from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from ctypes import byref, c_int, c_uint32, sizeof
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from log_manager import GlobalLogManager
from recognition_http_server.utils.capture_gimbal_frames import compute_frame_statistics, mask_source

DEFAULT_HIKVISION_ENV_PATH = ROOT / "recognition_http_server" / "hikvision.env"
DEFAULT_CAPTURE_CLEANUP_DIR = ROOT / "results" / "gimbal_capture"
DEFAULT_RECOVERY_LOG_DIR = DEFAULT_CAPTURE_CLEANUP_DIR / "logs"
DEFAULT_B_MINUS_G_THRESHOLD = 1.0
DEFAULT_GREEN_DEFICIT_THRESHOLD = 1.0
DEFAULT_WINDOW_SIZE = 5
DEFAULT_MIN_ERROR_FRAMES = 3
DAY_MODE = 0
NIGHT_MODE = 1
AUTO_MODE = 2
DAY_NIGHT_RESET_SEQUENCE = (NIGHT_MODE, DAY_MODE, AUTO_MODE)

SDK_PY_DIR = ROOT / "HK_SDK" / "HK_SDK_arm64_Linux"
if str(SDK_PY_DIR) not in sys.path:
    sys.path.insert(0, str(SDK_PY_DIR))

try:
    from HCNetSDK import *  # type: ignore  # noqa: F403
except Exception:  # pragma: no cover - local non-Linux test environments can still exercise injected paths.
    NET_DVR_GET_CCDPARAMCFG_EX = 3368
    NET_DVR_SET_CCDPARAMCFG_EX = 3369
    NET_DVR_CAMERAPARAMCFG_EX = None


Stats = Dict[str, float]
RecoveryCallback = Callable[[], None]


class StreamErrorWindow:
    """Track recent frame-level stream error decisions."""

    def __init__(
        self,
        window_size: int = DEFAULT_WINDOW_SIZE,
        min_error_frames: int = DEFAULT_MIN_ERROR_FRAMES,
        b_minus_g_threshold: float = DEFAULT_B_MINUS_G_THRESHOLD,
        green_deficit_threshold: float = DEFAULT_GREEN_DEFICIT_THRESHOLD,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if min_error_frames <= 0 or min_error_frames > window_size:
            raise ValueError("min_error_frames must be in 1..window_size")
        self.window_size = window_size
        self.min_error_frames = min_error_frames
        self.b_minus_g_threshold = b_minus_g_threshold
        self.green_deficit_threshold = green_deficit_threshold
        self._decisions: deque[bool] = deque(maxlen=window_size)

    def add(self, stats: Stats) -> bool:
        """Add one frame stats dict and return whether the window indicates an error."""
        self._decisions.append(is_stream_error_frame(stats, self.b_minus_g_threshold, self.green_deficit_threshold))
        return self.is_error

    def clear(self) -> None:
        self._decisions.clear()

    @property
    def is_error(self) -> bool:
        return len(self._decisions) == self.window_size and sum(self._decisions) >= self.min_error_frames


def compute_stream_error_score(stats: Stats) -> float:
    """Compute the stream color error score as the absolute B-G gap."""
    return abs(float(stats["b_minus_g"]))


def is_stream_error_frame(
    stats: Stats,
    b_minus_g_threshold: float = DEFAULT_B_MINUS_G_THRESHOLD,
    green_deficit_threshold: float = DEFAULT_GREEN_DEFICIT_THRESHOLD,
) -> bool:
    """Return True when the frame shows the IR-filter-stuck magenta color cast.

    Two conditions must both be met:
    1. |B - G| < b_minus_g_threshold — the blue channel has caught up to green
    (IR leaks primarily through R/B Bayer filters, pushing B toward G).
    2. green_deficit > green_deficit_threshold — a genuine magenta bias exists,
    ruling out naturally neutral scenes (gray walls, low-light, etc.).
    """
    b_gap = abs(float(stats["b_minus_g"]))
    green_deficit = float(stats["green_deficit"])
    return b_gap < b_minus_g_threshold and green_deficit > green_deficit_threshold


def cleanup_capture_images(capture_root: str | Path) -> int:
    """Delete captured frame image files under capture_root while keeping manifests and other files."""
    root = Path(capture_root)
    if not root.exists():
        return 0

    deleted = 0
    for pattern in ("frame_*.jpg", "frame_*.jpeg", "frame_*.png"):
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            path.unlink()
            deleted += 1
    return deleted


def reset_camera_day_night_mode(
    sdk,
    user_id: int,
    camera_param_cls=None,
    get_command: int = NET_DVR_GET_CCDPARAMCFG_EX,
    set_command: int = NET_DVR_SET_CCDPARAMCFG_EX,
    channel: int = 1,
    sequence: Sequence[int] = DAY_NIGHT_RESET_SEQUENCE,
    settle_seconds: float = 0.5,
) -> int:
    """Reset camera color state by switching day/night mode through night, day, then auto.

    Returns:
        The day/night mode value read before the reset sequence.
    """
    camera_param_cls = camera_param_cls or NET_DVR_CAMERAPARAMCFG_EX
    if camera_param_cls is None:
        raise RuntimeError("HCNetSDK NET_DVR_CAMERAPARAMCFG_EX is unavailable in this environment.")

    camera_param = camera_param_cls()
    camera_param.dwSize = sizeof(camera_param)
    returned = c_uint32(0)
    channel_arg = c_int(channel)

    ok = sdk.NET_DVR_GetDVRConfig(
        user_id,
        get_command,
        channel_arg,
        byref(camera_param),
        sizeof(camera_param),
        byref(returned),
    )
    if not ok:
        raise RuntimeError(f"NET_DVR_GET_CCDPARAMCFG_EX failed: error={sdk.NET_DVR_GetLastError()}")

    original_mode = int(camera_param.struDayNight.byDayNightFilterType)
    for index, mode in enumerate(sequence):
        camera_param.struDayNight.byDayNightFilterType = int(mode)
        ok = sdk.NET_DVR_SetDVRConfig(user_id, set_command, channel, byref(camera_param), sizeof(camera_param))
        if not ok:
            raise RuntimeError(f"NET_DVR_SET_CCDPARAMCFG_EX failed: mode={mode} error={sdk.NET_DVR_GetLastError()}")
        if settle_seconds > 0 and index < len(sequence) - 1:
            time.sleep(settle_seconds)
    return original_mode


def reset_hikvision_stream_color(config, settle_seconds: float = 0.5) -> int:
    """Login through HCNetSDK and apply the day/night reset sequence."""
    from recognition_http_server.hikvision_ptz import _bind_local_ip, _configure_sdk_paths, _load_sdk, _login

    if not config.host or not config.password:
        raise ValueError("Hikvision stream reset requires host and password.")

    sdk = _load_sdk(config.sdk_lib_dir)
    _configure_sdk_paths(sdk, config.sdk_lib_dir)
    if not sdk.NET_DVR_Init():
        raise RuntimeError(f"NET_DVR_Init failed: error={sdk.NET_DVR_GetLastError()}")

    user_id = -1
    try:
        if config.local_ip:
            _bind_local_ip(sdk, config.local_ip)
        user_id = _login(sdk, config)
        return reset_camera_day_night_mode(sdk, user_id, channel=config.channel, settle_seconds=settle_seconds)
    finally:
        if user_id >= 0:
            sdk.NET_DVR_Logout(user_id)
        sdk.NET_DVR_Cleanup()


def parse_hikvision_env(path: str | Path = DEFAULT_HIKVISION_ENV_PATH) -> dict[str, str]:
    """Parse a simple KEY=VALUE Hikvision env file."""
    values: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def config_from_env(path: str | Path = DEFAULT_HIKVISION_ENV_PATH):
    """Build a Hikvision PTZ config from recognition_http_server/hikvision.env."""
    from recognition_http_server.hikvision_ptz import HikvisionPTZConfig

    values = parse_hikvision_env(path)
    return HikvisionPTZConfig(
        host=values.get("HIK_HOST", ""),
        port=int(values.get("HIK_PORT", "8000")),
        username=values.get("HIK_USERNAME", "admin"),
        password=values.get("HIK_PASSWORD", ""),
        channel=int(values.get("HIK_CHANNEL", "1")),
        local_ip=values.get("HIK_LOCAL_IP", ""),
    )


def build_rtsp_source(config, stream: str = "main") -> str:
    """Build a Hikvision RTSP source URL for main or sub stream."""
    suffix = "01" if stream == "main" else "02"
    return f"rtsp://{config.username}:{config.password}@{config.host}:554/Streaming/Channels/{config.channel}{suffix}"


def iter_stream_statistics(source: int | str, interval: float = 1.0) -> Iterable[Stats]:
    """Yield frame statistics from an OpenCV-readable stream."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for stream error monitoring.") from exc

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open stream source: {mask_source(source)}")
    try:
        while True:
            ok, frame = capture.read()
            if ok and frame is not None:
                yield compute_frame_statistics(frame)
            if interval > 0:
                time.sleep(interval)
    finally:
        capture.release()


def _get_monitor_logger() -> logging.Logger:
    return GlobalLogManager.get_logger("stream.error")


def monitor_stream_and_recover(
    source: int | str,
    recover: RecoveryCallback,
    interval: float = 1.0,
    window_size: int = DEFAULT_WINDOW_SIZE,
    min_error_frames: int = DEFAULT_MIN_ERROR_FRAMES,
    b_minus_g_threshold: float = DEFAULT_B_MINUS_G_THRESHOLD,
    green_deficit_threshold: float = DEFAULT_GREEN_DEFICIT_THRESHOLD,
    max_frames: int | None = None,
    cooldown_seconds: float = 10.0,
    cleanup_dir: str | Path | None = DEFAULT_CAPTURE_CLEANUP_DIR,
) -> None:
    """Monitor stream color statistics and run recovery when the recent window is abnormal."""
    logger = _get_monitor_logger()
    detector = StreamErrorWindow(window_size, min_error_frames, b_minus_g_threshold, green_deficit_threshold)
    last_recovery_at = 0.0
    for index, stats in enumerate(iter_stream_statistics(source, interval)):
        b_gap = compute_stream_error_score(stats)
        is_error = detector.add(stats)
        logger.debug(
            "frame=%d b_gap=%.3f r_minus_g=%.3f b_minus_g=%.3f green_deficit=%.3f window_error=%s",
            index,
            b_gap,
            stats["r_minus_g"],
            stats["b_minus_g"],
            stats["green_deficit"],
            is_error,
        )
        now = time.monotonic()
        if is_error and now - last_recovery_at >= cooldown_seconds:
            logger.warning(
                "stream color error detected (b_gap=%.3f, green_deficit=%.3f); resetting camera day/night mode",
                b_gap,
                stats["green_deficit"],
            )
            recover()
            if cleanup_dir is not None:
                deleted = cleanup_capture_images(cleanup_dir)
                logger.info("deleted_capture_images=%d cleanup_dir=%s", deleted, cleanup_dir)
            last_recovery_at = now
            detector.clear()
        if max_frames is not None and index + 1 >= max_frames:
            return


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and recover Hikvision stream color disorder.")
    parser.add_argument("--env", type=Path, default=DEFAULT_HIKVISION_ENV_PATH, help="Path to hikvision.env.")
    parser.add_argument("--source", default="", help="Optional explicit RTSP/source override.")
    parser.add_argument("--stream", choices=("main", "sub"), default="main", help="RTSP stream to build from env.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between sampled frames.")
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--min-error-frames", type=int, default=DEFAULT_MIN_ERROR_FRAMES)
    parser.add_argument(
        "--b-minus-g-threshold",
        type=float,
        default=DEFAULT_B_MINUS_G_THRESHOLD,
        help="|B-G| below this value indicates B has caught up to G (IR leak).",
    )
    parser.add_argument(
        "--green-deficit-threshold",
        type=float,
        default=DEFAULT_GREEN_DEFICIT_THRESHOLD,
        help="green_deficit above this value confirms magenta cast rather than neutral scene.",
    )
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 means run forever.")
    parser.add_argument(
        "--capture-cleanup-dir",
        type=Path,
        default=DEFAULT_CAPTURE_CLEANUP_DIR,
        help="Delete frame_*.jpg/jpeg/png under this directory after a successful recovery.",
    )
    parser.add_argument(
        "--no-cleanup-captures", action="store_true", help="Do not delete captured images after recovery."
    )
    parser.add_argument("--dry-run", action="store_true", help="Detect only; do not reset camera parameters.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_env(args.env)
    source = args.source or build_rtsp_source(config, args.stream)

    GlobalLogManager.configure(
        log_dir=DEFAULT_RECOVERY_LOG_DIR,
        level=logging.INFO,
        log_file_name="recovery.log",
    )
    logger = _get_monitor_logger()

    def recover() -> None:
        if args.dry_run:
            logger.info("dry-run: skip camera day/night reset")
            return
        logger.warning("executing camera day/night reset sequence")
        reset_hikvision_stream_color(config)

    try:
        logger.info("monitor_source=%s", mask_source(source))
        monitor_stream_and_recover(
            source,
            recover,
            interval=args.interval,
            window_size=args.window_size,
            min_error_frames=args.min_error_frames,
            b_minus_g_threshold=args.b_minus_g_threshold,
            green_deficit_threshold=args.green_deficit_threshold,
            max_frames=args.max_frames or None,
            cooldown_seconds=args.cooldown_seconds,
            cleanup_dir=None if args.dry_run or args.no_cleanup_captures else args.capture_cleanup_dir,
        )
    except KeyboardInterrupt:
        logger.info("stream monitor interrupted by user")
    except Exception as exc:
        logger.exception("stream monitor error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
