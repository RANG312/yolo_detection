from __future__ import annotations

import ctypes
import os
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
from requests.auth import HTTPDigestAuth

from recognition_http_server.ptz_alignment import PTZAlignmentRequest, PTZZoomLimits

ROOT = Path(__file__).resolve().parents[1]
SDK_BASE_DIR = ROOT / "HK_SDK"
LEGACY_X86_SDK_ROOT = ROOT / "HK_SDK_x86_Linux" / "HCNetSDKV6.1.11.5_build20251204_linux64_ZH"
X86_SDK_ROOT_NAME = "HK_SDK_x86_Linux"
ARM64_SDK_ROOT_NAME = "HK_SDK_arm64_Linux"


def _default_sdk_lib_dir(machine: str | None = None) -> Path:
    arch = (machine or platform.machine()).lower()
    if arch in {"aarch64", "arm64"}:
        return SDK_BASE_DIR / ARM64_SDK_ROOT_NAME / "lib" / "linux"

    flat_x86_lib_dir = SDK_BASE_DIR / X86_SDK_ROOT_NAME / "lib" / "linux"
    if flat_x86_lib_dir.exists():
        return flat_x86_lib_dir

    nested_x86_lib_dir = SDK_BASE_DIR / X86_SDK_ROOT_NAME / "HCNetSDKV6.1.11.5_build20251204_linux64_ZH" / "库文件"
    if nested_x86_lib_dir.exists():
        return nested_x86_lib_dir

    return LEGACY_X86_SDK_ROOT / "库文件"


def resolve_sdk_lib_dir(override: str | Path | None = None) -> Path:
    configured = override or os.environ.get("HIK_SDK_LIB_DIR", "")
    if configured:
        return Path(configured).expanduser()
    return _default_sdk_lib_dir()


DEFAULT_LIB_DIR = resolve_sdk_lib_dir()

NET_DVR_SET_PTZPOS = 292
NET_DVR_GET_PTZPOS = 293
NET_DVR_GET_GISINFO = 3711
SDK_LONG = ctypes.c_int32
PTZ_COMMANDS = {
    "up": 21,
    "down": 22,
    "left": 23,
    "right": 24,
    "zoom-in": 11,
    "zoom-out": 12,
}


class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_byte * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        ("byStartChan", ctypes.c_byte),
        ("byAudioChanNum", ctypes.c_byte),
        ("byIPChanNum", ctypes.c_byte),
        ("byZeroChanNum", ctypes.c_byte),
        ("byMainProto", ctypes.c_byte),
        ("bySubProto", ctypes.c_byte),
        ("bySupport", ctypes.c_byte),
        ("bySupport1", ctypes.c_byte),
        ("bySupport2", ctypes.c_byte),
        ("wDevType", ctypes.c_uint16),
        ("bySupport3", ctypes.c_byte),
        ("byMultiStreamProto", ctypes.c_byte),
        ("byStartDChan", ctypes.c_byte),
        ("byStartDTalkChan", ctypes.c_byte),
        ("byHighDChanNum", ctypes.c_byte),
        ("bySupport4", ctypes.c_byte),
        ("byLanguageType", ctypes.c_byte),
        ("byVoiceInChanNum", ctypes.c_byte),
        ("byStartVoiceInChanNo", ctypes.c_byte),
        ("bySupport5", ctypes.c_byte),
        ("bySupport6", ctypes.c_byte),
        ("byMirrorChanNum", ctypes.c_byte),
        ("wStartMirrorChanNo", ctypes.c_uint16),
        ("bySupport7", ctypes.c_byte),
        ("byRes2", ctypes.c_byte),
    ]


class NET_DVR_LOCAL_SDK_PATH(ctypes.Structure):
    _fields_ = [
        ("sPath", ctypes.c_char * 256),
        ("byRes", ctypes.c_byte * 128),
    ]


class NET_DVR_PTZPOS(ctypes.Structure):
    _fields_ = [
        ("wAction", ctypes.c_uint16),
        ("wPanPos", ctypes.c_uint16),
        ("wTiltPos", ctypes.c_uint16),
        ("wZoomPos", ctypes.c_uint16),
    ]


class NET_DVR_STD_CONFIG(ctypes.Structure):
    _fields_ = [
        ("lpCondBuffer", ctypes.c_void_p),
        ("dwCondSize", ctypes.c_uint32),
        ("lpInBuffer", ctypes.c_void_p),
        ("dwInSize", ctypes.c_uint32),
        ("lpOutBuffer", ctypes.c_void_p),
        ("dwOutSize", ctypes.c_uint32),
        ("lpStatusBuffer", ctypes.c_void_p),
        ("dwStatusSize", ctypes.c_uint32),
        ("lpXmlBuffer", ctypes.c_void_p),
        ("dwXmlSize", ctypes.c_uint32),
        ("byDataType", ctypes.c_byte),
        ("byRes", ctypes.c_byte * 23),
    ]


class NET_PTZ_INFO(ctypes.Structure):
    _fields_ = [
        ("fPan", ctypes.c_float),
        ("fTilt", ctypes.c_float),
        ("fZoom", ctypes.c_float),
        ("dwFocus", ctypes.c_uint32),
        ("byRes", ctypes.c_byte * 4),
    ]


class NET_DVR_LLI_PARAM(ctypes.Structure):
    _fields_ = [
        ("fSec", ctypes.c_float),
        ("byDegree", ctypes.c_byte),
        ("byMinute", ctypes.c_byte),
        ("byRes", ctypes.c_byte * 6),
    ]


class NET_DVR_PTZPOS_PARAM(ctypes.Structure):
    _fields_ = [
        ("fPanPos", ctypes.c_float),
        ("fTiltPos", ctypes.c_float),
        ("fZoomPos", ctypes.c_float),
        ("byRes", ctypes.c_byte * 16),
    ]


class NET_DVR_SENSOR_PARAM(ctypes.Structure):
    _fields_ = [
        ("bySensorType", ctypes.c_byte),
        ("byRes", ctypes.c_byte * 31),
        ("fHorWidth", ctypes.c_float),
        ("fVerWidth", ctypes.c_float),
        ("fFold", ctypes.c_float),
    ]


class NET_DVR_GIS_INFO(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("fAzimuth", ctypes.c_float),
        ("fHorizontalValue", ctypes.c_float),
        ("fVerticalValue", ctypes.c_float),
        ("fVisibleRadius", ctypes.c_float),
        ("fMaxViewRadius", ctypes.c_float),
        ("byLatitudeType", ctypes.c_byte),
        ("byLongitudeType", ctypes.c_byte),
        ("byPTZPosExEnable", ctypes.c_byte),
        ("byRes1", ctypes.c_byte),
        ("struLatitude", NET_DVR_LLI_PARAM),
        ("struLongitude", NET_DVR_LLI_PARAM),
        ("struPtzPos", NET_DVR_PTZPOS_PARAM),
        ("struSensorParam", NET_DVR_SENSOR_PARAM),
        ("struPtzPosEx", NET_PTZ_INFO),
        ("fMinHorizontalValue", ctypes.c_float),
        ("fMaxHorizontalValue", ctypes.c_float),
        ("fMinVerticalValue", ctypes.c_float),
        ("fMaxVerticalValue", ctypes.c_float),
        ("byRes", ctypes.c_byte * 220),
    ]


@dataclass(frozen=True)
class HikvisionPTZConfig:
    host: str = ""
    port: int = 8000
    username: str = "admin"
    password: str = ""
    channel: int = 1
    snapshot_channel: int | None = None
    local_ip: str = ""
    sdk_lib_dir: Path = DEFAULT_LIB_DIR
    tilt_min_deg: float = 0.0
    tilt_max_deg: float = 90.0
    settle_seconds: float = 0.3
    nudge_speed: int = 1
    nudge_degrees_per_second: float = 8.0
    pan_nudge_degrees_per_second: float | None = None
    tilt_nudge_degrees_per_second: float | None = None
    nudge_min_seconds: float = 0.05
    nudge_max_seconds: float = 1.0
    nudge_max_steps: int = 2
    tilt_nudge_scale: float = 1.0
    zoom_nudge_speed: int = 1
    zoom_nudge_seconds: float = 0.3
    zoom_nudge_steps: int = 1
    zoom_focus_timeout: float = 2.0
    zoom_max_ratio: float = 0.0
    snapshot_timeout: float = 5.0


@dataclass(frozen=True)
class PTZPosition:
    pan_deg: float
    tilt_deg: float
    zoom_deg: float


@dataclass(frozen=True)
class HikvisionFieldOfView:
    horizontal_deg: float
    vertical_deg: float
    min_horizontal_deg: float
    max_horizontal_deg: float
    min_vertical_deg: float
    max_vertical_deg: float
    zoom: float

    @property
    def max_zoom(self) -> float:
        return _max_zoom_from_fov(
            self.min_horizontal_deg,
            self.max_horizontal_deg,
            self.min_vertical_deg,
            self.max_vertical_deg,
        )


class HikvisionPTZController:
    """Small HCNetSDK wrapper that keeps one SDK session for repeated PTZ operations."""

    def __init__(self, config: HikvisionPTZConfig, logger=None) -> None:
        self.config = config
        self.logger = logger
        self._lock = threading.Lock()
        self._sdk: ctypes.CDLL | None = None
        self._user_id = -1

    def close(self) -> None:
        """Release the persistent HCNetSDK login and SDK process state."""
        with self._lock:
            self._close_unlocked()

    def _ensure_session_unlocked(self) -> tuple[ctypes.CDLL, int]:
        if self._sdk is not None and self._user_id >= 0:
            return self._sdk, self._user_id

        sdk = _load_sdk(self.config.sdk_lib_dir)
        _configure_sdk_paths(sdk, self.config.sdk_lib_dir)
        if not sdk.NET_DVR_Init():
            raise RuntimeError(f"NET_DVR_Init failed: error={sdk.NET_DVR_GetLastError()}")
        try:
            if self.config.local_ip:
                _bind_local_ip(sdk, self.config.local_ip)
            user_id = _login(sdk, self.config)
        except Exception:
            sdk.NET_DVR_Cleanup()
            raise

        self._sdk = sdk
        self._user_id = user_id
        return sdk, user_id

    def _close_unlocked(self) -> None:
        sdk = self._sdk
        user_id = self._user_id
        self._sdk = None
        self._user_id = -1
        if sdk is None:
            return
        if user_id >= 0:
            sdk.NET_DVR_Logout(user_id)
        sdk.NET_DVR_Cleanup()

    def align(self, request: PTZAlignmentRequest) -> None:
        if not self.config.host or not self.config.password:
            raise ValueError("Hikvision PTZ alignment requires host and password.")

        with self._lock:
            try:
                sdk, user_id = self._ensure_session_unlocked()
                before = _read_ptz(sdk, user_id, self.config.channel)
                _nudge_by_delta(sdk, user_id, self.config, request.pan_delta_deg, request.tilt_delta_deg)
                after = _read_ptz(sdk, user_id, self.config.channel)
                if self.logger is not None:
                    self.logger.info(
                        "hikvision ptz aligned: pan %.3f->%.3f tilt %.3f->%.3f zoom %.3f->%.3f",
                        before.pan_deg,
                        after.pan_deg,
                        before.tilt_deg,
                        after.tilt_deg,
                        before.zoom_deg,
                        after.zoom_deg,
                    )
                if self.config.settle_seconds > 0:
                    time.sleep(self.config.settle_seconds)
            except Exception:
                self._close_unlocked()
                raise

    def zoom(self, request) -> None:
        if not self.config.host or not self.config.password:
            raise ValueError("Hikvision PTZ zoom requires host and password.")
        if request.zoom_direction not in {"in", "out"}:
            return

        with self._lock:
            try:
                sdk, user_id = self._ensure_session_unlocked()
                before = _read_ptz(sdk, user_id, self.config.channel)
                _nudge_zoom(sdk, user_id, self.config, request.zoom_direction)
                after = _read_ptz(sdk, user_id, self.config.channel)
                if self.logger is not None:
                    self.logger.info(
                        (
                            "hikvision ptz zoomed: direction=%s ratio %.3f->target %.3f "
                            "pan %.3f->%.3f tilt %.3f->%.3f zoom %.3f->%.3f"
                        ),
                        request.zoom_direction,
                        request.current_height_ratio,
                        request.target_height_ratio,
                        before.pan_deg,
                        after.pan_deg,
                        before.tilt_deg,
                        after.tilt_deg,
                        before.zoom_deg,
                        after.zoom_deg,
                    )
                if self.config.zoom_focus_timeout > 0:
                    time.sleep(self.config.zoom_focus_timeout)
            except Exception:
                self._close_unlocked()
                raise

    def read_field_of_view(self) -> HikvisionFieldOfView:
        if not self.config.host or not self.config.password:
            raise ValueError("Hikvision FOV reading requires host and password.")

        with self._lock:
            try:
                sdk, user_id = self._ensure_session_unlocked()
                return _read_gis_fov(sdk, user_id, self.config.channel)
            except Exception:
                self._close_unlocked()
                raise

    def read_zoom_limits(self) -> PTZZoomLimits | None:
        if not self.config.host or not self.config.password:
            raise ValueError("Hikvision zoom limit reading requires host and password.")

        with self._lock:
            try:
                sdk, user_id = self._ensure_session_unlocked()
                fov = _read_gis_fov(sdk, user_id, self.config.channel)
                if self.config.zoom_max_ratio > 0.0:
                    max_zoom = float(self.config.zoom_max_ratio)
                elif not _has_zoom_fov_range(fov):
                    return None
                else:
                    max_zoom = float(fov.max_zoom)
                return PTZZoomLimits(current_zoom=float(fov.zoom), max_zoom=max_zoom)
            except Exception:
                self._close_unlocked()
                raise

    def read_position(self) -> PTZPosition:
        if not self.config.host or not self.config.password:
            raise ValueError("Hikvision PTZ position reading requires host and password.")

        with self._lock:
            try:
                sdk, user_id = self._ensure_session_unlocked()
                return _read_ptz(sdk, user_id, self.config.channel)
            except Exception:
                self._close_unlocked()
                raise

    def set_position(self, position: PTZPosition) -> None:
        if not self.config.host or not self.config.password:
            raise ValueError("Hikvision PTZ position restore requires host and password.")

        with self._lock:
            try:
                sdk, user_id = self._ensure_session_unlocked()
                _set_ptz(sdk, user_id, self.config.channel, position.pan_deg, position.tilt_deg, position.zoom_deg)
                if self.logger is not None:
                    self.logger.info(
                        "hikvision ptz restored: pan=%.3f tilt=%.3f zoom=%.3f",
                        position.pan_deg,
                        position.tilt_deg,
                        position.zoom_deg,
                    )
                if self.config.settle_seconds > 0:
                    time.sleep(self.config.settle_seconds)
            except Exception:
                self._close_unlocked()
                raise

    def capture_image(self) -> np.ndarray:
        if not self.config.host or not self.config.password:
            raise ValueError("Hikvision snapshot requires host and password.")
        if self.config.snapshot_channel is not None:
            snapshot_channels = [self.config.snapshot_channel]
        else:
            snapshot_channels = [int(f"{self.config.channel}01"), int(f"{self.config.channel}02")]

        response = None
        for index, snapshot_channel in enumerate(snapshot_channels):
            url = f"http://{self.config.host}/ISAPI/Streaming/channels/{snapshot_channel}/picture"
            response = requests.get(
                url,
                auth=HTTPDigestAuth(self.config.username, self.config.password),
                timeout=self.config.snapshot_timeout,
            )
            try:
                response.raise_for_status()
                break
            except requests.HTTPError as exc:
                is_last_channel = index == len(snapshot_channels) - 1
                status_code = getattr(exc.response, "status_code", None)
                if is_last_channel or status_code != 503:
                    raise

        if response is None:
            raise RuntimeError("Hikvision snapshot did not return a response.")
        image = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Hikvision snapshot response is not a valid image.")
        return image


def _load_sdk(lib_dir: Path) -> ctypes.CDLL:
    sdk = ctypes.CDLL(str(lib_dir / "libhcnetsdk.so"))
    sdk.NET_DVR_SetSDKInitCfg.restype = ctypes.c_bool
    sdk.NET_DVR_Init.restype = ctypes.c_bool
    sdk.NET_DVR_GetLastError.restype = ctypes.c_uint32
    sdk.NET_DVR_GetLocalIP.restype = ctypes.c_bool
    sdk.NET_DVR_SetValidIP.restype = ctypes.c_bool
    sdk.NET_DVR_Login_V30.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint16,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(NET_DVR_DEVICEINFO_V30),
    ]
    sdk.NET_DVR_Login_V30.restype = SDK_LONG
    sdk.NET_DVR_GetDVRConfig.argtypes = [
        SDK_LONG,
        ctypes.c_uint32,
        SDK_LONG,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    sdk.NET_DVR_GetDVRConfig.restype = ctypes.c_bool
    sdk.NET_DVR_SetDVRConfig.argtypes = [SDK_LONG, ctypes.c_uint32, SDK_LONG, ctypes.c_void_p, ctypes.c_uint32]
    sdk.NET_DVR_SetDVRConfig.restype = ctypes.c_bool
    sdk.NET_DVR_GetSTDConfig.argtypes = [SDK_LONG, ctypes.c_uint32, ctypes.POINTER(NET_DVR_STD_CONFIG)]
    sdk.NET_DVR_GetSTDConfig.restype = ctypes.c_bool
    sdk.NET_DVR_PTZControlWithSpeed_Other.argtypes = [
        SDK_LONG,
        SDK_LONG,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    sdk.NET_DVR_PTZControlWithSpeed_Other.restype = ctypes.c_bool
    sdk.NET_DVR_Logout.argtypes = [SDK_LONG]
    sdk.NET_DVR_Logout.restype = ctypes.c_bool
    sdk.NET_DVR_Cleanup.restype = ctypes.c_bool
    return sdk


def _configure_sdk_paths(sdk: ctypes.CDLL, lib_dir: Path) -> None:
    lib_dir = lib_dir.resolve()
    lib_path = str(lib_dir).encode("utf-8")
    sdk_path = NET_DVR_LOCAL_SDK_PATH()
    sdk_path.sPath = lib_path
    sdk.NET_DVR_SetSDKInitCfg(2, ctypes.byref(sdk_path))
    crypto_path = _first_existing(lib_dir, ("libcrypto.so.3", "libcrypto.so.1.1", "libcrypto.so"))
    ssl_path = _first_existing(lib_dir, ("libssl.so.3", "libssl.so.1.1", "libssl.so"))
    sdk.NET_DVR_SetSDKInitCfg(3, ctypes.create_string_buffer(str(crypto_path).encode("utf-8")))
    sdk.NET_DVR_SetSDKInitCfg(4, ctypes.create_string_buffer(str(ssl_path).encode("utf-8")))


def _first_existing(lib_dir: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = lib_dir / name
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"missing Hikvision SDK dependency in {lib_dir}: {', '.join(names)}")


def _bind_local_ip(sdk: ctypes.CDLL, local_ip: str) -> None:
    ip_buffer = ctypes.create_string_buffer(16 * 64)
    count = ctypes.c_int(0)
    enable_bind = ctypes.c_bool(False)
    if not sdk.NET_DVR_GetLocalIP(ctypes.byref(ip_buffer), ctypes.byref(count), enable_bind):
        raise RuntimeError(f"NET_DVR_GetLocalIP failed: error={sdk.NET_DVR_GetLastError()}")

    for index in range(count.value):
        chunk = ip_buffer.raw[index * 64 : (index + 1) * 64].split(b"\x00", 1)[0]
        ips = chunk.decode("utf-8", errors="ignore").split()
        if local_ip in ips:
            if sdk.NET_DVR_SetValidIP(index, True):
                return
            raise RuntimeError(f"NET_DVR_SetValidIP failed: error={sdk.NET_DVR_GetLastError()}")
    raise RuntimeError(f"Local IP not found by HCNetSDK: {local_ip}")


def _login(sdk: ctypes.CDLL, config: HikvisionPTZConfig) -> int:
    device_info = NET_DVR_DEVICEINFO_V30()
    user_id = sdk.NET_DVR_Login_V30(
        config.host.encode("ascii"),
        config.port,
        config.username.encode("ascii"),
        config.password.encode("ascii"),
        ctypes.byref(device_info),
    )
    if user_id < 0:
        raise RuntimeError(f"NET_DVR_Login_V30 failed: host={config.host} error={sdk.NET_DVR_GetLastError()}")
    return int(user_id)


def _read_ptz(sdk: ctypes.CDLL, user_id: int, channel: int) -> PTZPosition:
    returned = ctypes.c_uint32(0)
    ptz = NET_DVR_PTZPOS()
    ok = sdk.NET_DVR_GetDVRConfig(
        user_id,
        NET_DVR_GET_PTZPOS,
        channel,
        ctypes.byref(ptz),
        ctypes.sizeof(ptz),
        ctypes.byref(returned),
    )
    if not ok:
        raise RuntimeError(f"NET_DVR_GetDVRConfig PTZ failed: error={sdk.NET_DVR_GetLastError()}")
    return PTZPosition(
        pan_deg=_decode_bcd_angle(ptz.wPanPos),
        tilt_deg=_decode_bcd_angle(ptz.wTiltPos),
        zoom_deg=_decode_bcd_angle(ptz.wZoomPos),
    )


def _read_gis_fov(sdk: ctypes.CDLL, user_id: int, channel: int) -> HikvisionFieldOfView:
    info = NET_DVR_GIS_INFO()
    info.dwSize = ctypes.sizeof(info)
    cond = ctypes.c_int(channel)
    status = ctypes.create_string_buffer(4096)
    cfg = NET_DVR_STD_CONFIG()
    cfg.lpCondBuffer = ctypes.cast(ctypes.byref(cond), ctypes.c_void_p)
    cfg.dwCondSize = ctypes.sizeof(cond)
    cfg.lpOutBuffer = ctypes.cast(ctypes.byref(info), ctypes.c_void_p)
    cfg.dwOutSize = ctypes.sizeof(info)
    cfg.lpStatusBuffer = ctypes.cast(status, ctypes.c_void_p)
    cfg.dwStatusSize = ctypes.sizeof(status)
    cfg.byDataType = 0

    ok = sdk.NET_DVR_GetSTDConfig(user_id, NET_DVR_GET_GISINFO, ctypes.byref(cfg))
    if not ok:
        status_text = status.value.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"NET_DVR_GetSTDConfig GIS FOV failed: error={sdk.NET_DVR_GetLastError()} status={status_text}"
        )
    fov = HikvisionFieldOfView(
        horizontal_deg=float(info.fHorizontalValue),
        vertical_deg=float(info.fVerticalValue),
        min_horizontal_deg=float(info.fMinHorizontalValue),
        max_horizontal_deg=float(info.fMaxHorizontalValue),
        min_vertical_deg=float(info.fMinVerticalValue),
        max_vertical_deg=float(info.fMaxVerticalValue),
        zoom=float(info.struPtzPos.fZoomPos),
    )
    if not _is_valid_fov(fov):
        raise RuntimeError(f"NET_DVR_GET_GISINFO returned invalid FOV: {fov}")
    return fov


def _is_valid_fov(fov: HikvisionFieldOfView) -> bool:
    return 0.0 < fov.horizontal_deg <= 180.0 and 0.0 < fov.vertical_deg <= 180.0


def _has_zoom_fov_range(fov: HikvisionFieldOfView) -> bool:
    return (fov.min_horizontal_deg > 0.0 and fov.max_horizontal_deg > fov.min_horizontal_deg) or (
        fov.min_vertical_deg > 0.0 and fov.max_vertical_deg > fov.min_vertical_deg
    )


def _max_zoom_from_fov(
    min_horizontal_deg: float,
    max_horizontal_deg: float,
    min_vertical_deg: float,
    max_vertical_deg: float,
) -> float:
    ratios = []
    if min_horizontal_deg > 0.0 and max_horizontal_deg > 0.0:
        ratios.append(max_horizontal_deg / min_horizontal_deg)
    if min_vertical_deg > 0.0 and max_vertical_deg > 0.0:
        ratios.append(max_vertical_deg / min_vertical_deg)
    if not ratios:
        return 1.0
    return max(1.0, max(ratios))


def _set_ptz(
    sdk: ctypes.CDLL,
    user_id: int,
    channel: int,
    pan_deg: float,
    tilt_deg: float,
    zoom_deg: float,
) -> None:
    ptz = NET_DVR_PTZPOS()
    ptz.wAction = 1
    ptz.wPanPos = _encode_bcd_angle(pan_deg)
    ptz.wTiltPos = _encode_bcd_angle(tilt_deg)
    ptz.wZoomPos = _encode_bcd_angle(zoom_deg)
    ok = sdk.NET_DVR_SetDVRConfig(user_id, NET_DVR_SET_PTZPOS, channel, ctypes.byref(ptz), ctypes.sizeof(ptz))
    if not ok:
        raise RuntimeError(f"NET_DVR_SetDVRConfig PTZ failed: error={sdk.NET_DVR_GetLastError()}")


def _nudge_by_delta(
    sdk: ctypes.CDLL,
    user_id: int,
    config: HikvisionPTZConfig,
    pan_delta_deg: float,
    tilt_delta_deg: float,
) -> None:
    if abs(pan_delta_deg) > 1e-6:
        direction = "right" if pan_delta_deg > 0 else "left"
        pan_dps = config.pan_nudge_degrees_per_second or config.nudge_degrees_per_second
        for duration in _durations_for_degrees(abs(pan_delta_deg), config, degrees_per_second=pan_dps):
            _nudge_ptz(sdk, user_id, config.channel, direction, duration, config.nudge_speed)
    if abs(tilt_delta_deg) > 1e-6:
        direction = "up" if tilt_delta_deg > 0 else "down"
        tilt_dps = config.tilt_nudge_degrees_per_second or config.nudge_degrees_per_second
        tilt_scale = 1.0 if config.tilt_nudge_degrees_per_second else config.tilt_nudge_scale
        for duration in _durations_for_degrees(
            abs(tilt_delta_deg),
            config,
            degrees_per_second=tilt_dps,
            scale=tilt_scale,
        ):
            _nudge_ptz(sdk, user_id, config.channel, direction, duration, config.nudge_speed)


def _nudge_zoom(sdk: ctypes.CDLL, user_id: int, config: HikvisionPTZConfig, direction: str) -> None:
    if config.zoom_nudge_steps < 1:
        raise ValueError("zoom_nudge_steps must be at least 1.")
    if config.zoom_nudge_seconds <= 0:
        raise ValueError("zoom_nudge_seconds must be positive.")
    command_direction = "zoom-in" if direction == "in" else "zoom-out"
    for _ in range(config.zoom_nudge_steps):
        _nudge_ptz(
            sdk,
            user_id,
            config.channel,
            command_direction,
            min(config.zoom_nudge_seconds, config.nudge_max_seconds),
            config.zoom_nudge_speed,
        )


def _durations_for_degrees(
    degrees: float,
    config: HikvisionPTZConfig,
    degrees_per_second: float | None = None,
    scale: float = 1.0,
) -> list[float]:
    degrees_per_second = degrees_per_second or config.nudge_degrees_per_second
    if degrees_per_second <= 0:
        raise ValueError("nudge degrees per second must be positive.")
    if config.nudge_max_steps < 1:
        raise ValueError("nudge_max_steps must be at least 1.")
    if scale <= 0:
        raise ValueError("nudge duration scale must be positive.")
    remaining = _clamp(
        float(degrees) / degrees_per_second,
        config.nudge_min_seconds,
        config.nudge_max_seconds * config.nudge_max_steps,
    )
    durations: list[float] = []
    while remaining > 1e-6 and len(durations) < config.nudge_max_steps:
        duration = min(config.nudge_max_seconds, remaining)
        clamped_duration = _clamp(
            max(config.nudge_min_seconds, duration) * scale,
            config.nudge_min_seconds,
            config.nudge_max_seconds,
        )
        durations.append(round(clamped_duration, 6))
        remaining -= duration
    return durations


def _nudge_ptz(
    sdk: ctypes.CDLL,
    user_id: int,
    channel: int,
    direction: str,
    duration: float,
    speed: int,
) -> None:
    command = PTZ_COMMANDS[direction]
    started = sdk.NET_DVR_PTZControlWithSpeed_Other(user_id, channel, command, 0, speed)
    if not started:
        raise RuntimeError(f"NET_DVR_PTZControlWithSpeed_Other start failed: error={sdk.NET_DVR_GetLastError()}")
    try:
        time.sleep(duration)
    finally:
        stopped = False
        stop_error = 0
        for attempt in range(3):
            stopped = sdk.NET_DVR_PTZControlWithSpeed_Other(user_id, channel, command, 1, speed)
            if stopped:
                break
            stop_error = int(sdk.NET_DVR_GetLastError())
            if attempt < 2:
                time.sleep(0.1)
    if not stopped:
        raise RuntimeError(f"NET_DVR_PTZControlWithSpeed_Other stop failed: error={stop_error}")


def _decode_bcd_angle(value: int) -> float:
    return int(f"{int(value):x}", 10) * 0.1


def _encode_bcd_angle(degrees: float) -> int:
    tenths = round(float(degrees) * 10.0)
    if tenths < 0:
        raise ValueError(f"PTZ BCD angle cannot be negative: {degrees}")
    return int(f"{tenths:d}", 16)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
