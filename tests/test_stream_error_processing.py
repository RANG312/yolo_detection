from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "recognition_http_server" / "utils" / "stream_error_processing.py"
SPEC = importlib.util.spec_from_file_location("stream_error_processing", SCRIPT_PATH)
stream_error_processing = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(stream_error_processing)


def test_compute_stream_error_score_returns_absolute_b_minus_g_gap() -> None:
    stats = {"r_minus_g": -0.6, "b_minus_g": -0.2, "green_deficit": 1.4}

    assert stream_error_processing.compute_stream_error_score(stats) == 0.2


def test_is_stream_error_frame_detects_ir_filter_stuck_signature() -> None:
    # Error: B close to G AND green_deficit elevated (magenta cast)
    assert stream_error_processing.is_stream_error_frame(
        {"r_minus_g": -0.8, "b_minus_g": -0.2, "green_deficit": 1.4},
    )
    # Normal: B far below G
    assert not stream_error_processing.is_stream_error_frame(
        {"r_minus_g": -2.0, "b_minus_g": -2.5, "green_deficit": 0.9},
    )
    # Gray wall: B close to G but green_deficit too low
    assert not stream_error_processing.is_stream_error_frame(
        {"r_minus_g": -0.1, "b_minus_g": -0.1, "green_deficit": 0.2},
    )
    # B close to G but green_deficit at boundary: not triggered
    assert not stream_error_processing.is_stream_error_frame(
        {"r_minus_g": -0.5, "b_minus_g": -0.5, "green_deficit": 1.0},
    )


def test_stream_error_window_requires_enough_recent_error_frames() -> None:
    detector = stream_error_processing.StreamErrorWindow(window_size=5, min_error_frames=3)

    error_frame = {"r_minus_g": -0.8, "b_minus_g": -0.2, "green_deficit": 1.4}
    normal_frame = {"r_minus_g": -2.0, "b_minus_g": -2.5, "green_deficit": 0.9}

    results = [
        detector.add(normal_frame),
        detector.add(error_frame),
        detector.add(error_frame),
        detector.add(normal_frame),
        detector.add(error_frame),
    ]

    assert results == [False, False, False, False, True]


def test_cleanup_capture_images_removes_only_frame_images(tmp_path: Path) -> None:
    capture_dir = tmp_path / "results" / "gimbal_capture" / "run_001"
    capture_dir.mkdir(parents=True)
    image = capture_dir / "frame_000001.jpg"
    nested_image = capture_dir / "nested" / "frame_000002.png"
    nested_image.parent.mkdir()
    manifest = capture_dir / "manifest.csv"
    unrelated = capture_dir / "preview.jpg"
    image.write_text("image", encoding="utf-8")
    nested_image.write_text("image", encoding="utf-8")
    manifest.write_text("manifest", encoding="utf-8")
    unrelated.write_text("preview", encoding="utf-8")

    deleted = stream_error_processing.cleanup_capture_images(tmp_path / "results" / "gimbal_capture")

    assert deleted == 2
    assert not image.exists()
    assert not nested_image.exists()
    assert manifest.exists()
    assert unrelated.exists()


def test_monitor_cleans_capture_images_after_successful_recovery(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    frame = {"r_minus_g": -0.5, "b_minus_g": -0.4, "green_deficit": 1.4}
    image = tmp_path / "run" / "frame_000001.jpg"
    image.parent.mkdir()
    image.write_text("image", encoding="utf-8")
    events: list[str] = []

    def fake_iter(source, interval):  # noqa: ANN001, ANN202
        yield frame

    monkeypatch.setattr(stream_error_processing, "iter_stream_statistics", fake_iter)

    stream_error_processing.monitor_stream_and_recover(
        "source",
        lambda: events.append("recover"),
        interval=0,
        window_size=1,
        min_error_frames=1,
        max_frames=1,
        cooldown_seconds=0,
        cleanup_dir=tmp_path,
    )

    assert events == ["recover"]
    assert not image.exists()


class FakeDayNight(ctypes.Structure):
    _fields_ = [("byDayNightFilterType", ctypes.c_byte)]


class FakeCameraParam(ctypes.Structure):
    _fields_ = [("dwSize", ctypes.c_uint32), ("struDayNight", FakeDayNight)]


class FakeSDK:
    def __init__(self) -> None:
        self.set_modes: list[int] = []
        self.last_error = 0

    def NET_DVR_GetDVRConfig(self, user_id, command, channel, param_ref, size, returned_ref):  # noqa: ANN001
        param = param_ref._obj
        param.struDayNight.byDayNightFilterType = 2
        returned_ref._obj.value = size
        self.get_call = (user_id, command, channel.value, size)
        return True

    def NET_DVR_SetDVRConfig(self, user_id, command, channel, param_ref, size):  # noqa: ANN001
        self.set_modes.append(param_ref._obj.struDayNight.byDayNightFilterType)
        self.set_call = (user_id, command, channel, size)
        return True

    def NET_DVR_GetLastError(self) -> int:
        return self.last_error


def test_reset_camera_day_night_mode_switches_night_day_auto(monkeypatch) -> None:  # noqa: ANN001
    fake_sdk = FakeSDK()
    sleeps: list[float] = []
    monkeypatch.setattr(stream_error_processing.time, "sleep", sleeps.append)

    original_mode = stream_error_processing.reset_camera_day_night_mode(
        fake_sdk,
        user_id=7,
        camera_param_cls=FakeCameraParam,
        get_command=3368,
        set_command=3369,
        channel=1,
        settle_seconds=0.2,
    )

    assert original_mode == 2
    assert fake_sdk.get_call == (7, 3368, 1, ctypes.sizeof(FakeCameraParam()))
    assert fake_sdk.set_modes == [1, 0, 2]
    assert sleeps == [0.2, 0.2]
