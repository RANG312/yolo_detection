from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "recognition_http_server" / "utils" / "capture_gimbal_frames.py"
SPEC = importlib.util.spec_from_file_location("capture_gimbal_frames", SCRIPT_PATH)
capture_gimbal_frames = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(capture_gimbal_frames)


def test_parse_source_converts_integer_device_index() -> None:
    assert capture_gimbal_frames.parse_source("0") == 0
    assert capture_gimbal_frames.parse_source("rtsp://192.168.1.64/stream") == "rtsp://192.168.1.64/stream"


def test_mask_source_hides_url_password() -> None:
    assert (
        capture_gimbal_frames.mask_source("rtsp://admin:secret@10.42.0.120:554/Streaming/Channels/101")
        == "rtsp://admin:***@10.42.0.120:554/Streaming/Channels/101"
    )


def test_frame_statistics_reports_rgb_mean_std_and_magenta_metrics() -> None:
    image_bgr = [
        [(120, 20, 120), (10, 10, 10)],
        [(100, 60, 100), (0, 255, 0)],
    ]

    stats = capture_gimbal_frames.compute_frame_statistics(image_bgr)

    assert stats["mean_r"] == 57.5
    assert stats["mean_g"] == 86.25
    assert stats["mean_b"] == 57.5
    assert round(stats["std_g"], 3) == 99.208
    assert stats["green_deficit"] == 35.0
    assert stats["magenta_pct"] == 50.0


def test_frame_statistics_can_limit_to_roi() -> None:
    image_bgr = [
        [(0, 255, 0), (120, 20, 120)],
        [(0, 255, 0), (10, 10, 10)],
    ]

    stats = capture_gimbal_frames.compute_frame_statistics(image_bgr, roi=(1, 0, 1, 1))

    assert stats["mean_r"] == 120.0
    assert stats["mean_g"] == 20.0
    assert stats["mean_b"] == 120.0
    assert stats["green_deficit"] == 100.0
    assert stats["magenta_pct"] == 100.0
