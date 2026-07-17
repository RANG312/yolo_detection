from __future__ import annotations

from pathlib import Path

import numpy as np

from recognition_http_server.ptz_alignment import PTZAlignmentRequest, PTZZoomRequest
from recognition_http_server.virtual_ptz import VirtualPTZConfig, VirtualPTZController


def test_virtual_ptz_records_motion_and_returns_static_frame(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "meter.jpg"
    image_path.touch()
    frame = np.zeros((16, 24, 3), dtype=np.uint8)
    monkeypatch.setattr("recognition_http_server.virtual_ptz.cv2.imread", lambda path: frame)
    controller = VirtualPTZController(VirtualPTZConfig(image_path=image_path))
    align_request = PTZAlignmentRequest(
        image_width=24,
        image_height=16,
        gauge_box=(1.0, 2.0, 8.0, 12.0),
        dx_px=2.0,
        dy_px=-1.0,
        pan_delta_deg=1.5,
        tilt_delta_deg=0.5,
        should_align=True,
    )
    zoom_request = PTZZoomRequest(
        image_width=24,
        image_height=16,
        gauge_box=(1.0, 2.0, 8.0, 12.0),
        gauge_height_px=10.0,
        current_height_ratio=0.625,
        target_height_ratio=0.8,
        ratio_tolerance=0.05,
        zoom_direction="in",
        should_zoom=True,
    )

    controller.align(align_request)
    controller.zoom(zoom_request)
    captured = controller.capture_image()

    assert controller.alignment_requests == [align_request]
    assert controller.zoom_requests == [zoom_request]
    assert np.array_equal(captured, frame)
    assert captured is not frame


def test_virtual_ptz_exposes_configured_field_of_view() -> None:
    controller = VirtualPTZController(
        VirtualPTZConfig(horizontal_fov_deg=12.5, vertical_fov_deg=7.25, zoom=3.0, max_zoom=8.0)
    )

    fov = controller.read_field_of_view()
    limits = controller.read_zoom_limits()

    assert fov.horizontal_deg == 12.5
    assert fov.vertical_deg == 7.25
    assert limits.current_zoom == 3.0
    assert limits.max_zoom == 8.0
