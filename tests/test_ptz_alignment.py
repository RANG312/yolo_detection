from __future__ import annotations

import numpy as np

from recognition_http_server.ptz_alignment import (
    PTZAlignmentConfig,
    PTZAlignmentRequest,
    PTZZoomRequest,
    compute_alignment_request,
    compute_zoom_request,
    maybe_align_gauge,
    maybe_zoom_gauge,
)


class FakeController:
    def __init__(self) -> None:
        self.requests: list[PTZAlignmentRequest] = []
        self.zoom_requests: list[PTZZoomRequest] = []

    def align(self, request: PTZAlignmentRequest) -> None:
        self.requests.append(request)

    def zoom(self, request: PTZZoomRequest) -> None:
        self.zoom_requests.append(request)


def test_compute_alignment_request_maps_image_offset_to_fov_degrees() -> None:
    request = compute_alignment_request(
        image_shape=(1000, 2000, 3),
        gauge_box=np.array([1200, 400, 1600, 800], dtype=np.float64),
        horizontal_fov_deg=60.0,
        vertical_fov_deg=40.0,
    )

    assert request.dx_px == 400.0
    assert request.dy_px == 100.0
    assert request.pan_delta_deg == 12.0
    assert request.tilt_delta_deg == -4.0


def test_maybe_align_gauge_skips_small_offsets() -> None:
    controller = FakeController()
    config = PTZAlignmentConfig(enabled=True, threshold_deg=5.0)

    request = maybe_align_gauge(
        controller,
        config,
        image_shape=(1000, 2000, 3),
        gauge_box=np.array([980, 490, 1020, 530], dtype=np.float64),
    )

    assert request is not None
    assert request.should_align is False
    assert controller.requests == []


def test_maybe_align_gauge_calls_controller_when_offset_exceeds_threshold() -> None:
    controller = FakeController()
    config = PTZAlignmentConfig(
        enabled=True,
        horizontal_fov_deg=60.0,
        vertical_fov_deg=40.0,
        threshold_deg=1.0,
        max_delta_deg=8.0,
    )

    request = maybe_align_gauge(
        controller,
        config,
        image_shape=(1000, 2000, 3),
        gauge_box=np.array([1200, 400, 1600, 800], dtype=np.float64),
    )

    assert request is not None
    assert request.should_align is True
    assert request.pan_delta_deg == 8.0
    assert request.tilt_delta_deg == -4.0
    assert controller.requests == [request]


def test_maybe_align_gauge_does_nothing_when_disabled() -> None:
    controller = FakeController()
    config = PTZAlignmentConfig(enabled=False)

    request = maybe_align_gauge(
        controller,
        config,
        image_shape=(1000, 2000, 3),
        gauge_box=np.array([1200, 400, 1600, 800], dtype=np.float64),
    )

    assert request is None
    assert controller.requests == []


def test_compute_zoom_request_zooms_in_when_gauge_height_is_below_target() -> None:
    request = compute_zoom_request(
        image_shape=(1000, 2000, 3),
        gauge_box=np.array([500, 350, 1500, 650], dtype=np.float64),
        target_height_ratio=0.8,
        ratio_tolerance=0.05,
    )

    assert request.gauge_height_px == 300.0
    assert request.current_height_ratio == 0.3
    assert request.target_height_ratio == 0.8
    assert request.zoom_direction == "in"
    assert request.should_zoom is True


def test_maybe_zoom_gauge_calls_controller_when_size_is_outside_tolerance() -> None:
    controller = FakeController()
    config = PTZAlignmentConfig(enabled=True, zoom_enabled=True, zoom_target_height_ratio=0.8)

    request = maybe_zoom_gauge(
        controller,
        config,
        image_shape=(1000, 2000, 3),
        gauge_box=np.array([500, 350, 1500, 650], dtype=np.float64),
    )

    assert request is not None
    assert request.should_zoom is True
    assert request.zoom_direction == "in"
    assert controller.zoom_requests == [request]
