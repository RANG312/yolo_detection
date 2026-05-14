from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class PTZAlignmentConfig:
    enabled: bool = False
    horizontal_fov_deg: float = 60.0
    vertical_fov_deg: float = 40.0
    threshold_deg: float = 1.0
    max_delta_deg: float = 10.0
    max_passes: int = 2
    zoom_enabled: bool = False
    zoom_target_height_ratio: float = 0.8
    zoom_ratio_tolerance: float = 0.05
    zoom_max_passes: int = 1


@dataclass(frozen=True)
class AutoPTZSettings:
    threshold_deg: float
    max_delta_deg: float
    nudge_degrees_per_second: float
    nudge_max_steps: int
    tilt_nudge_scale: float


@dataclass(frozen=True)
class PTZAlignmentRequest:
    image_width: int
    image_height: int
    gauge_box: tuple[float, float, float, float]
    dx_px: float
    dy_px: float
    pan_delta_deg: float
    tilt_delta_deg: float
    should_align: bool = False


@dataclass(frozen=True)
class PTZZoomRequest:
    image_width: int
    image_height: int
    gauge_box: tuple[float, float, float, float]
    gauge_height_px: float
    current_height_ratio: float
    target_height_ratio: float
    ratio_tolerance: float
    zoom_direction: str | None = None
    should_zoom: bool = False


class PTZAlignmentController(Protocol):
    def align(self, request: PTZAlignmentRequest) -> None:
        """Move the PTZ camera according to the requested relative angle."""

    def zoom(self, request: PTZZoomRequest) -> None:
        """Move the optical zoom according to the requested relative size change."""


def auto_tune_ptz_settings(horizontal_fov_deg: float, vertical_fov_deg: float) -> AutoPTZSettings:
    """Tune PTZ threshold and nudge settings from the current optical field of view."""
    narrow_fov = min(float(horizontal_fov_deg), float(vertical_fov_deg))
    wide_fov = max(float(horizontal_fov_deg), float(vertical_fov_deg))
    threshold_deg = round(max(0.02, min(0.1, narrow_fov / 80.0)), 2)
    max_delta_deg = round(max(0.3, min(5.0, wide_fov / 2.0)), 3)
    if wide_fov <= 5.0:
        nudge_degrees_per_second = 0.5
        nudge_max_steps = 4
        tilt_nudge_scale = 2.0
    elif wide_fov <= 15.0:
        nudge_degrees_per_second = 1.0
        nudge_max_steps = 3
        tilt_nudge_scale = 1.5
    else:
        nudge_degrees_per_second = 4.0
        nudge_max_steps = 2
        tilt_nudge_scale = 1.0
    return AutoPTZSettings(
        threshold_deg=threshold_deg,
        max_delta_deg=max_delta_deg,
        nudge_degrees_per_second=nudge_degrees_per_second,
        nudge_max_steps=nudge_max_steps,
        tilt_nudge_scale=tilt_nudge_scale,
    )


def compute_alignment_request(
    image_shape: tuple[int, ...],
    gauge_box: np.ndarray,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> PTZAlignmentRequest:
    """Map a gauge ROI center offset to approximate PTZ pan/tilt deltas."""
    height = int(image_shape[0])
    width = int(image_shape[1])
    box = np.asarray(gauge_box, dtype=np.float64)
    gauge_center_x = float((box[0] + box[2]) / 2.0)
    gauge_center_y = float((box[1] + box[3]) / 2.0)
    dx_px = gauge_center_x - width / 2.0
    dy_px = gauge_center_y - height / 2.0
    pan_delta_deg = dx_px / width * horizontal_fov_deg
    tilt_delta_deg = -dy_px / height * vertical_fov_deg
    return PTZAlignmentRequest(
        image_width=width,
        image_height=height,
        gauge_box=tuple(float(value) for value in box.tolist()),
        dx_px=float(dx_px),
        dy_px=float(dy_px),
        pan_delta_deg=float(pan_delta_deg),
        tilt_delta_deg=float(tilt_delta_deg),
    )


def compute_zoom_request(
    image_shape: tuple[int, ...],
    gauge_box: np.ndarray,
    target_height_ratio: float,
    ratio_tolerance: float,
) -> PTZZoomRequest:
    """Map a gauge ROI height ratio to an approximate zoom-in or zoom-out request."""
    height = int(image_shape[0])
    width = int(image_shape[1])
    box = np.asarray(gauge_box, dtype=np.float64)
    gauge_height_px = max(0.0, float(box[3] - box[1]))
    current_height_ratio = gauge_height_px / height if height > 0 else 0.0
    target_height_ratio = _clamp(target_height_ratio, 0.05, 0.98)
    ratio_tolerance = max(0.0, float(ratio_tolerance))
    zoom_direction = None
    should_zoom = False
    if current_height_ratio < target_height_ratio - ratio_tolerance:
        zoom_direction = "in"
        should_zoom = True
    elif current_height_ratio > target_height_ratio + ratio_tolerance:
        zoom_direction = "out"
        should_zoom = True
    return PTZZoomRequest(
        image_width=width,
        image_height=height,
        gauge_box=tuple(float(value) for value in box.tolist()),
        gauge_height_px=gauge_height_px,
        current_height_ratio=float(current_height_ratio),
        target_height_ratio=float(target_height_ratio),
        ratio_tolerance=float(ratio_tolerance),
        zoom_direction=zoom_direction,
        should_zoom=should_zoom,
    )


def maybe_align_gauge(
    controller: PTZAlignmentController | None,
    config: PTZAlignmentConfig,
    image_shape: tuple[int, ...],
    gauge_box: np.ndarray,
) -> PTZAlignmentRequest | None:
    """Compute and optionally execute PTZ alignment for a first-stage gauge ROI."""
    if not config.enabled:
        return None

    request = compute_alignment_request(
        image_shape=image_shape,
        gauge_box=gauge_box,
        horizontal_fov_deg=config.horizontal_fov_deg,
        vertical_fov_deg=config.vertical_fov_deg,
    )
    pan_delta = _clamp(request.pan_delta_deg, -config.max_delta_deg, config.max_delta_deg)
    tilt_delta = _clamp(request.tilt_delta_deg, -config.max_delta_deg, config.max_delta_deg)
    should_align = max(abs(pan_delta), abs(tilt_delta)) >= config.threshold_deg
    request = PTZAlignmentRequest(
        image_width=request.image_width,
        image_height=request.image_height,
        gauge_box=request.gauge_box,
        dx_px=request.dx_px,
        dy_px=request.dy_px,
        pan_delta_deg=pan_delta,
        tilt_delta_deg=tilt_delta,
        should_align=should_align,
    )
    if should_align and controller is not None:
        controller.align(request)
    return request


def maybe_zoom_gauge(
    controller: PTZAlignmentController | None,
    config: PTZAlignmentConfig,
    image_shape: tuple[int, ...],
    gauge_box: np.ndarray,
) -> PTZZoomRequest | None:
    """Compute and optionally execute optical zoom for a centered gauge ROI."""
    if not config.enabled or not config.zoom_enabled:
        return None

    request = compute_zoom_request(
        image_shape=image_shape,
        gauge_box=gauge_box,
        target_height_ratio=config.zoom_target_height_ratio,
        ratio_tolerance=config.zoom_ratio_tolerance,
    )
    if request.should_zoom and controller is not None and hasattr(controller, "zoom"):
        controller.zoom(request)
    return request


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))
