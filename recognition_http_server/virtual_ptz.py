from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from recognition_http_server.hikvision_ptz import HikvisionFieldOfView
from recognition_http_server.ptz_alignment import PTZAlignmentRequest, PTZZoomLimits, PTZZoomRequest


@dataclass(frozen=True)
class VirtualPTZConfig:
    """Configuration for a static-frame PTZ backend used without physical hardware."""

    image_path: Path | None = None
    horizontal_fov_deg: float = 60.0
    vertical_fov_deg: float = 40.0
    zoom: float = 1.0
    max_zoom: float = 100.0


class VirtualPTZController:
    """Record PTZ requests and return a static frame for deployment validation."""

    def __init__(self, config: VirtualPTZConfig, logger=None) -> None:
        self.config = config
        self.logger = logger
        self.alignment_requests: list[PTZAlignmentRequest] = []
        self.zoom_requests: list[PTZZoomRequest] = []

    def close(self) -> None:
        """Match the physical controller lifecycle API."""

    def align(self, request: PTZAlignmentRequest) -> None:
        self.alignment_requests.append(request)
        if self.logger is not None:
            self.logger.info(
                "virtual ptz aligned: pan_delta=%.3f tilt_delta=%.3f",
                request.pan_delta_deg,
                request.tilt_delta_deg,
            )

    def zoom(self, request: PTZZoomRequest) -> None:
        self.zoom_requests.append(request)
        if self.logger is not None:
            self.logger.info("virtual ptz zoomed: direction=%s", request.zoom_direction)

    def capture_image(self) -> np.ndarray:
        image_path = self.config.image_path
        if image_path is None:
            raise ValueError("Virtual PTZ capture requires --virtual-ptz-image.")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read virtual PTZ image: {image_path}")
        return image.copy()

    def read_field_of_view(self) -> HikvisionFieldOfView:
        horizontal = float(self.config.horizontal_fov_deg)
        vertical = float(self.config.vertical_fov_deg)
        return HikvisionFieldOfView(
            horizontal_deg=horizontal,
            vertical_deg=vertical,
            min_horizontal_deg=horizontal,
            max_horizontal_deg=horizontal,
            min_vertical_deg=vertical,
            max_vertical_deg=vertical,
            zoom=float(self.config.zoom),
        )

    def read_zoom_limits(self) -> PTZZoomLimits:
        return PTZZoomLimits(current_zoom=float(self.config.zoom), max_zoom=float(self.config.max_zoom))
