from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from recognition_http_server.dial_reading import (
    LEGACY_EXPECTED_CLASSES,
    METER_DATA_9K_EXPECTED_CLASSES,
    METER_DATA_9K_GAUGE_RETRY_SIZE,
    angle_from_center,
    box_center,
    ccw_distance,
    clip_box,
    compute_reading_from_detection_instance,
    crop_roi,
    detect_center_from_roi,
    detect_pointer_tip,
    detect_tick_point,
    infer_annotation_mode,
    load_model,
    save_canvas,
    select_arc_and_ratio,
)
from recognition_http_server.dial_reading import pipeline as _pipeline
from recognition_http_server.dial_reading.cli import main, parse_args, run_test_loop
from recognition_http_server.dial_reading.pipeline import predict_single_image
from ultralytics import YOLO


# 这个文件只保留旧入口兼容层。新代码应优先从
# `recognition_http_server.dial_reading` 子包导入具体能力。
def predict_image_instances(
    image_path: Path,
    model: YOLO,
    args: Any,
    annotation_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]], float, float]:
    """Backward-compatible wrapper for callers that monkeypatch this legacy module.

    Older tests and scripts patch `dial_reading.compute_reading_from_detection_instance`. The real implementation now
    lives in `recognition_http_server.dial_reading.pipeline`, so this wrapper mirrors the patched function into the
    pipeline before executing.
    """
    _pipeline.compute_reading_from_detection_instance = compute_reading_from_detection_instance
    return _pipeline.predict_image_instances(image_path, model, args, annotation_mode)


__all__ = [
    "LEGACY_EXPECTED_CLASSES",
    "METER_DATA_9K_EXPECTED_CLASSES",
    "METER_DATA_9K_GAUGE_RETRY_SIZE",
    "angle_from_center",
    "box_center",
    "ccw_distance",
    "clip_box",
    "compute_reading_from_detection_instance",
    "crop_roi",
    "detect_center_from_roi",
    "detect_pointer_tip",
    "detect_tick_point",
    "infer_annotation_mode",
    "load_model",
    "main",
    "parse_args",
    "predict_image_instances",
    "predict_single_image",
    "run_test_loop",
    "save_canvas",
    "select_arc_and_ratio",
]


if __name__ == "__main__":
    main()
