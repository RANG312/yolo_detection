from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types

import cv2
import numpy as np

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

import dial_reading


class FakeBoxes:
    def __init__(self, data: list[list[float]]) -> None:
        self.data = np.array(data, dtype=np.float64)

    def cpu(self) -> "FakeBoxes":
        return self

    def numpy(self) -> "FakeBoxes":
        return self


class FakeResult:
    names = {0: "gauge", 1: "center", 2: "pointer_tip", 3: "max_tick", 4: "min_tick"}

    def __init__(self, boxes: list[list[float]]) -> None:
        self.boxes = FakeBoxes(boxes)


class FakeModel:
    def __init__(self) -> None:
        self.sources: list[np.ndarray] = []

    def predict(self, source, imgsz, conf, device, verbose):  # noqa: ANN001, ANN201
        del imgsz, conf, device, verbose
        self.sources.append(source.copy())
        if len(self.sources) == 1:
            return [FakeResult([[20, 20, 180, 180, 0.95, 0]])]
        return [
            FakeResult(
                [
                    [300, 300, 340, 340, 0.90, 1],
                    [320, 120, 350, 150, 0.90, 2],
                    [500, 300, 530, 330, 0.90, 3],
                    [120, 300, 150, 330, 0.90, 4],
                ]
            )
        ]


class FakeRetryFailureModel:
    def __init__(self) -> None:
        self.sources: list[np.ndarray] = []

    def predict(self, source, imgsz, conf, device, verbose):  # noqa: ANN001, ANN201
        del imgsz, conf, device, verbose
        self.sources.append(source.copy())
        if len(self.sources) == 1:
            return [FakeResult([[20, 20, 180, 180, 0.95, 0]])]
        return [FakeResult([])]


def test_meter_data_9k_retries_on_resized_gauge_crop(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "meter.jpg"
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    def fake_compute(image, instance, annotation_mode, debug_center):  # noqa: ANN001, ANN202
        del annotation_mode, debug_center
        return {
            "gauge_box": instance["gauge"]["box"],
            "center_box": instance["center"]["box"],
            "min_tick_box": instance["min_tick"]["box"],
            "max_tick_box": instance["max_tick"]["box"],
            "pointer_tip_box": instance["pointer_tip"]["box"],
            "center": np.array([320.0, 320.0]),
            "start_tick": np.array([120.0, 320.0]),
            "end_tick": np.array([520.0, 320.0]),
            "pointer_tip": np.array([320.0, 120.0]),
            "ratio": 0.5,
            "arc_span": 180.0,
            "arc_mode": "test",
            "center_debug": {},
        }

    monkeypatch.setattr(dial_reading, "compute_reading_from_detection_instance", fake_compute)

    model = FakeModel()
    args = SimpleNamespace(imgsz=640, conf=0.25, device="cpu", min_value=0.0, max_value=1.0, debug_center=False)

    canvas, prediction_instances, _, _ = dial_reading.predict_image_instances(
        image_path, model, args, "meter_data_9k"
    )

    assert len(model.sources) == 2
    assert model.sources[1].shape[:2] == (640, 640)
    assert canvas.shape[:2] == (640, 640)
    assert prediction_instances[0]["error"] is None
    assert prediction_instances[0]["reading"] == 0.5


def test_meter_data_9k_returns_original_canvas_when_gauge_retry_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "meter.jpg"
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    model = FakeRetryFailureModel()
    args = SimpleNamespace(imgsz=640, conf=0.25, device="cpu", min_value=0.0, max_value=1.0, debug_center=False)

    canvas, prediction_instances, _, _ = dial_reading.predict_image_instances(
        image_path, model, args, "meter_data_9k"
    )

    assert len(model.sources) == 2
    assert model.sources[1].shape[:2] == (640, 640)
    assert canvas.shape[:2] == (240, 320)
    assert prediction_instances[0]["error"] == "Missing required detections: ['gauge']"
