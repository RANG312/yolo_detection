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
from recognition_http_server.ptz_alignment import PTZAlignmentConfig


def test_dial_reading_package_reexports_public_entrypoints() -> None:
    from recognition_http_server import dial_reading as package

    assert package.load_model is not None
    assert package.predict_image_instances is not None
    assert package.predict_single_image is not None
    assert package.save_canvas is not None


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


class FakeMultiGaugeModel:
    def __init__(self) -> None:
        self.sources: list[np.ndarray] = []

    def predict(self, source, imgsz, conf, device, verbose):  # noqa: ANN001, ANN201
        del imgsz, conf, device, verbose
        self.sources.append(source.copy())
        if len(self.sources) == 1:
            return [
                FakeResult(
                    [
                        [0, 0, 120, 240, 0.30, 0],
                        [200, 20, 319, 220, 0.96, 0],
                    ]
                )
            ]
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


class FakeTwoPassAlignmentModel:
    def __init__(self) -> None:
        self.sources: list[np.ndarray] = []

    def predict(self, source, imgsz, conf, device, verbose):  # noqa: ANN001, ANN201
        del imgsz, conf, device, verbose
        self.sources.append(source.copy())
        if len(self.sources) == 1:
            return [FakeResult([[220, 20, 319, 220, 0.96, 0]])]
        if len(self.sources) == 2:
            return [FakeResult([[170, 40, 300, 220, 0.96, 0]])]
        if len(self.sources) == 3:
            return [FakeResult([[120, 40, 260, 220, 0.96, 0]])]
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


class FakeAlignmentZoomModel(FakeTwoPassAlignmentModel):
    def predict(self, source, imgsz, conf, device, verbose):  # noqa: ANN001, ANN201
        del imgsz, conf, device, verbose
        self.sources.append(source.copy())
        if len(self.sources) == 1:
            return [FakeResult([[220, 20, 319, 220, 0.96, 0]])]
        if len(self.sources) == 2:
            return [FakeResult([[170, 40, 300, 220, 0.96, 0]])]
        if len(self.sources) == 3:
            return [FakeResult([[120, 40, 260, 220, 0.96, 0]])]
        if len(self.sources) == 4:
            return [FakeResult([[80, 20, 280, 230, 0.96, 0]])]
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


class FakeController:
    def __init__(self) -> None:
        self.requests = []
        self.zoom_requests = []

    def align(self, request):  # noqa: ANN001, ANN202
        self.requests.append(request)

    def zoom(self, request):  # noqa: ANN001, ANN202
        self.zoom_requests.append(request)


class FakeCaptureController(FakeController):
    def __init__(self, image: np.ndarray) -> None:
        super().__init__()
        self.image = image

    def capture_image(self) -> np.ndarray:
        return self.image.copy()


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


def test_meter_data_9k_retries_on_highest_confidence_gauge(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "meter.jpg"
    image = np.full((240, 320, 3), 20, dtype=np.uint8)
    image[:, 180:] = 220
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

    model = FakeMultiGaugeModel()
    args = SimpleNamespace(imgsz=640, conf=0.25, device="cpu", min_value=0.0, max_value=1.0, debug_center=False)

    _, prediction_instances, _, _ = dial_reading.predict_image_instances(image_path, model, args, "meter_data_9k")

    assert prediction_instances[0]["error"] is None
    assert len(model.sources) == 2
    assert float(model.sources[1].mean()) > 150.0


def test_meter_data_9k_runs_ptz_alignment_after_first_gauge_detection(tmp_path: Path, monkeypatch) -> None:
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

    controller = FakeController()
    model = FakeModel()
    args = SimpleNamespace(
        imgsz=640,
        conf=0.25,
        device="cpu",
        min_value=0.0,
        max_value=1.0,
        debug_center=False,
        ptz_alignment_config=PTZAlignmentConfig(
            enabled=True,
            horizontal_fov_deg=60.0,
            vertical_fov_deg=40.0,
            threshold_deg=0.1,
            max_delta_deg=10.0,
        ),
        ptz_controller=controller,
    )

    _, prediction_instances, _, _ = dial_reading.predict_image_instances(image_path, model, args, "meter_data_9k")

    assert prediction_instances[0]["error"] is None
    assert len(controller.requests) == 1
    assert controller.requests[0].should_align is True
    assert controller.requests[0].pan_delta_deg == -10.0


def test_meter_data_9k_runs_second_alignment_on_captured_image(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "meter.jpg"
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    captured_image = np.full((240, 320, 3), 180, dtype=np.uint8)
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

    controller = FakeCaptureController(captured_image)
    model = FakeTwoPassAlignmentModel()
    args = SimpleNamespace(
        imgsz=640,
        conf=0.25,
        device="cpu",
        min_value=0.0,
        max_value=1.0,
        debug_center=False,
        ptz_alignment_config=PTZAlignmentConfig(
            enabled=True,
            horizontal_fov_deg=60.0,
            vertical_fov_deg=40.0,
            threshold_deg=0.1,
            max_delta_deg=20.0,
            max_passes=2,
        ),
        ptz_controller=controller,
    )

    _, prediction_instances, _, _ = dial_reading.predict_image_instances(image_path, model, args, "meter_data_9k")

    assert prediction_instances[0]["error"] is None
    assert len(controller.requests) == 2
    assert len(model.sources) == 4
    assert np.array_equal(model.sources[1], captured_image)
    assert np.array_equal(model.sources[2], captured_image)


def test_meter_data_9k_saves_ptz_captures_and_final_aligned_image(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "meter.jpg"
    image = np.full((240, 320, 3), 255, dtype=np.uint8)
    captured_image = np.full((240, 320, 3), 180, dtype=np.uint8)
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

    controller = FakeCaptureController(captured_image)
    model = FakeAlignmentZoomModel()
    aligned_path = tmp_path / "meter_aligned.jpg"
    args = SimpleNamespace(
        imgsz=640,
        conf=0.25,
        device="cpu",
        min_value=0.0,
        max_value=1.0,
        debug_center=False,
        ptz_alignment_config=PTZAlignmentConfig(
            enabled=True,
            horizontal_fov_deg=60.0,
            vertical_fov_deg=40.0,
            threshold_deg=0.1,
            max_delta_deg=20.0,
            max_passes=2,
            zoom_enabled=True,
            zoom_target_height_ratio=0.8,
            zoom_ratio_tolerance=0.01,
            zoom_max_passes=1,
        ),
        ptz_controller=controller,
        ptz_capture_dir=tmp_path,
        ptz_capture_stem="meter",
        ptz_aligned_image_path=aligned_path,
    )

    _, prediction_instances, _, _ = dial_reading.predict_image_instances(image_path, model, args, "meter_data_9k")

    assert prediction_instances[0]["error"] is None
    assert len(controller.requests) == 2
    assert len(controller.zoom_requests) == 1
    assert (tmp_path / "meter_align_1.jpg").exists()
    assert (tmp_path / "meter_align_2.jpg").exists()
    assert (tmp_path / "meter_zoom_1.jpg").exists()
    assert not (tmp_path / "meter_align_after_zoom_1.jpg").exists()
    assert aligned_path.exists()


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
