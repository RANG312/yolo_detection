from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

cv2_stub = types.ModuleType("cv2")
cv2_stub.COLOR_BGR2RGB = 0
sys.modules.setdefault("cv2", cv2_stub)

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

dial_reading_stub = types.ModuleType("recognition_http_server.dial_reading")
dial_reading_stub.load_model = lambda *args, **kwargs: (object(), "stub")
dial_reading_stub.predict_image_instances = lambda *args, **kwargs: []
dial_reading_stub.save_canvas = lambda *args, **kwargs: None
sys.modules.setdefault("recognition_http_server.dial_reading", dial_reading_stub)

from recognition_http_server.constants import (
    TASK_KIND_FIRE_EXTINGUISHER,
    TASK_KIND_FIRE_PROTECTION_FACILITIES,
    TASK_KIND_PERSON_AND_CARS,
    TASK_KIND_PERSON_FALL_DOWN,
)
from recognition_http_server.service import RecognitionService


def test_new_detection_task_handlers_are_registered() -> None:
    service = object.__new__(RecognitionService)
    service.fire_protection_facilities_model = object()
    service.person_fall_down_model = object()
    service.fire_extinguisher_model = object()
    service.person_and_cars_model = object()
    service.config = SimpleNamespace()
    service.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    handlers = service._build_task_handlers()

    assert handlers[TASK_KIND_FIRE_PROTECTION_FACILITIES].description == "消防设施检测"
    assert handlers[TASK_KIND_PERSON_FALL_DOWN].description == "摔倒检测"
    assert handlers[TASK_KIND_FIRE_EXTINGUISHER].description == "灭火器检测"
    assert handlers[TASK_KIND_PERSON_AND_CARS].description == "人车检测"


def test_recognition_service_loads_new_detection_models() -> None:
    import recognition_http_server.service as service_module

    loaded_paths: list[str] = []

    class FakeYOLO:
        def __init__(self, model_path: str) -> None:
            loaded_paths.append(model_path)

    original_yolo = service_module.YOLO
    original_load_model = service_module.load_model
    service_module.YOLO = FakeYOLO
    service_module.load_model = lambda model_path: (f"meter:{model_path}", "stub")
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = SimpleNamespace(
                result_root=Path(temp_dir),
                model="meter.pt",
                fire_model="fire.pt",
                safehat_model="safehat.pt",
                fire_protection_facilities_model="facilities.pt",
                person_fall_down_model="fall.pt",
                fire_extinguisher_model="extinguisher.pt",
                person_and_cars_model="cars.pt",
            )

            service = RecognitionService(config)
    finally:
        service_module.YOLO = original_yolo
        service_module.load_model = original_load_model

    assert loaded_paths == [
        "fire.pt",
        "safehat.pt",
        "facilities.pt",
        "fall.pt",
        "extinguisher.pt",
        "cars.pt",
    ]
    assert service.fire_protection_facilities_model is not None
    assert service.person_fall_down_model is not None
    assert service.fire_extinguisher_model is not None
    assert service.person_and_cars_model is not None
