from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import types

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)

from recognition_http_server.service import RecognitionService, TaskHandler


def test_failed_recognition_copies_image_path_to_result_path(tmp_path: Path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not used")

    service = object.__new__(RecognitionService)
    service.input_root = tmp_path / "inputs"
    service.output_root = tmp_path / "outputs"
    service.config = SimpleNamespace(request_timeout=1)
    service.logger = SimpleNamespace(info=lambda *args, **kwargs: None, exception=lambda *args, **kwargs: None)

    def prepare_image(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return image_path

    def resolve_task_kind(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return "meter"

    def runner(local_image_path, data_type, image_result_path, extra_info, debug_center):  # noqa: ANN001, ANN202
        del local_image_path, image_result_path, extra_info, debug_center
        return [
            {
                "recognize_type": data_type["recognize_type"],
                "recognize_subtype": data_type["recognize_subtype"],
                "recognize_value": "",
                "confidence": "0",
                "recognize_desc": "识别失败: Missing required detections",
                "recognize_image_index": "1",
            }
        ]

    service._task_handlers = {"meter": TaskHandler("meter", "meter", "表计读数", runner)}

    import recognition_http_server.service as service_module

    original_prepare_image = service_module.prepare_image
    original_resolve_task_kind = service_module.resolve_task_kind
    try:
        service_module.prepare_image = prepare_image
        service_module.resolve_task_kind = resolve_task_kind
        requested_image_path = "data/original/input.jpg"
        result = service._process_single_image(
            "req-1",
            1,
            requested_image_path,
            {"recognize_type": "1", "recognize_subtype": "1"},
            {},
            False,
        )
    finally:
        service_module.prepare_image = original_prepare_image
        service_module.resolve_task_kind = original_resolve_task_kind

    assert result["image_path"] == requested_image_path
    assert result["image_path_result"] == requested_image_path
