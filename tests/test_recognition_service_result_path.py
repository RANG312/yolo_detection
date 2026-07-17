from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

ultralytics_stub = types.ModuleType("ultralytics")
ultralytics_stub.YOLO = object
sys.modules.setdefault("ultralytics", ultralytics_stub)
cv2_stub = types.ModuleType("cv2")
sys.modules.setdefault("cv2", cv2_stub)
dial_reading_stub = types.ModuleType("recognition_http_server.dial_reading")
dial_reading_stub.load_model = lambda *args, **kwargs: (None, "legacy")
dial_reading_stub.predict_image_instances = lambda *args, **kwargs: None
dial_reading_stub.save_canvas = lambda *args, **kwargs: None
sys.modules.setdefault("recognition_http_server.dial_reading", dial_reading_stub)
detection_handler_stub = types.ModuleType("recognition_http_server.handlers.detection")
detection_handler_stub.run_detection_recognition = lambda *args, **kwargs: None
sys.modules.setdefault("recognition_http_server.handlers.detection", detection_handler_stub)
meter_handler_stub = types.ModuleType("recognition_http_server.handlers.meter")
meter_handler_stub.run_digital_meter_recognition = lambda *args, **kwargs: None
meter_handler_stub.run_pointer_meter_recognition = lambda *args, **kwargs: None
sys.modules.setdefault("recognition_http_server.handlers.meter", meter_handler_stub)
hikvision_ptz_stub = types.ModuleType("recognition_http_server.hikvision_ptz")
hikvision_ptz_stub.HikvisionPTZConfig = object
hikvision_ptz_stub.HikvisionPTZController = object
hikvision_ptz_stub.resolve_sdk_lib_dir = lambda *args, **kwargs: None
sys.modules.setdefault("recognition_http_server.hikvision_ptz", hikvision_ptz_stub)
virtual_ptz_stub = types.ModuleType("recognition_http_server.virtual_ptz")
virtual_ptz_stub.VirtualPTZConfig = object
virtual_ptz_stub.VirtualPTZController = object
sys.modules.setdefault("recognition_http_server.virtual_ptz", virtual_ptz_stub)
ptz_alignment_stub = types.ModuleType("recognition_http_server.ptz_alignment")
ptz_alignment_stub.PTZAlignmentConfig = object
sys.modules.setdefault("recognition_http_server.ptz_alignment", ptz_alignment_stub)

from recognition_http_server.service import RecognitionService, TaskHandler, resolve_task_callback_url


def test_task_callback_url_prefers_request_extra_info_callback_url() -> None:
    assert (
        resolve_task_callback_url({"callback_url": "http://127.0.0.1:4567/callback"}, "10.0.0.1", 8088)
        == "http://127.0.0.1:4567/callback"
    )


def test_task_callback_url_falls_back_to_task_host_and_configured_port() -> None:
    assert resolve_task_callback_url({}, "10.0.0.1", 8088) == "http://10.0.0.1:8088/api/v1/recognition/callback"


def test_create_task_records_timestamped_result_dir_name(monkeypatch) -> None:
    service = object.__new__(RecognitionService)
    service._tasks = {}
    service._tasks_lock = threading.Lock()
    service.logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    import recognition_http_server.service as service_module

    class ThreadStub:
        def __init__(self, *args, **kwargs):
            pass

        def start(self) -> None:
            pass

    monkeypatch.setattr(service_module, "time", SimpleNamespace(strftime=lambda fmt: "20260611_143012"))
    monkeypatch.setattr(service_module.threading, "Thread", ThreadStub)

    task = service.create_task(
        {
            "req_id": "req-1",
            "image_path": "a.jpg,b.jpg",
            "data_type": [{"recognize_type": "1", "recognize_subtype": "1"}],
        },
        "127.0.0.1",
    )

    assert task.result_dir_name == "20260611_143012_req-1"


def test_failed_recognition_copies_image_path_to_result_path(tmp_path: Path) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not used")

    service = object.__new__(RecognitionService)
    service.input_root = tmp_path / "inputs"
    service.output_root = tmp_path / "outputs"
    service.config = SimpleNamespace(request_timeout=1)
    service.logger = SimpleNamespace(info=lambda *args, **kwargs: None, exception=lambda *args, **kwargs: None)

    def prepare_image(*args, **kwargs):
        return image_path

    def resolve_task_kind(*args, **kwargs):
        return "meter"

    def runner(local_image_path, data_type, image_result_path, extra_info, debug_center):
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


def test_successful_meter_recognition_uses_timestamped_result_dir(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "input.jpg"
    image_path.write_bytes(b"not used")
    aligned_path = image_path.with_name("input_aligned.jpg")
    aligned_path.write_bytes(b"aligned")

    service = object.__new__(RecognitionService)
    service.input_root = tmp_path / "inputs"
    service.output_root = tmp_path / "outputs"
    service.config = SimpleNamespace(request_timeout=1)
    service.logger = SimpleNamespace(info=lambda *args, **kwargs: None, exception=lambda *args, **kwargs: None)

    def prepare_image(*args, **kwargs):
        return image_path

    def resolve_task_kind(*args, **kwargs):
        return "meter"

    result_dir_names = []

    def runner(local_image_path, data_type, image_result_path, extra_info, debug_center):
        del local_image_path, extra_info, debug_center
        result_dir_names.append(image_result_path.parent.name)
        image_result_path.write_bytes(b"result")
        return [
            {
                "recognize_type": data_type["recognize_type"],
                "recognize_subtype": data_type["recognize_subtype"],
                "recognize_value": "1.000000",
                "confidence": "100",
                "recognize_desc": "ok",
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
        monkeypatch.setattr(
            service_module,
            "time",
            SimpleNamespace(strftime=lambda fmt: "20260611_143012", perf_counter=service_module.time.perf_counter),
        )
        result = service._process_single_image(
            "req-1",
            1,
            str(image_path),
            {"recognize_type": "1", "recognize_subtype": "1"},
            {},
            False,
        )
    finally:
        service_module.prepare_image = original_prepare_image
        service_module.resolve_task_kind = original_resolve_task_kind

    assert result_dir_names == ["20260611_143012_req-1"]
    assert result["image_path"] == str(aligned_path)
    assert result["image_path_result"] == str(image_path.with_name("input-detection.jpg"))
