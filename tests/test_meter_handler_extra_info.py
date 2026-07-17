from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from recognition_http_server.service import RecognitionService


def test_pointer_meter_task_forwards_extra_info_to_runner(monkeypatch, tmp_path: Path) -> None:
    service = object.__new__(RecognitionService)
    service.config = SimpleNamespace(max_value=1.0)

    received_extra_info = None

    def fake_run_pointer_meter_recognition(
        service_arg,
        local_image_path,
        data_types,
        visualize_path,
        extra_info,
        debug_center,
        scale,
    ):
        nonlocal received_extra_info
        del service_arg, local_image_path, data_types, visualize_path, debug_center, scale
        received_extra_info = extra_info
        return []

    import recognition_http_server.service as service_module

    monkeypatch.setattr(service_module, "run_pointer_meter_recognition", fake_run_pointer_meter_recognition)

    extra_info = {"_ptz_capture_dir": str(tmp_path)}
    service._handle_meter_task(
        tmp_path / "meter.jpg",
        {"recognize_type": "1", "recognize_subtype": "25"},
        tmp_path / "result.jpg",
        extra_info,
        False,
    )

    assert received_extra_info == extra_info
