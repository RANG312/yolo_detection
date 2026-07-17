from __future__ import annotations

from types import SimpleNamespace

from recognition_http_server.handlers.meter import resolve_current_ptz_alignment_config, resolve_current_ptz_runtime
from recognition_http_server.hikvision_ptz import HikvisionPTZConfig, HikvisionPTZController
from recognition_http_server.ptz_alignment import PTZAlignmentConfig


def test_resolve_current_ptz_alignment_config_reads_current_sdk_fov() -> None:
    class FakeController:
        def read_field_of_view(self):
            return SimpleNamespace(horizontal_deg=2.9, vertical_deg=1.63)

    service = SimpleNamespace(
        ptz_alignment_config=PTZAlignmentConfig(
            enabled=True,
            horizontal_fov_deg=60.0,
            vertical_fov_deg=40.0,
            threshold_deg=0.1,
            max_delta_deg=5.0,
            max_passes=3,
            zoom_enabled=True,
            zoom_target_height_ratio=0.8,
            zoom_ratio_tolerance=0.05,
            zoom_max_passes=2,
        ),
        ptz_controller=FakeController(),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    config = resolve_current_ptz_alignment_config(service)

    assert config.horizontal_fov_deg == 2.9
    assert config.vertical_fov_deg == 1.63
    assert config.threshold_deg == 0.02
    assert config.max_delta_deg == 1.45
    assert config.zoom_enabled is True


def test_resolve_current_ptz_runtime_preserves_device_motion_profile() -> None:
    class FakeController(HikvisionPTZController):
        def read_field_of_view(self):
            return SimpleNamespace(horizontal_deg=7.48, vertical_deg=4.21)

    service = SimpleNamespace(
        ptz_alignment_config=PTZAlignmentConfig(
            enabled=True,
            horizontal_fov_deg=60.0,
            vertical_fov_deg=40.0,
            threshold_deg=1.0,
            max_delta_deg=10.0,
        ),
        ptz_controller=FakeController(
            HikvisionPTZConfig(
                host="192.168.1.64",
                password="secret",
                pan_nudge_degrees_per_second=12.0,
                tilt_nudge_degrees_per_second=5.5,
            )
        ),
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    config, controller = resolve_current_ptz_runtime(service)

    assert config.horizontal_fov_deg == 7.48
    assert config.vertical_fov_deg == 4.21
    assert config.threshold_deg == 0.05
    assert config.max_delta_deg == 3.74
    assert controller.config.pan_nudge_degrees_per_second == 12.0
    assert controller.config.tilt_nudge_degrees_per_second == 5.5
    assert controller.config.nudge_max_steps == 2
