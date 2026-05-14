from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import test_http


def test_resolve_images_requires_images_without_hik_capture() -> None:
    args = Namespace(images=[], hik_capture=False)

    try:
        test_http.resolve_image_paths(args)
    except ValueError as exc:
        assert "--images is required unless --hik-capture is set." in str(exc)
    else:
        raise AssertionError("resolve_image_paths should reject empty images without --hik-capture")


def test_resolve_images_uses_hikvision_capture_path(monkeypatch, tmp_path: Path) -> None:  # noqa: ANN001
    captured_path = tmp_path / "captured.jpg"
    args = Namespace(images=[], hik_capture=True)

    def fake_capture(args):  # noqa: ANN001, ANN202
        return captured_path

    monkeypatch.setattr(test_http, "capture_hikvision_image", fake_capture)

    assert test_http.resolve_image_paths(args) == [str(captured_path)]


def test_restore_hikvision_position_after_delay_waits_and_restores(monkeypatch) -> None:  # noqa: ANN001
    calls = []
    position = SimpleNamespace(pan_deg=5.0, tilt_deg=1.0, zoom_deg=12.0)
    controller = type("FakeController", (), {"set_position": lambda self, value: calls.append(("set", value))})()
    args = Namespace(hik_restore_delay=10.0)

    monkeypatch.setattr(test_http.time, "sleep", lambda seconds: calls.append(("sleep", seconds)))
    monkeypatch.setattr(test_http, "create_hikvision_controller", lambda args_arg: controller)

    test_http.restore_hikvision_position_after_delay(args, position)

    assert calls == [("sleep", 10.0), ("set", position)]
