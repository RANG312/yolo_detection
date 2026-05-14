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


def test_read_initial_hikvision_position_continues_when_restore_probe_fails(monkeypatch, capsys) -> None:  # noqa: ANN001
    class BrokenController:
        def read_position(self):  # noqa: ANN201
            raise RuntimeError("NET_DVR_Login_V30 failed: host=192.168.1.64 error=9")

    args = Namespace(hik_capture=True, no_hik_restore=False)
    monkeypatch.setattr(test_http, "create_hikvision_controller", lambda args_arg: BrokenController())

    assert test_http.read_initial_hikvision_position(args) is None

    captured = capsys.readouterr()
    assert "WARNING: failed to read initial Hikvision PTZ position" in captured.out


def test_start_callback_server_updates_callback_port_when_ephemeral_port_is_requested() -> None:
    args = Namespace(callback_host="127.0.0.1", callback_port=0, callback_path="/callback")
    state = test_http.CallbackState()
    server = test_http.start_callback_server(args, state)
    try:
        assert args.callback_port > 0
        assert test_http.build_callback_url(
            Namespace(callback_url=None, host="127.0.0.1", callback_port=args.callback_port, callback_path="/callback")
        ) == f"http://127.0.0.1:{args.callback_port}/callback"
    finally:
        server.shutdown()
        server.server_close()
