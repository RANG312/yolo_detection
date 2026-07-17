from __future__ import annotations

import sys

from recognition_http_server.config import parse_args


def test_parse_args_uses_hikvision_environment_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["server.py"])
    monkeypatch.setenv("HIK_HOST", "10.42.0.120")
    monkeypatch.setenv("HIK_USERNAME", "operator")
    monkeypatch.setenv("HIK_PASSWORD", "secret")
    monkeypatch.setenv("HIK_PORT", "9000")
    monkeypatch.setenv("HIK_CHANNEL", "2")
    monkeypatch.setenv("HIK_LOCAL_IP", "192.168.2.171")

    args = parse_args()

    assert args.ptz_host == "10.42.0.120"
    assert args.ptz_username == "operator"
    assert args.ptz_password == "secret"
    assert args.ptz_port == 9000
    assert args.ptz_channel == 2
    assert args.ptz_local_ip == "192.168.2.171"


def test_parse_args_supports_virtual_ptz_controller(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["server.py", "--ptz-controller", "virtual", "--virtual-ptz-image", "/tmp/meter.jpg"],
    )

    args = parse_args()

    assert args.ptz_controller == "virtual"
    assert args.virtual_ptz_image == "/tmp/meter.jpg"
