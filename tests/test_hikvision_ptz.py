from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

import recognition_http_server.hikvision_ptz as hikvision_ptz
from recognition_http_server.hikvision_ptz import HikvisionFieldOfView, HikvisionPTZConfig, PTZPosition


def test_default_sdk_lib_dir_uses_arm64_sdk_for_aarch64(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(hikvision_ptz, "SDK_BASE_DIR", tmp_path / "HK_SDK")

    expected = tmp_path / "HK_SDK" / "HK_SDK_arm64_Linux" / "lib" / "linux"

    assert hikvision_ptz._default_sdk_lib_dir("aarch64") == expected


def test_default_sdk_lib_dir_prefers_flat_x86_sdk_when_present(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(hikvision_ptz, "SDK_BASE_DIR", tmp_path / "HK_SDK")
    expected = tmp_path / "HK_SDK" / "HK_SDK_x86_Linux" / "lib" / "linux"
    expected.mkdir(parents=True)

    assert hikvision_ptz._default_sdk_lib_dir("x86_64") == expected


def test_default_sdk_lib_dir_falls_back_to_nested_x86_sdk(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(hikvision_ptz, "SDK_BASE_DIR", tmp_path / "HK_SDK")

    expected = (
        tmp_path
        / "HK_SDK"
        / "HK_SDK_x86_Linux"
        / "HCNetSDKV6.1.11.5_build20251204_linux64_ZH"
        / "库文件"
    )
    expected.mkdir(parents=True)

    assert hikvision_ptz._default_sdk_lib_dir("x86_64") == expected


def test_resolve_sdk_lib_dir_uses_explicit_override() -> None:
    assert hikvision_ptz.resolve_sdk_lib_dir("/opt/hikvision/lib") == Path("/opt/hikvision/lib")


def test_configure_sdk_paths_accepts_openssl_1_1_dependencies(tmp_path) -> None:
    calls: list[tuple[int, bytes]] = []
    for name in ("libcrypto.so.1.1", "libssl.so.1.1"):
        (tmp_path / name).touch()

    class FakeSDKPathConfig:
        def NET_DVR_SetSDKInitCfg(self, cfg_type, value):  # noqa: ANN001, ANN202
            if cfg_type in {3, 4}:
                calls.append((cfg_type, ctypes.cast(value, ctypes.c_char_p).value))
            return True

    hikvision_ptz._configure_sdk_paths(FakeSDKPathConfig(), tmp_path)

    assert calls == [
        (3, str(tmp_path / "libcrypto.so.1.1").encode("utf-8")),
        (4, str(tmp_path / "libssl.so.1.1").encode("utf-8")),
    ]


class FakeSDK:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, int, int, int]] = []

    def NET_DVR_PTZControlWithSpeed_Other(self, user_id, channel, command, stop, speed):  # noqa: ANN001, ANN202
        self.calls.append((user_id, channel, command, stop, speed))
        return True


def test_nudge_by_delta_splits_large_pan_into_two_correction_steps(monkeypatch) -> None:
    sleep_durations: list[float] = []
    monkeypatch.setattr(hikvision_ptz.time, "sleep", sleep_durations.append)

    sdk = FakeSDK()
    config = HikvisionPTZConfig(
        channel=1,
        nudge_speed=1,
        nudge_degrees_per_second=8.0,
        nudge_max_seconds=1.0,
        nudge_max_steps=2,
    )

    hikvision_ptz._nudge_by_delta(sdk, 3, config, pan_delta_deg=20.0, tilt_delta_deg=0.0)

    assert sleep_durations == [1.0, 1.0]
    assert sdk.calls == [
        (3, 1, hikvision_ptz.PTZ_COMMANDS["right"], 0, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["right"], 1, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["right"], 0, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["right"], 1, 1),
    ]


def test_nudge_by_delta_scales_tilt_duration(monkeypatch) -> None:
    sleep_durations: list[float] = []
    monkeypatch.setattr(hikvision_ptz.time, "sleep", sleep_durations.append)

    sdk = FakeSDK()
    config = HikvisionPTZConfig(
        channel=1,
        nudge_speed=1,
        nudge_degrees_per_second=0.5,
        nudge_min_seconds=0.05,
        nudge_max_seconds=1.0,
        nudge_max_steps=4,
        tilt_nudge_scale=2.0,
    )

    hikvision_ptz._nudge_by_delta(sdk, 3, config, pan_delta_deg=0.0, tilt_delta_deg=0.02)

    assert sleep_durations == [0.1]
    assert sdk.calls == [
        (3, 1, hikvision_ptz.PTZ_COMMANDS["up"], 0, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["up"], 1, 1),
    ]


def test_nudge_zoom_sends_zoom_in_commands(monkeypatch) -> None:
    sleep_durations: list[float] = []
    monkeypatch.setattr(hikvision_ptz.time, "sleep", sleep_durations.append)

    sdk = FakeSDK()
    config = HikvisionPTZConfig(channel=1, zoom_nudge_speed=1, zoom_nudge_seconds=0.3, zoom_nudge_steps=2)

    hikvision_ptz._nudge_zoom(sdk, 3, config, "in")

    assert sleep_durations == [0.3, 0.3]
    assert sdk.calls == [
        (3, 1, hikvision_ptz.PTZ_COMMANDS["zoom-in"], 0, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["zoom-in"], 1, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["zoom-in"], 0, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["zoom-in"], 1, 1),
    ]


def test_nudge_ptz_retries_transient_stop_failure(monkeypatch) -> None:
    sleep_durations: list[float] = []
    monkeypatch.setattr(hikvision_ptz.time, "sleep", sleep_durations.append)

    class RetryStopSDK:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int, int, int, int]] = []

        def NET_DVR_PTZControlWithSpeed_Other(self, user_id, channel, command, stop, speed):  # noqa: ANN001, ANN202
            self.calls.append((user_id, channel, command, stop, speed))
            return stop == 0 or len([call for call in self.calls if call[3] == 1]) >= 2

        def NET_DVR_GetLastError(self) -> int:
            return 56

    sdk = RetryStopSDK()

    hikvision_ptz._nudge_ptz(sdk, 3, 1, "right", 0.2, 1)

    assert sleep_durations == [0.2, 0.1]
    assert sdk.calls == [
        (3, 1, hikvision_ptz.PTZ_COMMANDS["right"], 0, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["right"], 1, 1),
        (3, 1, hikvision_ptz.PTZ_COMMANDS["right"], 1, 1),
    ]


class FakeFOVSDK:
    def __init__(self) -> None:
        self.command = None
        self.channel = None

    def NET_DVR_GetSTDConfig(self, user_id, command, config):  # noqa: ANN001, ANN202
        self.command = command
        cfg = ctypes.cast(config, ctypes.POINTER(hikvision_ptz.NET_DVR_STD_CONFIG)).contents
        self.channel = ctypes.cast(cfg.lpCondBuffer, ctypes.POINTER(ctypes.c_int)).contents.value
        info = ctypes.cast(cfg.lpOutBuffer, ctypes.POINTER(hikvision_ptz.NET_DVR_GIS_INFO)).contents
        info.fHorizontalValue = 2.9
        info.fVerticalValue = 1.63
        info.fMinHorizontalValue = 2.9
        info.fMaxHorizontalValue = 54.93
        info.fMinVerticalValue = 1.63
        info.fMaxVerticalValue = 32.6
        info.struPtzPos.fZoomPos = 25.0
        return True


def test_read_gis_fov_returns_current_optical_field_of_view() -> None:
    sdk = FakeFOVSDK()

    fov = hikvision_ptz._read_gis_fov(sdk, user_id=3, channel=1)

    assert sdk.command == hikvision_ptz.NET_DVR_GET_GISINFO
    assert sdk.channel == 1
    assert fov.horizontal_deg == pytest.approx(2.9)
    assert fov.vertical_deg == pytest.approx(1.63)
    assert fov.min_horizontal_deg == pytest.approx(2.9)
    assert fov.max_horizontal_deg == pytest.approx(54.93)
    assert fov.min_vertical_deg == pytest.approx(1.63)
    assert fov.max_vertical_deg == pytest.approx(32.6)
    assert fov.zoom == pytest.approx(25.0)
    assert round(fov.max_zoom, 2) == 20.0


def test_controller_can_read_and_restore_absolute_ptz_position(monkeypatch) -> None:
    calls = []

    class FakeControllerSDK:
        def NET_DVR_Init(self) -> bool:
            return True

        def NET_DVR_GetLastError(self) -> int:
            return 0

        def NET_DVR_Logout(self, user_id):  # noqa: ANN001, ANN202
            calls.append(("logout", user_id))

        def NET_DVR_Cleanup(self):  # noqa: ANN202
            calls.append("cleanup")

    sdk = FakeControllerSDK()
    position = PTZPosition(pan_deg=5.0, tilt_deg=1.0, zoom_deg=12.0)

    monkeypatch.setattr(hikvision_ptz, "_load_sdk", lambda lib_dir: sdk)
    monkeypatch.setattr(hikvision_ptz, "_configure_sdk_paths", lambda sdk_arg, lib_dir: calls.append("configure"))
    monkeypatch.setattr(hikvision_ptz, "_bind_local_ip", lambda sdk_arg, local_ip: calls.append(("bind", local_ip)))
    monkeypatch.setattr(hikvision_ptz, "_login", lambda sdk_arg, config: 7)
    monkeypatch.setattr(hikvision_ptz, "_read_ptz", lambda sdk_arg, user_id, channel: position)

    def fake_set_ptz(sdk_arg, user_id, channel, pan_deg, tilt_deg, zoom_deg):  # noqa: ANN001, ANN202
        calls.append(("set", user_id, channel, pan_deg, tilt_deg, zoom_deg))

    monkeypatch.setattr(hikvision_ptz, "_set_ptz", fake_set_ptz)

    controller = hikvision_ptz.HikvisionPTZController(
        HikvisionPTZConfig(host="192.168.1.64", username="admin", password="secret", channel=1)
    )

    assert controller.read_position() == position
    controller.set_position(position)

    assert ("set", 7, 1, 5.0, 1.0, 12.0) in calls


def test_controller_reuses_sdk_session_until_closed(monkeypatch) -> None:
    calls = []

    class FakePersistentSDK:
        def NET_DVR_Init(self) -> bool:
            calls.append("init")
            return True

        def NET_DVR_GetLastError(self) -> int:
            return 0

        def NET_DVR_Logout(self, user_id):  # noqa: ANN001, ANN202
            calls.append(("logout", user_id))
            return True

        def NET_DVR_Cleanup(self):  # noqa: ANN202
            calls.append("cleanup")
            return True

    sdk = FakePersistentSDK()
    fov = HikvisionFieldOfView(7.48, 4.21, 2.9, 54.93, 1.63, 32.6, 9.1)

    monkeypatch.setattr(hikvision_ptz, "_load_sdk", lambda lib_dir: sdk)
    monkeypatch.setattr(hikvision_ptz, "_configure_sdk_paths", lambda sdk_arg, lib_dir: calls.append("configure"))
    monkeypatch.setattr(hikvision_ptz, "_bind_local_ip", lambda sdk_arg, local_ip: calls.append(("bind", local_ip)))
    monkeypatch.setattr(hikvision_ptz, "_login", lambda sdk_arg, config: 7)
    monkeypatch.setattr(hikvision_ptz, "_read_gis_fov", lambda sdk_arg, user_id, channel: fov)

    controller = hikvision_ptz.HikvisionPTZController(
        HikvisionPTZConfig(
            host="192.168.1.64",
            username="admin",
            password="secret",
            channel=1,
            local_ip="192.168.1.188",
        )
    )

    assert controller.read_field_of_view() == fov
    assert controller.read_field_of_view() == fov
    assert calls == ["configure", "init", ("bind", "192.168.1.188")]

    controller.close()

    assert calls == ["configure", "init", ("bind", "192.168.1.188"), ("logout", 7), "cleanup"]


def test_controller_read_zoom_limits_prefers_configured_max_zoom_ratio(monkeypatch) -> None:
    calls = []

    class FakePersistentSDK:
        def NET_DVR_Init(self) -> bool:
            calls.append("init")
            return True

        def NET_DVR_GetLastError(self) -> int:
            return 0

        def NET_DVR_Logout(self, user_id):  # noqa: ANN001, ANN202
            calls.append(("logout", user_id))
            return True

        def NET_DVR_Cleanup(self):  # noqa: ANN202
            calls.append("cleanup")
            return True

    sdk = FakePersistentSDK()
    fov = HikvisionFieldOfView(6.36, 3.58, 2.9, 54.93, 1.63, 32.6, 19.8)

    monkeypatch.setattr(hikvision_ptz, "_load_sdk", lambda lib_dir: sdk)
    monkeypatch.setattr(hikvision_ptz, "_configure_sdk_paths", lambda sdk_arg, lib_dir: calls.append("configure"))
    monkeypatch.setattr(hikvision_ptz, "_bind_local_ip", lambda sdk_arg, local_ip: calls.append(("bind", local_ip)))
    monkeypatch.setattr(hikvision_ptz, "_login", lambda sdk_arg, config: 7)
    monkeypatch.setattr(hikvision_ptz, "_read_gis_fov", lambda sdk_arg, user_id, channel: fov)

    controller = hikvision_ptz.HikvisionPTZController(
        HikvisionPTZConfig(host="192.168.1.64", password="secret", zoom_max_ratio=25.0)
    )

    limits = controller.read_zoom_limits()

    assert limits.current_zoom == 19.8
    assert limits.max_zoom == 25.0


def test_capture_image_uses_explicit_snapshot_channel(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        content = b"jpeg"

        def raise_for_status(self) -> None:
            captured["raised"] = True

    def fake_get(url, auth, timeout):  # noqa: ANN001, ANN202
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(hikvision_ptz.requests, "get", fake_get)
    monkeypatch.setattr(hikvision_ptz.cv2, "imdecode", lambda buffer, flags: object())

    controller = hikvision_ptz.HikvisionPTZController(
        HikvisionPTZConfig(host="10.42.0.120", password="secret", channel=1, snapshot_channel=102)
    )

    assert controller.capture_image() is not None
    assert captured == {
        "url": "http://10.42.0.120/ISAPI/Streaming/channels/102/picture",
        "timeout": 5.0,
        "raised": True,
    }


def test_capture_image_falls_back_to_secondary_stream_when_primary_is_unavailable(monkeypatch) -> None:
    urls = []

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code
            self.content = b"jpeg"

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                error = hikvision_ptz.requests.HTTPError(f"{self.status_code} error")
                error.response = self
                raise error

    def fake_get(url, auth, timeout):  # noqa: ANN001, ANN202
        urls.append(url)
        if url.endswith("/101/picture"):
            return FakeResponse(503)
        return FakeResponse(200)

    monkeypatch.setattr(hikvision_ptz.requests, "get", fake_get)
    monkeypatch.setattr(hikvision_ptz.cv2, "imdecode", lambda buffer, flags: object())

    controller = hikvision_ptz.HikvisionPTZController(
        HikvisionPTZConfig(host="10.42.0.120", password="secret", channel=1)
    )

    assert controller.capture_image() is not None
    assert urls == [
        "http://10.42.0.120/ISAPI/Streaming/channels/101/picture",
        "http://10.42.0.120/ISAPI/Streaming/channels/102/picture",
    ]
