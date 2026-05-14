from __future__ import annotations

import ctypes

import scripts.hk_sdk_probe as hk_sdk_probe


class FakeGISSDK:
    def __init__(self) -> None:
        self.command = None
        self.channel = None

    def NET_DVR_GetSTDConfig(self, user_id, command, config):  # noqa: ANN001, ANN202
        self.command = command
        cfg = ctypes.cast(config, ctypes.POINTER(hk_sdk_probe.NET_DVR_STD_CONFIG)).contents
        self.channel = ctypes.cast(cfg.lpCondBuffer, ctypes.POINTER(ctypes.c_int)).contents.value
        info = ctypes.cast(cfg.lpOutBuffer, ctypes.POINTER(hk_sdk_probe.NET_DVR_GIS_INFO)).contents
        info.fHorizontalValue = 12.34
        info.fVerticalValue = 5.67
        info.fMinHorizontalValue = 2.6
        info.fMaxHorizontalValue = 53.6
        info.fMinVerticalValue = 1.4
        info.fMaxVerticalValue = 30.6
        info.struPtzPos.fZoomPos = 25.0
        return True


def test_read_gis_fov_prints_current_field_of_view(capsys) -> None:
    sdk = FakeGISSDK()

    assert hk_sdk_probe.read_gis_fov(sdk, user_id=3, channel=1)

    captured = capsys.readouterr()
    assert sdk.command == hk_sdk_probe.NET_DVR_GET_GISINFO
    assert sdk.channel == 1
    assert "gis_fov_ok channel=1 hfov=12.34 vfov=5.67" in captured.out
    assert "hfov_range=2.60-53.60 vfov_range=1.40-30.60 zoom=25.0" in captured.out
