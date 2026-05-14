#!/usr/bin/env python3
"""Probe Hikvision HCNetSDK login and PTZ position."""

from __future__ import annotations

import argparse
import ctypes
import os
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = ROOT / "HK_SDK_x86_Linux" / "HCNetSDKV6.1.11.5_build20251204_linux64_ZH"
LIB_DIR = SDK_ROOT / "库文件"

NET_DVR_GET_PTZPOS = 293
NET_DVR_GET_GISINFO = 3711
NET_DVR_CHECK_USER_STATUS = 20005
NET_DVR_GET_SLAVECAMERA_CALIB_CAPABILITIES = 3603
NET_DVR_GET_SLAVECAMERA_CALIB_V51 = 8102
SDK_LONG = ctypes.c_int32
MAX_CALIB_NUM_EX = 20

PTZ_COMMANDS = {
    "up": 21,
    "down": 22,
    "left": 23,
    "right": 24,
    "up-left": 25,
    "up-right": 26,
    "down-left": 27,
    "down-right": 28,
    "zoom-in": 11,
    "zoom-out": 12,
    "focus-near": 13,
    "focus-far": 14,
    "down-zoom-in": 58,
    "down-zoom-out": 59,
    "left-zoom-in": 60,
    "left-zoom-out": 61,
    "right-zoom-in": 62,
    "right-zoom-out": 63,
    "up-left-zoom-in": 64,
    "up-left-zoom-out": 65,
    "up-right-zoom-in": 66,
    "up-right-zoom-out": 67,
    "down-left-zoom-in": 68,
    "down-left-zoom-out": 69,
    "down-right-zoom-in": 70,
    "down-right-zoom-out": 71,
    "up-zoom-in": 72,
    "up-zoom-out": 73,
}


class NET_DVR_DEVICEINFO_V30(ctypes.Structure):
    _fields_ = [
        ("sSerialNumber", ctypes.c_byte * 48),
        ("byAlarmInPortNum", ctypes.c_byte),
        ("byAlarmOutPortNum", ctypes.c_byte),
        ("byDiskNum", ctypes.c_byte),
        ("byDVRType", ctypes.c_byte),
        ("byChanNum", ctypes.c_byte),
        ("byStartChan", ctypes.c_byte),
        ("byAudioChanNum", ctypes.c_byte),
        ("byIPChanNum", ctypes.c_byte),
        ("byZeroChanNum", ctypes.c_byte),
        ("byMainProto", ctypes.c_byte),
        ("bySubProto", ctypes.c_byte),
        ("bySupport", ctypes.c_byte),
        ("bySupport1", ctypes.c_byte),
        ("bySupport2", ctypes.c_byte),
        ("wDevType", ctypes.c_uint16),
        ("bySupport3", ctypes.c_byte),
        ("byMultiStreamProto", ctypes.c_byte),
        ("byStartDChan", ctypes.c_byte),
        ("byStartDTalkChan", ctypes.c_byte),
        ("byHighDChanNum", ctypes.c_byte),
        ("bySupport4", ctypes.c_byte),
        ("byLanguageType", ctypes.c_byte),
        ("byVoiceInChanNum", ctypes.c_byte),
        ("byStartVoiceInChanNo", ctypes.c_byte),
        ("bySupport5", ctypes.c_byte),
        ("bySupport6", ctypes.c_byte),
        ("byMirrorChanNum", ctypes.c_byte),
        ("wStartMirrorChanNo", ctypes.c_uint16),
        ("bySupport7", ctypes.c_byte),
        ("byRes2", ctypes.c_byte),
    ]


class NET_DVR_LOCAL_SDK_PATH(ctypes.Structure):
    _fields_ = [
        ("sPath", ctypes.c_char * 256),
        ("byRes", ctypes.c_byte * 128),
    ]


class NET_DVR_PTZPOS(ctypes.Structure):
    _fields_ = [
        ("wAction", ctypes.c_uint16),
        ("wPanPos", ctypes.c_uint16),
        ("wTiltPos", ctypes.c_uint16),
        ("wZoomPos", ctypes.c_uint16),
    ]


class NET_DVR_STD_CONFIG(ctypes.Structure):
    _fields_ = [
        ("lpCondBuffer", ctypes.c_void_p),
        ("dwCondSize", ctypes.c_uint32),
        ("lpInBuffer", ctypes.c_void_p),
        ("dwInSize", ctypes.c_uint32),
        ("lpOutBuffer", ctypes.c_void_p),
        ("dwOutSize", ctypes.c_uint32),
        ("lpStatusBuffer", ctypes.c_void_p),
        ("dwStatusSize", ctypes.c_uint32),
        ("lpXmlBuffer", ctypes.c_void_p),
        ("dwXmlSize", ctypes.c_uint32),
        ("byDataType", ctypes.c_byte),
        ("byRes", ctypes.c_byte * 23),
    ]


class NET_DVR_STD_ABILITY(ctypes.Structure):
    _fields_ = [
        ("lpCondBuffer", ctypes.c_void_p),
        ("dwCondSize", ctypes.c_uint32),
        ("lpOutBuffer", ctypes.c_void_p),
        ("dwOutSize", ctypes.c_uint32),
        ("lpStatusBuffer", ctypes.c_void_p),
        ("dwStatusSize", ctypes.c_uint32),
        ("dwRetSize", ctypes.c_uint32),
        ("byRes", ctypes.c_byte * 32),
    ]


class NET_DVR_SLAVECAMERA_COND(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("dwChannel", ctypes.c_uint32),
        ("byID", ctypes.c_byte),
        ("byRes1", ctypes.c_byte * 3),
        ("dwSceneID", ctypes.c_uint32),
        ("byRes", ctypes.c_byte * 56),
    ]


class NET_VCA_POINT(ctypes.Structure):
    _fields_ = [
        ("fX", ctypes.c_float),
        ("fY", ctypes.c_float),
    ]


class NET_PTZ_INFO(ctypes.Structure):
    _fields_ = [
        ("fPan", ctypes.c_float),
        ("fTilt", ctypes.c_float),
        ("fZoom", ctypes.c_float),
        ("dwFocus", ctypes.c_uint32),
        ("byRes", ctypes.c_byte * 4),
    ]


class NET_DVR_LLI_PARAM(ctypes.Structure):
    _fields_ = [
        ("fSec", ctypes.c_float),
        ("byDegree", ctypes.c_byte),
        ("byMinute", ctypes.c_byte),
        ("byRes", ctypes.c_byte * 6),
    ]


class NET_DVR_PTZPOS_PARAM(ctypes.Structure):
    _fields_ = [
        ("fPanPos", ctypes.c_float),
        ("fTiltPos", ctypes.c_float),
        ("fZoomPos", ctypes.c_float),
        ("byRes", ctypes.c_byte * 16),
    ]


class NET_DVR_SENSOR_PARAM(ctypes.Structure):
    _fields_ = [
        ("bySensorType", ctypes.c_byte),
        ("byRes", ctypes.c_byte * 31),
        ("fHorWidth", ctypes.c_float),
        ("fVerWidth", ctypes.c_float),
        ("fFold", ctypes.c_float),
    ]


class NET_DVR_GIS_INFO(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("fAzimuth", ctypes.c_float),
        ("fHorizontalValue", ctypes.c_float),
        ("fVerticalValue", ctypes.c_float),
        ("fVisibleRadius", ctypes.c_float),
        ("fMaxViewRadius", ctypes.c_float),
        ("byLatitudeType", ctypes.c_byte),
        ("byLongitudeType", ctypes.c_byte),
        ("byPTZPosExEnable", ctypes.c_byte),
        ("byRes1", ctypes.c_byte),
        ("struLatitude", NET_DVR_LLI_PARAM),
        ("struLongitude", NET_DVR_LLI_PARAM),
        ("struPtzPos", NET_DVR_PTZPOS_PARAM),
        ("struSensorParam", NET_DVR_SENSOR_PARAM),
        ("struPtzPosEx", NET_PTZ_INFO),
        ("fMinHorizontalValue", ctypes.c_float),
        ("fMaxHorizontalValue", ctypes.c_float),
        ("fMinVerticalValue", ctypes.c_float),
        ("fMaxVerticalValue", ctypes.c_float),
        ("byRes", ctypes.c_byte * 220),
    ]


class NET_DVR_CALIB_PARAM(ctypes.Structure):
    _fields_ = [
        ("struPtzInfo", NET_PTZ_INFO),
        ("struCalibCoordinates", NET_VCA_POINT),
        ("iHorValue", ctypes.c_int),
        ("iVerValue", ctypes.c_int),
        ("byRes", ctypes.c_byte * 8),
    ]


class NET_DVR_SLAVECAMERA_CALIB_V51(ctypes.Structure):
    _fields_ = [
        ("dwSize", ctypes.c_uint32),
        ("byCalibMode", ctypes.c_byte),
        ("byRes", ctypes.c_byte * 3),
        ("struCalibParam", NET_DVR_CALIB_PARAM * MAX_CALIB_NUM_EX),
        ("byRes1", ctypes.c_byte * 512),
    ]


def bcd_angle(value: int) -> float:
    """Decode Hikvision PTZ BCD-like position value to degrees."""
    return int(f"{value:x}", 10) * 0.1


def load_sdk() -> ctypes.CDLL:
    sdk = ctypes.CDLL(str(LIB_DIR / "libhcnetsdk.so"))
    sdk.NET_DVR_SetSDKInitCfg.restype = ctypes.c_bool
    sdk.NET_DVR_Init.restype = ctypes.c_bool
    sdk.NET_DVR_GetLastError.restype = ctypes.c_uint32
    sdk.NET_DVR_GetLocalIP.restype = ctypes.c_bool
    sdk.NET_DVR_SetValidIP.restype = ctypes.c_bool
    sdk.NET_DVR_Login_V30.argtypes = [
        ctypes.c_char_p,
        ctypes.c_uint16,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.POINTER(NET_DVR_DEVICEINFO_V30),
    ]
    sdk.NET_DVR_Login_V30.restype = SDK_LONG
    sdk.NET_DVR_GetDVRConfig.argtypes = [
        SDK_LONG,
        ctypes.c_uint32,
        SDK_LONG,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    sdk.NET_DVR_GetDVRConfig.restype = ctypes.c_bool
    sdk.NET_DVR_GetSTDConfig.argtypes = [SDK_LONG, ctypes.c_uint32, ctypes.POINTER(NET_DVR_STD_CONFIG)]
    sdk.NET_DVR_GetSTDConfig.restype = ctypes.c_bool
    sdk.NET_DVR_GetSTDAbility.argtypes = [SDK_LONG, ctypes.c_uint32, ctypes.POINTER(NET_DVR_STD_ABILITY)]
    sdk.NET_DVR_GetSTDAbility.restype = ctypes.c_bool
    sdk.NET_DVR_RemoteControl.argtypes = [SDK_LONG, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
    sdk.NET_DVR_RemoteControl.restype = ctypes.c_bool
    sdk.NET_DVR_PTZControlWithSpeed_Other.argtypes = [
        SDK_LONG,
        SDK_LONG,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
    ]
    sdk.NET_DVR_PTZControlWithSpeed_Other.restype = ctypes.c_bool
    sdk.NET_DVR_Logout.argtypes = [SDK_LONG]
    sdk.NET_DVR_Logout.restype = ctypes.c_bool
    sdk.NET_DVR_Cleanup.restype = ctypes.c_bool
    return sdk


def configure_sdk_paths(sdk: ctypes.CDLL) -> None:
    lib_path = str(LIB_DIR).encode("utf-8")
    sdk_path = NET_DVR_LOCAL_SDK_PATH()
    sdk_path.sPath = lib_path
    sdk.NET_DVR_SetSDKInitCfg(2, ctypes.byref(sdk_path))
    sdk.NET_DVR_SetSDKInitCfg(3, ctypes.create_string_buffer(lib_path + b"/libcrypto.so.3"))
    sdk.NET_DVR_SetSDKInitCfg(4, ctypes.create_string_buffer(lib_path + b"/libssl.so.3"))


def bind_local_ip(sdk: ctypes.CDLL, local_ip: str) -> bool:
    ip_buffer = ctypes.create_string_buffer(16 * 64)
    count = ctypes.c_int(0)
    enable_bind = ctypes.c_bool(False)
    if not sdk.NET_DVR_GetLocalIP(ctypes.byref(ip_buffer), ctypes.byref(count), enable_bind):
        print(f"local_ip_list_failed error={sdk.NET_DVR_GetLastError()}")
        return False

    for index in range(count.value):
        chunk = ip_buffer.raw[index * 64 : (index + 1) * 64].split(b"\x00", 1)[0]
        ips = chunk.decode("utf-8", errors="ignore").split()
        print(f"local_ip_candidate index={index} ips={','.join(ips)}")
        if local_ip in ips:
            if sdk.NET_DVR_SetValidIP(index, True):
                print(f"local_ip_bound index={index} ip={local_ip}")
                return True
            print(f"local_ip_bind_failed index={index} ip={local_ip} error={sdk.NET_DVR_GetLastError()}")
            return False

    print(f"local_ip_not_found ip={local_ip}")
    return False


def read_ptz(sdk: ctypes.CDLL, user_id: int, channel: int) -> bool:
    returned = ctypes.c_uint32(0)
    ptz = NET_DVR_PTZPOS()
    ok = sdk.NET_DVR_GetDVRConfig(
        user_id,
        NET_DVR_GET_PTZPOS,
        channel,
        ctypes.byref(ptz),
        ctypes.sizeof(ptz),
        ctypes.byref(returned),
    )
    if not ok:
        print(f"ptz_read_failed channel={channel} error={sdk.NET_DVR_GetLastError()}")
        return False

    print(
        f"ptz_ok channel={channel} pan={bcd_angle(ptz.wPanPos):.1f} "
        f"tilt={bcd_angle(ptz.wTiltPos):.1f} zoom={bcd_angle(ptz.wZoomPos):.1f} "
        f"raw_pan=0x{ptz.wPanPos:04x} raw_tilt=0x{ptz.wTiltPos:04x} raw_zoom=0x{ptz.wZoomPos:04x}"
    )
    return True


def read_gis_fov(sdk: ctypes.CDLL, user_id: int, channel: int) -> bool:
    info = NET_DVR_GIS_INFO()
    info.dwSize = ctypes.sizeof(info)
    cond = ctypes.c_int(channel)
    status = ctypes.create_string_buffer(4096)
    cfg = NET_DVR_STD_CONFIG()
    cfg.lpCondBuffer = ctypes.cast(ctypes.byref(cond), ctypes.c_void_p)
    cfg.dwCondSize = ctypes.sizeof(cond)
    cfg.lpOutBuffer = ctypes.cast(ctypes.byref(info), ctypes.c_void_p)
    cfg.dwOutSize = ctypes.sizeof(info)
    cfg.lpStatusBuffer = ctypes.cast(status, ctypes.c_void_p)
    cfg.dwStatusSize = ctypes.sizeof(status)
    cfg.byDataType = 0

    ok = sdk.NET_DVR_GetSTDConfig(user_id, NET_DVR_GET_GISINFO, ctypes.byref(cfg))
    if not ok:
        status_text = status.value.decode("utf-8", errors="replace")
        print(f"gis_fov_read_failed channel={channel} error={sdk.NET_DVR_GetLastError()} status={status_text}")
        return False

    print(
        f"gis_fov_ok channel={channel} hfov={info.fHorizontalValue:.2f} vfov={info.fVerticalValue:.2f} "
        f"hfov_range={info.fMinHorizontalValue:.2f}-{info.fMaxHorizontalValue:.2f} "
        f"vfov_range={info.fMinVerticalValue:.2f}-{info.fMaxVerticalValue:.2f} "
        f"zoom={info.struPtzPos.fZoomPos:.1f}"
    )
    return True


def nudge_ptz(sdk: ctypes.CDLL, user_id: int, channel: int, direction: str, duration: float, speed: int) -> bool:
    command = PTZ_COMMANDS[direction]
    print(f"ptz_nudge_start channel={channel} direction={direction} duration={duration:.2f}s speed={speed}")
    started = sdk.NET_DVR_PTZControlWithSpeed_Other(user_id, channel, command, 0, speed)
    if not started:
        print(f"ptz_nudge_start_failed error={sdk.NET_DVR_GetLastError()}")
        return False

    try:
        time.sleep(duration)
    finally:
        stopped = sdk.NET_DVR_PTZControlWithSpeed_Other(user_id, channel, command, 1, speed)

    if not stopped:
        print(f"ptz_nudge_stop_failed error={sdk.NET_DVR_GetLastError()}")
        return False
    print("ptz_nudge_stop_ok")
    return True


def read_slave_calib(sdk: ctypes.CDLL, user_id: int, channel: int) -> bool:
    calib = NET_DVR_SLAVECAMERA_CALIB_V51()
    calib.dwSize = ctypes.sizeof(calib)
    cond = NET_DVR_SLAVECAMERA_COND()
    cond.dwSize = ctypes.sizeof(cond)
    cond.dwChannel = channel
    cond.byID = 1
    status = ctypes.create_string_buffer(4096)
    cfg = NET_DVR_STD_CONFIG()
    cfg.lpCondBuffer = ctypes.cast(ctypes.byref(cond), ctypes.c_void_p)
    cfg.dwCondSize = ctypes.sizeof(cond)
    cfg.lpOutBuffer = ctypes.cast(ctypes.byref(calib), ctypes.c_void_p)
    cfg.dwOutSize = ctypes.sizeof(calib)
    cfg.lpStatusBuffer = ctypes.cast(status, ctypes.c_void_p)
    cfg.dwStatusSize = ctypes.sizeof(status)
    cfg.byDataType = 0

    ok = sdk.NET_DVR_GetSTDConfig(user_id, NET_DVR_GET_SLAVECAMERA_CALIB_V51, ctypes.byref(cfg))
    if not ok:
        status_text = status.value.decode("utf-8", errors="replace")
        print(f"slave_calib_read_failed error={sdk.NET_DVR_GetLastError()} status={status_text}")
        return False

    print(f"slave_calib_ok mode={calib.byCalibMode} size={calib.dwSize}")
    for idx, param in enumerate(calib.struCalibParam):
        ptz = param.struPtzInfo
        point = param.struCalibCoordinates
        if (
            abs(ptz.fPan) > 0.0001
            or abs(ptz.fTilt) > 0.0001
            or abs(ptz.fZoom) > 0.0001
            or ptz.dwFocus
            or abs(point.fX) > 0.0001
            or abs(point.fY) > 0.0001
            or param.iHorValue
            or param.iVerValue
        ):
            print(
                f"calib[{idx}] pan={ptz.fPan:.3f} tilt={ptz.fTilt:.3f} zoom={ptz.fZoom:.3f} "
                f"focus={ptz.dwFocus} x={point.fX:.6f} y={point.fY:.6f} "
                f"hor={param.iHorValue} ver={param.iVerValue}"
            )
    return True


def read_slave_calib_ability(sdk: ctypes.CDLL, user_id: int, channel: int) -> bool:
    cond = NET_DVR_SLAVECAMERA_COND()
    cond.dwSize = ctypes.sizeof(cond)
    cond.dwChannel = channel
    cond.byID = 1
    out = ctypes.create_string_buffer(128 * 1024)
    status = ctypes.create_string_buffer(4096)
    ability = NET_DVR_STD_ABILITY()
    ability.lpCondBuffer = ctypes.cast(ctypes.byref(cond), ctypes.c_void_p)
    ability.dwCondSize = ctypes.sizeof(cond)
    ability.lpOutBuffer = ctypes.cast(out, ctypes.c_void_p)
    ability.dwOutSize = ctypes.sizeof(out)
    ability.lpStatusBuffer = ctypes.cast(status, ctypes.c_void_p)
    ability.dwStatusSize = ctypes.sizeof(status)

    ok = sdk.NET_DVR_GetSTDAbility(user_id, NET_DVR_GET_SLAVECAMERA_CALIB_CAPABILITIES, ctypes.byref(ability))
    if not ok:
        first_error = sdk.NET_DVR_GetLastError()
        out = ctypes.create_string_buffer(128 * 1024)
        status = ctypes.create_string_buffer(4096)
        ability = NET_DVR_STD_ABILITY()
        ability.lpOutBuffer = ctypes.cast(out, ctypes.c_void_p)
        ability.dwOutSize = ctypes.sizeof(out)
        ability.lpStatusBuffer = ctypes.cast(status, ctypes.c_void_p)
        ability.dwStatusSize = ctypes.sizeof(status)
        ok = sdk.NET_DVR_GetSTDAbility(user_id, NET_DVR_GET_SLAVECAMERA_CALIB_CAPABILITIES, ctypes.byref(ability))
        if not ok:
            status_text = status.value.decode("utf-8", errors="replace")
            print(
                f"slave_calib_ability_failed error={sdk.NET_DVR_GetLastError()} "
                f"first_error={first_error} status={status_text}"
            )
            return False

    text = out.value.decode("utf-8", errors="replace")
    print(f"slave_calib_ability_ok ret_size={ability.dwRetSize}")
    print(text)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--username", default=os.environ.get("HIK_USERNAME", "admin"))
    parser.add_argument("--password", default=os.environ.get("HIK_PASSWORD"))
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--local-ip", help="Bind HCNetSDK to a local interface IP before login.")
    parser.add_argument("--nudge-direction", choices=sorted(PTZ_COMMANDS))
    parser.add_argument("--nudge-duration", type=float, default=0.2)
    parser.add_argument("--nudge-speed", type=int, default=1)
    parser.add_argument("--read-slave-calib-ability", action="store_true")
    parser.add_argument("--read-slave-calib", action="store_true")
    parser.add_argument("--read-gis-fov", action="store_true", help="Read current optical FOV from NET_DVR_GET_GISINFO.")
    args = parser.parse_args()

    if not args.password:
        parser.error("pass --password or set HIK_PASSWORD")

    sdk = load_sdk()
    configure_sdk_paths(sdk)
    if not sdk.NET_DVR_Init():
        print(f"init_failed error={sdk.NET_DVR_GetLastError()}")
        return 2

    if args.local_ip and not bind_local_ip(sdk, args.local_ip):
        sdk.NET_DVR_Cleanup()
        return 2

    user_id = -1
    try:
        device_info = NET_DVR_DEVICEINFO_V30()
        user_id = sdk.NET_DVR_Login_V30(
            args.host.encode("ascii"),
            args.port,
            args.username.encode("ascii"),
            args.password.encode("ascii"),
            ctypes.byref(device_info),
        )
        if user_id < 0:
            print(f"login_failed host={args.host} error={sdk.NET_DVR_GetLastError()}")
            return 3

        serial = bytes(device_info.sSerialNumber).split(b"\x00", 1)[0].decode("ascii", errors="replace")
        online = sdk.NET_DVR_RemoteControl(user_id, NET_DVR_CHECK_USER_STATUS, None, 0)
        print(
            f"login_ok host={args.host} user_id={user_id} serial={serial} "
            f"dev_type={device_info.wDevType} start_chan={device_info.byStartChan} "
            f"start_dchan={device_info.byStartDChan} online={int(online)}"
        )

        if not read_ptz(sdk, user_id, args.channel):
            return 4

        if args.read_gis_fov and not read_gis_fov(sdk, user_id, args.channel):
            return 8

        if args.read_slave_calib_ability and not read_slave_calib_ability(sdk, user_id, args.channel):
            return 7

        if args.read_slave_calib and not read_slave_calib(sdk, user_id, args.channel):
            return 6

        if args.nudge_direction:
            if not 0.05 <= args.nudge_duration <= 2.0:
                print("nudge_duration_out_of_range")
                return 5
            if not 1 <= args.nudge_speed <= 7:
                print("nudge_speed_out_of_range")
                return 5
            if not nudge_ptz(sdk, user_id, args.channel, args.nudge_direction, args.nudge_duration, args.nudge_speed):
                return 5
            read_ptz(sdk, user_id, args.channel)

        return 0
    finally:
        if user_id >= 0:
            sdk.NET_DVR_Logout(user_id)
        sdk.NET_DVR_Cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
