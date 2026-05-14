# --*-- coding:utf-8 --*--
"""
海康球机 PTZ 控制 FastAPI 服务
接口：
  GET  /ptz/get          - 获取水平/垂直角度 + 变倍倍数
  POST /ptz/set          - 设置水平/垂直角度 + 变倍倍数
  POST /ptz/control      - 云台控制（方向、缩放等）
"""
import os
import platform
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from HCNetSDK import *
from ctypes import *

# ── 全局 SDK / 登录状态 ────────────────────────────────────────────
sdk = None
user_id = -1

# ── PTZ 控制命令映射 ───────────────────────────────────────────────
PTZ_COMMANDS = {
    "light_on":    LIGHT_PWRON,     # 接通灯光电源
    "zoom_in":     ZOOM_IN,         # 焦距变大(倍率变大)
    "zoom_out":    ZOOM_OUT,        # 焦距变小(倍率变小)
    "focus_near":  FOCUS_NEAR,      # 焦点前调
    "focus_far":   FOCUS_FAR,       # 焦点后调
    "iris_open":   IRIS_OPEN,       # 光圈扩大
    "iris_close":  IRIS_CLOSE,      # 光圈缩小
    "tilt_up":     TILT_UP,         # 云台上仰
    "tilt_down":   TILT_DOWN,       # 云台下俯
    "pan_left":    PAN_LEFT,        # 云台左转
    "pan_right":   PAN_RIGHT,       # 云台右转
    "up_left":     UP_LEFT,         # 上仰 + 左转
    "up_right":    UP_RIGHT,        # 上仰 + 右转
    "down_left":   DOWN_LEFT,       # 下俯 + 左转
    "down_right":  DOWN_RIGHT,      # 下俯 + 右转
    "pan_auto":    PAN_AUTO,        # 左右自动扫描
}

PRE_COMMANDS = {
    "add":    8,     # 添加预设位
    "del":    9,     # 删除预设位
    "goto":   39,    # 转到预设位
}

# ── 配置（可通过环境变量覆盖）─────────────────────────────────────
DEVICE_IP       = os.getenv("HK_DEVICE_IP", "10.42.0.120")
DEVICE_PORT     = int(os.getenv("HK_DEVICE_PORT", "8000"))
DEVICE_USER     = os.getenv("HK_DEVICE_USER", "admin")
DEVICE_PASSWORD = os.getenv("HK_DEVICE_PASSWORD", "oetsky@2023")
DEVICE_CHANNEL  = int(os.getenv("HK_DEVICE_CHANNEL", "1"))

WINDOWS_FLAG = platform.system() == "Windows"


# ── SDK 初始化 / 登录 / 注销 ──────────────────────────────────────
def _init_sdk():
    global sdk
    if WINDOWS_FLAG:
        os.chdir(os.path.join(os.path.dirname(__file__), "lib", "win"))
        sdk = ctypes.CDLL("./HCNetSDK.dll")
    else:
        lib_path = os.path.join(os.path.dirname(__file__), "lib", "linux")
        os.chdir(lib_path)
        sdk = cdll.LoadLibrary("./libhcnetsdk.so")

    # 设置组件库和 SSL 库路径
    str_path = os.getcwd().encode("gbk" if WINDOWS_FLAG else "utf-8")
    sdk_com_path = NET_DVR_LOCAL_SDK_PATH()
    sdk_com_path.sPath = str_path
    sdk.NET_DVR_SetSDKInitCfg(2, byref(sdk_com_path))
    sep = b"\\" if WINDOWS_FLAG else b"/"
    sdk.NET_DVR_SetSDKInitCfg(3, create_string_buffer(str_path + sep + (b"libcrypto-1_1-x64.dll" if WINDOWS_FLAG else b"libcrypto.so.1.1")))
    sdk.NET_DVR_SetSDKInitCfg(4, create_string_buffer(str_path + sep + (b"libssl-1_1-x64.dll" if WINDOWS_FLAG else b"libssl.so.1.1")))

    sdk.NET_DVR_Init()
    sdk.NET_DVR_SetLogToFile(3, b"./SdkLog_Python/", False)

    # 通用参数
    sdk_cfg = NET_DVR_LOCAL_GENERAL_CFG()
    sdk_cfg.byAlarmJsonPictureSeparate = 1
    sdk.NET_DVR_SetSDKLocalCfg(17, byref(sdk_cfg))


def _login() -> int:
    stru_login = NET_DVR_USER_LOGIN_INFO()
    stru_login.bUseAsynLogin = 0
    stru_login.sDeviceAddress = bytes(DEVICE_IP, "ascii")
    stru_login.wPort = DEVICE_PORT
    stru_login.sUserName = bytes(DEVICE_USER, "ascii")
    stru_login.sPassword = bytes(DEVICE_PASSWORD, "ascii")
    stru_login.byLoginMode = 0

    stru_dev_info = NET_DVR_DEVICEINFO_V40()
    uid = sdk.NET_DVR_Login_V40(byref(stru_login), byref(stru_dev_info))
    if uid < 0:
        raise RuntimeError(f"登录失败，错误码: {sdk.NET_DVR_GetLastError()}")
    return uid


def _logout():
    global user_id
    if sdk and user_id >= 0:
        sdk.NET_DVR_Logout(user_id)
        user_id = -1


# ── 请求/响应模型 ─────────────────────────────────────────────────
class PTZSetRequest(BaseModel):
    pan: int = Field(...,  description="水平角度（SDK 内部值，0-3600）")
    tilt: int = Field(..., description="垂直角度（SDK 内部值，0-3600）")
    zoom: int = Field(..., description="变倍倍数（SDK 内部值，0-1000）")
    channel: int = Field(DEVICE_CHANNEL, description="通道号")


class PTZControlRequest(BaseModel):
    command: str = Field(..., description=f"控制命令，可选: {', '.join(PTZ_COMMANDS.keys())}")
    stop: bool = Field(False, description="True=停止当前动作，False=执行动作")
    channel: int = Field(DEVICE_CHANNEL, description="通道号")

class PreControlRequest(BaseModel):
    command: str = Field(..., description=f"控制命令，可选: {', '.join(PRE_COMMANDS.keys())}")
    index: int = Field(..., ge=1, le=255, description="预设位索引（1-255）")
    channel: int = Field(DEVICE_CHANNEL, description="通道号")

# ── FastAPI 应用 ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化 SDK 并登录
    global user_id
    _init_sdk()
    user_id = _login()
    print(f"设备登录成功，UserID={user_id}")
    yield
    # 关闭时注销并释放 SDK
    _logout()
    sdk.NET_DVR_Cleanup()
    print("SDK 资源已释放")


app = FastAPI(title="海康球机 PTZ 控制服务", lifespan=lifespan)


# ── 接口 ───────────────────────────────────────────────────────────
@app.get("/ptz/get")
def get_ptz():
    """获取当前 PTZ 位置（水平角度、垂直角度、变倍倍数）"""
    stru_ptz = NET_DVR_PTZPOS()
    dw_returned = c_int(0)

    ok = sdk.NET_DVR_GetDVRConfig(
        user_id, NET_DVR_GET_PTZPOS, DEVICE_CHANNEL,
        byref(stru_ptz), sizeof(stru_ptz), byref(dw_returned),
    )
    if not ok:
        err = sdk.NET_DVR_GetLastError()
        raise HTTPException(status_code=500, detail=f"获取 PTZ 失败，错误码: {err}")

    pan  = int(hex(stru_ptz.wPanPos).replace("0x", ""), 16)
    tilt = int(hex(stru_ptz.wTiltPos).replace("0x", ""), 16)
    zoom = int(hex(stru_ptz.wZoomPos).replace("0x", ""), 16)

    return {"pan": pan, "tilt": tilt, "zoom": zoom}


@app.post("/ptz/set")
def set_ptz(req: PTZSetRequest):
    """设置 PTZ 绝对位置（水平角度、垂直角度、变倍倍数）"""
    stru_ptz = NET_DVR_PTZPOS()
    stru_ptz.wAction  = 1  # 同时设置 P、T、Z
    stru_ptz.wPanPos  = c_ushort(req.pan)
    stru_ptz.wTiltPos = c_ushort(req.tilt)
    stru_ptz.wZoomPos = c_ushort(req.zoom)

    ok = sdk.NET_DVR_SetDVRConfig(
        user_id, NET_DVR_SET_PTZPOS, req.channel,
        byref(stru_ptz), sizeof(stru_ptz),
    )
    if not ok:
        err = sdk.NET_DVR_GetLastError()
        raise HTTPException(status_code=500, detail=f"设置 PTZ 失败，错误码: {err}")

    return {"message": "设置成功", "pan": req.pan, "tilt": req.tilt, "zoom": req.zoom}


@app.post("/ptz/control")
def ptz_control(req: PTZControlRequest):
    """云台控制（方向、缩放、聚焦等）"""
    cmd_value = PTZ_COMMANDS.get(req.command)
    if cmd_value is None:
        raise HTTPException(
            status_code=400,
            detail=f"未知命令 '{req.command}'，可选: {', '.join(PTZ_COMMANDS.keys())}",
        )

    # stop=True 时 dwStop=1（停止），否则 dwStop=0（执行）
    dw_stop = 1 if req.stop else 0

    ok = sdk.NET_DVR_PTZControl_Other(user_id, req.channel, cmd_value, dw_stop)

    if not ok:
        err = sdk.NET_DVR_GetLastError()
        raise HTTPException(status_code=500, detail=f"云台控制失败，错误码: {err}")

    return {
        "message": "执行成功" if not req.stop else "停止成功",
        "command": req.command,
        "stop": req.stop,
    }


@app.post("/pre/control")
def pre_control(req: PreControlRequest):
    """预设位控制"""
    cmd_value = PRE_COMMANDS.get(req.command)
    if cmd_value is None:
        raise HTTPException(
            status_code=400,
            detail=f"未知命令 '{req.command}'，可选: {', '.join(PRE_COMMANDS.keys())}",
        )

    ok = sdk.NET_DVR_PTZPreset_Other(
        user_id, req.channel, cmd_value, req.index
    )
    if not ok:
        err = sdk.NET_DVR_GetLastError()
        raise HTTPException(status_code=500, detail=f"预设位控制失败，错误码: {err}")

    return {"message": "执行成功", "command": req.command, "index": req.index}


# ── 入口 ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ptz_server:app", host="0.0.0.0", port=9000, reload=False)
