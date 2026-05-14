from __future__ import annotations

from pathlib import Path

# 默认参数配置
DEFAULT_METER_MODEL_PATH = "runs/weights/1_dial_reading/best.pt"  # 指针表计读数权重路径
DEFAULT_FIRE_MODEL_PATH = "runs/weights/6_fire_and_smoke/best_fire.pt"  # 火源检测权重路径
DEFAULT_SAFEHAT_MODEL_PATH = "runs/weights/7_safe_hat/best_person.pt"  # 安全帽检测权重路径
DEFAULT_FIRE_PROTECTION_FACILITIES_MODEL_PATH = (
    "runs/weights/8_fire_protection_facilities/fire-fighting-facilitie_best.pt"
)  # 消防设施检测权重路径
DEFAULT_PERSON_FALL_DOWN_MODEL_PATH = "runs/weights/9_person_fall_down/fall_best.pt"  # 摔倒检测权重路径
DEFAULT_FIRE_EXTINGUISHER_MODEL_PATH = (
    "runs/weights/10_fire_extinguisher/extinguisher_best.pt"
)  # 灭火器检测权重路径
DEFAULT_PERSON_AND_CARS_MODEL_PATH = "runs/weights/11_person_and_cars/car_best.pt"  # 人车检测权重路径
DEFAULT_MIN_VALUE = 0.0  # 指针读数任务表盘起始刻度
DEFAULT_MAX_VALUE = 1.0  # 指针读数任务表盘末端刻度，默认设为 1（归一化）
DEFAULT_IMGSZ = 640  # YOLO 模型处理图片分辨率
DEFAULT_CONF = 0.25  # 默认置信度
DEFAULT_DEVICE = "0"  # 默认推理设备，GPU 0
DEFAULT_HOST = "0.0.0.0"  # HTTP 服务地址
DEFAULT_PORT = 3208  # HTTP 服务端口
DEFAULT_CALLBACK_PORT = 8088  # CALLBACK 端口
DEFAULT_CALLBACK_PATH = "/api/v1/recognition/callback"  # CALLBACK 地址
DEFAULT_RESULT_ROOT = Path("results/http_service")  # 结果保存路径
DEFAULT_REQUEST_TIMEOUT = 15  # 设置允许 timeout 时长
DEFAULT_LOG_DIR_NAME = "logs"  # HTTP 服务日志目录名
DEFAULT_PTZ_ALIGN_ENABLED = True  # 是否默认启用海康云台自动对齐
DEFAULT_PTZ_HORIZONTAL_FOV_DEG = 60.0  # 云台水平视场角兜底值，启用 PTZ 时优先从 SDK 读取当前值
DEFAULT_PTZ_VERTICAL_FOV_DEG = 40.0  # 云台垂直视场角兜底值，启用 PTZ 时优先从 SDK 读取当前值
DEFAULT_PTZ_ALIGN_THRESHOLD_DEG = 0.02  # pan/tilt 偏移小于该角度时不执行云台修正，窄视场下需要较小阈值
DEFAULT_PTZ_ALIGN_MAX_DELTA_DEG = 5.0  # 单轮 pan/tilt 修正允许的最大角度
DEFAULT_PTZ_ALIGN_MAX_PASSES = 3  # 最多执行几轮“检测表盘 -> 调整云台 -> 重新抓图”
DEFAULT_PTZ_HOST = "192.168.1.64"  # 海康设备 IP 地址或主机名
DEFAULT_PTZ_PORT = 8000  # 海康 SDK 登录端口
DEFAULT_PTZ_USERNAME = "admin"  # 海康 SDK 默认登录用户名
DEFAULT_PTZ_PASSWORD = "oetsky@2023"  # 海康 SDK 默认登录密码；不要在源码中保存真实密码，使用 HIK_PASSWORD 或 --ptz-password
DEFAULT_PTZ_CHANNEL = 1  # 海康通道号，抓图 URL 会使用 channel01
DEFAULT_PTZ_TILT_MIN_DEG = 0.0  # 云台 tilt 最小角度保留参数
DEFAULT_PTZ_TILT_MAX_DEG = 90.0  # 云台 tilt 最大角度保留参数
DEFAULT_PTZ_SETTLE_SECONDS = 0.5  # pan/tilt 调整后等待画面稳定的秒数
DEFAULT_PTZ_NUDGE_SPEED = 1  # 海康连续云台控制速度等级
DEFAULT_PTZ_NUDGE_DEGREES_PER_SECOND = 0.5  # 将目标角度换算为连续控制时长的估算速度，窄视场下取较低值
DEFAULT_PTZ_NUDGE_MIN_SECONDS = 0.05  # 单次 pan/tilt 连续控制最短时长
DEFAULT_PTZ_NUDGE_MAX_SECONDS = 1.0  # 单次 pan/tilt 连续控制最长时长
DEFAULT_PTZ_NUDGE_MAX_STEPS = 4  # 单轮 pan/tilt 修正最多拆分的连续控制次数
DEFAULT_PTZ_TILT_NUDGE_SCALE = 2.0  # tilt 控制时长倍率，用于补偿垂直方向灵敏度
DEFAULT_PTZ_ZOOM_ENABLED = True  # pan/tilt 对齐后是否默认启用光学变焦调整
DEFAULT_PTZ_ZOOM_TARGET_HEIGHT_RATIO = 0.8  # 目标表盘高度占整图高度的比例
DEFAULT_PTZ_ZOOM_RATIO_TOLERANCE = 0.05  # 表盘高度比例允许误差，误差内不调整 zoom
DEFAULT_PTZ_ZOOM_MAX_PASSES = 3  # 最多执行几轮“检测表盘 -> 调整 zoom -> 重新抓图”
DEFAULT_PTZ_ZOOM_NUDGE_SPEED = 1  # 海康 zoom 连续控制速度等级
DEFAULT_PTZ_ZOOM_NUDGE_SECONDS = 0.6  # 单次 zoom in/out 连续控制时长
DEFAULT_PTZ_ZOOM_NUDGE_STEPS = 1  # 单轮 zoom 调整连续控制次数
DEFAULT_PTZ_ZOOM_FOCUS_TIMEOUT = 2.0  # zoom 后等待自动对焦稳定的秒数

RECOGNIZE_TYPE_METER = "1"  # 表计读数任务 recognize_type 键值
RECOGNIZE_TYPE_FIRE = "6"  # 火源检测任务 recognize_type 键值
RECOGNIZE_TYPE_SAFEHAT = "7"  # 安全帽识别任务 recognize_type 键值
RECOGNIZE_TYPE_FIRE_PROTECTION_FACILITIES = "8"  # 消防设施检测任务 recognize_type 键值
RECOGNIZE_TYPE_PERSON_FALL_DOWN = "9"  # 摔倒检测任务 recognize_type 键值
RECOGNIZE_TYPE_FIRE_EXTINGUISHER = "10"  # 灭火器检测任务 recognize_type 键值
RECOGNIZE_TYPE_PERSON_AND_CARS = "11"  # 人车检测任务 recognize_type 键值

TASK_KIND_METER = "meter"
TASK_KIND_FIRE = "fire"
TASK_KIND_SAFEHAT = "safehat"
TASK_KIND_FIRE_PROTECTION_FACILITIES = "fire_protection_facilities"
TASK_KIND_PERSON_FALL_DOWN = "person_fall_down"
TASK_KIND_FIRE_EXTINGUISHER = "fire_extinguisher"
TASK_KIND_PERSON_AND_CARS = "person_and_cars"

DEFAULT_DATA_TYPE = {"recognize_type": RECOGNIZE_TYPE_METER, "recognize_subtype": "default"}
OCR_SUBTYPES = frozenset({"digital"})

RECOGNIZE_TYPE_ALIASES = {
    RECOGNIZE_TYPE_METER: TASK_KIND_METER,
    "dict_meter_type": TASK_KIND_METER,
    RECOGNIZE_TYPE_FIRE: TASK_KIND_FIRE,
    "fire": TASK_KIND_FIRE,
    RECOGNIZE_TYPE_SAFEHAT: TASK_KIND_SAFEHAT,
    "safehat": TASK_KIND_SAFEHAT,
    "person": TASK_KIND_SAFEHAT,
    RECOGNIZE_TYPE_FIRE_PROTECTION_FACILITIES: TASK_KIND_FIRE_PROTECTION_FACILITIES,
    "fire_protection_facilities": TASK_KIND_FIRE_PROTECTION_FACILITIES,
    RECOGNIZE_TYPE_PERSON_FALL_DOWN: TASK_KIND_PERSON_FALL_DOWN,
    "person_fall_down": TASK_KIND_PERSON_FALL_DOWN,
    RECOGNIZE_TYPE_FIRE_EXTINGUISHER: TASK_KIND_FIRE_EXTINGUISHER,
    "fire_extinguisher": TASK_KIND_FIRE_EXTINGUISHER,
    RECOGNIZE_TYPE_PERSON_AND_CARS: TASK_KIND_PERSON_AND_CARS,
    "person_and_cars": TASK_KIND_PERSON_AND_CARS,
}
