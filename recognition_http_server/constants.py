from __future__ import annotations

from pathlib import Path

# 默认参数配置
DEFAULT_METER_MODEL_PATH = "runs/train/meter_data_9k_yolov8m_best/weights/best.pt"  # 指针表计读数权重路径
DEFAULT_FIRE_MODEL_PATH = "runs/train/best_fire.pt"  # 火源检测权重路径
DEFAULT_SAFEHAT_MODEL_PATH = "runs/train/best_person.pt"  # 安全帽检测权重路径
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
DEFAULT_LOG_DIR_NAME = "logs"

RECOGNIZE_TYPE_METER = "1"  # 表计读数任务 recognize_type 键值
RECOGNIZE_TYPE_FIRE = "6"  # 火源检测任务 recognize_type 键值
RECOGNIZE_TYPE_SAFEHAT = "7"  # 安全帽识别任务 recognize_type 键值

TASK_KIND_METER = "meter"
TASK_KIND_FIRE = "fire"
TASK_KIND_SAFEHAT = "safehat"

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
}
