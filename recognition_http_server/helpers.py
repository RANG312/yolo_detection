from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from recognition_http_server.constants import (
    DEFAULT_DATA_TYPE,
    DEFAULT_MAX_VALUE,
    DEFAULT_MIN_VALUE,
    DEFAULT_CALLBACK_PATH,
    OCR_SUBTYPES,
    RECOGNIZE_TYPE_ALIASES,
    RECOGNIZE_TYPE_METER,
    TASK_KIND_METER,
)
from recognition_http_server.schemas import MeterSubtypeResolution


def now_text() -> str:
    """返回当前本地时间的格式化字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def split_image_paths(image_path_value: str) -> list[str]:
    """将逗号分隔的图片路径字段拆成规范化路径列表。"""
    return [item.strip() for item in image_path_value.split(",") if item.strip()]


def parse_extra_info(extra_info: Any) -> dict[str, Any]:
    """解析 `extra_info`，兼容字典对象和 JSON 字符串两种输入。"""
    if extra_info in (None, "", {}):
        return {}
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            parsed = json.loads(extra_info)
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra_info is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("extra_info JSON must decode to an object.")
        return parsed
    raise ValueError("extra_info must be an object or a JSON string.")


def is_http_url(value: str) -> bool:
    """判断输入值是否为 HTTP 或 HTTPS URL。"""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def make_filename_from_url(url: str, fallback: str) -> str:
    """从 URL 路径中提取文件名，失败时返回兜底名称。"""
    path_name = Path(urlparse(url).path).name
    if path_name:
        return path_name
    return fallback


def build_callback_url(callback_host: str, callback_port: int) -> str:
    """根据回调主机和配置端口拼接完整回调地址。"""
    return f"http://{callback_host}:{callback_port}{DEFAULT_CALLBACK_PATH}"


def resolve_task_kind(recognize_type: str) -> str:
    """将请求中的 `recognize_type` 映射为内部任务类型。"""
    recognize_type_text = str(recognize_type).strip()
    task_kind = RECOGNIZE_TYPE_ALIASES.get(recognize_type_text)
    if task_kind is None:
        raise ValueError(f"Unsupported recognize_type: {recognize_type_text}")
    return task_kind


def resolve_meter_subtype(recognize_subtype: str) -> MeterSubtypeResolution:
    """
    解析表计子类型，判断应走数码表链路还是指针表量程链路。

    规则如下：
    - 命中保留的 OCR subtype：走数码表路径
    - 其他有效值：按浮点量程解析
    - 空字符串或 default：视为非法输入
    """
    subtype_text = str(recognize_subtype or "").strip()
    if subtype_text in OCR_SUBTYPES:
        return MeterSubtypeResolution(mode="digital")
    if subtype_text in {"", "default"}:
        raise ValueError("Meter recognize_subtype must be an explicit float scale or a reserved OCR subtype.")
    try:
        return MeterSubtypeResolution(mode="pointer", scale=float(subtype_text))
    except ValueError as exc:
        raise ValueError(
            f"Unsupported recognize_subtype for meter task: {subtype_text}. "
            f"Expected one of {sorted(OCR_SUBTYPES)} or a float scale."
        ) from exc


def normalize_data_types(data_types: Any) -> list[dict[str, str]]:
    """标准化请求中的 `data_type` 列表，并补齐默认字段。"""
    if not data_types:
        return [DEFAULT_DATA_TYPE.copy()]
    if not isinstance(data_types, list):
        raise ValueError("data_type must be a list.")
    normalized: list[dict[str, str]] = []
    for item in data_types:
        if not isinstance(item, dict):
            raise ValueError("each data_type item must be an object.")
        recognize_type_raw = item.get("recognize_type", RECOGNIZE_TYPE_METER)
        recognize_type = str(recognize_type_raw).strip() if recognize_type_raw is not None else RECOGNIZE_TYPE_METER
        if not recognize_type:
            recognize_type = RECOGNIZE_TYPE_METER
        task_kind = resolve_task_kind(recognize_type)

        recognize_subtype_raw = item.get("recognize_subtype", "")
        recognize_subtype = "" if recognize_subtype_raw in (None, "") else str(recognize_subtype_raw).strip()
        normalized.append({"recognize_type": recognize_type, "recognize_subtype": recognize_subtype})
    return normalized


def align_data_types_to_images(image_paths: list[str], data_types: list[dict[str, str]]) -> list[dict[str, str]]:
    """按服务的多图规则将 `data_type` 列表与图片路径列表对齐。"""
    if not image_paths:
        raise ValueError("image_path is required and must contain at least one image.")
    if not data_types:
        raise ValueError("data_type must contain at least one item.")
    if len(data_types) == 1:
        return [data_types[0].copy() for _ in image_paths]
    if len(image_paths) != len(data_types):
        raise ValueError("When multiple data_type items are provided, the number of data_type items must match image_path.")
    return [item.copy() for item in data_types]


def build_meter_predict_args(
    config: Namespace,
    save_path: Path,
    min_value: float = DEFAULT_MIN_VALUE,
    max_value: float = DEFAULT_MAX_VALUE,
    debug_center: bool = False,
) -> SimpleNamespace:
    """构造旧版 `dial_reading.py` 推理链路所需的参数对象。"""
    return SimpleNamespace(
        imgsz=config.imgsz,
        conf=config.conf,
        device=config.device,
        min_value=min_value,
        max_value=max_value,
        debug_center=debug_center,
        show=False,
        save=save_path,
        annotation_mode=config.annotation_mode,
        test_loop=False,
        image=None,
        input_dir=None,
        batch_save_root=config.result_root,
        model=config.model,
    )


def build_success_data_entry(data_type: dict[str, str], reading: float, desc: str) -> dict[str, str]:
    """构造一条成功的 `recognize_data` 结果项。"""
    return {
        "recognize_type": data_type["recognize_type"],
        "recognize_subtype": data_type["recognize_subtype"],
        "recognize_value": f"{reading:.6f}",
        "confidence": "100",
        "recognize_desc": desc,
    }


def build_error_data_entry(data_type: dict[str, str], error_text: str) -> dict[str, str]:
    """构造一条失败的 `recognize_data` 结果项。"""
    return {
        "recognize_type": data_type["recognize_type"],
        "recognize_subtype": data_type["recognize_subtype"],
        "recognize_value": "",
        "confidence": "0",
        "recognize_desc": f"识别失败: {error_text}",
    }


def make_error_payload(req_id: str | None, message: str, status: str = "finished") -> dict[str, Any]:
    """构造 HTTP 和 ROS 层共用的标准错误响应体。"""
    return {
        "req_id": req_id,
        "code": 1,
        "resp_msg": message,
        "data": {"task_status": status},
    }
