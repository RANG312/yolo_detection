from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class TaskHandler:
    # 单类任务的路由配置，便于后续统一注册和扩展
    kind: str
    result_suffix: str
    description: str
    runner: Callable[[Path, dict[str, str], Path, dict[str, Any], bool], list[dict[str, str]]]


@dataclass(frozen=True)
class MeterSubtypeResolution:
    # 表计任务子类型解析结果
    mode: str
    scale: float | None = None


@dataclass
class TaskState:
    # 记录任务生命周期内的状态与回调信息
    req_id: str
    callback_host: str
    status: str = "pending"
    created_at: str = ""
    updated_at: str = ""
    request_payload: dict[str, Any] = field(default_factory=dict)
    callback_payload: dict[str, Any] | None = None
    callback_status_code: int | None = None
    callback_error: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class LoadedModels:
    meter_model: Any
    fire_model: Any
    safehat_model: Any
    annotation_mode: str
    config: Namespace
