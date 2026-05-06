from __future__ import annotations

import threading
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests
from ultralytics import YOLO

from log_manager import GlobalLogManager
from recognition_http_server.constants import TASK_KIND_FIRE, TASK_KIND_METER, TASK_KIND_SAFEHAT
from recognition_http_server.dial_reading import load_model
from recognition_http_server.handlers.detection import run_detection_recognition
from recognition_http_server.handlers.meter import run_digital_meter_recognition, run_pointer_meter_recognition
from recognition_http_server.helpers import (
    align_data_types_to_images,
    build_detection_result_path,
    build_error_data_entry,
    build_callback_url,
    normalize_data_types,
    now_text,
    parse_extra_info,
    resolve_meter_subtype,
    resolve_task_kind,
    split_image_paths,
)
from recognition_http_server.schemas import TaskHandler, TaskState
from recognition_http_server.utils.image_io import prepare_image


class RecognitionService:
    """识别服务的核心编排层，负责模型加载、任务状态和任务调度。"""

    def __init__(self, config) -> None:
        self.config = config
        self.result_root = config.result_root
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.logger = GlobalLogManager.get_logger("recognition.service")
        self.input_root = self.result_root / "inputs"
        self.output_root = self.result_root / "outputs"
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.logger.info("loading models")
        self.meter_model, self.config.annotation_mode = load_model(config.model)
        self.fire_model = YOLO(config.fire_model)
        self.safehat_model = YOLO(config.safehat_model)
        self.logger.info(
            "models loaded: meter=%s fire=%s safehat=%s annotation_mode=%s",
            config.model,
            config.fire_model,
            config.safehat_model,
            self.config.annotation_mode,
        )
        self._tasks: dict[str, TaskState] = {}
        self._tasks_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self._task_handlers = self._build_task_handlers()

    def _build_task_handlers(self) -> dict[str, TaskHandler]:
        """集中注册任务处理器，便于后续新增或删减检测类型。"""
        return {
            TASK_KIND_METER: TaskHandler(TASK_KIND_METER, "meter", "表计读数", self._handle_meter_task),
            TASK_KIND_FIRE: TaskHandler(TASK_KIND_FIRE, "fire", "火源检测", self._handle_fire_task),
            TASK_KIND_SAFEHAT: TaskHandler(TASK_KIND_SAFEHAT, "safehat", "安全帽检测", self._handle_safehat_task),
        }

    def create_task(self, payload: dict[str, Any], callback_host: str) -> TaskState:
        """校验请求参数、注册任务状态，并启动异步后台处理。"""
        req_id = str(payload.get("req_id") or uuid4())
        image_path_value = payload.get("image_path")
        if not isinstance(image_path_value, str) or not image_path_value.strip():
            raise ValueError("image_path is required and must be a non-empty string.")

        image_paths = split_image_paths(image_path_value)
        data_types = normalize_data_types(payload.get("data_type"))
        align_data_types_to_images(image_paths, data_types)
        parse_extra_info(payload.get("extra_info", ""))

        with self._tasks_lock:
            task = TaskState(
                req_id=req_id,
                callback_host=callback_host,
                status="processing",
                created_at=now_text(),
                updated_at=now_text(),
                request_payload=payload,
            )
            self._tasks[req_id] = task

        self.logger.info("task created: req_id=%s callback_host=%s", req_id, callback_host)
        worker = threading.Thread(target=self._process_task, args=(req_id,), daemon=True)
        worker.start()
        return task

    def get_task(self, req_id: str) -> TaskState | None:
        """根据请求 ID 返回当前任务状态。"""
        with self._tasks_lock:
            return self._tasks.get(req_id)

    def _update_task(self, req_id: str, **updates: Any) -> None:
        """原子更新任务字段，并刷新最后更新时间。"""
        with self._tasks_lock:
            task = self._tasks[req_id]
            for key, value in updates.items():
                setattr(task, key, value)
            task.updated_at = now_text()

    def _process_task(self, req_id: str) -> None:
        """完整执行一个异步任务，并在结束后统一发送回调结果。"""
        task = self.get_task(req_id)
        if task is None:
            return

        payload = task.request_payload

        image_paths = split_image_paths(str(payload["image_path"]))
        data_types = normalize_data_types(payload.get("data_type"))
        image_data_types = align_data_types_to_images(image_paths, data_types)
        extra_info = parse_extra_info(payload.get("extra_info", ""))
        callback_url = build_callback_url(task.callback_host, self.config.callback_port)
        debug_center = bool(extra_info.get("debug_center", False))

        try:
            self.logger.info(
                "task processing started: req_id=%s image_count=%s callback_url=%s debug_center=%s",
                req_id,
                len(image_paths),
                callback_url,
                debug_center,
            )
            data_result = []
            for index, (image_path, data_type) in enumerate(zip(image_paths, image_data_types), start=1):
                data_result.append(self._process_single_image(req_id, index, image_path, data_type, extra_info, debug_center))

            callback_payload = {
                "req_id": req_id,
                "data_result": data_result,
                "code": 0,
                "resp_msg": "Task finished successfully.",
                "finish_time": now_text(),
            }
            self._update_task(req_id, status="finished", callback_payload=callback_payload, error=None)
            self.logger.info("task processing finished: req_id=%s result_count=%s", req_id, len(data_result))
            self._send_callback(req_id, callback_url, callback_payload)
        except Exception as exc:
            callback_payload = {
                "req_id": req_id,
                "data_result": [],
                "code": 1,
                "resp_msg": str(exc),
                "finish_time": now_text(),
            }
            self._update_task(req_id, status="finished", callback_payload=callback_payload, error=str(exc))
            self.logger.exception("task processing failed: req_id=%s", req_id)
            self._send_callback(req_id, callback_url, callback_payload)

    def _process_single_image(
        self,
        req_id: str,
        index: int,
        image_path: str,
        data_type: dict[str, str],
        extra_info: dict[str, Any],
        debug_center: bool,
    ) -> dict[str, Any]:
        """处理单张图片，包括输入准备、handler 分发和结果组装。"""
        local_image_path = prepare_image(self.logger, self.input_root, self.config.request_timeout, req_id, index, image_path)
        image_output_dir = self.output_root / req_id
        image_output_dir.mkdir(parents=True, exist_ok=True)
        task_kind = resolve_task_kind(data_type["recognize_type"])
        handler = self._task_handlers.get(task_kind)
        if handler is None:
            raise ValueError(f"No handler registered for task kind: {task_kind}")

        image_result_path = image_output_dir / f"{local_image_path.stem}_result_{handler.result_suffix}.jpg"
        detection_result_path = build_detection_result_path(image_path)
        self.logger.info(
            "image processing started: req_id=%s index=%s task_kind=%s subtype=%s source=%s",
            req_id,
            index,
            task_kind,
            data_type["recognize_subtype"],
            image_path,
        )
        try:
            recognize_items = handler.runner(local_image_path, data_type, image_result_path, extra_info, debug_center)
            has_success = any(item.get("recognize_value") for item in recognize_items)
            result_image_path = str(image_result_path) if has_success else image_path
            if has_success and detection_result_path is not None and image_result_path.exists():
                detection_result_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_result_path, detection_result_path)
                result_image_path = str(detection_result_path)
                self.logger.info(
                    "secondary result image saved: req_id=%s index=%s task_kind=%s secondary_path=%s primary_path=%s",
                    req_id,
                    index,
                    task_kind,
                    detection_result_path,
                    image_result_path,
                )
            self.logger.info(
                "image processing finished: req_id=%s index=%s task_kind=%s result_path=%s items=%s",
                req_id,
                index,
                task_kind,
                result_image_path,
                len(recognize_items),
            )
        except Exception as exc:
            error_text = str(exc)
            recognize_item = build_error_data_entry(data_type, error_text)
            recognize_item["recognize_image_index"] = "0"
            recognize_item["recognize_value"] = ""
            recognize_item["confidence"] = ""
            recognize_items = [recognize_item]
            result_image_path = image_path
            self.logger.exception(
                "image processing failed and downgraded to error result: req_id=%s index=%s task_kind=%s source=%s",
                req_id,
                index,
                task_kind,
                image_path,
            )
        return {
            "image_path": image_path,
            "image_path_result": result_image_path,
            "recognize_data": recognize_items,
        }

    def _handle_meter_task(
        self,
        local_image_path: Path,
        data_type: dict[str, str],
        visualize_path: Path,
        extra_info: dict[str, Any],
        debug_center: bool,
    ) -> list[dict[str, str]]:
        """将表计请求分发到指针表或数码表处理逻辑。"""
        meter_subtype = resolve_meter_subtype(data_type["recognize_subtype"])
        if meter_subtype.mode == "digital":
            return run_digital_meter_recognition(self, local_image_path, [data_type], visualize_path, extra_info)
        return run_pointer_meter_recognition(
            self,
            local_image_path,
            [data_type],
            visualize_path,
            debug_center,
            meter_subtype.scale or self.config.max_value,
        )

    def _handle_fire_task(
        self,
        local_image_path: Path,
        data_type: dict[str, str],
        visualize_path: Path,
        extra_info: dict[str, Any],
        debug_center: bool,
    ) -> list[dict[str, str]]:
        """复用通用检测链路处理火源检测任务。"""
        del extra_info, debug_center
        return run_detection_recognition(self, local_image_path, self.fire_model, [data_type], visualize_path, "火源检测")

    def _handle_safehat_task(
        self,
        local_image_path: Path,
        data_type: dict[str, str],
        visualize_path: Path,
        extra_info: dict[str, Any],
        debug_center: bool,
    ) -> list[dict[str, str]]:
        """复用通用检测链路处理安全帽检测任务。"""
        del extra_info, debug_center
        return run_detection_recognition(self, local_image_path, self.safehat_model, [data_type], visualize_path, "安全帽检测")

    def _send_callback(self, req_id: str, callback_url: str, callback_payload: dict[str, Any]) -> None:
        """发送回调结果，并记录回调状态，但不回滚识别任务本身。"""
        try:
            self.logger.info("sending callback: req_id=%s url=%s", req_id, callback_url)
            response = requests.post(callback_url, json=callback_payload, timeout=self.config.request_timeout)
            self._update_task(req_id, callback_status_code=response.status_code, callback_error=None)
            self.logger.info("callback sent: req_id=%s url=%s status_code=%s", req_id, callback_url, response.status_code)
        except Exception as exc:
            self._update_task(req_id, callback_error=str(exc))
            self.logger.exception("callback failed: req_id=%s url=%s", req_id, callback_url)
