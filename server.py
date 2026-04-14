from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import cv2
import requests
from ultralytics import YOLO

from dial_reading import load_model, predict_image_instances, save_canvas
from log_manager import GlobalLogManager

# 默认参数配置
DEFAULT_METER_MODEL_PATH = "runs/train/meter_data_9k_yolov8m_best/weights/best.pt"    # 指针表计读数权重路径
DEFAULT_FIRE_MODEL_PATH = "runs/train/best_fire.pt"     # 火源检测权重路径 
DEFAULT_SAFEHAT_MODEL_PATH = "runs/train/best_person.pt"    # 安全帽检测权重路径
DEFAULT_MIN_VALUE = 0.0     # 指针读数任务表盘起始刻度
DEFAULT_MAX_VALUE = 1.0     # 指针读数任务表盘末端刻度， 默认设为1（归一化）
DEFAULT_IMGSZ = 640     # YOLO模型处理图片分辨率
DEFAULT_CONF = 0.25     # 默认置信度
DEFAULT_DEVICE = "0"    # 默认推理设备，GPU 0
DEFAULT_HOST = "0.0.0.0"    # HTTP服务地址
DEFAULT_PORT = 3208     # HTTP服务端口
DEFAULT_CALLBACK_PORT = 8088   # CALLBACK 端口
DEFAULT_CALLBACK_PATH = "/api/v1/recognition/callback"  # CALLBACK地址
DEFAULT_RESULT_ROOT = Path("results/http_service")   # 结果保存路径
DEFAULT_REQUEST_TIMEOUT = 15    # 设置允许timeout时长
DEFAULT_LOG_DIR_NAME = "logs"

RECOGNIZE_TYPE_METER = "1"      # 表计读数任务"recognize_type"键值
RECOGNIZE_TYPE_FIRE = "6"       # 火源检测任务"recognize_type"键值
RECOGNIZE_TYPE_SAFEHAT = "7"    # 安全帽识别任务"recognize_type"键值
TASK_KIND_METER = "meter"
TASK_KIND_FIRE = "fire"
TASK_KIND_SAFEHAT = "safehat"
DEFAULT_DATA_TYPE = {"recognize_type": RECOGNIZE_TYPE_METER, "recognize_subtype": "default"}    # 拼接请求体内参

def parse_args() -> argparse.Namespace:
    # 解析服务启动参数
    parser = argparse.ArgumentParser(description="Recognition HTTP service.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server bind port.")
    parser.add_argument("--model", default=DEFAULT_METER_MODEL_PATH, help="Meter model path (.pt or .onnx).")
    parser.add_argument("--fire-model", default=DEFAULT_FIRE_MODEL_PATH, help="Fire detection model path (.pt or .onnx).")
    parser.add_argument(
        "--safehat-model", default=DEFAULT_SAFEHAT_MODEL_PATH, help="Safehat/person detection model path (.pt or .onnx)."
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Inference confidence threshold.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, e.g. 'cpu' or '0'.")
    parser.add_argument("--min-value", type=float, default=DEFAULT_MIN_VALUE, help="Default meter minimum value.")
    parser.add_argument("--max-value", type=float, default=DEFAULT_MAX_VALUE, help="Default meter maximum value.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT, help="Result root directory.")
    parser.add_argument("--callback-port", type=int, default=DEFAULT_CALLBACK_PORT, help="Fixed callback port.")
    parser.add_argument(
        "--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP download/callback timeout in seconds."
    )
    return parser.parse_args()

def now_text() -> str:
    # 返回当前时间字符串
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def split_image_paths(image_path_value: str) -> list[str]:
    # 将逗号分隔的图片路径拆成列表
    return [item.strip() for item in image_path_value.split(",") if item.strip()]

def parse_extra_info(extra_info: Any) -> dict[str, Any]:
    # 解析 extra_info，兼容 dict 和 JSON 字符串
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
    # 判断路径是否为 HTTP/HTTPS URL
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

def make_filename_from_url(url: str, fallback: str) -> str:
    # 从 URL 提取文件名，提取失败时使用兜底名称
    path_name = Path(urlparse(url).path).name
    if path_name:
        return path_name
    return fallback

def build_callback_url(callback_host: str, callback_port: int) -> str:
    # 根据客户端地址和固定端口拼接回调地址
    return f"http://{callback_host}:{callback_port}{DEFAULT_CALLBACK_PATH}"

def resolve_task_kind(recognize_type: str) -> str:
    # 将 recognize_type 映射为内部任务类型
    recognize_type_text = str(recognize_type).strip()
    if recognize_type_text in {RECOGNIZE_TYPE_METER, "dict_meter_type"}:
        return TASK_KIND_METER
    if recognize_type_text in {RECOGNIZE_TYPE_FIRE, "fire"}:
        return TASK_KIND_FIRE
    if recognize_type_text in {RECOGNIZE_TYPE_SAFEHAT, "safehat", "person"}:
        return TASK_KIND_SAFEHAT
    raise ValueError(f"Unsupported recognize_type: {recognize_type_text}")

def normalize_data_types(data_types: Any) -> list[dict[str, str]]:
    # 标准化请求中的 data_type 列表，并补齐默认值
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
        if task_kind == TASK_KIND_METER and not recognize_subtype:
            recognize_subtype = "default"
        normalized.append({"recognize_type": recognize_type, "recognize_subtype": recognize_subtype})
    return normalized

def align_data_types_to_images(image_paths: list[str], data_types: list[dict[str, str]]) -> list[dict[str, str]]:
    # 将 data_type 与图片列表对齐，支持单个检测项复用到多张图
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
    config: argparse.Namespace,
    save_path: Path,
    min_value: float,
    max_value: float,
    debug_center: bool,
) -> SimpleNamespace:
    """
    构造传给 dial_reading.py 的预测参数对象。

    当前表计读数仍然复用 dial_reading.py 中的推理与几何后处理逻辑，
    但 server.py 自身并不直接解析 argparse 命令行参数，因此这里将服
    务启动参数重新封装成 SimpleNamespace，保持与原预测代码的调用
    方式一致。

    参数中的 save、imgsz、conf、device 等字段会直接影响表计推理；
    annotation_mode 由 load_model 自动识别，决定读数后处理走哪套标注
    规则。
    """
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
    # 构造单条成功识别结果
    return {
        "recognize_type": data_type["recognize_type"],
        "recognize_subtype": data_type["recognize_subtype"],
        "recognize_value": f"{reading:.6f}",
        "confidence": "100",
        "recognize_desc": desc,
    }


def build_error_data_entry(data_type: dict[str, str], error_text: str) -> dict[str, str]:
    # 构造单条失败识别结果
    return {
        "recognize_type": data_type["recognize_type"],
        "recognize_subtype": data_type["recognize_subtype"],
        "recognize_value": "",
        "confidence": "0",
        "recognize_desc": f"识别失败: {error_text}",
    }


@dataclass
class TaskState:
    # 记录任务生命周期内的状态与回调信息
    req_id: str
    callback_host: str
    status: str = "pending"
    created_at: str = field(default_factory=now_text)
    updated_at: str = field(default_factory=now_text)
    request_payload: dict[str, Any] = field(default_factory=dict)
    callback_payload: dict[str, Any] | None = None
    callback_status_code: int | None = None
    callback_error: str | None = None
    error: str | None = None


class RecognitionService:
    def __init__(self, config: argparse.Namespace) -> None:
        """
        初始化识别服务。

        这里会完成三类模型的加载：
        1. 表计读数模型
        2. 火源检测模型
        3. 安全帽检测模型

        同时初始化输入输出目录、任务状态表，以及一个全局预测锁。
        预测锁的作用是避免多线程同时抢占模型推理资源，降低显存竞争和
        推理过程中的不确定性，尤其适合 Jetson 这类边缘设备部署场景。
        """
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

    def create_task(self, payload: dict[str, Any], callback_host: str) -> TaskState:
        """
        创建异步识别任务并立即返回。

        该方法只负责参数校验、任务注册和后台线程启动，不在当前 HTTP
        请求线程中执行真正的模型推理。这样客户端提交任务后可以立即得
        到 accepted 响应，后续结果通过回调接口异步返回。

        这里会重点校验：
        - image_path 是否存在且非空
        - data_type 是否合法
        - 多张图片与 data_type 的映射关系是否符合“一图一检测项”规则
        - extra_info 是否可被解析
        """
        req_id = str(payload.get("req_id") or uuid4())
        image_path_value = payload.get("image_path")
        if not isinstance(image_path_value, str) or not image_path_value.strip():
            raise ValueError("image_path is required and must be a non-empty string.")

        image_paths = split_image_paths(image_path_value)
        data_types = normalize_data_types(payload.get("data_type"))
        align_data_types_to_images(image_paths, data_types)
        parse_extra_info(payload.get("extra_info", ""))

        with self._tasks_lock:
            # Temporary for local/Postman testing: allow reusing the same req_id.
            # if req_id in self._tasks:
            #     raise ValueError(f"req_id already exists: {req_id}")
            task = TaskState(req_id=req_id, callback_host=callback_host, status="processing", request_payload=payload)
            self._tasks[req_id] = task

        self.logger.info(
            "task created: req_id=%s callback_host=%s image_count=%s",
            req_id,
            callback_host,
            len(image_paths),
        )
        worker = threading.Thread(target=self._process_task, args=(req_id,), daemon=True)
        worker.start()
        return task

    def get_task(self, req_id: str) -> TaskState | None:
        # 按 req_id 查询任务状态
        with self._tasks_lock:
            return self._tasks.get(req_id)

    def _update_task(self, req_id: str, **updates: Any) -> None:
        # 原子更新任务状态，并刷新更新时间
        with self._tasks_lock:
            task = self._tasks[req_id]
            for key, value in updates.items():
                setattr(task, key, value)
            task.updated_at = now_text()
            update_keys = ",".join(sorted(updates))
            self.logger.debug("task updated: req_id=%s fields=%s", req_id, update_keys)

    def _process_task(self, req_id: str) -> None:
        """
        后台执行整个识别任务。

        一个任务可能包含多张图片。当前业务规则为：
        - 一张图只做一种检测
        - 如果 data_type 只有一项，则复用到所有图片
        - 如果 data_type 有多项，则按顺序与图片一一对应

        该方法会逐张图调用 _process_single_image，汇总成 data_result，
        最后统一构造回调 payload 并发送给客户端。
        """
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
        """
        处理单张图片的识别流程。

        单张图进入这里后，会先完成图片准备：
        - 若是 HTTP URL，则下载到本地临时目录
        - 若是本地路径，则直接校验并读取

        之后根据 data_type 中的 recognize_type 选择具体模型：
        - 1 -> 表计读数
        - 6 -> 火源检测
        - 7 -> 安全帽检测

        最终返回该图片对应的一条 data_result 记录，其中包含原图路径、
        结果图路径以及 recognize_data 列表。
        """
        local_image_path = self._prepare_image(req_id, index, image_path)
        image_output_dir = self.output_root / req_id
        image_output_dir.mkdir(parents=True, exist_ok=True)
        recognize_items: list[dict[str, str]] = []
        image_result_path: Path | None = None
        task_kind = resolve_task_kind(data_type["recognize_type"])
        self.logger.info(
            "image processing started: req_id=%s index=%s task_kind=%s source=%s",
            req_id,
            index,
            task_kind,
            image_path,
        )

        if task_kind == TASK_KIND_METER:
            meter_visualize_path = image_output_dir / f"{local_image_path.stem}_result_meter.jpg"
            recognize_items.extend(
                self._run_meter_recognition(
                    local_image_path,
                    [data_type],
                    meter_visualize_path,
                    debug_center,
                )
            )
            image_result_path = meter_visualize_path

        elif task_kind == TASK_KIND_FIRE:
            fire_visualize_path = image_output_dir / f"{local_image_path.stem}_result_fire.jpg"
            recognize_items.extend(
                self._run_detection_recognition(
                    local_image_path,
                    self.fire_model,
                    [data_type],
                    fire_visualize_path,
                    "火源检测",
                )
            )
            image_result_path = fire_visualize_path

        elif task_kind == TASK_KIND_SAFEHAT:
            safehat_visualize_path = image_output_dir / f"{local_image_path.stem}_result_safehat.jpg"
            recognize_items.extend(
                self._run_detection_recognition(
                    local_image_path,
                    self.safehat_model,
                    [data_type],
                    safehat_visualize_path,
                    "安全帽检测",
                )
            )
            image_result_path = safehat_visualize_path

        self.logger.info(
            "image processing finished: req_id=%s index=%s task_kind=%s result_path=%s items=%s",
            req_id,
            index,
            task_kind,
            image_result_path or local_image_path,
            len(recognize_items),
        )
        return {
            "image_path": image_path,
            "image_path_result": str(image_result_path or local_image_path),
            "recognize_data": recognize_items,
        }

    def _run_meter_recognition(
        self,
        local_image_path: Path,
        data_types: list[dict[str, str]],
        visualize_path: Path,
        debug_center: bool,
    ) -> list[dict[str, str]]:
        """
        执行表计读数任务。

        表计任务不是简单的目标检测，而是先通过 YOLO 定位表盘关键元素，
        再在 dial_reading.py 中完成几何后处理，计算得到归一化读数。

        返回结果中：
        - recognize_value 为归一化读数
        - confidence 当前固定为 100
        - recognize_desc 中会附带 recognize_image_index 和 arc_mode
        """
        predict_args = build_meter_predict_args(self.config, visualize_path, DEFAULT_MIN_VALUE, DEFAULT_MAX_VALUE, debug_center)
        self.logger.info("meter recognition running: image=%s output=%s", local_image_path, visualize_path)
        with self._predict_lock:
            canvas, prediction_instances, _, _ = predict_image_instances(
                local_image_path, self.meter_model, predict_args, self.config.annotation_mode
            )
        save_canvas(canvas, visualize_path)

        recognize_items: list[dict[str, str]] = []
        for meter_instance in prediction_instances:
            recognize_image_index = str(meter_instance["recognize_image_index"])
            if meter_instance.get("error") is None:
                normalized_reading = float(meter_instance["reading"])
                arc_mode = str(meter_instance["arc_mode"])
                for data_type in data_types:
                    desc = f"识别成功，表计#{recognize_image_index}归一化读数为{normalized_reading:.6f}，arc_mode={arc_mode}"
                    recognize_item = build_success_data_entry(data_type, normalized_reading, desc)
                    recognize_item["recognize_image_index"] = recognize_image_index
                    recognize_items.append(recognize_item)
            else:
                error_text = str(meter_instance["error"])
                for data_type in data_types:
                    recognize_item = build_error_data_entry(data_type, error_text)
                    recognize_item["recognize_image_index"] = recognize_image_index
                    recognize_items.append(recognize_item)
        self.logger.info(
            "meter recognition finished: image=%s meters=%s output=%s",
            local_image_path,
            len(recognize_items),
            visualize_path,
        )
        return recognize_items

    def _run_detection_recognition(
        self,
        local_image_path: Path,
        model: YOLO,
        data_types: list[dict[str, str]],
        visualize_path: Path,
        task_desc: str,
    ) -> list[dict[str, str]]:
        """
        执行通用目标检测任务。

        该方法同时服务于火源检测和安全帽检测两类场景。处理过程为：
        1. 使用对应 YOLO 模型对整图推理
        2. 将检测框绘制到结果图
        3. 把每个检测框组织为一条 recognize_data 记录

        对检测类任务来说：
        - recognize_value 为类别名
        - confidence 为模型置信度百分比
        - recognize_image_index 表示第几个检测框
        """
        image = cv2.imread(str(local_image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {local_image_path}")

        self.logger.info("detection recognition running: task=%s image=%s output=%s", task_desc, local_image_path, visualize_path)
        with self._predict_lock:
            results = model.predict(source=image, imgsz=self.config.imgsz, conf=self.config.conf, device=self.config.device, verbose=False)
        result = results[0]

        plotted = result.plot()
        save_canvas(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB), visualize_path)

        recognize_items: list[dict[str, str]] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            for data_type in data_types:
                recognize_item = build_error_data_entry(data_type, f"{task_desc}未检测到目标")
                recognize_item["recognize_image_index"] = "0"
                recognize_items.append(recognize_item)
            self.logger.warning("detection recognition found no targets: task=%s image=%s", task_desc, local_image_path)
            return recognize_items

        names = result.names if isinstance(result.names, dict) else {i: name for i, name in enumerate(result.names)}
        for detect_index, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
            label = str(names.get(cls_id, cls_id))
            for data_type in data_types:
                recognize_items.append(
                    {
                        "recognize_image_index": str(detect_index),
                        "recognize_type": data_type["recognize_type"],
                        "recognize_subtype": data_type["recognize_subtype"],
                        "recognize_value": label,
                        "confidence": str(int(round(confidence * 100))),
                        "recognize_desc": f"识别成功，{task_desc}检测到{label}，置信度={confidence:.4f}",
                    }
                )

        self.logger.info(
            "detection recognition finished: task=%s image=%s detections=%s output=%s",
            task_desc,
            local_image_path,
            len(boxes),
            visualize_path,
        )
        return recognize_items

    def _prepare_image(self, req_id: str, index: int, image_path: str) -> Path:
        """
        准备待识别图片并返回本地路径。

        支持两种输入：
        - HTTP/HTTPS 图片地址：先下载到 results/http_service/inputs/<req_id>/
        - 本地图片路径：直接解析为绝对路径并校验存在性

        这样下游推理逻辑始终只需要处理本地文件路径。
        """
        task_input_dir = self.input_root / req_id
        task_input_dir.mkdir(parents=True, exist_ok=True)

        if is_http_url(image_path):
            filename = make_filename_from_url(image_path, f"image_{index}.jpg")
            target_path = task_input_dir / filename
            self.logger.info("downloading image: req_id=%s index=%s url=%s target=%s", req_id, index, image_path, target_path)
            response = requests.get(image_path, timeout=self.config.request_timeout)
            response.raise_for_status()
            target_path.write_bytes(response.content)
            self.logger.info("image downloaded: req_id=%s index=%s target=%s bytes=%s", req_id, index, target_path, len(response.content))
            return target_path

        local_path = Path(image_path).expanduser()
        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path
        if not local_path.exists():
            raise FileNotFoundError(f"Image does not exist: {local_path}")
        self.logger.info("using local image: req_id=%s index=%s path=%s", req_id, index, local_path)
        return local_path

    def _send_callback(self, req_id: str, callback_url: str, callback_payload: dict[str, Any]) -> None:
        """
        向客户端发送回调结果。

        回调失败不会影响任务本身的 finished 状态，但会把异常记录到任务
        状态中，便于后续通过查询接口定位问题。
        """
        try:
            self.logger.info("sending callback: req_id=%s url=%s", req_id, callback_url)
            response = requests.post(callback_url, json=callback_payload, timeout=self.config.request_timeout)
            self._update_task(req_id, callback_status_code=response.status_code, callback_error=None)
            self.logger.info("callback sent: req_id=%s url=%s status_code=%s", req_id, callback_url, response.status_code)
        except Exception as exc:
            self._update_task(req_id, callback_error=str(exc))
            self.logger.exception("callback failed: req_id=%s url=%s", req_id, callback_url)

def make_error_payload(req_id: str | None, message: str, status: str = "finished") -> dict[str, Any]:
    # 构造统一错误响应体
    return {
        "req_id": req_id,
        "code": 1,
        "resp_msg": message,
        "data": {"task_status": status},
    }


class RecognitionHandler(BaseHTTPRequestHandler):
    server: "RecognitionAPIServer"

    @property
    def logger(self) -> logging.Logger:
        return GlobalLogManager.get_logger("recognition.http")

    def do_GET(self) -> None:  # noqa: N802
        """
        处理 GET 请求。

        支持两个接口：
        - /health：健康检查
        - /api/v1/recognition/tasks/{req_id}：查询任务状态
        """
        if self.path == "/health":
            self.logger.debug("health check requested: client=%s", self.client_address[0])
            self._send_json(
                HTTPStatus.OK,
                {"code": 0, "resp_msg": "ok", "data": {"status": "healthy", "time": now_text()}},
            )
            return

        if self.path.startswith("/api/v1/recognition/tasks/"):
            req_id = self.path.rsplit("/", 1)[-1]
            self.logger.info("task query requested: req_id=%s client=%s", req_id, self.client_address[0])
            task = self.server.service.get_task(req_id)
            if task is None:
                self._send_json(HTTPStatus.NOT_FOUND, make_error_payload(req_id, "Task not found."))
                return
            payload = {
                "req_id": task.req_id,
                "code": 0,
                "resp_msg": "success",
                "data": {
                    "task_status": task.status,
                    "created_at": task.created_at,
                    "updated_at": task.updated_at,
                    "callback_payload": task.callback_payload,
                    "callback_status_code": task.callback_status_code,
                    "callback_error": task.callback_error,
                    "error": task.error,
                },
            }
            self._send_json(HTTPStatus.OK, payload)
            return

        self._send_json(HTTPStatus.NOT_FOUND, make_error_payload(None, "Path not found."))

    def do_POST(self) -> None:  # noqa: N802
        """
        处理 POST 请求。

        当前只开放 /api/v1/recognition/tasks，用于提交异步识别任务。
        提交成功后不阻塞等待推理完成，而是立即返回 processing 状态，
        真正结果由后台线程完成后通过 callback 接口回传。
        """
        if self.path != "/api/v1/recognition/tasks":
            self._send_json(HTTPStatus.NOT_FOUND, make_error_payload(None, "Path not found."))
            return

        payload: dict[str, Any] | None = None
        try:
            payload = self._read_json_body()
            callback_host = self._get_callback_host()
            self.logger.info(
                "task submit requested: client=%s callback_host=%s req_id=%s",
                self.client_address[0],
                callback_host,
                payload.get("req_id"),
            )
            task = self.server.service.create_task(payload, callback_host)
        except ValueError as exc:
            req_id = None
            if isinstance(exc.args[0], str) and "req_id" in str(exc):
                req_id = str(payload.get("req_id")) if isinstance(payload, dict) else None
            self.logger.warning("bad request: path=%s req_id=%s error=%s", self.path, req_id, exc)
            self._send_json(HTTPStatus.BAD_REQUEST, make_error_payload(req_id, str(exc)))
            return
        except Exception as exc:
            req_id = payload.get("req_id") if isinstance(payload, dict) else None
            self.logger.exception("request handling failed: path=%s req_id=%s", self.path, req_id)
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, make_error_payload(req_id, str(exc)))
            return

        response = {
            "req_id": task.req_id,
            "code": 0,
            "resp_msg": "Task accepted successfully.",
            "data": {"task_status": task.status},
        }
        self.logger.info("task accepted: req_id=%s", task.req_id)
        self._send_json(HTTPStatus.OK, response)

    def _get_callback_host(self) -> str:
        # 从代理头或客户端连接信息中推断回调主机地址
        forwarded_for = self.headers.get("X-Forwarded-For", "").strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = self.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
        return self.client_address[0]

    def log_message(self, format: str, *args: Any) -> None:
        # 自定义 HTTP 访问日志格式
        message = format % args
        self.logger.info("access: client=%s message=%s", self.address_string(), message)

    def _read_json_body(self) -> dict[str, Any]:
        # 读取并校验 JSON 请求体
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is empty.")

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            raise ValueError("Content-Type must be application/json.")

        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Request body is not valid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("Request body JSON must be an object.")
        return payload

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        # 发送 JSON 响应
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RecognitionAPIServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], request_handler_class: type[RecognitionHandler], service: RecognitionService):
        # 注入业务服务实例，供 Handler 访问
        super().__init__(server_address, request_handler_class)
        self.service = service


def main() -> None:
    """
    服务入口函数。

    负责解析启动参数、初始化业务服务、创建 HTTP Server 并开始监听。
    启动时会打印当前加载的模型路径和表计标注模式，便于部署排查。
    """
    args = parse_args()
    args.result_root = args.result_root.resolve()
    log_dir = args.result_root / DEFAULT_LOG_DIR_NAME
    GlobalLogManager.configure(log_dir)
    logger = GlobalLogManager.get_logger("recognition.main")
    service = RecognitionService(args)
    server = RecognitionAPIServer((args.host, args.port), RecognitionHandler, service)
    logger.info("log file: %s", GlobalLogManager.get_log_file_path())
    logger.info("meter model loaded: %s", args.model)
    logger.info("fire model loaded: %s", args.fire_model)
    logger.info("safehat model loaded: %s", args.safehat_model)
    logger.info("meter annotation mode: %s", args.annotation_mode)
    logger.info("listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("server interrupted by keyboard")
    finally:
        server.server_close()
        logger.info("server closed")


if __name__ == "__main__":
    main()
