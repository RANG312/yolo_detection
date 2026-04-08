from __future__ import annotations

import argparse
import json
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


DEFAULT_MODEL_PATH = "runs/train/meter_data_9k_yolov8m_best/weights/best.pt"
DEFAULT_FIRE_MODEL_PATH = "runs/train/best_fire.pt"
DEFAULT_SAFEHAT_MODEL_PATH = "runs/train/best_person.pt"
DEFAULT_MIN_VALUE = 0.0
DEFAULT_MAX_VALUE = 1.0
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25
DEFAULT_DEVICE = "0"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 3208
DEFAULT_CALLBACK_PORT = 18080
DEFAULT_CALLBACK_PATH = "/api/v1/recognition/callback"
DEFAULT_RESULT_ROOT = Path("results/http_service")
DEFAULT_REQUEST_TIMEOUT = 15

RECOGNIZE_TYPE_METER = "1"
RECOGNIZE_TYPE_FIRE = "6"
RECOGNIZE_TYPE_SAFEHAT = "7"
TASK_KIND_METER = "meter"
TASK_KIND_FIRE = "fire"
TASK_KIND_SAFEHAT = "safehat"
DEFAULT_DATA_TYPE = {"recognize_type": RECOGNIZE_TYPE_METER, "recognize_subtype": "default"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dial reading HTTP service.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Server bind host.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server bind port.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Model path (.pt or .onnx).")
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
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def split_image_paths(image_path_value: str) -> list[str]:
    return [item.strip() for item in image_path_value.split(",") if item.strip()]


def parse_extra_info(extra_info: Any) -> dict[str, Any]:
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
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def make_filename_from_url(url: str, fallback: str) -> str:
    path_name = Path(urlparse(url).path).name
    if path_name:
        return path_name
    return fallback


def build_callback_url(callback_host: str, callback_port: int) -> str:
    return f"http://{callback_host}:{callback_port}{DEFAULT_CALLBACK_PATH}"


def resolve_task_kind(recognize_type: str) -> str:
    recognize_type_text = str(recognize_type).strip()
    if recognize_type_text in {RECOGNIZE_TYPE_METER, "dict_meter_type"}:
        return TASK_KIND_METER
    if recognize_type_text in {RECOGNIZE_TYPE_FIRE, "fire"}:
        return TASK_KIND_FIRE
    if recognize_type_text in {RECOGNIZE_TYPE_SAFEHAT, "safehat", "person"}:
        return TASK_KIND_SAFEHAT
    raise ValueError(f"Unsupported recognize_type: {recognize_type_text}")


def normalize_data_types(data_types: Any) -> list[dict[str, str]]:
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
    if not image_paths:
        raise ValueError("image_path is required and must contain at least one image.")
    if not data_types:
        raise ValueError("data_type must contain at least one item.")
    if len(data_types) == 1:
        return [data_types[0].copy() for _ in image_paths]
    if len(image_paths) != len(data_types):
        raise ValueError("When multiple data_type items are provided, the number of data_type items must match image_path.")
    return [item.copy() for item in data_types]


def build_predict_args(
    config: argparse.Namespace,
    save_path: Path,
    min_value: float,
    max_value: float,
    debug_center: bool,
) -> SimpleNamespace:
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
    return {
        "recognize_type": data_type["recognize_type"],
        "recognize_subtype": data_type["recognize_subtype"],
        "recognize_value": f"{reading:.6f}",
        "confidence": "100",
        "recognize_desc": desc,
    }


def build_error_data_entry(data_type: dict[str, str], error_text: str) -> dict[str, str]:
    return {
        "recognize_type": data_type["recognize_type"],
        "recognize_subtype": data_type["recognize_subtype"],
        "recognize_value": "",
        "confidence": "0",
        "recognize_desc": f"识别失败: {error_text}",
    }


@dataclass
class TaskState:
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


class DialReadingService:
    def __init__(self, config: argparse.Namespace) -> None:
        self.config = config
        self.result_root = config.result_root
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.input_root = self.result_root / "inputs"
        self.output_root = self.result_root / "outputs"
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.meter_model, self.config.annotation_mode = load_model(config.model)
        self.fire_model = YOLO(config.fire_model)
        self.safehat_model = YOLO(config.safehat_model)
        self._tasks: dict[str, TaskState] = {}
        self._tasks_lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def create_task(self, payload: dict[str, Any], callback_host: str) -> TaskState:
        req_id = str(payload.get("req_id") or uuid4())
        image_path_value = payload.get("image_path")
        if not isinstance(image_path_value, str) or not image_path_value.strip():
            raise ValueError("image_path is required and must be a non-empty string.")

        image_paths = split_image_paths(image_path_value)
        data_types = normalize_data_types(payload.get("data_type"))
        align_data_types_to_images(image_paths, data_types)
        parse_extra_info(payload.get("extra_info", ""))

        with self._tasks_lock:
            if req_id in self._tasks:
                raise ValueError(f"req_id already exists: {req_id}")
            task = TaskState(req_id=req_id, callback_host=callback_host, status="processing", request_payload=payload)
            self._tasks[req_id] = task

        worker = threading.Thread(target=self._process_task, args=(req_id,), daemon=True)
        worker.start()
        return task

    def get_task(self, req_id: str) -> TaskState | None:
        with self._tasks_lock:
            return self._tasks.get(req_id)

    def _update_task(self, req_id: str, **updates: Any) -> None:
        with self._tasks_lock:
            task = self._tasks[req_id]
            for key, value in updates.items():
                setattr(task, key, value)
            task.updated_at = now_text()

    def _process_task(self, req_id: str) -> None:
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
        local_image_path = self._prepare_image(req_id, index, image_path)
        image_output_dir = self.output_root / req_id
        image_output_dir.mkdir(parents=True, exist_ok=True)
        recognize_items: list[dict[str, str]] = []
        image_result_path: Path | None = None
        task_kind = resolve_task_kind(data_type["recognize_type"])

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
        predict_args = build_predict_args(self.config, visualize_path, DEFAULT_MIN_VALUE, DEFAULT_MAX_VALUE, debug_center)
        with self._predict_lock:
            canvas, prediction_instances, _, _ = predict_image_instances(
                local_image_path, self.meter_model, predict_args, self.config.annotation_mode
            )
        save_canvas(canvas, visualize_path)

        recognize_items: list[dict[str, str]] = []
        for meter_instance in prediction_instances:
            meter_index = str(meter_instance["meter_index"])
            if meter_instance.get("error") is None:
                normalized_reading = float(meter_instance["reading"])
                arc_mode = str(meter_instance["arc_mode"])
                for data_type in data_types:
                    desc = f"识别成功，表计#{meter_index}归一化读数为{normalized_reading:.6f}，arc_mode={arc_mode}"
                    recognize_item = build_success_data_entry(data_type, normalized_reading, desc)
                    recognize_item["meter_index"] = meter_index
                    recognize_items.append(recognize_item)
            else:
                error_text = str(meter_instance["error"])
                for data_type in data_types:
                    recognize_item = build_error_data_entry(data_type, error_text)
                    recognize_item["meter_index"] = meter_index
                    recognize_items.append(recognize_item)
        return recognize_items

    def _run_detection_recognition(
        self,
        local_image_path: Path,
        model: YOLO,
        data_types: list[dict[str, str]],
        visualize_path: Path,
        task_desc: str,
    ) -> list[dict[str, str]]:
        image = cv2.imread(str(local_image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {local_image_path}")

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
                recognize_item["meter_index"] = "0"
                recognize_items.append(recognize_item)
            return recognize_items

        names = result.names if isinstance(result.names, dict) else {i: name for i, name in enumerate(result.names)}
        for detect_index, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item()) if box.conf is not None else 0.0
            label = str(names.get(cls_id, cls_id))
            for data_type in data_types:
                recognize_items.append(
                    {
                        "meter_index": str(detect_index),
                        "recognize_type": data_type["recognize_type"],
                        "recognize_subtype": data_type["recognize_subtype"],
                        "recognize_value": label,
                        "confidence": str(int(round(confidence * 100))),
                        "recognize_desc": f"识别成功，{task_desc}检测到{label}，置信度={confidence:.4f}",
                    }
                )

        return recognize_items

    def _prepare_image(self, req_id: str, index: int, image_path: str) -> Path:
        task_input_dir = self.input_root / req_id
        task_input_dir.mkdir(parents=True, exist_ok=True)

        if is_http_url(image_path):
            filename = make_filename_from_url(image_path, f"image_{index}.jpg")
            target_path = task_input_dir / filename
            response = requests.get(image_path, timeout=self.config.request_timeout)
            response.raise_for_status()
            target_path.write_bytes(response.content)
            return target_path

        local_path = Path(image_path).expanduser()
        if not local_path.is_absolute():
            local_path = Path.cwd() / local_path
        if not local_path.exists():
            raise FileNotFoundError(f"Image does not exist: {local_path}")
        return local_path

    def _send_callback(self, req_id: str, callback_url: str, callback_payload: dict[str, Any]) -> None:
        try:
            response = requests.post(callback_url, json=callback_payload, timeout=self.config.request_timeout)
            self._update_task(req_id, callback_status_code=response.status_code, callback_error=None)
        except Exception as exc:
            self._update_task(req_id, callback_error=str(exc))


def make_error_payload(req_id: str | None, message: str, status: str = "finished") -> dict[str, Any]:
    return {
        "req_id": req_id,
        "code": 1,
        "resp_msg": message,
        "data": {"task_status": status},
    }


class RecognitionHandler(BaseHTTPRequestHandler):
    server: "RecognitionHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(
                HTTPStatus.OK,
                {"code": 0, "resp_msg": "ok", "data": {"status": "healthy", "time": now_text()}},
            )
            return

        if self.path.startswith("/api/v1/recognition/tasks/"):
            req_id = self.path.rsplit("/", 1)[-1]
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
        if self.path != "/api/v1/recognition/tasks":
            self._send_json(HTTPStatus.NOT_FOUND, make_error_payload(None, "Path not found."))
            return

        payload: dict[str, Any] | None = None
        try:
            payload = self._read_json_body()
            callback_host = self._get_callback_host()
            task = self.server.service.create_task(payload, callback_host)
        except ValueError as exc:
            req_id = None
            if isinstance(exc.args[0], str) and "req_id" in str(exc):
                req_id = str(payload.get("req_id")) if isinstance(payload, dict) else None
            self._send_json(HTTPStatus.BAD_REQUEST, make_error_payload(req_id, str(exc)))
            return
        except Exception as exc:
            req_id = payload.get("req_id") if isinstance(payload, dict) else None
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, make_error_payload(req_id, str(exc)))
            return

        response = {
            "req_id": task.req_id,
            "code": 0,
            "resp_msg": "Task accepted successfully.",
            "data": {"task_status": task.status},
        }
        self._send_json(HTTPStatus.OK, response)

    def _get_callback_host(self) -> str:
        forwarded_for = self.headers.get("X-Forwarded-For", "").strip()
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = self.headers.get("X-Real-IP", "").strip()
        if real_ip:
            return real_ip
        return self.client_address[0]

    def log_message(self, format: str, *args: Any) -> None:
        timestamp = now_text()
        message = format % args
        print(f"[{timestamp}] {self.address_string()} {message}")

    def _read_json_body(self) -> dict[str, Any]:
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
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class RecognitionHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], request_handler_class: type[RecognitionHandler], service: DialReadingService):
        super().__init__(server_address, request_handler_class)
        self.service = service


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    service = DialReadingService(args)
    server = RecognitionHTTPServer((args.host, args.port), RecognitionHandler, service)
    print(f"[{now_text()}] meter model loaded: {args.model}")
    print(f"[{now_text()}] fire model loaded: {args.fire_model}")
    print(f"[{now_text()}] safehat model loaded: {args.safehat_model}")
    print(f"[{now_text()}] meter annotation mode: {args.annotation_mode}")
    print(f"[{now_text()}] listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
