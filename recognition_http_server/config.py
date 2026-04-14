from __future__ import annotations

import argparse

from recognition_http_server.constants import (
    DEFAULT_CALLBACK_PORT,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_FIRE_MODEL_PATH,
    DEFAULT_HOST,
    DEFAULT_IMGSZ,
    DEFAULT_MAX_VALUE,
    DEFAULT_METER_MODEL_PATH,
    DEFAULT_MIN_VALUE,
    DEFAULT_PORT,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_RESULT_ROOT,
    DEFAULT_SAFEHAT_MODEL_PATH,
)


def parse_args() -> argparse.Namespace:
    """解析 HTTP 识别服务的命令行启动参数。"""
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
    parser.add_argument("--result-root", type=type(DEFAULT_RESULT_ROOT), default=DEFAULT_RESULT_ROOT, help="Result root directory.")
    parser.add_argument("--callback-port", type=int, default=DEFAULT_CALLBACK_PORT, help="Fixed callback port.")
    parser.add_argument(
        "--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP download/callback timeout in seconds."
    )
    return parser.parse_args()
