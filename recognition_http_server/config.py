from __future__ import annotations

import argparse
import os

from recognition_http_server.constants import (
    DEFAULT_CALLBACK_PORT,
    DEFAULT_CONF,
    DEFAULT_DEVICE,
    DEFAULT_FIRE_EXTINGUISHER_MODEL_PATH,
    DEFAULT_FIRE_MODEL_PATH,
    DEFAULT_FIRE_PROTECTION_FACILITIES_MODEL_PATH,
    DEFAULT_HOST,
    DEFAULT_IMGSZ,
    DEFAULT_MAX_VALUE,
    DEFAULT_METER_MODEL_PATH,
    DEFAULT_MIN_VALUE,
    DEFAULT_PERSON_AND_CARS_MODEL_PATH,
    DEFAULT_PERSON_FALL_DOWN_MODEL_PATH,
    DEFAULT_PTZ_ALIGN_ENABLED,
    DEFAULT_PTZ_ALIGN_MAX_DELTA_DEG,
    DEFAULT_PTZ_ALIGN_MAX_PASSES,
    DEFAULT_PTZ_ALIGN_THRESHOLD_DEG,
    DEFAULT_PTZ_CHANNEL,
    DEFAULT_PTZ_HOST,
    DEFAULT_PTZ_HORIZONTAL_FOV_DEG,
    DEFAULT_PTZ_PASSWORD,
    DEFAULT_PTZ_PAN_NUDGE_DEGREES_PER_SECOND,
    DEFAULT_PTZ_PORT,
    DEFAULT_PTZ_SETTLE_SECONDS,
    DEFAULT_PTZ_TILT_MAX_DEG,
    DEFAULT_PTZ_TILT_MIN_DEG,
    DEFAULT_PTZ_TILT_NUDGE_DEGREES_PER_SECOND,
    DEFAULT_PTZ_NUDGE_DEGREES_PER_SECOND,
    DEFAULT_PTZ_NUDGE_MAX_SECONDS,
    DEFAULT_PTZ_NUDGE_MAX_STEPS,
    DEFAULT_PTZ_NUDGE_MIN_SECONDS,
    DEFAULT_PTZ_NUDGE_SPEED,
    DEFAULT_PTZ_TILT_NUDGE_SCALE,
    DEFAULT_PTZ_USERNAME,
    DEFAULT_PTZ_VERTICAL_FOV_DEG,
    DEFAULT_PTZ_ZOOM_ENABLED,
    DEFAULT_PTZ_ZOOM_FOCUS_TIMEOUT,
    DEFAULT_PTZ_ZOOM_MAX_RATIO,
    DEFAULT_PTZ_ZOOM_MAX_PASSES,
    DEFAULT_PTZ_ZOOM_NUDGE_SECONDS,
    DEFAULT_PTZ_ZOOM_NUDGE_SPEED,
    DEFAULT_PTZ_ZOOM_NUDGE_STEPS,
    DEFAULT_PTZ_ZOOM_RATIO_TOLERANCE,
    DEFAULT_PTZ_ZOOM_TARGET_HEIGHT_RATIO,
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
    parser.add_argument(
        "--fire-model", default=DEFAULT_FIRE_MODEL_PATH, help="Fire detection model path (.pt or .onnx)."
    )
    parser.add_argument(
        "--safehat-model",
        default=DEFAULT_SAFEHAT_MODEL_PATH,
        help="Safehat/person detection model path (.pt or .onnx).",
    )
    parser.add_argument(
        "--fire-protection-facilities-model",
        default=DEFAULT_FIRE_PROTECTION_FACILITIES_MODEL_PATH,
        help="Fire protection facilities detection model path (.pt or .onnx).",
    )
    parser.add_argument(
        "--person-fall-down-model",
        default=DEFAULT_PERSON_FALL_DOWN_MODEL_PATH,
        help="Person fall-down detection model path (.pt or .onnx).",
    )
    parser.add_argument(
        "--fire-extinguisher-model",
        default=DEFAULT_FIRE_EXTINGUISHER_MODEL_PATH,
        help="Fire extinguisher detection model path (.pt or .onnx).",
    )
    parser.add_argument(
        "--person-and-cars-model",
        default=DEFAULT_PERSON_AND_CARS_MODEL_PATH,
        help="Person and cars detection model path (.pt or .onnx).",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Inference confidence threshold.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, e.g. 'cpu' or '0'.")
    parser.add_argument("--min-value", type=float, default=DEFAULT_MIN_VALUE, help="Default meter minimum value.")
    parser.add_argument("--max-value", type=float, default=DEFAULT_MAX_VALUE, help="Default meter maximum value.")
    parser.add_argument(
        "--result-root", type=type(DEFAULT_RESULT_ROOT), default=DEFAULT_RESULT_ROOT, help="Result root directory."
    )
    parser.add_argument("--callback-port", type=int, default=DEFAULT_CALLBACK_PORT, help="Fixed callback port.")
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP download/callback timeout in seconds.",
    )
    parser.add_argument(
        "--ptz-align-enabled",
        action="store_true",
        default=DEFAULT_PTZ_ALIGN_ENABLED,
        help="Enable Hikvision PTZ auto-alignment before pointer meter ROI retry.",
    )
    parser.add_argument("--ptz-host", default=os.environ.get("HIK_HOST", DEFAULT_PTZ_HOST), help="Hikvision device host.")
    parser.add_argument(
        "--ptz-port",
        type=int,
        default=int(os.environ.get("HIK_PORT", DEFAULT_PTZ_PORT)),
        help="Hikvision SDK port.",
    )
    parser.add_argument("--ptz-username", default=os.environ.get("HIK_USERNAME", DEFAULT_PTZ_USERNAME))
    parser.add_argument("--ptz-password", default=os.environ.get("HIK_PASSWORD", DEFAULT_PTZ_PASSWORD))
    parser.add_argument(
        "--ptz-channel",
        type=int,
        default=int(os.environ.get("HIK_CHANNEL", DEFAULT_PTZ_CHANNEL)),
    )
    parser.add_argument("--ptz-local-ip", default=os.environ.get("HIK_LOCAL_IP", ""))
    parser.add_argument(
        "--ptz-sdk-lib-dir",
        default=os.environ.get("HIK_SDK_LIB_DIR", ""),
        help="Hikvision SDK library directory. Defaults to HK_SDK by CPU architecture.",
    )
    parser.add_argument("--ptz-horizontal-fov-deg", type=float, default=DEFAULT_PTZ_HORIZONTAL_FOV_DEG)
    parser.add_argument("--ptz-vertical-fov-deg", type=float, default=DEFAULT_PTZ_VERTICAL_FOV_DEG)
    parser.add_argument("--ptz-align-threshold-deg", type=float, default=DEFAULT_PTZ_ALIGN_THRESHOLD_DEG)
    parser.add_argument("--ptz-align-max-delta-deg", type=float, default=DEFAULT_PTZ_ALIGN_MAX_DELTA_DEG)
    parser.add_argument("--ptz-align-max-passes", type=int, default=DEFAULT_PTZ_ALIGN_MAX_PASSES)
    parser.add_argument("--ptz-tilt-min-deg", type=float, default=DEFAULT_PTZ_TILT_MIN_DEG)
    parser.add_argument("--ptz-tilt-max-deg", type=float, default=DEFAULT_PTZ_TILT_MAX_DEG)
    parser.add_argument("--ptz-settle-seconds", type=float, default=DEFAULT_PTZ_SETTLE_SECONDS)
    parser.add_argument("--ptz-nudge-speed", type=int, default=DEFAULT_PTZ_NUDGE_SPEED)
    parser.add_argument("--ptz-nudge-degrees-per-second", type=float, default=DEFAULT_PTZ_NUDGE_DEGREES_PER_SECOND)
    parser.add_argument("--ptz-pan-nudge-degrees-per-second", type=float, default=DEFAULT_PTZ_PAN_NUDGE_DEGREES_PER_SECOND)
    parser.add_argument("--ptz-tilt-nudge-degrees-per-second", type=float, default=DEFAULT_PTZ_TILT_NUDGE_DEGREES_PER_SECOND)
    parser.add_argument("--ptz-nudge-min-seconds", type=float, default=DEFAULT_PTZ_NUDGE_MIN_SECONDS)
    parser.add_argument("--ptz-nudge-max-seconds", type=float, default=DEFAULT_PTZ_NUDGE_MAX_SECONDS)
    parser.add_argument("--ptz-nudge-max-steps", type=int, default=DEFAULT_PTZ_NUDGE_MAX_STEPS)
    parser.add_argument("--ptz-tilt-nudge-scale", type=float, default=DEFAULT_PTZ_TILT_NUDGE_SCALE)
    zoom_group = parser.add_mutually_exclusive_group()
    zoom_group.add_argument(
        "--ptz-zoom-enabled",
        dest="ptz_zoom_enabled",
        action="store_true",
        help="Enable optical zoom adjustment after PTZ pan/tilt alignment.",
    )
    zoom_group.add_argument(
        "--no-ptz-zoom-enabled",
        dest="ptz_zoom_enabled",
        action="store_false",
        help="Disable optical zoom adjustment after PTZ pan/tilt alignment.",
    )
    parser.set_defaults(ptz_zoom_enabled=DEFAULT_PTZ_ZOOM_ENABLED)
    parser.add_argument("--ptz-zoom-target-height-ratio", type=float, default=DEFAULT_PTZ_ZOOM_TARGET_HEIGHT_RATIO)
    parser.add_argument("--ptz-zoom-ratio-tolerance", type=float, default=DEFAULT_PTZ_ZOOM_RATIO_TOLERANCE)
    parser.add_argument("--ptz-zoom-max-passes", type=int, default=DEFAULT_PTZ_ZOOM_MAX_PASSES)
    parser.add_argument("--ptz-zoom-nudge-speed", type=int, default=DEFAULT_PTZ_ZOOM_NUDGE_SPEED)
    parser.add_argument("--ptz-zoom-nudge-seconds", type=float, default=DEFAULT_PTZ_ZOOM_NUDGE_SECONDS)
    parser.add_argument("--ptz-zoom-nudge-steps", type=int, default=DEFAULT_PTZ_ZOOM_NUDGE_STEPS)
    parser.add_argument("--ptz-zoom-focus-timeout", type=float, default=DEFAULT_PTZ_ZOOM_FOCUS_TIMEOUT)
    parser.add_argument(
        "--ptz-zoom-max-ratio",
        type=float,
        default=DEFAULT_PTZ_ZOOM_MAX_RATIO,
        help="Optional maximum zoom ratio. Use 0 to estimate from SDK FOV range.",
    )
    return parser.parse_args()
