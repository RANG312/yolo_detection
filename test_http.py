from __future__ import annotations

import argparse
import json
import os
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests


DEFAULT_SERVER = "http://127.0.0.1:3208"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_CALLBACK_HOST = "0.0.0.0"
DEFAULT_CALLBACK_PORT = 8088
DEFAULT_CALLBACK_PATH = "/api/v1/recognition/callback"
DEFAULT_TIMEOUT = 120
DEFAULT_OUTPUT = Path("results/http_service/test_callback_payload.json")
DEFAULT_HIK_OUTPUT_DIR = Path("results/http_service/hikvision_captures")

SCENE_METER = "meter"
SCENE_FIRE = "fire"
SCENE_SAFEHAT = "safehat"
SCENE_DEFAULT_DATA_TYPES = {
    SCENE_METER: [
        {"recognize_type": "1", "recognize_subtype": "3"},
        {"recognize_type": "1", "recognize_subtype": "5"},
    ],
    SCENE_FIRE: [{"recognize_type": "6", "recognize_subtype": ""}],
    SCENE_SAFEHAT: [{"recognize_type": "7", "recognize_subtype": ""}],
}


class CallbackState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.payload: dict[str, Any] | None = None
        self.received_at: str | None = None


class CallbackHandler(BaseHTTPRequestHandler):
    server: "CallbackServer"

    def do_POST(self) -> None:  # noqa: N802
        if self.path != self.server.callback_path:
            self._send_json(HTTPStatus.NOT_FOUND, {"code": 1, "resp_msg": "Path not found."})
            return

        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"code": 1, "resp_msg": str(exc)})
            return

        self.server.state.payload = payload
        self.server.state.received_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.server.state.event.set()
        self._send_json(HTTPStatus.OK, {"code": 0, "resp_msg": "Callback received."})

    def log_message(self, format: str, *args: Any) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] callback-server {self.address_string()} {format % args}")

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            raise ValueError("Request body is empty.")
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


class CallbackServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler_class: type[CallbackHandler],
        state: CallbackState,
        callback_path: str,
    ) -> None:
        super().__init__(server_address, request_handler_class)
        self.state = state
        self.callback_path = callback_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end HTTP test client for the recognition service.")
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help="Recognition service base URL, e.g. http://127.0.0.1:8000",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host used to build the advertised callback_url.")
    parser.add_argument(
        "--images",
        nargs="+",
        help="One or more local image paths or HTTP image URLs. Multiple images will be joined by comma.",
    )
    parser.add_argument(
        "--data-type",
        action="append",
        default=[],
        help="Recognition item in type[:subtype] format, e.g. 1:3, 6, 7, 8, 9, 10, 11. Can be repeated.",
    )
    parser.add_argument(
        "--recognize-type",
        action="append",
        default=[],
        help="Explicit recognize_type value. Can be repeated. Use with --recognize-subtype.",
    )
    parser.add_argument(
        "--recognize-subtype",
        action="append",
        default=[],
        help="Explicit recognize_subtype value. Can be repeated. Supports one shared value or one value per recognize_type.",
    )
    parser.add_argument(
        "--scene",
        choices=[SCENE_METER, SCENE_FIRE, SCENE_SAFEHAT],
        default=SCENE_METER,
        help="Preset request scene used when --data-type is not provided.",
    )
    parser.add_argument("--callback-host", default=DEFAULT_CALLBACK_HOST, help="Local callback server bind host.")
    parser.add_argument("--callback-port", type=int, default=DEFAULT_CALLBACK_PORT, help="Local callback server bind port.")
    parser.add_argument("--callback-path", default=DEFAULT_CALLBACK_PATH, help="Callback path.")
    parser.add_argument(
        "--callback-url",
        default=None,
        help="Explicit callback URL advertised to the recognition service. Defaults to http://<host>:<port><path>",
    )
    parser.add_argument("--debug-center", action="store_true", help="Pass debug_center=true in extra_info.")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Wait timeout in seconds for callback.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Where to save callback payload JSON.")
    parser.add_argument(
        "--hik-capture",
        action="store_true",
        help="Capture one snapshot from a Hikvision camera and use it as the request image.",
    )
    parser.add_argument("--hik-host", default=os.environ.get("HIK_HOST", ""), help="Hikvision device host.")
    parser.add_argument("--hik-port", type=int, default=8000, help="Hikvision SDK port.")
    parser.add_argument("--hik-username", default=os.environ.get("HIK_USERNAME", "admin"))
    parser.add_argument("--hik-password", default=os.environ.get("HIK_PASSWORD", ""))
    parser.add_argument("--hik-channel", type=int, default=1)
    parser.add_argument("--hik-snapshot-timeout", type=float, default=5.0, help="Hikvision snapshot timeout in seconds.")
    parser.add_argument(
        "--hik-output-dir",
        type=Path,
        default=DEFAULT_HIK_OUTPUT_DIR,
        help="Directory where --hik-capture stores the snapshot.",
    )
    parser.add_argument(
        "--hik-snapshot-name",
        default="",
        help="Optional filename for --hik-capture. Defaults to hik_YYYYmmdd_HHMMSS.jpg.",
    )
    parser.add_argument(
        "--hik-restore-delay",
        type=float,
        default=10.0,
        help="Seconds to wait after the test before restoring the initial Hikvision PTZ position.",
    )
    parser.add_argument(
        "--no-hik-restore",
        action="store_true",
        help="Do not restore the initial Hikvision PTZ position after --hik-capture test.",
    )
    return parser.parse_args()


def normalize_recognize_type(recognize_type: str) -> str:
    value = recognize_type.strip()
    mapping = {
        "1": "1",
        "meter": "1",
        "dict_meter_type": "1",
        "6": "6",
        "fire": "6",
        "7": "7",
        "safehat": "7",
        "person": "7",
        "8": "8",
        "fire_protection_facilities": "8",
        "9": "9",
        "person_fall_down": "9",
        "10": "10",
        "fire_extinguisher": "10",
        "11": "11",
        "person_and_cars": "11",
    }
    normalized = mapping.get(value.lower())
    if normalized is None:
        raise ValueError(f"Unsupported recognize_type: {recognize_type}")
    return normalized


def build_data_types_from_explicit_fields(raw_types: list[str], raw_subtypes: list[str]) -> list[dict[str, str]]:
    if not raw_types and not raw_subtypes:
        return []
    if not raw_types:
        raise ValueError("--recognize-subtype requires --recognize-type.")

    normalized_types = [normalize_recognize_type(item) for item in raw_types]
    if not raw_subtypes:
        resolved_subtypes = [""] * len(normalized_types)
    elif len(raw_subtypes) == 1:
        resolved_subtypes = [raw_subtypes[0]] * len(normalized_types)
    elif len(raw_subtypes) == len(normalized_types):
        resolved_subtypes = raw_subtypes
    else:
        raise ValueError("Provide either one shared --recognize-subtype or one --recognize-subtype per --recognize-type.")

    data_types = []
    for recognize_type, recognize_subtype_raw in zip(normalized_types, resolved_subtypes):
        recognize_subtype = "" if recognize_subtype_raw is None else str(recognize_subtype_raw).strip()
        if recognize_type == "1" and not recognize_subtype:
            recognize_subtype = "default"
        data_types.append({"recognize_type": recognize_type, "recognize_subtype": recognize_subtype})
    return data_types


def parse_data_types(raw_items: list[str], raw_types: list[str], raw_subtypes: list[str], scene: str) -> list[dict[str, str]]:
    if raw_items and (raw_types or raw_subtypes):
        raise ValueError("Use either --data-type or --recognize-type/--recognize-subtype, not both.")

    explicit_data_types = build_data_types_from_explicit_fields(raw_types, raw_subtypes)
    if explicit_data_types:
        return explicit_data_types

    if not raw_items:
        return [item.copy() for item in SCENE_DEFAULT_DATA_TYPES[scene]]

    data_types = []
    for item in raw_items:
        recognize_type_raw, sep, recognize_subtype_raw = item.partition(":")
        if not recognize_type_raw.strip():
            raise ValueError(f"Invalid --data-type value: {item}. Expected type[:subtype].")
        recognize_type = normalize_recognize_type(recognize_type_raw)
        recognize_subtype = recognize_subtype_raw.strip() if sep else ""
        if recognize_type == "1" and not recognize_subtype:
            recognize_subtype = "default"
        data_types.append({"recognize_type": recognize_type, "recognize_subtype": recognize_subtype})
    return data_types


def build_callback_url(args: argparse.Namespace) -> str:
    if args.callback_url:
        return args.callback_url
    return f"http://{args.host}:{args.callback_port}{args.callback_path}"


def create_hikvision_controller(args: argparse.Namespace):  # noqa: ANN201
    if not args.hik_host:
        raise ValueError("Pass --hik-host or set HIK_HOST when using --hik-capture.")
    if not args.hik_password:
        raise ValueError("Pass --hik-password or set HIK_PASSWORD when using --hik-capture.")

    from recognition_http_server.hikvision_ptz import HikvisionPTZConfig, HikvisionPTZController

    return HikvisionPTZController(
        HikvisionPTZConfig(
            host=args.hik_host,
            port=args.hik_port,
            username=args.hik_username,
            password=args.hik_password,
            channel=args.hik_channel,
            snapshot_timeout=args.hik_snapshot_timeout,
        )
    )


def capture_hikvision_image(args: argparse.Namespace) -> Path:
    import cv2

    output_dir = Path(args.hik_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_name = args.hik_snapshot_name.strip()
    if not snapshot_name:
        snapshot_name = f"hik_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    capture_path = output_dir / snapshot_name
    if not capture_path.suffix:
        capture_path = capture_path.with_suffix(".jpg")

    controller = create_hikvision_controller(args)
    image = controller.capture_image()
    if not cv2.imwrite(str(capture_path), image):
        raise RuntimeError(f"Failed to save Hikvision snapshot: {capture_path}")
    print(f"Hikvision snapshot saved to: {capture_path}")
    return capture_path


def resolve_image_paths(args: argparse.Namespace) -> list[str]:
    images = list(args.images or [])
    if args.hik_capture:
        capture_path = capture_hikvision_image(args)
        if images:
            print(f"--hik-capture is set; using captured image instead of --images: {capture_path}")
        return [str(capture_path)]
    if not images:
        raise ValueError("--images is required unless --hik-capture is set.")
    return images


def read_initial_hikvision_position(args: argparse.Namespace):  # noqa: ANN201
    if not args.hik_capture or args.no_hik_restore:
        return None
    position = create_hikvision_controller(args).read_position()
    print(
        "Initial Hikvision PTZ position: "
        f"pan={position.pan_deg:.3f} tilt={position.tilt_deg:.3f} zoom={position.zoom_deg:.3f}"
    )
    return position


def restore_hikvision_position_after_delay(args: argparse.Namespace, position) -> None:  # noqa: ANN001
    if position is None:
        return
    delay = max(0.0, float(args.hik_restore_delay))
    if delay > 0:
        print(f"Waiting {delay:.1f}s before restoring Hikvision PTZ position...")
        time.sleep(delay)
    create_hikvision_controller(args).set_position(position)
    print(
        "Hikvision PTZ position restored: "
        f"pan={position.pan_deg:.3f} tilt={position.tilt_deg:.3f} zoom={position.zoom_deg:.3f}"
    )


def build_request_payload(args: argparse.Namespace, callback_url: str) -> dict[str, Any]:
    extra_info: dict[str, Any] = {"callback_url": callback_url}
    if args.debug_center:
        extra_info["debug_center"] = True
    data_types = parse_data_types(args.data_type, args.recognize_type, args.recognize_subtype, args.scene)
    image_paths = resolve_image_paths(args)
    if len(data_types) not in {1, len(image_paths)}:
        raise ValueError(
            "One image uses one data_type. Provide either one shared item or one item per image for "
            "--data-type or --recognize-type/--recognize-subtype."
        )

    return {
        "req_id": str(uuid4()),
        "image_path": ",".join(image_paths),
        "data_type": data_types,
        "extra_info": json.dumps(extra_info, ensure_ascii=False),
    }


def start_callback_server(args: argparse.Namespace, state: CallbackState) -> CallbackServer:
    server = CallbackServer((args.callback_host, args.callback_port), CallbackHandler, state, args.callback_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Callback server listening on http://{args.callback_host}:{args.callback_port}{args.callback_path}")
    return server


def post_task(server_base: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = server_base.rstrip("/") + "/api/v1/recognition/tasks"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    print("Task submit response:")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return data


def poll_task(server_base: str, req_id: str) -> dict[str, Any] | None:
    url = server_base.rstrip("/") + f"/api/v1/recognition/tasks/{req_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def validate_callback_payload(payload: dict[str, Any]) -> None:
    required_top_fields = {"req_id", "data_result", "code", "resp_msg", "finish_time"}
    missing_top_fields = sorted(required_top_fields - set(payload))
    if missing_top_fields:
        raise ValueError(f"Callback payload missing top-level fields: {missing_top_fields}")

    if not isinstance(payload["data_result"], list):
        raise ValueError("Callback field data_result must be a list.")

    for image_result in payload["data_result"]:
        if not isinstance(image_result, dict):
            raise ValueError("Each data_result item must be an object.")
        for field_name in ("image_path", "image_path_result", "recognize_data"):
            if field_name not in image_result:
                raise ValueError(f"data_result item missing field: {field_name}")
        if not isinstance(image_result["recognize_data"], list):
            raise ValueError("recognize_data must be a list.")
        for recognize_item in image_result["recognize_data"]:
            for field_name in (
                "recognize_image_index",
                "recognize_type",
                "recognize_subtype",
                "recognize_value",
                "confidence",
                "recognize_desc",
            ):
                if field_name not in recognize_item:
                    raise ValueError(f"recognize_data item missing field: {field_name}")


def save_callback_payload(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Callback payload saved to: {output_path}")


def print_callback_summary(payload: dict[str, Any]) -> None:
    print("Callback payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("Summary:")
    for image_result in payload["data_result"]:
        image_path = image_result["image_path"]
        print(f"- image: {image_path}")
        for recognize_item in image_result["recognize_data"]:
            recognize_image_index = recognize_item["recognize_image_index"]
            recognize_type = recognize_item["recognize_type"]
            subtype = recognize_item["recognize_subtype"]
            value = recognize_item["recognize_value"]
            confidence = recognize_item["confidence"]
            desc = recognize_item["recognize_desc"]
            print(
                f"  recognize_image_index={recognize_image_index} type={recognize_type} subtype={subtype} "
                f"value={value} confidence={confidence} desc={desc}"
            )


def main() -> None:
    args = parse_args()
    state = CallbackState()
    callback_url = build_callback_url(args)
    initial_hikvision_position = read_initial_hikvision_position(args)
    callback_server = None

    try:
        payload = build_request_payload(args, callback_url)
        callback_server = start_callback_server(args, state)
        submit_response = post_task(args.server, payload)
        req_id = submit_response.get("req_id", payload["req_id"])
        deadline = time.time() + args.timeout

        while not state.event.wait(timeout=1.0):
            if time.time() >= deadline:
                raise TimeoutError(f"No callback received within {args.timeout} seconds.")
            task_state = poll_task(args.server, req_id)
            if task_state is not None:
                task_status = task_state.get("data", {}).get("task_status")
                if task_status:
                    print(f"Task status: {task_status}")

        callback_payload = state.payload
        if callback_payload is None:
            raise RuntimeError("Callback event was set but payload is empty.")

        validate_callback_payload(callback_payload)
        save_callback_payload(args.output, callback_payload)
        print_callback_summary(callback_payload)
    finally:
        if callback_server is not None:
            callback_server.shutdown()
            callback_server.server_close()
        restore_hikvision_position_after_delay(args, initial_hikvision_position)


if __name__ == "__main__":
    main()
