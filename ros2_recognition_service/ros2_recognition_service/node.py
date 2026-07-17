#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import Image
from std_msgs.msg import String

DEFAULT_REPO_ROOT = Path("/data/prj/yolov8_dial_reading/ultralytics")
repo_root = Path(os.environ.get("RECOGNITION_REPO_ROOT", DEFAULT_REPO_ROOT)).resolve()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ros2_recognition_service.srv import GetRecognitionTask, SubmitRecognitionTask
from server import (
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
    RecognitionService,
    TaskState,
    make_error_payload,
    now_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 recognition service node.")
    parser.add_argument("--node-name", default="recognition_service_node", help="ROS 2 node name.")
    parser.add_argument("--model", default=DEFAULT_METER_MODEL_PATH, help="Meter model path (.pt or .onnx).")
    parser.add_argument(
        "--fire-model", default=DEFAULT_FIRE_MODEL_PATH, help="Fire detection model path (.pt or .onnx)."
    )
    parser.add_argument(
        "--safehat-model",
        default=DEFAULT_SAFEHAT_MODEL_PATH,
        help="Safehat/person detection model path (.pt or .onnx).",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="Inference confidence threshold.")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Inference device, e.g. 'cpu' or '0'.")
    parser.add_argument("--min-value", type=float, default=DEFAULT_MIN_VALUE, help="Default meter minimum value.")
    parser.add_argument("--max-value", type=float, default=DEFAULT_MAX_VALUE, help="Default meter maximum value.")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT, help="Result root directory.")
    parser.add_argument(
        "--callback-port", type=int, default=DEFAULT_CALLBACK_PORT, help="Reserved callback port field for parity."
    )
    parser.add_argument(
        "--request-timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT, help="HTTP download timeout in seconds."
    )
    parser.add_argument(
        "--callback-topic",
        default="/api/v1/recognition/callback",
        help="Topic used to publish finished callback payloads as JSON strings.",
    )
    parser.add_argument(
        "--submit-service",
        default="/api/v1/recognition/tasks/submit",
        help="ROS 2 service name for task submission.",
    )
    parser.add_argument(
        "--status-service",
        default="/api/v1/recognition/tasks/get",
        help="ROS 2 service name for task status query.",
    )
    parser.add_argument(
        "--image-topic",
        default="",
        help="Optional sensor_msgs/msg/Image topic. When set, each received image will create a recognition task.",
    )
    parser.add_argument(
        "--image-qos-depth",
        type=int,
        default=10,
        help="Queue depth for the image subscription.",
    )
    parser.add_argument(
        "--image-recognize-type",
        default="1",
        help="recognize_type used for tasks created from subscribed images.",
    )
    parser.add_argument(
        "--image-recognize-subtype",
        default="default",
        help="recognize_subtype used for tasks created from subscribed images.",
    )
    cli_args = remove_ros_args(args=sys.argv)[1:]
    return parser.parse_args(cli_args)


class RosRecognitionService(RecognitionService):
    def __init__(self, config: argparse.Namespace, callback_publisher) -> None:
        self._callback_publisher = callback_publisher
        super().__init__(config)

    def create_task(self, payload: dict[str, Any], callback_host: str = "ros2") -> TaskState:
        return super().create_task(payload, callback_host)

    def _send_callback(self, req_id: str, callback_url: str, callback_payload: dict[str, Any]) -> None:
        del callback_url
        message = String()
        message.data = json.dumps(callback_payload, ensure_ascii=False)
        self._callback_publisher.publish(message)
        self._update_task(req_id, callback_status_code=0, callback_error=None)


class RecognitionServiceNode(Node):
    def __init__(self, config: argparse.Namespace) -> None:
        super().__init__(config.node_name)
        self._bridge = CvBridge()
        self._callback_publisher = self.create_publisher(String, config.callback_topic, 10)
        self.service = RosRecognitionService(config, self._callback_publisher)
        self._submit_service = self.create_service(
            SubmitRecognitionTask, config.submit_service, self._handle_submit_task
        )
        self._status_service = self.create_service(GetRecognitionTask, config.status_service, self._handle_get_task)
        self._image_subscription = None

        if config.image_topic:
            self._image_subscription = self.create_subscription(
                Image,
                config.image_topic,
                self._handle_image_message,
                config.image_qos_depth,
            )

        self.get_logger().info(f"meter model loaded: {config.model}")
        self.get_logger().info(f"fire model loaded: {config.fire_model}")
        self.get_logger().info(f"safehat model loaded: {config.safehat_model}")
        self.get_logger().info(f"meter annotation mode: {config.annotation_mode}")
        self.get_logger().info(f"submit service: {config.submit_service}")
        self.get_logger().info(f"status service: {config.status_service}")
        self.get_logger().info(f"callback topic: {config.callback_topic}")
        if config.image_topic:
            self.get_logger().info(f"image topic: {config.image_topic}")
            self.get_logger().info(
                f"image topic data_type: type={config.image_recognize_type}, subtype={config.image_recognize_subtype}"
            )

    def _handle_submit_task(self, request, response):
        try:
            payload = json.loads(request.request_json)
            if not isinstance(payload, dict):
                raise ValueError("request_json must decode to an object.")
            task = self.service.create_task(payload)
            result = {
                "req_id": task.req_id,
                "code": 0,
                "resp_msg": "Task accepted successfully.",
                "data": {"task_status": task.status},
            }
        except ValueError as exc:
            req_id = None
            response.response_json = json.dumps(make_error_payload(req_id, str(exc)), ensure_ascii=False)
            return response
        except Exception as exc:
            req_id = None
            response.response_json = json.dumps(make_error_payload(req_id, str(exc)), ensure_ascii=False)
            return response

        response.response_json = json.dumps(result, ensure_ascii=False)
        return response

    def _handle_get_task(self, request, response):
        req_id = request.req_id.strip()
        task = self.service.get_task(req_id)
        if task is None:
            response.response_json = json.dumps(make_error_payload(req_id, "Task not found."), ensure_ascii=False)
            return response

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
        response.response_json = json.dumps(payload, ensure_ascii=False)
        return response

    def _handle_image_message(self, message: Image) -> None:
        req_id = f"topic-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}"
        try:
            image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            image_path = self._save_subscribed_image(req_id, image)
            payload = {
                "req_id": req_id,
                "image_path": str(image_path),
                "data_type": [
                    {
                        "recognize_type": self.service.config.image_recognize_type,
                        "recognize_subtype": self.service.config.image_recognize_subtype,
                    }
                ],
            }
            task = self.service.create_task(payload)
            self.get_logger().info(f"created task from image topic: req_id={task.req_id}")
        except Exception as exc:
            self.get_logger().error(f"failed to process subscribed image: {exc}")

    def _save_subscribed_image(self, req_id: str, image) -> Path:
        topic_input_dir = self.service.input_root / "ros2_topic" / req_id
        topic_input_dir.mkdir(parents=True, exist_ok=True)
        image_path = topic_input_dir / "frame.jpg"
        if not cv2.imwrite(str(image_path), image):
            raise RuntimeError(f"failed to write subscribed image to {image_path}")
        return image_path


def main() -> None:
    args = parse_args()
    args.result_root = args.result_root.resolve()
    args.host = DEFAULT_HOST
    args.port = DEFAULT_PORT

    rclpy.init()
    node = RecognitionServiceNode(args)
    try:
        node.get_logger().info(f"node started at {now_text()}")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
