from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _read_config_file(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"launch config file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"launch config file must decode to a mapping: {config_path}")
    return data


def _resolve_value(context, key: str, default: str) -> str:
    override = LaunchConfiguration(key).perform(context).strip()
    return override if override else default


def _build_actions(context):
    config_file = Path(LaunchConfiguration("config_file").perform(context)).expanduser().resolve()
    file_config = _read_config_file(config_file)

    defaults = {
        "repo_root": "/data/prj/yolov8_dial_reading/ultralytics",
        "node_name": "recognition_service_node",
        "model": "runs/train/meter_data_9k_yolov8m_best/weights/best.pt",
        "fire_model": "runs/train/best_fire.pt",
        "safehat_model": "runs/train/best_person.pt",
        "imgsz": "640",
        "conf": "0.25",
        "device": "0",
        "min_value": "0.0",
        "max_value": "1.0",
        "result_root": "results/http_service",
        "callback_port": "18080",
        "request_timeout": "15",
        "callback_topic": "/api/v1/recognition/callback",
        "submit_service": "/api/v1/recognition/tasks/submit",
        "status_service": "/api/v1/recognition/tasks/get",
        "image_topic": "",
        "image_qos_depth": "10",
        "image_recognize_type": "1",
        "image_recognize_subtype": "default",
    }

    values = {}
    for key, fallback in defaults.items():
        file_value = file_config.get(key, fallback)
        values[key] = _resolve_value(context, key, str(file_value))

    return [
        SetEnvironmentVariable("RECOGNITION_REPO_ROOT", values["repo_root"]),
        Node(
            package="ros2_recognition_service",
            executable="recognition_node",
            name=values["node_name"],
            output="screen",
            arguments=[
                "--node-name",
                values["node_name"],
                "--model",
                values["model"],
                "--fire-model",
                values["fire_model"],
                "--safehat-model",
                values["safehat_model"],
                "--imgsz",
                values["imgsz"],
                "--conf",
                values["conf"],
                "--device",
                values["device"],
                "--min-value",
                values["min_value"],
                "--max-value",
                values["max_value"],
                "--result-root",
                values["result_root"],
                "--callback-port",
                values["callback_port"],
                "--request-timeout",
                values["request_timeout"],
                "--callback-topic",
                values["callback_topic"],
                "--submit-service",
                values["submit_service"],
                "--status-service",
                values["status_service"],
                "--image-topic",
                values["image_topic"],
                "--image-qos-depth",
                values["image_qos_depth"],
                "--image-recognize-type",
                values["image_recognize_type"],
                "--image-recognize-subtype",
                values["image_recognize_subtype"],
            ],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    package_share_dir = Path(get_package_share_directory("ros2_recognition_service"))
    default_config_file = package_share_dir / "config" / "recognition.yaml"

    configurable_keys = [
        "repo_root",
        "node_name",
        "model",
        "fire_model",
        "safehat_model",
        "imgsz",
        "conf",
        "device",
        "min_value",
        "max_value",
        "result_root",
        "callback_port",
        "request_timeout",
        "callback_topic",
        "submit_service",
        "status_service",
        "image_topic",
        "image_qos_depth",
        "image_recognize_type",
        "image_recognize_subtype",
    ]

    actions = [
        DeclareLaunchArgument(
            "config_file",
            default_value=str(default_config_file),
            description="YAML file used to configure the recognition node.",
        )
    ]

    for key in configurable_keys:
        actions.append(
            DeclareLaunchArgument(
                key,
                default_value="",
                description=f"Optional override for '{key}'. Empty means using the value from config_file.",
            )
        )

    actions.append(OpaqueFunction(function=_build_actions))
    return LaunchDescription(actions)
