#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE_DIR="${HOME}/prj/ros2_ws"

ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
PACKAGE_NAME="ros2_recognition_service"
CONFIG_FILE="${CONFIG_FILE:-${WORKSPACE_DIR}/config/recognition.yaml}"

source "${SCRIPT_DIR}/source_ros2_ws.bash"
exec ros2 launch "${PACKAGE_NAME}" recognition.launch.py "config_file:=${CONFIG_FILE}" "$@"
