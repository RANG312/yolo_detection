#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_WORKSPACE_DIR="${HOME}/prj/ros2_ws"

ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"

source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
source "${WORKSPACE_DIR}/install/setup.bash"
