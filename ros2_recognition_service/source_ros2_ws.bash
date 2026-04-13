#!/usr/bin/env bash
set -e

DEFAULT_WORKSPACE_DIR="${HOME}/prj/ros2_ws"
DEFAULT_CONDA_SH="${HOME}/miniconda3/etc/profile.d/conda.sh"

ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${DEFAULT_WORKSPACE_DIR}}"
USE_CONDA_ENV="${USE_CONDA_ENV:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-yolo-jetson}"
CONDA_SH_PATH="${CONDA_SH_PATH:-${DEFAULT_CONDA_SH}}"

if [[ "${USE_CONDA_ENV}" == "1" ]]; then
  if [[ ! -f "${CONDA_SH_PATH}" ]]; then
    echo "Conda init script not found: ${CONDA_SH_PATH}" >&2
    echo "Set CONDA_SH_PATH or disable conda activation with USE_CONDA_ENV=0" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONDA_SH_PATH}"
  conda activate "${CONDA_ENV_NAME}"
fi

source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
source "${WORKSPACE_DIR}/install/setup.bash"
