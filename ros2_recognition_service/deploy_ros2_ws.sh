#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PACKAGE_DIR="${REPO_ROOT}/ros2_recognition_service"

ROS_DISTRO_NAME="${ROS_DISTRO_NAME:-humble}"
WORKSPACE_DIR="${WORKSPACE_DIR:-${HOME}/prj/ros2_ws}"
PACKAGE_NAME="ros2_recognition_service"
LINK_MODE="${LINK_MODE:-symlink}"
DEVICE="${DEVICE:-0}"
BUILD_PYTHON="${BUILD_PYTHON:-/usr/bin/python3}"
IMAGE_TOPIC="${IMAGE_TOPIC:-}"
IMAGE_RECOGNIZE_TYPE="${IMAGE_RECOGNIZE_TYPE:-1}"
IMAGE_RECOGNIZE_SUBTYPE="${IMAGE_RECOGNIZE_SUBTYPE:-default}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --workspace PATH              ROS 2 workspace path. Default: ${WORKSPACE_DIR}
  --ros-distro NAME            ROS 2 distro name. Default: ${ROS_DISTRO_NAME}
  --link-mode MODE             'symlink' or 'copy'. Default: ${LINK_MODE}
  --device VALUE               Inference device, e.g. 0 or cpu. Default: ${DEVICE}
  --build-python PATH          Python used only for ROS 2 package build. Default: ${BUILD_PYTHON}
  --image-topic TOPIC          Optional sensor_msgs/msg/Image topic to subscribe.
  --image-recognize-type TYPE  recognize_type for subscribed images. Default: ${IMAGE_RECOGNIZE_TYPE}
  --image-recognize-subtype S  recognize_subtype for subscribed images. Default: ${IMAGE_RECOGNIZE_SUBTYPE}
  -h, --help                   Show this help.

Environment variables with the same names are also supported:
  WORKSPACE_DIR, ROS_DISTRO_NAME, LINK_MODE, DEVICE, BUILD_PYTHON,
  IMAGE_TOPIC, IMAGE_RECOGNIZE_TYPE, IMAGE_RECOGNIZE_SUBTYPE
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace)
      WORKSPACE_DIR="$2"
      shift 2
      ;;
    --ros-distro)
      ROS_DISTRO_NAME="$2"
      shift 2
      ;;
    --link-mode)
      LINK_MODE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --build-python)
      BUILD_PYTHON="$2"
      shift 2
      ;;
    --image-topic)
      IMAGE_TOPIC="$2"
      shift 2
      ;;
    --image-recognize-type)
      IMAGE_RECOGNIZE_TYPE="$2"
      shift 2
      ;;
    --image-recognize-subtype)
      IMAGE_RECOGNIZE_SUBTYPE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]]; then
  echo "ROS 2 setup file not found: /opt/ros/${ROS_DISTRO_NAME}/setup.bash" >&2
  exit 1
fi

if [[ ! -x "${BUILD_PYTHON}" ]]; then
  echo "Build python not found or not executable: ${BUILD_PYTHON}" >&2
  exit 1
fi

if [[ ! -d "${PACKAGE_DIR}" ]]; then
  echo "Package directory not found: ${PACKAGE_DIR}" >&2
  exit 1
fi

if [[ "${LINK_MODE}" != "symlink" && "${LINK_MODE}" != "copy" ]]; then
  echo "Unsupported --link-mode: ${LINK_MODE}" >&2
  exit 2
fi

MODEL_PATH="${REPO_ROOT}/runs/train/meter_data_9k_yolov8m_best/weights/best.pt"
FIRE_MODEL_PATH="${REPO_ROOT}/runs/train/best_fire.pt"
SAFEHAT_MODEL_PATH="${REPO_ROOT}/runs/train/best_person.pt"
RESULT_ROOT="${REPO_ROOT}/results/http_service"

for path in "${MODEL_PATH}" "${FIRE_MODEL_PATH}" "${SAFEHAT_MODEL_PATH}"; do
  if [[ ! -f "${path}" ]]; then
    echo "Required model file not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${WORKSPACE_DIR}/src"
TARGET_PACKAGE_PATH="${WORKSPACE_DIR}/src/${PACKAGE_NAME}"

if [[ -L "${TARGET_PACKAGE_PATH}" || -d "${TARGET_PACKAGE_PATH}" ]]; then
  rm -rf "${TARGET_PACKAGE_PATH}"
fi

if [[ "${LINK_MODE}" == "symlink" ]]; then
  ln -s "${PACKAGE_DIR}" "${TARGET_PACKAGE_PATH}"
else
  cp -a "${PACKAGE_DIR}" "${TARGET_PACKAGE_PATH}"
fi

mkdir -p "${WORKSPACE_DIR}/config"
WORKSPACE_CONFIG_FILE="${WORKSPACE_DIR}/config/recognition.yaml"
WORKSPACE_SOURCE_SCRIPT="${WORKSPACE_DIR}/source_ros2_ws.bash"
WORKSPACE_LAUNCH_SCRIPT="${WORKSPACE_DIR}/launch_recognition_service.sh"

cat > "${WORKSPACE_CONFIG_FILE}" <<EOF
repo_root: ${REPO_ROOT}
node_name: recognition_service_node

model: ${MODEL_PATH}
fire_model: ${FIRE_MODEL_PATH}
safehat_model: ${SAFEHAT_MODEL_PATH}

imgsz: 640
conf: 0.25
device: "${DEVICE}"
min_value: 0.0
max_value: 1.0

result_root: ${RESULT_ROOT}
callback_port: 18080
request_timeout: 15

callback_topic: /api/v1/recognition/callback
submit_service: /api/v1/recognition/tasks/submit
status_service: /api/v1/recognition/tasks/get

image_topic: "${IMAGE_TOPIC}"
image_qos_depth: 10
image_recognize_type: "${IMAGE_RECOGNIZE_TYPE}"
image_recognize_subtype: "${IMAGE_RECOGNIZE_SUBTYPE}"
EOF

cat > "${WORKSPACE_SOURCE_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -e
source /opt/ros/${ROS_DISTRO_NAME}/setup.bash
source ${WORKSPACE_DIR}/install/setup.bash
EOF
chmod +x "${WORKSPACE_SOURCE_SCRIPT}"

cat > "${WORKSPACE_LAUNCH_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -e
source /opt/ros/${ROS_DISTRO_NAME}/setup.bash
source ${WORKSPACE_DIR}/install/setup.bash
exec ros2 launch ${PACKAGE_NAME} recognition.launch.py config_file:=${WORKSPACE_CONFIG_FILE} "\$@"
EOF
chmod +x "${WORKSPACE_LAUNCH_SCRIPT}"

echo "[1/3] Sourcing ROS 2 environment: /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set +u
source "/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
set -u

echo "[2/4] Cleaning old build cache for ${PACKAGE_NAME}"
cd "${WORKSPACE_DIR}"
rm -rf "build/${PACKAGE_NAME}" "install/${PACKAGE_NAME}" log

echo "[3/4] Building workspace: ${WORKSPACE_DIR}"
echo "      Using build python: ${BUILD_PYTHON}"
colcon build --packages-select "${PACKAGE_NAME}" --cmake-args "-DPython3_EXECUTABLE=${BUILD_PYTHON}"

echo "[4/4] Deployment completed"
echo
echo "Workspace: ${WORKSPACE_DIR}"
echo "Package source: ${TARGET_PACKAGE_PATH}"
echo "Launch config: ${WORKSPACE_CONFIG_FILE}"
echo "Repo source script: ${PACKAGE_DIR}/source_ros2_ws.bash"
echo "Repo launch script: ${PACKAGE_DIR}/launch_recognition_service.sh"
echo "Workspace source script: ${WORKSPACE_SOURCE_SCRIPT}"
echo "Workspace launch script: ${WORKSPACE_LAUNCH_SCRIPT}"
echo "Build python: ${BUILD_PYTHON}"
echo
echo "Next commands:"
echo "  source ${WORKSPACE_SOURCE_SCRIPT}"
echo "  ${WORKSPACE_LAUNCH_SCRIPT}"
echo
echo "Equivalent manual launch command:"
echo "  source /opt/ros/${ROS_DISTRO_NAME}/setup.bash"
echo "  source ${WORKSPACE_DIR}/install/setup.bash"
echo "  ros2 launch ${PACKAGE_NAME} recognition.launch.py config_file:=${WORKSPACE_CONFIG_FILE}"
