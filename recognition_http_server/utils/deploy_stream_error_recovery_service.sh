#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="${ENV_NAME:-yolo-jetson}"
SYSTEMD_SERVICE_NAME="${SYSTEMD_SERVICE_NAME:-gimbal_stream_recovery.service}"
SYSTEMD_SERVICE_PATH="/etc/systemd/system/${SYSTEMD_SERVICE_NAME}"
SERVICE_SCRIPT="${PROJECT_DIR}/recognition_http_server/utils/stream_error_processing.py"
HIKVISION_ENV_PATH="${PROJECT_DIR}/recognition_http_server/hikvision.env"
RUN_USER="${RUN_USER:-root}"
STREAM_NAME="${STREAM_NAME:-main}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-1.0}"
WINDOW_SIZE="${WINDOW_SIZE:-5}"
MIN_ERROR_FRAMES="${MIN_ERROR_FRAMES:-3}"
B_MINUS_G_THRESHOLD="${B_MINUS_G_THRESHOLD:-1.0}"
GREEN_DEFICIT_THRESHOLD="${GREEN_DEFICIT_THRESHOLD:-1.0}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-10.0}"
ENABLE_CAPTURE_CLEANUP="${ENABLE_CAPTURE_CLEANUP:-1}"
CAPTURE_CLEANUP_DIR="${CAPTURE_CLEANUP_DIR:-${PROJECT_DIR}/results/gimbal_capture}"

log() {
  echo "[stream-recovery-deploy] $*" >&2
}

fail() {
  echo "[stream-recovery-deploy] ERROR: $*" >&2
  exit 1
}

find_conda_base() {
  local candidate=""

  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    candidate="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
  elif command -v conda > /dev/null 2>&1; then
    candidate="$(conda info --base 2> /dev/null || true)"
    if [[ -z "${candidate}" ]]; then
      candidate="$(cd "$(dirname "$(command -v conda)")/.." && pwd)"
    fi
  else
    for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/miniconda3"; do
      if [[ -x "${candidate}/bin/conda" ]]; then
        echo "${candidate}"
        return 0
      fi
    done
    return 1
  fi

  [[ -n "${candidate}" && -x "${candidate}/bin/conda" ]] || return 1
  echo "${candidate}"
}

find_conda_sh() {
  local base="$1"
  local conda_sh="${base}/etc/profile.d/conda.sh"
  [[ -f "${conda_sh}" ]] || fail "找不到 conda 初始化脚本: ${conda_sh}"
  echo "${conda_sh}"
}

find_hik_sdk_library_dir() {
  local arch=""
  local path=""

  if [[ -n "${HIK_SDK_LIB_DIR:-}" ]]; then
    [[ -f "${HIK_SDK_LIB_DIR}/libhcnetsdk.so" ]] || fail "HIK_SDK_LIB_DIR 无效: ${HIK_SDK_LIB_DIR}"
    echo "${HIK_SDK_LIB_DIR}"
    return 0
  fi

  arch="$(uname -m)"
  case "${arch}" in
    aarch64 | arm64)
      path="${PROJECT_DIR}/HK_SDK/HK_SDK_arm64_Linux/lib/linux"
      ;;
    x86_64 | amd64)
      for path in \
        "${PROJECT_DIR}/HK_SDK/HK_SDK_x86_Linux/lib/linux" \
        "${PROJECT_DIR}/HK_SDK/HK_SDK_x86_Linux/HCNetSDKV6.1.11.5_build20251204_linux64_ZH/库文件" \
        "${PROJECT_DIR}/HK_SDK_x86_Linux/HCNetSDKV6.1.11.5_build20251204_linux64_ZH/库文件"; do
        if [[ -f "${path}/libhcnetsdk.so" ]]; then
          echo "${path}"
          return 0
        fi
      done
      return 1
      ;;
    *)
      fail "不支持的 CPU 架构: ${arch}"
      ;;
  esac

  [[ -f "${path}/libhcnetsdk.so" ]] || fail "未找到海康 SDK 动态库: ${path}/libhcnetsdk.so"
  echo "${path}"
}

validate_inputs() {
  [[ -f "${SERVICE_SCRIPT}" ]] || fail "未找到异常恢复脚本: ${SERVICE_SCRIPT}"
  [[ -f "${HIKVISION_ENV_PATH}" ]] || fail "未找到海康配置文件: ${HIKVISION_ENV_PATH}"
  [[ "${STREAM_NAME}" == "main" || "${STREAM_NAME}" == "sub" ]] || fail "STREAM_NAME 只能是 main 或 sub"
  [[ "${WINDOW_SIZE}" =~ ^[0-9]+$ ]] || fail "WINDOW_SIZE 必须是正整数"
  [[ "${MIN_ERROR_FRAMES}" =~ ^[0-9]+$ ]] || fail "MIN_ERROR_FRAMES 必须是正整数"
  [[ "${ENABLE_CAPTURE_CLEANUP}" == "0" || "${ENABLE_CAPTURE_CLEANUP}" == "1" ]] || fail "ENABLE_CAPTURE_CLEANUP 只能是 0 或 1"
}

write_systemd_service_file() {
  local service_path="$1"
  local conda_sh="$2"
  local env_prefix="$3"
  local hik_sdk_lib_dir="$4"
  local cleanup_args=""

  if [[ "${ENABLE_CAPTURE_CLEANUP}" == "1" ]]; then
    cleanup_args="--capture-cleanup-dir \"${CAPTURE_CLEANUP_DIR}\""
  else
    cleanup_args="--no-cleanup-captures"
  fi

  sudo tee "${service_path}" > /dev/null << EOF
[Unit]
Description=Gimbal Stream Color Error Recovery
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${PROJECT_DIR}
UMask=0002
Environment=PYTHONNOUSERSITE=1
Environment=PYTHONUNBUFFERED=1
Environment=HIK_SDK_LIB_DIR=${hik_sdk_lib_dir}
Environment=LD_LIBRARY_PATH=${hik_sdk_lib_dir}
EnvironmentFile=${HIKVISION_ENV_PATH}
ExecStart=/bin/bash -lc 'source "${conda_sh}" && conda activate "${env_prefix}" && exec python "${SERVICE_SCRIPT}" --stream "${STREAM_NAME}" --interval "${SAMPLE_INTERVAL}" --window-size "${WINDOW_SIZE}" --min-error-frames "${MIN_ERROR_FRAMES}" --b-minus-g-threshold "${B_MINUS_G_THRESHOLD}" --green-deficit-threshold "${GREEN_DEFICIT_THRESHOLD}" --cooldown-seconds "${COOLDOWN_SECONDS}" ${cleanup_args}'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

  log "已写入 systemd 服务文件: ${service_path}"
}

install_service() {
  local conda_base=""
  local conda_sh=""
  local env_prefix=""
  local hik_sdk_lib_dir=""

  validate_inputs
  conda_base="$(find_conda_base)" || fail "未找到 miniconda/anaconda，请确认 conda 已安装"
  conda_sh="$(find_conda_sh "${conda_base}")"
  env_prefix="${conda_base}/envs/${ENV_NAME}"
  [[ -f "${env_prefix}/bin/python" ]] || fail "未找到 conda 环境: ${env_prefix}"
  hik_sdk_lib_dir="$(find_hik_sdk_library_dir)"

  log "项目目录: ${PROJECT_DIR}"
  log "conda 环境: ${env_prefix}"
  log "海康 SDK 动态库: ${hik_sdk_lib_dir}"
  if [[ "${ENABLE_CAPTURE_CLEANUP}" == "1" ]]; then
    log "恢复成功后自动清理采集图片: ${CAPTURE_CLEANUP_DIR}"
  else
    log "恢复成功后自动清理采集图片: disabled"
  fi
  log "安装 systemd 服务: ${SYSTEMD_SERVICE_NAME}"
  write_systemd_service_file "${SYSTEMD_SERVICE_PATH}" "${conda_sh}" "${env_prefix}" "${hik_sdk_lib_dir}"

  sudo systemctl daemon-reload
  sudo systemctl enable "${SYSTEMD_SERVICE_NAME}"
  sudo systemctl restart "${SYSTEMD_SERVICE_NAME}"
  sudo systemctl status --no-pager "${SYSTEMD_SERVICE_NAME}"
}

uninstall_service() {
  log "停止并卸载 systemd 服务: ${SYSTEMD_SERVICE_NAME}"
  sudo systemctl disable --now "${SYSTEMD_SERVICE_NAME}" || true
  sudo rm -f "${SYSTEMD_SERVICE_PATH}"
  sudo systemctl daemon-reload
  log "已卸载: ${SYSTEMD_SERVICE_NAME}"
}

show_status() {
  sudo systemctl status --no-pager "${SYSTEMD_SERVICE_NAME}"
}

usage() {
  cat << EOF
Usage: $(basename "$0") [install|uninstall|status]

Environment overrides:
  ENV_NAME=${ENV_NAME}
  SYSTEMD_SERVICE_NAME=${SYSTEMD_SERVICE_NAME}
  RUN_USER=${RUN_USER}
  STREAM_NAME=${STREAM_NAME}
  SAMPLE_INTERVAL=${SAMPLE_INTERVAL}
  WINDOW_SIZE=${WINDOW_SIZE}
  MIN_ERROR_FRAMES=${MIN_ERROR_FRAMES}
  B_MINUS_G_THRESHOLD=${B_MINUS_G_THRESHOLD}
  GREEN_DEFICIT_THRESHOLD=${GREEN_DEFICIT_THRESHOLD}
  COOLDOWN_SECONDS=${COOLDOWN_SECONDS}
  ENABLE_CAPTURE_CLEANUP=${ENABLE_CAPTURE_CLEANUP}
  CAPTURE_CLEANUP_DIR=${CAPTURE_CLEANUP_DIR}
  HIK_SDK_LIB_DIR=${HIK_SDK_LIB_DIR:-<auto>}
EOF
}

main() {
  local action="${1:-install}"

  case "${action}" in
    install)
      install_service
      ;;
    uninstall)
      uninstall_service
      ;;
    status)
      show_status
      ;;
    -h | --help | help)
      usage
      ;;
    *)
      usage
      fail "未知动作: ${action}"
      ;;
  esac
}

main "$@"
