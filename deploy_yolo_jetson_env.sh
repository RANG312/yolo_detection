#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/resources"
ARCHIVE_NAME="yolo-jetson.tar.gz"
ENV_NAME="yolo-jetson"
ARCHIVE_SOURCE_PATH="${RESOURCES_DIR}/${ARCHIVE_NAME}"
CUDSS_DEB_NAME="libcudss0-cuda-12_0.7.1.4-1_arm64.deb"
CUDSS_DEB_SOURCE_PATH="${RESOURCES_DIR}/${CUDSS_DEB_NAME}"
CUPTI_DEB_NAME="cuda-cupti-12-6_12.6.68-1_arm64.deb"
CUPTI_DEB_SOURCE_PATH="${RESOURCES_DIR}/${CUPTI_DEB_NAME}"
HIKVISION_ENV_PATH="${SCRIPT_DIR}/recognition_http_server/hikvision.env"
PROTECTED_BUILD_SCRIPT="${SCRIPT_DIR}/scripts/build_protected.py"
SYSTEMD_SERVICE_NAME="ai_detection.service"
SYSTEMD_SERVICE_PATH="/etc/systemd/system/${SYSTEMD_SERVICE_NAME}"

log() {
  echo "[deploy] $*" >&2
}

fail() {
  echo "[deploy] ERROR: $*" >&2
  exit 1
}

env_file_get() {
  local key="$1"
  local value=""

  [[ -f "${HIKVISION_ENV_PATH}" ]] || return 1
  value="$(awk -F= -v key="${key}" '$1 == key {print substr($0, index($0, "=") + 1); exit}' "${HIKVISION_ENV_PATH}")"
  [[ -n "${value}" ]] || return 1
  value="${value%\"}"
  value="${value#\"}"
  printf '%s' "${value}"
}

prompt_with_default() {
  local label="$1"
  local default_value="$2"
  local value=""

  if [[ -n "${default_value}" ]]; then
    read -r -p "${label} [${default_value}]: " value
    printf '%s' "${value:-${default_value}}"
  else
    read -r -p "${label}: " value
    printf '%s' "${value}"
  fi
}

prompt_secret_with_default() {
  local label="$1"
  local default_value="$2"
  local value=""
  local prompt="${label}"

  if [[ -n "${default_value}" ]]; then
    prompt="${prompt} [保留已有值请直接回车]"
  fi
  read -r -s -p "${prompt}: " value
  printf '\n' >&2
  printf '%s' "${value:-${default_value}}"
}

systemd_env_escape() {
  local value="$1"

  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '%s' "${value}"
}

write_hikvision_env_file() {
  local host="$1"
  local username="$2"
  local password="$3"
  local port="$4"
  local channel="$5"
  local local_ip="$6"

  mkdir -p "$(dirname "${HIKVISION_ENV_PATH}")"
  cat > "${HIKVISION_ENV_PATH}" <<EOF
HIK_HOST="$(systemd_env_escape "${host}")"
HIK_USERNAME="$(systemd_env_escape "${username}")"
HIK_PASSWORD="$(systemd_env_escape "${password}")"
HIK_PORT="$(systemd_env_escape "${port}")"
HIK_CHANNEL="$(systemd_env_escape "${channel}")"
HIK_LOCAL_IP="$(systemd_env_escape "${local_ip}")"
EOF
  chmod 600 "${HIKVISION_ENV_PATH}"
  log "已写入海康云台配置: ${HIKVISION_ENV_PATH}"
}

prompt_hikvision_config() {
  local default_host="${HIK_HOST:-}"
  local default_username="${HIK_USERNAME:-admin}"
  local default_password="${HIK_PASSWORD:-}"
  local default_port="${HIK_PORT:-8000}"
  local default_channel="${HIK_CHANNEL:-1}"
  local default_local_ip="${HIK_LOCAL_IP:-}"
  local host=""
  local username=""
  local password=""
  local port=""
  local channel=""
  local local_ip=""

  default_host="$(env_file_get HIK_HOST || printf '%s' "${default_host}")"
  default_username="$(env_file_get HIK_USERNAME || printf '%s' "${default_username}")"
  default_password="$(env_file_get HIK_PASSWORD || printf '%s' "${default_password}")"
  default_port="$(env_file_get HIK_PORT || printf '%s' "${default_port}")"
  default_channel="$(env_file_get HIK_CHANNEL || printf '%s' "${default_channel}")"
  default_local_ip="$(env_file_get HIK_LOCAL_IP || printf '%s' "${default_local_ip}")"

  log "请输入海康云台连接配置；直接回车使用方括号中的默认值"
  host="$(prompt_with_default "海康云台 IP/Host" "${default_host}")"
  username="$(prompt_with_default "海康云台用户名" "${default_username}")"
  password="$(prompt_secret_with_default "海康云台密码" "${default_password}")"
  port="$(prompt_with_default "海康 SDK 端口" "${default_port}")"
  channel="$(prompt_with_default "海康通道号" "${default_channel}")"
  local_ip="$(prompt_with_default "本机绑定 IP，可留空" "${default_local_ip}")"

  [[ -n "${host}" ]] || fail "海康云台 IP/Host 不能为空"
  [[ -n "${username}" ]] || fail "海康云台用户名不能为空"
  [[ -n "${password}" ]] || fail "海康云台密码不能为空"
  [[ "${port}" =~ ^[0-9]+$ ]] || fail "海康 SDK 端口必须是数字: ${port}"
  [[ "${channel}" =~ ^[0-9]+$ ]] || fail "海康通道号必须是数字: ${channel}"

  write_hikvision_env_file "${host}" "${username}" "${password}" "${port}" "${channel}" "${local_ip}"
}

find_conda_base() {
  local candidate=""

  if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
    candidate="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
  elif command -v conda >/dev/null 2>&1; then
    candidate="$(conda info --base 2>/dev/null || true)"
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

ensure_archive_in_envs() {
  local envs_dir="$1"
  local archive_target_path="${envs_dir}/${ARCHIVE_NAME}"

  mkdir -p "${envs_dir}"

  if [[ -f "${ARCHIVE_SOURCE_PATH}" ]]; then
    if [[ "${ARCHIVE_SOURCE_PATH}" -ef "${archive_target_path}" ]]; then
      log "复用已存在压缩包: ${archive_target_path}"
    else
      log "复制 ${ARCHIVE_SOURCE_PATH} -> ${archive_target_path}"
      cp -f "${ARCHIVE_SOURCE_PATH}" "${archive_target_path}"
    fi
  elif [[ -f "${archive_target_path}" ]]; then
    log "复用已存在压缩包: ${archive_target_path}"
  else
    fail "未找到环境压缩包: ${ARCHIVE_SOURCE_PATH} 或 ${archive_target_path}"
  fi

  echo "${archive_target_path}"
}

unpack_env_if_needed() {
  local archive_path="$1"
  local env_prefix="$2"

  if [[ -d "${env_prefix}" && -f "${env_prefix}/bin/python" ]]; then
    log "检测到已存在环境，跳过解压: ${env_prefix}"
    return 1
  fi

  mkdir -p "${env_prefix}"
  log "解压环境到 ${env_prefix}"
  tar -xzf "${archive_path}" -C "${env_prefix}"
  return 0
}

activate_env_cleanly() {
  local conda_sh="$1"
  local env_name="$2"

  # Some packaged env activation hooks source third-party scripts that are
  # not nounset-safe, so relax `set -u` only around conda shell operations.
  local had_nounset=0
  if [[ $- == *u* ]]; then
    had_nounset=1
    set +u
  fi

  # shellcheck disable=SC1090
  source "${conda_sh}"

  while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate
  done

  log "激活 conda 环境: ${env_name}"
  conda activate "${env_name}"

  if [[ "${had_nounset}" -eq 1 ]]; then
    set -u
  fi
}

run_conda_unpack_if_present() {
  if command -v conda-unpack >/dev/null 2>&1; then
    log "执行 conda-unpack 修正 prefix"
    conda-unpack
  else
    log "当前环境没有 conda-unpack，跳过"
  fi
}

find_cudss_library_dir() {
  local path=""

  for path in \
    /usr/lib/aarch64-linux-gnu \
    /usr/lib/aarch64-linux-gnu/12 \
    /usr/lib/aarch64-linux-gnu/libcudss \
    /usr/lib/aarch64-linux-gnu/libcudss/12 \
    /usr/local/cuda/lib64 \
    /usr/local/cuda/targets/aarch64-linux/lib; do
    if compgen -G "${path}/libcudss.so*" >/dev/null; then
      echo "${path}"
      return 0
    fi
  done

  path="$(ldconfig -p 2>/dev/null | awk '/libcudss\.so/ {print $NF; exit}')"
  if [[ -n "${path}" ]]; then
    dirname "${path}"
    return 0
  fi

  return 1
}

ensure_cudss_installed() {
  local lib_dir=""

  if lib_dir="$(find_cudss_library_dir)"; then
    log "已检测到 libcudss: ${lib_dir}"
    echo "${lib_dir}"
    return 0
  fi

  [[ -f "${CUDSS_DEB_SOURCE_PATH}" ]] || fail "未找到安装包: ${CUDSS_DEB_SOURCE_PATH}"
  log "未检测到 libcudss，执行 sudo apt install ${CUDSS_DEB_SOURCE_PATH}"
  sudo apt install -y "${CUDSS_DEB_SOURCE_PATH}" >&2

  lib_dir="$(find_cudss_library_dir)" || fail "libcudss 安装后仍未找到动态库"
  log "libcudss 安装完成: ${lib_dir}"
  echo "${lib_dir}"
}

find_cupti_library_dir() {
  local path=""

  for path in \
    /usr/local/cuda/targets/aarch64-linux/lib \
    /usr/local/cuda/lib64 \
    /usr/local/cuda/extras/CUPTI/lib64 \
    /usr/local/cuda-12/targets/aarch64-linux/lib \
    /usr/local/cuda-12/extras/CUPTI/lib64 \
    /usr/local/cuda-12.6/targets/aarch64-linux/lib \
    /usr/local/cuda-12.6/extras/CUPTI/lib64 \
    /usr/lib/aarch64-linux-gnu; do
    if compgen -G "${path}/libcupti.so*" >/dev/null; then
      echo "${path}"
      return 0
    fi
  done

  path="$(ldconfig -p 2>/dev/null | awk '/libcupti\.so/ {print $NF; exit}')"
  if [[ -n "${path}" ]]; then
    dirname "${path}"
    return 0
  fi

  return 1
}

ensure_cupti_installed() {
  local lib_dir=""

  if lib_dir="$(find_cupti_library_dir)"; then
    log "已检测到 libcupti: ${lib_dir}"
    echo "${lib_dir}"
    return 0
  fi

  [[ -f "${CUPTI_DEB_SOURCE_PATH}" ]] || fail "未找到安装包: ${CUPTI_DEB_SOURCE_PATH}"
  log "未检测到 libcupti，执行 sudo apt install ${CUPTI_DEB_SOURCE_PATH}"
  sudo apt install -y "${CUPTI_DEB_SOURCE_PATH}" >&2

  lib_dir="$(find_cupti_library_dir)" || fail "libcupti 安装后仍未找到动态库"
  log "libcupti 安装完成: ${lib_dir}"
  echo "${lib_dir}"
}

find_hik_sdk_library_dir() {
  local arch=""
  local path=""

  arch="$(uname -m)"
  case "${arch}" in
    aarch64|arm64)
      path="${SCRIPT_DIR}/HK_SDK/HK_SDK_arm64_Linux/lib/linux"
      ;;
    x86_64|amd64)
      for path in \
        "${SCRIPT_DIR}/HK_SDK/HK_SDK_x86_Linux/lib/linux" \
        "${SCRIPT_DIR}/HK_SDK/HK_SDK_x86_Linux/HCNetSDKV6.1.11.5_build20251204_linux64_ZH/库文件" \
        "${SCRIPT_DIR}/HK_SDK_x86_Linux/HCNetSDKV6.1.11.5_build20251204_linux64_ZH/库文件"; do
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

write_conda_hooks() {
  local env_prefix="$1"
  local cudss_lib_dir="$2"
  local cupti_lib_dir="$3"
  local activate_dir="${env_prefix}/etc/conda/activate.d"
  local deactivate_dir="${env_prefix}/etc/conda/deactivate.d"
  local activate_hook="${activate_dir}/cuda-runtime-libs.sh"
  local deactivate_hook="${deactivate_dir}/cuda-runtime-libs.sh"

  [[ -d "${cudss_lib_dir}" ]] || fail "Invalid libcudss path: ${cudss_lib_dir}"
  [[ -d "${cupti_lib_dir}" ]] || fail "Invalid libcupti path: ${cupti_lib_dir}"

  mkdir -p "${activate_dir}" "${deactivate_dir}"
  rm -f "${activate_dir}/libcudss.sh" "${deactivate_dir}/libcudss.sh"

  cat > "${activate_hook}" <<EOF
#!/usr/bin/env bash
export _YOLO_JETSON_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
for _YOLO_JETSON_LIB_DIR in \\
  "${cudss_lib_dir}" \\
  "${cupti_lib_dir}"; do
  case ":\${LD_LIBRARY_PATH:-}:" in
    *:"\${_YOLO_JETSON_LIB_DIR}":*) ;;
    *) export LD_LIBRARY_PATH="\${_YOLO_JETSON_LIB_DIR}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" ;;
  esac
done
unset _YOLO_JETSON_LIB_DIR
EOF

  cat > "${deactivate_hook}" <<'EOF'
#!/usr/bin/env bash
if [[ -n "${_YOLO_JETSON_OLD_LD_LIBRARY_PATH+x}" ]]; then
  export LD_LIBRARY_PATH="${_YOLO_JETSON_OLD_LD_LIBRARY_PATH}"
  unset _YOLO_JETSON_OLD_LD_LIBRARY_PATH
else
  unset LD_LIBRARY_PATH
fi
EOF

  chmod +x "${activate_hook}" "${deactivate_hook}"
  log "已写入 conda 长期配置: ${activate_hook}"
}

remove_resources_dir_if_present() {
  if [[ -d "${RESOURCES_DIR}" ]]; then
    log "删除部署资源目录: ${RESOURCES_DIR}"
    rm -rf "${RESOURCES_DIR}"
  else
    log "未找到资源目录，跳过删除: ${RESOURCES_DIR}"
  fi
}

remove_git_dir_if_present() {
  local git_dir="${SCRIPT_DIR}/.git"

  if [[ -d "${git_dir}" ]]; then
    log "删除 Git 元数据目录: ${git_dir}"
    rm -rf "${git_dir}"
  else
    log "未找到 Git 元数据目录，跳过删除: ${git_dir}"
  fi
}

write_systemd_service_file() {
  local service_path="$1"
  local run_user="$2"
  local working_dir="$3"
  local conda_sh="$4"
  local env_prefix="$5"
  local hik_sdk_lib_dir="$6"
  local server_script="${working_dir}/server.py"

  [[ -f "${server_script}" ]] || fail "未找到服务入口脚本: ${server_script}"
  [[ -d "${hik_sdk_lib_dir}" ]] || fail "Invalid Hikvision SDK path: ${hik_sdk_lib_dir}"

  sudo tee "${service_path}" >/dev/null <<EOF
[Unit]
Description=AI Detection Server
After=network.target

[Service]
Type=simple
User=${run_user}
WorkingDirectory=${working_dir}
UMask=0002
Environment=PYTHONNOUSERSITE=1
Environment=HIK_SDK_LIB_DIR=${hik_sdk_lib_dir}
Environment=LD_LIBRARY_PATH=${hik_sdk_lib_dir}
EnvironmentFile=${working_dir}/recognition_http_server/hikvision.env
ExecStart=/bin/bash -lc 'source "${conda_sh}" && conda activate "${env_prefix}" && exec python "${server_script}"'
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

  log "已写入 systemd 服务模板: ${service_path}"
}

install_systemd_service() {
  local service_path="$1"
  local run_user="$2"
  local working_dir="$3"
  local conda_sh="$4"
  local env_prefix="$5"
  local hik_sdk_lib_dir="$6"

  log "安装 systemd 服务: ${SYSTEMD_SERVICE_NAME}"
  write_systemd_service_file "${service_path}" "${run_user}" "${working_dir}" "${conda_sh}" "${env_prefix}" "${hik_sdk_lib_dir}"

  log "刷新 systemd 配置"
  sudo systemctl daemon-reload

  log "启用开机自启: ${SYSTEMD_SERVICE_NAME}"
  sudo systemctl enable "${SYSTEMD_SERVICE_NAME}"

  log "重启服务: ${SYSTEMD_SERVICE_NAME}"
  sudo systemctl restart "${SYSTEMD_SERVICE_NAME}"

  log "当前 ${SYSTEMD_SERVICE_NAME} 服务状态"
  sudo systemctl status --no-pager "${SYSTEMD_SERVICE_NAME}"
}

ensure_runtime_permissions() {
  local access_user="$1"
  local results_dir="${SCRIPT_DIR}/results"

  mkdir -p "${results_dir}"
  sudo chgrp -R "${access_user}" "${results_dir}"
  sudo chmod -R u+rwX,g+rwX,o+rX "${results_dir}"
  sudo find "${results_dir}" -type d -exec chmod g+s {} +
  log "已修正运行输出目录权限: ${results_dir} -> group ${access_user}, g+rwX, setgid"
}

ensure_cython_installed() {
  if python -c 'import Cython' >/dev/null 2>&1; then
    log "已检测到 Cython"
    return
  fi

  log "当前环境未安装 Cython，尝试通过 pip 安装"
  python -m pip install Cython || fail "Cython 安装失败，请检查网络或将 Cython 预装到 ${ENV_NAME} 环境"
}

build_protected_modules() {
  local enabled="${BUILD_PROTECTED:-1}"

  if [[ "${enabled}" == "0" ]]; then
    log "BUILD_PROTECTED=0，跳过核心源码编译"
    return
  fi

  [[ "${enabled}" == "1" ]] || fail "BUILD_PROTECTED 只支持 0 或 1: ${enabled}"
  [[ -f "${PROTECTED_BUILD_SCRIPT}" ]] || fail "未找到核心源码编译脚本: ${PROTECTED_BUILD_SCRIPT}"

  ensure_cython_installed
  log "编译核心源码为 Python 扩展模块"
  python "${PROTECTED_BUILD_SCRIPT}" build_ext --inplace
  python - <<'PY'
from pathlib import Path

import recognition_http_server.dial_reading.pipeline as pipeline
import recognition_http_server.service as service

modules = (service, pipeline)
for module in modules:
    path = Path(module.__file__)
    if path.suffix != ".so":
        raise SystemExit(f"Protected module did not load from .so: {module.__name__} -> {path}")
    print(f"[deploy] 已验证编译模块: {module.__name__} -> {path}", flush=True)
PY
}

main() {
  local conda_base=""
  local conda_sh=""
  local envs_dir=""
  local archive_path=""
  local env_prefix=""
  local cudss_lib_dir=""
  local cupti_lib_dir=""
  local hik_sdk_lib_dir=""
  local env_was_unpacked=0
  local run_user="root"
  local access_user=""

  access_user="$(stat -c '%U' "${SCRIPT_DIR}")"
  [[ -n "${access_user}" ]] || fail "无法检测目录属主: ${SCRIPT_DIR}"

  conda_base="$(find_conda_base)" || fail "未找到 miniconda/anaconda，请确认 conda 已安装"
  conda_sh="$(find_conda_sh "${conda_base}")"
  envs_dir="${conda_base}/envs"
  env_prefix="${envs_dir}/${ENV_NAME}"

  log "检测到 conda 根目录: ${conda_base}"
  archive_path="$(ensure_archive_in_envs "${envs_dir}")"
  if unpack_env_if_needed "${archive_path}" "${env_prefix}"; then
    env_was_unpacked=1
  fi

  activate_env_cleanly "${conda_sh}" "${ENV_NAME}"
  if [[ "${env_was_unpacked}" -eq 1 ]]; then
    run_conda_unpack_if_present
  else
    log "环境已存在，跳过 conda-unpack"
  fi

  cudss_lib_dir="$(ensure_cudss_installed)"
  [[ -d "${cudss_lib_dir}" ]] || fail "Invalid libcudss path: ${cudss_lib_dir}"
  cupti_lib_dir="$(ensure_cupti_installed)"
  [[ -d "${cupti_lib_dir}" ]] || fail "Invalid libcupti path: ${cupti_lib_dir}"
  write_conda_hooks "${CONDA_PREFIX}" "${cudss_lib_dir}" "${cupti_lib_dir}"

  hik_sdk_lib_dir="$(find_hik_sdk_library_dir)"
  log "检测到海康 SDK 动态库目录: ${hik_sdk_lib_dir}"
  prompt_hikvision_config

  log "重新激活环境以立即应用新配置"
  set +u
  conda deactivate
  conda activate "${ENV_NAME}"
  set -u

  build_protected_modules
  ensure_runtime_permissions "${access_user}"
  install_systemd_service "${SYSTEMD_SERVICE_PATH}" "${run_user}" "${SCRIPT_DIR}" "${conda_sh}" "${env_prefix}" "${hik_sdk_lib_dir}"

  remove_resources_dir_if_present
  remove_git_dir_if_present

  log "部署完成"
  log "CONDA_PREFIX=${CONDA_PREFIX}"
  log "CUDSS_LIB_DIR=${cudss_lib_dir}"
  log "CUPTI_LIB_DIR=${cupti_lib_dir}"
  log "HIK_SDK_LIB_DIR=${hik_sdk_lib_dir}"
  log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
}

main "$@"
