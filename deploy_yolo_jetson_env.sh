#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/resources"
ARCHIVE_NAME="yolo-jetson.tar.gz"
ENV_NAME="yolo-jetson"
ARCHIVE_SOURCE_PATH="${RESOURCES_DIR}/${ARCHIVE_NAME}"
DEB_NAME="libcudss0-cuda-12_0.7.1.4-1_arm64.deb"
DEB_SOURCE_PATH="${RESOURCES_DIR}/${DEB_NAME}"
SYSTEMD_SERVICE_NAME="yolo-jetson.service"
SYSTEMD_SERVICE_PATH="/etc/systemd/system/${SYSTEMD_SERVICE_NAME}"

log() {
  echo "[deploy] $*" >&2
}

fail() {
  echo "[deploy] ERROR: $*" >&2
  exit 1
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

  [[ -f "${DEB_SOURCE_PATH}" ]] || fail "未找到安装包: ${DEB_SOURCE_PATH}"
  log "未检测到 libcudss，执行 sudo apt install ${DEB_SOURCE_PATH}"
  sudo apt install -y "${DEB_SOURCE_PATH}"

  lib_dir="$(find_cudss_library_dir)" || fail "libcudss 安装后仍未找到动态库"
  log "libcudss 安装完成: ${lib_dir}"
  echo "${lib_dir}"
}

write_conda_hooks() {
  local env_prefix="$1"
  local lib_dir="$2"
  local activate_dir="${env_prefix}/etc/conda/activate.d"
  local deactivate_dir="${env_prefix}/etc/conda/deactivate.d"
  local activate_hook="${activate_dir}/libcudss.sh"
  local deactivate_hook="${deactivate_dir}/libcudss.sh"

  mkdir -p "${activate_dir}" "${deactivate_dir}"

  cat > "${activate_hook}" <<EOF
#!/usr/bin/env bash
export _YOLO_JETSON_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
case ":\${LD_LIBRARY_PATH:-}:" in
  *:"${lib_dir}":*) ;;
  *) export LD_LIBRARY_PATH="${lib_dir}\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" ;;
esac
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
  local server_script="${working_dir}/server.py"

  [[ -f "${server_script}" ]] || fail "未找到服务入口脚本: ${server_script}"

  sudo tee "${service_path}" >/dev/null <<EOF
[Unit]
Description=AI Detection Server
After=network.target

[Service]
Type=simple
User=${run_user}
WorkingDirectory=${working_dir}
Environment=PYTHONNOUSERSITE=1
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

  log "安装 systemd 服务: ${SYSTEMD_SERVICE_NAME}"
  write_systemd_service_file "${service_path}" "${run_user}" "${working_dir}" "${conda_sh}" "${env_prefix}"

  log "刷新 systemd 配置"
  sudo systemctl daemon-reload

  log "启用开机自启: ${SYSTEMD_SERVICE_NAME}"
  sudo systemctl enable "${SYSTEMD_SERVICE_NAME}"

  log "当前 ${SYSTEMD_SERVICE_NAME} 自启状态"
  sudo systemctl is-enabled "${SYSTEMD_SERVICE_NAME}"
}

main() {
  local conda_base=""
  local conda_sh=""
  local envs_dir=""
  local archive_path=""
  local env_prefix=""
  local cudss_lib_dir=""
  local env_was_unpacked=0
  local run_user="root"

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
  write_conda_hooks "${CONDA_PREFIX}" "${cudss_lib_dir}"

  log "重新激活环境以立即应用新配置"
  set +u
  conda deactivate
  conda activate "${ENV_NAME}"
  set -u

  install_systemd_service "${SYSTEMD_SERVICE_PATH}" "${run_user}" "${SCRIPT_DIR}" "${conda_sh}" "${env_prefix}"

  remove_resources_dir_if_present
  remove_git_dir_if_present

  log "部署完成"
  log "CONDA_PREFIX=${CONDA_PREFIX}"
  log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
}

main "$@"
