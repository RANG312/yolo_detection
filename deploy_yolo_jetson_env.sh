#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/resources"
ARCHIVE_NAME="yolo-jetson.tar.gz"
ENV_NAME="yolo-jetson"
ARCHIVE_SOURCE_PATH="${RESOURCES_DIR}/${ARCHIVE_NAME}"
DEB_NAME="libcudss0-cuda-12_0.7.1.4-1_arm64.deb"
DEB_SOURCE_PATH="${RESOURCES_DIR}/${DEB_NAME}"

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
    log "移动 ${ARCHIVE_SOURCE_PATH} -> ${archive_target_path}"
    mv -f "${ARCHIVE_SOURCE_PATH}" "${archive_target_path}"
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
    return 0
  fi

  mkdir -p "${env_prefix}"
  log "解压环境到 ${env_prefix}"
  tar -xzf "${archive_path}" -C "${env_prefix}"
}

activate_env_cleanly() {
  local conda_sh="$1"
  local env_name="$2"

  # shellcheck disable=SC1090
  source "${conda_sh}"

  while [[ "${CONDA_SHLVL:-0}" -gt 0 ]]; do
    conda deactivate
  done

  log "激活 conda 环境: ${env_name}"
  conda activate "${env_name}"
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

main() {
  local conda_base=""
  local conda_sh=""
  local envs_dir=""
  local archive_path=""
  local env_prefix=""
  local cudss_lib_dir=""

  conda_base="$(find_conda_base)" || fail "未找到 miniconda/anaconda，请确认 conda 已安装"
  conda_sh="$(find_conda_sh "${conda_base}")"
  envs_dir="${conda_base}/envs"
  env_prefix="${envs_dir}/${ENV_NAME}"

  log "检测到 conda 根目录: ${conda_base}"
  archive_path="$(ensure_archive_in_envs "${envs_dir}")"
  unpack_env_if_needed "${archive_path}" "${env_prefix}"

  activate_env_cleanly "${conda_sh}" "${ENV_NAME}"
  run_conda_unpack_if_present

  cudss_lib_dir="$(ensure_cudss_installed)"
  write_conda_hooks "${CONDA_PREFIX}" "${cudss_lib_dir}"

  log "重新激活环境以立即应用新配置"
  conda deactivate
  conda activate "${ENV_NAME}"

  log "部署完成"
  log "CONDA_PREFIX=${CONDA_PREFIX}"
  log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
}

main "$@"
