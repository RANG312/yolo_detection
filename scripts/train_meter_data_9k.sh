#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data/prj/yolov8_dial_reading/ultralytics"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="meter_data_9k_yolov8m_${RUN_TS}"

python "${ROOT_DIR}/scripts/train_dial_thermometer.py" \
  --data "${ROOT_DIR}/data/new_datas/meter_data_9k/meter_data_9k.yaml" \
  --batch 16 --workers 8 \
  --cache disk \
  --amp True \
  --patience 5 \
  --optimizer auto \
  --deterministic False \
  --project "${ROOT_DIR}/runs/train" \
  --name "${RUN_NAME}"
