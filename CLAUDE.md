# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO-based dial-gauge reading, fire detection, and safety-helmet detection service targeting NVIDIA Jetson Orin NX. Extends Ultralytics YOLO with custom recognition pipelines. Serves results via HTTP API and ROS2.

## Commands

```bash
# Development install
pip install -e ".[dev]"

# Start HTTP recognition service
python server.py --host 0.0.0.0 --port 3208
# or equivalently:
python recognition_http_server/app.py

# Run single-image dial reading (CLI)
python dial_reading.py --image path/to/image.jpg --device cpu

# Run integration test client (requires running service)
python test_http.py --server http://127.0.0.1:3208 --images path/to/a.jpg --data-type 1:25

# Run unit tests
pytest

# Export model to ONNX
python scripts/export_best_to_onnx.py

# Jetson deployment (auto-deploy conda env + systemd service)
bash deploy_yolo_jetson_env.sh

# ROS2 — build workspace, source it, launch
bash ros2_recognition_service/deploy_ros2_ws.sh
source ./ros2_recognition_service/source_ros2_ws.bash
bash ros2_recognition_service/launch_recognition_service.sh
```

## Architecture

The service supports 3 recognition tasks via a handler registry pattern (no `if/elif` chains for task dispatch):

| Task | `recognize_type` | Internal kind | Handler |
|------|------------------|---------------|---------|
| Dial meter reading | `"1"` | `meter` | `handlers/meter.py` → dispatches to pointer or digital |
| Fire detection | `"6"` | `fire` | `handlers/detection.py` (generic YOLO) |
| Safety helmet | `"7"` | `safehat` | `handlers/detection.py` (generic YOLO) |

### Key modules

- **`recognition_http_server/service.py`** — `RecognitionService`: model loading, task lifecycle (create → background thread → callback). Task handlers are registered in `_build_task_handlers()`.
- **`recognition_http_server/http_api.py`** — Flask HTTP layer (`POST /api/v1/recognition/tasks`, `GET .../tasks/{req_id}`, `GET /health`).
- **`recognition_http_server/dial_reading/pipeline.py`** — Pointer meter inference: YOLO detection → gauge crop retry (for `meter_data_9k` models) → instance assignment → geometric reading computation.
- **`recognition_http_server/helpers.py`** — Request parsing, task-type routing, callback URL construction, result formatting.
- **`recognition_http_server/constants.py`** — All default paths, ports, thresholds, and task type mappings. Update config here, not in handlers.

### Dial reading flow (pointer meter)

1. YOLO detects gauge box on full image
2. Crop gauge region and resize to `METER_DATA_9K_GAUGE_RETRY_SIZE` for second inference
3. Second inference detects center, ticks, pointer tip within the cropped ROI
4. `geometry.py` computes center estimation, arc ratio, and final reading
5. `visualization.py` draws annotated result image

Two annotation protocols are supported: `meter_data_9k` (two-stage with gauge retry) and `legacy` (single-stage). The pipeline auto-detects which to use via `infer_annotation_mode()`.

### Compatibility layers

- `server.py` — re-exports everything from `recognition_http_server` so ROS2 and old scripts don't break. Do not add new logic here.
- `dial_reading.py` — forwards to `recognition_http_server/dial_reading/` for CLI and legacy imports. Do not add new logic here.

### Image I/O

- Local paths are read directly; HTTP URLs are downloaded to `results/http_service/inputs/<req_id>/`.
- Results saved to `results/http_service/outputs/<req_id>/`.
- For local input paths, a copy of the result image is also saved alongside the original (with `-detection` suffix).

## Coding conventions

- Python 3.8+, 4-space indent, 120-char lines, Google-style docstrings
- Linting/formatting: `ruff`, `isort`, `yapf`, `docformatter` (configured in `pyproject.toml`)
- `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_CASE` for constants
- Task dispatch uses a handler dictionary; add new task types by registering a new `TaskHandler` in `RecognitionService._build_task_handlers()`
- Single-meter failure must not fail other meters on the same image

## Testing

- Test files: `test_*.py` in `tests/`; test functions: `test_*`
- `test_http.py` is the end-to-end integration client (starts a local callback server, submits tasks, validates responses)
- `pytest` runs unit tests with `--doctest-modules` per `pyproject.toml`
- Model weights are gitignored except for the 3 deployment models kept in `runs/train/`

## Jetson-specific notes

- Production target is Jetson Orin NX
- `resources/` holds conda archive and Jetson packages for deployment
- `deploy_yolo_jetson_env.sh` sets up the conda environment and a systemd auto-start service
- Service logs: `results/http_service/logs/server.log`
