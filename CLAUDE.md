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

# PTZ camera frame capture with RGB statistics
python recognition_http_server/utils/capture_gimbal_frames.py \
  --source rtsp://... --interval 1.0 --count 60

# Stream error recovery (IR-filter-stuck detection + day/night mode reset)
python recognition_http_server/utils/stream_error_processing.py \
  --stream main --interval 1.0

# Deploy stream error recovery as a systemd service (Jetson)
bash recognition_http_server/utils/deploy_stream_error_recovery_service.sh install
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
- **`recognition_http_server/hikvision_ptz.py`** — `HikvisionPTZController`: HCNetSDK wrapper for Hikvision PTZ cameras. Persistent SDK session, absolute/relative pan-tilt-zoom control, GIS field-of-view reading, snapshot capture via ISAPI HTTP. Cross-platform SDK path resolution (x86_64 + arm64).
- **`recognition_http_server/ptz_alignment.py`** — PTZ alignment logic: maps gauge ROI center offset to pan/tilt deltas from FOV, computes zoom requests from gauge height ratio, auto-tunes nudge parameters from FOV. Data classes: `PTZAlignmentConfig`, `PTZAlignmentRequest`, `PTZZoomRequest`, `PTZZoomLimits`.

### Dial reading flow (pointer meter)

1. YOLO detects gauge box on full image
2. Crop gauge region and resize to `METER_DATA_9K_GAUGE_RETRY_SIZE` for second inference
3. Second inference detects center, ticks, pointer tip within the cropped ROI
4. `geometry.py` computes center estimation, arc ratio, and final reading
5. `visualization.py` draws annotated result image

Two annotation protocols are supported: `meter_data_9k` (two-stage with gauge retry) and `legacy` (single-stage). The pipeline auto-detects which to use via `infer_annotation_mode()`.

When PTZ alignment is enabled, the pipeline calls `maybe_align_gauge()` / `maybe_zoom_gauge()` between stages 1 and 2 to center and zoom the camera on the gauge before the second inference pass.

### Utility modules

- **`recognition_http_server/utils/image_io.py`** — `prepare_image()` for image loading.
- **`recognition_http_server/utils/capture_gimbal_frames.py`** — CLI tool for timed snapshot capture from any OpenCV-compatible stream (RTSP/HTTP/device). Computes per-frame RGB statistics (mean, std, green deficit, magenta percentage, chroma) with optional ROI. Outputs timestamped JPEGs + manifest CSV.
- **`recognition_http_server/utils/stream_error_processing.py`** — Detects IR-filter-stuck magenta color cast on Hikvision streams via sliding-window B−G gap and green_deficit analysis. Automatically resets camera day/night mode (night → day → auto) via HCNetSDK to recover correct color. Optional captured-image cleanup after recovery. Configurable via CLI or as a long-running monitor.
- **`recognition_http_server/utils/deploy_stream_error_recovery_service.sh`** — Deploys `stream_error_processing.py` as a systemd service on Jetson. Auto-detects conda environment and Hikvision SDK paths. Supports `install`, `uninstall`, `status` actions.

### Compatibility layers

- `server.py` — re-exports everything from `recognition_http_server` so ROS2 and old scripts don't break. Do not add new logic here.
- `dial_reading.py` — forwards to `recognition_http_server/dial_reading/` for CLI and legacy imports. Do not add new logic here.

### Image I/O

- Local paths are read directly; HTTP URLs are downloaded to `results/http_service/inputs/<req_id>/`.
- Results saved to `results/http_service/outputs/<req_id>/`.
- For local input paths, a copy of the result image is also saved alongside the original (with `-detection` suffix).

## Hikvision SDK integration

### SDK layout

```
HK_SDK/
  HK_SDK_x86_Linux/    # x86_64 Linux SDK
  HK_SDK_arm64_Linux/  # Jetson (aarch64) Linux SDK
```

The SDK lib directory is resolved automatically from CPU architecture. Override with env var `HIK_SDK_LIB_DIR` or pass `resolve_sdk_lib_dir(override=...)`. Each SDK dir must contain `libhcnetsdk.so`, `libcrypto.so.*`, and `libssl.so.*`.

### PTZ controller (`hikvision_ptz.py`)

`HikvisionPTZController` wraps HCNetSDK with a persistent session (`NET_DVR_Login_V30` → `NET_DVR_Logout`). Key methods:

| Method | SDK calls | Purpose |
|--------|-----------|---------|
| `align(request)` | `NET_DVR_PTZControlWithSpeed_Other` | Nudge pan/tilt by delta degrees |
| `zoom(request)` | `NET_DVR_PTZControlWithSpeed_Other` | Nudge zoom in/out |
| `read_field_of_view()` | `NET_DVR_GetSTDConfig(GIS_INFO)` | Read optical FOV + zoom range |
| `read_zoom_limits()` | `NET_DVR_GetSTDConfig(GIS_INFO)` | Current zoom + max zoom ratio |
| `read_position()` | `NET_DVR_GetDVRConfig(PTZPOS)` | Read absolute pan/tilt/zoom (BCD) |
| `set_position(pos)` | `NET_DVR_SetDVRConfig(PTZPOS)` | Restore absolute PTZ position |
| `capture_image()` | ISAPI HTTP `/picture` | Snapshot via HTTP Digest auth |

Pan/tilt angles use BCD encoding (0.1° resolution, max 360.0°). Zoom values are arbitrary (device-dependent). Nudge operations split large movements into configurable steps and retry stop commands up to 3 times.

### Camera to env file

Camera connection parameters are read from `recognition_http_server/hikvision.env`:

```
HIK_HOST=192.168.1.64
HIK_PORT=8000
HIK_USERNAME=admin
HIK_PASSWORD=your_password
HIK_CHANNEL=1
HIK_LOCAL_IP=192.168.1.188
```

This file is gitignored. The template is at `recognition_http_server/hikvision.env`.

## Coding conventions

- Python 3.8+, 4-space indent, 120-char lines, Google-style docstrings
- Linting/formatting: `ruff`, `isort`, `yapf`, `docformatter` (configured in `pyproject.toml`)
- `snake_case` for functions/variables/modules, `PascalCase` for classes, `UPPER_CASE` for constants
- Task dispatch uses a handler dictionary; add new task types by registering a new `TaskHandler` in `RecognitionService._build_task_handlers()`
- Single-meter failure must not fail other meters on the same image
- New utility scripts go in `recognition_http_server/utils/`; tests in `tests/`

## Testing

- Test files: `test_*.py` in `tests/`; test functions: `test_*`
- `test_http.py` is the end-to-end integration client (starts a local callback server, submits tasks, validates responses)
- `pytest` runs unit tests with `--doctest-modules` per `pyproject.toml`
- SDK integration tests (`test_hikvision_ptz.py`, `test_stream_error_processing.py`) use monkeypatched fake SDK instances — no real camera required
- Model weights are gitignored except for the 3 deployment models kept in `runs/train/`

## Jetson-specific notes

- Production target is Jetson Orin NX
- `resources/` holds conda archive and Jetson packages for deployment
- `deploy_yolo_jetson_env.sh` sets up the conda environment and a systemd auto-start service
- Service logs: `results/http_service/logs/server.log`
- Stream error recovery logs: `results/gimbal_capture/logs/recovery.log`
- The arm64 Hikvision SDK ships with a Python wrapper in `HK_SDK/HK_SDK_arm64_Linux/` (used by `stream_error_processing.py` for `NET_DVR_CAMERAPARAMCFG_EX` ctypes struct access)
