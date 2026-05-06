# Repository Guidelines

## Project Structure & Module Organization

This repository targets NVIDIA Jetson Orin NX deployments and extends Ultralytics YOLO with custom dial-gauge reading, fire detection, and safety-helmet detection pipelines. Core library code lives in `ultralytics/`. The HTTP service is in `recognition_http_server/`; `server.py` is a compatibility entry point and should not grow new logic. Dial-reading compatibility code is in `dial_reading.py`, with the package implementation under `recognition_http_server/dial_reading/`. Training and export helpers live in `scripts/`. ROS2 integration lives under `ros2_recognition_service/`. Jetson deployment packages, including the packed conda environment, belong in `resources/`; runtime outputs/logs are written under `results/`.

## Architecture & Task Routing

The HTTP service uses a handler registry pattern for recognition tasks. Avoid adding `if`/`elif` task-dispatch chains; add new task types by registering a `TaskHandler` in `RecognitionService._build_task_handlers()`.

| Task | `recognize_type` | Internal kind | Handler |
|------|------------------|---------------|---------|
| Dial meter reading | `"1"` | `meter` | `recognition_http_server/handlers/meter.py` |
| Fire detection | `"6"` | `fire` | `recognition_http_server/handlers/detection.py` |
| Safety helmet | `"7"` | `safehat` | `recognition_http_server/handlers/detection.py` |

Key modules:

- `recognition_http_server/service.py`: `RecognitionService`, model loading, task lifecycle, background callbacks, and handler registration.
- `recognition_http_server/http_api.py`: Flask HTTP layer for `POST /api/v1/recognition/tasks`, `GET /api/v1/recognition/tasks/{req_id}`, and `GET /health`.
- `recognition_http_server/helpers.py`: request parsing, task-type routing, callback URL construction, and result formatting.
- `recognition_http_server/constants.py`: default paths, ports, thresholds, and task mappings. Update configuration here rather than duplicating it in handlers.
- `recognition_http_server/dial_reading/pipeline.py`: pointer-meter inference pipeline.

Pointer-meter reading runs YOLO detection on the full image, retries detection on a resized gauge crop for `meter_data_9k` models, assigns detected center/ticks/pointer elements, computes the reading geometrically, and saves an annotated visualization. Both `meter_data_9k` and `legacy` annotation protocols are supported through `infer_annotation_mode()`. A single-meter failure must not fail other meters on the same image.

Local input paths are read directly. HTTP URLs are downloaded to `results/http_service/inputs/<req_id>/`; result images are saved to `results/http_service/outputs/<req_id>/`. For local inputs, a `-detection` result copy is also saved alongside the original image.

## Build, Test, and Development Commands

- `pip install -e ".[dev]"`: install editable local development dependencies.
- `python server.py --host 0.0.0.0 --port 3208`: start the HTTP recognition service.
- `python recognition_http_server/app.py`: start the service through the package entry module.
- `python dial_reading.py --image path/to/image.jpg --device cpu`: run single-image dial reading.
- `python test_http.py --server http://127.0.0.1:3208 --images path/to/a.jpg --data-type 1:25`: run the local HTTP integration client against a running service.
- `pytest`: run Python tests and doctests using the project `pyproject.toml` settings.
- `python scripts/export_best_to_onnx.py`: export the best model to ONNX.
- `bash deploy_yolo_jetson_env.sh`: auto-deploy the Jetson Orin NX conda environment from packages in `resources/` and configure the systemd service.
- `bash ros2_recognition_service/deploy_ros2_ws.sh`: build/deploy the ROS2 workspace.
- `source ./ros2_recognition_service/source_ros2_ws.bash`: source the ROS2 workspace after deployment.
- `bash ros2_recognition_service/launch_recognition_service.sh`: launch the ROS2 recognition service.

## Coding Style & Naming Conventions

Use Python 3.8+ with 4-space indentation, 120-character lines, and Google-style docstrings. Tooling is configured in `pyproject.toml`: `ruff`, `isort`, `yapf`, and `docformatter`. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase constants such as `DEFAULT_PORT`. Keep service request parsing, routing, and image handling in `recognition_http_server/` instead of growing `server.py` or `dial_reading.py`.

## Testing Guidelines

Prefer focused tests for changed behavior. Name test files `test_*.py` and functions `test_*`; keep unit tests under `tests/`. `pytest` runs unit tests with doctests through the project `pyproject.toml` settings. For HTTP changes, exercise `test_http.py` with representative `meter`, `fire`, and `safehat` requests when models are available. Keep generated payloads, logs, and visual outputs under `results/`.

## Jetson Deployment Notes

Treat Jetson Orin NX as the production target. Keep the conda archive and Jetson packages under `resources/`, then use `deploy_yolo_jetson_env.sh` for repeatable setup. Service logs are written to `results/http_service/logs/server.log`.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries such as `modify readme`, `add results save to origin path`, and `debug scripts`. Keep commits small and behavior-focused. Pull requests should include a concise description, affected entry points, validation commands, linked issues when applicable, and screenshots or sample JSON for API or visual changes.

## Security & Configuration Tips

Do not commit model weights, generated datasets, credentials, or machine-specific paths. Model weights are gitignored except for the deployment models intentionally kept in `runs/train/`. Service defaults and task mappings are centralized in `recognition_http_server/constants.py`; update that file rather than duplicating configuration in handlers.
