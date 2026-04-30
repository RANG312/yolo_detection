# Repository Guidelines

## Project Structure & Module Organization

This repository targets NVIDIA Jetson Orin NX deployments and extends Ultralytics YOLO with dial-reading and recognition-service code. Core library code lives in `ultralytics/`. The HTTP service is in `recognition_http_server/`; `server.py` is the compatibility entry point. Dial-reading logic is in `dial_reading.py`, with training/export helpers in `scripts/`. ROS2 integration lives under `ros2_recognition_service/`. Jetson deployment packages, including the packed conda environment, belong in `resources/`; runtime outputs/logs are written under `results/`.

## Build, Test, and Development Commands

- `bash deploy_yolo_jetson_env.sh`: auto-deploy the Jetson Orin NX conda environment from packages in `resources/` and configure the service.
- `pip install -e ".[dev]"`: install editable local development dependencies.
- `python server.py --host 0.0.0.0 --port 3208`: start the HTTP recognition service.
- `python recognition_http_server/app.py`: start the service through the package entry module.
- `python test_http.py --images path/to/image.jpg --scene meter`: run the local HTTP integration client against a running service.
- `python dial_reading.py --image path/to/image.jpg --device cpu`: run single-image dial reading.
- `pytest`: run Python tests and doctests using the project `pyproject.toml` settings.
- `bash ros2_recognition_service/deploy_ros2_ws.sh`: build/deploy the ROS2 workspace.

## Coding Style & Naming Conventions

Use Python 3.8+ with 4-space indentation, 120-character lines, and Google-style docstrings. Tooling is configured in `pyproject.toml`: `ruff`, `isort`, `yapf`, and `docformatter`. Use `snake_case` for functions, variables, and modules; `PascalCase` for classes; and uppercase constants such as `DEFAULT_PORT`. Keep service request parsing, routing, and image handling in `recognition_http_server/` instead of growing `server.py`.

## Testing Guidelines

Prefer focused tests for changed behavior. Name test files `test_*.py` and functions `test_*`. For HTTP changes, exercise `test_http.py` with representative `meter`, `fire`, and `safehat` requests when models are available. Keep generated payloads, logs, and visual outputs under `results/`.

## Jetson Deployment Notes

Treat Jetson Orin NX as the production target. Keep the conda archive and Jetson packages under `resources/`, then use `deploy_yolo_jetson_env.sh` for repeatable setup.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries such as `modify readme`, `add results save to origin path`, and `debug scripts`. Keep commits small and behavior-focused. Pull requests should include a concise description, affected entry points, validation commands, linked issues when applicable, and screenshots or sample JSON for API or visual changes.

## Security & Configuration Tips

Do not commit model weights, generated datasets, credentials, or machine-specific paths. Service defaults and task mappings are centralized in `recognition_http_server/constants.py`; update that file rather than duplicating configuration in handlers.
