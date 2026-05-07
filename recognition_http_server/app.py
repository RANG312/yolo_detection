from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    # 兼容直接执行 `python recognition_http_server/app.py` 的场景。
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from log_manager import GlobalLogManager
from recognition_http_server.config import parse_args
from recognition_http_server.constants import DEFAULT_LOG_DIR_NAME
from recognition_http_server.http_api import RecognitionAPIServer, RecognitionHandler
from recognition_http_server.service import RecognitionService


def main() -> None:
    """初始化配置、日志和业务服务，并启动 HTTP 服务监听。"""
    args = parse_args()
    args.result_root = args.result_root.resolve()
    log_dir = args.result_root / DEFAULT_LOG_DIR_NAME
    GlobalLogManager.configure(log_dir)
    logger = GlobalLogManager.get_logger("recognition.main")
    service = RecognitionService(args)
    server = RecognitionAPIServer((args.host, args.port), RecognitionHandler, service)
    logger.info("log file: %s", GlobalLogManager.get_log_file_path())
    logger.info("meter model loaded: %s", args.model)
    logger.info("fire model loaded: %s", args.fire_model)
    logger.info("safehat model loaded: %s", args.safehat_model)
    logger.info("fire protection facilities model loaded: %s", args.fire_protection_facilities_model)
    logger.info("person fall-down model loaded: %s", args.person_fall_down_model)
    logger.info("fire extinguisher model loaded: %s", args.fire_extinguisher_model)
    logger.info("person and cars model loaded: %s", args.person_and_cars_model)
    logger.info("meter annotation mode: %s", args.annotation_mode)
    logger.info("listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("server interrupted by keyboard")
    finally:
        server.server_close()
        logger.info("server closed")


if __name__ == "__main__":
    main()
