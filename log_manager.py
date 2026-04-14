from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock


class GlobalLogManager:
    """Thread-safe global logger manager for service processes."""

    _lock = RLock()
    _configured = False
    _log_dir: Path | None = None
    _log_level = logging.INFO
    _log_file_name = "server.log"

    @classmethod
    def configure(
        cls,
        log_dir: str | Path,
        level: int = logging.INFO,
        log_file_name: str = "server.log",
    ) -> logging.Logger:
        with cls._lock:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(parents=True, exist_ok=True)

            root_logger = logging.getLogger()
            root_logger.setLevel(level)

            if cls._configured:
                cls._log_dir = log_dir_path
                cls._log_level = level
                cls._log_file_name = log_file_name
                for handler in root_logger.handlers:
                    handler.setLevel(level)
                return logging.getLogger("app")

            formatter = logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(name)s | %(threadName)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)

            file_handler = RotatingFileHandler(
                log_dir_path / log_file_name,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)

            root_logger.handlers.clear()
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)

            cls._configured = True
            cls._log_dir = log_dir_path
            cls._log_level = level
            cls._log_file_name = log_file_name
            return logging.getLogger("app")

    @classmethod
    def get_logger(cls, name: str = "app") -> logging.Logger:
        with cls._lock:
            if not cls._configured:
                cls.configure(Path("logs"))
            return logging.getLogger(name)

    @classmethod
    def get_log_file_path(cls) -> Path:
        with cls._lock:
            if cls._log_dir is None:
                cls.configure(Path("logs"))
            return cls._log_dir / cls._log_file_name
