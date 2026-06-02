#!/usr/bin/env python3
"""Compile recognition service business modules into importable extension modules."""

from __future__ import annotations

from pathlib import Path

from Cython.Build import cythonize
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_MODULES = (
    "recognition_http_server/service.py",
    "recognition_http_server/helpers.py",
    "recognition_http_server/schemas.py",
    "recognition_http_server/hikvision_ptz.py",
    "recognition_http_server/ptz_alignment.py",
    "recognition_http_server/virtual_ptz.py",
    "recognition_http_server/handlers/detection.py",
    "recognition_http_server/handlers/meter.py",
    "recognition_http_server/dial_reading/constants.py",
    "recognition_http_server/dial_reading/detections.py",
    "recognition_http_server/dial_reading/geometry.py",
    "recognition_http_server/dial_reading/pipeline.py",
    "recognition_http_server/dial_reading/visualization.py",
    "recognition_http_server/utils/image_io.py",
)


def module_name(relative_path: str) -> str:
    return str(Path(relative_path).with_suffix("")).replace("/", ".")


def main() -> None:
    extensions = [Extension(module_name(path), [str(ROOT / path)]) for path in PROTECTED_MODULES]
    setup(
        name="recognition-http-server-protected",
        ext_modules=cythonize(
            extensions,
            build_dir=str(ROOT / "build" / "cython"),
            compiler_directives={
                "language_level": "3",
                "embedsignature": False,
            },
        ),
    )


if __name__ == "__main__":
    main()
