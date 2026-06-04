#!/usr/bin/env python3
"""Compile recognition service business modules into importable extension modules."""

from __future__ import annotations

import shutil
import sys
import sysconfig
from pathlib import Path

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


def compiled_extension_path(source_path: Path) -> Path:
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        raise RuntimeError("Python EXT_SUFFIX is unavailable.")
    return source_path.with_name(f"{source_path.stem}{extension_suffix}")


def remove_protected_sources() -> None:
    """Remove protected Python sources only after every extension exists."""
    extension_paths = []
    for relative_path in PROTECTED_MODULES:
        source_path = ROOT / relative_path
        extension_path = compiled_extension_path(source_path)
        if not extension_path.is_file():
            raise RuntimeError(f"Refusing to remove source without compiled extension: {source_path}")
        extension_paths.append(extension_path)

    for relative_path in PROTECTED_MODULES:
        source_path = ROOT / relative_path
        source_path.unlink(missing_ok=True)
        pycache_dir = source_path.parent / "__pycache__"
        for cache_path in pycache_dir.glob(f"{source_path.stem}.*.pyc"):
            try:
                cache_path.unlink()
            except PermissionError:
                print(f"[protected-build] 无权限删除字节码缓存，跳过: {cache_path}", flush=True)

    shutil.rmtree(ROOT / "build", ignore_errors=True)
    for extension_path in extension_paths:
        print(f"[protected-build] 保留编译模块: {extension_path.relative_to(ROOT)}", flush=True)
    print("[protected-build] 已删除核心 Python 源码、字节码缓存和 Cython 中间文件", flush=True)


def main() -> None:
    remove_sources = "--remove-sources" in sys.argv
    if remove_sources:
        sys.argv.remove("--remove-sources")
    source_paths = [ROOT / path for path in PROTECTED_MODULES]
    missing_sources = [path for path in source_paths if not path.is_file()]
    if missing_sources:
        missing_extensions = [path for path in source_paths if not compiled_extension_path(path).is_file()]
        if missing_extensions:
            raise RuntimeError(f"Protected sources or extensions are incomplete: {missing_extensions}")
        print("[protected-build] 核心源码已删除，复用现有编译模块", flush=True)
        if remove_sources:
            remove_protected_sources()
        return
    from Cython.Build import cythonize
    from setuptools import Extension, setup

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
    if remove_sources:
        remove_protected_sources()


if __name__ == "__main__":
    main()
