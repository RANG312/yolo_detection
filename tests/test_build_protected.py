from __future__ import annotations

import sys
from pathlib import Path

import pytest

import scripts.build_protected as build_protected


def test_remove_protected_sources_refuses_to_delete_without_extension(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "package" / "core.py"
    source_path.parent.mkdir()
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(build_protected, "ROOT", tmp_path)
    monkeypatch.setattr(build_protected, "PROTECTED_MODULES", ("package/core.py",))

    with pytest.raises(RuntimeError, match="Refusing to remove source without compiled extension"):
        build_protected.remove_protected_sources()

    assert source_path.exists()


def test_remove_protected_sources_deletes_source_cache_and_build_dir(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "package" / "core.py"
    source_path.parent.mkdir()
    source_path.write_text("VALUE = 1\n", encoding="utf-8")
    extension_path = build_protected.compiled_extension_path(source_path)
    extension_path.touch()
    pycache_path = source_path.parent / "__pycache__" / "core.cpython-310.pyc"
    pycache_path.parent.mkdir()
    pycache_path.touch()
    build_file = tmp_path / "build" / "cython" / "core.c"
    build_file.parent.mkdir(parents=True)
    build_file.touch()
    monkeypatch.setattr(build_protected, "ROOT", tmp_path)
    monkeypatch.setattr(build_protected, "PROTECTED_MODULES", ("package/core.py",))

    build_protected.remove_protected_sources()

    assert not source_path.exists()
    assert extension_path.exists()
    assert not pycache_path.exists()
    assert not (tmp_path / "build").exists()


def test_main_reuses_extensions_when_sources_were_already_removed(monkeypatch, tmp_path: Path) -> None:
    source_path = tmp_path / "package" / "core.py"
    source_path.parent.mkdir()
    build_protected.compiled_extension_path(source_path).touch()
    monkeypatch.setattr(build_protected, "ROOT", tmp_path)
    monkeypatch.setattr(build_protected, "PROTECTED_MODULES", ("package/core.py",))
    monkeypatch.setattr(sys, "argv", ["build_protected.py"])

    build_protected.main()
