# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

from solprop.backends.julia import _source


def _sources(root):
    root.mkdir()
    (root / "src").mkdir()
    (root / "Project.toml").write_text('name = "engine"\n', encoding="utf-8")
    (root / "bridge.toml").write_text('bridge = "fixture"\n', encoding="utf-8")
    (root / "src" / "engine.jl").write_text("module engine\nend\n", encoding="utf-8")
    return root


def test_engine_digest_is_relocatable_and_includes_contract(tmp_path):
    original = _sources(tmp_path / "original")
    clone = tmp_path / "clone"
    shutil.copytree(original, clone)
    assert _source.engine_digest(original) == _source.engine_digest(clone)
    (clone / "bridge.toml").write_text('bridge = "changed"\n', encoding="utf-8")
    assert _source.engine_digest(original) != _source.engine_digest(clone)


def test_engine_digest_ignores_environment_and_generated_cache(tmp_path):
    root = _sources(tmp_path / "engine")
    expected = _source.engine_digest(root)
    (root / "Manifest.toml").write_text("fixture", encoding="utf-8")
    (root / "compiled").mkdir()
    (root / "compiled" / "engine.so").write_bytes(b"generated")
    assert _source.engine_digest(root) == expected


@pytest.mark.skipif(
    sys.platform == "win32", reason="creating symlinks needs Windows privileges"
)
def test_source_hash_cannot_escape_bundle(tmp_path):
    root = _sources(tmp_path / "engine")
    external = tmp_path / "external.jl"
    external.write_text("x = 1", encoding="utf-8")
    (root / "src" / "outside.jl").symlink_to(external)
    with pytest.raises(ValueError):
        _source.engine_digest(root)


def test_build_hook_can_load_source_module_without_package_context(tmp_path):
    root = _sources(tmp_path / "engine")
    path = Path(_source.__file__)
    spec = importlib.util.spec_from_file_location("standalone_source_digest", path)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    assert loaded.engine_digest(root) == _source.engine_digest(root)


def test_missing_source_is_not_a_valid_engine(tmp_path):
    with pytest.raises(FileNotFoundError):
        _source.engine_digest(tmp_path)


def test_source_location_distinguishes_editable_checkout_and_wheel(
    tmp_path, monkeypatch
):
    package = tmp_path / "src" / "solprop" / "backends" / "julia"
    monkeypatch.setattr(_source, "__file__", str(package / "_source.py"))
    assert _source.source_bundle() == package / "_bundle"
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "julia").mkdir()
    assert _source.source_bundle() == tmp_path / "julia"


def test_setup_inputs_always_name_required_installer_and_version_table(tmp_path):
    expected = {
        tmp_path / "scripts" / "instantiate.jl",
        tmp_path / "environments" / "versions.toml",
    }
    assert set(_source.setup_files(tmp_path, ())) == expected
