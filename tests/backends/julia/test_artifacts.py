# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import sys

import pytest

from solprop.backends.julia import _artifacts
from solprop.exceptions import BackendError


@pytest.mark.parametrize("value", [None, [], {1: "value"}])
def test_metadata_tables_require_string_keys(value):
    with pytest.raises(BackendError, match="invalid metadata table"):
        _artifacts.mapping(value, "metadata")


@pytest.mark.parametrize("value", [None, [], 1, ""])
def test_required_metadata_text_is_not_coerced(value):
    with pytest.raises(BackendError, match="invalid field"):
        _artifacts.text_value({"field": value}, "field")


@pytest.mark.parametrize("value", [None, "text", [1]])
def test_string_arrays_require_only_strings(value):
    with pytest.raises(BackendError, match="invalid field list"):
        _artifacts.string_list({"field": value}, "field")


def test_metadata_table_arrays_are_validated():
    with pytest.raises(BackendError, match="invalid field list"):
        _artifacts.table_list({"field": None}, "field")
    with pytest.raises(BackendError, match="invalid field table"):
        _artifacts.table_list({"field": [42]}, "field")


@pytest.mark.parametrize(
    "name", ["", "../escape", "/absolute", r"dir\file", "C:relative"]
)
def test_inventory_paths_cannot_escape_their_root(tmp_path, name):
    with pytest.raises(BackendError, match="unsafe inventory path"):
        _artifacts.child(tmp_path, name)


@pytest.mark.skipif(
    sys.platform == "win32", reason="creating symlinks needs Windows privileges"
)
def test_inventory_paths_cannot_escape_via_symlinks(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path / "outside")
    with pytest.raises(BackendError, match="escapes its root"):
        _artifacts.child(root, "escape")


@pytest.mark.skipif(
    sys.platform == "win32", reason="creating symlinks needs Windows privileges"
)
def test_broken_override_symlink_still_refuses(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "Overrides.toml").symlink_to(tmp_path / "absent")
    with pytest.raises(BackendError, match="overrides are incompatible"):
        _artifacts.refuse_overrides(tmp_path)
