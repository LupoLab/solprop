# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib
import pkgutil
import subprocess
import sys

import pytest

import solprop.output

RECORDS = ("records", "metadata", "serialisation", "provenance", "store", "capability")
"""The run-record modules translation and the backends import (layer S)."""


def test_the_record_modules_import_without_the_handlers():
    # A fresh interpreter: what matters is what importing them pulls in.
    imports = "; ".join(f"import solprop.output.{name}" for name in RECORDS)
    handlers = ("h5py", "solprop.output.hdf5", "solprop.output.memory")
    code = (
        f"import sys; {imports}; "
        f"print(sorted(m for m in {handlers!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]"


def test_every_submodule_is_an_attribute():
    on_disk = sorted(
        info.name
        for info in pkgutil.iter_modules(solprop.output.__path__)
        if not info.name.startswith("_")
    )
    assert sorted(solprop.output._SUBMODULES) == on_disk
    for name in on_disk:
        assert getattr(solprop.output, name) is importlib.import_module(
            f"solprop.output.{name}"
        )


@pytest.mark.parametrize("name", solprop.output.__all__)
def test_each_public_name_is_the_one_its_module_defines(name):
    home = importlib.import_module(f"solprop.output.{solprop.output._HOMES[name]}")
    assert getattr(solprop.output, name) is getattr(home, name)
    assert name in dir(solprop.output)


def test_the_public_names_are_exactly_those_with_a_home():
    assert sorted(solprop.output._HOMES) == sorted(solprop.output.__all__)


def test_an_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        _ = solprop.output.Nope  # pyright: ignore[reportAttributeAccessIssue]


def test_a_submodule_not_yet_bound_is_imported_on_first_use(monkeypatch):
    # Importing a submodule binds it on the package; unbind one to reach the lazy path.
    monkeypatch.delattr(solprop.output, "hdf5")
    assert solprop.output.hdf5 is importlib.import_module("solprop.output.hdf5")
