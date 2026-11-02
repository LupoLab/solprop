# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import importlib
import pkgutil
import subprocess
import sys

import pytest

import solprop.recipe

LIGHT = ("quantity", "paths", "document", "errors")
"""The modules a front end imports in a process that never runs a simulation."""

HEAVY = (
    "solprop.driver",
    "solprop.backends",
    "solprop.recipe.core",
    "solprop.recipe.metadata",
    "h5py",
    "scipy",
)


def test_the_light_modules_import_without_the_driver():
    # A fresh interpreter: what matters is what importing them pulls in.
    imports = "; ".join(f"import solprop.recipe.{name}" for name in LIGHT)
    code = (
        f"import sys; {imports}; "
        f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]"


def test_every_submodule_is_an_attribute():
    on_disk = sorted(
        info.name
        for info in pkgutil.iter_modules(solprop.recipe.__path__)
        if not info.name.startswith("_")
    )
    assert sorted(solprop.recipe._SUBMODULES) == on_disk
    for name in on_disk:
        assert getattr(solprop.recipe, name) is importlib.import_module(
            f"solprop.recipe.{name}"
        )


@pytest.mark.parametrize("name", solprop.recipe.__all__)
def test_each_public_name_is_the_one_its_module_defines(name):
    home = importlib.import_module(f"solprop.recipe.{solprop.recipe._HOMES[name]}")
    assert getattr(solprop.recipe, name) is getattr(home, name)
    assert name in dir(solprop.recipe)


def test_the_public_names_are_exactly_those_with_a_home():
    assert sorted(solprop.recipe._HOMES) == sorted(solprop.recipe.__all__)


def test_dir_lists_the_surface_and_not_the_imports():
    names = dir(solprop.recipe)
    assert "quantity" in names
    assert "importlib" not in names


def test_an_unknown_name_is_an_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'Nope'"):
        _ = solprop.recipe.Nope  # pyright: ignore[reportAttributeAccessIssue]
