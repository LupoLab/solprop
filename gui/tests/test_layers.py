# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The package's layer rules, read from its imports.

The design (docs/dev/gui/design.md, "Code structure") fixes three rules:

- ``model``, ``registry``, ``classes`` and ``help`` import no Qt;
- the window's process imports from solprop only the light parts of
  ``solprop.recipe`` and of ``solprop.service``; simulation code runs in the kernel and
  the workers, which live in ``solprop.service``;
- nothing imports a private solprop name, so the front end rests on solprop's public
  interface alone.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import solprop_gui

PACKAGE = Path(solprop_gui.__file__).parent

QT = ("PySide6", "shiboken6", "pyqtgraph")
WITHOUT_QT = ("model", "registry", "classes", "help")
# What the window's process may import from solprop: the light parts of the recipe, and
# the parts of the service that start, reach and describe its processes.
WINDOW_PROCESS_SOLPROP = (
    "solprop.recipe.document",
    "solprop.recipe.errors",
    "solprop.recipe.paths",
    "solprop.recipe.quantity",
    "solprop.service.client",
    "solprop.service.explore",
    "solprop.service.frames",
    "solprop.service.launch",
    "solprop.service.machines",
    "solprop.service.messages",
    "solprop.service.recipes",
    "solprop.service.reduce",
    "solprop.service.record",
    "solprop.service.remote",
    "solprop.service.runs",
    "solprop.service.states",
    "solprop.service.worker",
)


def imports(source: str) -> list[tuple[str, tuple[str, ...]]]:
    """Return each import in ``source`` as (module, names imported from it)."""
    found: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.append((node.module, tuple(alias.name for alias in node.names)))
    return found


def is_solprop(module: str) -> bool:
    return module == "solprop" or module.startswith("solprop.")


def violations(source: str, layer: str) -> list[str]:
    """Return the layer rules that ``source``, a module of ``layer``, breaks."""
    found: list[str] = []
    for module, names in imports(source):
        top = module.split(".")[0]
        if layer in WITHOUT_QT and top in QT:
            found.append(f"{layer} imports Qt ({module})")
        if not is_solprop(module):
            continue
        private = [
            part for part in (*module.split("."), *names) if part.startswith("_")
        ]
        if private:
            found.append(
                f"{layer} imports a private solprop name ({module}: {private})"
            )
        if module not in WINDOW_PROCESS_SOLPROP:
            found.append(f"{layer} imports {module} into the window's process")
    return found


def layer_of(path: Path) -> str:
    relative = path.relative_to(PACKAGE)
    return relative.parts[0] if len(relative.parts) > 1 else "top"


MODULES = sorted(PACKAGE.rglob("*.py"))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(PACKAGE)))
def test_each_module_keeps_its_layer_rules(path: Path):
    assert violations(path.read_text(encoding="utf-8"), layer_of(path)) == []


@pytest.mark.parametrize(
    ("source", "layer", "rule"),
    [
        ("from PySide6.QtCore import Qt", "classes", "imports Qt"),
        ("import pyqtgraph", "model", "imports Qt"),
        ("from solprop.driver._launch import x", "model", "private"),
        ("from solprop.recipe.core import _decode", "ui", "private"),
        ("from solprop.service.handlers import HANDLERS", "ui", "into the window's"),
        ("import solprop.driver", "ui", "into the window's process"),
        ("from solprop.recipe import Recipe", "model", "into the window's process"),
    ],
)
def test_a_broken_rule_is_reported(source: str, layer: str, rule: str):
    assert any(rule in message for message in violations(source, layer))


@pytest.mark.parametrize(
    ("source", "layer"),
    [
        ("from PySide6.QtWidgets import QLabel", "ui"),
        ("from solprop.recipe.quantity import parse", "model"),
        ("from solprop.service.messages import Reply", "ui"),
        ("import numpy as np", "registry"),
    ],
)
def test_an_allowed_import_passes(source: str, layer: str):
    assert violations(source, layer) == []


def test_the_window_process_loads_no_simulation_code():
    # What the window's own modules pull in when imported, in a fresh interpreter.
    window = [
        ".".join(("solprop_gui", *path.relative_to(PACKAGE).with_suffix("").parts))
        for path in MODULES
        if path.name != "__main__.py"
    ]
    window = [name.removesuffix(".__init__") for name in window]
    heavy = ("solprop.driver", "solprop.backends", "h5py", "numba", "optmat")
    script = (
        "import importlib, sys\n"
        f"for name in {window!r}: importlib.import_module(name)\n"
        f"print([m for m in {heavy!r} if m in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
