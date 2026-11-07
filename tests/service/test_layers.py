# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The service's place in solprop: on top of every layer, beneath none, and Qt-free.

`solprop.service` is front-end support. It may import the rest of solprop, but nothing
else in solprop may import it, it never imports Qt or the window's package, and the
modules a window's process imports load no simulation code (docs/dev/gui/design.md,
D20).
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import solprop

PACKAGE = Path(solprop.__file__).parent
SERVICE = PACKAGE / "service"
FORBIDDEN = ("PySide6", "shiboken6", "pyqtgraph", "solprop_gui")
LIGHT = (
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
"""The modules a window's process imports, which must stay light."""


def imported(path: Path) -> set[str]:
    """Return every module a source file imports."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            found.add(node.module)
    return found


@pytest.mark.parametrize("path", sorted(SERVICE.rglob("*.py")), ids=lambda p: p.name)
def test_the_service_imports_no_qt_and_no_window(path: Path):
    tops = {name.split(".")[0] for name in imported(path)}
    assert tops.isdisjoint(FORBIDDEN)


@pytest.mark.parametrize(
    "path", sorted(Path(__file__).parent.glob("*.py")), ids=lambda p: p.name
)
def test_the_service_s_tests_need_no_window(path: Path):
    # solprop's lanes install no Qt: a test that needs the window lives in gui/tests.
    tops = {name.split(".")[0] for name in imported(path)}
    assert tops.isdisjoint({*FORBIDDEN, "pytestqt"})


def test_nothing_else_in_solprop_imports_the_service():
    importers = [
        path.relative_to(PACKAGE).as_posix()
        for path in sorted(PACKAGE.rglob("*.py"))
        if SERVICE not in path.parents
        and any(
            name == "solprop.service" or name.startswith("solprop.service.")
            for name in imported(path)
        )
    ]
    assert importers == []


def test_the_window_s_modules_load_no_simulation_code():
    heavy = ("solprop.driver", "solprop.backends", "h5py", "optmat")
    script = (
        "import importlib, sys\n"
        f"for name in {LIGHT!r}: importlib.import_module(name)\n"
        f"print([m for m in {heavy!r} if m in sys.modules])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
