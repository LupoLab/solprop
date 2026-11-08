# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""solprop-native's standard-library modules, loaded without the native runtime.

A calibration's controller must not load the native extension or a device runtime
(solprop SEP 46 section 6.2), and importing ``solprop_native`` starts its OpenMP
runtime. Its store, topology, fingerprint, allocation, table and profile modules
import nothing but the standard library and each other, so they are loaded here by
path into a private package of their own, as the backend registry loads its processor
check (``backends/registry.py``). Nothing here runs ``solprop_native/__init__``.

Use them as ``from solprop.calibration import _native`` and then ``_native.store``,
``_native.profile`` and so on; each loads on first use.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from solprop.exceptions import BackendError

if TYPE_CHECKING:
    from solprop_native import _allocation as allocation
    from solprop_native import _cpu as cpu
    from solprop_native import _execution as execution
    from solprop_native import _fingerprint as fingerprint
    from solprop_native import _profile as profile
    from solprop_native import _store as store
    from solprop_native import _topology as topology

__all__ = [
    "allocation",
    "cpu",
    "execution",
    "fingerprint",
    "packages",
    "profile",
    "store",
    "topology",
    "variant",
]

_PACKAGE = "_solprop_calibration_native"
_MODULES = (
    "allocation",
    "cpu",
    "execution",
    "fingerprint",
    "profile",
    "store",
    "topology",
)


def packages() -> tuple[Path, ...]:
    """Return the installed ``solprop_native`` package's directories.

    An editable install keeps its sources and its built files (the extension,
    ``_build.json``) in separate directories; a wheel has one.

    Returns
    -------
    tuple of pathlib.Path
        The package's directories, in import order.

    Raises
    ------
    BackendError
        When solprop-native is not installed.
    """
    spec = importlib.util.find_spec("solprop_native")
    if spec is None or not spec.submodule_search_locations:
        raise BackendError(
            "a machine calibration needs the native backend: install solprop-native"
        )
    return tuple(Path(entry) for entry in spec.submodule_search_locations)


def variant() -> str:
    """Return the extension the native import chooses (``_cpu.select_variant``).

    Asked of the directory that holds the package's sources, as its ``__init__``
    asks it of its own.

    Returns
    -------
    str
        The extension's stem, ``"_core"`` for a build with one.
    """
    for directory in packages():
        if (directory / "_cpu.py").is_file():
            return __getattr__("cpu").select_variant(directory)
    return "_core"


def __getattr__(name: str) -> ModuleType:
    """Load one of the package's pure modules under the private package."""
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if _PACKAGE not in sys.modules:
        parent = ModuleType(_PACKAGE)
        parent.__path__ = [str(entry) for entry in packages()]
        sys.modules[_PACKAGE] = parent
    return importlib.import_module(f"{_PACKAGE}._{name}")
