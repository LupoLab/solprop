# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Experimental double-precision CPU kernels, independent of solprop-native."""

import os
import sys
from importlib.metadata import version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .backend import CPUNativeBackend

# libgomp may bind the importing thread as soon as the extension is loaded.
# Preserve both sides so solprop can distinguish that binding from its CPU grant,
# including a core place containing two SMT siblings.
PROCESS_CPUS = tuple(sorted(os.sched_getaffinity(0))) if sys.platform == "linux" else ()

from . import (  # noqa: E402 - capture affinity before loading OpenMP
    _core,  # pyright: ignore[reportMissingModuleSource]
)

# The delayed import above records affinity before loading OpenMP.

OPENMP_INITIAL_CPUS = (
    tuple(sorted(os.sched_getaffinity(0))) if sys.platform == "linux" else ()
)

__all__ = ["CPUNativeBackend", "__version__", "build_info", "check_build"]
__version__ = version("solprop-cpunative")


def build_info() -> dict[str, object]:
    """Return the compiler, CPU target, FFTW and kernel-schema build provenance.

    Returns
    -------
    dict
        Build settings, source digest, package version and loaded runtime paths.
    """
    return dict(_core.build_info()) | {"package_version": __version__}


def check_build() -> None:
    """Check that the installed engine and solprop share a version and schema.

    Raises
    ------
    solprop.exceptions.BackendError
        If the packages have different versions or kernel schemas.
    """
    from solprop.exceptions import BackendError
    from solprop.translate.problem import SCHEMA_VERSION

    if _core.build_info()["schema_version"] != SCHEMA_VERSION:
        raise BackendError("cpunative was built for a different kernel schema")
    if version("solprop") != __version__:
        raise BackendError(
            "solprop and solprop-cpunative must be installed from the same version"
        )


def __getattr__(name: str) -> object:
    if name == "CPUNativeBackend":
        from .backend import CPUNativeBackend

        return CPUNativeBackend
    raise AttributeError(name)
