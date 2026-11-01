# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultrafast nonlinear pulse propagation.

solprop models ultrafast pulse propagation with the unidirectional pulse propagation
equation (UPPE) and the generalised nonlinear Schrödinger equation (GNLSE), for the full
electric field or a complex envelope, in one mode, in many modes of a waveguide and
in radial or Cartesian free space. The design is described under ``docs/dev/`` and the
manual under ``docs/``.

The package installs a :class:`logging.NullHandler`; call
:func:`solprop.log.configure_logging` from a script to see progress and diagnostics.

Every submodule is an attribute of the package -- ``solprop.grid``, ``solprop.driver``,
``solprop.plotting`` -- loaded on first access (:pep:`562`), so ``dir(solprop)`` and tab
completion list them while ``import solprop`` stays within its import-time budget: the
modules that pull in h5py, matplotlib or a compiled backend are not imported until they
are used.
"""

import importlib
import logging
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Final

# The scheduler's CPU grant, read before any OpenMP runtime binds the main thread.
import solprop._process  # noqa: F401  # pyright: ignore[reportUnusedImport]
from solprop.exceptions import SolpropError
from solprop.log import configure_logging

__all__ = ["SolpropError", "__version__", "configure_logging"]

_SUBMODULES: Final = (
    "adiabatic",
    "backends",
    "boundaries",
    "capabilities",
    "cli",
    "constants",
    "coupling",
    "driver",
    "exceptions",
    "fourier",
    "gasflow",
    "grid",
    "hankel",
    "linear",
    "log",
    "modes",
    "monitor",
    "noise",
    "nonlinear",
    "optics",
    "output",
    "physdata",
    "plotting",
    "polarisation",
    "processing",
    "profiles",
    "pulses",
    "recipe",
    "scan",
    "sol",
    "spatial",
    "stats",
    "stepper",
    "tools",
    "translate",
)
"""The submodules reachable as attributes; a test keeps it equal to what is on disk."""


def __getattr__(name: str) -> ModuleType:
    """Import a submodule the first time it is asked for as an attribute.

    Parameters
    ----------
    name : str
        The attribute looked up on the package.

    Returns
    -------
    types.ModuleType
        The submodule, imported now and cached by the import system.

    Raises
    ------
    AttributeError
        If `name` is not a submodule of the package.
    """
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List the package's public surface, the submodules included whether or not loaded.

    The names in ``__all__``, the submodules and the dunders; not the imports this
    module happens to make, which are implementation and not part of the package.

    Returns
    -------
    list of str
        The sorted names ``dir`` and tab completion show.
    """
    names = {*globals(), *_SUBMODULES}
    return sorted(
        name
        for name in names
        if name.startswith("__") or name in __all__ or name in _SUBMODULES
    )


try:
    __version__ = version("solprop")
except PackageNotFoundError:  # pragma: no cover - only when run from a source checkout
    __version__ = "0.0.0"

logging.getLogger(__name__).addHandler(logging.NullHandler())
