# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ultrafast nonlinear pulse propagation.

solprop models ultrafast pulse propagation with the unidirectional pulse propagation
equation (UPPE) and the generalised nonlinear Schrödinger equation (GNLSE), for the full
electric field or a complex envelope, in one mode, in many modes of a waveguide and in
radial or Cartesian free space. The manual is under ``docs/guide/`` and the design under
``docs/dev/``.

- **Setting up and running**: :mod:`~solprop.driver` (the ``prop_*`` entry points and
  scans), :mod:`~solprop.recipe`, :mod:`~solprop.grid`, :mod:`~solprop.pulses`,
  :mod:`~solprop.profiles`, :mod:`~solprop.gasflow`, :mod:`~solprop.stepper`,
  :mod:`~solprop.boundaries` and :mod:`~solprop.noise`.
- **The physics**: :mod:`~solprop.modes`, :mod:`~solprop.nonlinear`,
  :mod:`~solprop.physdata`, :mod:`~solprop.linear`, :mod:`~solprop.spatial`,
  :mod:`~solprop.spatial_absorber`, :mod:`~solprop.adiabatic`,
  :mod:`~solprop.polarisation`, :mod:`~solprop.fourier`, :mod:`~solprop.hankel` and
  :mod:`~solprop.constants`.
- **Results**: :mod:`~solprop.output`, :mod:`~solprop.processing`,
  :mod:`~solprop.stats`, :mod:`~solprop.monitor` and :mod:`~solprop.transverse_monitor`,
  :mod:`~solprop.plotting`, :mod:`~solprop.coupling`, :mod:`~solprop.optics` and the
  design calculators of :mod:`~solprop.tools`.
- **Session and tooling**: :mod:`~solprop.sol` (the interactive namespace),
  :mod:`~solprop.log`, :mod:`~solprop.exceptions`, :mod:`~solprop.cli`,
  :mod:`~solprop.scan`, :mod:`~solprop.capabilities`, :mod:`~solprop.calibration` and
  :mod:`~solprop.service`.
- **Internals**: :mod:`~solprop.translate` and :mod:`~solprop.backends`.

Every submodule is an attribute of the package, imported on first access (:pep:`562`),
so ``dir(solprop)`` and tab completion list them while ``import solprop`` stays within
its import-time budget: h5py, matplotlib and a compiled backend load only when used. The
package installs a :class:`logging.NullHandler`; call
:func:`solprop.log.configure_logging` from a script to see progress and diagnostics.
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
    "calibration",
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
    "service",
    "sol",
    "spatial",
    "spatial_absorber",
    "stats",
    "stepper",
    "tools",
    "translate",
    "transverse_monitor",
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
