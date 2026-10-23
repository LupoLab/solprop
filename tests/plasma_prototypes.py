# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared inputs for the plasma prototypes, which live under ``tests/prototypes/``.

The three prototype modules (a fixed-partition scan, an eight-node tiled traversal and
the scan kernels) were rejected by measurement and stay beside their tests rather
than in the package; their tests reach them through this module, which also carries
the deterministic inputs and baseline arguments they share.
"""

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

from solprop.backends.bodies import PLAIN, RateArrays

PROTOTYPES = Path(__file__).resolve().parent / "prototypes"


def _load(name: str) -> ModuleType:
    """Load a prototype module from ``tests/prototypes/`` by path, not via sys.path.

    The prototypes import each other by their module names, so each is registered in
    ``sys.modules`` before its body runs; a module already loaded (by a benchmark
    script or an earlier test) is reused.

    Parameters
    ----------
    name : str
        The module's basename under ``tests/prototypes/``.

    Returns
    -------
    types.ModuleType
        The loaded module.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PROTOTYPES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_scans = _load("plasma_scan_kernels")
_scan = _load("plasma_scan")
_tiles = _load("plasma_tiles")
ScanKernels, make_scan_kernels = _scans.ScanKernels, _scans.make_scan_kernels
BlockedPlasma, PlasmaScalars, PlasmaWork, make_blocked_plasma = (
    _scan.BlockedPlasma,
    _scan.PlasmaScalars,
    _scan.PlasmaWork,
    _scan.make_blocked_plasma,
)
make_tiled_plasma = _tiles.make_tiled_plasma

__all__ = [
    "BlockedPlasma",
    "PlasmaScalars",
    "PlasmaWork",
    "ScanKernels",
    "historical_column",
    "inputs",
    "make_blocked_plasma",
    "make_scan_kernels",
    "make_tiled_plasma",
    "reference_arguments",
]


def inputs(samples, nodes, preionisation):
    axis = np.arange(samples, dtype=float)
    field = np.tile(3 * np.cos(axis / 3.1), (nodes, 1))
    field *= np.linspace(0.5, 1.5, nodes)[:, None]
    field[:, ::7] = 0.0
    if samples > 1:
        field[:, 1] = 1e-170
    coefficients = np.zeros((4, 2))
    coefficients[2] = 2.0
    coefficients[3] = np.log(4.0) + 2 * np.arange(2) * math.log(2.0)
    table = RateArrays(coefficients, 0.0, math.log(2.0), 1.0, 4.0, 64.0)
    scalars = PlasmaScalars(1e-4, 3.0, 0.2, 2.0, preionisation, 1e-3)
    coefficients.flags.writeable = False
    return field, table, scalars, np.linspace(0.1, 1.0, samples)


def reference_arguments(field, table, scalars, window):
    nodes, samples = field.shape
    return (
        field,
        np.full_like(field, np.nan),
        np.zeros((nodes, 2 * samples)),
        window,
        table,
        np.ones(samples, complex),
        scalars.kerr_factor,
        0.0,
        0.0,
        0.0,
        scalars.charge_mass_ratio,
        scalars.ionisation_potential,
        scalars.density,
        scalars.preionisation,
        scalars.dt,
        np.full((nodes, 2), -1.0),
    )


def historical_column(spec):
    """Return the NumPy trace for the rejected historical polarisation prototypes.

    SEP 27's production column returns J, whereas these retained SEP 25
    experiments integrate J to physical P. Their independent NumPy trace and
    original rounding envelope remain the comparison; production no longer
    carries an executable old plasma source.
    """
    bounds = _load("plasma_bounds")

    def evaluate(*args):
        field, out, drive, window, table = args[:5]
        samples = field.shape[-1]
        scalars = PlasmaScalars(
            args[6], args[10], args[11], args[12], args[13], args[14]
        )
        rates = np.fromiter(
            (PLAIN.rate_at(float(abs(value)), table) for value in field.flat),
            dtype=float,
            count=field.size,
        ).reshape(field.shape)
        budget, _ = bounds.column_budget(
            field, rates, window, scalars, raman=spec.raman
        )
        out[:] = budget["serial_trace"]
        if spec.raman:
            drive[:, :samples] = field * field
        if spec.stash:
            args[15][:, 0] = budget["serial_fraction"]
            args[15][:, 1] = rates.max(axis=-1)

    return evaluate
