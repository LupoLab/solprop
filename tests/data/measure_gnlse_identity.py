# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record a millimetre of both ``prop_gnlse`` routes, for a cross-tree identity.

The companion of :mod:`measure_power_identity`, for the entry point that has no
modes and no fill: it runs the **Taylor** route (a declared ``gamma`` and
``betas``) and the **mapped** route (``fibre=`` and ``n2=``, which goes through
``_gnlse_mode``), and saves both runs' fields.

Run it in separate processes with ``PYTHONPATH`` selecting each solprop source and
the matching Kokkos wheel, then compare the saved fields; ``--backend`` takes a
registry name, so this covers every backend the tree offers. Both routes step
adaptively, so two trees' runs part once a rounding difference changes a step: compare
each against a tighter-tolerance run of the same route, at an accuracy envelope fixed
in advance (``tests/bounds.py``'s ``AccuracyEnvelope``), not bit for bit.

Every number the two calls take is a literal here, so the measurement is
reproducible from this file alone: that is what makes the saved fields evidence.
"""

import argparse
import json
from pathlib import Path
from typing import cast

# Initialise Kokkos before importing the wheel-private pyFFTW OpenMP runtime, as
# `measure_power_identity` does; the default lane imports this module for its
# doctests and must not require the optional package.
try:
    # The optional package is out of tree and deliberately absent from the
    # static-analysis source, so the checker cannot resolve it on the base pair.
    import solprop_native as native  # pyright: ignore[reportMissingImports]
except ImportError:
    native = None

import numpy as np

import solprop
from solprop.driver import prop_gnlse
from solprop.modes import StepIndexMode
from solprop.output.memory import MemoryOutput

LENGTH = 1e-3
"""Millimetre, the length cap for a general modal measurement."""

LAMBDA0 = 835e-9
"""Reference and centre wavelength of both runs, m."""

GRID = {
    "wavelength": LAMBDA0,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
}
PULSE = {"width": 100e-15, "energy": 1e-9, "shape": "sech"}
RUN = {"n_save": 3, "noise": "input", "seed": 11, "progress": "none"}

GAMMA = 0.11
"""Nonlinear coefficient of the Taylor route, 1/(W m)."""

BETAS = (0.0, 0.0, -1.183e-26)
"""Taylor dispersion about ``wavelength``, s^n/m; Luna's soliton-fission case."""

N2 = 2.6e-20
"""Nonlinear index of the mapped route, m^2/W; fused silica."""

CORE_RADIUS = 2.5e-6
"""Core radius of the mapped route's step-index fibre, m."""


def runs(backend: str) -> list[np.ndarray]:
    """Return the saved fields of the two runs, in order.

    Parameters
    ----------
    backend : str
        Registry name of the backend to execute both runs on.

    Returns
    -------
    list of numpy.ndarray
        The Taylor route's saved fields and the mapped route's, each of shape
        ``(n_z, n_modes, n_pol, n_omega)`` in the square root of watts.
    """
    taylor = cast(
        "MemoryOutput",
        prop_gnlse(
            GAMMA,
            LENGTH,
            list(BETAS),
            shock=True,
            backend=backend,
            **{"noise": None, **GRID},
            **PULSE,
            **RUN,
        ),
    )
    mapped = cast(
        "MemoryOutput",
        prop_gnlse(
            None,
            LENGTH,
            fibre=StepIndexMode(CORE_RADIUS, profile_wavelength=LAMBDA0),
            n2=N2,
            shock=True,
            backend=backend,
            **{"noise": None, **GRID},
            **PULSE,
            **RUN,
        ),
    )
    return [
        np.ascontiguousarray(output.fields(), dtype=np.complex128)
        for output in (taylor, mapped)
    ]


def main() -> None:
    """Run both routes on one backend and write the fields and their record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    fields = runs(arguments.backend)
    np.savez(arguments.output, taylor=fields[0], mapped=fields[1])
    record = {
        "backend": arguments.backend,
        "solprop_source": solprop.__file__,
        "native_source": None if native is None else native.__file__,
        "length_m": LENGTH,
    }
    arguments.output.with_suffix(".json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    # Written, not printed: this module has no logger and the caller reads the
    # JSON beside the arrays.


if __name__ == "__main__":
    main()
