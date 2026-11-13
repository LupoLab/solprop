# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hollow planar (slab) guide's cases: an argon-filled gap, ten fixed steps.

Every run is ten fixed steps of a 10 cm argon-filled gap (2a = 127 um, Akturk et
al. 2009), under a collar weakened far below every bound here so that the explicit
nonlinear-scheme absorber does not cap the step (ADR-025, the pattern of
``tests/driver/test_coherent_capillary.py``). Energies are in **J/m**: the slab is
uniform in y. ``tests/driver/test_prop_slab.py`` runs them.
"""

import math
from typing import Any

import numpy as np

from bounds import fft_bound, gamma
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid
from solprop.modes import SlabMode, gaussian_coupling
from solprop.profiles import Linear
from solprop.stepper import DOPRI5
from solprop.stepper.schedule import FixedSchedule

LENGTH = 0.1
"""Guide length, m: B is a third of a radian at 0.5 J/m and dispersion negligible."""

STEPS = 10
"""The fixed schedule's steps: the ten-step rule."""

HALF_GAP = 63.5e-6
"""Akturk et al.'s 2a = 127 um, m."""

GRID = EnvelopeGrid.from_limits(LENGTH, 800e-9, (650e-9, 1000e-9), 1.2e-12)
"""One envelope grid for every run, so that launches and bounds share it."""

COLLAR = absorber(GRID, boundary_length=1.0e3)
"""A collar a million times weaker than the default, present but inert (ADR-025)."""


def options(**changes: Any) -> dict[str, Any]:
    """Return the slab runs' keywords: a 100 fs, 0.5 J/m pulse over ten fixed steps.

    `changes` replace any of them.
    """
    return {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": COLLAR,
        "fwhm": 100e-15,
        "energy": 0.5,
        "envelope": True,
        "plasma": False,
        "raman": False,
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        **changes,
    }


def launched(simulation: Simulation, shares: np.ndarray) -> Simulation:
    """Replace a single-mode launch by the Gaussian's projection onto every mode."""
    pulse = simulation._field0[0]
    initial = np.stack([math.sqrt(share) * pulse for share in shares])
    replaced = simulation.with_input(initial)
    # The record describes the replacement spectrum, not the single-mode pulse.
    assert "pulse" not in replaced.configuration
    return replaced


def parity_bound(n_nodes: int) -> float:
    """Relative bound on a ten-step final field between two backends.

    The legacy capillary bound (``legacy_entry_cases.legacy_parity_bound``):
    each of the ten steps' seven DOPRI5 stages re-forms the nonlinear term from a field
    the backends built in different orders -- one FFT round trip each way and the
    modal synthesis and projection over at most the rule's nodes -- and the errors of
    the stages add, since the linear step is unit-modulus and the nonlinear phase stays
    below one radian. Written before the runs were compared.
    """
    return STEPS * DOPRI5.n_stages * (2 * fft_bound(GRID.n_to) + 2 * gamma(n_nodes))


def projected_multimode(
    taper: bool, backend: str = "numpy_ref"
) -> tuple[Simulation, np.ndarray]:
    """Return TE1/TE3/TE5 (TE1/TE3 tapered) and the 0.735a Gaussian's shares of them.

    A centred beam reaches the odd orders only. The simulation launches the
    single-mode pulse; :func:`launched` replaces it by the Gaussian's projection onto
    every mode, at the returned coupled shares. The modes' 40 fs group-delay walk-off,
    which the inert collar cannot stop from wrapping, makes the constructor warn; ten
    steps of a 100 fs pulse in a 1.2 ps window never reach the edge. A taper couples
    TE1 to TE3 through ``<psi_m|d_z psi_n>``, which this route omits and says so once
    the launch is replaced. Returns the simulation on `backend` and the shares.
    """
    gap = Linear(HALF_GAP, 50e-6, LENGTH) if taper else HALF_GAP
    orders = (1, 3) if taper else (1, 3, 5)
    modes = [SlabMode(gap, cladding=1.45, order=order) for order in orders]
    shares = gaussian_coupling(modes, 0.735 * HALF_GAP)
    simulation = Simulation.from_waveguide(
        modes,
        LENGTH,
        "Ar",
        1.0,
        **({"noise": None, **options(backend=backend)}),
    )
    return simulation, shares
