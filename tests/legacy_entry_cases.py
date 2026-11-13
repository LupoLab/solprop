# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The legacy entry points' runs: ten fixed steps of a capillary and of a GNLSE fibre.

A wall is an opt-in leaf, so a run that declares none must be the run it was before the
leaf existed: ``tests/driver/test_coherent_capillary.py`` holds each run's final plane
to its record, and native's against the reference's within
:func:`legacy_parity_bound`. Both runs replay one fixed schedule at the Adams method's
fixed top order, so that a record is a record of one discrete march.
"""

import numpy as np

from bounds import fft_bound, gamma, step_evaluations
from coupled_flow_cases import GRID, STEPS, WEAK_COLLAR
from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import FIXED_ORDER
from solprop.driver import Simulation, prop_capillary, prop_gnlse
from solprop.grid import EnvelopeGrid
from solprop.stepper import DOPRI5, FixedSchedule

LEGACY_LENGTH = 2e-4
"""Length of the legacy-identity case, m: ten fixed steps of 20 micrometres."""

LEGACY_SCHEDULE = FixedSchedule(np.full(STEPS, LEGACY_LENGTH / STEPS))
"""The schedule both legacy runs replay, so that a record is a record of one march."""


def legacy_capillary_keywords(backend):
    """Return the keywords of the ten-step two-mode capillary run of the pre-wall route.

    At the Adams method's fixed top order, so that no order choice is left for
    rounding to move: a run is the recorded run's discrete march.
    """
    return {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-8,
        "modes": 2,
        "n_save": 2,
        "backend": backend,
        "dz_max": LEGACY_LENGTH / STEPS,
        "dz_init": LEGACY_LENGTH / STEPS,
        "schedule": LEGACY_SCHEDULE,
        "integrator": FIXED_ORDER,
        "plasma": False,
        "raman": False,
    }


def legacy_gnlse_keywords(backend):
    """Return the keywords of the ten-step GNLSE run of the soliton case.

    On the grid it was recorded on and without a collar, as the run was before
    the default collar: the record is of that march.
    """
    grid = EnvelopeGrid.from_limits(
        LEGACY_LENGTH, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return {
        "wavelength": 835e-9,
        "grid": grid,
        "absorber": bare_absorber(grid),
        "width": 100e-15,
        "peak_power": 9.8,
        "shape": "sech",
        "n_save": 2,
        "backend": backend,
        "dz_max": LEGACY_LENGTH / STEPS,
        "dz_init": LEGACY_LENGTH / STEPS,
        "schedule": LEGACY_SCHEDULE,
        "integrator": FIXED_ORDER,
    }


GNLSE_FIBRE = (0.11, LEGACY_LENGTH, [0.0, 0.0, -1.183e-26])
"""The legacy GNLSE run's ``gamma`` (1/(W m)), length (m) and Taylor betas."""


def legacy_capillary(backend):
    """Return a ten-step two-mode capillary run of the pre-wall route."""
    return prop_capillary(
        125e-6,
        LEGACY_LENGTH,
        "He",
        1.0,
        **{"noise": None, **legacy_capillary_keywords(backend)},
    )


def legacy_gnlse(backend):
    """Return a ten-step GNLSE run of the soliton case."""
    return prop_gnlse(*GNLSE_FIBRE, **{"noise": None, **legacy_gnlse_keywords(backend)})


def legacy_simulation(entry):
    """Return the simulation a legacy entry point runs, for its rounding budget."""
    if entry == "prop_capillary":
        return Simulation.from_capillary(
            125e-6,
            LEGACY_LENGTH,
            "He",
            1.0,
            **{"noise": None, **legacy_capillary_keywords("numpy_ref")},
        )
    return Simulation.from_gnlse(
        *GNLSE_FIBRE, **{"noise": None, **legacy_gnlse_keywords("numpy_ref")}
    )


def legacy_parity_bound(samples: int, simulation) -> float:
    """Relative bound on a legacy run's final field between two backends.

    Every evaluation of each of the ten steps -- DOPRI5's seven stages, or the Adams
    march's one, which reaches the step through its predictor's and corrector's
    weights, counted by their absolute sum (``bounds.step_evaluations``) --
    re-evaluates the nonlinear term
    from a field the two backends formed in different orders: one FFT round trip each
    way (``fft_bound``) and the modal synthesis and projection, whose sums run over at
    most the transform's nodes (``gamma`` of the time samples, an over-count of any
    node set here). Errors of the stages add; the linear step is a unit-modulus phase
    and the case's nonlinear phase is far below one radian, so nothing amplifies them.
    Derived before the local-build comparison was run.
    """
    per_stage = 2 * fft_bound(samples) + 2 * gamma(samples)
    evaluations = max(DOPRI5.n_stages, step_evaluations(simulation._run_request()))
    return STEPS * evaluations * per_stage
