# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Input shot noise beside every coupled linear flow: the routes (SEP 37, section 6).

Shared by ``tests/driver/test_coupled_input_noise.py`` and the catalogue's rows
(``tests/backends/flow_rows.py``), so that no row imports a test module: the coherent
wall and the taper coupling of a capillary, the moving graded-index guide and the split
block, each a builder of the noise model, the launch energy and the backend, on a fixed
schedule of ten steps (:data:`ROUTES`).
"""

import dataclasses
from typing import Any

import numpy as np

from capability_routes import (
    GRIN_GRID,
    LENGTH,
    capillary_taper,
    capillary_wall,
    general_moving,
)
from modal_provider_cases import GENERAL_Z_END, general_grid, split_guide
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.stepper import FixedSchedule, RungeKutta

STEPS = 10
"""Fixed steps every run takes; ten lands on the taper's kinks at 0.3 and 0.7 L."""


PULSE_ENERGY = 1e-12
"""Launch energy, J: weak, so that the pulse's Kerr phase on the seed stays small."""


def _fixed(length: float, backend: str) -> dict[str, Any]:
    """Return the fixed schedule every run here takes; z = 0 is saved anyway."""
    return {
        "schedule": FixedSchedule(np.full(STEPS, length / STEPS)),
        "dz_max": length / STEPS,
        "dz_init": length / STEPS,
        "save_planes": [length],
        "backend": backend,
    }


def _wall(noise: str | None, energy: float, backend: str) -> Simulation:
    # The split-step collar is unconditionally stable, so the default collar sits
    # beside steps of 20 um without capping them.
    return capillary_wall(
        noise, energy=energy, absorber_scheme="split_step", **_fixed(LENGTH, backend)
    )


def _taper(noise: str | None, energy: float, backend: str) -> Simulation:
    return capillary_taper(
        noise, energy=energy, absorber_scheme="split_step", **_fixed(LENGTH, backend)
    )


def _moving(noise: str | None, energy: float, backend: str) -> Simulation:
    # The general route has no split-step collar; a collar a thousand times longer
    # than the default keeps the explicit Runge-Kutta march stable at these steps.
    simulation = general_moving(
        noise,
        energy=energy,
        integrator=RungeKutta(),
        absorber=absorber(GRIN_GRID, boundary_length=1.0e3),
        **_fixed(LENGTH, backend),
    )
    # The 100 fs window holds the band-limited pulse with 2e-3 of its energy in the
    # guards from the launch on, which the boundary monitor would stop; the subject
    # here is the draw's transport, not the window (SEP 21).
    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    return simulation


def _split(noise: str | None, energy: float, backend: str) -> Simulation:
    grid = general_grid()
    return Simulation.from_waveguide(
        (),
        GENERAL_Z_END,
        provider=split_guide(grid),
        grid=grid,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=energy,
        shape="gauss",
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=None if noise is None else 37,
        integrator=RungeKutta(),
        absorber=absorber(grid, boundary_length=1.0e3),
        **_fixed(GENERAL_Z_END, backend),
    )


ROUTES = {
    "capillary (coherent wall)": _wall,
    "capillary (taper coupling)": _taper,
    "general guide (moving)": _moving,
    "general guide (split block)": _split,
}
