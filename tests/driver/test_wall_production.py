# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The wall note's pinned case on the production route.

The campaign itself is a development study run by hand, not shipped. This pins one
small admitted case of it -- a 150 micrometre evacuated bore at 3 micrometres,
cladding 1.3, the 80 micrometre Gaussian projected onto ten ``HE_1m`` channels -- over
ten steps of the production route, against the FEM numbers the study committed to
``tests/data/wall_production_reference.json``.
"""

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import Simulation
from solprop.exceptions import GrazingIncidenceWarning
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall, wall_operator
from solprop.stepper import FixedSchedule
from solprop.translate.linop import delta_phi

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "data" / "wall_production_reference.json").read_text()
)
"""The study's pinned case: launch, schedule, FEM power and projections."""

RADIUS = 150e-6
AMPLITUDE = 1e-9
"""Launch amplitude in sqrt(W): far below any response, and there is no fill."""

CONDITION = 100.0
"""The step's certified ``norm(U) norm(U^-1)``: the study caps the step at it."""


def pinned_final():
    """Run the pinned case's ten steps and return the optics-convention coefficients.

    The same declaration as the study's: the narrowest admissible field grid about
    3 micrometres, an evacuated helium capillary, the scalar wall at its default
    limits, one bin carrying the projected Gaussian. The common co-moving phase is
    removed as the study removes it, so the result is the paraxial field in the
    frame of the in-core wavenumber, which is the FEM's.
    """
    wavelength = REFERENCE["wavelength"]
    count = REFERENCE["count"]
    length = REFERENCE["length"]
    limits = (wavelength / 1.001, wavelength * 1.001)
    # The former geometry, on which the study chose its launch bin.
    grid = FieldGrid.from_limits(
        length, wavelength, limits, 50.0 * limits[0] / C, **LEGACY_GRID
    )
    index = REFERENCE["launch_bin"]
    simulation = Simulation.from_capillary(
        RADIUS,
        length,
        "He",
        0.0,
        lambda0=wavelength,
        grid=grid,
        absorber=absorber(grid, boundary_length=1e12),
        fwhm=30e-15,
        energy=1e-12,
        modes=count,
        wall=CoherentWall(),
        cladding=complex(REFERENCE["cladding"]),
        plasma=False,
        raman=False,
        backend="numpy_ref",
        dz_max=REFERENCE["dz"],
        dz_init=REFERENCE["dz"],
        schedule=FixedSchedule(np.full(REFERENCE["steps"], REFERENCE["dz"])),
        n_save=2,
        noise=None,
    )
    problem = simulation.translate()
    launch = np.array(REFERENCE["launch"])
    state = np.zeros_like(problem.initial.field0)
    state[0, :, 0, index] = AMPLITUDE * launch
    problem = replace(
        problem, initial=replace(problem.initial, field0=state, noise0=None)
    )
    driver = NumpyRefBackend(problem, grid=grid)
    driver.set_initial(problem.initial.field0, None)
    driver.propagate_to(length)
    final = np.conj(driver.observe().require_field()[0, :, 0, index]) / AMPLITUDE
    driver.close()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    span = float(problem.linear.phi_breakpoints[-1])
    rate = complex(delta_phi(problem.linear, 0.0, span)[0, 0, 0, index]) / span
    single = np.zeros_like(grid.active_mask)
    single[index] = True
    block = wall_operator(
        simulation.modes, grid.omega, active=single, validity_limit=np.inf
    )
    frame = np.conj(flow.generator[0, 0, 0, 0, index] + rate) - complex(
        block.generator[0, 0, index]
    )
    return final * np.exp(-frame * length), launch


@pytest.mark.medium
@pytest.mark.filterwarnings(
    f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
)
# A single excited bin is a continuous wave: nothing walks off the window, so the
# absorber is made negligible on purpose and its walk-off warning says so.
@pytest.mark.filterwarnings("ignore:the temporal absorber attenuates")
def test_the_production_wall_meets_the_fem_on_the_pinned_wall_note_case():
    """Ten steps of the production wall against the committed FEM numbers.

    Bounds, stated first. **Acceptance**: the power
    departs from the FEM's by at most 5e-4 of the launched power, and the field's
    FEM-mass-weighted L2 error is at most 1 % of the FEM field's norm; these are the
    targets set in advance, not fitted to the result. **Regression**: the same
    declaration reproduces the study's own final coefficients to
    ``steps x 6 stages x N x eps x kappa = 10 x 6 x 10 x 2.2e-16 x 100 = 1.3e-11`` of
    the launch norm -- one rounding per channel of each stage's Schur action,
    amplified at most by the certified condition ``kappa`` of that action -- so a
    changed operator, frame or amplitude map moves it by far more.
    """
    final, launch = pinned_final()
    study = np.array(REFERENCE["study_final"]["real"]) + 1j * np.array(
        REFERENCE["study_final"]["imag"]
    )
    bound = REFERENCE["steps"] * 6 * REFERENCE["count"] * 2.2e-16 * CONDITION
    assert np.max(abs(final - study)) <= bound * np.linalg.norm(launch)

    fem_power = REFERENCE["fem_power"]
    projection = np.array(REFERENCE["fem_projection"]["real"]) + 1j * np.array(
        REFERENCE["fem_projection"]["imag"]
    )
    gram = np.array(REFERENCE["fem_gram"])
    power_error = abs(float(np.sum(abs(final) ** 2)) - fem_power)
    squared = (
        fem_power
        - 2 * np.real(np.vdot(final, projection))
        + np.real(np.vdot(final, gram @ final))
    )
    field_l2 = math.sqrt(max(squared, 0.0) / fem_power)
    assert power_error <= 5e-4
    assert field_l2 <= 1e-2
    # The FEM comparison is recorded, not only passed: the study's own values. A
    # coefficient change of `delta` (the regression bound above) moves the power by
    # at most 2|c| delta, about 3e-11, and the squared L2 norm by the same over P,
    # so the norm itself by that over twice its value (above 1e-3 here): 1.5e-8.
    assert power_error == pytest.approx(REFERENCE["study_power_error"], abs=1e-10)
    assert field_l2 == pytest.approx(REFERENCE["study_field_l2"], abs=1e-7)
