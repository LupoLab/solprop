# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise tensor histories through local profiles, spatial transforms and batches.

On the modal routes -- a mode whose direction depends on position, and two envelope
plasma populations with a Raman kernel each -- native's first stage and two fixed steps
against the reference's are the catalogue's rows ``ladder modal raman ...``
(``backends/ladder_rows.py``); each backend's runs meeting their tolerance are held
here.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.host_cases_envelope_plasma import geometry_options
from backends.ladder_rows import DIRECTIONS
from backends.test_batch_responses import compare
from backends.test_radial_nonlinear import derivative
from former_raman import FORMER_H2, former_response
from sessions.runs import run as session_run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import ExecutionOptions
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble
from vector_raman_cases import (
    CHUNK,
    direction_translation,
    raman_populations,
    vector_translation,
)


def run(backend_class, problem, run_request, grid, options=None):
    """Return one source evaluation and a complete fixed-schedule run, the latter
    through the backend's own session."""
    rhs = derivative(backend_class, problem, run_request, grid, options)
    owner = backend_class(problem, run_request, grid=grid, options=options)
    ran = session_run(owner, problem, run_request, owns=True)
    assert ran.final.z == grid.length
    assert ran.met_tolerance
    return rhs, ran.final.field


def check_run(backend_class, problem, run_request, grid, options=None):
    reference = run(NumpyRefBackend, problem, run_request, grid, options)
    actual = run(backend_class, problem, run_request, grid, options)
    # Temporal transforms, Cartesian contraction and transverse reductions per
    # stage; 14 stages cover the two-step complete run and both compared paths.
    bound = 256 * (grid.n_to + problem.transform.n_nodes) * 14 * np.finfo(float).eps
    assert 0 < bound < 1e-6
    for got, expected in zip(actual, reference, strict=True):
        assert np.all(np.isfinite(got))
        assert_allclose(got, expected, rtol=0, atol=bound * np.max(abs(expected)))


@pytest.mark.medium
@pytest.mark.parametrize("geometry", ["radial", "free"])
@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
def test_spatial_vector_raman_and_actual_width_tails(
    geometry, family, spatial_equivalence_backend_class
):
    length = 1e-5
    constructor = EnvelopeGrid if family == "envelope" else FieldGrid
    grid = constructor.from_limits(length, 800e-9, (500e-9, 1.2e-6), 100e-15)
    space = (
        RadialGrid(6e-4, 8, collar_fraction=0.0)
        if geometry == "radial"
        else FreeGrid(3e-4, 4, 4e-4, 4, collar_fraction=0.0)
    )
    initial = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-7),
        80e-6,
        jones=(1, 0.35j),
    ).build(grid, space, medium=SpatialMedium())
    problem, run_request = assemble(
        grid,
        None,
        [
            KerrResponse(0.0, thg=family == "field_real"),
            former_response(FORMER_H2, thg=family == "field_real"),
        ],
        initial,
        density=1e25,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=length / 2,
        dz_init=length / 2,
        schedule=FixedSchedule.uniform(0.0, length, length / 2),
    )
    assert problem.transform.n_nodes % 3 != 0
    check_run(
        spatial_equivalence_backend_class,
        problem,
        run_request,
        grid,
        ExecutionOptions(node_chunk=3),
    )


def meets_its_tolerance(backend_class, problem, run_request, grid, options):
    """Assert one backend's run of the problem lands at its end within tolerance."""
    owner = backend_class(problem, run_request, grid=grid, options=options)
    ran = session_run(owner, problem, run_request, owns=True)
    assert ran.final.z == grid.length
    assert ran.met_tolerance


@pytest.mark.medium
@pytest.mark.parametrize("kind", sorted(DIRECTIONS))
def test_raman_acts_on_synthesised_position_dependent_direction(
    kind, spatial_backend_class
):
    problem, run_request, grid = direction_translation(kind, DIRECTIONS[kind])
    meets_its_tolerance(
        spatial_backend_class,
        problem,
        run_request,
        grid,
        ExecutionOptions(node_chunk=5),
    )


@pytest.mark.medium
@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
def test_vector_fixed_batches_preserve_modified_noise_members(family):
    # The reference's lockstep group; the native group runs each member as its own
    # native session (SEP 47 phase 5), with no shared arithmetic to check.
    problem, run_request, grid, _ = vector_translation(family, steps=2)
    compare(problem, run_request, grid, "numpy_ref", options=CHUNK)


@pytest.mark.medium
@pytest.mark.parametrize("geometry", ["radial", "free"])
@pytest.mark.parametrize("varying_density", [False, True])
def test_vector_raman_with_two_envelope_plasma_populations(
    geometry, varying_density, spatial_equivalence_backend_class
):
    if geometry == "free" and varying_density:
        with pytest.raises(
            TranslationError, match="varying density profiles are unsupported"
        ):
            raman_populations(varying_density, geometry)
        return
    problem, run_request, grid = raman_populations(varying_density, geometry)
    check_run(
        spatial_equivalence_backend_class,
        problem,
        run_request,
        grid,
        geometry_options(geometry),
    )


@pytest.mark.medium
@pytest.mark.parametrize("varying_density", [False, True])
def test_modal_raman_populations_meet_their_tolerance(
    varying_density, spatial_backend_class
):
    problem, run_request, grid = raman_populations(varying_density)
    meets_its_tolerance(
        spatial_backend_class, problem, run_request, grid, geometry_options("modal")
    )
