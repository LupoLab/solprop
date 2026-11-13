# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise tensor histories through local profiles, spatial transforms and batches.

On the modal routes -- a mode whose direction depends on position, and two envelope
plasma populations with a Raman kernel each -- native's first stage and two fixed steps
against the reference's are the catalogue's rows ``ladder modal raman ...``
(``backends/ladder_rows.py``); on the radial and Cartesian beams, the actual-width tails
and the two populations, they are the rows ``vector raman tails ...`` and ``vector
raman populations ...`` (``backends/vector_raman_rows.py``). Each backend's runs meeting
their tolerance are held here, with the Cartesian geometry's refusal of varying
populations.
"""

import pytest

from backends.host_cases_envelope_plasma import geometry_options
from backends.ladder_rows import DIRECTIONS
from backends.test_batch_responses import compare
from sessions.runs import run as session_run
from solprop.backends.session import ExecutionOptions
from solprop.exceptions import TranslationError
from vector_raman_cases import (
    CHUNK,
    direction_translation,
    raman_populations,
    spatial_tails,
    vector_translation,
)


def meets_its_tolerance(backend_class, problem, run_request, grid, options):
    """Assert one backend's run of the problem lands at its end within tolerance."""
    owner = backend_class(problem, run_request, grid=grid, options=options)
    ran = session_run(owner, problem, run_request, owns=True)
    assert ran.final.z == grid.length
    assert ran.met_tolerance


@pytest.mark.medium
@pytest.mark.parametrize("geometry", ["radial", "free"])
@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
def test_spatial_vector_raman_and_actual_width_tails_meet_their_tolerance(
    geometry, family, spatial_backend_class
):
    problem, run_request, grid = spatial_tails(geometry, family)
    assert problem.transform.n_nodes % 3 != 0
    meets_its_tolerance(
        spatial_backend_class,
        problem,
        run_request,
        grid,
        ExecutionOptions(node_chunk=3),
    )


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
@pytest.mark.parametrize(
    ("geometry", "varying_density"),
    [("radial", False), ("radial", True), ("free", False)],
)
def test_vector_raman_with_two_envelope_plasma_populations_meets_its_tolerance(
    geometry, varying_density, spatial_backend_class
):
    problem, run_request, grid = raman_populations(varying_density, geometry)
    meets_its_tolerance(
        spatial_backend_class,
        problem,
        run_request,
        grid,
        geometry_options(geometry),
    )


def test_a_cartesian_beam_refuses_varying_raman_populations_by_name():
    with pytest.raises(
        TranslationError, match="varying density profiles are unsupported"
    ):
        raman_populations(True, "free")


@pytest.mark.medium
@pytest.mark.parametrize("varying_density", [False, True])
def test_modal_raman_populations_meet_their_tolerance(
    varying_density, spatial_backend_class
):
    problem, run_request, grid = raman_populations(varying_density)
    meets_its_tolerance(
        spatial_backend_class, problem, run_request, grid, geometry_options("modal")
    )
