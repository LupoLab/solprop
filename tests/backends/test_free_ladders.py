# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Cartesian route's self-consistency: statistics read afresh, and the families.

Native against ``numpy_ref`` on the Cartesian ladders' cases is the case catalogue's
(``tests/backends/free_rows.py``); what stays here holds each backend to itself: a
plane's statistics read afresh from its state, the analytic family's one channel, and
the field and envelope families and the scalar and vector states on one matched grid.
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.free_rows import (
    CARTESIAN,
    FAMILIES,
    PLANE_STATISTICS,
    PLASMA_NAMES,
    REMAINDER,
    analytic_plasma_case,
    cartesian_case,
    free_amplitude_bound,
)
from backends.test_radial_nonlinear import (
    CHI3,
    LENGTH,
    make_problem,
    matched_parity_grids,
)
from bounds import gamma, ladder_bound
from sessions.executors import IDENTITY
from sessions.runs import launch_statistics, launched, run
from solprop.constants import C
from solprop.grid import FreeGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.stats import StatsSpec
from spatial_records import record_metric


@pytest.fixture(params=FAMILIES)
def free_case(request):
    """The Cartesian ladders' case of one response family (``free_rows.py``)."""
    return request.param, *cartesian_case(request.param)


@pytest.mark.parametrize("steps", [0, 8])
def test_cartesian_statistics_read_afresh_are_the_session_s(
    free_case, spatial_backend_class, steps
):
    """A plane's statistics, read afresh from the state it reached, are its own.

    The statistics a session's plane reports with the state it reached (the launch
    plane's at no steps), then those of that state loaded afresh, on each backend.
    Native against the reference is the catalogue's (``free_rows.py``).
    """
    family, grid, space, problem, run_request = free_case
    # Fresh statistics use the same initial state on both backends. Propagated
    # statistics add twice the amplitude ladder envelope for squared fields.
    bound = 2 * free_amplitude_bound(
        family, grid, space, spatial_backend_class.name, steps
    )
    bound += gamma(128 * space.n_columns + 32 * grid.n_to)
    names = PLANE_STATISTICS + (PLASMA_NAMES if family.endswith("plasma") else ())
    run_request = run_request.for_run(stats=StatsSpec(names=names))
    backend = spatial_backend_class(problem, run_request, grid=grid, options=REMAINDER)
    try:
        if steps:
            plane = run(backend, problem, run_request).final
            assert plane.z == LENGTH
            state = plane.field
            cached = plane.stats
        else:
            with launched(backend, problem, run_request) as (session, launch):
                state = np.array(session.export(launch.ticket, IDENTITY))
                cached = {name: np.array(value) for name, value in launch.stats.items()}
        fresh = launch_statistics(backend, problem, run_request, field=state)
        for name in names:
            assert_allclose(
                fresh[name],
                cached[name],
                rtol=0,
                atol=bound * np.max(np.abs(cached[name])),
            )
    finally:
        backend.close()


def test_the_analytic_family_s_one_channel_is_the_whole_electron_density(
    spatial_backend_class,
):
    """One-channel electron density on the analytic family is its species' sum.

    Native's statistics against the reference's are the catalogue's
    (``free_rows.analytic_plasma_case``).
    """
    built = analytic_plasma_case()
    problem, run_request, grid = built.problem, built.run_request, built.grid
    backend = spatial_backend_class(
        problem, run_request, grid=grid, options=built.options
    )
    try:
        # The launch plane's statistics, in the backend's own session.
        row = launch_statistics(backend, problem, run_request)
    finally:
        backend.close()
    bound = gamma(128 * CARTESIAN.n_columns + 32 * grid.n_to)
    assert_allclose(
        row["electron_density"],
        np.sum(row["species_electron_density"], axis=0),
        rtol=0,
        atol=bound * np.max(np.abs(row["electron_density"])),
    )


@pytest.mark.parametrize("steps", [1, pytest.param(8, marks=pytest.mark.medium)])
def test_cartesian_field_envelope_and_scalar_vector_parity(
    spatial_backend_class, steps, request
):
    field_grid, envelope_grid, carrier_index = matched_parity_grids()
    carrier = field_grid.omega[carrier_index]
    space = FreeGrid(3e-4, 12, 4e-4, 16, collar_fraction=0)
    waist = 80e-6
    spatial = np.fft.fft2(
        (2e8 * np.exp(-((space.r / waist) ** 2))).reshape(space.n_y, space.n_x)
    ).ravel()
    duration = 30e-15 / math.sqrt(2 * math.log(2))
    temporal = np.exp(-((field_grid.t_fine / duration) ** 2))
    carrier_phase = (-1) ** carrier_index
    # Envelope and field share the temporal period, exact carrier-bin shift and
    # physical band. The real-field opposite-frequency tail is bounded by the
    # Gaussian Fourier tail plus the finite temporal wall, before either run.
    tail = 4 * math.exp(-((duration * carrier / 2) ** 2))
    tail += 4 * math.exp(-((field_grid.t_fine[0] / duration) ** 2))
    evaluations = 7 * steps
    rounding = evaluations * (
        16 * np.finfo(float).eps * math.ceil(math.log2(field_grid.n_to))
        + 2 * gamma(space.n_columns)
    )
    rounding += gamma(64) * carrier / C * LENGTH + gamma(16 * field_grid.n_to)
    bound = ladder_bound(
        spatial_backend_class.name, rounding + tail, field_grid.n_to, evaluations
    )
    rows = []
    statistics = []
    for grid, n_pol in (
        (field_grid, 1),
        (envelope_grid, 1),
        (field_grid, 2),
        (envelope_grid, 2),
    ):
        if grid.kind == "field":
            samples = (
                temporal
                * np.cos(2 * np.pi * carrier_index * np.arange(grid.n_to) / grid.n_to)
                * carrier_phase
            )
            spectrum = np.fft.rfft(samples)[grid.crop_index] * grid.crop_scale
        else:
            spectrum = np.fft.fft(temporal)[grid.crop_index] * grid.crop_scale
        start = np.zeros((space.n_columns, n_pol, grid.n_omega), complex)
        start[:, 0] = spatial[:, None] * spectrum
        problem, run_request = make_problem(
            grid, space, start, [KerrResponse(CHI3, thg=False)], steps=steps
        )
        run_request = run_request.for_run(
            stats=StatsSpec(names=("energy", "on_axis_intensity", "peak_intensity"))
        )
        plane = run(
            spatial_backend_class(problem, run_request, grid=grid),
            problem,
            run_request,
            owns=True,
        ).final
        assert plane.z == LENGTH
        state = plane.field
        if n_pol == 2:
            np.testing.assert_array_equal(state[:, 1], np.zeros_like(state[:, 1]))
        indices = np.rint((grid.omega - carrier) / grid.d_omega).astype(int)
        active = grid.active_mask
        ordering = np.argsort(indices[active])
        values = state[:, 0, active][:, ordering] / grid.n_t
        if grid.kind == "field":
            values *= 2 * carrier_phase
        rows.append((indices[active][ordering], values))
        statistics.append(plane.stats)
    for label, (indices, values) in zip(
        ("envelope scalar", "field vector", "envelope vector"), rows[1:], strict=True
    ):
        np.testing.assert_array_equal(indices, rows[0][0])
        assert_allclose(
            values, rows[0][1], rtol=0, atol=bound * np.max(np.abs(rows[0][1]))
        )
        record_metric(
            request,
            f"Cartesian field-scalar parity with {label} (relative)",
            steps,
            np.max(np.abs(values - rows[0][1])) / np.max(np.abs(rows[0][1])),
            bound,
            spatial_backend_class.name,
        )
    for result in statistics[1:]:
        for name, reference in statistics[0].items():
            assert_allclose(result[name], reference, rtol=2 * bound, atol=0)
