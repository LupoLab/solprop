# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The line (x-only) free space on every backend: the ``"free"`` kind with ``n_y = 1``.

:class:`solprop.grid.LineGrid` runs through the Cartesian seam. These parity tests were
written first, to measure whether every backend runs it unchanged: ``numpy_ref`` and
``numba_cpu`` did, and native CPU needed only its ``n_y >= 2`` guard relaxed (no schema
change). The bounds are the plane's (``tests/backends/test_free_ladders.py``), whose FFT
depth counts ``log2(n_y) = 0`` for the line.
"""

import dataclasses
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.test_free_ladders import amplitude_bound
from backends.test_radial_nonlinear import (
    CHI3,
    LENGTH,
    QuadraticRate,
    derivative,
    make_problem,
)
from bounds import gamma
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.grid import EnvelopeGrid, FieldGrid, LineGrid
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.translate.ionisation import build_rate_table

# Ten steps is the qualification scope's propagation (AGENTS.md); zero is one
# nonlinear evaluation, one is a single step.
STEPS = (0, 1, 10)


@pytest.fixture(
    params=[
        "field_kerr",
        "envelope_kerr",
        "field_plasma",
        "envelope_raman",
        "field_kerr_collar",
    ]
)
def line_case(request):
    family = request.param
    constructor = (
        FieldGrid.from_limits
        if family.startswith("field")
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    # Resolve the held-current collar while preserving the non-plasma fixtures.
    window = 200e-15 if family == "field_plasma" else 150e-15
    grid = constructor(LENGTH, 800e-9, (600e-9, 1.2e-6), window)
    # The collar family puts the beam's wings in a 0.3 collar, so the window and
    # the per-column scaling are exercised along x; the others have no collar.
    space = LineGrid(
        3e-4 if family.endswith("collar") else 4e-4,
        24,
        window_factor=0.3 if family.endswith("collar") else 0.0,
    )
    # Energy per metre of y: 1e-7 J over the plane's 80 um beam is about 1e-3 J/m
    # over the same beam in x, the same peak intensity to a factor of order one.
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-3),
        80e-6,
        jones=(1,) if family.endswith("raman") else (1, 0.5j),
    )
    density = 1.0
    if family.endswith("plasma"):
        responses = [
            PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10)))
        ]
        density = 1e22
    elif family.endswith("raman"):
        responses = [
            RamanResponse(
                (DampedOscillator(1e-8, 2e14, undamped_dephasing()),), thg=False
            )
        ]
    else:
        responses = [KerrResponse(CHI3, thg=False)]
    problem = make_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        responses,
        density=density,
    )
    plasma_names = (
        ("electron_density", "peak_ionisation_rate")
        if family.endswith("plasma")
        else ()
    )
    problem = dataclasses.replace(
        problem,
        # Force a remainder chunk, independently of the memory-derived default.
        transform=dataclasses.replace(problem.transform, node_chunk=7),
        stats_spec=StatsSpec(names=plasma_names, periodic=plasma_names),
    )
    return family, grid, space, problem


def test_line_problem_is_the_free_kind_with_one_y_column(line_case):
    _, _, space, problem = line_case
    geometry = problem.transform.geometry
    assert geometry is not None
    assert geometry.kind == "free"
    assert geometry.shape == (space.n_x, 1)
    assert geometry.axes[1].tolist() == [0.0]
    assert geometry.spectral_axes[1].tolist() == [0.0]
    np.testing.assert_array_equal(geometry.spatial_weights, space.spatial_weights)
    assert problem.linear.separable is not None
    np.testing.assert_array_equal(problem.linear.separable.kperp_sq, space.k_x**2)


@pytest.mark.parametrize("steps", STEPS)
def test_line_nonlinear_ladder(line_case, spatial_backend_class, steps):
    family, grid, space, problem = line_case
    bound = amplitude_bound(family, grid, space, spatial_backend_class.name, steps)
    assert bound > 0
    if steps == 0:
        reference = derivative(NumpyRefBackend, problem, grid)
        actual = derivative(spatial_backend_class, problem, grid)
    else:
        schedule = problem.stepper.schedule
        assert schedule is not None
        problem = dataclasses.replace(
            problem,
            stepper=dataclasses.replace(
                problem.stepper,
                schedule=type(schedule).uniform(0, LENGTH, LENGTH / steps),
            ),
        )
        values = []
        for implementation in (NumpyRefBackend, spatial_backend_class):
            backend = implementation(problem, grid=grid)
            try:
                backend.set_initial(problem.initial.field0, None)
                backend.propagate_to(LENGTH)
                values.append(backend.observe().require_field().copy())
            finally:
                backend.close()
        reference, actual = values
    scale = np.max(np.abs(reference))
    assert scale > 0
    assert_allclose(actual, reference, rtol=0, atol=bound * scale)


@pytest.mark.parametrize("steps", [0, 10])
def test_line_statistics_fresh_and_cached(line_case, spatial_backend_class, steps):
    family, grid, space, problem = line_case
    # The plane's statistics bound (test_free_ladders): twice the amplitude ladder
    # for squared fields, plus the column and temporal reductions.
    bound = 2 * amplitude_bound(family, grid, space, spatial_backend_class.name, steps)
    bound += gamma(128 * space.n_columns + 32 * grid.n_to)
    names = (
        "energy",
        "peak_intensity",
        "on_axis_intensity",
        "on_axis_spectrum",
        "beam_radius",
        "fwhm_r",
    )
    if family.endswith("plasma"):
        names += ("electron_density", "peak_ionisation_rate")
    spec = StatsSpec(names=names)
    rows = []
    for implementation in (NumpyRefBackend, spatial_backend_class):
        backend = implementation(problem, grid=grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            if steps:
                backend.propagate_to(LENGTH)
            rows.append(backend.stats(spec))
            state = backend.observe().require_field().copy()
            backend.set_initial(state, None)
            fresh = backend.stats(spec)
            for name in names:
                assert_allclose(
                    fresh[name],
                    rows[-1][name],
                    rtol=0,
                    atol=bound * np.max(np.abs(rows[-1][name])),
                )
        finally:
            backend.close()
    for name in names:
        scale = np.max(np.abs(rows[0][name]))
        assert scale > 0, name
        assert_allclose(rows[1][name], rows[0][name], rtol=0, atol=bound * scale)
