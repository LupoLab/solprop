# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise tensor histories through local profiles, spatial transforms and batches."""

from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends import test_envelope_plasma_geometries as plasma_cases
from backends.test_batch_responses import compare
from backends.test_envelope_plasma_geometries import geometry_case
from backends.test_radial_nonlinear import derivative
from former_raman import FORMER_H2, FORMER_N2, former_response
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import KerrResponse
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble
from vector_raman_cases import vector_problem

envelope_mixture = plasma_cases.envelope_mixture


def run(backend_class, problem, grid):
    """Return one source evaluation and a complete fixed-schedule run."""
    rhs = derivative(backend_class, problem, grid)
    owner = backend_class(problem, grid=grid)
    try:
        owner.set_initial(problem.initial.field0, None)
        summary = owner.propagate_to(grid.length)
        assert summary.met_tolerance
        return rhs, owner.observe().require_field().copy()
    finally:
        owner.close()


def check_run(backend_class, problem, grid):
    reference = run(NumpyRefBackend, problem, grid)
    actual = run(backend_class, problem, grid)
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
        RadialGrid(6e-4, 8, window_factor=0.0)
        if geometry == "radial"
        else FreeGrid(3e-4, 4, 4e-4, 4, window_factor=0.0)
    )
    initial = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-7),
        80e-6,
        jones=(1, 0.35j),
    ).build(grid, space, medium=SpatialMedium())
    problem = assemble(
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
    problem = replace(problem, transform=replace(problem.transform, node_chunk=3))
    assert problem.transform.n_nodes % 3 != 0
    check_run(spatial_equivalence_backend_class, problem, grid)


@pytest.mark.medium
@pytest.mark.parametrize("kind, n", [("HE", 2), ("TE", 0), ("TM", 0)])
def test_raman_acts_on_synthesised_position_dependent_direction(
    kind, n, spatial_equivalence_backend_class
):
    original, grid, _ = vector_problem(steps=2)
    modes = [
        MarcatiliMode(125e-6, kind=kind, n=n, phi=0.37, cladding=1.45, loss=False),
        MarcatiliMode(125e-6, phi=0.19, cladding=1.45, loss=False),
    ]
    initial = original.initial.field0[0].transpose(1, 0, 2).copy()
    problem = assemble(
        grid,
        modes,
        [former_response(FORMER_H2, thg=False)],
        initial,
        density=1e25,
        quadrature=transverse_quadrature(modes, n_r=4, n_theta=8),
        controller=original.stepper.controller,
        save_planes=[grid.length],
        dz_max=grid.length / 2,
        dz_init=grid.length / 2,
        schedule=original.stepper.schedule,
    )
    assert problem.transform.n_pol * problem.shape[2] == 2
    problem = replace(problem, transform=replace(problem.transform, node_chunk=5))
    check_run(spatial_equivalence_backend_class, problem, grid)


@pytest.mark.medium
@pytest.mark.parametrize("family", ["field_real", "field_analytic", "envelope"])
def test_vector_fixed_batches_preserve_modified_noise_members(family, backend_class):
    problem, grid, _ = vector_problem(family, steps=2)
    compare(problem, grid, backend_class.name)


@pytest.mark.medium
@pytest.mark.parametrize("geometry", ["modal", "radial", "free"])
@pytest.mark.parametrize("varying_density", [False, True])
def test_vector_raman_with_two_envelope_plasma_populations(
    envelope_mixture, geometry, varying_density, spatial_equivalence_backend_class
):
    mixture, plasma = envelope_mixture
    if varying_density:
        mixture = replace(
            mixture,
            components=tuple(
                replace(
                    component,
                    density=Linear(
                        float(component.density(np.array([0.0]))[0]),
                        float(component.density(np.array([0.0]))[0]) * 0.2,
                        8e-6,
                    ),
                )
                for component in mixture.components
            ),
        )
    # Synthetic population labels with independently audited molecular kernels
    # isolate density routing from any proposed new molecular ionisation model.
    responses = plasma + tuple(
        replace(former_response(material, thg=False), species=component.label)
        for material, component in zip(
            (FORMER_H2, FORMER_N2), mixture.components, strict=True
        )
    )
    if geometry == "free" and varying_density:
        with pytest.raises(
            TranslationError, match="varying density profiles are unsupported"
        ):
            geometry_case((mixture, responses), geometry, 2)
        return
    problem, grid = geometry_case((mixture, responses), geometry, 2)
    assert len(problem.responses.raman_terms) == 2
    check_run(spatial_equivalence_backend_class, problem, grid)
