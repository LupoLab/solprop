# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response on the backends (phase 1, slice 1a).

``numpy_ref`` executes ``responses.dispersive_kerr`` on its fused node path: here its
response pass is held to the reference
:meth:`~solprop.nonlinear.responses.DispersiveKerrResponse.evaluate` on capillary
problems (field and envelope grids, both sectors, a polynomial mixture as the exact
per-species sum and a symmetric one under its mixture filter); its whole stage
derivative, with the dispersion switched off, to the constant cubic's on the
mode-averaged and radial routes; a short lossless propagation to the
work identity. ``numba_cpu`` and the native backend execute it on this path
(``tests/backends/test_numba_dispersive_kerr.py``,
``tests/backends/test_native_dispersive_kerr.py``); the free-space, planar and general
modal routes are ``tests/backends/test_dispersive_kerr_routes.py``'s.
"""

import dataclasses
import warnings
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest

from backends.matched import dispersive_response_bound
from bounds import assert_within
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.boundaries import Absorber
from solprop.constants import EPS0
from solprop.driver import Simulation
from solprop.exceptions import BackendError, MaterialRangeWarning
from solprop.grid import FieldGrid, RadialGrid
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    DispersiveWeights,
    KerrResponse,
    dispersive_cubic,
    mixture_filter,
)
from solprop.output import MemoryOutput
from solprop.physdata import MaterialSelection, Mixture
from solprop.physdata.nonlinear_data import DispersiveKerrModel, OptmatKerr
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController, RungeKutta
from solprop.translate.problem import assemble
from solprop.translate.responses import ResponseStage, dispersive_terms

try:
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]
except ImportError:
    NativeBackend = None

LENGTH = 2e-3
ARGON = OptmatKerr("Ar").dispersive_model()
FLAT = DispersiveKerrModel(
    "Ar", ARGON.gamma3_static, 0.0, 0.0, 0.0, ARGON.leg_bound, None, None,
    ARGON.material_records,
)  # fmt: skip
GRID_CASES: dict[str, dict[str, Any]] = {
    "field full": {"envelope": False, "thg": True},
    "field conserving": {"envelope": False, "thg": False},
    "envelope conserving": {"envelope": True, "thg": False},
    "envelope full": {"envelope": True, "thg": True},
}


def capillary(
    policy: str = "gas-polynomial",
    *,
    gas: str | Mapping[str, float] = "Ar",
    **overrides: object,
) -> Simulation:
    """Return a short single-mode argon capillary under an electronic policy."""
    # The keywords of one entry point, of as many types as it takes.
    arguments: dict[str, Any] = {
        "lambda0": 800e-9,
        "wavelength_limits": (150e-9, 4e-6),
        "trange": 150e-15,
        "fwhm": 10e-15,
        "energy": 30e-6,
        "materials": MaterialSelection(electronic=policy),
        "plasma": False,
        "raman": False,
        "backend": "numpy_ref",
        "n_save": 2,
    } | overrides
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(75e-6, LENGTH, gas, 2.0, **arguments)


def node_field(problem, bundle):
    """Return the launch field at the mode-averaged route's one node."""
    grids = problem.grids
    modal = passes.to_time(
        problem.initial.field0[0], grids.crop_index, grids.n_to, kind=problem.grid_kind
    )
    return passes.synthesise(
        modal, problem.transform.synthesis, bundle.taper_scale, problem.transform.n_pol
    )


def response_pass(problem, field_t, bundle):
    """Return the fused response pass over one chunk, with no temporal window."""
    return passes.nonlinear_polarisation(
        field_t,
        problem.responses,
        np.ones(problem.grids.n_to),
        kind=problem.grid_kind,
        density=bundle.density,
        preionisation=0.0,
        dt=problem.grids.dt_fine,
        carrier=problem.grids.carrier_2w0,
        scalars=bundle,
    )


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_the_response_pass_is_the_reference_evaluation(case, evaluator):
    """One evaluation: the compiled multipliers are the response's own.

    The pass and the reference share their arithmetic (``dispersive_cubic``) and differ
    in where their multipliers come from -- sampled once at translation, or on the
    reference's call -- so they agree within two evaluations' rounding
    (:func:`matched.dispersive_response_bound`).
    """
    policy = "gas-polynomial" if evaluator == "polynomial" else "symmetric-kerr"
    simulation = capillary(policy, **GRID_CASES[case])
    problem = simulation.translate()
    assert ResponseStage.DISPERSIVE_KERR in problem.responses.response_order
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.evaluator == evaluator
    assert (problem.grids.carrier_2w0 is not None) == (case == "envelope full")
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        bundle = backend._stage_scalars(0.0, 0.0)
    finally:
        backend.close()
    field_t = node_field(problem, bundle)
    got = response_pass(problem, field_t, bundle)
    response = simulation.responses[0]
    assert isinstance(response, DispersiveKerrResponse)
    (density,) = bundle.dispersive_kerr_densities
    expected = response.evaluate(
        field_t[0],
        density,
        kind=problem.grid_kind,
        omega=simulation.grid.omega_fine,
        carrier=problem.grids.carrier_2w0,
    )
    bound = dispersive_response_bound(problem.grids.n_to, evaluator)
    assert_within(got[0], expected, bound, what="polarisation")
    # The density table reproduces the fill's density to its interpolation tolerance.
    medium = cast("Mixture", simulation.configuration["medium"])
    fill = float(medium.components[0].density(np.zeros(1))[0])
    assert density == pytest.approx(fill, rel=1e-10)


def test_a_polynomial_mixture_is_the_sum_of_its_species():
    """Each species is its own channel with its own bound: the exact per-species sum."""
    simulation = capillary(gas={"He": 0.6, "Ar": 0.4})
    problem = simulation.translate()
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    assert leaf.n_channels == 2
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        bundle = backend._stage_scalars(0.0, 0.0)
    finally:
        backend.close()
    field_t = node_field(problem, bundle)
    got = response_pass(problem, field_t, bundle)
    responses = {
        response.species: response
        for response in simulation.responses
        if isinstance(response, DispersiveKerrResponse)
    }
    assert {one.model.material for one in responses.values()} == {"He", "Ar"}
    expected = sum(
        responses[label].evaluate(
            field_t[0], density, kind="field", omega=simulation.grid.omega_fine
        )
        for label, density in zip(
            leaf.labels, bundle.dispersive_kerr_densities, strict=True
        )
    )
    # Two channels summed in the pass's order and in the test's: one rounding each.
    scale = float(np.abs(expected).max())
    np.testing.assert_allclose(got[0], expected, rtol=0, atol=4e-16 * scale)
    helium, argon = (responses[label].model for label in leaf.labels)
    assert helium.leg_bound != argon.leg_bound


def test_a_symmetric_mixture_is_one_cubic_under_its_mixture_filter():
    """The symmetric evaluator's mixture: ``F G_mix[(G_mix E)^3]``, to its rounding.

    ``F`` and ``g_mix`` from the channels' coefficients and the stage's densities by
    ``mixture_filter``; its departure from the exact per-species sum is
    ``tests/nonlinear/test_dispersive_kerr_mixture.py``'s.
    """
    simulation = capillary("symmetric-kerr", gas={"He": 0.6, "Ar": 0.4})
    problem = simulation.translate()
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.n_channels == 2
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        bundle = backend._stage_scalars(0.0, 0.0)
    finally:
        backend.close()
    field_t = node_field(problem, bundle)
    got = response_pass(problem, field_t, bundle)
    total, curve = mixture_filter(
        leaf.filters,
        np.array(leaf.coefficients) * np.array(bundle.dispersive_kerr_densities),
    )
    expected = total * dispersive_cubic(
        field_t[0],
        DispersiveWeights("symmetric", leaf.positive, curve),
        kind="field",
        sector=leaf.sector,
    )
    bound = dispersive_response_bound(problem.grids.n_to, "symmetric")
    assert_within(got[0], expected, bound, what="polarisation")


def stage_derivative(problem, grid):
    """Return ``numpy_ref``'s stage derivative at the launch."""
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._hoist(0.0)
        return backend._nonlinear(
            backend._require_state(),
            backend._stage_scalars(0.0, 0.0),
            np.ones(problem.shape, complex),
        )
    finally:
        backend.close()


@pytest.mark.parametrize("thg", [True, False])
def test_without_dispersion_the_mode_averaged_derivative_is_the_constant_one(thg):
    """The whole fused path -- projection, transforms, prefactor -- around the leaf.

    With the coefficients zero the full cubic is the constant cubic, and the
    conserving sector is the ``thg=False`` rule wherever ``|u|^2 u`` has no
    negative-frequency content, which a 10 fs pulse at 800 nm has only in tails far
    below round-off. Measured 4.3e-16 and 9.1e-16 of the largest bin; the bound is
    1e-12, the few FFT round trips of 4096 points the leaf adds.
    """
    simulation = capillary("gas-polynomial", thg=thg)
    problem = simulation.translate()
    flat = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            dispersive_kerr=dispersive_terms(
                [DispersiveKerrResponse(FLAT, "polynomial", thg=thg)],
                problem.responses.labels,
                problem.responses.species_densities,
                simulation.grid,
            ),
        ),
    )
    constant = capillary("carrier-kerr", thg=thg).translate()
    constant = dataclasses.replace(
        constant,
        responses=dataclasses.replace(
            constant.responses,
            kerr_coefficients=dataclasses.replace(
                constant.responses.kerr_coefficients,  # pyright: ignore[reportArgumentType]
                coefficient=EPS0 * FLAT.gamma3_static,
            ),
        ),
    )
    got = stage_derivative(flat, simulation.grid)
    expected = stage_derivative(constant, simulation.grid)
    scale = float(np.abs(expected).max())
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12 * scale)


def test_the_radial_route_executes_the_leaf():
    """The radial route shares the fused node path; without dispersion it is Kerr's."""
    grid = FieldGrid.from_limits(1e-4, 800e-9, (400e-9, 2e-6), 100e-15)
    space = RadialGrid(4e-4, 16, window_factor=0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 1e-4
    )
    start = pulse.build(grid, space, medium=SpatialMedium()) * grid.active_mask

    def problem(response):
        return assemble(
            grid,
            None,
            [response],
            start,
            spatial_grid=space,
            medium=SpatialMedium(),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[1e-4],
            dz_max=1e-4,
            dz_init=1e-5,
            schedule=FixedSchedule.uniform(0.0, 1e-4, 1e-5),
            density=2.5e25,
        )

    got = stage_derivative(problem(DispersiveKerrResponse(FLAT, "symmetric")), grid)
    expected = stage_derivative(problem(KerrResponse(FLAT.gamma3_static)), grid)
    scale = float(np.abs(expected).max())
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12 * scale)


def energy_change(simulation) -> float:
    """Return the relative energy change of a run from its first to its last plane."""
    output = MemoryOutput()
    simulation.run(output)
    planes = output.planes()
    first, last = (
        float(np.sum(plane.stats["energy"])) for plane in planes[:: len(planes) - 1]
    )
    return last / first - 1.0


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
def test_a_lossless_run_conserves_energy(evaluator):
    """Section 4.9's work identity at propagation level, absorbers and loss off.

    Ten fixed DOPRI5 steps over 2 mm of argon at 2 bar, a 10 fs 30 uJ pulse. The
    number-conserving sector's energy holds to 1.4e-14 at ten steps; the full cubic's
    error is the integrator's, 1.5e-9 at ten steps and 1.0e-12 at twenty, falling at
    the method's order, which is what separates it from a response that did net work.
    """
    policy = "gas-polynomial" if evaluator == "polynomial" else "symmetric-kerr"
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (150e-9, 4e-6), 150e-15)
    lossless = Absorber(
        np.zeros(grid.n_omega),
        np.zeros(grid.n_t),
        np.zeros(grid.n_to),
        grid.active_mask,
        1.0,
    )

    def run(steps: int, thg: bool) -> float:
        return energy_change(
            capillary(
                policy,
                grid=grid,
                wavelength_limits=None,
                trange=None,
                thg=thg,
                loss=False,
                absorber=lossless,
                schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
                integrator=RungeKutta(),
            )
        )

    # Thirty roundings of a few thousand-term sum.
    assert abs(run(10, thg=False)) < 1e-13
    coarse, fine = run(10, thg=True), run(20, thg=True)
    assert abs(fine) < 1e-10
    assert abs(fine) < abs(coarse) / 30


@pytest.mark.skipif(NativeBackend is None, reason="solprop-native is not installed")
def test_native_admits_the_leaf_on_the_fused_path():
    simulation = capillary("symmetric-kerr")
    problem = simulation.translate()
    NativeBackend(problem, grid=simulation.grid).close()  # pyright: ignore[reportOptionalCall]


def test_numba_cpu_admits_the_leaf_on_the_fused_path():
    simulation = capillary("symmetric-kerr")
    problem = simulation.translate()
    NumbaCpuBackend(problem, grid=simulation.grid).close()


def test_the_pass_refuses_stage_inputs_that_do_not_match_the_leaf():
    """The densities, the harmonic carrier and the stage scalars are the leaf's own."""
    simulation = capillary("gas-polynomial", **GRID_CASES["envelope full"])
    problem = simulation.translate()
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.carrier
    backend = NumpyRefBackend(problem, grid=simulation.grid)
    try:
        bundle = backend._stage_scalars(0.0, 0.0)
    finally:
        backend.close()
    field_t = node_field(problem, bundle)
    with pytest.raises(BackendError, match="the stage carries 0 densities"):
        passes.dispersive_kerr_polarisation(
            field_t, leaf, (), kind="envelope", carrier=problem.grids.carrier_2w0
        )
    with pytest.raises(BackendError, match="no harmonic carrier was given"):
        passes.dispersive_kerr_polarisation(
            field_t, leaf, bundle.dispersive_kerr_densities, kind="envelope"
        )
    with pytest.raises(BackendError, match="from the stage scalars"):
        passes.nonlinear_polarisation(
            field_t,
            problem.responses,
            np.ones(problem.grids.n_to),
            kind=problem.grid_kind,
            density=bundle.density,
            preionisation=0.0,
            dt=problem.grids.dt_fine,
            carrier=problem.grids.carrier_2w0,
        )
