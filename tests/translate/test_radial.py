# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Radial cumulative expansions, scalar transform layout and early refusals."""

import dataclasses
from contextlib import nullcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad_vec

from hankel_oracles import gamma
from solprop.constants import C
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.linear import Frame
from solprop.profiles import Linear
from solprop.spatial import SpatialDispersion, SpatialMedium
from solprop.stepper import PIController
from solprop.translate import radial
from solprop.translate.chebyshev import DEFAULT_TOL
from solprop.translate.expansion import ParameterExpansion
from solprop.translate.linop import delta_phi, phi_rate
from solprop.translate.problem import assemble
from solprop.translate.radial import (
    radial_operators,
    radial_rate_bounds,
    radial_transform,
)


def adiabatic_oracle(medium, grid, space, z0, z1):
    """Return the endpoint amplitude exponent and its propagating set, independently.

    SEP 34 section 4.4: ``-1/2 ln(Re k_z(z1) / Re k_z(z0))`` on active bins whose
    longitudinal square has a positive real part, zero elsewhere, from the material
    index alone and a plain logarithm of the ratio.
    """
    omega = grid.omega[grid.active_mask]

    def square(z):
        return (omega * medium.index(omega, z) / C) ** 2 - space.k_r[:, None] ** 2

    start, end = square(z0), square(z1)
    propagating = np.real(start) > 0
    assert np.array_equal(propagating, np.real(end) > 0)
    ratio = np.ones_like(np.real(start))
    ratio[propagating] = np.real(np.sqrt(end[propagating])) / np.real(
        np.sqrt(start[propagating])
    )
    exponent = np.zeros((space.n_r, grid.n_omega))
    exponent[:, grid.active_mask] = -0.5 * np.log(ratio)
    mask = np.zeros_like(exponent, dtype=bool)
    mask[:, grid.active_mask] = propagating
    return exponent, mask


def endpoint_budget(exponent):
    """Bound the represented endpoint term's error against the oracle, absolutely.

    The endpoint function, a log-ratio of order 1e-4 for a gas, is resolved to a
    relative target at eight times its measured round-off floor, about 1e-10 here
    (each root carries a few rounding units against that small ratio); the Chebyshev
    aliasing factor two at two endpoints makes 4e-10, rounded up to 1e-9. The
    oracle's root, ratio and logarithm add 64 rounding units.
    """
    return 1e-9 * float(np.max(np.abs(exponent))) + gamma(64)


@pytest.fixture
def case():
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    return grid, RadialGrid(6e-4, 16, window_factor=0.0)


@pytest.mark.parametrize("comoving", [False, True])
def test_varying_radial_medium_uses_cumulative_expansion(case, comoving):
    grid, space = case
    medium = SpatialMedium("Ar", pressure=Linear(0.5, 1.5, grid.length))
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        frame = Frame.comoving(grid)
        if not comoving:
            frame = dataclasses.replace(frame, beta0=0.0, beta1=0.0)
        model = SpatialDispersion(medium, grid, frame)
        # The gas index is monotone over this pressure segment. Bound the worst
        # sqrt conditioning with the entrance's smallest propagating root; the frame
        # and material are no larger than the exit's optical wave number envelope.
        omega = grid.omega[grid.active_mask]
        wave_max = float(np.max(omega * np.real(medium.index(omega, grid.length)) / C))
        wave_min = float(np.min(omega * np.real(medium.index(omega, 0.0)) / C))
        root_min = np.sqrt(wave_min**2 - float(space.k_r.max()) ** 2)
        arithmetic = gamma(64) * (wave_max**2 / (2 * root_min) + 2 * wave_max)
        # The expansion's independent validation discards arithmetic-scale terms at
        # eight rounding budgets. Its per-element envelope is bounded here by the
        # full band operator scale, with a further factor from worst/best conditioning.
        smallest_diffraction = space.k_r[0] ** 2 / (2 * wave_max)
        conditioning_ratio = 2 * wave_max / smallest_diffraction if comoving else 1.0
        rate_bound = 8 * arithmetic * conditioning_ratio + 2 * wave_max * DEFAULT_TOL
        quad_budget = 1e-10  # absolute phase budget, below the predeclared table bound
        operator, prefactor = radial_operators(space, model, n_pol=2, z_max=grid.length)
        assert operator.phi_representation == "expansion"
        assert operator.phi_expansion is not None
        assert isinstance(prefactor, ParameterExpansion)
        assert operator.phi_expansion.parameters == ("density",)
        assert prefactor.parameters == ("density",)
        for left, right in (
            (0.0, grid.length),
            (0.21 * grid.length, 0.79 * grid.length),
        ):
            # Remove the exactly integrable constant before quadrature. The
            # uncentred optical carrier exhausted the oracle's unchanged error
            # budget through its positive-norm roundoff estimate (reported miss).
            centre = model.operator(space.k_r**2, (left + right) / 2)
            expected, quadrature_error = quad_vec(
                lambda z, centre=centre: (
                    model.operator(space.k_r**2, float(z)) - centre
                ),
                left,
                right,
                epsabs=quad_budget / 8,
                epsrel=1e-13,
            )
            assert quadrature_error <= quad_budget
            expected += centre * (right - left)
            # The adiabatic amplitude term is an endpoint quantity (SEP 34
            # section 4.4), added from the independent root ratio.
            amplitude, _ = adiabatic_oracle(medium, grid, space, left, right)
            expected += amplitude
            bound = (
                rate_bound * (right - left) + quad_budget + endpoint_budget(amplitude)
            )
            actual = delta_phi(operator, left, right)[0]
            assert_allclose(actual[:, 0], expected, rtol=0, atol=bound)
            np.testing.assert_array_equal(actual[:, 0], actual[:, 1])
        for z in (0.0, 0.37 * grid.length, grid.length):
            expected = model.prefactor(space.k_r**2, z) * grid.crop_scale
            assert_allclose(
                prefactor(z)[:, 0], expected, rtol=8 * DEFAULT_TOL + gamma(128), atol=0
            )


def test_radial_transform_keeps_polarisations_independent_and_chunks_nodes(case):
    grid, space = case
    budget = 2048 * grid.n_to
    transform = radial_transform(
        space, grid, n_pol=2, z_max=grid.length, node_budget=budget
    )
    assert transform.n_pol == 1
    assert transform.synthesis.shape == (space.n_r, space.n_r)
    assert transform.projection.shape == (space.n_r, space.n_r)
    assert transform.node_chunk <= space.n_r
    np.testing.assert_array_equal(transform.projection_coarse, transform.projection)
    assert transform.geometry is not None
    assert np.all(transform.geometry.absorber_rate == 0)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"spatial_grid": None}, "together"),
        ({"medium": None}, "together"),
        ({"components": ("y",)}, "components"),
        ({"shock": False}, "shock=False"),
        ({"prefactor_index": 1.0}, "prefactor_index"),
    ],
)
def test_radial_assembly_refuses_ambiguous_modal_options(case, options, match):
    grid, space = case
    keywords = {"spatial_grid": space, "medium": SpatialMedium()}
    keywords.update(options)
    with pytest.raises(TranslationError, match=match):
        assemble(
            grid,
            None,
            [],
            np.zeros((space.n_r, 1, grid.n_omega), dtype=np.complex128),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[grid.length],
            dz_max=grid.length,
            dz_init=grid.length,
            **keywords,
        )


@pytest.mark.parametrize("options", [{"z_max": 0.0}, {"margin": -1.0}, {"tol": 0.0}])
def test_radial_operator_refuses_invalid_span_or_tolerance(case, options):
    grid, space = case
    keywords = {"n_pol": 1, "z_max": grid.length}
    keywords.update(options)
    with pytest.raises(TranslationError, match="span and tolerance"):
        radial_operators(
            space,
            SpatialDispersion(SpatialMedium(), grid, Frame.comoving(grid)),
            **keywords,
        )


@pytest.mark.parametrize("failed_call", [1, 2])
def test_unresolved_density_expansion_refuses_a_silent_fallback(
    case, monkeypatch, failed_call
):
    grid, space = case
    medium = SpatialMedium("Ar", pressure=Linear(0.5, 1.5, grid.length))
    original = radial.build_expansion
    calls = 0

    def fail_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        return None if calls == failed_call else original(*args, **kwargs)

    monkeypatch.setattr(radial, "build_expansion", fail_one)
    # The registry's argon fit range warned here; optmat's argon is qualified
    # over the band (SEP 3, Phase 4a).
    with pytest.raises(TranslationError, match="could not resolve"):
        radial_operators(
            space,
            SpatialDispersion(medium, grid, Frame.comoving(grid)),
            n_pol=1,
            z_max=grid.length,
        )


def test_a_constant_radial_medium_carries_no_adiabatic_term(case):
    grid, space = case
    model = SpatialDispersion(
        SpatialMedium("Ar", pressure=1.0), grid, Frame.comoving(grid)
    )
    operator, _ = radial_operators(space, model, n_pol=1, z_max=grid.length)
    assert operator.phi_representation == "constant"
    assert operator.phi_constant is not None
    # The uniform medium's rate is the material operator itself, element by element:
    # no real endpoint term enters, so the exactly integrated D z is unchanged.
    np.testing.assert_array_equal(
        operator.phi_constant[0, :, 0], model.operator(space.k_r**2)
    )


@pytest.mark.parametrize("route", ["expansion", "table"])
def test_the_radial_adiabatic_term_matches_an_independent_oracle(
    case, monkeypatch, route
):
    grid, _ = case
    # Columns 0-2 cut off inside the band and the rest are evanescent throughout,
    # with no column crossing the cutoff along the pressure ramp.
    space = RadialGrid(1e-6, 16, window_factor=0.0)
    medium = SpatialMedium("Ar", pressure=Linear(0.5, 1.5, grid.length))
    model = SpatialDispersion(medium, grid, Frame.comoving(grid))
    if route == "table":
        monkeypatch.setattr(radial, "MAX_PARAMETERS", 0)
    operator, _ = radial_operators(space, model, n_pol=1, z_max=grid.length)
    assert operator.phi_representation == route
    omega = grid.omega[grid.active_mask]

    def attenuation(z):
        square = (omega * medium.index(omega, z) / C) ** 2 - space.k_r[:, None] ** 2
        root = np.sqrt(square)
        return np.where(
            np.real(square) > 0,
            np.imag(root),
            np.minimum(np.abs(np.imag(root)), model.evanescent_rate),
        )

    quad_budget = 1e-12  # absolute, on an attenuation integral of at most 0.2
    for left, right in ((0.0, grid.length), (0.3 * grid.length, 0.55 * grid.length)):
        amplitude, propagating = adiabatic_oracle(medium, grid, space, left, right)
        assert propagating[:3].any()
        assert not propagating[3:].any()
        loss, error = quad_vec(
            attenuation, left, right, epsabs=quad_budget / 8, epsrel=0.0
        )
        assert error <= quad_budget
        expected = amplitude.copy()
        expected[:, grid.active_mask] -= loss
        actual = np.real(delta_phi(operator, left, right)[0, :, 0])
        # Propagating bins: Re Phi is the endpoint term less the material loss
        # integral, which is zero for argon; the integral contributes no real part.
        assert_allclose(
            actual[propagating],
            expected[propagating],
            rtol=0,
            atol=endpoint_budget(amplitude) + quad_budget,
        )
        # Evanescent and inactive bins carry no endpoint term: their real part is
        # the capped attenuation alone, to the integral's representation bound
        # (relative 8 DEFAULT_TOL on a rate of at most the cap).
        rest = ~propagating
        assert_allclose(
            actual[rest],
            expected[rest],
            rtol=0,
            atol=8 * DEFAULT_TOL * model.evanescent_rate * (right - left) + quad_budget,
        )
        np.testing.assert_array_equal(actual[:, ~grid.active_mask], 0.0)


def test_a_cutoff_crossing_along_the_path_refuses(case):
    grid, _ = case
    medium = SpatialMedium("Ar", pressure=Linear(0.5, 1.5, grid.length))
    omega = grid.omega[grid.active_mask]
    middle = omega.size // 2
    low = float(omega[middle] * np.real(medium.index(omega, 0.0))[middle] / C)
    high = float(omega[middle] * np.real(medium.index(omega, grid.length))[middle] / C)
    assert high > low
    # Place the first column's transverse wave number between the entrance and exit
    # material wave numbers of one active bin, so its q = 0 lies inside the path.
    unit = RadialGrid(1.0, 16, window_factor=0.0)
    space = RadialGrid(float(unit.k_r[0]) / (0.5 * (low + high)), 16, window_factor=0.0)
    assert low < space.k_r[0] < high
    with pytest.raises(TranslationError, match="cutoff"):
        radial_operators(
            space,
            SpatialDispersion(medium, grid, Frame.comoving(grid)),
            n_pol=1,
            z_max=grid.length,
        )


def test_reused_radial_expansion_carries_the_adiabatic_term(case):
    grid, space = case
    frame = Frame.comoving(grid)
    source = SpatialDispersion(
        SpatialMedium("Ar", pressure=Linear(0.5, 1.5, grid.length)), grid, frame
    )
    linear, prefactor = radial_operators(space, source, n_pol=1, z_max=grid.length)
    assert linear.phi_expansion is not None
    assert linear.phi_expansion.endpoint_degrees is not None
    assert isinstance(prefactor, ParameterExpansion)
    medium = SpatialMedium("Ar", pressure=Linear(1.3, 0.7, grid.length))
    updated = SpatialDispersion(medium, grid, frame)
    reused, _ = radial_operators(
        space,
        updated,
        n_pol=1,
        z_max=grid.length,
        reuse_linear=linear.phi_expansion,
        reuse_prefactor=prefactor,
    )
    fresh, _ = radial_operators(space, updated, n_pol=1, z_max=grid.length)
    assert reused.phi_expansion is not None
    assert reused.phi_expansion.terms is linear.phi_expansion.terms
    for left, right in ((0.0, grid.length), (0.2 * grid.length, 0.9 * grid.length)):
        amplitude, _ = adiabatic_oracle(medium, grid, space, left, right)
        budget = endpoint_budget(amplitude)
        for operator in (reused, fresh):
            actual = np.real(delta_phi(operator, left, right)[0, :, 0])
            assert_allclose(actual, amplitude, rtol=0, atol=budget)


@pytest.mark.parametrize("route", ["expansion", "table"])
def test_radial_rate_bounds_include_the_adiabatic_term(case, monkeypatch, route):
    grid, space = case
    # Lossless argon whose pressure falls: Re k_z decreases, so the only real part
    # of D is the positive adiabatic rate -1/2 d ln(Re k_z)/dz.
    medium = SpatialMedium("Ar", pressure=Linear(1.5, 0.5, grid.length))
    if route == "table":
        monkeypatch.setattr(radial, "MAX_PARAMETERS", 0)
    model = SpatialDispersion(medium, grid, Frame.comoving(grid))
    operator, _ = radial_operators(space, model, n_pol=1, z_max=grid.length)
    assert operator.phi_representation == route
    bounds = radial_rate_bounds(operator)
    rates = np.stack(
        [
            np.real(phi_rate(operator, float(z)))
            for z in np.linspace(*operator.z_span, 33)
        ]
    )
    assert np.max(rates) > 0
    assert bounds.gain >= np.max(rates)
    assert bounds.loss >= max(-float(np.min(rates)), 0.0)
