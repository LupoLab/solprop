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
from solprop.translate.linop import delta_phi
from solprop.translate.problem import assemble
from solprop.translate.radial import radial_operators, radial_transform


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
            bound = rate_bound * (right - left) + quad_budget
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
