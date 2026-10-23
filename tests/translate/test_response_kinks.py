# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Response paths that kink become hard kinks of Phi."""

import dataclasses

import numpy as np
import pytest

from mixture_cases import AnalyticRate, mixture_case
from rate_oracles import adk_oracle
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.linear import Frame
from solprop.modes import MarcatiliMode
from solprop.nonlinear import EnvelopePlasmaResponse
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.profiles import (
    Constant,
    FromCallable,
    Linear,
    PiecewiseLinear,
    PressureGradient,
)
from solprop.pulses import GaussianPulse, SechPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import build_phi, constant_phi, delta_phi
from solprop.translate.problem import _walk_arrays, assemble
from solprop.translate.responses import response_kinks, unlanded_kinks

OPERATOR = np.array([0.0, -1.5j, 0.3 - 2.0j])
"""A constant ``D`` in 1/m with a lossy entry, so the real and imaginary parts count."""


def test_constant_phi_without_kinks_is_todays_single_interval():
    today = constant_phi(OPERATOR, 1.0)
    for kinks in ((), [0.0, 1.05, 2.0], np.array([-1.0])):
        again = constant_phi(OPERATOR, 1.0, kinks=kinks)
        for name in ("phi_breakpoints", "phi_coefficients", "phi_degrees"):
            np.testing.assert_array_equal(getattr(again, name), getattr(today, name))
        assert again.hard_kinks.size == 0


def test_constant_phi_split_at_kinks_stays_exact_and_keeps_its_constant():
    kinks = [0.61, 0.25, 0.25]
    split = constant_phi(OPERATOR, 1.0, kinks=kinks)
    whole = constant_phi(OPERATOR, 1.0)
    np.testing.assert_array_equal(split.phi_breakpoints, [0.0, 0.25, 0.61, 1.05])
    np.testing.assert_array_equal(split.hard_kinks, [0.25, 0.61])
    np.testing.assert_array_equal(split.phi_degrees, [1, 1, 1])
    np.testing.assert_array_equal(split.phi_constant, whole.phi_constant)
    # delta_phi reads phi_constant, so the flow is exactly the unsplit one.
    np.testing.assert_array_equal(
        delta_phi(split, 0.1, 0.7), delta_phi(whole, 0.1, 0.7)
    )
    # Each piece is D z exactly up to the rounding of its two coefficients, and
    # continuous across the kinks: a few ulp of |D| z.
    z = np.array([0.0, 0.1, 0.25, 0.4, 0.61, 0.9, 1.05])
    values = np.stack([split.polynomial(float(one)) for one in z])
    assert whole.phi_constant is not None
    expected = z[:, None, None, None, None] * whole.phi_constant[None]
    np.testing.assert_allclose(values, expected, rtol=0, atol=8 * np.finfo(float).eps)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_an_expansion_takes_the_further_kinks_beside_the_modes():
    grid = EnvelopeGrid.from_limits(0.05, 800e-9, (400e-9, 2e-6), 200e-15)
    fill = PressureGradient.two_point(0.05, 2.0, 1.0)
    mode = MarcatiliMode(125e-6, "He", fill, loss=False)
    frame = Frame.comoving(grid)
    plain = build_phi(mode, grid, frame, z_max=0.05)
    kinked = build_phi(mode, grid, frame, z_max=0.05, kinks=[0.0123, 0.05, 7.0])
    assert plain.phi_representation == kinked.phi_representation == "expansion"
    np.testing.assert_array_equal(plain.hard_kinks, [0.05])
    np.testing.assert_array_equal(kinked.hard_kinks, [0.0123, 0.05])


def test_response_kinks_name_each_path_and_skip_numbers():
    path = PiecewiseLinear(np.array([0.0, 0.3, 1.0]), np.array([0.0, 0.2, 0.2]))
    responses = (
        KerrResponse(1e-52),
        PlasmaResponse("Ar", adk_oracle("Ar"), preionisation=0.1),
        PlasmaResponse("Ar", adk_oracle("Ar"), preionisation=path, species="a"),
        EnvelopePlasmaResponse("Ar", adk_oracle("Ar"), preionisation=Constant(0.1)),
    )
    density = PiecewiseLinear(np.array([0.0, 0.6]), np.array([1e24, 2e24]))
    assert response_kinks(responses, density) == (
        (0.0, "preionisation:a"),
        (0.3, "preionisation:a"),
        (1.0, "preionisation:a"),
        (0.0, "density"),
        (0.6, "density"),
    )
    assert response_kinks(responses[:2], 1.0) == ()
    kinks = response_kinks(responses, density)
    assert unlanded_kinks(kinks, np.empty(0), 1.0, slack=1e-12) == (
        (0.3, "preionisation:a"),
        (0.6, "density"),
    )
    assert unlanded_kinks(kinks, np.array([0.3, 0.6]), 1.0, slack=1e-12) == ()


def stepped(length, start, end, low=0.02, high=0.2):
    """Return a fraction ramping from `low` to `high` between two kinks, in m."""
    return PiecewiseLinear(
        np.array([0.0, start, end, length]), np.array([low, low, high, high])
    )


def test_assemble_lands_a_preionisation_kink_and_a_number_is_unchanged():
    _, _, _, _, numbers = mixture_case(preionisation=(0.03, 0.1))
    _, _, _, _, constants = mixture_case(preionisation=(Constant(0.03), Constant(0.1)))
    _, _, _, _, kinked = mixture_case(
        preionisation=(stepped(1e-3, 2e-4, 3e-4), Constant(0.1))
    )
    # A constant path evaluates to its own number exactly, so every array of the
    # contract is the number's (structural: no arithmetic separates them).
    for (name, left), (other, right) in zip(
        _walk_arrays(numbers), _walk_arrays(constants), strict=True
    ):
        assert name == other
        np.testing.assert_array_equal(left, right, err_msg=name)
    assert numbers.linear.hard_kinks.size == 0
    np.testing.assert_array_equal(kinked.linear.hard_kinks, [2e-4, 3e-4, 1e-3])
    assert kinked.linear.phi_representation == "constant"
    assert kinked.responses.prefactor_representation == "constant"
    np.testing.assert_array_equal(
        kinked.linear.phi_constant, numbers.linear.phi_constant
    )


def test_an_explicit_density_path_kinks_phi_too():
    grid = EnvelopeGrid.from_limits(0.02, 835e-9, (500e-9, 2e-6), 1e-12)
    gnlse = problem_from([0.0, 0.0, -1.2e-26], omega_ref=grid.omega0, gamma=0.11)
    start = SechPulse(wavelength=835e-9, width=1e-13, peak_power=1.0).build(grid)
    density = PiecewiseLinear(np.array([0.0, 0.007, 0.02]), np.array([1.0, 1.0, 0.5]))
    problem = assemble(
        grid,
        gnlse.mode,
        [gnlse.kerr],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.02],
        dz_max=0.01,
        dz_init=1e-3,
        density=density,
        shock=False,
        prefactor_index=1.0,
    )
    np.testing.assert_array_equal(problem.linear.hard_kinks, [0.007, 0.02])


@dataclasses.dataclass(frozen=True)
class _InstantaneousRate(AnalyticRate):
    """The synthetic power law, declared an instantaneous isotropic rate."""

    convention: str = "instantaneous-isotropic"


def test_a_route_that_cannot_land_the_kink_refuses_it_by_name():
    mixture, _, _, _, _ = mixture_case(
        kerr=False, preionisation=(stepped(8e-6, 2e-6, 4e-6), 0.1)
    )
    responses = tuple(
        EnvelopePlasmaResponse(
            component.species,
            build_rate_table(_InstantaneousRate(scale=1e13), (1e7, 1e11), rtol=1e-7),
            preionisation=component.preionisation,
            species=component.label,
        )
        for component in mixture.components
    )
    grid = EnvelopeGrid.from_limits(8e-6, 800e-9, (500e-9, 1.2e-6), 80e-15)
    space = RadialGrid(6e-4, 8, window_factor=0.0)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=20e-15, energy=1e-7)
    with pytest.raises(
        TranslationError, match=r"preionisation:population-0 path kinks at z = 2e-06"
    ):
        assemble(
            grid,
            None,
            responses,
            SpatialPulse(pulse, 80e-6).build(grid, space),
            controller=PIController(rtol=1e-7, atol=1e-14),
            save_planes=[8e-6],
            dz_max=4e-6,
            dz_init=4e-6,
            spatial_grid=space,
            medium=SpatialMedium(mixture),
        )


def test_a_path_that_leaves_the_unit_interval_between_probes_is_caught_in_the_table():
    # The construction check samples 65 probes; a narrow bump between the first two
    # passes it, and the compiled table's interval-by-interval check refuses it.
    length = 1e-3
    centre, width = 0.5 * length / 64, length / 200

    def spike(z):
        return 0.1 + 1.5 * np.exp(-(((np.asarray(z) - centre) / width) ** 2))

    path = FromCallable(spike, (0.0, length))
    with pytest.raises(TranslationError, match=r"preionisation:population-0 must lie"):
        mixture_case(preionisation=(path, 0.1))


def test_a_constant_path_with_breakpoints_is_exactly_the_number():
    # A flat path evaluates to its value exactly (a + 0 t), so this is structural.
    flat = Linear(0.03, 0.03, 5e-4)
    assert (
        response_kinks([PlasmaResponse("Ar", adk_oracle("Ar"), preionisation=flat)])
        == ()
    )
    _, _, _, _, numbers = mixture_case(preionisation=(0.03, 0.1))
    _, _, _, _, flats = mixture_case(
        preionisation=(flat, PiecewiseLinear(np.array([0.0, 2e-4]), np.full(2, 0.1)))
    )
    assert flats.linear.hard_kinks.size == 0
    for (name, left), (_, right) in zip(
        _walk_arrays(numbers), _walk_arrays(flats), strict=True
    ):
        np.testing.assert_array_equal(left, right, err_msg=name)
