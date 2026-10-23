# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Power conversion bounds, each fixed before measurement."""

from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import pytest
from numpy.polynomial import chebyshev as chebyshev_basis

from bounds import EPS, assert_within
from solprop.exceptions import ModeError, TranslationError
from solprop.fourier import power_amplitude
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import Frame, log_power_weight
from solprop.modes import MarcatiliMode
from solprop.noise import modified_noise
from solprop.nonlinear.gnlse import SpecifiedMode, problem_from
from solprop.profiles import Linear, PressureGradient
from solprop.stepper import PIController
from solprop.translate.chebyshev import DEFAULT_TOL, build
from solprop.translate.expansion import (
    append_endpoint_expansion,
    build_expansion,
    parameter_roundoff_floor,
)
from solprop.translate.linop import (
    _add_phi_tables,
    _dispersion_phi,
    _log_weight_sampler,
    _roundoff_floor,
    build_phi,
    delta_phi,
    power_weight_at,
    power_weight_factors,
    with_power_weight,
)
from solprop.translate.problem import assemble


@pytest.mark.parametrize("index", [0.0, -1.0, np.inf, np.nan])
def test_log_power_weight_refuses_nonpositive_or_nonfinite_index(index):
    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45)
    with pytest.raises(ModeError, match="positive and finite"):
        log_power_weight(mode, grid.omega[grid.active_mask], index=index)


def test_log_power_weight_uses_the_prefactor_index():
    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(125e-6, "Ar", 20.0, cladding=1.45)
    omega = grid.omega[grid.active_mask]
    assert np.array_equal(
        log_power_weight(mode, omega, index=1.0), np.zeros_like(omega)
    )
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        actual = log_power_weight(mode, omega)
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        expected = np.log(np.real(mode.neff(omega)))
    # The same logarithm of the same index: a rounding of the index moves its
    # logarithm by about eps absolutely, so 4 eps absolute with 4 eps relative.
    np.testing.assert_allclose(actual, expected, rtol=4 * EPS, atol=4 * EPS)


def test_endpoint_terms_use_plain_products_and_integral_terms_keep_integrals():
    paths = {"density": Linear(1.0, 3.0, 1.0)}
    integral = build_expansion(
        lambda values: np.array([1j * values["density"]]),
        paths,
        0.0,
        1.0,
        integrate=True,
    )
    assert integral is not None
    combined = append_endpoint_expansion(
        integral,
        lambda values: np.array([values["density"] ** 2 + 0j]),
        paths,
        tol=1e-14,
    )
    assert combined is not None
    assert np.array_equal(combined.terms[: integral.n_terms], integral.terms)
    for z in np.random.default_rng(42).uniform(0.0, 1.0, 20):
        # Two low-degree Chebyshev transforms and a five-term reduction: 64 eps
        # bounds the absolute error at this unit parameter scale.
        expected = 1j * (z + z**2) + (1 + 2 * z) ** 2 - 1
        np.testing.assert_allclose(
            combined(float(z)), expected, rtol=0, atol=64 * np.finfo(float).eps
        )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("representation", ["expansion", "table"])
def test_phi_gains_the_exact_log_term(representation):

    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(
        125e-6,
        "Ar",
        PressureGradient.two_point(0.01, 20.0, 1.0),
        cladding=1.45,
        loss=False,
    )
    result = build_phi(
        mode, grid, Frame.comoving(grid), z_max=0.01, representation=representation
    )
    sample = _log_weight_sampler((mode,), grid, 1, None)
    # The same second-difference floor as build_phi, with the eight-rounding-unit
    # sampling/transform margin. Near-zero logarithms need an absolute epsilon floor.
    floor = _roundoff_floor(sample, result.z_span[1])
    scale = max(float(np.abs(sample(z)).max()) for z in (0.0, 0.005, 0.01))
    bound = 8 * max(floor * scale, np.finfo(float).eps)
    for z in np.random.default_rng(42).uniform(0.0, 0.01, 20):
        expected = np.real(sample(float(z)) - sample(0.0))
        actual = np.real(delta_phi(result, 0.0, float(z)))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_the_log_term_resolves_on_a_tapered_gradients_parameter_box():
    """The endpoint's tolerance is measured on the box it is resolved over.

    A 100 to 80 um taper beside a 5 to 0.2 bar argon gradient: the box's low-density,
    wide-bore corner, which the path never visits, holds bins near the long-wavelength
    crossing of ``Re n_eff = 1`` whose ``ln Re n_eff`` is a few 1e-6, so the same
    absolute rounding is a larger relative floor there than anywhere along z. The
    translation raised before the fix; the floor on the box is above the tolerance the
    path's floor gave, which is why no degree could meet it.
    """
    grid = FieldGrid.from_limits(0.1, 800e-9, (300e-9, 2e-6), 400e-15)
    mode = MarcatiliMode(
        Linear(100e-6, 80e-6, 0.1),
        "Ar",
        PressureGradient.two_point(0.1, 5.0, 0.2),
        loss=False,
    )
    result = build_phi(
        mode, grid, Frame.comoving(grid), z_max=0.1, representation="expansion"
    )
    expansion = result.phi_expansion
    assert expansion is not None and expansion.endpoint_degrees is not None
    sample = _log_weight_sampler((mode,), grid, 1, None)

    def value_at(values):
        return _log_weight_sampler([mode.at_parameters(values)], grid, 1, None)(0.0)

    path_floor = _roundoff_floor(sample, result.z_span[1])
    box_floor = parameter_roundoff_floor(
        value_at, expansion.parameters, expansion.ranges
    )
    assert box_floor > 8 * path_floor
    tol = max(8 * np.finfo(float).eps, 8 * path_floor, 8 * box_floor)
    scale = max(float(np.abs(sample(z)).max()) for z in (0.0, 0.05, 0.1))
    # The declared tolerance of the parameter fit and of the scalar weight tables,
    # relative to the value scale, with the factor two of the tail criterion.
    bound = 2.0 * (tol + DEFAULT_TOL) * scale
    for z in np.random.default_rng(42).uniform(0.0, 0.1, 20):
        expected = np.real(sample(float(z)) - sample(0.0))
        actual = np.real(delta_phi(result, 0.0, float(z)))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=bound)


def test_the_parameter_floor_is_zero_for_a_constant_and_stays_in_the_box():
    def vacuum_safe(values):
        if values["density"] < 0.0:
            raise AssertionError("sampled below a vacuum end")
        return np.array([2.0, 0.0])

    ranges = np.array([[0.0, 1.0], [1e-4, 2e-4]])
    assert parameter_roundoff_floor(vacuum_safe, ("density", "radius"), ranges) == 0.0
    assert (
        parameter_roundoff_floor(
            lambda values: np.zeros(3), ("density",), np.array([[0.0, 1.0]])
        )
        == 0.0
    )


def test_the_parameter_floor_is_relative_to_each_line_of_the_box():
    # ln(1 + 1e-4 rho (1 + a)) is a thousand times smaller on the low-density line
    # than on the high one, and its rounding is absolute, so measured against that
    # line's own values -- as the tail test measures the radius axis -- the floor is
    # about a thousand times the one of the same function without the density factor.
    ranges = np.array([[1e-3, 1.0], [0.0, 1.0]])
    names = ("density", "radius")
    floor = parameter_roundoff_floor(
        lambda values: np.array(
            [np.log(1.0 + 1e-4 * values["density"] * (1.0 + values["radius"]))]
        ),
        names,
        ranges,
    )
    uniform = parameter_roundoff_floor(
        lambda values: np.array([np.log(1.0 + 1e-4 * (1.0 + values["radius"]))]),
        names,
        ranges,
    )
    assert floor > 100.0 * uniform > 0.0


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_constant_fill_phi_is_unchanged():
    # The unit index's weight path and the plain one tabulate the same operator; the
    # coefficients agree normwise to the one-evaluation budget of 16 eps.
    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(125e-6, "Ar", 20.0, cladding=1.45)
    original = build_phi(mode, grid, Frame.comoving(grid), z_max=0.01, index=1.0)
    result = build_phi(mode, grid, Frame.comoving(grid), z_max=0.01)
    assert result.phi_coefficients is not None and original.phi_coefficients is not None
    assert_within(result.phi_coefficients, original.phi_coefficients, 16 * EPS)
    assert result.phi_constant is not None and original.phi_constant is not None
    assert_within(result.phi_constant, original.phi_constant, 16 * EPS)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_assemble_divides_true_power_launch_and_noise():

    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(125e-6, "Ar", 20.0, cladding=1.45)
    field = np.ones((1, 1, grid.n_omega), dtype=complex)
    # Noise and pulse builders remain in true sqrt(W); only assemble converts.
    noise = modified_noise(grid, np.random.default_rng(42))
    result = assemble(
        grid,
        mode,
        [],
        field,
        noise=noise,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[0.01],
        dz_max=0.01,
        dz_init=0.001,
    )
    root = np.ones(grid.n_omega)
    root[grid.active_mask] = np.sqrt(np.real(mode.neff(grid.omega[grid.active_mask])))
    # Division and the independently evaluated square root each round once.
    np.testing.assert_allclose(
        result.initial.field0[0], field / root, rtol=2 * np.finfo(float).eps, atol=0
    )
    assert result.initial.noise0 is not None
    np.testing.assert_allclose(
        result.initial.noise0[0],
        noise.spectrum / root,
        rtol=2 * np.finfo(float).eps,
        atol=0,
    )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("representation", ["constant", "expansion", "table"])
def test_weight_row_matches_direct_evaluation(representation):

    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(
        125e-6,
        "Ar",
        20.0
        if representation == "constant"
        else PressureGradient.two_point(0.01, 20.0, 1.0),
        cladding=1.45,
    )
    problem = assemble(
        grid,
        mode,
        [],
        np.ones((1, 1, grid.n_omega), dtype=complex),
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[0.01],
        dz_max=0.01,
        dz_init=0.001,
    )
    if representation == "table":
        phi = build_phi(
            mode, grid, Frame.comoving(grid), z_max=0.01, representation="table"
        )
        problem = replace(problem, linear=with_power_weight(phi, (mode,), grid, None))
    assert problem.linear.power_weight_representation == representation
    problem.validate()
    for z in np.random.default_rng(42).uniform(0.0, 0.01, 20):
        expected = np.ones((1, 1, grid.n_omega))
        expected[0, 0, grid.active_mask] = np.real(
            mode.neff(grid.omega[grid.active_mask], float(z))
        )
        # The bound is the table's own build tolerance, the default 1e-12 every
        # z-table is built to; evaluation round-off is far below it.
        np.testing.assert_allclose(
            power_weight_at(problem, float(z)), expected, rtol=1e-12, atol=0
        )
    # The factored form a saved run records is the same quantity as a dot product of
    # its basis row with the terms, held per element to twice the dot product's
    # gamma_n sum |a_i b_i| (both sides reduce n terms); the separable
    # representations need fewer rows than planes, the table one each.
    positions = np.linspace(0.0, 0.01, 7)
    terms, basis = power_weight_factors(problem, positions)
    assert basis.shape == (positions.size, terms.shape[0])
    assert terms.shape[1:] == (1, 1, grid.n_omega)
    assert (terms.shape[0] < positions.size) == (representation != "table")
    for row, z in zip(basis, positions, strict=True):
        reduction = np.tensordot(np.abs(row), np.abs(terms), axes=1)
        bound = 2 * 2 * terms.shape[0] * EPS * reduction
        np.testing.assert_array_less(
            np.abs(
                np.tensordot(row, terms, axes=1) - power_weight_at(problem, float(z))
            ),
            bound + np.finfo(float).tiny,
        )


def test_weight_factors_refuse_a_plane_outside_the_span():
    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    # No range warning: optmat's models are qualified here (SEP 3, Phase 4a).
    with nullcontext():
        mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45)
        problem = assemble(
            grid,
            mode,
            [],
            np.ones((1, 1, grid.n_omega), dtype=complex),
            controller=PIController(rtol=1e-6, atol=1e-12),
            save_planes=[0.01],
            dz_max=0.01,
            dz_init=0.001,
        )
    with pytest.raises(TranslationError, match="outside"):
        power_weight_factors(problem, np.array([0.0, 1.0]))


def test_gnlse_unit_index_phi_is_unchanged():
    # The GNLSE's own phi and the unit-index build tabulate the same operator: the
    # one-evaluation 16 eps, normwise.
    grid = EnvelopeGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = problem_from([0.0, 0.0, -1e-26], omega_ref=grid.omega0, gamma=0.1).mode
    old = _dispersion_phi(mode, grid, Frame.comoving(grid), z_max=0.01)
    result = build_phi(mode, grid, Frame.comoving(grid), z_max=0.01, index=1.0)
    assert old.phi_constant is not None and result.phi_constant is not None
    assert_within(result.phi_constant, old.phi_constant, 16 * EPS)
    assert old.phi_coefficients is not None and result.phi_coefficients is not None
    assert_within(result.phi_coefficients, old.phi_coefficients, 16 * EPS)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_unitary_noise_drift_stays_below_three_tenths_percent():
    grid = FieldGrid.from_limits(0.001, 800e-9, (400e-9, 2e-6), 100e-15)
    mode = MarcatiliMode(
        125e-6,
        "Ar",
        PressureGradient.two_point(grid.length, 20.0, 1.0),
        cladding=1.45,
        loss=False,
    )
    omega = grid.omega[grid.active_mask]
    ratio = np.sqrt(
        np.real(mode.neff(omega, grid.length)) / np.real(mode.neff(omega, 0.0))
    )
    # 0.3%, fixed before measurement. ADR-008 (docs/dev/decisions.md) rotates
    # internal noise unitarily, leaving this true-power drift uncompensated.
    assert np.max(np.abs(ratio - 1.0)) < 0.003
    assert np.max(np.abs(1.0 / ratio - 1.0)) < 0.003


def test_a_dispersion_specified_mode_needs_the_prefactor_index():
    # A Taylor beta carries no physical effective index: Re n_eff = beta c / omega
    # vanishes at the reference when beta(omega_ref) = 0, so the power row cannot be
    # built at all and the refusal has to name the override rather than read as a
    # numerical accident.
    grid = EnvelopeGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = SpecifiedMode.from_taylor([0.0, 0.0, 2e-26], omega_ref=grid.omega0)
    launch = np.ones((1, 1, grid.n_omega), dtype=complex)
    settings = {
        "controller": PIController(rtol=1e-6, atol=1e-12),
        "save_planes": [0.01],
        "dz_max": 0.01,
        "dz_init": 0.001,
    }
    with pytest.raises(ModeError, match=r"needs prefactor_index"):
        assemble(grid, mode, [], launch, **settings)
    # The remedy the message names: the GNLSE interface passes 1.0, and then the row
    # is exactly one everywhere and the launch is untouched.
    problem = assemble(grid, mode, [], launch, prefactor_index=1.0, **settings)
    np.testing.assert_array_equal(power_weight_at(problem, 0.0), 1.0)
    np.testing.assert_array_equal(problem.initial.field0[0], launch)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("representation", ["constant", "gradient"])
def test_the_entrance_conversion_reads_the_weight_row(representation):
    # The launch must be divided by the *row*, not by a second direct evaluation of
    # the mode: the two differ at the last bits on a gradient, by 3e-15 of the
    # requested launch.
    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(
        125e-6,
        "Ar",
        20.0
        if representation == "constant"
        else PressureGradient.two_point(0.01, 20.0, 1.0),
        cladding=1.45,
    )
    launch = np.linspace(1.0, 2.0, grid.n_omega)[None, None] * (1.0 + 0.5j)
    noise = modified_noise(grid, np.random.default_rng(42))
    problem = assemble(
        grid,
        mode,
        [],
        launch,
        noise=noise,
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[0.01],
        dz_max=0.01,
        dz_init=0.001,
    )
    weight = power_weight_at(problem, 0.0)
    # One rounding for the divide at the entrance and one for the multiply back: the
    # round trip through the same row cannot be worse than 2 eps, and is not.
    bound = 2 * np.finfo(float).eps
    np.testing.assert_allclose(
        power_amplitude(problem.initial.field0[0], weight), launch, rtol=bound, atol=0
    )
    assert problem.initial.noise0 is not None
    np.testing.assert_allclose(
        power_amplitude(problem.initial.noise0[0], weight),
        noise.spectrum,
        rtol=bound,
        atol=0,
    )


def test_added_phi_tables_keep_each_interval_its_own_degree():
    # A stretch that needs degree one and a stretch that needs a high degree, either
    # side of a mandatory edge: the sum must carry each interval's own degree. Padding
    # them all to the maximum made the diagnostic claim a cost that is not there and
    # would mislead any evaluator that trusted the field.
    def integrand(z):
        values = np.where(z < 0.5, 1.0, np.cos(40.0 * (z - 0.5)))
        return values[:, None, None, None, None] + 0j

    integral = build(integrand, 0.0, 1.0, mandatory_breakpoints=(0.5,)).integral()
    endpoint = build(
        lambda z: (3.0 * z)[:, None, None, None, None] + 0j,
        0.0,
        1.0,
        mandatory_breakpoints=(0.5,),
    )
    combined = _add_phi_tables(integral, endpoint)
    assert int(combined.degrees.min()) < int(combined.degrees.max())
    edges = zip(combined.breakpoints[:-1], combined.breakpoints[1:], strict=True)
    for number, (low, high) in enumerate(edges):
        old = integral.degrees[integral.interval_containing(float((low + high) / 2))]
        assert int(combined.degrees[number]) == max(
            1, int(old), int(endpoint.degrees[number])
        )
        # Everything beyond that degree is exact zero padding.
        keep = int(combined.degrees[number]) + 1
        np.testing.assert_array_equal(combined.coefficients[number, keep:], 0.0)
    # Because that padding is exact zeros, a Clenshaw pass over only degrees[i] + 1
    # coefficients reproduces the padded one exactly, the extra Clenshaw steps being
    # exact no-ops on exact zeros: the property that makes the
    # narrower degrees safe for an evaluator to act on.
    for z in np.linspace(0.0, 1.0, 37):
        number = combined.interval_containing(float(z))
        low = float(combined.breakpoints[number])
        high = float(combined.breakpoints[number + 1])
        keep = int(combined.degrees[number]) + 1
        np.testing.assert_array_equal(
            chebyshev_basis.chebval(
                (2.0 * z - low - high) / (high - low),
                combined.coefficients[number, :keep],
            ),
            combined(float(z)),
        )


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_the_gradient_phi_table_is_not_padded_to_one_degree():
    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(
        125e-6,
        "Ar",
        PressureGradient.two_point(0.01, 20.0, 1.0),
        cladding=1.45,
        loss=False,
    )
    table = build_phi(
        mode, grid, Frame.comoving(grid), z_max=0.01, representation="table"
    )
    dispersion = _dispersion_phi(
        mode, grid, Frame.comoving(grid), z_max=0.01, representation="table"
    )
    # The log term refines the intervals of Phi; adding it must not raise every
    # interval to the widest degree anywhere in the table.
    assert table.n_intervals >= dispersion.n_intervals
    assert int(table.phi_degrees.min()) < int(table.phi_degrees.max())


@pytest.mark.parametrize("index", [0.0, -1.0, np.inf, np.nan])
def test_phi_refuses_an_invalid_index_override(index):
    from solprop.exceptions import TranslationError  # noqa: PLC0415

    grid = FieldGrid.from_limits(0.01, 800e-9, (600e-9, 1e-6), 40e-15)
    mode = MarcatiliMode(125e-6, "Ar", 1.0, cladding=1.45)
    with pytest.raises(TranslationError, match="index must be positive and finite"):
        build_phi(mode, grid, Frame.comoving(grid), z_max=grid.length, index=index)
