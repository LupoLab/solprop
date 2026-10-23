# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import itertools
import logging
import math
from typing import cast

import numpy as np
import pytest
from numpy.polynomial.chebyshev import chebval
from scipy.integrate import quad, quad_vec

from bounds import EPS
from convergence_records import Column, Study, record
from solprop import fourier
from solprop.constants import C
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import (
    DEFAULT_LOSS_POLICY,
    Frame,
    dispersion_operator,
    frame_quantities,
    log_power_weight,
)
from solprop.modes import ChannelLabel, MarcatiliMode
from solprop.modes.protocol import Mode, ParameterisedMode
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.physdata import Gas
from solprop.profiles import (
    Constant,
    DensityPath,
    FromCallable,
    Linear,
    PressureGradient,
)
from solprop.pulses import GaussianPulse
from solprop.translate.chebyshev import DEFAULT_TOL, PiecewiseChebyshev
from solprop.translate.linop import (
    DEFAULT_MARGIN,
    ROUNDOFF_SAFETY,
    LinearOperator,
    _checked_representation,
    _operator_of_parameters,
    _operator_sampler,
    _roundoff_floor,
    build_phi,
    constant_phi,
    declared_paths,
    delta_phi,
    parameter_paths,
    profile_table,
    rate_weights,
    tabulate,
    unitary_phase,
    verify_phi,
)

# The active band runs from twice the longest requested wavelength to a tenth below the
# shortest, so a gas model is always extrapolated at its ends. That is a property of the
# band, not of what these tests measure.
pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

# quad_vec is asked for 1e-13 absolute and its own reported error is asserted before it
# is believed, in the style of tests/translate/test_chebyshev.py.
ORACLE_EPSABS = 1e-13


@pytest.fixture(scope="module")
def field_grid():
    return FieldGrid.from_limits(0.2, 800e-9, (400e-9, 2e-6), 400e-15)


@pytest.fixture(scope="module")
def gradient_mode():
    """A helium fill, optmat's, falling from 5 bar to 0.5 bar over the whole fibre."""
    return MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 5.0, 0.5), cladding=1.45
    )


@pytest.fixture(scope="module")
def gradient_table(field_grid, gradient_mode):
    return build_phi(gradient_mode, field_grid, Frame.comoving(field_grid), z_max=0.2)


@pytest.fixture(scope="module")
def gradient_z_table(field_grid, gradient_mode):
    """The same operator as the z table, the general fallback of contract v3."""
    return build_phi(
        gradient_mode,
        field_grid,
        Frame.comoving(field_grid),
        z_max=0.2,
        representation="table",
    )


@pytest.fixture(scope="module")
def gradient_sampler(field_grid, gradient_mode):
    return _operator_sampler(
        (gradient_mode,),
        field_grid,
        Frame.comoving(field_grid),
        1,
        DEFAULT_LOSS_POLICY,
        None,
    )


# ------------------------------------------------------------------- the constant path


def test_a_constant_operator_is_one_interval_of_degree_one():
    operator = np.array([0.0, -1j, -2j - 0.5])
    table = constant_phi(operator, 2.0)
    assert table.n_intervals == 1
    assert table.phi_degrees.tolist() == [1]
    assert table.phi_representation == "constant"
    assert table.phi_coefficients is not None
    assert table.phi_coefficients.shape == (1, 2, 1, 1, 1, 3)
    assert table.z_span == (0.0, 2.0 * (1.0 + DEFAULT_MARGIN))


@pytest.mark.parametrize("pair", [(0.0, 0.5), (0.25, 0.75), (0.5, 2.0)])
def test_the_constant_propagator_is_exactly_the_exponential_of_d_times_h(pair):
    """The fast path is the one product ``D (z2 - z1)``, not a difference of two sums.

    That form is the stable increment SEP 25 keeps protected, and one correctly rounded
    product is exact here by construction, so the contract is the form itself.
    """
    operator = np.array([-1j, -2j - 0.5, 3e-4 - 1j])
    table = constant_phi(operator, 2.0)
    z1, z2 = pair
    exact = np.exp(operator * (z2 - z1))
    assert np.array_equal(np.exp(delta_phi(table, z1, z2)).ravel(), exact)


def test_a_z_constant_mode_takes_the_constant_path():
    grid = EnvelopeGrid.from_limits(1.0, 1030e-9, (700e-9, 2e-6), 2e-12)
    mode = SpecifiedMode.from_taylor([0.0, 0.0, 2e-26], omega_ref=grid.omega0, loss=0.3)
    frame = Frame.comoving(grid)
    table = build_phi(mode, grid, frame, z_max=1.0)
    assert table.phi_representation == "constant"
    assert table.phi_constant is not None
    expected = dispersion_operator(mode, grid, frame)
    assert np.array_equal(table.phi_constant.ravel(), expected)


def test_the_constant_path_broadcasts_over_polarisation():
    table = constant_phi(np.array([-1j, -2j]), 1.0, n_pol=2)
    assert table.value_shape == (1, 1, 2, 2)
    assert table.phi_constant is not None
    assert np.array_equal(table.phi_constant[0, 0, 0], table.phi_constant[0, 0, 1])


def test_the_constant_path_accepts_the_canonical_layout():
    operator = np.zeros((1, 2, 1, 4), dtype=complex)
    assert constant_phi(operator, 1.0).value_shape == (1, 2, 1, 4)


def test_the_unitary_phase_of_a_constant_operator_is_the_imaginary_exponent():
    operator = np.array([-1j - 0.5, -2j])
    table = constant_phi(operator, 1.0)
    assert np.array_equal(unitary_phase(table, 0.4).ravel(), np.imag(operator) * 0.4)


# ------------------------------------------------------------------ the adaptive path


def test_the_table_covers_the_run_and_its_margin(gradient_table):
    assert gradient_table.z_span[0] == 0.0
    # Luna PR #440's five per cent, so the last step and the noise phase are defined at
    # the end of the fibre.
    assert gradient_table.z_span[1] == pytest.approx(1.05 * 0.2, rel=1e-15)


def test_the_hard_kinks_are_the_profiles_breakpoints(gradient_table, gradient_mode):
    interior = [
        value
        for value in gradient_mode.density.breakpoints
        if 0.0 < value < gradient_table.z_span[1]
    ]
    assert gradient_table.hard_kinks.tolist() == interior
    # And every kink is an interval edge, which is what the representation is for.
    edges = gradient_table.phi_breakpoints
    for kink in gradient_table.hard_kinks:
        assert np.min(np.abs(edges - kink)) < 1e-12


def test_a_taper_and_a_gradient_both_contribute_their_kinks(field_grid):
    mode = MarcatiliMode(
        Linear(125e-6, 75e-6, 0.15),
        "He",
        PressureGradient.from_stages([0.0, 0.1, 0.2], [5.0, 5.0, 0.5]),
        cladding=1.45,
    )
    table = build_phi(mode, field_grid, Frame.comoving(field_grid), z_max=0.2)
    # The taper's corner, the fill's middle stage and the fill's end, in order; the end
    # is interior to the table because of the margin beyond the fibre.
    assert table.hard_kinks.tolist() == [0.1, 0.15, 0.2]


# The record of the acceptance test below, the "cumulative operator Phi against
# quad_vec" row of `docs/dev/accuracy.md` section 1. Both representations of contract v3
# are measured against the same oracle on the same twenty intervals, which is where the
# page's two numbers come from. The deviation sits near 1e-10 relative, set by the
# oracle's own error estimate and the table's interpolation of an operator whose
# round-off floor is about 3e-10 per element (the frame's eight-digit cancellation), so
# it is *not* a platform-stable number to nine digits: the comparison is a factor of
# two, which still catches any change worth the name. Each interval is asserted against
# its own bound (the `bound` column), derived from the tolerance the table is built to.
PHI_QUADRATURE = Study(
    name="phi_quadrature",
    title="Cumulative operator against an independent quadrature of the same operator",
    case={
        "mode": "Marcatili 125 um, optmat's helium 5 bar to 0.5 bar over 0.2 m, "
        "cladding 1.45",
        "grid": "field grid, 800 nm, 400 nm to 2 um, 400 fs, 0.2 m",
        "frame": "co-moving",
        "oracle": "scipy.integrate.quad_vec at epsabs 1e-13, its own error asserted "
        "below 1e-9 of the value scale before it is believed",
        "intervals": "twenty random pairs from default_rng(20240905)",
    },
    axis={"name": "pair", "unit": "1", "label": "interval pair, sorted by width"},
    columns=(
        Column(
            name="deviation",
            unit="1 (relative to the value scale of the interval)",
            why="a factor of two: the quantity sits near 1e-10, at the oracle's own "
            "error and the operator's round-off, so its last digits are not "
            "reproducible, while any real change moves it by orders",
            rtol=1.0,
            atol=1e-15,
        ),
        Column(
            name="bound",
            unit="1 (relative to the value scale of the interval)",
            why="the table's build tolerance carried through the integral, plus the "
            "oracle's own error estimate; recorded for the figure, not compared, since "
            "the round-off floor it is built from is a noise measurement",
        ),
        Column(
            name="width",
            unit="m",
            why="the interval's length, an input of the ladder rather than a "
            "measurement, so it is compared exactly",
            rtol=0.0,
        ),
    ),
)


def test_phi_reproduces_an_independent_quadrature_of_the_operator(
    gradient_table,
    gradient_z_table,
    gradient_sampler,
    field_grid,
    gradient_mode,
    request,
):
    """The acceptance test of the table (`docs/dev/stepper.md` section 4, step 3).

    Both representations of contract v3 are measured, because
    `docs/dev/accuracy.md` section 1 quotes both and the figure draws both: the
    expansion, which is what a parameterised mode gets, and the z table, which is the
    general fallback for a mode without parameters.

    The bound is the table's own (SEP 3, Q34). `build_phi` resolves ``D`` to
    ``tol = max(DEFAULT_TOL, ROUNDOFF_SAFETY * floor)`` relative to its value scale,
    with the floor measured as the build measures it, so integrated over ``[z1, z2]``
    the table departs from the exact ``Phi`` by at most ``tol max|D| (z2 - z1)``; the
    oracle adds its own error estimate. Both are taken relative to the interval's
    value scale, as the deviation is.
    """
    floor = _roundoff_floor(gradient_sampler, gradient_table.z_span[1])
    tolerance = max(DEFAULT_TOL, ROUNDOFF_SAFETY * floor)
    rng = np.random.default_rng(20240905)
    pairs = [tuple(sorted(rng.uniform(0.0, 0.2, 2))) for _ in range(20)]
    tables = {"expansion": gradient_table, "z table": gradient_z_table}
    rows = []
    for index, (z1, z2) in enumerate(sorted(pairs, key=lambda pair: pair[1] - pair[0])):
        largest = [0.0]

        def sampled(z, largest=largest):
            value = gradient_sampler(float(z))
            largest[0] = max(largest[0], float(np.max(np.abs(value))))
            return value

        reference, error = quad_vec(sampled, z1, z2, epsabs=ORACLE_EPSABS)
        correction = np.zeros(field_grid.n_omega)
        active = field_grid.active_mask
        correction[active] = -0.5 * (
            log_power_weight(gradient_mode, field_grid.omega[active], z2)
            - log_power_weight(gradient_mode, field_grid.omega[active], z1)
        )
        reference = reference + correction
        scale = float(np.max(np.abs(reference)))
        # The oracle must be converged before it can be believed.
        assert float(np.max(np.abs(error))) < 1e-9 * scale
        bound = (tolerance * largest[0] * (z2 - z1) + float(error)) / scale
        for name, table in tables.items():
            deviation = np.abs(delta_phi(table, z1, z2) - reference)
            relative = float(np.max(deviation)) / scale
            assert relative < bound, (name, index)
            rows.append(
                {
                    "series": name,
                    "axis": index,
                    "deviation": relative,
                    "bound": bound,
                    "width": z2 - z1,
                }
            )
    # Measured on optmat's helium: the floor 2.8e-10 and so a tolerance of 2.8e-8; the
    # worst deviation 4.6e-11 for the expansion and 1.4e-10 for the z table, two
    # orders inside it. On the RDW capillary they are 4.8e-12 and 2.5e-12.
    record(PHI_QUADRATURE, request=request, rows=rows)


def test_the_unitary_phase_is_the_imaginary_part_of_phi(gradient_table):
    z = 0.137
    assert np.array_equal(
        unitary_phase(gradient_table, z),
        np.imag(gradient_table.phi_expansion(z)),
    )


def test_the_right_endpoint_rule_shows_the_predicted_error(
    gradient_table, gradient_sampler, field_grid, gradient_mode
):
    """A right-endpoint linear operator is first order in h, and by exactly this much.

    Evaluated at the later endpoint only, the accumulated exponent is `D(z1) h` rather
    than the integral and the error is `(h/2)[D(z1) - D(z0)] + O(h^2)`. Reproduced here
    as a regression witness, never as a scheme: the residual after subtracting the
    prediction falls one order faster than the prediction itself.
    """
    z0 = 0.05
    ratios = []
    for step in (0.02, 0.01, 0.005):
        z1 = z0 + step
        correction = np.zeros(field_grid.n_omega)
        active = field_grid.active_mask
        correction[active] = -0.5 * (
            log_power_weight(gradient_mode, field_grid.omega[active], z1)
            - log_power_weight(gradient_mode, field_grid.omega[active], z0)
        )
        rule = gradient_sampler(z1) * step + correction
        predicted = 0.5 * step * (gradient_sampler(z1) - gradient_sampler(z0))
        residual = rule - delta_phi(gradient_table, z0, z1) - predicted
        ratios.append(
            float(np.max(np.abs(residual))) / float(np.max(np.abs(predicted)))
        )
    # The prediction is O(h^2) and the residual O(h^3), so the ratio is O(h) and halves
    # with the step. Allow a factor of two either way on each halving.
    assert ratios[-1] < 0.02
    for coarse, fine in itertools.pairwise(ratios):
        assert 0.25 < fine / coarse < 1.0


def test_the_comoving_frame_holds_the_pulse_in_the_window(field_grid, gradient_mode):
    """A frame frozen at the entrance walks the pulse out; the co-moving one does not.

    The walk is exactly the group delay the frame fails to remove,
    `integral of (beta_1(z) - beta_1(0)) dz`.
    """
    comoving = Frame.comoving(field_grid)
    reference = frame_quantities(gradient_mode, comoving, 0.0)[1]
    frozen = Frame(omega0=field_grid.omega0, subtract_beta0=False, beta1=reference)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(field_grid)

    def centroid(frame):
        table = build_phi(gradient_mode, field_grid, frame, z_max=0.2)
        state = np.exp(delta_phi(table, 0.0, 0.2)[0, 0, 0]) * start[0, 0]
        signal = fourier.analytic_signal(fourier.inverse(state, kind="field"))
        power = np.abs(signal) ** 2
        return float(np.sum(field_grid.t * power) / np.sum(power))

    predicted = quad(
        lambda z: frame_quantities(gradient_mode, comoving, float(z))[1] - reference,
        0.0,
        0.2,
    )[0]
    # Measured -34.9 fs against a 455 fs window: a third of the way to the edge.
    assert centroid(frozen) == pytest.approx(predicted, rel=2e-3)
    # The co-moving frame leaves the centroid at 1.7e-17 s, four parts in 10^5 of the
    # 455 fs window and a numerical floor rather than a physical displacement.
    assert abs(centroid(comoving)) < 1e-16


def test_the_frame_quantities_are_tabulated_alongside_phi(
    gradient_table, gradient_mode, field_grid
):
    """A field grid keeps beta_0, so only beta_1 is removed and tabulated."""
    comoving = Frame.comoving(field_grid)
    assert gradient_table.beta0 is not None
    assert gradient_table.beta1 is not None
    assert gradient_table.beta0.z_span == gradient_table.z_span
    assert gradient_table.beta1.z_span == gradient_table.z_span
    # The fill's end is a kink of the frame quantities as much as of the operator, and
    # is declared rather than located: told about it the table needs 3 intervals and is
    # good to 7e-15; left to find it, 19 intervals and 1e-11 at the kink itself.
    assert gradient_table.beta1.hard_kinks.tolist() == [0.2]
    assert gradient_table.beta1.degrees.size <= 4
    for z in (0.0, 0.09, 0.2, 0.21):
        expected = frame_quantities(gradient_mode, comoving, z)[1]
        # Measured 6.8e-15 over 43 positions; the bound leaves over an order.
        assert float(gradient_table.beta1(z)) == pytest.approx(expected, rel=1e-13)
        assert float(gradient_table.beta0(z)) == 0.0


def test_every_mode_of_a_set_is_written_in_the_first_mode_s_frame(field_grid):
    """One frame for the set, so the modes stay simultaneous.

    The frame's ``beta_1 omega`` term is a change of the time variable. Given one mode
    at a time, ``Frame.comoving`` recomputes it from *that* mode, which puts every mode
    in its own time origin: the set is then no longer simultaneous, and because the
    nonlinear term multiplies the modes at the same time sample, the group-delay
    walk-off between them is silently thrown away. That is what
    ``docs/guide/numerics/linear-operator.md`` forbids by writing ``beta_0(z)`` and
    ``beta_1(z)`` without a mode index, and it was worth two to four orders of
    magnitude in the higher modes of the M2 problem.

    Asserted here on the operator itself: mode ``p``'s column must be the operator its
    own ``beta`` gives *minus the first mode's* frame term, and the difference between
    two columns must be exactly ``-i (beta_p - beta_1)``, which carries the walk-off.
    """
    modes = tuple(MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2, 3))
    frame = Frame.comoving(field_grid)
    sampler = _operator_sampler(modes, field_grid, frame, 1, DEFAULT_LOSS_POLICY, None)
    operator = sampler(0.0)[0, :, 0, :]
    active = field_grid.active_mask
    omega = field_grid.omega[active]
    shared = frame_quantities(modes[0], frame, 0.0)[1]
    for index, mode in enumerate(modes):
        beta = omega * np.real(np.asarray(mode.neff(omega, 0.0))) / C
        expected = -1j * (beta - shared * omega)
        # The imaginary part is a cancellation of beta ~ 8e6 down to ~1e2, so the
        # comparison is absolute at the round-off of that reduction, not relative.
        np.testing.assert_allclose(
            np.imag(operator[index][active]), np.imag(expected), rtol=0.0, atol=1e-6
        )
    # And the walk-off itself: at the carrier the two columns differ by the difference
    # of the propagation constants, 100.5 rad/m for HE_11 against HE_12 here. Written
    # in its own frame each mode would show only its dispersion about its own delay,
    # and this difference would be wrong by the group-delay term.
    carrier = int(np.argmin(abs(field_grid.omega - field_grid.omega0)))
    at_carrier = field_grid.omega[carrier : carrier + 1]

    def beta_of(mode):
        index = float(np.real(np.asarray(mode.neff(at_carrier, 0.0))[0]))
        return float(at_carrier[0]) * index / C

    betas = [beta_of(mode) for mode in modes]
    for index in (1, 2):
        difference = np.imag(operator[index][carrier] - operator[0][carrier])
        assert difference == pytest.approx(-(betas[index] - betas[0]), rel=1e-6)


def test_an_envelope_set_shares_the_first_mode_s_carrier_phase_too():
    """The same rule for ``beta_0``, which an envelope grid does remove.

    Per-mode frames would give each mode its own carrier phase as well as its own time
    origin; the third-harmonic term's ``exp(2 i omega_0 t)`` factor and the modes'
    relative phase both depend on there being one. The first mode's operator vanishes at
    the carrier, as it does for a single mode, and the others carry only their offset
    from it.
    """
    grid = EnvelopeGrid.from_limits(0.2, 800e-9, (400e-9, 2e-6), 400e-15)
    modes = tuple(MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2))
    frame = Frame.comoving(grid)
    assert frame.subtract_beta0
    operator = _operator_sampler(modes, grid, frame, 1, DEFAULT_LOSS_POLICY, None)(0.0)
    carrier = int(np.argmin(abs(grid.omega - grid.omega0)))
    columns = operator[0, :, 0, carrier]
    # The reference mode is exactly in its own frame at the carrier; the loss is all
    # that is left of it, and the bound is the round-off of beta ~ 8e6 - beta_0.
    assert abs(np.imag(columns[0])) < 1e-6
    at_carrier = grid.omega[carrier : carrier + 1]

    def beta_of(mode):
        index = float(np.real(np.asarray(mode.neff(at_carrier, 0.0))[0]))
        return float(at_carrier[0]) * index / C

    expected = -(beta_of(modes[1]) - beta_of(modes[0]))
    assert float(np.imag(columns[1])) == pytest.approx(expected, rel=1e-6)


def test_a_frame_given_both_quantities_is_used_as_it_is(field_grid):
    """A caller who fixes the frame keeps it: the set is not re-referenced."""
    modes = tuple(MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2))
    fixed = Frame(field_grid.omega0, subtract_beta0=False, beta0=0.0, beta1=0.0)
    operator = _operator_sampler(
        modes, field_grid, fixed, 1, DEFAULT_LOSS_POLICY, None
    )(0.0)[0, :, 0, :]
    active = field_grid.active_mask
    omega = field_grid.omega[active]
    for index, mode in enumerate(modes):
        beta = omega * np.real(np.asarray(mode.neff(omega, 0.0))) / C
        np.testing.assert_allclose(
            np.imag(operator[index][active]), -beta, rtol=1e-12, atol=0.0
        )


def test_an_envelope_grid_removes_and_tabulates_beta_zero(gradient_mode):
    grid = EnvelopeGrid.from_limits(0.2, 800e-9, (400e-9, 2e-6), 400e-15)
    frame = Frame.comoving(grid)
    table = build_phi(gradient_mode, grid, frame, z_max=0.2)
    assert table.beta0 is not None
    for z in (0.0, 0.13, 0.2):
        expected = frame_quantities(gradient_mode, frame, z)[0]
        assert expected > 0.0
        assert float(table.beta0(z)) == pytest.approx(expected, rel=1e-13)


# ----------------------------------------------------------------- tolerance and budget


def test_the_tolerance_is_raised_to_the_measured_round_off_floor(
    field_grid, gradient_mode, caplog
):
    with caplog.at_level(logging.INFO, logger="solprop.translate.linop"):
        build_phi(gradient_mode, field_grid, Frame.comoving(field_grid), z_max=0.2)
    assert "round-off floor" in caplog.text


def test_the_round_off_floor_of_an_exact_operator_is_zero():
    constant = np.array([[[[1.0 + 0j, 2.0]]]])
    assert _roundoff_floor(lambda z: constant, 1.0) == 0.0


def test_the_round_off_floor_of_a_vanishing_operator_is_zero():
    """A probe where the operator is identically zero has no scale, and is skipped."""
    zeros = np.zeros((1, 1, 1, 4), dtype=complex)
    assert _roundoff_floor(lambda z: zeros, 1.0) == 0.0


def test_the_floor_is_measured_relative_to_the_operator(field_grid, gradient_mode):
    sampler = _operator_sampler(
        (gradient_mode,),
        field_grid,
        Frame.comoving(field_grid),
        1,
        DEFAULT_LOSS_POLICY,
        None,
    )
    floor = _roundoff_floor(sampler, 0.21)
    # Measured 2.8e-10 here and 2.3e-7 on the M2 capillary with its spectral absorber.
    # The frame subtracts about eight digits from beta, so D itself is known to about
    # 1e-11 relative at the envelope; measured per element against that element's own
    # value, as the acceptance test measures the tail it must bound, the worst element
    # is orders worse and that is the number an interpolant cannot resolve past.
    assert 1e-12 < floor < 1e-5
    assert ROUNDOFF_SAFETY * floor > 1e-11


def test_an_exhausted_interval_budget_relaxes_the_tolerance_and_warns(
    field_grid, gradient_mode, caplog
):
    with caplog.at_level(logging.WARNING, logger="solprop.translate.linop"):
        table = build_phi(
            gradient_mode,
            field_grid,
            Frame.comoving(field_grid),
            z_max=0.2,
            n_max=4,
            max_intervals=1,
        )
    assert "interval budget" in caplog.text
    assert table.n_intervals >= 1


# --------------------------------------------------------------------------- tabulating


def test_a_profile_is_tabulated_with_its_breakpoints_as_kinks():
    table = profile_table(Linear(1.0, 3.0, 2.0), 0.0, 2.1)
    assert float(table(1.0)) == pytest.approx(2.0)
    assert table.hard_kinks.tolist() == [2.0]
    assert not table.coefficients.flags.writeable


def test_a_constant_profile_needs_no_kinks():
    table = profile_table(Constant(4.0), 0.0, 1.0)
    assert table.hard_kinks.size == 0
    assert float(table(0.5)) == 4.0


def test_a_kink_outside_the_range_is_dropped_rather_than_refused():
    """The margin asks a path for values beyond its own span, where it holds its end."""
    table = tabulate(np.asarray, 0.0, 1.0, kinks=[0.5, 2.0])
    assert table.hard_kinks.tolist() == [0.5]


def _edge_values(table):
    """Return each interval's value at its two ends, evaluated as the backends do."""
    return np.array([chebval(np.array([-1.0, 1.0]), row) for row in table.coefficients])


def _within_tol_of(table, fitted):
    """Whether each interval moved by at most DEFAULT_TOL of its largest coefficient."""
    change = np.abs(table.coefficients - fitted.coefficients).sum(axis=1)
    return bool(np.all(change <= DEFAULT_TOL * np.abs(fitted.coefficients).max(axis=1)))


def test_a_pressure_falling_to_vacuum_tabulates_an_exact_zero_density():
    """Helium at 1.5 bar falling linearly to 0 bar over a millimetre.

    Fitted alone, its density table is -1.0e11 m^-3 at the vacuum end, 3e-15 of its
    scale, and the problem's domain check refused the run (SEP 3, Q23).
    """
    path = DensityPath(Gas.He, Linear(1.5, 0.0, 1e-3))
    span = 1e-3 * (1.0 + DEFAULT_MARGIN)
    fitted = tabulate(path.__call__, 0.0, span, kinks=path.breakpoints)
    table = profile_table(path, 0.0, span)
    assert _edge_values(fitted)[0, 1] != 0.0
    assert _edge_values(table)[0, 1] == 0.0
    assert _edge_values(table)[0, 0] == pytest.approx(_edge_values(fitted)[0, 0])
    assert _within_tol_of(table, fitted)
    assert not table.coefficients.flags.writeable


def _vacuum_paths(length: float, scale: float, shape: float) -> list[FromCallable]:
    """Return smooth paths exactly zero at their ends or at a declared kink."""

    def unit(z):
        return z / length

    return [
        FromCallable(
            lambda z: scale * unit(z) * (1.0 - unit(z)) * np.exp(shape * unit(z)),
            (0.0, length),
        ),
        FromCallable(
            lambda z: scale * (1.0 - unit(z)) * (1.0 + 0.5 * shape * unit(z) ** 2),
            (0.0, length),
        ),
        FromCallable(
            lambda z: scale * np.abs(unit(z) - 0.5) * np.exp(shape * unit(z)),
            (0.0, length),
            kinks=np.array([0.5 * length]),
        ),
    ]


def test_every_exact_zero_of_a_path_at_an_edge_is_an_exact_zero_of_its_table():
    corrected = 0
    for length, scale, shape in itertools.product(
        (1e-3, 0.3, 2.0), (1e-3, 1.0, 3e25), (-1.5, 0.4, 2.0)
    ):
        for path in _vacuum_paths(length, scale, shape):
            fitted = tabulate(path.__call__, 0.0, length, kinks=path.breakpoints)
            table = profile_table(path, 0.0, length)
            vacuum = path(table.breakpoints) == 0.0
            edges = _edge_values(table)
            assert np.all(edges[vacuum[:-1], 0] == 0.0)
            assert np.all(edges[vacuum[1:], 1] == 0.0)
            assert _within_tol_of(table, fitted)
            corrected += not np.array_equal(table.coefficients, fitted.coefficients)
    # The fit alone already lands on some of these zeros; the rest are the test.
    assert corrected > 0


def test_a_square_root_vacuum_end_keeps_its_fitted_table():
    """Its end, sampled one ulp inside, is 1e-8 of the scale: that is not rounding."""
    path = FromCallable(lambda z: 3e25 * np.sqrt(np.maximum(1.0 - z, 0.0)), (0.0, 1.0))
    fitted = tabulate(path.__call__, 0.0, 1.0, kinks=path.breakpoints)
    table = profile_table(path, 0.0, 1.0)
    assert np.array_equal(table.coefficients, fitted.coefficients)
    assert _edge_values(table)[-1, 1] > 0.0


# ------------------------------------------------------------------------- verification


def test_verify_phi_reports_agreement_for_an_exact_table():
    operator = np.array([[[[-1j, -2j]]]])
    table = constant_phi(operator, 1.0)
    assert verify_phi(table, lambda z: operator, [(0.0, 1.0), (0.2, 0.6)]) < 1e-14


def test_verify_phi_reports_the_absolute_deviation_when_the_operator_vanishes():
    zeros = np.zeros((1, 1, 1, 2), dtype=complex)
    table = constant_phi(zeros, 1.0)
    assert verify_phi(table, lambda z: zeros, [(0.0, 1.0)]) == 0.0


def test_verify_phi_of_no_points_is_zero(gradient_table, gradient_sampler):
    assert verify_phi(gradient_table, gradient_sampler, []) == 0.0


# ------------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("z_max", "margin", "message"),
    [
        (0.0, 0.05, "z_max must be positive"),
        (math.inf, 0.05, "z_max must be positive"),
        (1.0, -0.1, "margin must be non-negative"),
        (1.0, math.nan, "margin must be non-negative"),
    ],
)
def test_constant_phi_rejects_a_bad_span(z_max, margin, message):
    with pytest.raises(TranslationError, match=message):
        constant_phi(np.zeros(4, dtype=complex), z_max, margin=margin)


def test_constant_phi_rejects_an_operator_of_the_wrong_rank():
    with pytest.raises(TranslationError, match="must have shape"):
        constant_phi(np.zeros((2, 4), dtype=complex), 1.0)


def test_constant_phi_rejects_an_impossible_polarisation_count():
    with pytest.raises(TranslationError, match="n_pol must be 1 or 2"):
        constant_phi(np.zeros(4, dtype=complex), 1.0, n_pol=3)


def test_constant_phi_rejects_a_non_finite_operator():
    with pytest.raises(TranslationError, match="must be finite"):
        constant_phi(np.array([0.0, np.nan], dtype=complex), 1.0)


def test_build_phi_rejects_an_empty_mode_set(field_grid):
    with pytest.raises(TranslationError, match="at least one mode"):
        build_phi([], field_grid, Frame.comoving(field_grid), z_max=1.0)


@pytest.mark.parametrize("z", [-0.1, 1.5])
def test_evaluating_outside_the_table_is_refused(z):
    table = constant_phi(np.zeros(4, dtype=complex), 1.0)
    with pytest.raises(TranslationError, match="lies outside the table"):
        delta_phi(table, 0.0, z)
    with pytest.raises(TranslationError, match="lies outside the table"):
        unitary_phase(table, z)


def test_the_group_refuses_a_breakpoint_count_that_does_not_match():
    with pytest.raises(TranslationError, match="phi_breakpoints has"):
        LinearOperator(
            phi_representation="table",
            phi_breakpoints=np.array([0.0, 0.5, 1.0]),
            phi_coefficients=np.zeros((1, 2, 1, 1, 1, 3), dtype=complex),
            phi_degrees=np.array([1]),
            hard_kinks=np.empty(0),
            phi_constant=None,
        )


def test_the_group_refuses_a_degree_count_that_does_not_match():
    """The degrees say how many intervals there are; the coefficients must agree."""
    with pytest.raises(TranslationError, match="phi_degrees has"):
        LinearOperator(
            phi_representation="table",
            phi_breakpoints=np.array([0.0, 1.0]),
            phi_coefficients=np.zeros((2, 2, 1, 1, 1, 3), dtype=complex),
            phi_degrees=np.array([1]),
            hard_kinks=np.empty(0),
            phi_constant=None,
        )


@pytest.mark.parametrize(
    ("kind", "constant"),
    [("constant", None), ("table", np.zeros((1, 1, 1, 3), dtype=complex))],
)
def test_the_group_refuses_a_constant_that_disagrees_with_the_kind(kind, constant):
    with pytest.raises(TranslationError, match="phi_constant must be present"):
        LinearOperator(
            phi_representation=kind,
            phi_breakpoints=np.array([0.0, 1.0]),
            phi_coefficients=np.zeros((1, 2, 1, 1, 1, 3), dtype=complex),
            phi_degrees=np.array([1]),
            hard_kinks=np.empty(0),
            phi_constant=constant,
        )


def test_the_group_refuses_a_constant_of_the_wrong_shape():
    with pytest.raises(TranslationError, match="phi_constant has shape"):
        LinearOperator(
            phi_representation="constant",
            phi_breakpoints=np.array([0.0, 1.0]),
            phi_coefficients=np.zeros((1, 2, 1, 1, 1, 3), dtype=complex),
            phi_degrees=np.array([1]),
            hard_kinks=np.empty(0),
            phi_constant=np.zeros((1, 1, 1, 4), dtype=complex),
        )


def test_the_group_reports_its_size_and_its_polynomial():
    table = constant_phi(np.zeros(4, dtype=complex), 1.0)
    assert table.nbytes == 2 * 4 * 16
    assert isinstance(table.polynomial, PiecewiseChebyshev)
    assert table.polynomial is table.polynomial


def test_a_tolerance_already_above_the_floor_is_left_alone(
    field_grid, gradient_mode, caplog
):
    with caplog.at_level(logging.INFO, logger="solprop.translate.linop"):
        build_phi(
            gradient_mode, field_grid, Frame.comoving(field_grid), z_max=0.2, tol=1e-6
        )
    assert "round-off floor" not in caplog.text


def test_the_build_verifies_itself_against_the_quadrature_at_debug_level(
    field_grid, gradient_mode, caplog
):
    """Step 3 of the construction runs as a spot check when the log asks for it."""
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.linop"):
        build_phi(gradient_mode, field_grid, Frame.comoving(field_grid), z_max=0.2)
    assert "quad_vec agreement" in caplog.text


def test_a_path_shared_by_two_modes_is_resolved_once(field_grid):
    """Two modes of one fibre share its radius and density objects, not copies."""
    fill = PressureGradient.two_point(0.2, 5.0, 0.5)
    modes = [MarcatiliMode(125e-6, "He", fill, m=m, cladding=1.45) for m in (1, 2)]
    assert modes[0].density is not modes[1].density
    paths = declared_paths(modes)
    # Distinct DensityPath objects built from one fill: both are resolved, and the
    # table carries the operator of both modes.
    assert len(paths) == 2
    table = build_phi(modes, field_grid, Frame.comoving(field_grid), z_max=0.2, n_pol=1)
    assert table.value_shape[1] == 2
    same = declared_paths([modes[0], modes[0]])
    assert len(same) == 1


# ------------------------------------------------------- expansion or z table


def test_a_gradient_capillary_takes_the_expansion(gradient_table):
    """`D` is analytic in the density, so the search resolves it in one box."""
    assert gradient_table.phi_representation == "expansion"
    assert gradient_table.phi_coefficients is None
    expansion = gradient_table.phi_expansion
    assert expansion is not None
    assert expansion.parameters == ("density",)
    # The z structure of Phi is the scalar tables' and nothing else.
    assert np.array_equal(gradient_table.phi_breakpoints, expansion.weights.breakpoints)
    assert np.array_equal(gradient_table.phi_degrees, expansion.weights.degrees)
    assert np.array_equal(gradient_table.hard_kinks, expansion.weights.hard_kinks)


def test_a_taper_and_a_gradient_expand_in_both_parameters(field_grid):
    mode = MarcatiliMode(
        Linear(125e-6, 75e-6, 0.15),
        "He",
        PressureGradient.from_stages([0.0, 0.1, 0.2], [5.0, 5.0, 0.5]),
        cladding=1.45,
    )
    table = build_phi(mode, field_grid, Frame.comoving(field_grid), z_max=0.2)
    expansion = table.phi_expansion
    assert expansion is not None
    assert expansion.parameters == ("core_radius", "density")
    assert expansion.endpoint_degrees is not None
    assert expansion.n_terms == int(
        np.prod(expansion.degrees + 1) + np.prod(expansion.endpoint_degrees + 1)
    )


def test_the_expansion_is_smaller_and_faster_to_evaluate_than_the_z_table(
    field_grid, gradient_mode, gradient_table
):
    """The point of the representation, on the case that motivated it.

    Measured on the RDW gradient capillary (1.5 m of helium 5 to 0.1 bar, 90 nm to
    4 um, 2049 bins): 9 terms and 0.32 MB against 8 intervals of degree 64 and 16.5 MB,
    a factor of 52. Here, on the smaller test capillary, the factor is about 20; the
    assertion is a factor of two, which no plausible change of the search could reach
    while still resolving the operator.
    """
    fallback = build_phi(
        gradient_mode,
        field_grid,
        Frame.comoving(field_grid),
        z_max=0.2,
        representation="table",
    )
    assert fallback.phi_representation == "table"
    assert gradient_table.nbytes * 2 < fallback.nbytes


def gradient_capillary():
    """A helium fill falling from 5 bar to 0.5 bar over the whole fibre."""
    return MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 5.0, 0.5), cladding=1.45
    )


def tapered_capillary():
    """A bore narrowing from 125 um to 75 um over three quarters of the fibre."""
    return MarcatiliMode(Linear(125e-6, 75e-6, 0.15), "He", 1.0, cladding=1.45)


def tapered_and_staged():
    """Both parameters vary, and the fill has a stage boundary as well as a corner."""
    return MarcatiliMode(
        Linear(125e-6, 75e-6, 0.15),
        "He",
        PressureGradient.from_stages([0.0, 0.1, 0.2], [5.0, 5.0, 0.5]),
        cladding=1.45,
    )


def argon_full_model():
    """The square-root model on the dispersive default cladding, not the reduced one."""
    return MarcatiliMode(125e-6, "Ar", PressureGradient.two_point(0.2, 5.0, 0.5))


@pytest.mark.parametrize(
    "mode_of",
    [gradient_capillary, tapered_capillary, tapered_and_staged, argon_full_model],
    ids=["gradient", "taper", "taper and gradient", "full model on argon"],
)
def test_the_expansion_and_the_z_table_agree_on_delta_phi(field_grid, mode_of):
    """Every problem the version-2 table produced must produce the same increment.

    The bound is the tolerance the tables are built to. The round-off floor of `D`
    raises that to about 1e-9 relative here, and the two representations are two
    independent interpolants of the same noisy operator, so they may differ by it.
    Measured worst over the four cases: 5.8e-11, one to two orders inside the bound.
    """
    z_max = 0.2
    mode = mode_of()
    frame = Frame.comoving(field_grid)
    expansion = build_phi(mode, field_grid, frame, z_max=z_max)
    table = build_phi(mode, field_grid, frame, z_max=z_max, representation="table")
    assert expansion.phi_representation == "expansion"
    rng = np.random.default_rng(20260906)
    worst = 0.0
    for _ in range(20):
        z1, z2 = sorted(rng.uniform(0.0, z_max, 2))
        reference = delta_phi(table, z1, z2)
        scale = float(np.max(np.abs(reference)))
        deviation = np.abs(delta_phi(expansion, z1, z2) - reference)
        worst = max(worst, float(np.max(deviation)) / scale)
    assert worst < 1e-9


def test_a_z_constant_tabulated_mode_takes_neither(field_grid):
    """A mode that declares no parameters is z-constant here, and exact."""
    mode = SpecifiedMode.from_taylor([0.0, 0.0, 2e-26], omega_ref=field_grid.omega0)
    table = build_phi(mode, field_grid, Frame.comoving(field_grid), z_max=0.2)
    assert table.phi_representation == "constant"
    assert table.phi_constant is not None


def test_a_mode_set_that_does_not_declare_its_parameters_takes_the_z_table(
    field_grid, gradient_mode, caplog
):
    """A GNLSE mode beside a capillary one: there is no shared parameter space.

    A dispersion-specified mode declares its own path, the effective area, so the two
    disagree about names rather than one declaring nothing; the outcome, the z table,
    is the one this test is about. The branch for a mode that declares nothing at all is
    :func:`test_a_mode_that_declares_no_parameters_at_all_has_no_shared_space`,
    which needs a double: every mode in the package declares its paths.
    """
    other = SpecifiedMode.from_taylor([0.0, 0.0, 2e-26], omega_ref=field_grid.omega0)
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.linop"):
        table = build_phi(
            [gradient_mode, other], field_grid, Frame.comoving(field_grid), z_max=0.2
        )
    assert table.phi_representation == "table"
    assert "declare different parameters" in caplog.text


def test_modes_that_disagree_about_a_path_take_the_z_table(field_grid, caplog):
    """Two different fills in one mode set have two operators, not one."""
    modes = [
        MarcatiliMode(
            125e-6, "He", PressureGradient.two_point(0.2, 5.0, end), cladding=1.45
        )
        for end in (0.5, 1.0)
    ]
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.linop"):
        table = build_phi(modes, field_grid, Frame.comoving(field_grid), z_max=0.2)
    assert table.phi_representation == "table"
    assert "disagree about the path" in caplog.text


def test_two_modes_of_one_fibre_share_one_parameter_axis(field_grid):
    """Their density objects are distinct, so agreement is checked, not assumed."""
    fill = PressureGradient.two_point(0.2, 5.0, 0.5)
    modes = [MarcatiliMode(125e-6, "He", fill, m=m, cladding=1.45) for m in (1, 2)]
    assert modes[0].density is not modes[1].density
    varying = parameter_paths(modes, 0.0, 0.21)
    assert varying is not None
    assert sorted(varying) == ["density"]
    table = build_phi(modes, field_grid, Frame.comoving(field_grid), z_max=0.2)
    assert table.phi_representation == "expansion"
    assert table.value_shape[1] == 2


def test_an_expansion_that_cannot_be_built_is_refused_when_it_is_demanded(field_grid):
    mode = SpecifiedMode.from_taylor([0.0, 0.0, 2e-26], omega_ref=field_grid.omega0)
    # A z-constant mode takes the exact path whatever is asked for, so demand it of a
    # set that genuinely has no shared parameter space.
    modes = [
        mode,
        MarcatiliMode(
            125e-6, "He", PressureGradient.two_point(0.2, 5.0, 0.5), cladding=1.45
        ),
    ]
    with pytest.raises(TranslationError, match="could not be built"):
        build_phi(
            modes,
            field_grid,
            Frame.comoving(field_grid),
            z_max=0.2,
            representation="expansion",
        )


def test_an_unknown_representation_is_refused():
    """Checked directly: the test session's type checker never lets one through."""
    with pytest.raises(TranslationError, match="representation must be"):
        _checked_representation("chebyshev")


def test_an_expansion_has_no_z_table_to_read(gradient_table):
    """A backend evaluates through the functions, never by reading the coefficients."""
    with pytest.raises(TranslationError, match="no z table"):
        _ = gradient_table.polynomial


def test_the_floor_is_measured_in_the_acceptance_tests_own_norm():
    """Measured against the global envelope, one small column hides the noise.

    A synthetic operator with two columns four orders apart, both carrying the same
    absolute noise. Against the global envelope the noise reads as 1e-4 of it; per
    element, the small column reports its own 1.0, which is what the tail test will see
    and what no interpolant can resolve past.
    """
    rng = np.random.default_rng(19)
    large = 1.0e4
    small = 1.0

    def operator_at(z):
        noise = rng.normal(scale=1e-4, size=(1, 1, 1, 2))
        return np.array([[[[large, small]]]]) * (1.0 + 1e-9 * z) + noise

    floor = _roundoff_floor(operator_at, 1.0)
    # The small column's own noise is 1e-4 of itself; the large column's is 1e-8 of
    # itself. A second difference of three draws is a few times one draw, so the floor
    # sits within an order of 1e-4 and nowhere near the 1e-8 a global norm would give.
    assert 1e-5 < floor < 1e-2


# ------------------------------------------------------ the size the change is for


def ctc_sized_problem(length=1.5):
    """Return ``(grid, modes)`` at the comb-to-continuum target's size.

    The target is a 16 um bore, 1.5 m, hydrogen, six HE_1m modes, 170 nm to 4 um on a 4
    ps window -- a field grid of 16 385 bins. Its fill is a constant 40 bar, which would
    take the exact constant path and have no table to compare, so the fill here is a
    gradient of the same size: what is measured is the representation's cost at that
    shape.
    """
    grid = FieldGrid.from_limits(length, 515e-9, (170e-9, 4e-6), 4e-12)
    fill = PressureGradient.two_point(length, 40.0, 0.4)
    return grid, [MarcatiliMode(16e-6, "H2", fill, m=m) for m in range(1, 7)]


@pytest.mark.benchmark
def test_the_ctc_sized_problem_costs_a_fraction_of_the_version_two_table():
    """The gigabyte, measured both ways at the size that motivated the change.

    Measured once: the z table is 8 intervals of degree 64 and 792.0 MB, built in 3.53
    s, with `delta_phi` at 17.80 ms; the expansion is 33 terms and 49.7 MB, built in
    0.88 s, with `delta_phi` at 1.20 ms. With the prefactor's 759.0 MB beside it the
    version-2 pair is 1.55 GB, the order of a gigabyte that motivated the expansion; the
    pair of expansions is 64.8 MB.

    Marked `benchmark` because it builds the gigabyte for real, so no lane runs it; run
    it by hand with
    ``pytest tests/translate/test_linop.py -m benchmark``. The slow lane guards the same
    property at a reduced size below.
    """
    grid, modes = ctc_sized_problem()
    assert grid.n_omega == 16385
    frame = Frame.comoving(grid)
    expansion = build_phi(modes, grid, frame, z_max=1.5)
    table = build_phi(modes, grid, frame, z_max=1.5, representation="table")
    assert expansion.phi_representation == "expansion"
    # An order of magnitude, against the sixteen measured: the assertion is what the
    # change is for and not the machine it was measured on.
    assert expansion.nbytes * 10 < table.nbytes
    assert table.nbytes > 500 * 2**20
    rng = np.random.default_rng(20260906)
    worst = 0.0
    for _ in range(10):
        z1, z2 = sorted(rng.uniform(0.0, 1.5, 2))
        reference = delta_phi(table, z1, z2)
        deviation = np.abs(delta_phi(expansion, z1, z2) - reference)
        worst = max(worst, float(np.max(deviation)) / float(np.max(np.abs(reference))))
    # The round-off floor of this operator is 5.4e-9, so both are built to 5.4e-7 and
    # may differ by it; measured 5.0e-9, a hundred times inside the bound.
    assert worst < 5.4e-7


@pytest.mark.slow
def test_the_expansion_is_a_fraction_of_the_z_table_at_the_ctc_shape():
    """The same property as the benchmark, at a size the slow lane can afford."""
    grid = FieldGrid.from_limits(0.2, 515e-9, (170e-9, 4e-6), 4e-13)
    fill = PressureGradient.two_point(0.2, 40.0, 0.4)
    modes = [MarcatiliMode(16e-6, "H2", fill, m=m) for m in range(1, 7)]
    frame = Frame.comoving(grid)
    expansion = build_phi(modes, grid, frame, z_max=0.2)
    table = build_phi(modes, grid, frame, z_max=0.2, representation="table")
    assert expansion.phi_representation == "expansion"
    assert expansion.nbytes * 5 < table.nbytes


@pytest.mark.slow
def test_the_two_representations_agree_on_the_m2_reduced_case():
    """The reduced M2 capillary, whose witness the driver tests run.

    A 125 um bore, 1 m of helium falling from 1.2 bar to the vacuum floor, two HE_1m
    modes, 100 nm to 1.5 um on a 200 fs window: the near-vacuum end is the square-root
    cusp the z table splits four orders into, and the expansion leaves it in the scalar
    tables where it costs nothing.
    """
    grid = FieldGrid.from_limits(1.0, 800e-9, (100e-9, 1.5e-6), 200e-15)
    fill = PressureGradient.two_point(1.0, 1.2, 0.0)
    modes = [MarcatiliMode(125e-6, "He", fill, m=m) for m in (1, 2)]
    frame = Frame.comoving(grid)
    expansion = build_phi(modes, grid, frame, z_max=1.0)
    table = build_phi(modes, grid, frame, z_max=1.0, representation="table")
    assert expansion.phi_representation == "expansion"
    assert expansion.nbytes * 5 < table.nbytes
    sampler = _operator_sampler(modes, grid, frame, 1, DEFAULT_LOSS_POLICY, None)
    floor = _roundoff_floor(sampler, 1.05)
    rng = np.random.default_rng(20260906)
    worst = 0.0
    for _ in range(20):
        z1, z2 = sorted(rng.uniform(0.0, 1.0, 2))
        reference = delta_phi(table, z1, z2)
        deviation = np.abs(delta_phi(expansion, z1, z2) - reference)
        worst = max(worst, float(np.max(deviation)) / float(np.max(np.abs(reference))))
    # Two independent interpolants of the same noisy operator, each built to
    # ROUNDOFF_SAFETY times its measured floor, may differ by that tolerance. Measured
    # 4.9e-9 against a tolerance of 4.8e-5, and 9 terms and 0.37 MB against 19 intervals
    # of degree 33 and 20.2 MB.
    assert worst < ROUNDOFF_SAFETY * floor


def spoiled_expansion(gradient_table, **overrides):
    """Return the fields of an expansion-backed operator, with some replaced."""
    parts = {
        "phi_representation": gradient_table.phi_representation,
        "phi_breakpoints": gradient_table.phi_breakpoints,
        "phi_coefficients": gradient_table.phi_coefficients,
        "phi_degrees": gradient_table.phi_degrees,
        "hard_kinks": gradient_table.hard_kinks,
        "phi_constant": None,
        "phi_expansion": gradient_table.phi_expansion,
    }
    return parts | overrides


def test_the_group_refuses_an_expansion_kind_without_an_expansion(gradient_table):
    with pytest.raises(TranslationError, match="phi_expansion must be present"):
        LinearOperator(**spoiled_expansion(gradient_table, phi_expansion=None))


def test_the_group_refuses_a_table_kind_without_a_table(gradient_table):
    """A kind that is neither is named by the check that finds it missing a member."""
    with pytest.raises(TranslationError, match="phi_coefficients must be present"):
        LinearOperator(
            **spoiled_expansion(
                gradient_table, phi_representation="table", phi_expansion=None
            )
        )


@pytest.mark.parametrize("field", ["phi_breakpoints", "phi_degrees", "hard_kinks"])
def test_the_group_refuses_a_z_structure_that_is_not_the_expansions(
    gradient_table, field
):
    """The z structure of Phi is its scalar tables' and nothing else."""
    stale = {
        "phi_breakpoints": np.array([0.0, 0.5, 0.21 * 1.0]),
        "phi_degrees": np.array([1, 1], dtype=np.intp),
        "hard_kinks": np.array([0.05]),
    }[field]
    with pytest.raises(TranslationError, match=f"{field} must be the expansion"):
        LinearOperator(**spoiled_expansion(gradient_table, **{field: stale}))


def test_the_unitary_phase_of_a_z_table_is_the_imaginary_part_of_its_clenshaw_sum(
    field_grid, gradient_mode
):
    table = build_phi(
        gradient_mode,
        field_grid,
        Frame.comoving(field_grid),
        z_max=0.2,
        representation="table",
    )
    z = 0.137
    assert np.array_equal(unitary_phase(table, z), np.imag(table.polynomial(z)))


def test_an_expansion_whose_scalar_tables_exhaust_their_budget_relaxes_and_warns(
    field_grid, gradient_mode, caplog
):
    """The scalar tables get the same decade relaxation the z table has always had."""
    with caplog.at_level(logging.WARNING, logger="solprop.translate.linop"):
        table = build_phi(
            gradient_mode,
            field_grid,
            Frame.comoving(field_grid),
            z_max=0.2,
            max_intervals=1,
        )
    assert "scalar tables of the expansion exhausted their budget" in caplog.text
    assert table.phi_representation == "expansion"


class _StubMode:
    """A mode declaring whatever parameters a test wants, and nothing that evaluates.

    Only :func:`parameter_paths` sees it: the two branches below are about the *shape*
    of a mode set's declaration, and no real mode in the package can make them, because
    `MarcatiliMode` is the only implementation and it always declares the same two
    paths.
    """

    z_constant = False
    scale_invariant = True
    azimuthal_order = 0
    channel_label = ChannelLabel("stub", None, None, None, None, "stub")

    def principal_component(self, z=0.0):
        return None

    def __init__(self, paths):
        self._paths = paths

    @property
    def parameter_paths(self):
        return self._paths

    def at_parameters(self, values):
        del values
        return self

    def neff(self, omega, z=0.0):
        return np.ones_like(omega, dtype=complex)

    def neff_minus_one(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def dneff_domega(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def aeff(self, z=0.0):
        return 1.0

    def dimlimits(self, z=0.0):
        return "polar", (0.0, 0.0), (1.0, 2 * np.pi)

    def field(self, points, z=0.0):
        return np.zeros((len(points), 2))


def test_modes_that_declare_different_parameters_have_no_shared_space(caplog):
    modes = [
        _StubMode({"density": Linear(1.0, 2.0, 1.0)}),
        _StubMode({"density": Linear(1.0, 2.0, 1.0), "core_radius": Constant(1e-4)}),
    ]
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.linop"):
        assert parameter_paths(modes, 0.0, 1.0) is None
    assert "declare different parameters" in caplog.text


def test_a_mode_that_declares_no_parameters_at_all_has_no_shared_space(caplog):
    # Every mode in the package declares its paths, so this branch needs a mode
    # that is a Mode without being a ParameterisedMode: the stub without its two
    # members.
    members = {
        name: value
        for name, value in vars(_StubMode).items()
        if name in ("__init__", "neff", "neff_minus_one", "dneff_domega", "aeff")
        or name in ("dimlimits", "field", "principal_component")
        or name in ("z_constant", "scale_invariant", "azimuthal_order", "channel_label")
    }
    bare = cast(
        "Mode",
        type("_Unparameterised", (), members)({"density": Linear(1.0, 2.0, 1.0)}),
    )
    assert not isinstance(bare, ParameterisedMode)
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.linop"):
        assert parameter_paths([bare], 0.0, 1.0) is None
    assert "do not declare their parameters" in caplog.text


def test_a_declaration_of_nothing_but_constants_has_no_dimension(caplog):
    modes = [_StubMode({"density": Constant(2.0)})]
    with caplog.at_level(logging.DEBUG, logger="solprop.translate.linop"):
        assert parameter_paths(modes, 0.0, 1.0) is None
    assert "is constant, yet the mode set is not" in caplog.text


@pytest.mark.parametrize("z", [0.0, 0.07, 0.2])
def test_the_parameter_sampler_reproduces_the_z_sampler(field_grid, z):
    """The expansion resolves the same operator the z table does, not a near one.

    The two samplers evaluate the same modes, frozen or at a position, so each
    propagation constant agrees to the one-evaluation 16 eps of ``accuracy.md``
    section 7; the operator subtracts the frame's ``beta_0 + beta_1 (omega -
    omega_0)``, three terms of that size, so each element is held absolutely to
    ``64 eps k_max`` with ``k_max = max omega / c`` (``n_eff < 1`` in a capillary).

    Including the *one frame for the set*: the frame quantities are frozen on the
    reference mode at these parameter values, exactly as :func:`_shared_frame` freezes
    them on it at this position, so a set expanded in its parameters stays as
    simultaneous as one tabulated along z.
    """
    fill = PressureGradient.two_point(0.2, 5.0, 0.5)
    modes = [MarcatiliMode(125e-6, "He", fill, m=m, cladding=1.45) for m in (1, 2, 3)]
    frame = Frame.comoving(field_grid)
    at_position = _operator_sampler(
        modes, field_grid, frame, 1, DEFAULT_LOSS_POLICY, None
    )
    at_parameters = _operator_of_parameters(
        modes, field_grid, frame, 1, DEFAULT_LOSS_POLICY, None
    )
    varying = parameter_paths(modes, 0.0, 0.21)
    assert varying is not None
    values = {name: float(path(np.array([z]))[0]) for name, path in varying.items()}
    k_max = float(np.max(np.abs(field_grid.omega))) / C
    np.testing.assert_allclose(
        at_parameters(values), at_position(z), rtol=0, atol=64 * EPS * k_max
    )


def test_a_z_table_has_no_fixed_terms_to_weight(gradient_table, gradient_z_table):
    # A resident Adams engine forms the operator's rate from the weights of its fixed
    # terms; an expansion has them and a z table, sampled along z, does not.
    assert rate_weights(gradient_table, 0.1).shape == (
        gradient_table.phi_expansion.terms.shape[0],
    )
    with pytest.raises(TranslationError, match="fixed terms to weight"):
        rate_weights(gradient_z_table, 0.1)
