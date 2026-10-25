# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import logging

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID
from solprop import fourier
from solprop.constants import C
from solprop.exceptions import GridError, ModeError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import (
    MAX_ATTENUATION_EXPONENT,
    Frame,
    LossPolicy,
    dispersion_operator,
    frame_quantities,
)
from solprop.modes import ChannelLabel
from solprop.nonlinear.gnlse import SpecifiedMode, attenuation_from_db

CENTRE = 1030e-9
OMEGA_REF = 2.0 * np.pi * C / CENTRE
PHASE_INDEX = 1.45
GROUP_INDEX = 1.47
GVD = 2.0e-26  # s^2/m
BETAS = [PHASE_INDEX * OMEGA_REF / C, GROUP_INDEX / C, GVD, -1.0e-41]

GRID = {
    "length": 0.1,
    "reference_wavelength": CENTRE,
    "wavelength_limits": (700e-9, 2.0e-6),
    "trange": 2e-12,
    # The grid the operator regression was recorded on.
    **LEGACY_GRID,
}


def make_mode(*, loss=0.0):
    return SpecifiedMode.from_taylor(BETAS, OMEGA_REF, loss=loss, aeff=8e-11)


@pytest.fixture
def mode():
    return make_mode()


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(**GRID)


@pytest.fixture
def envelope_grid():
    return EnvelopeGrid.from_limits(**GRID)


class _Broken:
    """A mode whose effective index is not finite, to exercise the guard."""

    z_constant = True
    scale_invariant = True
    azimuthal_order = 0
    channel_label = ChannelLabel("broken", None, None, None, None, "broken")

    def principal_component(self, z=0.0):
        return None

    def neff(self, omega, z=0.0):
        return np.full(omega.shape, np.nan, dtype=complex)

    def neff_minus_one(self, omega, z=0.0):
        return self.neff(omega, z) - 1.0

    def dneff_domega(self, omega, z=0.0):
        return np.zeros_like(omega, dtype=complex)

    def aeff(self, z=0.0):
        return 1.0

    def dimlimits(self, z=0.0):
        return "polar", (0.0, 0.0), (1.0, 2 * np.pi)

    def field(self, points, z=0.0):
        return np.zeros((len(points), 2))


# ------------------------------------------------------------------- loss policy


def test_the_default_loss_limit_is_the_underflow_of_the_whole_fibre():
    # exp(-alpha L / 2) has underflowed at the limit, so clamping there cannot change
    # a result. The number is derived, not chosen.
    assert LossPolicy().limit(1.5) == pytest.approx(
        2.0 * MAX_ATTENUATION_EXPONENT / 1.5, rel=1e-15
    )
    assert np.exp(-LossPolicy().limit(1.5) * 1.5 / 2.0) < 1e-307


def test_an_explicit_loss_limit_is_used_as_given():
    assert LossPolicy(max_alpha=3000.0).limit(0.01) == 3000.0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, np.nan])
def test_a_bad_loss_limit_is_rejected(bad):
    with pytest.raises(GridError, match="max_alpha must be positive and finite"):
        LossPolicy(max_alpha=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf])
def test_a_bad_length_is_rejected(bad):
    with pytest.raises(GridError, match="length must be positive and finite"):
        LossPolicy().limit(bad)


# ------------------------------------------------------------------------ frames


def test_the_comoving_frame_follows_the_grid_and_the_third_harmonic(
    field_grid, envelope_grid
):
    assert Frame.comoving(field_grid).subtract_beta0 is False
    assert Frame.comoving(field_grid, thg=True).subtract_beta0 is False
    assert Frame.comoving(envelope_grid).subtract_beta0 is True
    assert Frame.comoving(envelope_grid, thg=True).subtract_beta0 is False
    assert Frame.comoving(envelope_grid).omega0 == envelope_grid.omega0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_a_bad_frame_reference_is_rejected(bad):
    with pytest.raises(GridError, match="omega0 must be positive and finite"):
        Frame(bad, subtract_beta0=True)


def test_the_frame_quantities_are_the_analytic_phase_and_group_delays(mode):
    beta0, beta1 = frame_quantities(mode, Frame(OMEGA_REF, subtract_beta0=True))
    # beta_0 = n omega/c and beta_1 = n_g/c, both known in closed form from BETAS.
    assert beta0 == pytest.approx(PHASE_INDEX * OMEGA_REF / C, rel=1e-14)
    assert beta1 == pytest.approx(GROUP_INDEX / C, rel=1e-14)


def test_the_frame_reports_no_constant_when_it_does_not_subtract_one(mode):
    beta0, beta1 = frame_quantities(mode, Frame(OMEGA_REF, subtract_beta0=False))
    assert beta0 == 0.0
    assert beta1 == pytest.approx(GROUP_INDEX / C, rel=1e-14)


def test_fixed_frame_quantities_override_the_mode(mode):
    frame = Frame(OMEGA_REF, subtract_beta0=True, beta0=1.0, beta1=2.0)
    assert frame_quantities(mode, frame) == (1.0, 2.0)


def test_only_the_missing_frame_quantity_is_taken_from_the_mode(mode):
    beta0, beta1 = frame_quantities(
        mode, Frame(OMEGA_REF, subtract_beta0=True, beta1=2.0)
    )
    assert beta0 == pytest.approx(PHASE_INDEX * OMEGA_REF / C, rel=1e-14)
    assert beta1 == 2.0

    beta0, beta1 = frame_quantities(
        mode, Frame(OMEGA_REF, subtract_beta0=True, beta0=1.0)
    )
    assert beta0 == 1.0
    assert beta1 == pytest.approx(GROUP_INDEX / C, rel=1e-14)


def test_the_analytic_group_delay_matches_finite_differences(mode):
    # ADR-006's gate one level up from `dneff_domega`: beta_1 itself against a central
    # difference of beta, at three steps, checking the second-order scaling.
    _, analytic = frame_quantities(mode, Frame(OMEGA_REF, subtract_beta0=False))
    errors = []
    for relative in (9e-3, 3e-3, 1e-3):
        h = relative * OMEGA_REF
        differenced = (
            mode.beta(np.array([OMEGA_REF + h]))[0]
            - mode.beta(np.array([OMEGA_REF - h]))[0]
        ) / (2.0 * h)
        errors.append(abs(differenced / analytic - 1.0))
    # Second order, so a threefold smaller step cuts the error by nine. The residual
    # at the smallest step is that truncation error, not an error in beta_1.
    assert errors[0] / errors[1] == pytest.approx(9.0, rel=0.05)
    assert errors[1] / errors[2] == pytest.approx(9.0, rel=0.05)
    assert errors[2] < 1e-8


# ------------------------------------------------------------------- the operator


def round_off_bound(mode, grid):
    """Absolute bound on the round-off of the operator's own cancellation, in 1/m.

    D subtracts the frame from beta, and the two are of the same size, so what
    survives is a small difference of large numbers. The bound is Higham's n*eps with
    n taken as the handful of multiplies and adds in the expression, applied to the
    largest beta on the band. A relative comparison would be meaningless where D
    crosses zero inside the band (docs/dev/testing.md section 3). Measured worst
    deviation for the three cases below: 3.7e-9 against a bound of 1.8e-8.
    """
    beta = mode.beta(grid.omega[grid.active_mask])
    return 8.0 * np.finfo(float).eps * float(np.max(np.abs(beta)))


def deviation(actual, expected):
    """Largest absolute difference between two operators, in 1/m."""
    return float(np.max(np.abs(actual - expected)))


def expected_operator(mode, grid, *, subtract_beta0, origin):
    """Luna's own algebra, written out for comparison (LinearOps.jl:399-459)."""
    active = grid.active_mask
    omega = grid.omega[active]
    beta1 = GROUP_INDEX / C
    constant = mode.beta(np.array([OMEGA_REF]))[0] if subtract_beta0 else 0.0
    out = np.zeros(grid.n_omega, dtype=complex)
    out[active] = -1j * (
        mode.beta(omega) - constant - beta1 * (omega - origin)
    ) - 0.5 * mode.alpha(omega)
    return out


def test_the_field_grid_operator_removes_the_delay_and_not_the_carrier_phase(
    mode, field_grid
):
    operator = dispersion_operator(mode, field_grid, Frame.comoving(field_grid))
    expected = expected_operator(mode, field_grid, subtract_beta0=False, origin=0.0)
    assert deviation(operator, expected) < round_off_bound(mode, field_grid)


def test_the_envelope_grid_operator_removes_the_carrier_phase_too(mode, envelope_grid):
    operator = dispersion_operator(mode, envelope_grid, Frame.comoving(envelope_grid))
    expected = expected_operator(
        mode, envelope_grid, subtract_beta0=True, origin=OMEGA_REF
    )
    assert deviation(operator, expected) < round_off_bound(mode, envelope_grid)


def test_the_envelope_grid_keeps_the_carrier_phase_when_a_harmonic_is_present(
    mode, envelope_grid
):
    # Keeping the carrier phase whole means the delay is measured from zero frequency
    # too: from the carrier it would still remove the constant -beta_1 omega_ref, which
    # the harmonic's exp(2i omega0 t) does not share.
    frame = Frame.comoving(envelope_grid, thg=True)
    operator = dispersion_operator(mode, envelope_grid, frame)
    expected = expected_operator(mode, envelope_grid, subtract_beta0=False, origin=0.0)
    assert deviation(operator, expected) < round_off_bound(mode, envelope_grid)


def test_the_two_envelope_frames_differ_by_a_constant(mode, envelope_grid):
    active = envelope_grid.active_mask
    without = dispersion_operator(
        mode, envelope_grid, Frame.comoving(envelope_grid, thg=True)
    )[active]
    with_beta0 = dispersion_operator(
        mode, envelope_grid, Frame.comoving(envelope_grid)
    )[active]
    difference = with_beta0 - without
    assert deviation(difference, difference[0]) < round_off_bound(mode, envelope_grid)
    beta0 = mode.beta(np.array([OMEGA_REF]))[0]
    assert difference[0] == pytest.approx(
        1j * (beta0 - GROUP_INDEX / C * OMEGA_REF), rel=1e-12
    )


def test_the_operator_logs_what_it_built(mode, envelope_grid, caplog):
    # The debug line is the only place a run records how many bins were clamped, so
    # it has to survive a refactor of the operator.
    with caplog.at_level(logging.DEBUG, logger="solprop.linear"):
        dispersion_operator(
            make_mode(loss=1e6), envelope_grid, Frame.comoving(envelope_grid)
        )
    assert "bins active" in caplog.text
    assert "clamped" in caplog.text


def test_the_operator_is_zero_outside_the_active_band(mode, field_grid):
    operator = dispersion_operator(mode, field_grid, Frame.comoving(field_grid))
    assert np.array_equal(
        operator[~field_grid.active_mask],
        np.zeros(int(np.sum(~field_grid.active_mask)), dtype=complex),
    )
    assert np.any(operator[field_grid.active_mask] != 0.0)


def test_the_operator_regression(mode, field_grid, envelope_grid, ndarrays_regression):
    """Pin D on both grids, as a guard against silent drift.

    `tests/test_luna_linear.py` checks the same operator against Luna's own; this one
    is independent of that witness, so a change here is visible even if the witness is
    regenerated.

    `ndarrays_regression` rather than `num_regression`: the two are the same package
    and both compare with a tolerance, but `num_regression` goes through pandas, which
    is not an approved dependency (docs/dev/dependencies.md).
    """
    data = {}
    for name, grid in (("field", field_grid), ("envelope", envelope_grid)):
        operator = dispersion_operator(mode, grid, Frame.comoving(grid))
        data[f"{name}_real"] = np.real(operator)
        data[f"{name}_imag"] = np.imag(operator)
    # The operator is a cancellation of beta ~ 1e7 1/m down to ~1e2, so the last five
    # digits are round-off; anything above the sixth is a real change to the assembly.
    ndarrays_regression.check(data, default_tolerance={"rtol": 1e-9, "atol": 1e-8})


def test_the_frame_reference_must_lie_on_the_grid(mode, field_grid):
    with pytest.raises(GridError, match="lies outside the grid"):
        dispersion_operator(mode, field_grid, Frame(1e18, subtract_beta0=False))


def test_a_non_finite_effective_index_raises(field_grid):
    frame = Frame(OMEGA_REF, subtract_beta0=False, beta0=0.0, beta1=0.0)
    with pytest.raises(ModeError, match="non-finite effective index"):
        dispersion_operator(_Broken(), field_grid, frame)


# --------------------------------------------------------------------- the loss


def test_a_loss_below_the_limit_passes_through_untouched(envelope_grid):
    alpha = attenuation_from_db(20.0)
    operator = dispersion_operator(
        make_mode(loss=alpha), envelope_grid, Frame.comoving(envelope_grid)
    )
    active = envelope_grid.active_mask
    assert np.allclose(-2.0 * np.real(operator[active]), alpha, rtol=1e-15, atol=0.0)


def test_a_loss_above_the_limit_is_clamped_on_the_operator(envelope_grid):
    limit = LossPolicy().limit(envelope_grid.length)
    operator = dispersion_operator(
        make_mode(loss=10.0 * limit), envelope_grid, Frame.comoving(envelope_grid)
    )
    active = envelope_grid.active_mask
    # The clamp lands on D itself, so the exponent that the cumulative operator will
    # integrate is already limited; clamping the accumulated exponent instead would
    # depend on the step (docs/guide/numerics/linear-operator.md).
    assert np.allclose(-2.0 * np.real(operator[active]), limit, rtol=1e-15, atol=0.0)


def test_gain_is_represented_rather_than_silently_removed(envelope_grid):
    # Clamping alpha to [0, 3000] would make a negative loss lossless without a word.
    # solprop only limits from above.
    operator = dispersion_operator(
        make_mode(loss=-2.0), envelope_grid, Frame.comoving(envelope_grid)
    )
    active = envelope_grid.active_mask
    assert np.allclose(-2.0 * np.real(operator[active]), -2.0, rtol=1e-15, atol=0.0)


# ------------------------------------------------------------ the spectral absorber


def test_the_absorber_rate_is_carried_by_every_bin(mode, field_grid):
    rate = np.linspace(1.0, 5.0, field_grid.n_omega)
    bare = dispersion_operator(mode, field_grid, Frame.comoving(field_grid))
    absorbed = dispersion_operator(
        mode, field_grid, Frame.comoving(field_grid), absorber_rate=rate
    )
    # Out of band the operator is the absorber alone: with the collar as a rate those
    # bins must decay rather than propagate loss-free.
    assert np.allclose(absorbed - bare, -0.5 * rate, rtol=1e-14, atol=0.0)
    outside = ~field_grid.active_mask
    assert np.allclose(absorbed[outside], -0.5 * rate[outside], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("rate", "message"),
    [
        (np.full(8, np.nan), "finite everywhere"),
        (np.linspace(-1.0, 1.0, 8), "non-negative"),
    ],
)
def test_a_bad_absorber_rate_is_rejected(mode, field_grid, rate, message):
    full = np.resize(rate, field_grid.n_omega)
    with pytest.raises(GridError, match=message):
        dispersion_operator(
            mode, field_grid, Frame.comoving(field_grid), absorber_rate=full
        )


def test_an_absorber_rate_of_the_wrong_length_is_rejected(mode, field_grid):
    with pytest.raises(GridError, match="absorber_rate has shape"):
        dispersion_operator(
            mode,
            field_grid,
            Frame.comoving(field_grid),
            absorber_rate=np.ones(field_grid.n_omega - 1),
        )


# ------------------------------------------------------------------ the frame works


def centroid(power, t):
    """First moment of a non-negative distribution over t."""
    return float(np.sum(power * t) / np.sum(power))


def propagated_centroid(grid, frame, mode, distance):
    """Propagate a transform-limited pulse linearly and return its centre of mass."""
    t = grid.t
    duration = 150e-15
    sigma = duration / (2.0 * np.sqrt(np.log(2.0)))
    envelope = np.exp(-((t / sigma) ** 2))
    samples = (
        envelope.astype(complex)
        if grid.kind == "envelope"
        else np.real(envelope * np.exp(1j * grid.omega0 * t))
    )
    operator = dispersion_operator(mode, grid, frame)
    spectrum = fourier.forward(samples, kind=grid.kind) * np.exp(operator * distance)
    out = fourier.inverse(spectrum, kind=grid.kind)
    power = (
        np.abs(out if grid.kind == "envelope" else fourier.analytic_signal(out)) ** 2
    )
    return centroid(power, t)


@pytest.mark.parametrize("grid_name", ["field_grid", "envelope_grid"])
def test_the_comoving_frame_holds_the_pulse_in_the_window(grid_name, mode, request):
    grid = request.getfixturevalue(grid_name)
    moved = propagated_centroid(grid, Frame.comoving(grid), mode, grid.length)
    # A 10 cm fibre at a group index of 1.47 delays the pulse by 490 ps, 245 window
    # widths. In the co-moving frame it must not move by even one sample.
    assert abs(moved) < grid.dt


@pytest.mark.parametrize("grid_name", ["field_grid", "envelope_grid"])
def test_a_mistuned_frame_delays_the_pulse_by_the_difference(grid_name, mode, request):
    grid = request.getfixturevalue(grid_name)
    mismatch = 2.0e-12  # s/m, chosen so the residual delay is a fifth of the window
    frame = Frame(
        grid.omega0,
        subtract_beta0=grid.kind == "envelope",
        beta1=GROUP_INDEX / C - mismatch,
    )
    moved = propagated_centroid(grid, frame, mode, grid.length)
    # D carries -i beta_1 omega, and a delay by tau is exp(-i omega tau) in the
    # internal convention, so the residual group delay is exactly mismatch * length.
    assert moved == pytest.approx(mismatch * grid.length, abs=grid.dt)
