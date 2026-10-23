# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses
import math
import warnings

import numpy as np
import pytest

from bounds import scan_bound
from convergence_records import Column, Study, record
from plasma_cases import (
    AMPLITUDE,
    CARRIER,
    CHARGE_MASS_RATIO,
    DENSITY,
    DURATION,
    constant_field_reference,
    constant_rate,
    gaussian_pulse_and_derivative,
    power_rate,
    pulse,
    zero_rate,
)
from rate_oracles import HISTORICAL_ATOMS, adk_oracle
from solprop.constants import AU_ENERGY, E_CHARGE, EPS0, M_E
from solprop.exceptions import GridError, MaterialError
from solprop.grid import FieldGrid
from solprop.nonlinear.ionisation import EvaluableRate, ionisation_fraction
from solprop.nonlinear.responses import (
    PlasmaResponse,
    ResponseTraits,
    _check_plasma,
)
from solprop.physdata import Gas, IonisationData, ionisation_potential, species
from solprop.translate.ionisation import build_rate_table, clear_rate_table_cache


def trapezoid_plasma(*args, **kwargs):
    """Return the plasma response on the plain cumulative trapezoid.

    This module holds the trapezoid's own identities -- its second-order ladders, its
    exact multiplier, its strict causality -- and the response's properties evaluated
    through it; ``test_plasma_band.py`` holds the band rule's, which is the default.
    """
    return PlasmaResponse(*args, integration="trapezoid", **kwargs)


# A cumulative trapezoid accumulates one product per sample where a closed form
# multiplies once, so the two agree to Higham's n * eps rather than to machine epsilon:
# about 1e-12 over the few thousand samples used here.
FRACTION_RTOL = 1e-12

# hypot reaches the magnitude of a linearly polarised field by a different route from
# taking its absolute value, and an eighth-power rate amplifies that by eight, after
# which three cumulative sums over 8193 samples carry it. Measured worst peak-normalised
# deviation between the scalar and the rotated vector answer: 1.2e-13, at pi/4, and
# exactly zero at 0 and pi/2 where the two routes coincide.
PARITY_RTOL = 5e-13


@pytest.fixture
def response():
    return trapezoid_plasma(Gas.Ar, power_rate())


# ------------------------------------------------------------------------- traits


@pytest.mark.parametrize("preionisation", [0.0, 0.4])
def test_traits_are_declared_as_the_design_requires(preionisation):
    traits = trapezoid_plasma(Gas.Ar, power_rate(), preionisation=preionisation).traits
    # The pre-ionisation path is declared even when the value is zero: the translation
    # layer supplies it per plane, and a response that lies about its traits compiles
    # into a wrong kernel rather than an error.
    assert traits == ResponseTraits(
        domain="memory",
        grids=("field",),
        polarisation="vector_isotropic",
        z_profiles=("density", "preionisation"),
    )


# ------------------------------------------------------------- the analytic oracle

# The constant-field case is exact in closed form, so the only error is the trapezoid's
# own. Measured max relative deviation over the whole axis: 9.4e-8 at 1025 samples,
# falling by four at every doubling. The bound below is that value with a factor of two
# in hand; the convergence ratio is what actually gates the discretisation.
CONSTANT_FIELD_RTOL = 2e-7

# The two currents must stay within an order of magnitude of each other, or one of them
# hides an error in the other. It is W0*T that decides, not the field: at W0*T = 1 the
# ratio is 4.3, and by W0*T = 10 the free term has run away by a factor of 200.
BALANCE = (0.1, 10.0)


@pytest.mark.parametrize("preionisation", [0.0, 0.3])
def test_a_constant_field_reproduces_the_closed_form(preionisation):
    # Every integral is elementary here, so this is an oracle and not a second copy of
    # the implementation: it validates the whole chain, both prefactors included.
    amplitude, rate, duration = 1e9, 1e14, 1e-14
    potential = ionisation_potential(Gas.Ar)
    plasma = trapezoid_plasma(Gas.Ar, constant_rate(rate), preionisation=preionisation)

    t = np.linspace(0.0, duration, 2**12 + 1)
    field = np.full(t.size, amplitude)
    got = plasma.evaluate(field[None, :], 1.0, dt=float(t[1] - t[0]), kind="field")[0]
    fraction, expected = constant_field_reference(
        t, amplitude, rate, potential, preionisation
    )

    worst = np.max(np.abs(got - expected)) / np.max(np.abs(expected))
    assert 0.0 < worst < CONSTANT_FIELD_RTOL
    # The trapezoid accumulates W0 * dt 4097 times where the closed form multiplies
    # once, so the two differ by Higham's n * eps, about 1e-12.
    assert np.allclose(
        plasma.ionised_fraction(field[None, :], dt=float(t[1] - t[0]), kind="field"),
        fraction,
        rtol=FRACTION_RTOL,
        atol=0.0,
    )


def test_the_two_currents_are_balanced_in_the_closed_form_case():
    # Guards the oracle above against degenerating into a test of the free term alone.
    amplitude, rate, duration = 1e9, 1e14, 1e-14
    t = np.linspace(0.0, duration, 2**12 + 1)
    free = (
        CHARGE_MASS_RATIO
        * amplitude
        * (t**2 / 2.0 - (t - (1.0 - np.exp(-rate * t)) / rate) / rate)
    )
    loss = ionisation_potential(Gas.Ar) * (1.0 - np.exp(-rate * t)) / amplitude
    assert BALANCE[0] < loss[-1] / free[-1] < BALANCE[1]


# The record of the two ladders below, the "plasma column's cumulative trapezoid" row
# of `docs/dev/accuracy.md` section 1. Both are cumulative trapezoid sums of a few
# thousand samples whose finest rung is 6.2e-9 and 6.2e-6, seven and ten orders above
# the round-off floor of those sums, so the tolerance is tight; `atol` is set at that
# floor (about 8193 * eps) so it never becomes the thing being tested.
PLASMA_TRAPEZOID = Study(
    name="plasma_trapezoid",
    title="Second-order convergence of the plasma column in the sample spacing",
    case={
        "closed form": "argon, a constant 1e9 V/m field and a constant 1e14 1/s rate "
        "over 10 fs, against the closed-form polarisation",
        "energy identity": "the work the field does on the medium against Ip times "
        "the electrons, on a Gaussian pulse, He minus Ar sharing one rate",
    },
    axis={"name": "n_samples", "unit": "samples", "label": "samples in the window"},
    columns=(
        Column(
            name="deviation",
            unit="1 (relative)",
            why="seven orders above the round-off floor of a few thousand trapezoid "
            "terms; atol is that floor, about 8193 eps, so it is never what is tested",
            rtol=1e-9,
            atol=2e-12,
        ),
        Column(
            name="ratio",
            unit="1",
            why="the deviation's fall per doubling, a ratio of two numbers each known "
            "to 1e-9; null on the coarsest rung, which has no pair",
            rtol=1e-8,
        ),
    ),
)


def _trapezoid_rows(series, counts, deviations):
    """Return the record rows of one second-order ladder."""
    return [
        {
            "series": series,
            "axis": count,
            "deviation": deviation,
            "ratio": None if index == 0 else deviations[index - 1] / deviation,
        }
        for index, (count, deviation) in enumerate(zip(counts, deviations, strict=True))
    ]


def test_the_closed_form_is_reached_at_second_order_in_the_step(request):
    amplitude, rate, duration = 1e9, 1e14, 1e-14
    # The stored record was taken on argon's potential of solprop's former registry,
    # 0.5792 hartree (at dc4ca0d; optmat's first ionisation energy is 0.579155, 8e-5
    # lower), which moves the deviation by parts in 1e5 and the ratio past its 1e-8.
    # The ladder is about the trapezoid, not the potential, so that input is supplied
    # explicitly and the record kept.
    potential = HISTORICAL_ATOMS["Ar"][0] * AU_ENERGY
    argon = dataclasses.replace(
        species("Ar"),
        ionisation=IonisationData(
            potential, citation="solprop's former registry, the record's input"
        ),
    )
    plasma = trapezoid_plasma(argon, constant_rate(rate))

    def deviation(exponent):
        # 2^k + 1 samples, so every level shares the endpoint and the same interval.
        t = np.linspace(0.0, duration, 2**exponent + 1)
        field = np.full(t.size, amplitude)
        got = plasma.evaluate(field[None, :], 1.0, dt=float(t[1] - t[0]), kind="field")[
            0
        ]
        _, expected = constant_field_reference(t, amplitude, rate, potential, 0.0)
        return float(np.max(np.abs(got - expected)) / np.max(np.abs(expected)))

    coarse, middle, fine = deviation(10), deviation(11), deviation(12)
    # The trapezoid rule is second order, so each doubling divides the error by four;
    # measured 4.000 at both levels, from 9.450e-8, 2.363e-8 and 5.906e-9. The window
    # allows for the fourth-order term.
    assert 3.6 < coarse / middle < 4.4
    assert 3.6 < middle / fine < 4.4
    record(
        PLASMA_TRAPEZOID,
        request=request,
        rows=_trapezoid_rows("closed form", (1025, 2049, 4097), [coarse, middle, fine]),
    )


# --------------------------------------------------------------- the Drude limit

# The oscillating part of the polarisation of a cold plasma has susceptibility
# -omega_p^2 / Omega^2. This is the one test that pins e^2 / (eps0 m_e) rather than
# e^2 / m_e, and it needs no ionisation model at all. Measured deviation 3.7e-8 with
# 200001 samples, which is the trapezoid error on the double integral: theta^2 / 6 at
# theta = Omega dt = 4.7e-4, the small-angle limit of the multiplier below.
DRUDE_RTOL = 1e-7

# Two cumulative trapezoids applied to a sampled cosine reproduce the closed form
# exactly, with the multiplier (theta/2) cot(theta/2) per integral in place of one:
# the recurrence's own solution, so only round-off separates them. The second sum
# integrates the first one's rounding drift, so the deviation grows as n^2 eps
# tan(theta/2): measured worst peak-normalised 8.1e-14 at theta = 4.7e-4 over 200001
# samples, and 2.9e-12 and 3.4e-12 at the band top over 512.
MULTIPLIER_RTOL = 1e-12
BAND_TOP_RTOL = 1e-11


def trapezoid_multiplier(theta):
    """Return the cumulative trapezoid's multiplier relative to the exact integral."""
    return (theta / 2) / math.tan(theta / 2)


def test_a_pre_ionised_plasma_has_the_drude_susceptibility():
    fraction, amplitude = 0.3, 1e9
    t = np.linspace(0.0, 40e-15, 200001)
    dt = float(t[1] - t[0])
    # Cosine, so the double integral has no secular ramp, only a constant offset.
    field = amplitude * np.cos(CARRIER * t)
    plasma = trapezoid_plasma(Gas.Ar, zero_rate, preionisation=fraction)
    got = plasma.evaluate(field[None, :], DENSITY, dt=dt, kind="field")[0]

    offset = DENSITY * fraction * CHARGE_MASS_RATIO * amplitude / CARRIER**2
    expected = offset * (1.0 - np.cos(CARRIER * t))
    worst = np.max(np.abs(got - expected)) / np.max(np.abs(expected))
    assert 0.0 < worst < DRUDE_RTOL

    # And the oscillating part is -omega_p^2/Omega^2 times eps0 E, with the sign that
    # lowers the refractive index, times the two integrals' multiplier: exactly, so the
    # whole 3.7e-8 above is the multiplier and none of it is the coefficient.
    plasma_frequency_squared = DENSITY * fraction * E_CHARGE**2 / (EPS0 * M_E)
    assert offset == pytest.approx(
        plasma_frequency_squared * EPS0 * amplitude / CARRIER**2, rel=1e-14
    )
    multiplier = trapezoid_multiplier(CARRIER * dt) ** 2
    exact = multiplier * expected
    assert np.max(np.abs(got - exact)) < MULTIPLIER_RTOL * np.max(np.abs(exact))
    assert (got - multiplier * offset)[0] * field[0] < 0.0


@pytest.mark.parametrize(("top", "squared"), [(1.0, 0.8225), (1.1, 0.7866)])
def test_the_drude_polarisation_is_too_small_at_the_band_top(top, squared):
    # On the default field grid, dt_fine = 2 pi / (6 omega_max), so theta = pi/3 at
    # omega_max. Two cumulative trapezoids then give a pre-ionised plasma 0.82 of its
    # Drude susceptibility at omega_max and 0.79 at the top of the collar (accuracy.md
    # section 1). This test pins the present discretisation, not the physics: a band-
    # exact conversion of the currents would change both numbers to one.
    grid = FieldGrid.from_limits(1.0, 800e-9, (90e-9, 4e-6), 400e-15)
    dt = grid.dt_fine
    assert grid.omega_max * dt == pytest.approx(math.pi / 3, rel=1e-12)
    omega = top * grid.omega_max
    fraction, amplitude = 0.3, 1e9
    t = np.arange(512) * dt
    field = amplitude * np.cos(omega * t)
    plasma = trapezoid_plasma(Gas.Ar, zero_rate, preionisation=fraction)
    got = plasma.evaluate(field[None, :], DENSITY, dt=dt, kind="field")[0]

    offset = DENSITY * fraction * CHARGE_MASS_RATIO * amplitude / omega**2
    multiplier = trapezoid_multiplier(omega * dt) ** 2
    exact = multiplier * offset * (1.0 - np.cos(omega * t))
    assert np.max(np.abs(got - exact)) < BAND_TOP_RTOL * np.max(np.abs(exact))
    assert multiplier == pytest.approx(squared, abs=5e-5)


def test_a_fully_pre_ionised_gas_has_no_loss_term():
    # At f = 1 there is nothing left to ionise, so 1 - f is exactly zero and only the
    # free-electron current survives, whatever the rate says.
    _, dt, field = pulse()
    ionising = trapezoid_plasma(Gas.Ar, power_rate(), preionisation=1.0)
    inert = trapezoid_plasma(Gas.Ar, zero_rate, preionisation=1.0)
    fraction = ionising.ionised_fraction(field[None, :], dt=dt, kind="field")
    assert np.array_equal(fraction, np.ones_like(fraction))
    assert np.array_equal(
        ionising.evaluate(field[None, :], DENSITY, dt=dt, kind="field"),
        inert.evaluate(field[None, :], DENSITY, dt=dt, kind="field"),
    )


# ------------------------------------------------------------ energy bookkeeping

# The work the field does is -int E' P dt once the boundary term vanishes, and both
# sides here use the same trapezoid, so the identity converges at second order.
# Measured 9.9e-5 at 8193 samples, falling by four at each doubling.
ENERGY_RTOL = 2e-4


def work_per_atom(plasma, field, derivative, dt):
    """The work the field does on the medium, per atom, by parts."""
    polarisation = plasma.evaluate(field[None, :], DENSITY, dt=dt, kind="field")[0]
    return -float(np.trapezoid(derivative * polarisation, dx=dt)) / DENSITY


def energy_deviation(n_samples):
    """Relative deviation of the ionisation-loss energy from Ip times the electrons."""
    _, dt, field, derivative = gaussian_pulse_and_derivative(n_samples)
    # One rate shared by two gases, so the fraction and hence the free-electron work
    # are identical and cancel in the difference; what is left is the loss term alone,
    # taken from evaluate's own output rather than rebuilt from the same formula.
    rate = power_rate()
    difference = work_per_atom(
        trapezoid_plasma(Gas.He, rate), field, derivative, dt
    ) - work_per_atom(trapezoid_plasma(Gas.Ar, rate), field, derivative, dt)
    fraction = trapezoid_plasma(Gas.Ar, rate).ionised_fraction(
        field[None, :], dt=dt, kind="field"
    )
    electrons = float(fraction[-1] - fraction[0])
    potentials = ionisation_potential(Gas.He) - ionisation_potential(Gas.Ar)
    return abs(difference / (potentials * electrons) - 1.0)


def test_the_field_pays_one_ionisation_potential_for_every_electron():
    assert 0.0 < energy_deviation(2**13 + 1) < ENERGY_RTOL


def test_the_energy_identity_converges_at_second_order(request):
    """Measured 9.919e-5, 2.480e-5 and 6.199e-6 at 8193, 16385 and 32769 samples."""
    coarse, middle, fine = (energy_deviation(2**k + 1) for k in (13, 14, 15))
    assert 3.6 < coarse / middle < 4.4
    assert 3.6 < middle / fine < 4.4
    record(
        PLASMA_TRAPEZOID,
        request=request,
        rows=_trapezoid_rows(
            "energy identity", (8193, 16385, 32769), [coarse, middle, fine]
        ),
    )


# ------------------------------------------------------------------ polarisation


@pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, 0.7])
def test_a_linearly_polarised_field_gives_the_scalar_answer_rotated(angle, response):
    # The rate is taken on the magnitude and the loss current is directed along the
    # field, so a fixed-angle linear polarisation is the scalar problem in a rotated
    # frame. A sign-changing field is used deliberately: with a positive-only one an
    # implementation that took |E| in the free current would still pass.
    _, dt, field = pulse()
    scalar = response.evaluate(field[None, :], DENSITY, dt=dt, kind="field")[0]
    components = np.stack([np.cos(angle) * field, np.sin(angle) * field])
    vector = response.evaluate(components, DENSITY, dt=dt, kind="field")
    peak = np.max(np.abs(scalar))
    # A response that returned zeros would make every deviation vacuous, so the scale
    # is asserted rather than the deviation being non-zero: at zero and at pi/4 the two
    # routes to the magnitude agree exactly and the deviation is legitimately zero.
    assert peak > 0.0

    def deviation(got, expected):
        # Peak-normalised, as testing.md section 3 asks: the response starts at exactly
        # zero and passes through zero at every carrier crossing, where a relative
        # comparison is meaningless.
        return float(np.max(np.abs(got - expected)) / peak)

    assert deviation(vector[0], np.cos(angle) * scalar) < PARITY_RTOL
    assert deviation(vector[1], np.sin(angle) * scalar) < PARITY_RTOL


def test_the_orthogonal_component_of_a_linear_field_is_exactly_zero(response):
    _, dt, field = pulse()
    components = np.stack([field, np.zeros_like(field)])
    vector = response.evaluate(components, DENSITY, dt=dt, kind="field")
    assert np.array_equal(vector[1], np.zeros_like(vector[1]))


def test_a_rotating_field_ionises_by_its_magnitude(response):
    # Constant magnitude, so the fraction must be the one a constant scalar field of
    # the same magnitude gives. This is what pins the rate to hypot rather than to a
    # single component.
    t = np.linspace(0.0, 20e-15, 4097)
    dt = float(t[1] - t[0])
    rotating = AMPLITUDE * np.stack([np.cos(CARRIER * t), np.sin(CARRIER * t)])
    steady = np.full((1, t.size), AMPLITUDE)
    # hypot of a cosine and a sine reaches the amplitude by a different route, so the
    # two agree to a unit in the last place per sample rather than bit for bit.
    assert np.allclose(
        response.ionised_fraction(rotating, dt=dt, kind="field"),
        response.ionised_fraction(steady, dt=dt, kind="field"),
        rtol=FRACTION_RTOL,
        atol=0.0,
    )


def test_the_polarisation_is_odd_in_the_field(response):
    # The rate and the fraction see only the magnitude, and both currents are linear in
    # the signed components, so reversing the field reverses the answer exactly.
    _, dt, field = pulse()
    forward = response.evaluate(field[None, :], DENSITY, dt=dt, kind="field")
    backward = response.evaluate(-field[None, :], DENSITY, dt=dt, kind="field")
    assert np.array_equal(backward, -forward)


def test_the_polarisation_is_linear_in_the_density(response):
    _, dt, field = pulse()
    one = response.evaluate(field[None, :], 1.0, dt=dt, kind="field")
    many = response.evaluate(field[None, :], 7.0, dt=dt, kind="field")
    assert np.allclose(many, 7.0 * one, rtol=1e-15, atol=0.0)


# ------------------------------------------------------------- memory and guards


def test_the_response_has_memory_and_no_foresight(response):
    # The defining property of the `memory` trait: the polarisation at one sample
    # depends on every earlier sample and on none of the later ones. Structural, so
    # exact: a causal scan's prefixes never read a later sample.
    _, dt, field = pulse(n_samples=1025)
    cut = 600
    changed = field.copy()
    changed[cut:] *= -3.0
    before = response.evaluate(field[None, :], DENSITY, dt=dt, kind="field")
    after = response.evaluate(changed[None, :], DENSITY, dt=dt, kind="field")
    assert np.array_equal(after[:, :cut], before[:, :cut])
    assert not np.array_equal(after[:, cut:], before[:, cut:])


def test_padding_the_front_with_zeros_shifts_the_answer_and_nothing_else(response):
    _, dt, field = pulse(n_samples=1025)
    padded = np.concatenate([np.zeros(32), field])
    plain = response.evaluate(field[None, :], DENSITY, dt=dt, kind="field")
    shifted = response.evaluate(padded[None, :], DENSITY, dt=dt, kind="field")
    assert np.array_equal(shifted[:, :32], np.zeros((1, 32)))
    # The leading zeros add exact zeros, but a scan may group the shifted sums
    # differently: the plasma chain's cap of the largest value.
    scale = float(np.max(np.abs(plain)))
    assert scale > 0
    assert np.max(np.abs(shifted[:, 32:] - plain)) <= scan_bound(padded.size) * scale


def test_nothing_happens_before_the_field_is_strong_enough(response):
    # With no pre-ionisation the fraction is zero until the rate first turns on, so
    # both currents are exactly zero there and so is their integral.
    _, dt, field = pulse(n_samples=1025)
    rate = power_rate()(np.abs(field))
    first = int(np.argmax(rate > 0.0))
    assert first > 1
    out = response.evaluate(field[None, :], DENSITY, dt=dt, kind="field")
    # Exactly zero up to but not including the first sample with a rate: the trapezoid
    # reaches forward by one, so the current is already non-zero there and the second
    # integral picks it up at that same index.
    assert np.array_equal(out[:, :first], np.zeros((1, first)))
    assert out[0, first] != 0.0


def test_a_field_that_never_ionises_gives_exactly_nothing():
    _, dt, field = pulse(amplitude=1e8)
    plasma = trapezoid_plasma(Gas.Ar, power_rate())
    # The setup itself, so the test cannot pass for the wrong reason.
    assert float(power_rate()(np.max(np.abs(field)))) == 0.0
    out = plasma.evaluate(field[None, :], DENSITY, dt=dt, kind="field")
    assert np.array_equal(out, np.zeros_like(out))


def test_evaluating_the_response_raises_no_numerical_warnings():
    # The loss term divides by the squared field, and a pulse has exact zeros at every
    # carrier crossing. Divide-by-zero and invalid are errors in this suite.
    t = np.linspace(-3.0 * DURATION, 3.0 * DURATION, 2049)
    dt = float(t[1] - t[0])
    field = AMPLITUDE * np.exp(-((t / DURATION) ** 2)) * np.cos(CARRIER * t)
    field[512] = 0.0
    plasma = trapezoid_plasma(Gas.Ar, constant_rate(1e14))
    with (
        warnings.catch_warnings(),
        np.errstate(divide="raise", over="raise", invalid="raise"),
    ):
        warnings.simplefilter("error")
        out = plasma.evaluate(field[None, :], DENSITY, dt=dt, kind="field")
    assert np.all(np.isfinite(out))


def test_the_ionised_fraction_is_the_one_the_rates_module_defines():
    # One definition, not two: the response takes the magnitude and delegates, so the
    # pre-ionisation inside the exponential lives in exactly one place.
    _, dt, field = pulse()
    components = np.stack([0.6 * field, 0.8 * field])
    plasma = trapezoid_plasma(Gas.Ar, power_rate(), preionisation=0.2)
    assert np.array_equal(
        plasma.ionised_fraction(components, dt=dt, kind="field"),
        ionisation_fraction(
            power_rate(),
            np.hypot(components[0], components[1]),
            dt,
            preionisation=0.2,
        ),
    )


def test_the_fraction_is_monotone_and_starts_at_the_pre_ionised_value():
    _, dt, field = pulse()
    plasma = trapezoid_plasma(Gas.Ar, power_rate(), preionisation=0.2)
    fraction = plasma.ionised_fraction(field[None, :], dt=dt, kind="field")
    assert fraction[0] == 1.0 - (1.0 - 0.2)
    assert np.all(np.diff(fraction) >= 0.0)
    assert np.all(fraction <= 1.0)


def test_a_reversed_pulse_frees_the_same_electrons_but_not_the_same_polarisation():
    # The rate integral is reversal invariant, so the final fraction is; the response
    # is not, which is what having memory means.
    _, dt, field = pulse()
    plasma = trapezoid_plasma(Gas.Ar, power_rate())
    forward = plasma.ionised_fraction(field[None, :], dt=dt, kind="field")
    reversed_fraction = plasma.ionised_fraction(
        field[::-1][None, :], dt=dt, kind="field"
    )
    # A sum of the same terms in the opposite order: Higham's n*eps at 8193 samples.
    assert forward[-1] == pytest.approx(reversed_fraction[-1], rel=1e-12)
    out = plasma.evaluate(field[None, :], DENSITY, dt=dt, kind="field")
    other = plasma.evaluate(field[::-1][None, :], DENSITY, dt=dt, kind="field")
    assert not np.allclose(out, other[:, ::-1], rtol=1e-3, atol=0.0)


# --------------------------------------------------------------- the rate seam

# A table converged to 1e-6 in the rate carries that into the fraction through
# exp(-int W dt) and into both currents; measured worst deviation 6.6e-7 on this pulse.
TABLE_RTOL = 5e-6


def test_a_tabulated_rate_agrees_with_the_analytic_one_it_came_from():
    # The only test of the EvaluableRate seam with a real translation-layer object, and
    # what a propagation actually uses.
    clear_rate_table_cache()
    _, dt, field = pulse()
    analytic = adk_oracle("Ar")
    table = build_rate_table(analytic, (1e9, 1.5 * AMPLITUDE))
    response = trapezoid_plasma(Gas.Ar, analytic)
    direct = response.evaluate(field[None, :], DENSITY, dt=dt, kind="field")
    # The rate replaced by its table on the same response, as translation does: the
    # species keeps the energy per event the analytic rate bound to it.
    tabulated = dataclasses.replace(response, rate=table).evaluate(
        field[None, :], DENSITY, dt=dt, kind="field"
    )
    worst = np.max(np.abs(tabulated - direct)) / np.max(np.abs(direct))
    assert 0.0 < worst < TABLE_RTOL


def test_the_rate_protocol_accepts_any_callable():
    # EvaluableRate is a structural contract, not a validator: what the response checks
    # for itself is that the rate vanishes at zero field, which the loss term needs.
    assert isinstance(zero_rate, EvaluableRate)
    assert isinstance(adk_oracle("Ar"), EvaluableRate)


def test_a_response_can_be_built_on_a_grid_the_package_produces():
    # dt as a propagation supplies it, rather than from a bare linspace.
    grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 3e-6), 60e-15)
    field = (
        AMPLITUDE
        * np.exp(-((grid.t_fine / DURATION) ** 2))
        * np.cos(grid.omega0 * grid.t_fine)
    )
    plasma = trapezoid_plasma(Gas.Ar, adk_oracle("Ar"))
    out = plasma.evaluate(field[None, :], DENSITY, dt=grid.dt_fine, kind="field")
    assert out.shape == (1, grid.n_to)
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) > 0.0


# ------------------------------------------------------------------- error paths


def test_an_envelope_grid_is_rejected(response):
    with pytest.raises(GridError, match="field grid only"):
        response.evaluate(
            np.zeros((1, 8), dtype=complex), 1.0, dt=1e-16, kind="envelope"
        )
    with pytest.raises(GridError, match="field grid only"):
        response.ionised_fraction(
            np.zeros((1, 8), dtype=complex), dt=1e-16, kind="envelope"
        )


def test_a_complex_array_is_rejected(response):
    with pytest.raises(GridError, match="real field"):
        response.evaluate(np.zeros((1, 8), dtype=complex), 1.0, dt=1e-16, kind="field")


def test_too_many_polarisation_components_are_rejected(response):
    with pytest.raises(GridError, match="n_pol must be 1 or 2"):
        response.evaluate(np.zeros((3, 8)), 1.0, dt=1e-16, kind="field")


@pytest.mark.parametrize("shape", [(8,), (1, 2, 8)])
def test_an_array_that_is_not_two_dimensional_is_rejected(shape):
    # Through the guard directly: the jaxtyping hook rejects the rank first.
    with pytest.raises(GridError, match="field_t must have shape"):
        _check_plasma(np.zeros(shape), "field", 1e-16)


@pytest.mark.parametrize("bad", [0.0, -1e-16, math.inf, math.nan])
def test_a_bad_time_step_is_rejected(bad, response):
    with pytest.raises(GridError, match="dt must be positive"):
        response.evaluate(np.zeros((1, 8)), 1.0, dt=bad, kind="field")


def test_a_negative_density_is_rejected(response):
    with pytest.raises(GridError, match="density must be non-negative"):
        response.evaluate(np.zeros((1, 8)), -1.0, dt=1e-16, kind="field")


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_an_impossible_pre_ionisation_is_rejected(bad):
    with pytest.raises(MaterialError, match=r"preionisation must be in \[0, 1\]"):
        trapezoid_plasma(Gas.Ar, power_rate(), preionisation=bad)


def test_a_plane_with_another_pre_ionisation_is_a_replaced_response():
    # The pre-ionisation path is resolved in the translation layer; the oracle for a
    # plane where it differs is dataclasses.replace, which rebuilds and revalidates.
    plasma = trapezoid_plasma(Gas.Ar, power_rate(), preionisation=0.1)
    _, dt, field = pulse(n_samples=1025)
    at_plane = dataclasses.replace(plasma, preionisation=0.3)
    assert at_plane.preionisation == 0.3
    assert at_plane.rate is plasma.rate
    assert np.array_equal(
        at_plane.ionised_fraction(field[None, :], dt=dt, kind="field"),
        trapezoid_plasma(Gas.Ar, power_rate(), preionisation=0.3).ionised_fraction(
            field[None, :], dt=dt, kind="field"
        ),
    )
    with pytest.raises(MaterialError, match=r"preionisation must be in \[0, 1\]"):
        dataclasses.replace(plasma, preionisation=1.5)


@pytest.mark.parametrize("edge", [0.0, 1.0])
def test_the_ends_of_the_pre_ionisation_range_are_accepted(edge):
    assert (
        trapezoid_plasma(Gas.Ar, power_rate(), preionisation=edge).preionisation == edge
    )


def test_an_unknown_or_dataless_species_is_rejected():
    with pytest.raises(MaterialError, match="unknown gas"):
        trapezoid_plasma("Argon", power_rate())
    # Air is a mixture: each of its populations carries its own potential.
    with pytest.raises(MaterialError, match="no single ionisation potential"):
        trapezoid_plasma(Gas.Air, power_rate())


def test_a_rate_that_does_not_vanish_at_zero_field_is_rejected():
    # The loss term divides by the field; it is finite only because the rate is
    # identically zero where the field is too weak.
    def never_zero(field_amplitude):
        return np.full_like(np.asarray(field_amplitude, dtype=float), 1e14)

    with pytest.raises(MaterialError, match="exactly zero at zero field"):
        trapezoid_plasma(Gas.Ar, never_zero)


def test_a_rate_of_the_wrong_shape_is_rejected(response):
    def wrong(field_amplitude):
        # Right at zero field, so the constructor's own check passes and the shape
        # check is the one that fires.
        values = np.zeros_like(np.asarray(field_amplitude, dtype=float))
        return values if values.ndim == 0 else np.zeros(values.size + 1)

    with pytest.raises(GridError, match="the rate returned shape"):
        trapezoid_plasma(Gas.Ar, wrong).evaluate(
            np.zeros((1, 8)), 1.0, dt=1e-16, kind="field"
        )


@pytest.mark.parametrize("value", [math.nan, -1.0])
def test_a_rate_that_is_not_finite_and_non_negative_is_rejected(value):
    def bad(field_amplitude):
        magnitude = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(magnitude > 0.0, value, 0.0)

    with pytest.raises(GridError, match="finite and non-negative"):
        trapezoid_plasma(Gas.Ar, bad).evaluate(
            np.ones((1, 8)), 1.0, dt=1e-16, kind="field"
        )
