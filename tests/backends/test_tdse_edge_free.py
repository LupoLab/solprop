# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the edge-free gated TDSE source against analytic oracles (SEP 26 T2-T7).

Every case attaches a manufactured remainder ``S`` to an assembled capillary problem
(``tests/edge_source_cases.py``), so no atomic provider runs. Tolerances are absolute,
in the units of the source after the current factor (raw DFT units of ``S`` mapped by
the exactly known projection), and each is the sum of the implementation's own section
4.5 per-bin budget and the oracle's error, derived before the comparison. The tests are
fast: a few 4096-sample transforms each.
"""

import dataclasses
import json
import math

import mpmath
import numpy as np
import pytest
import scipy.fft
from numpy.testing import assert_array_equal
from scipy.special import erf, erfc
from tests.edge_source_cases import (
    EPS,
    ManufacturedBinding,
    evaluate,
    gate_values,
    make_case,
    spectral_derivative_oracle,
    steady_residual,
    truncating_source,
)
from tests.hacc_cases import plain

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.tdse import (
    FFT_ERROR_CONSTANT,
    atomic_edge_source,
    project_atomic_sources,
)
from solprop.exceptions import BackendError, TranslationError
from solprop.grid import planck_taper, planck_taper_derivative
from solprop.nonlinear.tdse import TDSEAtomicWindow
from solprop.translate.tdse import TDSEResult
from solprop.translate.tdse_source import (
    STEADY_CURRENT_CERTIFICATE_LIMIT,
    steady_current_certificate,
)

H = 1e-3
"""Steady remainder amplitude, A/m² per atom; every bound is linear in it."""

WINDOW = TDSEAtomicWindow(40e-15, 5e-15)
"""A 40 fs window with 5 fs tapers: 180 samples per taper on the fixture lattice."""


@pytest.fixture(scope="module", params=["window", "collar"])
def gated(request):
    """Both gates: an atomic window, and the grid collar of an unwindowed response."""
    window = WINDOW if request.param == "window" else None
    return make_case(window=window)


def _band(values, case):
    return float(np.linalg.norm(values[0, 0][case.active]))


# --- T2: steady and non-steady currents -----------------------------------------------


def test_t2a_a_globally_steady_current_cancels_to_the_budget_and_certificate(gated):
    case = gated
    steady = np.full(case.grid.n_to, H)
    held = evaluate(case, steady)
    truncating = truncating_source(case, steady)
    assert held.diagnostics["edge"]["steady_current_certificate"] <= 1e-15
    assert held.budget is not None
    # SEP 26 T2: the section 4.5 budget (absolute, per bin, here as a band norm) plus
    # the certified alias of S = 1 itself, H times its residual, in source units.
    # Measured: 6e-21 (window) and 7e-21 (collar), far inside the allowance.
    allowance = _band(held.budget, case) + _band(
        case.to_source(H * np.abs(steady_residual(case))), case
    )
    assert _band(held.source, case) <= allowance
    # The truncating source of the same current is not small: its switch-off.
    assert _band(truncating, case) > 1e6 * allowance


def test_t2b_a_zero_current_gives_exactly_zero(gated):
    for case in (gated,):
        result = evaluate(case, np.zeros(case.grid.n_to))
        # Structural zero: every operand is exactly zero, and 0 x anything is 0.
        assert_array_equal(result.source, np.zeros_like(result.source))
        edge = result.diagnostics["edge"]
        assert edge["leading_taper"]["ratio"] is None
        assert edge["leading_taper"]["reason"] == "zero remainder"
        active = edge["bands"]["active"]
        assert active["edge_over_truncating"] is None
        assert (
            active["edge_over_truncating_reason"] == "denominator below rounding floor"
        )
        assert active["edge_over_corrected"] is None


def _rise(case, centre=0.0, sigma=1e-15):
    """Return the T4 erf rise, its raw oracle and the T4 per-bin tolerance."""
    times, dt, omega, active = case.times, case.grid.dt_fine, case.omega, case.active
    rise = H / 2 * (1 + erf((times - centre) / sigma))
    _, one_low, one_high, _ = case.descriptor.source_breakpoints()
    distance = min(centre - one_low, one_high - centre)
    sign = (-1.0) ** np.arange(omega.size)
    oracle = np.zeros(omega.size, dtype=complex)
    band = omega[active]
    # (-1)^k/dt * H exp(-omega^2 sigma^2/4) exp(-i omega t_c) / (i omega): the raw DFT
    # of the held step, with the lattice's phase origin at n = 0 (SEP 26 T4).
    oracle[active] = (
        sign[active]
        / dt
        * H
        * np.exp(-(band**2) * sigma**2 / 4)
        * np.exp(-1j * band * centre)
        / (1j * band)
    )
    tail = np.zeros(omega.size)
    # The rise's mass outside the plateau and the alias beyond Nyquist, per bin.
    tail[active] = (
        H * erfc(distance / sigma) + H * math.exp(-((np.pi / dt * sigma) ** 2) / 4)
    ) / (dt * band)
    return rise, oracle, tail


def test_t2c_and_t4_an_interior_rise_keeps_its_analytic_held_spectrum():
    case = make_case(window=WINDOW)
    rise, oracle, tail = _rise(case)
    held = evaluate(case, rise)
    expected = case.to_source(oracle)
    assert held.budget is not None
    # T4 per bin: the rise's mass outside the plateau, H erfc(d/sigma)/(dt omega)
    # (d = 15 fs, sigma = 1 fs: 1e-100), the alias H exp(-(omega_N sigma)^2/4)/(dt
    # omega) (zero in double), the section 4.5 budget, the steady-current alias of the
    # trailing taper, at most H |r_k| with r the certified S = 1 residual, and 8 eps of
    # the oracle's own evaluation and the map's products. Measured: 1e-3 of it.
    tolerance = (
        np.abs(case.to_source(tail))
        + held.budget
        + np.abs(case.to_source(H * np.abs(steady_residual(case))))
        + 8 * EPS * np.abs(expected)
    )
    error = np.abs(held.source - expected)
    assert np.all(error[0, 0][case.active] <= tolerance[0, 0][case.active])
    # It differs from the truncating source of a globally steady current: the rise
    # is kept (T2 c).
    truncating = truncating_source(case, np.full(case.grid.n_to, H))
    assert _band(held.source - truncating, case) > 0.1 * _band(truncating, case)


def _synthetic(case, top):
    """Return T3's band-limited remainder: rise, drift, oscillation, taper slope.

    Every term is a Gaussian or an erf of width at least 0.8 fs, far above the
    17 fs / 1000 spacing, and the drift falls back to zero after the window, so the
    periodic extension is smooth and the spectral derivative exact to rounding.
    """
    times = case.times
    zero_low, one_low, one_high, zero_high = case.descriptor.source_breakpoints()
    drift = (
        H
        / 2
        * (
            erf((times - (one_low + 2e-15)) / 0.8e-15)
            - erf((times - (zero_high + 3e-15)) / 0.8e-15)
        )
    )
    centre = (one_low + one_high) / 2
    oscillation = (
        0.5 * H * np.cos(top * times) * np.exp(-(((times - centre) / 3e-15) ** 2))
    )
    middle, width = (one_high + zero_high) / 2, zero_high - one_high
    slope = (
        0.3
        * H
        * ((times - middle) / width)
        * np.exp(-(((times - middle) / (1.2 * width)) ** 2))
    )
    del zero_low
    return drift + oscillation + slope


def _identity_error(fft_size, multiple):
    """Compare the edge-free source with the spectral-derivative oracle."""
    probe = make_case(
        window=WINDOW,
        trange=64e-15,
        limits=(250e-9, 2e-6),
        fft_size=fft_size,
    )
    spacing = probe.grid.dt_fine
    # A ladder from 128 samples up; 1.0001 keeps 128 dt off a lattice tie.
    taper = 128 * multiple * spacing * 1.0001
    case = make_case(
        window=TDSEAtomicWindow(2 * taper + 12e-15, taper),
        trange=64e-15,
        limits=(250e-9, 2e-6),
        fft_size=fft_size,
    )
    top = float(case.omega[case.problem.grids.inband_mask].max())
    remainder = _synthetic(case, top)
    held = evaluate(case, remainder)
    oracle, oracle_budget = spectral_derivative_oracle(case, remainder)
    expected = case.to_source(oracle)
    assert held.budget is not None
    # T3: the sum of the two section 4.5 budgets, per bin.
    tolerance = held.budget + np.abs(case.to_source(oracle_budget * case.active))
    error = np.abs(held.source - expected)[0, 0][case.active]
    assert np.all(error <= tolerance[0, 0][case.active])
    return case, remainder, held, float(error.max() / np.abs(expected).max())


@pytest.mark.parametrize("fft_size", ["pow2", "smooth"], ids=["4096", "3840"])
def test_t3_the_identity_matches_a_spectral_derivative_at_every_resolution(fft_size):
    errors = [_identity_error(fft_size, multiple)[3] for multiple in (1, 2, 4, 8)]
    # Agreement does not worsen as the tapers are resolved: each finer taper stays
    # within the admission-limit error plus ten rounding units of the peak source.
    # Measured: 3.5e-16 to 8.3e-16 at 4096 samples, 4.2e-16 to 5.1e-16 at 3840.
    assert all(error <= errors[0] + 10 * EPS for error in errors[1:])


def test_t2d_and_t3_slope_through_the_taper_and_the_centred_difference_miss():
    case, remainder, held, error = _identity_error("pow2", 1)
    spacing = case.grid.dt_fine
    # The supporting study's second-order centred difference, in place of g'.
    centred = np.zeros(case.omega.size, dtype=complex)
    derivative = np.gradient(remainder, spacing)
    centred[case.active] = case.raw_dft(gate_values(case) * derivative)[case.active] / (
        1j * case.omega[case.active]
    )
    miss = float(
        np.max(np.abs(case.to_source(centred) - held.source)[0, 0][case.active])
        / np.max(np.abs(held.source))
    )
    # Its O((omega dt)^2) miss is recorded against (omega_max dt)^2/6 = 3.4e-3:
    # measured 1.7e-3, twelve orders above the identity form's 3.5e-16.
    predicted = (case.omega[case.active].max() * spacing) ** 2 / 6
    assert 1e-12 * miss > error
    assert 0.1 * predicted < miss <= predicted


def _reference_rfft(values):
    """Return the DFT of real samples to about one rounding, in double-double.

    Twiddles are evaluated at 120 bits by mpmath and split into two doubles; each
    product is exact by Dekker's splitting and each sum carried with an error-free
    transformation, so the result is the correctly rounded DFT to within a few units.
    """
    samples = values.size
    mpmath.mp.prec = 120
    twiddles = np.empty((4, samples))
    for index in range(samples):
        angle = 2 * mpmath.pi * index / samples
        cosine, sine = mpmath.cos(angle), mpmath.sin(angle)
        twiddles[0, index] = float(cosine)
        twiddles[1, index] = float(cosine - twiddles[0, index])
        twiddles[2, index] = float(sine)
        twiddles[3, index] = float(sine - twiddles[2, index])
    split = 134217729.0
    bins = np.arange(samples // 2 + 1)
    total = np.zeros((2, bins.size))
    correction = np.zeros((2, bins.size))
    for sample, value in enumerate(values):
        index = (sample * bins) % samples
        scaled = split * value
        high = scaled - (scaled - value)
        low = value - high
        for part, sign in ((0, 1.0), (1, -1.0)):
            leading, trailing = twiddles[2 * part, index], twiddles[2 * part + 1, index]
            product = value * leading
            scaled = split * leading
            leading_high = scaled - (scaled - leading)
            leading_low = leading - leading_high
            error = (
                (high * leading_high - product)
                + high * leading_low
                + low * leading_high
            ) + low * leading_low
            accumulated = total[part] + sign * product
            back = accumulated - total[part]
            correction[part] += (total[part] - (accumulated - back)) + (
                sign * product - back
            )
            correction[part] += sign * (error + value * trailing)
            total[part] = accumulated
    return (total[0] + correction[0]) + 1j * (total[1] + correction[1])


@pytest.mark.parametrize("samples", [4096, 3840], ids=["2^12", "2^8.3.5"])
def test_t3_the_fft_constant_holds_at_both_qualified_sizes(samples):
    values = np.random.default_rng(26).standard_normal(samples)
    reference = _reference_rfft(values)
    computed = scipy.fft.rfft(values)
    # The one-sided spectrum's full-spectrum 2-norm: interior bins count twice.
    weights = np.full(reference.size, 2.0)
    weights[0] = weights[-1] = 1.0
    error = math.sqrt(float(np.sum(weights * np.abs(computed - reference) ** 2)))
    norm = math.sqrt(float(np.sum(weights * np.abs(reference) ** 2)))
    observed = error / norm / (EPS * math.ceil(math.log2(samples)))
    # The assumed constant c_F = 8 of section 4.5 (FFT_ERROR_CONSTANT). Measured:
    # about 0.1 at both sizes.
    assert observed <= FFT_ERROR_CONSTANT


# --- T5, T6: a known oscillation and the leading taper --------------------------------


def test_t5_a_known_oscillation_inside_the_plateau_is_unchanged():
    held_case = make_case(window=WINDOW)
    times = held_case.times
    tau, carrier = 3e-15, float(held_case.omega[held_case.problem.grids.inband_mask][5])
    oscillation = H * np.cos(carrier * times) * np.exp(-((times / tau) ** 2))
    held = evaluate(held_case, oscillation)
    truncating = truncating_source(held_case, oscillation)
    _, one_low, one_high, _ = held_case.descriptor.source_breakpoints()
    reach = min(-one_low, one_high)
    weight = held_case.grid.dt_fine * float(
        np.sum(np.abs(held_case.descriptor.source_gate_derivative()))
    )
    omega, active = held_case.omega, held_case.active
    bound = np.zeros(omega.size)
    # Per bin, raw units: |E_k| <= sum |g'_n| |S_n| <= A exp(-(t_p/tau)^2) W/dt on the
    # tapers, then divided by omega_k (SEP 26 T5); plus the budget.
    bound[active] = (
        H
        * math.exp(-((reach / tau) ** 2))
        * weight
        / (held_case.grid.dt_fine * omega[active])
    )
    assert held.budget is not None
    tolerance = np.abs(held_case.to_source(bound)) + held.budget
    difference = np.abs(held.source - truncating)
    assert np.all(difference[0, 0][active] <= tolerance[0, 0][active])


def test_t6_the_leading_taper_ratio_and_both_bounds():
    case = make_case(window=WINDOW)
    times, dt = case.times, case.grid.dt_fine
    zero_low, one_low, _, zero_high = case.descriptor.source_breakpoints()
    lead = (times > zero_low) & (times < one_low)
    # A known remainder in the leading taper beside a larger one on the plateau.
    remainder = 0.02 * H * np.exp(-(((times - (zero_low + one_low) / 2) / 1e-15) ** 2))
    remainder += H * np.exp(-((times / 4e-15) ** 2))
    held = evaluate(case, remainder)
    record = held.diagnostics["edge"]["leading_taper"]
    support = (times >= zero_low) & (times <= zero_high)
    expected = np.max(np.abs(remainder[lead])) / np.max(np.abs(remainder[support]))
    # The same two maxima and one division: one rounding.
    assert record["ratio"] == pytest.approx(expected, rel=EPS, abs=0)
    assert record["reason"] is None
    assert record["lead_samples"] == int(np.count_nonzero(lead))
    derivative = case.descriptor.source_gate_derivative()
    lead_derivative = np.where(lead, derivative, 0.0)
    weight = dt * float(np.sum(np.abs(lead_derivative)))
    assert record["sampled_weight"] == pytest.approx(weight, rel=4 * EPS)
    omega, active = case.omega, case.active
    edge = case.raw_dft(lead_derivative * remainder)[active] / (1j * omega[active])
    peak = float(np.max(np.abs(remainder[lead])))
    # Per bin (section 4.3): |E_-,k/(i omega_k)| <= W_-/(dt omega_k) max_lead |S|, plus
    # the leading edge's own transform rounding, at most c_F eps log2 N sqrt(N)
    # ||g'_- S||_2 / omega_k per bin.
    samples = case.grid.n_to
    rounding = (
        FFT_ERROR_CONSTANT
        * EPS
        * math.log2(samples)
        * math.sqrt(samples)
        * float(np.linalg.norm(lead_derivative * remainder))
    )
    per_bin = (weight / dt * peak + rounding) / omega[active]
    assert np.all(np.abs(edge) <= per_bin)
    # Band norm, in the units of the edge norms: the recorded bound covers the leading
    # edge's actual share of the edge term.
    actual = np.zeros(omega.size, dtype=complex)
    actual[active] = edge
    for name, band in (("active", active), ("inband", case.problem.grids.inband_mask)):
        share = float(np.linalg.norm(case.to_source(actual)[0, 0][band]))
        # The recorded bound uses the absolute projection, at least |K| max |S|; the
        # same rounding allowance as the per-bin bound, relatively.
        assert share <= record["band_bound"][name] * (
            1 + rounding / (weight / dt * peak)
        )
        assert share > 0


# --- T7: admission and certificate ----------------------------------------------------


HACC_LIMITS = (9e-8, 1.6e-6)
"""The haCC lattice: 90 nm-1.6 um on a 60 fs window, N = 16384 with refinement 8."""


def _certificate(case):
    return steady_current_certificate(case.descriptor, case.omega, case.active)


def test_t7_the_review_counterexample_is_refused_by_its_certificate():
    # The review's counterexample, an 80 dt window with dt/4 tapers: no fixed sample
    # count refuses it any more; the certificate does. Its sampled g' is zero, so the
    # edge term cancels nothing and kappa is of order one (measured 0.92).
    probe = make_case(window=WINDOW, limits=HACC_LIMITS)
    spacing = probe.grid.dt_fine
    assert probe.grid.n_to == 16384
    assert spacing == pytest.approx(6.254e-18, rel=1e-3)
    case = make_case(
        window=TDSEAtomicWindow(80 * spacing, spacing / 4),
        limits=HACC_LIMITS,
    )
    assert _certificate(case) > 0.1
    with pytest.raises(TranslationError, match="not certified") as caught:
        case.problem.validate()
    assert "TDSESampling" in str(caught.value)


@pytest.mark.parametrize(
    ("samples", "admitted"), [(4, False), (40, False), (127.5, True)]
)
def test_t7_the_certificate_alone_decides_a_taper_on_the_hacc_lattice(
    samples, admitted
):
    # Measured kappa on this lattice: 1.5e-3 at 4 samples and 4.3e-10 at 40, both
    # alias-dominated and refused; 7.8e-17 at 127.5 samples, the taper SEP 26's fixed
    # 128-sample minimum refused, which is rounding level and admitted. The decision
    # sits more than two decades from the limit either side, so it does not depend on
    # the rounding of the transforms.
    probe = make_case(window=WINDOW, limits=HACC_LIMITS)
    case = make_case(
        window=TDSEAtomicWindow(40e-15, samples * probe.grid.dt_fine),
        limits=HACC_LIMITS,
    )
    certificate = _certificate(case)
    if admitted:
        assert certificate <= 1e-15
        case.problem.validate()
    else:
        assert certificate >= 4 * STEADY_CURRENT_CERTIFICATE_LIMIT
        with pytest.raises(TranslationError, match="not certified"):
            case.problem.validate()


def test_t7_a_76_sample_grid_collar_is_admitted():
    # An unrefined 200 nm-2 um, 40 fs lattice: its grid collar's tapers span 76 fine
    # samples. Measured kappa 3.4e-13, well inside the limit.
    case = make_case(limits=(200e-9, 2e-6), trange=40e-15, refinement=1)
    breakpoints = case.descriptor.source_breakpoints()
    taper = min(breakpoints[1] - breakpoints[0], breakpoints[3] - breakpoints[2])
    assert 76 <= taper / case.grid.dt_fine < 77
    assert _certificate(case) <= 1e-12
    case.problem.validate()


def test_t7_a_long_window_with_little_inband_gate_spectrum_is_admitted():
    # A 1 ps, 400 nm-2 um grid collar: almost none of the gate's spectrum lies in the
    # band, so the band-relative ratio SEP 26 first used compared two rounding-level
    # norms (1.4e-7, refused). Against the full transform of g it is 5.9e-17.
    case = make_case(limits=(400e-9, 2e-6), trange=1e-12, refinement=1)
    assert case.grid.n_to == 8192
    assert _certificate(case) <= 1e-15
    case.problem.validate()


def test_t7_a_resolved_taper_is_certified_in_the_metadata():
    probe = make_case(window=WINDOW)
    spacing = probe.grid.dt_fine
    case = make_case(window=TDSEAtomicWindow(40e-15, 128 * spacing))
    window = case.descriptor.atomic_window
    assert window is not None and window.taper == 128 * spacing
    backend = NumpyRefBackend(
        case.problem,
        grid=case.grid,
        tdse_binding=ManufacturedBinding(case.descriptor, 1),
    )
    try:
        record = plain(backend.metadata()["tdse_source"])
    finally:
        backend.close()
    certificate = record["steady_current_certificate"]
    assert certificate <= STEADY_CURRENT_CERTIFICATE_LIMIT
    # It never enters the descriptor or its identity.
    assert "certificate" not in case.descriptor.configuration_json
    assert "certificate" not in json.dumps(
        {name: str(value) for name, value in vars(case.descriptor).items()}
    )
    # An independent recomputation: numpy's own FFT of the gate and its derivative.
    # kappa is itself relative to ||DFT g||_2, so agreement within 10 eps log2 N is
    # absolute in it (SEP 26 T7). Measured difference: 1e-16.
    assert abs(certificate - _independent_certificate(case)) <= 10 * EPS * 12


def _independent_certificate(case):
    times = case.times
    breakpoints = case.descriptor.source_breakpoints()
    spectrum = np.fft.rfft(planck_taper(times, *breakpoints))
    gate = spectrum[case.problem.grids.crop_index]
    edge = np.fft.rfft(planck_taper_derivative(times, *breakpoints))[
        case.problem.grids.crop_index
    ]
    active, omega = case.active, case.omega
    corrected = gate[active] - edge[active] / (1j * omega[active])
    # Relative to the full one-sided transform of the gate, not its active band.
    return float(np.linalg.norm(corrected) / np.linalg.norm(spectrum))


def test_t7_an_alias_dominated_certificate_agrees_with_the_independent_one():
    # Any taper builds a descriptor, so the certificate can be measured where the
    # alias, not rounding, sets it: a 20-sample taper, kappa 1.1e-6.
    probe = make_case(window=WINDOW)
    case = make_case(window=TDSEAtomicWindow(40e-15, 20 * probe.grid.dt_fine))
    certificate = steady_current_certificate(case.descriptor, case.omega, case.active)
    assert certificate > STEADY_CURRENT_CERTIFICATE_LIMIT
    # Its numerator is a cancellation of two terms of at most the size of ||DFT g||_A
    # <= ||DFT g||_2, so the two transforms agree absolutely in kappa, within 10 eps
    # log2 N, and relatively only to about eps / kappa.
    independent = _independent_certificate(case)
    assert abs(certificate - independent) <= 10 * EPS * 12


def test_certificate_of_an_empty_band_is_zero():
    # Nothing is propagated, so nothing is left to certify; the denominator, the full
    # transform of the gate, stays positive.
    case = make_case(window=WINDOW)
    silent = np.zeros_like(case.active)
    assert steady_current_certificate(case.descriptor, case.omega, silent) == 0.0


# --- every electron once, the truncating formula and the seam's refusals -------------


def test_the_source_is_the_identity_of_one_remainder_and_j1_never_enters():
    case = make_case(window=WINDOW)
    times = case.times
    remainder = H * np.tanh(times / 3e-15) * np.exp(-((times / 12e-15) ** 2))
    plain = evaluate(case, remainder)
    # A first-order current 2^10 times larger cancels in J - j_1 and nowhere else.
    shadowed = evaluate(case, remainder, first=2.0**10 * H * np.sin(2.4e15 * times))
    gate = gate_values(case)
    derivative = case.descriptor.source_gate_derivative()
    active, omega = case.active, case.omega
    raw = np.zeros(omega.size, dtype=complex)
    raw[active] = case.raw_dft(gate * remainder)[active] - case.raw_dft(
        derivative * remainder
    )[active] / (1j * omega[active])
    expected = case.to_source(raw)
    assert plain.budget is not None and shadowed.budget is not None
    # DFT[gS] - DFT[g'S]/(i omega) with the same S, density and scale, once: within
    # the budget of each evaluation (the shadow's budget carries its raw operands)
    # plus 8 eps of the map's products.
    for result in (plain, shadowed):
        assert result.budget is not None
        tolerance = result.budget + 8 * EPS * np.abs(expected)
        assert np.all(
            np.abs(result.source - expected)[0, 0][active] <= tolerance[0, 0][active]
        )


def test_the_seam_refuses_a_missing_or_misplaced_edge_array():
    held = make_case(window=WINDOW)
    plain_case = make_case(kerr_model="optmat_cubic")
    remainder = np.full(held.grid.n_to, H)
    for case in (held, plain_case):
        rows = case.problem.transform.n_nodes
        values = np.broadcast_to(remainder, (rows, case.grid.n_to)).copy()
        result = TDSEResult(
            values,
            np.zeros_like(values),
            input_identity="manufactured",
            configuration_identity=case.descriptor.identity,
            valid=np.ones(values.shape, dtype=bool),
            shadow_orders=(1,),
        )
        current = values[:, None, :]
        edge = None if case is held else np.zeros_like(current)
        with pytest.raises(BackendError, match="edge array"):
            project_atomic_sources(
                np.zeros_like(current),
                current,
                result,
                case.problem,
                case.bundle,
                case.prefactor,
                edge=edge,
            )
        if case is held:
            for bad in (
                np.zeros_like(current)[:, :, :-1],
                np.full_like(current, np.nan),
            ):
                with pytest.raises(BackendError, match="finite and match"):
                    project_atomic_sources(
                        np.zeros_like(current),
                        current,
                        result,
                        case.problem,
                        case.bundle,
                        case.prefactor,
                        edge=bad,
                    )
        else:
            assert atomic_edge_source(result, case.problem, case.bundle) is None


def test_the_edge_assembly_refuses_a_foreign_or_nonfinite_result():
    case = make_case(window=WINDOW)
    rows = case.problem.transform.n_nodes
    values = np.full((rows, case.grid.n_to), H)

    def result(total, identity):
        return TDSEResult(
            total,
            np.zeros_like(values),
            input_identity="manufactured",
            configuration_identity=identity,
            valid=np.ones(values.shape, dtype=bool),
            shadow_orders=(1,),
        )

    with pytest.raises(BackendError, match="does not identify"):
        atomic_edge_source(result(values, "another"), case.problem, case.bundle)
    huge = np.full_like(values, np.finfo(float).max)
    # g' S overflows: the guard refuses it (the overflow itself is expected here).
    with np.errstate(over="ignore"), pytest.raises(BackendError, match="non-finite"):
        atomic_edge_source(
            result(huge, case.descriptor.identity), case.problem, case.bundle
        )
    bare = dataclasses.replace(case.problem, tdse=None, current_numerator=None)
    with pytest.raises(BackendError, match="requires a TDSE descriptor"):
        atomic_edge_source(result(values, "x"), bare, case.bundle)


def test_a_gate_with_no_sample_in_its_leading_taper_says_so():
    # The review's counterexample shape, 80 dt with dt/4 tapers, builds a descriptor
    # (its certificate refuses it at validation). The diagnostic of an evaluation
    # reports the empty taper rather than a ratio over no samples.
    probe = make_case(window=WINDOW)
    spacing = probe.grid.dt_fine
    case = make_case(window=TDSEAtomicWindow(80 * spacing, spacing / 4))
    record = evaluate(case, np.full(case.grid.n_to, H)).diagnostics["edge"]
    assert record["leading_taper"]["ratio"] is None
    assert record["leading_taper"]["reason"] == "empty leading taper"
    assert record["leading_taper"]["lead_samples"] == 0


def test_a_cubic_closure_records_an_ungated_source_and_no_certificate():
    # The cubic replacements are never gated (the source filter's removal): the run
    # metadata says so and carries no certificate to qualify a gate.
    from solprop.backends.numpy_ref.backend import _source_record  # noqa: PLC0415

    case = make_case(kerr_model="optmat_cubic")
    assert not case.descriptor.gated
    assert _source_record(case.problem) == {"gated": False}
