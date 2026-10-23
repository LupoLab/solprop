# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the haCC weak-field consistency protocol (SEP 20 section 7.1 items 11-13).

The reduction is checked against manufactured susceptibilities and against an
independent witness: a damped Lorentz oscillator integrated by SciPy, whose
susceptibility is analytic. That witness freezes protocol version 1 (Q16).
"""

import dataclasses
import json
import logging
import math
import subprocess
import sys
from typing import Any

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from tests.hacc_cases import (
    alpha_law,
    hacc_response,
    helium_alpha,
    install,
    preset_basis,
    tiny_basis,
)

from solprop.constants import E_CHARGE, EPS0, M_E, C
from solprop.driver import tdse, tdse_consistency
from solprop.driver.tdse_consistency import (
    PROTOCOL_VERSION,
    apply_outcome,
    cache_directory,
    check_consistency,
    cubic_evidence,
    evaluate_consistency,
    protocol_field,
    reduce_linear,
)
from solprop.exceptions import (
    BackendError,
    MemoryBudgetError,
    TDSEConsistencyError,
    TDSEConsistencyWarning,
)
from solprop.grid import FieldGrid
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEConsistency,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.physdata import species
from solprop.translate.tdse import TDSEDescriptor

_LAMBDA = 800e-9
_SUPPORT = (0.0, 2 * math.pi * C / 60e-9)
_REVISION = tdse.pinned_revision()


def _reference(omega):
    return helium_alpha(omega)


def _zero_law(field, spacing):
    return np.zeros_like(field), np.zeros_like(field)


def _reduction_valid(entry):
    return entry["reduction"]["valid"] is True


class _Source:
    """A diagnostic source following a manufactured response law."""

    def __init__(self, law):
        self.law = law
        self.calls = 0

    def diagnostic_currents(self, field, spacing):
        self.calls += 1
        first, total = self.law(field, spacing)
        return total[0], first[0]


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path))
    return tmp_path / "tdse-consistency"


def _check(source, basis=None, policy=None, **changes):
    values = {
        "reference_wavelength": _LAMBDA,
        "provider_revision": _REVISION,
        "computed_on": {"execution_space": "OpenMP", "build": {}, "threads": 1},
    }
    values.update(changes)
    return evaluate_consistency(
        source,
        tiny_basis() if basis is None else basis,
        TDSEConsistency() if policy is None else policy,
        **values,
    )


# --- item 11: the reduction ------------------------------------------------------


def test_protocol_field_is_the_frozen_version_1_pulse():
    spacing, field = protocol_field(_LAMBDA)
    assert PROTOCOL_VERSION == 1
    assert (spacing, field.size) == (25e-18, 1601)
    amplitude = math.sqrt(2 * 1e15 / (EPS0 * C))
    assert field[800] == pytest.approx(amplitude, rel=1e-15)
    # 5 FWHM from the centre: below the 1e-6 endpoint-leakage limit by far.
    assert np.max(np.abs(field[:32])) < 1e-14 * amplitude
    with pytest.raises(BackendError, match="wavelength"):
        protocol_field(-1.0)


@pytest.mark.parametrize("k1", [1.0, 1.05, 0.5, 2.0, -1.0])
def test_manufactured_susceptibility_gives_kappa_exactly_k1(k1):
    spacing, field = protocol_field(_LAMBDA)
    first, _ = alpha_law(k1)(field[None, :], spacing)
    reduction = reduce_linear(spacing, field, first[0], _reference, _SUPPORT)
    assert reduction.valid and reduction.reason is None
    kappa = np.array(reduction.kappa_real) + 1j * np.array(reduction.kappa_imag)
    # rfft(irfft(X)) returns X to about eps log2(N) |X|_max; at the band's 1e-3 edge
    # that is about 1e3 * 2.2e-16 * 11 ~ 3e-12 relative. 1e-9 bounds it with margin.
    assert np.max(np.abs(kappa - k1)) < 1e-9 * abs(k1)
    assert reduction.deviation == pytest.approx(abs(k1 - 1), abs=1e-9)
    assert reduction.worst_wavelength in reduction.wavelength


def test_band_rule_excludes_weak_bins_dc_and_unsupported_frequencies():
    spacing, field = protocol_field(_LAMBDA)
    first, _ = alpha_law(1.0)(field[None, :], spacing)
    full = reduce_linear(spacing, field, first[0], _reference, _SUPPORT)
    omega = 2 * math.pi * C / np.array(full.wavelength)
    spectrum = np.abs(np.fft.rfft(field))
    bins = 2 * math.pi * np.fft.rfftfreq(field.size, spacing)
    floor = 1e-3 * np.max(spectrum)
    expected = bins[(bins > 0) & (spectrum > floor)]
    assert np.allclose(omega, expected, rtol=1e-12, atol=0)
    narrow = (2.0e15, 2.6e15)
    limited = reduce_linear(spacing, field, first[0], _reference, narrow)
    limited_omega = 2 * math.pi * C / np.array(limited.wavelength)
    assert limited_omega.min() >= narrow[0] and limited_omega.max() <= narrow[1]
    assert len(limited.wavelength) < len(full.wavelength)
    # A unipolar pulse has its largest bin at DC, which is never divided by i omega.
    times = np.arange(field.size) * spacing
    unipolar = np.exp(-(((times - 20e-15) / 3e-15) ** 2))
    current = np.gradient(unipolar, spacing) * EPS0 * 2.6e-30
    dc = reduce_linear(spacing, unipolar, current, _reference, _SUPPORT)
    assert dc.valid or "floor" in str(dc.reason)
    assert all(math.isfinite(value) for value in dc.wavelength)


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("empty_band", "band is empty"),
        ("zero_current", "identically zero"),
        ("shape", "one shape"),
        ("nonfinite", "one shape"),
        ("current_leakage", "endpoint leakage of the first-order"),
        ("field_leakage", "endpoint leakage of the field"),
        ("rounding", "rounding floor"),
        ("reference_nan", "non-finite"),
    ],
)
def test_invalid_extractions_name_their_failed_condition(monkeypatch, case, reason):
    spacing, field = protocol_field(_LAMBDA)
    first = alpha_law(1.0)(field[None, :], spacing)[0][0]
    drive, support, reference = field, _SUPPORT, _reference
    if case == "empty_band":
        support = (1e17, 2e17)
    elif case == "zero_current":
        first = np.zeros_like(first)
    elif case == "shape":
        first = first[:-1]
    elif case == "nonfinite":
        first = first.copy()
        first[3] = np.nan
    elif case == "current_leakage":
        first = first + 1e-4 * np.max(np.abs(first))
    elif case == "field_leakage":
        drive = field + 1e-3 * np.max(np.abs(field))
    elif case == "rounding":
        # Raise the floor above every band bin: the rule, not a physical current.
        monkeypatch.setattr(tdse_consistency, "_ROUNDING_FACTOR", 1e16)
    elif case == "reference_nan":

        def reference(omega):
            values = helium_alpha(omega)
            values[len(values) // 2] = np.nan
            return values

    reduction = reduce_linear(spacing, drive, first, reference, support)
    assert not reduction.valid
    assert reason in str(reduction.reason)


def test_a_phase_mismatch_is_not_hidden_by_the_real_part():
    spacing, field = protocol_field(_LAMBDA)
    phase = 0.2
    first, _ = alpha_law(1.0, phase=phase)(field[None, :], spacing)
    reduction = reduce_linear(spacing, field, first[0], _reference, _SUPPORT)
    # kappa = exp(-i phase): its real part is within 2% of one, the ratio is not.
    expected = abs(np.exp(-1j * phase) - 1)
    assert reduction.deviation == pytest.approx(expected, rel=1e-9)
    assert max(abs(value - 1) for value in reduction.kappa_real) < 0.021


def _lorentz_witness(omega0, damping):
    """Integrate x'' + damping x' + omega0^2 x = -(e/m) E(t) independently."""
    spacing, field = protocol_field(_LAMBDA)
    centre = 20e-15
    amplitude = field[800]
    carrier = 2 * math.pi * C / _LAMBDA

    def electric(t):
        shifted = t - centre
        return (
            amplitude
            * math.exp(-2 * math.log(2) * (shifted / 4e-15) ** 2)
            * math.cos(carrier * shifted)
        )

    scale = E_CHARGE / M_E * amplitude / omega0**2  # characteristic displacement

    def rhs(t, state):
        x, v = state
        return [v, -E_CHARGE / M_E * electric(t) / scale - damping * v - omega0**2 * x]

    times = np.arange(field.size) * spacing
    solution = solve_ivp(
        rhs,
        (0.0, times[-1]),
        [0.0, 0.0],
        method="DOP853",
        t_eval=times,
        rtol=1e-12,
        atol=1e-12 * omega0,
    )
    assert solution.success
    current = -E_CHARGE * scale * solution.y[1]  # j = dp/dt = -e x'

    def alpha(omega):
        # Optics convention (d/dt -> -i omega): the absorptive part is positive.
        return E_CHARGE**2 / (
            M_E * EPS0 * (omega0**2 - omega**2 - 1j * damping * omega)
        )

    return spacing, field, current, alpha


@pytest.mark.parametrize("damping", [0.0, 2e14])
def test_independent_lorentz_witness_freezes_the_protocol(damping):
    spacing, field, current, alpha = _lorentz_witness(3.0e16, damping)
    reduction = reduce_linear(spacing, field, current, alpha, _SUPPORT)
    assert reduction.valid
    kappa = np.array(reduction.kappa_real) + 1j * np.array(reduction.kappa_imag)
    # The integration is accurate to about 1e-12 of the peak current; at the band's
    # 1e-3 edge the ratio amplifies that to about 1e-9. The DFT ratio equals the
    # continuous one because both records are 1e-15 transients at the window ends.
    assert np.max(np.abs(kappa - 1)) < 1e-6
    if damping:
        # The convention is right: a sign error in i omega or the conjugation would
        # put kappa near conj(alpha)/alpha, far from one at the resonance's tail.
        assert np.max(np.abs(np.array(reduction.kappa_imag))) < 1e-6


def test_invalid_diagnostic_refuses_whatever_the_bounds_say():
    source = _Source(
        lambda field, spacing: (np.zeros_like(field), np.zeros_like(field))
    )
    result = _check(source, policy=TDSEConsistency(warn=100.0, refuse=1e6))
    assert result.outcome == "invalid"
    with pytest.raises(TDSEConsistencyError, match="invalid") as error:
        apply_outcome(result)
    assert error.value.result is result


def test_cubic_evidence_recovers_k3_leakage_and_the_ladder_check():
    spacing, field = protocol_field(_LAMBDA)
    gamma = species("He").kerr_coefficient(_LAMBDA)
    assert gamma is not None
    k3 = 0.95
    omega = 2 * math.pi * np.fft.rfftfreq(field.size, spacing)

    def cubic(scale):
        cube = np.fft.rfft((scale * field) ** 3)
        return np.fft.irfft(1j * omega * EPS0 * gamma / k3 * cube, n=field.size)

    first = alpha_law(1.0)(field[None, :], spacing)[0][0]
    # The nonlinear current is about 1.5e-5 of the linear one at 1e11 W/cm^2, so a
    # 1e-10 leak is about 1e-5 of the cubic term: c3_hat's 30/24 share of it biases
    # kappa_3 by about 1e-5, well inside the 1e-3 tolerance below.
    leak = 1e-10
    fifth = 1e-3 * np.max(np.abs(cubic(1.0))) / np.max(np.abs(field)) ** 5

    def remainder(scale, seventh=0.0):
        values = scale * leak * first + cubic(scale) + fifth * (scale * field) ** 5
        if seventh:
            values = values + seventh * (scale * field) ** 7
        return values

    evidence = cubic_evidence(
        spacing, field, {a: remainder(a) for a in (1, 2, 4)}, gamma
    )
    # The seventh order is absent, so only the leakage biases kappa_3.
    assert evidence.kappa_3 == pytest.approx(k3, rel=1e-3)
    assert evidence.residual < 1e-2
    # Without a seventh order, the leak alone misses the lambda = 4 row: c3_hat and
    # c5_hat carry +30/24 and -6/24 of c1, so the prediction holds -176 c1 against
    # the row's +4 c1, a residual of exactly 180 ||c1|| / ||r_4||.
    expected = 180 * np.linalg.norm(leak * first) / np.linalg.norm(remainder(4))
    assert evidence.ladder_ok
    assert evidence.ladder_residual == pytest.approx(expected, rel=1e-6)
    # The fifth order is 1e-3 of the cubic here, so the geometric seventh-order
    # floor, 64 (||c5_hat|| / ||c3_hat||)^2, lies above this tiny leak: unresolved.
    c3_hat = (32 * remainder(1) - remainder(2)) / 24
    c5_hat = (remainder(2) - 8 * remainder(1)) / 24
    floor = 64 * (np.linalg.norm(c5_hat) / np.linalg.norm(c3_hat)) ** 2
    assert evidence.seventh_order_floor == pytest.approx(floor, rel=1e-12)
    assert evidence.leakage < evidence.seventh_order_floor
    assert not evidence.leakage_resolved
    ratio = np.linalg.norm(leak * first) / np.linalg.norm(cubic(1.0))
    assert evidence.leakage == pytest.approx(ratio, rel=1e-6)
    lower, upper = evidence.band_wavelength
    assert lower < _LAMBDA < upper
    # A resolvable seventh order trips the lambda = 4 ladder check.
    strong = 1e-2 * np.max(np.abs(cubic(1.0))) / np.max(np.abs(field)) ** 7
    tripped = cubic_evidence(
        spacing,
        field,
        {a: remainder(a, seventh=strong) for a in (1, 2, 4)},
        gamma,
    )
    assert not tripped.ladder_ok and tripped.ladder_residual > 1e-4


def test_the_ladder_limit_admits_perturbative_scaling_and_resolves_a_leak():
    spacing, field = protocol_field(_LAMBDA)
    gamma = species("He").kerr_coefficient(_LAMBDA)
    assert gamma is not None
    omega = 2 * math.pi * np.fft.rfftfreq(field.size, spacing)
    cube = np.fft.rfft(field**3)
    cubic = np.fft.irfft(1j * omega * EPS0 * gamma * cube, n=field.size)
    first = alpha_law(1.0)(field[None, :], spacing)[0][0]
    peak_field = np.max(np.abs(field))
    peak_cubic = np.max(np.abs(cubic))
    # Successive orders 1e-4 apart, as at 1e11 W/cm^2: c5 ~ 1e-4 c3, c7 ~ 1e-8 c3.
    fifth = 1e-4 * peak_cubic / peak_field**5
    seventh = 1e-8 * peak_cubic / peak_field**7

    def remainders(leak):
        return {
            a: a * leak * first
            + a**3 * cubic
            + fifth * (a * field) ** 5
            + seventh * (a * field) ** 7
            for a in (1, 2, 4)
        }

    clean = cubic_evidence(spacing, field, remainders(0.0), gamma)
    # The seventh order alone misses the lambda = 4 row by about 11520 c7 / r4,
    # of order 1e-6: well inside the 1e-4 limit, which a 1e-2 limit never tested.
    assert clean.ladder_ok and clean.ladder_residual < 1e-5
    # c5_hat = c5 + 5 c7 and c3_hat = c3 - 4 c7: the floor is 64 times the inputs'
    # order ratio squared, to 2 * 5 c7 / c5 ~ 1e-3 from the squared fifth order.
    ratio = np.linalg.norm(fifth * field**5) / np.linalg.norm(cubic)
    assert clean.seventh_order_floor == pytest.approx(64 * ratio**2, rel=2e-3)
    # A leak above the floor is resolved and still inside the ladder limit.
    leak = 1e-6 * np.linalg.norm(cubic) / np.linalg.norm(first)
    leaky = cubic_evidence(spacing, field, remainders(leak), gamma)
    assert leaky.leakage_resolved and leaky.ladder_ok
    # The fitted c1 is the leak plus exactly 64 c7, which bounds the difference.
    bias = 64 * np.linalg.norm(seventh * field**7) / np.linalg.norm(cubic)
    assert abs(leaky.leakage - 1e-6) <= bias * (1 + 1e-6)


def test_third_harmonic_coefficient_leaves_the_fundamental_band_kappa_3():
    spacing, field = protocol_field(_LAMBDA)
    gamma = species("He").kerr_coefficient(_LAMBDA)
    assert gamma is not None
    omega = 2 * math.pi * np.fft.rfftfreq(field.size, spacing)
    # Split E^3 into its fundamental (3/4 |a|^2 Re a) and third-harmonic
    # (1/4 Re a^3) parts with the analytic signal a.
    spectrum = np.fft.fft(field)
    spectrum[1 : field.size // 2 + 1] *= 2
    spectrum[field.size // 2 + 1 :] = 0
    analytic = np.fft.ifft(spectrum)
    fundamental = 0.75 * np.abs(analytic) ** 2 * np.real(analytic)
    harmonic = 0.25 * np.real(analytic**3)

    def current(scale, third):
        cube = scale**3 * (fundamental + third * harmonic)
        return np.fft.irfft(1j * omega * EPS0 * gamma * np.fft.rfft(cube), n=field.size)

    plain = cubic_evidence(
        spacing, field, {a: current(a, 1.0) for a in (1, 2, 4)}, gamma
    )
    shifted = cubic_evidence(
        spacing, field, {a: current(a, 1.05) for a in (1, 2, 4)}, gamma
    )
    # The third-harmonic spectrum is below 1e-6 of its peak on the fundamental band,
    # so a 5% change of its coefficient moves kappa_3 by less than 1e-6.
    assert shifted.kappa_3 == pytest.approx(plain.kappa_3, rel=1e-6)
    assert plain.kappa_3 == pytest.approx(1.0, rel=1e-2)


def test_cubic_evidence_refuses_an_incomplete_or_degenerate_ladder():
    spacing, field = protocol_field(_LAMBDA)
    with pytest.raises(BackendError, match="ladder"):
        cubic_evidence(spacing, field, {1: field, 2: field}, 1e-52)
    zero = np.zeros_like(field)
    with pytest.raises(BackendError, match="degenerate"):
        cubic_evidence(spacing, field, {1: zero, 2: zero, 4: zero}, 1e-52)


# --- item 12: the gate -------------------------------------------------------------


@pytest.mark.parametrize(
    ("k1", "policy", "outcome"),
    [
        (1.05, TDSEConsistency(), "passed"),
        (0.95, TDSEConsistency(), "passed"),
        (1.2, TDSEConsistency(), "warned"),
        (0.8, TDSEConsistency(), "warned"),
        (0.5, TDSEConsistency(), "refused"),
        (2.0, TDSEConsistency(), "refused"),
        (-1.0, TDSEConsistency(), "refused"),
        (0.5, TDSEConsistency(refuse=0.6), "warned"),
        (1.05, TDSEConsistency(warn=0.03), "warned"),
    ],
)
def test_gate_decisions_follow_the_section_5_4_rules(k1, policy, outcome):
    result = _check(_Source(alpha_law(k1)), policy=policy)
    assert result.outcome == outcome
    if outcome == "passed":
        apply_outcome(result)
    elif outcome == "warned":
        with pytest.warns(TDSEConsistencyWarning, match="kappa_1") as record:
            apply_outcome(result)
        assert len(record) == 1
        assert "nm" in str(record[0].message)
    else:
        with pytest.raises(TDSEConsistencyError, match="widen") as error:
            apply_outcome(result)
        assert error.value.result is result


def test_gate_boundaries_are_exact():
    decide = tdse_consistency._decide
    policy = TDSEConsistency()
    assert decide(0.1, policy) == "passed"
    assert decide(math.nextafter(0.1, 1.0), policy) == "warned"
    assert decide(0.3, policy) == "warned"
    assert decide(math.nextafter(0.3, 1.0), policy) == "refused"


def test_skipped_check_makes_no_call_and_warns_once(cache):
    source = _Source(alpha_law(1.0))
    result = _check(source, policy=TDSEConsistency(enforce=False))
    assert result.outcome == "skipped" and source.calls == 0
    assert result.reduction is None and result.cache_key is None
    assert not cache.exists()
    with pytest.warns(TDSEConsistencyWarning, match="skipped") as record:
        apply_outcome(result)
    assert len(record) == 1


def test_preset_and_custom_bases_are_checked_alike_and_kappa_3_never_decides():
    for basis in (preset_basis(1), tiny_basis()):
        plain = _check(_Source(alpha_law(1.05)), basis=basis)
        cubic = _check(_Source(alpha_law(1.05, cubic=10.0)), basis=basis, refresh=True)
        assert plain.outcome == cubic.outcome == "passed"
        assert (plain.cubic_evidence is None) is (basis.preset is None)
    with pytest.raises(BackendError, match="atomic model"):
        _check(None, refresh=True)


def test_result_record_is_json_serialisable():
    result = _check(_Source(alpha_law(1.0)))
    record = json.loads(json.dumps(result.to_record(), allow_nan=False))
    assert record["outcome"] == "passed"
    assert record["protocol_version"] == 1
    assert record["reduction"]["valid"] is True
    assert record["computed"]["execution_space"] == "OpenMP"


# --- item 13: the cache ------------------------------------------------------------


def test_first_check_writes_one_entry_and_the_second_hits(cache):
    source = _Source(alpha_law(1.02))
    first = _check(source)
    assert source.calls == 1 and not first.from_cache
    entries = list(cache.glob("*.json"))
    assert [entry.name for entry in entries] == [f"{first.cache_key}.json"]
    stored = json.loads(entries[0].read_text(encoding="utf-8"))
    assert stored["computed"]["execution_space"] == "OpenMP"
    assert stored["computed"]["threads"] == 1
    assert stored["inputs"]["provider_revision"] == _REVISION
    # Another execution space, build or thread count reuses the entry.
    second = _check(
        source,
        computed_on={"execution_space": "Cuda", "build": {"x": 1}, "threads": 64},
    )
    assert source.calls == 1 and second.from_cache
    assert second.reduction == first.reduction
    assert second.computed == first.computed


@pytest.mark.parametrize(
    "change",
    ["preset", "custom_field", "ions", "revision", "optmat", "protocol", "wavelength"],
)
def test_any_key_input_change_misses(monkeypatch, change):
    source = _Source(alpha_law(1.02))
    basis = preset_basis(1)
    _check(source, basis=basis)
    changes = {}
    if change == "preset":
        basis = dataclasses.replace(basis, preset="hacc-helium-1-v2-0123456789abcdef")
    elif change == "custom_field":
        basis = dataclasses.replace(basis, preset=None, cut_energy=basis.cut_energy * 2)
    elif change == "ions":
        basis = preset_basis(2)
    elif change == "revision":
        changes["provider_revision"] = "0" * 40
    elif change == "wavelength":
        changes["reference_wavelength"] = 1030e-9
    elif change == "optmat":
        original = tdse_consistency._helium_reference

        def other_optmat():
            reference, support, identity = original()
            return reference, support, {**identity, "optmat_version": "0.9.0"}

        monkeypatch.setattr(tdse_consistency, "_helium_reference", other_optmat)
    else:
        monkeypatch.setattr(tdse_consistency, "PROTOCOL_VERSION", 2)
    result = _check(source, basis=basis, **changes)
    assert source.calls == 2 and not result.from_cache


def test_changed_bounds_hit_and_reapply_the_decision():
    source = _Source(alpha_law(1.2))
    assert _check(source).outcome == "warned"
    stricter = _check(source, policy=TDSEConsistency(warn=0.05, refuse=0.1))
    assert stricter.from_cache and stricter.outcome == "refused"
    assert source.calls == 1


@pytest.mark.parametrize("damage", ["truncated", "json", "inputs", "values", "invalid"])
def test_unusable_entries_are_ignored_logged_and_recomputed(cache, caplog, damage):
    source = _Source(alpha_law(1.02))
    first = _check(source)
    path = cache / f"{first.cache_key}.json"
    text = path.read_text(encoding="utf-8")
    entry = json.loads(text)
    if damage == "truncated":
        path.write_text(text[: len(text) // 2], encoding="utf-8")
    elif damage == "json":
        path.write_text("not json", encoding="utf-8")
    else:
        if damage == "inputs":
            entry["inputs"]["basis_identity"] = "another"
        elif damage == "values":
            entry["reduction"]["kappa_real"][0] = "nan"
        else:
            entry["reduction"]["valid"] = False
        path.write_text(json.dumps(entry), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="solprop.driver.tdse_consistency"):
        again = _check(source)
    assert "ignoring" in caplog.text
    assert source.calls == 2 and not again.from_cache
    assert json.loads(path.read_text(encoding="utf-8"))["inputs"] == entry[
        "inputs"
    ] or (damage == "inputs")


def test_refresh_recomputes_and_overwrites():
    source = _Source(alpha_law(1.02))
    _check(source)
    refreshed = _check(source, refresh=True)
    assert source.calls == 2 and not refreshed.from_cache


def test_invalid_results_are_never_cached(cache):
    source = _Source(
        lambda field, spacing: (np.zeros_like(field), np.zeros_like(field))
    )
    assert _check(source).outcome == "invalid"
    assert not list(cache.glob("*.json"))


def test_cache_write_is_atomic(cache, monkeypatch):
    source = _Source(alpha_law(1.02))
    first = _check(source)
    path = cache / f"{first.cache_key}.json"
    before = path.read_text(encoding="utf-8")

    def interrupted(entry, stream, **kwargs):
        stream.write('{"inputs": ')
        raise KeyboardInterrupt

    monkeypatch.setattr(tdse_consistency.json, "dump", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _check(source, refresh=True)
    # The reader never sees the partial file: the entry is unchanged and no
    # temporary file is left behind.
    assert path.read_text(encoding="utf-8") == before
    assert [entry.name for entry in cache.iterdir()] == [path.name]


def test_cache_location_falls_back_from_solprop_to_xdg_to_home(monkeypatch, tmp_path):
    assert cache_directory() == tmp_path / "tdse-consistency"
    monkeypatch.delenv("SOLPROP_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_directory() == tmp_path / "xdg" / "solprop" / "tdse-consistency"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))  # Windows' home
    assert cache_directory() == (
        tmp_path / "home" / ".cache" / "solprop" / "tdse-consistency"
    )


# --- the binding's diagnostic and the standalone check -----------------------------


def test_binding_diagnostic_is_one_charged_one_row_call(monkeypatch):
    fake = install(monkeypatch)
    fake.response = alpha_law(1.0)
    grid = FieldGrid.from_limits(1e-3, 800e-9, (400e-9, 1600e-9), 10e-15)
    descriptor = TDSEDescriptor.from_response(hacc_response(preset_basis(1)), grid)
    binding: Any = tdse.prepare_binding(
        descriptor, TDSEExecution(max_atomic_calls=5), 2
    )
    spacing, field = protocol_field(_LAMBDA)
    total, first = binding.diagnostic_currents(field[None, :], spacing)
    assert total.shape == first.shape == field.shape
    assert fake.plan_arguments[-1]["rows"] == 1
    assert fake.plan_arguments[-1]["samples"] == field.size
    assert fake.events[-1] == f"close-plan-1-{field.size}"
    ledger = binding.ledger
    assert (ledger.attempted, ledger.diagnostic_calls, ledger.reserved) == (1, 1, 0)
    fake.failure = "raise_driven"
    with pytest.raises(BackendError, match=r"diagnostic failed.*\[native\]"):
        binding.diagnostic_currents(field[None, :], spacing)
    assert binding.ledger.failed_calls == 1 and binding.ledger.attempted == 2
    with pytest.raises(BackendError, match="float64"):
        binding.diagnostic_currents(field[None, :].astype(np.float32), spacing)
    assert binding.ledger.attempted == 2
    fake.failure = None
    fake.plan_bytes = 10**12
    with pytest.raises(MemoryBudgetError):
        tdse.prepare_binding(descriptor, TDSEExecution(memory_limit=10**9), 2)
    binding.close()


def test_diagnostic_memory_is_admitted_beside_the_propagation_plans(monkeypatch):
    fake = install(monkeypatch)
    grid = FieldGrid.from_limits(1e-3, 800e-9, (400e-9, 1600e-9), 10e-15)
    descriptor = TDSEDescriptor.from_response(hacc_response(preset_basis(1)), grid)
    estimate = sum(
        tdse.estimate_binding_memory(descriptor, TDSEExecution(), 1).values()
    )
    limit = estimate + fake.model_bytes + fake.plan_bytes + 10
    binding: Any = tdse.prepare_binding(
        descriptor, TDSEExecution(memory_limit=limit), 1
    )
    spacing, field = protocol_field(_LAMBDA)
    with pytest.raises(MemoryBudgetError):
        binding.diagnostic_currents(field[None, :], spacing)
    assert binding.ledger.attempted == 0
    binding.close()


def test_standalone_check_uses_the_cache_and_returns_its_ledger(monkeypatch, cache):
    fake = install(monkeypatch)
    fake.response = alpha_law(1.01)
    response = hacc_response(preset_basis(1))
    first = check_consistency(
        response, TDSEExecution(max_atomic_calls=1), reference_wavelength=_LAMBDA
    )
    assert first.outcome == "passed" and not first.from_cache
    assert first.ledger is not None and first.ledger.diagnostic_calls == 1
    assert fake.events[-1] == "close-model"
    again = check_consistency(response, TDSEExecution(), reference_wavelength=_LAMBDA)
    assert again.from_cache and again.ledger is not None
    assert again.ledger.attempted == 0
    skipped = check_consistency(
        hacc_response(preset_basis(1), consistency=TDSEConsistency(enforce=False)),
        TDSEExecution(),
        reference_wavelength=_LAMBDA,
    )
    assert skipped.outcome == "skipped" and skipped.ledger is None
    refreshed = check_consistency(
        response, TDSEExecution(), reference_wavelength=_LAMBDA, refresh=True
    )
    assert not refreshed.from_cache


def test_standalone_check_refuses_sae_and_reports_provider_failures(monkeypatch):
    fake = install(monkeypatch)
    sae_basis = TDSEBasis("sae", (0.0, 1e-9), (4,), 3, 1e-17, source="sae control")
    sae = TDSEGasResponse(
        "helium",
        sae_basis,
        1e-19,
        "tdse_native",
        1.0,
        TDSECalibration(
            "sae",
            "research",
            sae_basis.identity,
            (100e-9, 2e-6),
            source="test",
            limitations=("test",),
        ),
        TDSESampling(),
    )
    with pytest.raises(BackendError, match="haCC responses only"):
        check_consistency(sae, TDSEExecution(), reference_wavelength=_LAMBDA)
    response = hacc_response(preset_basis(1))
    fake.failure = "plan_driven"
    with pytest.raises(BackendError, match=r"diagnostic preparation failed.*native"):
        check_consistency(response, TDSEExecution(), reference_wavelength=_LAMBDA)
    fake.failure = None
    fake.preset_changes = {"identity": "hacc-helium-1-v2-0123456789abcdef"}
    with pytest.raises(BackendError, match="expected"):
        check_consistency(response, TDSEExecution(), reference_wavelength=_LAMBDA)
    fake.preset_changes = {}
    fake.model_bytes = 10**9
    with pytest.raises(MemoryBudgetError):
        check_consistency(
            response, TDSEExecution(memory_limit=10**6), reference_wavelength=_LAMBDA
        )


# --- cached entries are validated explicitly, also under python -O --------------


def _valid_record():
    spacing, field = protocol_field(_LAMBDA)
    first, _ = alpha_law(1.02)(field[None, :], spacing)
    reduction = reduce_linear(spacing, field, first[0], _reference, _SUPPORT)
    return json.loads(json.dumps(dataclasses.asdict(reduction)))


_CHANGES = {
    "missing_key": lambda r: r.pop("kappa_imag"),
    "short_real": lambda r: r.update(kappa_real=r["kappa_real"][:-1]),
    "short_imag": lambda r: r.update(kappa_imag=r["kappa_imag"][1:]),
    "empty": lambda r: r.update(wavelength=[], kappa_real=[], kappa_imag=[]),
    "string_value": lambda r: r["kappa_real"].__setitem__(0, "1.02"),
    "integer_value": lambda r: r["kappa_imag"].__setitem__(0, 0),
    "invalid_flag": lambda r: r.update(valid=False),
    "reason": lambda r: r.update(reason="was invalid"),
    "deviation": lambda r: r.update(deviation=r["deviation"] / 2),
    "worst_wavelength": lambda r: r.update(worst_wavelength=r["wavelength"][0] * 1.5),
    "worst_kappa": lambda r: r.update(worst_kappa=[1.0, 0.0]),
    "worst_kappa_length": lambda r: r.update(worst_kappa=[1.0]),
    "worst_kappa_none": lambda r: r.update(worst_kappa=None),
    "deviation_none": lambda r: r.update(deviation=None),
}


def _damaged(change):
    record = _valid_record()
    if change == "not_a_dict":
        return [record]
    _CHANGES[change](record)
    return record


_DAMAGES = [
    "not_a_dict",
    "missing_key",
    "short_real",
    "short_imag",
    "empty",
    "string_value",
    "integer_value",
    "invalid_flag",
    "reason",
    "deviation",
    "worst_wavelength",
    "worst_kappa",
    "worst_kappa_length",
    "worst_kappa_none",
    "deviation_none",
]


def test_a_well_formed_cached_reduction_is_admitted():
    reduction = tdse_consistency._reduction_from(_valid_record())
    assert reduction is not None and reduction.valid


@pytest.mark.parametrize("change", _DAMAGES)
def test_malformed_cached_reductions_are_refused(change):
    assert tdse_consistency._reduction_from(_damaged(change)) is None


@pytest.mark.parametrize("change", ["deviation", "short_real", "computed"])
def test_malformed_cache_entries_are_ignored_logged_and_recomputed(
    cache, caplog, change
):
    source = _Source(alpha_law(1.02))
    first = _check(source)
    path = cache / f"{first.cache_key}.json"
    entry = json.loads(path.read_text(encoding="utf-8"))
    if change == "computed":
        entry["computed"] = "not a record"
    else:
        entry["reduction"] = _damaged(change)
    path.write_text(json.dumps(entry), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="solprop.driver.tdse_consistency"):
        again = _check(source)
    assert "invalid values" in caplog.text
    assert source.calls == 2 and not again.from_cache


def test_cached_entries_are_validated_without_assertions_under_optimisation(
    tmp_path,
):
    # Under ``python -O`` assertions vanish; the explicit checks must still refuse.
    records = {change: _damaged(change) for change in _DAMAGES}
    path = tmp_path / "records.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    code = (
        "import json, sys; "
        "from solprop.driver.tdse_consistency import _reduction_from; "
        "records = json.load(open(sys.argv[1], encoding='utf-8')); "
        "print(json.dumps({k: _reduction_from(v) is None for k, v in records.items()}))"
    )
    completed = subprocess.run(
        [sys.executable, "-O", "-c", code, str(path)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    refused = json.loads(completed.stdout.strip().splitlines()[-1])
    assert refused == dict.fromkeys(_DAMAGES, True)


_SAE_BASIS = TDSEBasis("sae", (0.0, 1e-9), (4,), 3, 1e-17, source="sae")


def test_explicit_guards_replace_the_remaining_assertions():
    with pytest.raises(BackendError, match="no deviation"):
        tdse_consistency._deviation(None)
    with pytest.raises(BackendError, match="haCC responses only"):
        tdse_consistency.gate_run(
            _Source(alpha_law(1.0)),
            TDSEGasResponse(
                "helium",
                _SAE_BASIS,
                1e-19,
                "tdse_native",
                1.0,
                TDSECalibration(
                    "sae",
                    "research",
                    TDSEBasis(
                        "sae", (0.0, 1e-9), (4,), 3, 1e-17, source="sae"
                    ).identity,
                    (100e-9, 2e-6),
                    source="test",
                    limitations=("test",),
                ),
                TDSESampling(),
            ),
            reference_wavelength=_LAMBDA,
            provider_revision=_REVISION,
            computed_on={},
        )
    result = _check(_Source(alpha_law(1.2)))
    assert result.reduction is not None
    incomplete = dataclasses.replace(
        result,
        reduction=dataclasses.replace(result.reduction, worst_kappa=None),
    )
    assert "invalid" in tdse_consistency.describe(incomplete)
