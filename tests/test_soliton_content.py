# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scattering content and constituent phase, independent of propagation backends."""

import math
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest
from jaxtyping import TypeCheckError
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from solprop import tools
from solprop.constants import C
from solprop.exceptions import ModeError, ToolsError
from solprop.modes import MarcatiliMode, TubeLattice, VincettiMode
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.physdata import n2
from solprop.physdata.selection import MaterialSelection

PUMP = 800e-9
OMEGA = 2 * math.pi * C / PUMP


def content(order, shape: tools.PulseShape = "sech"):
    # Unit natural width, |beta2| and gamma make the input power N^2.
    fwhm = 1 / tools.pulse_width(1.0, shape)
    return tools.soliton_content(order**2, fwhm, 1.0, -1.0, shape)


@pytest.mark.parametrize("order", [0.7, 1.0, 3.0, 10.0])
def test_sech_constituent_is_the_analytic_scattering_solution(order):
    result = content(order)
    assert result.count == math.ceil(order - 0.5)
    # Closed forms use only a handful of operations; allow ten ulps.
    assert result.largest_eigenvalue == pytest.approx(order - 0.5, rel=2e-15)
    assert result.largest_peak_power == pytest.approx((2 * order - 1) ** 2, rel=2e-15)
    assert result.largest_width == pytest.approx(1 / (2 * order - 1), rel=2e-15)
    assert result.largest_nonlinear_wavenumber == pytest.approx(
        (2 * order - 1) ** 2 / 2, rel=2e-15
    )
    with pytest.raises(FrozenInstanceError):
        setattr(result, "count", 0)  # noqa: B010 - exercise frozen dataclass refusal


@pytest.mark.parametrize("shape", ["sech", "gauss"])
@pytest.mark.parametrize("threshold", [0.5, 1.5, 2.5])
@pytest.mark.parametrize("offset", [-1e-6, 0.0, 1e-6])
def test_count_excludes_the_zero_eigenvalue_at_threshold(shape, threshold, offset):
    factor = 1.0 if shape == "sech" else math.sqrt(2 / math.pi)
    result = content((threshold + offset) / factor, shape)
    assert result.count == int(threshold - 0.5) + (offset > 0)
    if result.count == 0:
        assert result.largest_eigenvalue is None
        assert result.largest_peak_power is None
        assert result.largest_width is None
        assert result.largest_nonlinear_wavenumber is None


def scattering_residual(eta, order, extent, tolerance):
    """Match independently integrated left/right Jost amplitude vectors."""

    def rhs(x, vector):
        q = order * math.exp(-x * x / 2)
        return [eta * vector[0] + q * vector[1], -q * vector[0] - eta * vector[1]]

    vectors = []
    for boundary, initial in ((-extent, [1.0, 0.0]), (extent, [0.0, 1.0])):
        solution = solve_ivp(
            rhs,
            (boundary, 0),
            initial,
            method="DOP853",
            rtol=tolerance,
            atol=tolerance * 0.01,
        )
        assert solution.success
        vector = solution.y[:, -1]
        vectors.append(vector / np.linalg.norm(vector))
    return np.linalg.det(np.stack(vectors))


@pytest.mark.parametrize("order", [0.7, 1.0, 3.0])
def test_gaussian_eigenvalue_converges_against_two_sided_amplitude_matching(order):
    eta = content(order, "gauss").largest_eigenvalue
    assert eta is not None
    references = [
        brentq(
            lambda trial, extent=extent, tol=tol: scattering_residual(
                trial, order, extent, tol
            ),
            eta * 0.9,
            eta * 1.1,
            xtol=1e-13,
        )
        for extent, tol in ((10, 1e-8), (12, 1e-10), (14, 1e-12))
    ]
    # Independent amplitude integration at 1e-12 agrees with the phase equation's
    # 1e-11 local tolerance, allowing accumulated integration and root error.
    assert eta == pytest.approx(references[-1], abs=1e-10)
    assert abs(references[1] - references[2]) < abs(references[0] - references[2])
    assert abs(scattering_residual(eta, order, 14, 1e-12)) < 1e-9


@pytest.mark.parametrize("shape", ["sech", "gauss"])
def test_constituent_scaling_preserves_the_fundamental_soliton_identity(shape):
    first = tools.soliton_content(4e9, 30e-15, 2e-10, -1e-29, shape)
    second = tools.soliton_content(8e9, 60e-15, 1e-10, -4e-29, shape)
    assert first.largest_peak_power is not None and first.largest_width is not None
    assert second.largest_peak_power is not None and second.largest_width is not None
    # Same dimensionless scattering problem, so scaling introduces only rounding.
    assert second.largest_peak_power == pytest.approx(
        2 * first.largest_peak_power, rel=1e-12
    )
    assert second.largest_width == pytest.approx(2 * first.largest_width, rel=1e-12)
    assert first.largest_peak_power * first.largest_width**2 == pytest.approx(
        1e-29 / 2e-10, rel=1e-12
    )


@pytest.mark.parametrize("dispersion", [0.0, 1.0, math.nan, math.inf])
def test_content_requires_anomalous_finite_dispersion(dispersion):
    with pytest.raises(ToolsError, match="anomalous"):
        tools.soliton_content(1.0, 1.0, 1.0, dispersion)


@pytest.mark.parametrize("shape", ["sech", "gauss"])
def test_default_rdw_phase_uses_the_largest_constituent_in_the_soliton_mode(shape):
    soliton = SpecifiedMode.from_taylor([0, 0, -1e-26, 5e-41], OMEGA)
    wave = SpecifiedMode.from_taylor([0, 0, -2e-26, 5e-41], OMEGA)
    power, gamma, fwhm = 2e4, 1e-3, 100e-15
    expected = tools.soliton_content(power, fwhm, gamma, -1e-26, shape)
    assert expected.largest_nonlinear_wavenumber is not None
    result = tools.phase_mismatch(
        wave,
        np.array([OMEGA]),
        PUMP,
        soliton_mode=soliton,
        peak_power=power,
        gamma=gamma,
        fwhm=fwhm,
        shape=shape,
    )
    # Taylor-mode beta2 comes from a Richardson difference, with ~1e-8 accuracy.
    assert result[0] == pytest.approx(-expected.largest_nonlinear_wavenumber, rel=1e-8)
    new = tools.lambda_rdw(
        soliton, PUMP, peak_power=power, gamma=gamma, fwhm=fwhm, shape=shape
    )
    supplied = tools.lambda_rdw(
        soliton,
        PUMP,
        peak_power=expected.largest_peak_power,
        gamma=gamma,
        nonlinear_phase="supplied_power",
    )
    assert new == pytest.approx(supplied, rel=1e-8)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"peak_power": 1, "gamma": 1}, "needs fwhm"),
        ({"fwhm": 1}, "fwhm requires"),
        ({"gamma": 1}, "both peak_power and gamma"),
        (
            {"peak_power": 1, "gamma": 1, "nonlinear_phase": "linear"},
            "requires omitting",
        ),
        (
            {
                "peak_power": 1,
                "gamma": 1,
                "fwhm": 1,
                "nonlinear_phase": "supplied_power",
            },
            "does not use",
        ),
        ({"nonlinear_phase": "wrong"}, "nonlinear_phase must"),
        ({"shape": "wrong"}, "shape must"),
        ({"peak_power": 1, "gamma": 1, "fwhm": -1}, "fwhm must"),
    ],
)
def test_phase_declarations_fail_before_search(kwargs, message):
    mode = MarcatiliMode(125e-6, "He", 1.0, cladding=1.45)
    if kwargs.get("shape") == "wrong" or kwargs.get("nonlinear_phase") == "wrong":
        with pytest.raises(TypeCheckError):
            tools.lambda_rdw(mode, PUMP, **kwargs)
    else:
        with pytest.raises(ToolsError, match=message):
            tools.lambda_rdw(mode, PUMP, **kwargs)


def test_default_refuses_a_pulse_without_a_discrete_soliton():
    mode = SpecifiedMode.from_taylor([0, 0, -1e-26], OMEGA)
    with pytest.raises(ToolsError, match="no discrete soliton"):
        tools.lambda_rdw(mode, PUMP, peak_power=1, gamma=1e-3, fwhm=1e-15)
    assert (
        tools.phase_mismatch(mode, np.array([OMEGA]), PUMP, nonlinear_phase="linear")[0]
        == 0.0
    )


def test_the_inner_diameter_ratio_uses_the_inner_resonator_diameter():
    lattice = TubeLattice.from_inner_diameter_ratio(20e-6, 0.3e-6, 0.6, 6)
    # Geometry is just a product and sum: rounding only.
    assert lattice.tube_radius == pytest.approx(12.3e-6, rel=1e-15)
    assert lattice.n_tubes == 6
    assert lattice.gap(20e-6) > 0


@pytest.mark.parametrize(
    "radius, wall, ratio, count",
    [
        (-1, 1e-7, 0.6, 6),
        (20e-6, -1e-7, 0.6, 6),
        (20e-6, 1e-7, 0, 6),
        (20e-6, 1e-7, 1.2, 6),
        (20e-6, 1e-7, 0.6, 2),
        (20e-6, 1e-7, math.nan, 6),
    ],
)
def test_invalid_diameter_geometry_is_rejected(radius, wall, ratio, count):
    with pytest.raises(ModeError):
        TubeLattice.from_inner_diameter_ratio(radius, wall, ratio, count)


@pytest.mark.parametrize("shape", ["sech", "gauss"])
@pytest.mark.parametrize("vincetti", [False, True])
def test_pressure_search_inverts_constituent_phase_for_both_modes(shape, vincetti):
    radius, pressure = (20e-6 if vincetti else 125e-6), 2.0
    lattice = (
        TubeLattice.from_inner_diameter_ratio(radius, 0.3e-6, 0.6, 6)
        if vincetti
        else None
    )
    mode = (
        MarcatiliMode(radius, "He", pressure, cladding=1.45)
        if lattice is None
        else VincettiMode(radius, "He", pressure, cladding=1.45, lattice=lattice)
    )
    # The nonlinear index the search takes: optmat's helium at the pump since SEP 3's
    # Phase 4a, as the mode's own fill.
    index = n2("He", pressure=pressure, wavelength=PUMP, selection=MaterialSelection())
    gamma = tools.nonlinear_coefficient(mode, index, PUMP)
    power, fwhm = 1e9, 30e-15
    target = tools.lambda_rdw(
        mode,
        PUMP,
        peak_power=power,
        gamma=gamma,
        fwhm=fwhm,
        shape=shape,
        wavelength_limits=(150e-9, 700e-9),
    )
    found = tools.pressure_rdw(
        radius,
        "He",
        target,
        PUMP,
        peak_power=power,
        fwhm=fwhm,
        shape=shape,
        lattice=lattice,
        pressure_limits=(1.5, 2.5),
        cladding=1.45,
    )
    # A numerical beta2 feeds the constituent; tolerate its ~1e-8 relative noise.
    assert found == pytest.approx(pressure, rel=1e-6)


def test_pressure_search_rejects_ambiguous_or_inadmissible_roots():
    with pytest.raises(ToolsError, match=r"2 roots, at 2, 3 bar"):
        tools._rdw_pressure_root(lambda p: (p - 2) * (p - 3), (1, 4))
    with pytest.raises(ToolsError, match="no pressure"):
        tools._rdw_pressure_root(lambda p: None if 2 < p < 3 else p - 2.5, (1, 4))
    with pytest.raises(ToolsError, match="not finite"):
        tools._rdw_pressure_root(lambda p: math.nan, (1, 4))
    with pytest.raises(ToolsError, match="ordered"):
        tools._rdw_pressure_root(lambda p: p, (4, 1))


def test_params_report_constituents_and_normal_dispersion_unavailability():
    for dispersion in (-1e-26, 1e-26):
        mode = SpecifiedMode.from_taylor([0, 0, dispersion], OMEGA)
        params = tools.soliton_params(
            mode, "He", 1.0, energy=1e-6, fwhm=30e-15, wavelength=PUMP
        )
        assert (params.soliton_content is None) == (dispersion > 0)
        assert "CONSTITUENTS" in str(params)
        if dispersion > 0:
            assert "unavailable (normal dispersion)" in str(params)


def test_wavelength_search_computes_gaussian_scattering_only_once(monkeypatch):
    original = tools._gaussian_eigenvalue
    calls = []

    def counted(order):
        calls.append(order)
        return original(order)

    monkeypatch.setattr(tools, "_gaussian_eigenvalue", counted)
    mode = SpecifiedMode.from_taylor([0, 0, -1e-26, 5e-41], OMEGA)
    tools.lambda_rdw(
        mode, PUMP, peak_power=2e4, gamma=1e-3, fwhm=100e-15, shape="gauss"
    )
    assert len(calls) == 1


@pytest.mark.parametrize("limits", [(2, 1), (0, 1), (1, 101), (1, math.nan)])
def test_pressure_limits_reject_invalid_ranges(limits):
    with pytest.raises(ToolsError, match="pressure_limits"):
        tools.pressure_rdw(
            125e-6, "He", 200e-9, PUMP, pressure_limits=limits, cladding=1.45
        )


def test_pressure_constituent_search_rejects_subthreshold_pulses():
    with pytest.raises(ToolsError, match="no pressure"):
        tools.pressure_rdw(
            125e-6,
            "He",
            200e-9,
            PUMP,
            peak_power=1,
            fwhm=1e-15,
            pressure_limits=(0.1, 0.2),
            cladding=1.45,
        )


def test_failed_scattering_integration_is_reported(monkeypatch):
    monkeypatch.setattr(
        tools,
        "solve_ivp",
        lambda *args, **kwargs: SimpleNamespace(
            success=False, message="test integration failure"
        ),
    )
    with pytest.raises(ToolsError, match="scattering integration failed"):
        content(1, "gauss")


def test_failed_eigenvalue_bracket_is_reported(monkeypatch):
    def failed(*args, **kwargs):
        raise ValueError("test bracket failure")

    monkeypatch.setattr(tools, "brentq", failed)
    with pytest.raises(ToolsError, match="eigenvalue refinement failed"):
        content(1, "gauss")


def test_unrepresentable_area_is_rejected():
    with pytest.raises(ToolsError, match="pulse area must be finite"):
        tools.soliton_content(1.0, 1e100, 1.0, -1e-200)


def test_unresolved_positive_eigenvalue_is_rejected():
    # This is above creation but below the absolute eigenvalue solver tolerance.
    with pytest.raises(ToolsError, match="too close to threshold"):
        content(math.sqrt(math.pi / 8) * (1 + 1e-14), "gauss")


def test_pressure_refinement_refuses_an_unresolved_no_soliton_gap():
    root = 2.123456
    with pytest.raises(ToolsError, match="crossed a no-soliton region"):
        tools._rdw_pressure_root(
            lambda p: None if abs(p - root) < 1e-5 else p - root, (1.0, 4.0)
        )


def test_pressure_search_requires_a_nonempty_bracket():
    with pytest.raises(ToolsError, match="pressure_max must exceed"):
        tools.pressure_rdw(125e-6, "He", 200e-9, PUMP, pressure_max=1e-7)
    with pytest.raises(ToolsError, match="no pressure between"):
        tools._pressure_root(lambda p: 1.0, 100.0, "an unreachable target")


@pytest.mark.parametrize(
    "dispersion, message",
    [(math.nan, "dispersion is not finite"), (1e-26, "no pressure")],
)
def test_pressure_search_rejects_invalid_pump_dispersion(
    monkeypatch, dispersion, message
):
    monkeypatch.setattr(tools.base, "beta2", lambda *args: np.array([dispersion]))
    with pytest.raises(ToolsError, match=message):
        tools.pressure_rdw(
            125e-6, "He", 200e-9, PUMP, peak_power=1e9, fwhm=30e-15, cladding=1.45
        )


def test_the_phase_keyword_is_checked_without_the_type_hook():
    # The type hook refuses a wrong literal at the public door; production has none.
    with pytest.raises(ToolsError, match="nonlinear_phase must"):
        tools._check_phase("wrong")
