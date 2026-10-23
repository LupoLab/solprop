# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Steady gas flow: the choked end, its independent witness, and the flow regime."""

import dataclasses
import logging
import math
import warnings

import numpy as np
import pytest
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from solprop.constants import BAR, K_B, N_A
from solprop.exceptions import GasFlowWarning, MaterialError, TranslationError
from solprop.gasflow import (
    KNUDSEN_SLIP_LIMIT,
    KNUDSEN_TRANSITION_LIMIT,
    REYNOLDS_LAMINAR_LIMIT,
    FlowReport,
    _check_regime,
    _checked_choice,
    chamber_pressure,
    choked_pressure,
    flow_report,
    knudsen_number,
    mass_flow,
    mbar_litres_per_second,
    reynolds_number,
    standard_litres_per_minute,
    steady_fill,
    throughput,
)
from solprop.physdata import Gas, IdealGas, Mixture, Species, gamma
from solprop.physdata.transport import (
    heat_capacity_ratio,
    molar_mass,
    sound_speed,
    viscosity,
)
from solprop.profiles import (
    FromCallable,
    Linear,
    PiecewiseLinear,
    PressureGradient,
    hydraulic_integral,
    profile_identity,
)

GAS_CONSTANT = K_B * N_A
# solprop's default temperature, at which every call below without one is evaluated.
ROOM = 293.15

# Gebhardt et al., Opt. Express 29, 6957 (2021), Fig. 6(c): 500 um bore, 5 m, N2.
GEBHARDT_RADIUS = 250e-6
GEBHARDT_LENGTH = 5.0


def _factor(gas, radius, length, *, exit_radius=None):
    """k = 1 / (16 eta c a_e^2 R) of the module docstring, in Pa^-1."""
    resistance = hydraulic_integral(radius, [0.0, length])[0]
    exit_radius = radius if exit_radius is None else exit_radius
    eta, c = viscosity(gas, ROOM), sound_speed(gas, ROOM)
    return 1.0 / (16.0 * eta * c * exit_radius**2 * resistance)


# ------------------------------------------------------------- the choked quadratic


@pytest.mark.parametrize(
    ("gas", "radius", "length", "high"),
    [("N2", 250e-6, 5.0, 4.2), ("Ar", 15e-6, 1.0, 20.0), ("He", 125e-6, 3.0, 5.0)],
)
def test_the_choked_pressure_is_the_exact_root(gas, radius, length, high):
    k = _factor(gas, radius, length)
    high_pa = high * BAR
    root = brentq(
        lambda p: k * (high_pa**2 - p**2) - p, 0.0, high_pa, xtol=1e-300, rtol=1e-15
    )
    # The closed form and a bracketing root solve of the same quadratic: both are
    # correctly rounded to a few ulps.
    got = choked_pressure(gas, radius, length, high)
    assert got * BAR == pytest.approx(root, rel=1e-14)


@pytest.mark.parametrize(
    ("gas", "radius", "length", "high"),
    [("N2", 250e-6, 5.0, 4.2), ("Ar", 15e-6, 1.0, 20.0), ("He", 125e-6, 3.0, 5.0)],
)
def test_the_choked_pressure_reduces_to_hisols_small_exit_limit(
    gas, radius, length, high
):
    # HiSol.jl's choked_pressure: D^2 p0^2 / (64 eta L c), which drops p_e^2 from
    # k (p0^2 - p_e^2) = p_e. The exact root is k p0^2 (1 - (p_e/p0)^2 + ...), so the
    # relative difference is (p_e/p0)^2 to leading order.
    eta, c = viscosity(gas, ROOM), sound_speed(gas, ROOM)
    hisol = (2 * radius) ** 2 * (high * BAR) ** 2 / (64 * eta * length * c)
    got = choked_pressure(gas, radius, length, high) * BAR
    ratio = got / (high * BAR)
    assert 1.0 - got / hisol == pytest.approx(ratio**2, rel=0.1)


def test_the_worked_choked_pressures():
    # Independent numbers (pyfluids at 294 K): 223 mbar, 79 mbar and 41 mbar. They
    # were computed at 294 K, so solprop is asked for that temperature.
    def at_294(*args):
        return choked_pressure(*args, temperature=294.0)

    assert at_294("N2", 250e-6, 5.0, 4.2) == pytest.approx(0.223, abs=1e-3)
    assert at_294("Ar", 15e-6, 1.0, 20.0) == pytest.approx(0.079, abs=1e-3)
    assert at_294("He", 125e-6, 3.0, 5.0) == pytest.approx(0.041, abs=1e-3)


def test_the_exit_end_of_a_taper_sets_the_sonic_throughput():
    taper = Linear(125e-6, 75e-6, 1.0)
    at_end = choked_pressure("Ar", taper, 1.0, 5.0)
    at_start = choked_pressure("Ar", taper, 1.0, 5.0, exit_end="start")
    # p_e is about k p_h^2 with k proportional to 1 / a_e^2.
    assert at_end / at_start == pytest.approx((125.0 / 75.0) ** 2, rel=0.05)
    # The literal is enforced by the annotation under the test suite's type hook; the
    # check a caller without a type checker meets is the module's own.
    with pytest.raises(TranslationError, match="exit_end must be one of"):
        _checked_choice("middle", "exit_end", ("start", "end"))


# --------------------------------------------------------- the independent witness


def _witness(gas, high_bar, length, radius, slope, *, adiabatic):
    """Integrate the 1-D steady compressible laminar flow until it chokes at L.

    Continuity rho u A = mdot, momentum rho u du = -dp - (8 eta u / a^2) dz with the
    acceleration term, the ideal-gas law, and either an isothermal wall (T = T0) or an
    adiabatic one (c_p T + u^2/2 = c_p T0). Eliminating rho and p gives

        du/dz (1 - M^2) = u (8 eta u / (a^2 p) - A'/A),

    with M the isothermal Mach number u / sqrt(R_s T0) on an isothermal wall and the
    adiabatic one u / sqrt(gamma R_s T) on an adiabatic wall; the flow chokes where
    M = 1. Integrated in u, z(u) is regular up to the choke. The mass flow is the one
    whose choke falls exactly at z = L; the exit pressure follows from it.
    """
    specific = GAS_CONSTANT / molar_mass(gas)
    ratio = heat_capacity_ratio(gas, ROOM)
    heat = ratio * specific / (ratio - 1.0)
    eta = viscosity(gas, ROOM)

    def a(z):
        return radius + slope * z

    def temperature(u):
        return ROOM - u**2 / (2.0 * heat) if adiabatic else ROOM

    choke = (
        math.sqrt(2.0 * ratio * specific * ROOM / (ratio + 1.0))
        if adiabatic
        else math.sqrt(specific * ROOM)
    )

    def mach2(u):
        t = temperature(u)
        return u**2 / (ratio * specific * t) if adiabatic else u**2 / (specific * t)

    def reach(mdot):
        start = mdot * specific * ROOM / (high_bar * BAR * math.pi * radius**2)

        def rhs(u, y):
            z = y[0]
            area = math.pi * a(z) ** 2
            p = mdot * specific * temperature(u) / (u * area)
            friction = 8.0 * eta * u / (a(z) ** 2 * p)
            return [(1.0 - mach2(u)) / (u * (friction - 2.0 * slope / a(z)))]

        solution = solve_ivp(
            rhs, (start, choke * (1 - 1e-12)), [0.0], method="DOP853", rtol=1e-10,
            atol=1e-13 * length,
        )  # fmt: skip
        return solution.y[0, -1] - length

    guess = mass_flow(gas, throughput(gas, Linear(radius, radius + slope * length,
                      length), length, high_bar, 0.0), 0.0)  # fmt: skip
    mdot = brentq(reach, 0.2 * guess, 1.5 * guess, xtol=1e-30, rtol=1e-12)
    exit_area = math.pi * a(length) ** 2
    t_exit = temperature(choke)
    return mdot * specific * t_exit / (choke * exit_area) / BAR, mdot


WITNESS_CASES = [
    # Gebhardt's uniform capillary at 4.2 bar, and an argon down-taper at 5 bar.
    ("N2", 4.2, GEBHARDT_LENGTH, GEBHARDT_RADIUS, 0.0),
    ("Ar", 5.0, 1.0, 125e-6, -50e-6),
]


@pytest.mark.medium
@pytest.mark.parametrize(("gas", "high", "length", "radius", "slope"), WITNESS_CASES)
def test_the_choked_end_agrees_with_an_independent_compressible_flow(
    gas, high, length, radius, slope
):
    bore = Linear(radius, radius + slope * length, length)
    closed = choked_pressure(gas, bore, length, high)
    poiseuille = mass_flow(gas, throughput(gas, bore, length, high, closed), closed)
    isothermal, mdot_isothermal = _witness(
        gas, high, length, radius, slope, adiabatic=False
    )
    adiabatic, mdot_adiabatic = _witness(
        gas, high, length, radius, slope, adiabatic=True
    )
    root_gamma = math.sqrt(heat_capacity_ratio(gas, ROOM))
    # Recorded agreement: the Poiseuille mass flow is within 2.2 % (N2) and
    # 3.4 % (Ar taper) of the compressible one -- the acceleration term matters only
    # near the exit -- so 4 % bounds both walls.
    assert mdot_isothermal == pytest.approx(poiseuille, rel=0.04)
    assert mdot_adiabatic == pytest.approx(poiseuille, rel=0.04)
    # The exit pressure is set by where the flow chokes. An isothermal wall chokes at
    # the isothermal sound speed c / sqrt(gamma), and the closed form with that speed
    # matches it to 2.1 % (N2) and 3.1 % (Ar taper): 4 %.
    assert isothermal == pytest.approx(closed * root_gamma, rel=0.04)
    # The adopted closed form, with the adiabatic speed at the fill temperature, lies
    # between the two walls: the isothermal witness is 15.7 % and 24.8 % above it, the
    # adiabatic one, which chokes where the gas has cooled, 10.5 % and 15.7 % below.
    assert adiabatic < closed < isothermal
    assert 0.80 < adiabatic / closed < 0.92
    assert 1.10 < isothermal / closed < 1.30


# ------------------------------------------------------------------ the flow regime


def _peak_reynolds(high):
    """The peak Reynolds number of Gebhardt's capillary, choked into vacuum."""
    fill = PressureGradient.from_stages(
        [0.0, GEBHARDT_LENGTH],
        [high, choked_pressure("N2", GEBHARDT_RADIUS, GEBHARDT_LENGTH, high)],
    )
    return flow_report(fill, "N2", radius=GEBHARDT_RADIUS).peak_reynolds


def test_gebhardts_capillary_warns_at_4p2_bar_and_not_at_2p9_bar():
    # 2542 with optmat's nitrogen viscosity since SEP 3's Phase 4a (2539 on the
    # registry's: the dilute viscosity is 0.08 % lower), at 294 K; 2561 at 293.15 K,
    # the default since 2026-10 (the density 0.3 % higher, the viscosity 0.2 % lower).
    with pytest.warns(GasFlowWarning, match="Reynolds number of 2561"):
        steady_fill("N2", [0.0, GEBHARDT_LENGTH], [4.2, 0.0], radius=GEBHARDT_RADIUS)
    with warnings.catch_warnings():
        warnings.simplefilter("error", GasFlowWarning)
        steady_fill("N2", [0.0, GEBHARDT_LENGTH], [2.9, 0.0], radius=GEBHARDT_RADIUS)
    # The worked number, Re about 2560 at 4.2 bar.
    assert _peak_reynolds(4.2) == pytest.approx(2560, rel=0.005)


def test_the_laminar_limit_falls_at_gebhardts_threshold_pressure():
    # Their eq. (4), for a negligible exit pressure: 3.75 bar at Re = 2000 for this
    # capillary. Their viscosity is not stated; 5 % covers that and the choked exit.
    threshold = brentq(
        lambda p: _peak_reynolds(p) - REYNOLDS_LAMINAR_LIMIT, 2.0, 6.0, rtol=1e-12
    )
    assert threshold == pytest.approx(3.75, rel=0.05)
    # Eq. (4) with this module's own viscosity and molar mass differs only by the
    # choked exit (p_e^2 / p_1^2 ~ 0.3 %) and the real-gas density at the exit.
    eta, mass = viscosity("N2", ROOM), molar_mass("N2")
    diameter = 2 * GEBHARDT_RADIUS
    own = 8 * math.sqrt(
        GAS_CONSTANT * ROOM * eta**2 / mass * GEBHARDT_LENGTH / diameter**3 * 2000
    )
    assert threshold == pytest.approx(own / BAR, rel=0.005)


def test_the_peak_reynolds_number_of_a_taper_sits_at_its_waist():
    waist = PiecewiseLinear.from_stages(
        [0.0, 0.3, 0.45, 0.55, 0.7, 1.0], [150e-6, 150e-6, 60e-6, 60e-6, 150e-6, 150e-6]
    )
    fill = steady_fill("Ar", [0.0, 1.0], [3.0, 0.0], radius=waist)
    stage = flow_report(fill, "Ar", radius=waist).stages[0]
    assert 0.45 <= stage.peak_reynolds_position <= 0.55
    # Re = 2 mdot / (pi a eta): the waist carries 150/60 times the entrance's.
    entrance = reynolds_number("Ar", stage.mass_flow, waist, np.array([0.0]))[0]
    assert stage.peak_reynolds / entrance == pytest.approx(2.5, rel=1e-12)


def test_a_rarefied_stage_warns_and_a_continuum_one_logs_its_exit(caplog):
    with caplog.at_level(logging.INFO, logger="solprop.gasflow"):
        steady_fill("He", [0.0, 3.0], [5.0, 0.0], radius=125e-6)
    # 0.0188 at 294 K, 0.0186 at 293.15 K.
    assert any("exit Knudsen number 0.0186" in r.message for r in caplog.records)
    # 0.05 bar to 0.03 bar in a 5 um bore: at the mean 0.04 bar the mean free path is
    # 1.6 um, Kn = 0.162, so slip alone moves the throughput by more than 100 %.
    with pytest.warns(GasFlowWarning, match=r"narrowest bore is 0\.162"):
        fill = steady_fill("N2", [0.0, 0.1], [0.05, 0.03], radius=5e-6)
    stage = flow_report(fill, "N2", radius=5e-6).stages[0]
    expected = float(knudsen_number("N2", 0.04, 5e-6))
    assert stage.mean_knudsen == pytest.approx(expected, rel=1e-14)
    assert stage.mean_knudsen > KNUDSEN_SLIP_LIMIT


@pytest.mark.parametrize(
    ("length", "entrance", "micrometres"),
    # The M2 case and Luna's RDW example: helium, 125 um bore, to vacuum. Their choked
    # exits are transitional (Kn 0.36 and 0.11) over 128 um and 5.7 um, a millionth
    # of their gas, so neither warns (the physics review measured 0.13 mm and 6 um).
    # optmat's helium transport since SEP 3's Phase 4a: 6.238 um (6.27 on the
    # registry's); the 1.5 m case moves within its printed digits. At 293.15 K, the
    # default since 2026-10, 127.6 um and 5.716 um (128.6 and 6.238 at 294 K): the
    # length is a difference of squares near the exit, so it moves by 0.8 % and 8 %.
    [(1.5, 0.8, 127.6), (1.0, 1.2, 5.716)],
)
def test_the_helium_gradients_are_transitional_only_at_their_exit(
    length, entrance, micrometres
):
    with warnings.catch_warnings():
        warnings.simplefilter("error", GasFlowWarning)
        fill = steady_fill("He", [0.0, length], [entrance, 0.0], radius=125e-6)
    stage = flow_report(fill, "He", radius=125e-6).stages[0]
    exit_pressure = fill.pressures[-1]
    # Kn > 0.1 where p < p_t = lambda(1 bar) / (0.2 a); p^2 is linear in z on a
    # uniform bore, so the region is L (p_t^2 - p_e^2) / (p_0^2 - p_e^2) long.
    threshold = float(knudsen_number("He", 1.0, 125e-6)) / KNUDSEN_TRANSITION_LIMIT
    expected = (
        length * (threshold**2 - exit_pressure**2) / (entrance**2 - exit_pressure**2)
    )
    assert stage.transitional_length == pytest.approx(expected, rel=1e-9)
    assert stage.transitional_length * 1e6 == pytest.approx(micrometres, rel=1e-3)
    assert 0.0 < stage.transitional_fraction < 1e-5
    assert stage.exit_knudsen > KNUDSEN_TRANSITION_LIMIT
    assert stage.mean_knudsen < KNUDSEN_SLIP_LIMIT


def test_a_stage_with_much_transitional_gas_warns():
    fill = PressureGradient.from_stages([0.0, 1.0], [1.0, 0.5])
    report = flow_report(fill, "He", radius=125e-6)
    stage = dataclasses.replace(report.stages[0], transitional_fraction=0.02)
    with pytest.warns(GasFlowWarning, match="0.02 of its gas"):
        _check_regime(dataclasses.replace(report, stages=(stage,)))


def test_the_knudsen_number_is_the_mean_free_path_over_the_diameter():
    got = knudsen_number("N2", np.array([1.0, 0.5, 0.0]), 50e-6)
    eta, mass = viscosity("N2", ROOM), molar_mass("N2") / N_A
    lam = eta / BAR * math.sqrt(math.pi * K_B * ROOM / (2 * mass))
    np.testing.assert_allclose(got[:2], [lam / 100e-6, 2 * lam / 100e-6], rtol=1e-14)
    assert got[2] == math.inf
    with pytest.raises(TranslationError, match="not negative"):
        knudsen_number("N2", -1.0, 50e-6)
    with pytest.raises(TranslationError, match="positive"):
        knudsen_number("N2", 1.0, 0.0)


def test_a_floored_vacuum_end_does_not_warn():
    # The floor is numerical and rarefied by construction, but over micrometres
    # holding a negligible part of the gas, which the column rule measures.
    with warnings.catch_warnings():
        warnings.simplefilter("error", GasFlowWarning)
        fill = steady_fill(
            "He", [0.0, 3.0], [5.0, 0.0], radius=125e-6, vacuum_end="floor"
        )
    assert fill.pressures[-1] == pytest.approx(5e-4, rel=1e-15)


# ----------------------------------------------------------------- the fill builder


def test_the_floor_route_is_todays_fill():
    # Without a radius the floor route builds the same profile from the same stage
    # values: an explicitly exact no-op route, so the values agree exactly.
    today = PressureGradient.from_stages([0.0, 0.4, 1.0], [0.0, 2.0, 0.0], floor=None)
    built = steady_fill(
        "Ar", [0.0, 0.4, 1.0], [0.0, 2.0, 0.0], radius=100e-6, vacuum_end="floor"
    )
    z = np.linspace(-0.1, 1.1, 97)
    assert np.array_equal(built.pressures, today.pressures)
    assert np.array_equal(built(z), today(z))
    assert np.array_equal(built.derivative(z), today.derivative(z))
    assert built.radius is None and built.floor == today.floor
    true_vacuum = steady_fill(
        "Ar", [0.0, 1.0], [2.0, 0.0], radius=100e-6, vacuum_end="floor", floor=0.0
    )
    assert true_vacuum.pressures[-1] == 0.0


def test_a_zero_end_is_choked_and_logged_at_info(caplog):
    with caplog.at_level(logging.INFO, logger="solprop.gasflow"):
        fill = steady_fill("He", [0.0, 3.0], [5.0, 0.0], radius=125e-6)
    choked = choked_pressure("He", 125e-6, 3.0, 5.0)
    assert fill.pressures[-1] == choked
    assert any(
        record.levelno == logging.INFO and "choked pressure" in record.message
        for record in caplog.records
    )


def test_both_outer_ends_choke_and_the_interior_keeps_its_value():
    fill = steady_fill("He", [0.0, 1.0, 3.0], [0.0, 5.0, 0.0], radius=125e-6)
    assert fill.pressures[1] == 5.0
    assert fill.pressures[0] == pytest.approx(choked_pressure("He", 125e-6, 1.0, 5.0))
    assert fill.pressures[2] == pytest.approx(choked_pressure("He", 125e-6, 2.0, 5.0))
    report = flow_report(fill, "He", radius=125e-6)
    assert report.start_is_choked and report.end_is_choked


def test_a_declared_end_below_the_choke_is_raised_with_a_warning_log(caplog):
    choked = choked_pressure("He", 125e-6, 3.0, 5.0)
    with caplog.at_level(logging.WARNING, logger="solprop.gasflow"):
        fill = steady_fill("He", [0.0, 3.0], [5.0, choked / 2], radius=125e-6)
    assert fill.pressures[-1] == choked
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert f"{choked / 2:.4g} bar" in record.message
    assert f"{choked:.4g} bar" in record.message


def test_a_declared_end_above_the_choke_is_kept(caplog):
    with caplog.at_level(logging.INFO, logger="solprop.gasflow"):
        fill = steady_fill("He", [0.0, 3.0], [5.0, 0.5], radius=125e-6)
    assert fill.pressures[-1] == 0.5
    assert not any("choked" in r.message for r in caplog.records)
    assert not flow_report(fill, "He", radius=125e-6).end_is_choked


def test_an_interior_zero_keeps_the_numerical_floor(caplog):
    with caplog.at_level(logging.WARNING, logger="solprop.profiles"):
        fill = steady_fill("He", [0.0, 1.0, 2.0], [2.0, 0.0, 2.0], radius=125e-6)
    assert fill.pressures[1] == fill.floor == pytest.approx(2e-4)
    assert any("raised from" in record.message for record in caplog.records)


def test_a_taper_fill_follows_the_bore():
    taper = Linear(125e-6, 75e-6, 1.0)
    fill = steady_fill("Ar", [0.0, 1.0], [5.0, 0.0], radius=taper)
    assert fill.radius is taper
    assert fill.pressures[-1] == choked_pressure("Ar", taper, 1.0, 5.0)


def test_a_mixture_fill_uses_the_mixtures_flow_properties():
    fractions = {"N2": 0.7885, "O2": 0.2115}
    fill = steady_fill(fractions, [0.0, 2.0], [1.0, 0.0], radius=100e-6)
    assert fill.pressures[-1] == pytest.approx(
        choked_pressure(fractions, 100e-6, 2.0, 1.0), rel=1e-15
    )
    mixture = Mixture.fill(fractions, fill)
    report = flow_report(fill, mixture, radius=100e-6)
    assert report.viscosity == viscosity(fractions, ROOM)
    assert report.sound_speed == sound_speed(fractions, ROOM)


def test_independent_populations_are_not_one_flow():
    independent = Mixture.partial_pressures({"N2": 0.8, "O2": 0.2})
    fill = PressureGradient.two_point(1.0, 1.0, 0.1)
    with pytest.raises(MaterialError, match="not one flow"):
        flow_report(fill, independent, radius=100e-6)


def test_the_builder_refuses_what_it_cannot_honour():
    with pytest.raises(TranslationError, match="vacuum_end must be one of"):
        _checked_choice("sealed", "vacuum_end", ("choked", "floor"))
    custom = Species("Mystery", gamma(Gas.He), IdealGas())
    with pytest.raises(MaterialError, match="vacuum_end='floor'"):
        steady_fill(custom, [0.0, 1.0], [1.0, 0.0], radius=1e-4)
    with pytest.raises(MaterialError, match="declared for"):
        steady_fill("He", [0.0, 1.0], [1.0, 0.0], radius=1e-4, temperature=100.0)


def test_the_floor_route_builds_without_transport_data(caplog):
    custom = Species("Mystery", gamma(Gas.He), IdealGas())
    with caplog.at_level(logging.INFO, logger="solprop.gasflow"):
        fill = steady_fill(
            custom, [0.0, 1.0], [1.0, 0.0], radius=1e-4, vacuum_end="floor"
        )
    assert fill.pressures[-1] == pytest.approx(1e-4)
    assert any("without its flow-regime check" in r.message for r in caplog.records)


# ------------------------------------------------------------ throughput and units


def test_the_throughput_is_the_poiseuille_law():
    eta = viscosity("N2", ROOM)
    expected = math.pi * GEBHARDT_RADIUS**4 * (4.2 * BAR) ** 2 / (16 * eta * 5.0)
    assert throughput("N2", GEBHARDT_RADIUS, 5.0, 4.2, 0.0) == pytest.approx(
        expected, rel=1e-14
    )
    # A taper against a quadrature of a^-4.
    resistance, _ = quad(lambda z: (125e-6 - 50e-6 * z) ** -4, 0.0, 1.0, epsrel=1e-13)
    taper = throughput("Ar", Linear(125e-6, 75e-6, 1.0), 1.0, 1.0, 2.0)
    expected = math.pi * 3.0 * BAR**2 / (16 * viscosity("Ar", ROOM) * resistance)
    assert taper == pytest.approx(expected, rel=1e-12)
    with pytest.raises(TranslationError, match="length"):
        throughput("N2", 1e-4, 0.0, 1.0, 0.0)
    with pytest.raises(TranslationError, match="start_bar"):
        throughput("N2", 1e-4, 1.0, -1.0, 0.0)


def test_the_mass_flow_uses_the_real_gas_density_at_the_reference_state():
    ideal = molar_mass("Ar") / (GAS_CONSTANT * ROOM)
    assert mass_flow("Ar", 2.0, 0.0) == pytest.approx(2.0 * ideal, rel=1e-15)
    # At 20 bar argon's compressibility is about 0.99, so the real gas is denser.
    assert mass_flow("Ar", 2.0, 20.0) > 2.0 * ideal * 1.005


def test_the_throughput_units():
    assert mbar_litres_per_second(1.0) == 10.0
    # 1 Pa m^3/s at 0 C is 1/101325 m^3 at one atmosphere per second.
    assert standard_litres_per_minute(1.0, 273.15) == pytest.approx(
        60e3 / 101325, rel=1e-15
    )
    assert standard_litres_per_minute(1.0, 294.0) == pytest.approx(
        60e3 / 101325 * 273.15 / 294.0, rel=1e-15
    )


def test_the_chamber_pressure_with_a_constant_pump_speed():
    assert chamber_pressure(1.0, 0.01) == pytest.approx(1.0 / 0.01 / BAR, rel=1e-15)
    with pytest.raises(TranslationError, match="pump speed"):
        chamber_pressure(1.0, 0.0)


def test_the_chamber_pressure_with_a_pump_characteristic():
    # S(p) = S0 p / (p + p0) falls at low pressure: p S(p) = Q is a quadratic.
    s0, p0, q = 0.01, 1e-3, 0.5

    def speed(p):
        return s0 * p / (p + p0)

    a, b = s0 * BAR, -q / BAR
    expected = (-b * BAR + math.sqrt((b * BAR) ** 2 + 4 * a * q * p0)) / (2 * a)
    assert chamber_pressure(q, speed) == pytest.approx(expected, rel=1e-10)
    assert chamber_pressure(0.0, speed) == 0.0
    with pytest.raises(TranslationError, match="does not reach"):
        chamber_pressure(1e9, speed)


def test_the_report_prints_its_flow():
    fill = steady_fill("He", [0.0, 3.0], [5.0, 0.0], radius=125e-6)
    report = flow_report(fill, "He", radius=125e-6)
    assert isinstance(report, FlowReport)
    assert repr(report).startswith("FlowReport(1 stage, peak Re")
    assert "end choked" in repr(report)
    text = str(report)
    assert "mbar l/s" in text and "choked pressure" in text
    stage = report.stages[0]
    assert stage.throughput_mbar_litres_per_second == pytest.approx(
        10 * stage.throughput
    )
    assert stage.throughput_slpm > 0.0
    assert "StageFlow(" in repr(stage)


def test_a_smooth_radius_is_accepted_by_the_calculators():
    bore = FromCallable(
        lambda z: 100e-6 * (1 - 0.3 * np.sin(np.pi * z)),
        (0.0, 1.0),
        slope=lambda z: -30e-6 * np.pi * np.cos(np.pi * z),
    )
    fill = steady_fill("Ar", [0.0, 1.0], [2.0, 0.0], radius=bore)
    stage = flow_report(fill, "Ar", radius=bore).stages[0]
    assert stage.peak_reynolds_position == pytest.approx(0.5, abs=1e-2)
    with pytest.raises(TranslationError, match="mass_flow"):
        reynolds_number("Ar", -1.0, bore, np.array([0.0]))
    with pytest.raises(TranslationError, match="positive"):
        reynolds_number("Ar", 1.0, -1e-6, np.array([0.0]))


def test_the_report_names_its_largest_knudsen_number():
    fill = steady_fill("He", [0.0, 1.0, 3.0], [0.0, 5.0, 0.0], radius=125e-6)
    report = flow_report(fill, "He", radius=125e-6)
    assert report.peak_knudsen == max(stage.peak_knudsen for stage in report.stages)
    # The longer stage chokes lower, so the rarest gas is at its end.
    assert report.peak_knudsen == pytest.approx(
        float(knudsen_number("He", fill.pressures[-1], 125e-6)), rel=1e-12
    )


def test_a_mixture_fill_refuses_what_it_cannot_honour():
    with pytest.raises(MaterialError, match="vacuum_end must be"):
        Mixture.fill({"He": 1.0}, (1.0, 0.0), length=1.0, vacuum_end="sealed")
    with pytest.raises(MaterialError, match="needs the core radius"):
        Mixture.fill({"He": 1.0}, (1.0, 0.0), length=1.0, vacuum_end="choked")
    with pytest.raises(MaterialError, match="pass pressure="):
        Mixture.fill({"He": 1.0}, 1.0).with_fill(radius=1e-4)
    # Without a radius, floor is today's fill exactly: the same profile from the same
    # stages (an explicitly exact no-op route).
    floor = Mixture.fill({"He": 1.0}, (1.0, 0.0), length=1.0, vacuum_end="floor")
    today = Mixture.fill({"He": 1.0}, (1.0, 0.0), length=1.0)
    assert isinstance(floor.shared, PressureGradient)
    assert isinstance(today.shared, PressureGradient)
    assert np.array_equal(floor.shared.pressures, today.shared.pressures)


def test_a_mixture_refill_takes_the_bore():
    mixture = Mixture.fill({"N2": 0.8, "O2": 0.2}, 1.0)
    refilled = mixture.with_fill(pressure=(1.0, 0.0), length=2.0, radius=100e-6)
    assert isinstance(refilled.shared, PressureGradient)
    fractions = {"N2": 0.8, "O2": 0.2}
    assert refilled.shared.pressures[-1] == pytest.approx(
        choked_pressure(fractions, 100e-6, 2.0, 1.0), rel=1e-14
    )


def test_a_choked_fill_is_marked_and_a_composition_change_refused():
    fractions = {"N2": 0.8, "O2": 0.2}
    mixture = Mixture.fill(fractions, (1.0, 0.0), length=2.0, radius=100e-6)
    fill = mixture.shared
    assert isinstance(fill, PressureGradient)
    assert fill.choked_ends == (False, True)
    assert profile_identity(fill)["choked_ends"] == ["False", "True"]
    assert "choked" in str(fill)
    with pytest.raises(MaterialError, match="old composition"):
        mixture.with_fill({"N2": 0.5, "O2": 0.5})
    # A new pressure rebuilds the flow for the new fractions.
    refilled = mixture.with_fill(
        {"N2": 0.5, "O2": 0.5}, (1.0, 0.0), length=2.0, radius=100e-6
    )
    assert isinstance(refilled.shared, PressureGradient)
    assert refilled.shared.pressures[-1] == pytest.approx(
        choked_pressure({"N2": 0.5, "O2": 0.5}, 100e-6, 2.0, 1.0), rel=1e-14
    )
    # A fill whose ends are the declared ones does not depend on the gas.
    floored = Mixture.fill(
        fractions, (1.0, 0.0), length=2.0, radius=100e-6, vacuum_end="floor"
    )
    assert floored.with_fill({"N2": 0.5, "O2": 0.5}).shared is floored.shared
    with pytest.raises(TranslationError, match="one flag"):
        dataclasses.replace(fill, choked_ends=(True,))


def test_a_boreless_profile_beside_a_varying_radius_is_refused_at_the_mixture():
    taper = Linear(125e-6, 75e-6, 1.0)
    boreless = PressureGradient.two_point(1.0, 1.0, 0.0)
    with pytest.raises(MaterialError, match=r"from_stages\(\.\.\., radius=\)"):
        Mixture.fill({"He": 1.0}, boreless, radius=taper)
    with pytest.raises(MaterialError, match="built without the bore"):
        Mixture.air(boreless, radius=taper)
    with pytest.raises(MaterialError, match="built without the bore"):
        Mixture.fill({"He": 1.0}, 1.0).with_fill(pressure=boreless, radius=taper)
    # A profile that follows the bore, a constant pressure or a constant radius is fine.
    bored = PressureGradient.two_point(1.0, 1.0, 0.0, radius=taper)
    assert Mixture.fill({"He": 1.0}, bored, radius=taper).shared is bored
    assert Mixture.fill({"He": 1.0}, 2.0, radius=taper).shared is not None
    assert Mixture.fill({"He": 1.0}, boreless, radius=1e-4).shared is boreless
