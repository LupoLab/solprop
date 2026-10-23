# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Gas temperature along z in the density path and the steady-flow law."""

import logging
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pytest
from numpy.polynomial import Chebyshev
from scipy.integrate import quad
from scipy.optimize import brentq

from solprop.constants import BAR, ROOM_TEMPERATURE
from solprop.exceptions import MaterialError, MaterialStatusWarning, TranslationError
from solprop.gasflow import (
    _viscosity_law,
    choked_pressure,
    flow_report,
    steady_fill,
    temperature_weight,
)
from solprop.physdata import Gas, Species, density, species
from solprop.physdata.equation_of_state import _check_temperature
from solprop.physdata.equation_of_state import density as eos_density
from solprop.physdata.species import OptmatGas
from solprop.physdata.transport import (
    mean_free_path,
    molar_mass,
    sound_speed,
    temperature_range,
    viscosity,
)
from solprop.profiles import (
    Constant,
    DensityPath,
    FromCallable,
    Linear,
    PiecewiseLinear,
    PressureGradient,
    SeriesOfPath,
    _checked_temperature,
    hydraulic_integral,
    profile_identity,
)

FILL = PressureGradient.two_point(1.0, 10.0, 0.5)
RAMP = PiecewiseLinear(np.array([0.0, 0.4, 1.0]), np.array([294.0, 360.0, 330.0]))
"""A temperature with an interior kink, rising then falling, over the fill."""
WARM = Linear(294.0, 400.0, 1.0)
TAPER = Linear(125e-6, 75e-6, 1.0)


def points(path):
    """Return evenly spaced positions over [0, 1] m with the path's kinks."""
    return np.unique(np.concatenate([np.linspace(0.0, 1.0, 41), path.breakpoints]))


def value(path, z):
    """Return a path's value at one position."""
    return float(path(np.array([z]))[0])


# ------------------------------------------------------------------ density paths


@pytest.mark.parametrize("gas", ["Ar", "N2", "He", "Xe"])
def test_the_density_follows_pressure_and_temperature_like_direct_calls(gas):
    path = DensityPath(gas, FILL, RAMP)
    z = points(RAMP)
    direct = np.array([density(gas, value(FILL, x), value(RAMP, x)) for x in z])
    # The compressibility series converges to 1e-13 per axis; the declared bound
    # is 1e-9 relative.
    np.testing.assert_allclose(path(z), direct, rtol=1e-9, atol=0.0)
    np.testing.assert_array_equal(path.breakpoints, [0.0, 0.4, 1.0])


@pytest.mark.parametrize("gas", ["Ar", "He"])
def test_the_density_derivative_is_the_chain_rule_through_both_paths(gas):
    path = DensityPath(gas, FILL, RAMP)
    assert path.derivative_is_analytic
    # Inside each piece of the kinked temperature, away from its breakpoint.
    z = np.array([0.05, 0.2, 0.35, 0.45, 0.7, 0.95])
    step = 1e-4
    wide = (path(z + step) - path(z - step)) / (2 * step)
    narrow = (path(z + step / 2) - path(z - step / 2)) / step
    # A Richardson difference of the tabulated density: its truncation is O(h^4)
    # and its rounding eps rho / h ~ 1e-12 relative; 1e-7 is the profiles' bound.
    np.testing.assert_allclose(
        path.derivative(z), (4 * narrow - wide) / 3, rtol=1e-7, atol=0.0
    )


def test_a_constant_temperature_path_is_the_number_exactly():
    # A constant path is normalised to its number, so the density runs the number's
    # computation on the same inputs: an explicitly exact no-op, compared exactly.
    z = points(FILL)
    number = DensityPath("Ar", FILL, 294.0)
    for constant in (Constant(294.0), Linear(294.0, 294.0, 1.0)):
        path = DensityPath("Ar", FILL, constant)
        assert path.temperature == 294.0 and isinstance(path.temperature, float)
        np.testing.assert_array_equal(path(z), number(z))
        np.testing.assert_array_equal(path.derivative(z), number.derivative(z))
        assert path.breakpoints is FILL.breakpoints
        assert profile_identity(path) == profile_identity(number)
        assert repr(path) == repr(number)


@dataclass
class CountingState:
    """An equation of state that counts its calls, around a built-in gas's."""

    gas: Gas
    calls: list[int] = field(default_factory=lambda: [0])

    def density(self, pressure_bar: float, temperature: float) -> float:
        """Return the built-in (optmat) density, counting the call."""
        self.calls[0] += 1
        return eos_density(self.gas, pressure_bar, temperature)

    def pressure(self, number_density: float, temperature: float) -> float:
        """Not used here."""
        raise NotImplementedError


def test_a_temperature_path_costs_fewer_equation_of_state_calls_than_a_number():
    state = CountingState(Gas.Ar)
    material = Species(
        "counted argon", species("Ar").susceptibility, equation_of_state=state
    )
    DensityPath(material, FILL, 294.0)
    constant = state.calls[0]
    state.calls[0] = 0
    path = DensityPath(material, FILL, RAMP)
    along = state.calls[0]
    assert constant == 1024
    assert path._table is not None
    # Each axis is raised from degree 8 until converged; every trial is re-sampled.
    assert along < constant
    z = points(RAMP)
    direct = [eos_density(Gas.Ar, value(FILL, x), value(RAMP, x)) for x in z]
    np.testing.assert_allclose(path(z), direct, rtol=1e-9, atol=0.0)


def test_an_ideal_species_follows_the_temperature_exactly():
    material = Species("ideal", species("He").susceptibility)
    path = DensityPath(material, FILL, WARM)
    z = points(WARM)
    expected = FILL(z) * BAR / (1.380649e-23 * WARM(z))
    np.testing.assert_allclose(path(z), expected, rtol=1e-13, atol=0.0)


def test_a_state_outside_the_table_is_evaluated_directly_and_logged(caplog):
    # A narrow warm spike between the probes: the table never saw 500 K.
    spike = PiecewiseLinear(
        np.array([0.0, 0.5, 0.5001, 0.5002, 1.0]),
        np.array([294.0, 294.0, 500.0, 294.0, 294.0]),
    )
    tilted = FromCallable(lambda z: spike(z) + z, (0.0, 1.0), np.array([0.0, 1.0]))
    path = DensityPath("Ar", FILL, tilted)
    with caplog.at_level(logging.WARNING, logger="solprop.profiles"):
        rho = path(np.array([0.50010]))
    assert "outside the range sampled at construction" in caplog.text
    expected = density("Ar", value(FILL, 0.5001), value(tilted, 0.5001))
    assert rho[0] == pytest.approx(expected, rel=1e-14)


def test_a_temperature_on_an_unbounded_pressure_takes_its_own_span():
    path = DensityPath("N2", Constant(2.0), WARM)
    assert path.z_span == (0.0, 1.0)
    assert not path.is_constant
    np.testing.assert_array_equal(path.breakpoints, [0.0, 1.0])
    vacuum = DensityPath("N2", Constant(0.0), WARM)
    assert vacuum.is_constant
    np.testing.assert_array_equal(vacuum(np.array([0.2, 0.8])), [0.0, 0.0])


@pytest.mark.parametrize(
    "temperature",
    [
        0.0,
        -3.0,
        math.nan,
        Linear(294.0, -1.0, 1.0),
        FromCallable(
            lambda z: np.where(z > 0.5, np.nan, 300.0), (0.0, 1.0), np.array([0.0, 1.0])
        ),
    ],
)
def test_a_temperature_that_is_not_positive_is_refused(temperature):
    with pytest.raises(MaterialError, match="temperature of a density path"):
        DensityPath("Ar", FILL, temperature)


def test_the_identity_and_the_repr_carry_the_temperature_path():
    path = DensityPath("Ar", FILL, WARM)
    identity = profile_identity(path)
    assert identity["temperature"] == profile_identity(WARM)
    assert "Linear(294 to 400 over 1 m) K" in repr(path)
    assert profile_identity(DensityPath("Ar", FILL, RAMP)) != identity


def test_the_equation_of_state_refuses_a_temperature_path_by_name():
    with pytest.raises(MaterialError, match="one temperature in K"):
        _check_temperature(WARM)


# ------------------------------------------------------------------ series of a path


def test_a_series_of_a_path_evaluates_and_differentiates_through_it():
    law = Chebyshev([1.0, 0.5, 0.25], domain=[294.0, 400.0])
    weight = SeriesOfPath(RAMP, law)
    z = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(weight(z), law(RAMP(z)), rtol=1e-15)
    np.testing.assert_allclose(
        weight.derivative(z), law.deriv()(RAMP(z)) * RAMP.derivative(z), rtol=1e-15
    )
    np.testing.assert_array_equal(weight.breakpoints, RAMP.breakpoints)
    assert weight.z_span == RAMP.z_span and not weight.is_constant
    assert weight.derivative_is_analytic
    assert SeriesOfPath(Constant(300.0), law).is_constant
    identity = profile_identity(weight)
    assert identity["path"] == profile_identity(RAMP)
    other = SeriesOfPath(RAMP, Chebyshev([1.0, 0.5, 0.26], domain=[294.0, 400.0]))
    assert profile_identity(other) != identity
    assert "degree 2" in repr(weight)


def test_the_temperature_weight_is_viscosity_times_temperature():
    weight = temperature_weight("N2", WARM, [0.0, 1.0])
    assert isinstance(weight, SeriesOfPath)
    z = np.linspace(0.0, 1.0, 17)
    direct = [viscosity("N2", value(WARM, x)) * value(WARM, x) for x in z]
    # The viscosity law is checked to 1e-6 relative (its data are good to 0.3 %).
    np.testing.assert_allclose(weight(z), direct, rtol=1e-6, atol=0.0)
    constant = temperature_weight("N2", 300.0, [0.0, 1.0])
    assert isinstance(constant, Constant)
    assert constant.value == viscosity("N2", 300.0) * 300.0
    # A name is optmat's nitrogen, declared from 63.151 K to 2000 K.
    with pytest.raises(MaterialError, match=r"declared for 63\.151 K to 2000 K"):
        temperature_weight("N2", Linear(294.0, 2100.0, 1.0), [0.0, 1.0])


# ------------------------------------------------------------------ the flow law


def flow_weight(gas, temperature, radius):
    """Return the integrand eta(T) T / a^4 at one position, from direct calls."""

    def integrand(z):
        kelvin = value(temperature, z)
        return viscosity(gas, kelvin) * kelvin / value(radius, z) ** 4

    return integrand


@pytest.mark.parametrize("radius", [Constant(125e-6), TAPER])
def test_the_fill_follows_the_temperature_weight_against_a_quadrature(radius):
    fill = steady_fill("Ar", [0.0, 1.0], [5.0, 1.0], radius=radius, temperature=WARM)
    assert fill.weight is not None
    integrand = flow_weight("Ar", WARM, radius)
    total, error = quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=1e-13, limit=200)
    for z in (0.1, 0.37, 0.5, 0.83, 1.0):
        part, _ = quad(integrand, 0.0, z, epsabs=0.0, epsrel=1e-13, limit=200)
        fraction = part / total
        expected = math.sqrt((1.0 - fraction) * 25.0 + fraction * 1.0)
        # The weight's viscosity law is checked to 1e-6 and enters p^2 through a
        # ratio of integrals, which moves by at most twice that.
        assert value(fill, z) == pytest.approx(expected, rel=2e-6)
    assert error < 1e-10 * total
    # The derivative stays analytic: (p1^2 - p0^2) w a^-4 / (2 R p).
    z = np.array([0.25, 0.6])
    slope = (1.0 - 25.0) * np.array([integrand(x) for x in z]) / (2 * total * fill(z))
    # The weight at the point and the stage's integral, each good to 1e-6.
    np.testing.assert_allclose(fill.derivative(z), slope, rtol=2e-6)


def test_a_constant_temperature_path_gives_the_unweighted_fill_exactly():
    # A constant temperature and a scalar weight are both dropped (no weight), so the
    # fill runs the unweighted computation: an explicitly exact no-op.
    z = np.linspace(0.0, 1.0, 33)
    number = steady_fill("He", [0.0, 3.0], [5.0, 0.0], radius=TAPER)
    constant = steady_fill(
        "He",
        [0.0, 3.0],
        [5.0, 0.0],
        radius=TAPER,
        temperature=Constant(ROOM_TEMPERATURE),
    )
    assert constant.weight is None
    np.testing.assert_array_equal(constant.pressures, number.pressures)
    np.testing.assert_array_equal(constant(z), number(z))
    np.testing.assert_array_equal(constant.derivative(z), number.derivative(z))
    assert profile_identity(constant) == profile_identity(number)
    weighted = PressureGradient.from_stages([0.0, 1.0], [5.0, 1.0], weight=3.0)
    plain = PressureGradient.from_stages([0.0, 1.0], [5.0, 1.0])
    assert weighted.weight is None
    np.testing.assert_array_equal(weighted(z), plain(z))
    assert profile_identity(weighted) == profile_identity(plain)


def test_a_weighted_fill_breaks_where_the_temperature_does():
    fill = PressureGradient.from_stages(
        [0.0, 0.6, 1.0], [5.0, 3.0, 1.0], weight=temperature_weight("He", RAMP, [0, 1])
    )
    np.testing.assert_array_equal(fill.breakpoints, [0.0, 0.4, 0.6, 1.0])
    assert "weighted" in repr(fill)
    weight = profile_identity(fill)["weight"]
    assert isinstance(weight, dict) and weight["kind"] == "SeriesOfPath"
    assert hydraulic_integral(125e-6, [0.0, 1.0], weight=2.0)[0] == pytest.approx(
        2.0 / 125e-6**4, rel=1e-15
    )


def test_a_weight_that_is_not_positive_is_refused():
    with pytest.raises(TranslationError, match="flow weight"):
        PressureGradient.from_stages(
            [0.0, 1.0], [5.0, 1.0], weight=Linear(1.0, -1.0, 1.0)
        )
    with pytest.raises(TranslationError, match="flow weight"):
        hydraulic_integral(125e-6, [0.0, 1.0], weight=-1.0)


# ------------------------------------------------------------------ the choke


def test_a_hot_exit_chokes_with_its_own_viscosity_and_sound_speed():
    gas, length, radius, high = "He", 3.0, 125e-6, 5.0
    temperature = Linear(294.0, 394.0, length)
    fill = steady_fill(
        gas, [0.0, length], [high, 0.0], radius=radius, temperature=temperature
    )
    assert fill.choked_ends == (False, True)
    # Independently: mdot = pi M (p_h^2 - p_e^2) / (16 R int eta T a^-4 dz) meets the
    # sonic mdot = p_e M pi a^2 c_e / (R T_e), with c_e the exit's sound speed.
    integrand = flow_weight(gas, temperature, Constant(radius))
    resistance, _ = quad(integrand, 0.0, length, epsabs=0.0, epsrel=1e-13)
    exit_temperature = 394.0
    factor = exit_temperature / (
        16.0 * sound_speed(gas, exit_temperature) * radius**2 * resistance
    )
    top = high * BAR
    expected = brentq(
        lambda p: factor * (top**2 - p**2) - p, 0.0, top, xtol=1e-20, rtol=1e-15
    )
    # The viscosity law's 1e-6 carried through the resistance: p_e ~ 1/W.
    assert fill.pressures[-1] * BAR == pytest.approx(expected, rel=2e-6)
    # A warm exit's choked pressure lies below the room-temperature one.
    assert fill.pressures[-1] < choked_pressure(gas, radius, length, high)


def test_the_report_takes_the_local_viscosity_and_mean_free_path():
    gas, radius = "Ar", TAPER
    fill = steady_fill(gas, [0.0, 1.0], [5.0, 0.5], radius=radius, temperature=WARM)
    report = flow_report(fill, gas, radius=radius, temperature=WARM)
    (stage,) = report.stages
    integrand = flow_weight(gas, WARM, radius)
    resistance, _ = quad(integrand, 0.0, 1.0, epsabs=0.0, epsrel=1e-13)
    low_t, low_p = 400.0, 0.5
    throughput = (
        math.pi * low_t * ((5.0 * BAR) ** 2 - (low_p * BAR) ** 2) / (16.0 * resistance)
    )
    # The throughput is quoted at the low end's temperature; every quantity below
    # carries the viscosity law's 1e-6 once or twice (the resistance, the local eta).
    assert stage.temperature == low_t
    assert stage.throughput == pytest.approx(throughput, rel=2e-6)
    # The fill's own equation of state: optmat's for a name since SEP 3's Phase 4a.
    number = species(gas).equation_of_state.density(low_p, low_t)
    mass = throughput * number * molar_mass(gas) / 6.02214076e23
    assert stage.mass_flow == pytest.approx(mass / (low_p * BAR), rel=2e-6)
    # The narrowest, hottest point: Re = 2 mdot / (pi a eta(T)).
    peak = stage.peak_reynolds_position
    assert peak == pytest.approx(1.0)
    expected = 2 * stage.mass_flow / (math.pi * 75e-6 * viscosity(gas, 400.0))
    assert stage.peak_reynolds == pytest.approx(expected, rel=2e-6)
    # Kn at the exit with the exit's mean free path.
    knudsen = mean_free_path(gas, low_p, low_t) / (2 * 75e-6)
    assert stage.knudsen_end == pytest.approx(knudsen, rel=2e-6)
    assert report.viscosity == viscosity(gas, 294.0)


def test_a_temperature_path_needs_the_gas_transport_data_whatever_the_end():
    material = Species("inert", species("He").susceptibility)
    with pytest.raises(MaterialError, match="temperature path weights"):
        steady_fill(
            material,
            [0.0, 1.0],
            [5.0, 1.0],
            radius=125e-6,
            temperature=WARM,
            vacuum_end="floor",
        )
    registered = Species(
        "argon", species("Ar").susceptibility, equation_of_state=OptmatGas(Gas.Ar)
    )
    fill = steady_fill(
        registered, [0.0, 1.0], [5.0, 1.0], radius=125e-6, temperature=WARM
    )
    assert fill.weight is not None


# ------------------------------------------------------------------ edge cases


FLAT = FromCallable(lambda z: 300.0 + 0.0 * z, (0.0, 1.0), np.array([0.0, 1.0]))
"""A path that never varies but cannot say so: a callable is never known constant."""


def test_a_path_that_never_varies_is_tabulated_on_a_widened_axis():
    path = DensityPath("Ar", FILL, FLAT)
    assert path._table is not None
    z = points(FILL)
    direct = [density("Ar", value(FILL, x), 300.0) for x in z]
    np.testing.assert_allclose(path(z), direct, rtol=1e-9, atol=0.0)
    fill = steady_fill("Ar", [0.0, 1.0], [5.0, 1.0], radius=125e-6, temperature=FLAT)
    plain = steady_fill("Ar", [0.0, 1.0], [5.0, 1.0], radius=125e-6, temperature=300.0)
    # A weight that is constant in value cancels from the fill to rounding.
    np.testing.assert_allclose(fill(z), plain(z), rtol=1e-14)


@dataclass
class KinkedState:
    """An equation of state whose compressibility kinks at 5 bar: not smooth."""

    def density(self, pressure_bar: float, temperature: float) -> float:
        """Return a density whose compressibility has a kink."""
        factor = 1.0 + 0.05 * abs(pressure_bar - 5.0)
        return pressure_bar * BAR / (1.380649e-23 * temperature * factor)

    def pressure(self, number_density: float, temperature: float) -> float:
        """Not used here."""
        raise NotImplementedError


def test_a_compressibility_that_does_not_converge_is_refused_by_name():
    material = Species(
        "kinked", species("Ar").susceptibility, equation_of_state=KinkedState()
    )
    with pytest.raises(MaterialError, match="did not converge by Chebyshev degree 64"):
        DensityPath(material, FILL, WARM)


def test_a_flow_weight_with_an_undeclared_kink_is_refused_by_name():
    kinked = FromCallable(
        lambda z: 1.0 + np.abs(z - 0.5), (0.0, 1.0), np.array([0.0, 1.0])
    )
    with pytest.raises(
        TranslationError, match="did not converge by Chebyshev degree 512"
    ):
        PressureGradient.from_stages([0.0, 1.0], [5.0, 1.0], weight=kinked)


def test_the_temperature_check_takes_numbers_and_paths_only():
    assert _checked_temperature(np.array(300.0)) == 300.0
    assert isinstance(_checked_temperature(np.float64(300.0)), float)
    assert _checked_temperature(300) == 300
    for bad in (True, "hot", [300.0]):
        with pytest.raises(MaterialError, match="a number in K or a ZProfile"):
            _checked_temperature(bad)


def test_the_printed_blocks_name_the_weight_and_the_series():
    fill = steady_fill("Ar", [0.0, 1.0], [5.0, 1.0], radius=TAPER, temperature=WARM)
    assert fill.weight is not None
    text = str(fill)
    assert "flow weight" in text and "weighted" in text
    assert "SeriesOfPath: a Chebyshev series of a path" in str(fill.weight)


# ------------------------------------------------------------------ viscosity laws


TRANSPORT_GASES = [pytest.param(gas, id=gas.value) for gas in Gas if gas is not Gas.Air]


@pytest.mark.parametrize("gas", TRANSPORT_GASES)
def test_every_gas_has_a_viscosity_law_over_its_whole_range(gas):
    gas = species(gas)
    low, high = temperature_range(gas)
    if gas.name == "CH4":
        # optmat's CH4 viscosity is provisional, and says so once per evaluation.
        with pytest.warns(MaterialStatusWarning, match="transport/CH4"):
            law = _viscosity_law(gas, low, high)
    else:
        law = _viscosity_law(gas, low, high)
    checks = np.linspace(low, high, 301)
    with warnings.catch_warnings():
        # The provisional status is asserted above; the reference values repeat it.
        warnings.simplefilter("ignore", MaterialStatusWarning)
        direct = np.array([viscosity(gas, float(t)) for t in checks])
    # The law is accepted at 1e-6 on its own check grid; this grid is another.
    np.testing.assert_allclose(law(checks), direct, rtol=2e-6, atol=0.0)


def test_helium_and_a_helium_xenon_mix_have_viscosity_laws():
    # The reviewer's cases: CoolProp's helium is not smooth at 1e-9, so the old
    # coefficient-tail criterion never converged over 250 to 580 K.
    for gas in ("He", {"He": 0.95, "Xe": 0.05}):
        law = _viscosity_law(gas, 250.0, 580.0)
        checks = np.linspace(250.0, 580.0, 211)
        direct = np.array([viscosity(gas, float(t)) for t in checks])
        np.testing.assert_allclose(law(checks), direct, rtol=2e-6, atol=0.0)
    weight = temperature_weight("He", Linear(250.0, 580.0, 1.0), [0.0, 1.0])
    assert isinstance(weight, SeriesOfPath)


def test_a_viscosity_law_that_does_not_meet_the_tolerance_is_refused_by_name():
    # A quadratic cannot follow optmat's helium viscosity from 110 K to 1500 K to
    # 1e-6, and the refusal names the gas and the bound.
    with pytest.raises(MaterialError, match="was not reproduced to 1e-06"):
        _viscosity_law(species("He"), 110.0, 1500.0, degrees=(2,))
