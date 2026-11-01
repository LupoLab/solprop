# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Design calculators: shape factors, soliton numbers, phase matching."""

import json
import math
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.integrate import quad
from scipy.special import jn_zeros, jv

from solprop import tools
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.constants import EPS0, ROOM_TEMPERATURE, C
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning, ToolsError
from solprop.grid import FieldGrid
from solprop.modes import MarcatiliMode, base
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.physdata import (
    CompositeGlass,
    Gas,
    MaterialSelection,
    Mixture,
    PoleSumSusceptibility,
    ReferenceState,
    SellmeierGlass,
    Species,
    n2,
    species,
)
from solprop.physdata.glasses import OptmatGlass, fused_silica
from solprop.physdata.nonlinear_data import chi3 as gas_chi3
from solprop.physdata.species import OptmatGas
from solprop.spatial import critical_power
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule

PUMP = 800e-9
OMEGA0 = 2.0 * math.pi * C / PUMP
# A constant cladding keeps every index request inside a model's range, so the suite's
# warnings-are-errors rule needs no filter here.
GLASS = 1.45
WITNESS = json.loads(
    (Path(__file__).parent / "data" / "luna_tools_optics.json").read_text()
)


def quiet(call, *arguments, **keywords):
    """Call something whose grid reaches beyond a gas fit, without the range warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return call(*arguments, **keywords)


# Luna's inputs for its witnesses below, supplied as a caller's own data: the published
# fits solprop's former registry carried (dc4ca0d), which Luna implements too, and the
# registry's former helium and krypton gamma3 (Lehmeier et al. 1985).
BAR_0C = ReferenceState(pressure_bar=1.0, temperature=273.15)
ATM_0C = ReferenceState(pressure_bar=1.01325, temperature=273.15)
HELIUM_BORZSONYI = PoleSumSusceptibility(
    gas=Gas.He,
    source="borzsonyi",
    reference_state=BAR_0C,
    wavelength_range=(400e-9, 1000e-9),
    citation="A. Börzsönyi et al., Appl. Opt. 47, 4856 (2008), Luna's :HeB",
    b=(4977.77e-8, 1856.94e-8),
    c=(28.54e-6, 7.76e-3),
)
HELIUM_ERMOLOV = PoleSumSusceptibility(
    gas=Gas.He,
    source="ermolov",
    reference_state=ATM_0C,
    wavelength_range=(90e-9, 1000e-9),
    citation="A. Ermolov et al., Phys. Rev. A 92, 033821 (2015), Luna's :He",
    b=(2.16463842e-05, 2.10561127e-07, 4.75092720e-05),
    c=(-6.80769781e-04, 5.13251289e-03, 3.18621354e-03),
)
KRYPTON_KINGSTON = PoleSumSusceptibility(
    gas=Gas.Kr,
    source="kingston",
    reference_state=ATM_0C,
    wavelength_range=(130e-9, 1000e-9),
    citation="three-pole fit to Kingston (1964) and Börzsönyi (2008), Luna PR #425",
    b=(7.156207185088921e-6, 0.00023020614937540427, 0.000604323109684193),
    c=(0.016126479267946525, 0.01348822574907029, 0.004204161467601559),
)
LEHMEIER_GAMMA3 = {"He": 5.109203705817757e-53, "Kr": 3.25915498422145e-51}
"""m^5/V^2: solprop's former registry values (dc4ca0d), Lehmeier et al. (1985)."""
MALITSON = SellmeierGlass(
    name="SiO2",
    source="malitson (test data)",
    wavelength_range=(0.21e-6, 3.71e-6),
    citation="I. H. Malitson, J. Opt. Soc. Am. 55, 1205 (1965)",
    poles=((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)),
)


# The Luna witnesses below are frozen (``docs/dev/testing.md``), so every input they
# depend on is stated rather than defaulted. The temperature is solprop's room
# temperature, deliberately, against Luna's 294 K: the bounds below absorb it.
WITNESS_TEMPERATURE = 293.15


def former_fill(law, pressure):
    """Return a fill of one supplied law over optmat's equation of state."""
    name = law.gas.value
    material = Species(name, law, OptmatGas(law.gas), gamma3=LEHMEIER_GAMMA3[name])
    return Mixture.fill({material: 1.0}, pressure, temperature=WITNESS_TEMPERATURE)


def former_silica():
    """Return solprop's former silica as a caller would build it: Malitson's real
    part with Franta's absorption (optmat's franta_2016 model, which the former table
    sampled to 2e-5)."""
    franta = fused_silica(model="franta_2016/udm_lithosil_q2")
    return CompositeGlass(
        name="SiO2",
        source="malitson+franta (test data)",
        wavelength_range=(
            max(MALITSON.wavelength_range[0], franta.wavelength_range[0]),
            min(MALITSON.wavelength_range[1], franta.wavelength_range[1]),
        ),
        citation="Malitson (1965) with Franta et al. (2016)",
        real_part=MALITSON,
        absorption=franta,
    )


def helium(pressure=1.0, **keywords):
    return MarcatiliMode(125e-6, "He", pressure, cladding=GLASS, **keywords)


# ------------------------------------------------------------------ the pulse shapes


@pytest.mark.parametrize(
    ("shape", "profile"),
    [
        ("sech", lambda t, tau: 1.0 / math.cosh(t / tau) ** 2),
        ("gauss", lambda t, tau: math.exp(-((t / tau) ** 2))),
    ],
)
def test_the_shape_factors_are_the_integrals_of_the_shapes(shape, profile):
    fwhm, energy = 10e-15, 1e-3
    tau = tools.pulse_width(fwhm, shape)
    # The half maximum sits at fwhm / 2 by the definition of the natural width.
    assert profile(fwhm / 2.0, tau) == pytest.approx(0.5, rel=1e-14)
    integral = quad(profile, -40.0 * tau, 40.0 * tau, args=(tau,), epsabs=0.0)[0]
    # quad's default relative tolerance, 1.5e-8, is what limits the comparison.
    assert tools.pulse_peak_power(energy, fwhm, shape) * integral == pytest.approx(
        energy, rel=1e-7
    )


def test_the_closed_forms_of_the_two_widths():
    assert tools.pulse_width(1.0, "sech") == pytest.approx(
        1.0 / (2.0 * math.log(1.0 + math.sqrt(2.0))), rel=1e-15
    )
    assert tools.pulse_width(1.0, "gauss") == pytest.approx(
        1.0 / (2.0 * math.sqrt(math.log(2.0))), rel=1e-15
    )
    assert tools.pulse_peak_power(1.0, 1.0, "gauss") == pytest.approx(
        2.0 * math.sqrt(math.log(2.0) / math.pi), rel=1e-15
    )


@pytest.mark.parametrize("shape", ["sech", "gauss"])
def test_the_conventional_unit_order_has_equal_lengths(shape):
    fwhm, beta2, gamma = 30e-15, -2e-29, 3e-10
    power = 1.0 / (gamma * tools.dispersion_length(fwhm, beta2, shape))
    # Products and one square root of order-one numbers: a few ulps.
    assert tools.soliton_order(power, fwhm, gamma, beta2, shape) == pytest.approx(
        1.0, rel=1e-14
    )
    assert tools.fission_length(power, fwhm, gamma, beta2, shape) == pytest.approx(
        tools.nonlinear_length(power, gamma), rel=1e-14
    )


@given(
    energy=st.floats(1e-9, 1e-2),
    fwhm=st.floats(5e-15, 1e-12),
    scale=st.floats(1.1, 50.0),
)
def test_the_order_scales_as_root_energy_and_as_duration(energy, fwhm, scale):
    gamma, beta2 = 1e-10, -1e-29

    def order(pulse_energy, width):
        power = tools.pulse_peak_power(pulse_energy, width)
        return tools.soliton_order(power, width, gamma, beta2)

    # N^2 = gamma P0 tau0^2 / |beta2| with P0 = E / (2 tau0): N ~ sqrt(E tau).
    assert order(scale * energy, fwhm) == pytest.approx(
        math.sqrt(scale) * order(energy, fwhm), rel=1e-12
    )
    assert order(energy, scale * fwhm) == pytest.approx(
        math.sqrt(scale) * order(energy, fwhm), rel=1e-12
    )
    assert order(energy, fwhm) * tools.fission_length(
        tools.pulse_peak_power(energy, fwhm), fwhm, gamma, beta2
    ) == pytest.approx(tools.dispersion_length(fwhm, beta2), rel=1e-12)


@given(field=st.floats(1.0, 1e13))
def test_field_and_intensity_round_trip(field):
    # One multiplication, one division and a square root: a couple of ulps.
    assert tools.intensity_to_field(tools.field_to_intensity(field)) == pytest.approx(
        field, rel=1e-14
    )


def test_the_critical_power_is_the_package_townes_power_times_the_index():
    index, nonlinear = 1.0003, 3.2e-23
    expected = 1.86226 * PUMP**2 / (4.0 * math.pi * index * nonlinear)
    # The same constant reached through chi3 and back: rounding only.
    assert tools.self_focusing_power(PUMP, index, nonlinear) == pytest.approx(
        expected, rel=1e-13
    )
    chi3 = 4.0 * nonlinear * EPS0 * C * index**2 / 3.0
    # Both are the physical power, n0 times the index-free field threshold.
    assert tools.self_focusing_power(PUMP, index, nonlinear) == pytest.approx(
        critical_power(PUMP, chi3, index=index), rel=1e-9
    )
    assert critical_power(PUMP, chi3, index=index) == pytest.approx(
        index * critical_power(PUMP, chi3, index=1.0), rel=1e-13
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: tools._check_shape("lorentz"),
        lambda: tools.pulse_width(0.0),
        lambda: tools.pulse_peak_power(-1.0, 10e-15),
        lambda: tools.dispersion_length(10e-15, 0.0),
        lambda: tools.dispersion_length(10e-15, math.nan),
        lambda: tools.nonlinear_length(1.0, 0.0),
        lambda: tools.nonlinear_coefficient(helium(), -1.0, PUMP),
        lambda: tools.intensity_to_field(-1.0),
        lambda: tools.self_focusing_power(PUMP, 1.0, 0.0),
        lambda: tools.capillary_params(
            -1e-6, "He", 1.0, energy=1e-6, fwhm=1e-14, wavelength=PUMP
        ),
    ],
)
def test_unusable_numbers_are_refused_by_name(call):
    with pytest.raises(ToolsError):
        call()


# --------------------------------------------------------------- soliton parameters


@pytest.fixture(scope="module")
def luna_case():
    """Luna's own test case: 300 uJ, 10 fs, 125 um, 0.4 bar helium, reduced model."""
    return tools.capillary_params(
        125e-6,
        "He",
        0.4,
        energy=300e-6,
        fwhm=10e-15,
        wavelength=PUMP,
        model="reduced",
        cladding=GLASS,
    )


def test_the_numbers_are_consistent_with_their_definitions(luna_case):
    p = luna_case
    assert p.gas == "He"
    assert p.pressure == pytest.approx(0.4, rel=1e-12)
    assert p.width == tools.pulse_width(p.fwhm, "sech")
    assert p.omega == pytest.approx(OMEGA0, rel=1e-15)
    assert p.n2 == pytest.approx(n2("He", pressure=0.4, wavelength=PUMP), rel=1e-12)
    assert p.gamma == p.n2 * p.omega / (C * p.effective_area)
    assert p.intensity == p.peak_power / p.effective_area
    assert p.soliton_order**2 == pytest.approx(
        p.dispersion_length / p.nonlinear_length, rel=1e-13
    )
    assert p.loss_length == pytest.approx(
        float(base.loss_length(p.mode, np.array([p.omega]))[0]), rel=1e-15
    )
    assert p.beta2 < 0.0


def test_lunas_own_test_case_is_reproduced_on_the_same_helium_fit():
    # Luna test/test_tools.jl, regenerated rather than pasted: params(300 uJ, 10 fs,
    # 800 nm) in a 125 um capillary of 0.4 bar of its :HeB helium, the Borzsonyi fit,
    # with the reduced Marcatili model and Luna's own silica cladding.
    order, fission, zdw, critical, beta2, gamma, n2_luna, area = WITNESS["params"][:8]
    mode = MarcatiliMode(
        125e-6,
        "He",
        0.4,
        model="reduced",
        loss=True,
        susceptibility=HELIUM_BORZSONYI,
        # Deliberately solprop's silica against Luna's table; it does not enter the
        # real part of beta in the reduced model.
        cladding=fused_silica(model="fit/SiO2"),
        temperature=WITNESS_TEMPERATURE,
    )
    # Luna's numbers on Luna's inputs, supplied: the fill's chi3 is the former
    # registry's default helium measurement, the one Luna carries too.
    p = quiet(
        tools.soliton_params,
        mode,
        former_fill(HELIUM_BORZSONYI, 0.4),
        energy=300e-6,
        fwhm=10e-15,
        wavelength=PUMP,
        shape="sech",
        temperature=WITNESS_TEMPERATURE,
    )
    # The area is a pure number times a^2.
    assert p.effective_area == pytest.approx(area, rel=1e-13)
    # beta2: solprop's matches the analytic second derivative of the reduced model to
    # 2e-8; Luna differences beta twice and lands 1.3e-4 away. The cladding does not
    # enter the real part of beta in this model. N and the fission length inherit half
    # of this and half of the chi3 difference below.
    assert p.beta2 == pytest.approx(beta2, rel=5e-4)
    assert p.soliton_order == pytest.approx(order, rel=3e-3)
    assert p.fission_length == pytest.approx(fission, rel=3e-3)
    # n2 and gamma: Luna's chi3 of helium against the same measurement.
    assert p.n2 == pytest.approx(n2_luna, rel=5e-3)
    assert p.gamma == pytest.approx(gamma, rel=5e-3)
    assert p.peak_power / p.critical_power == pytest.approx(critical, rel=5e-3)
    assert p.zdw == pytest.approx(zdw, rel=2e-3)


def test_lunas_other_numbers_within_the_difference_of_the_gas_models():
    # Krypton at 15 bar in a 14 um capillary: Luna's zero-dispersion point on its own
    # krypton index, against the former registry's krypton fit, supplied; and the
    # helium dispersive wave likewise, on the former helium and silica. The argon
    # zero-dispersion pressure is a built-in gas's only, optmat's argon since SEP 3:
    # 0.6 % from Luna's (measured; 1.5 % covers the index models, as before).
    params = quiet(
        tools.capillary_params,
        14e-6,
        former_fill(KRYPTON_KINGSTON, 15.0),
        energy=6e-9,
        fwhm=20e-15,
        wavelength=PUMP,
        shape="sech",
        temperature=WITNESS_TEMPERATURE,
        kind="HE",
        n=1,
        m=1,
        model="full",
        loss=True,
        cladding=fused_silica(model="fit/SiO2"),
    )
    assert params.zdw == pytest.approx(WITNESS["capillary"][2], rel=5e-3)
    mode = MarcatiliMode(
        125e-6,
        "He",
        1.0,
        model="full",
        loss=True,
        susceptibility=HELIUM_ERMOLOV,
        cladding=former_silica(),
        temperature=WITNESS_TEMPERATURE,
    )
    assert quiet(
        tools.lambda_rdw, mode, PUMP, shape="sech", nonlinear_phase="largest"
    ) == pytest.approx(WITNESS["rdw"][0], rel=1.5e-2)
    # Deliberately optmat's argon (a built-in gas): the difference is what is measured.
    assert quiet(
        tools.pressure_zdw,
        125e-6,
        "Ar",
        500e-9,
        pressure_max=100.0,
        temperature=WITNESS_TEMPERATURE,
        m=1,
        model="full",
        cladding=fused_silica(model="fit/SiO2"),
    ) == pytest.approx(WITNESS["pzdw"], rel=1.5e-2)


def test_the_on_axis_intensity_of_the_fundamental_is_the_bessel_ratio(luna_case):
    # For a field profile F(r) = J0(u r / a) the on-axis intensity per watt times the
    # effective area is |F(0)|^2 int|F|^2 / int|F|^4 * ... = (int J0^2 x dx)
    # / (int J0^4 x dx) over 0 < x < u, a pure number (1.768).
    u = jn_zeros(0, 1)[0]
    squared = quad(lambda x: jv(0, x) ** 2 * x, 0.0, u)[0]
    fourth = quad(lambda x: jv(0, x) ** 4 * x, 0.0, u)[0]
    # The mode's own area factor and its normalised profile are closed forms; quad is
    # good to 1e-10 here.
    assert luna_case.peak_intensity / luna_case.intensity == pytest.approx(
        squared / fourth, rel=1e-9
    )


def test_a_gaussian_runs_through_the_whole_chain():
    sech = tools.capillary_params(
        125e-6, "He", 0.4, energy=300e-6, fwhm=10e-15, wavelength=PUMP, cladding=GLASS
    )
    gauss = tools.capillary_params(
        125e-6,
        "He",
        0.4,
        energy=300e-6,
        fwhm=10e-15,
        wavelength=PUMP,
        shape="gauss",
        cladding=GLASS,
    )
    assert gauss.shape == "gauss"
    assert gauss.peak_power / sech.peak_power == pytest.approx(
        2.0
        * tools.pulse_width(1.0, "sech")
        / (math.sqrt(math.pi) * gauss.width / 1e-14),
        rel=1e-12,
    )
    assert gauss.soliton_order**2 == pytest.approx(
        gauss.gamma * gauss.peak_power * gauss.width**2 / abs(gauss.beta2), rel=1e-12
    )


def test_the_zero_dispersion_point_survives_a_resonance_in_the_window():
    # Krypton flips the sign of beta2 twice near 120 nm, which defeats a single
    # bracket over 100 nm to 3 um. Luna's test value is 769.30 nm on its own krypton
    # index; 0.5 % covers the difference between the two index models.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        params = tools.capillary_params(
            14e-6, "Kr", 15.0, energy=6e-9, fwhm=20e-15, wavelength=PUMP
        )
    assert params.zdw == pytest.approx(769.3e-9, rel=5e-3)
    assert float(
        base.beta2(params.mode, np.array([2.0 * math.pi * C / params.zdw]))[0]
    ) == pytest.approx(0.0, abs=1e-6 * abs(params.beta2))


def test_a_mode_without_a_zero_dispersion_point_reports_nan():
    mode = SpecifiedMode.from_taylor([0.0, 0.0, -1e-26], OMEGA0, aeff=1e-10)
    params = tools.soliton_params(
        mode, "Ar", 1.0, energy=1e-9, fwhm=50e-15, wavelength=PUMP
    )
    assert math.isnan(params.zdw)
    # An area and no transverse profile: a mode-averaged intensity and no on-axis one.
    assert math.isnan(params.peak_intensity)
    assert params.intensity == params.peak_power / 1e-10
    assert params.beta2 == pytest.approx(-1e-26, rel=1e-6)


def test_fills_are_read_as_the_run_interfaces_read_them():
    mode = helium()

    def params(fill, pressure=None):
        return tools.soliton_params(
            mode, fill, pressure, energy=1e-6, fwhm=30e-15, wavelength=PUMP
        )

    by_name = params("Ar", 2.0)
    by_mixture = params(Mixture.fill({"Ar": 1.0}, 2.0))
    assert by_name.n2 == by_mixture.n2
    assert by_name.n2 == pytest.approx(
        n2("Ar", pressure=2.0, wavelength=PUMP), rel=1e-12
    )
    halves = params({"Ar": 0.5, "Ne": 0.5}, 2.0)
    assert halves.gas == "Ar+Ne"
    assert n2("Ne", pressure=1.0, wavelength=PUMP) < halves.n2 < by_name.n2
    assert params("air", 1.0).gas == "N2+O2"
    assert params("He").pressure == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ToolsError, match="carries its own pressure"):
        params(Mixture.fill({"Ar": 1.0}, 2.0), 1.0)


@pytest.mark.parametrize(
    "tool",
    [
        lambda fill, **kw: tools.soliton_params(helium(), fill, **kw),
        lambda fill, **kw: tools.capillary_params(125e-6, fill, **kw),
    ],
    ids=["soliton_params", "capillary_params"],
)
def test_a_mixture_fill_refuses_a_temperature_other_than_its_own(tool):
    pulse = {"energy": 1e-6, "fwhm": 30e-15, "wavelength": PUMP}
    warm = Mixture.fill({"Ar": 1.0}, 2.0, temperature=300.0)
    # Not given, or given as the mixture's own: the mixture's temperature.
    assert tool(warm, **pulse).temperature == 300.0
    assert tool(warm, temperature=300.0, **pulse).temperature == 300.0
    assert tool("Ar", **pulse).temperature == ROOM_TEMPERATURE
    with pytest.raises(ToolsError, match=r"own temperature \(300 K\).*omit"):
        tool(warm, temperature=ROOM_TEMPERATURE, **pulse)


def test_gas_ratio_names_its_two_numbers():
    ratio = tools.gas_ratio("Ar", "He", PUMP)
    # optmat's coefficients at the pump (23.5 on solprop's former registry, 28.4 here).
    assert ratio.chi3 == pytest.approx(
        n2("Ar", pressure=1.0, wavelength=PUMP, selection=MaterialSelection())
        / n2("He", pressure=1.0, wavelength=PUMP, selection=MaterialSelection()),
        rel=1e-3,
    )
    inverse = tools.gas_ratio("He", "Ar", PUMP)
    assert ratio.beta2 * inverse.beta2 == pytest.approx(1.0, rel=1e-12)
    # Pressure cancels from both ratios to the accuracy of the ideal-gas limit.
    assert tools.gas_ratio("Ar", "He", PUMP, pressure=0.1).beta2 == pytest.approx(
        ratio.beta2, rel=2e-3
    )


# -------------------------------------------------------------------- phase matching


def taylor_mode(beta2=-1e-26, beta3=5e-41):
    return SpecifiedMode.from_taylor([0.0, 0.0, beta2, beta3], OMEGA0)


def test_the_linear_mismatch_has_a_double_root_at_the_pump():
    mode = helium()
    step = 1e-3 * OMEGA0
    around = tools.phase_mismatch(
        mode, OMEGA0 + step * np.array([-1.0, 0.0, 1.0]), PUMP
    )
    # beta is of order 1e7 1/m, so its rounding is 1e-9 1/m, against 2e-5 1/m here.
    assert around[1] == pytest.approx(0.0, abs=1e-8)
    curvature = float(base.beta2(mode, np.array([OMEGA0]))[0]) * step**2 / 2.0
    # Even about the pump up to the third-order term, beta3 step / (3 beta2), which
    # is a few parts in a thousand at this step.
    assert around[0] == pytest.approx(curvature, rel=1e-2)
    assert around[2] == pytest.approx(curvature, rel=1e-2)
    assert around[0] + around[2] == pytest.approx(2.0 * curvature, rel=1e-3)


def test_a_cubic_dispersion_has_the_closed_form_root():
    beta2, beta3 = -1e-26, 5e-41
    mode = taylor_mode(beta2, beta3)
    limits = (2.0 * math.pi * C / (OMEGA0 + 1e15), 0.95 * PUMP)
    found = tools.lambda_rdw(mode, PUMP, wavelength_limits=limits)
    expected = OMEGA0 - 3.0 * beta2 / beta3
    # brentq's default xtol is 2e-12 absolute on a root of order 1e15.
    assert 2.0 * math.pi * C / found == pytest.approx(expected, rel=1e-12)


def test_the_kerr_term_gives_the_root_of_the_cubic():
    beta2, beta3 = -1e-26, 5e-41
    mode = taylor_mode(beta2, beta3)
    power, gamma = 2e4, 1e-3
    limits = (2.0 * math.pi * C / (OMEGA0 + 1e15), 0.95 * PUMP)
    found = tools.lambda_rdw(
        mode,
        PUMP,
        peak_power=power,
        gamma=gamma,
        nonlinear_phase="supplied_power",
        wavelength_limits=limits,
    )
    roots = np.roots([beta3 / 6.0, beta2 / 2.0, 0.0, -0.5 * gamma * power])
    real = roots[np.abs(roots.imag) < 1e-6 * np.abs(roots.real)].real
    expected = OMEGA0 + float(np.max(real))
    # numpy.roots on a well-separated cubic: 1e-10 relative is generous.
    assert 2.0 * math.pi * C / found == pytest.approx(expected, rel=1e-10)
    linear = tools.lambda_rdw(mode, PUMP, wavelength_limits=limits)
    assert found < linear


def test_the_kerr_shift_is_to_the_blue_and_vanishes_with_the_power():
    mode = helium()
    gamma = tools.nonlinear_coefficient(
        mode, n2("He", pressure=1.0, wavelength=PUMP), PUMP
    )
    linear = tools.lambda_rdw(mode, PUMP)
    shifts = [
        linear
        - tools.lambda_rdw(
            mode, PUMP, peak_power=power, gamma=gamma, nonlinear_phase="supplied_power"
        )
        for power in (1e10, 1e9, 1e8)
    ]
    assert shifts[0] > shifts[1] > shifts[2] > 0.0
    # The shift is linear in the power once it is small: a decade for a decade.
    assert shifts[1] / shifts[2] == pytest.approx(10.0, rel=1e-2)


def test_the_nonlinear_term_needs_both_of_its_numbers():
    with pytest.raises(ToolsError, match="both peak_power and gamma"):
        tools.phase_mismatch(helium(), np.array([OMEGA0]), PUMP, peak_power=1e9)


def test_intermodal_matching_moves_the_root_as_the_index_difference_says():
    fundamental, second = helium(), helium(m=2)
    same = tools.lambda_rdw(fundamental, PUMP)
    # A wave in HE12 has the lower index, so against a soliton in HE11 the mismatch is
    # lowered everywhere and its blue root, where it rises through zero, moves bluer.
    cross = tools.lambda_rdw(second, PUMP, soliton_mode=fundamental)
    assert cross < same


def test_no_root_in_the_window_is_refused_with_the_window_named():
    with pytest.raises(ToolsError, match=r"300-700 nm.*never changes sign"):
        tools.lambda_rdw(helium(), PUMP, wavelength_limits=(300e-9, 700e-9))
    assert (
        tools.lambda_rdw_all(helium(), PUMP, wavelength_limits=(300e-9, 700e-9)) == ()
    )
    with pytest.raises(ToolsError, match="positive and ordered"):
        tools.lambda_rdw_all(helium(), PUMP, wavelength_limits=(700e-9, 300e-9))


def test_every_root_is_returned_in_order():
    # A quartic dispersion with two zero-dispersion points has a dispersive wave on
    # each side of the pump.
    mode = SpecifiedMode.from_taylor([0.0, 0.0, -1e-26, 0.0, 4e-55], OMEGA0)
    detuning = math.sqrt(12.0 * 1e-26 / 4e-55)
    limits = (
        2.0 * math.pi * C / (OMEGA0 + 1.5 * detuning),
        2.0 * math.pi * C / (OMEGA0 - 1.5 * detuning),
    )
    grid = np.linspace(OMEGA0 - 1.5 * detuning, OMEGA0 + 1.5 * detuning, 7)
    assert np.all(np.isfinite(tools.phase_mismatch(mode, grid, PUMP)))
    roots = tools.lambda_rdw_all(mode, PUMP, wavelength_limits=limits)
    expected = sorted(2.0 * math.pi * C / (OMEGA0 + s * detuning) for s in (1.0, -1.0))
    # The pump itself is a double root and changes no sign, so it is not reported.
    assert roots == pytest.approx(expected, rel=1e-9)
    assert tools.lambda_rdw(mode, PUMP, wavelength_limits=limits) == roots[-1]


@settings(max_examples=6, deadline=None)
@given(pressure=st.floats(1.0, 3.0), radius=st.floats(100e-6, 150e-6))
def test_the_pressure_solvers_invert_the_forward_calculations(pressure, radius):
    mode = MarcatiliMode(radius, "He", pressure, cladding=GLASS)
    target = tools.lambda_rdw(mode, PUMP)
    # brentq stops at xtol = 2e-12 bar; the slope of the residual carries that to the
    # pressure with a condition number of order one.
    assert tools.pressure_rdw(
        radius, "He", target, PUMP, cladding=GLASS
    ) == pytest.approx(pressure, rel=1e-8)
    zero = base.zdw(mode, wavelength_limits=(200e-9, 990e-9))
    # beta2 is a Richardson difference good to 5e-9 (modes.base), which the root
    # inherits through the slope of beta2 in pressure.
    assert tools.pressure_zdw(radius, "He", zero, cladding=GLASS) == pytest.approx(
        pressure, rel=1e-5
    )


def test_the_pressure_with_the_kerr_term_is_higher():
    target = tools.lambda_rdw(helium(), PUMP)
    linear = tools.pressure_rdw(125e-6, "He", target, PUMP, cladding=GLASS)
    kerr = tools.pressure_rdw(
        125e-6,
        "He",
        target,
        PUMP,
        peak_power=2e10,
        nonlinear_phase="supplied_power",
        cladding=GLASS,
    )
    # The Kerr term moves the wave to the blue at fixed pressure, and so does lowering
    # the pressure; to stay on target the pressure has to rise.
    assert linear == pytest.approx(1.0, rel=1e-8)
    assert kerr > linear


def test_a_pressure_bracket_without_a_root_is_refused():
    with pytest.raises(ToolsError, match="no pressure between"):
        tools.pressure_rdw(125e-6, "He", 186e-9, PUMP, pressure_max=0.5, cladding=GLASS)
    with pytest.raises(ToolsError, match="pressure_max must exceed"):
        tools.pressure_zdw(125e-6, "He", 400e-9, pressure_max=1e-7, cladding=GLASS)


# ------------------------------------------------------------- the soliton design


def design(**keywords):
    """A 125 um argon capillary with its zero at 500 nm and a 10 fs pump at 800 nm."""
    return tools.capillary_design(
        125e-6,
        "Ar",
        zdw_wavelength=500e-9,
        fwhm=10e-15,
        wavelength=PUMP,
        cladding=GLASS,
        **keywords,
    )


@pytest.mark.parametrize("shape", ["sech", "gauss"])
def test_the_design_is_what_the_forward_calculation_reports(shape):
    designed = design(soliton_order=3.0, shape=shape)
    forward = tools.capillary_params(
        125e-6,
        "Ar",
        designed.pressure,
        energy=designed.energy,
        fwhm=10e-15,
        wavelength=PUMP,
        shape=shape,
        cladding=GLASS,
    )
    # The energy is a closed form in beta2 and gamma at the pressure, so only
    # rounding separates the order from its target.
    assert designed.soliton_order == pytest.approx(3.0, rel=1e-12)
    assert forward.soliton_order == pytest.approx(3.0, rel=1e-12)
    # pressure_zdw stops at xtol = 2e-12 bar and beta2 is a Richardson difference
    # good to 5e-9 (modes.base); the zero inherits both through slopes of order one.
    assert designed.zdw == pytest.approx(500e-9, rel=1e-6)
    assert designed.pressure == tools.pressure_zdw(125e-6, "Ar", 500e-9, cladding=GLASS)


def test_the_energy_goes_as_the_square_of_the_order_at_one_pressure():
    once, twice = design(soliton_order=1.5), design(soliton_order=3.0)
    assert twice.pressure == once.pressure
    # N^2 is linear in the energy: rounding only.
    assert twice.energy / once.energy == pytest.approx(4.0, rel=1e-12)


@pytest.mark.parametrize("shape", ["sech", "gauss"])
@pytest.mark.parametrize("count", [1, 2, 3])
def test_a_soliton_count_puts_the_area_in_the_middle_of_its_band(shape, count):
    content = design(soliton_count=count, shape=shape).soliton_content
    assert content is not None
    assert content.count == count
    assert content.area_order == pytest.approx(count, rel=1e-12)


@pytest.mark.parametrize("shape", ["sech", "gauss"])
@pytest.mark.parametrize("eigenvalue", [0.05, 0.4207333062232459, 1.0, 9.0])
def test_a_largest_eigenvalue_is_met(shape, eigenvalue):
    content = design(largest_eigenvalue=eigenvalue, shape=shape).soliton_content
    assert content is not None
    # The Gaussian eigenvalue is good to about 1e-10 absolute (soliton_content), and
    # the inverse solve stops at xtol = 1e-12 in N with d(eta)/dN close to one.
    assert content.largest_eigenvalue == pytest.approx(eigenvalue, abs=1e-9)


def test_the_gaussian_inverse_returns_the_documented_unit_order():
    # docs/guide/workflows/design-tools.md: a Gaussian with N = 1 has eta = 0.4207333.
    assert tools._gaussian_order(0.4207333062232459) == pytest.approx(1.0, abs=1e-9)


def test_a_gaussian_inverse_without_a_root_in_its_bracket_is_refused(monkeypatch):
    monkeypatch.setattr(tools, "_gaussian_eigenvalue", lambda order: 0.0)
    with pytest.raises(ToolsError, match="no Gaussian soliton order"):
        tools._gaussian_order(1.0)


def test_a_conventional_order_in_normal_dispersion_is_designed():
    # A zero at 1 um puts the 800 nm pump in normal dispersion, where N is still
    # the conventional measure but there is no soliton content.
    normal = tools.capillary_design(
        125e-6,
        "Ar",
        zdw_wavelength=1000e-9,
        fwhm=10e-15,
        wavelength=PUMP,
        soliton_order=2.0,
        cladding=GLASS,
    )
    assert normal.beta2 > 0.0
    assert normal.soliton_content is None
    assert normal.soliton_order == pytest.approx(2.0, rel=1e-12)


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({}, "got none"),
        ({"soliton_order": 2.0, "soliton_count": 2}, "soliton_order, soliton_count"),
        ({"soliton_order": -1.0}, "soliton_order"),
        ({"soliton_count": 0}, "at least 1"),
        ({"largest_eigenvalue": 0.0}, "largest_eigenvalue"),
    ],
)
def test_an_unusable_design_target_is_refused_by_name(keywords, message):
    with pytest.raises(ToolsError, match=message):
        design(**keywords)


@pytest.mark.parametrize("target", [{"soliton_count": 1}, {"largest_eigenvalue": 0.5}])
def test_a_content_target_in_normal_dispersion_is_refused(target):
    with pytest.raises(ToolsError, match="anomalous dispersion"):
        tools.capillary_design(
            125e-6,
            "Ar",
            zdw_wavelength=1000e-9,
            fwhm=10e-15,
            wavelength=PUMP,
            cladding=GLASS,
            **target,
        )


def test_a_root_outside_a_models_range_still_warns():
    # The search window is probed silently; the answer is not. With Malitson's silica
    # (a caller's own) the helium dispersive wave at 200 nm lies below the 210 nm of
    # the fit; optmat's silica reaches 24.8 nm.
    mode = MarcatiliMode(125e-6, "He", 1.0, cladding=MALITSON)
    with pytest.warns(MaterialRangeWarning, match="SiO2"):
        tools.lambda_rdw(mode, PUMP)


# ------------------------------------------------ agreement with what a run reports


def pole_mode(pole, strength=1e-12):
    """Return a dispersion whose second derivative changes sign through a pole only.

    ``beta = A (w - wp) ln|w - wp|`` has ``beta1 = A (ln|w - wp| + 1)`` and
    ``beta2 = A / (w - wp)``: one sign change in the window and no zero.
    """
    return SpecifiedMode(
        dispersion=lambda w: strength * (w - pole) * np.log(np.abs(w - pole)),
        dispersion_derivative=lambda w: strength * (np.log(np.abs(w - pole)) + 1.0),
        loss=0.0,
        effective_area=1.0,
    )


def test_a_sign_change_through_a_pole_is_not_a_zero_dispersion_point():
    # Placed off the sample grid, so that no sample lands on the singularity itself.
    pole = 2.0 * math.pi * C / 431.7e-9
    assert math.isnan(tools._zdw_near(pole_mode(pole), OMEGA0, 0.0))


def test_a_fill_without_kerr_data_is_refused_by_name():
    inert = replace(species("He"), gamma3=None, kerr=None)
    fill = Mixture.fill({inert: 1.0}, 1.0)
    with pytest.raises(ToolsError, match="no nonlinear index"):
        tools.soliton_params(helium(), fill, energy=1e-6, fwhm=30e-15, wavelength=PUMP)


def test_the_predicted_peak_intensity_is_the_statistic_a_run_reports_at_its_entrance():
    energy, fwhm = 50e-6, 30e-15
    simulation = quiet(
        Simulation.from_capillary,
        125e-6,
        0.01,
        "He",
        1.0,
        wavelength=PUMP,
        wavelength_limits=(400e-9, 990e-9),
        trange=400e-15,
        fwhm=fwhm,
        energy=energy,
        shape="sech",
        cladding=GLASS,
        plasma=False,
        noise=None,
    )
    problem = quiet(simulation.translate)
    backend = NumpyRefBackend(
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    backend.set_initial(problem.initial.field0, None)
    reported = float(
        backend.stats(StatsSpec(names=("peak_intensity",)))["peak_intensity"][0]
    )
    predicted = tools.capillary_params(
        125e-6, "He", 1.0, energy=energy, fwhm=fwhm, wavelength=PUMP, cladding=GLASS
    ).peak_intensity
    # The run's pulse is the sampled analytic signal of a real sech pulse normalised
    # to its energy on the grid: the peak of a 30 fs pulse on a 0.7 fs grid is read to
    # a few parts in ten thousand, and the statistic is the oversampled maximum.
    assert reported == pytest.approx(predicted, rel=2e-3)


def nan_mode():
    """Return a dispersion that is not a number above 3e15 rad/s."""
    return SpecifiedMode(
        dispersion=lambda w: np.where(w < 3.0e15, 1e-26 * (w - OMEGA0) ** 2, np.nan),
        dispersion_derivative=lambda w: np.where(
            w < 3.0e15, 2e-26 * (w - OMEGA0), np.nan
        ),
        loss=0.0,
        effective_area=1.0,
    )


def test_a_window_the_model_does_not_cover_is_refused_rather_than_searched():
    with pytest.raises(ToolsError, match=r"not finite at .* samples"):
        tools.lambda_rdw_all(nan_mode(), PUMP, wavelength_limits=(400e-9, 700e-9))


def test_a_residual_that_is_not_a_number_is_a_tools_error():
    with pytest.raises(ToolsError, match="not finite at the ends"):
        tools._pressure_root(lambda pressure: math.nan, 10.0, "anything")


def test_the_tools_build_the_capillary_the_run_interface_builds():
    # capillary_params cannot import the driver (P against D), so it carries a copy of
    # the fill and mode construction of Simulation.from_capillary; this holds the copy
    # to the original.
    params = quiet(
        tools.capillary_params,
        125e-6,
        "Ar",
        2.0,
        energy=1e-6,
        fwhm=30e-15,
        wavelength=PUMP,
    )
    run = quiet(
        Simulation.from_capillary,
        125e-6,
        0.1,
        "Ar",
        2.0,
        wavelength=PUMP,
        wavelength_limits=(300e-9, 2e-6),
        trange=200e-15,
        fwhm=30e-15,
        energy=1e-6,
        noise=None,
    )
    omega = np.linspace(1.0e15, 4.0e15, 9)
    assert repr(params.mode) == repr(run.modes[0])
    # The same mode built twice: its index evaluated by the same arithmetic on the same
    # fill, a few roundings of n_eff ~ 1 (16 eps).
    np.testing.assert_allclose(
        quiet(params.mode.neff, omega),
        quiet(run.modes[0].neff, omega),
        rtol=16 * np.finfo(float).eps,
        atol=0,
    )


# ------------------------------------------- designs under a material selection (Q7)

SELECTION = MaterialSelection()


def selected_run(gas, pressure, limits, length=1e-3):
    """Four fixed steps of the capillary a design under `SELECTION` names.

    The weak collar keeps four fixed steps of a millimetre inside its stability
    limit, as for SEP 3's baseline witnesses.
    """
    grid = FieldGrid.from_limits(length, PUMP, limits, 200e-15)
    simulation = quiet(
        Simulation.from_capillary,
        125e-6,
        length,
        gas,
        pressure,
        wavelength=PUMP,
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0),
        fwhm=30e-15,
        energy=1e-9,
        plasma=False,
        schedule=FixedSchedule(np.full(4, length / 4)),
        n_save=2,
        backend="numpy_ref",
        materials=SELECTION,
        noise=None,
    )
    output = quiet(simulation.run, progress="none")
    assert output.n_saved == 2
    return simulation


@pytest.mark.medium
def test_a_zero_dispersion_pressure_designed_under_a_selection_holds_in_the_run():
    # SEP 3, Q7: the design and the run under one selection take the same gas index,
    # density and wall. The run's zero-dispersion wavelength is the target to the
    # tool's own accuracy (beta2 is a Richardson difference good to 5e-9).
    target = 600e-9
    pressure = quiet(tools.pressure_zdw, 125e-6, "Ar", target, materials=SELECTION)
    run = selected_run("Ar", pressure, (400e-9, 1.2e-6))
    found = quiet(base.zdw, run.modes[0], wavelength_limits=(400e-9, 990e-9))
    assert found == pytest.approx(target, rel=1e-7)


@pytest.mark.medium
def test_a_phase_matching_pressure_designed_under_a_selection_holds_in_the_run():
    target = 200e-9
    pressure = quiet(
        tools.pressure_rdw, 125e-6, "He", target, PUMP, materials=SELECTION
    )
    run = selected_run("He", pressure, (150e-9, 1.2e-6))
    # brentq's 2e-12 bar carries to the wavelength with a condition number of order
    # one, as for any design.
    assert quiet(tools.lambda_rdw, run.modes[0], PUMP) == pytest.approx(
        target, rel=1e-8
    )
    # With the Kerr term the nonlinear coefficient follows the selected fill.
    kerr = quiet(
        tools.pressure_rdw,
        125e-6,
        "He",
        target,
        PUMP,
        peak_power=2e10,
        nonlinear_phase="supplied_power",
        materials=SELECTION,
    )
    assert kerr > pressure


def test_the_design_numbers_under_a_selection_are_the_selected_fills():
    params = quiet(
        tools.capillary_params,
        125e-6,
        "He",
        0.4,
        energy=300e-6,
        fwhm=10e-15,
        wavelength=PUMP,
        materials=SELECTION,
    )
    assert isinstance(params.mode.cladding, OptmatGlass)
    expected = n2("He", pressure=0.4, selection=SELECTION, wavelength=PUMP)
    assert params.n2 == pytest.approx(expected, rel=1e-12)
    density = species("He", selection=SELECTION).equation_of_state.density(
        0.4, ROOM_TEMPERATURE
    )
    assert params.density == pytest.approx(density, rel=1e-12)
    # materials=None is the same selection.
    default = quiet(
        tools.capillary_params,
        125e-6,
        "He",
        0.4,
        energy=300e-6,
        fwhm=10e-15,
        wavelength=PUMP,
    )
    assert default.n2 == params.n2
    # Air and a fraction mapping resolve as the drivers resolve them.
    air = quiet(
        tools.soliton_params,
        params.mode,
        "air",
        1.0,
        energy=1e-6,
        fwhm=30e-15,
        wavelength=PUMP,
        materials=SELECTION,
    )
    assert air.gas == "N2+O2"
    mapping = quiet(
        tools.soliton_params,
        params.mode,
        {"Ar": 0.5, "He": 0.5},
        1.0,
        energy=1e-6,
        fwhm=30e-15,
        wavelength=PUMP,
        materials=SELECTION,
    )
    assert mapping.gas == "Ar+He"


def test_a_gas_ratio_under_a_selection_takes_optmats_gases():
    ratio = quiet(tools.gas_ratio, "Ar", "He", PUMP, materials=SELECTION)
    chi3 = [
        gas_chi3(gas, pressure=1.0, selection=SELECTION, wavelength=PUMP)
        for gas in ("Ar", "He")
    ]
    assert ratio.chi3 == pytest.approx(chi3[0] / chi3[1], rel=1e-12)
    assert quiet(tools.gas_ratio, "Ar", "He", PUMP).beta2 == ratio.beta2


def test_a_design_under_a_selection_refuses_another_carrier():
    other = MaterialSelection(carrier_wavelength=1030e-9)
    with pytest.raises(ToolsError, match="carrier_wavelength"):
        tools.capillary_params(
            125e-6,
            "He",
            1.0,
            energy=1e-6,
            fwhm=30e-15,
            wavelength=PUMP,
            materials=other,
        )
    with pytest.raises(ToolsError, match="carrier_wavelength"):
        tools.gas_ratio("Ar", "He", PUMP, materials=other)
    with pytest.raises(ToolsError, match="carrier_wavelength"):
        tools.pressure_rdw(125e-6, "He", 200e-9, PUMP, materials=other)
    # The selection's own carrier, when it is the design's, is accepted.
    same = MaterialSelection(carrier_wavelength=PUMP)
    assert quiet(tools.gas_ratio, "Ar", "He", PUMP, materials=same).chi3 > 1


@pytest.mark.medium
def test_air_and_a_tube_lattice_design_under_a_selection():
    air = quiet(tools.pressure_zdw, 125e-6, "Air", 600e-9, materials=SELECTION)
    assert air == quiet(tools.pressure_zdw, 125e-6, "Air", 600e-9)
    assert 0 < air < math.inf
    # A tube lattice is built as prop_capillary builds it under the selection.
    tubes = tools._design_capillary(
        15e-6,
        "Ar",
        temperature=294.0,
        m=1,
        model="full",
        cladding=None,
        lattice=TubeLattice(180e-9, 10.38e-6, 6),
        materials=SELECTION,
    )
    assert isinstance(tubes, VincettiMode)
    assert isinstance(tubes.cladding, OptmatGlass) and tubes.mixture is not None
