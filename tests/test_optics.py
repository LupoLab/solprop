# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Transfer functions: sign convention, closed forms, materials, mirrors, witnesses."""

import json
import math
import warnings
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from solprop import fourier, optics, pulses
from solprop.constants import C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.optics import Chain, ChirpedMirror, Material, TaylorPhase, stored_factor
from solprop.physdata import (
    Mixture,
    SellmeierGlass,
    calcium_fluoride,
    fused_silica,
    glass,
)
from solprop.processing import time
from solprop.processing._examples import example_grid
from solprop.processing.fields import _to_time
from solprop.pulses import GaussianPulse

WAVELENGTH = 800e-9
OMEGA0 = 2.0 * math.pi * C / WAVELENGTH
FWHM = 30e-15
WITNESS = json.loads(
    (Path(__file__).parent / "data" / "luna_tools_optics.json").read_text()
)
# optmat's malitson_1963/sellmeier, 230 nm to 9.7 um: wider than the example grid's
# active band, so no range warning.
WINDOW = calcium_fluoride(model="malitson_1963/sellmeier")


def former_glass(name, wavelength_range, poles):
    """Return a caller's Sellmeier glass: a published expansion, air-relative."""
    return SellmeierGlass(
        name=name,
        source="test data",
        wavelength_range=wavelength_range,
        citation="the published coefficients solprop's former registry carried",
        poles=poles,
    )


# The glasses the Luna witness was recorded against: the published, air-relative
# expansions solprop's former registry carried (dc4ca0d), supplied as a caller's own
# data. solprop's built-in glasses are optmat's and absolute (SEP 3), above these by
# the reference gas's index, 2e-4 to 3e-4.
FORMER_GLASSES = {
    # Malitson (1965), J. Opt. Soc. Am. 55, 1205.
    "SiO2": former_glass(
        "SiO2",
        (0.21e-6, 3.71e-6),
        ((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)),
    ),
    # SCHOTT N-BK7, Zemax catalogue of 2017-01-20.
    "BK7": former_glass(
        "N-BK7",
        (0.3e-6, 2.5e-6),
        (
            (1.03961212, 0.07746417668832478),
            (0.231792344, 0.14148467902921502),
            (1.01046945, 10.176475470417055),
        ),
    ),
    # Daimon and Masumura (2002), Appl. Opt. 41, 5275, 20 C.
    "CaF2": former_glass(
        "CaF2",
        (0.138e-6, 2.326e-6),
        (
            (0.443749998, 0.04219334710591233),
            (0.444930066, 0.0887995529831091),
            (0.150133991, 0.11140892738017003),
            (8.85319946, 52.46219353019849),
        ),
    ),
    # Malitson (1964), J. Opt. Soc. Am. 54, 628.
    "BaF2": former_glass(
        "BaF2",
        (0.2652e-6, 10.346e-6),
        ((0.643356, 0.057789), (0.506762, 0.10968), (3.8261, 46.3864)),
    ),
    # Dodge (1984), Appl. Opt. 23, 1980, ordinary ray.
    "MgF2": former_glass(
        "MgF2",
        (0.2e-6, 7e-6),
        ((0.48755108, 0.04338408), (0.39875031, 0.09461442), (2.3120353, 23.793604)),
    ),
}


def built(kind, **keywords):
    """Return a grid and the stored spectrum of a 30 fs Gaussian on it."""
    grid = example_grid(kind)
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=FWHM, energy=1e-6, **keywords)
    return grid, pulse.build(grid)


def power_after(grid, stored, transfer=None, window=None):
    """Return the time axis and the power after an element and a spectral window."""
    if window is not None:
        stored = stored * window
    t, amplitude = _to_time(
        grid, stored, oversampling=4, bandpass=None, transfer=transfer
    )
    return t, np.sum(np.abs(amplitude) ** 2, axis=(0, 1))


def half_masks(grid):
    """Return the stored-axis masks of the red and the blue half of the spectrum."""
    offset = np.abs(grid.omega) - OMEGA0
    return (offset < 0.0) & grid.active_mask, (offset > 0.0) & grid.active_mask


def phase_derivatives(transfer, omega, step):
    """Return the first and second derivative of an element's phase by differences."""
    phase = np.unwrap(np.angle(transfer(omega + step * np.array([-1.0, 0.0, 1.0]))))
    first = (phase[2] - phase[0]) / (2.0 * step)
    return first, (phase[0] - 2.0 * phase[1] + phase[2]) / step**2


# ------------------------------------------------------------- the sign convention


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_positive_gdd_broadens_a_gaussian_by_the_closed_form(kind):
    grid, stored = built(kind)
    gdd = 400e-30
    t, power = power_after(grid, stored, TaylorPhase((0.0, 0.0, gdd), WAVELENGTH))
    expected = FWHM * math.sqrt(1.0 + (4.0 * math.log(2.0) * gdd / FWHM**2) ** 2)
    # The half-maximum crossings are interpolated on a grid four times finer than the
    # stored one; for a 38 fs pulse that reads the width to a few parts in 1e5.
    assert time.fwhm(t, power) == pytest.approx(expected, rel=2e-4)


@pytest.mark.parametrize("kind", ["envelope", "field"])
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_positive_gdd_makes_the_blue_arrive_later(kind, sign):
    grid, stored = built(kind)
    red, blue = half_masks(grid)
    element = TaylorPhase((0.0, 0.0, sign * 2000e-30), WAVELENGTH)
    arrivals = [
        time.moment(*power_after(grid, stored, element, window=mask))
        for mask in (red, blue)
    ]
    # conventions.md section 2: group delay is +d(phi)/d(omega), so with positive GDD
    # the delay rises with frequency.
    assert sign * (arrivals[1] - arrivals[0]) > 10e-15


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_a_positive_delay_coefficient_arrives_later(kind):
    grid, stored = built(kind)
    delay = 40e-15
    before = time.moment(*power_after(grid, stored))
    after = time.moment(
        *power_after(grid, stored, TaylorPhase((0.0, delay), WAVELENGTH))
    )
    # A pure linear phase is an exact shift of a band-limited pulse.
    assert after - before == pytest.approx(delay, rel=1e-9)


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_a_slab_of_glass_chirps_positively_in_the_visible(kind):
    grid, stored = built(kind)
    red, blue = half_masks(grid)
    slab = Material(WINDOW, 20e-3, WAVELENGTH)
    arrivals = [
        time.moment(*power_after(grid, stored, slab, window=mask))
        for mask in (red, blue)
    ]
    assert arrivals[1] > arrivals[0]


def test_a_chirp_built_into_a_pulse_is_removed_by_the_opposite_taylor_phase():
    grid, clean = built("envelope")
    _, chirped = built("envelope", phase=(0.0, 0.0, 700e-30, 9000e-45))
    undo = TaylorPhase((0.0, 0.0, -700e-30, -9000e-45), WAVELENGTH)
    restored = chirped * stored_factor(grid, undo)
    # One shared phase formula evaluated with opposite coefficients: the two unit
    # phasors multiply to one within a few ulps of a phase of tens of radians.
    np.testing.assert_allclose(
        restored, clean, rtol=0.0, atol=1e-12 * np.abs(clean).max()
    )


def test_the_pulse_phase_factor_is_what_it_was():
    coefficients, delay = np.array([0.3, 0.0, 250e-30, -4000e-45]), 12e-15
    for kind in ("field", "envelope"):
        grid = example_grid(kind)
        # The loop the builders carried before the formula moved to solprop.optics.
        detuning = grid.omega - OMEGA0
        phase = grid.omega * delay
        for order, coefficient in enumerate(coefficients):
            phase = phase + coefficient * detuning**order / math.factorial(order)
        expected = fourier.to_internal(np.exp(1j * phase))
        # The same sum of five terms, in whatever order the helper forms it: a few
        # roundings of the largest phase (16 eps of it), then one complex exponential
        # of unit modulus (a few ulps), compared absolutely, the peak being one.
        bound = 16 * np.finfo(float).eps * float(np.max(np.abs(phase)))
        np.testing.assert_allclose(
            pulses._phase_factor(grid, coefficients, delay, OMEGA0),
            expected,
            rtol=0,
            atol=bound + 8 * np.finfo(float).eps,
        )


def test_stored_factor_is_the_conjugate_on_the_active_band_and_one_elsewhere():
    grid = example_grid("field")
    element = TaylorPhase((0.2, 1e-15, 80e-30), WAVELENGTH)
    factor = stored_factor(grid, element)
    active = grid.active_mask
    # The element evaluated again on the same frequencies, conjugated: the same
    # arithmetic, to a few ulps of its unit modulus. Outside the band the factor is
    # one by construction, exactly.
    np.testing.assert_allclose(
        factor[active],
        np.conj(element(grid.omega[active])),
        rtol=0,
        atol=8 * np.finfo(float).eps,
    )
    assert np.all(factor[~active] == 1.0)


# ----------------------------------------------------------------- invariants


@pytest.mark.parametrize("kind", ["envelope", "field"])
@pytest.mark.parametrize(
    "element",
    [
        TaylorPhase((0.0, 0.0, 300e-30, 5000e-45), WAVELENGTH),
        Material(WINDOW, 5e-3, WAVELENGTH),
        Chain((TaylorPhase((0.0, 0.0, 100e-30), WAVELENGTH), Material(WINDOW, -1e-3))),
    ],
)
def test_phase_only_elements_conserve_energy_and_the_spectrum(kind, element):
    grid, stored = built(kind)
    after = stored * stored_factor(grid, element)
    # |exp(i phi)| is one to an ulp, the complex product adds another, and the modulus
    # a third: a few 1e-16 per bin, and so for the energy, which is their sum.
    np.testing.assert_allclose(np.abs(after), np.abs(stored), rtol=1e-15, atol=0.0)


def test_removing_a_material_undoes_inserting_it():
    omega = np.linspace(0.6, 1.6, 101) * OMEGA0
    forward, back = (
        Material(WINDOW, 3e-3, WAVELENGTH),
        Material(WINDOW, -3e-3, WAVELENGTH),
    )
    # Two unit phasors of opposite phase, each a few thousand radians.
    np.testing.assert_allclose(forward(omega) * back(omega), 1.0, rtol=0.0, atol=1e-12)
    assert np.all(Chain(())(omega) == 1.0)


# ------------------------------------------------------------------- materials


def test_a_material_is_exp_i_beta_l_with_the_group_delay_removed_analytically():
    omega = np.linspace(0.7, 1.5, 9) * OMEGA0
    length = 2e-3
    index = np.real(WINDOW.index(omega))
    reference = float(np.real(WINDOW.index(OMEGA0)))
    slope = float(np.real(WINDOW.dindex_domega(OMEGA0)))
    delay = (reference + OMEGA0 * slope) / C
    expected = np.exp(1j * (omega * index / C - delay * (omega - OMEGA0)) * length)
    # The same expression written twice; the phases are of order 1e4 rad.
    np.testing.assert_allclose(
        Material(WINDOW, length, WAVELENGTH)(omega), expected, rtol=0.0, atol=1e-10
    )
    kept = Material(WINDOW, length)(omega)
    np.testing.assert_allclose(
        kept, np.exp(1j * omega * index * length / C), rtol=0.0, atol=1e-10
    )


def test_the_group_delay_at_the_reference_is_zero_and_the_gdd_is_silicas():
    silica = fused_silica(model="malitson_1965/sellmeier")
    step = 1e-3 * OMEGA0
    delay, gdd = phase_derivatives(Material(silica, 1e-3, WAVELENGTH), OMEGA0, step)
    # A central difference of a phase whose third derivative is ~1e-43 s^3/mm: the
    # residual delay is step^2/6 of that, 1e-19 s, against a group delay of 5e-12 s.
    assert abs(delay) < 1e-18
    # 36.16 fs^2/mm, the textbook figure for fused silica at 800 nm.
    assert gdd == pytest.approx(36.16e-30, rel=1e-3)
    infrared = 2.0 * math.pi * C / 1550e-9
    assert phase_derivatives(Material(silica, 1e-3, 1550e-9), infrared, step)[1] < 0.0


def test_absorption_is_beers_law_and_is_off_by_default():
    silica = fused_silica()
    omega = np.array([2.0 * math.pi * C / 3.4e-6])
    length = 5e-3
    extinction = float(np.imag(silica.index(omega))[0])
    assert extinction > 0.0
    lossy = Material(silica, length, loss=True)(omega)
    assert abs(lossy[0]) ** 2 == pytest.approx(
        math.exp(-2.0 * omega[0] * extinction * length / C), rel=1e-12
    )
    assert abs(Material(silica, length)(omega)[0]) == pytest.approx(1.0, rel=1e-15)


def test_a_metre_of_air_has_about_twenty_femtoseconds_squared():
    air = Material("Air", 1.0, WAVELENGTH, pressure=1.01325)
    _, gdd = phase_derivatives(air, OMEGA0, 1e-2 * OMEGA0)
    # optmat's dry air gives 20.8 fs^2/m; the figure usually quoted is 21.3 fs^2/m at
    # 800 nm and 1 atm (solprop's former Börzsönyi air gave 20.2).
    assert 19.0e-30 < gdd < 22.5e-30


def test_a_gas_path_is_the_same_through_a_mixture():
    omega = np.linspace(0.8, 1.3, 7) * OMEGA0
    named = Material("Ar", 2.0, WAVELENGTH, pressure=1.5)(omega)
    mixed = Material(Mixture.fill({"Ar": 1.0}, 1.5), 2.0, WAVELENGTH)(omega)
    # n = sqrt(1 + chi) formed two ways from one susceptibility; a phase of 2e7 rad.
    np.testing.assert_allclose(mixed, named, rtol=0.0, atol=1e-7)


def test_a_reflectivity_of_exactly_zero_is_a_zero_and_not_a_warning():
    table = np.array([[800.0, 0.0, 50.0], [801.0, 8.0, 2.0]])
    wavelength, mean = optics._mean_columns(table, geometric=True)
    np.testing.assert_allclose(wavelength, [800e-9, 801e-9])
    np.testing.assert_allclose(mean, [0.0, 4.0], rtol=1e-15)


def test_a_material_is_one_where_there_is_no_index():
    response = Material(WINDOW, 1e-3, WAVELENGTH)(np.array([-OMEGA0, 0.0, OMEGA0]))
    assert response[0] == 1.0
    assert response[1] == 1.0
    assert response[2] != 1.0
    assert np.all(Material(WINDOW, 1e-3)(np.array([-1.0, 0.0])) == 1.0)


def test_a_glass_name_is_resolved_and_a_range_is_still_warned_about():
    # optmat's adapters, the defaults since SEP 3's Phase 4a, are equal by model.
    assert Material("caf2", 1e-3).medium == glass("CaF2")
    with pytest.warns(MaterialRangeWarning, match="N-BK7"):
        Material("BK7", 1e-3)(np.array([2.0 * math.pi * C / 250e-9]))


@pytest.mark.parametrize(
    "build",
    [
        lambda: Material(WINDOW, math.inf),
        lambda: Material(WINDOW, 1e-3, -800e-9),
        lambda: Material("Ar", 1.0),
        lambda: Material(WINDOW, 1e-3, pressure=1.0),
        lambda: TaylorPhase((0.0, math.nan), WAVELENGTH),
        lambda: TaylorPhase((0.0,), 0.0),
    ],
)
def test_unusable_elements_are_refused(build):
    with pytest.raises(MaterialError):
        build()


def test_lunas_material_transfer_function_is_the_conjugate():
    record = WITNESS["material"]
    omega = 2.0 * math.pi * C / np.array(record["wavelength"])
    luna = np.array(record["real"]) + 1j * np.array(record["imag"])
    # Luna's lossless transfer function, stated (``docs/dev/testing.md``).
    ours = Material(
        FORMER_GLASSES["SiO2"], record["thickness"], record["reference"], loss=False
    )(omega)
    difference = np.unwrap(np.angle(np.conj(luna) / ours))
    detuning = (omega - OMEGA0) * 1e-15
    constant_and_delay = np.polyval(np.polyfit(detuning, difference, 1), detuning)
    # Luna's :SiO2 is a spline through measured index data, not Malitson's expression,
    # and it differences that spline for the group delay it removes. The two codes
    # therefore differ by a delay (0.85 fs over this millimetre, 1.7e-4 of beta1) and a
    # constant, which are fitted out here, and by the 9e-6 by which the measured index
    # departs from Malitson's at 400 nm, 0.14 rad over a millimetre.
    assert abs(np.polyfit(detuning, difference, 1)[0]) < 1.0
    np.testing.assert_allclose(difference, constant_and_delay, rtol=0.0, atol=0.15)


def test_lunas_glass_indices_where_the_sources_agree():
    wavelengths = np.array(WITNESS["glass_wavelengths"])
    omega = 2.0 * math.pi * C / wavelengths
    cases = (
        # Luna's silica is a spline through measured data; Malitson is within 1e-5.
        ("SiO2", 1e-5),
        # The same published coefficients in both codes.
        ("BK7", 1e-12),
        ("CaF2", 1e-12),
        # Different published fits (Luna: the Li 1980 forms; here Malitson 1964 and
        # Dodge 1984 from the pinned database), agreeing to the fits' own accuracy.
        ("BaF2", 1e-4),
        # Luna's uncited magnesium fluoride fit leaves Dodge's by 3.6e-4 at 1.5 um,
        # twenty times its distance in the visible; which is right there is open.
        ("MgF2", 4e-4),
    )
    # The comparison is of the two codes' published coefficients, relative to air,
    # through solprop's SellmeierGlass.
    for name, tolerance in cases:
        ours = np.real(FORMER_GLASSES[name].index(omega))
        np.testing.assert_allclose(ours, WITNESS["glasses"][name], rtol=tolerance)


# --------------------------------------------------------------------- mirrors


def synthetic(gdd=-60e-30, **keywords):
    wavelength = np.linspace(600e-9, 1000e-9, 801)
    return ChirpedMirror.from_data(
        wavelength,
        keywords.pop("reflectivity", None),
        gdd=np.full(wavelength.size, gdd),
        reference_wavelength=WAVELENGTH,
        band=(650e-9, 950e-9),
        **keywords,
    )


def test_constant_gdd_data_give_a_taylor_phase_in_the_band():
    gdd = -60e-30
    mirror = synthetic(gdd)
    omega = 2.0 * math.pi * C / np.linspace(660e-9, 940e-9, 57)
    expected = TaylorPhase((0.0, 0.0, gdd), WAVELENGTH)(omega)
    # The trapezoid rule is exact for a constant integrand and for the linear function
    # its first integral is, so the phase is the Taylor one to rounding (3e-14 rad
    # measured). The tolerance is there to catch a conjugated element, which
    # fails it by two.
    np.testing.assert_allclose(mirror(omega), expected, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(mirror.gdd(omega), gdd, rtol=1e-6)
    np.testing.assert_allclose(
        mirror.group_delay(omega), gdd * (omega - OMEGA0), atol=1e-19
    )


def test_a_dispersionless_mirror_is_a_mirror_and_not_an_index_error():
    # A flat zero phase fits to a polynomial whose trailing zeros numpy trims away.
    wavelength = np.linspace(600e-9, 1000e-9, 401)
    mirror = ChirpedMirror.from_data(
        wavelength,
        np.full(401, 0.25),
        gdd=np.zeros(401),
        reference_wavelength=WAVELENGTH,
        band=(650e-9, 950e-9),
    )
    omega = 2.0 * math.pi * C / np.linspace(670e-9, 930e-9, 9)
    np.testing.assert_allclose(mirror(omega), 0.5, rtol=1e-14)


def test_group_delay_data_give_the_same_mirror():
    wavelength = np.linspace(600e-9, 1000e-9, 801)
    gdd = -60e-30
    delay = gdd * (2.0 * math.pi * C / wavelength - OMEGA0) + 3e-15
    mirror = ChirpedMirror.from_data(
        wavelength,
        None,
        group_delay=delay,
        reference_wavelength=WAVELENGTH,
        band=(650e-9, 950e-9),
    )
    omega = 2.0 * math.pi * C / np.linspace(660e-9, 940e-9, 23)
    # The constant 3 fs of delay is one of the two integration constants removed.
    np.testing.assert_allclose(
        mirror(omega), synthetic(gdd)(omega), rtol=0.0, atol=1e-4
    )


def test_reflections_multiply_and_the_inverse_is_phase_only():
    reflectivity = np.full(801, 0.98)
    mirror = synthetic(reflectivity=reflectivity)
    inside = 2.0 * math.pi * C / np.linspace(670e-9, 930e-9, 17)
    outside = 2.0 * math.pi * C / np.array([500e-9, 620e-9, 990e-9, 1200e-9])
    three = mirror.bounces(3)
    np.testing.assert_allclose(three(inside), mirror(inside) ** 3, rtol=1e-13)
    np.testing.assert_allclose(np.abs(three(inside)), 0.98**1.5, rtol=1e-13)
    undone = three(inside) * mirror.bounces(-3)(inside)
    # The phases cancel to rounding and the inverse carries no amplitude.
    np.testing.assert_allclose(undone, 0.98**1.5, rtol=0.0, atol=1e-13)
    assert np.all(mirror.bounces(0)(outside) == 1.0)
    # Out of band both directions return zero: Luna's inverse returns one there.
    assert np.all(three(outside) == 0.0)
    assert np.all(mirror.bounces(-3)(outside) == 0.0)
    assert np.all(mirror(np.array([-OMEGA0, 0.0])) == 0.0)


def test_the_taper_falls_smoothly_between_the_band_and_its_edge():
    mirror = synthetic(taper_width=30e-9)
    edge = 2.0 * math.pi * C / np.array([955e-9, 965e-9, 975e-9])
    modulus = np.abs(mirror(edge))
    assert 1.0 > modulus[0] > modulus[1] > modulus[2] > 0.0


@pytest.mark.parametrize("name", optics.chirped_mirrors())
def test_every_shipped_set_reproduces_lunas_transfer_function(name):
    record = WITNESS["mirrors"][name]
    omega = 2.0 * math.pi * C / np.array(record["wavelength"])
    luna = np.array(record["real"]) + 1j * np.array(record["imag"])
    ours = ChirpedMirror.lookup(name)(omega)
    # Same data, same integration, same fit and the same amplitude spline, so the
    # moduli agree to rounding at every witness point (7e-16 measured; no
    # shipped set's spline exceeds one anywhere). The phase spline is in omega here and
    # Luna's B-spline is in wavelength, which is felt at PC70's ragged short-wavelength
    # edge, 2.3e-4 rad, and at 4e-7 rad or below on the other six sets.
    np.testing.assert_allclose(np.abs(ours), np.abs(luna), rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        np.conj(luna) / np.abs(luna), ours / np.abs(ours), rtol=0.0, atol=5e-4
    )


@pytest.mark.parametrize("name", optics.chirped_mirrors())
def test_a_shipped_set_has_a_band_a_name_and_a_physical_reflectivity(name):
    mirror = ChirpedMirror.lookup(name)
    assert mirror.name == name
    assert mirror.reflections == 1
    zero_low, one_low, one_high, zero_high = mirror.band
    assert zero_low < one_low < one_high < zero_high
    omega = 2.0 * math.pi * C / np.linspace(one_low, one_high, 101)
    modulus = np.abs(mirror(omega))
    # One ulp above unity is the unit phasor's own rounding.
    assert np.all(modulus <= 1.0 + 4e-16)
    assert float(np.median(modulus)) > 0.99


@pytest.mark.parametrize(
    "build",
    [
        lambda: ChirpedMirror.lookup("HD60"),
        lambda: ChirpedMirror(
            np.array([4.0, 3.0, 2.0, 1.0]) * 1e15,
            np.zeros(4),
            np.empty(0),
            np.empty(0),
            (600e-9, 650e-9, 950e-9, 1000e-9),
        ),
        lambda: synthetic(group_delay=np.zeros(801)),
        lambda: ChirpedMirror.from_data(
            np.linspace(600e-9, 1000e-9, 11),
            None,
            reference_wavelength=WAVELENGTH,
            band=(650e-9, 950e-9),
        ),
        lambda: synthetic(reflectivity=np.full(801, 1.2)),
        lambda: ChirpedMirror.from_data(
            np.array([700e-9, 800e-9, 750e-9, 900e-9, 950e-9]),
            None,
            gdd=np.zeros(5),
            reference_wavelength=WAVELENGTH,
            band=(720e-9, 900e-9),
        ),
        lambda: ChirpedMirror.from_data(
            np.linspace(700e-9, 900e-9, 3),
            None,
            gdd=np.zeros(3),
            reference_wavelength=WAVELENGTH,
            band=(720e-9, 880e-9),
        ),
        lambda: ChirpedMirror.from_data(
            np.linspace(700e-9, 900e-9, 9),
            np.full(8, 0.9),
            gdd=np.zeros(9),
            reference_wavelength=WAVELENGTH,
            band=(720e-9, 880e-9),
        ),
    ],
)
def test_unusable_mirror_data_are_refused(build):
    with pytest.raises(MaterialError):
        build()


def test_an_unreadable_data_row_is_refused_with_the_file_named():
    record = optics._MIRRORS["HD120"]
    good = "header\n900,81.5,900,-2585.9\n"
    table = optics._parse_columns(good, "x.csv", record, (0, 1))
    assert table.tolist() == [[900.0, 81.5]]
    with pytest.raises(MaterialError, match=r"x\.csv: unreadable data row"):
        optics._parse_columns(good + "901,,901,-2600\n", "x.csv", record, (0, 1))
    with pytest.raises(MaterialError, match="unreadable"):
        optics._parse_columns(good + "901\n", "x.csv", record, (0, 1))


def test_the_shipped_files_carry_their_provenance():
    # Through the resource API `optics._read_columns` uses, so that the nine files are
    # found where an installed package looks for them, not only beside the source.
    folder = resources.files("solprop.physdata") / "data" / "mirrors"
    files = sorted(entry.name for entry in folder.iterdir() if entry.name != "NOTICE")
    assert len(files) == 9
    # The data sets are Luna.jl's, redistributed under its MIT licence.
    notice = (folder / "NOTICE").read_text(encoding="utf-8")
    assert "Luna.jl" in notice and "MIT License" in notice
    for name in files:
        head = (folder / name).read_text(encoding="utf-8").splitlines()[:7]
        assert all(line.startswith("# ") for line in head)
        keys = [line[2:].split(":")[0] for line in head]
        assert keys == [
            "mirror",
            "source",
            "source-sha256",
            "columns",
            "origin",
            "changes",
            "generated-by",
        ]


def test_nothing_here_warns_about_the_window_glass():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        grid, stored = built("field")
        power_after(grid, stored, Material(WINDOW, 1e-3, WAVELENGTH))
