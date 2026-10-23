# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Gas linear susceptibility: optmat's built-in laws and a caller's supplied ones.

Every built-in gas law is optmat's (SEP 3; the adapter's agreement with optmat itself
is in test_optmat_gases.py). These tests hold the built-in route to the physics a gas
index must have -- above one, normally dispersive in the visible, linear in density,
``n - 1`` without the cancellation -- and the supplied law types
(:class:`PoleSumSusceptibility`, :class:`InversePoleIndex`,
:class:`WavenumberPoleIndex`) to their closed forms, on the published fits of
tests/physdata/former_gas_laws.py.
"""

import math
import re
import warnings

import numpy as np
import pytest
from former_gas_laws import (
    ALL_LAWS,
    ATM_IN_BAR,
    HE_BORZSONYI,
    HE_ERMOLOV,
    QUANFU_HE_COEFFICIENTS,
    quanfu_he,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from solprop.constants import C
from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.physdata import (
    Gas,
    GasSusceptibility,
    chi1,
    dchi1_domega,
    density,
    drefractive_index_domega,
    gamma,
    refractive_index,
    refractive_index_minus_one,
)

GASES = list(Gas)
# Wavelengths inside every built-in gas's qualified intervals: the infrared-active
# molecules' first window starts at 320 nm (N2O) and every fit reaches 1123 nm.
INSIDE = np.array([350e-9, 532e-9, 800e-9, 1030e-9])
MID_BAND = 700e-9

LAW_IDS = [f"{law.gas.value}-{law.source}" for law in ALL_LAWS]


def omega_of(wavelength):
    return 2.0 * math.pi * C / np.asarray(wavelength, dtype=float)


def mid_band_omega(model):
    """An angular frequency at the geometric centre of a law's validity range."""
    shortest, longest = model.wavelength_range
    return float(omega_of(math.sqrt(shortest * longest)))


# ------------------------------------------------------- the built-in route


@pytest.mark.parametrize("gas", GASES)
def test_the_built_in_law_is_optmats(gas):
    model = gamma(gas)
    assert isinstance(model, GasSusceptibility)
    assert model.gas is gas
    assert "optmat" in model.citation


@pytest.mark.parametrize("gas", GASES)
def test_index_is_above_one_and_normally_dispersive(gas):
    index = refractive_index(gas, omega_of(INSIDE), pressure=1.0)
    assert np.all(index > 1.0)
    # Normal dispersion: n rises towards shorter wavelengths everywhere in the
    # visible and near infrared, below every gas's first resonance.
    assert np.all(np.diff(index) < 0.0)
    assert np.all(drefractive_index_domega(gas, omega_of(INSIDE), pressure=1.0) > 0.0)


@pytest.mark.parametrize("gas", GASES)
def test_susceptibility_is_the_law_times_the_density(gas):
    # The whole point of the per-particle form: one spectral profile, scaled by rho.
    model = gamma(gas)
    omega = omega_of(MID_BAND)
    profile = float(model.susceptibility(omega))
    for bar in (0.1, 1.0, 10.0):
        got = float(chi1(gas, omega, pressure=bar))
        # One product of the same two numbers: exact to a rounding.
        assert got == pytest.approx(profile * density(gas, bar), rel=1e-15)


@pytest.mark.parametrize("gas", GASES)
def test_chi1_is_linear_in_number_density(gas):
    omega = omega_of(MID_BAND)
    low = float(chi1(gas, omega, pressure=0.01))
    high = float(chi1(gas, omega, pressure=0.02))
    ratio = density(gas, 0.02) / density(gas, 0.01)
    # A ratio of two products sharing one factor: a few roundings.
    assert high / low == pytest.approx(ratio, rel=1e-14)


@pytest.mark.parametrize("gas", GASES)
def test_refractive_index_minus_one_avoids_the_cancellation(gas):
    omega = omega_of(MID_BAND)
    direct = float(refractive_index_minus_one(gas, omega, pressure=1.0))
    naive = float(refractive_index(gas, omega, pressure=1.0)) - 1.0
    # Both are correct to their own accuracy; the naive form loses the low bits of a
    # ~1e-4 quantity formed from a number near 1, so they agree only to ~1e-12.
    assert direct == pytest.approx(naive, rel=1e-11)
    # The direct form is the one that stays exact as the pressure falls: at 1e-6 bar
    # n - 1 is ~1e-10 and the subtraction would keep about six figures.
    tiny = float(refractive_index_minus_one(gas, omega, pressure=1e-6))
    assert tiny == pytest.approx(
        direct * density(gas, 1e-6) / density(gas, 1.0), rel=1e-9
    )


@pytest.mark.parametrize("gas", GASES)
def test_scalar_and_array_arguments_agree(gas):
    omega = omega_of(MID_BAND)
    scalar = refractive_index(gas, omega, pressure=1.0)
    array = refractive_index(gas, np.full(4, omega), pressure=1.0)
    assert scalar.shape == ()
    assert array.shape == (4,)
    # The vectorised and the scalar evaluation of one closed form: a vector loop may
    # round its few operations differently, held to 4 eps of an index near one.
    np.testing.assert_allclose(array, float(scalar), rtol=4 * np.finfo(float).eps)


@pytest.mark.parametrize("gas", GASES)
def test_index_derivative_follows_from_the_susceptibility_derivative(gas):
    omega = omega_of(MID_BAND)
    index = float(refractive_index(gas, omega, pressure=3.0))
    dchi = float(dchi1_domega(gas, omega, pressure=3.0))
    got = float(drefractive_index_domega(gas, omega, pressure=3.0))
    assert got == pytest.approx(dchi / (2.0 * index), rel=1e-15)


@pytest.mark.parametrize("gas", list(QUANFU_HE_COEFFICIENTS))
def test_the_built_in_index_of_the_quanfu_he_gases_is_a_gas_index(gas):
    # Luna PhysData.jl:138 returns n / rho in place of (n^2 - 1) / rho for these three
    # gases and so reports n = 1.4037 for methane at 1 bar instead of 1.0004; the
    # built-in route must give a gas's index.
    index = float(refractive_index(gas, omega_of(500e-9), pressure=1.0))
    assert 1.0 < index < 1.001
    assert index != pytest.approx(1.4037, rel=1e-3)


# ------------------------------------------------------------- range warnings


def test_warning_names_the_range_and_the_model():
    with pytest.warns(MaterialRangeWarning) as record:
        refractive_index(Gas.Xe, omega_of(100e-9), pressure=1.0)
    message = str(record[0].message)
    assert "148 nm" in message
    assert "fit/Xe" in message


def test_a_single_warning_is_raised_for_a_whole_array():
    with pytest.warns(MaterialRangeWarning) as record:
        refractive_index(
            Gas.Xe, omega_of(np.geomspace(100e-9, 3e-6, 512)), pressure=1.0
        )
    assert len(record) == 1


# ----------------------------------------------------------------- error paths


def test_unknown_gas_is_rejected():
    with pytest.raises(MaterialError, match="unknown gas"):
        gamma("Argon")
    with pytest.raises(MaterialError, match="unknown gas"):
        refractive_index("Argon", 1e15, pressure=1.0)


@pytest.mark.parametrize("bad", [-1.0, np.array([1e15, -1e15])])
def test_negative_frequency_is_rejected(bad):
    with pytest.raises(MaterialError, match="must not be negative"):
        refractive_index(Gas.Ar, bad, pressure=1.0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, np.array([1e15, np.nan])])
def test_non_finite_frequency_is_rejected(bad):
    with pytest.raises(MaterialError, match="must be finite"):
        refractive_index(Gas.Ar, bad, pressure=1.0)


@pytest.mark.parametrize("bad", [1e15 + 1j, np.array([1e15 + 0j])])
def test_complex_frequency_is_rejected(bad):
    # numpy would cast a complex argument to float with a ComplexWarning rather than
    # refusing it, so it is caught at the boundary instead.
    with pytest.raises(MaterialError, match="must be real, got a complex value"):
        refractive_index(Gas.Ar, bad, pressure=1.0)


@pytest.mark.parametrize("bad", ["800 nm", ["a", "b"]])
def test_non_numeric_frequency_is_rejected(bad):
    with pytest.raises(MaterialError, match="must be real"):
        refractive_index(Gas.Ar, bad, pressure=1.0)


def test_extrapolation_onto_a_pole_is_rejected():
    # optmat's xenon fit, qualified from 148 nm, has a pole near 119.7 nm. Just past
    # it the susceptibility at 50 bar is about -1300 and the index would be
    # imaginary; a NaN must not escape into a linear operator.
    with pytest.warns(MaterialRangeWarning), pytest.raises(MaterialError, match="pole"):
        refractive_index(Gas.Xe, omega_of(119.72e-9), pressure=50.0)


# ------------------------------------------------------------------- properties


@settings(max_examples=40, deadline=None)
@given(
    pressure=st.floats(min_value=1e-3, max_value=50.0),
    fraction=st.floats(min_value=0.0, max_value=1.0),
)
def test_index_is_above_one_and_grows_with_pressure(pressure, fraction):
    omega = omega_of(INSIDE[0] * (INSIDE[-1] / INSIDE[0]) ** fraction)
    low = float(refractive_index(Gas.Ar, omega, pressure=pressure))
    high = float(refractive_index(Gas.Ar, omega, pressure=pressure * 1.5))
    assert 1.0 < low < high


def _domain_top_bar(gas, temperature):
    """Return the top of a gas's single-phase domain at a temperature, in bar."""
    from optmat.gas.eos import air_equation_of_state, equation_of_state  # noqa: PLC0415

    eos = air_equation_of_state() if gas is Gas.Air else equation_of_state(gas.value)
    return eos.phase_domain(temperature)[1] / 1e5


@settings(max_examples=300, deadline=None)
@given(
    gas_index=st.integers(min_value=0, max_value=len(GASES) - 1),
    fraction=st.floats(min_value=0.0, max_value=1.0),
    pressure_fraction=st.floats(min_value=1e-6, max_value=1.0),
    temperature=st.floats(min_value=250.0, max_value=500.0),
)
def test_index_is_physical_over_the_whole_parameter_space(
    gas_index, fraction, pressure_fraction, temperature
):
    """No gas state produces a non-physical or non-finite index.

    Each gas is drawn up to 100 bar or 1 % below the top of its own gas domain at the
    drawn temperature, whichever is lower: optmat's equations of state refuse a
    condensed state (sulfur hexafluoride's vapour pressure at 250 K is about 8 bar),
    while helium, argon or nitrogen are still exercised to 100 bar.
    """
    gas = GASES[gas_index]
    pressure = pressure_fraction * min(100.0, 0.99 * _domain_top_bar(gas, temperature))
    omega = omega_of(INSIDE[0] * (INSIDE[-1] / INSIDE[0]) ** fraction)
    index = float(
        refractive_index(gas, omega, pressure=pressure, temperature=temperature)
    )
    refractivity = float(
        refractive_index_minus_one(
            gas, omega, pressure=pressure, temperature=temperature
        )
    )
    assert math.isfinite(index)
    assert index > 1.0
    assert refractivity > 0.0
    # The naive subtraction is right only to a few ulps of 1.0, i.e. to an absolute
    # 1e-16, which at 1e-4 bar is a relative error of 1e-8 on a refractivity of 3e-8.
    # Asserting an absolute agreement rather than a relative one is the statement that
    # the direct form loses nothing the subtraction does not.
    assert abs(refractivity - (index - 1.0)) < 4.0 * np.finfo(float).eps


# ------------------------------------------------------------ supplied laws


@pytest.mark.parametrize("law", ALL_LAWS, ids=LAW_IDS)
def test_a_supplied_law_declares_a_range_a_state_and_a_citation(law):
    shortest, longest = law.wavelength_range
    assert 0.0 < shortest < longest
    assert law.reference_state.pressure_bar > 0.0
    assert law.reference_state.temperature > 0.0
    assert "doi:" in law.citation or "Luna" in law.citation
    assert not law.absorbing


@pytest.mark.parametrize("law", ALL_LAWS, ids=LAW_IDS)
def test_a_supplied_reference_density_is_the_state_of_the_source(law):
    assert law.reference_density == density(
        law.gas, law.reference_state.pressure_bar, law.reference_state.temperature
    )


@pytest.mark.parametrize("law", ALL_LAWS, ids=LAW_IDS)
def test_a_supplied_susceptibility_is_finite_at_zero_frequency(law):
    # u = (k omega)^2, so the static limit is the sum of the pole strengths and needs
    # no special case. A formulation in wavelength would divide by zero here.
    with pytest.warns(MaterialRangeWarning):
        static = float(law.susceptibility(0.0))
    assert math.isfinite(static)
    assert static > 0.0


# Central differences have error (h^2/6)|f'''| + eps|f|/h. With a relative step
# h = r * omega the truncation term falls as r^2 and the round-off term rises as 1/r.
# Measured over every law here the worst relative error is 3.2e-7 at r = 1e-3 and
# 3.2e-9 at r = 1e-4, i.e. exactly second order, with the floor reached near r = 1e-5
# at 7e-10. These gases disperse weakly (gamma varies by ~1 % across the band), so the
# floor is eps/(r * omega f'/f) ~ 2e-9 rather than eps/r.
DERIVATIVE_BOUNDS = {1e-3: 1e-6, 1e-4: 1e-8, 1e-5: 1e-8}


@pytest.mark.parametrize("law", ALL_LAWS, ids=LAW_IDS)
def test_analytic_derivative_matches_central_differences(law):
    omega = mid_band_omega(law)
    analytic = float(law.dsusceptibility_domega(omega))
    for step, bound in DERIVATIVE_BOUNDS.items():
        h = step * omega
        difference = float(law.susceptibility(omega + h)) - float(
            law.susceptibility(omega - h)
        )
        assert difference / (2.0 * h) == pytest.approx(analytic, rel=bound)


@pytest.mark.parametrize("law", ALL_LAWS, ids=LAW_IDS)
def test_derivative_error_falls_as_the_square_of_the_step(law):
    # The gate that distinguishes a correct analytic derivative from a nearly correct
    # one: a wrong closed form leaves a constant error that does not shrink with h.
    omega = mid_band_omega(law)
    analytic = float(law.dsusceptibility_domega(omega))

    def error(step):
        h = step * omega
        difference = float(law.susceptibility(omega + h)) - float(
            law.susceptibility(omega - h)
        )
        return abs(difference / (2.0 * h) / analytic - 1.0)

    ratio = error(1e-3) / error(1e-4)
    assert 70.0 < ratio < 150.0


@pytest.mark.parametrize("law", ALL_LAWS, ids=LAW_IDS)
def test_analytic_derivative_holds_across_the_whole_range(law):
    """Check the closed-form derivative everywhere, not only at mid band.

    Sweeping the range catches a derivative that is right in the middle and wrong near
    a pole -- which is where an algebraic slip in a rational function would show up.
    A plain central difference is not accurate enough here: its truncation error is
    (h^2 / 6) times the third derivative, which diverges towards the pole just below
    the short-wavelength bound. Richardson extrapolation of two steps cancels that h^2
    term and leaves a fourth-order estimate, which agrees with the closed form to
    1.0e-9 at worst over every law and every wavelength in range; 5e-9 keeps a factor
    of five in hand.
    """
    shortest, longest = law.wavelength_range
    # Inset from the ends so the step does not cross the validity bound and warn.
    omega = omega_of(np.geomspace(shortest * 1.01, longest * 0.99, 200))
    analytic = law.dsusceptibility_domega(omega)

    def central(step):
        return (law.susceptibility(omega + step) - law.susceptibility(omega - step)) / (
            2.0 * step
        )

    step = 1e-4 * omega
    richardson = (4.0 * central(step / 2.0) - central(step)) / 3.0
    assert np.max(np.abs(richardson / analytic - 1.0)) < 5e-9


@pytest.mark.parametrize("law", ALL_LAWS, ids=LAW_IDS)
def test_out_of_range_wavelengths_warn(law):
    shortest, longest = law.wavelength_range
    for wavelength in (shortest * 0.5, longest * 2.0):
        with pytest.warns(MaterialRangeWarning, match=law.gas.value):
            law.susceptibility(omega_of(wavelength))
        with pytest.warns(MaterialRangeWarning, match=re.escape(law.source)):
            law.dsusceptibility_domega(omega_of(wavelength))


@pytest.mark.parametrize("law", ALL_LAWS, ids=LAW_IDS)
def test_in_range_wavelengths_do_not_warn(law):
    shortest, longest = law.wavelength_range
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        law.susceptibility(omega_of(np.geomspace(shortest, longest, 8)))


def mansfield_peck_helium(wavelength):
    """Helium refractivity at 1 atm and 273.15 K, an independent primary source.

    C. R. Mansfield and E. R. Peck, J. Opt. Soc. Am. 59, 199 (1969),
    doi:10.1364/JOSA.59.000199, valid 0.4801-2.0586 um. Not used by any law here; it
    is purely an arbiter.
    """
    micron = wavelength * 1e6
    return 0.01470091 / (423.98 - 1.0 / micron**2)


@pytest.mark.parametrize("law", [HE_ERMOLOV, HE_BORZSONYI, None], ids=str)
def test_helium_matches_an_independent_measurement(law):
    """An end-to-end absolute check against a source no law here uses.

    Coefficients, reference state, equation of state and the n - 1 formation
    together. The two supplied fits sit within 0.25 % of Mansfield and Peck over its
    whole range; normalising either by the 1 bar rather than the 1 atm density would
    move the Ermolov expansion to +1.2 %, well outside this bound. optmat's built-in
    helium (``None``) sits within it too: 0.24 % at worst, measured.
    """
    wavelengths = np.geomspace(480.1e-9, 1000e-9, 12)
    if law is None:
        got = refractive_index_minus_one(
            Gas.He, omega_of(wavelengths), pressure=ATM_IN_BAR, temperature=273.15
        )
    else:
        chi = law.susceptibility(omega_of(wavelengths)) * density(
            Gas.He, ATM_IN_BAR, 273.15
        )
        got = chi / (np.sqrt(1.0 + chi) + 1.0)
    assert np.all(np.abs(got / mansfield_peck_helium(wavelengths) - 1.0) < 2.5e-3)


@pytest.mark.parametrize("gas", list(QUANFU_HE_COEFFICIENTS))
def test_a_wavenumber_pole_law_reproduces_the_published_refractivity(gas):
    """He et al. 2021 tabulate ``(n - 1) 1e8 = a + b / (c - nu^2)``, nu in cm^-1.

    At the paper's own state, 1 atm and 288.15 K, re-deriving the refractivity
    straight from the paper checks the whole chain of the supplied type -- the
    wavenumber conversion, the ``(1 + dn)^2 - 1`` step Luna omits, the reference
    density and the density scaling -- against the published expression rather than
    against our own arithmetic. The only difference from the published number is the
    ``(1 + dn)^2 - 1`` round trip, exact to rounding.
    """
    a, b, c, _ = QUANFU_HE_COEFFICIENTS[gas]
    wavelength = 500e-9
    wavenumber = 1e-2 / wavelength
    published = 1e-8 * (a + b / (c - wavenumber**2))
    law = quanfu_he(gas)
    chi = float(law.susceptibility(omega_of(wavelength))) * density(
        gas, ATM_IN_BAR, 288.15
    )
    got = chi / (math.sqrt(1.0 + chi) + 1.0)
    assert got == pytest.approx(published, rel=1e-12)
