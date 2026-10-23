# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The rate protocol, its conventions and the ionisation diagnostics.

solprop keeps no ionisation-rate formula of its own (SEP 3, D1, Phase 4b): the rates
themselves are optmat's (``tests/nonlinear/test_optmat_rates.py``). What is tested here
is what the package still owns -- the protocol a rate meets, the optical-cycle
convention and species a rate declares, the ceiling, and the diagnostics
:func:`keldysh`, :func:`barrier_suppression_field` and :func:`ionisation_fraction`.
Where a fixed closed-form rate is wanted for the numerics, it is the test-only ADK
oracle (``tests/rate_oracles.py``).
"""

import math

import numpy as np
import pytest

from rate_oracles import adk_oracle
from solprop.constants import AU_EFIELD, AU_ENERGY, AU_TIME, C
from solprop.exceptions import MaterialError
from solprop.nonlinear.ionisation import (
    MAX_RATE,
    EvaluableRate,
    IonisationRate,
    barrier_suppression_field,
    ionisation_fraction,
    keldysh,
    rate_convention,
    rate_species,
)
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.physdata import Gas, ionisation_potential, species
from solprop.translate.ionisation import build_rate_table

NOBLE = [Gas.He, Gas.Ne, Gas.Ar, Gas.Kr, Gas.Xe]
NOBLE_IDS = [gas.value for gas in NOBLE]

WAVELENGTH = 800e-9


def omega0_au(wavelength=WAVELENGTH):
    """Driving angular frequency in atomic units."""
    return 2.0 * math.pi * C / wavelength * AU_TIME


def anonymous_rate(field_amplitude):
    """A rate that declares nothing: no convention, species or energy."""
    return np.zeros_like(np.asarray(field_amplitude, dtype=float))


# ------------------------------------------------------------------- the protocol


@pytest.mark.parametrize("gas", NOBLE, ids=NOBLE_IDS)
def test_optmats_rate_satisfies_the_rate_protocol(gas):
    rate = OptmatRate(gas, WAVELENGTH)
    assert isinstance(rate, IonisationRate)
    # The narrow protocol a consumer needs is satisfied too, and by anything else that
    # maps a field to a rate: that is what lets the plasma response accept a table
    # from the translation layer without the physics layer importing it.
    assert isinstance(rate, EvaluableRate)
    assert rate.threshold > 0.0
    assert rate.maximum == MAX_RATE
    assert len(rate.citation) > 40


def test_a_rate_table_is_evaluable_but_not_a_full_ionisation_rate():
    table = build_rate_table(adk_oracle("He"), (2e9, 1e11), rtol=1e-3)
    assert isinstance(table, EvaluableRate)
    # It carries no threshold, ceiling, citation or parameters of its own, so it is
    # deliberately not the full protocol the table builder itself consumes.
    assert not isinstance(table, IonisationRate)


def test_any_callable_is_evaluable():
    assert isinstance(anonymous_rate, EvaluableRate)
    assert not isinstance(anonymous_rate, IonisationRate)


def test_the_ceiling_is_one_ionisation_per_atomic_unit_of_time():
    assert MAX_RATE == 1.0 / AU_TIME
    # 4.13e16 s^-1: far above any quasistatic rate that means anything.
    assert 4.1e16 < MAX_RATE < 4.2e16


# ---------------------------------------------------- conventions and the species


def test_the_convention_is_read_as_declared_and_never_guessed():
    assert rate_convention(OptmatRate("Ar", WAVELENGTH)) == "instantaneous-isotropic"
    assert rate_convention(adk_oracle("Ar")) == "instantaneous-isotropic"
    assert rate_convention(adk_oracle("Ar", cycle_averaged=True)) == "cycle-averaged"
    # Nothing declared, or something unrecognised, is unspecified rather than assumed.
    assert rate_convention(anonymous_rate) == "unspecified"

    class Odd:
        """A rate declaring a convention the package does not know."""

        convention = "per-half-cycle"

        def __call__(self, field_amplitude):
            return anonymous_rate(field_amplitude)

    assert rate_convention(Odd()) == "unspecified"


def test_the_species_is_read_from_whatever_the_rate_declares():
    # optmat's rate names its species and its energy per event.
    optmat = OptmatRate("Ar", WAVELENGTH)
    assert rate_species(optmat) == ("Ar", optmat.ionisation_potential)
    # A caller's rate with a species name and a potential (the oracle's are the
    # historical inputs, 0.5792 hartree for argon).
    assert rate_species(adk_oracle("Ar")) == ("Ar", 0.5792 * AU_ENERGY)
    # A table keeps what its source rate declared.
    table = build_rate_table(adk_oracle("He"), (2e9, 1e11), rtol=1e-3)
    assert rate_species(table) == ("He", 0.9036 * AU_ENERGY)
    # An anonymous callable declares neither; its response supplies the material.
    assert rate_species(anonymous_rate) == (None, None)


@pytest.mark.parametrize("gas", ["Ar", Gas.Ar])
def test_a_rate_naming_its_gas_takes_the_species_potential(gas):
    class Named:
        """A caller's rate that names a gas and nothing else."""

        def __init__(self, gas):
            self.gas = gas

        def __call__(self, field_amplitude):
            return anonymous_rate(field_amplitude)

    assert rate_species(Named(gas)) == ("Ar", ionisation_potential("Ar"))
    argon = species("Ar")
    assert argon.ionisation is not None
    assert rate_species(Named(argon)) == ("Ar", argon.ionisation.potential)


# -------------------------------------------------------- barrier suppression


def test_barrier_suppression_field_is_the_classical_value():
    # E_b = Ip^2 / (4 Z) in atomic units: the field at which the top of the Coulomb
    # barrier is pulled down to the level.
    fields = {}
    for gas in NOBLE:
        ip_au = ionisation_potential(gas) / AU_ENERGY
        fields[gas] = barrier_suppression_field(ionisation_potential(gas), 1.0)
        assert fields[gas] == pytest.approx(ip_au**2 / 4.0 * AU_EFIELD, rel=1e-15)
    # Helium binds hardest, so it needs the strongest field.
    assert fields[Gas.He] == max(fields.values())
    # Twice the charge halves it: the barrier is pulled down twice as fast.
    argon = ionisation_potential(Gas.Ar)
    assert barrier_suppression_field(argon, 2.0) == pytest.approx(
        barrier_suppression_field(argon, 1.0) / 2.0, rel=1e-15
    )


# ------------------------------------------------------------- the helpers


def test_keldysh_is_the_ratio_of_the_two_times():
    field = 5e10
    got = float(keldysh(Gas.Ar, WAVELENGTH, field))
    ip_au = ionisation_potential(Gas.Ar) / AU_ENERGY
    expected = omega0_au() * math.sqrt(2.0 * ip_au) / (field / AU_EFIELD)
    assert got == pytest.approx(expected, rel=1e-15)
    # It falls as 1/E, so a ten-fold field is a tenth of the parameter.
    assert float(keldysh(Gas.Ar, WAVELENGTH, 10.0 * field)) == pytest.approx(
        got / 10.0, rel=1e-15
    )
    assert keldysh(Gas.Ar, WAVELENGTH, np.full(3, field)).shape == (3,)
    # The sign of the field cannot matter.
    assert float(keldysh(Gas.Ar, WAVELENGTH, -field)) == got


def test_keldysh_is_the_one_optmats_rate_reports():
    # One definition: the diagnostic and optmat's rate model agree on gamma for the
    # same first ionisation energy, to the rounding of two assemblies of the formula.
    rate = OptmatRate("Ar", WAVELENGTH)
    for field in (1e10, 3e10, 8e10):
        assert float(keldysh("Ar", WAVELENGTH, field)) == pytest.approx(
            rate.keldysh(field), rel=1e-12
        )


def test_ionisation_fraction_is_monotone_and_bounded():
    time = np.linspace(-3e-14, 3e-14, 513)
    field = 6e10 * np.exp(-((time / 8e-15) ** 2))
    fraction = ionisation_fraction(adk_oracle("Ar"), field, float(time[1] - time[0]))
    assert np.all(np.diff(fraction) >= 0.0)
    assert 0.0 <= fraction[0] < fraction[-1] <= 1.0


def test_ionisation_fraction_is_the_survival_of_the_trapezoid_integral():
    # A constant rate over a constant field: the trapezoid is exact, so the fraction is
    # 1 - exp(-W t) to Higham's n * eps over the 257 samples.
    time = np.linspace(0.0, 1e-14, 257)
    field = np.full(time.size, 1e9)
    rate = 1e14

    def constant(field_amplitude):
        magnitude = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(magnitude > 0.0, rate, 0.0)

    fraction = ionisation_fraction(constant, field, float(time[1] - time[0]))
    np.testing.assert_allclose(fraction, -np.expm1(-rate * time), rtol=1e-12, atol=0)


def test_pre_ionisation_sits_inside_the_exponential():
    # f = 1 - (1 - f0) exp(-int W dt), so f can never exceed one. Luna adds f0 outside
    # the exponential (Nonlinear.jl:280), where it can.
    time = np.linspace(-3e-14, 3e-14, 513)
    dt = float(time[1] - time[0])
    field = 6e10 * np.exp(-((time / 8e-15) ** 2))
    rate = adk_oracle("Ar")
    plain = ionisation_fraction(rate, field, dt)
    preionised = ionisation_fraction(rate, field, dt, preionisation=0.4)
    assert preionised[0] == pytest.approx(0.4)
    assert np.all(preionised <= 1.0)
    # Every atom already ionised is one the pulse cannot ionise again.
    assert np.allclose(1.0 - preionised, 0.6 * (1.0 - plain))
    assert np.all(ionisation_fraction(rate, field, dt, preionisation=1.0) == 1.0)


def test_no_field_ionises_nothing():
    time = np.linspace(-1e-14, 1e-14, 65)
    fraction = ionisation_fraction(
        adk_oracle("He"), np.zeros_like(time), float(time[1] - time[0])
    )
    assert np.array_equal(fraction, np.zeros_like(time))


# --------------------------------------------------------------- error paths


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_a_non_positive_wavelength_is_rejected(bad):
    with pytest.raises(MaterialError, match="wavelength must be positive"):
        keldysh(Gas.He, bad, 1e10)


def test_an_unknown_or_dataless_species_is_rejected():
    with pytest.raises(MaterialError, match="unknown gas"):
        keldysh("Argon", WAVELENGTH, 1e10)
    # Air is a mixture: each of its populations has its own potential and the mixture
    # has none.
    with pytest.raises(MaterialError, match="no single ionisation potential"):
        keldysh(Gas.Air, WAVELENGTH, 1e10)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_non_finite_field_is_rejected(bad):
    with pytest.raises(MaterialError, match="must be finite"):
        keldysh(Gas.He, WAVELENGTH, bad)
    with pytest.raises(MaterialError, match="must be finite"):
        keldysh(Gas.He, WAVELENGTH, np.array([1e10, bad]))


@pytest.mark.parametrize("bad", [0.0, -1.0, math.inf, math.nan])
def test_a_non_positive_time_step_is_rejected(bad):
    with pytest.raises(MaterialError, match="dt must be positive"):
        ionisation_fraction(adk_oracle("He"), np.zeros(4), bad)


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_an_impossible_pre_ionisation_is_rejected(bad):
    with pytest.raises(MaterialError, match=r"preionisation must be in \[0, 1\]"):
        ionisation_fraction(adk_oracle("He"), np.zeros(4), 1e-16, preionisation=bad)
