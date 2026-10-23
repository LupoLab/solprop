# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Raman data types a caller supplies, and the status of optmat's responses.

Every built-in Raman response is optmat's (SEP 3, Phase 4b): solprop's own molecular
constants and its silica line set were removed with :mod:`solprop.physdata.raman_data`.
What is left is the supplied types of :mod:`solprop.physdata.raman_types` -- their
linewidth law and their validation -- and :func:`raman_status`, which the capillary
driver's default reads.
"""

import math
from dataclasses import FrozenInstanceError

import pytest

from solprop.constants import AMAGAT, BAR, STD_ATMOSPHERE
from solprop.exceptions import MaterialError
from solprop.nonlinear.optmat_raman import raman_status
from solprop.physdata import (
    Gas,
    IntermediateBroadeningRaman,
    Linewidth,
    VibrationalRaman,
    ideal_gas_density,
    linewidth_from_dephasing_time,
)

# Hydrogen's rotational S(1) linewidth law, Hz per amagat: Bischel and Dyer, Phys. Rev.
# A 33, 3113 (1986), as solprop's former registry carried it (dc4ca0d). Test data.
HYDROGEN_ROTATIONAL = Linewidth(inverse_density=6.15e6, density=1.14e8)

# Hydrogen's vibrational Q(1) constants as the former registry carried them (dc4ca0d):
# a caller's own supplied record, test data.
HYDROGEN_VIBRATIONAL = VibrationalRaman(
    frequency=7.826769158409088e14,
    polarisability_derivative=1.3e-20,
    reduced_mass=8.367788476101665e-28,
    linewidth=Linewidth(inverse_density=3.09e8, density=5.22e7),
    citation="test data: Phys. Rev. A 92, 063828 (2015); Phys. Rev. A 33, 3113 (1986)",
)


# ------------------------------------------------------------------ linewidths


def test_dephasing_time_round_trips_a_measured_value():
    assert linewidth_from_dephasing_time(6e-12).dephasing_time(1.0) == pytest.approx(
        6e-12, rel=1e-15
    )


def test_a_constant_linewidth_does_not_depend_on_density():
    linewidth = linewidth_from_dephasing_time(6.6e-12)
    assert linewidth.dephasing_time(0.1) == linewidth.dephasing_time(100.0)


def test_collisional_broadening_shortens_the_dephasing_time():
    linewidth = Linewidth(density=3.3e9)
    assert linewidth.dephasing_time(10.0) < linewidth.dephasing_time(1.0)


def test_dicke_narrowing_lengthens_it_at_low_density():
    # The inverse-density term is Dicke narrowing: below the crossover the line gets
    # narrower as the density rises, which is why hydrogen's rotational coherence is
    # longest around a few amagat rather than in the dilute limit.
    linewidth = HYDROGEN_ROTATIONAL
    crossover = math.sqrt(linewidth.inverse_density / linewidth.density)
    assert linewidth.dephasing_time(crossover) > linewidth.dephasing_time(0.1)
    assert linewidth.dephasing_time(crossover) > linewidth.dephasing_time(
        10.0 * crossover
    )


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_zero_density_is_rejected_rather_than_returning_nan(bad):
    # Luna divides by the density without a guard in the vibrational case, so a fill
    # starting from vacuum yields tau2 = 0 and a NaN polarisation in the first sample.
    with pytest.raises(MaterialError, match="density must be positive"):
        HYDROGEN_VIBRATIONAL.linewidth.dephasing_time(bad)


@pytest.mark.parametrize(
    "linewidth", [HYDROGEN_ROTATIONAL, HYDROGEN_VIBRATIONAL.linewidth]
)
def test_dephasing_times_are_picoseconds_to_nanoseconds_at_one_amagat(linewidth):
    assert 1e-12 < linewidth.dephasing_time(1.0) < 1e-8


def test_the_amagat_is_the_loschmidt_constant():
    assert (
        pytest.approx(ideal_gas_density(STD_ATMOSPHERE / BAR, 273.15), rel=1e-15)
        == AMAGAT
    )


def test_entries_are_immutable():
    field = "frequency"  # not a literal, so the type checker leaves the assignment be
    with pytest.raises(FrozenInstanceError):
        setattr(HYDROGEN_VIBRATIONAL, field, 0.0)


@pytest.mark.parametrize("bad", [0.0, -1e-12])
def test_a_non_positive_dephasing_time_is_rejected(bad):
    with pytest.raises(MaterialError, match="dephasing time must be positive"):
        linewidth_from_dephasing_time(bad)


# ------------------------------------------------------------ a supplied line set


def test_a_supplied_line_set_counts_its_lines():
    lines = IntermediateBroadeningRaman(
        frequencies=(1.0e13, 2.0e13),
        amplitudes=(1.0, 0.5),
        gaussian_widths=(1.0e12, 2.0e12),
        lorentzian_widths=(5.0e11, 1.0e12),
        citation="test",
    )
    assert lines.n_lines == 2


def test_a_line_set_with_mismatched_lengths_is_rejected():
    with pytest.raises(MaterialError, match="same length"):
        IntermediateBroadeningRaman(
            frequencies=(1.0, 2.0),
            amplitudes=(1.0,),
            gaussian_widths=(1.0, 2.0),
            lorentzian_widths=(1.0, 2.0),
            citation="test",
        )


def test_an_empty_line_set_is_rejected():
    with pytest.raises(MaterialError, match="at least one line"):
        IntermediateBroadeningRaman(
            frequencies=(),
            amplitudes=(),
            gaussian_widths=(),
            lorentzian_widths=(),
            citation="test",
        )


# ------------------------------------------------------- optmat's responses


@pytest.mark.parametrize(
    ("gas", "status"),
    [
        (Gas.N2, "validated"),
        (Gas.O2, "validated"),
        (Gas.H2, "validated"),
        (Gas.D2, "validated"),
        (Gas.CH4, "validated"),
        (Gas.SF6, "validated"),
        (Gas.N2O, "provisional"),
        (Gas.He, "atom"),
        (Gas.Air, "atom"),
    ],
)
def test_raman_status_names_what_the_driver_default_needs(gas, status):
    # The driver's raman=None is on for "validated", warns for "provisional".
    assert raman_status(gas.value) == status


@pytest.mark.parametrize("gas", ["CO", "O3"])
def test_a_molecule_optmat_curates_no_raman_response_for_has_none(gas):
    # optmat knows these as molecules but has no Raman response of them, so the
    # driver's default leaves Raman off for them without a warning.
    assert raman_status(gas) == "none"
