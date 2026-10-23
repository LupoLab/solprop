# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Comparison of the Raman response models against Luna.

The stored file is a regression witness, not truth. It was generated once, by
``tests/data/generate_luna_raman.py``, and is frozen: never regenerated, every input the
comparison depends on stated in the test (``docs/dev/testing.md``); the file records the
Luna commit it came from and Luna's own physical constants, so that the CODATA
difference between the two packages is accounted for rather than hidden in a loose
tolerance.

Three convention/constant conversions stand between the two sides:

* Luna's ``h`` carries ``eps0`` twice where solprop's carries it once, so a stored
  response becomes a solprop one on multiplication by ``eps0_solprop / eps0_luna^2``;
* the vibrational scale factor goes as ``1 / m_u``, which moved by 1.7e-8;
* the rotational populations go as ``exp(-E_J hbar / (hbar k_B T))``, whose ``hbar/k_B``
  moved by 3.3e-7. That one is not a monomial in the response, so it is not rescaled
  but bounded, from the recorded constants and the ladder's own Boltzmann exponents.

There is also a physical correction: solprop's scalar vibration amplitude is twice
Luna's inherited one. The original witness is preserved and only its
vibrational component receives that factor. The independent quantum-impulse tests,
not this comparison, establish the correction's physical basis.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from former_raman import (
    FORMER_CH4,
    FORMER_D2,
    FORMER_H2,
    FORMER_N2,
    FORMER_SF6,
    FORMER_SILICA_LINES,
    former_response,
)
from solprop.constants import AMAGAT, ATOMIC_MASS, EPS0, HBAR, K_B, C
from solprop.nonlinear.raman import (
    DampedOscillator,
    IntermediateBroadening,
    MolecularOptions,
    normalised_damped_oscillator,
    rotational_nonrigid,
    vibrational,
)
from solprop.physdata import Gas, RotationalRaman

REFERENCE = json.loads((Path(__file__).parents[1] / "data/luna_raman.json").read_text())
LUNA = REFERENCE["luna"]
PROVENANCE = REFERENCE["provenance"]
CASES = REFERENCE["cases"]
TIME_AXES = {
    name: np.asarray(values) for name, values in REFERENCE["time_axes"].items()
}

TEMPERATURE = PROVENANCE["temperature_k"]
AMAGAT_MULTIPLE = PROVENANCE["amagat_multiple"]
MIN_J = PROVENANCE["min_j"]
MAX_J = PROVENANCE["max_j"]
OPTIONS = {
    key: MolecularOptions(
        temperature=TEMPERATURE,
        rotation=case["rotation"],
        vibration=case["vibration"],
        min_j=MIN_J,
        max_j=MAX_J,
    )
    for key, case in CASES.items()
}

LUNA_EPS0, LUNA_HBAR, LUNA_KB, LUNA_C, LUNA_MU, LUNA_AMG, LUNA_ATM = LUNA["CONSTANTS"]

# The shift in hbar/k_B between CODATA 2014 and CODATA 2022, which is the whole of what
# separates the two packages' Boltzmann factors. Everything else in a population is a
# tabulated molecular constant or the exact speed of light.
BOLTZMANN_SHIFT = abs((LUNA_HBAR / LUNA_KB) / (HBAR / K_B) - 1.0)

# Levels below this share of the most populated one contribute nothing measurable to
# the response, so their (large) Boltzmann exponents must not set the bound.
POPULATION_FLOOR = 1e-6

# The witness's molecular inputs. solprop's own Raman data was deleted in SEP 3, Phase
# 4b: a built-in gas's response is now optmat's. This witness compares solprop's Raman
# *numerics* with Luna's, so it supplies the constants both sides were run with as
# explicit user data: solprop's former registry at dc4ca0d, bit for bit, held in
# tests/former_raman.py and taken through a caller's species (the supplied route).
WITNESS_RAMAN = {
    Gas.H2: FORMER_H2,
    Gas.N2: FORMER_N2,
    Gas.D2: FORMER_D2,
    Gas.CH4: FORMER_CH4,
    Gas.SF6: FORMER_SF6,
}

LADDER_GASES = {"H2": Gas.H2, "N2": Gas.N2, "D2": Gas.D2}
GASES = {"H2": Gas.H2, "N2": Gas.N2, "D2": Gas.D2, "CH4": Gas.CH4, "SF6": Gas.SF6}

INPUTS = {"CONSTANTS"}
DIVERGENT: dict[str, str] = {}


def stored(key: str) -> np.ndarray:
    """The stored Luna values for a key, as an array."""
    return np.asarray(LUNA[key], dtype=float)


def as_solprop_response(values: np.ndarray) -> np.ndarray:
    """Luna's ``h`` (or a scale factor of it) in solprop's units and constants.

    Luna's response is quadratic in ``eps0`` and solprop's is linear, so the conversion
    divides by Luna's ``eps0`` once for the convention and once more, times solprop's,
    for the CODATA difference.
    """
    return values * (EPS0 / LUNA_EPS0**2)


def rotational_parameters(gas: Gas) -> RotationalRaman:
    """The rotational constants of a gas that has them."""
    part = WITNESS_RAMAN[gas].rotational
    assert part is not None
    return part


def ladder_of(gas: Gas):
    """The solprop rotational ladder built at the witness's own state."""
    return rotational_nonrigid(
        rotational_parameters(gas), TEMPERATURE, min_j=MIN_J, max_j=MAX_J
    )


def population_bound(gas: Gas) -> float:
    """The relative error a 3.3e-7 shift in ``hbar/k_B`` leaves in a ladder.

    Normalising the populations removes the common mode, so what survives is the
    spread of the Boltzmann exponent about its population-weighted mean, over the
    levels that carry any population at all.
    """
    parameters = rotational_parameters(gas)
    lines = rotational_nonrigid(parameters, TEMPERATURE, min_j=MIN_J, max_j=MAX_J)
    exponent = lines.energies / (K_B * TEMPERATURE)
    mean = float(lines.populations @ exponent)
    carried = lines.populations > POPULATION_FLOOR * lines.populations.max()
    spread = float(np.max(np.abs(exponent[carried] - mean)))
    # expm1 rather than the first-order product: at a spread of 22 the second-order
    # term is 1.1e-11, which is 2 ulp of the bound itself and enough to cross it.
    return abs(math.expm1(BOLTZMANN_SHIFT * spread))


def worst_relative(actual: np.ndarray, expected: np.ndarray) -> float:
    """The largest deviation, relative to the peak of the reference.

    Relative to the peak rather than element by element: these responses oscillate
    through zero, where a per-element relative error is meaningless.
    """
    scale = float(np.max(np.abs(expected)))
    assert scale > 0.0
    return float(np.max(np.abs(actual - expected))) / scale


# --------------------------------------------------------------------- provenance


def test_provenance_is_recorded():
    assert len(PROVENANCE["luna_commit"]) == 40
    assert PROVENANCE["luna_branch"] == "modal-fixed"
    assert PROVENANCE["julia_version"].startswith("julia version")


def test_every_stored_quantity_is_classified():
    # A new witness must be triaged as an input, a comparison or a documented
    # divergence, never silently ignored.
    compared = set(LUNA) - INPUTS - set(DIVERGENT)
    assert set(LUNA) == INPUTS | compared | set(DIVERGENT)
    assert compared


def test_the_recorded_constants_are_the_expected_codata_2014():
    # If Luna ever changes constant set, every rescaling below silently becomes wrong,
    # so the shifts themselves are asserted rather than assumed.
    assert LUNA_C == C
    assert pytest.approx(101325.0, rel=1e-15) == LUNA_ATM
    assert abs(LUNA_EPS0 / EPS0 - 1.0) == pytest.approx(1.33e-10, rel=0.05)
    assert abs(LUNA_MU / ATOMIC_MASS - 1.0) == pytest.approx(1.74e-8, rel=0.05)
    assert pytest.approx(3.31e-7, rel=0.05) == BOLTZMANN_SHIFT


def test_the_amagat_difference_is_the_boltzmann_one():
    # amg = atm / (k_B * 273.15) and atm is exact, so the amagat carries only k_B.
    # Giving the density in amagat therefore makes the dephasing time identical on
    # both sides, which is why the witness does that instead of quoting a pressure.
    assert pytest.approx(1.0, rel=1e-15) == LUNA_AMG / LUNA_ATM * LUNA_KB * 273.15


# ---------------------------------------------------------- the damped oscillator


def test_the_normalised_oscillator_scale_matches_luna():
    amplitude, frequency, tau2 = PROVENANCE["normalised_oscillator"]
    line = normalised_damped_oscillator(amplitude, frequency, tau2)
    # A pure number: no physical constant enters, so the two must agree to round-off.
    assert line.amplitude == pytest.approx(float(stored("SDO")[0]), rel=4e-16)


def test_the_silica_scale_matches_luna():
    model = IntermediateBroadening(FORMER_SILICA_LINES)
    # Also constant-free. solprop's is the analytic integral and Luna's an adaptive
    # quadrature to 1 ns, so the agreement is set by Luna's quadrature, not by ours.
    assert model.scale == pytest.approx(float(stored("SILICA_SCALE")[0]), rel=1e-10)


# ------------------------------------------------------------- the vibrational mode


@pytest.mark.parametrize("name", sorted(GASES))
def test_the_vibrational_frequency_matches_luna(name):
    mode = WITNESS_RAMAN[GASES[name]].vibrational
    assert mode is not None
    # Tabulated wavenumbers times the exact speed of light: nothing to rescale.
    assert mode.frequency == pytest.approx(float(stored(f"VIB-{name}")[1]), rel=1e-15)


@pytest.mark.parametrize("name", sorted(GASES))
def test_the_vibrational_scale_includes_the_independent_amplitude_correction(name):
    mode = WITNESS_RAMAN[GASES[name]].vibrational
    assert mode is not None
    # K goes as eps0 / mu, both monomials, so both are rescaled exactly and what is
    # left is round-off after applying the independently verified factor of two.
    expected = (
        2 * as_solprop_response(stored(f"VIB-{name}")[0]) * (LUNA_MU / ATOMIC_MASS)
    )
    assert vibrational(mode).amplitude == pytest.approx(
        float(expected), rel=1e-14, abs=0.0
    )


def test_the_vibrational_scale_needs_the_constant_correction():
    # Without rescaling m_u the same comparison is out by 1.7e-8: the correction is
    # doing real work, not decorating a tolerance that would have passed anyway.
    mode = WITNESS_RAMAN[Gas.H2].vibrational
    assert mode is not None
    uncorrected = 2 * as_solprop_response(stored("VIB-H2")[0])
    residual = abs(vibrational(mode).amplitude / float(uncorrected) - 1.0)
    assert 1e-9 < residual < 1e-7


# --------------------------------------------------------------- the level ladder


@pytest.mark.parametrize("name", sorted(LADDER_GASES))
def test_the_retained_levels_match_luna(name):
    lines = ladder_of(LADDER_GASES[name])
    assert np.array_equal(lines.quantum_numbers, stored(f"LADDER-{name}-J"))


@pytest.mark.parametrize("name", sorted(LADDER_GASES))
def test_the_level_energies_match_luna(name):
    lines = ladder_of(LADDER_GASES[name])
    # E_J = 2 pi hbar c (...): linear in hbar, so exactly rescalable.
    expected = stored(f"LADDER-{name}-EJ") * (HBAR / LUNA_HBAR)
    assert np.allclose(lines.energies, expected, rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("name", sorted(LADDER_GASES))
def test_the_line_frequencies_match_luna_at_round_off(name):
    lines = ladder_of(LADDER_GASES[name])
    # Omega = (E_{J+2} - E_J)/hbar: hbar cancels and c is exact in both constant sets,
    # so this is the one rotational quantity that carries no CODATA content at all.
    # The subtraction of two nearly equal energies costs a few digits at low J.
    expected = stored(f"LADDER-{name}-OMEGA")
    assert np.allclose(lines.frequencies, expected, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("name", sorted(LADDER_GASES))
def test_the_populations_match_luna_within_the_boltzmann_shift(name):
    gas = LADDER_GASES[name]
    lines = ladder_of(gas)
    expected = stored(f"LADDER-{name}-POP")
    carried = lines.populations > POPULATION_FLOOR * lines.populations.max()
    worst = float(np.max(np.abs(lines.populations[carried] / expected[carried] - 1.0)))
    # Not merely bounded by the constant shift but *equal* to it: the level whose
    # Boltzmann exponent lies furthest from the population-weighted mean is the one
    # that moves most, and the prediction matches the measurement to eight digits.
    # A real error in the ladder would break this far more visibly than a loose bound.
    assert worst == pytest.approx(population_bound(gas), rel=1e-3)


@pytest.mark.parametrize("name", sorted(LADDER_GASES))
def test_the_line_weights_match_luna_within_the_boltzmann_shift(name):
    gas = LADDER_GASES[name]
    lines = ladder_of(gas)
    # The weights go as eps0 / hbar times a population difference; the first two are
    # rescaled and the third is what the bound allows for.
    expected = as_solprop_response(stored(f"LADDER-{name}-K")) * (LUNA_HBAR / HBAR)
    assert 0.0 < worst_relative(lines.amplitudes, expected) < population_bound(gas)


# ------------------------------------------------------------- the sampled response


def response_of(key):
    """The solprop response for one stored case."""
    return former_response(WITNESS_RAMAN[Gas(CASES[key]["gas"])], OPTIONS[key])


def sampled(key, axis):
    """The solprop response sampled on one of the stored delay axes."""
    return response_of(key).impulse(TIME_AXES[axis], AMAGAT_MULTIPLE * AMAGAT)


VIBRATION_ONLY = sorted(k for k, c in CASES.items() if not c["rotation"])
WITH_ROTATION = sorted(k for k, c in CASES.items() if c["rotation"])


def phase_bound(frequency: float, axis: str, expected: np.ndarray) -> float:
    """The deviation one unit in the last place of a line frequency can produce.

    Both packages build the same angular frequency from the same tabulated wavenumber,
    but multiply the factors in a different order, so the two can differ by an ulp.
    Sampling a sine of that frequency turns the difference into a phase error that
    grows as ``Omega t``, weighted by how much response is left at that delay. For
    nitrogen the two orders happen to agree exactly and the bound is never approached;
    for methane, whose phase reaches 33000 radians before its coherence dies, it is
    what the comparison is limited by.
    """
    epsilon = float(np.finfo(float).eps)
    weight = np.abs(expected) / float(np.max(np.abs(expected)))
    accumulated = float(np.max(frequency * TIME_AXES[axis] * weight))
    # Four epsilon: one ulp in the frequency, one in the delay product, and the
    # argument reduction of the sine itself. The floor is the round-off of the four
    # products that build h from its scale factor.
    return 4.0 * epsilon * accumulated + 8.0 * epsilon


@pytest.mark.parametrize("key", VIBRATION_ONLY)
@pytest.mark.parametrize("axis", sorted(TIME_AXES))
def test_a_vibrational_response_is_twice_the_corrected_luna_witness(key, axis):
    # Every constant in a vibrational response is a monomial and rescaled exactly, and
    # the density is given in amagat so the dephasing time is identical, so what is
    # left is the accumulated phase of the sine.
    expected = (
        2 * as_solprop_response(stored(f"{key}-{axis}")) * (LUNA_MU / ATOMIC_MASS)
    )
    line = response_of(key).models[0]
    assert isinstance(line, DampedOscillator)
    assert worst_relative(sampled(key, axis), expected) < phase_bound(
        line.frequency, axis, expected
    )


@pytest.mark.parametrize("key", WITH_ROTATION)
@pytest.mark.parametrize("axis", sorted(TIME_AXES))
def test_a_rotational_response_matches_luna_within_the_boltzmann_shift(key, axis):
    gas = Gas(CASES[key]["gas"])
    # Keep the original mixed witness. Add only the change in its separately
    # stored vibrational component: mass-constant rescaling and the amplitude doubling.
    expected = as_solprop_response(stored(f"{key}-{axis}"))
    if CASES[key]["vibration"]:
        vibration_key = f"{gas.value.lower()}-vibration-{axis}"
        expected += as_solprop_response(stored(vibration_key)) * (
            2 * LUNA_MU / ATOMIC_MASS - 1
        )
    if gas is Gas.D2:
        assert not CASES[key]["vibration"]  # The archive has only D2 rotation.
        # solprop corrects Luna's pressure/amagat transcription for D2. The stored D2
        # rotation witness keeps its original damping; adjust that component
        # analytically, without changing the archive or relaxing its bound.
        parameters = rotational_parameters(gas)
        new_decay = 1.0 / parameters.linewidth.dephasing_time(AMAGAT_MULTIPLE)
        old_decay = math.pi * 4e-3 * 100 * C * AMAGAT_MULTIPLE
        expected *= np.exp(-(new_decay - old_decay) * TIME_AXES[axis])
    worst = worst_relative(sampled(key, axis), expected)
    assert 0.0 < worst < population_bound(gas)


def test_the_rotational_difference_is_the_boltzmann_shift_and_not_a_bug():
    # Sharpen the statement: the deviation must be well below the bound as well as
    # above round-off, so a real error in the line weights could not hide inside it.
    worst = worst_relative(
        sampled("h2-rotation", "fast"),
        as_solprop_response(stored("h2-rotation-fast")),
    )
    assert 1e-8 < worst < 1e-5
