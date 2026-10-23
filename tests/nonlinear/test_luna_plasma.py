# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Comparison against the plasma polarisation Luna computes on this machine.

The stored file is a regression witness, not truth: where solprop deliberately differs
from Luna the difference is asserted exactly, not merely tolerated. It was generated
once, by ``tests/data/generate_luna_plasma.py``, and is frozen: never regenerated, every
input the comparison depends on stated in the test (``docs/dev/testing.md``); the file
records the Luna commit it came from. Luna's column is the plain cumulative trapezoid,
so every response here states ``integration="trapezoid"``, the legacy rule solprop keeps
for these comparisons (SEP 18).
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from bounds import scan_bound
from solprop.constants import E_CHARGE, M_E
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import IonisationData, species

REFERENCE = json.loads(
    (Path(__file__).parents[1] / "data/luna_plasma.json").read_text()
)
LUNA = REFERENCE["luna"]
PARAMETERS = REFERENCE["parameters"]
SIGNALS = REFERENCE["signals"]

# The witness is frozen (``docs/dev/testing.md``): every response below states the
# inputs Luna was given -- the potential, the rate and no preionisation unless a case
# asks for it -- rather than taking a solprop default.

CHARGE_MASS_RATIO = E_CHARGE**2 / M_E
LUNA_CHARGE_MASS_RATIO, _, _ = LUNA["CONSTANTS"]

# The ionisation potential the generator handed Luna, recorded in the fixture (solprop's
# former registry value for argon, 0.5792 hartree, at the time of the run). The witness
# supplies it as explicit species data, so the comparison does not move with optmat's
# curated potential (SEP 3, Phase 4b).
IONISATION_POTENTIAL = PARAMETERS["ionisation_potential"]
ARGON = replace(
    species("Ar"),
    ionisation=IonisationData(
        IONISATION_POTENTIAL, citation="tests/data/luna_plasma.json, the Luna input"
    ),
)

# Luna's constants come from PhysicalConstants.CODATA2014 and solprop's from
# scipy.constants, which is CODATA 2022. The plasma carries no vacuum permittivity: the
# free-electron current is linear in e^2/m_e, which moved by -4.2e-10 relative, and the
# ionisation-loss current is linear in the ionisation potential, which the generator
# hands to Luna rather than letting it look one up. Every case is therefore stored
# twice, once with a zero potential, so each half can be rescaled by the one constant it
# carries. Rescaling brings the comparison from 4.2e-10 to 5.4e-14, which is round-off
# over a thousand samples rather than a known constant hidden behind a tolerance.
CHARGE_MASS_RESCALE = CHARGE_MASS_RATIO / LUNA_CHARGE_MASS_RATIO

# Three cumulative sums over 1025 samples, formed in a different order on each side:
# Higham's n*eps is 2e-13. Measured worst peak-normalised deviation 5.4e-14.
ANALYTIC_RTOL = 2e-13

# The fraction and the rate carry no constants at all when the rate is the analytic one,
# so they compare at pure round-off. Measured 3.8e-16.
EXACT_RTOL = 1e-14


# The ionisation-loss current is reconstructed as a difference of two Luna runs, which
# costs a relative accuracy of eps divided by this ratio. Measured 0.026 to 0.028 for
# the stored cases, so the reconstruction keeps about fourteen digits.
MIN_LOSS_OVER_FREE = 1e-2

SCALAR_CASES = ("SCALAR", "ZEROS")
STAGES = ("RATE", "FRACTION", "J", "P", "P_FUSED")
VECTOR_STAGES = (
    "RATE",
    "FRACTION",
    "J_X",
    "J_Y",
    "P_X",
    "P_Y",
    "P_FUSED_X",
    "P_FUSED_Y",
)

INPUTS = {"CONSTANTS"}
DIVERGENT = {
    f"PREIONISED{suffix}_{stage}" for suffix in ("", "_FREE") for stage in STAGES
}
COMPARED = {
    f"{case}{suffix}_{stage}"
    for case in (*SCALAR_CASES, "PPT")
    for suffix in ("", "_FREE")
    for stage in STAGES
} | {f"VECTOR{suffix}_{stage}" for suffix in ("", "_FREE") for stage in VECTOR_STAGES}


def stored(key):
    """One stored Luna array."""
    return np.asarray(LUNA[key], dtype=float)


def signal(name):
    """One of the fields both sides were given."""
    return np.asarray(SIGNALS[name], dtype=float)


def rescaled(case, stage):
    """Luna's value as it would be with solprop's constants.

    The run with a zero ionisation potential is exactly the free-electron part, which is
    linear in ``e^2/m_e``; what is left is the loss part, which carries no constant at
    all because Luna was given the potential.
    """
    free = stored(f"{case}_FREE_{stage}")
    return free * CHARGE_MASS_RESCALE + (stored(f"{case}_{stage}") - free)


def analytic_rate(field_amplitude):
    """The rate the generator spelled identically on both sides."""
    magnitude = np.abs(np.asarray(field_amplitude, dtype=float))
    above = magnitude >= PARAMETERS["rate_threshold"]
    return np.where(
        above,
        PARAMETERS["rate_scale"] * (magnitude / PARAMETERS["rate_reference"]) ** 8,
        0.0,
    )


@dataclass(frozen=True)
class RecordedRate:
    """Luna's PPT rate as it recorded it, sample for sample along the stored field.

    solprop no longer has a PPT formula of its own (SEP 3, Phase 4b): the rate of the
    one real-model case is supplied from the fixture, as a function of the field's
    magnitude interpolated between the recorded samples. At the samples themselves it
    is exactly Luna's value.
    """

    magnitude: np.ndarray
    rate: np.ndarray

    @classmethod
    def from_fixture(cls):
        magnitude = np.abs(signal("field"))
        order = np.argsort(magnitude, kind="stable")
        return cls(magnitude[order], stored("PPT_RATE")[order])

    def __call__(self, field_amplitude):
        magnitude = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.interp(magnitude, self.magnitude, self.rate, left=0.0)


def deviation(got, expected):
    """Peak-normalised deviation, which a response passing through zero needs."""
    return float(np.max(np.abs(got - expected)) / np.max(np.abs(expected)))


# ------------------------------------------------------------------- provenance


def test_provenance_is_recorded():
    provenance = REFERENCE["provenance"]
    assert len(provenance["luna_commit"]) == 40
    assert provenance["luna_branch"] == "modal-fixed"
    assert provenance["julia_version"].startswith("julia version")
    assert PARAMETERS["gas"] == "Ar"
    assert ARGON.ionisation is not None
    assert ARGON.ionisation.potential == IONISATION_POTENTIAL


def test_every_stored_quantity_is_classified():
    # A new witness must be triaged as an input, a comparison or a documented
    # divergence, never silently ignored.
    assert set(LUNA) == INPUTS | COMPARED | DIVERGENT


def test_the_loss_current_is_large_enough_to_reconstruct():
    # It is recovered as a difference of two runs, so a case where it were a rounding of
    # the total would compare nothing.
    for case, ratio in REFERENCE["provenance"]["loss_over_free"].items():
        assert ratio > MIN_LOSS_OVER_FREE, case


# --------------------------------------------------------------- the analytic rate


@pytest.mark.parametrize("case", SCALAR_CASES)
def test_the_scalar_stages_match_luna(case):
    field = signal("field" if case == "SCALAR" else "zeroed")
    plasma = PlasmaResponse(
        ARGON, analytic_rate, preionisation=0.0, integration="trapezoid"
    )
    dt = PARAMETERS["dt"]

    # The rate and the fraction carry no constants, so they compare at round-off.
    assert deviation(analytic_rate(np.abs(field)), stored(f"{case}_RATE")) < EXACT_RTOL
    fraction = plasma.ionised_fraction(field[None, :], dt=dt, kind="field")
    assert 0.0 < deviation(fraction, stored(f"{case}_FRACTION")) < EXACT_RTOL

    got = plasma.evaluate(field[None, :], 1.0, dt=dt, kind="field")[0]
    assert 0.0 < deviation(got, rescaled(case, "P")) < ANALYTIC_RTOL


def test_the_vector_stages_match_luna():
    components = np.stack([signal("field_x"), signal("field_y")])
    plasma = PlasmaResponse(
        ARGON, analytic_rate, preionisation=0.0, integration="trapezoid"
    )
    dt = PARAMETERS["dt"]

    magnitude = np.hypot(components[0], components[1])
    assert deviation(analytic_rate(magnitude), stored("VECTOR_RATE")) < EXACT_RTOL
    fraction = plasma.ionised_fraction(components, dt=dt, kind="field")
    assert 0.0 < deviation(fraction, stored("VECTOR_FRACTION")) < EXACT_RTOL

    got = plasma.evaluate(components, 1.0, dt=dt, kind="field")
    for index, axis in enumerate(("X", "Y")):
        assert (
            0.0
            < deviation(got[index], rescaled("VECTOR", f"P_{axis}"))
            < (ANALYTIC_RTOL)
        )


@pytest.mark.parametrize("case", [*SCALAR_CASES, "PPT"])
def test_lunas_fused_kernel_is_the_reference_it_claims_to_be(case):
    # Luna's design note says its two-pass column kernel is the multi-pass form solprop
    # ports, reordered. Both are stored, so the claim is witnessed rather than assumed,
    # at the plasma chain's rounding (SEP 25) -- and if it ever stops holding, the
    # oracle solprop was built from is not the one a Luna propagation runs.
    for suffix in ("", "_FREE"):
        fused_witness(stored(f"{case}{suffix}_P_FUSED"), stored(f"{case}{suffix}_P"))


def fused_witness(fused, reference):
    """Assert a fused column within the plasma chain's cap of the multi-pass one."""
    scale = float(np.max(np.abs(reference)))
    assert scale > 0
    bound = scan_bound(np.shape(reference)[-1]) * scale
    assert np.max(np.abs(np.asarray(fused) - np.asarray(reference))) <= bound


def test_the_vector_fused_kernel_is_the_reference_it_claims_to_be():
    for suffix in ("", "_FREE"):
        for axis in ("X", "Y"):
            fused_witness(
                stored(f"VECTOR{suffix}_P_FUSED_{axis}"),
                stored(f"VECTOR{suffix}_P_{axis}"),
            )


def test_a_field_with_an_exact_zero_matches_luna_sample_for_sample():
    # The only configuration that reaches the division by the field at |E| = 0 in the
    # middle of a pulse. Luna guards it with `abs(E) > 0` and solprop with a `where`;
    # this is the witness that the two guards agree.
    index = PARAMETERS["zeroed_index"]
    assert signal("zeroed")[index] == 0.0
    assert signal("field")[index] != 0.0
    plasma = PlasmaResponse(
        ARGON, analytic_rate, preionisation=0.0, integration="trapezoid"
    )
    got = plasma.evaluate(
        signal("zeroed")[None, :], 1.0, dt=PARAMETERS["dt"], kind="field"
    )[0]
    assert deviation(got, rescaled("ZEROS", "P")) < ANALYTIC_RTOL
    assert np.all(np.isfinite(got))


# ------------------------------------------------------------------ the PPT rate


def test_the_polarisation_of_a_real_rate_matches_luna():
    # Luna's direct PPT (no barrier-suppression correction), supplied as the rate it
    # recorded: with the model taken out, what is compared is the plasma arithmetic on a
    # real rate's shape, so the analytic cases' round-off bound applies. (solprop's own
    # PPT, which this case once compared at 2e-7, was deleted in SEP 3, Phase 4b.)
    field = signal("field")
    rate = RecordedRate.from_fixture()
    assert np.array_equal(rate(field), stored("PPT_RATE"))
    got = PlasmaResponse(
        ARGON, rate, preionisation=0.0, integration="trapezoid"
    ).evaluate(field[None, :], 1.0, dt=PARAMETERS["dt"], kind="field")[0]
    assert 0.0 < deviation(got, rescaled("PPT", "P")) < ANALYTIC_RTOL


# ------------------------------------------------------- the pre-ionisation fix


def test_lunas_ionised_fraction_is_solprops_plus_the_pre_ionised_offset():
    # DIVERGENT by design. Luna adds the pre-ionised fraction outside the exponential,
    # so `f_Luna = f0 + 1 - exp(-int W dt)` while solprop has
    # `f = 1 - (1 - f0) exp(-int W dt)`. Since solprop's fraction
    # at f0 = 0 is exactly `1 - exp(-int W dt)`, the two differ by exactly f0, so
    # Luna's tends to 1 + f0 rather than to 1 as the pulse saturates.
    field = signal("field")
    plain = PlasmaResponse(
        ARGON, analytic_rate, preionisation=0.0, integration="trapezoid"
    ).ionised_fraction(field[None, :], dt=PARAMETERS["dt"], kind="field")
    assert np.allclose(
        stored("PREIONISED_FRACTION"),
        PARAMETERS["preionfrac"] + plain,
        rtol=0.0,
        atol=4.0 * np.finfo(float).eps,
    )


def test_lunas_pre_ionised_polarisation_is_reconstructible_from_solprops():
    # The consequence of the fraction above, asserted exactly rather than tolerated.
    # Both currents are affine in the fraction, so Luna's answer is solprop's at zero
    # pre-ionisation plus f0 times the two integrals below: the free current an
    # unweighted double integral of the field, the loss current the term Luna keeps
    # because its `1 - f` has gone down by f0.
    field = signal("field")
    dt, offset = PARAMETERS["dt"], PARAMETERS["preionfrac"]
    plain = PlasmaResponse(
        ARGON, analytic_rate, preionisation=0.0, integration="trapezoid"
    ).evaluate(field[None, :], 1.0, dt=dt, kind="field")[0]

    def integrate(values):
        return cumulative_trapezoid(values, dx=dt, initial=0)

    rate = analytic_rate(np.abs(field))
    squared = np.abs(field) ** 2
    loss = np.zeros_like(squared)
    np.divide(
        IONISATION_POTENTIAL * rate,
        squared,
        out=loss,
        where=(rate > 0.0) & (squared > 0.0),
    )
    extra = CHARGE_MASS_RATIO * integrate(integrate(field)) - integrate(loss * field)
    assert deviation(plain + offset * extra, rescaled("PREIONISED", "P")) < (
        ANALYTIC_RTOL
    )


def test_the_pre_ionisation_fix_changes_the_answer_by_a_documented_amount():
    # What the fix costs in comparability: solprop's pre-ionised polarisation is 11 %
    # away from Luna's on this pulse, and the whole of that is the placement of f0.
    field = signal("field")
    got = PlasmaResponse(
        ARGON,
        analytic_rate,
        preionisation=PARAMETERS["preionfrac"],
        integration="trapezoid",
    ).evaluate(field[None, :], 1.0, dt=PARAMETERS["dt"], kind="field")[0]
    assert deviation(got, rescaled("PREIONISED", "P")) == pytest.approx(0.113, abs=0.01)
