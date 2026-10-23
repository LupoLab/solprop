# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from witness_materials import borzsonyi, malitson_silica

from solprop.constants import C
from solprop.exceptions import MaterialRangeWarning
from solprop.modes import MarcatiliMode
from solprop.modes.base import zdw

REFERENCE = json.loads((Path(__file__).parents[1] / "data/luna_modes.json").read_text())
SYMBOLIC = json.loads(
    (Path(__file__).parents[1] / "data/symbolic_marcatili.json").read_text()
)

WAVELENGTHS = np.array(REFERENCE["wavelengths"])
OMEGA = 2.0 * math.pi * C / WAVELENGTHS

# Cases generated with a constant cladding index: nothing about the two implementations
# differs there, so the effective index must agree to rounding.
CONSTANT = [key for key in REFERENCE["index_cases"] if key.endswith("-const")]
# Cases generated with each side's own default silica: the deviation is the replaced
# cladding data and is measured rather than tolerated.
SILICA = [key for key in REFERENCE["index_cases"] if key.endswith("-silica")]

# Luna was run at 294 K (``generate_luna_modes.py``), solprop's former default; the
# witness is not regenerated, so every mode compared with it is built at 294 K.
LUNA_TEMPERATURE = 294.0


def build(spec):
    """Build the solprop mode a stored Luna case describes."""
    return MarcatiliMode(
        spec["radius_m"],
        spec["gas"],
        spec["pressure_bar"],
        susceptibility=gas_index(spec),
        kind=spec["kind"],
        n=spec["n"],
        m=spec["m"],
        model=spec["model"],
        loss=spec["loss"],
        cladding=cladding(spec),
        temperature=LUNA_TEMPERATURE,
    )


def gas_index(spec):
    """Return the gas index Luna was run with, as explicit test-local data."""
    assert spec["source"] == "borzsonyi"
    return borzsonyi(spec["gas"])


def cladding(spec):
    """Return the constant cladding, or the silica the silica cases measured.

    That is solprop's former default silica (its real part, Malitson's law), supplied
    explicitly since SEP 3's Phase 4b deleted it, against Luna's unreferenced table.
    """
    return 1.45 if spec["cladding"] == "1.45" else malitson_silica()


def stored_index(key):
    """Return the stored complex effective index of one case."""
    values = np.asarray(REFERENCE["neff"][key])
    return values[0::2] + 1j * values[1::2]


@pytest.fixture(autouse=True)
def _allow_extrapolation():
    """Let the material models be asked for wavelengths outside their stated ranges.

    Luna states no validity range for any index model, so a like-for-like comparison
    has to ask solprop for exactly the wavelengths Luna was asked for -- 250 nm to
    2 um, where the Boerzsoenyi fits (400 nm to 1 um) and Malitson (210 nm to 3.71 um)
    are being extrapolated. That warning is the subject of the physdata and glasses
    tests; here it is noise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        yield


# ------------------------------------------------------------------ provenance


def test_provenance_is_recorded():
    provenance = REFERENCE["provenance"]
    assert len(provenance["luna_commit"]) == 40
    assert provenance["luna_branch"] == "modal-fixed"
    assert provenance["julia_version"].startswith("julia version")
    assert provenance["temperature_k"] == LUNA_TEMPERATURE
    assert provenance["constant_cladding"] == "1.45"


def test_the_symbolic_witness_records_how_it_was_derived():
    provenance = SYMBOLIC["provenance"]
    assert provenance["derived_with"].startswith("sympy ")
    assert provenance["gas"] == "Ar"
    assert provenance["index_source"] == "borzsonyi"
    assert provenance["wavelength_m"] == 800e-9


def test_every_stored_case_is_classified():
    # A new Luna case must be triaged into "nothing differs" or "the cladding data
    # differs", not silently skipped.
    assert set(CONSTANT) | set(SILICA) == set(REFERENCE["index_cases"])
    assert set(REFERENCE["neff"]) == set(REFERENCE["index_cases"])
    assert set(REFERENCE["zdw"]) == set(REFERENCE["zdw_cases"])
    assert set(REFERENCE["area"]) == set(REFERENCE["area_cases"])


# ------------------------------------------- the effective index, cladding matched


@pytest.mark.parametrize("key", CONSTANT)
def test_neff_matches_luna_to_rounding_when_the_cladding_matches(key):
    """With a constant cladding the two implementations share every input.

    The gas index is Börzsönyi's fit on both sides, supplied explicitly
    (``witness_materials``). What is left is the equation of state -- optmat's against
    CoolProp, and CODATA 2018 against 2014 -- which enters only through the ratio of
    the fill density to the fit's reference density and moves the refractivity by a
    small fraction of its value. In n_eff itself that is invisible: the refractivity
    is 1e-5 of unity, so 1e-15 is the whole budget and the observed agreement is
    4e-16, which is rounding.
    """
    spec = REFERENCE["index_cases"][key]
    got = build(spec).neff(OMEGA)
    assert np.max(np.abs(got / stored_index(key) - 1.0)) < 1e-14


@pytest.mark.parametrize("key", [key for key in CONSTANT if "noloss" not in key])
def test_the_loss_matches_luna_to_rounding_when_the_cladding_matches(key):
    """And it is loss on both sides: a sign slip would turn absorption into gain."""
    spec = REFERENCE["index_cases"][key]
    reference = stored_index(key)
    got = build(spec).neff(OMEGA)
    assert np.all(np.imag(reference) > 0.0)
    assert np.max(np.abs(np.imag(got) / np.imag(reference) - 1.0)) < 1e-14


@pytest.mark.parametrize("key", CONSTANT)
def test_the_refractivity_beats_lunas_by_the_cancellation_it_avoids(key):
    """The direct measurement of the cancellation in n_eff - 1, with Luna as control.

    n_eff agrees to 4e-16 above, so any disagreement in n_eff - 1 is the round trip
    through unity, and it is 1e-11 at a helium fill -- five orders worse than the
    quantity it is derived from. An earlier prototype measured the same cancellation
    capping a Chebyshev tail at 1.3e-12 instead of 1.9e-16, which is why solprop forms
    the refractivity directly. The gate has two sides: the difference must be small
    enough to be that and nothing else, and large enough that it is really happening.
    """
    spec = REFERENCE["index_cases"][key]
    difference = np.max(
        np.abs(build(spec).neff_minus_one(OMEGA) / (stored_index(key) - 1.0) - 1.0)
    )
    assert 1e-14 < difference < 1e-9


# ------------------------------------------- the effective index, cladding replaced


@pytest.mark.parametrize("key", SILICA)
def test_the_cladding_replacement_moves_neff_by_the_documented_amount(key):
    """Luna's default cladding is an unreferenced table; solprop's is cited.

    Measured over 250 nm to 2 um: the real part of the refractivity moves by at most
    1.5e-10 relative and the loss by at most 0.75 %. The loss is the sensitive one
    because it is carried entirely by v_n, which depends on the cladding permittivity
    alone, while the real part is dominated by the gas. Both bands are asserted from
    both sides, so this fails if the data are silently changed again -- in either
    direction, and whether the change makes the agreement worse or spuriously perfect.
    """
    spec = REFERENCE["index_cases"][key]
    reference = stored_index(key)
    got = build(spec).neff(OMEGA)
    refractivity = np.max(np.abs(np.real(got - 1.0) / np.real(reference - 1.0) - 1.0))
    assert 1e-13 < refractivity < 1e-9
    loss = np.max(np.abs(np.imag(got) / np.imag(reference) - 1.0))
    assert 1e-3 < loss < 2e-2


# ------------------------------------------------------------------------- ZDW


@pytest.mark.parametrize("key", sorted(REFERENCE["zdw_cases"]))
def test_zdw_matches_luna(key):
    """Luna's own gate on these is 1 nm; the agreement is 1e-7 relative, or 5e-5 nm.

    The residual is Luna's, not solprop's: its beta_2 is a seven-point finite
    difference of beta and its root finder stops at a relative tolerance of 1e-6. The
    gate here is 1e-6 relative, an order above the observed shift.
    """
    spec = REFERENCE["zdw_cases"][key]
    mode = MarcatiliMode(
        spec["radius_m"],
        spec["gas"],
        spec["pressure_bar"],
        susceptibility=gas_index(spec),
        kind="HE",
        n=1,
        m=1,
        model=spec["model"],
        loss=True,
        cladding=cladding(spec),
        temperature=LUNA_TEMPERATURE,
    )
    # Luna's search window, today's default, stated.
    wavelength = zdw(mode, wavelength_range=(100e-9, 3e-6))
    assert wavelength == pytest.approx(REFERENCE["zdw"][key][0], rel=1e-6)


# --------------------------------------------------- effective area and normalisation


@pytest.mark.parametrize("key", sorted(REFERENCE["area_cases"]))
def test_the_effective_area_matches_luna_exactly(key):
    """Purely geometric: no gas, no cladding, no constants -- so it must be rounding."""
    spec = REFERENCE["area_cases"][key]
    mode = MarcatiliMode(spec["radius_m"], kind=spec["kind"], n=spec["n"], m=spec["m"])
    assert mode.aeff() == pytest.approx(REFERENCE["area"][key][0], rel=1e-13)


@pytest.mark.parametrize("key", sorted(REFERENCE["area_cases"]))
def test_the_normalisation_matches_luna_to_the_constants(key):
    """The only difference is CODATA: Luna uses the 2014 set, scipy the 2018 one.

    N carries sqrt(eps_0/mu_0), whose value moved by about 1e-10 between the two
    releases, and the agreement is exactly that. Bit-identity with Luna is impossible
    for anything carrying a physical constant, which the physdata port already
    documents.
    """
    spec = REFERENCE["area_cases"][key]
    mode = MarcatiliMode(spec["radius_m"], kind=spec["kind"], n=spec["n"], m=spec["m"])
    assert mode.normalisation() == pytest.approx(REFERENCE["area"][key][1], rel=1e-9)
