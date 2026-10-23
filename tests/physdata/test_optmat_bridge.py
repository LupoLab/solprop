# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The optmat bridge: records, units, the warning and error policy, lazy import."""

import math
import subprocess
import sys
import warnings

import numpy as np
import pytest
from optmat.conditions import State
from optmat.exceptions import ModelAbsorptionWarning, ModelRangeWarning, StateError
from optmat.gas.eos import equation_of_state
from optmat.gas.models.registry import default_model, model
from optmat.gas.transport import transport_model

from solprop.constants import BAR, C
from solprop.exceptions import (
    MaterialError,
    MaterialRangeWarning,
    MaterialStatusWarning,
)
from solprop.physdata import _optmat
from solprop.physdata.selection import (
    DiluteScalingCheck,
    MaterialRecord,
    MaterialSelection,
    _choice,
)

ARGON = default_model("Ar")


def test_importing_solprop_and_physdata_does_not_import_optmat():
    # optmat's registries take more than a second to import; solprop imports them
    # only where a material is resolved (SEP 3, Phase 0).
    code = (
        "import sys, solprop, solprop.physdata, solprop.physdata.mixtures, "
        "solprop.physdata.selection, solprop.modes; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in "
        "('optmat', 'CoolProp')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert out.stdout.strip() == "[]"


def test_bar_converts_to_pascal_exactly_and_back():
    # The interface unit is bar; optmat takes pascals, by the exact factor 1e5.
    assert BAR == 1e5
    for value in (0.0, 1.0, 3.7, 60.0, 1e-9):
        assert _optmat.bar_to_pa(value) == value * 1e5
        assert _optmat.pa_to_bar(_optmat.bar_to_pa(value)) == pytest.approx(
            value, rel=2e-16
        )


def test_angular_frequency_and_vacuum_wavelength_are_inverse_with_the_static_limit():
    omega = np.array([0.0, 1e14, 2.356e15, 1e17])
    wavelength = _optmat.omega_to_wavelength(omega)
    assert math.isinf(wavelength[0])
    np.testing.assert_allclose(wavelength[1:], 2 * math.pi * C / omega[1:], rtol=0)
    # One division each way: two roundings.
    np.testing.assert_allclose(
        _optmat.wavelength_to_omega(wavelength), omega, rtol=4e-16, atol=0
    )


def test_selection_defaults_and_validation():
    selection = MaterialSelection()
    assert (selection.linear, selection.electronic, selection.raman) == (
        "recommended",
        "recommended",
        "recommended",
    )
    assert selection.ionisation == "recommended"
    assert selection.energy_loss == "peak-weighted"
    assert selection.raman_shift is None
    assert MaterialSelection(linear="  fit/Ar ").linear == "fit/Ar"
    assert MaterialSelection(linear=ARGON).linear is ARGON
    assert MaterialSelection(carrier_wavelength=800e-9).carrier_wavelength == 800e-9
    assert MaterialSelection(raman_shift="reference-density").raman_shift


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"linear": ""}, "empty string"),
        ({"raman": 3}, "optmat model object, got int"),
        ({"electronic": "full"}, "electronic must be one of"),
        ({"energy_loss": "mean"}, "energy_loss must be one of"),
        ({"raman_shift": "always"}, "raman_shift must be one of"),
        ({"carrier_wavelength": -1e-6}, "carrier_wavelength"),
        ({"carrier_wavelength": math.inf}, "carrier_wavelength"),
        ({"carrier_wavelength": True}, "carrier_wavelength"),
    ],
)
def test_selection_refuses_invalid_fields(changes, match):
    with pytest.raises(MaterialError, match=match):
        MaterialSelection(**changes)


def test_a_model_choice_is_a_name_or_an_identified_object():
    assert _choice(ARGON, "linear") is ARGON
    with pytest.raises(MaterialError, match="optmat model object, got float"):
        _choice(1.5, "raman")


def test_a_record_needs_an_optmat_identity():
    with pytest.raises(MaterialError, match="a str is not an optmat model"):
        _optmat.record("fit/Ar")


def test_selection_record_holds_names_and_model_identities():
    record = MaterialSelection(linear=ARGON, carrier_wavelength=1e-6).to_dict()
    assert record["raman"] == "recommended"
    assert record["carrier_wavelength"] == 1e-6
    assert isinstance(record["linear"], dict)
    assert record["linear"]["digest"] == ARGON.identity.digest


def test_material_record_is_the_identity_in_plain_values():
    entry = _optmat.record(ARGON)
    identity = ARGON.identity
    assert entry == MaterialRecord.from_identity(identity)
    assert (entry.name, entry.kind, entry.material) == (
        "fit/Ar",
        "gas_refractive",
        "Ar",
    )
    assert entry.digest == identity.digest
    assert entry.status == "validated"
    assert entry.optmat_version == identity.optmat_version == _optmat.optmat_version()
    as_dict = entry.to_dict()
    assert as_dict["modified"] == [] and as_dict["base"] == ""
    assert set(as_dict) == {
        "name",
        "kind",
        "material",
        "digest",
        "status",
        "status_reason",
        "optmat_version",
        "modified",
        "base",
        "sources",
        "reference_temperature",
    }
    assert as_dict["sources"] == list(identity.sources) and as_dict["sources"]
    assert as_dict["reference_temperature"] == identity.reference_temperature


def test_an_override_record_carries_status_user_its_changes_and_its_base():
    narrowed = ARGON.with_parameters(intervals=((120e-9, math.inf),))
    entry = _optmat.record(narrowed)
    assert entry.status == "user"
    assert entry.base == ARGON.identity.digest
    assert entry.digest != ARGON.identity.digest
    assert entry.modified == (("intervals", '[[1.2e-07,"inf"]]'),)
    assert entry.to_dict()["modified"] == [["intervals", '[[1.2e-07,"inf"]]']]


@pytest.mark.parametrize(
    ("changes", "match"),
    [({"digest": "abc"}, "64 hexadecimal"), ({"status": "trusted"}, "status")],
)
def test_material_record_refuses_a_malformed_digest_or_status(changes, match):
    fields = _optmat.record(ARGON).to_dict() | changes
    fields["modified"] = ()
    fields["sources"] = ()
    with pytest.raises(MaterialError, match=match):
        MaterialRecord(**fields)


def test_dilute_check_record():
    check = DiluteScalingCheck("Ar", "Ar", 2.5e25, 294.0, 8e-7, -1.9e-4, 1e-3)
    assert check.measured and not check.exceeded
    assert dict(check.to_dict())["relative_error"] == -1.9e-4
    unmeasured = DiluteScalingCheck(
        "Ar", "Ar", 2.5e25, 250.0, math.nan, math.nan, 1e-3, "optmat refused"
    )
    assert not unmeasured.measured and not unmeasured.exceeded
    record = dict(unmeasured.to_dict())
    assert record["relative_error"] is None and record["wavelength"] is None
    assert record["reason"] == "optmat refused" and record["measured"] is False


def test_a_warning_is_raised_once_per_evaluation_context():
    with pytest.warns(MaterialRangeWarning) as caught, _optmat.material_context():
        for _ in range(3):
            _optmat.warn_once(("x",), "once", MaterialRangeWarning)
        # A nested context shares the outer one.
        with _optmat.material_context():
            _optmat.warn_once(("x",), "once", MaterialRangeWarning)
    assert len(caught) == 1
    # Outside any context, each call is its own.
    with pytest.warns(MaterialRangeWarning) as caught:
        _optmat.warn_once(("x",), "again", MaterialRangeWarning)
        _optmat.warn_once(("x",), "again", MaterialRangeWarning)
    assert len(caught) == 2


def test_status_policy_warns_for_provisional_and_records_the_rest():
    methane = transport_model("CH4")
    assert methane.status == "provisional"
    with pytest.warns(MaterialStatusWarning, match="provisional"):
        entry = _optmat.check_status(methane, "CH4")
    assert entry.status == "provisional"
    # A curated alternative and a caller's override are chosen, so recorded silently.
    comparison = model("borzsonyi_2008/sellmeier_Ar")
    assert _optmat.check_status(comparison, "Ar").status == "comparison"
    narrowed = ARGON.with_parameters(intervals=((120e-9, math.inf),))
    assert _optmat.check_status(narrowed, "Ar").status == "user"


def test_outside_parts_names_each_part_and_gap_reason():
    intervals = ((100e-9, 200e-9), (300e-9, 400e-9), (500e-9, 600e-9))
    inside = np.array([150e-9, 350e-9, 550e-9])
    assert _optmat.outside_parts(inside, intervals) == ()
    band = np.array([50e-9, 250e-9, 450e-9, 700e-9, np.nan])
    parts = _optmat.outside_parts(band, intervals, ("absorption",))
    assert len(parts) == 4
    assert "below its shortest" in parts[0] and "50 nm" in parts[0]
    assert "a gap (absorption)" in parts[1]
    assert "a gap (unqualified)" in parts[2]
    assert "beyond its longest" in parts[3]
    static = _optmat.outside_parts(np.array([math.inf]), intervals)
    assert static == ("600 nm to inf, beyond its longest qualified wavelength",)


def test_translated_turns_optmat_range_warnings_into_one_material_warning():
    band = np.array([50e-9, 800e-9])
    with (
        pytest.warns(MaterialRangeWarning) as caught,
        _optmat.translated(ARGON, "polarizability", material="Ar", wavelength=band),
    ):
        ARGON.complex_polarizability(band)
        ARGON.dpolarizability_domega(band)
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "fit/Ar" in message and "below its shortest" in message
    assert "optmat:" in message


def test_translated_without_a_band_names_the_range():
    with (
        pytest.warns(MaterialRangeWarning, match="qualified range"),
        _optmat.translated(ARGON, "polarizability", material="Ar"),
    ):
        warnings.warn("outside", ModelRangeWarning, stacklevel=1)


def test_translated_passes_other_warnings_through():
    with (
        pytest.warns(RuntimeWarning, match="unrelated"),
        _optmat.translated(ARGON, "polarizability", material="Ar"),
    ):
        warnings.warn("unrelated", RuntimeWarning, stacklevel=1)


def test_translated_converts_absorption_warnings():
    with (
        pytest.warns(MaterialRangeWarning, match="Ar: absorbs"),
        _optmat.translated(ARGON, "polarizability", material="Ar"),
    ):
        warnings.warn("absorbs", ModelAbsorptionWarning, stacklevel=1)


def test_translated_turns_optmat_errors_into_material_errors():
    sf6 = equation_of_state("SF6")
    with (
        pytest.raises(MaterialError, match="SF6: optmat refuses the number density"),
        _optmat.translated(
            sf6, "number density", material="SF6", state="30 bar and 294 K"
        ),
    ):
        sf6.number_density(State(30e5, 294.0))
    with (
        pytest.raises(MaterialError) as error,
        _optmat.translated(sf6, "number density", material="SF6"),
    ):
        raise StateError("refused")
    assert isinstance(error.value.__cause__, StateError)


def test_named_compositions_and_their_refusal():
    air = _optmat.composition_fractions("dry air (N2, O2, Ar)")
    assert list(air) == ["N2", "O2", "Ar"]
    assert math.fsum(air.values()) == pytest.approx(1.0, abs=4e-16)
    with pytest.raises(MaterialError, match="no composition"):
        _optmat.composition_fractions("wet air")
