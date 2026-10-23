# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep SEP 3's attributed changed-results record whole (``sep3_attribution.json``).

The attribution script, run by hand at solprop ``8d116c8`` and deleted in SEP 3's
Phase 4b with the registry it switched families on, attributed the change of each
frozen witness (``sep3_baseline.*``) to the material families one at a time, checked
the numerical error of the comparison by step refinement, and recorded the solids' GVD
changes and the decomposition of the rate ratios (SEP 3, Phase 4a, second part). The
record is now a historical one. This test re-runs no propagation: it checks that the
record is well formed, that its all-registry rows are the frozen witnesses, and that
its numbers are consistent with one another. Its SHA-256 comparisons are provenance (a
record naming the stored bytes it was made against), computed from stored files, never
from a fresh computation.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"
RECORD = json.loads((DATA / "sep3_attribution.json").read_text())
BASELINE = json.loads((DATA / "sep3_baseline.json").read_text())
CASES = sorted(BASELINE["cases"])
FAMILIES = list(RECORD["families"])


def test_the_record_names_its_revisions_and_every_part():
    """The record pins solprop's commit, the witnesses' commit and optmat's version."""
    assert len(RECORD["solprop_commit"]) == 40
    assert RECORD["baseline_commit"] == BASELINE["solprop_commit"]
    assert RECORD["optmat_version"] == RECORD["versions"]["optmat"]
    assert RECORD["steps"] == 10 and RECORD["refined_steps"] == 20
    assert RECORD["backend"] == "numpy_ref"
    for part in ("propagation", "refinement", "solid_gvd", "capillary_gvd", "rates"):
        assert part in RECORD
    assert sorted(RECORD["propagation"]) == CASES
    assert sorted(RECORD["refinement"]) == CASES


@pytest.mark.parametrize("case", CASES)
def test_the_registry_rows_are_the_frozen_witnesses(case):
    """The registry rows name the stored witness by its digest, as the record says.

    The record states that the registry runs reproduced the witness at the time
    (``reproduced``, a zero difference); the digest ties those rows to the stored
    ``sep3_baseline.npz`` they were compared with: provenance, exactly.
    """
    entry = RECORD["propagation"][case]
    registry = entry["runs"]["registry"]
    with np.load(DATA / "sep3_baseline.npz") as arrays:
        witness = np.ascontiguousarray(arrays[f"{case}/field_out"], dtype=np.complex128)
    sha = hashlib.sha256(witness.tobytes()).hexdigest()
    assert entry["baseline"]["reproduced"] is True
    assert entry["baseline"]["max_abs_difference"] == 0.0
    assert entry["baseline"]["field_out_sha256"] == sha
    assert registry["field_out_sha256"] == sha
    assert registry["switched"] == [] and not registry["default"]
    assert registry["models"] == {}
    assert all(value == 0.0 for value in registry["change"].values())


@pytest.mark.parametrize("case", CASES)
def test_each_family_is_run_or_said_not_to_apply(case):
    """A family that applies has its row; one that does not has its reason."""
    entry = RECORD["propagation"][case]
    assert list(entry["applicable"]) == FAMILIES
    for family, reason in entry["applicable"].items():
        if reason is None:
            assert entry["runs"][family]["switched"] == [family]
        else:
            assert isinstance(reason, str) and reason
            assert family not in entry["runs"]
    default = entry["runs"]["default"]
    assert default["default"] is True
    assert entry["default_equals_all_families"] == 0.0
    for row in entry["runs"].values():
        assert math.isfinite(row["change"]["spectrum"])
        assert {"energy_j", "centroid_thz", "rms_width_thz"} <= set(row["observables"])
    if any(reason is None for reason in entry["applicable"].values()):
        # The default run records an optmat model for every family it switched.
        assert default["models"]
        assert all(len(digest) == 64 for digest in default["models"].values())


@pytest.mark.parametrize("case", CASES)
def test_the_attributed_changes_stand_above_the_comparison_error(case):
    """Refined from ten to twenty steps, a resolved change moves far less than itself.

    A change of the spectrum above 1e-6 is resolved when the comparison's own change
    with refinement is at most a tenth of it.
    """
    entry = RECORD["refinement"][case]
    assert entry["steps"] == [10, 20]
    for comparison in entry["comparisons"].values():
        if comparison["attributed_20"] > 1e-6:
            assert comparison["ratio"] is not None
            assert comparison["ratio"] > 10.0


@pytest.mark.parametrize("gas", sorted(RECORD["rates"]["gases"]))
def test_each_rate_decomposition_closes(gas):
    """The chain ends on optmat's rate, and its factors multiply to the whole ratio."""
    entry = RECORD["rates"]["gases"][gas]
    assert entry["closure"] < 1e-9
    factors = np.prod([entry["factors"][link] for link in entry["links"]], axis=0)
    np.testing.assert_allclose(factors, entry["ratio"], rtol=1e-9)
    grouped = np.prod(list(entry["groups"].values()), axis=0)
    np.testing.assert_allclose(grouped, entry["ratio"], rtol=1e-9)
    # solprop's own form against optmat's named form of it: the same rate.
    np.testing.assert_allclose(entry["groups"]["parity"], 1.0, rtol=1e-8)
    assert set(entry["links"]) <= set(RECORD["rates"]["links"])
    # The ratio decomposed is the one sep3_changes.json measured, of the same model.
    changes = json.loads((DATA / "sep3_changes.json").read_text())["rates"][gas]
    assert entry["digest"] == changes["digest"]
    measured = [row["ratio"] for row in changes["ratios"].values()]
    np.testing.assert_allclose(entry["ratio"], measured, rtol=1e-12)


def test_every_solid_and_axis_has_its_gvd_at_every_wavelength():
    """Five wavelengths per solid; the new rays have no former model to compare."""
    for name, entry in RECORD["solid_gvd"].items():
        assert len(entry["points"]) == 5
        for point in entry["points"].values():
            assert point["optmat"]["in_range"]
            assert max(point["optmat"]["difference_error"]) < 1e-2
            if name.endswith(" e"):
                assert point["solprop"] is None
            else:
                assert point["solprop"]["in_range"]
                assert "beta2_difference_fs2_per_mm" in point
    capillary = RECORD["capillary_gvd"]
    assert abs(capillary["beta2_difference_fs2_per_m"]) < 1e-3
