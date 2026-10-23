# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep SEP 3's frozen baseline witnesses whole (``tests/data/sep3_baseline.*``).

The witnesses are short runs on the material defaults before SEP 3 changed them; Phase
4a attributed each change against them. Their generator last ran at solprop
``eae0ed8`` and was deleted in Phase 4b with the data it ran on, so the arrays are a
historical record. This test checks only that the record and the arrays agree with each
other, never their physics.
"""

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / "data"
RECORD = json.loads((DATA / "sep3_baseline.json").read_text())
CASES = sorted(RECORD["cases"])


def test_the_record_names_its_commit_and_every_case():
    """The record pins a full solprop commit and the generator's nine cases."""
    assert len(RECORD["solprop_commit"]) == 40
    assert len(CASES) == 9


@pytest.mark.parametrize("case", CASES)
def test_each_case_has_finite_input_and_output_spectra(case):
    """Every case stores two planes of one shape, finite, on its declared grid."""
    with np.load(DATA / "sep3_baseline.npz") as arrays:
        field_in = arrays[f"{case}/field_in"]
        field_out = arrays[f"{case}/field_out"]
        z = arrays[f"{case}/z"]
    entry = RECORD["cases"][case]
    assert field_in.shape == field_out.shape
    assert field_in.shape[-1] == entry["grid"]["n_omega"]
    assert np.all(np.isfinite(field_in)) and np.all(np.isfinite(field_out))
    assert z[0] == 0.0 and z[-1] > 0.0
    assert entry["steps"] == 10 and entry["backend"] == "numpy_ref"
