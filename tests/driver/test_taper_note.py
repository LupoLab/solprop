# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin the taper-note campaign's small case on the production route.

The campaign itself is the study command
``tests/oracles/modal_design/taper_production.py``, run by hand; the receipt of its
small case is committed as ``tests/data/taper_note.json``. This module re-runs the one
case of it that is test-sized -- the 1 cm Robin taper at a quarter of the radius (the
paraxial similarity: radii times 0.25, lengths times 0.0625, so 31.25 to 15.625
micrometres over 0.625 mm between 0.3125 mm straight ends), four channels, twelve fixed
steps on ``numpy_ref`` -- and compares it with the numbers the study committed.

At four channels the retained basis is far from the FEM's answer (the study records
it: 35 % on ``HE_13``, because the connection is truncated at second order in the
contraction and more channels do not remove that), so the FEM is not the
comparison here; the full-scale rows of the study are. What this pins is the
production route's answer, and its agreement with the prototype's independent DOP853
solve of the same condensed model.
"""

import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from modal_oracles import register_modal_oracles

RECEIPT = Path(__file__).resolve().parents[1] / "data" / "taper_note.json"
EPSILON = float(np.finfo(float).eps)


def _study():
    """Import the taper study under the package name the oracles use."""
    register_modal_oracles()
    return importlib.import_module("modal_design.taper_production")


@pytest.mark.medium
def test_the_scaled_taper_note_case_reproduces_the_committed_study():
    """Twelve steps reproduce the study's number and the prototype's solve.

    Two bounds, stated first.

    * **The pin.** The run is the study's coarse schedule exactly, so the only
      admissible difference is rounding in a different LAPACK or BLAS build: about
      ``64 N eps`` per step for the four-channel products and exponentials, over twelve
      steps, 6.8e-13 of a fraction of order one. The tolerance is a hundredfold that,
      because a Schur decomposition's blocking may differ between builds.
    * **The model.** The prototype's DOP853 solve (``taper.modal``, rtol 1e-10) is an
      independent integration of the same condensed equation. Twelve steps depart
      from it by at most their own schedule error plus the finer schedule's, both
      measured by the study: ``|f_12 - f_24| + |f_24 - f_proto|``, with half again
      for the fifth-order estimate being crude at twelve steps.
    """
    study = _study()
    receipt = json.loads(RECEIPT.read_text())
    row = receipt["rows"][0]
    committed = row["production"]
    case = study.CASES[receipt["case"]["name"]]
    assert committed["coarse"]["steps"] == 12
    assert committed["count"] == 4

    result = study.production(case, 4, case.per_straight, committed["validity_limit"])

    fractions = np.asarray(result["fractions"])
    assert result["steps"] == 12
    assert np.all(np.isfinite(fractions))
    pinned = 12 * 64 * 4 * EPSILON
    assert pinned < 1e-12
    np.testing.assert_allclose(
        fractions, committed["coarse"]["fractions"], rtol=0.0, atol=100 * pinned
    )

    prototype = np.asarray(receipt["fem"]["prototype"][0]["fractions"])
    fine = np.asarray(committed["fractions"])
    schedule_bound = np.max(abs(np.asarray(committed["coarse"]["fractions"]) - fine))
    bound = 1.5 * (schedule_bound + np.max(abs(fine - prototype)))
    assert 0.0 < bound < 1e-3
    assert np.max(abs(fractions - prototype)) <= bound
    # The power is the sum of the pinned fractions: four times their bound.
    assert result["power"] == pytest.approx(
        committed["coarse"]["power"], abs=4 * 100 * pinned
    )
