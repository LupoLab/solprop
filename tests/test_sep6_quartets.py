# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep SEP 6's material record (``tests/data/sep6_quartets.json``) reproducible.

``tests/sep6_quartets.py`` evaluates optmat's hyperpolarizability models into the
tables of SEP 6, Part I. This test recomputes them and compares with the record, and
checks the two facts Phase 0 confirmed: solprop's SI conversion is optmat's, and the
leg bound of section 4.10 sits beyond every leg a quartet within the fitted sums can
have. A changed optmat model is reported as such: rerun the generator and re-check
the SEP's tables against the new record.
"""

import json
import math

import numpy as np
import pytest

from sep6_quartets import GASES, RECORD, _coefficients, tables

STORED = json.loads(RECORD.read_text())


@pytest.fixture(scope="module")
def current():
    return tables()


def flatten(table, prefix=""):
    if isinstance(table, dict):
        for key, value in table.items():
            yield from flatten(value, f"{prefix}/{key}")
    else:
        yield prefix, table


@pytest.mark.medium
def test_the_models_are_the_recorded_ones(current):
    """The tables are only comparable while optmat's fits are the recorded ones."""
    assert current["models"] == STORED["models"], (
        "optmat's hyperpolarizability models changed: rerun tests/sep6_quartets.py and "
        "re-check SEP 6's tables"
    )


@pytest.mark.medium
def test_every_table_reproduces(current):
    """Elementwise arithmetic and percentiles of fixed draws: equal to rounding.

    1e-12 relative allows the summation order of another NumPy build; the tables are
    otherwise the same operations on the same inputs.
    """
    stored = dict(flatten({k: STORED[k] for k in current if k != "models"}))
    for key, value in flatten({k: v for k, v in current.items() if k != "models"}):
        assert value == pytest.approx(stored[key], rel=1e-12, abs=1e-12), key


def test_solprop_si_conversion_is_optmats():
    """The Taylor factor and unit conversions agree with optmat's ``gamma3``.

    Both sides evaluate one formula in a different order; a few units in the last
    place, recorded at 3.3e-16, against a bound of 1e-14.
    """
    assert max(STORED["si_adapter"].values()) < 1e-14


@pytest.mark.parametrize("gas", GASES)
def test_leg_bound_lies_beyond_the_kerr_diagonal(gas):
    """``sqrt(3 W2_max / 4)`` is ``sqrt(3)`` times the Kerr diagonal's leg."""
    w2_max = _coefficients(gas)[3]
    bound_nm = STORED["leg_bounds_nm"][gas]
    diagonal_nm = 1e7 / math.sqrt(w2_max / 4)
    assert diagonal_nm / bound_nm == pytest.approx(math.sqrt(3), rel=1e-12)
    assert np.isfinite(bound_nm)
