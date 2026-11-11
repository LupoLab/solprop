# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the in-process measurement of the native session (SEP 47 section 7.3).

``benchmarks.native_session`` runs a ``sep47`` case through the native session; a short
window of the period-one row case must show what it was built to measure: rows taken
natively, with no return to Python for any of them.
"""

import pytest
from benchmarks.native_session import measure

pytest.importorskip("solprop_native")


@pytest.mark.parametrize("case", ["stats_period1_rk", "stats_period1_adams"])
def test_period_one_rows_take_no_return_to_python(case):
    native = measure(case, 1, 0.25)["native-session"]
    assert native["attempts"] > 0
    assert native["statistics_rows"] > 0
    # Rows taken natively: the native session returns at its planes and end alone,
    # and at its time bound (DEFAULT_DEADLINE), which a loaded host can reach between
    # two planes: a return the clock makes, not a row.
    returns = dict(native["native_returns"])
    yielded = returns.pop("session_progress:deadline", 0)
    assert native["batches"] == native["planes"] + 1 + yielded
    assert returns == {
        "session_observation": native["planes"],
        "session_terminal": 1,
    }
    assert sum(native["march_calls"].values()) >= native["statistics_rows"]
    assert native["field_bytes"] > 0
    assert native["seconds"]["median"] >= native["native_seconds"]["median"]


def test_a_refused_case_runs_admitted():
    record = measure("radial_filament_256_free_memory", 1, 0.02, admitted=True)
    assert record["admitted"]
    # The widths are finished natively now, and kept (SEP 47 section 6.4).
    assert {"beam_radius", "fwhm_r"} <= set(record["statistics"])
    assert record["native-session"]["attempts"] > 0
    assert record["native-session"]["field_bytes"] == 0
