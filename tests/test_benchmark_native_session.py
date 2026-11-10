# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the in-process comparison of the native and driver sessions (SEP 47 7.4).

``benchmarks.native_session`` runs a ``sep47`` case through both sessions; a short
window of the period-one row case must show what it was built to measure: the driver's
returns to Python for each row, none for the native session's, and the same work.
"""

import pytest
from benchmarks.native_session import measure

pytest.importorskip("solprop_native")


@pytest.mark.parametrize("case", ["stats_period1_rk", "stats_period1_adams"])
def test_period_one_rows_return_to_python_only_on_the_driver(case):
    record = measure(case, 1, 0.25)
    driver, native = record["driver"], record["native-session"]
    assert driver["attempts"] == native["attempts"] > 0
    assert driver["statistics_rows"] == native["statistics_rows"] > 0
    assert driver["native_returns"]["host_step:statistics"] > 0
    # Rows taken natively: the native session returns at its planes and end alone.
    assert native["batches"] == native["planes"] + 1
    assert native["native_returns"] == {
        "session_observation": native["planes"],
        "session_terminal": 1,
    }
    assert sum(native["march_calls"].values()) >= native["statistics_rows"]
    assert native["field_bytes"] == driver["field_bytes"] > 0
    for executor in (driver, native):
        assert executor["seconds"]["median"] >= executor["native_seconds"]["median"]


def test_a_refused_case_runs_admitted_for_both():
    record = measure("radial_filament_256_free_memory", 1, 0.02, admitted=True)
    assert record["admitted"]
    assert "beam_radius" not in record["statistics"]
    assert record["driver"]["attempts"] == record["native-session"]["attempts"]
    assert record["native-session"]["field_bytes"] == 0
