# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a scan's results page draws."""

import numpy as np

from solprop_gui.model.scans import result_title, scan_lines, shown_results


def test_results_that_move_come_first_and_read_plainly() -> None:
    results = {
        "stats/core_radius": {"values": np.ones(3)},
        "stats/fwhm_t": {"values": np.arange(3.0)},
        "stats/energy": {"values": np.array([1.0, np.nan, 2.0])},
        "stats/species": {"values": np.array([0.0, 1.0, 2.0])},
    }
    assert shown_results(results) == ["stats/energy", "stats/fwhm_t", "stats/species"]
    assert result_title("stats/omega0") == "Centre frequency"
    assert shown_results({"stats/a": {"values": [np.nan]}}) == ["stats/a"]


def test_one_variable_is_one_line_with_its_failures() -> None:
    scan = {
        "variables": [{"name": "radius", "values": [1e-4, 2e-4, 3e-4]}],
        "status": ["done", "failed", "done"],
        "results": {"stats/energy": {"units": "J", "values": [1.0, np.nan, 3.0]}},
    }
    drawn = scan_lines(scan, "stats/energy", {"radius": "Core radius"})
    assert drawn.xlabel == "Core radius"
    assert drawn.xunit == "m"
    assert drawn.unit == "J"
    assert [title for title, _y in drawn.lines] == [""]
    assert drawn.failed.tolist() == [2e-4]


def test_a_third_variable_is_read_at_its_first_value() -> None:
    values = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    scan = {
        "variables": [
            {"name": "pressure", "values": [1.0, 2.0]},
            {"name": "gamma", "values": [1.0, 2.0, 3.0]},
            {"name": "seed", "values": [0.0, 1.0]},
        ],
        "status": np.full((2, 3, 2), "done", dtype=object),
        "results": {"stats/x": {"units": "", "values": values}},
    }
    drawn = scan_lines(scan, "stats/x")
    assert [title for title, _y in drawn.lines] == [
        "gamma 1",
        "gamma 2",
        "gamma 3",
    ]
    assert drawn.lines[2][1].tolist() == [4.0, 10.0]
    assert drawn.failed.size == 0
