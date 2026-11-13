# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spatial checks meet their derived bounds; the record keeps each series' margin."""

import pytest

import spatial_records as records
from convergence_records import RecordError


@pytest.fixture
def captured(monkeypatch):
    written = []
    monkeypatch.setattr(
        records, "record", lambda *args, **kwargs: written.extend(kwargs["rows"])
    )
    monkeypatch.setattr(records, "_LARGEST", {})
    return written


def test_a_measurement_within_its_bound_records_its_fraction(captured, monkeypatch):
    monkeypatch.setattr(records, "writing", lambda: False)
    records.record_metric(None, "error", "case", 0.5, 2.0)
    assert captured == [
        {"series": "numpy_ref: error", "axis": "largest", "fraction": 0.25}
    ]


def test_a_write_session_keeps_each_series_largest_fraction(captured, monkeypatch):
    monkeypatch.setattr(records, "writing", lambda: True)
    for measured in (0.5, 1.5, 1.0):
        records.record_metric(None, "error", measured, measured, 2.0)
    records.record_metric(None, "other", "case", 0.2, 2.0)
    assert [row["fraction"] for row in captured] == [0.25, 0.75, 0.75, 0.1]


@pytest.mark.parametrize(
    "measured,bound",
    [(1.1, 1.0), (float("nan"), 1.0), (0.0, float("inf")), (0.0, 0.0), (-0.1, 1.0)],
)
def test_a_miss_or_an_invalid_error_or_bound_is_refused(captured, measured, bound):
    with pytest.raises(RecordError, match="against the derived bound"):
        records.record_metric(None, "error", "case", measured, bound)
    assert not captured
