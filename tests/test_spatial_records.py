# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep physical error limits fixed while reporting local bound evaluation."""

import pytest

import spatial_records as records
from convergence_records import RecordError


@pytest.fixture
def captured(monkeypatch, tmp_path):
    path = tmp_path / "records.json"
    path.write_text("existing record")
    monkeypatch.setattr(records, "path_of", lambda _: path)
    monkeypatch.setattr(
        records,
        "load",
        lambda _: {
            "rows": [{"series": "numpy_ref: error", "axis": "case", "bound": 1.0}]
        },
    )
    written = []
    monkeypatch.setattr(
        records, "record", lambda *args, **kwargs: written.extend(kwargs["rows"])
    )
    return written


@pytest.mark.parametrize("derived", [0.75, 1.25])
def test_spatial_record_preserves_fixed_limit_and_reports_local_bound(
    captured, derived
):
    records.record_metric(None, "error", "case", 0.5, derived)
    assert captured == [
        {
            "series": "numpy_ref: error",
            "axis": "case",
            "deviation": 0.5,
            "bound": 1.0,
            "derived_bound": derived,
        }
    ]


@pytest.mark.parametrize("measured,derived", [(1.1, 1.25), (0.9, 0.75)])
def test_spatial_record_rejects_a_miss_of_either_limit(captured, measured, derived):
    with pytest.raises(RecordError, match=r"fixed limit 1\.0"):
        records.record_metric(None, "error", "case", measured, derived)
    assert not captured


def test_new_spatial_case_uses_its_predeclared_bound(captured):
    records.record_metric(None, "new error", "case", 0.5, 0.75)
    assert captured[0]["bound"] == captured[0]["derived_bound"] == 0.75


@pytest.mark.parametrize(
    "measured,derived",
    [(float("nan"), 1.0), (0.0, float("inf")), (0.0, 0.0), (-0.1, 1.0)],
)
def test_spatial_record_rejects_invalid_error_or_bound(captured, measured, derived):
    with pytest.raises(RecordError):
        records.record_metric(None, "error", "case", measured, derived)
    assert not captured
