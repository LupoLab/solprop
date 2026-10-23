# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The scan worker keeps only the fields a scan reads (SEP 22 section 5.1, Q3).

A point and the same simulation run alone are one discrete march on a prescribed
schedule at a fixed order, so the scan's stored arrays are compared with the direct run
within that march's rounding budget (``march_bounds``), whatever fields the worker kept.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from driver.test_field_free_planes import cartesian, low_storage
from driver.test_scans import build_point
from march_bounds import STATISTIC_FACTOR, assert_matched, simulation_bound
from solprop.driver._scan_run import _Capture, _Options, _retained
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import OutputError
from solprop.output.schema import FieldsChoice
from solprop.processing import scans


def _statistics_only(output) -> np.ndarray:
    """Read the last plane's energy, and prove the spooled run kept no field history.

    The spool keeps the fields the scan keeps anyway (a built-in reduction's target
    plane), and no other.
    """
    with pytest.raises(OutputError, match="this run was saved with fields="):
        output["field"]
    return np.asarray(output["stats/energy"][-1].sum())


def _reads_the_field(output) -> np.ndarray:
    """Read the last plane's field, which a reduction declaring so may."""
    return np.asarray(np.abs(output.fields()[-1]).sum())


def retention_of(options: _Options, tmp_path: Path):
    simulation = build_point(Scan("scan", power=[1.0]).point(0))
    capture = _Capture(options, simulation, tmp_path)
    return _retained(simulation, capture).settings


@pytest.mark.parametrize(
    ("fields", "field_planes", "reductions", "expected"),
    [
        ("all", (), (), ("all", ())),
        ("none", (), (), ("none", ())),
        ("planes", (0, -1), (), ("planes", (0, 2))),
        ("none", (), (scans.peak_power(plane=1),), ("planes", (1,))),
        ("planes", (0,), (scans.mode_energies(plane=-1),), ("planes", (0, 2))),
        (
            "none",
            (),
            (scans.custom("s", _statistics_only, (), "J", reads_fields=False),),
            ("none", ()),
        ),
        ("none", (), (scans.custom("f", _reads_the_field, (), "1"),), ("all", ())),
    ],
)
def test_the_worker_keeps_the_fields_the_scan_reads(
    tmp_path, fields, field_planes, reductions, expected
):
    options = _Options(tuple(reductions), fields, field_planes, "all")
    settings = retention_of(options, tmp_path)
    assert (settings.fields, tuple(settings.field_planes)) == expected


def test_the_worker_keeps_only_the_scanned_fields_beside_the_low_storage_march(
    tmp_path,
):
    # That march holds its last field-free plane (SEP 22, Q14), so the worker keeps
    # what the scan reads, as for any other march.
    simulation = low_storage()
    options = _Options((), "none", (), "all")
    retained = _retained(simulation, _Capture(options, simulation, tmp_path))
    assert retained is not simulation
    assert retained.settings.fields == "none"
    assert simulation.settings.fields == "all"


def test_a_field_free_scan_stores_the_point_s_own_statistics(tmp_path):
    scan = Scan("scan", power=[1.0, 2.0])
    reductions = [
        scans.peak_power(),
        scans.custom("energy", _statistics_only, (), "J", reads_fields=False),
        scans.custom("abs_sum", _reads_the_field, (), "1"),
    ]
    rules: tuple[tuple[FieldsChoice, tuple[int, ...]], ...] = (
        ("none", ()),
        ("planes", (-1,)),
    )
    for fields, field_planes in rules:
        result = run_scan(
            scan,
            build_point,
            output=tmp_path / f"{fields}.h5",
            reductions=reductions[:2] if fields == "none" else reductions[::2],
            fields=fields,
            field_planes=field_planes,
            progress="none",
        )
        assert result.status.tolist() == ["done", "done"]
        for index in range(2):
            direct = build_point(scan.point(index)).run(progress="none")
            point = result.point(index)
            bound = STATISTIC_FACTOR * simulation_bound(build_point(scan.point(index)))
            assert_matched(point["stats/energy"], direct["stats/energy"], bound)
            assert_matched(point["reduced/peak_power"], reductions[0](direct), bound)
            if fields == "none":
                assert_matched(
                    point["reduced/energy"],
                    np.asarray(direct["stats/energy"][-1].sum()),
                    bound,
                )
            else:
                assert_matched(
                    point["reduced/abs_sum"], _reads_the_field(direct), bound
                )
                assert_matched(
                    point["fields/values"],
                    direct.fields()[[-1]],
                    simulation_bound(build_point(scan.point(index))),
                )


def test_a_builder_that_sets_fields_is_refused_by_name(tmp_path):
    def keeping_fields(point):
        simulation = build_point(point)
        simulation.settings = dataclasses.replace(simulation.settings, fields="none")
        return simulation

    result = run_scan(
        Scan("scan", power=[1.0]),
        keeping_fields,
        output=tmp_path / "scan.h5",
        progress="none",
    )
    assert result.status.tolist() == ["failed"]
    assert "drop fields and field_planes" in str(result.point(0)["error"])


@pytest.mark.medium
def test_a_statistic_reduction_reads_a_field_free_plane(tmp_path):
    def build(point):
        return cartesian("numpy_ref")

    result = run_scan(
        Scan("scan", power=[1.0]),
        build,
        output=tmp_path / "scan.h5",
        reductions=[scans.beam_radius(), scans.on_axis_intensity(plane=1)],
        fields="none",
        progress="none",
    )
    assert result.status.tolist() == ["done"]
    direct = cartesian("numpy_ref").run(progress="none")
    point = result.point(0)
    # One process, one backend, one replayed march: the scan's reduction is the
    # direct run's statistic to the replay's rounding (relative 1e-9 is a thousand
    # times the case's ten-step replay bound).
    assert_matched(point["reduced/beam_radius"], direct["stats/beam_radius"][-1], 1e-9)
    assert_matched(
        point["reduced/on_axis_intensity"], direct["stats/on_axis_intensity"][1], 1e-9
    )
