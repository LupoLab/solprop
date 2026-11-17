# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a scan records at each point beyond its statistics, and how it reads back."""

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from service.cases import TINY
from service.test_service import Collector
from solprop.driver.scan import run_scan
from solprop.recipe import Recipe
from solprop.service import results
from solprop.service.run import _point_value
from solprop.service.scan_reductions import (
    AXIS_POINTS,
    reduction_axis,
    reductions,
)
from solprop.service.worker import RunSpec

SPECS: Any = [
    {"kind": "energy_in_band", "band": [900e-9, 600e-9]},
    {"kind": "peak_power_in_band", "band": [600e-9, 900e-9]},
    {"kind": "spectrum", "z": 0.025},
    {"kind": "pulse"},
]


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> str:
    text = TINY + '\n[scan]\nname = "p"\npressure = ["0.5 bar", "2 bar"]\n'
    recipe = Recipe.loads(text)
    scan = recipe.scan()
    path = tmp_path_factory.mktemp("recorded") / "scan.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_scan(
            scan.scan,
            scan.build,
            output=path,
            reductions=reductions(SPECS, recipe),
            progress="none",
            **scan.keywords,  # pyright: ignore[reportArgumentType]
        )
    return str(path)


def test_each_reduction_is_recorded_at_every_point(recorded: str) -> None:
    found: Any = results.scan_overview({"path": recorded})
    got = found["results"]
    energy = got["reduced/energy_in_band"]
    assert energy["units"] == "J"
    # Most of a 100 uJ pulse at 800 nm lies within 600 to 900 nm.
    assert np.all((energy["values"] > 80e-6) & (energy["values"] < 100e-6))
    assert np.all(got["reduced/peak_power_in_band"]["values"] > 1e9)
    spectrum = got["reduced/spectrum"]
    assert spectrum["values"].shape == (2, AXIS_POINTS)
    peak = spectrum["x"][int(np.argmax(spectrum["values"][0]))]
    assert 700e-9 < peak < 900e-9
    pulse = got["reduced/pulse"]
    assert pulse["x"][int(np.argmax(pulse["values"][0]))] == pytest.approx(
        0, abs=20e-15
    )


def test_the_axes_are_even_and_cover_the_run_s_window() -> None:
    recipe = Recipe.loads(TINY)
    wavelength = reduction_axis(recipe, "spectrum")
    assert np.allclose(np.diff(wavelength), np.diff(wavelength)[0])
    assert wavelength[0] < 800e-9 < wavelength[-1]
    t = reduction_axis(recipe, "pulse")
    assert t[0] < 0 < t[-1]


def test_a_point_of_a_gradient_reads_back_in_its_unit(tmp_path: Path) -> None:
    text = TINY.replace('pressure = "1 bar"', 'pressure = ["1 bar", "0 bar"]') + (
        '\n[scan]\nname = "g"\n"pressure[0]" = ["0.5 bar", "2 bar"]\n'
    )
    spec = RunSpec("g1", str(tmp_path / "g1"), recipe=text, base=str(tmp_path))
    collector = Collector(spec).start()
    collector.collect()
    started = next(e.data for e in collector.events if e.kind == "scan_started")
    assert started["names"] == ["pressure[0]"]
    assert started["values"] == [["0.5 bar", "2 bar"]]
    first = next(e.data for e in collector.events if e.kind == "point")
    assert first["values"] == {"pressure[0]": "0.5 bar"} and first["place"] == [0]
    final = collector.events[-1].data
    found: Any = results.scan_overview({"path": final["path"]})
    axis = found["variables"][0]
    assert axis["values"].tolist() == [0.5, 2.0]
    assert axis["unit"] == "bar"


def test_a_range_is_written_as_numbers_and_reductions_reach_the_worker(
    tmp_path: Path,
) -> None:
    text = TINY + (
        '\n[scan]\nname = "e"\nenergy = { start = "50 uJ", stop = "100 uJ", num = 2 }\n'
    )
    spec = RunSpec(
        "e1",
        str(tmp_path / "e1"),
        recipe=text,
        base=str(tmp_path),
        reductions=({"kind": "peak_power_in_band", "band": [600e-9, 900e-9]},),
    )
    collector = Collector(spec).start()
    collector.collect(until="point", then="stop")
    started = next(e.data for e in collector.events if e.kind == "scan_started")
    assert started["values"] == [["5e-05", "0.0001"]]
    assert _point_value("x") == "x"


def test_a_beam_s_scan_is_refused_the_readers_of_a_guided_run() -> None:
    beam = Recipe.example("filament")
    band = {"kind": "energy_in_band", "band": [600e-9, 900e-9]}
    with pytest.raises(ValueError, match="statistics only"):
        reductions([band], beam)
    assert reductions([], beam) == []  # its statistics alone are fine


def test_a_reduction_of_no_known_kind_is_refused():
    from solprop.service.scan_reductions import _reader  # noqa: PLC0415 - only here

    with pytest.raises(KeyError):
        _reader({"kind": "colour"}, None)
