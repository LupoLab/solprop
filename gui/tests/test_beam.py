# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A free-space plane measured: in the kernel, in a run's worker, and in the views."""

import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pytestqt.qtbot import QtBot
from solprop.exceptions import SolpropError
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.recipe import Recipe

from solprop_gui.model.plot_options import PlotOptions
from solprop_gui.service import handlers
from solprop_gui.service import run as worker
from solprop_gui.service.beam import across
from solprop_gui.service.outbox import Outbox
from solprop_gui.service.reduce import TIME_OVERSAMPLING, Display, Reducer
from solprop_gui.service.worker import RunSpec
from solprop_gui.views.live import OverviewView
from solprop_gui.views.plane import PlaneView
from test_results import beam, run  # noqa: F401 - fixtures
from test_service import Collector
from test_ui import shot


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def ask(kind: str, body: Mapping[str, object]) -> Any:
    return handlers.HANDLERS[kind](body)


def test_the_nodes_across_a_beam_s_centre():
    name, x, values = across(RadialGrid(1e-3, 8), np.arange(8.0))
    assert name == "r" and x.size == values.size == 8
    name, _, values = across(LineGrid(1e-3, 8), np.arange(8.0))
    assert name == "x" and values.tolist() == list(range(8))
    grid = FreeGrid(1e-3, 4, 2e-3, 6)
    nodes = np.arange(24.0).reshape(6, 4)  # y by x, as the statistics order them
    name, x, values = across(grid, nodes.reshape(-1))
    row = int(np.argmin(np.abs(np.asarray(grid.y))))
    assert name == "x" and x.size == 4 and values.tolist() == nodes[row].tolist()
    with pytest.raises(TypeError, match="no profile"):
        across(object(), np.zeros(2))


def test_a_beam_s_plane_its_profile_and_its_slice(beam: str, run: str):  # noqa: F811
    found = ask("plane", {"path": beam, "index": 1})
    profile = found["profile"]
    assert profile["axis"] == "x" and profile["x"].size == profile["y"].size == 32
    assert profile["y"].max() > 0
    # The spectrum is the grid's active band only, not every positive bin.
    assert found["spectrum"]["x"].max() < 1e-5  # not every bin out to 100 um
    rows = ask("overview", {"path": beam})["rows"]
    assert all("profile" in row for row in rows)
    sliced = ask("beam", {"path": beam, "index": 1, "bandpass": [7e-7, 9e-7]})
    assert sliced["values"].shape == (sliced["x"].size, sliced["t"].size)
    assert sliced["values"].dtype == np.float32 and sliced["axis"] == "x"
    # A beam's pulse is drawn as finely as a modal one: four samples per grid step.
    pulse_t = np.asarray(found["power"]["t"])
    assert pulse_t.size == np.asarray(found["power"]["y"]).size
    step = float(pulse_t[1] - pulse_t[0])
    assert float(sliced["t"][1] - sliced["t"][0]) == pytest.approx(step)
    assert pulse_t.size % TIME_OVERSAMPLING == 0
    with pytest.raises(SolpropError, match="only a free-space run"):
        ask("beam", {"path": run, "index": 0})


def test_a_free_space_run_streams_its_beam(tmp_path: Path):
    text = Recipe.example("cartesian_beam").to_text()
    band = (7e-7, 9e-7)
    spec = RunSpec(
        "b", str(tmp_path / "b"), recipe=text, display=Display(bandpass=band)
    )
    collector = Collector(spec).start()
    collector.collect()
    rows = [event.data for event in collector.events if event.kind == "row"]
    assert collector.kinds()[-1] == "finished" and len(rows) == 2
    assert rows[0]["profile"]["axis"] == "x" and rows[0]["profile"]["y"].max() > 0
    assert set(rows[0]["density"]) == {"wavelength", "f"}


def test_a_continued_free_space_run_sends_no_rows():
    class Plane:
        basis = "spatial"

    observer = worker.LiveObserver("r", Reducer(Display()), Outbox())
    observer.observe_plane(0.0, Plane())  # no grids kept: nothing measured
    assert observer.rows == []
    observer.measure(object())  # no transverse grid: measured by its modes
    assert observer.space is None


def row(index: int) -> dict[str, object]:
    x = np.linspace(-1e-3, 1e-3, 9)
    return {
        "index": index,
        "z": 0.01 * index,
        "density": {
            axis: np.ones(16, dtype=np.float32) for axis in ("wavelength", "f")
        },
        "power": np.ones(8, dtype=np.float32),
        "stats": {"energy": 1e-6},
        "profile": {"axis": "r", "x": x, "y": np.exp(-((x / 3e-4) ** 2)) * (index + 1)},
    }


AXES = {
    "wavelength": np.linspace(4e-7, 1.2e-6, 16),
    "f": np.linspace(2.5e14, 7.5e14, 16),
    "time": np.linspace(-1e-13, 1e-13, 8),
    "bandpass": None,
}


def test_the_overview_maps_a_beam_across_its_centre(qtbot: QtBot, tmp_path: Path):
    view = OverviewView()
    qtbot.addWidget(view)
    view.resize(1100, 700)
    assert not view.spatial and view.profile_map.plot not in view.maps.ci.items
    view.set_rows(AXES, [row(0), row(1), row(2)])
    assert view.spatial and view.profile_map.plot in view.maps.ci.items
    assert view.profile_map.data.shape == (3, 9)
    assert view.profile_map.plot.getAxis("bottom").labelText == "r (mm)"
    xs, ys = view.profile_curve.getData()
    assert xs is not None and ys is not None and float(ys.max()) == pytest.approx(3.0)
    shot(view, tmp_path, "overview-beam")
    modal_rows = [{k: v for k, v in row(i).items() if k != "profile"} for i in range(2)]
    view.set_rows(AXES, modal_rows)  # a modal file again: the map goes
    assert not view.spatial and view.profile_map.plot not in view.maps.ci.items


def test_the_plane_view_draws_a_beam_s_intensity_slice(qtbot: QtBot):
    view = PlaneView()
    qtbot.addWidget(view)
    view.resize(1000, 600)
    t = np.linspace(-5e-14, 5e-14, 20)
    x = np.linspace(-1e-3, 1e-3, 11)
    data = {"t": t, "x": x, "axis": "x", "values": np.ones((11, 20))}
    view.show_spectrogram(data)
    plot = view.spectrogram
    assert plot.plot.getAxis("left").labelText == "x (mm)"
    assert "across the beam" in plot.plot.titleLabel.text
    hit = plot.lookup(0.0, 0.5)  # mm
    assert hit is not None
    view.apply_options(PlotOptions(axis="f", spectral_limits=(1e14, 2e14)))
    assert plot.plot.getViewBox().viewRange()[1][1] == pytest.approx(1.0, abs=0.2)
