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
from solprop.service import handlers

from solprop_gui.model.plot_options import PlotOptions
from solprop_gui.views.live import OverviewView
from solprop_gui.views.plane import PlaneView
from test_ui import shot


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def ask(kind: str, body: Mapping[str, object]) -> Any:
    return handlers.HANDLERS[kind](body)


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
