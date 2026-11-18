# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A free-space beam in the Overview view, offscreen."""

import warnings
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from solprop_gui.views.live import OverviewView
from test_ui import shot


@pytest.fixture(autouse=True)
def _quiet() -> Iterator[None]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


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


def test_the_overview_names_a_beam_s_readings(qtbot: QtBot):
    view = OverviewView()
    qtbot.addWidget(view)
    view.set_spatial(True)
    assert "Axial power" in view.power_map.plot.titleLabel.text
    assert "Energy per bin" in view.spectrum_map.plot.titleLabel.text
    view.set_spatial(False)
    assert view.power_line.titleLabel.text == "Power at z"
