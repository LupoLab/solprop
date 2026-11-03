# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Plane and Statistics views, alone and with a real run, offscreen."""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QBuffer, QIODevice
from PySide6.QtGui import QPixmap
from pytestqt.qtbot import QtBot

from solprop_gui.service.messages import Reply
from solprop_gui.ui.main_window import MainWindow
from solprop_gui.views.plane import PlaneView
from solprop_gui.views.statistics import StatisticsView
from test_results import run  # noqa: F401 - a fixture
from test_ui import WAIT_MS, private_settings, shot  # noqa: F401 - a fixture

OMEGA = 2 * np.pi * 299_792_458.0 / np.linspace(1.2e-6, 5e-7, 50)


def png() -> bytes:
    pixmap = QPixmap(40, 30)
    pixmap.fill()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    pixmap.save(buffer, "PNG")
    return bytes(buffer.data().data())


def plane_reply(*, phase: bool = True) -> dict[str, object]:
    wavelength = 2 * np.pi * 299_792_458.0 / OMEGA
    density = np.exp(-(((wavelength - 8e-7) / 5e-8) ** 2))
    t = np.linspace(-1e-13, 1e-13, 64)
    return {
        "z": 0.05,
        "labels": ["HE1,1", "HE1,2"],
        "spectrum": {"x": wavelength, "y": density},
        "power": {"t": t, "y": 1e9 * np.exp(-((t / 1e-14) ** 2))},
        "phase": {"omega": OMEGA, "y": np.linspace(0, 3, OMEGA.size)}
        if phase
        else {"error": "a sum has no phase"},
    }


def test_the_plane_view_asks_and_draws(qtbot: QtBot, tmp_path: Path):
    view = PlaneView()
    qtbot.addWidget(view)
    view.resize(1000, 600)
    assert not view.slider.isEnabled()
    asked = []
    view.plane_wanted.connect(lambda *args: asked.append(("plane", *args)))
    view.spectrogram_wanted.connect(lambda *args: asked.append(("sg", *args)))
    view.compression_wanted.connect(lambda *args: asked.append(("comp", *args)))
    view.figure_wanted.connect(lambda figure: asked.append(("figure", figure)))
    view.set_planes([0, 2], [0.0, 0.05], ["HE1,1", "HE1,2"])
    assert asked[-1] == ("plane", 2, "sum") and "5 cm" in view.where.text()
    view.slider.setValue(0)
    assert asked[-1] == ("plane", 0, "sum")
    view.modes.setCurrentIndex(2)
    assert asked[-1] == ("plane", 0, 1)
    view.modes.setCurrentIndex(0)
    view.show_plane(plane_reply())
    assert "Peak power 975.1 MW" in view.status.text()  # the 64-sample peak
    assert "first mode" in view.status.text()  # a sum of two modes
    ys = np.asarray(view.phase_curve.getData()[1], dtype=float)
    assert np.isnan(ys).any() and np.isfinite(ys).any()  # dark edges left out
    view.log.setChecked(False)
    view.log.setChecked(True)
    view.ask_spectrogram()
    assert asked[-1] == ("sg", 0.0, "sum")
    view.show_spectrogram(png())
    view.spectrogram.resize(300, 200)
    assert view.spectrogram.pixmap() is not None
    view.ask_compression()
    assert asked[-1] == ("comp", 0, 2, "sum")
    t = np.linspace(-1e-13, 1e-13, 64)
    view.show_compression(
        {
            "t": t,
            "after": np.ones(64),
            "limit": np.ones(64),
            "fwhm_before": 1e-14,
            "fwhm_after": 9e-15,
            "fwhm_limit": 8e-15,
            "peak_power_after": 1e9,
            "peak_power_limit": 1.1e9,
            "coefficients": [0.0, 0.0, 1e-30],
        }
    )
    assert "φ2 = 1 fs^2" in view.status.text() and len(view.compressed) == 2
    shot(view, tmp_path, "plane")
    view.export_data(tmp_path / "plane.csv")
    lines = (tmp_path / "plane.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("spectrum: wavelength (nm)") and len(lines) == 1 + 64
    view.show_plane(plane_reply(phase=False))
    assert "No phase" in view.status.text() and view.compressed == []
    view.export_figure("x.pdf", "spec_1d")
    assert asked[-1][1]["plot"] == "spec_1d"
    view.show_error("refused")
    assert view.status.text() == "refused"
    view.clear()
    assert view.planes == [] and "No plane" in view.where.text()
    view.ask_spectrogram()  # still at the last z: asks again
    count = len(asked)
    view.ask_compression()  # no planes: nothing asked
    view._ask()
    assert len(asked) == count


def test_a_spatial_plane_shows_its_fluence(qtbot: QtBot, tmp_path: Path):
    view = PlaneView()
    qtbot.addWidget(view)
    view.set_planes([0, 1], [0.0, 0.02])
    x, y = np.linspace(-1e-3, 1e-3, 4), np.linspace(-2e-3, 2e-3, 5)
    reply = plane_reply()
    spatial = {
        **reply,
        "spectrum": {**reply["spectrum"], "unit": "J per bin"},  # pyright: ignore[reportGeneralTypeIssues]
        "fluence": {
            "axes": {"y": y, "x": x},
            "values": np.ones((5, 4)),
            "unit": "J/m^2",
        },
    }
    view.show_plane(spatial)
    assert (
        view.image.isVisible()
        and "Energy per bin" in view.spectrum_plot.getAxis("left").labelText
    )
    shot(view, tmp_path, "fluence")
    radial = {"axes": {"r": x}, "values": np.ones(4), "unit": "J/m^2"}
    view.show_plane({**spatial, "fluence": radial})
    assert not view.image.isVisible()
    view.show_plane({**spatial, "fluence": {"error": "not rebuilt"}})
    assert "No fluence: not rebuilt" in view.status.text()
    # A planar plane: no spectrum, solprop says why; the phase still drawn.
    view.show_plane({**reply, "spectrum": {"error": "planar planes are not exchanged"}})
    assert "No spectrum: planar" in view.status.text()
    view.set_refused({"compression": "no compression on a beam"})
    assert not view.compress_button.isEnabled()
    assert view.compress_button.toolTip() == "no compression on a beam"
    view.set_refused({})
    assert view.compress_button.isEnabled()


def test_the_statistics_view_draws_every_statistic(qtbot: QtBot, tmp_path: Path):
    view = StatisticsView()
    qtbot.addWidget(view)
    view.resize(900, 600)
    reply = {
        "z": [0.0, 0.01, 0.02],
        "stats": [
            {
                "name": "energy",
                "summary": "Energy.",
                "units": "J",
                "columns": [
                    {"label": "HE1,1", "y": [1e-6, 1e-6, 0.9e-6]},
                    {"label": "HE1,2", "y": [0.0, 1e-8, 2e-8]},
                ],
            },
            {
                "name": "density",
                "summary": "Density.",
                "units": "m^-3",
                "columns": [{"label": "density", "y": [1, 2, 3]}],
            },
        ],
        "steps": {
            "z": [0.0, 0.01, 0.02],
            "h": [1e-4, 2e-4, 2e-4],
            "rejected_z": [0.015],
            "rejected_h": [5e-4],
        },
    }
    view.set_statistics(reply)
    assert len(view.curves) == 2 and "1 rejected" in view.summary.text()
    hit = view.lookup(1.0, 1e-6)
    assert hit is not None and "z = 1 cm" in hit[2]
    view.names.setCurrentRow(1)
    assert "m^-3" in view.plot.getAxis("left").labelText
    shot(view, tmp_path, "statistics")
    seen = []
    view.figure_wanted.connect(seen.append)
    view.export_figure("s.pdf")
    assert seen == [{"file": "s.pdf", "plot": "plot_stats"}]
    view.clear()
    assert view.curves == [] and view.lookup(0.0, 0.0) is None
    assert not view.figure_button.isEnabled()


def test_the_results_open_a_run_s_planes_and_statistics(
    qtbot: QtBot,
    tmp_path: Path,
    run: str,  # noqa: F811 - the fixture
):
    window = MainWindow(workspace=tmp_path / "workspace")
    qtbot.addWidget(window)
    window.show()
    results = window.results
    results.open(run)
    qtbot.waitUntil(
        lambda: "Peak power" in results.plane.status.text(), timeout=WAIT_MS
    )
    assert results.plane.modes.count() == 3
    qtbot.waitUntil(lambda: results.statistics.names.count() == 3, timeout=WAIT_MS)
    window.show_view("plane")
    results.plane.ask_spectrogram()
    qtbot.waitUntil(
        lambda: results.plane.spectrogram.original is not None, timeout=WAIT_MS
    )
    results.plane.ask_compression()
    qtbot.waitUntil(lambda: "FWHM" in results.plane.status.text(), timeout=WAIT_MS)
    figure = tmp_path / "power.pdf"
    results.plane.export_figure(str(figure), "time_1d")
    qtbot.waitUntil(lambda: "Wrote" in results.info.text(), timeout=WAIT_MS)
    assert figure.stat().st_size > 1000
    window.show_view("statistics")
    shot(window, tmp_path, "statistics-window")
    # A failed plane says so in the view; a reply about another file is dropped.
    results._pending[-5] = ("plane", run)
    results.on_reply(Reply("plane", -5, 0, error="it failed"))
    assert "it failed" in results.plane.status.text()
    results._pending[-6] = ("plane", "other.h5")
    results.on_reply(Reply("plane", -6, 0, value={"z": 0}))
    results._pending[-7] = ("statistics", run)
    results.on_reply(Reply("statistics", -7, 0, value="not a mapping"))
    window.close()


def test_the_overview_names_a_beam_s_readings(qtbot: QtBot):
    from solprop_gui.views.live import OverviewView  # noqa: PLC0415 - only here

    view = OverviewView()
    qtbot.addWidget(view)
    view.set_spatial(True)
    assert "Axial power" in view.power_map.plot.titleLabel.text
    assert "Energy per bin" in view.spectrum_map.plot.titleLabel.text
    view.set_spatial(False)
    assert view.power_line.titleLabel.text == "Power at z"
