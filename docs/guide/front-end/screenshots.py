# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Take the screenshots of the front end's chapter of the user guide.

This script is **not** run by the docs build. ``poe guide-screenshots`` runs it by hand
whenever the window changes; the PNGs it writes into ``images/`` beside it are committed
and the pages only include them, so the manual builds on an install with no Qt. It needs
the front end (``uv sync --extra gui``) and the native backend, because it runs the
examples it shows.

Each picture is the real window, offscreen, at 1280 by 820, driven through the stories
the pages tell: a capillary from Home through a typed target, its run and its analysis,
a scan of its pressure, a waveguide designed without a pulse, and a radial filament.
Every number in them is the one solprop computes for the example; nothing is staged.
The Run stage is shown after its run ends, since a frame caught on the way depends on
the machine's speed. A screenshot is not compared with what is committed (fonts and
antialiasing differ between machines), so a stale one is caught by reading the page.
Home is also written, cut to its tiles, as the showcase in ``docs/_static`` that the
repository's README and the manual's front page show.
"""

import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRect, QSettings, QStandardPaths
from PySide6.QtWidgets import QApplication, QWidget
from solprop_gui.classes.workflow import Stage
from solprop_gui.model.workspace import DEFAULT_WORKSPACE
from solprop_gui.ui.workflow_window import WorkflowWindow
from solprop_gui.views.page_plots import LinePlot

IMAGES = Path(__file__).resolve().parent / "images"
"""Where the screenshots are written."""

SHOWCASE = Path(__file__).resolve().parents[2] / "_static" / "readme-gui-home.png"
"""Home, cut to its tiles, for the README and the manual's front page."""

SHOWCASE_HEIGHT = 640
"""How much of the window's height the showcase keeps, in pixels: down to the tiles'
buttons."""

SIZE = (1280, 820)
"""The window's size, in pixels."""

WAIT_S = 600.0
"""The longest any step may take, s."""


def _spin(app: QApplication, done: Callable[[], bool], timeout: float = WAIT_S) -> None:
    """Process events until a condition holds.

    Parameters
    ----------
    app : PySide6.QtWidgets.QApplication
        The application.
    done : callable
        The condition.
    timeout : float, optional
        How long to wait, s.

    Raises
    ------
    TimeoutError
        If the condition does not hold in time.
    """
    start = time.monotonic()
    while not done():
        if time.monotonic() - start > timeout:
            msg = "the window did not reach the state a screenshot needs"
            raise TimeoutError(msg)
        app.processEvents()
        time.sleep(0.005)


def _settle(app: QApplication, seconds: float = 1.0) -> None:
    """Process events for a while, so that the window finishes drawing.

    Parameters
    ----------
    app : PySide6.QtWidgets.QApplication
        The application.
    seconds : float, optional
        How long, s.
    """
    start = time.monotonic()
    while time.monotonic() - start < seconds:
        app.processEvents()
        time.sleep(0.005)


def _save(widget: QWidget, name: str) -> None:
    """Write a widget as it is drawn.

    Parameters
    ----------
    widget : PySide6.QtWidgets.QWidget
        The widget.
    name : str
        The file's name, without its extension.
    """
    widget.grab().save(str(IMAGES / f"{name}.png"))
    sys.stdout.write(f"wrote {name}.png\n")


def main() -> None:  # noqa: PLR0915 - one walk through the window
    """Walk the window through the guide's stories and save what it shows."""
    IMAGES.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp())
    QStandardPaths.setTestModeEnabled(True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(scratch / "settings")
    )
    app = QApplication(sys.argv[:1])
    window = WorkflowWindow(workspace=scratch / "runs")
    window.resize(*SIZE)
    window.show()
    window.document.confirm = lambda _title, _text: "discard"  # pyright: ignore[reportAttributeAccessIssue]
    sheet = window.sheet

    def ready() -> bool:
        """Return whether solprop accepts the setup shown.

        Returns
        -------
        bool
            Whether the Ready line says so.
        """
        return "Ready" in sheet.ready.text()

    def open_class(key: str) -> None:
        """Open a class's sheet and wait until it is ready.

        Parameters
        ----------
        key : str
            The class.
        """
        window.begin_class(key)
        _spin(app, lambda: window.problem is not None and window.problem.key == key)
        _spin(app, ready)
        _settle(app, 2.5)

    def run_to_analyse() -> None:
        """Run the setup and wait for its results."""
        window.primary.click()
        _spin(app, lambda: window.stage is Stage.ANALYSE)
        _spin(app, lambda: bool(window.analyse.buttons))
        _settle(app, 2.0)

    # Home, naming the folder a new user's runs go in rather than this script's own.
    window.home.show_workspace(DEFAULT_WORKSPACE)
    _spin(app, lambda: bool(sheet._examples))
    _settle(app, 1.0)
    _save(window, "home")
    window.grab(QRect(0, 0, SIZE[0], SHOWCASE_HEIGHT)).save(str(SHOWCASE))
    sys.stdout.write(f"wrote {SHOWCASE.name}\n")

    # The capillary's sheet, and a target typed into its soliton order.
    open_class("capillary")
    _save(window, "capillary-sheet")
    button = sheet.derived["soliton_order"]
    button.click()  # pyright: ignore[reportAttributeAccessIssue]
    _settle(app, 0.5)
    _save(window, "capillary-target")
    sheet.target_editor.hide()
    _settle(app, 0.3)

    # Its run and its analysis. The Run stage is taken once the run has ended: a frame
    # caught on the way depends on the machine's speed.
    run_to_analyse()
    _save(window, "capillary-overview")
    window.show_stage(Stage.RUN)
    _settle(app, 1.0)
    _save(window, "capillary-run")
    window.show_stage(Stage.ANALYSE)
    _settle(app, 0.5)
    analyse = window.analyse
    analyse.show_page("plane")
    _settle(app, 2.0)
    page = analyse._page()
    assert page is not None
    pulse = [plot for plot in page.findChildren(LinePlot) if plot.isVisible()][1]
    x, y = np.asarray(pulse.xs), np.asarray(pulse.ys[0])
    peak = int(np.nanargmax(y))
    pulse.measure_width(float(x[peak]), float(y[peak]))
    pulse.box.setXRange(float(x[peak]) - 6.0, float(x[peak]) + 6.0)
    pulse.zoomed = True
    _settle(app, 0.5)
    _save(window, "capillary-plane")
    menu = pulse.show_menu(QPointF(float(x[peak]), float(y[peak])), QPoint(0, 0))
    _settle(app, 0.3)
    _save(menu, "plot-menu")
    menu.close()
    analyse.show_page("along")
    _settle(app, 2.0)
    _save(window, "capillary-along")

    # A scan of its pressure.
    window.show_stage(Stage.SETUP)
    window.document.edit_table(
        "scan",
        {"name": "pressure", "pressure": ["0.6 bar", "0.8 bar", "1 bar", "1.2 bar"]},
    )
    window.show_stage(Stage.SCAN)
    record = window.scans.record
    record.band.setChecked(True)
    record.low.set_value("120 nm")
    record.high.set_value("250 nm")
    _settle(app, 0.5)
    window.primary.click()
    browser = window.scans.browser
    _spin(app, lambda: browser.result.count() > 0 and not window.run_panel.active)
    for index in range(browser.result.count()):
        if "band" in browser.result.itemText(index).lower():
            browser.result.setCurrentIndex(index)
            break
    _settle(app, 1.0)
    _save(window, "scan")
    window.document.edit_table("scan", None)

    # A waveguide designed with no pulse.
    window.show_stage(Stage.HOME)
    window.home.design_chosen.emit()
    _settle(app, 4.0)
    _save(window, "design")

    # A radial filament: its sheet and its analysis.
    open_class("filament")
    _save(window, "filament-sheet")
    run_to_analyse()
    _save(window, "filament-overview")

    model = window.document.model
    if model is not None:
        model._saved = model.revision
    window.close()
    app.quit()


if __name__ == "__main__":
    main()
