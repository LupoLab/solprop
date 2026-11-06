# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``solprop-gui`` command."""

import argparse
import sys
from collections.abc import Sequence

from solprop_gui import __version__

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Open the window and run until it closes.

    Parameters
    ----------
    argv : sequence of str or None, optional
        The command-line arguments after the program name; ``None`` reads
        ``sys.argv``.

    Returns
    -------
    int
        The exit status: 0 when the window closed normally.
    """
    parser = argparse.ArgumentParser(
        prog="solprop-gui", description="A desktop front end for solprop."
    )
    parser.add_argument("--version", action="version", version=__version__)
    # Accepted, and no longer needed: the workflow window is the only one.
    parser.add_argument("--workflow", action="store_true", help=argparse.SUPPRESS)
    options = parser.parse_args(argv)

    # Qt is imported here, after the arguments: `--version` and `--help` answer without
    # loading it.
    from pyqtgraph.util.garbage_collector import (  # noqa: PLC0415 - see above
        GarbageCollector,
    )
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415 - see above

    from solprop_gui.ui.identity import (  # noqa: PLC0415 - see above
        apply_identity,
        set_windows_app_id,
    )
    from solprop_gui.ui.workflow_window import (  # noqa: PLC0415 - see above
        WorkflowWindow,
    )

    # Before the first window, so that Windows shows this application's taskbar button.
    set_windows_app_id()
    application = QApplication.instance() or QApplication(sys.argv[:1])
    apply_identity(application)  # pyright: ignore[reportArgumentType]
    del options
    # Garbage is collected on a timer, between events, never by Python in the middle of
    # one: a plot dropped while its scene still had an event queued was otherwise freed
    # inside that event's handler, deleting the scene under Qt's feet.
    collector = GarbageCollector(interval=1.0)
    window = WorkflowWindow()
    window.show()
    try:
        return application.exec()
    finally:
        collector.timer.stop()
