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
    from PySide6.QtWidgets import QApplication  # noqa: PLC0415 - see above

    from solprop_gui.ui.workflow_window import (  # noqa: PLC0415 - see above
        WorkflowWindow,
    )

    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("solprop")
    application.setApplicationVersion(__version__)
    del options
    window = WorkflowWindow()
    window.show()
    return application.exec()
