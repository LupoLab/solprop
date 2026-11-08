# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ask the user what ``ssh`` asks: a password, a passphrase or a one-time code.

``ssh`` runs the program named by ``SSH_ASKPASS`` with its question and reads the
answer from its output. The window names a small script (:func:`askpass_program`) that
runs this module, which shows the question in a dialog of its own. Nothing is stored:
the answer goes to ``ssh`` and nowhere else, and with a master connection per machine
it is asked once per connection.
"""

import sys
from pathlib import Path

__all__ = ["askpass_program", "main"]


def askpass_program() -> str | None:
    """Return the program ``ssh`` should ask, written beside the window's settings.

    Returns
    -------
    str or None
        The program's path; ``None`` where it cannot be written, so ``ssh`` asks in
        its own way (a terminal, or not at all).
    """
    folder = Path.home() / ".solprop-gui"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":  # pragma: no cover - Windows only
            path = folder / "askpass.cmd"
            path.write_text(
                f'@"{sys.executable}" -m solprop_gui.askpass %*\r\n', encoding="utf-8"
            )
        else:
            path = folder / "askpass.sh"
            path.write_text(
                f"#!/bin/sh\nexec '{sys.executable}' -m solprop_gui.askpass \"$@\"\n",
                encoding="utf-8",
            )
            path.chmod(0o700)
    except OSError:
        return None
    return str(path)


def main(argv: list[str] | None = None) -> int:
    """Show ``ssh``'s question and print the answer.

    Parameters
    ----------
    argv : list of str or None, optional
        The question's words; ``None`` reads the command line.

    Returns
    -------
    int
        0 with an answer, 1 when the user cancels.
    """
    from PySide6.QtWidgets import QApplication, QInputDialog, QLineEdit  # noqa: PLC0415

    question = " ".join(sys.argv[1:] if argv is None else argv).strip() or "Password:"
    application = QApplication.instance() or QApplication([])
    del application
    secret = any(word in question.lower() for word in ("password", "passphrase", "pin"))
    mode = QLineEdit.EchoMode.Password if secret else QLineEdit.EchoMode.Normal
    answer, given = QInputDialog.getText(None, "solprop: SSH", question, mode)
    if not given:
        return 1
    sys.stdout.write(answer + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - run by ssh
    raise SystemExit(main())
