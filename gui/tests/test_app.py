# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The command and the main window."""

import pytest

from solprop_gui import __version__
from solprop_gui.app import main


def test_version_answers_and_exits(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == __version__
