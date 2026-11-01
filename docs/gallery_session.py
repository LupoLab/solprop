# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reset between gallery scripts, in a module so that it can be named in conf.py."""

import sys


def leave_session(gallery_conf: object, fname: str) -> None:
    """Undo an imported `solprop.sol`, so that a script importing it starts one.

    Autosummary imports `solprop.sol` before the galleries run, and importing it
    silences `MaterialRangeWarning` for the rest of the process, so a warning a page
    should show would go missing. Each script starts with no session, as a script run
    alone does.

    Parameters
    ----------
    gallery_conf : object
        The sphinx-gallery configuration, unused.
    fname : str
        The script about to run, unused.
    """
    del gallery_conf, fname
    session = sys.modules.pop("solprop.sol", None)
    if session is not None:
        session.end_session()
