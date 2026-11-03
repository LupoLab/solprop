# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The editable recipe: the authored document, its revisions, undo and dirty state.

This layer imports no Qt. In the window's process it may use only the light parts of
``solprop.recipe`` (its quantity parser and writer, parameter paths and the TOML
document); everything that builds or checks a simulation runs in the kernel process.
"""

__all__: list[str] = []
