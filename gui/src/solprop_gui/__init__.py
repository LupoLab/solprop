# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A desktop front end for solprop.

The window sets up, runs and reads solprop simulations without Python, while every
setup can leave it as a recipe file or a script and every results file solprop writes
can be opened in it. solprop does the physics and the window none: every number shown
comes from a public solprop call, made in a separate process. The design is
``docs/dev/gui/design.md`` in the solprop repository.

The package's layers are ``model``, ``registry``, ``classes`` and ``help``, which
import no Qt, and ``ui`` and ``views``, which build the window. The service the window
reaches its kernel and its runs through is ``solprop.service``, in solprop.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__: str = version("solprop-gui")
except PackageNotFoundError:  # pragma: no cover - a tree never installed
    __version__ = "0.0.0"
