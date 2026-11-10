# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare and inspect the optional Julia backend without starting its runtime.

Install ``solprop[julia]``, then call :func:`setup` explicitly. A run selects a
published installation through :func:`select_setup`; neither selection nor
:func:`status` imports JuliaCall or downloads packages.
"""

from solprop.backends.julia._identity import SetupRecord, SetupStatus
from solprop.backends.julia._setup import select_setup, setup, status

__all__ = ["SetupRecord", "SetupStatus", "select_setup", "setup", "status"]
