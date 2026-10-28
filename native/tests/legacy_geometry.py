# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The grid geometry from before solprop SEP 21, for tests not about boundaries.

SEP 21 moved the temporal collar outside the requested window and made
``fft_size="auto"`` the default. A test sized for the former geometry -- a pinned
transform length, a pulse placed against the window's edge -- states it here.
"""

from __future__ import annotations

from typing import Literal, TypedDict


class LegacyGrid(TypedDict):
    """The ``from_limits`` keywords of the former geometry."""

    fft_size: str
    collar_fraction: float
    collar_placement: Literal["inside"]


LEGACY_GRID: LegacyGrid = {
    "fft_size": "pow2",
    "collar_fraction": 0.05,
    "collar_placement": "inside",
}
"""``from_limits`` keywords that rebuild a grid as solprop built it before SEP 21."""
