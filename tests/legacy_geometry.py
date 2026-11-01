# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The grid and boundary settings from before SEP 21, for tests not about boundaries.

SEP 21 moved the temporal collar outside the requested window, made ``fft_size="auto"``
the default and gave GNLSE and spatial runs the default collar. A test whose subject is
something else, and whose fixture was sized for the former geometry, states that
geometry here rather than chasing the new one.
"""

from __future__ import annotations

from solprop.boundaries import Absorber, no_absorber
from solprop.grid import TimeGrid

LEGACY_GRID = {
    "fft_size": "pow2",
    "collar_fraction": 0.05,
    "collar_placement": "inside",
}
"""``from_limits`` keywords that rebuild a grid as solprop built it before SEP 21."""


def bare_absorber(grid: TimeGrid) -> Absorber:
    """Return what a spatial or GNLSE run without an absorber had before SEP 21.

    No temporal or spectral collar, and the real-space collar off; the hard band mask,
    which the translation applies in any case, is kept.

    Parameters
    ----------
    grid : TimeGrid
        The run's grid.

    Returns
    -------
    Absorber
        The bare boundaries.
    """
    return no_absorber(grid)
