# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar families: an argon- or nitrogen-filled gap's launches and steps.

A 127 um gap (2a, Akturk et al. 2009) over ten centimetres, each family a response and
an absorber scheme with its grid, launch, rule, y axis and fixed schedule. The oracle's
own tests (``test_planar_hybrid.py``), the compiled backends' planar cases
(``planar_cases.py``) and the case catalogue's planar rows (``planar_rows.py``) build on
them.
"""

from legacy_geometry import LEGACY_GRID
from solprop.grid import EnvelopeGrid, FieldGrid

LENGTH = 0.1
"""The gap's length, m."""

HALF_GAP = 63.5e-6
"""Half the gap, m: Akturk et al.'s 2a = 127 um."""

ENVELOPE = EnvelopeGrid.from_limits(
    LENGTH, 800e-9, (650e-9, 1000e-9), 1.2e-12, **LEGACY_GRID
)
"""The envelope families' grid."""

FAMILIES = {
    # name: (grid, gas, energy J/m, fwhm s, plasma, raman, n_r, n_y, steps, scheme)
    "kerr": (ENVELOPE, "Ar", 0.5, 100e-15, False, False, None, 4, 10, "nonlinear"),
    "kerr-split": (
        ENVELOPE,
        "Ar",
        0.5,
        100e-15,
        False,
        False,
        None,
        4,
        10,
        "split_step",
    ),
    "raman": (ENVELOPE, "N2", 0.5, 100e-15, False, True, None, 4, 10, "nonlinear"),
    # Forty steps: optmat's argon rate (SEP 3, Phase 4a) is 2.5 to 9 times the former
    # PPT's over this launch's fields, and fixed steps must keep the plasma's loss
    # inside the explicit history's stability. Twenty sufficed for the Runge-Kutta
    # march; the default Adams method, whose interval at order five is forty times
    # smaller, diverged there (to 4e192, finite: a replay cannot reject) and is stable
    # from forty, eighty and a hundred and sixty agreeing to 1e-4.
    "field-plasma": (
        FieldGrid.from_limits(
            LENGTH, 800e-9, (400e-9, 2000e-9), 600e-15, **LEGACY_GRID
        ),
        "Ar",
        2.0,
        50e-15,
        True,
        False,
        None,
        4,
        80,
        "nonlinear",
    ),
    # The envelope plasma's cycle moments dominate its cost, so the rule and the y
    # axis are the smallest that still exercise every path.
    "envelope-plasma": (
        ENVELOPE,
        "Ar",
        0.5,
        50e-15,
        True,
        False,
        12,
        2,
        2,
        "nonlinear",
    ),
}
"""Each family's launch, rule, y axis and schedule."""

ORDERS = {
    # The envelope plasma's smaller rule resolves TE1's quartic products alone.
    "envelope-plasma": (1,),
}
"""The slab orders of a family that does not carry TE1, TE3 and TE5."""
