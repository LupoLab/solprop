# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The free-space routes' shared builders: a short spatial problem and an analytic rate.

Every free-space ladder (``backends/free_rows.py``), the planar plates, the plasma
column's spatial cases and the low-storage march's spatial rows assemble their
problems with :func:`make_problem` over :data:`LENGTH`, and ionise under
:class:`QuadraticRate`, which reads no material fit.
"""

import numpy as np

from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble

LENGTH = 1e-4
"""The spatial ladders' length, m."""

CHI3 = 1e-22
"""The spatial ladders' Kerr susceptibility, m^2/V^2."""


def make_problem(grid, space, start, responses, *, steps=8, density=1.0):
    """Assemble a spatial problem over :data:`LENGTH` on `steps` uniform fixed steps."""
    return assemble(
        grid,
        None,
        responses,
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
        density=density,
    )


class QuadraticRate:
    """An analytic ionisation law, independent of material fit data."""

    threshold = 1e3
    maximum = 1e17
    citation = "Analytic test rate W=3e12*(E/1e9)^2 s^-1"
    parameters = (
        ("model", "quadratic-radial-oracle"),
        ("reference", 1e9),
        ("scale", 3e12),
    )

    def __call__(self, field_amplitude):
        values = np.abs(np.asarray(field_amplitude))
        return np.where(
            values < self.threshold,
            0.0,
            np.minimum(self.maximum, 3e12 * (values / 1e9) ** 2),
        )
