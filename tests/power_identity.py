# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Build the constant-fill version of the reduced M2 window."""

from solprop.driver import Simulation
from solprop.grid import FieldGrid
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule


def constant_fill_m2():
    """Keep M2's spectral/transverse physics over ten fixed steps in one millimetre."""
    # The committed scan case has a pressure gradient and cannot retain its trajectory
    # under the required real Phi term. This constant 1.2 bar replacement exercises
    # Kerr, plasma and noise.
    # Retain the original metre-long grid/collar; only the observed replay window
    # is shortened. Rescaling the collar to one millimetre changes its stability.
    grid = FieldGrid.from_limits(1.0, 800e-9, (100e-9, 1.5e-6), 200e-15)
    return Simulation.from_capillary(
        125e-6,
        1.0,
        "He",
        1.2,
        lambda0=800e-9,
        grid=grid,
        fwhm=7.5e-15,
        energy=275e-6,
        modes=2,
        n_save=2,
        noise="modified",
        seed=31,
        stats=StatsSpec(names=()),
        schedule=FixedSchedule.uniform(0.0, 1.0, 1e-4),
        backend="numpy_ref",
    )
