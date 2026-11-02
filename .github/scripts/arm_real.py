# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: a user's run on Windows on Arm, with no thread settings of their own."""
import faulthandler, sys
faulthandler.enable()
faulthandler.dump_traceback_later(240, exit=True)
from solprop.boundaries import SpatialAbsorber
from solprop.driver import SimulationSettings, prop_radial
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController
import numpy as np
sizes = set()
_solve = np.linalg.solve
def noted(a, b, **kw):
    sizes.add(np.shape(a)[-1])
    print("solve size", np.shape(a)[-1], flush=True)
    return _solve(a, b, **kw)
np.linalg.solve = noted
n_r = int(sys.argv[1])
backend = sys.argv[2]
grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (500e-9, 1.6e-6), 100e-15, thg=False)
space = RadialGrid(4e-4, n_r, collar_fraction=1.0)
pulse = SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 8e-5)
settings = SimulationSettings(
    controller=PIController(rtol=1e-6, atol=1e-12),
    save_planes=[1e-4],
    dz_max=2.5e-5,
    dz_init=2.5e-5,
    backend=backend,
    spatial_absorber=SpatialAbsorber(),
)
print("start", n_r, backend, flush=True)
prop_radial(grid, space, SpatialMedium("Ar", 1.0), [], pulse, settings=settings, progress="none")
print("finished", n_r, backend, "solve sizes", sorted(sizes), flush=True)
