# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: an ordinary scan on Pool, with no thread settings of the user's own."""
import faulthandler, os, sys, tempfile
from solprop.boundaries import SpatialAbsorber
from solprop.driver import SimulationSettings, Simulation
from solprop.driver.executors import Pool
from solprop.driver.scan import Scan, ScanPoint, run_scan
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController


def build_point(point):
    n_r = int(sys.argv[1])
    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (500e-9, 1.6e-6), 100e-15, thg=False)
    space = RadialGrid(4e-4, n_r, collar_fraction=1.25)
    pulse = SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=float(point.values["energy"])), 8e-5)
    return Simulation.from_radial(grid, space, SpatialMedium("Ar", 1.0), [], pulse, settings=SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[1e-4], dz_max=2.5e-5, dz_init=2.5e-5,
        backend=sys.argv[2], spatial_absorber=SpatialAbsorber(),
    ))


if __name__ == "__main__":
    faulthandler.enable()
    faulthandler.dump_traceback_later(int(sys.argv[4]) - 5, exit=True)
    print("start", sys.argv[1:], flush=True)
    with tempfile.TemporaryDirectory() as folder:
        run_scan(Scan("energy", energy=[1e-9, 2e-9]), build_point, output=os.path.join(folder, "s.h5"), executor=Pool(workers=2, threads=int(sys.argv[3])))
    print("finished", sys.argv[1:], flush=True)
