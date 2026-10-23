# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
Scan helium pressure and pulse energy
=====================================

A reduced version of a resonant-dispersive-wave (RDW) set-up shows how to vary
pressure and
energy together and collect
a compact result from each run. Point indices make the scan reproducible, while
shared setup and per-worker reductions control its execution cost. The 125 um
capillary and
the 7.5 fs, 800 nm pulse are those of the helium RDW case of
:ref:`sphx_glr_introduction_lessons_plot_01_rdw.py`, but this scan uses
only 2 mm of propagation, one mode and a 200 nm lower wavelength limit to keep its
execution short. It is not a converged RDW prediction. The padded grid extends beyond
the fitted helium and silica ranges, so MaterialRangeWarning notices are expected.
Helium has no Raman response; plasma and third-harmonic generation are explicitly
disabled here.
"""

from functools import cache
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

from solprop.driver import Simulation
from solprop.driver.executors import Local
from solprop.driver.scan import Scan, ScanPoint, run_scan
from solprop.physdata import Mixture
from solprop.processing.scans import energy_in_band, peak_power
from solprop.pulses import GaussianPulse


@cache
def base() -> Simulation:
    """Translate geometry and response tables once for this worker."""
    simulation = Simulation.from_capillary(
        125e-6,
        0.002,
        Mixture.fill({"He": 1.0}, 1.0),
        lambda0=800e-9,
        wavelength_limits=(200e-9, 1.5e-6),
        trange=200e-15,
        fwhm=7.5e-15,
        energy=200e-6,
        modes=1,
        raman=False,
        plasma=False,
        thg=False,
        n_save=3,
        backend="numpy_ref",
    )
    simulation.translate()
    return simulation


def build(point: ScanPoint) -> Simulation:
    """Update pressure and input energy while retaining the same geometry."""
    mixture = Mixture.fill({"He": 1.0}, float(cast(float, point.values["pressure"])))
    pulse = GaussianPulse(
        wavelength=800e-9,
        fwhm=7.5e-15,
        energy=float(cast(float, point.values["energy"])),
    )
    return base().with_mixture(mixture).with_input(pulse)


# %%
# Local is explicit so running this gallery never starts processes. In a standalone
# script, select Pool under the usual main guard to reuse one base per worker.
scan = Scan("helium_pressure_energy", pressure=[0.8, 1.2], energy=[200e-6, 275e-6])
with TemporaryDirectory(prefix="solprop-gallery-scan-") as directory:
    output = run_scan(
        scan,
        build,
        executor=Local(),
        output=Path(directory) / "scan.h5",
        reductions=[peak_power(), energy_in_band(250e-9, 400e-9)],
        fields="none",
        progress="none",
    )
    if not np.all(output.status == "done"):
        raise RuntimeError(f"gallery scan failed: {output.point(0)['error']}")
    powers = output.as_grid("reduced/peak_power")

figure, axis = plt.subplots()
for index, pressure in enumerate(scan.variables["pressure"]):
    axis.plot(
        scan.variables["energy"] * 1e6,
        powers[index] * 1e-9,
        "o-",
        label=f"{pressure:g} bar",
    )
axis.set(xlabel="Input energy (uJ)", ylabel="Final peak power (GW)")
axis.legend(title="Helium pressure")
figure.tight_layout()

# %%
# Related reading
# ---------------
# Continue with :doc:`/guide/workflows/scans` for the model, options and limits,
# and :doc:`/guide/numerics/accuracy` for convergence checks.
# For a paced propagation lesson, return to
# :doc:`/introduction/lessons/plot_01_rdw`.
