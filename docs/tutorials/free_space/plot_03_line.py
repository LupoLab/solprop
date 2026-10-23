# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
A line beam: diffraction and the (1+1)-D Kerr soliton
=====================================================

A :class:`~solprop.grid.LineGrid` propagates a beam that is uniform in y, per metre of
y: free space resolved in x alone. Its energies are in J/m and its powers in W/m.
With one free transverse dimension Kerr self-focusing does not collapse; the stationary
beam is the soliton ``sech(x/x0)``, whose power per metre of y is
``8 eps0 c / (3 k0**2 chi3 x0)``. Here a long pulse carries that power at its peak
through three diffraction lengths in vacuum with an absolute Kerr susceptibility, and
at half the power the same beam diffracts. Ten fixed steps each; the run takes seconds.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from solprop.constants import EPS0, C
from solprop.driver import SimulationSettings, prop_free
from solprop.exceptions import GridError
from solprop.grid import EnvelopeGrid, LineGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.spatial import (
    SpatialMedium,
    critical_power,
    spatial_energy,
    spatial_intensity,
)
from solprop.stepper import FixedSchedule, PIController

# %%
# The soliton's power per metre of y, and the line. A critical power has no meaning
# here, and asking for one on a line is refused by name.
wavelength, chi3, x0 = 800e-9, 3e-26, 50e-6
k0 = 2 * math.pi / wavelength
soliton = 8 * EPS0 * C / (3 * k0**2 * chi3 * x0)
length = 3 * k0 * x0**2
space = LineGrid(20 * x0, 128, window_factor=0)
print(f"soliton power {soliton:.3g} W/m; three diffraction lengths {length:.3g} m")
try:
    critical_power(wavelength, chi3, space=space)
except GridError as error:
    print(f"refused: {str(error)[:72]}...")

# %%
# A long pulse whose peak carries ``P'`` (or half of it) in a ``sech(x/x0)`` beam: the
# transverse profile is normalised so that its power per metre of y is the temporal
# pulse's power.
grid = EnvelopeGrid.from_limits(length, wavelength, (770e-9, 832e-9), 3e-12, thg=False)
profile = 1 / np.cosh(space.x / x0)
spectrum = np.fft.fft(profile)
norm = 0.5 * EPS0 * C * np.sum(space.spectral_weights * np.abs(spectrum) ** 2)
settings = SimulationSettings(
    controller=PIController(rtol=1e-8, atol=1e-14),
    save_planes=[length],
    dz_init=length / 10,
    dz_max=length,
    schedule=FixedSchedule.uniform(0.0, length, length / 10),
    backend="numpy_ref",
)
figure, axis = plt.subplots(figsize=(6, 4))
for fraction, colour in ((1.0, "C0"), (0.5, "C1")):
    temporal = GaussianPulse(
        wavelength=wavelength, fwhm=500e-15, peak_power=fraction * soliton
    ).build(grid)
    start = spectrum[:, None, None] / math.sqrt(norm) * temporal
    result = prop_free(
        grid,
        space,
        SpatialMedium(),
        [KerrResponse(chi3, thg=False)],
        start,
        settings=settings,
    )
    first, last = result.fields("fft")
    before = spatial_intensity(grid, space, first)
    after = spatial_intensity(grid, space, last)
    peak = float(np.max(before))
    column = int(np.argmax(np.max(after, axis=0)))
    axis.plot(
        space.x * 1e6,
        after[:, column] / peak,
        colour,
        label=f"{fraction:g} P' after 3 k x0²",
    )
    print(
        f"{fraction:g} P': energy {spatial_energy(grid, space, last):.4g} J/m, "
        f"peak intensity ratio {np.max(after) / peak:.4f}"
    )
axis.plot(space.x * 1e6, profile**2, "k--", label="sech²(x/x0), launched")
axis.set(
    xlabel="x (µm)", ylabel="intensity at the pulse peak (norm.)", xlim=(-250, 250)
)
axis.legend()
figure.tight_layout()

# %%
# Related reading
# ---------------
# Continue with :doc:`/guide/free-space/model` for the model, options and limits,
# and :doc:`/guide/numerics/accuracy` for convergence checks.
# For a paced propagation lesson, return to
# :doc:`/introduction/lessons/plot_04_crossing_beams`.
