# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
A gas-filled hollow rectangular waveguide
=========================================

A hollow silica channel 250 um wide and 100 um high, filled with 5 bar of argon, guides
a 30 fs, 800 nm pulse in its lowest x-polarised mode. The mode is supplied as an object
(:class:`~solprop.modes.rectangular.RectMode`) to the waveguide route, which binds it
to the route's fill, and the run is a plain
:func:`~solprop.driver.interface.prop_waveguide`: the waveguide route's starting point,
and the call its example recipe mirrors (``Recipe.example("hollow_rectangular")``).

The first chart is the mode's loss and dispersion across the band; the second the
spectrum at the entrance and after 20 cm, broadened by self-phase modulation.
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from solprop.constants import C
from solprop.driver import prop_waveguide
from solprop.modes import RectMode
from solprop.modes.base import beta2, db_per_m
from solprop.plotting import spec_1d
from solprop.stats import StatsSpec

started = perf_counter()

# %%
# The mode
# --------
# The half-widths are 125 um in x and 50 um in y; the field is polarised along x, the
# wide side, which has the lower loss: 3 dB/m at 800 nm. The loss grows as the cube of
# the inverse height, so half the height is eight times the loss. The cladding is fused
# silica by default. Built here with the fill, the mode's loss and dispersion can be
# drawn before any run; the route rebinds it to its own fill.

mode = RectMode(125e-6, 50e-6, "Ar", 5.0)
wavelength = np.linspace(450e-9, 1200e-9, 300)
omega = 2 * np.pi * C / wavelength
figure, (loss_axis, dispersion_axis) = plt.subplots(1, 2, figsize=(9, 3.5))
loss_axis.plot(wavelength * 1e9, db_per_m(mode, omega))
loss_axis.set(xlabel="Wavelength (nm)", ylabel="Loss (dB/m)")
dispersion_axis.plot(wavelength * 1e9, beta2(mode, omega) * 1e30)
dispersion_axis.axhline(0.0, color="0.6", linewidth=0.8)
dispersion_axis.set(xlabel="Wavelength (nm)", ylabel="β₂ (fs²/m)")
figure.tight_layout()

# %%
# The run
# -------
# One mode, so the launch needs no table; the energy is a literal. The envelope grid
# leaves out the third harmonic, which this short guide does not phase-match, and argon
# has no Raman response. The transverse quadrature of the nonlinear overlap uses 12 by
# 12 nodes.

run = prop_waveguide(
    [RectMode(125e-6, 50e-6)],
    0.2,
    "Ar",
    5.0,
    wavelength=800e-9,
    wavelength_limits=(450e-9, 1200e-9),
    trange=200e-15,
    fwhm=30e-15,
    energy=20e-6,
    envelope=True,
    plasma=False,
    n_r=12,
    n_theta=12,
    n_save=11,
    stats=StatsSpec(names=("energy",)),
    seed=1,
)
spec_1d(run, [0.0, 0.2], spec_range=(600e-9, 1050e-9))
print(f"Transmitted energy: {run['stats/energy'][-1].sum() * 1e6:.4f} uJ")
print(f"Complete hollow rectangular example: {perf_counter() - started:.3f} s")

# %%
# Related reading
# ---------------
# :doc:`/guide/waveguides/rectangular` gives the model and its limits, and
# :doc:`/guide/setup/recipes` the recipe that builds this run.
