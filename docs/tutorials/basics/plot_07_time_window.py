# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
Is my time window big enough?
=============================

The time window is periodic: light that walks past one edge comes back at the other.
Every run therefore carries a soft absorbing **collar** just outside the window you
ask for (``trange``), and a **boundary monitor** that watches the outer 5 % of the
window at each edge, the **guards**, and the energy the collar removes. This tutorial
uses both to size a window, on a short fibre where a Raman soliton decelerates
towards the window's late edge.

The fibre: 40 cm of a solid-core fibre at 835 nm (``beta2 = -11.83 ps^2/km``, a little
third-order dispersion, ``gamma = 0.11 /W/m``), pumped by a 50 fs sech pulse of
soliton order 1.7 with Raman on. The pulse settles into a soliton, which shifts to
the red and slows down, and sheds dispersive radiation near 700 nm, which travels
slower still. The runs below switch the input shot noise off
(``noise=None``), so that windows of different sizes see the same input; with noise,
each grid draws its own realisation.

The :doc:`boundaries guide </guide/numerics/boundaries>` describes the collar,
the monitor and their settings.
"""

# %%
# A window that is too small
# --------------------------
# One function builds the fibre on a requested window ``trange``, optionally with the
# pulse launched at ``t0`` and with the monitor's thresholds changed.

import dataclasses
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from solprop.driver import Simulation
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.monitor import BoundaryCheck
from solprop.processing import get_power
from solprop.pulses import SechPulse

LENGTH = 0.4
BETAS = [0.0, 0.0, -11.83e-27, 8.1e-41]
WIDTH = 28e-15
"""sech half-width, s (50 fs full width at half maximum)."""
PEAK = 3 * abs(BETAS[2]) / (0.11 * WIDTH**2)
"""Peak power, W: three times the fundamental soliton's, a soliton order of 1.7."""


def fibre(trange, *, t0=0.0, check=None):
    """Return the fibre on a requested window `trange`, s."""
    simulation = Simulation.from_gnlse(
        0.11,
        LENGTH,
        BETAS,
        lambda0=835e-9,
        wavelength_limits=(600e-9, 1600e-9),
        trange=trange,
        raman=True,
        n_save=21,
        noise=None,
        pulses=[SechPulse(wavelength=835e-9, width=WIDTH, peak_power=PEAK, t0=t0)],
    )
    if check is not None:
        simulation.settings = dataclasses.replace(
            simulation.settings, boundary_check=check
        )
    return simulation


# %%
# On a 0.5 ps window the soliton reaches the late edge before the end of the fibre.
# The monitor warns once the field in a guard passes its budget (a field error of
# 1e-3, so a millionth of the energy), and stops the run at ten times the budget:

try:
    fibre(0.5e-12).run(progress="none")
except BoundaryError as error:
    print(f"stopped: {error}")

# %%
# Reading what the monitor saw
# ----------------------------
# A relaxed check lets the run finish so that its record can be read. The record sits
# in the run's execution record, with a row for every saved plane: the collar's
# occupancy there, each guard's largest occupancy since the previous plane, and the
# fraction of the launch energy the collar has removed so far. (The occupancies are
# counted above the shot-noise floor, which these runs do not carry.)

relaxed = BoundaryCheck(budget=1.0, stop_factor=1.0)


def watched(trange, **keywords):
    """Run the fibre with a relaxed check; return the run, its record and wall time."""
    simulation = fibre(trange, check=relaxed, **keywords)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        run = simulation.run(progress="none")
    wall = time.perf_counter() - started
    return run, run.outcome.execution["boundary_monitor"], wall, simulation.grid


small, small_record, _, _ = watched(0.5e-12)
modest, modest_record, _, _ = watched(1e-12)

fig, (guards, removed) = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
for record, label in ((small_record, "0.5 ps"), (modest_record, "1 ps")):
    rows = record["planes"]
    z = 100 * np.asarray(rows["z"])
    guards.semilogy(
        z, np.maximum(rows["guard_high"], 1e-16), label=f"{label}, late guard"
    )
    removed.semilogy(z, np.maximum(rows["removed_fraction"], 1e-16), label=label)
guards.axhline(1e-6, color="0.5", ls=":", label="budget (field error 1e-3)")
guards.set(xlabel="z (cm)", ylabel="guard occupancy", ylim=(1e-14, 1))
removed.set(xlabel="z (cm)", ylabel="fraction removed by the collar", ylim=(1e-8, 1))
guards.legend(fontsize=8)
removed.legend(fontsize=8)
plt.show()

# %%
# On 0.5 ps the soliton fills the late guard and the collar takes most of the
# energy. On 1 ps the guards stay far below the budget, yet the collar still removes
# about a quarter of a per cent of the energy: light that is not the soliton walks
# out. The monitor reports that at the end of the run as a warning, since its square
# root is above the budget; whether it matters depends on what you measure.

print(
    f"1 ps window: guards peaked at {modest_record['guard_high_max']:.1e}, "
    f"the collar removed {modest_record['removed_fraction']:.1e} of the launch energy"
)

# %%
# What is walking out
# -------------------
# A wide window answers that. On 8 ps nothing reaches the collar, and the exit power
# shows the soliton at about +0.34 ps and a weak tail far later: the dispersive
# radiation shed at fission, near 700 nm, which a 1 ps window cannot hold.

wide, wide_record, _, _ = watched(8e-12)
t, power = get_power(wide, plane=-1)
total = power.sum()
print(f"8 ps window: the collar removed {wide_record['removed_fraction']:.1e}")
for edge in (0.25e-12, 0.5e-12, 1e-12):
    print(
        f"energy later than {1e12 * edge:.2f} ps: {power[t > edge].sum() / total:.1e}, "
        f"earlier than {-1e12 * edge:.2f} ps: {power[t < -edge].sum() / total:.1e}"
    )

fig, ax = plt.subplots(figsize=(7, 3.4), layout="constrained")
ax.semilogy(1e12 * t, power / power.max(), lw=0.8, color="k")
for edge, colour in ((0.25e-12, "C3"), (0.5e-12, "C1"), (1.5e-12, "C2")):
    ax.axvspan(-edge, edge, color=colour, alpha=0.08)
    ax.axvline(edge, color=colour, lw=0.8, label=f"{2e12 * edge:g} ps request")
ax.set(xlabel="time (ps)", ylabel="power (normalised)", xlim=(-1, 3), ylim=(1e-12, 2))
ax.legend(fontsize=8)
plt.show()

# %%
# Cost against what is kept
# -------------------------
# Widening the window costs transform points and time. The fibre loses energy to the
# Raman response itself, so what the window costs is read against the wide run's
# exit energy. The soliton is the same on every window that holds it; the radiation
# needs 3 ps.


def kept(run):
    """Return the exit energy over the launch energy."""
    energy = np.asarray(run["stats/energy"]).reshape(len(run["z"]), -1).sum(axis=1)
    return energy[-1] / energy[0]


reference = kept(wide)
print(" window   points   wall (s)   removed   exit energy against 8 ps")
for trange in (0.5e-12, 1e-12, 2e-12, 3e-12, 4e-12):
    run, record, wall, grid = watched(trange)
    print(
        f"{1e12 * trange:5.1f} ps  {grid.n_t:6d}   {wall:8.2f}   "
        f"{record['removed_fraction']:.1e}   {kept(run) / reference - 1:+.1e}"
    )

# %%
# Placing the launch
# ------------------
# The soliton walks one way, to later times, so launching the pulse early moves it
# away from the late edge. On 0.75 ps, a launch 0.2 ps early brings the soliton's exit
# from near the edge back towards the middle. The guard and the collar barely notice:
# what they report is the radiation, which walks far further than the soliton and
# crosses the late guard whatever the launch. Placing the launch helps when the light
# you measure is the only light that walks; here only a wider window keeps it all.

for t0 in (0.0, -0.2e-12):
    run, record, _, _ = watched(0.75e-12, t0=t0)
    t, power = get_power(run, plane=-1)
    guard, removed = record["guard_high_max"], record["removed_fraction"]
    print(
        f"0.75 ps, launched at {1e12 * t0:+.2f} ps: soliton at "
        f"{1e12 * t[power.argmax()]:+.2f} ps, late guard {guard:.1e}, "
        f"collar removed {removed:.1e}"
    )

# %%
# **In short.** A monitor that stays quiet means no light reached the guards; a
# removed fraction near zero means none left through the collar either. When either
# is not, run the same case on a wider window, see what walks out, and decide whether
# it is part of what you measure. ``boundary_check`` relaxes or switches off the
# check, ``absorber=False`` the collar.
