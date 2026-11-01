# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
Is my transverse window big enough?
===================================

A free-space grid ends at a boundary: a radial grid has a hard wall at its outer
radius, which reflects, and a Cartesian or line grid is periodic, so light that leaves
one side comes back at the other. By default solprop puts no absorber there. It folds a
**safety ramp** into the operator, which removes light only at the grid's largest
transverse wavenumbers, and its **boundary monitor** reads the outer 5 % of the grid,
the transverse **edge**. A run whose light reaches the edge warns, and stops at ten
times the budget, naming a size to try. This tutorial sizes a radial window with the
monitor, checks the result against a much wider window, and then asks for a designed
real-space collar instead. It is the spatial companion of
:doc:`/auto_tutorials/basics/plot_07_time_window`.

The beam: a 30 fs, 800 nm Gaussian pulse focused to a 35 um waist in argon at one bar,
with the Kerr response at 0.9 of the critical power, over 1 cm. That is a few
Rayleigh lengths, so the beam spreads well beyond its waist, and the self-focusing
throws a little light out at larger angles. The input shot noise is off
(``noise=None``), so that grids of different sizes see the same launch.

The free-space model's section on transverse boundaries
(:doc:`/guide/free-space/model`) describes the ramp, the monitor and the collar.
"""

# %%
# A window that is too small
# --------------------------
# One function builds the beam on a requested radius, three waists by default, with
# ``fraction`` of that radius added outside it as room and ``absorber`` the transverse
# boundary. The radial spacing stays the same on every grid.

import math
import time
import warnings

import numpy as np
from scipy.special import j0

from solprop.boundaries import SpatialAbsorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import FieldGrid, RadialGrid
from solprop.monitor import DEFAULT_BOUNDARY_CHECK
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium, critical_power
from solprop.stepper import PIController

LENGTH = 0.01
WAIST = 35.36e-6
RADIUS = 3 * WAIST
"""The requested radius: three waists."""
SAMPLES = 64
"""Radial samples over the requested radius."""


def beam(fraction=0.0, absorber=None, check=True):
    """Return the beam on `RADIUS` with `fraction` of it added outside as room."""
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (500e-9, 1.6e-6), 250e-15)
    space = RadialGrid(
        RADIUS, round(SAMPLES * (1 + fraction)), collar_fraction=fraction
    )
    medium = SpatialMedium("Ar", 1.0)
    chi3 = gamma3("Ar", wavelength=800e-9)
    density = float(medium.density(np.array([0.0]))[0])
    index = float(np.real(medium.index(np.array([2 * math.pi * C / 800e-9]))[0]))
    power = 0.9 * critical_power(800e-9, chi3 * density, index=index)
    energy = power * 30e-15 * math.sqrt(math.pi / (4 * math.log(2)))
    settings = SimulationSettings(
        controller=PIController(rtol=1e-7, atol=0.0),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH * 1e-4,
    )
    return Simulation.from_radial(
        grid,
        space,
        medium,
        [KerrResponse(chi3, thg=False)],
        SpatialPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy), WAIST
        ),
        settings=settings,
        noise=None,
        spatial_absorber=absorber,
        boundary_check=DEFAULT_BOUNDARY_CHECK if check else None,
    )


# %%
# On three waists the beam reaches the wall well before the end, and the monitor
# stops the run. The message names a radius to try, from how far the light walked
# in the distance it ran, and the samples that keep the spacing:

try:
    with warnings.catch_warnings():
        # It warns first, as the edge passes the warning; the stop says it all.
        warnings.simplefilter("ignore", BoundaryWarning)
        beam().run(progress="none")
except BoundaryError as error:
    print(f"stopped: {error}")

# %%
# Reading what the monitor saw
# ----------------------------
# Room outside the requested radius moves the wall out. ``collar_fraction`` adds it
# as a fraction of the radius, with no absorber unless one is asked for, so
# ``fraction=1`` puts the wall at twice the radius. Each run's record holds the
# edge's ``transverse_status`` and the largest share of the launch energy it read
# there above its launch, ``edge_max``.

RADII = np.linspace(0.0, RADIUS, 481)
"""Where the fields are compared: the requested radius."""


def run(simulation):
    """Run; return the field at the end on `RADII`, the edge's record and wall time."""
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        output = simulation.run(progress="none")
    wall = time.perf_counter() - started
    monitor = output.outcome.execution.get("boundary_monitor") or {}
    record = monitor.get("transverse", {})
    # The Fourier-Bessel series of the output's columns, exact on any radii, so that
    # grids of different sizes are compared on the same points.
    hankel = simulation.spatial_grid.hankel
    coefficient = 2.0 / (hankel.radius**2 * hankel.j1**2)
    spectrum = np.asarray(output["field"][-1])[:, 0, :]
    field = j0(np.outer(RADII, hankel.k_r)) @ (coefficient[:, None] * spectrum)
    return field, record, wall


for fraction in (1.0, 2.0):
    _, record, _ = run(beam(fraction))
    print(
        f"wall at {1 + fraction:g} R: {record['transverse_status']}, "
        f"largest edge share {max(record['edge_max']):.1e} over {record['reads']} reads"
    )

# %%
# At twice the radius the edge reads a few times 1e-5 of the energy, above the
# warning of 1e-6, so the run warns; at three times it is clear.
#
# Checking against a wider window
# -------------------------------
# The monitor reads light *at* the edge, not light that came back from it. A run on a
# wall far further out says what each smaller window cost: the field error on the
# requested radius, against a wall at eight times the radius.

reference, _, reference_wall = run(beam(7.0, check=False))


def error(field):
    """Return the area-weighted field error on the requested radius."""
    weight = 2 * np.pi * RADII
    difference = np.sum(weight[:, None] * np.abs(field - reference) ** 2)
    return math.sqrt(difference / np.sum(weight[:, None] * np.abs(reference) ** 2))


print("  wall    status    edge share   field error   wall time (s)")
for fraction in (0.0, 1.0, 2.0):
    # Three waists would be stopped: it runs to the end unwatched here.
    field, record, wall = run(beam(fraction, check=fraction > 0))
    status = record.get("transverse_status", "off")
    share = max(record["edge_max"]) if record.get("edge_max") else float("nan")
    print(
        f"{1 + fraction:5g} R  {status:9s}  {share:10.1e}   {error(field):10.1e}"
        f"   {wall:8.2f}"
    )
print(f"    8 R  reference                              {reference_wall:8.2f}")

# %%
# With the monitor off, three waists end about a tenth wrong: the stop was right. Twice
# the radius ends a few times 1e-5 wrong in field, as its warning said, and three times
# is clean, at about 1e-7. The monitor errs on the cautious side, since light at the
# edge has not yet come back; the radius the stop named, 3.7 radii, is a lower bound for
# the light seen so far.
#
# A collar on request
# -------------------
# Instead of a wider wall, a real-space collar in the room outside the radius can
# absorb what reaches it. ``SpatialAbsorber()`` designs one from the launch's band of
# transverse wavenumbers and the room's width, and checks it at translation by
# propagating the launch through it; a design that cannot hold its budget is refused
# by name. Here it needs 2.6 radii of room:

field, record, wall = run(beam(2.6, SpatialAbsorber()))
print(
    f"collar in 2.6 R of room: {record['transverse_status']}, field error "
    f"{error(field):.1e}, removed {record['spatial_removed']:.1e} of the energy, "
    f"{wall:.2f} s"
)

# %%
# The collar holds the error near 1e-6 and reads clear: its monitor reads the light it
# sends back, rather than the edge. For this beam it is not the better choice. A bare
# wall at three radii is more accurate and several times faster, because the collar's
# rate limits the step and adds work to each one. A collar earns its place where room
# is what you cannot afford: fast light at an angle, such as four-wave-mixing products
# or conical emission, which would need a far wider window to keep
# (:doc:`plot_06_four_wave_mixing_scan`).
#
# What the monitor cannot judge
# -----------------------------
# The stop names a size from the light the run has seen. Light generated later, by a
# filament forming near the end, may need more. A weak signal at an angle, such as a
# four-wave-mixing sideband, can be far below the warning and still be what you
# measure: declare it, with ``SpatialAbsorber(kappa=...)`` or ``angle=...``, so that
# a collar is designed for it, or give it room and check it against a wider run, as
# above.
#
# **In short.** A run that finishes ``clear`` had no light at its edge above a
# millionth of its energy. When it warns or stops, give it the radius the message
# names, or room and a ``SpatialAbsorber``, and check what you measure against one
# wider run. ``spatial_absorber=False`` removes the safety ramp, and
# ``boundary_check=None`` the monitor.
