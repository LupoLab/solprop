# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
A planar guide resolved in y
============================

The hybrid planar transform: the slab modes of a 127 um argon-filled gap (Akturk et
al. 2009) in x, and a periodic, collared y axis. Unlike the slab uniform in y of
:ref:`sphx_glr_auto_tutorials_waveguides_plot_08_planar.py`, the beam here has a
finite height, so the run integrates over y and reports energy in J, and the model can
describe
diffraction, whole-beam self-focusing and modulational instability in y,
subject to the retained modes, grid, boundary and response assumptions. A 1 mJ, 100 fs
pulse,
Gaussian in both directions (``w_x = 0.735a`` projected onto TE1, TE3 and TE5;
``w_y = 0.5 mm``), propagates 10 cm with the Kerr effect under the step controller;
the run takes seconds on ``numpy_ref``, the reference backend. ``numba_cpu`` and the
native backend execute the geometry too.
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

from solprop.boundaries import absorber
from solprop.driver import SimulationSettings, prop_planar
from solprop.grid import EnvelopeGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, PlanarPulse
from solprop.spatial import planar_fluence, planar_mode_energy
from solprop.stats import default_spec
from solprop.stepper import PIController

started = perf_counter()
half_gap, length = 63.5e-6, 0.1

# %%
# The grid carries the slab channels, their rule in x and the y axis with its
# absorbing collar; the launch projects its x profile onto the modes, with signs, and
# its energy is the incident beam's, so the modes carry the coupled share.
grid = EnvelopeGrid.from_limits(length, 800e-9, (650e-9, 1000e-9), 400e-15)
modes = [SlabMode(half_gap, "Ar", 1.0, order=p) for p in (1, 3, 5)]
space = PlanarGrid(modes, 1.5e-3, 64)
launch = PlanarPulse(
    GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
    0.5e-3,
    waist_x=0.735 * half_gap,
)
print(space)

# %%
# The Kerr effect of argon, whose per-particle hyperpolarisability the run scales by
# the fill's density. The collar absorbs whatever reaches it. Its damping is
# integrated explicitly in every nonlinear stage, so a step longer than about 7 mm
# here is unstable (:doc:`/guide/free-space/model`): the step controller's error
# estimate keeps its steps below that, while a fixed schedule of ten 1 cm steps would
# be refused at translation.
settings = SimulationSettings(
    controller=PIController(rtol=1e-3, atol=1e-9),
    save_planes=[length],
    dz_max=length / 10,
    dz_init=1e-3,
    absorber=absorber(grid, boundary_length=1.0),
    stats=default_spec("free"),
    backend="numpy_ref",
)
output = prop_planar(
    grid,
    space,
    [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
    launch,
    settings=settings,
    progress="none",
)
first, last = output.fields("fft")
energy = np.ravel(output["stats/energy"])
print(f"energy {energy[0] * 1e3:.4f} mJ -> {energy[-1] * 1e3:.4f} mJ")
for mode, value in zip(modes, planar_mode_energy(grid, space, last), strict=True):
    print(f"  {mode.channel_label.display_label}: {value * 1e3:.5f} mJ")

# %%
# The fluence at the exit on the rule's (x, y) nodes, and across y after integrating
# over the gap. The widths in y are the run's ``beam_radius`` statistic.
fluence = planar_fluence(grid, space, last).reshape(space.n_y, space.x.size)
figure, (section, profile) = plt.subplots(1, 2, figsize=(9, 3.6))
section.pcolormesh(space.x * 1e6, space.y * 1e3, fluence * 1e-4, shading="nearest")
section.set(xlabel="x (um)", ylabel="y (mm)", title="exit fluence (J/cm²)")
for label, plane in (("entrance", first), ("exit", last)):
    across = planar_fluence(grid, space, plane).reshape(space.n_y, -1) @ space.x_weights
    profile.plot(space.y * 1e3, across, label=label)
profile.set(xlabel="y (mm)", ylabel="fluence across the gap (J/m)")
profile.legend()
figure.tight_layout()
radius = np.ravel(output["stats/beam_radius"])[-1]
print(f"1/e^2 half-width in y at the exit: {radius * 1e3:.3f} mm")

# %%
print(f"ran in {perf_counter() - started:.1f} s on numpy_ref")

# %%
# Related reading
# ---------------
# Continue with :doc:`/guide/waveguides/planar` for the model, options and limits,
# and :doc:`/guide/numerics/accuracy` for convergence checks.
# For a paced propagation lesson, return to
# :doc:`/introduction/lessons/plot_04_crossing_beams`.
