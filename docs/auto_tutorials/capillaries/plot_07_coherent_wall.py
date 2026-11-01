# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
The coherent wall of a capillary, against the incoherent one
=============================================================

Marcatili's loss is **incoherent**: every ``HE_1m`` channel of a capillary attenuates
on its own and the wall never moves power between them. The **coherent** wall,
``prop_capillary(..., wall=CoherentWall())``, condenses the outgoing Robin boundary
condition of the exact radial problem onto the retained channels instead, so the
wall becomes one rank-one term that couples every channel to every other with a
definite phase (:doc:`/guide/waveguides/coherent-wall`).

This example shows the two things that follow, each with the number the run should
reproduce:

* **the loss.** Each channel of the block still carries its own Marcatili loss, to a
  departure that can be written down before anything is run -- ``-1.2036e-5`` of it
  for one channel of a 125 um bore at 800 nm;
* **the conversion.** Launched into ``HE_11`` alone, ten fixed steps of a 50 um bore
  leave ``2.6e-7`` and ``6.3e-7`` of the launched energy in ``HE_12`` and ``HE_13``
  with the wall, where the incoherent route -- which has no linear wall conversion --
  leaves
  only what the Kerr overlap puts there, eleven orders below.

**The approximation, in one sentence:** the block is the quasi-static, paraxial
condensation of the wall, so each channel's loss departs from Marcatili's by
``-(4 N^2 t^2 + u_m^2/(2 (ka)^2))`` relative, with ``t = v_n/(k_0 a)`` -- the first term
being the square of the quasi-static ratio ``2N/|a k_eff|`` the run records, which is
why raising the channel count ``N`` makes it larger rather than smaller.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import prop_capillary
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall, MarcatiliMode, wall_operator
from solprop.stepper import FixedSchedule

# %%
# The loss, without a run
# -----------------------
#
# :func:`~solprop.modes.capillary.wall_operator` builds the block itself, in 1/m and
# in the optics convention, so the field-loss rate of a channel is minus the real part
# of its diagonal entry. Against the evacuated bore's own Marcatili rate
# ``(omega/c) Im n_eff`` it departs by the prediction above, for one channel and for
# the lowest channel of a four-channel block, whose ``4 N^2`` is sixteen times larger.
radius = 125e-6
wavelengths = np.geomspace(0.4e-6, 3e-6, 9)
omega = 2.0 * np.pi * C / wavelengths
departures = {}
for count in (1, 4):
    modes = [MarcatiliMode(radius, None, 0.0, m=index + 1) for index in range(count)]
    block = wall_operator(modes, omega, validity_limit=1.0)
    measured = -np.real(block.generator[0, 0])
    marcatili = omega / C * np.imag(modes[0].neff(omega))
    t = 1.0 / np.abs(radius * block.impedance)
    ka = np.abs(block.wavenumber) * radius
    predicted = -(4.0 * count**2 * t**2 + modes[0].u_nm ** 2 / (2.0 * ka**2))
    departures[count] = (measured / marcatili - 1.0, predicted)

at_800 = MarcatiliMode(radius, None, 0.0, m=1)
single = wall_operator([at_800], np.array([2.0 * np.pi * C / 800e-9]))
rate = -float(np.real(single.generator[0, 0, 0]))
reference = float(2.0 * np.pi / 800e-9 * np.imag(at_800.neff(single.omega)[0]))
print(f"one channel, 125 um, 800 nm: relative departure {rate / reference - 1.0:.5e}")

fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
for count, colour in ((1, "C0"), (4, "C1")):
    measured, predicted = departures[count]
    ax.loglog(
        wavelengths * 1e6,
        -predicted,
        "-",
        color=colour,
        label=f"predicted, N = {count}",
    )
    ax.loglog(
        wavelengths * 1e6, -measured, "o", color=colour, label=f"block, N = {count}"
    )
ax.set(
    xlabel="Wavelength (um)",
    ylabel="Loss rate: -(block / Marcatili - 1)",
    title="HE_11 of a 125 um bore: the condensation's own loss error",
)
ax.legend(fontsize="small", frameon=False)
plt.show()

# %%
# The conversion, through the entry point
# ---------------------------------------
#
# Ten fixed steps of 20 um of a 50 um helium-filled bore, three channels, launched
# into ``HE_11`` alone: once with ``wall=CoherentWall()`` and once with the ordinary
# incoherent loss. The small radius is what makes a fifth of a millimetre show a
# measurable wall; it sits near the edge of the block's validity range on purpose,
# and the construction says so with a warning, as does the band's reach past the
# fitted ranges of helium and silica. The collar is weakened so that the twenty
# micrometre steps are stable: the absorber is an explicit term in the step, so its
# strength limits the step it allows.
length, steps = 2e-4, 10
grid = FieldGrid.from_limits(length, 800e-9, (400e-9, 1.6e-6), 300e-15)


def run(wall):
    """Propagate the three-channel bore ten fixed steps, with or without a wall."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prop_capillary(
            50e-6,
            length,
            "He",
            1.0,
            wavelength=800e-9,
            grid=grid,
            absorber=absorber(grid, boundary_length=1e3),
            fwhm=30e-15,
            energy=1e-9,
            modes=3,
            n_save=6,
            backend="numpy_ref",
            wall=wall,
            schedule=FixedSchedule(np.full(steps, length / steps)),
            dz_max=length / steps,
            dz_init=length / steps,
            plasma=False,
            raman=False,
            progress="none",
            noise=None,
        )


walled, plain = run(CoherentWall()), run(None)


def channel_energy(output):
    """Return the energy in each channel at every save plane, and the positions."""
    planes = output.planes()
    energy = np.array(
        [np.asarray(plane.stats["energy"]).reshape(3, -1)[:, 0] for plane in planes]
    )
    return np.array([plane.z for plane in planes]), energy


for name, output in (("CoherentWall()", walled), ("wall=None", plain)):
    _, energy = channel_energy(output)
    shares = energy[-1] / energy[0, 0]
    lost = 1.0 - energy[-1].sum() / energy[0].sum()
    print(
        f"{name:15s} energy lost {lost:.4e}; "
        f"HE_12 {shares[1]:.2e}, HE_13 {shares[2]:.2e}"
    )

# %%
# What the run recorded. ``modal.flow`` is part of the run's recorded identity, so a
# file written by this run says which block it propagated, and a resume against a
# different one is refused, naming the entry that differs.
flow = walled.metadata.modal["flow"]
print(flow["model"], flow["section_methods"], f"quasi_static={flow['quasi_static']}")

fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
for output, style, label in (
    (plain, ":", "wall=None"),
    (walled, "-", "CoherentWall()"),
):
    z, energy = channel_energy(output)
    # The launch plane holds exactly nothing in the upper channels, which a
    # logarithmic axis cannot show, so that point is left out of their curves.
    shares = energy / energy[0, 0]
    shares = np.where(shares > 0.0, shares, np.nan)
    for channel in range(3):
        ax.semilogy(
            z * 1e6,
            shares[:, channel],
            style,
            color=f"C{channel}",
            label=f"HE_1{channel + 1}, {label}",
        )
ax.set(
    xlabel="z (um)",
    ylabel="Fraction of the launched energy",
    title="Only the coherent wall moves power between channels",
    ylim=(1e-22, 3.0),
)
ax.legend(frameon=False, fontsize="small", ncol=2)
plt.show()

# %%
# Related reading
# ---------------
# Continue with :doc:`/guide/waveguides/coherent-wall` for the model, options and
# limits,
# and :doc:`/guide/numerics/accuracy` for convergence checks.
# For a paced propagation lesson, return to
# :doc:`/introduction/lessons/plot_01_rdw`.
