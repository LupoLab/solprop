# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
The vector wall: two impedances, two families, one conversion
=============================================================

The scalar coherent wall gives the glass one impedance, the average a linearly
polarised ``HE_1m`` field sees around the bore. The wall imposes two conditions, not
one: at grazing incidence the field **tangential** to the glass sees
``d_r E = i k_o E`` and the field **normal** to it sees ``d_r E = i (k_o/n_r^2) E``.
``CoherentWall(impedance="te_tm")`` condenses both
(:doc:`/guide/waveguides/vector-wall`). Because the wall is rotationally symmetric the
total
azimuthal order ``nu`` is conserved, and an order has at most two radial families,
so the block is one order's two families: ``TE_0m`` and ``TM_0m`` for ``nu = 0``,
``HE_nu,m`` and ``EH_nu,m`` for ``nu >= 1``.

This example shows the two things a vector wall adds, each with the number the run
should reproduce:

* **each family keeps its own loss.** ``TE_01`` and ``TM_01`` of a 125 um bore at
  800 nm lose ``0.1161`` and ``0.2440`` 1/m -- a factor ``eps`` of the silica wall
  apart, which no impedance-averaged block can produce (it gives both ``0.1800``);
* **the anisotropy couples the two hybrid families.** Ten fixed steps of a 50 um bore,
  launched into ``HE_11``, leave ``2.8e-8`` of the arriving energy in ``EH_11``, where
  the uncoupled route leaves nothing measurable, and the transfer scales as the
  square of the half-difference ``delta`` of the two impedances.

**The approximation, in one sentence:** this is the scalar block's quasi-static,
paraxial condensation applied to the **grazing-impedance** model of the wall, checked
against the exact vector Maxwell roots family by family (to the size of this package's
own Marcatili error) and through the coupled eigenvalue; it has not yet been checked
against a resolved-cladding Maxwell solve excited in a mixed-family state.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import prop_capillary
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall, MarcatiliMode, vector_wall_operator
from solprop.stepper import FixedSchedule

# %%
# Each family's own loss, without a run
# -------------------------------------
#
# One channel of each family, alone, in an evacuated 125 um bore with a silica-like
# wall of index 1.45. A negative azimuthal index is this package's spelling of the
# ``EH`` family. The vector block's decay rate of each is set beside this package's
# own Marcatili rate for the same channel, and beside what an impedance-averaged
# block would give the two transverse families: the same channel with the hybrid
# cladding term ``v_HE = (v_TE + v_TM)/2`` in place of its own.
radius, wavelength = 125e-6, 800e-9
omega = np.array([2.0 * np.pi * C / wavelength])
families = {
    "TE_01": ("TE", 0),
    "TM_01": ("TM", 0),
    "HE_11": ("HE", 1),
    "EH_11": ("HE", -1),
}
cladding = 1.45 + 0j


def cladding_term(kind, order):
    """Return this package's Marcatili cladding term `v_n` for one family."""
    mode = MarcatiliMode(radius, None, 0.0, kind=kind, n=order, cladding=cladding)
    return complex(mode.sampled_terms(omega).cladding_term[0])


hybrid = cladding_term("HE", 1)
rows = {}
for name, (kind, order) in families.items():
    mode = MarcatiliMode(radius, None, 0.0, kind=kind, n=order, m=1, cladding=cladding)
    block = vector_wall_operator([mode], omega, validity_limit=2.0)
    vector = -float(np.real(block.generator[0, 0, 0]))
    marcatili = float(np.imag(mode.neff(omega)[0]) * omega[0] / C)
    averaged = marcatili * hybrid.real / cladding_term(kind, order).real
    rows[name] = (vector, marcatili, averaged)
    print(
        f"{name}: vector block {vector:.6e} 1/m, Marcatili {marcatili:.6e}, "
        f"impedance-averaged {averaged:.4e}"
    )
print(
    f"TM_01 / TE_01 = {rows['TM_01'][0] / rows['TE_01'][0]:.4f}, "
    f"eps = {cladding.real**2:.4f}"
)

fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
positions = np.arange(len(rows))
ax.bar(positions - 0.2, [row[0] for row in rows.values()], 0.4, label="vector block")
ax.bar(
    positions + 0.2, [row[2] for row in rows.values()], 0.4, label="impedance averaged"
)
ax.plot(
    positions, [row[1] for row in rows.values()], "k_", markersize=28, label="Marcatili"
)
ax.set_xticks(positions, list(rows))
ax.set(
    ylabel="Field-loss rate (1/m)", title="125 um bore, 800 nm: each family's own loss"
)
ax.legend(fontsize="small", frameon=False)
plt.show()

# %%
# The conversion, through the entry point
# ---------------------------------------
#
# The order ``nu = 1``, two channels of each family: ``HE_11``, ``HE_12``, ``EH_11``,
# ``EH_12``, contiguous from ``m = 1`` in each. A **supplied** mode set is the run's
# geometry, so ``prop_capillary``'s own radius argument must agree with the modes'
# radius or the call is refused by name; the two are the same here.
# Ten fixed steps of 20 um, launched into ``HE_11``, with the vector wall and with
# the ordinary incoherent loss. The scalar block cannot carry this set at all -- it
# spans two families -- and is refused by name.
length, steps = 2e-4, 10
bore = 50e-6
grid = FieldGrid.from_limits(length, 800e-9, (400e-9, 1.6e-6), 300e-15)


def run(wall, modes):
    """Propagate `modes` ten fixed steps of the 50 um bore, with or without a wall."""
    # A marginal geometry on purpose, and a band past the fitted ranges: both warn.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return prop_capillary(
            bore,
            length,
            "He",
            1.0,
            wavelength=800e-9,
            grid=grid,
            absorber=absorber(grid, boundary_length=1e3),
            fwhm=30e-15,
            energy=1e-9,
            modes=modes,
            n_save=11,
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


order_one = [
    MarcatiliMode(bore, "He", 1.0, n=order, m=radial + 1)
    for order in (1, -1)
    for radial in range(2)
]
labels = ["HE_11", "HE_12", "EH_11", "EH_12"]
walled = run(CoherentWall(impedance="te_tm"), order_one)
plain = run(None, order_one)
flow = walled.metadata.modal["flow"]
print(flow["model"], "|", walled.metadata.modal["basis"], "|", flow["gauge_id"])


def channel_energy(output, count):
    """Return the positions and the energy in each channel at every save plane."""
    planes = output.planes()
    energy = np.array(
        [np.asarray(plane.stats["energy"]).reshape(count, -1)[:, 0] for plane in planes]
    )
    return np.array([plane.z for plane in planes]), energy


_, energy = channel_energy(walled, 4)
print(
    f"vector wall: energy lost {1.0 - energy[-1].sum() / energy[0].sum():.6e}, "
    f"EH_11 holds {energy[-1, 2] / energy[-1].sum():.4e} of what arrives"
)

# %%
# The conversion is the anisotropy's. Moving the cladding index from 1.45 to 1.2099
# shrinks ``delta``, the half-difference of the two impedances, by a factor 1.54;
# a coupling through ``delta`` must shrink the ``EH_11`` share by that squared.
ratios = []
for index in (1.45 + 0j, 1.2099 + 0j):
    pair = [
        MarcatiliMode(bore, "He", 1.0, n=order, m=1, cladding=index)
        for order in (1, -1)
    ]
    block = vector_wall_operator(pair, grid.omega, active=grid.active_mask)
    impedance = block.impedance[:, grid.active_mask]
    delta = float(np.max(np.abs(0.5 * (impedance[0] - impedance[1]))))
    field = run(CoherentWall(impedance="te_tm"), pair).planes()[-1].field[:, 0]
    share = np.sum(np.abs(field) ** 2, axis=-1)
    ratios.append((delta, float(share[1] / share[0])))
(delta_a, share_a), (delta_b, share_b) = ratios
print(
    f"delta shrinks by {delta_a / delta_b:.4f}; "
    f"the EH_11 share by {share_a / share_b:.6f} "
    f"against delta squared, {(delta_a / delta_b) ** 2:.6f}"
)

fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
for output, style, name in ((plain, ":", "wall=None"), (walled, "-", "vector wall")):
    z, energy = channel_energy(output, 4)
    shares = energy / energy[0, 0]
    shares = np.where(shares > 0.0, shares, np.nan)
    for channel in range(4):
        ax.semilogy(
            z * 1e6,
            np.maximum(shares[:, channel], 1e-60),
            style,
            color=f"C{channel}",
            label=f"{labels[channel]}, {name}",
        )
ax.set(
    xlabel="z (um)",
    ylabel="Fraction of the launched energy",
    title="The wall's anisotropy converts HE_11 into EH_11",
    ylim=(1e-55, 3.0),
)
ax.legend(frameon=False, fontsize="small", ncol=2)
plt.show()

# %%
# Related reading
# ---------------
# Continue with :doc:`/guide/waveguides/vector-wall` for the model, options and limits,
# and :doc:`/guide/numerics/accuracy` for convergence checks.
# For a paced propagation lesson, return to
# :doc:`/introduction/lessons/plot_01_rdw`.
