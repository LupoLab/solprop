# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
Package overview
================

Check the installation, see which backend a run will use, and propagate a pulse: a
two-centimetre run through a helium-filled capillary, read back through the same
accessors every later example uses. The tutorials that follow take each part in
turn -- pulses, analysing a run, the interactive namespace and scans; then gas-filled
capillaries, other waveguides, free space and the compression tools -- and the
:doc:`guide <../../guide/index>` says which model to use for what.
"""

# %%
# The package version, and the backend ``backend="auto"`` resolves to here: ``native``
# where the ``solprop-native`` package is installed, otherwise ``numba_cpu``.

import warnings

import matplotlib.pyplot as plt
import numpy as np

import solprop
from solprop.backends.registry import available, resolve_name
from solprop.driver import prop_capillary
from solprop.exceptions import MaterialRangeWarning
from solprop.processing import energy, fwhm_t, get_spectral_density

print(f"solprop {solprop.__version__}")
print(f"installed backends {available()}; auto resolves to {resolve_name('auto')}")

# %%
# One call builds the grid, the mode, the gas, the responses and the pulse, and
# runs them: two centimetres of a 125 um helium-filled capillary at 1 bar, pumped
# by a 10 fs, 20 uJ pulse at 800 nm. The gas index is asked for beyond its fitted
# range at the short end of the window, which it reports; that notice is silenced.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", MaterialRangeWarning)
    run = prop_capillary(
        125e-6,
        0.02,
        "He",
        1.0,
        wavelength=800e-9,
        wavelength_limits=(200e-9, 2e-6),
        trange=150e-15,
        fwhm=10e-15,
        energy=20e-6,
        n_save=5,
        progress="none",
        seed=20261027,
    )
print(run)

# %%
# The run is read through :mod:`solprop.processing`: the energy and duration at
# each save plane, and the spectrum at the entrance and the exit. At this energy
# and length the pulse barely changes, which is the point: every later example
# reads a run the same way.

print("energy (uJ):", np.round(np.asarray(energy(run)).ravel() * 1e6, 4))
print("duration (fs):", np.round(np.asarray(fwhm_t(run)).ravel() * 1e15, 3))

fig, ax = plt.subplots(figsize=(5.6, 3.2))
for plane, label in ((0, "entrance"), (-1, "exit")):
    wavelength, density = get_spectral_density(run, "wavelength", plane=plane)
    order = np.argsort(wavelength)
    ax.plot(wavelength[order] * 1e9, np.ravel(density)[order], label=label)
ax.set_xlim(600.0, 1000.0)
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("spectral energy density (J/m)")
ax.legend(frameon=False)
fig.tight_layout()

# %%
# Related reading
# ---------------
# Continue with :doc:`/guide/setup/running` for the model, options and limits,
# and :doc:`/guide/numerics/accuracy` for convergence checks.
# For a paced propagation lesson, return to
# :doc:`/introduction/lessons/plot_01_rdw`.
