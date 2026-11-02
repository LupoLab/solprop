# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
Four-wave mixing into the vacuum ultraviolet
============================================

Two pulses of different colours travelling together through a gas-filled capillary
mix through the third-order susceptibility. In degenerate four-wave mixing two pump
photons are converted into a *seed* photon and an *idler* photon,
:math:`2\omega_p = \omega_s + \omega_i`. With a deep-ultraviolet pump at 267 nm (the
third harmonic of a titanium-sapphire laser) and its 800 nm fundamental as the seed,
the idler is at

.. math:: \lambda_i = \left(\frac{2}{\lambda_p} - \frac{1}{\lambda_s}\right)^{-1}
          \approx 160\ \text{nm},

in the vacuum ultraviolet. The process is efficient only if it is **phase matched**:
the waveguide's anomalous contribution to the dispersion must cancel the gas's normal
contribution, so that
:math:`2\beta(\omega_p) = \beta(\omega_s) + \beta(\omega_i)`; this is
achieved by choosing the gas pressure for the capillary's core size. Because the
capillary keeps the beams overlapped and the interaction length long, a large fraction
of the pump energy can be converted to the vacuum ultraviolet in a single pass -- up
to the photon-number limit :math:`\omega_i / 2\omega_p \approx 0.83` of it, since
every idler photon costs two pump photons.

This example launches **two pulses at once** with ``pulses=[...]``: a 35 fs, 95 uJ pump
at 267 nm and a 35 fs, 45 uJ seed at 800 nm, into 1.5 m of 75 um-radius capillary filled
with helium at 0.445 bar. The grid's reference wavelength is the pump's.

**What to look for.** In the spectrum, the idler appears near 160 nm and grows along
the fibre while the pump and seed exchange energy with it; cross-phase modulation
broadens the seed. In time, the pump and the seed walk off from each other slowly,
which limits how long they can mix. At the exit about 44 % of the pump energy is in
the idler. Phase matching is sharp. This example's original set-up used 0.48 bar;
the pressure moved to 0.445 bar because optmat's helium index (SEP 3) puts the linear
phase matching of this capillary at 0.457 bar, and the pump's Kerr phase moves the
exit optimum to 0.445 bar. At 0.48 bar the idler would now carry under a tenth of the
pump. The experiment found its optimum at 46.7 kPa (0.467 bar), in a 0.5 m fibre of the
same bore.

**What is approximate.** Only the fundamental mode is carried. The helium index is
optmat's fit, qualified over the whole band (from 60 nm).

This is panel (c) of the survey of nonlinear processes in the review by Travers et al.;
the experiment it is based on is reported by Forbes et al., Optics Letters 49 (2024).

Use :doc:`phase-matching tools </guide/workflows/design-tools>` to locate
a starting pressure, then refine the nonlinear calculation around it.
The :doc:`frequency-dependent Kerr chapter </guide/media/dispersive-kerr>`
explains the carrier-coefficient approximation used here.
"""

# %%
# The run
# -------

import time
import warnings

import matplotlib.pyplot as plt
import numpy as np

from solprop.driver import prop_capillary
from solprop.exceptions import MaterialRangeWarning
from solprop.plotting import prop_2d, spec_1d, time_1d
from solprop.processing import bandpass, energy, grid_of
from solprop.pulses import GaussianPulse

pump = GaussianPulse(wavelength=267e-9, fwhm=35e-15, energy=95e-6)
seed = GaussianPulse(wavelength=800e-9, fwhm=35e-15, energy=45e-6)
idler_wavelength = 1 / (2 / pump.wavelength - 1 / seed.wavelength)
print(f"idler at {idler_wavelength * 1e9:.1f} nm")

started = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", MaterialRangeWarning)
    run = prop_capillary(
        75e-6,
        1.5,
        "He",
        0.445,
        wavelength=267e-9,
        pulses=[pump, seed],
        trange=400e-15,
        wavelength_limits=(120e-9, 3e-6),
        n_save=301,
        progress="none",
        seed=20261027,
    )

# %%
# Spectral and temporal evolution
# -------------------------------

figures = prop_2d(
    run,
    "wavelength",
    spec_range=(130e-9, 1.2e-6),
    time_range=(-60e-15, 60e-15),
    db_min=-40,
)
figures[-1].set_size_inches(9.5, 3.6)
figures[-1].tight_layout()

# %%
# The idler
# ---------
# The energy in a band around the idler along the fibre, and the input and output
# spectra.

idler = bandpass(grid_of(run), (150e-9, 170e-9))
z = np.asarray(run["z"])
idler_energy = np.asarray(
    [float(np.sum(energy(run, plane=index, bandpass=idler))) for index in range(z.size)]
)

fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 3.6))
left.plot(z, idler_energy * 1e6)
left.set_xlabel("z (m)")
left.set_ylabel("150-170 nm energy (uJ)")
spec_1d(run, z=[0.0, 1.5], ax=right, spec_range=(130e-9, 1e-6))
fig.tight_layout()

launched = float(np.sum(energy(run, plane=0)))
print(
    f"idler at the exit {idler_energy[-1] * 1e6:.2f} uJ, "
    f"{idler_energy[-1] / (pump.energy or 1.0):.2%} of the pump energy; "
    f"launched {launched * 1e6:.0f} uJ in all"
)

# %%
# The two input pulses in time, as launched. Their coherent power beats at the
# difference frequency. The width of one beat is not the 35 fs duration of either
# input pulse, so the automatic single-peak width label is disabled.

fig, ax = plt.subplots(figsize=(6.0, 3.2))
time_1d(run, z=0.0, time_range=(-60e-15, 60e-15), widths=False, ax=ax)
fig.tight_layout()

# %%
# Runtime and reproducibility
# ----------------------------

print(f"{time.perf_counter() - started:.0f} s; backend {run.metadata.backend.name}")
