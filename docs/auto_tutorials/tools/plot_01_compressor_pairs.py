# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""
Grating spacing and prism insertion
===================================

Recover known compressor settings from synthetic chirped pulses. No propagation is
needed: an inverse pair phase makes a test input and the bounded compressor finds the
physical stage setting that removes it. Independent ray and dispersion references,
rather than this round trip, establish the phase model in the test suite.

A fixed working band selects the same spectrum for every trial. The prism search
moves only the second prism, perpendicular to its base, at fixed separation stage.
"""

# %%
# Build a small envelope grid and a transform-limited 15 fs pulse.
import matplotlib.pyplot as plt
import numpy as np

from solprop import fourier
from solprop.grid import EnvelopeGrid
from solprop.optics import (
    GratingPair,
    PrismGeometry,
    PrismPair,
    brewster_angle,
    brewster_apex,
    littrow_angle,
    stored_factor,
)
from solprop.physdata import fused_silica
from solprop.processing import compress_gratings, compress_prisms
from solprop.pulses import GaussianPulse

wavelength = 800e-9
grid = EnvelopeGrid.from_limits(1e-3, wavelength, (400e-9, 2e-6), 2e-12)
clean = GaussianPulse(wavelength=wavelength, fwhm=15e-15, energy=1e-6).build(grid)
band = (600e-9, 1100e-9)

gratings = GratingPair(
    600e3, 400e-6, littrow_angle(600e3, wavelength), wavelength, band=band
)
glass = fused_silica(model="malitson_1965/sellmeier")
prisms = PrismPair(
    glass,
    PrismGeometry(
        separation=0.6,
        insertion=0.015,
        first_face_distance=0.003,
        apex_angle=brewster_apex(glass, wavelength),
        incidence_angle=brewster_angle(glass, wavelength),
    ),
    wavelength,
    band=band,
)

# %%
# Stored spectra use the internal Fourier convention. The conjugate of
# ``stored_factor`` applies the inverse phase and the same fixed band selection.
grating_input = clean * stored_factor(grid, gratings).conj()
prism_input = clean * stored_factor(grid, prisms).conj()
grating_fit = compress_gratings(
    grid=grid, spectrum=grating_input, pair=gratings, separation_range=(0.0, 1e-3)
)
prism_fit = compress_prisms(
    grid=grid, spectrum=prism_input, pair=prisms, insertion_range=(0.008, 0.021)
)
assert grating_fit.success and prism_fit.success
assert min(grating_fit.quality, prism_fit.quality) >= 1 - 1e-6

# %%
# Compare the selected pulses before and after compression. Both fitted settings
# recover the transform-limited pulse. Absolute CEP and transit time were removed.
figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for axis, stored, fitted, title in zip(
    axes,
    (grating_input, prism_input),
    (grating_fit, prism_fit),
    ("Gratings: tune normal spacing", "Prisms: tune second-prism insertion"),
    strict=True,
):
    for label, transfer in (("Before", None), ("Compressed", fitted.transfer)):
        factor = 1.0 if transfer is None else stored_factor(grid, transfer)
        field = fourier.inverse(stored * factor, kind="envelope")
        power = np.sum(abs(field) ** 2, axis=(0, 1))
        axis.plot(grid.t * 1e15, power * 1e-6, label=label)
    axis.set(xlim=(-200, 200), xlabel="Time (fs)", title=title)
    axis.legend(frameon=False)
axes[0].set_ylabel("Power (MW)")
figure.tight_layout()

# %%
# Related reading
# ---------------
# Continue with :doc:`/guide/workflows/compression` for the model, options and limits,
# and :doc:`/guide/numerics/accuracy` for convergence checks.
# For a paced propagation lesson, return to
# :doc:`/introduction/lessons/plot_02_soliton`.
