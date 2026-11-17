# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The radial collar's cases: SEP 40's real-space collar on a radial beam.

A collar requested with a :class:`~solprop.boundaries.SpatialAbsorber` on a grid with
room outside its aperture, its rate the Planck profile ``alpha0 (1 - W)`` on the nodes,
and the independent Fourier-Bessel operator it should be. ``test_radial_absorber.py``
holds the collar to that operator; the case catalogue's ramped-rate rows
(``backends/free_rows.py``) build on the same cases.
"""

from functools import partial

import numpy as np
from scipy.special import j0

from legacy_geometry import LEGACY_GRID
from solprop.boundaries import Absorber, SpatialAbsorber
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble
from solprop.translate.routes import transverse_of

LENGTH = 1e-4
COLLAR = SpatialAbsorber(kappa=8e4)
"""A declared band: kappa w = 24 on the case's 300 um collar, above the floor 22.3."""


def radial_absorber_case(family):
    """Return a family's radial collar case and its independent Bessel operator."""
    constructor = (
        partial(EnvelopeGrid.from_limits, thg=False)
        if family == "envelope"
        else FieldGrid.from_limits
    )
    grid = constructor(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15, **LEGACY_GRID)
    # Room as wide as the aperture, and enough columns that the declared band lies
    # below the safety ramp.
    space = RadialGrid(300e-6, 24, collar_fraction=1.0)
    reference_length = 5 * LENGTH
    boundary = Absorber(
        np.zeros(grid.n_omega),
        np.zeros(grid.n_t),
        np.zeros(grid.n_to),
        grid.active_mask,
        reference_length,
    )
    index = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    start = np.zeros((space.n_r, 2, grid.n_omega), dtype=np.complex128)
    start[:, 0, index] = np.exp(-(((np.arange(space.n_r) - 4) / 3) ** 2))
    start[:, 1, index] = 0.7j * start[:, 0, index]
    responses = (
        [KerrResponse(chi3=0.0, thg=False)] if family == "field_analytic" else []
    )
    design = transverse_of(
        assembled(grid, space, boundary, start, responses, 8).problem.route
    )
    assert design is not None and design.has_collar
    # Direct Fourier-Bessel forward and inverse formulas, independent of the
    # backend's composed matrix and with no temporal pad/crop factors.
    kernel = j0(space.r[:, None] * space.k_r[None, :])
    forward = kernel.T * space.radial_weights[None, :] / (2 * np.pi)
    inverse = kernel * space.spectral_weights[None, :] / (2 * np.pi)
    rate = design.alpha0 * (1 - space.window)
    absorption = forward @ (-0.5 * rate[:, None] * inverse)
    wave = grid.omega[index] / C
    # The safety ramp rides the diagonal operator as -alpha_s / 2 on each column.
    phase = 1j * space.k_r**2 / (
        np.sqrt(wave**2 - space.k_r**2) + wave
    ) - 0.5 * np.asarray(design.ramp_rate)
    return grid, space, boundary, start, index, absorption, phase, responses


def assembled(grid, space, boundary, start, responses, steps, **options):
    """The case's problem and request on a schedule of `steps`, with the collar."""
    options.setdefault("spatial_absorber", COLLAR)
    return assemble(
        grid,
        None,
        responses,
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        absorber=boundary,
        controller=PIController(rtol=1e-5, atol=1e-12),
        save_planes=[LENGTH],
        dz_init=LENGTH / steps,
        dz_max=LENGTH,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps),
        **options,
    )


def problem_for(case, steps, **options):
    grid, space, boundary, start, _, _, _, responses = case
    return assembled(grid, space, boundary, start, responses, steps, **options)
