# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hybrid planar problems for the compiled backends' parity.

Each case is a hybrid family (``tests/backends/test_planar_hybrid.py``) launched
as a Gaussian in y rather than uniform in it, so every ``k_y`` column is populated
and the y transforms, the rows' blocks and the collar all act: the slab set's modal
amplitudes of the uniform launch times ``exp(-(y/w)^2)``, ``w`` a third of the y
radius, on an argon- or nitrogen-filled 127 um gap over at most ten fixed steps.
"""

import dataclasses
import math

import numpy as np

from backends.test_planar_hybrid import FAMILIES, HALF_GAP, LENGTH, ORDERS
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid
from solprop.modes import ModeCollection, PlanarGrid, gaussian_coupling
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.stepper.schedule import FixedSchedule

RADIUS_Y = 1e-3
"""Half-width of the periodic y window, m."""

CASES = {
    # name: (hybrid family, n_y, window_factor, collar boundary length m or None,
    # responses: the family's, none, or Kerr without its harmonic)
    "kerr": ("kerr", 6, 0.0, None, "family"),
    # A 10 m collar keeps its in-stage rate, 1.3 1/m, inside the default Adams
    # method's stability over one whole-length step (h alpha / 2 = 0.06 < 0.0877): the
    # cases test the collar's arithmetic, not its strength.
    "kerr-collar": ("kerr", 6, 0.3, 10.0, "family"),
    "kerr-split": ("kerr-split", 6, 0.3, 10.0, "family"),
    "raman": ("raman", 6, 0.0, None, "family"),
    "field-plasma": ("field-plasma", 4, 0.0, None, "family"),
    "field-kerr": ("field-plasma", 4, 0.0, None, "analytic"),
    "envelope-plasma": ("envelope-plasma", 3, 0.0, None, "family"),
    "collar-only": ("kerr", 6, 0.3, 10.0, "none"),
}
"""Each case's family, y axis, collar and responses.

``"analytic"`` puts a field grid on the analytic-signal column family (Kerr without
its harmonic), which no hybrid family compiles; ``"none"`` leaves the collar's
in-stage term as the whole derivative.
"""


def planar_case(name, *, steps=None, node_chunk=None):
    """Return ``(grid, space, problem)`` for one case, on a replayed schedule.

    Parameters
    ----------
    name : str
        A key of :data:`CASES`.
    steps : int, optional
        Fixed steps over :data:`LENGTH`; the family's own count by default. Zero is a
        one-step schedule for a problem that is only evaluated, never stepped.
    node_chunk : int, optional
        Override for the transform's node chunk, to force a partial last block.
    """
    family, n_y, window_factor, boundary, respond = CASES[name]
    grid, gas, energy, fwhm, plasma, raman, n_r, _, count, scheme = FAMILIES[family]
    steps = count if steps is None else max(1, steps)
    orders = ORDERS.get(family, (1, 3, 5))
    modes = ModeCollection.slab(HALF_GAP, orders, gas, 1.0).modes
    simulation = Simulation.from_waveguide(
        list(modes),
        LENGTH,
        gas,
        1.0,
        lambda0=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=boundary or 1.0e3),
        fwhm=fwhm,
        energy=energy,
        envelope=isinstance(grid, EnvelopeGrid),
        plasma=plasma,
        raman=raman,
        n_save=2,
        n_r=n_r,
        dz_max=LENGTH / steps,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule(np.full(steps, LENGTH / steps)),
        backend="numpy_ref",
    )
    shares = gaussian_coupling(modes, 0.735 * HALF_GAP)
    pulse = simulation._field0[0]
    modal = simulation.with_input(
        np.stack([math.sqrt(share) * pulse for share in shares])
    ).translate()
    space = PlanarGrid(modes, RADIUS_Y, n_y, window_factor=window_factor, n_x=n_r)
    profile = np.exp(-((space.y / (RADIUS_Y / 3)) ** 2))
    field0 = (
        np.fft.fft(profile)[:, None, None, None] * modal.initial.field0[0][None]
    ).reshape(space.n_columns, 1, grid.n_omega)
    settings = dataclasses.replace(
        simulation.settings,
        absorber_scheme=scheme,
        quadrature=None,
        components=None,
        stats=None,
    )
    if respond != "family":
        # The family's plasma range has no rate to tabulate over without it.
        settings = dataclasses.replace(settings, ionisation_field_range=None)
    responses = {
        "family": list(simulation.responses),
        "none": [],
        "analytic": [KerrResponse(gamma3(gas, wavelength=800e-9), thg=False)],
    }[respond]
    problem = Simulation.from_planar(
        grid, space, responses, field0, settings=settings
    ).translate()
    if node_chunk is not None:
        problem = dataclasses.replace(
            problem,
            transform=dataclasses.replace(problem.transform, node_chunk=node_chunk),
        )
    return grid, space, problem
