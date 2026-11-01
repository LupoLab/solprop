# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hybrid planar problems for the compiled backends' parity.

Each case is a hybrid family (``tests/backends/test_planar_hybrid.py``) launched
as a Gaussian in y rather than uniform in it, so every ``k_y`` column is populated
and the y transforms, the rows' blocks and the collar all act: the slab set's modal
amplitudes of the uniform launch times ``exp(-(y/w)^2)``, ``w`` a third of the y
radius, on an argon- or nitrogen-filled 127 um gap over at most ten fixed steps. A
collar case requests SEP 40's real-space collar (:data:`COLLAR`) in the y room.
"""

import dataclasses
import math

import numpy as np

from backends.test_planar_hybrid import FAMILIES, HALF_GAP, LENGTH, ORDERS
from solprop.boundaries import SpatialAbsorber, absorber
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid
from solprop.modes import ModeCollection, PlanarGrid, gaussian_coupling
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.stepper.schedule import FixedSchedule

RADIUS_Y = 1e-3
"""Half-width of the periodic y window, m."""

COLLAR = SpatialAbsorber(kappa=5.6e3, attenuation=3.0)
"""The collar cases' y collar: a declared band, kappa w = 22.4 on the 4 mm room.

Three nepers at the band's edge make a collar of 1.3 1/m, inside the default Adams
method's stability over one whole-length step (h alpha / 2 = 0.07 < 0.0877): the
cases test the collar's arithmetic, not its strength or its design. Twenty rows put the
band below the safety ramp, which starts at 0.9 k_y,max = 5.65e3 rad/m. They do not
resolve the launch for SEP 40's design check, which a collar case's translation skips
(:func:`planar_case`)."""

CASES = {
    # name: (hybrid family, n_y, collar_fraction, temporal collar boundary length m
    # or None, responses: the family's, none, or Kerr without its harmonic); a
    # nonzero collar_fraction requests COLLAR in the room.
    "kerr": ("kerr", 6, 0.0, None, "family"),
    "kerr-collar": ("kerr", 20, 4.0, 10.0, "family"),
    "kerr-split": ("kerr-split", 20, 4.0, 10.0, "family"),
    "raman": ("raman", 6, 0.0, None, "family"),
    "field-plasma": ("field-plasma", 4, 0.0, None, "family"),
    "field-kerr": ("field-plasma", 4, 0.0, None, "analytic"),
    "envelope-plasma": ("envelope-plasma", 3, 0.0, None, "family"),
    "collar-only": ("kerr", 20, 4.0, 10.0, "none"),
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
    family, n_y, collar_fraction, boundary, respond = CASES[name]
    grid, gas, energy, fwhm, plasma, raman, n_r, _, count, scheme = FAMILIES[family]
    steps = count if steps is None else max(1, steps)
    orders = ORDERS.get(family, (1, 3, 5))
    modes = ModeCollection.slab(HALF_GAP, orders, gas, 1.0).modes
    simulation = Simulation.from_waveguide(
        list(modes),
        LENGTH,
        gas,
        1.0,
        wavelength=800e-9,
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
        noise=None,
    )
    shares = gaussian_coupling(modes, 0.735 * HALF_GAP)
    pulse = simulation._field0[0]
    modal = simulation.with_input(
        np.stack([math.sqrt(share) * pulse for share in shares])
    ).translate()
    space = PlanarGrid(modes, RADIUS_Y, n_y, collar_fraction=collar_fraction, n_x=n_r)
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
    collar = collar_fraction > 0
    # A collar case is translated on a zero launch, whose design check is pending,
    # and then given its launch, which translation keeps as it is: the parity cases
    # compare the backends' arithmetic on the collar, not the design's return.
    problem = Simulation.from_planar(
        grid,
        space,
        responses,
        np.zeros_like(field0) if collar else field0,
        settings=settings,
        spatial_absorber=COLLAR if collar else None,
    ).translate()
    if collar:
        problem = dataclasses.replace(
            problem, initial=dataclasses.replace(problem.initial, field0=field0[None])
        )
    if node_chunk is not None:
        problem = dataclasses.replace(
            problem,
            transform=dataclasses.replace(problem.transform, node_chunk=node_chunk),
        )
    return grid, space, problem
