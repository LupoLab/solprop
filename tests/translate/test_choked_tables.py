# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 14 phase 5: tables at a choked gas end.

Near a choked exit the steady-flow pressure goes as the square root of the distance to
the exit. Sampled in floating-point z, a function of that kind carries a relative noise
of about ``spacing(z) / (2 d)`` at a distance ``d``: within half a millimetre of a 4.2 m
exit it exceeds the default tolerance, the coefficient tail cannot converge, and the
split used to tile the region at its width floor. The builder now accepts a tail at the
sampling-noise floor of the interval's own positions, so the exit refines geometrically
as the entrance does.
"""

import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.gasflow import steady_fill
from solprop.translate.chebyshev import build
from solprop.translate.linop import delta_phi

EXIT = 4.2
STAGE = 2.1


@pytest.fixture(scope="module")
def fill():
    """The canonical capillary's fill: vacuum at both ends, 0.2 bar in the middle."""
    return steady_fill("He", [0.0, STAGE, EXIT], [0.0, 0.2, 0.0], radius=256e-6)


def exact(fill, distance):
    """The exit stage's pressure, written in the exact distance from the exit."""
    middle, end = float(fill.pressures[1]), float(fill.pressures[2])
    return np.sqrt(end**2 + (middle**2 - end**2) * distance / STAGE)


def test_the_exit_refines_as_the_entrance_does(fill):
    exit_table = build(fill.__call__, STAGE, EXIT)
    entrance_table = build(fill.__call__, 0.0, STAGE)
    # The two ends are mirror images; only the precision of z differs between them.
    # Measured 14 against 15 intervals; 37 at the exit before the sampling floor.
    assert exit_table.degrees.size <= entrance_table.degrees.size + 3
    widths = np.diff(exit_table.breakpoints)
    assert widths.min() > 1e-5


def test_the_exit_table_is_accurate_to_its_sampling_noise(fill):
    table = build(fill.__call__, STAGE, EXIT)
    rng = np.random.default_rng(5)
    distance = np.concatenate([np.geomspace(1e-7, 1e-2, 200), rng.uniform(0, 2.0, 200)])
    z = EXIT - distance
    values = np.array([float(table(one)) for one in z])
    reference = exact(fill, EXIT - z)
    relative = np.abs(values / reference - 1.0)
    # The noise floor of a position at distance d is spacing(4.2) / (2 d); the table
    # stays within a small multiple of it, and of the 1e-12 tolerance elsewhere.
    floor = np.maximum(1e-12, np.spacing(EXIT) / (2.0 * (EXIT - z)))
    assert np.all(relative <= 20.0 * floor)


def test_the_canonical_phi_table_is_unchanged_but_for_its_intervals():
    simulation = Simulation.from_capillary(
        256e-6,
        4.2,
        "He",
        steady_fill("He", [0.0, STAGE, EXIT], [0.0, 0.2, 0.0], radius=256e-6),
        lambda0=800e-9,
        fwhm=6e-15,
        energy=1e-3,
        modes=1,
        trange=60e-15,
        wavelength_limits=(90e-9, 1.6e-6),
        noise=None,
        n_save=2,
        backend="numpy_ref",
    )
    linear = simulation.translate().linear
    edges = linear.phi_breakpoints
    # 30 intervals, six in the last centimetre (123 and 99 before).
    assert linear.n_intervals <= 40
    assert np.count_nonzero((edges > 4.19) & (edges < EXIT)) <= 10
    # Phi itself: the exit's last micrometre is the same integral as before, to the
    # round-off of an 850 rad exponent (3.6e-13 rad measured between the builds).
    near = delta_phi(linear, EXIT - 1e-6, EXIT)
    assert np.all(np.isfinite(near))
    assert np.max(np.abs(near)) < 1e-3
