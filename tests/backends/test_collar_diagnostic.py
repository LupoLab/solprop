# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Cartesian and planar collar diagnostic, accumulated by block of frequency.

The diagnostic is the collar's share of the beam's electric-field norm, a boundary
check rather than a fraction of physical fluence (SEP 34 §5.1). SEP 22 section 6.3:
each node's field norm is a sum over frequency, so blocks accumulate
the whole-volume diagnostic exactly up to the order of the sums. The comparison is
class R (``docs/dev/testing.md`` section 3): Higham's ``gamma_m`` of each sum of ``m``
non-negative terms, and the transverse transform's own bound, fixed before measuring.
"""

import dataclasses
import tracemalloc

import numpy as np
import pytest

from bounds import assert_within, fft_bound, higham_gamma
from driver.test_field_free_planes import cartesian
from driver.test_prop_planar import GRID, pulse, settings, slab_modes
from solprop import fourier
from solprop.backends import spatial
from solprop.backends.spatial import collar_block, collar_field_fraction
from solprop.driver import Simulation
from solprop.modes.slab import PlanarGrid


def whole_volume_fraction(problem, field) -> float:
    """Return the field-norm diagnostic computed over the whole volume at once."""
    transform = problem.transform
    geometry = transform.geometry
    full = spatial._free_physical(transform, field, problem.grids.pad_scale)
    assert geometry is not None and full is not None
    dt = problem.grids.dt_fine * problem.grids.pad_scale
    fluence = np.sum(fourier.energy_omega(full, dt, kind=problem.grid_kind), axis=1)
    energy = geometry.spatial_weights * fluence
    return float(np.sum(energy[geometry.window < 1]) / np.sum(energy))


def planar(**changes):
    space = PlanarGrid(slab_modes(), 1.5e-3, 32, collar_fraction=0.3)
    return Simulation.from_planar(
        GRID, space, [], pulse(), settings=settings(**changes)
    )


@pytest.mark.parametrize("kind", ["free", "slab"])
@pytest.mark.parametrize("bins", [1, 7, None])
def test_the_blockwise_fraction_is_the_whole_volume_one(monkeypatch, kind, bins):
    simulation = cartesian("numpy_ref") if kind == "free" else planar()
    problem = simulation.translate()
    field = problem.initial.field0[0] * (1 + 0.3j)
    # A beam that reaches the collar: shift it towards a corner of the window.
    field = field * np.exp(1j * np.linspace(0.0, 3.0, field.shape[0]))[:, None, None]
    expected = whole_volume_fraction(problem, field)
    assert expected > 0
    if bins is not None:
        monkeypatch.setattr(spatial, "collar_block", lambda problem: bins)
    measured = collar_field_fraction(problem, field)
    terms = field.size
    # Two sums of `terms` non-negative terms each (gamma_m apiece), their quotient,
    # and a physical field taken by a different grouping of the same transforms.
    bound = 2 * higham_gamma(terms) + 4 * fft_bound(problem.transform.n_nodes)
    assert_within(measured, expected, bound, what=f"{kind}, {bins} bins")


def test_the_block_is_bounded_by_the_node_chunk_and_so_is_the_transient():
    simulation = cartesian("numpy_ref")
    # Four nodes' columns of the response pass: the Cartesian translation charges
    # twenty fine temporal columns per node (translate/free.py).
    simulation.settings = dataclasses.replace(
        simulation.settings, node_budget=4 * 20 * 16 * simulation.grid.n_to
    )
    problem = simulation.translate()
    assert problem.transform.node_chunk == 4
    transform = problem.transform
    n_pol, n_omega = problem.shape[2], problem.shape[3]
    block = collar_block(problem)
    assert 1 <= block < n_omega
    assert 3 * transform.n_nodes * n_pol * block <= max(
        transform.node_chunk * n_pol * problem.grids.n_to, 3 * transform.n_nodes * n_pol
    )
    field = np.ascontiguousarray(problem.initial.field0[0])
    collar_field_fraction(problem, field)  # warm any first-call allocation
    tracemalloc.start()
    collar_field_fraction(problem, field)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    volume = 16 * transform.n_nodes * n_pol * n_omega
    working = 16 * 3 * transform.n_nodes * n_pol * block
    # One block's working set (three block volumes) and a few node- and
    # frequency-length vectors, against the whole volume the diagnostic used to hold.
    assert peak <= working + 64 * (transform.n_nodes + n_omega) + 65536
    assert peak < volume


def test_a_small_problem_takes_its_whole_band_in_few_blocks():
    problem = cartesian("numpy_ref").translate()
    block = collar_block(problem)
    assert block >= problem.shape[3] // 3
