# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar transform on native CPU: its plates, ledger and threads.

Native executes schema 15's ``"slab"`` kind in every combination ``numpy_ref`` does: the
y FFT around the slab rule's synthesis and projection in x, each column's operator with
its channel's ``kappa_sq``, the y collar in both absorber schemes, every response family
and the spatial and plasma statistics. The parity of each case with ``numpy_ref`` -- at
one nonlinear evaluation, one step and ten fixed steps, under :func:`amplitude_bound`
derived before the run, and of a planar driver run end to end -- is the case
catalogue's planar rows (``tests/backends/planar_rows.py``), on these cases
(``tests/backends/planar_plate_cases.py``).
Here: that the cases exercise the slab route, the ledger's planar buffers, and one
thread against four.

The cases are vacuum-filled plates of constant index (so no material fit is read and
the responses carry their own density, as the Cartesian ladders'), three slab orders
with the beam off centre in x so that odd and even orders are both driven, and a node
chunk of four y rows over 11, so that the last chunk is partial.
"""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.forced_threads import assert_threaded
from backends.matched import native_thread_fields
from backends.planar_plate_cases import (
    FAMILIES,
    N_Y,
    ORDERS,
    ROWS_PER_CHUNK,
    amplitude_bound,
    chunked,
    native_class,
    planar_case,
)
from solprop.backends.registry import available
from solprop.translate.routes import transverse_of

pytestmark = [
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
]

PROBE = Path(__file__).with_name("native_planar_thread_probe.py")


@pytest.fixture(params=FAMILIES, scope="module")
def case(request):
    return request.param, *planar_case(request.param)


def test_the_cases_exercise_the_slab_route(case):
    """The problems are the ``"slab"`` kind, with a partial final chunk."""
    family, _, space, problem, _ = case
    geometry = problem.transform.geometry
    assert geometry is not None
    assert geometry.kind == "slab"
    assert problem.linear.separable is not None
    assert problem.linear.separable.kappa_sq is not None
    assert problem.transform.synthesis.shape == (space.x.size, len(ORDERS))
    assert space.n_y % ROWS_PER_CHUNK != 0
    if family.startswith("collar"):
        assert (
            transverse_of(problem.route) is not None
            and (transverse := transverse_of(problem.route)) is not None
            and transverse.has_collar
        )
        assert np.any(geometry.absorber_rate > 0)


def test_the_ledger_counts_the_planar_buffers():
    """The y scratch, the slab rule and the wall terms, at their real byte cost."""
    grid, space, problem, run_request = planar_case("envelope_kerr")
    backend = native_class()(problem, run_request, grid=grid, options=chunked(space))
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    n_x, n_p = space.x.size, space.n_p
    assert ledger["transverse_scratch"] == grid.n_omega * n_p * N_Y * 16
    assert ledger["transform_matrices"] == 2 * n_x * n_p * 8
    separable = problem.linear.separable
    assert separable is not None
    assert separable.kappa_sq is not None
    # The free-space vectors, the per-column collar rate, the safety ramp's rate
    # (kernel schema 31, zeros here: assembled with no spatial_absorber) and the wall
    # terms.
    expected = (
        4 * grid.n_omega * 16
        + 3 * grid.n_omega * 8
        + 4 * space.n_columns * 8
        + separable.kappa_sq.nbytes
    )
    assert ledger["spatial_coefficients"] == expected
    # One chunk of y rows in time, not every column, and no modal tables.
    assert ledger["modal_time"] == ROWS_PER_CHUNK * n_p * grid.n_to * 16
    assert ledger["fixed_phi_terms"] == ledger["fixed_prefactor_terms"] == 0


@pytest.mark.slow
def test_the_planar_kind_agrees_across_thread_counts(tmp_path):
    """One thread against four, ten Runge-Kutta steps, each in a fresh process.

    The same method on the same fixed schedule, so the two differ by the order of
    their threaded transforms and contractions alone: the ten-step amplitude bound
    (:func:`amplitude_bound`), as against the reference.
    """
    names = ["envelope_kerr", "field_plasma", "collar_split_step"]
    one = native_thread_fields(PROBE, 1, names, tmp_path)
    four = native_thread_fields(PROBE, 4, names, tmp_path)
    assert_threaded(four["sizing"], 4, "kernels.stage", "kernels.response", "fft.fft")
    for name in names:
        grid, space, _, run_request = planar_case(name)
        assert run_request.stepper.integrator is None
        bound = amplitude_bound(grid, space, name, 10)
        scale = float(np.max(np.abs(one[name])))
        assert scale > 0
        assert_allclose(four[name], one[name], rtol=0, atol=bound * scale)
