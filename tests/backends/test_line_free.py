# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The line (x-only) free space: the ``"free"`` kind with ``n_y = 1``.

:class:`solprop.grid.LineGrid` runs through the Cartesian seam. Native against
``numpy_ref`` on the line's ladders is the case catalogue's
(``tests/backends/free_rows.py``), whose bounds are the plane's with an FFT depth of
``log2(n_y) = 0``; what stays here is the translation of the line, and each backend's
statistics read afresh against its own session's.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.free_rows import (
    LINE_FAMILIES,
    LINE_STATISTICS,
    PLASMA_NAMES,
    REMAINDER,
    free_amplitude_bound,
    line_case,
)
from bounds import gamma
from sessions.runs import run as session_run
from solprop.backends.session import Start
from solprop.stats import StatsSpec


@pytest.fixture(params=LINE_FAMILIES)
def line_case_of(request):
    """The line ladders' case of one family (``free_rows.line_case``)."""
    return request.param, *line_case(request.param)


def test_line_problem_is_the_free_kind_with_one_y_column(line_case_of):
    _, _, space, problem, _ = line_case_of
    geometry = problem.transform.geometry
    assert geometry is not None
    assert geometry.kind == "free"
    assert geometry.shape == (space.n_x, 1)
    assert geometry.axes[1].tolist() == [0.0]
    assert geometry.spectral_axes[1].tolist() == [0.0]
    np.testing.assert_array_equal(geometry.spatial_weights, space.spatial_weights)
    assert problem.linear.separable is not None
    np.testing.assert_array_equal(problem.linear.separable.kperp_sq, space.k_x**2)


@pytest.mark.parametrize("steps", [0, 10])
def test_line_statistics_read_afresh_are_the_session_s(
    line_case_of, spatial_backend_class, steps
):
    """A line plane's statistics, read afresh from its state, are its own.

    The plane's statistics in the backend's own session (the launch's, or the end's
    after the case's eight steps), then the same state's afresh: the launch of a new
    session started from it there. The bound is the plane's for ten steps, as the old
    ladder held it.
    """
    family, grid, space, problem, run_request = line_case_of
    # The plane's statistics bound: twice the amplitude ladder for squared fields,
    # plus the column and temporal reductions.
    bound = 2 * free_amplitude_bound(
        family, grid, space, spatial_backend_class.name, steps
    )
    bound += gamma(128 * space.n_columns + 32 * grid.n_to)
    names = LINE_STATISTICS + (PLASMA_NAMES if family.endswith("plasma") else ())
    run_request = run_request.for_run(stats=StatsSpec(names=names))
    backend = spatial_backend_class(problem, run_request, grid=grid, options=REMAINDER)
    try:
        ran = session_run(
            backend, problem, run_request, owns=False, until=None if steps else 0.0
        )
        plane = ran.final if steps else ran.launch
        assert plane is not None
        again = Start(
            field=plane.field,
            z=plane.z,
            plane_index=int(
                np.searchsorted(run_request.stepper.save_planes, plane.z, "right")
            ),
        )
        relaunch = session_run(
            backend, problem, run_request, again, owns=False, until=plane.z
        ).launch
        assert relaunch is not None
        for name in names:
            assert_allclose(
                relaunch.stats[name],
                plane.stats[name],
                rtol=0,
                atol=bound * np.max(np.abs(plane.stats[name])),
            )
    finally:
        backend.close()
