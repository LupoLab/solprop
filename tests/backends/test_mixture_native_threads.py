# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the optional native backend's two-channel run across processes and threads."""

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from backends.matched import native_thread_fields
from backends.native_mixture_thread_probe import NAMES, STEPS, problem_and_grid
from bounds import assert_within, device_backend

pytest.importorskip("solprop_native")

PROBE = Path(__file__).with_name("native_mixture_thread_probe.py")


@pytest.mark.skipif(
    device_backend("native") != "none", reason="a device build's host space is serial"
)
def test_two_channels_are_repeatable_and_thread_invariant(tmp_path):
    """One thread twice and four threads, eight fixed Runge-Kutta steps each.

    The same method on the same schedule, so the runs differ by the order of their
    threaded transforms and reductions alone. The bound is the mixture ladder's
    (``test_mixture_ladder.py``), fixed from its operation counts: 64 eps per fine
    sample, per channel, per stage, seven stages a step, eight steps; the statistics
    are held to it relatively, the ionised fractions absolutely as well.
    """
    problem, grid = problem_and_grid()
    assert problem.stepper.integrator is None
    names = ["threads", "steps", "rows", "field"] + [
        f"{kind}:{name}" for kind in ("rows", "stats") for name in NAMES
    ]
    runs = [
        native_thread_fields(PROBE, threads, names, tmp_path) for threads in (1, 1, 4)
    ]
    # The work done is accounting, exact: a prescribed schedule runs its steps.
    assert [int(run["threads"]) for run in runs] == [1, 1, 4]
    assert all(int(run["steps"]) == int(run["rows"]) == STEPS for run in runs)
    bound = 64 * grid.n_to * 2 * 7 * STEPS * np.finfo(float).eps
    first = runs[0]
    for run in runs[1:]:
        assert_within(run["field"], first["field"], bound, what="field")
        for key in names[4:]:
            absolute = bound if key.endswith("species_ionisation") else 0.0
            assert_allclose(
                run[key], first[key], rtol=bound, atol=absolute, err_msg=key
            )
