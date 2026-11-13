# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native on the retired kernels' synthetic separable operators (SEP 43, 31, 32, 34).

The Numba kernels were called alone on small synthetic leaves: loss and gain, an
exactly zero longitudinal wave number, evanescent columns, an inactive bin, a wall with
gain and a negative real part, low words of 1e-17 and a safety ramp, all at once. No
real medium has those together, so no native run met them. Native's kernels cannot be
called alone, but its operator is read from the leaf: these tests put the same values
on the leaf of a small real problem, with order-one steps and an order-one state, and
compare a whole step, on the low-storage march and on the Runge-Kutta march, with
``numpy_ref``, whose operator is ``separable_values``: the catalogue's rows
(``backends/geometry_rows.py``), on the leaves of ``backends/synthetic_leaf_cases.py``.
What stays here is
what a row does not compare: the inactive bin's hard zero, the norm the march hands its
accept, and a diverged march's refusal.

**The bound.** The original's was 128 eps per kernel on order-one operands, with no
transform between them. A step here has four evaluations of a transform pair, so it
takes :func:`low_storage_cases.parity_bound`, the same count over the whole step, with
the growth computed from the leaf: ``exp(max Re D dz)``, where the original's operands
did not grow. **Weaker than the original.** A pass isn't read alone, so a fold's
intermediate registers aren't compared, only what the whole step leaves.

Each backend runs the step in its own session (SEP 47 phase 5). The squared norm a
native low-storage step hands its accept stays inside the native session's loop, so it
is read from the resident low-storage engine, one step on the backend's native state.
"""

import dataclasses
import os

import numpy as np
import pytest

from backends.geometry_rows import SYNTHETIC_CASES, synthetic
from backends.low_storage_cases import MARCH
from backends.synthetic_leaf_cases import (
    KERR,
    free_leaf,
    free_problem,
    on_leaf,
)
from bounds import gamma
from sessions.runs import load_native
from sessions.runs import run as session_run
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import ConvergenceError

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

solprop_native = pytest.importorskip("solprop_native")
backend_module = pytest.importorskip("solprop_native.backend")
NativeBackend = solprop_native.NativeBackend


def run(implementation, problem, run_request, grid):
    """Replay the schedule to its end in the backend's session; return the final state.

    A failed run raises its error, as the run loop does.
    """
    backend = implementation(problem, run_request, grid=grid)
    try:
        landed = session_run(backend, problem, run_request)
    finally:
        backend.close()
    error = landed.terminal.error
    if error is not None:
        raise error
    assert landed.observations[-1].z == pytest.approx(
        float(run_request.stepper.save_planes[-1]), rel=1e-12
    )
    return landed.fields[-1]


def resident_step(problem, run_request, grid):
    """Take the schedule's one step on native's resident low-storage engine.

    Returns the squared norm the engine hands the accept, and the state after the
    backend's finish, as the low-storage march accepts a step.
    """
    edges = np.asarray(run_request.stepper.schedule.edges)
    assert edges.size == 2, "one step"
    dz = float(edges[1] - edges[0])
    backend = NativeBackend(problem, run_request, grid=grid)
    try:
        state = load_native(backend, problem.initial.field0, clear_fsal=False)
        engine = backend_module._ResidentLowStorage(backend, MARCH)
        squared = engine.step(float(edges[0]), dz)
        state.finish(dz)
        field = np.asarray(state.field_to_host()).reshape(problem.shape)
    finally:
        backend.close()
    return squared, field


@pytest.mark.parametrize(("case", "mode", "n_pol"), SYNTHETIC_CASES)
def test_the_inactive_bin_stays_a_hard_zero(case, mode, n_pol):
    """Class P: after the step the inactive bin is a hard zero on both backends.

    The step itself is held to the reference by the catalogue's rows
    (``backends/geometry_rows.py``).
    """
    built = synthetic(case, mode, n_pol)
    leaf = built.problem.linear.separable
    assert leaf is not None
    inactive = ~np.asarray(leaf.active)
    for implementation in (NumpyRefBackend, NativeBackend):
        state = run(implementation, built.problem, built.run_request, built.grid)
        np.testing.assert_array_equal(state[..., inactive], 0.0)


def with_masked_bin(problem, leaf):
    """Return `problem` with its fullest active bin out of the norm, and the mask."""
    power = np.sum(np.abs(problem.initial.field0) ** 2, axis=(0, 1))
    power[~np.asarray(leaf.active)] = 0.0
    mask = np.array(problem.grids.norm_mask)
    mask[int(np.argmax(power))] = False
    return (
        dataclasses.replace(
            problem, grids=dataclasses.replace(problem.grids, norm_mask=mask)
        ),
        mask,
    )


def test_the_march_s_norm_reads_a_mask_with_a_hard_zero(monkeypatch):
    """Inventory 31 (b): the squared norm the march returns, with a populated bin out.

    The original read it from the fold's partials. Here it is what the engine hands
    the accept, on each backend, against the sum over the mask of the state that step
    leaves, within one sum's rounding: the reference's in its session, native's from
    its resident engine, whose state is the native session's own to the bit.
    """
    problem, run_request, grid = free_problem(responses=[KERR])
    leaf = free_leaf(problem.linear.separable)
    problem, mask = with_masked_bin(on_leaf(problem, leaf), leaf)
    seen = []

    def capture(self, z, dz, squared, accept=HostLowStorageMarch._accept):
        seen.append(squared)
        return accept(self, z, dz, squared)

    monkeypatch.setattr(HostLowStorageMarch, "_accept", capture)
    reference = run(NumpyRefBackend, problem, run_request, grid)
    assert len(seen) == 1
    squared, resident = resident_step(problem, run_request, grid)
    session = run(NativeBackend, problem, run_request, grid)
    if "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }:
        # One thread fixes every reduction's order: the probe is the run's step.
        np.testing.assert_array_equal(resident, session)
    for norm, state in ((seen[0], reference), (squared, resident)):
        masked = float(np.sum(np.abs(state[..., mask]) ** 2))
        whole = float(np.sum(np.abs(state) ** 2))
        reductions = gamma(state.size)
        assert norm == pytest.approx(masked, rel=reductions)
        # Vacuous unless the masked bin moves the sum by far more than that.
        assert abs(whole - masked) > 1e3 * reductions * whole


@pytest.mark.filterwarnings("ignore::solprop.exceptions.ExponentBudgetWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("implementation", ["numpy_ref", "native"])
def test_a_diverged_march_is_refused_by_name(implementation):
    """Inventory 31 (b)'s other half: a gain past overflow is refused, not returned.

    Native had no diverged march. Every active bin's ``k^2`` carries a gain of about
    ``e^650`` over the step, which translation admits (it refuses a propagator that
    itself overflows), and whose squared norm overflows on the first step.
    """
    problem, run_request, grid = free_problem(responses=[KERR])
    leaf = free_leaf(problem.linear.separable)
    leaf = dataclasses.replace(
        leaf, k_squared=np.where(leaf.active, 1.0 - 9.4e6j, leaf.k_squared)
    )
    problem = on_leaf(problem, leaf)
    backend = {"numpy_ref": NumpyRefBackend, "native": NativeBackend}[implementation]
    with pytest.raises(ConvergenceError, match="diverged"):
        run(backend, problem, run_request, grid)
