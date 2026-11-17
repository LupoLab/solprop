# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's engines against the reference's, through solprop's engine protocols.

SEP 52 sections 6.5 and 7: each case is written once against
:class:`~solprop.backends.engines.RungeKuttaEngine` or
:class:`~solprop.backends.engines.AdamsEngine` and played on both backends' engines
over a loaded workspace (``engine_pairs``). The Runge-Kutta case takes rejected and
accepted trials, reads the stage registers, the accept's split collar's sums and the
boundary reading; the low-storage case its in-flight interval; the Adams cases replay
the calls the reference's own march made on a run -- the startup, rejected trials,
accepts with the history they keep, the carrier chord and the step's carrier rotations
-- on the interaction formulation with transported history, the exponential one, and
the exponential one in the carrier frame.

**Budgets** (SEP 25, fixed before the comparison). Each step's both-sides rounding is
``tests/march_bounds.py``'s one-step share of :func:`march_bounds.march_bound` at the
case's measured nonlinear phase, ``s``; after ``k`` matched steps the states agree to
``k s`` normwise. A trial's scaled norm divides the embedded difference by about
``rtol ||a||``, so its rounding and the states' departure reach it as ``k s / rtol``,
times ``(1 + W Phi / S)`` for the departure's own effect on the stages (``W`` the
step's weights, ``Phi / S`` its share of the phase). The split collar's sums and the
boundary reading are quadratic in the field: four times the field's budget
(:data:`march_bounds.STATISTIC_FACTOR`). A stage register is one evaluation's rounding
(:func:`march_bounds.evaluation_rounding`) with the margin, plus three times the step's
budget for the cubic response to its input's departure; it is compared on the first
step only, where no recycled stage is involved.

**Non-vacuity.** Every compared term is asserted to exceed its bound by a wide factor,
and the split sums' measurement point is shown to matter: the same reduction of the
finished field (after the split) misses native's sum by more than the bound.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from typing import Any

import numpy as np
import pytest

from backend_corpus_cases import CARRIER, FIXED_ADAMS, capillary, walking_fibre
from bounds import EPS, assert_within, step_evaluations
from engine_pairs import AdamsCall, Opened, opened, play_adams, record_adams
from march_bounds import (
    MARGIN,
    STATISTIC_FACTOR,
    evaluation_rounding,
    march_bound,
    measured_scales,
)
from solprop.backends.engines import AcceptRecord, BoundaryReading
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.stepper import LowStorageRungeKutta, Multistep, RungeKutta

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="the native backend is not installed"
)

BACKENDS = ("numpy_ref", "native")

STEPS = 3
"""Accepted steps each Runge-Kutta case takes."""

ADAMS_ACCEPTS = 6
"""Accepted steps each Adams case replays: past the startup, into a full history."""

WIDE = 100.0
"""How many bounds a compared term must exceed to mean something."""


class Budget:
    """A case's budgets, from its measured phase, fixed before anything is compared."""

    def __init__(self, simulation: Simulation, steps: int | None = None) -> None:
        problem = simulation.translate()
        request = simulation.run_request()
        phi, kappa = measured_scales(simulation)
        phi = max(phi, 0.1)
        schedule = request.stepper.schedule
        total = int(schedule.n_steps) if schedule is not None else int(steps or 1)
        whole = march_bound(
            problem, request, total, phi=phi, kappa=kappa, backend="native"
        )
        self.step = whole / total
        self.rtol = float(request.stepper.controller.rtol)
        self.departure = 1.0 + step_evaluations(request) * phi / total
        self.stage = (
            MARGIN
            * kappa
            * evaluation_rounding(
                int(problem.grids.n_to),
                int(problem.transform.n_nodes),
                int(problem.shape[0]),
                raman=bool(problem.flags.has_raman),
                plasma=bool(problem.flags.has_plasma),
            )
            + 3 * self.step
        )

    def field(self, k: int) -> float:
        """The states' normwise budget after `k` matched steps."""
        return k * self.step

    def norm(self, k: int) -> float:
        """A scaled norm's absolute budget at the `k`-th trial."""
        return k * self.step * self.departure / self.rtol


def _check_norm(native: float, reference: float, bound: float, what: str) -> None:
    assert reference > WIDE * bound, f"{what}: the norm is too small to compare"
    assert abs(native - reference) <= bound, (
        f"{what}: {native!r} against {reference!r}, bound {bound:.3g}"
    )


def _check_record(
    native: AcceptRecord,
    reference: AcceptRecord,
    bound: float,
    what: str,
) -> float:
    """The split collar's sums of one accepted step, as the reference took them.

    Each sum weighs the field's energy over part of the window by weights of at most
    one, so the states' normwise departure bounds it absolutely, relative to the
    state's energy (``found[0]``), not to the sum: `bound` times the energy. Returns
    that absolute bound on the removal.
    """
    assert reference.found is not None and native.found is not None, what
    energy = float(reference.found[0])
    limit = bound * energy
    # What the collar took must stand well clear of what the bound allows.
    assert reference.removed > 10 * limit, f"{what}: the collar took too little"
    assert abs(native.removed - reference.removed) <= limit, what
    assert_within(native.found, reference.found, limit, scale=1.0, what=what)
    return limit


def _check_reading(
    native: BoundaryReading, reference: BoundaryReading, bound: float, what: str
) -> None:
    """A boundary reading: its weighted energies bounded relative to the state's."""
    limit = bound * reference.total
    assert reference.total > 0.0
    assert abs(native.total - reference.total) <= limit, what
    assert np.max(reference.regions) > 0.0, what
    assert_within(native.regions, reference.regions, limit, scale=1.0, what=what)
    assert reference.occupancy is not None and native.occupancy is not None, what
    assert_within(native.occupancy, reference.occupancy, limit, scale=1.0, what=what)


# -- the Runge-Kutta engine ---------------------------------------------------------


def _runge_kutta_case(case: Opened, simulation: Simulation) -> list[dict[str, Any]]:
    """Take :data:`STEPS` steps, each after a rejected trial; what each read."""
    engine = case.runge_kutta
    assert engine is not None
    request = simulation.run_request()
    schedule = request.stepper.schedule
    assert schedule is not None
    n_stages = request.stepper.tableau.n_stages
    steps = []
    for index in range(STEPS):
        z, dz = float(schedule.edges[index]), float(schedule.steps[index])
        before = case.field()
        rejected = engine.trial(z, 0.5 * dz)
        norm = engine.trial(z, dz)
        # The commit contract, on each backend: a trial commits nothing.
        assert np.array_equal(case.field(), before)
        assert not engine.in_flight
        stages = [case.stage(i) for i in range(n_stages)] if index == 0 else []
        record = engine.accept(dz)
        assert math.isnan(record.stiffness)
        steps.append({
            "dz": dz,
            "rejected": rejected,
            "norm": norm,
            "stages": stages,
            "record": record,
            "field": case.field(),
            "reading": engine.reading(),
        })  # fmt: skip
    return steps


@pytest.mark.medium
def test_a_runge_kutta_engine_steps_reads_and_splits_as_the_reference_s() -> None:
    simulation = walking_fibre("numpy_ref", RungeKutta())
    budget = Budget(simulation)
    with ExitStack() as stack:
        played = {}
        monitors = {}
        for backend in BACKENDS:
            case = stack.enter_context(
                opened(walking_fibre(backend, RungeKutta()), backend)
            )
            played[backend] = _runge_kutta_case(case, simulation)
            monitors[backend] = case.monitor
    launch = simulation.translate().initial.field0
    for k, (ours, theirs) in enumerate(
        zip(played["native"], played["numpy_ref"], strict=True), start=1
    ):
        what = f"step {k}"
        _check_norm(ours["rejected"], theirs["rejected"], budget.norm(k), what)
        _check_norm(ours["norm"], theirs["norm"], budget.norm(k), what)
        for index, (stage, expected) in enumerate(
            zip(ours["stages"], theirs["stages"], strict=True)
        ):
            assert_within(stage, expected, budget.stage, what=f"{what} stage {index}")
        assert_within(ours["field"], theirs["field"], budget.field(k), what=what)
        statistic = STATISTIC_FACTOR * budget.field(k)
        limit = _check_record(ours["record"], theirs["record"], statistic, what)
        _check_reading(ours["reading"], theirs["reading"], statistic, what)
        # The sums are the masked field's before the split: the same reduction of the
        # finished field, after it, is another number by more than the bound.
        monitor = monitors["numpy_ref"]
        assert monitor is not None
        after = float(monitor.split_sums(theirs["field"], ours["dz"])[0])
        assert abs(after - ours["record"].removed) > 10 * limit, what
    # The stages were read, and the state moved by far more than its budget.
    assert len(played["native"][0]["stages"]) > 1
    moved = np.linalg.norm(played["numpy_ref"][-1]["field"] - launch)
    assert moved > WIDE * budget.field(STEPS) * np.linalg.norm(launch)


def test_a_dense_extension_leaves_the_trial_and_its_accept_alone() -> None:
    # `extend` holds the trial's continuous extension for a dense observation, which
    # only the session reads; through the protocol, a held extension changes neither
    # the committed state nor what the accept commits, on either backend.
    simulation = capillary("numpy_ref", RungeKutta(), interpolated=True)
    budget = Budget(simulation, steps=1)
    dz = float(simulation.run_request().stepper.dz_init)
    found = {}
    for backend in BACKENDS:
        with opened(
            capillary(backend, RungeKutta(), interpolated=True), backend
        ) as case:
            engine = case.runge_kutta
            assert engine is not None
            before = case.field()
            engine.trial(0.0, dz)
            engine.extend(0.0, dz)
            assert np.array_equal(case.field(), before)
            engine.accept(dz)
            found[backend] = case.field()
    assert_within(found["native"], found["numpy_ref"], budget.field(1))


# -- the low-storage engine -----------------------------------------------------------


@pytest.mark.medium
def test_a_low_storage_engine_is_in_flight_and_commits_as_the_reference_s() -> None:
    simulation = walking_fibre("numpy_ref", LowStorageRungeKutta())
    budget = Budget(simulation)
    schedule = simulation.run_request().stepper.schedule
    assert schedule is not None
    played: dict[str, list[tuple[AcceptRecord, np.ndarray]]] = {}
    for backend in BACKENDS:
        with opened(walking_fibre(backend, LowStorageRungeKutta()), backend) as case:
            engine = case.runge_kutta
            assert engine is not None
            played[backend] = []
            for index in range(STEPS):
                z, dz = float(schedule.edges[index]), float(schedule.steps[index])
                assert not engine.in_flight
                # "Not estimated": the step overwrites the state, the register A.
                assert engine.trial(z, dz) == 0.0
                assert engine.in_flight
                record = engine.accept(dz)
                assert not engine.in_flight
                played[backend].append((record, case.field()))
    for k, ((ours, field), (theirs, expected)) in enumerate(
        zip(played["native"], played["numpy_ref"], strict=True), start=1
    ):
        what = f"step {k}"
        # The stiffness diagnostic is the reference engine's alone (SEP 52 5.2).
        assert math.isfinite(theirs.stiffness) and theirs.stiffness > 0.0
        assert math.isnan(ours.stiffness)
        assert_within(field, expected, budget.field(k), what=what)
        _check_record(ours, theirs, STATISTIC_FACTOR * budget.field(k), what)


# -- the Adams engine ---------------------------------------------------------------

ADAMS = {
    "interaction": lambda backend: capillary(
        backend, Multistep(variable_order=False, formulation="interaction"), modes=3
    ),
    "exponential": lambda backend: capillary(backend, FIXED_ADAMS),
    "carrier": lambda backend: capillary(backend, CARRIER, interpolated=True),
}
"""The Adams cases: retained-history transport and the split collar; the exponential
formulation and the split collar; the exponential formulation in the carrier frame,
adaptive, whose trials form carrier rotations."""


@pytest.mark.medium
@pytest.mark.parametrize("name", list(ADAMS))
def test_an_adams_engine_replays_the_reference_s_march_as_the_reference_does(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation = ADAMS[name]("numpy_ref")
    calls = record_adams(simulation, monkeypatch, accepts=ADAMS_ACCEPTS)
    accepts = sum(call.name == "accept" for call in calls)
    assert accepts == ADAMS_ACCEPTS
    budget = Budget(simulation, steps=accepts)
    played = {}
    for backend in BACKENDS:
        with opened(ADAMS[name](backend), backend) as case:
            played[backend] = play_adams(case, calls)
    _compare_adams(name, calls, played["native"], played["numpy_ref"], budget)


def _compare_adams(
    name: str,
    calls: list[AdamsCall],
    ours: list[tuple[str, Any]],
    theirs: list[tuple[str, Any]],
    budget: Budget,
) -> None:
    """Hold native's replay to the reference's, call by call."""
    # The single-mode capillary's collar takes about 1e-11 of the energy at a step,
    # below what the states' budget can resolve; the three-mode one's group delays
    # put a thousand times more there. The carrier case has the in-stage collar.
    split = name == "interaction"
    k = 1
    carriers = 0
    for (call, value), (other, expected) in zip(ours, theirs, strict=True):
        assert call == other
        what = f"{name} {call} at step {k}"
        if call == "trial":
            reference = np.asarray(expected)
            bound = budget.norm(k)
            assert reference[0] > WIDE * bound, f"{what}: the norm is too small"
            np.testing.assert_allclose(value, reference, rtol=0.0, atol=bound)
        elif call == "accept":
            (record, field), (wanted, state) = value, expected
            assert_within(field, state, budget.field(k), what=what)
            if split:
                _check_record(record, wanted, STATISTIC_FACTOR * budget.field(k), what)
            k += 1
        elif call == "chord_phase":
            assert value == pytest.approx(expected, rel=64 * EPS, abs=64 * EPS)
        else:
            reference = np.asarray(expected)
            assert np.shape(value) == reference.shape, what
            if reference.size:
                carriers += 1
                # A rotation's phase, the shift times a node's span, carries its
                # chord's rounding: a few eps of the shift, rad.
                assert reference[0] != 0.0, f"{what}: no carrier shift"
                tolerance = 64 * EPS * max(1.0, abs(float(reference[0])))
                np.testing.assert_allclose(value, reference, rtol=0.0, atol=tolerance)
    if name == "carrier":
        assert carriers > 0, "no trial formed carrier rotations"
    if name == "interaction":
        # The transport is exercised: accepts kept more than one entry.
        kept = [len(call.args[3]) for call in calls if call.name == "accept"]
        assert max(kept) > 1
