# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The new native seam (solprop SEP 52 section 5.3): records, workspace, probe.

The field tables hold each Python record that crosses to its C++ table
(``_core.schemas()``); each record round-trips through the converter, and the
converter refuses a malformed record by its field's path. A ``_core.Workspace`` built
from one ``Problem`` with its set-once data, started from one run record, runs a session
to the same fields as the backend's own session over the same problem; the probe repeats
its trials, reads back an owning copy, refuses the low-storage march's field, and is
refused while a session holds the workspace. The
records are the ones the backend built its own workspace from, sent through the
bindings directly.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pytest
from conftest import PhaseACase
from session_runs import run
from solprop.backends import engines as protocol
from solprop.backends import records as shared
from solprop.backends.session import Start
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import MarchPolicy, Multistep
from solprop.stepper.schedule import FixedSchedule

from solprop_native import _core
from solprop_native._load import load
from solprop_native.backend import (
    FRESH_WATCH,
    NativeBackend,
    _lower,  # pyright: ignore[reportPrivateUsage]
    _LowerPlan,  # pyright: ignore[reportPrivateUsage]
)

_testing: Any = cast(Any, _core)._testing
EPS = float(np.finfo(np.float64).eps)

# -- the schema test ------------------------------------------------------------------

_RECORDS: list[tuple[type, str]] = [
    (shared.RungeKuttaPolicy, "RungeKuttaPolicy"),
    (PIController, "PIController"),
    (MarchPolicy, "MarchPolicy"),
    (shared.LowStorageSettings, "LowStorageSettings"),
    (FixedSchedule, "FixedSchedule"),
    (shared.PlanePlan, "PlanePlan"),
    (shared.MonitorConfig, "MonitorConfig"),
    (shared.EdgeConfig, "EdgeConfig"),
    (shared.Resumption, "Resumption"),
    (shared.EdgeResumption, "EdgeResumption"),
    (shared.MonitorContinuation, "MonitorContinuation"),
    (shared.MonitorRecords, "MonitorRecords"),
    (shared.RunRecords, "RunRecords"),
    (protocol.HistoryEntry, "march.Entry"),
]
"""Each Python record that crosses to native, and the C++ table that reads it."""

_DTYPES = {"float64": "float64", "complex128": "complex128", "uint64": "uint64"}


def _optional(hint: object) -> tuple[bool, object]:
    """Return whether `hint` admits None, and what it is without None."""
    if isinstance(hint, types.UnionType) or typing.get_origin(hint) is typing.Union:
        arguments = [one for one in typing.get_args(hint) if one is not type(None)]
        none = len(arguments) < len(typing.get_args(hint))
        return none, arguments[0] if len(arguments) == 1 else hint
    return False, hint


def _array_facts(hint: object) -> tuple[set[str], list[str]] | None:
    """Return a jaxtyping annotation's dtypes and axis names, or None for another."""
    dtypes = getattr(hint, "dtypes", None)
    dims = getattr(hint, "dim_str", None)
    if dtypes is None or dims is None:
        return None
    return set(cast("tuple[str, ...]", dtypes)), str(dims).split()


@pytest.mark.parametrize(("record", "name"), _RECORDS, ids=[one[1] for one in _RECORDS])
def test_each_python_record_is_held_to_its_native_table(
    record: type, name: str
) -> None:
    """A renamed field, a lost optional, a changed dtype or axis name fails here."""
    table = _core.schemas()[name]
    entries = {entry["name"]: entry for entry in table["fields"]}
    fields = [one.name for one in dataclasses.fields(record)]
    assert [entry["name"] for entry in table["fields"]] == fields
    hints = typing.get_type_hints(record)
    for field in fields:
        entry = entries[field]
        none, bare = _optional(hints[field])
        facts = _array_facts(bare)
        if entry["kind"] == "array":
            assert facts is not None, f"{name}.{field} is an array natively only"
            dtypes, axes = facts
            assert entry["dtype"] in dtypes, (name, field, entry["dtype"], dtypes)
            assert entry["optional"] == none, (name, field)
            # Each named axis is the native table's relation, symbol for symbol.
            assert len(axes) == len(entry["extents"]), (name, field)
            for axis, extent in zip(axes, entry["extents"], strict=True):
                if not axis[0].isdigit():
                    assert axis == extent, (name, field, axis, extent)
        else:
            assert facts is None, f"{name}.{field} is an array in Python only"
            assert (entry["kind"] == "optional") == none, (name, field, entry)


def test_the_schemas_state_the_relations_a_transposed_table_breaks() -> None:
    raman = {
        entry["name"]: entry for entry in _core.schemas()["Problem.Raman"]["fields"]
    }
    assert raman["tensor_weights"]["extents"] == ["n_terms", "2"]
    table = {entry["name"]: entry for entry in _core.schemas()["stage.Table"]["fields"]}
    assert table["coefficients"]["extents"] == ["n_edges-1", "n_terms", "n_values"]
    assert _core.schemas()["stage.Table"]["counts"] == {
        "n_terms": "n_terms",
        "n_values": "n_values",
    }


# -- the converter -------------------------------------------------------------------


def _same(left: object, right: object) -> None:
    """Assert that a written-back record holds what was read, by value."""
    if dataclasses.is_dataclass(left) and not isinstance(left, type):
        left = {one.name: getattr(left, one.name) for one in dataclasses.fields(left)}
    if isinstance(left, Mapping):
        assert isinstance(right, Mapping)
        left_map = cast("Mapping[str, object]", left)
        right_map = cast("Mapping[str, object]", right)
        assert set(left_map) == set(right_map)
        for key, value in left_map.items():
            _same(value, right_map[key])
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert right.dtype == left.dtype
        np.testing.assert_array_equal(right.reshape(left.shape), left)
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, list)
        assert len(right) == len(left)  # pyright: ignore[reportUnknownArgumentType]
        for one, other in zip(left, right, strict=True):  # pyright: ignore[reportUnknownArgumentType]
            _same(one, other)
    elif isinstance(left, float) and math.isnan(left):
        assert isinstance(right, float)
        assert math.isnan(right)
    else:
        assert right == left, (left, right)


def test_a_run_s_records_round_trip(phase_a_case: PhaseACase) -> None:
    problem, request, _ = phase_a_case
    records = shared.run_records(problem, request, Start.entrance(problem))
    back = _testing.round_trip("RunRecords", records)
    _same(records, back)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LowStorageSettings", shared.LowStorageSettings(2, True, 0.25)),
        ("FixedSchedule", FixedSchedule(np.array([0.1, 0.2, 0.3]), 0.5)),
        (
            "Resumption",
            shared.Resumption(
                record=True,
                collar_quiet=True,
                numbers={"launch_energy": 2.0, "steps": 3.0},
                unwatched=((0.0, 0.1),),
                resumed_at=(0.1,),
                planes=((0.1,), (0.2,), (0.3,), (0.4,), (0.5,)),
            ),
        ),
        (
            "EdgeResumption",
            shared.EdgeResumption(
                record=True,
                numbers={"current": 1.5},
                edge_baseline=(1.0, 2.0),
                reads=4,
                transverse_status="clear",
            ),
        ),
        (
            "stage.Table",
            {
                "edges": np.array([0.0, 0.5, 1.0]),
                "lower": 0.0,
                "upper": 1.0,
                "coefficients": np.arange(12.0).reshape(2, 3, 2),
            },
        ),
        (
            "ZTable",
            {
                "kind": "phi",
                "edges": np.array([0.0, 1.0]),
                "lower": 0.0,
                "upper": 1.0,
                "coefficients": np.ones((1, 2, 3), dtype=np.complex128),
            },
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_records_round_trip(name: str, value: object) -> None:
    _same(value, _testing.round_trip(name, value))


def _table(n_edges: int = 3) -> dict[str, object]:
    return {
        "edges": np.linspace(0.0, 1.0, n_edges),
        "lower": 0.0,
        "upper": 1.0,
        "coefficients": np.zeros((n_edges - 1, 2, 1)),
    }


@pytest.mark.parametrize(
    ("spoil", "error", "message"),
    [
        (lambda t: t.pop("lower"), ValueError, r"stage\.Table\.lower is missing"),
        (lambda t: t.update(extra=1), ValueError, "unknown field 'extra'"),
        (
            lambda t: t.update(coefficients=np.zeros((2, 2, 1), dtype=np.float32)),
            TypeError,
            r"coefficients must be a C-contiguous float64 array of rank 3",
        ),
        (
            lambda t: t.update(coefficients=np.zeros((2, 2))),
            ValueError,
            "of rank 3",
        ),
        # The right element count with its axes swapped breaks the relation.
        (
            lambda t: t.update(coefficients=np.zeros((1, 2, 2))),
            ValueError,
            r"axis 0 gives n_edges = 2, where the record has 3",
        ),
        (
            lambda t: t.update(coefficients=np.zeros((2, 4, 1))[:, ::2]),
            ValueError,
            "not a strided or overlapping view",
        ),
        (
            lambda t: t.update(coefficients=np.broadcast_to(np.zeros(1), (2, 2, 1))),
            ValueError,
            "not a strided or overlapping view",
        ),
        (lambda t: t.update(lower="0"), TypeError, r"stage\.Table\.lower must be a"),
    ],
)
def test_the_converter_refuses_a_malformed_record_by_its_path(
    spoil: Any, error: type[Exception], message: str
) -> None:
    table = _table()
    spoil(table)
    with pytest.raises(error, match=message):
        _testing.round_trip("stage.Table", table)


def test_the_converter_admits_read_only_arrays_and_copies_them() -> None:
    table = _table()
    coefficients = cast(np.ndarray, table["coefficients"])
    coefficients.setflags(write=False)
    back = _testing.round_trip("stage.Table", table)
    assert back["coefficients"].flags.owndata or back["coefficients"].base is not None
    assert not np.shares_memory(back["coefficients"], coefficients)


def test_a_variant_is_told_apart_by_its_fields(phase_a_case: PhaseACase) -> None:
    problem, request, _ = phase_a_case
    records = shared.run_records(problem, request, Start.entrance(problem))
    stranger = {
        "policy": {"order": 1},
        **{
            k: getattr(records, k)
            for k in ("low_storage", "schedule", "planes", "monitor")
        },
    }
    with pytest.raises(
        TypeError, match="matches none of MarchPolicy or RungeKuttaPolicy"
    ):
        _testing.round_trip("RunRecords", stranger)


# -- the workspace: a backend's own records, through the bindings directly ------------

_PLANNING = {"mode": "legacy"}
_OBSERVATION, _TERMINAL = 0, 3


def _workspace(backend: Any, problem: Any, request: Any) -> _core.Workspace:
    """Build a second workspace from the records the backend built its own from."""
    native = _lower(
        problem,
        request,
        _LowerPlan(
            terms=backend._terms,
            flow=backend._flow,
            tables=backend._z_tables,
            node_chunk=backend._node_chunk,
            general_chunk=backend._general_chunk,
            cpu_execution=backend._applied_execution,
            fft_planning=_PLANNING,
            setup=backend._setup(),
        ),
    )
    return _core.Workspace(native)


def _run_start(backend: Any, problem: Any, request: Any) -> dict[str, object]:
    """Return one run's start in the new seam's record, at the entrance."""
    start = Start.entrance(problem)
    records = shared.run_records(problem, request, start)
    loaded = load(
        problem,
        request.stepper,
        np.asarray(start.take_field()),
        start.noise,
        z=float(start.z),
        resume=start.resume,
    )
    proposal = float(request.stepper.dz_init)
    adams = runge_kutta = None
    if isinstance(request.stepper.integrator, Multistep):
        adams = backend._adams.start_state(loaded.z, proposal, loaded.cursor)
    else:
        runge_kutta = {
            "z": float(loaded.z),
            "proposal": proposal,
            "norms": [],
            "watch": dict(FRESH_WATCH),
            "cursor": int(loaded.cursor),
        }
    return {
        "records": records,
        "field": np.ascontiguousarray(loaded.field.reshape(-1)),
        "noise": np.ascontiguousarray(loaded.native_noise()),
        "adams": adams,
        "runge_kutta": runge_kutta,
        "rows": [],
        "plane_statistics": None,
        "row_statistics": None,
        "floor": None,
        "edge_noise": None,
        "monitor": None,
        "edge": None,
    }


def _drive(session: Any, elements: int) -> tuple[list[tuple[float, np.ndarray]], Any]:
    """Advance a native session to its end, exporting each observed state."""
    fields: list[tuple[float, np.ndarray]] = []
    recipe = {"power": False, "optics": False, "weight": None}
    while True:
        batch = session.advance(4096, 3600.0)
        event = batch.event
        if event.kind == _OBSERVATION and event.ticket:
            out = np.empty(elements, dtype=np.complex128)
            session.export((batch.sequence, event.generation), recipe, out)
            fields.append((float(event.z), out))
        session.acknowledge(batch.sequence)
        if event.kind == _TERMINAL:
            return fields, event


def _case(phase_a_case: PhaseACase, march: str) -> tuple[Any, Any, Any]:
    problem, request, grid = phase_a_case
    stepper = request.stepper
    if march == "adams":
        stepper = dataclasses.replace(stepper, integrator=Multistep())
    elif march == "low_storage":
        target = float(stepper.save_planes[-1])
        stepper = dataclasses.replace(
            stepper,
            integrator=LowStorageRungeKutta(),
            schedule=FixedSchedule.uniform(0.0, target, target / 4),
        )
    return problem, request.with_stepper(stepper), grid


@pytest.mark.parametrize("march", ["runge_kutta", "adams", "low_storage"])
def test_a_workspace_runs_a_session_as_the_old_seam_does(
    phase_a_case: PhaseACase, march: str
) -> None:
    problem, request, grid = _case(phase_a_case, march)
    backend = NativeBackend(problem, request, grid=grid, fft_planning=_PLANNING)
    try:
        old = run(backend, problem, request)
        workspace = _workspace(backend, problem, request)
        start = _run_start(backend, problem, request)
    finally:
        backend.close()
    try:
        session = workspace.start(start)
        fields, terminal = _drive(session, int(np.prod(problem.shape)))
        session.close()
        assert terminal.outcome == 1  # completed
        old_planes = old.all_planes()
        assert [z for z, _ in fields] == [plane.z for plane in old_planes]
        # The same operations on the same build, over the same tables: the matched
        # computation, compared bitwise here as a single-build diagnostic (SEP 25).
        for (_, field), plane in zip(fields, old_planes, strict=True):
            np.testing.assert_array_equal(field, plane.field.reshape(-1))
    finally:
        workspace.close()


def test_a_workspace_is_reused_across_starts_and_its_arrays_outlive_it(
    phase_a_case: PhaseACase,
) -> None:
    problem, request, grid = phase_a_case
    backend = NativeBackend(problem, request, grid=grid, fft_planning=_PLANNING)
    try:
        workspace = _workspace(backend, problem, request)
        first_start = _run_start(backend, problem, request)
        second_start = _run_start(backend, problem, request)
    finally:
        backend.close()
    n = int(np.prod(problem.shape))
    session = workspace.start(first_start)
    # The launch's batch, taken by hand: its arrays are kept past everything below.
    batch = session.advance(1, 3600.0)
    attempts = batch.attempts
    kept = np.array(attempts, copy=True)
    out = np.empty(n, dtype=np.complex128)
    recipe = {"power": False, "optics": False, "weight": None}
    session.export((batch.sequence, batch.event.generation), recipe, out)
    session.acknowledge(batch.sequence)
    # A second start is refused while the session holds the workspace.
    with pytest.raises(RuntimeError, match="leased"):
        workspace.start(second_start)
    with pytest.raises(RuntimeError, match="leased"):
        workspace.probe(second_start)
    first, _ = _drive(session, n)
    first.insert(0, (float(batch.event.z), out))
    session.close()
    again = workspace.start(second_start)
    second, _ = _drive(again, n)
    again.close()
    assert len(first) == len(second)
    for (_, one), (_, other) in zip(first, second, strict=True):
        np.testing.assert_array_equal(one, other)
    workspace.close()
    assert workspace.closed
    # A batch's arrays own their storage: valid after acknowledgment, reuse and close.
    np.testing.assert_array_equal(attempts, kept)
    with pytest.raises(RuntimeError):
        workspace.start(second_start)
    with pytest.raises(_core.SessionLifecycleError):
        again.advance(1, 1.0)


def test_diagnostics_hold_what_the_metadata_reads(phase_a_case: PhaseACase) -> None:
    problem, request, grid = phase_a_case
    backend = NativeBackend(problem, request, grid=grid, fft_planning=_PLANNING)
    try:
        workspace = _workspace(backend, problem, request)
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    try:
        diagnostics = workspace.diagnostics()
        assert diagnostics["method"] == "runge_kutta"
        assert set(diagnostics["memory_ledger"]) >= {"state"}
        assert diagnostics["memory_ledger"]["state"] == ledger["state"]
        assert set(diagnostics["crossed_bytes"]) is not None
        for key in ("cpu_execution", "fft_planning", "execution_sizing"):
            assert diagnostics[key]
    finally:
        workspace.close()


# -- the probe ------------------------------------------------------------------------


def test_the_probe_repeats_its_trials_and_reads_back_an_owning_copy(
    phase_a_case: PhaseACase,
) -> None:
    problem, request, grid = phase_a_case
    dz = float(request.stepper.dz_init)
    backend = NativeBackend(problem, request, grid=grid, fft_planning=_PLANNING)
    try:
        old = backend.timing_probe(Start.entrance(problem))
        old.trial(0.0, dz)
        old_field = old.field()
        workspace = _workspace(backend, problem, request)
        start = _run_start(backend, problem, request)
    finally:
        backend.close()
    try:
        probe = workspace.probe(start)
        first = probe.trial(0.0, dz)
        # The same trial from the same loaded state, held to a few roundings of the
        # norm (SEP 25), which on one build is its bits.
        assert probe.trial(0.0, dz) == pytest.approx(first, rel=64 * EPS)
        field = probe.field()
        assert field.flags.owndata or field.base is not None
        # The same trial and finish as the old probe's, on the same build.
        np.testing.assert_array_equal(field.reshape(problem.shape), old_field)
        field[...] = 0.0  # an owning copy: nothing native reads it
        assert probe.trial(0.0, dz) == pytest.approx(first, rel=64 * EPS)
        assert probe.trials == 3
        # Refused while the probe holds the workspace.
        with pytest.raises(RuntimeError, match="leased"):
            workspace.start(start)
        probe.close()
        assert probe.closed
        assert not workspace.closed
        session = workspace.start(start)
        with pytest.raises(RuntimeError, match="leased"):
            workspace.probe(start)
        session.close()
    finally:
        workspace.close()


def test_the_low_storage_probe_steps_in_place_and_closes_its_workspace(
    phase_a_case: PhaseACase,
) -> None:
    problem, request, grid = _case(phase_a_case, "low_storage")
    backend = NativeBackend(problem, request, grid=grid, fft_planning=_PLANNING)
    try:
        workspace = _workspace(backend, problem, request)
        start = _run_start(backend, problem, request)
    finally:
        backend.close()
    schedule = request.stepper.schedule
    assert schedule is not None
    probe = workspace.probe(start)
    assert probe.low_storage
    assert probe.trial(0.0, float(schedule.steps[0])) == 0.0
    # Four evaluations a step: the timing a probe of this march measures.
    assert probe.evaluations == LowStorageRungeKutta().evaluations
    with pytest.raises(_core.BackendRefusal, match="no trial field to read back"):
        probe.field()
    probe.close()
    assert workspace.closed


# -- the engines and the test hooks ---------------------------------------------------


def test_the_engines_take_the_probe_s_trial_through_solprop_s_protocol(
    phase_a_case: PhaseACase,
) -> None:
    problem, request, grid = phase_a_case
    dz = float(request.stepper.dz_init)
    backend = NativeBackend(problem, request, grid=grid, fft_planning=_PLANNING)
    try:
        workspace = _workspace(backend, problem, request)
        start = _run_start(backend, problem, request)
    finally:
        backend.close()
    try:
        probe = workspace.probe(start)
        expected = probe.trial(0.0, dz)
        probe.close()
        engines = _testing.engines(workspace)
        # The engines hold the workspace as a session does.
        with pytest.raises(RuntimeError, match="leased"):
            workspace.start(start)
        loaded = engines.field()
        assert loaded.shape == (1, problem.shape[-1])
        assert engines.rk_trial(0.0, dz) == pytest.approx(expected, rel=64 * EPS)
        assert not engines.rk_in_flight
        stage = engines.stage(0)
        assert stage.shape == loaded.shape
        record = engines.rk_accept(dz)
        assert isinstance(record, protocol.AcceptRecord)
        assert math.isnan(record.stiffness)
        # The committed state moved; its reading is the protocol's record.
        assert not np.array_equal(engines.field(), loaded)
        with pytest.raises(RuntimeError, match=r"no boundary readings|monitor"):
            engines.rk_reading()
        engines.close()
        session = workspace.start(start)
        session.close()
    finally:
        workspace.close()


def test_an_injected_fault_fails_the_run_at_its_stage(phase_a_case: PhaseACase) -> None:
    problem, request, grid = _case(phase_a_case, "low_storage")
    backend = NativeBackend(problem, request, grid=grid, fft_planning=_PLANNING)
    try:
        workspace = _workspace(backend, problem, request)
        start = _run_start(backend, problem, request)
    finally:
        backend.close()
    try:
        with pytest.raises(ValueError, match="a march fault's stage"):
            _testing.inject_fault(workspace, "finish", 0, "value")
        # The second step's trial fails, after the first committed.
        _testing.inject_fault(workspace, "trial", 1, "value")
        session = workspace.start(start)
        _, terminal = _drive(session, int(np.prod(problem.shape)))
        session.close()
        assert terminal.outcome == 3  # failed
        assert "injected" in terminal.message
    finally:
        workspace.close()
