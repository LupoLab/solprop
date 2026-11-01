# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve terminal states and accounting independently of regular observations."""

import dataclasses

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from solprop.exceptions import OutputError
from solprop.output import (
    HDF5Output,
    MemoryOutput,
    RunMetadata,
    RunOutcome,
    SavedPlane,
    StepHistory,
    TerminalCheckpoint,
    open_file,
    resume,
)
from solprop.output.schema import (
    StepStats,
    _example_metadata,
    _prepend_checkpoint_history,
)


@pytest.fixture
def metadata():
    return _example_metadata()


def plane(metadata, z, *, history=False):
    field = np.full((1, 1, metadata.grid.n_omega), 1 + 2j)
    return SavedPlane(
        z=z,
        field=field,
        step_history=StepHistory(
            z=np.array([z - 0.125]),
            h=np.array([0.125]),
            norm=np.array([0.25]),
            accepted=np.array([True]),
            wall_time=np.array([0.01]),
        )
        if history
        else StepHistory.empty(),
        step_stats=StepStats(
            z=np.array([z]),
            step=np.array([0]),
            values={"energy": np.ones((1, 1, 1))},
        )
        if history
        else StepStats.empty(),
    )


def test_memory_checkpoint_round_trip_preserves_counts_and_fft(metadata, tmp_path):
    memory = MemoryOutput()
    memory.open(metadata)
    memory.save(plane(metadata, 0.0))
    checkpoint = TerminalCheckpoint(plane(metadata, 0.25, history=True), 0)
    outcome = RunOutcome(
        "budget_exhausted",
        0.25,
        "atomic cap reached",
        logical_calls={"startup": 1, "accepted": 2},
        atomic_calls={"attempted": 6, "reserved": 0, "failed_calls": 1},
    )
    memory.record_outcome(outcome, checkpoint)
    memory.close()
    assert memory.n_saved == 1
    assert_array_equal(memory["z"], [0.0])
    path = tmp_path / "terminal.h5"
    memory.to_file(path)
    with open_file(path) as stored:
        assert stored.n_saved == 1
        assert stored.outcome == outcome
        assert stored.terminal_checkpoint is not None
        assert_array_equal(
            stored.terminal_checkpoint.plane.field, checkpoint.plane.field
        )
    state = resume(path)
    assert state.n_saved == 1
    assert state.plane.z == 0.25
    assert state.checkpoint is not None and state.checkpoint.next_save_index == 0
    assert state.history.n_steps == 1
    assert state.outcome == outcome


def test_checkpoint_history_moves_to_next_regular_save_once(metadata, tmp_path):
    path = tmp_path / "resume.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.save(plane(metadata, 0.0))
        handler.record_outcome(
            RunOutcome("failed", 0.25),
            TerminalCheckpoint(plane(metadata, 0.25, history=True), 0),
        )
    with HDF5Output(path, append=True) as handler:
        handler.open(metadata)
        handler.record_outcome(RunOutcome("running", 0.25))
        handler.save(plane(metadata, 0.5, history=True))
        handler.record_outcome(RunOutcome("complete", 0.5))
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is None
        assert stored.outcome is not None and stored.outcome.status == "complete"
        assert_array_equal(stored["z"], [0.0, 0.5])
        assert_array_equal(stored["step_history/counts"], [0, 2])
        assert_array_equal(stored["step_stats/step"], [0, 1])
        assert_array_equal(stored["step_history/z"], [0.125, 0.375])
    assert resume(path).history.n_steps == 2


def test_second_stop_retains_previous_unsaved_attempts(metadata, tmp_path):
    path = tmp_path / "repeated.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.save(plane(metadata, 0.0))
        handler.record_outcome(
            RunOutcome("failed", 0.25),
            TerminalCheckpoint(plane(metadata, 0.25, history=True), 0),
        )
    with HDF5Output(path, append=True) as handler:
        handler.open(metadata)
        handler.record_outcome(RunOutcome("running", 0.25))
        handler.record_outcome(
            RunOutcome("failed", 0.375),
            TerminalCheckpoint(plane(metadata, 0.375, history=True), 0),
        )
    state = resume(path)
    assert state.n_saved == 1 and state.history.n_steps == 2
    assert_array_equal(state.plane.step_stats.step, [0, 1])


def test_checkpoint_can_exist_before_first_regular_save(metadata, tmp_path):
    path = tmp_path / "entrance-failure.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.record_outcome(
            RunOutcome("failed", 0.0), TerminalCheckpoint(plane(metadata, 0.0), 0)
        )
    state = resume(path)
    assert state.n_saved == 0 and state.plane.z == 0.0
    assert state.checkpoint is not None


def test_scientific_launch_and_tdse_identity_control_resume(metadata):
    changed = dataclasses.replace(metadata, initial_z=0.1)
    assert changed.configuration_hash() != metadata.configuration_hash()
    atomic = dataclasses.replace(metadata, tdse={"identity": "model-one", "scale": 2.0})
    assert atomic.configuration_hash() != metadata.configuration_hash()
    assert (
        dataclasses.replace(
            atomic, tdse={"identity": "model-two", "scale": 2.0}
        ).configuration_hash()
        != atomic.configuration_hash()
    )
    assert (
        RunMetadata.from_json(atomic.to_json()).configuration_hash()
        == atomic.configuration_hash()
    )


def test_checkpoint_validation_preserves_interpolated_classification(
    metadata, tmp_path
):
    with pytest.raises(OutputError, match="nonnegative integer"):
        TerminalCheckpoint(plane(metadata, 0.0), -1)
    with pytest.raises(OutputError, match="finite"):
        TerminalCheckpoint(
            dataclasses.replace(
                plane(metadata, 0.0),
                field=np.full((1, 1, metadata.grid.n_omega), np.nan + 0j),
            ),
            0,
        )
    memory = MemoryOutput()
    memory.open(metadata)
    with pytest.raises(OutputError, match="positions must agree"):
        memory.record_outcome(
            RunOutcome("failed", 0.2), TerminalCheckpoint(plane(metadata, 0.1), 0)
        )
    memory.save(plane(metadata, 0.25))
    with pytest.raises(OutputError, match="precedes"):
        memory.record_outcome(
            RunOutcome("failed", 0.1), TerminalCheckpoint(plane(metadata, 0.1), 0)
        )
    path = tmp_path / "interpolated.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.record_outcome(
            RunOutcome("failed", 0.1),
            TerminalCheckpoint(plane(metadata, 0.1), 0, landed=False),
        )
    state = resume(path)
    assert state.checkpoint is not None and not state.checkpoint.landed
    assert state.n_saved == 0 and state.plane.z == 0.1
    assert state.outcome is not None and state.outcome.z == state.plane.z
    assert_array_equal(state.plane.field, plane(metadata, 0.1).field)
    with h5py.File(path, "r+") as file:
        file["terminal_checkpoint"].attrs["landed"] = True
        del file["meta/outcome"]
    with pytest.raises(OutputError, match="matching execution outcome"):
        resume(path)


def test_execution_records_own_json_values_and_keep_resume_history(metadata, tmp_path):
    facts = {"native": {"device": "cpu", "plans": [2, 1]}}
    outcome = RunOutcome("failed", 0.25, execution=facts)
    facts["native"]["plans"].append(99)
    native = outcome.execution["native"]
    assert isinstance(native, dict) and native["plans"] == [2, 1]
    for invalid in (np.array([1]), float("nan"), object()):
        with pytest.raises(OutputError, match="finite JSON"):
            RunOutcome("failed", 0.0, execution={"bad": invalid})
    path = tmp_path / "executions.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.save(plane(metadata, 0.0))
        handler.record_outcome(outcome, TerminalCheckpoint(plane(metadata, 0.25), 0))
    with HDF5Output(path, append=True) as handler:
        handler.open(metadata)
        handler.record_outcome(
            RunOutcome("running", 0.25, execution={"device": "cuda"})
        )
        handler.save(plane(metadata, 0.5))
        handler.record_outcome(
            RunOutcome("complete", 0.5, execution={"device": "cuda"})
        )
    with open_file(path) as stored:
        assert stored.outcome is not None
        assert stored.outcome.execution == {"device": "cuda"}
        assert len(stored.outcome.previous_executions) == 1
        assert stored.outcome.previous_executions[0]["execution"] == outcome.execution


@pytest.mark.parametrize("z", [-1.0, np.nan, np.inf])
def test_outcome_and_metadata_refuse_invalid_launch_positions(metadata, z):
    with pytest.raises(OutputError, match="finite and nonnegative"):
        RunOutcome("failed", z)
    with pytest.raises(OutputError, match="finite and nonnegative"):
        dataclasses.replace(metadata, initial_z=z)


def test_outcome_revalidates_unknown_status_without_test_only_literal_check():
    outcome = RunOutcome("failed", 0.0)
    object.__setattr__(outcome, "status", "unclassified")
    with pytest.raises(OutputError, match="unknown run outcome"):
        outcome.__post_init__()


def test_checkpoint_must_match_regular_field_layout(metadata):
    memory = MemoryOutput()
    memory.open(metadata)
    memory.save(plane(metadata, 0.0))
    changed = dataclasses.replace(
        plane(metadata, 0.1), field=np.ones((2, 1, metadata.grid.n_omega), complex)
    )
    with pytest.raises(OutputError, match="field shape differs"):
        memory.record_outcome(RunOutcome("failed", 0.1), TerminalCheckpoint(changed, 0))
    assert memory.outcome is None and memory.terminal_checkpoint is None


@pytest.mark.parametrize("before_has_rows", [True, False])
def test_checkpoint_periodic_rows_survive_when_only_one_side_has_rows(
    metadata, before_has_rows
):
    previous = plane(metadata, 0.25, history=before_has_rows)
    current = plane(metadata, 0.5, history=not before_has_rows)
    merged = _prepend_checkpoint_history(previous, current)
    assert merged.step_stats.n_rows == merged.step_history.n_steps == 1
    assert_array_equal(merged.step_stats.step, [0])
    assert_array_equal(merged.step_stats.values["energy"], np.ones((1, 1, 1)))
    assert_array_equal(merged.field, current.field)


def test_checkpoint_history_refuses_changed_periodic_statistic_names(metadata):
    previous = plane(metadata, 0.25, history=True)
    current = plane(metadata, 0.5, history=True)
    changed = dataclasses.replace(
        current,
        step_stats=dataclasses.replace(
            current.step_stats, values={"peak_power": np.ones((1, 1, 1))}
        ),
    )
    with pytest.raises(OutputError, match="statistic names changed"):
        _prepend_checkpoint_history(previous, changed)


def test_checkpoint_roundtrip_retains_available_statistics_and_noise(
    metadata, tmp_path
):
    original = plane(metadata, 0.25, history=True)
    assert original.field is not None
    saved = dataclasses.replace(
        original,
        noise_phase=np.full(original.field.shape, 0.75),
        stats={"energy": np.ones((1, 1))},
    )
    path = tmp_path / "available-diagnostics.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.record_outcome(
            RunOutcome("failed", saved.z), TerminalCheckpoint(saved, 0)
        )
    restored = resume(path).plane
    assert_array_equal(restored.noise_phase, saved.noise_phase)
    assert_array_equal(restored.stats["energy"], saved.stats["energy"])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("z", [0.1]),
        ("z", "0.1"),
        ("next_save_index", [0]),
        ("next_save_index", 0.0),
        ("landed", [True]),
        ("landed", 1),
    ],
)
def test_read_refuses_malformed_checkpoint_scalar_attributes(
    metadata, tmp_path, name, value
):
    path = tmp_path / "malformed-scalar.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.record_outcome(
            RunOutcome("failed", 0.1), TerminalCheckpoint(plane(metadata, 0.1), 0)
        )
    with h5py.File(path, "r+") as stored:
        stored["terminal_checkpoint"].attrs[name] = value
    with pytest.raises(OutputError, match="typed scalars"):
        resume(path)


def test_new_checkpoint_replaces_an_interrupted_staging_group(metadata, tmp_path):
    path = tmp_path / "interrupted-write.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
    with h5py.File(path, "r+") as stored:
        stored.create_group("_terminal_checkpoint_writing")
    with HDF5Output(path, append=True) as handler:
        handler.open(metadata)
        handler.record_outcome(
            RunOutcome("failed", 0.1), TerminalCheckpoint(plane(metadata, 0.1), 0)
        )
    assert resume(path).plane.z == 0.1
    with h5py.File(path) as stored:
        assert "_terminal_checkpoint_writing" not in stored


def test_resume_refuses_checkpoint_older_than_last_regular_save(metadata, tmp_path):
    path = tmp_path / "outdated-checkpoint.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.save(plane(metadata, 0.25))
        handler.record_outcome(
            RunOutcome("failed", 0.5), TerminalCheckpoint(plane(metadata, 0.5), 0)
        )
    with h5py.File(path, "r+") as stored:
        stored["terminal_checkpoint"].attrs["z"] = 0.1
    with pytest.raises(OutputError, match="precedes the last saved plane"):
        resume(path)


def test_storage_hook_can_clear_checkpoint_without_rewriting_outcome(
    metadata, tmp_path
):
    path = tmp_path / "clear-checkpoint.h5"
    outcome = RunOutcome("failed", 0.1)
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.save(plane(metadata, 0.0))
        handler.record_outcome(outcome, TerminalCheckpoint(plane(metadata, 0.1), 0))
        handler._write_outcome(None, None)
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is None
        assert stored.outcome == outcome
