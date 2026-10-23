# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Write and stream a TDSE response archive exactly (SEP 20 section 5.8, 7.1 21-22).

These tests drive the writer and reader directly with manufactured provider
records; the binding and run tests check what a run writes. Every comparison is
exact: the archive copies values, so a stored array equals its source bitwise,
including NaN and infinite samples, with its dtype and shape.
"""

import enum
import errno
import gc
import inspect
import json
import subprocess
import sys
import textwrap
import tracemalloc
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tests.tdse_archive_cases import (
    RECORDS,
    assert_record_equal,
    begin,
    hacc_full,
    sae_compact,
    sae_full,
    writer,
)

from solprop.exceptions import OutputError, TDSEArchiveError
from solprop.output import TDSE_ARCHIVE_SCHEMA, TDSEArchive, open_tdse_archive
from solprop.output.tdse_archive import (
    ARCHIVE_METADATA_CACHE_BYTES,
    HACC_GRID_RESPONSE,
    SAE_GRID_RESPONSE,
    SAE_TOTAL_CURRENT_RESPONSE,
    record_serializer,
    structured_json,
    writer_memory,
)


@pytest.fixture
def rng():
    return np.random.default_rng(20)


def _hdf5(path, mode="r") -> Any:
    """Open an HDF5 file for inspection, untyped as h5py's indexing is."""
    return h5py.File(path, mode)


# --- the configuration record -------------------------------------------------------


def test_the_request_adopts_a_path_and_validates_its_selection():
    archive = TDSEArchive("a.h5")
    assert archive.path == Path("a.h5") and archive.outputs == "full"
    assert TDSEArchive(Path("b.h5"), outputs="required").outputs == "required"
    # Under the test session's type hook the annotations refuse first; the explicit
    # checks are what a production caller meets.
    init = inspect.unwrap(TDSEArchive.__init__)
    blank = object.__new__(TDSEArchive)
    with pytest.raises(TypeError, match="path or a string"):
        init(blank, 3)
    with pytest.raises(ValueError, match="empty"):
        TDSEArchive("")
    with pytest.raises(ValueError, match="'full' or 'required'"):
        init(blank, "a.h5", outputs="all")
    with pytest.raises(TypeError):
        TDSEArchive("a.h5", "full")  # pyright: ignore[reportCallIssue] - keyword only
    assert issubclass(TDSEArchiveError, OutputError)


def test_each_model_and_selection_has_one_versioned_serializer():
    assert record_serializer("sae", "full") is SAE_GRID_RESPONSE
    assert record_serializer("sae", "total_current") is SAE_TOTAL_CURRENT_RESPONSE
    assert record_serializer("hacc", "full") is HACC_GRID_RESPONSE
    assert HACC_GRID_RESPONSE.serializer == "solprop.tdse-archive.hacc.grid_response.v1"
    with pytest.raises(TDSEArchiveError, match="no TDSE archive serializer"):
        record_serializer("argon", "full")
    # Units and order labels travel with the fields (C4 for haCC).
    fields = {item.name: item for item in HACC_GRID_RESPONSE.fields}
    assert fields["outside"].unit == "1" and fields["total_current"].unit == "C m/s"
    assert fields["neglected_current_bound"].optional
    shadow = fields["shadow"].record
    assert shadow is not None and shadow.fields[0].kind == "labels"


def test_structured_records_become_valid_json_whatever_they_hold():
    class Colour(enum.Enum):
        RED = "red"

    class Opaque:
        def __repr__(self):
            return "Opaque()"

    text = structured_json(
        {
            "nan": float("nan"),
            "inf": np.float64(np.inf),
            "array": np.array([1, 2], dtype=np.int64),
            "enum": Colour.RED,
            "namespace": SimpleNamespace(a=(1, 2.5)),
            "opaque": Opaque(),
            "flag": np.bool_(True),
            1: None,
        }
    )
    assert json.loads(text) == {
        "1": None,
        "array": [1, 2],
        "enum": "red",
        "flag": True,
        "inf": "nonfinite:inf",
        "namespace": {"a": [1, 2.5]},
        "nan": "nonfinite:nan",
        "opaque": {"repr": "Opaque()", "type": Opaque.__qualname__},
    }


def test_the_writer_memory_is_the_capped_cache_and_one_copy():
    assert writer_memory(rows=4, samples=100) == {
        "archive.metadata_cache": ARCHIVE_METADATA_CACHE_BYTES,
        "archive.conversion_buffer": 8 * 3 * 4 * 100,
    }


# --- round trips --------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_every_record_type_reads_back_exactly(tmp_path, rng, name):
    build, schema = RECORDS[name]
    raw = build(rng)
    # Raw invalid and non-finite samples are stored as returned: the source would
    # refuse them, the archive keeps them.
    current = raw.plasma if name == "sae_full" else raw
    current.total_current[0, -1] = np.nan
    current.total_current[1, 0] = -np.inf
    raw.valid[0, -1] = False
    archive = writer(tmp_path / "a.h5")
    identifier, prewindow, drives = begin(archive, rng, chunks=2)
    archive.write_result(identifier, 0, raw, schema)
    second = build(rng)
    archive.write_result(identifier, 1, second, schema)
    archive.finish_evaluation(identifier, "returned")
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        assert reader.schema == TDSE_ARCHIVE_SCHEMA and reader.run_uuid == "run-1"
        (evaluation,) = reader.evaluations()
        assert evaluation.complete and not reader.truncated
        assert evaluation.outcome == "returned" and evaluation.error is None
        assert [chunk.serializer for chunk in evaluation.chunks] == [
            schema.serializer
        ] * 2
        assert_record_equal(raw, reader.raw(evaluation, 0), schema)
        assert_record_equal(second, reader.raw(evaluation, 1), schema)
        assert_array_equal(reader.load(evaluation, "prewindow_field"), prewindow)
        for index, (start, stop, drive) in enumerate(drives):
            stored = reader.load(evaluation, "drive", chunk=index)
            assert stored.dtype == np.float64
            assert_array_equal(stored, drive, strict=True)
            assert evaluation.chunks[index].row_start == start
            assert evaluation.chunks[index].row_stop == stop
        path = "plasma/total_current" if name == "sae_full" else "total_current"
        total = reader.load(evaluation, path, chunk=0)
        assert_array_equal(total, current.total_current, strict=True)
        assert np.isnan(total[0, -1]) and total[1, 0] == -np.inf


def test_absent_fields_are_marked_and_present_ones_keep_their_dtypes(tmp_path, rng):
    raw = hacc_full(rng, closure=False)
    archive = writer(tmp_path / "a.h5")
    identifier, _, _ = begin(archive, rng)
    archive.write_result(identifier, 0, raw, HACC_GRID_RESPONSE)
    archive.finish_evaluation(identifier, "returned")
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        (evaluation,) = reader.evaluations()
        assert evaluation.chunks[0].absent == ("neglected_current_bound",)
        stored: Any = reader.raw(evaluation, 0)
        assert stored["neglected_current_bound"] is None
        assert stored["shadow"]["total_current"] is None
        assert stored["monitors"]["non_finite"].dtype == np.bool_
        assert stored["monitors"]["monitored_step"].dtype == np.int64
        assert stored["valid"].dtype == np.bool_
        assert stored["shadow"]["orders"] == (1,)
        with _hdf5(tmp_path / "a.h5") as file:
            group = file["sessions/session_000000/evaluations/evaluation_000000000"]
            dataset = group["chunk_000/raw/outside"]
            assert (
                dataset.attrs["unit"] == "1" and dataset.attrs["axes"] == "n_row n_time"
            )
            assert group["chunk_000/raw/shadow/current"].attrs["axes"] == (
                "n_row n_order n_time"
            )


def test_labels_and_shared_values_are_stored_once_and_read_back(tmp_path, rng):
    archive = writer(tmp_path / "a.h5")
    for _ in range(3):
        identifier, _, _ = begin(archive, rng, trial_id=None, density=None)
        archive.write_result(
            identifier, 0, sae_compact(rng), SAE_TOTAL_CURRENT_RESPONSE
        )
        archive.finish_evaluation(identifier, "returned")
    archive.close()
    with _hdf5(tmp_path / "a.h5") as file:
        session = file["sessions/session_000000"]
        # One gate, one time axis, one binding record and one info record.
        assert len(session["shared"]) == 4
        evaluation = session["evaluations/evaluation_000000002"]
        shared = [session["shared"][name] for name in session["shared"]]
        # Hard links: the time axis and the gate are the shared datasets themselves.
        assert any(evaluation["chunk_000/raw/time"] == item for item in shared)
        assert any(evaluation["gate"] == item for item in shared)
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        evaluations = list(reader.evaluations())
        assert [item.evaluation_id for item in evaluations] == [0, 1, 2]
        assert all(item.trial is None and item.density is None for item in evaluations)
        assert evaluations[0].z == 0.25 and evaluations[0].role == "predictor"
        assert json.loads(str(evaluations[0].labels["absent_labels"])) == [
            "density",
            "trial_id",
        ]
        gate = reader.load(evaluations[1], "gate")
        assert_array_equal(gate, np.linspace(0.0, 1.0, 16), strict=True)
        times = [reader.load(item, "time", chunk=0) for item in evaluations]
        assert all(np.array_equal(times[0], value) for value in times)


def test_a_session_records_its_totals_and_reference(tmp_path, rng):
    optical = tmp_path / "out" / "run.h5"
    optical.parent.mkdir()
    (tmp_path / "side").mkdir()
    archive = writer(tmp_path / "side" / "a.h5", optical=optical)
    identifier, _, _ = begin(archive, rng)
    archive.write_result(identifier, 0, sae_full(rng), SAE_GRID_RESPONSE)
    archive.finish_evaluation(identifier, "returned")
    archive.record_trial(0, accepted=True, evaluations=[identifier])
    reference: Any = archive.reference()
    assert reference["path"] == "../side/a.h5"
    assert reference["relative_to"] == "optical_file"
    assert reference["schema"] == TDSE_ARCHIVE_SCHEMA
    assert reference["evaluations"] == 1 and reference["bytes_written"] > 0
    assert reference["array_bytes"] > 0 and reference["write_seconds"] >= 0.0
    archive.close()
    archive.close()
    with open_tdse_archive(tmp_path / "side" / "a.h5") as reader:
        (session,) = reader.sessions
        assert session.closed and session.kind == "run"
        assert session.optical_reference == {
            "path": "../side/a.h5",
            "relative_to": "optical_file",
        }
        assert session.context == {"test": True}
        assert session.trials == {0: "accepted"}
        totals: Any = session.totals
        assert totals["evaluations"] == 1
        assert totals["bytes_written"] > 0
    # Without an optical file the reference is the absolute path.
    alone = writer(tmp_path / "b.h5")
    assert alone.reference()["relative_to"] is None
    assert alone.reference()["path"] == (tmp_path / "b.h5").resolve().as_posix()
    alone.close()


def test_a_rejected_trial_relabels_its_predictor_and_corrector(tmp_path, rng):
    archive = writer(tmp_path / "a.h5")
    ids = []
    for role, trial in (
        ("startup", 0),
        ("predictor", 0),
        ("predictor", 1),
        ("corrector", 2),
        ("predictor", 3),
    ):
        identifier, _, _ = begin(archive, rng, role=role, trial_id=trial)
        archive.write_result(
            identifier, 0, sae_compact(rng), SAE_TOTAL_CURRENT_RESPONSE
        )
        archive.finish_evaluation(identifier, "returned")
        ids.append(identifier)
    archive.record_trial(0, accepted=False, evaluations=ids[:2])
    archive.record_trial(1, accepted=True, evaluations=ids[2:3])
    archive.record_trial(2, accepted=False, evaluations=ids[3:4])
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        roles = [
            (item.role, item.evaluated_role, item.trial_outcome)
            for item in reader.evaluations()
        ]
    assert roles == [
        ("startup", "startup", "rejected"),
        ("rejected", "predictor", "rejected"),
        ("predictor", "predictor", "accepted"),
        ("rejected", "corrector", "rejected"),
        ("predictor", "predictor", None),
    ]


# --- sessions, resumes and refusals -------------------------------------------------


def test_a_resume_appends_a_session_and_continues_the_identifiers(tmp_path, rng):
    path = tmp_path / "a.h5"
    first = writer(path)
    begin(archive=first, rng=rng)
    first.close()
    second = writer(path, kind="resume", resume=True)
    identifier, _, _ = begin(second, rng)
    assert identifier == 1
    second.close()
    with open_tdse_archive(path) as reader:
        assert [session.kind for session in reader.sessions] == ["run", "resume"]
        assert {session.run_uuid for session in reader.sessions} == {"run-1"}
        assert [item.session for item in reader.evaluations()] == [0, 1]


@pytest.mark.parametrize(
    "case", ["text", "hdf5", "other_run", "not_a_resume", "directory"]
)
def test_an_existing_unrelated_file_is_refused_and_left_unchanged(tmp_path, case):
    path = tmp_path / "a.h5"
    if case == "text":
        path.write_text("not an archive", encoding="utf-8")
    elif case == "hdf5":
        with h5py.File(path, "w") as file:
            file["x"] = 1
    elif case == "directory":
        path.mkdir()
    else:
        writer(path, run_uuid="other").close()
    before = None if case == "directory" else path.read_bytes()
    with pytest.raises(TDSEArchiveError, match="never overwritten"):
        writer(path, resume=case != "not_a_resume")
    if before is not None:
        assert path.read_bytes() == before


def test_a_session_missing_its_groups_is_read_and_continued(tmp_path, rng):
    path = tmp_path / "a.h5"
    first = writer(path)
    begin(first, rng)
    first.close()
    with _hdf5(path, "r+") as file:
        del file["sessions/session_000000/evaluations"]
        del file["sessions/session_000000/trials"]
    second = writer(path, kind="resume", resume=True)
    # The damaged session holds no evaluation, so numbering starts again at zero.
    identifier, _, _ = begin(second, rng)
    assert identifier == 0
    second.close()
    with open_tdse_archive(path) as reader:
        assert [item.session for item in reader.evaluations()] == [1]
        assert reader.sessions[0].trials == {}


def test_an_uncreatable_archive_is_refused(tmp_path):
    with pytest.raises(TDSEArchiveError, match="cannot open the TDSE archive"):
        writer(tmp_path / "missing" / "a.h5")


# --- failures -----------------------------------------------------------------------


def test_a_raised_evaluation_keeps_its_input_and_exception(tmp_path, rng):
    archive = writer(tmp_path / "a.h5")
    identifier, prewindow, _ = begin(archive, rng, chunks=2)
    archive.write_result(identifier, 0, sae_compact(rng), SAE_TOTAL_CURRENT_RESPONSE)
    archive.write_failure(
        identifier,
        chunk=1,
        error=RuntimeError("native failure"),
        classification="native",
    )
    archive.finish_evaluation(identifier, "raised")
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        (evaluation,) = reader.evaluations()
        assert evaluation.complete and evaluation.outcome == "raised"
        assert evaluation.error == {
            "chunk": 1,
            "classification": "native",
            "message": "native failure",
            "type": "RuntimeError",
        }
        assert [chunk.serializer is None for chunk in evaluation.chunks] == [
            False,
            True,
        ]
        assert_array_equal(reader.load(evaluation, "prewindow_field"), prewindow)
        with pytest.raises(TDSEArchiveError, match="has no record"):
            reader.raw(evaluation, 1)
        with pytest.raises(TDSEArchiveError, match="stores no array"):
            reader.load(evaluation, "missing", chunk=0)


def _missing_field(raw):
    del raw.cut_term
    return raw


def _required_none(raw):
    raw.plasma.total_current = None
    return raw


def _object_array(raw):
    raw.current = np.array([object()] * 3, dtype=object)
    return raw


def _not_an_array(raw):
    raw.outside = [0.0, 1.0]
    return raw


def _bad_labels(raw):
    raw.shadow.orders = (1, True)
    return raw


def _bad_outcomes(raw):
    raw.outcomes = (SimpleNamespace(failure="none"),)
    return raw


def _bad_outcome_container(raw):
    raw.outcomes = "none"
    return raw


def _bad_scalar(raw):
    raw.cut_term = "small"
    return raw


@pytest.mark.parametrize(
    "malform",
    [
        _missing_field,
        _required_none,
        _object_array,
        _not_an_array,
        _bad_labels,
        _bad_outcomes,
        _bad_outcome_container,
        _bad_scalar,
    ],
)
def test_a_record_its_serializer_cannot_store_fails_the_writer(tmp_path, rng, malform):
    archive = writer(tmp_path / "a.h5")
    identifier, _, _ = begin(archive, rng)
    with pytest.raises(TDSEArchiveError, match="cannot serialize raw"):
        archive.write_result(identifier, 0, malform(sae_full(rng)), SAE_GRID_RESPONSE)
    assert archive.failed and archive.reference()["failed"] is True
    # Every later write, and the check before each provider call, now refuses.
    with pytest.raises(TDSEArchiveError, match="failed earlier"):
        archive.ensure_writable()
    with pytest.raises(TDSEArchiveError, match="failed earlier"):
        archive.finish_evaluation(identifier, "returned")
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        (evaluation,) = reader.evaluations()
        assert not evaluation.complete and reader.truncated
        assert not reader.sessions[0].closed


def test_a_full_disk_is_named_and_stops_the_writer(tmp_path, rng, monkeypatch):
    archive = writer(tmp_path / "a.h5")
    identifier, _, _ = begin(archive, rng)

    def full(self, name, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(h5py.Group, "create_dataset", full)
    with pytest.raises(TDSEArchiveError, match="the disk is full") as error:
        archive.write_result(
            identifier, 0, sae_compact(rng), SAE_TOTAL_CURRENT_RESPONSE
        )
    assert isinstance(error.value.__cause__, OSError)
    for write in (
        lambda: archive.record_trial(0, accepted=True, evaluations=[]),
        lambda: archive.write_failure(
            identifier, chunk=0, error=ValueError("x"), classification="contract"
        ),
        lambda: begin(archive, rng),
    ):
        with pytest.raises(TDSEArchiveError, match="failed earlier"):
            write()
    monkeypatch.undo()
    archive.close()


@pytest.mark.parametrize("step", ["begin", "failure", "outcome", "trial", "close"])
def test_each_write_reports_its_own_failure(tmp_path, rng, monkeypatch, step):
    archive = writer(tmp_path / "a.h5")
    identifier = 0
    if step != "begin":
        identifier, _, _ = begin(archive, rng)

    def broken(*args, **kwargs):
        raise OSError("synthetic write failure")

    if step == "begin":
        monkeypatch.setattr(h5py.Group, "create_group", broken)
        with pytest.raises(TDSEArchiveError, match="evaluation 0"):
            begin(archive, rng)
    elif step == "failure":
        monkeypatch.setattr(h5py.Group, "create_group", broken)
        with pytest.raises(TDSEArchiveError, match="failure"):
            archive.write_failure(
                identifier, chunk=None, error=ValueError("x"), classification="contract"
            )
    elif step == "outcome":
        monkeypatch.setattr(h5py.File, "flush", broken)
        with pytest.raises(TDSEArchiveError, match="outcome"):
            archive.finish_evaluation(identifier, "returned")
    elif step == "trial":
        monkeypatch.setattr(h5py.Group, "create_group", broken)
        with pytest.raises(TDSEArchiveError, match="trial 0"):
            archive.record_trial(0, accepted=False, evaluations=[identifier])
    else:
        monkeypatch.setattr(h5py.File, "flush", broken)
        with pytest.raises(TDSEArchiveError, match="session totals"):
            archive.close()
    monkeypatch.undo()
    archive.close()
    assert archive.failed


def test_an_evaluation_not_begun_here_is_refused(tmp_path, rng):
    archive = writer(tmp_path / "a.h5")
    with pytest.raises(TDSEArchiveError, match="not begun"):
        archive.finish_evaluation(7, "returned")
    # A misuse stops the writer like a failed write: nothing follows it.
    with pytest.raises(TDSEArchiveError, match="failed earlier"):
        archive.ensure_writable()
    archive.close()
    closed = writer(tmp_path / "b.h5")
    closed.close()
    with pytest.raises(TDSEArchiveError, match="is closed"):
        closed.ensure_writable()


def test_a_session_that_cannot_start_is_refused(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("synthetic")

    monkeypatch.setattr(h5py.Group, "create_dataset", broken)
    with pytest.raises(TDSEArchiveError, match="cannot start a session"):
        writer(tmp_path / "a.h5")


# --- truncation ---------------------------------------------------------------------


def test_an_incomplete_last_group_is_flagged_not_raised(tmp_path, rng):
    archive = writer(tmp_path / "a.h5")
    done, _, _ = begin(archive, rng)
    archive.write_result(done, 0, sae_compact(rng), SAE_TOTAL_CURRENT_RESPONSE)
    archive.finish_evaluation(done, "returned")
    begin(archive, rng, chunks=2)
    # The second evaluation never ends: as after a crash, it has no outcome.
    archive.close()
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        first, last = reader.evaluations()
        assert first.complete and not last.complete and last.outcome is None
        assert reader.truncated


def test_unreadable_and_partial_groups_are_flagged(tmp_path, rng):
    archive = writer(tmp_path / "a.h5")
    for _ in range(3):
        identifier, _, _ = begin(archive, rng, chunks=2)
        archive.write_result(
            identifier, 0, sae_compact(rng), SAE_TOTAL_CURRENT_RESPONSE
        )
        archive.write_result(
            identifier, 1, sae_compact(rng), SAE_TOTAL_CURRENT_RESPONSE
        )
        archive.finish_evaluation(identifier, "returned")
    archive.record_trial(0, accepted=True, evaluations=[0])
    archive.close()
    with _hdf5(tmp_path / "a.h5", "r+") as file:
        session = file["sessions/session_000000"]
        evaluations = session["evaluations"]
        del evaluations["evaluation_000000001/chunk_001"]
        # A group whose labels cannot be read at all.
        evaluations["evaluation_000000002"].attrs["n_chunks"] = "garbled"
        session["trials/trial_000000000"].attrs.__delitem__("outcome")
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        good, partial, broken = reader.evaluations()
        assert good.complete and good.trial_outcome is None
        assert not partial.complete and len(partial.chunks) == 1
        assert not broken.complete and broken.evaluated_role == "unreadable"
        assert reader.sessions[0].trials == {}


def test_a_crash_leaves_every_flushed_evaluation_readable(tmp_path):
    path = tmp_path / "crash.h5"
    script = textwrap.dedent(
        f"""
        import os
        import numpy as np
        from tests.tdse_archive_cases import begin, sae_compact, writer
        from solprop.output.tdse_archive import SAE_TOTAL_CURRENT_RESPONSE as SCHEMA
        rng = np.random.default_rng(1)
        archive = writer({str(path)!r})
        for _ in range(2):
            identifier, _, _ = begin(archive, rng, chunks=2)
            for chunk in (0, 1):
                archive.write_result(identifier, chunk, sae_compact(rng), SCHEMA)
            archive.finish_evaluation(identifier, "returned")
        identifier, _, _ = begin(archive, rng, chunks=2)
        archive.write_result(identifier, 0, sae_compact(rng), SCHEMA)
        os._exit(3)
        """
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=root, check=False, capture_output=True
    )
    assert result.returncode == 3, result.stderr.decode()
    with open_tdse_archive(path) as reader:
        evaluations = list(reader.evaluations())
        assert [item.complete for item in evaluations] == [True, True, False]
        assert reader.truncated and not reader.sessions[0].closed
        assert str(reader.raw(evaluations[2], 0)["serializer"]).endswith("v1")


# --- the reader's refusals and bounded memory ----------------------------------------


def test_the_reader_refuses_what_is_not_an_archive(tmp_path):
    with pytest.raises(TDSEArchiveError, match="cannot read"):
        open_tdse_archive(tmp_path / "missing.h5")
    with h5py.File(tmp_path / "other.h5", "w") as file:
        file.attrs["schema"] = "something-else"
    with pytest.raises(TDSEArchiveError, match="is not a"):
        open_tdse_archive(tmp_path / "other.h5")


def test_the_writer_keeps_no_array_and_its_memory_stays_bounded(tmp_path, rng):
    archive = writer(tmp_path / "a.h5")
    rows, samples = 2, 2000
    raw = sae_full(rng, rows=rows, samples=samples)
    witness = weakref.ref(raw.plasma.total_current)
    identifier, _, _ = begin(archive, rng, rows=rows, samples=samples)
    archive.write_result(identifier, 0, raw, SAE_GRID_RESPONSE)
    archive.finish_evaluation(identifier, "returned")
    del raw
    gc.collect()
    assert witness() is None
    record_bytes = 40 * rows * samples * 8
    tracemalloc.start()
    try:
        for _ in range(3):
            identifier, _, _ = begin(archive, rng, rows=rows, samples=samples)
            archive.write_result(
                identifier,
                0,
                sae_full(rng, rows=rows, samples=samples),
                SAE_GRID_RESPONSE,
            )
            archive.finish_evaluation(identifier, "returned")
        gc.collect()
        settled, _ = tracemalloc.get_traced_memory()
        for _ in range(10):
            identifier, _, _ = begin(archive, rng, rows=rows, samples=samples)
            archive.write_result(
                identifier,
                0,
                sae_full(rng, rows=rows, samples=samples),
                SAE_GRID_RESPONSE,
            )
            archive.finish_evaluation(identifier, "returned")
        gc.collect()
        current, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    archive.close()
    # Ten more records of about 1.3 MB each: retaining even one would add a full
    # record. The bound is half of one, far above the per-group bookkeeping.
    assert current - settled < record_bytes / 2
