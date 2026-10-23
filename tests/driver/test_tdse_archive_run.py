# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Archive a haCC run's provider evaluations end to end (SEP 20 7.1 items 21-22).

The run is the natural-output fixture of item 20: a manufactured adaptive response
on a capillary with a pressure step, which rejects at least one trial, through the
real binding and the fake provider. Archiving must leave the march, the saved
fields and both call ledgers unchanged: within each run the ledgers count exactly the
evaluations made and archived, and across the adaptive runs the work agrees to a band
and the field to an accuracy envelope (SEP 25 section 4.4); the archive must label
every evaluation and trial, survive a resume as a new session of the same run, refuse
an unrelated file and stop the run at its last checkpoint when a write fails.
"""

import dataclasses
import inspect
import warnings
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tests.driver.test_tdse_natural_output import _simulation as natural_simulation
from tests.driver.test_tdse_natural_output import (
    assert_no_extra_call,
    assert_same_work,
    reference_field,
)
from tests.hacc_cases import alpha_law, install, plain, preset_basis

from march_bounds import assert_in_envelope
from solprop.driver import Simulation, prop_capillary
from solprop.driver import tdse as runtime
from solprop.exceptions import (
    BackendError,
    DriverError,
    GasFlowWarning,
    MaterialRangeWarning,
    ResponseBudgetError,
    TDSEArchiveError,
    TDSEDomainWarning,
)
from solprop.nonlinear.tdse import TDSEExecution
from solprop.output import (
    TDSE_ARCHIVE_SCHEMA,
    MemoryOutput,
    TDSEArchive,
    open_file,
    open_tdse_archive,
    resume,
)
from solprop.output import tdse_archive as archive_module
from solprop.stats import StatsSpec

pytestmark = pytest.mark.medium


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def fake(monkeypatch):
    fake = install(monkeypatch)
    fake.response = alpha_law(1.0, lambda5=2.0**-200)
    return fake


def _run(simulation, output=None, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEDomainWarning)
        return simulation.run(output, progress="none", **kwargs)


def _limited(limit):
    simulation = natural_simulation()
    simulation.settings = dataclasses.replace(
        simulation.settings,
        tdse_execution=TDSEExecution(max_atomic_calls=limit),
        resume_history=False,
    )
    return simulation


# --- item 21: the march is unchanged and every evaluation is labelled ----------------


def test_archiving_changes_neither_the_march_nor_the_ledgers(
    fake, monkeypatch, tmp_path
):
    runs = {}
    for name in ("none", "required", "full"):
        monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / f"cache-{name}"))
        before = len(fake.calls)
        archive = (
            None
            if name == "none"
            else TDSEArchive(tmp_path / f"{name}.h5", outputs=name)
        )
        output = _run(natural_simulation(), tdse_archive=archive)
        runs[name] = (output, fake.calls[before:])
    reference, calls = runs["none"]
    assert reference.outcome is not None
    assert plain(reference.outcome.logical_calls)["rejected_steps"] >= 1
    assert_no_extra_call(reference.outcome)
    assert len(calls) == plain(reference.outcome.atomic_calls)["attempted"]
    converged = reference_field(monkeypatch, tmp_path)
    for name in ("required", "full"):
        output, archived_calls = runs[name]
        assert output.outcome is not None
        logical = plain(output.outcome.logical_calls)
        atomic = plain(output.outcome.atomic_calls)
        # Within the run: one evaluation per attempted step, each made of the provider.
        assert_no_extra_call(output.outcome)
        assert len(archived_calls) == atomic["attempted"]
        # Across the adaptive runs: the same work to a band, the same field to the
        # accuracy envelope.
        assert_same_work(reference.outcome, output.outcome)
        assert_in_envelope(
            [output.fields(convention="fft")[-1]],
            converged,
            reference.fields(convention="fft")[-1],
            what=f"the {name} run's final field",
        )
        with open_tdse_archive(tmp_path / f"{name}.h5") as reader:
            evaluations = list(reader.evaluations())
            (session,) = reader.sessions
            assert session.closed and session.outputs == name
            assert all(item.complete for item in evaluations)
            roles = [item.role for item in evaluations]
            assert roles.count("diagnostic") == atomic["diagnostic_calls"] == 1
            assert roles.count("startup") == logical["startup"]
            # Each evaluation spent on a rejected step reads back as rejected.
            assert roles.count("rejected") == logical["rejected"]
            assert (
                roles.count("rejected") + roles.count("predictor")
                == logical["predictor"]
            )
            assert sum(len(item.chunks) for item in evaluations) == atomic["attempted"]
            outcomes = list(session.trials.values())
            assert outcomes.count("accepted") == logical["accepted_steps"]
            assert outcomes.count("rejected") == logical["rejected_steps"]
            # Every propagation evaluation carries its trial, position and density.
            propagation = [item for item in evaluations if item.role != "diagnostic"]
            assert all(item.trial is not None for item in propagation)
            assert all(
                item.density is not None and item.density > 0 for item in propagation
            )
            # The provider received exactly the archived drives, in order.
            drives = [
                reader.load(item, "drive", chunk=chunk.index)
                for item in evaluations
                for chunk in item.chunks
            ]
            assert len(drives) == len(archived_calls)
            for drive, call in zip(drives, archived_calls, strict=True):
                assert_array_equal(drive, call, strict=True)
            assert session.optical_reference is None
        record = plain(output.outcome.execution)["tdse"]["archive"]
        assert record["outputs"] == name and record["relative_to"] is None
        assert record["native_outputs"] == (
            "full" if name == "full" else "total_current"
        )


def test_the_optical_file_records_the_archive_and_reads_without_it(fake, tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "side").mkdir()
    optical = tmp_path / "out" / "run.h5"
    sidecar = tmp_path / "side" / "run.tdse.h5"
    _run(natural_simulation(), optical, tdse_archive=TDSEArchive(sidecar))
    with open_tdse_archive(sidecar) as reader:
        run_uuid = reader.run_uuid
        (session,) = reader.sessions
        assert session.optical_reference == {
            "path": "../side/run.tdse.h5",
            "relative_to": "optical_file",
        }
        context: Any = session.context
        assert context["descriptor_identity"]
        assert context["quadrature"]["n_nodes"] == 2
        assert len(context["node_weights_m2"]) == 2
    sidecar.unlink()
    with open_file(optical) as saved:
        assert saved.metadata is not None and saved.metadata.backend is not None
        record = plain(saved.metadata.backend.settings)["tdse"]["archive"]
        assert record["schema"] == TDSE_ARCHIVE_SCHEMA
        assert record["run_uuid"] == run_uuid
        assert record["path"] == "../side/run.tdse.h5"
        assert saved.n_saved > 2 and np.all(np.isfinite(saved.fields(convention="fft")))
    final = resume(optical)
    assert final.outcome is not None and final.outcome.status == "complete"
    closing = plain(final.outcome.execution)["tdse"]["archive"]
    assert closing["run_uuid"] == run_uuid and closing["evaluations"] > 0


# --- item 22: sessions, refusals and failures --------------------------------------


def test_a_resume_appends_a_session_under_the_same_run_uuid(fake, tmp_path):
    optical = tmp_path / "run.h5"
    sidecar = tmp_path / "run.tdse.h5"
    with pytest.raises(ResponseBudgetError):
        _run(_limited(30), optical, tdse_archive=TDSEArchive(sidecar))
    stopped = resume(optical)
    assert stopped.outcome is not None and stopped.outcome.status == "budget_exhausted"
    with open_tdse_archive(sidecar) as reader:
        first_count = len(list(reader.evaluations()))
        run_uuid = reader.run_uuid
    simulation = _limited(10_000)
    _run(simulation, optical, resume=True, tdse_archive=TDSEArchive(sidecar))
    with open_tdse_archive(sidecar) as reader:
        sessions = reader.sessions
        assert [session.kind for session in sessions] == ["run", "resume"]
        assert {session.run_uuid for session in sessions} == {run_uuid}
        assert all(session.closed for session in sessions)
        evaluations = list(reader.evaluations())
        identifiers = [item.evaluation_id for item in evaluations]
        assert identifiers == list(range(len(identifiers)))
        assert [item.session for item in evaluations[:first_count]] == [0] * first_count
        resumed = evaluations[first_count:]
        assert resumed and {item.session for item in resumed} == {1}
        # The cached consistency check is not repeated, and is not archived again.
        assert all(item.role != "diagnostic" for item in resumed)
        # A fresh startup restarts the history after the resume.
        assert resumed[0].role == "startup" and resumed[0].trial == 0
    final = resume(optical)
    assert final.outcome is not None and final.outcome.status == "complete"
    assert plain(final.outcome.execution)["tdse"]["archive"]["session"] == 1


def test_an_archive_first_requested_by_a_resume_starts_its_own_file(fake, tmp_path):
    optical = tmp_path / "run.h5"
    with pytest.raises(ResponseBudgetError):
        _run(_limited(30), optical)
    sidecar = tmp_path / "late.h5"
    _run(_limited(10_000), optical, resume=True, tdse_archive=TDSEArchive(sidecar))
    with open_tdse_archive(sidecar) as reader:
        assert [session.kind for session in reader.sessions] == ["resume"]
        run_uuid = reader.run_uuid
    final = resume(optical)
    assert final.outcome is not None
    # The resumed execution's record names the archive it started.
    assert plain(final.outcome.execution)["tdse"]["archive"]["run_uuid"] == run_uuid
    # The first execution archived nothing.
    assert all(
        "archive" not in plain(previous)["execution"].get("tdse", {})
        for previous in final.outcome.previous_executions
    )


@pytest.mark.parametrize("existing", ["text", "previous_run"])
def test_an_existing_unrelated_file_is_refused_before_any_atomic_work(
    fake, tmp_path, existing
):
    sidecar = tmp_path / "run.tdse.h5"
    if existing == "text":
        sidecar.write_text("keep me", encoding="utf-8")
    else:
        _run(natural_simulation(), tdse_archive=TDSEArchive(sidecar))
    before = sidecar.read_bytes()
    calls = len(fake.calls)
    optical = tmp_path / "run.h5"
    with pytest.raises(TDSEArchiveError, match="never overwritten"):
        _run(natural_simulation(), optical, tdse_archive=TDSEArchive(sidecar))
    assert len(fake.calls) == calls
    assert sidecar.read_bytes() == before
    assert not optical.exists()


def test_archive_requests_the_run_cannot_honour_are_refused(fake, tmp_path):
    optical = tmp_path / "run.h5"
    with pytest.raises(DriverError, match="must differ from the optical"):
        _run(natural_simulation(), optical, tdse_archive=TDSEArchive(optical))
    with pytest.raises(DriverError, match="TDSEArchive"):
        inspect.unwrap(Simulation.run)(
            natural_simulation(), progress="none", tdse_archive="x.h5"
        )
    gnlse = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        pulseshape="sech",
        n_save=2,
    )
    with pytest.raises(DriverError, match="no TDSE response"):
        gnlse.run(progress="none", tdse_archive=TDSEArchive(tmp_path / "a.h5"))
    assert fake.calls == []


def test_a_failed_write_stops_the_run_at_its_last_checkpoint(
    fake, tmp_path, monkeypatch
):
    original = archive_module.TDSEArchiveWriter._write_record
    written = []

    def failing(self, group, raw, schema, where):
        if where == "raw":
            written.append(len(fake.calls))
            if len(written) == 12:
                raise OSError(28, "No space left on device")
        original(self, group, raw, schema, where)

    monkeypatch.setattr(archive_module.TDSEArchiveWriter, "_write_record", failing)
    optical = tmp_path / "run.h5"
    sidecar = tmp_path / "run.tdse.h5"
    with pytest.raises(TDSEArchiveError, match="disk is full"):
        _run(natural_simulation(), optical, tdse_archive=TDSEArchive(sidecar))
    # The failing write followed the provider call it records; nothing came after.
    assert len(fake.calls) == written[-1]
    stopped = resume(optical)
    assert stopped.outcome is not None and stopped.outcome.status == "failed"
    assert "TDSEArchiveError" in stopped.outcome.message
    assert stopped.checkpoint is not None or stopped.n_saved >= 1
    with open_tdse_archive(sidecar) as reader:
        assert reader.truncated
        assert not reader.sessions[0].closed
        # Every evaluation before the failed one is complete; it alone is flagged.
        evaluations = list(reader.evaluations())
        assert [item.complete for item in evaluations] == [True] * (
            len(evaluations) - 1
        ) + [False]
        assert sum(len(item.chunks) for item in evaluations) == len(written)


def test_a_closing_failure_raises_after_success_and_is_a_note_after_an_error(
    fake, tmp_path, monkeypatch
):
    original = archive_module.TDSEArchiveWriter.close

    def failing(self):
        original(self)
        raise TDSEArchiveError("synthetic close failure")

    monkeypatch.setattr(archive_module.TDSEArchiveWriter, "close", failing)
    with pytest.raises(TDSEArchiveError, match="synthetic close failure"):
        _run(natural_simulation(), tdse_archive=TDSEArchive(tmp_path / "a.h5"))
    with pytest.raises(ResponseBudgetError) as error:
        _run(_limited(30), tdse_archive=TDSEArchive(tmp_path / "b.h5"))
    assert "could not be closed" in "\n".join(error.value.__notes__)


def test_a_failed_preparation_closes_the_archive(fake, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise BackendError("synthetic preparation failure")

    monkeypatch.setattr(runtime, "prepare_binding", refuse)
    sidecar = tmp_path / "a.h5"
    with pytest.raises(BackendError, match="synthetic preparation failure"):
        _run(natural_simulation(), tdse_archive=TDSEArchive(sidecar))
    with open_tdse_archive(sidecar) as reader:
        (session,) = reader.sessions
        assert session.closed and list(reader.evaluations()) == []


def test_prop_capillary_archives_through_the_run(fake, tmp_path):
    from tests.hacc_cases import hacc_response  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        warnings.simplefilter("ignore", GasFlowWarning)
        warnings.simplefilter("ignore", TDSEDomainWarning)
        output: Any = prop_capillary(
            256e-6,
            6 * 2.0**-20,
            "He",
            0.1,
            lambda0=800e-9,
            wavelength_limits=(400e-9, 1.6e-6),
            trange=40e-15,
            fwhm=6e-15,
            energy=1e-12,
            radial_rule="field_scale",
            n_r=2,
            noise=None,
            dz_init=2.0**-20,
            dz_max=2.0**-20,
            stats=StatsSpec(names=("energy",)),
            tdse=hacc_response(preset_basis(1), atomic_window="auto"),
            tdse_execution=TDSEExecution(max_atomic_calls=10_000),
            progress="none",
            output=MemoryOutput(),
            tdse_archive=TDSEArchive(tmp_path / "a.h5", outputs="required"),
        )
    assert output.outcome is not None and output.outcome.status == "complete"
    with open_tdse_archive(tmp_path / "a.h5") as reader:
        assert reader.sessions[0].outputs == "required"
        assert len(list(reader.evaluations())) > 2
