# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a restart keeps and restores is recorded, and warned about when it falls short.

SEP 47 section 5.5: every persisted plane and checkpoint states whether its method
history was retained, omitted or is not applicable, and why; a resumed execution states
what it restored (``restored``, ``reset`` or ``not_applicable``) and how its boundary
monitor was rebuilt. A resume warns once when an Adams history it expected was omitted
by the automatic size limit or cannot be restored, and not when it was set aside on
request. A file written before the record says nothing, and is read as unknown.
"""

import json
import math
import warnings
from typing import Any, cast

import h5py
import pytest

import solprop.driver.run as run_module
from driver.test_abandoned_step import built as low_storage
from driver.test_resume import run_until
from driver.test_resume_history import AFRESH, soliton
from solprop.backends.driver import StepDriver
from solprop.backends.driver_session import DriverSession
from solprop.backends.session import SessionError
from solprop.exceptions import SolpropWarning
from solprop.output import (
    HDF5Output,
    SnapshotController,
    open_file,
    read_snapshot,
    resume,
)
from solprop.output.hdf5 import _dataset
from solprop.stepper import Multistep, RungeKutta

# The fixtures launch noise-free into tight windows; the edge is not what is tested.
pytestmark = [
    pytest.mark.medium,
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]


def restart(path) -> dict[str, Any]:
    """Return the restart record of a closed file's latest execution record."""
    with open_file(path) as stored:
        assert stored.outcome is not None
        return cast("dict[str, Any]", dict(stored.outcome.execution).get("restart", {}))


def resumed(build, path) -> list[warnings.WarningMessage]:
    """Resume `path` to its end and return the history warnings it gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build().run(path, resume=True, progress="none")
    return [
        one
        for one in caught
        if issubclass(one.category, SolpropWarning) and AFRESH in str(one.message)
    ]


def interrupted(build, path) -> float:
    """Run into `path`, stop after two planes and return where it stopped."""
    run_until(build(), path, after=2)
    return resume(path).plane.z


@pytest.mark.parametrize(
    ("policy", "reason"), [(None, "automatic_policy"), (True, "explicit_policy")]
)
def test_a_kept_adams_history_is_recorded_and_restored_without_a_warning(
    build_simulation, tmp_path, policy, reason
):
    build = soliton(build_simulation, "numpy_ref", resume_history=policy)
    path = tmp_path / "run.h5"
    z = interrupted(build, path)
    retention = restart(path)["retention"]
    assert retention["state"] == "retained"
    assert retention["reason"] == reason
    assert retention["z"] == z
    assert retention["nbytes"] > 0
    assert "restoration" not in restart(path)  # a fresh execution restores nothing
    assert resumed(build, path) == []
    record = restart(path)
    assert record["restoration"] == {
        "state": "restored",
        "reason": "retained_history",
        "z": z,
        "retention": retention,
    }
    assert record["retention"]["state"] == "retained"


def test_a_history_set_aside_on_request_is_recorded_without_a_warning(
    build_simulation, tmp_path
):
    build = soliton(build_simulation, "numpy_ref", resume_history=False)
    path = tmp_path / "run.h5"
    z = interrupted(build, path)
    assert restart(path)["retention"] == {
        "state": "omitted",
        "reason": "explicit_policy",
        "z": z,
    }
    assert resumed(build, path) == []
    restoration = restart(path)["restoration"]
    assert restoration["state"] == "reset"
    assert restoration["reason"] == "explicit_policy"


def test_a_kept_history_set_aside_on_resume_is_recorded_without_a_warning(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    interrupted(soliton(build_simulation, "numpy_ref"), path)
    assert (
        resumed(soliton(build_simulation, "numpy_ref", resume_history=False), path)
        == []
    )
    restoration = restart(path)["restoration"]
    assert (restoration["state"], restoration["reason"]) == ("reset", "explicit_policy")
    assert restoration["retention"]["state"] == "retained"


def test_a_history_over_the_automatic_limit_warns_once_on_resume(
    build_simulation, tmp_path, monkeypatch
):
    monkeypatch.setattr(run_module, "MARCH_SNAPSHOT_AUTO_BYTES", 1)
    build = soliton(build_simulation, "numpy_ref")
    path = tmp_path / "run.h5"
    z = interrupted(build, path)
    retention = restart(path)["retention"]
    assert retention["state"] == "omitted"
    assert retention["reason"] == "automatic_size_limit"
    assert retention["z"] == z
    assert retention["limit"] == 1
    caught = resumed(build, path)
    assert len(caught) == 1
    assert "resume_history=True keeps it" in str(caught[0].message)
    restoration = restart(path)["restoration"]
    assert (restoration["state"], restoration["reason"]) == (
        "reset",
        "automatic_size_limit",
    )


def test_a_kept_history_that_cannot_be_read_back_warns_once(build_simulation, tmp_path):
    build = soliton(build_simulation, "numpy_ref")
    path = tmp_path / "run.h5"
    interrupted(build, path)
    with h5py.File(path, "r+") as file:
        del file["march_snapshot"]
    caught = resumed(build, path)
    assert len(caught) == 1
    assert "cannot be read back" in str(caught[0].message)
    restoration = restart(path)["restoration"]
    assert (restoration["state"], restoration["reason"]) == (
        "reset",
        "unavailable_history",
    )


def test_a_history_this_backend_cannot_install_is_recorded_as_incompatible(
    build_simulation, tmp_path, monkeypatch
):
    build = soliton(build_simulation, "numpy_ref")
    path = tmp_path / "run.h5"
    interrupted(build, path)
    monkeypatch.setattr(
        StepDriver, "restore_march", lambda self, snapshot: "refused by the test"
    )
    caught = resumed(build, path)
    assert len(caught) == 1
    assert "refused by the test" in str(caught[0].message)
    restoration = restart(path)["restoration"]
    assert (restoration["state"], restoration["reason"]) == (
        "reset",
        "incompatible_representation",
    )


def test_a_runge_kutta_resume_records_its_reset_controller(build_simulation, tmp_path):
    build = soliton(build_simulation, "numpy_ref", integrator=RungeKutta())
    path = tmp_path / "run.h5"
    z = interrupted(build, path)
    assert restart(path)["retention"] == {
        "state": "omitted",
        "reason": "unsupported_method",
        "z": z,
    }
    assert resumed(build, path) == []
    assert restart(path)["restoration"] == {
        "state": "reset",
        "reason": "unsupported_method",
        "z": z,
    }


def test_a_low_storage_resume_has_no_history_to_record(tmp_path):
    path = tmp_path / "run.h5"

    def build():
        return low_storage(fields="all")

    run_until(build(), path, after=1)
    z = resume(path).plane.z
    assert restart(path)["retention"] == {
        "state": "not_applicable",
        "reason": "no_method_history",
        "z": z,
    }
    assert resumed(build, path) == []
    assert restart(path)["restoration"] == {
        "state": "not_applicable",
        "reason": "no_method_history",
        "z": z,
    }


@pytest.mark.parametrize("march", [True, False])
def test_a_file_without_the_record_reads_as_unknown_and_resumes(
    build_simulation, tmp_path, march
):
    # A file written before the record: no reason is invented for it, and a missing
    # history is not warned about, as before.
    build = soliton(build_simulation, "numpy_ref")
    path = tmp_path / "legacy.h5"
    z = interrupted(build, path)
    with h5py.File(path, "r+") as file:
        outcome = json.loads(str(_dataset(file, "meta/outcome").asstr()[()]))
        for record in (outcome, *outcome["previous_executions"]):
            record["execution"].pop("restart", None)
        del file["meta/outcome"]
        file.create_dataset(
            "meta/outcome", data=json.dumps(outcome), dtype=h5py.string_dtype()
        )
        if not march:
            del file["march_snapshot"]
    state = resume(path)
    assert state.outcome is not None and "restart" not in state.outcome.execution
    assert run_module.retention_at(state.outcome.execution, z) is None
    assert resumed(build, path) == []
    restoration = restart(path)["restoration"]
    assert restoration["retention"] == {"state": "unknown", "reason": "legacy"}
    assert (restoration["state"], restoration["reason"]) == (
        ("restored", "retained_history") if march else ("reset", "unavailable_history")
    )


def test_a_resumed_monitor_states_its_reconstruction(build_simulation, tmp_path):
    build = soliton(build_simulation, "numpy_ref")
    path = tmp_path / "run.h5"
    interrupted(build, path)
    with open_file(path) as stored:
        assert stored.outcome is not None
        assert "boundary_monitor" in stored.outcome.execution
    resumed(build, path)
    monitor = restart(path)["monitor"]
    assert monitor["state"] == "reconstructed"
    assert "check_cadence" in monitor["restarted"]
    assert "cumulative_totals" in monitor["restored"]


def test_a_required_snapshot_records_its_retention(tmp_path, monkeypatch):
    from driver.test_snapshots import ordinary  # noqa: PLC0415 - its case

    monkeypatch.setattr(run_module, "MARCH_SNAPSHOT_AUTO_BYTES", 0)
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    ordinary(integrator=Multistep()).run(
        tmp_path / "working.h5", snapshots=controller, progress="none"
    )
    assert controller.last_result is not None
    assert controller.last_result.generation is not None
    info = read_snapshot(controller.last_result.generation)
    retention = restart(info.files.output)["retention"]
    assert (retention["state"], retention["reason"]) == (
        "retained",
        "snapshot_required",
    )
    assert retention["z"] == info.z


def test_a_history_write_cut_short_warns_once_on_resume(
    build_simulation, tmp_path, monkeypatch
):
    # The second write of the history fails part-way, leaving the torn group a reader
    # takes for none; the run stops there, and its resume warns that the history it
    # expected is lost.
    save = HDF5Output.save_march
    calls = []

    def torn(self, snapshot):
        calls.append(snapshot.z)
        if len(calls) == 1:
            return save(self, snapshot)
        self._file.require_group("march_snapshot").attrs["z"] = math.nan
        raise OSError("the disk filled")

    build = soliton(build_simulation, "numpy_ref")
    path = tmp_path / "run.h5"
    with monkeypatch.context() as patch:
        patch.setattr(HDF5Output, "save_march", torn)
        with pytest.raises(OSError, match="the disk filled"):
            build().run(path, progress="none")
    retention = restart(path)["retention"]
    assert (retention["state"], retention["reason"]) == ("omitted", "write_failed")
    assert resume(path).march is None
    caught = resumed(build, path)
    assert len(caught) == 1
    assert "write failed" in str(caught[0].message)
    restoration = restart(path)["restoration"]
    assert (restoration["state"], restoration["reason"]) == (
        "reset",
        "unavailable_history",
    )


def test_a_monitor_continuation_is_given_once_and_before_it_watches():
    session = object.__new__(DriverSession)
    session._attached = False
    given = {"record": None, "plane": True, "unwatched": ()}
    session._monitor_resume = given
    session.continue_monitor({"record": {"other": 1}, "plane": False, "unwatched": ()})
    assert session._monitor_resume is given  # a Start's continuation is kept
    session._monitor_resume = None
    session.continue_monitor(given)
    assert session._monitor_resume is given
    session._attached = True
    with pytest.raises(SessionError, match="before it watches"):
        session.continue_monitor(given)
