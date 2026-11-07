# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Runs started without a window, and their states read from their folders alone.

Slurm is faked by small scripts on ``PATH`` that answer as ``sbatch``, ``squeue``,
``sacct`` and ``scancel`` do; the direct runs are real, on the tiny capillary.
"""

import json
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from service.cases import TINY
from solprop.exceptions import ServiceError
from solprop.service import launch, runs
from solprop.service.job import main as job_main
from solprop.service.launch import (
    SUBMITTED,
    SlurmResources,
    SlurmState,
    cancel_slurm,
    prepare,
    slurm_script,
    slurm_states,
    start_direct,
    submit_slurm,
)
from solprop.service.record import (
    JOB,
    STATUS,
    Job,
    read_events,
    request_stop,
    write_job,
    write_json,
)
from solprop.service.runs import list_runs, slurm_jobs, summarise
from solprop.service.worker import SPEC, RunSpec

SLOW = TINY.replace('length = "5 cm"', 'length = "40 cm"').replace(
    "n_save = 6", "n_save = 40"
)
WAIT = 120.0


def wait_for(check, timeout: float = WAIT) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.05)


def fake(folder: Path, name: str, script: str) -> None:
    """Put a fake command on the path the tests give their children."""
    path = folder / name
    path.write_text("#!/bin/sh\n" + script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def slurm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake Slurm commands, first on the path."""
    if sys.platform == "win32":
        pytest.skip("the fake commands are shell scripts")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    monkeypatch.setenv("PATH", f"{bin_}{os.pathsep}{os.environ['PATH']}")
    return bin_


def test_a_prepared_folder_holds_the_spec_and_refuses_a_second_run(tmp_path):
    spec = RunSpec("r1", str(tmp_path / "r1"), recipe=TINY, threads=2)
    folder = prepare(spec)
    written = json.loads((folder / SPEC).read_text(encoding="utf-8"))
    assert written["threads"] == 2 and written["recipe"] == TINY
    (folder / "run.h5").write_bytes(b"")
    with pytest.raises(ServiceError, match="already holds a run"):
        prepare(spec)


def test_the_batch_script_asks_for_what_the_template_says(tmp_path):
    resources = SlurmResources(
        partition="gpu",
        time="1-00:00:00",
        cpus=8,
        memory="16G",
        gres="gpu:1",
        extra=("--account=abc", " "),
        setup=("module load gcc", ""),
    )
    script = slurm_script(tmp_path / "a run", "/env/bin/python", resources, threads=4)
    lines = script.splitlines()
    for wanted in (
        "#SBATCH --partition=gpu",
        "#SBATCH --time=1-00:00:00",
        "#SBATCH --cpus-per-task=8",
        "#SBATCH --mem=16G",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --account=abc",
        "#SBATCH --signal=B:TERM@120",
        "module load gcc",
        "export SOLPROP_THREADS=4",
    ):
        assert wanted in lines
    assert "#SBATCH --job-name=solprop-a_run" in lines
    assert lines[-1].startswith("exec /env/bin/python -m solprop.service.job '")
    assert resources.brief == "gpu · 8 CPUs · 16G · gpu:1 · 1-00:00:00"
    plain = slurm_script(tmp_path, "python", SlurmResources())
    assert "export SOLPROP_THREADS=${SLURM_CPUS_PER_TASK:-1}" in plain
    assert SlurmResources().brief == "1 CPU"


def test_a_job_is_submitted_and_its_states_read_from_queue_and_accounts(
    tmp_path, slurm
):
    fake(slurm, "sbatch", 'echo "4242;cluster"\n')
    fake(
        slurm,
        "squeue",
        'echo "4242|PENDING|Priority"\necho "7|RUNNING|None"\necho garbled\n',
    )
    fake(slurm, "sacct", 'echo "9|CANCELLED by 1000"\necho "9.batch|CANCELLED"\n')
    spec = RunSpec("r1", str(tmp_path / "r1"), recipe=TINY)
    folder = prepare(spec)
    assert submit_slurm(folder, "python", SlurmResources(cpus=2)) == "4242"
    submitted = json.loads((folder / SUBMITTED).read_text(encoding="utf-8"))
    assert submitted["slurm_job"] == "4242" and submitted["resources"]["cpus"] == 2
    assert (folder / launch.SLURM_SCRIPT).exists()
    states = slurm_states(["4242", "7", "9", "11"])
    assert states == {
        "4242": SlurmState("PENDING", "Priority"),
        "7": SlurmState("RUNNING", ""),
        "9": SlurmState("CANCELLED"),
    }
    assert slurm_states([]) == {}
    summary = summarise(folder, states)
    assert summary["state"] == "queued" and summary["detail"] == "Priority"
    assert slurm_jobs([folder]) == ["4242"]


def test_slurm_s_refusals_are_said(tmp_path, slurm):
    fake(slurm, "sbatch", 'echo "sbatch: error: invalid partition" >&2\nexit 1\n')
    fake(slurm, "scancel", 'echo "scancel: error: no such job" >&2\nexit 1\n')
    folder = prepare(RunSpec("r1", str(tmp_path / "r1"), recipe=TINY))
    with pytest.raises(ServiceError, match="invalid partition"):
        submit_slurm(folder, "python", SlurmResources())
    with pytest.raises(ServiceError, match="no such job"):
        cancel_slurm("1")
    fake(slurm, "scancel", "exit 0\n")
    cancel_slurm("1")
    fake(slurm, "squeue", "exit 1\n")
    fake(slurm, "sacct", "exit 1\n")
    assert slurm_states(["1"]) == {}


def test_a_missing_or_silent_scheduler_is_said(monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ServiceError, match="is not on this machine"):
        slurm_states(["1"])

    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired("squeue", 1)

    monkeypatch.setattr(launch.subprocess, "run", slow)
    with pytest.raises(ServiceError, match="did not answer"):
        cancel_slurm("1")


@pytest.mark.parametrize(
    ("known", "has_job", "fresh", "state"),
    [
        (SlurmState("RUNNING"), True, True, "running"),
        (SlurmState("RUNNING"), False, False, "starting"),
        (SlurmState("TIMEOUT"), True, False, "lost"),
        (SlurmState("COMPLETED"), True, True, "lost"),
        (None, True, True, "running"),
        (None, True, False, "lost"),
    ],
)
def test_a_slurm_run_s_state(tmp_path, known, has_job, fresh, state):
    folder = tmp_path / "r1"
    folder.mkdir()
    write_json(folder / SUBMITTED, {"launcher": "slurm", "slurm_job": "5"})
    if has_job:
        write_job(folder, Job("r1", "slurm", "node7", 99, None, "5", time.time()))
    updated = time.time() - (0 if fresh else 10 * runs.STALE)
    write_json(folder / STATUS, {"run_id": "r1", "updated": updated})
    states = {} if known is None else {"5": known}
    found = summarise(folder, states)
    assert found["state"] == state
    if known is not None and known.state == "TIMEOUT":
        assert found["detail"] == "timeout"


def test_a_direct_run_s_state_from_its_process(tmp_path):
    folder = tmp_path / "r1"
    folder.mkdir()
    here = socket.gethostname()
    write_json(
        folder / SUBMITTED, {"launcher": "direct", "host": here, "pid": os.getpid()}
    )
    assert summarise(folder)["state"] == "starting"
    write_json(folder / SUBMITTED, {"launcher": "direct", "host": here, "pid": 0})
    found = summarise(folder)
    assert found["state"] == "lost" and "before it started" in found["detail"]
    write_job(folder, Job.here("r1", "direct"))
    assert summarise(folder)["state"] == "running"
    write_job(folder, Job("r1", "direct", here, 0, None, None, time.time()))
    assert summarise(folder)["state"] == "lost"
    # A run on another machine, seen through a shared folder: its heartbeat says.
    write_job(folder, Job("r1", "direct", "elsewhere", 5, None, None, time.time()))
    write_json(folder / STATUS, {"run_id": "r1", "updated": time.time()})
    assert summarise(folder)["state"] == "running"
    write_json(folder / STATUS, {"run_id": "r1", "updated": "never"})
    assert summarise(folder)["state"] == "lost"


def test_folders_without_a_record_are_listed_for_what_they_hold(tmp_path):
    assert list_runs(tmp_path / "nowhere", None) == []
    old = tmp_path / "old"
    old.mkdir()
    (old / "recipe.toml").write_text('[recipe]\ntitle = "An old run"\n', "utf-8")
    copied = tmp_path / "copied"
    copied.mkdir()
    (copied / "run.h5").write_bytes(b"")
    (tmp_path / "other").mkdir()
    os.utime(old, (1, 1))
    listed = list_runs(tmp_path, None)
    assert [run["run_id"] for run in listed] == ["copied", "old"]
    assert listed[1]["title"] == "An old run" and listed[1]["state"] == "empty"
    assert listed[0]["state"] == "unknown" and listed[0]["title"] == "copied"
    (old / "recipe.toml").write_text("[recipe]\nroute = 1\n", "utf-8")
    assert summarise(old)["title"] == "old"


@pytest.mark.medium
def test_a_direct_run_outlives_its_starter_and_records_its_ending(tmp_path):
    folder = prepare(RunSpec("r1", str(tmp_path / "r1"), recipe=TINY))
    start_direct(folder)
    wait_for(lambda: summarise(folder)["state"] == "finished")
    kinds = [event.kind for event in read_events(folder)[0]]
    assert kinds.count("row") == 6 and kinds[-1] == "finished"
    job = json.loads((folder / JOB).read_text(encoding="utf-8"))
    assert job["launcher"] == "direct" and job["ended"] is not None
    summary = list_runs(tmp_path, None)[0]
    assert summary["output"].endswith("run.h5") and summary["record"]


@pytest.mark.medium
def test_a_direct_run_stops_on_its_stop_file(tmp_path):
    folder = prepare(RunSpec("r1", str(tmp_path / "r1"), recipe=SLOW))
    start_direct(folder)
    wait_for(
        lambda: (folder / STATUS).exists() and summarise(folder)["state"] == "running"
    )
    request_stop(folder)
    assert summarise(folder)["stopping"]
    wait_for(lambda: summarise(folder)["state"] == "stopped")


def test_the_job_command_needs_a_folder_with_a_spec(tmp_path, capsys):
    assert job_main([]) == 2
    assert job_main([str(tmp_path)]) == 2
    assert "no run to start" in capsys.readouterr().err
    (tmp_path / SPEC).write_text('{"run_id": "r1", "nonsense": 1}', encoding="utf-8")
    assert job_main([str(tmp_path)]) == 2


def test_the_job_command_runs_in_this_process(tmp_path, monkeypatch):
    folder = prepare(RunSpec("r1", str(tmp_path / "r1"), recipe=TINY, threads=1))
    monkeypatch.setenv("SLURM_JOB_ID", "77")
    monkeypatch.setenv("SOLPROP_THREADS", "2")  # restored after the test
    assert job_main([str(folder)]) == 0
    assert os.environ["SOLPROP_THREADS"] == "1"
    job = json.loads((folder / JOB).read_text(encoding="utf-8"))
    assert (job["launcher"], job["slurm_job"]) == ("slurm", "77")
    assert summarise(folder)["state"] == "finished"
