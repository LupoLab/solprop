# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run emitted Slurm task bodies locally without a scheduler or accelerator."""

import inspect
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import solprop.scan as cli
from driver.test_scan_batch import build
from solprop.driver.executors import Local
from solprop.driver.executors._request import ScanRequest
from solprop.driver.executors.pool import Pool
from solprop.driver.executors.queue import Queue
from solprop.driver.executors.slurm import Slurm
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import DriverError
from solprop.processing.scans import ScanOutput
from solprop.scan import current_executor, main

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the Slurm (and its Queue) executor is POSIX only and refuses Windows",
)


@pytest.mark.parametrize("mode", ["queue", "chunk"])
@pytest.mark.medium
def test_slurm_task_body_runs_durable_scan(tmp_path, mode):
    script = tmp_path / "scan with ' spaces$.py"
    output = tmp_path / "result.h5"
    queue = tmp_path / "queue with spaces"
    script.write_text(
        """from solprop.driver import Simulation
from solprop.driver.scan import Scan, run_scan

def build(point):
    return Simulation.from_gnlse(0.11, 0.001, [0., 0., -1.183e-26],
        lambda0=835e-9, wavelength_limits=(500e-9, 2e-6), trange=1e-12,
        width=100e-15, pulseshape="sech", peak_power=point.values["power"],
        n_save=2, backend="numpy_ref")

if __name__ == "__main__":
    run_scan(Scan("slurm", power=[1., 2., 3., 4.]), build,
        output="""
        + repr(str(output))
        + """, progress="none")
"""
    )
    policy = Slurm(script, ntasks=2, mode=mode, queue_path=queue, workers=1)
    batch = tmp_path / "job.sh"
    batch.write_text(policy.render())
    subprocess.run(["bash", "-n", str(batch)], check=True)
    for task in (1, 2):
        subprocess.run(
            ["bash", str(batch)],
            env=dict(os.environ, SLURM_ARRAY_TASK_ID=str(task)),
            check=True,
            capture_output=True,
            text=True,
        )
    assert ScanOutput(output).status.tolist() == ["done"] * 4
    assert (queue / ".done").exists()
    assert main(["collect", str(queue), "--output", str(output)]) == 0


@pytest.mark.medium
def test_slurm_instances_receive_distinct_gpu_assignments(tmp_path):
    script = tmp_path / "inspect.py"
    script.write_text(
        """import json, os
from pathlib import Path
record = {key: os.environ[key] for key in (
    "SOLPROP_SCAN_INSTANCE", "CUDA_VISIBLE_DEVICES", "NUMBA_NUM_THREADS",
    "SOLPROP_SCAN_CPU_OFFSET", "SOLPROP_SCAN_CPU_COUNT")}
Path("""
        + repr(str(tmp_path))
        + """, record["SOLPROP_SCAN_INSTANCE"] + ".json").write_text(json.dumps(record))
"""
    )
    policy = Slurm(
        script,
        instances=2,
        cpus_per_task=4,
        gres="gpu:h200:2",
        queue_path=tmp_path / "queue",
    )
    batch = tmp_path / "instances.sh"
    batch.write_text(policy.render())
    subprocess.run(
        ["bash", str(batch)],
        env=dict(
            os.environ,
            SLURM_ARRAY_TASK_ID="1",
            CUDA_VISIBLE_DEVICES="GPU-first,GPU-second",
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    first = json.loads((tmp_path / "0.json").read_text())
    second = json.loads((tmp_path / "1.json").read_text())
    assert first["CUDA_VISIBLE_DEVICES"] == "GPU-first"
    assert second["CUDA_VISIBLE_DEVICES"] == "GPU-second"
    assert [first["SOLPROP_SCAN_CPU_OFFSET"], second["SOLPROP_SCAN_CPU_OFFSET"]] == [
        "0",
        "2",
    ]
    assert first["NUMBA_NUM_THREADS"] == second["NUMBA_NUM_THREADS"] == "1"
    # The controller keeps its share; only its Pool workers get the one-thread budget.
    assert "export SOLPROP_THREADS" not in policy.render()


def test_slurm_submission_uses_argument_vector(tmp_path, monkeypatch):
    script = tmp_path / "example.py"
    script.write_text("pass\n")
    policy = Slurm(
        script, queue_path=tmp_path / "queue", partition="partition with spaces"
    )
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        return subprocess.CompletedProcess(arguments, 0, stdout="12345;cluster\n")

    monkeypatch.setattr(subprocess, "run", run)
    assert policy.submit() == "12345;cluster"
    assert calls[0][0] == [
        "sbatch",
        "--parsable",
        str(tmp_path / "queue/submission.slurm"),
    ]
    assert "#SBATCH --partition='partition with spaces'" in policy.render()
    report = next((tmp_path / "queue/submissions").glob("*.json"))
    assert json.loads(report.read_text())["job_id"] == "12345;cluster"


@pytest.mark.parametrize(
    "options",
    [
        {"ntasks": 0},
        {"mem": "0"},
        {"instances": 2},
        {"partition": "bad\n#SBATCH --exclusive"},
    ],
)
def test_slurm_refuses_invalid_resources(options):
    with pytest.raises(DriverError):
        Slurm("script.py", **options)


def test_cli_emits_script_without_executing_input(tmp_path):
    script = tmp_path / "not_executed.py"
    script.write_text('raise AssertionError("must not execute during emission")\n')
    target = tmp_path / "job.sh"
    assert (
        main(
            [
                str(script),
                "--emit-slurm",
                str(target),
                "--ntasks",
                "2",
                "--mode",
                "chunk",
            ]
        )
        == 0
    )
    assert "#SBATCH --array=1-2" in target.read_text()


@pytest.mark.parametrize("mode", ["queue", "chunk"])
def test_slurm_script_snapshot(tmp_path, monkeypatch, file_regression, mode):
    monkeypatch.setattr("solprop.driver.executors.slurm.sys.executable", "/python")
    policy = Slurm(
        tmp_path / "scan 'quoted'.py",
        ntasks=3,
        cpus_per_task=8,
        mem="12G",
        time="00:30:00",
        partition="cpu queue",
        gres="gpu:h200:2",
        mode=mode,
        instances=2,
        queue_path=tmp_path / "point queue",
        workers=1,
        threads=2,
        fresh=True,
        workdir=tmp_path,
    )
    file_regression.check(
        policy.render().replace(str(tmp_path), "/shared/scan"), extension=".sh"
    )


def test_slurm_single_instance_gpu_snapshot_pins_one_worker(
    tmp_path, monkeypatch, file_regression
):
    monkeypatch.setattr("solprop.driver.executors.slurm.sys.executable", "/python")
    # instances=0 is the default: one controller in the task, holding the one
    # device the allocation was granted.
    policy = Slurm(
        tmp_path / "single.py",
        cpus_per_task=8,
        gres="gpu:h200:1",
        queue_path=tmp_path / "single queue",
        threads=2,
        workdir=tmp_path,
    )
    script = policy.render()
    assert "--workers 1" in script
    # The per-instance device split is a different question, and one instance
    # does not ask it: Slurm already scoped CUDA_VISIBLE_DEVICES to the task.
    assert "CUDA_VISIBLE_DEVICES" not in script
    monkeypatch.setattr(Queue, "prepare", lambda self, request: None)
    policy.prepare(ScanRequest(Scan("single", x=[1]), {}, tmp_path / "out.h5", abs))
    assert policy._queue is not None
    assert policy._queue.pool.workers == 1
    file_regression.check(
        script.replace(str(tmp_path), "/shared/scan"), extension=".sh"
    )


def test_cli_worker_controls_choose_pool_and_refuse_local_conflicts(monkeypatch):
    from_policy = []
    monkeypatch.setattr(
        "solprop.scan.runpy.run_path",
        lambda *a, **k: from_policy.append(current_executor()),
    )
    assert main(["scan.py", "--workers", "2"]) == 0
    assert isinstance(from_policy[0], Pool)
    assert from_policy[0].workers == 2
    with pytest.raises(SystemExit):
        main(["scan.py", "--local", "--workers", "2"])
    with pytest.raises(SystemExit):
        main(["scan.py", "--submit", "--memory-limit", "1000"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--local", "--queue", "queue"],
        ["--submit", "--chunk", "0/2"],
        ["--chunk", "invalid"],
        ["--chunk", "one/two"],
    ],
)
def test_cli_rejects_conflicting_or_malformed_routes(arguments):
    with pytest.raises(SystemExit):
        main(["scan.py", *arguments])


def test_cli_chunk_binds_global_partition_and_restores_context(monkeypatch):
    previous = sys.argv
    captured = []

    def execute(*args, **kwargs):
        captured.append(current_executor())
        assert sys.argv[0].endswith("scan.py")
        raise RuntimeError("script failed")

    monkeypatch.setattr("solprop.scan.runpy.run_path", execute)
    with pytest.raises(RuntimeError, match="script failed"):
        main(["scan.py", "--chunk", "1/3", "--fresh"])
    assert sys.argv is previous
    assert current_executor() is None
    policy = captured[0]
    assert isinstance(policy, Queue)
    assert policy.chunk == (1, 3)
    assert policy.pool.fresh


@pytest.mark.parametrize("workers", [0, -1])
def test_slurm_rejects_nonpositive_workers(workers):
    with pytest.raises(DriverError, match="workers must be positive"):
        Slurm("scan.py", workers=workers)


@pytest.mark.parametrize(
    "error", [FileNotFoundError("sbatch"), subprocess.CalledProcessError(1, "sbatch")]
)
def test_slurm_submission_failure_keeps_reviewable_script(tmp_path, monkeypatch, error):
    def refuse(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", refuse)
    policy = Slurm(tmp_path / "scan.py")
    destination = tmp_path / "job.sh"
    with pytest.raises(DriverError, match="submission failed"):
        policy.submit(destination)
    assert destination.read_text() == policy.render()
    assert policy.job_id is None
    assert not (tmp_path / "scan.queue/submissions").exists()


def test_slurm_requires_prepare_before_dispatch():
    with pytest.raises(DriverError, match="must be prepared"):
        list(Slurm("scan.py").run([], abs))


def test_slurm_prepares_worker_chunk_and_drains_inner_queue(tmp_path, monkeypatch):
    monkeypatch.setenv("SOLPROP_SLURM_WORKER", "1")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "2")
    monkeypatch.setenv("SOLPROP_SCAN_INSTANCE", "1")
    prepared = []
    monkeypatch.setattr(
        Queue, "prepare", lambda self, request: prepared.append(request)
    )
    monkeypatch.setattr(
        Queue, "run", lambda self, points, work: iter([work(p) for p in points])
    )
    policy = Slurm(
        "scan.py", ntasks=3, instances=2, cpus_per_task=4, mode="chunk", gres="gpu:2"
    )
    request = ScanRequest(Scan("tiny", x=[1]), {}, tmp_path / "scan.h5", abs)
    policy.validate(abs)
    policy.prepare(request)
    assert prepared == [request]
    assert policy._queue is not None
    assert policy._queue.chunk == (3, 6)
    assert policy._queue.pool.workers == 1
    assert policy._queue.pool.memory_limit == 2 * 1024**3
    assert list(policy.run([-2], abs)) == [2]
    monkeypatch.delenv("SOLPROP_SLURM_WORKER")
    submitted = []
    monkeypatch.setattr(policy, "submit", lambda: submitted.append(True))
    assert list(policy.run([-2], abs)) == []
    assert submitted == [True]


def test_slurm_submission_route_collects_pending_manifest(tmp_path, monkeypatch):
    monkeypatch.delenv("SOLPROP_SLURM_WORKER", raising=False)
    policy = Slurm(tmp_path / "scan.py", workdir=tmp_path)
    submissions = []
    monkeypatch.setattr(policy, "submit", lambda: submissions.append(True))
    output = run_scan(
        Scan("submission", power=[1.0]),
        build,
        output=tmp_path / "scan.h5",
        executor=policy,
        progress="none",
    )
    assert submissions == [True]
    assert output.status.tolist() == ["todo"]
    assert policy._queue is not None
    assert policy._queue.chunk is None


def test_cli_help_documents_the_collect_mode(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    assert stopped.value.code == 0
    printed = capsys.readouterr().out
    assert "collect QUEUE --output OUT.h5" in printed
    with pytest.raises(SystemExit):
        main(["collect", "--help"])
    assert "collect QUEUE --output OUT.h5" in capsys.readouterr().out


def test_slurm_production_guard_refuses_unknown_mode_without_session_hook():
    with pytest.raises(DriverError, match="mode must be"):
        inspect.unwrap(Slurm.__init__)(Slurm.__new__(Slurm), "scan.py", mode="unknown")


def test_cli_module_entry_runs_script_and_emission_requires_slurm(
    tmp_path, monkeypatch
):
    script = tmp_path / "scan.py"
    script.write_text("value = 42\n")
    monkeypatch.setattr(sys, "argv", ["solprop.scan", str(script), "--local"])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(Path(cli.__file__)), run_name="__main__")
    assert stopped.value.code == 0
    monkeypatch.setattr(cli, "_policy", lambda *args: Local())
    with pytest.raises(SystemExit):
        main([str(script), "--emit-slurm", str(tmp_path / "job.sh")])
