# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Require the real packaged runtime matrix when the Julia lane is enabled."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from solprop.backends.julia._identity import read_record, write_record
from solprop.backends.julia._setup import select_setup

PROBE = Path(__file__).with_name("runtime_probe.py")
pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _copy_publication(source, depot):
    directory = depot / "setups" / source.setup_id
    shutil.copytree(source.engine.parents[1], directory, symlinks=True)
    copied = read_record(directory)
    relocated = replace(copied, depot=depot.resolve())
    write_record(relocated, directory)
    return relocated


@pytest.fixture(scope="module")
def runtime_installation(tmp_path_factory):
    setup_id = os.environ.get("SOLPROP_TEST_JULIA_SETUP_ID")
    assert setup_id, "the enabled Julia lane must publish a managed setup identity"
    assert os.environ.get("SOLPROP_JULIA_DEPOT"), "the Julia lane needs a private depot"
    source = select_setup(setup_id=setup_id)
    depot = tmp_path_factory.mktemp("julia-runtime") / "depot"
    depot.mkdir()
    # Runtime caches and every mutable test input stay in pytest's directory. The
    # supplied managed installation is read only, including its compiled cache.
    for name in ("packages", "artifacts", "compiled"):
        origin = source.depot / name
        if origin.exists():
            shutil.copytree(
                origin,
                depot / name,
                symlinks=True,
                ignore=shutil.ignore_patterns("*.pidfile"),
            )
    return _copy_publication(source, depot)


def _environment(record, threads, directory):
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("JULIA_", "PYTHON_JULIACALL_", "SOLPROP_NATIVE_"))
    }
    environment.update(
        SOLPROP_TEST_JULIA="1",
        SOLPROP_TEST_JULIA_SETUP_ID=record.setup_id,
        SOLPROP_JULIA_DEPOT=str(record.depot),
        SOLPROP_THREADS=str(threads),
        SOLPROP_NATIVE_THREADS=str(threads),
        SOLPROP_CALIBRATION="off",
        OMP_PROC_BIND="false",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(directory),
        TMP=str(directory),
        TEMP=str(directory),
    )
    return environment


def _termination_ready(process, output, mode, threads, timeout):
    entered = output.with_suffix(".term-entered")
    deadline = time.monotonic() + timeout
    while True:
        assert process.poll() is None, "SIGTERM child exited before verified entry"
        if entered.is_file():
            assert entered.read_text(encoding="utf-8") == "entered"
            result = json.loads(output.read_text(encoding="utf-8"))
            handler, phase = mode.split(":")
            assert result["case"] == "sigterm" and result["pid"] == process.pid
            assert result["stage"] == "ready-for-accepted-termination"
            assert result["initial_handler"] == handler and result["phase"] == phase
            assert result["requested_threads"] == result["actual_threads"] == threads
            return result
        assert time.monotonic() < deadline, "SIGTERM child never verified entry"
        time.sleep(0.002)


def _signal_group(process, number):
    # Popen(start_new_session=True) makes this child's PID our private PGID. Do
    # not signal a group inherited from pytest or any other unowned process.
    assert os.getpgid(process.pid) == process.pid, (
        "SIGTERM child lost its private group"
    )
    os.killpg(process.pid, number)


def _termination_exit(process, grace, reap_timeout):
    expired = False
    killed_at = None
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        expired = True
        if process.poll() is None:
            try:
                _signal_group(process, signal.SIGKILL)
            except ProcessLookupError:
                # Fatal TERM can complete between the timeout and kill. Its
                # actual returncode still must pass the narrow check below.
                pass
            else:
                killed_at = time.monotonic()
        process.wait(timeout=reap_timeout)
    return expired, killed_at


def _supervise_sigterm(
    process, output, mode, threads, *, ready_timeout=180.0, grace=2.0
):
    started = time.monotonic()
    result = _termination_ready(process, output, mode, threads, ready_timeout)
    entered_at = time.monotonic()
    _signal_group(process, signal.SIGTERM)
    term_at = time.monotonic()
    expired, killed_at = _termination_exit(process, grace, 15.0)
    finished = time.monotonic()
    code = process.returncode
    result["supervision"] = {
        "ready_observed": True,
        "entered_observed": True,
        "process_group": process.pid,
        "entered_seconds": entered_at - started,
        "term_sent": True,
        "term_seconds": term_at - started,
        "term_grace_seconds": grace,
        "term_grace_expired": expired,
        "forced_escalation": killed_at is not None,
        "kill_seconds": None if killed_at is None else killed_at - started,
        "elapsed_seconds": finished - started,
        "after_term_seconds": finished - term_at,
        "returncode": code,
        "exit_signal": -code if code < 0 else None,
        "exit_status": code if code >= 0 else None,
        "reaped": True,
    }
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    allowed = {-signal.SIGTERM, 128 + signal.SIGTERM}
    if killed_at is not None:
        allowed.add(-signal.SIGKILL)
    assert code in allowed, f"unexpected supervised SIGTERM exit: {code}"
    return result


def _run_terminating(record, directory, command, output, mode, threads):
    assert not output.exists() and not output.with_suffix(".term-entered").exists()
    stdout_path = output.with_suffix(".stdout")
    stderr_path = output.with_suffix(".stderr")
    # Files cannot block a fatal Julia stack trace behind an undrained pipe while
    # the supervisor waits for entry or enforces the TERM grace deadline.
    with (
        stdout_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(
            command,
            env=_environment(record, threads, directory),
            cwd=directory,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            result = _supervise_sigterm(process, output, mode, threads)
        except BaseException as failure:
            if process.poll() is None:
                try:
                    _signal_group(process, signal.SIGKILL)
                    process.wait(timeout=15)
                except (AssertionError, OSError, subprocess.TimeoutExpired) as cleanup:
                    failure.add_note(f"SIGTERM child cleanup failed: {cleanup}")
            failure.add_note(f"stdout: {stdout_path}; stderr: {stderr_path}")
            raise
    diagnostics = stderr_path.read_text(encoding="utf-8")
    assert "Exception ignored in" not in diagnostics, diagnostics
    return result


def _run(record, directory, case, threads, *, mode="", terminating=False):
    output = directory / f"{case}-{mode or 'normal'}-{threads}.json"
    command = [sys.executable, str(PROBE), case, str(output), "--mode", mode]
    if terminating:
        return _run_terminating(record, directory, command, output, mode, threads)
    process = subprocess.Popen(
        command,
        env=_environment(record, threads, directory),
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        stdout, stderr = process.communicate(timeout=15)
        pytest.fail(f"{case} timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")
    assert process.returncode == 0, (
        f"{case}/{mode}/{threads}: exit {process.returncode}\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert "Exception ignored in" not in stderr, stderr
    for diagnostic in (
        "error in running finalizer:",
        "atexit hook threw an error:",
        "error during exit cleanup:",
    ):
        assert diagnostic not in stderr, stderr
    assert output.is_file(), f"{case} exited without its verified result: {stderr}"
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["case"] == case and result["requested_threads"] == threads
    if case != "preflight":
        assert result["actual_threads"] == threads
    if case == "shutdown":
        assert output.with_suffix(".owner-released").read_text(encoding="utf-8") == (
            "released"
        )
        assert output.with_suffix(".julia-finalized").read_text(encoding="utf-8") == (
            "finalized"
        )
    return result


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "case",
    [
        "worker_first",
        "startup_race",
        "cancellation",
        "last_owner",
        "worker_refusals",
        "worker_refusals_gc",
        "owner_drop",
        "sanitized_error",
        "polyester",
        "shutdown",
    ],
)
def test_runtime_threads_gc_cancellation_and_shutdown(
    runtime_installation, tmp_path, case, threads, record_property
):
    result = _run(runtime_installation, tmp_path, case, threads)
    record_property("runtime_probe", json.dumps(result, sort_keys=True))


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("mode", ["custom", "ignored", "default"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX process signal delivery")
def test_actual_sigint_handlers_survive_threaded_gc(
    runtime_installation, tmp_path, mode, threads, record_property
):
    result = _run(runtime_installation, tmp_path, "sigint", threads, mode=mode)
    assert result["deliveries"] == 4
    assert result["custom_callbacks"] == (3 if mode == "custom" else 1)
    record_property("runtime_probe", json.dumps(result, sort_keys=True))


@pytest.mark.skipif(os.name != "posix", reason="POSIX process signal delivery")
@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("mode", ["custom", "ignored", "default"])
@pytest.mark.parametrize("phase", ["idle", "work"])
def test_sigterm_retains_the_accepted_process_termination_limitation(
    runtime_installation, tmp_path, mode, phase, threads, record_property
):
    result = _run(
        runtime_installation,
        tmp_path,
        "sigterm",
        threads,
        mode=f"{mode}:{phase}",
        terminating=True,
    )
    assert result["stage"] == "ready-for-accepted-termination"
    assert result["phase"] == phase
    supervision = result["supervision"]
    assert supervision["term_sent"] and supervision["entered_observed"]
    assert supervision["reaped"]
    assert supervision["entered_seconds"] <= supervision["term_seconds"]
    if supervision["returncode"] == -signal.SIGKILL:
        assert supervision["forced_escalation"] and supervision["term_grace_expired"]
    record_property("runtime_probe", json.dumps(result, sort_keys=True))


@pytest.mark.parametrize("threads", [1, 2])
def test_fresh_spawned_workers_own_their_runtime(
    runtime_installation, tmp_path, threads, record_property
):
    result = _run(runtime_installation, tmp_path, "spawn", threads)
    record_property("runtime_probe", json.dumps(result, sort_keys=True))


@pytest.mark.parametrize("threads", [1, 2])
def test_main_lazily_loads_each_installed_device_after_worker_refusal(
    runtime_installation, tmp_path, threads, record_property
):
    result = _run(runtime_installation, tmp_path, "devices", threads)
    installed = set(dict(runtime_installation.projects)) - {"cpu"}
    assert set(result["loaded_devices"]) == installed
    assert set(result["refused_devices"]) == {"cuda", "rocm", "metal"} - installed
    assert set(result["worker_refused_devices"]) == {"cuda", "rocm", "metal"}
    record_property("runtime_probe", json.dumps(result, sort_keys=True))


@pytest.mark.parametrize("mode", ["complete", "incomplete"])
def test_preflight_never_starts_julia_a_subprocess_or_network(
    runtime_installation, tmp_path, mode, record_property
):
    record = runtime_installation
    if mode == "incomplete":
        record = _copy_publication(record, tmp_path / "incomplete-depot")
    result = _run(record, tmp_path, "preflight", 1, mode=mode)
    assert result["network_calls"] == 0 and result["subprocess_calls"] == 0
    record_property("runtime_probe", json.dumps(result, sort_keys=True))


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize("mode", ["before", "after"])
def test_native_and_julia_fft_runtimes_coexist_in_both_import_orders(
    runtime_installation, tmp_path, mode, threads, record_property
):
    result = _run(runtime_installation, tmp_path, "native", threads, mode=mode)
    assert result["native_threads"] == threads and result["fft_rounds"] == 3
    record_property("runtime_probe", json.dumps(result, sort_keys=True))
