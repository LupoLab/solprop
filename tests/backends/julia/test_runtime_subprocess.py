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


def _run(record, directory, case, threads, *, mode="", terminating=False):
    output = directory / f"{case}-{mode or 'normal'}-{threads}.json"
    command = [sys.executable, str(PROBE), case, str(output), "--mode", mode]
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
    expected = {-signal.SIGTERM, 128 + signal.SIGTERM} if terminating else {0}
    assert process.returncode in expected, (
        f"{case}/{mode}/{threads}: exit {process.returncode}\n"
        f"stdout:\n{stdout}\nstderr:\n{stderr}"
    )
    assert "Exception ignored in" not in stderr, stderr
    if not terminating:
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
