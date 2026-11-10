# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A published profile applies at native import, and runs record it (solprop SEP 46
section 7 items 7 and 8). Each case runs in a fresh interpreter, since process
settings are fixed when the runtime starts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from solprop_native._store import Generation, Row, Store

REPORT = """
import json, solprop_native as n
from solprop_native.backend import NativeBackend
from solprop.grid import EnvelopeGrid
from solprop.nonlinear.gnlse import problem_from
from solprop.pulses import SechPulse
from solprop.stepper import PIController
from solprop.translate.problem import assemble
grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
model = problem_from([0.0, 0.0, -11.83e-27], omega_ref=grid.omega0, gamma=0.11,
                     shock=False)
field = SechPulse(wavelength=835e-9, width=100e-15, peak_power=1e4).build(grid)
problem, run_request = assemble(
    grid, model.mode, [model.kerr], field, save_planes=[1e-3], density=1.0,
    shock=False, prefactor_index=1.0, controller=PIController(rtol=1e-8, atol=1e-16),
    dz_max=1e-3, dz_init=1e-4)
backend = NativeBackend(problem, run_request, grid=grid)
calibration = backend.metadata()["calibration"]
backend.close()
print(json.dumps({
    "fingerprint": n.PROFILE.fingerprint and n.PROFILE.fingerprint.digest,
    "build": n.PROFILE.fingerprint and n.PROFILE.fingerprint.build,
    "allocation": n.PROFILE.allocation and n.PROFILE.allocation.record(),
    "threads": n.build_info().runtime_threads,
    "wait": n.WAIT_POLICY,
    "stage": n.EXECUTION.stage_threshold,
    "source": n.EXECUTION_SOURCE,
    "calibration": calibration,
}))
"""


def _report(store: Path, **environ: str) -> dict[str, Any]:
    clean = {
        name: value
        for name, value in os.environ.items()
        if name
        not in {
            "SOLPROP_CALIBRATION",
            "SOLPROP_THREADS",
            "SOLPROP_NATIVE_THREADS",
            "OMP_WAIT_POLICY",
            "GOMP_SPINCOUNT",
            "KMP_BLOCKTIME",
            "SOLPROP_NATIVE_WAIT",
            "SOLPROP_CALIBRATION_PATH",
        }
    }
    completed = subprocess.run(
        [sys.executable, "-c", REPORT],
        capture_output=True,
        text=True,
        check=False,
        env=clean | {"SOLPROP_CACHE_DIR": str(store)} | environ,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def baseline(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """The process with an empty store: the built-in table."""
    return _report(tmp_path_factory.mktemp("empty"))


def _publish(store: Path, baseline: dict[str, Any], row: Row) -> None:
    fingerprint = baseline["fingerprint"]
    assert isinstance(fingerprint, str)
    Store(store).publish(
        Generation(
            fingerprint=fingerprint, name="g1", created="2026-11-08", rows=(row,)
        )
    )


def test_without_a_profile_the_process_runs_the_built_in_table(
    baseline: dict[str, Any],
) -> None:
    assert baseline["source"] == "no profile for this machine and build"
    # The fingerprint holds the build: its record and the extension's digest.
    build = baseline["build"]
    assert isinstance(build, dict)
    assert build["extension"] is not None
    assert isinstance(build["record"], dict)
    calibration = baseline["calibration"]
    assert isinstance(calibration, dict)
    assert calibration["profile"] == "built-in"


def test_an_exact_row_sets_the_pool_the_wait_and_the_workspace(
    tmp_path: Path, baseline: dict[str, Any]
) -> None:
    allocation = cast(dict[str, object], baseline["allocation"])
    # The row's width within the process's budget, which a small runner holds to one;
    # its workspace settings were fitted under the width the process then runs.
    width = min(2, baseline["threads"])
    row = Row(
        allocation=allocation,
        level="standard",
        process={"threads": 2, "wait": "passive"},
        workspace={"stage_threshold": 1234, "stage_full": 1234},
        fitted_under={"threads": width, "wait": "passive"},
    )
    _publish(tmp_path, baseline, row)
    applied = _report(tmp_path)
    assert (applied["threads"], applied["wait"]) == (width, "passive")
    assert (applied["stage"], applied["source"]) == (1234, "profile")
    calibration = cast(dict[str, Any], applied["calibration"])
    profile = cast(dict[str, Any], calibration["profile"])
    assert profile["generation"] == "g1"
    assert profile["exact"] is True
    # Read back from the workspace, beside the built-in value.
    assert calibration["workspace"]["stage_threshold"]["applied"] == 1234
    assert calibration["process"] == {"threads": width, "wait": "passive"}

    # SOLPROP_CALIBRATION=off: the table, whatever the store holds.
    off = _report(tmp_path, SOLPROP_CALIBRATION="off")
    assert off["threads"] == baseline["threads"]
    assert off["stage"] == baseline["stage"]

    # The user's thread count wins, and settings fitted on more threads give way; on a
    # one-CPU runner the row was fitted on the one thread the user asks for.
    narrowed = _report(tmp_path, SOLPROP_THREADS="1")
    assert narrowed["threads"] == 1
    if width > 1:
        assert narrowed["stage"] == baseline["stage"]
        assert "fitted on 2 threads" in str(narrowed["source"])
    else:
        assert narrowed["stage"] == 1234

    # The user's wait wins, and settings fitted under passive waiting give way.
    spinning = _report(tmp_path, OMP_WAIT_POLICY="active")
    assert spinning["wait"] == "environment: OMP_WAIT_POLICY=active"
    assert spinning["stage"] == baseline["stage"]


def test_a_shared_read_only_directory_supplies_a_profile(
    tmp_path: Path, baseline: dict[str, Any]
) -> None:
    allocation = cast(dict[str, object], baseline["allocation"])
    shared = tmp_path / "shared"
    _publish(
        shared,
        baseline,
        Row(
            allocation=allocation,
            level="thorough",
            workspace={"medium_threads": 3},
            fitted_under={"wait": baseline["wait"]},
        ),
    )
    applied = _report(tmp_path / "mine", SOLPROP_CALIBRATION_PATH=str(shared))
    assert applied["source"] == "profile"
    calibration = applied["calibration"]
    assert isinstance(calibration, dict)
    assert calibration["workspace"]["medium_threads"]["applied"] == 3
