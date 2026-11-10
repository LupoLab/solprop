# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compare Julia scalar decisions with Python on identical recorded readings."""

import math
import os
import struct
import subprocess
from pathlib import Path

import pytest
import tomli_w

from solprop.backends.julia._setup import select_setup
from solprop.backends.rk_core import RungeKuttaPolicy, decide, propose, reject, start_up
from solprop.exceptions import ConvergenceError
from solprop.stepper.controller import START_GROWTH, PIController

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]


def _canonical(value):
    if isinstance(value, bool):
        return str(value).lower()
    if math.isnan(value):
        return "nan"
    return struct.pack(">d", value).hex()


def _reference_call(kind, policy, norm, history):
    if kind == "propose":
        return propose(policy, 0.001, history)
    if kind == "start":
        return start_up(policy, 0.001, norm)
    if kind == "reject":
        return reject(policy, 0.001, norm)
    return decide(policy, norm, 0.0007, 0.001)


def _reference_cases():
    cases = []
    readings = (0.0, 0.2, 1.0, 2.0, 1e-200, 1e200, math.inf, math.nan, 0.99, 1.01)
    for lag in range(4):
        controller = PIController(rtol=1e-8, atol=0.0, lag=lag)
        policy = RungeKuttaPolicy(controller, 1e-9, 0.1, True)
        fields = {
            name: getattr(controller, name)
            for name in (
                "safety",
                "max_factor",
                "min_factor",
                "beta1",
                "beta2",
                "lag",
                "reject_free",
                "estimator_order",
            )
        } | {
            "start_growth": START_GROWTH,
            "dz_min": policy.dz_min,
            "dz_max": policy.dz_max,
            "auto_init": policy.auto_init,
        }
        for index, norm in enumerate(readings):
            history = [0.8] * (index % 6) + [norm]
            for kind, arguments in (
                ("propose", (0.001, history)),
                ("start", (0.001, norm)),
                ("reject", (0.001, norm)),
                ("decide", (norm, 0.0007, 0.001)),
            ):
                expected, error = [], ""
                try:
                    result = _reference_call(kind, policy, norm, history)
                    results = result if isinstance(result, tuple) else (result,)
                    expected = [_canonical(value) for value in results]
                except ConvergenceError:
                    error = "StepFloorError"
                except OverflowError:
                    error = "OverflowError"
                cases.append(
                    {
                        "kind": kind,
                        "policy": fields,
                        "history": history,
                        "arguments": list(
                            arguments[:1] if kind == "propose" else arguments
                        ),
                        "expected": expected,
                        "error": error,
                    }
                )
    # The proposal floor must fail transactionally at each site, and acceptance
    # must use the actual executed norm even in the reject-free variant.
    for case in list(cases):
        if case["kind"] == "decide":
            policy = dict(case["policy"], reject_free=True)
            norm, taken, proposal = case["arguments"]
            reference = RungeKuttaPolicy(
                PIController(rtol=1e-8, atol=0.0, reject_free=True), 1e-9, 0.1, True
            )
            results = decide(reference, norm, taken, proposal)
            cases.append(
                case | {"policy": policy, "expected": [_canonical(v) for v in results]}
            )
        elif case["kind"] == "reject" and case["arguments"][1] == math.inf:
            cases.append(
                case
                | {
                    "policy": dict(case["policy"], dz_min=0.0005),
                    "expected": [],
                    "error": "StepFloorError",
                }
            )
    for kind in ("start", "propose"):
        template = next(
            case
            for case in cases
            if case["kind"] == kind and case["history"][-1] == 1.0
        )
        cases.append(
            template
            | {
                "policy": dict(template["policy"], dz_min=0.0009, lag=0),
                "history": [1.0],
                "arguments": [0.001, 1.0] if kind == "start" else [0.001],
                "expected": [],
                "error": "StepFloorError",
            }
        )
    return cases


@pytest.mark.parametrize("threads", [1, 2])
def test_scalar_control_uses_identical_recorded_decisions(tmp_path, threads):
    setup_id = os.environ.get("SOLPROP_TEST_JULIA_SETUP_ID")
    assert setup_id, "the enabled Julia lane needs its published setup identity"
    record = select_setup(setup_id=setup_id)
    root = Path(__file__).resolve().parents[3]
    fixture = tmp_path / "readings.toml"
    fixture.write_text(tomli_w.dumps({"cases": _reference_cases()}), encoding="utf-8")
    project = dict(record.projects)["cpu"]
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("JULIA_")
    }
    environment.update(
        JULIA_DEPOT_PATH=os.pathsep.join((str(tmp_path / "depot"), str(record.depot))),
        JULIA_LOAD_PATH=os.pathsep.join((str(project), "@stdlib")),
        JULIA_PKG_OFFLINE="true",
        JULIA_CPU_TARGET=record.cpu_target,
    )
    completed = subprocess.run(
        [
            str(record.julia_executable),
            "--startup-file=no",
            "--history-file=no",
            "--check-bounds=yes",
            f"--threads={threads},0",
            f"--project={project}",
            str(root / "julia/test/control_oracle.jl"),
            str(fixture),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Identical recorded Python readings" in completed.stdout
