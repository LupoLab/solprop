# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Adams march takes the boundary checks itself (SEP 45 sections 4.8, 7.3).

While the boundary monitor is busy it checks every committed step. The native session
takes those checks inside its march and returns to Python only where solprop must act,
so the run returns far less often than it checks. Its decisions are held to the
recorded identical-input traces (``tests/backends/adams_traces.py``); here the run is
the same twice -- its steps, its field and the monitor's whole record, bit for bit on
one thread -- and warns as the reference's independent march does on the same walk,
its field within a hundred times the tolerance of the reference's (two adaptive runs,
SEP 25 class T). Until SEP 47's phase 5 it was held bit for bit to solprop's Python
march, a private switch retired with the transitional bridge.
"""

from __future__ import annotations

import dataclasses
import json
import os
import warnings
from typing import Any

import numpy as np
import pytest

from solprop.backends import available
from solprop.backends.events import Progress
from solprop.exceptions import BoundaryWarning
from solprop.monitor import BoundaryCheck
from solprop.stepper.multistep import Formulation, Multistep
from test_monitor import TRANGE, walker

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
NativeSession = pytest.importorskip("solprop_native.session").NativeSession

BUSY = BoundaryCheck(stop_factor=1000.0)
"""The default budget, so faint light at the edge keeps the monitor busy and warns once,
and a stop far beyond what the walk reaches."""


def _busy(scheme: str, formulation: Formulation) -> Any:
    return walker(
        0.65 * TRANGE,
        check=BUSY,
        backend="native",
        integrator=Multistep(formulation=formulation),
        absorber_scheme=scheme,
        n_save=6,
    )


def _run(build, monkeypatch, backend: str = "native") -> dict[str, Any]:
    """Run on one backend; return its record, steps, field and returns to Python."""
    calls: list[int] = []
    advance = NativeSession.advance

    def counted(self, *args, **keywords):
        batch = advance(self, *args, **keywords)
        # A bounded yield's progress batch is the clock's or the buffer's return,
        # never a check's.
        if not all(isinstance(event, Progress) for event in batch.events):
            calls.append(len(batch.events))
        return batch

    simulation = build()
    simulation.settings = dataclasses.replace(simulation.settings, backend=backend)
    rtol = simulation._run_request().stepper.controller.rtol
    with monkeypatch.context() as patch:
        patch.setattr(NativeSession, "advance", counted)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", BoundaryWarning)
            run = simulation.run(progress="none")
    outcome = run.outcome
    assert outcome is not None and outcome.execution is not None
    return {
        "march": outcome.execution.get("adams_march"),
        "rtol": float(rtol),
        "record": json.dumps(outcome.execution["boundary_monitor"], sort_keys=True),
        "monitor": outcome.execution["boundary_monitor"],
        "z": np.asarray(run["step_history/z"]),
        "h": np.asarray(run["step_history/h"]),
        "accepted": np.asarray(run["step_history/accepted"]),
        "field": run.plane_field(run.n_saved - 1, "fft"),
        "planes": run.n_saved,
        "calls": calls,
        "warnings": [
            str(one.message)
            for one in caught
            if issubclass(one.category, BoundaryWarning)
        ],
    }


def _one_thread() -> bool:
    return "1" in {
        os.environ.get("SOLPROP_NATIVE_THREADS"),
        os.environ.get("SOLPROP_THREADS"),
    }


@pytest.mark.parametrize("formulation", ["exponential", "interaction"])
@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_a_busy_monitor_s_checks_stay_in_the_native_march(
    monkeypatch, scheme, formulation
):
    build = lambda: _busy(scheme, formulation)  # noqa: E731
    native = _run(build, monkeypatch)
    again = _run(build, monkeypatch)
    reference = _run(build, monkeypatch, "numpy_ref")
    assert native["march"] == "native"
    if _one_thread():
        for name in ("z", "h", "accepted", "field"):
            np.testing.assert_array_equal(native[name], again[name], name)
        assert native["record"] == again["record"]
    assert native["warnings"] == again["warnings"] == reference["warnings"]
    peak = float(np.max(np.abs(reference["field"])))
    departure = float(np.max(np.abs(native["field"] - reference["field"])))
    assert departure <= 100.0 * native["rtol"] * peak
    # The monitor was busy, and warned once, while the march seldom returned.
    monitor = native["monitor"]
    accepted = int(native["accepted"].sum())
    assert monitor["steps"] + monitor["skipped"] == accepted
    assert monitor["steps"] > accepted // 2
    guards = [one for one in native["warnings"] if "guard occupancy" in one]
    assert len(guards) == 1
    # The session returns at its planes, its one warning and its end alone.
    assert len(native["calls"]) == native["planes"] + len(native["warnings"]) + 1
