# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native Adams march takes the boundary checks itself (SEP 45 sections 4.8, 7.3).

While the boundary monitor is busy it checks every committed step. The native march now
takes those checks inside a call, buffers the readings and returns only where solprop
must act; solprop replays the readings through its own monitor when it drains the call.
On one build the run must be the Python march's to the bit -- its steps, its fields and
the monitor's whole record -- while the march returns far less often than it checks.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
import pytest

from solprop.backends import available
from solprop.exceptions import BoundaryWarning
from solprop.monitor import BoundaryCheck
from solprop.stepper.multistep import Formulation, Multistep
from test_monitor import TRANGE, walker

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)
backend_module = pytest.importorskip("solprop_native.backend")
NativeAdamsMarch = backend_module._NativeAdamsMarch

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


def _run(build, march: str, monkeypatch) -> dict[str, Any]:
    """Run on one march; return its record, steps, field and the native calls."""
    calls: list[int] = []
    run_native = NativeAdamsMarch._native_run

    def counted(self, state, bounds):
        result = run_native(self, state, bounds)
        calls.append(int(result["reason"]))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(backend_module, "_ADAMS_MARCH", march)
        patch.setattr(NativeAdamsMarch, "_native_run", counted)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", BoundaryWarning)
            run = build().run(progress="none")
    outcome = run.outcome
    assert outcome is not None and outcome.execution is not None
    assert outcome.execution["adams_march"] == march
    return {
        "record": json.dumps(outcome.execution["boundary_monitor"], sort_keys=True),
        "monitor": outcome.execution["boundary_monitor"],
        "z": np.asarray(run["step_history/z"]),
        "h": np.asarray(run["step_history/h"]),
        "accepted": np.asarray(run["step_history/accepted"]),
        "field": run.plane_field(run.n_saved - 1, "fft"),
        "calls": calls,
        "warnings": [
            str(one.message)
            for one in caught
            if issubclass(one.category, BoundaryWarning)
        ],
    }


@pytest.mark.parametrize("formulation", ["exponential", "interaction"])
@pytest.mark.parametrize("scheme", ["nonlinear", "split_step"])
def test_a_busy_monitor_s_checks_stay_in_the_native_march(
    monkeypatch, scheme, formulation
):
    build = lambda: _busy(scheme, formulation)  # noqa: E731
    native = _run(build, "native", monkeypatch)
    host = _run(build, "host", monkeypatch)
    # The same steps, the same field and the same record, on one build (SEP 45 4.9).
    np.testing.assert_array_equal(native["z"], host["z"])
    np.testing.assert_array_equal(native["h"], host["h"])
    np.testing.assert_array_equal(native["accepted"], host["accepted"])
    np.testing.assert_array_equal(native["field"], host["field"])
    assert native["record"] == host["record"]
    assert native["warnings"] == host["warnings"]
    # The monitor was busy, and warned once, while the march seldom returned.
    monitor = native["monitor"]
    accepted = int(native["accepted"].sum())
    assert monitor["steps"] + monitor["skipped"] == accepted
    assert monitor["steps"] > accepted // 2
    guards = [one for one in native["warnings"] if "guard occupancy" in one]
    assert len(guards) == 1
    assert len(native["calls"]) < monitor["steps"] // 4
