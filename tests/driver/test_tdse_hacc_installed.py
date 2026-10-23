# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the real haCC provider surface when it is installed (SEP 20, 7.1).

Skipped unless Pyroraptor at the pinned revision is importable. The check runs in a
subprocess with one provider and one OpenMP thread, so the provider's native runtime
never shares this test process: it resolves both presets through ``hacc_preset``,
checks their identities and SI conversion against the records the fake-provider tests
use, and runs one call of the provider's own tiny consumer-example haCC(1) basis
through the binding (C3 to C5). It qualifies nothing about the atomic physics.
"""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.medium

_SCRIPT = """
import dataclasses, json, warnings
import numpy as np
from solprop.driver import tdse
from solprop.exceptions import TDSEUnqualifiedModelWarning
from solprop.grid import FieldGrid
from solprop.nonlinear.tdse import TDSEAtomicWindow, TDSEExecution
from solprop.translate.tdse import TDSEDescriptor
from tests.hacc_cases import hacc_response, preset_basis, tiny_basis

report = {"presets": {}}
for ions in (1, 2):
    for variant in ("standard", "fast"):
        preset = tdse.hacc_preset(ions, variant)
        expected = preset_basis(ions, variant)
        report["presets"][f"{ions}-{variant}"] = {
            "identity": preset.identity,
            "equal": dataclasses.replace(preset.basis, source=expected.source)
            == expected,
            "variant": preset.basis.variant,
            "step_margin": preset.step_margin,
            "accuracy": preset.accuracy,
            "window": preset.window,
            "step": preset.max_atomic_step,
        }
grid = FieldGrid.from_limits(1e-3, 800e-9, (400e-9, 1600e-9), 10e-15)
response = hacc_response(tiny_basis(), atomic_window=TDSEAtomicWindow(8e-15, 1e-15))
descriptor = TDSEDescriptor.from_response(response, grid)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    binding = tdse.prepare_binding(descriptor, TDSEExecution(max_batch_rows=2), 3)
report["warnings"] = [one.category.__name__ for one in caught]
t = grid.t_fine
field = 2e9 * np.exp(-((t / 3e-15) ** 2)) * np.cos(2.35e15 * t)
binding.reserve(1)
result = binding.evaluate(
    np.stack([field, 0.5 * field, 0.25 * field]), z=0.0, purpose="startup"
)
window = descriptor.atomic_window
before = result.total_current[:, : window.first]
after = result.total_current[:, window.last + 1 :]
outside = np.concatenate([before, after], axis=1)
metadata = binding.metadata()
report.update(
    shape=list(result.total_current.shape),
    finite=bool(np.all(np.isfinite(result.total_current))),
    peak=float(np.max(np.abs(result.total_current))),
    outside=float(np.max(np.abs(outside))),
    closure=metadata["hacc"]["closure"],
    attempted=metadata["ledger"]["attempted"],
    plans=sorted(metadata["plans"]),
    native_outputs=metadata["hacc"]["native_outputs"],
)
binding.close()

# A mixed gauge with no edge from the third onward below Rg (here Rg = 1 bohr) has
# no escaped-electron closure. The binding detects that from the provider's refusal
# of a compact order-one plan, so pin the message it relies on, then check that the
# binding takes the full-output route for such a basis.
from pyroraptor import hacc
from tests.hacc_cases import A0

control = tiny_basis(gauge_radius=1.0 * A0)
atom, native, gauge = tdse._hacc_objects(tdse._load_provider(), hacc, control)
with hacc.prepare(atom, native, gauge=gauge, cut_energy=control.cut_energy) as model:
    try:
        hacc.ResponsePlan(
            model, rows=1, spacing=float(grid.dt_fine), samples=16,
            max_step_au=0.05, shadow_order=1, outputs="total_current",
        )
    except ValueError as error:
        report["closure_message"] = str(error)
    else:
        report["closure_message"] = None
response = hacc_response(control, atomic_window=TDSEAtomicWindow(8e-15, 1e-15))
descriptor = TDSEDescriptor.from_response(response, grid)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    binding = tdse.prepare_binding(descriptor, TDSEExecution(), 1)
binding.reserve(1)
result = binding.evaluate(field[None, :], z=0.0, purpose="startup")
metadata = binding.metadata()
report["control"] = {
    "closure": metadata["hacc"]["closure"],
    "native_outputs": metadata["hacc"]["native_outputs"],
    "plan_outputs": [plan["outputs"] for plan in metadata["plans"].values()],
    "warning": [str(one.message) for one in caught],
    "finite": bool(np.all(np.isfinite(result.total_current))),
}
binding.close()
print(json.dumps(report))
"""


def _pinned_provider_installed() -> bool:
    if importlib.util.find_spec("pyroraptor") is None:
        return False
    from solprop.driver import tdse  # noqa: PLC0415 - only after the spec check

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pyroraptor; print(pyroraptor.build_info().source_revision)",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        env=_environment(),
        check=False,
    )
    revision = probe.stdout.strip().splitlines()[-1] if probe.stdout.strip() else ""
    return probe.returncode == 0 and tdse.pinned_revision().startswith(revision or "-")


def _environment() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        **os.environ,
        "PYTHONPATH": str(root),
        "PYRORAPTOR_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OMP_PROC_BIND": "false",
    }


@pytest.mark.skipif(
    not _pinned_provider_installed(),
    reason="Pyroraptor at the pinned revision is not installed",
)
def test_installed_provider_resolves_presets_and_runs_a_tiny_custom_call():
    completed = subprocess.run(
        [sys.executable, "-W", "ignore::UserWarning", "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=600,
        env=_environment(),
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    presets = report["presets"]
    assert presets["1-standard"]["identity"] == "hacc-helium-1-v1-28f0eae146b1d665"
    assert presets["2-standard"]["identity"] == "hacc-helium-5-v1-5853d5d50c187aa9"
    assert presets["1-fast"]["identity"] == "hacc-helium-1-fast-v1-e38292e1b30729c1"
    assert presets["2-fast"]["identity"] == "hacc-helium-5-fast-v1-e741d3ce524adc7e"
    assert all(record["equal"] for record in presets.values())
    for name, record in presets.items():
        variant = name.split("-")[1]
        assert record["variant"] == variant
        assert record["step_margin"] == (0.62 if variant == "standard" else 0.9)
    # The provider's fast records add its measured 1.5e-4 to the standard 0.0122;
    # both are sums of the same two binary64 literals there and in the fake.
    assert presets["1-fast"]["accuracy"] == 0.0122 + 1.5e-4
    assert report["warnings"] == ["TDSEUnqualifiedModelWarning"]
    assert report["shape"] == [3, 64] and report["finite"]
    assert report["peak"] > 0 and report["outside"] == 0.0
    assert report["closure"] is True and report["native_outputs"] == "total_current"
    assert report["attempted"] == 2 and report["plans"] == ["1", "2"]
    # The binding's closure detection depends on this provider message: a rewording
    # must fail here rather than turn a research control into a refusal.
    assert "needs the escaped-electron closure" in report["closure_message"]
    control = report["control"]
    assert control["closure"] is False
    assert control["native_outputs"] == "full" and control["plan_outputs"] == ["full"]
    assert control["finite"]
    assert any("escaped-electron closure" in message for message in control["warning"])
