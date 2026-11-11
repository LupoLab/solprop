# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validate execution devices before calibration or numerical preparation."""

import dataclasses
import sys
from types import ModuleType, SimpleNamespace

import pytest

from solprop.backends import registry
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import ExecutionOptions
from solprop.calibration import first_use
from solprop.exceptions import BackendError
from solprop.stepper import PIController
from solprop.translate.precision import SINGLE

pytestmark = pytest.mark.real_auto


def _native_build(monkeypatch, device):
    module = ModuleType("solprop_native")
    monkeypatch.setattr(
        module,
        "build_info",
        lambda: SimpleNamespace(device_backend=device, device_count=1, blas_library=""),
        raising=False,
    )
    monkeypatch.setitem(sys.modules, "solprop_native", module)
    monkeypatch.setattr(registry, "_satisfied", lambda _entry: True)
    monkeypatch.setattr(registry, "_native_baseline_shortfall", lambda: None)


def _forbidden(*_args, **_kwargs):
    pytest.fail("a refused device reached calibration or numerical construction")


@pytest.mark.parametrize("name", ["numpy_ref", "julia"])
@pytest.mark.parametrize("device", ["cuda", "rocm", "metal"])
def test_unqualified_devices_refuse_without_entering_preparation(
    name, device, linear_problem, narrow_grid, monkeypatch
):
    problem, request = linear_problem(narrow_grid, save_planes=[0.001])
    monkeypatch.setattr(registry, "_construct", _forbidden)
    monkeypatch.setattr(first_use, "ensure", _forbidden)
    message = (
        "requires precision='single'"
        if name == "julia" and device == "metal"
        else "qualified devices: cpu"
    )
    with pytest.raises(BackendError, match=message):
        registry.select(name, problem, request, options=ExecutionOptions(device=device))


def test_metal_binary32_still_requires_a_qualified_extension(
    linear_problem, narrow_grid, monkeypatch
):
    problem, request = linear_problem(
        narrow_grid,
        save_planes=[0.001],
        precision=SINGLE,
        controller=PIController(rtol=1e-4, atol=1e-10),
    )
    monkeypatch.setattr(registry, "_construct", _forbidden)
    with pytest.raises(BackendError, match="qualified devices: cpu"):
        registry.select(
            "julia", problem, request, options=ExecutionOptions(device="metal")
        )


@pytest.mark.parametrize("built", ["none", "cuda", "hip"])
@pytest.mark.parametrize("requested", ["cpu", "cuda", "rocm", "metal"])
def test_native_explicit_selection_must_match_the_actual_build(
    built, requested, linear_problem, narrow_grid, monkeypatch
):
    _native_build(monkeypatch, built)
    problem, request = linear_problem(narrow_grid, save_planes=[0.001])
    effective = {"none": "cpu", "hip": "rocm"}.get(built, built)
    if requested == effective:
        reference = NumpyRefBackend(problem, request, grid=narrow_grid)
        calls = []
        monkeypatch.setattr(first_use, "ensure", lambda: calls.append("calibrate"))
        monkeypatch.setattr(registry, "_construct", lambda *_args, **_kwargs: reference)
        try:
            assert (
                registry.select(
                    "auto", problem, request, options=ExecutionOptions(device=requested)
                )
                is reference
            )
            assert calls == ["calibrate"]
        finally:
            reference.close()
    else:
        monkeypatch.setattr(first_use, "ensure", _forbidden)
        monkeypatch.setattr(registry, "_construct", _forbidden)
        with pytest.raises(BackendError, match=r"native.*device"):
            registry.select(
                "auto", problem, request, options=ExecutionOptions(device=requested)
            )


@pytest.mark.parametrize("device", [None, "cpu"])
def test_cpu_reference_metadata_records_effective_target(
    device, linear_problem, narrow_grid
):
    problem, request = linear_problem(narrow_grid, save_planes=[0.001])
    backend = registry.select(
        "numpy_ref",
        problem,
        request,
        grid=narrow_grid,
        options=ExecutionOptions(device=device),
    )
    try:
        assert backend.metadata()["device"] == "cpu"
    finally:
        backend.close()


def test_direct_reference_construction_also_refuses_a_gpu(linear_problem, narrow_grid):
    problem, request = linear_problem(narrow_grid, save_planes=[0.001])
    with pytest.raises(BackendError, match="numpy_ref executes on device 'cpu'"):
        NumpyRefBackend(problem, request, options=ExecutionOptions(device="cuda"))


@pytest.mark.parametrize(
    ("requirement", "device"),
    [
        ("cuda", "cuda"),
        ("hip", "rocm"),
        ("metal", "metal"),
        ("any", "cuda"),
        ("any", "rocm"),
        ("any", "metal"),
    ],
)
def test_job_guard_checks_explicit_julia_targets_without_selecting_them(
    requirement, device, monkeypatch
):
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, requirement)
    registry._check_required_device("julia", device)
    with pytest.raises(BackendError, match="julia device 'cpu'"):
        registry._check_required_device("julia")
    with pytest.raises(BackendError, match="numpy_ref"):
        registry._check_required_device("numpy_ref", device)


def test_injectable_capability_declaration_does_not_enable_the_public_julia_entry(
    linear_problem, narrow_grid, monkeypatch
):
    # Future extensions may declare targets after their qualification. This stand-in
    # tests the registry boundary only; the real declaration remains CPU-only.
    entry = registry.BACKENDS["julia"]
    assert entry.devices == frozenset({"cpu"})
    monkeypatch.setitem(
        registry.BACKENDS,
        "julia",
        dataclasses.replace(entry, devices=frozenset({"cpu", "cuda"})),
    )
    monkeypatch.setenv(registry.REQUIRE_DEVICE_SETTING, "cuda")
    monkeypatch.setattr(registry, "_satisfied", lambda _entry: True)
    problem, request = linear_problem(narrow_grid, save_planes=[0.001])
    backend = NumpyRefBackend(problem, request, grid=narrow_grid)
    seen = []

    def construct(name, _problem, _request, **keywords):
        seen.append((name, keywords["options"].device))
        return backend

    monkeypatch.setattr(registry, "_construct", construct)
    try:
        assert (
            registry.select(
                "julia", problem, request, options=ExecutionOptions(device="cuda")
            )
            is backend
        )
        assert seen == [("julia", "cuda")]
    finally:
        backend.close()
