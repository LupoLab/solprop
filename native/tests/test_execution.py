# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A workspace's CPU execution record and the built-in table (solprop SEP 46)."""

from __future__ import annotations

import dataclasses
import os
import platform

import pytest
from conftest import PhaseACase

import solprop_native as native
from solprop_native import _core
from solprop_native._execution import (
    SPIN_DIVISOR,
    SPIN_MAX_TEAM,
    ExecutionRecord,
    builtin,
)
from solprop_native._wait import SPIN_MILLISECONDS
from solprop_native.backend import (
    NativeBackend,
    _native_problem,  # pyright: ignore[reportPrivateUsage]
)

# The constants of core.hpp before SEP 46 moved them here (solprop PR #576), Linux and
# Windows, with the transforms' shape rule of solprop issue #587: every value of the
# table, frozen so that a change to one is deliberate.
LINUX = {
    "stage_threshold": 4096,
    "stage_full": 4096,
    "response_threshold": 32768,
    "response_full": 196608,
    "dense_threshold": 1048576,
    "dense_full": 1048576,
    "fft_threshold": 8192,
    "fft_full": 4194304,
    "raman_threshold": 65536,
    "raman_full": 262144,
    "copy_threshold": 65536,
    "copy_full": 65536,
    "flow_threshold": 32768,
    "flow_full": 32768,
    "modal_threshold": 65536,
    "modal_full": 65536,
    "adams_threshold": 8192,
    "adams_full": 65536,
    "response_full_samples": 65536,
    "response_full_columns": 40,
    "medium_threads": 4,
    "compact_threads": 8,
    "compact_max_fine_samples": 65536,
    "stable_kernel_team": True,
    "fft_thread_elements": 4096,
    "fft_long_row": 16384,
}
# macOS: its own stage and FFT starts and no stable kernel team.
APPLE = LINUX | {
    "stage_threshold": 8192,
    "stage_full": 8192,
    "fft_threshold": 32768,
    "stable_kernel_team": False,
    "fft_thread_elements": 32768,
    "fft_long_row": 65536,
}
# macOS on a spinning team of at most eight: every threshold and full size divided by
# 16 (ADR-073), each full size kept at least its threshold.
APPLE_SPINNING = APPLE | {
    "stage_threshold": 512,
    "stage_full": 512,
    "response_threshold": 2048,
    "response_full": 12288,
    "dense_threshold": 65536,
    "dense_full": 65536,
    "fft_threshold": 2048,
    "fft_full": 262144,
    "raman_threshold": 4096,
    "raman_full": 16384,
    "copy_threshold": 4096,
    "copy_full": 4096,
    "flow_threshold": 2048,
    "flow_full": 2048,
    "modal_threshold": 4096,
    "modal_full": 4096,
    "adams_threshold": 512,
    "adams_full": 4096,
    "fft_thread_elements": 2048,
    "fft_long_row": 4096,
}


@pytest.mark.parametrize("system", ["Linux", "Windows", "FreeBSD"])
def test_the_table_off_macos_is_the_former_constants(system: str) -> None:
    assert builtin(system).native() == LINUX
    # Off macOS a spin never divides.
    assert builtin(system, spin_milliseconds=1, width=4).native() == LINUX


def test_the_table_on_macos_divides_only_on_a_spinning_top_tier() -> None:
    assert builtin("Darwin").native() == APPLE
    assert builtin("Darwin", spin_milliseconds=0, width=6).native() == APPLE
    assert (
        builtin("Darwin", spin_milliseconds=1, width=SPIN_MAX_TEAM).native()
        == APPLE_SPINNING
    )
    assert (
        builtin("Darwin", spin_milliseconds=1, width=SPIN_MAX_TEAM + 1).native()
        == APPLE
    )
    assert SPIN_DIVISOR == 16


def test_a_users_fft_start_replaces_the_tables_undivided() -> None:
    # SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES: the start as given, never divided, its
    # full size the table's undivided one or the start where that is larger.
    spinning = builtin("Darwin", spin_milliseconds=1, width=6, fft_threshold=1000)
    assert (spinning.fft_threshold, spinning.fft_full) == (1000, 4194304)
    assert spinning.stage_threshold == APPLE_SPINNING["stage_threshold"]
    large = builtin("Linux", fft_threshold=1 << 20)
    assert (large.fft_threshold, large.fft_full) == (1 << 20, 1 << 22)


def test_the_process_record_is_its_platforms_table() -> None:
    # The import's inputs: the system, the wait it chose, the runtime's width and the
    # user's FFT start.
    start = os.environ.get("SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES", "").strip()
    expected = builtin(
        platform.system(),
        spin_milliseconds=SPIN_MILLISECONDS if native.WAIT_POLICY == "spin" else 0,
        width=native.build_info().runtime_threads,
        fft_threshold=int(start) if start else None,
    )
    assert native.BUILTIN_EXECUTION == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stage_threshold", 0),
        ("stage_full", 4095),
        ("fft_threshold", -1),
        ("medium_threads", 0),
        ("compact_threads", True),
        ("response_full_columns", 1.5),
        ("stable_kernel_team", 1),
        ("fft_thread_elements", 0),
        ("fft_long_row", -4),
    ],
)
def test_a_record_refuses_a_value_out_of_range(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field.split("_")[0]):
        builtin("Linux").updated({field: value})


def test_records_round_trip_and_name_their_differences() -> None:
    table = builtin("Linux")
    assert ExecutionRecord.from_native(table.native()) == table
    with pytest.raises(ValueError, match=r"missing \['stable_kernel_team'\]"):
        ExecutionRecord.from_native(
            {k: v for k, v in table.native().items() if k != "stable_kernel_team"}
        )
    with pytest.raises(ValueError, match=r"unknown \['extra'\]"):
        ExecutionRecord.from_native(table.native() | {"extra": 1})
    with pytest.raises(ValueError, match="unknown CPU execution settings"):
        table.updated({"divisor": 2})
    changed = table.updated({"stage_threshold": 1024, "medium_threads": 6})
    assert changed.differences(table) == {
        "stage_threshold": (1024, 4096),
        "medium_threads": (6, 4),
    }
    assert table.differences(table) == {}


def _payload(case: PhaseACase) -> dict[str, object]:
    problem, grid = case
    backend = NativeBackend(problem, grid=grid)
    try:
        return _native_problem(problem, backend._terms)  # pyright: ignore[reportPrivateUsage]
    finally:
        backend.close()


def test_a_state_reads_back_the_record_it_was_built_with(
    phase_a_case: PhaseACase,
) -> None:
    payload = _payload(phase_a_case)
    assert payload["cpu_execution"] == native.BUILTIN_EXECUTION.native()
    serial = dataclasses.replace(
        native.BUILTIN_EXECUTION,
        **{f"{name}_threshold": 1 << 40 for name in ("stage", "copy", "adams")},
        **{f"{name}_full": 1 << 40 for name in ("stage", "copy", "adams")},
    )
    first = _core.State(payload)
    second = _core.State(payload | {"cpu_execution": serial.native()})
    try:
        # Two live workspaces hold their own records (solprop SEP 46 section 7 item 7).
        assert first.cpu_execution == native.BUILTIN_EXECUTION.native()
        assert second.cpu_execution == serial.native()
        if native.build_info().cpu_size_policy_enabled:
            assert all(
                choice["threads"] == 1
                for name, choice in second.execution_sizing.items()
                if name.startswith(("kernels.stage.", "kernels.copy."))
            )
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cpu_execution": None}, "must be a dict"),
        ({"cpu_execution": {"stage_threshold": 1}}, "has 26 fields, got 1"),
        ("missing", "needs its CPU execution record"),
    ],
)
def test_a_state_refuses_a_missing_or_malformed_record(
    phase_a_case: PhaseACase, change: dict[str, object] | str, message: str
) -> None:
    payload = _payload(phase_a_case)
    if change == "missing":
        del payload["cpu_execution"]
    else:
        assert isinstance(change, dict)
        payload |= change
    with pytest.raises(ValueError, match=message):
        _core.State(payload)


def test_a_state_refuses_a_record_the_native_core_refuses(
    phase_a_case: PhaseACase,
) -> None:
    payload = _payload(phase_a_case)
    record = native.BUILTIN_EXECUTION.native()
    with pytest.raises(ValueError, match="positive threshold"):
        _core.State(payload | {"cpu_execution": record | {"dense_full": 1}})
    with pytest.raises(ValueError, match="positive response bounds"):
        _core.State(payload | {"cpu_execution": record | {"medium_threads": 0}})
    renamed = {k: v for k, v in record.items() if k != "adams_full"}
    with pytest.raises(ValueError, match="misses a field: adams_full"):
        _core.State(payload | {"cpu_execution": renamed | {"adams_fulll": 1}})


def test_a_backend_given_a_record_builds_with_it_and_records_it(
    phase_a_case: PhaseACase,
) -> None:
    problem, grid = phase_a_case
    given = native.BUILTIN_EXECUTION.updated({"medium_threads": 2})
    backend = NativeBackend(problem, grid=grid, cpu_execution=given)
    default = NativeBackend(problem, grid=grid)
    try:
        calibration = backend.metadata()["calibration"]
        assert isinstance(calibration, dict)
        assert calibration["workspace_source"] == "given"
        assert calibration["workspace"]["medium_threads"] == {
            "applied": 2,
            "built_in": native.BUILTIN_EXECUTION.medium_threads,
        }
        recorded = default.metadata()["calibration"]
        assert isinstance(recorded, dict)
        # The suite runs with SOLPROP_CALIBRATION=off: the built-in table.
        assert recorded["profile"] == "built-in"
        assert recorded["mode"] == "off"
    finally:
        backend.close()
        default.close()
