# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Require direct native preparation to match its installed execution target."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from solprop.backends.session import ExecutionOptions
from solprop.exceptions import BackendError

from solprop_native import NativeBackend, backend, build_info

if TYPE_CHECKING:
    from conftest import PhaseACase


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_build_device_is_reported_and_accepted(
    phase_a_case: PhaseACase, explicit: bool
) -> None:
    """Exercise the actual compiled build, preserving its ordinary digest check."""
    problem, request, grid = phase_a_case
    built = str(build_info().device_backend)
    actual = {"none": "cpu", "hip": "rocm"}.get(built, built)
    # Resolving through the public guard returns the canonical Literal type too.
    selected = ExecutionOptions().check_device("native", actual)
    owner = NativeBackend(
        problem,
        request,
        grid=grid,
        options=ExecutionOptions(device=selected if explicit else None),
    )
    try:
        assert owner.metadata()["device"] == actual
        assert owner.metadata()["device_backend"] == built
    finally:
        owner.close()


def test_direct_mismatch_refuses_before_numerical_admission(
    phase_a_case: PhaseACase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metal never matches a native build and must fail before preparing any state."""
    problem, request, _grid = phase_a_case

    def forbidden(*_arguments: object, **_keywords: object) -> None:
        pytest.fail("a mismatched device reached numerical admission")

    monkeypatch.setattr(backend, "_refuse_unsupported", forbidden)
    with pytest.raises(BackendError, match="device='metal' was requested"):
        NativeBackend(problem, request, options=ExecutionOptions(device="metal"))
