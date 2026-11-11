# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify native held-current assembly, complete rows and resident buffers."""

import dataclasses

import numpy as np
import pytest

from backends.test_band_plasma_parity import CASES, bound, case, options_for
from backends.test_radial_nonlinear import derivative
from gated_source import _gated_source
from sessions.runs import native_trial
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError
from solprop.translate.plasma_integration import design_running_integral

native = pytest.importorskip("solprop_native")

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_current_assembly_and_ledger_on_each_route(name, rule):
    grid, problem, run_request = case(name)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses, plasma_integral=design_running_integral(grid, rule)
        ),
    )
    # The established backend ladder accounts for scans, transforms and products;
    # its independent budget is fixed before evaluating the native source.
    allowance = bound(problem, grid, "native", 0)
    assert allowance > 0
    options = options_for(name)
    expected = derivative(NumpyRefBackend, problem, run_request, grid, options)
    actual = derivative(native.NativeBackend, problem, run_request, grid, options)
    scale = float(np.max(abs(expected)))
    assert scale > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=allowance * scale)
    assert np.count_nonzero(actual[..., ~grid.active_mask]) == 0
    backend = native.NativeBackend(problem, run_request, grid=grid, options=options)
    try:
        ledger = backend.memory_ledger()
        native_trial(backend, 0.0, 0.0, field=problem.initial.field0)
        native_trial(backend, 0.0, 0.0, field=np.zeros_like(problem.initial.field0))
        assert np.count_nonzero(backend._native_state.stage_to_host(0)) == 0
        # A second evaluation neither plans nor grows a resident source buffer.
        assert backend.memory_ledger() == ledger
    finally:
        backend.close()


def test_gated_comparison_is_refused_before_native_allocation():
    grid, problem, run_request = case("mode_averaged")
    with (
        _gated_source(),
        pytest.raises(BackendError, match="gated plasma benchmark requires numpy_ref"),
    ):
        native.NativeBackend(problem, run_request, grid=grid)


@pytest.mark.gpu
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_device_current_assembly_and_ledger(name, rule):
    test_current_assembly_and_ledger_on_each_route(name, rule)
