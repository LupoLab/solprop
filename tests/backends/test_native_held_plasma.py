# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify native held-current assembly, complete rows and resident buffers."""

import dataclasses

import numpy as np
import pytest

from backends.host_cases_envelope_plasma import build_envelope_mixture, geometry_case
from backends.test_band_plasma_parity import CASES, bound, case, march, options_for
from backends.test_radial_nonlinear import derivative
from gated_source import _gated_source
from sessions.runs import native_trial
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError
from solprop.profiles import Constant
from solprop.stepper import FixedSchedule
from solprop.translate.linop import profile_table
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


def _gated_case(name, rule, steps=None):
    """Return a case with field plasma, its rule, ten per cent pre-ionisation and,
    for a march, a fixed schedule of `steps` over the parity ladder's window."""
    grid, problem, run_request = case(name)
    # Pre-ionised, so the free current, and with it the gated integral, is live
    # from the first sample rather than a rounding-level population.
    initial = profile_table(Constant(0.1), *problem.linear.z_span)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            plasma_integral=design_running_integral(grid, rule),
            plasma_channels=tuple(
                dataclasses.replace(channel, preionisation=initial)
                for channel in problem.responses.plasma_channels
            ),
        ),
    )
    if steps is not None:
        original_end = float(run_request.stepper.save_planes[-1])
        # The planar fixture's own window is 80 Adams steps long; the parity ladder's
        # 1e-7 m window keeps its cold-plasma phase below 0.028 rad.
        end = 1e-7 if name == "planar" else original_end
        run_request = run_request.with_stepper(
            dataclasses.replace(
                run_request.stepper,
                save_planes=np.asarray(run_request.stepper.save_planes)
                * (end / original_end),
                schedule=FixedSchedule.uniform(0.0, end, end / steps),
            )
        )
    return grid, problem, run_request


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_the_gated_benchmark_source_matches_the_reference(name, rule):
    """One nonlinear evaluation of the private gated form, against ``numpy_ref``.

    The gated form replaces the held current's two transforms and products by one
    filter and one trapezoid scan of the same current rows and a product with the
    window: at most ``gamma(2 width n_to + 2)`` of the integral's magnitudes, inside
    the ladder's per-call ``gamma(12 width n_to)``, so the held form's budget, fixed
    before the evaluation, holds it (``docs/dev/testing.md`` section 3).
    """
    grid, problem, run_request = _gated_case(name, rule)
    allowance = bound(problem, grid, "native", 0)
    assert allowance > 0
    options = options_for(name)
    with _gated_source():
        expected = derivative(NumpyRefBackend, problem, run_request, grid, options)
        actual = derivative(native.NativeBackend, problem, run_request, grid, options)
    held = derivative(native.NativeBackend, problem, run_request, grid, options)
    scale = float(np.max(abs(expected)))
    assert scale > 0
    np.testing.assert_allclose(actual, expected, rtol=0, atol=allowance * scale)
    assert np.count_nonzero(actual[..., ~grid.active_mask]) == 0
    # The comparison discriminates: the production held current differs from the
    # gated form by 5.4e-7 of the source or more on these cases (measured), a
    # thousand times the allowance and more, so a native run that ignored the flag
    # would fail above.
    assert float(np.max(abs(held - actual))) > 1e3 * allowance * scale


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
@pytest.mark.parametrize("steps", [1, 8])
def test_the_gated_benchmark_march_matches_the_reference(name, rule, steps):
    """One and eight fixed steps of the gated form, against ``numpy_ref``.

    The bound is the band-rule ladder's at seven source evaluations a step, fixed
    before the march, as for the one evaluation above.
    """
    grid, problem, run_request = _gated_case(name, rule, steps)
    limit = bound(problem, grid, "native", steps)
    assert limit > 0
    options = options_for(name)
    with _gated_source():
        expected, _ = march(
            NumpyRefBackend, problem, run_request, grid, steps=steps, options=options
        )
        actual, _ = march(
            native.NativeBackend,
            problem,
            run_request,
            grid,
            steps=steps,
            options=options,
        )
    scale = float(np.max(np.abs(expected)))
    assert scale > 0
    assert float(np.max(np.abs(actual - expected))) <= limit * scale


def test_the_gated_benchmark_names_itself_and_holds_no_held_streams():
    """The record names the private form, and no held-current stream is allocated."""
    grid, problem, run_request = _gated_case("mode_averaged", "trapezoid")
    with _gated_source():
        backend = native.NativeBackend(problem, run_request, grid=grid)
    try:
        record = backend.metadata()
        assert record["_benchmark_plasma_source"] == "gated_nonresumable"
        assert record["plasma_source"] == {
            "form": "gated_benchmark",
            "resumable": False,
        }
        ledger = backend.memory_ledger()
        assert ledger["held_current_gj"] == ledger["held_current_edge"] == 0
        assert ledger["gated_current_prefix"] > 0
    finally:
        backend.close()


def test_the_gated_benchmark_is_refused_by_name_on_an_envelope_plasma():
    """The gated form is a field column's: native names the refusal before it builds."""
    problem, run_request, grid = geometry_case(build_envelope_mixture(), "modal", 1)
    with (
        _gated_source(),
        pytest.raises(BackendError, match=r"gated plasma benchmark.*envelope plasma"),
    ):
        native.NativeBackend(problem, run_request, grid=grid)


@pytest.mark.gpu
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
def test_device_current_assembly_and_ledger(name, rule):
    test_current_assembly_and_ledger_on_each_route(name, rule)
