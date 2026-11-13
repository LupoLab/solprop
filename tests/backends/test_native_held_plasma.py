# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What native's held and gated plasma sources need beyond the parity ladder.

The ladder itself -- one evaluation of the held current on every route under both
rules, and the private gated form's evaluation and march, against ``numpy_ref`` -- is
the catalogue's (``tests/backends/plasma_rows.py``). Here: native's source lands
nothing outside the active band, a second evaluation neither plans nor grows a resident
buffer, native runs the gated form when it is asked to, the gated form's record and
ledger, and its refusal on an envelope plasma.
"""

import numpy as np
import pytest

import parity_ladder
from backends.host_cases_envelope_plasma import build_envelope_mixture, geometry_case
from backends.plasma_column_cases import CASES
from backends.plasma_rows import PREIONISED, RULES, built_column, one_evaluation, rows
from gated_source import _gated_source
from sessions.runs import native_trial
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.exceptions import BackendError

native = pytest.importorskip("solprop_native")

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", RULES)
def test_the_held_current_lands_in_the_band_and_plans_once(name, rule):
    built = built_column(name, rule)
    actual = parity_ladder.derivative(native.NativeBackend, built)
    assert np.count_nonzero(actual[..., ~built.grid.active_mask]) == 0
    problem = built.problem
    backend = native.NativeBackend(
        problem, built.run_request, grid=built.grid, options=built.options
    )
    try:
        ledger = backend.memory_ledger()
        native_trial(backend, 0.0, 0.0, field=problem.initial.field0)
        native_trial(backend, 0.0, 0.0, field=np.zeros_like(problem.initial.field0))
        assert np.count_nonzero(backend._native_state.stage_to_host(0)) == 0
        # A second evaluation neither plans nor grows a resident source buffer.
        assert backend.memory_ledger() == ledger
    finally:
        backend.close()


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", RULES)
def test_native_runs_the_gated_form_when_asked(name, rule):
    """Native's gated source lands nothing outside the band, and it is not the held one.

    The comparison discriminates: the production held current differs from the gated
    form by 5.4e-7 of the source or more on these cases (re-measured: 5.4e-7 on the
    radial band case), a thousand times the allowance and more (7,563 times on the
    planar band case, the least), so a native run that ignored the flag would fail the
    gated rows' ``derivative`` rung.
    """
    built = built_column(name, rule, preionisation=PREIONISED)
    allowance = one_evaluation(built)
    with _gated_source():
        expected = parity_ladder.derivative(NumpyRefBackend, built)
        gated = parity_ladder.derivative(native.NativeBackend, built)
    held = parity_ladder.derivative(native.NativeBackend, built)
    assert np.count_nonzero(gated[..., ~built.grid.active_mask]) == 0
    scale = float(np.max(abs(expected)))
    assert scale > 0
    assert float(np.max(abs(held - gated))) > 1e3 * allowance * scale


def test_the_gated_benchmark_names_itself_and_holds_no_held_streams():
    """The record names the private form, and no held-current stream is allocated."""
    built = built_column("mode_averaged", "trapezoid", preionisation=PREIONISED)
    with _gated_source():
        backend = native.NativeBackend(
            built.problem, built.run_request, grid=built.grid
        )
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
@pytest.mark.parametrize("rule", RULES)
def test_device_current_assembly_and_ledger(name, rule):
    (row,) = [row for row in rows() if row.id == f"plasma held current {name} {rule}"]
    parity_ladder.check(native.NativeBackend, row)
    test_the_held_current_lands_in_the_band_and_plans_once(name, rule)
