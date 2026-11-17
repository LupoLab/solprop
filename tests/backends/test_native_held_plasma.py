# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What native's held plasma source needs beyond the parity ladder.

The ladder itself -- one evaluation of the held current on every route under both
rules, against ``numpy_ref`` -- is the catalogue's (``tests/backends/plasma_rows.py``).
Here: native's source lands nothing outside the active band, and a second evaluation
neither plans nor grows a resident buffer.
"""

import numpy as np
import pytest

import parity_ladder
from backends.plasma_column_cases import CASES
from backends.plasma_rows import RULES, built_column, rows
from sessions.runs import native_trial

native = pytest.importorskip("solprop_native")
_testing = pytest.importorskip("solprop_native._core")._testing

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
        assert np.count_nonzero(_testing.stage_to_host(backend._workspace, 0)) == 0
        # A second evaluation neither plans nor grows a resident source buffer.
        assert backend.memory_ledger() == ledger
    finally:
        backend.close()


@pytest.mark.gpu
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", RULES)
def test_device_current_assembly_and_ledger(name, rule):
    (row,) = [row for row in rows() if row.id == f"plasma held current {name} {rule}"]
    parity_ladder.check(native.NativeBackend, row)
    test_the_held_current_lands_in_the_band_and_plans_once(name, rule)
