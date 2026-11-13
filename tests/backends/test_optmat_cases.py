# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the optmat cases need beyond the parity ladder: a batch and native's ledger.

The ladder itself -- every case on native against ``numpy_ref``, rung by rung, with
what each case carries -- is the catalogue's (``tests/backends/optmat_rows.py``).
"""

import pytest

from backends.test_batch_responses import compare
from optmat_parity_cases import build_case


@pytest.mark.medium
@pytest.mark.parametrize("name", ["n2_field", "ar_envelope"])
def test_a_batch_equals_its_members(name):
    # Within the rounding budget tests/backends/test_batch_responses.py holds its own
    # cases to: a lockstep batch's members run the same arithmetic as independent
    # runs. The reference's lockstep group: the native group runs each member as its
    # own native session (SEP 47 phase 5), with no shared arithmetic to check.
    case = build_case(name)
    compare(case.problem, case.run_request, case.grid, "numpy_ref", options=None)


@pytest.mark.medium
def test_native_holds_every_row_of_the_composition_leaf():
    # The rows are a view of n_terms x 2 n_to doubles and the per-stage decay rates
    # one per row and stage: both scale with the row count, so no fixed term count
    # is assumed (SEP 3, section 6). The two host backends read the problem's array.
    native = pytest.importorskip("solprop_native")
    case = build_case("air_floor")
    problem, run_request = case.problem, case.run_request
    backend = native.NativeBackend(
        problem, run_request, grid=case.grid, modes=case.modes
    )
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    rows = case.n_rows
    assert ledger["raman_undamped"] == rows * 2 * problem.grids.n_to * 8
    # Two slots, the step and the Adams sample, share one packed upload per trial.
    assert (
        ledger["stage_raman_rates"]
        == 2 * run_request.stepper.tableau.n_stages * rows * 8
    )
