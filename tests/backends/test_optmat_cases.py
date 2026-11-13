# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the optmat cases need beyond the parity ladder: a batch and native's ledger.

The ladder itself -- every case on native against ``numpy_ref``, rung by rung, with
what each case carries -- is the catalogue's (``tests/backends/optmat_rows.py``).
"""

import numpy as np
import pytest

from backends.test_batch_responses import compare
from bounds import EPS, assert_within
from builders import launch_derivative
from optmat_parity_cases import build_case
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.passes import apply_hard_mask


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


@pytest.mark.medium
def test_the_lab_frame_sample_is_the_hoisted_derivative_on_the_masked_launch():
    # The reference's two evaluations of the launch's source -- its lab-frame sample
    # (the ladder's source rung) and its hoisted first stage (the derivative rung) --
    # run the same passes on the same scalars, so on the same input they agree to
    # rounding: a few ulps, normwise relative (measured: bit for bit). The sample is
    # taken after the anchor was hoisted elsewhere, so a stale anchor would show. The
    # 4.3e-13 they differed by was the input: the derivative evaluates the loaded state,
    # masked to the active band, and the unmasked sample the launch's out-of-band
    # content, which the masked launch must therefore change.
    case = build_case("ar_field")
    problem = case.problem
    backend = NumpyRefBackend(
        problem, case.run_request, grid=case.grid, modes=case.modes
    )
    try:
        hoisted = np.array(launch_derivative(backend, problem))
        launch = problem.initial.field0[None]
        masked = apply_hard_mask(launch, problem.grids.active_mask)
        assert np.any(masked != launch)
        backend._hoist(1e-3)
        sample = np.array(backend._adams_lab_source(masked, 0.0))
    finally:
        backend.close()
    assert_within(sample, hoisted, 8 * EPS, what="lab-frame sample")
