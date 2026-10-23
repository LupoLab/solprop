# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's gas Raman responses on the compiled backends against numpy_ref.

SEP 3, Phase 2, after the handoff of Phase 1c: dry air's 126-row composition leaf on a
pressure gradient to the vacuum floor and, on density paths to exact vacuum, with N2's
and O2's decay rates zero on the last plane; hydrogen at constant density and under the
reference-density shift with its Dicke cap finite at vacuum; nitrogen's 72 isotropic
and rank-two rows on two polarisations. The bounds are ``tests/optmat_parity.py``'s,
derived from the operation count before any measurement; the Raman rows enter them
as one more rounding per row and sample.

Medium tier: ten fixed steps per case, a second or two each with its translation.

Measured on the development machine against bounds of order 1e-10 to 1e-8, the worst
deviation of any field rung was 7.1e-16 (the first stage of
``n2_vector_envelope``) and of any statistic 2.7e-14
(``quadrature_error``).
"""

import dataclasses

import numpy as np
import pytest

from optmat_parity import check_ladder, derivative, deviation, stage_bound
from optmat_parity_cases import build_case
from solprop.backends.numpy_ref import NumpyRefBackend

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

CASES = (
    "air_floor",
    "air_vacuum",
    "h2_constant",
    "h2_reference",
    "n2_vector",
    "n2_vector_envelope",
)

DETECTION = 1e3
"""How far above the nonlinear rung's bound the Raman term must reach."""


@pytest.mark.medium
@pytest.mark.parametrize("name", CASES)
def test_every_rung_agrees_with_the_reference(name, equivalence_backend_class):
    # air_vacuum's three independent density paths take the z table, which the native
    # CPU build evaluates natively (SEP 29 section 6.6), as numba_cpu does on the host.
    check_ladder(name, equivalence_backend_class)


@pytest.mark.parametrize("name", CASES)
def test_the_first_stage_sees_the_raman_leaf(name):
    # The leaf's share of N(a_0), against the same problem with its rows zeroed on
    # the reference: far above the nonlinear rung's bound.
    case = build_case(name)
    responses = case.problem.responses
    silent = dataclasses.replace(
        case.problem,
        responses=dataclasses.replace(
            responses,
            raman_terms=tuple(
                dataclasses.replace(term, undamped=np.zeros_like(term.undamped))
                for term in responses.raman_terms
            ),
        ),
    )
    share = deviation(
        derivative(NumpyRefBackend, dataclasses.replace(case, problem=silent)),
        derivative(NumpyRefBackend, case),
    )
    assert share > DETECTION * stage_bound(case)


@pytest.mark.parametrize(
    ("name", "rows", "zero_at_vacuum"),
    [
        ("air_floor", 126, False),
        ("air_vacuum", 126, True),
        ("h2_reference", 4, False),
        ("n2_vector", 72, True),
        ("n2_vector_envelope", 72, True),
    ],
)
def test_the_leaves_are_the_handoffs(name, rows, zero_at_vacuum):
    # What the parity runs carry: the row counts of Phase 1c, mixed isotropic and
    # rank-two rows, and the decay paths at the vacuum end -- zero for N2's and O2's
    # groups, H2's Dicke cap finite, the floor's small and positive.
    case = build_case(name)
    (leaf,) = case.problem.responses.raman_terms
    assert leaf.n_terms == rows
    assert {tuple(weights) for weights in leaf.tensor_weights} == {
        (1.0, 0.0),
        (0.25, 0.75),
    }
    end = case.problem.stepper.save_planes[-1]
    rates = np.array([float(table(end)) for table in leaf.decay_rates])
    if zero_at_vacuum:
        assert leaf.density is not None
        assert float(leaf.density(end)) == 0.0
        assert np.all(rates == 0.0)
    else:
        assert np.all(rates > 0.0)


@pytest.mark.medium
def test_native_holds_every_row_of_the_composition_leaf():
    # The rows are a view of n_terms x 2 n_to doubles and the per-stage decay rates
    # one per row and stage: both scale with the row count, so no fixed term count
    # is assumed (SEP 3, section 6). The two host backends read the problem's array.
    native = pytest.importorskip("solprop_native")
    case = build_case("air_floor")
    problem = case.problem
    backend = native.NativeBackend(problem, grid=case.grid, modes=case.modes)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    rows = case.n_rows
    assert ledger["raman_undamped"] == rows * 2 * problem.grids.n_to * 8
    assert ledger["stage_raman_rates"] == problem.stepper.tableau.n_stages * rows * 8
