# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""optmat's ionisation rates on the compiled backends against numpy_ref (SEP 3).

Argon and nitrogen under ``MaterialSelection()``, field and envelope, and argon
launched above its rate model's maximum field so that the table's hold is inside the
run (the handoff of Phase 1d). Every rung of ``tests/optmat_parity.py`` -- one linear
application, one nonlinear evaluation, one step, eight steps, each statistic and the
driver's complete ten-step run -- is compared with its bound, which that module derives
from the operation count before any measurement.

Medium tier: each case is a real run of ten steps, a few seconds with its translation;
the reference's envelope plasma (cycle quadrature on the host) is the largest share.

Measured on the development machine against bounds of order 1e-10 to 1e-7, the worst
deviation of any field rung was 2.1e-15 (the first stage's derivative, ``n2_field`` on
Kokkos) and of any statistic 6.1e-14 (``quadrature_error``, ``n2_field`` on ``native``).
"""

import numpy as np
import pytest

from backends.test_batch_responses import compare
from optmat_parity import (
    check_ladder,
    derivative,
    deviation,
    reference,
    stage_bound,
)
from optmat_parity_cases import build_case, variant
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse
from solprop.translate.ionisation import DEFAULT_RTOL

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

CASES = ("ar_field", "ar_envelope", "ar_hold", "n2_field", "n2_envelope")

DETECTION = 1e3
"""How far above a rung's bound the term it qualifies must reach.

A backend that dropped the term would then miss the reference by a thousand bounds;
measured shares are 1e-4 to 1e-1 against stage bounds near 1e-10.
"""


@pytest.mark.medium
@pytest.mark.parametrize("name", CASES)
def test_every_rung_agrees_with_the_reference(name, equivalence_backend_class):
    check_ladder(name, equivalence_backend_class)


@pytest.mark.medium
@pytest.mark.parametrize("name", CASES)
def test_the_first_stage_sees_the_plasma(name):
    # The plasma's share of N(a_0), against the same run without it on the
    # reference: far above the nonlinear rung's bound, so the rung can see it.
    case = build_case(name)
    share = deviation(
        derivative(NumpyRefBackend, variant(name, plasma=False)),
        derivative(NumpyRefBackend, case),
    )
    assert share > DETECTION * stage_bound(case)


@pytest.mark.medium
def test_the_hold_above_the_maximum_field_is_inside_the_run():
    # Every plane's peak rate is the held rate W(E_m): the field exceeds the
    # model's maximum field at each of them, and every backend ran the hold.
    case = build_case("ar_hold")
    assert case.simulation is not None
    (rate,) = [
        response.rate
        for response in case.simulation.responses
        if isinstance(response, PlasmaResponse)
    ]
    assert isinstance(rate, OptmatRate)
    held = float(rate(rate.maximum_field))
    assert float(rate(2.0 * rate.maximum_field)) == held
    peaks = reference("ar_hold").run_statistics["peak_ionisation_rate"]
    # The table is within its build tolerance of direct evaluation.
    np.testing.assert_allclose(peaks, held, rtol=DEFAULT_RTOL)


@pytest.mark.medium
# A native lockstep batch replays its Adams members on solprop's march, with the
# warning native/tests/test_batch.py asserts (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("name", ["n2_field", "ar_envelope"])
def test_a_batch_equals_its_members(name, equivalence_backend_name):
    # Within the rounding budget tests/backends/test_batch_responses.py holds its own
    # cases to: a lockstep batch's members run the same arithmetic as independent
    # runs. The compiled backends only: the reference runs a batch member by member.
    case = build_case(name)
    compare(case.problem, case.grid, equivalence_backend_name)
