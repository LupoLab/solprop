# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every row of the case catalogue on native against ``numpy_ref``, rung by rung.

The catalogue, its rungs and their bounds are ``tests/capability_routes.py`` and
``tests/parity_ladder.py``. A row is fast when it evaluates the source or takes one
step, medium when it marches its window, and slow when it says it is slow by cost.
"""

import pytest

import parity_ladder
from backends.adams_source_cases import one_evaluation
from backends.planar_rows import without_responses
from capability_routes import catalogue

solprop_native = pytest.importorskip("solprop_native")


def _param(row: parity_ladder.Row):
    marks = [pytest.mark.filterwarnings(spec) for spec in row.quiet]
    if row.slow:
        marks.append(pytest.mark.slow)
    elif row.medium:
        marks.append(pytest.mark.medium)
    return pytest.param(row, id=row.id, marks=marks)


@pytest.mark.parametrize("row", [_param(row) for row in catalogue()])
def test_every_rung_agrees_with_the_reference(row, request):
    parity_ladder.check(solprop_native.NativeBackend, row, request)


# Medium: the argon capillary's translation takes about a second.
@pytest.mark.medium
def test_a_guard_that_switches_off_every_response_of_several_is_refused():
    """Kerr beside the plasma: removing both would let Kerr clear a plasma guard."""
    built = one_evaluation("envelope plasma")
    assert parity_ladder.responses_held(built.problem) == ("kerr", "envelope plasma")
    row = parity_ladder.Row(
        id="every response of two",
        why="The envelope plasma's source is the reference's.",
        axes={"response": "envelope plasma"},
        build=lambda: built,
        rungs={"source": lambda _: 1.0},
        switch_off=without_responses,
    )
    with pytest.raises(AssertionError, match=r"holds \('kerr', 'envelope plasma'\)"):
        parity_ladder.check(solprop_native.NativeBackend, row)
