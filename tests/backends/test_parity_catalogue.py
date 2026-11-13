# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every row of the case catalogue on native against ``numpy_ref``, rung by rung.

The catalogue, its rungs and their bounds are ``tests/capability_routes.py`` and
``tests/parity_ladder.py``. A row is fast when it evaluates the source or takes one
step, medium when it marches its window.
"""

import pytest

import parity_ladder
from capability_routes import catalogue

solprop_native = pytest.importorskip("solprop_native")


def _param(row: parity_ladder.Row):
    marks = [pytest.mark.filterwarnings(spec) for spec in row.quiet]
    if row.medium:
        marks.append(pytest.mark.medium)
    return pytest.param(row, id=row.id, marks=marks)


@pytest.mark.parametrize("row", [_param(row) for row in catalogue()])
def test_every_rung_agrees_with_the_reference(row, request):
    parity_ladder.check(solprop_native.NativeBackend, row, request)
