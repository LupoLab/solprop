# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The NumPy z-table evaluator and the persistent host adapter agree."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from solprop.backends.numba_cpu.tables import HostTable
from solprop.translate.chebyshev import PiecewiseChebyshev


@pytest.mark.parametrize("degree", [0, 1, 5, 32])
def test_host_clenshaw_preserves_the_reference_polynomial(degree):
    rng = np.random.default_rng(34)
    coefficients = rng.normal(size=(2, degree + 1, 1, 1, 1, 7)) + 1j * rng.normal(
        size=(2, degree + 1, 1, 1, 1, 7)
    )
    table = PiecewiseChebyshev(
        coefficients,
        np.array([0.0, 0.4, 1.0]),
        np.full(2, degree),
        np.array([0.4]),
        False,
    )
    adapter = HostTable(table)
    # Clenshaw is a sequence of at most 8n real operations per complex coefficient;
    # use its absolute coefficient envelope to remain valid near cancellation.
    bound = (
        8
        * (degree + 1) ** 2
        * np.finfo(float).eps
        * float(np.sum(np.abs(coefficients)))
    )
    assert bound > 0
    owned = (id(adapter.work), id(adapter.value), id(adapter.left))
    for z in [0.0, 0.2, 0.4, 0.7, 1.0]:
        assert_allclose(adapter.at(z), table(z), rtol=0, atol=bound)
        assert_allclose(
            adapter.delta(0.0, z), table(z) - table(0.0), rtol=0, atol=2 * bound
        )
        assert owned == (id(adapter.work), id(adapter.value), id(adapter.left))
    assert adapter.nbytes == 5 * np.prod(table.value_shape) * 16
    assert_array_equal(table.coefficients, coefficients)
