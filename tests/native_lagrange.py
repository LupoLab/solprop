# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's Lagrange basis and Adams trial tables, reached for tests.

The compiled Adams march forms its basis and a trial's coefficient tables in C++
(``lagrange_basis.hpp``) every trial, and the native helper record names neither (SEP
52). These wrappers reach the same C++ through its private test bindings
(``solprop_native._core._testing``), so that tests can hold it to solprop's plain forms
and to the exact tables. Call them only where solprop-native is installed.
"""

from typing import Any

import numpy as np

from solprop._helpers import Helpers


def _testing() -> Any:
    """Return the native extension's private test module."""
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _core,  # pyright: ignore[reportMissingModuleSource]
    )

    return _core._testing


def native_basis(points: np.ndarray, upper: float, integrate: bool):
    """Return native's Lagrange basis on `points`: its coefficients, or its integrals.

    The C++ basis the compiled Adams march forms every trial, through its private test
    binding (``_core._testing.basis_rows``); the native record names no basis.
    """
    testing = _testing()
    count = points.shape[0]
    rows, weights = np.empty((count, count)), np.empty(count)
    testing.basis_rows(points, upper, integrate, rows, weights)
    return rows, weights


def native_tables(nodes: np.ndarray, order: int, orders: list[int], compensated: bool):
    """Return native's exponential Adams trial tables, formed in one C++ call.

    The arrays of :func:`solprop.backends.numpy_ref.adams_tables.trial_tables`, in its
    order, through the private test binding (``_core._testing.trial_tables``).
    """
    testing = _testing()
    width, count = 1 + max(orders), len(orders)
    predictor, corrector = np.empty((order, order)), np.empty((width, width))
    factored = np.empty((count, 2 * width))
    if compensated:
        spans, carried = np.empty(width - 1), np.empty(order)
        endpoint, defects = np.empty(order + 1), np.empty((count, width))
    else:
        spans, carried, endpoint = np.empty(0), np.empty(0), np.empty(0)
        defects = np.empty((0, 0))
    testing.trial_tables(
        nodes,
        order,
        orders,
        compensated,
        predictor,
        corrector,
        factored,
        spans,
        carried,
        endpoint,
        defects,
    )
    return predictor, corrector, factored, spans, carried, endpoint, defects


def native_basis_record() -> Helpers:
    """Return a test record whose basis is native's C++ one, its rows the plain ones."""
    from solprop._helpers import PLAIN  # noqa: PLC0415 - beside the record it copies

    return Helpers("native", lambda: PLAIN.points, lambda: native_basis)
