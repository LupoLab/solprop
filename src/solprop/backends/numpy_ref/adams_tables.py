# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coefficient tables of an Adams trial (SEP 13, SEP 16, ADR-042).

Every weight an Adams trial applies depends only on the history's normalised nodes and
the orders it scores, never on the state. The interaction formulation integrates its
interpolants: :func:`defect_rows` (each scored order's corrector less predictor, and the
step's own corrector) and :func:`fused_tables`. The exponential formulation works in the
monomial basis of its moments: :func:`exponential_corrector` and
:func:`exponential_rows`, and for native's factored defects
:func:`factored_defect_rows`; :func:`compensation_tables` are a varying operator's
transported compensation, and :func:`exponential_tables` and :func:`trial_tables` gather
them as native's march forms them in C++. The reference's engines
(:mod:`~solprop.backends.numpy_ref.multistep`) read these with the plain basis; native's
parity tests read them with its own (``helpers``).
"""

import numpy as np

from solprop._helpers import PLAIN, Helpers
from solprop.stepper.multistep import integrated_weights, lagrange_coefficients

__all__ = [
    "compensation_tables",
    "defect_rows",
    "exponential_corrector",
    "exponential_rows",
    "exponential_tables",
    "factored_defect_rows",
    "fused_tables",
    "trial_tables",
]


def defect_rows(
    nodes: list[float], orders: list[int], *, helpers: Helpers = PLAIN
) -> tuple[np.ndarray, np.ndarray]:
    """Return each order's defect weights and the step's own corrector weights.

    Over ``(source, newest, ...)``, order ``k``'s defect is its ``k + 1``-node
    corrector through the new source less its ``k``-node predictor, as integrated
    weights of the interaction formulation.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    orders : list of int
        The orders scored, the step's own first.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the Adams basis; the plain forms by default.

    Returns
    -------
    rows : numpy.ndarray
        The defect rows, shape ``(len(orders), 1 + max(orders))``.
    corrector : numpy.ndarray
        The step's own corrector weights, zero past its nodes.
    """
    width = 1 + max(orders)
    rows = np.zeros((len(orders), width))
    corrector = np.zeros(width)
    for index, order in enumerate(orders):
        stencil = nodes[:order]
        through = integrated_weights([1.0, *stencil], check=False, helpers=helpers)
        rows[index, : order + 1] = through
        rows[index, 1 : order + 1] -= integrated_weights(
            stencil, check=False, helpers=helpers
        )
        if index == 0:
            corrector[: order + 1] = through
    return rows, corrector


def fused_tables(
    nodes: list[float], order: int, orders: list[int], *, helpers: Helpers = PLAIN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an interaction-formulation trial's coefficient tables, as one record.

    What :class:`~solprop.backends.numpy_ref.multistep.InteractionHistory`'s trial
    forms; native's march forms the same tables in C++, and its parity tests read
    them here.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    order : int
        The step's order.
    orders : list of int
        The orders scored, the step's own first.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the Adams basis; the plain forms by default.

    Returns
    -------
    tuple
        The predictor's integrated weights over the ``order`` newest rows; the
        corrector's weights over ``(source, newest, ...)``, zero past its nodes; and
        one defect row per scored order over the same samples.
    """
    rows, corrector = defect_rows(nodes, orders, helpers=helpers)
    predictor = integrated_weights(nodes[:order], check=False, helpers=helpers)
    return predictor, corrector, rows


def exponential_corrector(
    nodes: list[float], order: int, width: int, *, helpers: Helpers = PLAIN
) -> np.ndarray:
    """Return the step's own corrector's Lagrange coefficients, padded to `width`.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    order : int
        The step's order.
    width : int
        ``1 + max(orders)``.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the Adams basis; the plain forms by default.

    Returns
    -------
    numpy.ndarray
        The coefficients over ``X = (source, newest, ...)``, shape
        ``(width, width)``, zero past its nodes.
    """
    corrector = np.zeros((width, width))
    corrector[: order + 1, : order + 1] = lagrange_coefficients(
        [1.0, *nodes[:order]], check=False, helpers=helpers
    )
    return corrector


def exponential_rows(
    nodes: list[float], orders: list[int], width: int, *, helpers: Helpers = PLAIN
) -> np.ndarray:
    """Return each order's defect coefficients in the monomial basis.

    Over ``X = (source, newest, ...)`` in the monomial basis of the moments, order
    ``k``'s defect is its ``k + 1``-node corrector through the new source less its
    ``k``-node predictor, as for :func:`defect_rows`; the weights are the Lagrange
    coefficients, which the trial's moments turn into exponential weights.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    orders : list of int
        The orders scored, the step's own first.
    width : int
        ``1 + max(orders)``.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the Adams basis; the plain forms by default.

    Returns
    -------
    numpy.ndarray
        The defect rows, shape ``(len(orders), width, width)``.
    """
    rows = np.zeros((len(orders), width, width))
    for index, order in enumerate(orders):
        stencil = nodes[:order]
        rows[index, : order + 1, : order + 1] = lagrange_coefficients(
            [1.0, *stencil], check=False, helpers=helpers
        )
        rows[index, 1 : order + 1, :order] -= lagrange_coefficients(
            stencil, check=False, helpers=helpers
        )
    return rows


def factored_defect_rows(
    nodes: list[float], orders: list[int], width: int
) -> np.ndarray:
    """Return each exponential defect factored: a divided difference and a polynomial.

    Order ``k``'s defect, its ``k + 1``-node corrector through the new sample less its
    ``k``-node predictor, vanishes at the predictor's nodes, so it is
    ``f[1, theta_0, ..., theta_{k-1}] prod_{j<k} (theta - theta_j)``: one divided
    difference of the samples times one polynomial, whose integral against the flow is
    one combination of the moments. That is ``2 (k + 1)`` products an element instead
    of ``(k + 1) n_k``, the monomial rows of :func:`exponential_rows`.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    orders : list of int
        The orders scored, the step's own first.
    width : int
        ``1 + max(orders)``: the samples and the moments of the trial.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(orders), 2 width)``: the divided-difference coefficients over
        ``(source, newest, ...)``, then the node polynomial's ascending monomial
        coefficients.
    """
    rows = np.zeros((len(orders), 2 * width))
    for index, order in enumerate(orders):
        points = [1.0, *nodes[:order]]
        row = rows[index]
        for t, node in enumerate(points):
            denominator = 1.0
            for s, other in enumerate(points):
                if s != t:
                    denominator *= node - other
            row[t] = 1.0 / denominator
        # The node polynomial one linear factor at a time, newest node first, as the
        # Adams basis builds its numerators (`_basis_rows`); in plain floats, since
        # `numpy.polynomial.polyfromroots` costs four fifths of this function. Every
        # node is at or below zero, so every coefficient is positive: no cancellation.
        coefficients = [1.0]
        for root in nodes[:order]:
            product = [0.0, *coefficients]
            for power, value in enumerate(coefficients):
                product[power] -= root * value
            coefficients = product
        row[width : width + order + 1] = coefficients
    return rows


def exponential_tables(
    nodes: list[float], order: int, orders: list[int], *, helpers: Helpers = PLAIN
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an exponential trial's coefficient tables.

    What :class:`~solprop.backends.numpy_ref.multistep.ExponentialHistory`'s trial
    forms, but for the defects, which the reference applies in the monomial basis
    (:func:`exponential_rows`) and native in the factored form.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    order : int
        The step's order.
    orders : list of int
        The orders scored, the step's own first.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the Adams basis; the plain forms by default.

    Returns
    -------
    tuple
        The predictor's Lagrange coefficients over the ``order`` newest rows, shape
        ``(order, order)``; the corrector's over ``(source, newest, ...)``, shape
        ``(width, width)`` with ``width = 1 + max(orders)``; and the factored defect
        rows of :func:`factored_defect_rows`.
    """
    width = 1 + max(orders)
    corrector = exponential_corrector(nodes, order, width, helpers=helpers)
    predictor = lagrange_coefficients(nodes[:order], check=False, helpers=helpers)
    return predictor, corrector, factored_defect_rows(nodes, orders, width)


def compensation_tables(
    nodes: list[float], order: int, orders: list[int], *, helpers: Helpers = PLAIN
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a trial's transported-compensation tables.

    The weights the reference's compensation applies (SEP 16 section 4.2, ADR-042),
    over the
    carried rows ``exp((1 - theta_j) x) (D(z_j) - x / h) u_j`` of the history
    samples and the endpoint term ``(D(z + h) - x / h) u_pred``, each sum still to be
    multiplied by the step.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    order : int
        The step's order.
    orders : list of int
        The orders scored, the step's own first.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the Adams basis; the plain forms by default.

    Returns
    -------
    tuple
        The spans ``1 - theta_j`` of the ``width - 1`` carried samples; the
        predictor's weights over the ``order`` newest of them; the corrector's over
        ``(endpoint, newest, ...)``; and each scored order's defect weights over the
        same, shape ``(len(orders), width)``, with ``width = 1 + max(orders)``.
    """
    width = 1 + max(orders)
    spans = np.array([1.0 - node for node in nodes[: width - 1]])
    # Each count's predictor and corrector once: the step's own pair is also its
    # defect's, so a trial forms two weights per scored order, not two more.
    weights = {
        count: (
            integrated_weights(nodes[:count], check=False, helpers=helpers),
            integrated_weights([1.0, *nodes[:count]], check=False, helpers=helpers),
        )
        for count in dict.fromkeys([order, *orders])
    }
    rows = np.zeros((len(orders), width))
    for index, scored in enumerate(orders):
        predicted, through = weights[scored]
        rows[index, : scored + 1] = through
        rows[index, 1 : scored + 1] -= predicted
    predictor, corrector = weights[order]
    return spans, predictor, corrector, rows


def trial_tables(
    nodes: list[float],
    order: int,
    orders: list[int],
    *,
    compensated: bool,
    helpers: Helpers = PLAIN,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Return all of an exponential trial's coefficient tables, as native reads them.

    :func:`exponential_tables`' three and, compensated, :func:`compensation_tables`'
    four, composed over the record's basis, read once a trial instead of from a dozen
    small calls. The native march forms the same tables in C++ with the same
    operations in the same order.

    Parameters
    ----------
    nodes : list of float
        The history's normalised positions, newest first.
    order : int
        The step's order.
    orders : list of int
        The orders scored, the step's own first.
    compensated : bool
        Whether the trial carries a varying operator's transported compensation.
    helpers : solprop._helpers.Helpers, optional
        The implementation of the Adams basis; the plain forms by default.

    Returns
    -------
    tuple
        The predictor, the corrector and the factored defect rows; then the
        compensation's spans, predictor, corrector and defect rows, shapes ``(0,)``
        and ``(0, 0)`` when not compensated.
    """
    predictor, corrector, factored = exponential_tables(
        nodes, order, orders, helpers=helpers
    )
    if not compensated:
        return predictor, corrector, factored, *_NO_COMPENSATION
    spans, carried, endpoint, defects = compensation_tables(
        nodes, order, orders, helpers=helpers
    )
    return predictor, corrector, factored, spans, carried, endpoint, defects


_NO_COMPENSATION = (np.empty(0), np.empty(0), np.empty(0), np.empty((0, 0)))
"""An uncompensated trial's compensation tables: spans, weights and defect rows."""
