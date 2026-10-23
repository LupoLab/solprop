# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Explicit Runge-Kutta tableaux with embedded error estimates.

Two order-5(4) pairs are provided: ``DOPRI5`` [1]_ (the default scheme)
and ``TSIT5`` [2]_. Both are first-same-as-last (FSAL), so a step costs six
evaluations of the nonlinear term and the embedded companion supplies the local error
estimate for free (``docs/dev/stepper.md`` sections 2 and 3).

A tableau is data, not behaviour: it carries the nodes, the Butcher matrix, the
propagated weights and the error weights, and it validates its own structure.
:func:`check_order_conditions` measures the order the coefficients actually attain,
from the rooted-tree conditions [3]_. Applying a tableau to a state is a backend kernel
and does not live here.

Every rational coefficient is written as the exact rational of its reference and
evaluated once, at import, so each constant is a single correctly-rounded division and
no division happens per step.

References
----------
.. [1] J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae",
       J. Comput. Appl. Math. 6, 19 (1980). doi:10.1016/0771-050X(80)90013-3
.. [2] Ch. Tsitouras, "Runge-Kutta pairs of order 5(4) satisfying only the first column
       simplifying assumption", Comput. Math. Appl. 62, 770 (2011).
       doi:10.1016/j.camwa.2011.06.002
.. [3] J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*, 3rd ed.,
       Wiley (2016), chapter 3.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from jaxtyping import Float

from solprop.exceptions import StepperError

__all__ = [
    "DOPRI5",
    "DOPRI5_DENSE",
    "MAX_CHECKED_ORDER",
    "TSIT5",
    "OrderConditionReport",
    "Tableau",
    "check_order_conditions",
    "continuous_extension",
    "extension_coefficients",
    "extension_value",
]

MAX_CHECKED_ORDER = 5
"""Highest order for which :func:`check_order_conditions` knows the rooted trees."""

# Structural checks compare sums of up to seven coefficients whose magnitudes reach ~13
# for Tsit5, whose published values themselves carry ~1e-16 of relative representation
# error; the bound 7 * 13 * eps is ~1e-14. The worst residual measured over both
# tableaux is 7.8e-16, so this leaves an order of magnitude of margin while still
# catching a mistyped digit.
_STRUCTURE_TOLERANCE = 1e-14

# The rooted-tree order conditions through order 5, as (order, label, 1/gamma(t)). The
# right-hand sides are the reciprocals of the tree densities, evaluated once here rather
# than divided out at every call. Labels read as elementary weights: "c" is the node
# vector and "A" the Butcher matrix, so "b.AAc" means
# sum_i b_i sum_j a_ij sum_k a_jk c_k.
_ORDER_CONDITIONS: tuple[tuple[int, str, float], ...] = (
    (1, "b.1", 1.0),
    (2, "b.c", 1 / 2),
    (3, "b.c2", 1 / 3),
    (3, "b.Ac", 1 / 6),
    (4, "b.c3", 1 / 4),
    (4, "b.c.Ac", 1 / 8),
    (4, "b.Ac2", 1 / 12),
    (4, "b.AAc", 1 / 24),
    (5, "b.c4", 1 / 5),
    (5, "b.c2.Ac", 1 / 10),
    (5, "b.(Ac)2", 1 / 20),
    (5, "b.c.Ac2", 1 / 15),
    (5, "b.Ac3", 1 / 20),
    (5, "b.c.AAc", 1 / 30),
    (5, "b.A(c.Ac)", 1 / 40),
    (5, "b.AAc2", 1 / 60),
    (5, "b.AAAc", 1 / 120),
)


def _readonly(values: Float[np.ndarray, "*shape"]) -> Float[np.ndarray, "*shape"]:
    """Return a private float64 copy that cannot be written through.

    Parameters
    ----------
    values : numpy.ndarray
        Array of any shape, dimensionless.

    Returns
    -------
    numpy.ndarray
        A float64 copy with the write flag cleared.
    """
    copy = np.array(values, dtype=np.float64)
    copy.setflags(write=False)
    return copy


# PR02 is suppressed because numpydoc's static hook cannot see the __init__ that
# @dataclass generates, so it reads every documented field as an unknown parameter.
@dataclass(frozen=True, eq=False)
class Tableau:  # numpydoc ignore=PR02
    """An explicit Runge-Kutta tableau with an embedded companion.

    The stage derivatives are ``k_i = f(z + c_i h, y + h sum_j a_ij k_j)``; the step is
    ``y_new = y + h sum_i b_i k_i`` and the local error estimate is
    ``e = h sum_i (b_i - b_hat_i) k_i``.

    Equality is by identity (``eq=False``): the fields are arrays, for which ``==`` is
    elementwise and would make a generated ``__eq__`` raise.

    Parameters
    ----------
    name : str
        Name of the scheme, used in error messages.
    nodes : numpy.ndarray
        Stage abscissae ``c`` of shape ``(n_stages,)``, as fractions of the step;
        ``c[0]`` is 0 for an explicit method.
    matrix : numpy.ndarray
        Butcher matrix ``A`` of shape ``(n_stages, n_stages)``, strictly lower
        triangular and dimensionless.
    solution_weights : numpy.ndarray
        Weights ``b`` of shape ``(n_stages,)`` of the propagated solution, which is the
        one of order ``order``. Dimensionless, summing to 1.
    error_weights : numpy.ndarray
        Weights ``b - b_hat`` of shape ``(n_stages,)`` of the error estimate,
        dimensionless, summing to 0. The *difference* is stored rather than ``b_hat``
        because it is the only combination ever used, and forming it once avoids a
        cancellation between two nearly equal vectors at every step.
    order : int
        Order of the propagated solution.
    embedded_order : int
        Order of the embedded companion ``b_hat``, which sets the controller exponent.
    first_same_as_last : bool
        Whether the last stage derivative equals the first derivative of the next step
        (FSAL), so it can be reused instead of recomputed.

    Raises
    ------
    StepperError
        If the shapes disagree, an entry is not finite, ``A`` is not strictly lower
        triangular, a row of ``A`` does not sum to its node, ``b`` does not sum to 1,
        ``b - b_hat`` does not sum to 0 or is zero throughout, the orders are not
        ``0 < embedded < order``, or the FSAL flag is set but the last row of ``A`` is
        not ``b``.

    Notes
    -----
    Two transcription errors are easy to make: swapping ``b`` and ``b_hat``, which
    propagates the lower-order solution, and storing an error vector in the slot
    labelled as the higher-order weights. The two sum rules checked here catch both: an
    error vector does not sum to 1 and a weight vector does not sum to 0. See
    ``docs/dev/stepper.md`` section 10.

    Examples
    --------
    >>> from solprop.stepper import DOPRI5
    >>> DOPRI5.n_stages, DOPRI5.order, DOPRI5.embedded_order
    (7, 5, 4)
    >>> DOPRI5.first_same_as_last
    True
    >>> bool(abs(DOPRI5.embedded_weights.sum() - 1.0) < 1e-15)
    True
    """

    # The four arrays carry independent axis names on purpose: their agreement is
    # checked by _check_shapes with a StepperError naming the culprit, which must also
    # hold in production, where the test session's runtime type checker is absent.
    # Tying them to one axis name would let the checker forestall that validation.
    name: str
    nodes: Float[np.ndarray, " n_stages"]
    matrix: Float[np.ndarray, "n_rows n_cols"]
    solution_weights: Float[np.ndarray, " n_solution"]
    error_weights: Float[np.ndarray, " n_error"]
    order: int
    embedded_order: int
    first_same_as_last: bool

    def __post_init__(self) -> None:
        """Validate the structure, then adopt read-only copies of every array."""
        for attribute in ("nodes", "matrix", "solution_weights", "error_weights"):
            object.__setattr__(self, attribute, _readonly(getattr(self, attribute)))
        self._check_shapes()
        self._check_structure()
        self._check_orders()

    def _check_shapes(self) -> None:
        """Check that the arrays describe one stage count and are finite."""
        n_stages = self.nodes.size
        wrong = [
            name
            for name, bad in (
                ("matrix", self.matrix.shape != (n_stages, n_stages)),
                ("solution_weights", self.solution_weights.shape != (n_stages,)),
                ("error_weights", self.error_weights.shape != (n_stages,)),
            )
            if bad
        ]
        if wrong:
            raise StepperError(
                f"{self.name}: {', '.join(wrong)} inconsistent with {n_stages} nodes"
            )
        for name in ("nodes", "matrix", "solution_weights", "error_weights"):
            if not np.all(np.isfinite(getattr(self, name))):
                raise StepperError(f"{self.name}: {name} has a non-finite entry")

    def _check_structure(self) -> None:
        """Check explicitness, the row sums and the two weight sum rules."""
        if self.nodes[0] != 0.0:
            raise StepperError(f"{self.name}: an explicit method needs nodes[0] == 0")
        if np.any(np.triu(self.matrix) != 0.0):
            raise StepperError(f"{self.name}: matrix is not strictly lower triangular")
        row_defect = float(np.max(np.abs(self.matrix.sum(axis=1) - self.nodes)))
        if row_defect > _STRUCTURE_TOLERANCE:
            raise StepperError(
                f"{self.name}: matrix rows do not sum to the nodes"
                f" (worst {row_defect:.2e})"
            )
        weight_defect = abs(float(self.solution_weights.sum()) - 1.0)
        if weight_defect > _STRUCTURE_TOLERANCE:
            raise StepperError(
                f"{self.name}: solution weights sum to 1 + {weight_defect:.2e}, not 1"
            )
        error_defect = abs(float(self.error_weights.sum()))
        if error_defect > _STRUCTURE_TOLERANCE:
            raise StepperError(
                f"{self.name}: error weights sum to {error_defect:.2e}, not 0; a vector"
                " of weights has been stored where the difference b - b_hat belongs"
            )
        if not np.any(self.error_weights):
            # An all-zero difference passes the sum rule and scores every step zero,
            # so the adaptive march grows the step at its cap without end
            # (docs/dev/stepper.md section 10, trap 11). A scheme without an estimate
            # is stepped by LowStorageRungeKutta, which takes no tableau (SEP 24).
            raise StepperError(
                f"{self.name}: every error weight is zero, so the embedded estimate "
                "is identically zero and the controller would grow the step without "
                "limit; a scheme without an estimate is replayed by "
                "LowStorageRungeKutta() instead"
            )

    def _check_orders(self) -> None:
        """Check the declared orders and the FSAL claim."""
        if not 0 < self.embedded_order < self.order:
            raise StepperError(
                f"{self.name}: need 0 < embedded_order < order, got"
                f" {self.embedded_order} and {self.order}"
            )
        if not self.first_same_as_last:
            return
        last_row_is_b = np.array_equal(self.matrix[-1, :-1], self.solution_weights[:-1])
        if not last_row_is_b or self.solution_weights[-1] != 0.0:
            raise StepperError(
                f"{self.name}: first_same_as_last requires the last matrix row to equal"
                " the solution weights and the last weight to be zero"
            )

    @property
    def n_stages(self) -> int:
        """Return the number of stages, counting the reused FSAL stage.

        Returns
        -------
        int
            Number of stages.
        """
        return int(self.nodes.size)

    @property
    def embedded_weights(self) -> Float[np.ndarray, " n_stages"]:
        """Return the weights of the embedded companion.

        Returns
        -------
        numpy.ndarray
            Weights ``b_hat = b - (b - b_hat)`` of shape ``(n_stages,)``,
            dimensionless, summing to 1.
        """
        return self.solution_weights - self.error_weights


def _dot(left: Float[np.ndarray, " n"], right: Float[np.ndarray, " n"]) -> float:
    """Return an exactly rounded inner product.

    The order conditions are a statement about the coefficients, so the check must not
    add error of its own: ``math.fsum`` is correctly rounded, whereas a BLAS dot product
    reassociates and costs ~1e-15 on the Tsit5 order-5 trees, enough to read a genuine
    order 5 as order 4.

    Parameters
    ----------
    left, right : numpy.ndarray
        Vectors of the same length, dimensionless.

    Returns
    -------
    float
        The inner product, rounded once.
    """
    return math.fsum(left * right)


def _matvec(
    matrix: Float[np.ndarray, "n n"], vector: Float[np.ndarray, " n"]
) -> Float[np.ndarray, " n"]:
    """Return ``matrix @ vector``, each row reduced with :func:`math.fsum`.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square matrix, dimensionless.
    vector : numpy.ndarray
        Vector of matching length, dimensionless.

    Returns
    -------
    numpy.ndarray
        The product, of the same length as ``vector``.
    """
    return np.array([_dot(row, vector) for row in matrix])


def _elementary_weights(
    nodes: Float[np.ndarray, " n_stages"],
    matrix: Float[np.ndarray, "n_stages n_stages"],
    weights: Float[np.ndarray, " n_stages"],
) -> dict[str, float]:
    """Evaluate the 17 elementary weights of the trees through order 5.

    Parameters
    ----------
    nodes : numpy.ndarray
        Stage abscissae ``c`` of shape ``(n_stages,)``, dimensionless.
    matrix : numpy.ndarray
        Butcher matrix ``A`` of shape ``(n_stages, n_stages)``, dimensionless.
    weights : numpy.ndarray
        Weight vector of shape ``(n_stages,)``, dimensionless.

    Returns
    -------
    dict
        Elementary weight of every tree, keyed by tree label.
    """
    node = nodes
    node2 = node * node
    node3 = node2 * node
    a_node = _matvec(matrix, node)
    a_node2 = _matvec(matrix, node2)
    aa_node = _matvec(matrix, a_node)
    return {
        "b.1": math.fsum(weights),
        "b.c": _dot(weights, node),
        "b.c2": _dot(weights, node2),
        "b.Ac": _dot(weights, a_node),
        "b.c3": _dot(weights, node3),
        "b.c.Ac": _dot(weights, node * a_node),
        "b.Ac2": _dot(weights, a_node2),
        "b.AAc": _dot(weights, aa_node),
        "b.c4": _dot(weights, node2 * node2),
        "b.c2.Ac": _dot(weights, node2 * a_node),
        "b.(Ac)2": _dot(weights, a_node * a_node),
        "b.c.Ac2": _dot(weights, node * a_node2),
        "b.Ac3": _dot(weights, _matvec(matrix, node3)),
        "b.c.AAc": _dot(weights, node * aa_node),
        "b.A(c.Ac)": _dot(weights, _matvec(matrix, node * a_node)),
        "b.AAc2": _dot(weights, _matvec(matrix, a_node2)),
        "b.AAAc": _dot(weights, _matvec(matrix, aa_node)),
    }


def _residuals(
    nodes: Float[np.ndarray, " n_stages"],
    matrix: Float[np.ndarray, "n_stages n_stages"],
    weights: Float[np.ndarray, " n_stages"],
) -> Mapping[str, float]:
    """Return the absolute order-condition residuals, keyed by tree label.

    Parameters
    ----------
    nodes : numpy.ndarray
        Stage abscissae ``c`` of shape ``(n_stages,)``, dimensionless.
    matrix : numpy.ndarray
        Butcher matrix ``A`` of shape ``(n_stages, n_stages)``, dimensionless.
    weights : numpy.ndarray
        Weight vector of shape ``(n_stages,)``, dimensionless.

    Returns
    -------
    collections.abc.Mapping
        Absolute residual of every tree, keyed by tree label.
    """
    attained = _elementary_weights(nodes, matrix, weights)
    return MappingProxyType(
        {label: abs(attained[label] - target) for _, label, target in _ORDER_CONDITIONS}
    )


def _attained_order(residuals: Mapping[str, float], tolerance: float) -> int:
    """Return the highest order whose conditions all sit within ``tolerance``.

    Parameters
    ----------
    residuals : collections.abc.Mapping
        Absolute residuals keyed by tree label, dimensionless.
    tolerance : float
        Threshold below which a condition counts as satisfied, dimensionless.

    Returns
    -------
    int
        The attained order, 0 if even the first condition fails.
    """
    attained = 0
    for order in range(1, MAX_CHECKED_ORDER + 1):
        satisfied = all(
            residuals[label] <= tolerance
            for condition_order, label, _ in _ORDER_CONDITIONS
            if condition_order == order
        )
        if not satisfied:
            break
        attained = order
    return attained


def _worst_residual(residuals: Mapping[str, float], up_to: int) -> float:
    """Return the largest residual among the conditions of order ``up_to`` and below.

    Parameters
    ----------
    residuals : collections.abc.Mapping
        Absolute residuals keyed by tree label, dimensionless.
    up_to : int
        Highest condition order to include.

    Returns
    -------
    float
        The largest residual in that set.
    """
    return max(
        residuals[label] for order, label, _ in _ORDER_CONDITIONS if order <= up_to
    )


@dataclass(frozen=True)
class OrderConditionReport:
    """The order a tableau's coefficients attain, measured tree by tree.

    Attributes
    ----------
    tolerance : float
        Residual below which a condition counts as satisfied, dimensionless.
    solution_residuals : collections.abc.Mapping
        Absolute residual ``|sum_i b_i Phi_i(t) - 1/gamma(t)|`` of every rooted tree
        through order ``MAX_CHECKED_ORDER``, keyed by tree label, for the propagated
        weights. Dimensionless.
    embedded_residuals : collections.abc.Mapping
        The same for the embedded companion ``b_hat``.
    solution_order : int
        Highest order, at most ``MAX_CHECKED_ORDER``, whose conditions the
        propagated weights all satisfy within ``tolerance``.
    embedded_order : int
        The same for the embedded companion.
    max_residual : float
        Largest residual among the conditions the tableau *declares* it satisfies (up to
        ``Tableau.order`` for the propagated weights and ``Tableau.embedded_order`` for
        the companion). Independent of ``tolerance``, so it is a measurement rather than
        a restatement of the threshold.
    matches_declared_orders : bool
        Whether the attained orders equal the declared ones.
    """

    tolerance: float
    solution_residuals: Mapping[str, float]
    embedded_residuals: Mapping[str, float]
    solution_order: int
    embedded_order: int
    max_residual: float
    matches_declared_orders: bool


def check_order_conditions(
    tableau: Tableau, tolerance: float = 1e-15
) -> OrderConditionReport:
    """Measure the order the tableau's coefficients attain.

    A tableau of order ``p`` satisfies ``sum_i b_i Phi_i(t) = 1/gamma(t)`` for every
    rooted tree ``t`` of order up to ``p``. There are 17 such trees through order 5. The
    check is applied to the propagated weights and to the embedded companion, so the
    labelling of the pair is verified rather than assumed, which catches a swapped or
    mislabelled pair.

    This function reports; it does not raise. A caller that wants a hard failure asserts
    on ``matches_declared_orders``.

    Parameters
    ----------
    tableau : Tableau
        The tableau to measure.
    tolerance : float, optional
        Residual below which a condition counts as satisfied, dimensionless. The
        default 1e-15 is above the worst residual of either shipped tableau
        (8.0e-16, in Tsit5's order-5 conditions, whose coefficients are published
        decimals rather than exact rationals) and eleven orders below the residual
        of an order that is genuinely not attained (8.7e-4 for both embedded
        companions at order 5).

    Returns
    -------
    OrderConditionReport
        The residuals, the attained orders and whether they match the declared ones.

    Examples
    --------
    >>> from solprop.stepper import DOPRI5, check_order_conditions
    >>> report = check_order_conditions(DOPRI5)
    >>> report.solution_order, report.embedded_order
    (5, 4)
    >>> report.matches_declared_orders
    True
    >>> bool(report.max_residual < 1e-15)
    True
    """
    solution = _residuals(tableau.nodes, tableau.matrix, tableau.solution_weights)
    embedded = _residuals(tableau.nodes, tableau.matrix, tableau.embedded_weights)
    solution_order = _attained_order(solution, tolerance)
    embedded_order = _attained_order(embedded, tolerance)
    return OrderConditionReport(
        tolerance=tolerance,
        solution_residuals=solution,
        embedded_residuals=embedded,
        solution_order=solution_order,
        embedded_order=embedded_order,
        max_residual=max(
            _worst_residual(solution, tableau.order),
            _worst_residual(embedded, tableau.embedded_order),
        ),
        matches_declared_orders=(
            solution_order == tableau.order and embedded_order == tableau.embedded_order
        ),
    )


# The Dormand-Prince RK5(4)7M pair [1]_. The last matrix row and the solution weights
# are written out independently, rather than one derived from the other, so that the
# FSAL identity is a real check on the labelling and not a tautology.
_DOPRI5_NODES = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_DOPRI5_MATRIX = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1 / 5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3 / 40, 9 / 40, 0.0, 0.0, 0.0, 0.0, 0.0],
        [44 / 45, -56 / 15, 32 / 9, 0.0, 0.0, 0.0, 0.0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0.0, 0.0, 0.0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0.0, 0.0],
        [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0],
    ]
)
_DOPRI5_SOLUTION = np.array(
    [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]
)
_DOPRI5_EMBEDDED = np.array(
    [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
)

DOPRI5 = Tableau(
    name="DOPRI5",
    nodes=_DOPRI5_NODES,
    matrix=_DOPRI5_MATRIX,
    solution_weights=_DOPRI5_SOLUTION,
    error_weights=_DOPRI5_SOLUTION - _DOPRI5_EMBEDDED,
    order=5,
    embedded_order=4,
    first_same_as_last=True,
)
"""Dormand-Prince RK5(4)7M, the default scheme."""

DOPRI5_DENSE = np.array(
    [
        -12715105075 / 11282082432,
        0.0,
        87487479700 / 32700410799,
        -10690763975 / 1880347072,
        701980252875 / 199316789632,
        -1453857185 / 822651844,
        69997945 / 29380423,
    ]
)
"""DOPRI5's dense-output weights ``d``, Hairer, Noersett and Wanner's (II.6.12).

With ``r1 = y0``, ``r2 = y1 - y0``, ``r3 = h k1 - r2``, ``r4 = r2 - h k7 - r3`` and
``r5 = h sum_i d_i k_i``, the fourth-order continuous extension at ``theta`` in
``[0, 1]`` is ``r1 + theta (r2 + (1 - theta) (r3 + theta (r4 + (1 - theta) r5)))``
(:func:`continuous_extension`; SEP 14 phase 3).
"""


def continuous_extension(tableau: Tableau) -> Float[np.ndarray, " n_stages"] | None:
    """Return a tableau's dense-output weights, or ``None`` if it has none here.

    Parameters
    ----------
    tableau : Tableau
        The scheme.

    Returns
    -------
    numpy.ndarray or None
        :data:`DOPRI5_DENSE` for DOPRI5; ``None`` for any other tableau.

    Examples
    --------
    >>> from solprop.stepper.tableau import DOPRI5, TSIT5, continuous_extension
    >>> continuous_extension(DOPRI5).shape, continuous_extension(TSIT5)
    ((7,), None)
    """
    return DOPRI5_DENSE if tableau is DOPRI5 else None


def extension_coefficients(
    tableau: Tableau,
    start: np.ndarray,
    stages: Sequence[np.ndarray],
    dz: float,
) -> list[np.ndarray]:
    """Contract an accepted DOPRI5 step into its continuous extension's coefficients.

    Parameters
    ----------
    tableau : Tableau
        DOPRI5, the scheme with a continuous extension here.
    start : numpy.ndarray
        The state at the step's start, ``y0``.
    stages : sequence of numpy.ndarray
        The step's stage derivatives ``k_i``, one per stage.
    dz : float
        The step ``h``.

    Returns
    -------
    list of numpy.ndarray
        ``[y0, y1 - y0, h k1 - (y1 - y0), (y1 - y0) - h k7 - (h k1 - (y1 - y0)),
        h sum_i d_i k_i]``, with ``y1 = y0 + h sum_i b_i k_i``.

    Raises
    ------
    ValueError
        If the tableau has no continuous extension here.

    Examples
    --------
    A constant derivative extends linearly: half a step of ``y' = 2`` from 1 is 2.

    >>> import numpy as np
    >>> from solprop.stepper.tableau import (
    ...     DOPRI5, extension_coefficients, extension_value)
    >>> stages = [np.array([2.0])] * 7
    >>> coefficients = extension_coefficients(DOPRI5, np.array([1.0]), stages, 1.0)
    >>> extension_value(coefficients, 0.5).round(12).tolist()
    [2.0]
    """
    weights = continuous_extension(tableau)
    if weights is None:
        raise ValueError(f"{tableau.name} has no continuous extension here")
    span = np.zeros_like(start)
    dense = np.zeros_like(start)
    for solution, extension, stage in zip(
        tableau.solution_weights, weights, stages, strict=True
    ):
        span = span + float(solution) * stage
        dense = dense + float(extension) * stage
    span = dz * span
    first = dz * stages[0] - span
    last = span - dz * stages[-1] - first
    return [start, span, first, last, dz * dense]


def extension_value(coefficients: Sequence[np.ndarray], theta: float) -> np.ndarray:
    """Evaluate a continuous extension at a fraction of its step.

    Parameters
    ----------
    coefficients : sequence of numpy.ndarray
        From :func:`extension_coefficients`.
    theta : float
        The fraction of the step, in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        ``r1 + theta (r2 + (1 - theta) (r3 + theta (r4 + (1 - theta) r5)))``.
    """
    start, span, first, last, dense = coefficients
    rest = 1.0 - theta
    return start + theta * (span + rest * (first + theta * (last + rest * dense)))


# The Tsitouras RK5(4) pair [2]_. Its coefficients solve the order conditions
# numerically and have no exact rational form, so what is written here are the
# published decimals of the paper's tableau, to the 17 significant digits the standard
# reference implementations carry. What verifies them is the order conditions
# themselves, not the transcription: check_order_conditions puts the worst residual at
# 8.0e-16. The one exception is the last error weight, which is exactly 1/66.
#
# Transcriptions of this tableau exist in which the vector labelled "b5" is the error
# vector, not a set of weights, with the sign of its last entry also wrong; they must
# not be used as the source. See docs/dev/stepper.md section 10.
_TSIT5_NODES = np.array([0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0])
_TSIT5_MATRIX = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.161, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.008480655492356989, 0.335480655492357, 0.0, 0.0, 0.0, 0.0, 0.0],
        [
            2.8971530571054935,
            -6.359448489975075,
            4.3622954328695815,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            5.325864828439257,
            -11.748883564062828,
            7.4955393428898365,
            -0.09249506636175525,
            0.0,
            0.0,
            0.0,
        ],
        [
            5.86145544294642,
            -12.92096931784711,
            8.159367898576159,
            -0.071584973281401,
            -0.028269050394068383,
            0.0,
            0.0,
        ],
        [
            0.09646076681806523,
            0.01,
            0.4798896504144996,
            1.379008574103742,
            -3.290069515436081,
            2.324710524099774,
            0.0,
        ],
    ]
)
_TSIT5_SOLUTION = np.array(
    [
        0.09646076681806523,
        0.01,
        0.4798896504144996,
        1.379008574103742,
        -3.290069515436081,
        2.324710524099774,
        0.0,
    ]
)
_TSIT5_ERROR = np.array(
    [
        -0.00178001105222577714,
        -0.0008164344596567469,
        0.007880878010261995,
        -0.1447110071732629,
        0.5823571654525552,
        -0.45808210592918697,
        1 / 66,
    ]
)

TSIT5 = Tableau(
    name="Tsit5",
    nodes=_TSIT5_NODES,
    matrix=_TSIT5_MATRIX,
    solution_weights=_TSIT5_SOLUTION,
    error_weights=_TSIT5_ERROR,
    order=5,
    embedded_order=4,
    first_same_as_last=True,
)
"""Tsitouras RK5(4), the alternative scheme."""
