# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Fixed transverse quadrature rules and the mode matrices built on them.

The multimode transform integrates over the cross-section with one fixed rule chosen
once per problem, never an adaptive one (ADR-010). That is not a compromise: the Kerr
polarisation is cubic in the field, so its projection is a *quartic* in the transverse
profiles, and for Marcatili modes those are Bessel functions of :math:`r/a` times
:math:`\cos h\theta` -- analytic, and integrated to rounding by a Gauss rule of modest
order. An adaptive rule buys nothing and costs a data-dependent time per step and an
error estimate that can be fooled.

The rules:

**Radial**, :math:`n_r` Gauss-Legendre nodes on :math:`\xi = r/a \in (0, 1)`, with the
Jacobian and the azimuthal :math:`2\pi` folded into the weights,
:math:`w_i = 2\pi a^2 \xi_i w^{\rm GL}_i / 2`. With
``radial_rule="gauss_kronrod"`` the count is rounded up to :math:`2n+1` and the rule
becomes the Kronrod extension of the :math:`n`-point Gauss rule, whose Gauss subset is
an *embedded coarse rule* for the error estimate. Nodes are strictly interior, so the
wall is never sampled.

**Field scale** (SEP 19), for one Marcatili HE_11 mode in a transversely uniform
medium, where every node sees the axial waveform scaled by
:math:`\eta = J_0(u_{01}\xi)`: ``"field_scale"`` is the Gauss rule in :math:`\eta` for
the HE_11 projection measure and ``"field_scale_radau"`` its Radau variant with a node
on the axis (:func:`field_scale_rule`). They exist for responses so expensive that the
number of atoms per call matters, carry *area* weights, which are finite at the axis
and integrate the source rather than the area, and have no embedded coarse rule.

**Azimuthal**, :math:`n_\theta` equispaced nodes offset by half a step,
:math:`\theta_j = (j - \tfrac12) 2\pi/n_\theta`: the periodic trapezoid rule, which is
exact for trigonometric polynomials of degree at most :math:`n_\theta - 1`. The offset
matters. On-grid nodes at :math:`0, \pi/2, \pi, 3\pi/2` annihilate products such as
:math:`E_xE_y` for a superposition of hybrid modes, which can make an adaptive cubature
such as ``pcubature`` report convergence on a wrong value; the offset rule cannot be
fooled that way and there is no convergence decision in it to fool. A quartic product of
modes with azimuthal order up to :math:`h` reaches degree :math:`4h`, so
:func:`transverse_quadrature` warns below :math:`n_\theta = 4h + 1`.

**Cartesian**, Gauss-Legendre in both directions, for rectangular waveguides. A
product of trigonometric eigenfunctions has no finite exactness degree there, so the
bound is the measured :func:`cartesian_nodes`, :math:`6h + 6` per axis, and
:func:`transverse_quadrature` warns below it.

**Slab**, Gauss-Legendre in x alone on :math:`[-a, a]`, for the hollow planar guide
uniform in y (:mod:`solprop.modes.slab`). The rule emits ``(x, 0)`` points with the
weights :math:`a\,w_i` times a unit y weight, so everything downstream of the rule is
the Cartesian pipeline with one y node: an integral over it is a per-unit-width one. The
same :func:`cartesian_nodes` bound holds -- re-measured on the one-dimensional quartic
products -- and with ``radial_rule="gauss_kronrod"`` the x rule is the Kronrod
extension, so the slab has the quadrature alarm a Cartesian product rule lacks.

The Kronrod extension is computed, not tabulated, by Laurie's algorithm [1]_: build the
Jacobi-Kronrod matrix from the Legendre recurrence coefficients and take its eigenvalues
and first eigenvector components. That gives any order rather than the two that
:mod:`scipy.integrate` happens to ship, and it is checked node by node against QuadGK's
values in ``tests/data/luna_modes.json``.

**Tapers.** For a scale-invariant mode the profile depends on ``r`` only through
``r / a`` and the normalisation goes as :math:`a^2`, so at nodes placed at
:math:`r = a\xi` the normalised field is exactly proportional to :math:`1/a` and the
weights to :math:`a^2`. Moving along a taper therefore rescales the synthesis matrix by
:math:`a(0)/a(z)` and the projection matrix by :math:`a(z)/a(0)`, exactly, instead of
re-evaluating thousands of Bessel functions: :meth:`ModeMatrices.rescaled`.

**The dual pair.** The propagated basis is :math:`e_m/\sqrt{N_m}` with
:math:`N_m = (\varepsilon_0 c/2)\int|e_m|^2\,dA` (``conventions.md`` section 4), so its
overlaps are :math:`2/(\varepsilon_0 c)` rather than one, and the coefficient of a field
in that basis is :math:`(\varepsilon_0 c/2)\int P\,e_m/\sqrt{N_m}\,dA`. The projection
matrix therefore carries :data:`BASIS_NORMALISATION` beside the quadrature weights, so
that ``projection @ synthesis`` is the identity on an orthonormal set and a backend
applies the pair as it is.

The manual page is ``docs/guide/waveguides/mode-sets.md``.

References
----------
.. [1] D. P. Laurie, "Calculation of Gauss-Kronrod quadrature rules", Math. Comp. 66,
       1133-1145 (1997), doi:10.1090/S0025-5718-97-00861-2.
"""

import functools
import itertools
import logging
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, get_args

import numpy as np
from jaxtyping import Float
from scipy.linalg import eigh_tridiagonal, solve_banded
from scipy.optimize import brentq
from scipy.special import j0, jn_zeros, roots_legendre

from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes.protocol import CrossSection, Mode

__all__ = [
    "BASIS_NORMALISATION",
    "DEFAULT_N_THETA",
    "FIELD_SCALE_RULES",
    "ModeMatrices",
    "RadialRule",
    "TransverseQuadrature",
    "cartesian_nodes",
    "check_field_scale_modes",
    "field_scale_rule",
    "gauss_kronrod",
    "mode_matrix",
    "on_axis_rule",
    "quadrature_rule",
    "taper_scale",
    "transverse_quadrature",
]

BASIS_NORMALISATION: Final = EPS0 * C / 2.0
"""The factor that makes the projection the dual of the synthesis, in F/(m s).

The propagated basis ``e_m / sqrt(N_m)`` has overlaps ``2 / (eps0 c)`` and not one, so
the coefficient of a field in it is ``(eps0 c / 2) int P e_m/sqrt(N_m) dA``:
:func:`mode_matrix` folds this into both projections beside the quadrature weights.
Without it a multimode run is wrong by the constant ``753.5``.
"""

logger = logging.getLogger(__name__)

DEFAULT_N_THETA: Final = 16
"""Default azimuthal node count, exact for cubic products up to azimuthal order 3."""

_COMPONENTS: Final = ("x", "y")
"""The transverse Cartesian field components, in the order the kernel layout uses."""

RadialRule = Literal["gauss_kronrod", "gauss", "field_scale", "field_scale_radau"]
"""The first-coordinate rule families (SEP 19).

``"gauss_kronrod"`` and ``"gauss"`` are Gauss-Legendre on ``xi = r / a``, the former
with the Kronrod extension and its embedded coarse rule. ``"field_scale"`` and
``"field_scale_radau"`` are Gauss and Gauss-Radau rules in the HE_11 field scale
``eta = J_0(u_01 xi)`` (:func:`field_scale_rule`), valid for one Marcatili HE_11 mode
in a transversely uniform medium only."""

FIELD_SCALE_RULES: Final = ("field_scale", "field_scale_radau")
"""The rule families that are Gaussian in the HE_11 field scale."""

_U01: Final = float(jn_zeros(0, 1)[0])
"""The first zero of J_0: the HE_11 profile is ``J_0(_U01 r / a)``."""

_MINIMUM_NODES: Final = {
    "gauss_kronrod": 2,
    "gauss": 2,
    "field_scale": 1,
    "field_scale_radau": 4,
}
"""The smallest admitted ``n_r`` of each family (SEP 19 section 5.1).

One field-scale node is the effective-area special rule; a Radau rule is exact for the
Kerr term from three nodes, but three are 23% off on the SAE TDSE source, so four is
the floor."""

_MOMENT_TOLERANCE: Final = 1e-12
"""Relative moment residual a field-scale rule must reach against its refined measure.

SEP 19 section 7's target. The independent review of the prototypes measured a largest
residual of 5.6e-13 over orders 2 to 32 against adaptive Bessel integrals."""

_MEASURE_COUNTS: Final = (128, 256, 512, 1024, 2048, 4096, 8192)
"""Gauss-Legendre counts in xi tried for the discretised field-scale measure.

The integrands ``xi J_0(u xi)^(k+1)`` are entire, so the discrete moments converge
spectrally; the first count at which the recurrence coefficients agree with the next
to ``_RECURRENCE_TOLERANCE`` is taken, and a rule whose coefficients never settle
is refused rather than returned."""

_RECURRENCE_TOLERANCE: Final = 1e-13
"""Agreement of the recurrence coefficients between two measure refinements.

Relative to the largest coefficient. The coefficients settle by 256 points for every
order up to 64, and successive refinements then differ at a rounding floor of 1e-14 to
2e-14 that grows with the point count (2e-13 at 8192); the moment check of
:func:`field_scale_rule`, at ``_MOMENT_TOLERANCE``, is the certificate."""


def cartesian_nodes(order: int) -> int:
    """Return the nodes per axis a Cartesian quartic product needs.

    A product of trigonometric eigenfunctions has no finite polynomial
    exactness degree, so this is a measured bound rather than a derived one:
    ``6 * order + 6`` covers the counts that reach a relative 1e-12 on the
    quartic field product of a rectangular mode -- 12 nodes at order 1, 18 at
    2, 22 at 3, 26 at 4 and 35 at 6. Higher orders need their own resolution
    check.

    The one-dimensional quartic products of a slab set (:mod:`solprop.modes.slab`)
    were measured separately against 30-digit ``mpmath`` integrals: every product
    of orders up to ``h`` reaches 1e-12 of the ``int V^4`` scale at 12, 17, 22, 26,
    30 and 34 Gauss nodes for ``h = 1`` to 6, and stays there at every count up to
    ``6 h + 6``, so the same bound serves the slab's x rule.

    Parameters
    ----------
    order : int
        Highest transverse order of the set, ``max(n, m)``, >= 0.

    Returns
    -------
    int
        Nodes per axis for quartic products of that order.

    Examples
    --------
    >>> from solprop.modes.quadrature import cartesian_nodes
    >>> cartesian_nodes(1), cartesian_nodes(3)
    (12, 24)
    """
    return 6 * order + 6


def _legendre_recurrence(
    count: int,
) -> tuple[Float[np.ndarray, " n_coefficients"], Float[np.ndarray, " n_coefficients"]]:
    """Return the three-term recurrence coefficients of the Legendre polynomials.

    Parameters
    ----------
    count : int
        How many coefficients to return.

    Returns
    -------
    tuple of numpy.ndarray
        ``alpha`` (all zero, the weight being symmetric) and ``beta``, with
        ``beta[0]`` the measure of the interval.
    """
    alpha = np.zeros(count)
    beta = np.zeros(count)
    beta[0] = 2.0
    index = np.arange(1, count, dtype=float)
    beta[1:] = 1.0 / (4.0 - index**-2.0)
    return alpha, beta


def _jacobi_kronrod(
    n: int,
) -> tuple[Float[np.ndarray, " n_kronrod"], Float[np.ndarray, " n_kronrod"]]:
    """Return the Jacobi-Kronrod matrix coefficients by Laurie's mixed recursion.

    A direct transcription of Laurie's algorithm [1]_ as published by Gautschi in the
    OPQ suite; the index arithmetic is delicate and is not simplified here, so that it
    can be read against the source. It is validated against QuadGK's own nodes and
    weights, which use the same algorithm independently.

    Parameters
    ----------
    n : int
        Order of the underlying Gauss rule; the extension has ``2 n + 1`` points.

    Returns
    -------
    tuple of numpy.ndarray
        The diagonal and off-diagonal recurrence coefficients, length ``2 n + 1``.

    References
    ----------
    .. [1] D. P. Laurie, Math. Comp. 66, 1133-1145 (1997).
    """
    half_up = math.ceil(3 * n / 2)
    half_down = math.floor(3 * n / 2)
    alpha0, beta0 = _legendre_recurrence(half_up + 1)
    alpha = np.zeros(2 * n + 1)
    beta = np.zeros(2 * n + 1)
    alpha[: half_down + 1] = alpha0[: half_down + 1]
    beta[: half_up + 1] = beta0[: half_up + 1]
    size = n // 2 + 2
    s = np.zeros(size)
    t = np.zeros(size)
    t[1] = beta[n + 1]
    for m in range(n - 1):
        total = 0.0
        for k in range((m + 1) // 2, -1, -1):
            index = m - k
            total += (
                (alpha[k + n + 1] - alpha[index]) * t[k + 1]
                + beta[k + n + 1] * s[k]
                - beta[index] * s[k + 1]
            )
            s[k + 1] = total
        s, t = t, s
    for j in range(n // 2, -1, -1):
        s[j + 1] = s[j]
    j = 0
    for m in range(n - 1, 2 * n - 2):
        total = 0.0
        for k in range(m + 1 - n, (m - 1) // 2 + 1):
            index = m - k
            j = n - 1 - index
            total -= (
                (alpha[k + n + 1] - alpha[index]) * t[j + 1]
                + beta[k + n + 1] * s[j + 1]
                - beta[index] * s[j + 2]
            )
            s[j + 1] = total
        if m % 2 == 0:
            k = m // 2
            alpha[k + n + 1] = (
                alpha[k] + (s[j + 1] - beta[k + n + 1] * s[j + 2]) / t[j + 2]
            )
        else:
            k = (m + 1) // 2
            beta[k + n + 1] = s[j + 1] / s[j + 2]
        s, t = t, s
    alpha[2 * n] = alpha[n - 1] - beta[2 * n] * s[1] / t[1]
    return alpha, beta


def gauss_kronrod(
    n: int,
) -> tuple[
    Float[np.ndarray, " n_points"],
    Float[np.ndarray, " n_weights"],
    Float[np.ndarray, " n_coarse"],
]:
    """Return the ``2 n + 1`` point Kronrod extension of the n-point Gauss rule.

    Parameters
    ----------
    n : int
        Order of the underlying Gauss rule, >= 1.

    Returns
    -------
    tuple of numpy.ndarray
        Nodes on ``[-1, 1]`` in increasing order, their weights, and the embedded
        coarse weights: the Gauss weights at the Gauss nodes and zero elsewhere, so
        that the two rules share every node they have in common and their difference
        is an error estimate.

    Raises
    ------
    ModeError
        If `n` is less than 1.

    Examples
    --------
    >>> from solprop.modes.quadrature import gauss_kronrod
    >>> nodes, weights, coarse = gauss_kronrod(7)
    >>> nodes.size, int((coarse > 0.0).sum())
    (15, 7)
    >>> bool(abs(weights.sum() - 2.0) < 16 * np.spacing(2.0))
    True
    """
    if n < 1:
        raise ModeError(f"a Kronrod extension needs n >= 1, got {n}")
    alpha, beta = _jacobi_kronrod(n)
    nodes, vectors = eigh_tridiagonal(alpha, np.sqrt(beta[1 : 2 * n + 1]))
    weights = beta[0] * vectors[0, :] ** 2
    coarse = np.zeros_like(weights)
    # The Gauss nodes are every other Kronrod node, starting from the second: the
    # extension interlaces them. Placing the Gauss weights there and nothing elsewhere
    # makes the coarse rule a strict subset of the fine one.
    coarse[1::2] = roots_legendre(n)[1]
    return nodes, weights, coarse


def _radial_rule(
    n_r: int, kronrod: bool
) -> tuple[
    Float[np.ndarray, " n_r"], Float[np.ndarray, " n_r"], Float[np.ndarray, " n_r"]
]:
    """Return the radial nodes on ``(0, 1)`` with their fine and coarse weights.

    Parameters
    ----------
    n_r : int
        Requested node count; rounded up to ``2 n + 1`` for a Kronrod rule.
    kronrod : bool
        Whether to use the Kronrod extension and its embedded coarse rule.

    Returns
    -------
    tuple of numpy.ndarray
        ``xi``, ``w`` and ``w_coarse`` on ``xi = r / a``. Without a Kronrod extension
        there is no embedded rule and the coarse weights equal the fine ones.
    """
    if kronrod:
        # The smallest n with 2n + 1 >= n_r.
        nodes, weights, coarse = gauss_kronrod(max(-(-(n_r - 1) // 2), 1))
    else:
        nodes, weights = roots_legendre(n_r)
        coarse = weights
    # Map [-1, 1] onto (0, 1); the Jacobian of the map is the halving of the weights.
    return (nodes + 1.0) / 2.0, weights / 2.0, coarse / 2.0


def _field_scale_measure(
    count: int,
) -> tuple[Float[np.ndarray, " n_points"], Float[np.ndarray, " n_points"]]:
    """Return the HE_11 projection measure in the field scale, discretised.

    ``d mu = J_0(u xi) xi d xi`` written on ``eta = J_0(u xi)``: the points are the
    field scales at a ``count``-point Gauss-Legendre rule in ``xi`` on ``(0, 1)`` and
    the weights carry ``xi eta`` and the Jacobian of the map.

    Parameters
    ----------
    count : int
        Gauss-Legendre points in ``xi``.

    Returns
    -------
    tuple of numpy.ndarray
        ``eta`` and the discrete measure, both of length `count`.
    """
    nodes, weights = roots_legendre(count)
    xi = 0.5 * (nodes + 1.0)
    eta = j0(_U01 * xi)
    return eta, 0.5 * weights * xi * eta


def _lanczos(
    points: Float[np.ndarray, " n_points"],
    measure: Float[np.ndarray, " n_points"],
    n: int,
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
    """Return the first ``n`` recurrence coefficients of a discrete measure.

    The Lanczos process on ``diag(points)`` started from ``sqrt(measure)``
    (Gautschi, section 2.2.3.2 of the reference in :func:`field_scale_rule`), with
    full reorthogonalisation, applied twice: it produces the Jacobi matrix of the
    discrete measure, and unlike the discretised Stieltjes procedure it does not lose
    orthogonality at high order. ``beta[0]`` is the total mass.

    Parameters
    ----------
    points, measure : numpy.ndarray
        The support and the positive masses of the measure.
    n : int
        Coefficients wanted, >= 1 and at most the number of points.

    Returns
    -------
    tuple of numpy.ndarray
        ``alpha`` and ``beta``, each of length `n`.
    """
    alpha = np.zeros(n)
    beta = np.zeros(n)
    beta[0] = float(np.sum(measure))
    basis = np.zeros((points.size, n))
    basis[:, 0] = np.sqrt(measure / beta[0])
    for k in range(n):
        vector = points * basis[:, k]
        alpha[k] = float(basis[:, k] @ vector)
        if k + 1 < n:
            for _ in range(2):
                vector -= basis[:, : k + 1] @ (basis[:, : k + 1].T @ vector)
            norm = float(np.linalg.norm(vector))
            beta[k + 1] = norm**2
            basis[:, k + 1] = vector / norm
    return alpha, beta


def _field_scale_recurrence(
    n: int,
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"], int]:
    """Return converged recurrence coefficients of the field-scale measure.

    Parameters
    ----------
    n : int
        Coefficients wanted, >= 1.

    Returns
    -------
    tuple
        ``alpha``, ``beta`` and the measure's point count at which they converged.

    Raises
    ------
    ModeError
        If no count in ``_MEASURE_COUNTS`` settles the coefficients.
    """
    previous = None
    for count in _MEASURE_COUNTS:
        alpha, beta = _lanczos(*_field_scale_measure(count), n)
        if previous is not None:
            scale = max(float(np.max(np.abs(alpha))), float(np.max(beta)))
            change = max(
                float(np.max(np.abs(alpha - previous[0]))),
                float(np.max(np.abs(beta - previous[1]))),
            )
            if change <= _RECURRENCE_TOLERANCE * scale:
                return alpha, beta, count
        previous = (alpha, beta)
    msg = (
        f"the field-scale recurrence of order {n} did not settle within "
        f"{_MEASURE_COUNTS[-1]} measure points"
    )
    raise ModeError(msg)


@functools.cache
def field_scale_rule(
    n: int, *, radau: bool = False
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
    r"""Return a Gauss or Gauss-Radau rule in the HE_11 field scale.

    For one HE_11 mode in a transversely uniform medium every node sees the same
    waveform scaled by :math:`\eta = J_0(u_{01}\xi)`, so the radial projection is
    :math:`\int_0^1 F[\eta A]\,d\mu(\eta)` with
    :math:`d\mu = J_0(u_{01}\xi)\,\xi\,d\xi` (SEP 19 section 4). The Gauss rule for
    :math:`\mu` is computed from its recurrence coefficients, by the Lanczos process on
    a refined Gauss-Legendre discretisation in :math:`\xi` [1]_, and the
    Golub-Welsch eigenvalue method [2]_; it is exact for polynomials in
    :math:`\eta` of degree :math:`2n - 1`, which from two nodes includes the linear
    and Kerr terms. The Radau rule fixes one node at :math:`\eta = 1`, the axis, by
    replacing the last diagonal entry with
    :math:`1 - \beta_{n-1} e^T(I - J_{n-1})^{-1}e` [3]_ and is exact to degree
    :math:`2n - 2`. One Gauss node is the effective-area special rule, not the one-point
    Gauss rule: :math:`\eta_e^2 = M_3/M_1` and :math:`\lambda_e = M_3/\eta_e^3`, with
    :math:`M_k = \int\eta^k d\mu`, which matches moments 1 and 3.

    Every rule is checked before it is returned: finite positive weights, distinct
    nodes in :math:`(0, 1]`, and every moment it should integrate against a measure
    refined once more, to a relative ``1e-12``.

    Parameters
    ----------
    n : int
        Nodes, >= 1 for the Gauss family and >= 2 for Radau.
    radau : bool, optional
        Fix one node on the axis.

    Returns
    -------
    tuple of numpy.ndarray
        ``eta`` in decreasing order (increasing radius) and the weights ``lambda`` of
        :math:`\mu`, read-only.

    Raises
    ------
    ModeError
        If the order is too small or the rule fails its own check.

    References
    ----------
    .. [1] W. Gautschi, *Orthogonal Polynomials: Computation and Approximation*
       (Oxford University Press, 2004), doi:10.1093/oso/9780198506720.001.0001.
    .. [2] G. H. Golub and J. H. Welsch, Math. Comp. 23, 221 (1969),
       doi:10.1090/S0025-5718-69-99647-1.
    .. [3] G. H. Golub, SIAM Rev. 15, 318 (1973), doi:10.1137/1015032.

    Examples
    --------
    >>> from solprop.modes.quadrature import field_scale_rule
    >>> eta, weights = field_scale_rule(5)
    >>> eta.size, bool(eta[0] < 1.0)
    (5, True)
    >>> radau, _ = field_scale_rule(5, radau=True)
    >>> float(radau[0])
    1.0
    """
    if n < (2 if radau else 1):
        msg = f"a field-scale {'Radau ' if radau else ''}rule needs more nodes, got {n}"
        raise ModeError(msg)
    if n == 1:
        points, measure = _field_scale_measure(_MEASURE_COUNTS[0])
        first = float(np.sum(measure * points))
        third = float(np.sum(measure * points**3))
        eta = np.array([math.sqrt(third / first)])
        weights = np.array([third / eta[0] ** 3])
        degree = None
        count = _MEASURE_COUNTS[0]
    else:
        alpha, beta, count = _field_scale_recurrence(n)
        if radau:
            # The leading block minus the identity is tridiagonal: solve, never invert.
            off = np.sqrt(beta[1 : n - 1])
            banded = np.zeros((3, n - 1))
            banded[0, 1:] = -off
            banded[1, :] = 1.0 - alpha[: n - 1]
            banded[2, :-1] = -off
            unit = np.zeros(n - 1)
            unit[-1] = 1.0
            alpha = alpha.copy()
            alpha[n - 1] = 1.0 - beta[n - 1] * float(
                solve_banded((1, 1), banded, unit)[-1]
            )
        eta, vectors = eigh_tridiagonal(alpha, np.sqrt(beta[1:n]))
        weights = beta[0] * vectors[0, :] ** 2
        order = np.argsort(eta)[::-1]
        eta, weights = eta[order], weights[order]
        if radau:
            if abs(eta[0] - 1.0) > 1e-10:
                msg = f"the Radau rule of order {n} lost its axis node ({eta[0]!r})"
                raise ModeError(msg)
            eta[0] = 1.0
        degree = 2 * n - 2 if radau else 2 * n - 1
    _certify(n, eta, weights, degree, count)
    for values in (eta, weights):
        values.setflags(write=False)
    return eta, weights


def _certify(
    n: int,
    eta: Float[np.ndarray, " n"],
    weights: Float[np.ndarray, " n"],
    degree: int | None,
    count: int,
) -> None:
    """Check a field-scale rule against a measure refined once more.

    Parameters
    ----------
    n : int
        The order, for the message.
    eta, weights : numpy.ndarray
        The rule.
    degree : int or None
        The polynomial degree it must integrate exactly; ``None`` for the one-node
        special rule, which must match moments 1 and 3 only.
    count : int
        The measure point count the rule was built from.

    Raises
    ------
    ModeError
        If a weight is not finite and positive, a node is outside ``(0, 1]`` or
        repeated, or a moment misses ``_MOMENT_TOLERANCE``.
    """
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or np.any(eta <= 0.0)
        or np.any(eta > 1.0)
        or np.any(np.diff(eta) >= 0.0)
    ):
        msg = f"the field-scale rule of order {n} has an invalid node or weight"
        raise ModeError(msg)
    points, measure = _field_scale_measure(2 * count)
    powers = (1, 3) if degree is None else range(degree + 1)
    for k in powers:
        exact = float(np.sum(measure * points**k))
        residual = abs(float(np.sum(weights * eta**k)) - exact)
        if residual > _MOMENT_TOLERANCE * exact:
            msg = (
                f"the field-scale rule of order {n} misses moment {k} by "
                f"{residual / exact:.1e} relative"
            )
            raise ModeError(msg)


def _xi_of_eta(eta: float) -> float:
    """Return the radius ``xi = r / a`` at which the HE_11 field scale equals `eta`.

    Parameters
    ----------
    eta : float
        A field scale in ``(0, 1]``.

    Returns
    -------
    float
        The root of ``J_0(u xi) = eta`` in ``[0, 1)``; exactly 0 on the axis.
    """
    if eta >= 1.0:
        return 0.0
    return float(
        brentq(
            lambda xi: float(j0(_U01 * xi)) - eta,
            0.0,
            1.0,
            xtol=4 * np.finfo(float).eps,
            rtol=4 * np.finfo(float).eps,
        )
    )


def _field_scale_radial(
    n: int, *, radau: bool
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
    """Return a field-scale rule as radii and dimensionless radial area weights.

    Parameters
    ----------
    n : int
        Nodes.
    radau : bool
        Fix one node on the axis.

    Returns
    -------
    tuple of numpy.ndarray
        ``xi`` in increasing order and ``q = lambda / eta``, the weights of
        ``int f xi d xi``: the area form, finite at the axis, which the ``d xi`` form
        would need to divide by ``xi`` (SEP 19 section 4.2). They are source-adapted
        and do not integrate a constant area density exactly.
    """
    eta, weights = field_scale_rule(n, radau=radau)
    xi = np.array([_xi_of_eta(float(value)) for value in eta])
    return xi, weights / eta


def check_field_scale_modes(modes: Sequence[Mode]) -> None:
    """Refuse a field-scale rule for anything but one Marcatili HE_11 profile.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set the rule would integrate.

    Raises
    ------
    ModeError
        Naming the mode set, if it has more than one mode, or a mode whose transverse
        profile is not Marcatili's ``J_0(u_01 r / a)``: a mode merely labelled HE_11
        with another radial profile does not qualify.
    """
    from solprop.modes.capillary import MarcatiliMode  # noqa: PLC0415 - a cycle

    mode = modes[0] if len(modes) == 1 else None
    if (
        mode is None
        or not isinstance(mode, MarcatiliMode)
        or type(mode).field is not MarcatiliMode.field
        or (mode.kind, mode.n, mode.m) != ("HE", 1, 1)
    ):
        names = ", ".join(repr(mode) for mode in modes)
        msg = (
            "a field-scale radial rule needs exactly one Marcatili HE_11 mode, whose "
            f"field at every node is the axial waveform scaled by J_0; got [{names}]"
        )
        raise ModeError(msg)


def _azimuthal_rule(
    n_theta: int, full: bool
) -> tuple[
    Float[np.ndarray, " n_theta"],
    Float[np.ndarray, " n_theta"],
    Float[np.ndarray, " n_theta"],
]:
    """Return the azimuthal nodes with their fine and coarse weights.

    Parameters
    ----------
    n_theta : int
        Node count for the full rule; ignored when `full` is false.
    full : bool
        Whether the integrand varies with the azimuthal angle.

    Returns
    -------
    tuple of numpy.ndarray
        ``theta``, ``w`` and ``w_coarse``. The azimuthally symmetric rule is one node
        at ``theta = 0`` carrying the whole ``2 pi``.
    """
    if not full:
        single = np.array([2.0 * math.pi])
        return np.zeros(1), single, single
    step = 2.0 * math.pi / n_theta
    # Offset by half a step: on-grid nodes at 0, pi/2, pi and 3 pi/2 annihilate the
    # very products this rule exists to integrate.
    theta = (np.arange(n_theta) + 0.5) * step
    weights = np.full(n_theta, step)
    if n_theta % 2 == 0 and n_theta >= 4:
        coarse = np.where(np.arange(n_theta) % 2 == 0, 2.0 * step, 0.0)
    else:
        coarse = weights
    return theta, weights, coarse


@dataclass(frozen=True, eq=False)
class TransverseQuadrature:
    """A fixed product rule over the cross-section of a waveguide.

    Node ``p = i + j * n_r`` is coordinate-1 node ``i`` and coordinate-2 node ``j``, so
    the radial index varies fastest.

    The nodes and weights here are on the reference cross-section; :meth:`nodes` and
    :meth:`weights` map them onto the geometry a mode declares, folding the Jacobian
    into the weights.

    Attributes
    ----------
    kind : str
        ``"polar"``, ``"cartesian"`` or ``"slab"``.
    full : bool
        Whether the second coordinate is resolved. A polar rule with ``full=False`` is
        radial only, valid when every product of the mode fields is azimuthally
        symmetric -- any set of HE_1m modes. A slab rule is never full: its one y
        node sits at ``y = 0`` with unit weight.
    n_r : int
        Nodes in the first coordinate, after any rounding up for a Kronrod rule.
        With `breaks` it is the TOTAL count across the panels, each of which
        carries the requested count, so a rule with one break holds twice what
        was asked for.
    n_theta : int
        Nodes in the second coordinate; 1 for a radial-only rule.
    kronrod : bool
        Whether the first coordinate carries an embedded coarse rule.
    xi_r : numpy.ndarray
        First-coordinate nodes on the reference interval.
    w_r, w_r_coarse : numpy.ndarray or None
        First-coordinate weights for ``d xi`` (polar) or ``dx`` on the reference
        interval; ``None`` when the rule carries area weights instead.
    xi_theta, w_theta, w_theta_coarse : numpy.ndarray
        Second-coordinate nodes and weights.
    breaks : tuple of float
        Interior radial panel boundaries on the unit interval, in ascending order.
    radial_rule : str
        The first-coordinate family, one of :data:`RadialRule`.
    area_r, area_r_coarse : numpy.ndarray or None
        Dimensionless radial **area** weights ``q`` of ``int f xi d xi``, for a
        field-scale rule (SEP 19 section 4.2), whose Radau member has a node on the
        axis where the ``d xi`` form would divide by zero. Exactly one of ``w_r`` and
        ``area_r`` is present. They are source-adapted: they integrate the linear and
        Kerr source exactly, not a constant area density, and must not be
        renormalised to the geometrical area.
    requested_n_r : int or None
        The first-coordinate count asked of :func:`quadrature_rule`, before any
        Kronrod rounding or panel multiplication; ``None`` for a rule built directly.

    Raises
    ------
    ModeError
        If both or neither weight representation is present, or area weights are
        given for anything but a radial-only polar rule without panels.
    """

    kind: CrossSection
    full: bool
    n_r: int
    n_theta: int
    kronrod: bool
    xi_r: Float[np.ndarray, " n_first"]
    w_r: Float[np.ndarray, " n_first_weights"] | None
    w_r_coarse: Float[np.ndarray, " n_first_coarse"] | None
    xi_theta: Float[np.ndarray, " n_second"]
    w_theta: Float[np.ndarray, " n_second_weights"]
    w_theta_coarse: Float[np.ndarray, " n_second_coarse"]
    breaks: tuple[float, ...] = ()
    radial_rule: RadialRule = "gauss"
    area_r: Float[np.ndarray, " n_first_area"] | None = None
    area_r_coarse: Float[np.ndarray, " n_first_area_coarse"] | None = None
    requested_n_r: int | None = None

    def __post_init__(self) -> None:
        """Check that exactly one radial weight representation is active.

        Raises
        ------
        ModeError
            As described in the class docstring.
        """
        legacy = self.w_r is not None and self.w_r_coarse is not None
        area = self.area_r is not None and self.area_r_coarse is not None
        partial = (self.w_r is None) != (self.w_r_coarse is None) or (
            self.area_r is None
        ) != (self.area_r_coarse is None)
        if partial or legacy == area:
            msg = (
                "a transverse rule carries either d-xi weights (w_r and w_r_coarse) "
                "or radial area weights (area_r and area_r_coarse), exactly one pair"
            )
            raise ModeError(msg)
        if (self.radial_rule in FIELD_SCALE_RULES) != area:
            msg = (
                "area weights belong to the field-scale rules and to no other, got "
                f"{self.radial_rule!r} with "
                f"{'area' if area else 'd-xi'} weights"
            )
            raise ModeError(msg)
        if area and (self.kind != "polar" or self.full or self.breaks):
            msg = "area weights need a radial-only polar rule without panels"
            raise ModeError(msg)

    @property
    def n_nodes(self) -> int:
        """Return the number of quadrature nodes.

        Returns
        -------
        int
            ``n_r * n_theta``.
        """
        return self.n_r * self.n_theta

    @property
    def has_embedded_rule(self) -> bool:
        """Return whether the coarse weights differ from the fine ones.

        Returns
        -------
        bool
            ``True`` when there is a genuine embedded rule, so that the difference of
            the two projections is an error estimate rather than identically zero.
        """
        fine, coarse = (
            (self.w_r, self.w_r_coarse)
            if self.area_r is None
            else (self.area_r, self.area_r_coarse)
        )
        assert fine is not None and coarse is not None  # one pair, by construction
        return not (
            np.array_equal(fine, coarse)
            and np.array_equal(self.w_theta, self.w_theta_coarse)
        )

    def _limits(
        self, dimlimits: tuple[CrossSection, tuple[float, ...], tuple[float, ...]]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Check that a mode's geometry matches this rule and return its limits.

        Parameters
        ----------
        dimlimits : tuple
            As returned by :meth:`solprop.modes.protocol.Mode.dimlimits`.

        Returns
        -------
        tuple
            The lower and upper limits.

        Raises
        ------
        ModeError
            If the cross-section kind does not match the rule.
        """
        kind, lower, upper = dimlimits
        if kind != self.kind:
            raise ModeError(
                f"this rule is {self.kind}, but the mode's cross-section is {kind}"
            )
        return lower, upper

    def nodes(
        self, dimlimits: tuple[CrossSection, tuple[float, ...], tuple[float, ...]]
    ) -> Float[np.ndarray, "n_nodes 2"]:
        """Return the quadrature nodes on a mode's cross-section.

        Parameters
        ----------
        dimlimits : tuple
            As returned by :meth:`solprop.modes.protocol.Mode.dimlimits`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_nodes, 2)``: ``(r, theta)`` in m and rad, or ``(x, y)`` in m;
            a slab rule's ``y`` is zero at every node.

        Raises
        ------
        ModeError
            If the cross-section kind does not match the rule.

        Examples
        --------
        >>> import math
        >>> from solprop.modes.quadrature import quadrature_rule
        >>> rule = quadrature_rule("polar", full=False, n_r=4)
        >>> rule.nodes(("polar", (0.0, 0.0), (100e-6, 2 * math.pi))).shape
        (4, 2)
        """
        lower, upper = self._limits(dimlimits)
        if self.kind == "polar":
            first = upper[0] * self.xi_r
            second = self.xi_theta
        elif self.kind == "slab":
            first = (
                0.5 * (upper[0] + lower[0]) + 0.5 * (upper[0] - lower[0]) * self.xi_r
            )
            second = self.xi_theta
        else:
            first = (
                0.5 * (upper[0] + lower[0]) + 0.5 * (upper[0] - lower[0]) * self.xi_r
            )
            second = (
                0.5 * (upper[1] + lower[1])
                + 0.5 * (upper[1] - lower[1]) * self.xi_theta
            )
        grid_first, grid_second = np.meshgrid(first, second, indexing="ij")
        return np.stack(
            [grid_first.ravel(order="F"), grid_second.ravel(order="F")], axis=-1
        )

    def record(
        self, dimlimits: tuple[CrossSection, tuple[float, ...], tuple[float, ...]]
    ) -> dict[str, object]:
        """Return the rule's identity for the run metadata and its resume key.

        Parameters
        ----------
        dimlimits : tuple
            As returned by :meth:`solprop.modes.protocol.Mode.dimlimits`.

        Returns
        -------
        dict
            The family, the requested and realised counts, the total node count,
            whether there is an embedded rule, the panels, whether this is the
            one-node effective-area rule of SEP 19, and the nodes and physical
            weights on that cross-section (m and m^2, or m for a slab).

        Examples
        --------
        >>> import math
        >>> from solprop.modes.quadrature import quadrature_rule
        >>> rule = quadrature_rule(
        ...     "polar", full=False, n_r=32, radial_rule="gauss_kronrod"
        ... )
        >>> record = rule.record(("polar", (0.0, 0.0), (100e-6, 2 * math.pi)))
        >>> record["radial_rule"], record["requested_n_r"], record["n_nodes"]
        ('gauss_kronrod', 32, 33)
        """
        return {
            "radial_rule": self.radial_rule,
            "kind": self.kind,
            "full": self.full,
            "requested_n_r": self.requested_n_r,
            "n_r": self.n_r,
            "n_theta": self.n_theta,
            "n_nodes": self.n_nodes,
            "embedded_rule": self.has_embedded_rule,
            "breaks": list(self.breaks),
            "effective_area_node": self.radial_rule == "field_scale" and self.n_r == 1,
            "nodes": self.nodes(dimlimits),
            "weights": self.weights(dimlimits),
        }

    def weights(
        self,
        dimlimits: tuple[CrossSection, tuple[float, ...], tuple[float, ...]],
        *,
        coarse: bool = False,
    ) -> Float[np.ndarray, " n_nodes"]:
        """Return the quadrature weights, with the area element folded in.

        Parameters
        ----------
        dimlimits : tuple
            As returned by :meth:`solprop.modes.protocol.Mode.dimlimits`.
        coarse : bool, optional
            Return the embedded coarse rule's weights instead of the fine ones.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_nodes,)``. Polar: ``a^2 xi_i w_i w_j``, the Jacobian ``r`` of
            ``r dr dtheta`` included, or ``a^2 q_i w_j`` for a rule carrying radial
            area weights ``q``, which already include it. Cartesian: the product of
            the half-widths and the two weights. Slab: the half-gap times the x
            weight, times the unit y weight, in m -- a per-unit-width measure.

        Raises
        ------
        ModeError
            If the cross-section kind does not match the rule.

        Examples
        --------
        >>> import math
        >>> from solprop.modes.quadrature import quadrature_rule
        >>> rule = quadrature_rule("polar", full=False, n_r=8)
        >>> limits = ("polar", (0.0, 0.0), (100e-6, 2 * math.pi))
        >>> area = float(rule.weights(limits).sum())
        >>> bool(abs(area / (math.pi * 100e-6**2) - 1.0) < 1e-15)
        True
        """
        lower, upper = self._limits(dimlimits)
        second = self.w_theta_coarse if coarse else self.w_theta
        if self.area_r is not None:
            area = self.area_r_coarse if coarse else self.area_r
            assert area is not None
            return np.asarray(np.outer(upper[0] ** 2 * area, second).ravel(order="F"))
        first = self.w_r_coarse if coarse else self.w_r
        assert first is not None
        if self.kind == "polar":
            radial = upper[0] ** 2 * self.xi_r * first
        elif self.kind == "slab":
            radial = 0.5 * (upper[0] - lower[0]) * first
        else:
            radial = 0.5 * (upper[0] - lower[0]) * first
            second = 0.5 * (upper[1] - lower[1]) * second
        return np.asarray(np.outer(radial, second).ravel(order="F"))


def _check_request(
    kind: str,
    full: bool,
    n_r: object,
    n_theta: int,
    radial_rule: str,
    breaks: tuple[float, ...],
) -> None:
    """Refuse a transverse rule request that :func:`quadrature_rule` cannot build.

    Typed loosely on purpose: it is the guard behind :func:`quadrature_rule`'s own
    ``Literal`` annotations, so it must see and name a value those would reject.

    Parameters
    ----------
    kind : str
        The cross-section, as :func:`quadrature_rule`.
    full : bool
        Whether the second coordinate is resolved.
    n_r : object
        The first-coordinate count, checked to be an integer here.
    n_theta : int
        The second-coordinate count.
    radial_rule : str
        The first-coordinate family, checked against :data:`RadialRule` here.
    breaks : tuple of float
        Radial panel boundaries.

    Raises
    ------
    ModeError
        As :func:`quadrature_rule` describes.
    """
    if kind not in ("polar", "cartesian", "slab"):
        raise ModeError(f"kind must be 'polar', 'cartesian' or 'slab', got {kind!r}")
    if radial_rule not in get_args(RadialRule):
        msg = (
            f"radial_rule must be one of {', '.join(get_args(RadialRule))}, got "
            f"{radial_rule!r}"
        )
        raise ModeError(msg)
    if isinstance(n_r, bool) or not isinstance(n_r, (int, np.integer)):
        raise ModeError(f"n_r must be an integer, got {n_r!r}")
    if radial_rule in FIELD_SCALE_RULES and (kind != "polar" or full or breaks):
        msg = (
            f"the {radial_rule!r} rule is radial-only on a polar core without panels: "
            f"got a {'full ' if full else ''}{kind} rule"
            f"{' with breaks' if breaks else ''}"
        )
        raise ModeError(msg)
    if kind == "cartesian" and not full:
        raise ModeError("a cartesian rule has no azimuthally symmetric form")
    if kind == "slab" and full:
        raise ModeError(
            "a slab rule is uniform in y and has no second coordinate to resolve: "
            "its one y node carries a unit weight (per unit width)"
        )
    if n_r < _MINIMUM_NODES[radial_rule]:
        msg = (
            f"a {radial_rule!r} rule needs n_r >= {_MINIMUM_NODES[radial_rule]}, "
            f"got {n_r}"
        )
        raise ModeError(msg)
    if full and kind != "slab" and n_theta < 2:
        raise ModeError(f"a full rule needs n_theta >= 2, got {n_theta}")
    if breaks and (
        kind != "polar"
        or not all(math.isfinite(value) and 0 < value < 1 for value in breaks)
        or any(right <= left for left, right in itertools.pairwise(breaks))
    ):
        raise ModeError(
            "radial breaks must increase strictly inside (0, 1) on a polar rule"
        )


def quadrature_rule(
    kind: CrossSection,
    *,
    full: bool,
    n_r: int,
    n_theta: int = DEFAULT_N_THETA,
    radial_rule: RadialRule = "gauss",
    breaks: tuple[float, ...] = (),
) -> TransverseQuadrature:
    """Build a transverse rule from its geometry alone.

    :func:`transverse_quadrature` is the entry point for a mode set; this is the one
    for a bare cross-section, which the tests and the rectangular waveguides use.

    Parameters
    ----------
    kind : str
        ``"polar"``, ``"cartesian"`` or ``"slab"``.
    full : bool
        Whether to resolve the second coordinate. A Cartesian rule must; a slab
        rule must not, having no y to resolve.
    n_r : int
        Nodes in the first coordinate: >= 2, rounded up to the smallest odd count
        (at least 3) for ``"gauss_kronrod"``; >= 1 for ``"field_scale"``, whose one-node
        member is the effective-area rule; >= 4 for ``"field_scale_radau"``.
    n_theta : int, optional
        Nodes in the second coordinate, >= 2 when `full`; ignored by a slab rule.
    radial_rule : str, optional
        The first-coordinate family, one of :data:`RadialRule`.
        ``"gauss_kronrod"`` uses the Kronrod extension, giving an embedded coarse rule
        for the transform's error estimate; a Cartesian product rule has none and
        reads it as ``"gauss"``. The field-scale families need a radial-only polar
        rule without `breaks`; which modes they may integrate is checked where the
        modes are known (:func:`check_field_scale_modes`).
    breaks : tuple of float, optional
        Strictly increasing radial panel boundaries inside (0, 1), for polar rules.
        Each panel receives `n_r` nodes before any Kronrod rounding.

    Returns
    -------
    TransverseQuadrature
        The rule.

    Raises
    ------
    ModeError
        If the kind or the rule family is unknown, a Cartesian rule is asked not to
        be full, a slab rule is asked to be full, a node count is not an integer or
        is too small, or a field-scale rule is asked of anything but a radial-only
        polar cross-section without panels.

    Examples
    --------
    >>> from solprop.modes.quadrature import quadrature_rule
    >>> rule = quadrature_rule("polar", full=True, n_r=16, n_theta=8)
    >>> rule.n_nodes
    128
    >>> slab = quadrature_rule("slab", full=False, n_r=12)
    >>> limits = ("slab", (-50e-6,), (50e-6,))
    >>> slab.n_nodes, bool(slab.nodes(limits)[:, 1].any())
    (12, False)
    >>> f"{slab.weights(limits).sum():.1e}"
    '1.0e-04'
    """
    _check_request(kind, full, n_r, n_theta, radial_rule, breaks)
    field_scale = radial_rule in FIELD_SCALE_RULES
    kronrod = radial_rule == "gauss_kronrod"
    area_r = None
    if field_scale:
        xi_r, area_r = _field_scale_radial(
            int(n_r), radau=radial_rule == "field_scale_radau"
        )
        w_r = w_r_coarse = None
        xi_theta, w_theta, w_theta_coarse = _azimuthal_rule(n_theta, full)
    elif kind == "polar":
        xi_r, w_r, w_r_coarse = _radial_rule(n_r, kronrod)
        if breaks:
            edges = (0.0, *breaks, 1.0)
            widths = np.diff(edges)
            xi_r = np.concatenate(
                [
                    left + width * xi_r
                    for left, width in zip(edges, widths, strict=False)
                ]
            )
            w_r = np.concatenate([width * w_r for width in widths])
            w_r_coarse = np.concatenate([width * w_r_coarse for width in widths])
        xi_theta, w_theta, w_theta_coarse = _azimuthal_rule(n_theta, full)
    elif kind == "slab":
        if kronrod:
            # The smallest n with 2n + 1 >= n_r, as the radial rule rounds.
            xi_r, w_r, w_r_coarse = gauss_kronrod(max(-(-(n_r - 1) // 2), 1))
        else:
            xi_r, w_r = roots_legendre(n_r)
            w_r_coarse = w_r
        # One node at y = 0 carrying unit width: the integral over the rule is an
        # integral over x per metre of y.
        one = np.ones(1)
        xi_theta, w_theta, w_theta_coarse = np.zeros(1), one, one
    else:
        nodes, weights = roots_legendre(n_r)
        xi_r, w_r, w_r_coarse = nodes, weights, weights
        second_nodes, second_weights = roots_legendre(n_theta)
        xi_theta, w_theta, w_theta_coarse = (
            second_nodes,
            second_weights,
            second_weights,
        )
    return TransverseQuadrature(
        kind=kind,
        full=full,
        n_r=xi_r.size,
        n_theta=xi_theta.size,
        kronrod=kronrod and kind in ("polar", "slab"),
        xi_r=xi_r,
        w_r=w_r,
        w_r_coarse=w_r_coarse,
        xi_theta=xi_theta,
        w_theta=w_theta,
        w_theta_coarse=w_theta_coarse,
        breaks=breaks,
        radial_rule=radial_rule if field_scale or kind != "cartesian" else "gauss",
        area_r=area_r,
        area_r_coarse=area_r,
        requested_n_r=int(n_r),
    )


def transverse_quadrature(
    modes: Sequence[Mode],
    *,
    n_r: int,
    n_theta: int = DEFAULT_N_THETA,
    radial_rule: RadialRule = "gauss",
    full: bool | None = None,
    z: float = 0.0,
) -> TransverseQuadrature:
    """Build the transverse rule a mode set needs.

    The geometry comes from the first mode's cross-section, and whether the azimuthal
    coordinate has to be resolved from the modes' azimuthal orders: a set of HE_1m
    modes has azimuthally symmetric products and needs the radial rule only.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set the rule will integrate, non-empty.
    n_r : int
        Nodes in the first coordinate; the minimum depends on the family
        (:func:`quadrature_rule`).
    n_theta : int, optional
        Nodes in the azimuthal coordinate, >= 2. Ignored by a radial-only rule.
    radial_rule : str, optional
        The first-coordinate family, one of :data:`RadialRule`: ``"gauss_kronrod"``
        gives an embedded coarse rule; ``"field_scale"`` and ``"field_scale_radau"``
        are Gaussian in the HE_11 field scale and need exactly one Marcatili HE_11
        mode (SEP 19).
    full : bool or None, optional
        Force the azimuthal coordinate on or off. ``None`` decides from the modes; a
        slab set is never full.
    z : float, optional
        Position along the waveguide the geometry is taken at, m.

    Returns
    -------
    TransverseQuadrature
        The rule.

    Raises
    ------
    ModeError
        If `modes` is empty, a field-scale rule is asked of anything but one
        Marcatili HE_11 mode, or the rule cannot be built.

    Warns
    -----
    UserWarning
        If the azimuthal rule is too coarse for a quartic product of these modes, which
        needs ``n_theta >= 4 h + 1`` with ``h`` the highest azimuthal order, or if
        either axis of a Cartesian rule, or the x rule of a slab, is below
        :func:`cartesian_nodes` of the highest transverse order. The rule is still
        built, because a caller may know the products it will actually integrate;
        the warning is the only thing standing between an aliased Kerr term and a
        plausible-looking answer.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.quadrature import transverse_quadrature
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2)]
    >>> rule = transverse_quadrature(modes, n_r=24)
    >>> rule.full, rule.n_nodes
    (False, 24)
    """
    if not modes:
        raise ModeError("a transverse rule needs at least one mode")
    if radial_rule in FIELD_SCALE_RULES:
        check_field_scale_modes(modes)
    # One call: dimlimits is an eigensolve for a step-index mode, and both the
    # cross-section kind and the outer radius the breaks are scaled by come
    # from it.
    limits = modes[0].dimlimits(z)
    kind = limits[0]
    orders = [mode.azimuthal_order for mode in modes]
    highest = max(orders)
    if kind == "slab":
        resolve = False if full is None else full
    else:
        resolve = (highest > 0 or kind == "cartesian") if full is None else full
    if resolve and kind == "polar" and n_theta < 4 * highest + 1:
        warnings.warn(
            f"n_theta = {n_theta} is below the exactness bound 4 * {highest} + 1 for "
            "quartic products of modes with azimuthal order up to "
            f"{highest}; the Kerr and Raman terms will alias. Use n_theta >= "
            f"{4 * highest + 1}.",
            UserWarning,
            stacklevel=2,
        )
    if kind == "cartesian" and min(n_r, n_theta) < cartesian_nodes(highest):
        warnings.warn(
            f"n_r = {n_r} and n_theta = {n_theta} are the x and y node counts of a "
            f"Cartesian rule, and one is below the measured bound "
            f"6 * {highest} + 6 for quartic products of modes with transverse order "
            f"up to {highest}; the Kerr and Raman terms will alias. Use "
            f"{cartesian_nodes(highest)} nodes per axis.",
            UserWarning,
            stacklevel=2,
        )
    if kind == "slab" and n_r < cartesian_nodes(highest):
        warnings.warn(
            f"n_r = {n_r} is the x node count of a slab rule and is below the measured "
            f"bound 6 * {highest} + 6 for quartic products of slab modes of order up "
            f"to {highest}; the Kerr and Raman terms will alias. Use "
            f"{cartesian_nodes(highest)} nodes.",
            UserWarning,
            stacklevel=2,
        )
    radial_breaks: Callable[[float], tuple[float, ...]] | None = getattr(
        modes[0], "radial_breaks", None
    )
    breaks = (
        tuple(value / limits[2][0] for value in radial_breaks(z))
        if radial_breaks is not None
        else ()
    )
    rule = quadrature_rule(
        kind,
        full=resolve,
        n_r=n_r,
        n_theta=n_theta,
        radial_rule=radial_rule,
        breaks=breaks,
    )
    logger.debug(
        "transverse rule: %s, full=%s, n_r=%d, n_theta=%d, radial rule %s, %d nodes, "
        "highest azimuthal order %d",
        rule.kind,
        rule.full,
        rule.n_r,
        rule.n_theta,
        rule.radial_rule,
        rule.n_nodes,
        highest,
    )
    return rule


@dataclass(frozen=True, eq=False)
class ModeMatrices:
    """Synthesis and projection matrices in the layout the kernel contract expects.

    Synthesis takes modal amplitudes to the field at the quadrature nodes; projection
    takes a polarisation at the nodes back to modal amplitudes, with the quadrature
    weights folded in. Column ``i * n_pol + p`` of the projection -- row of the
    synthesis -- is polarisation component ``p`` at node ``i``, so that a
    ``(n_to, n_pol, n_nodes)`` view of the real-space arrays is free and each node's
    column is contiguous (``docs/dev/kernel-contract.md`` section 4).

    Attributes
    ----------
    synthesis : numpy.ndarray
        Shape ``(n_nodes * n_pol, n_modes)``: the normalised transverse fields.
    projection : numpy.ndarray
        Shape ``(n_modes, n_nodes * n_pol)``: the dual basis -- the same fields with
        the quadrature weights and :data:`BASIS_NORMALISATION` folded in, so that
        ``projection @ synthesis`` is the identity on an orthonormal set.
    projection_coarse : numpy.ndarray
        The projection under the embedded coarse rule; equal to `projection` when the
        rule has no embedded pair, in which case the error estimate is identically
        zero and must not be read as convergence.
    n_nodes : int
        Quadrature nodes.
    n_pol : int
        Polarisation components carried, 1 or 2.
    """

    synthesis: Float[np.ndarray, "n_rows n_modes"]
    projection: Float[np.ndarray, "n_modes n_columns"]
    projection_coarse: Float[np.ndarray, "n_modes_coarse n_columns_coarse"]
    n_nodes: int
    n_pol: int

    def rescaled(self, scale: float) -> "ModeMatrices":  # noqa: UP037 - bound after the class body
        """Return the matrices for a scale-invariant mode set at a new core radius.

        Parameters
        ----------
        scale : float
            The ratio ``a(z_reference) / a(z)``, > 0.

        Returns
        -------
        ModeMatrices
            Synthesis multiplied by `scale` and projection divided by it, which is
            exact: the normalised field goes as ``1 / a`` at nodes placed at
            ``r = a xi``, and the weights as ``a^2``.

        Raises
        ------
        ModeError
            If `scale` is not positive.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.modes.quadrature import mode_matrix, transverse_quadrature
        >>> modes = [MarcatiliMode(100e-6, "He", 1.0)]
        >>> rule = transverse_quadrature(modes, n_r=8)
        >>> matrices = mode_matrix(modes, rule)
        >>> matrices.rescaled(2.0).synthesis.shape
        (16, 1)
        """
        if not scale > 0.0:
            raise ModeError(f"a taper rescaling must be positive, got {scale}")
        return ModeMatrices(
            synthesis=self.synthesis * scale,
            projection=self.projection / scale,
            projection_coarse=self.projection_coarse / scale,
            n_nodes=self.n_nodes,
            n_pol=self.n_pol,
        )


def on_axis_rule(kind: CrossSection) -> "TransverseQuadrature":  # noqa: UP037 - bound after the class body
    """Return a one-node rule at the transverse origin of a waveguide.

    The axis is not a quadrature node -- a Gauss rule on ``(0, 1)`` never places one
    there -- so the on-axis intensity needs the mode fields evaluated at ``r = 0``. The
    rule is the vehicle for that: :func:`solprop.modes.quadrature.mode_matrix` is what
    knows how to normalise a mode's profile, and giving it a rule keeps that knowledge
    in the physics layer where it belongs. Its weights are never used.

    Parameters
    ----------
    kind : str
        The polar, Cartesian or slab cross-section the mode set declares; a slab's
        origin is its mid-plane ``x = 0``.

    Returns
    -------
    solprop.modes.quadrature.TransverseQuadrature
        A rule with one node at the origin.

    Raises
    ------
    ModeError
        If the cross-section kind is unknown.
    """
    if kind not in ("polar", "cartesian", "slab"):
        raise ModeError(f"unknown cross-section {kind!r} for on-axis intensity")
    one = np.array([0.0])
    return TransverseQuadrature(
        kind=kind,
        full=kind != "slab",
        n_r=1,
        n_theta=1,
        kronrod=False,
        xi_r=one,
        w_r=one,
        w_r_coarse=one,
        xi_theta=one,
        w_theta=one,
        w_theta_coarse=one,
    )


def mode_matrix(
    modes: Sequence[Mode],
    quadrature: TransverseQuadrature,
    z: float = 0.0,
    *,
    components: Sequence[str] = _COMPONENTS,
) -> ModeMatrices:
    """Build the synthesis and projection matrices of a mode set on a rule.

    Takes the rule rather than bare nodes: the projection needs the weights, and
    passing the rule keeps the two from ever being paired wrongly.

    Every mode is evaluated on the cross-section of the *first*, which is what makes
    one product rule serve the whole set; a set whose limits disagree is rejected.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set, non-empty.
    quadrature : TransverseQuadrature
        The rule, whose cross-section kind must match the modes'.
    z : float, optional
        Position along the waveguide, m.
    components : sequence of str, optional
        Which transverse components to carry: ``("x", "y")`` for a vector run, or one
        of them for a scalar run.

    Returns
    -------
    ModeMatrices
        The matrices.

    Raises
    ------
    ModeError
        If `modes` is empty, the components are not a subset of ``("x", "y")``, the
        modes do not share a cross-section, or a field-scale rule is paired with
        anything but one Marcatili HE_11 mode -- checked here as well as in
        :func:`transverse_quadrature`, so that a rule built by :func:`quadrature_rule`
        cannot reach a translation with the wrong profile.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.quadrature import mode_matrix, transverse_quadrature
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2, 3)]
    >>> rule = transverse_quadrature(modes, n_r=24)
    >>> matrices = mode_matrix(modes, rule, components=("x",))
    >>> matrices.synthesis.shape, matrices.projection.shape
    ((24, 3), (3, 24))
    """
    if not modes:
        raise ModeError("mode matrices need at least one mode")
    if quadrature.radial_rule in FIELD_SCALE_RULES:
        check_field_scale_modes(modes)
    if not components or any(name not in _COMPONENTS for name in components):
        raise ModeError(
            f"components must be a non-empty subset of {_COMPONENTS}, got "
            f"{tuple(components)}"
        )
    limits = modes[0].dimlimits(z)
    # islice rather than a slice, because a mode set may be an ordered collection whose
    # indexing is by position only.
    for mode in itertools.islice(modes, 1, None):
        if mode.dimlimits(z) != limits:
            raise ModeError(
                "every mode in a set must share a cross-section; "
                f"{limits} and {mode.dimlimits(z)} differ"
            )
    points = quadrature.nodes(limits)
    weights = quadrature.weights(limits)
    coarse = quadrature.weights(limits, coarse=True)
    indices = [_COMPONENTS.index(name) for name in components]
    n_pol = len(indices)
    n_nodes = points.shape[0]
    fields = np.empty((n_nodes * n_pol, len(modes)))
    for column, mode in enumerate(modes):
        # Exy = e / sqrt(N): the normalisation that makes a modal amplitude squared a
        # power in watts (docs/dev/conventions.md section 4).
        profile = mode.field(points, z) / math.sqrt(_normalisation(mode, z))
        fields[:, column] = profile[:, indices].ravel()
    return ModeMatrices(
        synthesis=fields,
        projection=BASIS_NORMALISATION
        * (fields * np.repeat(weights, n_pol)[:, None]).T,
        projection_coarse=BASIS_NORMALISATION
        * (fields * np.repeat(coarse, n_pol)[:, None]).T,
        n_nodes=n_nodes,
        n_pol=n_pol,
    )


def _normalisation(mode: Mode, z: float) -> float:
    """Return the power carried by a mode's unnormalised transverse profile.

    Taken from the mode's own closed form. Integrating it on the same rule the
    matrices use would make the Gram matrix exactly the identity whatever the rule,
    which would turn the orthonormality test into a tautology; the closed form is what
    makes that test say something.

    Parameters
    ----------
    mode : Mode
        The mode, which must provide ``normalisation(z)``.
    z : float
        Position along the waveguide, m.

    Returns
    -------
    float
        ``N`` in W.

    Raises
    ------
    ModeError
        If the mode has no closed-form normalisation.
    """
    closed_form: Callable[[float], float] | None = getattr(mode, "normalisation", None)
    if not callable(closed_form):
        raise ModeError(
            f"{type(mode).__name__} provides no normalisation(z); without it the "
            "modal amplitudes would not be in square-root watts"
        )
    return float(closed_form(z))


def taper_scale(mode: Mode, z: float, reference: float = 0.0) -> float:
    """Return the factor a scale-invariant mode's matrices rescale by.

    Parameters
    ----------
    mode : Mode
        The mode, which must report ``scale_invariant``.
    z : float
        Position along the waveguide, m.
    reference : float, optional
        Position the matrices were built at, m.

    Returns
    -------
    float
        ``a(reference) / a(z)``, to be passed to :meth:`ModeMatrices.rescaled`. For a
        slab it is the square root of that: the normalised field goes as
        ``1/sqrt(a)`` there, because ``N`` goes as ``a`` and not ``a^2``, and the
        weights as ``a``.

    Raises
    ------
    ModeError
        If the mode is not scale invariant.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.modes.quadrature import taper_scale
    >>> from solprop.profiles import Linear
    >>> mode = MarcatiliMode(Linear(100e-6, 50e-6, 1.0), "He", 1.0)
    >>> f"{taper_scale(mode, 1.0):.1f}"
    '2.0'
    """
    if not mode.scale_invariant:
        raise ModeError(
            f"{type(mode).__name__} is not scale invariant, so its matrices must be "
            "rebuilt rather than rescaled"
        )
    kind, lower, upper = mode.dimlimits(z)
    if kind == "slab":
        # N = eps0 c a / 2 per unit width, so the normalised profile at nodes
        # x = a xi goes as a^(-1/2) and the weights as a: the pair rescales by the
        # square root of the gap ratio, which keeps rescaled() exact.
        _, reference_lower, reference_upper = mode.dimlimits(reference)
        return math.sqrt(
            (reference_upper[0] - reference_lower[0]) / (upper[0] - lower[0])
        )
    if kind == "cartesian":
        _, reference_lower, reference_upper = mode.dimlimits(reference)
        return float((reference_upper[0] - reference_lower[0]) / (upper[0] - lower[0]))
    return float(mode.dimlimits(reference)[2][0] / upper[0])
