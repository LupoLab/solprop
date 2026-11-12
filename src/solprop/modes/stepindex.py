# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Solve the exact, lossless vector eigenproblem of a circular step-index fibre.

The characteristic equation is evaluated after clearing its Bessel denominators;
sign changes at Bessel poles are therefore never accepted as eigenvalues.
TE means ``A + B = 0`` and TM means ``A + (n_cl/n_co)**2 B = 0``.
The transverse profile is evaluated at an explicitly declared wavelength.
See ``docs/guide/waveguides/step-index.md`` for the Snyder and Love references.
"""

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass, field, replace
from functools import cache, cached_property
from types import MappingProxyType
from typing import Literal

import numpy as np
from jaxtyping import Complex, Float
from numpy.typing import ArrayLike
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import jn_zeros, jv, jvp, kve

from solprop._display import Displayed, engineering, line
from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes.protocol import ChannelLabel, CrossSection
from solprop.physdata.glasses import GlassIndex, fused_silica, lossless_index
from solprop.profiles import ZProfile, as_profile

__all__ = ["PROFILE_MIN_W", "StepIndexMode"]

PROFILE_MIN_W = 0.1
"""Smallest cladding decay the frozen transverse profile is evaluated at.

The outer radius is ``a max(2, 1 + log(1e8)/w)``, so ``w = 0.1`` puts
it at about 185 core radii and anything smaller integrates a nearly unguided
field over a disc no fixed rule resolves. It bounds the *frozen profile* only:
:meth:`StepIndexMode.neff` has no floor and reports whatever the eigenvalue is.
"""

logger = logging.getLogger(__name__)

_Kind = Literal["HE", "EH", "TE", "TM"]
_Loss = Callable[[Float[np.ndarray, " n_omega"]], Float[np.ndarray, " n_omega"]]
_EPS = np.finfo(float).eps
_MAX_V = 20000.0
"""Largest fibre frequency V at which the root scan runs.

The scan for radial order m covers u from the family's first cutoff to its
(m+1)-th, so its cost grows with m and not with V: HE11 over a 256-point band
takes about 27 ms at V = 1663, 4976 and 19905 alike (Apple M5 Max, 2026-11-02).
Resolution is not what sets the limit either: with each family's own factor
bracketing the roots the product loses (:func:`_recovered`), every count was
complete and aligned at each V tried up to 20000, for n = 0 to 3 and 40 to 300,
and the closest pair of the two families was still 3.4e-4 apart at V = 20000.
The limit is where the scan is qualified, by complete counts over V and against
a 40-digit solve of the unreduced equation up to V = 19990
(``tests/modes/test_stepindex.py``).
"""


def _k_ratio(
    n: int, w: Float[np.ndarray, "*shape"] | float
) -> Float[np.ndarray, "*shape"]:
    """Return ``K_(n-1)(w)/(w K_n(w))`` for a positive order, finite at any w > 0.

    Every ratio of modified Bessel functions of one argument is formed from
    ``kve``, which carries the factor ``exp(w)`` that cancels in the ratio: the
    unscaled ``kv`` underflows to zero from an argument of about 700. At small w
    and large n, ``kve(n, w)`` itself overflows (``K_n(w) ~ (n-1)!/2 (2/w)**n``),
    and there the quotient ``q_n = K_(n-1)/K_n`` is carried up from
    ``q_1 = K_0/K_1`` by ``q_(v+1) = 1/(q_v + 2 v/w)``, the recurrence
    ``K_(v+1) = K_(v-1) + (2 v/w) K_v`` divided through. K is the dominant
    solution of that recurrence in increasing order, so the forward quotient is
    stable: an error in ``q_v`` is multiplied by ``q_(v+1)**2 < 1``.

    - Returns: the positive ratio, with the shape of `w`.
    """
    upper = kve(n, w)
    finite = np.isfinite(upper)
    if finite.all():
        return np.asarray(kve(n - 1, w) / (w * upper))
    w, upper = np.asarray(w, dtype=float), np.asarray(upper)
    ratio = np.empty_like(w)
    ratio[finite] = kve(n - 1, w[finite]) / (w[finite] * upper[finite])
    small = w[~finite]
    quotient = kve(0, small) / kve(1, small)
    for order in range(1, n):
        quotient = 1 / (quotient + 2 * order / small)
    ratio[~finite] = quotient / small
    return ratio


def _log_slope(
    n: int, w: Float[np.ndarray, "*shape"] | float
) -> Float[np.ndarray, "*shape"]:
    """Return the logarithmic derivative K_n'(w)/K_n(w), negative and finite at w > 0.

    ``K_n' = -(K_(n-1) + K_(n+1))/2`` from scaled functions where ``kve(n + 1, w)``
    is finite, and ``K_n'/K_n = -K_(n-1)/K_n - n/w`` from :func:`_k_ratio` where
    it overflows.
    """
    upper = kve(n + 1, w)
    finite = np.isfinite(upper)
    if finite.all():
        return np.asarray(-(kve(abs(n - 1), w) + upper) / (2 * kve(n, w)))
    w, upper = np.asarray(w, dtype=float), np.asarray(upper)
    # n >= 1 here: kve(1, w) is finite for every representable w > 0.
    slope = np.empty_like(w)
    near = w[finite]
    slope[finite] = -(kve(abs(n - 1), near) + upper[finite]) / (2 * kve(n, near))
    small = w[~finite]
    slope[~finite] = -(small * _k_ratio(n, small) + n / small)
    return slope


def _k_decay(
    order: int, w: float, radius: Float[np.ndarray, " n_points"]
) -> Float[np.ndarray, " n_points"]:
    """Return ``K_order(w r)/K_order(w)`` for r >= 1, finite where kve overflows.

    The direct form takes scaled functions and the explicit ``exp(-w (r - 1))``
    of their two arguments. Where ``kve(order, w)`` overflows, the ratio is
    built up from order 1 by the same quotient recurrence as :func:`_k_ratio`,
    run at both arguments: each step multiplies by
    ``(K_(v+1)/K_v)(w r)/(K_(v+1)/K_v)(w)``, at most one for r >= 1, so the
    product can only underflow, to the negligible field it describes.

    - ``radius``: dimensionless radii r/a, at least one.
    - Returns: the ratio, between zero and one for order >= 1, with the shape of
      `radius`.
    """
    decay = np.exp(-w * (radius - 1))
    denominator = kve(order, w)
    if math.isfinite(denominator):
        return np.asarray(kve(order, w * radius) / denominator * decay)
    argument = w * radius
    ratio = kve(1, argument) / kve(1, w) * decay
    outer, inner = kve(0, argument) / kve(1, argument), kve(0, w) / kve(1, w)
    for step in range(1, order):
        grown_outer, grown_inner = outer + 2 * step / argument, inner + 2 * step / w
        ratio = ratio * grown_outer / grown_inner
        outer, inner = 1 / grown_outer, 1 / grown_inner
    return np.asarray(ratio)


def _characteristic(
    u: Float[np.ndarray, "*shape"] | float,
    v: Float[np.ndarray, "*shape"] | float,
    ratio: Float[np.ndarray, "*shape"] | float,
    n: int,
    kind: _Kind,
) -> Float[np.ndarray, "*shape"]:
    """Evaluate the pole-free product with positive Bessel and endpoint factors removed.

    - ``u``: dimensionless core transverse frequency, strictly between zero and V.
    - ``ratio``: squared real cladding/core index ratio, between zero and one.
    """
    w = np.sqrt((v - u) * (v + u))
    bessel = jv(n, u)
    if kind in ("TE", "TM"):
        first = u * w**2 * jvp(n, u)
        second = u**2 * w * bessel * _log_slope(n, w)
        return np.asarray(first + (1.0 if kind == "TE" else ratio) * second)
    # Expand with J_n'=n J_n/u-J_(n+1), K_n'=-n K_n/w-K_(n-1).
    # Cancelling the endpoint factors symbolically prevents rounding in two
    # nearly equal products from manufacturing roots at u=0 or w=0.
    h = jv(n + 1, u) / u
    k = _k_ratio(n, w)
    return np.asarray(
        -2 * n**2 * (1 + ratio) * bessel**2
        - n * h * bessel * (2 * w**2 - (1 + ratio) * u**2)
        - n * k * bessel**2 * ((1 + ratio) * w**2 - 2 * ratio * u**2)
        + u**2 * w**2 * (h**2 + (1 + ratio) * h * k * bessel + ratio * k**2 * bessel**2)
    )


def _partials(
    u: float, w: float, ratio: float, n: int, kind: _Kind
) -> tuple[float, float, float]:
    """Differentiate with respect to u, w and the squared index ratio.

    - ``u``: dimensionless core transverse frequency, strictly between zero and V.
    - ``ratio``: squared real cladding/core index ratio, between zero and one.
    - Returns: partial derivatives with respect to u, w and the squared index ratio.

    Notes
    -----
    The scaling differs from :func:`_characteristic` by ``u**2 w**2``: this
    function differentiates ``u**4 w**4 J_n**2`` times the cleared equation,
    where :func:`_characteristic` evaluates ``u**2 w**2 J_n**2`` times it. Both
    quotients are the same at a root, where the extra positive factor cancels,
    and a test ties them together at three roots.
    """
    bessel, slope, curvature = float(jv(n, u)), float(jvp(n, u)), float(jvp(n, u, 2))
    logarithmic = float(_log_slope(n, w))
    # K_n'' = (K_(n-2) + 2 K_n + K_(n+2))/4, scaled like the first derivative;
    # where kve(n + 2, w) overflows, Bessel's equation gives the same quotient,
    # K_n''/K_n = 1 + n**2/w**2 - (K_n'/K_n)/w.
    top = kve(n + 2, w)
    curvature_k = (
        (kve(abs(n - 2), w) + 2 * kve(n, w) + top) / (4 * kve(n, w))
        if math.isfinite(top)
        else 1 + n**2 / w**2 - logarithmic / w
    )
    dlogarithmic = float(curvature_k) - logarithmic**2
    first = u * w**2 * slope
    second = u**2 * w * bessel * logarithmic
    first_u, first_w = w**2 * (slope + u * curvature), 2 * u * w * slope
    second_u = w * logarithmic * (2 * u * bessel + u**2 * slope)
    second_w = u**2 * bessel * (logarithmic + w * dlogarithmic)
    if kind in ("TE", "TM"):
        factor = 1.0 if kind == "TE" else ratio
        return (
            first_u + factor * second_u,
            first_w + factor * second_w,
            0.0 if kind == "TE" else second,
        )
    left, right = first + second, first + ratio * second
    total, weighted = u**2 + w**2, w**2 + ratio * u**2
    du = (first_u + second_u) * right + left * (first_u + ratio * second_u)
    du -= n**2 * (
        2 * u * (weighted + ratio * total) * bessel**2
        + 2 * total * weighted * bessel * slope
    )
    dw = (first_w + second_w) * right + left * (first_w + ratio * second_w)
    dw -= 2 * n**2 * w * (weighted + total) * bessel**2
    dratio = left * second - n**2 * total * u**2 * bessel**2
    return du, dw, dratio


def _family(u: float, v: float, ratio: float, n: int) -> _Kind:
    """Identify the vanishing branch of the quadratic in J_n'/(u J_n).

    - ``u``: dimensionless core transverse frequency, strictly between zero and V.
    - ``v``: positive dimensionless fibre frequency V, matching u in shape.
    - ``ratio``: squared real cladding/core index ratio, between zero and one.
    """
    w = math.sqrt((v - u) * (v + u))
    a = float(jvp(n, u) / (u * jv(n, u)))
    b = float(_log_slope(n, w)) / w
    centre = a + (1.0 + ratio) * b / 2.0
    radical = math.sqrt(
        ((1.0 - ratio) * b / 2.0) ** 2
        + n**2 * (1 / u**2 + 1 / w**2) * (1 / u**2 + ratio / w**2)
    )
    return "HE" if abs(centre + radical) < abs(centre - radical) else "EH"


@cache
def _zero_block(order: int, size: int) -> Float[np.ndarray, " n_zeros"]:
    """Cache one block of the first zeros of one Bessel order.

    - ``size``: a power of two (:func:`_zeros` rounds up).
    - Returns: the first `size` zeros of J_order, dimensionless, read-only.
    """
    block = jn_zeros(order, size)
    block.flags.writeable = False
    return block


def _zeros(order: int, count: int) -> Float[np.ndarray, " n_zeros"]:
    """Return the first zeros of one Bessel order from a cached block.

    The cutoffs ask for every prefix length up to the mode count, about V/pi, so
    one cache entry per length would cost quadratically in V. ``jn_zeros``
    returns the same leading zeros, bit for bit, whatever count it is asked for,
    so a block rounded up to a power of two is computed once and sliced.

    - Returns: the first `count` zeros of J_order, dimensionless, read-only.
    """
    return _zero_block(order, max(64, 1 << (count - 1).bit_length()))[:count]


@cache
def _bessel_nodes(n: int, count: int) -> Float[np.ndarray, " n_zeros"]:
    """Cache the Bessel-zero scan nodes: the zeros of orders n-1, n and n+1."""
    return np.concatenate([_zeros(abs(order), count) for order in (n - 1, n, n + 1)])


def _cutoffs(
    v: float, ratio: float, n: int, kind: _Kind, count: int | None = None
) -> list[float]:
    """Return one family's exact cutoffs below V, in increasing order.

    A family has exactly one guided root per cutoff below V, and its m-th root
    lies between its m-th and (m+1)-th cutoff, so this list is both
    the expected root count and the interval each root must fall in.

    - ``ratio``: squared real cladding/core index ratio, between zero and one.
    - ``count``: positive number of lowest cutoffs wanted; all of them below V when
      omitted.
    - Returns: cutoffs of that family strictly below V, at most `count`, dimensionless.
    """
    if kind in ("TE", "TM", "EH"):
        limit = max(1, math.ceil(v / math.pi) + 1) if count is None else count
        zeros = _zeros(0 if kind in ("TE", "TM") else n, limit)
        return zeros[zeros < v].tolist()
    # The HE cutoffs rise with m (0 or a J_1 zero for n = 1, and inside
    # [j_(n-2,m), j_(n-1,m)] otherwise), so counting up terminates at the first
    # cutoff at or above V.
    found: list[float] = []
    while (count is None or len(found) < count) and (
        cutoff := _cutoff("HE", n, len(found) + 1, ratio)
    ) < v:
        found.append(cutoff)
    return found


def _check_frequency(v: float) -> None:
    """Refuse a fibre frequency above the largest one the root scan is qualified at."""
    if v > _MAX_V:
        raise ModeError(
            f"V={v:.16g} exceeds {_MAX_V:g}, the largest V the step-index root "
            "scan supports: its roots are qualified against an independent "
            "high-precision solve and by complete counts only up to that limit"
        )


def _refined_roots(  # noqa: PLR0917 - called positionally
    nodes: Float[np.ndarray, " n_nodes"],
    values: Float[np.ndarray, " n_nodes"],
    v: float,
    ratio: float,
    n: int,
    kind: _Kind,
) -> dict[str, list[float]]:
    """Refine every sign change on a node set and label each root by family.

    The hybrid families share one product, so one scan serves HE and EH.

    - ``ratio``: squared real cladding/core index ratio, between zero and one.
    - Returns: family name to its roots u in increasing order.
    """

    def equation(one: float) -> float:
        """Evaluate the characteristic at one trial root."""
        return float(_characteristic(one, v, ratio, n, kind))

    found: dict[str, list[float]] = {"HE": [], "EH": [], "TE": [], "TM": []}
    for index in np.flatnonzero(np.signbit(values[:-1]) != np.signbit(values[1:])):
        root = brentq(
            equation,
            float(nodes[index]),
            float(nodes[index + 1]),
            xtol=4 * math.ulp(float(nodes[index + 1])),
            rtol=4 * _EPS,
        )
        label = kind if kind in ("TE", "TM") else _family(root, v, ratio, n)
        found[label].append(root)
    return found


def _branch(
    u: Float[np.ndarray, "*shape"] | float, v: float, ratio: float, n: int, kind: _Kind
) -> Float[np.ndarray, "*shape"]:
    """Evaluate the one factor of the hybrid product that vanishes on one family.

    With ``x = J_n'/(u J_n)``, ``b = K_n'/(w K_n)`` and ``R`` the radical of
    :func:`_family`, the hybrid product is ``u**2 w**2`` times
    ``(J_n'/u - J_n x_HE)(J_n'/u - J_n x_EH)``, where ``x_HE = -(1 + ratio) b/2 - R``
    and ``x_EH = -(1 + ratio) b/2 + R``. Each factor is pole-free, has the roots
    of its own family only, and so has one root per cutoff interval: two roots of
    different families, however close, are two separate sign changes here,
    while in the product they cancel inside one scan interval.

    The HE factor is written with ``R - (1 + ratio) b/2`` rationalised, so that the
    terms in ``1/w**4`` that cancel near cutoff are cancelled symbolically.

    - ``u``: dimensionless core transverse frequencies, strictly between zero and V.
    - ``ratio``: squared real cladding/core index ratio, between zero and one.
    - Returns: the family's factor at each u, with the shape of `u`; only its sign and
      its roots are used.
    """
    w = np.sqrt((v - u) * (v + u))
    bessel = jv(n, u)
    slope = n * bessel / u**2 - jv(n + 1, u) / u
    # -b = n/w**2 + K_(n-1)/(w K_n) > 0, from K_n' = -n K_n/w - K_(n-1).
    k = _k_ratio(n, w)
    minus_b = n / w**2 + k
    radical = np.sqrt(
        ((1 - ratio) * minus_b / 2) ** 2
        + n**2 * (1 / u**2 + 1 / w**2) * (1 / u**2 + ratio / w**2)
    )
    total = radical + (1 + ratio) * minus_b / 2
    if kind == "EH":
        return np.asarray(slope - bessel * total)
    # R**2 - ((1 + ratio) b/2)**2 with the n**2 ratio/w**4 terms cancelled.
    difference = (
        n**2 / u**4
        + n**2 * (1 + ratio) / (u**2 * w**2)
        - 2 * ratio * n * k / w**2
        - ratio * k**2
    )
    return np.asarray(slope + bessel * difference / total)


def _recovered(
    nodes: Float[np.ndarray, " n_nodes"],
    roots: list[float],
    limits: list[float],
    scan: tuple[float, float, int, _Kind],
) -> list[float]:
    """Add the hybrid roots the product scan lost, one per empty cutoff interval.

    A root of one family and a root of the other closer together than the scan
    spacing are two sign changes of the product in one interval, which cancel.
    Each family's own factor (:func:`_branch`) changes sign there once, so it
    brackets the lost root wherever the product cannot. Only cutoff intervals
    holding no root yet are searched, so a root the product found is never
    replaced and the alignment check that follows still sees every root's
    radial order.

    - ``scan``: V, the squared index ratio, the azimuthal order n and the family (HE or
      EH).
    - Raises :class:`ModeError` if the factor changes sign more than once in an empty
      cutoff interval: one guided root per interval rules that out, so which candidate
      is the root cannot be decided.
    """
    v, ratio, n, kind = scan
    bounds = np.asarray(limits)
    occupied = np.searchsorted(bounds, roots, side="left") - 1
    empty = np.setdiff1d(np.arange(bounds.size - 1), occupied)
    if not empty.size:
        # Too many roots rather than too few: the alignment check refuses them.
        return roots
    # The factor is evaluated only on the scan intervals that meet an empty
    # cutoff interval, which is all a lost root can be in.
    first = np.maximum(np.searchsorted(nodes, bounds[empty], side="right") - 1, 0)
    last = np.minimum(
        np.searchsorted(nodes, bounds[empty + 1], side="left"), nodes.size - 1
    )
    intervals = np.unique(
        np.concatenate(
            [np.arange(low, high) for low, high in zip(first, last, strict=True)]
        )
    ).astype(int)
    factor = np.ones_like(nodes)
    touched = np.union1d(intervals, intervals + 1)
    factor[touched] = _branch(nodes[touched], v, ratio, n, kind)
    changes = intervals[
        np.signbit(factor[intervals]) != np.signbit(factor[intervals + 1])
    ]
    candidates: dict[int, list[float]] = {}
    for index in changes:
        root = brentq(
            lambda one: float(_branch(one, v, ratio, n, kind)),
            float(nodes[index]),
            float(nodes[index + 1]),
            xtol=4 * math.ulp(float(nodes[index + 1])),
            rtol=4 * _EPS,
        )
        order = int(np.searchsorted(bounds, root, side="left")) - 1
        if order in empty:
            candidates.setdefault(order, []).append(root)
    for order, found in candidates.items():
        if len(found) > 1:
            raise ModeError(
                f"the {kind} factor changes sign {len(found)} times at V={v:.16g} "
                f"between the cutoffs {limits[order]:.16g} and "
                f"{limits[order + 1]:.16g}, which hold one guided root: the "
                f"radial order of a root cannot be assigned (candidates {found})"
            )
    return sorted([*roots, *(found[0] for found in candidates.values())])


def _scan_nodes(
    v: float, n: int, low: float, high: float
) -> Float[np.ndarray, " n_nodes"]:
    """Return the scan nodes that bracket every root between two values of u.

    The nodes are the part of one fixed grid over (0, V), of spacing at most 0.1,
    that covers [low, high], with the Bessel zeros of orders n-1, n and n+1 in that
    range added. Each node is computed exactly as the whole grid would compute
    it (``numpy.linspace`` is ``start + i step``, its last node set to the end),
    so a bracket, and the root refined from it, does not depend on how much of
    the grid is taken.

    - ``low``: smallest u to cover, dimensionless, at least zero.
    - ``high``: largest u to cover, dimensionless, at most V.
    """
    # Endpoints exclude the trivial factors at u=0 and w=0, which are not
    # guided modes.
    edge = 32 * _EPS
    start, stop = edge * v, (1.0 - edge) * v
    count = max(64, math.ceil(v / 0.1) + 1)
    step = (stop - start) / (count - 1)
    first = max(0, math.floor((low - start) / step) - 1)
    last = min(count - 1, math.ceil((high - start) / step) + 1)
    base = np.arange(first, last + 1) * step + start
    if last == count - 1:
        base[-1] = stop
    zeros = _bessel_nodes(n, max(1, math.ceil(base[-1] / math.pi) + 1))
    interior = zeros[(zeros > base[0]) & (zeros < base[-1])]
    return np.unique(np.concatenate((base, interior)))


def _roots(
    v: float, ratio: float, n: int, kind: _Kind, orders: int | None = None
) -> list[float]:
    """Bracket the pole-free equation and select one vector family.

    The scan is a grid of spacing at most 0.1 with the Bessel zeros
    added, brackets at every sign change. It covers u from the family's first
    cutoff, below which no guided root lies, to the cutoff above the highest
    radial order wanted (or V). Starting there is also what keeps ``J_n``
    representable: ``J_n(u) ~ (u/2)**n/n!`` underflows to zero at small u for
    large n, and the cutoffs all exceed ``n - 2``. The scan is then held to the
    family's exact cutoffs, because two roots can share one interval:
    ``J_(-1) = -J_1`` makes HE_(1,m+1) and EH_(1,m) emerge from the same cutoff
    ``j_(1,m)``, both as roots of the same n = 1 product, and their two sign
    changes then cancel. The same happens away from any shared cutoff from V of
    about 26, where a root of each family can be closer than the scan spacing
    (measured with a numerical aperture of 0.22: 0.024 apart at V = 67,
    1.3e-3 at V = 1663 and 4.8e-4 at V = 4990, nearest V). While a hybrid
    family's count is short or misaligned, each cutoff interval holding none of
    its roots is searched with the family's own factor of the product
    (:func:`_recovered`), which has no partner root to cancel against, and every
    root so found is kept.

    What is then checked is alignment, which is stronger than the count: the
    i-th root must lie between the i-th and (i+1)-th cutoff, so every returned
    root carries its true radial order. Misalignment is raised rather than
    returned. A count still short with the alignment intact can only be missing
    roots above the last one found, whose ``u`` is within float64 rounding of V
    at its own cutoff; the caller reports that as an unresolvable mode.

    - ``v``: positive dimensionless fibre frequency V, at most :data:`_MAX_V`.
    - ``ratio``: squared real cladding/core index ratio, between zero and one.
    - ``orders``: positive number of lowest radial orders wanted; every guided root when
      omitted. The cost grows with the orders scanned, not with V.
    - Returns: guided roots u in increasing order, hence descending effective index: the
      lowest `orders` of them, or fewer where fewer are guided.
    """
    _check_frequency(v)
    cutoffs = _cutoffs(v, ratio, n, kind, None if orders is None else orders + 1)
    if not cutoffs:
        return []
    if orders is not None and len(cutoffs) > orders:
        expected, top = cutoffs[:orders], cutoffs[orders]
    else:
        expected, top = cutoffs, v
    nodes = _scan_nodes(v, n, expected[0], top)
    values = _characteristic(
        nodes, np.full_like(nodes, v), np.full_like(nodes, ratio), n, kind
    )
    # Roots above the top cutoff are of radial orders not asked for.
    roots = [
        root
        for root in _refined_roots(nodes, values, v, ratio, n, kind)[kind]
        if root <= top
    ]
    limits = [*expected, top]

    def aligned(found: list[float]) -> bool:
        """Say whether every root lies between its own cutoff and the next.

        - Returns: True when the i-th root is in the i-th cutoff interval for every i.
        """
        return len(found) <= len(expected) and all(
            limits[order] < root <= limits[order + 1]
            for order, root in enumerate(found)
        )

    if kind in ("HE", "EH") and (len(roots) < len(expected) or not aligned(roots)):
        # Only now is the second factor worth its cost: the scan has lost at
        # least one pair whose two sign changes cancelled inside one interval.
        roots = _recovered(nodes, roots, limits, (v, ratio, n, kind))
    if not aligned(roots):
        raise ModeError(
            f"the {kind} scan found {len(roots)} roots at V={v:.16g} that do not "
            f"align with the {len(expected)} exact cutoffs below V: the radial "
            f"order of a root cannot be assigned (roots {roots}, cutoffs "
            f"{expected})"
        )
    if len(roots) < len(expected):
        if top < v:
            raise ModeError(
                f"the {kind} scan found {len(roots)} of the {len(expected)} roots "
                f"below the cutoff {top:.16g} at V={v:.16g}: a root away from V "
                "could not be resolved"
            )
        # The missing roots are above the last one found, at their own cutoffs,
        # where V - u is below float64 resolution. The caller refuses those m.
        logger.debug(
            "the %s scan found %d of %d roots at V=%.16g; the highest are within "
            "rounding of V",
            kind,
            len(roots),
            len(expected),
            v,
        )
    return roots


@dataclass(frozen=True)
class _Profile:
    """Hold the lossless eigenpair fixing a transverse profile at one radius."""

    radius: float
    u: float
    w: float
    ratio: float
    f2: float


def _cutoff(kind: _Kind, n: int, m: int, ratio: float) -> float:
    """Return the exact cutoff at a fixed squared material-index ratio.

    - ``m``: positive radial order, counted from the highest effective index.
    - ``ratio``: squared real cladding/core index ratio, between zero and one.
    """
    if kind in ("TE", "TM"):
        return float(_zeros(0, m)[-1])
    if kind == "EH":
        return float(_zeros(n, m)[-1])
    if n == 1:
        return 0.0 if m == 1 else float(_zeros(1, m - 1)[-1])
    contrast = (1 - ratio) / (1 + ratio)
    lower, upper = float(_zeros(n - 2, m)[-1]), float(_zeros(n - 1, m)[-1])
    # J_(n-2) + contrast*J_n = 0 follows from the w -> 0 limit.
    # Only at zero contrast is the cutoff a zero of J_(n-2).
    return brentq(
        lambda v: float(jv(n - 2, v) + contrast * jv(n, v)),
        lower,
        upper,
        xtol=4 * math.ulp(upper),
        rtol=4 * _EPS,
    )


@dataclass(frozen=True, eq=False, kw_only=True)
class _ApertureIndex(GlassIndex):
    """Implement the numerical-aperture core index and its material slope."""

    cladding: GlassIndex | float
    numerical_aperture: float

    def _index_and_derivative(
        self, omega: Float[np.ndarray, "*shape"]
    ) -> tuple[Complex[np.ndarray, "*shape"], Complex[np.ndarray, "*shape"]]:
        """Return the index and its frequency derivative.

        - ``omega``: positive angular frequencies in rad/s.
        - Returns: complex refractive index and slope in s/rad, with the input shape.
        """
        index, derivative = _material(self.cladding, omega)
        addition = self.numerical_aperture**2 / (2 * index)
        return np.asarray(index + addition, dtype=complex), np.asarray(
            derivative * (1 - addition / index), dtype=complex
        )


def _material(
    material: GlassIndex | float, omega: Float[np.ndarray, "*shape"]
) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
    """Sample the real index and its analytic slope for the lossless eigenproblem.

    An optmat index is taken through its transparent approximation
    (:func:`~solprop.physdata.glasses.lossless_index`), which records the ``k`` it
    drops and warns beyond its bound; any other glass's real part is taken as
    documented on :class:`StepIndexMode`.

    - ``material``: index model or positive constant refractive index.
    - ``omega``: positive angular frequencies in rad/s.
    - Returns: real index and derivative in s/rad, with the input shape.
    """
    if isinstance(material, GlassIndex):
        lossless = lossless_index(material)
        return np.asarray(np.real(lossless.index(omega))), np.asarray(
            np.real(lossless.dindex_domega(omega))
        )
    return np.full_like(omega, material), np.zeros_like(omega)


def _checked_omega(omega: ArrayLike | float) -> Float[np.ndarray, " n_omega"]:
    """Require a nonempty vector of positive finite real angular frequencies.

    - ``omega``: positive angular frequencies in rad/s.
    """
    if np.iscomplexobj(omega):
        raise ModeError("omega must be real")
    values = np.atleast_1d(np.asarray(omega, dtype=float))
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values <= 0)
    ):
        raise ModeError(
            "omega must be a nonempty vector of positive finite frequencies"
        )
    return values


@dataclass(frozen=True, eq=False)
class StepIndexMode(Displayed):
    """Describe an exact vector mode with an explicitly frozen profile wavelength.

    Solve the lossless Snyder and Love eigenproblem using the real parts of the
    material indices. Material absorption is not inferred; ``loss`` supplies the
    power attenuation independently. An optmat index
    (:class:`~solprop.physdata.glasses.OptmatGlass`) is held as its transparent
    approximation (:func:`~solprop.physdata.glasses.lossless_index`): the real part,
    with the dropped ``k`` recorded for a run's manifest and a
    :class:`~solprop.exceptions.MaterialRangeWarning` once per run where the dropped
    absorption exceeds its bound (SEP 3, Q3). TE obeys ``A+B=0`` and TM obeys
    ``A+(n_cl/n_co)**2*B=0``, where ``A=J_0'/(u J_0)`` and
    ``B=K_0'/(w K_0)``. The public frequency methods require positive frequencies.

    Attributes
    ----------
    radius : float or ZProfile
        Core radius in m, positive throughout the declared path.
    core : GlassIndex or float or None
        Core refractive index; None selects fused silica. An optmat index given here
        is held as its transparent counterpart.
    cladding : GlassIndex or float or None
        Cladding refractive index; None selects the constant index 1. An optmat
        index given here is held as its transparent counterpart.
    kind : {'HE', 'EH', 'TE', 'TM'}
        Vector mode family in Snyder and Love's convention.
    n : int
        Azimuthal order, at least 1 for hybrids and 0 for TE/TM.
    m : int
        Radial order, at least 1, counted from highest effective index.
    parity : {'even', 'odd'}
        Hybrid orientation; TE/TM have no distinct parity partner.
    profile_wavelength : float
        Required wavelength in m for the transverse profile, inside both material
        validity ranges and above the selected mode's cutoff in frequency.
    loss : float or callable
        Non-negative power attenuation in 1/m, constant or alpha(omega).
        A callable may supply its analytic slope as a ``derivative`` attribute.
    core_radius : ZProfile
        Resolved radius path in m.

    References
    ----------
    A. W. Snyder and J. D. Love, Optical Waveguide Theory (1983), chapter 12,
    Table 12-3. The book's equation and table numbering has not been independently
    checked; the characteristic equation, the vector boundary conditions and the exact
    cutoffs were checked against F. Le Kien et al., Phys. Rev. A 97, 013821 (2018),
    Appendix C.

    Examples
    --------
    >>> mode = StepIndexMode(1.25e-6, profile_wavelength=835e-9)
    >>> mode.z_constant, mode.scale_invariant
    (True, False)
    """

    radius: float | ZProfile
    _: KW_ONLY
    core: GlassIndex | float | None = None
    cladding: GlassIndex | float | None = None
    kind: _Kind = "HE"
    n: int = 1
    m: int = 1
    parity: Literal["even", "odd"] = "even"
    profile_wavelength: float
    loss: float | _Loss = 0.0
    core_radius: ZProfile = field(init=False, repr=False)
    _core: GlassIndex | float = field(init=False, repr=False)
    _cladding: GlassIndex | float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate inputs and resolve the required profile eigenpair.

        Raises
        ------
        ModeError
            If geometry, indices, family, parity, loss or profile are invalid.
        """
        if self.kind not in ("HE", "EH", "TE", "TM"):
            raise ModeError("kind must be HE, EH, TE or TM")
        if (self.kind in ("TE", "TM") and self.n != 0) or (
            self.kind in ("HE", "EH") and self.n < 1
        ):
            raise ModeError("TE/TM require n=0; HE/EH require n>=1")
        if self.m < 1:
            raise ModeError("radial order m must be at least 1")
        if self.parity not in ("even", "odd"):
            raise ModeError("parity must be even or odd")
        if not math.isfinite(self.profile_wavelength) or self.profile_wavelength <= 0:
            raise ModeError("profile_wavelength must be positive and finite")
        if not callable(self.loss) and (not math.isfinite(self.loss) or self.loss < 0):
            raise ModeError("loss must be a non-negative finite coefficient in 1/m")
        object.__setattr__(self, "core_radius", as_profile(self.radius))
        # The eigenproblem is lossless: an optmat index is held as its transparent
        # counterpart, in the declared field too, so that the mode carries -- and a
        # run's manifest records -- the material it is solved with (SEP 3, Q3).
        for name in ("core", "cladding"):
            declared = getattr(self, name)
            if declared is not None:
                object.__setattr__(self, name, lossless_index(declared))
        object.__setattr__(
            self,
            "_core",
            lossless_index(fused_silica()) if self.core is None else self.core,
        )
        object.__setattr__(
            self, "_cladding", 1.0 if self.cladding is None else self.cladding
        )
        self._check_materials()
        for position in np.unique(
            np.concatenate(
                (
                    self.core_radius.breakpoints,
                    np.asarray(self.core_radius.z_span),
                    [0.0],
                )
            )
        ):
            if math.isfinite(float(position)):
                self._radius(float(position))
        self._profile_at(0.0)
        if callable(self.loss) and not callable(getattr(self.loss, "derivative", None)):
            logger.warning(
                "the loss callable has no derivative; only its contribution to "
                "dneff_domega is Richardson differenced"
            )

    def _check_materials(self) -> None:
        """Check material indices and profile wavelength support."""
        for material in (self._core, self._cladding):
            if isinstance(material, GlassIndex):
                low, high = material.wavelength_range
                if not low <= self.profile_wavelength <= high:
                    raise ModeError(
                        f"profile_wavelength is outside {material.name}'s validity "
                        f"range {material.wavelength_range}"
                    )
            elif not math.isfinite(material) or material <= 0:
                raise ModeError("material indices must be positive and finite")

    @classmethod
    def from_numerical_aperture(
        cls,
        radius: float | ZProfile,
        numerical_aperture: float,
        *,
        cladding: GlassIndex | float | None = None,
        kind: _Kind = "HE",
        n: int = 1,
        m: int = 1,
        parity: Literal["even", "odd"] = "even",
        profile_wavelength: float,
        loss: float | _Loss = 0.0,
    ) -> "StepIndexMode":
        """Construct a fibre with the core index n_co = n_cl + NA^2/(2 n_cl).

        Parameters
        ----------
        radius : float or ZProfile
            Positive core radius in m.
        numerical_aperture : float
            Positive finite numerical aperture, dimensionless.
        cladding : GlassIndex or float or None, optional
            Cladding index; None selects fused silica.
        kind : {'HE', 'EH', 'TE', 'TM'}, optional
            Vector mode family.
        n, m : int, optional
            Azimuthal and radial indices, as in the class attributes.
        parity : {'even', 'odd'}, optional
            Hybrid orientation.
        profile_wavelength : float
            Required profile wavelength in m.
        loss : float or callable, optional
            Power attenuation in 1/m.

        Returns
        -------
        StepIndexMode
            Fibre with an analytic material slope for its derived core index.

        Raises
        ------
        ModeError
            If the aperture or mode arguments are invalid.
        """
        if not math.isfinite(numerical_aperture) or numerical_aperture <= 0:
            raise ModeError("numerical_aperture must be positive and finite")
        resolved = lossless_index(fused_silica() if cladding is None else cladding)
        core = _ApertureIndex(
            name="NA core",
            source="numerical_aperture",
            wavelength_range=resolved.wavelength_range
            if isinstance(resolved, GlassIndex)
            else (0.0, math.inf),
            citation="n_co = n_cl + NA^2 / (2 n_cl) from the numerical aperture",
            cladding=resolved,
            numerical_aperture=numerical_aperture,
        )
        return cls(
            radius,
            core=core,
            cladding=resolved,
            kind=kind,
            n=n,
            m=m,
            parity=parity,
            profile_wavelength=profile_wavelength,
            loss=loss,
        )

    def _radius(self, z: float) -> float:
        """Evaluate and validate the radius at every requested position.

        - ``z``: position along the fibre in m.
        - Returns: positive core radius in m.
        """
        radius = float(self.core_radius(np.asarray([z]))[0])
        if not math.isfinite(radius) or radius <= 0:
            raise ModeError("the core radius must be positive and finite")
        return radius

    def _eigenpairs(
        self, omega: Float[np.ndarray, " n_omega"], z: float
    ) -> tuple[Float[np.ndarray, " n_omega"], Float[np.ndarray, " n_omega"]]:
        """Select the m-th root of the requested vector family at each frequency.

        - ``omega``: positive angular frequencies in rad/s.
        - ``z``: position along the fibre in m.
        - Returns: effective indices and their transverse roots u, shape (n_omega,).
        """
        core, _ = _material(self._core, omega)
        cladding, _ = _material(self._cladding, omega)
        if (
            not np.all(np.isfinite(core))
            or not np.all(np.isfinite(cladding))
            or np.any(cladding <= 0)
            or np.any(core <= cladding)
        ):
            raise ModeError(
                "core index must exceed the positive cladding index "
                "everywhere on the band"
            )
        scale = self._radius(z) * omega / C
        v = scale * np.sqrt((core - cladding) * (core + cladding))
        ratio = (cladding / core) ** 2
        # One check for the whole band, before any frequency is solved.
        _check_frequency(float(np.max(v)))
        result = np.empty_like(omega)
        transverse = np.empty_like(omega)
        for index in range(omega.size):
            roots = _roots(
                float(v[index]), float(ratio[index]), self.n, self.kind, self.m
            )
            if len(roots) < self.m:
                raise ModeError(
                    self._missing_root(float(v[index]), float(ratio[index]))
                )
            u = float(roots[self.m - 1])
            decay = math.sqrt((float(v[index]) - u) * (float(v[index]) + u))
            # Both branches are the same eigenvalue; near cutoff, where w is the
            # small parameter, n_eff - n_cl is what the guidance guard and the
            # cladding profile need and the w branch conditions it far better.
            value = (
                math.sqrt(float(cladding[index] ** 2 + (decay / scale[index]) ** 2))
                if decay < u
                else math.sqrt(float(core[index] ** 2 - (u / scale[index]) ** 2))
            )
            if not cladding[index] < value < core[index]:
                raise ModeError(
                    f"{self.kind}{self.n}{self.m} is too close to cutoff to resolve "
                    f"a guided index in float64 at V={v[index]:.16g}"
                )
            result[index] = value
            transverse[index] = roots[self.m - 1]
        return result, transverse

    def _missing_root(self, v: float, ratio: float) -> str:
        """Return the diagnostic for a radial root the scan did not find.

        - ``ratio``: squared real cladding/core index ratio, between zero and one.
        - Returns: a below-cutoff message when V is at or below the exact cutoff;
          otherwise the root exists but lies within float64 rounding of V (see
          :func:`_roots`), and the message says so rather than calling the mode below
          cutoff.
        """
        cutoff = _cutoff(self.kind, self.n, self.m, ratio)
        name = f"{self.kind}{self.n}{self.m}"
        if v > cutoff:
            return (
                f"{name} is above cutoff at V={v:.16g} (cutoff V={cutoff:.16g}), "
                f"but its radial root m={self.m} is not resolvable in float64 this "
                "close to cutoff"
            )
        return (
            f"{name} is below cutoff at V={v:.16g}: no radial root m={self.m}; "
            f"cutoff V={cutoff:.16g}"
        )

    def _loss_values(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> tuple[Float[np.ndarray, " n_omega"], Float[np.ndarray, " n_omega"]]:
        """Return absolute power attenuation and its slope.

        - ``omega``: positive angular frequencies in rad/s.
        - Returns: attenuation in 1/m and its derivative in s/(rad m).
        """
        if not callable(self.loss):
            return np.full_like(omega, self.loss), np.zeros_like(omega)
        attenuation = np.asarray(self.loss(omega), dtype=float)
        if (
            attenuation.shape != omega.shape
            or not np.all(np.isfinite(attenuation))
            or np.any(attenuation < 0)
        ):
            raise ModeError(
                "loss callable must return finite non-negative coefficients "
                "with the shape of omega"
            )
        analytic = getattr(self.loss, "derivative", None)
        if callable(analytic):
            slope = np.asarray(analytic(omega), dtype=float)
        else:
            # Same fourth-order difference and step as the existing absolute-loss
            # path; the lossless eigenvalue derivative is always analytic.
            step = omega * 1e-5
            wide = (self.loss(omega + step) - self.loss(omega - step)) / (2 * step)
            narrow = (self.loss(omega + step / 2) - self.loss(omega - step / 2)) / step
            slope = (4 * narrow - wide) / 3
        if slope.shape != omega.shape or not np.all(np.isfinite(slope)):
            raise ModeError("loss derivative must be finite with the shape of omega")
        return attenuation, slope

    def neff(
        self, omega: ArrayLike | float, z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the effective index, with positive imaginary part for loss.

        Parameters
        ----------
        omega : array_like or float
            Positive finite angular frequencies in rad/s, shape (n_omega,).
        z : float, optional
            Position along the fibre in m.

        Returns
        -------
        numpy.ndarray
            Complex dimensionless index, shape (n_omega,).

        Raises
        ------
        ModeError
            If inputs are invalid or any frequency is below cutoff.
        """
        values = _checked_omega(omega)
        attenuation, _ = self._loss_values(values)
        return np.asarray(
            self._eigenpairs(values, z)[0] + 0.5j * C * attenuation / values
        )

    def neff_minus_one(
        self, omega: ArrayLike | float, z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return neff minus one by subtraction, appropriate for a solid core.

        Unlike a dilute gas, a solid core's refractivity is ordinarily of order
        unity and does not require the capillary's cancellation-free expression.

        Parameters
        ----------
        omega : array_like or float
            Positive finite angular frequencies in rad/s, shape (n_omega,).
        z : float, optional
            Position along the fibre in m.

        Returns
        -------
        numpy.ndarray
            Complex refractivity, shape (n_omega,).
        """
        return self.neff(omega, z) - 1.0

    def dneff_domega(
        self, omega: ArrayLike | float, z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Differentiate the exact eigenvalue equation analytically.

        Only an absolute-loss callable without a derivative attribute is differenced.

        Parameters
        ----------
        omega : array_like or float
            Positive finite angular frequencies in rad/s, shape (n_omega,).
        z : float, optional
            Position along the fibre in m.

        Returns
        -------
        numpy.ndarray
            Complex index derivative in s/rad, shape (n_omega,).

        Raises
        ------
        ModeError
            If the eigenpair or its derivative cannot be resolved.
        """
        values = _checked_omega(omega)
        index, u = self._eigenpairs(values, z)
        core, dcore = _material(self._core, values)
        cladding, dcladding = _material(self._cladding, values)
        square = (self._radius(z) * values / C) ** 2
        w = np.sqrt(square * (core - cladding) * (core + cladding) - u**2)
        ratio = (cladding / core) ** 2
        slope = np.empty_like(values)
        for row in range(values.size):
            du, dw, dratio = _partials(
                float(u[row]), float(w[row]), float(ratio[row]), self.n, self.kind
            )
            partial_index = square[row] * index[row] * (-du / u[row] + dw / w[row])
            partial_frequency = du * (
                u[row] / values[row] + square[row] * core[row] * dcore[row] / u[row]
            )
            partial_frequency += dw * (
                w[row] / values[row]
                - square[row] * cladding[row] * dcladding[row] / w[row]
            )
            partial_frequency += (
                dratio
                * 2
                * ratio[row]
                * (dcladding[row] / cladding[row] - dcore[row] / core[row])
            )
            if partial_index == 0 or not math.isfinite(float(partial_index)):
                raise ModeError("eigenvalue derivative is singular")
            slope[row] = -partial_frequency / partial_index
        attenuation, dattenuation = self._loss_values(values)
        result = np.asarray(
            slope + 0.5j * C * (dattenuation * values - attenuation) / values**2
        )
        if not np.all(np.isfinite(result)):
            raise ModeError("eigenvalue derivative must be finite")
        return result

    @property
    def z_constant(self) -> bool:  # numpydoc ignore=RT01
        """Whether the radius path is constant."""
        return self.core_radius.is_constant

    @property
    def radius_is_constant(self) -> bool:  # numpydoc ignore=RT01
        """Whether the transverse geometry is constant along z."""
        return self.core_radius.is_constant

    @property
    def scale_invariant(self) -> bool:  # numpydoc ignore=RT01
        """Whether the profile scales exactly with the core radius.

        Always ``False``: changing the radius changes the normalised frequency V.
        """
        return False

    @property
    def channel_label(self) -> ChannelLabel:  # numpydoc ignore=RT01
        """Exact vector family, its indices and its parity.

        Family, azimuthal and radial order, and the hybrid parity. An axially
        symmetric TE0m or TM0m channel has no parity partner, so its parity is
        ``None`` rather than the orientation its constructor defaulted to.

        Examples
        --------
        >>> from solprop.modes import StepIndexMode
        >>> mode = StepIndexMode(1.25e-6, profile_wavelength=835e-9)
        >>> mode.channel_label.display_label
        'HE1,1:even'
        """
        parity = self.parity if self.n else None
        return ChannelLabel(
            self.kind,
            self.n,
            self.m,
            parity,
            None,
            f"{self.kind}{self.n},{self.m}" + (f":{parity}" if parity else ""),
        )

    @property
    def azimuthal_order(self) -> int:  # numpydoc ignore=RT01
        """Azimuthal mode index n."""
        return self.n

    def principal_component(self, z: float = 0.0) -> None:
        """Declare that the exact vector profile needs both Cartesian components.

        Parameters
        ----------
        z : float, optional
            Position along the fibre in m; orientation is independent of it.

        Returns
        -------
        None
            Both transverse components are resolved.
        """
        del z

    @property
    def orthogonal_polarisation(self) -> "StepIndexMode | None":  # numpydoc ignore=RT01
        """Other hybrid parity, or None for TE and TM."""
        if self.n == 0:
            return None
        return replace(self, parity="odd" if self.parity == "even" else "even")

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:  # numpydoc ignore=RT01
        """The core-radius path, m, carrying the entire z dependence."""
        return MappingProxyType({"core_radius": self.core_radius})

    def at_parameters(self, values: Mapping[str, float]) -> "StepIndexMode":
        """Freeze the supplied geometry parameter.

        Parameters
        ----------
        values : collections.abc.Mapping
            Any subset of core_radius, in m and strictly positive.

        Returns
        -------
        StepIndexMode
            Mode with the named radius frozen and its profile recomputed.

        Raises
        ------
        ModeError
            If a name is unknown or the radius is invalid.
        """
        if set(values) - {"core_radius"}:
            raise ModeError(f"unknown mode parameters: {set(values) - {'core_radius'}}")
        return replace(self, radius=values.get("core_radius", self.radius))

    def _solve_profile(self, z: float) -> _Profile:
        """Evaluate the transverse-profile parameters at the declared wavelength.

        - ``z``: position along the fibre in m.
        """
        omega = np.asarray([2 * math.pi * C / self.profile_wavelength])
        _, transverse = self._eigenpairs(omega, z)
        core = float(_material(self._core, omega)[0][0])
        cladding = float(_material(self._cladding, omega)[0][0])
        radius = self._radius(z)
        scale = radius * float(omega[0]) / C
        u = float(transverse[0])
        w = math.sqrt(scale**2 * (core - cladding) * (core + cladding) - u**2)
        if w < PROFILE_MIN_W:
            raise ModeError(
                f"{self.kind}{self.n}{self.m} has cladding decay w={w:.6g} at the "
                f"profile wavelength {self.profile_wavelength * 1e9:.6g} nm, below "
                f"the floor {PROFILE_MIN_W}: the field's 1e-8 tail would put the "
                "outer radius beyond 185 core radii, which no transverse "
                "integral resolves. Freeze the profile nearer the guided band."
            )
        a = float(jvp(self.n, u) / (u * jv(self.n, u)))
        b = float(_log_slope(self.n, w)) / w
        f2 = self.n * (1 / u**2 + 1 / w**2) / (a + b) if self.n else 0.0
        return _Profile(radius, u, w, (cladding / core) ** 2, f2)

    @cached_property
    def _entrance_profile(self) -> _Profile:
        """Keep the required entrance profile on this immutable mode."""
        return self._solve_profile(0.0)

    def _profile_at(self, z: float) -> _Profile:
        """Reuse a constant geometry's profile and recompute a tapered one.

        - ``z``: position along the fibre in m.
        """
        return (
            self._entrance_profile
            if self.radius_is_constant or z == 0.0
            else self._solve_profile(z)
        )

    def radial_breaks(self, z: float = 0.0) -> tuple[float, ...]:
        """Return the core boundary where the radial electric field jumps.

        Parameters
        ----------
        z : float, optional
            Position along the fibre in m.

        Returns
        -------
        tuple of float
            One radius in m.
        """
        return (self._radius(z),)

    def cutoff_parameter(self) -> float:
        """Return the exact cutoff V for the profile wavelength's material indices.

        For dispersive materials the cutoff curve changes with frequency. This
        diagnostic uses the declared profile wavelength; the solver determines
        guidance using the material indices at each requested frequency.

        Returns
        -------
        float
            Dimensionless cutoff V. HE11 has cutoff zero. Hybrid orders n>=2
            use the exact contrast-dependent equation, not its weak-guidance limit.

        References
        ----------
        F. Le Kien et al., Phys. Rev. A 97, 013821 (2018), Appendix C,
        doi:10.1103/PhysRevA.97.013821.
        """
        return _cutoff(self.kind, self.n, self.m, self._entrance_profile.ratio)

    def _materials(self) -> tuple[str, str]:
        """Return the resolved core and cladding as text, for printing.

        - Returns: a glass by its name and source, anything else by its ``repr``.
        """
        names = []
        for model in (self._core, self._cladding):
            name = getattr(model, "name", None)
            source = getattr(model, "source", "")
            names.append(f"{name} ({source})" if name else repr(model))
        return names[0], names[1]

    def __repr__(self) -> str:
        """Return a one-line description: the mode, the radius and the materials.

        Returns
        -------
        str
            For example ``StepIndexMode(HE11 even, a=1.25 um, profile at 835 nm)``.
        """
        radius = float(self.core_radius(np.array([0.0]))[0])
        label = f"{self.kind}{self.n}{self.m}"
        if self.kind in ("HE", "EH"):
            label += f" {self.parity}"
        return line(
            "StepIndexMode",
            [
                ("", label),
                ("a", engineering(radius, "m")),
                ("", f"profile at {engineering(self.profile_wavelength, 'm')}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        radius = float(self.core_radius(np.array([0.0]))[0])
        core, cladding = self._materials()
        return f"StepIndexMode {self.kind}{self.n}{self.m}", [
            ("core radius", engineering(radius, "m")),
            ("core", core),
            ("cladding", cladding),
            ("parity", self.parity),
            ("profile frozen at", engineering(self.profile_wavelength, "m")),
            ("loss", self.loss if isinstance(self.loss, (int, float)) else "callable"),
        ]

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, float], tuple[float, float]]:
        """Return a polar disc enclosing the cladding tail to the declared decay.

        Parameters
        ----------
        z : float, optional
            Position along the fibre in m.

        Returns
        -------
        tuple
            Polar limits in m and rad, with R_out=max(2a,a*(1+log(1e8)/w)).
        """
        profile = self._profile_at(z)
        outer = profile.radius * max(2.0, 1.0 + math.log(1e8) / profile.w)
        return "polar", (0.0, 0.0), (outer, 2 * math.pi)

    def _radial_fields(
        self, radius: Float[np.ndarray, "*shape"], profile: _Profile
    ) -> tuple[Float[np.ndarray, "*shape"], Float[np.ndarray, "*shape"]]:
        """Evaluate radial coefficients before the parity factors are applied.

        - ``radius``: dimensionless non-negative radii r/a.
        - Returns: unnormalised radial and tangential coefficients, with the radius
          shape.
        """
        inside = radius <= 1.0
        radial, tangential = np.zeros_like(radius), np.zeros_like(radius)
        if self.n == 0:
            # The TE/TM limiting solutions are separate: Table 12-3's hybrid
            # coefficients divide by n. Tangential E (TE) and radial D (TM)
            # match at the boundary; the arbitrary core boundary amplitude is 1.
            amplitude = np.empty_like(radius)
            amplitude[inside] = jv(1, profile.u * radius[inside]) / jv(1, profile.u)
            # K_1(w r)/K_1(w) from scaled functions: exp(-w (r - 1)) carries the
            # decay that would otherwise underflow both kv values at large w.
            amplitude[~inside] = _k_decay(1, profile.w, radius[~inside])
            if self.kind == "TE":
                tangential = amplitude
            else:
                radial = np.where(inside, amplitude, amplitude / profile.ratio)
            return radial, tangential
        first, second = (profile.f2 - 1) / 2, (profile.f2 + 1) / 2
        lower = jv(self.n - 1, profile.u * radius[inside])
        upper = jv(self.n + 1, profile.u * radius[inside])
        radial[inside] = -(first * lower + second * upper) / jv(self.n, profile.u)
        tangential[inside] = -(first * lower - second * upper) / jv(self.n, profile.u)
        # The exterior coefficients are K_(n -+ 1)(w r)/K_n(w), each taken as
        # K_(n -+ 1)(w r)/K_(n -+ 1)(w) times a one-argument quotient at w, so
        # both stay finite where kv underflows (large w) and where kve
        # overflows (large n at small w).
        outside = radius[~inside]
        quotient = profile.w * float(_k_ratio(self.n, profile.w))
        lower = _k_decay(self.n - 1, profile.w, outside) * quotient
        upper = _k_decay(self.n + 1, profile.w, outside) * (
            quotient + 2 * self.n / profile.w
        )
        factor = -profile.u / profile.w
        radial[~inside] = factor * (first * lower - second * upper)
        tangential[~inside] = factor * (first * lower + second * upper)
        return radial, tangential

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Return unnormalised Cartesian fields with the core value at r=a.

        Parameters
        ----------
        points : numpy.ndarray
            Polar (r, theta) points in m and rad, shape (n_points, 2), r>=0.
        z : float, optional
            Position along the fibre in m.

        Returns
        -------
        numpy.ndarray
            Real Cartesian components (E_x, E_y), shape (n_points, 2).

        Raises
        ------
        ModeError
            If points are non-finite, malformed or have negative radii.
        """
        points = np.asarray(points, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
            or np.any(points[:, 0] < 0)
        ):
            raise ModeError("points must be finite (r, theta) pairs with r>=0")
        profile = self._profile_at(z)
        radial, tangential = self._radial_fields(points[:, 0] / profile.radius, profile)
        theta = points[:, 1]
        if self.n:
            if self.parity == "even":
                radial = radial * np.cos(self.n * theta)
                tangential = -tangential * np.sin(self.n * theta)
            else:
                radial = radial * np.sin(self.n * theta)
                tangential = tangential * np.cos(self.n * theta)
        return np.column_stack(
            (
                radial * np.cos(theta) - tangential * np.sin(theta),
                radial * np.sin(theta) + tangential * np.cos(theta),
            )
        )

    def electromagnetic_field(
        self, xy: ArrayLike, omega: float, z: float = 0.0
    ) -> tuple[Complex[np.ndarray, "n_points 3"], Complex[np.ndarray, "n_points 3"]]:
        """Return unit-power E/H in the internal FFT convention.

        Parameters
        ----------
        xy : array_like
            Cartesian (x, y) positions in m, shape (n_points, 2). Named apart from
            the polar `points` of :meth:`field`, which this method does not share:
            these are Cartesian pairs. Interface
            points take the core value; the axis is evaluated by its analytic limit.
        omega : float
            Positive angular frequency in rad/s in the bound guided band.
        z : float, optional
            Position in m within the radius path.

        Returns
        -------
        tuple of numpy.ndarray
            Complex Cartesian E and H, each shape (n_points, 3), including
            longitudinal components. Units are V/m and A/m per sqrt(W); the
            full-domain Poynting flux is one, independently of the supplied points.

        Raises
        ------
        ModeError
            If the positions or frequency are invalid, the mode is too near cutoff,
            or attenuation/leaky fields are requested without a qualified adjoint.

        Notes
        -----
        This separate vector API leaves the frozen transverse ``field`` unchanged.
        Maxwell curl equations use exp(i omega t - i beta z). The radial
        roots, family counting and analytic dispersion derivative are unchanged.
        """
        from solprop.modes import _stepindex_vector  # noqa: PLC0415 - type cycle

        omega, z = float(omega), float(z)
        # A public boundary owes its caller a ModeError, not the AttributeError a
        # list of pairs used to raise on the first `.ndim` inside the field helper.
        points = np.asarray(xy, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ModeError(
                f"xy must be Cartesian (x, y) pairs of shape (n_points, 2), got "
                f"{points.shape}"
            )
        electric, magnetic = _stepindex_vector._fields(self, points, omega, z)
        power, error = _stepindex_vector._power(self, omega, z)
        if power <= error:
            raise ModeError("full-domain power norm is non-positive or unresolved")
        return electric / np.sqrt(power), magnetic / np.sqrt(power)

    def _integrals(self, z: float) -> tuple[float, float]:
        """Integrate intensity and its square after azimuthal integration.

        - ``z``: position along the fibre in m.
        - Returns: Squared-field and quartic-field area integrals in m².
        """
        profile = self._profile_at(z)
        outer = self.dimlimits(z)[2][0] / profile.radius

        def integrand(radius: float, power: int) -> float:
            """Evaluate the radial intensity moment.

            - ``radius``: dimensionless non-negative radius r/a.
            """
            radial, tangential = self._radial_fields(np.asarray(radius), profile)
            x, y = float(radial) ** 2, float(tangential) ** 2
            if not self.n:
                angular = 2 * math.pi * (x + y) ** power
            elif power == 1:
                angular = math.pi * (x + y)
            else:
                angular = math.pi * (3 * (x**2 + y**2) + 2 * x * y) / 4
            return radius * angular

        # Dimensionless integration avoids an absolute m² tolerance overwhelming
        # a micron-scale area. The target is well above QUADPACK's 50-eps floor.
        integrals = tuple(
            quad(
                integrand,
                0.0,
                outer,
                args=(power,),
                points=[1.0],
                epsabs=0.0,
                epsrel=2e-13,
                limit=200,
            )[0]
            * profile.radius**2
            for power in (1, 2)
        )
        return integrals[0], integrals[1]

    @cached_property
    def _entrance_integrals(self) -> tuple[float, float]:
        """Cache the two profile integrals of a constant geometry.

        - Returns: Squared-field and quartic-field area integrals in m².
        """
        return self._integrals(0.0)

    def normalisation(self, z: float = 0.0) -> float:
        """Return the power normalisation of the transverse electric profile.

        Parameters
        ----------
        z : float, optional
            Position along the fibre in m.

        Returns
        -------
        float
            (epsilon0*c/2) times the transverse squared-field integral.
        """
        squared, _ = (
            self._entrance_integrals if self.radius_is_constant else self._integrals(z)
        )
        return 0.5 * EPS0 * C * squared

    def aeff(self, z: float = 0.0) -> float:
        """Return the effective area of the frozen transverse profile.

        Parameters
        ----------
        z : float, optional
            Position along the fibre in m.

        Returns
        -------
        float
            Squared intensity integral divided by the quartic integral, in m².
        """
        squared, quartic = (
            self._entrance_integrals if self.radius_is_constant else self._integrals(z)
        )
        return squared**2 / quartic
