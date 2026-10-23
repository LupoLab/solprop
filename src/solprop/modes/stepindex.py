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
from scipy.special import jn_zeros, jv, jvp, kv, kvp

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
_ROOT_REFINEMENTS = 6
"""Halvings of the suspect scan intervals allowed while a root count is short.

Six levels divide every equal-sign interval of the coarse scan into 64, so a
pair of roots further apart than about 1.6e-3 in u is separated. A closer pair
is within about 1e-3 of its shared cutoff, where the newborn root's guidance is
below what float64 resolves; the scan then reports a short count, and the
alignment check still refuses to mislabel anything.
"""


def _characteristic(
    u: Float[np.ndarray, "*shape"] | float,
    v: Float[np.ndarray, "*shape"] | float,
    ratio: Float[np.ndarray, "*shape"] | float,
    n: int,
    kind: _Kind,
) -> Float[np.ndarray, "*shape"]:
    """Evaluate the pole-free product with positive Bessel and endpoint factors removed.

    Parameters
    ----------
    u : float or numpy.ndarray
        Dimensionless core transverse frequency, strictly between zero and V.
    v : float or numpy.ndarray
        Positive dimensionless fibre frequency V, matching u in shape.
    ratio : float or numpy.ndarray
        Squared real cladding/core index ratio, between zero and one.
    n : int
        Non-negative azimuthal order.
    kind : str
        HE, EH, TE or TM family.

    Returns
    -------
    numpy.ndarray
        Pole-free characteristic value, with the broadcast input shape.
    """
    w = np.sqrt((v - u) * (v + u))
    bessel = jv(n, u)
    if kind in ("TE", "TM"):
        first = u * w**2 * jvp(n, u)
        second = u**2 * w * bessel * kvp(n, w) / kv(n, w)
        return np.asarray(first + (1.0 if kind == "TE" else ratio) * second)
    # Expand with J_n'=n J_n/u-J_(n+1), K_n'=-n K_n/w-K_(n-1).
    # Cancelling the endpoint factors symbolically prevents rounding in two
    # nearly equal products from manufacturing roots at u=0 or w=0.
    h = jv(n + 1, u) / u
    k = kv(n - 1, w) / (w * kv(n, w))
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

    Parameters
    ----------
    u : float or numpy.ndarray
        Dimensionless core transverse frequency, strictly between zero and V.
    w : float
        Positive dimensionless cladding decay parameter.
    ratio : float or numpy.ndarray
        Squared real cladding/core index ratio, between zero and one.
    n : int
        Non-negative azimuthal order.
    kind : str
        HE, EH, TE or TM family.

    Returns
    -------
    tuple of float
        Partial derivatives with respect to u, w and the squared index ratio.

    Notes
    -----
    The scaling differs from :func:`_characteristic` by ``u**2 w**2``: this
    function differentiates ``u**4 w**4 J_n**2`` times the cleared equation,
    where :func:`_characteristic` evaluates ``u**2 w**2 J_n**2`` times it. Both
    quotients are the same at a root, where the extra positive factor cancels,
    and a test ties them together at three roots.
    """
    bessel, slope, curvature = float(jv(n, u)), float(jvp(n, u)), float(jvp(n, u, 2))
    logarithmic = float(kvp(n, w) / kv(n, w))
    dlogarithmic = float(kvp(n, w, 2) / kv(n, w)) - logarithmic**2
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

    Parameters
    ----------
    u : float or numpy.ndarray
        Dimensionless core transverse frequency, strictly between zero and V.
    v : float or numpy.ndarray
        Positive dimensionless fibre frequency V, matching u in shape.
    ratio : float or numpy.ndarray
        Squared real cladding/core index ratio, between zero and one.
    n : int
        Non-negative azimuthal order.

    Returns
    -------
    str
        HE or EH according to the vanishing quadratic branch.
    """
    w = math.sqrt((v - u) * (v + u))
    a = float(jvp(n, u) / (u * jv(n, u)))
    b = float(kvp(n, w) / (w * kv(n, w)))
    centre = a + (1.0 + ratio) * b / 2.0
    radical = math.sqrt(
        ((1.0 - ratio) * b / 2.0) ** 2
        + n**2 * (1 / u**2 + 1 / w**2) * (1 / u**2 + ratio / w**2)
    )
    return "HE" if abs(centre + radical) < abs(centre - radical) else "EH"


@cache
def _zeros(order: int, count: int) -> Float[np.ndarray, " n_zeros"]:
    """Cache the first zeros of one Bessel order.

    Parameters
    ----------
    order : int
        Non-negative Bessel order.
    count : int
        Positive number of zeros.

    Returns
    -------
    numpy.ndarray
        The first `count` zeros of J_order, dimensionless.
    """
    return jn_zeros(order, count)


@cache
def _bessel_nodes(n: int, count: int) -> Float[np.ndarray, " n_zeros"]:
    """Cache frequency-independent Bessel-zero scan nodes.

    Parameters
    ----------
    n : int
        Non-negative azimuthal order.
    count : int
        Positive number of zeros per Bessel order.

    Returns
    -------
    numpy.ndarray
        Bessel zeros for orders n-1, n and n+1, dimensionless.
    """
    return np.concatenate([_zeros(abs(order), count) for order in (n - 1, n, n + 1)])


def _cutoffs(v: float, ratio: float, n: int, kind: _Kind) -> list[float]:
    """Return one family's exact cutoffs below V, in increasing order.

    A family has exactly one guided root per cutoff below V, and its m-th root
    lies between its m-th and (m+1)-th cutoff, so this list is both
    the expected root count and the interval each root must fall in.

    Parameters
    ----------
    v : float
        Positive dimensionless fibre frequency V.
    ratio : float
        Squared real cladding/core index ratio, between zero and one.
    n : int
        Non-negative azimuthal order.
    kind : str
        HE, EH, TE or TM family.

    Returns
    -------
    list of float
        Cutoffs of that family strictly below V, dimensionless.
    """
    limit = max(1, math.ceil(v / math.pi) + 1)
    if kind in ("TE", "TM", "EH"):
        zeros = _zeros(0 if kind in ("TE", "TM") else n, limit)
        return zeros[zeros < v].tolist()
    # The HE cutoffs rise with m (0 or a J_1 zero for n = 1, and inside
    # [j_(n-2,m), j_(n-1,m)] otherwise), so counting up terminates at the first
    # cutoff at or above V.
    found: list[float] = []
    while (cutoff := _cutoff("HE", n, len(found) + 1, ratio)) < v:
        found.append(cutoff)
    return found


def _refined_roots(
    nodes: Float[np.ndarray, " n_nodes"],
    values: Float[np.ndarray, " n_nodes"],
    v: float,
    ratio: float,
    n: int,
    kind: _Kind,
) -> dict[str, list[float]]:
    """Refine every sign change on a node set and label each root by family.

    Parameters
    ----------
    nodes : numpy.ndarray
        Increasing scan nodes in (0, V), dimensionless.
    values : numpy.ndarray
        The characteristic at those nodes, evaluated once by the caller.
    v : float
        Positive dimensionless fibre frequency V.
    ratio : float
        Squared real cladding/core index ratio, between zero and one.
    n : int
        Non-negative azimuthal order.
    kind : str
        HE, EH, TE or TM family; the hybrid families share one product.

    Returns
    -------
    dict
        Family name to its roots u in increasing order.
    """

    def equation(one: float) -> float:
        """Evaluate the characteristic at one trial root.

        Parameters
        ----------
        one : float
            Trial dimensionless transverse frequency u.

        Returns
        -------
        float
            Characteristic value for the scalar root search.
        """
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


def _missing_region(
    roots: list[float], expected: list[float], v: float
) -> tuple[float, float]:
    """Return the interval in u that must contain the lowest missing root.

    Every root sits between its own cutoff and the next. If the found
    roots align with the cutoffs from the bottom, the missing ones are above the
    next cutoff; where alignment first fails, a root of that order is missing
    between its cutoff and the root that took its place.

    Parameters
    ----------
    roots : list of float
        Roots found so far, in increasing order.
    expected : list of float
        The family's exact cutoffs below V, in increasing order.
    v : float
        Positive dimensionless fibre frequency V.

    Returns
    -------
    tuple of float
        Lower and upper bound in u of the region to sample more finely.
    """
    limits = [*expected, v]
    for order, root in enumerate(roots):
        if not limits[order] < root <= limits[order + 1]:
            return expected[order], root
    return expected[len(roots)], v


def _roots(v: float, ratio: float, n: int, kind: _Kind) -> list[float]:
    """Bracket the pole-free equation and select one vector family.

    The scan is a grid of spacing at most 0.1 with the Bessel zeros
    added, brackets at every sign change. It is then held to the family's exact
    cutoffs, because two roots can share one interval: ``J_(-1) = -J_1`` makes
    HE_(1,m+1) and EH_(1,m) emerge from the same cutoff ``j_(1,m)``, both as
    roots of the same n = 1 product, and their two sign changes then cancel.
    While the count is short of the cutoff count, every interval whose endpoints
    share a sign -- the only place a cancelled pair can hide -- is sampled twice
    as finely, up to :data:`_ROOT_REFINEMENTS` halvings.

    What is then checked is alignment, which is stronger than the count: the
    i-th root must lie between the i-th and (i+1)-th cutoff, so every returned
    root carries its true radial order. Misalignment is raised rather than
    returned. A count still short with the alignment intact can only be missing
    roots above the last one found, whose ``u`` is within float64 rounding of V
    at its own cutoff; the caller reports that as an unresolvable mode.

    Parameters
    ----------
    v : float or numpy.ndarray
        Positive dimensionless fibre frequency V, matching u in shape.
    ratio : float or numpy.ndarray
        Squared real cladding/core index ratio, between zero and one.
    n : int
        Non-negative azimuthal order.
    kind : str
        HE, EH, TE or TM family.

    Returns
    -------
    list of float
        Guided roots u in increasing order, hence descending effective index.

    Raises
    ------
    ModeError
        If the refined roots do not align one for one with the family's cutoffs.
    """
    # Endpoints exclude the trivial factors at u=0 and w=0, which are not
    # guided modes.
    edge = 32 * _EPS
    base = np.linspace(edge * v, (1.0 - edge) * v, max(64, math.ceil(v / 0.1) + 1))
    zeros = _bessel_nodes(n, max(1, math.ceil(v / math.pi) + 1))
    interior = zeros[(zeros > base[0]) & (zeros < base[-1])]
    nodes = np.unique(np.concatenate((base, interior)))
    expected = _cutoffs(v, ratio, n, kind)
    values = _characteristic(
        nodes, np.full_like(nodes, v), np.full_like(nodes, ratio), n, kind
    )
    roots = _refined_roots(nodes, values, v, ratio, n, kind)[kind]
    if len(roots) < len(expected):
        # Only now is the equal-sign bookkeeping worth its cost: the scan has
        # missed a pair whose two sign changes cancelled inside one interval.
        # The cutoffs say where: the missing root of the lowest unfilled order
        # lies between that order's cutoff and either the root that displaced
        # it or V, so only the equal-sign intervals meeting that region are
        # sampled more finely.
        equal = np.flatnonzero(np.signbit(values[:-1]) == np.signbit(values[1:]))
        for level in range(1, _ROOT_REFINEMENTS + 1):
            if len(roots) >= len(expected):
                break
            low, high = _missing_region(roots, expected, v)
            selected = equal[(nodes[equal + 1] > low) & (nodes[equal] < high)]
            if not selected.size:
                break
            step = (nodes[selected + 1] - nodes[selected]) / 2**level
            extra = (
                nodes[selected, None] + step[:, None] * np.arange(1, 2**level)
            ).ravel()
            dense = np.unique(np.concatenate((nodes, extra)))
            roots = _refined_roots(
                dense,
                _characteristic(
                    dense, np.full_like(dense, v), np.full_like(dense, ratio), n, kind
                ),
                v,
                ratio,
                n,
                kind,
            )[kind]
    limits = [*expected, v]
    if len(roots) > len(expected) or any(
        not limits[order] < root <= limits[order + 1]
        for order, root in enumerate(roots)
    ):
        raise ModeError(
            f"the {kind} scan found {len(roots)} roots at V={v:.16g} that do not "
            f"align with the {len(expected)} exact cutoffs below V: the radial "
            f"order of a root cannot be assigned (roots {roots}, cutoffs "
            f"{expected})"
        )
    if len(roots) < len(expected):
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

    Parameters
    ----------
    kind : str
        HE, EH, TE or TM family.
    n : int
        Non-negative azimuthal order.
    m : int
        Positive radial order, counted from highest effective index.
    ratio : float or numpy.ndarray
        Squared real cladding/core index ratio, between zero and one.

    Returns
    -------
    float
        Exact dimensionless cutoff for the fixed material-index ratio.
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

        Parameters
        ----------
        omega : array_like or float
            Positive angular frequencies in rad/s.

        Returns
        -------
        tuple of numpy.ndarray
            Complex refractive index and slope in s/rad, with the input shape.
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

    Parameters
    ----------
    material : GlassIndex or float
        Index model or positive constant refractive index.
    omega : array_like or float
        Positive angular frequencies in rad/s.

    Returns
    -------
    tuple of numpy.ndarray
        Real index and derivative in s/rad, with the input shape.
    """
    if isinstance(material, GlassIndex):
        lossless = lossless_index(material)
        return np.asarray(np.real(lossless.index(omega))), np.asarray(
            np.real(lossless.dindex_domega(omega))
        )
    return np.full_like(omega, material), np.zeros_like(omega)


def _checked_omega(omega: ArrayLike | float) -> Float[np.ndarray, " n_omega"]:
    """Require a nonempty vector of positive finite real angular frequencies.

    Parameters
    ----------
    omega : array_like or float
        Positive angular frequencies in rad/s.

    Returns
    -------
    numpy.ndarray
        Validated frequency vector, shape (n_omega,).
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
    def from_numerical_aperture(  # noqa: PLR0913 - the constructor mirrors the class's keyword surface
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
    ) -> "StepIndexMode":  # noqa: UP037 - the test hook resolves the class while its body runs
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

        Parameters
        ----------
        z : float
            Position along the fibre in m.

        Returns
        -------
        float
            Positive core radius in m.
        """
        radius = float(self.core_radius(np.asarray([z]))[0])
        if not math.isfinite(radius) or radius <= 0:
            raise ModeError("the core radius must be positive and finite")
        return radius

    def _eigenpairs(
        self, omega: Float[np.ndarray, " n_omega"], z: float
    ) -> tuple[Float[np.ndarray, " n_omega"], Float[np.ndarray, " n_omega"]]:
        """Select the m-th root of the requested vector family at each frequency.

        Parameters
        ----------
        omega : array_like or float
            Positive angular frequencies in rad/s.
        z : float
            Position along the fibre in m.

        Returns
        -------
        tuple of numpy.ndarray
            Effective indices and their transverse roots u, shape (n_omega,).
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
        result = np.empty_like(omega)
        transverse = np.empty_like(omega)
        for index in range(omega.size):
            roots = _roots(float(v[index]), float(ratio[index]), self.n, self.kind)
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

        Parameters
        ----------
        v : float
            Normalised frequency V at which the root is missing.
        ratio : float
            Squared real cladding/core index ratio, between zero and one.

        Returns
        -------
        str
            A below-cutoff message when V is at or below the exact cutoff; otherwise
            the root exists but lies within float64 rounding of V (see
            :func:`_roots`), and the message says so rather than calling the mode
            below cutoff.
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

        Parameters
        ----------
        omega : array_like or float
            Positive angular frequencies in rad/s.

        Returns
        -------
        tuple of numpy.ndarray
            Attenuation in 1/m and its derivative in s/(rad m).
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
    def z_constant(self) -> bool:
        """Return whether the radius path is constant.

        Returns
        -------
        bool
            True if the radius path is constant.
        """
        return self.core_radius.is_constant

    @property
    def radius_is_constant(self) -> bool:
        """Return whether the transverse geometry is constant along z.

        Returns
        -------
        bool
            True if the geometry is fixed along z.
        """
        return self.core_radius.is_constant

    @property
    def scale_invariant(self) -> bool:
        """Return False because changing the radius changes the normalised frequency.

        Returns
        -------
        bool
            False, since V changes with core radius.
        """
        return False

    @property
    def channel_label(self) -> ChannelLabel:
        """Return the exact vector family, its indices and its parity.

        Returns
        -------
        ChannelLabel
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
    def azimuthal_order(self) -> int:
        """Return the azimuthal mode index n.

        Returns
        -------
        int
            The mode index n, dimensionless.
        """
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
    def orthogonal_polarisation(self) -> "StepIndexMode | None":  # noqa: UP037 - runtime test hook
        """Return the other hybrid parity, or None for TE and TM.

        Returns
        -------
        StepIndexMode or None
            Opposite hybrid parity; None for TE and TM.
        """
        if self.n == 0:
            return None
        return replace(self, parity="odd" if self.parity == "even" else "even")

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the core-radius path carrying the entire z dependence.

        Returns
        -------
        collections.abc.Mapping
            The core_radius path in m.
        """
        return MappingProxyType({"core_radius": self.core_radius})

    def at_parameters(self, values: Mapping[str, float]) -> "StepIndexMode":  # noqa: UP037 - runtime test hook
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

        Parameters
        ----------
        z : float
            Position along the fibre in m.

        Returns
        -------
        _Profile
            Radius and dimensionless transverse eigenpair at profile_wavelength.
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
        b = float(kvp(self.n, w) / (w * kv(self.n, w)))
        f2 = self.n * (1 / u**2 + 1 / w**2) / (a + b) if self.n else 0.0
        return _Profile(radius, u, w, (cladding / core) ** 2, f2)

    @cached_property
    def _entrance_profile(self) -> _Profile:
        """Keep the required entrance profile on this immutable mode.

        Returns
        -------
        _Profile
            Entrance eigenpair at profile_wavelength.
        """
        return self._solve_profile(0.0)

    def _profile_at(self, z: float) -> _Profile:
        """Reuse a constant geometry's profile and recompute a tapered one.

        Parameters
        ----------
        z : float
            Position along the fibre in m.

        Returns
        -------
        _Profile
            Eigenpair at this geometry and the declared profile wavelength.
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

        Returns
        -------
        tuple of str
            A glass by its name and source, anything else by its ``repr``.
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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

        Parameters
        ----------
        radius : numpy.ndarray
            Dimensionless non-negative radii r/a.
        profile : _Profile
            Lossless eigenpair at the declared profile wavelength.

        Returns
        -------
        tuple of numpy.ndarray
            Unnormalised radial and tangential coefficients, with the radius shape.
        """
        inside = radius <= 1.0
        radial, tangential = np.zeros_like(radius), np.zeros_like(radius)
        if self.n == 0:
            # The TE/TM limiting solutions are separate: Table 12-3's hybrid
            # coefficients divide by n. Tangential E (TE) and radial D (TM)
            # match at the boundary; the arbitrary core boundary amplitude is 1.
            amplitude = np.empty_like(radius)
            amplitude[inside] = jv(1, profile.u * radius[inside]) / jv(1, profile.u)
            amplitude[~inside] = kv(1, profile.w * radius[~inside]) / kv(1, profile.w)
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
        lower = kv(self.n - 1, profile.w * radius[~inside])
        upper = kv(self.n + 1, profile.w * radius[~inside])
        factor = -profile.u / (profile.w * kv(self.n, profile.w))
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

        Parameters
        ----------
        z : float
            Position along the fibre in m.

        Returns
        -------
        tuple of float
            Squared-field and quartic-field area integrals in m².
        """
        profile = self._profile_at(z)
        outer = self.dimlimits(z)[2][0] / profile.radius

        def integrand(radius: float, power: int) -> float:
            """Evaluate the radial intensity moment.

            Parameters
            ----------
            radius : float
                Dimensionless non-negative radius r/a.
            power : int
                Intensity moment, one or two.

            Returns
            -------
            float
                Radial moment including its Jacobian and exact azimuthal integral.
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

        Returns
        -------
        tuple of float
            Squared-field and quartic-field area integrals in m².
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
