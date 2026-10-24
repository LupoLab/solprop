# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adaptive piecewise Chebyshev representation of functions of z.

The solver rests on the cumulative linear operator ``Phi(omega, z) = int D dz``: because
``D`` is diagonal in omega it commutes with itself, so ``exp(Phi(z2) - Phi(z1))`` is the
*exact* linear flow over any step. This module builds the representation that
carries it — a piecewise Chebyshev polynomial in z with array-valued coefficients, built
by Chebfun-style adaptive splitting, integrated exactly in the Chebyshev basis and
evaluated by Clenshaw recursion.

The design, the evidence behind each choice and the traps that are silent are in
``docs/dev/stepper.md`` section 4; the manual page is
``docs/guide/numerics/linear-operator.md``.
"""

import functools
import inspect
import itertools
import logging
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from jaxtyping import Float, Inexact, Int
from numpy.polynomial import chebyshev as chebyshev_basis

from solprop._helpers import NUMBA, Helpers
from solprop.exceptions import TranslationError

__all__ = [
    "DEFAULT_MAX_DEGREE",
    "DEFAULT_MAX_INTERVALS",
    "DEFAULT_TOL",
    "ENVELOPE_FLOOR",
    "INITIAL_DEGREE",
    "MIN_WIDTH_FRACTION",
    "SAMPLING_NOISE_FACTOR",
    "PiecewiseChebyshev",
    "build",
    "build_from_paths",
    "clenshaw",
    "interleave_axis",
    "lobatto_coefficients",
    "lobatto_points",
    "refinement_points",
    "resolve_axis",
    "trim_axis",
]

logger = logging.getLogger(__name__)

DEFAULT_TOL: Final = 1e-12
"""Default tolerance: the bound on the uniform error relative to the value scale."""

DEFAULT_MAX_DEGREE: Final = 64
"""Default highest Chebyshev degree tried on one interval before it is bisected."""

DEFAULT_MAX_INTERVALS: Final = 512
"""Default interval budget; exceeding it sets the fallback flag rather than raising."""

MIN_WIDTH_FRACTION: Final = 1e-6
"""Default narrowest interval the split may produce, as a fraction of the total span."""

INITIAL_DEGREE: Final = 8
"""Degree the adaptive search starts from, here and in the parameter expansion."""

_N_TAIL: Final = 3
ENVELOPE_FLOOR: Final = 1e-10
"""Comparison-scale floor of the acceptance test, as a fraction of the global envelope.

An element whose own values are negligible beside the rest must not drive the
refinement, so its comparison scale is floored here. Exposed because the round-off floor
`solprop.translate.linop` measures has to be measured in this same norm to be comparable
with the tail it is meant to bound.
"""

SAMPLING_NOISE_FACTOR: Final = 4.0
"""Margin of the local sampling-noise floor of the acceptance test (SEP 14 phase 5).

An interval is also accepted when twice its coefficient tail is within this factor of
the relative change its interpolant makes across the rounding of its own node positions,
``eps |z| max|f'| / |f|``: below that the sampled function carries no information about
its own variation. Near a choked gas end, where the pressure goes as the square root of
the distance to the exit, that floor rises above the requested tolerance within half a
millimetre of a 4.2 m exit, and without it the split tiled that region at its width
floor.
"""


def _clenshaw_columns(x: float, c: np.ndarray, out: np.ndarray) -> None:
    """Write NumPy's ``chebval`` of each column of `c` at `x` into `out`.

    Operation for operation NumPy's Clenshaw recursion, so that compiled it gives the
    same values bit for bit. Compiled by :func:`_clenshaw`; plain Python otherwise.

    Parameters
    ----------
    x : float
        Point in [-1, 1].
    c : numpy.ndarray
        Coefficients, shape ``(degree + 1, n_values)``, float64 or complex128.
    out : numpy.ndarray
        The values, shape ``(n_values,)``, of `c`'s dtype; overwritten.
    """
    n, m = c.shape
    if n == 1:
        for j in range(m):
            out[j] = c[0, j] + 0.0 * x
        return
    if n == 2:
        for j in range(m):
            out[j] = c[0, j] + c[1, j] * x
        return
    x2 = 2 * x
    for j in range(m):
        c0 = c[n - 2, j]
        c1 = c[n - 1, j]
        for i in range(3, n + 1):
            tmp = c0
            c0 = c[n - i, j] - c1
            c1 = tmp + c1 * x2
        out[j] = c0 + c1 * x


@functools.cache
def _clenshaw() -> Callable[[float, np.ndarray, np.ndarray], None]:
    """Return :func:`_clenshaw_columns` compiled.

    Compiled, a degree-64 table of 18 values costs under a microsecond against NumPy's
    36 µs, which was most of a trial's host work. numba is imported here, at the first
    evaluation, so that importing solprop does not load it.

    Returns
    -------
    collections.abc.Callable
        The compiled evaluation.
    """
    from solprop._numba_cache import njit  # noqa: PLC0415 - numba on first use

    # The function itself, not a runtime type checker's wrapper around it (the test
    # suite's import hook adds one), which numba can neither compile nor cache.
    # Strict (SEP 25 section 4.2): the tables it evaluates include the integrated
    # weights whose difference is a stage's phase increment.
    return njit(cache=True, strict=True)(inspect.unwrap(_clenshaw_columns))


def clenshaw(
    x: float, coefficients: Inexact[np.ndarray, " n_deg *value"]
) -> Inexact[np.ndarray, " *value"]:
    """Evaluate one interval's Chebyshev series at one point, as NumPy's ``chebval``.

    The same values bit for bit, compiled: a degree-64 series of 18 values costs under a
    microsecond against NumPy's 36 µs. Until numba is loaded NumPy evaluates instead:
    a run on the native backend needs numba for nothing else, and importing it costs
    more than a set-up's thousand NumPy evaluations.

    Parameters
    ----------
    x : float
        Point in the interval's own coordinate, in [-1, 1].
    coefficients : numpy.ndarray
        The series' coefficients, lowest degree first, shape ``(n_terms, *value)``,
        float64 or complex128.

    Returns
    -------
    numpy.ndarray
        The value, shape ``value``, of the coefficients' dtype.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.chebyshev import clenshaw
    >>> float(clenshaw(0.5, np.array([1.0, 2.0, 3.0])))
    0.5
    """
    if "numba" not in sys.modules:
        return np.asarray(chebyshev_basis.chebval(float(x), coefficients))
    flat = np.ascontiguousarray(coefficients).reshape(coefficients.shape[0], -1)
    value = np.empty(flat.shape[1], dtype=coefficients.dtype)
    _clenshaw()(float(x), flat, value)
    return value.reshape(coefficients.shape[1:])


def _clenshaw_points(  # noqa: PLR0912 - one compiled loop, its cases inline
    z: np.ndarray,
    edges: np.ndarray,
    lower: float,
    upper: float,
    c: np.ndarray,
    out: np.ndarray,
) -> int:
    """Write :func:`_clenshaw_columns` at each of several positions into `out`'s rows.

    Position ``p`` takes the interval :meth:`PiecewiseChebyshev.interval_containing`
    takes, the same map onto ``[-1, 1]`` as :meth:`PiecewiseChebyshev.__call__` and
    the recursion of :func:`_clenshaw_columns` operation for operation, so a batch of
    positions gives the values of one call per position bit for bit. The NumPy
    reference backend runs it as written, ``numba_cpu`` compiled by
    :func:`_clenshaw_batch`, and the native backend runs its C++ transcription
    (``solprop._helpers``).

    Parameters
    ----------
    z : numpy.ndarray
        Positions, m, float64, shape ``(n_points,)``.
    edges : numpy.ndarray
        The breakpoints, float64, shape ``(n_intervals + 1,)``, increasing.
    lower, upper : float
        The span widened by its edge slack, m; a position outside is refused.
    c : numpy.ndarray
        Coefficients, shape ``(n_intervals, degree + 1, n_values)``, float64 or
        complex128.
    out : numpy.ndarray
        The values, shape ``(n_points, n_values)``, of `c`'s dtype; overwritten.

    Returns
    -------
    int
        The first position outside ``[lower, upper]``, or ``-1`` when none is (the
        rows from it on are then not written).
    """
    n, m = c.shape[1], c.shape[2]
    last = edges.shape[0] - 2
    for p in range(z.shape[0]):
        zp = z[p]
        if zp < lower or zp > upper:
            return p
        if zp != zp:  # noqa: PLR0124 - NaN sorts last, as in `searchsorted`
            k = last
        else:
            # `searchsorted(edges, zp, side="right") - 1`, clamped to an interval.
            low, high = 0, edges.shape[0]
            while low < high:
                middle = (low + high) // 2
                if edges[middle] <= zp:
                    low = middle + 1
                else:
                    high = middle
            k = min(max(low - 1, 0), last)
        lo = edges[k]
        hi = edges[k + 1]
        xp = (2.0 * zp - lo - hi) / (hi - lo)
        if n == 1:
            for j in range(m):
                out[p, j] = c[k, 0, j] + 0.0 * xp
            continue
        if n == 2:
            for j in range(m):
                out[p, j] = c[k, 0, j] + c[k, 1, j] * xp
            continue
        x2 = 2 * xp
        for j in range(m):
            c0 = c[k, n - 2, j]
            c1 = c[k, n - 1, j]
            for i in range(3, n + 1):
                tmp = c0
                c0 = c[k, n - i, j] - c1
                c1 = tmp + c1 * x2
            out[p, j] = c0 + c1 * xp
    return -1


@functools.cache
def _clenshaw_batch() -> Callable[
    [np.ndarray, np.ndarray, float, float, np.ndarray, np.ndarray], int
]:
    """Return :func:`_clenshaw_points` compiled.

    Returns
    -------
    collections.abc.Callable
        The compiled evaluation.
    """
    from solprop._numba_cache import njit  # noqa: PLC0415 - numba on first use

    # Strict, as _clenshaw_columns is: the stage bundles it serves must equal the
    # one-point reads operation for operation, and include the integrated weights whose
    # difference is a stage's phase increment (SEP 25 section 4.2).
    return njit(cache=True, strict=True)(inspect.unwrap(_clenshaw_points))


_RECENT_POSITIONS: Final = 8
"""Positions whose values a table keeps; a march revisits only the last few."""

_EDGE_SLACK_FRACTION: Final = 1e-10
_TRIM_NOISE_FACTOR: Final = 8.0
_MIN_WIDTH_ULPS: Final = 8.0

_SampleFunction = Callable[
    [Float[np.ndarray, " n_z"]], Inexact[np.ndarray, " n_z *value"]
]
_Outcome = tuple[Inexact[np.ndarray, " n_deg *value"], bool]

_BATCH_BYTES: Final = 1 << 25
"""Most bytes of samples one batched call of a sampler returns (32 MiB)."""

_LOOK_AHEAD_LEVELS: Final = 4
"""Levels below the intervals it must visit that the split resolves early."""


@dataclass(frozen=True)
class PiecewiseChebyshev:
    """A piecewise Chebyshev polynomial in z with array-valued coefficients.

    On interval ``i`` the value is ``sum_j coefficients[i, j] T_j(x)`` with
    ``x = (2 z - lo - hi) / (hi - lo)`` mapping ``[lo, hi]`` onto ``[-1, 1]``. Intervals
    that need a lower degree carry zero coefficients in the high-order slots, which
    Clenshaw handles exactly and at no accuracy cost, so one vectorised pass evaluates
    every interval.

    ``coefficients`` is listed first because jaxtyping resolves the symbolic axis
    ``n_intervals+1`` of ``breakpoints`` only once ``n_intervals`` is bound by an
    earlier argument. Shape agreement between the fields is enforced by those
    annotations under the test session's runtime checker; construction validates what
    shapes cannot express.

    Attributes
    ----------
    coefficients : numpy.ndarray
        Chebyshev coefficients, shape ``(n_intervals, n_deg, *value_shape)``, float64 or
        complex128. Units are those of the represented function.
    breakpoints : numpy.ndarray
        Interval edges in z, shape ``(n_intervals + 1,)``, m, strictly increasing.
    degrees : numpy.ndarray
        Degree actually used on each interval, shape ``(n_intervals,)``. Diagnostic: the
        zero padding makes it unnecessary for evaluation.
    hard_kinks : numpy.ndarray
        Positions in z, m, where the represented function itself jumps or kinks — stage
        boundaries, taper corners. Every entry is an *interior* breakpoint. These are
        tracked apart from the far more numerous edges the adaptive split introduces to
        resolve smooth variation, because steps must land on them and must not
        be capped by the others.
    budget_exceeded : bool, optional
        True when the split ran out of its interval budget and accepted unresolved
        intervals. The caller falls back to per-substep Gauss-Legendre on it.

    Raises
    ------
    TranslationError
        If the interval count is zero, the breakpoints are not finite and strictly
        increasing, the coefficient dtype is neither float64 nor complex128, or a hard
        kink is not an interior breakpoint.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.chebyshev import build
    >>> phi = build(lambda z: 2.0 * z, 0.0, 1.0).integral()
    >>> float(phi(0.5))
    0.25
    """

    coefficients: Inexact[np.ndarray, " n_intervals n_deg *value"]
    breakpoints: Float[np.ndarray, " n_intervals+1"]
    degrees: Int[np.ndarray, " n_intervals"]
    hard_kinks: Float[np.ndarray, " n_kinks"]
    budget_exceeded: bool = False

    def __post_init__(self) -> None:
        """Validate the fields that the shape annotations cannot express.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        if self.coefficients.shape[0] < 1:
            msg = "coefficients: need at least one interval"
            raise TranslationError(msg)
        if not np.all(np.isfinite(self.breakpoints)):
            msg = "breakpoints: every edge must be finite"
            raise TranslationError(msg)
        if not np.all(np.diff(self.breakpoints) > 0.0):
            msg = f"breakpoints must be strictly increasing, got {self.breakpoints}"
            raise TranslationError(msg)
        if self.coefficients.dtype not in (np.float64, np.complex128):
            msg = (
                "coefficients must be float64 or complex128, "
                f"got {self.coefficients.dtype}"
            )
            raise TranslationError(msg)
        # A hard kink that is not an interval edge means the interpolant was built
        # straight through the kink, which is the one thing this representation exists
        # to prevent.
        interior = self.breakpoints[1:-1]
        slack = _EDGE_SLACK_FRACTION * (self.z_span[1] - self.z_span[0])
        for kink in self.hard_kinks:
            if interior.size == 0 or np.min(np.abs(interior - kink)) > slack:
                msg = (
                    f"hard kink at z = {float(kink)!r} is not an interior breakpoint, "
                    "so the interpolant was built across it"
                )
                raise TranslationError(msg)
        # The values at the last few positions: a march evaluates each position
        # several times (a trial's end, its sample, the next anchor), and a Clenshaw
        # pass at degree 64 costs more than the rest of a trial's host work.
        object.__setattr__(self, "_recent", {})

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return the shape of one value.

        Returns
        -------
        tuple of int
            Shape of one value; ``()`` for a scalar-valued function.
        """
        return self.coefficients.shape[2:]

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the ends of the represented span.

        Returns
        -------
        tuple of float
            First and last breakpoint, m.
        """
        return float(self.breakpoints[0]), float(self.breakpoints[-1])

    def __call__(
        self, z: float, *, helpers: Helpers | None = None
    ) -> Inexact[np.ndarray, " *value"]:
        """Evaluate at one position by Clenshaw recursion.

        Clenshaw rather than Horner on monomials: the operation count is the same but it
        is far better conditioned at the degrees the builder reaches, and it keeps the
        coefficients in the basis the exact Chebyshev antiderivative produces. The last
        few positions' values are kept and returned again, read-only, whichever
        implementation formed them (:meth:`at_positions`).

        Parameters
        ----------
        z : float
            Position, m, within the represented span (a few ulp outside is tolerated).
        helpers : solprop._helpers.Helpers or None, optional
            A backend's implementation, for a stage's scalar tables read by the backend
            that runs the stage (SEP 29 section 6.6): the position is then evaluated
            as a batch of one by :meth:`at_positions`. ``None``, for translation and
            every caller outside a backend, takes the shared compiled pass
            (:func:`clenshaw`).

        Returns
        -------
        numpy.ndarray
            The value, shape ``value_shape``.

        Raises
        ------
        TranslationError
            If ``z`` lies outside the represented span.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.translate.chebyshev import build
        >>> f = build(lambda z: np.cos(z), 0.0, 1.0)
        >>> bool(abs(f(0.3) - np.cos(0.3)) < 1e-14)
        True
        """
        recent: dict[float, np.ndarray] = self.__dict__["_recent"]
        kept = recent.get(z)
        if kept is not None:
            return kept
        if helpers is not None:
            return self.at_positions(np.array([z], dtype=np.float64), helpers=helpers)[
                0, ...
            ]
        index = self.interval_containing(z)
        lo = float(self.breakpoints[index])
        hi = float(self.breakpoints[index + 1])
        x = (2.0 * z - lo - hi) / (hi - lo)
        value = clenshaw(x, self.coefficients[index])
        value.flags.writeable = False
        if len(recent) >= _RECENT_POSITIONS:
            recent.clear()
        recent[z] = value
        return value

    def at_positions(
        self, z: Float[np.ndarray, " n_points"], *, helpers: Helpers = NUMBA
    ) -> Inexact[np.ndarray, " n_points *value"]:
        """Evaluate at several positions, one Clenshaw pass for all of them.

        Each row is what calling the table returns at that position, bit for bit: the
        same interval choice (an interior breakpoint belongs to the interval starting
        there), the same map onto ``[-1, 1]`` and the same recursion, so a caller may
        batch the positions it knows without changing a value. Rows are kept as the
        last positions' values, as a call of the table keeps them, so the next step's
        anchor finds its weights already formed.

        The pass is the caller's backend's (`helpers`, SEP 29 section 6.6): a stage's
        rows are evaluated by the backend that runs the stage. The kept rows are shared
        whichever implementation filled them, because every implementation performs the
        operations of a one-point call in its order under strict arithmetic, so a kept
        row is the value any of them would form.

        Parameters
        ----------
        z : numpy.ndarray
            Positions, m, float64, shape ``(n_points,)``, each within the represented
            span (a few ulp outside is tolerated).
        helpers : solprop._helpers.Helpers, optional
            The evaluating backend's implementation; the shared Numba one by default,
            for callers outside a backend.

        Returns
        -------
        numpy.ndarray
            The values, shape ``(n_points, *value_shape)``, read-only.

        Raises
        ------
        TranslationError
            If a position lies outside the represented span.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.translate.chebyshev import build
        >>> f = build(lambda z: np.cos(z), 0.0, 1.0)
        >>> rows = f.at_positions(np.array([0.3, 0.7]))
        >>> bool(rows[0] == f(0.3)) and bool(rows[1] == f(0.7))
        True
        """
        positions = np.ascontiguousarray(z, dtype=np.float64)
        held = self.__dict__.get("_batch")
        if held is None:
            # Formed at the first batch rather than at construction: only the scalar
            # tables are read this way, and a z table's coefficients are large.
            n_intervals, n_deg = self.coefficients.shape[:2]
            held = (
                *self.z_span,
                np.ascontiguousarray(self.breakpoints),
                np.ascontiguousarray(self.coefficients).reshape(n_intervals, n_deg, -1),
            )
            object.__setattr__(self, "_batch", held)
        lo, hi, edges, flat = held
        slack = _EDGE_SLACK_FRACTION * (hi - lo)
        values = np.empty((positions.size, flat.shape[2]), dtype=flat.dtype)
        bad = helpers.points(positions, edges, lo - slack, hi + slack, flat, values)
        if bad >= 0:
            msg = (
                f"z = {float(positions[bad])!r} is outside the represented span "
                f"[{lo}, {hi}]"
            )
            raise TranslationError(msg)
        values = values.reshape(positions.size, *self.value_shape)
        values.flags.writeable = False
        recent: dict[float, np.ndarray] = self.__dict__["_recent"]
        if len(recent) + positions.size > _RECENT_POSITIONS:
            recent.clear()
        for row in range(max(positions.size - _RECENT_POSITIONS, 0), positions.size):
            recent[float(positions[row])] = values[row, ...]
        return values

    def interval_containing(self, z: float) -> int:
        """Return the index of the interval that contains ``z``.

        A position sitting exactly on an interior breakpoint belongs to the interval
        *starting* there, and the right end of the span resolves to the last interval.

        Parameters
        ----------
        z : float
            Position, m. Positions up to ``1e-10`` of the span outside are tolerated,
            because step schedules land on the ends and rounding puts them a few ulp
            out.

        Returns
        -------
        int
            Interval index in ``[0, n_intervals - 1]``.

        Raises
        ------
        TranslationError
            If ``z`` lies further outside the span than that tolerance.

        Examples
        --------
        >>> from solprop.translate.chebyshev import build
        >>> f = build(lambda z: z, 0.0, 1.0, mandatory_breakpoints=(0.5,))
        >>> f.interval_containing(0.5)
        1
        """
        lo, hi = self.z_span
        slack = _EDGE_SLACK_FRACTION * (hi - lo)
        if z < lo - slack or z > hi + slack:
            msg = f"z = {z!r} is outside the represented span [{lo}, {hi}]"
            raise TranslationError(msg)
        index = int(np.searchsorted(self.breakpoints, z, side="right")) - 1
        return min(max(index, 0), self.breakpoints.size - 2)

    def interval_width(self, z: float) -> float:
        """Return the width of the interval containing ``z``.

        This is the *local* step cap of ``docs/dev/stepper.md`` section 3. The narrowest
        interval anywhere is not a cap: the split resolves a square-root entrance with
        intervals four orders of magnitude below the median, and capping globally on
        those would cost four orders of magnitude in step count.

        Parameters
        ----------
        z : float
            Position, m.

        Returns
        -------
        float
            Width of the containing interval, m.

        Examples
        --------
        >>> from solprop.translate.chebyshev import build
        >>> f = build(lambda z: z, 0.0, 1.0, mandatory_breakpoints=(0.25,))
        >>> f.interval_width(0.1)
        0.25
        """
        index = self.interval_containing(z)
        return float(self.breakpoints[index + 1] - self.breakpoints[index])

    def next_hard_kink(self, z: float) -> float | None:
        """Return the first hard kink beyond ``z``, or None if there is none.

        Strictly beyond, with the same edge tolerance as :meth:`interval_containing`, so
        a step starting on a kink clears it instead of being capped to zero length.

        Parameters
        ----------
        z : float
            Position, m.

        Returns
        -------
        float or None
            Position of the next hard kink, m, or None when none remains.

        Examples
        --------
        >>> from solprop.translate.chebyshev import build
        >>> f = build(lambda z: z, 0.0, 1.0, mandatory_breakpoints=(0.4,))
        >>> f.next_hard_kink(0.0), f.next_hard_kink(0.4)
        (0.4, None)
        """
        lo, hi = self.z_span
        slack = _EDGE_SLACK_FRACTION * (hi - lo)
        beyond = self.hard_kinks[self.hard_kinks > z + slack]
        return float(np.min(beyond)) if beyond.size > 0 else None

    def integral(self) -> "PiecewiseChebyshev":  # noqa: UP037 - bound after the class body; beartype evaluates eagerly
        """Return the antiderivative, continuous across intervals and zero at the start.

        Integration is a coefficient recurrence in the Chebyshev basis, not a
        quadrature, so it is exact for the polynomial it is given: the only
        approximation left in the cumulative operator is the interpolation of the
        integrand. The constant of integration lives entirely in the ``T_0`` coefficient
        and is fixed from the left, interval by interval, so the result is continuous by
        construction.

        Returns
        -------
        PiecewiseChebyshev
            The antiderivative with respect to z, vanishing at the first breakpoint,
            with the same breakpoints and hard kinks and one more coefficient per
            interval.

        Examples
        --------
        >>> from solprop.translate.chebyshev import build
        >>> phi = build(lambda z: 3.0 * z**2, 0.0, 2.0).integral()
        >>> bool(abs(phi(2.0) - 8.0) < 1e-14)
        True
        """
        running = np.zeros(self.value_shape, dtype=self.coefficients.dtype)
        antiderivatives = []
        for index in range(self.degrees.size):
            lo = float(self.breakpoints[index])
            hi = float(self.breakpoints[index + 1])
            piece = chebyshev_basis.chebint(
                self.coefficients[index], m=1, k=0, axis=0, scl=0.5 * (hi - lo)
            )
            piece[0] += running - np.asarray(chebyshev_basis.chebval(-1.0, piece))
            antiderivatives.append(piece)
            running = np.asarray(chebyshev_basis.chebval(1.0, piece))
        return PiecewiseChebyshev(
            np.stack(antiderivatives),
            self.breakpoints,
            self.degrees + 1,
            self.hard_kinks,
            self.budget_exceeded,
        )

    def derivative(self) -> "PiecewiseChebyshev":  # noqa: UP037 - bound after the class body; beartype evaluates eagerly
        """Return the derivative with respect to z.

        Returns
        -------
        PiecewiseChebyshev
            The derivative, with the same breakpoints and hard kinks and one fewer
            coefficient per interval (a single zero coefficient at the floor).

        Examples
        --------
        >>> from solprop.translate.chebyshev import build
        >>> f = build(lambda z: z**2, 0.0, 1.0).derivative()
        >>> bool(abs(f(0.5) - 1.0) < 1e-14)
        True
        """
        widths = np.diff(self.breakpoints)
        pieces = [
            chebyshev_basis.chebder(
                self.coefficients[index], m=1, axis=0, scl=2.0 / float(widths[index])
            )
            for index in range(self.degrees.size)
        ]
        return PiecewiseChebyshev(
            np.stack(pieces),
            self.breakpoints,
            np.maximum(self.degrees - 1, 0),
            self.hard_kinks,
            self.budget_exceeded,
        )


@dataclass(frozen=True)
class _Piece:
    """One resolved interval and the Chebyshev coefficients on it.

    Attributes
    ----------
    lo, hi : float
        Interval ends in z, m.
    coefficients : numpy.ndarray
        Coefficients on ``[lo, hi]``, shape ``(n_deg, *value_shape)``.
    """

    lo: float
    hi: float
    coefficients: np.ndarray


def build(  # noqa: PLR0913 - the documented signature takes every option by name
    f: _SampleFunction,
    z0: float,
    z1: float,
    *,
    tol: float = DEFAULT_TOL,
    mandatory_breakpoints: Sequence[float] | np.ndarray = (),
    n_max: int = DEFAULT_MAX_DEGREE,
    min_width: float | None = None,
    max_intervals: int = DEFAULT_MAX_INTERVALS,
) -> PiecewiseChebyshev:
    """Build an adaptive piecewise Chebyshev representation of ``f`` on ``[z0, z1]``.

    The split is Chebfun-style: sample at Chebyshev-Lobatto points, accept when the
    coefficient tail is negligible, otherwise double the degree up to ``n_max``,
    otherwise bisect. Declared breakpoints are mandatory edges and become the hard
    kinks, so a jump that is known in advance is never crossed; a jump that is not
    declared is *located* by the bisection down to ``min_width``.

    Parameters
    ----------
    f : callable
        Sampled vectorised: ``f(z)`` takes positions of shape ``(n_z,)`` in m and
        returns an array of shape ``(n_z, *value_shape)``, float64 or complex128.
        Sampling is vectorised because the operator is evaluated over a whole omega grid
        at once.
    z0, z1 : float
        Ends of the domain, m, with ``z1 > z0``.
    tol : float, optional
        Target bound on the uniform error relative to the value scale, ``> 0``. An
        interval is accepted when twice the relative coefficient tail is at most
        ``tol``; the factor two is the Chebyshev aliasing bound, so ``tol`` bounds the
        error itself. Use ``1e-12`` when the frame quantities are analytic, ``1e-8``
        when differenced.
    mandatory_breakpoints : array_like, optional
        Positions in z, m, inside ``[z0, z1]`` that must be interval edges: stage
        boundaries, taper corners. The interior ones become the
        :class:`PiecewiseChebyshev` hard kinks.
    n_max : int, optional
        Highest degree tried on one interval before bisecting, ``>= 3``.
    min_width : float or None, optional
        Narrowest interval the split may produce, m. None means ``MIN_WIDTH_FRACTION *
        (z1 - z0)``. It is absolute and computed once from the whole span: derived per
        call it lets an already tiny segment recurse several further decades and exhaust
        the budget without getting closer to resolving anything.
    max_intervals : int, optional
        Interval budget, ``>= 1``. Exceeding it stops the split, accepts what is left
        and sets the :class:`PiecewiseChebyshev` fallback flag rather than raising.
        Mandatory breakpoints are honoured whatever the budget, so a domain declaring
        more of them than the budget allows still gets one interval per segment.

    Returns
    -------
    PiecewiseChebyshev
        The representation of ``f``. Call :meth:`PiecewiseChebyshev.integral` for its
        cumulative integral.

    Raises
    ------
    TranslationError
        If the domain, the tolerance, the degree or the budget is invalid, if a
        mandatory breakpoint lies outside the domain, or if ``f`` returns the wrong
        leading length, an empty value shape, or a non-finite sample.

    Notes
    -----
    What the build reproduces exactly, to round-off, whatever ``tol`` is:

    - any polynomial in z of degree at most ``n_max - 3`` on each declared segment. The
      interpolation at the Chebyshev-Lobatto points is exact for it and the tail test
      then sees three vanishing coefficients. Degrees ``n_max - 2`` to ``n_max`` are
      still represented exactly but are not *recognised* as resolved, so the split
      bisects them until it reaches ``min_width``;
    - a constant or piecewise-constant function, as one interval of degree 0 per
      declared segment, whose integral is one interval of degree 1;
    - the round trip ``build(f).integral().derivative()``, which returns the original
      coefficients up to the round-off of the two recurrences.

    The other round trip, the one the interaction picture needs -- ``exp(-dPhi)`` as the
    exact inverse of ``exp(+dPhi)`` -- is the caller's: negate one evaluated exponent
    rather than evaluating the table twice.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.chebyshev import build
    >>> f = build(lambda z: np.exp(-z), 0.0, 1.0)
    >>> int(f.degrees.size)
    1
    >>> bool(abs(f(0.25) - np.exp(-0.25)) < 1e-14)
    True
    """
    width_floor = _validate(z0, z1, tol, n_max, min_width, max_intervals)
    edges = _mandatory_edges(z0, z1, mandatory_breakpoints)
    pieces, exceeded = _split_segments(
        f,
        edges,
        tol=tol,
        n_max=n_max,
        min_width=width_floor,
        max_intervals=max_intervals,
    )
    return _assemble(pieces, edges[1:-1], exceeded)


def build_from_paths(  # noqa: PLR0913 - the documented signature takes every option
    paths: Sequence[Callable[[Float[np.ndarray, " n_z"]], Float[np.ndarray, " n_z"]]],
    f_of_params: Callable[..., Inexact[np.ndarray, " n_z *value"]],
    z0: float,
    z1: float,
    *,
    tol: float = DEFAULT_TOL,
    mandatory_breakpoints: Sequence[float] | np.ndarray = (),
    n_max: int = DEFAULT_MAX_DEGREE,
    min_width: float | None = None,
    max_intervals: int = DEFAULT_MAX_INTERVALS,
) -> PiecewiseChebyshev:
    """Resolve the scalar paths first, then the array-valued function on their union.

    The scalar parameter paths — density, core radius, any other profile of z — are
    cheap to evaluate and carry the structure that makes the operator hard: they are
    resolved first, within each declared segment separately so that a declared kink is
    never crossed by a path fit, and the union of all the edges they produce becomes the
    mandatory split of the expensive array-valued function.

    Parameters
    ----------
    paths : sequence of callable
        Scalar paths, each mapping positions of shape ``(n_z,)`` in m to values of shape
        ``(n_z,)``.
    f_of_params : callable
        Called as ``f_of_params(z, *values)`` with the positions and one array of shape
        ``(n_z,)`` per path, returning an array of shape ``(n_z, *value_shape)``. The
        positions come first so that an operator which also depends on z directly -- a
        co-moving frame quantity, say -- needs no path of its own, and so that a build
        with no paths at all is exactly :func:`build`.
    z0, z1 : float
        Ends of the domain, m, with ``z1 > z0``.
    tol : float, optional
        Target bound on the uniform error relative to the value scale; see
        :func:`build`.
    mandatory_breakpoints : array_like, optional
        Positions in z, m, that must be interval edges; the interior ones become the
        hard kinks.
    n_max : int, optional
        Highest degree tried on one interval before bisecting, ``>= 3``.
    min_width : float or None, optional
        Narrowest interval the split may produce, m; see :func:`build`.
    max_intervals : int, optional
        Interval budget for each stage of the build, ``>= 1``.

    Returns
    -------
    PiecewiseChebyshev
        The representation of ``f_of_params`` evaluated along the paths.

    Raises
    ------
    TranslationError
        For the same reasons as :func:`build`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.chebyshev import build_from_paths
    >>> density = lambda z: 1.0 + z
    >>> f = build_from_paths([density], lambda z, rho: rho**2, 0.0, 1.0)
    >>> bool(abs(f(0.5) - 2.25) < 1e-14)
    True
    """
    width_floor = _validate(z0, z1, tol, n_max, min_width, max_intervals)
    mandatory = _mandatory_edges(z0, z1, mandatory_breakpoints)
    path_edges = list(mandatory)
    exceeded = False
    for path in paths:
        pieces, over = _split_segments(
            path,
            mandatory,
            tol=tol,
            n_max=n_max,
            min_width=width_floor,
            max_intervals=max_intervals,
        )
        exceeded = exceeded or over
        path_edges.extend(piece.hi for piece in pieces)

    pieces, over = _split_segments(
        functools.partial(_along_paths, paths, f_of_params),
        np.unique(np.asarray(path_edges, dtype=np.float64)),
        tol=tol,
        n_max=n_max,
        min_width=width_floor,
        max_intervals=max_intervals,
    )
    return _assemble(pieces, mandatory[1:-1], exceeded or over)


def _along_paths(
    paths: Sequence[Callable[[Float[np.ndarray, " n_z"]], Float[np.ndarray, " n_z"]]],
    f_of_params: Callable[..., Inexact[np.ndarray, " n_z *value"]],
    z: Float[np.ndarray, " n_z"],
) -> Inexact[np.ndarray, " n_z *value"]:
    """Evaluate the paths at ``z`` and hand ``z`` and their values to ``f_of_params``.

    Parameters
    ----------
    paths : sequence of callable
        Scalar paths of z.
    f_of_params : callable
        Called as ``f_of_params(z, *values)``, one array per path.
    z : numpy.ndarray
        Positions, m, shape ``(n_z,)``.

    Returns
    -------
    numpy.ndarray
        Samples of shape ``(n_z, *value_shape)``.
    """
    return f_of_params(z, *(path(z) for path in paths))


def _validate(
    z0: float,
    z1: float,
    tol: float,
    n_max: int,
    min_width: float | None,
    max_intervals: int,
) -> float:
    """Check the build arguments and return the absolute width floor.

    Parameters
    ----------
    z0, z1 : float
        Ends of the domain, m.
    tol : float
        Target bound on the uniform relative error.
    n_max : int
        Highest degree tried before bisecting.
    min_width : float or None
        Narrowest interval allowed, m, or None for the default fraction of the span.
    max_intervals : int
        Interval budget.

    Returns
    -------
    float
        The width floor, m, absolute and derived once from the whole span.

    Raises
    ------
    TranslationError
        Naming the offending argument.
    """
    if not z1 > z0:
        msg = f"need z1 > z0, got z0 = {z0!r}, z1 = {z1!r}"
        raise TranslationError(msg)
    if not tol > 0.0:
        msg = f"tol must be positive, got {tol!r}"
        raise TranslationError(msg)
    if n_max < _N_TAIL:
        msg = f"n_max must be at least {_N_TAIL}, got {n_max!r}"
        raise TranslationError(msg)
    if max_intervals < 1:
        msg = f"max_intervals must be at least 1, got {max_intervals!r}"
        raise TranslationError(msg)
    span = z1 - z0
    # The endpoint samples are inset by one ulp of the interval scale; an interval that
    # narrow would have its two ends collapse onto each other.
    floor = _MIN_WIDTH_ULPS * float(np.spacing(max(abs(z0), abs(z1), span)))
    if min_width is None:
        return MIN_WIDTH_FRACTION * span
    if min_width < floor:
        msg = (
            f"min_width must be at least {floor!r} m on this domain, got {min_width!r}"
        )
        raise TranslationError(msg)
    return min_width


def _mandatory_edges(
    z0: float, z1: float, mandatory_breakpoints: Sequence[float] | np.ndarray
) -> Float[np.ndarray, " n_edges"]:
    """Return the sorted, deduplicated mandatory edges including both domain ends.

    Parameters
    ----------
    z0, z1 : float
        Ends of the domain, m.
    mandatory_breakpoints : array_like
        Declared breakpoints, m.

    Returns
    -------
    numpy.ndarray
        Edges in increasing order, m, starting at ``z0`` and ending at ``z1``.

    Raises
    ------
    TranslationError
        If a declared breakpoint is not finite or lies outside ``[z0, z1]``.
    """
    declared = np.asarray(mandatory_breakpoints, dtype=np.float64).reshape(-1)
    if declared.size > 0 and not np.all(np.isfinite(declared)):
        msg = f"mandatory_breakpoints must be finite, got {declared}"
        raise TranslationError(msg)
    if declared.size > 0 and (declared.min() < z0 or declared.max() > z1):
        msg = (
            f"mandatory_breakpoints must lie inside [{z0}, {z1}], "
            f"got [{declared.min()}, {declared.max()}]"
        )
        raise TranslationError(msg)
    return np.unique(np.concatenate([np.asarray([z0, z1], dtype=np.float64), declared]))


def _split_segments(
    f: _SampleFunction,
    edges: Float[np.ndarray, " n_edges"],
    *,
    tol: float,
    n_max: int,
    min_width: float,
    max_intervals: int,
) -> tuple[list[_Piece], bool]:
    """Resolve ``f`` on every segment between consecutive edges.

    Parameters
    ----------
    f : callable
        The vectorised sampler.
    edges : numpy.ndarray
        Mandatory edges in increasing order, m.
    tol : float
        Target bound on the uniform relative error.
    n_max : int
        Highest degree tried before bisecting.
    min_width : float
        Narrowest interval allowed, m.
    max_intervals : int
        Interval budget, shared across the segments.

    Returns
    -------
    list of _Piece
        The resolved intervals, in increasing z.
    bool
        True when the budget was exceeded.
    """
    pieces: list[_Piece] = []
    exceeded = False
    for lo, hi in itertools.pairwise(edges):
        found, over = _split(
            f,
            float(lo),
            float(hi),
            tol=tol,
            n_max=n_max,
            min_width=min_width,
            budget=max_intervals - len(pieces),
        )
        pieces.extend(found)
        exceeded = exceeded or over
    return pieces, exceeded


class _BatchedSampler:
    """Sample several groups of positions in as few calls of ``f`` as memory allows.

    The first call of a walk samples its first group alone, which fixes the size of one
    sample; after that groups are concatenated into calls of at most
    :data:`_BATCH_BYTES` of samples each, and the samples split back per group.

    Parameters
    ----------
    f : callable
        The vectorised sampler.
    """

    def __init__(self, f: _SampleFunction) -> None:
        self.f = f
        self.per_position = 0

    def __call__(
        self, groups: Sequence[Float[np.ndarray, " n_z"]]
    ) -> list[Inexact[np.ndarray, " n_z *value"]]:
        """Return the samples of each group, as :func:`_sample` returns them.

        Parameters
        ----------
        groups : sequence of numpy.ndarray
            Positions, m, one array per interval.

        Returns
        -------
        list of numpy.ndarray
            The samples of each group, each C-contiguous.
        """
        sampled: list[Inexact[np.ndarray, " n_z *value"]] = []
        remaining = list(groups)
        if self.per_position == 0:
            first = _sample(self.f, remaining.pop(0))
            self.per_position = max(first[0].nbytes, 1)
            sampled.append(first)
        chunk: list[Float[np.ndarray, " n_z"]] = []
        held = 0
        for positions in remaining:
            if chunk and (held + positions.size) * self.per_position > _BATCH_BYTES:
                sampled.extend(self._joined(chunk))
                chunk, held = [], 0
            chunk.append(positions)
            held += positions.size
        if chunk:
            sampled.extend(self._joined(chunk))
        return sampled

    def capacity(self, n_max: int) -> int:
        """Return how many intervals' samples at degree `n_max` fit in one batch.

        Parameters
        ----------
        n_max : int
            Highest degree tried.

        Returns
        -------
        int
            At least one; one until the first sample fixes the size of a sample.
        """
        if self.per_position == 0:
            return 1
        return max(_BATCH_BYTES // ((n_max + 1) * self.per_position), 1)

    def _joined(
        self, groups: Sequence[Float[np.ndarray, " n_z"]]
    ) -> list[Inexact[np.ndarray, " n_z *value"]]:
        """Sample the concatenated groups in one call and split the samples back.

        Parameters
        ----------
        groups : sequence of numpy.ndarray
            Positions, m, one array per interval.

        Returns
        -------
        list of numpy.ndarray
            The samples of each group, each a contiguous copy.
        """
        if len(groups) == 1:
            return [_sample(self.f, groups[0])]
        values = _sample(self.f, np.concatenate(groups))
        ends = np.cumsum([positions.size for positions in groups])[:-1]
        return [np.ascontiguousarray(part) for part in np.split(values, ends)]


def _split(  # noqa: PLR0913 - the split is driven by the public keywords
    f: _SampleFunction,
    lo: float,
    hi: float,
    *,
    tol: float,
    n_max: int,
    min_width: float,
    budget: int,
) -> tuple[list[_Piece], bool]:
    """Bisect ``[lo, hi]`` until every part is resolved, at the floor, or out of budget.

    The walk is depth-first and resolves one interval at a time, but the sampling is
    batched: when it reaches an interval not yet resolved, :func:`_look_ahead` resolves
    it together with the intervals still pending and, a few levels down, the halves
    the walk is about to visit, sampling each level in one call of ``f``. The outcome
    of :func:`_resolve` depends on its interval alone, so the walk's decisions, and
    the pieces, are those of resolving interval by interval wherever ``f`` is pointwise
    in z. If a batched call raises, the walk falls back to one interval at a time and
    raises at the interval it always did.

    Parameters
    ----------
    f : callable
        The vectorised sampler.
    lo, hi : float
        Ends of the segment, m.
    tol : float
        Target bound on the uniform relative error.
    n_max : int
        Highest degree tried before bisecting.
    min_width : float
        Narrowest interval allowed, m.
    budget : int
        Intervals this segment may produce; the split stops before exceeding it.

    Returns
    -------
    list of _Piece
        The resolved intervals, in increasing z.
    bool
        True when the budget was exceeded.
    """
    # An explicit stack, never recursion: a square-root entrance refines geometrically
    # and would otherwise risk the recursion limit. Pushing the right half first and
    # popping from the end walks the segment depth-first from the left, so pieces come
    # out sorted.
    pieces: list[_Piece] = []
    pending = [(lo, hi)]
    exceeded = False
    ahead: dict[tuple[float, float], _Outcome] = {}
    sample: _BatchedSampler | None = _BatchedSampler(f)
    while pending:
        left, right = pending.pop()
        # Each split adds one to this count and an acceptance leaves it unchanged, so
        # `budget - count` is how many more splits the walk can make.
        count = len(pieces) + len(pending) + 1
        if sample is not None and (left, right) not in ahead:
            try:
                _look_ahead(
                    sample,
                    [(left, right), *reversed(pending)],
                    ahead,
                    tol=tol,
                    n_max=n_max,
                    min_width=min_width,
                    splits=budget - count,
                )
            except Exception:
                sample = None
                ahead.clear()
        outcome = ahead.pop((left, right), None)
        if outcome is None:
            outcome = _resolve(f, left, right, tol=tol, n_max=n_max)
        coefficients, resolved = outcome
        out_of_budget = count >= budget
        if resolved or right - left <= min_width or out_of_budget:
            exceeded = exceeded or (out_of_budget and not resolved)
            pieces.append(_Piece(left, right, _trim(coefficients)))
        else:
            middle = 0.5 * (left + right)
            pending.extend([(middle, right), (left, middle)])
    if exceeded:
        logger.warning(
            "piecewise Chebyshev split exceeded its budget of %d intervals on "
            "[%.6g, %.6g] m; the representation is unresolved there and the caller "
            "should fall back to per-substep Gauss-Legendre",
            budget,
            lo,
            hi,
        )
    return pieces, exceeded


def _look_ahead(  # noqa: PLR0913 - the split's keywords plus the look-ahead's bound
    sample: _BatchedSampler,
    frontier: Sequence[tuple[float, float]],
    ahead: dict[tuple[float, float], _Outcome],
    *,
    tol: float,
    n_max: int,
    min_width: float,
    splits: int,
) -> None:
    """Resolve the intervals the walk is about to visit, a level at a time.

    The frontier is the interval the walk has reached, first, and those still pending:
    it visits all of them whatever their outcomes. Below them it visits both halves of
    every interval it splits, and it splits an interval exactly when the interval is
    unresolved, wider than the floor and within the budget. The look-ahead follows the
    first two conditions for :data:`_LOOK_AHEAD_LEVELS` levels, resolving each level's
    new intervals in one batch, and stops once it has found as many splits as the
    budget still allows. It therefore resolves an interval the walk never visits only
    where the budget runs out inside the levels it looked into. It holds at most
    :meth:`_BatchedSampler.capacity` intervals ahead of the walk.

    Parameters
    ----------
    sample : _BatchedSampler
        The batched sampler.
    frontier : sequence of tuple of float
        ``(lo, hi)`` in m, of intervals the walk will visit, the one it needs now
        first.
    ahead : dict
        Outcomes of :func:`_resolve` by interval ``(lo, hi)``, already resolved ahead
        of the walk; the new ones are added to it.
    tol : float
        Target bound on the uniform relative error.
    n_max : int
        Highest degree tried before bisecting.
    min_width : float
        Narrowest interval allowed, m.
    splits : int
        How many more splits the budget allows.
    """
    level = list(frontier)
    found = 0
    for _ in range(_LOOK_AHEAD_LEVELS + 1):
        new = [interval for interval in level if interval not in ahead]
        room = sample.capacity(n_max) - len(ahead)
        partial = len(new) > max(room, 1)
        if partial:
            # Held ahead of the walk, the coefficients of a level of large samples
            # would outgrow the memory one interval at a time needs: take the first
            # interval the walk needs and as many more as fit, and look no deeper.
            new = new[: max(room, 1)]
        if new:
            ahead.update(
                zip(new, _resolve_many(sample, new, tol=tol, n_max=n_max), strict=True)
            )
        if partial:
            return
        following: list[tuple[float, float]] = []
        for left, right in level:
            if not ahead[left, right][1] and right - left > min_width:
                found += 1
                middle = 0.5 * (left + right)
                following.extend([(left, middle), (middle, right)])
        if not following or found > splits:
            return
        level = following


def _resolve_many(
    sample: _BatchedSampler,
    intervals: Sequence[tuple[float, float]],
    *,
    tol: float,
    n_max: int,
) -> list[_Outcome]:
    """Run :func:`_resolve` on several intervals, sampling each round in one call.

    Interval by interval the arithmetic is :func:`_resolve`'s: the same positions, the
    same transform and acceptance test, the same degree doubling. Only the calls of
    ``f`` are shared: each round's positions of every interval still being refined are
    sampled together by `sample`.

    Parameters
    ----------
    sample : _BatchedSampler
        The batched sampler.
    intervals : sequence of tuple of float
        ``(lo, hi)`` in m.
    tol : float
        Target bound on the uniform relative error.
    n_max : int
        Highest degree tried.

    Returns
    -------
    list of tuple
        Per interval, the coefficients and whether the interval is resolved, as
        :func:`_resolve` returns them.
    """
    degree = min(INITIAL_DEGREE, n_max)
    values = sample([_sample_positions(degree, lo, hi) for lo, hi in intervals])
    outcomes: list[_Outcome | None] = [None] * len(intervals)
    active = list(range(len(intervals)))
    while active:
        refine: list[int] = []
        tested = _accept_many(
            [values[index] for index in active],
            [intervals[index] for index in active],
            tol=tol,
        )
        for index, (coefficients, accepted) in zip(active, tested, strict=True):
            if accepted:
                outcomes[index] = (coefficients, True)
            elif 2 * degree > n_max:
                outcomes[index] = (coefficients, False)
            else:
                refine.append(index)
        if refine:
            fine = sample(
                [_refinement_positions(degree, *intervals[index]) for index in refine]
            )
            for index, added in zip(refine, fine, strict=True):
                values[index] = _interleave(values[index], added)
            degree *= 2
        active = refine
    return [outcome for outcome in outcomes if outcome is not None]


def _accept_many(
    values: Sequence[Inexact[np.ndarray, " n_deg *value"]],
    intervals: Sequence[tuple[float, float]],
    *,
    tol: float,
) -> list[_Outcome]:
    """Transform each interval's samples and apply :func:`_resolve`'s acceptance test.

    Parameters
    ----------
    values : sequence of numpy.ndarray
        Samples per interval, all of shape ``(n_deg, *value_shape)``.
    intervals : sequence of tuple of float
        ``(lo, hi)`` in m, per interval.
    tol : float
        Target bound on the uniform relative error.

    Returns
    -------
    list of tuple
        Per interval, the coefficients and whether the tail passes the test.
    """
    coefficients, ratios, noises = _tails_many(values, intervals)
    floors = SAMPLING_NOISE_FACTOR * noises
    passed = 2.0 * ratios <= np.maximum(tol, floors)
    return [
        (one, bool(accepted))
        for one, accepted in zip(coefficients, passed, strict=True)
    ]


def _tails_many(
    values: Sequence[Inexact[np.ndarray, " n_deg *value"]],
    intervals: Sequence[tuple[float, float]],
) -> tuple[
    list[Inexact[np.ndarray, " n_deg *value"]],
    Float[np.ndarray, " n_intervals"],
    Float[np.ndarray, " n_intervals"],
]:
    """Return the coefficients, :func:`_tail_ratio` and :func:`_sampling_noise` of each.

    Intervals of one round share a degree, so their samples stack along a new axis
    after the first, and the transform, the tail and the sampling-noise floor are
    computed for all of them at once with each interval's own arithmetic: the transform
    and every elementwise operation act column by column, the maxima are exact, and a
    sum over the degree axis of an array with more than one column accumulates row by
    row whatever the number of columns. The exception is a single-element value, whose
    sum NumPy takes pairwise; such intervals, and a lone interval, are taken one at a
    time.

    Parameters
    ----------
    values : sequence of numpy.ndarray
        Samples per interval, all of shape ``(n_deg, *value_shape)``.
    intervals : sequence of tuple of float
        ``(lo, hi)`` in m, per interval.

    Returns
    -------
    list of numpy.ndarray
        The coefficients of each interval, C-contiguous.
    numpy.ndarray
        The relative coefficient tail of each.
    numpy.ndarray
        The relative sampling noise of each.
    """
    if len(values) < 2 or values[0][0].size < 2:
        coefficients = [lobatto_coefficients(samples) for samples in values]
        ratios = np.array([_tail_ratio(one) for one in coefficients])
        noises = np.array(
            [
                _sampling_noise(one, lo, hi)
                for one, (lo, hi) in zip(coefficients, intervals, strict=True)
            ]
        )
        return coefficients, ratios, noises
    stacked = lobatto_coefficients(np.stack(values, axis=1))
    count = len(values)
    lo = np.array([interval[0] for interval in intervals], dtype=np.float64)
    hi = np.array([interval[1] for interval in intervals], dtype=np.float64)
    per = (count,) + (1,) * (stacked.ndim - 2)
    magnitude = np.abs(stacked)
    envelope = magnitude.max(axis=0)
    global_envelope = envelope.reshape(count, -1).max(axis=1)
    scale = np.maximum(envelope, (global_envelope * ENVELOPE_FLOOR).reshape(per))
    order = np.arange(stacked.shape[0], dtype=np.float64)
    order = order.reshape((-1,) + (1,) * (stacked.ndim - 1))
    slope = (2.0 / (hi - lo)).reshape(per) * (order**2 * magnitude).sum(axis=0)
    spacing = np.spacing(np.maximum(np.abs(lo), np.abs(hi))).reshape(per)
    tail = magnitude[-_N_TAIL:].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        noises = (spacing * slope / scale).reshape(count, -1).max(axis=1)
        ratios = (tail / scale).reshape(count, -1).max(axis=1)
    zero = global_envelope == 0.0
    coefficients = [np.ascontiguousarray(stacked[:, index]) for index in range(count)]
    return coefficients, np.where(zero, 0.0, ratios), np.where(zero, 0.0, noises)


def _resolve(
    f: _SampleFunction, lo: float, hi: float, *, tol: float, n_max: int
) -> tuple[Inexact[np.ndarray, " n_deg *value"], bool]:
    """Raise the degree on ``[lo, hi]`` until the coefficient tail is negligible.

    Parameters
    ----------
    f : callable
        The vectorised sampler.
    lo, hi : float
        Ends of the interval, m.
    tol : float
        Target bound on the uniform relative error.
    n_max : int
        Highest degree tried.

    Returns
    -------
    numpy.ndarray
        Chebyshev coefficients on ``[lo, hi]``, shape ``(n_deg, *value_shape)``.
    bool
        True when the interval is resolved to ``tol``.
    """
    degree = min(INITIAL_DEGREE, n_max)
    values = _sample(f, _sample_positions(degree, lo, hi))
    while True:
        coefficients = lobatto_coefficients(values)
        # The factor two is the Chebyshev aliasing bound |f - p_n| <= 2 sum_{k>n} |a_k|
        # (Trefethen, Approximation Theory and Approximation Practice, theorem 8.2), so
        # tol bounds the uniform error itself rather than the coefficient tail. A tail
        # at the sampling noise of the interval's own positions is resolved too: no
        # finer sampling in floating-point z can reduce it.
        floor = SAMPLING_NOISE_FACTOR * _sampling_noise(coefficients, lo, hi)
        if 2.0 * _tail_ratio(coefficients) <= max(tol, floor):
            return coefficients, True
        if 2 * degree > n_max:
            return coefficients, False
        values = _interleave(values, _sample(f, _refinement_positions(degree, lo, hi)))
        degree *= 2


def _sampling_noise(
    coefficients: Inexact[np.ndarray, " n_deg *value"], lo: float, hi: float
) -> float:
    """Return the relative change of an interpolant across the rounding of its nodes.

    A node at ``z`` is a floating-point position, uncertain by ``spacing(z)``, so a
    function sampled there is uncertain by ``spacing(z) |f'(z)|``. Relative to the
    acceptance test's per-element scale this bounds the noise floor of the coefficient
    tail. ``|f'|`` is bounded from the coefficients themselves, ``(2 / w) sum_k k**2
    |a_k|`` on an interval of width ``w``; a noisy tail contributes to it only at the
    level of the noise times the spacing, which is negligible.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Coefficients on the interval, shape ``(n_deg, *value_shape)``.
    lo, hi : float
        Ends of the interval, m.

    Returns
    -------
    float
        The worst relative floor over the value shape; zero for a zero series.
    """
    magnitude = np.abs(coefficients)
    envelope = magnitude.max(axis=0)
    global_envelope = float(envelope.max())
    if global_envelope == 0.0:
        return 0.0
    scale = np.maximum(envelope, global_envelope * ENVELOPE_FLOOR)
    order = np.arange(coefficients.shape[0], dtype=np.float64)
    order = order.reshape((-1,) + (1,) * (coefficients.ndim - 1))
    slope = (2.0 / (hi - lo)) * (order**2 * magnitude).sum(axis=0)
    spacing = float(np.spacing(max(abs(lo), abs(hi))))
    return float(np.max(spacing * slope / scale))


def _sample(
    f: _SampleFunction, positions: Float[np.ndarray, " n_z"]
) -> Inexact[np.ndarray, " n_z *value"]:
    """Evaluate ``f`` at ``positions`` and check the result.

    Parameters
    ----------
    f : callable
        The vectorised sampler.
    positions : numpy.ndarray
        Sample positions, m, shape ``(n_z,)``.

    Returns
    -------
    numpy.ndarray
        Samples of shape ``(n_z, *value_shape)``, float64 or complex128.

    Raises
    ------
    TranslationError
        If the leading length is wrong, the value shape is empty, or a sample is not
        finite.
    """
    raw = np.asarray(f(positions))
    values = raw.astype(np.complex128 if np.iscomplexobj(raw) else np.float64)
    if values.ndim < 1 or values.shape[0] != positions.size:
        msg = (
            f"f must return one value per position: expected leading length "
            f"{positions.size}, got shape {values.shape}"
        )
        raise TranslationError(msg)
    if values.size == 0:
        msg = f"f returned an empty value shape {values.shape[1:]}"
        raise TranslationError(msg)
    finite = np.all(np.isfinite(values), axis=tuple(range(1, values.ndim)))
    if not np.all(finite):
        bad = float(positions[np.argmin(finite)])
        msg = f"f returned a non-finite value at z = {bad!r} m"
        raise TranslationError(msg)
    return values


def _nodes(degree: int) -> Float[np.ndarray, " n_deg"]:
    """Return the Chebyshev-Lobatto points of ``degree``, ascending on ``[-1, 1]``.

    Points of the second kind because they include both endpoints, which the continuity
    fix needs, and because they nest: the degree ``2 d`` set contains the degree ``d``
    set at even indices, bitwise, so a degree doubling costs ``d`` new evaluations.

    Parameters
    ----------
    degree : int
        Polynomial degree, ``>= 1``.

    Returns
    -------
    numpy.ndarray
        The ``degree + 1`` points, increasing from ``-1`` to ``+1``.
    """
    return -np.cos(np.pi * np.arange(degree + 1) / degree)


def _sample_positions(degree: int, lo: float, hi: float) -> Float[np.ndarray, " n_z"]:
    """Map the Chebyshev-Lobatto points onto ``[lo, hi]``, insetting the ends one ulp.

    A profile written ``z >= z_stage`` returns the post-jump value *at* ``z_stage``, so
    without the inset the interval that ends there samples the wrong side of its own
    discontinuity; the search then reads that one contaminated sample as unresolvable
    structure and bisects toward the breakpoint until it hits the width floor. The inset
    is one ulp of the interval's own scale rather than of an endpoint, so it stays a
    real perturbation when an end sits at zero.

    Parameters
    ----------
    degree : int
        Polynomial degree, ``>= 1``.
    lo, hi : float
        Ends of the interval, m.

    Returns
    -------
    numpy.ndarray
        Sample positions, m, increasing, strictly inside ``[lo, hi]`` at the ends.
    """
    positions = 0.5 * (lo + hi) + 0.5 * (hi - lo) * _nodes(degree)
    inset = float(np.spacing(max(abs(lo), abs(hi), hi - lo)))
    positions[0] = lo + inset
    positions[-1] = hi - inset
    return positions


def _refinement_positions(
    degree: int, lo: float, hi: float
) -> Float[np.ndarray, " n_z"]:
    """Return the positions the degree doubling adds, all interior.

    Parameters
    ----------
    degree : int
        Current degree; the result completes the grid of degree ``2 * degree``.
    lo, hi : float
        Ends of the interval, m.

    Returns
    -------
    numpy.ndarray
        The ``degree`` new positions, m, increasing. They are interior, so they need no
        endpoint inset.
    """
    odd = -np.cos(np.pi * np.arange(1, 2 * degree, 2) / (2 * degree))
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * odd


def _interleave(
    coarse: Inexact[np.ndarray, " n_z *value"],
    fine: Inexact[np.ndarray, " n_odd *value"],
) -> Inexact[np.ndarray, " n_fine *value"]:
    """Interleave a nested grid's samples with the ones the doubling added.

    Parameters
    ----------
    coarse : numpy.ndarray
        Samples on the degree ``d`` grid, shape ``(d + 1, *value_shape)``.
    fine : numpy.ndarray
        Samples at the odd points of the degree ``2 d`` grid, shape ``(d,
        *value_shape)``.

    Returns
    -------
    numpy.ndarray
        Samples on the degree ``2 d`` grid, shape ``(2 d + 1, *value_shape)``.
    """
    merged = np.empty(
        (2 * coarse.shape[0] - 1, *coarse.shape[1:]),
        dtype=np.result_type(coarse.dtype, fine.dtype),
    )
    merged[0::2] = coarse
    merged[1::2] = fine
    return merged


def lobatto_coefficients(
    values: Inexact[np.ndarray, " n_z *value"],
) -> Inexact[np.ndarray, " n_deg *value"]:
    """Transform samples at the Chebyshev-Lobatto points into Chebyshev coefficients.

    The discrete Chebyshev transform is the type-I DCT of the samples up to the end
    weights; taking it as the FFT of the mirror extension lets complex-valued integrands
    go through the same path as real ones. The coefficients are those
    :func:`numpy.polynomial.chebyshev.chebval` and
    :class:`PiecewiseChebyshev` evaluate, so one transform serves the adaptive
    builder here and the fixed-section expansions of
    :mod:`solprop.translate.coupled`.

    Parameters
    ----------
    values : numpy.ndarray
        Samples at :func:`lobatto_points` in increasing position, shape
        ``(n_deg, *value_shape)``.

    Returns
    -------
    numpy.ndarray
        Coefficients of ``sum_j c_j T_j``, shape ``(n_deg, *value_shape)``, real when
        the samples are real.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.chebyshev import lobatto_coefficients, lobatto_points
    >>> nodes = lobatto_points(4, -1.0, 1.0)
    >>> lobatto_coefficients(nodes**2).round(12) + 0.0
    array([0.5, 0. , 0.5, 0. , 0. ])
    """
    n_deg = values.shape[0]
    degree = n_deg - 1
    mirrored = np.concatenate([values[degree:0:-1], values[0:degree]], axis=0)
    coefficients = np.fft.ifft(mirrored, axis=0)[:n_deg].copy()
    coefficients[1:degree] *= 2.0
    if np.iscomplexobj(values):
        return coefficients
    return np.ascontiguousarray(np.real(coefficients))


def _tail_ratio(coefficients: Inexact[np.ndarray, " n_deg *value"]) -> float:
    """Return the relative coefficient tail, the worst over the value shape.

    The tail is a *sum* over the last three coefficients: two because an even or odd
    function has an exactly vanishing coefficient of the wrong parity that a shorter
    test reads as convergence, and three because the sum is a usable surrogate for the
    whole neglected series. It is measured per trailing element, so a mode with a small
    operator is not swamped by one with a large operator, with the comparison scale
    floored at a fixed fraction of the global envelope so that masked-out or negligible
    elements do not drive the refinement.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Coefficients, shape ``(n_deg, *value_shape)``.

    Returns
    -------
    float
        The tail relative to the value scale; zero for an identically zero series.
    """
    magnitude = np.abs(coefficients)
    envelope = magnitude.max(axis=0)
    global_envelope = float(envelope.max())
    if global_envelope == 0.0:
        return 0.0
    tail = magnitude[-_N_TAIL:].sum(axis=0)
    scale = np.maximum(envelope, global_envelope * ENVELOPE_FLOOR)
    return float(np.max(tail / scale))


def _trim(
    coefficients: Inexact[np.ndarray, " n_deg *value"],
) -> Inexact[np.ndarray, " n_kept *value"]:
    """Drop trailing coefficients that are rounding noise of the transform.

    The mirror-extension FFT has a noise floor of order ``log2(2 d) * eps`` times the
    value scale, so coefficients below a small multiple of it carry no information. Only
    that noise is dropped, never any of the tolerance budget, which is what collapses a
    constant operator to a single coefficient and a linear one to two — the exact,
    cheapest fast path of the cumulative operator.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Coefficients, shape ``(n_deg, *value_shape)``.

    Returns
    -------
    numpy.ndarray
        The leading coefficients that are not noise, at least one.
    """
    magnitude = np.abs(coefficients)
    envelope = magnitude.max(axis=0)
    scale = np.maximum(envelope, float(envelope.max()) * ENVELOPE_FLOOR)
    noise = (
        _TRIM_NOISE_FACTOR
        * math.log2(2 * coefficients.shape[0])
        * float(np.finfo(np.float64).eps)
        * scale
    )
    kept = coefficients.shape[0]
    dropped = np.zeros_like(envelope)
    while kept > 1:
        trial = dropped + magnitude[kept - 1]
        if np.any(trial > noise):
            break
        dropped = trial
        kept -= 1
    return coefficients[:kept]


def _assemble(
    pieces: list[_Piece], hard_kinks: Float[np.ndarray, " n_kinks"], exceeded: bool
) -> PiecewiseChebyshev:
    """Pack the resolved intervals into one rectangular representation.

    Parameters
    ----------
    pieces : list of _Piece
        The resolved intervals, in increasing z.
    hard_kinks : numpy.ndarray
        Interior positions, m, where the represented function itself jumps or kinks.
    exceeded : bool
        Whether the split ran out of its interval budget.

    Returns
    -------
    PiecewiseChebyshev
        The representation, with the low-degree intervals zero-padded.
    """
    n_deg = max(piece.coefficients.shape[0] for piece in pieces)
    dtype = np.result_type(*(piece.coefficients.dtype for piece in pieces))
    coefficients = np.zeros(
        (len(pieces), n_deg, *pieces[0].coefficients.shape[1:]), dtype=dtype
    )
    for index, piece in enumerate(pieces):
        coefficients[index, : piece.coefficients.shape[0]] = piece.coefficients
    representation = PiecewiseChebyshev(
        coefficients,
        np.asarray([pieces[0].lo, *(piece.hi for piece in pieces)], dtype=np.float64),
        np.asarray(
            [piece.coefficients.shape[0] - 1 for piece in pieces], dtype=np.intp
        ),
        np.asarray(hard_kinks, dtype=np.float64),
        exceeded,
    )
    _log_summary(representation)
    return representation


def _log_summary(representation: PiecewiseChebyshev) -> None:
    """Log the split's interval count, degrees and widths at DEBUG level.

    Parameters
    ----------
    representation : PiecewiseChebyshev
        The assembled representation.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    widths = np.diff(representation.breakpoints)
    degrees, counts = np.unique(representation.degrees, return_counts=True)
    logger.debug(
        "piecewise Chebyshev on [%.6g, %.6g] m: %d intervals, degrees %s, "
        "widths min %.3g median %.3g max %.3g m, %d hard kinks, budget exceeded %s",
        *representation.z_span,
        representation.degrees.size,
        dict(zip(degrees.tolist(), counts.tolist(), strict=True)),
        widths.min(),
        float(np.median(widths)),
        widths.max(),
        representation.hard_kinks.size,
        representation.budget_exceeded,
    )


# --------------------------------------------------------------------------------------
# The primitives a second builder needs.
#
# `solprop.translate.expansion` resolves an array-valued function of one or two *scalar
# parameters* rather than of z, on a tensor grid of Chebyshev-Lobatto points. The
# arithmetic is the same in every dimension -- the same node set, the same transform,
# the same tail criterion, the same noise trim -- so it is written once, here, and
# applied to whichever axis is being resolved. Everything below is that arithmetic
# exposed one axis at a time; nothing about z enters.


def lobatto_points(degree: int, lo: float, hi: float) -> Float[np.ndarray, " n_deg"]:
    """Return the Chebyshev-Lobatto points of ``degree`` mapped onto ``[lo, hi]``.

    Unlike the z sampler these are not inset at the ends: a parameter range has no
    discontinuity sitting on its own boundary, and its ends are values the paths
    actually reach, where the function must be evaluated as it stands.

    Parameters
    ----------
    degree : int
        Polynomial degree, ``>= 1``.
    lo, hi : float
        Ends of the range, in the parameter's own units, with ``hi > lo``.

    Returns
    -------
    numpy.ndarray
        The ``degree + 1`` points, increasing from `lo` to `hi`.

    Examples
    --------
    >>> from solprop.translate.chebyshev import lobatto_points
    >>> [round(value, 12) for value in lobatto_points(2, 0.0, 1.0).tolist()]
    [0.0, 0.5, 1.0]
    """
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * _nodes(degree)


def refinement_points(degree: int, lo: float, hi: float) -> Float[np.ndarray, " n_deg"]:
    """Return the points a degree doubling adds to :func:`lobatto_points`.

    Parameters
    ----------
    degree : int
        Current degree; the result completes the grid of degree ``2 * degree``.
    lo, hi : float
        Ends of the range, in the parameter's own units.

    Returns
    -------
    numpy.ndarray
        The ``degree`` new points, increasing, all interior.

    Examples
    --------
    >>> from solprop.translate.chebyshev import refinement_points
    >>> [round(value, 6) for value in refinement_points(1, 0.0, 1.0).tolist()]
    [0.5]
    """
    return _refinement_positions(degree, lo, hi)


def interleave_axis(
    # The two differ in length along `axis` alone, so they share no axis names.
    coarse: Inexact[np.ndarray, " *coarse_shape"],
    fine: Inexact[np.ndarray, " *fine_shape"],
    axis: int,
) -> Inexact[np.ndarray, " *doubled"]:
    """Merge a nested grid's samples with the ones a doubling added, along one axis.

    Parameters
    ----------
    coarse : numpy.ndarray
        Samples on the degree ``d`` grid, with ``d + 1`` entries along `axis`.
    fine : numpy.ndarray
        Samples at the ``d`` points :func:`refinement_points` added, same shape
        otherwise.
    axis : int
        The axis the doubling ran along.

    Returns
    -------
    numpy.ndarray
        Samples on the degree ``2 d`` grid, with ``2 d + 1`` entries along `axis`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.chebyshev import interleave_axis
    >>> coarse = np.array([[0.0, 2.0]])
    >>> interleave_axis(coarse, np.array([[1.0]]), 1).tolist()
    [[0.0, 1.0, 2.0]]
    """
    return np.moveaxis(
        _interleave(np.moveaxis(coarse, axis, 0), np.moveaxis(fine, axis, 0)), 0, axis
    )


def resolve_axis(
    values: Inexact[np.ndarray, " *shape"], axis: int, *, tol: float
) -> tuple[Inexact[np.ndarray, " *coefficients"], bool]:
    """Transform one axis to Chebyshev coefficients and judge its tail.

    The samples along the other axes are left as they are, so the acceptance test
    measures the tail **per element** against a genuine function value, as it does for
    the state in the one-dimensional build, rather than against a coefficient that may
    itself be negligible.

    Parameters
    ----------
    values : numpy.ndarray
        Samples at the Chebyshev-Lobatto points of `axis`, of any shape.
    axis : int
        The axis to transform.
    tol : float
        Target bound on the uniform error relative to the value scale, ``> 0``. The
        criterion is :func:`build`'s: twice the relative coefficient tail, the factor
        two being the Chebyshev aliasing bound.

    Returns
    -------
    numpy.ndarray
        The same array with `axis` holding Chebyshev coefficients.
    bool
        Whether that axis is resolved to `tol`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.chebyshev import lobatto_points, resolve_axis
    >>> x = lobatto_points(8, -1.0, 1.0)
    >>> coefficients, resolved = resolve_axis(2.0 * x**2 - 1.0, 0, tol=1e-12)
    >>> resolved, [abs(round(value, 12)) for value in coefficients[:3].tolist()]
    (True, [0.0, 0.0, 1.0])
    """
    moved = np.moveaxis(values, axis, 0)
    coefficients = lobatto_coefficients(moved)
    return np.moveaxis(coefficients, 0, axis), 2.0 * _tail_ratio(coefficients) <= tol


def trim_axis(
    coefficients: Inexact[np.ndarray, " *shape"], axis: int
) -> Inexact[np.ndarray, " *trimmed"]:
    """Drop the trailing coefficients of one axis that are transform noise.

    Parameters
    ----------
    coefficients : numpy.ndarray
        Chebyshev coefficients along `axis`.
    axis : int
        The axis to trim.

    Returns
    -------
    numpy.ndarray
        The same array with `axis` shortened to the coefficients that carry signal; at
        least one is always kept.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.chebyshev import trim_axis
    >>> trim_axis(np.array([[1.0, 0.0, 0.0]]), 1).shape
    (1, 1)
    """
    return np.moveaxis(_trim(np.moveaxis(coefficients, axis, 0)), 0, axis)
