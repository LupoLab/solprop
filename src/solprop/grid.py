# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time and frequency grids for field-resolved and envelope propagation.

Two grids implement one protocol. :class:`FieldGrid` carries a real field ``E(t)`` and
stores its one-sided spectrum; :class:`EnvelopeGrid` carries a complex envelope ``A(t)``
and stores the full spectrum in FFT order. Both carry an oversampled *fine* grid on
which nonlinear responses are evaluated, the masks that say which bins are physical, and
the Planck-taper collars that absorb what leaves the window.

The construction arithmetic, the invariants, the layout rules and the sign convention
are fixed by ``docs/dev/conventions.md`` §2-§3, and the manual page
``docs/guide/setup/grids.md`` derives them. Four hazards are guarded against, each noted
where
it is handled: overlapping coarse spectral bins when the transform size is not a power
of two, a degenerate temporal collar when the realised window equals the requested one,
an envelope fine spacing that ignores the low-frequency edge, and an envelope crop that
is derived from the upper edge alone and then applied symmetrically.
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final, Literal

import numpy as np
from jaxtyping import Bool, Complex, Float

from solprop import fourier
from solprop._display import Displayed, engineering, line
from solprop.constants import C
from solprop.exceptions import GridError
from solprop.fourier import GridKind
from solprop.hankel import QDHT

__all__ = [
    "DEFAULT_COLLAR_FRACTION",
    "FFT_SIZES",
    "Collar",
    "CollarPlacement",
    "EnvelopeGrid",
    "FieldGrid",
    "FreeGrid",
    "LineGrid",
    "RadialGrid",
    "TimeGrid",
    "fft_cost",
    "planck_taper",
    "planck_taper_derivative",
    "planck_taper_derivative_error",
]

logger = logging.getLogger(__name__)

# The frequency window is widened past the requested maximum so that the spectral collar
# has somewhere to fall to zero.
WINDOW_FACTOR = 1.1

# Relative tolerance for the grid invariants. They are exact in real arithmetic and are
# built from products and quotients of a handful of exactly representable integers, so
# only a few units in the last place separate them from equality.
INVARIANT_RTOL = 1e-12


def planck_taper(
    x: Float[np.ndarray, " n"],
    zero_low: float,
    one_low: float,
    one_high: float,
    zero_high: float,
) -> Float[np.ndarray, " n"]:
    """Evaluate a Planck taper that rises then falls between four breakpoints.

    The window is 0 at or below ``zero_low``, rises smoothly to 1 at ``one_low``, stays
    1 up to ``one_high``, and falls back to 0 at ``zero_high``. Every derivative is
    continuous, which is what distinguishes a Planck taper from the Tukey window that
    :mod:`scipy.signal.windows` provides, and why it is written out here.

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate the window, shape ``(n,)``, in the same units as the
        breakpoints: rad/s for a spectral collar, s for a temporal one.
    zero_low : float
        Lower point at which the window reaches 0.
    one_low : float
        Lower point at which the window reaches 1. Must exceed ``zero_low``.
    one_high : float
        Upper point at which the window leaves 1. Must be at least ``one_low``.
    zero_high : float
        Upper point at which the window returns to 0. Must exceed ``one_high``.

    Returns
    -------
    numpy.ndarray
        Window values in ``[0, 1]``, shape ``(n,)``.

    Raises
    ------
    GridError
        If the four breakpoints are not ordered, so that a taper edge would have zero
        width and the window would step rather than taper.

    References
    ----------
    D. J. A. McKechan, C. Robinson and B. S. Sathyaprakash, "A tapering window for
    time-domain templates and simulated signals in the detection of gravitational waves
    from coalescing compact binaries", Class. Quantum Grav. 27, 084020 (2010), eq. (7).

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import planck_taper
    >>> window = planck_taper(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), 0.0, 1.0, 3.0, 4.0)
    >>> window.tolist()
    [0.0, 1.0, 1.0, 1.0, 0.0]
    """
    if not zero_low < one_low <= one_high < zero_high:
        msg = (
            "planck_taper breakpoints must satisfy "
            "zero_low < one_low <= one_high < zero_high, got "
            f"({zero_low!r}, {one_low!r}, {one_high!r}, {zero_high!r})"
        )
        raise GridError(msg)
    values = np.asarray(x, dtype=float)
    window = np.zeros_like(values)
    window[(values >= one_low) & (values <= one_high)] = 1.0
    # The taper exponent diverges at every breakpoint, so it is evaluated only on the
    # points strictly inside the rising and falling intervals. Evaluating it everywhere
    # and masking afterwards would raise a divide-by-zero warning, which is an error in
    # the test session.
    rising = (values > zero_low) & (values < one_low)
    window[rising] = _taper_edge(values[rising], zero_low, one_low)
    falling = (values > one_high) & (values < zero_high)
    window[falling] = _taper_edge(values[falling], zero_high, one_high)
    return window


def _taper_edge(
    x: Float[np.ndarray, " n"], outer: float, inner: float
) -> Float[np.ndarray, " n"]:
    """Evaluate one Planck taper edge, strictly between the two breakpoints.

    Parameters
    ----------
    x : numpy.ndarray
        Points strictly between ``outer`` and ``inner``, shape ``(n,)``.
    outer : float
        Breakpoint at which the edge reaches 0.
    inner : float
        Breakpoint at which the edge reaches 1.

    Returns
    -------
    numpy.ndarray
        Edge values in ``(0, 1)``, shape ``(n,)``.
    """
    width = inner - outer
    exponent = width / (x - outer) + width / (x - inner)
    # exp(709) is the largest finite double, and 1/(1 + exp(700)) is already exactly 0,
    # so clipping here costs no accuracy and avoids an overflow warning.
    return 1.0 / (1.0 + np.exp(np.clip(exponent, -700.0, 700.0)))


# Below this distance from a breakpoint, in units of the edge width, the derivative
# returns its asymptotic zero without forming a reciprocal (SEP 26 section 4.2). There
# |zeta| >= 740 - 1/(1 - 1/740) > 738, so 1/(4 cosh^2(zeta/2)) <= 4 exp(-738) while
# the rational factor 1/s^2 + 1/(1 - s)^2 is at most 2 * 740^2: the analytic value is
# below 1e-300 / |width|, which bounds what returning zero discards.
_DERIVATIVE_CUTOFF = 1.0 / 740.0

# The largest |g'| of an edge is 2/|width|, at its midpoint (h(1/2) = 2); this keeps
# that peak finite.
_DERIVATIVE_PEAK = 2.0

# The discarded analytic value in the cut-off region, in units of 1/|width|.
_CUTOFF_RESIDUE = 1e-300

# First-order relative rounding bound of the evaluated edge derivative, as
# (_DERIVATIVE_ROUNDING + _EXPONENT_AMPLIFICATION * |zeta|) * eps, for the operation
# sequence of _taper_edge_derivative. With u = eps/2: s and r carry 3u each (the
# subtraction, the width and the division); their reciprocals 4u. Then
# zeta = 1/s - 1/r has an absolute error of at most 4u (1/s + 1/r) + u |zeta|, and
# 1/s + 1/r = |zeta| + 2 min(1/s, 1/r) <= |zeta| + 4, so 5u |zeta| + 16u. The
# exponential adds 2u (one correctly rounded libm result and its argument's rounding),
# e/(1 + e)^2 is at most once as sensitive to e as e is to zeta, and adds three
# roundings; 1/s^2 + 1/r^2 carries 10u; the product and the division by the width 3u.
# In all 5u |zeta| + 34u = (2.5 |zeta| + 17) eps, rounded up here to (3 |zeta| + 18)
# eps for the second-order terms. SEP 26 section 4.5 wrote (8 + |zeta|) eps from a
# shorter count; this derivation replaces it (phase 1 report).
_DERIVATIVE_ROUNDING = 18.0
_EXPONENT_AMPLIFICATION = 3.0


def planck_taper_derivative(
    x: Float[np.ndarray, " n"],
    zero_low: float,
    one_low: float,
    one_high: float,
    zero_high: float,
) -> Float[np.ndarray, " n"]:
    r"""Evaluate the analytic derivative of :func:`planck_taper` with respect to ``x``.

    On an edge with outer breakpoint ``a`` and inner ``b``, ``width = b - a`` (negative
    on the falling edge), ``s = (x - a) / width`` and ``r = (b - x) / width``, both in
    ``(0, 1)``. The gate is ``g = 1 / (1 + exp(zeta))`` with ``zeta = 1/s - 1/r``, and

    .. math::

        g'(x) = \frac{h(s)}{\mathrm{width}},\qquad
        h = \frac{1}{4\cosh^2(\zeta/2)}\left(\frac{1}{s^2}+\frac{1}{r^2}\right) > 0,

    so ``g'`` is positive on the rising edge, negative on the falling edge, integrates
    to ``+1`` and ``-1`` over them, and is zero outside the tapers. The factor
    ``1/(4 cosh^2(zeta/2))`` is evaluated as ``e/(1 + e)^2`` with ``e = exp(-|zeta|)``,
    which never overflows. ``s`` and ``r`` are formed separately, so neither is
    obtained by cancellation. Where ``min(s, r) <= 1/740`` the asymptotic zero is
    returned and no reciprocal is formed: there the analytic value is below
    ``1e-300 / |width|``. Points at or beyond the breakpoints return exactly zero.
    The evaluation never forms an infinite intermediate (SEP 26 section 4.2).

    Parameters
    ----------
    x : numpy.ndarray
        Points at which to evaluate the derivative, shape ``(n,)``, in the units of
        the breakpoints.
    zero_low, one_low, one_high, zero_high : float
        Breakpoints of :func:`planck_taper`, in the same units.

    Returns
    -------
    numpy.ndarray
        ``dg/dx`` in reciprocal units of ``x``, shape ``(n,)``.

    Raises
    ------
    GridError
        If the breakpoints are not ordered, or an edge is so narrow that its peak
        derivative ``2 / width`` would overflow.

    See Also
    --------
    planck_taper_derivative_error : The rounding bound of this evaluation.

    References
    ----------
    D. J. A. McKechan, C. Robinson and B. S. Sathyaprakash, Class. Quantum Grav. 27,
    084020 (2010), eq. (7), differentiated.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import planck_taper_derivative
    >>> slope = planck_taper_derivative(np.array([0.5, 2.0, 3.5]), 0.0, 1.0, 3.0, 4.0)
    >>> slope.tolist()
    [2.0, 0.0, -2.0]
    """
    values = np.asarray(x, dtype=float)
    derivative = np.zeros_like(values)
    for outer, inner, inside in _derivative_edges(
        values, zero_low, one_low, one_high, zero_high
    ):
        derivative[inside], _ = _taper_edge_derivative(values[inside], outer, inner)
    return derivative


def planck_taper_derivative_error(
    x: Float[np.ndarray, " n"],
    zero_low: float,
    one_low: float,
    one_high: float,
    zero_high: float,
) -> Float[np.ndarray, " n"]:
    """Bound the absolute error of :func:`planck_taper_derivative` at each point.

    The bound is ``(18 + 3 |zeta|) eps |g'|`` where the derivative is evaluated, a
    first-order count of the operations of that evaluation (its comments derive it),
    and ``1e-300 / |width|`` in the cut-off region next to a breakpoint, where the
    returned zero discards at most that much. It is zero outside the tapers, where the
    derivative is exactly zero.

    Parameters
    ----------
    x : numpy.ndarray
        Points, shape ``(n,)``, in the units of the breakpoints.
    zero_low, one_low, one_high, zero_high : float
        Breakpoints of :func:`planck_taper`.

    Returns
    -------
    numpy.ndarray
        Nonnegative bound in reciprocal units of ``x``, shape ``(n,)``.

    Raises
    ------
    GridError
        As :func:`planck_taper_derivative`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import planck_taper_derivative_error
    >>> bound = planck_taper_derivative_error(np.array([0.5, 2.0]), 0.0, 1.0, 3.0, 4.0)
    >>> bool(0.0 < bound[0] < 1e-13), float(bound[1])
    (True, 0.0)
    """
    values = np.asarray(x, dtype=float)
    bound = np.zeros_like(values)
    epsilon = float(np.finfo(float).eps)
    for outer, inner, inside in _derivative_edges(
        values, zero_low, one_low, one_high, zero_high
    ):
        slope, exponent = _taper_edge_derivative(values[inside], outer, inner)
        evaluated = (
            _DERIVATIVE_ROUNDING + _EXPONENT_AMPLIFICATION * np.abs(exponent)
        ) * epsilon
        residue = _CUTOFF_RESIDUE / abs(inner - outer)
        bound[inside] = np.where(np.isnan(exponent), residue, evaluated * np.abs(slope))
    return bound


def _derivative_edges(
    values: Float[np.ndarray, " n"],
    zero_low: float,
    one_low: float,
    one_high: float,
    zero_high: float,
) -> tuple[tuple[float, float, Bool[np.ndarray, " n"]], ...]:
    """Validate the breakpoints and select the points strictly inside each edge.

    Parameters
    ----------
    values : numpy.ndarray
        Points, shape ``(n,)``.
    zero_low, one_low, one_high, zero_high : float
        Breakpoints of :func:`planck_taper`.

    Returns
    -------
    tuple
        ``(outer, inner, mask)`` for the rising and the falling edge.
    """
    if not zero_low < one_low <= one_high < zero_high:
        msg = (
            "planck_taper_derivative breakpoints must satisfy "
            "zero_low < one_low <= one_high < zero_high, got "
            f"({zero_low!r}, {one_low!r}, {one_high!r}, {zero_high!r})"
        )
        raise GridError(msg)
    largest = float(np.finfo(float).max)
    for width in (one_low - zero_low, zero_high - one_high):
        if not width * largest > _DERIVATIVE_PEAK:
            msg = f"a Planck taper edge of width {width!r} has an unrepresentable slope"
            raise GridError(msg)
    return (
        (zero_low, one_low, (values > zero_low) & (values < one_low)),
        (zero_high, one_high, (values > one_high) & (values < zero_high)),
    )


def _taper_edge_derivative(
    x: Float[np.ndarray, " n"], outer: float, inner: float
) -> tuple[Float[np.ndarray, " n"], Float[np.ndarray, " n"]]:
    """Evaluate one edge's derivative strictly between its two breakpoints.

    Parameters
    ----------
    x : numpy.ndarray
        Points strictly between ``outer`` and ``inner``, shape ``(n,)``.
    outer : float
        Breakpoint at which the edge reaches 0.
    inner : float
        Breakpoint at which the edge reaches 1.

    Returns
    -------
    slope : numpy.ndarray
        ``dg/dx``, shape ``(n,)``; the asymptotic zero in the cut-off region.
    exponent : numpy.ndarray
        ``zeta`` where the slope is evaluated, NaN in the cut-off region, where no
        reciprocal is formed.
    """
    width = inner - outer
    # Both distances are formed from x directly. r = 1 - s would cancel next to the
    # inner breakpoint and amplify the rounding of s by 1/r.
    near = (x - outer) / width
    far = (inner - x) / width
    slope = np.zeros_like(x)
    exponent = np.full_like(x, np.nan)
    kept = np.minimum(near, far) > _DERIVATIVE_CUTOFF
    near, far = near[kept], far[kept]
    zeta = 1.0 / near - 1.0 / far
    # 1/(4 cosh^2(zeta/2)) = e/(1 + e)^2 with e = exp(-|zeta|) <= 1: no overflow at
    # |zeta| up to 740, where cosh^2 itself would exceed the largest double.
    decay = np.exp(-np.abs(zeta))
    slope[kept] = decay / (1.0 + decay) ** 2 * (1.0 / near**2 + 1.0 / far**2) / width
    exponent[kept] = zeta
    return slope, exponent


@dataclass(frozen=True, eq=False)
class RadialGrid(Displayed):
    """Construct an axisymmetric spatial grid with a declared outer collar.

    Attributes
    ----------
    radius : float
        Outer Dirichlet boundary in m, finite and positive.
    n_r : int
        Number of interior radial and spectral samples, at least 2.
    window_factor : float
        Fraction of the outer radius occupied by the absorbing collar, in [0, 1).
        Zero disables the spatial collar; the default 0.1 uses its outer tenth.
    hankel : solprop.hankel.QDHT
        The raw transform, its physical scalings and integration weights.
    window : numpy.ndarray
        Planck window on the radial samples, shape ``(n_r,)``, dimensionless.

    Raises
    ------
    GridError
        If a dimension or the collar fraction is invalid.

    Examples
    --------
    >>> from solprop.grid import RadialGrid
    >>> grid = RadialGrid(1e-3, 64)
    >>> grid.r.shape, grid.k_r.shape, bool(grid.r[0] > 0)
    ((64,), (64,), True)
    """

    radius: float
    n_r: int
    window_factor: float = 0.1
    hankel: QDHT = field(init=False, repr=False)
    window: Float[np.ndarray, " n_r"] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the transform and an immutable collar profile."""
        if not math.isfinite(self.window_factor) or not 0 <= self.window_factor < 1:
            msg = "window_factor must be finite and lie in [0, 1)"
            raise GridError(msg)
        hankel = QDHT(self.radius, self.n_r)
        window = (
            np.ones(self.n_r)
            if self.window_factor == 0
            else planck_taper(
                hankel.r,
                -self.radius,
                0.0,
                self.radius * (1 - self.window_factor),
                self.radius,
            )
        )
        window.setflags(write=False)
        object.__setattr__(self, "hankel", hankel)
        object.__setattr__(self, "window", window)
        self.validate()

    @property
    def r(self) -> Float[np.ndarray, " n_r"]:
        """Return the positive interior radial samples.

        Returns
        -------
        numpy.ndarray
            Radii in m, shape ``(n_r,)``; zero and the outer boundary are excluded.
        """
        return self.hankel.r

    @property
    def k_r(self) -> Float[np.ndarray, " n_r"]:
        """Return the transverse spectral samples.

        Returns
        -------
        numpy.ndarray
            Positive angular frequencies in rad/m, shape ``(n_r,)``.
        """
        return self.hankel.k_r

    @property
    def radial_weights(self) -> Float[np.ndarray, " n_r"]:
        """Return the quadrature weights of the transverse area integral.

        Returns
        -------
        numpy.ndarray
            Weights of ``2*pi*r*dr``, in m^2, shape ``(n_r,)``.
        """
        return self.hankel.radial_weights

    @property
    def spectral_weights(self) -> Float[np.ndarray, " n_r"]:
        """Return the quadrature weights of the transverse spectral integral.

        Returns
        -------
        numpy.ndarray
            Weights of ``2*pi*k_r*dk_r``, in m^-2, shape ``(n_r,)``.
        """
        return self.hankel.spectral_weights

    @property
    def n_columns(self) -> int:
        """Return the number of transverse spectral columns.

        Returns
        -------
        int
            The radial sample count.
        """
        return self.n_r

    def validate(self) -> None:
        """Check the radial axes, weights and spatial window.

        Raises
        ------
        GridError
            If an axis is unordered or an array is mutable, nonfinite or inconsistent.
        """
        for name in ("r", "k_r", "radial_weights", "spectral_weights", "window"):
            values = getattr(self, name)
            if (
                values.shape != (self.n_r,)
                or values.dtype != np.float64
                or values.flags.writeable
                or not values.flags.c_contiguous
                or not np.all(np.isfinite(values))
            ):
                msg = (
                    f"RadialGrid.{name} must be finite, read-only float64 "
                    "of shape (n_r,)"
                )
                raise GridError(msg)
            if name != "window" and np.any(values <= 0):
                msg = f"RadialGrid.{name} must be strictly positive"
                raise GridError(msg)
        if np.any(np.diff(self.r) <= 0) or np.any(np.diff(self.k_r) <= 0):
            msg = "RadialGrid axes must be strictly increasing"
            raise GridError(msg)
        if self.r[-1] >= self.radius or np.any((self.window < 0) | (self.window > 1)):
            msg = "RadialGrid samples must lie inside radius and window in [0, 1]"
            raise GridError(msg)

    def __repr__(self) -> str:
        """Return a one-line description: the aperture and the sample count.

        Returns
        -------
        str
            For example ``RadialGrid(radius=2 mm, n_r=256)``.
        """
        return line(
            "RadialGrid", [("radius", engineering(self.radius, "m")), ("n_r", self.n_r)]
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "RadialGrid: a radial grid on the zeros of J0", [
            ("aperture radius", engineering(self.radius, "m")),
            ("samples", self.n_r),
            (
                "innermost, outermost",
                f"{engineering(float(self.r[0]), 'm')}, "
                f"{engineering(float(self.r[-1]), 'm')}",
            ),
            ("largest k_r", f"{float(self.k_r[-1]):.4g} rad/m"),
            ("window factor", f"{self.window_factor:g}"),
        ]


def _periodic_axis(
    owner: str, radius: float, count: int, window_factor: float
) -> tuple[
    Float[np.ndarray, " n"], Float[np.ndarray, " n"], Float[np.ndarray, " n"], float
]:
    """Build one centred periodic Cartesian axis, its FFT frequencies and its collar.

    Parameters
    ----------
    owner : str
        The grid class, named in the error messages.
    radius : float
        Requested physical half-width in m, finite and positive.
    count : int
        Number of samples, at least 1 (validated by the caller).
    window_factor : float
        Added collar width divided by the half-width, in [0, 1).

    Returns
    -------
    tuple
        The centred axis in m, the angular frequencies in rad/m in unshifted FFT
        order, the Planck collar, each of shape ``(count,)``, and the spacing in m.

    Raises
    ------
    GridError
        If the extent, the spacing or the wave-number squares are not representable.
    """
    outer = radius * (1 + window_factor)
    spacing = 2 * (outer / count)
    if not math.isfinite(outer) or not math.isfinite(spacing) or spacing <= 0:
        msg = f"{owner} extent or spacing is not representable"
        raise GridError(msg)
    axis = (np.arange(count) - count / 2) * spacing
    if spacing < math.pi / math.sqrt(np.finfo(float).max / 2):
        msg = f"{owner} transverse wave-number squares are not representable"
        raise GridError(msg)
    frequencies = np.fft.fftfreq(count) * (2 * np.pi / spacing)
    window = (
        np.ones(count)
        if window_factor == 0
        else planck_taper(axis, -outer, -radius, radius, outer)
    )
    return axis, frequencies, window, spacing


@dataclass(frozen=True, eq=False)
class FreeGrid(Displayed):
    """Construct a periodic Cartesian grid with x fastest in flattened columns.

    Attributes
    ----------
    radius_x, radius_y : float
        Requested physical half-widths in m, finite and positive.
    n_x, n_y : int
        Independent sample counts, each an integer of at least 2.
    window_factor : float
        Added collar width divided by each requested half-width, in [0, 1).
        The periodic half-width is ``radius * (1 + window_factor)``. Zero
        disables the collar. The default is 0.1.
    x, y : numpy.ndarray
        Centred physical axes in m, shapes ``(n_x,)`` and ``(n_y,)``. Odd
        sample counts place zero halfway between samples.
    k_x, k_y : numpy.ndarray
        Angular transverse frequencies in rad/m, in unshifted FFT order.
    r : numpy.ndarray
        Distance from the origin in m, shape ``(n_nodes,)``, x fastest.
    spatial_weights, spectral_weights : numpy.ndarray
        Area and raw-DFT Parseval weights in m^2, shapes ``(n_nodes,)``.
    window : numpy.ndarray
        Product of the two Planck collars, shape ``(n_nodes,)``.
    on_axis : numpy.ndarray
        Complex reconstruction weights at x=y=0, shape ``(n_nodes,)``.
    kperp_sq : numpy.ndarray
        Transverse wave-number squares in m^-2, shape ``(n_nodes,)``.

    Raises
    ------
    GridError
        If dimensions, the collar, or derived floating-point arrays are invalid.

    Notes
    -----
    Public axes are x before y. Column ``ix + n_x*iy`` represents ``(x[ix],
    y[iy])``. A C-order reshape for an FFT therefore has shape ``(n_y, n_x)``.
    Forward transforms are unscaled; inverse transforms divide by ``n_x*n_y``.
    """

    radius_x: float
    n_x: int
    radius_y: float
    n_y: int
    window_factor: float = 0.1
    x: Float[np.ndarray, " n_x"] = field(init=False, repr=False)
    y: Float[np.ndarray, " n_y"] = field(init=False, repr=False)
    k_x: Float[np.ndarray, " n_x"] = field(init=False, repr=False)
    k_y: Float[np.ndarray, " n_y"] = field(init=False, repr=False)
    r: Float[np.ndarray, " n_nodes"] = field(init=False, repr=False)
    spatial_weights: Float[np.ndarray, " n_nodes"] = field(init=False, repr=False)
    spectral_weights: Float[np.ndarray, " n_nodes"] = field(init=False, repr=False)
    window: Float[np.ndarray, " n_nodes"] = field(init=False, repr=False)
    on_axis: Complex[np.ndarray, " n_nodes"] = field(init=False, repr=False)
    kperp_sq: Float[np.ndarray, " n_nodes"] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Derive the periodic axes and their discrete integration measure."""
        for name in ("radius_x", "radius_y"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                msg = f"FreeGrid.{name} must be finite and positive"
                raise GridError(msg)
        for name in ("n_x", "n_y"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                msg = f"FreeGrid.{name} must be an integer of at least 2"
                raise GridError(msg)
        if not math.isfinite(self.window_factor) or not 0 <= self.window_factor < 1:
            msg = "FreeGrid.window_factor must be finite and lie in [0, 1)"
            raise GridError(msg)
        x, k_x, window_x, spacing_x = _periodic_axis(
            "FreeGrid", self.radius_x, self.n_x, self.window_factor
        )
        y, k_y, window_y, spacing_y = _periodic_axis(
            "FreeGrid", self.radius_y, self.n_y, self.window_factor
        )
        self._adopt(x, y, (k_x, k_y), (window_x, window_y), spacing_x * spacing_y)

    def _adopt(
        self,
        x: Float[np.ndarray, " n_x"],
        y: Float[np.ndarray, " n_y"],
        frequencies: tuple[Float[np.ndarray, " n_x"], Float[np.ndarray, " n_y"]],
        windows: tuple[Float[np.ndarray, " n_x"], Float[np.ndarray, " n_y"]],
        area: float,
    ) -> None:
        """Derive the flattened columns from the two axes and freeze them.

        Parameters
        ----------
        x, y : numpy.ndarray
            Centred physical axes in m.
        frequencies : tuple of numpy.ndarray
            ``(k_x, k_y)`` in rad/m, in unshifted FFT order.
        windows : tuple of numpy.ndarray
            The x and y Planck collars.
        area : float
            The measure of one column: ``dx*dy`` in m^2, or ``dx`` times the unit
            y measure for a line.

        Raises
        ------
        GridError
            If the measure or a derived array is not representable.
        """
        k_x, k_y = frequencies
        window_x, window_y = windows
        if not math.isfinite(area) or area / self.n_columns <= 0:
            msg = f"{type(self).__name__} area weights are not representable"
            raise GridError(msg)
        values = {
            "x": x,
            "y": y,
            "k_x": k_x,
            "k_y": k_y,
            "r": np.hypot(x[None, :], y[:, None]).ravel(),
            "spatial_weights": np.full(self.n_columns, area),
            "spectral_weights": np.full(self.n_columns, area / self.n_columns),
            "window": (window_y[:, None] * window_x[None, :]).ravel(),
            "on_axis": (
                np.exp(-1j * (k_y[:, None] * y[0] + k_x[None, :] * x[0]))
                / self.n_columns
            ).ravel(),
            "kperp_sq": (k_y[:, None] ** 2 + k_x[None, :] ** 2).ravel(),
        }
        for name, array in values.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        self.validate()

    @property
    def n_columns(self) -> int:
        """Return the number of flattened transverse spectral columns.

        Returns
        -------
        int
            ``n_x*n_y``.
        """
        return self.n_x * self.n_y

    def validate(self) -> None:
        """Validate the immutable derived axes and area measures.

        Raises
        ------
        GridError
            If a derived array has invalid shape, dtype, finiteness or mutability.
        """
        for name in (
            "x",
            "y",
            "k_x",
            "k_y",
            "r",
            "spatial_weights",
            "spectral_weights",
            "window",
            "on_axis",
            "kperp_sq",
        ):
            values = getattr(self, name)
            size = (
                self.n_x
                if name in ("x", "k_x")
                else self.n_y
                if name in ("y", "k_y")
                else self.n_columns
            )
            dtype = np.complex128 if name == "on_axis" else np.float64
            if (
                values.shape != (size,)
                or values.dtype != dtype
                or values.flags.writeable
                or not values.flags.c_contiguous
                or not np.all(np.isfinite(values))
            ):
                msg = (
                    f"{type(self).__name__}.{name} must be finite and read-only "
                    "with its declared shape and dtype"
                )
                raise GridError(msg)
        if (
            np.any(np.diff(self.x) <= 0)
            or np.any(np.diff(self.y) <= 0)
            or np.any(self.spatial_weights <= 0)
            or np.any(self.spectral_weights <= 0)
            or np.any(self.kperp_sq < 0)
            or np.any((self.window < 0) | (self.window > 1))
        ):
            msg = (
                f"{type(self).__name__} axes, area weights, frequencies or window "
                "are invalid"
            )
            raise GridError(msg)

    def __repr__(self) -> str:
        """Return a one-line description: the two half-widths and sample counts.

        Returns
        -------
        str
            For example ``FreeGrid(x=2 mm x 128, y=2 mm x 128)``.
        """
        return line(
            "FreeGrid",
            [
                ("x", f"{engineering(self.radius_x, 'm')} x {self.n_x}"),
                ("y", f"{engineering(self.radius_y, 'm')} x {self.n_y}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "FreeGrid: a Cartesian transverse grid", [
            (
                "half-width in x",
                f"{engineering(self.radius_x, 'm')}, {self.n_x} samples",
            ),
            (
                "half-width in y",
                f"{engineering(self.radius_y, 'm')}, {self.n_y} samples",
            ),
            (
                "spacing",
                f"{engineering(float(self.x[1] - self.x[0]), 'm')}, "
                f"{engineering(float(self.y[1] - self.y[0]), 'm')}",
            ),
            ("window factor", f"{self.window_factor:g}"),
        ]


@dataclass(frozen=True, eq=False)
class LineGrid(FreeGrid):
    """Construct a periodic line grid in x for a beam uniform in y, per metre of y.

    The line is the x axis of a :class:`FreeGrid` -- the same centred samples, the
    same unshifted FFT frequencies and the same Planck collar -- with y reduced to
    one sample at ``y = 0`` carrying ``k_y = 0``, no collar and a unit measure. It
    therefore runs as the ``"free"`` transform with ``n_y = 1``, and every integral
    over the transverse plane becomes an integral over x per metre of y: energy in
    J/m and power in W/m, while intensity and fluence stay local, in W/m^2 and
    J/m^2 (:attr:`transverse_measure`). This is x-only free space;
    ``docs/guide/free-space/model.md`` derives how its physics
    differs from the plane's.

    Attributes
    ----------
    radius_x : float
        Requested physical half-width in m, finite and positive.
    n_x : int
        Number of samples in x, an integer of at least 2.
    window_factor : float
        Added collar width divided by the half-width, in [0, 1). The periodic
        half-width is ``radius_x * (1 + window_factor)``; zero disables the collar.
        The default is 0.1.
    radius_y : float
        ``math.inf``: the field is uniform and unbounded in y.
    n_y : int
        1.
    x, y : numpy.ndarray
        The centred x axis in m, shape ``(n_x,)``, and ``y = [0.0]``.
    k_x, k_y : numpy.ndarray
        Angular frequencies in rad/m in unshifted FFT order, and ``k_y = [0.0]``.
    r : numpy.ndarray
        ``|x|`` in m, shape ``(n_x,)``.
    spatial_weights, spectral_weights : numpy.ndarray
        Line and raw-DFT Parseval weights in m (the spacing times one metre of y
        per metre of y), shapes ``(n_x,)``.
    window : numpy.ndarray
        The x collar, shape ``(n_x,)``.
    on_axis : numpy.ndarray
        Complex reconstruction weights at ``x = 0``, shape ``(n_x,)``.
    kperp_sq : numpy.ndarray
        ``k_x**2`` in m^-2, shape ``(n_x,)``.

    Raises
    ------
    GridError
        If the half-width, the sample count or the collar is invalid.

    Notes
    -----
    It is a :class:`FreeGrid` so that everything that runs a Cartesian grid runs
    it unchanged, the backends included; :class:`FreeGrid` itself keeps two samples
    per axis. Refusals that are specific to the line, such as
    :func:`solprop.spatial.critical_power`, test for this class by name.

    Examples
    --------
    >>> from solprop.grid import LineGrid
    >>> line = LineGrid(1e-3, 64)
    >>> line.n_columns, line.y.tolist(), line.transverse_measure
    (64, [0.0], 'per_unit_width')
    >>> f"{line.spatial_weights[0]:.4e} m"
    '3.4375e-05 m'
    """

    radius_y: float = field(init=False, repr=False)
    n_y: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Derive the periodic x axis and the unit y measure."""
        if not math.isfinite(self.radius_x) or self.radius_x <= 0:
            msg = "LineGrid.radius_x must be finite and positive"
            raise GridError(msg)
        if isinstance(self.n_x, bool) or not isinstance(self.n_x, int) or self.n_x < 2:
            msg = "LineGrid.n_x must be an integer of at least 2"
            raise GridError(msg)
        if not math.isfinite(self.window_factor) or not 0 <= self.window_factor < 1:
            msg = "LineGrid.window_factor must be finite and lie in [0, 1)"
            raise GridError(msg)
        object.__setattr__(self, "radius_y", math.inf)
        object.__setattr__(self, "n_y", 1)
        x, k_x, window_x, spacing_x = _periodic_axis(
            "LineGrid", self.radius_x, self.n_x, self.window_factor
        )
        # One y sample of unit measure: every transverse integral is then per
        # metre of y, and k_y = 0 makes the operator and the collar the x axis's.
        self._adopt(
            x, np.zeros(1), (k_x, np.zeros(1)), (window_x, np.ones(1)), spacing_x
        )

    @property
    def transverse_measure(self) -> Literal["per_unit_width"]:
        """Return what the grid's transverse integrals measure.

        Returns
        -------
        str
            ``"per_unit_width"``: energies are J/m and powers W/m, per metre of y
            (:data:`solprop.modes.protocol.TransverseMeasure`).
        """
        return "per_unit_width"

    def __repr__(self) -> str:
        """Return a one-line description: the half-width and the sample count.

        Returns
        -------
        str
            For example ``LineGrid(x=2 mm x 128)``.
        """
        return line(
            "LineGrid", [("x", f"{engineering(self.radius_x, 'm')} x {self.n_x}")]
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "LineGrid: an x line, uniform in y (per metre of y)", [
            (
                "half-width in x",
                f"{engineering(self.radius_x, 'm')}, {self.n_x} samples",
            ),
            ("spacing", engineering(float(self.x[1] - self.x[0]), "m")),
            ("window factor", f"{self.window_factor:g}"),
            ("measure", "per metre of y: energy J/m, power W/m"),
        ]


@dataclass(frozen=True)
class Collar:
    """Geometry of one Planck-taper collar, as four breakpoints.

    A collar is where the propagation is absorbed rather than modelled: the spectral
    collar outside the requested wavelength limits, and the temporal collar outside the
    requested time window. :mod:`solprop.boundaries` turns these profiles into absorber
    rates; here they define the profile itself.

    Attributes
    ----------
    zero_low : float
        Lower point at which the profile reaches 0.
    one_low : float
        Lower point at which the profile reaches 1.
    one_high : float
        Upper point at which the profile leaves 1.
    zero_high : float
        Upper point at which the profile returns to 0.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import Collar
    >>> collar = Collar(0.0, 1.0, 3.0, 4.0)
    >>> collar.window(np.array([0.5, 2.0])).round(3).tolist()
    [0.5, 1.0]
    """

    zero_low: float
    one_low: float
    one_high: float
    zero_high: float

    @property
    def width_low(self) -> float:
        """Return the width of the rising edge, in the collar's own units.

        Returns
        -------
        float
            The distance from ``zero_low`` to ``one_low``.
        """
        return self.one_low - self.zero_low

    @property
    def width_high(self) -> float:
        """Return the width of the falling edge, in the collar's own units.

        Returns
        -------
        float
            The distance from ``one_high`` to ``zero_high``.
        """
        return self.zero_high - self.one_high

    def window(self, x: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Evaluate the collar's Planck taper at the given points.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate the profile, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            Window values in ``[0, 1]``, shape ``(n,)``.
        """
        return planck_taper(
            x, self.zero_low, self.one_low, self.one_high, self.zero_high
        )

    def window_derivative(self, x: Float[np.ndarray, " n"]) -> Float[np.ndarray, " n"]:
        """Evaluate the analytic derivative of the collar's profile.

        Parameters
        ----------
        x : numpy.ndarray
            Points at which to evaluate it, shape ``(n,)``.

        Returns
        -------
        numpy.ndarray
            ``d window / dx`` in reciprocal units of ``x``, shape ``(n,)``; see
            :func:`planck_taper_derivative`.
        """
        return planck_taper_derivative(
            x, self.zero_low, self.one_low, self.one_high, self.zero_high
        )


FFT_SIZES: Final = ("auto", "pow2", "smooth")
"""The transform-size rules of :meth:`FieldGrid.from_limits` and its envelope twin."""

DEFAULT_COLLAR_FRACTION: Final = 0.1
"""Width of each temporal collar, as a fraction of the realised window (SEP 21 D3)."""

_RADIX_PENALTY: Final = {3: 1.05, 5: 1.04}
"""Relative cost per factor of 3 and of 5 in an FFT's length.

Fitted to FFTW's batched real transforms, measured planning, every even
2-3-5-smooth length from 4096 to 262144, on Intel and Apple CPUs (SEP 21 phase 0a,
finding 14): within 10 per cent of the cheapest length for most requests.
"""


def fft_cost(n: int) -> float:
    """Return the modelled relative cost of a real transform of length `n`.

    ``n log2 n``, times ``_RADIX_PENALTY`` for each factor of 3 and of 5.

    Parameters
    ----------
    n : int
        Transform length, at least 2.

    Returns
    -------
    float
        Relative cost, comparable between lengths only.

    Examples
    --------
    >>> from solprop.grid import fft_cost
    >>> fft_cost(4096) < fft_cost(4374)  # 4374 = 2 * 3^7
    True
    """
    cost = n * math.log2(n)
    for prime, penalty in _RADIX_PENALTY.items():
        while n % prime == 0:
            n //= prime
            cost *= penalty
    return cost


def _smooth_sizes(low: int, high: int) -> list[int]:
    """Return every even 2-3-5-smooth integer in ``[low, high]``.

    Parameters
    ----------
    low, high : int
        Inclusive bounds.

    Returns
    -------
    list of int
        The sizes, increasing.
    """
    found = []
    size = low
    while size <= high:
        found.append(size)
        size = _next_smooth_even(size + 1)
    return found


CollarPlacement = Literal["outside", "inside"]
"""Where the temporal collar lies against the requested window."""


def _needed_window(trange: float, collar_fraction: float, placement: str) -> float:
    """Return the window a grid must realise at least, s.

    Parameters
    ----------
    trange : float
        Requested window, s.
    collar_fraction : float
        Width of each temporal collar as a fraction of the realised window.
    placement : {'outside', 'inside'}
        Where the collar lies against the request.

    Returns
    -------
    float
        ``trange / (1 - 2 collar_fraction)`` outside, ``trange`` inside.

    Raises
    ------
    GridError
        If the placement is neither.
    """
    if placement == "outside":
        return trange / (1.0 - 2.0 * collar_fraction)
    if placement == "inside":
        return trange
    msg = f"collar_placement must be 'outside' or 'inside', got {placement!r}"
    raise GridError(msg)


def _next_smooth_even(n: int) -> int:
    """Return the smallest even 2-3-5-smooth integer that is at least ``n``.

    Parameters
    ----------
    n : int
        Lower bound on the result.

    Returns
    -------
    int
        The smallest even integer of the form ``2^a 3^b 5^c`` with ``a >= 1`` that is at
        least ``n``.
    """
    # scipy.fft.next_fast_len also admits factors 7 and 11 (next_fast_len(4100) is
    # 4116 = 2^2 * 3 * 7^3), so it cannot be used where 2-3-5-smooth is required.
    power_of_two = 1 << max(1, (max(n, 2) - 1).bit_length())
    best = power_of_two
    five = 1
    while five < best:
        fifteen = five
        while fifteen < best:
            two = 2
            while two * fifteen < n:
                two *= 2
            best = min(best, two * fifteen)
            fifteen *= 3
        five *= 5
    return best


def _fft_size(needed: float, fft_size: str) -> int:
    """Round a required sample count up to a transform size.

    Parameters
    ----------
    needed : float
        Number of samples the request implies.
    fft_size : {'auto', 'pow2', 'smooth'}
        Round up to the next power of two, or to the next even 2-3-5-smooth size
        (``'auto'`` rounds as ``'smooth'`` here; its choice is made in
        :func:`_choose_sizes`).

    Returns
    -------
    int
        The rounded transform size, even and at least 4.

    Raises
    ------
    GridError
        If ``fft_size`` is not one of :data:`FFT_SIZES`.
    """
    return max(fourier.MIN_N_T, _next_size(needed, fft_size))


def _next_size(needed: float, fft_size: str) -> int:
    """Round a count up to the next size of the chosen family, with no lower bound.

    Parameters
    ----------
    needed : float
        Lower bound on the result.
    fft_size : {'auto', 'pow2', 'smooth'}
        Round up to the next power of two, or to the next even 2-3-5-smooth size
        (``'auto'`` rounds as ``'smooth'`` here; its choice is made in
        :func:`_choose_sizes`).

    Returns
    -------
    int
        The rounded size. Used for the oversampling multiplier, where a floor of four
        would inflate the fine grid fourfold.

    Raises
    ------
    GridError
        If ``fft_size`` is not one of :data:`FFT_SIZES`.
    """
    if fft_size not in FFT_SIZES:
        msg = f"fft_size must be one of {FFT_SIZES}, got {fft_size!r}"
        raise GridError(msg)
    count = max(1, math.ceil(needed))
    if fft_size == "pow2":
        return 1 << (count - 1).bit_length()
    return _next_smooth_even(count)


def _coarse_size(needed: int, n_to: int) -> int:
    """Return the smallest even divisor of ``n_to`` that is at least ``needed``.

    Choosing the coarse transform size from the divisors of the fine one guarantees
    ``n_to % n_t == 0``, and hence that the coarse time samples are a subset of the fine
    ones. When ``n_to`` is a power of two this is the ``2^ceil(log2(...))`` rounding
    exactly; when it is merely 2-3-5-smooth, that rounding could exceed ``n_to`` and
    duplicate spectral bins, which this cannot.

    Parameters
    ----------
    needed : int
        Lower bound on the coarse size.
    n_to : int
        Fine transform size, even.

    Returns
    -------
    int
        The chosen coarse size, clamped to ``n_to`` when nothing smaller is big
        enough.
    """
    if needed >= n_to:
        return n_to
    divisors = set()
    for candidate in range(1, math.isqrt(n_to) + 1):
        if n_to % candidate == 0:
            divisors.add(candidate)
            divisors.add(n_to // candidate)
    return min(d for d in divisors if d >= needed and d % 2 == 0)


def _choose_sizes(
    half_bins: Callable[[float, int], int],
    trange: float,
    dt_fine: float,
    fft_size: str,
) -> tuple[int, int]:
    """Choose the coarse and fine transform sizes together.

    The fine size follows from the requested window; the coarse size must be large
    enough to hold the apodised frequency window *and* must divide the fine size, so
    that the coarse time samples coincide with fine ones. Taking the coarse size from
    the divisors of the fine one satisfies both, and for a power-of-two fine size it
    is the power-of-two rounding exactly. A 2-3-5-smooth fine size can have too few
    divisors for that to leave any oversampling at all, and only then is the fine grid
    widened to a multiple of the coarse size the request implies.

    Parameters
    ----------
    half_bins : callable
        Function of the bin spacing in rad/s and the fine size, returning the number of
        positive-frequency bins the coarse grid must hold.
    trange : float
        Requested time window, s.
    dt_fine : float
        Fine sample spacing, s.
    fft_size : {'pow2', 'smooth'}
        Rounding rule for both sizes.

    Returns
    -------
    tuple of int
        The coarse and fine numbers of time samples, ``(n_t, n_to)``.
    """
    if fft_size == "auto":
        # The cheapest pair by the transform-cost model, among the smooth fine sizes up
        # to the next power of two (which is one of them).
        needed = trange / dt_fine
        ceiling = _fft_size(needed, "pow2")
        pairs = []
        for n_fine in _smooth_sizes(_fft_size(needed, "smooth"), ceiling):
            pair = _choose_sizes(half_bins, n_fine * dt_fine, dt_fine, "smooth")
            pairs.append((fft_cost(pair[1]) + fft_cost(pair[0]), pair[1], pair))
        return min(pairs)[2]
    n_to = _fft_size(trange / dt_fine, fft_size)
    half = half_bins(2.0 * np.pi / (n_to * dt_fine), n_to)
    n_t = _coarse_size(2 * half, n_to)
    if n_t == n_to and 2 * half <= n_to // 2:
        # The fine size's divisors were too coarse and the oversampling was lost, which
        # can happen only for a 2-3-5-smooth size with few divisors (2*5^4, say).
        # Widen the fine grid to a multiple of the coarse size the request implies.
        ideal = _fft_size(float(2 * half), fft_size)
        wider = ideal * _next_size(n_to / ideal, fft_size)
        wider_half = half_bins(2.0 * np.pi / (wider * dt_fine), wider)
        # The widening always leaves some oversampling: `ideal` divides `wider` and is
        # strictly smaller than it, and the entry condition 2*half <= n_to/2 bounds the
        # regrown `wider_half` below ideal/2, so `_coarse_size` can never be driven all
        # the way back up to `wider`.
        return _coarse_size(2 * wider_half, wider), wider
    return n_t, n_to


@dataclass(frozen=True, eq=False)
class TimeGrid(Displayed):
    """Shared state of the field and envelope grids.

    The eleven stored attributes below are the whole of a grid's state; every other
    quantity named in ``docs/dev/kernel-contract.md`` §2 is a property derived from
    them, so each has exactly one definition. Instances are frozen and hold private
    read-only copies of their axes, so a grid can be shared without defensive copying.
    Equality is by identity, because comparing whole axes elementwise is never what a
    caller means.

    Build one with :meth:`FieldGrid.from_limits` or :meth:`EnvelopeGrid.from_limits`
    rather than by hand; the constructor is the low-level path and only
    :meth:`validate` stands behind it.

    Attributes
    ----------
    length : float
        Propagation distance the grid was built for, m.
    reference_wavelength : float
        Reference wavelength, m; fixes ``omega0`` and the co-moving frame.
    t : numpy.ndarray
        Coarse time axis, s, shape ``(n_t,)``, centred on zero.
    omega : numpy.ndarray
        Coarse angular frequency axis, rad/s, shape ``(n_omega,)``. Non-negative and
        increasing on a field grid; in FFT order about ``omega0`` on an envelope grid.
    t_fine : numpy.ndarray
        Fine (oversampled) time axis, s, shape ``(n_to,)``, centred on zero.
    omega_fine : numpy.ndarray
        Fine angular frequency axis, rad/s, shape ``(n_omega_fine,)``.
    dt : float
        Coarse sample spacing, s.
    dt_fine : float
        Fine sample spacing, s.
    ffac : float
        Oversampling factor: the fine grid is sampled at ``ffac`` times the highest
        frequency of the window. At least 2.
    omega_collar : Collar
        Spectral collar geometry, rad/s: rises over ``[omega_min/2, omega_min]`` and
        falls over ``[omega_max, 1.1 omega_max]``.
    t_collar : Collar
        Temporal collar geometry, s, symmetric about zero and guaranteed non-degenerate.
    kind : {'field', 'envelope'}
        Spectral layout, fixed by the subclass and passed to :mod:`solprop.fourier`.

    Examples
    --------
    >>> from solprop.grid import FieldGrid, TimeGrid
    >>> isinstance(FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15), TimeGrid)
    True
    """

    length: float
    reference_wavelength: float
    t: Float[np.ndarray, " n_t"] = field(repr=False)
    omega: Float[np.ndarray, " n_omega"] = field(repr=False)
    t_fine: Float[np.ndarray, " n_to"] = field(repr=False)
    omega_fine: Float[np.ndarray, " n_omega_fine"] = field(repr=False)
    dt: float
    dt_fine: float
    ffac: float
    omega_collar: Collar
    t_collar: Collar

    kind: ClassVar[GridKind] = "field"

    def __post_init__(self) -> None:
        """Replace the axes with private read-only copies, so the grid is frozen."""
        for name in ("t", "omega", "t_fine", "omega_fine"):
            axis = np.array(getattr(self, name), dtype=np.float64, copy=True)
            axis.setflags(write=False)
            object.__setattr__(self, name, axis)

    # --- sizes ---------------------------------------------------------------------

    @property
    def n_t(self) -> int:
        """Return the number of coarse time samples.

        Returns
        -------
        int
            The length of ``t``.
        """
        return self.t.size

    @property
    def n_to(self) -> int:
        """Return the number of fine time samples.

        Returns
        -------
        int
            The length of ``t_fine``.
        """
        return self.t_fine.size

    @property
    def n_omega(self) -> int:
        """Return the number of coarse spectral bins.

        Returns
        -------
        int
            The length of ``omega``.
        """
        return self.omega.size

    @property
    def n_omega_fine(self) -> int:
        """Return the number of fine spectral bins.

        Returns
        -------
        int
            The length of ``omega_fine``.
        """
        return self.omega_fine.size

    # --- frequencies ---------------------------------------------------------------

    @property
    def omega0(self) -> float:
        """Return the reference angular frequency, rad/s.

        Returns
        -------
        float
            ``2 pi c / reference_wavelength``.
        """
        return 2.0 * np.pi * C / self.reference_wavelength

    @property
    def d_omega(self) -> float:
        """Return the spectral bin spacing, rad/s.

        Returns
        -------
        float
            ``2 pi / (n_to dt_fine)``, the same on the coarse and the fine grid.
        """
        # The axes are built from the fine window, and dt = dt_fine (n_to // n_t) is
        # rounded, so 2 pi / (n_t dt) can differ from this in the last place on a grid
        # that is not a power of two; the record rebuilds the axes from this value.
        return 2.0 * np.pi / (self.n_to * self.dt_fine)

    @property
    def omega_min(self) -> float:
        """Return the lowest in-band angular frequency, rad/s.

        Returns
        -------
        float
            The frequency of the longest requested wavelength.
        """
        return self.omega_collar.one_low

    @property
    def omega_max(self) -> float:
        """Return the highest in-band angular frequency, rad/s.

        Returns
        -------
        float
            The frequency of the shortest requested wavelength.
        """
        return self.omega_collar.one_high

    @property
    def omega_max_window(self) -> float:
        """Return the upper edge of the apodised frequency window, rad/s.

        Returns
        -------
        float
            ``1.1 omega_max``, where the spectral collar reaches zero.
        """
        return self.omega_collar.zero_high

    # --- masks and windows -----------------------------------------------------------

    @property
    def active_mask(self) -> Bool[np.ndarray, " n_omega"]:
        """Return the bins where the physics lives, inside the apodised window.

        The dispersion relation, the noise and the error norm are
        evaluated here and nowhere else.

        Returns
        -------
        numpy.ndarray
            Boolean mask over the coarse frequency axis, shape ``(n_omega,)``.
        """
        return (self.omega > 0.5 * self.omega_min) & (
            self.omega < self.omega_max_window
        )

    @property
    def inband_mask(self) -> Bool[np.ndarray, " n_omega"]:
        """Return the bins inside the requested wavelength limits.

        Returns
        -------
        numpy.ndarray
            Boolean mask over the coarse frequency axis, shape ``(n_omega,)``, true
            where ``omega_min <= omega <= omega_max``.
        """
        return (self.omega >= self.omega_min) & (self.omega <= self.omega_max)

    @property
    def omega_window(self) -> Float[np.ndarray, " n_omega"]:
        """Return the spectral collar profile on the coarse frequency axis.

        Returns
        -------
        numpy.ndarray
            Window values in ``[0, 1]``, shape ``(n_omega,)``, equal to 1 in band.
        """
        return self.omega_collar.window(self.omega)

    @property
    def t_window(self) -> Float[np.ndarray, " n_t"]:
        """Return the temporal collar profile on the coarse time axis.

        Returns
        -------
        numpy.ndarray
            Window values in ``[0, 1]``, shape ``(n_t,)``, equal to 1 inside the window.
        """
        return self.t_collar.window(self.t)

    @property
    def t_window_fine(self) -> Float[np.ndarray, " n_to"]:
        """Return the temporal collar profile on the fine time axis.

        Both temporal windows come from the one collar geometry, so the coarse and fine
        apodisations agree exactly wherever their samples coincide, rather than each
        window being built from its own axis's extremes, which differ at the upper edge.

        Returns
        -------
        numpy.ndarray
            Window values in ``[0, 1]``, shape ``(n_to,)``, 1 inside the window.
        """
        return self.t_collar.window(self.t_fine)

    # --- oversampling -----------------------------------------------------------------

    @property
    def crop_index(self) -> np.ndarray:
        """Return the positions of the coarse spectral bins inside the fine spectrum.

        Returns
        -------
        numpy.ndarray
            Index array of ``numpy.intp``, shape ``(n_omega,)``.
        """
        return fourier.crop_index(self.n_t, self.n_to, kind=self.kind)

    @property
    def pad_scale(self) -> float:
        """Return the factor applied when padding the coarse spectrum out.

        Returns
        -------
        float
            ``n_to / n_t``.
        """
        return fourier.pad_scale(self.n_t, self.n_to)

    @property
    def crop_scale(self) -> float:
        """Return the factor applied when cropping the fine spectrum back.

        Returns
        -------
        float
            ``n_t / n_to``.
        """
        return fourier.crop_scale(self.n_t, self.n_to)

    # --- validation -------------------------------------------------------------------

    def validate(self) -> None:
        """Check every grid invariant, raising on the first that fails.

        The invariants are those of ``docs/dev/conventions.md`` §3. Both ``from_limits``
        classmethods call this, so a grid they return is always valid; call it again
        after building or loading one by hand.

        Raises
        ------
        GridError
            If any invariant fails, naming the offending quantity.

        Examples
        --------
        >>> from solprop.grid import FieldGrid
        >>> FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15).validate()
        """
        for name, actual, expected in (
            ("n_omega", self.n_omega, fourier.n_omega_for(self.n_t, self.kind)),
            (
                "n_omega_fine",
                self.n_omega_fine,
                fourier.n_omega_for(self.n_to, self.kind),
            ),
        ):
            if actual != expected:
                msg = f"{name} is {actual}, expected {expected} on a {self.kind} grid"
                raise GridError(msg)
        if self.n_to % self.n_t:
            msg = (
                f"n_to must be an integer multiple of n_t so that the coarse time "
                f"samples coincide with fine ones, got n_to={self.n_to} and "
                f"n_t={self.n_t}"
            )
            raise GridError(msg)
        ratio = self.n_to // self.n_t
        if not math.isclose(self.dt / self.dt_fine, ratio, rel_tol=INVARIANT_RTOL):
            msg = (
                f"dt/dt_fine is {self.dt / self.dt_fine!r}, expected n_to/n_t = {ratio}"
            )
            raise GridError(msg)
        # The coarse samples are the fine ones taken every n_to/n_t: exact in real
        # arithmetic, and within a few units in the last place of max|t| in practice,
        # because the two axes are built from different products of the same integers.
        if not np.allclose(
            self.t,
            self.t_fine[::ratio],
            rtol=0.0,
            atol=INVARIANT_RTOL * float(np.max(np.abs(self.t_fine))),
        ):
            msg = "t does not coincide with every n_to/n_t-th sample of t_fine"
            raise GridError(msg)
        if self.ffac < 2.0:
            msg = (
                f"ffac must be at least 2 so that the fine grid resolves the frequency "
                f"window itself, got {self.ffac!r}"
            )
            raise GridError(msg)
        self._validate_spectrum()

    def _validate_spectrum(self) -> None:
        """Check the frequency-axis invariants that depend on the grid's layout.

        Raises
        ------
        NotImplementedError
            Always; each subclass supplies the invariants for its own layout.
        """
        raise NotImplementedError

    def _band(self) -> str:
        """Return the spectral band as wavelengths, for printing.

        Returns
        -------
        str
            ``"150-4000 nm"``, with ``inf`` where the band reaches zero frequency.
        """
        shortest = 2.0e9 * np.pi * C / self.omega_max
        longest = 2.0e9 * np.pi * C / self.omega_min if self.omega_min > 0.0 else np.inf
        return f"{shortest:.4g}-{longest:.4g} nm"

    def __repr__(self) -> str:
        """Return a one-line description: samples, window, band and reference.

        Returns
        -------
        str
            For example
            ``FieldGrid(n_t=8192, window=400 fs, band=150-4000 nm, reference=800 nm)``.
        """
        return line(
            type(self).__name__,
            [
                ("n_t", self.n_t),
                ("window", engineering(self.n_t * self.dt, "s")),
                ("band", self._band()),
                ("reference", engineering(self.reference_wavelength, "m")),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        kind = "a real field" if self.kind == "field" else "an envelope"
        collar = self.t_collar
        return f"{type(self).__name__}: {kind} grid", [
            ("time samples", f"{self.n_t} ({self.n_to} oversampled)"),
            (
                "time window",
                f"{engineering(self.n_t * self.dt, 's')}, "
                f"step {engineering(self.dt, 's')}",
            ),
            (
                "undamped window",
                f"{engineering(collar.one_low, 's')} to "
                f"{engineering(collar.one_high, 's')}",
            ),
            (
                "frequency samples",
                f"{self.n_omega}, step {self.d_omega / (2.0e12 * np.pi):.4g} THz",
            ),
            ("spectral band", self._band()),
            ("reference", engineering(self.reference_wavelength, "m")),
            ("length", engineering(self.length, "m")),
        ]


def _frequency_limits(wavelength_limits: Sequence[float]) -> tuple[float, float]:
    """Convert wavelength limits, m, to angular frequency limits, rad/s.

    The parameter is a general sequence rather than a two-tuple on purpose: this is the
    run-time guard on the pair's length, and the test session's ``beartype`` hook would
    otherwise reject a wrong-length tuple before this check is reached.

    Parameters
    ----------
    wavelength_limits : sequence of float
        Shortest and longest wavelength of the frequency window, m.

    Returns
    -------
    tuple of float
        The lowest and highest angular frequency, rad/s.

    Raises
    ------
    GridError
        If the pair is not two strictly increasing positive wavelengths.
    """
    if len(wavelength_limits) != 2:
        msg = (
            f"wavelength_limits must hold exactly two values, got {wavelength_limits!r}"
        )
        raise GridError(msg, parameter="wavelength_limits")
    low, high = (float(value) for value in wavelength_limits)
    if not 0.0 < low < high:
        msg = (
            "wavelength_limits must be a strictly increasing pair of positive "
            f"wavelengths in metres, got {wavelength_limits!r}"
        )
        raise GridError(msg, parameter="wavelength_limits")
    return 2.0 * np.pi * C / high, 2.0 * np.pi * C / low


def _check_common(
    length: float,
    reference_wavelength: float,
    trange: float,
    dt: float | None,
    ffac: float,
    collar_fraction: float,
) -> None:
    """Validate the constructor arguments both grids share, naming each failure.

    Parameters
    ----------
    length : float
        Propagation distance, m.
    reference_wavelength : float
        Reference wavelength, m.
    trange : float
        Requested time window, s.
    dt : float or None
        Requested sample spacing, s.
    ffac : float
        Fine-grid oversampling factor.
    collar_fraction : float
        Minimum temporal collar width as a fraction of the window.

    Raises
    ------
    GridError
        If any argument is outside its valid range.
    """
    if length < 0.0:
        msg = f"length must be a non-negative distance in metres, got {length!r}"
        raise GridError(msg, parameter="length")
    if reference_wavelength <= 0.0:
        msg = (
            "reference_wavelength must be positive, in metres, got "
            f"{reference_wavelength!r}"
        )
        raise GridError(msg, parameter="reference_wavelength")
    if trange <= 0.0:
        msg = f"trange must be a positive time window in seconds, got {trange!r}"
        raise GridError(msg, parameter="trange")
    if dt is not None and dt <= 0.0:
        msg = f"dt must be a positive sample spacing in seconds, got {dt!r}"
        raise GridError(msg, parameter="dt")
    if ffac < 2.0:
        msg = (
            "ffac must be at least 2 so that the fine grid resolves the frequency "
            f"window itself, got {ffac!r}"
        )
        raise GridError(msg, parameter="ffac")
    if not 0.0 <= collar_fraction < 0.5:
        msg = (
            "collar_fraction must lie in [0, 0.5): zero leaves no temporal collar, and "
            f"half or more leaves no window, got {collar_fraction!r}"
        )
        raise GridError(msg, parameter="collar_fraction")


def _temporal_collar(
    trange: float, trange_even: float, collar_fraction: float
) -> Collar:
    """Build the temporal collar, widened if the realised window leaves it too narrow.

    Putting the flat region at ``+/- trange/2`` and the zeros at the ends of the time
    axis degenerates to a hard edge whenever the transform size makes the
    realised window equal to the requested one — the common case on a power-of-two grid.
    Requiring at least ``collar_fraction`` of the full window at each end removes that
    case (conventions.md §3).

    Parameters
    ----------
    trange : float
        Requested time window, s.
    trange_even : float
        Realised time window, s, at least ``trange``.
    collar_fraction : float
        Minimum width of each collar as a fraction of the full window.

    Returns
    -------
    Collar
        The temporal collar geometry, symmetric about zero.
    """
    half = 0.5 * trange_even
    inner = min(0.5 * trange, half * (1.0 - 2.0 * collar_fraction))
    return Collar(-half, -inner, inner, half)


def _log_grid(kind: str, n_t: int, n_to: int, dt: float, trange_even: float) -> None:
    """Record the realised grid at debug level.

    Parameters
    ----------
    kind : str
        Which grid was built.
    n_t : int
        Coarse number of time samples.
    n_to : int
        Fine number of time samples.
    dt : float
        Coarse sample spacing, s.
    trange_even : float
        Realised time window, s.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "%s grid: n_t=%d, n_to=%d, dt=%.3f as, window=%.1f fs",
            kind,
            n_t,
            n_to,
            dt * 1e18,
            trange_even * 1e15,
        )


@dataclass(frozen=True, eq=False, repr=False)
class FieldGrid(TimeGrid):
    """Grid for field-resolved propagation of a real electric field.

    The state is the one-sided spectrum on ``omega_k = k d_omega``, ``k = 0 ... n_t/2``,
    so ``n_omega = n_t/2 + 1`` and the time samples are real. Build one with
    :meth:`from_limits`; see :class:`TimeGrid` for the stored attributes.

    Examples
    --------
    >>> from solprop.grid import FieldGrid
    >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
    >>> grid.n_t, grid.n_to, grid.n_omega
    (5120, 10240, 2561)
    """

    kind: ClassVar[GridKind] = "field"

    # The suppression below is for PLR0913: the arguments are the grid's physical
    # specification, each named separately in conventions.md §3, so grouping them into a
    # dataclass would hide the API rather than clarify it.
    @classmethod
    def from_limits(  # noqa: PLR0913
        cls,
        length: float,
        reference_wavelength: float,
        wavelength_limits: tuple[float, float],
        trange: float,
        *,
        dt: float | None = None,
        ffac: float = 6.0,
        fft_size: str = "auto",
        collar_fraction: float = DEFAULT_COLLAR_FRACTION,
        collar_placement: CollarPlacement = "outside",
        # The return annotation stays quoted: beartype resolves it while the class body
        # is still executing, when the bare class name is not yet bound.
    ) -> "FieldGrid":  # noqa: UP037
        """Build a field grid from the wavelength limits and the requested time window.

        The fine grid is sampled at ``ffac`` times the highest frequency of the window,
        its length is rounded up to a transform size, and the coarse grid keeps the fine
        bins out to just past ``1.1 omega_max``. Note that ``ffac`` changes the *whole*
        grid — the bin spacing, the realised window and ``n_omega`` — not only the fine
        one, so two values of ``ffac`` give two different grids that both satisfy the
        request; compare them as physical spectra on their common band, never array to
        array.

        Parameters
        ----------
        length : float
            Propagation distance, m.
        reference_wavelength : float
            Reference wavelength, m, for example the pump's centre wavelength.
        wavelength_limits : tuple of float
            Shortest and longest wavelength of the frequency window, m, strictly
            increasing and positive.
        trange : float
            Requested extent of the time window, s. The realised window is at least this
            long, because the transform size is rounded up.
        dt : float or None, optional
            Requested fine sample spacing, s. The value used is this or the value the
            frequency window demands, whichever is smaller.
        ffac : float, optional
            Fine-grid oversampling factor, at least 2. The default 6 resolves a cubic
            response, whose third harmonic reaches ``3 omega_max``; 4 suffices for a
            response that generates no third harmonic, such as ``|E_a|^2 E``. At exactly
            2 the fine Nyquist frequency is ``omega_max`` itself, so the apodisation
            collar above it is truncated rather than represented.
        fft_size : {'auto', 'pow2', 'smooth'}, optional
            How the transform sizes are rounded up: ``'auto'``, the default, takes the
            cheapest even 2-3-5-smooth size between the requirement and the next power
            of two by a transform-cost model (:func:`fft_cost`); ``'pow2'`` the next
            power of two; ``'smooth'`` the next even 2-3-5-smooth size.
        collar_fraction : float, optional
            Width of each temporal collar as a fraction of the realised window, in
            ``[0, 0.5)``. The collar lies **outside** the requested window: the grid is
            widened to at least ``trange / (1 - 2 collar_fraction)`` so that the whole
            request stays flat (SEP 21). Zero leaves no room for a temporal collar, for
            a run without one. The default is 0.1.
        collar_placement : {'outside', 'inside'}, optional
            ``'outside'``, the default, widens the window so that the collar lies
            outside the request. ``'inside'`` is the geometry before SEP 21, kept to
            reproduce frozen witnesses: the window is the request rounded up, and the
            collar takes at least `collar_fraction` of it, inside the request when the
            rounding leaves too little slack.

        Returns
        -------
        FieldGrid
            A validated grid.

        Raises
        ------
        GridError
            If any argument is outside its valid range, or an invariant fails.

        Examples
        --------
        >>> from solprop.grid import FieldGrid
        >>> grid = FieldGrid.from_limits(1.5, 800e-9, (90e-9, 4e-6), 400e-15)
        >>> grid.n_t, grid.n_to
        (5120, 10240)
        >>> float(grid.dt / grid.dt_fine)
        2.0
        """
        _check_common(length, reference_wavelength, trange, dt, ffac, collar_fraction)
        omega_min, omega_max = _frequency_limits(wavelength_limits)
        omega_max_window = WINDOW_FACTOR * omega_max

        required = 2.0 * np.pi / (ffac * omega_max)
        dt_fine = required if dt is None else min(required, dt)

        def half_bins(d_omega: float, n_to: int) -> int:
            """Return the coarse half-size needed to hold the apodised window.

            Parameters
            ----------
            d_omega : float
                Bin spacing, rad/s.
            n_to : int
                Fine transform size.

            Returns
            -------
            int
                Number of positive-frequency bins the coarse grid must hold.
            """
            # Keep every fine bin out to the first one past the apodised window edge,
            # so the collar is fully represented on the coarse grid.
            return min(math.floor(omega_max_window / d_omega) + 2, n_to // 2 + 1)

        needed = _needed_window(trange, collar_fraction, collar_placement)
        n_t, n_to = _choose_sizes(half_bins, needed, dt_fine, fft_size)
        trange_even = n_to * dt_fine
        d_omega = 2.0 * np.pi / trange_even
        # dt comes from the integer ratio rather than from pi/omega[-1], so that the
        # dt/dt_fine == n_to/n_t invariant holds by construction, not to round-off.
        dt_coarse = dt_fine * (n_to // n_t)

        _log_grid("field", n_t, n_to, dt_coarse, trange_even)
        grid = cls(
            length=float(length),
            reference_wavelength=float(reference_wavelength),
            t=(np.arange(n_t, dtype=float) - n_t / 2) * dt_coarse,
            omega=np.arange(n_t // 2 + 1, dtype=float) * d_omega,
            t_fine=(np.arange(n_to, dtype=float) - n_to / 2) * dt_fine,
            omega_fine=np.arange(n_to // 2 + 1, dtype=float) * d_omega,
            dt=dt_coarse,
            dt_fine=float(dt_fine),
            ffac=float(ffac),
            omega_collar=Collar(
                0.5 * omega_min, omega_min, omega_max, omega_max_window
            ),
            t_collar=_temporal_collar(trange, trange_even, collar_fraction),
        )
        grid.validate()
        return grid

    def _validate_spectrum(self) -> None:
        """Check that the frequency axis is the one-sided rfft axis of the time axis.

        Raises
        ------
        GridError
            If ``dt`` is not ``pi/omega[-1]``, or ``omega`` is not ``k d_omega``.
        """
        expected_dt = np.pi / self.omega[-1]
        if not math.isclose(self.dt, expected_dt, rel_tol=INVARIANT_RTOL):
            msg = f"dt is {self.dt!r}, expected pi/omega[-1] = {expected_dt!r}"
            raise GridError(msg)
        expected = np.arange(self.n_omega, dtype=float) * self.d_omega
        if not np.allclose(self.omega, expected, rtol=INVARIANT_RTOL, atol=0.0):
            msg = "omega is not the non-negative rfft axis k*d_omega, k = 0 ... n_t/2"
            raise GridError(msg)


@dataclass(frozen=True, eq=False, repr=False)
class EnvelopeGrid(TimeGrid):
    """Grid for propagation of a complex envelope about a carrier.

    The state is the full spectrum on ``omega = omega0 + v`` with ``v`` in FFT order, so
    ``n_omega = n_t`` and the time samples are complex. The kernels never shift; only
    processing and plotting do. Build one with :meth:`from_limits`; see
    :class:`TimeGrid` for the stored attributes.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 20e-12)
    >>> grid.n_omega == grid.n_t, grid.n_to == grid.n_t
    (True, True)
    """

    kind: ClassVar[GridKind] = "envelope"

    # The suppression below is for PLR0913: the arguments are the grid's physical
    # specification, each named separately in conventions.md §3, so grouping them into a
    # dataclass would hide the API rather than clarify it.
    @classmethod
    def from_limits(  # noqa: PLR0913
        cls,
        length: float,
        reference_wavelength: float,
        wavelength_limits: tuple[float, float],
        trange: float,
        *,
        dt: float | None = None,
        thg: bool = False,
        ffac: float | None = None,
        fft_size: str = "auto",
        collar_fraction: float = DEFAULT_COLLAR_FRACTION,
        collar_placement: CollarPlacement = "outside",
        # The return annotation stays quoted: beartype resolves it while the class body
        # is still executing, when the bare class name is not yet bound.
    ) -> "EnvelopeGrid":  # noqa: UP037
        """Build an envelope grid from the wavelength limits and the time window.

        Without third harmonic the grid is sampled at twice the widest relative
        frequency it must carry, so the fine and coarse grids coincide and there is no
        oversampling unless a finer ``dt`` is asked for. With ``thg=True`` it is sampled
        at six times that, and the coarse grid is cropped back to the apodised window.

        The relative-frequency span is measured symmetrically, as the larger of
        ``1.1 omega_max - omega0`` (``omega_max - omega0`` with ``thg=True``, where the
        factor 6 already leaves room for the collar) and ``omega0 - omega_min/2``.
        Measuring only the
        upper side and applying the result to both would silently drop the whole
        low-frequency half of the band whenever the reference wavelength sits near the
        blue edge of the window.

        Parameters
        ----------
        length : float
            Propagation distance, m.
        reference_wavelength : float
            Reference wavelength, m; the carrier the envelope is written about.
        wavelength_limits : tuple of float
            Shortest and longest wavelength of the frequency window, m, strictly
            increasing and positive.
        trange : float
            Requested extent of the time window, s.
        dt : float or None, optional
            Requested fine sample spacing, s. The value used is this or the value the
            frequency window demands, whichever is smaller; a finer value here turns on
            oversampling.
        thg : bool, optional
            Leave room for third-harmonic generation. Sets the default ``ffac`` to 6
            instead of 2 and turns on oversampling.
        ffac : float or None, optional
            Fine-grid oversampling factor, at least 2. Defaults to 6 with ``thg`` and 2
            without.
        fft_size : {'auto', 'pow2', 'smooth'}, optional
            How the transform sizes are rounded up, as for
            :meth:`FieldGrid.from_limits`; ``'auto'`` by default.
        collar_fraction : float, optional
            Width of each temporal collar as a fraction of the realised window, in
            ``[0, 0.5)``, outside the requested window, as for
            :meth:`FieldGrid.from_limits`. The default is 0.1.
        collar_placement : {'outside', 'inside'}, optional
            Where the collar lies against the request, as for
            :meth:`FieldGrid.from_limits`; ``'outside'`` by default.

        Returns
        -------
        EnvelopeGrid
            A validated grid.

        Raises
        ------
        GridError
            If any argument is outside its valid range, or an invariant fails.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 20e-12)
        >>> grid.n_t == grid.n_to
        True
        >>> limits = (400e-9, 2e-6)
        >>> thg = EnvelopeGrid.from_limits(1.0, 835e-9, limits, 20e-12, thg=True)
        >>> thg.n_to // thg.n_t
        2
        """
        ffac = (6.0 if thg else 2.0) if ffac is None else ffac
        _check_common(length, reference_wavelength, trange, dt, ffac, collar_fraction)
        omega_min, omega_max = _frequency_limits(wavelength_limits)
        omega_max_window = WINDOW_FACTOR * omega_max
        omega0 = 2.0 * np.pi * C / reference_wavelength

        # Without third harmonic the window must still reach 1.1*omega_max, because the
        # spectral collar lives there; with it, the factor 6 already leaves the room.
        upper = (omega_max if thg else omega_max_window) - omega0
        # The active band runs down to omega_min/2, so the grid must reach that far
        # below the carrier too, not only `upper` cropped symmetrically.
        span = max(abs(upper), abs(omega0 - 0.5 * omega_min))
        required = 2.0 * np.pi / (ffac * span)
        dt_fine = required if dt is None else min(required, dt)
        oversampling = thg or dt_fine < required

        def half_bins(d_omega: float, n_to: int) -> int:
            """Return the coarse half-size needed to hold the apodised window.

            Parameters
            ----------
            d_omega : float
                Bin spacing, rad/s.
            n_to : int
                Fine transform size.

            Returns
            -------
            int
                Number of relative-frequency bins the coarse grid must hold each side.
            """
            if not oversampling:
                return n_to // 2
            reach = max(abs(omega_max_window - omega0), abs(omega0 - 0.5 * omega_min))
            return min(max(math.ceil(reach / d_omega), 1), n_to // 2)

        needed = _needed_window(trange, collar_fraction, collar_placement)
        n_t, n_to = _choose_sizes(half_bins, needed, dt_fine, fft_size)
        trange_even = n_to * dt_fine
        d_omega = 2.0 * np.pi / trange_even
        dt_coarse = dt_fine * (n_to // n_t)

        omega_fine = _fft_order(n_to) * d_omega + omega0
        omega = omega_fine[fourier.crop_index(n_t, n_to, kind="envelope")]

        _log_grid("envelope", n_t, n_to, dt_coarse, trange_even)
        grid = cls(
            length=float(length),
            reference_wavelength=float(reference_wavelength),
            t=(np.arange(n_t, dtype=float) - n_t / 2) * dt_coarse,
            omega=omega,
            t_fine=(np.arange(n_to, dtype=float) - n_to / 2) * dt_fine,
            omega_fine=omega_fine,
            dt=dt_coarse,
            dt_fine=float(dt_fine),
            ffac=float(ffac),
            omega_collar=Collar(
                0.5 * omega_min, omega_min, omega_max, omega_max_window
            ),
            t_collar=_temporal_collar(trange, trange_even, collar_fraction),
        )
        grid.validate()
        return grid

    def _validate_spectrum(self) -> None:
        """Check that the frequency axis is the FFT-order axis of the time axis.

        The field grid's ``omega_fine[-1]/omega[-1]`` has no meaning in FFT order, where
        the last bin is the one just below the carrier on both grids; the ratio that
        carries the information is the one of the minima.

        Raises
        ------
        GridError
            If ``dt`` is not ``2 pi / (n_omega d_omega)``, or ``omega`` is not the
            FFT-order axis about ``omega0``.
        """
        relative = self.omega - self.omega0
        expected_dt = -np.pi / relative.min()
        if not math.isclose(self.dt, expected_dt, rel_tol=INVARIANT_RTOL):
            msg = (
                f"dt is {self.dt!r}, expected -pi/min(omega - omega0) = {expected_dt!r}"
            )
            raise GridError(msg)
        ratio = (self.omega_fine - self.omega0).min() / relative.min()
        if not math.isclose(ratio, self.n_to / self.n_t, rel_tol=INVARIANT_RTOL):
            msg = (
                f"min(omega_fine - omega0)/min(omega - omega0) is {ratio!r}, expected "
                f"n_to/n_t = {self.n_to / self.n_t!r}"
            )
            raise GridError(msg)
        expected = _fft_order(self.n_t) * self.d_omega + self.omega0
        if not np.allclose(self.omega, expected, rtol=INVARIANT_RTOL, atol=0.0):
            msg = "omega is not the FFT-order axis omega0 + k*d_omega"
            raise GridError(msg)


def _fft_order(n: int) -> Float[np.ndarray, " n"]:
    """Return the integer bin indices of an ``n``-point DFT, in FFT order.

    Parameters
    ----------
    n : int
        Transform length, even.

    Returns
    -------
    numpy.ndarray
        The values ``0, 1, ..., n/2 - 1, -n/2, ..., -1`` as floats, shape ``(n,)``.
    """
    index = np.arange(n, dtype=float)
    index[n // 2 :] -= n
    return index
