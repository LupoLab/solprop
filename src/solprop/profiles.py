# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Scalar paths along the propagation axis, with breakpoints and derivatives.

A ``ZProfile`` is a scalar function of ``z``: the gas density in a pressure
gradient, the core radius of a taper, a pre-ionisation fraction. The cumulative
linear operator is built by resolving these paths first and taking the union of the
edges they produce as the mandatory split of the expensive array-valued operator
(:func:`solprop.translate.chebyshev.build_from_paths`, ``docs/dev/stepper.md`` section
4), so a profile has to offer three things:

* **vectorised evaluation**, because the builder samples at Chebyshev points in blocks;
* **declared breakpoints**, because a stage boundary is a hard kink that the adaptive
  split must never straddle and the stepper must land on;
* an **analytic derivative** where one exists, and an honest declaration where it does
  not, so that a caller can tell a converged quantity from a differenced one.

A profile is directly usable as a path: ``build_from_paths([profile], ...)`` works
through ``__call__``, and the caller passes the union of the profiles' ``breakpoints``
as ``mandatory_breakpoints``.

**Pressures are in bar** at this interface, the one documented departure from SI
(``docs/dev/conventions.md`` section 1); positions are in metres and densities in m^-3.

Vacuum ends. The steady-flow fill :math:`p(z) = \sqrt{p_0^2 + (z/L)(p_1^2 - p_0^2)}` has
:math:`\mathrm{d}p/\mathrm{d}z = (p_1^2 - p_0^2)/(2Lp)`, which diverges where
:math:`p \to 0`. A fill that reaches exactly zero therefore caps every integration
scheme at about third order however small the step (``docs/dev/stepper.md`` section 10,
trap 5), silently. :class:`PressureGradient` refuses such a stage end and substitutes a
small floor, logging what it did; the floor is relative to the highest pressure of the
fill, so it behaves the same for a 0.8 bar gradient and a 40 bar one. A fill built by
:func:`solprop.gasflow.steady_fill` from the gas and the bore replaces a vacuum end by
its choked pressure instead, which is physical rather than numerical.

Tapers. Given the core radius, :class:`PressureGradient` makes :math:`p^2` linear in the
hydraulic resistance :math:`\int a^{-4}\,\mathrm{d}z` rather than in ``z`` (Pennetta et
al., APL Photonics 4, 056105 (2019), eq. 6), so a narrow waist carries most of the
pressure drop; :func:`hydraulic_integral` evaluates that resistance.
"""

import hashlib
import itertools
import logging
import math
import numbers
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Bool, Float, Int
from numpy.polynomial import Chebyshev
from scipy.interpolate import CubicSpline

from solprop._display import Displayed, engineering, line, summarise
from solprop.constants import BAR, K_B, ROOM_TEMPERATURE
from solprop.exceptions import MaterialError, TranslationError
from solprop.physdata.equation_of_state import density
from solprop.physdata.gases import Gas, as_gas
from solprop.physdata.species import Species

__all__ = [
    "DEFAULT_PRESSURE_FLOOR_FRACTION",
    "MIN_STAGE_PRESSURE",
    "Constant",
    "DensityPath",
    "FromCallable",
    "Linear",
    "PiecewiseLinear",
    "PressureGradient",
    "Scaled",
    "SeriesOfPath",
    "Sum",
    "ZProfile",
    "as_profile",
    "hydraulic_integral",
    "profile_identity",
]

logger = logging.getLogger(__name__)

DEFAULT_PRESSURE_FLOOR_FRACTION: Final = 1e-4
"""Default vacuum-end pressure floor, as a fraction of the highest stage pressure.

At 1e-4 of the peak the residual gas contributes a relative refractivity of 1e-4 of the
fill -- nothing, for a gradient whose point is that the end is empty -- while the slope
of the steady-flow profile stays finite and cheap for the adaptive Chebyshev split to
resolve. Being relative rather than absolute, it means the same thing for a 0.8 bar
gradient as for a 40 bar one.
"""
MIN_STAGE_PRESSURE: Final = 1e-150
"""Smallest highest-stage pressure a fill accepts, bar.

The steady-flow profile is linear in ``p^2`` and its slope divides by ``p``, so every
pressure the fill evaluates -- the default floor of
:data:`DEFAULT_PRESSURE_FLOOR_FRACTION` times the highest stage included -- must square
to a normal float. ``(1e-4 * 1e-150)^2 = 1e-308`` is the smallest normal double; below
this the floored end squares to zero and the slope there is infinite. Found by
hypothesis: a fill from 0 to 6.5e-159 bar passed the old exact-zero guard and returned
an infinite derivative at the entrance.
"""

_DENSITY_SAMPLES: Final = 1024
"""Pressure samples used to tabulate the equation of state for :class:`DensityPath`.

The real-gas density is smooth and nearly linear in
pressure, so a cubic spline through 1024 points reproduces direct calls to better than
1e-12 relative, which the tests assert.
"""

_FD_RELATIVE_STEP: Final = 1e-5
"""Relative step of the Richardson difference used when no derivative is supplied."""

_BORE_DEGREES: Final = (16, 32, 64, 128, 256)
"""Chebyshev degrees tried in turn for the bore weight of a radius that is not linear.

A taper's weight ``a^-4`` is analytic on each piece between breakpoints, with its
nearest singularity where the continued radius would reach zero, so its Chebyshev
coefficients fall geometrically; a linear 125 um to 75 um piece converges by degree 16,
a 16 um to 5 um one by degree 64. A radius that has not converged by degree 256 has an
undeclared kink or a near-zero bore and is refused rather than integrated badly.
"""

_BORE_TOLERANCE: Final = 1e-13
"""Relative size of the last Chebyshev coefficients at which the bore weight converged.

The integral of the interpolant is then exact to about this relative accuracy, a few
hundred ulps, which is far below anything the fill's pressure could resolve.
"""

_WEIGHTED_DEGREES: Final = (*_BORE_DEGREES, 512)
"""Chebyshev degrees tried for a bore weight multiplied by a flow weight.

A temperature's flow weight ``eta(T) T`` is a Chebyshev series in ``T`` of degree up
to 65 over a gas's whole range (helium's viscosity law needed 64 on the former
data), so along a linear ``T(z)`` the product with ``a^-4`` can need a degree beyond
the bore's own.
"""

_TEMPERATURE_PROBES: Final = 65
"""Evenly spaced positions a temperature path is checked and ranged at.

The count the density paths of a mixture component are sampled at. Beside them every
breakpoint is sampled, so a piecewise-linear path's extremes are always found.
"""

_Z_DEGREES: Final = (8, 16, 32, 64)
"""Chebyshev degrees tried on each axis of the compressibility table.

The compressibility factor ``Z(p, T)`` of every built-in gas is analytic and close to
one over a fill: argon from 0 to 40 bar and 250 to 400 K converges to 1e-13 by degree
16 in pressure and 32 in temperature, 17 x 33 equation-of-state calls against the
1024 of the constant-temperature table.
"""

_Z_TOLERANCE: Final = 1e-13
"""Size of the last Chebyshev coefficients, relative to the largest, at which an axis
of the compressibility table has converged; the density then reproduces direct
equation-of-state calls to about 1e-12 relative, which the tests assert at 1e-9."""


@runtime_checkable
class ZProfile(Protocol):
    """A scalar path along the propagation axis.

    Implementations are frozen dataclasses. Every one is callable on an array of
    positions and returns an array of the same shape, so that it can be handed
    straight to :func:`solprop.translate.chebyshev.build_from_paths` as a path.
    """

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the positions in m where the path jumps or kinks, increasing."""
        ...

    @property
    def z_span(self) -> tuple[float, float]:
        """The interval in m over which the path is defined, as ``(start, end)``."""
        ...

    @property
    def derivative_is_analytic(self) -> bool:
        """Whether :meth:`derivative` is exact rather than a finite difference."""
        ...

    @property
    def is_constant(self) -> bool:
        """Whether the path takes the same value everywhere on its span."""
        ...

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the value of the path at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions along the waveguide in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The value at each position, shape ``(n_z,)``.
        """
        ...

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the derivative of the path with respect to z at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions along the waveguide in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            ``d(value)/dz`` per metre at each position, shape ``(n_z,)``.
        """
        ...


def _as_positions(z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
    """Validate a position argument and return it as a float array.

    Parameters
    ----------
    z : numpy.ndarray
        Positions along the waveguide in m, shape ``(n_z,)``.

    Returns
    -------
    numpy.ndarray
        The same values as a finite float array.

    Raises
    ------
    TranslationError
        If any position is not finite.
    """
    values = np.asarray(z, dtype=float)
    if not np.all(np.isfinite(values)):
        raise TranslationError("positions along z must be finite")
    return values


def _checked_edges(
    positions: Sequence[float] | np.ndarray, what: str
) -> Float[np.ndarray, " n_breakpoints"]:
    """Validate a sequence of stage positions and return it as an array.

    Parameters
    ----------
    positions : array_like
        Stage boundaries in m; at least two, finite and strictly increasing.
    what : str
        The name used in error messages.

    Returns
    -------
    numpy.ndarray
        The positions as a float array.

    Raises
    ------
    TranslationError
        If there are fewer than two, if any is not finite, or if they are not strictly
        increasing.
    """
    edges = np.asarray(positions, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise TranslationError(f"{what} needs at least two positions, got {edges.size}")
    if not np.all(np.isfinite(edges)):
        raise TranslationError(f"{what} positions must be finite")
    if np.any(np.diff(edges) <= 0.0):
        # An unsorted argument would silently interpolate the wrong stage.
        raise TranslationError(
            f"{what} positions must be strictly increasing, got {edges.tolist()}"
        )
    return edges


def _checked_pressures(
    edges: Float[np.ndarray, " n_positions"],
    pressures: Sequence[float] | np.ndarray,
) -> Float[np.ndarray, " n_pressures"]:
    """Validate the stage pressures of a fill and return them as an array.

    Parameters
    ----------
    edges : numpy.ndarray
        Stage boundaries in m, already validated.
    pressures : array_like
        Absolute pressure in bar at each boundary.

    Returns
    -------
    numpy.ndarray
        The pressures as a float array.

    Raises
    ------
    TranslationError
        If the lengths disagree, a pressure is negative or not finite, every pressure
        is zero, or the highest is so small that its square underflows.
    """
    values = np.asarray(pressures, dtype=float)
    if values.shape != edges.shape:
        raise TranslationError(
            f"PressureGradient needs one pressure per position, got {values.size} "
            f"pressures for {edges.size} positions"
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise TranslationError("stage pressures must be finite and not negative")
    highest = float(np.max(values))
    if highest <= 0.0:
        raise TranslationError(
            "a pressure fill needs at least one non-zero stage pressure"
        )
    if highest < MIN_STAGE_PRESSURE:
        # The steady-flow profile is linear in p^2 and its slope divides by p: the
        # floored end must square to a normal float, or the slope there is infinite.
        raise TranslationError(
            f"the highest stage pressure {highest} bar is too small for the "
            "steady-flow profile, which is linear in p^2 and divides by p; the floored "
            f"end must stay above {MIN_STAGE_PRESSURE:g} bar"
        )
    return values


@dataclass(frozen=True, eq=False)
class Constant:
    """A path that does not vary along z.

    Attributes
    ----------
    value : float
        The constant value, in whatever unit the consumer expects.
    span : tuple of float
        The interval in m the path is declared over. The default covers any
        propagation, since a constant needs no domain.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import Constant
    >>> radius = Constant(125e-6)
    >>> radius(np.array([0.0, 1.0]))
    array([0.000125, 0.000125])
    >>> radius.derivative(np.array([0.5]))
    array([0.])
    """

    value: float
    span: tuple[float, float] = (0.0, math.inf)

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the interval edges, which for a constant are none.

        Returns
        -------
        numpy.ndarray
            An empty array: a constant has no kinks.
        """
        return np.empty(0, dtype=float)

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the path is defined over.

        Returns
        -------
        tuple of float
            ``(start, end)`` in m.
        """
        return self.span

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether the derivative is exact.

        Returns
        -------
        bool
            Always ``True``: the derivative is identically zero.
        """
        return True

    @property
    def is_constant(self) -> bool:
        """Return whether the path is the same everywhere.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the constant value at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The value, broadcast to shape ``(n_z,)``.
        """
        return np.full_like(_as_positions(z), self.value)

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return zero at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            Zeros of shape ``(n_z,)``.
        """
        return np.zeros_like(_as_positions(z))


@dataclass(frozen=True, eq=False)
class PiecewiseLinear(Displayed):
    """A path linear between given positions and constant outside them.

    Every position is a breakpoint, because the slope jumps there. Clamping outside the
    declared range rather than extrapolating keeps the path bounded when the Chebyshev
    table is built slightly beyond the fibre end.

    Attributes
    ----------
    positions : numpy.ndarray
        Stage boundaries in m, strictly increasing, at least two.
    values : numpy.ndarray
        The value at each position.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import PiecewiseLinear
    >>> taper = PiecewiseLinear.from_stages([0.0, 1.0], [125e-6, 75e-6])
    >>> taper(np.array([0.0, 0.5, 1.0, 2.0])) * 1e6
    array([125., 100.,  75.,  75.])
    >>> taper.derivative(np.array([0.5])) * 1e6
    array([-50.])
    """

    positions: Float[np.ndarray, " n_positions"]
    values: Float[np.ndarray, " n_values"]

    def __repr__(self) -> str:
        """Return a one-line description with the arrays summarised, never dumped.

        Returns
        -------
        str
            For example ``PiecewiseLinear(3 points, 0 m to 1.5 m, values 1 to 0)``.
        """
        values = np.asarray(self.values, dtype=float)
        return line(
            "PiecewiseLinear",
            [
                (
                    "",
                    summarise(self.positions, "m")
                    .strip("<>")
                    .replace("values", "points"),
                ),
                ("values", f"{values[0]:.4g} to {values[-1]:.4g}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        pairs = zip(self.positions, self.values, strict=True)
        rows: list[tuple[str, object]] = [
            (engineering(float(z), "m"), f"{float(v):.6g}") for z, v in pairs
        ]
        shown = rows if len(rows) <= 8 else [*rows[:4], ("...", "..."), *rows[-3:]]
        return f"PiecewiseLinear: {len(rows)} points, constant outside them", shown

    @classmethod
    def from_stages(
        cls,
        positions: Sequence[float] | np.ndarray,
        values: Sequence[float] | np.ndarray,
    ) -> "PiecewiseLinear":  # noqa: UP037 - evaluated while the class body still runs
        """Build a path from any sequence of positions and values.

        Parameters
        ----------
        positions : array_like
            Stage boundaries in m, strictly increasing, at least two.
        values : array_like
            The value at each position.

        Returns
        -------
        PiecewiseLinear
            The path.

        Examples
        --------
        >>> from solprop.profiles import PiecewiseLinear
        >>> PiecewiseLinear.from_stages([0.0, 1.0], [1.0, 3.0]).values
        array([1., 3.])
        """
        return cls(np.asarray(positions, dtype=float), np.asarray(values, dtype=float))

    def __post_init__(self) -> None:
        """Validate and normalise the stage table.

        Raises
        ------
        TranslationError
            If the positions are not strictly increasing or the lengths disagree.
        """
        edges = _checked_edges(self.positions, "PiecewiseLinear")
        heights = np.asarray(self.values, dtype=float)
        if heights.shape != edges.shape:
            raise TranslationError(
                f"PiecewiseLinear needs one value per position, got {heights.size} "
                f"values for {edges.size} positions"
            )
        if not np.all(np.isfinite(heights)):
            raise TranslationError("PiecewiseLinear values must be finite")
        object.__setattr__(self, "positions", edges)
        object.__setattr__(self, "values", heights)

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the stage boundaries, every one of which is a kink.

        Returns
        -------
        numpy.ndarray
            The positions, in m.
        """
        return self.positions

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the stages cover.

        Returns
        -------
        tuple of float
            ``(first position, last position)`` in m.
        """
        return float(self.positions[0]), float(self.positions[-1])

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether the derivative is exact.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @property
    def is_constant(self) -> bool:
        """Return whether every stage carries the same value.

        Returns
        -------
        bool
            ``True`` when the path never varies.
        """
        return bool(np.all(self.values == self.values[0]))

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the interpolated value at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The value at each position, clamped outside the stage range.
        """
        return np.interp(_as_positions(z), self.positions, self.values)

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the slope at each position.

        The value at a breakpoint is the slope of the stage that starts there, and zero
        outside the declared range; the one-sided choice is arbitrary at a kink and the
        Chebyshev builder never samples one, because a breakpoint is an interval edge.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            ``d(value)/dz`` per metre, shape ``(n_z,)``.
        """
        positions = _as_positions(z)
        slopes = np.diff(self.values) / np.diff(self.positions)
        index = np.clip(
            np.searchsorted(self.positions, positions, side="right") - 1,
            0,
            slopes.size - 1,
        )
        inside = (positions >= self.positions[0]) & (positions <= self.positions[-1])
        return np.where(inside, slopes[index], 0.0)


@dataclass(frozen=True, eq=False)
class Linear(Displayed):
    """A path varying linearly between two ends and constant outside them.

    A two-stage :class:`PiecewiseLinear`, named for the common case of a taper.

    Attributes
    ----------
    start : float
        Value at ``z = 0``.
    end : float
        Value at ``z = length``.
    length : float
        Length of the varying section in m, > 0.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 75e-6, 1.5)
    >>> taper(np.array([0.0, 0.75, 1.5])) * 1e6
    array([125., 100.,  75.])
    """

    start: float
    end: float
    length: float
    _stages: PiecewiseLinear = field(init=False, repr=False)

    def __repr__(self) -> str:
        """Return a one-line description: the two end values and the length.

        Returns
        -------
        str
            For example ``Linear(1 to 0 over 1.5 m)``.
        """
        span = engineering(self.length, "m")
        return line("Linear", [("", f"{self.start:.6g} to {self.end:.6g} over {span}")])

    def __post_init__(self) -> None:
        """Build the equivalent two-stage table.

        Raises
        ------
        TranslationError
            If the length is not positive.
        """
        if not self.length > 0.0:
            raise TranslationError(f"Linear needs a positive length, got {self.length}")
        object.__setattr__(
            self,
            "_stages",
            PiecewiseLinear(
                np.array([0.0, float(self.length)]),
                np.array([float(self.start), float(self.end)]),
            ),
        )

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the two ends of the varying section.

        Returns
        -------
        numpy.ndarray
            ``[0, length]`` in m.
        """
        return self._stages.breakpoints

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the path varies over.

        Returns
        -------
        tuple of float
            ``(0, length)`` in m.
        """
        return self._stages.z_span

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether the derivative is exact.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @property
    def is_constant(self) -> bool:
        """Return whether the two ends carry the same value.

        Returns
        -------
        bool
            ``True`` when ``start == end``.
        """
        return self._stages.is_constant

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the interpolated value at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The value at each position, clamped outside ``[0, length]``.
        """
        return self._stages(z)

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the slope at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            ``(end - start) / length`` inside the section and zero outside.
        """
        return self._stages.derivative(z)


def _linear_bore(
    start: Float[np.ndarray, " n_z"],
    radius_start: Float[np.ndarray, " n_z"],
    z: Float[np.ndarray, " n_z"],
    radius_z: Float[np.ndarray, " n_z"],
) -> Float[np.ndarray, " n_z"]:
    """Integrate ``a^-4`` exactly from `start` to `z` on a piece where ``a`` is linear.

    Parameters
    ----------
    start : numpy.ndarray
        Left edge of each piece in m.
    radius_start : numpy.ndarray
        Radius at `start` in m, > 0.
    z : numpy.ndarray
        Upper limit of each integral in m, on the same piece.
    radius_z : numpy.ndarray
        Radius at `z` in m, > 0.

    Returns
    -------
    numpy.ndarray
        ``int_start^z a^-4 dz`` in m^-3.
    """
    # (a0^-3 - a^-3) / (3 a') rewritten with a' = (a - a0) / dz: the difference
    # cancels against the slope, leaving a form with no subtraction that is also right
    # on a piece of constant radius.
    return (
        (z - start)
        * (radius_start**2 + radius_start * radius_z + radius_z**2)
        / (3.0 * radius_start**3 * radius_z**3)
    )


@dataclass(frozen=True, eq=False)
class _BoreIntegral:
    """The cumulative hydraulic resistance ``int w a^-4 dz`` of a fill, by stage.

    The fill is cut at the union of its stage boundaries and the breakpoints of the
    radius and the flow weight ``w`` inside it, so both are smooth on every piece and
    no piece straddles a stage end. Each piece is integrated in closed form where the
    radius is linear on it and there is no flow weight, and otherwise through a
    converged Chebyshev interpolant of ``w a^-4``, integrated exactly. Integrals are
    accumulated from each stage's own start, never from the fill's, so a short stage
    after a long one loses no digits to a difference.

    Attributes
    ----------
    radius : ZProfile or None
        The core radius in m, or ``None`` for a uniform bore, whose ``a^-4`` is a
        constant that cancels from the fill.
    factor : ZProfile or None
        The flow weight ``w`` multiplying ``a^-4``, or ``None`` for none.
    edges : numpy.ndarray
        Piece boundaries in m, strictly increasing, the fill's breakpoints.
    stage_edges : numpy.ndarray
        The stage boundaries in m.
    piece_stage : numpy.ndarray
        The stage each piece belongs to.
    offsets : numpy.ndarray
        ``int w a^-4 dz`` from each piece's stage start to its left edge.
    stage_totals : numpy.ndarray
        ``int w a^-4 dz`` over each stage: in m^-3 times the unit of `factor`.
    series : tuple
        The exact antiderivative of each piece's interpolant, ``None`` on a piece
        integrated in closed form.
    """

    radius: ZProfile | None
    factor: ZProfile | None
    edges: Float[np.ndarray, " n_edges"]
    stage_edges: Float[np.ndarray, " n_positions"]
    piece_stage: Int[np.ndarray, " n_pieces"]
    offsets: Float[np.ndarray, " n_pieces"]
    stage_totals: Float[np.ndarray, " n_stages"]
    series: tuple[Chebyshev | None, ...]

    @classmethod
    def build(
        cls,
        radius: ZProfile | None,
        stage_edges: Float[np.ndarray, " n_positions"],
        factor: ZProfile | None = None,
    ) -> "_BoreIntegral":  # noqa: UP037 - evaluated while the class body still runs
        """Cut the fill into pieces and integrate the bore weight over each.

        Parameters
        ----------
        radius : ZProfile or None
            Core radius in m, positive over the fill; ``None`` for a uniform bore.
        stage_edges : numpy.ndarray
            Stage boundaries in m, validated and strictly increasing.
        factor : ZProfile or None, optional
            Flow weight multiplying ``a^-4``, positive over the fill.

        Returns
        -------
        _BoreIntegral
            The cumulative resistance.

        Raises
        ------
        TranslationError
            If the radius or the weight is not finite and positive over the fill, or
            their product does not converge on a piece by the highest degree tried.
        """
        start, end = float(stage_edges[0]), float(stage_edges[-1])
        kinks = [stage_edges]
        for path in (radius, factor):
            if path is not None:
                inner = np.asarray(path.breakpoints, dtype=float)
                kinks.append(inner[(inner > start) & (inner < end)])
        edges = np.unique(np.concatenate(kinks))
        probes = np.concatenate([edges, 0.5 * (edges[:-1] + edges[1:])])
        weight = _flow_weight(radius, factor)
        weight(probes)
        closed = factor is None and isinstance(
            radius, (Constant, Linear, PiecewiseLinear)
        )
        series: list[Chebyshev | None] = []
        pieces = np.empty(edges.size - 1)
        for k, (left, right) in enumerate(itertools.pairwise(edges)):
            if closed:
                assert radius is not None
                series.append(None)
                ends = radius(np.array([left, right]))
                pieces[k] = _linear_bore(
                    np.array([left]), ends[:1], np.array([right]), ends[1:]
                )[0]
            else:
                antiderivative = _bore_series(
                    weight,
                    float(left),
                    float(right),
                    _BORE_DEGREES if factor is None else _WEIGHTED_DEGREES,
                    repr(radius)
                    if factor is None
                    else f"{radius!r} weighted by {factor!r}",
                )
                series.append(antiderivative)
                pieces[k] = float(antiderivative(right))
        piece_stage = np.searchsorted(stage_edges, edges[:-1], side="right") - 1
        offsets = np.zeros_like(pieces)
        stage_totals = np.zeros(stage_edges.size - 1)
        for k, stage in enumerate(piece_stage):
            offsets[k] = stage_totals[stage]
            stage_totals[stage] += pieces[k]
        return cls(
            radius=radius,
            factor=factor,
            edges=edges,
            stage_edges=stage_edges,
            piece_stage=np.asarray(piece_stage),
            offsets=offsets,
            stage_totals=stage_totals,
            series=tuple(series),
        )

    def weight(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the integrand ``w a^-4`` at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, inside the fill.

        Returns
        -------
        numpy.ndarray
            ``w(z) a(z)^-4``; ``a(z)^-4`` in m^-4 without a flow weight.
        """
        if self.factor is None:
            assert self.radius is not None
            return np.asarray(self.radius(z), dtype=float) ** -4
        return _flow_weight(self.radius, self.factor)(z)

    def from_stage_start(
        self, z: Float[np.ndarray, " n_z"], stage: Int[np.ndarray, " n_z"]
    ) -> Float[np.ndarray, " n_z"]:
        """Return ``int w a^-4 dz`` from the start of each position's stage.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, each inside its stage.
        stage : numpy.ndarray
            The stage index of each position.

        Returns
        -------
        numpy.ndarray
            The resistance from the stage start to each position.
        """
        piece = np.clip(
            np.searchsorted(self.edges, z, side="right") - 1, 0, self.offsets.size - 1
        )
        # A position on a stage's upper boundary belongs to that stage but sits at the
        # left edge of the next stage's first piece: step back into its own stage.
        piece = np.where(self.piece_stage[piece] > stage, piece - 1, piece)
        left = self.edges[piece]
        result = self.offsets[piece].copy()
        if self.series[0] is None:
            assert self.radius is not None
            result += _linear_bore(left, self.radius(left), z, self.radius(z))
            return result
        for k in np.unique(piece):
            chosen = piece == k
            antiderivative = self.series[k]
            assert antiderivative is not None
            result[chosen] += antiderivative(z[chosen])
        return result

    def stage_fraction(
        self, z: Float[np.ndarray, " n_z"], stage: Int[np.ndarray, " n_z"]
    ) -> Float[np.ndarray, " n_z"]:
        """Return each position's share of its stage's resistance.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, each clamped into its stage.
        stage : numpy.ndarray
            The stage index of each position.

        Returns
        -------
        numpy.ndarray
            ``R(z_i, z) / R(z_i, z_{i+1})``, in [0, 1].
        """
        return np.clip(
            self.from_stage_start(z, stage) / self.stage_totals[stage], 0.0, 1.0
        )


def _checked_radius(radius: ZProfile, z: Float[np.ndarray, " n_z"]) -> None:
    """Refuse a radius that is not finite and positive at the given positions.

    Parameters
    ----------
    radius : ZProfile
        Core radius in m.
    z : numpy.ndarray
        Positions in m.

    Raises
    ------
    TranslationError
        If any sampled radius is not finite or not positive.
    """
    values = np.asarray(radius(z), dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise TranslationError(
            "the core radius a steady-flow fill follows must be finite and positive "
            "over the fill"
        )


def _flow_weight(
    radius: ZProfile | None, factor: ZProfile | None
) -> Callable[[Float[np.ndarray, " n_z"]], Float[np.ndarray, " n_z"]]:
    """Return the integrand ``w a^-4`` of the hydraulic resistance, checking both.

    Parameters
    ----------
    radius : ZProfile or None
        Core radius in m; ``None`` for a uniform bore, whose ``a^-4`` cancels.
    factor : ZProfile or None
        Flow weight ``w``; ``None`` for none.

    Returns
    -------
    callable
        ``z -> w(z) a(z)^-4``, raising :class:`~solprop.exceptions.TranslationError`
        where the radius or the weight is not finite and positive.
    """

    def weight(z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the integrand at the given positions, refusing a bad path.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            ``w(z) a(z)^-4``.
        """
        result = np.ones_like(np.asarray(z, dtype=float))
        if radius is not None:
            values = np.asarray(radius(z), dtype=float)
            _checked_radius(radius, z)
            result = values**-4
        if factor is not None:
            scale = np.asarray(factor(z), dtype=float)
            if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
                raise TranslationError(
                    "the flow weight a steady-flow fill follows must be finite and "
                    f"positive over the fill, and {factor!r} is not"
                )
            result = result * scale
        return result

    return weight


def _bore_series(
    weight: Callable[[Float[np.ndarray, " n_z"]], Float[np.ndarray, " n_z"]],
    left: float,
    right: float,
    degrees: tuple[int, ...],
    subject: str,
) -> Chebyshev:
    """Return the exact antiderivative of a converged interpolant of a bore weight.

    Parameters
    ----------
    weight : callable
        The integrand ``w a^-4``, smooth and positive on the piece; it refuses a
        radius or weight that is not.
    left, right : float
        The piece in m.
    degrees : tuple of int
        The Chebyshev degrees tried in turn.
    subject : str
        The radius's repr, or a description of the radius and weight, for the error.

    Returns
    -------
    numpy.polynomial.Chebyshev
        ``int_left^z w a^-4 dz`` as a Chebyshev series in ``z``, zero at `left`.

    Raises
    ------
    TranslationError
        If the interpolant has not converged by the highest degree tried.
    """
    for degree in degrees:
        interpolant = Chebyshev.interpolate(weight, degree, domain=[left, right])
        coefficients = np.abs(interpolant.coef)
        if np.max(coefficients[-3:]) <= _BORE_TOLERANCE * np.max(coefficients):
            return interpolant.integ(lbnd=left)
    raise TranslationError(
        f"the bore weight a^-4 of the radius {subject} did not converge by Chebyshev "
        f"degree {degrees[-1]} between z = {left:.6g} m and {right:.6g} m: a "
        "steady-flow fill needs a radius that is smooth between its declared "
        "breakpoints. Declare the kink as a breakpoint, or give the radius as a "
        "PiecewiseLinear table"
    )


def hydraulic_integral(
    radius: float | ZProfile,
    positions: Sequence[float] | np.ndarray,
    *,
    weight: float | ZProfile | None = None,
) -> Float[np.ndarray, " n_stages"]:
    """Return the hydraulic resistance ``int w a^-4 dz`` of each stage of a bore.

    The Poiseuille throughput of a stage is ``pi (p_high^2 - p_low^2) / (16 eta R)``
    with ``R`` this integral without a weight: ``length / a^4`` for a uniform bore.
    A gas whose temperature varies along the fibre takes the weight
    ``w = eta(T) T`` (:class:`PressureGradient`).

    Parameters
    ----------
    radius : float or ZProfile
        Core radius in m, > 0 over the stages.
    positions : array_like
        Stage boundaries in m, strictly increasing, at least two.
    weight : float, ZProfile or None, optional
        A flow weight multiplying ``a^-4``, > 0 over the stages, in any unit.
        ``None`` (the default) is no weight.

    Returns
    -------
    numpy.ndarray
        ``int_{z_i}^{z_{i+1}} w a^-4 dz`` in m^-3 times the unit of `weight`, one
        per stage.

    Raises
    ------
    TranslationError
        If the positions are invalid, or the radius or the weight is not positive
        or cannot be integrated.

    Examples
    --------
    >>> from solprop.profiles import Linear, hydraulic_integral
    >>> f"{hydraulic_integral(125e-6, [0.0, 1.0])[0]:.6e}"
    '4.096000e+15'
    >>> taper = hydraulic_integral(Linear(125e-6, 75e-6, 1.0), [0.0, 1.0])
    >>> f"{taper[0]:.6e}"
    '1.238914e+16'
    """
    edges = _checked_edges(positions, "hydraulic_integral")
    path = as_profile(radius)
    factor = None if weight is None else as_profile(weight)
    if factor is not None and factor.is_constant:
        scale = float(factor(edges[:1])[0])
        if not (math.isfinite(scale) and scale > 0.0):
            raise TranslationError(
                f"the flow weight must be finite and positive, got {scale}"
            )
        return scale * hydraulic_integral(path, edges)
    if path.is_constant and factor is None:
        value = float(path(edges[:1])[0])
        _checked_radius(path, edges[:1])
        return np.diff(edges) / value**4
    return _BoreIntegral.build(path, edges, factor).stage_totals.copy()


@dataclass(frozen=True, eq=False)
class PressureGradient(Displayed):
    r"""A steady-flow pressure fill, in bar, with every stage boundary a breakpoint.

    Laminar flow of a compressible gas through a narrow tube of constant bore makes
    :math:`p^2` linear in ``z``, so between two stage ends

    .. math:: p(z) = \sqrt{p_i^2 + \frac{z - z_i}{z_{i+1} - z_i}(p_{i+1}^2 - p_i^2)}

    evaluated as the convex combination :math:`\sqrt{(1-f)\,p_i^2 + f\,p_{i+1}^2}`,
    which cannot cancel, and the pressure is held constant outside the declared range.
    The derivative
    :math:`\mathrm{d}p/\mathrm{d}z = (p_{i+1}^2 - p_i^2)/(2\,\Delta z\,p)` is
    exact.

    Given a varying core radius :math:`a(z)`, the mass flow of an isothermal ideal gas
    in Poiseuille flow, :math:`\dot m \propto a^4\,p\,\mathrm{d}p/\mathrm{d}z`, is
    constant along each stage, so :math:`p^2` is linear in the hydraulic resistance
    instead (Pennetta et al., APL Photonics 4, 056105 (2019), eq. 6):

    .. math:: p^2 = (1 - f)\,p_i^2 + f\,p_{i+1}^2, \quad
       f(z) = \frac{R(z_i, z)}{R(z_i, z_{i+1})}, \quad
       R(u, v) = \int_u^v a^{-4}\,\mathrm{d}z,

    with :math:`\mathrm{d}p/\mathrm{d}z = (p_{i+1}^2 - p_i^2)\,a^{-4}/(2 R p)`, still
    analytic. Every stage is an independent flow between two gas cells. A constant or
    omitted radius is the uniform law above, bit for bit.

    A gas whose temperature varies along the fibre carries the mass flow
    :math:`\dot m \propto (p/T)\,a^4\,\mathrm{d}p/\mathrm{d}z/\eta(T)`, so the
    integrand of :math:`R` becomes :math:`w\,a^{-4}` with the flow weight
    :math:`w = \eta(T)\,T`, which :func:`solprop.gasflow.steady_fill` builds from the
    gas's viscosity; the fill itself needs only its variation, since a constant
    factor cancels from :math:`f`. A constant or omitted weight is the law above, bit
    for bit.

    A stage pressure of zero makes that derivative infinite and caps every integration
    scheme at about third order (``docs/dev/stepper.md`` section 10, trap 5). Such a
    pressure is raised to `floor` and the substitution is logged. Passing ``floor=0.0``
    keeps the true vacuum end, for reproducing a result that assumes one, and logs a
    stronger warning. A physical vacuum end is choked rather than floored: build the
    fill with :func:`solprop.gasflow.steady_fill`, which knows the gas.

    Attributes
    ----------
    positions : numpy.ndarray
        Stage boundaries in m, strictly increasing, at least two.
    pressures : numpy.ndarray
        Absolute pressure in bar at each boundary, >= 0, after any flooring.
    floor : float
        The pressure floor in bar that was applied.
    radius : ZProfile or None
        The core radius in m the flow follows, or ``None`` for a uniform bore. A
        constant radius is stored as ``None``: it is the uniform law exactly.
    choked_ends : tuple of bool
        Whether the first and the last end pressure is a choked pressure set by
        :func:`solprop.gasflow.steady_fill`, which depends on the gas; ``(False,
        False)`` for a fill whose ends are the declared ones.
    weight : ZProfile or None
        The flow weight multiplying ``a^-4`` in the resistance, positive, in any unit
        (``eta(T) T`` in Pa s K for a temperature path), or ``None`` for none. A
        constant weight is stored as ``None``: it cancels exactly.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import PressureGradient
    >>> fill = PressureGradient.two_point(1.5, 0.8, 0.0)
    >>> float(fill.floor)
    8e-05
    >>> [f"{value:.4f}" for value in fill(np.array([0.0, 0.75, 1.5]))]
    ['0.8000', '0.5657', '0.0001']
    >>> fill.breakpoints
    array([0. , 1.5])
    """

    positions: Float[np.ndarray, " n_positions"]
    pressures: Float[np.ndarray, " n_pressures"]
    floor: float
    radius: ZProfile | None = None
    choked_ends: tuple[bool, ...] = (False, False)
    weight: ZProfile | None = None
    _bore: _BoreIntegral | None = field(init=False, repr=False, default=None)

    def __repr__(self) -> str:
        """Return a one-line description: the end pressures and the length.

        Returns
        -------
        str
            For example ``PressureGradient(1.2 bar to 0 bar over 1 m, 2 points)``,
            with ``following the bore`` for a fill that knows a varying radius and
            ``weighted`` for one that carries a flow weight.
        """
        positions = np.asarray(self.positions, dtype=float)
        pressures = np.asarray(self.pressures, dtype=float)
        span = engineering(float(positions[-1] - positions[0]), "m")
        fields = [
            ("", f"{pressures[0]:.4g} bar to {pressures[-1]:.4g} bar over {span}"),
            ("", f"{positions.size} points"),
        ]
        if self.radius is not None:
            fields.append(("", "following the bore"))
        if self.weight is not None:
            fields.append(("", "weighted"))
        return line("PressureGradient", fields)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        pairs = zip(self.positions, self.pressures, strict=True)
        rows: list[tuple[str, object]] = [
            (engineering(float(z), "m"), f"{float(p):.6g} bar") for z, p in pairs
        ]
        rows.append(("floor", f"{self.floor:.3g} bar"))
        if any(self.choked_ends):
            ends = [
                n for n, c in zip(("first", "last"), self.choked_ends, strict=True) if c
            ]
            rows.append(("choked", " and ".join(ends) + " end"))
        title = "PressureGradient: steady flow between fixed pressures"
        if self.radius is not None:
            rows.append(("bore", repr(self.radius)))
            title += ", following the bore"
        if self.weight is not None:
            rows.append(("flow weight", repr(self.weight)))
            title += ", weighted"
        return title, rows

    def __post_init__(self) -> None:
        """Validate the stage table and integrate the bore of a varying radius.

        Raises
        ------
        TranslationError
            If the positions are not strictly increasing, the lengths disagree, a
            pressure is negative, no stage pressure is usable, or the radius is not
            positive or cannot be integrated.
        """
        edges = _checked_edges(self.positions, "PressureGradient")
        object.__setattr__(self, "positions", edges)
        object.__setattr__(self, "pressures", _checked_pressures(edges, self.pressures))
        chokes = tuple(bool(flag) for flag in self.choked_ends)
        if len(chokes) != 2:
            raise TranslationError("choked_ends needs one flag for each outer end")
        object.__setattr__(self, "choked_ends", chokes)
        # A constant bore or weight is the uniform law exactly, and taking today's
        # path keeps it bitwise and keeps the fill's identity unchanged.
        radius = None if self.radius is None else as_profile(self.radius)
        if radius is not None and radius.is_constant:
            radius = None
        weight = None if self.weight is None else as_profile(self.weight)
        if weight is not None and weight.is_constant:
            weight = None
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "weight", weight)
        if radius is None and weight is None:
            return
        object.__setattr__(self, "_bore", _BoreIntegral.build(radius, edges, weight))

    @classmethod
    def two_point(
        cls,
        length: float,
        start_bar: float,
        end_bar: float,
        *,
        floor: float | None = None,
        radius: float | ZProfile | None = None,
        weight: float | ZProfile | None = None,
    ) -> "PressureGradient":  # noqa: UP037 - evaluated while the class body still runs
        """Build a single-stage fill from one end pressure to the other.

        Parameters
        ----------
        length : float
            Length of the fill in m, > 0.
        start_bar, end_bar : float
            Absolute pressures in bar at ``z = 0`` and ``z = length``, >= 0.
        floor : float or None, optional
            Smallest pressure in bar any stage end may take. ``None`` uses
            :data:`DEFAULT_PRESSURE_FLOOR_FRACTION` of the highest stage pressure.
        radius : float, ZProfile or None, optional
            Core radius in m, > 0, which the flow follows; ``None`` or a constant is
            the uniform bore.
        weight : float, ZProfile or None, optional
            Flow weight multiplying ``a^-4``, > 0 (the class docstring); ``None`` or
            a constant is none.

        Returns
        -------
        PressureGradient
            The fill.

        Raises
        ------
        TranslationError
            If the length is not positive, or a pressure is negative.

        Examples
        --------
        >>> from solprop.profiles import PressureGradient
        >>> PressureGradient.two_point(1.0, 2.0, 2.0).pressures
        array([2., 2.])
        """
        if not length > 0.0:
            raise TranslationError(
                f"PressureGradient needs a positive length, got {length}"
            )
        return cls.from_stages(
            [0.0, float(length)],
            [start_bar, end_bar],
            floor=floor,
            radius=radius,
            weight=weight,
        )

    @classmethod
    def from_stages(
        cls,
        positions: Sequence[float] | np.ndarray,
        pressures: Sequence[float] | np.ndarray,
        *,
        floor: float | None = None,
        radius: float | ZProfile | None = None,
        weight: float | ZProfile | None = None,
    ) -> "PressureGradient":  # noqa: UP037 - evaluated while the class body still runs
        """Build a multi-stage fill, applying and logging the vacuum-end floor.

        Parameters
        ----------
        positions : array_like
            Stage boundaries in m, strictly increasing, at least two.
        pressures : array_like
            Absolute pressure in bar at each boundary, >= 0.
        floor : float or None, optional
            Smallest pressure in bar any stage end may take. ``None`` uses
            :data:`DEFAULT_PRESSURE_FLOOR_FRACTION` of the highest stage pressure;
            ``0.0`` keeps a true vacuum end and logs that the order of the scheme is
            then capped.
        radius : float, ZProfile or None, optional
            Core radius in m, > 0, which the flow follows (the class docstring);
            ``None`` or a constant is the uniform bore. A radius that is neither
            constant nor linear between its breakpoints is integrated by Chebyshev
            interpolation and refused if that does not converge.
        weight : float, ZProfile or None, optional
            Flow weight multiplying ``a^-4``, > 0 (the class docstring), integrated
            with the bore by Chebyshev interpolation; ``None`` or a constant is none.

        Returns
        -------
        PressureGradient
            The fill, with any pressure below `floor` raised to it.

        Raises
        ------
        TranslationError
            If the stage table is invalid, `floor` is negative, every stage pressure is
            zero, the highest is too small for its square to be representable, or the
            radius is not positive or cannot be integrated.

        Examples
        --------
        >>> from solprop.profiles import PressureGradient
        >>> fill = PressureGradient.from_stages([0.0, 0.6, 1.0], [0.0, 5.0, 0.0])
        >>> fill.pressures
        array([5.e-04, 5.e+00, 5.e-04])
        """
        edges = _checked_edges(positions, "PressureGradient")
        stage_pressures = _checked_pressures(edges, pressures)
        highest = float(np.max(stage_pressures))
        if floor is None:
            applied = DEFAULT_PRESSURE_FLOOR_FRACTION * highest
        elif floor < 0.0:
            raise TranslationError(
                f"the pressure floor must not be negative, got {floor}"
            )
        else:
            applied = float(floor)
        floored = np.maximum(stage_pressures, applied)
        raised = ~np.isclose(floored, stage_pressures, rtol=0.0, atol=0.0)
        if applied == 0.0 and np.any(stage_pressures == 0.0):
            logger.warning(
                "pressure fill reaches exactly zero at z = %s m and floor=0 was "
                "requested: dp/dz diverges there, which caps every integration scheme "
                "at about third order however small the step "
                "(docs/dev/stepper.md section 10, trap 5)",
                edges[stage_pressures == 0.0].tolist(),
            )
        elif np.any(raised):
            logger.warning(
                "pressure fill raised from %s bar to the floor %.3e bar at z = %s m: "
                "dp/dz diverges at a true vacuum end, which caps every integration "
                "scheme at about third order (docs/dev/stepper.md section 10, trap 5). "
                "Pass floor= to choose another value.",
                stage_pressures[raised].tolist(),
                applied,
                edges[raised].tolist(),
            )
        return cls(
            positions=edges,
            pressures=floored,
            floor=applied,
            radius=None if radius is None else as_profile(radius),
            weight=None if weight is None else as_profile(weight),
        )

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the stage boundaries and the bore's kinks, every one a hard kink.

        Returns
        -------
        numpy.ndarray
            The positions in m: the stage boundaries, joined for a fill that follows
            a varying bore or carries a flow weight by their breakpoints inside the
            fill, where the slope of the pressure kinks with theirs.
        """
        if self._bore is None:
            return self.positions
        return self._bore.edges

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the fill covers.

        Returns
        -------
        tuple of float
            ``(first position, last position)`` in m.
        """
        return float(self.positions[0]), float(self.positions[-1])

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether the derivative is exact.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True

    @property
    def is_constant(self) -> bool:
        """Return whether every stage carries the same pressure.

        Returns
        -------
        bool
            ``True`` when the fill never varies.
        """
        return bool(np.all(self.pressures == self.pressures[0]))

    def _stage_index(self, z: Float[np.ndarray, " n_z"]) -> Int[np.ndarray, " n_z"]:
        """Return the index of the stage each position falls in.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            Stage indices, clamped to the declared range.
        """
        return np.asarray(
            np.clip(
                np.searchsorted(self.positions, z, side="right") - 1,
                0,
                self.positions.size - 2,
            )
        )

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the pressure in bar at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            Absolute pressure in bar, clamped to the end pressures outside the fill.
        """
        positions = _as_positions(z)
        index = self._stage_index(positions)
        lower, upper = self.positions[index], self.positions[index + 1]
        squared_lower = self.pressures[index] ** 2
        squared_upper = self.pressures[index + 1] ** 2
        if self._bore is None:
            fraction = np.clip((positions - lower) / (upper - lower), 0.0, 1.0)
        else:
            fraction = self._bore.stage_fraction(
                np.clip(positions, lower, upper), index
            )
        # The convex combination rather than p_i^2 + f (p_{i+1}^2 - p_i^2): that form
        # cancels catastrophically as f -> 1 when the fill runs into a floor far below
        # its start, and can return a pressure below the floor.
        return np.sqrt((1.0 - fraction) * squared_lower + fraction * squared_upper)

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the pressure gradient in bar per metre at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            ``dp/dz = (p_next^2 - p^2) / (2 dz p)`` on a uniform bore and
            ``(p_next^2 - p^2) w a^-4 / (2 R p)`` on a varying one or with a flow
            weight ``w``, and zero outside
            the fill. It is infinite at a stage end held at exactly zero pressure,
            which is why the constructor floors one by default.
        """
        positions = _as_positions(z)
        index = self._stage_index(positions)
        lower, upper = self.positions[index], self.positions[index + 1]
        difference = self.pressures[index + 1] ** 2 - self.pressures[index] ** 2
        inside = (positions >= self.positions[0]) & (positions <= self.positions[-1])
        if self._bore is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                slope = difference / ((upper - lower) * 2.0 * self(positions))
            return np.where(inside, slope, 0.0)
        # The derivative needs the weight w a^-4 itself, never the radius's slope, so
        # it stays analytic for any radius and weight the fill can integrate.
        weight = self._bore.weight(np.clip(positions, lower, upper))
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (
                difference
                * weight
                / (2.0 * self._bore.stage_totals[index] * self(positions))
            )
        return np.where(inside, slope, 0.0)


def _chebyshev_nodes(degree: int) -> Float[np.ndarray, " n_nodes"]:
    """Return the Chebyshev points of the first kind on ``[-1, 1]``.

    Parameters
    ----------
    degree : int
        The interpolation degree; ``degree + 1`` points, none at an end.

    Returns
    -------
    numpy.ndarray
        The points, decreasing.
    """
    return np.cos(np.pi * (np.arange(degree + 1) + 0.5) / (degree + 1))


@dataclass(frozen=True, eq=False)
class _Compressibility:
    """The compressibility factor ``Z(p, T)`` of a gas as a tensor Chebyshev series.

    ``Z = p / (rho k_B T)`` is analytic and close to one over the states a fill takes,
    so a product of Chebyshev series in pressure and temperature converges fast and
    carries exact partial derivatives; the density ``p / (k_B T Z)`` then follows with
    its ideal-gas part exact. Nodes are Chebyshev points of the first kind, which never
    fall on a zero-pressure end, where the equation of state is undefined.

    Attributes
    ----------
    pressures : tuple of float
        The pressure interval in bar.
    temperatures : tuple of float
        The temperature interval in K.
    coefficients : numpy.ndarray
        The series coefficients, shape ``(n_p + 1, n_T + 1)``.
    """

    pressures: tuple[float, float]
    temperatures: tuple[float, float]
    coefficients: Float[np.ndarray, "n_p n_temperature"]

    @classmethod
    def build(
        cls,
        density_at: Callable[[float, float], float],
        pressures: tuple[float, float],
        temperatures: tuple[float, float],
        what: str,
    ) -> "_Compressibility":  # noqa: UP037 - evaluated while the class body still runs
        """Tabulate ``Z`` at Chebyshev nodes, raising each axis's degree to convergence.

        Parameters
        ----------
        density_at : callable
            ``(pressure in bar, temperature in K) -> number density in m^-3``.
        pressures : tuple of float
            The pressure interval in bar, ``low < high``.
        temperatures : tuple of float
            The temperature interval in K, ``low < high``.
        what : str
            The gas, for the error.

        Returns
        -------
        _Compressibility
            The converged series.

        Raises
        ------
        MaterialError
            If an axis has not converged by the highest degree tried.
        """
        degrees = [_Z_DEGREES[0], _Z_DEGREES[0]]
        while True:
            coefficients = cls._fit(density_at, pressures, temperatures, degrees)
            scale = float(np.max(np.abs(coefficients)))
            tails = (
                float(np.max(np.abs(coefficients[-3:, :]))),
                float(np.max(np.abs(coefficients[:, -3:]))),
            )
            unresolved = [
                axis for axis, tail in enumerate(tails) if tail > _Z_TOLERANCE * scale
            ]
            if not unresolved:
                return cls(pressures, temperatures, coefficients)
            for axis in unresolved:
                position = _Z_DEGREES.index(degrees[axis])
                if position + 1 == len(_Z_DEGREES):
                    raise MaterialError(
                        f"the compressibility factor of {what} over {pressures[0]:.6g} "
                        f"to {pressures[1]:.6g} bar and {temperatures[0]:.6g} to "
                        f"{temperatures[1]:.6g} K did not converge by Chebyshev degree "
                        f"{_Z_DEGREES[-1]}: the equation of state is not smooth there "
                        "(a phase boundary?)"
                    )
                degrees[axis] = _Z_DEGREES[position + 1]

    @staticmethod
    def _fit(
        density_at: Callable[[float, float], float],
        pressures: tuple[float, float],
        temperatures: tuple[float, float],
        degrees: Sequence[int],
    ) -> Float[np.ndarray, "n_p n_temperature"]:
        """Return the tensor Chebyshev coefficients of ``Z`` at the given degrees.

        Parameters
        ----------
        density_at : callable
            ``(pressure in bar, temperature in K) -> number density in m^-3``.
        pressures, temperatures : tuple of float
            The intervals in bar and K.
        degrees : sequence of int
            The degree in pressure and in temperature.

        Returns
        -------
        numpy.ndarray
            Coefficients, shape ``(degrees[0] + 1, degrees[1] + 1)``.
        """
        unit_p = _chebyshev_nodes(degrees[0])
        unit_t = _chebyshev_nodes(degrees[1])
        nodes_p = (
            0.5 * (pressures[0] + pressures[1])
            + 0.5 * (pressures[1] - pressures[0]) * unit_p
        )
        nodes_t = (
            0.5 * (temperatures[0] + temperatures[1])
            + 0.5 * (temperatures[1] - temperatures[0]) * unit_t
        )
        values = np.array(
            [
                [p * BAR / (K_B * t * density_at(float(p), float(t))) for t in nodes_t]
                for p in nodes_p
            ]
        )
        from_p = np.linalg.solve(
            np.polynomial.chebyshev.chebvander(unit_p, degrees[0]), values
        )
        return np.linalg.solve(
            np.polynomial.chebyshev.chebvander(unit_t, degrees[1]), from_p.T
        ).T

    def _unit(
        self,
        pressure: Float[np.ndarray, " n_z"],
        temperature: Float[np.ndarray, " n_z"],
    ) -> tuple[Float[np.ndarray, " n_z"], Float[np.ndarray, " n_z"]]:
        """Map states onto the series' unit square.

        Parameters
        ----------
        pressure : numpy.ndarray
            Pressures in bar.
        temperature : numpy.ndarray
            Temperatures in K.

        Returns
        -------
        tuple of numpy.ndarray
            The two coordinates on ``[-1, 1]``.
        """
        (p0, p1), (t0, t1) = self.pressures, self.temperatures
        return (2.0 * pressure - (p0 + p1)) / (p1 - p0), (
            2.0 * temperature - (t0 + t1)
        ) / (t1 - t0)

    def covers(
        self,
        pressure: Float[np.ndarray, " n_z"],
        temperature: Float[np.ndarray, " n_z"],
    ) -> Bool[np.ndarray, " n_z"]:
        """Return which states lie inside the tabulated rectangle.

        Parameters
        ----------
        pressure : numpy.ndarray
            Pressures in bar.
        temperature : numpy.ndarray
            Temperatures in K.

        Returns
        -------
        numpy.ndarray
            ``True`` where the series interpolates.
        """
        (p0, p1), (t0, t1) = self.pressures, self.temperatures
        return (
            (pressure >= p0)
            & (pressure <= p1)
            & (temperature >= t0)
            & (temperature <= t1)
        )

    def values(
        self,
        pressure: Float[np.ndarray, " n_z"],
        temperature: Float[np.ndarray, " n_z"],
    ) -> tuple[
        Float[np.ndarray, " n_z"], Float[np.ndarray, " n_z"], Float[np.ndarray, " n_z"]
    ]:
        """Return ``Z`` and its partial derivatives at each state.

        Parameters
        ----------
        pressure : numpy.ndarray
            Pressures in bar, inside the table.
        temperature : numpy.ndarray
            Temperatures in K, inside the table.

        Returns
        -------
        tuple of numpy.ndarray
            ``Z``, ``dZ/dp`` in 1/bar and ``dZ/dT`` in 1/K.
        """
        chebyshev = np.polynomial.chebyshev
        x, y = self._unit(pressure, temperature)
        (p0, p1), (t0, t1) = self.pressures, self.temperatures
        along_p = chebyshev.chebder(self.coefficients, axis=0) * (2.0 / (p1 - p0))
        along_t = chebyshev.chebder(self.coefficients, axis=1) * (2.0 / (t1 - t0))
        return (
            chebyshev.chebval2d(x, y, self.coefficients),
            chebyshev.chebval2d(x, y, along_p),
            chebyshev.chebval2d(x, y, along_t),
        )


@dataclass(frozen=True, eq=False)
class DensityPath(Displayed):
    """The number density along z of a gas following a pressure path.

    The equation of state is a scalar call into optmat (CoolProp) that costs about a
    millisecond, and the Chebyshev builder samples a path thousands of times, so the
    density is tabulated once over the pressure range the path spans and interpolated
    with a cubic spline. The spline's derivative is exact, so
    ``d(rho)/dz = rho'(p) dp/dz`` stays analytic whenever the pressure path is.

    A temperature that varies along z makes the density ``rho(p(z), T(z))``. The
    compressibility factor ``Z(p, T) = p / (rho k_B T)`` is then tabulated as a
    tensor Chebyshev series over the rectangle of pressures and temperatures the two
    paths span, converged to 1e-13 on each axis (a few hundred equation-of-state
    calls, fewer than the constant-temperature spline's 1024), and
    ``rho = p / (k_B T Z)`` keeps its ideal-gas part exact. The derivative is
    ``rho_p p' + rho_T T'`` with the series' exact partial derivatives. A constant
    temperature, a number or a constant path, takes the spline above, bit for bit.

    Attributes
    ----------
    gas : Gas or str or Species
        The species: a built-in gas's name takes optmat's equation of state, a
        species its own.
    pressure : ZProfile
        The pressure path in bar.
    temperature : float or ZProfile
        Temperature in K, > 0: a number, or a path along z. A constant path is
        stored as its value.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.physdata import Gas
    >>> from solprop.profiles import DensityPath, Linear, PressureGradient
    >>> fill = PressureGradient.two_point(1.0, 1.0, 0.5)
    >>> path = DensityPath(Gas.He, fill)
    >>> f"{float(path(np.array([0.0]))[0]):.6e}"
    '2.469537e+25'
    >>> bool(path.derivative(np.array([0.5]))[0] < 0.0)
    True
    >>> warm = DensityPath(Gas.He, fill, Linear(293.15, 350.0, 1.0))
    >>> f"{float(warm(np.array([0.0]))[0]):.6e}"
    '2.469537e+25'
    """

    gas: Gas | str | Species
    pressure: ZProfile
    temperature: float | ZProfile = ROOM_TEMPERATURE
    _spline: CubicSpline | None = field(init=False, repr=False, default=None)
    _table: _Compressibility | None = field(init=False, repr=False, default=None)

    def __repr__(self) -> str:
        """Return a one-line description: the gas, its pressure path, its temperature.

        Returns
        -------
        str
            For example ``DensityPath(He, Constant(value=1.0, span=None), 293.15 K)``.
        """
        gas = getattr(self.gas, "name", None) or str(self.gas)
        return line(
            "DensityPath",
            [
                ("", gas),
                ("", repr(self.pressure)),
                ("", _temperature_text(self.temperature)),
            ],
        )

    def __post_init__(self) -> None:
        """Tabulate the equation of state over the pressure range of the path.

        Raises
        ------
        MaterialError
            If the gas is unknown, the temperature is not positive, or the
            compressibility table of a temperature path does not converge.
        """
        species = self.gas if isinstance(self.gas, Species) else as_gas(self.gas)
        object.__setattr__(self, "gas", species)
        temperature = _checked_temperature(self.temperature, "a density path")
        object.__setattr__(self, "temperature", temperature)
        start, end = self.pressure.z_span
        # The steady-flow profile is monotonic on each stage, so the extremes over the
        # whole path are attained at the stage boundaries; sampling the interior as
        # well costs nothing and covers a path that is not. A constant fill declares an
        # unbounded span, where there is nothing to sample between the ends.
        interior = (
            np.linspace(start, end, 65)
            if math.isfinite(start) and math.isfinite(end)
            else np.array([start])
        )
        probes = np.concatenate(
            [np.asarray(self.pressure.breakpoints, dtype=float), interior]
        )
        boundaries = np.unique(
            np.concatenate(
                (
                    self.pressure.breakpoints,
                    np.array([start, end]) if math.isfinite(end) else np.array([start]),
                )
            )
        )
        if np.any(self.pressure(boundaries) < 0.0):
            raise MaterialError(
                "pressure must not be negative at declared profile boundaries"
            )
        if isinstance(temperature, ZProfile):
            self._tabulate(species, temperature)
            return
        values = np.maximum(self.pressure(probes), 0.0)
        lowest, highest = float(np.min(values)), float(np.max(values))
        if highest == lowest:
            highest = lowest * (1.0 + 1e-9) + 1e-12
        grid = np.linspace(lowest, highest, _DENSITY_SAMPLES)
        densities = np.array(
            [
                species.equation_of_state.density(float(value), temperature)
                if isinstance(species, Species)
                else density(species, float(value), temperature)
                for value in grid
            ]
        )
        object.__setattr__(self, "_spline", CubicSpline(grid, densities))

    def _tabulate(self, species: Gas | Species, temperature: ZProfile) -> None:
        """Tabulate the compressibility over the states a temperature path spans.

        Parameters
        ----------
        species : Gas or Species
            The resolved species.
        temperature : ZProfile
            The varying temperature path in K.
        """
        probes = _probe_positions(self)
        pressures = np.maximum(self.pressure(probes), 0.0)
        low_p, high_p = float(np.min(pressures)), float(np.max(pressures))
        low_t, high_t = _temperature_range(temperature, probes)
        # A degenerate axis (a constant pressure beside a varying temperature) is
        # widened by a relative hair, as the spline's grid is: the series is then of
        # low degree along it and its derivative there is never multiplied by a
        # nonzero slope.
        if high_p == low_p:
            high_p = low_p * (1.0 + 1e-9) + 1e-12
        if high_t == low_t:
            high_t = low_t * (1.0 + 1e-9)
        table = _Compressibility.build(
            self._density_at,
            (low_p, high_p),
            (low_t, high_t),
            getattr(species, "name", str(species)),
        )
        object.__setattr__(self, "_table", table)

    def _density_at(self, pressure: float, temperature: float) -> float:
        """Return the equation of state's density at one state.

        Parameters
        ----------
        pressure : float
            Pressure in bar, non-negative.
        temperature : float
            Temperature in K, > 0.

        Returns
        -------
        float
            Number density in m^-3.
        """
        species = self.gas
        if isinstance(species, Species):
            return float(species.equation_of_state.density(pressure, temperature))
        return float(density(species, pressure, temperature))

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the breakpoints of the pressure path, and of a temperature path.

        Returns
        -------
        numpy.ndarray
            The positions in m; the density kinks exactly where the pressure or the
            temperature does.
        """
        if not isinstance(self.temperature, ZProfile):
            return self.pressure.breakpoints
        start, end = self.z_span
        inner = np.asarray(self.temperature.breakpoints, dtype=float)
        return np.unique(
            np.concatenate(
                [
                    np.asarray(self.pressure.breakpoints, dtype=float),
                    inner[(inner >= start) & (inner <= end)],
                ]
            )
        )

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the interval the path covers.

        Returns
        -------
        tuple of float
            ``(start, end)`` in m, from the pressure path; from a temperature path
            where the pressure's is unbounded.
        """
        span = self.pressure.z_span
        if isinstance(self.temperature, ZProfile) and not math.isfinite(span[1]):
            return self.temperature.z_span
        return span

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether the derivative is exact.

        Returns
        -------
        bool
            ``True`` when the pressure path's derivative is, and a temperature path's:
            the tables contribute exact derivatives of their interpolants.
        """
        if isinstance(self.temperature, ZProfile):
            return (
                self.pressure.derivative_is_analytic
                and self.temperature.derivative_is_analytic
            )
        return self.pressure.derivative_is_analytic

    @property
    def is_constant(self) -> bool:
        """Return whether the density never varies.

        Returns
        -------
        bool
            ``True`` when the pressure never varies and neither does the temperature,
            or the pressure is identically zero.
        """
        if isinstance(self.temperature, ZProfile):
            return self.pressure.is_constant and not np.any(
                self.pressure(np.array([self.z_span[0]])) > 0.0
            )
        return self.pressure.is_constant

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the number density at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            Number density in m^-3, shape ``(n_z,)``.
        """
        positions = np.clip(_as_positions(z), *self.z_span)
        pressures = np.maximum(self.pressure(positions), 0.0)
        if self._table is not None:
            return self._along_temperature(positions, pressures)[0]
        assert self._spline is not None
        values = np.maximum(np.asarray(self._spline(self._within(pressures))), 0.0)
        outside = ~self._covered(pressures)
        if np.any(outside):
            # A pressure the construction probes never saw: an extremum of the path
            # narrower than the probe spacing. Extrapolating the spline there is
            # unbounded (a constant fill's knots nearly coincide), so evaluate the
            # equation of state itself, as ``Mixture.at_parameters`` does.
            values = values.copy()
            values[outside] = [self._density(float(p)) for p in pressures[outside]]
            self._warn_outside(outside)
        # The spline through vacuum can carry a sub-ulp residual. A zero population
        # is an exact zero, including a component deliberately retained for updates.
        return np.where(pressures == 0.0, 0.0, values)

    def _warn_outside(self, outside: Bool[np.ndarray, " n_z"]) -> None:
        """Log that some states lay outside the table built at construction.

        Parameters
        ----------
        outside : numpy.ndarray
            Which positions were evaluated directly.
        """
        if self._table is None:
            assert self._spline is not None
            extent = f"{self._spline.x[0]:.6g} to {self._spline.x[-1]:.6g} bar"
        else:
            (p0, p1), (t0, t1) = self._table.pressures, self._table.temperatures
            extent = f"{p0:.6g} to {p1:.6g} bar and {t0:.6g} to {t1:.6g} K"
        logger.warning(
            "density path of %s: %d of %d positions have states outside the range "
            "sampled at construction (%s); the equation of state was evaluated there "
            "directly. Declare the extreme as a breakpoint of the profile to keep the "
            "table's arithmetic.",
            getattr(self.gas, "name", self.gas),
            int(np.count_nonzero(outside)),
            outside.size,
            extent,
        )

    def _along_temperature(
        self, positions: Float[np.ndarray, " n_z"], pressures: Float[np.ndarray, " n_z"]
    ) -> tuple[Float[np.ndarray, " n_z"], Float[np.ndarray, " n_z"]]:
        """Return the density and its z derivative under a temperature path.

        Parameters
        ----------
        positions : numpy.ndarray
            Positions in m, inside the span.
        pressures : numpy.ndarray
            The pressure in bar at each, non-negative.

        Returns
        -------
        tuple of numpy.ndarray
            ``rho`` in m^-3 and ``rho_p p' + rho_T T'`` in m^-4.
        """
        table = self._table
        temperature = self.temperature
        assert table is not None and isinstance(temperature, ZProfile)
        temperatures = np.asarray(temperature(positions), dtype=float)
        inside = table.covers(pressures, temperatures)
        held_p = np.clip(pressures, *table.pressures)
        held_t = np.clip(temperatures, *table.temperatures)
        factor, factor_p, factor_t = table.values(held_p, held_t)
        ideal = BAR / (K_B * held_t * factor)
        values = pressures * BAR / (K_B * temperatures * factor)
        # rho = p / (k_B T Z): d(rho)/dp = (1 - p Z_p / Z) / (k_B T Z), finite at p = 0,
        # and d(rho)/dT = -rho (1/T + Z_T / Z). Outside the table the slope is held at
        # its nearer edge, as the constant-temperature spline's is.
        slope_p = ideal * (1.0 - held_p * factor_p / factor)
        slope_t = -values * (1.0 / held_t + factor_t / factor)
        if np.any(~inside):
            values = values.copy()
            values[~inside] = [
                self._density_at(float(p), float(t))
                for p, t in zip(pressures[~inside], temperatures[~inside], strict=True)
            ]
            self._warn_outside(~inside)
        derivative = slope_p * self.pressure.derivative(
            positions
        ) + slope_t * temperature.derivative(positions)
        return np.where(pressures == 0.0, 0.0, values), derivative

    def _covered(
        self, pressures: Float[np.ndarray, " n_z"]
    ) -> Bool[np.ndarray, " n_z"]:
        """Return which pressures lie within the range the spline was sampled on.

        Parameters
        ----------
        pressures : numpy.ndarray
            Pressures in bar, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            ``True`` where the spline interpolates rather than extrapolates.
        """
        assert self._spline is not None
        x = self._spline.x
        return (pressures >= x[0]) & (pressures <= x[-1])

    def _within(
        self, pressures: Float[np.ndarray, " n_z"]
    ) -> Float[np.ndarray, " n_z"]:
        """Return the pressures clamped to the spline's sampled range.

        Parameters
        ----------
        pressures : numpy.ndarray
            Pressures in bar, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The same values, held at the range's ends outside it.
        """
        assert self._spline is not None
        x = self._spline.x
        return np.clip(pressures, x[0], x[-1])

    def _density(self, pressure: float) -> float:
        """Return the equation of state's density at one pressure.

        Parameters
        ----------
        pressure : float
            Pressure in bar, non-negative.

        Returns
        -------
        float
            Number density in m^-3.
        """
        species = self.gas
        assert not isinstance(self.temperature, ZProfile)
        if isinstance(species, Species):
            return float(species.equation_of_state.density(pressure, self.temperature))
        return float(density(species, pressure, self.temperature))

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the density gradient at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            ``d(rho)/dz`` in m^-3 per metre, by the chain rule through the pressure
            path, and through a temperature path ``rho_p p' + rho_T T'``.
        """
        original = _as_positions(z)
        positions = np.clip(original, *self.z_span)
        values = self.pressure(positions)
        if self._table is not None:
            derivative = self._along_temperature(positions, np.maximum(values, 0.0))[1]
            return np.where((original != positions) | (values < 0.0), 0.0, derivative)
        assert self._spline is not None
        # Outside the sampled range the spline's slope is held at the nearer end:
        # the compressibility varies slowly, and __call__ has already warned there.
        derivative = np.asarray(
            self._spline(self._within(np.maximum(values, 0.0)), 1)
        ) * self.pressure.derivative(positions)
        return np.where((original != positions) | (values < 0.0), 0.0, derivative)


@dataclass(frozen=True, eq=False)
class SeriesOfPath(Displayed):
    """A smooth function of another path, ``f(g(z))``, with ``f`` a Chebyshev series.

    The steady-flow fill of a gas whose temperature varies along the fibre is weighted
    by ``eta(T) T`` (:class:`PressureGradient`): :func:`solprop.gasflow.steady_fill`
    represents that weight as a converged Chebyshev series in the temperature,
    evaluated on the temperature path. The derivative ``f'(g) g'`` is analytic
    whenever the inner path's is.

    Attributes
    ----------
    path : ZProfile
        The inner path ``g``, in whatever unit `series` takes.
    series : numpy.polynomial.Chebyshev
        The outer function ``f``, with its domain in the unit of `path`; outside the
        domain it extrapolates, so it should cover the path's range.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.polynomial import Chebyshev
    >>> from solprop.profiles import Linear, SeriesOfPath
    >>> square = Chebyshev.fit([1.0, 2.0, 3.0], [1.0, 4.0, 9.0], 2, domain=[1.0, 3.0])
    >>> weight = SeriesOfPath(Linear(1.0, 3.0, 2.0), square)
    >>> [round(float(v), 12) for v in weight(np.array([0.0, 1.0]))]
    [1.0, 4.0]
    >>> round(float(weight.derivative(np.array([1.0]))[0]), 12)
    4.0
    """

    path: ZProfile
    series: Chebyshev

    def __repr__(self) -> str:
        """Return a one-line description: the degree and the inner path.

        Returns
        -------
        str
            For example ``SeriesOfPath(degree 16 of Linear(300 to 400 over 1 m))``.
        """
        return line(
            "SeriesOfPath", [("", f"degree {self.series.degree()} of {self.path!r}")]
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        low, high = (float(v) for v in self.series.domain)
        return "SeriesOfPath: a Chebyshev series of a path", [
            ("path", repr(self.path)),
            ("degree", self.series.degree()),
            ("domain", f"{low:.6g} to {high:.6g}"),
        ]

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the inner path's breakpoints.

        Returns
        -------
        numpy.ndarray
            The positions in m; a smooth function kinks where its argument does.
        """
        return self.path.breakpoints

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the inner path's interval.

        Returns
        -------
        tuple of float
            ``(start, end)`` in m.
        """
        return self.path.z_span

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether the derivative is exact.

        Returns
        -------
        bool
            ``True`` when the inner path's is: the series has an exact derivative.
        """
        return self.path.derivative_is_analytic

    @property
    def is_constant(self) -> bool:
        """Return whether the inner path is constant.

        Returns
        -------
        bool
            ``True`` when the argument never varies.
        """
        return self.path.is_constant

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return ``f(g(z))`` at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The values, shape ``(n_z,)``.
        """
        return np.asarray(self.series(self.path(_as_positions(z))), dtype=float)

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return ``f'(g(z)) g'(z)`` at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The slope per metre, shape ``(n_z,)``.
        """
        positions = _as_positions(z)
        slope = np.asarray(self.series.deriv()(self.path(positions)), dtype=float)
        return slope * self.path.derivative(positions)


def _probe_positions(path: ZProfile) -> Float[np.ndarray, " n_probes"]:
    """Return a path's breakpoints and evenly spaced positions over its span.

    Parameters
    ----------
    path : ZProfile
        Any path.

    Returns
    -------
    numpy.ndarray
        Sorted unique positions in m: the breakpoints, and
        :data:`_TEMPERATURE_PROBES` positions over a finite span (its start alone
        for an unbounded one).
    """
    start, end = path.z_span
    interior = (
        np.linspace(start, end, _TEMPERATURE_PROBES)
        if math.isfinite(start) and math.isfinite(end)
        else np.array([start])
    )
    return np.unique(
        np.concatenate([np.asarray(path.breakpoints, dtype=float), interior])
    )


def _checked_temperature(value: object, owner: str = "") -> float | ZProfile:
    """Validate a gas temperature, a number or a path of it along z, in K.

    A path that is constant is returned as the number it holds, so that every
    consumer takes its constant-temperature code, bit for bit.

    Parameters
    ----------
    value : object
        A temperature in K, finite and > 0: a real number, or a ``ZProfile`` finite
        and positive at its breakpoints and at evenly spaced probes over its span.
    owner : str, optional
        What carries the temperature, for the message.

    Returns
    -------
    float or ZProfile
        A Python number as given, a NumPy scalar as a float, a constant path as its
        value, a varying path itself.

    Raises
    ------
    MaterialError
        If `value` is a boolean, neither a real number nor a ``ZProfile``, or not
        finite and positive where it is sampled.
    """
    where = f" of {owner}" if owner else ""
    if isinstance(value, ZProfile):
        probes = _probe_positions(value)
        samples = np.asarray(value(probes), dtype=float)
        bad = ~np.isfinite(samples) | (samples <= 0.0)
        if np.any(bad):
            first = int(np.argmax(bad))
            raise MaterialError(
                f"the temperature{where} must be finite and positive in K at every "
                f"position; the path reaches {samples[first]:g} at "
                f"z = {probes[first]:g} m"
            )
        if value.is_constant:
            return float(samples[0])
        return value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if isinstance(value, bool | np.bool_) or not isinstance(value, numbers.Real):
        raise MaterialError(
            f"the temperature{where} must be a number in K or a ZProfile of it, got "
            f"{type(value).__name__}"
        )
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise MaterialError(
            f"the temperature{where} must be finite and positive, got {value} K"
        )
    # A Python number is kept as given, so a run record states what was declared.
    return cast("float", value) if type(value) in (float, int) else number


def _same_temperature(given: float | ZProfile, own: float | ZProfile) -> bool:
    """Return whether a declared temperature is a mixture's own.

    Parameters
    ----------
    given : float or ZProfile
        A validated temperature in K (a constant path already its number).
    own : float or ZProfile
        The mixture's.

    Returns
    -------
    bool
        ``True`` for the same path object, or two equal numbers.
    """
    if isinstance(given, ZProfile) or isinstance(own, ZProfile):
        return given is own
    return float(given) == float(own)


def _temperature_text(temperature: float | ZProfile) -> str:
    """Return a temperature for printing: ``'293.15 K'`` or the path's own ``repr``.

    Parameters
    ----------
    temperature : float or ZProfile
        A temperature in K, or a path of it along z.

    Returns
    -------
    str
        The text.
    """
    if isinstance(temperature, ZProfile):
        return f"{temperature!r} K"
    return f"{temperature:g} K"


def _temperature_range(
    temperature: ZProfile, positions: Float[np.ndarray, " n_z"]
) -> tuple[float, float]:
    """Return the lowest and highest temperature a path takes at given positions.

    Parameters
    ----------
    temperature : ZProfile
        The temperature path in K.
    positions : numpy.ndarray
        Positions in m, the path's breakpoints among them.

    Returns
    -------
    tuple of float
        ``(lowest, highest)`` in K.
    """
    values = np.asarray(temperature(positions), dtype=float)
    return float(np.min(values)), float(np.max(values))


@dataclass(frozen=True, eq=False)
class Scaled:
    """Multiply a path and its derivative by a constant.

    Attributes
    ----------
    path : ZProfile
        Underlying path, with its units and endpoint continuation.
    factor : float
        Finite multiplier; the output units include those of this multiplier.
    """

    path: ZProfile
    factor: float

    def __post_init__(self) -> None:
        """Reject a non-finite multiplier."""
        if not math.isfinite(self.factor):
            raise TranslationError("path scale factor must be finite")

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the underlying breakpoints in m.

        Returns
        -------
        numpy.ndarray
            Positions of derivative discontinuities.
        """
        return self.path.breakpoints

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the underlying span in m.

        Returns
        -------
        tuple of float
            Start and end positions.
        """
        return self.path.z_span

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether the underlying derivative is analytic.

        Returns
        -------
        bool
            The underlying path's declaration.
        """
        return self.path.derivative_is_analytic

    @property
    def is_constant(self) -> bool:
        """Return whether the scaled path is constant.

        Returns
        -------
        bool
            True for a constant path or a zero multiplier.
        """
        return self.factor == 0.0 or self.path.is_constant

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Evaluate the scaled path.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Scaled values, with the input shape.
        """
        return self.path(z) * self.factor

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Evaluate the scaled derivative.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Scaled derivatives per metre, with the input shape.
        """
        return self.path.derivative(z) * self.factor


@dataclass(frozen=True, eq=False)
class Sum:
    """Add compatible paths in their declared order.

    Attributes
    ----------
    paths : tuple of ZProfile
        Non-empty sequence of paths with compatible units. Their spans are joined;
        outside an individual span that path supplies its endpoint continuation.
    """

    paths: tuple[ZProfile, ...]

    def __post_init__(self) -> None:
        """Require at least one path and retain an immutable sequence."""
        if not self.paths:
            raise TranslationError("Sum needs at least one path")
        object.__setattr__(self, "paths", tuple(self.paths))

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the union of the component breakpoints in m.

        Returns
        -------
        numpy.ndarray
            Sorted unique positions.
        """
        return np.unique(np.concatenate([p.breakpoints for p in self.paths]))

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the union of the component spans in m.

        Returns
        -------
        tuple of float
            Earliest start and latest end.
        """
        return min(p.z_span[0] for p in self.paths), max(
            p.z_span[1] for p in self.paths
        )

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether every component derivative is analytic.

        Returns
        -------
        bool
            Conjunction of the component declarations.
        """
        return all(p.derivative_is_analytic for p in self.paths)

    @property
    def is_constant(self) -> bool:
        """Return whether every component is constant.

        Returns
        -------
        bool
            Conjunction of the component declarations.
        """
        return all(p.is_constant for p in self.paths)

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Evaluate the sum in component order.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Summed values with the input shape.
        """
        result = self.paths[0](z)
        for path in self.paths[1:]:
            result = result + path(z)
        return result

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Evaluate the derivative sum in component order.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Summed derivatives per metre with the input shape.
        """
        result = self.paths[0].derivative(z)
        for path in self.paths[1:]:
            result = result + path.derivative(z)
        return result


@dataclass(frozen=True, eq=False)
class FromCallable:
    """An arbitrary vectorised path, with its breakpoints declared by the caller.

    The escape hatch for a measured or exotic profile. Because the adaptive Chebyshev
    split finds smooth structure but cannot see a kink it was not told about, the
    breakpoints are a required argument rather than something guessed.

    With no `slope` the derivative is a Richardson-extrapolated central difference,
    ``derivative_is_analytic`` is ``False``, and a caller that needs an analytic
    ``beta_1`` (ADR-006) must supply one.

    Attributes
    ----------
    func : collections.abc.Callable
        The path, vectorised over an array of positions in m.
    span : tuple of float
        The interval in m the path is defined over, as ``(start, end)``.
    kinks : numpy.ndarray
        Positions in m where the path jumps or kinks, strictly increasing. May be
        empty.
    slope : collections.abc.Callable or None
        The analytic derivative, if there is one.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import FromCallable
    >>> path = FromCallable(np.sin, (0.0, 1.0), slope=np.cos)
    >>> path.derivative_is_analytic
    True
    >>> bool(abs(path.derivative(np.array([0.3]))[0] - np.cos(0.3)) < 1e-15)
    True
    """

    func: Callable[[Float[np.ndarray, " n_z"]], Float[np.ndarray, " n_z"]]
    span: tuple[float, float]
    kinks: Float[np.ndarray, " n_kinks"] = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    slope: Callable[[Float[np.ndarray, " n_z"]], Float[np.ndarray, " n_z"]] | None = (
        None
    )

    def __post_init__(self) -> None:
        """Validate the declared span and breakpoints.

        Raises
        ------
        TranslationError
            If the span is empty or a breakpoint lies outside it.
        """
        start, end = float(self.span[0]), float(self.span[1])
        if not end > start:
            raise TranslationError(f"FromCallable needs end > start, got {self.span}")
        edges = np.asarray(self.kinks, dtype=float)
        if edges.size and (
            np.any(np.diff(edges) <= 0.0) or edges[0] < start or edges[-1] > end
        ):
            raise TranslationError(
                "FromCallable breakpoints must be strictly increasing and inside the "
                f"span {self.span}, got {edges.tolist()}"
            )
        object.__setattr__(self, "span", (start, end))
        object.__setattr__(self, "kinks", edges)

    @property
    def breakpoints(self) -> Float[np.ndarray, " n_breakpoints"]:
        """Return the declared breakpoints.

        Returns
        -------
        numpy.ndarray
            The positions in m.
        """
        return self.kinks

    @property
    def z_span(self) -> tuple[float, float]:
        """Return the declared span.

        Returns
        -------
        tuple of float
            ``(start, end)`` in m.
        """
        return self.span

    @property
    def derivative_is_analytic(self) -> bool:
        """Return whether the derivative is exact.

        Returns
        -------
        bool
            ``True`` only when a `slope` was supplied.
        """
        return self.slope is not None

    @property
    def is_constant(self) -> bool:
        """Return whether the path is known to be constant.

        Returns
        -------
        bool
            Always ``False``: an arbitrary callable cannot be inspected, and the
            conservative answer only costs a re-evaluation.
        """
        return False

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the value of the wrapped callable at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The value at each position, shape ``(n_z,)``.
        """
        return np.asarray(self.func(_as_positions(z)), dtype=float)

    def derivative(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Return the derivative at each position.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape ``(n_z,)``.

        Returns
        -------
        numpy.ndarray
            The analytic derivative when one was supplied, otherwise a
            Richardson-extrapolated central difference, whose relative accuracy is
            about 1e-10 for a smooth path and no better.
        """
        positions = _as_positions(z)
        if self.slope is not None:
            return np.asarray(self.slope(positions), dtype=float)
        start, end = self.span
        step = _FD_RELATIVE_STEP * max(
            end - start, float(np.max(np.abs(positions))), 1.0
        )
        wide = (self(positions + step) - self(positions - step)) / (2.0 * step)
        half = step / 2.0
        narrow = (self(positions + half) - self(positions - half)) / (2.0 * half)
        # Richardson extrapolation of two central differences cancels the h^2
        # truncation term and leaves a fourth-order estimate, the same construction the
        # material models are tested with.
        return (4.0 * narrow - wide) / 3.0


def as_profile(
    value: float | ZProfile, span: tuple[float, float] = (0.0, math.inf)
) -> ZProfile:
    """Coerce a number to a constant path, and pass a path through unchanged.

    Parameters
    ----------
    value : float or ZProfile
        A constant value, or a path already.
    span : tuple of float, optional
        The interval in m given to a :class:`Constant` built from a number.

    Returns
    -------
    ZProfile
        The path.

    Raises
    ------
    TranslationError
        If `value` is neither a real number nor a path.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.profiles import as_profile
    >>> as_profile(125e-6)(np.array([0.0, 2.0])) * 1e6
    array([125., 125.])
    """
    if isinstance(value, ZProfile):
        return value
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return Constant(float(value), span)
    raise TranslationError(
        f"expected a number or a ZProfile, got {type(value).__name__}"
    )


_IDENTITY_ARRAY_LIMIT: Final = 16
"""Longest array a profile identity writes out value by value.

A two-stage taper or a four-stage fill is more useful read than digested, while a
measured 1200-knot path is not: ``numpy``'s own ``repr`` summarises anything over
1000 elements, so two physically different long paths print identically and compare
equal. Above this length the identity carries the SHA-256
of the float64 bytes instead, which separates them exactly.
"""


def _number_identity(value: float) -> str:
    """Return the exact hexadecimal literal of a double.

    Parameters
    ----------
    value : float
        Any float, including an infinity.

    Returns
    -------
    str
        ``float.hex`` of the value: exact, round-trippable and independent of
        ``numpy.set_printoptions``.
    """
    return float(value).hex()


def _array_identity(values: Float[np.ndarray, " n_values"]) -> dict[str, object]:
    """Return the exact identity of a float array, by value or by digest.

    Parameters
    ----------
    values : numpy.ndarray
        Finite or infinite float values, shape ``(n_values,)``.

    Returns
    -------
    dict
        ``{'n': size, 'values': [hex, ...]}`` for a short array, otherwise
        ``{'n': size, 'sha256': digest}`` of the contiguous float64 bytes.
    """
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.size <= _IDENTITY_ARRAY_LIMIT:
        return {"n": int(array.size), "values": [_number_identity(v) for v in array]}
    return {"n": int(array.size), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


def profile_identity(profile: float | ZProfile) -> dict[str, object]:
    """Return the structural identity of a path, exact to the last bit.

    Two paths give the same identity exactly when they are the same class built
    from the same numbers. Unlike ``repr``, the result does not depend on
    ``numpy.set_printoptions`` and does not summarise a long array, so a 1200-knot
    :class:`PiecewiseLinear` is distinguished from any other. Every value is a
    string, an integer or a nested mapping of them, so the result is JSON-able and
    can go straight into a provider's ``identity``.

    Parameters
    ----------
    profile : float or ZProfile
        A constant value, or any path declared in this module.

    Returns
    -------
    dict
        ``{'kind': <class name>, ...}`` with the defining numbers as exact
        hexadecimal float literals and long arrays as SHA-256 digests.

    Raises
    ------
    TranslationError
        If the path is an arbitrary callable (:class:`FromCallable`) or an
        implementation this module does not know: neither has a structural
        identity, and a ``repr`` is not one.

    Examples
    --------
    >>> from solprop.profiles import Linear, profile_identity
    >>> profile_identity(Linear(125e-6, 75e-6, 1.5))["kind"]
    'Linear'
    >>> profile_identity(2.0) == profile_identity(2.0)
    True
    """
    path = as_profile(profile)
    identity: dict[str, object] | None = None
    if isinstance(path, Constant):
        identity = {
            "value": _number_identity(path.value),
            "span": [_number_identity(v) for v in path.span],
        }
    elif isinstance(path, PiecewiseLinear):
        identity = {
            "positions": _array_identity(path.positions),
            "values": _array_identity(path.values),
        }
    elif isinstance(path, Linear):
        identity = {
            "start": _number_identity(path.start),
            "end": _number_identity(path.end),
            "length": _number_identity(path.length),
        }
    elif isinstance(path, PressureGradient):
        identity = {
            "positions": _array_identity(path.positions),
            "pressures": _array_identity(path.pressures),
            "floor": _number_identity(path.floor),
        }
        if path.radius is not None:
            # Only a varying bore is recorded: a constant one is stored as None, so a
            # uniform fill keeps the identity it had before fills knew the bore.
            identity["radius"] = profile_identity(path.radius)
        if path.weight is not None:
            identity["weight"] = profile_identity(path.weight)
        if any(path.choked_ends):
            identity["choked_ends"] = [str(flag) for flag in path.choked_ends]
    elif isinstance(path, DensityPath):
        identity = {
            "gas": getattr(path.gas, "name", None) or str(path.gas),
            # A constant temperature is stored as its number, so a density path keeps
            # the identity it had before temperatures could vary along z.
            "temperature": profile_identity(path.temperature)
            if isinstance(path.temperature, ZProfile)
            else _number_identity(path.temperature),
            "pressure": profile_identity(path.pressure),
        }
    elif isinstance(path, SeriesOfPath):
        identity = {
            "path": profile_identity(path.path),
            "coefficients": _array_identity(path.series.coef),
            "domain": [_number_identity(v) for v in path.series.domain],
            "window": [_number_identity(v) for v in path.series.window],
        }
    elif isinstance(path, Scaled):
        identity = {
            "factor": _number_identity(path.factor),
            "path": profile_identity(path.path),
        }
    elif isinstance(path, Sum):
        identity = {"paths": [profile_identity(p) for p in path.paths]}
    if identity is None:
        raise TranslationError(
            f"{type(path).__name__} has no structural identity: an arbitrary callable "
            "cannot be identified by its values, and its repr is not an identity. "
            "Declare the path as one of "
            "Constant, PiecewiseLinear, Linear, PressureGradient, DensityPath, "
            "SeriesOfPath, Scaled or Sum."
        )
    return {"kind": type(path).__name__, **identity}
