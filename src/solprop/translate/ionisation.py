# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ionisation rates as converged, clipped, log-spaced spline tables.

Evaluating a rate model directly -- optmat's, through
:class:`solprop.nonlinear.optmat_rates.OptmatRate` -- costs of order a microsecond per
sample, special functions per point, per stage. A propagation
evaluates the rate once per sample of the fine time grid on every column of every stage,
so the rate has to be a table. This module builds it.

Three properties of the table matter:

* **Log-spaced knots.** ``log W`` is nearly linear in ``log E`` at the low-field turn-on
  and flattens at high field, so uniform knots in ``log E`` put the samples where the
  curvature is and let one table span four decades. Knots uniform in ``E`` over the
  same span reach only about one per cent.
* **A converged knot count.** The count doubles until the interpolant meets the
  requested relative tolerance against direct evaluation everywhere, rather than being
  fixed in advance.
* **Clipping rather than throwing.** A rate that raised above its maximum field would
  stop a run in the middle of a propagation. Here the table is exactly
  zero below ``field_min`` -- which the plasma loss term ``Ip W (1-f) / E`` relies on --
  and holds its last value above ``field_max``, so a diverged step cannot extrapolate a
  cubic in ``log W`` into an overflow (``docs/dev/stepper.md`` section 10, trap 9).

The manual page is ``docs/guide/media/ionisation.md``; the kernel's view of the table is
``docs/dev/kernel-contract.md`` section 5.
"""

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from typing import Final

import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike
from scipy import interpolate

from solprop.exceptions import TranslationError
from solprop.nonlinear.ionisation import (
    IonisationRate,
    RateConvention,
    rate_convention,
    rate_species,
)

__all__ = [
    "DEFAULT_INTERVALS",
    "DEFAULT_RTOL",
    "MAX_KNOTS",
    "SCHEMA_VERSION",
    "RateTable",
    "build_rate_table",
    "clear_rate_table_cache",
    "rate_table_key",
]

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final = 2
"""Version of the table layout, part of the cache key.

Bump it when the knot placement, the spline convention or the clipping changes, so that
a table built by an older version can never be mistaken for one built by this one.
"""

DEFAULT_RTOL: Final = 1e-6
"""Default bound on the relative error of the table against direct evaluation."""

DEFAULT_INTERVALS: Final = 256
"""Knot intervals tried first; the count doubles until the tolerance is met."""

MAX_KNOTS: Final = 1 << 20
"""Knot budget before the build gives up.

A million knots over four decades is a spacing of 1e-5 in ``log E``; needing more means
the rate is not smooth, which for these models means a kink, not a fine feature.
"""

_CLIP_SCAN_POINTS: Final = 4096
"""Samples used to find where a rate reaches its own ceiling inside the field range."""

_PROBE_OFFSETS: Final = (0.25, 0.5, 0.75)
"""Where inside each interval the interpolant is probed, as a fraction of its width.

A cubic interpolant is exact at its knots, so the error has to be measured between them;
it peaks near the middle of an interval and the quarter points catch the shoulders.
"""

_SPLINE_DEGREE: Final = 4
"""Coefficients per interval of a cubic spline, as SciPy stores them: four."""

_TABLES: dict[tuple[object, ...], RateTable] = {}
"""Tables already built in this process, keyed by :func:`rate_table_key`."""


@dataclass(frozen=True)
class RateTable:
    """A cubic-spline table of ``log W`` against ``log E``, on uniform log-spaced knots.

    Knots are uniform in ``log E``, so locating one is arithmetic rather than a search:
    ``i = floor((log|E| - log_field_min) / d_log_field)``. On interval ``i`` the value
    is ``sum_j coefficients[j, i] t^(3-j)`` with ``t = log|E| - log_field_min - i
    d_log_field``, which is how :class:`scipy.interpolate.CubicSpline` stores it. The
    coefficients rather than a spline object are the table, because they are what
    crosses into the kernel contract.

    Attributes
    ----------
    log_field_min : float
        Natural logarithm of ``field_min``, with the field in V/m.
    d_log_field : float
        Knot spacing in ``log E``, positive.
    log_rate : numpy.ndarray
        Natural logarithm of the rate at each knot, shape ``(n_table,)``, s^-1.
        Diagnostic: evaluation goes through the coefficients.
    coefficients : numpy.ndarray
        Spline coefficients, shape ``(n_deg, n_intervals)``, highest power first.
    field_min : float
        Field amplitude in V/m below which the table is exactly zero.
    field_max : float
        Field amplitude in V/m above which the table holds ``rate_max``.
    rate_max : float
        The rate at ``field_max``, s^-1, and the value returned above it.
    error : float
        The largest relative deviation from direct evaluation measured during the build.
    key : str
        A digest of the rate's parameters and the build settings, stable across
        processes. Names a cache entry; two tables with the same key are the same table.
    convention : str
        Optical-cycle convention retained from the source rate. Legacy manually
        constructed tables default to ``'unspecified'``.
    species_identity : str or None
        Chemical species name, when declared by the source rate.
    ionisation_potential : float or None
        First ionisation potential in J, when declared by the source rate.

    Raises
    ------
    TranslationError
        If the knot spacing is not positive, the coefficient array does not have four
        rows and one column per interval, the field range is not increasing, or anything
        stored is not finite.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.translate.ionisation import build_rate_table
    >>> table = build_rate_table(OptmatRate("Ar", 800e-9), (1e9, 5e10), rtol=1e-4)
    >>> float(table(1e6))
    0.0
    >>> abs(float(table(3e10)) / float(OptmatRate("Ar", 800e-9)(3e10)) - 1.0) < 1e-4
    True
    """

    log_rate: Float[np.ndarray, " n_table"]
    coefficients: Float[np.ndarray, " n_deg n_intervals"]
    log_field_min: float
    d_log_field: float
    field_min: float
    field_max: float
    rate_max: float
    error: float
    key: str
    convention: RateConvention = "unspecified"
    species_identity: str | None = None
    ionisation_potential: float | None = None

    def __post_init__(self) -> None:
        """Validate what the shape annotations cannot express.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        if self.convention not in (
            "instantaneous-isotropic",
            "cycle-averaged",
            "unspecified",
        ):
            raise TranslationError("invalid rate-table optical-cycle convention")
        if self.ionisation_potential is not None and not (
            math.isfinite(self.ionisation_potential) and self.ionisation_potential > 0
        ):
            raise TranslationError(
                "rate-table ionisation potential must be finite and positive"
            )
        if self.species_identity is not None and not self.species_identity.strip():
            raise TranslationError("rate-table species identity must not be empty")
        if not self.d_log_field > 0.0:
            msg = f"d_log_field must be positive, got {self.d_log_field!r}"
            raise TranslationError(msg)
        if self.coefficients.shape[0] != _SPLINE_DEGREE:
            msg = (
                f"coefficients must have {_SPLINE_DEGREE} rows, one per power of a "
                f"cubic, got {self.coefficients.shape[0]}"
            )
            raise TranslationError(msg)
        if self.coefficients.shape[1] != self.log_rate.size - 1:
            msg = (
                f"coefficients must have one column per interval: "
                f"{self.log_rate.size - 1} expected, got {self.coefficients.shape[1]}"
            )
            raise TranslationError(msg)
        if not 0.0 < self.field_min < self.field_max:
            msg = (
                "field_min and field_max must satisfy 0 < field_min < field_max, got "
                f"({self.field_min!r}, {self.field_max!r})"
            )
            raise TranslationError(msg)
        if not (
            np.all(np.isfinite(self.log_rate))
            and np.all(np.isfinite(self.coefficients))
            and math.isfinite(self.rate_max)
        ):
            msg = "the table must be finite everywhere; a rate underflowed or diverged"
            raise TranslationError(msg)

    @property
    def n_knots(self) -> int:
        """Return the number of knots.

        Returns
        -------
        int
            The knot count, one more than the number of intervals.
        """
        return int(self.log_rate.size)

    @property
    def log_field(self) -> Float[np.ndarray, " n_table"]:
        """Return the knot positions in ``log E``.

        Returns
        -------
        numpy.ndarray
            Uniformly spaced logarithms of the field amplitude, shape ``(n_table,)``.
        """
        return self.log_field_min + self.d_log_field * np.arange(self.n_knots)

    @property
    def field(self) -> Float[np.ndarray, " n_table"]:
        """Return the knot positions as field amplitudes.

        Returns
        -------
        numpy.ndarray
            Geometrically spaced field amplitudes in V/m, shape ``(n_table,)``.
        """
        return np.exp(self.log_field)

    def log_evaluate(
        self, log_field: Float[np.ndarray, " n_probe"]
    ) -> Float[np.ndarray, " n_probe"]:
        """Return the interpolated ``log W`` at logarithmic field amplitudes.

        This is the interpolant itself, without the clipping and without the
        exponential. The build measures its error here rather than on the rate, because
        a difference of logarithms *is* the relative error and does not underflow where
        the rate is 1e-300.

        Parameters
        ----------
        log_field : numpy.ndarray
            Natural logarithms of field amplitudes in V/m, shape ``(n_probe,)``, inside
            ``[log(field_min), log(field_max)]``.

        Returns
        -------
        numpy.ndarray
            Interpolated ``log W`` with the rate in s^-1, shape ``(n_probe,)``.
        """
        index = np.clip(
            ((log_field - self.log_field_min) / self.d_log_field).astype(np.intp),
            0,
            self.coefficients.shape[1] - 1,
        )
        local = log_field - (self.log_field_min + index * self.d_log_field)
        c = self.coefficients
        return ((c[0, index] * local + c[1, index]) * local + c[2, index]) * local + c[
            3, index
        ]

    @property
    def plasma_energy(self) -> float | None:
        """Return the energy per ionisation event the source rate declared, in J.

        A table keeps the species and the energy of the rate it tabulates, so that a
        plasma response built on the table pays the same energy as one built on the
        rate (:func:`solprop.nonlinear.ionisation.bind_rate_energy`, SEP 3, D8).

        Returns
        -------
        float or None
            :attr:`ionisation_potential` when the source rate named its species and
            energy; ``None`` otherwise.
        """
        if self.species_identity is None:
            return None
        return self.ionisation_potential

    def __call__(
        self, field_amplitude: ArrayLike | float
    ) -> Float[np.ndarray, " *shape"]:
        """Return the tabulated rate for a field amplitude.

        Parameters
        ----------
        field_amplitude : array_like or float
            Field amplitude in V/m; the sign is ignored, as for the rate itself.

        Returns
        -------
        numpy.ndarray
            Rate per atom in s^-1, with the shape of ``field_amplitude``. Exactly zero
            below :attr:`field_min` and equal to :attr:`rate_max` above
            :attr:`field_max`.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.optmat_rates import OptmatRate
        >>> from solprop.translate.ionisation import build_rate_table
        >>> table = build_rate_table(OptmatRate("He", 800e-9), (2e9, 1e11), rtol=1e-4)
        >>> np.round(np.log10(table(np.array([5e10, 1e11]))), 3).tolist()
        [11.207, 14.237]
        """
        field = np.asarray(np.abs(np.asarray(field_amplitude, dtype=np.float64)))
        flat = np.atleast_1d(field).ravel()
        rate = np.zeros_like(flat)
        above = flat > self.field_max
        inside = (flat >= self.field_min) & ~above
        if bool(np.any(inside)):
            rate[inside] = np.exp(self.log_evaluate(np.log(flat[inside])))
        rate[above] = self.rate_max
        return rate.reshape(field.shape)


def rate_table_key(
    rate: IonisationRate, field_range: tuple[float, float], rtol: float
) -> tuple[object, ...]:
    """Return the deterministic cache key of a table.

    Every setting that changes the table appears once, in name order, so the key cannot
    depend on the order in which the rate's keyword arguments were written: the same
    settings in a different order give the same cache file.

    Parameters
    ----------
    rate : IonisationRate
        The rate to tabulate.
    field_range : tuple of float
        Lower and upper field amplitude in V/m.
    rtol : float
        Requested relative tolerance.

    Returns
    -------
    tuple
        A hashable key, beginning with the schema version.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.translate.ionisation import rate_table_key
    >>> first = OptmatRate("Ar", 800e-9)
    >>> second = OptmatRate("Ar", 800e-9)
    >>> rate_table_key(first, (1e9, 1e11), 1e-6) == rate_table_key(
    ...     second, (1e9, 1e11), 1e-6
    ... )
    True
    """
    settings = {
        "field_max": float(field_range[1]),
        "field_min": float(field_range[0]),
        "rtol": float(rtol),
        "convention": rate_convention(rate),
        "species": rate_species(rate),
    }
    return (SCHEMA_VERSION, *sorted(settings.items()), *rate.parameters)


def clear_rate_table_cache() -> None:
    """Forget every table built in this process.

    The cache exists so that two responses sharing a rate share its table; it is a
    memo, not a store, and nothing is written to disk. Tests that count builds clear it
    first.

    Examples
    --------
    >>> from solprop.translate.ionisation import clear_rate_table_cache
    >>> clear_rate_table_cache()
    """
    _TABLES.clear()


def _validate_build(field_range: tuple[float, float], rtol: float) -> None:
    """Check the build arguments.

    Parameters
    ----------
    field_range : tuple of float
        Lower and upper field amplitude in V/m.
    rtol : float
        Requested relative tolerance.

    Raises
    ------
    TranslationError
        Naming the offending argument.
    """
    low, high = float(field_range[0]), float(field_range[1])
    if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
        msg = (
            "field_range must be two finite amplitudes with 0 < low < high, got "
            f"({field_range[0]!r}, {field_range[1]!r})"
        )
        raise TranslationError(msg)
    if not (math.isfinite(rtol) and rtol > 0.0):
        msg = f"rtol must be positive and finite, got {rtol!r}"
        raise TranslationError(msg)


def _usable_range(
    rate: IonisationRate, field_range: tuple[float, float]
) -> tuple[float, float]:
    """Return the field range the table is actually built over.

    The lower end is raised to the rate's threshold, below which the rate is identically
    zero and its logarithm does not exist. The upper end is lowered to a rate's declared
    ``maximum_field``, above which it holds its value, and to just below the first field
    at which the rate reaches its own ceiling: the clip puts a corner in
    ``log W`` that a cubic interpolant resolves only at first order, so a table that
    straddled it would spend its whole budget there. Above ``field_max`` the table
    returns its last value, which is the ceiling to within one sample.

    Parameters
    ----------
    rate : IonisationRate
        The rate to tabulate.
    field_range : tuple of float
        Lower and upper field amplitude in V/m, already validated.

    Returns
    -------
    tuple of float
        The field range to build over.

    Raises
    ------
    TranslationError
        If nothing is left of the range: the rate is zero throughout, it is held from
        below the first field, or it is at its ceiling from the first field.
    """
    # The margin above the threshold is what makes the lowest knot survive the round
    # trip through log and exp: the rate is exactly zero below the threshold, and one
    # unit in the last place of the exponential is enough to land there and put a
    # logarithm of zero into the fit. A part in 1e9 of the field changes the rate by
    # about a part in 1e6 -- of a number near 1e-308.
    field_min = max(float(field_range[0]), rate.threshold * (1.0 + 1e-9))
    if field_min >= float(field_range[1]):
        msg = (
            f"the rate is exactly zero below {rate.threshold:.3e} V/m, which is at or "
            f"above the top of field_range {field_range!r}; nothing to tabulate"
        )
        raise TranslationError(msg)
    # A rate that declares a maximum field holds its value above it (an optmat rate,
    # SEP 3), so the table stops there and holds its last knot, which is the same
    # value: a knot range straddling the hold's corner would converge only at first
    # order there.
    ceiling = float(getattr(rate, "maximum_field", math.inf))
    field_max = min(float(field_range[1]), ceiling)
    if field_min >= field_max:
        msg = (
            f"the rate is held above its maximum field {ceiling:.3e} V/m, which is at "
            f"or below the bottom of the tabulated fields {field_min:.3e} V/m; "
            "nothing to tabulate"
        )
        raise TranslationError(msg)
    scan = np.geomspace(field_min, field_max, _CLIP_SCAN_POINTS)
    # A rate that is clipped is equal to its ceiling to the last bit, because the clip
    # is a minimum against that exact number; the slack covers the ceiling being reached
    # by the formula itself rather than by the clip.
    clipped = rate(scan) >= rate.maximum * (1.0 - 1e-12)
    if not bool(np.any(clipped)):
        return field_min, field_max
    first = int(np.argmax(clipped))
    if first == 0:
        msg = (
            f"the rate is at its ceiling of {rate.maximum:.3e} 1/s from the bottom of "
            f"field_range {field_range!r}; raise the rate's maximum or lower the range"
        )
        raise TranslationError(msg)
    return field_min, float(scan[first - 1])


def _fit(
    rate: IonisationRate, field_min: float, field_max: float, n_intervals: int
) -> tuple[Float[np.ndarray, " n_table"], Float[np.ndarray, " n_deg n_intervals"]]:
    """Fit the cubic spline of ``log W`` on ``n_intervals`` uniform intervals in log E.

    Parameters
    ----------
    rate : IonisationRate
        The rate to tabulate.
    field_min, field_max : float
        Ends of the range in V/m.
    n_intervals : int
        Number of knot intervals.

    Returns
    -------
    tuple
        The knot values of ``log W`` and the spline coefficients.
    """
    log_field = np.linspace(
        math.log(field_min), math.log(field_max), n_intervals + 1, dtype=np.float64
    )
    log_rate = np.log(rate(np.exp(log_field)))
    spline = interpolate.CubicSpline(log_field, log_rate)
    return log_rate, np.asarray(spline.c, dtype=np.float64)


def _probe_points(
    log_field_min: float, d_log_field: float, n_intervals: int
) -> Float[np.ndarray, " n_probe"]:
    """Return the logarithmic field amplitudes at which the interpolant is checked.

    Parameters
    ----------
    log_field_min : float
        Natural logarithm of the lowest field amplitude.
    d_log_field : float
        Knot spacing in ``log E``.
    n_intervals : int
        Number of knot intervals.

    Returns
    -------
    numpy.ndarray
        Probe positions in ``log E``, three per interval, shape ``(n_probe,)``.
    """
    left = log_field_min + d_log_field * np.arange(n_intervals, dtype=np.float64)
    return (left[:, None] + d_log_field * np.array(_PROBE_OFFSETS)[None, :]).ravel()


def build_rate_table(
    rate: IonisationRate,
    field_range: tuple[float, float],
    *,
    rtol: float = DEFAULT_RTOL,
    n_intervals: int = DEFAULT_INTERVALS,
    max_knots: int = MAX_KNOTS,
    use_cache: bool = True,
) -> RateTable:
    """Build a converged spline table of an ionisation rate.

    The knot count doubles until the interpolant is within ``rtol`` of direct evaluation
    at every probe point, which is what makes the table's accuracy a stated property
    rather than a consequence of a fixed sample count.

    Parameters
    ----------
    rate : IonisationRate
        The rate to tabulate, in s^-1 for a field amplitude in V/m.
    field_range : tuple of float
        Lower and upper field amplitude in V/m. The lower end is raised to the rate's
        threshold and the upper end lowered below any clipping; see the table's
        ``field_min`` and ``field_max`` for what was used.
    rtol : float, optional
        Bound on the relative deviation from direct evaluation, measured as the largest
        absolute difference of logarithms.
    n_intervals : int, optional
        Knot intervals to try first.
    max_knots : int, optional
        Knot budget before the build gives up.
    use_cache : bool, optional
        Reuse a table already built in this process for the same key, and record this
        one under it.

    Returns
    -------
    RateTable
        The converged table.

    Raises
    ------
    TranslationError
        If the arguments are inconsistent, nothing is left of the field range, or the
        tolerance is not reached within ``max_knots``.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.translate.ionisation import build_rate_table
    >>> table = build_rate_table(OptmatRate("Ar", 800e-9), (1e9, 5e10), rtol=1e-6)
    >>> table.error < 1e-6
    True
    """
    _validate_build(field_range, rtol)
    key = rate_table_key(rate, (float(field_range[0]), float(field_range[1])), rtol)
    if use_cache and key in _TABLES:
        return _TABLES[key]

    started = time.perf_counter()
    field_min, field_max = _usable_range(rate, field_range)
    log_field_min, log_field_max = math.log(field_min), math.log(field_max)
    intervals = int(n_intervals)
    while intervals + 1 <= max_knots:
        log_rate, coefficients = _fit(rate, field_min, field_max, intervals)
        d_log_field = (log_field_max - log_field_min) / intervals
        table = RateTable(
            log_rate=log_rate,
            coefficients=coefficients,
            log_field_min=log_field_min,
            d_log_field=d_log_field,
            field_min=field_min,
            field_max=field_max,
            rate_max=float(np.exp(log_rate[-1])),
            error=math.inf,
            key=_digest(key),
            convention=rate_convention(rate),
            species_identity=rate_species(rate)[0],
            ionisation_potential=rate_species(rate)[1],
        )
        probes = _probe_points(log_field_min, d_log_field, intervals)
        # The difference of logarithms is the relative error of the rate to first order,
        # and unlike a ratio it stays meaningful where the rate itself is 1e-300.
        error = float(
            np.max(np.abs(table.log_evaluate(probes) - np.log(rate(np.exp(probes)))))
        )
        if error <= rtol:
            table = _finalise(table, error)
            _log_summary(table, rate, time.perf_counter() - started)
            if use_cache:
                _TABLES[key] = table
            return table
        intervals *= 2
    msg = (
        f"the rate table did not reach rtol={rtol:g} within {max_knots} knots over "
        f"[{field_min:.3e}, {field_max:.3e}] V/m; the rate is not smooth there"
    )
    raise TranslationError(msg)


def _finalise(table: RateTable, error: float) -> RateTable:
    """Return a copy of the table with the measured error recorded.

    Parameters
    ----------
    table : RateTable
        The table built at the winning knot count.
    error : float
        The largest deviation measured on the probe grid.

    Returns
    -------
    RateTable
        The same table, with :attr:`RateTable.error` set.
    """
    return RateTable(
        log_rate=table.log_rate,
        coefficients=table.coefficients,
        log_field_min=table.log_field_min,
        d_log_field=table.d_log_field,
        field_min=table.field_min,
        field_max=table.field_max,
        rate_max=table.rate_max,
        error=error,
        key=table.key,
        convention=table.convention,
        species_identity=table.species_identity,
        ionisation_potential=table.ionisation_potential,
    )


def _digest(key: tuple[object, ...]) -> str:
    """Return a stable hexadecimal digest of a cache key.

    Through :mod:`hashlib` rather than :func:`hash`, whose value for a string is salted
    per process: the digest names a cache entry that may outlive this process.

    Parameters
    ----------
    key : tuple
        The output of :func:`rate_table_key`.

    Returns
    -------
    str
        Sixteen hexadecimal characters.
    """
    return hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).hexdigest()


def _log_summary(table: RateTable, rate: IonisationRate, elapsed: float) -> None:
    """Record the table's size, span and accuracy at debug level.

    Parameters
    ----------
    table : RateTable
        The finished table.
    rate : IonisationRate
        The rate it came from.
    elapsed : float
        Wall time of the build in seconds.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "rate table %s: %d knots over [%.3e, %.3e] V/m (%.2f decades), rate "
            "[%.3e, %.3e] 1/s, error %.2e, built in %.2f s from %s",
            table.key,
            table.n_knots,
            table.field_min,
            table.field_max,
            math.log10(table.field_max / table.field_min),
            float(np.exp(table.log_rate[0])),
            table.rate_max,
            table.error,
            elapsed,
            rate.citation,
        )
