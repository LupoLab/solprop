# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How numbers are shown: scales, ranges and short texts, with no physics in them."""

import math
from collections.abc import Sequence

import numpy as np

from solprop_gui.registry.units import si

__all__ = [
    "decibels",
    "duration_text",
    "even_rows",
    "length_unit",
    "pulse_window",
    "scaled_unit",
    "size_text",
    "steady_limits",
    "wants_log",
]

_TINY = 1e-30
"""What a zero reads as on a dB scale, relative to the largest value."""


def decibels(values: np.ndarray) -> np.ndarray:
    """Return values in dB below their largest.

    Parameters
    ----------
    values : numpy.ndarray
        The values, not negative.

    Returns
    -------
    numpy.ndarray
        In dB, at most 0; all ``-300`` where every value is zero.

    Examples
    --------
    >>> decibels(np.array([1.0, 0.1, 0.0])).round(1).tolist()
    [0.0, -10.0, -300.0]
    """
    top = float(np.max(values)) if values.size else 0.0
    if top <= 0:
        return np.full(values.shape, -300.0)
    return 10.0 * np.log10(np.maximum(values / top, _TINY))


def pulse_window(
    t: np.ndarray, power: np.ndarray, *, fraction: float = 1e-3, margin: float = 1.2
) -> float:
    """Return the half-width of the time range that shows a pulse.

    Parameters
    ----------
    t : numpy.ndarray
        The times.
    power : numpy.ndarray
        The power, on rows of planes or one trace, against ``t`` on its last axis.
    fraction : float, optional
        The share of the peak that counts as pulse.
    margin : float, optional
        How much wider than that to show.

    Returns
    -------
    float
        The half-width, in the unit of ``t``; the whole window with no pulse in it.

    Examples
    --------
    >>> t = np.linspace(-100.0, 100.0, 201)
    >>> round(pulse_window(t, np.exp(-((t / 10.0) ** 2))), 1)
    31.2
    """
    trace = power if power.ndim == 1 else power.max(axis=0)
    top = float(trace.max()) if trace.size else 0.0
    whole = float(np.abs(t).max()) if t.size else 1.0
    if top <= 0:
        return whole
    reached = np.abs(t[trace >= fraction * top]).max()
    return min(float(reached) * margin, whole)


def wants_log(values: np.ndarray, decades: float = 3.0) -> bool:
    """Say whether a quantity reads better on a log axis.

    Parameters
    ----------
    values : numpy.ndarray
        The values.
    decades : float, optional
        How wide a positive range calls for one.

    Returns
    -------
    bool
        Whether its positive values span more than that, none being negative.

    Examples
    --------
    >>> wants_log(np.array([1e-3, 1.0, 1e3])), wants_log(np.array([1.0, 2.0]))
    (True, False)
    >>> wants_log(np.array([-1.0, 1e6]))
    False
    """
    finite = values[np.isfinite(values)]
    positive = finite[finite > 0]
    if positive.size == 0 or (finite < 0).any():
        return False
    return math.log10(float(positive.max()) / float(positive.min())) > decades


def steady_limits(
    values: Sequence[float] | np.ndarray, *, noise: float = 1e-6, padding: float = 0.06
) -> tuple[float, float]:
    """Return a quantity's axis limits, which do not magnify noise.

    A quantity that holds still to within ``noise`` of its size is drawn as the constant
    it is, on an axis a few percent either side of it, rather than filling the axis
    with rounding (design 11.3).

    Parameters
    ----------
    values : Sequence of float or numpy.ndarray
        The values.
    noise : float, optional
        The relative change below which it counts as constant.
    padding : float, optional
        The share of its range added at each end.

    Returns
    -------
    tuple of float
        The lower and upper limits.

    Examples
    --------
    >>> [round(v, 3) for v in steady_limits([200.0, 200.00000005])]
    [194.0, 206.0]
    >>> [round(v, 2) for v in steady_limits([0.0, 1.0])]
    [-0.06, 1.06]
    >>> steady_limits([0.0, 0.0]), steady_limits([])
    ((-1.0, 1.0), (0.0, 1.0))
    """
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return 0.0, 1.0
    low, high = float(data.min()), float(data.max())
    size = max(abs(low), abs(high))
    if size == 0:
        return -1.0, 1.0
    if high - low <= noise * size:
        middle = 0.5 * (low + high)
        return middle - 0.03 * size, middle + 0.03 * size
    extra = padding * (high - low)
    return low - extra, high + extra


def size_text(size: float) -> str:
    """Write a number of bytes briefly.

    Parameters
    ----------
    size : float
        Bytes.

    Returns
    -------
    str
        With an SI prefix.

    Examples
    --------
    >>> size_text(2_846_616), size_text(3.5e9), size_text(2.81e8)
    ('2.85 MB', '3.5 GB', '281 MB')
    """
    return si(size, "B", digits=3)


def duration_text(seconds: float) -> str:
    """Write a duration as a clock reads.

    Parameters
    ----------
    seconds : float
        Seconds.

    Returns
    -------
    str
        Seconds below ten, then ``m:ss``, and ``h:mm:ss`` from an hour.

    Examples
    --------
    >>> duration_text(0.96), duration_text(42.4), duration_text(3725.0)
    ('0.96 s', '0:42', '1:02:05')
    """
    if seconds < 10:
        return f"{seconds:.2g} s"
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def length_unit(length: float) -> tuple[str, float]:
    """Return the unit a length along z reads best in.

    Parameters
    ----------
    length : float
        The length, m.

    Returns
    -------
    tuple
        The unit's name, and what metres are multiplied by to read in it.

    Examples
    --------
    >>> length_unit(1.0), length_unit(0.25), length_unit(0.005)
    (('m', 1.0), ('cm', 100.0), ('mm', 1000.0))
    """
    if length >= 1.0:
        return "m", 1.0
    return ("cm", 100.0) if length >= 0.01 else ("mm", 1000.0)


_PREFIXED = frozenset({"J", "W", "m", "s", "Hz", "V"})
"""The units a physicist reads with an SI prefix; others take a power of ten."""

_READ_IN = {"W/m²": ("W/cm²", 1e-4), "1/m³": ("cm⁻³", 1e-6), "1/m^3": ("cm⁻³", 1e-6)}
"""The units laboratory quantities are read in, by the unit solprop gives them in."""

_SUPERSCRIPTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def scaled_unit(values: np.ndarray, unit: str) -> tuple[str, float]:
    """Return the unit a quantity reads best in.

    Parameters
    ----------
    values : numpy.ndarray
        The values, in ``unit``.
    unit : str
        Their unit, unprefixed; empty for a pure number.

    Returns
    -------
    tuple
        The unit to write, and what the values are multiplied by to read in it.

    Examples
    --------
    >>> scaled_unit(np.array([2.75e-4]), "J")
    ('µJ', 1000000.0)
    >>> label, factor = scaled_unit(np.array([3e18]), "W/m²")
    >>> label, round(3e18 * factor, 6)
    ('10¹⁴ W/cm²', 3.0)
    >>> label, factor = scaled_unit(np.array([4e24]), "1/m³")
    >>> label, round(4e24 * factor, 6)
    ('10¹⁸ cm⁻³', 4.0)
    >>> scaled_unit(np.array([0.5]), ""), scaled_unit(np.array([0.0]), "J")
    (('', 1.0), ('J', 1.0))
    """
    shown, scale = _READ_IN.get(unit, (unit, 1.0))
    finite = np.abs(values[np.isfinite(values)]) * scale
    top = float(finite.max()) if finite.size else 0.0
    if not shown or top == 0:
        return shown, scale
    if shown in _PREFIXED:
        power = max(min(math.floor(math.log10(top) / 3), 6), -5)
        prefix = {
            6: "E", 5: "P", 4: "T", 3: "G", 2: "M", 1: "k", 0: "",
            -1: "m", -2: "µ", -3: "n", -4: "p", -5: "f",
        }[power]  # fmt: skip
        return f"{prefix}{shown}", scale * 10.0 ** (-3 * power)
    exponent = math.floor(math.log10(top))
    if exponent == 0:
        return shown, scale
    return (
        f"10{str(exponent).translate(_SUPERSCRIPTS)} {shown}",
        scale * 10.0**-exponent,
    )


def even_rows(z: np.ndarray, count: int = 256) -> tuple[np.ndarray, float, float]:
    """Return which plane each row of a map along z shows.

    Saved planes need not be even along z, and an image's rows are. Even planes map to
    themselves.

    Parameters
    ----------
    z : numpy.ndarray
        The planes' positions, ascending.
    count : int, optional
        The rows of a map of uneven planes.

    Returns
    -------
    tuple
        The plane of each row, where the first row's middle is and the rows' spacing.

    Examples
    --------
    >>> rows, start, step = even_rows(np.array([0.0, 1.0, 2.0]))
    >>> rows.tolist(), start, step
    ([0, 1, 2], 0.0, 1.0)
    >>> rows, start, step = even_rows(np.array([0.0, 1.0, 4.0]), count=5)
    >>> rows.tolist(), step
    ([0, 1, 1, 2, 2], 1.0)
    >>> even_rows(np.array([0.5]))[1:]
    (0.5, 1.0)
    """
    if z.size < 2:
        return np.zeros(z.size, dtype=int), float(z[0]) if z.size else 0.0, 1.0
    spacing = np.diff(z)
    if np.ptp(spacing) <= 1e-6 * abs(float(spacing.mean())):
        return np.arange(z.size), float(z[0]), float(spacing.mean())
    even = np.linspace(float(z[0]), float(z[-1]), count)
    nearest = np.abs(even[:, None] - z[None, :]).argmin(axis=1)
    return nearest, float(even[0]), float(even[1] - even[0])
