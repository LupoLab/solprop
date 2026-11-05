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
    "pulse_window",
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
    >>> steady_limits([200.0, 200.00000005])
    (194.0, 206.0)
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
    >>> size_text(2_846_616), size_text(3.5e9)
    ('2.8 MB', '3.5 GB')
    """
    return si(size, "B", digits=2)


def duration_text(seconds: float) -> str:
    """Write a duration as a clock reads.

    Parameters
    ----------
    seconds : float
        Seconds.

    Returns
    -------
    str
        ``m:ss``, or ``h:mm:ss`` from an hour.

    Examples
    --------
    >>> duration_text(42.4), duration_text(3725.0)
    ('0:42', '1:02:05')
    """
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"
