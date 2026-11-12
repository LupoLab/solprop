# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reduce a saved plane to what the live and overview views draw (design 13.3, 12.8).

A plane's spectral density and power come from solprop (a
:class:`~solprop.driver.plane_view.PlaneView` during a run, the file's
:mod:`solprop.processing` functions after it), the density on both spectral axes
(wavelength and frequency, solprop applying the Jacobian, so a view switches between
them at once) and the power within the display's bandpass; this module only puts them
on the display's bins. Each display bin keeps the **largest**
value that falls in it, so a narrow peak survives the reduction; a bin no sample falls
in takes the value interpolated at its centre. The reduction is for display only, and
says so: exports and "Copy as Python" are computed from the file.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

__all__ = [
    "SPECTRAL_AXES",
    "TIME_OVERSAMPLING",
    "C",
    "Display",
    "Reducer",
    "bin_max",
    "light_range",
    "stats_row",
]

TIME_OVERSAMPLING = 4
"""How much finer than the run's own grid a pulse is drawn in time.

solprop's readers resample the time axis by zero-padding the spectrum, which adds
no information but draws a short pulse smoothly and finds its true peak between
samples; four is the factor solprop's own plots use, as Luna's do.
"""

C = 299_792_458.0
"""The speed of light in vacuum, m/s: to turn the wavelength window to frequency."""


@dataclass(frozen=True)
class Display:
    """The bins a run's planes are reduced onto.

    Attributes
    ----------
    wavelength_range : tuple of float or None
        The wavelength window, m; ``None`` takes the first plane's band.
    n_wavelength : int
        Wavelength bins.
    time_range : tuple of float or None
        The time window, s; ``None`` takes the first plane's whole window.
    n_time : int
        Time bins: four times the wavelength bins, because a short pulse fills a
        small part of its time window (a 5 fs pulse in a 200 fs window spans 50
        of them).
    bandpass : tuple of float or None
        A wavelength band, m, the power is computed within (solprop's ``bandpass``
        keyword); ``None`` for the whole spectrum.

    Examples
    --------
    >>> Display(wavelength_range=(200e-9, 2e-6)).n_wavelength
    512
    """

    wavelength_range: tuple[float, float] | None = None
    n_wavelength: int = 512
    time_range: tuple[float, float] | None = None
    n_time: int = 2048
    bandpass: tuple[float, float] | None = None

    def within(self, keywords: Mapping[str, object]) -> "Display":
        """Return the display with the run's wavelength window where it has none.

        Parameters
        ----------
        keywords : Mapping
            The run's build keywords, in SI; ``wavelength_limits`` is read, or else
            the band of a ``grid`` (its ``omega_min`` and ``omega_max``).

        Returns
        -------
        Display
            This display, its wavelength range taken from the keywords when it has
            none and they give one.

        Examples
        --------
        >>> Display().within({"wavelength_limits": (2e-7, 2e-6)}).wavelength_range
        (2e-07, 2e-06)
        """
        if self.wavelength_range is not None:
            return self
        limits = keywords.get("wavelength_limits")
        if not isinstance(limits, tuple | list):
            grid = keywords.get("grid")
            high, low = getattr(grid, "omega_max", 0.0), getattr(grid, "omega_min", 0.0)
            if low <= 0.0 or high <= low:
                return self
            limits = (2.0 * np.pi * C / high, 2.0 * np.pi * C / low)
        low, high = (float(limit) for limit in limits)  # pyright: ignore[reportArgumentType]
        return replace(self, wavelength_range=(min(low, high), max(low, high)))


def bin_max(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Reduce samples onto bins, keeping each bin's largest value.

    Parameters
    ----------
    x : numpy.ndarray
        The abscissae, in any order.
    y : numpy.ndarray
        The values, one per abscissa.
    edges : numpy.ndarray
        The bin edges, ascending; ``len(edges) - 1`` bins.

    Returns
    -------
    numpy.ndarray
        One value per bin: the largest sample in it, or, for a bin no sample falls
        in, the value interpolated at its centre (zero outside the samples' span).

    Examples
    --------
    >>> import numpy as np
    >>> bin_max(np.array([0.1, 0.2, 0.9]), np.array([1.0, 3.0, 2.0]),
    ...         np.array([0.0, 0.5, 1.0])).tolist()
    [3.0, 2.0]
    """
    order = np.argsort(x)
    xs = np.asarray(x, dtype=float)[order]
    ys = np.asarray(y, dtype=float)[order]
    n_bins = edges.size - 1
    out = np.full(n_bins, -np.inf)
    index = np.searchsorted(edges, xs, side="right") - 1
    index[xs == edges[-1]] = n_bins - 1  # the last bin includes its upper edge
    inside = (index >= 0) & (index < n_bins)
    np.maximum.at(out, index[inside], ys[inside])
    empty = ~np.isfinite(out)
    if empty.any():
        centres = 0.5 * (edges[:-1] + edges[1:])
        out[empty] = np.interp(centres[empty], xs, ys, left=0.0, right=0.0)
    return out


def even_rows(
    x: np.ndarray, values: np.ndarray, window: tuple[float, float], count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Put a map's rows on evenly spaced abscissae within a window, for an image.

    An image is drawn on even bins; a map on wavelength is not even. Each column is
    interpolated at the new rows, which is for display only, as every reduction here.

    Parameters
    ----------
    x : numpy.ndarray
        The rows' abscissae, ascending.
    values : numpy.ndarray
        The map, shape ``(x.size, n_columns)``.
    window : tuple of float
        The range of the new rows.
    count : int
        How many rows.

    Returns
    -------
    tuple of numpy.ndarray
        The even abscissae and the map on them, zero outside ``x``.

    Examples
    --------
    >>> import numpy as np
    >>> rows, image = even_rows(np.array([1.0, 2.0, 4.0]), np.array([[0.0], [1.0],
    ...                         [3.0]]), (1.0, 4.0), 4)
    >>> rows.tolist(), image[:, 0].tolist()
    ([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0])
    """
    rows = np.linspace(window[0], window[1], count)
    image = np.stack(
        [
            np.interp(rows, x, values[:, column], left=0.0, right=0.0)
            for column in range(values.shape[1])
        ],
        axis=1,
    )
    return rows, image


def stats_row(stats: Mapping[str, object]) -> dict[str, float]:
    """Return a plane's statistics as one number each, summed over modes.

    Parameters
    ----------
    stats : Mapping
        The statistics row, by name, as solprop records it.

    Returns
    -------
    dict
        The finite totals, by name; a statistic that is not a number is left out.

    Examples
    --------
    >>> import numpy as np
    >>> stats_row({"energy": np.array([[1.0, 2.0]]), "label": "x"})
    {'energy': 3.0}
    """
    row: dict[str, float] = {}
    for name, value in stats.items():
        try:
            total = float(np.sum(np.asarray(value, dtype=float)))
        except TypeError, ValueError:
            continue
        if np.isfinite(total):
            row[name] = total
    return row


SPECTRAL_AXES = ("wavelength", "f")
"""The spectral axes every plane is reduced on, by solprop's names."""


class Reducer:
    """Put each plane's spectral densities and power on fixed display bins.

    The bins are fixed by the first plane reduced (or by the :class:`Display`), so
    every row of a run shares them. The density is binned on both spectral axes: the
    display's wavelength window, and the frequency window it spans.

    Parameters
    ----------
    display : Display
        The windows, bin counts and bandpass.

    Examples
    --------
    >>> import numpy as np
    >>> reducer = Reducer(Display(wavelength_range=(1e-6, 2e-6), n_wavelength=1))
    >>> t = np.array([-1e-15, 1e-15])
    >>> spectra = {"wavelength": (np.array([1.5e-6]), np.ones(1)),
    ...            "f": (np.array([2e14]), np.ones(1))}
    >>> density, power = reducer.reduce(spectra, t, np.ones(2))
    >>> sorted(density), power.shape
    (['f', 'wavelength'], (2048,))
    >>> [round(float(v) / 1e12, 1) for v in reducer.axes()["f"]]
    [249.9]
    """

    def __init__(self, display: Display) -> None:
        """Hold the display; the bins are made on the first plane."""
        self.display = display
        self._edges: dict[str, np.ndarray] = {}

    def _window(self, axis: str, x: np.ndarray) -> tuple[float, float]:
        """Return the spectral window on an axis.

        Parameters
        ----------
        axis : {'wavelength', 'f'}
            The axis.
        x : numpy.ndarray
            The first plane's positive values on it, m or Hz.

        Returns
        -------
        tuple of float
            The window, ascending: the display's wavelength range (on frequency,
            from the plane's lowest frequency to the range's highest), or the
            plane's whole band.
        """
        limits = self.display.wavelength_range
        if limits is None:
            return float(x.min()), float(x.max())
        if axis == "wavelength":
            return limits
        # On frequency, from the plane's lowest: a field grid reaches zero, where
        # no wavelength window can say.
        return float(x.min()), C / min(limits)

    def _bins(self, axis: str, x: np.ndarray, count: int) -> np.ndarray:
        """Return an axis's bin edges, made from the first plane's values.

        Parameters
        ----------
        axis : str
            ``"wavelength"``, ``"f"`` or ``"time"``.
        x : numpy.ndarray
            The plane's values on it.
        count : int
            The bins.

        Returns
        -------
        numpy.ndarray
            The edges.
        """
        if axis not in self._edges:
            if axis == "time":
                low, high = self.display.time_range or (x.min(), x.max())
            else:
                low, high = self._window(axis, x[np.isfinite(x) & (x > 0)])
            self._edges[axis] = np.linspace(low, high, count + 1)
        return self._edges[axis]

    def reduce(
        self,
        spectra: Mapping[str, tuple[np.ndarray, np.ndarray]],
        t: np.ndarray,
        power: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Reduce one plane.

        Parameters
        ----------
        spectra : Mapping
            By axis (``"wavelength"``, ``"f"``), the axis (m or Hz) and the spectral
            energy density on it, summed over modes.
        t : numpy.ndarray
            The time axis, s.
        power : numpy.ndarray
            The power on it, W, summed over modes.

        Returns
        -------
        tuple
            The densities on the display bins by axis, and the power on them, as
            float32.
        """
        reduced: dict[str, np.ndarray] = {}
        for axis, (given_x, given_y) in spectra.items():
            x = np.asarray(given_x, dtype=float)
            y = np.asarray(given_y, dtype=float)
            edges = self._bins(axis, x, self.display.n_wavelength)
            keep = np.isfinite(x) & (x > 0)
            reduced[axis] = bin_max(x[keep], y[keep], edges).astype(np.float32)
        t = np.asarray(t, dtype=float)
        edges = self._bins("time", t, self.display.n_time)
        return reduced, bin_max(t, np.asarray(power), edges).astype(np.float32)

    def axes(self) -> dict[str, Any]:
        """Return the bin centres.

        Returns
        -------
        dict
            ``wavelength`` (m), ``f`` (Hz) and ``time`` (s), each an array of bin
            centres, as far as planes have made them, and the ``bandpass`` the
            power is computed within.

        Raises
        ------
        RuntimeError
            Before the first plane is reduced, when the bins are not yet known.
        """
        if "time" not in self._edges:
            msg = "the bins are made by the first plane reduced"
            raise RuntimeError(msg)
        found: dict[str, Any] = {
            axis: 0.5 * (edges[:-1] + edges[1:]) for axis, edges in self._edges.items()
        }
        found["bandpass"] = self.display.bandpass
        return found


def light_range(
    x: np.ndarray, rows: np.ndarray, *, floor: float = 1e-6, margin: float = 0.0
) -> tuple[float, float]:
    """Return the part of a spectral axis that holds light in any row.

    Parameters
    ----------
    x : numpy.ndarray
        The positions, ascending.
    rows : numpy.ndarray
        The values, with the positions along the last axis.
    floor : float, optional
        What holds light, as a share of the peak.
    margin : float, optional
        What to add at each end, as a share of the span.

    Returns
    -------
    tuple of float
        The range; the whole axis when nothing holds light.

    Examples
    --------
    >>> light_range(np.arange(5.0), np.array([[0.0, 1.0, 2.0, 0.0, 0.0]]))
    (1.0, 2.0)
    >>> light_range(np.arange(5.0), np.zeros((1, 5)))
    (0.0, 4.0)
    """
    held = np.asarray(rows, dtype=float).reshape(-1, x.size).max(axis=0)
    top = float(held.max())
    light = np.flatnonzero(held > floor * top) if top > 0 else np.zeros(0, dtype=int)
    if light.size == 0:
        return float(x[0]), float(x[-1])
    low, high = float(x[light[0]]), float(x[light[-1]])
    pad = margin * (high - low)
    return max(low - pad, float(x[0])), min(high + pad, float(x[-1]))
