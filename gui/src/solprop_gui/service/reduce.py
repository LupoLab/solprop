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

__all__ = ["SPECTRAL_AXES", "C", "Display", "Reducer", "bin_max", "stats_row"]

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
        Time bins.
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
    n_time: int = 512
    bandpass: tuple[float, float] | None = None

    def within(self, keywords: Mapping[str, object]) -> Display:
        """Return the display with the run's wavelength window where it has none.

        Parameters
        ----------
        keywords : Mapping
            The run's build keywords, in SI; ``wavelength_limits`` is read.

        Returns
        -------
        Display
            This display, its wavelength range taken from the keywords when it has
            none and they give a pair.

        Examples
        --------
        >>> Display().within({"wavelength_limits": (2e-7, 2e-6)}).wavelength_range
        (2e-07, 2e-06)
        """
        limits = keywords.get("wavelength_limits")
        if self.wavelength_range is not None or not isinstance(limits, tuple | list):
            return self
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
    (['f', 'wavelength'], (512,))
    >>> [round(float(v) / 1e12, 1) for v in reducer.axes()["f"]]
    [224.8]
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
            The window, ascending: the display's wavelength range on the axis, or
            the plane's whole band.
        """
        limits = self.display.wavelength_range
        if limits is None:
            return float(x.min()), float(x.max())
        if axis == "wavelength":
            return limits
        low, high = sorted(C / limit for limit in limits)
        return low, high

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
