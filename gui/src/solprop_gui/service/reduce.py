# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reduce a saved plane to what the live and overview views draw (design 13.3, 12.8).

A plane's spectral density and power come from solprop (a
:class:`~solprop.driver.plane_view.PlaneView` during a run, the file's
:mod:`solprop.processing` functions after it); this module only puts them on the
display's bins. Each display bin keeps the **largest** value that falls in it, so a
narrow peak survives the reduction; a bin no sample falls in takes the value
interpolated at its centre. The reduction is for display only, and says so: exports and
"Copy as Python" are computed from the file.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["Display", "Reducer", "bin_max", "stats_row"]


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

    Examples
    --------
    >>> Display(wavelength_range=(200e-9, 2e-6)).n_wavelength
    512
    """

    wavelength_range: tuple[float, float] | None = None
    n_wavelength: int = 512
    time_range: tuple[float, float] | None = None
    n_time: int = 512


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


class Reducer:
    """Put each plane's spectral density and power on fixed display bins.

    The bins are fixed by the first plane reduced (or by the :class:`Display`), so
    every row of a run shares them.

    Parameters
    ----------
    display : Display
        The windows and bin counts.

    Examples
    --------
    >>> import numpy as np
    >>> reducer = Reducer(Display(n_wavelength=2, n_time=2))
    >>> lam = np.array([1e-6, 2e-6]); t = np.array([-1e-15, 1e-15])
    >>> density, power = reducer.reduce(lam, np.ones(2), t, np.ones(2))
    >>> density.shape, reducer.axes()["time"].shape
    ((2,), (2,))
    """

    def __init__(self, display: Display) -> None:
        """Hold the display; the bins are made on the first plane."""
        self.display = display
        self._lam_edges: np.ndarray | None = None
        self._t_edges: np.ndarray | None = None

    def _edges(self, lam: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the bin edges, making them from the first plane's axes.

        Parameters
        ----------
        lam : numpy.ndarray
            The plane's wavelength axis, m.
        t : numpy.ndarray
            Its time axis, s.

        Returns
        -------
        tuple of numpy.ndarray
            The wavelength and time edges.
        """
        if self._lam_edges is None or self._t_edges is None:
            display = self.display
            finite = lam[np.isfinite(lam) & (lam > 0)]
            low, high = display.wavelength_range or (finite.min(), finite.max())
            self._lam_edges = np.linspace(low, high, display.n_wavelength + 1)
            t_low, t_high = display.time_range or (t.min(), t.max())
            self._t_edges = np.linspace(t_low, t_high, display.n_time + 1)
        return self._lam_edges, self._t_edges

    def reduce(
        self,
        lam: np.ndarray,
        density: np.ndarray,
        t: np.ndarray,
        power: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce one plane.

        Parameters
        ----------
        lam : numpy.ndarray
            The wavelength axis, m.
        density : numpy.ndarray
            The spectral energy density on it, J/m, summed over modes.
        t : numpy.ndarray
            The time axis, s.
        power : numpy.ndarray
            The power on it, W, summed over modes.

        Returns
        -------
        tuple of numpy.ndarray
            The density and the power on the display bins, as float32.
        """
        lam_edges, t_edges = self._edges(lam, t)
        keep = np.isfinite(lam) & (lam > 0)
        reduced_density = bin_max(lam[keep], density[keep], lam_edges)
        reduced_power = bin_max(t, power, t_edges)
        return reduced_density.astype(np.float32), reduced_power.astype(np.float32)

    def axes(self) -> dict[str, np.ndarray]:
        """Return the bin centres.

        Returns
        -------
        dict
            ``wavelength`` (m) and ``time`` (s), each an array of bin centres.

        Raises
        ------
        RuntimeError
            Before the first plane is reduced, when the bins are not yet known.
        """
        if self._lam_edges is None or self._t_edges is None:
            msg = "the bins are made by the first plane reduced"
            raise RuntimeError(msg)
        return {
            "wavelength": 0.5 * (self._lam_edges[:-1] + self._lam_edges[1:]),
            "time": 0.5 * (self._t_edges[:-1] + self._t_edges[1:]),
        }
