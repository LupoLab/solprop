# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plot options shared by the live run, the Overview and the Analyse pages.

One :class:`PlotOptions` holds the spectral axis (wavelength or frequency), limits for
the spectral and time axes, the spectral scale and its floor, a bandpass for the
time-domain views and the spectrogram's gate. Values are kept in SI (m, Hz, s); the
views show them in nm, THz and fs (:data:`UNITS`).

The axis and the bandpass change what solprop computes (the spectral density on the
axis, the power within the band), so they travel to the kernel; the limits, scale and
floor only change how a view draws what it has.
"""

from dataclasses import dataclass
from typing import Final, Literal

__all__ = ["AXES", "UNITS", "PlotOptions", "parse_limit", "to_display"]

Axis = Literal["wavelength", "f"]
Scale = Literal["dB", "linear"]

AXES: Final[dict[str, str]] = {"wavelength": "Wavelength", "f": "Frequency"}
"""The spectral axes, by solprop's name, with their labels."""

UNITS: Final[dict[str, tuple[str, float]]] = {
    "wavelength": ("nm", 1e-9),
    "f": ("THz", 1e12),
    "time": ("fs", 1e-15),
}
"""Each axis's display unit and its size in SI."""


def to_display(axis: str, value: float) -> float:
    """Return an SI value in its axis's display unit.

    Parameters
    ----------
    axis : str
        ``"wavelength"``, ``"f"`` or ``"time"``.
    value : float
        The value in m, Hz or s.

    Returns
    -------
    float
        It in nm, THz or fs.

    Examples
    --------
    >>> round(to_display("wavelength", 800e-9), 6), round(to_display("time", 3e-15), 6)
    (800.0, 3.0)
    """
    return value / UNITS[axis][1]


def parse_limit(text: str) -> float | None:
    """Read a typed limit: a number, or nothing for automatic.

    Parameters
    ----------
    text : str
        What was typed.

    Returns
    -------
    float or None
        The number; ``None`` for a blank or ``auto``.

    Raises
    ------
    ValueError
        If it is neither a number nor blank.

    Examples
    --------
    >>> parse_limit(" 1200 "), parse_limit(""), parse_limit("auto")
    (1200.0, None, None)
    """
    text = text.strip()
    if not text or text.lower() == "auto":
        return None
    return float(text)


Limits = tuple[float | None, float | None]


@dataclass(frozen=True)
class PlotOptions:
    """What the views draw with, and what they ask solprop for.

    Attributes
    ----------
    axis : {'wavelength', 'f'}
        The spectral axis of spectra, maps and spectrograms.
    spectral_limits : tuple
        The spectral axis's shown range, in m or Hz as ``axis`` says; either end
        ``None`` for automatic.
    time_limits : tuple
        The time axis's shown range, s; either end ``None`` for automatic.
    scale : {'dB', 'linear'}
        How spectra are drawn.
    floor : float
        The bottom of the dB scale, dB.
    bandpass : tuple of float or None
        A wavelength band, m, the time-domain views are computed within; ``None``
        for the whole spectrum.
    gate_fwhm : float or None
        The spectrogram's gate, s; ``None`` for solprop's default (a twentieth of
        the time range).
    positions : int
        The spectrogram's gate positions.

    Examples
    --------
    >>> PlotOptions().label()
    'Wavelength (nm)'
    """

    axis: Axis = "wavelength"
    spectral_limits: Limits = (None, None)
    time_limits: Limits = (None, None)
    scale: Scale = "dB"
    floor: float = -60.0
    bandpass: tuple[float, float] | None = None
    gate_fwhm: float | None = None
    positions: int = 256

    def shown_limits(self, which: str) -> Limits:
        """Return an axis's limits in its display unit, ascending.

        Parameters
        ----------
        which : {'spectral', 'time'}
            The axis.

        Returns
        -------
        tuple
            The limits in nm or THz, or fs; ``None`` where automatic.

        Examples
        --------
        >>> PlotOptions(time_limits=(-1e-13, None)).shown_limits("time")
        (-100.0, None)
        """
        axis = self.axis if which == "spectral" else "time"
        limits = self.spectral_limits if which == "spectral" else self.time_limits
        shown = [None if v is None else to_display(axis, v) for v in limits]
        return shown[0], shown[1]

    def label(self) -> str:
        """Return the spectral axis's label, with its unit.

        Returns
        -------
        str
            ``"Wavelength (nm)"`` or ``"Frequency (THz)"``.

        Examples
        --------
        >>> PlotOptions(axis="f").label()
        'Frequency (THz)'
        """
        return f"{AXES[self.axis]} ({UNITS[self.axis][0]})"

    def describe(self) -> str:
        """Return a short statement of what differs from the defaults.

        Returns
        -------
        str
            For a view's status line; empty when nothing is set.

        Examples
        --------
        >>> PlotOptions(bandpass=(700e-9, 900e-9)).describe()
        'time domain within 700-900 nm'
        """
        if self.bandpass is None:
            return ""
        low, high = (to_display("wavelength", v) for v in self.bandpass)
        return f"time domain within {low:.0f}-{high:.0f} nm"
