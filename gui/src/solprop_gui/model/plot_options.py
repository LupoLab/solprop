# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The plot options shared by the live run, the Overview and the plane views.

One :class:`PlotOptions` holds what the plot options panel sets: the spectral axis
(wavelength or frequency), typed limits for the spectral and time axes, the spectral
scale and its floor, and a bandpass for the time-domain views. Values are kept in SI
(m, Hz, s); the panel shows them in nm, THz and fs (:data:`UNITS`).

The axis and the bandpass change what solprop computes (the spectral density on the
axis, the power within the band), so they travel to the kernel and the run's worker
(:meth:`PlotOptions.request`); the limits, scale and floor only change how a view
draws what it has.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

__all__ = [
    "AXES",
    "SCALES",
    "UNITS",
    "PlotOptions",
    "from_display",
    "from_mapping",
    "parse_limit",
    "to_display",
]

Axis = Literal["wavelength", "f"]
Scale = Literal["dB", "linear"]

AXES: Final[dict[str, str]] = {"wavelength": "Wavelength", "f": "Frequency"}
"""The spectral axes, by solprop's name, with their labels."""

SCALES: Final = ("dB", "linear")
"""The spectral scales."""

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


def from_display(axis: str, value: float) -> float:
    """Return a value in its axis's display unit in SI.

    Parameters
    ----------
    axis : str
        ``"wavelength"``, ``"f"`` or ``"time"``.
    value : float
        The value in nm, THz or fs.

    Returns
    -------
    float
        It in m, Hz or s.

    Examples
    --------
    >>> from_display("f", 375.0)
    375000000000000.0
    """
    return value * UNITS[axis][1]


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
    """What the plot options panel sets.

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

    Examples
    --------
    >>> PlotOptions().request()
    {'axis': 'wavelength', 'bandpass': None}
    """

    axis: Axis = "wavelength"
    spectral_limits: Limits = (None, None)
    time_limits: Limits = (None, None)
    scale: Scale = "dB"
    floor: float = -60.0
    bandpass: tuple[float, float] | None = None

    def request(self) -> dict[str, object]:
        """Return what solprop is asked to compute differently.

        Returns
        -------
        dict
            ``axis`` and ``bandpass``, as the kernel's handlers and the run's
            display take them.
        """
        return {"axis": self.axis, "bandpass": self.bandpass}

    def same_request(self, other: PlotOptions) -> bool:
        """Return whether another set of options asks solprop for the same data.

        Parameters
        ----------
        other : PlotOptions
            The other options.

        Returns
        -------
        bool
            True when only the drawing differs.

        Examples
        --------
        >>> PlotOptions().same_request(PlotOptions(scale="linear"))
        True
        >>> PlotOptions().same_request(PlotOptions(axis="f"))
        False
        """
        return self.request() == other.request()

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


def from_mapping(values: Mapping[str, object]) -> PlotOptions:
    """Return options saved as a mapping, defaults where a value is missing or bad.

    Parameters
    ----------
    values : Mapping
        What :func:`dataclasses.asdict` gave, as settings keep it.

    Returns
    -------
    PlotOptions
        The options.

    Examples
    --------
    >>> options = from_mapping({"axis": "f", "floor": "-40", "scale": "nonsense"})
    >>> options.axis, options.floor, options.scale
    ('f', -40.0, 'dB')
    """
    axis = values.get("axis")
    scale = values.get("scale")

    def limits(name: str) -> Limits:
        """Return a saved pair of limits; automatic where it is not a pair.

        Parameters
        ----------
        name : str
            The field.

        Returns
        -------
        tuple
            The limits.
        """
        pair = values.get(name)
        if not isinstance(pair, list | tuple) or len(pair) != 2:
            return (None, None)
        low, high = (None if v is None else float(v) for v in pair)  # pyright: ignore[reportArgumentType]
        return low, high

    band = values.get("bandpass")
    bandpass = None
    if isinstance(band, list | tuple) and len(band) == 2:
        bandpass = (float(band[0]), float(band[1]))  # pyright: ignore[reportArgumentType]
    try:
        floor = float(values.get("floor", -60.0))  # pyright: ignore[reportArgumentType]
    except TypeError, ValueError:
        floor = -60.0
    return PlotOptions(
        axis=axis if axis in AXES else "wavelength",  # pyright: ignore[reportArgumentType]
        spectral_limits=limits("spectral_limits"),
        time_limits=limits("time_limits"),
        scale=scale if scale in SCALES else "dB",  # pyright: ignore[reportArgumentType]
        floor=floor,
        bandpass=bandpass,
    )
