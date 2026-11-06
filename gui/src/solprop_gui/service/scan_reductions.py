# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a scan records at each point beyond its statistics (design 12.7).

A scan's reductions are solprop's (:mod:`solprop.processing.scans`), declared by the
front end from plain data, so a run's spec carries them: the energy and the peak power
within a wavelength band, and the spectrum or the pulse at a distance. Each is read at
the saved plane nearest the distance, the last without one. A spectrum or a pulse is
held on an axis fixed by the recipe, the same at every point, so a scan's points line
up; :func:`reduction_axis` rebuilds it from the recipe the scan records. The readers
are of a guided run's field: a beam's scan records its statistics only, and is refused
these (:data:`~solprop_gui.model.recipe.BEAM_ROUTES`).
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Final

import numpy as np
from solprop.constants import C
from solprop.output import Output
from solprop.processing import get_power, get_spectral_density
from solprop.processing.scans import Reduction, custom
from solprop.recipe import Recipe

from solprop_gui.model.recipe import BEAM_ROUTES
from solprop_gui.service.reduce import TIME_OVERSAMPLING

__all__ = [
    "AXIS_POINTS",
    "KINDS",
    "reduction_axis",
    "reductions",
]

AXIS_POINTS: Final = 400
"""The samples of a spectrum's or a pulse's fixed axis."""

KINDS: Final = {
    "energy_in_band": "J",
    "peak_power_in_band": "W",
    "spectrum": "J/m",
    "pulse": "W",
}
"""The reductions offered, by name, with their units."""

Spec = Mapping[str, object]


def reduction_axis(recipe: Recipe, kind: str) -> np.ndarray:
    """Return the axis a spectrum or a pulse is held on.

    Parameters
    ----------
    recipe : solprop.recipe.Recipe
        The scan's recipe.
    kind : str
        ``"spectrum"`` or ``"pulse"``.

    Returns
    -------
    numpy.ndarray
        Wavelengths across the run's window (m), or times across its time window (s).
    """
    grid = recipe.build().grid
    if kind == "spectrum":
        omega = np.asarray(grid.omega)
        active = (
            omega[np.asarray(grid.inband_mask)]
            if hasattr(grid, "inband_mask")
            else omega
        )
        live = active[active > 0]
        # Evenly spaced, so that a map of it is drawn true.
        return np.linspace(
            2 * np.pi * C / float(live.max()),
            2 * np.pi * C / float(live.min()),
            AXIS_POINTS,
        )
    t = np.asarray(grid.t)
    return np.linspace(float(t.min()), float(t.max()), AXIS_POINTS)


def _where(spec: Spec) -> dict[str, float | int]:
    """Return where a reduction reads: a distance, or nothing for the last plane.

    Parameters
    ----------
    spec : Mapping
        The reduction's spec, with ``z`` (m) or none.

    Returns
    -------
    dict
        The readers' ``z``, or the last ``plane``.
    """
    z = spec.get("z")
    return {"plane": -1} if z is None else {"z": float(z)}  # pyright: ignore[reportArgumentType]


def _band(spec: Spec) -> tuple[float, float]:
    """Return a reduction's band.

    Parameters
    ----------
    spec : Mapping
        The reduction's spec, with ``band``, two wavelengths in m.

    Returns
    -------
    tuple of float
        The band, ascending.
    """
    low, high = spec["band"]  # pyright: ignore[reportGeneralTypeIssues]
    return min(float(low), float(high)), max(float(low), float(high))


def _reader(spec: Spec, axis: np.ndarray | None) -> Callable[[Output], np.ndarray]:
    """Return what a reduction computes from a point's run.

    Parameters
    ----------
    spec : Mapping
        ``kind``, and ``band`` or ``z`` as it takes them.
    axis : numpy.ndarray or None
        A spectrum's or a pulse's fixed axis.

    Returns
    -------
    callable
        From the point's output to its value.
    """
    kind = str(spec["kind"])
    where = _where(spec)
    on: np.ndarray = np.zeros(0) if axis is None else axis

    def energy_in_band(output: Output) -> np.ndarray:
        """Return the energy within the band.

        Parameters
        ----------
        output : solprop.output.Output
            The point's run.

        Returns
        -------
        numpy.ndarray
            The energy within the band.
        """
        low, high = _band(spec)
        x, density = get_spectral_density(output, "wavelength", modes="sum", **where)  # pyright: ignore[reportArgumentType]
        inside = (np.asarray(x) >= low) & (np.asarray(x) <= high)
        values = np.asarray(density)[inside]
        return np.asarray(
            np.trapezoid(values, np.asarray(x)[inside]) if values.size > 1 else 0.0
        )

    def peak_power_in_band(output: Output) -> np.ndarray:
        """Return the peak power of the pulse filtered to the band.

        Parameters
        ----------
        output : solprop.output.Output
            The point's run.

        Returns
        -------
        numpy.ndarray
            The peak power of the pulse filtered to the band.
        """
        _t, power = get_power(
            output,
            modes="sum",
            bandpass=_band(spec),
            oversampling=TIME_OVERSAMPLING,
            **where,  # pyright: ignore[reportArgumentType]
        )  # pyright: ignore[reportArgumentType]
        return np.asarray(float(np.max(power)))

    def spectrum(output: Output) -> np.ndarray:
        """Return the spectral energy density on the fixed axis.

        Parameters
        ----------
        output : solprop.output.Output
            The point's run.

        Returns
        -------
        numpy.ndarray
            The spectral energy density on the fixed axis.
        """
        x, density = get_spectral_density(output, "wavelength", modes="sum", **where)  # pyright: ignore[reportArgumentType]
        order = np.argsort(x)
        return np.interp(on, np.asarray(x)[order], np.asarray(density)[order], 0.0, 0.0)

    def pulse(output: Output) -> np.ndarray:
        """Return the power on the fixed axis.

        Parameters
        ----------
        output : solprop.output.Output
            The point's run.

        Returns
        -------
        numpy.ndarray
            The power on the fixed axis.
        """
        t, power = get_power(
            output,
            modes="sum",
            oversampling=TIME_OVERSAMPLING,
            **where,  # pyright: ignore[reportArgumentType]
        )  # pyright: ignore[reportArgumentType]
        return np.interp(on, np.asarray(t), np.asarray(power), 0.0, 0.0)

    return {
        "energy_in_band": energy_in_band,
        "peak_power_in_band": peak_power_in_band,
        "spectrum": spectrum,
        "pulse": pulse,
    }[kind]


def reductions(specs: Sequence[Spec], recipe: Recipe) -> list[Reduction]:
    """Declare a scan's reductions from their specs.

    Parameters
    ----------
    specs : Sequence of Mapping
        Each ``kind`` (of :data:`KINDS`), with ``band`` (m) for the band's, and
        optionally ``z`` (m).
    recipe : solprop.recipe.Recipe
        The scan's recipe, whose axes the spectrum and the pulse are held on.

    Returns
    -------
    list of solprop.processing.scans.Reduction
        The reductions, each named by its kind.

    Raises
    ------
    ValueError
        For a beam's scan given any, before a point runs: each point's reading would
        fail.
    """
    if specs and recipe.route in BEAM_ROUTES:
        msg = (
            f"a scan of a {recipe.route} beam records its statistics only: the band, "
            "spectrum and pulse readers are of a guided run"
        )
        raise ValueError(msg)
    declared: list[Reduction] = []
    for spec in specs:
        kind = str(spec["kind"])
        axis = reduction_axis(recipe, kind) if kind in ("spectrum", "pulse") else None
        shape = () if axis is None else (axis.size,)
        declared.append(custom(kind, _reader(spec, axis), shape, KINDS[kind]))
    return declared
