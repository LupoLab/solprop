# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A free-space plane, measured: what the live run, the Overview and the Plane show.

A spatial run's columns are field samples across the beam, which the modal readers
refuse; it is measured by :mod:`solprop.spatial`'s functionals of the stored spectrum
with the run's ``SpatialFlux`` (its grids and medium at the plane), and by the energy
per frequency bin of the plane as ``processing.field_plane`` reads it. Both the run's
worker (from the plane observer's ``PlaneView.stored`` and ``field_plane``) and the
kernel (from the file) measure a plane here, so the live view and the results agree.

Three readings make a free-space plane's row:

- the energy per bin over the grid's active band, on wavelength and on frequency
  (energy per bin, not a density: the same bins read by either);
- the axial power through the plane (``spatial_power``), within a bandpass if one is
  given (the stored spectrum windowed by solprop's ``bandpass``);
- the fluence across the beam's centre (``spatial_fluence``): along r for a radial
  grid, along x for a line, and along the row nearest y = 0 for a Cartesian grid.
"""

from collections.abc import Mapping

import numpy as np
from solprop.grid import FreeGrid, LineGrid, RadialGrid, TimeGrid
from solprop.processing.fields import time_axis
from solprop.processing.spectra import bandpass as spectral_window
from solprop.spatial import (
    SpatialFlux,
    spatial_fluence,
    spatial_intensity,
    spatial_power,
)

from solprop_gui.service.reduce import TIME_OVERSAMPLING, C

__all__ = ["across", "beam_reading", "beam_slice", "profile_row"]


def across(space: object, values: np.ndarray) -> tuple[str, np.ndarray, np.ndarray]:
    """Return the nodes across a beam's centre, and values there.

    Parameters
    ----------
    space : solprop.grid.RadialGrid, LineGrid or FreeGrid
        The transverse grid.
    values : numpy.ndarray
        Values at its nodes, first axis in node order (``(n_nodes, ...)``).

    Returns
    -------
    tuple
        The axis's name (``"r"`` or ``"x"``), its coordinates (m) and the values on
        it: a Cartesian grid's row nearest y = 0.

    Raises
    ------
    TypeError
        For another transverse grid.
    """
    if isinstance(space, RadialGrid):
        return "r", np.asarray(space.r), values
    if isinstance(space, LineGrid):
        return "x", np.asarray(space.x), values
    if isinstance(space, FreeGrid):
        grid = values.reshape(space.n_y, space.n_x, *values.shape[1:])
        row = int(np.argmin(np.abs(np.asarray(space.y))))
        return "x", np.asarray(space.x), grid[row]
    msg = f"no profile across a {type(space).__name__}"
    raise TypeError(msg)


def _flux(grid: TimeGrid, space: object, medium: object, z: float) -> SpatialFlux:
    """Return the plane's measure: the run's grids and medium at the position.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The run's time grid.
    space : object
        Its transverse grid.
    medium : object
        Its medium.
    z : float
        The plane's position, m.

    Returns
    -------
    solprop.spatial.SpatialFlux
        The measure.
    """
    return SpatialFlux.resolve(grid, space, medium, z=z)  # pyright: ignore[reportArgumentType]


def beam_reading(  # noqa: PLR0913 - the run's grids, medium and the plane
    grid: TimeGrid,
    space: object,
    medium: object,
    z: float,
    stored: np.ndarray,
    energy: np.ndarray,
    *,
    bandpass: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Measure one free-space plane.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The run's time grid.
    space : object
        Its transverse grid.
    medium : object
        Its medium.
    z : float
        The plane's position, m.
    stored : numpy.ndarray
        The plane's stored spectrum, ``(n_modes, n_pol, n_omega)``.
    energy : numpy.ndarray
        Its energy per frequency bin (``field_plane(...).spectral_energy()``).
    bandpass : tuple of float or None, optional
        A wavelength band, m, the power is measured within.

    Returns
    -------
    dict
        ``spectra``, by axis (``"wavelength"``, ``"f"``), the bins' abscissae and
        energies over the active band; ``t`` and ``power`` (W, or W/m on a line);
        and ``profile``, the fluence across the centre: ``axis``, ``x`` (m) and
        ``y`` (J/m^2, or J/m on a line).
    """
    flux = _flux(grid, space, medium, z)
    omega = np.asarray(grid.omega)
    active = np.asarray(grid.active_mask) & (omega > 0)
    frequency = omega[active] / (2 * np.pi)
    per_bin = np.asarray(energy)[active]
    windowed = stored if bandpass is None else stored * spectral_window(grid, bandpass)
    axis, coordinates, fluence = across(
        space, np.asarray(spatial_fluence(stored, flux))
    )
    return {
        "spectra": {
            "wavelength": (C / frequency, per_bin),
            "f": (frequency, per_bin),
        },
        "t": time_axis(grid, TIME_OVERSAMPLING),
        "power": np.asarray(
            spatial_power(windowed, flux, oversampling=TIME_OVERSAMPLING)
        ),
        "profile": {"axis": axis, "x": coordinates, "y": fluence},
    }


def beam_slice(
    grid: TimeGrid,
    space: object,
    medium: object,
    z: float,
    stored: np.ndarray,
    *,
    bandpass: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Return the intensity across a free-space beam's centre against time.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The run's time grid.
    space : object
        Its transverse grid.
    medium : object
        Its medium.
    z : float
        The plane's position, m.
    stored : numpy.ndarray
        The plane's stored spectrum.
    bandpass : tuple of float or None, optional
        A wavelength band, m, the intensity is measured within.

    Returns
    -------
    dict
        ``t`` (s), ``x`` (m, across the centre), ``axis`` (``"r"`` or ``"x"``) and
        ``values``, the axial flux (``spatial_intensity``, W/m^2), shape
        ``(x.size, t.size)``.
    """
    flux = _flux(grid, space, medium, z)
    windowed = stored if bandpass is None else stored * spectral_window(grid, bandpass)
    intensity = spatial_intensity(windowed, flux, oversampling=TIME_OVERSAMPLING)
    axis, coordinates, values = across(space, np.asarray(intensity))
    return {
        "t": time_axis(grid, TIME_OVERSAMPLING),
        "x": coordinates,
        "axis": axis,
        "values": values,
    }


def profile_row(reading: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Return a reading's fluence profile as a row carries it.

    Parameters
    ----------
    reading : Mapping
        :func:`beam_reading`'s result.

    Returns
    -------
    dict
        ``axis``, and ``x`` and ``y`` as float32.

    Examples
    --------
    >>> import numpy as np
    >>> row = profile_row({"profile": {"axis": "r", "x": np.ones(2), "y": np.ones(2)}})
    >>> row["axis"], row["y"].dtype
    ('r', dtype('float32'))
    """
    profile: Mapping[str, object] = reading["profile"]  # pyright: ignore[reportAssignmentType]
    return {
        "axis": profile["axis"],  # pyright: ignore[reportReturnType]
        "x": np.asarray(profile["x"], dtype=np.float32),
        "y": np.asarray(profile["y"], dtype=np.float32),
    }
