# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read a radial beam from its file: across it, on its axis and in its far field.

A spatial file's columns are field samples, so the modal readers refuse it. These read
what those columns mean, each from the file's own record of the plane measure, with the
measures of :mod:`solprop.spatial`: the same local pairing of ``E`` and ``G`` as the
run's statistics. The far field is the plane's angular spectrum: the transverse
wavevector is conserved through a window into vacuum, so ``sin(theta) = c k_r / omega``
is the angle outside the medium, and a column beyond it has left no light there.
"""

from collections.abc import Mapping
from typing import cast

import numpy as np
from jaxtyping import Float

from solprop.constants import C
from solprop.exceptions import ProcessingError
from solprop.grid import RadialGrid
from solprop.output.schema import Output
from solprop.processing.axes import (
    SpectralAxis,
    grid_of,
    spatial_grid_of,
    spectral_axis,
)
from solprop.processing.fields import (
    DEFAULT_GATE_POSITIONS,
    _plane_field,
    _to_time,
    field_spectrogram,
    real_field_oversampling,
)
from solprop.spatial import (
    SpatialFlux,
    column_spectrum,
    node_spectrum,
    on_axis_field,
    on_axis_spectrum,
    on_axis_trace,
    spatial_fluence,
    spatial_intensity,
)
from solprop.translate.flux import flux_from_record

__all__ = [
    "FAR_FIELD_ANGLES",
    "far_field",
    "fluence_along",
    "on_axis_power",
    "on_axis_spectra",
    "on_axis_spectrogram",
    "radial_intensity",
    "radial_spectrum",
]

FAR_FIELD_ANGLES = 256
"""The angles a far field is read onto."""


def _radial_space(output: Output) -> RadialGrid:
    """Return a file's radial grid, from its record.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.

    Returns
    -------
    solprop.grid.RadialGrid
        The grid.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the run is not on a radial grid.
    """
    try:
        space = spatial_grid_of(output)
    except ProcessingError:
        space = None
    if not isinstance(space, RadialGrid):
        msg = "these readers are of a radial beam, and this run is not on a radial grid"
        raise ProcessingError(msg)
    return space


def _measured_plane(
    output: Output, plane: int | None, z: float | None
) -> tuple[np.ndarray, SpatialFlux]:
    """Return one saved plane's spectrum and its measure, from the file alone.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.
    plane : int or None
        The plane, negative from the end; the last when neither it nor ``z`` is given.
    z : float or None
        Its position, m: the nearest saved plane.

    Returns
    -------
    tuple
        The stored spectrum, ``(n_r, n_pol, n_omega)``, and the plane's measure.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the file's record carries no plane measure.
    """
    positions = np.asarray(output["z"], dtype=float)
    if z is not None:
        index = int(np.argmin(np.abs(positions - z)))
    else:
        index = -1 if plane is None else plane
    space = _radial_space(output)
    metadata = output.metadata
    assert metadata is not None  # _radial_space read it
    spatial = cast("Mapping[str, object]", metadata.spatial)
    if "flux_plan" not in spatial:
        msg = "this spatial output carries no plane measure (flux_plan, SEP 34)"
        raise ProcessingError(msg)
    plan = flux_from_record(cast("Mapping[str, object]", spatial["flux_plan"]))
    at = float(positions[index])
    flux = SpatialFlux(
        grid_of(output),
        space,
        None,  # a measure from the record alone
        at,
        None,
        plan.at(at),
    )
    return _plane_field(output, index, None, convention="fft"), flux


def _on_axis(
    omega: np.ndarray, axis: SpectralAxis
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the positive frequencies on a spectral axis, ascending.

    Parameters
    ----------
    omega : numpy.ndarray
        The grid's angular frequencies, rad/s.
    axis : {'f', 'omega', 'wavelength'}
        The abscissa.

    Returns
    -------
    tuple
        Which bins, the abscissa and the Jacobian from per rad/s, all ascending.
    """
    bins = np.flatnonzero(omega > 0)
    x, jacobian = spectral_axis(omega[bins], axis)
    order = np.argsort(x)
    return bins[order], x[order], jacobian[order]


def on_axis_spectra(
    output: Output, axis: SpectralAxis = "wavelength"
) -> tuple[Float[np.ndarray, " n_x"], Float[np.ndarray, "n_z n_x"]]:
    """Return the on-axis spectral fluence density of every plane that kept its field.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.
    axis : {'f', 'omega', 'wavelength'}, optional
        The spectral abscissa.

    Returns
    -------
    x : numpy.ndarray
        The abscissa, ascending.
    values : numpy.ndarray
        The density in J/m^2 per unit of ``x``, shape ``(n_planes, n_x)``, one row per
        plane that kept its field.
    """
    grid = grid_of(output)
    bins, x, jacobian = _on_axis(np.asarray(grid.omega), axis)
    rows = []
    for index in np.asarray(output["field_planes"]):
        spectrum, flux = _measured_plane(output, int(index), None)
        rows.append(on_axis_spectrum(spectrum, flux)[bins] * jacobian)
    return x, np.asarray(rows)


def radial_spectrum(
    output: Output,
    axis: SpectralAxis = "wavelength",
    *,
    plane: int | None = None,
    z: float | None = None,
) -> tuple[
    Float[np.ndarray, " n_r"], Float[np.ndarray, " n_x"], Float[np.ndarray, "n_r n_x"]
]:
    """Return one plane's spectral fluence density across the beam.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.
    axis : {'f', 'omega', 'wavelength'}, optional
        The spectral abscissa.
    plane : int or None, optional
        The plane, negative from the end; the last when neither it nor ``z`` is given.
    z : float or None, optional
        Its position, m: the nearest saved plane.

    Returns
    -------
    r : numpy.ndarray
        The radial nodes, m.
    x : numpy.ndarray
        The abscissa, ascending.
    values : numpy.ndarray
        The density in J/m^2 per unit of ``x``, shape ``(n_r, n_x)``; signed, as
        :func:`solprop.spatial.node_spectrum` says.
    """
    spectrum, flux = _measured_plane(output, plane, z)
    bins, x, jacobian = _on_axis(np.asarray(flux.grid.omega), axis)
    space = cast("RadialGrid", flux.space)
    return (
        np.asarray(space.r),
        x,
        node_spectrum(spectrum, flux)[:, bins] * jacobian,
    )


def far_field(
    output: Output,
    axis: SpectralAxis = "wavelength",
    *,
    plane: int | None = None,
    z: float | None = None,
    angles: int = FAR_FIELD_ANGLES,
) -> tuple[
    Float[np.ndarray, " n_theta"],
    Float[np.ndarray, " n_x"],
    Float[np.ndarray, "n_theta n_x"],
]:
    """Return one plane's far field: its energy per solid angle and per unit of ``x``.

    Conical emission reads as a ring in this map. Light at each frequency is placed by
    its transverse wavevector, ``sin(theta) = c k_r / omega``, the angle outside the
    medium, and the angles reach what the grid resolves at the highest frequency that
    holds light.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.
    axis : {'f', 'omega', 'wavelength'}, optional
        The spectral abscissa.
    plane : int or None, optional
        The plane, negative from the end; the last when neither it nor ``z`` is given.
    z : float or None, optional
        Its position, m: the nearest saved plane.
    angles : int, optional
        How many angles to read onto, evenly spaced from the axis.

    Returns
    -------
    theta : numpy.ndarray
        The angles, rad.
    x : numpy.ndarray
        The abscissa, ascending.
    values : numpy.ndarray
        The energy in J/sr per unit of ``x``, shape ``(n_theta, n_x)``.
    """
    spectrum, flux = _measured_plane(output, plane, z)
    grid = flux.grid
    omega = np.asarray(grid.omega)
    bins, x, jacobian = _on_axis(omega, axis)
    space = cast("RadialGrid", flux.space)
    k_r = np.asarray(space.k_r)
    table = column_spectrum(spectrum, flux)
    per_bin = table[:, bins] / float(grid.d_omega)
    edges = np.gradient(k_r)
    lit = per_bin.sum(axis=0) > 1e-6 * per_bin.sum(axis=0).max()
    reach = float(np.max(np.arcsin(np.minimum(C * k_r[-1] / omega[bins][lit], 1.0))))
    theta = np.linspace(0.0, reach, angles)
    values = np.zeros((angles, bins.size))
    for place, frequency in enumerate(omega[bins]):
        real = C * k_r / frequency < 1.0
        if not real.any():
            continue
        at, density = _per_solid_angle(
            per_bin[real, place], k_r[real], edges[real], float(frequency)
        )
        values[:, place] = np.interp(theta, at, density, right=0.0)
    return theta, x, values * jacobian


def _per_solid_angle(
    energy: np.ndarray, k_r: np.ndarray, widths: np.ndarray, omega: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return the energy in transverse-wavevector columns per unit solid angle.

    With :math:`k_r = k\sin\theta`, :math:`dk_r = k\cos\theta\,d\theta`, so a column
    spans :math:`2\pi k_r\,dk_r = k^2\cos\theta\,d\Omega`: its solid angle is
    :math:`2\pi k_r\,dk_r / (k^2\cos\theta)`, which grows towards grazing angles.

    Parameters
    ----------
    energy : numpy.ndarray
        Each column's energy, J per unit of the spectral axis.
    k_r : numpy.ndarray
        The columns' transverse wavevectors, 1/m, each below ``omega / C``.
    widths : numpy.ndarray
        Their widths in ``k_r``, 1/m.
    omega : float
        The angular frequency, rad/s.

    Returns
    -------
    tuple of numpy.ndarray
        The angles, rad, and the energy per steradian there.
    """
    sine = C * k_r / omega
    cosine = np.sqrt(1.0 - sine**2)
    solid = 2.0 * np.pi * sine * (C / omega) * widths / cosine
    return np.arcsin(sine), energy / solid


def fluence_along(
    output: Output,
) -> tuple[
    Float[np.ndarray, " n_planes"],
    Float[np.ndarray, " n_r"],
    Float[np.ndarray, "n_planes n_r"],
]:
    """Return the fluence across the beam at every plane that kept its field.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.

    Returns
    -------
    z : numpy.ndarray
        The planes' positions, m.
    r : numpy.ndarray
        The radial nodes, m.
    values : numpy.ndarray
        The signed fluence in J/m^2, shape ``(n_planes, n_r)``.
    """
    space = _radial_space(output)
    positions = np.asarray(output["z"], dtype=float)
    kept = np.asarray(output["field_planes"])
    rows = []
    for index in kept:
        spectrum, flux = _measured_plane(output, int(index), None)
        rows.append(spatial_fluence(spectrum, flux))
    return positions[kept], np.asarray(space.r), np.asarray(rows)


def radial_intensity(
    output: Output, *, plane: int | None = None, z: float | None = None
) -> tuple[
    Float[np.ndarray, " n_r"], Float[np.ndarray, " n_t"], Float[np.ndarray, "n_r n_t"]
]:
    """Return one plane's axial flux across the beam and in time.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.
    plane : int or None, optional
        The plane, negative from the end; the last when neither it nor ``z`` is given.
    z : float or None, optional
        Its position, m: the nearest saved plane.

    Returns
    -------
    r : numpy.ndarray
        The radial nodes, m.
    t : numpy.ndarray
        The times, s.
    values : numpy.ndarray
        The signed flux in W/m^2, shape ``(n_r, n_t)``.
    """
    spectrum, flux = _measured_plane(output, plane, z)
    space = cast("RadialGrid", flux.space)
    return (
        np.asarray(space.r),
        np.asarray(flux.grid.t),
        spatial_intensity(spectrum, flux),
    )


def on_axis_power(
    output: Output, *, plane: int | None = None, z: float | None = None
) -> tuple[Float[np.ndarray, " n_t"], Float[np.ndarray, " n_t"]]:
    """Return one plane's axial flux on the beam's axis against time.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.
    plane : int or None, optional
        The plane, negative from the end; the last when neither it nor ``z`` is given.
    z : float or None, optional
        Its position, m: the nearest saved plane.

    Returns
    -------
    t : numpy.ndarray
        The times, s.
    values : numpy.ndarray
        The flux in W/m^2.
    """
    spectrum, flux = _measured_plane(output, plane, z)
    return np.asarray(flux.grid.t), on_axis_trace(spectrum, flux)


def on_axis_spectrogram(  # noqa: PLR0913 - a plane, its axis and the gate
    output: Output,
    axis: SpectralAxis = "wavelength",
    *,
    plane: int | None = None,
    z: float | None = None,
    time_range: tuple[float, float] | None = None,
    gate_fwhm: float | None = None,
    positions: int = DEFAULT_GATE_POSITIONS,
) -> tuple[
    Float[np.ndarray, " n_slices"],
    Float[np.ndarray, " n_x"],
    Float[np.ndarray, "n_x n_slices"],
]:
    """Return the spectrogram of the field on the beam's axis.

    One point of a beam has one field, so this is well defined where a spectrogram of a
    spatial file's columns is not. Two polarisations are refused, as two fields.

    Parameters
    ----------
    output : solprop.output.schema.Output
        The run's output.
    axis : {'f', 'omega', 'wavelength'}, optional
        The spectral abscissa.
    plane : int or None, optional
        The plane, negative from the end; the last when neither it nor ``z`` is given.
    z : float or None, optional
        Its position, m: the nearest saved plane.
    time_range : tuple of float or None, optional
        The time range the gate positions cover, s; the whole window by default.
    gate_fwhm : float or None, optional
        Width of the gate, s; a twentieth of the time range by default.
    positions : int, optional
        How many gate positions to place across the time range.

    Returns
    -------
    t : numpy.ndarray
        The gate positions within the time range, s.
    x : numpy.ndarray
        The abscissa, ascending.
    values : numpy.ndarray
        The map, shape ``(n_x, n_slices)``, arbitrary units per unit of ``x``.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the beam has two polarisations, or as
        :func:`~solprop.processing.fields.field_spectrogram` does.
    """
    spectrum, flux = _measured_plane(output, plane, z)
    field = on_axis_field(spectrum, flux)
    if field.shape[0] != 1:
        msg = "a spectrogram is of one field, and this beam has two polarisations"
        raise ProcessingError(msg)
    grid = flux.grid
    t, amplitude = _to_time(
        grid,
        field,
        oversampling=real_field_oversampling(grid),
        bandpass=None,
        transfer=None,
        transform_limit=False,
    )
    trace = amplitude[0]
    if grid.kind == "envelope":
        # The stored envelope has its carrier removed: the real field has it back.
        trace = trace * np.exp(1j * float(grid.omega0) * t)
    return field_spectrogram(
        t,
        np.real(trace),
        axis,
        time_range=time_range,
        gate_fwhm=gate_fwhm,
        positions=positions,
    )
