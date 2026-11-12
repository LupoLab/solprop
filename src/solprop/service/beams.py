# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a radial beam's pages show, read from its file with solprop's readers."""

from collections.abc import Mapping

import numpy as np

from solprop.output import open_file
from solprop.processing import (
    far_field,
    fluence_along,
    fluence_map,
    on_axis_power,
    on_axis_spectra,
    on_axis_spectrogram,
    radial_intensity,
    radial_spectrum,
)
from solprop.service.reduce import TIME_OVERSAMPLING, even_rows
from solprop.service.results import _window

__all__ = [
    "BEAM_HANDLERS",
    "beam_far_field",
    "beam_fluence",
    "beam_line",
    "beam_on_axis",
    "beam_overview",
    "beam_plane",
]

Payload = Mapping[str, object]

_ROWS = 400
"""The spectral bins of a beam's maps, even on their axis."""


def _even(
    path: str, axis: str, x: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Put a map's columns on even bins within the run's window.

    - ``axis``: the spectral axis.
    - ``values``: the map, shape ``(n_rows, x.size)``.
    - Returns: the even positions, and the map on them as single precision.
    """
    bins, image = even_rows(x, values.T, _window(path, axis, x, values.T), _ROWS)
    return bins, image.T.astype(np.float32)


def beam_overview(payload: Payload) -> dict[str, object]:
    """Return the fluence along the propagation and the on-axis spectrum.

    Parameters
    ----------
    payload : Mapping
        ``path``, and optionally ``axis``.

    Returns
    -------
    dict
        ``z``, ``r`` (m), ``fluence`` (J/m^2, ``(n_planes, n_r)``), ``axis``, ``x`` and
        ``on_axis``, the on-axis spectral density on even bins, ``(n_planes, x.size)``.
    """
    path = str(payload["path"])
    axis = str(payload.get("axis") or "wavelength")
    with open_file(path) as output:
        z, r, fluence = fluence_along(output)
        x, on_axis = on_axis_spectra(output, axis)  # pyright: ignore[reportArgumentType]
    bins, image = _even(path, axis, x, on_axis)
    return {
        "z": z,
        "r": r,
        "fluence": fluence.astype(np.float32),
        "axis": axis,
        "x": bins,
        "on_axis": image,
    }


def beam_plane(payload: Payload) -> dict[str, object]:
    """Return one plane across the beam: in time and in its spectrum.

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index``, and optionally ``axis``.

    Returns
    -------
    dict
        ``r`` and ``t``, ``intensity`` (W/m^2, ``(n_r, n_t)``), ``fluence`` (J/m^2, that
        integrated in time), ``axis``, ``x`` and ``spectrum``, the spectral fluence
        density on even bins, ``(n_r, x.size)``.
    """
    path = str(payload["path"])
    axis = str(payload.get("axis") or "wavelength")
    index = int(payload["index"])  # pyright: ignore[reportArgumentType]
    with open_file(path) as output:
        r, t, intensity = radial_intensity(output, plane=index)
        _r, x, spectrum = radial_spectrum(output, axis, plane=index)  # pyright: ignore[reportArgumentType]
    bins, image = _even(path, axis, x, spectrum)
    return {
        "r": r,
        "t": t,
        "intensity": intensity.astype(np.float32),
        "fluence": intensity.sum(axis=1) * float(t[1] - t[0]),
        "axis": axis,
        "x": bins,
        "spectrum": image,
    }


def beam_on_axis(payload: Payload) -> dict[str, object]:
    """Return one plane on the beam's axis: its pulse, spectrum and spectrogram.

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index``, and optionally ``axis``, ``gate_fwhm`` and ``positions``.

    Returns
    -------
    dict
        ``t`` and ``power`` (W/m^2), ``axis``, ``x`` and ``spectrum``, and
        ``spectrogram``: its ``t``, ``x``, ``values`` and ``gate_fwhm``.
    """
    path = str(payload["path"])
    axis = str(payload.get("axis") or "wavelength")
    index = int(payload["index"])  # pyright: ignore[reportArgumentType]
    gate = payload.get("gate_fwhm")
    with open_file(path) as output:
        t, power = on_axis_power(output, plane=index, oversampling=TIME_OVERSAMPLING)
        kept = np.asarray(output["field_planes"])
        x, rows = on_axis_spectra(output, axis)  # pyright: ignore[reportArgumentType]
        spectrum = rows[int(np.flatnonzero(kept == index)[0])]
        map_t, map_x, values = on_axis_spectrogram(
            output,
            axis,  # pyright: ignore[reportArgumentType]
            plane=index,
            gate_fwhm=None if gate is None else float(gate),  # pyright: ignore[reportArgumentType]
            positions=int(payload.get("positions") or 256),  # pyright: ignore[reportArgumentType]
        )
    bins, image = even_rows(map_x, values, _window(path, axis, map_x, values), _ROWS)
    span = float(map_t[-1] - map_t[0])
    # In arbitrary units, and a field in V/m squares beyond single precision.
    image = image / (float(image.max()) or 1.0)
    return {
        "t": t,
        "power": power,
        "axis": axis,
        "x": x,
        "spectrum": spectrum,
        "spectrogram": {
            "t": map_t,
            "x": bins,
            "values": image.astype(np.float32),
            "gate_fwhm": float(gate) if gate is not None else span / 20.0,  # pyright: ignore[reportArgumentType]
        },
    }


def beam_far_field(payload: Payload) -> dict[str, object]:
    """Return one plane's far field.

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index``, and optionally ``axis``.

    Returns
    -------
    dict
        ``theta`` (rad), ``axis``, ``x`` and ``values`` (J/sr per unit of ``x``, on even
        bins, ``(theta.size, x.size)``).
    """
    path = str(payload["path"])
    axis = str(payload.get("axis") or "wavelength")
    index = int(payload["index"])  # pyright: ignore[reportArgumentType]
    with open_file(path) as output:
        theta, x, values = far_field(output, axis, plane=index)  # pyright: ignore[reportArgumentType]
    bins, image = _even(path, axis, x, values)
    return {"theta": theta, "axis": axis, "x": bins, "values": image}


def beam_fluence(payload: Payload) -> dict[str, object]:
    """Return one plane's fluence across a Cartesian beam.

    Parameters
    ----------
    payload : Mapping
        ``path`` and ``index``, the plane.

    Returns
    -------
    dict
        ``z`` (m), ``x`` and ``y`` (m) and ``fluence`` (J/m^2, rows along y).
    """
    with open_file(str(payload["path"])) as output:
        found = fluence_map(output, int(payload["index"]))  # pyright: ignore[reportArgumentType]
    axes = dict(zip(found.axis_names, found.axes, strict=True))
    return {
        "z": float(found.z),
        "x": np.asarray(axes["x"]),
        "y": np.asarray(axes["y"]),
        "fluence": np.asarray(found.fluence),
    }


def beam_line(payload: Payload) -> dict[str, object]:
    """Return a line beam's fluence across x at every saved plane.

    Parameters
    ----------
    payload : Mapping
        ``path``.

    Returns
    -------
    dict
        ``z`` (m, the planes), ``x`` (m) and ``fluence`` (J/m^2 per plane, rows along
        z).
    """
    with open_file(str(payload["path"])) as output:
        count = int(np.asarray(output["z"]).size)
        planes = [fluence_map(output, index) for index in range(count)]
    return {
        "z": np.array([plane.z for plane in planes], dtype=float),
        "x": np.asarray(planes[0].axes[0]),
        "fluence": np.array([np.asarray(plane.fluence) for plane in planes]),
    }


BEAM_HANDLERS = {
    "beam_far_field": beam_far_field,
    "beam_fluence": beam_fluence,
    "beam_line": beam_line,
    "beam_on_axis": beam_on_axis,
    "beam_overview": beam_overview,
    "beam_plane": beam_plane,
}
"""The beam pages' requests, by kind."""
