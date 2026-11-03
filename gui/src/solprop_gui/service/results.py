# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's results requests: one plane, the statistics, a spectrogram, figures.

Each request opens a results file and asks :mod:`solprop.processing` (or
:mod:`solprop.plotting`, for what has no data function) for one view's numbers (design
11.3): a plane's spectrum, power and spectral phase; every recorded statistic and the
step history against z; a plane's spectrogram as data; and a figure of any of these
written by solprop's own plotting
(design 11.4). Like the other handlers, this module computes nothing of its own, and
runs only in the kernel process.
"""

import contextlib
import functools
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from solprop import capabilities, plotting
from solprop.driver import Simulation
from solprop.exceptions import SolpropError
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.output import Output, open_file
from solprop.processing import (
    ModeSelection,
    channel_labels,
    field_plane,
    get_phase,
    get_power,
    get_spectral_density,
    node_fluence,
    transverse_units,
)
from solprop.processing.fields import DEFAULT_GATE_POSITIONS, get_spectrogram
from solprop.processing.scans import ScanOutput
from solprop.processing.spectra import bandpass as spectral_window
from solprop.recipe import Recipe
from solprop.spatial import SpatialFlux, spatial_power
from solprop.stats import CATALOGUE

from solprop_gui.service.reduce import even_rows

__all__ = ["RESULTS_HANDLERS", "band_of", "spatial_plane"]

Payload = Mapping[str, object]

C = 299_792_458.0


def _modes(payload: Payload) -> ModeSelection:
    """Return the mode selection a payload asks for.

    Parameters
    ----------
    payload : Mapping
        Optionally ``modes``: ``"sum"`` or a mode's index.

    Returns
    -------
    int or str
        The selection, ``"sum"`` by default.
    """
    modes = payload.get("modes", "sum")
    return "sum" if modes == "sum" else int(modes)  # pyright: ignore[reportArgumentType]


def _labels(handler: Output) -> list[str]:
    """Return a file's mode labels, or none for a file without modes.

    Parameters
    ----------
    handler : solprop.output.Output
        The open file.

    Returns
    -------
    list of str
        Each channel's display label.
    """
    try:
        return [label.display_label for label in channel_labels(handler)]
    except SolpropError:
        return []


@functools.lru_cache(maxsize=4)
def _simulation(path: str, stamp: float) -> Simulation:
    """Return a run rebuilt from the recipe it records, for its grids and medium.

    Parameters
    ----------
    path : str
        The results file.
    stamp : float
        Its modification time, so that a rewritten file is rebuilt.

    Returns
    -------
    solprop.driver.Simulation
        The run's simulation, built and not run.
    """
    del stamp  # the cache's key only
    return Recipe.from_output(path, resolved=True).build()


def _space(path: str) -> object:
    """Return a spatial run's transverse grid (:func:`_simulation`'s).

    Parameters
    ----------
    path : str
        The results file.

    Returns
    -------
    object
        The ``spatial_grid``.
    """
    return _simulation(path, Path(path).stat().st_mtime).spatial_grid


def band_of(payload: Payload) -> tuple[float, float] | None:
    """Return a request's bandpass: a wavelength band, m, or none.

    Parameters
    ----------
    payload : Mapping
        The request; ``bandpass`` is read.

    Returns
    -------
    tuple of float or None
        The band, as solprop's ``bandpass`` keyword takes a pair.

    Examples
    --------
    >>> band_of({"bandpass": [7e-7, 9e-7]}), band_of({})
    ((7e-07, 9e-07), None)
    """
    band = payload.get("bandpass")
    if band is None:
        return None
    low, high = band  # pyright: ignore[reportGeneralTypeIssues]
    return float(low), float(high)


def spatial_plane(
    path: str,
    handler: Output,
    index: int,
    *,
    axis: str = "wavelength",
    bandpass: tuple[float, float] | None = None,
) -> dict[str, object]:
    """Return a spatial plane's physical spectrum and power, as solprop reads them.

    The column readers refuse a spatial file (its columns are field samples); the
    physical readings are the plane's energy per frequency bin
    (``field_plane(...).spectral_energy``) and the axial flux through it
    (``spatial.spatial_power``), the latter with the run's grids and medium.

    Parameters
    ----------
    path : str
        The results file.
    handler : solprop.output.Output
        It, open.
    index : int
        The plane.
    axis : {'wavelength', 'f'}, optional
        The spectrum's abscissa: each bin's wavelength (m) or frequency (Hz).
    bandpass : tuple of float or None, optional
        A wavelength band, m: the power is the flux of the field windowed by
        solprop's bandpass (:func:`solprop.processing.spectra.bandpass`).

    Returns
    -------
    dict
        ``spectrum`` (``x``, the wavelength or frequency of each in-band bin,
        ascending, and ``y``, J per bin) and ``power`` (``t`` and ``y`` in W), each
        with ``error`` instead where solprop gives none (a planar run's spectrum).
    """
    simulation = _simulation(path, Path(path).stat().st_mtime)
    z = float(np.asarray(handler["z"])[index])
    found: dict[str, object] = {}
    try:
        energy = np.asarray(field_plane(handler, index).spectral_energy())
        omega = np.asarray(simulation.grid.omega)
        light = omega > 0
        x = omega[light] / (2 * np.pi)
        if axis == "wavelength":
            x = C / x
        order = np.argsort(x)  # ascending, as the modal readers give it
        found["spectrum"] = {
            "x": x[order],
            "y": energy[light][order],
            "unit": "J per bin",
        }
    except SolpropError as error:
        found["spectrum"] = {"error": str(error)}
    flux = SpatialFlux.resolve(
        simulation.grid,
        simulation.spatial_grid,  # pyright: ignore[reportArgumentType]
        simulation.medium,  # pyright: ignore[reportArgumentType]
        z=z,
    )
    stored = np.asarray(handler.fields("fft")[index])  # pyright: ignore[reportAttributeAccessIssue]
    if bandpass is not None:
        stored = stored * spectral_window(simulation.grid, bandpass)
    found["power"] = {
        "t": np.asarray(simulation.grid.t),
        "y": np.asarray(spatial_power(stored, flux)),
    }
    return found


def _fluence(path: str, handler: Output, index: int) -> dict[str, object]:
    """Return a spatial plane's fluence on its transverse axes.

    Parameters
    ----------
    path : str
        The results file.
    handler : solprop.output.Output
        It, open.
    index : int
        The plane.

    Returns
    -------
    dict
        ``axes`` (their names and values, m: ``r``, ``x``, or ``y`` and ``x``),
        ``values`` (``node_fluence`` on them) and ``unit``; ``error`` instead
        where the run's grid cannot be rebuilt or laid out.
    """
    try:
        space = _space(path)
        values = np.asarray(node_fluence(handler))[index]
    except SolpropError as error:
        return {"error": str(error)}
    unit = transverse_units(handler).fluence
    if isinstance(space, RadialGrid):
        axes = {"r": np.asarray(space.r)}
    elif isinstance(space, LineGrid):
        axes = {"x": np.asarray(space.x)}
    elif isinstance(space, FreeGrid):
        axes = {"y": np.asarray(space.y), "x": np.asarray(space.x)}
        values = values.reshape(space.n_y, space.n_x)
    else:
        return {"error": "this run's transverse grid has no fluence map here"}
    return {"axes": axes, "values": values, "unit": unit}


def plane(payload: Payload) -> dict[str, object]:
    """Return one plane's spectrum, power and spectral phase.

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index`` (the plane), optionally ``modes`` (``"sum"`` or a
        mode's index), ``axis`` (``"wavelength"``, ``"f"`` or ``"omega"``) and
        ``bandpass`` (a wavelength band, m, the power is computed within).

    Returns
    -------
    dict
        ``z``, ``axis``, ``labels`` (the file's modes), ``spectrum`` (``x`` and
        ``y``, the spectral density on the axis), ``power`` (``t`` and ``y``) and
        ``phase`` (``omega`` and ``y``, of one mode; with ``error`` instead for a
        sum, which has no phase). On a spatial file the spectrum is the plane's
        energy per bin and the power the axial flux (:func:`spatial_plane`), with
        ``fluence`` (:func:`_fluence`) and no phase.
    """
    index = int(payload["index"])  # pyright: ignore[reportArgumentType]
    modes = _modes(payload)
    axis = str(payload.get("axis", "wavelength"))
    band = band_of(payload)
    path = str(payload["path"])
    caps = capabilities.output(path)
    with open_file(path) as handler:
        if caps.basis == "spatial":
            return {
                "z": float(np.asarray(handler["z"])[index]),
                "axis": axis,
                "labels": [],
                **spatial_plane(path, handler, index, axis=axis, bandpass=band),
                "phase": {"error": "a beam's columns are field samples: no phase"},
                "fluence": _fluence(path, handler, index),
            }
        x, density = get_spectral_density(handler, axis, plane=index, modes=modes)  # pyright: ignore[reportArgumentType]
        t, power = get_power(handler, plane=index, modes=modes, bandpass=band)
        phase: dict[str, object]
        try:
            omega, values = get_phase(
                handler, plane=index, modes=0 if modes == "sum" else modes
            )
            phase = {"omega": np.asarray(omega), "y": np.asarray(values)}
        except SolpropError as error:
            phase = {"error": str(error)}
        return {
            "z": float(np.asarray(handler["z"])[index]),
            "axis": axis,
            "labels": _labels(handler),
            "spectrum": {"x": np.asarray(x), "y": np.asarray(density)},
            "power": {"t": np.asarray(t), "y": np.asarray(power)},
            "phase": phase,
        }


def statistics(payload: Payload) -> dict[str, object]:
    """Return every recorded statistic, and the step history, against z.

    Parameters
    ----------
    payload : Mapping
        ``path``.

    Returns
    -------
    dict
        ``z``; ``stats``, per statistic its ``name``, ``summary``, ``units`` and
        ``columns`` (each a ``label`` and the ``y`` against z, one per mode and
        component of a statistic that has several); and ``steps``, the accepted
        steps' ``z`` and ``h`` with the rejected ones' (``rejected_z``,
        ``rejected_h``).
    """
    with open_file(str(payload["path"])) as handler:
        z = np.asarray(handler["z"])
        labels = _labels(handler)
        stats = []
        for name in handler.stat_names:
            values = np.asarray(handler[f"stats/{name}"], dtype=float)
            flat = values.reshape(values.shape[0], -1)
            columns = []
            for column in range(flat.shape[1]):
                position = np.unravel_index(column, values.shape[1:])
                if flat.shape[1] == 1:
                    text = name
                elif labels and len(position) >= 1 and position[0] < len(labels):
                    rest = ", ".join(str(int(p)) for p in position[1:])
                    text = labels[position[0]] + (f" [{rest}]" if rest else "")
                else:
                    text = f"{name} {list(map(int, position))}"
                columns.append({"label": text, "y": flat[:, column]})
            entry = CATALOGUE.get(name)
            stats.append(
                {
                    "name": name,
                    "summary": "" if entry is None else entry.summary,
                    "units": "" if entry is None else entry.units,
                    "columns": columns,
                }
            )
        accepted = np.asarray(handler["step_history/accepted"], dtype=bool)
        steps_z = np.asarray(handler["step_history/z"], dtype=float)
        steps_h = np.asarray(handler["step_history/h"], dtype=float)
    return {
        "z": z,
        "stats": stats,
        "steps": {
            "z": steps_z[accepted],
            "h": steps_h[accepted],
            "rejected_z": steps_z[~accepted],
            "rejected_h": steps_h[~accepted],
        },
    }


def _drawn(handler: Output, payload: Payload) -> list[Figure]:
    """Draw one of solprop's plots of a file.

    Parameters
    ----------
    handler : solprop.output.Output
        The open file.
    payload : Mapping
        ``plot``: ``"prop_2d"``, ``"spec_1d"``, ``"time_1d"``, ``"spectrogram"``
        or ``"plot_stats"``; ``z`` (a plane's position, for the slices and the
        spectrogram); and optionally ``modes``.

    Returns
    -------
    list of matplotlib.figure.Figure
        The figures: one for a slice or a spectrogram, one per map or statistic
        for the others.

    Raises
    ------
    ValueError
        If the plot is not one of these.
    """
    kind = str(payload["plot"])
    modes = _modes(payload)
    z = payload.get("z")
    position = None if z is None else float(z)  # pyright: ignore[reportArgumentType]
    drawn: Figure | list[Figure]
    if kind == "prop_2d":
        drawn = plotting.prop_2d(handler, modes=modes)
    elif kind == "spec_1d":
        drawn = plotting.spec_1d(handler, position, modes=modes)
    elif kind == "time_1d":
        drawn = plotting.time_1d(handler, position, modes=modes)
    elif kind == "spectrogram":
        # A field, so one mode: a sum of modes has no phase to gate.
        drawn = plotting.spectrogram(
            handler,
            position,
            modes=0 if modes == "sum" else modes,
            log=bool(payload.get("log", True)),
        )
    elif kind == "plot_stats":
        drawn = plotting.plot_stats(handler)
    else:
        msg = f"no plot named {kind!r}"
        raise ValueError(msg)
    return drawn if isinstance(drawn, list) else [drawn]


SPECTROGRAM_ROWS = 400
"""The rows of a spectrogram's image: its spectral bins, as the window shows them."""


def spectrogram(payload: Payload) -> dict[str, object]:
    """Return a plane's spectrogram as data, for an interactive image.

    The map is solprop's (``processing.get_spectrogram``), per unit of the axis asked
    for; its rows are put on even bins within the run's wavelength window for the
    image (:func:`.reduce.even_rows`).

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index`` (the plane), and optionally ``modes`` (a sum takes the
        first mode: modal fields do not add away from a point), ``axis``
        (``"wavelength"`` or ``"f"``), ``time_range`` (a pair, s), ``gate_fwhm``
        (s), ``positions`` and ``bandpass`` (a wavelength band, m).

    Returns
    -------
    dict
        ``t`` (the gate positions, s), ``x`` (the even rows, m or Hz), ``values``
        (the map, shape ``(x.size, t.size)``), ``axis``, ``mode`` (the mode's
        index) and ``gate_fwhm`` (the gate solprop used, s).

    Raises
    ------
    solprop.exceptions.SolpropError
        For a free-space run, whose columns are field samples rather than modes.
    """
    path = str(payload["path"])
    caps = capabilities.output(path)
    if caps.basis == "spatial":
        msg = (
            "a spectrogram is of one field, and a free-space run's columns are "
            "field samples across the beam: none is drawn for it"
        )
        raise SolpropError(msg)
    axis = str(payload.get("axis") or "wavelength")
    modes = _modes(payload)
    mode = 0 if modes == "sum" else modes
    time_range = band_of({"bandpass": payload.get("time_range")})
    gate = payload.get("gate_fwhm")
    with open_file(path) as handler:
        t, x, values = get_spectrogram(
            handler,
            axis,  # pyright: ignore[reportArgumentType]
            plane=int(payload["index"]),  # pyright: ignore[reportArgumentType]
            modes=mode,
            time_range=time_range,
            gate_fwhm=None if gate is None else float(gate),  # pyright: ignore[reportArgumentType]
            positions=int(payload.get("positions") or DEFAULT_GATE_POSITIONS),  # pyright: ignore[reportArgumentType]
            bandpass=band_of(payload),
        )
    window = _window(path, axis, x, values)
    rows, image = even_rows(x, values, window, SPECTROGRAM_ROWS)
    span = float(t[-1] - t[0]) if time_range is None else abs(np.ptp(time_range))
    return {
        "t": t,
        "x": rows,
        "values": image.astype(np.float32),
        "axis": axis,
        "mode": int(mode),  # pyright: ignore[reportArgumentType]
        "gate_fwhm": float(gate) if gate is not None else span / 20.0,  # pyright: ignore[reportArgumentType]
    }


def _window(
    path: str, axis: str, x: np.ndarray, values: np.ndarray
) -> tuple[float, float]:
    """Return the spectral window of a spectrogram's image.

    Parameters
    ----------
    path : str
        The results file.
    axis : str
        The map's axis.
    x : numpy.ndarray
        Its rows' abscissae, ascending.
    values : numpy.ndarray
        The map.

    Returns
    -------
    tuple of float
        The run's wavelength window on the axis, where its recipe gives one; else
        the rows that hold light (above a millionth of the peak).
    """
    limits = None
    if capabilities.output(path).recipe:
        with contextlib.suppress(SolpropError):
            limits = Recipe.from_output(path).keywords().get("wavelength_limits")
    if isinstance(limits, tuple | list) and len(limits) == 2:
        low, high = sorted(float(v) for v in limits)  # pyright: ignore[reportArgumentType]
        if axis == "wavelength":
            return low, high
        return C / high, C / low
    light = np.flatnonzero(values.max(axis=1) > 1e-6 * values.max())
    return float(x[light[0]]), float(x[light[-1]])


def figure(payload: Payload) -> list[str]:
    """Write one of solprop's plots of a file as figures.

    Parameters
    ----------
    payload : Mapping
        ``path`` (the results file), ``file`` (the figure, ``.png``, ``.svg`` or
        ``.pdf``), and the plot's own keys (:func:`_drawn`).

    Returns
    -------
    list of str
        The figures written: the file, or for several figures the file's name
        numbered (``maps-1.pdf``, ``maps-2.pdf``).
    """
    target = Path(str(payload["file"]))
    with open_file(str(payload["path"])) as handler:
        drawn = _drawn(handler, payload)
    written = []
    for number, one in enumerate(drawn, start=1):
        path = (
            target
            if len(drawn) == 1
            else target.with_name(f"{target.stem}-{number}{target.suffix}")
        )
        one.savefig(path, dpi=200)
        written.append(str(path))
    return written


def scan_overview(payload: Payload) -> dict[str, object]:
    """Return a scan's variables, its points' status and every result over the grid.

    Each result a point records (``stats/*``, and ``reduced/*`` where the scan asked
    for reductions) is laid out over the scan's axes by ``ScanOutput.as_grid``; per
    point the value at its last saved plane, summed over its modes and components,
    is the number a browser maps. A point not done is NaN.

    Parameters
    ----------
    payload : Mapping
        ``path``, the scan's results file.

    Returns
    -------
    dict
        ``variables`` (each ``name`` and ``values``, in its keyword's unit, in the
        scan's axis order),
        ``shape``, ``status`` (each point's, laid out over the shape) and
        ``results`` (by name: ``summary``, ``units`` and ``values`` over the shape).
    """
    found = ScanOutput(str(payload["path"]))
    record = found.recipe
    if record is None:
        msg = "the scan records no recipe, so its variables are not known"
        raise SolpropError(msg)
    variables = Recipe.loads(record.authored).scan().scan.variables
    shape = tuple(int(n) for n in found.shape)
    status = np.asarray(found.status).reshape(shape)
    done = np.flatnonzero(np.asarray(found.status) == "done")
    names = (
        []
        if done.size == 0
        else sorted(
            name
            for name in found.point(int(done[0]))
            if name.startswith(("stats/", "reduced/"))
        )
    )
    values: dict[str, object] = {}
    for name in names:
        grid = found.as_grid(name)
        last = np.ma.take(grid, -1, axis=len(shape)) if grid.ndim > len(shape) else grid
        while last.ndim > len(shape):
            last = last.sum(axis=-1)
        entry = CATALOGUE.get(name.split("/", 1)[1])
        values[name] = {
            "summary": "" if entry is None else entry.summary,
            "units": "" if entry is None else entry.units,
            "values": np.ma.filled(np.ma.asarray(last, dtype=float), np.nan),
        }
    return {
        "variables": [
            {"name": str(k), "values": np.asarray(v, dtype=float)}
            for k, v in variables.items()
        ],
        "shape": list(shape),
        "status": status.tolist(),
        "results": values,
    }


RESULTS_HANDLERS = {
    "plane": plane,
    "statistics": statistics,
    "spectrogram": spectrogram,
    "result_figure": figure,
    "scan_overview": scan_overview,
}
"""The results requests, by kind."""
