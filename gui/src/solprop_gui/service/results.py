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
from solprop.exceptions import OutputError, SolpropError
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
    physical_energy,
    transverse_units,
)
from solprop.processing.fields import DEFAULT_GATE_POSITIONS, get_spectrogram
from solprop.processing.scans import ScanOutput
from solprop.processing.spectra import bandpass as spectral_window
from solprop.processing.time import energy, fwhm_t
from solprop.recipe import Recipe
from solprop.spatial import SpatialFlux, spatial_power
from solprop.stats import CATALOGUE

from solprop_gui.model.display import light_range
from solprop_gui.model.quantities import split
from solprop_gui.model.recipe import without_table
from solprop_gui.service.beam import beam_reading, beam_slice, profile_row
from solprop_gui.service.reduce import even_rows
from solprop_gui.service.scan_reductions import KINDS, reduction_axis

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
        ``spectrum`` (``x``, the wavelength or frequency of each bin of the active
        band, ascending, and ``y``, J per bin), ``power`` (``t`` and ``y`` in W) and
        ``profile`` (the fluence across the centre, :func:`.beam.profile_row`); a
        spectrum with ``error`` instead, and no profile, where solprop reads no
        field plane (a planar run).
    """
    simulation = _simulation(path, Path(path).stat().st_mtime)
    z = float(np.asarray(handler["z"])[index])
    stored = np.asarray(handler.fields("fft")[index])  # pyright: ignore[reportAttributeAccessIssue]
    space = (simulation.grid, simulation.spatial_grid, simulation.medium)
    try:
        energy = np.asarray(field_plane(handler, index).spectral_energy())
    except SolpropError as error:
        # A plane solprop does not read as a field plane (a planar run): its flux.
        flux = SpatialFlux.resolve(*space, z=z)  # pyright: ignore[reportArgumentType]
        windowed = stored
        if bandpass is not None:
            windowed = stored * spectral_window(simulation.grid, bandpass)
        return {
            "spectrum": {"error": str(error)},
            "power": {
                "t": np.asarray(simulation.grid.t),
                "y": np.asarray(spatial_power(windowed, flux)),
            },
        }
    reading = beam_reading(*space, z, stored, energy, bandpass=bandpass)
    spectra: Mapping[str, tuple[np.ndarray, np.ndarray]] = reading["spectra"]  # pyright: ignore[reportAssignmentType]
    x, y = spectra["f" if axis == "f" else "wavelength"]
    order = np.argsort(x)  # ascending, as the modal readers give it
    return {
        "spectrum": {"x": x[order], "y": y[order], "unit": "J per bin"},
        "power": {"t": reading["t"], "y": reading["power"]},
        "profile": profile_row(reading),
    }


def beam(payload: Payload) -> dict[str, object]:
    """Return a free-space plane's intensity across the beam's centre against time.

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index`` (the plane) and optionally ``bandpass`` (a wavelength
        band, m).

    Returns
    -------
    dict
        :func:`.beam.beam_slice`'s: ``t``, ``x``, ``axis`` and ``values`` (W/m^2),
        the values as float32.

    Raises
    ------
    solprop.exceptions.SolpropError
        For a run that is not a free-space one.
    """
    path = str(payload["path"])
    if capabilities.output(path).basis != "spatial":
        msg = "only a free-space run has a beam to slice"
        raise SolpropError(msg)
    simulation = _simulation(path, Path(path).stat().st_mtime)
    index = int(payload["index"])  # pyright: ignore[reportArgumentType]
    with open_file(path) as handler:
        z = float(np.asarray(handler["z"])[index])
        stored = np.asarray(handler.fields("fft")[index])  # pyright: ignore[reportAttributeAccessIssue]
    found = beam_slice(
        simulation.grid,
        simulation.spatial_grid,
        simulation.medium,
        z,
        stored,
        bandpass=band_of(payload),
    )
    return {**found, "values": np.asarray(found["values"], dtype=np.float32)}


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
        sum, which has no phase), ``window`` (the run's own range on the axis),
        ``energy`` (J) and ``fwhm`` (s, of the power shown);
        with a band, ``full_power`` (the power of the whole spectrum) and
        ``band_energy`` (J). On a spatial file the spectrum is the plane's
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
        reply: dict[str, object] = {
            "z": float(np.asarray(handler["z"])[index]),
            "axis": axis,
            "labels": _labels(handler),
            "spectrum": {"x": np.asarray(x), "y": np.asarray(density)},
            "power": {"t": np.asarray(t), "y": np.asarray(power)},
            "phase": phase,
            "window": _window(path, axis, np.asarray(x), np.asarray(density)),
            "energy": _energy_of(handler, index, modes),
            "fwhm": float(fwhm_t(handler, plane=index, modes=modes, bandpass=band)),
        }
        if band is not None:
            _t, full = get_power(handler, plane=index, modes=modes)
            reply["full_power"] = np.asarray(full)
            reply["band_energy"] = float(
                energy(handler, plane=index, modes=modes, bandpass=band)
            )
        return reply


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
                    # Only what varies names a column: a lone polarisation does not.
                    rest = ", ".join(
                        str(int(p))
                        for p, size in zip(position[1:], values.shape[2:], strict=True)
                        if size > 1
                    )
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


RUN_LOG = "run.log"
"""The log a run in the workspace keeps beside its results."""

SPECTROGRAM_ROWS = 400
"""The rows of a spectrogram's image: its spectral bins, as the window shows them."""


def _health_facts(execution: Mapping[str, object]) -> dict[str, object]:
    """Return what a run's record says about how it ran.

    Parameters
    ----------
    execution : Mapping
        The run's ``execution`` record.

    Returns
    -------
    dict
        ``backend``, ``precision``, ``threads``, ``wall_time`` (s), ``peak_memory``
        (bytes) and ``plasma_source`` (its check and limit), each ``None`` where the
        record has none.
    """
    resources = execution.get("resources")
    plasma = execution.get("plasma_source")
    held = resources if isinstance(resources, Mapping) else {}
    source = plasma if isinstance(plasma, Mapping) else {}
    return {
        "backend": execution.get("backend"),
        "precision": execution.get("precision"),
        "threads": execution.get("thread_count"),
        "wall_time": held.get("wall_time_s"),
        "peak_memory": held.get("peak_rss_bytes"),
        "plasma_source": (
            None
            if "certificate" not in source
            else [source["certificate"], source.get("limit")]
        ),
    }


def _boundary_of(execution: Mapping[str, object]) -> dict[str, object] | None:
    """Return what the boundary monitor recorded along z.

    Parameters
    ----------
    execution : Mapping
        The run's ``execution`` record.

    Returns
    -------
    dict or None
        ``z``, ``removed`` (the share of the light the absorber took, by position)
        and ``budget``; ``None`` where no monitor ran.
    """
    monitor = execution.get("boundary_monitor")
    planes = monitor.get("planes") if isinstance(monitor, Mapping) else None
    if not isinstance(monitor, Mapping) or not isinstance(planes, Mapping):
        return None
    return {
        "z": np.asarray(planes.get("z", ()), dtype=float),
        "removed": np.asarray(planes.get("removed_fraction", ()), dtype=float),
        "budget": monitor.get("budget"),
    }


def run_health(payload: Payload) -> dict[str, object]:
    """Return how a run went: its steps, its boundary and its record.

    Parameters
    ----------
    payload : Mapping
        ``path``.

    Returns
    -------
    dict
        ``status`` and ``message`` (how it ended), ``steps`` (as :func:`statistics`
        gives them), ``boundary`` (:func:`_boundary_of`), ``facts``
        (:func:`_health_facts`) and ``log``, the run's log file where it is beside
        the results.
    """
    path = Path(str(payload["path"]))
    with open_file(str(path)) as handler:
        outcome = handler.outcome
        accepted = np.asarray(handler["step_history/accepted"], dtype=bool)
        steps_z = np.asarray(handler["step_history/z"], dtype=float)
        steps_h = np.asarray(handler["step_history/h"], dtype=float)
    execution = {} if outcome is None else outcome.execution
    log = path.with_name(RUN_LOG)
    return {
        "status": None if outcome is None else outcome.status,
        "message": None if outcome is None else outcome.message,
        "steps": {
            "z": steps_z[accepted],
            "h": steps_h[accepted],
            "rejected_z": steps_z[~accepted],
            "rejected_h": steps_h[~accepted],
        },
        "boundary": _boundary_of(execution),
        "facts": _health_facts(execution),
        "log": log.read_text(encoding="utf-8") if log.exists() else None,
    }


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
        The run's wavelength window on the axis, where its recipe gives one (on
        frequency from the map's lowest row, as a field grid reaches zero); else
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
        # On frequency, from the transform's lowest: a field grid reaches zero.
        return float(x[0]), C / low
    # A file with no mode axis gives one column; a map has its positions in rows.
    return light_range(x, np.moveaxis(values.reshape(x.size, -1), 0, -1))


def _energy_of(handler: Output, index: int, modes: ModeSelection) -> float:
    """Return a plane's energy, through the metric where the modes' basis needs it.

    Parameters
    ----------
    handler : solprop.output.Output
        The open file.
    index : int
        The plane.
    modes : ModeSelection
        The modes.

    Returns
    -------
    float
        The energy, J: a general modal guide's is its physical energy, of every mode.
    """
    try:
        return float(energy(handler, plane=index, modes=modes))
    except OutputError:
        return float(physical_energy(handler, plane=index))


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
    recipe = _scan_recipe(str(payload["path"]))
    found = ScanOutput(str(payload["path"]))
    variables = recipe.scan().scan.variables
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
    values = {name: _scan_result(found, name, shape, recipe) for name in names}
    return {
        "variables": [_scan_axis(str(k), v) for k, v in variables.items()],
        "shape": list(shape),
        "status": status.tolist(),
        "results": values,
    }


def _scan_recipe(path: str) -> Recipe:
    """Return the recipe a scan's file records, as it was written.

    Parameters
    ----------
    path : str
        The scan's results file.

    Returns
    -------
    solprop.recipe.Recipe
        The recipe, with its ``[scan]`` table.

    Raises
    ------
    solprop.exceptions.SolpropError
        If the file records no recipe.
    """
    record = ScanOutput(path).recipe
    if record is None:
        msg = "the scan records no recipe, so its variables are not known"
        raise SolpropError(msg)
    return Recipe.loads(record.authored)


def scan_point(payload: Payload) -> dict[str, object]:
    """Return one scan point's setup: the scan's recipe with that point's values.

    Parameters
    ----------
    payload : Mapping
        ``path``, the scan's results file, and ``index``, the point's flat index.

    Returns
    -------
    dict
        ``text``, the recipe without its ``[scan]`` table and with the point's values
        set, and ``values``, those values by path.
    """
    recipe = _scan_recipe(str(payload["path"]))
    point = recipe.scan().scan.point(int(payload["index"]))  # pyright: ignore[reportArgumentType]
    values = {path: value for path, value in point.values.items() if path != "seed"}
    single = Recipe.loads(without_table(recipe.to_text(), "scan"))
    for path, value in values.items():
        single.set(path, value)
    return {"text": single.to_text(), "values": values}


def _scan_axis(name: str, values: object) -> dict[str, object]:
    """Return a scan variable's values as numbers, with the unit they were written in.

    A keyword's values are read in SI; a nested path's (``pressure[0]``) are the
    quantities as the recipe writes them.

    Parameters
    ----------
    name : str
        The variable's path.
    values : object
        Its values.

    Returns
    -------
    dict
        ``name``, ``values`` and ``unit`` (``""`` where they are SI numbers).

    Examples
    --------
    >>> _scan_axis("pressure[0]", np.array(["0.8 bar", "1.2 bar"]))["values"].tolist()
    [0.8, 1.2]
    >>> _scan_axis("energy", np.array([1e-4]))["unit"]
    ''
    """
    held = np.asarray(values).ravel()
    if held.dtype.kind in "fiu":
        return {"name": name, "values": held.astype(float), "unit": ""}
    parts = [split(str(value)) for value in held]
    return {
        "name": name,
        "values": np.array([float(number) for number, _unit in parts]),
        "unit": parts[0][1] if parts else "",
    }


def _scan_result(
    found: ScanOutput, name: str, shape: tuple[int, ...], recipe: Recipe
) -> dict[str, object]:
    """Return one recorded result over a scan's points.

    Parameters
    ----------
    found : solprop.processing.scans.ScanOutput
        The scan.
    name : str
        The result: a statistic (read at the last plane, summed over its other axes)
        or a reduction (a value, or a spectrum or pulse on its fixed axis).
    shape : tuple of int
        The scan's shape.
    recipe : solprop.recipe.Recipe
        The scan's recipe, whose axes a spectrum or a pulse is held on.

    Returns
    -------
    dict
        ``summary``, ``units`` and ``values``; a spectrum or a pulse also its ``x``
        (m or s).
    """
    grid = found.as_grid(name)
    short = name.split("/", 1)[1]
    if name.startswith("reduced/"):
        entry: dict[str, object] = {
            "summary": _REDUCED.get(short, short),
            "units": KINDS.get(short, ""),
            "values": np.ma.filled(np.ma.asarray(grid, dtype=float), np.nan),
        }
        if grid.ndim > len(shape):
            entry["x"] = reduction_axis(recipe, short)
        return entry
    last = np.ma.take(grid, -1, axis=len(shape)) if grid.ndim > len(shape) else grid
    while last.ndim > len(shape):
        last = last.sum(axis=-1)
    catalogued = CATALOGUE.get(short)
    return {
        "summary": "" if catalogued is None else catalogued.summary,
        "units": "" if catalogued is None else catalogued.units,
        "values": np.ma.filled(np.ma.asarray(last, dtype=float), np.nan),
    }


_REDUCED = {
    "energy_in_band": "Energy within the band.",
    "peak_power_in_band": "Peak power of the pulse filtered to the band.",
    "spectrum": "Spectral energy density.",
    "pulse": "Power against time.",
}
"""What each of the front end's reductions records."""


RESULTS_HANDLERS = {
    "plane": plane,
    "run_health": run_health,
    "statistics": statistics,
    "spectrogram": spectrogram,
    "beam": beam,
    "result_figure": figure,
    "scan_overview": scan_overview,
    "scan_point": scan_point,
}
"""The results requests, by kind."""
