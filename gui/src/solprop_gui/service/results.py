# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's results requests: one plane, the statistics, compression, figures.

Each request opens a results file and asks :mod:`solprop.processing` (or
:mod:`solprop.plotting`, for what has no data function) for one view's numbers (design
11.3): a plane's spectrum, power and spectral phase; every recorded statistic and the
step history against z; the spectrogram, drawn by solprop; the best Taylor-phase
compression of a plane; and a figure of any of these written by solprop's own plotting
(design 11.4). Like the other handlers, this module computes nothing of its own, and
runs only in the kernel process.
"""

import io
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from solprop import plotting
from solprop.exceptions import SolpropError
from solprop.output import Output, open_file
from solprop.processing import (
    ModeSelection,
    channel_labels,
    compress_taylor,
    get_phase,
    get_power,
    get_spectral_density,
)
from solprop.stats import CATALOGUE

__all__ = ["RESULTS_HANDLERS"]

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


def plane(payload: Payload) -> dict[str, object]:
    """Return one plane's spectrum, power and spectral phase.

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index`` (the plane), optionally ``modes`` (``"sum"`` or a
        mode's index) and ``axis`` (``"wavelength"``, ``"f"`` or ``"omega"``).

    Returns
    -------
    dict
        ``z``, ``labels`` (the file's modes), ``spectrum`` (``x`` and ``y``, the
        spectral density on the axis), ``power`` (``t`` and ``y``) and ``phase``
        (``omega`` and ``y``, of one mode; with ``error`` instead for a sum, which
        has no phase).
    """
    index = int(payload["index"])  # pyright: ignore[reportArgumentType]
    modes = _modes(payload)
    axis = str(payload.get("axis", "wavelength"))
    with open_file(str(payload["path"])) as handler:
        x, density = get_spectral_density(handler, axis, plane=index, modes=modes)  # pyright: ignore[reportArgumentType]
        t, power = get_power(handler, plane=index, modes=modes)
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


def _png(figure: Figure) -> bytes:
    """Return a Matplotlib figure as PNG bytes.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.

    Returns
    -------
    bytes
        The image.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110)
    return buffer.getvalue()


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


def spectrogram(payload: Payload) -> dict[str, object]:
    """Return a plane's spectrogram as solprop draws it.

    Parameters
    ----------
    payload : Mapping
        ``path``, ``z`` and optionally ``modes`` and ``log``.

    Returns
    -------
    dict
        ``png``, the image; of the first mode where a sum is asked for, which has
        no field.
    """
    with open_file(str(payload["path"])) as handler:
        (drawn,) = _drawn(handler, {**payload, "plot": "spectrogram"})
    return {"png": _png(drawn)}


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


def compression(payload: Payload) -> dict[str, object]:
    """Return a plane's best Taylor-phase compression (``compress_taylor``).

    Parameters
    ----------
    payload : Mapping
        ``path``, ``index`` (the plane), optionally ``order`` (2 by default) and
        ``modes``.

    Returns
    -------
    dict
        The widths and peak powers before, after and at the transform limit, the
        ``coefficients`` (the phase, from order two, in s^n), its ``quality`` and
        ``message``; and ``t`` with the power ``before``, ``after`` and ``limit``.
    """
    index = int(payload["index"])  # pyright: ignore[reportArgumentType]
    modes = payload.get("modes")
    selection = None if modes in (None, "sum") else int(modes)  # pyright: ignore[reportArgumentType]
    with open_file(str(payload["path"])) as handler:
        found = compress_taylor(
            handler,
            order=int(payload.get("order", 2)),  # pyright: ignore[reportArgumentType]
            plane=index,
            modes=selection,
        )
        summed = "sum" if selection is None else selection
        t, before = get_power(handler, plane=index, modes=summed)
        _, after = get_power(
            handler, plane=index, modes=summed, transfer=found.transfer
        )
        _, limit = get_power(handler, plane=index, modes=summed, transform_limit=True)
    return {
        "fwhm_before": float(found.fwhm_before),
        "fwhm_after": float(found.fwhm_after),
        "fwhm_limit": float(found.fwhm_limit),
        "peak_power_before": float(found.peak_power_before),
        "peak_power_after": float(found.peak_power_after),
        "peak_power_limit": float(found.peak_power_limit),
        "coefficients": [float(c) for c in np.atleast_1d(found.coefficients)],
        "quality": float(found.quality),
        "message": str(found.message),
        "t": np.asarray(t),
        "before": np.asarray(before),
        "after": np.asarray(after),
        "limit": np.asarray(limit),
    }


RESULTS_HANDLERS = {
    "plane": plane,
    "statistics": statistics,
    "spectrogram": spectrogram,
    "compression": compression,
    "result_figure": figure,
}
"""The results requests, by kind."""
