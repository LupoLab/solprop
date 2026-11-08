# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The kernel's figure request: a view's plot drawn by Matplotlib (design 11.4).

A view describes what it shows as plain data (its curves, marker lines and points,
its axis titles and log axes) and the kernel draws that with Matplotlib's object
interface, which needs no window, into a PNG, SVG or PDF file. So an exported figure
is the view as seen, at print quality, without the view's own toolkit in the way.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

__all__ = ["FIGURE_HANDLERS", "figure"]

FORMATS = (".png", ".svg", ".pdf")
"""The file types a figure is written as."""

Payload = Mapping[str, object]


def figure(payload: Payload) -> str:
    """Draw a view's description and write it.

    Parameters
    ----------
    payload : Mapping
        ``path``, the file (``.png``, ``.svg`` or ``.pdf``; another suffix is
        written as PNG), and ``spec``, the view's description
        (the window's ``solprop_gui.views.plotting.figure_spec``): ``xlabel``,
        ``ylabel``, ``logx``, ``logy``, ``curves``, ``lines`` and ``points``.

    Returns
    -------
    str
        The file written.
    """
    spec: Mapping[str, object] = payload["spec"]  # pyright: ignore[reportAssignmentType]
    path = Path(str(payload["path"]))
    if path.suffix.lower() not in FORMATS:
        path = path.with_suffix(".png")
    canvas = Figure(figsize=(6.4, 4.2), layout="constrained")
    axes = canvas.add_subplot()
    curves: Sequence[Mapping[str, object]] = spec.get("curves") or []  # pyright: ignore[reportAssignmentType]
    for curve in curves:
        axes.plot(
            np.asarray(curve["x"], dtype=float),
            np.asarray(curve["y"], dtype=float),
            color=str(curve["colour"]),
            linestyle=str(curve["style"]),
            linewidth=float(curve["width"]),  # pyright: ignore[reportArgumentType]
            label=str(curve["label"]) or None,
        )
    lines: Sequence[Mapping[str, object]] = spec.get("lines") or []  # pyright: ignore[reportAssignmentType]
    for line in lines:
        style = {
            "color": str(line["colour"]),
            "linestyle": str(line["style"]),
            "linewidth": float(line["width"]),  # pyright: ignore[reportArgumentType]
        }
        if "x" in line:
            axes.axvline(float(line["x"]), **style)  # pyright: ignore[reportArgumentType]
            if line["label"]:
                axes.annotate(
                    str(line["label"]),
                    (float(line["x"]), 1.0),  # pyright: ignore[reportArgumentType]
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -12),
                    textcoords="offset points",
                    color=str(line["colour"]),
                    fontsize="small",
                )
        else:
            axes.axhline(float(line["y"]), **style)  # pyright: ignore[reportArgumentType]
    points: Sequence[Mapping[str, object]] = spec.get("points") or []  # pyright: ignore[reportAssignmentType]
    for point in points:
        axes.plot(
            [float(point["x"])],  # pyright: ignore[reportArgumentType]
            [float(point["y"])],  # pyright: ignore[reportArgumentType]
            marker="+",
            markersize=12,
            markeredgewidth=2,
            linestyle="none",
            color=str(point["colour"]),
            label=str(point["label"]) or None,
        )
    if spec.get("logx"):
        axes.set_xscale("log")
    if spec.get("logy"):
        axes.set_yscale("log")
    axes.set_xlabel(str(spec.get("xlabel") or ""))
    axes.set_ylabel(str(spec.get("ylabel") or ""))
    axes.grid(alpha=0.2)
    if any(curve["label"] for curve in curves) or points:
        axes.legend(fontsize="small", frameon=False)
    canvas.savefig(path, dpi=200)
    return str(path)


FIGURE_HANDLERS = {"figure": figure}
"""The figure request, by kind."""
