# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Bind a transverse edge in a native workspace and read it, as a native run does.

The edge tests (``test_native_edge_reading.py``, ``test_native_edge_collar.py``) hold
native's own reading of a loaded state to the host's. A run's start binds the monitor
and its edge (``Workspace::start``); these helpers bind them alone, through the private
test module (``solprop_native._core._testing``), and lay the reading out as the edge's
judging reads it (:class:`solprop.transverse_monitor._EdgeReading`):

- :func:`bind_edge_monitor` binds a boundary monitor's weights, the problem's walk-off
  buckets and the monitor's edge (``_testing.bind_monitor``);
- :func:`edge_reading` reads the state for the edge (``_testing.edge_reading``).
"""

from typing import Any

import numpy as np

from solprop.boundaries import SPLIT_RATE_BUCKETS
from solprop.monitor import BoundaryMonitor
from solprop.transverse_monitor import TransverseEdge, _EdgeReading

__all__ = ["bind_edge_monitor", "edge_reading"]


def _testing() -> Any:
    """Return the native extension's private test module."""
    from solprop_native._core import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
        _testing,
    )

    return _testing


def bind_edge_monitor(backend: Any, monitor: BoundaryMonitor) -> None:
    """Hold a monitor's weights, buckets and edge tables in a native workspace.

    What a native run's start binds for its monitor: the backend's
    ``_monitor_binding``, the walk-off occupancy of a monitor with a crossing and the
    tables of its transverse edge (``_edge_binding``), if it has one.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend, its workspace loaded (``sessions.runs.load_native``,
        which unbinds the monitor: bind it after).
    monitor : solprop.monitor.BoundaryMonitor
        The monitor.
    """
    crossing = monitor.crossing
    occupancy = (
        None
        if crossing is None
        else {
            "buckets": np.ascontiguousarray(crossing.buckets, dtype=np.uint8).reshape(
                -1
            ),
            "count": SPLIT_RATE_BUCKETS,
        }
    )
    edge = monitor.transverse
    _testing().bind_monitor(
        backend._workspace,
        backend._monitor_binding(monitor),
        None if edge is None else backend._edge_binding(edge),
        occupancy,
    )


def edge_reading(
    backend: Any,
    edge: TransverseEdge,
    *,
    faces: bool = True,
    current: bool = False,
    full: bool = False,
) -> Any:  # an `_EdgeReading` whose fields the flags fill, which its type can't say
    """Read a native workspace's state for the transverse edge, where it lives.

    Only numbers cross. A launch's reading (`full`) adds the clip and the totals; a
    requested collar's read adds the totals and its return strip.

    Parameters
    ----------
    backend : solprop_native.NativeBackend
        The constructed backend, its edge bound (:func:`bind_edge_monitor`).
    edge : solprop.transverse_monitor.TransverseEdge
        The bound edge.
    faces : bool, optional
        Read the faces, columns and (with a collar) the strip; ``False`` reads only the
        current.
    current : bool, optional
        Read the collar's rectified inward current.
    full : bool, optional
        Read as a launch does.

    Returns
    -------
    solprop.transverse_monitor._EdgeReading
        The reading.
    """
    collar = edge.every_step
    sums, columns, strip, rate = _testing().edge_reading(
        backend._workspace, faces, full, collar, faces and collar, current
    )
    rate_value = float(rate) if current else None
    if not faces:
        return _EdgeReading(current=rate_value)
    values = np.asarray(sums).reshape(-1)
    if full:
        clip, totals = values[4:8], values[8:12]
    else:
        clip, totals = None, values[4:8] if collar else None
    return _EdgeReading(
        faces=values[:4],
        clip=clip,
        totals=totals,
        columns=np.asarray(columns).reshape(-1),
        strip=np.asarray(strip).reshape(-1, 2) if collar else None,
        current=rate_value,
    )
