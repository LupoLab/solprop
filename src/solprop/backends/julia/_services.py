# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Copy Julia-owned observation data across a generation-checked delivery boundary."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from solprop.backends.julia._bridge import invoke
from solprop.backends.julia._observations import ObservationTransfer
from solprop.backends.julia._ownership import JuliaHandle
from solprop.backends.julia._runtime import JuliaRuntime
from solprop.backends.protocol import StepStats
from solprop.exceptions import BackendError

__all__ = ["ObservationPayload", "observations"]


@dataclass(frozen=True)
class ObservationPayload:
    """Retain a complete detached observation chunk from one numerical generation.

    Attributes
    ----------
    generation : int
        Pending engine generation copied without draining its numerical buffers.
    rows : solprop.backends.protocol.StepStats
        Periodic values with their exact relative attempt indices.
    plane : mapping
        Read-only plane arrays in public catalogue shapes, empty without a plane.
    """

    generation: int
    rows: StepStats
    plane: Mapping[str, np.ndarray]


def _observations(
    runtime: JuliaRuntime,
    handle: JuliaHandle,
    generation: int,
    transfer: ObservationTransfer,
) -> ObservationPayload:
    """Detach one packed payload inside the guarded main-thread Julia entry.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized process runtime.
    handle : JuliaHandle
        Session holding the pending generation.
    generation : int
        Bound engine generation.
    transfer : ObservationTransfer
        Immutable public shapes and packed offsets.

    Returns
    -------
    ObservationPayload
        Independently owned arrays; no Julia wrapper leaves this helper.

    Raises
    ------
    BackendError
        If the numerical payload disagrees with its prepared catalogue.
    """
    engine = runtime.engine
    session = handle.borrow()
    result = engine.session_observation_status(session, generation)
    count, width, plane_width = (int(value) for value in result)
    selected = tuple(transfer.slots[index - 1] for index in transfer.periodic_indices)
    if (
        count < 0
        or width != sum(slot.count for slot in selected)
        or plane_width not in (0, transfer.width)
    ):
        raise BackendError("Julia observation buffers disagree with their preparation")
    positions = np.empty(count, dtype=np.float64)
    steps = np.empty(count, dtype=np.int64)
    packed = np.empty((count, width), dtype=np.float64)
    plane = np.empty(plane_width, dtype=np.float64)
    getattr(engine, "copy_session_observations!")(
        positions, steps, packed, plane, session, generation
    )
    periodic: dict[str, np.ndarray] = {}
    offset = 0
    for slot in selected:
        periodic[slot.name] = packed[:, offset : offset + slot.count].reshape(
            (count, *slot.shape)
        )
        offset += slot.count
    plane.setflags(write=False)
    values = (
        {
            slot.name: plane[slot.offset - 1 : slot.offset - 1 + slot.count].reshape(
                slot.shape
            )
            for slot in transfer.slots
        }
        if plane_width
        else {}
    )
    return ObservationPayload(generation, StepStats(positions, steps, periodic), values)


def observations(
    runtime: JuliaRuntime,
    handle: JuliaHandle,
    generation: int,
    transfer: ObservationTransfer | None,
) -> ObservationPayload:
    """Copy a complete pending observation chunk without consuming it.

    Parameters
    ----------
    runtime : JuliaRuntime
        Initialized main-thread runtime.
    handle : JuliaHandle
        Owned session holding its pending generation.
    generation : int
        Receipt-bound engine generation.
    transfer : ObservationTransfer or None
        Immutable catalogue, absent for a session without observation services.

    Returns
    -------
    ObservationPayload
        Detached periodic and plane values, safe to retain after acknowledgment.
    """
    if transfer is None:
        return ObservationPayload(generation, StepStats.empty(), {})
    runtime.check()
    return invoke(
        lambda: _observations(runtime, handle, generation, transfer),
        "the julia observation copy failed",
    )
