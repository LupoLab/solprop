# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prototype the ordered time-outer eight-node plasma traversal (rejected).

An experiment kept with its measurements. The production
column does not dispatch here. Packing and unpacking belong to the
experimental caller and must be included when this traversal is priced. A parallel
closure is never disk-cached.
"""

from typing import Any, no_type_check

import numpy as np
from jaxtyping import Float
from numba import prange
from plasma_scan import PlasmaScalars

from solprop._numba_cache import njit
from solprop.backends.bodies import RateArrays
from solprop.backends.numba_cpu.kernels import HOST, ColumnSpec
from solprop.exceptions import BackendError

__all__ = ["make_tiled_plasma"]


@njit(cache=True, nogil=True, inline="always")
@no_type_check
def _stash_block(
    history: Float[np.ndarray, "eight n_blocks eight"],
    stashed: Float[np.ndarray, "n_stash_nodes two"],
    block: int,
    active: int,
) -> None:
    """Write the two statistics from completed active lane histories.

    Parameters
    ----------
    history : numpy.ndarray
        Completed per-lane recurrence values.
    stashed : numpy.ndarray
        Caller-owned final fractions and largest rates, one row per node.
    block : int
        Zero-based eight-node block index.
    active : int
        Number of active lanes in the block, from one through eight.
    """
    for lane in range(active):
        node = block * 8 + lane
        stashed[node, 0] = history[3, block, lane]
        stashed[node, 1] = history[2, block, lane]


def make_tiled_plasma(spec: ColumnSpec) -> Any:  # noqa: PLR0915 - ordered fused recurrence
    """Compile an ordered plasma column over contiguous lanes of eight nodes.

    Parameters
    ----------
    spec : ColumnSpec
        Scalar real-field plasma flags. Raman uses the real squared-field drive;
        other families and component counts receive an explicit refusal.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled traversal. Inputs and output use time-major packed storage;
        every mutable array is passed directly to the compiled function.

    Raises
    ------
    BackendError
        If the specialisation is outside this prototype's scalar real domain.
    """
    if spec.family != "field_real" or spec.n_pol != 1 or not spec.plasma:
        msg = "tiled plasma prototype requires scalar field_real plasma"
        raise BackendError(msg)
    if spec.raman and not spec.raman_thg:
        msg = "tiled plasma prototype requires real-field Raman drive"
        raise BackendError(msg)
    loop = prange if spec.parallel else range
    rate_at, trapezoid = HOST.rate_at, HOST.trapezoid
    fraction_at, loss_scale = HOST.fraction_at, HOST.loss_scale
    kerr_real = HOST.kerr_real
    raman, stash = spec.raman, spec.stash

    @njit(cache=not spec.parallel, nogil=True, parallel=spec.parallel)
    @no_type_check
    def column(  # noqa: PLR0913, PLR0917 - explicit kernel buffers
        packed: Float[np.ndarray, "n_to n_capacity"],
        output: Float[np.ndarray, "n_to n_capacity"],
        drive: Float[np.ndarray, "n_drive_nodes n_doubled"],
        window: Float[np.ndarray, " n_to"],
        table: RateArrays,
        scalars: PlasmaScalars,
        stashed: Float[np.ndarray, "n_stash_nodes two"],
        history: Float[np.ndarray, "eight n_blocks eight"],
        count: int,
    ) -> None:
        """Evaluate each node's unchanged recurrence inside a time-outer block.

        Parameters
        ----------
        packed : numpy.ndarray
            Real fields in V/m, shaped (n_to, n_capacity). Only columns before
            count are active; inactive columns are never read.
        output : numpy.ndarray
            Caller-owned time-major polarisation, matching packed storage.
            Inactive columns are untouched.
        drive : numpy.ndarray
            Existing node-major doubled Raman drive. Its leading time samples
            are written only when Raman is enabled, with the original formula.
        window : numpy.ndarray
            Dimensionless temporal collar, shaped (n_to,).
        table : RateArrays
            Unchanged compiled ionisation rate.
        scalars : PlasmaScalars
            Existing scalar stage parameters, in SI units.
        stashed : numpy.ndarray
            Final fractions and largest rates, untouched without stash.
        history : numpy.ndarray
            Eight recurrence variables for every lane of each block, shaped
            (8, ceil(n_capacity/8), 8). Active lanes are initialized by the kernel.
        count : int
            Active physical node count, positive and no greater than n_capacity.
            Shape agreement and capacity are internal caller responsibilities.
        """
        for block in loop((count + 7) // 8):
            first_node = block * 8
            active = min(8, count - first_node)
            for state in range(8):
                for lane in range(active):
                    history[state, block, lane] = 0.0
            for k in range(packed.shape[0]):
                for lane in range(active):
                    node = first_node + lane
                    field = packed[k, node]
                    magnitude = abs(field)
                    rate = rate_at(magnitude, table)
                    exposure = history[0, block, lane]
                    if k > 0:
                        exposure += trapezoid(rate, history[1, block, lane], scalars.dt)
                    fraction = fraction_at(exposure, scalars.preionisation)
                    guard = loss_scale(
                        rate,
                        fraction,
                        magnitude * magnitude,
                        scalars.ionisation_potential,
                    )
                    weight = fraction * scalars.charge_mass_ratio
                    integrand = weight * field
                    free = history[4, block, lane]
                    if k > 0:
                        free += trapezoid(
                            integrand, history[5, block, lane], scalars.dt
                        )
                    current = free + guard * field
                    outer = history[6, block, lane]
                    if k > 0:
                        outer += trapezoid(current, history[7, block, lane], scalars.dt)
                    total = scalars.density * outer + kerr_real(
                        field, field * field, scalars.kerr_factor
                    )
                    if raman:
                        drive[node, k] = field * field
                        output[k, node] = total
                    else:
                        output[k, node] = total * window[k]
                    history[0, block, lane] = exposure
                    history[1, block, lane] = rate
                    if stash and rate > history[2, block, lane]:
                        history[2, block, lane] = rate
                    history[3, block, lane] = fraction
                    history[4, block, lane] = free
                    history[5, block, lane] = integrand
                    history[6, block, lane] = outer
                    history[7, block, lane] = current
            if stash:
                _stash_block(history, stashed, block, active)

    return column
