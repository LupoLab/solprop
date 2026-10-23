# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prototype the complete fixed-partition plasma column (rejected).

An experiment kept with its measurements, beside the
production column it lost to. It is confined to scalar real-field columns and does not
change the production dispatcher. All physics uses the existing shared scalar bodies;
only the association of the three temporal prefixes changes.
A parallel closure is never disk-cached.
"""

from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Float
from numba import prange
from plasma_scan_kernels import make_scan_kernels

from solprop._numba_cache import njit
from solprop.backends.bodies import RateArrays
from solprop.backends.numba_cpu.kernels import HOST, ColumnSpec
from solprop.exceptions import BackendError

__all__ = ["BlockedPlasma", "PlasmaScalars", "PlasmaWork", "make_blocked_plasma"]


class BlockedPlasma(NamedTuple):
    """Expose the full experiment and its unchanged rate pass for input budgets.

    Attributes
    ----------
    column, rates : callable
        Complete response and shared-body rate pass. The latter lets the caller
        derive input-conditioned budgets before evaluating the complete response.
    """

    # Numba dispatchers do not supply callable stubs.
    column: Any
    rates: Any


class PlasmaScalars(NamedTuple):
    """Group the existing column's host scalar parameters without changing units.

    Attributes
    ----------
    kerr_factor : float
        Kerr coefficient times density, in the existing column's units.
    charge_mass_ratio : float
        Electron charge-to-mass factor from the compiled plasma response.
    ionisation_potential : float
        Ionisation energy in J.
    density : float
        Number density in m^-3.
    preionisation : float
        Initial ionised fraction in [0, 1].
    dt : float
        Positive fine-grid sample interval in s.
    """

    kerr_factor: float
    charge_mass_ratio: float
    ionisation_potential: float
    density: float
    preionisation: float
    dt: float


class PlasmaWork(NamedTuple):
    """Name every caller-owned array of the blocked scalar plasma prototype.

    Attributes
    ----------
    rates, increments, prefix, fraction, loss : numpy.ndarray
        Five float64 arrays shaped (n_nodes, n_to). Rates and increments are
        reused after their last read; the three scans share prefix storage.
    totals, maxima : numpy.ndarray
        Float64 arrays shaped (n_nodes, n_segments), for scan carries and rate
        maxima. The latter preserves statistics before rate storage is reused.
    """

    rates: Float[np.ndarray, "n_nodes n_to"]
    increments: Float[np.ndarray, "n_nodes n_to"]
    prefix: Float[np.ndarray, "n_nodes n_to"]
    fraction: Float[np.ndarray, "n_nodes n_to"]
    loss: Float[np.ndarray, "n_nodes n_to"]
    totals: Float[np.ndarray, "n_nodes n_segments"]
    maxima: Float[np.ndarray, "n_nodes n_segments"]


def _make_finish(spec: ColumnSpec, n_to: int, segments: int) -> Any:
    """Specialise the final Kerr, drive, collar and stash pass.

    Parameters
    ----------
    spec : ColumnSpec
        Response flags and parallel-kernel selection.
    n_to : int
        Power-of-two temporal extent.
    segments : int
        Fixed temporal partition count, independent of workers.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        Compiled pass with all work arrays supplied by the caller.
    """
    width = n_to // segments
    loop = prange if spec.parallel else range
    kerr_real = HOST.kerr_real
    raman, stash = spec.raman, spec.stash

    @njit(cache=not spec.parallel, nogil=True, parallel=spec.parallel)
    @no_type_check
    def finish(  # noqa: PLR0913, PLR0917 - caller-owned kernel buffers
        nodes: Float[np.ndarray, "n_nodes n_to"],
        output: Float[np.ndarray, "n_nodes n_to"],
        drive: Float[np.ndarray, "n_nodes n_doubled"],
        window: Float[np.ndarray, " n_to"],
        scalars: PlasmaScalars,
        stashed: Float[np.ndarray, "n_nodes two"],
        prefix: Float[np.ndarray, "n_nodes n_to"],
        fraction: Float[np.ndarray, "n_nodes n_to"],
        maxima: Float[np.ndarray, "n_nodes n_segments"],
    ) -> None:
        """Finish with the existing Kerr addition, drive, collar and statistics.

        Parameters
        ----------
        nodes : numpy.ndarray
            Real node fields, shaped (n_nodes, n_to), in V/m.
        output : numpy.ndarray
            Caller-owned polarisation output, shaped (n_nodes, n_to).
        drive : numpy.ndarray
            Doubled Raman drive; only its leading n_to samples are written.
        window : numpy.ndarray
            Dimensionless temporal collar, shaped (n_to,).
        scalars : PlasmaScalars
            Existing host scalar values for this stage, in SI units.
        stashed : numpy.ndarray
            Per-node final fraction and largest rate; untouched without stash.
        prefix, fraction : numpy.ndarray
            Final plasma integral and ionised fractions, shaped (n_nodes, n_to).
        maxima : numpy.ndarray
            Per-segment rate maxima, shaped (n_nodes, n_segments).
        """
        for task in loop(nodes.shape[0] * segments):
            node, segment = task // segments, task % segments
            for k in range(segment * width, (segment + 1) * width):
                field = nodes[node, k]
                total = scalars.density * prefix[node, k] + kerr_real(
                    field, field * field, scalars.kerr_factor
                )
                if raman:
                    drive[node, k] = field * field
                    output[node, k] = total
                else:
                    output[node, k] = total * window[k]
            if stash and segment == 0:
                largest = 0.0
                for part in range(segments):
                    largest = max(largest, maxima[node, part])
                stashed[node, 0] = fraction[node, n_to - 1]
                stashed[node, 1] = largest

    return finish


def _make_increments(spec: ColumnSpec, n_to: int, segments: int) -> Any:
    """Specialise the unchanged trapezoid increments for a fixed partition.

    Parameters
    ----------
    spec : ColumnSpec
        Response flags and parallel-kernel selection.
    n_to : int
        Power-of-two temporal extent.
    segments : int
        Fixed temporal partition count, independent of workers.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        Compiled pass with all work arrays supplied by the caller.
    """
    width = n_to // segments
    loop = prange if spec.parallel else range
    trapezoid = HOST.trapezoid

    @njit(cache=not spec.parallel, nogil=True, parallel=spec.parallel)
    @no_type_check
    def increments(
        values: Float[np.ndarray, "n_nodes n_to"],
        output: Float[np.ndarray, "n_nodes n_to"],
        dt: float,
    ) -> None:
        """Form the existing trapezoid increments with an exact zero first sample.

        Parameters
        ----------
        values : numpy.ndarray
            Input samples, shaped (n_nodes, n_to).
        output : numpy.ndarray
            Caller-owned polarisation output, shaped (n_nodes, n_to).
        dt : float
            Positive sample spacing in s.
        """
        for task in loop(values.shape[0] * segments):
            node, segment = task // segments, task % segments
            for k in range(segment * width, (segment + 1) * width):
                output[node, k] = (
                    trapezoid(values[node, k], values[node, k - 1], dt)
                    if k > 0
                    else 0.0
                )

    return increments


def _make_rates(spec: ColumnSpec, n_to: int, segments: int) -> Any:
    """Specialise the unchanged rate pass and fixed-segment maxima.

    Parameters
    ----------
    spec : ColumnSpec
        Response flags and parallel-kernel selection.
    n_to : int
        Power-of-two temporal extent.
    segments : int
        Fixed temporal partition count, independent of workers.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        Compiled rate pass using caller-owned arrays.
    """
    width = n_to // segments
    loop = prange if spec.parallel else range
    rate_at = HOST.rate_at
    stash = spec.stash

    @njit(cache=not spec.parallel, nogil=True, parallel=spec.parallel)
    @no_type_check
    def rates(
        nodes: Float[np.ndarray, "n_nodes n_to"],
        table: RateArrays,
        values: Float[np.ndarray, "n_nodes n_to"],
        maxima: Float[np.ndarray, "n_nodes n_segments"],
    ) -> None:
        """Write unchanged rates and fixed-segment maxima to caller-owned storage.

        Parameters
        ----------
        nodes : numpy.ndarray
            Real node fields, shaped (n_nodes, n_to), in V/m.
        table : RateArrays
            Unchanged compiled scalar ionisation-rate coefficients.
        values : numpy.ndarray
            Caller-owned rate output, shaped (n_nodes, n_to).
        maxima : numpy.ndarray
            Caller-owned segment maxima, shaped (n_nodes, n_segments).
        """
        for task in loop(nodes.shape[0] * segments):
            node, segment = task // segments, task % segments
            largest = 0.0
            for k in range(segment * width, (segment + 1) * width):
                rate = rate_at(abs(nodes[node, k]), table)
                values[node, k] = rate
                if stash and rate > largest:
                    largest = rate
            maxima[node, segment] = largest

    return rates


def make_blocked_plasma(spec: ColumnSpec, n_to: int) -> BlockedPlasma:
    """Compile the complete experimental column with fixed temporal partitions.

    Parameters
    ----------
    spec : ColumnSpec
        Scalar real-field plasma specialisation. Raman and stash flags remain
        closure constants; unsupported families receive an explicit refusal.
    n_to : int
        Positive power-of-two temporal extent. The caller supplies a PlasmaWork
        with min(16, n_to) segments and buffers matching the actual chunk width.

    Returns
    -------
    BlockedPlasma
        Allocation-free host coordinator and rate pass. The column's arguments
        are nodes, output, Raman drive, window, rate table, scalar bundle, stash
        and workspace, in that order. It performs no construction or planning.

    Raises
    ------
    BackendError
        If the specialisation is outside this prototype's domain or the time
        extent is not a positive power of two.
    """
    if spec.family != "field_real" or spec.n_pol != 1 or not spec.plasma:
        msg = "blocked plasma prototype requires scalar field_real plasma"
        raise BackendError(msg)
    if spec.raman and not spec.raman_thg:
        msg = "blocked plasma prototype requires real-field Raman drive"
        raise BackendError(msg)
    scans = make_scan_kernels(n_to, parallel=spec.parallel)
    segments = scans.segments
    width = n_to // segments
    loop = prange if spec.parallel else range
    fraction_at, loss_scale = HOST.fraction_at, HOST.loss_scale

    rate_pass = _make_rates(spec, n_to, segments)

    def rates(
        nodes: Float[np.ndarray, "n_nodes n_to"],
        table: RateArrays,
        work: PlasmaWork,
    ) -> None:
        """Unpack host workspace arrays for the compiled rate pass.

        Parameters
        ----------
        nodes : numpy.ndarray
            Real fields, shaped (n_nodes, n_to), in V/m.
        table : RateArrays
            Compiled ionisation rate.
        work : PlasmaWork
            Host bundle whose output arrays are passed directly to the kernel.
        """
        rate_pass(nodes, table, work.rates, work.maxima)

    increments = _make_increments(spec, n_to, segments)

    @njit(cache=not spec.parallel, nogil=True, parallel=spec.parallel)
    @no_type_check
    def fraction(  # noqa: PLR0913, PLR0917 - explicit kernel array outputs
        nodes: Float[np.ndarray, "n_nodes n_to"],
        scalars: PlasmaScalars,
        rates: Float[np.ndarray, "n_nodes n_to"],
        prefix: Float[np.ndarray, "n_nodes n_to"],
        ionised_values: Float[np.ndarray, "n_nodes n_to"],
        integrands: Float[np.ndarray, "n_nodes n_to"],
        loss: Float[np.ndarray, "n_nodes n_to"],
    ) -> None:
        """Form the unchanged fraction, free-current integrand and loss current.

        Parameters
        ----------
        nodes : numpy.ndarray
            Real node fields, shaped (n_nodes, n_to), in V/m.
        scalars : PlasmaScalars
            Existing host scalar values for this stage, in SI units.
        rates, prefix : numpy.ndarray
            Shared-body rates and exposure prefixes, shaped (n_nodes, n_to).
        ionised_values, integrands, loss : numpy.ndarray
            Caller-owned fraction, free integrand and loss-current outputs.
        """
        for task in loop(nodes.shape[0] * segments):
            node, segment = task // segments, task % segments
            for k in range(segment * width, (segment + 1) * width):
                field = nodes[node, k]
                magnitude = abs(field)
                ionised = fraction_at(prefix[node, k], scalars.preionisation)
                guard = loss_scale(
                    rates[node, k],
                    ionised,
                    magnitude * magnitude,
                    scalars.ionisation_potential,
                )
                ionised_values[node, k] = ionised
                integrands[node, k] = (ionised * scalars.charge_mass_ratio) * field
                loss[node, k] = guard * field

    @njit(cache=not spec.parallel, nogil=True, parallel=spec.parallel)
    @no_type_check
    def current(
        prefix: Float[np.ndarray, "n_nodes n_to"],
        loss: Float[np.ndarray, "n_nodes n_to"],
    ) -> None:
        """Add the loss current to the free-current prefix in its original order.

        Parameters
        ----------
        prefix, loss : numpy.ndarray
            Free-current prefix and loss term; prefix receives the ordered sum.
        """
        for task in loop(prefix.shape[0] * segments):
            node, segment = task // segments, task % segments
            for k in range(segment * width, (segment + 1) * width):
                prefix[node, k] += loss[node, k]

    finish = _make_finish(spec, n_to, segments)

    def scan(values: Float[np.ndarray, "n_nodes n_to"], work: PlasmaWork) -> None:
        """Execute the three fixed-prefix passes on the same owned workspace.

        Parameters
        ----------
        values : numpy.ndarray
            Input samples, shaped (n_nodes, n_to).
        work : PlasmaWork
            Caller-owned full-chunk scratch and fixed-segment reductions.
        """
        scans.local(values, work.prefix, work.totals)
        scans.carries(work.totals)
        scans.apply(work.prefix, work.totals)

    def column(  # noqa: PLR0913, PLR0917 - caller-owned kernel buffers
        nodes: Float[np.ndarray, "n_nodes n_to"],
        output: Float[np.ndarray, "n_nodes n_to"],
        drive: Float[np.ndarray, "n_nodes n_doubled"],
        window: Float[np.ndarray, " n_to"],
        table: RateArrays,
        scalars: PlasmaScalars,
        stashed: Float[np.ndarray, "n_nodes two"],
        work: PlasmaWork,
    ) -> None:
        """Evaluate all three plasma scans with their complete pass overhead.

        Parameters
        ----------
        nodes : numpy.ndarray
            Real node fields, shaped (n_nodes, n_to), in V/m.
        output : numpy.ndarray
            Caller-owned polarisation output, shaped (n_nodes, n_to).
        drive : numpy.ndarray
            Doubled Raman drive; only its leading n_to samples are written.
        window : numpy.ndarray
            Dimensionless temporal collar, shaped (n_to,).
        table : RateArrays
            Unchanged compiled scalar ionisation-rate coefficients.
        scalars : PlasmaScalars
            Existing host scalar values for this stage, in SI units.
        stashed : numpy.ndarray
            Per-node final fraction and largest rate; untouched without stash.
        work : PlasmaWork
            Caller-owned full-chunk scratch and fixed-segment reductions.
        """
        rates(nodes, table, work)
        increments(work.rates, work.increments, scalars.dt)
        scan(work.increments, work)
        fraction(
            nodes,
            scalars,
            work.rates,
            work.prefix,
            work.fraction,
            work.increments,
            work.loss,
        )
        increments(work.increments, work.rates, scalars.dt)
        scan(work.rates, work)
        current(work.prefix, work.loss)
        increments(work.prefix, work.increments, scalars.dt)
        scan(work.increments, work)
        finish(
            nodes,
            output,
            drive,
            window,
            scalars,
            stashed,
            work.prefix,
            work.fraction,
            work.maxima,
        )

    return BlockedPlasma(column, rates)
