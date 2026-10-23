# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute separable spatial coefficients and bounded Cartesian column transfers."""

from collections.abc import Collection
from functools import cache
from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Complex
from numba import prange

from solprop import fourier
from solprop._numba_cache import njit
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.spatial import (
    FarFieldLayout,
    far_field_bands,
    far_field_layout,
    far_field_partials,
)
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.spatial import ABSORBED_CHANNELS, FAR_FIELD_STATISTICS, ApertureTables
from solprop.translate.problem import KernelProblem
from solprop.translate.spatial import SeparableOperator

__all__ = [
    "FarFieldPass",
    "SpatialKernels",
    "absorber_columns",
    "accumulate_stage",
    "crop_columns",
    "far_field_sum",
    "finish_material",
    "flux_measure",
    "make_far_field",
    "make_kernels",
    "pack_columns",
    "rotate_noise",
    "write_exponent",
    "write_phase",
    "write_propagator",
]

_ELEMENT = HOST.spatial_element
_PAIR = HOST.propagator_pair
_ROOT = HOST.longitudinal_root
_RADIUS = HOST.mask_plane_radius
_WEIGHT = HOST.aperture_weight
_MEMBER = HOST.region_member


class SpatialKernels(NamedTuple):
    """Group the independently specialised Cartesian stage operations.

    Attributes
    ----------
    write_propagator, write_exponent, finish_material, accumulate_stage : callable
        Wide stage traversals, serial or parallel across independent columns.
    pack_columns, crop_columns : callable
        Temporal-buffer transfers, with their own parallelism choice.
    """

    write_propagator: Any
    write_exponent: Any
    finish_material: Any
    accumulate_stage: Any
    pack_columns: Any
    crop_columns: Any


@cache
def make_kernels(*, parallel: bool, parallel_transfers: bool = False) -> SpatialKernels:
    """Compile spatial passes with independent stage and copy scheduling.

    Parameters
    ----------
    parallel : bool
        Whether wide stage operations run across independent columns.
    parallel_transfers : bool, optional
        Whether temporal-buffer copies run across independent columns. Efficient
        contiguous field copies remain serial; large strided envelope copies benefit.

    Returns
    -------
    SpatialKernels
        Compiled operations with unchanged within-column arithmetic.
    """
    # Capture the iterator itself, so the serial/parallel choice is also present
    # in Numba's closure cache key, rather than only in dispatcher options. Only
    # the serial kernels are disk-cached: a parallel closure is not.
    wide_columns = prange if parallel else range
    copy_columns = prange if parallel_transfers else range

    @njit(cache=not parallel, nogil=True, parallel=parallel)
    @no_type_check
    def write_propagator(
        data: Any, distance: float, forward: np.ndarray, inverse: np.ndarray
    ) -> None:
        """Evaluate both propagation signs without constructing a coefficient table.

        Parameters
        ----------
        data : tuple
            Shared schema-5 arrays, evanescent rate, polarisation count and the
            schema-31 transverse rate per column.
        distance : float
            Propagation increment in m.
        forward, inverse : numpy.ndarray
            Preallocated propagation-factor arrays, complex128.
        """
        (
            square,
            transverse,
            square_low,
            transverse_low,
            frame,
            rate,
            numerator,
            clamped,
            active,
            evanescent,
            n_pol,
            kperp,
        ) = data
        for column in wide_columns(forward.shape[0]):
            for omega in range(forward.shape[1]):
                value, _ = _ELEMENT(
                    square[omega],
                    transverse[column // n_pol],
                    square_low[omega],
                    transverse_low[column // n_pol],
                    frame[omega],
                    rate[omega],
                    numerator[omega],
                    clamped[omega],
                    active[omega],
                    evanescent,
                    kperp[column // n_pol],
                )
                exponent = distance * value
                fr, fi, br, bi = _PAIR(exponent.real, exponent.imag)
                forward[column, omega] = complex(fr, fi)
                inverse[column, omega] = complex(br, bi)

    @njit(cache=not parallel, nogil=True, parallel=parallel)
    @no_type_check
    def write_exponent(data: Any, distance: float, exponent: np.ndarray) -> None:
        """Evaluate the exponent ``D distance`` that :func:`write_propagator` takes.

        Parameters
        ----------
        data : tuple
            As for :func:`write_propagator`.
        distance : float
            Propagation increment in m.
        exponent : numpy.ndarray
            Preallocated exponent, complex128, ``(n_columns, n_omega)``; overwritten.
        """
        (
            square,
            transverse,
            square_low,
            transverse_low,
            frame,
            rate,
            numerator,
            clamped,
            active,
            evanescent,
            n_pol,
            kperp,
        ) = data
        for column in wide_columns(exponent.shape[0]):
            for omega in range(exponent.shape[1]):
                value, _ = _ELEMENT(
                    square[omega],
                    transverse[column // n_pol],
                    square_low[omega],
                    transverse_low[column // n_pol],
                    frame[omega],
                    rate[omega],
                    numerator[omega],
                    clamped[omega],
                    active[omega],
                    evanescent,
                    kperp[column // n_pol],
                )
                exponent[column, omega] = distance * value

    @njit(cache=not parallel_transfers, nogil=True, parallel=parallel_transfers)
    @no_type_check
    def pack_columns(
        source: np.ndarray,
        crop: np.ndarray,
        fine: np.ndarray,
        scale: float,
        analytic: bool,
    ) -> None:
        """Pad one physical-space chunk for an unnormalised temporal inverse.

        Parameters
        ----------
        source : numpy.ndarray
            Coarse physical spectra, complex128.
        crop : numpy.ndarray
            Coarse-to-fine frequency indices.
        fine : numpy.ndarray
            Planned temporal inverse input, overwritten including unused rows.
        scale : float
            Explicit temporal inverse scaling.
        analytic : bool
            Whether to double the bins strictly between the zero frequency and the
            fine grid's Nyquist bin, the analytic-signal rule of
            :func:`scipy.signal.hilbert` on the fine grid and of the modal path's
            ``_double_positive_bins``.
        """
        half = fine.shape[1] // 2
        for column in copy_columns(fine.shape[0]):
            for omega in range(fine.shape[1]):
                fine[column, omega] = 0j
            if column < source.shape[0]:
                for omega in range(source.shape[1]):
                    factor = 2.0 if analytic and 0 < crop[omega] < half else 1.0
                    fine[column, crop[omega]] = source[column, omega] * (scale * factor)

    @njit(cache=not parallel_transfers, nogil=True, parallel=parallel_transfers)
    @no_type_check
    def crop_columns(fine: np.ndarray, crop: np.ndarray, target: np.ndarray) -> None:
        """Return one transformed physical chunk to its coarse spectral scratch.

        Parameters
        ----------
        fine : numpy.ndarray
            Planned temporal forward output, complex128.
        crop : numpy.ndarray
            Coarse-to-fine frequency indices.
        target : numpy.ndarray
            Coarse scratch rows, overwritten in place.
        """
        for column in copy_columns(target.shape[0]):
            for omega in range(target.shape[1]):
                target[column, omega] = fine[column, crop[omega]]

    @njit(cache=not parallel, nogil=True, parallel=parallel)
    @no_type_check
    def finish_material(
        data: Any,
        spectrum: np.ndarray,
        window: np.ndarray,
        inverse: np.ndarray,
        stage: np.ndarray,
    ) -> None:
        """Apply the separable prefactor and interaction factor directly to a stage.

        Parameters
        ----------
        data : tuple
            Shared schema-5 arrays, evanescent rate, polarisation count and the
            schema-31 transverse rate per column.
        spectrum : numpy.ndarray
            Transverse and temporal transform of the nonlinear polarisation.
        window : numpy.ndarray
            Spectral taper, dimensionless.
        inverse : numpy.ndarray
            Inverse interaction factor.
        stage : numpy.ndarray
            Stage derivative overwritten in place.
        """
        (
            square,
            transverse,
            square_low,
            transverse_low,
            frame,
            rate,
            numerator,
            clamped,
            active,
            evanescent,
            n_pol,
            kperp,
        ) = data
        for column in wide_columns(stage.shape[0]):
            for omega in range(stage.shape[1]):
                _, prefactor = _ELEMENT(
                    square[omega],
                    transverse[column // n_pol],
                    square_low[omega],
                    transverse_low[column // n_pol],
                    frame[omega],
                    rate[omega],
                    numerator[omega],
                    clamped[omega],
                    active[omega],
                    evanescent,
                    kperp[column // n_pol],
                )
                stage[column, omega] = inverse[column, omega] * (
                    prefactor * window[omega] * spectrum[column, omega]
                )

    @njit(cache=not parallel, nogil=True, parallel=parallel)
    @no_type_check
    def accumulate_stage(
        stage: np.ndarray,
        weights: tuple[float, float],
        solution: np.ndarray,
        error: np.ndarray,
    ) -> None:
        """Accumulate a complete Cartesian stage into solution and embedded error.

        Parameters
        ----------
        stage : numpy.ndarray
            Complete interaction-picture derivative.
        weights : tuple of float
            Solution and embedded-error tableau weights.
        solution, error : numpy.ndarray
            Persistent accumulators, updated in the existing stage order.
        """
        for column in wide_columns(stage.shape[0]):
            for omega in range(stage.shape[1]):
                solution[column, omega] += weights[0] * stage[column, omega]
                error[column, omega] += weights[1] * stage[column, omega]

    return SpatialKernels(
        write_propagator,
        write_exponent,
        finish_material,
        accumulate_stage,
        pack_columns,
        crop_columns,
    )


_SERIAL = make_kernels(parallel=False)
write_propagator = _SERIAL.write_propagator
write_exponent = _SERIAL.write_exponent
finish_material = _SERIAL.finish_material
accumulate_stage = _SERIAL.accumulate_stage
pack_columns = _SERIAL.pack_columns
crop_columns = _SERIAL.crop_columns


@njit(cache=True, nogil=True)
@no_type_check
def absorber_columns(
    samples: np.ndarray,
    rates: tuple[np.ndarray, np.ndarray],
    first: int,
    n_pol: int,
    output: np.ndarray,
) -> None:
    """Apply spatial and temporal rates without a broadcast-sized allocation.

    Parameters
    ----------
    samples : numpy.ndarray
        Physical time samples, real for a field grid.
    rates : tuple of numpy.ndarray
        Spatial and temporal power rates in m^-1.
    first : int
        First physical node of this chunk.
    n_pol : int
        State polarisation count.
    output : numpy.ndarray
        Planned forward-FFT input, overwritten for the active chunk.
    """
    for column in range(samples.shape[0]):
        for time in range(samples.shape[1]):
            rate = rates[0][first + column // n_pol] + rates[1][time]
            output[column, time] = -0.5 * rate * samples[column, time]


@njit(cache=True, nogil=True)
@no_type_check
def write_phase(data: Any, z: float, phase: np.ndarray) -> None:
    """Evaluate the unitary noise phase without a coefficient volume.

    Parameters
    ----------
    data : tuple
        Shared spatial coefficients, evanescent rate, polarisation count and the
        schema-31 transverse rate per column.
    z : float
        Absolute propagation position in m.
    phase : numpy.ndarray
        Real phase buffer, overwritten in radians.
    """
    (
        square,
        transverse,
        square_low,
        transverse_low,
        frame,
        rate,
        numerator,
        clamped,
        active,
        evanescent,
        n_pol,
        kperp,
    ) = data
    for column in range(phase.shape[0]):
        for omega in range(phase.shape[1]):
            value, _ = _ELEMENT(
                square[omega],
                transverse[column // n_pol],
                square_low[omega],
                transverse_low[column // n_pol],
                frame[omega],
                rate[omega],
                numerator[omega],
                clamped[omega],
                active[omega],
                evanescent,
                kperp[column // n_pol],
            )
            phase[column, omega] = z * value.imag


@njit(cache=True, nogil=True)
@no_type_check
def rotate_noise(noise: np.ndarray, phase: np.ndarray, output: np.ndarray) -> None:
    """Rotate a fixed noise realization into a persistent stage buffer.

    Parameters
    ----------
    noise : numpy.ndarray
        Unadvanced complex noise spectrum.
    phase : numpy.ndarray
        Unitary phase in radians.
    output : numpy.ndarray
        Advanced spectrum, overwritten in place; may own the phase view.
    """
    for column in range(output.shape[0]):
        for omega in range(output.shape[1]):
            angle = phase[column, omega]
            output[column, omega] = noise[column, omega] * complex(
                np.cos(angle), np.sin(angle)
            )


# ------------------------------------------------- the plane measure (SEP 34)


@cache
def make_flux_columns(*, parallel: bool) -> Any:
    """Compile the per-column pass of the plane measure: energy and the closure.

    Parameters
    ----------
    parallel : bool
        Whether the columns run in a ``prange``; each column's sum is formed by one
        thread in ascending frequency order, so the partials do not depend on the
        thread count.

    Returns
    -------
    callable
        ``flux_columns(field, squares, kappa, active, omega, weights, energy,
        closure, partials)``, documented on the compiled function.
    """
    columns = prange if parallel else range

    @njit(cache=not parallel, nogil=True, parallel=parallel)
    @no_type_check
    def flux_columns(  # noqa: PLR0913, PLR0917 - one fused pass
        field: np.ndarray,
        squares: Any,
        kappa: np.ndarray,
        active: np.ndarray,
        omega: np.ndarray,
        weights: np.ndarray,
        energy: np.ndarray,
        closure: np.ndarray,
        partials: np.ndarray,
    ) -> None:
        """Weigh each column by the plane's admittance (SEP 34 section 4.3).

        Per column, ``w_c sum_k nu_k Re(eta) sum_q |E|^2`` into `partials`, and,
        when `closure` has rows, the closure ``conj(eta) E`` of the internal
        spectrum. ``eta = c k_z / omega`` is
        :func:`solprop.translate.spatial.axial_admittance`'s, from the root of
        ``longitudinal_root``'s compensated square, the wall term last.

        Parameters
        ----------
        field : numpy.ndarray
            One internal spectrum, shape ``(n_modes, n_pol, n_omega)``.
        squares : tuple
            The material square and its residual (complex, ``(n_omega,)``) and the
            transverse square and its residual (real, ``(n_modes,)``).
        kappa : numpy.ndarray
            A planar run's wall terms, ``(n_channels, n_omega)``; no rows elsewhere.
        active : numpy.ndarray
            The active band, bool, ``(n_omega,)``.
        omega : numpy.ndarray
            Physical angular frequencies in rad/s.
        weights : numpy.ndarray
            The columns' spectral weights.
        energy : numpy.ndarray
            ``fourier.energy_weights`` of the grid.
        closure : numpy.ndarray
            Overwritten with the closure, the field's shape, or no rows to skip it.
        partials : numpy.ndarray
            Overwritten with each column's weighted energy sum, ``(n_modes,)``.
        """
        square, square_low, transverse, transverse_low = squares
        n_pol = field.shape[1]
        n_omega = field.shape[2]
        n_channels = kappa.shape[0]
        closed = closure.shape[0] > 0
        for column in columns(field.shape[0]):
            total = 0.0
            for bin_ in range(n_omega):
                longitudinal = (square[bin_] - transverse[column]) + (
                    square_low[bin_] - transverse_low[column]
                )
                if n_channels > 0:
                    longitudinal = longitudinal - kappa[column % n_channels, bin_]
                closing = 0j
                if active[bin_] and longitudinal.real > 0.0 and omega[bin_] > 0.0:
                    closing = (C * np.sqrt(longitudinal).conjugate()) / omega[bin_]
                power = 0.0
                for pol in range(n_pol):
                    value = field[column, pol, bin_]
                    power += value.real * value.real + value.imag * value.imag
                    if closed:
                        closure[column, pol, bin_] = closing * value
                total += energy[bin_] * (closing.real * power)
            partials[column] = weights[column] * total

    return flux_columns


def flux_measure(
    plane: SeparableOperator,
    grid: TimeGrid,
    weights: np.ndarray,
    field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    *,
    closure: bool,
    parallel: bool,
) -> tuple[float, Complex[np.ndarray, "n_modes n_pol n_omega"] | None]:
    """Return a spectrum's axial-flux energy and, when asked, its closure (SEP 34).

    The compiled form of ``collect_spatial``'s ``energy`` and closure.

    Parameters
    ----------
    plane : solprop.translate.spatial.SeparableOperator
        The plane's coefficients (``FluxPlan.at``).
    grid : solprop.grid.TimeGrid
        The temporal grid.
    weights : numpy.ndarray
        The geometry's spectral weights.
    field : numpy.ndarray
        One internal spectrum.
    closure : bool
        Whether to form the closure ``conj(eta) E``.
    parallel : bool
        Whether the columns run in a ``prange``.

    Returns
    -------
    tuple
        The energy in J and the closure, or ``None``.
    """
    n_omega = field.shape[2]
    kappa = (
        np.empty((0, n_omega), np.complex128)
        if plane.kappa_sq is None
        else np.ascontiguousarray(plane.kappa_sq)
    )
    held = np.empty_like(field) if closure else np.empty((0, 1, 1), np.complex128)
    partials = np.empty(field.shape[0])
    make_flux_columns(parallel=parallel)(
        field,
        (plane.k_squared, plane.k_squared_low, plane.kperp_sq, plane.kperp_sq_low),
        kappa,
        plane.active,
        np.ascontiguousarray(grid.omega),
        np.ascontiguousarray(weights),
        fourier.energy_weights(n_omega, grid.dt, kind=grid.kind),
        held,
        partials,
    )
    return 0.5 * EPS0 * C * float(np.sum(partials)), held if closure else None


# ----------------------------------------------- far-field statistics (SEP 23)


@cache
def make_far_field(*, parallel: bool) -> Any:  # noqa: PLR0915 - one closure
    """Compile stage one of the far-field pass: every band's partial sums.

    Parameters
    ----------
    parallel : bool
        Whether the bands run in a ``prange``. The bands, and the order of every sum
        inside one, are fixed by the shapes
        (:func:`~solprop.backends.spatial.far_field_bands`), so the partials do not
        depend on the thread count.

    Returns
    -------
    callable
        ``far_field_bands(field, edges, columns, apertures, regions, operator,
        layout, partials)``, documented on the compiled function.
    """
    bands = prange if parallel else range

    @njit(cache=not parallel, nogil=True, parallel=parallel)
    @no_type_check
    def far_field_bands(  # noqa: PLR0912, PLR0913, PLR0915, PLR0917 - one fused pass
        field: np.ndarray,
        edges: np.ndarray,
        columns: Any,
        apertures: Any,
        regions: np.ndarray,
        operator: Any,
        layout: np.ndarray,
        partials: np.ndarray,
    ) -> None:
        """Accumulate one band of columns per row of the partials (SEP 23 section 6).

        Each band walks its columns in ascending order with frequency innermost and
        adds, per slot and bin, the sums of equations (5) and (7) to (11) without
        their constant factors, measured by the medium's axial flux (SEP 34):
        ``w_c Re(eta) sum_q |E|^2`` (times ``W^p``, a rate or a region's
        membership), and the complex products ``alpha_c W E`` and their magnetic
        closure ``alpha_c W conj(eta) E``. The admittance ``eta = c k_z/omega`` is
        the operator's own root (``longitudinal_root``) on active propagating bins
        at positive frequency and zero elsewhere, formed per element: no column by
        frequency table is held.

        Parameters
        ----------
        field : numpy.ndarray
            One internal spectrum, shape ``(n_x*n_y, n_pol, n_omega)``.
        edges : numpy.ndarray
            The bands' column edges, int64, shape ``(n_bands + 1,)``.
        columns : tuple
            The stored axes ``k_x`` and ``k_y`` in rad/m, the spectral weights and
            the complex on-axis weights.
        apertures : tuple
            Centres ``(n_apertures, 2)``, half-widths, reciprocal widths, edges
            (orders), distances, the scale table ``(n_apertures, n_omega)``, the
            profile code, the exponent ``p`` and whether every aperture shares its
            centre and distance (the radius is then evaluated once per element).
        regions : numpy.ndarray
            Boxes of physical transverse wavevector, shape ``(n_regions, 4)``.
        operator : tuple
            The separable operator's squares, residuals, active mask, evanescent
            cap, spectral rate and transverse rate, and the physical frequencies.
        layout : numpy.ndarray
            :meth:`solprop.backends.spatial.FarFieldLayout.array`.
        partials : numpy.ndarray
            Overwritten, shape ``(n_bands, n_slots, n_omega)``.
        """
        k_x, k_y, weights, on_axis = columns
        (
            centres,
            half_widths,
            inverse_widths,
            orders,
            distances,
            scale,
            code,
            exponent,
            shared,
        ) = apertures
        (
            square,
            transverse,
            square_low,
            transverse_low,
            active,
            evanescent,
            spectral_rate,
            kperp,
            omega,
        ) = operator
        beam = layout[0]
        absorbed = layout[1]
        region = layout[2]
        aperture = layout[3]
        stride = layout[4]
        collected = layout[5]
        amplitude = layout[6]
        axis = layout[7]
        n_x = k_x.shape[0]
        n_pol = field.shape[1]
        n_omega = field.shape[2]
        n_apertures = half_widths.shape[0] if aperture >= 0 else 0
        n_regions = regions.shape[0] if region >= 0 else 0
        radius = 0.0
        origin = 0.0
        for band in bands(partials.shape[0]):
            for slot in range(partials.shape[1]):
                for bin_ in range(n_omega):
                    partials[band, slot, bin_] = 0.0
            for column in range(edges[band], edges[band + 1]):
                kx = k_x[column % n_x]
                ky = k_y[column // n_x]
                weight = weights[column]
                alpha = on_axis[column]
                for bin_ in range(n_omega):
                    power = 0.0
                    for pol in range(n_pol):
                        value = field[column, pol, bin_]
                        power += value.real * value.real + value.imag * value.imag
                    longitudinal = (square[bin_] - transverse[column]) + (
                        square_low[bin_] - transverse_low[column]
                    )
                    root, propagating, attenuation = _ROOT(longitudinal, evanescent)
                    live = active[bin_]
                    # The conjugate admittance the internal spectrum's closure takes.
                    closing = 0j
                    if live and propagating and omega[bin_] > 0.0:
                        closing = (C * root.conjugate()) / omega[bin_]
                    weighted = weight * power * closing.real
                    if beam >= 0:
                        partials[band, beam, bin_] += weighted
                    if absorbed >= 0:
                        partials[band, absorbed, bin_] += weighted * spectral_rate[bin_]
                        if live:
                            partials[band, absorbed + 1, bin_] += (
                                weighted * kperp[column]
                            )
                            channel = absorbed + 3 if propagating else absorbed + 2
                            partials[band, channel, bin_] += weighted * (
                                2.0 * attenuation
                            )
                    for row in range(n_regions):
                        if _MEMBER(
                            kx,
                            ky,
                            regions[row, 0],
                            regions[row, 1],
                            regions[row, 2],
                            regions[row, 3],
                        ):
                            partials[band, region + row, bin_] += weighted
                    if axis >= 0:
                        for pol in range(n_pol):
                            product = alpha * field[column, pol, bin_]
                            partials[band, axis + 2 * pol, bin_] += product.real
                            partials[band, axis + 2 * pol + 1, bin_] += product.imag
                            closed = alpha * (closing * field[column, pol, bin_])
                            slot = axis + 2 * (n_pol + pol)
                            partials[band, slot, bin_] += closed.real
                            partials[band, slot + 1, bin_] += closed.imag
                    if n_apertures > 0 and shared:
                        radius = _RADIUS(
                            scale[0, bin_], kx, ky, centres[0, 0], centres[0, 1]
                        )
                        origin = _RADIUS(scale[0, bin_], kx, ky, 0.0, 0.0)
                    for index in range(n_apertures):
                        reach = scale[index, bin_]
                        if not shared:
                            radius = _RADIUS(
                                reach, kx, ky, centres[index, 0], centres[index, 1]
                            )
                            origin = _RADIUS(reach, kx, ky, 0.0, 0.0)
                        transmission = _WEIGHT(
                            radius,
                            origin,
                            reach,
                            half_widths[index],
                            inverse_widths[index],
                            orders[index],
                            distances[index],
                            code,
                        )
                        if transmission == 0.0:
                            continue
                        base = aperture + index * stride
                        if collected >= 0:
                            partials[band, base + collected, bin_] += weighted * (
                                transmission
                                if exponent == 1
                                else transmission * transmission
                            )
                        if amplitude >= 0:
                            slot = base + amplitude
                            for pol in range(n_pol):
                                through = transmission * field[column, pol, bin_]
                                product = alpha * through
                                partials[band, slot + 2 * pol, bin_] += product.real
                                partials[band, slot + 2 * pol + 1, bin_] += product.imag
                                closed = alpha * (closing * through)
                                again = slot + 2 * (n_pol + pol)
                                partials[band, again, bin_] += closed.real
                                partials[band, again + 1, bin_] += closed.imag

    return far_field_bands


@njit(cache=True, nogil=True)
@no_type_check
def far_field_sum(partials: np.ndarray, totals: np.ndarray) -> None:
    """Stage two of the far-field pass: sum the bands serially, in ascending order.

    Parameters
    ----------
    partials : numpy.ndarray
        Stage one's partials, shape ``(n_bands, n_slots, n_omega)``.
    totals : numpy.ndarray
        Overwritten with their sum over bands, shape ``(n_slots, n_omega)``.
    """
    for slot in range(totals.shape[0]):
        for bin_ in range(totals.shape[1]):
            totals[slot, bin_] = 0.0
    for band in range(partials.shape[0]):
        for slot in range(totals.shape[0]):
            for bin_ in range(totals.shape[1]):
                totals[slot, bin_] += partials[band, slot, bin_]


def _read_only(values: np.ndarray) -> np.ndarray:
    """Return a read-only view, so every call hands the pass one array type.

    Parameters
    ----------
    values : numpy.ndarray
        Any array.

    Returns
    -------
    numpy.ndarray
        A view that cannot be written through.
    """
    view = np.asarray(values).view()
    view.setflags(write=False)
    return view


class FarFieldPass:
    """Form the far-field statistics and the on-axis amplitude in one banded pass.

    SEP 23 section 6 on ``numba_cpu``: stage one (:func:`make_far_field`) writes one
    row of partials per band of columns, in a ``prange`` when `parallel`; stage two
    (:func:`far_field_sum`) adds the bands in ascending order; a finishing step on the
    host applies ``epsilon_0 c/2``, the spectral-density factor, the energy weights of
    ``absorbed_power`` and the pairing ``Re(A conj(G))`` of each on-axis amplitude
    with its magnetic closure (SEP 34). The partials are sized from the problem's own
    request at construction (:func:`solprop.backends.spatial.far_field_partials`) and
    reused by every call, the periodic channel's included; a call that needs more slots
    than were sized for (a spec handed to the run loop with other tables) uses
    buffers of its own.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A Cartesian or line problem, whose separable operator and columns the pass
        reads.
    parallel : bool
        Whether stage one runs its bands in a ``prange``.

    Raises
    ------
    BackendError
        If the problem is not a Cartesian or line problem with a separable operator
        that is its own plane measure.
    """

    def __init__(self, problem: KernelProblem, *, parallel: bool) -> None:
        """Hold the operator's arrays and allocate the partials once."""
        geometry = problem.transform.geometry
        operator = problem.linear.separable
        if geometry is None or geometry.kind != "free" or operator is None:
            msg = "the far-field pass needs a Cartesian or line separable problem"
            raise BackendError(msg)
        flux = problem.flux
        if flux is None or flux.base is not operator or flux.material is not None:
            # The pass reads the operator's root as the plane's (SEP 34): a uniform
            # Cartesian or line medium's measure is its own separable record.
            msg = "the far-field pass needs the run's own separable plane measure"
            raise BackendError(msg)
        n_slots, n_bands = far_field_partials(problem)
        self._n_x, self._n_y = geometry.shape
        self._n_omega = problem.shape[3]
        self._kernel = make_far_field(parallel=parallel)
        self._partials = np.empty(n_bands * n_slots * self._n_omega)
        self._totals = np.empty(n_slots * self._n_omega)
        k_x, k_y = geometry.spectral_axes
        self._columns = tuple(
            _read_only(np.ascontiguousarray(values))
            for values in (k_x, k_y, geometry.spectral_weights, geometry.on_axis)
        )
        self._operator = tuple(
            _read_only(np.ascontiguousarray(values))
            for values in (
                operator.k_squared,
                operator.kperp_sq,
                operator.k_squared_low,
                operator.kperp_sq_low,
                operator.active,
            )
        )
        self._rates = (
            float(operator.evanescent_rate),
            _read_only(np.ascontiguousarray(operator.spectral_rate)),
            _read_only(np.ascontiguousarray(operator.kperp_rate)),
        )
        empty = _read_only(np.empty(0))
        self._none = (
            _read_only(np.empty((0, 2))),
            empty,
            empty,
            empty,
            empty,
            _read_only(np.empty((0, self._n_omega))),
            0,
            2,
            False,
        )
        self._no_regions = _read_only(np.empty((0, 4)))
        self._held: tuple[ApertureTables, tuple[Any, ...]] | None = None

    @property
    def nbytes(self) -> int:
        """Return the bytes of the partials and their band sum held for the run.

        Returns
        -------
        int
            The ``statistics_partials`` row of the memory ledger.
        """
        return int(self._partials.nbytes + self._totals.nbytes)

    def _apertures(self, tables: ApertureTables | None) -> tuple[Any, ...]:
        """Return the aperture constants a call reads, formed once per tables.

        Parameters
        ----------
        tables : solprop.spatial.ApertureTables or None
            The call's tables.

        Returns
        -------
        tuple
            What stage one's ``apertures`` argument documents.
        """
        if tables is None or tables.n_apertures == 0:
            return self._none
        if self._held is not None and self._held[0] is tables:
            return self._held[1]
        shared = bool(
            np.all(tables.centres == tables.centres[0])
            and np.all(tables.distances == tables.distances[0])
        )
        constants = (
            tables.centres,
            tables.half_widths,
            _read_only(tables.inverse_widths),
            tables.edges,
            tables.distances,
            tables.scale,
            tables.profile_code,
            tables.exponent,
            shared,
        )
        self._held = (tables, constants)
        return constants

    def __call__(
        self,
        names: Collection[str],
        grid: TimeGrid,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        tables: ApertureTables | None,
        *,
        shared: bool | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        tuple[
            Complex[np.ndarray, "n_pol n_omega"], Complex[np.ndarray, "n_pol n_omega"]
        ]
        | None,
    ]:
        """Form the requested far-field statistics and the on-axis amplitude.

        Parameters
        ----------
        names : collections.abc.Collection of str
            The statistics asked for. The far-field ones and ``on_axis_spectrum`` are
            formed here; ``on_axis_intensity`` reads the returned amplitudes.
        grid : solprop.grid.TimeGrid
            The temporal grid: its spacing, kind and frequencies.
        field : numpy.ndarray
            One internal spectrum, shape ``(n_x*n_y, n_pol, n_omega)``.
        tables : solprop.spatial.ApertureTables or None
            The call's far-field tables; ``None`` when it names no far-field
            statistic.
        shared : bool or None, optional
            Force (``True``) or forbid (``False``) the path that evaluates the radius
            once per element when every aperture shares its centre and distance;
            ``None`` takes it exactly then. Forcing it with distinct apertures is
            refused.

        Returns
        -------
        tuple
            The statistics formed here, as :func:`solprop.spatial.aperture_statistics`
            and :func:`solprop.spatial.on_axis_spectrum` define them, and the on-axis
            amplitude ``sum_c alpha_c E_c`` in V/m with its magnetic closure
            ``sum_c alpha_c conj(eta_c) E_c``, each of shape ``(n_pol, n_omega)``, or
            ``None`` when no axis statistic is asked for.

        Raises
        ------
        BackendError
            If a far-field statistic is asked for without its tables, or a forced
            shared path does not apply.
        """
        wanted = set(names)
        if wanted & set(FAR_FIELD_STATISTICS) and tables is None:
            msg = (
                f"the far-field statistics {sorted(wanted & set(FAR_FIELD_STATISTICS))}"
                " need the tables the backend forms at construction"
            )
            raise BackendError(msg)
        n_pol = field.shape[1]
        layout = far_field_layout(
            wanted,
            n_apertures=0 if tables is None else tables.n_apertures,
            n_regions=0 if tables is None else tables.n_regions,
            n_pol=n_pol,
        )
        if layout is None:
            return {}, None
        apertures = self._apertures(tables)
        if shared is not None:
            if shared and not apertures[-1]:
                msg = "the shared-mask path needs one centre and one distance"
                raise BackendError(msg)
            apertures = (*apertures[:-1], shared)
        edges = far_field_bands(self._n_x, self._n_y, layout.n_slots, self._n_omega)
        size = (edges.size - 1) * layout.n_slots * self._n_omega
        partials = (
            self._partials[:size] if size <= self._partials.size else np.empty(size)
        ).reshape(edges.size - 1, layout.n_slots, self._n_omega)
        totals = (
            self._totals[: layout.n_slots * self._n_omega]
            if layout.n_slots * self._n_omega <= self._totals.size
            else np.empty(layout.n_slots * self._n_omega)
        ).reshape(layout.n_slots, self._n_omega)
        operator = (
            *self._operator,
            *self._rates,
            _read_only(np.ascontiguousarray(grid.omega)),
        )
        self._kernel(
            field,
            edges,
            self._columns,
            apertures,
            self._no_regions if tables is None else tables.regions,
            operator,
            layout.array(),
            partials,
        )
        far_field_sum(partials, totals)
        return _finish(wanted, grid, totals, layout, tables, n_pol)


def _finish(
    wanted: set[str],
    grid: TimeGrid,
    totals: np.ndarray,
    layout: FarFieldLayout,
    tables: ApertureTables | None,
    n_pol: int,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, np.ndarray] | None]:
    """Apply the constant factors to the band sums.

    Parameters
    ----------
    wanted : set of str
        The statistics asked for.
    grid : solprop.grid.TimeGrid
        The temporal grid.
    totals : numpy.ndarray
        The band sums, shape ``(n_slots, n_omega)``.
    layout : solprop.backends.spatial.FarFieldLayout
        Where each quantity lives.
    tables : solprop.spatial.ApertureTables or None
        The call's tables.
    n_pol : int
        Polarisation components.

    Returns
    -------
    tuple
        The statistics and the on-axis amplitude with its closure, as
        :meth:`FarFieldPass.__call__` documents.
    """
    conversion = 0.5 * EPS0 * C
    n_omega = totals.shape[1]
    # The spectral density is |E|^2 times a constant of the grid, formed by the one
    # definition rather than restated.
    density = fourier.spectral_density(
        np.ones(n_omega, dtype=np.complex128), grid.dt, kind=grid.kind
    )
    values: dict[str, np.ndarray] = {}
    if "beam_spectrum" in wanted:
        values["beam_spectrum"] = conversion * (density * totals[layout.beam])
    if "absorbed_power" in wanted:
        energy = fourier.energy_weights(n_omega, grid.dt, kind=grid.kind)
        rows = totals[layout.absorbed : layout.absorbed + len(ABSORBED_CHANNELS)]
        values["absorbed_power"] = conversion * np.sum(rows * energy, axis=1)
    if "region_spectrum" in wanted and tables is not None:
        rows = totals[layout.regions : layout.regions + tables.n_regions]
        values["region_spectrum"] = conversion * (density * rows)
    if tables is not None and layout.apertures >= 0:
        slots = layout.apertures + layout.stride * np.arange(tables.n_apertures)
        if "aperture_spectrum" in wanted:
            values["aperture_spectrum"] = conversion * (
                density * totals[slots + layout.collected]
            )
        if "aperture_on_axis_spectrum" in wanted:
            values["aperture_on_axis_spectrum"] = np.stack(
                [
                    _axis_spectrum(
                        grid, *_amplitude(totals, slot + layout.amplitude, n_pol)
                    )
                    for slot in slots.tolist()
                ]
            )
    axis = None if layout.axis < 0 else _amplitude(totals, layout.axis, n_pol)
    if axis is not None and "on_axis_spectrum" in wanted:
        values["on_axis_spectrum"] = _axis_spectrum(grid, *axis)
    return values, axis


def _amplitude(
    totals: np.ndarray, first: int, n_pol: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return a complex amplitude and its closure from their real and imaginary slots.

    Parameters
    ----------
    totals : numpy.ndarray
        The band sums, shape ``(n_slots, n_omega)``.
    first : int
        The amplitude's first slot: real then imaginary part per polarisation, then
        the closure's.
    n_pol : int
        Polarisation components.

    Returns
    -------
    tuple of numpy.ndarray
        The amplitude in V/m and its magnetic closure ``G / (eps0 c)``, complex,
        each of shape ``(n_pol, n_omega)``.
    """
    stop = first + 2 * n_pol
    end = stop + 2 * n_pol
    return (
        totals[first:stop:2] + 1j * totals[first + 1 : stop : 2],
        totals[stop:end:2] + 1j * totals[stop + 1 : end : 2],
    )


def _axis_spectrum(
    grid: TimeGrid, amplitude: np.ndarray, closure: np.ndarray
) -> np.ndarray:
    """Return the axial-flux spectral fluence of an on-axis amplitude (SEP 34).

    The same-frequency pairing ``Re(A conj(G))`` of the amplitude with its closure,
    polarisations summed.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The temporal grid.
    amplitude : numpy.ndarray
        The amplitude, shape ``(n_pol, n_omega)``.
    closure : numpy.ndarray
        Its magnetic closure, same shape.

    Returns
    -------
    numpy.ndarray
        J s/m^2 per bin, shape ``(n_omega,)``.
    """
    factor = fourier.spectral_density(
        np.ones(grid.n_omega, dtype=np.complex128), grid.dt, kind=grid.kind
    )
    return (
        0.5 * EPS0 * C * factor * np.sum(np.real(amplitude * np.conj(closure)), axis=0)
    )
