# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute the hybrid planar transform's compiled passes.

Kernel schema 15's ``"slab"`` kind on ``numba_cpu``: slab modes in x on the
transform's own synthesis and projection, an FFT in y, the state's columns
``(p, k_y)`` with the channel index fastest (``docs/dev/kernel-contract.md`` section
4). What is here is what the kind has and free space has not:

- the exact longitudinal operator with each channel's complex wall term
  ``kappa_sq``, evaluated per element by the shared body
  :func:`~solprop.backends.bodies.define_bodies`'s ``slab_element`` for the
  propagator, the noise phase and the stage's prefactor, never tabulated;
- the y transform: one in-place FFTW pair over the first axis of the
  ``(n_y, n_p * n_pol, n_omega)`` column volume, and, when the in-stage absorber is
  on, a second half of the same buffer that carries the absorber's rows through the
  same forward transform;
- the stage's finish, which forms ``(prefactor * window * material + mask *
  absorbed) * exp(-dPhi)`` per column in the reference's order
  (the reference backend's ``_nonlinear_slab``).

The temporal transforms, the response column, the Raman convolution, the envelope
plasma and the accumulation are the backend's own, shared with every other route: a
``(p, y)`` row is a modal amplitude whose time signal is a real field's, so only
``n_p * n_y`` rows are taken to time and back, and the x synthesis and projection are
real GEMMs in time, one per y row.
"""

import time
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple, no_type_check

import numpy as np
from numba import prange

from solprop._numba_cache import njit
from solprop.backends.numba_cpu.fft import PLAN_FLAGS, _heap_bytes
from solprop.backends.numba_cpu.kernels import HOST
from solprop.backends.numba_cpu.openmp import pyfftw
from solprop.exceptions import BackendError
from solprop.translate.problem import KernelProblem

__all__ = [
    "PlanarKernels",
    "SlabPlans",
    "build_slab_plans",
    "make_kernels",
    "slab_data",
    "write_phase",
]

_SLAB = HOST.slab_element
_PAIR = HOST.propagator_pair


class PlanarKernels(NamedTuple):
    """Group the hybrid planar kind's wide stage operations.

    Attributes
    ----------
    write_propagator, finish_stage : callable
        Traversals over independent columns, serial or parallel.
    """

    write_propagator: Any
    finish_stage: Any


@dataclass(frozen=True)
class SlabPlans:
    """Own the planar column volume and its y transforms.

    Attributes
    ----------
    spectrum : numpy.ndarray
        Aligned complex scratch ``(n_halves, n_y, n_p * n_pol, n_omega)``: the
        material half and, when the in-stage absorber is on, the absorber's half.
    rows, absorbed : numpy.ndarray
        The two halves as ``(n_columns, n_omega)`` views; ``absorbed`` is the first
        half again when there is no second one, and is then never read.
    inverse, forward : pyfftw.FFTW
        Unnormalised in-place y transforms of the material half.
    forward_all : pyfftw.FFTW
        The forward y transform of both halves together, or ``forward``.
    plan_heap_growth : int
        Process allocator net growth during planning in bytes, including opaque FFTW
        work areas; a measured count, not a bound.
    planning_seconds : float
        Time spent planning, in seconds.
    """

    spectrum: Any
    rows: Any
    absorbed: Any
    inverse: Any
    forward: Any
    forward_all: Any
    plan_heap_growth: int
    planning_seconds: float


def build_slab_plans(
    shape: tuple[int, int, int],
    *,
    absorber: bool,
    flag: str,
    threads: int,
) -> SlabPlans:
    """Plan the in-place y transforms of the planar column volume, before any state.

    Parameters
    ----------
    shape : tuple of int
        ``(n_y, n_p * n_pol, n_omega)``, all positive.
    absorber : bool
        Whether to allocate the absorber's half and transform it with the material.
    flag : str
        A key of :data:`solprop.backends.numba_cpu.fft.PLAN_FLAGS`.
    threads : int
        FFTW's pool for these plans, positive.

    Returns
    -------
    SlabPlans
        The volume, its views and its three plans.

    Raises
    ------
    BackendError
        If the planning flag is not one of the supported ones.
    """
    if flag not in PLAN_FLAGS:
        msg = f"{flag!r} is not a planning flag"
        raise BackendError(msg)
    chosen = PLAN_FLAGS[flag]
    started = time.perf_counter()
    spectrum = pyfftw.empty_aligned((2 if absorber else 1, *shape), "complex128")
    heap_before = _heap_bytes()

    def plan(buffer: Any, axis: int, direction: str) -> Any:
        """Build one unnormalised in-place plan over the y axis.

        Parameters
        ----------
        buffer : numpy.ndarray
            The volume or its material half, transformed in place.
        axis : int
            The y axis of `buffer`.
        direction : str
            ``"FFTW_FORWARD"`` or ``"FFTW_BACKWARD"``.

        Returns
        -------
        pyfftw.FFTW
            The plan, bound to `buffer`.
        """
        return pyfftw.FFTW(
            buffer,
            buffer,
            axes=(axis,),
            direction=direction,
            flags=(chosen,),
            threads=threads,
        )

    inverse = plan(spectrum[0], 0, "FFTW_BACKWARD")
    forward = plan(spectrum[0], 0, "FFTW_FORWARD")
    forward_all = plan(spectrum, 1, "FFTW_FORWARD") if absorber else forward
    columns = shape[0] * shape[1]
    return SlabPlans(
        spectrum=spectrum,
        rows=spectrum[0].reshape(columns, shape[2]),
        absorbed=spectrum[-1].reshape(columns, shape[2]),
        inverse=inverse,
        forward=forward,
        forward_all=forward_all,
        plan_heap_growth=max(0, _heap_bytes() - heap_before),
        planning_seconds=time.perf_counter() - started,
    )


def slab_data(problem: KernelProblem) -> tuple[Any, ...]:
    """Pack the separable leaf of a hybrid planar problem as its kernels take it.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A validated problem whose geometry is ``"slab"``.

    Returns
    -------
    tuple
        The free-space record's arrays with ``kappa_sq`` after the transverse
        residual, then the evanescent rate, the state's polarisation count, the
        channel count and the safety ramp's rate per column (SEP 40).
    """
    data = problem.linear.separable
    assert data is not None and data.kappa_sq is not None
    return (
        data.k_squared,
        data.kperp_sq,
        data.k_squared_low,
        data.kperp_sq_low,
        data.kappa_sq,
        data.frame,
        data.spectral_rate,
        data.prefactor_numerator,
        data.prefactor_clamped,
        data.active,
        data.evanescent_rate,
        data.n_pol,
        data.n_channels,
        data.kperp_rate,
    )


@cache
def make_kernels(*, parallel: bool) -> PlanarKernels:
    """Compile the planar stage operations, serial or across independent columns.

    Parameters
    ----------
    parallel : bool
        Whether the column loop is a ``prange``. Each column's arithmetic is the same
        either way, so the two variants agree to the last bit.

    Returns
    -------
    PlanarKernels
        The compiled operations.
    """
    # The iterator is captured so the choice is in the closure's cache key; only the
    # serial variants are disk-cached.
    wide_columns = prange if parallel else range

    @njit(cache=not parallel, nogil=True, parallel=parallel)
    @no_type_check
    def write_propagator(
        data: Any, distance: float, forward: np.ndarray, inverse: np.ndarray
    ) -> None:
        """Evaluate both propagation signs of every column without a table.

        Parameters
        ----------
        data : tuple
            :func:`slab_data`'s record.
        distance : float
            Propagation increment in m.
        forward, inverse : numpy.ndarray
            Preallocated propagation-factor arrays ``(n_columns, n_omega)``,
            complex128, overwritten.
        """
        (
            square,
            transverse,
            square_low,
            transverse_low,
            wall,
            frame,
            rate,
            numerator,
            clamped,
            active,
            evanescent,
            n_pol,
            n_channels,
            ramp,
        ) = data
        for column in wide_columns(forward.shape[0]):
            mode = column // n_pol
            channel = mode % n_channels
            for omega in range(forward.shape[1]):
                value, _ = _SLAB(
                    square[omega],
                    transverse[mode],
                    square_low[omega],
                    transverse_low[mode],
                    wall[channel, omega],
                    frame[omega],
                    rate[omega],
                    numerator[omega],
                    clamped[omega],
                    active[omega],
                    evanescent,
                    ramp[mode],
                )
                exponent = distance * value
                fr, fi, br, bi = _PAIR(exponent.real, exponent.imag)
                forward[column, omega] = complex(fr, fi)
                inverse[column, omega] = complex(br, bi)

    @njit(cache=not parallel, nogil=True, parallel=parallel)
    @no_type_check
    def finish_stage(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        data: Any,
        material: np.ndarray,
        absorbed: np.ndarray,
        window: np.ndarray,
        mask: np.ndarray,
        inverse: np.ndarray,
        stage: np.ndarray,
        flags: tuple[bool, bool],
    ) -> None:
        """Form one stage derivative from the transformed sources, in place.

        ``(prefactor * window * material + mask * absorbed) * exp(-dPhi)``, in the
        reference's order; a term whose flag is off is not formed, so a stage with
        the absorber alone is ``(mask * absorbed) * exp(-dPhi)``.

        Parameters
        ----------
        data : tuple
            :func:`slab_data`'s record.
        material : numpy.ndarray
            The polarisation's columns ``(n_columns, n_omega)``.
        absorbed : numpy.ndarray
            The in-stage absorber's columns, the same shape.
        window : numpy.ndarray
            Spectral taper, dimensionless.
        mask : numpy.ndarray
            The active band, boolean.
        inverse : numpy.ndarray
            Inverse interaction factor ``exp(-dPhi)``.
        stage : numpy.ndarray
            Stage derivative, overwritten.
        flags : tuple of bool
            Whether the material and the absorber terms are present.
        """
        (
            square,
            transverse,
            square_low,
            transverse_low,
            wall,
            frame,
            rate,
            numerator,
            clamped,
            active,
            evanescent,
            n_pol,
            n_channels,
            ramp,
        ) = data
        for column in wide_columns(stage.shape[0]):
            mode = column // n_pol
            channel = mode % n_channels
            for omega in range(stage.shape[1]):
                value = 0j
                if flags[0]:
                    _, prefactor = _SLAB(
                        square[omega],
                        transverse[mode],
                        square_low[omega],
                        transverse_low[mode],
                        wall[channel, omega],
                        frame[omega],
                        rate[omega],
                        numerator[omega],
                        clamped[omega],
                        active[omega],
                        evanescent,
                        ramp[mode],
                    )
                    value = prefactor * window[omega] * material[column, omega]
                if flags[1] and mask[omega]:
                    value = value + absorbed[column, omega]
                stage[column, omega] = value * inverse[column, omega]

    return PlanarKernels(write_propagator, finish_stage)


_SERIAL = make_kernels(parallel=False)


@njit(cache=True, nogil=True)
@no_type_check
def write_phase(data: Any, z: float, phase: np.ndarray) -> None:
    """Evaluate the unitary noise phase ``z Im D`` of every column.

    Parameters
    ----------
    data : tuple
        :func:`slab_data`'s record.
    z : float
        Absolute propagation position in m.
    phase : numpy.ndarray
        Real phase buffer ``(n_columns, n_omega)``, overwritten in radians.
    """
    (
        square,
        transverse,
        square_low,
        transverse_low,
        wall,
        frame,
        rate,
        numerator,
        clamped,
        active,
        evanescent,
        n_pol,
        n_channels,
        ramp,
    ) = data
    for column in range(phase.shape[0]):
        mode = column // n_pol
        channel = mode % n_channels
        for omega in range(phase.shape[1]):
            value, _ = _SLAB(
                square[omega],
                transverse[mode],
                square_low[omega],
                transverse_low[mode],
                wall[channel, omega],
                frame[omega],
                rate[omega],
                numerator[omega],
                clamped[omega],
                active[omega],
                evanescent,
                ramp[mode],
            )
            phase[column, omega] = z * value.imag
