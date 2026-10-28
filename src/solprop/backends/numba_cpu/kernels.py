# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The fused stage kernels: seven numbered passes in a handful of traversals.

``docs/dev/architecture.md`` section 8 names seven passes and the reference backend
performs each of them as its own traversal of the state. Every one of those traversals
reads a field-sized array from memory and writes another, and at the sizes this package
targets none of them fits a cache. The kernels do the same arithmetic in the same
order with the intermediate arrays never formed: the stage combination, the lab frame,
the noise and the pad are one traversal; the whole nonlinear term at one node -- the
rate lookup, the plasma column's two running sums, the loss-current guard, the Kerr
cubic and the Raman drive -- is one traversal of ``t``; and the crop, the prefactor, the
windows, the return to the interaction picture and both accumulations are one more.

**Specialisation is by closure constant, never by argument.** A branch in the innermost
loop is not free: the Raman flag was an argument first and the runtime test cost the RDW
step 38 per cent (3.79 ms against 5.24). Every flag a kernel needs is bound when the
factory is called, so numba folds the branch away and a problem without a term pays
nothing for it.

**Reductions are two-stage.** A kernel writes one partial per column and the driver
sums them; the order of either sum is free (SEP 25: a norm is held to its summation
bound, not to bits). The column loop is a ``prange`` in the parallel variants, which a
state of many columns takes (:data:`ACCEPT_ELEMENTS`). Reassociating the masked sums
within a column was measured and gained nothing: the mask's select keeps the loop
scalar, and multiplying by the mask instead would turn a non-finite value outside the
band into a NaN norm.

**The floating-point licence** is SEP 25's, applied by ``solprop._numba_cache.njit``:
every kernel contracts multiply-adds; the protected bodies (``define_bodies``'s
``protect``) are compiled strict and called, never inlined. Nothing in here allocates,
logs, raises or touches a Python object. ``numba.prange`` falls back to ``range`` when a
function is compiled without ``parallel=True``, so the serial and parallel variants of
``docs/dev/backends.md`` section 3.2 come from one body rather than from a copy of it.

**Every kernel carries** ``@no_type_check``: the test session's jaxtyping hook appends
its checker innermost, and ``njit`` would otherwise compile the checker's wrapper.
"""

import math
from functools import cache
from typing import Any, NamedTuple, no_type_check

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Num
from numba import prange

from solprop._numba_cache import njit
from solprop.backends.bodies import RateArrays, define_bodies

__all__ = [
    "ACCEPT_COLUMNS",
    "ACCEPT_ELEMENTS",
    "BAND_DIAGNOSTICS",
    "BAND_STATS_ROWS",
    "BOUNDARY_SUMS",
    "HOST",
    "ColumnSpec",
    "accept_and_partials",
    "accept_and_partials_parallel",
    "accept_kernels",
    "add_temporal_absorber",
    "apply_hard_mask",
    "band_work_rows",
    "boundary_kernels",
    "boundary_partials",
    "boundary_partials_parallel",
    "column_spec",
    "exponentiate",
    "make_band_plasma_column",
    "make_band_stats_column",
    "make_combine_and_pad",
    "make_finish_stage",
    "make_raman_finish",
    "make_response_column",
    "make_stats_column",
    "peak_reduce",
    "removal_partials",
    "removal_partials_parallel",
    "scale_samples",
    "scale_spectrum",
    "scale_tensor_spectrum",
    "weighted_residual",
    "weighted_residual_parallel",
    "write_temporal_absorber",
]

HOST = define_bodies(njit(inline="always"), njit(strict=True, nogil=True))
"""The compiled body family every kernel inlines, and the protected bodies it calls.

Not disk-cached: a kernel inlines these bodies' IR and caches the result, and
a body that captures another would pickle differently in every process, so its own entry
would never be read back. The protected bodies are compiled strict, on their own, and
called (SEP 25 section 5).
"""

_RAMAN_DRIVE = HOST.raman_drive
_RAMAN_FINISH = HOST.raman_finish
_RAMAN_TENSOR_DRIVE = HOST.raman_tensor_drive
_RAMAN_TENSOR_FINISH = HOST.raman_tensor_finish


@njit(cache=True, nogil=True)
@no_type_check
def add_temporal_absorber(
    samples: Inexact[np.ndarray, "n_columns n_to"],
    rate: Float[np.ndarray, " n_to"],
    output: Inexact[np.ndarray, "n_columns n_to"],
) -> None:
    """Add the temporal rate to a precomputed spatial absorber term in place.

    Parameters
    ----------
    samples : numpy.ndarray
        Fine temporal samples, shape ``(n_columns, n_to)``.
    rate : numpy.ndarray
        Power absorption rates in m^-1, shape ``(n_to,)``.
    output : numpy.ndarray
        Existing derivative term; receives ``-rate*samples/2``.
    """
    for column in range(samples.shape[0]):
        for time in range(samples.shape[1]):
            output[column, time] += -0.5 * rate[time] * samples[column, time]


@njit(cache=True, nogil=True)
@no_type_check
def write_temporal_absorber(
    samples: Inexact[np.ndarray, "n_columns n_to"],
    rate: Float[np.ndarray, " n_to"],
    output: Inexact[np.ndarray, "n_columns n_to"],
) -> None:
    """Write the temporal absorber's time-domain term, ``-rate*samples/2``.

    The product is grouped as :func:`solprop.boundaries.nonlinear_absorber_term`
    groups it, ``(-0.5 * rate) * samples``, so the samples agree with the reference's
    bit for bit.

    Parameters
    ----------
    samples : numpy.ndarray
        Fine temporal samples, real for a field grid (an analytic signal's real part
        may be a strided view), shape ``(n_columns, n_to)``.
    rate : numpy.ndarray
        Power absorption rates in m^-1, shape ``(n_to,)``.
    output : numpy.ndarray
        The planned forward transform's input, overwritten.
    """
    for column in range(samples.shape[0]):
        for time in range(samples.shape[1]):
            output[column, time] = -0.5 * rate[time] * samples[column, time]


# Bound to module names so a kernel closes over the function itself: numba cannot type
# the Bodies record as a global, and reaching through it would be a Python attribute
# lookup in compiled code.
_RATE_AT = HOST.rate_at
_TRAPEZOID = HOST.trapezoid
_FRACTION_AT = HOST.fraction_at
_LOSS_SCALE = HOST.loss_scale
_KERR_REAL = HOST.kerr_real
_KERR_ANALYTIC = HOST.kerr_analytic
_PROPAGATOR_PAIR = HOST.propagator_pair


class ColumnSpec(NamedTuple):
    """Which response column a problem compiles to, and how it is executed.

    Every field is a closure constant of the kernel the factory returns, so two
    problems that differ in any of them get two compiled kernels and neither pays a
    branch for the other.

    Attributes
    ----------
    family : str
        ``"field_real"``, ``"field_analytic"`` or ``"envelope"``; see
        :data:`solprop.backends.numba_cpu.fft.ColumnFamily`.
    plasma : bool
        Whether the ionisation rate, the plasma column and the loss current are
        evaluated. Only a field grid has them.
    raman : bool
        Whether the column writes the Raman drive out onto the doubled grid.
    raman_thg : bool
        Whether the drive keeps the third harmonic (``E^2`` rather than
        ``|E_analytic|^2 / 2``). Meaningless on an envelope grid, which cannot carry it
        in a quadratic drive.
    harmonic : bool
        Whether the envelope's third-harmonic Kerr term is added with the carrier.
    n_pol : int
        Transverse components at one node, 1 or 2. The Kerr sums run over them.
    parallel : bool
        Whether the node loop is a ``prange``.
    stash : bool
        Whether the plasma branch leaves the final ionised fraction and the largest
        rate of each node behind for the between-planes statistics channel. A closure
        constant like every other flag, so a run without the channel compiles the two
        stores away and pays nothing.
    raman_drives : int
        One scalar or isotropic trace history, or three trace/difference/cross
        histories for a vector rank-two response.
    band : bool
        Whether the plasma column integrates by the band rule (SEP 18). The plasma
        then runs in :func:`make_band_plasma_column`'s passes, and the fused column is
        compiled without plasma and writes non-plasma polarisation separately.
    preloaded : bool
        Whether the fused column starts each sample's total from what ``out`` already
        holds: an existing non-plasma contribution rather than zero.
    """

    family: str
    plasma: bool
    raman: bool
    raman_thg: bool
    harmonic: bool
    n_pol: int
    parallel: bool
    stash: bool = False
    n_channels: int = 1
    n_raman: int = 1
    raman_drives: int = 1
    band: bool = False
    preloaded: bool = False


# --- the small kernels ---------------------------------------------------------


@njit(cache=True, nogil=True)
@no_type_check
def exponentiate(
    exponent: Complex[np.ndarray, "n_columns n_omega"],
    forward: Complex[np.ndarray, "n_columns n_omega"],
    inverse: Complex[np.ndarray, "n_columns n_omega"],
) -> None:
    """Form ``exp(+dPhi)`` and ``exp(-dPhi)`` from one real ``exp`` and one ``sincos``.

    A complex exponential per sign, which is what the reference evaluates, costs the
    transcendental twice for numbers that share all their trigonometry.

    Parameters
    ----------
    exponent : numpy.ndarray
        ``dPhi`` for one stage fraction, complex, shape ``(n_columns, n_omega)``.
    forward : numpy.ndarray
        Written with ``exp(+dPhi)``, same shape and dtype.
    inverse : numpy.ndarray
        Written with ``exp(-dPhi)``, same shape and dtype.
    """
    n_columns, n_omega = exponent.shape
    for column in range(n_columns):
        for j in range(n_omega):
            value = exponent[column, j]
            f_re, f_im, i_re, i_im = _PROPAGATOR_PAIR(value.real, value.imag)
            forward[column, j] = complex(f_re, f_im)
            inverse[column, j] = complex(i_re, i_im)


@njit(cache=True, nogil=True)
@no_type_check
def apply_hard_mask(
    state: Complex[np.ndarray, "n_columns n_omega"],
    mask: Bool[np.ndarray, " n_omega"],
    out: Complex[np.ndarray, "n_columns n_omega"],
) -> None:
    """Zero every bin outside the active band, into ``out``.

    The active-band mask, applied once per accepted step. It is deliberately *not* part
    of a stage's pass 7: the reference applies it in its own finishing pass, and a fused
    pass 7 that added it would mask the stage derivatives as well as the state.

    Parameters
    ----------
    state : numpy.ndarray
        The state, complex, shape ``(n_columns, n_omega)``.
    mask : numpy.ndarray
        The active band, boolean, shape ``(n_omega,)``.
    out : numpy.ndarray
        Written with the masked state. May be ``state`` itself.
    """
    n_columns, n_omega = state.shape
    for column in range(n_columns):
        for j in range(n_omega):
            out[column, j] = state[column, j] if mask[j] else 0.0 + 0.0j


@njit(cache=True, nogil=True)
@no_type_check
def scale_samples(
    samples: Inexact[np.ndarray, "n_columns n_t"],
    factor: Float[np.ndarray, " n_t"],
) -> None:
    """Multiply the coarse time samples by the absorber's collar, in place.

    Parameters
    ----------
    samples : numpy.ndarray
        The state in time on the coarse grid, shape ``(n_columns, n_t)``.
    factor : numpy.ndarray
        ``exp(-rate dz / 2)`` on the coarse time axis, shape ``(n_t,)``.
    """
    n_columns, n_t = samples.shape
    for column in range(n_columns):
        for k in range(n_t):
            samples[column, k] *= factor[k]


@njit(cache=True, nogil=True)
@no_type_check
def scale_spectrum(
    spectrum: Complex[np.ndarray, "n_rows n_omega_doubled"],
    kernel: Complex[np.ndarray, " n_omega_doubled"],
    scale: float,
) -> None:
    """Multiply the Raman drive's spectrum by the damped kernel, in place.

    Parameters
    ----------
    spectrum : numpy.ndarray
        The drive's one-sided spectrum on the doubled grid, complex, shape
        ``(n_rows, n_to + 1)``.
    kernel : numpy.ndarray
        The damped kernel's spectrum at this position, shape ``(n_to + 1,)``.
    scale : float
        The ``1 / (2 n_to)`` that :func:`numpy.fft.irfft` applies and pyfftw's
        ``execute`` does not -- the same fold as the modal pair's, with its own factor.
    """
    n_rows, n_omega = spectrum.shape
    for row in range(n_rows):
        for j in range(n_omega):
            spectrum[row, j] = spectrum[row, j] * kernel[j] * scale


@njit(cache=True, nogil=True)
@no_type_check
def scale_tensor_spectrum(
    spectrum: Complex[np.ndarray, "n_rows n_omega_doubled"],
    kernel: Complex[np.ndarray, "n_raman_kernels n_omega_doubled"],
    scale: float,
) -> None:
    """Multiply interleaved tensor drives by trace or traceless kernel spectra.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Owned FFT rows, one or three consecutive rows per physical node.
    kernel : numpy.ndarray
        Trace-only or trace/traceless spectra, including dt.
    scale : float
        Explicit inverse-transform normalization ``1 / (2 n_to)``.
    """
    n_drives = 1 if kernel.shape[0] == 1 else 3
    for row in range(spectrum.shape[0]):
        channel = 0 if row % n_drives == 0 else 1
        for j in range(spectrum.shape[1]):
            spectrum[row, j] = spectrum[row, j] * kernel[channel, j] * scale


ACCEPT_ELEMENTS = 1 << 16
"""State size from which the accept pass and the weighted residual thread their columns.

Both run a column per ``prange`` iteration in their parallel variants, which a state of
at least this many elements in at least :data:`ACCEPT_COLUMNS` columns takes when the
kernel pool has more than one thread. Measured on six threads of an Apple-silicon laptop
under load: 8 x 8192 elements 104 -> 80 us, 64 x 2048 224 -> 107 us, 2048 x 1024
3.5 -> 0.96 ms; 4 x 8192 slower (50 -> 63 us). Each column's sum is the serial one, so
the variants agree exactly.
"""

ACCEPT_COLUMNS = 8
"""Columns the parallel accept pass needs: with fewer the threads have too little."""


@njit(cache=True, nogil=True)
@no_type_check
def accept_and_partials(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
    state: Complex[np.ndarray, "n_columns n_omega"],
    propagator: Complex[np.ndarray, "n_columns n_omega"],
    dz: float,
    solution: Complex[np.ndarray, "n_columns n_omega"],
    error: Complex[np.ndarray, "n_columns n_omega"],
    mask: Bool[np.ndarray, " n_omega"],
    advanced: Complex[np.ndarray, "n_columns n_omega"],
    partials: Float[np.ndarray, "three n_columns"],
) -> None:
    """Take the accept pass and the error norm's partials in one traversal.

    ``partials`` holds three sums per column -- of ``|y|^2``, ``|y_new|^2`` and
    ``|propagator (h e)|^2`` over the in-band mask -- and the driver sums them over the
    columns. The column loop is a ``prange``: :func:`accept_and_partials_parallel` is
    the same body compiled parallel.

    Parameters
    ----------
    state : numpy.ndarray
        The state at the start of the step, complex, shape ``(n_columns, n_omega)``.
    propagator : numpy.ndarray
        ``exp(dPhi)`` over the whole step, same shape.
    dz : float
        The step, m.
    solution : numpy.ndarray
        The weighted stage sum of the solution, same shape.
    error : numpy.ndarray
        The weighted stage sum of the embedded difference, same shape.
    mask : numpy.ndarray
        The in-band mask the norm is taken over, boolean, shape ``(n_omega,)``.
    advanced : numpy.ndarray
        Written with the state at the end of the step.
    partials : numpy.ndarray
        Written with the three sums, shape ``(3, n_columns)``.
    """
    n_columns, n_omega = state.shape
    for column in prange(n_columns):
        old = 0.0
        new = 0.0
        residual = 0.0
        for j in range(n_omega):
            end = propagator[column, j] * (state[column, j] + dz * solution[column, j])
            advanced[column, j] = end
            if mask[j]:
                before = state[column, j]
                scaled = propagator[column, j] * (dz * error[column, j])
                old += before.real * before.real + before.imag * before.imag
                new += end.real * end.real + end.imag * end.imag
                residual += scaled.real * scaled.real + scaled.imag * scaled.imag
        partials[0, column] = old
        partials[1, column] = new
        partials[2, column] = residual


# Not disk-cached: numba names a parfor's gufunc per process (`backends.md` 3.2).
accept_and_partials_parallel = njit(cache=False, nogil=True, parallel=True)(
    # The plain function under ``NUMBA_DISABLE_JIT``, which has no ``py_func``.
    getattr(accept_and_partials, "py_func", accept_and_partials)
)
""":func:`accept_and_partials` with its columns threaded (:data:`ACCEPT_ELEMENTS`)."""


@njit(cache=True, nogil=True)
@no_type_check
def weighted_residual(
    propagator: Complex[np.ndarray, "n_columns n_omega"],
    dz: float,
    error: Complex[np.ndarray, "n_columns n_omega"],
    mask: Bool[np.ndarray, " n_omega"],
    scale: Float[np.ndarray, " n_omega"],
    partials: Float[np.ndarray, " n_columns"],
) -> None:
    """Sum the error norm's residual with a **per-bin** scale, one partial per column.

    :meth:`solprop.stepper.controller.PIController.scale` returns one number when the
    controller has no weights and one number per bin when it has, and the weighted form
    divides *before* the norm is taken: the weights redistribute the tolerance across
    the band rather than rescaling the whole of it. A single summed residual divided by
    a scalar cannot express that, so a weighted run pays this second traversal -- and
    only a weighted run does.

    Parameters
    ----------
    propagator : numpy.ndarray
        ``exp(dPhi)`` over the whole step.
    dz : float
        The step, m.
    error : numpy.ndarray
        The weighted stage sum of the embedded difference.
    mask : numpy.ndarray
        The in-band mask the norm is taken over.
    scale : numpy.ndarray
        The controller's per-bin scale.
    partials : numpy.ndarray
        Written with one sum of squares per column, shape ``(n_columns,)``.
    """
    n_columns, n_omega = error.shape
    for column in prange(n_columns):
        total = 0.0
        for j in range(n_omega):
            if mask[j]:
                value = propagator[column, j] * (dz * error[column, j]) / scale[j]
                total += value.real * value.real + value.imag * value.imag
        partials[column] = total


weighted_residual_parallel = njit(cache=False, nogil=True, parallel=True)(
    # The plain function under ``NUMBA_DISABLE_JIT``, which has no ``py_func``.
    getattr(weighted_residual, "py_func", weighted_residual)
)
""":func:`weighted_residual` with its columns threaded (:data:`ACCEPT_ELEMENTS`)."""


def accept_kernels(threads: int, n_columns: int, n_omega: int) -> tuple[Any, Any]:
    """Choose the serial or the column-threaded accept pass and weighted residual.

    Parameters
    ----------
    threads : int
        Threads the kernel pool has.
    n_columns : int
        Columns of the state the pass traverses.
    n_omega : int
        Bins per column.

    Returns
    -------
    tuple
        :func:`accept_and_partials` and :func:`weighted_residual`, or their parallel
        variants when the state is large enough to share (:data:`ACCEPT_ELEMENTS`).

    Examples
    --------
    >>> from solprop.backends.numba_cpu import kernels
    >>> kernels.accept_kernels(6, 4, 8192)[0] is kernels.accept_and_partials
    True
    """
    threaded = (
        threads > 1
        and n_columns >= ACCEPT_COLUMNS
        and n_columns * n_omega >= ACCEPT_ELEMENTS
    )
    if threaded:
        return accept_and_partials_parallel, weighted_residual_parallel
    return accept_and_partials, weighted_residual


@njit(cache=True, nogil=True)
@no_type_check
def peak_reduce(
    columns: Complex[np.ndarray, "n_components n_t"],
    partials: Float[np.ndarray, " one"],
) -> None:
    """Reduce ``max_t sum_c |x_c(t)|^2`` over a set of columns.

    The reduction behind the on-axis peak intensity. The sum over components is inside
    the maximum, so the answer is the largest total at any one instant and not the sum
    of separately timed maxima.

    Parameters
    ----------
    columns : numpy.ndarray
        The analytic field at one point, complex, shape ``(n_components, n_t)``.
    partials : numpy.ndarray
        Written with the running maximum, shape ``(1,)``.
    """
    n_components, n_t = columns.shape
    largest = 0.0
    for k in range(n_t):
        total = 0.0
        for component in range(n_components):
            value = columns[component, k]
            total += value.real * value.real + value.imag * value.imag
        largest = max(largest, total)
    partials[0] = largest


BOUNDARY_SUMS = 5
"""The sums the boundary monitor reads per row: the energy and its four weighted
energies (:class:`solprop.monitor.BoundaryEnergies`)."""


@njit(cache=True, nogil=True)
@no_type_check
def boundary_partials(
    samples: Inexact[np.ndarray, "n_rows n_t"],
    row_weights: Float[np.ndarray, " n_weighted"],
    weights: Float[np.ndarray, "four n_t"],
    partials: Float[np.ndarray, " n_partials"],
) -> None:
    """Sum each row's power and its four weighted powers, one set of partials per row.

    The boundary monitor's reading of a chunk of rows on the coarse time axis
    (:meth:`solprop.monitor.BoundaryMonitor.energies`): ``sum_t |x(t)|^2`` and
    ``sum_t w_k(t) |x(t)|^2`` for the monitor's four weight rows, each times the row's
    quadrature weight. The driver sums the partials over rows and chunks; the
    reference sums over rows first and over time second, and either order is held to
    its summation bound (SEP 25). The row loop is a ``prange``:
    :func:`boundary_partials_parallel` is the same body compiled parallel.

    Parameters
    ----------
    samples : numpy.ndarray
        The rows in time, real on a field grid and complex on an envelope grid, shape
        ``(n_rows, n_t)``.
    row_weights : numpy.ndarray
        Each row's quadrature weight, shape ``(n_rows,)``, or empty for unit weights.
    weights : numpy.ndarray
        The monitor's four weight rows, shape ``(4, n_t)``
        (:attr:`solprop.monitor.BoundaryMonitor.weights`).
    partials : numpy.ndarray
        Written with :data:`BOUNDARY_SUMS` sums per row, row by row: the energy, then
        the four weighted energies. At least ``BOUNDARY_SUMS * n_rows`` long.
    """
    n_rows, n_t = samples.shape
    weighted = row_weights.shape[0] > 0
    for row in prange(n_rows):
        total = 0.0
        low = 0.0
        high = 0.0
        collar = 0.0
        rate = 0.0
        for k in range(n_t):
            value = samples[row, k]
            power = value.real * value.real + value.imag * value.imag
            total += power
            low += weights[0, k] * power
            high += weights[1, k] * power
            collar += weights[2, k] * power
            rate += weights[3, k] * power
        scale = row_weights[row] if weighted else 1.0
        base = BOUNDARY_SUMS * row
        partials[base] = scale * total
        partials[base + 1] = scale * low
        partials[base + 2] = scale * high
        partials[base + 3] = scale * collar
        partials[base + 4] = scale * rate


# Not disk-cached: numba names a parfor's gufunc per process (`backends.md` 3.2).
boundary_partials_parallel = njit(cache=False, nogil=True, parallel=True)(
    # The plain function under ``NUMBA_DISABLE_JIT``, which has no ``py_func``.
    getattr(boundary_partials, "py_func", boundary_partials)
)
""":func:`boundary_partials` with its rows threaded (:data:`ACCEPT_ELEMENTS`)."""


@njit(cache=True, nogil=True)
@no_type_check
def removal_partials(
    samples: Inexact[np.ndarray, "n_rows n_t"],
    row_weights: Float[np.ndarray, " n_weighted"],
    taken: Float[np.ndarray, " n_t"],
    partials: Float[np.ndarray, " n_partials"],
) -> None:
    """Sum what a split-step collar takes of each row, one partial per row.

    ``sum_t (1 - exp(-alpha(t) h)) |x(t)|^2`` of the rows the split is about to
    damp, each times the row's quadrature weight
    (:meth:`solprop.monitor.BoundaryMonitor.split_removal`). The row loop is a
    ``prange``: :func:`removal_partials_parallel` is the same body compiled parallel.

    Parameters
    ----------
    samples : numpy.ndarray
        The rows in time before the split, real on a field grid and complex on an
        envelope grid, shape ``(n_rows, n_t)``.
    row_weights : numpy.ndarray
        Each row's quadrature weight, shape ``(n_rows,)``, or empty for unit weights.
    taken : numpy.ndarray
        ``1 - exp(-alpha(t) h)`` on the coarse time axis, shape ``(n_t,)``.
    partials : numpy.ndarray
        Written with one sum per row; at least ``n_rows`` long.
    """
    n_rows, n_t = samples.shape
    weighted = row_weights.shape[0] > 0
    for row in prange(n_rows):
        total = 0.0
        for k in range(n_t):
            value = samples[row, k]
            total += taken[k] * (value.real * value.real + value.imag * value.imag)
        partials[row] = (row_weights[row] if weighted else 1.0) * total


removal_partials_parallel = njit(cache=False, nogil=True, parallel=True)(
    # The plain function under ``NUMBA_DISABLE_JIT``, which has no ``py_func``.
    getattr(removal_partials, "py_func", removal_partials)
)
""":func:`removal_partials` with its rows threaded (:data:`ACCEPT_ELEMENTS`)."""


def boundary_kernels(threads: int, n_rows: int, n_t: int) -> tuple[Any, Any]:
    """Choose the serial or the row-threaded boundary-monitor reductions.

    One traversal of the rows each, as the accept pass's, so the same measured
    thresholds decide (:data:`ACCEPT_ELEMENTS`, :data:`ACCEPT_COLUMNS`).

    Parameters
    ----------
    threads : int
        Threads the kernel pool has.
    n_rows : int
        Rows one call traverses: a chunk's.
    n_t : int
        Coarse time samples per row.

    Returns
    -------
    tuple
        :func:`boundary_partials` and :func:`removal_partials`, or their parallel
        variants when a chunk is large enough to share.

    Examples
    --------
    >>> from solprop.backends.numba_cpu import kernels
    >>> kernels.boundary_kernels(1, 64, 8192)[0] is kernels.boundary_partials
    True
    >>> threaded = kernels.boundary_kernels(6, 64, 8192)
    >>> threaded[1] is kernels.removal_partials_parallel
    True
    """
    threaded = (
        threads > 1 and n_rows >= ACCEPT_COLUMNS and n_rows * n_t >= ACCEPT_ELEMENTS
    )
    if threaded:
        return boundary_partials_parallel, removal_partials_parallel
    return boundary_partials, removal_partials


def column_spec(
    responses: Any, kind: str, n_pol: int, *, parallel: bool, stash: bool = False
) -> ColumnSpec:
    """Read the specialisation off a compiled response set.

    One place decides which kernel a problem gets, so the backend and the tests cannot
    disagree about it.

    Parameters
    ----------
    responses : solprop.translate.responses.CompiledResponses
        The compiled response set the contract carries.
    kind : {'field', 'envelope'}
        The spectral layout.
    n_pol : int
        Transverse components at one node, 1 or 2.
    parallel : bool
        Whether the node loop is a ``prange``, which the threading regime decides.
    stash : bool, optional
        Whether the plasma branch leaves each node's final ionised fraction and largest
        rate behind, which the between-planes statistics channel reads.

    Returns
    -------
    ColumnSpec
        The specialisation.

    Examples
    --------
    >>> from solprop.backends.numba_cpu.kernels import ColumnSpec
    >>> ColumnSpec("envelope", False, False, False, True, 1, False).harmonic
    True
    """
    kerr = responses.kerr_coefficients
    dispersive = getattr(responses, "dispersive_kerr", None)
    raman = responses.raman_terms[0] if responses.raman_terms else None
    if kind == "envelope":
        family = "envelope"
    elif (kerr is not None and kerr.analytic_signal) or (
        dispersive is not None and dispersive.sector == "conserving"
    ):
        # SEP 6's number-conserving sector is formed on analytic signals, of the
        # filtered fields and, for the polynomial, of the bare one; a Raman drive
        # beside it is the cycle-averaged one, which reads the analytic signal too.
        family = "field_analytic"
    else:
        # A field grid with no Kerr term reads only the real field: the plasma takes
        # its magnitude and the Raman drive its square, and neither wants a Hilbert
        # transform. So the real family is the right default and not a fallback.
        family = "field_real"
    return ColumnSpec(
        family=family,
        plasma=bool(responses.plasma_channels),
        raman=raman is not None,
        # An envelope grid's drive is |A|^2/2 whatever the response says, because a
        # quadratic drive cannot carry the harmonic there; the compiled flag is about
        # a field grid's drive and is read only for one.
        raman_thg=bool(raman.thg) and kind != "envelope"
        if raman is not None
        else False,
        harmonic=kerr is not None and kerr.harmonic_factor != 0.0,
        n_pol=n_pol,
        parallel=parallel,
        stash=stash and bool(responses.plasma_channels),
        n_channels=len(responses.plasma_channels),
        n_raman=len(responses.raman_terms),
        raman_drives=3
        if n_pol == 2
        and any(np.any(c.tensor_weights[:, 1]) for c in responses.raman_terms)
        else 1,
        band=bool(responses.plasma_channels)
        and getattr(responses, "plasma_integral", None) is not None
        and responses.plasma_integral.policy == "band",
    )


# --- the factories -------------------------------------------------------------


@cache
def make_combine_and_pad(
    *, contiguous: bool, noise: bool, parallel: bool = False
) -> Any:
    """Compile pass 1: stage combine, lab frame, noise and pad, in one traversal.

    Parameters
    ----------
    contiguous : bool
        Whether the crop index is the leading block of the fine spectrum, which it is
        on a field grid and is not on an envelope grid, where the negative frequencies
        sit at the far end. The contiguous form writes the pad tail as one run.
    noise : bool
        Whether the modified noise model's realisation is added at every stage.
    parallel : bool, optional
        Whether independent columns execute in parallel. The small modal path
        retains the serial default.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel.
    """
    columns = prange if parallel else range

    @no_type_check
    def combine_and_pad(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        state: Complex[np.ndarray, "n_columns n_omega"],
        stages: Complex[np.ndarray, "n_stages n_columns n_omega"],
        weights: Float[np.ndarray, " n_stages"],
        n_previous: int,
        dz: float,
        propagator: Complex[np.ndarray, "n_columns n_omega"],
        drawn: Complex[np.ndarray, "n_columns n_omega"],
        crop: Any,
        fine: Complex[np.ndarray, "n_columns n_omega_fine"],
        scale: float,
    ) -> None:
        """Combine the stages, return to the lab frame and scatter onto the fine grid.

        The whole of ``fine`` is written, not only the bins that carry data: a
        complex-to-real plan destroys its input while it executes, so a tail zeroed
        once at construction would be gone after the first stage and the pad would be
        reading whatever the transform left behind.

        Parameters
        ----------
        state : numpy.ndarray
            The state at the start of the step, complex.
        stages : numpy.ndarray
            The stage derivatives so far.
        weights : numpy.ndarray
            The tableau row for this stage.
        n_previous : int
            How many stages the row combines.
        dz : float
            The step, m.
        propagator : numpy.ndarray
            ``exp(dPhi_i)`` for this stage.
        drawn : numpy.ndarray
            The noise advanced to this stage's position, or any array when there is
            none: the branch is folded away at compile time.
        crop : numpy.ndarray
            Where each coarse bin sits in the fine spectrum, ``numpy.intp``.
        fine : numpy.ndarray
            Written with the padded fine spectrum.
        scale : float
            The ``1 / n_to`` that :func:`numpy.fft.irfft` applies and pyfftw's
            ``execute`` does not.
        """
        n_columns, n_omega = state.shape
        n_fine = fine.shape[1]
        for column in columns(n_columns):
            if contiguous:
                for j in range(n_omega, n_fine):
                    fine[column, j] = 0.0 + 0.0j
            else:
                for j in range(n_fine):
                    fine[column, j] = 0.0 + 0.0j
            for j in range(n_omega):
                accumulated = 0.0 + 0.0j
                for previous in range(n_previous):
                    accumulated = (
                        accumulated + weights[previous] * stages[previous, column, j]
                    )
                combined = propagator[column, j] * (state[column, j] + dz * accumulated)
                if noise:
                    combined = combined + drawn[column, j]
                fine[column, crop[j]] = combined * scale

    # A parallel closure is never disk-cached: numba names its gufunc
    # from an address and a per-process counter, so a cached variant can call
    # another variant's body.
    return njit(cache=not parallel, nogil=True, parallel=parallel)(combine_and_pad)


def make_finish_stage(*, contiguous: bool, absorber: bool = False) -> Any:
    """Compile passes 6 and 7: crop, prefactor, window, frame, both accumulations.

    Parameters
    ----------
    contiguous : bool
        Whether the crop index is the leading block of the fine spectrum.
    absorber : bool, optional
        Whether the in-stage temporal absorber contributes a second term. It is
        a compile-time flag rather than a zero array so that a problem without a collar
        pays neither the read nor the add.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel.
    """

    @no_type_check
    def finish_stage(  # noqa: PLR0913, PLR0917 - buffers are a kernel's parameters
        spectrum: Complex[np.ndarray, "n_columns n_omega_fine"],
        crop: Any,
        prefactor: Complex[np.ndarray, "n_columns n_omega"],
        window: Float[np.ndarray, " n_omega"],
        inverse: Complex[np.ndarray, "n_columns n_omega"],
        damping: Complex[np.ndarray, "n_columns n_omega"],
        weight: float,
        error_weight: float,
        stage: Complex[np.ndarray, "n_columns n_omega"],
        solution: Complex[np.ndarray, "n_columns n_omega"],
        error: Complex[np.ndarray, "n_columns n_omega"],
    ) -> None:
        """Crop the fine spectrum back and accumulate the stage into both sums.

        The hard band mask is deliberately absent: the reference applies it once per
        accepted step and not to the stage derivatives, and a pass 7 that added it
        would change what the embedded estimate scores.

        Parameters
        ----------
        spectrum : numpy.ndarray
            The forward transform's output on the fine grid.
        crop : numpy.ndarray
            Where each coarse bin sits in the fine spectrum, ``numpy.intp``.
        prefactor : numpy.ndarray
            The UPPE prefactor at this stage's position, times the crop scale.
        window : numpy.ndarray
            The spectral window ``grids.omega_window``.
        inverse : numpy.ndarray
            ``exp(-dPhi_i)`` for this stage.
        damping : numpy.ndarray
            The temporal absorber's own term on the coarse grid, already transformed
            and cropped, when `absorber` was compiled in; a buffer the kernel never
            reads otherwise. It sits beside the normalised polarisation rather than
            inside it because it is a rate on the field and not a polarisation, so the
            prefactor does not apply to it.
        weight : float
            The tableau's solution weight for this stage.
        error_weight : float
            The tableau's embedded-difference weight.
        stage : numpy.ndarray
            Written with the stage derivative.
        solution : numpy.ndarray
            Accumulated into.
        error : numpy.ndarray
            Accumulated into.
        """
        n_columns, n_omega = stage.shape
        for column in range(n_columns):
            for j in range(n_omega):
                index = j if contiguous else crop[j]
                normalised = prefactor[column, j] * window[j] * spectrum[column, index]
                if absorber:
                    normalised = normalised + damping[column, j]
                value = inverse[column, j] * normalised
                stage[column, j] = value
                solution[column, j] += weight * value
                error[column, j] += error_weight * value

    return njit(cache=True, nogil=True)(finish_stage)


def _make_gather(spec: ColumnSpec, *, require_magnitude: bool = True) -> Any:
    """Compile the per-sample gather of one node's transverse components.

    The sums run over the **polarisation** axis of one node, which is what the
    vector-isotropic coupling means: two components of the field at one point of the
    cross-section, never two different points. The groupings are numpy's own, so that
    the sums agree with :func:`solprop.backends.numpy_ref.passes.kerr_polarisation` to
    the last bit that its own formulation allows rather than merely resembling it.

    Parameters
    ----------
    spec : ColumnSpec
        The specialisation, whose family and component count are folded in.
    require_magnitude : bool, optional
        Form the vector real-field magnitude for plasma. Statistics retain the
        default; a propagation column without plasma can omit this unused value.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled device function.
    """
    real_samples = spec.family == "field_real"
    two = spec.n_pol == 2

    @no_type_check
    def gather(
        nodes: Num[np.ndarray, "n_rows n_to"], base: int, k: int
    ) -> tuple[Any, float, float, float, float]:
        """Read one sample of one node and form the polarisation sums.

        Parameters
        ----------
        nodes : numpy.ndarray
            The chunk's field, rows ``node * n_pol + component``.
        base : int
            The first row of this node.
        k : int
            The time sample.

        Returns
        -------
        tuple
            The first component as the Kerr term wants it -- the real sample on a
            field grid that radiates the harmonic and the complex one otherwise -- then
            ``sum_q |a_q|^2``, the real and imaginary parts of ``sum_q a_q^2``, and
            ``|E|`` of the **real** field, which is what the plasma reads.
        """
        first = nodes[base, k]
        if real_samples:
            if two:
                second = nodes[base + 1, k]
                return (
                    first,
                    first * first + second * second,
                    0.0,
                    0.0,
                    math.hypot(first, second) if require_magnitude else 0.0,
                )
            return first, first * first, 0.0, 0.0, abs(first)
        real, imag = first.real, first.imag
        if two:
            second = nodes[base + 1, k]
            other_real, other_imag = second.real, second.imag
            return (
                first,
                (real * real + imag * imag)
                + (other_real * other_real + other_imag * other_imag),
                (real * real - imag * imag)
                + (other_real * other_real - other_imag * other_imag),
                2.0 * real * imag + 2.0 * other_real * other_imag,
                math.hypot(real, other_real) if require_magnitude else 0.0,
            )
        return (
            first,
            real * real + imag * imag,
            real * real - imag * imag,
            2.0 * real * imag,
            abs(real),
        )

    return njit(inline="always")(gather)


@cache
def make_response_column(  # noqa: PLR0915 - see the kernel
    spec: ColumnSpec, *, reuse_unchanged_fraction: bool = False
) -> Any:
    """Compile passes 4 and 5: the whole nonlinear term, one traversal of ``t``.

    The kernel this backend exists for. At one node it reads the field once and forms
    the ionisation rate, the plasma column's two running sums, the loss-current
    guard, the Kerr cubic and the Raman drive from the same registers, where the
    reference makes a separate pass over the chunk for each -- two of them with
    :func:`scipy.integrate.cumulative_trapezoid`, which cannot be given a running sum.

    Every flag is a closure constant. The Raman one was an argument first and the
    runtime test in the innermost loop cost the RDW step 38 per cent (3.79 ms against
    5.24); numba folds a constant away and a problem without a term pays nothing.

    Parameters
    ----------
    spec : ColumnSpec
        The specialisation.
    reuse_unchanged_fraction : bool, optional
        Reuse the one-channel fraction when the rounded exposure is unchanged.
        Sample zero always evaluates the original scalar body. The production
        caller selects the measured component and traversal domain.
        Multiple channels keep their independent original calculation.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel.
    """
    # The Kerr sums do not consume the plasma magnitude. The sampled
    # vector Kerr column retained the otherwise unused libm hypot call.
    gather = _make_gather(spec, require_magnitude=spec.plasma or spec.n_pol != 2)
    real_samples = spec.family == "field_real"
    take_real = spec.family != "envelope"
    plasma = spec.plasma
    raman = spec.raman
    raman_thg = spec.raman_thg
    harmonic = spec.harmonic
    two = spec.n_pol == 2
    n_pol = spec.n_pol
    stash = spec.stash
    n_channels = spec.n_channels
    n_drives = spec.raman_drives
    preloaded = spec.preloaded

    @no_type_check
    def column(  # noqa: PLR0913, PLR0917, PLR0912, PLR0915 - one fused traversal
        nodes: Num[np.ndarray, "n_rows n_to"],
        out: Num[np.ndarray, "n_rows n_to"],
        drive: Float[np.ndarray, "n_nodes n_doubled"],
        window: Float[np.ndarray, " n_to"],
        table: Any,
        carrier: Complex[np.ndarray, " n_to"],
        kerr_factor: float,
        self_factor: float,
        conjugate_factor: float,
        harmonic_factor: float,
        charge_mass_ratio: float,
        ionisation_potential: float,
        density: float,
        preionisation: float,
        dt: float,
        stashed: Float[np.ndarray, "n_nodes n_population_values"],
        scratch: Any = None,
        current_out: Any = None,
    ) -> None:
        """Write non-plasma polarisation and plasma current at every chunk node.

        Parameters
        ----------
        nodes : numpy.ndarray
            The field at the chunk's nodes, rows ``node * n_pol + component``: real on
            a field grid that radiates the harmonic, and the analytic signal or the
            envelope otherwise.
        out : numpy.ndarray
            Written with the polarisation, same shape; real except on an envelope grid.
        drive : numpy.ndarray
            The Raman drive on the **doubled** grid, whose leading ``n_to`` samples are
            written here and whose tail stays zero. Any array when there is no Raman
            term.
        window : numpy.ndarray
            The temporal collar ``grids.t_window_fine``. Applied here when there is no
            Raman term and in :func:`make_raman_finish`'s kernel when there is, because
            the reference applies it to the total.
        table : RateArrays
            The compiled ionisation rate. Any value when there is no plasma.
        carrier : numpy.ndarray
            ``exp(2 i omega_0 t)`` on the fine grid. Any array without a harmonic term.
        kerr_factor : float
            ``coefficient * density``.
        self_factor, conjugate_factor, harmonic_factor : float
            The weights of the analytic form's three terms.
        charge_mass_ratio : float
            The plasma's ``q/m``.
        ionisation_potential : float
            The potential the freed electron is charged, J.
        density : float
            The density path at this position, m^-3.
        preionisation : float
            The fraction already ionised at this position.
        dt : float
            Sample spacing of the fine time grid, s.
        stashed : numpy.ndarray
            Written, when the specialisation asks for it, with each node's final
            ionised fraction in column 0 and its largest rate in column 1 -- the two
            numbers the between-planes statistics channel reduces, left behind by the
            pass that already computed them rather than found again by a second
            synthesis of every node. One row per node of *this chunk*. A
            one-row dummy when the specialisation does not ask, which numba types and
            then never writes.
        scratch : numpy.ndarray or None, optional
            Multi-channel state, shape (n_channels, n_nodes_chunk, 4), for exposure,
            previous rate, final fraction and peak rate. None on the scalar channel
            specialization, which retains its original register state.
        current_out : numpy.ndarray or None, optional
            Required for a plasma specialization: real, ungated total current in
            A/m², same shape as ``out``. Every sample of every active row is written.
            None when no plasma is compiled.
        """
        n_rows, n_to = nodes.shape
        n_nodes = n_rows // n_pol
        if n_channels > 1:
            packed_rates, rate_metadata = table
        for node in prange(n_nodes):
            base = node * n_pol
            exposure = 0.0
            last_rate = 0.0
            largest_rate = 0.0
            fraction = preionisation if n_channels == 1 else 0.0
            if n_channels > 1:
                for channel in range(n_channels):
                    scratch[channel, node, 0] = 0.0
                    scratch[channel, node, 1] = 0.0
                    scratch[channel, node, 2] = preionisation[channel]
                    scratch[channel, node, 3] = 0.0
            free_first = 0.0
            free_second = 0.0
            last_first = 0.0
            last_second = 0.0
            for k in range(n_to):
                value, modulus, square_real, square_imag, magnitude = gather(
                    nodes, base, k
                )
                # Typed from the output, not from the sample: a field grid that works
                # on the analytic signal still writes a real polarisation, so a total
                # seeded from a complex sample would not fit the buffer it goes to.
                if preloaded:
                    # The band plasma column's polarisation, already in `out`, is
                    # the first term of the total, as in the reference's order.
                    first = out[base, k]
                    second = out[base + 1, k] if two else 0.0
                elif take_real:
                    first = 0.0
                    second = 0.0
                else:
                    first = 0.0 + 0.0j
                    second = 0.0 + 0.0j
                if plasma:
                    if n_channels == 1:
                        rate = _RATE_AT(magnitude, table)
                        previous_exposure = exposure
                        if k > 0:
                            exposure += _TRAPEZOID(rate, last_rate, dt)
                        last_rate = rate
                        if stash and rate > largest_rate:
                            largest_rate = rate
                        # Reuse identical rounded arguments; the original
                        # 1-(1-p) need not retain all of the initial p's low bits.
                        if (
                            not reuse_unchanged_fraction
                            or k == 0
                            or exposure != previous_exposure
                        ):
                            fraction = _FRACTION_AT(exposure, preionisation)
                        guard = _LOSS_SCALE(
                            rate, fraction, magnitude * magnitude, ionisation_potential
                        )
                        weight = fraction * charge_mass_ratio
                    else:
                        weight = 0.0
                        work = 0.0
                        for channel in range(n_channels):
                            intervals = int(rate_metadata[channel, 5])
                            coefficients = packed_rates[
                                channel, : 4 * intervals
                            ].reshape((4, intervals))
                            channel_table = RateArrays(
                                coefficients,
                                rate_metadata[channel, 0],
                                rate_metadata[channel, 1],
                                rate_metadata[channel, 2],
                                rate_metadata[channel, 3],
                                rate_metadata[channel, 4],
                            )
                            rate = _RATE_AT(magnitude, channel_table)
                            if k > 0:
                                scratch[channel, node, 0] += _TRAPEZOID(
                                    rate, scratch[channel, node, 1], dt
                                )
                            scratch[channel, node, 1] = rate
                            fraction_channel = _FRACTION_AT(
                                scratch[channel, node, 0], preionisation[channel]
                            )
                            scratch[channel, node, 2] = fraction_channel
                            if stash and rate > scratch[channel, node, 3]:
                                scratch[channel, node, 3] = rate
                            weight += (
                                density[channel]
                                * fraction_channel
                                * charge_mass_ratio[channel]
                            )
                            work += (
                                density[channel]
                                * ionisation_potential[channel]
                                * rate
                                * (1.0 - fraction_channel)
                            )
                        guard = _LOSS_SCALE(1.0, 0.0, magnitude * magnitude, work)
                    field = value if real_samples else value.real
                    integrand = weight * field
                    if k > 0:
                        free_first += _TRAPEZOID(integrand, last_first, dt)
                    current = free_first + guard * field
                    last_first = integrand
                    current_out[base, k] = (
                        density * current if n_channels == 1 else current
                    )
                    if two:
                        other = nodes[base + 1, k]
                        field = other if real_samples else other.real
                        integrand = weight * field
                        if k > 0:
                            free_second += _TRAPEZOID(integrand, last_second, dt)
                        current = free_second + guard * field
                        last_second = integrand
                        current_out[base + 1, k] = (
                            density * current if n_channels == 1 else current
                        )
                if real_samples:
                    first = first + _KERR_REAL(value, modulus, kerr_factor)
                    if two:
                        second = second + _KERR_REAL(
                            nodes[base + 1, k], modulus, kerr_factor
                        )
                else:
                    real, imag = _KERR_ANALYTIC(
                        value.real,
                        value.imag,
                        modulus,
                        square_real,
                        square_imag,
                        kerr_factor,
                        self_factor,
                        conjugate_factor,
                    )
                    if harmonic:
                        # The reference adds this after the whole fundamental term and
                        # in this order: (factor * m) * carrier * squares * a.
                        extra = (
                            ((kerr_factor * harmonic_factor) * carrier[k])
                            * complex(square_real, square_imag)
                            * value
                        )
                        real += extra.real
                        imag += extra.imag
                    first = first + (real if take_real else complex(real, imag))
                    if two:
                        other = nodes[base + 1, k]
                        real, imag = _KERR_ANALYTIC(
                            other.real,
                            other.imag,
                            modulus,
                            square_real,
                            square_imag,
                            kerr_factor,
                            self_factor,
                            conjugate_factor,
                        )
                        if harmonic:
                            extra = (
                                ((kerr_factor * harmonic_factor) * carrier[k])
                                * complex(square_real, square_imag)
                                * other
                            )
                            real += extra.real
                            imag += extra.imag
                        second = second + (real if take_real else complex(real, imag))
                if raman:
                    if two:
                        other = nodes[base + 1, k]
                        trace, difference, cross = _RAMAN_TENSOR_DRIVE(
                            value.real, value.imag, other.real, other.imag, raman_thg
                        )
                        drive[n_drives * node, k] = trace
                        if n_drives == 3:
                            drive[n_drives * node + 1, k] = difference
                            drive[n_drives * node + 2, k] = cross
                    else:
                        drive[node, k] = _RAMAN_DRIVE(value.real, value.imag, raman_thg)
                    out[base, k] = first
                else:
                    out[base, k] = first * window[k]
                if two:
                    out[base + 1, k] = second if raman else second * window[k]
            if stash:
                # What the accepted step leaves behind for the statistics channel:
                # the fraction at the end of the window and the largest rate, per
                # node, written once after the traversal that already had both.
                if n_channels == 1:
                    stashed[node, 0] = fraction
                    stashed[node, 1] = largest_rate
                else:
                    for channel in range(n_channels):
                        stashed[node, 2 * channel] = scratch[channel, node, 2]
                        stashed[node, 2 * channel + 1] = scratch[channel, node, 3]

    # Never disk-cached: the closure captures dispatchers, whose pickled
    # form differs per process, so its cache key never matches and every process
    # would append an entry; its parallel variants would also risk another's gufunc.
    return njit(cache=False, nogil=True, parallel=spec.parallel)(column)


def band_work_rows(spec: ColumnSpec) -> int:  # noqa: ARG001 - one layout for every spec
    """Return the rows of ``n_to`` doubles the multipass plasma column needs per node.

    The raw rate, the weight ``f q/m`` (summed over channels when there are several),
    the loss guard, the free-current integrand, and the filtered integrand a scan
    reads. The total current is written directly to the caller's output. The components
    and the channels are processed one after another, so
    the count does not grow with them.

    Parameters
    ----------
    spec : ColumnSpec
        The specialisation.

    Returns
    -------
    int
        Five.

    Examples
    --------
    >>> from solprop.backends.numba_cpu.kernels import ColumnSpec, band_work_rows
    >>> band_work_rows(ColumnSpec("field_real", True, False, False, False, 2, False))
    5
    """
    return 5


@cache
def _make_band_passes() -> Any:
    """Compile the stencil the multipass plasma columns share, as inlined bodies.

    Returns
    -------
    tuple
        ``(filter_range, filtered_at)``, dispatchers compiled ``inline="always"``.
    """
    inline = njit(inline="always")

    @inline
    @no_type_check
    def filtered_at(source: Any, index: int, n_to: int, taps: Any) -> float:
        """Return one filtered sample, its window's ends closed by point reflection.

        ``c_0 y_n + sum_k c_k (y_{n-k} + y_{n+k})`` in the order of
        :meth:`solprop.nonlinear.ionisation.RunningIntegral.filter` -- the centre
        first, then each pair added before it is weighted -- with ``2 y_0 - y_m`` and
        ``2 y_{N-1} - y_{N-1-m}`` past either end. :func:`filter_range` calls it only
        between its live window's ends, writing the exact zeros outside them itself.

        Parameters
        ----------
        source : numpy.ndarray
            The integrand, one row of ``n_to`` samples.
        index : int
            The sample to filter.
        n_to : int
            The row's length.
        taps : numpy.ndarray
            ``(c_0, ..., c_K)``.

        Returns
        -------
        float
            The filtered integrand at `index`.
        """
        total = taps[0] * source[index]
        last = n_to - 1
        for k in range(1, taps.shape[0]):
            low = index - k
            high = index + k
            before = source[low] if low >= 0 else 2.0 * source[0] - source[-low]
            after = (
                source[high]
                if high <= last
                else 2.0 * source[last] - source[2 * last - high]
            )
            total = total + taps[k] * (before + after)
        return total

    @inline
    @no_type_check
    def filter_range(  # noqa: PLR0913, PLR0917 - a pass's operands
        source: Any, target: Any, low: int, high: int, taps: Any, start: int, stop: int
    ) -> None:
        """Write the filtered samples ``[low, high)`` of `source` into `target`.

        Parameters
        ----------
        source, target : numpy.ndarray
            One row each, ``n_to`` samples.
        low, high : int
            The samples written.
        taps : numpy.ndarray
            ``(c_0, ..., c_K)``.
        start, stop : int
            The first and last samples whose stencil may read a non-zero value.
        """
        n_to = source.shape[0]
        half = taps.shape[0] - 1
        zero_end = min(high, max(low, start))
        live_end = max(zero_end, min(high, stop + 1))
        for k in range(low, zero_end):
            target[k] = 0.0
        for k in range(live_end, high):
            target[k] = 0.0
        first = max(zero_end, half)
        last = min(live_end, n_to - half)
        for k in range(zero_end, min(live_end, first)):
            target[k] = filtered_at(source, k, n_to, taps)
        centre = taps[0]
        for k in range(first, last):
            total = centre * source[k]
            for j in range(1, half + 1):
                total = total + taps[j] * (source[k - j] + source[k + j])
            target[k] = total
        for k in range(max(first, last), live_end):
            target[k] = filtered_at(source, k, n_to, taps)

    return filter_range, filtered_at


@cache
def make_band_plasma_column(spec: ColumnSpec) -> Any:  # noqa: PLR0915 - the kernel
    """Compile the plasma column as whole-row passes (SEP 18's rule, SEP 25's form).

    Each of the two running integrals is a filter of its integrand followed by a
    trapezoid scan, and each integrand is known only once the scan before it is done,
    so the column runs in passes along ``t``, each a contiguous traversal of a row of
    scratch: the rate; the exposure (its filter and scan) with the fraction, the loss
    guard and the free-current weight; for each component, the free current (its
    integrand's filter and scan) with the total current. The stencil reads the row the
    pass before it finished, so there
    is no ring to keep. A pass whose whole stencil reads zeros -- before the rate
    first leaves zero, after it last does -- writes exact zeros without reading. The
    exposure is floored at zero where the fraction is formed; the loss current keeps
    the raw rate. With the trapezoid's one identity tap it is the trapezoid's column.

    The total current is written into ``out`` uncollared. The fused non-plasma
    column separately forms Kerr and Raman polarisation, with its usual collar.
    Source assembly consumes gJ and g'J; there is no third running integral.

    Parameters
    ----------
    spec : ColumnSpec
        The specialisation: field family, polarisation count, channels, stash and the
        node loop's parallelism.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel.
    """
    gather = _make_gather(spec)
    filter_range = _make_band_passes()[0]
    real_samples = spec.family == "field_real"
    n_pol = spec.n_pol
    stash = spec.stash
    n_channels = spec.n_channels
    multiple = n_channels > 1

    @no_type_check
    def band_column(  # noqa: PLR0913, PLR0917, PLR0912, PLR0915 - the column's passes
        nodes: Num[np.ndarray, "n_rows n_to"],
        out: Float[np.ndarray, "n_rows n_to"],
        table: Any,
        taps: Float[np.ndarray, " n_taps"],
        charge_mass_ratio: Any,
        ionisation_potential: Any,
        density: Any,
        preionisation: Any,
        dt: float,
        stashed: Float[np.ndarray, "n_nodes n_population_values"],
        work: Float[np.ndarray, "n_nodes n_work n_to"],
    ) -> None:
        """Write the total plasma current at every node of a chunk.

        Parameters
        ----------
        nodes : numpy.ndarray
            The field at the chunk's nodes, rows ``node * n_pol + component``.
        out : numpy.ndarray
            Written with total current density in A/m², uncollared, same shape.
        table : RateArrays or tuple
            The sole rate, or stacked coefficient storage and channel metadata.
        taps : numpy.ndarray
            The problem's taps, ``(c_0, ..., c_K)``; ``(1.0,)`` for the trapezoid.
        charge_mass_ratio, ionisation_potential, density, preionisation : object
            The channel constants and this position's scalars: floats, or arrays of
            one value per channel when there are several.
        dt : float
            Sample spacing of the fine time grid, s.
        stashed : numpy.ndarray
            Each node's final fraction and largest rate per channel, when stashed.
        work : numpy.ndarray
            Per-node scratch, :func:`band_work_rows` rows of ``n_to`` doubles.
        """
        n_rows, n_to = nodes.shape
        n_nodes = n_rows // n_pol
        half = taps.shape[0] - 1
        if multiple:
            packed_rates, rate_metadata = table
        for node in prange(n_nodes):
            base = node * n_pol
            rates = work[node, 0]
            weights = work[node, 1]
            guards = work[node, 2]
            first_buffer = work[node, 3]
            filtered_row = work[node, 4]
            for channel in range(n_channels):
                if multiple:
                    intervals = int(rate_metadata[channel, 5])
                    channel_table = RateArrays(
                        packed_rates[channel, : 4 * intervals].reshape((4, intervals)),
                        rate_metadata[channel, 0],
                        rate_metadata[channel, 1],
                        rate_metadata[channel, 2],
                        rate_metadata[channel, 3],
                        rate_metadata[channel, 4],
                    )
                    initial = preionisation[channel]
                else:
                    channel_table = table
                    initial = preionisation
                # The rate, and where it first and last leaves zero.
                lead = n_to
                tail = -1
                largest = 0.0
                for k in range(n_to):
                    rate = _RATE_AT(gather(nodes, base, k)[4], channel_table)
                    rates[k] = rate
                    if rate != 0.0:
                        tail = k
                        lead = min(lead, k)
                    largest = max(largest, rate)
                if stash:
                    stashed[node, 2 * channel + 1] = largest
                start = lead - half
                stop = tail + half
                # The exposure, floored; the fraction; the guard (one channel) or the
                # summed weight and work (several), which the last channel turns into
                # the guard.
                exposure = 0.0
                previous = 0.0
                held = 0.0
                fraction = initial
                filter_range(rates, filtered_row, 0, n_to, taps, start, stop)
                for k in range(n_to):
                    filtered = filtered_row[k]
                    if k > 0:
                        exposure += _TRAPEZOID(filtered, previous, dt)
                    previous = filtered
                    # The same rounded exposure gives the same fraction; only a
                    # change costs the exponential.
                    if k == 0 or exposure != held:
                        fraction = _FRACTION_AT(
                            exposure if exposure > 0.0 else 0.0, initial
                        )
                    held = exposure
                    if multiple:
                        if channel == 0:
                            weights[k] = 0.0
                            guards[k] = 0.0
                        weights[k] += (
                            density[channel] * fraction * charge_mass_ratio[channel]
                        )
                        guards[k] += (
                            density[channel]
                            * ionisation_potential[channel]
                            * rates[k]
                            * (1.0 - fraction)
                        )
                        if channel == n_channels - 1:
                            magnitude = gather(nodes, base, k)[4]
                            guards[k] = _LOSS_SCALE(
                                1.0, 0.0, magnitude * magnitude, guards[k]
                            )
                    else:
                        magnitude = gather(nodes, base, k)[4]
                        guards[k] = _LOSS_SCALE(
                            rates[k],
                            fraction,
                            magnitude * magnitude,
                            ionisation_potential,
                        )
                        weights[k] = fraction * charge_mass_ratio
                if stash:
                    stashed[node, 2 * channel] = fraction
            for component in range(n_pol):
                row = base + component
                # The free-current integrand.
                lead = n_to
                tail = -1
                for k in range(n_to):
                    sample = nodes[row, k]
                    field = sample if real_samples else sample.real
                    value = weights[k] * field
                    first_buffer[k] = value
                    if value != 0.0:
                        tail = k
                        lead = min(lead, k)
                start = lead - half
                stop = tail + half
                # The free current, and the current: free plus loss.
                free = 0.0
                previous = 0.0
                filter_range(first_buffer, filtered_row, 0, n_to, taps, start, stop)
                for k in range(n_to):
                    filtered = filtered_row[k]
                    if k > 0:
                        free += _TRAPEZOID(filtered, previous, dt)
                    previous = filtered
                    sample = nodes[row, k]
                    field = sample if real_samples else sample.real
                    value = free + guards[k] * field
                    out[row, k] = value if multiple else density * value

    # Never disk-cached, for the response column's two reasons.
    return njit(cache=False, nogil=True, parallel=spec.parallel)(band_column)


BAND_DIAGNOSTICS = 6
"""Rows the band statistics column appends after the channels' two each: the largest
exposure dip and rise, the largest ``|paid - freed|`` and ``freed``, and the largest
``|f E|`` in the first ``2K`` samples and anywhere, per node
(:func:`solprop.backends.numpy_ref.passes.plasma_integration_diagnostics`)."""

BAND_STATS_ROWS = 2
"""Rows of ``n_to`` doubles the band statistics column needs per node: the raw rate and
the filtered rate's exposure."""


@cache
def make_band_stats_column(spec: ColumnSpec) -> Any:  # noqa: PLR0915 - the kernel
    """Compile the plasma statistics of a plane under a running-integral rule.

    The band rule's first stage alone -- the filtered exposure, floored, and the
    fraction -- for the final fraction and the largest rate of every channel, with
    the six per-node parts of SEP 18's run diagnostics beside them, in whole-row
    passes as :func:`make_band_plasma_column` runs it: the rate, its filter, the
    exposure's scan with its increments' extremes, then the fraction, the work paid
    and the integrand's levels. Given the trapezoid's one identity tap it is the plain
    column with the diagnostics, which is how a trapezoid run that asks for them is
    answered.

    Parameters
    ----------
    spec : ColumnSpec
        The specialisation: field family, polarisation and plasma channel counts.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel.
    """
    gather = _make_gather(spec)
    filter_range = _make_band_passes()[0]
    real_samples = spec.family == "field_real"
    n_pol = spec.n_pol
    n_channels = spec.n_channels

    @no_type_check
    def band_stats_column(  # noqa: PLR0913, PLR0917, PLR0912, PLR0915 - the passes
        nodes: Num[np.ndarray, "n_rows n_to"],
        table: Any,
        taps: Float[np.ndarray, " n_taps"],
        preionisation: Any,
        dt: float,
        partials: Float[np.ndarray, "n_partials n_nodes"],
        work: Float[np.ndarray, "n_nodes n_work n_to"],
    ) -> None:
        """Write each node's final fraction, largest rate and diagnostic parts.

        Parameters
        ----------
        nodes : numpy.ndarray
            The field at every node, rows ``node * n_pol + component``.
        table : RateArrays or tuple
            The sole rate, or stacked coefficient storage and channel metadata.
        taps : numpy.ndarray
            The rule's taps, ``(c_0, ..., c_K)``.
        preionisation : float or numpy.ndarray
            Initial fraction, scalar for one channel or one value per channel.
        dt : float
            Sample spacing of the fine time grid, s.
        partials : numpy.ndarray
            Rows ``2c`` and ``2c + 1``: channel c's final fraction and largest rate;
            then :data:`BAND_DIAGNOSTICS` rows, each the largest over the channels.
        work : numpy.ndarray
            Per-node scratch, :data:`BAND_STATS_ROWS` rows of ``n_to`` doubles.
        """
        n_rows, n_to = nodes.shape
        n_nodes = n_rows // n_pol
        half = taps.shape[0] - 1
        reach = 2 * half
        diagnostics = 2 * n_channels
        if n_channels > 1:
            packed_rates, metadata = table
        for node in prange(n_nodes):
            base = node * n_pol
            rates = work[node, 0]
            exposed = work[node, 1]
            for row in range(BAND_DIAGNOSTICS):
                partials[diagnostics + row, node] = 0.0
            for channel in range(n_channels):
                if n_channels == 1:
                    channel_table = table
                    initial = preionisation
                else:
                    count = int(metadata[channel, 5])
                    channel_table = RateArrays(
                        packed_rates[channel, : 4 * count].reshape((4, count)),
                        metadata[channel, 0],
                        metadata[channel, 1],
                        metadata[channel, 2],
                        metadata[channel, 3],
                        metadata[channel, 4],
                    )
                    initial = preionisation[channel]
                largest = 0.0
                for k in range(n_to):
                    rate = _RATE_AT(gather(nodes, base, k)[4], channel_table)
                    rates[k] = rate
                    largest = max(largest, rate)
                filter_range(rates, exposed, 0, n_to, taps, 0, n_to)
                # The exposure's scan, with its increments' extremes.
                exposure = 0.0
                dip = 0.0
                rise = 0.0
                last_filtered = 0.0
                for k in range(n_to):
                    filtered = exposed[k]
                    if k > 0:
                        step = _TRAPEZOID(filtered, last_filtered, dt)
                        exposure += step
                        dip = max(dip, -step)
                        rise = max(rise, step)
                    last_filtered = filtered
                    exposed[k] = exposure
                fraction = initial
                first_fraction = initial
                paid = 0.0
                last_paid = 0.0
                early = 0.0
                peak = 0.0
                for k in range(n_to):
                    exposure = exposed[k]
                    fraction = _FRACTION_AT(
                        exposure if exposure > 0.0 else 0.0, initial
                    )
                    if k == 0:
                        first_fraction = fraction
                    owed = rates[k] * (1.0 - fraction)
                    if k > 0:
                        paid += _TRAPEZOID(owed, last_paid, dt)
                    last_paid = owed
                    for component in range(n_pol):
                        sample = nodes[base + component, k]
                        level = abs(
                            fraction * (sample if real_samples else sample.real)
                        )
                        peak = max(peak, level)
                        if k < reach:
                            early = max(early, level)
                partials[2 * channel, node] = fraction
                partials[2 * channel + 1, node] = largest
                freed = fraction - first_fraction
                for row, part in enumerate(
                    (dip, rise, abs(paid - freed), freed, early, peak)
                ):
                    partials[diagnostics + row, node] = max(
                        partials[diagnostics + row, node], part
                    )

    # Never disk-cached, for the response column's two reasons.
    return njit(cache=False, nogil=True, parallel=spec.parallel)(band_stats_column)


def make_raman_finish(spec: ColumnSpec) -> Any:
    """Compile the delayed Raman term and the temporal collar, in one traversal.

    ``P_R = eps0 rho E(t) (h * D)(t)``, with the convolution already done on the
    doubled grid. The only pass the Raman term costs beyond its transform pair: the
    drive was written by the response column and the total is already in ``out``.

    Parameters
    ----------
    spec : ColumnSpec
        Field family, physical component count and Raman history count. A field
        grid multiplies the delayed response matrix by the **real** field; an
        envelope grid multiplies it by the complex envelope.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel.
    """
    real_field = spec.family != "envelope"
    n_pol = spec.n_pol
    n_drives = spec.raman_drives

    @no_type_check
    def raman_finish(
        out: Num[np.ndarray, "n_nodes n_to"],
        nodes: Num[np.ndarray, "n_nodes n_to"],
        convolved: Float[np.ndarray, "n_nodes n_to"],
        window: Float[np.ndarray, " n_to"],
        factor: float,
    ) -> None:
        """Add the delayed polarisation to the chunk's total and window it.

        Parameters
        ----------
        out : numpy.ndarray
            The chunk's polarisation so far, added to in place.
        nodes : numpy.ndarray
            The chunk's field, whose real part a field grid multiplies by.
        convolved : numpy.ndarray
            The leading ``n_to`` samples of the convolution on the doubled grid.
        window : numpy.ndarray
            The temporal collar ``grids.t_window_fine``.
        factor : float
            ``eps0 * density``.
        """
        n_rows, n_to = out.shape
        for node in range(n_rows // n_pol):
            for k in range(n_to):
                base = node * n_pol
                row = node * n_drives
                sample = nodes[base, k]
                field = sample.real if real_field else sample
                if n_pol == 1:
                    out[base, k] = _RAMAN_FINISH(
                        out[base, k], field, convolved[row, k], factor, window[k]
                    )
                else:
                    sample_y = nodes[base + 1, k]
                    other = sample_y.real if real_field else sample_y
                    trace = convolved[row, k]
                    difference = convolved[row + 1, k] if n_drives == 3 else 0.0
                    cross = convolved[row + 2, k] if n_drives == 3 else 0.0
                    out[base, k] = _RAMAN_TENSOR_FINISH(
                        out[base, k],
                        field,
                        other,
                        trace + difference,
                        cross,
                        factor,
                        window[k],
                    )
                    out[base + 1, k] = _RAMAN_TENSOR_FINISH(
                        out[base + 1, k],
                        other,
                        field,
                        trace - difference,
                        cross,
                        factor,
                        window[k],
                    )

    return njit(cache=True, nogil=True)(raman_finish)


@cache
def make_stats_column(spec: ColumnSpec) -> Any:
    """Compile the plasma statistics of a plane: one traversal for both of them.

    ``electron_density`` and ``peak_ionisation_rate`` are read at the node of largest
    ionised fraction, so both need the rate and the exposure at every node -- which is
    the response column's first half and nothing else. The reference re-synthesises
    **every node at once** and then runs the multipass scan, which was measured at two
    thirds of a whole CtC step on an x86-64 server; a run with many save planes cannot
    afford that.

    Parameters
    ----------
    spec : ColumnSpec
        The specialisation: field family, polarisation and plasma channel counts.

    Returns
    -------
    numba.core.dispatcher.Dispatcher
        The compiled kernel.
    """
    gather = _make_gather(spec)
    n_pol = spec.n_pol
    n_channels = spec.n_channels

    @no_type_check
    def stats_column(
        nodes: Num[np.ndarray, "n_rows n_to"],
        table: Any,
        preionisation: Any,
        dt: float,
        partials: Float[np.ndarray, "n_partials n_nodes"],
    ) -> None:
        """Return the final ionised fraction and the peak rate at every node.

        Parameters
        ----------
        nodes : numpy.ndarray
            The field at every node, rows ``node * n_pol + component``.
        table : RateArrays or tuple
            The sole rate, or stacked coefficient storage and channel metadata.
        preionisation : float or numpy.ndarray
            Initial fraction, scalar for one channel or one value per channel.
        dt : float
            Sample spacing of the fine time grid, s.
        partials : numpy.ndarray
            Final fraction and largest rate in rows ``2*c`` and ``2*c + 1`` for
            channel c, one column per node.
        """
        n_rows, n_to = nodes.shape
        n_nodes = n_rows // n_pol
        if n_channels > 1:
            packed_rates, metadata = table
            for node in prange(n_nodes):
                base = node * n_pol
                for channel in range(n_channels):
                    count = int(metadata[channel, 5])
                    coefficients = packed_rates[channel, : 4 * count].reshape(
                        (4, count)
                    )
                    channel_table = RateArrays(
                        coefficients,
                        metadata[channel, 0],
                        metadata[channel, 1],
                        metadata[channel, 2],
                        metadata[channel, 3],
                        metadata[channel, 4],
                    )
                    exposure = 0.0
                    last_rate = 0.0
                    largest = 0.0
                    fraction = preionisation[channel]
                    for k in range(n_to):
                        magnitude = gather(nodes, base, k)[4]
                        rate = _RATE_AT(magnitude, channel_table)
                        if k > 0:
                            exposure += _TRAPEZOID(rate, last_rate, dt)
                        last_rate = rate
                        fraction = _FRACTION_AT(exposure, preionisation[channel])
                        largest = max(largest, rate)
                    partials[2 * channel, node] = fraction
                    partials[2 * channel + 1, node] = largest
        else:
            for node in prange(n_nodes):
                base = node * n_pol
                exposure = 0.0
                last_rate = 0.0
                largest = 0.0
                fraction = preionisation
                for k in range(n_to):
                    magnitude = gather(nodes, base, k)[4]
                    rate = _RATE_AT(magnitude, table)
                    if k > 0:
                        exposure += _TRAPEZOID(rate, last_rate, dt)
                    last_rate = rate
                    fraction = _FRACTION_AT(exposure, preionisation)
                    largest = max(largest, rate)
                partials[0, node] = fraction
                partials[1, node] = largest

    # Never disk-cached, for the response column's two reasons.
    return njit(cache=False, nogil=True, parallel=spec.parallel)(stats_column)
