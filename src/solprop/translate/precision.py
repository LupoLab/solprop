# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The arithmetic a run is held in, and the range rules of single precision (SEP 33).

A run is ``"double"`` (binary64, the default and the only arithmetic of ``numba_cpu``
and ``native``) or ``"single"``: the propagated state, the stage values, every transform
and the responses in IEEE binary32, with the controller, every host scalar and the
linear propagator's exponent in binary64
(``docs/dev/seps/sep-0033-single-precision-mlx.md`` section 4.2, ADR-052). This module
holds what every backend that admits single precision shares, so that they compute the
same exponents and refuse the same things:

* :class:`Precision`, the record a :class:`~solprop.translate.problem.KernelProblem`
  carries, with named fields so that a later mixed mode (a binary32 source beside a
  binary64 state) is a third public value rather than a schema redesign;
* :func:`admit_precision`, the shared admission of the defaults and refusals;
* the range helpers of SEP 33 section 4.3: :func:`block_exponent` and :func:`narrow`
  for every table narrowed at upload, :func:`state_scale` for the state's exact
  power-of-two scale, :class:`Coefficient` and :func:`combine` for the scalar factors
  combined on the host, and :func:`pairwise_sum`, :func:`prefix_sum` and
  :func:`scaled_square_sum`, the reduction orders the single-precision oracle defines
  (SEP 33, Q5).

Nothing here changes a binary64 run: the translation layer's tables stay binary64, and a
backend narrows them on upload.
"""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np

from solprop import fourier
from solprop.exceptions import TranslationError
from solprop.fourier import GridKind

__all__ = [
    "DOUBLE",
    "SINGLE",
    "SINGLE_PRECISION_BACKENDS",
    "SINGLE_RTOL_FLOOR",
    "Coefficient",
    "Narrowed",
    "Precision",
    "PrecisionName",
    "admit_precision",
    "block_exponent",
    "combine",
    "launch_peak",
    "narrow",
    "pairwise_sum",
    "precision_named",
    "prefix_sum",
    "scaled_square_sum",
    "state_scale",
    "study_admission",
]

PrecisionName = Literal["double", "single"]
"""The public values of ``precision``: binary64 or binary32 arithmetic."""

SINGLE_RTOL_FLOOR = 1e-6
"""The smallest relative tolerance a single-precision run admits (SEP 33, Q1).

A local error target below the per-step rounding of binary32, about ten times its unit
roundoff, buys steps without accuracy. An explicit tolerance below the floor is refused,
and the floor is every interface's default under single precision.
"""

SINGLE_PRECISION_BACKENDS: tuple[str, ...] = ("numpy_ref", "mlx")
"""The backends that admit ``precision="single"``.

``numpy_ref`` is the single-precision oracle (SEP 33 phase 1) and ``mlx`` the Apple-GPU
backend (phase 2). ``numba_cpu`` and ``native`` refuse it by name until a later
proposal adds their own modes.
"""


@dataclass(frozen=True, slots=True)
class Precision:
    """The arithmetic of a run's propagated state and of its source evaluation.

    The public strings map to equal fields: ``"double"`` is binary64 throughout and
    ``"single"`` binary32 throughout, with the host scalars, the controller and the
    propagator's exponent in binary64 either way. A mixed mode would set the two fields
    apart and is not admitted.

    Attributes
    ----------
    state : {'double', 'single'}
        The propagated state, the stage values and the error estimate's array.
    source : {'double', 'single'}
        The transforms and the responses of a source evaluation.

    Examples
    --------
    >>> from solprop.translate.precision import Precision, precision_named
    >>> precision_named("single")
    Precision(state='single', source='single')
    >>> Precision().name, precision_named("single").single
    ('double', True)
    """

    state: PrecisionName = "double"
    source: PrecisionName = "double"

    def __post_init__(self) -> None:
        """Refuse an unknown name or a mixed record.

        Raises
        ------
        TranslationError
            If either field is not a public value, or the two differ.
        """
        names = get_args(PrecisionName)
        if self.state not in names or self.source not in names:
            msg = (
                f"precision fields must each be one of {names}, got state="
                f"{self.state!r}, source={self.source!r}"
            )
            raise TranslationError(msg)
        if self.state != self.source:
            msg = (
                "a mixed precision (a binary32 source beside a binary64 state, or the "
                "reverse) is not admitted; use precision='double' or 'single'"
            )
            raise TranslationError(msg)

    @property
    def name(self) -> PrecisionName:
        """Return the public value this record stands for.

        Returns
        -------
        {'double', 'single'}
            The shared value of the two fields.
        """
        return self.state

    @property
    def single(self) -> bool:
        """Return whether the state is held in binary32.

        Returns
        -------
        bool
            True under ``precision="single"``.
        """
        return self.state == "single"

    @property
    def complex_dtype(self) -> np.dtype:
        """Return the dtype of the state and of every complex pass.

        Returns
        -------
        numpy.dtype
            ``complex64`` under single precision, ``complex128`` otherwise.
        """
        return np.dtype(np.complex64 if self.single else np.complex128)

    @property
    def real_dtype(self) -> np.dtype:
        """Return the dtype of every real pass.

        Returns
        -------
        numpy.dtype
            ``float32`` under single precision, ``float64`` otherwise.
        """
        return np.dtype(np.float32 if self.single else np.float64)


DOUBLE = Precision()
"""Binary64 throughout: the default."""

SINGLE = Precision("single", "single")
"""Binary32 for the state, the stages and the source evaluation."""


def precision_named(name: PrecisionName | Precision) -> Precision:
    """Return the record for a public precision value.

    Parameters
    ----------
    name : {'double', 'single'} or Precision
        The public value, or a record, which is returned unchanged.

    Returns
    -------
    Precision
        :data:`DOUBLE` or :data:`SINGLE`.

    Raises
    ------
    TranslationError
        If `name` is not a public value.

    Examples
    --------
    >>> from solprop.translate.precision import precision_named
    >>> precision_named("double").name
    'double'
    """
    if isinstance(name, Precision):
        return name
    if name == "double":
        return DOUBLE
    if name == "single":
        return SINGLE
    msg = f"precision must be 'double' or 'single', got {name!r}"
    raise TranslationError(msg)


_STUDY: ContextVar[bool] = ContextVar("solprop_precision_study", default=False)


@contextmanager
def study_admission() -> Iterator[None]:
    """Admit, inside the block, what the survival study measures and the public refuses.

    The private, test-only route of SEP 33 section 5.2: the study runs the Adams
    integrator under single precision through it, to measure whether Adams can be
    admitted (Q2). No public interface sets it.

    Yields
    ------
    None
        Nothing; the admission is relaxed until the block exits.
    """
    token = _STUDY.set(True)
    try:
        yield
    finally:
        _STUDY.reset(token)


def admit_precision(
    precision: Precision,
    *,
    backend: str,
    adams: bool,
    rtol: float | None,
    schedule: bool,
) -> None:
    """Apply the single-precision refusals every interface and backend shares.

    SEP 33 section 5.2. A binary64 run passes unchanged. A single-precision run is
    refused on a backend that does not admit it (:data:`SINGLE_PRECISION_BACKENDS`),
    with the Adams integrator until the survival study admits it, and with an adaptive
    tolerance below :data:`SINGLE_RTOL_FLOOR`. The routes and responses no backend runs
    in binary32 are refused by
    :meth:`~solprop.translate.problem.KernelProblem.validate`, where they are known.

    Parameters
    ----------
    precision : Precision
        The run's precision.
    backend : str
        The concrete registry name the run executes on.
    adams : bool
        Whether the integrator is an Adams method.
    rtol : float or None
        The controller's relative tolerance, or ``None`` where the controller has none.
    schedule : bool
        Whether a fixed schedule is replayed, which reads no tolerance.

    Raises
    ------
    TranslationError
        Naming the refusal and what to do instead.

    Examples
    --------
    >>> from solprop.translate.precision import SINGLE, admit_precision
    >>> admit_precision(SINGLE, backend="numpy_ref", adams=False, rtol=1e-6,
    ...                 schedule=False) is None
    True
    """
    if not precision.single:
        return
    if backend not in SINGLE_PRECISION_BACKENDS:
        msg = (
            f"backend {backend!r} computes in binary64 only and refuses "
            f"precision='single'; the backends that admit it are "
            f"{', '.join(SINGLE_PRECISION_BACKENDS)} (SEP 33)"
        )
        raise TranslationError(msg)
    if adams and not _STUDY.get():
        msg = (
            "the Adams integrator (Multistep) is not admitted under precision='single' "
            "until its single-precision survival is established; use "
            "integrator=RungeKutta(), the single-precision default"
        )
        raise TranslationError(msg)
    if not schedule and rtol is not None and rtol < SINGLE_RTOL_FLOOR:
        msg = (
            f"rtol={rtol:g} is below the single-precision floor {SINGLE_RTOL_FLOOR:g}: "
            "a tighter local error target buys steps without accuracy in binary32; "
            "take rtol at least the floor, or precision='double'"
        )
        raise TranslationError(msg)


def block_exponent(values: np.ndarray | float) -> int:
    """Return the power of two that brings a buffer's largest magnitude into [1, 2).

    SEP 33 section 4.3, rule 1: a buffer narrowed to binary32 is stored as
    ``values * 2**-e``, which is `values` rounded once, with no range loss for entries
    within ``2**-126`` of its largest.

    Parameters
    ----------
    values : numpy.ndarray or float
        The binary64 buffer, real or complex, finite.

    Returns
    -------
    int
        ``e`` with ``max |values| * 2**-e`` in ``[1, 2)``, or 0 for an all-zero buffer.

    Raises
    ------
    TranslationError
        If the buffer holds a non-finite value.

    Examples
    --------
    >>> from solprop.translate.precision import block_exponent
    >>> [block_exponent(x) for x in (1.0, 3.0, 0.25, 0.0)]
    [0, 1, -2, 0]
    """
    largest = float(np.max(np.abs(values), initial=0.0))
    if not math.isfinite(largest):
        raise TranslationError("a buffer narrowed to binary32 must be finite")
    if largest == 0.0:
        return 0
    return math.frexp(largest)[1] - 1


@dataclass(frozen=True, slots=True)
class Narrowed:
    """A binary64 buffer narrowed to binary32 with its own power-of-two exponent.

    The buffer stands for ``values * 2**exponent``; the exponent is a host integer and
    is folded into the binary64 coefficient the buffer is multiplied by
    (:func:`combine`), never applied to the binary32 values.

    Attributes
    ----------
    values : numpy.ndarray
        The scaled buffer, ``float32`` or ``complex64``, read-only, with its largest
        magnitude in ``[1, 2)`` (or zero).
    exponent : int
        The block exponent, :func:`block_exponent` of the binary64 buffer.
    """

    values: np.ndarray
    exponent: int

    def widened(self) -> np.ndarray:
        """Return the buffer the narrowed one stands for, in binary64.

        Returns
        -------
        numpy.ndarray
            ``values * 2**exponent``, exactly, as ``float64`` or ``complex128``.
        """
        wide = self.values.astype(np.result_type(self.values.dtype, np.float64))
        return _ldexp(wide, self.exponent)


def narrow(values: np.ndarray) -> Narrowed:
    """Narrow a binary64 buffer to binary32 with its block exponent.

    Multiplying by a power of two is exact in binary64 for these magnitudes, so the
    narrowed values are the buffer rounded once.

    Parameters
    ----------
    values : numpy.ndarray
        The binary64 buffer, real or complex, finite.

    Returns
    -------
    Narrowed
        The read-only ``float32`` or ``complex64`` buffer and its exponent.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.precision import narrow
    >>> kernel = narrow(np.array([5.2e-50, 1.0e-50]))
    >>> kernel.values.dtype, kernel.exponent, bool(np.all(kernel.values > 0))
    (dtype('float32'), -164, True)
    """
    exponent = block_exponent(values)
    source = np.asarray(values)
    dtype = np.complex64 if np.iscomplexobj(source) else np.float32
    scaled = source * math.ldexp(1.0, -exponent)
    result = np.ascontiguousarray(scaled, dtype=dtype)
    result.setflags(write=False)
    return Narrowed(result, exponent)


def launch_peak(field0: np.ndarray, kind: GridKind) -> float:
    """Return the peak temporal amplitude of a launch, in the state's own units.

    The quantity the state scale is chosen from (SEP 33 section 4.3, rule 2): every
    column of the canonical state transformed to time on its own grid, without
    padding, in binary64.

    Parameters
    ----------
    field0 : numpy.ndarray
        The launch, complex, in the canonical layout ``(..., n_omega)``.
    kind : {'field', 'envelope'}
        The spectral layout.

    Returns
    -------
    float
        The largest temporal magnitude, non-negative.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.precision import launch_peak
    >>> launch_peak(np.full((1, 1, 1, 4), 2.0 + 0.0j), "envelope")
    2.0
    """
    values = np.asarray(field0, dtype=np.complex128)
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(fourier.inverse(values, kind=kind)), initial=0.0))


def state_scale(peak: float) -> float:
    """Return the exact power-of-two scale of a single-precision run's state.

    SEP 33 section 4.3, rule 2: the state is held as ``a / s`` with ``s = 2**k`` chosen
    from the launch so that the launch's peak temporal amplitude over ``s`` lies in
    ``[1, 2)``; ``s = 1`` for an identically zero launch. Being a function of the
    problem, not of the state at ``set_initial``, it is the same for a run, its restarts
    and its resumes.

    Parameters
    ----------
    peak : float
        The launch's peak temporal amplitude, in the state's units, non-negative.

    Returns
    -------
    float
        ``2**k``.

    Examples
    --------
    >>> from solprop.translate.precision import state_scale
    >>> state_scale(3.0e4), state_scale(0.0)
    (16384.0, 1.0)
    """
    return math.ldexp(1.0, block_exponent(peak))


@dataclass(frozen=True, slots=True)
class Coefficient:
    """A scalar factor combined on the host as a binary64 mantissa and an exponent.

    SEP 33 section 4.3, rule 3: the density, an amplitude, a prefactor's constant, the
    state scale's powers and the buffers' exponents are combined in binary64 into one
    coefficient per product, kept as a mantissa in ``[0.5, 1)`` (or zero) and a
    separate integer exponent, so that the combination itself cannot leave binary64's
    range whatever the exponents.

    Attributes
    ----------
    mantissa : float
        The binary64 mantissa, ``0`` or of magnitude in ``[0.5, 1)``.
    exponent : int
        The power of two it multiplies.
    """

    mantissa: float
    exponent: int

    @property
    def value(self) -> float:
        """Return the coefficient as one binary64 number.

        Returns
        -------
        float
            ``mantissa * 2**exponent``.

        Raises
        ------
        TranslationError
            If the coefficient is outside binary64's range.
        """
        try:
            return math.ldexp(self.mantissa, self.exponent)
        except OverflowError as error:
            raise TranslationError(
                f"a combined coefficient 2**{self.exponent} overflows binary64"
            ) from error

    def binary32(self) -> float:
        """Return the coefficient rounded once to binary32, as a Python float.

        Returns
        -------
        float
            The value of ``numpy.float32(value)``.

        Raises
        ------
        TranslationError
            If the coefficient is not a finite normal binary32 number (or zero): it
            would overflow or lose its precision, so the pass that applies it has not
            kept its intermediates in range.
        """
        value = self.value
        narrowed = float(np.float32(value))
        tiny = float(np.finfo(np.float32).tiny)
        if not math.isfinite(narrowed) or (value != 0.0 and abs(narrowed) < tiny):
            msg = (
                f"a combined coefficient {value:.3e} leaves binary32's normal range; "
                "the pass applying it must apply it where its intermediate is largest"
            )
            raise TranslationError(msg)
        return narrowed


def combine(*factors: float, exponent: int = 0) -> Coefficient:
    """Combine scalar factors and power-of-two exponents in binary64.

    Each factor is split into its mantissa and exponent first, so the product of the
    mantissas stays in ``[2**-n, 1)`` for ``n`` factors and the exponents add as
    integers: a density of ``2.5e25`` times a kernel exponent of ``-164`` never forms
    the intermediate ``2.5e25 * 2**-164`` out of range.

    Parameters
    ----------
    *factors : float
        Finite binary64 factors.
    exponent : int, optional
        A further power of two, such as a buffer's block exponent or a power of the
        state scale's exponent.

    Returns
    -------
    Coefficient
        The product, normalised.

    Examples
    --------
    >>> from solprop.translate.precision import combine
    >>> c = combine(2.5e25, 8.854e-12, exponent=-164)
    >>> abs(c.value - 2.5e25 * 8.854e-12 * 2.0**-164) < 1e-60
    True
    """
    mantissa = 1.0
    total = exponent
    for factor in factors:
        part, power = math.frexp(float(factor))
        mantissa *= part
        total += power
        if mantissa == 0.0:
            return Coefficient(0.0, 0)
        mantissa, power = math.frexp(mantissa)
        total += power
    return Coefficient(mantissa, total)


def pairwise_sum(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Sum along an axis as a balanced binary tree, in the values' own precision.

    The reduction order of the single-precision oracle (SEP 33 section 4.2, Q5): the
    axis is padded with zeros to a power of two and halved repeatedly, each level one
    vectorised addition. The error of each sum is bounded by ``gamma_d`` times the sum
    of magnitudes, ``d = ceil(log2 n)`` the depth of the tree.

    Parameters
    ----------
    values : numpy.ndarray
        The terms; the result has the same dtype.
    axis : int, optional
        The axis summed over.

    Returns
    -------
    numpy.ndarray
        The sums, with `axis` removed.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.precision import pairwise_sum
    >>> pairwise_sum(np.arange(5, dtype=np.float32)).item()
    10.0
    """
    moved = np.moveaxis(np.asarray(values), axis, -1)
    n = moved.shape[-1]
    if n == 0:
        return np.zeros(moved.shape[:-1], dtype=moved.dtype)
    width = 1 << (n - 1).bit_length()
    level = np.zeros((*moved.shape[:-1], width), dtype=moved.dtype)
    level[..., :n] = moved
    while level.shape[-1] > 1:
        level = level[..., 0::2] + level[..., 1::2]
    return level[..., 0]


def prefix_sum(values: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return the running sums along an axis by doubling, in the values' own precision.

    The running-integral order of the single-precision oracle (SEP 33 section 4.2,
    Q5): ``ceil(log2 n)`` vectorised levels, at level ``k`` each entry adding the entry
    ``2**k`` before it (a Hillis-Steele scan). Every prefix is then a sum of depth at
    most ``ceil(log2 n)``, bounded by ``gamma_d`` times its sum of magnitudes, where a
    sequential binary32 sum of ``10**4`` terms could lose ``10**4 u``.

    Parameters
    ----------
    values : numpy.ndarray
        The increments; the result has the same dtype and shape.
    axis : int, optional
        The axis scanned along.

    Returns
    -------
    numpy.ndarray
        ``out[..., i] = sum(values[..., :i + 1])`` along `axis`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.precision import prefix_sum
    >>> prefix_sum(np.arange(5, dtype=np.float32)).tolist()
    [0.0, 1.0, 3.0, 6.0, 10.0]
    """
    scan = np.array(np.moveaxis(np.asarray(values), axis, -1), copy=True)
    n = scan.shape[-1]
    shift = 1
    while shift < n:
        shifted = scan[..., :-shift].copy()
        scan[..., shift:] += shifted
        shift *= 2
    return np.moveaxis(scan, -1, axis)


def scaled_square_sum(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Return ``sum(w |v|**2)`` without squaring a value out of range.

    SEP 33 section 4.3, rule 6: the values are divided by a power of two chosen from
    their largest magnitude before squaring, as LAPACK's ``nrm2`` does, the squares
    summed in the values' precision by :func:`pairwise_sum`, and the scale restored on
    the host in binary64.

    Parameters
    ----------
    values : numpy.ndarray
        The values, real or complex, of any precision.
    weights : numpy.ndarray or None, optional
        Non-negative weights broadcastable against `values`, in the values' real
        precision; None weighs every value by one.

    Returns
    -------
    float
        The weighted sum of squares, binary64.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.precision import scaled_square_sum
    >>> big = np.full(4, 3.0e30, dtype=np.float32)
    >>> total = scaled_square_sum(big)
    >>> abs(total / (4 * float(np.float32(3.0e30)) ** 2) - 1.0) < 1e-6
    True
    """
    array = np.asarray(values)
    exponent = block_exponent(array)
    scaled = _ldexp(array, -exponent)
    squares = np.real(scaled) * np.real(scaled)
    if np.iscomplexobj(scaled):
        squares = squares + np.imag(scaled) * np.imag(scaled)
    if weights is not None:
        squares = squares * weights
    total = pairwise_sum(squares.reshape(-1))
    return math.ldexp(float(total), 2 * exponent)


def _ldexp(values: np.ndarray, exponent: int) -> np.ndarray:
    """Multiply by ``2**exponent`` exactly, in the values' own precision.

    Parameters
    ----------
    values : numpy.ndarray
        Real or complex values.
    exponent : int
        The power of two.

    Returns
    -------
    numpy.ndarray
        The scaled values, same dtype; exact unless a value over- or underflows.
    """
    if np.iscomplexobj(values):
        result = np.empty_like(values)
        result.real = np.ldexp(np.real(values), exponent)
        result.imag = np.ldexp(np.imag(values), exponent)
        return result
    return np.ldexp(values, exponent)
