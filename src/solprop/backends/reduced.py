# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The host-side range work every reduced-precision backend shares (SEP 36 6.3).

Single and mixed precision evaluate the source in binary32 on the state narrowed at the
launch's exact power-of-two scale, with every table narrowed once and every scalar
factor combined in binary64 (``docs/dev/seps/archive/sep-0033-single-precision-mlx.md``
section 4.3). ``numpy_ref`` (the oracles), ``mlx`` and ``native`` take these from here,
so that the three compute the same exponents, round the same tables and refuse the
same ranges:

* :class:`LaunchScale`, the state's (or the narrowed input's) scale ``s = 2**k``;
* :func:`narrowed_table`, a binary64 table rounded once to binary32, refused where
  narrowing it directly would lose an entry a block exponent keeps;
* :func:`narrowed_kernel`, the Raman kernel's spectrum narrowed with its exponent kept;
* :func:`kerr_factor` and :func:`raman_factor`, the cubic terms' coefficients with
  their homogeneity factors, combined on the host and rounded once.
"""

import math
from dataclasses import dataclass

import numpy as np
from jaxtyping import Complex, Inexact

from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.translate.precision import (
    block_exponent,
    combine,
    launch_peak,
    state_scale,
)
from solprop.translate.problem import KernelProblem
from solprop.translate.responses import KerrCoefficients

__all__ = [
    "LaunchScale",
    "check_range",
    "kerr_factor",
    "narrowed_kernel",
    "narrowed_table",
    "raman_factor",
]

_TINY = float(np.finfo(np.float32).tiny)
_HUGE = float(np.finfo(np.float32).max)


@dataclass(frozen=True, slots=True)
class LaunchScale:
    """The exact power-of-two scale ``s = 2**k`` of a reduced-precision run.

    Chosen from the launch, so that the launch's peak temporal amplitude over ``s``
    lies in ``[1, 2)`` (:func:`solprop.translate.precision.state_scale`): a function of
    the problem, the same for a run, its restarts and its resumes.

    Attributes
    ----------
    scale : float
        ``s``.
    exponent : int
        ``k``.

    Examples
    --------
    >>> from solprop.backends.reduced import LaunchScale
    >>> LaunchScale.of_peak(3.0e4)
    LaunchScale(scale=16384.0, exponent=14)
    """

    scale: float
    exponent: int

    @classmethod
    def of_peak(cls, peak: float) -> "LaunchScale":  # noqa: UP037 - bound after the class body
        """Return the scale for a launch's peak temporal amplitude.

        Parameters
        ----------
        peak : float
            The peak, in the state's units, non-negative.

        Returns
        -------
        LaunchScale
            The scale and its exponent.
        """
        scale = state_scale(peak)
        return cls(scale, math.frexp(scale)[1] - 1)

    @classmethod
    def of(cls, problem: KernelProblem) -> "LaunchScale":  # noqa: UP037 - bound after the class body
        """Return the scale of a problem's launch.

        Parameters
        ----------
        problem : solprop.translate.problem.KernelProblem
            The problem.

        Returns
        -------
        LaunchScale
            The scale of :func:`solprop.translate.precision.launch_peak`.
        """
        return cls.of_peak(launch_peak(problem.initial.field0, problem.grid_kind))


def check_range(source: np.ndarray, what: str) -> None:
    """Refuse narrowing an array that would lose range a block exponent keeps.

    Narrowing directly keeps every entry the block-exponent form keeps when the largest
    magnitude is at least one, and when no entry falls below binary32's smallest normal
    number; an entry flushed then lies below ``2**-126`` of the largest, inside SEP 33's
    underflow budget.

    Parameters
    ----------
    source : numpy.ndarray
        The binary64 array.
    what : str
        What it is, for the message.

    Raises
    ------
    BackendError
        If its largest magnitude overflows binary32, or is below one beside an entry
        below binary32's smallest normal number.
    """
    magnitude = np.abs(source)
    largest = float(np.max(magnitude, initial=0.0))
    if not math.isfinite(largest) or largest >= _HUGE:
        msg = f"{what} reaches {largest:.3e}, beyond binary32's range"
        raise BackendError(msg)
    if largest < 1.0 and bool(np.any((magnitude > 0.0) & (magnitude < _TINY))):
        msg = (
            f"{what} holds entries below binary32's smallest normal number beside a "
            f"largest of {largest:.3e}: its pass must carry its block exponent, and "
            "none does (SEP 33 section 4.3)"
        )
        raise BackendError(msg)


def narrowed_table(values: Inexact[np.ndarray, "*shape"], what: str) -> np.ndarray:
    """Round a binary64 table once to binary32, keeping its range.

    Parameters
    ----------
    values : numpy.ndarray
        The binary64 table, real or complex.
    what : str
        What it is, for a refusal.

    Returns
    -------
    numpy.ndarray
        The contiguous ``float32`` or ``complex64`` table.

    Raises
    ------
    BackendError
        As :func:`check_range`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.backends.reduced import narrowed_table
    >>> narrowed_table(np.array([3.0, 1e-41]), "probe").dtype
    dtype('float32')
    """
    source = np.asarray(values)
    check_range(source, what)
    dtype = np.complex64 if np.iscomplexobj(source) else np.float32
    return np.ascontiguousarray(source, dtype=dtype)


def narrowed_kernel(
    spectrum: Complex[np.ndarray, "*shape"],
) -> tuple[Complex[np.ndarray, "*shape"], int]:
    """Narrow a Raman kernel's spectrum with its block exponent kept.

    A molecular kernel's spectrum of order ``1e-50`` is all zero in binary32; scaled by
    ``2**-e`` first it keeps every bin, and ``e`` is folded into the Raman coefficient
    (:func:`raman_factor`).

    Parameters
    ----------
    spectrum : numpy.ndarray
        The binary64 spectrum.

    Returns
    -------
    narrowed : numpy.ndarray
        ``spectrum * 2**-e`` rounded once, complex64.
    exponent : int
        ``e``, :func:`solprop.translate.precision.block_exponent` of the spectrum.
    """
    exponent = block_exponent(spectrum)
    narrowed = (spectrum * math.ldexp(1.0, -exponent)).astype(np.complex64)
    return narrowed, exponent


def kerr_factor(kerr: KerrCoefficients, density: float, scale_exponent: int) -> float:
    """Return the Kerr term's coefficient on the scaled field, rounded once.

    ``coefficient * density * s**2``: the cubic term's homogeneity factor (SEP 33
    section 4.3, rule 4), combined in binary64 with the exponents added as integers.

    Parameters
    ----------
    kerr : solprop.translate.responses.KerrCoefficients
        The compiled coefficients.
    density : float
        The Kerr density at the stage, m^-3 (or one).
    scale_exponent : int
        ``k`` with ``s = 2**k``.

    Returns
    -------
    float
        The binary32 value, as a Python float.
    """
    return combine(kerr.coefficient, density, exponent=2 * scale_exponent).binary32()


def raman_factor(density: float, scale_exponent: int, kernel_exponent: int) -> float:
    """Return the Raman term's coefficient on the scaled field, rounded once.

    ``eps0 * density * s**2 * 2**e``, ``e`` the narrowed kernel's block exponent.

    Parameters
    ----------
    density : float
        The Raman density at the stage, m^-3 (or one for a mixture's rows).
    scale_exponent : int
        ``k`` with ``s = 2**k``.
    kernel_exponent : int
        The kernel spectrum's block exponent.

    Returns
    -------
    float
        The binary32 value, as a Python float.
    """
    return combine(
        EPS0, density, exponent=2 * scale_exponent + kernel_exponent
    ).binary32()
