# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Retain the original launch's precision and exact physical amplitude scales."""

import math
from dataclasses import dataclass
from weakref import WeakKeyDictionary

from solprop.translate.precision import PrecisionName, launch_peak, state_scale
from solprop.translate.problem import PropagationProblem

__all__ = ["ArithmeticTransfer", "lower_arithmetic"]


@dataclass(frozen=True)
class ArithmeticTransfer:
    """Declare state precision and the physical units of prepared numerical arrays.

    Attributes
    ----------
    name : {'double', 'single'}
        Shared state and source precision; wide arithmetic is a separate execution
        choice, not part of this scientific record.
    state_exponent, field_exponent : int
        Stored modal and reconstructed electric amplitudes represent physical
        amplitudes divided by these powers of two. Both exponents are zero in
        binary64. Binary32 folds map coefficients at preparation, so the exponents
        coincide and come from the original translated launch.
    """

    name: PrecisionName = "double"
    state_exponent: int = 0
    field_exponent: int = 0


_ARITHMETIC: WeakKeyDictionary[PropagationProblem, ArithmeticTransfer] = (
    WeakKeyDictionary()
)


def lower_arithmetic(problem: PropagationProblem) -> ArithmeticTransfer:
    """Derive physical scaling from the immutable launch using the shared SEP 33 rule.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        Scientific descriptor with the original launch in internal spectral units.
        A replacement starting field or snapshot is deliberately not an input.

    Returns
    -------
    ArithmeticTransfer
        Binary64 unit scales or the original launch's exact binary32 power of two.
        An identically zero launch also has unit scales.
    """
    if not problem.precision.single:
        return ArithmeticTransfer()
    if problem in _ARITHMETIC:
        return _ARITHMETIC[problem]
    scale = state_scale(launch_peak(problem.initial.field0, problem.grid_kind))
    exponent = math.frexp(scale)[1] - 1
    result = ArithmeticTransfer("single", exponent, exponent)
    _ARITHMETIC[problem] = result
    return result
