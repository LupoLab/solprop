# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep Julia's physical scale tied to the scientific launch, not execution choices."""

import gc
import pickle
import weakref
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from backends.julia.cases import cpu_case
from solprop.backends.julia import _precision
from solprop.backends.julia._descriptors import lower_modal
from solprop.backends.julia._precision import ArithmeticTransfer, lower_arithmetic
from solprop.translate.precision import SINGLE


def test_binary64_scales_need_no_launch_transform(monkeypatch):
    case = cpu_case()

    def forbidden(*_):
        pytest.fail("binary64 preparation must not transform the launch for scaling")

    monkeypatch.setattr(_precision, "launch_peak", forbidden)
    assert lower_arithmetic(case.problem) == ArithmeticTransfer()
    assert lower_modal(case.problem).arithmetic == ArithmeticTransfer()


@pytest.mark.parametrize("shift", [-500, -126, 0, 80, 500])
def test_binary32_launch_scale_is_covariant_under_exact_powers(shift):
    problem = replace(cpu_case().problem, precision=SINGLE)
    baseline = lower_arithmetic(problem)
    field = np.asarray(problem.initial.field0, dtype=np.complex128)
    changed = np.ldexp(field.real, shift) + 1j * np.ldexp(field.imag, shift)
    scaled = replace(problem, initial=replace(problem.initial, field0=changed))
    result = lower_arithmetic(scaled)
    # Exact binary scaling preserves the FFT's represented operations on these
    # normal-range fixtures. The launch exponent must shift by that same integer.
    assert result == ArithmeticTransfer(
        "single", baseline.state_exponent + shift, baseline.field_exponent + shift
    )
    assert pickle.loads(pickle.dumps(result)) == result
    with pytest.raises(FrozenInstanceError):
        # Exercise the forbidden mutation without bypassing the frozen setter.
        result.state_exponent = 0  # pyright: ignore[reportAttributeAccessIssue]


def test_zero_launch_retains_unit_scales_and_no_mutable_state():
    problem = replace(cpu_case().problem, precision=SINGLE)
    zero = replace(
        problem,
        initial=replace(problem.initial, field0=np.zeros_like(problem.initial.field0)),
    )
    assert lower_arithmetic(zero) == ArithmeticTransfer("single", 0, 0)
    assert set(ArithmeticTransfer.__dataclass_fields__) == {
        "name",
        "state_exponent",
        "field_exponent",
    }


def test_launch_arithmetic_cache_reuses_lowering_without_retaining_the_problem(
    monkeypatch,
):
    cache = weakref.WeakKeyDictionary()
    monkeypatch.setattr(_precision, "_ARITHMETIC", cache)
    problem = replace(cpu_case().problem, precision=SINGLE)
    first = lower_arithmetic(problem)
    reference = weakref.ref(problem)

    def forbidden(*_):
        pytest.fail("an immutable problem's launch must be lowered only once")

    monkeypatch.setattr(_precision, "launch_peak", forbidden)
    assert lower_arithmetic(problem) is first
    assert len(cache) == 1
    del problem
    gc.collect()
    assert reference() is None
    assert not cache
