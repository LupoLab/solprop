# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The shared refusals of plain argument conditions, in their one wording."""

import math

import numpy as np
import pytest

from runtime_types import unchecked
from solprop._validation import choice, finite, positive, shape
from solprop.exceptions import ModeError, PulseError, ToolsError


@pytest.mark.parametrize("value", [2, 2.5, np.float64(3.0), True])
def test_positive_returns_the_number_as_a_float(value):
    result = positive(value, "width", ToolsError)
    assert type(result) is float
    assert result == float(value)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf, "1.0", b"1"])
def test_positive_refuses_by_name_in_the_callers_class(value):
    # A string reaches the check only past the annotation, as in production.
    with pytest.raises(ToolsError, match=r"width must be finite and positive, got "):
        unchecked(positive)(value, "width", ToolsError)


def test_positive_carries_the_parameter_and_can_refuse_a_boolean():
    with pytest.raises(PulseError, match="energy must be finite and positive") as hit:
        positive(-1.0, "energy", PulseError, parameter="energy")
    assert hit.value.parameter == "energy"
    with pytest.raises(ModeError, match="cut must be finite and positive, got True"):
        positive(True, "cut", ModeError, reject_bool=True)


def test_finite_returns_a_float_and_refuses_what_is_not_finite():
    assert finite(-2, "z", ToolsError) == -2.0
    for value in (math.nan, -math.inf, "0"):
        with pytest.raises(ToolsError, match="z must be finite, got "):
            unchecked(finite)(value, "z", ToolsError)


def test_choice_lists_every_allowed_value():
    choice("b", "kind", ("a", "b"), ToolsError)
    with pytest.raises(ToolsError, match="kind must be 'a', 'b' or 'c', got 'd'"):
        choice("d", "kind", ("a", "b", "c"), ToolsError)
    with pytest.raises(ToolsError, match="kind must be 'a', got 'd'"):
        choice("d", "kind", ("a",), ToolsError)


def test_shape_refuses_another_shape():
    shape((2, 3), (2, 3), "table", ModeError)
    expected = r"table must have shape \(2, 3\), got \(3, 2\)"
    with pytest.raises(ModeError, match=expected):
        shape((3, 2), (2, 3), "table", ModeError)
