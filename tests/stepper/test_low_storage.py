# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage RK4IP specification (SEP 24 section 5)."""

import dataclasses

import numpy as np
import pytest

from solprop.boundaries import STABILITY_RADIUS
from solprop.exceptions import StepperError
from solprop.stepper import LowStorageRungeKutta
from solprop.stepper.low_storage import LOW_STORAGE_REAL_INTERVAL


def stability_function(x):
    """RK4's ``R(x) = 1 + x + x^2/2 + x^3/6 + x^4/24``."""
    return 1.0 + x + x**2 / 2.0 + x**3 / 6.0 + x**4 / 24.0


def test_the_specification_names_and_counts_itself():
    method = LowStorageRungeKutta()
    assert repr(method) == "LowStorageRungeKutta(scheme='rk4ip')"
    assert method.name == "low-storage-rk4ip"
    assert (method.order, method.evaluations) == (4, 4)
    assert method.real_stability_interval == LOW_STORAGE_REAL_INTERVAL == 2.785
    # Frozen, and equal by value: a lockstep batch compares members' integrators.
    assert method == LowStorageRungeKutta(scheme="rk4ip")
    with pytest.raises(dataclasses.FrozenInstanceError):
        method.scheme = "rk4ip"  # type: ignore[misc]


def test_the_interval_is_rk4_s_on_the_negative_real_axis():
    # Rounded down: |R| <= 1 on [-2.785, 0] and > 1 just beyond (the root is -2.7853).
    inside = np.linspace(-LOW_STORAGE_REAL_INTERVAL, 0.0, 20001)
    assert np.max(np.abs(stability_function(inside))) <= 1.0
    assert abs(stability_function(-2.786)) > 1.0
    # Tighter than DOPRI5's, so the march admits shorter steps beside a collar.
    assert LOW_STORAGE_REAL_INTERVAL < STABILITY_RADIUS


def test_an_unknown_scheme_is_refused_where_annotations_are_not_checked():
    # The Literal annotation refuses it first under the test session's type checker,
    # so the instance is built without its initialiser to reach the check itself.
    method = object.__new__(LowStorageRungeKutta)
    object.__setattr__(method, "scheme", "rk4")
    with pytest.raises(StepperError, match="scheme must be 'rk4ip'"):
        method.__post_init__()
