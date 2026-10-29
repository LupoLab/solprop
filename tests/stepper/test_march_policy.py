# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The Adams march's policy record (SEP 39 section 4.2)."""

import dataclasses
import re
from pathlib import Path

import pytest

from solprop.exceptions import StepperError
from solprop.stepper.multistep import (
    CARRIER_PHASE_CAP,
    COLLAR_STABILITY,
    ORDER_HYSTERESIS,
    PEC_REAL_INTERVAL,
    REJECTION_GROWTH,
    RESTART_AFTER,
    MarchPolicy,
    Multistep,
    march_policy,
)
from solprop.stepper.schedule import LANDING_ULPS

CONTRACT = Path(__file__).parents[2] / "docs" / "dev" / "kernel-contract.md"

SETTINGS = {
    "dz_min": 1e-9,
    "dz_max": 1e-2,
    "stretch": 1.1,
    "z_end": 1.0,
    "save_planes": [0.25, 0.5, 1.0],
    "hard_kinks": [0.4],
    "breakpoints": [0.0, 0.3, 0.6, 1.0],
    "interpolate": False,
}


def test_the_contract_lists_the_record_s_fields():
    text = CONTRACT.read_text(encoding="utf-8")
    documented = set(re.findall(r"`MarchPolicy\.([a-z_]+)`", text))
    assert documented == {entry.name for entry in dataclasses.fields(MarchPolicy)}


def test_the_record_carries_the_method_settings_and_constants():
    method = Multistep(order=4, max_growth=1.5, max_calls=100)
    policy = march_policy(method, collar_rate=12.5, carrier=True, **SETTINGS)
    assert (policy.order, policy.max_growth, policy.max_calls) == (4, 1.5, 100)
    assert policy.rejection_growth == REJECTION_GROWTH == 0.9
    assert (policy.hysteresis, policy.restart_after) == (
        ORDER_HYSTERESIS,
        RESTART_AFTER,
    )
    assert (policy.collar_stability, policy.real_intervals) == (
        COLLAR_STABILITY,
        PEC_REAL_INTERVAL,
    )
    assert (policy.carrier, policy.carrier_cap) == (True, CARRIER_PHASE_CAP)
    assert policy.landing_ulps == LANDING_ULPS
    assert policy.collar_rate == 12.5
    assert policy.save_planes == (0.25, 0.5, 1.0)
    assert policy.hard_kinks == (0.4,)


def test_the_interval_cap_and_dense_planes_shape_the_landings():
    uncapped = march_policy(Multistep(), **SETTINGS)
    assert uncapped.breakpoints == (0.0, 1.0)
    capped = march_policy(Multistep(interval_cap=True), **SETTINGS)
    assert capped.breakpoints == (0.0, 0.3, 0.6, 1.0)
    dense = march_policy(Multistep(), **{**SETTINGS, "interpolate": True})
    assert dense.save_planes == ()
    assert dense.interpolate


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("order", 7),
        ("real_intervals", (1.0,)),
        ("min_factor", 1.0),
        ("rejection_growth", 1.5),
        ("hysteresis", 0.5),
        ("restart_after", 0),
        ("collar_rate", -1.0),
        ("max_calls", 0),
        ("dz_min", 1.0),
        ("stretch", 0.9),
        ("breakpoints", (0.0,)),
        ("landing_ulps", 0),
        ("quiet_interval", 0.0),
    ],
)
def test_a_record_out_of_range_is_refused_by_its_field(field, value):
    policy = march_policy(Multistep(), **SETTINGS)
    with pytest.raises(StepperError, match=f"policy has {field} = "):
        dataclasses.replace(policy, **{field: value})
