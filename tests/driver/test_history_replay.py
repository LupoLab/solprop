# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Replaying a run's own step history with ``FixedSchedule.from_history``.

An adaptive run is not held to the in-stage collar's stable step, because an empty
collar produces no error, so a default run's history is refused as a schedule. The
refusal names the two remedies, and each replays: the split collar in both runs, or
``dz_max`` in the discovery run just under the printed limit. Each run is a third-order
soliton over 0.2 m of fibre on ``numpy_ref``, under a second.
"""

import re

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.driver import prop_gnlse
from solprop.exceptions import TranslationError
from solprop.stepper import FixedSchedule, Multistep, RungeKutta

pytestmark = pytest.mark.medium

INTEGRATORS = {"runge_kutta": RungeKutta, "multistep": Multistep}


def soliton(integrator: str, **changes):
    """Run the guide's third-order soliton with one integrator."""
    return prop_gnlse(
        0.11,
        0.2,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=2e-12,
        width=100e-15,
        peak_power=9 * 11.83e-27 / (0.11 * 1e-26),
        shape="sech",
        n_save=3,
        noise=None,
        backend="numpy_ref",
        progress="none",
        integrator=INTEGRATORS[integrator](),
        **changes,
    )


def steps_of(run) -> FixedSchedule:
    """Return a run's accepted steps as a schedule."""
    return FixedSchedule.from_history(
        np.asarray(run["step_history/z"]),
        np.asarray(run["step_history/h"]),
        np.asarray(run["step_history/accepted"]),
    )


def refused_limit(integrator: str, **changes) -> float:
    """Return the limit a default run's replayed history is refused at, in m."""
    schedule = steps_of(soliton(integrator, **changes))
    with pytest.raises(TranslationError) as refusal:
        soliton(integrator, schedule=schedule, **changes)
    message = str(refusal.value)
    assert "FixedSchedule.from_history" in message
    assert 'absorber_scheme="split_step"' in message
    assert "dz_max just under" in message
    found = re.search(r"only up to ([0-9.e+-]+) m", message)
    assert found is not None
    limit = float(found.group(1))
    assert schedule.max_step > limit
    return limit


def assert_replays(integrator: str, **changes) -> None:
    """Assert that a run's history replays, and a replay's replay agrees with it."""
    schedule = steps_of(soliton(integrator, **changes))
    first = soliton(integrator, schedule=schedule, **changes)
    second = soliton(integrator, schedule=steps_of(first), **changes)
    # The positions are the schedule's: a structural contract, so exact.
    np.testing.assert_array_equal(steps_of(first).steps, schedule.steps)
    np.testing.assert_array_equal(steps_of(second).steps, schedule.steps)
    one, other = np.asarray(first["field"])[-1], np.asarray(second["field"])[-1]
    # Two replays of one schedule are one discretisation, so they differ by rounding
    # alone: a budget of a few hundred ulp of the peak over the run's 50-1500 steps.
    assert_allclose(other, one, rtol=0.0, atol=1e-12 * float(np.max(np.abs(one))))


@pytest.mark.parametrize("integrator", list(INTEGRATORS))
def test_a_default_runs_history_is_refused_naming_both_remedies(integrator):
    refused_limit(integrator)


@pytest.mark.parametrize("integrator", list(INTEGRATORS))
def test_the_split_collar_in_both_runs_replays(integrator):
    assert_replays(integrator, absorber_scheme="split_step")


@pytest.mark.parametrize("integrator", list(INTEGRATORS))
def test_dz_max_under_the_printed_limit_replays(integrator):
    cap = 0.99 * refused_limit(integrator)
    assert_replays(integrator, dz_max=cap)
