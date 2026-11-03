# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Replaying a run's own step history with ``FixedSchedule.from_history``.

An adaptive run is not held to a fixed schedule's collar limits, so its history can be
refused as a schedule. Each refusal a replayed history can reach names its remedies,
and each remedy replays:

- the temporal collar's stable step: the split collar in both runs, or ``dz_max`` at
  the printed limit in the discovery run (a third-order soliton over 0.2 m of fibre,
  under a second);
- a requested spatial collar's stable step, and its fixed-schedule damping bound:
  ``dz_max`` at the printed limit (a linear radial run over a centimetre of argon, a
  few seconds).

The printed limit is rounded down, so ``dz_max`` equal to it is the test.
"""

import re
from decimal import Decimal

import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings, prop_gnlse
from solprop.exceptions import TranslationError
from solprop.grid import FieldGrid, RadialGrid
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from solprop.translate.problem import _printed_down

pytestmark = pytest.mark.medium

INTEGRATORS = {"runge_kutta": RungeKutta, "multistep": Multistep}
NUMBER = r"([0-9.]+e[+-][0-9]+)"


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


def radial(integrator: str, **changes):
    """Run a linear beam through a centimetre of argon inside a requested collar.

    The transverse monitor is off, and the first step long, so that the adaptive steps
    grow past the fixed-schedule damping bound within the run, as they would on a
    longer one; the monitor itself caps an adaptive step at a sixteenth of the length.
    """
    length = 0.01
    grid = FieldGrid.from_limits(length, 800e-9, (500e-9, 1.6e-6), 250e-15)
    space = RadialGrid(106.07e-6, 230, collar_fraction=2.6)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9)
    settings = {
        "integrator": INTEGRATORS[integrator](),
        "controller": PIController(rtol=1e-6, atol=0.0),
        "save_planes": [length],
        "dz_max": length,
        "dz_init": 1e-3,
        "backend": "numpy_ref",
        **changes,
    }
    simulation = Simulation.from_radial(
        grid,
        space,
        SpatialMedium("Ar", 1.0),
        [],
        SpatialPulse(pulse, 35.36e-6),
        settings=SimulationSettings(**settings),
        noise=None,
        spatial_absorber=SpatialAbsorber(),
        boundary_check=None,
    )
    return simulation.run(progress="none")


ROUTES = {"soliton": soliton, "radial": radial}


def steps_of(run) -> FixedSchedule:
    """Return a run's accepted steps as a schedule."""
    return FixedSchedule.from_history(
        np.asarray(run["step_history/z"]),
        np.asarray(run["step_history/h"]),
        np.asarray(run["step_history/accepted"]),
    )


def refusal_of(route: str, integrator: str, **changes) -> str:
    """Return the message a default run's replayed history is refused with."""
    run = ROUTES[route]
    schedule = steps_of(run(integrator, **changes))
    with pytest.raises(TranslationError) as refusal:
        run(integrator, schedule=schedule, **changes)
    message = str(refusal.value)
    assert "FixedSchedule.from_history" in message
    assert "dz_max at or under" in message
    return message


def printed_limit(message: str, pattern: str) -> float:
    """Return the limit a refusal prints, checking that it is the one it advises."""
    found = re.search(pattern.format(number=NUMBER), message)
    assert found is not None, message
    printed = found.group(1)
    assert f"dz_max at or under {printed} m" in message
    return float(printed)


def assert_replays(route: str, integrator: str, **changes) -> None:
    """Assert that a run's history replays, and a replay's replay agrees with it."""
    run = ROUTES[route]
    schedule = steps_of(run(integrator, **changes))
    first = run(integrator, schedule=schedule, **changes)
    second = run(integrator, schedule=steps_of(first), **changes)
    # The positions are the schedule's: a structural contract, so exact.
    np.testing.assert_array_equal(steps_of(first).steps, schedule.steps)
    np.testing.assert_array_equal(steps_of(second).steps, schedule.steps)
    one, other = np.asarray(first["field"])[-1], np.asarray(second["field"])[-1]
    # Two replays of one schedule are one discretisation, so they differ by rounding
    # alone: a budget of a few hundred ulp of the peak over the runs' 10-1500 steps.
    assert_allclose(other, one, rtol=0.0, atol=1e-12 * float(np.max(np.abs(one))))


@pytest.mark.parametrize("integrator", list(INTEGRATORS))
def test_a_default_runs_history_is_refused_naming_both_remedies(integrator):
    message = refusal_of("soliton", integrator)
    assert 'absorber_scheme="split_step" to both' in message
    assert "run the discovery with that integrator" in message


@pytest.mark.parametrize("integrator", list(INTEGRATORS))
def test_the_split_collar_in_both_runs_replays(integrator):
    assert_replays("soliton", integrator, absorber_scheme="split_step")


@pytest.mark.parametrize("integrator", list(INTEGRATORS))
def test_dz_max_at_the_printed_temporal_limit_replays(integrator):
    message = refusal_of("soliton", integrator)
    cap = printed_limit(message, r"only up to {number} m")
    assert_replays("soliton", integrator, dz_max=cap)


def test_the_spatial_collar_stability_refusal_names_the_dz_max_remedy():
    # Multistep's stability interval puts the collar's stable step below the steps
    # the adaptive run takes; the split scheme does not lift a spatial limit.
    message = refusal_of("radial", "multistep")
    assert "spatial collar stability limit" in message
    assert 'absorber_scheme="split_step" to both' not in message
    assert "run the discovery with that integrator" in message
    cap = printed_limit(message, r"stability limit {number} m")
    assert_replays("radial", "multistep", dz_max=cap)


def test_the_collar_damping_refusal_names_the_dz_max_remedy():
    # Under Runge-Kutta the adaptive steps stay inside the collar's stable step but
    # grow past the damping bound a fixed schedule is held to, which is the
    # integrator's no more than the crossing limit is.
    message = refusal_of("radial", "runge_kutta")
    assert "the damping bound" in message
    assert 'absorber_scheme="split_step" to both' not in message
    assert "integrator" not in message.split("FixedSchedule.from_history")[1]
    cap = printed_limit(message, r"collar, {number} m")
    assert_replays("radial", "runge_kutta", dz_max=cap)


@pytest.mark.parametrize(
    ("value", "printed"),
    [
        # Rounding to nearest would print 5.239e-03, a step still above the limit.
        (5.2386e-3, "5.238e-03"),
        (9.99996e-4, "9.999e-04"),
        (1.0, "1.000e+00"),
        # The double nearest 3.48e-5 lies just below it, so its floor is 3.479e-05.
        (3.48e-5, "3.479e-05"),
    ],
)
def test_a_printed_limit_rounds_down(value, printed):
    assert _printed_down(value) == printed
    assert Decimal(printed) <= Decimal(value)
    # Never further down than one unit in the fourth digit.
    assert float(printed) > value * (1 - 1e-3)
