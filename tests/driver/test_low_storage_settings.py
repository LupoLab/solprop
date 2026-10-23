# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage march's admission in the driver's settings (SEP 24 section 5)."""

import pytest

from solprop.backends.registry import resolve_name
from solprop.driver import SimulationSettings
from solprop.driver.simulation import LOW_STORAGE_BACKENDS
from solprop.exceptions import DriverError
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, PIController

SCHEDULE = FixedSchedule.uniform(0.0, 1.0, 0.25)


def settings(**overrides):
    """Return settings for the march, any field overridden."""
    arguments = {
        "controller": PIController(rtol=1e-6, atol=1e-12),
        "save_planes": [1.0],
        "dz_max": 0.25,
        "dz_init": 0.25,
        "integrator": LowStorageRungeKutta(),
        "schedule": SCHEDULE,
    } | overrides
    return SimulationSettings(**arguments)


def test_auto_resolves_as_for_every_integrator():
    # Every registered backend executes the march (SEP 24 phase 3), so "auto" takes
    # the registry's choice: ``native`` when it is installed, otherwise numba_cpu.
    assert settings().backend == resolve_name("auto")
    for name in LOW_STORAGE_BACKENDS:
        assert settings(backend=name).backend == name


def test_the_march_needs_a_schedule_and_lands_on_it():
    with pytest.raises(DriverError, match=r"replays a prescribed schedule.*uniform"):
        settings(schedule=None)
    with pytest.raises(DriverError, match="drop interpolate_planes"):
        settings(interpolate_planes=True)


def test_the_tdse_response_is_refused_at_the_settings():
    # Its explicit execution resources need an Adams integrator, before translation.
    from solprop.nonlinear.tdse import TDSEExecution  # noqa: PLC0415

    with pytest.raises(DriverError, match=r"tdse_execution requires.*Multistep"):
        settings(backend="numpy_ref", tdse_execution=TDSEExecution())
