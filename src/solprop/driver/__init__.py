# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The simulation driver: physics objects in, a written run out.

Layer D of ``docs/dev/architecture.md`` section 3.
:class:`~solprop.driver.simulation.Simulation` holds a run and translates it once;
:func:`~solprop.driver.run.run_problem` is the loop over save planes;
:func:`~solprop.driver.interface.prop_gnlse` is the keyword interface for a GNLSE fibre.
:mod:`solprop.driver.scaffold` stays as the independent fixed-step integrator the
backend tests cross-check against.

The manual page is ``docs/guide/setup/running.md``.
"""

from solprop.driver.interface import (
    prop_capillary,
    prop_free,
    prop_gnlse,
    prop_planar,
    prop_radial,
    prop_waveguide,
)
from solprop.driver.run import (
    LogReporter,
    ProgressReporter,
    plane_from_observation,
    run_problem,
)
from solprop.driver.simulation import (
    Simulation,
    SimulationRunner,
    SimulationSettings,
    metadata_for,
)
from solprop.driver.tdse import HaccPreset, hacc_basis, hacc_preset
from solprop.driver.tdse_consistency import ConsistencyResult, check_consistency

__all__ = [
    "ConsistencyResult",
    "HaccPreset",
    "LogReporter",
    "ProgressReporter",
    "Simulation",
    "SimulationRunner",
    "SimulationSettings",
    "check_consistency",
    "hacc_basis",
    "hacc_preset",
    "metadata_for",
    "plane_from_observation",
    "prop_capillary",
    "prop_free",
    "prop_gnlse",
    "prop_planar",
    "prop_radial",
    "prop_waveguide",
    "run_problem",
]
