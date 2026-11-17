# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The simulation driver: physics objects in, a written run out.

Layer D of ``docs/dev/architecture.md`` section 3. The package re-exports ``Simulation``
(:mod:`~solprop.driver.simulation`, a run that translates once and writes a resumable
file), ``SimulationSettings`` (:mod:`~solprop.driver.settings`), ``run_problem``
(:mod:`~solprop.driver.run`, the loop over save planes), ``SimulationRunner``
(:mod:`~solprop.driver.runner`), the keyword interfaces ``prop_gnlse``,
``prop_capillary``, ``prop_waveguide``, ``prop_free`` and ``prop_planar``
(:mod:`~solprop.driver.interface`), and the TDSE presets and consistency check
(:mod:`~solprop.driver.tdse`, :mod:`~solprop.driver.tdse_consistency`). The
constructors' keyword routes are :mod:`solprop.driver.routes`; scans and their executors
are :mod:`solprop.driver.scan` and :mod:`solprop.driver.executors`; the session's
service loop, inputs, defaults, metadata, resume, summary and estimate have a module
each. :mod:`solprop.driver.scaffold` stays as the independent fixed-step integrator the
backend tests cross-check against. The manual page is ``docs/guide/setup/running.md``.
"""

from solprop.driver.interface import (
    prop_capillary,
    prop_free,
    prop_gnlse,
    prop_planar,
    prop_waveguide,
)
from solprop.driver.run import run_problem
from solprop.driver.runner import SimulationRunner
from solprop.driver.settings import SimulationSettings
from solprop.driver.simulation import Simulation
from solprop.driver.tdse import HaccPreset, hacc_basis, hacc_preset
from solprop.driver.tdse_consistency import ConsistencyResult, check_consistency

__all__ = [
    "ConsistencyResult",
    "HaccPreset",
    "Simulation",
    "SimulationRunner",
    "SimulationSettings",
    "check_consistency",
    "hacc_basis",
    "hacc_preset",
    "prop_capillary",
    "prop_free",
    "prop_gnlse",
    "prop_planar",
    "prop_waveguide",
    "run_problem",
]
