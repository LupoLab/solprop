# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A moving guide through the entry point on ``native`` (SEP 43).

From the retired ``test_numba_moving_guide.py``: the statistic a user reads, beside the
field the surviving parity already holds.
"""

import dataclasses
import warnings
from typing import cast

import numpy as np
import pytest

pytest.importorskip("solprop_native")


from coupled_flow_cases import propagate
from modal_provider_cases import grin_grid
from moving_guide_cases import (
    GRIN_ENERGY,
    GRIN_PULSE,
    grin_kinked,
    grin_taper,
    kinked_radius,
    moving_parity_bound,
    ten_step_simulation,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.output.memory import MemoryOutput


@pytest.mark.slow
def test_a_moving_guide_runs_end_to_end_on_native_as_on_the_reference():
    """``Simulation.from_waveguide(provider=..., backend="native")``, ten steps.

    The entry point's physical energy at the exit against the reference's run, within
    the ten-step bound of the state (an energy is a quadratic form of it, so twice). The
    retired bound was the flow's ten steps; a moving guide's own adds the maps'
    combination (:func:`moving_parity_bound`), as native's field parity takes it.
    """
    case = grin_kinked()
    expected = propagate(case, NumpyRefBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    scale = float(np.max(np.abs(case.problem.initial.field0)))
    bound = moving_parity_bound(case, expected, linear) / scale
    energies = {}
    for backend in ("numpy_ref", "native"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            simulation = ten_step_simulation(
                grin_taper(kinked_radius(), kerr=True),
                grin_grid(),
                **(GRIN_PULSE | {"energy": GRIN_ENERGY}),
            )
            simulation.settings = dataclasses.replace(
                simulation.settings, absorber=None, backend=backend, boundary_check=None
            )
            planes = cast("MemoryOutput", simulation.run(progress="none")).planes()
        energies[backend] = float(np.sum(planes[-1].stats["physical_energy"]))
    assert energies["numpy_ref"] > 0.0
    assert energies["native"] == pytest.approx(energies["numpy_ref"], rel=2.0 * bound)
