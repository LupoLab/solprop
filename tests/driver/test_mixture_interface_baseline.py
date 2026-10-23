# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The launch and log-weight conventions of a capillary's translated problem.

Until SEP 3 Phase 4b this module also held the byte witness of every translated array
of four legacy capillary calls (``tests/data/mixture_interface_baseline.json`` and the
``cases`` of its Linux x86-64 twin), recorded on solprop's former registry materials.
Those inputs are deleted and the flow's density now needs optmat's equation of state
and transport, which a caller's species cannot supply, so the bytes cannot be
reproduced and the witness was retired. What remains is its independent check: the
power-weighted launch and the exact log-weight term, against the dispersion-only
problem, on optmat's materials.
"""

import dataclasses
from contextlib import nullcontext

import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.exceptions import GasFlowWarning
from solprop.linear import Frame
from solprop.translate.linop import _dispersion_phi, delta_phi, power_weight_at

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)


@pytest.mark.parametrize(
    "gas, plasma, raman",
    [
        ("He", False, False),
        ("Ar", True, False),
        ("N2", True, True),
        ("O2", True, False),
    ],
)
def test_the_launch_and_the_log_weight_reconstruct_the_dispersion_problem(
    gas, plasma, raman
):
    # A millimetre carrying a bar of difference is turbulent for all but helium.
    turbulent = (
        nullcontext() if gas == "He" else pytest.warns(GasFlowWarning, match="Reynolds")
    )
    with turbulent:
        simulation = Simulation.from_capillary(
            30e-6,
            1e-3,
            gas,
            (2.0, 1.0),
            vacuum_end="floor",
            lambda0=800e-9,
            wavelength_limits=(400e-9, 1600e-9),
            trange=400e-15,
            fwhm=20e-15,
            energy=1e-9,
            n_r=8,
            n_save=2,
            plasma=plasma,
            raman=raman,
            thg=False,
            loss=False,
            cladding=1.45,
        )
        problem = simulation.translate()
    assert simulation.settings.absorber is not None
    dispersion = _dispersion_phi(
        simulation.modes,
        simulation.grid,
        Frame.comoving(simulation.grid),
        z_max=simulation.grid.length,
        absorber_rate=simulation.settings.absorber.spectral_rate,
    )
    original = dataclasses.replace(
        problem,
        linear=dispersion,
        initial=dataclasses.replace(problem.initial, field0=simulation._field0[None]),
    )
    root = np.sqrt(power_weight_at(problem, 0.0))
    # The launch is divided by this same row, so the entrance is one division: exact.
    np.testing.assert_array_equal(
        problem.initial.field0, original.initial.field0 / root
    )
    for z in (0.0, simulation.grid.length / 2, simulation.grid.length):
        correction = -0.5 * np.log(
            power_weight_at(problem, z) / power_weight_at(problem, 0.0)
        )
        # The inherited D fit has its original 1e-10 accuracy; the separately fitted
        # log term is tested to its round-off floor in test_power_weight.py.
        np.testing.assert_allclose(
            delta_phi(problem.linear, 0.0, z),
            delta_phi(dispersion, 0.0, z) + correction,
            rtol=1e-10,
            atol=1e-12,
        )
    # The Raman term carries its explicit contraction weights (schema 10).
    weights = {
        f"raman_terms[{index}]"
        for index, term in enumerate(problem.responses.raman_terms)
        if term.tensor_weights is not None
    }
    assert bool(weights) == raman
