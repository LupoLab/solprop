# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coherent wall's loss depends on the relative phase of the channels.

The phase-sensitive leakage pair (``tests/driver/test_coherent_capillary.py``:
``test_phase_sensitive_leakage``). Two launches carry the same power in the same two
channels, ``HE_11`` and ``HE_12``, and differ only in their relative phase, 0 or pi. A
diagonal loss -- one rate per channel -- removes the same power from both, whatever the
phase, because the power each channel loses is its own. The coherent wall's generator
has a Hermitian part with off-diagonal entries (the wall is one boundary that every
channel reaches, not one per channel), so the power it removes is ``a^H (G + G^H) a``,
and the pair loses different amounts.

The run is linear -- one bin of the field grid at ``1e-9`` sqrt(W), whose Kerr phase
over the length is below 1e-12 rad (``test_coherent_capillary.linear_launch``) -- over
ten fixed steps of the fifth of a millimetre, with no temporal collar, so that nothing
but the flow and the carrier acts on the bin.
"""

from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import expm

from driver.test_coherent_capillary import LENGTH, capillary, propagated
from flow_parity import UNIT
from solprop.exceptions import GrazingIncidenceWarning, MaterialRangeWarning
from solprop.modes import CoherentWall
from solprop.translate.linop import delta_phi, power_weight_at

pytestmark = [
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    # The helper builds the weak collar before it is removed, and it says so.
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]
"""The qualification case's deliberate warnings (``test_coherent_capillary``)."""

AMPLITUDE = 1e-9
"""The launched amplitude, sqrt(W), split equally between the two channels."""


def uncollared(**overrides):
    """Return the qualification capillary with no temporal collar at all.

    ``absorber=None`` at the entry point means the default collar; the collar is
    removed from the built simulation instead, as the moving guide's tests do.
    """
    simulation = capillary(**overrides)
    simulation.settings = replace(simulation.settings, absorber=None)
    return simulation


def pair_launch(problem, grid, sign):
    """Return the problem launched into ``(HE_11 + sign HE_12) / sqrt(2)``, one bin."""
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    state = np.zeros_like(problem.initial.field0)
    state[0, 0, 0, index] = AMPLITUDE / np.sqrt(2.0)
    state[0, 1, 0, index] = sign * AMPLITUDE / np.sqrt(2.0)
    return index, replace(
        problem, initial=replace(problem.initial, field0=state, noise0=None)
    )


def transmitted(problem, grid, sign):
    """Return the fraction of the launched true power that reaches the end."""
    index, launched = pair_launch(problem, grid, sign)
    start = launched.initial.field0[0, :, 0, index]
    final = propagated(launched, grid)[0, :, 0, index]
    weights = [power_weight_at(problem, z)[:, 0, index] for z in (0.0, LENGTH)]
    return float(
        np.sum(weights[1] * abs(final) ** 2) / np.sum(weights[0] * abs(start) ** 2)
    )


def predicted(problem, grid, sign):
    """Return the transmitted fraction from ``scipy.linalg.expm`` of the block.

    The route's carrier is one scalar per bin (``delta_phi`` is the same for every
    channel beside a wall), so the exact propagator is ``exp(Phi) expm(G L)`` with
    ``G`` the declared constant block: Pade scaling and squaring on the full matrix,
    independent of the Schur factors and the Taylor actions the backend applies.
    """
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    carrier = delta_phi(problem.linear, 0.0, LENGTH)[0, :, 0, index]
    assert np.all(carrier == carrier[0])
    start = np.zeros(flow.n_modes, dtype=complex)
    start[:2] = AMPLITUDE / np.sqrt(2.0) * np.array([1.0, sign])
    final = np.exp(carrier[0]) * (
        expm(flow.generator[0, 0, :, :, index] * LENGTH) @ start
    )
    weights = [power_weight_at(problem, z)[:, 0, index] for z in (0.0, LENGTH)]
    return float(
        np.sum(weights[1] * abs(final) ** 2) / np.sum(weights[0] * abs(start) ** 2)
    )


@pytest.mark.medium
def test_two_phases_of_one_superposition_lose_differently_through_the_wall(
    record_property,
):
    """The pair's transmissions differ, as ``expm`` of the block says they do.

    Bound, stated first, on each transmission against :func:`predicted`, relative to
    the launched power: **2e-12** -- the Kerr phase of the launch, below 1e-12 rad,
    moves at most that fraction of the power between the two phases; ``expm`` of a
    block of norm ``||G L||`` below one is accurate to a few hundred ``u``, and ten
    applications of the backend's actions to a few tens (``tests/flow_parity.py``'s
    ``eps_apply``), both below 1e-13. The two phases' difference is asserted a
    thousand times larger than twice that bound, so the pair is resolved.
    """
    simulation = uncollared(wall=CoherentWall())
    problem, grid = simulation.translate(), simulation.grid
    bound = 1e-12 + 1e-13 + 1e-13
    got = {sign: transmitted(problem, grid, sign) for sign in (1.0, -1.0)}
    expected = {sign: predicted(problem, grid, sign) for sign in (1.0, -1.0)}
    departures = {sign: abs(value - expected[sign]) for sign, value in got.items()}
    record_property("departure", max(departures.values()))
    for sign, departure in departures.items():
        assert departure <= bound, sign
    difference = got[1.0] - got[-1.0]
    record_property("in_phase", got[1.0])
    record_property("out_of_phase", got[-1.0])
    record_property("difference", difference)
    assert abs(expected[1.0] - expected[-1.0]) > 1e3 * 2 * bound
    assert np.sign(difference) == np.sign(expected[1.0] - expected[-1.0])


@pytest.mark.medium
def test_a_diagonal_loss_takes_the_same_power_from_either_phase(record_property):
    """Without a wall, each channel keeps its own Marcatili rate: no phase matters.

    Bound, stated first: the two transmissions agree to **2e-12** of the launched
    power -- the same Kerr allowance, and the ten steps' rounding of a diagonal
    exponential, ``10 * 32 u`` per run -- while each run loses more than 1e-4 of it,
    so the loss is there and is simply not phase-sensitive.
    """
    simulation = uncollared()
    problem, grid = simulation.translate(), simulation.grid
    assert problem.linear.coupled_flow is None
    got = {sign: transmitted(problem, grid, sign) for sign in (1.0, -1.0)}
    record_property("difference", got[1.0] - got[-1.0])
    assert abs(got[1.0] - got[-1.0]) <= 1e-12 + 2 * 10 * 32 * UNIT
    assert got[1.0] < 1.0 - 1e-4
