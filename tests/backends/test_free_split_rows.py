# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A Cartesian envelope's split collar in place on the state, at any block of rows.

The factor depends on time alone, so native transforms the state's own rows, a block at
a time, and holds no coarse buffers. The block is a memory and launch choice: at any
size each march takes the reference's steps to its fields and its monitor record, within
a rounding budget fixed before the comparison.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from backends.low_storage_cases import parity_bound
from bounds import assert_within, device_backend, ladder_bound
from capability_routes import LENGTH, STEP
from monitor_cases import MARCHES, STEPS, as_driver, watched
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, TimeGrid
from solprop.monitor import BoundaryCheck
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import KernelProblem

pytest.importorskip("solprop_native")

BLOCKS = [1, 5, None]
"""Node chunks: one row at a time, one that leaves a shorter last block (192 columns
hold no multiple of 5), and the default, every row at once."""

BLOCK_IDS = ["one row", "with a tail", "every row"]

MONITORED = ("raw_removed_fraction", "removed_fraction", "collar_max", "guard_high_max")
"""What the monitor reads of the split: its own removal and what it found."""


def _beam(grid: TimeGrid, march: object | None = None) -> Simulation:
    """Return a Kerr beam on a 12 x 16 plane under the split collar.

    The route's case of ``tests/capability_routes.py``, on the given grid; with a march,
    on a fixed schedule, and the monitor relaxed: :func:`monitor_cases.watched` launches
    a copy near the window's edge on purpose.
    """
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[LENGTH],
        dz_max=STEP,
        dz_init=STEP,
        backend="numpy_ref",
        spatial_absorber=False,
        absorber_scheme="split_step",
        boundary_check=BoundaryCheck(budget=1.0, stop_factor=1.0),
        **(
            {}
            if march is None
            else {
                "integrator": march,
                "schedule": FixedSchedule.uniform(
                    0.0, LENGTH, LENGTH / STEPS, save_planes=[LENGTH]
                ),
            }
        ),
    )
    return Simulation.from_free(
        grid,
        FreeGrid(3e-4, 12, 4e-4, 16, collar_fraction=0),
        SpatialMedium("Ar", 1.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 80e-6),
        settings=settings,
    )


ENVELOPE = EnvelopeGrid.from_limits(
    LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
)


def _blocked(simulation: Simulation, chunk: int | None) -> KernelProblem:
    """Return the simulation's problem at a node chunk, or at its own."""
    problem = simulation.translate()
    if chunk is None:
        return problem
    return dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=chunk)
    )


def _marched(
    simulation: Simulation, name: str, problem: KernelProblem
) -> tuple[np.ndarray, dict[str, Any], dict[str, int]]:
    """Return the state, the monitor's record and the ledger after the fixed march."""
    backend = watched(simulation, name, problem)
    try:
        backend.propagate_to(LENGTH)
        driver = as_driver(backend)
        monitor = driver._monitor
        assert monitor is not None
        field = np.array(driver._observed_field(driver._require_state()))
        return field, monitor.summary(), backend.memory_ledger()
    finally:
        backend.close()


@pytest.fixture(scope="module", params=sorted(MARCHES))
def reference(request: pytest.FixtureRequest) -> tuple[str, np.ndarray, dict[str, Any]]:
    """Return a march's name and the reference's state and record on the beam."""
    simulation = _beam(ENVELOPE, MARCHES[request.param])
    field, record, _ = _marched(simulation, "numpy_ref", simulation.translate())
    return request.param, field, record


def _check_block(
    reference: tuple[str, np.ndarray, dict[str, Any]], chunk: int | None
) -> None:
    """Compare the native march at a node chunk with the reference's."""
    march, expected, theirs = reference
    simulation = _beam(ENVELOPE, MARCHES[march])
    got, ours, ledger = _marched(simulation, "native", _blocked(simulation, chunk))
    # The low-storage bound of four evaluations a step, scaled to the seven of the
    # longest tableau; the split's pair and product are less than one evaluation's, and
    # its factor's size is inside the bound's margin. A device build adds its
    # transforms' declared envelope, eight round trips a step at most. A reading sums
    # squares of the same samples, at twice the state's relative rounding.
    problem = simulation.translate()
    bound = ladder_bound(
        "native",
        7 / 4 * parity_bound(problem, transverse=problem.transform.n_nodes),
        simulation.grid.n_t,
        8 * STEPS,
    )
    assert_within(got, expected, bound, what=f"{march} at node chunk {chunk}")
    assert theirs["raw_removed_fraction"] > 1e-6
    for name in MONITORED:
        assert ours[name] == pytest.approx(theirs[name], rel=2 * bound), name
    # The rows are the state's own: no spectrum or time buffer of the split's.
    assert ledger["split_absorber_spectrum"] == 0
    assert ledger["split_absorber_time"] == 0


@pytest.mark.parametrize("chunk", BLOCKS, ids=BLOCK_IDS)
def test_a_block_of_any_size_takes_the_reference_s_steps(
    reference: tuple[str, np.ndarray, dict[str, Any]], chunk: int | None
) -> None:
    _check_block(reference, chunk)


@pytest.mark.gpu
@pytest.mark.parametrize("chunk", BLOCKS, ids=BLOCK_IDS)
def test_a_device_block_of_any_size_takes_the_reference_s_steps(
    reference: tuple[str, np.ndarray, dict[str, Any]], chunk: int | None
) -> None:
    if device_backend("native") == "none":
        pytest.skip("the installed solprop-native build has no accelerator backend")
    _check_block(reference, chunk)


def test_real_rows_keep_their_buffers() -> None:
    # A real time row is not a complex spectrum's row: a field grid takes the chunked
    # pair through the coarse buffers still.
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 150e-15)
    simulation = _beam(grid)
    problem = simulation.translate()
    assert problem.grid_kind == "field"
    backend = watched(simulation, "native", problem)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    assert ledger["split_absorber_spectrum"] > 0
    assert ledger["split_absorber_time"] > 0
