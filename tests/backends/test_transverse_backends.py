# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The transverse edge on native CPU and ``numba_cpu``, against ``numpy_ref`` (SEP 40).

Both backends read the edge on the host from their state, as ``numpy_ref`` does; the
native Adams march holds its steps to the edge's next read and hands the state back
there, and ``numba_cpu`` steps from the host under the driver's ceiling (section 5.3).
The same run on two backends must then stop at the same read, read the same number of
times and keep the same record; a requested collar must remove the same energy.
"""

import dataclasses
import math
import warnings
from typing import Any

import numpy as np
import pytest

from solprop.backends.registry import available
from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryWarning
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.monitor import BoundaryCheck
from solprop.output.schema import PlaneStore
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from test_transverse_reads import (
    MISS,
    REREVIEW,
    committed,
    edge_record,
    finished,
    packet_run,
    stopped_at,
    true_stop,
)

BACKENDS = [
    pytest.param(
        "native",
        marks=pytest.mark.skipif(
            "native" not in available(), reason="native is not installed"
        ),
    ),
    "numba_cpu",
]
"""The backends compared with ``numpy_ref``."""

RTOL = 1e-9
"""Two backends' readings of one run: rounding apart, as the matched cases are."""
FLOOR = 1e-15
"""Rounding in a share of the launch energy: a share far below it is rounding alone."""


def on(simulation: Simulation, backend: str) -> Simulation:
    """Return `simulation` on `backend`."""
    simulation.settings = dataclasses.replace(simulation.settings, backend=backend)
    return simulation


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("integrator", [Multistep(), RungeKutta()], ids=["ms", "rk"])
def test_a_backend_stops_the_packet_where_numpy_ref_does(
    integrator: Multistep | RungeKutta, backend: str
) -> None:
    packet = REREVIEW
    reference = stopped_at(packet_run(packet, integrator=integrator))
    native = stopped_at(on(packet_run(packet, integrator=integrator), backend))
    assert native == pytest.approx(reference, rel=RTOL, abs=0.0)
    assert native - packet.length / 16 * (1 + 1e-9) < true_stop(packet) <= native


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_march_is_held_to_the_reads(backend: str) -> None:
    # The scaled miss: unmonitored, the Multistep march never reads the packet on
    # the wrap; under the ceiling the march lands on each read and stops at the
    # first past the exact flow's crossing.
    packet = MISS
    reference = stopped_at(packet_run(packet, integrator=Multistep()))
    native = stopped_at(on(packet_run(packet, integrator=Multistep()), backend))
    assert native == pytest.approx(reference, rel=RTOL, abs=0.0)
    assert native - packet.length / 16 * (1 + 1e-9) < true_stop(packet) <= native


def records(
    simulation: Simulation, backend: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run `simulation` on ``numpy_ref`` and on `backend`; return the edge records."""
    reference = edge_record(finished(simulation))
    native = edge_record(finished(on(simulation, backend)))
    return reference, native


def assert_same(reference: dict[str, Any], native: dict[str, Any]) -> None:
    """Assert that two edge records agree: counts exactly, numbers to rounding."""
    for key in ("transverse_status", "faces", "reads"):
        assert native[key] == reference[key], key
    for key in ("largest_gap", "ramp_removed", "raw_ramp_removed", "spatial_removed"):
        assert native[key] == pytest.approx(reference[key], rel=RTOL, abs=FLOOR), key
    for key in ("edge_baseline", "edge_max"):
        np.testing.assert_allclose(native[key], reference[key], rtol=1e-7, atol=FLOOR)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_backend_reads_the_miss_as_numpy_ref(backend: str) -> None:
    # The budget relaxed, the run completes: sixteen reads at most L/16 apart.
    simulation = packet_run(
        MISS, integrator=Multistep(), check=BoundaryCheck(transverse_budget=10.0)
    )
    reference, native = records(simulation, backend)
    assert native["reads"] >= 16
    assert native["largest_gap"] <= MISS.length / 16 * (1 + 1e-9)
    assert_same(reference, native)


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_replay_is_read_at_its_own_steps(backend: str) -> None:
    # A fixed schedule is not held to the reads: a replay reads where its steps
    # land past each read, and reports coarse with the same largest gap.
    packet = MISS
    edges = np.r_[0.0, 1e-4 * (2.0 ** np.arange(1, 13) - 1), 0.70475, 1.0]
    schedule = FixedSchedule(np.diff(edges) * packet.length)
    reference, native = records(
        packet_run(packet, integrator=Multistep(), schedule=schedule), backend
    )
    assert native["transverse_status"] == "coarse"
    assert_same(reference, native)


def radial_collar(backend: str, integrator: Multistep | RungeKutta) -> Simulation:
    """Return a linear radial run with a requested collar outside its aperture."""
    length = 2e-3
    grid = EnvelopeGrid.from_limits(length, 800e-9, (700e-9, 900e-9), 200e-15)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=50e-15, energy=1e-9), 40e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-7, atol=0.0),
        save_planes=[length / 2, length],
        dz_max=length / 8,
        dz_init=length * 1e-3,
        integrator=integrator,
        backend=backend,
        spatial_absorber=SpatialAbsorber(kappa=8e4),
    )
    return Simulation.from_radial(
        grid,
        RadialGrid(300e-6, 48, collar_fraction=1.0),
        SpatialMedium(1.0),
        [],
        pulse,
        settings=settings,
        noise=None,
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("integrator", [Multistep(), RungeKutta()], ids=["ms", "rk"])
def test_a_requested_collar_runs_as_on_numpy_ref(
    integrator: Multistep | RungeKutta, backend: str
) -> None:
    outputs: list[PlaneStore] = []
    for name in ("numpy_ref", backend):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BoundaryWarning)
            output = radial_collar(name, integrator).run(progress="none")
        assert isinstance(output, PlaneStore)
        outputs.append(output)
    reference, native = outputs
    np.testing.assert_allclose(committed(native), committed(reference), rtol=RTOL)
    expected = np.asarray(reference["field"][-1])
    found = np.asarray(native["field"][-1])
    error = float(np.linalg.norm(found - expected) / np.linalg.norm(expected))
    assert error < 1e-10
    first, second = edge_record(reference), edge_record(native)
    assert second["reads"] == first["reads"]
    assert second["transverse_status"] == first["transverse_status"]
    # The collar removes what reached it, and both backends take the same share.
    assert first["spatial_removed"] > 0.0
    assert math.isclose(
        second["spatial_removed"], first["spatial_removed"], rel_tol=1e-8, abs_tol=FLOOR
    )
    assert math.isclose(
        second["current"], first["current"], rel_tol=1e-8, abs_tol=FLOOR
    )


def test_a_backend_that_does_not_read_the_edge_runs_unwatched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Section 6: mlx does not read the edge; numba_cpu with its reading switched off
    # stands in for it. The run warns before its first step, is not stopped by the
    # packet on the wrap, and records that it was unwatched.
    from solprop.backends.numba_cpu import NumbaCpuBackend  # noqa: PLC0415

    monkeypatch.setattr(NumbaCpuBackend, "reads_transverse_edge", False)
    simulation = on(packet_run(REREVIEW, integrator=Multistep()), "numba_cpu")
    with pytest.warns(BoundaryWarning, match="does not read the transverse edge"):
        output = simulation.run(progress="none")
    assert isinstance(output, PlaneStore)
    record = edge_record(output)
    assert record["transverse_status"] == "unwatched"
    assert record["reads"] == 0
