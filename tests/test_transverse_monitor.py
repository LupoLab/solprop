# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor's transverse edge (SEP 40 section 5.3, Part II's witnesses).

The scaled suite case (lengths scaled paraxially by 1/8, linear here) reaches its
wall within the run as configured; a bare wall at f = 1 reads an edge share near
2.9e-6 (Part II, factor 3) and warns, and a requested collar at f = 2.6 completes
clear.
"""

import dataclasses
import math
import warnings
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from solprop.boundaries import SpatialAbsorber, SpatialAbsorberDesign
from solprop.driver import Simulation
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import LineGrid
from solprop.log import NullReporter
from solprop.monitor import BoundaryCheck
from solprop.output.hdf5 import resume
from solprop.output.schema import PlaneStore
from solprop.transverse_monitor import TransverseEdge
from test_transverse_resolution import LENGTH, suite


def edge_record(output: object) -> dict[str, Any]:
    """Return a finished run's transverse edge record."""
    assert isinstance(output, PlaneStore)
    assert output.outcome is not None
    monitor = output.outcome.execution["boundary_monitor"]
    assert isinstance(monitor, dict)
    return cast("dict[str, Any]", monitor["transverse"])


def quietly(simulation: Simulation) -> tuple[object, list[str]]:
    """Run, returning the output and the boundary warnings' messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", BoundaryWarning)
        output = simulation.run(progress="none")
    return output, [
        str(one.message) for one in caught if issubclass(one.category, BoundaryWarning)
    ]


def test_a_run_reaching_its_wall_is_stopped_naming_a_size() -> None:
    with (
        warnings.catch_warnings(),
        pytest.raises(BoundaryError, match="transverse edge") as caught,
    ):
        warnings.simplefilter("ignore", BoundaryWarning)
        suite(0.0, None).run(progress="none")
    message = str(caught.value)
    assert "a radius of at least" in message
    assert "n_r of at least" in message
    assert "lower bound" in message


def test_a_bare_wall_at_twice_the_radius_warns_near_the_witness() -> None:
    output, messages = quietly(suite(1.0, None))
    record = edge_record(output)
    assert record["transverse_status"] == "warned"
    assert 2.9e-6 / 3 <= max(record["edge_max"]) <= 3 * 2.9e-6
    assert any("reached the transverse edge" in one for one in messages)


def test_a_bare_wall_further_out_completes_clear() -> None:
    output, messages = quietly(suite(1.5, None))
    record = edge_record(output)
    assert record["transverse_status"] == "clear"
    assert max(record["edge_max"]) < 1e-6
    assert not [one for one in messages if "transverse edge" in one]


def test_a_requested_collar_completes_clear_and_credits_its_removal() -> None:
    output, _ = quietly(suite(2.6, SpatialAbsorber()))
    record = edge_record(output)
    assert record["transverse_status"] == "clear"
    assert record["spatial_removed"] > 0.0
    assert record["edge_max"] == [0.0]


def test_the_reads_are_spaced_by_the_ceiling() -> None:
    # Section 5.3: at most L/16 apart, enforced as a step ceiling, so a low-source run
    # that would step further reads at least sixteen times.
    output, _ = quietly(suite(1.5, None))
    record = edge_record(output)
    assert record["reads"] >= 16
    assert record["largest_gap"] <= LENGTH / 16 * (1 + 1e-9)


def test_numba_cpu_reads_the_edge_as_numpy_ref_does() -> None:
    # A backend that does not read it is tested in tests/backends/
    # test_transverse_backends.py, with numba_cpu's reading switched off.
    records = []
    for backend in ("numpy_ref", "numba_cpu"):
        simulation = suite(1.5, None)
        simulation.settings = dataclasses.replace(simulation.settings, backend=backend)
        output, messages = quietly(simulation)
        assert not any("does not read the transverse edge" in one for one in messages)
        records.append(edge_record(output))
    reference, found = records
    assert found["transverse_status"] == reference["transverse_status"]
    assert found["reads"] == reference["reads"] > 0


def test_the_transverse_budget_relaxes_the_edge_alone() -> None:
    simulation = suite(0.0, None)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck(transverse_budget=0.1),
    )
    # The edge reaches about 1e-4 of the launch: past the default stop, inside a
    # budget of 0.1 (1e-2 in energy). Time keeps its own budget.
    output, _ = quietly(simulation)
    assert edge_record(output)["transverse_status"] == "clear"


def _ramp_launch(share: float) -> Simulation:
    """Return the scaled suite with `share` of its launch in the top k_r column."""
    simulation = suite(1.5, None)
    problem = simulation.translate()
    field = np.array(problem.initial.field0[0])
    weights = problem.transform.geometry.spectral_weights  # pyright: ignore[reportOptionalMemberAccess]
    energy = float(np.sum(weights[:, None, None] * np.abs(field) ** 2))
    top = np.zeros_like(field)
    top[-1] = field[0]  # the beam's own spectrum, so the time window is not the issue
    top_energy = float(np.sum(weights[-1] * np.abs(top[-1]) ** 2))
    field = math.sqrt(1 - share) * field + math.sqrt(share * energy / top_energy) * top
    return simulation.with_input(field)


def test_a_launch_in_the_ramp_s_band_is_refused() -> None:
    with pytest.raises(BoundaryError, match="safety ramp's band"):
        quietly(_ramp_launch(1e-2))


def test_a_small_share_in_the_ramp_s_band_warns() -> None:
    _, messages = quietly(_ramp_launch(1e-5))
    assert any("safety ramp's band" in one for one in messages)


class _StopAtPlane(NullReporter):
    """Raise at the second saved plane."""

    def __init__(self) -> None:
        self.seen = 0

    def plane(self, z: float, summary: object) -> None:
        del z, summary
        self.seen += 1
        if self.seen == 2:
            raise KeyError("stopped")


def test_a_resume_judges_against_the_original_launch(tmp_path: Path) -> None:
    def build() -> Simulation:
        simulation = suite(1.0, None)
        simulation.settings = dataclasses.replace(
            simulation.settings, save_planes=[LENGTH / 2, LENGTH]
        )
        return simulation

    whole, _ = quietly(build())
    path = tmp_path / "run.h5"
    with warnings.catch_warnings(), pytest.raises(KeyError):
        warnings.simplefilter("ignore", BoundaryWarning)
        build().run(path, progress=_StopAtPlane())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        build().run(path, resume=True, progress="none")
    outcome = resume(path).outcome
    assert outcome is not None
    resumed = cast("dict[str, Any]", outcome.execution["boundary_monitor"])[
        "transverse"
    ]
    expected = edge_record(whole)
    assert resumed["edge_baseline"] == expected["edge_baseline"]
    assert resumed["launch_energy"] == pytest.approx(expected["launch_energy"])
    assert max(resumed["edge_max"]) == pytest.approx(max(expected["edge_max"]), rel=0.1)


def test_the_start_of_run_summary_says_what_the_edge_can_report() -> None:
    from solprop.driver.summary import run_summary  # noqa: PLC0415
    from solprop.output import MemoryOutput  # noqa: PLC0415

    simulation = suite(1.5, None)
    problem, metadata = simulation.translate(), simulation.metadata()

    def row(backend: str, check: BoundaryCheck | None) -> str:
        summary = run_summary(
            problem,
            metadata,
            backend=backend,
            execution={},
            output=MemoryOutput(),
            boundary_check=check,
        )
        return str(dict(summary.rows)["transverse edge"])

    assert "watched" in row("numpy_ref", BoundaryCheck())
    assert "unwatched on numba_cpu" in row("numba_cpu", BoundaryCheck())
    assert row("numpy_ref", None).endswith("off (boundary_check=None)")


def _line_edge() -> tuple[TransverseEdge, LineGrid]:
    """Return a line's edge: a collar of 100 um outside R = 200 um, 1 um spacing."""
    space = LineGrid(200e-6, 600, collar_fraction=0.5)
    design = SpatialAbsorberDesign(
        SpatialAbsorber(kappa=3e5),
        np.zeros(space.n_columns),
        1.0,
        (1e6,),
        collar_rate=1e3 * (1 - np.asarray(space.window)),
        alpha0=1e3,
        width=(100e-6,),
        kappa_p=(3e5,),
    )
    edge = TransverseEdge(
        space, design, 1e-3, 10.0, 0.01, np.arange(1), 2 * math.pi / 800e-9
    )
    return edge, space


def _packet(space: LineGrid, centre: float, kappa: float) -> np.ndarray:
    """Return a Gaussian packet at `centre` with transverse wavenumber `kappa`."""
    x = np.asarray(space.x)
    field = np.exp(-(((x - centre) / 10e-6) ** 2) + 1j * kappa * x)
    return np.fft.fft(field)[:, None, None]


def test_the_strip_reads_inward_flow_and_not_outward() -> None:
    # Section 5.3: the Kaiser(8)-windowed strip resolves direction above 16 / s; a
    # packet in the strip moving outward reads below 1e-6 of itself, one moving
    # inward reads most of itself.
    edge, space = _line_edge()
    launch = _packet(space, 0.0, 0.0)
    edge.launch(launch, 0.0, noise=None, steps=None, reads=True, declared=False)
    # solprop propagates as exp(-i k_z z): a packet exp(i kappa x) travels to -x.
    outward, _ = edge._strip_inward(_packet(space, 230e-6, -1e6))  # pyright: ignore[reportPrivateUsage]
    inward, _ = edge._strip_inward(_packet(space, 230e-6, 1e6))  # pyright: ignore[reportPrivateUsage]
    assert outward < 1e-6
    assert inward > 0.1


def test_the_onset_current_counts_inward_flow_only() -> None:
    edge, space = _line_edge()
    edge.launch(
        _packet(space, 0.0, 0.0),
        0.0,
        noise=None,
        steps=None,
        reads=True,
        declared=False,
    )
    leaving = edge._onset_current(_packet(space, 200e-6, -1e6))  # pyright: ignore[reportPrivateUsage]
    returning = edge._onset_current(_packet(space, 200e-6, 1e6))  # pyright: ignore[reportPrivateUsage]
    assert leaving < 1e-20
    assert returning > 1e3 * leaving


def test_the_ramp_s_removal_stops_a_run_at_ten_budgets() -> None:
    # Q8, decided on the under-resolved filament's measurements: the ramp's removal
    # warns above the budget and stops above ten, as the edge does (section 5.3).
    space = LineGrid(200e-6, 256)
    ramp = np.where(
        np.abs(np.asarray(space.k_x)) > 0.9 * np.max(np.abs(space.k_x)), 1e4, 0.0
    )
    design = SpatialAbsorberDesign(
        SpatialAbsorber(real_space=False), ramp, 1e4, (0.9 * float(np.max(space.k_x)),)
    )
    edge = TransverseEdge(
        space, design, 1e-3, 10.0, 0.01, np.arange(1), 2 * math.pi / 800e-9
    )
    clean = _packet(space, 0.0, 0.0)
    edge.launch(clean, 0.0, noise=None, steps=None, reads=True, declared=False)
    filament = clean.copy()
    top = int(np.argmax(np.abs(np.asarray(space.k_x))))
    filament[top] = 0.05 * float(np.max(np.abs(clean)))  # content generated at the edge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        edge.read(filament, 1e-4)
        with pytest.raises(BoundaryError, match="safety ramp has removed"):
            edge.read(filament, 1e-3)
    assert edge.status == "stopped"
