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

from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.log import NullReporter
from solprop.monitor import BoundaryCheck
from solprop.output.hdf5 import resume
from solprop.output.schema import PlaneStore
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


def test_a_backend_that_does_not_read_the_edge_says_so() -> None:
    simulation = suite(1.5, None)
    simulation.settings = dataclasses.replace(simulation.settings, backend="numba_cpu")
    output, messages = quietly(simulation)
    assert edge_record(output)["transverse_status"] == "unwatched"
    assert any("does not read the transverse edge" in one for one in messages)


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
