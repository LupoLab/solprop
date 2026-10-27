# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preallocated output stores, the HDF5 saved-plane count and the output's admission.

A store sizes itself for the planes a run reserves and grows past them by doubling. An
HDF5 file records the planes actually written in its ``n_saved`` attribute, written
after each plane's data, so a hard stop at a plane boundary or part-way through a plane
leaves a file that reads, resumes and continues exactly; a clean close trims every
dataset to its count, and a file without the attribute reads as it always did. What is
stored does not depend on any of this: the same planes give the same arrays.
"""

import subprocess
import sys
import textwrap
from typing import Any

import h5py
import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.driver.memory import _catalogued_entries, admit_memory, plane_bytes
from solprop.exceptions import MemoryBudgetError
from solprop.grid import EnvelopeGrid
from solprop.output import HDF5Output, MemoryOutput, open_file, resume
from solprop.output.schema import (
    GridRecord,
    RunMetadata,
    SavedPlane,
    StepHistory,
    StepperRecord,
    StepStats,
)
from solprop.stats import CATALOGUE

N_MODES = 2
N_POL = 1


@pytest.fixture(scope="module")
def grid():
    return EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)


@pytest.fixture(scope="module")
def metadata(grid):
    return RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=(),
        simulation_kind="multimode",
        configuration={"gamma": 0.11},
        stepper=StepperRecord(tableau="DOPRI5", controller={"rtol": 1e-8, "atol": 0.0}),
    )


def make_plane(grid, index):
    """Build one plane with statistics, a noise phase, history and rows.

    The attempts (1 to 3) and the between-planes rows (0 to 2, none at the entrance)
    vary from plane to plane, so that every column is ragged.
    """
    rng = np.random.default_rng(index)
    shape = (N_MODES, N_POL, grid.n_omega)
    attempts = 1 + index % 3
    n_rows = index % 3
    return SavedPlane(
        z=0.1 * index,
        field=rng.standard_normal(shape) + 1j * rng.standard_normal(shape),
        stats={
            "energy": rng.random((N_MODES, N_POL)),
            "peak_power_total": np.asarray(rng.random()),
        },
        step_history=StepHistory(
            z=0.1 * index + 0.01 * np.arange(attempts),
            h=np.full(attempts, 0.05),
            norm=rng.random(attempts),
            accepted=np.arange(attempts) % 2 == 0,
            wall_time=rng.random(attempts),
        ),
        step_stats=StepStats(
            z=0.1 * index + 0.01 * np.arange(1, n_rows + 1),
            step=np.arange(n_rows),
            values={
                "energy": rng.random((n_rows, N_MODES, N_POL)),
                "peak_power_total": rng.random(n_rows),
            },
        ),
        noise_phase=rng.random(shape),
    )


KEYS = (
    "z",
    "field",
    "noise_phase",
    "stats/energy",
    "stats/peak_power_total",
    *(f"step_history/{name}" for name in ("counts", "z", "h", "norm", "accepted")),
    "step_history/wall_time",
    *(f"step_stats/{name}" for name in ("counts", "z", "step", "energy")),
    "step_stats/peak_power_total",
)
"""Every array a store holds for the planes of :func:`make_plane`."""


def stored(handler) -> dict[str, np.ndarray]:
    """Return every array of :data:`KEYS` a handler holds."""
    return {key: handler[key] for key in KEYS}


def assert_identical(left: dict[str, np.ndarray], right: dict[str, np.ndarray]):
    """Assert two stores hold the same arrays: shapes, dtypes and bytes."""
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert value.dtype == right[key].dtype, key
        assert value.shape == right[key].shape, key
        assert value.tobytes() == right[key].tobytes(), key


def write(handler, metadata, planes, *, reserve=None, close=True):
    """Open a handler, reserve if asked, save the planes and close if asked."""
    handler.open(metadata)
    if reserve is not None:
        handler.reserve(reserve)
    for plane in planes:
        handler.save(plane)
    if close:
        handler.close()
    return handler


def extents(path) -> dict[str, tuple[int, ...]]:
    """Return the shape of every plane, history and rows dataset in a file."""
    shapes: dict[str, tuple[int, ...]] = {}
    with h5py.File(path, "r") as file:

        def visit(name: str, node: Any) -> None:
            if isinstance(node, h5py.Dataset) and not name.startswith("meta"):
                shapes[name] = node.shape

        file.visititems(visit)
    return shapes


# ------------------------------------------------------------- the stored values


def test_the_same_planes_are_stored_identically_however_the_store_is_sized(
    grid, metadata, tmp_path
):
    planes = [make_plane(grid, index) for index in range(7)]
    reference = write(MemoryOutput(block=1), metadata, planes)
    expected = stored(reference)
    stores = [
        write(MemoryOutput(), metadata, planes, reserve=7),
        write(MemoryOutput(), metadata, planes, reserve=3),
        write(MemoryOutput(), metadata, planes, reserve=40),
    ]
    for store in stores:
        assert_identical(stored(store), expected)
    for index, reserve in enumerate((None, 7, 3, 40)):
        path = tmp_path / f"run-{index}.h5"
        write(HDF5Output(path), metadata, planes, reserve=reserve)
        with open_file(path) as file:
            assert file.n_saved == 7
            assert_identical(stored(file), expected)


def test_a_finished_file_is_trimmed_to_the_layout_of_an_unreserved_one(
    grid, metadata, tmp_path
):
    planes = [make_plane(grid, index) for index in range(5)]
    write(HDF5Output(tmp_path / "plain.h5"), metadata, planes)
    write(HDF5Output(tmp_path / "reserved.h5"), metadata, planes, reserve=50)
    assert extents(tmp_path / "reserved.h5") == extents(tmp_path / "plain.h5")
    with h5py.File(tmp_path / "reserved.h5", "r") as file:
        assert file.attrs["n_saved"] == 5
        field: Any = file["field"]
        assert field.shape == (5, N_MODES, N_POL, grid.n_omega)
        assert field.maxshape == (None, N_MODES, N_POL, grid.n_omega)
        assert field.chunks == (1, N_MODES, N_POL, grid.n_omega)
        counts: Any = file["step_history/counts"]
        history: Any = file["step_history/z"]
        rows: Any = file["step_stats/z"]
        energies: Any = file["step_stats/energy"]
        assert history.shape == (int(counts[()].sum()),)
        assert rows.shape == energies.shape[:1]


def test_a_file_reserved_for_fewer_planes_grows_by_doubling(
    grid, metadata, tmp_path, caplog
):
    planes = [make_plane(grid, index) for index in range(10)]
    path = tmp_path / "grown.h5"
    with caplog.at_level("DEBUG", logger="solprop.output.hdf5"):
        write(HDF5Output(path), metadata, planes, reserve=3)
    grown = [
        record.getMessage()
        for record in caplog.records
        if "grew its plane datasets" in record.getMessage()
    ]
    assert [message.rsplit(" ", 2)[-2] for message in grown] == ["6", "12"]
    with open_file(path) as file:
        assert file.n_saved == 10
        assert_identical(stored(file), stored(write(MemoryOutput(), metadata, planes)))
    assert extents(path)["field"] == (10, N_MODES, N_POL, grid.n_omega)


def test_a_memory_store_reserved_for_fewer_planes_grows_past_them(grid, metadata):
    planes = [make_plane(grid, index) for index in range(10)]
    store = write(MemoryOutput(), metadata, planes, reserve=3)
    assert store.n_saved == 10
    assert [plane.z for plane in store.planes()] == [plane.z for plane in planes]


def test_a_reservation_only_rises_and_is_never_negative():
    store = MemoryOutput()
    store.reserve(9)
    store.reserve(4)
    assert store.held_bytes(2, 10) == 90
    with pytest.raises(Exception, match="non-negative"):
        store.reserve(-1)


# ---------------------------------------------------------------- interruptions


def interrupt(path, *, planned: int, mid_plane: bool) -> None:
    """Turn a finished file into one a hard stop would have left.

    The plane-indexed datasets are grown back to the planned count, as the writer
    allocated them, and the history and rows columns get slack past their counts.
    With `mid_plane`, the next plane's data is half written -- its position, field,
    attempts and history rows -- but not counted, as a stop between the data and the
    count leaves it.
    """
    with h5py.File(path, "r+") as file:
        attributes: Any = file.attrs
        saved = int(attributes["n_saved"])

        def grow(name: str, node: Any) -> None:
            if not isinstance(node, h5py.Dataset) or name.startswith("meta"):
                return
            group, _, leaf = name.rpartition("/")
            per_plane = group in ("", "stats") or leaf == "counts"
            node.resize(planned if per_plane else 2 * node.shape[0] + 8, axis=0)

        file.visititems(grow)
        if mid_plane:
            junk = np.full(file["field"].shape[1:], 7.0 + 7.0j)  # type: ignore[union-attr]
            file["z"][saved] = 99.0  # type: ignore[index]
            file["field"][saved] = junk  # type: ignore[index]
            counts: Any = file["step_history/counts"]
            counts[saved] = 5
            start = int(counts[:saved].sum())
            history: Any = file["step_history/z"]
            history[start : start + 5] = 99.0
            rows: Any = file["step_stats/counts"]
            rows[saved] = 4


@pytest.mark.parametrize("mid_plane", [False, True], ids=["boundary", "mid-plane"])
def test_an_interrupted_file_reads_resumes_and_continues_exactly(
    grid, metadata, tmp_path, mid_plane
):
    planes = [make_plane(grid, index) for index in range(8)]
    whole = tmp_path / "whole.h5"
    write(HDF5Output(whole), metadata, planes)
    path = tmp_path / "interrupted.h5"
    write(HDF5Output(path), metadata, planes[:5])
    interrupt(path, planned=8, mid_plane=mid_plane)
    with open_file(path) as file:
        assert file.n_saved == 5
        assert_identical(
            stored(file), stored(write(MemoryOutput(), metadata, planes[:5]))
        )
    state = resume(path)
    assert state.n_saved == 5
    assert state.plane.z == planes[4].z
    np.testing.assert_array_equal(state.plane.field, planes[4].field)
    np.testing.assert_array_equal(state.plane.step_history.z, planes[4].step_history.z)
    np.testing.assert_array_equal(
        state.history.z, np.concatenate([plane.step_history.z for plane in planes[:5]])
    )
    np.testing.assert_array_equal(state.plane.step_stats.z, planes[4].step_stats.z)
    write(HDF5Output(path, append=True), metadata, planes[5:])
    assert extents(path) == extents(whole)
    with open_file(path) as continued, open_file(whole) as reference:
        assert_identical(stored(continued), stored(reference))


def test_a_hard_stop_part_way_through_a_plane_loses_only_that_plane(metadata, tmp_path):
    """A child process saves three planes, fails part-way through the fourth and exits.

    It exits with ``os._exit``, so no handler is closed and nothing is trimmed: the
    file is what the writer left. The fourth plane's statistic has the wrong shape, so
    its write fails after its position and field were written.
    """
    path = tmp_path / "stopped.h5"
    code = textwrap.dedent(
        f"""
        import os
        import numpy as np
        from solprop.output import HDF5Output
        from solprop.output.schema import RunMetadata, SavedPlane

        metadata = RunMetadata.from_json({metadata.to_json()!r})
        n_omega = metadata.grid.n_omega
        handler = HDF5Output({str(path)!r})
        handler.open(metadata)
        handler.reserve(6)
        for index in range(4):
            value = np.full(3 if index == 3 else 2, float(index))
            plane = SavedPlane(
                z=0.1 * index,
                field=np.full((1, 1, n_omega), index + 1j),
                stats={{"energy": value}},
            )
            try:
                handler.save(plane)
            except Exception:
                os._exit(3)
        os._exit(0)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 3, completed.stderr[-2000:]
    assert extents(path)["field"][0] == 6
    with open_file(path) as file:
        assert file.n_saved == 3
        np.testing.assert_array_equal(file["z"], [0.0, 0.1, 0.2])
        np.testing.assert_array_equal(file["stats/energy"][:, 0], [0.0, 1.0, 2.0])
    assert resume(path).plane.z == 0.2


def test_a_file_without_the_count_reads_by_its_extents(grid, metadata, tmp_path):
    planes = [make_plane(grid, index) for index in range(4)]
    path = tmp_path / "older.h5"
    write(HDF5Output(path), metadata, planes)
    before = extents(path)
    with h5py.File(path, "r+") as file:
        del file.attrs["n_saved"]
    expected = stored(write(MemoryOutput(), metadata, planes))
    with open_file(path) as file:
        assert file.n_saved == 4
        assert_identical(stored(file), expected)
    state = resume(path)
    assert state.n_saved == 4
    assert state.plane.z == planes[3].z
    more = [make_plane(grid, index) for index in range(4, 6)]
    write(HDF5Output(path, append=True), metadata, more, reserve=6)
    with h5py.File(path, "r") as file:
        assert file.attrs["n_saved"] == 6
    assert extents(path)["field"][0] == before["field"][0] + 2
    with open_file(path) as file:
        assert_identical(
            stored(file), stored(write(MemoryOutput(), metadata, planes + more))
        )


# ------------------------------------------------------------- memory admission


@pytest.fixture(scope="module")
def problem():
    simulation = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        pulseshape="sech",
        n_save=5,
        noise=None,
    )
    return simulation.translate()


def test_admission_charges_the_planned_memory_store(problem):
    without = admit_memory(problem, "numpy_ref", "none")
    assert "output" not in without.estimate.groups
    admission = admit_memory(problem, "numpy_ref", "none", output=MemoryOutput())
    planned = len(problem.stepper.save_planes) + 1
    assert admission.estimate.groups["output"] == planned * plane_bytes(problem)
    assert admission.estimate.total == without.estimate.total + planned * plane_bytes(
        problem
    )
    assert admission.record()["groups"]["output"] == planned * plane_bytes(problem)  # type: ignore[index]


def test_admission_charges_a_file_output_nothing(problem, tmp_path):
    without = admit_memory(problem, "numpy_ref", "none")
    admission = admit_memory(
        problem, "numpy_ref", "none", output=HDF5Output(tmp_path / "run.h5")
    )
    assert admission.estimate == without.estimate


def test_a_memory_store_over_the_budget_is_refused_naming_the_output(problem):
    total = admit_memory(problem, "numpy_ref", "none").estimate.total
    admit_memory(problem, "numpy_ref", total, output=None)
    with pytest.raises(MemoryBudgetError, match="output"):
        admit_memory(problem, "numpy_ref", total, output=MemoryOutput())


def test_a_plane_is_charged_what_the_memory_store_holds_for_it(problem):
    simulation_planes = len(problem.stepper.save_planes) + 1
    store = MemoryOutput()
    store.reserve(simulation_planes)
    # One plane of the problem's shape, with every statistic it records.
    entries = int(np.prod(problem.shape[1:]))
    expected = 16 * entries + 8 * 3
    if problem.initial.noise_model != "none":
        expected += 8 * entries
    for name in problem.stats_spec.names:
        expected += 8 * _catalogued_entries(
            CATALOGUE[name].shape,
            {
                "n_modes": problem.shape[1],
                "n_pol": problem.shape[2],
                "n_omega": problem.shape[3],
                "n_windows": problem.stats_spec.n_windows,
                "n_species": 0,
                "n_nodes": 0,
            },
        )
    assert plane_bytes(problem) == expected
    assert store.held_bytes(simulation_planes, 1) == simulation_planes


def test_every_catalogued_statistic_shape_has_known_axes():
    sizes: dict[str, int] = dict.fromkeys(
        (
            "n_modes",
            "n_pol",
            "n_omega",
            "n_windows",
            "n_apertures",
            "n_regions",
            "n_species",
            "n_nodes",
        ),
        3,
    )
    for entry in CATALOGUE.values():
        assert _catalogued_entries(entry.shape, sizes) >= 1
    with pytest.raises(Exception, match="n_unknown"):
        _catalogued_entries("(n_unknown,)", sizes)


@pytest.mark.medium
def test_a_run_in_memory_records_its_output_in_the_admission(tmp_path):
    simulation = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        pulseshape="sech",
        n_save=3,
        backend="numpy_ref",
        noise=None,
    )
    handler = simulation.run(MemoryOutput(), progress="none", memory_budget="none")
    assert handler.metadata is not None
    assert handler.metadata.backend is not None
    record: Any = handler.metadata.backend.settings["memory"]
    assert record["groups"]["output"] == 3 * plane_bytes(simulation.translate())
    file = simulation.run(tmp_path / "run.h5", progress="none", memory_budget="none")
    with open_file(tmp_path / "run.h5") as stored_file:
        assert stored_file.metadata is not None
        assert stored_file.metadata.backend is not None
        record = stored_file.metadata.backend.settings["memory"]
    assert "output" not in record["groups"]
    assert file.n_saved == 3
