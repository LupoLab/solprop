# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the two output handlers: what they store, and how they resume."""

import dataclasses
import signal
import subprocess
import sys
from typing import Any

import h5py
import numpy as np
import pytest

from solprop import fourier
from solprop.exceptions import OutputError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.noise import input_noise
from solprop.output import (
    COMPRESSIONS,
    HISTORY_FIELDS,
    HDF5Output,
    MemoryOutput,
    Output,
    PlaneStore,
    check_cache,
    open_file,
    resume,
)
from solprop.output import hdf5 as hdf5_module
from solprop.output.schema import (
    PEAK_FIELDS_ANNOTATION,
    SCHEMA_VERSION,
    BackendRecord,
    GridRecord,
    NoiseRecord,
    RunMetadata,
    SavedPlane,
    StepHistory,
    StepLimits,
    StepperRecord,
    StepStats,
)
from solprop.stepper import DOPRI5, FixedSchedule, PIController


def kept(field: np.ndarray | None) -> np.ndarray:
    """Return a saved plane's field: every plane of these runs keeps its own."""
    assert field is not None
    return field


def item(parent, *names) -> Any:
    """Return one h5py object, loosely typed so that a test may inspect it."""
    value = parent
    for name in names:
        value = value[name]
    return value


def given(value) -> Any:
    """Return a value that must not be None, narrowed for the type checker."""
    assert value is not None
    return value


WAVELENGTH = 835e-9
LIMITS = (400e-9, 2e-6)
TRANGE = 2e-12
N_MODES = 2
N_POL = 1


@pytest.fixture
def grid():
    return EnvelopeGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE)


@pytest.fixture
def schedule():
    return FixedSchedule(np.array([0.25, 0.25, 0.25, 0.25]))


@pytest.fixture
def metadata(grid, schedule):
    return make_metadata(grid, schedule)


def make_metadata(grid, schedule, **overrides):
    """Build a mode-averaged record over one grid and schedule."""
    arguments = {
        "grid": GridRecord.from_grid(grid),
        "modes": (),
        "simulation_kind": "multimode",
        "configuration": {"gamma": 0.11},
        "stepper": StepperRecord.from_settings(
            DOPRI5,
            PIController(rtol=1e-8, atol=0.0),
            schedule=schedule,
            limits=StepLimits(dz_max=0.3),
        ),
    }
    return RunMetadata(**(arguments | overrides))


def make_plane(grid, index, *, with_noise=False, with_rows=False, seed=0):
    """Build one synthetic plane with statistics and a short step history."""
    rng = np.random.default_rng(seed + index)
    shape = (N_MODES, N_POL, grid.n_omega)
    field = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    attempts = 1 + index % 3
    history = StepHistory(
        z=np.full(attempts, 0.1 * index),
        h=np.full(attempts, 0.05),
        norm=np.linspace(0.1, 0.9, attempts),
        accepted=np.ones(attempts, dtype=bool),
        wall_time=np.full(attempts, 1e-3),
    )
    return SavedPlane(
        z=0.1 * index,
        field=field,
        stats={
            "energy": rng.random((N_MODES, N_POL)),
            "peak_power_total": np.asarray(rng.random()),
        },
        step_history=history,
        step_stats=make_rows(index, rng) if with_rows else StepStats.empty(),
        noise_phase=rng.random(shape) if with_noise else None,
    )


def make_rows(index, rng):
    """Build the between-planes rows of one plane, the entrance plane carrying none."""
    n_rows = index % 3
    return StepStats(
        z=0.1 * index + 0.01 * np.arange(1, n_rows + 1),
        step=np.arange(n_rows),
        values={
            "energy": rng.random((n_rows, N_MODES, N_POL)),
            "peak_power_total": rng.random(n_rows),
        },
    )


def run_into(handler, grid, metadata, n_planes, *, with_noise=False):
    """Open a handler, save `n_planes` synthetic planes and return them."""
    handler.open(metadata)
    planes = [make_plane(grid, i, with_noise=with_noise) for i in range(n_planes)]
    for plane in planes:
        handler.save(plane)
    return planes


# ------------------------------------------------------------------- the protocol


@pytest.mark.parametrize("factory", ["memory", "file"])
def test_both_handlers_satisfy_the_protocol(factory, tmp_path):
    handler = MemoryOutput() if factory == "memory" else HDF5Output(tmp_path / "a.h5")
    assert isinstance(handler, Output)
    assert isinstance(handler, PlaneStore)
    assert (handler.n_saved, handler.metadata, handler.closed) == (0, None, False)


def test_the_two_handlers_store_the_same_planes_bitwise(grid, metadata, tmp_path):
    memory = MemoryOutput()
    file = HDF5Output(tmp_path / "run.h5")
    run_into(memory, grid, metadata, 5, with_noise=True)
    run_into(file, grid, metadata, 5, with_noise=True)
    for key in ("field", "z", "stats/energy", "stats/peak_power_total"):
        # The two handlers apply the same conversion to the same arrays, so equality is
        # bitwise, not approximate.
        assert np.array_equal(memory[key], file[key]), key
    for name in ("counts", *HISTORY_FIELDS):
        assert np.array_equal(
            memory[f"step_history/{name}"], file[f"step_history/{name}"]
        )
    file.close()
    memory.close()


def test_a_handler_refuses_to_save_before_it_is_opened(grid, tmp_path):
    plane = make_plane(grid, 0)
    for handler in (MemoryOutput(), HDF5Output(tmp_path / "run.h5")):
        with pytest.raises(OutputError, match="call open"):
            handler.save(plane)


def test_file_save_handles_a_subclass_closing_after_the_append(
    grid, metadata, tmp_path
):
    class ClosingOutput(HDF5Output):
        def _append(self, plane, stored):
            super()._append(plane, stored)
            self.close()

    path = tmp_path / "run.h5"
    handler = ClosingOutput(path)
    handler.open(metadata)
    plane = make_plane(grid, 0)
    handler.save(plane)
    assert handler.closed and handler.n_saved == 1
    with open_file(path) as saved:
        np.testing.assert_array_equal(saved.fields("fft")[0], plane.field)


def test_a_handler_refuses_to_work_after_it_is_closed(grid, metadata, tmp_path):
    for handler in (MemoryOutput(), HDF5Output(tmp_path / "run.h5")):
        handler.open(metadata)
        handler.close()
        assert handler.closed
        # Closing twice is harmless, which lets a driver close in a finally block.
        handler.close()
        with pytest.raises(OutputError, match="is closed"):
            handler.save(make_plane(grid, 0))
        with pytest.raises(OutputError, match="is closed"):
            handler.open(metadata)


def test_an_unknown_key_names_the_ones_that_exist(grid, metadata):
    memory = MemoryOutput()
    run_into(memory, grid, metadata, 1)
    with pytest.raises(OutputError, match="unknown key 'banana'"):
        memory["banana"]
    with pytest.raises(OutputError, match="unknown key"):
        memory["stats/"]


@pytest.mark.parametrize("factory", ["memory", "file"])
def test_membership_answers_for_exactly_the_keys_the_accessor_accepts(
    factory, grid, metadata, tmp_path
):
    # Without __contains__ Python iterated the handler with integer keys and
    # the key parser failed on an int. Each candidate is checked against the accessor.
    handler = handler_of(factory, tmp_path)
    run_into(handler, grid, metadata, 2)
    held = (
        "field",
        "z",
        "stats/energy",
        "stats/peak_power_total",
        "step_history/counts",
        "step_history/h",
        "step_stats/z",
    )
    missing = ("banana", "stats/", "stats/z", "step_history/banana", "noise_phase")
    for key in held:
        assert key in handler, key
        handler[key]
    for key in missing:
        assert key not in handler, key
        with pytest.raises(OutputError):
            handler[key]
    # A key of the wrong type is simply not held; the membership test never raises.
    assert 0 not in handler
    assert None not in handler
    handler.close()
    if factory == "file":
        assert "z" not in handler


# --------------------------------------------------------------- the conventions


@pytest.mark.parametrize("grid_kind", ["envelope", "field"])
def test_the_stored_field_is_the_optics_spectrum_of_the_internal_one(
    grid_kind, schedule, tmp_path
):
    grid = (
        EnvelopeGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE)
        if grid_kind == "envelope"
        else FieldGrid.from_limits(1.0, WAVELENGTH, LIMITS, TRANGE)
    )
    metadata = make_metadata(grid, schedule)
    handler = HDF5Output(tmp_path / "run.h5")
    planes = run_into(handler, grid, metadata, 3)
    handler.close()
    with h5py.File(tmp_path / "run.h5", "r") as file:
        stored = np.asarray(item(file, "field")[()])
        assert file.attrs["spectral_convention"] == "optics"
        assert item(file, "field").attrs["spectral_convention"] == "optics"
    expected = np.stack(
        [
            fourier.spectrum_to_optics(kept(plane.field), kind=grid.kind)
            for plane in planes
        ]
    )
    # The boundary is a conjugation and a sign flip per bin, both exact in binary.
    assert np.array_equal(stored, expected)


def test_the_fft_accessor_returns_the_internal_arrays_exactly(grid, metadata):
    memory = MemoryOutput()
    planes = run_into(memory, grid, metadata, 3)
    internal = memory.fields(convention="fft")
    assert np.array_equal(internal, np.stack([kept(plane.field) for plane in planes]))
    assert np.array_equal(memory.fields(), memory["field"])


def test_the_noise_phase_stays_in_the_internal_convention(grid, metadata, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    planes = run_into(handler, grid, metadata, 2, with_noise=True)
    handler.close()
    with h5py.File(tmp_path / "run.h5", "r") as file:
        assert item(file, "noise_phase").attrs["spectral_convention"] == "fft"
        stored = np.asarray(item(file, "noise_phase")[()])
    assert np.array_equal(stored, np.stack([given(p.noise_phase) for p in planes]))


def test_the_noise_phase_is_read_back_through_the_accessor(grid, metadata, tmp_path):
    """The background processing restores is reachable by name."""
    memory = MemoryOutput()
    planes = run_into(memory, grid, metadata, 3, with_noise=True)
    expected = np.stack([given(p.noise_phase) for p in planes])
    assert np.array_equal(memory["noise_phase"], expected)
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 3, with_noise=True)
    assert handler["noise_phase"].shape == expected.shape
    assert handler["noise_phase"].dtype == np.float64
    handler.close()


def test_a_run_without_a_noise_model_has_no_noise_phase_to_read(
    grid, metadata, tmp_path
):
    memory = MemoryOutput()
    run_into(memory, grid, metadata, 2)
    with pytest.raises(OutputError, match="no noise phase was stored"):
        memory["noise_phase"]
    empty = MemoryOutput()
    empty.open(metadata)
    with pytest.raises(OutputError, match="no plane has been saved"):
        empty["noise_phase"]
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 2)
    with pytest.raises(OutputError, match="nothing named 'noise_phase'"):
        handler["noise_phase"]
    handler.close()


def test_both_handlers_list_the_statistics_they_store(grid, metadata, tmp_path):
    memory = MemoryOutput()
    assert memory.stat_names == ()
    run_into(memory, grid, metadata, 2)
    assert memory.stat_names == ("energy", "peak_power_total")
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 2)
    assert handler.stat_names == memory.stat_names
    handler.close()
    continued = HDF5Output(tmp_path / "run.h5", append=True)
    continued.open(metadata)
    assert continued.stat_names == memory.stat_names
    continued.close()
    with open_file(tmp_path / "run.h5") as stored:
        assert stored.stat_names == memory.stat_names


def test_open_file_reads_a_finished_run_through_the_protocol(grid, metadata, tmp_path):
    memory = MemoryOutput()
    planes = run_into(memory, grid, metadata, 3, with_noise=True)
    memory.to_file(tmp_path / "run.h5")
    with open_file(tmp_path / "run.h5") as stored:
        assert isinstance(stored, Output)
        assert stored.n_saved == 3
        assert (
            given(stored.metadata).configuration_hash() == metadata.configuration_hash()
        )
        for key in ("field", "z", "noise_phase", "stats/energy", "step_history/z"):
            assert np.array_equal(stored[key], memory[key]), key
        assert np.array_equal(stored.fields("fft"), memory.fields("fft"))
        with pytest.raises(OutputError, match="reading only"):
            stored.open(metadata)
        with pytest.raises(OutputError, match="reading only"):
            stored.save(planes[0])
    assert stored.closed
    with pytest.raises(OutputError, match="not open"):
        stored["field"]


def test_open_file_refuses_what_is_not_a_run(tmp_path):
    with pytest.raises(OutputError, match="cannot read"):
        open_file(tmp_path / "missing.h5")
    with h5py.File(tmp_path / "empty.h5", "w"):
        pass
    with pytest.raises(OutputError, match="not a solprop output file"):
        open_file(tmp_path / "empty.h5")
    with pytest.raises(OutputError, match="not both"):
        HDF5Output(tmp_path / "run.h5", append=True, readonly=True)


def test_the_noise_realisation_travels_in_the_metadata(grid, schedule, tmp_path):
    realisation = input_noise(grid, np.random.default_rng(4), n_modes=N_MODES)
    metadata = make_metadata(
        grid, schedule, noise=NoiseRecord.from_realisation(realisation)
    )
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 1)
    handler.close()
    stored = resume(tmp_path / "run.h5").metadata.noise
    assert stored is not None
    assert stored.spectral_convention == "fft"
    assert np.array_equal(stored.spectrum, realisation.spectrum)


# ------------------------------------------------------------------ storage layout


def test_the_file_is_chunked_one_plane_at_a_time_and_resizes(grid, metadata, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 50)
    assert handler.n_saved == 50
    handler.close()
    with h5py.File(tmp_path / "run.h5", "r") as file:
        field = item(file, "field")
        assert field.shape == (50, N_MODES, N_POL, grid.n_omega)
        assert field.chunks == (1, N_MODES, N_POL, grid.n_omega)
        assert field.maxshape == (None, N_MODES, N_POL, grid.n_omega)
        assert item(file, "stats", "energy").shape == (50, N_MODES, N_POL)
        assert item(file, "stats", "peak_power_total").shape == (50,)
        assert item(file, "step_history", "counts").shape == (50,)
        # Each plane records 1 + index % 3 attempts, 49 of them over 50 planes.
        assert item(file, "step_history", "z").shape == (99,)


def test_the_memory_store_grows_by_doubling(grid, metadata):
    memory = MemoryOutput(block=8)
    run_into(memory, grid, metadata, 50)
    assert memory.n_saved == 50
    assert memory["field"].shape[0] == 50
    assert memory["z"].shape == (50,)
    assert memory["step_history/z"].shape == (99,)


def test_the_memory_store_refuses_an_empty_block():
    with pytest.raises(OutputError, match="at least one plane"):
        MemoryOutput(block=0)


@pytest.mark.parametrize("compression", COMPRESSIONS, ids=["none", "gzip", "lzf"])
def test_every_compression_option_round_trips(compression, grid, metadata, tmp_path):
    options = 4 if compression == "gzip" else None
    handler = HDF5Output(
        tmp_path / "run.h5", compression=compression, compression_opts=options
    )
    planes = run_into(handler, grid, metadata, 3)
    handler.close()
    with h5py.File(tmp_path / "run.h5", "r") as file:
        assert item(file, "field").compression == compression
        stored = np.asarray(item(file, "field")[()])
    expected = np.stack(
        [
            fourier.spectrum_to_optics(kept(plane.field), kind=grid.kind)
            for plane in planes
        ]
    )
    # Both filters are lossless, so the round trip is bitwise.
    assert np.array_equal(stored, expected)


@pytest.mark.parametrize(
    ("compression", "options", "message"),
    [
        ("blosc", None, "compression must be one of"),
        ("lzf", 4, "only meaningful for gzip"),
        (None, 4, "only meaningful for gzip"),
        ("gzip", 11, "between 0 and 9"),
    ],
    ids=["unknown", "lzf-level", "none-level", "gzip-range"],
)
def test_an_unsupported_compression_request_is_refused(
    compression, options, message, tmp_path
):
    with pytest.raises(OutputError, match=message):
        HDF5Output(
            tmp_path / "run.h5", compression=compression, compression_opts=options
        )


def test_read_back_indexing_has_the_documented_shapes(grid, metadata, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 4)
    assert handler["field"].shape == (4, N_MODES, N_POL, grid.n_omega)
    assert handler["z"].shape == (4,)
    assert handler["stats/energy"].shape == (4, N_MODES, N_POL)
    assert handler["stats/peak_power_total"].shape == (4,)
    assert handler["step_history/counts"].tolist() == [1, 2, 3, 1]
    assert handler["step_history/accepted"].dtype == np.bool_
    assert handler.fields(convention="fft").shape == handler["field"].shape
    handler.close()


def test_the_handler_names_what_it_holds_when_a_key_is_missing(
    grid, metadata, tmp_path
):
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 1)
    with pytest.raises(OutputError, match="nothing named 'banana' in stats"):
        handler["stats/banana"]
    with pytest.raises(OutputError, match="nothing named 'nope' in step_history"):
        handler["step_history/nope"]
    handler.close()


def test_the_file_names_a_group_that_does_not_exist_yet(grid, metadata, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    handler.open(metadata)
    with pytest.raises(OutputError, match="no 'stats' group yet"):
        handler["stats/energy"]
    with pytest.raises(OutputError, match="nothing named 'field'"):
        handler["field"]
    handler.close()


def test_the_memory_store_names_what_it_holds_when_a_key_is_missing(grid, metadata):
    memory = MemoryOutput()
    run_into(memory, grid, metadata, 1)
    with pytest.raises(OutputError, match="no statistic 'banana'"):
        memory["stats/banana"]
    with pytest.raises(OutputError, match="no step-history column 'banana'"):
        memory["step_history/banana"]


def test_the_memory_store_refuses_a_field_before_the_first_plane(metadata):
    memory = MemoryOutput()
    memory.open(metadata)
    with pytest.raises(OutputError, match="no plane has been saved"):
        memory["field"]
    assert memory["z"].shape == (0,)
    assert memory["step_history/z"].shape == (0,)
    assert memory.noise_phase is None
    assert memory.stat_names == ()


def test_a_closed_file_handler_cannot_be_read(grid, metadata, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 1)
    handler.close()
    with pytest.raises(OutputError, match="is not open"):
        handler["field"]
    assert handler.path == tmp_path / "run.h5"


def test_a_never_opened_file_handler_says_so(tmp_path):
    with pytest.raises(OutputError, match="call open"):
        HDF5Output(tmp_path / "run.h5")["field"]


# -------------------------------------------------------------------- consistency

# The layout check lives in the base class, so both handlers refuse by name rather
# than the file handler failing deep inside h5py on a broadcast or a missing dataset.


def handler_of(factory, tmp_path):
    """Build one handler of either kind."""
    return MemoryOutput() if factory == "memory" else HDF5Output(tmp_path / "run.h5")


@pytest.mark.parametrize("factory", ["memory", "file"])
def test_a_plane_of_a_different_shape_is_refused(factory, grid, metadata, tmp_path):
    handler = handler_of(factory, tmp_path)
    run_into(handler, grid, metadata, 1)
    odd = make_plane(grid, 1)
    with pytest.raises(OutputError, match="plane 1 has field shape"):
        handler.save(
            SavedPlane(
                z=1.0,
                field=kept(odd.field)[:1],
                stats=odd.stats,
                step_history=odd.step_history,
            )
        )
    handler.close()


@pytest.mark.parametrize("factory", ["memory", "file"])
def test_a_plane_with_different_statistics_is_refused(
    factory, grid, metadata, tmp_path
):
    handler = handler_of(factory, tmp_path)
    run_into(handler, grid, metadata, 1)
    odd = make_plane(grid, 1)
    with pytest.raises(OutputError, match="reports statistics \\['energy'\\]"):
        handler.save(
            SavedPlane(
                z=1.0, field=odd.field, stats={"energy": np.real(kept(odd.field))}
            )
        )
    handler.close()


@pytest.mark.parametrize("factory", ["memory", "file"])
def test_a_plane_that_gains_or_loses_a_noise_phase_is_refused(
    factory, grid, metadata, tmp_path
):
    handler = handler_of(factory, tmp_path)
    run_into(handler, grid, metadata, 1, with_noise=True)
    plain = make_plane(grid, 1)
    with pytest.raises(OutputError, match="does not carry a noise phase"):
        handler.save(plain)
    handler.close()


def test_a_plane_that_gains_a_noise_phase_is_refused_too(grid, metadata):
    memory = MemoryOutput()
    run_into(memory, grid, metadata, 1)
    with pytest.raises(OutputError, match="carries a noise phase"):
        memory.save(make_plane(grid, 1, with_noise=True))


# -------------------------------------------------------------- metadata handling


def test_metadata_is_written_once_and_a_repeat_is_a_no_op(
    grid, metadata, schedule, tmp_path
):
    for handler in (MemoryOutput(), HDF5Output(tmp_path / "run.h5")):
        handler.open(metadata)
        handler.open(make_metadata(grid, schedule))
        assert handler.metadata is not None
        handler.close()


def test_a_different_run_is_refused_unless_forced(grid, schedule, tmp_path):
    one = make_metadata(grid, schedule)
    two = make_metadata(grid, schedule, configuration={"gamma": 0.22})
    handler = HDF5Output(tmp_path / "run.h5")
    handler.open(one)
    with pytest.raises(OutputError, match=r"configuration\.gamma"):
        handler.open(two)
    handler.open(two, force=True)
    assert handler.metadata is two
    handler.save(make_plane(grid, 0))
    handler.close()
    assert resume(tmp_path / "run.h5").metadata.configuration_hash() == (
        two.configuration_hash()
    )


def test_the_file_carries_the_identifying_attributes(grid, metadata, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 1)
    handler.close()
    with h5py.File(tmp_path / "run.h5", "r") as file:
        assert file.attrs["schema_version"] == metadata.schema_version
        assert file.attrs["spectral_convention"] == "optics"
        assert file.attrs["field_layout"] == "(n_z, n_modes, n_pol, n_omega)"
        assert file.attrs["configuration_hash"] == metadata.configuration_hash()
        assert file.attrs["simulation_kind"] == "multimode"
        assert file.attrs["grid_kind"] == "envelope"
        assert file.attrs["created"] == metadata.created


def test_an_existing_file_is_replaced_with_a_warning(grid, metadata, tmp_path, caplog):
    path = tmp_path / "run.h5"
    path.write_bytes(b"not an hdf5 file")
    handler = HDF5Output(path)
    with caplog.at_level("WARNING", logger=hdf5_module.__name__):
        handler.open(metadata)
    handler.close()
    assert "replacing the existing output file" in caplog.text


def test_the_file_is_opened_once_per_run(grid, metadata, tmp_path, monkeypatch):
    opened = []
    real = h5py.File

    def counting(*args, **kwargs):
        opened.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(hdf5_module.h5py, "File", counting)
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 10)
    handler["field"]
    handler.close()
    # One open for the whole run, ten saves and a read included, rather than a reopen
    # on every save and every metadata call.
    assert len(opened) == 1


# ------------------------------------------------------------------ cache and resume


def test_check_cache_reports_no_file(tmp_path, metadata):
    assert check_cache(tmp_path / "absent.h5", metadata) is False


def test_check_cache_accepts_the_same_run(grid, metadata, schedule, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, metadata, 2)
    handler.close()
    assert check_cache(tmp_path / "run.h5", make_metadata(grid, schedule))


def test_check_cache_refuses_a_changed_configuration_by_name(grid, schedule, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    run_into(handler, grid, make_metadata(grid, schedule), 2)
    handler.close()
    changed = make_metadata(grid, schedule, configuration={"gamma": 0.22})
    with pytest.raises(
        OutputError, match=r"these fields changed: configuration\.gamma"
    ):
        check_cache(tmp_path / "run.h5", changed)


def test_check_cache_refuses_a_file_that_is_not_an_output(tmp_path, metadata):
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("something", data=np.zeros(3))
    with pytest.raises(OutputError, match="not a solprop output file"):
        check_cache(path, metadata)


def test_check_cache_refuses_an_unreadable_file(tmp_path, metadata):
    path = tmp_path / "broken.h5"
    path.write_bytes(b"not an hdf5 file at all")
    with pytest.raises(OutputError, match="cannot read the output file"):
        check_cache(path, metadata)


def test_a_tampered_digest_is_refused(grid, metadata, tmp_path):
    path = tmp_path / "run.h5"
    handler = HDF5Output(path)
    run_into(handler, grid, metadata, 1)
    handler.close()
    with h5py.File(path, "r+") as file:
        file.attrs["configuration_hash"] = "0" * 64
    with pytest.raises(OutputError, match="is inconsistent"):
        check_cache(path, metadata)


def test_resume_restores_the_last_plane_the_schedule_and_the_history(
    grid, metadata, tmp_path, schedule
):
    handler = HDF5Output(tmp_path / "run.h5")
    planes = run_into(handler, grid, metadata, 4, with_noise=True)
    handler.close()
    state = resume(tmp_path / "run.h5")
    assert state.n_saved == 4
    assert state.plane.z == pytest.approx(planes[-1].z)
    # The field comes back through the inverse of the storage boundary, which is exact.
    assert np.array_equal(kept(state.plane.field), kept(planes[-1].field))
    assert np.array_equal(given(state.plane.noise_phase), given(planes[-1].noise_phase))
    assert np.array_equal(state.plane.stats["energy"], planes[-1].stats["energy"])
    assert np.array_equal(state.plane.step_history.z, planes[-1].step_history.z)
    assert state.history.n_steps == sum(p.step_history.n_steps for p in planes)
    stored = state.metadata.stepper.schedule
    assert stored is not None
    assert np.array_equal(stored, schedule.steps)


def test_resume_refuses_a_file_with_no_plane(metadata, tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    handler.open(metadata)
    handler.close()
    with pytest.raises(OutputError, match="holds no saved plane"):
        resume(tmp_path / "run.h5")


# ------------------------------------------------------------------------ append


def test_an_appending_handler_continues_the_file_where_it_stopped(
    grid, metadata, schedule, tmp_path
):
    path = tmp_path / "run.h5"
    first = HDF5Output(path, compression="lzf")
    planes = run_into(first, grid, metadata, 2, with_noise=True)
    first.close()
    with h5py.File(path, "r") as file:
        created = file.attrs["created"]

    again = HDF5Output(path, append=True)
    # A record built afresh for the same run: a later wall-clock time, and the same
    # hash, which is what "the same run" means.
    again.open(make_metadata(grid, schedule))
    assert again.n_saved == 2
    later = [make_plane(grid, 2, with_noise=True), make_plane(grid, 3, with_noise=True)]
    for plane in later:
        again.save(plane)
    assert again.n_saved == 4
    assert again["step_history/counts"].tolist() == [1, 2, 3, 1]
    again.close()

    state = resume(path)
    assert state.n_saved == 4
    assert np.array_equal(kept(state.plane.field), kept(later[-1].field))
    assert state.history.n_steps == sum(
        p.step_history.n_steps for p in (*planes, *later)
    )
    with h5py.File(path, "r") as file:
        # The file keeps its own record and creation time; nothing was rewritten.
        assert file.attrs["created"] == created
        assert item(file, "field").compression == "lzf"
        assert item(file, "field").shape[0] == 4
        assert item(file, "z")[()].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])


def test_an_appending_handler_refuses_a_different_run(grid, schedule, tmp_path):
    path = tmp_path / "run.h5"
    handler = HDF5Output(path)
    run_into(handler, grid, make_metadata(grid, schedule), 1)
    handler.close()
    again = HDF5Output(path, append=True)
    with pytest.raises(
        OutputError, match=r"cannot be appended to; these fields changed: configuration"
    ):
        again.open(make_metadata(grid, schedule, configuration={"gamma": 0.22}))
    again.close()
    # The refusal left the file as it was.
    assert resume(path).n_saved == 1


def test_an_appending_handler_needs_a_file_to_append_to(metadata, tmp_path):
    again = HDF5Output(tmp_path / "absent.h5", append=True)
    with pytest.raises(OutputError, match="cannot append to"):
        again.open(metadata)
    assert not (tmp_path / "absent.h5").exists()


def test_an_appending_handler_checks_the_next_plane_against_the_file(
    grid, metadata, tmp_path
):
    path = tmp_path / "run.h5"
    handler = HDF5Output(path)
    run_into(handler, grid, metadata, 1, with_noise=True)
    handler.close()
    again = HDF5Output(path, append=True)
    again.open(metadata)
    with pytest.raises(OutputError, match="plane 1 does not carry a noise phase"):
        again.save(make_plane(grid, 1))
    odd = make_plane(grid, 1, with_noise=True)
    with pytest.raises(OutputError, match="plane 1 has field shape"):
        again.save(
            SavedPlane(
                z=0.1,
                field=kept(odd.field)[:1],
                stats=odd.stats,
                noise_phase=given(odd.noise_phase)[:1],
            )
        )
    again.close()


def test_an_appending_handler_accepts_a_file_with_metadata_but_no_plane(
    grid, metadata, tmp_path
):
    path = tmp_path / "run.h5"
    handler = HDF5Output(path)
    handler.open(metadata)
    handler.close()
    again = HDF5Output(path, append=True)
    again.open(metadata)
    assert again.n_saved == 0
    again.save(make_plane(grid, 0))
    again.close()
    assert resume(path).n_saved == 1


# ------------------------------------------------------------------------ to_file


def test_to_file_writes_what_the_file_handler_would_have(grid, metadata, tmp_path):
    memory = MemoryOutput()
    planes = run_into(memory, grid, metadata, 6, with_noise=True)
    memory.to_file(tmp_path / "from_memory.h5")
    direct = HDF5Output(tmp_path / "direct.h5")
    direct.open(metadata)
    for plane in planes:
        direct.save(plane)
    direct.close()
    with (
        h5py.File(tmp_path / "from_memory.h5", "r") as one,
        h5py.File(tmp_path / "direct.h5", "r") as two,
    ):
        assert sorted(one) == sorted(two)
        for key in ("field", "z", "noise_phase"):
            assert np.array_equal(item(one, key)[()], item(two, key)[()]), key
        for name in ("counts", *HISTORY_FIELDS):
            path = f"step_history/{name}"
            assert np.array_equal(item(one, path)[()], item(two, path)[()]), path
        assert item(one, "meta/json")[()] == item(two, "meta/json")[()]


def test_to_file_passes_its_compression_through(grid, metadata, tmp_path):
    memory = MemoryOutput()
    run_into(memory, grid, metadata, 2)
    memory.to_file(tmp_path / "run.h5", compression="gzip", compression_opts=1)
    with h5py.File(tmp_path / "run.h5", "r") as file:
        assert item(file, "field").compression == "gzip"


def test_to_file_refuses_a_store_that_was_never_opened(tmp_path):
    with pytest.raises(OutputError, match="call open"):
        MemoryOutput().to_file(tmp_path / "run.h5")


def test_planes_round_trip_through_the_memory_store(grid, metadata):
    memory = MemoryOutput()
    planes = run_into(memory, grid, metadata, 3, with_noise=True)
    back = memory.planes()
    assert len(back) == 3
    for original, restored in zip(planes, back, strict=True):
        assert restored.z == pytest.approx(original.z)
        assert np.array_equal(kept(restored.field), kept(original.field))
        assert np.array_equal(given(restored.noise_phase), given(original.noise_phase))
        assert np.array_equal(restored.stats["energy"], original.stats["energy"])
        assert restored.step_history.n_steps == original.step_history.n_steps
    assert memory.stat_names == ("energy", "peak_power_total")


def test_closing_a_file_handler_that_never_opened_is_harmless(tmp_path):
    handler = HDF5Output(tmp_path / "run.h5")
    handler.close()
    assert handler.closed
    assert not (tmp_path / "run.h5").exists()


def test_the_memory_store_grows_its_noise_phase_too(grid, metadata):
    memory = MemoryOutput(block=2)
    planes = run_into(memory, grid, metadata, 5, with_noise=True)
    stored = given(memory.noise_phase)
    assert stored.shape[0] == 5
    assert np.array_equal(stored, np.stack([given(p.noise_phase) for p in planes]))


# ------------------------------------------------------- what executed the run


def test_the_backend_group_survives_both_handlers(tmp_path, grid, schedule):
    """A saved run says what executed it, and reads it back the same.

    The record is free-form beyond its two named fields, because the pools, the plan
    flag and the wisdom digest are a backend's own vocabulary; what the schema fixes is
    that they are there and that they come back.
    """
    record = BackendRecord.from_metadata(
        {
            "backend": "numba_cpu",
            "propagator_mode": "recompute",
            "numba_threads": 4,
            "plan_flag": "measure",
            "wisdom_digest": "0" * 16,
        }
    )
    metadata = dataclasses.replace(make_metadata(grid, schedule), backend=record)
    plane = make_plane(grid, 0)
    path = tmp_path / "executed.h5"
    for handler in (MemoryOutput(), HDF5Output(path)):
        handler.open(metadata)
        handler.save(plane)
        handler.close()
    for read in (handler, open_file(path)):
        record = read.metadata
        assert record is not None
        stored = record.backend
        assert stored is not None
        assert stored.name == "numba_cpu"
        assert stored.propagator_mode == "recompute"
        assert stored.settings["numba_threads"] == 4
        assert stored.settings["wisdom_digest"] == "0" * 16
    with h5py.File(path, "r") as file:
        assert file.attrs["backend"] == "numba_cpu"


def test_the_backend_group_is_outside_the_resume_key(grid, schedule):
    """A run continued on the other backend is the same run.

    On a replayed schedule the continued run is the same march as either uninterrupted
    one, within the rounding budget that separates the two backends' uninterrupted runs
    (``tests/driver/test_resume.py``), so the resume key holds the problem and the
    state alone and the execution is recorded beside it rather than inside it.
    """
    base = make_metadata(grid, schedule)
    one = dataclasses.replace(
        base, backend=BackendRecord("numpy_ref", "tabulate", {"threads": 1})
    )
    other = dataclasses.replace(
        base, backend=BackendRecord("numba_cpu", "recompute", {"threads": 18})
    )
    assert one.configuration_hash() == other.configuration_hash()
    assert one.differences(other) == ()
    assert base.configuration_hash() == one.configuration_hash()


def test_a_backend_record_names_what_it_needs():
    """The two named fields are required; everything else is the backend's own."""
    with pytest.raises(OutputError, match="must name the backend"):
        BackendRecord.from_metadata({"propagator_mode": "tabulate"})
    with pytest.raises(OutputError, match="propagator mode"):
        BackendRecord.from_metadata({"backend": "numpy_ref"})


# -------------------------------------------- the statistics channel between planes


def save_rows(handler, grid, metadata, n_planes=4):
    """Open `handler`, save `n_planes` planes carrying rows, and close it."""
    handler.open(metadata)
    planes = [make_plane(grid, index, with_rows=True) for index in range(n_planes)]
    for plane in planes:
        handler.save(plane)
    handler.close()
    return planes


def test_both_handlers_store_the_channel_alike(grid, metadata, tmp_path):
    """The rows are concatenated over planes exactly as the step history is."""
    memory = MemoryOutput()
    planes = save_rows(memory, grid, metadata)
    path = tmp_path / "rows.h5"
    save_rows(HDF5Output(path), grid, metadata)
    expected = [plane.step_stats.n_rows for plane in planes]
    with hdf5_module.open_file(path) as stored:
        assert memory["step_stats/counts"].tolist() == expected
        assert stored["step_stats/counts"].tolist() == expected
        for key in ("z", "step", "energy", "peak_power_total"):
            assert np.array_equal(
                memory[f"step_stats/{key}"], stored[f"step_stats/{key}"]
            ), key
        assert stored.step_stat_names == ("energy", "peak_power_total")
    assert memory.step_stat_names == ("energy", "peak_power_total")


def test_the_stored_rows_have_the_documented_shapes(grid, metadata, tmp_path):
    """One row axis, the statistic's own shape after it, and int64 step indices."""
    path = tmp_path / "rows.h5"
    planes = save_rows(HDF5Output(path), grid, metadata)
    total = sum(plane.step_stats.n_rows for plane in planes)
    with hdf5_module.open_file(path) as stored:
        assert stored["step_stats/z"].shape == (total,)
        assert stored["step_stats/step"].shape == (total,)
        assert stored["step_stats/step"].dtype == np.int64
        assert stored["step_stats/energy"].shape == (total, N_MODES, N_POL)
        assert stored["step_stats/peak_power_total"].shape == (total,)


def test_a_run_that_recorded_no_channel_reads_back_empty(grid, metadata, tmp_path):
    """The default, on both handlers: asking is not an error.

    A file written before schema version 3 has no ``step_stats`` group at all, and a
    run that recorded no periodic subset has an empty one; both answer the same way, so
    that processing and plotting can ask unconditionally and fall back to the planes.
    """
    memory = MemoryOutput()
    memory.open(metadata)
    memory.save(make_plane(grid, 0))
    memory.close()
    path = tmp_path / "plain.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.save(make_plane(grid, 0))
    with hdf5_module.open_file(path) as stored:
        for handler in (memory, stored):
            assert handler.step_stat_names == ()
            assert handler["step_stats/z"].size == 0
            assert handler["step_stats/energy"].size == 0
            assert handler["step_stats/counts"].tolist() == [0]


def test_a_file_older_than_the_channel_reads_back_empty(grid, metadata, tmp_path):
    """A schema-2 file has no group at all, and the reader must not fall over it."""
    path = tmp_path / "old.h5"
    with HDF5Output(path) as handler:
        handler.open(metadata)
        handler.save(make_plane(grid, 0))
    with h5py.File(path, "r+") as file:
        del file["step_stats"]
    with hdf5_module.open_file(path) as stored:
        assert stored.step_stat_names == ()
        assert stored["step_stats/z"].size == 0
    # And it still resumes as far as the reader is concerned: what refuses such a file
    # is the configuration hash, which carries the schema version, not the reader.
    state = hdf5_module.resume(path)
    assert state.plane.step_stats.n_rows == 0


def test_appending_to_a_file_older_than_the_channel_is_the_hash_s_refusal(
    grid, schedule, metadata, tmp_path
):
    """Continuing a schema-2 file opens it without the group and refuses by its hash.

    The handler adopts the groups the file has and skips the one it lacks, so the
    refusal is the readable one naming the schema version, not a missing group.
    """
    path = tmp_path / "old.h5"
    with HDF5Output(path) as handler:
        handler.open(make_metadata(grid, schedule, schema_version=2))
        handler.save(make_plane(grid, 0))
    with h5py.File(path, "r+") as file:
        del file["step_stats"]
    continued = HDF5Output(path, append=True)
    with pytest.raises(OutputError, match="schema_version"):
        continued.open(metadata)
    continued.close()


def test_a_plane_with_an_empty_channel_keeps_the_columns_aligned(grid, metadata):
    """Every column gets one piece per plane, so one cumulative sum locates them all.

    A plane may record no rows at all after one that did -- a resumed run whose first
    plane is shorter than its period does exactly that, because the driver's buffer is
    cleared on a load and the handler's layout comes from the file. The store must
    still be able to slice any plane out of any column.
    """
    memory = MemoryOutput()
    memory.open(metadata)
    memory.save(make_plane(grid, 1, with_rows=True))
    memory.save(
        dataclasses.replace(
            make_plane(grid, 2, with_rows=True), step_stats=StepStats.empty()
        )
    )
    memory.save(make_plane(grid, 3, with_rows=True))
    assert memory.step_stat_names == ("energy", "peak_power_total")
    assert memory["step_stats/counts"].tolist() == [1, 0, 0]
    for name in memory.step_stat_names:
        assert memory[f"step_stats/{name}"].shape[0] == 1
    restored = memory.planes()
    assert [plane.step_stats.n_rows for plane in restored] == [1, 0, 0]
    assert restored[1].step_stats.names == memory.step_stat_names


def test_an_entrance_plane_without_names_keeps_the_columns_aligned(grid, metadata):
    """The driver's entrance plane carries no rows and no names, as a run saves it.

    The statistic names are learnt only at the first plane with rows, so the planes
    before it must still get their empty pieces in those columns, or every later plane
    reads another plane's rows.
    """
    memory = MemoryOutput()
    memory.open(metadata)
    planes = [
        dataclasses.replace(make_plane(grid, 0), step_stats=StepStats.empty()),
        *(make_plane(grid, index, with_rows=True) for index in (1, 2, 3)),
    ]
    for plane in planes:
        memory.save(plane)
    assert memory["step_stats/counts"].tolist() == [0, 1, 2, 0]
    restored = memory.planes()
    assert [plane.step_stats.n_rows for plane in restored] == [0, 1, 2, 0]
    for original, plane in zip(planes[1:], restored[1:], strict=True):
        for name, value in original.step_stats.values.items():
            np.testing.assert_array_equal(plane.step_stats.values[name], value)
    assert restored[0].step_stats.values["energy"].shape == (0, N_MODES, N_POL)


def test_the_memory_store_names_the_row_columns_it_holds(grid, metadata):
    """A run *with* a channel still says which statistics it recorded."""
    memory = MemoryOutput()
    save_rows(memory, grid, metadata)
    with pytest.raises(OutputError, match="no step statistic 'banana'"):
        memory["step_stats/banana"]


def test_planes_round_trip_with_their_rows(grid, metadata, tmp_path):
    """`planes()` rebuilds each plane's own rows out of the concatenated columns."""
    memory = MemoryOutput()
    originals = save_rows(memory, grid, metadata)
    for original, restored in zip(originals, memory.planes(), strict=True):
        assert restored.step_stats.n_rows == original.step_stats.n_rows
        assert restored.step_stats.names == original.step_stats.names
        np.testing.assert_array_equal(restored.step_stats.z, original.step_stats.z)
        for name, value in original.step_stats.values.items():
            np.testing.assert_array_equal(restored.step_stats.values[name], value)
    # And through the file, which is what `to_file` feeds.
    path = tmp_path / "round.h5"
    memory.to_file(path)
    with hdf5_module.open_file(path) as stored:
        assert np.array_equal(stored["step_stats/z"], memory["step_stats/z"])


def test_an_appending_handler_continues_the_channel_where_it_stopped(
    grid, metadata, tmp_path
):
    """A resume appends rows and keeps `counts` one entry per plane."""
    path = tmp_path / "resumed.h5"
    first = [make_plane(grid, index, with_rows=True) for index in range(2)]
    with HDF5Output(path) as handler:
        handler.open(metadata)
        for plane in first:
            handler.save(plane)
    state = hdf5_module.resume(path)
    # The last plane comes back with its own rows, not the whole run's.
    assert state.plane.step_stats.n_rows == first[-1].step_stats.n_rows
    np.testing.assert_array_equal(state.plane.step_stats.z, first[-1].step_stats.z)

    rest = [make_plane(grid, index, with_rows=True) for index in (2, 3)]
    with HDF5Output(path, append=True) as handler:
        handler.open(metadata)
        for plane in rest:
            handler.save(plane)
    every = [*first, *rest]
    with hdf5_module.open_file(path) as stored:
        assert stored["step_stats/counts"].tolist() == [
            plane.step_stats.n_rows for plane in every
        ]
        assert stored["step_stats/z"].size == sum(
            plane.step_stats.n_rows for plane in every
        )
        np.testing.assert_array_equal(
            stored["step_stats/z"],
            np.concatenate([plane.step_stats.z for plane in every]),
        )


def test_a_plane_whose_channel_changes_its_statistics_is_refused(grid, metadata):
    """The layout guard covers the rows as it covers the plane statistics."""
    memory = MemoryOutput()
    memory.open(metadata)
    memory.save(make_plane(grid, 1, with_rows=True))
    spoiled = dataclasses.replace(
        make_plane(grid, 2, with_rows=True),
        step_stats=StepStats(
            z=np.array([0.5]), step=np.array([0]), values={"energy": np.ones((1, 1, 1))}
        ),
    )
    with pytest.raises(OutputError, match="reports step statistics"):
        memory.save(spoiled)


def test_the_schema_version_is_twelve_for_integrated_run_records(metadata):
    """Combined launch, TDSE, step-limit and saved-plane policies version the file."""
    assert SCHEMA_VERSION == 26
    assert metadata.schema_version == 26


@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no SIGKILL")
def test_planes_reach_the_disk_before_a_hard_stop(tmp_path):
    """Each saved plane is flushed, so a SIGKILL loses at most the plane in flight.

    A child process runs a short fibre with six save planes into a file and kills
    itself with SIGKILL from the reporter's ``plane`` call after the third plane was
    saved -- no ``finally`` runs, no handler is closed. The parent then reads the
    file: the entrance and the three planes are there and readable.
    """
    path = tmp_path / "killed.h5"
    code = f"""
import os, signal
from solprop.driver import prop_gnlse
from solprop.log import NullReporter

class Killer(NullReporter):
    planes = 0
    def plane(self, z, summary):
        self.planes += 1
        if self.planes == 3:
            os.kill(os.getpid(), signal.SIGKILL)

prop_gnlse(
    gamma=0.11, length=1e-3, betas=[0.0, 0.0, -11.83e-27], lambda0=835e-9,
    wavelength_limits=(500e-9, 2e-6), trange=1e-12, width=100e-15,
    peak_power=9.8, pulseshape="sech", shock=False, n_save=7,
    output={str(path)!r}, progress=Killer(),
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == -signal.SIGKILL, completed.stderr[-2000:]
    stored = open_file(path)
    try:
        # The entrance and the three planes saved before the kill; `n_save=7` would
        # have written seven planes in all, so the file is genuinely partial.
        assert stored.n_saved == 4
        assert stored["z"].shape == (4,)
        assert stored["z"][-1] > 0.0
    finally:
        stored.close()


def test_file_save_does_not_flush_a_handle_closed_by_the_append_hook(
    grid, metadata, tmp_path
):
    class ClosingOutput(HDF5Output):
        def _append(self, plane, stored):
            super()._append(plane, stored)
            self.close()

    path = tmp_path / "closed-after-append.h5"
    output = ClosingOutput(path)
    plane = make_plane(grid, 0)
    output.open(metadata)
    output.save(plane)
    assert output.closed and output.n_saved == 1
    with open_file(path) as result:
        np.testing.assert_array_equal(
            result["field"][0],
            fourier.spectrum_to_optics(kept(plane.field), kind=grid.kind),
        )


# ------------------------------------------------- post-run annotations (SEP 3, Q13)

REPORT = [{"label": "Ar", "largest_field": 3.2e10, "departure": None, "ok": True}]


def test_an_annotation_round_trips_through_memory_and_a_file(tmp_path, grid, metadata):
    memory = MemoryOutput()
    run_into(memory, grid, metadata, 2)
    memory.close()
    # A store may be annotated once its run has closed, which is when one is known.
    memory.annotate(PEAK_FIELDS_ANNOTATION, REPORT)
    assert memory.annotations == {PEAK_FIELDS_ANNOTATION: REPORT}
    # What is handed back is a copy.
    given(memory.annotations[PEAK_FIELDS_ANNOTATION]).append("changed")
    assert memory.annotations[PEAK_FIELDS_ANNOTATION] == REPORT
    memory.to_file(tmp_path / "copy.h5")
    with open_file(tmp_path / "copy.h5") as stored:
        assert stored.annotations == memory.annotations
    path = tmp_path / "run.h5"
    handler = HDF5Output(path)
    run_into(handler, grid, metadata, 2)
    handler.annotate("note", {"open": 1})
    handler.close()
    handler.annotate(PEAK_FIELDS_ANNOTATION, REPORT)
    handler.annotate("note", {"closed": 2})
    assert handler.annotations == {
        "note": {"closed": 2},
        PEAK_FIELDS_ANNOTATION: REPORT,
    }
    with open_file(path) as stored:
        assert stored.annotations == handler.annotations
        # Outside the record: the digest and the resume gate are unchanged.
        assert (
            given(stored.metadata).configuration_hash() == metadata.configuration_hash()
        )
    assert check_cache(path, metadata)
    continued = HDF5Output(path, append=True)
    continued.open(metadata)
    assert continued.annotations == handler.annotations
    continued.close()


def test_a_file_without_annotations_reads_as_before(tmp_path, grid, metadata):
    path = tmp_path / "run.h5"
    handler = HDF5Output(path)
    run_into(handler, grid, metadata, 1)
    handler.close()
    with h5py.File(path, "r") as file:
        assert "annotations" not in item(file, "meta")
    with open_file(path) as stored:
        assert dict(stored.annotations) == {}
        assert stored.n_saved == 1


def test_an_annotation_is_refused_where_it_cannot_be_written(tmp_path, grid, metadata):
    with pytest.raises(OutputError, match="call open"):
        MemoryOutput().annotate("note", 1)
    memory = MemoryOutput()
    memory.open(metadata)
    for name in ("", "a/b"):
        with pytest.raises(OutputError, match="non-empty name without"):
            memory.annotate(name, 1)
    with pytest.raises(OutputError, match="not plain JSON"):
        memory.annotate("note", {"array": np.zeros(2)})
    path = tmp_path / "run.h5"
    handler = HDF5Output(path)
    run_into(handler, grid, metadata, 1)
    handler.close()
    with open_file(path) as stored, pytest.raises(OutputError, match="reading only"):
        stored.annotate("note", 1)
    path.unlink()
    with pytest.raises(OutputError, match="cannot annotate"):
        handler.annotate("note", 1)
