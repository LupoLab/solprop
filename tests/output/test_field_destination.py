# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A plane store's destinations for an exported field (SEP 47 section 5.4).

A run's session exports a kept field straight into the stored form, the true-power
amplitude in the optics convention, into the room the store gives it
(:meth:`~solprop.output.schema.PlaneStore.field_destination`): ``MemoryOutput``'s
own block row, so the field is written once where it is kept, or a new array for
``HDF5Output``. :meth:`~solprop.output.schema.PlaneStore.save_exported` then stores
exactly what :meth:`~solprop.output.schema.PlaneStore.save` would have stored for the
same field: equalities here are exact, structural contracts of lossless storage.
"""

import numpy as np
import pytest

from output.test_handlers import N_MODES, N_POL, make_metadata, make_plane
from solprop import fourier
from solprop.exceptions import OutputError
from solprop.grid import EnvelopeGrid
from solprop.output import HDF5Output, MemoryOutput, open_file
from solprop.output.schema import FieldRetention, SavedPlane, StoredField
from solprop.stepper import FixedSchedule


@pytest.fixture
def grid():
    return EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)


@pytest.fixture
def metadata(grid):
    return make_metadata(grid, FixedSchedule(np.array([0.25, 0.25, 0.25, 0.25])))


def _bare(plane: SavedPlane) -> SavedPlane:
    """Return a plane's position, statistics and history, without its field."""
    return SavedPlane(
        z=plane.z,
        field=None,
        stats=plane.stats,
        step_history=plane.step_history,
        step_stats=plane.step_stats,
    )


def _stored(plane: SavedPlane, kind: str) -> StoredField:
    """Return a plane's field and noise phase in the stored form, as an export makes."""
    assert plane.field is not None
    return StoredField(
        fourier.spectrum_to_optics(plane.field, kind=kind),  # pyright: ignore[reportArgumentType]
        plane.noise_phase,
    )


def _shape(grid) -> tuple[int, int, int]:
    return (N_MODES, N_POL, grid.n_omega)


@pytest.mark.parametrize("with_noise", [False, True], ids=["plain", "noise"])
def test_a_memory_row_is_filled_once_where_the_field_is_kept(
    grid, metadata, with_noise
):
    """The row is the block's, written by the export alone, and stays as written."""
    exported = MemoryOutput(block=2)  # grows past its block on the third plane
    exported.open(metadata)
    converted = MemoryOutput(block=2)
    converted.open(metadata)
    rows: list[tuple[np.ndarray, bytes]] = []
    for index in range(5):
        plane = make_plane(grid, index, with_noise=with_noise)
        converted.save(plane)
        bare = _bare(plane)
        row = exported.field_destination(bare, _shape(grid))
        assert row.flags.c_contiguous and row.flags.writeable
        assert row.dtype == np.complex128 and row.shape == _shape(grid)
        stored = _stored(plane, grid.kind)
        np.copyto(row, stored.values)  # what the session's export does
        exported.save_exported(bare, StoredField(row, stored.noise_phase))
        rows.append((row, row.tobytes()))
    for name in ("field", "z", "stats/energy", "step_history/z"):
        assert np.asarray(exported[name]).tobytes() == (
            np.asarray(converted[name]).tobytes()
        ), name
    if with_noise:
        assert np.asarray(exported["noise_phase"]).tobytes() == (
            np.asarray(converted["noise_phase"]).tobytes()
        )
    # Each row was handed out once, holds its plane's field, and was never rewritten;
    # the rows before the block grew still hold what they were given.
    for (row, written), stored in zip(rows, exported.fields("optics"), strict=True):
        assert row.tobytes() == written == stored.tobytes()
    assert len({row.__array_interface__["data"][0] for row, _ in rows}) == len(rows)


def test_a_field_exported_elsewhere_is_copied_into_the_row(grid, metadata):
    handler = MemoryOutput()
    handler.open(metadata)
    plane = make_plane(grid, 0)
    stored = _stored(plane, grid.kind)
    handler.save_exported(_bare(plane), stored)
    assert not np.shares_memory(handler["field"], stored.values)
    assert handler["field"][0].tobytes() == stored.values.tobytes()


def test_only_a_plane_that_keeps_its_field_is_given_a_row(grid, metadata):
    handler = MemoryOutput()
    handler.retain(FieldRetention("planes", (1,)), _shape(grid))
    handler.open(metadata)
    first = _bare(make_plane(grid, 0))
    with pytest.raises(OutputError, match="plane 0 keeps no field"):
        handler.field_destination(first, _shape(grid))
    handler.save(first)
    second = make_plane(grid, 1)
    with pytest.raises(OutputError, match="carries no field of its own"):
        handler.field_destination(second, _shape(grid))
    row = handler.field_destination(_bare(second), _shape(grid))
    stored = _stored(second, grid.kind).values
    np.copyto(row, stored)
    handler.save_exported(_bare(second), StoredField(row))
    assert handler.plane_field(1).tobytes() == stored.tobytes()


def test_a_closed_or_unopened_store_gives_no_row(grid, metadata):
    unopened = MemoryOutput()
    with pytest.raises(OutputError, match="open"):
        unopened.field_destination(_bare(make_plane(grid, 0)), _shape(grid))
    closed = MemoryOutput()
    closed.open(metadata)
    closed.close()
    with pytest.raises(OutputError):
        closed.field_destination(_bare(make_plane(grid, 0)), _shape(grid))


def test_an_exported_plane_carries_no_field_of_its_own(grid, metadata):
    handler = MemoryOutput()
    handler.open(metadata)
    plane = make_plane(grid, 0)
    with pytest.raises(OutputError, match="without a field of its own"):
        handler.save_exported(plane, _stored(plane, grid.kind))
    assert handler.n_saved == 0


def test_a_stored_field_s_noise_phase_has_its_shape():
    with pytest.raises(OutputError, match="noise_phase has shape"):
        StoredField(np.ones((1, 1, 4), complex), np.zeros((1, 1, 2)))


def test_a_file_stores_an_exported_plane_as_it_stores_a_plane(grid, metadata, tmp_path):
    """A file is handed a new array (no room of its own), written as it is."""
    paths = {name: tmp_path / f"{name}.h5" for name in ("save", "exported")}
    handlers = {name: HDF5Output(path) for name, path in paths.items()}
    for handler in handlers.values():
        handler.open(metadata)
    for index in range(3):
        plane = make_plane(grid, index, with_noise=True)
        handlers["save"].save(plane)
        bare = _bare(plane)
        assert handlers["exported"].field_destination(bare, _shape(grid)) is None
        handlers["exported"].save_exported(bare, _stored(plane, grid.kind))
    for handler in handlers.values():
        handler.close()
    with open_file(paths["save"]) as saved, open_file(paths["exported"]) as exported:
        for name in ("field", "noise_phase", "z", "stats/energy", "step_history/h"):
            assert np.asarray(saved[name]).tobytes() == (
                np.asarray(exported[name]).tobytes()
            ), name


def test_a_read_only_file_refuses_an_exported_plane(grid, metadata, tmp_path):
    path = tmp_path / "run.h5"
    writer = HDF5Output(path)
    writer.open(metadata)
    writer.save(make_plane(grid, 0))
    writer.close()
    reader = HDF5Output(path, readonly=True)
    plane = make_plane(grid, 1)
    with pytest.raises(OutputError, match="for reading only"):
        reader.save_exported(_bare(plane), _stored(plane, grid.kind))
    reader.close()


def test_a_subclass_with_its_own_save_is_given_every_field_there():
    class Watching(MemoryOutput):
        def save(self, plane: SavedPlane) -> None:
            super().save(plane)

    class Declared(MemoryOutput):
        _takes_exports = True

        def save(self, plane: SavedPlane) -> None:
            super().save(plane)

    class Plain(HDF5Output):
        pass

    assert MemoryOutput().takes_exports
    assert not Watching().takes_exports
    assert Declared().takes_exports
    assert Plain.__dict__.get("_takes_exports") is None and Plain._takes_exports
