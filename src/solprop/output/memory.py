# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The in-memory output handler.

Everything in RAM: the handler for fast, small problems and for the members of a scan,
where a file per run would cost more than the run. It stores exactly what
:class:`solprop.output.hdf5.HDF5Output` stores, in the same layout and the same sign
convention -- both go through :class:`solprop.output.schema.PlaneStore` -- so a test can
compare the two bitwise and :meth:`MemoryOutput.to_file` writes the file the other
handler would have written.

The plane-indexed arrays are allocated once, for the planes the run reserved
(:meth:`~solprop.output.schema.PlaneStore.reserve`; the run loop reserves the entrance
and every save plane), or for a block of :data:`INITIAL_BLOCK` planes when nothing was
reserved, and grow by doubling past that rather than by concatenating on every save,
which would be quadratic in the number of planes. The field block follows the run's
field retention (:meth:`~solprop.output.schema.PlaneStore.retain`): every plane,
exactly the named planes, or none at all, in which case no field block is allocated.
The step history is variable in length per plane, so it is kept as a
list of pieces, concatenated when read, with a per-plane count beside it.

The manual page is ``docs/guide/results/output.md``.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from jaxtyping import Complex

from solprop.exceptions import OutputError
from solprop.fourier import GridKind, spectrum_to_internal
from solprop.output.schema import (
    HISTORY_FIELDS,
    STEP_STATS_INDEX,
    PlaneStore,
    RunMetadata,
    SavedPlane,
    StepHistory,
    StepStats,
    stored_spectrum,
)

__all__ = ["INITIAL_BLOCK", "MemoryOutput"]

logger = logging.getLogger(__name__)

INITIAL_BLOCK = 16
"""Planes preallocated when no run size was reserved (``architecture.md`` §13)."""


def _grown(block: np.ndarray, used: int) -> np.ndarray:
    """Return a copy of a block with twice the room, keeping the used entries.

    - ``block``: the current block, whose leading axis is the plane index.
    """
    grown = np.zeros((2 * block.shape[0], *block.shape[1:]), dtype=block.dtype)
    grown[:used] = block[:used]
    return grown


def _same_array(one: np.ndarray, other: np.ndarray) -> bool:
    """Return whether two arrays are views of exactly the same memory.

    - Returns: whether they start at the same address with the same shape and strides.
    """
    return (
        one.__array_interface__["data"][0] == other.__array_interface__["data"][0]
        and one.shape == other.shape
        and one.strides == other.strides
    )


def _history_slice(columns: Mapping[str, np.ndarray], piece: slice) -> StepHistory:
    """Return one plane's step history out of the concatenated columns.

    - ``columns``: the concatenated history columns, keyed by field name.
    """
    return StepHistory(**{name: columns[name][piece] for name in HISTORY_FIELDS})


def _joined(pieces: Sequence[np.ndarray], dtype: type) -> np.ndarray:
    """Return the pieces of one column as one array, keeping its dtype when empty.

    - ``dtype``: the column's dtype, which an empty concatenation would otherwise lose.
    """
    if not pieces:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(pieces)


def _step_stats_slice(
    columns: Mapping[str, list[np.ndarray]], index: int, names: Sequence[str]
) -> StepStats:
    """Return one plane's statistics rows out of the per-plane pieces.

    - ``columns``: the row columns, keyed by column name, each holding one piece per
      plane.
    - ``names``: the statistics the run recorded between planes.
    - Returns: the rows of that plane, with no rows when it recorded none. Every plane
      contributes a piece to both index columns, so this is always indexable.
    """
    return StepStats(
        z=columns["z"][index],
        step=columns["step"][index],
        values={name: columns[name][index] for name in names},
    )


class MemoryOutput(PlaneStore):
    """Hold a whole run in memory, in the layout an HDF5 file would use.

    Reading stays available after :meth:`~solprop.output.schema.PlaneStore.close`,
    unlike the file handler: nothing is
    released, so a driver may close in a ``finally`` and a caller keep reading.

    Parameters
    ----------
    block : int, optional
        Planes to preallocate when the run reserved none
        (:meth:`~solprop.output.schema.PlaneStore.reserve`); a reserved count takes its
        place. The store doubles whenever it is full.

    Raises
    ------
    OutputError
        If ``block`` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output import MemoryOutput
    >>> from solprop.output.schema import (
    ...     GridRecord, RunMetadata, SavedPlane, StepperRecord
    ... )
    >>> from solprop.stepper import DOPRI5, PIController
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> metadata = RunMetadata(
    ...     grid=GridRecord.from_grid(grid),
    ...     modes=(),
    ...     simulation_kind="mode_averaged",
    ...     configuration={},
    ...     stepper=StepperRecord.from_settings(
    ...         DOPRI5, PIController(rtol=1e-8, atol=0.0)
    ...     ),
    ... )
    >>> handler = MemoryOutput()
    >>> handler.open(metadata)
    >>> handler.save(SavedPlane(z=0.0, field=np.ones((1, 1, grid.n_omega), complex)))
    >>> handler.n_saved, handler["field"].shape
    (1, (1, 1, 1, 2560))
    >>> handler.close()
    """

    _takes_exports = True
    """Its ``_append`` and ``_stored_field`` keep to the exported path's contract: a
    run hands it each kept field already stored, into its own row."""

    def __init__(self, block: int = INITIAL_BLOCK) -> None:
        """Start an empty store with no room allocated yet."""
        super().__init__()
        if block < 1:
            msg = f"the preallocated block must hold at least one plane, got {block}"
            raise OutputError(msg)
        self._block = block
        self._capacity = 0
        self._z = np.zeros(0)
        self._counts = np.zeros(0, dtype=np.int64)
        # The field block and the noise phase beside it are allocated at the first
        # plane that keeps its field, for the planes that will.
        self._fields = np.zeros((0, 0, 0, 0), dtype=np.complex128)
        self._noise_phase: np.ndarray | None = None
        self._n_fields = 0
        self._stats: dict[str, np.ndarray] = {}
        self._history: dict[str, list[np.ndarray]] = {
            name: [] for name in HISTORY_FIELDS
        }
        # The between-planes rows are kept the way the history is: a list of pieces per
        # column with a per-plane count beside them, because how many rows a plane
        # contributes depends on how many steps it took.
        self._step_counts = np.zeros(0, dtype=np.int64)
        self._step_stats: dict[str, list[np.ndarray]] = {
            name: [] for name in STEP_STATS_INDEX
        }

    def _write_metadata(self, metadata: RunMetadata) -> None:
        """Record that a run has begun; the base class keeps the metadata itself."""
        # Guarded: the hash costs seconds on a large grid, and only this line needs it.
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "memory output opened for %s", metadata.configuration_hash()[:12]
            )

    def _allocate(self, plane: SavedPlane, shape: tuple[int, ...] | None) -> None:
        """Preallocate every plane-indexed array from the shapes of the first plane.

        The room is the planes the run reserved, or the handler's block when it
        reserved none.

        - ``shape``: its field's shape, which with its noise phase fixes the field
          block; ``None`` on a field-free plane.
        """
        room = self._planned or self._block
        self._capacity = room
        self._z = np.zeros(room)
        self._counts = np.zeros(room, dtype=np.int64)
        self._step_counts = np.zeros(room, dtype=np.int64)
        if shape is not None:
            self._allocate_fields(plane, shape)
        self._stats = {
            name: np.zeros((room, *np.shape(value)), dtype=np.float64)
            for name, value in plane.stats.items()
        }

    def _allocate_fields(self, plane: SavedPlane, shape: tuple[int, ...]) -> None:
        """Preallocate the field block, and the noise phase's, at the first kept field.

        Every plane's room for a run that keeps every field, which then grows with
        the planes; exactly the named planes for one that keeps some.

        - ``shape``: its field's shape, ``(n_modes, n_pol, n_omega)``.
        """
        retention = self.retention
        rows = (
            self._capacity if retention.fields == "all" else len(retention.field_planes)
        )
        self._fields = np.zeros((rows, *shape), dtype=np.complex128)
        if plane.noise_phase is not None:
            self._noise_phase = np.zeros((rows, *shape), dtype=np.float64)

    def _expand(self) -> None:
        """Double the room in every plane-indexed array."""
        used = self._n_saved
        self._z = _grown(self._z, used)
        self._counts = _grown(self._counts, used)
        self._step_counts = _grown(self._step_counts, used)
        if self.retention.fields == "all" and self._fields.shape[0]:
            self._fields = _grown(self._fields, used)
            if self._noise_phase is not None:
                self._noise_phase = _grown(self._noise_phase, used)
        self._stats = {name: _grown(block, used) for name, block in self._stats.items()}
        self._capacity *= 2
        logger.debug("memory output grew to room for %d planes", self._capacity)

    def _make_room(self, plane: SavedPlane, shape: tuple[int, ...] | None) -> None:
        """Grow the blocks so that they have room for the next plane.

        - ``shape``: its field's shape, or ``None`` when it keeps none.
        """
        if self._capacity == 0:
            self._allocate(plane, shape)
        elif self._n_saved == self._capacity:
            self._expand()
        if shape is not None and self._n_fields == 0 and not self._fields.shape[0]:
            self._allocate_fields(plane, shape)
        if (
            plane.noise_phase is not None
            and self._noise_phase is None
            and self._n_fields == 0
        ):
            # A room given before the plane's noise phase was known
            # (`field_destination`): the first kept plane fixes whether one is stored.
            self._noise_phase = np.zeros(self._fields.shape, dtype=np.float64)

    def field_destination(
        self, plane: SavedPlane, shape: tuple[int, ...]
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the block's row that the next plane's field is to be exported into.

        The row is allocated with the block, for the planes the run reserved, so a
        kept field is written once, where it is kept (SEP 47 section 5.4). It is never
        given to another plane, and nothing writes to it once the plane is stored.

        Parameters
        ----------
        plane : SavedPlane
            The plane about to be stored, without its field.
        shape : tuple of int
            ``(n_modes, n_pol, n_omega)`` of the field.

        Returns
        -------
        numpy.ndarray
            The row: writeable, C-contiguous complex128 of `shape` once the block is
            allocated for that shape.

        Raises
        ------
        OutputError
            If the handler is closed or was never opened, the run keeps no field at
            this plane, or `shape` is not the run's field shape.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.output import MemoryOutput
        >>> from solprop.output.schema import (
        ...     GridRecord, RunMetadata, SavedPlane, StepperRecord, StoredField
        ... )
        >>> from solprop.stepper import DOPRI5, PIController
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
        >>> handler = MemoryOutput()
        >>> handler.open(RunMetadata(
        ...     grid=GridRecord.from_grid(grid),
        ...     modes=(),
        ...     simulation_kind="mode_averaged",
        ...     configuration={},
        ...     stepper=StepperRecord.from_settings(
        ...         DOPRI5, PIController(rtol=1e-8, atol=0.0)
        ...     ),
        ... ))
        >>> plane = SavedPlane(z=0.0, field=None)
        >>> row = handler.field_destination(plane, (1, 1, grid.n_omega))
        >>> row[...] = 1.0
        >>> handler.save_exported(plane, StoredField(row))
        >>> stored = handler.fields("optics")
        >>> stored.shape, bool((stored == 1.0).all())
        ((1, 1, 1, 2560), True)
        """
        self._check_destination(plane, shape)
        self._make_room(plane, shape)
        return self._fields[self._n_fields]

    def _take_exported(
        self,
        plane: SavedPlane,
        exported: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the row holding an exported field, copying it in from elsewhere."""
        self._make_room(plane, exported.shape)
        row = self._fields[self._n_fields]
        if not _same_array(row, exported):
            np.copyto(row, exported)
        return row

    def _stored_field(
        self, plane: SavedPlane, kind: GridKind
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Convert a plane's field straight into its room in the block.

        - ``kind``: the grid's spectral layout.
        """
        assert plane.field is not None
        self._make_room(plane, plane.field.shape)
        return stored_spectrum(plane.field, kind, self._fields[self._n_fields])

    def _append(
        self,
        plane: SavedPlane,
        stored: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    ) -> None:
        """Append one plane to the blocks, growing them when they are full.

        - ``plane``: the plane: its field in the internal convention, or ``None`` where
          the field was exported straight into the stored form
          (:meth:`~solprop.output.schema.PlaneStore.save_exported`) and `stored` alone
          holds it. A subclass whose own ``_append`` reads the plane's field does not
          take exports (``PlaneStore.takes_exports``).
        - ``stored``: that field converted to the optics convention, in its row of the
          block (:meth:`_stored_field`); ``None`` on a field-free plane, which stores
          none.
        """
        self._make_room(plane, None if stored is None else stored.shape)
        index = self._n_saved
        self._z[index] = plane.z
        if stored is not None:
            if self._noise_phase is not None:
                self._noise_phase[self._n_fields] = plane.noise_phase
            self._n_fields += 1
        for name, value in plane.stats.items():
            self._stats[name][index] = value
        self._counts[index] = plane.step_history.n_steps
        for name in HISTORY_FIELDS:
            self._history[name].append(getattr(plane.step_history, name))
        self._append_rows(plane.step_stats)

    def _append_rows(self, rows: StepStats) -> None:
        """Append one plane's between-planes statistics rows.

        A plane may contribute none -- the entrance plane never does, and at a period
        above one a short plane need not -- and it still contributes an empty piece to
        every column, so that one cumulative sum of the counts locates every plane in
        every column.
        """
        index = self._n_saved
        self._step_counts[index] = rows.n_rows
        for name in STEP_STATS_INDEX:
            self._step_stats[name].append(getattr(rows, name))
        for name in self.step_stat_names:
            if name not in self._step_stats:
                # The names are learnt from the first plane with rows, so the planes
                # before it -- the entrance plane at least -- get their empty pieces
                # now, keeping this column one piece per plane like the index columns.
                shape = np.shape(rows.values[name])[1:]
                self._step_stats[name] = [np.zeros((0, *shape)) for _ in range(index)]
            pieces = self._step_stats[name]
            if name in rows.values:
                pieces.append(rows.values[name])
            else:
                shape = next((piece.shape[1:] for piece in pieces if piece.size), ())
                pieces.append(np.zeros((0, *shape)))

    def _read(self, group: str, name: str) -> np.ndarray:
        """Return one stored array, trimmed to the planes actually saved.

        - ``group``: ``''``, ``'stats'``, ``'step_history'`` or ``'step_stats'``.
        """
        saved = self._n_saved
        if group == "":
            return self._read_plane_array(name, saved)
        if group == "stats":
            if name not in self._stats:
                msg = (
                    f"no statistic {name!r} was stored; the run has "
                    f"{sorted(self._stats)}"
                )
                raise OutputError(msg)
            return self._stats[name][:saved]
        if group == "step_stats":
            return self._read_rows(name, saved)
        if name == "counts":
            return self._counts[:saved]
        if name not in self._history:
            msg = (
                f"no step-history column {name!r}; the run has "
                f"{['counts', *HISTORY_FIELDS]}"
            )
            raise OutputError(msg)
        pieces = self._history[name][:saved]
        return _joined(pieces, HISTORY_FIELDS[name])

    def _read_plane_array(self, name: str, saved: int) -> np.ndarray:
        """Return one of the arrays stored once per plane.

        - ``name``: ``'z'``, ``'field'`` or ``'noise_phase'``.
        """
        if name == "z":
            return self._z[:saved]
        if self._capacity == 0:
            msg = f"no plane has been saved, so there is no {name} to read"
            raise OutputError(msg)
        if name == "field":
            return self._fields[: self._n_fields]
        if self._noise_phase is None:
            msg = (
                "no noise phase was stored: the run carried no noise model, so "
                "there is no background to restore"
            )
            raise OutputError(msg)
        return self._noise_phase[: self._n_fields]

    def _read_rows(self, name: str, saved: int) -> np.ndarray:
        """Return one column of the between-planes statistics.

        A run that recorded no periodic subset reads back as an **empty channel**
        rather than an error, so that processing and plotting can ask for it
        unconditionally and fall back to the planes.
        """
        if name == "counts":
            return self._step_counts[:saved]
        if name in STEP_STATS_INDEX:
            return _joined(self._step_stats[name][:saved], STEP_STATS_INDEX[name])
        if name not in self._step_stats:
            if not self.step_stat_names:
                return np.zeros(0)
            msg = (
                f"no step statistic {name!r}; the run recorded "
                f"{['counts', *STEP_STATS_INDEX, *self.step_stat_names]}"
            )
            raise OutputError(msg)
        return _joined(self._step_stats[name][:saved], np.float64)

    def _release(self) -> None:
        """Release nothing: the arrays stay readable after the handler is closed."""

    @property
    def noise_phase(self) -> np.ndarray | None:  # numpydoc ignore=RT01
        """Stored noise phases, or ``None`` when the run had no noise.

        Shape ``(n_z, n_modes, n_pol, n_omega)``, rad, in the internal FFT
        convention, which is where the noise field lives.

        Raises
        ------
        OutputError
            If the run did not keep its field, and so its noise phase, on every plane.
        """
        if self._noise_phase is None:
            return None
        return self["noise_phase"]

    def planes(self) -> list[SavedPlane]:
        """Return every stored plane, back in the internal convention.

        The inverse of the whole save path: the field comes back through
        :func:`solprop.fourier.spectrum_to_internal`, so a plane that comes out is the
        plane that went in. This is how :meth:`to_file` re-feeds the file handler. A
        plane that kept no field comes back with ``field=None``.

        Returns
        -------
        list
            One :class:`solprop.output.schema.SavedPlane` per saved plane, in order.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> MemoryOutput().planes()
        []
        """
        if self._n_saved == 0:
            return []
        kind = self._require_metadata().grid.kind
        internal = (
            spectrum_to_internal(self._fields[: self._n_fields], kind=kind)
            if self._n_fields
            else None
        )
        phases = (
            None if self._noise_phase is None else self._noise_phase[: self._n_fields]
        )
        offsets = np.concatenate([[0], np.cumsum(self._counts[: self._n_saved])])
        columns = {name: self[f"step_history/{name}"] for name in HISTORY_FIELDS}
        built = []
        slot = 0
        for index in range(self._n_saved):
            piece = slice(int(offsets[index]), int(offsets[index + 1]))
            kept = self.retention.keeps(index)
            built.append(
                SavedPlane(
                    z=float(self._z[index]),
                    field=None if internal is None or not kept else internal[slot],
                    stats={
                        name: np.asarray(block[index])
                        for name, block in self._stats.items()
                    },
                    step_history=_history_slice(columns, piece),
                    step_stats=_step_stats_slice(
                        self._step_stats, index, self.step_stat_names
                    ),
                    noise_phase=None if phases is None or not kept else phases[slot],
                )
            )
            slot += int(kept)
        return built

    def to_file(
        self,
        path: Path | str,
        *,
        compression: str | None = None,
        compression_opts: int | None = None,
    ) -> None:
        """Write the file an :class:`solprop.output.hdf5.HDF5Output` would have written.

        Parameters
        ----------
        path : pathlib.Path or str
            Where to write. An existing file is replaced.
        compression : {None, 'gzip', 'lzf'}, optional
            Dataset compression; ``None`` for none.
        compression_opts : int or None, optional
            Level for ``'gzip'``, 0 to 9; not accepted for the other choices.

        Raises
        ------
        OutputError
            If the handler was never opened.

        Examples
        --------
        >>> import numpy as np, tempfile
        >>> from pathlib import Path
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.output import HDF5Output, MemoryOutput
        >>> from solprop.output.hdf5 import resume
        >>> from solprop.output.schema import (
        ...     GridRecord, RunMetadata, SavedPlane, StepperRecord
        ... )
        >>> from solprop.stepper import DOPRI5, PIController
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
        >>> metadata = RunMetadata(
        ...     grid=GridRecord.from_grid(grid),
        ...     modes=(),
        ...     simulation_kind="mode_averaged",
        ...     configuration={},
        ...     stepper=StepperRecord.from_settings(
        ...         DOPRI5, PIController(rtol=1e-8, atol=0.0)
        ...     ),
        ... )
        >>> handler = MemoryOutput()
        >>> handler.open(metadata)
        >>> start = np.ones((1, 1, grid.n_omega), complex)
        >>> handler.save(SavedPlane(z=0.5, field=start))
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     path = Path(folder) / "run.h5"
        ...     handler.to_file(path)
        ...     resume(path).plane.z
        0.5
        """
        # Imported here rather than at the top so that this module, and everything
        # that reaches it through solprop.stats, does not pull in h5py.
        from solprop.output.hdf5 import HDF5Output  # noqa: PLC0415

        metadata = self._require_metadata()
        handler = HDF5Output(
            path, compression=compression, compression_opts=compression_opts
        )
        try:
            handler.open(metadata)
            handler.reserve(self._n_saved)
            handler.retain(self.retention, self.field_shape)
            for plane in self.planes():
                handler.save(plane)
            if self.outcome is not None:
                handler.record_outcome(self.outcome, self.terminal_checkpoint)
            for name, value in self.annotations.items():
                handler.annotate(name, value)
        finally:
            handler.close()
