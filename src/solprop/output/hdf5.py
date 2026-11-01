# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The streaming HDF5 output handler, with cache and resume.

One file per run, kept open between optional snapshot transactions. Saves and
metadata calls reuse the handle. Without snapshots the file opens only once.

**What is on disk, and where.** Root attributes carry the identifying scalars --
``schema_version``, ``spectral_convention``, ``field_layout``, ``configuration_hash``,
``simulation_kind``, ``grid_kind``, ``created`` and, once a backend has executed the
run, ``backend`` -- and ``meta/json`` carries the
whole :class:`solprop.output.schema.RunMetadata` as one JSON string. The metadata goes
in a *dataset* rather than an attribute because HDF5 caps an attribute in the object
header at 64 kB and a serialised configuration -- a capillary with a tabulated
cladding glass, a stored noise realisation -- goes past that; the identifying scalars
stay attributes, where a reader looks for them first. ``field`` is ``(n_z, n_modes,
n_pol, n_omega)`` complex, chunked one plane at a time and extendable along ``n_z``,
in the **optics** convention and marked so. ``noise_phase``, when the run
has noise, has the same shape and stays in the internal convention. ``z`` is the plane
positions, ``stats/<name>`` one dataset per statistic, each with a leading plane axis,
and ``step_history/<column>`` the attempts of the whole run concatenated, with
``step_history/counts`` saying how many belong to each plane.

**Preallocation and the saved-plane count.** The plane-indexed datasets are created at
the first save for every plane the run reserved
(:meth:`~solprop.output.schema.PlaneStore.reserve`), with an unlimited leading axis, and
grow by doubling if the run saves more; the step-history and between-planes columns
grow by doubling too. The root attribute ``n_saved`` counts the planes actually
written: each plane's data is written and flushed first and the count last, so a hard
stop leaves the count at the last complete plane. The attempts and rows belonging to
those planes are the sums of ``step_history/counts`` and ``step_stats/counts`` over
them, so the count alone fixes every column's length. Every reader and :func:`resume`
read only the first ``n_saved`` planes. A clean close trims every dataset to its count,
so a finished file holds no unused rows.

**Field retention** (SEP 22). The root attribute ``fields`` says
which planes keep their field (:class:`~solprop.output.schema.FieldRetention`), and
``field_shape`` the shape of one plane's field whether or not it is stored. A run that
keeps every field stores ``field`` and ``noise_phase`` as above. One that keeps the
named planes stores exactly those, in plane order, with a ``field_planes`` dataset of
their indices beside them; one that keeps none has no ``field`` and no ``noise_phase``
dataset. :func:`resume` continues only from a plane that carries a field: a
terminal checkpoint always does, and a file whose last regular plane is field-free and
which holds no checkpoint is refused by name when a run is continued from it.

**Cache and resume.** :func:`check_cache` compares the file's record with the requested
one path by path (:meth:`~solprop.output.schema.RunMetadata.compare`, SEP 42): the
layout and the physics must agree, numbers to a rounding-level tolerance, so resuming
with different physics of the same shape is refused, by naming what changed; the
numerics may change, and provenance -- the optmat version, a model's name -- is
ignored, while an optmat model whose numbers changed is refused by name. A file of
another output schema is not read at all
(:data:`~solprop.output.schema.SCHEMA_VERSION`).
:func:`resume` returns the metadata, the last plane -- converted back to the internal
convention -- and the step history. A handler built with ``append=True`` then carries
on in the same file: it opens it read-write instead of replacing it, compares the
record offered with the file's unless the driver's gate already has, holds the file's
record as its own, and continues numbering planes and attempts where the file left
off. :func:`open_file`
opens a finished file **read-only** through the same protocol it was written with, which
is how :mod:`solprop.processing` reads a run after the driver has closed it.

Compression is restricted to what h5py ships: ``None``, ``'gzip'`` or ``'lzf'``. Blosc
is not offered, because it needs a filter plugin solprop does not depend on.

The manual page is ``docs/guide/results/output.md``.
"""

import json
import logging
import math
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import h5py
import numpy as np
from jaxtyping import Complex

from solprop import fourier
from solprop._display import Displayed, line, tail
from solprop.exceptions import OutputError, SolpropWarning
from solprop.output._writer_lock import WriterLock
from solprop.output.compatibility import RecordComparison
from solprop.output.schema import (
    HISTORY_FIELDS,
    STEP_STATS_INDEX,
    FieldRetention,
    FieldsChoice,
    JsonValue,
    MarchSnapshot,
    PlaneLayout,
    PlaneStore,
    RecipeFile,
    RecipeRecord,
    RunMetadata,
    RunOutcome,
    SavedPlane,
    StepHistory,
    StepStats,
    TerminalCheckpoint,
)

__all__ = [
    "COMPRESSIONS",
    "HDF5Output",
    "ResumeState",
    "check_cache",
    "open_file",
    "read_recipe",
    "resume",
    "write_recipe",
]

logger = logging.getLogger(__name__)

COMPRESSIONS = (None, "gzip", "lzf")
"""Compression filters h5py ships with; no plugin, so no Blosc (``dependencies.md``)."""

_MAX_GZIP_LEVEL = 9
"""Largest gzip level HDF5 accepts."""

_COLUMN_CHUNK = 256
"""Entries per chunk of a one-dimensional column; one per plane would be wasteful."""

_METADATA_KEY = "meta/json"

_ANNOTATIONS_KEY = "meta/annotations"
"""Group of the post-run annotations, one JSON string dataset each; optional."""
"""Dataset holding the whole run record as JSON."""

_OUTCOME_KEY = "meta/outcome"
_CHECKPOINT_KEY = "terminal_checkpoint"
_MARCH_KEY = "march_snapshot"
"""The group holding the Adams march's state at the last saved position."""


def _read_outcome(file: h5py.File) -> RunOutcome | None:
    """Read the execution status a file records, if it records one.

    Parameters
    ----------
    file : h5py.File
        Open saved output.

    Returns
    -------
    RunOutcome or None
        Validated execution status; ``None`` when the file records none.
    """
    if _OUTCOME_KEY not in file:
        return None
    stored = _dataset(file, _OUTCOME_KEY)[()]
    text = stored.decode("utf-8") if isinstance(stored, bytes) else str(stored)
    record = json.loads(text)
    record["previous_executions"] = tuple(record["previous_executions"])
    return RunOutcome(**record)


def _overwrite(group: h5py.Group, name: str, data: np.ndarray) -> None:
    """Write a dataset in place when its shape and type are unchanged, else anew.

    In place, a snapshot rewritten at every plane does not grow the file: HDF5 does not
    reclaim the space of a deleted dataset.

    Parameters
    ----------
    group : h5py.Group
        Its group.
    name : str
        Its name.
    data : numpy.ndarray
        What it holds.
    """
    if name in group:
        existing = group[name]
        if (
            isinstance(existing, h5py.Dataset)
            and existing.shape == data.shape
            and existing.dtype == data.dtype
        ):
            existing[...] = data
            return
        del group[name]
    group.create_dataset(name, data=data)


def _write_march(group: h5py.Group, snapshot: MarchSnapshot) -> None:
    """Store a march snapshot, its position last, so that a torn write reads as none.

    Parameters
    ----------
    group : h5py.Group
        The snapshot's group, possibly holding an earlier snapshot.
    snapshot : solprop.output.schema.MarchSnapshot
        The snapshot.
    """
    group.attrs["z"] = math.nan
    _overwrite(group, "state", np.asarray(snapshot.state, dtype=np.complex128))
    _overwrite(group, "positions", np.asarray(snapshot.positions, dtype=np.float64))
    _overwrite(group, "slots", np.asarray(snapshot.slots, dtype=np.int64))
    rows = group.require_group("rows")
    for name in [one for one in rows if one not in snapshot.rows]:
        del rows[name]
    for name, row in snapshot.rows.items():
        _overwrite(rows, name, np.asarray(row))
    group.attrs["identities"] = json.dumps(list(snapshot.identities))
    group.attrs["method"] = snapshot.method
    group.attrs["proposal"] = snapshot.proposal
    group.attrs["order"] = snapshot.order
    group.attrs["rejections"] = snapshot.rejections
    group.attrs["last_rejected"] = snapshot.last_rejected
    group.attrs["collar_was_quiet"] = snapshot.collar_was_quiet
    if snapshot.history_key is not None:
        group.attrs["history_key"] = snapshot.history_key
    elif "history_key" in group.attrs:
        del group.attrs["history_key"]
    group.attrs["z"] = snapshot.z


def _read_march(file: h5py.File, z: float) -> MarchSnapshot | None:
    """Read the march snapshot a file holds at `z`, or ``None``.

    Parameters
    ----------
    file : h5py.File
        Open saved output.
    z : float
        The position the run resumes from, m: a snapshot taken elsewhere, or one
        whose write was cut short, is not returned.

    Returns
    -------
    solprop.output.schema.MarchSnapshot or None
        The snapshot taken at `z`, if the file holds one.
    """
    if _MARCH_KEY not in file:
        return None
    group = _group(file, _MARCH_KEY)
    stored = float(np.asarray(group.attrs.get("z", math.nan)))
    if stored != z:
        return None
    try:
        rows = _group(group, "rows")
        return MarchSnapshot(
            z=stored,
            state=np.asarray(_dataset(group, "state")[()]),
            proposal=float(np.asarray(group.attrs["proposal"])),
            method=str(group.attrs["method"]),
            order=int(np.asarray(group.attrs["order"])),
            rejections=int(np.asarray(group.attrs["rejections"])),
            last_rejected=bool(np.asarray(group.attrs["last_rejected"])),
            positions=np.asarray(_dataset(group, "positions")[()]),
            identities=tuple(json.loads(str(group.attrs["identities"]))),
            slots=np.asarray(_dataset(group, "slots")[()]),
            rows={
                str(name): np.asarray(_dataset(rows, str(name))[()]) for name in rows
            },
            collar_was_quiet=bool(np.asarray(group.attrs["collar_was_quiet"])),
            history_key=(
                str(group.attrs["history_key"])
                if "history_key" in group.attrs
                else None
            ),
        )
    except (KeyError, ValueError, OutputError) as error:
        logger.warning("ignoring the unreadable march snapshot in the file: %s", error)
        return None


def _write_checkpoint(
    group: h5py.Group, checkpoint: TerminalCheckpoint, metadata: RunMetadata
) -> None:
    """Store a terminal field using the same Fourier boundary as regular saves.

    Parameters
    ----------
    group : h5py.Group
        Empty destination for the terminal record.
    checkpoint : TerminalCheckpoint
        Accepted field and available history, in internal FFT convention.
    metadata : RunMetadata
        Grid and output amplitude conventions.
    """
    plane = checkpoint.plane
    # A checkpoint always carries its field (TerminalCheckpoint refuses one without).
    assert plane.field is not None
    group.attrs["z"] = plane.z
    group.attrs["next_save_index"] = checkpoint.next_save_index
    group.attrs["landed"] = checkpoint.landed
    field = group.create_dataset(
        "field", data=fourier.spectrum_to_optics(plane.field, kind=metadata.grid.kind)
    )
    field.attrs["spectral_convention"] = "optics"
    field.attrs["power_convention"] = metadata.power_convention
    if plane.noise_phase is not None:
        phase = group.create_dataset("noise_phase", data=plane.noise_phase)
        phase.attrs["spectral_convention"] = "fft"
    statistics = group.create_group("stats")
    for name, value in plane.stats.items():
        statistics.create_dataset(name, data=value)
    history = group.create_group("step_history")
    for name in HISTORY_FIELDS:
        history.create_dataset(name, data=getattr(plane.step_history, name))
    rows = group.create_group("step_stats")
    rows.create_dataset("z", data=plane.step_stats.z)
    rows.create_dataset("step", data=plane.step_stats.step)
    for name, value in plane.step_stats.values.items():
        rows.create_dataset(name, data=value)


def _read_checkpoint(
    file: h5py.File, metadata: RunMetadata
) -> TerminalCheckpoint | None:
    """Read a separate terminal field in internal FFT convention.

    Parameters
    ----------
    file : h5py.File
        Open saved output.
    metadata : RunMetadata
        Grid and output amplitude conventions.

    Returns
    -------
    TerminalCheckpoint or None
        Validated accepted field, absent when no terminal state is stored.
    """
    if _CHECKPOINT_KEY not in file:
        return None
    group = _group(file, _CHECKPOINT_KEY)
    statistics = _group(group, "stats")
    history = _group(group, "step_history")
    rows = _group(group, "step_stats")
    position = np.asarray(group.attrs["z"])
    index = np.asarray(group.attrs["next_save_index"])
    landed = np.asarray(group.attrs["landed"])
    if (
        position.shape != ()
        or position.dtype.kind not in "fiu"
        or index.shape != ()
        or index.dtype.kind not in "iu"
        or landed.shape != ()
        or landed.dtype.kind != "b"
    ):
        raise OutputError("terminal checkpoint attributes must be typed scalars")
    plane = SavedPlane(
        z=float(position.item()),
        field=fourier.spectrum_to_internal(
            np.asarray(_dataset(group, "field")[()]), kind=metadata.grid.kind
        ),
        stats={
            str(name): np.asarray(_dataset(statistics, str(name))[()])
            for name in statistics
        },
        step_history=StepHistory(
            **{name: np.asarray(_dataset(history, name)[()]) for name in HISTORY_FIELDS}
        ),
        step_stats=StepStats(
            z=np.asarray(_dataset(rows, "z")[()]),
            step=np.asarray(_dataset(rows, "step")[()]),
            values={
                str(name): np.asarray(_dataset(rows, str(name))[()])
                for name in rows
                if str(name) not in STEP_STATS_INDEX
            },
        ),
        noise_phase=np.asarray(_dataset(group, "noise_phase")[()])
        if "noise_phase" in group
        else None,
    )
    return TerminalCheckpoint(
        plane=plane,
        next_save_index=int(index.item()),
        landed=bool(landed.item()),
    )


def _check_compression(compression: str | None, compression_opts: int | None) -> None:
    """Check a compression request against what h5py can do without a plugin.

    Parameters
    ----------
    compression : {None, 'gzip', 'lzf'}
        The filter.
    compression_opts : int or None
        Level for ``'gzip'``, 0 to 9; ``None`` for the others.

    Raises
    ------
    OutputError
        If the filter is unknown, or the level is given for a filter that has none, or
        is outside the range gzip accepts.
    """
    if compression not in COMPRESSIONS:
        msg = f"compression must be one of {COMPRESSIONS}, got {compression!r}"
        raise OutputError(msg)
    if compression != "gzip" and compression_opts is not None:
        msg = f"compression_opts is only meaningful for gzip, not for {compression!r}"
        raise OutputError(msg)
    if compression_opts is not None and not 0 <= compression_opts <= _MAX_GZIP_LEVEL:
        msg = f"the gzip level must be between 0 and 9, got {compression_opts}"
        raise OutputError(msg)


def _dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
    """Return one dataset of a group, narrowed for the type checker.

    h5py types every lookup as a group, a dataset or a named type; solprop's files hold
    only the first two and every name below is one this module created, so the cast is
    total.

    Parameters
    ----------
    parent : h5py.Group
        The group to look in.
    name : str
        Name of the dataset.

    Returns
    -------
    h5py.Dataset
        The dataset.
    """
    return cast("h5py.Dataset", parent[name])


def _group(parent: h5py.Group, name: str) -> h5py.Group:
    """Return one subgroup of a group, narrowed for the type checker.

    Parameters
    ----------
    parent : h5py.Group
        The group to look in.
    name : str
        Name of the subgroup.

    Returns
    -------
    h5py.Group
        The subgroup.
    """
    return cast("h5py.Group", parent[name])


def _kind_of(key: str, *, every_field: bool = True) -> str:
    """Return which count a dataset's leading axis follows.

    Parameters
    ----------
    key : str
        The dataset's accessor key: ``'z'``, ``'field'``, ``'noise_phase'``, or a
        name prefixed by ``'stats/'``, ``'step_history/'`` or ``'step_stats/'``.
    every_field : bool, optional
        Whether the run keeps its field on every plane; if not, the field and the
        noise phase have one entry per plane that keeps its field.

    Returns
    -------
    str
        ``'plane'`` for one entry per saved plane (the per-plane ``counts`` columns
        among them), ``'field'`` for one per kept field, ``'history'`` for a
        step-history column and ``'rows'`` for a between-planes statistics column.
    """
    group, _, name = key.rpartition("/")
    if group == "" and name in ("field", "noise_phase") and not every_field:
        return "field"
    if group in ("", "stats") or name == "counts":
        return "plane"
    return "history" if group == "step_history" else "rows"


def _stored_counts(file: h5py.File) -> tuple[int, int, int]:
    """Return how many planes, attempts and between-planes rows a file holds.

    A file holds as many planes as its ``n_saved`` attribute says, and the attempts
    and rows belonging to them, the sums of ``step_history/counts`` and
    ``step_stats/counts`` over them; any rows past those were allocated ahead or cut
    off by a hard stop, and are not part of the run.

    Parameters
    ----------
    file : h5py.File
        An open file holding at least one plane's datasets.

    Returns
    -------
    tuple of int
        The planes, the step-history attempts and the between-planes rows.
    """
    history = _group(file, "step_history")
    rows = _group(file, "step_stats")
    n_saved = int(cast("int", file.attrs["n_saved"]))
    attempts = int(np.sum(_dataset(history, "counts")[:n_saved], dtype=np.int64))
    entries = int(np.sum(_dataset(rows, "counts")[:n_saved], dtype=np.int64))
    return n_saved, attempts, entries


def _stored_retention(file: h5py.File) -> FieldRetention:
    """Return which planes of a file keep their field.

    Parameters
    ----------
    file : h5py.File
        An open file.

    Returns
    -------
    solprop.output.schema.FieldRetention
        The rule its ``fields`` attribute and ``field_planes`` dataset record.
    """
    fields = file.attrs["fields"]
    planes = (
        tuple(int(index) for index in np.asarray(_dataset(file, "field_planes")[()]))
        if "field_planes" in file
        else ()
    )
    return FieldRetention(cast("FieldsChoice", str(fields)), planes)


def _stored_shape(file: h5py.File) -> tuple[int, ...]:
    """Return the shape of one plane's field in a file, stored or not.

    Parameters
    ----------
    file : h5py.File
        An open file holding at least one plane.

    Returns
    -------
    tuple of int
        ``(n_modes, n_pol, n_omega)``, from the ``field_shape`` attribute.
    """
    return tuple(int(size) for size in np.asarray(file.attrs["field_shape"]))


def _write_json(file: h5py.File, text: str) -> None:
    """Write, or replace, the metadata document of an open file.

    Parameters
    ----------
    file : h5py.File
        The open file.
    text : str
        The document.
    """
    if _METADATA_KEY in file:
        del file[_METADATA_KEY]
    file.create_dataset(_METADATA_KEY, data=text, dtype=h5py.string_dtype())


_RECIPE_KEY = "recipe"
"""The top-level group of the recipe a run was built from (SEP 41, S2)."""


def _text(stored: object) -> str:
    """Return a stored string dataset's value as text.

    Parameters
    ----------
    stored : object
        What h5py read: bytes or str.

    Returns
    -------
    str
        The text.
    """
    return stored.decode("utf-8") if isinstance(stored, bytes) else str(stored)


def read_recipe(file: h5py.File) -> RecipeRecord | None:
    """Return the recipe an open file records, or ``None`` for a run from a script.

    The ``recipe`` group of an optical file or a scan file (SEP 41, S2).

    Parameters
    ----------
    file : h5py.File
        The open file.

    Returns
    -------
    RecipeRecord or None
        The record.
    """
    if _RECIPE_KEY not in file:
        return None
    group = _group(file, _RECIPE_KEY)
    files: dict[str, RecipeFile] = {}
    if "files" in group:
        stored = _group(group, "files")
        for name in stored:
            entry = stored[str(name)]
            data = (
                None
                if not isinstance(entry, h5py.Dataset) or entry.shape is None
                else bytes(np.asarray(entry[()], dtype=np.uint8))
            )
            files[str(name)] = RecipeFile(
                _text(entry.attrs["path"]), _text(entry.attrs["sha256"]), data
            )
    return RecipeRecord(
        int(cast("int", group.attrs["format"])),
        _text(_dataset(group, "authored")[()]),
        _text(_dataset(group, "resolved")[()]),
        files,
    )


def write_recipe(file: h5py.File, record: RecipeRecord) -> None:
    """Write the recipe group of an open file.

    Parameters
    ----------
    file : h5py.File
        The file, open for writing, without a recipe group.
    record : RecipeRecord
        The recipe.
    """
    group = file.create_group(_RECIPE_KEY)
    group.attrs["format"] = record.format
    group.attrs["authored_sha256"] = record.authored_sha256
    group.attrs["resolved_sha256"] = record.resolved_sha256
    group.create_dataset("authored", data=record.authored, dtype=h5py.string_dtype())
    group.create_dataset("resolved", data=record.resolved, dtype=h5py.string_dtype())
    files = group.create_group("files")
    for name, entry in record.files.items():
        dataset = (
            files.create_dataset(name, data=h5py.Empty("u1"))
            if entry.data is None
            else files.create_dataset(
                name, data=np.frombuffer(entry.data, dtype=np.uint8)
            )
        )
        dataset.attrs["path"] = entry.path
        dataset.attrs["sha256"] = entry.sha256


def _read_annotations(file: h5py.File) -> dict[str, JsonValue]:
    """Return the post-run annotations an open file carries.

    Parameters
    ----------
    file : h5py.File
        The open file.

    Returns
    -------
    dict
        Each annotation's JSON values, by name; empty when the file has none.
    """
    if _ANNOTATIONS_KEY not in file:
        return {}
    found: dict[str, JsonValue] = {}
    group = _group(file, _ANNOTATIONS_KEY)
    for name in group:
        stored = _dataset(group, str(name))[()]
        text = stored.decode("utf-8") if isinstance(stored, bytes) else str(stored)
        found[str(name)] = cast("JsonValue", json.loads(text))
    return found


def _read_json(file: h5py.File) -> str:
    """Return the metadata document of an open file.

    Parameters
    ----------
    file : h5py.File
        The open file.

    Returns
    -------
    str
        The document.

    Raises
    ------
    OutputError
        If the file carries no metadata, so it is not a solprop output file.
    """
    if _METADATA_KEY not in file:
        msg = (
            f"{file.filename} carries no {_METADATA_KEY!r} dataset, so it is not a "
            "solprop output file"
        )
        raise OutputError(msg)
    stored = _dataset(file, _METADATA_KEY)[()]
    return stored.decode("utf-8") if isinstance(stored, bytes) else str(stored)


@dataclass(frozen=True, eq=False)
class ResumeState(Displayed):
    """Everything a driver needs to carry on from a file it wrote earlier.

    Equality is by identity (``eq=False``) because the plane holds arrays.

    Attributes
    ----------
    metadata : solprop.output.schema.RunMetadata
        The record of the interrupted run, including its step schedule.
    plane : solprop.output.schema.SavedPlane
        The accepted terminal checkpoint when present, otherwise the last regular
        plane, with its field back in the **internal** FFT convention. Modal
        amplitudes remain in output units until the driver applies the weight at z.
    n_saved : int
        Number of regular planes stored. The next scheduled index is determined
        from the actual position, independently of this count.
    history : solprop.output.schema.StepHistory
        Every attempt of the whole run so far, concatenated.
    outcome : solprop.output.schema.RunOutcome or None
        Completion status and cumulative call ledgers; ``None`` when the file
        records none.
    checkpoint : solprop.output.schema.TerminalCheckpoint or None
        Separate accepted terminal state, preferred over the last regular save.
    march : solprop.output.schema.MarchSnapshot or None
        The Adams march's state where the run stopped, when the file kept one there.

    Examples
    --------
    >>> from solprop.output.hdf5 import ResumeState
    >>> ResumeState.__doc__.splitlines()[0]
    'Everything a driver needs to carry on from a file it wrote earlier.'
    """

    metadata: RunMetadata
    plane: SavedPlane
    n_saved: int
    history: StepHistory
    outcome: RunOutcome | None = None
    checkpoint: TerminalCheckpoint | None = None
    march: MarchSnapshot | None = None

    @property
    def interpolated(self) -> bool:
        """Whether the plane the file resumes from was interpolated inside a step.

        A landed plane is the end of an accepted step of the run; an interpolated one
        lies inside a step, and a resume from it needs ``resume="interpolated"``.
        The entrance is the initial field.

        Returns
        -------
        bool
            Whether the plane is not a state the march passed through.
        """
        if self.checkpoint is not None:
            return not self.checkpoint.landed
        if self.n_saved <= 1:
            return False
        history = self.history
        # The history records where each attempt started and how long it was.
        ends = (history.z + history.h)[history.accepted]
        return not bool(np.any(np.isclose(ends, self.plane.z, rtol=1e-12, atol=0.0)))

    def __repr__(self) -> str:
        """Return a one-line description: where a run stopped.

        Returns
        -------
        str
            For example ``ResumeState(12 planes saved, last SavedPlane(...))``.
        """
        return line(
            "ResumeState",
            [("", f"{self.n_saved} planes saved"), ("last", repr(self.plane))],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return f"ResumeState: {self.n_saved} planes saved", [
            ("last plane", repr(self.plane)),
            ("history", repr(self.history)),
            ("run", repr(self.metadata)),
        ]


class HDF5Output(PlaneStore):
    """Stream a run to one HDF5 file, opened once and extended plane by plane.

    Reading through ``__getitem__`` and
    :meth:`~solprop.output.schema.PlaneStore.fields` works while the handler is
    open and stops when it is closed, because the arrays live in the file rather than in
    memory; :func:`open_file` opens a finished file for reading afterwards.

    Parameters
    ----------
    path : pathlib.Path or str
        Where to write. An existing file is replaced, with a warning, when the run
        opens.
    compression : {None, 'gzip', 'lzf'}, optional
        Dataset compression. ``'lzf'`` is fast and modest; ``'gzip'`` is slower and
        smaller. Nothing else: a Blosc filter would need a plugin that solprop does not
        depend on.
    compression_opts : int or None, optional
        Level for ``'gzip'``, 0 to 9; not accepted for the other choices.
    append : bool, optional
        Continue the run the file already holds instead of replacing the file: the
        resume path, after :func:`resume` has handed the driver the last plane. The
        record given to ``open`` is compared with the file's
        (:meth:`~solprop.output.schema.RunMetadata.compare`) and refused when the
        layout or a compared value differs; the file's own record is kept, creation
        time included, and becomes the handler's ``metadata``, and the run's layout is
        read from the file so that the next plane is checked against the ones already
        there.
    readonly : bool, optional
        Open the file at once, for reading only: the handler comes back open, with
        the file's own record as its ``metadata``, and refuses ``open`` and ``save``.
        This is what :func:`open_file` builds.
    gated : bool, optional
        With ``append``: the driver's resume gate has already compared the records and
        decided to continue, forced or not, so the record given to ``open`` is not
        compared again (SEP 42 section 5.3). Without it, appending compares.

    Raises
    ------
    OutputError
        If the compression request is not one h5py can serve, if both ``append`` and
        ``readonly`` are asked for, or if a read-only file cannot be read.

    Examples
    --------
    >>> import numpy as np, tempfile
    >>> from pathlib import Path
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output import HDF5Output
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
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     handler = HDF5Output(Path(folder) / "run.h5", compression="lzf")
    ...     handler.open(metadata)
    ...     start = np.ones((1, 1, grid.n_omega), complex)
    ...     handler.save(SavedPlane(z=0.0, field=start))
    ...     shape = handler["field"].shape
    ...     handler.close()
    >>> shape
    (1, 1, 1, 2560)
    """

    def __init__(
        self,
        path: Path | str,
        *,
        compression: str | None = None,
        compression_opts: int | None = None,
        append: bool = False,
        readonly: bool = False,
        gated: bool = False,
    ) -> None:
        """Prepare a handler; only a read-only one opens its file now."""
        super().__init__()
        _check_compression(compression, compression_opts)
        if append and readonly:
            msg = "a handler continues a run or reads one, not both"
            raise OutputError(msg)
        self._path = Path(path)
        self._compression = compression
        self._compression_opts = compression_opts
        self._append_mode = append
        self._readonly = readonly
        self._gated = gated
        self._adopted: RunMetadata | None = None
        self._file: h5py.File | None = None
        self._writer_lock = WriterLock(self._path)
        self._suspended_identity: tuple[int, int] | None = None
        self._history_length = 0
        self._step_stats_length = 0
        # The open datasets the writer fills, by accessor key, and how many entries
        # each kind has room for: plane-indexed, step-history and step-statistics rows.
        self._handles: dict[str, h5py.Dataset] = {}
        self._plane_room = 0
        self._history_room = 0
        self._rows_room = 0
        if readonly:
            self._open_readonly()

    @property
    def path(self) -> Path:
        """Return where the file is written.

        Returns
        -------
        pathlib.Path
            The path given to the constructor.
        """
        return self._path

    def _state_text(self) -> str:
        """Return the state of the file, for printing, without raising.

        Returns
        -------
        str
            Open, closed or read-only, or a note that the file is gone.
        """
        if self._file is None and not self._path.exists():
            return (
                "file missing" if self._n_saved or self._closed else "not written yet"
            )
        state = "closed" if self._file is None else "open"
        return f"{state}, read-only" if self._readonly else state

    def __repr__(self) -> str:
        """Return a one-line description: the file, what it holds, and its state.

        Returns
        -------
        str
            For example ``HDF5Output('run.h5', 201 planes, closed)``.
        """
        return line(
            "HDF5Output",
            [
                ("", repr(self._path.name)),
                ("", f"{self._n_saved} planes"),
                ("", self._state_text()),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows, the file's path first.
        """
        title, rows = super()._describe()
        return title, [("file", tail(str(self._path))), *rows]

    def open(self, metadata: RunMetadata, *, force: bool = False) -> None:
        """Begin a run, recording what produced it; refused on a read-only handler.

        Parameters
        ----------
        metadata : RunMetadata
            The record of the run about to start.
        force : bool, optional
            Whether to overwrite metadata already held for a different run.

        Raises
        ------
        OutputError
            If the handler was built with ``readonly=True``.
        """
        self._refuse_readonly()
        super().open(metadata, force=force)
        if self._adopted is not None:
            # The session holds the file's record, which the exact checks of a
            # suspend, a reopen and a snapshot compare the file with (SEP 42 §5.3).
            self._metadata = self._adopted

    def save(self, plane: SavedPlane) -> None:
        """Store one saved plane and flush it to disk; refused on a read-only handler.

        The flush is what makes a hard stop -- a SIGKILL, an out-of-memory kill, a lost
        node -- lose at most the plane being written: the plane's data is flushed before
        the file's ``n_saved`` count is raised, so a plane cut off half-way is not
        counted, and readers do not see it. An ordinary interruption
        (``KeyboardInterrupt``, an error) never needed it: the run loop closes the
        handler in a ``finally`` and closing writes everything. One flush per plane
        costs nothing against the steps between planes.

        Parameters
        ----------
        plane : SavedPlane
            The position, field, statistics and step history to store.

        Raises
        ------
        OutputError
            If the handler was built with ``readonly=True``.
        """
        self._refuse_readonly()
        super().save(plane)
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        """Close the owned handle, including a failed post-run annotation handle.

        Repeated calls are harmless. A failed close retains the handle and writer
        ownership for a later retry, even if propagation already closed the store.

        Raises
        ------
        OutputError
            If the handle cannot be closed safely.
        """
        self._release()
        self._closed = True

    def save_march(self, snapshot: MarchSnapshot) -> None:
        """Keep the Adams march's state at the position just saved, for a resume.

        One snapshot is kept, rewritten in place at every call, beside the planes and
        outside the record: it says where the march stood, not what was computed, and
        a file without one resumes by starting the history afresh.

        Parameters
        ----------
        snapshot : solprop.output.schema.MarchSnapshot
            The march's state at the last saved plane or terminal checkpoint.

        Raises
        ------
        OutputError
            If the handler was built with ``readonly=True`` or is not open.
        """
        self._refuse_readonly()
        if self._file is None:
            raise OutputError("save_march called before open")
        _write_march(self._file.require_group(_MARCH_KEY), snapshot)
        self._file.flush()

    def _refuse_readonly(self) -> None:
        """Refuse to write through a handler that only reads.

        Raises
        ------
        OutputError
            If the handler was built with ``readonly=True``.
        """
        if self._readonly:
            msg = (
                f"{self._path} is open for reading only; build a handler with "
                "append=True to continue the run it holds"
            )
            raise OutputError(msg)

    def _open_readonly(self) -> None:
        """Open the file for reading, adopting its state and its record.

        Raises
        ------
        OutputError
            If the file cannot be opened, or carries no metadata.
        """
        try:
            file = h5py.File(self._path, "r")
        except OSError as error:
            msg = f"cannot read the output file {self._path}: {error}"
            raise OutputError(msg) from error
        try:
            self._metadata = RunMetadata.from_json(_read_json(file))
        except OutputError:
            file.close()
            raise
        self._file = file
        self._adopt_state(file)
        self._annotations = _read_annotations(file)
        self._recipe = read_recipe(file)

    def _open_file(self) -> h5py.File:
        """Open the file once: fresh, or read-write to continue what it holds.

        Returns
        -------
        h5py.File
            The open handle, which the handler keeps until it is closed.

        Raises
        ------
        OutputError
            If appending and the file does not exist or cannot be opened.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._writer_lock.acquire()
        try:
            return self._open_owned_file()
        except BaseException:
            if self._file is None:
                self._writer_lock.release()
            raise

    def _open_owned_file(self) -> h5py.File:
        """Open a file after acquiring its independent writer lease.

        Returns
        -------
        h5py.File
            Owned live handle.
        """
        if self._append_mode:
            try:
                file = h5py.File(self._path, "r+")
            except OSError as error:
                msg = f"cannot append to {self._path}: {error}"
                raise OutputError(msg) from error
            try:
                self._adopt_state(file)
                self._annotations = _read_annotations(file)
                self._recipe = read_recipe(file)
                if "z" in file:
                    self._adopt_handles(file)
            except BaseException as error:
                self._file = file
                self._handles = {}
                try:
                    file.close()
                except (OSError, RuntimeError) as cleanup:
                    error.add_note(f"rejected append handle could not close: {cleanup}")
                else:
                    self._file = None
                raise
            return file
        if self._path.exists():
            logger.warning("replacing the existing output file %s", self._path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        return h5py.File(self._path, "w")

    def suspend(self) -> None:
        """Close the HDF5 handle while retaining the live writer and its ownership.

        Raises
        ------
        OutputError
            If the writer is not open or its handle cannot be closed safely.
        """
        self._refuse_readonly()
        file = self._require_file()
        try:
            file.flush()
            stat = self.path.stat()
            file.close()
        except (OSError, RuntimeError) as error:
            raise OutputError(f"cannot suspend optical writer: {error}") from error
        self._file = None
        self._handles = {}
        self._suspended_identity = (stat.st_dev, stat.st_ino)

    def reopen(self) -> None:
        """Rebind a suspended writer without beginning a new execution.

        Raises
        ------
        OutputError
            If the file changed while suspended or cannot be reopened safely.
        """
        if self._suspended_identity is None:
            raise OutputError("optical writer is not suspended")
        try:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) != self._suspended_identity:
                raise OutputError("suspended optical file was replaced")
            file = h5py.File(self.path, "r+")
            try:
                metadata = RunMetadata.from_json(_read_json(file))
                if (
                    metadata.configuration_hash()
                    != self._require_metadata().configuration_hash()
                    or _stored_counts(file)
                    != (self._n_saved, self._history_length, self._step_stats_length)
                    or _read_outcome(file) != self._outcome
                ):
                    raise OutputError("suspended optical checkpoint changed")
                self._adopt_handles(file)
            except BaseException as error:
                self._handles = {}
                self._file = file
                try:
                    file.close()
                except (OSError, RuntimeError) as cleanup:
                    error.add_note(
                        f"rejected optical handle could not close: {cleanup}"
                    )
                else:
                    self._file = None
                raise
            self._file = file
            self._suspended_identity = None
        except OutputError:
            raise
        except (OSError, RuntimeError) as error:
            raise OutputError(f"cannot reopen optical writer: {error}") from error

    def _adopt_state(self, file: h5py.File) -> None:
        """Take the plane count, the history length and the layout from the file.

        Parameters
        ----------
        file : h5py.File
            The file being continued.
        """
        metadata = RunMetadata.from_json(_read_json(file))
        self._outcome = _read_outcome(file)
        self._resume_outcome_pending = self._append_mode and self._outcome is not None
        self._terminal_checkpoint = _read_checkpoint(file, metadata)
        self._checkpoint_pending = (
            self._append_mode and self._terminal_checkpoint is not None
        )
        if "z" not in file:
            return
        self._n_saved, self._history_length, self._step_stats_length = _stored_counts(
            file
        )
        self._retention = _stored_retention(file)
        self._field_shape = _stored_shape(file)
        rows = _group(file, "step_stats")
        self._layout = PlaneLayout(
            field_shape=self._field_shape,
            stat_names=tuple(sorted(str(name) for name in _group(file, "stats"))),
            has_noise_phase="noise_phase" in file,
            step_stat_names=tuple(
                sorted(
                    str(name)
                    for name in rows
                    if str(name) not in ("counts", *STEP_STATS_INDEX)
                )
            ),
            retention=self._retention,
        )

    def _write_outcome(
        self, outcome: RunOutcome | None, checkpoint: TerminalCheckpoint | None
    ) -> None:
        """Flush execution status and a separate accepted checkpoint to disk.

        Parameters
        ----------
        outcome : RunOutcome or None
            Execution status and accounting, or no status to write.
        checkpoint : TerminalCheckpoint or None
            Accepted field to store, or removal of an existing terminal state.
        """
        self._refuse_readonly()
        file = self._require_file()
        if checkpoint is not None:
            pending = "_terminal_checkpoint_writing"
            if pending in file:
                del file[pending]
            group = file.create_group(pending)
            _write_checkpoint(group, checkpoint, self._require_metadata())
            file.flush()
            if _CHECKPOINT_KEY in file:
                del file[_CHECKPOINT_KEY]
            file.move(pending, _CHECKPOINT_KEY)
        elif _CHECKPOINT_KEY in file:
            del file[_CHECKPOINT_KEY]
        if outcome is not None:
            if _OUTCOME_KEY in file:
                del file[_OUTCOME_KEY]
            file.create_dataset(
                _OUTCOME_KEY,
                data=json.dumps(asdict(outcome), sort_keys=True),
                dtype=h5py.string_dtype(),
            )
        file.flush()

    def _require_file(self) -> h5py.File:
        """Return the open file, refusing once the handler is closed.

        Returns
        -------
        h5py.File
            The open handle.

        Raises
        ------
        OutputError
            If the handler was never opened, or has been closed.
        """
        if self._file is None:
            action = "reopen it" if self._closed else "call open(metadata) first"
            msg = f"the output file {self._path} is not open; {action}"
            raise OutputError(msg)
        return self._file

    def _write_metadata(self, metadata: RunMetadata) -> None:
        """Open the file if it is not open yet, and write the run record.

        When appending, the record the file already carries is kept -- its creation
        time included -- and becomes the session's; the one offered is compared with
        it, unless the driver's gate has done so (``gated``).

        Parameters
        ----------
        metadata : RunMetadata
            The record to store.

        Raises
        ------
        OutputError
            If appending without the gate, and the file's layout or a compared value
            differs; the message names what changed.
        """
        if self._file is None:
            self._file = self._open_file()
        file = self._file
        if self._append_mode:
            stored = RunMetadata.from_json(_read_json(file))
            if not self._gated:
                comparison = stored.compare(metadata)
                if not comparison.continues:
                    msg = (
                        f"{self._path} holds a different run and cannot be appended "
                        f"to; {comparison.summary()}"
                    )
                    raise OutputError(msg)
            self._adopted = stored
            logger.debug("continuing %s from plane %d", self._path, self._n_saved)
            return
        file.attrs["schema_version"] = metadata.schema_version
        file.attrs["spectral_convention"] = metadata.spectral_convention
        file.attrs["field_layout"] = metadata.field_layout
        file.attrs["configuration_hash"] = metadata.configuration_hash()
        file.attrs["simulation_kind"] = metadata.simulation_kind
        file.attrs["grid_kind"] = metadata.grid.kind
        # A modal amplitude is the square root of a power, which on a run uniform
        # in y is a power per metre of y; area runs keep the
        # attribute they always had.
        file.attrs["field_units"] = (
            str(metadata.spatial["stored_field_units"])
            if metadata.spatial is not None
            else "sqrt(W/m)"
            if metadata.transverse_measure == "per_unit_width"
            else "sqrt(W)"
        )
        file.attrs["transverse_spectral_axis"] = (
            "mode"
            if metadata.spatial is None
            else "k_r"
            if metadata.simulation_kind == "radial"
            else "p,k_y (p fastest)"
            if metadata.simulation_kind == "slab"
            else "k_x,k_y (x fastest)"
        )
        file.attrs["created"] = metadata.created
        integration = metadata.configuration.get("plasma_integration")
        if isinstance(integration, Mapping):
            # SEP 18, Q9: how the plasma was integrated, readable without
            # translating the run again. Absent for the plain trapezoid.
            file.attrs["plasma_integration"] = str(integration["policy"])
            file.attrs["plasma_integration_half_width"] = int(
                cast("int", integration["half_width"])
            )
            file.attrs["plasma_integration_theta_b"] = float(
                cast("float", integration["theta_b"])
            )
            file.attrs["plasma_integration_taps"] = np.asarray(
                integration["taps"], dtype=np.float64
            )
            file.attrs["plasma_integration_certified_error"] = float(
                cast("float", integration["certified_error"])
            )
        if metadata.backend is not None:
            # The one thing a reader wants without parsing the whole record: which
            # backend produced the file. The rest of the group is in `meta/json`.
            file.attrs["backend"] = metadata.backend.name
        _write_json(file, metadata.to_json())
        logger.debug("opened %s for %s", self._path, metadata.configuration_hash()[:12])

    def _new_dataset(
        self,
        parent: h5py.Group,
        key: str,
        shape: tuple[int, ...],
        dtype: object,
        room: int,
    ) -> h5py.Dataset:
        """Create one extendable dataset with the handler's compression, and keep it.

        Parameters
        ----------
        parent : h5py.Group
            Where to create it.
        key : str
            Its accessor key: the dataset's name, prefixed by its group's.
        shape : tuple of int
            Shape of one entry, without the leading plane or attempt axis.
        dtype : object
            NumPy dtype of the entries.
        room : int
            Entries to allocate along the leading axis, zero-filled until written.

        Returns
        -------
        h5py.Dataset
            The extendable dataset, also kept as the handle the writer fills.
        """
        chunks = (1, *shape) if shape else (_COLUMN_CHUNK,)
        dataset = parent.create_dataset(
            key.rpartition("/")[2],
            shape=(room, *shape),
            maxshape=(None, *shape),
            chunks=chunks,
            dtype=dtype,
            compression=self._compression,
            compression_opts=self._compression_opts,
        )
        self._handles[key] = dataset
        return dataset

    def _adopt_handles(self, file: h5py.File) -> None:
        """Keep the datasets of a file being continued.

        Parameters
        ----------
        file : h5py.File
            The file, open for writing, holding at least one plane's datasets.
        """
        for key in ("z", "field", "noise_phase"):
            if key in file:
                self._handles[key] = _dataset(file, key)
        # A file of this schema holds all three groups from its first plane.
        for group in ("stats", "step_history", "step_stats"):
            for name in _group(file, group):
                key = f"{group}/{name}"
                self._handles[key] = _dataset(file, key)
        self._plane_room = int(self._handles["z"].shape[0])
        self._history_room = int(self._handles["step_history/z"].shape[0])
        self._rows_room = (
            int(self._handles["step_stats/z"].shape[0])
            if "step_stats/z" in self._handles
            else 0
        )

    def _create_datasets(
        self,
        plane: SavedPlane,
        stored: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    ) -> None:
        """Lay out every dataset from the shapes of the first plane.

        Parameters
        ----------
        plane : SavedPlane
            The first plane, which fixes the statistics.
        stored : numpy.ndarray or None
            Its field in the optics convention, which fixes the field's shape and
            whether a noise phase is stored; ``None`` on a field-free plane, whose
            field datasets wait for the first plane that keeps one.
        """
        file = self._require_file()
        # The count goes first: a file whose datasets exist always says how many of
        # their planes are real, even if it is stopped while they are being created.
        file.attrs["n_saved"] = 0
        retention = self.retention
        file.attrs["fields"] = retention.fields
        file.attrs["field_shape"] = np.asarray(self.field_shape, dtype=np.int64)
        if retention.fields == "planes":
            file.create_dataset(
                "field_planes", data=np.asarray(retention.field_planes, dtype=np.int64)
            )
        room = max(self._planned, 1)
        self._plane_room = room
        self._new_dataset(file, "z", (), np.float64, room)
        if stored is not None:
            self._create_field_datasets(plane, stored)
        statistics = file.create_group("stats")
        for name, value in plane.stats.items():
            self._new_dataset(
                statistics, f"stats/{name}", np.shape(value), np.float64, room
            )
        history = file.create_group("step_history")
        self._new_dataset(history, "step_history/counts", (), np.int64, room)
        for name, dtype in HISTORY_FIELDS.items():
            self._new_dataset(history, f"step_history/{name}", (), dtype, 0)
        # The between-planes channel's index columns exist from the first plane, so
        # that `counts` has one entry per plane whether or not that plane had rows;
        # the named statistics wait for the first plane that does, because the
        # entrance plane at z = 0 never has any.
        rows = file.create_group("step_stats")
        self._new_dataset(rows, "step_stats/counts", (), np.int64, room)
        for name, dtype in STEP_STATS_INDEX.items():
            self._new_dataset(rows, f"step_stats/{name}", (), dtype, 0)

    def _create_field_datasets(
        self, plane: SavedPlane, stored: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> None:
        """Create the field dataset, and the noise phase's, at the first kept field.

        Sized for the planes the run reserved when it keeps every field, and growing
        with them; for exactly the named planes when it keeps some.

        Parameters
        ----------
        plane : SavedPlane
            The first plane that keeps its field.
        stored : numpy.ndarray
            Its field in the optics convention.
        """
        file = self._require_file()
        retention = self.retention
        room = (
            self._plane_room
            if retention.fields == "all"
            else len(retention.field_planes)
        )
        field = self._new_dataset(file, "field", stored.shape, np.complex128, room)
        field.attrs["spectral_convention"] = "optics"
        assert self.metadata is not None
        field.attrs["power_convention"] = self.metadata.power_convention
        if plane.noise_phase is not None:
            phase = self._new_dataset(
                file, "noise_phase", stored.shape, np.float64, room
            )
            phase.attrs["spectral_convention"] = "fft"

    def _grow(self, kind: str, needed: int, room: int) -> int:
        """Make room for `needed` entries in every dataset of one kind, by doubling.

        Parameters
        ----------
        kind : {'plane', 'history', 'rows'}
            Which datasets: the plane-indexed ones, the step-history columns or the
            between-planes statistics columns.
        needed : int
            Entries the datasets must hold.
        room : int
            Entries they hold now.

        Returns
        -------
        int
            The entries they hold after growing: `room` when it was enough, otherwise
            the largest of `needed`, twice `room` and, for planes, the reserved count.
        """
        if needed <= room:
            return room
        grown = max(needed, 2 * room, self._planned if kind == "plane" else 0)
        every_field = self.retention.fields == "all"
        for key, dataset in self._handles.items():
            if _kind_of(key, every_field=every_field) == kind:
                dataset.resize(grown, axis=0)
        logger.debug("%s grew its %s datasets to %d entries", self._path, kind, grown)
        return grown

    def _field_room(self, slot: int) -> None:
        """Make room for the field of row `slot` under a rule that keeps some planes.

        Every field is grown with the planes (:meth:`_grow`). The named planes' fields
        are laid out for all of them at once, but a clean close trims them to the rows
        written, so a run continued after a close grows them back here, to every named
        plane, before its next kept field.

        Parameters
        ----------
        slot : int
            The row the next kept field is written to.
        """
        if self.retention.fields == "all":
            return
        room = max(slot + 1, len(self.retention.field_planes))
        for key in ("field", "noise_phase"):
            dataset = self._handles.get(key)
            if dataset is not None and dataset.shape[0] < room:
                dataset.resize(room, axis=0)

    def _append(
        self,
        plane: SavedPlane,
        stored: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    ) -> None:
        """Write one plane into the file, then count it.

        The datasets are created on the first call. The plane's data is written and
        flushed before the file's ``n_saved`` count is raised, so that a stop part-way
        leaves the count at the last complete plane.

        Parameters
        ----------
        plane : SavedPlane
            The plane, whose field is still in the internal convention.
        stored : numpy.ndarray or None
            That field converted to the optics convention; ``None`` on a field-free
            plane, which stores none.
        """
        file = self._require_file()
        if not self._handles:
            self._create_datasets(plane, stored)
        handles = self._handles
        index = self._n_saved
        self._plane_room = self._grow("plane", index + 1, self._plane_room)
        handles["z"][index] = plane.z
        if stored is not None:
            if "field" not in handles:
                self._create_field_datasets(plane, stored)
            slot = self._fields_stored()
            self._field_room(slot)
            handles["field"][slot] = stored
            if plane.noise_phase is not None:
                handles["noise_phase"][slot] = plane.noise_phase
        for name, value in plane.stats.items():
            handles[f"stats/{name}"][index] = value
        attempts = plane.step_history.n_steps
        handles["step_history/counts"][index] = attempts
        start = self._history_length
        self._history_room = self._grow("history", start + attempts, self._history_room)
        if attempts:
            for name in HISTORY_FIELDS:
                handles[f"step_history/{name}"][start : start + attempts] = getattr(
                    plane.step_history, name
                )
        rows_end = self._append_rows(file, plane.step_stats, index)
        file.flush()
        file.attrs["n_saved"] = index + 1
        # Advanced only once the plane is counted, so that a write that fails
        # part-way leaves the next plane to overwrite it.
        self._history_length = start + attempts
        self._step_stats_length = rows_end

    def _append_rows(self, file: h5py.File, rows: StepStats, index: int) -> int:
        """Write one plane's between-planes statistics rows into the open file.

        Parameters
        ----------
        file : h5py.File
            The open file.
        rows : solprop.output.schema.StepStats
            The rows of the plane being appended, possibly none.
        index : int
            The plane's index.

        Returns
        -------
        int
            The rows the file holds with this plane's.
        """
        group = _group(file, "step_stats")
        handles = self._handles
        for name in rows.names:
            key = f"step_stats/{name}"
            if key not in handles:
                self._new_dataset(
                    group,
                    key,
                    np.shape(rows.values[name])[1:],
                    np.float64,
                    self._rows_room,
                )
        handles["step_stats/counts"][index] = rows.n_rows
        start = self._step_stats_length
        self._rows_room = self._grow("rows", start + rows.n_rows, self._rows_room)
        if rows.n_rows:
            for name in (*STEP_STATS_INDEX, *self.step_stat_names):
                handles[f"step_stats/{name}"][start : start + rows.n_rows] = (
                    getattr(rows, name)
                    if name in STEP_STATS_INDEX
                    else rows.values[name]
                )
        return start + rows.n_rows

    def _trim(self, file: h5py.File) -> None:
        """Cut every dataset the writer filled down to what the file counts.

        Run on a clean close, so that a finished file has no unused rows and its
        datasets' extents are the counts. The lengths are the file's own
        (:func:`_stored_counts`), so a plane whose write failed part-way, and was
        never counted, is cut off with the rest.

        Parameters
        ----------
        file : h5py.File
            The open file.
        """
        planes, attempts, entries = _stored_counts(file)
        lengths = {
            "plane": planes,
            "field": self._fields_stored(),
            "history": attempts,
            "rows": entries,
        }
        every_field = self.retention.fields == "all"
        for key, dataset in self._handles.items():
            length = lengths[_kind_of(key, every_field=every_field)]
            if dataset.shape[0] != length:
                dataset.resize(length, axis=0)

    def _read(self, group: str, name: str) -> np.ndarray:
        """Return one stored array, read out of the open file.

        Parameters
        ----------
        group : str
            ``''``, ``'stats'``, ``'step_history'`` or ``'step_stats'``.
        name : str
            Name within the group.

        Returns
        -------
        numpy.ndarray
            The stored array.

        Raises
        ------
        OutputError
            If the file is not open, or holds nothing of that name.
        """
        file = self._require_file()
        if group == "step_stats" and (
            group not in file
            or (name not in _group(file, group) and not self.step_stat_names)
        ):
            # A run that recorded no periodic subset, or a file with no plane yet,
            # reads back as an empty channel rather than an error, so that a reader
            # can ask for it unconditionally and fall back to the planes.
            return np.zeros(0, dtype=STEP_STATS_INDEX.get(name, np.float64))
        if group and group not in file:
            msg = f"the file holds no {group!r} group yet; save a plane first"
            raise OutputError(msg)
        parent = file if group == "" else _group(file, group)
        if name not in parent:
            where = group or "the file"
            msg = f"nothing named {name!r} in {where}; it holds {sorted(parent)}"
            raise OutputError(msg)
        length = {
            "plane": self._n_saved,
            "field": self._fields_stored(),
            "history": self._history_length,
            "rows": self._step_stats_length,
        }[
            _kind_of(
                f"{group}/{name}" if group else name,
                every_field=self.retention.fields == "all",
            )
        ]
        return np.asarray(_dataset(parent, name)[:length])

    def _read_field(self, slot: int) -> np.ndarray:
        """Return one stored field by its row, reading that row alone.

        Parameters
        ----------
        slot : int
            The row in the ``field`` dataset.

        Returns
        -------
        numpy.ndarray
            The stored field, optics convention.
        """
        return np.asarray(_dataset(self._require_file(), "field")[slot])

    def _write_recipe(self, record: RecipeRecord) -> None:
        """Write the recipe group into the open file (SEP 41, S2).

        Parameters
        ----------
        record : RecipeRecord
            The recipe.
        """
        self._refuse_readonly()
        write_recipe(self._require_file(), record)

    def _write_annotation(self, name: str, text: str) -> None:
        """Write one post-run annotation into the file, replacing one of that name.

        A closed handler reopens its file for the write and closes it again, so a
        run's driver can annotate the file after the run has closed it.

        Parameters
        ----------
        name : str
            The annotation's name.
        text : str
            Its JSON document.

        Raises
        ------
        OutputError
            If the handler only reads, or the file cannot be opened for writing.
        """
        self._refuse_readonly()
        if self._suspended_identity is not None:
            raise OutputError("cannot annotate a suspended optical writer")
        if self._file is not None:
            _write_annotation(self._file, name, text)
            self._file.flush()
            return
        try:
            self._writer_lock.acquire()
            failure: BaseException | None = None
            try:
                self._file = h5py.File(self._path, "r+")
                _write_annotation(self._file, name, text)
            except BaseException as error:
                failure = error
                raise
            finally:
                try:
                    self._release()
                except BaseException as cleanup:
                    if failure is None:
                        raise
                    failure.add_note(
                        f"annotation handle closure also failed: {cleanup}"
                    )
        except OSError as error:
            msg = f"cannot annotate {self._path}: {error}"
            raise OutputError(msg) from error

    def _release(self) -> None:
        """Trim what the writer filled to its counts, and close the file handle."""
        if self._file is None:
            self._suspended_identity = None
            self._writer_lock.release()
            return
        failure: BaseException | None = None
        try:
            if self._handles:
                self._trim(self._file)
        except BaseException as error:
            failure = error
            raise
        finally:
            self._handles = {}
            try:
                self._file.close()
            except (OSError, RuntimeError) as error:
                # Retain ownership until a successful retry or process exit.
                if failure is None:
                    raise OutputError(
                        f"cannot close optical writer: {error}"
                    ) from error
                failure.add_note(f"optical handle closure also failed: {error}")
            else:
                self._file = None
                self._suspended_identity = None
                self._writer_lock.release()


def _write_annotation(file: h5py.File, name: str, text: str) -> None:
    """Write, or replace, one post-run annotation of an open file.

    Parameters
    ----------
    file : h5py.File
        The file, open for writing.
    name : str
        The annotation's name.
    text : str
        Its JSON document.
    """
    group = file.require_group(_ANNOTATIONS_KEY)
    if name in group:
        del group[name]
    group.create_dataset(name, data=text, dtype=h5py.string_dtype())


def _stored_metadata(path: Path | str) -> tuple[RunMetadata, str]:
    """Return the record a file carries and the digest stored beside it.

    Parameters
    ----------
    path : pathlib.Path or str
        The file to read.

    Returns
    -------
    tuple
        The :class:`solprop.output.schema.RunMetadata` and the ``configuration_hash``
        attribute the writer stored, which are checked against each other.

    Raises
    ------
    OutputError
        If the file cannot be opened, carries no metadata, or its stored digest does
        not match the record beside it.
    """
    try:
        with h5py.File(path, "r") as file:
            metadata = RunMetadata.from_json(_read_json(file))
            stamped = str(file.attrs.get("configuration_hash", ""))
    except OSError as error:
        msg = f"cannot read the output file {path}: {error}"
        raise OutputError(msg) from error
    if stamped and stamped != metadata.configuration_hash():
        msg = (
            f"{path} is inconsistent: its configuration_hash attribute is {stamped} "
            f"but its metadata digests to {metadata.configuration_hash()}"
        )
        raise OutputError(msg)
    return metadata, stamped


def _compare_cache(
    path: Path | str, metadata: RunMetadata
) -> tuple[RunMetadata, RecordComparison] | None:
    """Return a file's record and its comparison with a requested one.

    The checks that refuse before any comparison come first: the output schema and
    the stamp (:func:`_stored_metadata`), the gated plasma benchmark and the general
    modal accessors.

    Parameters
    ----------
    path : pathlib.Path or str
        The file to check.
    metadata : RunMetadata
        The record of the run about to start.

    Returns
    -------
    tuple or None
        The stored record and the comparison, or ``None`` when there is no file.
    """
    if not Path(path).exists():
        logger.debug("no cache at %s; starting from the beginning", path)
        return None
    stored, _ = _stored_metadata(path)
    _check_plasma_benchmark(stored)
    _check_accessors(path, stored, metadata)
    return stored, stored.compare(metadata)


def _resume_refusal(
    path: Path | str, stored: RunMetadata, comparison: RecordComparison
) -> str:
    """Return the message refusing a resume across a comparison's differences.

    Parameters
    ----------
    path : pathlib.Path or str
        The file.
    stored : RunMetadata
        Its record.
    comparison : RecordComparison
        What separates it from the requested record.

    Returns
    -------
    str
        What differs, and what to do instead: a new file for a layout change, a new
        segment or ``resume="force"`` for a compared one.
    """
    text = f"{path} cannot be resumed; {comparison.summary()}"
    noise = stored.noise
    if (
        noise is not None
        and noise.seed is not None
        and any(change.path[:1] == ("noise",) for change in comparison.refused)
    ):
        text += (
            f"\nThe stored run drew its {noise.model} noise from seed {noise.seed}: "
            f"pass seed={noise.seed} to draw it again."
        )
    if any(change.path[:1] == ("materials",) for change in comparison.refused):
        text += (
            "\nNote that the material manifest differs: an optmat model is "
            "compared by the digest of its numbers, and a material policy as recorded."
        )
    if comparison.layout:
        return (
            text + "\nThe layout of the stored arrays differs, which nothing continues "
            "across: write a new file."
        )
    return (
        text + "\nTo change the physics, launch a new segment from a saved plane; to "
        'continue in this file regardless, pass resume="force".'
    )


def _stored_stat_shapes(path: Path | str) -> dict[str, tuple[int, ...]]:
    """Return the shape one plane's entry of each stored statistic has.

    Parameters
    ----------
    path : pathlib.Path or str
        The file.

    Returns
    -------
    dict
        Each ``stats/<name>`` dataset's shape without its plane axis; empty for a
        file that has saved no plane.
    """
    with h5py.File(path, "r") as file:
        stats = file.get("stats")
        if not isinstance(stats, h5py.Group):
            return {}
        return {
            str(name): tuple(int(size) for size in dataset.shape[1:])
            for name, dataset in stats.items()
            if isinstance(dataset, h5py.Dataset)
        }


def check_cache(path: Path | str, metadata: RunMetadata) -> bool:
    """Say whether a file may be continued as the run ``metadata`` describes.

    The resume gate (SEP 42). The two records are compared path by path
    (:meth:`~solprop.output.schema.RunMetadata.compare`): the layout of the stored
    arrays and every compared value -- physics and launch parameters -- must agree,
    numbers to a rounding-level tolerance; the numerics (``rtol``, the integrator) may
    differ, and provenance (package and optmat versions, model names) is ignored. A
    refusal names what differs. The checks that depend on the file's state, such as
    whether any work is left, are the driver's.

    Parameters
    ----------
    path : pathlib.Path or str
        The file to check.
    metadata : RunMetadata
        The record of the run about to start.

    Returns
    -------
    bool
        ``True`` when the file may be continued; ``False`` when there is no file to
        resume from.

    Raises
    ------
    OutputError
        If the file's layout or a compared value differs, or it was written at
        another output schema; or if the file's general modal record names an
        accessor this build no longer admits (``_check_accessors``). An optmat model
        whose numbers changed is named by its material and kind.

    Examples
    --------
    >>> from solprop.output.hdf5 import check_cache
    >>> from solprop.output.schema import _example_metadata
    >>> check_cache("no-such-file.h5", _example_metadata())
    False
    """
    found = _compare_cache(path, metadata)
    if found is None:
        return False
    stored, comparison = found
    if comparison.continues:
        return True
    raise OutputError(_resume_refusal(path, stored, comparison))


def _check_plasma_benchmark(metadata: RunMetadata) -> None:
    """Keep the historical gated equation out of every continuation path.

    Parameters
    ----------
    metadata : RunMetadata
        Stored run identity, including any private benchmark source label.

    Raises
    ------
    OutputError
        If the output was produced by the non-resumable gated benchmark.
    """
    if metadata.configuration.get("_benchmark_plasma_source"):
        raise OutputError("gated plasma benchmark output cannot be resumed")


def _check_accessors(
    path: Path | str, stored: RunMetadata, metadata: RunMetadata
) -> None:
    """Refuse a resume whose file names an accessor this build cannot supply.

    The published accessor list is **not** in the resume key: it says what the
    package version that wrote the file could compute, not what the run computed,
    and keeping it in the key would make every accessor this package gains
    invalidate every general modal file ever written. What
    matters is the one direction that is a real loss -- a file naming a capability
    this build has dropped -- so the list is checked as a **subset** here instead.
    A file written before an accessor was added resumes; one written by a build
    that had an accessor this one does not is refused by the name it is missing.

    Parameters
    ----------
    path : pathlib.Path or str
        The file, for the message.
    stored : solprop.output.schema.RunMetadata
        The record the file carries.
    metadata : solprop.output.schema.RunMetadata
        The record of the run about to start.

    Raises
    ------
    OutputError
        Naming the accessors this build cannot supply.
    """
    if stored.modal is None or metadata.modal is None:
        return
    theirs = stored.modal.get("accessors")
    mine = metadata.modal.get("accessors")
    if not isinstance(theirs, list) or not isinstance(mine, list):
        return
    missing = sorted(str(one) for one in theirs if one not in mine)
    if not missing:
        return
    msg = (
        f"{path} was written by a build whose general modal basis admitted "
        f"{', '.join(missing)}, which this build does not: a resume would silently "
        "drop a statistic the file's own record says the run could ask for. The "
        "accessor list is not in the resume key -- a file written before an "
        "accessor existed resumes -- so this is a capability that was removed, not "
        "one that was added"
    )
    raise OutputError(msg)


def _last_history(file: h5py.File, index: int) -> tuple[StepHistory, StepHistory]:
    """Return the whole step history and the part belonging to one plane.

    Parameters
    ----------
    file : h5py.File
        The open file.
    index : int
        Index of the plane whose slice is wanted.

    Returns
    -------
    tuple
        The whole history and the slice, both as
        :class:`solprop.output.schema.StepHistory`.
    """
    group = _group(file, "step_history")
    counts = np.asarray(_dataset(group, "counts")[: index + 1], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    # The history up to the end of this plane: rows past it were allocated ahead,
    # or written for a plane a hard stop cut off before it was counted.
    columns: Mapping[str, np.ndarray] = {
        name: np.asarray(_dataset(group, name)[: int(offsets[-1])])
        for name in HISTORY_FIELDS
    }
    piece = slice(int(offsets[index]), int(offsets[index + 1]))
    whole = StepHistory(**dict(columns))
    part = StepHistory(**{name: columns[name][piece] for name in HISTORY_FIELDS})
    return whole, part


def _last_step_stats(file: h5py.File, index: int) -> StepStats:
    """Return the between-planes statistics rows belonging to one plane.

    Parameters
    ----------
    file : h5py.File
        The open file.
    index : int
        Index of the plane whose rows are wanted.

    Returns
    -------
    solprop.output.schema.StepStats
        The rows of that plane, empty for a run that recorded no periodic subset.
    """
    group = _group(file, "step_stats")
    names = [
        str(name) for name in group if str(name) not in ("counts", *STEP_STATS_INDEX)
    ]
    counts = np.asarray(_dataset(group, "counts")[: index + 1], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    piece = slice(int(offsets[index]), int(offsets[index + 1]))
    return StepStats(
        z=np.asarray(_dataset(group, "z")[piece]),
        step=np.asarray(_dataset(group, "step")[piece]),
        values={name: np.asarray(_dataset(group, name)[piece]) for name in names},
    )


def resume(path: Path | str) -> ResumeState:
    """Read a file's resumable state and cumulative work, preferring its checkpoint.

    The last plane comes back in the **internal** FFT convention, through
    :func:`solprop.fourier.spectrum_to_internal`, so a driver can hand it straight to a
    backend without conjugating anything itself.
    A separate terminal checkpoint contributes no regular observation but supplies
    its actual position, pending attempt history and cumulative call ledgers.
    Reading preserves its landing classification; the driver requires explicit
    ``resume="interpolated"`` consent before restarting from an interpolated field.

    A run that keeps no field on its last regular plane (SEP 22) comes back with
    that plane's ``field=None`` unless a terminal checkpoint, which always carries the
    field, follows it; the driver refuses to continue from it by name.

    Parameters
    ----------
    path : pathlib.Path or str
        The file to read.

    Returns
    -------
    ResumeState
        The metadata, the last plane, how many planes the file holds, and every attempt
        the stepper made.

    Raises
    ------
    OutputError
        If the file cannot be read, carries no metadata, or holds no plane.

    Examples
    --------
    >>> import numpy as np, tempfile
    >>> from pathlib import Path
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output import HDF5Output
    >>> from solprop.output.hdf5 import resume
    >>> from solprop.output.schema import SavedPlane, _example_metadata
    >>> metadata = _example_metadata()
    >>> n_omega = metadata.grid.n_omega
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     handler = HDF5Output(Path(folder) / "run.h5")
    ...     handler.open(metadata)
    ...     handler.save(SavedPlane(z=0.25, field=np.ones((1, 1, n_omega), complex)))
    ...     handler.close()
    ...     state = resume(Path(folder) / "run.h5")
    >>> state.n_saved, state.plane.z
    (1, 0.25)

    Carry on in the same file with a handler built with ``append=True``:

    >>> with tempfile.TemporaryDirectory() as folder:
    ...     handler = HDF5Output(Path(folder) / "run.h5")
    ...     handler.open(metadata)
    ...     handler.save(SavedPlane(z=0.0, field=np.ones((1, 1, n_omega), complex)))
    ...     handler.close()
    ...     again = HDF5Output(Path(folder) / "run.h5", append=True)
    ...     again.open(metadata)
    ...     again.save(SavedPlane(z=0.5, field=np.ones((1, 1, n_omega), complex)))
    ...     again.close()
    ...     resume(Path(folder) / "run.h5").n_saved
    2
    """
    metadata, _ = _stored_metadata(path)
    _check_plasma_benchmark(metadata)
    with h5py.File(path, "r") as file:
        checkpoint = _read_checkpoint(file, metadata)
        outcome = _read_outcome(file)
        n_saved = _stored_counts(file)[0] if "z" in file else 0
        if n_saved:
            plane, whole = _regular_resume_plane(file, metadata, n_saved - 1)
        elif checkpoint is not None:
            plane, whole = checkpoint.plane, StepHistory.empty()
        else:
            msg = f"{path} holds no saved plane, so there is nothing to resume from"
            raise OutputError(msg)
        if checkpoint is not None:
            if n_saved and checkpoint.plane.z < plane.z:
                raise OutputError("terminal checkpoint precedes the last saved plane")
            if outcome is None or checkpoint.plane.z != outcome.z:
                raise OutputError(
                    "terminal checkpoint has no matching execution outcome"
                )
            plane = checkpoint.plane
            whole = StepHistory(
                **{
                    name: np.concatenate(
                        (getattr(whole, name), getattr(plane.step_history, name))
                    )
                    for name in HISTORY_FIELDS
                }
            )
        march = _read_march(file, plane.z)
    logger.debug(
        "resuming %s at z = %.6e m with %d regular planes", path, plane.z, n_saved
    )
    return ResumeState(
        metadata=metadata,
        plane=plane,
        n_saved=n_saved,
        history=whole,
        outcome=outcome,
        checkpoint=checkpoint,
        march=march,
    )


def _regular_resume_plane(
    file: h5py.File, metadata: RunMetadata, index: int
) -> tuple[SavedPlane, StepHistory]:
    """Read the last regular plane and its complete stored attempt history.

    Parameters
    ----------
    file : h5py.File
        Open saved output with regular observations.
    metadata : RunMetadata
        Grid and output amplitude conventions.
    index : int
        Nonnegative regular-plane index to restore.

    Returns
    -------
    tuple
        Restored plane and the complete attempt history through that plane; the
        plane's field and noise phase are ``None`` where the run kept no field.
    """
    # A stored rule's indices are already counted from the entrance.
    retention = _stored_retention(file)
    internal = phase = None
    if retention.keeps(index):
        slot = (
            index if retention.fields == "all" else retention.field_planes.index(index)
        )
        internal = fourier.spectrum_to_internal(
            np.asarray(_dataset(file, "field")[slot]), kind=metadata.grid.kind
        )
        if "noise_phase" in file:
            phase = np.asarray(_dataset(file, "noise_phase")[slot])
    group = _group(file, "stats")
    whole, part = _last_history(file, index)
    return SavedPlane(
        z=float(_dataset(file, "z")[index]),
        field=internal,
        stats={
            str(name): np.asarray(_dataset(group, str(name))[index]) for name in group
        },
        step_history=part,
        step_stats=_last_step_stats(file, index),
        noise_phase=phase,
    ), whole


def open_file(path: Path | str) -> HDF5Output:
    """Open a finished run for reading, through the protocol it was written with.

    The handler comes back open and read-only: ``__getitem__``, ``fields``,
    ``stat_names``, ``n_saved`` and ``metadata`` answer from the file, ``open`` and
    ``save`` are refused, and ``close`` releases the file -- a ``with`` block does that
    on the way out. It is how :mod:`solprop.processing` reads a file after the run that
    wrote it has ended; :func:`resume` reads the last plane only, and a handler built
    with ``append=True`` opens the file read-write to continue the run.

    Parameters
    ----------
    path : pathlib.Path or str
        The file to read.

    Returns
    -------
    HDF5Output
        An open, read-only handler.

    Raises
    ------
    OutputError
        If the file cannot be opened, or carries no metadata.

    Warns
    -----
    SolpropWarning
        Once, for a file a session continued with ``resume="force"``, naming where
        (SEP 42 section 5.5).

    Examples
    --------
    >>> import numpy as np, tempfile
    >>> from pathlib import Path
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output import MemoryOutput, open_file
    >>> from solprop.output.schema import (
    ...     GridRecord, RunMetadata, SavedPlane, StepperRecord
    ... )
    >>> from solprop.stepper import DOPRI5, PIController
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> memory = MemoryOutput()
    >>> memory.open(RunMetadata(
    ...     grid=GridRecord.from_grid(grid), modes=(), simulation_kind="mode_averaged",
    ...     configuration={},
    ...     stepper=StepperRecord.from_settings(
    ...         DOPRI5, PIController(rtol=1e-8, atol=0.0)
    ...     ),
    ... ))
    >>> memory.save(SavedPlane(z=0.0, field=np.ones((1, 1, grid.n_omega), complex)))
    >>> with tempfile.TemporaryDirectory() as folder:
    ...     memory.to_file(Path(folder) / "run.h5")
    ...     with open_file(Path(folder) / "run.h5") as stored:
    ...         shape, planes = stored["field"].shape, stored.n_saved
    >>> shape, planes, stored.closed
    ((1, 1, 1, 2560), 1, True)
    """
    handler = HDF5Output(path, readonly=True)
    outcome = handler.outcome
    forced = () if outcome is None else outcome.forced_positions
    if forced:
        where = ", ".join(f"{z:.6g}" for z in forced)
        warnings.warn(
            f"{path} was continued with resume='force' at z = {where} m: past there "
            "its record does not describe its planes. stats/* hold; anything rebuilt "
            "from the record does not, and those planes are not exchanged",
            SolpropWarning,
            stacklevel=2,
        )
    return handler
