# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Archive every raw TDSE provider evaluation of a run in an HDF5 sidecar (SEP 20).

A TDSE run's optical output holds the propagated field; this append-only sidecar
holds what the atomic provider actually returned for each evaluation, before any
validation, shadow subtraction, scaling, padding or reduction, beside the exact
input it was given. It is a history of evaluations, not a restart store: the
optical file's call ledger stays authoritative for calls, budgets and resume, and
the sidecar does no charge accounting.

**Layout** (schema :data:`TDSE_ARCHIVE_SCHEMA`). The root carries ``schema`` and
``run_uuid``. ``sessions/session_NNNNNN`` is one run or resume (or one standalone
consistency check), with its context and, once it closes, the bytes written and
the write time. Inside it ``evaluations/evaluation_NNNNNNNNN`` is one logical
provider evaluation: its labels as attributes, ``prewindow_field`` (the full
nodal field of the evaluation, V/m), ``gate`` when an atomic window applies, one
``chunk_NNN`` group per provider call holding ``drive`` (the exact float64 field
passed, V/m) and ``raw`` (the returned record, through a versioned serializer),
an ``exception`` group when the evaluation raised, and finally the ``outcome``
attribute. ``trials/trial_NNNNNNNNN`` records whether an optical trial was
accepted or rejected, appended when the stepper decides. Shared immutable values
-- sample times, gates and structured records -- are stored once per session under
``shared`` and hard-linked where they are used, so they read back unchanged.

**Truncation.** Groups are flushed as they complete. A crash leaves every earlier
evaluation readable; an evaluation without its ``outcome``, or missing a chunk or a
dataset, is reported as incomplete by :class:`TDSEArchiveReader` rather than
failing the read.

The manual page is ``docs/guide/media/tdse.md``.
"""

import contextlib
import dataclasses
import errno
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from types import SimpleNamespace, TracebackType
from typing import Literal, cast

import h5py
import numpy as np
from jaxtyping import Float

from solprop.exceptions import OutputError, TDSEArchiveError
from solprop.output._writer_lock import WriterLock

__all__ = [
    "ARCHIVE_METADATA_CACHE_BYTES",
    "HACC_GRID_RESPONSE",
    "HACC_TOTAL_CURRENT_RESPONSE",
    "SAE_GRID_RESPONSE",
    "SAE_TOTAL_CURRENT_RESPONSE",
    "TDSE_ARCHIVE_SCHEMA",
    "ArchiveOutputs",
    "ArchivedChunk",
    "ArchivedEvaluation",
    "ArchivedSession",
    "RecordField",
    "RecordSchema",
    "SessionKind",
    "TDSEArchive",
    "TDSEArchiveReader",
    "TDSEArchiveWriter",
    "open_tdse_archive",
    "record_serializer",
    "structured_json",
    "writer_memory",
]

logger = logging.getLogger(__name__)

TDSE_ARCHIVE_SCHEMA = "solprop.tdse-archive.v1"
"""Schema identifier stored at the root of every archive this module writes."""

ARCHIVE_METADATA_CACHE_BYTES = 4 * 1024 * 1024
"""Upper bound of the HDF5 metadata cache of one open archive, bytes.

The writer stores every array as a contiguous, uncompressed dataset, so HDF5 keeps
no raw-data chunk cache for it; its metadata cache is capped at this size when the
file is opened. It is admitted with the binding's memory.
"""

ArchiveOutputs = Literal["full", "required"]
SessionKind = Literal["run", "resume", "consistency_check"]

_Roles = Literal["startup", "predictor", "corrector", "diagnostic", "rejected"]
_ARRAY_KINDS = frozenset("biufc")
# Errors HDF5, NumPy or JSON raise when a value cannot be written; each is reported
# as a TDSEArchiveError naming the evaluation, so the run stops before its next call.
_WRITE_ERRORS = (OSError, ValueError, TypeError, RuntimeError, KeyError)
_ARCHIVE_ERRORS = (*_WRITE_ERRORS, TDSEArchiveError)


# --------------------------------------------------------------------- configuration


@dataclass(frozen=True, init=False)
class TDSEArchive:
    """Ask a TDSE run to archive every raw provider evaluation (SEP 20 section 5.8).

    Passed as ``tdse_archive=`` to :meth:`solprop.driver.simulation.Simulation.run`,
    :func:`solprop.driver.interface.prop_capillary` or
    :func:`solprop.driver.tdse_consistency.check_consistency`. The archive is a
    separate, append-only HDF5 file; it changes neither the propagation nor the call
    ledger, and its path and output selection are execution provenance, not part of
    any physical identity.

    Parameters
    ----------
    path : pathlib.Path or str
        Where the archive is written. It must differ from the optical HDF5 file. An
        existing file is continued only when it is this run's archive (a resume);
        any other existing file is refused, never overwritten.
    outputs : {'full', 'required'}, optional
        ``"full"`` (the default) asks the same provider evaluations for the
        provider's full observable record and stores all of it; ``"required"``
        keeps the provider's output selection that the source policy needs and
        stores every field it returns. Neither adds a provider call, a shadow order
        or a change to the atomic state.

    Attributes
    ----------
    path : pathlib.Path
        The archive's path, as given.
    outputs : {'full', 'required'}
        The output selection.

    Raises
    ------
    TypeError
        If `path` is not a path or a string.
    ValueError
        If `path` is empty or `outputs` is neither selection.

    Examples
    --------
    >>> from solprop.output import TDSEArchive
    >>> archive = TDSEArchive("run.tdse.h5", outputs="required")
    >>> archive.path.name, archive.outputs
    ('run.tdse.h5', 'required')
    """

    path: Path
    outputs: ArchiveOutputs

    def __init__(self, path: Path | str, *, outputs: ArchiveOutputs = "full") -> None:
        """Adopt the path as a :class:`pathlib.Path` and check the selection."""
        candidate: object = path
        if not isinstance(candidate, Path | str):
            msg = f"TDSEArchive path must be a path or a string, got {type(path)!r}"
            raise TypeError(msg)
        if str(candidate) == "":
            raise ValueError("TDSEArchive path must not be empty")
        if outputs not in ("full", "required"):
            msg = f"TDSEArchive outputs must be 'full' or 'required', got {outputs!r}"
            raise ValueError(msg)
        object.__setattr__(self, "path", Path(candidate))
        object.__setattr__(self, "outputs", outputs)


# ----------------------------------------------------------------------- serializers


@dataclass(frozen=True)
class RecordField:
    """Describe one attribute of a provider record and how it is stored.

    Attributes
    ----------
    name : str
        The attribute name on the provider record.
    kind : {'array', 'scalar', 'labels', 'outcomes', 'structured', 'record'}
        An owning array stored with its dtype and shape; a number stored as a
        zero-dimensional dataset of its dtype; a tuple of integer order labels; the
        per-row outcome records; a structured record (``info``) stored as JSON; or a
        nested record with its own schema.
    unit : str
        SI unit of an array or scalar (``"1"`` for a dimensionless one, ``""`` for
        none, such as a validity mask or a step index).
    axes : str
        Axis names of an array, in the shared vocabulary.
    optional : bool
        Whether the provider may return ``None``; an absent value is listed in the
        stored group's ``absent`` attribute.
    record : RecordSchema or None
        The schema of a nested record.
    """

    name: str
    kind: Literal["array", "scalar", "labels", "outcomes", "structured", "record"]
    unit: str = ""
    axes: str = ""
    optional: bool = False
    # Quoted: RecordSchema is defined below and refers back to this class.
    record: "RecordSchema | None" = None  # noqa: UP037 - forward reference


@dataclass(frozen=True)
class RecordSchema:
    """Name, version and fields of one explicit provider-record serializer.

    Attributes
    ----------
    name : str
        Record name, such as ``"sae.grid_response"``.
    version : int
        Serializer version; a change of the stored fields or their meaning bumps it.
    fields : tuple of RecordField
        Every attribute stored, in order.
    """

    name: str
    version: int
    fields: tuple[RecordField, ...]

    @property
    def serializer(self) -> str:
        """Return the identifier written beside every record stored with this schema.

        Returns
        -------
        str
            ``solprop.tdse-archive.<name>.v<version>``.
        """
        return f"solprop.tdse-archive.{self.name}.v{self.version}"


def _array(
    name: str, unit: str, axes: str = "n_row n_time", *, optional: bool = False
) -> RecordField:
    """Return an array field.

    Parameters
    ----------
    name, unit, axes : str
        Attribute name, SI unit and axis names.
    optional : bool
        Whether ``None`` is admitted.

    Returns
    -------
    RecordField
        The field.
    """
    return RecordField(name, "array", unit, axes, optional)


_ROW_MONITOR_AXES = "n_row"
_SAE_MONITORS = RecordSchema(
    "sae.row_monitors",
    1,
    (
        _array("initial_norm", "1", _ROW_MONITOR_AXES),
        _array("norm", "1", _ROW_MONITOR_AXES),
        _array("peak_norm", "1", _ROW_MONITOR_AXES),
        _array("field_free_energy", "J", _ROW_MONITOR_AXES),
        _array("monitored_step", "", _ROW_MONITOR_AXES),
        _array("removed_component", "1", _ROW_MONITOR_AXES, optional=True),
        _array("step_error_estimate", "1", _ROW_MONITOR_AXES, optional=True),
        _array("peak_step_error_estimate", "1", _ROW_MONITOR_AXES, optional=True),
    ),
)
_HACC_MONITORS = RecordSchema(
    "hacc.row_monitors",
    1,
    (
        _array("initial_norm", "1", _ROW_MONITOR_AXES),
        _array("norm", "1", _ROW_MONITOR_AXES),
        _array("peak_norm", "1", _ROW_MONITOR_AXES),
        _array("monitored_step", "", _ROW_MONITOR_AXES),
        _array("non_finite", "", _ROW_MONITOR_AXES),
    ),
)
_ORDERED = "n_row n_order n_time"
_SAE_PLASMA = RecordSchema(
    "sae.plasma_response",
    1,
    (
        RecordField("radius", "scalar", "m"),
        RecordField("potential_slope_bound", "scalar", "J/m"),
        _array("inbox_acceleration", "C m/s^2"),
        _array("free_acceleration", "C m/s^2"),
        _array("total_acceleration", "C m/s^2"),
        _array("inbox_current", "C m/s"),
        _array("free_current", "C m/s"),
        _array("total_current", "C m/s"),
        _array("outside_integral", "s"),
        _array("neglected_acceleration_bound", "C m/s^2"),
        _array("neglected_current_bound", "C m/s"),
    ),
)
_SAE_SHADOW = RecordSchema(
    "sae.perturbative_shadow",
    1,
    (
        RecordField("orders", "labels"),
        RecordField("radius", "scalar", "m"),
        _array("charge_dipole", "C m", _ORDERED),
        _array("current", "C m/s", _ORDERED),
        _array("total_acceleration", "C m/s^2", _ORDERED),
        _array("total_current", "C m/s", _ORDERED, optional=True),
    ),
)
_HACC_SHADOW = RecordSchema(
    "hacc.shadow",
    1,
    (
        RecordField("orders", "labels"),
        _array("charge_dipole", "C m", _ORDERED),
        _array("current", "C m/s", _ORDERED),
        # With the escaped-electron closure the order-one part of the closed total
        # current, one order; None without the closure.
        _array("total_current", "C m/s", _ORDERED, optional=True),
    ),
)
_COMMON_TAIL = (
    RecordField("outcomes", "outcomes"),
    _array("valid", ""),
    RecordField("info", "structured"),
)

SAE_GRID_RESPONSE = RecordSchema(
    "sae.grid_response",
    1,
    (
        _array("time", "s", "n_time"),
        _array("vector_potential", "V s/m"),
        _array("electric_field", "V/m"),
        _array("charge_dipole", "C m"),
        _array("current", "C m/s"),
        _array("dipole_acceleration", "C m/s^2"),
        _array("outside", "1"),
        RecordField("plasma", "record", record=_SAE_PLASMA),
        RecordField("cut_term", "scalar", "1"),
        RecordField("monitors", "record", record=_SAE_MONITORS),
        *_COMMON_TAIL,
        RecordField("shadow", "record", optional=True, record=_SAE_SHADOW),
    ),
)
"""The SAE provider's full record: dipole, currents, accelerations, plasma, shadow."""

SAE_TOTAL_CURRENT_RESPONSE = RecordSchema(
    "sae.total_current_response",
    1,
    (
        _array("time", "s", "n_time"),
        _array("total_current", "C m/s"),
        _array("shadow_total_current", "C m/s", _ORDERED, optional=True),
        RecordField("shadow_orders", "labels"),
        RecordField("radius", "scalar", "m"),
        RecordField("potential_slope_bound", "scalar", "J/m"),
        _array("neglected_acceleration_bound", "C m/s^2"),
        _array("neglected_current_bound", "C m/s"),
        RecordField("monitors", "record", record=_SAE_MONITORS),
        *_COMMON_TAIL,
    ),
)
"""The SAE provider's compact record: the total current and its odd shadow orders."""

HACC_GRID_RESPONSE = RecordSchema(
    "hacc.grid_response",
    1,
    (
        _array("time", "s", "n_time"),
        _array("vector_potential", "V s/m"),
        _array("electric_field", "V/m"),
        _array("charge_dipole", "C m"),
        _array("current", "C m/s"),
        _array("total_current", "C m/s"),
        # The lost norm as the provider defines it (initial minus current norm
        # inside R0), not an ionisation probability.
        _array("outside", "1"),
        _array("neglected_current_bound", "C m/s", optional=True),
        RecordField("monitors", "record", record=_HACC_MONITORS),
        *_COMMON_TAIL,
        RecordField("shadow", "record", optional=True, record=_HACC_SHADOW),
    ),
)
"""The haCC provider's full record: dipole, in-box and total currents, lost norm."""

HACC_TOTAL_CURRENT_RESPONSE = RecordSchema(
    "hacc.total_current_response",
    1,
    (
        _array("time", "s", "n_time"),
        _array("total_current", "C m/s"),
        _array("shadow_total_current", "C m/s", _ORDERED, optional=True),
        RecordField("shadow_orders", "labels"),
        RecordField("radius", "scalar", "m"),
        _array("neglected_current_bound", "C m/s", optional=True),
        RecordField("monitors", "record", record=_HACC_MONITORS),
        *_COMMON_TAIL,
    ),
)
"""The haCC provider's compact record: the total current and its order one."""

_SERIALIZERS: dict[tuple[str, str], RecordSchema] = {
    ("sae", "full"): SAE_GRID_RESPONSE,
    ("sae", "total_current"): SAE_TOTAL_CURRENT_RESPONSE,
    ("hacc", "full"): HACC_GRID_RESPONSE,
    ("hacc", "total_current"): HACC_TOTAL_CURRENT_RESPONSE,
}


def record_serializer(model: str, outputs: str) -> RecordSchema:
    """Return the serializer of a model's record for a native output selection.

    Parameters
    ----------
    model : {'sae', 'hacc'}
        The atomic model.
    outputs : {'full', 'total_current'}
        The provider's construction-time output selection.

    Returns
    -------
    RecordSchema
        The explicit, versioned serializer.

    Raises
    ------
    TDSEArchiveError
        If no serializer exists for the pair.

    Examples
    --------
    >>> from solprop.output.tdse_archive import record_serializer
    >>> record_serializer("hacc", "full").serializer
    'solprop.tdse-archive.hacc.grid_response.v1'
    """
    try:
        return _SERIALIZERS[model, outputs]
    except KeyError:
        msg = f"no TDSE archive serializer for model {model!r} outputs {outputs!r}"
        raise TDSEArchiveError(msg) from None


def structured_json(value: object) -> str:
    """Encode a structured provider record as sorted, valid JSON.

    Dataclasses and namespaces become objects, arrays and tuples become lists,
    non-finite floats become the strings ``"nonfinite:nan"``, ``"nonfinite:inf"``
    and ``"nonfinite:-inf"``, and any other object becomes ``{"type": ..., "repr":
    ...}``, so a structured diagnostic never stops the archive.

    Parameters
    ----------
    value : object
        The record.

    Returns
    -------
    str
        JSON text.
    """
    return json.dumps(_plain(value), sort_keys=True, allow_nan=False)


def _plain(value: object) -> object:  # noqa: PLR0911 - one return per value kind
    """Return a JSON-compatible copy of one structured value.

    Parameters
    ----------
    value : object
        Any value.

    Returns
    -------
    object
        Plain containers and scalars.
    """
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else f"nonfinite:{value}"
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, Enum):
        return _plain(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _plain(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, SimpleNamespace):
        return {str(key): _plain(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): _plain(item) for key, item in mapping.items()}
    if isinstance(value, list | tuple):
        sequence = cast("Sequence[object]", value)
        return [_plain(item) for item in sequence]
    return {"type": type(value).__qualname__, "repr": repr(value)}


# --------------------------------------------------------------------------- writer


def _now() -> str:
    """Return the current UTC time in ISO 8601.

    Returns
    -------
    str
        Timestamp.
    """
    return datetime.now(UTC).isoformat()


def _digest(data: bytes) -> str:
    """Return a short content address for a shared value.

    Parameters
    ----------
    data : bytes
        The value's bytes, with its dtype and shape folded in by the caller.

    Returns
    -------
    str
        Thirty-two hexadecimal characters of its SHA-256.
    """
    return hashlib.sha256(data).hexdigest()[:32]


def _admit_array(value: object, where: str) -> np.ndarray:
    """Return an array the archive can store with its exact dtype, or refuse it.

    Parameters
    ----------
    value : object
        A provider array.
    where : str
        Its path in the record, for the error.

    Returns
    -------
    numpy.ndarray
        The same array.

    Raises
    ------
    TDSEArchiveError
        If it is not a NumPy array of a numeric or boolean dtype.
    """
    if not isinstance(value, np.ndarray):
        msg = f"cannot serialize {where}: expected a NumPy array, got {type(value)!r}"
        raise TDSEArchiveError(msg)
    if value.dtype.kind not in _ARRAY_KINDS or value.dtype.hasobject:
        msg = f"cannot serialize {where}: dtype {value.dtype} is not numeric"
        raise TDSEArchiveError(msg)
    return value


def _admit_scalar(value: object, where: str) -> np.ndarray:
    """Return a numeric scalar as a zero-dimensional array of its own dtype.

    Parameters
    ----------
    value : object
        A Python or NumPy number.
    where : str
        Its path in the record, for the error.

    Returns
    -------
    numpy.ndarray
        Zero-dimensional array.

    Raises
    ------
    TDSEArchiveError
        If it is not a number.
    """
    if isinstance(value, bool | int | float | np.number | np.bool_):
        return np.asarray(value)
    msg = f"cannot serialize {where}: expected a number, got {type(value)!r}"
    raise TDSEArchiveError(msg)


def _admit_labels(value: object, where: str) -> np.ndarray:
    """Return integer order labels as an int64 vector.

    Parameters
    ----------
    value : object
        A tuple of integers.
    where : str
        Its path in the record, for the error.

    Returns
    -------
    numpy.ndarray
        The labels.

    Raises
    ------
    TDSEArchiveError
        If the value is not a tuple of plain integers.
    """
    if not isinstance(value, tuple) or not all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in cast("tuple[object, ...]", value)
    ):
        msg = f"cannot serialize {where}: order labels must be a tuple of integers"
        raise TDSEArchiveError(msg)
    return np.asarray(value, dtype=np.int64).reshape(-1)


def _outcomes_json(value: object, where: str) -> str:
    """Encode the per-row outcome records.

    Parameters
    ----------
    value : object
        A sequence of records with ``failure`` and ``failure_step``.
    where : str
        Its path in the record, for the error.

    Returns
    -------
    str
        JSON list of ``{"failure": ..., "failure_step": ...}``.

    Raises
    ------
    TDSEArchiveError
        If an outcome lacks either attribute.
    """
    if not isinstance(value, tuple | list):
        msg = f"cannot serialize {where}: outcomes must be a sequence"
        raise TDSEArchiveError(msg)
    rows: list[dict[str, object]] = []
    for item in cast("Sequence[object]", value):
        try:
            rows.append(
                {
                    "failure": _plain(getattr(item, "failure")),  # noqa: B009 - explicit field
                    "failure_step": _plain(getattr(item, "failure_step")),  # noqa: B009
                }
            )
        except AttributeError as error:
            msg = f"cannot serialize {where}: {error}"
            raise TDSEArchiveError(msg) from error
    return json.dumps(rows, sort_keys=True, allow_nan=False)


@dataclass
class _Counters:
    """Accumulate what one session wrote.

    Attributes
    ----------
    array_bytes : int
        Bytes of array data written.
    seconds : float
        Wall time spent writing, s.
    evaluations : int
        Evaluation groups begun.
    """

    array_bytes: int = 0
    seconds: float = 0.0
    evaluations: int = 0


def _relative_reference(archive: Path, optical: Path | None) -> dict[str, object]:
    """Return where the optical file finds its archive.

    Parameters
    ----------
    archive : pathlib.Path
        The archive path.
    optical : pathlib.Path or None
        The optical HDF5 file, or None for an in-memory output.

    Returns
    -------
    dict
        ``path`` relative to the optical file's directory with ``relative_to =
        "optical_file"``, or the absolute path with ``relative_to = None`` when there
        is no optical file or no relative path exists.
    """
    absolute = archive.resolve()
    if optical is not None:
        try:
            relative = os.path.relpath(absolute, optical.resolve().parent)
        except ValueError:  # pragma: no cover - only across Windows drives
            pass
        else:
            return {"path": Path(relative).as_posix(), "relative_to": "optical_file"}
    return {"path": absolute.as_posix(), "relative_to": None}


class TDSEArchiveWriter:
    """Write one session of a TDSE archive: one synchronous, append-only writer.

    Built with :meth:`open`. Every method writes through to the file and flushes at
    the end of each group, keeping no array after it returns. The first failure
    marks the writer failed: every later write, and :meth:`ensure_writable`, which
    the binding calls before each provider call, raise
    :class:`~solprop.exceptions.TDSEArchiveError`, so the run stops before its next
    provider call.

    Parameters
    ----------
    file : h5py.File
        The open archive.
    archive : TDSEArchive
        The request.
    run_uuid : str
        The run's identity, shared with the optical file.
    session : h5py.Group
        This session's group.
    index : int
        This session's index in the file.
    first_evaluation : int
        The identifier of this session's first evaluation.
    optical : pathlib.Path or None
        The optical HDF5 file, for the relative reference.
    start_size : int
        The file's size before this session, bytes.
    """

    def __init__(  # noqa: PLR0913 - the open file and its session
        self,
        file: h5py.File,
        archive: TDSEArchive,
        *,
        run_uuid: str,
        session: h5py.Group,
        index: int,
        first_evaluation: int,
        optical: Path | None,
        start_size: int,
    ) -> None:
        self._file: h5py.File | None = file
        self._archive = archive
        self._run_uuid = run_uuid
        self._session: h5py.Group | None = session
        self._session_name = session.name
        self._writer_lock = WriterLock(archive.path)
        self._suspended_identity: tuple[int, int] | None = None
        self._index = index
        self._next_evaluation = first_evaluation
        self._optical = optical
        self._counters = _Counters()
        self._failure: TDSEArchiveError | None = None
        self._open_evaluations: dict[int, h5py.Group] = {}
        self._start_size = start_size
        self._bytes_written = 0
        self._closed = False

    # -- opening ---------------------------------------------------------------------

    @classmethod
    def open(
        cls,
        archive: TDSEArchive,
        *,
        run_uuid: str,
        kind: SessionKind,
        optical: Path | None,
        context: Mapping[str, object],
        resume: bool = False,
    ) -> "TDSEArchiveWriter":  # noqa: UP037 - the runtime type hook resolves the class
        """Open or create the archive and append a new session group.

        Parameters
        ----------
        archive : TDSEArchive
            Path and output selection.
        run_uuid : str
            The run's identity. An existing file is continued only with ``resume``
            and only when it carries this identity.
        kind : {'run', 'resume', 'consistency_check'}
            What the session records.
        optical : pathlib.Path or None
            The optical HDF5 file, or None.
        context : mapping
            JSON-compatible facts of the session: the descriptor, quadrature,
            execution and provider pin.
        resume : bool, optional
            Whether an existing archive of this run may be continued.

        Returns
        -------
        TDSEArchiveWriter
            The writer of the new session.

        Raises
        ------
        TDSEArchiveError
            If the path exists and is not this run's archive (or is not to be
            continued), or the file cannot be created or written.
        """
        lease = WriterLock(archive.path)
        try:
            lease.acquire()
        except OutputError as error:
            raise TDSEArchiveError(
                f"cannot open the TDSE archive {archive.path}: {error}"
            ) from error
        path = archive.path
        file: h5py.File | None = None
        action = "open the TDSE archive"
        try:
            exists = path.exists()
            if exists and (not path.is_file() or not h5py.is_hdf5(path)):
                raise TDSEArchiveError(
                    f"{path} exists and is not a TDSE archive; it is never "
                    "overwritten: remove it or choose another path"
                )
            file = h5py.File(path, "r+" if exists else "x", libver="earliest")
            if exists:
                cls._validate_existing(file, path, run_uuid, resume=resume)
            else:
                file.attrs["schema"] = TDSE_ARCHIVE_SCHEMA
                file.attrs["run_uuid"] = run_uuid
                file.attrs["created"] = _now()
                file.create_group("sessions")
            _cap_metadata_cache(file)
            file.flush()
            start_size = path.stat().st_size
            action = "start a session in the TDSE archive"
            sessions = cast("h5py.Group", file["sessions"])
            index = len(sessions)
            first = _next_evaluation_id(sessions)
            session = sessions.create_group(f"session_{index:06d}")
            session.attrs["index"] = index
            session.attrs["kind"] = kind
            session.attrs["started"] = _now()
            session.attrs["outputs"] = archive.outputs
            session.attrs["run_uuid"] = run_uuid
            session.attrs["first_evaluation"] = first
            reference = _relative_reference(path, optical)
            session.attrs["optical_reference"] = json.dumps(
                None if optical is None else reference
            )
            session.create_dataset(
                "context", data=structured_json(context), dtype=h5py.string_dtype()
            )
            session.create_group("shared")
            session.create_group("evaluations")
            session.create_group("trials")
            file.flush()
            writer = cls(
                file,
                archive,
                run_uuid=run_uuid,
                session=session,
                index=index,
                first_evaluation=first,
                optical=optical,
                start_size=start_size,
            )
            writer._writer_lock = lease
            return writer
        except BaseException as error:
            failure = (
                TDSEArchiveError(f"cannot {action} {path}: {error}")
                if isinstance(error, _WRITE_ERRORS)
                else error
            )
            cls._close_failed_open(file, lease, failure)
            if failure is error:
                raise
            raise failure from error

    @staticmethod
    def _close_failed_open(
        file: h5py.File | None, lease: WriterLock, failure: BaseException
    ) -> None:
        """Release a failed factory's ownership only after confirmed closure.

        Parameters
        ----------
        file : h5py.File or None
            Handle opened before setup failed, if any.
        lease : WriterLock
            Factory's acquired lease.
        failure : BaseException
            Primary error, with cleanup failures attached as notes.
        """
        if file is not None:
            try:
                file.close()
            except BaseException as cleanup:
                # No writer can be returned for a retry. Fail closed until exit.
                failure.add_note(
                    f"archive setup handle could not close: {cleanup}; "
                    "writer ownership retained until process exit"
                )
                return
        lease.release()

    @staticmethod
    def _validate_existing(
        file: h5py.File, path: Path, run_uuid: str, *, resume: bool
    ) -> None:
        """Validate an existing archive while its caller owns the handle.

        Parameters
        ----------
        file : h5py.File
            The file, open read-write under the persistent writer lease.
        path : pathlib.Path
            The existing path.
        run_uuid : str
            The identity it must carry.
        resume : bool
            Whether continuing is allowed at all.

        Raises
        ------
        TDSEArchiveError
            If the file is not this run's archive, or this is not a resume.
        """
        schema = file.attrs.get("schema")
        stored = file.attrs.get("run_uuid")
        if schema != TDSE_ARCHIVE_SCHEMA or "sessions" not in file:
            msg = (
                f"{path} exists and is not a {TDSE_ARCHIVE_SCHEMA} archive; it is "
                "never overwritten"
            )
            raise TDSEArchiveError(msg)
        if not resume or stored != run_uuid:
            msg = (
                f"{path} is the TDSE archive of another run ({stored}); an archive "
                "is continued only by a resume of its own run and is never "
                "overwritten: remove it or choose another path"
            )
            raise TDSEArchiveError(msg)

    # -- properties ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Return the archive's path.

        Returns
        -------
        pathlib.Path
            As requested.
        """
        return self._archive.path

    @property
    def outputs(self) -> ArchiveOutputs:
        """Return the output selection.

        Returns
        -------
        {'full', 'required'}
            The request's selection.
        """
        return self._archive.outputs

    @property
    def run_uuid(self) -> str:
        """Return the run's identity.

        Returns
        -------
        str
            Shared with the optical file.
        """
        return self._run_uuid

    @property
    def failed(self) -> bool:
        """Return whether a write has failed.

        Returns
        -------
        bool
            True once any write raised.
        """
        return self._failure is not None

    def memory_ledger(self, *, rows: int, samples: int) -> dict[str, int]:
        """Return the writer's bounded host buffers for memory admission.

        Parameters
        ----------
        rows : int
            Largest provider chunk, rows.
        samples : int
            Largest per-row sample count.

        Returns
        -------
        dict
            :func:`writer_memory` of the largest chunk.
        """
        return writer_memory(rows=rows, samples=samples)

    def reference(self) -> dict[str, object]:
        """Return the record of this archive the optical file keeps.

        Returns
        -------
        dict
            The schema, the run identity, the path (relative to the optical file
            when there is one), the output selection, the session, and the bytes
            written, array bytes, write time and evaluation count so far.
        """
        return {
            "schema": TDSE_ARCHIVE_SCHEMA,
            "run_uuid": self._run_uuid,
            **_relative_reference(self.path, self._optical),
            "outputs": self.outputs,
            "session": self._index,
            "bytes_written": self._bytes_written,
            "array_bytes": self._counters.array_bytes,
            "write_seconds": self._counters.seconds,
            "evaluations": self._counters.evaluations,
            "failed": self.failed,
        }

    # -- writing ---------------------------------------------------------------------

    def ensure_writable(self) -> None:
        """Refuse to continue after a write failure.

        Raises
        ------
        TDSEArchiveError
            The first failure, again, so no provider call follows it.
        """
        if self._failure is not None:
            msg = f"the TDSE archive {self.path} failed earlier: {self._failure}"
            raise TDSEArchiveError(msg) from self._failure
        if self._closed:
            msg = f"the TDSE archive {self.path} is closed"
            raise TDSEArchiveError(msg)
        if self._suspended_identity is not None:
            raise TDSEArchiveError("the TDSE archive is suspended")

    def _require_session(self) -> h5py.Group:
        """Return the live session; suspended writers cannot issue HDF5 operations.

        Returns
        -------
        h5py.Group
            Live session group.
        """
        self.ensure_writable()
        assert self._session is not None
        return self._session

    def suspend(self) -> None:
        """Close the handle without ending this session or releasing ownership.

        Raises
        ------
        TDSEArchiveError
            If an evaluation is in flight or the writer cannot be suspended.
        """
        file = self._require_file()
        if self._open_evaluations:
            raise TDSEArchiveError("cannot suspend an archive with an open evaluation")
        try:
            session = self._require_session()
            session.attrs["array_bytes"] = self._counters.array_bytes
            session.attrs["evaluations"] = self._counters.evaluations
            session.attrs["write_seconds"] = self._counters.seconds
            session.attrs["bytes_written"] = self._bytes_written
            file.flush()
            stat = self.path.stat()
            file.close()
        except _WRITE_ERRORS as error:
            raise self._fail(error, "snapshot suspension") from error
        self._file = None
        self._session = None
        self._suspended_identity = (stat.st_dev, stat.st_ino)

    def reopen(self) -> None:
        """Reopen the same session, retaining all evaluation and trial identifiers.

        Raises
        ------
        TDSEArchiveError
            If the file changed or cannot be reopened safely.
        """
        if self._suspended_identity is None:
            raise TDSEArchiveError("TDSE archive is not suspended")
        try:
            stat = self.path.stat()
            if (stat.st_dev, stat.st_ino) != self._suspended_identity:
                raise TDSEArchiveError("suspended TDSE archive was replaced")
            file = h5py.File(self.path, "r+", libver="earliest")
            try:
                self._validate_existing(file, self.path, self.run_uuid, resume=True)
                session = cast("h5py.Group", file[self._session_name])
                if (
                    session.attrs.get("closed", False)
                    or len(cast("h5py.Group", session["evaluations"]))
                    != self._counters.evaluations
                    or _next_evaluation_id(cast("h5py.Group", file["sessions"]))
                    != self._next_evaluation
                ):
                    raise TDSEArchiveError("suspended TDSE session changed")
                _cap_metadata_cache(file)
            except BaseException as error:
                self._file = file
                try:
                    file.close()
                except _WRITE_ERRORS as cleanup:
                    error.add_note(
                        f"rejected archive handle could not close: {cleanup}"
                    )
                else:
                    self._file = None
                raise
            self._file, self._session = file, session
            self._suspended_identity = None
        except _ARCHIVE_ERRORS as error:
            raise self._fail(error, "snapshot reopening") from error

    def _fail(self, error: BaseException, what: str) -> TDSEArchiveError:
        """Mark the writer failed and return the error to raise.

        Parameters
        ----------
        error : BaseException
            What went wrong.
        what : str
            What was being written.

        Returns
        -------
        TDSEArchiveError
            The failure, chained to `error`.
        """
        if isinstance(error, TDSEArchiveError):
            failure = error
        else:
            reason = "the disk is full" if _disk_full(error) else str(error)
            failure = TDSEArchiveError(
                f"cannot write {what} to the TDSE archive {self.path}: {reason}"
            )
            failure.__cause__ = error
        # Every write first checks `ensure_writable`, so this is the first failure.
        self._failure = failure
        return failure

    def _measure(self, started: float) -> None:
        """Add one write's duration and the file's growth to the counters.

        Parameters
        ----------
        started : float
            ``time.perf_counter()`` when the write began.
        """
        self._counters.seconds += time.perf_counter() - started
        # A file removed while open has no size to report; the counters stand.
        with contextlib.suppress(OSError):
            self._bytes_written = self.path.stat().st_size - self._start_size

    def _require_file(self) -> h5py.File:
        """Return the open file.

        Returns
        -------
        h5py.File
            The file.
        """
        self.ensure_writable()
        assert self._file is not None
        return self._file

    def _shared(self, value: np.ndarray | str) -> h5py.Dataset:
        """Store an immutable value once in the session and return its dataset.

        Parameters
        ----------
        value : numpy.ndarray or str
            An array (stored with its dtype and shape) or JSON text.

        Returns
        -------
        h5py.Dataset
            The stored dataset, to hard-link where it is used.
        """
        shared = cast("h5py.Group", self._require_session()["shared"])
        if isinstance(value, str):
            key = "json_" + _digest(value.encode())
            if key not in shared:
                shared.create_dataset(key, data=value, dtype=h5py.string_dtype())
        else:
            header = f"{value.dtype.str}{value.shape}".encode()
            key = "array_" + _digest(header + np.ascontiguousarray(value).tobytes())
            if key not in shared:
                shared.create_dataset(key, data=value)
                self._counters.array_bytes += value.nbytes
        return cast("h5py.Dataset", shared[key])

    def _dataset(
        self, group: h5py.Group, name: str, value: np.ndarray, **attrs: str
    ) -> None:
        """Write one array exactly, uncompressed and contiguous, with its labels.

        Parameters
        ----------
        group : h5py.Group
            Where it goes.
        name : str
            Dataset name.
        value : numpy.ndarray
            The array; its dtype and shape are kept.
        **attrs : str
            Unit and axis labels.
        """
        dataset = group.create_dataset(name, data=value)
        for key, label in attrs.items():
            dataset.attrs[key] = label
        self._counters.array_bytes += value.nbytes

    def begin_evaluation(
        self,
        labels: Mapping[str, object],
        *,
        prewindow: Float[np.ndarray, "n_nodes n_lattice"],
        drives: Sequence[tuple[int, int, Float[np.ndarray, "_ _"]]],
        gate: Float[np.ndarray, " n_lattice"] | None,
        binding: Mapping[str, object],
    ) -> int:
        """Begin one evaluation group with its labels and complete input.

        Written before the first provider call of the evaluation, so an evaluation
        that raises still has its input.

        Parameters
        ----------
        labels : mapping
            Scalar labels: role, trial, z, density, identities, time map.
        prewindow : numpy.ndarray
            The evaluation's full nodal field in V/m, before slicing and gating.
        drives : sequence of tuple
            ``(row_start, row_stop, drive)`` for every provider chunk, with the exact
            float64 field it will be passed, V/m.
        gate : numpy.ndarray or None
            The atomic window's gate on the lattice, or None.
        binding : mapping
            Build, execution, plan and chunk records, stored once per session.

        Returns
        -------
        int
            The evaluation's identifier, increasing through the file.

        Raises
        ------
        TDSEArchiveError
            If the input cannot be written.
        """
        file = self._require_file()
        started = time.perf_counter()
        identifier = self._next_evaluation
        try:
            evaluations = cast("h5py.Group", self._require_session()["evaluations"])
            group = evaluations.create_group(f"evaluation_{identifier:09d}")
            self._next_evaluation += 1
            self._counters.evaluations += 1
            group.attrs["evaluation_id"] = identifier
            group.attrs["session"] = self._index
            group.attrs["n_chunks"] = len(drives)
            group.attrs["chunk_rows"] = np.asarray(
                [(start, stop) for start, stop, _ in drives], dtype=np.int64
            ).reshape(-1, 2)
            for key, value in labels.items():
                if value is not None:
                    group.attrs[key] = value
            group.attrs["absent_labels"] = json.dumps(
                sorted(key for key, value in labels.items() if value is None)
            )
            group["binding"] = self._shared(structured_json(binding))
            self._dataset(
                group,
                "prewindow_field",
                _admit_array(prewindow, "prewindow_field"),
                unit="V/m",
                axes="n_nodes n_lattice",
            )
            if gate is not None:
                group["gate"] = self._shared(_admit_array(gate, "gate"))
            for index, (start, stop, drive) in enumerate(drives):
                chunk = group.create_group(f"chunk_{index:03d}")
                chunk.attrs["row_start"] = start
                chunk.attrs["row_stop"] = stop
                self._dataset(
                    chunk,
                    "drive",
                    _admit_array(drive, "drive"),
                    unit="V/m",
                    axes="n_row n_time",
                )
            file.flush()
        except _ARCHIVE_ERRORS as error:
            raise self._fail(error, f"evaluation {identifier}") from error
        finally:
            self._measure(started)
        self._open_evaluations[identifier] = group
        return identifier

    def _group(self, evaluation: int) -> h5py.Group:
        """Return an evaluation group begun in this session.

        Parameters
        ----------
        evaluation : int
            Its identifier.

        Returns
        -------
        h5py.Group
            The group.
        """
        try:
            return self._open_evaluations[evaluation]
        except KeyError:
            msg = f"evaluation {evaluation} was not begun in this archive session"
            raise TDSEArchiveError(msg) from None

    def write_result(
        self, evaluation: int, chunk: int, raw: object, schema: RecordSchema
    ) -> None:
        """Store one provider chunk's returned record, before any validation.

        Parameters
        ----------
        evaluation : int
            The evaluation identifier.
        chunk : int
            The chunk index.
        raw : object
            The provider's returned record, exactly as returned.
        schema : RecordSchema
            Its explicit serializer.

        Raises
        ------
        TDSEArchiveError
            If the record does not match its serializer, or cannot be written.
        """
        file = self._require_file()
        started = time.perf_counter()
        try:
            group = cast("h5py.Group", self._group(evaluation)[f"chunk_{chunk:03d}"])
            stored = group.create_group("raw")
            self._write_record(stored, raw, schema, "raw")
            file.flush()
        except _ARCHIVE_ERRORS as error:
            raise self._fail(error, f"evaluation {evaluation} chunk {chunk}") from error
        finally:
            self._measure(started)

    def _write_record(
        self, group: h5py.Group, raw: object, schema: RecordSchema, where: str
    ) -> None:
        """Store every field of one record under its serializer.

        Parameters
        ----------
        group : h5py.Group
            The record's group.
        raw : object
            The record.
        schema : RecordSchema
            Its serializer.
        where : str
            Its path, for errors.
        """
        group.attrs["serializer"] = schema.serializer
        absent: list[str] = []
        for item in schema.fields:
            path = f"{where}.{item.name}"
            try:
                value = getattr(raw, item.name)
            except AttributeError as error:
                msg = f"cannot serialize {path}: the record has no such field"
                raise TDSEArchiveError(msg) from error
            if value is None:
                if not item.optional:
                    msg = f"cannot serialize {path}: a required field is None"
                    raise TDSEArchiveError(msg)
                absent.append(item.name)
                continue
            self._write_field(group, item, value, path)
        group.attrs["absent"] = json.dumps(absent)

    def _write_field(
        self, group: h5py.Group, item: RecordField, value: object, where: str
    ) -> None:
        """Store one present field.

        Parameters
        ----------
        group : h5py.Group
            The record's group.
        item : RecordField
            The field's description.
        value : object
            Its value.
        where : str
            Its path, for errors.
        """
        if item.kind == "array":
            array = _admit_array(value, where)
            if item.name == "time":
                group[item.name] = self._shared(array)
            else:
                self._dataset(group, item.name, array, unit=item.unit, axes=item.axes)
        elif item.kind == "scalar":
            self._dataset(group, item.name, _admit_scalar(value, where), unit=item.unit)
        elif item.kind == "labels":
            self._dataset(group, item.name, _admit_labels(value, where), axes="n_order")
        elif item.kind == "outcomes":
            group.create_dataset(
                item.name, data=_outcomes_json(value, where), dtype=h5py.string_dtype()
            )
        elif item.kind == "structured":
            group[item.name] = self._shared(structured_json(value))
        else:
            assert item.record is not None
            nested = group.create_group(item.name)
            self._write_record(nested, value, item.record, where)

    def write_failure(
        self,
        evaluation: int,
        *,
        chunk: int | None,
        error: BaseException,
        classification: str,
    ) -> None:
        """Store the exception that ended an evaluation.

        Parameters
        ----------
        evaluation : int
            The evaluation identifier.
        chunk : int or None
            The chunk whose call or validation raised, or None.
        error : BaseException
            The exception.
        classification : str
            ``native``, ``interface``, ``contract`` or ``unexpected``.

        Raises
        ------
        TDSEArchiveError
            If the record cannot be written; the caller keeps its own error primary.
        """
        file = self._require_file()
        started = time.perf_counter()
        try:
            group = self._group(evaluation).create_group("exception")
            group.attrs["type"] = type(error).__qualname__
            group.attrs["message"] = str(error)
            group.attrs["classification"] = classification
            group.attrs["chunk"] = -1 if chunk is None else chunk
            file.flush()
        except _ARCHIVE_ERRORS as failure:
            raise self._fail(failure, f"evaluation {evaluation}'s failure") from failure
        finally:
            self._measure(started)

    def finish_evaluation(
        self, evaluation: int, outcome: Literal["returned", "raised"]
    ) -> None:
        """Mark an evaluation complete: the last write of its group.

        Parameters
        ----------
        evaluation : int
            The evaluation identifier.
        outcome : {'returned', 'raised'}
            Whether every chunk returned and validated, or the evaluation raised.

        Raises
        ------
        TDSEArchiveError
            If the attribute cannot be written.
        """
        file = self._require_file()
        started = time.perf_counter()
        try:
            group = self._group(evaluation)
            group.attrs["outcome"] = outcome
            file.flush()
            del self._open_evaluations[evaluation]
        except _ARCHIVE_ERRORS as error:
            raise self._fail(error, f"evaluation {evaluation}'s outcome") from error
        finally:
            self._measure(started)

    def record_trial(
        self, trial: int, *, accepted: bool, evaluations: Sequence[int]
    ) -> None:
        """Append the decision on one optical trial.

        Parameters
        ----------
        trial : int
            The trial's identifier in this session.
        accepted : bool
            Whether the stepper accepted it.
        evaluations : sequence of int
            The evaluations made in it.

        Raises
        ------
        TDSEArchiveError
            If the record cannot be written.
        """
        file = self._require_file()
        started = time.perf_counter()
        try:
            trials = cast("h5py.Group", self._require_session()["trials"])
            group = trials.create_group(f"trial_{trial:09d}")
            group.attrs["trial_id"] = trial
            group.attrs["outcome"] = "accepted" if accepted else "rejected"
            group.attrs["evaluations"] = np.asarray(evaluations, dtype=np.int64)
            file.flush()
        except _ARCHIVE_ERRORS as error:
            raise self._fail(error, f"trial {trial}") from error
        finally:
            self._measure(started)

    def close(self) -> None:
        """Record the session's totals and close the file; repeated calls are harmless.

        Raises
        ------
        TDSEArchiveError
            If totals or closure fail. Closure is always attempted; a handle that
            cannot close retains writer ownership until a successful retry.
        """
        file = self._file
        if file is None:
            self._suspended_identity = None
            self._writer_lock.release()
            self._closed = True
            return
        session = self._session
        self._file = None
        self._closed = True
        started = time.perf_counter()
        failure: TDSEArchiveError | None = None
        try:
            if self._failure is None:
                assert session is not None
                session.attrs["ended"] = _now()
                session.attrs["array_bytes"] = self._counters.array_bytes
                session.attrs["evaluations"] = self._counters.evaluations
                session.attrs["write_seconds"] = self._counters.seconds
                file.flush()
                self._measure(started)
                session.attrs["bytes_written"] = self._bytes_written
                session.attrs["closed"] = True
        except _WRITE_ERRORS as error:
            failure = self._fail(error, "the session totals")
            raise failure from error
        finally:
            try:
                file.close()
            except _WRITE_ERRORS as error:
                self._file = file
                if failure is None:
                    raise self._fail(error, "archive closure") from error
                failure.add_note(f"archive handle closure also failed: {error}")
            else:
                self._session = None
                self._suspended_identity = None
                self._writer_lock.release()
            logger.info(
                "TDSE archive %s: %d evaluations, %d bytes written in %.3f s",
                self.path,
                self._counters.evaluations,
                self._bytes_written,
                self._counters.seconds,
            )


def writer_memory(*, rows: int, samples: int) -> dict[str, int]:
    """Return an archive writer's bounded host buffers, for memory admission.

    Parameters
    ----------
    rows : int
        Largest provider chunk, rows.
    samples : int
        Largest per-row sample count.

    Returns
    -------
    dict
        ``archive.metadata_cache``, the HDF5 metadata cache cap
        (:data:`ARCHIVE_METADATA_CACHE_BYTES`), and ``archive.conversion_buffer``,
        one copy of the largest stored array (an ``(n_row, n_order, n_time)``
        float64 shadow of up to three orders), bytes. The writer keeps no array
        after a write returns.

    Examples
    --------
    >>> from solprop.output.tdse_archive import writer_memory
    >>> writer_memory(rows=2, samples=100)["archive.conversion_buffer"]
    4800
    """
    return {
        "archive.metadata_cache": ARCHIVE_METADATA_CACHE_BYTES,
        "archive.conversion_buffer": 8 * 3 * rows * samples,
    }


def _disk_full(error: BaseException) -> bool:
    """Return whether an error reports a full disk.

    Parameters
    ----------
    error : BaseException
        A write error.

    Returns
    -------
    bool
        True for ``ENOSPC``.
    """
    return isinstance(error, OSError) and error.errno == errno.ENOSPC


def _cap_metadata_cache(file: h5py.File) -> None:
    """Bound the open file's HDF5 metadata cache by the admitted size.

    Parameters
    ----------
    file : h5py.File
        The open archive.
    """
    config = file.id.get_mdc_config()
    config.set_initial_size = True
    config.initial_size = ARCHIVE_METADATA_CACHE_BYTES // 4
    config.min_size = ARCHIVE_METADATA_CACHE_BYTES // 4
    config.max_size = ARCHIVE_METADATA_CACHE_BYTES
    file.id.set_mdc_config(config)


def _next_evaluation_id(sessions: h5py.Group) -> int:
    """Return the identifier after every evaluation already in the file.

    Parameters
    ----------
    sessions : h5py.Group
        The file's sessions.

    Returns
    -------
    int
        One more than the largest stored identifier, or 0.
    """
    largest = -1
    for name in sessions:
        try:
            evaluations = cast("h5py.Group", sessions[name])["evaluations"]
            names = list(cast("h5py.Group", evaluations))
        except KeyError, OSError:
            continue
        for entry in names:
            largest = max(largest, int(entry.removeprefix("evaluation_")))
    return largest + 1


# --------------------------------------------------------------------------- reader


@dataclass(frozen=True)
class ArchivedChunk:
    """Describe one provider call of an archived evaluation.

    Attributes
    ----------
    index : int
        Chunk index within the evaluation.
    row_start, row_stop : int
        Its radial rows, start inclusive, stop exclusive.
    serializer : str or None
        The serializer of its stored record, or None when no record was stored
        (the call raised, or the group is incomplete).
    absent : tuple of str
        Optional top-level fields the provider returned as None.
    """

    index: int
    row_start: int
    row_stop: int
    serializer: str | None
    absent: tuple[str, ...]


@dataclass(frozen=True)
class ArchivedEvaluation:
    """Describe one archived provider evaluation without loading its arrays.

    Attributes
    ----------
    evaluation_id : int
        Increasing through the file.
    session : int
        The session it belongs to.
    trial : int or None
        The optical trial within its session, or None for a consistency diagnostic.
    role : {'startup', 'predictor', 'corrector', 'diagnostic', 'rejected'}
        The evaluation's role, ``rejected`` for a predictor or corrector whose
        trial the stepper then rejected.
    evaluated_role : str
        The role as evaluated, before any rejection.
    trial_outcome : {'accepted', 'rejected'} or None
        The stepper's decision on the trial, or None when none was recorded.
    z : float or None
        Optical position in m, or None for a consistency diagnostic.
    density : float or None
        Number density at the evaluation in m^-3, or None when not supplied.
    labels : mapping
        Every stored label: identities, model, output selections and time map.
    chunks : tuple of ArchivedChunk
        The provider calls, in row order.
    outcome : {'returned', 'raised'} or None
        How the evaluation ended; None for an incomplete group.
    error : mapping or None
        The exception record of an evaluation that raised.
    complete : bool
        Whether the group holds everything its outcome promises; False for a
        group truncated by a crash or a failed write.
    """

    evaluation_id: int
    session: int
    trial: int | None
    role: _Roles
    evaluated_role: str
    trial_outcome: Literal["accepted", "rejected"] | None
    z: float | None
    density: float | None
    labels: Mapping[str, object]
    chunks: tuple[ArchivedChunk, ...]
    outcome: Literal["returned", "raised"] | None
    error: Mapping[str, object] | None
    complete: bool


@dataclass(frozen=True)
class ArchivedSession:
    """Describe one session: a run, a resume or a standalone consistency check.

    Attributes
    ----------
    index : int
        Position in the file.
    kind : {'run', 'resume', 'consistency_check'}
        What it records.
    run_uuid : str
        The run's identity.
    started : str
        ISO 8601 start time.
    outputs : {'full', 'required'}
        The output selection.
    optical_reference : mapping or None
        The archive's path as the optical file records it, or None.
    context : mapping
        The session's descriptor, quadrature, execution and provider facts.
    trials : mapping of int to str
        Each decided trial's outcome.
    closed : bool
        Whether the session closed normally and recorded its totals.
    totals : mapping
        ``bytes_written``, ``array_bytes``, ``write_seconds`` and ``evaluations``
        of a closed session; empty otherwise.
    """

    index: int
    kind: SessionKind
    run_uuid: str
    started: str
    outputs: ArchiveOutputs
    optical_reference: Mapping[str, object] | None
    context: Mapping[str, object]
    trials: Mapping[int, str]
    closed: bool
    totals: Mapping[str, object]


def _attr(group: h5py.HLObject, name: str) -> object:
    """Return an attribute as a plain Python value.

    Parameters
    ----------
    group : h5py.HLObject
        The owner: a file, group or dataset.
    name : str
        Attribute name.

    Returns
    -------
    object
        The value, with NumPy scalars as Python ones and arrays as lists.
    """
    value = group.attrs[name]
    if isinstance(value, np.ndarray):
        return cast("object", value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _text(dataset: object) -> str:
    """Return the text of a string dataset.

    Parameters
    ----------
    dataset : object
        An h5py string dataset.

    Returns
    -------
    str
        Its text.
    """
    return str(cast("h5py.Dataset", dataset).asstr()[()])


class TDSEArchiveReader:
    """Read a TDSE archive one evaluation at a time.

    Built by :func:`open_tdse_archive`. Enumerating sessions and evaluations reads
    only their labels; arrays are loaded on request, one at a time, so a reader never
    holds a whole run. Use it as a context manager or call :meth:`close`.

    Parameters
    ----------
    path : pathlib.Path or str
        The archive.

    Raises
    ------
    TDSEArchiveError
        If the file is not a TDSE archive of a known schema.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        try:
            self._file = h5py.File(self._path, "r")
        except OSError as error:
            msg = f"cannot read the TDSE archive {self._path}: {error}"
            raise TDSEArchiveError(msg) from error
        schema = self._file.attrs.get("schema")
        if schema != TDSE_ARCHIVE_SCHEMA or "sessions" not in self._file:
            self._file.close()
            msg = f"{self._path} is not a {TDSE_ARCHIVE_SCHEMA} archive ({schema!r})"
            raise TDSEArchiveError(msg)

    def __enter__(self) -> "TDSEArchiveReader":  # noqa: UP037 - runtime type hook
        """Return the reader.

        Returns
        -------
        TDSEArchiveReader
            Itself.
        """
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the file.

        Parameters
        ----------
        kind, value, traceback : object
            The exception being propagated, if any.
        """
        self.close()

    def close(self) -> None:
        """Close the file; repeated calls are harmless."""
        self._file.close()

    @property
    def schema(self) -> str:
        """Return the archive's schema identifier.

        Returns
        -------
        str
            :data:`TDSE_ARCHIVE_SCHEMA`.
        """
        return str(_attr(self._file, "schema"))

    @property
    def run_uuid(self) -> str:
        """Return the run's identity, shared with its optical file.

        Returns
        -------
        str
            The UUID.
        """
        return str(_attr(self._file, "run_uuid"))

    def _sessions(self) -> list[h5py.Group]:
        """Return the session groups in order.

        Returns
        -------
        list of h5py.Group
            The groups.
        """
        sessions = cast("h5py.Group", self._file["sessions"])
        return [cast("h5py.Group", sessions[name]) for name in sorted(sessions)]

    @property
    def sessions(self) -> tuple[ArchivedSession, ...]:
        """Return every session of the archive, in order.

        Returns
        -------
        tuple of ArchivedSession
            Their labels, trial decisions and totals.
        """
        return tuple(self._session(group) for group in self._sessions())

    def _session(self, group: h5py.Group) -> ArchivedSession:
        """Read one session's labels.

        Parameters
        ----------
        group : h5py.Group
            The session group.

        Returns
        -------
        ArchivedSession
            Its description.
        """
        trials: dict[int, str] = {}
        if "trials" in group:
            for name in cast("h5py.Group", group["trials"]):
                try:
                    trial = cast("h5py.Group", group["trials"])[name]
                    trials[int(cast("int", _attr(trial, "trial_id")))] = str(
                        _attr(trial, "outcome")
                    )
                except KeyError, OSError:
                    continue
        totals = {
            name: _attr(group, name)
            for name in ("bytes_written", "array_bytes", "write_seconds", "evaluations")
            if name in group.attrs
        }
        reference = json.loads(str(_attr(group, "optical_reference")))
        return ArchivedSession(
            index=int(cast("int", _attr(group, "index"))),
            kind=cast("SessionKind", _attr(group, "kind")),
            run_uuid=str(_attr(group, "run_uuid")),
            started=str(_attr(group, "started")),
            outputs=cast("ArchiveOutputs", _attr(group, "outputs")),
            optical_reference=reference,
            context=json.loads(_text(group["context"])),
            trials=trials,
            closed=bool(group.attrs.get("closed", False)),
            totals=totals,
        )

    def evaluations(self) -> Iterator[ArchivedEvaluation]:
        """Yield every evaluation of every session in order, labels only.

        Yields
        ------
        ArchivedEvaluation
            One per evaluation group; an incomplete group is yielded with
            ``complete=False``, not raised.
        """
        for group in self._sessions():
            trials = self._session(group).trials
            evaluations = group.get("evaluations")
            if not isinstance(evaluations, h5py.Group):
                continue
            for name in sorted(evaluations):
                yield _evaluation(evaluations, name, trials)

    @property
    def truncated(self) -> bool:
        """Return whether any evaluation group is incomplete.

        Returns
        -------
        bool
            True when a crash or a failed write left a group incomplete.
        """
        return any(not item.complete for item in self.evaluations())

    def _evaluation_group(self, evaluation: ArchivedEvaluation) -> h5py.Group:
        """Return the group of an evaluation.

        Parameters
        ----------
        evaluation : ArchivedEvaluation
            The evaluation.

        Returns
        -------
        h5py.Group
            Its group.
        """
        session = self._sessions()[evaluation.session]
        evaluations = cast("h5py.Group", session["evaluations"])
        return cast(
            "h5py.Group", evaluations[f"evaluation_{evaluation.evaluation_id:09d}"]
        )

    def load(
        self, evaluation: ArchivedEvaluation, name: str, *, chunk: int | None = None
    ) -> np.ndarray:
        """Load one array of an evaluation, exactly as stored.

        Parameters
        ----------
        evaluation : ArchivedEvaluation
            The evaluation, from :meth:`evaluations`.
        name : str
            ``"prewindow_field"`` or ``"gate"`` without a chunk; with a chunk,
            ``"drive"`` or a path into the raw record, such as ``"total_current"``,
            ``"plasma/total_current"`` or ``"shadow/current"``.
        chunk : int or None, optional
            The chunk index, for a chunk's arrays.

        Returns
        -------
        numpy.ndarray
            The array with its stored dtype and shape (zero-dimensional for a
            scalar).

        Raises
        ------
        TDSEArchiveError
            If no such array was stored.
        """
        group = self._evaluation_group(evaluation)
        if chunk is None:
            path = name
        elif name == "drive":
            path = f"chunk_{chunk:03d}/drive"
        else:
            path = f"chunk_{chunk:03d}/raw/{name}"
        item = group.get(path)
        if not isinstance(item, h5py.Dataset) or item.dtype.kind in "OSU":
            msg = f"evaluation {evaluation.evaluation_id} stores no array {path!r}"
            raise TDSEArchiveError(msg)
        return cast("np.ndarray", item[()])

    def raw(self, evaluation: ArchivedEvaluation, chunk: int) -> dict[str, object]:
        """Load one chunk's whole stored record.

        Parameters
        ----------
        evaluation : ArchivedEvaluation
            The evaluation.
        chunk : int
            The chunk index.

        Returns
        -------
        dict
            Field name to array (stored dtype), scalar, order labels (tuple),
            outcomes (list of dicts), ``info`` (dict) or nested record (dict); an
            absent optional field is ``None``. The ``serializer`` key names the
            serializer.

        Raises
        ------
        TDSEArchiveError
            If the chunk stored no record.
        """
        group = self._evaluation_group(evaluation).get(f"chunk_{chunk:03d}/raw")
        if not isinstance(group, h5py.Group):
            msg = f"evaluation {evaluation.evaluation_id} chunk {chunk} has no record"
            raise TDSEArchiveError(msg)
        return _read_record(group)


def _evaluation(
    evaluations: h5py.Group, name: str, trials: Mapping[int, str]
) -> ArchivedEvaluation:
    """Read one evaluation's labels, flagging anything incomplete.

    Parameters
    ----------
    evaluations : h5py.Group
        The session's evaluations.
    name : str
        The group's name.
    trials : mapping
        The session's trial decisions.

    Returns
    -------
    ArchivedEvaluation
        Its description.
    """
    identifier = int(name.removeprefix("evaluation_"))
    try:
        group = cast("h5py.Group", evaluations[name])
        labels = {key: _attr(group, key) for key in group.attrs}
        chunks, chunks_complete = _chunks(group, labels)
        error = None
        if "exception" in group:
            exception = cast("h5py.Group", group["exception"])
            error = {key: _attr(exception, key) for key in exception.attrs}
    except KeyError, OSError, ValueError, TypeError:
        return _broken(identifier, evaluations)
    outcome = cast("Literal['returned', 'raised'] | None", labels.get("outcome"))
    complete = (
        outcome is not None
        and "prewindow_field" in group
        and (chunks_complete if outcome == "returned" else error is not None)
    )
    trial = cast("int | None", labels.get("trial_id"))
    evaluated = str(labels.get("role", "diagnostic"))
    decided = None if trial is None else trials.get(trial)
    role = (
        "rejected"
        if decided == "rejected" and evaluated in ("predictor", "corrector")
        else evaluated
    )
    return ArchivedEvaluation(
        evaluation_id=identifier,
        session=int(cast("int", labels.get("session", -1))),
        trial=trial,
        role=cast("_Roles", role),
        evaluated_role=evaluated,
        trial_outcome=cast("Literal['accepted', 'rejected'] | None", decided),
        z=cast("float | None", labels.get("z")),
        density=cast("float | None", labels.get("density")),
        labels=labels,
        chunks=chunks,
        outcome=outcome,
        error=error,
        complete=bool(complete),
    )


def _broken(identifier: int, evaluations: h5py.Group) -> ArchivedEvaluation:
    """Describe an evaluation group that cannot be read at all.

    Parameters
    ----------
    identifier : int
        Its identifier.
    evaluations : h5py.Group
        The session's evaluations, for its session index.

    Returns
    -------
    ArchivedEvaluation
        Marked incomplete, with no labels.
    """
    session = evaluations.parent
    index = int(cast("int", _attr(session, "index")))
    return ArchivedEvaluation(
        evaluation_id=identifier,
        session=index,
        trial=None,
        role="diagnostic",
        evaluated_role="unreadable",
        trial_outcome=None,
        z=None,
        density=None,
        labels={},
        chunks=(),
        outcome=None,
        error=None,
        complete=False,
    )


def _chunks(
    group: h5py.Group, labels: Mapping[str, object]
) -> tuple[tuple[ArchivedChunk, ...], bool]:
    """Read an evaluation's chunk descriptions.

    Parameters
    ----------
    group : h5py.Group
        The evaluation group.
    labels : mapping
        Its labels, with ``n_chunks``.

    Returns
    -------
    tuple
        The chunks present, and whether every expected chunk stored its input and
        its record.
    """
    expected = int(cast("int", labels.get("n_chunks", 0)))
    chunks: list[ArchivedChunk] = []
    complete = True
    for index in range(expected):
        item = group.get(f"chunk_{index:03d}")
        if not isinstance(item, h5py.Group) or "drive" not in item:
            complete = False
            continue
        raw = item.get("raw")
        stored = isinstance(raw, h5py.Group) and "absent" in raw.attrs
        complete = complete and stored
        chunks.append(
            ArchivedChunk(
                index=index,
                row_start=int(cast("int", _attr(item, "row_start"))),
                row_stop=int(cast("int", _attr(item, "row_stop"))),
                serializer=str(_attr(cast("h5py.Group", raw), "serializer"))
                if stored
                else None,
                absent=tuple(json.loads(str(_attr(cast("h5py.Group", raw), "absent"))))
                if stored
                else (),
            )
        )
    return tuple(chunks), complete


def _read_record(group: h5py.Group) -> dict[str, object]:
    """Read one stored record back into plain containers.

    Parameters
    ----------
    group : h5py.Group
        The record's group.

    Returns
    -------
    dict
        Field name to value, ``None`` for an absent field.
    """
    record: dict[str, object] = {"serializer": str(_attr(group, "serializer"))}
    for name in json.loads(str(_attr(group, "absent"))):
        record[name] = None
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            record[name] = _read_record(item)
        elif item.dtype.kind in "OSU":
            text = _text(item)
            record[name] = json.loads(text)
        elif name in ("orders", "shadow_orders"):
            record[name] = tuple(int(value) for value in item[()])
        else:
            record[name] = item[()]
    return record


def open_tdse_archive(path: Path | str) -> TDSEArchiveReader:
    """Open a TDSE archive for streaming reads.

    Parameters
    ----------
    path : pathlib.Path or str
        The archive written beside a run.

    Returns
    -------
    TDSEArchiveReader
        The reader; close it, or use it as a context manager.

    Raises
    ------
    TDSEArchiveError
        If the file cannot be read or is not a TDSE archive.

    Examples
    --------
    >>> from solprop.output import open_tdse_archive
    >>> with open_tdse_archive(path) as archive:  # doctest: +SKIP
    ...     for evaluation in archive.evaluations():
    ...         current = archive.load(evaluation, "total_current", chunk=0)
    """
    return TDSEArchiveReader(path)
