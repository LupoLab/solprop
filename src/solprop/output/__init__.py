# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output handlers, the run record they write, and the files a run leaves.

- The record of a run (layer S, imported by translation and the backends without the
  handlers): :mod:`~solprop.output.records` (the record types of a saved run and its
  planes), :mod:`~solprop.output.metadata`
  (:class:`~solprop.output.metadata.RunMetadata` and the resume key),
  :mod:`~solprop.output.serialisation`, :mod:`~solprop.output.provenance`,
  :mod:`~solprop.output.store` (the save protocol,
  :class:`~solprop.output.store.Output`, and field retention) and
  :mod:`~solprop.output.capability` (what a saved basis can be asked).
- The handlers: :mod:`~solprop.output.memory`
  (:class:`~solprop.output.memory.MemoryOutput`), :mod:`~solprop.output.hdf5`
  (:class:`~solprop.output.hdf5.HDF5Output` with :func:`~solprop.output.hdf5.open_file`,
  :func:`~solprop.output.hdf5.read_resume_state` and
  :func:`~solprop.output.hdf5.check_cache`) and :mod:`~solprop.output.callback`
  (:class:`~solprop.output.callback.CallbackOutput`, for a run reduced plane by plane).
- The other files a run leaves: :mod:`~solprop.output.tdse_archive` (the raw TDSE
  response archive), :mod:`~solprop.output.snapshots` (checkpoints published while a run
  computes), :mod:`~solprop.output.resumable` (whether a file can continue), and a
  scan's collected file and per-point artifacts (:mod:`~solprop.output.scan`,
  :mod:`~solprop.output.scan_artifacts`).

Every handler shares one save protocol, one metadata schema and one sign convention on
the way out: stored fields are in the optics convention and carry the attribute
``spectral_convention = "optics"`` (``docs/dev/conventions.md`` §2, ADR-017). The
manual page is ``docs/guide/results/output.md``. The package's names are imported on
first use (:pep:`562`), so that importing a record module does not load h5py or the
handlers.
"""

import importlib
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from solprop.output.callback import CallbackOutput
    from solprop.output.hdf5 import (
        HDF5Output,
        ResumeState,
        check_cache,
        open_file,
        read_resume_state,
    )
    from solprop.output.memory import MemoryOutput
    from solprop.output.metadata import SCHEMA_VERSION, RunMetadata
    from solprop.output.provenance import BackendRecord, GitProvenance
    from solprop.output.records import (
        GridRecord,
        ModeRecord,
        NoiseRecord,
        RunOutcome,
        SavedPlane,
        StepHistory,
        StepLimits,
        StepperRecord,
    )
    from solprop.output.resumable import Resumability, resumability
    from solprop.output.snapshots import (
        SnapshotController,
        SnapshotFiles,
        SnapshotInfo,
        SnapshotResult,
        read_snapshot,
        restore_snapshot,
    )
    from solprop.output.store import Output
    from solprop.output.tdse_archive import (
        TDSEArchive,
        TDSEArchiveReader,
        open_tdse_archive,
    )

__all__ = [
    "SCHEMA_VERSION",
    "BackendRecord",
    "CallbackOutput",
    "GitProvenance",
    "GridRecord",
    "HDF5Output",
    "MemoryOutput",
    "ModeRecord",
    "NoiseRecord",
    "Output",
    "Resumability",
    "ResumeState",
    "RunMetadata",
    "RunOutcome",
    "SavedPlane",
    "SnapshotController",
    "SnapshotFiles",
    "SnapshotInfo",
    "SnapshotResult",
    "StepHistory",
    "StepLimits",
    "StepperRecord",
    "TDSEArchive",
    "TDSEArchiveReader",
    "check_cache",
    "open_file",
    "open_tdse_archive",
    "read_resume_state",
    "read_snapshot",
    "restore_snapshot",
    "resumability",
]

_HOMES: Final = {
    "SCHEMA_VERSION": "metadata",
    "BackendRecord": "provenance",
    "CallbackOutput": "callback",
    "GitProvenance": "provenance",
    "GridRecord": "records",
    "HDF5Output": "hdf5",
    "MemoryOutput": "memory",
    "ModeRecord": "records",
    "NoiseRecord": "records",
    "Output": "store",
    "Resumability": "resumable",
    "ResumeState": "hdf5",
    "RunMetadata": "metadata",
    "RunOutcome": "records",
    "SavedPlane": "records",
    "SnapshotController": "snapshots",
    "SnapshotFiles": "snapshots",
    "SnapshotInfo": "snapshots",
    "SnapshotResult": "snapshots",
    "StepHistory": "records",
    "StepLimits": "records",
    "StepperRecord": "records",
    "TDSEArchive": "tdse_archive",
    "TDSEArchiveReader": "tdse_archive",
    "check_cache": "hdf5",
    "open_file": "hdf5",
    "open_tdse_archive": "tdse_archive",
    "read_resume_state": "hdf5",
    "read_snapshot": "snapshots",
    "restore_snapshot": "snapshots",
    "resumability": "resumable",
}
"""The submodule each name of ``__all__`` is defined in, imported on first access."""

_SUBMODULES: Final = (
    "callback",
    "capability",
    "hdf5",
    "memory",
    "metadata",
    "provenance",
    "records",
    "resumable",
    "scan",
    "scan_artifacts",
    "serialisation",
    "snapshots",
    "store",
    "tdse_archive",
)
"""The submodules reachable as attributes; a test keeps it equal to what is on disk."""


def __getattr__(name: str) -> object:
    """Import a name or submodule the first time it is asked for (:pep:`562`).

    Parameters
    ----------
    name : str
        The attribute looked up on the package.

    Returns
    -------
    object
        The name from the submodule that defines it, or the submodule itself.

    Raises
    ------
    AttributeError
        If `name` is neither a public name nor a submodule of the package.
    """
    if name in _HOMES:
        module = importlib.import_module(f"{__name__}.{_HOMES[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    """List the package's public surface, whether or not it has been loaded yet.

    Returns
    -------
    list of str
        The sorted names ``dir`` and tab completion show: ``__all__``, the submodules
        and the dunders.
    """
    names = {*globals(), *__all__, *_SUBMODULES}
    return sorted(
        name
        for name in names
        if name.startswith("__") or name in __all__ or name in _SUBMODULES
    )
