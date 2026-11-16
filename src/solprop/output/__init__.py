# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output handlers and the metadata schema.

In-memory and HDF5 outputs sharing one save protocol
(:class:`solprop.output.store.PlaneStore`), one metadata schema and one sign convention
on the way out: stored fields are in the optics convention and carry the attribute
``spectral_convention = "optics"`` (see ``docs/dev/decisions.md``). The manual page is
``docs/guide/results/output.md``. A run reduced plane by plane stores nothing, and gives
each plane to a callable instead (:class:`~solprop.output.callback.CallbackOutput`;
``docs/guide/setup/running.md``). A TDSE run can also write its raw provider
evaluations to a separate archive
(:class:`~solprop.output.tdse_archive.TDSEArchive`, read with
:func:`~solprop.output.tdse_archive.open_tdse_archive`;
``docs/guide/media/tdse.md``).
"""

from solprop.output.callback import CallbackOutput
from solprop.output.hdf5 import HDF5Output, ResumeState, check_cache, open_file, resume
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
    "read_snapshot",
    "restore_snapshot",
    "resumability",
    "resume",
]
