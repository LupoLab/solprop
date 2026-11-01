# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output handlers and the metadata schema.

In-memory and HDF5 outputs sharing one save protocol
(:class:`solprop.output.schema.PlaneStore`), one metadata schema and one sign convention
on the way out: stored fields are in the optics convention and carry the attribute
``spectral_convention = "optics"`` (see ``docs/dev/decisions.md``). The manual page is
``docs/guide/results/output.md``. A TDSE run can also write its raw provider
evaluations to a separate archive
(:class:`~solprop.output.tdse_archive.TDSEArchive`, read with
:func:`~solprop.output.tdse_archive.open_tdse_archive`;
``docs/guide/media/tdse.md``).
"""

from solprop.output.hdf5 import (
    COMPRESSIONS,
    HDF5Output,
    ResumeState,
    check_cache,
    open_file,
    resume,
)
from solprop.output.memory import MemoryOutput
from solprop.output.resumability import Resumability, resumability
from solprop.output.schema import (
    FIELD_LAYOUT,
    HISTORY_FIELDS,
    INTERNAL_CONVENTION,
    SCHEMA_VERSION,
    SPECTRAL_CONVENTION,
    BackendRecord,
    CheckpointOutput,
    GitProvenance,
    GridRecord,
    ModeRecord,
    NoiseRecord,
    Output,
    PlaneLayout,
    PlaneStore,
    RunMetadata,
    RunOutcome,
    SavedPlane,
    StepHistory,
    StepLimits,
    StepperRecord,
    TerminalCheckpoint,
    package_versions,
    resolve_metadata_write,
    serialise,
)
from solprop.output.snapshots import (
    SnapshotController,
    SnapshotFiles,
    SnapshotInfo,
    SnapshotResult,
    read_snapshot,
    restore_snapshot,
)
from solprop.output.tdse_archive import (
    TDSE_ARCHIVE_SCHEMA,
    TDSEArchive,
    TDSEArchiveReader,
    open_tdse_archive,
)

__all__ = [
    "COMPRESSIONS",
    "FIELD_LAYOUT",
    "HISTORY_FIELDS",
    "INTERNAL_CONVENTION",
    "SCHEMA_VERSION",
    "SPECTRAL_CONVENTION",
    "TDSE_ARCHIVE_SCHEMA",
    "BackendRecord",
    "CheckpointOutput",
    "GitProvenance",
    "GridRecord",
    "HDF5Output",
    "MemoryOutput",
    "ModeRecord",
    "NoiseRecord",
    "Output",
    "PlaneLayout",
    "PlaneStore",
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
    "TerminalCheckpoint",
    "check_cache",
    "open_file",
    "open_tdse_archive",
    "package_versions",
    "read_snapshot",
    "resolve_metadata_write",
    "restore_snapshot",
    "resumability",
    "resume",
    "serialise",
]
