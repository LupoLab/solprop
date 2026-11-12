# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""TEMPORARY: the old import path of the records split into five modules; delete it.

This module was split into :mod:`solprop.output.serialisation`,
:mod:`solprop.output.provenance`, :mod:`solprop.output.records`,
:mod:`solprop.output.metadata` and :mod:`solprop.output.store`. It re-exports only the
names that ``driver/simulation.py`` and ``driver/interface.py`` still import from
here while another branch splits those files; the merge rewrites them and deletes this
module.
"""

from solprop.output.metadata import (
    INTERPOLATED_RESUMES_ANNOTATION,
    PEAK_FIELDS_ANNOTATION,
    RunMetadata,
)
from solprop.output.records import (
    DefaultsResolution,
    GridRecord,
    MarchSnapshot,
    ModeRecord,
    NoiseRecord,
    NoiseResolution,
    RecipeRecord,
    StepLimits,
    StepperRecord,
)
from solprop.output.serialisation import (
    StatedInput,
    _recorded_inline,  # pyright: ignore[reportPrivateUsage] - for simulation.py
    recorded_input,
    serialise,
)
from solprop.output.store import (
    CheckpointOutput,
    FieldRetention,
    FieldsChoice,
    Output,
    PlaneStore,
    field_retention_refusal,
)

__all__ = [
    "INTERPOLATED_RESUMES_ANNOTATION",
    "PEAK_FIELDS_ANNOTATION",
    "CheckpointOutput",
    "DefaultsResolution",
    "FieldRetention",
    "FieldsChoice",
    "GridRecord",
    "MarchSnapshot",
    "ModeRecord",
    "NoiseRecord",
    "NoiseResolution",
    "Output",
    "PlaneStore",
    "RecipeRecord",
    "RunMetadata",
    "StatedInput",
    "StepLimits",
    "StepperRecord",
    "_recorded_inline",
    "field_retention_refusal",
    "recorded_input",
    "serialise",
]
