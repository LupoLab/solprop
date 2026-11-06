# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execution backends behind one seam.

The NumPy reference and native backends
(``docs/dev/backends.md``). This package exports the seam itself — the
:class:`~solprop.backends.protocol.Backend` protocol and the record types it returns —
and :mod:`solprop.backends.registry`, which selects a backend by name and says why one
is not usable, and the thread budget every backend's pools are held within.
"""

from solprop._process import ThreadBudget, thread_budget
from solprop.backends.protocol import (
    Backend,
    BackendHistory,
    BatchBackend,
    Observation,
    StepStats,
    StepSummary,
)
from solprop.backends.registry import available, resolve_name, select, select_batch

__all__ = [
    "Backend",
    "BackendHistory",
    "BatchBackend",
    "Observation",
    "StepStats",
    "StepSummary",
    "ThreadBudget",
    "available",
    "resolve_name",
    "select",
    "select_batch",
    "thread_budget",
]
