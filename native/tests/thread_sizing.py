# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Hold a thread-count comparison to the widths its operations really took.

Native sizes each operation by the work it holds and records the choice in
``execution_sizing``, so a comparison whose operation stayed on one thread would pass
without testing anything. Every probe here reports its backends' records.
"""

from __future__ import annotations

from collections.abc import Iterable

Sizing = dict[str, dict[str, int]]
"""A backend's ``execution_sizing``: ``"pool.operation.elements"`` to its choice."""


def widest(sizing: Sizing, operation: str) -> int:
    """Return the most threads any run of `operation` took; 0 if it never ran."""
    return max(
        (
            choice["threads"]
            for name, choice in sizing.items()
            if name.startswith(f"{operation}.")
        ),
        default=0,
    )


def assert_threaded(sizings: Iterable[Sizing], threads: int, *operations: str) -> None:
    """Assert every backend of a run took `threads` for each of `operations`."""
    records = list(sizings)
    assert records
    for index, sizing in enumerate(records):
        for operation in operations:
            assert widest(sizing, operation) == threads, (index, operation, sizing)
