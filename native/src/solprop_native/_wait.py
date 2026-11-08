# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How the OpenMP runtime's idle threads wait between parallel regions (ADR-071).

A native step opens tens of short parallel regions. A team whose idle threads sleep at
once wakes them for each: 16 us a region at 6 threads and 55 us at 18 on an Apple M5
Max (LLVM's runtime), 36 us at 16 threads and 72 us at 36 on two Intel Xeon Gold 6240
(GNU's). A team that spins a little before sleeping enters the next region in 0.6 to
12 us. Over windows of the benchmark cases a short spin ran them in about 0.8 of the
time on the Mac's six top-tier cores, the two Xeon Gold 6240 and 16 cores of a shared
AMD EPYC 7763 node, and in 0.87 on 16 cores of an AMD EPYC 9B45 under Windows
(ADR-071).

ADR-056 had made idle threads sleep at once, after spinning cost an EPYC 7763 1.4 to
3.3 times at 10,000 spins while the backend ran several thread teams in turn (Kokkos's,
the threaded FFTs', the BLAS's, NumPy's and numba's). Those now share one runtime and
NumPy's BLAS holds one thread, and the same test no longer loses.

So, unless the user chose, the import sets before the core loads the runtime:

- ``GOMP_SPINCOUNT=10000`` for GNU's runtime (Linux), which reads it once for the
  process: about ten microseconds of spinning before a thread sleeps.
- ``KMP_BLOCKTIME=0`` for LLVM's (macOS, Windows, and Linux builds with Clang): idle
  threads sleep at once outside native work, and the core spins them for
  :data:`SPIN_MILLISECONDS` while a native call runs (``kmp_set_blocktime``).

Each runtime ignores the other's variable. Idle threads sleep at once
(``OMP_WAIT_POLICY=passive``), as before, where the team is wider than a chip's
fastest cores: spinning on an M5 Max's
second-tier cores ran most cases 1.1 to 1.35 times slower at 18 threads. Any of
:data:`VARIABLES` already set is the user's choice, and nothing is changed; a process
started from this one inherits the settings with :data:`MARKER`, which tells them from
a user's.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

VARIABLES = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME")
"""The variables by which a user chooses how OpenMP's idle threads wait."""

SPIN_COUNT = "10000"
"""GNU's runtime's spins before an idle thread sleeps, by default (ADR-071)."""

SPIN_MILLISECONDS = 1
"""LLVM's runtime's block time inside native calls, by default (ADR-071)."""

MARKER = "SOLPROP_NATIVE_WAIT"
"""Set beside the variables this module sets, naming the choice it made."""

REQUEST = "SOLPROP_NATIVE_WAIT_REQUEST"
"""solprop's own request for a process, ``spin`` or ``passive`` (solprop SEP 46).

A scan's workers and a calibration's children ask for a wait through it. A user's
:data:`VARIABLES` still win; the request replaces a calibration's and the default
rule's choice.
"""

_SETTINGS = {
    "spin": {"GOMP_SPINCOUNT": SPIN_COUNT, "KMP_BLOCKTIME": "0"},
    "passive": {"OMP_WAIT_POLICY": "passive"},
}


@dataclass(frozen=True)
class WaitPolicy:
    """How idle threads wait, as chosen at import.

    ``description`` goes into a run's metadata; ``spin_milliseconds`` is the block time
    the core keeps inside native calls on LLVM's runtime (zero: the process's).
    """

    description: str
    spin_milliseconds: int = 0


def _ours(environ: MutableMapping[str, str]) -> bool:
    """Whether the wait variables set are exactly those this module set."""
    settings = _SETTINGS.get(environ.get(MARKER, ""))
    if settings is None:
        return False
    return all(
        environ.get(name, "").strip() == settings.get(name, "") for name in VARIABLES
    )


def requested_wait(environ: Mapping[str, str] = os.environ) -> str | None:
    """Return :data:`REQUEST`'s ``spin`` or ``passive``, or ``None``.

    Raises
    ------
    ValueError
        For another value.
    """
    value = environ.get(REQUEST, "").strip()
    if not value:
        return None
    if value not in _SETTINGS:
        raise ValueError(f"{REQUEST} must be spin or passive, got {value!r}")
    return value


def wait_policy(
    environ: MutableMapping[str, str] = os.environ,
    *,
    threads: int = 1,
    top_tier: int | None = None,
    preferred: str | None = None,
) -> WaitPolicy:
    """Set the default wait in `environ` unless the user chose; say which.

    `threads` is the native pool's size and `top_tier` the cores of the chip's fastest
    kind where it has several (``None`` otherwise). `preferred`, ``"spin"`` or
    ``"passive"``, is a machine calibration's choice (solprop SEP 46), which replaces
    the default rule. Returns ``"spin"`` or ``"passive"`` when set here, else
    ``"environment: NAME=value"`` for the first of :data:`VARIABLES` the user set.
    """
    if not _ours(environ):
        for name in VARIABLES:
            value = environ.get(name, "").strip()
            if value:
                return WaitPolicy(f"environment: {name}={value}")
    for name in (*VARIABLES, MARKER):
        environ.pop(name, None)
    wider = top_tier is not None and threads > top_tier
    choice = preferred if preferred in _SETTINGS else "passive" if wider else "spin"
    environ.update(_SETTINGS[choice])
    environ[MARKER] = choice
    return WaitPolicy(choice, SPIN_MILLISECONDS if choice == "spin" else 0)
