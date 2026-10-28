# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Leave the OpenMP runtime's threads unbound, and say so, unless the user chose.

Kokkos's OpenMP backend prints a four-line notice to standard error when it starts with
``OMP_PROC_BIND`` unset, recommending ``spread`` and ``threads``. Every first native run
printed it. Binding brought the native backend no measurable gain on one socket; it ties
the main thread to one CPU, which solprop then has to undo for ``numba_cpu``
(``solprop._process.allowed_cpus``), and on several sockets whole processes are better
placed with ``numactl`` or ``taskset``. Unbound threads are also what both GNU's and
LLVM's runtimes give when nothing is set.

So, before the core starts Kokkos, the import sets ``OMP_PROC_BIND=false``: the threads
stay where the scheduler puts them, as before, and the notice is not printed. The
setting stays in the environment, so that a runtime that reads it later and a solprop
worker process run alike. Any of ``OMP_PROC_BIND``, ``OMP_PLACES``,
``GOMP_CPU_AFFINITY`` and ``KMP_AFFINITY`` already set is the user's choice of
placement, and nothing is changed.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

VARIABLES = ("OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY", "KMP_AFFINITY")
"""The variables by which a user chooses where OpenMP's threads run."""


def proc_bind(environ: MutableMapping[str, str] = os.environ) -> str:
    """Set ``OMP_PROC_BIND=false`` in `environ` unless the user chose; say which.

    Returns ``"false"`` when it was set here, else ``"environment: NAME=value"`` for
    the first of :data:`VARIABLES` the user set.
    """
    for name in VARIABLES:
        value = environ.get(name, "").strip()
        if value:
            return f"environment: {name}={value}"
    environ["OMP_PROC_BIND"] = "false"
    return "false"
