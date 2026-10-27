# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Make the OpenMP runtime's idle threads sleep at once, unless the user chose.

Between parallel regions an OpenMP thread spins for a while before it sleeps: GNU's
runtime by default for about 300,000 spins, LLVM's for 200 ms. The native backend runs
several thread teams in turn (Kokkos', the threaded FFTs' and the BLAS's), and when its
threads fill the CPUs one team's spinning threads take the cores the next team needs.
On 16 cores of an AMD EPYC 7763, with the runtime's own waiting, the four-mode helium
capillary took 334 s against 13.5 s with idle threads sleeping, the argon blueshift
432 s against 37.3 s and the radial filament 72 s against 5.9 s; even 10,000 spins cost
them 1.4 to 3.3 times.

Spinning is faster where hardware threads are spare: on 36 cores of two Intel Xeon Gold
6240 (72 hardware threads) the runtime's own waiting took the supercontinuum from 94 s
to 49 s and the helium capillary from 14.7 s to 9.7 s. But the runtime reads the policy
once, when it loads, and work that arrives on the machine later takes those spare
threads, so a run that spins can end up where the EPYC one did. Sleeping is never
catastrophic, so it is the default, and ``OMP_WAIT_POLICY=active`` is the opt-in for a
machine a run has to itself.

So, before the core loads the runtime, the import sets ``OMP_WAIT_POLICY=passive``,
which both runtimes honour (LLVM's then sets its block time to zero). The setting stays
in the environment, so that a runtime that reads it later (LLVM's, at its first
parallel region; Numba's) and a solprop worker process wait alike. Any of
``OMP_WAIT_POLICY``, ``GOMP_SPINCOUNT`` and ``KMP_BLOCKTIME`` already set is the user's
choice, and nothing is changed.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

VARIABLES = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME")
"""The variables by which a user chooses how OpenMP's idle threads wait."""


def wait_policy(environ: MutableMapping[str, str] = os.environ) -> str:
    """Set the passive wait policy in `environ` unless the user chose; say which.

    Returns ``"passive"`` when it was set here, else ``"environment: NAME=value"`` for
    the first of :data:`VARIABLES` the user set.
    """
    for name in VARIABLES:
        value = environ.get(name, "").strip()
        if value:
            return f"environment: {name}={value}"
    environ["OMP_WAIT_POLICY"] = "passive"
    return "passive"
