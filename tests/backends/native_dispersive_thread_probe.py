# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save the field of ten dispersive steps per case on ``native``, at this thread count.

Run by the thread-count tests of ``test_native_dispersive_kerr.py`` and
``test_native_dispersive_routes.py`` in a subprocess, because the Kokkos runtime reads
its thread count once per process (``backends/matched.py``). The first argument is the
``.npz`` file to write; each further argument is an evaluator (the mode-averaged route)
or ``route:evaluator`` with the route ``free``, ``planar`` or ``general``. Every case
runs at a fixed order on its fixed schedule, so two thread counts take the same
discrete method.
"""

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

# The runtime loads before solprop's FFTs, as in conftest. Without the optional
# wheel the doctest collection still imports this module, and nothing runs.
if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

from backends.matched import fixed_order
from backends.test_numba_dispersive_kerr import averaged, run

# The doctest collection imports every module under tests/, and the arguments it
# was given are not evaluators: only a subprocess run saves the fields.


def _runtime_threads() -> np.ndarray:
    """Return the Kokkos pool this probe ran with, for the caller to check."""
    return np.array(
        importlib.import_module("solprop_native").build_info().runtime_threads
    )


if __name__ == "__main__":
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    warnings.simplefilter("ignore")
    fields = {}
    for index, case in enumerate(sys.argv[2:]):
        if ":" in case:
            from backends.test_native_dispersive_routes import (
                route_case,
            )

            route, evaluator = case.split(":")
            problem, grid = route_case(route, evaluator)
        else:
            problem, grid = averaged(case, envelope=False, thg=True)
        fields[f"case{index}"], _ = run(NativeBackend, fixed_order(problem), grid)
    np.savez(sys.argv[1], runtime_threads=_runtime_threads(), **fields)
