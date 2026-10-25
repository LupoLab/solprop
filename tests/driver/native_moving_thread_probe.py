# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save each case's state after ten moving-guide steps on ``native`` per thread count.

Run by ``test_native_moving_guide.py``'s thread-count test in a subprocess, because the
Kokkos runtime reads its thread count once per process. The first argument is the
``.npz`` file to write, the rest the case names.
"""

import importlib
import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

# The runtime loads before solprop's FFTs, as in conftest. Without the optional
# wheel the doctest collection still imports this module, and nothing runs.
if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

from driver.test_native_coupled_flow import native_class, propagate
from driver.test_native_moving_guide import CASES

# The doctest collection imports every module under tests/, and the arguments it
# was given are not case names: only a subprocess run writes the states.


def _runtime_threads() -> np.ndarray:
    """Return the Kokkos pool this probe ran with, for the caller to check."""
    return np.array(
        importlib.import_module("solprop_native").build_info().runtime_threads
    )


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    states = {name: propagate(CASES[name](), native_class()) for name in sys.argv[2:]}
    np.savez(
        sys.argv[1], allow_pickle=False, runtime_threads=_runtime_threads(), **states
    )
