# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save the field of the low-storage march's Cartesian cases on ``native``, at this
thread count.

Run by ``test_native_low_storage.py::test_the_march_agrees_across_thread_counts`` in a
subprocess, because the Kokkos runtime reads its thread count once per process
(``backends/matched.py``). The first argument is the ``.npz`` file to write, the others
the case names, ``"<mode>-<collar>"`` with the collar ``"plain"`` or ``"collar"``.
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

from backends.test_low_storage import cartesian_march
from backends.test_numba_low_storage import replay

# The doctest collection imports every module under tests/, and the arguments it
# was given are not case names: only a subprocess run saves the fields.


def _runtime_threads() -> np.ndarray:
    """Return the Kokkos pool this probe ran with, for the caller to check."""
    return np.array(
        importlib.import_module("solprop_native").build_info().runtime_threads
    )


if __name__ == "__main__":
    # The optional plugin is required only by this explicitly launched probe.
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    warnings.simplefilter("ignore")
    fields = {}
    for index, name in enumerate(sys.argv[2:]):
        mode, collar = name.split("-")
        problem, grid, _ = cartesian_march(mode, collar=collar == "collar")
        fields[f"case{index}"] = replay(NativeBackend, problem, grid)
    np.savez(sys.argv[1], runtime_threads=_runtime_threads(), **fields)
