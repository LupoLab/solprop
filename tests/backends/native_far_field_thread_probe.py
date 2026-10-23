# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Save the Kokkos far-field statistics of one state at this thread count.

Run by the thread-count test of ``test_native_far_field.py`` in a subprocess,
because the Kokkos runtime reads its thread count once per process
(``backends/matched.py``). The first argument is the ``.npz`` file to write, the
others the statistics' names.
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

# The doctest collection imports every module under tests/, and the arguments it
# was given are not names: only a subprocess run saves the statistics.
if __name__ == "__main__":
    from backends.test_native_far_field import native_stats, thread_case

    warnings.simplefilter("ignore")
    grid, problem, field = thread_case()
    names = sys.argv[2:]
    values, _ = native_stats(problem, grid, field, names)
    np.savez(
        sys.argv[1],
        **{f"case{index}": values[name] for index, name in enumerate(names)},
    )
