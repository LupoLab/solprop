# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run one case of ``forced_thread_cases.py`` and save its arrays and its sizing.

Run by :func:`forced_threads.forced_run` in a subprocess, because the Kokkos runtime
reads its pools once per process (``backends/matched.py``). The first argument is the
``.npz`` file to write, the second the case's name, ``"<builder>"`` or
``"<builder>:<option>"``.
"""

import importlib
import importlib.util
import json
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
# was given are not case names: only a subprocess run saves the arrays.

if __name__ == "__main__":
    from backends.forced_thread_cases import CASES

    warnings.simplefilter("ignore")
    builder, _, option = sys.argv[2].partition(":")
    arrays, sizing = CASES[builder](option) if option else CASES[builder]()
    pool = importlib.import_module("solprop_native").build_info().runtime_threads
    np.savez(sys.argv[1], pool=np.array(pool), sizing=json.dumps(sizing), **arrays)
