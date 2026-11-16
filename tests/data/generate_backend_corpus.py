# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate ``tests/data/backend_corpus.npz``: both backends' runs, pinned.

Recorded from the tree as it stood before SEP 52 restructured the reference and native's
seam (its phase 2, on top of phase 1's removal of the groups), and committed. It is what
the restructured backends are compared with (SEP 52 section 7, "A pinned corpus
first"; ``tests/backends/test_backend_corpus.py``): every case of
``tests/backend_corpus_cases.py`` on ``numpy_ref`` and on the native CPU build, each
backend held to its own record.

Regenerating it would bless whatever the tree then computes. SEP 52 section 4 allows it
only for a change of the reference's numbers that its phase report records -- the
computation it changes, the largest difference on these fixed schedules against that
computation's SEP 25 budget, and whether native follows -- and that is accepted on that
evidence. So the script refuses to overwrite the file unless given ``--force``, and the
commit that uses it names the change. It needs the native package installed.

Every thread pool is pinned to one thread, the default lane's setting, so that the
record's bits are those a one-thread run repeats; the comparison reports bitwise
agreement as a diagnostic only on the same build and thread setting
(:func:`backend_corpus_cases.identity`), and fails only beyond a rounding budget or an
accuracy envelope.

Usage::

    uv run python tests/data/generate_backend_corpus.py [--force]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS))

# Before anything reads them: one thread per pool, as the default lane runs, and no
# machine calibration (the suite's own setting, tests/conftest.py).
os.environ["SOLPROP_CALIBRATION"] = "off"
for _name in (
    "SOLPROP_THREADS",
    "SOLPROP_NATIVE_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ[_name] = "1"

import importlib  # noqa: E402 - after the environment above

# The native runtime loads before solprop's FFTs, as in tests/conftest.py.
importlib.import_module("solprop_native")

import json  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from backend_corpus_cases import (  # noqa: E402
    CASES,
    THREAD_VARIABLES,
    identity,
    record,
)
from march_bounds import measured_scales  # noqa: E402

OUT = Path(__file__).with_name("backend_corpus.npz")

BACKENDS = ("numpy_ref", "native")


def _commit() -> str:
    """Return the checkout's head commit, marked when its sources are modified."""
    root = TESTS.parent
    head = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", "src", "native/src"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


def main() -> int:
    """Write the corpus, refusing to overwrite it without ``--force``."""
    if OUT.exists() and "--force" not in sys.argv[1:]:
        sys.stderr.write(
            f"{OUT} exists; pass --force only for a change SEP 52 section 4 records\n"
        )
        return 1
    assert set(THREAD_VARIABLES) <= set(os.environ)
    arrays: dict[str, Any] = {}
    cases: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as directory:
        for case in CASES:
            # The launch's nonlinear phase and the source's conditioning, which the
            # comparison's rounding budget is formed from (march_bounds.march_bound),
            # read once by the reference before anything is compared.
            phi, kappa = measured_scales(case.build("numpy_ref"))
            entry: dict[str, Any] = {
                "route": case.route,
                "march": case.march,
                "phi": phi,
                "kappa": kappa,
                "backends": {},
            }
            for backend in BACKENDS:
                found = record(case, backend, Path(directory))
                for name, value in found.arrays.items():
                    arrays[f"{case.name}/{backend}/{name}"] = value
                entry["backends"][backend] = found.records
                sys.stdout.write(f"{case.name} on {backend}\n")
            cases[case.name] = entry
    records = {
        "generated_from": _commit(),
        "identity": {backend: identity(backend) for backend in BACKENDS},
        "cases": cases,
    }
    text = json.dumps(records, sort_keys=True, allow_nan=False)
    arrays["records"] = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(OUT, **arrays)
    sys.stdout.write(f"wrote {OUT}: {len(cases)} cases, {OUT.stat().st_size} bytes\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
