# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: a user's run in a fresh process, with a time limit, under each thread setting."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
python, backend = sys.argv[1], sys.argv[2]
# What the pool executor sets in a worker before its interpreter starts (pool.py).
def worker(width):
    return dict.fromkeys(("NUMBA_NUM_THREADS", "SOLPROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), width)
for label, env in (("no-settings", {}), ("scan-worker-1", worker("1")), ("scan-worker-2", worker("2")), ("scan-worker-4", worker("4"))):
    for n_r in (12, 14, 16, 24, 28):
        try:
            done = subprocess.run([python, os.path.join(here, "arm_real.py"), str(n_r), backend], env={**os.environ, **env}, capture_output=True, text=True, timeout=150)
        except subprocess.TimeoutExpired:
            print(backend, label, n_r, "HANG", flush=True)
            continue
        text = (done.stdout + done.stderr).splitlines()
        tail = [x for x in text if x.startswith("finished") or "Error" in x or "fatal" in x][-1:] or text[-1:]
        print(backend, label, n_r, "exit", done.returncode, " ".join(tail)[:170], flush=True)
