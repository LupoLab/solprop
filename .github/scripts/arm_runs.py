# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: a user's runs in fresh processes, with a time limit that also ends workers."""
import os, subprocess, sys, tempfile
here = os.path.dirname(os.path.abspath(__file__))
python, backend = sys.argv[1], sys.argv[2]
def go(label, script, args, env, limit):
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as out:
        child = subprocess.Popen([python, os.path.join(here, script), *args], env={**os.environ, **env}, stdout=out, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)], capture_output=True)
            child.wait()
            code = "HANG"
        out.seek(0)
        text = out.read().splitlines()
    keep = [x for x in text if x.startswith("finished") or "Error" in x or "fatal" in x or "exited" in x][-2:] or text[-1:]
    print(backend, label, "exit", code, " ".join(keep)[:300], flush=True)
# What the pool executor sets in a worker before its interpreter starts (pool.py).
def worker(width):
    return dict.fromkeys(("SOLPROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"), width)
for threads in (1, 2, 3, 4):
    for n_r in (18, 24, 28):
        go(f"pool-scan threads={threads} n_r={n_r}", "arm_scan.py", [str(n_r), backend, str(threads), "90"], {}, 90)
go("warning OMP=1", "arm_warn.py", [], {"OMP_NUM_THREADS": "1"}, 60)
go("warning none", "arm_warn.py", [], {}, 60)
