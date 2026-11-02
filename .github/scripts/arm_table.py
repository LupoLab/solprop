# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: run arm_one.py across a table, one fresh process a cell, with a time limit."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
python = sys.argv[1]
print("cpus", os.cpu_count(), flush=True)
def cell(env, how, kind, n, limit=45):
    full = {**os.environ, **env}
    try:
        code = subprocess.run([python, os.path.join(here, "arm_one.py"), how, kind, str(n)], env=full, capture_output=True, timeout=limit).returncode
    except subprocess.TimeoutExpired:
        return "HANG"
    return {0: ".", 3221225477: "AV"}.get(code, str(code))
def row(label, env, how, kinds, sizes):
    for kind in kinds:
        print(label, kind, " ".join(f"{n}={cell(env, how, kind, n)}" for n in sizes), flush=True)
fine = list(range(2, 66)) + [80, 96, 128, 192, 256, 512]
usual = [16, 24, 32, 64, 128, 256]
if sys.argv[2] == "fine":
    row("env1", {"OPENBLAS_NUM_THREADS": "1"}, "plain", ["solve"], fine)
    row("limit1", {}, "limit", ["solve"], fine)
    row("env1-thread", {"OPENBLAS_NUM_THREADS": "1"}, "thread", ["solve"], usual)
    row("limit1-thread", {}, "limit-thread", ["solve"], usual)
elif sys.argv[2] == "widths":
    for width in ("2", "3", "4"):
        row("env" + width, {"OPENBLAS_NUM_THREADS": width}, "plain", ["solve", "inv", "matmul"], usual)
    row("limit2", {}, "limit2", ["solve", "inv", "matmul"], usual)
    row("unset", {}, "plain", ["solve", "inv", "matmul"], usual)
else:
    row("numpy-" + sys.argv[2], {"OPENBLAS_NUM_THREADS": "1"}, "plain", ["solve"], usual)
