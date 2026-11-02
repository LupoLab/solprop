# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: run arm_one.py across a table, one fresh process a cell, with a time limit."""
import os, subprocess, sys
here = os.path.dirname(os.path.abspath(__file__))
python = sys.argv[1]
print("cpus", os.cpu_count(), flush=True)
errors = {}
def cell(env, how, kind, n, limit=45):
    full = {**os.environ, **env}
    try:
        done = subprocess.run([python, os.path.join(here, "arm_one.py"), how, kind, str(n)], env=full, capture_output=True, text=True, timeout=limit)
    except subprocess.TimeoutExpired:
        return "HANG"
    if done.returncode == 1:
        errors[how] = done.stderr[-400:]
    return {0: ".", 7: "WRONG", 3221225477: "AV"}.get(done.returncode, str(done.returncode))
def row(label, env, how, kinds, sizes):
    for kind in kinds:
        print(label, kind, " ".join(f"{n}={cell(env, how, kind, n)}" for n in sizes), flush=True)
band = list(range(20, 35))
usual = [16, 24, 32, 64, 128, 256]
one = {"OPENBLAS_NUM_THREADS": "1"}
if sys.argv[2] == "vars":
    for name in ("OMP_NUM_THREADS", "GOTO_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        for width in ("1", "2"):
            row(f"{name}={width}", {name: width}, "plain", ["solve"], band)
    row("all-six=1", dict.fromkeys(("SOLPROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"), "1"), "plain", ["solve", "inv"], band)
elif sys.argv[2] == "right":
    row("unset", {}, "plain", ["solve", "complex_solve", "inv"], band + usual)
    row("limit1", {}, "limit", ["solve", "complex_solve", "inv"], band + usual)
    row("limit2", {}, "limit2", ["solve", "complex_solve", "inv"], band + usual)
    row("env3", {"OPENBLAS_NUM_THREADS": "3"}, "plain", ["solve", "complex_solve", "inv"], band + usual)
    row("env4-limit1", {"OPENBLAS_NUM_THREADS": "4"}, "limit", ["solve", "complex_solve", "inv"], band + usual)
    row("env1-stack64M", one, "thread", ["solve"], band)
    row("unset-stack64M", {}, "thread", ["solve"], band)
for how, text in errors.items():
    print("error", how, text.replace("\n", " | "), flush=True)
