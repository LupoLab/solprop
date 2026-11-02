# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: which call crashes, with which libraries loaded."""
import faulthandler, os, sys
faulthandler.enable()
mode = sys.argv[1]
if "native" in mode:
    import solprop_native
    print("native", solprop_native.__file__, flush=True)
import numpy as np
print("numpy", np.__version__, os.environ.get("OPENBLAS_NUM_THREADS"), flush=True)
if mode.endswith("config"):
    np.show_config()
rng = np.random.default_rng(1)
for n in (8, 16, 24, 32, 48, 64, 96, 128, 192, 256):
    for kind in ("real", "complex", "eigvec"):
        a = rng.standard_normal((n, n))
        b = rng.standard_normal(n) + 0j
        if kind != "real":
            a = a + 1j * rng.standard_normal((n, n))
        if kind == "eigvec":
            _, a = np.linalg.eig(a)
        print(n, kind, end=" ", flush=True)
        x = np.linalg.solve(a, b if kind != "real" else b.real)
        print(float(np.max(abs(a @ x - b))), flush=True)
print("done", mode, flush=True)
