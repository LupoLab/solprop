# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spike only: one LAPACK call in a fresh process. Exit 0 means it survived."""
import faulthandler, sys, threading
faulthandler.enable()
how, kind, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
import numpy as np
if "limit" in how:
    import threadpoolctl
    threadpoolctl.threadpool_limits(1)
rng = np.random.default_rng(n)
a = rng.standard_normal((n, n))
if "complex" in kind:
    a = a + 1j * rng.standard_normal((n, n))
b = rng.standard_normal(n)
def call():
    if "solve" in kind:
        np.linalg.solve(a, b)
    elif kind == "inv":
        np.linalg.inv(a)
    elif kind == "det":
        np.linalg.det(a)
    elif kind == "eig":
        np.linalg.eig(a)
    elif kind == "svd":
        np.linalg.svd(a)
    elif kind == "qr":
        np.linalg.qr(a)
    elif kind == "scipy":
        import scipy.linalg
        scipy.linalg.solve(a, b)
    elif kind == "matmul":
        a @ a
if "thread" in how:
    threading.stack_size(256 * 1024 * 1024)
    worker = threading.Thread(target=call)
    worker.start()
    worker.join()
else:
    call()
