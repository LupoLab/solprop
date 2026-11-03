#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check an installed solprop outside any checkout: its version, ``auto``, a saved run.

``python -I native/ci/check_install.py VERSION BACKEND``. SEP 43 section 7's fresh
installs and minimum systems, and ``wheels.yml``'s distributions.
"""

from __future__ import annotations

import platform
import sys
import tempfile
from pathlib import Path

import optmat

import solprop
from solprop.backends.registry import available, resolve_name
from solprop.driver import prop_gnlse

version, backend = sys.argv[1:]
print("solprop", solprop.__version__, "with optmat", optmat.__version__)
print("on", platform.platform(), platform.machine(), "Python", platform.python_version())
print("backends:", available(), "- auto means", resolve_name("auto"))
assert solprop.__version__ == version, f"expected solprop {version}"
assert resolve_name("auto") == backend, f"expected auto to mean {backend}"
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "run.h5"
    prop_gnlse(
        0.11,
        1e-4,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        fwhm=50e-15,
        peak_power=1e3,
        shape="sech",
        output=path,
        progress="none",
    )
    assert path.stat().st_size > 0
print("a short run propagated and saved")
