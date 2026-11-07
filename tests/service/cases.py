# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The recipe and the pipe the service's tests share."""

import multiprocessing
from multiprocessing.connection import Connection
from typing import cast

TINY = """\
[recipe]
format = 1
route = "capillary"
title = "Tiny capillary"

[build]
radius = "125 um"
length = "5 cm"
gas = "He"
pressure = "1 bar"
mode_average = true
wavelength = "800 nm"
wavelength_limits = ["200 nm", "2000 nm"]
trange = "300 fs"
fwhm = "10 fs"
energy = "100 uJ"
n_save = 6
backend = "numpy_ref"
"""
"""A mode-averaged capillary that runs in about a second."""


def pipe(*, duplex: bool = True) -> tuple[Connection, Connection]:
    """Return a pipe's two ends, typed as the service takes them on every platform."""
    return cast("tuple[Connection, Connection]", multiprocessing.Pipe(duplex))
