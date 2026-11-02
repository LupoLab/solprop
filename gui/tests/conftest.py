# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Shared test configuration: widgets are drawn without a display."""

import multiprocessing
import os
import shutil
from multiprocessing.connection import Connection
from pathlib import Path
from typing import cast

import pytest

# Before any Qt application exists: the tests run headless on every platform, and a
# developer's display is never used. An explicit setting (a debugging session) wins.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


@pytest.fixture(scope="session")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a completed run's results file, made once per session."""
    from solprop.recipe import Recipe  # noqa: PLC0415 - only the tests that need it

    path = tmp_path_factory.mktemp("finished") / "run.h5"
    Recipe.loads(TINY).run(path)
    return path


@pytest.fixture
def run_copy(finished_run: Path, tmp_path: Path) -> Path:
    """Return a private copy of the completed run's file."""
    target = tmp_path / "run.h5"
    shutil.copy(finished_run, target)
    return target


def pipe(*, duplex: bool = True) -> tuple[Connection, Connection]:
    """Return a pipe's two ends, typed as the service takes them on every platform."""
    return cast("tuple[Connection, Connection]", multiprocessing.Pipe(duplex))
