# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixtures of the service's tests: one finished run, made once per session."""

import shutil
import warnings
from pathlib import Path

import pytest

from service.cases import MULTIMODE, STATS, TINY
from solprop.recipe import Recipe


@pytest.fixture(scope="session")
def finished_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a completed run's results file, made once per session."""
    path = tmp_path_factory.mktemp("finished") / "run.h5"
    Recipe.loads(TINY).run(path)
    return path


@pytest.fixture
def run_copy(finished_run: Path, tmp_path: Path) -> Path:
    """Return a private copy of the completed run's file."""
    target = tmp_path / "run.h5"
    shutil.copy(finished_run, target)
    return target


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a two-mode run's results file, with statistics, made once."""
    path = tmp_path_factory.mktemp("results") / "run.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Recipe.loads(MULTIMODE.replace("n_save = 3", "n_save = 3" + STATS)).run(path)
    return str(path)


@pytest.fixture(scope="module")
def beam(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Return a Cartesian beam's results file (solprop's example), made once."""
    path = tmp_path_factory.mktemp("beam") / "beam.h5"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Recipe.example("cartesian_beam").run(path)
    return str(path)
