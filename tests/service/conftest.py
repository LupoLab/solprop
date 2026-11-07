# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixtures of the service's tests: one finished run, made once per session."""

import shutil
from pathlib import Path

import pytest

from service.cases import TINY
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
