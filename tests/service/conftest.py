# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixtures of the service's tests: one finished run, made once per session.

No test reaches a real Slurm, wherever the suite runs: the ``SLURM_*`` variables of an
enclosing job are removed, and the Slurm commands found first on ``PATH`` are ones that
refuse, so no test can ask a real scheduler anything (or cancel the job the suite
itself runs in). A test that fakes Slurm puts its own commands before them.
"""

import os
import shutil
import stat
import sys
import warnings
from pathlib import Path

import pytest

from service.cases import MULTIMODE, STATS, TINY
from solprop.recipe import Recipe

_SLURM_COMMANDS = ("sacct", "salloc", "sbatch", "scancel", "sinfo", "squeue", "srun")


@pytest.fixture(scope="session")
def _no_slurm_bin(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a folder of Slurm commands that refuse, each saying so."""
    folder = tmp_path_factory.mktemp("no-slurm")
    for name in _SLURM_COMMANDS:
        path = folder / name
        path.write_text(
            f'#!/bin/sh\necho "{name}: the service tests reach no real Slurm" >&2\n'
            "exit 1\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return folder


@pytest.fixture(autouse=True)
def _no_real_slurm(_no_slurm_bin: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep every test away from the Slurm of the machine the suite runs on."""
    for name in [name for name in os.environ if name.startswith("SLURM_")]:
        monkeypatch.delenv(name)
    if sys.platform != "win32":
        monkeypatch.setenv("PATH", f"{_no_slurm_bin}{os.pathsep}{os.environ['PATH']}")


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
