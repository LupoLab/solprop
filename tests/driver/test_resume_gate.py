# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A resume continues on an exact configuration hash and otherwise refuses (ADR-066).

Hashes are identity metadata, so every comparison here is exact (class P): a rebuilt run
must give the same bits, in a fresh process too, since a refusal stops a long job.
"""

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

import record_routes
from solprop.cli import main
from solprop.driver import Simulation
from solprop.driver.simulation import _check_resume_choice
from solprop.exceptions import OutputError
from solprop.log import NullReporter
from solprop.output import check_cache, open_file, resumability
from solprop.output.serialisation import serialise
from solprop.recipe import Recipe, resume_file

pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LENGTH = 0.02
"""The soliton's fibre, m: a few steps, on the reference backend."""


def soliton(*, rtol: float = 1e-6, gamma: float = 0.11) -> Simulation:
    """Return a short soliton fibre on the reference backend."""
    return Simulation.from_gnlse(
        gamma,
        LENGTH,
        [0.0, 0.0, -11.83e-27],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        n_save=3,
        backend="numpy_ref",
        absorber=False,
        noise=None,
        rtol=rtol,
    )


def stopped(path: Path) -> None:
    """Run into a file, and stop at the first saved plane."""
    flag = threading.Event()

    class Stop(NullReporter):
        def plane(self, z, summary) -> None:
            flag.set()

    soliton().run(path, progress=Stop(), stop=flag)


def test_a_matching_hash_continues_the_file(tmp_path):
    path = tmp_path / "run.h5"
    stopped(path)
    with open_file(path) as first:
        created = first.metadata.created if first.metadata else None
        planes = first.n_saved
    assert 0 < planes < 3
    soliton().run(path, resume=True, progress="none")
    with open_file(path) as stored:
        assert stored.n_saved == 3
        assert stored.metadata is not None and stored.metadata.created == created
        assert stored.outcome is not None and stored.outcome.status == "complete"
        assert "resume" not in stored.outcome.execution


@pytest.mark.parametrize(
    ("changes", "named"),
    [
        ({"rtol": 1e-5}, r"stepper\.controller\.rtol"),
        ({"gamma": 0.12}, r"gamma"),
    ],
    ids=["numerics", "physics"],
)
def test_a_different_hash_is_refused_by_name_and_leaves_the_file(
    tmp_path, changes, named
):
    path = tmp_path / "run.h5"
    stopped(path)
    before = path.read_bytes()
    with pytest.raises(OutputError, match=rf"cannot be resumed; .*{named}") as refusal:
        soliton(**changes).run(path, resume=True, progress="none")
    assert "new segment from a saved plane" in str(refusal.value)
    assert path.read_bytes() == before


def test_a_finished_file_comes_back_only_under_its_own_hash(tmp_path):
    path = tmp_path / "run.h5"
    soliton().run(path, progress="none")
    before = path.read_bytes()
    handler = soliton().run(path, resume=True, progress="none")
    assert handler.n_saved == 3
    assert path.read_bytes() == before
    # A tightened tolerance must not hand back the loose result.
    with pytest.raises(OutputError, match=r"stepper\.controller\.rtol"):
        soliton(rtol=1e-8).run(path, resume=True, progress="none")


@pytest.mark.parametrize("word", ["force", "yes", 1])
def test_a_resume_that_is_not_one_of_the_three_values_is_refused(word):
    # Directly: the test session's type hook would refuse the word at `run`'s door.
    with pytest.raises(ValueError, match="resume must be False, True or"):
        _check_resume_choice(word)


def test_the_command_line_has_no_force(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(["resume", str(tmp_path / "run.h5"), "--force"])
    assert "--force" in capsys.readouterr().err


def test_the_gate_says_what_a_resume_would_say(tmp_path):
    path = tmp_path / "soliton.h5"
    flag = threading.Event()

    class Stop(NullReporter):
        def plane(self, z, summary) -> None:
            flag.set()

    Recipe.example("soliton").run(path, progress=Stop(), stop=flag)
    answer = resumability(path, gate=True)
    assert answer.resumable and answer.mode is True and answer.differences == ""
    assert resumability(path).differences is None
    resume_file(path, progress="none")
    with open_file(path) as stored:
        assert stored.outcome is not None and stored.outcome.status == "complete"


def test_an_unset_optional_field_is_recorded_at_its_default():
    """A new recorded field needs no step beyond adding it (ADR-051, ADR-066)."""

    @dataclass(frozen=True)
    class Added:
        optional: int | None = None

    assert cast("dict[str, Any]", serialise(Added()))["optional"] is None


def _route_hashes(**environment: str) -> dict[str, str]:
    """Return each route's configuration hash, as a fresh process builds it."""
    script = Path(record_routes.__file__)
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(script.parent), os.environ.get("PYTHONPATH", "")]
            ),
            **environment,
        },
    )
    return json.loads(result.stdout)


@pytest.mark.medium
@pytest.mark.parametrize(
    "environment",
    [
        {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"},
        {"PYTHONHASHSEED": "12345"},
    ],
    ids=["one-thread", "four-threads", "hash-seed"],
)
def test_a_rebuilt_run_has_its_hash_in_a_fresh_process(environment):
    """On each route: a resume is another process, and a hash has no tolerance."""
    built = {
        name: build().metadata().configuration_hash()
        for name, build in record_routes.ROUTES.items()
    }
    assert _route_hashes(**environment) == built


def test_check_cache_says_there_is_nothing_to_resume(tmp_path):
    assert check_cache(tmp_path / "absent.h5", soliton().metadata()) is False
