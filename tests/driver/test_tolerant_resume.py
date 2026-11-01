# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resuming by what matters to the run: numerics, finished files and force (SEP 42).

Through real files and the driver's gate, on the short soliton fibre: a changed
tolerance continues and is recorded, a finished file is returned only under the
numerics it ran, a changed physics is refused unless forced, and a forced file is
marked so that later sessions and readers respect it.
"""

import json
import logging
import warnings

import h5py
import pytest

from march_bounds import assert_in_envelope
from solprop.driver.simulation import _check_resume_choice
from solprop.exceptions import BoundaryWarning, OutputError, SolpropWarning
from solprop.log import NullReporter
from solprop.output.hdf5 import open_file, resume
from solprop.stepper import RungeKutta
from solprop.stepper.multistep import Multistep

# Tight time windows for speed, as in test_resume.py: the boundary monitor's warning
# is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LENGTH = 0.01  # m, a sixteenth of the soliton period
N_SAVE = 5  # the entrance and four planes
RTOL = 1e-8


class CrashError(RuntimeError):
    """What an interrupted run raises, standing in for a machine going away."""


class Interrupt(NullReporter):
    """A progress reporter that stops the run once it has seen enough planes."""

    def __init__(self, after):
        self.after = after
        self.seen = 0

    def plane(self, z, summary):
        self.seen += 1
        if self.seen >= self.after:
            raise CrashError(f"stopped after plane {self.seen}")


def fibre(build_simulation, **keywords):
    """The short soliton fibre, four planes, at `RTOL` unless overridden."""
    return build_simulation(
        **({"length": LENGTH, "n_save": N_SAVE, "rtol": RTOL} | keywords)
    )


def stop(simulation, path, *, after=2, **keywords):
    """Run into a file and stop once `after` planes of this session have been saved."""
    with pytest.raises(CrashError):
        simulation.run(path, progress=Interrupt(after), **keywords)


def entries(path):
    """Return the ``resume`` entries a file's sessions wrote, oldest first."""
    outcome = resume(path).outcome
    assert outcome is not None
    return outcome.resumes


def quietly(simulation, path, **keywords):
    """Run without letting a warning through, so a test sees the ones it expects."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", SolpropWarning)
        warnings.simplefilter("ignore", BoundaryWarning)
        return simulation.run(path, **keywords)


# ------------------------------------------------------------------- numerics


def test_a_run_resumes_under_a_looser_tolerance_and_records_it(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation), path)
    stopped = resume(path).n_saved
    looser = fibre(build_simulation, rtol=10 * RTOL)
    with pytest.warns(SolpropWarning, match="with changed numerics") as caught:
        handler = looser.run(path, resume=True)
    assert "stepper: controller.rtol 1e-08 -> 1e-07" in str(caught[0].message)
    [entry] = entries(path)
    assert entry["numerics"] == {"stepper.controller.rtol": [RTOL, 10 * RTOL]}
    assert entry["forced"] is False
    assert entry["compared"] == ""
    assert entry["from_plane"] == stopped - 1
    # The session held the file's record, which the file's stamp still matches.
    stored = resume(path)
    assert handler.metadata.configuration_hash() == stored.metadata.configuration_hash()
    assert stored.n_saved == N_SAVE
    assert stored.outcome is not None and stored.outcome.status == "complete"


def test_a_resumed_run_meets_the_envelope_of_its_looser_tolerance(
    build_simulation, tmp_path
):
    """A continuation under two tolerances is no better than the looser (SEP 42 §4).

    Class T: runs stopped part-way and resumed under the other of 1e-8 and 1e-9, each
    against the fibre at a hundredth of the looser tolerance, within the envelope of
    the uninterrupted run at the looser one (``tests/march_bounds.py``). Measured on
    ``numpy_ref``: 2.2e-11 and 2.6e-10 against the envelope's 5.1e-10. A loosening to
    1e-7 is not tested this way: the uninterrupted 1e-7 run happens to land at 3.7e-11,
    nearer the reference than the 1e-8 one, so its envelope does not describe a
    trajectory restarted at the join, which lands at 9e-9, well inside its tolerance.
    """
    looser = RTOL
    fibre(build_simulation, rtol=looser).run(tmp_path / "whole.h5")
    fibre(build_simulation, rtol=looser / 100).run(tmp_path / "tight.h5")
    for first, second in ((RTOL, RTOL / 10), (RTOL / 10, RTOL)):
        path = tmp_path / f"part-{first:g}.h5"
        stop(fibre(build_simulation, rtol=first), path)
        with pytest.warns(SolpropWarning, match="with changed numerics"):
            fibre(build_simulation, rtol=second).run(path, resume=True)

    def last(name):
        return open_file(tmp_path / name)["field"][-1]

    assert_in_envelope(
        [last(f"part-{RTOL:g}.h5"), last(f"part-{RTOL / 10:g}.h5")],
        last("tight.h5"),
        last("whole.h5"),
        what="the soliton",
    )


def test_a_finished_file_is_returned_only_under_the_numerics_it_ran(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    fibre(build_simulation).run(path)
    handler = quietly(fibre(build_simulation), path, resume=True)
    assert handler.n_saved == N_SAVE
    # A tightened tolerance must not hand back the loose result.
    with pytest.raises(
        OutputError, match="not computed throughout under the requested"
    ):
        fibre(build_simulation, rtol=RTOL / 10).run(path, resume=True)


def test_a_file_finished_under_other_numerics_matches_neither(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation), path)
    with pytest.warns(SolpropWarning, match="with changed numerics"):
        fibre(build_simulation, rtol=10 * RTOL).run(path, resume=True)
    # The stored record still matches the original script, but neither tolerance
    # ran the whole file.
    for rtol in (RTOL, 10 * RTOL):
        with pytest.raises(OutputError, match="not computed throughout"):
            fibre(build_simulation, rtol=rtol).run(path, resume=True)


def test_a_file_stopped_after_its_last_plane_has_no_work_left(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    fibre(build_simulation).run(path)
    # As if killed after its last plane: the status still reads running.
    with h5py.File(path, "r+") as file:
        outcome = json.loads(file["meta/outcome"][()])
        outcome["status"] = "running"
        del file["meta/outcome"]
        file["meta/outcome"] = json.dumps(outcome)
    handler = quietly(fibre(build_simulation), path, resume=True)
    assert handler.n_saved == N_SAVE


def test_a_file_with_an_unknown_group_and_attribute_reads_and_resumes(
    build_simulation, tmp_path
):
    # What a later release of the same schema may add without a bump (SEP 42 §5.8):
    # a top-level group written once and a root attribute.
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation), path)
    with h5py.File(path, "r+") as file:
        file.create_group("recipe").attrs["source"] = "a later release"
        file.attrs["future_attribute"] = 1
    handler = quietly(fibre(build_simulation), path, resume=True)
    assert handler.n_saved == N_SAVE
    with open_file(path) as reader:
        assert reader.n_saved == N_SAVE


@pytest.mark.parametrize("value", ["yes", "Force", 1, None])
def test_a_mistyped_resume_is_refused(value):
    # A mistyped word would otherwise read as True. Checked on the helper, because the
    # test session's runtime type check refuses it at the call before the driver does.
    with pytest.raises(ValueError, match="resume must be False, True"):
        _check_resume_choice(value)
    for valid in (False, True, "interpolated", "force"):
        _check_resume_choice(valid)


# ------------------------------------------------- the numerics behind the gate


def test_an_adams_run_continues_under_runge_kutta(build_simulation, tmp_path):
    # The charged ledger stays with the earlier execution as provenance; a
    # Runge-Kutta march keeps none (SEP 42 section 5.6).
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation, integrator=Multistep()), path)
    charged = resume(path).outcome
    assert charged is not None and charged.logical_calls
    with pytest.warns(SolpropWarning, match="with changed numerics"):
        handler = fibre(build_simulation, integrator=RungeKutta()).run(
            path, resume=True
        )
    assert handler.n_saved == N_SAVE
    outcome = resume(path).outcome
    assert outcome is not None and outcome.status == "complete"
    assert not outcome.logical_calls
    assert outcome.previous_executions[-1]["logical_calls"] == charged.logical_calls
    [entry] = outcome.resumes
    assert "stepper.method" in entry["numerics"]


def test_a_raised_call_budget_keeps_the_adams_history(
    build_simulation, tmp_path, caplog
):
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation, integrator=Multistep()), path)
    assert resume(path).march is not None
    with (
        caplog.at_level(logging.INFO, logger="solprop.driver.simulation"),
        pytest.warns(SolpropWarning, match="with changed numerics"),
    ):
        fibre(build_simulation, integrator=Multistep(max_calls=10**6)).run(
            path, resume=True
        )
    assert "continuing the Adams history" in caplog.text
    assert "starts afresh" not in caplog.text


def test_a_changed_order_starts_the_adams_history_afresh(build_simulation, tmp_path):
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation, integrator=Multistep()), path)
    with pytest.warns(SolpropWarning) as caught:
        fibre(build_simulation, integrator=Multistep(order=4)).run(path, resume=True)
    messages = [str(warning.message) for warning in caught]
    assert any("the Adams history starts afresh" in one for one in messages)
    assert any("history was kept by" in one for one in messages)


# ----------------------------------------------------------------------- force


def test_a_changed_physics_is_refused_and_can_be_forced(build_simulation, tmp_path):
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation), path)
    changed = fibre(build_simulation, gamma=0.12)
    with pytest.raises(OutputError, match='pass resume="force"') as refused:
        changed.run(path, resume=True)
    assert "these compared values differ: configuration:" in str(refused.value)
    with pytest.warns(SolpropWarning, match="forcing a resume") as caught:
        handler = changed.run(path, resume="force")
    message = str(caught[0].message)
    assert "The Adams history starts afresh" in message
    assert "anything rebuilt from the record does not" in message
    assert handler.n_saved == N_SAVE
    [entry] = entries(path)
    assert entry["forced"] is True
    assert "configuration:" in str(entry["compared"])
    # A complete forced file is returned to the original script, with a warning.
    with pytest.warns(SolpropWarning, match="was continued with resume='force'"):
        fibre(build_simulation).run(path, resume=True)


def test_a_stopped_force_is_respected_by_later_sessions(build_simulation, tmp_path):
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation), path)
    changed = fibre(build_simulation, gamma=0.12)
    with pytest.warns(SolpropWarning, match="forcing a resume"):
        stop(changed, path, after=1, resume="force")
    forced_at = resume(path).outcome.forced_positions  # pyright: ignore[reportOptionalMemberAccess] - a stopped run has an outcome
    assert len(forced_at) == 1
    # The original script matches the stored record, but the file is no longer its
    # run: it is refused, by position.
    with pytest.raises(OutputError, match="was continued with resume='force' at z ="):
        fibre(build_simulation).run(path, resume=True)
    with pytest.warns(SolpropWarning, match="forcing a resume"):
        handler = fibre(build_simulation, gamma=0.12).run(path, resume="force")
    assert handler.n_saved == N_SAVE


def test_force_does_not_cross_the_layout(build_simulation, tmp_path):
    path = tmp_path / "run.h5"
    stop(fibre(build_simulation), path)
    with pytest.raises(OutputError, match="the layout differs") as refused:
        fibre(build_simulation, n_save=N_SAVE + 1).run(path, resume="force")
    assert "write a new file" in str(refused.value)


def test_force_needs_work_left(build_simulation, tmp_path):
    path = tmp_path / "run.h5"
    fibre(build_simulation).run(path)
    with pytest.raises(OutputError, match="continues only a file with work left"):
        fibre(build_simulation, gamma=0.12).run(path, resume="force")
