# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resume: continuing a file the same run wrote, and refusing one it did not."""

import dataclasses
import re
from typing import Any

import h5py
import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import FIXED_ORDER, assert_in_envelope, assert_matched, march_bound
from solprop.backends.driver import StepDriver
from solprop.backends.registry import select
from solprop.exceptions import BackendError, DriverError, OutputError
from solprop.grid import EnvelopeGrid
from solprop.log import NullReporter
from solprop.output import MemoryOutput
from solprop.output.hdf5 import resume
from solprop.stepper import FixedSchedule

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

# The columns two matched runs of one problem are compared on, within the march's
# rounding budget (SEP 25). The wall time is left out: the same steps are taken and the
# same norms scored, but never in the same number of microseconds.
STORED_COLUMNS = (
    "field",
    "z",
    "stats/energy",
    "stats/omega0",
    "stats/peak_power",
    "stats/fwhm_t",
    "stats/fwhm_t_outer",
    "step_history/z",
    "step_history/h",
    "step_history/norm",
    "step_history/accepted",
    "step_history/counts",
)

# What a replayed schedule fixes exactly, whatever the arithmetic that fills it: the
# positions and the steps it prescribes (a structural contract, compared exactly).
SCHEDULED = ("z", "step_history/z", "step_history/h", "step_history/accepted")

LENGTH = 0.01  # m, a sixteenth of the soliton period
N_SAVE = 5  # the entrance and four planes


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

    def finished(self, total):  # pragma: no cover - the run never gets this far
        raise AssertionError("an interrupted run does not finish")


def item(parent, *names) -> Any:
    """Return one h5py object or serialised entry, loosely typed for inspection."""
    value = parent
    for name in names:
        value = value[name]
    return value


def read_file(path, key):
    """Return one stored array out of a closed HDF5 output file."""
    with h5py.File(path, "r") as file:
        return np.asarray(item(file, *key.split("/"))[()])


def scheduled(build_simulation, **keywords):
    """Return a simulation of the soliton fibre that replays a fixed schedule.

    At the Adams method's fixed top order, so that a resume that continues the
    history is the same discrete computation as the uninterrupted run: the same steps,
    orders and history, differing by rounding alone (SEP 25 section 4.4). On the
    former grid and without a collar: a short fixed schedule is refused beside the
    default collar, and these tests are about resuming, not boundaries.
    """
    planes = np.linspace(0.0, LENGTH, N_SAVE)[1:]
    grid = EnvelopeGrid.from_limits(
        LENGTH, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return build_simulation(
        length=LENGTH,
        grid=grid,
        absorber=bare_absorber(grid),
        n_save=N_SAVE,
        schedule=FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / 40, save_planes=list(planes)
        ),
        **({"integrator": FIXED_ORDER} | keywords),
    )


def budget(simulation):
    """Return the matched-march budget of a scheduled soliton run.

    The launch's nonlinear phase over the sixteenth of a period is ``L / L_NL = 0.88``
    for the third-order soliton, below the module's default declared ``Phi = 1``.
    """
    return march_bound(simulation.translate())


def assert_column(key, got, expected, bound, rtol):
    """Assert one stored column of two matched runs agrees within the march's budget.

    The fields and statistics by :func:`march_bounds.assert_matched` (a statistic
    quadratic in the field, twice the field's budget); the error norm is a scaled
    defect near round-off, ``||d|| / (rtol |y|)``, so the budget relative to the
    state moves it by up to ``bound / rtol`` absolutely.
    """
    if key == "step_history/norm":
        np.testing.assert_allclose(got, expected, rtol=0, atol=bound / rtol)
    elif key == "field":
        assert_matched(got, expected, bound, what=key)
    else:
        assert_matched(got, expected, 2 * bound, what=key)


def afresh(simulation):
    """Return `simulation` set to keep no Adams march state, so a resume restarts it."""
    simulation.settings = dataclasses.replace(simulation.settings, resume_history=False)
    return simulation


def run_until(simulation, path, *, after):
    """Run into a file and stop once ``after`` planes have been saved."""
    with pytest.raises(CrashError):
        simulation.run(path, progress=Interrupt(after))


# ------------------------------------------------------------------- the two runs


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu"])
def test_a_replayed_schedule_resumed_matches_the_uninterrupted_run(
    build_simulation, tmp_path, backend
):
    """Through a second simulation object, as a new process would.

    A replayed schedule takes the same steps whatever it is resumed from: the state
    round trip through the file is a conjugation and a sign flip (ADR-017), the
    controller's own state does not enter a replay, and the kept Adams history makes
    the resumed march the uninterrupted one's. So the positions and the steps agree
    exactly, and the values to the march's rounding budget: ``numba_cpu`` keeps a
    step's last stage as the next one's first across planes (ADR-039), which a resume
    evaluates afresh, and no run is promised bitwise anyway (SEP 25).
    """
    whole, interrupted, again = (
        scheduled(build_simulation, backend=backend) for _ in range(3)
    )
    bound = budget(whole)
    whole.run(tmp_path / "whole.h5")
    run_until(interrupted, tmp_path / "part.h5", after=2)
    assert read_file(tmp_path / "part.h5", "z").size == 3
    again.run(tmp_path / "part.h5", resume=True)
    for key in STORED_COLUMNS:
        expected = read_file(tmp_path / "whole.h5", key)
        got = read_file(tmp_path / "part.h5", key)
        if key in SCHEDULED:
            assert np.array_equal(expected, got), key
        elif key == "step_history/counts":
            # Integer accounting of the prescribed steps: exact.
            assert np.array_equal(expected, got), key
        else:
            assert_column(key, got, expected, bound, rtol=1e-8)


def test_an_adaptive_run_resumed_agrees_to_the_controller_s_tolerance(
    build_simulation, tmp_path, soliton
):
    """With no history kept the controller restarts: an accuracy comparison.

    A resume with no snapshot of the Adams march (``resume_history=False`` here; Kokkos,
    which keeps its history resident, always) starts the history afresh at order one
    and re-derives its step sizes instead of continuing the sequence it was on, so the
    two trajectories differ by the local error the controller was asked for
    (``docs/guide/numerics/adams.md``). Each is compared with the same fibre at a
    hundredth of the tolerance, within four times the uninterrupted run's own error
    against it (SEP 25 section 4.4; ``tests/march_bounds.py``).
    """
    rtol = 1e-8
    whole, interrupted, again, tight = (
        afresh(build_simulation(length=LENGTH, n_save=N_SAVE, rtol=tolerance))
        for tolerance in (rtol, rtol, rtol, rtol / 100)
    )
    whole.run(tmp_path / "whole.h5")
    tight.run(tmp_path / "tight.h5")
    run_until(interrupted, tmp_path / "part.h5", after=2)
    again.run(tmp_path / "part.h5", resume=True)
    uninterrupted = read_file(tmp_path / "whole.h5", "field")[-1]
    resumed = read_file(tmp_path / "part.h5", "field")[-1]
    reference = read_file(tmp_path / "tight.h5", "field")[-1]
    assert_in_envelope([resumed], reference, uninterrupted, what="the resumed soliton")
    # An interruption that changed nothing at all would mean the resume never happened.
    assert soliton.error(resumed, uninterrupted, phase=0.0) > 0.0


def test_an_input_noise_run_resumed_does_not_add_its_noise_twice(
    build_simulation, tmp_path
):
    """The state at a plane already carries the entrance realisation (ADR-008).

    The ``input`` model adds its realisation once, at ``z = 0``. A resumed run loads a
    state from a plane past the entrance, so adding it again would double the shot noise
    and the two files would differ by the realisation itself, many orders above the
    matched march's rounding budget that they must agree within instead. The same
    stored realisation on both sides, so this is a matched comparison (SEP 25 4.5).
    """
    whole, interrupted, again = (
        scheduled(build_simulation, noise="input", seed=17, backend="numpy_ref")
        for _ in range(3)
    )
    bound = budget(whole)
    whole.run(tmp_path / "whole.h5")
    run_until(interrupted, tmp_path / "part.h5", after=2)
    again.run(tmp_path / "part.h5", resume=True)
    for key in ("field", "noise_phase"):
        assert_matched(
            read_file(tmp_path / "part.h5", key),
            read_file(tmp_path / "whole.h5", key),
            bound,
            what=key,
        )


def test_a_default_noise_run_without_a_seed_resumes_from_the_seed_it_recorded(
    build_simulation, tmp_path
):
    """Re-running the same unseeded call continues its file (the maintainers' ruling).

    Each unseeded construction draws its own entropy, so the call that resumes holds
    another realisation until it takes the recorded seed; it then continues the run
    the file holds, which the uninterrupted run seeded with that entropy matches to
    the matched march's rounding budget (SEP 25 4.5).
    """
    path = tmp_path / "part.h5"
    run_until(
        scheduled(build_simulation, noise="default", backend="numpy_ref"),
        path,
        after=2,
    )
    stored = resume(path).metadata.noise
    assert stored is not None and stored.seed is not None
    again = scheduled(build_simulation, noise="default", backend="numpy_ref")
    assert again.noise is not None and again.noise.seed != stored.seed
    again.run(path, resume=True)
    assert again.noise.seed == stored.seed
    assert resume(path).n_saved == N_SAVE
    whole = scheduled(
        build_simulation, noise="default", seed=stored.seed, backend="numpy_ref"
    )
    whole.run(tmp_path / "whole.h5")
    for key in ("field", "noise_phase"):
        assert_matched(
            read_file(path, key),
            read_file(tmp_path / "whole.h5", key),
            budget(whole),
            what=key,
        )


def test_a_recorded_seed_is_taken_only_for_the_model_the_file_drew(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    scheduled(build_simulation, noise="input").run(path)
    # Already drawn from the recorded seed: a second resume of the same run has
    # nothing to redraw and finds the file complete.
    again = scheduled(build_simulation, noise="input")
    again.run(path, resume=True)
    seed = again.noise.seed if again.noise is not None else None
    again.run(path, resume=True)
    assert again.noise is not None and again.noise.seed == seed
    # A file of another noise model, or of none, keeps the call's own draw, and the
    # resume is refused as a different run.
    modified = scheduled(build_simulation, noise="modified")
    drawn = modified.noise
    with pytest.raises(OutputError, match=r"noise\.model"):
        modified.run(path, resume=True)
    assert modified.noise is drawn
    quiet = tmp_path / "quiet.h5"
    scheduled(build_simulation).run(quiet)
    noisy = scheduled(build_simulation, noise="input")
    with pytest.raises(OutputError, match="these fields changed: noise"):
        noisy.run(quiet, resume=True)


def test_a_seeded_realisation_is_not_redrawn_and_the_refusal_names_the_seed(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    scheduled(build_simulation, noise="input", seed=17).run(path)
    other = scheduled(build_simulation, noise="input", seed=18)
    assert not other._reseed(17)
    with pytest.raises(OutputError, match="cannot be resumed") as refusal:
        other.run(path, resume=True)
    # The spectrum's hundreds of changed bins are one entry, counted, not listed.
    message = str(refusal.value)
    assert re.search(r"noise\.seed, noise\.spectrum \(\d+ elements\)", message)
    assert "[0][0]" not in message
    assert message.endswith("pass seed=17 to draw it again")


def test_a_resumed_run_continues_the_file_rather_than_replacing_it(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    run_until(scheduled(build_simulation), path, after=1)
    created = resume(path).metadata.created
    scheduled(build_simulation).run(path, resume=True)
    state = resume(path)
    assert state.n_saved == N_SAVE
    # The file keeps its own record, creation time included: it is the same run.
    assert state.metadata.created == created


def test_a_run_interrupted_before_its_first_plane_resumes_without_a_second_entrance(
    build_simulation, tmp_path, monkeypatch
):
    """The entrance is saved when the handler holds nothing, not when the run starts.

    A run that fails before reaching its first save plane leaves a file holding the
    plane at z = 0 and nothing else. Resuming it starts at plane zero again, so keying
    the entrance off the start plane would store it twice.
    """
    path = tmp_path / "run.h5"
    reference = tmp_path / "whole.h5"
    scheduled(build_simulation).run(reference)

    def fail_before_first_step(self, target):
        raise CrashError("interrupted before the first step")

    # Fail the actual march so its terminal checkpoint and regular observations
    # agree; truncating an advanced file leaves its later checkpoint in place.
    with monkeypatch.context() as patch:
        patch.setattr(StepDriver, "propagate_to", fail_before_first_step)
        with pytest.raises(CrashError, match="before the first step"):
            scheduled(build_simulation).run(path, progress="none")
    stopped = resume(path)
    assert stopped.n_saved == 1 and stopped.plane.z == 0.0
    assert stopped.checkpoint is not None and stopped.checkpoint.landed
    scheduled(build_simulation).run(path, resume=True)
    assert np.array_equal(read_file(path, "z"), read_file(reference, "z"))
    # Restarted from the entrance on the same schedule: the same march, to rounding.
    assert_matched(
        read_file(path, "field"),
        read_file(reference, "field"),
        budget(scheduled(build_simulation)),
        what="field",
    )


def test_resuming_a_finished_run_saves_nothing_more(build_simulation, tmp_path):
    path = tmp_path / "run.h5"
    scheduled(build_simulation).run(path)
    before = read_file(path, "field")
    scheduled(build_simulation).run(path, resume=True)
    # Nothing is recomputed or rewritten, so the stored values are untouched, exactly.
    assert np.array_equal(read_file(path, "field"), before)


# -------------------------------------------------------------------- the refusals


def test_a_file_written_for_another_run_is_refused_before_anything_is_written(
    build_simulation, tmp_path
):
    path = tmp_path / "run.h5"
    scheduled(build_simulation).run(path)
    before = read_file(path, "field")
    # The same fibre and the same planes, a different input: a stronger soliton.
    changed = scheduled(build_simulation, order=4)
    with pytest.raises(OutputError, match="cannot be resumed"):
        changed.run(path, resume=True)
    # Refused by check_cache before the handler opened the file, so nothing moved.
    assert np.array_equal(read_file(path, "field"), before)


def test_only_a_path_can_be_resumed(build_simulation):
    with pytest.raises(DriverError, match="only a path can be resumed"):
        build_simulation(n_save=2).run(MemoryOutput(), resume=True)


def test_resuming_with_no_file_starts_from_the_beginning(build_simulation, tmp_path):
    path = tmp_path / "fresh.h5"
    build_simulation(length=LENGTH, n_save=3).run(path, resume=True)
    assert read_file(path, "z").tolist() == [0.0, LENGTH / 2, LENGTH]


def test_set_initial_refuses_a_position_outside_the_table(build_simulation):
    simulation = build_simulation(length=LENGTH, n_save=2)
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=simulation.grid)
    with pytest.raises(BackendError, match="operator table"):
        backend.set_initial(problem.initial.field0, None, z=2 * LENGTH)
    with pytest.raises(BackendError, match="operator table"):
        backend.set_initial(problem.initial.field0, None, z=-LENGTH)
    backend.close()


def test_set_initial_refuses_a_position_off_a_replayed_schedule(build_simulation):
    simulation = scheduled(build_simulation)
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=simulation.grid)
    # Half a step past an edge: a replay could not continue in step from there.
    with pytest.raises(BackendError, match="one of its edges"):
        backend.set_initial(problem.initial.field0, None, z=LENGTH / 80)
    backend.close()


def test_a_state_loaded_at_a_plane_starts_the_run_there(build_simulation):
    """The position and the schedule index come from `z`, and nothing else moves."""
    simulation = scheduled(build_simulation)
    problem = simulation.translate()
    backend = select("numpy_ref", problem, grid=simulation.grid)
    backend.set_initial(problem.initial.field0, None, z=LENGTH / 2)
    assert backend.observe().z == LENGTH / 2
    summary = backend.propagate_to(LENGTH)
    # Half the fibre at forty steps over the whole of it: twenty steps left.
    assert summary.n_steps == 20
    backend.close()


# ------------------------------------------------------ across the two backends


def test_a_run_continued_on_the_other_backend_is_the_same_run(
    build_simulation, tmp_path
):
    """The resume key holds the problem and the state, not what executed them.

    The backend's name is not in the run's ``configuration`` and therefore not in the
    resume key, so a file written by one backend can be continued by the other. Keying
    on it would protect against nothing measurable: on a replayed schedule the resumed
    run is the same discrete march as the uninterrupted run on either backend, so the
    two agree within the matched march's rounding budget, which is what bounds two
    uninterrupted runs on the two backends as well.

    What executed the run is recorded instead, outside the key, so the file
    still says which backend wrote which half.
    """
    whole = scheduled(build_simulation, backend="numpy_ref")
    started = scheduled(build_simulation, backend="numba_cpu")
    finished = scheduled(build_simulation, backend="numpy_ref")
    whole.run(tmp_path / "whole.h5")
    run_until(started, tmp_path / "mixed.h5", after=2)
    # The refusal this test exists to remove would fire here, before a plane is read.
    finished.run(tmp_path / "mixed.h5", resume=True)
    reference = read_file(tmp_path / "whole.h5", "field")
    mixed = read_file(tmp_path / "mixed.h5", "field")
    assert_matched(mixed, reference, budget(whole), what="field")
    assert np.array_equal(
        read_file(tmp_path / "whole.h5", "z"), read_file(tmp_path / "mixed.h5", "z")
    )
    # The file keeps the record it was opened with, creation time and all, so the
    # backend it names is the one that *started* it. That is append mode's rule and not
    # a special case for this group; what matters here is that the resume key never
    # saw either name.
    with h5py.File(tmp_path / "mixed.h5", "r") as file:
        assert file.attrs["backend"] == "numba_cpu"
    assert (
        whole.metadata().configuration_hash() == started.metadata().configuration_hash()
    )
