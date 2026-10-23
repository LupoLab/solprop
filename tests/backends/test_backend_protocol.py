# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import pytest

from solprop.backends import (
    Backend,
    Observation,
    StepHistory,
    StepStats,
    StepSummary,
)
from solprop.exceptions import BackendError
from solprop.log import NullReporter
from solprop.stats import StatsSpec


def history(n_steps=2, *, accepted=None):
    """A well-formed step history of `n_steps` steps."""
    return StepHistory(
        z=np.linspace(0.1, 0.1 * n_steps, n_steps),
        dz=np.full(n_steps, 0.1),
        error_norm=np.full(n_steps, 0.4),
        accepted=np.ones(n_steps, bool) if accepted is None else accepted,
        wall_time=np.linspace(0.01, 0.02, n_steps),
    )


# ------------------------------------------------------------------------ step history


def test_a_history_counts_its_steps_and_its_rejections():
    record = history(3, accepted=np.array([True, False, True]))
    assert record.n_steps == 3
    assert record.n_rejected == 1
    assert not record.z.flags.writeable


def test_an_empty_history_is_a_valid_record():
    """Two save planes with no step between them is a legitimate observation."""
    empty = StepHistory(
        np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0)
    )
    assert (empty.n_steps, empty.n_rejected) == (0, 0)


@pytest.mark.parametrize("name", ["dz", "error_norm", "accepted", "wall_time"])
def test_a_history_of_ragged_arrays_is_refused(name):
    fields = {
        "z": np.zeros(3),
        "dz": np.ones(3),
        "error_norm": np.zeros(3),
        "accepted": np.ones(3, bool),
        "wall_time": np.zeros(3),
    }
    fields[name] = fields[name][:2]
    with pytest.raises(BackendError, match="one entry per step"):
        StepHistory(**fields)


def test_a_history_with_a_non_finite_entry_is_refused():
    record = history()
    with pytest.raises(BackendError, match="not finite"):
        StepHistory(
            record.z,
            record.dz,
            np.array([np.nan, 0.4]),
            record.accepted,
            record.wall_time,
        )


def test_a_step_of_no_length_is_refused():
    record = history()
    with pytest.raises(BackendError, match="dz must be positive"):
        StepHistory(
            record.z,
            np.array([0.1, 0.0]),
            record.error_norm,
            record.accepted,
            record.wall_time,
        )


# -------------------------------------------------------------------------- observation


def test_an_observation_freezes_its_field_and_agrees_with_its_history():
    record = history(2, accepted=np.array([True, False]))
    plane = Observation(0.2, np.zeros((1, 1, 1, 8), complex), None, record, 2, 1)
    assert plane.z == 0.2
    assert not plane.require_field().flags.writeable
    assert plane.require_field().flags.c_contiguous
    assert plane.noise_phase is None


def test_a_noise_phase_is_carried_in_the_state_layout():
    plane = Observation(
        0.1,
        np.zeros((2, 1, 1, 4), complex),
        np.zeros((2, 1, 1, 4)),
        history(1),
        1,
        0,
    )
    assert plane.noise_phase is not None
    assert plane.noise_phase.shape == plane.require_field().shape
    assert not plane.noise_phase.flags.writeable


def test_a_diverged_field_is_refused_rather_than_observed():
    spoiled = np.zeros((1, 1, 1, 4), complex)
    spoiled[0, 0, 0, 2] = np.nan
    with pytest.raises(BackendError, match="a diverged run must raise"):
        Observation(0.1, spoiled, None, history(1), 1, 0)


def test_a_noise_phase_that_does_not_match_the_field_is_refused():
    with pytest.raises(BackendError, match="noise_phase has shape"):
        Observation(
            0.1,
            np.zeros((1, 1, 1, 4), complex),
            np.zeros((1, 1, 1, 8)),
            history(1),
            1,
            0,
        )


@pytest.mark.parametrize(
    ("n_steps", "n_rejected", "message"),
    [(3, 0, "n_steps is 3"), (2, 1, "n_rejected is 1")],
)
def test_counts_that_disagree_with_the_history_are_refused(
    n_steps, n_rejected, message
):
    with pytest.raises(BackendError, match=message):
        Observation(
            0.2, np.zeros((1, 1, 1, 4), complex), None, history(2), n_steps, n_rejected
        )


# --------------------------------------------------------------------------- step stats


def rows(n_rows=2, *, step=None):
    """A well-formed statistics channel of `n_rows` rows."""
    return StepStats(
        z=np.linspace(0.05, 0.05 * n_rows, n_rows),
        step=np.arange(n_rows) if step is None else step,
        values={"energy": np.ones((n_rows, 1, 1, 1))},
    )


def test_a_channel_counts_its_rows_and_names_its_statistics():
    channel = rows(3)
    assert channel.n_rows == 3
    assert channel.names == ("energy",)
    assert not channel.z.flags.writeable
    assert not channel.values["energy"].flags.writeable
    assert channel.step.dtype == np.int64


def test_an_empty_channel_is_what_a_run_without_a_periodic_subset_carries():
    channel = StepStats.empty()
    assert channel.n_rows == 0
    assert channel.names == ()
    assert (
        Observation(
            0.1, np.zeros((1, 1, 1, 4), complex), None, history(1), 1, 0
        ).step_stats.n_rows
        == 0
    )


def test_a_channel_at_a_position_that_is_not_finite_is_refused():
    """A row is a point on the axis a reader plots against; NaN is not one."""
    with pytest.raises(BackendError, match="z has entries that are not finite"):
        StepStats(np.array([0.1, np.nan]), np.array([0, 1]), {})


def test_a_channel_whose_index_columns_disagree_is_refused():
    with pytest.raises(BackendError, match="step has 3 entries and z has 2"):
        StepStats(np.array([0.1, 0.2]), np.array([0, 1, 2]), {})


@pytest.mark.parametrize(
    ("column", "z", "step"),
    [
        ("z", np.array([0.2, 0.1]), np.array([0, 1])),
        ("step", np.array([0.1, 0.2]), np.array([1, 1])),
    ],
)
def test_a_channel_whose_rows_do_not_advance_is_refused(column, z, step):
    """The rows are taken in the order the run took its steps and are read that way."""
    with pytest.raises(BackendError, match=f"{column} must increase"):
        StepStats(z, step, {})


def test_a_channel_with_a_row_axis_that_is_not_the_row_count_is_refused():
    with pytest.raises(BackendError, match="'energy' rows have leading axis"):
        StepStats(np.array([0.1, 0.2]), np.array([0, 1]), {"energy": np.ones((3, 1))})


def test_a_scalar_value_is_refused_because_it_carries_no_row_axis():
    with pytest.raises(BackendError, match="'energy' rows have leading axis"):
        StepStats(np.array([0.1]), np.array([0]), {"energy": np.asarray(1.0)})


def test_a_row_that_indexes_past_the_history_it_is_flushed_with_is_refused():
    """The two channels describe one interval, so a row must name an attempt in it."""
    with pytest.raises(BackendError, match="indexes step 5 of a history with 2"):
        Observation(
            0.2,
            np.zeros((1, 1, 1, 4), complex),
            None,
            history(2),
            2,
            0,
            rows(1, step=np.array([5])),
        )


# ------------------------------------------------------------------------- step summary


def test_a_summary_records_what_the_call_cost():
    summary = StepSummary(1.0, 40, 2, 0.9, 0.5, 0.31, True)
    assert (summary.z, summary.n_steps, summary.n_rejected) == (1.0, 40, 2)
    assert summary.met_tolerance


@pytest.mark.parametrize(("n_steps", "n_rejected"), [(-1, 0), (2, 3), (2, -1)])
def test_impossible_step_counts_are_refused(n_steps, n_rejected):
    with pytest.raises(BackendError, match="0 <= n_rejected <= n_steps"):
        StepSummary(1.0, n_steps, n_rejected, 0.0, 0.0, 0.0, True)


@pytest.mark.parametrize("name", ["max_norm", "mean_norm", "wall_time"])
def test_a_negative_or_infinite_measure_is_refused(name):
    fields = {"max_norm": 0.5, "mean_norm": 0.4, "wall_time": 0.1}
    fields[name] = -1.0
    with pytest.raises(BackendError, match=f"{name} must be non-negative"):
        StepSummary(1.0, 1, 0, met_tolerance=True, **fields)
    fields[name] = np.inf
    with pytest.raises(BackendError, match=f"{name} must be non-negative"):
        StepSummary(1.0, 1, 0, met_tolerance=True, **fields)


# ------------------------------------------------------------------------- the protocol


class Stub:
    """The smallest object that satisfies the seam."""

    name = "stub"
    max_batch = 1

    def set_initial(self, field, noise):
        self.field = field

    def restore_call_ledgers(self, logical, atomic=None):
        return None

    def set_progress(self, reporter):
        self.reporter = reporter

    def propagate_to(self, z):
        return StepSummary(z, 1, 0, 0.5, 0.5, 0.0, True)

    def observe(self, *, field=True):
        return Observation(0.0, self.field if field else None, None, history(1), 1, 0)

    def stats(self, spec):
        return {name: np.zeros(1) for name in spec.names}

    def metadata(self):
        return {"backend": self.name, "propagator_mode": "tabulate"}

    def memory_ledger(self):
        return {"state": self.field.nbytes}

    def available_device_memory(self):
        return None

    def close(self):
        return None


def test_a_minimal_backend_satisfies_the_protocol():
    backend = Stub()
    assert isinstance(backend, Backend)
    backend.set_initial(np.zeros((1, 1, 1, 4), complex), None)
    assert backend.restore_call_ledgers({}) is None
    backend.set_progress(NullReporter())
    assert backend.propagate_to(0.5).z == 0.5
    assert backend.observe().require_field().shape == (1, 1, 1, 4)
    assert backend.stats(StatsSpec(names=("energy",))) == {"energy": pytest.approx(0.0)}
    assert backend.memory_ledger() == {"state": 64}
    assert backend.available_device_memory() is None
    assert backend.metadata()["backend"] == "stub"
    assert backend.close() is None


def test_an_object_missing_a_method_is_not_a_backend():
    class Partial:
        name = "partial"
        max_batch = 1

        def set_initial(self, field, noise):
            return None

    assert not isinstance(Partial(), Backend)
