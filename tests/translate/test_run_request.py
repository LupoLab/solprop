# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run request a backend prepares beside its problem (SEP 47 section 6.3).

:func:`~solprop.translate.run_request.describe_run` classifies the method and resolves
every policy before preparation. These tests pin the classification of the
benchmark-only comparison family, which only the reference runs, the refusal of a
start plane the problem does not have, and how a run's request is derived from the one
translation returned (SEP 47 section 5.2).
"""

import dataclasses

import numpy as np
import pytest

from sessions.executors import soliton
from solprop.exceptions import TranslationError
from solprop.output.schema import FieldRetention
from solprop.stats import StatsSpec
from solprop.stepper import Multistep
from solprop.stepper.multistep import ComparisonMultistep
from solprop.translate.problem import StepperSettings
from solprop.translate.run_request import HistoryPolicy, MethodKind, describe_run


@pytest.mark.parametrize(
    ("integrator", "kind", "reference_only"),
    [
        (None, MethodKind.RUNGE_KUTTA, False),
        (Multistep(), MethodKind.ADAMS, False),
        (ComparisonMultistep(order=3), MethodKind.ADAMS, True),
        # The PEC variant one order higher is the selected recurrence, but the
        # comparison family is benchmark-only as a whole (SEP 47 section 6.3).
        (
            ComparisonMultistep(corrector="pec", order=3, corrector_order=4),
            MethodKind.ADAMS,
            True,
        ),
    ],
    ids=["runge-kutta", "adams", "comparison", "comparison-pec"],
)
def test_only_the_comparison_family_is_reference_only(integrator, kind, reference_only):
    integration = soliton(integrator=integrator).request.integration
    assert integration.kind is kind
    assert integration.reference_only is reference_only


@pytest.mark.parametrize("start_plane", [-1, 5])
def test_a_start_plane_the_problem_does_not_have_is_refused(start_plane):
    run_request = soliton().request
    assert len(run_request.stepper.save_planes) == 4
    with pytest.raises(TranslationError, match=f"start_plane {start_plane}"):
        run_request.for_run(start_plane=start_plane)
    with pytest.raises(TranslationError, match=f"start_plane {start_plane}"):
        describe_run(run_request.stepper, start_plane=start_plane)


def test_the_plane_after_the_last_is_a_start_plane():
    # A run resumed at its final plane starts beyond every plane, with none to save.
    request = describe_run(soliton().request.stepper, start_plane=4)
    assert request.observation.start_plane == 4


@pytest.mark.parametrize("integrator", [None, Multistep()])
def test_the_stepper_section_is_the_one_translation_resolved(integrator):
    """The request gathers back, field for field, the section it was built from."""
    run_request = soliton(integrator=integrator).request
    again = describe_run(
        run_request.stepper, stats=run_request.observation.stats
    ).stepper
    for entry in dataclasses.fields(StepperSettings):
        one, other = (
            getattr(run_request.stepper, entry.name),
            getattr(again, entry.name),
        )
        if isinstance(one, np.ndarray):
            np.testing.assert_array_equal(one, other)
        else:
            assert one is other or one == other, entry.name
    assert run_request.stepper is run_request.stepper, "built once per request"


def test_a_run_s_request_keeps_the_policy_it_was_prepared_with():
    run_request = soliton(step_planes=3).request
    retention = FieldRetention("none")
    spec = StatsSpec(names=("energy",))
    history = HistoryPolicy.from_choice(choice=True)
    run = run_request.for_run(
        stats=spec, retention=retention, start_plane=2, history=history, snapshots=True
    )
    assert run.integration is run_request.integration
    assert run.error_control is run_request.error_control
    assert run.replay is run_request.replay
    assert run.observation.save_planes is run_request.observation.save_planes
    assert run.observation.step_planes == 3
    assert run.observation.stats is spec
    assert run.observation.retention is retention
    assert run.observation.start_plane == 2
    assert run.history is history
    assert run.snapshots
    # The default keeps the request's own statistics and resets the run's choices.
    plain = run.for_run()
    assert plain.observation.stats is spec
    assert plain.observation.start_plane == 0
    assert not plain.snapshots


def test_another_stepper_section_keeps_the_run_s_choices():
    run_request = soliton().request.for_run(start_plane=1, snapshots=True)
    shorter = dataclasses.replace(run_request.stepper, dz_max=1e-3)
    changed = run_request.with_stepper(shorter)
    assert changed.error_control.dz_max == 1e-3
    assert changed.observation.start_plane == 1
    assert changed.observation.stats is run_request.observation.stats
    assert changed.snapshots
