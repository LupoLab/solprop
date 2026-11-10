# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run request a backend prepares beside its problem (SEP 47 section 6.3).

:func:`~solprop.translate.run_request.describe_run` classifies the method and resolves
every policy before preparation. These tests pin the classification of the
benchmark-only comparison family, which only the reference runs, and the refusal of a
start plane the problem does not have.
"""

import pytest

from sessions.executors import soliton
from solprop.exceptions import TranslationError
from solprop.stepper import Multistep
from solprop.stepper.multistep import ComparisonMultistep
from solprop.translate.run_request import MethodKind, describe_run


@pytest.mark.parametrize(
    ("integrator", "kind", "reference_only"),
    [
        (None, MethodKind.RUNGE_KUTTA, False),
        (Multistep(), MethodKind.ADAMS, False),
        (ComparisonMultistep(order=3), MethodKind.ADAMS, True),
    ],
    ids=["runge-kutta", "adams", "comparison"],
)
def test_only_the_comparison_family_is_reference_only(integrator, kind, reference_only):
    integration = describe_run(soliton(integrator=integrator)).integration
    assert integration.kind is kind
    assert integration.reference_only is reference_only


@pytest.mark.parametrize("start_plane", [-1, 5])
def test_a_start_plane_the_problem_does_not_have_is_refused(start_plane):
    problem = soliton()
    assert len(problem.stepper.save_planes) == 4
    with pytest.raises(TranslationError, match=f"start_plane {start_plane}"):
        describe_run(problem, start_plane=start_plane)


def test_the_plane_after_the_last_is_a_start_plane():
    # A run resumed at its final plane starts beyond every plane, with none to save.
    request = describe_run(soliton(), start_plane=4)
    assert request.observation.start_plane == 4
