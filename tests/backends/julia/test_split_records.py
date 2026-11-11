# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep split damping in accepted-step services and out of nonlinear stages."""

from dataclasses import replace

import numpy as np
import pytest

from backends.julia.split_cases import split_case
from solprop.backends.julia._descriptors import lower_modal
from solprop.backends.julia._lower import lower_run
from solprop.backends.julia.backend import JuliaBackend
from solprop.exceptions import TranslationError


@pytest.mark.parametrize("kind", ["field", "envelope"])
@pytest.mark.parametrize("method", ["rk", "adams", "low_storage"])
def test_split_alone_attaches_services_without_a_statistic_or_monitor(kind, method):
    case, grid, mode = split_case(kind, method)
    assert not case.request.observation.stats.names
    assert case.request.boundary.monitor is None
    lowered = lower_modal(case.problem)
    assert lowered.temporal_rate is None
    run = lower_run(case.problem, case.request)
    assert run.source_absent
    if run.adams is not None:
        assert run.adams.collar_rate == 0
    owner = JuliaBackend(case.problem, case.request, grid=grid, modes=(mode,))
    try:
        observation = owner._observation
        assert observation is not None and observation.split_rate is not None
        assert not observation.can_rest
        np.testing.assert_array_equal(
            observation.split_rate, case.problem.responses.temporal_absorber
        )
        assert not observation.split_rate.flags.writeable
        assert not np.shares_memory(
            observation.split_rate, case.problem.responses.temporal_absorber
        )
    finally:
        owner.close()


def test_an_all_zero_collar_retains_the_shared_no_absorber_translation():
    case, _, _ = split_case(rate="zero")
    assert case.problem.responses.temporal_absorber is None
    assert case.problem.responses.temporal_absorber_fine is None
    owner = JuliaBackend(case.problem, case.request)
    try:
        assert owner._observation is None
        assert lower_modal(case.problem).temporal_rate is None
    finally:
        owner.close()


def test_split_interpolation_refuses_through_shared_admission():
    case, _, _ = split_case(method="adams")
    request = replace(
        case.request,
        replay=None,
        observation=replace(case.request.observation, dense=True),
    )
    with pytest.raises(TranslationError, match="per-step temporal collar"):
        JuliaBackend(case.problem, request)
