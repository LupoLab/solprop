# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The one admission list of the general modal route and the coupled flow.

:func:`solprop.translate.admission.check_general_route` runs in the problem's own
validation, so a combination is admitted or refused once, for both backends.
"""

import copy
import dataclasses

import pytest

from coupled_flow_cases import guide_simulation
from driver.test_coherent_capillary import varying
from modal_provider_cases import general_grid, split_guide
from solprop.exceptions import (
    GrazingIncidenceWarning,
    MaterialRangeWarning,
    TranslationError,
)
from solprop.modes import CoherentWall
from solprop.translate.admission import check_general_route

pytestmark = [
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]


@pytest.fixture(scope="module")
def split_problem():
    """The rotated split block: a general guide carrying a coupled flow."""
    grid = general_grid()
    return guide_simulation(split_guide(grid), grid).translate()


@pytest.fixture(scope="module")
def frozen_problem():
    """A tapered wall integrated by the frozen-residual split."""
    return varying(wall=CoherentWall()).translate()


def test_every_flow_method_executed_is_admitted(split_problem, frozen_problem):
    assert check_general_route(split_problem) is None
    assert check_general_route(frozen_problem) is None


@pytest.mark.parametrize("name", ["kerr_coefficients", "raman_terms"])
def test_a_legacy_response_beside_a_general_guide_is_refused(split_problem, name):
    """The route's source replaces the node pipeline that would read it.

    The field is set on a copy past the group's own validation: what is under test
    is the list's refusal, which reads only whether the field is present.
    """
    responses = copy.copy(split_problem.responses)
    object.__setattr__(
        responses, name, object() if name == "kerr_coefficients" else (object(),)
    )
    spoiled = dataclasses.replace(split_problem, responses=responses)
    with pytest.raises(
        TranslationError, match=rf"^solprop executes .*responses\.{name}"
    ):
        check_general_route(spoiled)
