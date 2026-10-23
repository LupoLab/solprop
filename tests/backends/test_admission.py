# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The one admission list of the general modal route and the coupled flow.

Every backend runs
:func:`solprop.backends.numpy_ref.admission.refuse_general_route` under its own name,
so a combination is admitted or refused once, and each refusal names the caller.
"""

import copy
import dataclasses

import pytest

from driver.test_coherent_capillary import CF4_WALL, varying
from driver.test_split_block import build
from modal_provider_cases import general_grid, split_guide
from solprop.backends.numpy_ref.admission import (
    FLOW_METHODS,
    refuse_coupled_flow,
    refuse_general_route,
)
from solprop.exceptions import (
    BackendError,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
)

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
    return build(split_guide(grid), grid).translate()


@pytest.fixture(scope="module")
def cf4_problem():
    """A tapered wall integrated by CF4."""
    return varying(wall=CF4_WALL).translate()


def test_the_reference_admits_every_flow_method_it_executes(split_problem, cf4_problem):
    assert set(FLOW_METHODS) == {"constant_schur", "frozen_residual", "cf4"}
    assert refuse_general_route(split_problem) is None
    assert refuse_general_route(cf4_problem, backend="numpy_ref") is None


def test_a_section_method_the_caller_has_not_built_is_refused_by_its_name(
    cf4_problem,
):
    """A backend narrows the flow's methods and is refused in its own name."""
    with pytest.raises(BackendError, match=r"^compiled does not execute a 'cf4'"):
        refuse_coupled_flow(
            cf4_problem,
            backend="compiled",
            flow_methods=("constant_schur", "frozen_residual"),
        )
    # The full list refuses it the same way, flow first.
    with pytest.raises(
        BackendError, match="not yet implemented on the compiled backends"
    ):
        refuse_general_route(
            cf4_problem, backend="compiled", flow_methods=("frozen_residual",)
        )


@pytest.mark.parametrize(
    "name", ["kerr_coefficients", "raman_terms", "spectral_responses"]
)
def test_a_legacy_response_beside_a_general_guide_is_refused_by_every_caller(
    split_problem, name
):
    """The route's source replaces the node pipeline that would read it.

    The field is set on a copy past the group's own validation: what is under test
    is the list's refusal, which reads only whether the field is present.
    """
    responses = copy.copy(split_problem.responses)
    object.__setattr__(
        responses, name, object() if name == "kerr_coefficients" else (object(),)
    )
    spoiled = dataclasses.replace(split_problem, responses=responses)
    for backend in ("numpy_ref", "native"):
        with pytest.raises(
            BackendError, match=rf"^{backend} executes .*responses\.{name}"
        ):
            refuse_general_route(spoiled, backend=backend)
