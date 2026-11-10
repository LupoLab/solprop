# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A plane measure the far-field pass can't read is refused on ``native`` (SEP 43).

``numba_cpu``'s pass read the operator's root as the plane's, and any other measure was
refused. Native has that guard at the payload (``_flux_payload``), and no test held it.
"""

import copy
import dataclasses

import numpy as np
import pytest

pytest.importorskip("solprop_native")


from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

from backends.test_native_far_field import (
    holes,
    problem_of,
    random_field,
    spec_of,
    time_grid,
)
from solprop.exceptions import BackendError, TranslationError
from solprop.grid import FreeGrid
from solprop.translate.routes import flux_of


def separable_case():
    """Return a Cartesian problem that asks for a far-field statistic, and its grid."""
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 8, 3e-4, 8, collar_fraction=0)
    field = random_field(space, 1, grid)
    return (*problem_of(grid, space, field, spec_of(holes("tanh"))), grid)


def test_a_foreign_plane_measure_is_refused():
    """A measure on another operator than the run's own: the payload's guard."""
    problem, run_request, grid = separable_case()
    flux = flux_of(problem.route)
    assert flux is not None
    separable = dataclasses.replace(
        flux.base, k_squared=np.array(flux.base.k_squared) * (1 + 1e-6)
    )
    foreign = dataclasses.replace(
        problem,
        route=dataclasses.replace(
            problem.route, flux=dataclasses.replace(flux, base=separable)
        ),
    )
    with pytest.raises(BackendError, match="this plane measure is another"):
        NativeBackend(foreign, run_request, grid=grid)


def test_a_missing_plane_measure_is_refused():
    """``numba_cpu``'s refusal of ``flux=None``, made here by the shared contract.

    Native's payload would take a missing measure for an empty one, but a spatial
    problem without one doesn't reach it: the problem's own check refuses, on either
    backend.
    """
    problem, run_request, grid = separable_case()
    # The route refuses a missing measure on construction; set one past it.
    route = copy.copy(problem.route)
    object.__setattr__(route, "flux", None)
    foreign = dataclasses.replace(problem, route=route)
    with pytest.raises(TranslationError, match="plane measure"):
        NativeBackend(foreign, run_request, grid=grid)
