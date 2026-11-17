# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A general guide under the split collar, whole spectrum, native beside the reference.

SEP 43 deleted the only field comparison under ``absorber_scheme="split_step"`` on a
general guide on a compiled backend, ``numba_cpu``'s entry-point test. It is here on
native, beside what #230 asks: a launch holding content outside the active band, on the
constant and the moving guide, on either march. Each backend must drop that content at
the load, so the state's out-of-band bins are held to exactly zero, a structural
contract. The constant guide's field on every other bin, a linear guide's and the entry
point's saved plane are the catalogue's rows (``backends/general_rows.py``); the moving
guide's field is compared here, at an absolute bound read from more reference runs.

The roadmap's and #90's report that the reference carries the content while native
zeroes it is what these settle.
"""

import dataclasses
import importlib.util

import numpy as np
import pytest

from backends.general_rows import MARCHES, split_collar, with_content_out_of_band
from coupled_flow_cases import propagate
from general_parity import LENGTH
from moving_guide_cases import grin_kinked, moving_parity_bound
from sessions.runs import run as session_run
from solprop.backends.numpy_ref import NumpyRefBackend

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("solprop_native") is None,
        reason="solprop_native is not installed",
    ),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.GeneralModalBasisWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.SolpropWarning"),
]


def native_class():
    """Return the optional wheel's backend class."""
    # Optional wheel: absent from the base environment.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    return NativeBackend


def run(backend_class, problem, run_request, grid, length=LENGTH):
    """Return the final state of the problem's fixed steps on one backend.

    The backend's own session runs them (SEP 47 phase 5) to the last plane, `length`.
    """
    backend = backend_class(problem, run_request, grid=grid)
    try:
        landed = session_run(backend, problem, run_request)
    finally:
        backend.close()
    assert landed.observations[-1].z == pytest.approx(length, rel=1e-12)
    return landed.fields[-1]


def departure(got, expected):
    """Return the relative L2 departure of two states."""
    return float(np.linalg.norm(got - expected) / np.linalg.norm(expected))


@pytest.mark.medium
@pytest.mark.parametrize("march", sorted(MARCHES))
def test_the_split_collar_on_a_constant_guide_zeroes_the_content_out_of_band(march):
    """Ten steps on each backend: every bin outside the active band exactly zero.

    The field on the other bins is the catalogue's row ``general pair split collar``.
    """
    built = split_collar(march)
    active = np.asarray(built.problem.grids.active_mask)
    for backend_class in (NumpyRefBackend, native_class()):
        field = run(backend_class, built.problem, built.run_request, built.grid)
        assert np.all(field[..., ~active] == 0.0)


@pytest.mark.slow
@pytest.mark.parametrize("march", sorted(MARCHES))
def test_the_split_collar_on_a_moving_guide_is_the_reference_on_every_bin(march):
    """Ten steps of the kinked taper, at the moving guide's own parity bound.

    The case keeps no collar of its own, so the temporal one goes on as the entry
    point's: a moving guide's content out of band has not been injected before.
    """
    case = grin_kinked()
    problem = dataclasses.replace(
        case.problem,
        responses=dataclasses.replace(
            case.problem.responses, absorber_scheme="split_step"
        ),
    )
    case = dataclasses.replace(
        case,
        run_request=case.run_request.with_stepper(
            dataclasses.replace(case.run_request.stepper, integrator=MARCHES[march])
        ),
    )
    active = np.asarray(problem.grids.active_mask)
    assert np.count_nonzero(~active) > 0
    loud = dataclasses.replace(case, problem=with_content_out_of_band(problem))
    quiet = dataclasses.replace(case, problem=problem)
    expected = propagate(loud, NumpyRefBackend)
    got = propagate(loud, native_class())
    linear = 1e3 * propagate(quiet, NumpyRefBackend, problem.initial.field0 * 1e-3)
    assert np.all(expected[..., ~active] == 0.0)
    assert np.all(got[..., ~active] == 0.0)
    bound = moving_parity_bound(quiet, propagate(quiet, NumpyRefBackend), linear)
    assert float(np.max(np.abs(got - expected))) <= bound
