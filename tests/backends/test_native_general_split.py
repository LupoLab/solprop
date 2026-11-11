# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A general guide under the split collar, whole spectrum, native against the reference.

SEP 43 deleted the only field comparison under ``absorber_scheme="split_step"`` on a
general guide on a compiled backend, ``numba_cpu``'s entry-point test. It is here on
native, beside what #230 asks: a launch holding content outside the active band, on the
constant and the moving guide, on either march, compared on every bin. Each backend must
drop that content at the load, so the state's out-of-band bins are held to exactly zero,
a structural contract, and the rest to the route's rounding bound.

The roadmap's and #90's report that the reference carries the content while native
zeroes it is what these settle.
"""

import dataclasses
import importlib.util
import warnings

import numpy as np
import pytest

from driver.test_native_moving_guide import grin_kinked, moving_parity_bound, propagate
from general_parity import (
    LENGTH,
    conditioning,
    launch,
    nonlinear_scale,
    problem_for,
    scaled_case,
    source_bound,
    ten_step_bound,
)
from modal_provider_cases import general_grid, split_guide
from sessions.runs import run as session_run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import _general_maps
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import GeneralModalBasisWarning
from solprop.output.memory import MemoryOutput
from solprop.stepper import FixedSchedule
from solprop.stepper.multistep import Multistep
from solprop.translate.general import general_modal_route

pytestmark = [
    pytest.mark.skipif(
        importlib.util.find_spec("solprop_native") is None,
        reason="solprop_native is not installed",
    ),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.GeneralModalBasisWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.SolpropWarning"),
]

OUT_OF_BAND = 1e-3
"""The launch's out-of-band content, relative to its peak: #230's."""

MARCHES = {"runge_kutta": None, "adams": Multistep(variable_order=False)}
"""Both marches on a fixed schedule; a fixed order keeps Adams one discrete march."""


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


def with_content_out_of_band(problem, active):
    """Return `problem` launching into every bin outside the active band as well."""
    launched = np.array(problem.initial.field0, dtype=complex)
    launched[..., ~active] = OUT_OF_BAND * np.max(np.abs(launched))
    return dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=launched)
    )


@pytest.mark.medium
@pytest.mark.parametrize("march", sorted(MARCHES))
def test_the_split_collar_on_a_constant_guide_is_the_reference_on_every_bin(march):
    """Ten steps, at the route's ten-step bound on the launch's own nonlinear scale."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        grid, guide, route, state = scaled_case("pair")
    problem, run_request = problem_for(
        grid,
        guide,
        route,
        state,
        absorber=absorber(grid),
        absorber_scheme="split_step",
        integrator=MARCHES[march],
    )
    maps = _general_maps(problem)
    assert maps is not None
    bound = ten_step_bound(
        nonlinear_scale(maps, state, grid),
        source_bound(maps, grid.n_to, conditioning(maps, state, grid)),
    )
    active = np.asarray(problem.grids.active_mask)
    assert np.count_nonzero(~active) > 0
    loud = with_content_out_of_band(problem, active)
    expected = run(NumpyRefBackend, loud, run_request, grid)
    got = run(native_class(), loud, run_request, grid)
    assert np.all(expected[..., ~active] == 0.0)
    assert np.all(got[..., ~active] == 0.0)
    assert departure(got, expected) <= bound


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
    loud = dataclasses.replace(case, problem=with_content_out_of_band(problem, active))
    quiet = dataclasses.replace(case, problem=problem)
    expected = propagate(loud, NumpyRefBackend)
    got = propagate(loud, native_class())
    linear = 1e3 * propagate(quiet, NumpyRefBackend, problem.initial.field0 * 1e-3)
    assert np.all(expected[..., ~active] == 0.0)
    assert np.all(got[..., ~active] == 0.0)
    bound = moving_parity_bound(quiet, propagate(quiet, NumpyRefBackend), linear)
    assert float(np.max(np.abs(got - expected))) <= bound


@pytest.mark.medium
def test_a_run_through_the_entry_point_is_the_references_on_native():
    """``Simulation.from_waveguide(..., backend="native")`` end to end, split collar.

    ``numba_cpu``'s test, on its input: the user's door, with a shot-noise draw, the run
    loop's admission, the output and the statistics. The saved field agrees with the
    reference's run within the ten-step bound at the launch's own nonlinear scale over
    the millimetre, which covers the shorter run, and the recorded ``physical_energy``
    within the same relative bound.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        grid, guide, _, _ = scaled_case("pair")
    # The entry point's default collar is stable in-stage under the default Adams
    # history only below 0.7 um steps, so it is split off and integrated exactly.
    short = LENGTH / 50
    runs = {}
    for backend in ("numpy_ref", "native"):
        simulation = Simulation.from_waveguide(
            (),
            short,
            provider=guide,
            grid=grid,
            wavelength=835e-9,
            fwhm=20e-15,
            energy=2e-11,
            shape="gauss",
            noise="input",
            seed=5,
            schedule=FixedSchedule.uniform(0.0, short, short / 10, save_planes=[short]),
            save_planes=[short],
            backend=backend,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber_scheme="split_step"
        )
        output = simulation.run(progress="none")
        assert isinstance(output, MemoryOutput)
        runs[backend] = output
    problem = Simulation.from_waveguide(
        (),
        short,
        provider=guide,
        grid=grid,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=2e-11,
        shape="gauss",
        save_planes=[short],
        backend="numpy_ref",
        noise=None,
    ).translate()
    maps = _general_maps(problem)
    assert maps is not None
    state = problem.initial.field0
    bound = ten_step_bound(
        nonlinear_scale(maps, state, grid),
        source_bound(maps, grid.n_to, conditioning(maps, state, grid)),
    )
    reference, compiled = (runs[one].planes()[-1] for one in ("numpy_ref", "native"))
    assert departure(compiled.field, reference.field) <= bound
    energies = [
        float(np.ravel(plane.stats["physical_energy"])[0])
        for plane in (reference, compiled)
    ]
    assert energies[0] > 0.0
    assert energies[1] == pytest.approx(energies[0], rel=bound, abs=0.0)


@pytest.mark.medium
def test_a_linear_guide_skips_the_source_and_steps_as_the_reference():
    """No responding region: no source pass, and the linear flow is the reference's.

    ``numba_cpu``'s test on its input: the split block in the provider's own
    coordinates, which declares no response at all, at the ten-step bound at ``Phi =
    0``. The retired test also made the source raise, and that half has no witness
    here: native's counters count a stage's dispatch whether it has a source to form or
    not, so a source pass that adds nothing is not distinguished, only one that adds
    something.
    """
    grid = general_grid()
    guide = split_guide(grid, rotated=False)
    route = general_modal_route(
        guide, grid, z_max=LENGTH, omega_reference=float(grid.omega0)
    )
    problem, run_request = problem_for(
        grid, guide, route, launch(grid, 2), absorber=None, integrator=None
    )
    got = run(native_class(), problem, run_request, grid)
    expected = run(NumpyRefBackend, problem, run_request, grid)
    assert departure(got, expected) <= ten_step_bound(0.0, 0.0)
