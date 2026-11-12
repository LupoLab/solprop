# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 14 phase 2: the Adams march on the compiled backends, against the reference.

The march is shared (``solprop.backends.numpy_ref.multistep``); only the one source
evaluation and the propagator differ: Kokkos evaluates through its general engine's
stage pipeline. Parity is the reference's trajectory to round-off: the same steps,
orders and calls, and fields that agree to the accumulated rounding of the evaluation
orders.
"""

import dataclasses
import math

import numpy as np
import pytest

from backends.matched import fixed_order
from bounds import EPS, assert_within, fft_bound
from sessions.runs import adams_ledger, native_source
from sessions.runs import run as session_run
from solprop.backends import available
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.passes import apply_hard_mask
from solprop.backends.session import Start
from solprop.boundaries import absorber
from solprop.driver.simulation import Simulation, SimulationSettings
from solprop.exceptions import DriverError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.noise import modified_noise
from solprop.stepper import PIController
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.linop import delta_phi

COMPILED: list[type] = []
if "native" in available():
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    COMPILED.append(NativeBackend)

# The soliton fixture's period (tests/backends/conftest.py).
SOLITON_PERIOD = math.pi / 2 * (100e-15) ** 2 / 11.83e-27


def run(backend_class, problem, run_request, grid, start=None):
    """Run the session to the last plane; return its field, ledger and step sizes.

    The step sizes are every attempt's, rejected ones included.
    """
    backend = backend_class(problem, run_request, grid=grid)
    record: dict[str, object] = {}
    try:
        result = session_run(backend, problem, start=start, record=record)
        ledger = adams_ledger(backend, record)
    finally:
        backend.close()
    return result.fields[-1], ledger, np.asarray(result.joined[0].dz)


@pytest.mark.parametrize(
    "method",
    [
        Multistep(formulation="interaction"),
        Multistep(order=3, variable_order=False, formulation="interaction"),
        # The comparison variants' general path is the reference's alone (SEP 47).
        "interpolated",
    ],
    ids=["selected", "pec3/4", "dense"],
)
@pytest.mark.parametrize("backend_class", COMPILED, ids=lambda one: one.name)
def test_a_compiled_backend_follows_the_reference_trajectory(
    narrow_grid, soliton_problem, method, backend_class
):
    interpolate = method == "interpolated"
    problem, run_request = soliton_problem(
        narrow_grid,
        2,
        save_planes=[SOLITON_PERIOD / 8, SOLITON_PERIOD / 4],
        controller=PIController(rtol=1e-6, atol=1e-18),
        integrator=Multistep(formulation="interaction") if interpolate else method,
        interpolate_planes=interpolate,
    )
    reference, reference_ledger, reference_steps = run(
        NumpyRefBackend, problem, run_request, narrow_grid
    )
    field, ledger, steps = run(backend_class, problem, run_request, narrow_grid)
    # The same decisions: every step, order and call.
    assert ledger.orders == reference_ledger.orders
    assert ledger.total == reference_ledger.total
    # The evaluation orders round differently and the controller passes that on to
    # the step sizes: measured at most 1.5e-10 relative over 47-99 steps (Kokkos's
    # PECE4).
    np.testing.assert_allclose(steps, reference_steps, rtol=1e-9)
    # Measured 8e-16 to 1.8e-15 of the peak over the quarter period.
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-13 * scale)


@pytest.mark.parametrize("formulation", ["interaction", "exponential"])
@pytest.mark.parametrize("backend_class", COMPILED, ids=lambda one: one.name)
def test_a_compiled_backend_steps_the_modified_noise_as_the_reference(
    narrow_grid, soliton_problem, formulation, backend_class
):
    # SEP 16 phase 3: the draw enters each sample at its own position, through the
    # same stage pipeline the Runge-Kutta march uses.
    noise = modified_noise(narrow_grid, np.random.default_rng(3))
    problem, run_request = soliton_problem(
        narrow_grid,
        2,
        save_planes=[SOLITON_PERIOD / 8],
        noise=noise,
        controller=PIController(rtol=1e-7, atol=1e-18),
        integrator=Multistep(formulation=formulation),  # pyright: ignore[reportArgumentType]
    )

    start = Start.entrance(problem, noise=noise)
    reference, reference_ledger, _ = run(
        NumpyRefBackend, problem, run_request, narrow_grid, start
    )
    start = Start.entrance(problem, noise=noise)
    field, ledger, _ = run(backend_class, problem, run_request, narrow_grid, start)
    assert ledger.total == reference_ledger.total
    # As for the noiseless trajectory (above): rounding of the evaluation orders.
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-12 * scale)


def test_the_settings_admit_the_compiled_backends():
    control = PIController(rtol=1e-6, atol=1e-12)
    common = {"save_planes": [1.0], "dz_max": 0.1, "dz_init": 1e-3}
    settings = SimulationSettings(
        control,
        backend="native",
        integrator=Multistep(formulation="interaction"),
        **common,
    )
    assert settings.backend == "native"
    with pytest.raises(DriverError, match="numpy_ref and native only"):
        SimulationSettings(
            control,
            backend="cuda",
            integrator=Multistep(formulation="interaction"),
            **common,
        )


def assert_marches_without_responses(
    backend_class, narrow_grid, linear_problem, collar
):
    """Assert a problem with no response marches as on the reference, to round-off."""
    # No nonlinear response: the source is zero, or the in-stage collar's term alone.
    extra = (
        {"absorber": absorber(narrow_grid), "absorber_scheme": "nonlinear"}
        if collar
        else {}
    )
    problem, run_request = linear_problem(
        narrow_grid,
        save_planes=[SOLITON_PERIOD / 8],
        integrator=Multistep(formulation="interaction"),
        **extra,
    )
    reference, _, _ = run(NumpyRefBackend, problem, run_request, narrow_grid)
    field, _, _ = run(backend_class, problem, run_request, narrow_grid)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-13 * scale)


@pytest.mark.parametrize("collar", [False, True], ids=["no collar", "in-stage collar"])
def test_native_marches_a_problem_without_responses(
    narrow_grid, linear_problem, collar
):
    native = pytest.importorskip("solprop_native")
    assert_marches_without_responses(
        native.NativeBackend, narrow_grid, linear_problem, collar
    )


ROUTES = {
    "analytic field": {"thg": False, "plasma": False},
    "envelope plasma": {"envelope": True, "plasma": True},
}
"""The stage pipeline's two other column families, for one evaluation of the source."""


SOURCE_STEPS = 8
"""The fixed steps of :func:`assert_sources_the_reference`'s matched run."""

SOURCE_LENGTH = SOURCE_STEPS * 5e-6
"""Its length, m: steps inside the in-stage collar's stability limit of 7e-6 m."""

SOURCE_AT = 1e-3
"""Where the one evaluation is taken, m, a step from the anchor at the launch."""

SOURCE_SHARE_BOUND = 1e-10
"""The matched runs' source contributions' departure, relative to the reference's.

Measured 5.4e-12 (analytic field) and 7.5e-12 (envelope plasma) on native. The floor is
the fields' own rounding over the source's share of them (1.1e-5): an eps of the field
is 2e-11 of the source, and the linear runs alone part by 2e-17 to 5e-17 of their
field. 1e-10 is about five eps of the field, 13 times the worst measured.
"""


def source_case(route, length, **settings):
    """Return `route`'s helium capillary of `length` and its edged launch.

    The window's own grid, whose collar is the original evaluation's. The launch is the
    pulse plus half of it rolled two fifths of the window along, which reaches into the
    temporal collar.
    """
    limits = (0.01, 800e-9, (200e-9, 2e-6), 100e-15)
    grid = (
        EnvelopeGrid.from_limits(*limits, thg=False)
        if route.get("envelope")
        else FieldGrid.from_limits(*limits)
    )
    simulation = Simulation.from_capillary(
        150e-6,
        length,
        "He",
        1.0,
        wavelength=800e-9,
        fwhm=10e-15,
        energy=5e-5,
        grid=grid,
        n_save=2,
        noise=None,
        integrator=Multistep(formulation="interaction"),
        **settings,
        **route,
    )
    problem = simulation.translate()
    pulse = np.fft.ifft(problem.initial.field0, axis=-1)
    edge = np.roll(pulse, pulse.shape[-1] * 2 // 5, axis=-1)
    return simulation, problem, np.fft.fft(pulse + 0.5 * edge, axis=-1)


def assert_evaluates_the_reference(backend_class, route):
    """Assert one source evaluation and one propagator are the reference's.

    The evaluation: the reference's untransported sample (``_adams_lab_source``)
    against the native state's own, a zero-length trial's first stage at the same
    position (:func:`sessions.runs.native_source`), at the launch as the
    native state loads it, hard-masked to the active band (a change of 4e-16 of it
    here). Untransported, the two are one transform round trip apart: measured 6.3e-16
    and 4.7e-16 normwise against the one-evaluation budget ``fft_bound(n_to)``, about
    1e-12, as the host-callable evaluation SEP 47 deleted measured 8.2e-16 and 5.1e-16.
    The source is compared before any transport back to the anchor: ``U(anchor,
    position)`` reaches 1e26 on the active band's edges (the capillary's loss over the
    millimetre), where the in-stage collar's broad term is small against its own peak,
    so transported, that small content and the whole term's rounding with it become the
    source's peak, and a peak-relative comparison measures the grid more than the
    backend.

    The propagator, ``exp(Phi(position) - Phi(anchor))``: the native stage kernels'
    phase sum over its ``Phi`` terms for the step's first stage row, both read from the
    native state (``adams_stage_scalars``, ``phase_sum_to_host``), against the
    reference's ``_adams_propagator``. An exponent rounded within a few eps of its size
    and an exponential's few eps, so 16 eps (1 + max |delta Phi|) relative; measured
    equal to the bit.
    """
    simulation, problem, lab = source_case(route, 0.01)
    run_request = simulation._run_request()
    launch = apply_hard_mask(lab[None], problem.grids.active_mask)[0]
    context = {
        "grid": simulation.grid,
        "modes": simulation.modes,
        "profiles": simulation.profiles,
    }
    reference = NumpyRefBackend(problem, run_request, **context)
    try:
        reference.set_initial(launch, None)
        expected = np.array(reference._adams_lab_source(launch[None], SOURCE_AT))
        reference_back = np.array(reference._adams_propagator(0.0, SOURCE_AT))
    finally:
        reference.close()
    backend = backend_class(problem, run_request, **context)
    try:
        found = native_source(backend, launch, SOURCE_AT)
        backend._staged_rows()
        state = backend._native_state
        stages = run_request.stepper.tableau.n_stages
        rows = np.asarray(
            state.adams_stage_scalars(0.0, SOURCE_AT)["phi_delta"], dtype=np.float64
        ).reshape(stages + 1, -1)
        exponent = np.array(state.phase_sum_to_host(np.ascontiguousarray(rows[0])))
    finally:
        backend.close()
    assert_within(
        found,
        expected,
        fft_bound(problem.grids.n_to),
        what="untransported source",
    )
    back = np.exp(exponent).reshape(reference_back.shape)
    size = np.abs(delta_phi(problem.linear, SOURCE_AT, 0.0))
    bound = 16 * EPS * (1 + float(np.max(size)))
    assert_within(back, reference_back, bound, what="propagator")


def assert_sources_the_reference(backend_class, route):
    """Assert `backend_class`'s Adams march sources a column family as the reference.

    The Adams source through the stage pipeline's two other column families: the
    analytic signal of a field without THG, and the envelope with its plasma assembly,
    at a launch that reaches into the temporal collar. One evaluation and the
    propagator are compared as before SEP 47 (:func:`assert_evaluates_the_reference`);
    then the march's use of them: both backends replay :data:`SOURCE_STEPS` fixed steps
    at the method's top order through their sessions, the same discrete method and
    history (``backends/matched.py``), so they part by rounding alone. Each backend's
    source contribution is its run's field less its own run of the launch scaled to the
    linear limit (``faint``), where the source's share falls a millionfold; the two
    contributions are held to :data:`SOURCE_SHARE_BOUND` of the reference's, and the
    fields to the compiled backend's matched-window bound (``test_carrier_frame.py``:
    512 eps a step relative to the field). The source moves the field by 1.1e-5 of it
    over the run, seven decades above that bound.
    """
    assert_evaluates_the_reference(backend_class, route)
    simulation, problem, lab = source_case(
        route,
        SOURCE_LENGTH,
        schedule=FixedSchedule.uniform(0.0, SOURCE_LENGTH, 5e-6),
    )
    run_request = fixed_order(simulation._run_request())
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=lab)
    )
    faint = 1e-3
    linear = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=faint * lab)
    )
    found = []
    for evaluating, launched in (
        (NumpyRefBackend, problem),
        (NumpyRefBackend, linear),
        (backend_class, problem),
        (backend_class, linear),
    ):
        backend = evaluating(
            launched,
            run_request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            found.append(session_run(backend, launched).fields[-1])
        finally:
            backend.close()
    reference, unsourced, field, unsourced_here = found
    bound = SOURCE_STEPS * 512 * EPS
    share = reference - unsourced / faint
    assert np.linalg.norm(share) / np.linalg.norm(reference) > 1e6 * bound
    assert_within(field, reference, bound, what="matched Adams run")
    assert_within(
        field - unsourced_here / faint,
        share,
        SOURCE_SHARE_BOUND,
        what="the run's source contribution",
    )


@pytest.mark.parametrize("route", ROUTES)
def test_native_sources_the_analytic_field_and_the_envelope_plasma(route):
    native = pytest.importorskip("solprop_native")
    assert_sources_the_reference(native.NativeBackend, ROUTES[route])
