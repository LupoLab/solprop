# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 16 phase 4: the exponential formulation on varying diagonal operators.

A trial-local constant reference ``B_n = (Phi(z + h) - Phi(z)) / h`` carries the exact
endpoint flow, and the linear compensation ``(D(z_j) - B_n) u_j`` of every sample is
integrated in the reference flow's frame (the transported compensation, SEP 16 section
4.2 and Q8). These tests check the representation's rate, the exact linear-only path,
agreement with the Runge-Kutta march and the interaction formulation on gradient
capillaries, the untransported witness, order on a schedule, dense observation and
the compiled backends' parity.
"""

import dataclasses
import math

import numpy as np
import pytest

from solprop.backends import available
from solprop.backends.multistep import AdamsMarch
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver.simulation import Simulation
from solprop.gasflow import steady_fill
from solprop.profiles import Linear
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.chebyshev import build
from solprop.translate.linop import delta_phi, phi_rate

pytestmark = pytest.mark.medium

EXPONENTIAL = Multistep(formulation="exponential")
LENGTH = 0.02


def gradient(
    *,
    nonlinear: bool = True,
    integrator: Multistep | RungeKutta = EXPONENTIAL,
    rtol: float = 1e-8,
    **overrides,
) -> Simulation:
    """Return a 2 cm helium capillary whose pressure falls from 2 to 0.5 bar.

    The per-step temporal collar: the pulse stays clear of it over 2 cm, and the
    in-stage collar's stability limit would hold an Adams step to tens of micrometres
    (``docs/guide/numerics/adams.md``), where rounding-level differences of the
    estimate move the trajectory.
    """
    settings = {
        "lambda0": 800e-9,
        "fwhm": 10e-15,
        "energy": 20e-6 if nonlinear else 1e-9,
        "trange": 200e-15,
        "wavelength_limits": (200e-9, 2e-6),
        "backend": "numpy_ref",
        "integrator": integrator,
        "noise": None,
        "save_planes": [LENGTH / 2, LENGTH],
        "rtol": rtol,
        "kerr": nonlinear,
        "plasma": False,
        "raman": False,
        "absorber_scheme": "split_step",
    } | overrides
    radius = settings.pop("radius", 100e-6)
    pressure = settings.pop("pressure", (2.0, 0.5))
    return Simulation.from_capillary(radius, LENGTH, "He", pressure, **settings)


def march(simulation, backend_class=NumpyRefBackend, *, compensation=None, plane=-1):
    """Run to the last plane; return a plane's field, the march and the steps."""
    problem = simulation.translate()
    backend = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
    if compensation is not None:
        method = problem.stepper.integrator
        backend._adams = AdamsMarch(backend, method, compensation=compensation)
    try:
        backend.set_initial(problem.initial.field0, None)
        steps, fields = [], []
        for position in problem.stepper.save_planes:
            backend.propagate_to(float(position))
            observed = backend.observe()
            steps += list(observed.step_history.dz)
            fields.append(observed.field)
        return fields[plane], backend._adams, np.array(steps)
    finally:
        backend.close()


def test_the_rate_is_the_derivative_of_the_flow():
    problem = gradient().translate()
    linear = problem.linear
    assert linear.phi_representation == "expansion"
    z, dz = 0.007, 1e-6
    # A centred difference of Phi against the representation's own derivative:
    # O(dz^2 D''') truncation and Phi's rounding over dz, measured below 1e-6.
    centred = (delta_phi(linear, z - dz, z + dz)) / (2 * dz)
    np.testing.assert_allclose(
        phi_rate(linear, z), centred, rtol=0, atol=1e-6 * np.abs(centred).max()
    )


@pytest.mark.parametrize("route", ["cartesian", "gnlse"])
def test_a_constant_rate_is_the_flow_per_metre(route):
    # A separable or constant operator's rate is D itself, wherever it is read.
    from capability_routes import ROUTES  # noqa: PLC0415

    linear = ROUTES[route](None).translate().linear
    expected = "separable" if route == "cartesian" else "constant"
    assert linear.phi_representation == expected
    flow = delta_phi(linear, 0.0, 1e-4) / 1e-4
    for before in (False, True):
        np.testing.assert_allclose(
            phi_rate(linear, 5e-5, before=before), flow, rtol=1e-12, atol=0
        )


def test_a_structurally_absent_source_takes_the_exact_flow():
    # No response and no collar: the source is structurally absent, so the step is
    # the chord's exact homogeneous flow (SEP 16 section 4.2).
    parent = gradient()
    simulation = Simulation(
        parent.grid,
        parent.modes,
        [],
        parent._field0,
        settings=dataclasses.replace(parent.settings, absorber=None),
        profiles=parent.profiles,
        frame=parent.frame,
        noise=None,
    )
    problem = simulation.translate()
    assert len(problem.responses.response_order) == 1
    field, adams, _ = march(simulation)
    assert adams is not None and not adams._engine.compensated
    start = problem.initial.field0 * problem.grids.active_mask
    exact = np.exp(delta_phi(problem.linear, 0.0, LENGTH)) * start
    # Round-off of the flows' products, step by step.
    np.testing.assert_allclose(field, exact, rtol=0, atol=1e-12 * np.abs(exact).max())


@pytest.mark.parametrize(
    "integrator", [Multistep(formulation="interaction"), EXPONENTIAL]
)
def test_both_formulations_agree_with_runge_kutta(integrator):
    reference, _, _ = march(gradient(integrator=RungeKutta(), rtol=1e-11))
    field, adams, _ = march(gradient(integrator=integrator, rtol=1e-8))
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    # A local tolerance of 1e-8 over a few hundred steps: measured below 1e-6.
    assert error < 1e-5, error
    assert adams is not None and adams.ledger.startup == 1


def test_the_untransported_witness_agrees_with_the_transported_compensation():
    # The gate found the two compensations equal to two significant digits on the
    # full physical runs; on a short gradient they agree within the tolerance.
    transported, first, _ = march(gradient())
    untransported, second, _ = march(gradient(), compensation="untransported")
    scale = np.linalg.norm(transported)
    assert np.linalg.norm(untransported - transported) / scale < 1e-6
    assert first is not None and second is not None
    # The same steps to within a few: the estimate differs only by the compensation.
    assert abs(first.ledger.total - second.ledger.total) <= 0.05 * first.ledger.total


@pytest.mark.parametrize("order", [2, 4])
def test_order_on_a_schedule(order):
    # No collar: the per-step one is a first-order splitting wherever it holds any
    # field, and the in-stage one bounds the steps by stability.
    def collarless(**settings):
        simulation = gradient(**settings)
        simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
        return simulation

    # Measured at the midpoint plane: over the last millimetres the steady-flow
    # fill steepens towards its exit, the rate's higher derivatives grow forty-fold
    # and uniform steps lose order there whatever interpolates the compensation
    # (an adaptive run sees it, the compensation being in the defects).
    method = Multistep(order=order, variable_order=False, formulation="exponential")
    reference, _, _ = march(collarless(integrator=RungeKutta(), rtol=1e-12), plane=0)

    def solve(steps):
        h = LENGTH / steps
        ramp = h * 2.0 ** -np.arange(1, 13)[::-1]
        rest = LENGTH / 2 - ramp.sum()
        count = math.ceil(rest / h)
        first = np.concatenate((ramp, np.full(count, rest / count)))
        schedule = FixedSchedule(np.concatenate((first, np.full(steps // 2, h))))
        field, _, _ = march(
            collarless(
                integrator=method, schedule=schedule, dz_max=h, dz_init=h / 4096
            ),
            plane=0,
        )
        return field

    power = np.abs(reference) ** 2
    strong = power > 1e-8 * power.max()
    errors = [
        np.linalg.norm((solve(n) - reference)[..., strong])
        / np.linalg.norm(reference[..., strong])
        for n in (40, 80, 160)
    ]
    observed = [math.log2(errors[i] / errors[i + 1]) for i in range(2)]
    # PEC q/(q+1) is order q + 1 with the transported compensation too (section 4.2).
    assert all(order + 1 - 0.4 <= one <= order + 2 for one in observed), observed


def test_dense_observation_matches_landing():
    # The in-stage collar: dense observation is refused beside the per-step one.
    landed, _, _ = march(gradient(rtol=1e-9, absorber_scheme="nonlinear"))
    observed, adams, _ = march(
        gradient(rtol=1e-9, absorber_scheme="nonlinear", interpolate_planes=True)
    )
    assert adams is not None and adams.ledger.dense_observations >= 1
    error = np.linalg.norm(observed - landed) / np.linalg.norm(landed)
    # Both within the run's accuracy of the truth: measured below 1e-7.
    assert error < 1e-6, error


COMPILED: dict[str, type] = {"numba_cpu": NumbaCpuBackend}
if "native" in available():
    from solprop_native import NativeBackend  # pyright: ignore[reportMissingImports]

    COMPILED["native"] = NativeBackend


@pytest.mark.parametrize("backend_name", sorted(COMPILED))
def test_a_compiled_backend_follows_the_reference(backend_name):
    reference, first, steps = march(gradient())
    field, second, compiled_steps = march(gradient(), COMPILED[backend_name])
    assert first is not None and second is not None
    assert second.ledger.total == first.ledger.total
    # As on the constant routes (tests/backends/test_exponential_routes.py).
    np.testing.assert_allclose(compiled_steps, steps, rtol=1e-3)
    scale = np.max(np.abs(reference))
    np.testing.assert_allclose(field, reference, rtol=0, atol=1e-9 * scale)


def _tabulated(problem):
    """Return the problem with ``Phi`` as a z table of the expansion's own values."""
    linear = problem.linear
    built = build(
        lambda z: np.stack([delta_phi(linear, 0.0, float(one)) for one in z]),
        *linear.z_span,
        tol=1e-11,
    )
    table = dataclasses.replace(
        linear,
        phi_expansion=None,
        phi_representation="table",
        phi_breakpoints=built.breakpoints,
        phi_coefficients=built.coefficients,
        phi_degrees=built.degrees,
        hard_kinks=built.hard_kinks,
        phi_constant=None,
    )
    return dataclasses.replace(problem, linear=table)


def test_a_z_table_steps_as_the_expansion_it_tabulates():
    simulation = gradient()
    problem = simulation.translate()
    tabulated = _tabulated(problem)
    assert tabulated.linear.phi_representation == "table"
    fields = {}
    for name, candidate, backend_class in (
        ("expansion", problem, NumpyRefBackend),
        ("table", tabulated, NumpyRefBackend),
        ("table, numba_cpu", tabulated, NumbaCpuBackend),
    ):
        backend = backend_class(candidate, grid=simulation.grid, modes=simulation.modes)
        try:
            backend.set_initial(candidate.initial.field0, None)
            for position in candidate.stepper.save_planes:
                backend.propagate_to(float(position))
            fields[name] = backend.observe().require_field()
        finally:
            backend.close()
    scale = np.linalg.norm(fields["expansion"])
    # The table resolves Phi to 1e-11, its rate to what that implies; the run's own
    # tolerance is 1e-8: measured below 1e-7.
    assert np.linalg.norm(fields["table"] - fields["expansion"]) / scale < 1e-6
    # numba_cpu evaluates the table through its own host evaluator (a reused buffer,
    # which the march copies): the reference's trajectory to rounding.
    peak = np.max(np.abs(fields["table"]))
    np.testing.assert_allclose(
        fields["table, numba_cpu"], fields["table"], rtol=0, atol=1e-9 * peak
    )


def test_an_interior_hard_kink_restarts_the_history():
    # A gas port in the middle: the density kinks there, so does D, and the history
    # restarts on the far side as in the interaction formulation.
    fill = steady_fill("He", [0.0, LENGTH / 2, LENGTH], [0.5, 2.0, 0.5], radius=100e-6)
    planes = [0.3 * LENGTH, LENGTH]
    reference, _, _ = march(
        gradient(integrator=RungeKutta(), rtol=1e-11, pressure=fill, save_planes=planes)
    )
    field, adams, _ = march(gradient(pressure=fill, save_planes=planes))
    assert adams is not None and adams.ledger.restarts >= 1
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    assert error < 1e-5, error


def test_the_modified_noise_along_a_gradient_with_the_same_draw():
    # The compensation acts on the propagated state, never on the draw (SEP 16
    # section 4.4); the draw's phase is each sample's own position's.
    reference, _, _ = march(
        gradient(integrator=RungeKutta(), rtol=1e-11, noise="modified", seed=3)
    )
    field, _, _ = march(gradient(noise="modified", seed=3))
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    assert error < 1e-5, error


def test_an_adiabatic_taper():
    # A single-mode core taper: a varying effective area and normalisation, with no
    # coupling to omit.
    taper = Linear(100e-6, 80e-6, LENGTH)
    reference, _, _ = march(gradient(integrator=RungeKutta(), rtol=1e-11, radius=taper))
    field, adams, _ = march(gradient(radius=taper))
    assert adams is not None and adams._engine.compensated
    error = np.linalg.norm(field - reference) / np.linalg.norm(reference)
    assert error < 1e-5, error


@pytest.mark.parametrize(
    "integrator", [Multistep(formulation="interaction"), EXPONENTIAL]
)
def test_a_kink_on_a_save_plane_restarts_the_history(integrator):
    # A kink that is also a save plane: the landing is reported as the plane, and the
    # history must still restart there rather than cross the jump in D (which cost
    # three rejections and a failure restart before).
    fill = steady_fill("He", [0.0, LENGTH / 2, LENGTH], [0.5, 2.0, 0.5], radius=100e-6)
    _, adams, _ = march(gradient(integrator=integrator, pressure=fill))
    assert adams is not None
    assert adams.ledger.restarts >= 1
    assert adams.ledger.failure_restarts == 0
