# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Polarisation-route equivalence and full-azimuthal propagation."""

import math
from dataclasses import dataclass

import numpy as np
import pytest

from bounds import gamma, ladder_bound, record_backend
from convergence_records import record
from convergence_studies import BACKEND_EQUIVALENCE
from rate_oracles import adk_oracle
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.constants import CHARGE_MASS_RATIO
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import delta_phi
from solprop.translate.problem import assemble

EPS = np.finfo(float).eps
LENGTH = 0.002
N_RADIAL = 24


@pytest.fixture(scope="module", params=["field_real", "field_analytic", "envelope"])
def polarised_case(request):
    family = request.param
    grid_class = EnvelopeGrid if family == "envelope" else FieldGrid
    grid = grid_class.from_limits(LENGTH, 800e-9, (300e-9, 2e-6), 100e-15)
    # Vacuum index and fixed cladding make this an analytic propagation test without
    # extrapolating any gas's index fit. The specified chi3 is absolute (density=1).
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    response = KerrResponse(chi3=1e-26, thg=family == "field_real")
    pulse = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-6).build(grid)
    return family, grid, mode, response, pulse


def route_problem(
    case, route, *, angle=math.pi / 5, phase=math.pi / 3, steps=8, density=1.0
):
    _, grid, mode, response, pulse = case
    polarised = np.concatenate(
        [
            math.cos(angle) * pulse,
            math.sin(angle) * np.exp(1j * phase) * pulse,
        ],
        axis=1,
    )
    if route == "rows":
        modes = [mode, mode.rotated(math.pi / 2)]
        initial = polarised.transpose(1, 0, 2).copy()
    else:
        modes = [mode]
        initial = polarised
    rule = transverse_quadrature(modes, n_r=N_RADIAL)
    schedule = (
        None if steps is None else FixedSchedule.uniform(0.0, LENGTH, LENGTH / steps)
    )
    problem = assemble(
        grid,
        modes,
        [response],
        initial,
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / 8,
        quadrature=rule,
        density=density,
        schedule=schedule,
        stats_spec=StatsSpec(names=("energy", "peak_power_total", "peak_intensity")),
    )
    return problem, modes


def run_route(backend_class, case, route, **options):
    problem, modes = route_problem(case, route, **options)
    backend = backend_class(problem, grid=case[1], modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(LENGTH)
        field = backend.observe().require_field().copy().reshape(2, -1)
        statistics = backend.stats(problem.stats_spec)
        return field, statistics, summary
    finally:
        backend.close()


def relative_gap(values, reference):
    scale = float(np.max(np.abs(reference)))
    assert scale > 0
    return float(np.max(np.abs(values - reference)) / scale)


def route_bound(backend_name, grid, evaluations):
    # Each evaluation has two FFTs, synthesis and projection. The transforms have
    # 16 epsilon per FFT depth; the two matrix reductions add their gamma_n bounds.
    per_evaluation = 16 * EPS * math.ceil(math.log2(grid.n_to)) + 2 * gamma(N_RADIAL)
    return ladder_bound(
        backend_name, evaluations * per_evaluation, grid.n_to, evaluations
    )


def record_rung(request, backend_name, family, label, measured, bound):
    record(
        BACKEND_EQUIVALENCE,
        request=request,
        backend=record_backend(backend_name),
        rows=[
            {
                "series": f"polarisation ({record_backend(backend_name)}, {family})",
                "axis": label,
                "deviation": measured,
                "bound": float(bound),
                "order": 10,
            }
        ],
    )


def first_derivative(backend_class, case, route):
    problem, modes = route_problem(case, route)
    return problem_derivative(backend_class, problem, case[1], modes)


def problem_derivative(backend_class, problem, grid, modes):
    backend = backend_class(problem, grid=grid, modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        if isinstance(backend, NumpyRefBackend):
            backend._hoist(0.0)
            bundle = backend._stage_scalars(0.0, 0.0)
            derivative = backend._nonlinear(
                problem.initial.field0, bundle, np.ones(problem.shape, complex)
            )
        else:
            # The first stage of a trial is exactly N(initial, 0). Observation reads
            # it without finite-differencing a propagated field or changing the stepper.
            backend._trial_step(0.0, LENGTH / 8)
            if backend.name == "native":
                derivative = backend._native_state.stage_to_host(0)
            else:
                derivative = backend._stages[0].copy()
        return derivative.reshape(2, -1)
    finally:
        backend.close()


def plasma_work_bound(grid, tau, amplitude, potential):
    # Write J_loss = Ip W0/Eref^2 E exp(-F), with E a circular Gaussian.
    # A prefix trapezoid's absolute error is <= h^2/12 integral |J_loss''|.
    # Gaussian zeroth, first and second moments bound that integral by Ip W0 E0
    # / Eref^2 times curvature below. Work by parts adds integral |E'| <=
    # E0 (omega tau sqrt(pi) + 2). The maximum W/f is its on-axis value.
    rate = 3e12 * (amplitude / 1e10) ** 2
    fraction = -math.expm1(-rate * tau * math.sqrt(math.pi / 2))
    angular = grid.omega0 + rate
    curvature = (
        angular**2 * tau * math.sqrt(math.pi)
        + 4 * angular
        + 4 * math.sqrt(math.pi) / tau
        + 2 * rate * math.sqrt(math.pi)
    )
    trapezoid = (
        grid.dt_fine**2
        / 12
        * rate
        / fraction
        * (grid.omega0 * tau * math.sqrt(math.pi) + 2)
        * curvature
    )
    # He-Ar subtraction also cancels the Drude term. Bound its conditioning by
    # integrating |E| twice over the entire window; no observed scale enters.
    window = grid.n_to * grid.dt_fine
    conditioning = 1 + CHARGE_MASS_RATIO * amplitude**2 * window**2 / (
        potential * fraction
    )
    rounding = 64 * gamma(grid.n_to) * conditioning
    # The field is below exp(-64) at the boundaries; the nearest excluded spectral
    # bin is over eight Gaussian widths from the carrier. Their work tails are
    # bounded by Cauchy-Schwarz with this conservative shared amplitude envelope.
    tail = 16 * math.exp(-32) * conditioning
    return trapezoid + rounding + tail


def plasma_energy_at(backend_class, backend_name, requested_dt):
    tau, amplitude, density = 6e-15, 1e10, 1e22
    grid = FieldGrid.from_limits(
        LENGTH, 800e-9, (200e-9, 4e-6), 100e-15, dt=requested_dt
    )
    # The work is the difference of two energies per event, which the rate declares
    # (SEP 3, D8): the former registry's helium and argon potentials, 0.9036 and
    # 0.5792 hartree (dc4ca0d, ``rate_oracles.HISTORICAL_ATOMS``), on which the record's
    # bounds were derived; optmat's ionisation energies would move every bound by 1e-5.
    energies = {gas: adk_oracle(gas).ionisation_potential for gas in ("He", "Ar")}
    potential = energies["He"] - energies["Ar"]
    bound = plasma_work_bound(grid, tau, amplitude, potential)
    bound = ladder_bound(backend_name, bound, grid.n_to, 1)
    assert 0 < bound < 0.1
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    rule = transverse_quadrature([mode], n_r=N_RADIAL)
    profile = mode.field(rule.nodes(mode.dimlimits()))[:, 0]
    fractions = -np.expm1(-3e12 * tau * math.sqrt(math.pi / 2) * profile**2)
    expected = (
        density * potential * float(np.dot(rule.weights(mode.dimlimits()), fractions))
    )
    # Two real Cartesian components each have peak E0. Their combined analytic
    # intensity is twice a single component's, with the mode's sqrt(W) scaling.
    envelope = (
        amplitude * math.sqrt(mode.normalisation()) * np.exp(-((grid.t / tau) ** 2))
    )
    temporal = envelope * np.stack(
        [np.cos(grid.omega0 * grid.t), np.sin(grid.omega0 * grid.t)]
    )
    initial = np.fft.rfft(temporal, axis=-1)[None]
    derivatives = []
    for gas in ("He", "Ar"):
        table = build_rate_table(QuadraticRate(gas, energies[gas]), (1e3, 1e11))
        problem = assemble(
            grid,
            [mode],
            # The trapezoid's second-order ladder; the band rule's error here is a
            # few 1e-7 at every rung, inside the bound but not falling by four.
            [PlasmaResponse(gas, table, integration="trapezoid")],
            initial,
            controller=PIController(rtol=1e-9, atol=1e-15),
            save_planes=[LENGTH],
            dz_max=LENGTH,
            dz_init=LENGTH / 8,
            quadrature=rule,
            density=density,
            prefactor_index=1.0,
        )
        derivatives.append(problem_derivative(backend_class, problem, grid, [mode]))
    difference = derivatives[0] - derivatives[1]
    # U = dt/nt sum_{positive bins} 4 |Ew|^2, so dU/dz has coefficient 8.
    # End bins carry weight one instead of four; their derivative is identically 0.
    assert np.all(difference[:, (0, -1)] == 0)
    work = (
        -8
        * grid.dt
        / grid.n_t
        * float(np.sum(np.real(np.conj(initial[0, :, 1:-1]) * difference[:, 1:-1])))
    )
    measured = abs(work / expected - 1)
    assert measured < bound
    return grid.n_to, measured, bound


def test_circular_plasma_pays_the_ionisation_energy(
    backend_class, backend_name, request
):
    rungs = [
        plasma_energy_at(backend_class, backend_name, spacing)
        for spacing in (0.04e-15, 0.02e-15, 0.01e-15)
    ]
    # SEP 27 removes the third trapezoid integral. The work error on these fine
    # grids reaches the fixed spectral projection floor, so its ratio no longer
    # measures the integrator's order. Retain the analytic work bound at every
    # resolution; the running-integral tests separately establish convergence.
    for samples, measured, bound in rungs:
        record_rung(
            request, backend_name, "circular plasma energy", samples, measured, bound
        )


@dataclass(frozen=True)
class QuadraticRate:
    """A log-linear table with an analytic constant-magnitude ionisation solution.

    Anonymous by default; given a species and an energy per event, it declares them
    as a rate does (SEP 3, D8), and a response on it, or on its table, pays that energy.
    """

    species_identity: str | None = None
    ionisation_potential: float | None = None

    @property
    def threshold(self) -> float:
        """Return the field, V/m, below which the rate is zero."""
        return 1e5

    @property
    def maximum(self) -> float:
        """Return the ceiling on the rate, s^-1."""
        return 1e17

    @property
    def citation(self) -> str:
        """Return the rate's description."""
        return "Analytic test rate W = 3e12 (E / 1e10)^2 s^-1"

    @property
    def plasma_energy(self) -> float | None:
        """Return the declared energy per event, J, or None for the anonymous rate."""
        return self.ionisation_potential

    @property
    def parameters(self) -> tuple[tuple[str, float | str], ...]:
        """Return every setting that changes the table, the species among them."""
        named = (
            ()
            if self.species_identity is None
            else (("species", self.species_identity),)
        )
        return (
            ("model", "quadratic-oracle"),
            ("reference", 1e10),
            ("scale", 3e12),
            *named,
        )

    def __call__(self, field_amplitude):
        field = np.abs(np.asarray(field_amplitude, dtype=float))
        return np.where(
            field < self.threshold,
            0.0,
            np.minimum(self.maximum, 3e12 * (field / 1e10) ** 2),
        )


@pytest.fixture(scope="module")
def circular_plasma_case():
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 2e-6), 100e-15)
    mode = MarcatiliMode(125e-6, cladding=1.45, loss=False)
    rate = build_rate_table(QuadraticRate(), (1e7, 1e11))
    template = np.zeros((1, 1, grid.n_omega), dtype=complex)
    carrier = int(np.argmin(np.abs(grid.omega - grid.omega0)))
    # Each real component has amplitude E0 after the two routes divide by sqrt(2).
    # The same physical E0 is specified using a true-power input; the analytic rate
    # and its round-off bound are unaffected.
    weight = float(np.real(mode.neff(np.array([grid.omega[carrier]])))[0])
    template[0, 0, carrier] = (
        grid.n_t * 1e10 * math.sqrt(mode.normalisation() / 2) * math.sqrt(weight)
    )
    return "field_real", grid, mode, PlasmaResponse("Ar", rate), template


def test_circular_plasma_ionises_by_magnitude_through_the_backend(
    backend_class, backend_name, circular_plasma_case, request
):
    case = circular_plasma_case
    _, grid, mode, _, _ = case
    density = 1e22
    # The quadratic log-rate is linear in log E, so there is no spline truncation.
    # A fixed-order time sum, exp and the two transforms set the rounding budget.
    bound = ladder_bound(
        backend_name,
        8 * gamma(grid.n_to) + 64 * EPS * math.ceil(math.log2(grid.n_to)),
        grid.n_to,
        1,
    )
    rule = transverse_quadrature([mode], n_r=N_RADIAL)
    point = rule.nodes(mode.dimlimits())[:1]
    peak_amplitude = 1e10 * float(mode.field(point)[0, 0])
    expected_rate = float(QuadraticRate()(peak_amplitude))
    duration = (grid.n_to - 1) * grid.dt_fine
    expected_density = density * (-math.expm1(-expected_rate * duration))
    specification = StatsSpec(names=("electron_density", "peak_ionisation_rate"))
    results = []
    statistics_error = 0.0
    for route in ("rows", "product"):
        problem, modes = route_problem(
            case, route, angle=math.pi / 4, phase=math.pi / 2, density=density
        )
        backend = backend_class(problem, grid=grid, modes=modes)
        try:
            backend.set_initial(problem.initial.field0, None)
            actual = backend.stats(specification)
            assert relative_gap(actual["electron_density"], expected_density) < bound
            assert relative_gap(actual["peak_ionisation_rate"], expected_rate) < bound
            statistics_error = max(
                statistics_error,
                relative_gap(actual["electron_density"], expected_density),
                relative_gap(actual["peak_ionisation_rate"], expected_rate),
            )
            backend.propagate_to(LENGTH)
            results.append(backend.observe().require_field().copy().reshape(2, -1))
        finally:
            backend.close()
    propagation_bound = route_bound(backend_name, grid, 7 * 8)
    assert relative_gap(results[0], results[1]) < propagation_bound
    record_rung(
        request,
        backend_name,
        "circular plasma",
        "rate and fraction",
        statistics_error,
        bound,
    )
    record_rung(
        request,
        backend_name,
        "circular plasma",
        "8 steps",
        relative_gap(results[0], results[1]),
        propagation_bound,
    )


def test_polarisation_one_nonlinear_evaluation(
    backend_class, backend_name, polarised_case, request
):
    bound = route_bound(backend_name, polarised_case[1], 1)
    product = first_derivative(backend_class, polarised_case, "product")
    rows = first_derivative(backend_class, polarised_case, "rows")
    oracle = first_derivative(NumpyRefBackend, polarised_case, "product")
    measured = max(relative_gap(rows, product), relative_gap(product, oracle))
    assert measured < bound
    record_rung(
        request,
        backend_name,
        polarised_case[0],
        "one nonlinear evaluation",
        measured,
        bound,
    )


@pytest.mark.parametrize("steps", [1, 8, None], ids=["one", "eight", "adaptive"])
def test_polarisation_routes_propagate_the_same_field(
    backend_class, backend_name, polarised_case, steps, request
):
    # 256 evaluations bound the reduced adaptive case; assert the count as well as
    # the field, so the predeclared rounding budget cannot silently be exceeded.
    evaluations = 256 if steps is None else 7 * steps
    bound = route_bound(backend_name, polarised_case[1], evaluations)
    product, stats_product, summary = run_route(
        backend_class, polarised_case, "product", steps=steps
    )
    rows, stats_rows, other = run_route(
        backend_class, polarised_case, "rows", steps=steps
    )
    oracle, _, _ = run_route(NumpyRefBackend, polarised_case, "product", steps=steps)
    assert 7 * (summary.n_steps + summary.n_rejected) <= evaluations
    assert (summary.n_steps, summary.n_rejected) == (other.n_steps, other.n_rejected)
    measured = max(relative_gap(rows, product), relative_gap(product, oracle))
    assert measured < bound
    # Each quadratic statistic adds at most two reductions over its time samples.
    statistics_bound = 2 * bound + 2 * gamma(polarised_case[1].n_t)
    worst_statistic = max(
        relative_gap(stats_rows[name].ravel(), stats_product[name].ravel())
        for name in stats_product
    )
    assert worst_statistic < statistics_bound
    label = "adaptive run" if steps is None else f"{steps} steps"
    record_rung(request, backend_name, polarised_case[0], label, measured, bound)
    record_rung(
        request,
        backend_name,
        polarised_case[0],
        f"statistics after {label}",
        worst_statistic,
        statistics_bound,
    )


@pytest.mark.parametrize(
    "kinds, orders, forbidden",
    [
        (["HE", "HE", "HE"], [1, 1, 2], 2),
        (["TM", "HE"], [0, 1], 1),
    ],
)
def test_full_azimuthal_propagation_obeys_selection_rules(
    backend_class, backend_name, kinds, orders, forbidden, request
):
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 2e-6), 100e-15)
    indices = [1, 2, 1] if len(kinds) == 3 else [1, 1]
    modes = [
        MarcatiliMode(125e-6, kind=kind, n=n, m=m, cladding=1.45, loss=False)
        for kind, n, m in zip(kinds, orders, indices, strict=True)
    ]
    field = np.zeros((len(modes), 1, grid.n_omega), complex)
    field[0] = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-6).build(grid)[0]
    # Cubic products reach 4h, so five angular nodes integrate h=1 exactly. The
    # rounding budget counts projection over all 24*5 nodes through eight RK steps.
    bound = 8 * 7 * (2 * gamma(24 * 5) + 16 * EPS * math.ceil(math.log2(grid.n_to)))
    rule = transverse_quadrature(modes, n_r=24, n_theta=5)
    problem = assemble(
        grid,
        modes,
        [KerrResponse(chi3=1e-26, thg=True)],
        field,
        controller=PIController(rtol=1e-8, atol=1e-15),
        density=1.0,
        save_planes=[LENGTH],
        dz_max=LENGTH / 8,
        dz_init=LENGTH / 8,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8),
        quadrature=rule,
    )
    backend = backend_class(problem, grid=grid, modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        result = backend.observe().require_field()
    finally:
        backend.close()
    measured = float(np.max(np.abs(result[0, forbidden])) / np.max(np.abs(result)))
    assert measured < bound
    record_rung(
        request,
        backend_name,
        f"azimuthal {kinds[0]}",
        "forbidden mode",
        measured,
        bound,
    )
    if len(modes) == 3:
        assert np.max(np.abs(result[0, 1])) / np.max(np.abs(result)) > bound


@pytest.mark.parametrize("kind", ["TE", "TM"])
def test_transverse_family_linear_flow_matches_its_neff(
    backend_class, backend_name, kind, request
):
    grid = EnvelopeGrid.from_limits(LENGTH, 800e-9, (400e-9, 2e-6), 100e-15)
    mode = MarcatiliMode(125e-6, kind=kind, n=0, cladding=1.45, loss=True)
    field = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-6).build(grid)
    rule = transverse_quadrature([mode], n_r=24, n_theta=5)
    problem = assemble(
        grid,
        [mode],
        [],
        field,
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[LENGTH],
        dz_max=LENGTH / 8,
        dz_init=LENGTH / 8,
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8),
        quadrature=rule,
    )
    exponent = delta_phi(problem.linear, 0.0, LENGTH)
    bound = 32 * EPS * (1 + float(np.max(np.abs(exponent))))
    expected = problem.initial.field0 * np.exp(exponent)
    backend = backend_class(problem, grid=grid, modes=[mode])
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        measured = relative_gap(backend.observe().require_field(), expected)
        assert measured < bound
    finally:
        backend.close()
    record_rung(request, backend_name, f"linear {kind}", "8 steps", measured, bound)


def test_an_underresolved_angular_rule_creates_a_forbidden_harmonic(
    backend_class, backend_name, request
):
    # HE_11 cannot drive HE_31: its Cartesian field is angle-independent, while
    # HE_31 carries harmonic 2. Two midpoint samples alias cos(2 theta) to -1.
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 2e-6), 100e-15)
    modes = [MarcatiliMode(125e-6, n=n, cladding=1.45, loss=False) for n in (1, 3)]
    field = np.zeros((2, 1, grid.n_omega), complex)
    field[0] = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-6).build(grid)[0]
    bound = 7 * (2 * gamma(24 * 9) + 16 * EPS * math.ceil(math.log2(grid.n_to)))
    rules = [transverse_quadrature(modes, n_r=24, n_theta=9)]
    with pytest.warns(UserWarning, match="n_theta >= 9"):
        rules.append(transverse_quadrature(modes, n_r=24, n_theta=2))
    leakage = []
    for rule in rules:
        problem = assemble(
            grid,
            modes,
            [KerrResponse(chi3=1e-26, thg=True)],
            field,
            controller=PIController(rtol=1e-8, atol=1e-15),
            density=1.0,
            save_planes=[LENGTH],
            dz_max=LENGTH,
            dz_init=LENGTH,
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH),
            quadrature=rule,
        )
        backend = backend_class(problem, grid=grid, modes=modes)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(LENGTH)
            result = backend.observe().require_field()
            leakage.append(float(np.max(np.abs(result[0, 1])) / np.max(np.abs(result))))
        finally:
            backend.close()
    assert leakage[0] < bound
    assert leakage[1] > bound
    record_rung(
        request, backend_name, "angular alias control", "resolved", leakage[0], bound
    )
    record_rung(
        request,
        backend_name,
        "angular alias control",
        "aliased (must exceed bound)",
        leakage[1],
        bound,
    )
