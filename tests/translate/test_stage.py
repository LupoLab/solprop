# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The per-stage bundle against the three functions it is the device form of.

``translate.stage`` splits what a stage needs into the fixed arrays a backend holds and
the host scalars the translation layer computes per stage. The claim that makes the
split safe is that it is the *same arithmetic*: combining the bundle's weights over the
terms reproduces :func:`~solprop.translate.linop.delta_phi`,
:func:`~solprop.translate.linop.unitary_phase` and
:func:`~solprop.translate.responses.prefactor_at` to the rounding of that one linear
combination, on every representation that has a device form. That is asserted here at
random stage pairs on the P0 case, the gradient capillary and the reduced M2 case, at
:func:`combination_bound`: a backend may contract, vectorise or reorder the sum.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from bounds import gamma
from former_raman import FORMER_N2, former_response
from solprop.exceptions import BackendError, TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.gnlse import SpecifiedMode, problem_from
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.raman import MolecularOptions
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import Gas
from solprop.profiles import PressureGradient
from solprop.pulses import GaussianPulse, SechPulse
from solprop.stepper import PIController
from solprop.translate.chebyshev import PiecewiseChebyshev, build
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import delta_phi, unitary_phase
from solprop.translate.problem import Flags, KernelProblem, assemble
from solprop.translate.responses import prefactor_at
from solprop.translate.stage import (
    anchor,
    combine,
    require_device_ready,
    scalars,
    scalars_at,
    terms,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)


def combination_bound(weights, fixed):
    """Return how far two evaluations of ``sum_k w_k T_k`` may differ, elementwise.

    The bundle is a rearrangement of the same products, not an approximation of them:
    a sum of ``n`` products in any order is within ``gamma_n sum_k |w_k| |T_k|`` of the
    exact one (SEP 25 section 4.1; the ladder's ``gamma`` is in eps, twice Higham's),
    and two such evaluations within twice it. A zero combination stays exactly zero.
    """
    weights = np.asarray(weights, dtype=np.float64)
    magnitude = np.tensordot(np.abs(weights), np.abs(fixed), axes=1)
    return 2 * gamma(weights.size) * magnitude


def assert_combined(got, want, weights, fixed):
    """Assert one combination matches its reference within :func:`combination_bound`."""
    bound = combination_bound(weights, fixed)
    error = np.abs(np.asarray(got) - np.asarray(want))
    assert np.all(error <= bound), float(np.max(error - bound))


def soliton_problem():
    """P0/P1: the GNLSE soliton, whose operator and prefactor are both constant."""
    grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    gnlse = problem_from(
        [0.0, 0.0, -11.83e-27], omega_ref=grid.omega0, gamma=0.11, shock=False
    )
    start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=1.0).build(grid)
    return assemble(
        grid,
        gnlse.mode,
        [gnlse.kerr],
        start,
        controller=PIController(rtol=1e-10, atol=1e-18),
        save_planes=[1.0],
        dz_max=0.1,
        dz_init=1e-4,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
    )


def gradient_capillary():
    """A two-point pressure gradient: both quantities are parameter expansions."""
    grid = FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)
    mode = MarcatiliMode(
        125e-6, "He", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=1e-6).build(grid)
    return assemble(
        grid,
        mode,
        [KerrResponse(chi3=2.2e-52, thg=True)],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[0.1, 0.2],
        dz_max=0.01,
        dz_init=1e-4,
    )


def reduced_m2():
    """The reduced M2 capillary: two modes, helium to a vacuum floor."""
    grid = FieldGrid.from_limits(1.0, 800e-9, (100e-9, 1.5e-6), 200e-15)
    fill = PressureGradient.two_point(1.0, 1.2, 0.0)
    modes = [MarcatiliMode(125e-6, "He", fill, m=m) for m in (1, 2)]
    start = np.repeat(
        GaussianPulse(wavelength=800e-9, fwhm=7.5e-15, energy=275e-6).build(grid), 2, 0
    )
    return assemble(
        grid,
        modes,
        [KerrResponse(chi3=2.2e-52, thg=True)],
        start,
        controller=PIController(rtol=1e-6, atol=1e-10),
        save_planes=[1.0],
        dz_max=0.05,
        dz_init=1e-4,
        quadrature=transverse_quadrature(modes, n_r=8, radial_rule="gauss_kronrod"),
    )


CASES = {
    "soliton": soliton_problem,
    "gradient capillary": gradient_capillary,
    "reduced M2": reduced_m2,
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request):
    """Each device-ready problem the bundle is asserted on, once."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CASES[request.param]()


def stage_pairs(problem, count=12, seed=20260907):
    """Return random ``(z0, z1)`` stage spans inside the tabulated range."""
    low, high = problem.linear.z_span
    rng = np.random.default_rng(seed)
    pairs = [(0.0, 0.0)]
    for _ in range(count):
        z0, z1 = sorted(rng.uniform(low, high, 2))
        pairs.append((float(z0), float(z1)))
    return pairs


# ------------------------------------------------------- the bundle is the same numbers


def test_the_bundle_reproduces_delta_phi(case):
    """One linear combination over the weights, and the propagator is unchanged."""
    held = terms(case)
    assert held.phi is not None
    for z0, z1 in stage_pairs(case):
        bundle = scalars(case, z0, z1)
        assert bundle.phi_delta is not None
        got = combine(bundle.phi_delta, held.phi)
        want = delta_phi(case.linear, z0, z1)
        assert_combined(got, want, bundle.phi_delta, held.phi)


def test_the_bundle_reproduces_the_unitary_phase(case):
    """The noise reads the same weights at the stage position, not a second table."""
    held = terms(case)
    assert held.phi is not None
    for _, z1 in stage_pairs(case):
        bundle = scalars(case, 0.0, z1)
        assert bundle.phi_at is not None
        got = np.imag(combine(bundle.phi_at, held.phi))
        assert_combined(got, unitary_phase(case.linear, z1), bundle.phi_at, held.phi)


def test_the_bundle_reproduces_the_prefactor(case):
    """A constant's weight is exactly one, so the stored array comes back unchanged."""
    held = terms(case)
    assert held.prefactor is not None
    for _, z1 in stage_pairs(case):
        bundle = scalars(case, 0.0, z1)
        assert bundle.prefactor is not None
        got = combine(bundle.prefactor, held.prefactor)
        want = prefactor_at(case.responses, z1)
        assert_combined(got, want, bundle.prefactor, held.prefactor)


def test_a_constant_operator_is_the_one_term_case(case):
    """The distance is the weight, which is what makes ``D h`` exact."""
    if case.linear.phi_representation != "constant":
        pytest.skip("this case took the expansion")
    held = terms(case)
    bundle = scalars(case, 0.25, 0.75)
    assert held.phi is not None
    assert bundle.phi_delta is not None
    assert bundle.phi_at is not None
    assert bundle.prefactor is not None
    assert held.phi.shape[0] == 1
    assert bundle.phi_delta.tolist() == [0.5]
    assert bundle.phi_at.tolist() == [0.75]
    assert bundle.prefactor.tolist() == [1.0]


def test_an_expansion_carries_one_weight_per_term(case):
    """The counts are the representation's, and the two builds are independent."""
    if case.linear.phi_representation != "expansion":
        pytest.skip("this case took the constant fast path")
    held = terms(case)
    bundle = scalars(case, 0.1, 0.2)
    expansion = case.linear.phi_expansion
    assert held.phi is not None
    assert held.prefactor is not None
    assert expansion is not None
    assert bundle.phi_delta is not None
    assert bundle.prefactor is not None
    assert held.phi.shape[0] == expansion.n_terms
    assert bundle.phi_delta.size == held.phi.shape[0]
    assert bundle.prefactor.size == held.prefactor.shape[0]


# ------------------------------------------------------------------- the scalar paths


def test_the_scalar_paths_are_read_at_the_stage_position(case):
    """Every path the responses read, at ``z1`` and not at the step's anchor."""
    for _, z1 in stage_pairs(case):
        bundle = scalars(case, 0.0, z1)
        assert bundle.z1 == z1
        assert bundle.density == float(case.responses.density(z1))
        assert bundle.taper_scale == (
            1.0
            if case.transform.taper_scale is None
            else float(case.transform.taper_scale(z1))
        )
        assert bundle.raman_rates == ()
        assert bundle.preionisation == 0.0
        assert math.isfinite(bundle.density)


def test_the_raman_rates_come_off_the_leaf_in_its_own_order(raman_problem):
    """One rate per undamped row, hashable, so a backend can memoise on it."""
    coefficients = raman_problem.responses.raman_coefficients
    bundle = scalars(raman_problem, 0.0, 0.01)
    assert len(bundle.raman_rates) == coefficients.n_terms
    assert bundle.raman_rates == tuple(
        float(table(0.01)) for table in coefficients.decay_rates
    )
    assert hash(bundle.raman_rates)


def test_the_preionisation_comes_off_the_plasma_leaf(plasma_problem):
    """Zero without a plasma response, and the leaf's value with one."""
    bundle = scalars(plasma_problem, 0.0, 0.05)
    plasma = plasma_problem.responses.plasma_coefficients
    assert plasma is not None
    assert bundle.preionisation == float(plasma.preionisation(0.05))


@pytest.fixture(scope="module")
def raman_problem():
    """A nitrogen Raman response, for the decay rates the bundle carries."""
    grid = EnvelopeGrid.from_limits(0.1, 835e-9, (650e-9, 1.2e-6), 1e-12)
    mode = SpecifiedMode.from_taylor(
        [0.0, 0.0, 2.0e-26], omega_ref=grid.omega0, aeff=1.0e-9
    )
    field = np.zeros((1, 1, grid.n_omega), dtype=complex)
    field[0, 0, 0] = grid.n_t
    return assemble(
        grid,
        mode,
        [
            KerrResponse(chi3=1.0e-49, thg=False),
            former_response(FORMER_N2, MolecularOptions(rotation=True), thg=False),
        ],
        field,
        controller=PIController(rtol=1e-9, atol=1e-22),
        save_planes=[0.02],
        dz_max=0.005,
        dz_init=1e-3,
        density=2.5e25,
        shock=False,
        prefactor_index=1.0,
    )


@pytest.fixture(scope="module")
def plasma_problem():
    """An ionising argon capillary with a pre-ionised fraction."""
    grid = FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)
    mode = MarcatiliMode(
        125e-6, "Ar", PressureGradient.two_point(0.2, 2.0, 0.2), cladding=1.45
    )
    table = build_rate_table(OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov"), (1e9, 2e11))
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=200e-6).build(grid)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return assemble(
            grid,
            mode,
            [
                KerrResponse(chi3=2.2e-52, thg=True),
                PlasmaResponse(Gas.Ar, table, preionisation=0.25),
            ],
            start,
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[0.1],
            dz_max=0.01,
            dz_init=1e-4,
        )


# --------------------------------------------------------------- the device predicate


def test_a_device_ready_problem_has_both_kinds_of_term(case):
    """Constant or expansion, both are a weighted sum of fixed arrays."""
    assert case.device_ready
    assert terms(case).device_ready
    require_device_ready(case, "native")


def test_a_z_table_is_not_a_device_form(tabulated):
    """The general representation has no split into terms and weights."""
    assert not tabulated.device_ready
    held = terms(tabulated)
    assert held.phi is None
    assert not held.device_ready
    bundle = scalars(tabulated, 0.05, 0.1)
    assert bundle.phi_delta is None
    assert bundle.phi_at is None
    assert bundle.prefactor is None
    # The scalar paths are still there: only the two array-valued quantities lose
    # their device form, and a host backend reads them through `delta_phi` instead.
    assert bundle.density == float(tabulated.responses.density(0.1))


def test_an_accelerator_build_refuses_a_z_table_by_name(tabulated):
    """The message says which quantity, its size, why, and what runs it instead.

    Only an accelerator build calls the refusal (SEP 29 section 6.6): every CPU
    backend runs the table, the native backend's CPU build first among them.
    """
    with pytest.raises(BackendError, match="native cannot run this problem") as bad:
        require_device_ready(tabulated, "native")
    message = str(bad.value)
    assert "on an accelerator" in message
    assert "the linear operator Phi" in message
    assert "the response prefactor" in message
    size = (
        tabulated.linear.phi_coefficients.nbytes
        + tabulated.responses.prefactor.coefficients.nbytes
    )
    assert f"{size / 1e6:.3g} MB" in message
    assert "ParameterisedMode" in message
    assert message.index("a CPU build of native") < message.index("numba_cpu")
    assert 'backend="numpy_ref"' in message


@pytest.mark.parametrize("quantity", ["phi", "prefactor"])
def test_an_accelerator_refusal_sizes_the_one_table_taken(quantity):
    """Either quantity may be the table beside the other's expansion: its size alone."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = gradient_capillary()
    if quantity == "phi":
        table = dataclasses.replace(problem, linear=_tabulated_phi(problem))
        assert table.linear.phi_coefficients is not None
        size = table.linear.phi_coefficients.nbytes
        assert table.responses.prefactor is None
    else:
        prefactor = _tabulated_prefactor(problem)
        table = dataclasses.replace(
            problem,
            responses=dataclasses.replace(
                problem.responses,
                prefactor_representation="table",
                prefactor=prefactor,
                prefactor_expansion=None,
            ),
        )
        size = prefactor.coefficients.nbytes
        assert table.linear.phi_coefficients is None
    with pytest.raises(BackendError, match="on an accelerator") as bad:
        require_device_ready(table, "native")
    message = str(bad.value)
    assert f"{size / 1e6:.3g} MB" in message
    assert ("the linear operator Phi" in message) == (quantity == "phi")
    assert ("the response prefactor" in message) == (quantity == "prefactor")


@pytest.fixture(scope="module")
def tabulated():
    """The gradient capillary with both quantities on the z table, as a fallback gets.

    Nothing this package builds today takes the z table for a capillary -- the mode
    declares its parameter paths, so the expansion serves it -- so the tables are built
    here from the values the expansions give and substituted, the way a mode set that
    declares nothing would present them.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = gradient_capillary()
    linear = _tabulated_phi(problem)
    responses = dataclasses.replace(
        problem.responses,
        prefactor_representation="table",
        prefactor=_tabulated_prefactor(problem),
        prefactor_expansion=None,
        prefactor_constant=None,
    )
    rebuilt = KernelProblem(
        schema_version=problem.schema_version,
        grid_kind=problem.grid_kind,
        shape=problem.shape,
        flags=Flags.from_content(
            problem.shape, linear, problem.transform, responses, problem.initial
        ),
        grids=problem.grids,
        linear=linear,
        transform=problem.transform,
        responses=responses,
        initial=problem.initial,
        stepper=problem.stepper,
        stats_spec=problem.stats_spec,
    )
    rebuilt.validate()
    return rebuilt


def _frozen(values):
    """Return a read-only C-contiguous copy, as every array crossing the seam is."""
    frozen = np.ascontiguousarray(values)
    frozen.setflags(write=False)
    return frozen


def _tabulated_phi(problem):
    """Return the operator with ``Phi`` as a z table of the expansion's own values."""
    span = problem.linear.z_span
    built = build(
        lambda z: np.stack([delta_phi(problem.linear, 0.0, float(one)) for one in z]),
        *span,
        tol=1e-10,
    )
    return dataclasses.replace(
        problem.linear,
        phi_expansion=None,
        phi_representation="table",
        phi_breakpoints=built.breakpoints,
        phi_coefficients=built.coefficients,
        phi_degrees=built.degrees,
        hard_kinks=built.hard_kinks,
        phi_constant=None,
    )


def _tabulated_prefactor(problem):
    """Return the prefactor as a z table of the expansion's own values."""
    span = problem.responses.prefactor_span
    built = build(
        lambda z: np.stack([prefactor_at(problem.responses, float(one)) for one in z]),
        *span,
        tol=1e-10,
    )
    return PiecewiseChebyshev(
        _frozen(built.coefficients),
        _frozen(built.breakpoints),
        _frozen(built.degrees),
        _frozen(built.hard_kinks),
    )


# --------------------------------------------------------------------- the guard rails


def test_a_weight_count_that_does_not_match_the_terms_is_refused():
    """The two come from one representation; a mismatch is a wiring error."""
    with pytest.raises(BackendError, match="cannot combine"):
        combine(np.array([1.0, 2.0]), np.zeros((3, 4)))


def test_a_position_outside_the_range_is_refused(case):
    """The bundle is as strict about its span as the functions it replaces."""
    beyond = case.linear.z_span[1] + 1.0
    with pytest.raises(TranslationError, match="outside the"):
        scalars(case, 0.0, beyond)


# ------------------------------------------------------------------ the hoisted anchor


def test_hoisting_the_anchor_changes_nothing_but_the_work(case):
    """Every stage of a step subtracts the same weights, so they are taken once.

    The Clenshaw pass over the scalar tables is the largest part of what the host does
    per step on a gradient -- 0.97 ms of the RDW case's 1.73 ms column -- and a sixth
    of it was the anchor, re-evaluated at every stage. Hoisting it is exact, because
    it is the same subtraction of the same numbers, computed once (structural).
    """
    held = anchor(case, 0.05)
    for fraction in (0.0, 0.2, 0.5, 1.0):
        z1 = 0.05 + fraction * 0.01
        alone = scalars(case, 0.05, z1)
        hoisted = scalars(case, 0.05, z1, at=held)
        for name in ("phi_delta", "phi_at", "prefactor"):
            mine, theirs = getattr(alone, name), getattr(hoisted, name)
            assert (mine is None) == (theirs is None), name
            if mine is not None:
                assert np.array_equal(mine, theirs), name


def test_an_anchor_from_another_position_is_refused(case):
    """It is the step's own anchor or the increment is against the wrong place."""
    with pytest.raises(BackendError, match="the same position"):
        scalars(case, 0.05, 0.06, at=anchor(case, 0.04))


def test_the_anchor_of_a_z_table_carries_nothing(tabulated):
    """The fallback has no weights to hoist, and the bundle says so consistently."""
    held = anchor(tabulated, 0.05)
    assert held.z == 0.05
    assert held.phi is None
    assert scalars(tabulated, 0.05, 0.06, at=held).phi_delta is None


# ------------------------------------------------------ the stages of a step together


def _tables(problem):
    """Every scalar table a bundle reads, for forgetting their kept values."""
    responses = problem.responses
    found = [responses.density, problem.transform.taper_scale]
    found += [table for term in responses.raman_terms for table in term.decay_rates]
    found += [term.density for term in responses.raman_terms]
    found += [
        one
        for c in responses.population_channels
        for one in (c.density, c.preionisation)
    ]
    if responses.kerr_coefficients is not None:
        found.append(responses.kerr_coefficients.density)
    if responses.dispersive_kerr is not None:
        found += list(responses.dispersive_kerr.densities)
    for expansion in (problem.linear.phi_expansion, responses.prefactor_expansion):
        if expansion is not None:
            found.append(expansion.weights)
    return [table for table in found if isinstance(table, PiecewiseChebyshev)]


def _forget(problem):
    """Drop every table's kept values, so each evaluation is formed afresh."""
    for table in _tables(problem):
        table.__dict__["_recent"].clear()


def _same(mine, theirs):
    """Assert two bundles are equal field for field and bit for bit."""
    for field in dataclasses.fields(mine):
        a, b = getattr(mine, field.name), getattr(theirs, field.name)
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            assert a is not None and b is not None, field.name
            assert a.dtype == b.dtype and a.shape == b.shape, field.name
            assert np.array_equal(a.view(np.uint8), b.view(np.uint8)), field.name
        else:
            assert type(a) is type(b), field.name
            assert a == b, field.name


def _step_positions(problem, z0, dz):
    """A trial's positions: its nodes and its end, as a Runge-Kutta trial forms them."""
    nodes = problem.stepper.tableau.nodes
    return [z0 + dz, *(z0 + float(node) * dz for node in nodes)]


def _boundaries(problem):
    """Interior breakpoints of every table, where the interval choice matters."""
    edges = {
        float(edge) for table in _tables(problem) for edge in table.breakpoints[1:-1]
    }
    return sorted(edges)


@pytest.fixture(
    scope="module",
    params=["soliton", "gradient capillary", "reduced M2", "raman", "plasma"],
)
def every_case(request, raman_problem, plasma_problem):
    """The device-ready cases and the two leaves with their own scalar tables."""
    if request.param == "raman":
        return raman_problem
    if request.param == "plasma":
        return plasma_problem
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CASES[request.param]()


def test_the_stages_taken_together_are_the_stages_taken_alone(every_case):
    """``scalars_at`` is ``scalars`` per position, bit for bit (SEP 25 phase 2).

    The batch keeps every operation of the per-position pass -- the interval choice,
    the map onto ``[-1, 1]``, the recursion and the increment formed from the
    integrated weights or ``z1 - z0`` -- so the comparison has no tolerance. Steps
    across and onto every table's interior breakpoints, steps a relative 1e-12 long,
    and steps anchored at the far end of the range, where the anchor dwarfs the step.
    """
    problem = every_case
    low, high = problem.linear.z_span
    steps = [(z0, z1 - z0) for z0, z1 in stage_pairs(problem, count=6) if z1 > z0]
    for edge in _boundaries(problem)[:6]:
        steps += [
            (edge, (high - edge) * 1e-3),
            (edge - (edge - low) * 1e-3, (edge - low) * 1e-3),
        ]
    steps += [(high - high * 2e-12, high * 1e-12), (0.5 * (low + high), 1e-12 * high)]
    steps += [(low, high - low)]
    for z0, dz in steps:
        positions = [min(z, high) for z in _step_positions(problem, z0, dz)]
        held = anchor(problem, z0)
        _forget(problem)
        alone = [scalars(problem, z0, z1, at=held) for z1 in positions]
        _forget(problem)
        together = scalars_at(problem, z0, positions, at=held)
        assert len(together) == len(positions)
        for mine, theirs in zip(together, alone, strict=True):
            _same(mine, theirs)
        _forget(problem)
        for mine, theirs in zip(scalars_at(problem, z0, positions), alone, strict=True):
            _same(mine, theirs)


def test_the_stages_taken_together_refuse_what_one_stage_refuses(case):
    """Outside the range, and against another position's anchor, as ``scalars``."""
    beyond = case.linear.z_span[1] + 1.0
    with pytest.raises(TranslationError, match="outside the"):
        scalars_at(case, 0.0, [0.0, beyond])
    with pytest.raises(BackendError, match="the same position"):
        scalars_at(case, 0.05, [0.06], at=anchor(case, 0.04))
    assert scalars_at(case, 0.05, []) == ()


def test_the_stages_of_a_z_table_taken_together_carry_no_weights(tabulated):
    """The fallback's bundles say so consistently in a batch too."""
    for bundle in scalars_at(tabulated, 0.05, [0.05, 0.06]):
        assert bundle.phi_delta is None
        assert bundle.phi_at is None
