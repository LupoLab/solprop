# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 35 phase 1: the carrier frame of the exponential Adams history, on numpy_ref.

The carrier frame interpolates, inside each step, ``G_j = exp(sigma (1 - theta_j)) F_j``
and integrates it against the moments at ``x - sigma`` (section 4.1). These tests are
section 7.1's on the reference backend: the interface and its refusals, the reference
rule, exactness on a carrier-following source (the march's own arithmetic, dense
planes and the step cap included), the aliasing and stability witnesses of sections
4.3 and 4.4, varying operators and coupled sections, the grids and routes, resume, and
the rounding budget the compiled backends' parity will use.

The manufactured-source runs replace the backend's lab-frame source by a function of
``z`` alone, ``F = sum p exp(i rho z)`` per bin: every history sample is then exact
whatever the march's state, and the solution is closed-form.
"""

import dataclasses
import functools
import math
import warnings
from types import SimpleNamespace
from typing import Literal

import mpmath
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm

import flow_parity
from bounds import EPS, assert_within, carrier_increment_budget
from solprop.backends import available
from solprop.backends.exponential import (
    CarrierReference,
    _carrier_refusal,  # pyright: ignore[reportPrivateUsage]
    _nearest_bin,  # pyright: ignore[reportPrivateUsage]
    carrier_reference,
    exponential_weights,
    phi_functions,
)
from solprop.backends.multistep import (
    AdamsMarch,
    CoupledBlock,
    ExponentialKernels,
    HostHistory,
    coupled_moments,
)
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import BackendError, SolpropWarning, StepperError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import Frame
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.stepper.multistep import (
    CARRIER_PHASE_CAP,
    PEC_REAL_INTERVAL,
    ComparisonMultistep,
    Multistep,
    lagrange_coefficients,
)
from solprop.translate.coupled import schur_blocks, schur_eigenbasis

OFF = Multistep()
SCALAR = Multistep(carrier_frame="scalar")

# --- fixtures -------------------------------------------------------------------------

THG = {"field": None, "envelope": False, "envelope_thg": True}
"""The third-harmonic choice per grid: the field grid's default (on), off, on."""


@functools.cache
def capillary(  # noqa: PLR0913 - the case's keywords, cached together
    integrator: Multistep | RungeKutta,
    length: float = 2e-3,
    *,
    n_save: int = 5,
    interpolate: bool = True,
    rtol: float = 1e-8,
    grid_kind: str = "field",
    energy: float = 1e-7,
    modes: int = 1,
    frame_offset: float = 0.0,
    collar: str = "spectral",
) -> Simulation:
    """Return a 16 um, 40 bar hydrogen capillary on a 200 fs grid: kappa near -5.5e3/m.

    The dense gas of SEP 35's cases C and D, reduced: Kerr only (no Raman, no plasma),
    so that a short window runs in seconds. `frame_offset` moves a fixed custom frame's
    reference frequency by that fraction of the grid's. `collar` is ``"spectral"`` (no
    temporal collar: a manufactured source replaces the whole source, the collar's
    term with it), ``"weak"`` (a 1 m reference length, so that accuracy sets a short
    window's step) or ``"default"`` (the window's own in-stage collar, which then binds
    the step: 1.4 um at PEC5/6 on 3 mm).
    """
    limits = (200e-9, 2e-6)
    grid = (
        FieldGrid.from_limits(length, 800e-9, limits, 200e-15)
        if grid_kind == "field"
        else EnvelopeGrid.from_limits(
            length, 800e-9, limits, 200e-15, thg=grid_kind == "envelope_thg"
        )
    )
    collars = {
        "spectral": absorber(grid, temporal=False),
        "weak": absorber(grid, boundary_length=1.0),
        "default": None,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        simulation = Simulation.from_capillary(
            16e-6,
            length,
            "H2",
            40.0,
            grid=grid,
            lambda0=800e-9,
            fwhm=30e-15,
            energy=energy,
            modes=modes,
            rtol=rtol,
            atol=1e-12,
            noise=None,
            backend="numpy_ref",
            n_save=n_save,
            integrator=integrator,
            raman=False,
            plasma=False,
            thg=THG[grid_kind],
            interpolate_planes=interpolate,
            absorber=collars[collar],
        )
    if frame_offset:
        simulation.frame = Frame(
            grid.omega0 * (1.0 + frame_offset), subtract_beta0=False, beta1=None
        )
    return simulation


def reference_rate(problem) -> float:
    """Return ``Im D`` at the reference bin of a constant operator, 1/m (``-kappa``)."""
    grids = problem.grids
    index = _nearest_bin(grids.omega, grids.omega0)
    return float(problem.linear.phi_constant[0, 0, 0, index].imag)


@dataclasses.dataclass(frozen=True)
class Manufactured:
    """A source of ``z`` alone, ``F = sum_m p_m exp(i rho_m z)``, and its solution."""

    profiles: tuple[np.ndarray, ...]
    rates: tuple[np.ndarray, ...]

    def source(self, z: float) -> np.ndarray:
        """Return ``F(z)``."""
        total = np.zeros_like(self.profiles[0])
        for profile, rate in zip(self.profiles, self.rates, strict=True):
            total = total + profile * np.exp(1j * rate * z)
        return total

    def exact(self, start: np.ndarray, operator: np.ndarray, z: float) -> np.ndarray:
        """Return ``exp(z D) u_0 + int_0^z exp((z - s) D) F(s) ds``, in closed form."""
        total = np.exp(z * operator) * start
        for profile, rate in zip(self.profiles, self.rates, strict=True):
            total = total + np.exp(z * operator) * profile * _integral(
                1j * rate - operator, z
            )
        return total


def _integral(a, z):
    """Return ``(exp(z a) - 1) / a``, with its limit ``z`` where ``a`` vanishes."""
    safe = np.where(a == 0, 1.0, a)
    return np.where(a == 0, z, np.expm1(z * a) / safe)


def random_profile(problem, seed=35) -> np.ndarray:
    """Return a unit-scale random amplitude per active bin, the source's spectrum."""
    rng = np.random.default_rng(seed)
    shape = problem.linear.phi_constant.shape
    profile = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 1e3
    return profile * problem.grids.active_mask


def manufactured(problem, harmonic=0.0, stationary=False) -> Manufactured:
    """Return a carrier-following source on the problem's active bins.

    Its rate is the reference bin's ``Im D`` everywhere, the carrier's; ``harmonic``
    adds a component of that relative amplitude at three times the rate (the
    third-harmonic remainder of section 4.2), and ``stationary`` makes the source
    constant instead (the residual ``kappa`` the shift gives a stationary part).
    """
    profile = random_profile(problem)
    shape = profile.shape
    rate = 0.0 if stationary else reference_rate(problem)
    profiles, rates = [profile], [np.full(shape, rate)]
    if harmonic:
        profiles.append(harmonic * profile)
        rates.append(np.full(shape, 3 * rate))
    return Manufactured(tuple(profiles), tuple(rates))


def per_band(problem) -> Manufactured:
    """Return a source rotating at three times the carrier's rate above 2 omega0.

    The developed third harmonic's rotation, which the scalar shift leaves a residual
    2 kappa.
    """
    profile = random_profile(problem)
    grids = problem.grids
    multiple = np.where(grids.omega >= 2.0 * grids.omega0, 3.0, 1.0)
    rates = multiple * reference_rate(problem) * np.ones(profile.shape)
    return Manufactured((profile,), (rates,))


def march(
    simulation,
    source=None,
    *,
    schedule=None,
    method=None,
    prepare=None,
    backend_class=NumpyRefBackend,
):
    """Run a simulation's problem on numpy_ref; return planes, errors, accepted steps.

    With `source`, the lab-frame source is replaced by it and each plane's error
    against the closed form is the largest active bin's, relative to the field.
    `prepare` may rewrite the problem first (a shifted or non-fundamental launch);
    `backend_class` is the backend run.
    """
    problem = simulation.translate()
    stepper = problem.stepper
    if schedule is not None:
        stepper = dataclasses.replace(stepper, schedule=schedule)
    if method is not None:
        stepper = dataclasses.replace(stepper, integrator=method)
    problem = dataclasses.replace(problem, stepper=stepper)
    if prepare is not None:
        problem = prepare(problem)
    backend = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
    active = problem.grids.active_mask
    try:
        if source is not None:

            def lab_source(lab, position, *, purpose="predictor"):
                del lab, purpose
                return source.source(position)

            backend._adams_lab_source = lab_source  # pyright: ignore[reportAttributeAccessIssue]
        start = problem.initial.field0
        backend.set_initial(start, None)
        fields, errors, steps = [], [], []
        for plane in stepper.save_planes:
            backend.propagate_to(float(plane))
            observed = backend.observe()
            field = observed.require_field()
            fields.append(field)
            history = observed.step_history
            steps += list(
                np.asarray(history.dz)[np.asarray(history.accepted, dtype=bool)]
            )
            if source is not None:
                exact = source.exact(start, problem.linear.phi_constant, float(plane))
                errors.append(
                    float(np.max(np.abs((field - exact) * active)))
                    / float(np.max(np.abs(exact * active)))
                )
        return SimpleNamespace(
            fields=fields,
            errors=errors,
            steps=np.array(steps),
            march=backend._adams,
            problem=problem,
        )
    finally:
        backend.close()


def carrier_of(march) -> CarrierReference | None:
    """Return a march's carrier reference, which its host engine holds."""
    assert march is not None
    engine = march._engine
    assert isinstance(engine, HostHistory)
    return engine.carrier


def shifted(march) -> bool:
    """Return whether a march's carrier frame can shift its samples."""
    carrier = carrier_of(march)
    return carrier is not None and carrier.active


def relative(field, reference) -> float:
    """Return the normwise relative departure of two fields."""
    return float(np.linalg.norm(field - reference) / np.linalg.norm(reference))


# --- the interface --------------------------------------------------------------------


def test_the_policy_is_a_field_of_the_method():
    assert OFF.carrier_frame == "off"
    assert OFF.name == "exponential-adams-pec5/6-variable"
    assert SCALAR.name == "exponential-adams-pec5/6-variable-scalar-frame"
    # The representation names the policy, so the run's record and a snapshot do.
    assert "carrier_frame='scalar'" in repr(SCALAR)
    assert SCALAR != OFF
    assert CARRIER_PHASE_CAP == 1.0


def test_a_shift_needs_the_exponential_formulation():
    with pytest.raises(StepperError, match="needs the exponential formulation"):
        Multistep(formulation="interaction", carrier_frame="scalar")
    with pytest.raises(StepperError, match="needs the exponential formulation"):
        ComparisonMultistep(carrier_frame="scalar")


def test_the_refusals_name_the_policy_and_the_route():
    everything = frozenset({"scalar"})
    refusal = _carrier_refusal("scalar", "native (device build)", frozenset(), None)
    assert refusal is not None
    assert refusal.startswith(
        "native (device build): Multistep(carrier_frame='scalar') is not delivered"
    )
    for kind in ("free", "radial", "slab"):
        spatial = _carrier_refusal("scalar", "numpy_ref", everything, kind)
        assert spatial is not None and f"free-space route ('{kind}'" in spatial
    # Off, or delivered on a guided route: nothing to refuse.
    assert _carrier_refusal("off", "numba_cpu", frozenset(), "free") is None
    assert _carrier_refusal("scalar", "numpy_ref", everything, None) is None


def test_both_compiled_cpu_backends_accept_the_shift():
    for method in (OFF, SCALAR):
        built = capillary(method)
        NumbaCpuBackend(built.translate(), grid=built.grid, modes=built.modes).close()


def test_a_problem_of_the_older_schema_is_refused():
    simulation = capillary(SCALAR)
    stale = dataclasses.replace(simulation.translate(), schema_version=35)
    with pytest.raises(BackendError, match="version 35"):
        NumpyRefBackend(stale, grid=simulation.grid, modes=simulation.modes)


# --- the reference rule (section 6.1) -------------------------------------------------


def test_the_reference_is_the_active_bin_nearest_the_reference_frequency():
    problem = capillary(SCALAR).translate()
    grids = problem.grids
    reference = carrier_reference(problem, SCALAR)
    assert reference is not None and reference.active
    assert reference.bin == int(np.argmin(np.abs(grids.omega - grids.omega0)))
    assert reference.omega == grids.omega[reference.bin]
    # Equally near bins: the tie goes to the lower index.
    assert _nearest_bin(np.array([0.5, 0.9, 1.1, 2.0]), 1.0) == 1
    assert carrier_reference(problem, OFF) is None


def test_the_policy_is_inert_where_no_carrier_phase_is_kept():
    envelope = carrier_reference(
        capillary(SCALAR, grid_kind="envelope").translate(), SCALAR
    )
    assert envelope is not None and not envelope.active
    assert "without the third harmonic" in envelope.reason
    # With the third harmonic the operator is the field grid's.
    thg = carrier_reference(
        capillary(SCALAR, grid_kind="envelope_thg").translate(), SCALAR
    )
    assert thg is not None and thg.active
    # An inactive reference bin: sigma = 0 for the run, with its reason.
    problem = capillary(SCALAR).translate()
    nearest = _nearest_bin(problem.grids.omega, problem.grids.omega0)
    mask = problem.grids.active_mask.copy()
    mask[nearest] = False
    grids = dataclasses.replace(
        problem.grids, active_mask=mask, norm_mask=problem.grids.norm_mask & mask
    )
    holed = dataclasses.replace(problem, grids=grids)
    inactive = carrier_reference(holed, SCALAR)
    assert inactive is not None and not inactive.active
    assert f"({nearest}) is inactive" in inactive.reason
    with pytest.raises(BackendError, match="inert"):
        inactive.shift(np.zeros((1, 1, 1, 3), dtype=complex))
    assert inactive.phase(np.ones((1, 1, 1, 3), dtype=complex)) == 0.0


def test_the_shift_is_per_member_and_the_cap_reads_the_largest():
    exponent = np.zeros((2, 3, 2, 4), dtype=complex)
    exponent[0, 0, 0, 1] = -0.1 + 0.3j
    exponent[1, 0, 0, 1] = -0.2 - 0.7j
    # Other modes and polarisations never set the shift.
    exponent[:, 1:, :, :] = 5j
    exponent[:, :, 1, :] = 9j
    reference = CarrierReference("scalar", 1, 1.0, "")
    shift = reference.shift(exponent)
    assert shift.shape == (2, 1, 1, 1)
    # Imaginary only: the loss is never shifted (section 10).
    assert shift.ravel().tolist() == [0.3j, -0.7j]
    assert reference.phase(exponent) == pytest.approx(0.7)


# --- exactness (sections 4.1 and 7.1) --------------------------------------------


def manufactured_budget(run) -> float:
    """Return the rounding budget of a manufactured run, relative to the field.

    A step's increment carries a few hundred ulp of its scale, the moments' 256
    (``bounds.PHI_ULPS``) dominating; the flow ``exp(x)`` and the sums add a handful
    more. Every step's rounding is then carried by flows of modulus at most one (the
    operator is lossy, never amplifying), so the steps' contributions add: 512 eps a
    step, fixed before the comparison.
    """
    return len(run.steps) * 512 * EPS


TOLERANCE_LEVEL = 1e-11
"""Below the 2e-10 to 1e-9 the unexact runs measure at rtol 1e-8, far above rounding."""


def test_a_source_following_the_frame_is_stepped_exactly():
    # F = p exp(i rho z) with rho the frame's own rate: the rotated samples are
    # constant, so every order's polynomial is exact and the step is exact to
    # rounding, at the planes interpolated inside steps as at the step ends
    # (section 4.1's dense formula with the accepted step's sigma).
    simulation = capillary(SCALAR)
    problem = simulation.translate()
    source = manufactured(problem)
    run = march(simulation, source)
    assert run.march.ledger.dense_observations > 0
    assert max(run.errors) <= manufactured_budget(run)
    # Unshifted, the same source is interpolated while it rotates: a tolerance-level
    # error.
    unshifted = march(simulation, source, method=OFF)
    assert min(unshifted.errors) > TOLERANCE_LEVEL
    # The exact steps grow until the cap holds them at one radian of carrier phase;
    # a capped step is not a landing, so the history never restarts.
    phases = run.steps * abs(reference_rate(problem))
    assert np.max(phases) <= CARRIER_PHASE_CAP * (1 + 1e-9)
    assert np.max(phases) >= 0.999 * CARRIER_PHASE_CAP
    assert run.march.ledger.restarts == 0
    # Measured 19 against 37 calls: the unshifted steps resolve the rotation.
    assert run.march.ledger.total < unshifted.march.ledger.total


def test_the_scalar_shift_is_not_exact_for_the_harmonic_band():
    # A source with bins at three times the rate above 2 omega0, a developed third
    # harmonic: the scalar shift leaves them a residual 2 kappa, which it resolves to
    # the tolerance, not exactly.
    simulation = capillary(SCALAR)
    run = march(simulation, per_band(simulation.translate()))
    assert min(run.errors) > TOLERANCE_LEVEL
    assert max(run.errors) < 1e-7


def test_the_dense_formula_is_the_integral_it_stands_for():
    # Section 4.1's interpolated plane against independent quadrature, on irregular
    # nodes, a lossy rotating exponent and an imaginary shift, at interior fractions
    # and the endpoint: the production moments and passes.
    nodes = np.array([1.0, 0.0, -0.7, -1.9])
    x = np.array([-0.4 + 7.0j, -1.5 + 0.3j]).reshape((1, 1, 1, 2))
    sigma = 2.3j
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(4, 1, 1, 1, 2)) + 1j * rng.normal(size=(4, 1, 1, 1, 2))
    rotated = [
        np.exp(sigma * (1 - node)) * one
        for node, one in zip(nodes, samples, strict=True)
    ]
    coefficients = lagrange_coefficients(list(nodes))
    kernels = ExponentialKernels()
    count = len(nodes)

    def polynomial(theta, element):
        """The rotated samples' interpolant at ``theta``."""
        basis = np.array([theta**k for k in range(count)])
        values = np.array([one[0, 0, 0, element] for one in rotated])
        return (coefficients @ basis) @ values

    for xi in (0.2, 0.63, 1.0):
        moments = kernels.moments(xi * (x - sigma), count)
        powers = np.array([xi ** (k + 1) for k in range(count)])
        scaled = moments * powers.reshape((count, 1, 1, 1, 1))
        found = np.exp(-sigma * (1 - xi)) * kernels.apply(
            scaled, coefficients, rotated, 1.0
        )
        for element in range(2):
            rate = complex(x[0, 0, 0, element])

            def integrand(s, part, rate=rate, element=element, xi=xi):
                value = (
                    np.exp((xi - s) * rate)
                    * np.exp(-sigma * (1 - s))
                    * polynomial(s, element)
                )
                return value.real if part == 0 else value.imag

            exact = complex(
                quad(integrand, 0.0, xi, args=(0,), epsabs=1e-15, epsrel=1e-13)[0],
                quad(integrand, 0.0, xi, args=(1,), epsabs=1e-15, epsrel=1e-13)[0],
            )
            assert abs(found[0, 0, 0, element] - exact) <= 1e-13 * max(1.0, abs(exact))


# --- aliasing and resolution (section 4.3) --------------------------------------------


def one_step(rate, components, h, order, sigma):
    """One exponential PEC step on exact, unit-spaced history; return error and defect.

    ``u' = D u + F(z)`` with ``F = sum a exp(i r z)`` (`components` of ``(a, r)``): the
    corrector through the exact endpoint sample against the closed-form integral, and
    its defect against the ``order``-node predictor, both with the production
    weights at ``h D - sigma`` over the rotated samples.
    """
    nodes = [-float(j) for j in range(order)]
    x = h * rate

    def rotated(theta):
        value = sum(a * np.exp(1j * r * h * theta) for a, r in components)
        return np.exp(sigma * (1 - theta)) * value

    predictor = exponential_weights(nodes, x - sigma)
    corrector = exponential_weights([1.0, *nodes], x - sigma)
    predicted = h * sum(w * rotated(t) for w, t in zip(predictor, nodes, strict=True))
    corrected = h * sum(
        w * rotated(t) for w, t in zip(corrector, [1.0, *nodes], strict=True)
    )
    exact = sum(
        a * np.exp(h * rate) * complex(_integral(1j * r - rate, h))
        for a, r in components
    )
    return abs(corrected - exact), abs(corrected - predicted)


MIXED = ((1.0, 1.0), (0.1, 3.0))
"""The mixed witness: a carrier with a residual 2 kappa component, kappa = 1."""

STATIONARY = ((1.0, 0.0),)
"""A stationary source: the shift gives it a residual kappa."""

TABLE = {
    # h |kappa|, q: error and defect shifted, error and defect unshifted (section 4.3).
    (0.25, 3): (4.0e-5, 1.1e-3, 2.2e-4, 5.1e-3),
    (0.5, 5): (5.9e-4, 1.3e-2, 5.3e-3, 7.8e-2),
    (1.0, 3): (2.8e-2, 1.8e-1, 1.1e-1, 5.4e-1),
    (1.0, 5): (4.2e-2, 4.5e-1, 1.6e-1, 7.6e-1),
    (2.0, 5): (3.7e-1, 1.3, 9.7e-1, 8.2),
}


@pytest.mark.parametrize(("phase", "order"), TABLE, ids=[f"{h}-{q}" for h, q in TABLE])
def test_the_mixed_witness_reproduces_section_4_3(phase, order):
    expected = TABLE[phase, order]
    found = (
        *one_step(1j, MIXED, phase, order, 1j * phase),
        *one_step(1j, MIXED, phase, order, 0.0),
    )
    # The table's two significant figures.
    for value, table in zip(found, expected, strict=True):
        assert value == pytest.approx(table, rel=0.06)


def test_aliasing_blinds_the_defect_in_either_formulation():
    # At pi the shifted step's residual 2 kappa h is 2 pi; unshifted, a
    # carrier-following source at h D = 2 pi i is the mirror image (section 4.3).
    error, defect = one_step(1j, MIXED, math.pi, 5, 1j * math.pi)
    assert error > 0.3 and defect < 1e-14
    for order in (1, 3, 5):
        error, defect = one_step(2j * math.pi, ((1.0, 2 * math.pi),), 1.0, order, 0.0)
        assert error == pytest.approx(1.0, rel=1e-9) and defect < 1e-14


@pytest.mark.parametrize("order", range(1, 7))
@pytest.mark.parametrize("phase", [0.25, 0.5, 0.75, CARRIER_PHASE_CAP])
def test_under_the_cap_the_defect_bounds_the_error_at_every_order(order, phase):
    # SEP 35 Q6: the mixed and stationary witnesses, at every order the method runs.
    # Measured, the defect exceeds the error by 2.8 to 13 at the cap (mixed) and by
    # 5.9 to 27 (stationary), growing with order; the mixed witness's defect first
    # falls below its error at h |kappa| = 2.05 (q = 1) to 2.5 (q = 6). One cap
    # serves every order.
    for components in (MIXED, STATIONARY):
        error, defect = one_step(1j, components, phase, order, 1j * phase)
        assert defect >= 2.5 * error


# --- stability (section 4.4) ----------------------------------------------------------


def spectral_radius(order: int, shifted: complex, damping: float) -> float:
    """Return the spectral radius of exponential PEC q/(q+1), ``u' = D u + lambda u``.

    After the unit-modulus change of coordinates the shifted recurrence is the
    unshifted one at ``x' = h D - sigma`` and the same ``h lambda`` (section 4.4); the
    source ``lambda u`` is evaluated at the predicted state, as in the method.
    """
    nodes = [-float(j) for j in range(order)]
    predictor = exponential_weights(nodes, shifted)
    corrector = exponential_weights([1.0, *nodes], shifted)
    flow = np.exp(shifted)
    size = order + 1
    matrix = np.zeros((size, size), dtype=complex)
    predicted = np.zeros(size, dtype=complex)
    predicted[0] = flow
    predicted[1:] = predictor
    new = damping * predicted
    matrix[0] = corrector[0] * new
    matrix[0, 0] += flow
    matrix[0, 1:] += corrector[1:]
    matrix[1] = new
    for row in range(2, size):
        matrix[row, row - 1] = 1.0
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def stable(order: int, shifted: complex) -> bool:
    """Return whether every ``h |lambda|`` up to the order's interval is stable."""
    dampings = -np.linspace(1e-4, PEC_REAL_INTERVAL[order - 1], 40)
    return all(
        spectral_radius(order, complex(shifted.real, sign * shifted.imag), damping)
        <= 1 + 1e-12
        for sign in (1, -1)
        for damping in dampings
    )


def test_a_residual_phase_can_destabilise_pec():
    # The review's counterexample: PEC3/4 at h lambda = -0.25, inside its interval.
    assert spectral_radius(3, 0j, -0.25) == pytest.approx(0.8879010401, rel=1e-9)
    assert spectral_radius(3, 4j, -0.25) == pytest.approx(1.0245058313, rel=1e-9)


THRESHOLDS = {
    # q: the largest |Im x'| keeping PEC_REAL_INTERVAL valid at Re x' = 0, -0.1, -0.5;
    # None is "> 5.9", beyond the range scanned (section 4.4).
    1: (None, None, None),
    2: (5.1, None, None),
    3: (3.5, None, None),
    4: (3.2, 3.4, None),
    5: (3.0, 3.1, None),
}


@pytest.mark.parametrize("order", sorted(THRESHOLDS))
def test_the_residual_phase_thresholds_of_section_4_4(order):
    for real, threshold in zip((0.0, -0.1, -0.5), THRESHOLDS[order], strict=True):
        if threshold is None:
            assert stable(order, complex(real, 5.9))
        else:
            assert stable(order, complex(real, threshold - 0.05))
            assert not stable(order, complex(real, threshold + 0.1))


# --- adaptive marches on manufactured sources -----------------------------------------


@pytest.mark.parametrize(
    "kind", ["mixed", "stationary"], ids=["third-harmonic remainder", "stationary part"]
)
def test_the_adaptive_march_resolves_what_the_shift_leaves(kind):
    # A source the shift does not make smooth: a residual 2 kappa (a tenth of the
    # carrier's amplitude) or a stationary part (residual kappa), from startup, with
    # the controller's variable steps, and then replayed on its own accepted steps.
    simulation = capillary(SCALAR, rtol=1e-6, n_save=2, interpolate=False)
    problem = simulation.translate()
    source = (
        manufactured(problem, harmonic=0.1)
        if kind == "mixed"
        else manufactured(problem, stationary=True)
    )
    run = march(simulation, source)
    ledger = run.march.ledger
    assert ledger.accepted_steps > 10
    assert len(set(np.round(run.steps / run.steps[-1], 6))) > 3
    # An accuracy envelope against the closed form, fixed before the run: the
    # tolerance per step, accumulated over the steps taken.
    envelope = 1e-6 * ledger.accepted_steps
    assert max(run.errors) < envelope
    phases = run.steps * abs(reference_rate(problem))
    assert np.max(phases) <= CARRIER_PHASE_CAP * (1 + 1e-9)
    # Replay on the recorded schedule: no estimate, the same envelope. The last step
    # closes the sum's rounding onto the plane, which a schedule must land on.
    steps = run.steps.copy()
    steps[-1] = float(problem.stepper.save_planes[-1]) - float(np.sum(steps[:-1]))
    replay = march(simulation, source, schedule=FixedSchedule(steps))
    assert max(replay.errors) < envelope


def test_the_controller_recovers_from_rejections_under_the_frame():
    # A first proposal far too long for a strong 2 kappa remainder at a tight
    # tolerance: the cap shortens it to one radian and the controller then rejects
    # and shrinks, as without the frame.
    simulation = capillary(SCALAR, rtol=1e-10, n_save=2, interpolate=False)
    problem = simulation.translate()
    source = manufactured(problem, harmonic=0.5)

    def long_first_step(problem):
        stepper = dataclasses.replace(problem.stepper, dz_init=1e-3)
        return dataclasses.replace(problem, stepper=stepper)

    run = march(simulation, source, prepare=long_first_step)
    assert run.march.ledger.rejected_steps > 0
    assert max(run.errors) < 1e-10 * run.march.ledger.accepted_steps


# --- the inert policy -----------------------------------------------------------------


def test_the_policy_is_the_present_arithmetic_where_inert():
    # On an envelope grid without the third harmonic sigma = 0: bit for bit the
    # unshifted march, a structural contract (the same operations in the same order).
    off = march(capillary(OFF, grid_kind="envelope", energy=1e-6, length=5e-3))
    scalar = march(capillary(SCALAR, grid_kind="envelope", energy=1e-6, length=5e-3))
    engine = scalar.march._engine
    assert engine.carrier is not None and not engine.carrier.active
    for one, other in zip(off.fields, scalar.fields, strict=True):
        assert np.array_equal(one, other)
    assert np.array_equal(off.steps, scalar.steps)


# --- physical windows: grids, collar, launches, a custom frame ------------------------

WINDOW = 3e-3
"""A short physical window, m: a strong 2 uJ pulse broadening in 40 bar of hydrogen."""


@functools.cache
def converged(
    grid_kind="field", modes=1, frame_offset=0.0, roll=0, launch_mode=0, collar="weak"
):
    """Return the Runge-Kutta reference of a window, three orders tighter."""
    simulation = capillary(
        RungeKutta(),
        WINDOW,
        rtol=1e-11,
        grid_kind=grid_kind,
        energy=2e-6,
        n_save=2,
        interpolate=False,
        modes=modes,
        frame_offset=frame_offset,
        collar=collar,
    )
    return march(simulation, prepare=launched(roll, launch_mode)).fields[-1]


def window(method, **keywords):
    """Return a window's run under `method` at rtol 1e-8."""
    roll, launch_mode = keywords.pop("roll", 0), keywords.pop("launch_mode", 0)
    keywords.setdefault("collar", "weak")
    simulation = capillary(
        method, WINDOW, rtol=1e-8, energy=2e-6, n_save=2, interpolate=False, **keywords
    )
    return march(simulation, prepare=launched(roll, launch_mode))


def launched(roll: int, mode: int):
    """Return a problem rewrite moving the launch `roll` bins and into `mode`."""
    if not roll and not mode:
        return None

    def prepare(problem):
        start = problem.initial.field0
        moved = np.zeros_like(start)
        moved[:, mode] = np.roll(start[:, 0], roll, axis=-1)
        moved = moved * problem.grids.active_mask
        initial = dataclasses.replace(problem.initial, field0=moved)
        return dataclasses.replace(problem, initial=initial)

    return prepare


def within_envelope(run, reference) -> bool:
    """Return whether a window run meets its envelope: rtol per accepted step."""
    return relative(run.fields[-1], reference) < 1e-8 * run.march.ledger.accepted_steps


@pytest.mark.medium
@pytest.mark.parametrize("grid_kind", ["field", "envelope_thg"])
def test_both_grids_that_keep_the_carrier_meet_their_references(grid_kind):
    # Section 7.1: the same physics on the field grid and on the envelope grid with
    # the third harmonic, each against its own converged reference (not each other).
    reference = converged(grid_kind)
    for method in (OFF, SCALAR):
        run = window(method, grid_kind=grid_kind)
        assert within_envelope(run, reference)
    assert shifted(run.march)


@pytest.mark.medium
def test_the_in_stage_collar_shows_no_growth_under_the_frame():
    # SEP 35 Q7: the order weight's collar bound (PEC_REAL_INTERVAL) under the
    # policy. The window's own temporal collar is integrated in the source and binds
    # the step. No growth beyond the reference's: the policy's field meets its
    # accuracy envelope and its energy is the reference's.
    run = window(SCALAR, collar="default")
    assert run.march._collar_rate > 0.0
    reference = converged(collar="default")
    assert within_envelope(run, reference)
    energy = np.sum(np.abs(run.fields[-1]) ** 2)
    assert energy <= np.sum(np.abs(reference) ** 2) * (1 + 1e-8)


@pytest.mark.medium
@pytest.mark.parametrize(
    ("roll", "launch_mode", "modes"),
    [(-40, 0, 1), (40, 0, 1), (0, 1, 2)],
    ids=["red launch", "blue launch", "non-fundamental launch"],
)
def test_a_launch_away_from_the_reference(roll, launch_mode, modes):
    # Section 6.1 and Q5: the pulse need not sit at the frame's reference frequency or
    # in its mode. The shift then removes only part of the rotation, which changes
    # efficiency, never exactness: each meets its envelope.
    reference = converged(modes=modes, roll=roll, launch_mode=launch_mode)
    run = window(SCALAR, modes=modes, roll=roll, launch_mode=launch_mode)
    assert within_envelope(run, reference)


@pytest.mark.medium
def test_a_custom_frame_changes_only_how_much_rotation_is_removed():
    # A fixed frame whose reference is 5 % above the grid's: kappa is the frame's,
    # the shift is read at the grid's reference bin, and the run meets its envelope.
    reference = converged(frame_offset=0.05)
    run = window(SCALAR, frame_offset=0.05)
    assert within_envelope(run, reference)


# --- varying operators (section 4.5) --------------------------------------------------


@functools.cache
def gradient(integrator, rtol: float) -> Simulation:
    """Return a 3 mm hydrogen gradient, 40 to 20 bar, weak collar: varying operator."""
    grid = FieldGrid.from_limits(WINDOW, 800e-9, (200e-9, 2e-6), 200e-15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        return Simulation.from_capillary(
            16e-6,
            WINDOW,
            "H2",
            (40.0, 20.0),
            grid=grid,
            lambda0=800e-9,
            fwhm=30e-15,
            energy=2e-6,
            modes=1,
            rtol=rtol,
            atol=1e-12,
            noise=None,
            backend="numpy_ref",
            n_save=2,
            integrator=integrator,
            raman=False,
            plasma=False,
            absorber=absorber(grid, boundary_length=1.0),
        )


def gradient_run(
    method, compensation: Literal["transported", "untransported"] = "transported"
):
    """Run the gradient window with the given compensation; return field and march."""
    simulation = gradient(method, 1e-8)
    problem = simulation.translate()
    backend = NumpyRefBackend(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        backend._adams = AdamsMarch(backend, method, compensation=compensation)
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        return backend.observe().require_field(), backend._adams
    finally:
        backend.close()


@pytest.mark.medium
@pytest.mark.parametrize("compensation", ["transported", "untransported"])
def test_a_varying_operator_is_compensated_under_the_frame(compensation):
    # Section 4.5: the transported compensation's formula is unchanged and its
    # endpoint is the shifted predictor; the untransported witness is rotated as the
    # samples. Both against a converged Runge-Kutta reference.
    reference = march(gradient(RungeKutta(), 1e-11)).fields[-1]
    field, run = gradient_run(SCALAR, compensation)
    assert run._engine.compensated and shifted(run)
    assert relative(field, reference) < 1e-8 * run.ledger.accepted_steps


# --- coupled sections (section 4.5) ---------------------------------------------------


def shifted_moment(shifted: np.ndarray, k: int) -> np.ndarray:
    """Return ``int_0^1 exp((1 - s) X) s^k ds`` by quadrature, entry by entry."""
    size = shifted.shape[0]
    result = np.empty((size, size), dtype=complex)
    for i in range(size):
        for j in range(size):

            def part(s, index, i=i, j=j):
                value = expm((1 - s) * shifted)[i, j] * s**k
                return value.real if index == 0 else value.imag

            result[i, j] = complex(
                quad(part, 0.0, 1.0, args=(0,), epsabs=1e-13)[0],
                quad(part, 0.0, 1.0, args=(1,), epsabs=1e-13)[0],
            )
    return result


def test_the_shifted_block_keeps_the_true_flow():
    # Moments of h B - sigma I by quadrature, and the flow exp(h B) itself, not the
    # shifted block's: a nonnormal block, on the augmented and eigenbasis paths.
    rng = np.random.default_rng(35)
    generator = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    generator[np.tril_indices(3, -1)] *= 0.05
    vectors, triangular = schur_blocks(generator[:, :, np.newaxis])
    diagonal = -0.4 + 2.5j
    exponent = np.full((1, 3, 1, 1), diagonal)
    sigma = np.full((1, 1, 1, 1), 2.1j)
    block = CoupledBlock(
        active=np.array([True]), vectors=vectors, triangular=triangular, section=0
    )
    dz = 0.3
    total = diagonal * np.eye(3) + dz * generator
    shifted = total - 2.1j * np.eye(3)
    exact = [shifted_moment(shifted, k) for k in range(4)]
    routed = dataclasses.replace(block, basis=schur_eigenbasis(vectors, triangular))
    for candidate in (block, routed):
        moments = coupled_moments(exponent, 4, candidate, dz, carrier=sigma)
        np.testing.assert_allclose(moments.flow_matrix[0], expm(total), atol=1e-13)
        np.testing.assert_allclose(moments.flow_scalar, np.exp(exponent), rtol=1e-15)
        for k in range(4):
            np.testing.assert_allclose(moments.matrices[k, 0], exact[k], atol=1e-11)


def coupled_run(name, integrator, rtol, backend_class=NumpyRefBackend):
    """Run a flow-parity case adaptively to its end; return the field and the march."""
    with warnings.catch_warnings():
        for category in (*flow_parity.QUIET, SolpropWarning):
            warnings.simplefilter("ignore", category)
        simulation = flow_parity.build(
            name,
            schedule=None,
            dz_init=1e-6,
            dz_max=flow_parity.LENGTH,
            rtol=rtol,
            energy=2e-3,
        )
        simulation.settings = dataclasses.replace(
            simulation.settings, integrator=integrator
        )
        problem = simulation.translate()
    backend = backend_class(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        for plane in problem.stepper.save_planes:
            backend.propagate_to(float(plane))
        return backend.observe().require_field(), backend._adams
    finally:
        backend.close()


@pytest.mark.medium
@pytest.mark.parametrize(
    ("name", "bound"),
    [("scalar", 1e-10), ("te_tm", 1e-10), ("gradient", 1e-10), ("taper", 1e-9)],
)
def test_coupled_sections_agree_with_runge_kutta_under_the_frame(name, bound):
    # The constant-Schur cases and the frozen-residual ones (a gradient; a taper
    # crossing section edges, where the history's residual is rebuilt), with the
    # unshifted formulation's own bounds (tests/backends/test_exponential_coupled.py).
    reference, _ = coupled_run(name, RungeKutta(), 1e-11)
    field, run = coupled_run(name, SCALAR, 1e-9)
    assert run is not None and shifted(run)
    assert relative(field, reference) < bound


# --- resume and the method's identity (section 6.5) -----------------------------------


def test_a_snapshot_continues_only_under_its_own_policy():
    simulation = capillary(SCALAR, energy=1e-6, interpolate=False, n_save=3)
    problem = simulation.translate()
    planes = [float(z) for z in problem.stepper.save_planes]
    first = NumpyRefBackend(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        first.set_initial(problem.initial.field0, None)
        first.propagate_to(planes[1])
        stored = first.march_snapshot()
        first.propagate_to(planes[-1])
        continued = first.observe().require_field()
    finally:
        first.close()
    assert stored is not None
    # Under the same policy the saved history continues; with the same steps, orders
    # and history it agrees with the in-memory continuation within a rounding budget
    # of the steps after the plane (SEP 25's matched-method bound, 512 eps a step).
    second = NumpyRefBackend(problem, grid=simulation.grid, modes=simulation.modes)
    try:
        second.set_initial(problem.initial.field0, None, z=stored.z)
        assert second.restore_march(stored) is None
        second.propagate_to(planes[-1])
        resumed = second.observe().require_field()
        assert second._adams is not None
        steps = second._adams.ledger.accepted_steps
    finally:
        second.close()
    assert_within(resumed, continued, max(steps, 1) * 512 * EPS, what="resumed history")
    # Under another policy, or before this SEP (the representation differs): refused.
    for method in (OFF, Multistep(carrier_frame="scalar", order=4)):
        other = dataclasses.replace(
            problem, stepper=dataclasses.replace(problem.stepper, integrator=method)
        )
        backend = NumpyRefBackend(other, grid=simulation.grid, modes=simulation.modes)
        try:
            backend.set_initial(problem.initial.field0, None, z=stored.z)
            reason = backend.restore_march(stored)
        finally:
            backend.close()
        assert reason is not None and "carrier_frame='scalar'" in reason


# --- the parity rounding budget -------------------------------------------------------


def test_the_rounding_budget_bounds_one_shifted_increment():
    # The budget the compiled backends' parity will use (SEP 35 section 7.1), against
    # a 40-digit evaluation of the same increment from the same inputs.
    rng = np.random.default_rng(25)
    nodes = np.array([0.0, -0.8, -1.7, -2.4, -3.1])
    coefficients = lagrange_coefficients(list(nodes))
    count = len(nodes)
    exponent = (
        rng.normal(size=6) * 0.3 - 0.05 + 1j * rng.uniform(-40.0, 40.0, size=6)
    ).reshape((1, 1, 1, 6))
    sigma = 0.9j  # under the cap
    samples = rng.normal(size=(count, 1, 1, 1, 6)) + 1j * rng.normal(
        size=(count, 1, 1, 1, 6)
    )
    shifted = exponent - sigma
    moments = ExponentialKernels().moments(shifted, count)
    rotated = np.exp(sigma * (1.0 - nodes)).reshape((count, 1, 1, 1, 1)) * samples
    found = sum(
        moments[k] * np.tensordot(coefficients[:, k], rotated, axes=1)
        for k in range(count)
    )
    with mpmath.workdps(40):
        exact = np.array(
            [
                exact_increment(
                    complex(exponent[0, 0, 0, element]) - sigma,
                    sigma,
                    nodes,
                    coefficients,
                    samples[:, 0, 0, 0, element],
                )
                for element in range(6)
            ]
        ).reshape((1, 1, 1, 6))
    budget = carrier_increment_budget(
        coefficients, moments, samples, nodes, sigma, shifted
    )
    departure = np.abs(found - exact)
    assert np.all(budget > 0)
    assert np.all(departure <= budget)
    # A phase formed with an error widens the budget by that error alone.
    wider = carrier_increment_budget(
        coefficients, moments, samples, nodes, sigma, shifted, phase_rounding=1e-12
    )
    assert np.all(wider > budget)


def exact_increment(shifted, sigma, nodes, coefficients, samples) -> complex:
    """Return ``sum_k k! phi_{k+1}(x') sum_j c_jk G_j`` in the working precision."""
    argument = mpmath.mpc(shifted)
    total = mpmath.mpc(0)
    for k in range(len(nodes)):
        # k! phi_{k+1}(x') = int_0^1 exp((1 - t) x') t^k dt.
        moment = mpmath.quad(
            lambda t, k=k, argument=argument: mpmath.exp((1 - t) * argument) * t**k,
            [0, 1],
        )
        inner = mpmath.mpc(0)
        for j, node in enumerate(nodes):
            rotation = mpmath.exp(mpmath.mpc(sigma) * (1 - mpmath.mpf(float(node))))
            inner += (
                mpmath.mpf(float(coefficients[j, k]))
                * rotation
                * mpmath.mpc(complex(samples[j]))
            )
        total += moment * inner
    return complex(total)


def test_the_moments_at_the_reference_bin_take_the_near_branch():
    # At the reference bin x' = Re x, small, whatever |Im x| was: the scaled series,
    # not the upward recurrence (section 6.6's changed moment work).
    shifted = np.array([-1e-3 + 0j, -0.5 + 0j])
    phi = phi_functions(shifted, 3)
    np.testing.assert_allclose(phi[1], np.expm1(shifted) / shifted, rtol=1e-15)


# --- the compiled backends (phases 2 and 3) ----------------------------------------

COMPILED: dict[str, type] = {"numba_cpu": NumbaCpuBackend}
"""The compiled backends that deliver the scalar shift."""
if "native" in available():
    from solprop_native import (  # pyright: ignore[reportMissingImports]
        NativeBackend as _NativeBackend,
    )

    COMPILED["native"] = _NativeBackend


def one_trial_inputs(seed=35):
    """Return a random trial: exponent, shift, state, store, slots and coefficients."""
    rng = np.random.default_rng(seed)
    shape = (2, 2, 1, 64)
    exponent = rng.normal(size=shape) * 0.2 - 0.05 + 1j * rng.uniform(-30, 30, shape)
    # One shift per batch member, from its own reference element, under the cap.
    shift = (1j * np.array([0.8, -0.6])).reshape((2, 1, 1, 1))
    state = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    store = rng.normal(size=(7, *shape)) + 1j * rng.normal(size=(7, *shape))
    nodes = [0.0, -0.7, -1.5, -2.6]
    slots = np.array([3, 0, 5, 1], dtype=np.int64)
    return exponent, shift, state, store, nodes, slots


def test_the_compiled_passes_follow_the_reference_within_the_budget():
    # One trial's moments, flow, predictor and corrector on numba_cpu against the
    # reference, from the same inputs: within twice the increment's rounding budget
    # (bounds.carrier_increment_budget) plus the flow's few ulp of the state.
    from solprop.backends.multistep import (  # noqa: PLC0415 - the private helpers
        _carrier_factors,  # pyright: ignore[reportPrivateUsage]
    )
    from solprop.backends.numba_cpu.multistep import (  # noqa: PLC0415
        NumbaExponentialKernels,
    )

    exponent, shift, state, store, nodes, slots = one_trial_inputs()
    order = 3
    width = len(slots) + 1
    reference, compiled = ExponentialKernels(), NumbaExponentialKernels()
    expected = reference.moments(exponent, width, shift=shift)
    found = compiled.moments(exponent, width, shift=shift)
    np.testing.assert_allclose(found, expected, rtol=0, atol=512 * EPS)
    # The flow is the unshifted step's, formed beside the shifted moments.
    np.testing.assert_allclose(compiled.flow(exponent), np.exp(exponent), rtol=4 * EPS)
    factors = _carrier_factors(nodes, len(slots), shift)
    coefficients = lagrange_coefficients(nodes[:order])
    predicted = (
        reference.predict(
            state,
            np.exp(exponent),
            expected,
            store,
            slots[:order],
            coefficients,
            1.0,
            factors=factors[:order],
        ),
        compiled.predict(
            state,
            compiled.flow(exponent),
            found,
            store,
            slots[:order],
            coefficients,
            1.0,
            factors=factors[:order],
        ),
    )
    samples = np.stack([store[slot] for slot in slots[:order]])
    budget = 2 * carrier_increment_budget(
        coefficients, expected, samples, nodes[:order], shift, exponent - shift
    ) + 8 * EPS * np.abs(np.exp(exponent) * state)
    assert np.all(np.abs(predicted[1] - predicted[0]) <= budget)
    # Unrotated rows are a different sum: the factors are applied, not ignored.
    unrotated = compiled.predict(
        state, compiled.flow(exponent), found, store, slots[:order], coefficients, 1.0
    )
    assert np.max(np.abs(unrotated - predicted[1])) > 1e3 * np.max(budget)


def test_numba_cpu_steps_the_frame_exactly():
    # The manufactured carrier-following source through the compiled march: exact to
    # rounding, dense planes and the cap included, in the reference's calls. (The
    # native engine evaluates its source itself, so a host substitute cannot reach it;
    # its dense planes are compared with the reference's below.)
    backend_name = "numba_cpu"
    simulation = capillary(SCALAR)
    source = manufactured(simulation.translate())
    expected = march(simulation, source)
    run = march(simulation, source, backend_class=COMPILED[backend_name])
    assert max(run.errors) <= manufactured_budget(run)
    assert run.march.ledger.total == expected.march.ledger.total
    assert np.max(run.steps) * abs(reference_rate(run.problem)) <= 1 + 1e-9


@pytest.mark.medium
@pytest.mark.parametrize("backend_name", sorted(COMPILED))
def test_a_compiled_backend_follows_the_reference_on_a_matched_window(backend_name):
    # A physical window replayed on the reference's own accepted steps at a fixed
    # order, so both run the same discrete method and history (SEP 25): they differ
    # by rounding alone, 512 eps a step relative to the field.
    method = Multistep(variable_order=False, carrier_frame="scalar")
    simulation = capillary(
        method,
        WINDOW,
        rtol=1e-8,
        energy=2e-6,
        n_save=2,
        interpolate=False,
        collar="weak",
    )
    adaptive = march(simulation)
    steps = adaptive.steps.copy()
    steps[-1] = WINDOW - float(np.sum(steps[:-1]))
    schedule = FixedSchedule(steps)
    expected = march(simulation, schedule=schedule)
    found = march(simulation, schedule=schedule, backend_class=COMPILED[backend_name])
    assert_within(
        found.fields[-1],
        expected.fields[-1],
        len(steps) * 512 * EPS,
        what=f"{backend_name} against numpy_ref",
    )


@pytest.mark.medium
@pytest.mark.parametrize("backend_name", sorted(COMPILED))
@pytest.mark.parametrize("name", ["scalar", "taper"])
def test_a_compiled_backend_follows_the_reference_beside_a_coupled_flow(
    backend_name, name
):
    # The shifted block's moments and true flow, on the compiled coupled passes, in
    # the reference's calls; the bound is the unshifted comparison's
    # (tests/backends/test_exponential_coupled.py).
    expected, reference = coupled_run(name, SCALAR, 1e-8)
    found, run = coupled_run(name, SCALAR, 1e-8, COMPILED[backend_name])
    assert reference is not None and run is not None
    assert run.ledger.total == reference.ledger.total
    scale = np.max(np.abs(expected))
    np.testing.assert_allclose(found, expected, rtol=0, atol=1e-11 * scale)


@pytest.mark.medium
@pytest.mark.parametrize("backend_name", sorted(COMPILED))
def test_a_compiled_backend_interpolates_planes_as_the_reference_does(backend_name):
    # Dense planes inside steps (section 4.1's formula with the accepted step's
    # sigma, held across later trials): the same adaptive run on both backends takes
    # the same steps here, so each plane differs by rounding; a wrong prefactor or an
    # unrotated sample would differ by the step's carrier phase, order one.
    simulation = capillary(
        SCALAR,
        WINDOW,
        rtol=1e-8,
        energy=2e-6,
        n_save=6,
        interpolate=True,
        collar="weak",
    )
    expected = march(simulation)
    found = march(simulation, backend_class=COMPILED[backend_name])
    assert found.march.ledger.dense_observations > 0
    assert np.array_equal(found.steps, expected.steps) or len(found.steps) == len(
        expected.steps
    )
    for one, other in zip(found.fields, expected.fields, strict=True):
        assert_within(
            one, other, len(expected.steps) * 512 * EPS, what=f"{backend_name} plane"
        )


@pytest.mark.parametrize("backend_name", ["numpy_ref", *sorted(COMPILED)])
@pytest.mark.parametrize("operator", ["constant", "gradient"])
def test_the_chord_is_the_engine_s_exponent_at_the_reference(backend_name, operator):
    # Section 6.1: the host's one-element chord, which the cap reads and the native
    # engine takes its shift from, against the backend's own exponent, a constant
    # operator's and a gas gradient's expansion: to rounding of the exponent.
    backends = {"numpy_ref": NumpyRefBackend, **COMPILED}
    simulation = capillary(SCALAR) if operator == "constant" else gradient(SCALAR, 1e-8)
    problem = simulation.translate()
    reference = carrier_reference(problem, SCALAR)
    assert reference is not None and reference.chord is not None
    assert reference.bin is not None
    backend = backends[backend_name](
        problem, grid=simulation.grid, modes=simulation.modes
    )
    try:
        for z1, z2 in ((0.0, 1e-4), (1e-3, 1.7e-3)):
            full = backend._delta_phi(backend._stage_scalars(z1, z2))
            expected = np.asarray(full)[:, 0, 0, reference.bin]
            found = reference.chord.exponent(z1, z2)
            # The same representation evaluated at one element: a few roundings of the
            # exponent (an expansion's combination, a constant's product).
            bound = 64 * EPS * (1 + np.abs(expected))
            assert np.all(np.abs(found - expected) <= bound), (found, expected)
    finally:
        backend.close()


@pytest.mark.medium
@pytest.mark.parametrize("backend_name", sorted(COMPILED))
def test_a_compiled_backend_follows_the_reference_on_a_gradient(backend_name):
    # The transported compensation beside the shift (section 4.5), on the reference's
    # own accepted steps at a fixed order: rounding alone, 512 eps a step.
    method = Multistep(variable_order=False, carrier_frame="scalar")
    simulation = gradient(method, 1e-8)
    adaptive = march(simulation)
    assert adaptive.march._engine.compensated
    steps = adaptive.steps.copy()
    steps[-1] = WINDOW - float(np.sum(steps[:-1]))
    schedule = FixedSchedule(steps)
    expected = march(simulation, schedule=schedule)
    found = march(simulation, schedule=schedule, backend_class=COMPILED[backend_name])
    assert_within(
        found.fields[-1],
        expected.fields[-1],
        len(steps) * 512 * EPS,
        what=f"{backend_name} against numpy_ref on a gradient",
    )
