# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native on the retired kernels' synthetic separable operators (SEP 43, 31, 32, 34).

The Numba kernels were called alone on small synthetic leaves: loss and gain, an
exactly zero longitudinal wave number, evanescent columns, an inactive bin, a wall with
gain and a negative real part, low words of 1e-17 and a safety ramp, all at once. No
real medium has those together, so no native run met them. Native's kernels cannot be
called alone, but its operator is read from the leaf: these tests put the same values
on the leaf of a small real problem, with order-one steps and an order-one state, and
compare a whole step, on the low-storage march and on the Runge-Kutta march, with
``numpy_ref``, whose operator is ``separable_values``.

**The bound.** The original's was 128 eps per kernel on order-one operands, with no
transform between them. A step here has four evaluations of a transform pair, so it
takes :func:`low_storage_cases.parity_bound`, the same count over the whole step, with
the growth computed from the leaf: ``exp(max Re D dz)``, where the original's operands
did not grow. **Weaker than the original.** A pass isn't read alone, so a fold's
intermediate registers aren't compared, only what the whole step leaves.
"""

import dataclasses

import numpy as np
import pytest

from backends.low_storage_cases import parity_bound
from backends.test_low_storage import MARCH
from bounds import assert_within, gamma, propagation_growth, step_evaluations
from solprop.backends.host.low_storage import HostLowStorageMarch
from solprop.backends.low_storage import LowStorageMarch
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.exceptions import ConvergenceError
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble
from solprop.translate.spatial import separable_values

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

STEP = 0.3
"""An order-one step, m, so every term of the table is of the same size."""

KERR = KerrResponse(8e5, thg=False)
"""A cubic source of the order of the order-one random state, on these unit axes."""

ACTIVE = (True, True, True, True, False)
"""The original's frequencies: four active and an inactive one, repeated."""


def cycled(values, size):
    """Return `values` repeated to `size`, as an array."""
    return np.resize(np.asarray(values), size)


def common_terms(leaf, wave):
    """Return the leaf's shared arrays: the frame, rates, prefactor and inactive bin."""
    n_omega = leaf.k_squared.size
    return {
        "k_squared": wave,
        "frame": np.resize(np.arange(5, dtype=float) / 8, n_omega),
        "spectral_rate": np.resize(np.arange(5, dtype=float) / 16, n_omega),
        "prefactor_numerator": np.full(n_omega, -2j),
        "prefactor_clamped": np.full(n_omega, -0.5j),
        "active": cycled(ACTIVE, n_omega),
        "evanescent_rate": 0.5,
    }


def free_leaf(leaf):
    """The original's Cartesian leaf on `leaf`'s axes, whose ``kperp^2`` it must keep.

    Loss and gain, a column with Re q exactly zero and an exactly zero one (bins at the
    axes' own values), evanescent columns and an inactive bin, with a safety ramp's rate
    on the outer columns.
    """
    n_omega = leaf.k_squared.size
    axis = np.unique(leaf.kperp_sq)
    wave = cycled(
        [1 + 0j, axis[2] + 1j, axis[3] - 1j, axis[1] + 0j, axis[-1] + 1 + 0j], n_omega
    )
    ramp = np.where(leaf.kperp_sq > 0.5 * axis[-1], 1.5, 0.25)
    return dataclasses.replace(
        leaf,
        **common_terms(leaf, wave),
        k_squared_low=np.zeros(n_omega, complex),
        kperp_rate=np.where(leaf.kperp_sq == 0.0, 0.0, ramp),
    )


def slab_leaf(leaf):
    """The original's planar leaf on `leaf`'s axes.

    As :func:`free_leaf`, with the wall: gain, a negative real part and a zero, low
    words of 1e-17 and a ramp together.
    """
    n_omega = leaf.k_squared.size
    axis = np.unique(leaf.kperp_sq)
    wave = cycled([1 + 0j, 4 + 1j, 4 - 1j, axis[1] + 0j, 9 + 0j], n_omega)
    walls = (
        [0.25 + 0.01j, 0.5 - 0.02j, 1.0 + 0.1j, 5.0, 0],
        [1.5, 0.5j, -0.5, 0.1j, 0],
    )
    n_channels = leaf.kappa_sq.shape[0]
    ramp = np.where(leaf.kperp_sq > 0.5 * axis[-1], 1.0, 0.5)
    return dataclasses.replace(
        leaf,
        **common_terms(leaf, wave),
        k_squared_low=cycled([1e-17, 0, -1e-17, 0, 0], n_omega).astype(complex),
        kperp_sq_low=np.where(leaf.kperp_sq == axis[1], 1e-17, 0.0),
        kappa_sq=np.stack(
            [cycled(walls[row % 2], n_omega) for row in range(n_channels)]
        ).astype(complex),
        kperp_rate=np.where(leaf.kperp_sq == 0.0, 0.0, ramp),
    )


def free_problem(*, responses, n_pol=1, mode="auto"):
    """Return a small Cartesian problem on metre-wide axes: ``kperp^2`` of order one."""
    grid = EnvelopeGrid.from_limits(STEP, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(3.0, 9, 4.0, 12, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        0.8,
        jones=(1,) if n_pol == 1 else (1, 0.35j),
    )
    return (
        assemble(
            grid,
            None,
            responses,
            pulse.build(grid, space, medium=SpatialMedium()),
            spatial_grid=space,
            medium=SpatialMedium(),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[STEP],
            dz_max=STEP,
            dz_init=STEP,
            schedule=FixedSchedule.uniform(0.0, STEP, STEP * 1.0000001),
            integrator=MARCH,
            propagator_mode=mode,  # pyright: ignore[reportArgumentType]
            absorber_scheme="nonlinear",
        ),
        grid,
    )


def slab_problem(*, responses, mode="auto", collar=False, integrator=MARCH):
    """Return a small slab problem on a metre-wide y axis."""
    grid = EnvelopeGrid.from_limits(STEP, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    modes = [SlabMode(63.5e-6, order=p, cladding=1.45) for p in (1, 2, 3)]
    space = PlanarGrid(modes, 3.0, 6)
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        0.8,
        waist_x=45e-6,
        offset_x=10e-6,
    )
    field = pulse.build(grid, space, medium=SpatialMedium())
    return (
        assemble(
            grid,
            None,
            responses,
            field,
            spatial_grid=space,
            medium=SpatialMedium(),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[STEP],
            dz_max=STEP,
            dz_init=STEP,
            schedule=FixedSchedule.uniform(0.0, STEP, STEP * 1.0000001),
            integrator=integrator,
            propagator_mode=mode,  # pyright: ignore[reportArgumentType]
            absorber=absorber(grid, boundary_length=1.0) if collar else None,
            absorber_scheme="nonlinear",
        ),
        grid,
    )


def on_leaf(problem, leaf, *, seed=3):
    """Return `problem` on `leaf`, with an order-one random state on its active bins."""
    rng = np.random.default_rng(seed)
    shape = problem.shape
    field = rng.standard_normal(shape[1:]) + 1j * rng.standard_normal(shape[1:])
    field = field * np.asarray(leaf.active)
    return dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, separable=leaf),
        responses=dataclasses.replace(problem.responses, separable=leaf),
        grids=dataclasses.replace(problem.grids, active_mask=leaf.active),
        flux=dataclasses.replace(problem.flux, base=leaf),
        initial=dataclasses.replace(problem.initial, field0=field[None]),
    )


def run(implementation, problem, grid):
    """Replay the schedule to its end, and return the final state."""
    backend = implementation(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(float(problem.stepper.save_planes[-1]))
        return backend.observe().require_field().copy()
    finally:
        backend.close()


def nonlinear_rate(problem, grid):
    """Return ``3 |N| / |a|`` on the launch, the Kerr term's Lipschitz bound, 1/m."""
    reference = NumpyRefBackend(problem, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)
        state = reference._require_state()
        stage = reference._nonlinear(
            state, reference._stage_scalars(0.0, 0.0), np.ones(problem.shape, complex)
        )[0]
        return 3.0 * float(np.max(np.abs(stage)) / np.max(np.abs(state)))
    finally:
        reference.close()


def step_limit(problem, grid, leaf):
    """Return the one step's relative bound, fixed before native runs."""
    gain = max(0.0, float(np.max(np.real(separable_values(leaf)[0]))))
    rate = gain + nonlinear_rate(problem, grid)
    limit = parity_bound(
        problem,
        growth=propagation_growth(rate, STEP),
        transverse=int(np.prod(problem.transform.geometry.shape)),
    )
    # parity_bound counts the march's four evaluations; a step takes this many.
    return limit * step_evaluations(problem) / 4.0


def assert_the_leaf_is_pathological(leaf):
    """Check the inputs do occur: every case the original named, on this leaf."""
    operator, prefactor = separable_values(leaf)
    active = np.asarray(leaf.active)
    assert not np.all(active)
    # Gain and loss on active bins, and an inactive bin's prefactor exactly zero.
    assert np.max(np.real(operator)[..., active]) > 0.0
    assert np.min(np.real(operator)[..., active]) < 0.0
    assert np.all(prefactor[..., ~active] == 0.0)
    q_real = np.real(leaf.k_squared)[None] - leaf.kperp_sq[:, None]
    if leaf.kappa_sq is not None:
        walled = np.real(leaf.kappa_sq)
        assert np.any(walled < 0.0)
        assert np.any(np.imag(leaf.kappa_sq) < 0.0)
        assert np.any(leaf.k_squared_low != 0.0)
        assert np.any(leaf.kperp_sq_low != 0.0)
    else:
        # Re q exactly zero at a live bin, and evanescent elsewhere.
        assert np.any(q_real[:, active] == 0.0)
    assert np.any(q_real[:, active] < 0.0)
    assert np.any(leaf.kperp_rate != 0.0)


def assert_step_agrees(problem, grid, leaf):
    """One step on native within the bound of ``numpy_ref``, and the mask exact."""
    limit = step_limit(problem, grid, leaf)
    expected = run(NumpyRefBackend, problem, grid)
    actual = run(NativeBackend, problem, grid)
    assert_within(actual, expected, limit, what="against numpy_ref")
    # Class P: the inactive bin's state is a hard zero on both, whatever went in.
    inactive = ~np.asarray(leaf.active)
    np.testing.assert_array_equal(expected[..., inactive], 0.0)
    np.testing.assert_array_equal(actual[..., inactive], 0.0)


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_the_cartesian_march_on_the_original_leaf_is_the_reference_s(n_pol, mode):
    """Inventory 31 (a): loss, gain, Re q = 0, evanescent rows and an inactive bin."""
    problem, grid = free_problem(responses=[KERR], n_pol=n_pol, mode=mode)
    leaf = free_leaf(problem.linear.separable)
    assert_the_leaf_is_pathological(leaf)
    assert_step_agrees(on_leaf(problem, leaf), grid, leaf)


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_planar_march_on_the_original_leaf_is_the_reference_s(mode):
    """Inventory 32: the wall's gain and negative part, 1e-17 low words, the ramp.

    One scalar channel per column: translation refuses ``n_pol = 2`` on the slab, so
    the original's second polarisation has nothing to reach.
    """
    problem, grid = slab_problem(responses=[KERR], mode=mode)
    leaf = slab_leaf(problem.linear.separable)
    assert_the_leaf_is_pathological(leaf)
    assert_step_agrees(on_leaf(problem, leaf), grid, leaf)


@pytest.mark.parametrize(
    ("material", "absorbed"),
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_the_planar_stage_on_the_original_leaf_is_the_reference_s(material, absorbed):
    """Inventory 34: gain, an inactive bin and a ramp together, on the seven stages.

    The original's ``finish_stage`` with each term present or absent: a response and an
    in-stage absorber here. The channels are distinct: the same ``k_y`` is another rate.
    """
    problem, grid = slab_problem(
        responses=[KERR] if material else [],
        collar=absorbed,
        integrator=None,  # pyright: ignore[reportArgumentType]
    )
    leaf = slab_leaf(problem.linear.separable)
    assert_the_leaf_is_pathological(leaf)
    operator = separable_values(leaf)[0][0]
    assert not np.allclose(operator[0], operator[1])
    assert_step_agrees(on_leaf(problem, leaf), grid, leaf)


def with_masked_bin(problem, leaf):
    """Return `problem` with its fullest active bin out of the norm, and the mask."""
    power = np.sum(np.abs(problem.initial.field0) ** 2, axis=(0, 1, 2))
    power[~np.asarray(leaf.active)] = 0.0
    mask = np.array(problem.grids.norm_mask)
    mask[int(np.argmax(power))] = False
    return (
        dataclasses.replace(
            problem, grids=dataclasses.replace(problem.grids, norm_mask=mask)
        ),
        mask,
    )


def test_the_march_s_norm_reads_a_mask_with_a_hard_zero(monkeypatch):
    """Inventory 31 (b): the squared norm the march returns, with a populated bin out.

    The original read it from the fold's partials. Here it is what the engine hands
    ``_accept``, on each backend, against the sum over the mask of the state that step
    leaves, within one sum's rounding.
    """
    problem, grid = free_problem(responses=[KERR])
    leaf = free_leaf(problem.linear.separable)
    problem, mask = with_masked_bin(on_leaf(problem, leaf), leaf)
    seen = []
    # The reference's march and the native bridge's each accept for themselves.
    for march in (HostLowStorageMarch, LowStorageMarch):

        def capture(self, z, dz, squared, accept=march._accept):
            seen.append(squared)
            return accept(self, z, dz, squared)

        monkeypatch.setattr(march, "_accept", capture)
    for implementation in (NumpyRefBackend, NativeBackend):
        seen.clear()
        state = run(implementation, problem, grid)
        assert len(seen) == 1
        masked = float(np.sum(np.abs(state[..., mask]) ** 2))
        whole = float(np.sum(np.abs(state) ** 2))
        reductions = gamma(state.size)
        assert seen[0] == pytest.approx(masked, rel=reductions)
        # Vacuous unless the masked bin moves the sum by far more than that.
        assert abs(whole - masked) > 1e3 * reductions * whole


@pytest.mark.filterwarnings("ignore::solprop.exceptions.ExponentBudgetWarning")
@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.parametrize("implementation", ["numpy_ref", "native"])
def test_a_diverged_march_is_refused_by_name(implementation):
    """Inventory 31 (b)'s other half: a gain past overflow is refused, not returned.

    Native had no diverged march. Every active bin's ``k^2`` carries a gain of about
    ``e^650`` over the step, which translation admits (it refuses a propagator that
    itself overflows), and whose squared norm overflows on the first step.
    """
    problem, grid = free_problem(responses=[KERR])
    leaf = free_leaf(problem.linear.separable)
    leaf = dataclasses.replace(
        leaf, k_squared=np.where(leaf.active, 1.0 - 9.4e6j, leaf.k_squared)
    )
    problem = on_leaf(problem, leaf)
    backend = {"numpy_ref": NumpyRefBackend, "native": NativeBackend}[implementation]
    with pytest.raises(ConvergenceError, match="diverged"):
        run(backend, problem, grid)
