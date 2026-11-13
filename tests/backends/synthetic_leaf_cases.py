# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The retired kernels' synthetic separable leaves on small real problems (SEP 43).

The Numba kernels were called alone on small synthetic leaves: loss and gain, an
exactly zero longitudinal wave number, evanescent columns, an inactive bin, a wall with
gain and a negative real part, low words of 1e-17 and a safety ramp, all at once. These
builders put the same values on the leaf of a small Cartesian or planar problem
(:func:`free_problem`, :func:`slab_problem`, :func:`on_leaf`), with order-one steps and
an order-one state, and bound one step of it before native runs (:func:`step_limit`).
The case catalogue's geometry rows (``geometry_rows.py``) compare native with the
reference on them; ``test_native_synthetic_operator.py`` keeps what a row does not.
"""

import dataclasses

import numpy as np

from backends.low_storage_cases import MARCH, parity_bound
from bounds import propagation_growth, step_evaluations
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.boundaries import absorber
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble
from solprop.translate.routes import flux_of
from solprop.translate.spatial import separable_values

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
    problem, run_request = assemble(
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
    )
    return problem, run_request, grid


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
    problem, run_request = assemble(
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
    )
    return problem, run_request, grid


def on_leaf(problem, leaf, *, seed=3):
    """Return `problem` on `leaf`, with an order-one random state on its active bins."""
    rng = np.random.default_rng(seed)
    shape = problem.shape
    field = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    field = field * np.asarray(leaf.active)
    return dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, separable=leaf),
        responses=dataclasses.replace(problem.responses, separable=leaf),
        grids=dataclasses.replace(problem.grids, active_mask=leaf.active),
        initial=dataclasses.replace(problem.initial, field0=field),
        route=dataclasses.replace(
            problem.route, flux=dataclasses.replace(_flux(problem), base=leaf)
        ),
    )


def nonlinear_rate(problem, run_request, grid):
    """Return ``3 |N| / |a|`` on the launch, the Kerr term's Lipschitz bound, 1/m."""
    reference = NumpyRefBackend(problem, run_request, grid=grid)
    try:
        reference.set_initial(problem.initial.field0, None)
        reference._hoist(0.0)
        state = reference._require_state()
        stage = reference._nonlinear(
            state,
            reference._stage_scalars(0.0, 0.0),
            np.ones((1, *problem.shape), complex),
        )[0]
        return 3.0 * float(np.max(np.abs(stage)) / np.max(np.abs(state)))
    finally:
        reference.close()


def step_limit(problem, run_request, grid, leaf):
    """Return the one step's relative bound, fixed before native runs."""
    gain = max(0.0, float(np.max(np.real(separable_values(leaf)[0]))))
    rate = gain + nonlinear_rate(problem, run_request, grid)
    limit = parity_bound(
        problem,
        run_request,
        growth=propagation_growth(rate, STEP),
        transverse=int(np.prod(problem.transform.geometry.shape)),
    )
    # parity_bound counts the march's four evaluations; a step takes this many.
    return limit * step_evaluations(run_request) / 4.0


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


def _flux(problem):
    """Return a spatial problem's flux plan."""
    flux = flux_of(problem.route)
    assert flux is not None
    return flux
