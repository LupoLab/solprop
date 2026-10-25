# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The low-storage RK4IP march on ``numpy_ref``: SEP 24 section 7.1.

The march replays a schedule with classical RK4 in the midpoint interaction picture,
three registers and no estimate. Every agreement below is a bound fixed before the
comparison (SEP 25): the linear limit's rounding bound of section 4.5, rounding bounds
for the lifecycle's identities, and the truncation errors of the order ladders. None is
bitwise.
"""

import copy
import dataclasses
import itertools
import logging
import math
import signal
import warnings
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from backends.conftest import soliton_field
from backends.test_free_diffraction import flux_width, gaussian_reference, moment_bound
from backends.test_numpy_ref_p2 import (
    BETA2_UNIT,
    Z0,
    dispersion_integral,
    dissipative,  # noqa: F401 - the P2 fixture, read through request
    exact_spectrum,
    smooth_loss_integral,
)
from bounds import (
    EPS,
    UNIT_ROUNDOFF,
    fft_bound,
    gamma,
    propagation_growth,
    step_budget,
)
from solprop.backends.driver import LANDING_ULPS, StepDriver
from solprop.backends.low_storage import HostLowStorage, schedule_key, table_steps
from solprop.backends.memory import estimate_memory, ledger_groups
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import resolve_name
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver.scaffold import gnlse_source
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
    SolpropWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.noise import spatial_noise
from solprop.nonlinear.responses import KerrResponse
from solprop.output import MemoryOutput
from solprop.output.schema import MarchSnapshot
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, beam_radius
from solprop.stats import StatsSpec
from solprop.stepper import (
    DOPRI5,
    FixedSchedule,
    LowStorageRungeKutta,
    PIController,
    calibrate,
)
from solprop.translate.linop import delta_phi
from solprop.translate.problem import PropagatorMode, StepperSettings, assemble

MARCH = LowStorageRungeKutta()

# Section 4.5's argument coefficient, fixed before any measurement. Each flow's exponent
# and the reference's carry about eps |Delta Phi|; the table is built from its key, the
# run's mean step (Q10), so a reused flow is off by |D| |h - h_t| / 2, a few ulps of the
# position times |D|, whose sum over a run telescopes to the run's own phase (the
# |Phi(z_N)| term); and the products add 4 N eps. 16 covers the 8 + 1 of phase 1's
# per-step key with a factor of about two.
ARGUMENT_FACTOR = 16


def marched(problem, schedule, *, mode="auto", **stepper):
    """Return a problem stepped by the low-storage march over `schedule`."""
    settings = dataclasses.replace(
        problem.stepper,
        integrator=MARCH,
        schedule=schedule,
        propagator_mode=mode,
        dz_max=max(schedule.max_step, problem.stepper.dz_max),
        dz_init=min(problem.stepper.dz_init, schedule.max_step),
        **stepper,
    )
    return dataclasses.replace(problem, stepper=settings)


def run(problem, grid, *, planes=None, backend=None):
    """Replay a problem through its save planes; return the final state and backend."""
    made = NumpyRefBackend(problem, grid=grid) if backend is None else backend
    made.set_initial(problem.initial.field0, None)
    for plane in problem.stepper.save_planes if planes is None else planes:
        made.propagate_to(float(plane))
    return made.observe().require_field().copy(), made


def linear_bound(problem, schedule):
    """Section 4.5's per-element bound, ``eps (4N + c (N |D| h + |Phi(z_N)|))``.

    Relative to each element's input magnitude; ``N |D| h`` is summed step by step,
    so a varying operator's bound holds too.
    """
    edges = schedule.edges
    steps = schedule.n_steps
    swept = sum(
        np.abs(delta_phi(problem.linear, float(a), float(b)))
        for a, b in itertools.pairwise(edges)
    )
    final = np.abs(delta_phi(problem.linear, float(edges[0]), float(edges[-1])))
    shape = problem.shape
    return EPS * (
        4 * steps
        + ARGUMENT_FACTOR
        * (np.broadcast_to(swept, shape) + np.broadcast_to(final, shape))
    )


def assert_linear_limit(problem, final, schedule):
    """The march's state is the exact flow of its input within section 4.5's bound."""
    start = problem.initial.field0 * problem.grids.active_mask
    flow = np.exp(delta_phi(problem.linear, float(schedule.z_start), schedule.z_end))
    expected = flow * start
    error = np.abs(final - expected)
    limit = linear_bound(problem, schedule) * np.abs(start)
    # Outside the active band the input is masked to zero and stays exactly zero.
    assert np.all(final[..., ~problem.grids.active_mask] == 0.0)
    assert np.all(error <= limit), float(np.max(error / np.where(limit, limit, 1)))
    assert np.any(error > 0)  # a nontrivial flow, not a no-op


# --- the linear limit (section 7.1) ------------------------------------------------


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
@pytest.mark.parametrize("silent", [False, True])
def test_the_linear_limit_is_the_exact_constant_flow(
    linear_problem, narrow_grid, zero_kerr, mode, silent
):
    """No source: two half-step flows a step compose to the exact flow (section 4.5).

    ``chi3 = 0`` runs every pass and multiplies by nothing; no response at all removes
    the evaluation. Both are the constant modal operator's exact flow.
    """
    length = 0.6
    schedule = FixedSchedule.uniform(0.0, length, length / 200)
    base = linear_problem(
        narrow_grid, save_planes=[length], responses=() if silent else [zero_kerr]
    )
    problem = marched(base, schedule, mode=mode)
    final, backend = run(problem, narrow_grid)
    assert backend.metadata()["propagator_mode"] == mode
    backend.close()
    assert_linear_limit(problem, final, schedule)


@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_linear_limit_holds_on_a_varying_operator(
    linear_problem, soliton_grid, substitute, synthetic_expansion, mode
):
    """P2's analytic operator, ``P != Q``: the flows still telescope to the exact one.

    ``Phi = i T0^2 zeta(Z) Omega^2 / 2 - Gamma(Z) / 2``, with the dispersion and the
    loss
    of ``docs/dev/stepper.md`` section 11's rung P2, varying independently along z.
    """
    length = 1.27
    schedule = FixedSchedule.uniform(0.0, length, length / 60)
    base = linear_problem(soliton_grid, save_planes=[length])
    detuning = base.grids.omega - base.grids.omega0
    active = base.grids.active_mask
    terms = [0.5j * BETA2_UNIT * Z0 * detuning**2 * active, -0.5 * active]

    def weights_of(z):
        scale = np.asarray(z, dtype=float) / Z0
        return np.stack(
            [dispersion_integral(scale), smooth_loss_integral(scale)], axis=-1
        )

    varying = substitute(base, linear=synthetic_expansion(base, terms, weights_of))
    assert not varying.flags.constant_linop
    problem = marched(varying, schedule, mode=mode)
    final, backend = run(problem, soliton_grid)
    # A varying operator's "auto" recomputes its flows (section 4.3).
    assert backend.metadata()["propagator_mode"] == mode
    backend.close()
    assert_linear_limit(problem, final, schedule)


@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_the_cartesian_diffraction_reproduces_its_width_and_gouy_phase(fraction):
    """``test_free_diffraction``'s Gaussian beam, stepped by the march (eight steps).

    Diffraction is the linear flow, which the march applies exactly; the analytic beam
    and its Fourier-series bounds are that test's own.
    """
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(6e-4, 48, 7.5e-4, 60, window_factor=0)
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    wave, waist = grid.omega[index] / C, 80e-6
    distance = fraction * wave * waist**2 / 2
    continuum, axis, spectrum, bound = gaussian_reference(space, waist, wave, distance)
    start = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    start[:, 0, index] = spectrum
    schedule = FixedSchedule.uniform(0.0, distance, distance / 8)
    problem = assemble(
        grid,
        None,
        [],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[distance],
        dz_max=distance,
        dz_init=distance / 8,
        schedule=schedule,
        integrator=MARCH,
    )
    state, backend = run(problem, grid)
    # The free kinds admit the table under this march, and "auto" takes it.
    assert backend.metadata()["propagator_mode"] == "tabulate"
    backend.close()
    state = state[0]
    physical = np.fft.ifft2(state[:, 0, index].reshape(space.n_y, space.n_x)).ravel()
    assert np.max(np.abs(physical - continuum)) < bound
    measured_axis = np.sum(space.on_axis * state[:, 0, index])
    assert abs(np.angle(measured_axis / axis)) < math.asin(bound / abs(axis))
    # The axial-flux fluence's width (test_free_diffraction.flux_width), with the
    # width model's next order, (k0 w)^-4 of it, in the bound.
    expected_width = flux_width(waist, wave, distance)
    radius_bound = moment_bound(space, continuum, bound, expected_width)
    radius_bound += 1e-11 * expected_width
    flux = SpatialFlux.resolve(grid, space, SpatialMedium())
    assert abs(beam_radius(state, flux) - expected_width) < radius_bound


# --- order four (section 7.1) -------------------------------------------------------

# The march must read four within the spread IP-DOPRI5's observed order shows about its
# own classical value on the same ladder, and never more loosely than this (section
# 7.1).
ORDER_SPREAD_CAP = 0.3


def populated(spectra, fraction=1e-2):
    """Return the bins holding at least `fraction` of the run's peak amplitude."""
    peak = np.max(np.abs(np.asarray(spectra)), axis=0)
    return peak >= fraction * np.max(peak)


def driven_phase(problem, step, bins):
    """Return ``max |Im D| h / 2`` over `bins`: section 4.4's asymptotic range."""
    rate = np.broadcast_to(delta_phi(problem.linear, 0.0, step) / step, problem.shape)
    selected = np.broadcast_to(bins, problem.shape)
    return float(np.max(np.abs(np.imag(rate))[selected]) * step / 2)


def ladder(evaluate, base, integrator, floor):
    """Return ``calibrate``'s three-level reading for one integrator on a ladder."""
    return calibrate(
        base,
        lambda schedule: evaluate(schedule, integrator),
        order=4 if integrator is MARCH else 5,
        requested=1e-8,
        floor=floor,
    )


def round_off_floor(evaluate, base):
    """Return a ladder's round-off floor, fixed before its order is read.

    Two flow treatments on one schedule differ by rounding alone (section 4.7), and an
    observable normalised to one carries at least an ``eps`` per source evaluation of
    the finest rung; the larger is the floor ``calibrate`` is given.
    """
    measured = abs(
        evaluate(base, MARCH, "tabulate") - evaluate(base, MARCH, "recompute")
    )
    return max(measured, 4 * base.refined(4).n_steps * EPS)


def assert_order_four(evaluate, base, floor):
    """Read the march's order and IP-DOPRI5's on one ladder, and compare them."""
    march = ladder(evaluate, base, MARCH, floor)
    dopri = ladder(evaluate, base, None, floor)
    assert march.conclusive and dopri.conclusive, (march.message, dopri.message)
    spread = min(ORDER_SPREAD_CAP, abs(dopri.observed_order - 5.0))
    assert abs(march.observed_order - 4.0) <= spread, (
        march.observed_order,
        dopri.observed_order,
    )
    return march, dopri


@pytest.mark.slow
def test_the_third_order_soliton_converges_at_order_four(
    soliton_problem, soliton_grid, scales
):
    """P1, phase-resolved: the recurrence error at 200, 400 and 800 steps.

    The observable is the recurrence error with its phase, ``||A - e^{-i pi/4} A_0||
    / ||A_0||`` (trap 12), whose exact value is zero, so the ladder's Richardson
    estimate of the base error is compared with the analytic error too. The steps are
    inside section 4.4's asymptotic range: over every bin holding a hundredth of the
    run's peak amplitude, ``|Im D| h / 2`` is 0.80 at 200 steps (asserted below one).
    """
    period = scales.period
    found = {}

    def evaluate(schedule, integrator, mode="auto"):
        key = (schedule.n_steps, integrator, mode)
        if key in found:
            return found[key]
        problem = soliton_problem(
            soliton_grid,
            3,
            save_planes=[period],
            dz_max=schedule.max_step,
            dz_init=schedule.max_step,
            schedule=schedule,
            integrator=integrator,
            propagator_mode=mode,
        )
        final, backend = run(problem, soliton_grid)
        backend.close()
        start = problem.initial.field0 * problem.grids.active_mask
        exact = np.exp(1j * scales.recurrence_phase) * start
        found[key] = float(np.linalg.norm(final - exact) / np.linalg.norm(start))
        return found[key]

    base = FixedSchedule.uniform(0.0, period, period / 200)
    planes = [period * k / 8 for k in range(1, 9)]
    probe = soliton_problem(
        soliton_grid,
        3,
        save_planes=planes,
        dz_max=period,
        schedule=FixedSchedule.uniform(0.0, period, period / 400, save_planes=planes),
        integrator=MARCH,
    )
    backend = NumpyRefBackend(probe, grid=soliton_grid)
    backend.set_initial(probe.initial.field0, None)
    spectra = [probe.initial.field0]
    for plane in planes:
        backend.propagate_to(plane)
        spectra.append(backend.observe().require_field())
    backend.close()
    assert driven_phase(probe, base.max_step, populated(spectra)) < 1.0
    march, _ = assert_order_four(evaluate, base, round_off_floor(evaluate, base))
    # Measured: the march reads 3.96 (errors 7.64e-3, 4.93e-4, 3.36e-5) and IP-DOPRI5
    # 5.08 on the same ladder. The Richardson estimate of the base error is the
    # analytic error to 0.1 %; 5 % is asserted.
    assert march.achieved_error == pytest.approx(evaluate(base, MARCH), rel=0.05)


@pytest.mark.medium
def test_the_dissipative_varying_problem_converges_at_order_four(request, soliton_grid):
    """P2: z-varying loss and dispersion with an analytic answer, at 40 to 160 steps.

    The observable is the error against ``test_numpy_ref_p2``'s exact solution. Over the
    bins holding a thousandth of the peak amplitude ``|Im D| h / 2`` is at most 0.53 at
    40 steps (asserted below one).
    """
    build = request.getfixturevalue("dissipative")
    problem, distance, loss_integral = build(stepped=False)
    exact = exact_spectrum(soliton_grid, distance, loss_integral)
    exact = exact * problem.grids.active_mask

    def evaluate(schedule, integrator, mode="auto"):
        stepper = dataclasses.replace(
            problem.stepper,
            integrator=integrator,
            schedule=schedule,
            dz_max=schedule.max_step,
            dz_init=min(problem.stepper.dz_init, schedule.max_step),
            propagator_mode=mode,
        )
        final, backend = run(
            dataclasses.replace(problem, stepper=stepper), soliton_grid
        )
        backend.close()
        return float(np.linalg.norm(final[0, 0, 0] - exact) / np.linalg.norm(exact))

    base = FixedSchedule.uniform(0.0, distance, distance / 40)
    bins = populated([problem.initial.field0[0, 0, 0], exact], 1e-3)
    # The operator varies: take each step's own rate.
    edges = base.edges
    phase = max(
        float(
            np.max(
                np.abs(np.imag(delta_phi(problem.linear, float(a), float(b))))[
                    ..., bins
                ]
            )
            / 2
        )
        for a, b in itertools.pairwise(edges)
    )
    assert phase < 1.0
    # Measured: the march reads 3.96 (errors 8.93e-6, 5.73e-7, 3.62e-8) and IP-DOPRI5
    # 6.24, above its asymptotic order on this ladder, so the cap applies.
    assert_order_four(evaluate, base, round_off_floor(evaluate, base))


@pytest.mark.medium
def test_a_cartesian_kerr_beam_converges_at_order_four():
    """Diffraction and Kerr on ``test_free_ladders``' grids, at 10 to 40 steps.

    A beam of 0.2 uJ over 1 cm, 0.4 Rayleigh lengths: the Kerr phase is the only term
    the march integrates numerically. The observable is the input's overlap with the
    output, rotated so that both its amplitude and its phase enter at first order.
    ``|Im D| h / 2`` is at most 0.14 at 10 steps over the bins holding a thousandth of
    the peak amplitude (asserted below one).
    """
    length = 1e-2
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=2e-7), 80e-6
    )
    start = pulse.build(grid, space, medium=SpatialMedium())

    def evaluate(schedule, integrator, mode="auto"):
        problem = assemble(
            grid,
            None,
            [KerrResponse(1e-22, thg=False)],
            start,
            spatial_grid=space,
            medium=SpatialMedium(),
            controller=PIController(rtol=1e-8, atol=1e-14),
            save_planes=[length],
            dz_max=schedule.max_step,
            dz_init=schedule.max_step,
            schedule=schedule,
            integrator=integrator,
            propagator_mode=cast(
                "PropagatorMode", mode if integrator is MARCH else "auto"
            ),
        )
        final, backend = run(problem, grid)
        backend.close()
        initial = problem.initial.field0
        overlap = np.vdot(initial, final) / np.vdot(initial, initial).real
        return float(np.real(np.exp(0.25j * np.pi) * overlap))

    base = FixedSchedule.uniform(0.0, length, length / 10)
    probe = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=length,
        dz_init=length / 40,
        schedule=base.refined(4),
        integrator=MARCH,
    )
    final, backend = run(probe, grid)
    backend.close()
    bins = populated([probe.initial.field0, final], 1e-3)
    assert driven_phase(probe, base.max_step, bins) < 1.0
    # Measured: the march reads 3.98 and IP-DOPRI5 4.55, below its classical order on
    # this ladder, so the cap applies.
    assert_order_four(evaluate, base, round_off_floor(evaluate, base))


# --- the independent RK4IP (section 7.1) -----------------------------------------


def evaluation_difference(problem, grid, gnlse):
    """Return one source evaluation's departure from the scaffold's, relative.

    Returns the departure and the masked input it was taken at.
    """
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    state = backend._require_state()
    march = backend._adams_lab_source(state, 0.0)[0, 0, 0]
    backend.close()
    scaffold = gnlse_source(gnlse, grid)(state[0, 0, 0])
    assert np.linalg.norm(scaffold) > 0
    return float(np.linalg.norm(march - scaffold) / np.linalg.norm(scaffold)), state


def test_one_evaluation_agrees_with_the_scaffold_s(
    soliton_problem, soliton_grid, gnlse
):
    """The march's source and the scaffold's, at one state, within one pass's bound.

    ``numpy_ref`` multiplies the source by ``grids.omega_window`` and the scaffold does
    not; with the window all ones (the default) the two compose the same response, so
    they agree within an FFT round trip's rounding (``fft_bound`` of the fine grid).
    """
    problem = soliton_problem(soliton_grid, 3, save_planes=[0.1])
    assert np.all(problem.grids.omega_window == 1.0)
    problem = marched(problem, FixedSchedule.uniform(0, 0.1, 0.01))
    difference, _ = evaluation_difference(problem, soliton_grid, gnlse)
    assert difference <= fft_bound(soliton_grid.n_to)


@pytest.mark.medium
def test_whole_runs_agree_with_the_scaffold_far_below_the_truncation_error(
    soliton_problem, soliton_grid, gnlse, scales
):
    """The march against ``propagate_fixed_rk4`` on the N = 3 soliton, 400 steps.

    Not bitwise: the two accumulate the step's bracket in different orders. The bound
    is derived from the measured per-evaluation difference and the step count -- four
    evaluations and about sixteen register operations a step, each relative to the
    state -- carried by the growth ``exp(gamma P_0 z)`` a perturbation can suffer over
    the period (``gamma P_0 = N^2 / L_D``, a stated assumption of the case). It must,
    and does, sit orders below the truncation error the same run reports.
    """
    from solprop.driver.scaffold import propagate_fixed_rk4  # noqa: PLC0415

    steps = 400
    period = scales.period
    schedule = FixedSchedule.uniform(0.0, period, period / steps)
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[period],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
    )
    difference, start = evaluation_difference(problem, soliton_grid, gnlse)
    final, backend = run(problem, soliton_grid)
    backend.close()
    start = start[0, 0, 0]
    scaffold = propagate_fixed_rk4(gnlse, soliton_grid, start, period, steps)[1][-1]
    gap = float(np.linalg.norm(final[0, 0, 0] - scaffold) / np.linalg.norm(start))
    exact = np.exp(1j * scales.recurrence_phase) * start
    truncation = float(np.linalg.norm(scaffold - exact) / np.linalg.norm(start))
    per_step = 4 * difference / UNIT_ROUNDOFF + 16
    growth = propagation_growth(9 / scales.dispersion_length, period)
    bound = step_budget(per_step, steps, growth)
    # Measured: the gap is 7.0e-12, the derived bound 1.0e-6 and the scaffold's own
    # truncation error 4.93e-4.
    assert gap <= bound <= 1e-2 * truncation


# --- lifecycle (section 7.1) ------------------------------------------------------

# A fundamental soliton over two dispersion lengths, 40 uneven steps: a uniform
# schedule's steps differ in their last bits, so a rebuilt flow table differs from
# the one it replaces by rounding.
LIFECYCLE_STEPS = 40


@pytest.fixture
def lifecycle(soliton_problem, narrow_grid, scales):
    """Return ``(**stepper) -> (problem, length)``: the march on an N = 1 soliton."""
    length = 2 * scales.dispersion_length
    planes = [length / 2, length]
    schedule = FixedSchedule.uniform(
        0.0, length, length / LIFECYCLE_STEPS * 1.0000001, save_planes=planes
    )

    def build(**stepper):
        problem = soliton_problem(
            narrow_grid,
            1,
            save_planes=planes,
            dz_max=schedule.max_step,
            dz_init=schedule.max_step,
            schedule=schedule,
            integrator=MARCH,
        )
        if stepper:
            problem = dataclasses.replace(
                problem, stepper=dataclasses.replace(problem.stepper, **stepper)
            )
        return problem, length

    return build


def rounding_bound(problem, schedule, scales):
    """A rounding bound for two replays of one schedule, normwise and relative.

    Section 4.5's ``eps (4N + c (N |D| h + |Phi(z_N)|))`` over the bins holding a
    thousandth of the peak amplitude, carried by ``exp(gamma P_0 z) = exp(z / L_D)``,
    the growth a perturbation of the fundamental soliton can suffer (a stated
    assumption).
    """
    bins = populated([problem.initial.field0], 1e-3)
    limit = linear_bound(problem, schedule)[bins]
    growth = propagation_growth(1 / scales.dispersion_length, schedule.z_end)
    return float(np.max(limit)) * growth


def test_planes_at_natural_steps_match_the_uninterrupted_run(
    lifecycle, narrow_grid, scales
):
    """``step_planes`` stops on the schedule's own edges and changes nothing else."""
    problem, length = lifecycle()
    uninterrupted, backend = run(problem, narrow_grid)
    backend.close()
    natural, _ = lifecycle(step_planes=7)
    backend = NumpyRefBackend(natural, grid=narrow_grid)
    backend.set_initial(natural.initial.field0, None)
    stops = []
    while backend._z < length:
        summary = backend.propagate_to(length)
        stops.append(summary.z)
    final = backend.observe().require_field()
    backend.close()
    edges = problem.stepper.schedule.edges
    # On the same edges, exactly: the 7th, 14th, ... edge, and the end.
    assert stops == [float(edges[k]) for k in (7, 14, 21, 28, 35)] + [length]
    bound = rounding_bound(problem, problem.stepper.schedule, scales)
    assert np.linalg.norm(final - uninterrupted) <= bound * np.linalg.norm(
        uninterrupted
    )


def test_a_lockstep_batch_matches_its_members_run_alone(
    lifecycle, narrow_grid, scales, caplog
):
    """Each member owns its march (section 6.5); the batch is its solo runs.

    The runner holds no march of its own, and its closing debug line still says the
    run has no estimate rather than printing the members' 0.0 as a ratio.
    """
    from solprop.backends.registry import select_batch  # noqa: PLC0415

    problem, length = lifecycle()
    members, alone = [], []
    for factor in (1.0, 1.1):
        member = dataclasses.replace(
            problem,
            initial=dataclasses.replace(
                problem.initial, field0=problem.initial.field0 * factor
            ),
        )
        final, backend = run(member, narrow_grid)
        backend.close()
        alone.append(final)
        loaded = NumpyRefBackend(member, grid=narrow_grid)
        loaded.set_initial(member.initial.field0, None)
        members.append(loaded)
    runner = select_batch(members)
    try:
        with caplog.at_level(logging.DEBUG, logger=type(runner).__module__):
            for plane in problem.stepper.save_planes:
                runner.propagate_to(float(plane))
        observed = runner.observe_members()
        metadata = runner.member_metadata()
    finally:
        runner.close()
    bound = rounding_bound(problem, problem.stepper.schedule, scales)
    for observation, expected, record in zip(observed, alone, metadata, strict=True):
        assert observation.z == length
        assert observation.step_history.n_steps == LIFECYCLE_STEPS
        assert np.all(observation.step_history.error_norm == 0.0)
        assert np.linalg.norm(observation.field - expected) <= bound * np.linalg.norm(
            expected
        )
        assert record["source_evaluations"] == 4 * LIFECYCLE_STEPS
    ratios = [
        record.getMessage()
        for record in caplog.records
        if record.name == type(runner).__module__
        and "tolerance ratio" in record.getMessage()
    ]
    assert len(ratios) == len(problem.stepper.save_planes)
    assert all(line.endswith("not estimated (replayed schedule)") for line in ratios)


@pytest.mark.parametrize("edge", ["plane", "mid-segment"])
def test_a_resume_at_an_edge_agrees_to_rounding(lifecycle, narrow_grid, scales, edge):
    """A resume rebuilds the flow table from its own first step (section 4.3).

    At a save plane and at a mid-segment edge (a natural step's), where the table is
    built from a different step of the segment than the uninterrupted run's: the two
    agree within section 4.5's bound, not bit for bit.
    """
    problem, length = lifecycle()
    uninterrupted, backend = run(problem, narrow_grid)
    backend.close()
    resumable, _ = lifecycle(step_planes=7 if edge == "mid-segment" else 0)
    backend = NumpyRefBackend(resumable, grid=narrow_grid)
    backend.set_initial(resumable.initial.field0, None)
    summary = backend.propagate_to(length / 2)
    stopped = backend.observe()
    backend.close()
    edges = problem.stepper.schedule.edges
    expected_stop = float(edges[7]) if edge == "mid-segment" else length / 2
    assert summary.z == stopped.z == expected_stop
    resumed = NumpyRefBackend(problem, grid=narrow_grid)
    resumed.set_initial(stopped.require_field(), None, z=stopped.z)
    for plane in problem.stepper.save_planes:
        if plane > stopped.z:
            resumed.propagate_to(float(plane))
    final = resumed.observe().require_field()
    assert resumed.march_snapshot() is None
    # A snapshot from an Adams run is not continued: this march keeps no history.
    snapshot = MarchSnapshot(
        z=stopped.z,
        state=stopped.require_field(),
        proposal=1e-3,
        method=repr(MARCH),
        order=1,
        rejections=0,
        last_rejected=False,
        positions=np.array([stopped.z]),
        identities=("accepted",),
        slots=np.array([0]),
        rows={},
    )
    assert "low-storage march, which keeps no history" in str(
        resumed.restore_march(snapshot)
    )
    resumed.close()
    bound = rounding_bound(problem, problem.stepper.schedule, scales)
    assert np.linalg.norm(final - uninterrupted) <= bound * np.linalg.norm(
        uninterrupted
    )


# --- refusals, before allocation (section 5) ---------------------------------------


def test_the_translation_refuses_what_the_march_cannot_replay(
    soliton_problem, narrow_grid
):
    """No schedule, interpolated planes, a step past dz_max: each named and cured."""
    length = 0.1
    schedule = FixedSchedule.uniform(0.0, length, length / 10)
    with pytest.raises(
        TranslationError, match=r"replays a prescribed schedule.*uniform"
    ):
        soliton_problem(narrow_grid, 1, save_planes=[length], integrator=MARCH)
    with pytest.raises(TranslationError, match="drop interpolate_planes"):
        StepperSettings(
            DOPRI5,
            PIController(rtol=1e-8, atol=1e-14),
            1e-9,
            length,
            1e-3,
            1.1,
            schedule,
            [length],
            "auto",
            integrator=MARCH,
            interpolate_planes=True,
        )
    with pytest.raises(TranslationError, match="raise dz_max"):
        soliton_problem(
            narrow_grid,
            1,
            save_planes=[length],
            dz_max=length / 20,
            dz_init=length / 20,
            schedule=schedule,
            integrator=MARCH,
        )


def test_the_translation_refuses_a_coupled_flow():
    """A coupled flow's half-step flows are not exponentials of commuting values."""
    import flow_parity  # noqa: PLC0415

    with warnings.catch_warnings():
        # The qualification case's own warnings (flow_parity.QUIET) and its collar's.
        for category in (*flow_parity.QUIET, SolpropWarning):
            warnings.simplefilter("ignore", category)
        simulation = flow_parity.build("scalar", integrator=MARCH)
        with pytest.raises(
            TranslationError, match=r"diagonal linear flow only.*RungeKutta"
        ):
            simulation.translate()


def test_the_tdse_response_is_refused_by_name(soliton_problem, narrow_grid):
    """Through ``validate``, before the TDSE route's own checks would name Adams.

    The descriptor is the Adams route's; the march's check reads only its presence and
    the current numerator beside it, so stand-ins suffice (the full TDSE path is the
    route's own tests').
    """
    length = 0.1
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[length],
        schedule=FixedSchedule.uniform(0.0, length, length / 10),
        integrator=MARCH,
    )
    spoiled = copy.copy(problem)
    object.__setattr__(spoiled, "tdse", object())
    object.__setattr__(spoiled, "current_numerator", np.zeros(problem.shape[1:]))
    with pytest.raises(TranslationError, match="does not step the TDSE response"):
        spoiled.validate()


def test_a_step_past_rk4_s_stability_interval_is_refused(narrow_grid, gnlse):
    """The in-stage collar's limit is RK4's 2.785, below DOPRI5's 3.3 (section 4.6)."""
    length = 0.2
    collar = absorber(narrow_grid, boundary_N=40)
    rate = None
    for steps in range(5, 200):
        schedule = FixedSchedule.uniform(0.0, length, length / steps)
        arguments = {
            "controller": PIController(rtol=1e-8, atol=1e-16),
            "save_planes": [length],
            "dz_max": schedule.max_step,
            "dz_init": schedule.max_step,
            "schedule": schedule,
            "absorber": collar,
            "absorber_scheme": "nonlinear",
            "density": 1.0,
            "shock": False,
            "prefactor_index": 1.0,
        }
        start = soliton_field(narrow_grid, 1)
        try:
            problem = assemble(
                narrow_grid,
                gnlse.mode,
                [gnlse.kerr],
                start,
                integrator=MARCH,
                **arguments,
            )
        except TranslationError as error:
            assert "below 2.785" in str(error)
            continue
        assert problem.responses.temporal_absorber is not None
        rate = float(np.max(problem.responses.temporal_absorber))
        # The first admitted schedule is the first inside RK4's interval ...
        assert schedule.max_step * rate / 2 <= 2.785
        previous = FixedSchedule.uniform(0.0, length, length / (steps - 1))
        assert previous.max_step * rate / 2 > 2.785
        # ... and DOPRI5 admits the schedule the march refused.
        assemble(
            narrow_grid,
            gnlse.mode,
            [gnlse.kerr],
            start,
            **arguments
            | {
                "schedule": previous,
                "dz_max": previous.max_step,
                "dz_init": previous.max_step,
            },
        )
        break
    assert rate is not None


def spatial_collar_problem(steps, integrator):
    """Return a Cartesian Kerr problem whose in-stage damping is the spatial collar's.

    The temporal collar is the split-step one, applied once per step outside the
    source, so the spatial collar alone sets the stability limit.
    """
    length = 1e-3
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0.25)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    schedule = FixedSchedule.uniform(0.0, length, length / steps)
    arguments = {} if integrator is None else {"integrator": integrator}
    return assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        absorber=absorber(grid),
        absorber_scheme="split_step",
        **arguments,
    ), schedule


def test_a_step_past_rk4_s_interval_beside_the_spatial_collar_is_refused():
    """The spatial collar's limit takes RK4's 2.785 too, through the translation.

    Its effective rate on a Cartesian grid is the collar's largest rate, times one plus
    the check's rounding allowance; the first schedule inside ``h alpha / 2 <= 2.785``
    is admitted, the one before it refused, and DOPRI5 admits that one at 3.3.
    """
    admitted, _ = spatial_collar_problem(400, MARCH)
    geometry = admitted.transform.geometry
    assert geometry is not None and admitted.responses.absorber_scheme == "split_step"
    rounding = 128 * admitted.transform.n_modes * EPS
    largest = float(np.max(geometry.absorber_rate)) * (1 + rounding / (1 - rounding))

    def inside(steps, interval):
        schedule = FixedSchedule.uniform(0.0, 1e-3, 1e-3 / steps)
        return schedule.max_step * largest / 2 <= interval

    first = next(steps for steps in range(2, 400) if inside(steps, 2.785))
    assert not inside(first - 1, 2.785) and inside(first - 1, 3.3)
    spatial_collar_problem(first, MARCH)
    with pytest.raises(TranslationError, match="spatial collar stability limit"):
        spatial_collar_problem(first - 1, MARCH)
    spatial_collar_problem(first - 1, None)


def test_numba_cpu_runs_the_march_with_no_runge_kutta_buffer(
    soliton_problem, narrow_grid
):
    """A widened integrator never reads as none and runs DOPRI5 (section 6.4).

    ``numba_cpu`` builds the march's resident engine (phase 2) and none of the
    Runge-Kutta march's buffers, and its first-same-as-last switch stays off (Q13).
    """
    from solprop.backends.numba_cpu import NumbaCpuBackend  # noqa: PLC0415
    from solprop.backends.numba_cpu.low_storage import NumbaLowStorage  # noqa: PLC0415

    length = 0.1
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[length],
        schedule=FixedSchedule.uniform(0.0, length, length / 10),
        integrator=MARCH,
    )
    backend = NumbaCpuBackend(problem, grid=narrow_grid)
    assert backend._low_storage is not None
    assert isinstance(backend._low_storage.engine, NumbaLowStorage)
    assert not backend._fsal
    ledger = backend.memory_ledger()
    for row in (*RUNGE_KUTTA_ROWS, "advanced", "exponent", "recycled_stage"):
        assert ledger[row] == 0, row
    backend.close()


# --- the step in flight (section 6.2, Q6) -----------------------------------------


class AbandonedError(RuntimeError):
    """The exception injected between two passes of a step."""


def abandon_after(backend, evaluations):
    """Make the backend's source raise after `evaluations` evaluations."""
    original = backend._adams_lab_source
    count = [0]

    def evaluate(lab, position, **keywords):
        count[0] += 1
        if count[0] > evaluations:
            raise AbandonedError
        return original(lab, position, **keywords)

    backend._adams_lab_source = evaluate
    return lambda: setattr(backend, "_adams_lab_source", original)


def test_an_abandoned_step_leaves_every_reader_refusing(lifecycle, narrow_grid, scales):
    """Between passes the state is neither end of the step: nothing may read it.

    ``observe``, ``propagate_to``, ``stats`` and ``march_snapshot`` refuse it, naming
    the step; ``set_initial`` alone clears it, and a resume from the last plane agrees
    with the uninterrupted run to rounding.
    """
    problem, length = lifecycle()
    uninterrupted, backend = run(problem, narrow_grid)
    backend.close()
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(length / 2)
    plane = backend.observe()
    # Two steps and one pass more: the third step is abandoned after its first pass,
    # which has already overwritten the state with its interaction-picture image.
    restore = abandon_after(backend, 4 * 2 + 1)
    with pytest.raises(AbandonedError):
        backend.propagate_to(length)
    restore()
    step = problem.stepper.schedule.steps[LIFECYCLE_STEPS // 2 + 2]
    message = rf"step of {step:.6g} m from z = .* abandoned in flight"
    for reader in (
        backend.observe,
        lambda: backend.propagate_to(length),
        lambda: backend.stats(StatsSpec(names=("energy",))),
        backend.march_snapshot,
    ):
        with pytest.raises(BackendError, match=message):
            reader()
    # The two committed steps are counted; the abandoned one is not.
    assert backend.metadata()["source_evaluations"] == 4 * (LIFECYCLE_STEPS // 2 + 2)
    backend.set_initial(plane.require_field(), None, z=plane.z)
    backend.propagate_to(length)
    final = backend.observe().require_field()
    backend.close()
    bound = rounding_bound(problem, problem.stepper.schedule, scales)
    assert np.linalg.norm(final - uninterrupted) <= bound * np.linalg.norm(
        uninterrupted
    )


def test_a_lockstep_member_left_in_flight_refuses_the_next_step(lifecycle, narrow_grid):
    from solprop.backends.registry import select_batch  # noqa: PLC0415

    problem, length = lifecycle()
    members = []
    for _ in range(2):
        member = NumpyRefBackend(problem, grid=narrow_grid)
        member.set_initial(problem.initial.field0, None)
        members.append(member)
    runner = select_batch(members)
    abandon_after(members[1], 1)
    with pytest.raises(AbandonedError):
        runner.propagate_to(length / 2)
    # Member 0 committed the shared step before member 1 was abandoned; the next call
    # must refuse before stepping it again from there.
    position, state = members[0]._z, members[0]._state.copy()
    with pytest.raises(BackendError, match="abandoned in flight"):
        runner.propagate_to(length / 2)
    assert members[0]._z == position
    assert np.array_equal(members[0]._state, state)
    assert members[0].metadata()["source_evaluations"] == 4
    with pytest.raises(BackendError, match="abandoned in flight"):
        runner.observe_members()
    with pytest.raises(BackendError, match="abandoned in flight"):
        runner.stats(StatsSpec(names=("energy",)))
    runner.close()


def test_a_diverged_state_stops_the_replay_in_flight(lifecycle, narrow_grid):
    """No estimate to watch: the accepted state's norm is the divergence check."""
    problem, length = lifecycle()
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    original = backend._adams_lab_source

    def overflowing(lab, position, **keywords):
        # An overflowed source, as a step past a stability limit leaves one.
        return np.full_like(original(lab, position, **keywords), np.nan)

    backend._adams_lab_source = overflowing
    with pytest.raises(ConvergenceError, match=r"non-finite state.*lost"):
        backend.propagate_to(length)
    with pytest.raises(BackendError, match="abandoned in flight"):
        backend.observe()
    # The diverged step was not committed, so its evaluations are not counted.
    assert backend.metadata()["source_evaluations"] == 0
    backend.close()


@pytest.mark.skipif(
    not hasattr(signal, "SIGINT"), reason="Ctrl-C is delivered by a POSIX signal"
)
@pytest.mark.parametrize("atomic", [False, True], ids=["host engine", "atomic engine"])
def test_an_atomic_engine_commits_its_step_whole(lifecycle, narrow_grid, atomic):
    """Section 6.7: a signal during an atomic engine's step waits for the commit.

    An engine whose step is one native call (``atomic``, schema 32) has the step, the
    finish and the move committed together, so a Ctrl-C raised while it steps lands
    after the step: the backend is not in flight and stands at the step's end. The
    host engine runs Python between its passes, so there the signal abandons the step
    at once, in flight.
    """
    from solprop._interrupts import handled  # noqa: PLC0415

    problem, length = lifecycle()
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert isinstance(engine, HostLowStorage) and not engine.atomic
    engine.atomic = atomic
    original = engine.step

    def interrupted(z, dz):
        squared = original(z, dz)
        # Ctrl-C in-process; os.kill would terminate the process on Windows.
        signal.raise_signal(signal.SIGINT)
        return squared

    engine.step = interrupted
    with handled(), pytest.raises(KeyboardInterrupt):
        backend.propagate_to(length)
    schedule = problem.stepper.schedule
    assert schedule is not None
    if atomic:
        assert backend._in_flight is None
        assert backend._z == float(schedule.edges[1])
        assert backend.metadata()["source_evaluations"] == 4
        backend.observe()
    else:
        assert backend._in_flight == (0.0, float(schedule.steps[0]))
        with pytest.raises(BackendError, match="abandoned in flight"):
            backend.observe()
    backend.close()


def test_the_run_writes_no_checkpoint_from_a_step_in_flight(tmp_path, caplog):
    """The run loop's failure path says why it wrote none; a resume continues."""
    import solprop.driver.run as run_module  # noqa: PLC0415
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.output.hdf5 import open_file  # noqa: PLC0415

    length = 0.02
    schedule = FixedSchedule.uniform(0.0, length, length / 8, save_planes=[length / 2])

    def simulation():
        return Simulation.from_gnlse(
            0.11,
            length,
            [0.0, 0.0, -11.83e-27],
            lambda0=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=1e-12,
            width=100e-15,
            peak_power=9.8,
            pulseshape="sech",
            save_planes=[length / 2, length],
            schedule=schedule,
            integrator=MARCH,
            backend="numpy_ref",
            dz_max=schedule.max_step,
            dz_init=schedule.max_step,
            # No collar: a few fixed steps are refused beside the default one.
            absorber=False,
        )

    uninterrupted = simulation().run(progress="none")
    path = tmp_path / "run.h5"
    original = NumpyRefBackend._adams_lab_source
    count = [0]

    def evaluate(self, lab, position, **keywords):
        count[0] += 1
        if count[0] == 4 * 5 + 2:
            raise AbandonedError
        return original(self, lab, position, **keywords)

    try:
        NumpyRefBackend._adams_lab_source = evaluate
        with (
            caplog.at_level(logging.WARNING, logger=run_module.__name__),
            pytest.raises(AbandonedError) as caught,
        ):
            simulation().run(path, progress="none")
    finally:
        NumpyRefBackend._adams_lab_source = original
    assert any("no terminal checkpoint" in note for note in caught.value.__notes__)
    assert "abandoned in flight" in caplog.text
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is None
        assert stored.outcome is not None
        assert stored.outcome.status == "failed"
        assert stored.outcome.z == length / 2
        assert "no terminal checkpoint" in stored.outcome.message
        assert stored["z"].tolist() == [0.0, length / 2]
    resumed = simulation().run(path, progress="none", resume=True)
    with open_file(path) as stored:
        final = stored["field"][-1]
        assert stored["z"].tolist() == [0.0, length / 2, length]
        # Physical evaluations, four a step of the resumed half, and no logical
        # response calls for the resume to restore (section 5).
        assert stored.outcome is not None
        assert stored.outcome.execution["source_evaluations"] == 4 * 4
        assert "response_calls" not in stored.outcome.execution
        assert not stored.outcome.logical_calls
    del resumed
    expected = uninterrupted["field"][-1]
    # Eight steps of an N = 1 soliton over 0.024 dispersion lengths: a resume at the
    # plane rebuilds the table from the same step length, and the two agree within
    # section 4.5's bound, carried by the soliton's negligible growth here.
    problem = simulation().translate()
    bins = populated([expected], 1e-3).reshape(-1)
    bound = float(np.max(linear_bound(problem, schedule)[..., bins]))
    assert np.linalg.norm(final - expected) <= bound * np.linalg.norm(expected)


def held_soliton(fields="all"):
    """The soliton of the test above on its schedule, keeping `fields`."""
    from solprop.driver import Simulation  # noqa: PLC0415

    length = 0.02
    schedule = FixedSchedule.uniform(0.0, length, length / 8, save_planes=[length / 2])
    simulation = Simulation.from_gnlse(
        0.11,
        length,
        [0.0, 0.0, -11.83e-27],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        pulseshape="sech",
        save_planes=[length / 2, length],
        schedule=schedule,
        integrator=MARCH,
        backend="numpy_ref",
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        # No collar: a few fixed steps are refused beside the default one.
        absorber=False,
    )
    if fields != "all":
        simulation.settings = dataclasses.replace(
            simulation.settings, fields=fields, field_planes=(0, -1)
        )
    return simulation, schedule, length


def test_the_held_plane_is_charged_at_admission_when_fields_are_dropped():
    from solprop.driver.memory import admit_memory  # noqa: PLC0415
    from solprop.output.schema import FieldRetention  # noqa: PLC0415

    simulation, _, _ = held_soliton("planes")
    problem = simulation.translate()
    kept = FieldRetention("planes", (0, -1)).resolved(3)
    admission = admit_memory(problem, "numpy_ref", "none", retention=kept)
    assert admission.estimate.groups["held_plane"] == 16 * math.prod(problem.shape)
    every = FieldRetention("all", ()).resolved(3)
    groups = admit_memory(problem, "numpy_ref", "none", retention=every).estimate.groups
    assert "held_plane" not in groups


def test_a_field_free_plane_is_held_and_checkpointed_from_a_step_in_flight(tmp_path):
    """SEP 22, Q14: the state of the last field-free plane is held, one volume.

    A step abandoned in flight after a plane that kept no field leaves a terminal
    checkpoint at that plane, taken from the held state, and the run resumes from it.
    The ledger lists the held volume from the first field-free plane.
    """
    from solprop.output.hdf5 import open_file  # noqa: PLC0415

    plain, schedule, length = held_soliton()
    uninterrupted = plain.run(progress="none")
    kept, _, _ = held_soliton("planes")
    problem = kept.translate()
    path = tmp_path / "run.h5"
    original = NumpyRefBackend._adams_lab_source
    original_observe = NumpyRefBackend.observe
    count = [0]
    ledgers = []

    def observe(self, **keywords):
        observation = original_observe(self, **keywords)
        ledgers.append(self.memory_ledger()["held_plane"])
        return observation

    def evaluate(self, lab, position, **keywords):
        count[0] += 1
        if count[0] == 4 * 5 + 2:
            raise AbandonedError
        return original(self, lab, position, **keywords)

    try:
        NumpyRefBackend._adams_lab_source = evaluate
        NumpyRefBackend.observe = observe
        with pytest.raises(AbandonedError) as caught:
            kept.run(path, progress="none")
    finally:
        NumpyRefBackend._adams_lab_source = original
        NumpyRefBackend.observe = original_observe
    # The entrance kept its field and held nothing; the half plane held one volume.
    assert ledgers == [0, 16 * math.prod(problem.shape)]
    assert not any("no terminal checkpoint" in note for note in caught.value.__notes__)
    with open_file(path) as stored:
        checkpoint = stored.terminal_checkpoint
        assert checkpoint is not None and checkpoint.plane.field is not None
        assert checkpoint.plane.z == length / 2
        assert checkpoint.plane.step_history.n_steps == 0
        assert stored["z"].tolist() == [0.0, length / 2]
    kept.run(path, progress="none", resume=True)
    with open_file(path) as stored:
        assert stored["z"].tolist() == [0.0, length / 2, length]
        final = stored.plane_field(2)
    expected = uninterrupted["field"][-1]
    # As the resume from a kept plane above: the held state is the plane's exactly,
    # and the resumed half agrees within section 4.5's bound.
    bins = populated([expected], 1e-3).reshape(-1)
    bound = float(np.max(linear_bound(problem, schedule)[..., bins]))
    assert np.linalg.norm(final - expected) <= bound * np.linalg.norm(expected)


def test_a_held_plane_that_cannot_be_observed_keeps_the_original_failure(tmp_path):
    """The failure path must not replace the original exception (SEP 22, Q14).

    If the held state cannot be observed, the step in flight is recorded as abandoned,
    with no checkpoint, and the reason is noted on the original exception.
    """
    from solprop.output.hdf5 import open_file  # noqa: PLC0415

    kept, _, _ = held_soliton("planes")
    path = tmp_path / "run.h5"
    original = NumpyRefBackend._adams_lab_source
    original_held = NumpyRefBackend.held_observation
    count = [0]

    def evaluate(self, lab, position, **keywords):
        count[0] += 1
        if count[0] == 4 * 5 + 2:
            raise AbandonedError
        return original(self, lab, position, **keywords)

    def held(self):
        raise FloatingPointError("held field is not finite")

    try:
        NumpyRefBackend._adams_lab_source = evaluate
        NumpyRefBackend.held_observation = held
        with pytest.raises(AbandonedError) as caught:
            kept.run(path, progress="none")
    finally:
        NumpyRefBackend._adams_lab_source = original
        NumpyRefBackend.held_observation = original_held
    notes = caught.value.__notes__
    assert any("held plane could not be observed" in note for note in notes)
    with open_file(path) as stored:
        assert stored.terminal_checkpoint is None


# --- memory (section 7.3) ------------------------------------------------------------

# The Runge-Kutta march's own field-sized rows, empty under the low-storage march.
RUNGE_KUTTA_ROWS = ("stages", "accumulator_solution", "accumulator_error")

# The rows that scale with the whole state, as section 6.8 counts them.
FIELD_ROWS = (
    "state",
    "transverse_scratch",
    "register_accumulator",
    "register_stage",
    "propagators",
    "noise",
    "stages",
    "accumulator_solution",
    "accumulator_error",
)


def cartesian_march(mode, *, noise=None, collar=False):
    """Return a Cartesian Kerr problem under the march, node chunk 7 of 108.

    With the collars, the spatial and the temporal one in-stage, it takes the 100
    steps their stability limits ask for; otherwise 4.
    """
    length = 1e-3
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0.25 if collar else 0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    start = (
        pulse.build(grid, space, medium=SpatialMedium())
        if noise is None
        else np.zeros_like(noise.spectrum)
    )
    schedule = FixedSchedule.uniform(0.0, length, length / (100 if collar else 4))
    problem = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        noise=noise,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
        propagator_mode=mode,
        absorber=absorber(grid) if collar else None,
        absorber_scheme="nonlinear",
    )
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=7)
    )
    return problem, grid, space


def held_volumes(backend, field):
    """Return the bytes every state-sized array the backend and its engine hold owns.

    Each attribute of the backend and of the march's engine, and each array inside a
    tuple attribute, is followed to the memory it views: a broadcast view of one
    element owns one element, however large it reads.
    """
    found = {}
    owners = {"backend": backend, "engine": backend._low_storage.engine}
    for owner_name, owner in owners.items():
        for name, value in vars(owner).items():
            items = value if isinstance(value, tuple) else (value,)
            for item in items:
                if not isinstance(item, np.ndarray):
                    continue
                root = item
                while isinstance(root.base, np.ndarray):
                    root = root.base
                if root.nbytes >= field:
                    found[f"{owner_name}.{name}"] = root.nbytes
    return found


@pytest.mark.parametrize("collar", [False, True])
@pytest.mark.parametrize("mode", ["tabulate", "recompute"])
def test_the_ledger_holds_three_registers_the_scratch_and_the_table(mode, collar):
    """Five volumes tabulated, four recomputed; no stage, accumulator or pair (7.3).

    Beside the in-stage collars too: the march holds no row for the material output
    while the absorber term is evaluated (section 4.3). The rows are checked against
    the arrays the backend and its engine actually hold after the run: the state, the
    two registers and the table, and nothing else of a state's size; the transverse
    scratch ``S`` is a per-call allocation on this backend, counted at its peak.
    """
    problem, grid, _ = cartesian_march(mode, collar=collar)
    if collar:
        assert problem.transform.geometry is not None
        assert np.any(problem.transform.geometry.absorber_rate)
        assert problem.responses.temporal_absorber_fine is not None
    field = 16 * math.prod(problem.shape)
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(float(problem.stepper.save_planes[-1]))
    assert np.all(np.isfinite(backend.observe().require_field()))
    ledger = backend.memory_ledger()
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert isinstance(engine, HostLowStorage)
    assert engine.held() == {
        row: ledger[row]
        for row in ("register_accumulator", "register_stage", "propagators")
    }
    expected = {"backend._state", "engine._accumulator", "engine._stage"}
    if mode == "tabulate":
        expected.add("engine._table")
    assert held_volumes(backend, field) == dict.fromkeys(expected, field)
    backend.close()
    # Closing releases the registers and the table with the state.
    assert held_volumes(backend, field) == {}
    assert set(engine.held().values()) == {0}
    volumes = {row: ledger[row] for row in FIELD_ROWS}
    assert volumes["state"] == volumes["transverse_scratch"] == field
    assert volumes["register_accumulator"] == volumes["register_stage"] == field
    assert volumes["propagators"] == (field if mode == "tabulate" else 0)
    for row in ("stages", "accumulator_solution", "accumulator_error", "noise"):
        assert volumes[row] == 0, row
    assert sum(volumes.values()) == (5 if mode == "tabulate" else 4) * field
    # Every other row is a node chunk's, bounded by the chunk and not the state.
    assert all(size < field for row, size in ledger.items() if row not in FIELD_ROWS)
    # The construction-free estimate is the ledger to the byte.
    estimate = estimate_memory(problem, "numpy_ref")
    assert estimate.groups == {
        name: size for name, size in ledger_groups(ledger).items() if name != "measured"
    }


def test_the_noise_adds_one_volume_and_the_estimate_names_it():
    noise = spatial_noise(
        SpatialFlux.resolve(
            EnvelopeGrid.from_limits(
                1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
            ),
            FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0),
            SpatialMedium(),
        ),
        np.random.default_rng(7),
        model="modified",
    )
    problem, grid, _ = cartesian_march("tabulate", noise=noise)
    field = 16 * math.prod(problem.shape)
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    ledger = backend.memory_ledger()
    backend.close()
    assert sum(ledger[row] for row in FIELD_ROWS) == 6 * field
    estimate = estimate_memory(problem, "numpy_ref")
    assert ledger_groups(ledger)["base"] == estimate.groups["base"] + field
    assert any("noise field" in item for item in estimate.not_counted)


def test_a_run_too_large_for_the_adams_history_fits_under_the_march(
    soliton_problem, narrow_grid
):
    """``_admit_history`` refuses the default integrator and admits the march (7.3)."""
    from solprop.driver.memory import MemoryLimit, _admit_history  # noqa: PLC0415
    from solprop.exceptions import MemoryBudgetError  # noqa: PLC0415
    from solprop.stepper import Multistep  # noqa: PLC0415

    length = 0.1
    schedule = FixedSchedule.uniform(0.0, length, length / 10)
    adams = estimate_memory(
        soliton_problem(
            narrow_grid,
            1,
            save_planes=[length],
            schedule=schedule,
            integrator=Multistep(),
        ),
        "numpy_ref",
    )
    march = estimate_memory(
        soliton_problem(
            narrow_grid, 1, save_planes=[length], schedule=schedule, integrator=MARCH
        ),
        "numpy_ref",
    )
    assert march.groups["adams_history"] == 0
    limit = MemoryLimit(adams.total - 1, "warn", "machine")
    with pytest.raises(MemoryBudgetError, match="RungeKutta"):
        _admit_history(adams, limit)
    assert limit.limit is not None and march.total <= limit.limit
    _admit_history(march, limit)


# --- the noise (section 7.1, Q3) ----------------------------------------------------


def test_the_noise_draw_never_enters_the_propagating_field():
    """With the draw and no response the state is the pure flow of its input."""
    length = 1e-3
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    noise = spatial_noise(
        SpatialFlux.resolve(grid, space, SpatialMedium()),
        np.random.default_rng(3),
        model="modified",
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    schedule = FixedSchedule.uniform(0.0, length, length / 8)
    problem = assemble(
        grid,
        None,
        [],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        noise=noise,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
    )
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, noise)
    backend.propagate_to(length)
    final = backend.observe().require_field()
    backend.close()
    assert_linear_limit(problem, final, schedule)


def test_each_pass_carries_the_draw_at_its_own_position(monkeypatch):
    """The draw is advanced to z_n, z_m, z_m and z_n + h, against a hand-formed one.

    The phase is formed independently, the vacuum's ``z k_perp^2 / (k_z + k)`` on
    every active bin, as ``test_free_noise`` forms it; the bound is that test's.
    """
    length = 1e-4
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False
    )
    space = FreeGrid(3e-4, 9, 4e-4, 12, window_factor=0)
    noise = spatial_noise(
        SpatialFlux.resolve(grid, space, SpatialMedium()),
        np.random.default_rng(94),
        model="modified",
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6
    )
    schedule = FixedSchedule.uniform(0.0, length, length / 2)
    problem = assemble(
        grid,
        None,
        [KerrResponse(1e-22, thg=False)],
        pulse.build(grid, space, medium=SpatialMedium()),
        spatial_grid=space,
        medium=SpatialMedium(),
        noise=noise,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        schedule=schedule,
        integrator=MARCH,
    )
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, noise)
    inputs, carried = [], []
    source, nonlinear = backend._adams_lab_source, backend._nonlinear

    def lab_source(lab, position, **keywords):
        inputs.append((position, lab.copy()))
        return source(lab, position, **keywords)

    def capture(lab, bundle, inverse):
        carried.append(lab.copy())
        return nonlinear(lab, bundle, inverse)

    monkeypatch.setattr(backend, "_adams_lab_source", lab_source)
    monkeypatch.setattr(backend, "_nonlinear", capture)
    backend.propagate_to(length)
    backend.close()
    z, h = float(schedule.edges[1]), float(schedule.steps[1])
    positions = [position for position, _ in inputs[4:]]
    assert positions == [z, z + 0.5 * h, z + 0.5 * h, z + h]
    wave = grid.omega[grid.active_mask] / C
    root = np.sqrt(wave[None, :] ** 2 - space.kperp_sq[:, None])
    for (position, lab), field_in in zip(inputs[4:], carried[4:], strict=True):
        phase = position * space.kperp_sq[:, None] / (root + wave[None, :])
        expected = np.zeros_like(noise.spectrum)
        expected[..., grid.active_mask] = noise.spectrum[
            ..., grid.active_mask
        ] * np.exp(1j * phase[:, None, :])
        bound = gamma(128 * (space.n_columns + grid.n_to))
        bound += gamma(64) * position * float(grid.omega.max()) / C
        drawn = (field_in - lab)[0]
        scale = np.max(np.abs(expected))
        assert np.max(np.abs(drawn - expected)) <= bound * scale


# --- records, flows and the stiffness indicator --------------------------------------


@pytest.mark.parametrize("backend", ["auto", "numpy_ref"])
def test_the_run_records_the_march_and_no_estimate(caplog, backend):
    """The method names the march, the norms are 0.0, the logs say why (section 6.3).

    ``"auto"`` resolves as for every integrator (phase 3): ``native`` when installed.
    """
    from solprop.driver import Simulation  # noqa: PLC0415

    length = 0.01
    schedule = FixedSchedule.uniform(0.0, length, length / 5)
    simulation = Simulation.from_gnlse(
        0.11,
        length,
        [0.0, 0.0, -11.83e-27],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        pulseshape="sech",
        n_save=2,
        schedule=schedule,
        integrator=MARCH,
        backend=backend,
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        # No collar: a few fixed steps are refused beside the default one.
        absorber=False,
    )
    assert simulation.settings.backend == resolve_name(backend)
    # The package logger's own level too: a doctest may have left it raised.
    with caplog.at_level(logging.DEBUG, logger="solprop"):
        output = simulation.run(progress="log")
    metadata = output.metadata
    assert metadata is not None
    assert metadata.stepper.method == "LowStorageRungeKutta(scheme='rk4ip')"
    assert metadata.stepper.tableau == "DOPRI5"
    norms = output["step_history/norm"]
    assert norms.shape == (5,) and np.all(norms == 0.0)
    messages = [record.getMessage() for record in caplog.records]
    ratio_lines = [line for line in messages if "tolerance ratio" in line]
    assert ratio_lines
    assert all("not estimated (replayed schedule)" in line for line in ratio_lines)
    assert not any("too coarse" in line for line in messages)
    assert isinstance(output, MemoryOutput)
    assert output.outcome is not None
    assert output.outcome.execution["source_evaluations"] == 20
    assert output.outcome.execution["propagator_mode"] == "tabulate"


@pytest.mark.parametrize("problem_kind", ["soliton", "cartesian"])
def test_the_two_flow_treatments_agree_to_rounding(
    lifecycle, narrow_grid, scales, problem_kind
):
    """The table and the per-finish flow agree within a rounding bound (section 4.3)."""
    if problem_kind == "soliton":
        problem, _ = lifecycle()
        grid = narrow_grid
        bound = rounding_bound(problem, problem.stepper.schedule, scales)
    else:
        problem, grid, _ = cartesian_march("tabulate")
        # Four steps of a weakly nonlinear beam: section 4.5's bound without growth.
        bins = populated([problem.initial.field0], 1e-3)
        bound = float(np.max(linear_bound(problem, problem.stepper.schedule)[bins]))
    fields = {}
    for mode in ("tabulate", "recompute"):
        stepper = dataclasses.replace(problem.stepper, propagator_mode=mode)
        fields[mode], backend = run(dataclasses.replace(problem, stepper=stepper), grid)
        backend.close()
    reference = fields["recompute"]
    assert np.linalg.norm(fields["tabulate"] - reference) <= bound * np.linalg.norm(
        reference
    )


def test_the_table_is_keyed_on_its_run_and_rebuilt_where_the_run_changes(
    lifecycle, narrow_grid
):
    """One table per run of equal steps; a discovered schedule rebuilds it (Q10).

    A uniform schedule's steps are differences of rounded positions and differ by up
    to an ulp of the *position*, many ulps of the step far from the origin; keyed on
    the run's mean step (``table_steps``) the table is built once per run, where a key
    on each step's own value within ulps of the step was rebuilt at most steps.
    """
    problem, length = lifecycle()
    schedule = problem.stepper.schedule
    assert schedule is not None
    keys = table_steps(schedule)
    # The segments cut at the planes have the same nominal step: one run, one key, and
    # its steps are not all equal.
    assert len(set(keys.tolist())) == 1
    assert len(set(schedule.steps.tolist())) > 1
    assert abs(float(keys[0]) - length / LIFECYCLE_STEPS) <= 4 * EPS * length
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert isinstance(engine, HostLowStorage)
    backend.set_initial(problem.initial.field0, None)
    built = []
    propagator = backend._adams_propagator

    def counted(anchor, position):
        built.append((anchor, position))
        return propagator(anchor, position)

    backend._adams_propagator = counted
    backend.propagate_to(length / 2)
    backend.propagate_to(length)
    assert len(built) == 1
    assert engine._table_step == float(keys[0])
    backend.close()
    # Steps that all differ have runs of one: the table is rebuilt every step.
    steps = np.geomspace(1.0, 2.0, 12)
    varied_schedule = FixedSchedule(steps / steps.sum() * length)
    assert np.array_equal(table_steps(varied_schedule), varied_schedule.steps)
    varied = marched(problem, varied_schedule, step_planes=0)
    varied = dataclasses.replace(
        varied, stepper=dataclasses.replace(varied.stepper, save_planes=[length])
    )
    backend = NumpyRefBackend(varied, grid=narrow_grid)
    backend.set_initial(varied.initial.field0, None)
    built.clear()
    propagator = backend._adams_propagator
    backend._adams_propagator = counted
    backend.propagate_to(length)
    assert len(built) == varied_schedule.n_steps
    # A load drops the table, which the next step builds again.
    backend.set_initial(varied.initial.field0, None)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert isinstance(engine, HostLowStorage) and engine._table is None
    backend.close()


def test_a_run_s_key_is_its_mean_step_and_far_runs_keep_one_key():
    """Far from the origin a uniform segment's steps spread over many step ulps."""
    schedule = FixedSchedule.uniform(10.0, 10.3, 1e-4)
    spread = np.ptp(schedule.steps) / np.spacing(schedule.steps[0])
    assert spread > LANDING_ULPS
    keys = table_steps(schedule)
    assert len(set(keys.tolist())) == 1
    # The key is the run's mean: the run's steps sum to its keys to rounding.
    assert abs(float(np.sum(keys) - np.sum(schedule.steps))) <= 4 * np.spacing(10.3)
    assert schedule_key(keys, schedule.edges, 10.0, float(schedule.steps[0])) == float(
        keys[0]
    )
    # Two segments of different nominal steps are two runs.
    keys = table_steps(FixedSchedule.uniform(0.0, 1.0, 0.1, save_planes=[0.25, 1.0]))
    assert len(set(keys.tolist())) == 2


class LinearSource(StepDriver):
    """A stand-in backend whose source is ``lambda u`` and whose flow is one.

    Built without the driver's initialiser, which needs a translated problem: the host
    engine reads only the members set here.
    """

    def __init__(self, rate, state):
        self.rate = rate
        self._state = state
        self._low_storage = None
        self._problem = SimpleNamespace(
            stepper=SimpleNamespace(propagator_mode="recompute"),
            flags=SimpleNamespace(constant_linop=True),
            grids=SimpleNamespace(norm_mask=np.ones(state.shape[-1], dtype=bool)),
        )

    def _trial_step(self, z, dz):
        raise NotImplementedError

    def _apply_mask(self, state):
        raise NotImplementedError

    def stats(self, spec):
        raise NotImplementedError

    def _hoist(self, z0):
        del z0

    def _stage_bundles(self, z0, positions):
        del z0, positions
        return ()

    def _adams_propagator(self, anchor, position):
        del anchor, position
        return np.ones_like(self._state)

    def _adams_lab_source(self, lab, position, *, purpose="predictor"):
        del position, purpose
        return self.rate * lab


@pytest.mark.parametrize("rate", [-3.0, -40.0 + 0j, 25j, -10.0 + 10.0j])
def test_a_linear_step_is_rk4_s_stability_function_and_reads_its_stiffness(rate):
    """For ``u' = lambda u`` a step is ``R(h lambda)`` and ``h rho = |h lambda|``.

    RK4's stability function ``1 + x + x^2/2 + x^3/6 + x^4/24`` is the scheme itself on
    a linear source, and the indicator of section 6.3 is exact there; both to a few
    roundings of the passes.
    """
    rng = np.random.default_rng(5)
    state = rng.standard_normal((1, 1, 1, 64)) + 1j * rng.standard_normal((1, 1, 1, 64))
    start = state.copy()
    step = 0.05
    backend = LinearSource(rate, state)
    engine = HostLowStorage(backend, MARCH)
    squared = engine.step(0.0, step)
    x = step * rate
    expected = (1 + x + x**2 / 2 + x**3 / 6 + x**4 / 24) * start
    assert np.allclose(state, expected, rtol=64 * EPS, atol=0)
    assert squared == pytest.approx(float(np.vdot(expected, expected).real), rel=1e-13)
    assert engine.stiffness == pytest.approx(abs(x), rel=64 * EPS * 1e3)
    engine.reset()
    assert math.isnan(engine.stiffness)


def test_the_march_warns_once_when_a_step_passes_the_interval(
    lifecycle, narrow_grid, caplog
):
    """A stiffness past 2.785 is logged once per replay, never in the norm column."""
    problem, length = lifecycle()
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    assert backend._low_storage is not None
    engine = backend._low_storage.engine
    assert isinstance(engine, HostLowStorage)
    step = engine.step

    def stiff(z, dz):
        squared = step(z, dz)
        engine.stiffness = 3.0
        return squared

    engine.step = stiff
    with caplog.at_level(logging.WARNING, logger="solprop.backends.low_storage"):
        backend.propagate_to(length / 2)
    history = backend.observe().step_history
    backend.close()
    warnings_logged = [
        record.getMessage()
        for record in caplog.records
        if "stiffness indicator" in record.getMessage()
    ]
    assert len(warnings_logged) == 1
    assert "h rho = 3 passed RK4's stability interval 2.785" in warnings_logged[0]
    assert np.all(history.error_norm == 0.0)


def test_a_replay_that_runs_out_says_so(lifecycle, narrow_grid):
    problem, length = lifecycle()
    backend = NumpyRefBackend(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    with pytest.raises(BackendError, match="cannot reach"):
        backend.propagate_to(length * 1.02)
    # Every step was taken and committed: nothing is left in flight.
    assert backend.observe().z == length
    backend.close()


def test_a_failure_record_that_cannot_be_written_keeps_the_original_error(caplog):
    """The in-flight failure path never replaces the run's own exception."""
    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.exceptions import DriverError  # noqa: PLC0415

    class Unwritable(MemoryOutput):
        def record_outcome(self, outcome, checkpoint=None):
            if outcome.status == "failed":
                raise DriverError("manufactured record failure")
            super().record_outcome(outcome, checkpoint)

    length = 0.01
    schedule = FixedSchedule.uniform(0.0, length, length / 4)
    simulation = Simulation.from_gnlse(
        0.11,
        length,
        [0.0, 0.0, -11.83e-27],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        pulseshape="sech",
        n_save=2,
        schedule=schedule,
        integrator=MARCH,
        backend="numpy_ref",
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        # No collar: a few fixed steps are refused beside the default one.
        absorber=False,
    )
    original = NumpyRefBackend._adams_lab_source
    count = [0]

    def evaluate(self, lab, position, **keywords):
        count[0] += 1
        if count[0] == 3:
            raise AbandonedError
        return original(self, lab, position, **keywords)

    try:
        NumpyRefBackend._adams_lab_source = evaluate
        with pytest.raises(AbandonedError) as caught:
            simulation.run(Unwritable(), progress="none")
    finally:
        NumpyRefBackend._adams_lab_source = original
    notes = caught.value.__notes__
    assert any("no terminal checkpoint" in note for note in notes)
    assert any("manufactured record failure" in note for note in notes)
    assert "could not record the failure" in caplog.text


def test_the_held_plane_reuses_its_buffer():
    """A second field-free plane copies into the volume the first allocated."""
    problem, grid, _ = cartesian_march("tabulate")
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        state = backend._state
        assert state is not None
        backend._hold(state)
        first = backend._held
        backend._hold(None)
        backend._hold(2.0 * state)
        assert backend._held is first
        np.testing.assert_array_equal(
            backend._held_field(), backend._observed_field(2.0 * state)
        )
    finally:
        backend.close()
