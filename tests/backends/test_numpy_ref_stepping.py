# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How the step loop lands, replays and converges.

``docs/dev/stepper.md`` section 3: the controller's proposal is its state, a landing
never restarts it from a sliver, and a replayed schedule still scores every step so that
one too coarse for its tolerance is reported. Plus the two checks that the stage passes
compose into the scheme they claim: a single step against an independent integrator, and
the same Kerr propagation on both grids.
"""

import logging
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from solprop import fourier
from solprop.exceptions import BackendError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.linear import Frame, dispersion_operator
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.normalisation import mode_average_prefactor, sqrtw_to_vm
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.stepper import FixedSchedule, PIController, observed_order
from solprop.translate.problem import assemble

SPACING = 1e-4  # m between save planes in the stretch test
N_PLANES = 200
# The sliver test needs planes several natural steps apart, so that landing really
# does clip and the step after a plane has something to be compared with.
WIDE_SPACING = 2e-3  # m


def run(backend_class, problem, grid, planes):
    """Propagate through a list of planes and return the summaries and observations."""
    backend = backend_class(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    summaries, observations = [], []
    for plane in planes:
        summaries.append(backend.propagate_to(plane))
        observations.append(backend.observe())
    backend.close()
    return summaries, observations


# ------------------------------------------------------------------ the landing rules


def test_a_march_over_many_planes_takes_one_step_each(
    backend_class, linear_problem, narrow_grid
):
    """The stretch: a plane just beyond the proposal is landed on, not overshot and
    then reached by a sliver."""
    planes = [SPACING * (index + 1) for index in range(N_PLANES)]
    problem = linear_problem(
        narrow_grid,
        save_planes=planes,
        # A growth cap that barely moves pins the proposal just short of the spacing,
        # which is the configuration the stretch exists for.
        controller=PIController(rtol=1e-8, atol=1e-18, max_factor=1.0 + 1e-7),
        dz_init=0.98 * SPACING,
        dz_max=10 * SPACING,
    )
    summaries, observations = run(backend_class, problem, narrow_grid, planes)
    steps = np.concatenate([plane.step_history.dz for plane in observations])
    assert sum(summary.n_steps for summary in summaries) == N_PLANES
    # Every step is the stretched one, so none is a sliver: the shortest is the spacing.
    assert float(np.min(steps)) == pytest.approx(SPACING, rel=1e-12)


def test_the_step_after_a_plane_is_the_proposal_not_a_sliver(
    backend_class, soliton_problem, narrow_grid
):
    """The equivalent-norm rule: the controller's state is the proposal, so a landing
    does not restart it from the sliver it had to take."""
    planes = [WIDE_SPACING * (index + 1) for index in range(20)]
    problem = soliton_problem(
        narrow_grid,
        3,
        save_planes=planes,
        controller=PIController(rtol=1e-10, atol=1e-18),
        dz_max=10 * WIDE_SPACING,
        dz_init=1e-4,
    )
    _, observations = run(backend_class, problem, narrow_grid, planes)
    # Skip the early planes, over which the controller is still growing from dz_init.
    # Each plane's first step against that plane's own steps: the soliton's spectrum
    # reaches the scored margin as it goes, so the steps shrink along the run.
    later = observations[8:]
    for plane in later:
        steps = np.asarray(plane.step_history.dz, dtype=float)
        assert steps[0] > 0.8 * float(np.median(steps))


def test_a_plane_is_landed_on_exactly(backend_class, soliton_problem, narrow_grid):
    """Saved fields are exact step endpoints; there is no dense output."""
    planes = [0.1 / 3, 0.1 / 2, 0.1]
    problem = soliton_problem(
        narrow_grid, 1, save_planes=planes, dz_max=0.004, dz_init=1e-4
    )
    _, observations = run(backend_class, problem, narrow_grid, planes)
    for plane, observation in zip(planes, observations, strict=True):
        assert observation.z == plane
        assert float(observation.step_history.z[-1]) == plane


def test_a_step_lands_on_a_hard_kink(backend_class, capillary_problem, field_grid):
    """The fill's stage boundary is a kink, and a step must end exactly on it."""
    problem = capillary_problem()
    _, observations = run(backend_class, problem, field_grid, [0.2])
    assert 0.2 in observations[0].step_history.z.tolist()


# ------------------------------------------------------------------ replayed schedules


def test_a_schedule_fine_enough_meets_its_tolerance(
    backend_class, soliton_problem, soliton_grid, caplog
):
    problem = soliton_problem(
        soliton_grid,
        1,
        save_planes=[0.05],
        controller=PIController(rtol=1e-8, atol=1e-18),
        schedule=FixedSchedule.uniform(0.0, 0.05, 0.05 / 200),
    )
    with caplog.at_level(logging.WARNING):
        summaries, observations = run(backend_class, problem, soliton_grid, [0.05])
    assert summaries[0].n_steps == 200
    assert summaries[0].met_tolerance
    assert not caplog.records
    assert np.all(observations[0].step_history.accepted)


def test_a_schedule_too_coarse_is_reported_rather_than_silent(
    backend_class, soliton_problem, soliton_grid, caplog
):
    """The fixed-mode counterpart of the dz_min error: the estimate costs nothing."""
    problem = soliton_problem(
        soliton_grid,
        1,
        save_planes=[0.05],
        # The schedule is perfectly reasonable and the tolerance asked of it is not,
        # which is the case the report exists for.
        controller=PIController(rtol=1e-14, atol=1e-30),
        schedule=FixedSchedule.uniform(0.0, 0.05, 0.05 / 8),
    )
    with caplog.at_level(logging.WARNING):
        summaries, observations = run(backend_class, problem, soliton_grid, [0.05])
    assert summaries[0].n_steps == 8
    assert not summaries[0].met_tolerance
    assert summaries[0].max_norm > 1.0
    assert any("too coarse" in record.message for record in caplog.records)
    # Every step is still recorded with its norm, which is what makes the report.
    assert np.all(observations[0].step_history.error_norm > 0.0)


def test_a_schedule_that_runs_out_is_an_error(
    backend_class, soliton_problem, soliton_grid
):
    problem = soliton_problem(
        soliton_grid,
        1,
        save_planes=[0.05],
        schedule=FixedSchedule.uniform(0.0, 0.05, 0.05 / 4),
    )
    backend = backend_class(problem, grid=soliton_grid)
    backend.set_initial(problem.initial.field0, None)
    with pytest.raises(BackendError, match="cannot reach"):
        backend.propagate_to(0.052)
    backend.close()


# ---------------------------------------------------- one step against an oracle


SPM_GRID = {
    "length": 1.0,
    "reference_wavelength": 1030e-9,
    "wavelength_limits": (700e-9, 2.0e-6),
    "trange": 2e-12,
}
SPM_GAMMA = 0.01  # 1/(W m)
SPM_DISTANCE = 0.2  # m


def spm_problem(grid, *, distance=SPM_DISTANCE, **overrides):
    """A dispersionless Kerr fibre, whose solution is a pure self-phase modulation."""
    fibre = problem_from(
        [0.0, 0.0, 0.0], omega_ref=grid.omega0, gamma=SPM_GAMMA, shock=False
    )
    start = GaussianPulse(wavelength=1030e-9, fwhm=120e-15, energy=9e-11).build(grid)
    settings = {
        "controller": PIController(rtol=1e-12, atol=1e-30),
        "dz_max": distance / 20,
        "dz_init": distance / 200,
    } | overrides
    return fibre, assemble(
        grid,
        fibre.mode,
        [KerrResponse(chi3=fibre.kerr.chi3, thg=False)],
        start,
        save_planes=[distance],
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        **settings,
    )


def envelope_of(state, grid, kind):
    """Return the complex envelope of a state, on either grid."""
    samples = fourier.inverse(state, kind=kind)
    if kind == "envelope":
        return samples
    return fourier.analytic_signal(samples)


def spm_phase(backend_class, problem, grid, kind, distance):
    """Return the measured nonlinear phase, the input power and where the pulse is.

    The phase is read only where the pulse actually is: the ratio of two envelopes is
    meaningless in a bin the pulse never occupied, and a nonlinear phase measured there
    would be the round-off of one empty sample divided by another.
    """
    _, observations = run(backend_class, problem, grid, [distance])
    start = envelope_of(
        problem.initial.field0[0, 0, 0] * problem.grids.active_mask, grid, kind
    )
    end = envelope_of(observations[0].field[0, 0, 0], grid, kind)
    power = np.abs(start) ** 2
    core = power > 1e-3 * np.max(power)
    measured = np.zeros_like(power)
    measured[core] = np.angle(end[core] / start[core])
    return measured, power, core


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_self_phase_modulation_is_reproduced_on_either_grid(backend_class, kind):
    """Parity: the two grids are different discretisations of one physical answer.

    With no dispersion the exact solution is ``A(z) = A(0) exp(-i gamma |A|^2 z)`` in
    the internal convention, on the envelope grid for ``A`` and on the field grid for
    the analytic signal of the real field, which is the same object with its carrier
    left in (``conventions.md`` section 4).
    """
    grid = (
        EnvelopeGrid.from_limits(**SPM_GRID, thg=False)
        if kind == "envelope"
        else FieldGrid.from_limits(**SPM_GRID)
    )
    _, problem = spm_problem(grid)
    measured, power, core = spm_phase(backend_class, problem, grid, kind, SPM_DISTANCE)
    expected = -SPM_GAMMA * power * SPM_DISTANCE
    peak = float(np.max(np.abs(expected)))
    # The measured phase is compared as a fraction of the largest nonlinear phase in the
    # pulse; away from the peak the phase is small and a relative measure there would be
    # dominated by the wings' own discretisation. The pulse energy is chosen to keep the
    # peak phase below pi, so that no sample needs unwrapping.
    assert 1.0 < peak < math.pi
    assert np.max(np.abs(measured[core] - expected[core])) / peak < 1e-8


def test_the_two_grids_agree_on_the_peak_nonlinear_phase(backend_class):
    """The peak sits at ``t = 0``, a sample on both grids, so it is comparable without
    interpolating between two different time axes."""
    phases = {}
    for kind in ("envelope", "field"):
        grid = (
            EnvelopeGrid.from_limits(**SPM_GRID, thg=False)
            if kind == "envelope"
            else FieldGrid.from_limits(**SPM_GRID)
        )
        _, problem = spm_problem(grid)
        measured, _, _ = spm_phase(backend_class, problem, grid, kind, SPM_DISTANCE)
        phases[kind] = float(measured[grid.n_t // 2])
    assert phases["field"] == pytest.approx(phases["envelope"], rel=1e-8)


def reference_step(fibre, grid, problem, state, dz):
    """Integrate one step with an independent high-order solver.

    The right-hand side is composed from the physics layer the way
    ``solprop.driver.scaffold`` composes it -- oversample, transform, convert to volts
    per metre, evaluate the response, transform back, crop, multiply by the prefactor --
    so it shares no code with the backend's passes beyond the response itself.
    """
    operator = dispersion_operator(fibre.mode, grid, Frame.comoving(grid))
    prefactor = mode_average_prefactor(
        fibre.mode, grid, shock=False, index=fibre.prefactor_index
    )
    to_field = sqrtw_to_vm(fibre.aeff)
    mask = problem.grids.active_mask
    n_t, n_to = grid.n_t, grid.n_to

    def rhs(_z, packed):
        spectrum = (packed[: packed.size // 2] + 1j * packed[packed.size // 2 :]) * mask
        fine = fourier.oversample(spectrum, n_to, kind="envelope")
        envelope = fourier.inverse(fine, kind="envelope") * to_field
        polarisation = fibre.kerr.evaluate(envelope[np.newaxis], 1.0, kind="envelope")
        transformed = fourier.forward(polarisation[0], kind="envelope")
        derivative = operator * spectrum + prefactor * fourier.crop(
            transformed, n_t, kind="envelope"
        )
        return np.concatenate([derivative.real, derivative.imag])

    packed = np.concatenate([state.real, state.imag])
    solution = solve_ivp(
        rhs,
        (0.0, dz),
        packed,
        method="DOP853",
        rtol=1e-13,
        atol=1e-14 * float(np.max(np.abs(state))),
        dense_output=False,
    )
    final = solution.y[:, -1]
    return final[: final.size // 2] + 1j * final[final.size // 2 :]


def test_one_step_matches_an_independent_integrator_at_the_scheme_s_order(
    backend_class,
):
    """The local error of one step falls as ``h**6``, which is what an order-5 pair
    promises and what no amount of agreement at a single step size would show."""
    # 256 bins over a 400 fs window: small enough that a high-order reference is
    # cheap, wide enough that the pulse is not chopped by the temporal collar.
    grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-13)
    fibre = problem_from(
        [0.0, 0.0, -11.83e-27], omega_ref=grid.omega0, gamma=0.11, shock=False
    )
    start = GaussianPulse(wavelength=835e-9, fwhm=20e-15, energy=5e-11).build(grid)
    # Coarse enough that even base/4 leaves a local error well above the reference's
    # own 1e-13, and fine enough that the coarsest rung is already asymptotic.
    base = 5e-4  # m
    errors = []
    for level in range(3):
        dz = base / 2**level
        problem = assemble(
            grid,
            fibre.mode,
            [fibre.kerr],
            start,
            controller=PIController(rtol=1e-6, atol=1e-30),
            save_planes=[dz],
            dz_max=dz,
            dz_init=dz,
            schedule=FixedSchedule(np.array([dz])),
            density=1.0,
            shock=False,
            prefactor_index=1.0,
        )
        _, observations = run(backend_class, problem, grid, [dz])
        masked = problem.initial.field0[0, 0, 0] * problem.grids.active_mask
        oracle = reference_step(fibre, grid, problem, masked, dz)
        errors.append(float(np.max(np.abs(observations[0].field[0, 0, 0] - oracle))))
    scale = float(np.max(np.abs(start)))
    orders = [
        observed_order(errors[index], errors[index + 1])
        for index in range(len(errors) - 1)
    ]
    assert errors[0] / scale < 1e-6
    # An order-5 method has a local error of order h**6. Measured relative errors
    # 1.34e-8, 1.23e-10 and 1.45e-12 at 5e-4, 2.5e-4 and 1.25e-4 m, so orders 6.76 and
    # 6.41 -- above six because the leading term has not quite taken over at the top of
    # the ladder. The gate of testing.md section 1 rejects anything above p + 3; below,
    # an order under 5 would mean the stage combination is wrong rather than inaccurate.
    assert all(5.0 < order < 8.0 for order in orders), orders
