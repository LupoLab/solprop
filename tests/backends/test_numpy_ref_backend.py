# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the reference backend refuses, holds and reports.

The seam of ``docs/dev/backends.md`` section 1, from the outside: the protocol, the
refusals that stop a declared term from being silently dropped, the memory ledger, the
noise, the statistics and the two propagator modes.
"""

import dataclasses
import logging

import numpy as np
import pytest

from bounds import EPS, assert_within, departure, gamma
from solprop.backends import Backend
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref import backend as backend_module
from solprop.exceptions import BackendError, TranslationError
from solprop.noise import modified_noise
from solprop.output.schema import SavedPlane
from solprop.stats import StatsSpec, collect
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.linop import unitary_phase
from solprop.translate.problem import SCHEMA_VERSION
from solprop.translate.responses import prefactor_at

LENGTH = 0.01  # m

SCHEDULE = FixedSchedule.uniform(0.0, LENGTH, LENGTH / 8)
"""Eight replayed steps to :data:`LENGTH`, for two runs that take the same steps."""

REPEAT_BOUND = 8 * 16 * EPS
"""The eight-step rung's budget (``docs/dev/accuracy.md`` section 7): two runs of one
method on :data:`SCHEDULE` differ by rounding alone (SEP 25)."""

STATISTICS_BOUND = 16 * 1024 * EPS
"""Two statistics evaluations of one state: two transforms and reductions over the
narrow grid's 1024 samples and the FWHM interpolation, 16 n eps (the Kokkos row of
:func:`test_statistics_reproduce_the_reference`)."""


def run_to(backend_class, problem, grid, z, **options):
    """Propagate a problem to a plane and return the backend and the observation."""
    backend = backend_class(problem, grid=grid, **options)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(z)
    return backend, backend.observe()


# ------------------------------------------------------------------- the protocol


def test_the_backend_satisfies_the_seam(
    backend_class, backend_name, linear_problem, narrow_grid
):
    problem = linear_problem(narrow_grid, save_planes=[LENGTH])
    backend = backend_class(problem)
    assert isinstance(backend, Backend)
    assert (backend.name, backend.max_batch) == (backend_name, 1)
    backend.close()


def test_the_observation_is_the_state_in_the_fft_convention(
    backend_class, soliton_problem, narrow_grid
):
    """Nothing conjugates behind the seam, so the initial plane is the input itself."""
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    plane = backend.observe()
    expected = problem.initial.field0 * problem.grids.active_mask
    assert np.array_equal(plane.field, expected)
    assert plane.z == 0.0
    assert plane.step_history.n_steps == 0
    assert plane.noise_phase is None
    backend.close()


def test_a_plane_already_reached_costs_no_steps(
    backend_class, soliton_problem, narrow_grid
):
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(0.0)
    assert (summary.n_steps, summary.max_norm, summary.met_tolerance) == (0, 0.0, True)
    backend.close()


def test_the_history_covers_only_what_happened_since_the_last_observation(
    backend_class, soliton_problem, narrow_grid
):
    problem = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH, 2 * LENGTH], dz_max=LENGTH / 3
    )
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(LENGTH)
    first = backend.observe()
    backend.propagate_to(2 * LENGTH)
    second = backend.observe()
    assert first.step_history.n_steps > 0
    assert second.step_history.n_steps > 0
    assert float(first.step_history.z[-1]) == LENGTH
    assert float(second.step_history.z[0]) > LENGTH
    backend.close()


# --------------------------------------------------------------------- the refusals


def declare(problem, substitute, **overrides):
    """Return the problem with one or more flags declared, as a compiler would."""
    return substitute(
        problem,
        flags=dataclasses.replace(problem.flags, **overrides),
        validate=False,
    )


def test_a_declared_term_this_backend_cannot_execute_is_refused_by_name(
    backend_class, soliton_problem, narrow_grid, substitute
):
    """A term silently dropped would run, converge and be wrong. The Kerr, Raman and
    plasma terms execute; the spectral-domain responses do not."""
    problem = declare(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH]),
        substitute,
        has_spectral_responses=True,
    )
    with pytest.raises(BackendError, match="spectral responses"):
        backend_class(problem)


def test_a_declared_raman_kernel_is_executed_rather_than_refused(
    backend_class, backend_name, linear_problem, narrow_grid
):
    """Construct a compiled Raman term; a flag without a leaf is not executable."""
    from solprop.nonlinear.raman import response_for  # noqa: PLC0415

    problem = linear_problem(
        narrow_grid,
        save_planes=[LENGTH],
        responses=[response_for("SiO2", amplitude=0.0)],
    )
    assert problem.flags.has_raman
    backend = backend_class(problem)
    assert backend.name == backend_name
    backend.close()


@pytest.mark.parametrize("declared", [2, 3, SCHEMA_VERSION - 1, 99])
def test_an_unknown_contract_version_is_refused_before_validation(
    backend_class, soliton_problem, narrow_grid, declared
):
    """The version is checked first: validate assumes the version it knows.

    The previous version is refused too: the check is an exact match, so a consumer of
    the previous version equally refuses a problem of this one.

    Version 2 is refused by name and not merely unrecognised: its section 3 and section
    5 carry z tables where version 3 carries expansions, and its Raman rows are three
    fields rather than a leaf, so a backend that read it would read the wrong things.
    """
    problem = dataclasses.replace(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH]), schema_version=declared
    )
    with pytest.raises(
        BackendError,
        match=f"version {SCHEMA_VERSION} and the problem declares version {declared}",
    ):
        backend_class(problem)


def test_a_transform_and_a_state_cannot_both_carry_the_components(
    backend_class, soliton_problem, narrow_grid, substitute
):
    """The components live in the transform's rows or in the state, never in both."""
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    transform = dataclasses.replace(
        problem.transform,
        synthesis=np.ones((2, 1)),
        projection=np.ones((1, 2)),
        projection_coarse=np.ones((1, 2)),
        node_chunk=1,
        _n_pol=2,
        kind="modal",
    )
    spoiled = substitute(problem, transform=transform, validate=False)
    object.__setattr__(spoiled, "shape", (1, 1, 2, narrow_grid.n_omega))
    with pytest.raises(BackendError, match=r"belong to one of the two"):
        backend_class(spoiled)


def test_a_batch_larger_than_one_is_refused(backend_class, batched_problem):
    with pytest.raises(BackendError, match=r"max_batch = 1"):
        backend_class(batched_problem)


def test_both_absorber_schemes_run_on_both_backends(
    backend_class, field_grid, capillary_problem
):
    """The temporal absorber runs both ways behind one switch, on every backend.

    The in-stage term is the default the comparison chose; the split factor stays
    reachable by name, because a scheme nobody can run is a scheme nobody can
    re-measure.
    """
    for scheme in ("nonlinear", "split_step"):
        problem = capillary_problem(scheme=scheme)
        assert problem.responses.absorber_scheme == scheme
        backend, plane = run_to(backend_class, problem, field_grid, 0.1)
        assert np.all(np.isfinite(plane.field))
        backend.close()


def test_an_inconsistent_problem_still_fails_validation(
    backend_class, soliton_problem, narrow_grid, substitute
):
    """The backend validates as its own defence, not as the caller's courtesy."""
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    # A constant prefactor that disagrees with its own table: the group accepts it (the
    # shapes match) and only validate can tell.
    wrong = np.ascontiguousarray(2.0 * problem.responses.prefactor(0.0))
    broken = substitute(
        problem,
        responses=dataclasses.replace(problem.responses, prefactor_constant=wrong),
        validate=False,
    )
    with pytest.raises(TranslationError, match="prefactor_constant"):
        backend_class(broken)


# -------------------------------------------------------------------- use and misuse


def test_the_initial_state_must_match_the_contract(
    backend_class, soliton_problem, narrow_grid
):
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend = backend_class(problem)
    with pytest.raises(BackendError, match="expected"):
        backend.set_initial(np.zeros((1, 1, 1, 3), dtype=complex), None)
    backend.close()


def test_propagating_without_a_state_is_refused(
    backend_class, soliton_problem, narrow_grid
):
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend = backend_class(problem)
    with pytest.raises(BackendError, match="call set_initial"):
        backend.propagate_to(LENGTH)
    backend.close()


@pytest.mark.parametrize("target", [-1.0, 10.0, float("nan")])
def test_an_unreachable_plane_is_refused(
    backend_class, soliton_problem, narrow_grid, target
):
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend = backend_class(problem)
    backend.set_initial(problem.initial.field0, None)
    with pytest.raises(BackendError):
        backend.propagate_to(target)
    backend.close()


def test_close_is_idempotent_and_use_after_close_is_refused(
    backend_class, soliton_problem, narrow_grid
):
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    backend.close()
    backend.close()
    for call in (
        lambda: backend.propagate_to(LENGTH),
        backend.observe,
        lambda: backend.stats(StatsSpec(names=("energy",))),
        lambda: backend.set_initial(problem.initial.field0, None),
    ):
        with pytest.raises(BackendError, match="closed"):
            call()


# ------------------------------------------------------------------- the memory ledger


def test_the_memory_ledger_names_every_buffer(
    backend_class, soliton_problem, narrow_grid
):
    """Every buffer the backend works in, named and sized, even when it is empty.

    Asserted as a superset rather than as an exact set: the rows below are the ones
    ``architecture.md`` section 8's budget is measured in and every backend must
    declare, and a backend that preallocates has more of its own -- its plan work
    areas, the state a trial step writes before it is known to be kept -- which the
    reference, allocating per pass, never has to name.
    """
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend = backend_class(problem, grid=narrow_grid)
    ledger = backend.memory_ledger()
    required = {
        "state",
        "stages",
        "accumulator_solution",
        "accumulator_error",
        "noise",
        "propagators",
        "fine_spectrum",
        "modal_time",
        "projected",
        "node_field",
        "node_polarisation",
        "statistics_power_weight",
    }
    assert required <= set(ledger)
    field_bytes = int(np.prod(problem.shape)) * 16
    assert ledger["state"] == field_bytes
    assert ledger["stages"] == problem.stepper.tableau.n_stages * field_bytes
    assert ledger["noise"] == 0
    # One float64 weight per column and none per batch member: the members of one
    # problem share the fibre, so they share `Re n_eff` (kernel-contract.md §5).
    columns = int(np.prod(problem.shape[1:]))
    assert ledger["statistics_power_weight"] == columns * 8
    assert all(value >= 0 for value in ledger.values())
    backend.close()
    if backend.name == "numpy_ref":
        # Released with everything else, and still named: a closed backend owes the
        # budget a row of zero, not a missing row. `numba_cpu` asserts the same in its
        # own ledger test; native refuses a ledger read once the native state is gone.
        assert backend.memory_ledger()["statistics_power_weight"] == 0


def test_the_ledger_counts_the_noise_buffer_when_one_is_carried(
    backend_class, soliton_problem, narrow_grid
):
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=modified_noise(narrow_grid, np.random.default_rng(3)),
    )
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    assert backend.memory_ledger()["noise"] == problem.initial.field0.nbytes
    backend.close()


def test_recomputing_the_propagators_costs_less_memory(
    backend_class, soliton_problem, narrow_grid
):
    tabulated = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH], propagator_mode="tabulate"
    )
    recomputed = soliton_problem(
        narrow_grid, 1, save_planes=[LENGTH], propagator_mode="recompute"
    )
    first = backend_class(tabulated).memory_ledger()["propagators"]
    second = backend_class(recomputed).memory_ledger()["propagators"]
    assert second < first


def test_the_propagator_mode_defaults_to_the_backends_own_choice(
    backend_class, soliton_problem, narrow_grid
):
    """The flag is the problem's, the choice is the backend's, and it is recorded.

    The reference forms two complex exponentials per stage and keeps the six distinct
    exponents of a step; ``numba_cpu`` takes both signs from one ``exp`` and a
    ``sincos`` in the stage that formed them, so the table buys residency and nothing
    else and it recomputes. Either can still be asked for by name, and the two agree
    to rounding, so what the default changes is memory rather than numbers.
    """
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    assert problem.stepper.propagator_mode == "auto"
    backend = backend_class(problem, grid=narrow_grid)
    try:
        resolved = backend.metadata()["propagator_mode"]
        assert resolved in {"tabulate", "recompute"}
        assert (
            resolved
            == {
                "native": "recompute",
                "numpy_ref": "tabulate",
                "numba_cpu": "recompute",
            }[backend.name]
        )
        named = backend_class(
            soliton_problem(
                narrow_grid, 1, save_planes=[LENGTH], propagator_mode=resolved
            ),
            grid=narrow_grid,
        )
        assert (
            backend.memory_ledger()["propagators"]
            == named.memory_ledger()["propagators"]
        )
        named.close()
    finally:
        backend.close()


# -------------------------------------------------------------------------- the noise


def seeded_finals(backend_class, soliton_problem, narrow_grid, seeds):
    """Return the final field of a replayed run per noise seed."""
    finals = []
    for seed in seeds:
        problem = soliton_problem(
            narrow_grid,
            1,
            save_planes=[LENGTH],
            schedule=SCHEDULE,
            noise=modified_noise(narrow_grid, np.random.default_rng(seed)),
        )
        backend, plane = run_to(backend_class, problem, narrow_grid, LENGTH)
        backend.close()
        finals.append(plane.field)
    return finals


def test_the_same_seed_gives_the_same_run(backend_class, soliton_problem, narrow_grid):
    """The realisation and the seed are the whole of the run's randomness.

    The same stored realisation twice, on one schedule: the eight-step rung's
    rounding budget (:data:`REPEAT_BOUND`).
    """
    finals = seeded_finals(backend_class, soliton_problem, narrow_grid, (17, 17))
    assert_within(finals[1], finals[0], REPEAT_BOUND, what="same seed")


def test_a_different_seed_gives_a_different_run(
    backend_class, soliton_problem, narrow_grid
):
    """Another realisation moves the run by far more than its rounding."""
    finals = seeded_finals(backend_class, soliton_problem, narrow_grid, (17, 18))
    assert departure(finals[1], finals[0]) > 1e3 * REPEAT_BOUND


def test_the_propagating_field_never_carries_the_modified_noise(
    backend_class, soliton_problem, narrow_grid
):
    """The noise is added at every stage, never to the propagating state."""
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=modified_noise(narrow_grid, np.random.default_rng(5)),
    )
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    start = backend.observe()
    assert np.array_equal(
        start.field, problem.initial.field0 * problem.grids.active_mask
    )
    backend.propagate_to(LENGTH)
    plane = backend.observe()
    assert plane.noise_phase is not None
    assert plane.noise_phase.shape == problem.shape
    backend.close()


def test_the_input_noise_model_adds_the_realisation_once(
    backend_class, soliton_problem, narrow_grid
):
    """The traditional model puts the noise in the field and then forgets about it."""
    realisation = modified_noise(narrow_grid, np.random.default_rng(9))
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=dataclasses.replace(realisation, model="input"),
    )
    backend = backend_class(problem, grid=narrow_grid)
    backend.set_initial(problem.initial.field0, None)
    start = backend.observe()
    added = start.field - problem.initial.field0 * problem.grids.active_mask
    assert np.max(np.abs(added)) > 0.0
    backend.close()


# --------------------------------------------------------------------- the statistics


def test_statistics_reproduce_the_reference(
    backend_class, soliton_problem, narrow_grid
):
    """There is one definition of each statistic in the package; a backend calls it.

    Not bitwise on any backend (SEP 25): the backend's own transform and reductions
    and the reference's differ in their order. The spectral sums contain at most n_t
    terms; 16 n eps covers two transforms/reductions and the FWHM interpolation,
    fixed before any value is observed.
    """
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend, plane = run_to(backend_class, problem, narrow_grid, LENGTH)
    spec = StatsSpec(
        names=("energy", "peak_power", "fwhm_t", "omega0"),
        energy_windows=np.array([[2.0e15, 2.5e15]]),
    )
    got = backend.stats(spec)
    want = collect(spec, narrow_grid, SavedPlane(z=plane.z, field=plane.field[0]))
    assert set(got) == set(want)
    for name, value in want.items():
        np.testing.assert_allclose(
            got[name][0],
            value,
            rtol=16 * narrow_grid.n_t * EPS,
            atol=0.0,
            equal_nan=True,
        )
        assert got[name].shape == (1, *value.shape)
    backend.close()


def test_statistics_without_a_grid_say_what_is_missing(
    backend_class, soliton_problem, narrow_grid
):
    """The contract carries no grid, so the backend has to be told which one it is."""
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    backend = backend_class(problem)
    backend.set_initial(problem.initial.field0, None)
    with pytest.raises(BackendError, match="grid=grid"):
        backend.stats(StatsSpec(names=("energy",)))
    backend.close()


# ------------------------------------------------------- the two propagator modes


def test_the_two_propagator_modes_agree(backend_class, soliton_problem, soliton_grid):
    """Memoising the exponents within a step changes the work, not the numbers.

    The memoised path evaluates delta_phi once per distinct stage fraction and the
    other once per stage, on the same constant table: one method by two routes on
    one schedule, within the eight-step rung's budget.
    """
    tabulated = soliton_problem(
        soliton_grid, 3, save_planes=[LENGTH], schedule=SCHEDULE
    )
    recomputed = soliton_problem(
        soliton_grid,
        3,
        save_planes=[LENGTH],
        schedule=SCHEDULE,
        propagator_mode="recompute",
    )
    backend, first = run_to(backend_class, tabulated, soliton_grid, LENGTH)
    backend.close()
    backend, second = run_to(backend_class, recomputed, soliton_grid, LENGTH)
    backend.close()
    assert_within(second.field, first.field, REPEAT_BOUND, what="propagator modes")


# -------------------------------------------------- the third harmonic on an envelope


def test_the_envelope_third_harmonic_term_is_executed(
    backend_class, thg_problem, thg_grid
):
    """The one compiled Kerr path nothing else exercises; the carrier is built here."""
    with_harmonic = thg_problem(thg=True)
    without = thg_problem(thg=False)
    assert with_harmonic.responses.kerr_coefficients.harmonic_factor == 0.25
    assert without.responses.kerr_coefficients.harmonic_factor == 0.0
    _, first = run_to(backend_class, with_harmonic, thg_grid, 1e-3)
    _, second = run_to(backend_class, without, thg_grid, 1e-3)
    difference = np.max(np.abs(first.field - second.field))
    assert difference > 0.0
    # The harmonic is radiated near 3 omega_0, far outside the band the fundamental
    # occupies, so it must show up above the pump's own bins there and nowhere else.
    assert np.max(np.abs(first.field)) == pytest.approx(
        np.max(np.abs(second.field)), rel=1e-3
    )


# ------------------------------------------------------------- the gradient capillary


def test_a_gradient_capillary_propagates_on_the_field_grid(
    backend_class, capillary_problem, field_grid, caplog
):
    """A z-varying Phi, a tabulated prefactor, a hard kink and a temporal absorber."""
    problem = capillary_problem()
    assert not problem.flags.constant_linop
    assert problem.responses.prefactor_constant is None
    assert problem.flags.has_temporal_absorber
    assert problem.linear.hard_kinks.tolist() == [0.2]
    with caplog.at_level(logging.WARNING):
        backend, plane = run_to(backend_class, problem, field_grid, 0.1)
    assert np.all(np.isfinite(plane.field))
    assert plane.z == 0.1
    assert not caplog.records
    # Steps land on the kink at 0.2 m, so continuing past it must reach it exactly.
    backend.propagate_to(0.2)
    after = backend.observe()
    assert 0.2 in after.step_history.z.tolist()
    backend.close()


def test_the_alarm_is_weighted_at_the_stage_it_was_scored_from(
    backend_class, capillary_problem, field_grid
):
    """The prefactor varies along this fibre, so the alarm's weight has to follow it.

    Regression: ``numba_cpu`` held the position of its last stage in a field that was
    set to zero at construction and never written, so the alarm was weighted by
    ``prefactor_at(responses, 0.0)`` for the whole run and its warning always named
    ``z = 0``. Invisible on a uniform fill and wrong on any gradient. The bundle is
    what makes it right: the alarm reads the scalars of the stage it was scored from.
    """
    problem = capillary_problem()
    backend, _ = run_to(backend_class, problem, field_grid, 0.1)
    bundle = backend._last_bundle
    assert bundle is not None
    stage_z = bundle.z1
    # The stage the alarm was scored from is the last one of the last accepted step,
    # which lands on the plane rather than sitting at the entrance.
    assert stage_z == pytest.approx(0.1, abs=1e-12)
    weight = prefactor_at(problem.responses, stage_z)
    assert not np.allclose(weight, prefactor_at(problem.responses, 0.0))
    backend.close()


def test_a_z_table_runs_on_the_numba_host_adapter_and_the_oracle(
    tabulated_problem, field_grid
):
    """Numba's host adapter qualifies the general CPU baseline beside the oracle."""
    assert not tabulated_problem.device_ready
    # The host Clenshaw recurrence follows the reference operation order; budget
    # 64 eps per coefficient and sample over seven stages and 1000 accepted steps.
    bound = (
        64
        * field_grid.n_to
        * len(tabulated_problem.linear.phi_coefficients[0])
        * 7
        * 1000
        * np.finfo(float).eps
    )
    assert bound > 0
    compiled, measured = run_to(NumbaCpuBackend, tabulated_problem, field_grid, 0.05)
    backend, plane = run_to(NumpyRefBackend, tabulated_problem, field_grid, 0.05)
    assert np.all(np.isfinite(plane.require_field()))
    assert (
        np.linalg.norm(measured.require_field() - plane.require_field())
        / np.linalg.norm(plane.require_field())
        < bound
    )
    assert compiled._host_phi is not None and compiled._host_prefactor is not None
    compiled.close()
    backend.close()


# ------------------------------------------------------------- the remaining paths


def test_construction_logs_what_it_was_given(soliton_problem, narrow_grid, caplog):
    """The translation summary a debug run wants: grid, shape, table, modes."""
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    with caplog.at_level(logging.DEBUG, logger="solprop.backends.numpy_ref.backend"):
        NumpyRefBackend(problem, grid=narrow_grid).close()
    assert any(
        "numpy_ref: envelope grid" in record.message for record in caplog.records
    )


def test_a_step_too_large_for_the_tolerance_is_rejected_and_retried(
    backend_class, soliton_problem, soliton_grid
):
    """A step the tolerance rejects is shrunk and retried, not clamped and accepted."""
    problem = soliton_problem(
        soliton_grid,
        3,
        save_planes=[0.05],
        controller=PIController(rtol=1e-12, atol=1e-30),
        dz_max=0.05,
        dz_init=0.05,
    )
    backend = backend_class(problem, grid=soliton_grid)
    backend.set_initial(problem.initial.field0, None)
    summary = backend.propagate_to(0.05)
    plane = backend.observe()
    backend.close()
    assert summary.n_rejected > 0
    assert not np.all(plane.step_history.accepted)
    # A rejection shrinks the proposal at once, without the controller's lag, so the
    # retry is shorter than the attempt it replaces.
    rejected = np.flatnonzero(~plane.step_history.accepted)[0]
    assert plane.step_history.dz[rejected + 1] < plane.step_history.dz[rejected]
    assert summary.met_tolerance


def test_the_realisation_passed_to_set_initial_is_the_one_carried(
    backend_class, soliton_problem, narrow_grid
):
    """The protocol loads the noise; the problem's own realisation is the fallback."""
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        noise=modified_noise(narrow_grid, np.random.default_rng(21)),
    )
    other = modified_noise(narrow_grid, np.random.default_rng(22))
    finals = []
    for realisation in (None, other):
        backend = backend_class(problem, grid=narrow_grid)
        backend.set_initial(problem.initial.field0, realisation)
        backend.propagate_to(LENGTH)
        finals.append(backend.observe().require_field())
        backend.close()
    assert not np.array_equal(finals[0], finals[1])


def test_the_weighted_norm_is_the_weak_one_when_the_weights_are_all_ones(
    backend_class, soliton_problem, narrow_grid
):
    """The weights redistribute the tolerance; the magnitude stays global either way.

    On one schedule, so that the two norms are of the same steps: the per-bin form
    divides each bin by one and reduces in its own order, so the norms agree to the
    reduction's ``gamma(n_omega)`` (twice, one per side) and the fields, which the
    norm does not touch on a replayed schedule, to the eight-step rung's budget.
    """
    ones = np.ones(narrow_grid.n_omega)
    weighted = soliton_problem(
        narrow_grid,
        3,
        save_planes=[LENGTH],
        schedule=SCHEDULE,
        controller=PIController(rtol=1e-9, atol=1e-18, weights=ones),
    )
    weak = soliton_problem(
        narrow_grid,
        3,
        save_planes=[LENGTH],
        schedule=SCHEDULE,
        controller=PIController(rtol=1e-9, atol=1e-18),
    )
    backend, first = run_to(backend_class, weighted, narrow_grid, LENGTH)
    backend.close()
    backend, second = run_to(backend_class, weak, narrow_grid, LENGTH)
    backend.close()
    assert_within(first.field, second.field, REPEAT_BOUND, what="field")
    np.testing.assert_allclose(
        first.step_history.error_norm,
        second.step_history.error_norm,
        rtol=2 * gamma(narrow_grid.n_omega),
    )


def test_tightening_one_band_s_weight_shortens_the_steps(
    backend_class, soliton_problem, narrow_grid
):
    """A weight below one tightens that bin, which is what the option is for."""
    weights = np.ones(narrow_grid.n_omega)
    weights[narrow_grid.omega > narrow_grid.omega0] = 1e-3
    tightened = soliton_problem(
        narrow_grid,
        3,
        save_planes=[LENGTH],
        controller=PIController(rtol=1e-9, atol=1e-18, weights=weights),
    )
    loose = soliton_problem(
        narrow_grid,
        3,
        save_planes=[LENGTH],
        controller=PIController(rtol=1e-9, atol=1e-18),
    )
    _, tight = run_to(backend_class, tightened, narrow_grid, LENGTH)
    _, relaxed = run_to(backend_class, loose, narrow_grid, LENGTH)
    assert tight.n_steps > relaxed.n_steps


def test_a_harmonic_term_without_its_carrier_is_refused(thg_problem):
    """The carrier is the contract's now; a problem missing it cannot be run."""
    problem = thg_problem(thg=True)
    grids = dataclasses.replace(problem.grids)
    object.__setattr__(grids, "carrier_2w0", None)
    stripped = dataclasses.replace(problem)
    object.__setattr__(stripped, "grids", grids)
    with pytest.raises(BackendError, match="carrier_2w0 is absent"):
        backend_module._harmonic_carrier(stripped)


def test_the_harmonic_carrier_comes_from_the_contract(thg_problem):
    problem = thg_problem(thg=True)
    assert problem.grids.carrier_2w0 is not None
    assert backend_module._harmonic_carrier(problem) is problem.grids.carrier_2w0
    assert backend_module._harmonic_carrier(thg_problem(thg=False)) is None


def test_the_z_table_answers_the_noise_phase_too(tabulated_problem, field_grid):
    """The fallback covers all three evaluations, not only the propagator.

    A problem the bundle cannot carry reads ``Phi``, the prefactor *and* the noise's
    unitary phase from the z table. The first two have their own tests above; this is
    the third, which only a run with the ``modified`` noise model reaches in anger.
    """
    backend = NumpyRefBackend(tabulated_problem, grid=field_grid)
    try:
        bundle = backend._stage_scalars(0.0, 0.05)
        assert bundle.phi_at is None
        got = backend._unitary_phase(bundle)
        want = unitary_phase(tabulated_problem.linear, 0.05)
    finally:
        backend.close()
    # One table evaluation by two routes: a few roundings of the phase.
    assert_within(got, want, 8 * EPS, what="unitary phase")


# ------------------------------------------- the statistics channel between planes


PERIODIC = ("energy", "peak_power", "fwhm_t", "omega0")
"""The subset the channel tests ask for: two spectral reductions and two temporal."""


def with_channel(problem, *, period=1, names=PERIODIC):
    """Return `problem` with `names` asked for at planes and between them."""
    spec = StatsSpec(names=names, periodic=names, period=period)
    return dataclasses.replace(problem, stats_spec=spec)


def test_a_periodic_row_at_a_plane_is_the_plane_s_own_statistic(
    soliton_problem, narrow_grid, backend_class
):
    """The same function on the same state, so the same numbers to its rounding.

    The row is evaluated by the same :meth:`Backend.stats` the save plane uses, on the
    state the accepted step committed, so when a row lands on a plane the two are the
    same call on the same array: never one definition for the host and another for
    the device, where the two could drift. Two calls may order their reductions
    differently (SEP 25), so they agree within :data:`STATISTICS_BOUND`.
    """
    problem = with_channel(
        soliton_problem(narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH])
    )
    backend = backend_class(problem, grid=narrow_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH / 2)
        observed = backend.observe()
        at_plane = backend.stats(problem.stats_spec)
        rows = observed.step_stats
        # At period one the last accepted step is the landing on the plane.
        assert rows.step[-1] == observed.step_history.n_steps - 1
        for name in PERIODIC:
            np.testing.assert_allclose(
                rows.values[name][-1],
                at_plane[name],
                rtol=STATISTICS_BOUND,
                atol=0,
                err_msg=name,
            )
    finally:
        backend.close()


def test_the_rows_stand_at_the_accepted_positions_of_the_history(
    soliton_problem, narrow_grid, backend_class
):
    """One row per accepted step at period one, at that step's own end position."""
    problem = with_channel(soliton_problem(narrow_grid, 1, save_planes=[LENGTH]))
    backend = backend_class(problem, grid=narrow_grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend.propagate_to(LENGTH)
        observed = backend.observe()
    finally:
        backend.close()
    accepted = np.flatnonzero(observed.step_history.accepted)
    rows = observed.step_stats
    np.testing.assert_array_equal(rows.step, accepted)
    np.testing.assert_array_equal(rows.z, observed.step_history.z[accepted])


@pytest.mark.parametrize("period", [1, 4])
def test_a_run_with_the_channel_is_the_run_without(
    soliton_problem, narrow_grid, backend_class, period
):
    """The channel reads the state and writes nothing of the step's.

    ``wall_time`` is excluded because it is a clock reading rather than a result. The
    other four history columns and the exit field are the whole of what the step loop
    computed, and the channel must not move any of them -- which on ``numba_cpu`` also
    means its statistics pass must leave the plan buffers in a state the next stage
    rewrites completely. The two runs replay one schedule: the schedule and the
    acceptances are exact, the field within the eight-step rung's budget and the
    error norm, a defect over ``rtol``, within seven stages of that over ``rtol``.
    """
    columns = ("z", "dz", "accepted")

    def run(spec_of):
        problem = spec_of(
            soliton_problem(
                narrow_grid, 1, save_planes=[LENGTH / 2, LENGTH], schedule=SCHEDULE
            )
        )
        backend = backend_class(problem, grid=narrow_grid)
        backend.set_initial(problem.initial.field0, None)
        fields, histories = [], []
        for plane in (LENGTH / 2, LENGTH):
            backend.propagate_to(plane)
            observed = backend.observe()
            fields.append(observed.field)
            histories.append(observed.step_history)
        backend.close()
        return fields, histories

    plain_fields, plain_history = run(lambda problem: problem)
    watched_fields, watched_history = run(
        lambda problem: with_channel(problem, period=period)
    )
    rtol = soliton_problem(narrow_grid, 1, save_planes=[LENGTH]).stepper.controller.rtol
    for plain, watched in zip(plain_fields, watched_fields, strict=True):
        assert_within(watched, plain, REPEAT_BOUND, what="field")
    for plain, watched in zip(plain_history, watched_history, strict=True):
        for column in columns:
            assert np.array_equal(getattr(plain, column), getattr(watched, column))
        np.testing.assert_allclose(
            watched.error_norm, plain.error_norm, rtol=0, atol=7 * REPEAT_BOUND / rtol
        )
