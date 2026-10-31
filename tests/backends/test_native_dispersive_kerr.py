# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response on native CPU's fused node path.

solprop-native evaluates ``responses.dispersive_kerr`` with the structure ``numba_cpu``
takes (``backends/numba_cpu/dispersive.py``): the spectral multipliers on the modal
columns, the products at the nodes under the collar, projected and landed in the
stage's own polarisation spectrum, the embedded rule's finished when it is scored, a
lone symmetric response on the stage's own transforms, the symmetric mixture filter
rebuilt when the densities change. ``numpy_ref``, which applies the multipliers at
every node, is the oracle, and ``numba_cpu`` is checked beside it. The ladder of
``docs/dev/backends.md`` section 3.7 for the response, as ``numba_cpu``'s
(``test_numba_dispersive_kerr.py``): one nonlinear evaluation (the projected
polarisation on both quadrature rules), one step and ten steps, on the multimode
capillary, the mode-averaged and radial routes and a two-component field, on field and
envelope grids, in both sectors, for both evaluators; mixtures of constant and varying
composition; Raman, plasma and envelope plasma beside the response. Then what is the
backend's own: repeatability, the thread count and the memory ledger.

**The bounds are ``numba_cpu``'s** (:func:`evaluation_bound`, :func:`steps_bound`),
derived from the transforms and node reductions both backends perform and counted for
the reference's node-by-node arithmetic too, before any run: the native response takes
the same transforms per channel (fewer when a lone symmetric response is the stage's
own polarisation) and the same reductions over the nodes. Against ``numba_cpu`` the
same bound holds, both compiled backends sitting inside it from the reference's side.
Each test states what it measured.
"""

import dataclasses
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backends.matched import fixed_order, native_thread_fields
from backends.test_numba_dispersive_kerr import (
    GRID_CASES,
    LENGTH,
    POLICIES,
    STEPS,
    Evaluation,
    averaged,
    capillary,
    disagreement,
    evaluate,
    radial,
    run,
)
from backends.test_numba_dispersive_kerr import evaluation_bound as cpu_evaluation_bound
from bounds import assert_within, dispersive_device_bound, step_evaluations
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.physdata import MaterialSelection, Mixture
from solprop.profiles import Linear
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.translate.problem import KernelProblem
from solprop.translate.stage import scalars

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

ON_DEVICE = solprop_native.build_info().device_backend != "none"
host_only = pytest.mark.skipif(
    ON_DEVICE,
    reason="the thread count is a host notion; a device build is held run to run",
)


def evaluation_bound(problem: KernelProblem) -> float:
    """``numba_cpu``'s bound, or on a device build the predeclared device envelope."""
    return dispersive_device_bound(cpu_evaluation_bound(problem), problem)


def steps_bound(problem: KernelProblem) -> float:
    """Bound a replayed window of fixed steps: every evaluation's rounding, added."""
    return STEPS * step_evaluations(problem) * evaluation_bound(problem)


PROBE = Path(__file__).parent / "native_dispersive_thread_probe.py"

DISPERSIVE_ROWS = (
    "dispersive_filtered_time",
    "dispersive_node_fields",
    "dispersive_products",
    "dispersive_coarse_products",
    "dispersive_product_spectra",
    "dispersive_projection_chunks",
    "dispersive_multipliers",
)
"""The response's rows of the native memory ledger, named on every problem."""


# --- the measurements ------------------------------------------------------------


def embedded(problem: KernelProblem) -> bool:
    """Return whether the quadrature has an embedded rule of its own, as packed."""
    transform = problem.transform
    return transform.n_nodes > 1 and not np.array_equal(
        transform.projection, transform.projection_coarse
    )


def native_evaluation(problem: KernelProblem, grid: Any) -> Evaluation:
    """Evaluate the source once at the launch and return the native projected spectra.

    The engine keeps both rules' projected polarisation spectra of its last
    evaluation, unnormalised, as the reference's projections' transforms are; the
    embedded rule's dispersive share is finished when it is read.
    """
    backend = NativeBackend(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._adams_lab_source(problem.initial.field0, 0.0)
        native = backend._native_state
        return Evaluation(
            native.projected_to_host(False),
            native.projected_to_host(True) if embedded(problem) else None,
        )
    finally:
        backend.close()


def assert_close(ours: Evaluation, theirs: Evaluation, bound: float) -> None:
    """Both rules' projected spectra within a bound, and the same rules present."""
    assert disagreement(ours.projected, theirs.projected) < bound
    assert (ours.coarse is None) == (theirs.coarse is None)
    if ours.coarse is not None and theirs.coarse is not None:
        assert disagreement(ours.coarse, theirs.coarse) < bound


def assert_one_evaluation(
    problem: KernelProblem, grid: Any, *, numba: bool = True
) -> None:
    """One evaluation, both rules, against the reference and ``numba_cpu``."""
    ours = native_evaluation(problem, grid)
    bound = evaluation_bound(problem)
    assert bound > 0
    assert_close(ours, evaluate(NumpyRefBackend, problem, grid), bound)
    if numba:
        assert_close(ours, evaluate(NumbaCpuBackend, problem, grid), bound)


def source(backend_class: type, problem: KernelProblem, grid: Any) -> np.ndarray:
    """Return one evaluation's whole stage source at the launch."""
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        return np.array(backend._adams_lab_source(problem.initial.field0, 0.0))
    finally:
        backend.close()


def assert_steps(problem: KernelProblem, grid: Any, *, numba: bool = True) -> None:
    """A replayed window of fixed steps, the same steps, against both backends."""
    ours, summary = run(NativeBackend, problem, grid)
    bound = steps_bound(problem)
    assert bound > 0
    others = [NumpyRefBackend, NumbaCpuBackend] if numba else [NumpyRefBackend]
    for backend_class in others:
        theirs, reference = run(backend_class, problem, grid)
        assert summary.n_steps == reference.n_steps
        assert disagreement(ours, theirs) < bound


def path_of(problem: KernelProblem, grid: Any) -> dict[str, bool] | None:
    """Return how the native response runs: alone, in place, borrowed."""
    backend = NativeBackend(problem, grid=grid)
    try:
        path = backend.metadata()["dispersive_path"]
    finally:
        backend.close()
    return path


# --- the ladder --------------------------------------------------------------------


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_one_evaluation_on_the_multimode_capillary_matches_the_reference(
    case, evaluator
):
    """Both rules' projected polarisation, under the in-stage absorber.

    Alone, the response holds its own filtered field beside the bare one the collar
    reads; a symmetric one projects through the stage unless its products are a field
    grid's analytic signals. Measured 2.3-4.3 eps on both rules against the reference
    and 0.5-2.3 eps against ``numba_cpu``, under bounds of 1060-1870 eps.
    """
    simulation = capillary(evaluator, **GRID_CASES[case])
    problem = simulation.translate()
    assert problem.transform.n_nodes == 33
    path = path_of(problem, simulation.grid)
    analytic = case == "field conserving"
    assert path == {
        "alone": True,
        "in_place": False,
        "borrowed": evaluator == "symmetric" and not analytic,
    }
    assert_one_evaluation(problem, simulation.grid)


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_one_step_on_the_multimode_capillary_matches_the_reference(case, evaluator):
    """One Runge-Kutta step, seven evaluations through the whole stage.

    Under the split-step absorber a lone symmetric response filters the stage's own
    spectrum in place. Measured 1.2-1.5 eps against bounds of 7400-13100 eps.
    """
    simulation = capillary(
        evaluator,
        stepped=True,
        integrator=RungeKutta(),
        save_planes=[LENGTH / STEPS],
        n_save=None,
        **GRID_CASES[case],
    )
    problem = simulation.translate()
    assert path_of(problem, simulation.grid) == {
        "alone": True,
        "in_place": evaluator == "symmetric",
        "borrowed": evaluator == "symmetric" and case != "field conserving",
    }
    ours, summary = run(NativeBackend, problem, simulation.grid)
    theirs, _ = run(NumpyRefBackend, problem, simulation.grid)
    assert summary.n_steps == 1
    bound = step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    assert disagreement(ours, theirs) < bound


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_ten_steps_on_the_multimode_capillary_match_the_reference(case, evaluator):
    """The default integrator (the resident Adams engine) over ten fixed steps.

    Measured 5.8-6.7 eps against the reference and 5.6-6.7 eps against ``numba_cpu``,
    under bounds of 155 000-273 000 eps.
    """
    simulation = capillary(evaluator, stepped=True, **GRID_CASES[case])
    assert_steps(simulation.translate(), simulation.grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("sector", ["field full", "field conserving"])
def test_the_quadrature_alarm_is_the_reference_s(sector, evaluator):
    """The statistic the embedded rule exists for, after ten steps.

    The response's share of the embedded projection is finished when the step is
    scored; without it the alarm would be near one. Both alarms agree to the window's
    and an evaluation's bounds.
    """
    simulation = capillary(evaluator, stepped=True, **GRID_CASES[sector])
    problem = simulation.translate()
    spec = StatsSpec(names=("quadrature_error",))
    alarms = []
    for backend_class in (NativeBackend, NumpyRefBackend):
        backend = backend_class(problem, grid=simulation.grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(LENGTH)
            alarms.append(float(backend.stats(spec)["quadrature_error"][0]))
        finally:
            backend.close()
    ours, theirs = alarms
    assert theirs > 0
    assert abs(ours - theirs) < steps_bound(problem) + evaluation_bound(problem)


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize(
    ("case", "raman"),
    [
        ("field full", False),
        ("field conserving", False),
        ("field conserving", True),
        ("envelope full", False),
    ],
)
def test_node_chunks_accumulate_as_one(case, raman, evaluator):
    """Five chunks of eight nodes, the last of one: every product accumulates.

    The response packs each chunk's columns of the scaled projections once a stage.
    """
    simulation = capillary(
        evaluator, gas="N2" if raman else "Ar", raman=raman, **GRID_CASES[case]
    )
    problem = simulation.translate()
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=8)
    )
    assert_one_evaluation(problem, simulation.grid, numba=False)


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("thg", [True, False])
def test_a_state_with_zero_frequency_content_matches_the_reference(thg, evaluator):
    """The polynomial subtracts the bare analytic signal's zero-frequency constant.

    ``Pi_+`` excludes the zero-frequency bin, which the engine's analytic family keeps.
    """
    simulation = capillary(evaluator, thg=thg)
    problem = simulation.translate()
    start = problem.initial.field0.copy()
    start[..., 0] = 1e-3 * np.abs(start).max()
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=start)
    )
    assert_one_evaluation(problem, simulation.grid, numba=False)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_the_mode_averaged_route_matches_the_reference(case, evaluator):
    """One node: one evaluation and ten steps."""
    problem, grid = averaged(evaluator, **GRID_CASES[case])
    assert problem.transform.n_nodes == 1
    assert_one_evaluation(problem, grid, numba=False)
    assert_steps(problem, grid, numba=False)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", ["field full", "field conserving", "envelope full"])
def test_a_two_component_field_matches_the_reference(case, evaluator):
    """The isotropic Kleinman contraction of an elliptical launch, on the quadrature."""
    problem, grid = averaged(evaluator, vector=True, **GRID_CASES[case])
    assert problem.transform.n_pol * problem.shape[2] == 2
    assert_one_evaluation(problem, grid, numba=False)
    assert_steps(problem, grid, numba=False)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("envelope", [False, True])
def test_the_radial_route_matches_the_reference(envelope, evaluator):
    """The Hankel route, sixteen nodes."""
    problem, grid = radial(evaluator, envelope=envelope, thg=envelope)
    assert problem.transform.n_nodes == 16
    assert_one_evaluation(problem, grid, numba=False)
    assert_steps(problem, grid, numba=False)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", ["field full", "envelope conserving"])
def test_a_mixture_matches_the_reference(case, evaluator):
    """Helium and argon: one product under the mixture filter, or the polynomial's sum.

    The symmetric evaluator's filter is rebuilt natively from the channels' densities by
    the reference's arithmetic; the polynomial's ``E^3`` is shared and weighed by both
    channels' output multipliers.
    """
    simulation = capillary(evaluator, gas={"He": 0.6, "Ar": 0.4}, **GRID_CASES[case])
    problem = simulation.translate()
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.n_channels == 2
    assert_one_evaluation(problem, simulation.grid)
    stepped = capillary(
        evaluator, gas={"He": 0.6, "Ar": 0.4}, stepped=True, **GRID_CASES[case]
    )
    assert_steps(stepped.translate(), stepped.grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_a_mixture_of_varying_composition_matches_the_reference(evaluator):
    """Helium falling and argon rising along the guide: the filter rebuilt per stage."""
    fill = Mixture.partial_pressures(
        {"He": Linear(2.0, 0.5, LENGTH), "Ar": Linear(0.5, 2.0, LENGTH)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            75e-6,
            LENGTH,
            fill,
            None,
            wavelength=800e-9,
            wavelength_limits=(150e-9, 4e-6),
            trange=150e-15,
            fwhm=10e-15,
            energy=30e-6,
            modes=3,
            materials=MaterialSelection(electronic=POLICIES[evaluator]),
            plasma=False,
            raman=False,
            backend="numpy_ref",
            n_save=2,
            absorber_scheme="split_step",
            schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
            noise=None,
        )
    problem = simulation.translate()
    start = scalars(problem, 0.0, 0.0).dispersive_kerr_densities
    end = scalars(problem, 0.0, LENGTH).dispersive_kerr_densities
    assert start[0] / start[1] == pytest.approx(4.0, rel=1e-2)
    assert end[0] / end[1] == pytest.approx(0.25, rel=1e-2)
    assert_steps(problem, simulation.grid)


@pytest.mark.medium
@pytest.mark.parametrize("case", ["field full", "field conserving", "envelope full"])
def test_species_with_one_leg_bound_share_their_derivative_fields(case):
    """Krypton and xenon: one leg bound, one group of the polynomial.

    One ``D E`` and one ``D^2 E`` with their coefficient tables added,
    density-weighted: an exact rewriting of the reference's per-species sum.
    """
    band = {"wavelength_limits": (180e-9, 4e-6), "energy": 5e-6}
    simulation = capillary(
        "polynomial", gas={"Kr": 0.5, "Xe": 0.5}, **band, **GRID_CASES[case]
    )
    problem = simulation.translate()
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.n_channels == 2
    assert leaf.groups() == ((0, 1),)
    ledger = NativeBackend(problem, grid=simulation.grid)
    try:
        held = ledger.memory_ledger()
    finally:
        ledger.close()
    n_columns, n_to = problem.shape[1] * problem.shape[2], problem.grids.n_to
    item = 8 if case == "field full" else 16
    # Two filtered fields and three products: one group.
    assert held["dispersive_filtered_time"] == 2 * n_columns * n_to * item
    assert held["dispersive_products"] == 3 * n_columns * n_to * item
    assert_one_evaluation(problem, simulation.grid)
    stepped = capillary(
        "polynomial",
        gas={"Kr": 0.5, "Xe": 0.5},
        stepped=True,
        **band,
        **GRID_CASES[case],
    )
    assert_steps(stepped.translate(), stepped.grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("thg", [True, False])
def test_raman_beside_the_response_reads_the_bare_field(thg, evaluator):
    """Nitrogen's Raman response beside it: the response must not filter the drive."""
    simulation = capillary(evaluator, gas="N2", raman=True, thg=thg)
    problem = simulation.translate()
    assert problem.flags.has_raman
    assert path_of(problem, simulation.grid) == {
        "alone": False,
        "in_place": False,
        "borrowed": False,
    }
    assert_one_evaluation(problem, simulation.grid)
    stepped = capillary(evaluator, gas="N2", raman=True, thg=thg, stepped=True)
    assert_steps(stepped.translate(), stepped.grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("envelope", [False, True])
def test_plasma_beside_the_response_reads_the_bare_field(envelope, evaluator):
    """The field grid's plasma column, or the envelope's own current streams.

    Compared on the whole stage source, which holds the envelope currents on both
    backends (the engine assembles them into the projected spectrum, the reference
    beside it). The plasma column and the prefactor's band edges already disagree with
    the reference's, so the bound adds the carrier-policy run's disagreement on this
    backend (268 and 178 eps measured); the dispersive runs sit within 0.2 eps of it. A
    plasma current the response filtered would be wrong by the multiplier's departure
    from one, about 1e-3, far above either. Ten steps: 6.8-19.5 eps against bounds of
    155 000-273 000 eps (the carrier policy's own: 6.6-6.7 eps).
    """
    kind = {"plasma": True, "thg": False, "envelope": envelope, "energy": 60e-6}
    simulation = capillary(evaluator, **kind)
    problem = simulation.translate()
    assert (problem.responses.envelope_plasma is not None) == envelope
    baseline = capillary("carrier-kerr", **kind)
    carrier = baseline.translate()
    floor = disagreement(
        source(NativeBackend, carrier, baseline.grid),
        source(NumpyRefBackend, carrier, baseline.grid),
    )
    bound = evaluation_bound(problem) + floor
    ours = source(NativeBackend, problem, simulation.grid)
    assert disagreement(ours, source(NumpyRefBackend, problem, simulation.grid)) < bound
    stepped = capillary(evaluator, stepped=True, **kind)
    stepped_problem = stepped.translate()
    stepped_carrier = capillary("carrier-kerr", stepped=True, **kind)
    carrier_problem = stepped_carrier.translate()
    step_floor = disagreement(
        run(NativeBackend, carrier_problem, stepped_carrier.grid)[0],
        run(NumpyRefBackend, carrier_problem, stepped_carrier.grid)[0],
    )
    ours_field, summary = run(NativeBackend, stepped_problem, stepped.grid)
    theirs_field, reference = run(NumpyRefBackend, stepped_problem, stepped.grid)
    assert summary.n_steps == reference.n_steps
    assert (
        disagreement(ours_field, theirs_field)
        < steps_bound(stepped_problem) + step_floor
    )


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_runge_kutta_steps_match_the_reference(evaluator):
    """The Runge-Kutta march: seven stages a step, the first recycled."""
    simulation = capillary(evaluator, stepped=True, integrator=RungeKutta())
    problem = simulation.translate()
    assert problem.stepper.integrator is None
    assert_steps(problem, simulation.grid)


# --- what is this backend's alone ---------------------------------------------------


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_two_runs_of_one_build_agree(evaluator):
    """Two runs of one build: the same fixed-order method on the same fixed schedule.

    A repeated run is the same discrete computation, so it is held to the window's
    rounding budget (:func:`steps_bound`), not to its bits: a threaded reduction
    may order its sums differently from one run to the next.
    """
    simulation = capillary(evaluator, gas="N2", raman=True, stepped=True)
    problem = fixed_order(simulation.translate())
    first, _ = run(NativeBackend, problem, simulation.grid)
    second, _ = run(NativeBackend, problem, simulation.grid)
    assert_within(second, first, steps_bound(problem), what="repeat run")


@pytest.mark.medium
@host_only
def test_one_thread_and_four_agree(tmp_path):
    """Ten steps at one thread and at four, each in a fresh process.

    The same fixed-order method on the same schedule, so the two differ by the order
    of their threaded transforms and node reductions alone: the window's rounding
    budget (:func:`steps_bound`, the transforms and reductions of every evaluation),
    as against the reference. A repeat at one thread is held to the same budget.
    """
    names = ["symmetric", "polynomial"]
    one = native_thread_fields(PROBE, 1, names, tmp_path)
    again = native_thread_fields(PROBE, 1, names, tmp_path)
    four = native_thread_fields(PROBE, 4, names, tmp_path)
    for name in names:
        problem, _ = averaged(name, envelope=False, thg=True)
        bound = steps_bound(fixed_order(problem))
        assert_within(again[name], one[name], bound, what=f"{name} repeat")
        assert_within(four[name], one[name], bound, what=f"{name} four threads")


# --- the memory ledger -------------------------------------------------------------


def ledger_of(problem: KernelProblem, grid: Any) -> dict[str, int]:
    """Return the backend's ledger and check its rows are released on close."""
    backend = NativeBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    assert all(row in ledger for row in DISPERSIVE_ROWS)
    assert all(ledger[row] >= 0 for row in DISPERSIVE_ROWS)
    return ledger


def test_a_lone_symmetric_response_without_an_in_stage_absorber_holds_no_field():
    """SEP 6's memory target: the stage's own buffers, filtered in place."""
    simulation = capillary("symmetric", stepped=True)
    problem = simulation.translate()
    ledger = ledger_of(problem, simulation.grid)
    for row in DISPERSIVE_ROWS[:-1]:
        assert ledger[row] == 0, row
    n_bins = problem.grids.n_to // 2 + 1
    # Its input and output rows, their stage copies and its weights at least.
    assert ledger["dispersive_multipliers"] >= 5 * n_bins * 8


def test_a_lone_symmetric_mixture_holds_no_field_either():
    """A mixture is one product under its filter, so it takes the lone path too."""
    simulation = capillary("symmetric", gas={"He": 0.6, "Ar": 0.4}, stepped=True)
    problem = simulation.translate()
    ledger = ledger_of(problem, simulation.grid)
    for row in DISPERSIVE_ROWS[:-1]:
        assert ledger[row] == 0, row


def test_a_lone_symmetric_response_beside_the_collar_holds_one_field():
    """The in-stage absorber reads the bare modal field: one filtered field besides.

    Its spectrum is the stage's projected spectrum, free until the stage's forward
    transform.
    """
    simulation = capillary("symmetric")
    problem = simulation.translate()
    assert problem.responses.absorber_scheme == "nonlinear"
    ledger = ledger_of(problem, simulation.grid)
    n_columns, n_to = problem.shape[1] * problem.shape[2], problem.grids.n_to
    assert ledger["dispersive_filtered_time"] == n_columns * n_to * 8
    for row in ("dispersive_node_fields", "dispersive_products"):
        assert ledger[row] == 0, row
    assert ledger["dispersive_product_spectra"] == 0


def test_the_polynomial_holds_two_derivative_fields_and_its_products():
    """Two filtered fields, two node slots and three products per rule."""
    simulation = capillary("polynomial", thg=False)
    problem = simulation.translate()
    ledger = ledger_of(problem, simulation.grid)
    n_columns, n_to = problem.shape[1] * problem.shape[2], problem.grids.n_to
    field = n_columns * n_to * 16  # the analytic family: complex fields
    assert ledger["dispersive_filtered_time"] == 2 * field
    assert ledger["dispersive_products"] == 3 * field
    assert ledger["dispersive_coarse_products"] == 3 * field
    assert ledger["dispersive_product_spectra"] == 3 * field
    assert ledger["dispersive_node_fields"] == 2 * ledger["node_field"]
    # One chunk: the stage's scaled projections are read in place.
    assert ledger["dispersive_projection_chunks"] == 0


def test_a_response_beside_raman_holds_its_own_node_field_and_products():
    """The bare field stays the column's; the products are landed in its spectrum."""
    simulation = capillary("symmetric", gas="N2", raman=True, thg=False)
    problem = simulation.translate()
    ledger = ledger_of(problem, simulation.grid)
    n_columns, n_to = problem.shape[1] * problem.shape[2], problem.grids.n_to
    assert ledger["dispersive_node_fields"] == ledger["node_field"]
    assert ledger["dispersive_products"] == n_columns * n_to * 16
    assert ledger["dispersive_product_spectra"] == n_columns * n_to * 16


def test_chunked_nodes_pack_their_projections():
    """More than one chunk: each chunk's columns of both scaled projections, once."""
    simulation = capillary("polynomial")
    problem = simulation.translate()
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=8)
    )
    ledger = ledger_of(problem, simulation.grid)
    chunks = -(-problem.transform.n_nodes // 8)
    per_rule = chunks * problem.shape[1] * 8 * problem.transform.n_pol * 8
    assert ledger["dispersive_projection_chunks"] == 2 * per_rule


def test_a_problem_without_the_leaf_names_the_rows_at_zero():
    simulation = capillary("carrier-kerr")
    problem = simulation.translate()
    backend = NativeBackend(problem, grid=simulation.grid)
    try:
        ledger = backend.memory_ledger()
        metadata = backend.metadata()
    finally:
        backend.close()
    assert all(ledger[row] == 0 for row in DISPERSIVE_ROWS)
    assert metadata["dispersive_path"] is None
