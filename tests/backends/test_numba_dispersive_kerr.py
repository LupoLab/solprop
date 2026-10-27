# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 6's dispersive electronic response on ``numba_cpu``'s fused node path.

``numba_cpu`` evaluates ``responses.dispersive_kerr`` with its spectral multipliers on
the modal columns rather than at the nodes (``backends/numba_cpu/dispersive.py``);
``numpy_ref``, which applies them at every node, is the oracle. The ladder of
``docs/dev/backends.md`` section 3.7 for the response: one nonlinear evaluation (the
projected polarisation, on both quadrature rules), one step and ten steps, on the
multimode capillary, the mode-averaged and radial routes and a two-component field,
on field and envelope grids, in both sectors, for both evaluators; a two-species
mixture, of constant and of varying composition; Raman, plasma and envelope plasma
beside the response. Then what is this backend's own: agreement across thread counts
and repeated runs, at the window's rounding budget, and the memory ledger.

**The bounds are derived from summation lengths.** One evaluation of the constant
cubic is two transforms, ``16 eps ceil(log2 n_to)`` (``test_polarised_runs.py``), and
two matrix reductions over the nodes, ``gamma(n_nodes)`` each. The response adds
transforms and reductions per channel: the symmetric evaluator one inverse transform
of the filtered field and one forward transform of its product, one synthesis and one
projection per rule; the polynomial two inverse and three forward transforms, two
syntheses and three projections. The bounds still count the output inverse transform
the response had before its output joined the stage's spectrum (one per channel), so
they are a transform looser than they need be. The reference takes as many again at
every node, so each count is doubled. Each bound states what it was measured against.
"""

import dataclasses
import math
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from backends.matched import fixed_order
from bounds import assert_within, gamma, step_evaluations
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numba_cpu.dispersive import DISPERSIVE_LEDGER_ROWS
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.exceptions import MaterialRangeWarning
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.nonlinear.responses import DispersiveKerrResponse
from solprop.physdata import MaterialSelection, Mixture
from solprop.physdata.nonlinear_data import OptmatKerr
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from solprop.translate.problem import KernelProblem, assemble
from solprop.translate.stage import scalars

EPS = np.finfo(float).eps
LENGTH = 2e-3
STEPS = 10
ARGON = OptmatKerr("Ar").dispersive_model()
DENSITY = 4.9e25
"""Two bar of argon at room temperature, m^-3, for the routes built by hand."""

POLICIES = {"symmetric": "symmetric-kerr", "polynomial": "gas-polynomial"}
GRID_CASES: dict[str, dict[str, Any]] = {
    "field full": {"envelope": False, "thg": True},
    "field conserving": {"envelope": False, "thg": False},
    "envelope conserving": {"envelope": True, "thg": False},
    "envelope full": {"envelope": True, "thg": True},
}


# --- problems -------------------------------------------------------------------


def capillary(
    evaluator: str,
    *,
    gas: str | Mapping[str, float] = "Ar",
    stepped: bool = False,
    **overrides: object,
) -> Simulation:
    """Return a short three-mode argon capillary under a dispersive policy.

    Its quadrature has 33 nodes with an embedded rule. A stepped one takes the
    split-step absorber, whose collar is stable at a tenth of the guide, and ten fixed
    steps. Nitrogen's band starts at 180 nm, inside its index fit.
    """
    arguments: dict[str, Any] = {
        "lambda0": 800e-9,
        "wavelength_limits": (180e-9 if gas == "N2" else 150e-9, 4e-6),
        "trange": 150e-15,
        "fwhm": 10e-15,
        "energy": 30e-6,
        "modes": 3,
        "materials": MaterialSelection(electronic=POLICIES.get(evaluator, evaluator)),
        "plasma": False,
        "raman": False,
        "backend": "numpy_ref",
        "n_save": 2,
    }
    if stepped:
        arguments["absorber_scheme"] = "split_step"
        arguments["schedule"] = FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS)
    arguments |= overrides
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(75e-6, LENGTH, gas, 2.0, **arguments)


def grid_for(envelope: bool) -> FieldGrid | EnvelopeGrid:
    """Return the hand-built routes' grid, oversampled for an envelope's harmonic."""
    if envelope:
        return EnvelopeGrid.from_limits(
            LENGTH, 800e-9, (150e-9, 4e-6), 150e-15, thg=True
        )
    return FieldGrid.from_limits(LENGTH, 800e-9, (150e-9, 4e-6), 150e-15)


def averaged(
    evaluator: str, *, envelope: bool, thg: bool, vector: bool = False
) -> tuple[KernelProblem, FieldGrid | EnvelopeGrid]:
    """Return a mode-averaged argon problem, one node, or a two-component quadrature.

    The two-component field is the polarised route of ``test_polarised_runs.py``: one
    mode, a 24-point radial rule, an elliptical launch.
    """
    grid = grid_for(envelope)
    mode = MarcatiliMode(75e-6, cladding=1.45, loss=False)
    pulse = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=2e-6).build(grid)
    start = pulse
    quadrature = None
    if vector:
        angle, phase = math.pi / 5, math.pi / 3
        start = np.concatenate(
            [math.cos(angle) * pulse, math.sin(angle) * np.exp(1j * phase) * pulse],
            axis=1,
        )
        quadrature = transverse_quadrature([mode], n_r=24)
    problem = assemble(
        grid,
        [mode] if vector else mode,
        [DispersiveKerrResponse(ARGON, evaluator, thg=thg)],
        start,
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH / STEPS,
        quadrature=quadrature,
        density=DENSITY,
        absorber_scheme="split_step",
        integrator=Multistep(),
        schedule=FixedSchedule.uniform(0.0, LENGTH, LENGTH / STEPS),
    )
    return problem, grid


def radial(
    evaluator: str, *, envelope: bool, thg: bool
) -> tuple[KernelProblem, FieldGrid | EnvelopeGrid]:
    """Return a radial free-space problem: sixteen Hankel nodes, no embedded rule.

    On an envelope grid this is the backend's real radial GEMM.
    """
    if envelope:
        grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (300e-9, 2e-6), 100e-15, thg=True)
    else:
        grid = FieldGrid.from_limits(1e-4, 800e-9, (400e-9, 2e-6), 100e-15)
    space = RadialGrid(4e-4, 16, window_factor=0.0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 1e-4
    )
    start = pulse.build(grid, space, medium=SpatialMedium()) * grid.active_mask
    problem = assemble(
        grid,
        None,
        [DispersiveKerrResponse(ARGON, evaluator, thg=thg)],
        start,
        spatial_grid=space,
        medium=SpatialMedium(),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[1e-4],
        dz_max=1e-4,
        dz_init=1e-5,
        absorber_scheme="split_step",
        integrator=Multistep(),
        schedule=FixedSchedule.uniform(0.0, 1e-4, 1e-5),
        density=DENSITY,
    )
    return problem, grid


# --- the measurements ------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """One source evaluation's projected polarisation spectrum on both rules."""

    projected: np.ndarray
    coarse: np.ndarray | None


def evaluate(backend_class: type, problem: KernelProblem, grid: Any) -> Evaluation:
    """Evaluate the source once at the launch and return its projected spectrum.

    The reference keeps both rules' projections of its last evaluation in time; their
    unnormalised transforms are what this backend's stage holds in frequency, where
    the response joins every other's (``backends/numba_cpu/dispersive.py``). The
    embedded rule is finished when a step is accepted, which is done here by hand.
    """
    backend = backend_class(problem, grid=grid)
    forward = np.fft.rfft if problem.grid_kind == "field" else np.fft.fft
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._adams_lab_source(problem.initial.field0, 0.0)
        if isinstance(backend, NumpyRefBackend):
            coarse = backend._last_coarse
            return Evaluation(
                forward(np.asarray(backend._last_projected), axis=-1),
                None if coarse is None else forward(np.asarray(coarse), axis=-1),
            )
        plans = backend._plans
        if plans.coarse_to_frequency is not None:
            backend._coarse_spectrum(backend._last_bundle)
        coarse = plans.coarse_projected_spectrum
        return Evaluation(
            plans.projected_spectrum.copy(),
            None if plans.coarse_to_frequency is None else coarse.copy(),
        )
    finally:
        backend.close()


def run(
    backend_class: type, problem: KernelProblem, grid: Any, **options: object
) -> tuple[np.ndarray, Any]:
    """Propagate to the last save plane and return the field and the summary."""
    backend = backend_class(problem, grid=grid, **options)
    try:
        backend.set_initial(problem.initial.field0, None)
        summary = backend.propagate_to(float(problem.stepper.save_planes[-1]))
        return backend.observe().require_field().copy(), summary
    finally:
        backend.close()


def disagreement(ours: np.ndarray, theirs: np.ndarray) -> float:
    """Return the relative L2 difference of two arrays of the same size."""
    ours, theirs = np.ravel(ours), np.ravel(theirs)
    scale = float(np.linalg.norm(theirs))
    assert scale > 0
    return float(np.linalg.norm(ours - theirs) / scale)


def evaluation_bound(problem: KernelProblem) -> float:
    """Bound one evaluation's rounding: transforms and node reductions, both backends.

    The constant cubic's two transforms at ``8 eps ceil(log2 n_to)`` each and its two
    reductions over the nodes, plus the response's per channel (module docstring),
    doubled for the reference's node-by-node arithmetic.
    """
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None
    symmetric = leaf.evaluator == "symmetric"
    transforms = 2 + (3 if symmetric else 6) * leaf.n_channels
    reductions = 2 + (2 if symmetric else 5) * leaf.n_channels
    per_transform = 8 * EPS * math.ceil(math.log2(problem.grids.n_to))
    return 2 * (
        transforms * per_transform + reductions * gamma(problem.transform.n_nodes)
    )


def steps_bound(problem: KernelProblem) -> float:
    """Bound a replayed window of fixed steps: every evaluation's rounding, added."""
    return STEPS * step_evaluations(problem) * evaluation_bound(problem)


def assert_one_evaluation(
    problem: KernelProblem, grid: Any, baseline: KernelProblem | None = None
) -> None:
    """One evaluation, both rules, against the reference.

    With a `baseline` -- the same problem under the carrier policy -- the bound is
    what the other responses already disagree by plus the response's own rounding.
    """
    ours = evaluate(NumbaCpuBackend, problem, grid)
    theirs = evaluate(NumpyRefBackend, problem, grid)
    floor = (None, None)
    if baseline is not None:
        before = evaluate(NumbaCpuBackend, baseline, grid)
        reference = evaluate(NumpyRefBackend, baseline, grid)
        floor = (
            disagreement(before.projected, reference.projected),
            None
            if before.coarse is None or reference.coarse is None
            else disagreement(before.coarse, reference.coarse),
        )
    bound = evaluation_bound(problem)
    assert bound > 0
    assert disagreement(ours.projected, theirs.projected) < bound + (floor[0] or 0.0)
    assert (ours.coarse is None) == (theirs.coarse is None)
    if ours.coarse is not None and theirs.coarse is not None:
        assert disagreement(ours.coarse, theirs.coarse) < bound + (floor[1] or 0.0)


def assert_steps(problem: KernelProblem, grid: Any) -> None:
    """A replayed window of fixed steps, the same steps, against the reference."""
    ours, summary = run(NumbaCpuBackend, problem, grid)
    theirs, reference = run(NumpyRefBackend, problem, grid)
    assert summary.n_steps == reference.n_steps
    bound = steps_bound(problem)
    assert bound > 0
    assert disagreement(ours, theirs) < bound


# --- the ladder --------------------------------------------------------------------


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_one_evaluation_on_the_multimode_capillary_matches_the_reference(
    case, evaluator
):
    """Both rules' projected polarisation, under the in-stage absorber.

    The response holds its own filtered field beside the bare one the collar reads.
    Measured 2.3-4.0 eps on the fine rule and 2.1-3.9 on the embedded one against a
    bound of about 1600 (symmetric) and 2400 eps (polynomial).
    """
    simulation = capillary(evaluator, **GRID_CASES[case])
    problem = simulation.translate()
    assert problem.transform.n_nodes == 33
    assert_one_evaluation(problem, simulation.grid)


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_one_step_on_the_multimode_capillary_matches_the_reference(case, evaluator):
    """One Runge-Kutta step, seven evaluations through the whole stage.

    Bound: the seven evaluations' rounding added, 7400-13000 eps; measured 0.9-1.7 eps.
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
    ours, summary = run(NumbaCpuBackend, problem, simulation.grid)
    theirs, _ = run(NumpyRefBackend, problem, simulation.grid)
    assert summary.n_steps == 1
    bound = step_evaluations(problem) * evaluation_bound(problem)
    assert bound > 0
    assert disagreement(ours, theirs) < bound


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("sector", ["field full", "field conserving"])
def test_the_quadrature_alarm_is_the_reference_s(sector, evaluator):
    """The statistic the embedded rule exists for, after ten steps.

    The alarm is the relative distance between the two rules' projections of the last
    stage's polarisation, which here holds the response's share, finished lazily on
    the accepted step: a response missing from the embedded projection would put the
    alarm near one. Both backends' projections agree to the evaluation bound and the
    states to the window's, so the alarms differ by at most their sum. Measured: alarms
    of 5e-15 (conserving) and 7e-13 (full cubic) agreeing to 3.4e-17.
    """
    simulation = capillary(evaluator, stepped=True, **GRID_CASES[sector])
    problem = simulation.translate()
    spec = StatsSpec(names=("quadrature_error",))
    alarms = []
    for backend_class in (NumbaCpuBackend, NumpyRefBackend):
        backend = backend_class(problem, grid=simulation.grid)
        try:
            backend.set_initial(problem.initial.field0, None)
            backend.propagate_to(LENGTH)
            alarms.append(float(backend.stats(spec)["quadrature_error"][0]))
        finally:
            backend.close()
    ours, theirs = alarms
    assert theirs > 0
    bound = steps_bound(problem) + evaluation_bound(problem)
    assert abs(ours - theirs) < bound


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_ten_steps_on_the_multimode_capillary_match_the_reference(case, evaluator):
    """The default integrator over a replayed window of ten steps.

    Under the split-step absorber, so a lone symmetric response filters the stage's own
    spectrum in place. Measured 3.0-8.2 eps.
    """
    simulation = capillary(evaluator, stepped=True, **GRID_CASES[case])
    assert_steps(simulation.translate(), simulation.grid)


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

    A real product shares the backend's chunk scratch; a complex one beside a real
    polarisation holds its own.
    """
    simulation = capillary(
        evaluator, gas="N2" if raman else "Ar", raman=raman, **GRID_CASES[case]
    )
    problem = simulation.translate()
    problem = dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=8)
    )
    assert_one_evaluation(problem, simulation.grid)


@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("thg", [True, False])
def test_a_state_with_zero_frequency_content_matches_the_reference(thg, evaluator):
    """A field grid's state carries a zero-frequency bin, which Pi_+ leaves out.

    The active band excludes it, but the split-step absorber's product in time puts
    some there at every accepted step. The backend's analytic family keeps it in the
    bare analytic signal, which the polynomial reads, so the response subtracts its
    constant at the nodes; without that this disagrees at 1e-6.
    """
    simulation = capillary(evaluator, thg=thg)
    problem = simulation.translate()
    start = problem.initial.field0.copy()
    start[..., 0] = 1e-3 * np.abs(start).max()
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=start)
    )
    assert_one_evaluation(problem, simulation.grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", list(GRID_CASES))
def test_the_mode_averaged_route_matches_the_reference(case, evaluator):
    """One node: one evaluation and ten steps."""
    problem, grid = averaged(evaluator, **GRID_CASES[case])
    assert problem.transform.n_nodes == 1
    assert_one_evaluation(problem, grid)
    assert_steps(problem, grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", ["field full", "field conserving", "envelope full"])
def test_a_two_component_field_matches_the_reference(case, evaluator):
    """The isotropic Kleinman contraction of an elliptical launch, on the quadrature."""
    problem, grid = averaged(evaluator, vector=True, **GRID_CASES[case])
    assert problem.transform.n_pol * problem.shape[2] == 2
    assert_one_evaluation(problem, grid)
    assert_steps(problem, grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("envelope", [False, True])
def test_the_radial_route_matches_the_reference(envelope, evaluator):
    """The Hankel route; on an envelope grid, the backend's real radial GEMM."""
    problem, grid = radial(evaluator, envelope=envelope, thg=envelope)
    assert problem.transform.n_nodes == 16
    assert_one_evaluation(problem, grid)
    assert_steps(problem, grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("case", ["field full", "envelope conserving"])
def test_a_mixture_matches_the_reference(case, evaluator):
    """Helium and argon, two channels with their own multipliers and leg bounds.

    The symmetric evaluator takes one product under the mixture filter, built from the
    channels' densities at each stage by the reference's arithmetic. The polynomial's
    ``E^3`` is shared by the channels and weighted in the spectrum by both channels'
    output multipliers; every other product is per channel.
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
    """Helium falling and argon rising along the guide: the filter is rebuilt per stage.

    Independent partial pressures, 2 to 0.5 bar of helium and 0.5 to 2 bar of argon, so
    the composition, and with it the symmetric evaluator's mixture filter, changes at
    every stage; ten steps against the reference, which rebuilds it per evaluation.
    """
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
            lambda0=800e-9,
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
    # Four to one and back, but for the gases' compressibility.
    assert start[0] / start[1] == pytest.approx(4.0, rel=1e-2)
    assert end[0] / end[1] == pytest.approx(0.25, rel=1e-2)
    assert_steps(problem, simulation.grid)


@pytest.mark.medium
@pytest.mark.parametrize("case", ["field full", "field conserving", "envelope full"])
def test_species_with_one_leg_bound_share_their_derivative_fields(case):
    """Krypton and xenon: one leg bound, so one group of the polynomial.

    Their coefficient tables add, density-weighted, over one ``D E`` and one
    ``D^2 E`` -- an exact rewriting of the per-species sum, which the reference
    evaluates; the backend agrees with it to the evaluation and step bounds.
    """
    # Xenon's index fit starts at 148 nm; a sixth of the argon runs' energy keeps
    # the heavier gases' ten fixed steps stable.
    band = {"wavelength_limits": (180e-9, 4e-6), "energy": 5e-6}
    simulation = capillary(
        "polynomial", gas={"Kr": 0.5, "Xe": 0.5}, **band, **GRID_CASES[case]
    )
    problem = simulation.translate()
    leaf = problem.responses.dispersive_kerr
    assert leaf is not None and leaf.n_channels == 2
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        workspace = backend._dispersive
        assert workspace is not None
        assert workspace.terms.groups == ((0, 1),)
        assert workspace.n_fields == 2 and workspace.n_products == 3
    finally:
        backend.close()
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
    """Nitrogen's Raman response beside it: the response must not filter the drive.

    The field grid's conserving sector puts both on the analytic family, whose Raman
    drive is the cycle-averaged one.
    """
    simulation = capillary(evaluator, gas="N2", raman=True, thg=thg)
    problem = simulation.translate()
    assert problem.flags.has_raman
    assert_one_evaluation(problem, simulation.grid)
    stepped = capillary(evaluator, gas="N2", raman=True, thg=thg, stepped=True)
    assert_steps(stepped.translate(), stepped.grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("envelope", [False, True])
def test_plasma_beside_the_response_reads_the_bare_field(envelope, evaluator):
    """The field grid's plasma column, or the envelope's own current streams.

    A field run with plasma takes the conserving sector, the analytic family. The
    plasma column's own scan already disagrees with the reference's, by 916 eps on the
    fine rule here, so the bound is the carrier-policy run's disagreement plus the
    response's. A plasma current filtered by the response would be wrong by the
    multiplier's departure from one, about 1e-3, far above either.
    """
    simulation = capillary(
        evaluator, plasma=True, thg=False, envelope=envelope, energy=60e-6
    )
    problem = simulation.translate()
    assert (problem.responses.envelope_plasma is not None) == envelope
    baseline = capillary(
        "carrier-kerr", plasma=True, thg=False, envelope=envelope, energy=60e-6
    ).translate()
    assert_one_evaluation(problem, simulation.grid, baseline)
    stepped = capillary(
        evaluator,
        plasma=True,
        thg=False,
        envelope=envelope,
        energy=60e-6,
        stepped=True,
    )
    assert_steps(stepped.translate(), stepped.grid)


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
def test_runge_kutta_steps_match_the_reference(evaluator):
    """The Runge-Kutta march: seven stages a step, the first recycled."""
    simulation = capillary(evaluator, stepped=True, integrator=RungeKutta())
    problem = simulation.translate()
    assert problem.stepper.integrator is None
    assert_steps(problem, simulation.grid)


# --- what is this backend's alone ---------------------------------------------------


@pytest.mark.medium
@pytest.mark.parametrize("evaluator", sorted(POLICIES))
@pytest.mark.parametrize("raman", [False, True])
def test_one_thread_and_four_agree(thread_budget_lifted, raman, evaluator):
    """The parallel node loops, forced, against the serial ones, and two runs alike.

    The same fixed-order method on the same fixed schedule, so the runs differ by the
    order of their threaded reductions alone: the window's rounding budget
    (:func:`steps_bound`), as against the reference.
    """
    simulation = capillary(
        evaluator, gas="N2" if raman else "Ar", raman=raman, stepped=True
    )
    problem = fixed_order(simulation.translate())
    serial, one = run(
        NumbaCpuBackend, problem, simulation.grid, threads=1, regime="large"
    )
    parallel, many = run(
        NumbaCpuBackend, problem, simulation.grid, threads=4, regime="large"
    )
    again, _ = run(NumbaCpuBackend, problem, simulation.grid, threads=4, regime="large")
    assert one.n_steps == many.n_steps
    bound = steps_bound(problem)
    assert_within(parallel, serial, bound, what="four threads")
    assert_within(again, parallel, bound, what="repeat run")


def ledger_of(problem: KernelProblem, grid: Any) -> dict[str, int]:
    """Return the backend's ledger and check it is released on close."""
    backend = NumbaCpuBackend(problem, grid=grid)
    try:
        ledger = backend.memory_ledger()
        layout = backend._dispersive
        assert layout is not None
    finally:
        backend.close()
    closed = backend.memory_ledger()
    assert all(closed[row] == 0 for row in DISPERSIVE_LEDGER_ROWS)
    assert all(ledger[row] >= 0 for row in DISPERSIVE_LEDGER_ROWS)
    return ledger


def test_a_lone_symmetric_response_without_an_in_stage_absorber_holds_no_field():
    """SEP 6's memory target met: the stage's own buffers and its spectrum, in place.

    Real products project into the stage's projected buffer and the filter acts on
    the stage's spectrum, so no field-sized buffer is added.
    """
    simulation = capillary("symmetric", stepped=True)
    problem = simulation.translate()
    ledger = ledger_of(problem, simulation.grid)
    for row in (
        "dispersive_filtered_time",
        "dispersive_node_fields",
        "dispersive_products",
        "dispersive_coarse_products",
        "dispersive_product_spectra",
        "dispersive_chunk_scratch",
        "dispersive_synthesis",
    ):
        assert ledger[row] == 0, row
    # The multipliers: one input and one output row of the one-sided layout.
    n_bins = problem.grids.n_to // 2 + 1
    assert ledger["dispersive_multipliers"] >= 2 * n_bins * 8


def test_a_lone_symmetric_mixture_holds_no_field_either():
    """A mixture is one product under its filter, so it takes the lone path too."""
    simulation = capillary("symmetric", gas={"He": 0.6, "Ar": 0.4}, stepped=True)
    problem = simulation.translate()
    ledger = ledger_of(problem, simulation.grid)
    for row in (
        "dispersive_filtered_time",
        "dispersive_node_fields",
        "dispersive_products",
        "dispersive_product_spectra",
    ):
        assert ledger[row] == 0, row


def test_a_lone_symmetric_response_beside_the_collar_holds_one_field():
    """The in-stage absorber reads the bare modal field: one filtered field besides."""
    simulation = capillary("symmetric")
    problem = simulation.translate()
    assert problem.responses.absorber_scheme == "nonlinear"
    ledger = ledger_of(problem, simulation.grid)
    n_columns, n_to = problem.shape[1] * problem.shape[2], problem.grids.n_to
    assert ledger["dispersive_filtered_time"] == n_columns * n_to * 8
    assert ledger["dispersive_products"] == ledger["dispersive_node_fields"] == 0


def test_the_polynomial_holds_two_derivative_fields_and_its_products():
    """Two filtered fields, two node buffers and three products per rule.

    Alone, its summed output lands in the stage's own spectrum and projected buffer.
    """
    simulation = capillary("polynomial", thg=False)
    problem = simulation.translate()
    ledger = ledger_of(problem, simulation.grid)
    n_columns, n_to = problem.shape[1] * problem.shape[2], problem.grids.n_to
    field = n_columns * n_to * 16  # the analytic family: complex fields
    assert ledger["dispersive_filtered_time"] == 2 * field
    assert ledger["dispersive_products"] == 3 * field
    assert ledger["dispersive_coarse_products"] == 3 * field
    assert ledger["dispersive_product_spectra"] == 3 * field
    # The cube, then the two derivative fields, share two buffers of the chunk's size.
    assert ledger["dispersive_node_fields"] == 2 * ledger["node_field"]


def test_a_response_beside_raman_holds_its_own_node_field_and_output():
    """The bare field stays the column's; the output is added to the projection."""
    simulation = capillary("symmetric", gas="N2", raman=True, thg=False)
    problem = simulation.translate()
    ledger = ledger_of(problem, simulation.grid)
    n_columns, n_to = problem.shape[1] * problem.shape[2], problem.grids.n_to
    assert ledger["dispersive_node_fields"] == ledger["node_field"]
    assert ledger["dispersive_products"] == n_columns * n_to * 16
    assert ledger["dispersive_product_spectra"] == n_columns * n_to * 16


def test_a_problem_without_the_leaf_names_the_rows_at_zero():
    simulation = capillary("carrier-kerr")
    problem = simulation.translate()
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        ledger = backend.memory_ledger()
        metadata = backend.metadata()
    finally:
        backend.close()
    assert all(ledger[row] == 0 for row in DISPERSIVE_LEDGER_ROWS)
    assert metadata["dispersive_alone"] is None


def test_the_metadata_records_whether_the_response_ran_alone():
    for raman, alone in ((False, True), (True, False)):
        simulation = capillary("symmetric", gas="N2", raman=raman)
        problem = simulation.translate()
        backend = NumbaCpuBackend(problem, grid=simulation.grid)
        try:
            assert backend.metadata()["dispersive_alone"] is alone
        finally:
            backend.close()


@pytest.mark.parametrize("evaluator", ["polynomial", "symmetric"])
def test_the_embedded_rule_is_finished_once_and_only_where_it_exists(evaluator):
    """A second finish of the same evaluation, or one without the rule, does nothing.

    The embedded products are transformed once per evaluation, when the step is
    accepted; a repeated call must not land them twice. The radial route has no
    embedded rule, so finishing and combining it leave the stage untouched. Both are
    explicit no-ops, so exact.
    """
    simulation = capillary(evaluator)
    problem = simulation.translate()
    backend = NumbaCpuBackend(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._adams_lab_source(problem.initial.field0, 0.0)
        workspace, bundle = backend._dispersive, backend._last_bundle
        assert workspace is not None and workspace.coarse is not None
        assert bundle is not None
        backend._coarse_spectrum(bundle)
        landed = backend._plans.coarse_projected_spectrum.copy()
        workspace.finish_coarse(bundle)
        np.testing.assert_array_equal(backend._plans.coarse_projected_spectrum, landed)
    finally:
        backend.close()
    radial_problem, grid = radial(evaluator, envelope=False, thg=True)
    backend = NumbaCpuBackend(radial_problem, grid=grid)
    try:
        backend.set_initial(radial_problem.initial.field0, None)
        backend._adams_lab_source(radial_problem.initial.field0, 0.0)
        workspace, bundle = backend._dispersive, backend._last_bundle
        assert workspace is not None and workspace.coarse is None
        assert bundle is not None
        before = backend._plans.projected_spectrum.copy()
        workspace.finish_coarse(bundle)
        workspace.combine_coarse()
        np.testing.assert_array_equal(backend._plans.projected_spectrum, before)
    finally:
        backend.close()
