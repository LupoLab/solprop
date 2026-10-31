# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A moving solid guide, end to end: its route, its doors and its physics.

A step-index or graded-index guide whose geometry moves along z propagates on the
general modal route with its own mode fields at every position: the synthesis and
projection maps expanded along z on the path's sections and combined per stage (kernel
schema 14's ``StageScalars.transform_weights``), and the connection ``Gamma = W^-1 S``
in the ``"frozen_residual"`` coupled flow beside the generator. ``numpy_ref`` executes
it, and so do ``numba_cpu`` (``tests/backends/test_numba_moving_guide.py``) and Kokkos
(``tests/driver/test_native_moving_guide.py``).

The default tier runs the **graded-index** taper, whose solves take milliseconds. The
step-index validations -- the dense reference that re-solves the guide at every
position, the single channel against the ``SpecifiedMode`` path and the complex basis in
which the connection's conjugation matters -- are ``slow``. Every bound is stated before
the number it bounds, and no reference used here contains the route's formula: the dense
reference integrates the coupled-mode equations by DOP853 from fresh solves and a
separately sampled connection, the adiabatic limit is a quadrature of fresh propagation
constants, and the ``SpecifiedMode`` path is the scalar model of the same fibre.

**No run here is longer than a millimetre**, and the default-tier runs are ten steps;
the refinement studies are linear single-bin runs of a few seconds each.
"""

import dataclasses
import re
from typing import cast

import numpy as np
import pytest
from scipy.integrate import simpson, solve_ivp

from modal_provider_cases import (
    GENERAL_CHI3,
    general_guide,
    grin_grid,
    grin_provider,
    make_provider,
)
from solprop.backends import select
from solprop.backends.memory import estimate_memory, ledger_groups
from solprop.backends.numpy_ref.admission import refuse_general_route
from solprop.backends.numpy_ref.backend import NumpyRefBackend
from solprop.backends.numpy_ref.batch import NumpyBatchBackend
from solprop.constants import C
from solprop.driver import Simulation, prop_gnlse
from solprop.exceptions import (
    BackendError,
    DriverError,
    ModeError,
    OutputError,
    SurrogateAreaWarning,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import StepIndexMode
from solprop.modes.connection import modal_connection
from solprop.modes.guide import (
    ModalGuide,
    ProviderChannelMode,
    circular_pair,
    provider_channels,
)
from solprop.modes.material import MaterialKerr
from solprop.nonlinear.gnlse import GnlseProblem, SpecifiedMode, _warn_surrogate_area
from solprop.nonlinear.normalisation import modal_kerr_coefficient
from solprop.output.hdf5 import check_cache, resume
from solprop.output.memory import MemoryOutput
from solprop.output.schema import ModeRecord
from solprop.processing.modal import (
    _plane_reference,
    _plane_rule,
    electric_field,
    magnetic_field,
    plane_nodes,
    reconstructed_flux,
)
from solprop.profiles import Linear, PiecewiseLinear, ZProfile
from solprop.stats import analytic_field
from solprop.stepper import FixedSchedule, RungeKutta
from solprop.translate.coupled import flow_weights, residual_truncation
from solprop.translate.general import general_modal_route
from solprop.translate.modal_path import path_rule, path_snapshot
from solprop.translate.modal_tables import apply_channel_basis, channel_area
from solprop.translate.problem import MOVING_FLOW_CERTIFICATE, _carrier
from solprop.translate.stage import scalars

LENGTH = 1e-3
"""Length of every run here, m: a millimetre."""

STEPS = 10
"""Steps of every default-tier run."""

KINK = 4e-4
"""The kinked taper's corner, m: a multiple of ``LENGTH / STEPS``."""

GRIN_PULSE = {
    "wavelength": 1.03e-6,
    "fwhm": 60e-15,
    "energy": 1e-12,
    "shape": "gauss",
}


def grin_knots(count=13):
    """Return knots over the graded-index grid's active band, in rad/s."""
    grid = grin_grid()
    active = grid.omega[grid.active_mask]
    return np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), count)


def grin_taper(radius, *, kerr=False, sections=1, knots=13):
    """Return the graded-index sector on a declared radius path, as a guide."""
    provider = dataclasses.replace(
        grin_provider(
            elements=32, kerr=MaterialKerr(chi3=GENERAL_CHI3) if kerr else None
        ),
        radius=radius,
    )
    frequencies = grin_knots(knots)
    return ModalGuide.from_provider(
        provider,
        frequencies,
        nodes=provider.node_rule(float(frequencies[0]), 0.0, n_r=2),
        sections=sections,
    )


def kinked_radius():
    """Return the kinked taper: 25 um, 24 um by the corner, 22.5 um at the end."""
    return PiecewiseLinear(
        np.array([0.0, KINK, LENGTH]), np.array([25e-6, 24e-6, 22.5e-6])
    )


def build(guide, grid, *, steps=STEPS, length=LENGTH, **overrides):
    """Return a general modal simulation of a guide on a fixed schedule."""
    simulation = Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grid,
        **(
            {
                "noise": None,
                **{
                    "save_planes": [length],
                    "dz_max": length / steps,
                    "dz_init": length / steps,
                    "schedule": FixedSchedule(np.full(steps, length / steps)),
                    "backend": "numpy_ref",
                }
                | GRIN_PULSE
                | overrides,
            }
        ),
    )
    # The subject is the moving guide; the 60 fs pulse fills its window to the edge.
    simulation.settings = dataclasses.replace(simulation.settings, boundary_check=None)
    return simulation


@pytest.fixture(scope="module")
def kinked():
    """The kinked graded-index taper with a Kerr core."""
    return grin_taper(kinked_radius(), kerr=True)


@pytest.fixture(scope="module")
def kinked_problem(kinked):
    """Its ten-step problem, with no absorber, and the grid."""
    grid = grin_grid()
    simulation = build(kinked, grid)
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    return simulation, simulation.translate()


def test_the_problem_carries_moving_maps_a_frozen_flow_and_the_kink(kinked_problem):
    """What crosses the seam for a moving guide, and nothing it should not.

    Two sections, one per kink interval, carry the maps and the flow on the same
    edges; the flow is ``"frozen_residual"`` on each with the connection present;
    ``Phi`` is the one-term expansion of the reference channel's entrance dispersion,
    a multiple of the identity in the channel axis, and its one hard kink is the
    taper's corner. A stage's bundle carries the maps' weights beside the flow's.
    """
    _, problem = kinked_problem
    tables = problem.transform.modal_tables
    flow = problem.linear.coupled_flow
    assert tables is not None and flow is not None
    assert tables.synthesis is not None
    assert tables.synthesis.shape[:2] == (2, 5)
    assert flow.section_methods == ("frozen_residual", "frozen_residual")
    assert flow.connection is not None
    assert flow.certificate_id == MOVING_FLOW_CERTIFICATE
    assert np.array_equal(flow.z_edges, tables.z_edges)
    assert problem.linear.phi_representation == "expansion"
    assert problem.linear.hard_kinks.tolist() == [KINK]
    phi = problem.linear.phi_expansion
    assert phi is not None
    assert np.array_equal(phi.terms[:, :, 0], phi.terms[:, :, 1])
    bundle = scalars(problem, 0.0, 0.55 * LENGTH)
    assert bundle.transform_weights is not None
    assert bundle.transform_weights.shape == (10,)
    assert not np.any(bundle.transform_weights[:5])
    assert bundle.flow_weights is not None
    assert np.array_equal(
        bundle.flow_weights, flow_weights(flow, 0.55 * LENGTH, bundle.flow_section)
    )
    assert not problem.transform.modal_tables.frozen_rule
    identity = tables.identity
    assert identity["map_degree"] == 4
    assert identity["map_z_edges"] == [0.0, KINK, LENGTH]


@pytest.mark.medium
def test_a_moving_guide_runs_ten_steps_end_to_end_and_records_its_path(
    kinked, kinked_problem
):
    """``Simulation.from_waveguide(provider=...)``, ten steps on ``numpy_ref``.

    With no absorber the guide is lossless and the connection conserves the flux
    norm exactly in the continuous limit (it is anti-Hermitian), so what the run loses
    of its physical energy is the flow's own ten-step truncation (:func:`_truncation`);
    the Kerr term's step error at a picojoule is far below it. The record says what the
    guide was: its path's sections and degree, and a flow labelled a moving guide's.
    The truncation is the Runge-Kutta march's frozen-residual bound, so the march is
    named; :func:`test_a_moving_guide_conserves_its_energy_under_the_default` runs the
    default.
    """
    simulation = build(kinked, grin_grid(), integrator=RungeKutta())
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    problem = simulation.translate()
    run = cast("MemoryOutput", simulation.run(progress="none"))
    energy = [float(np.sum(plane.stats["physical_energy"])) for plane in run.planes()]
    assert len(energy) == 2
    drift = abs(energy[-1] / energy[0] - 1.0)
    # The Kerr term conserves energy exactly in the continuous limit and at 1 pJ its
    # step error is far below the flow's, so the flow's own bound is the budget.
    assert drift < _truncation(problem)
    modal = simulation.metadata().modal
    assert modal is not None
    guide = modal["guide"]
    assert isinstance(guide, dict)
    path = guide["sampling"]["path"]
    assert path["z_edges"] == [0.0, KINK, LENGTH]
    assert path["degree"] == 4
    flow = modal["flow"]
    assert isinstance(flow, dict)
    assert flow["model"] == "moving guide"
    assert flow["connection"] is True
    assert 0.0 <= flow["connection_certificate"] < 1e-6
    sampling = kinked.sampling
    assert sampling is not None and sampling.path is not None
    # The field accessors rebuild a plane on its own modes at its own nodes, which
    # the reconstruction tests below check against a constant guide of that shape.
    tables = general_modal_route(
        kinked, grin_grid(), z_max=LENGTH, omega_reference=float(grin_grid().omega0)
    ).tables
    nodes = path_rule(kinked.field_provider, sampling.nodes, LENGTH).weights.size
    assert electric_field(run, tables, plane=-1, guide=kinked).shape == (
        nodes,
        3,
        grin_grid().n_t,
    )


def test_every_compiled_backend_admits_moving_maps_and_a_new_one_refuses_them(
    kinked_problem,
):
    """Both compiled backends admit the moving maps; the default refuses them by name.

    A backend that calls the shared admission without declaring ``moving_maps``
    has not built the per-stage combination, and is refused by name.
    """
    simulation, problem = kinked_problem
    refuse_general_route(problem, backend="numpy_ref", moving_maps=True)
    with pytest.raises(BackendError, match=r"new executes .*moving maps.*not build"):
        refuse_general_route(problem, backend="new")
    select("numba_cpu", problem, grid=simulation.grid).close()
    pytest.importorskip("solprop_native")
    select("native", problem, grid=simulation.grid).close()


def test_the_reference_refuses_a_cf4_flow_beside_moving_maps(kinked_problem):
    """CF4 composes on paper and is not yet built or measured on this route.

    CF4 is the Runge-Kutta march's section method: the default Adams history refuses
    it before the route is reached.
    """
    simulation, problem = kinked_problem
    problem = dataclasses.replace(
        problem, stepper=dataclasses.replace(problem.stepper, integrator=None)
    )
    flow = problem.linear.coupled_flow
    assert flow is not None
    spoiled_flow = dataclasses.replace(flow)
    object.__setattr__(spoiled_flow, "section_methods", ("cf4", "cf4"))
    linear = dataclasses.replace(problem.linear)
    object.__setattr__(linear, "coupled_flow", spoiled_flow)
    spoiled = dataclasses.replace(problem)
    object.__setattr__(spoiled, "linear", linear)
    with pytest.raises(BackendError, match="CF4 beside moving maps"):
        NumpyRefBackend(spoiled, grid=simulation.grid)


def test_a_lockstep_batch_refuses_a_moving_guide(kinked_problem):
    """The batch applies neither half of a flow, so a member carrying one is refused."""
    simulation, problem = kinked_problem
    member = NumpyRefBackend(
        problem, grid=simulation.grid, modes=list(simulation.modes), profiles=None
    )
    with pytest.raises(BackendError, match="lockstep batch does not execute"):
        NumpyBatchBackend([member])
    member.close()


def test_the_modified_noise_model_is_refused_beside_a_moving_guide(kinked):
    """A kept-apart draw's transport through the moving flow is not measured.

    The input draw is admitted (``tests/driver/test_coupled_input_noise.py``).
    """
    with pytest.raises(TranslationError, match="refuses the 'modified' noise model"):
        build(kinked, grin_grid(), noise="modified", seed=3).translate()


def test_a_fixed_schedule_straddling_the_kink_is_refused(kinked):
    """Seven steps over the millimetre do not land on the corner at 0.4 mm."""
    with pytest.raises((DriverError, TranslationError), match="hard kink"):
        build(kinked, grin_grid(), steps=7).translate()


def test_a_run_longer_than_the_declared_path_is_refused(kinked):
    """The geometry is declared over the provider's z domain and not past it."""
    with pytest.raises((DriverError, TranslationError, Exception), match="path"):
        build(kinked, grin_grid(z_end=2 * LENGTH), length=2 * LENGTH).translate()


def test_the_memory_estimate_counts_the_moving_maps_to_the_byte(kinked_problem):
    """Sections times degree of each map, and the two a stage evaluates."""
    simulation, problem = kinked_problem
    estimate = estimate_memory(problem, "numpy_ref")
    backend = select("numpy_ref", problem, grid=simulation.grid)
    try:
        ledger = backend.memory_ledger()
        held = ledger_groups(ledger)
    finally:
        backend.close()
    for group, value in held.items():
        if group != "measured":
            assert estimate.groups[group] == value, group
    tables = problem.transform.modal_tables
    assert tables is not None and tables.synthesis is not None
    one = tables.synthesis[0, 0].nbytes
    assert ledger["general_synthesis"] == 10 * one
    assert ledger["general_stage_maps"] == 2 * one


@pytest.mark.medium
def test_a_resume_across_a_changed_path_is_refused_by_the_field_that_moved(
    kinked, tmp_path
):
    """Two sections per kink interval is a different guide, and the file says so."""
    grid = grin_grid()
    written = tmp_path / "moving.h5"
    first = build(kinked, grid)
    first.settings = dataclasses.replace(first.settings, absorber=None)
    first.run(written, progress="none")
    stored = resume(written).metadata
    other = build(grin_taper(kinked_radius(), kerr=True, sections=2), grid)
    other.settings = dataclasses.replace(other.settings, absorber=None)
    metadata = other.metadata()
    changed = stored.differences(metadata)
    assert any(one.startswith("modal.guide") for one in changed), changed
    assert any(one.startswith("modal.tables") for one in changed), changed
    with pytest.raises(OutputError, match="these fields changed"):
        check_cache(written, metadata)


def test_a_moving_guide_s_channels_are_the_guide_s_own_and_a_mode_set_is_refused(
    kinked,
):
    """``modes=`` beside a provider is empty or the guide's own, as on any guide."""
    grid = grin_grid()
    with pytest.raises(DriverError, match="supplies its own"):
        Simulation.from_waveguide(
            [kinked.mode_collection.modes[0]],
            LENGTH,
            provider=kinked,
            grid=grid,
            save_planes=[LENGTH],
            **{"noise": None, **GRIN_PULSE},
        )


# ------------------------------------------------------------ the adiabatic limit


def _single_bin(guide, grid, length, steps, index, **overrides):
    """Return a linear single-bin run's final channel amplitudes, launched in one.

    The launch is ``1e-9`` sqrt(W) in channel 0 at one bin: linear, a continuous wave
    in time, with the collar removed so that no absorber touches it: the temporal
    absorber is a convolution in frequency and would attenuate the channels' spectra
    differently.
    """
    simulation = build(guide, grid, steps=steps, length=length, **overrides)
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    problem = simulation.translate()
    state = np.zeros_like(problem.initial.field0)
    state[0, 0, 0, index] = 1e-9
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=state)
    )
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(length)
    final = backend.observe().require_field()[0, :, 0, index].copy()
    backend.close()
    return final / 1e-9, problem


def _truncation(problem):
    """Return what ten steps may move the flux norm by, from the leaf's own rows.

    The connection conserves the flux norm exactly where it is anti-Hermitian (the split
    block's identity with ``W = I``), so a run departs from it by two things: the
    explicit residual's local frozen defect, ``residual_truncation(h K cond)`` per step
    with ``K`` and ``cond`` the flow's certificate rows (the frozen-residual bound), and
    the stored connection's own departure from anti-Hermitian -- its measured
    certificate carried through the expansions -- enclosed by ``sum_k ||c_k + c_k^H||``
    over the Chebyshev coefficients (``|T_k| <= 1``) and applied over the length, twice,
    as it moves ``|b|^2``.
    """
    flow = problem.linear.coupled_flow
    scale = float(np.max(flow.residual_norm_bound * flow.inverse_condition_bound))
    step = float(problem.stepper.dz_max)
    link = flow.connection
    hermitian = np.linalg.norm(
        np.moveaxis(link + np.conj(np.swapaxes(link, 2, 3)), -1, 2),
        ord=2,
        axis=(-2, -1),
    )
    enclosure = float(np.max(np.sum(hermitian, axis=1)))
    length = STEPS * step
    return STEPS * residual_truncation(step * scale) + 2.0 * length * enclosure


def _beta_departure(guide, omega, positions):
    """Return each channel's tabulated propagation constant less a fresh solve's, 1/m.

    What the route's own frequency interpolation puts into ``beta`` at each position,
    against the provider asked at the bin itself.
    """
    sampling = guide.sampling
    provider = guide.field_provider
    rows = []
    for z in positions:
        rule = path_rule(provider, sampling.nodes, float(z))
        table = path_snapshot(provider, sampling.omega, float(z), rule)
        tabulated = np.diag(table.evaluate(omega, float(z)).generator)
        fresh = np.diag(provider.solve(omega, float(z), nodes=rule).generator)
        rows.append(np.real(1j * (tabulated - fresh)))
    return np.asarray(rows)


@pytest.mark.medium
def test_a_moving_guide_conserves_its_energy_under_the_default(kinked):
    """The kinked taper, adaptive under the default integrator, to its tolerance.

    Lossless, so the run's physical energy moves by its integration error alone: the
    Kerr source carries the frozen residual's compensation into the Adams samples
    (ADR-042, ADR-044), where the controller scores it. Measured 5.2e-11 in 103
    steps at ``rtol=1e-8``; the bound is twenty times that.
    """
    simulation = build(kinked, grin_grid(), schedule=None, rtol=1e-8)
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    run = cast("MemoryOutput", simulation.run(progress="none"))
    energy = [float(np.sum(plane.stats["physical_energy"])) for plane in run.planes()]
    assert abs(energy[-1] / energy[0] - 1.0) < 1e-9


@pytest.mark.medium
def test_a_slow_taper_is_adiabatic_its_conversion_falling_as_one_over_its_length():
    """No conversion in the adiabatic limit, and each mode's own phase.

    The same radius change over 0.25, 0.5 and 1 mm: the connection is ``a'/a``
    times a shape factor, so the converted amplitude, set by ``Gamma/(beta_1 -
    beta_2)``, falls as ``1/L`` -- a factor of 2 per doubling, gated at 1.8 to 2.2.
    The flux norm is conserved by the connection exactly (it is anti-Hermitian), so a
    run departs from it by the explicit residual's truncation alone
    (:func:`_truncation`).

    **The phase** of the launched channel against a constant guide at the entrance
    is ``-int (beta_1(z) - beta_1(0)) dz`` by a 201-point quadrature of fresh solves,
    which contains nothing of the route. The bound, before the number: the route's
    ``beta`` is the frequency interpolant's at every position, whose departure from a
    fresh solve varies along z, so ``L max_z |d beta(z) - d beta(0)|`` measured at nine
    positions; plus the second-order back-reaction of the conversion on the launched
    channel, bounded by ``|b_2/b_1|``; plus the path's own held-out error times
    ``|beta| L``.

    The runs are linear and the bound is the Runge-Kutta march's frozen-residual
    truncation, so the march is named: the default Adams history samples the flow's
    residual with the source, and with none steps the frozen block alone (ADR-042).
    """
    grid = grin_grid()
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    omega = float(grid.omega[index])
    constant = grin_taper(25e-6)
    ratios = []
    for length in (0.25e-3, 0.5e-3, LENGTH):
        guide = grin_taper(Linear(25e-6, 22.5e-6, length))
        moving, problem = _single_bin(
            guide, grid, length, STEPS, index, integrator=RungeKutta()
        )
        still, _ = _single_bin(
            constant, grid, length, STEPS, index, integrator=RungeKutta()
        )
        ratios.append(abs(moving[1] / moving[0]))
        assert abs(float(np.sum(abs(moving) ** 2)) - 1.0) < _truncation(problem)
        provider = guide.field_provider
        sampling = guide.sampling
        assert sampling is not None and sampling.path is not None
        positions = np.linspace(0.0, length, 201)
        beta = np.array(
            [
                float(
                    np.real(
                        1j
                        * provider.solve(
                            omega,
                            float(z),
                            nodes=provider.node_rule(
                                float(sampling.omega[0]), float(z), n_r=2
                            ),
                        ).generator[0, 0]
                    )
                )
                for z in positions
            ]
        )
        expected = -simpson(beta - beta[0], x=positions)
        measured = float(np.angle(moving[0] / still[0]))
        drift = _beta_departure(guide, omega, np.linspace(0.0, length, 9))[:, 0]
        bound = (
            length * float(np.max(abs(drift - drift[0])))
            + ratios[-1]
            + sampling.path.z_error * abs(beta[0]) * length
        )
        assert abs(measured - expected) < bound, (length, measured, expected, bound)
    assert 1.8 < ratios[0] / ratios[1] < 2.2, ratios
    assert 1.8 < ratios[1] / ratios[2] < 2.2, ratios


@pytest.mark.medium
def test_a_graded_index_taper_converts_and_converges_under_step_refinement():
    """A moving profile's conversion, and its order under the step.

    Linear, one bin, 1 mm, launched in the lower channel: the converted amplitude is
    a property of the taper (of order 1e-2 here, set by the connection over the
    splitting) and must not depend on the schedule. Successive differences at 10, 20
    and 40 steps fall faster than 16 per halving, so the embedded pair integrates the
    explicit residual -- the moving generator and the whole connection -- at its own
    high order, and the ten-step answer is within 1e-8 of the forty-step one. The
    order is the Runge-Kutta march's: the default Adams history samples the flow's
    residual with the source, and with none steps the frozen block alone (ADR-042).
    """
    grid = grin_grid()
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    guide = grin_taper(Linear(25e-6, 22.5e-6, LENGTH))
    runs = [
        _single_bin(guide, grid, LENGTH, steps, index, integrator=RungeKutta())[0]
        for steps in (10, 20, 40)
    ]
    conversion = abs(runs[-1][1] / runs[-1][0])
    assert 1e-3 < conversion < 1e-1
    first = float(np.max(abs(runs[0] - runs[1])))
    second = float(np.max(abs(runs[1] - runs[2])))
    assert first < 1e-8
    assert second == 0.0 or first / second > 16.0, (first, second)


# ------------------------------------------------- the step-index validations (slow)


STEP_RADIUS = Linear(4e-6, 3.6e-6, LENGTH)
"""A 10 % step-index taper: two HE_1m channels bound over the whole active band."""


STEP_BASIS = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.GeneralModalBasisWarning"
)
"""Silence the active-band basis warning on the step-index cases.

Their seventeen knots span the 835 nm grid's active band, which reaches 1.66 um, and
the interpolated basis is 1.8e-4 from orthonormal there -- just over the 1e-4 budget,
at a bin these single-bin and 835 nm runs never occupy. The gate the route applies,
weighted by the launch, passes; the warning covers the whole active band, which a
broadening run may reach.
"""


def step_grid():
    """Return the 835 nm grid the step-index cases propagate on."""
    return FieldGrid.from_limits(LENGTH, 835e-9, (815e-9, 860e-9), 60e-15)


def step_guide(descriptors, *, basis_block=None, blocks=()):
    """Return the tapered silica-in-1.40 step-index guide over the active band."""
    grid = step_grid()
    active = grid.omega[grid.active_mask]
    provider = make_provider(
        radius=STEP_RADIUS, indices=(1.45, 1.40), descriptors=descriptors
    )
    frequencies = np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 17)
    return ModalGuide.from_provider(
        provider,
        frequencies,
        nodes=provider.node_rule(float(frequencies[0]), 0.0, n_r=24),
        degenerate_blocks=blocks,
        basis=None
        if basis_block is None
        else circular_pair(provider_channels(provider), basis_block),
    )


def _step_run(guide, steps, index, launch=None):
    """Return a step-index linear single-bin run's final amplitudes.

    On the Runge-Kutta march, whose embedded pair integrates the flow's explicit
    residual at its own order: a linear run has no source, so the default Adams history
    would step the frozen block alone, at second order and unseen by its controller
    (``docs/guide/numerics/adams.md``, a coupled flow with no source), and these bounds
    are
    the march's.
    """
    grid = step_grid()
    simulation = Simulation.from_waveguide(
        (),
        LENGTH,
        provider=guide,
        grid=grid,
        wavelength=835e-9,
        fwhm=20e-15,
        energy=1e-12,
        shape="gauss",
        save_planes=[LENGTH],
        backend="numpy_ref",
        dz_max=LENGTH / steps,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule(np.full(steps, LENGTH / steps)),
        integrator=RungeKutta(),
        noise=None,
    )
    simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
    problem = simulation.translate()
    state = np.zeros_like(problem.initial.field0)
    if launch is None:
        state[0, 0, 0, index] = 1e-9
    else:
        state[0, :, 0, index] = 1e-9 * np.asarray(launch)
    problem = dataclasses.replace(
        problem, initial=dataclasses.replace(problem.initial, field0=state)
    )
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None)
    backend.propagate_to(LENGTH)
    final = backend.observe().require_field()[0, :, 0, index].copy()
    backend.close()
    return final / 1e-9, problem


def _dense_reference(provider, omega, *, basis=None, samples=33):
    """Return ``b(L)`` of the coupled-mode equations, solved independently by DOP853.

    ``b' = (G(z) - G_11(0) I - Gamma(z)) b`` in the provider's own true-power
    amplitude, launched ``b(0) = e_1``. ``G`` is the **modes' own** propagation
    constants at every position the solver asks for -- the eigenproblem, not a table
    -- and ``Gamma`` is :func:`solprop.modes.connection.modal_connection` at its own
    default resolution, sampled at ``samples`` Chebyshev-**Gauss** points (not the
    route's Lobatto ones) and interpolated at degree ``samples - 1``. A declared
    `basis` rotates both. Nothing of the route's tables, maps, expansions or flow is
    in it.
    """
    count = len(provider.modes)
    gauss = 0.5 * LENGTH * (1 - np.cos(np.pi * (np.arange(samples) + 0.5) / samples))
    links = np.array(
        [modal_connection(provider, omega, float(z)).matrix for z in gauss]
    )
    unitary = np.eye(count) if basis is None else basis.transform
    links = np.einsum("ij,zjk,kl->zil", unitary.conj().T, links, unitary)
    coefficients = np.polynomial.chebyshev.chebfit(
        2 * gauss / LENGTH - 1, links.reshape(samples, -1), samples - 1
    )

    def generator(z):
        diagonal = np.array(
            [
                -1j * omega / C * np.conj(mode.neff(np.array([omega]), z)[0])
                for mode in provider.modes
            ]
        )
        return unitary.conj().T @ np.diag(diagonal) @ unitary

    reference = generator(0.0)[0, 0]

    def derivative(z, state):
        amplitude = state[:count] + 1j * state[count:]
        link = np.polynomial.chebyshev.chebval(2 * z / LENGTH - 1, coefficients)
        change = (
            generator(z) - reference * np.eye(count) - link.reshape(count, count)
        ) @ amplitude
        return np.concatenate([change.real, change.imag])

    start = np.zeros(2 * count)
    start[0] = 1.0
    solved = solve_ivp(
        derivative, (0.0, LENGTH), start, method="DOP853", rtol=1e-12, atol=1e-18
    )
    return solved.y[:count, -1] + 1j * solved.y[count:, -1]


@STEP_BASIS
@pytest.mark.slow
@pytest.mark.xdist_group(name="moving-guide-dense-reference")
def test_a_multimode_step_index_taper_matches_a_dense_reference_and_converges():
    """``HE_11`` and ``HE_12`` on a 10 % taper, against an independent DOP853 solve.

    Bounds, first. The route and the reference share the provider and nothing else;
    they differ by (i) the route's representation of ``G`` and ``Gamma`` -- the path's
    Chebyshev expansion along z and the tables' cubic in frequency -- and (ii) its
    step. (i) is bounded by ``L max_z ||dG(z)|| + ||dGamma(z)||``, the departures of
    the route's own flow leaf at the bin from fresh values at five held-out positions,
    because the propagator of a lossless guide is unitary and a perturbation ``dA``
    moves ``b`` by at most ``int ||dA|| dz``. (ii) falls at the pair's high order:
    the successive differences over 20, 40 and 80 steps are gated at 16 per halving.
    The quantity compared is ``b_2/b_1``, which the co-moving frame and the carrier's
    common phase leave alone. Measured: 6.5e-6, 8.8e-8 and 5.8e-9 at 20,
    40 and 80 steps against a ratio of 3.1e-3.
    """
    guide = step_guide((("HE", 1, 1, "even"), ("HE", 1, 2, "even")))
    grid = step_grid()
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    omega = float(grid.omega[index])
    routed = {}
    problem = None
    for steps in (20, 40, 80):
        routed[steps], problem = _step_run(guide, steps, index)
    reference = _dense_reference(guide.field_provider, omega)
    expected = reference[1] / reference[0]
    departures = [abs(routed[s][1] / routed[s][0] - expected) for s in (20, 40, 80)]
    assert problem is not None
    bound = _representation_bound(guide, problem, index)
    assert departures[-1] < bound + abs(
        routed[40][1] / routed[40][0] - routed[80][1] / routed[80][0]
    ), (departures, bound)
    assert departures[0] / departures[1] > 16.0, departures
    # The flux norm, conserved by the connection alone (it is anti-Hermitian).
    assert abs(float(np.sum(abs(routed[80]) ** 2)) - 1.0) < 1e-6
    # And conversion happened: the taper is not adiabatic at this length.
    assert abs(expected) > 1e-3


STEP_TERM = 1e-7
"""The forty-step schedule's share of a single-bin ratio's error, dimensionless.

Measured on the two-channel case: ``b_2/b_1`` moves by 8.8e-8 between 40 steps and the
dense reference and by 6e-9 at 80, falling faster than 16 per halving from 20, so at
40 steps the schedule's own error is below 1e-7.
"""


def _representation_bound(guide, problem, index):
    """Return ``L max_z (||dG|| + ||dGamma||)`` of a route's flow leaf at one bin."""
    flow = problem.linear.coupled_flow
    provider = guide.field_provider
    sampling = guide.sampling
    omega = float(problem.grids.omega[index])
    count = flow.n_modes
    entrance = path_snapshot(
        provider,
        sampling.omega,
        0.0,
        sampling.nodes,
        degenerate_blocks=sampling.degenerate_blocks,
        basis=sampling.basis,
    )
    reference = entrance.evaluate(omega, 0.0).generator[0, 0]
    worst = 0.0
    for z in np.linspace(0.07, 0.93, 5) * LENGTH:
        weights = flow_weights(flow, float(z))
        generator = np.tensordot(
            weights, flow.generator[..., index].reshape(-1, count, count), axes=1
        )
        link = np.tensordot(
            weights, flow.connection[..., index].reshape(-1, count, count), axes=1
        )
        rule = path_rule(provider, sampling.nodes, float(z))
        fresh = provider.solve(omega, float(z), nodes=rule)
        if sampling.basis is not None:
            fresh = apply_channel_basis(fresh, sampling.basis)
        measured = modal_connection(
            provider,
            omega,
            float(z),
            degenerate_blocks=sampling.degenerate_blocks,
            basis=sampling.basis,
        ).matrix
        worst = max(
            worst,
            float(
                np.linalg.norm(
                    generator - (fresh.generator - reference * np.eye(count)), 2
                )
            )
            + float(np.linalg.norm(link - measured, 2)),
        )
    return LENGTH * worst


@STEP_BASIS
@pytest.mark.slow
@pytest.mark.xdist_group(name="moving-guide-complex-connection")
def test_the_connection_is_stored_unconjugated_as_a_complex_basis_decides(
    record_property,
):
    """The leaf carries ``Gamma``, not ``conj Gamma``; a circular basis bites.

    **The derivation.** The provider's fields are internal-convention and the
    synthesis stores them unconjugated, so the kernel's amplitude ``a`` is the
    coefficient of the internal field on them: ``E(r, z) = sum_n a_n e_n(r; z)``.
    Differentiating at fixed physical field gives ``a' = ... - W^-1 <e|d_z e> a``
    with ``<.|.>`` the package's flux pairing -- exactly ``Gamma`` as measured, with
    no conjugate. In the optics convention the field and the basis are both
    conjugated and the connection would be ``conj Gamma``; the kernel is not in that
    convention (``docs/dev/conventions.md`` section 2).

    **Where it bites.** In a real basis ``Gamma`` is real and the two are equal. The
    circular pair of ``HE_11`` beside the parity pair of ``HE_12`` puts ``+-i/sqrt 2``
    into the rotated connection's off-diagonal between the two groups, so ``Gamma``
    and ``conj Gamma`` differ there by the whole of their imaginary part.

    **The test.** The same physical launch -- ``HE_11`` left-handed -- propagated in
    the circular basis, against the dense reference solved in the provider's own
    real basis and rotated; and against the same route with the stored connection
    conjugated, which must be wrong by the conversion's own size. Bound: the
    representation bound of the multimode case above, computed for this guide, plus
    :data:`STEP_TERM`; the conjugated route must miss by ten times that. Measured:
    6.2e-8 against a bound of 1.3e-5.
    """
    descriptors = (
        ("HE", 1, 1, "even"),
        ("HE", 1, 1, "odd"),
        ("HE", 1, 2, "even"),
        ("HE", 1, 2, "odd"),
    )
    guide = step_guide(descriptors, basis_block=(0, 1), blocks=((0, 1), (2, 3)))
    sampling = guide.sampling
    assert sampling is not None and sampling.basis is not None
    grid = step_grid()
    index = int(np.argmin(abs(grid.omega - grid.omega0)))
    omega = float(grid.omega[index])
    routed, problem = _step_run(guide, 40, index)
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.connection is not None
    imaginary = float(np.max(abs(np.imag(flow.connection[..., index]))))
    assert imaginary > 1.0
    reference = _dense_reference(guide.field_provider, omega, basis=sampling.basis)
    # The quantity compared is each channel relative to the launched one, which the
    # co-moving frame and the carrier's common phase leave alone.
    expected = reference[1:] / reference[0]
    departure = float(np.max(abs(routed[1:] / routed[0] - expected)))
    bound = _representation_bound(guide, problem, index) + STEP_TERM
    record_property("tc801_departure", departure)
    record_property("tc801_bound", bound)
    assert departure < bound
    conjugated_flow = dataclasses.replace(flow)
    conjugated = np.conj(flow.connection)
    conjugated.setflags(write=False)
    object.__setattr__(conjugated_flow, "connection", conjugated)
    linear = dataclasses.replace(problem.linear)
    object.__setattr__(linear, "coupled_flow", conjugated_flow)
    wrong_problem = dataclasses.replace(problem)
    object.__setattr__(wrong_problem, "linear", linear)
    backend = NumpyRefBackend(wrong_problem, grid=grid)
    backend.set_initial(wrong_problem.initial.field0, None)
    backend.propagate_to(LENGTH)
    wrong = backend.observe().require_field()[0, :, 0, index] / 1e-9
    backend.close()
    wrong_departure = float(np.max(abs(wrong[1:] / wrong[0] - expected)))
    record_property("tc801_conjugated_departure", wrong_departure)
    # Conjugating flips the sign of the imaginary couplings, so the channels they fill
    # arrive with the wrong sign: a departure of the order of their own size.
    assert wrong_departure > 0.5 * float(np.max(abs(expected[1:])))
    assert wrong_departure > 10 * bound


@STEP_BASIS
@pytest.mark.slow
@pytest.mark.xdist_group(name="moving-guide-single-channel")
def test_one_step_index_channel_agrees_with_the_specified_mode_path(record_property):
    """One ``HE_11`` channel on a weakly guiding taper against the scalar model.

    The ``SpecifiedMode`` of the same fibre (``SpecifiedMode.from_mode`` of the tapered
    ``StepIndexMode``) carries its dispersion along z and its effective area beside it,
    and its nonlinear coefficient is ``gamma A(0)/A(z)``. Each route is
    compared with **its own** run on the constant entrance guide, and each nonlinear
    run with its own dispersion-only partner, which removes the frame, the grid, the
    carrier and everything the two routes do differently at constant geometry; what is
    left is what the taper does to the self-phase, in each model.

    That ratio -- moving over constant -- is the path average of ``gamma(z)/gamma(0)``:
    the physics sanity check that the effective coefficient follows the mode area as
    the core shrinks. The general route's is the vector overlap's and the
    ``SpecifiedMode``'s the scalar area's; on a weakly guiding fibre the two differ by
    the measured ``max_z |gamma_vec(z)/gamma_vec(0) - A(0)/A(z)|`` (fresh solves at nine
    positions, asserted below 5e-3 as the weak-guidance premise). A 150 fs pulse over
    a millimetre sees no dispersion to speak of (``L/L_D`` of order 1e-3), so the
    peak's power is the launch's in all four runs to that order. Bound: the departure,
    plus ``L/L_D`` measured from the dispersion-only runs' own peak powers, plus
    ``4 rtol``.
    """
    radius = Linear(4e-6, 3.2e-6, LENGTH)
    window, trange = (790e-9, 890e-9), 800e-15
    field_grid = FieldGrid.from_limits(LENGTH, 835e-9, window, trange)
    envelope_grid = EnvelopeGrid.from_limits(LENGTH, 835e-9, window, trange)
    band = (
        min(
            float(field_grid.omega[field_grid.active_mask].min()),
            float(envelope_grid.omega[envelope_grid.active_mask].min()),
        ),
        max(
            float(field_grid.omega[field_grid.active_mask].max()),
            float(envelope_grid.omega[envelope_grid.active_mask].max()),
        ),
    )
    frequencies = np.linspace(0.99 * band[0], 1.01 * band[1], 13)

    def guide_for(profile, *, kerr):
        provider = make_provider(
            radius=profile,
            indices=(1.45, 1.44),
            kerr=MaterialKerr(chi3=GENERAL_CHI3) if kerr else None,
        )
        return ModalGuide.from_provider(
            provider,
            frequencies,
            nodes=provider.node_rule(float(frequencies[0]), 0.0, n_r=24),
        )

    common = {
        "save_planes": [LENGTH],
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "rtol": 1e-6,
        "wavelength": 835e-9,
        "fwhm": 150e-15,
        "energy": 4e-9,
        "shape": "gauss",
        "progress": "none",
    }

    def general(profile, *, kerr):
        simulation = Simulation.from_waveguide(
            (),
            LENGTH,
            provider=guide_for(profile, kerr=kerr),
            grid=field_grid,
            **(
                {
                    "noise": None,
                    **{
                        key: value for key, value in common.items() if key != "progress"
                    },
                }
            ),
        )
        simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
        return analytic_field(
            field_grid,
            cast(
                "np.ndarray",
                cast("MemoryOutput", simulation.run(progress="none"))
                .planes()[-1]
                .field,
            )[0, 0],
        )

    def specified(profile, gamma):
        mode = SpecifiedMode.from_mode(
            StepIndexMode(profile, core=1.45, cladding=1.44, profile_wavelength=835e-9)
        )
        run = cast(
            "MemoryOutput",
            prop_gnlse(
                gamma,
                LENGTH,
                grid=envelope_grid,
                fibre=mode,
                shock=True,
                **{"noise": None, **common},
            ),
        )
        return analytic_field(
            envelope_grid, cast("np.ndarray", run.planes()[-1].field)[0, 0]
        )

    def turned(nonlinear, linear):
        peak = int(np.argmax(abs(linear) ** 2))
        return float(np.angle(nonlinear[peak] / linear[peak])), float(
            abs(linear[peak]) ** 2
        )

    omega0 = float(field_grid.omega0)
    fibre = StepIndexMode(radius, core=1.45, cladding=1.44, profile_wavelength=835e-9)
    moving = guide_for(radius, kerr=True)
    provider = moving.field_provider
    sampling = moving.sampling
    assert sampling is not None
    positions = np.linspace(0.0, LENGTH, 9)
    vector, scalar = [], []
    for z in positions:
        rule = path_rule(provider, sampling.nodes, float(z))
        sample = provider.solve(omega0, float(z), nodes=rule)
        vector.append(
            modal_kerr_coefficient(
                sample.electric[:, :, 0],
                rule.weights,
                np.where(rule.region_index == 0, GENERAL_CHI3, 0.0),
                omega0,
                metric=float(np.real(sample.flux_metric[0, 0])),
            )
        )
        # The scalar model's own area: the StepIndexMode the SpecifiedMode is built
        # from, at its profile wavelength -- the frequency the vector coefficient is
        # taken at -- and not the provider channel's surrogate, which is taken at the
        # median knot of the band.
        scalar.append(1.0 / fibre.aeff(float(z)))
    vector_ratio = np.asarray(vector) / vector[0]
    scalar_ratio = np.asarray(scalar) / scalar[0]
    departure = float(np.max(abs(vector_ratio - scalar_ratio)))
    record_property("moving_gamma_vector_scalar_departure", departure)
    # Weak guidance, the premise: the two models' coefficients move alike to a few
    # parts in a thousand over a 20 % taper (measured 2.1e-3).
    assert departure < 5e-3
    assert vector_ratio[-1] > 1.2

    gamma0 = float(vector[0])
    phases = {}
    powers = {}
    for name, profile in (("moving", radius), ("constant", 4e-6)):
        phases[f"general {name}"], powers[f"general {name}"] = turned(
            general(profile, kerr=True), general(profile, kerr=False)
        )
        phases[f"specified {name}"], powers[f"specified {name}"] = turned(
            specified(profile, gamma0), specified(profile, 0.0)
        )
    general_ratio = phases["general moving"] / phases["general constant"]
    scalar_ratio_run = phases["specified moving"] / phases["specified constant"]
    mean_gamma = float(simpson(vector_ratio, x=positions) / LENGTH)
    for name, value in (
        ("general", general_ratio),
        ("specified", scalar_ratio_run),
        ("mean_gamma", mean_gamma),
    ):
        record_property(f"moving_self_phase_ratio_{name}", value)
    dispersion = max(
        abs(powers["general moving"] / powers["general constant"] - 1.0),
        abs(powers["specified moving"] / powers["specified constant"] - 1.0),
    )
    bound = departure + dispersion + 4e-6
    record_property("moving_self_phase_bound", bound)
    # The coefficient follows the area: the self-phase grows with the path average.
    assert abs(general_ratio - mean_gamma) < 2 * bound + 1e-2 * abs(mean_gamma - 1.0)
    assert abs(general_ratio - scalar_ratio_run) < bound + 1e-2 * abs(mean_gamma - 1.0)


def test_a_moving_route_record_refuses_what_its_flow_cannot_be(kinked):
    """The route's own doors on the moving flow's arrays and measurements."""
    route = general_modal_route(
        kinked, grin_grid(), z_max=LENGTH, omega_reference=float(grin_grid().omega0)
    )
    assert route.moving
    assert route.moving_generator is not None and route.moving_connection is not None
    for change, message in (
        ({"moving_connection": None}, "supplied together"),
        ({"moving_error": -1.0}, "moving_error is a measured"),
        ({"connection_certificate": float("nan")}, "connection_certificate"),
        ({"moving_generator": route.moving_generator[:1]}, "finite expansion"),
        (
            {
                "coupled_generator": np.ones(
                    (route.n_modes, route.n_modes, route.tables.omega.size), complex
                )
            },
            "not both",
        ),
    ):
        with pytest.raises(TranslationError, match=message):
            dataclasses.replace(route, **change)


def test_the_carrier_of_phi_is_the_entrance_channel_at_every_position(kinked_problem):
    """What ``Phi`` is built from on a moving route: the entrance, and the kinks."""
    simulation, problem = kinked_problem
    route = simulation.general
    assert route is not None
    mode = simulation.modes[0]
    assert isinstance(mode, ProviderChannelMode)
    carrier = _carrier(mode, route)
    entrance = mode.entrance()
    omega = np.array([float(grin_grid().omega0)])
    assert np.array_equal(carrier.neff(omega, LENGTH), entrance.neff(omega))
    assert np.array_equal(
        carrier.neff_minus_one(omega, LENGTH), entrance.neff_minus_one(omega)
    )
    assert np.array_equal(
        carrier.dneff_domega(omega, LENGTH), entrance.dneff_domega(omega)
    )
    assert carrier.aeff(LENGTH) == entrance.aeff()
    assert carrier.dimlimits(LENGTH) == entrance.dimlimits()
    assert carrier.channel_label == entrance.channel_label
    assert carrier.azimuthal_order == entrance.azimuthal_order
    assert carrier.principal_component(LENGTH) is None
    assert not carrier.z_constant and not carrier.scale_invariant
    assert carrier.at_parameters({"position": 0.7 * LENGTH}) is carrier.entrance
    assert cast("ProviderChannelMode", carrier.entrance).table is entrance.table
    assert list(carrier.parameter_paths) == ["position"]
    assert carrier.parameter_paths["position"].breakpoints.tolist() == [KINK]
    with pytest.raises(TranslationError, match="declares one path"):
        carrier.at_parameters({"core_radius": 1e-6})
    with pytest.raises(ModeError, match="provider channel"):
        carrier.field(np.zeros((1, 2)), 0.0)
    assert problem.linear.hard_kinks.tolist() == [KINK]


MID = 6e-4
"""A saved plane inside the kinked taper's second section, m: a schedule edge."""


@pytest.fixture(scope="module")
def kinked_run(kinked):
    """Ten steps of the kinked taper, saved at the entrance, `MID` and the end."""
    grid = grin_grid()
    simulation = Simulation.from_waveguide(
        (),
        LENGTH,
        provider=kinked,
        grid=grid,
        save_planes=[MID, LENGTH],
        dz_max=LENGTH / STEPS,
        dz_init=LENGTH / STEPS,
        schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        backend="numpy_ref",
        **{"noise": None, **GRIN_PULSE},
    )
    # As `kinked_problem`: ten 1e-4 m steps are beyond the default collar's explicit
    # stability, and the collar is not what is under test.
    # The boundary monitor is off too: the pulse fills the tight window to its edge.
    simulation.settings = dataclasses.replace(
        simulation.settings, absorber=None, boundary_check=None
    )
    run = cast("MemoryOutput", simulation.run(progress="none"))
    tables = general_modal_route(
        kinked, grid, z_max=LENGTH, omega_reference=float(grid.omega0)
    ).tables
    return run, tables


def _constant_reference(radius):
    """Return a constant guide of one radius and its reference tables."""
    grid = grin_grid()
    constant = grin_taper(radius, kerr=True)
    tables = general_modal_route(
        constant, grid, z_max=LENGTH, omega_reference=float(grid.omega0)
    ).tables
    return constant, tables


def test_a_moving_guide_s_plane_is_reconstructed_on_that_plane_s_own_modes(
    kinked, kinked_run, record_property
):
    """A plane's fields are the guide's at that plane's position.

    At `MID` the moving guide's reconstruction must be what a **constant** guide of
    the radius there reconstructs from the same amplitudes: the same provider,
    solved at the same geometry on the same rule (node ``q`` has moved to that
    position's ``x_q``), in the same gauge. Both are one solve and one contraction,
    so they agree to rounding; the entrance reference, which the accessors used to
    rebuild every plane from, is measurably not that field.
    """
    run, tables = kinked_run
    grid = grin_grid()
    index = int(np.argmin(np.abs(run["z"] - MID)))
    assert run["z"][index] == pytest.approx(MID, abs=1e-15)
    radius = float(kinked_radius()(np.array([MID]))[0])
    _, reference = _constant_reference(radius)
    moved = path_rule(kinked.field_provider, kinked.sampling.nodes, MID)
    np.testing.assert_array_equal(moved.xy, reference.node_xy[0])
    amplitudes = np.sum(run.fields("fft")[index], axis=1)
    rounding = 64 * np.finfo(float).eps
    for accessor, stored in (
        (electric_field, "electric"),
        (magnetic_field, "magnetic"),
    ):
        rebuilt = accessor(
            run, tables, z=MID, analytic=True, convention="fft", guide=kinked
        )
        expected = analytic_field(
            grid, np.einsum("qcmw,mw->qcw", getattr(reference, stored)[0], amplitudes)
        )
        scale = float(np.max(np.abs(expected)))
        # One solve and one contraction on each side, in the same order: rounding.
        np.testing.assert_allclose(rebuilt, expected, rtol=0, atol=rounding * scale)
        entrance = analytic_field(
            grid, np.einsum("qcmw,mw->qcw", getattr(tables, stored)[0], amplitudes)
        )
        moved_by = float(np.max(np.abs(entrance - expected))) / scale
        record_property(f"{stored}_entrance_departure", moved_by)
        # The radius has moved by 4 % there: the entrance shape is far from it.
        assert moved_by > 1e-2


def _gram_departure(reference, weights):
    """Return the largest ``||W(w) - I||_2`` of a reference's same-frequency flux."""
    electric, magnetic = reference
    gram = 0.5 * (
        np.einsum("q,qmw,qnw->mnw", weights, electric[:, 0], np.conj(magnetic[:, 1]))
        - np.einsum("q,qmw,qnw->mnw", weights, electric[:, 1], np.conj(magnetic[:, 0]))
    )
    support = np.any(np.abs(electric) > 0.0, axis=(0, 1, 2))
    identity = np.eye(gram.shape[0])[:, :, None]
    departure = np.moveaxis((gram - identity)[:, :, support], -1, 0)
    return float(np.max(np.linalg.norm(departure, ord=2, axis=(1, 2))))


@pytest.mark.parametrize("position", [0.0, MID])
def test_a_moving_guide_s_reconstructed_flux_integrates_to_its_physical_energy(
    kinked, kinked_run, position, record_property
):
    """Parseval through the plane's own fields, at the entrance and inside the taper.

    The time integral of the reconstructed cycle-averaged flux is the same-frequency
    sum ``sum_w a^H W(w) a`` of the plane's own flux metric, the cross-frequency
    terms integrating to nothing over the window (as measured for a constant guide);
    the run records ``sum_w |a|^2``, its metric being the identity (a declared basis
    on a moving guide keeps the identity metric, not a dense metric expanded along
    z). So they differ by at most ``max_w ||W(w) - I||_2`` of the
    reference at that plane, which is computed here from a constant guide of that
    radius before the reconstruction is read.
    """
    run, tables = kinked_run
    grid = grin_grid()
    index = int(np.argmin(np.abs(run["z"] - position)))
    radius = float(kinked_radius()(np.array([position]))[0])
    _, reference = _constant_reference(radius)
    bound = _gram_departure(
        (reference.electric[0], reference.magnetic[0]), reference.weights[0]
    )
    record_property("gram_departure", bound)
    flux = reconstructed_flux(run, tables, plane=index, guide=kinked)
    integrated = float(np.sum(flux)) * float(grid.dt)
    recorded = float(np.ravel(run.planes()[index].stats["physical_energy"])[0])
    departure = abs(integrated - recorded) / recorded
    record_property("energy_against_the_flux_integral", departure)
    # The Gram bound plus the arithmetic of a 288-bin, node-weighted reduction.
    assert departure < bound + 1e-12


def test_a_moving_reference_needs_the_path_and_a_position_the_provider_solves(
    kinked, kinked_run
):
    """The two refusals of a plane's own reference, by name."""
    _, tables = kinked_run
    constant, _ = _constant_reference(25e-6)
    with pytest.raises(OutputError, match="records no path"):
        _plane_reference(tables, constant, MID)
    with pytest.raises(OutputError, match="cannot be solved again"):
        _plane_reference(tables, kinked, 2 * LENGTH)


def test_a_moving_plane_s_nodes_are_where_its_flux_density_lives(kinked, kinked_run):
    """A moving guide's plane is reconstructed on that plane's own rule.

    `plane_nodes` returns the rule the provider builds at the plane's position, the
    one the field accessors solve on, so the reconstructed flux density integrated
    with its weights is the accessor's flux to the rounding of one reduction; at the
    entrance the nodes are the entrance rule's, and 0.6 mm in they have moved.
    """
    run, tables = kinked_run
    index = int(np.argmin(np.abs(run["z"] - MID)))
    xy, weights = plane_nodes(run, tables, z=MID, guide=kinked)
    moved = path_rule(kinked.field_provider, kinked.sampling.nodes, MID)
    np.testing.assert_array_equal(xy, moved.xy)
    np.testing.assert_array_equal(weights, moved.weights)
    entrance, _ = plane_nodes(run, tables, plane=0, guide=kinked)
    np.testing.assert_array_equal(entrance, kinked.sampling.nodes.xy)
    assert not np.array_equal(entrance, xy)
    density = reconstructed_flux(run, tables, plane=index, density=True, guide=kinked)
    flux = reconstructed_flux(run, tables, plane=index, guide=kinked)
    # One weighted sum over the node axis on each side: rounding, 64 eps of cover.
    np.testing.assert_allclose(
        weights @ density, flux, rtol=0, atol=64 * np.finfo(float).eps * flux.max()
    )


def test_a_moving_plane_s_rule_needs_the_path_and_a_position_it_solves(
    kinked, kinked_run
):
    """The node accessor refuses as the field accessors do, by name."""
    _, tables = kinked_run
    constant, _ = _constant_reference(25e-6)
    with pytest.raises(OutputError, match="records no path"):
        _plane_rule(tables, constant, MID)
    with pytest.raises(OutputError, match="cannot be solved again"):
        _plane_rule(tables, kinked, 2 * LENGTH)


def _pump_area(guide, omega, z):
    """Return worked example 05's ``effective_area``, verbatim but for its rule."""
    source = guide.field_provider
    rule = source.node_rule(omega, z, n_r=2, n_theta=2)
    field = source.solve(omega, z, nodes=rule).electric[:, :, 0]
    density = np.sum(np.abs(field) ** 2, axis=1)
    return float(
        np.sum(rule.weights * density) ** 2 / np.sum(rule.weights * density**2)
    )


def test_a_provider_channel_s_area_is_solved_again_at_the_pump(kinked):
    """The surrogate area at a chosen frequency, from a fresh solve.

    Against worked example 05's own helper on the graded-index taper: the same
    solve, on the same rule -- the provider's rule does not depend on the frequency
    it is asked at -- and the same sum, so bit for bit. The median-knot surrogate
    the channel otherwise reports is 23 % above it here, which is the trap.
    """
    channel = kinked.mode_collection.modes[0]
    omega0 = float(grin_grid().omega0)
    for z in (0.0, KINK, MID, LENGTH):
        solved = channel.solved_area(omega0, z)
        assert solved == _pump_area(kinked, omega0, z)
        assert channel.aeff(z) > 1.15 * solved
    # A snapshot ignores the position and is solved where it was sampled.
    entrance = channel.entrance()
    assert entrance.solved_area(omega0, MID) == channel.solved_area(omega0, 0.0)
    assert entrance.area_at(omega0) == channel.solved_area(omega0, 0.0)


def test_a_moving_channel_s_pump_area_is_a_path_at_its_sampled_positions(kinked):
    """At every sampled position the path is the fresh solve; between, the polynomial.

    Declared exactly as the channel's own median-knot `area_path` is -- the same
    span and section edges -- so the fibre built from it splits on the same kink.
    """
    channel = kinked.mode_collection.modes[0]
    omega0 = float(grin_grid().omega0)
    path = channel.area_at(omega0)
    assert isinstance(path, ZProfile)
    median = channel.area_path
    assert median is not None
    assert path.z_span == median.z_span
    np.testing.assert_array_equal(path.breakpoints, median.breakpoints)
    positions = kinked.sampling.path.positions.ravel()
    np.testing.assert_array_equal(
        path(positions), [channel.solved_area(omega0, float(z)) for z in positions]
    )
    # Between the sampled positions the degree-d polynomial through them, which on
    # a smooth taper is the fresh solve to well inside the path's own budget.
    between = 0.5 * (positions[0] + positions[1])
    assert float(path(np.array([between]))[0]) == pytest.approx(
        channel.solved_area(omega0, between), rel=1e-6
    )


def test_from_mode_takes_the_area_at_the_wavelength_and_warns_without_one(kinked):
    """A specified fibre from a provider channel: the pump's area, or a warning.

    `wavelength` hands over the channel's area path at the pump beside its
    dispersion path; neither `aeff` nor `wavelength` keeps the median-knot surrogate
    and warns, naming the frequency and how to pass the pump; an explicit `aeff`
    wins without a warning.
    """
    channel = kinked.mode_collection.modes[0]
    wavelength = GRIN_PULSE["wavelength"]
    omega = 2 * np.pi * C / wavelength
    fibre = SpecifiedMode.from_mode(channel, wavelength=wavelength)
    assert not fibre.dispersion_is_constant
    for z in (0.0, KINK, LENGTH):
        assert fibre.aeff(z) == pytest.approx(channel.solved_area(omega, z), rel=1e-12)
    knot = channel.table.source_identity["channel_area_omega"]
    with pytest.warns(
        SurrogateAreaWarning, match=rf"median knot.*{re.escape(f'{knot:.6e}')} rad/s"
    ):
        surrogate = SpecifiedMode.from_mode(channel)
    assert surrogate.aeff(KINK) == pytest.approx(channel.aeff(KINK), rel=1e-12)
    declared = SpecifiedMode.from_mode(
        channel,
        aeff=PiecewiseLinear(
            np.array([0.0, KINK, LENGTH]), np.array([1e-10, 1.5e-10, 2e-10])
        ),
        wavelength=wavelength,
    )
    assert declared.aeff(LENGTH) == pytest.approx(2e-10)
    with pytest.raises(ModeError, match="wavelength must be positive"):
        SpecifiedMode.from_mode(channel, wavelength=-1.0)
    bare = dataclasses.replace(channel.entrance(), source=None)
    with pytest.raises(ModeError, match="built without its provider"):
        SpecifiedMode.from_mode(bare, wavelength=wavelength)
    # A table that does not record its surrogate's frequency is named as such.
    with pytest.warns(SurrogateAreaWarning, match="an undeclared frequency"):
        _warn_surrogate_area("LP0,1", None)


def test_prop_gnlse_assembles_its_gamma_from_the_pump_s_area(kinked):
    """`from_gnlse` with a provider channel and ``n2``: gamma is the pump's.

    The coefficient mapped at the entrance is ``n2 omega0 / (c A)`` with the area
    solved at ``wavelength``, not the median knot's, and the fibre's area path is the
    pump's along the taper, not the median-knot surrogate's.
    """
    channel = kinked.mode_collection.modes[0]
    wavelength = GRIN_PULSE["wavelength"]
    omega = 2 * np.pi * C / wavelength
    n2 = 2.7e-20
    simulation = Simulation.from_gnlse(
        None,
        LENGTH,
        fibre=channel,
        n2=n2,
        grid=EnvelopeGrid.from_limits(LENGTH, wavelength, (1.0e-6, 1.06e-6), 100e-15),
        fwhm=GRIN_PULSE["fwhm"],
        energy=GRIN_PULSE["energy"],
        wavelength=wavelength,
        backend="numpy_ref",
        noise=None,
    )
    mapping = simulation.configuration["gnlse"]
    assert isinstance(mapping, GnlseProblem)
    pump = channel.solved_area(omega, 0.0)
    assert mapping.aeff == pump
    # Two SI conversions between the n2 declared and the gamma recorded.
    assert mapping.gamma == pytest.approx(
        n2 * omega / (C * pump), rel=8 * np.finfo(float).eps
    )
    assert mapping.mode.aeff(LENGTH) == pytest.approx(
        channel.solved_area(omega, LENGTH), rel=1e-12
    )


def test_wavelength_is_ignored_where_the_area_is_not_a_function_of_it():
    """Every other mode's area is declared once; the wavelength changes nothing."""
    mode = StepIndexMode(4e-6, core=1.45, cladding=1.44, profile_wavelength=835e-9)
    assert SpecifiedMode.from_mode(mode, wavelength=1.5e-6).aeff() == mode.aeff()


def test_a_channel_s_provider_is_not_part_of_its_record(kinked):
    """The source is a live handle: the run records the provider once, beside it."""
    record = ModeRecord.from_mode(kinked.mode_collection.modes[0])
    assert "source" not in record.configuration
    assert kinked.mode_collection.modes[0].source is not None


def test_a_declared_basis_is_applied_before_a_channel_s_area_is_read():
    """A circular channel's pump area is the circular member's, not a parity one's.

    The fresh solve is put in the guide's declared basis first, as the path's
    samples are, so the area is that of ``(x + i y)/sqrt 2``: a ring-free HE11 of
    both parities, whose ``|E|^2`` is their mean and whose area is therefore not a
    parity channel's.
    """
    guide = general_guide(
        descriptors=(("HE", 1, 1, "even"), ("HE", 1, 1, "odd")),
        blocks=((0, 1),),
        circular=(0, 1),
    )
    sampling = guide.sampling
    assert sampling is not None and sampling.basis is not None
    omega = float(sampling.omega[len(sampling.omega) // 2 - 1])
    solved = guide.field_provider.solve(omega, 0.0, nodes=sampling.nodes)
    rotated = apply_channel_basis(solved, sampling.basis)
    for index, channel in enumerate(guide.mode_collection.modes):
        assert isinstance(channel, ProviderChannelMode)
        area = channel.solved_area(omega)
        assert area == channel_area(rotated, sampling.nodes, index)
        assert area != channel_area(solved, sampling.nodes, index)
