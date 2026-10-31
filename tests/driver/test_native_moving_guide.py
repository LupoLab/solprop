# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A moving solid guide on the native CPU backend, against ``numpy_ref``.

Native executes a moving guide's synthesis and projection maps (kernel schema 14): the
maps' Chebyshev coefficients along z are uploaded once at construction and combined per
stage from the bundle's ``transform_weights`` by a native kernel -- the weighted sum
over the stage's section's ``n_deg`` terms, one team per map row and the row's bins its
vector lanes -- which feeds the general modal synthesis, contraction and projection
unchanged. The connection rides in the frozen-residual coupled flow exactly as the
capillary flows execute it.

The cases are those of ``tests/driver/test_moving_guide.py``: the kinked graded-index
taper, the step-index HE11/HE12 taper, and the same taper in a complex basis, where the
connection's conjugation matters (a circular HE11 pair beside the parity pair of HE12).
Each carries a Kerr core and a launch strong enough that the source moves the field
measurably, and each is run ten fixed steps over the millimetre on both backends from
the same problem -- over the millimetre for the graded-index taper and over its first
fifth for the step-index ones (:data:`STEP_LENGTH`); the fields are compared against a
bound written before the run (:func:`moving_parity_bound`). The step-index cases
translate in minutes on a busy machine (the connection at every position and knot), so
they are ``slow``.
"""

import dataclasses
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from driver.test_moving_guide import (
    GRIN_PULSE,
    LENGTH,
    STEP_RADIUS,
    STEPS,
    grin_taper,
    kinked_radius,
    step_grid,
)
from driver.test_native_coupled_flow import (
    Case,
    native_class,
    parity_bound,
    propagate,
    reference_and_linear,
    thread_states,
)
from modal_provider_cases import GENERAL_CHI3, grin_grid, make_provider
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.modes.guide import ModalGuide, circular_pair, provider_channels
from solprop.modes.material import MaterialKerr
from solprop.output.memory import MemoryOutput
from solprop.stepper import FixedSchedule
from solprop.translate.problem import KernelProblem

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="native is not installed"
)

UNIT = 2.0**-53
"""Unit roundoff of float64."""

GRIN_ENERGY = 1e-8
"""Launch energy of the graded-index case, J: a Kerr phase of a few hundredths."""

STEP_ENERGY = 1e-9
"""Launch energy of the step-index cases, J."""


STEP_LENGTH = 2e-4
"""Window of the step-index cases, m: the first fifth of their millimetre's path.

The ten-step bound carries a perturbation through each step's explicit residual,
amplified by ``1 + dz K`` per step, and the step-index taper's ``K`` is 1.1e4 1/m
(``tests/driver/test_moving_guide.py``): ten steps over the whole millimetre take
``dz K`` = 1.1, where the bound's growth factor exceeds a thousand and says nothing. Ten
steps of 20 um keep ``dz K`` near a fifth, inside the taper's advisory step, as a user's
run must be.
"""


def _simulation(guide, grid, length=LENGTH, **pulse) -> Simulation:
    """Return a ten-step general modal simulation of a guide, with no absorber."""
    with warnings.catch_warnings():
        # The guides' own basis and range warnings are tested in test_moving_guide.py.
        warnings.simplefilter("ignore")
        simulation = Simulation.from_waveguide(
            (),
            length,
            provider=guide,
            grid=grid,
            save_planes=[length],
            dz_max=length / STEPS,
            dz_init=length / STEPS,
            schedule=FixedSchedule(np.full(STEPS, length / STEPS)),
            backend="numpy_ref",
            **{"noise": None, **pulse},
        )
        simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
        simulation.translate()
    return simulation


def _case(simulation: Simulation, length=LENGTH) -> Case:
    """Wrap a simulation's problem as the coupled-flow parity harness's case."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = simulation.translate()
    return Case(problem, simulation.grid, length)


def grin_kinked() -> Case:
    """The kinked graded-index taper with a Kerr core, two sections of degree four."""
    grid = grin_grid()
    guide = grin_taper(kinked_radius(), kerr=True)
    return _case(_simulation(guide, grid, **(GRIN_PULSE | {"energy": GRIN_ENERGY})))


def _step_guide(descriptors, *, basis_block=None, blocks=()) -> ModalGuide:
    """Return ``test_moving_guide``'s step-index taper with a Kerr core."""
    grid = step_grid()
    active = grid.omega[grid.active_mask]
    provider = make_provider(
        radius=STEP_RADIUS,
        indices=(1.45, 1.40),
        descriptors=descriptors,
        kerr=MaterialKerr(chi3=GENERAL_CHI3),
    )
    frequencies = np.linspace(0.99 * float(active[0]), 1.01 * float(active[-1]), 17)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ModalGuide.from_provider(
            provider,
            frequencies,
            nodes=provider.node_rule(float(frequencies[0]), 0.0, n_r=24),
            degenerate_blocks=blocks,
            basis=None
            if basis_block is None
            else circular_pair(provider_channels(provider), basis_block),
        )


STEP_PULSE = {
    "wavelength": 835e-9,
    "fwhm": 20e-15,
    "energy": STEP_ENERGY,
    "shape": "gauss",
}


def step_taper() -> Case:
    """HE11 and HE12 on the 10 % step-index taper, with a Kerr core."""
    guide = _step_guide((("HE", 1, 1, "even"), ("HE", 1, 2, "even")))
    return _case(
        _simulation(guide, step_grid(), STEP_LENGTH, **STEP_PULSE), STEP_LENGTH
    )


def complex_basis() -> Case:
    """The circular HE11 pair beside the HE12 parity pair, a Kerr core."""
    descriptors = (
        ("HE", 1, 1, "even"),
        ("HE", 1, 1, "odd"),
        ("HE", 1, 2, "even"),
        ("HE", 1, 2, "odd"),
    )
    guide = _step_guide(descriptors, basis_block=(0, 1), blocks=((0, 1), (2, 3)))
    return _case(
        _simulation(guide, step_grid(), STEP_LENGTH, **STEP_PULSE), STEP_LENGTH
    )


CASES = {
    "grin_kinked": grin_kinked,
    "step_taper": step_taper,
    "complex_basis": complex_basis,
}

SLOW = frozenset({"step_taper", "complex_basis"})
"""The step-index cases, whose translation takes tens of seconds each."""


def _gamma(count: int) -> float:
    """Return Higham's ``gamma_n = n u / (1 - n u)``."""
    return count * UNIT / (1.0 - count * UNIT)


def map_conditioning(terms: np.ndarray) -> float:
    """Return ``max sum_k |c_k| / min_z max |sum_k w_k c_k|`` over a map's sections.

    `terms` is a moving map's coefficients, shape ``(n_intervals, n_deg, ...)``. A
    stage's weights are Chebyshev values on the section, ``|T_k(x)| <= 1``, so the
    sum of the absolute products is at most ``sum_k |c_k|`` element by element and the
    combined map's largest element is at least ``max |c_0| - sum_(k>0) max |c_k|``:
    the ratio bounds the combination's normwise conditioning at every stage of the
    run without evaluating any.
    """
    worst = 0.0
    for section in terms:
        absolute = np.sum(np.abs(section), axis=0)
        peaks = np.max(np.abs(section.reshape(section.shape[0], -1)), axis=1)
        floor = float(peaks[0] - np.sum(peaks[1:]))
        assert floor > 0.0, "the maps' Chebyshev terms must fall off"
        worst = max(worst, float(np.max(absolute)) / floor)
    return worst


def combination_bound(problem: KernelProblem) -> float:
    """Return the maps' combination's relative rounding on the source, both backends.

    Written before any measurement. Each backend forms a combined map element as a
    sum of ``n_deg`` products of a real weight and a complex term (``numpy_ref``'s
    ``tensordot``, Kokkos's per-lane loop in term order; the other sections' zero
    weights add exact zeros), each part within ``gamma(n_deg)`` of the sum of its
    absolute products, so the two are within ``2 gamma(n_deg) kappa`` of each other
    relative to the map, with ``kappa`` from :func:`map_conditioning`. The source is
    cubic in the synthesis and linear in the projection, so it moves by three times
    the synthesis's relative error plus the projection's.
    """
    tables = problem.transform.modal_tables
    assert tables is not None
    assert tables.synthesis is not None and tables.projection is not None
    n_deg = int(tables.synthesis.shape[1])
    synthesis = 2 * _gamma(n_deg) * map_conditioning(tables.synthesis)
    projection = 2 * _gamma(n_deg) * map_conditioning(tables.projection)
    return 3 * synthesis + projection


def moving_parity_bound(case: Case, expected: np.ndarray, linear: np.ndarray) -> float:
    """Return the ten-step bound on ``native``'s departure from ``numpy_ref``, absolute.

    The coupled-flow bound (:func:`parity_bound`: the shared linear work, the
    flow's formation and carries, the source's transforms, contraction and projection
    on the stage's nonlinear increment ``rho``, the explicit residual, ten steps each
    amplifying by ``1 + 3 rho + r``) plus, per stage, the maps' combination
    (:func:`combination_bound`) carried on the same increment ``rho`` and amplified
    by the same growth.
    """
    problem = case.problem
    flow = problem.linear.coupled_flow
    assert flow is not None
    scale = float(np.max(np.abs(problem.initial.field0)))
    dz = case.length / STEPS
    rho = float(np.max(np.abs(expected - linear))) / scale / STEPS
    residual = float(np.max(flow.residual_norm_bound)) * dz
    growth = (1 + 3 * rho + residual) ** STEPS
    stages = problem.stepper.tableau.n_stages
    combination = STEPS * stages * rho * combination_bound(problem) * growth * scale
    return parity_bound(case, expected, linear) + combination


def measured(name: str) -> dict[str, float]:
    """Return a case's Kokkos departure and bound, relative to the launch, and rho."""
    case = CASES[name]()
    expected = propagate(case, NumpyRefBackend)
    actual = propagate(case, native_class())
    # The launch scaled down by a thousand and back up is the same launch propagated
    # linearly, to a millionth of its nonlinear increment.
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    scale = float(np.max(np.abs(case.problem.initial.field0)))
    rho = float(np.max(np.abs(expected - linear))) / scale / STEPS
    bound = moving_parity_bound(case, expected, linear)
    return {
        "departure": float(np.max(np.abs(actual - expected))) / scale,
        "bound": bound / scale,
        "combination": combination_bound(case.problem),
        "rho": rho,
    }


def _params():
    """Every case, the step-index ones marked slow."""
    return [
        pytest.param(
            name, marks=pytest.mark.slow if name in SLOW else pytest.mark.medium
        )
        for name in sorted(CASES)
    ]


@pytest.mark.parametrize("name", _params())
def test_native_runs_ten_steps_of_a_moving_guide_as_numpy_ref_does(
    name, record_property
):
    """Ten fixed steps of each moving guide on ``native``, within the derived bound."""
    result = measured(name)
    for key, value in result.items():
        record_property(key, value)
    assert result["bound"] > 0.0
    assert result["rho"] > 1e-5, "the source must move the field through the maps"
    assert result["departure"] <= result["bound"]


@pytest.mark.medium
def test_the_moving_maps_are_combined_per_stage_not_frozen():
    """The stages of one step read different maps: a frozen map would not be caught.

    Kokkos run with every stage's weights replaced by the step's first stage's --
    the maps frozen at the step's start -- departs from ``numpy_ref`` by far more than
    the parity bound, so the parity test above measures the combination and not
    only the rest of the route. The stages are the Runge-Kutta march's: an Adams step
    evaluates the source once, at the step's end.
    """
    built = grin_kinked()
    stepper = dataclasses.replace(built.problem.stepper, integrator=None)
    case = dataclasses.replace(
        built, problem=dataclasses.replace(built.problem, stepper=stepper)
    )
    expected = propagate(case, NumpyRefBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    bound = moving_parity_bound(case, expected, linear)
    backend = native_class()(case.problem, grid=case.grid)
    original = backend._stage_bundles
    single = backend._stage_scalars

    def frozen(z0, positions):
        first = single(z0, z0).transform_weights
        return tuple(
            dataclasses.replace(bundle, transform_weights=first)
            for bundle in original(z0, positions)
        )

    backend._stage_bundles = frozen
    try:
        backend.set_initial(case.problem.initial.field0, None)
        backend.propagate_to(case.length)
        actual = backend.observe().require_field().copy()
    finally:
        backend.close()
    assert float(np.max(np.abs(actual - expected))) > 100 * bound


@pytest.mark.medium
def test_auto_runs_a_moving_guide_on_native():
    """``backend="auto"`` resolves to Kokkos, which now executes the moving maps."""
    guide = grin_taper(kinked_radius(), kerr=True)
    grid = grin_grid()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        simulation = Simulation.from_waveguide(
            (),
            LENGTH,
            provider=guide,
            grid=grid,
            save_planes=[LENGTH],
            dz_max=LENGTH / STEPS,
            dz_init=LENGTH / STEPS,
            rtol=1e-8,
            backend="auto",
            **{"noise": None, **GRIN_PULSE},
        )
        # The bare guide, no absorber and so no window watch: the light reaches the
        # edge of this short window by design.
        simulation.settings = dataclasses.replace(
            simulation.settings, absorber=None, boundary_check=None
        )
        run = cast(MemoryOutput, simulation.run(progress="none"))
    assert run.metadata is not None
    record = run.metadata.backend
    assert record is not None
    assert record.name == "native"
    energy = [float(np.sum(plane.stats["physical_energy"])) for plane in run.planes()]
    assert len(energy) == 2
    # As the default's end-to-end run on the reference: lossless, so the run's
    # physical energy moves by its integration error alone, 5.2e-11 there.
    assert abs(energy[-1] / energy[0] - 1.0) < 1e-9


PROBE = Path(__file__).with_name("native_moving_thread_probe.py")
"""The subprocess that runs the cases at one thread count and saves their states."""


@pytest.mark.slow
def test_a_moving_guide_is_invariant_under_the_thread_count(tmp_path):
    """One thread against four on the three cases, within the parity bound.

    The combination is a fixed-order loop inside one output element, as every other
    kernel of the route is, and the route's transforms are below the FFT plan
    threshold. Bitwise invariance is not promised, though (SEP 25): two thread
    layouts are two orderings of one discrete march, held to the bound of two
    implementations of it, :func:`moving_parity_bound`.
    """
    names = sorted(CASES)
    one = thread_states(PROBE, 1, names, tmp_path)
    four = thread_states(PROBE, 4, names, tmp_path)
    for name in names:
        case = CASES[name]()
        bound = moving_parity_bound(case, *reference_and_linear(case))
        assert np.any(one[name])
        departure = float(np.max(np.abs(four[name] - one[name])))
        assert departure <= bound, (name, departure, bound)
