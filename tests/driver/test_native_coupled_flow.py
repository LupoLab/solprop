# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flow on the native CPU backend, against ``numpy_ref``.

Native executes ``linear.coupled_flow`` in the combinations the reference admits but the
commutator-free CF4 flow, which is not yet implemented on the compiled backends: the
scalar wall, the TE/TM wall and the split block, each one constant-Schur section, and
the frozen-residual flow of a varying guide. The section's actions are formed natively
from the eigenbasis of its Schur factor, or by the reference's scaling and squaring
where that basis is ill-conditioned, and applied natively; the explicit residual is
combined and added natively, and the whole step's conditioning is measured natively and
checked against the certificate per step as the reference checks it.

Every case propagates ten fixed steps on both backends from the same problem, and the
fields are compared against a bound written before the run (:func:`parity_bound`).
The launches are strong enough that the nonlinear half is exercised through the flow:
the bound is taken on each case's measured nonlinear increment, which is asserted
non-negligible.
"""

import math
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from driver.test_coherent_capillary import (
    CF4_WALL,
    LENGTH,
    STEPS,
    VECTOR_WALL,
    capillary,
    run_capillary,
    varying,
    vector_modes,
)
from driver.test_split_block import build
from flow_parity import UNIT, formation_bound, radial_family
from modal_provider_cases import (
    GENERAL_CHI3,
    GENERAL_Z_END,
    SPLIT_DESCRIPTORS,
    general_grid,
    make_provider,
    split_basis,
    split_guide,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.driver import Simulation
from solprop.exceptions import (
    BackendError,
    DriverError,
    GasFlowWarning,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
)
from solprop.grid import TimeGrid
from solprop.modes.capillary import CoherentWall
from solprop.modes.guide import ModalGuide
from solprop.modes.material import MaterialKerr
from solprop.translate.problem import KernelProblem

pytestmark = [
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
    pytest.mark.filterwarnings(
        f"ignore::{MaterialRangeWarning.__module__}.MaterialRangeWarning"
    ),
    pytest.mark.filterwarnings(
        f"ignore::{GrazingIncidenceWarning.__module__}.GrazingIncidenceWarning"
    ),
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]

ENERGY = 1e-3
"""Launch energy of the strong capillary cases, J."""

PRESSURE = 10.0
"""Helium fill of the strong capillary cases, bar.

A launch whose Kerr phase over the ten steps is a few parts in a thousand.

The reference suite's wall cases launch a nanojoule, whose nonlinear
increment is below the rounding this comparison is about; a millijoule at ten bar is
what makes the source carried back through the flow a measurable part of the step.
"""


@dataclass(frozen=True)
class Case:
    """One coupled-flow problem, its grid and the length it is run to."""

    problem: KernelProblem
    grid: TimeGrid
    length: float


def _capillary_case(simulation: Simulation) -> Case:
    """Wrap a capillary simulation's problem."""
    return Case(simulation.translate(), simulation.grid, LENGTH)


def scalar_wall() -> Case:
    """The scalar wall's one constant section over three channels."""
    return _capillary_case(
        capillary(wall=CoherentWall(), energy=ENERGY, pressure=PRESSURE)
    )


def te_tm_wall() -> Case:
    """The TE/TM wall's coupled azimuthal block over two families."""
    return _capillary_case(
        capillary(
            wall=VECTOR_WALL, modes=vector_modes(), energy=ENERGY, pressure=PRESSURE
        )
    )


def te_wall() -> Case:
    """The TE/TM wall beside ``TE_01`` and ``TE_02``.

    The transform resolves E_x and E_y; the flow never reads it.
    """
    return _capillary_case(
        capillary(
            wall=VECTOR_WALL,
            modes=radial_family("TE"),
            energy=ENERGY,
            pressure=PRESSURE,
        )
    )


def tm_wall() -> Case:
    """The TE/TM wall beside ``TM_01`` and ``TM_02``."""
    return _capillary_case(
        capillary(
            wall=VECTOR_WALL,
            modes=radial_family("TM"),
            energy=ENERGY,
            pressure=PRESSURE,
        )
    )


def taper_gradient() -> Case:
    """A taper under a pressure gradient, a frozen residual per section."""
    # Nine bar across a few millimetres of taper is not laminar, which the fill says;
    # the parity under test does not depend on it.
    with pytest.warns(GasFlowWarning, match="Reynolds"):
        simulation = varying(
            wall=CoherentWall(), pressure=(PRESSURE, 1.0), energy=ENERGY
        )
    return _capillary_case(simulation)


def lossless_taper() -> Case:
    """The taper coupling of a lossless three-channel taper, no wall."""
    return _capillary_case(
        varying(loss=False, cladding=1.45, pressure=PRESSURE, energy=ENERGY)
    )


def split_block() -> Case:
    """The declared rotation of a split pair, with a Kerr core.

    The guide of ``tests/driver/test_split_block.py`` in its rotated basis, whose
    generator couples, with the core responding as the general modal parity case does
    (``tests/general_parity.py``), so that the source is carried through the flow.
    """
    grid = general_grid()
    provider = make_provider(
        radius=6e-6,
        descriptors=SPLIT_DESCRIPTORS,
        indices=(1.45, 1.40),
        kerr=MaterialKerr(chi3=GENERAL_CHI3),
    )
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    guide = ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, 17),
        nodes=provider.node_rule(0.99 * lo, n_r=24),
        degenerate_blocks=((0, 1),),
        basis=split_basis(),
    )
    simulation = build(guide, grid, energy=2e-9)
    return Case(simulation.translate(), simulation.grid, GENERAL_Z_END)


CASES = {
    "scalar_wall": scalar_wall,
    "te_tm_wall": te_tm_wall,
    "te_wall": te_wall,
    "tm_wall": tm_wall,
    "taper_gradient": taper_gradient,
    "lossless_taper": lossless_taper,
    "split_block": split_block,
}


def native_class():
    """Return the optional wheel's backend class."""
    # Optional wheel: absent from the base environment.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBackend,
    )

    return NativeBackend


def propagate(case: Case, backend, field0=None) -> np.ndarray:
    """Run one case to its end on one backend and return the final state."""
    problem = case.problem
    driver = backend(problem, grid=case.grid)
    try:
        driver.set_initial(problem.initial.field0 if field0 is None else field0, None)
        driver.propagate_to(case.length)
        return driver.observe().require_field().copy()
    finally:
        driver.close()


def parity_bound(case: Case, expected: np.ndarray, linear: np.ndarray) -> float:
    """Return the ten-step bound on the native backend's departure from ``numpy_ref``.

    Written before any measurement. The two backends form their flow actions from
    the same Schur factors, the engine natively, so the actions differ by
    the formation bound ``eps_form`` of ``tests/flow_parity.py``; each carry of a
    vector by an action is then within ``2 n kappa u + kappa eps_form`` of the
    reference's:

    * per step, ``16 u`` of the launch for the shared linear work (the engine's
      established trial bound) and the whole step's carry of the state
      and the solution, one carry each, with ``n`` channels and ``kappa`` the
      certificate's declared ``norm(U) norm(U^-1)``;
    * per stage, the source's rounding on the stage's nonlinear increment ``rho`` --
      the window's departure from the same launch propagated linearly, over the step
      count: two transforms at the approved ``512 u log2 n_to`` each and
      ``(n + 84 + 3 Q) u`` for the synthesis, the tensor and the projection over
      ``Q`` nodes (a form which covers the mode-set route's scalar cubic) -- and
      the carry forward and back, ``2 n kappa u`` each, on the same increment;
    * per stage, the explicit residual of a varying flow, whose share of the step is
      ``r = dz max ||R||`` from the certificate's declared residual bound, rounded in
      ``(T + 2 n) u`` for ``T`` terms and the two carries.

    Ten steps add, each amplifying the error carried in by ``1 + 3 rho + r``.
    """
    problem = case.problem
    flow = problem.linear.coupled_flow
    assert flow is not None
    eps = float(np.finfo(float).eps)
    _, channels, _, _ = problem.shape
    tables = problem.transform.modal_tables
    nodes = (
        problem.transform.n_nodes if tables is None else int(tables.region_index.size)
    )
    kappa = float(np.max(flow.inverse_condition_bound))
    scale = float(np.max(np.abs(problem.initial.field0)))
    dz = case.length / STEPS
    rho = float(np.max(np.abs(expected - linear))) / scale / STEPS
    residual = float(np.max(flow.residual_norm_bound)) * dz
    assert flow.generator is not None
    terms = int(flow.generator.shape[0] * flow.generator.shape[1])
    carry = 2 * channels * kappa * eps + kappa * formation_bound(problem, dz)
    source = (
        2 * 512 * eps * math.log2(problem.grids.n_to)
        + (channels + 84 + 3 * nodes) * eps
    )
    stages = problem.stepper.tableau.n_stages
    per_step = 16 * eps + 2 * carry
    per_stage = rho * (source + 2 * carry) + residual * (terms + 2 * channels) * eps
    growth = (1 + 3 * rho + residual) ** STEPS
    return STEPS * (per_step + stages * per_stage) * growth * scale


def measured(name: str) -> tuple[float, float, float]:
    """Return one case's Kokkos departure and its bound, relative, and ``rho``."""
    case = CASES[name]()
    expected = propagate(case, NumpyRefBackend)
    actual = propagate(case, native_class())
    # The launch scaled down by a thousand and back up is the same launch
    # propagated linearly, to a millionth of its nonlinear increment.
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    scale = float(np.max(np.abs(case.problem.initial.field0)))
    rho = float(np.max(np.abs(expected - linear))) / scale / STEPS
    bound = parity_bound(case, expected, linear)
    return float(np.max(np.abs(actual - expected))) / scale, bound / scale, rho


@pytest.mark.medium
@pytest.mark.parametrize("name", sorted(CASES))
def test_native_runs_ten_steps_of_the_flow_as_numpy_ref_does(name, record_property):
    """Ten fixed steps of each admitted flow on ``native``, within the derived bound."""
    departure, bound, rho = measured(name)
    record_property("departure", departure)
    record_property("bound", bound)
    record_property("rho", rho)
    assert bound > 0.0
    assert rho > 1e-5, "the nonlinear half must be exercised through the flow"
    assert departure <= bound


# Off the eigenbasis route the flow keeps the host engine, which runs solprop's march
# with the warning tests/backends/test_native_march_admission.py asserts (SEP 39).
@pytest.mark.filterwarnings("ignore::solprop.exceptions.HostMarchWarning")
@pytest.mark.parametrize("route", ["eigenbasis", "scaling_and_squaring"])
def test_the_flow_s_native_actions_are_the_reference_s_within_the_formation_bound(
    route, monkeypatch
):
    """The engine forms every stage's pair within ``eps_form`` of the reference's.

    Bound, stated first (``tests/flow_parity.py``): the native action and
    the reference's are each within their route's formation bound of the exact
    ``Z exp(l T) Z^H``, so within the sum of the two relative to the action; the
    whole step's measured conditioning is the reference's singular-value product
    within ``32 n u``, half the certificate's own allowance. Every bin of the taper
    takes the eigenbasis; the second route sets the eigenbasis limit to zero so that
    every bin is formed by the native scaling and squaring.
    """
    # Optional wheel: absent from the base environment.
    from solprop_native.backend import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        _host_flow,
    )

    from solprop.backends.numpy_ref.backend import (  # noqa: PLC0415 - private oracle
        _coupled_flow_maps,
        _flow_action,
    )

    if route == "scaling_and_squaring":
        monkeypatch.setattr(
            "solprop.translate.coupled.eigenbasis_limit", lambda n_modes: 0.0
        )
    case = taper_gradient()
    reference = _coupled_flow_maps(case.problem)
    held = _host_flow(case.problem)
    assert reference is not None and held is not None
    assert np.all(held.route) if route == "eigenbasis" else not np.any(held.route)
    nodes = case.problem.stepper.tableau.nodes
    fractions = np.array([*(float(node) for node in nodes), 1.0])
    dz = case.length / STEPS
    bound = formation_bound(case.problem, dz)
    backend = native_class()(case.problem, grid=case.grid)
    try:
        native = backend._native_state
        shape = (fractions.size, -1, reference.n_modes, reference.n_modes)
        for section in range(reference.frozen.shape[0]):
            measured = native.form_flow_actions(section, dz, fractions)[0]
            forward = native.flow_actions_to_host(inverse=False).reshape(shape)
            inverse = native.flow_actions_to_host(inverse=True).reshape(shape)
            for slot, fraction in enumerate(fractions):
                if fraction == 0.0:
                    continue
                expected = _flow_action(reference, section, float(fraction) * dz)
                for got, want in zip(
                    (forward[slot], inverse[slot]), expected, strict=True
                ):
                    error = np.linalg.norm(got - want, ord=2, axis=(-2, -1))
                    scale = np.linalg.norm(want, ord=2, axis=(-2, -1))
                    assert np.max(error / scale) <= bound
            pair = _flow_action(reference, section, dz)
            whole = np.linalg.norm(pair[0], ord=2, axis=(-2, -1)) * np.linalg.norm(
                pair[1], ord=2, axis=(-2, -1)
            )
            assert np.max(abs(measured / whole - 1.0)) <= 32 * reference.n_modes * UNIT
    finally:
        backend.close()


def test_an_action_needing_too_many_squarings_is_refused_by_name_on_native():
    """The reference's squaring limit, on the section's row-sum norm, in its words."""
    case = scalar_wall()
    backend = native_class()(case.problem, grid=case.grid)
    try:
        backend.set_initial(case.problem.initial.field0, None)
        whole = backend._stage_scalars(0.0, case.length / STEPS)
        with pytest.raises(BackendError, match="native cannot take a coupled flow"):
            backend._flow_trial(1e30, whole, [whole])
        assert backend._flow_key is None
    finally:
        backend.close()


def test_a_step_longer_than_the_certificate_is_refused_by_name_on_native():
    """The whole step's conditioning is measured per step and checked, as there.

    The certificate is declared over ``dz_max``; a flow whose declared bound is
    lowered below what the step's own action measures is refused by name before the
    native trial runs, and the message names this backend. The certificate guards the
    Runge-Kutta march's inverse flow action; the default exponential formulation acts
    forward only (ADR-044) and does not measure it.
    """
    case = scalar_wall()
    flow = case.problem.linear.coupled_flow
    assert flow is not None
    shrunk = replace(
        flow, inverse_condition_bound=np.ones_like(flow.inverse_condition_bound)
    )
    problem = replace(
        case.problem,
        linear=replace(case.problem.linear, coupled_flow=shrunk),
        stepper=replace(case.problem.stepper, integrator=None),
    )
    backend = native_class()(problem, grid=case.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        with pytest.raises(BackendError, match="native measured a coupled flow"):
            backend.propagate_to(case.length)
    finally:
        backend.close()


def test_cf4_is_refused_by_name_at_the_backend_and_the_entry_point():
    """CF4 stays the reference's: both doors name it and the backend."""
    simulation = varying(wall=CF4_WALL)
    problem = simulation.translate()
    with pytest.raises(BackendError, match=r"native does not execute a 'cf4' section"):
        native_class()(problem, grid=simulation.grid)
    for name in ("native", "auto"):
        with pytest.raises(DriverError, match=r"method='cf4'.*numpy_ref"):
            varying(wall=CF4_WALL, backend=name).translate()


def test_a_native_batch_refuses_a_coupled_flow_by_name():
    """The fused accepted update applies no flow action, so a batch is refused."""
    # Optional wheel: absent from the base environment.
    from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeBatchBackend,
    )

    case = scalar_wall()
    members = [native_class()(case.problem, grid=case.grid) for _ in range(2)]
    for member in members:
        member.set_initial(case.problem.initial.field0, None)
    try:
        with pytest.raises(
            BackendError, match=r"does not execute linear\.coupled_flow"
        ):
            NativeBatchBackend(members)
    finally:
        for member in members:
            member.close()


def test_the_flow_s_ledger_rows_are_its_views_and_zero_elsewhere():
    """Every persistent flow View appears in the ledger at its real byte cost."""
    case = taper_gradient()
    flow = case.problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    backend = native_class()(case.problem, grid=case.grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    _, channels, _, bins = case.problem.shape
    active = int(np.count_nonzero(np.any(flow.generator != 0.0, axis=(0, 1, 2, 3))))
    stages = case.problem.stepper.tableau.n_stages
    block = active * channels * channels * 16
    terms = int(flow.generator.shape[0] * flow.generator.shape[1])
    assert ledger["flow_forward"] == (stages + 1) * block
    assert ledger["flow_inverse"] == (stages + 1) * block
    assert ledger["flow_frozen"] == len(flow.section_methods) * block
    assert ledger["flow_residual_terms"] == terms * block
    assert ledger["flow_weights"] == stages * terms * 8
    assert ledger["flow_drive"] == channels * bins * 16
    other = capillary(energy=ENERGY, pressure=PRESSURE)
    plain = native_class()(other.translate(), grid=other.grid)
    try:
        rows = dict(plain.memory_ledger())
    finally:
        plain.close()
    assert all(size == 0 for name, size in rows.items() if name.startswith("flow_"))


@pytest.mark.medium
def test_auto_runs_a_wall_on_native_through_the_entry_point():
    """`"auto"` resolves to Kokkos where it is installed, and ``native`` runs the wall.

    The record and the final plane are the reference's: the same schedule, the same
    flow, within the parity bound's order (a millionth of the launch is generous for
    ten steps of a nanojoule, where the source is below rounding).
    """
    reference = run_capillary(wall=CoherentWall())
    automatic = run_capillary(wall=CoherentWall(), backend="auto")
    assert automatic.metadata is not None
    record = automatic.metadata.backend
    assert record is not None
    assert record.name == "native"
    expected = reference.planes()[-1].field
    actual = automatic.planes()[-1].field
    start = reference.planes()[0].field
    assert expected is not None and actual is not None and start is not None
    scale = float(np.max(np.abs(start)))
    assert float(np.max(np.abs(actual - expected))) <= 1e-12 * scale
    grid = general_grid()
    simulation = build(split_guide(grid), grid, backend="auto")
    assert simulation.translate().linear.coupled_flow is not None


PROBE = Path(__file__).with_name("native_flow_thread_probe.py")
"""The subprocess that runs every case at one thread count and saves its states."""


def thread_states(probe: Path, threads: int, names, directory: Path):
    """Run a probe at one thread count and return each case's final state.

    The lanes' one-thread budget is lifted and oversubscription allowed (ADR-052), and
    the probe's Kokkos pool is checked, so a silent cap cannot compare one thread with
    one thread.
    """
    path = directory / f"states-{threads}.npz"
    environment = os.environ | {
        "SOLPROP_NATIVE_THREADS": str(threads),
        "SOLPROP_THREADS": "",
        "SOLPROP_OVERSUBSCRIBE": "1",
        "OMP_PROC_BIND": "false",
    }
    subprocess.run(
        [sys.executable, str(probe), str(path), *names],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        cwd=Path(__file__).parents[1],
    )
    with np.load(path) as stored:
        assert int(stored["runtime_threads"]) == threads
        return {name: stored[name] for name in names}


def reference_and_linear(case: Case) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference's final state and the launch's linear propagation."""
    expected = propagate(case, NumpyRefBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    return expected, linear


@pytest.mark.slow
def test_the_flow_is_invariant_under_the_thread_count(tmp_path):
    """One thread against three, four and eight, within the parity bound.

    The flow's own kernels are fixed-order loops inside one output element, and the
    engine plans transforms as small as the split block's 256-point two-column inverse
    on one FFTW thread. Bitwise invariance is not promised, though (SEP 25): two thread
    layouts are two orderings of one discrete march, held to the bound that holds two
    implementations of it, :func:`parity_bound`, fixed from the reference before the
    threads are compared.
    """
    names = sorted(CASES)
    one = thread_states(PROBE, 1, names, tmp_path)
    for name in names:
        case = CASES[name]()
        bound = parity_bound(case, *reference_and_linear(case))
        assert np.any(one[name])
        for threads in (3, 4, 8):
            other = thread_states(PROBE, threads, [name], tmp_path)[name]
            departure = float(np.max(np.abs(other - one[name])))
            assert departure <= bound, (name, threads, departure, bound)
