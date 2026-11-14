# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled flow on the native CPU backend, against ``numpy_ref``.

Native executes ``linear.coupled_flow`` in every combination the reference admits: the
scalar wall, the TE/TM wall and the split block, each one constant-Schur section, and
the frozen-residual flow of a varying guide. The section's actions are formed natively
from the eigenbasis of its Schur factor, or by the reference's scaling and squaring
where that basis is ill-conditioned, and applied natively; the explicit residual is
combined and added natively, and the whole step's conditioning is measured natively and
checked against the certificate per step as the reference checks it.

The strong cases' ten fixed steps against the reference are rows of the case catalogue
(``tests/backends/flow_rows.py``, ``coupled flow ...``), which also guard the flow.
What stays here: the split block's run, which is adaptive (#809); the actions' formation
on either route; the refusal of a step past the certificate; a native group, a batched
scan and the ledger; the entry point's ``auto`` run; and the thread count.
"""

import subprocess
import sys
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from backends.forced_threads import (
    assert_threaded,
    saved_sizing,
    thread_environment,
)
from coupled_flow_cases import (
    CASES,
    ENERGY,
    PRESSURE,
    STEPS,
    capillary,
    guide_simulation,
    native_class,
    parity_bound,
    propagate,
    reference_and_linear,
    scalar_wall,
    taper_gradient,
)
from driver.test_coherent_capillary import run_capillary
from flow_parity import UNIT, formation_bound
from march_bounds import assert_points_matched
from modal_provider_cases import (
    general_grid,
    split_guide,
)
from sessions.runs import bare, final_field
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.backends.session import Start
from solprop.driver.executors import Batch, Local
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import (
    BackendError,
    GrazingIncidenceWarning,
    MaterialRangeWarning,
)
from solprop.modes.capillary import CoherentWall

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
def test_the_split_block_s_adaptive_run_is_the_reference_s(record_property):
    """The split block's run on ``native``, within :func:`parity_bound`.

    Not a fixed schedule: the case declares none, and its controller takes 147 steps
    (two rejected) of the default collar's stable length, not ten. The bound counts ten
    steps, so it is not derived for this run (#807); an adaptive comparison waits for
    #809's policy. The other cases' ten fixed steps are catalogue rows.
    """
    departure, bound, rho = measured("split_block")
    record_property("departure", departure)
    record_property("bound", bound)
    record_property("rho", rho)
    assert bound > 0.0
    assert rho > 1e-5, "the nonlinear half must be exercised through the flow"
    assert departure <= bound


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
        # Each problem's decomposition is kept while it lives, routed by the limit
        # in force when it was formed: this route forms its own.
        monkeypatch.setattr(
            "solprop_native.backend._HOST_FLOWS", weakref.WeakKeyDictionary()
        )
    case = taper_gradient()
    reference = _coupled_flow_maps(case.problem)
    held = _host_flow(case.problem)
    assert reference is not None and held is not None
    assert np.all(held.route) if route == "eigenbasis" else not np.any(held.route)
    nodes = case.run_request.stepper.tableau.nodes
    fractions = np.array([*(float(node) for node in nodes), 1.0])
    dz = case.length / STEPS
    bound = formation_bound(case.problem, case.run_request, dz)
    backend = native_class()(case.problem, case.run_request, grid=case.grid)
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
        case.problem, linear=replace(case.problem.linear, coupled_flow=shrunk)
    )
    run_request = case.run_request.with_stepper(
        replace(case.run_request.stepper, integrator=None)
    )
    backend = native_class()(problem, run_request, grid=case.grid)
    with pytest.raises(BackendError, match="native measured a coupled flow"):
        final_field(backend, problem, bare(run_request), ended=True, owns=True)


def test_a_native_group_admits_a_coupled_flow():
    """A coupled flow runs in a native group (SEP 47 section 6.7): each member's flow
    is its own session's, so nothing of its step is fused. The members' numerics are
    held to their lone sessions in ``native/tests/test_coupled_flow.py``."""
    from solprop_native.group import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
        NativeGroupSession,
        refusal,
    )

    from solprop.backends.sessions import start_group  # noqa: PLC0415

    simulation = capillary(wall=CoherentWall(), energy=ENERGY, pressure=PRESSURE)
    problem, request = simulation.translate(), simulation._run_request()
    problems, requests = [problem] * 2, [request] * 2
    starts = [Start.entrance(problem) for _ in range(2)]
    assert refusal(problems, requests, starts) is None
    members = [
        native_class()(
            problem,
            request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        for _ in range(2)
    ]
    group = start_group(members, problems, requests, starts)
    try:
        assert isinstance(group, NativeGroupSession)
    finally:
        group.close()


def _scan_point(point):
    """Return one point of a power scan of the scalar wall, natively."""
    return capillary(
        wall=CoherentWall(), backend="native", energy=1e-9 * point.values["power"]
    )


def test_a_batched_scan_of_coupled_flows_runs_as_the_local_loop(tmp_path):
    """A batched scan groups coupled-flow points into one native group, and each
    point is the local loop's to the matched-march bound (SEP 25)."""
    scan = Scan("coupled", power=[1.0, 2.0])
    local = run_scan(
        scan,
        _scan_point,
        output=tmp_path / "local.h5",
        executor=Local(),
        progress="none",
    )
    batched = run_scan(
        scan,
        _scan_point,
        output=tmp_path / "batch.h5",
        executor=Batch(2),
        progress="none",
    )
    assert local.status.tolist() == batched.status.tolist() == ["done"] * 2
    assert_points_matched(batched, local, scan, _scan_point, what="batch member")


def test_the_flow_s_ledger_rows_are_its_views_and_zero_elsewhere():
    """Every persistent flow View appears in the ledger at its real byte cost."""
    case = taper_gradient()
    flow = case.problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    backend = native_class()(case.problem, case.run_request, grid=case.grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    channels, _, bins = case.problem.shape
    active = int(np.count_nonzero(np.any(flow.generator != 0.0, axis=(0, 1, 2, 3))))
    stages = case.run_request.stepper.tableau.n_stages
    block = active * channels * channels * 16
    terms = int(flow.generator.shape[0] * flow.generator.shape[1])
    assert ledger["flow_forward"] == (stages + 1) * block
    assert ledger["flow_inverse"] == (stages + 1) * block
    assert ledger["flow_frozen"] == len(flow.section_methods) * block
    assert ledger["flow_residual_terms"] == terms * block
    assert ledger["flow_weights"] == stages * terms * 8
    assert ledger["flow_drive"] == channels * bins * 16
    other = capillary(energy=ENERGY, pressure=PRESSURE)
    plain = native_class()(other.translate(), other._run_request(), grid=other.grid)
    try:
        rows = dict(plain.memory_ledger())
    finally:
        plain.close()
    assert all(size == 0 for name, size in rows.items() if name.startswith("flow_"))


@pytest.mark.medium
def test_auto_runs_a_wall_on_native_through_the_entry_point():
    """`"auto"` resolves to Kokkos where it is installed, and ``native`` runs the wall.

    The record names native; the split guide's flow is translated for it. The final
    plane against the reference's is a row of the case catalogue
    (``tests/backends/flow_rows.py``, ``scalar wall driver``).
    """
    automatic = run_capillary(wall=CoherentWall(), backend="auto")
    assert automatic.metadata is not None
    record = automatic.metadata.backend
    assert record is not None
    assert record.name == "native"
    grid = general_grid()
    simulation = guide_simulation(split_guide(grid), grid, backend="auto")
    assert simulation.translate().linear.coupled_flow is not None


PROBE = Path(__file__).with_name("native_flow_thread_probe.py")
"""The subprocess that runs every case at one thread count and saves its states."""


def thread_states(probe: Path, threads: int, names, directory: Path):
    """Run a probe at one thread count and return each case's final state.

    The lanes' one-thread budget is lifted and oversubscription allowed (ADR-052), every
    pool is forced, since these sizes would otherwise keep their flow passes serial, and
    the probe's Kokkos pool and the flow's own width are checked, so a silent cap cannot
    compare one thread with one thread.
    """
    path = directory / f"states-{threads}.npz"
    subprocess.run(
        [sys.executable, str(probe), str(path), *names],
        check=True,
        capture_output=True,
        text=True,
        env=thread_environment(threads, None),
        cwd=Path(__file__).parents[1],
    )
    assert_threaded(saved_sizing(path), threads, "kernels.flow", "kernels.stage")
    with np.load(path) as stored:
        assert int(stored["runtime_threads"]) == threads
        return {name: stored[name] for name in names}


@pytest.mark.slow
def test_the_flow_is_invariant_under_the_thread_count(tmp_path):
    """One thread against three, four and eight, within the parity bound.

    Every pool is forced (:func:`thread_states`). The flow's own kernels are fixed-order
    loops inside one output element, and the engine plans transforms as small as the
    split block's 256-point two-column inverse on one FFTW thread. Bitwise invariance is
    not promised, though (SEP 25): two thread layouts are two orderings of one discrete
    march, held to the bound that holds two implementations of it, :func:`parity_bound`,
    fixed from the reference before the threads are compared.
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
