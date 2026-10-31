# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general modal route of solprop on the native backend, against ``numpy_ref``.

The route's source is a second execution path beside the
node pipeline (``src/general_modal.cpp``); everything before and after it -- the
lab-frame stage input, the in-stage absorber, the prefactor and the interaction
picture -- is the engine's existing code. Each case propagates ten fixed steps on
both backends from the same problem and compares the fields against a bound derived
before the run (:func:`_parity_bound`).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pytest
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.general_modal import (
    general_node_chunk,
    node_kerr_tensor,
)
from solprop.driver import Simulation
from solprop.exceptions import BackendError, GeneralModalBasisWarning
from solprop.grid import FieldGrid
from solprop.modes import (
    GradedIndexProvider,
    IndexMaterial,
    PowerLawPermittivity,
    StepIndexMode,
)
from solprop.modes.guide import (
    ModalGuide,
    StepIndexProvider,
    circular_pair,
    provider_channels,
)
from solprop.modes.material import MaterialKerr, MaterialRegion
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.profiles import Linear
from solprop.stats import StatsSpec
from solprop.stepper import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.modal import DEFAULT_NODE_BUDGET
from solprop.translate.problem import KernelProblem

from solprop_native import NativeBackend, NativeBatchBackend, build_info

type ComplexArray = npt.NDArray[np.complex128]

POOL_VARIABLES = (
    "SOLPROP_NATIVE_KERNEL_THREADS",
    "SOLPROP_NATIVE_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_FFT_THREADS",
    "SOLPROP_NATIVE_RAMAN_TAIL_FFT_THREADS",
)

EPS = float(np.finfo(np.float64).eps)
CHI3 = 2.2e-22
"""Core susceptibility of the step-index cases, m^2/V^2 (solprop's test value)."""
FIBRE = 1e-3
"""Declared guide length, m: solprop's millimetre, which sets the collar's rate."""
Z_END = 2e-4
"""End of the ten-step window, m: ten steps of 20 um, inside the 26 um the spectral
collar's in-stage scheme is stable to on the step-index grid of a millimetre."""
STEPS = 10
WAVELENGTH = 835e-9

HOST_ONLY = pytest.mark.skipif(
    build_info().device_backend != "none",
    reason="the thread count is the host build's; a device build has one stream",
)
"""Marks the thread-count probes, which a CUDA or HIP build has no counterpart of."""


def _grid() -> FieldGrid:
    return FieldGrid.from_limits(FIBRE, WAVELENGTH, (815e-9, 860e-9), 60e-15)


def _step_index_guide(
    grid: FieldGrid,
    *,
    pair: bool = False,
    raman: SilicaTensorResponse | None = None,
    circular: bool = False,
) -> ModalGuide:
    """Build solprop's 1.5 um silica strand over the grid's active band."""
    parities: tuple[Literal["even", "odd"], ...] = (
        ("even", "odd") if pair else ("even",)
    )
    modes = tuple(
        StepIndexMode(
            1.5e-6,
            core=1.45,
            cladding=1.0,
            profile_wavelength=WAVELENGTH,
            kind="HE",
            n=1,
            m=1,
            parity=parity,
        )
        for parity in parities
    )
    core, cladding = IndexMaterial(1.45), IndexMaterial(1.0)
    responses = (MaterialKerr(chi3=CHI3),) + (() if raman is None else (raman,))
    regions = (
        MaterialRegion("core", 0.0, modes[0].core_radius, core, responses),
        MaterialRegion("cladding", modes[0].core_radius, None, cladding),
    )
    guide = ModalGuide.from_step_index(
        modes, core=core, cladding=cladding, regions=regions
    )
    provider = guide.field_provider
    assert isinstance(provider, StepIndexProvider)
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    blocks = ((0, 1),) if pair else ()
    return ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, 9),
        nodes=provider.node_rule(0.99 * lo, n_r=16),
        degenerate_blocks=blocks,
        basis=circular_pair(provider_channels(provider), (0, 1)) if circular else None,
    )


def _grin_grid() -> FieldGrid:
    return FieldGrid.from_limits(FIBRE, 1.03e-6, (1.0e-6, 1.06e-6), 100e-15)


def _grin_guide(grid: FieldGrid) -> ModalGuide:
    """Build solprop's parabolic LP sector with a Kerr core, coarsely."""
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.45 * math.sqrt(1.0 - 2.0 * 0.005)),
        radius=25e-6,
        family="LP",
        count=2,
        elements=64,
        extent=2.0,
        core_responses=(MaterialKerr(chi3=CHI3),),
    )
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    return ModalGuide.from_provider(
        provider,
        np.linspace(0.99 * lo, 1.01 * hi, 9),
        nodes=provider.node_rule(0.99 * lo, n_r=2),
    )


@dataclasses.dataclass(frozen=True)
class Case:
    """One general modal problem, its launch and the context both backends need."""

    simulation: Simulation
    problem: KernelProblem
    state: ComplexArray


def _case(
    guide: ModalGuide,
    grid: FieldGrid,
    *,
    mix: complex | None = None,
    kerr_form: Literal["real", "analytic"] = "real",
    wavelength: float = WAVELENGTH,
) -> Case:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        simulation = Simulation.from_waveguide(
            (),
            FIBRE,
            provider=guide,
            grid=grid,
            backend="numpy_ref",
            kerr_form=kerr_form,
            save_planes=[Z_END],
            wavelength=wavelength,
            fwhm=20e-15,
            energy=2e-9,
            shape="gauss",
            noise=None,
        )
        # No collar: ten replayed steps are far beyond the in-stage collar's limit
        # under solprop's default Adams history, and the split-step scheme treats a
        # launch's content outside the active band differently here and on
        # numpy_ref (an open question on solprop's roadmap); the route is the subject.
        simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
        problem = simulation.translate()
    problem = _prescribed(problem, FixedSchedule(np.full(STEPS, Z_END / STEPS)))
    state = np.array(problem.initial.field0, copy=True)
    if mix is not None:
        # A second channel carrying part of the launch, so every cross term of the
        # cubic and of the delayed pair history is exercised.
        state[:, 1] = mix * state[:, 0]
    return Case(simulation, problem, state)


def _prescribed(problem: KernelProblem, schedule: FixedSchedule) -> KernelProblem:
    """Return `problem` on `schedule` with its method's order held fixed.

    A prescribed schedule alone does not fix the discrete computation: a
    variable-order Adams march chooses its orders from error estimates, which rounding
    can move. Holding the order fixed makes two runs of one problem -- two backends,
    thread counts, a batch member and its solo run, a repeat -- the same discrete
    computation, compared by rounding budgets (solprop SEP 25 section 4.4).
    """
    integrator = problem.stepper.integrator
    if isinstance(integrator, Multistep):
        integrator = dataclasses.replace(integrator, variable_order=False)
    return dataclasses.replace(
        problem,
        stepper=dataclasses.replace(
            problem.stepper, schedule=schedule, integrator=integrator
        ),
    )


def _run(
    backend_class: type[NumpyRefBackend] | type[NativeBackend], case: Case
) -> tuple[ComplexArray, dict[str, int]]:
    backend = backend_class(case.problem, grid=case.simulation.grid)
    try:
        backend.set_initial(case.state, None)
        backend.propagate_to(Z_END)
        return backend.observe().require_field(), backend.memory_ledger()
    finally:
        backend.close()


def _fft_bound(samples: int) -> float:
    """The derived FFT bound (``tests/test_numerics.py``)."""
    return 512 * EPS * math.ceil(math.log2(samples))


def _gamma(operations: int) -> float:
    scaled = operations * EPS
    return scaled / (1.0 - scaled)


def _linear(case: Case) -> ComplexArray:
    """Propagate the same launch on ``numpy_ref`` with every region linear."""
    problem = dataclasses.replace(
        case.problem,
        responses=dataclasses.replace(case.problem.responses, regional_responses=None),
    )
    return _run(NumpyRefBackend, dataclasses.replace(case, problem=problem))[0]


def _parity_bound(case: Case, reference: ComplexArray, linear: ComplexArray) -> float:
    """Return the absolute bound on the ten-step field difference, fixed in advance.

    Per step, the shared linear work -- stage combination, propagator, tableau and
    prefactor -- is the established sixteen-epsilon trial bound of the node route
    (``tests/test_numerics.py``), relative to the field. The general source adds a
    relative rounding ``e_src`` of its own: two transforms (the approved FFT bound
    each, the delayed pair two more on the doubled axis), the synthesis sum over
    channels, the cubic contraction over ``C^4`` and ``C`` terms (the dense count,
    which also bounds the real cubic's symmetric contraction, ``gamma_17``) and the
    projection sum over ``Q C`` terms, the last two relative to the sums of their
    absolute terms, which the dense tensor and the quadrature do not cancel. It is
    carried on one step's nonlinear increment, whose relative size ``rho`` is bounded
    by the whole window's departure from the same launch propagated linearly, divided
    by the step count, and every stage contributes at most that much. Ten steps add;
    the linear flow is unitary and a nonlinear increment of ``rho`` amplifies a
    perturbation by at most ``1 + 3 rho`` per step, which the last factor carries.
    """
    tables = case.problem.transform.modal_tables
    assert tables is not None
    n_to = case.problem.grids.n_to
    components = int(tables.electric.shape[2])
    nodes = int(tables.region_index.size)
    modes = case.problem.shape[1]
    delayed = case.problem.responses.regional_responses
    source = (
        _gamma(modes)
        + 2 * _fft_bound(n_to)
        + _gamma(components**4 + components)
        + _gamma(nodes * components)
    )
    if delayed is not None and delayed.n_terms:
        source += 2 * _fft_bound(2 * n_to) + _gamma(components**3)
    scale = float(np.max(np.abs(case.state)))
    rho = float(np.max(np.abs(reference - linear))) / scale / STEPS
    stages = case.problem.stepper.tableau.n_stages
    per_step = 16 * EPS + stages * source * rho
    return STEPS * per_step * (1.0 + 3.0 * rho) ** STEPS * scale


CASES: dict[str, Callable[[], Case]] = {
    "strand_real": lambda: _case(_step_index_guide(_grid()), _grid()),
    "strand_analytic": lambda: _case(
        _step_index_guide(_grid()), _grid(), kerr_form="analytic"
    ),
    "parity_pair": lambda: _case(
        _step_index_guide(_grid(), pair=True), _grid(), mix=0.6 * np.exp(0.3j)
    ),
    "delayed_tensor": lambda: _case(
        _step_index_guide(_grid(), pair=True, raman=SilicaTensorResponse()),
        _grid(),
        mix=0.6 * np.exp(0.3j),
    ),
    "delayed_strand": lambda: _case(
        _step_index_guide(_grid(), raman=SilicaTensorResponse()), _grid()
    ),
    "circular_basis": lambda: _case(
        _step_index_guide(_grid(), pair=True, circular=True),
        _grid(),
        mix=0.5j,
    ),
    "grin": lambda: _case(
        _grin_guide(_grin_grid()), _grin_grid(), mix=0.4, wavelength=1.03e-6
    ),
}


@pytest.fixture(scope="module", params=sorted(CASES))
def case(request: pytest.FixtureRequest) -> Case:
    return CASES[cast(str, request.param)]()


def test_ten_steps_match_the_reference_within_the_derived_bound(case: Case) -> None:
    expected, _ = _run(NumpyRefBackend, case)
    actual, _ = _run(NativeBackend, case)
    bound = _parity_bound(case, expected, _linear(case))
    departure = float(np.max(np.abs(actual - expected)))
    assert departure <= bound


def test_the_run_repeats_to_rounding_and_the_ledger_is_the_routes(
    case: Case,
) -> None:
    """A second run of the same prescribed march agrees with the first to rounding.

    The same discrete computation twice, held to the reference comparison's own
    derived bound (:func:`_parity_bound`), not to its bits (solprop SEP 25).
    """
    first, ledger = _run(NativeBackend, case)
    second, _ = _run(NativeBackend, case)
    bound = _parity_bound(case, first, _linear(case))
    assert float(np.max(np.abs(second - first))) <= bound
    tables = case.problem.transform.modal_tables
    assert tables is not None
    components = int(tables.electric.shape[2])
    nodes = int(tables.region_index.size)
    n_to = case.problem.grids.n_to
    chunk = min(
        nodes,
        general_node_chunk(
            components, n_to, int(tables.omega_fine.size), DEFAULT_NODE_BUDGET
        ),
    )
    assert tables.synthesis is not None
    assert ledger["general_synthesis"] == tables.synthesis[0, 0].nbytes
    assert (
        ledger["general_tensor"]
        == node_kerr_tensor(tables, case.problem.responses.regional_responses).nbytes
    )
    assert ledger["general_node_field"] == chunk * components * n_to * 16
    delayed = case.problem.responses.regional_responses
    has_delayed = delayed is not None and bool(delayed.n_terms)
    assert (ledger["general_node_drive"] > 0) == has_delayed


def test_the_general_statistics_are_the_references() -> None:
    case = CASES["circular_basis"]()
    spec = StatsSpec(names=("physical_energy", "analytic_flux", "spectral_flux"))
    values: list[dict[str, npt.NDArray[np.float64]]] = []
    for backend_class in (NumpyRefBackend, NativeBackend):
        member: NumpyRefBackend | NativeBackend = backend_class(
            case.problem, grid=case.simulation.grid
        )
        try:
            member.set_initial(case.state, None)
            values.append(member.stats(spec))
        finally:
            member.close()
    # The launch's statistics on both backends: each a reduction of squared moduli
    # over the channels and samples (``gamma`` of four operations a term, the terms
    # nonnegative) after at most one transform round trip each way, fixed beforehand.
    bound = 2 * _fft_bound(case.problem.grids.n_to) + _gamma(4 * case.state.size)
    for name in spec.names:
        np.testing.assert_allclose(
            values[1][name], values[0][name], rtol=bound, atol=0.0, err_msg=name
        )
    backend = NativeBackend(case.problem, grid=case.simulation.grid)
    try:
        backend.set_initial(case.state, None)
        with pytest.raises(BackendError, match="energy"):
            backend.stats(StatsSpec(names=("energy",)))
    finally:
        backend.close()


def _replaced(
    problem: KernelProblem,
    *,
    linear: dict[str, object] | None = None,
    responses: dict[str, object] | None = None,
) -> KernelProblem:
    """Return the problem with some rows of its linear and response groups replaced."""
    return dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, **(linear or {})),
        responses=dataclasses.replace(problem.responses, **(responses or {})),
    )


def test_every_combination_the_reference_refuses_is_refused_by_name() -> None:
    case = CASES["delayed_tensor"]()
    problem = case.problem
    tables = problem.transform.modal_tables
    metric = problem.linear.modal_metric
    regional = problem.responses.regional_responses
    assert tables is not None
    assert metric is not None
    assert regional is not None
    weight = problem.linear.power_weight_constant
    assert weight is not None
    spoiled = {
        "linear.modal_metric": _replaced(problem, linear={"modal_metric": None}),
        "above": _replaced(
            problem,
            linear={
                "modal_metric": dataclasses.replace(
                    metric, error=np.full_like(metric.error, 1.0)
                )
            },
        ),
        "model_id": _replaced(
            problem,
            responses={
                "regional_responses": dataclasses.replace(
                    regional, model_id="unknown-kerr"
                )
            },
        ),
        "not uniform": _replaced(
            problem,
            responses={
                "regional_responses": dataclasses.replace(
                    regional,
                    temporal_weight=regional.temporal_weight
                    * np.linspace(1.0, 2.0, regional.temporal_weight.size),
                )
            },
        ),
        "power_weight": _replaced(
            problem,
            linear={"power_weight_constant": 2.0 * weight},
        ),
    }
    for match, one in spoiled.items():
        with pytest.raises(BackendError, match=match):
            NativeBackend(one, grid=case.simulation.grid)


def _probe(
    threads: int, names: tuple[str, ...], out: Path
) -> tuple[dict[str, object], dict[str, ComplexArray]]:
    script = Path(__file__).with_name("general_modal_thread_probe.py")
    completed = subprocess.run(
        [sys.executable, str(script), str(out), *names],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "OMP_PROC_BIND": "false",
            "SOLPROP_NATIVE_THREADS": str(threads),
            "SOLPROP_OVERSUBSCRIBE": "1",
            "SOLPROP_THREADS": "",
            **{name: str(threads) for name in POOL_VARIABLES},
        },
    )
    assert completed.returncode == 0, completed.stderr
    record = cast("dict[str, object]", json.loads(completed.stdout.splitlines()[-1]))
    with np.load(out) as stored:
        fields = {name: cast(ComplexArray, stored[name]) for name in stored.files}
    return record, fields


def _agree_across_threads(
    names: tuple[str, ...], counts: tuple[int, int], tmp_path: Path
) -> None:
    """Run `names` at both thread `counts`; hold each case to its derived bound.

    Both sides march the same prescribed, fixed-order steps, so the thread count moves
    the field by rounding only (solprop SEP 25), within :func:`_parity_bound`.
    """
    runs = [_probe(count, names, tmp_path / f"fields-{count}.npz") for count in counts]
    assert [record["threads"] for record, _ in runs] == list(counts)
    (_, one), (_, other) = runs
    for name in names:
        case = CASES[name]()
        bound = _parity_bound(case, one[name], _linear(case))
        assert float(np.max(np.abs(other[name] - one[name]))) <= bound, name


@HOST_ONLY
def test_one_and_four_threads_agree_to_rounding_on_one_channel(
    tmp_path: Path,
) -> None:
    """Every native reduction on the route sums within one element.

    The route's own transforms are planned alike at any count (the delayed pair on the
    Raman provider); the comparison holds whatever order a future kernel sums in.
    """
    names = ("strand_real", "strand_analytic", "delayed_strand")
    _agree_across_threads(names, (1, 4), tmp_path)


@HOST_ONLY
@pytest.mark.parametrize("threads", [3, 4, 8])
def test_the_two_channel_cases_agree_to_rounding_under_the_thread_count(
    threads: int, tmp_path: Path
) -> None:
    """One thread against three, four and eight on the two-channel cases.

    The engine's in-stage absorber reads a two-column inverse transform of 256
    points, whose FFTW estimate plan changed its factorisation at three and four
    threads (radix 4 with 64-point leaves against radix 8 with 32-point ones), which
    moved the answer by rounding. Transforms that small are now planned on one FFTW
    thread, where a threaded plan only cost time; any plan is held to the bound.
    """
    names = ("parity_pair", "delayed_tensor", "circular_basis")
    _agree_across_threads(names, (1, threads), tmp_path)


def test_a_batch_of_general_members_agrees_with_its_members_alone() -> None:
    """The fixed-schedule batch shares only the accepted update across members.

    Each member and its solo run march the same prescribed, fixed-order steps, held to
    the member's own :func:`_parity_bound` (solprop SEP 25).
    """
    case = CASES["delayed_tensor"]()
    launches = [case.state, 0.5 * case.state[:, ::-1].copy()]
    cases = [dataclasses.replace(case, state=state) for state in launches]
    singles = [_run(NativeBackend, one)[0] for one in cases]
    bounds = [
        _parity_bound(one, single, _linear(one))
        for one, single in zip(cases, singles, strict=True)
    ]
    members = [NativeBackend(case.problem, grid=case.simulation.grid) for _ in launches]
    for member, state in zip(members, launches, strict=True):
        member.set_initial(state, None)
    batch = NativeBatchBackend(members)
    try:
        batch.propagate_to(Z_END)
        for actual, single, bound in zip(
            batch.observe_members(), singles, bounds, strict=True
        ):
            assert float(np.max(np.abs(actual.require_field() - single))) <= bound
    finally:
        batch.close()
        for member in members:
            member.close()


def _moving_grin() -> tuple[Simulation, KernelProblem]:
    """Build solprop's graded-index sector on a 10 % taper over the millimetre.

    Solprop's moving solid guide: the core radius follows
    ``Linear(25 -> 22.5 um)``, so the provider re-solves the modes along z and the
    route carries moving synthesis and projection maps (one section of degree four)
    with the connection in a frozen-residual coupled flow; ten fixed steps, a Kerr
    core, a launch whose Kerr phase is a few hundredths, and no absorber.
    """
    grid = _grin_grid()
    provider = GradedIndexProvider(
        permittivity=PowerLawPermittivity(IndexMaterial(1.45), 0.005),
        exterior=IndexMaterial(1.45 * math.sqrt(1.0 - 2.0 * 0.005)),
        radius=Linear(25e-6, 22.5e-6, FIBRE),
        family="LP",
        count=2,
        elements=32,
        extent=2.0,
        core_responses=(MaterialKerr(chi3=CHI3),),
    )
    active = grid.omega[grid.active_mask]
    lo, hi = float(active[0]), float(active[-1])
    knots = np.linspace(0.99 * lo, 1.01 * hi, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guide = ModalGuide.from_provider(
            provider, knots, nodes=provider.node_rule(float(knots[0]), 0.0, n_r=2)
        )
        simulation = Simulation.from_waveguide(
            (),
            FIBRE,
            provider=guide,
            grid=grid,
            backend="numpy_ref",
            save_planes=[FIBRE],
            dz_max=FIBRE / STEPS,
            dz_init=FIBRE / STEPS,
            schedule=FixedSchedule(np.full(STEPS, FIBRE / STEPS)),
            wavelength=1.03e-6,
            fwhm=60e-15,
            energy=1e-8,
            shape="gauss",
            noise=None,
        )
        simulation.settings = dataclasses.replace(simulation.settings, absorber=None)
        problem = simulation.translate()
    return simulation, _prescribed(
        problem, FixedSchedule(np.full(STEPS, FIBRE / STEPS))
    )


def _moving_run(
    backend_class: type[NumpyRefBackend] | type[NativeBackend],
    simulation: Simulation,
    problem: KernelProblem,
    field0: ComplexArray | None = None,
) -> tuple[ComplexArray, dict[str, int]]:
    backend = backend_class(problem, grid=simulation.grid)
    try:
        backend.set_initial(problem.initial.field0 if field0 is None else field0, None)
        backend.propagate_to(FIBRE)
        return np.asarray(
            backend.observe().require_field()
        ).copy(), backend.memory_ledger()
    finally:
        backend.close()


def _moving_bound(
    problem: KernelProblem, expected: ComplexArray, linear: ComplexArray
) -> float:
    """Return the ten-step bound relative to the launch; fixed before the first run.

    solprop's derivation (``tests/driver/test_native_moving_guide.py``): the coupled
    flow's (the formation bound, the carries, the residual) with the general source's
    rounding on ``Q`` nodes, plus per stage the maps' combination, ``n_deg`` products
    per element on each backend, ``2 gamma(n_deg)`` relative to a map whose Chebyshev
    terms are bounded by their sum (``|T_k| <= 1``), entering the cubic source three
    times through the synthesis and once through the projection.
    """
    from test_coupled_flow import (
        _formation_bound,  # pyright: ignore[reportPrivateUsage]
    )

    flow = problem.linear.coupled_flow
    tables = problem.transform.modal_tables
    assert flow is not None
    assert flow.generator is not None
    assert tables is not None
    assert tables.synthesis is not None
    assert tables.projection is not None
    _, channels, _, _ = problem.shape
    dz = FIBRE / STEPS
    kappa = float(np.max(flow.inverse_condition_bound))
    scale = float(np.max(np.abs(problem.initial.field0)))
    rho = float(np.max(np.abs(expected - linear))) / scale / STEPS
    residual = float(np.max(flow.residual_norm_bound)) * dz
    terms = int(flow.generator.shape[0] * flow.generator.shape[1])
    carry = 2 * channels * kappa * EPS + kappa * _formation_bound(problem, dz)
    nodes = int(tables.region_index.size)
    source = (
        2 * 512 * EPS * math.log2(problem.grids.n_to)
        + (channels + 84 + 3 * nodes) * EPS
    )
    n_deg = int(tables.synthesis.shape[1])

    def conditioning(maps: ComplexArray) -> float:
        worst = 0.0
        for index in range(maps.shape[0]):
            section: ComplexArray = maps[index]
            magnitude: npt.NDArray[np.float64] = np.abs(section)
            peaks: npt.NDArray[np.float64] = np.max(
                magnitude.reshape(section.shape[0], -1), axis=1
            )
            floor = float(peaks[0] - np.sum(peaks[1:]))
            assert floor > 0.0
            worst = max(worst, float(np.max(np.sum(magnitude, axis=0))) / floor)
        return worst

    combination = (
        2
        * _gamma(n_deg)
        * (3 * conditioning(tables.synthesis) + conditioning(tables.projection))
    )
    stages = problem.stepper.tableau.n_stages
    per_step = 16 * EPS + 2 * carry
    per_stage = (
        rho * (source + 2 * carry + combination)
        + residual * (terms + 2 * channels) * EPS
    )
    return STEPS * (per_step + stages * per_stage) * (1 + 3 * rho + residual) ** STEPS


def test_a_moving_guide_runs_ten_steps_as_the_reference_does() -> None:
    """Moving maps combined natively per stage, within the derived bound (E10c).

    The maps' Chebyshev coefficients are uploaded once; each stage's
    ``StageScalars.transform_weights`` (kernel schema 14) combine the stage's
    section's terms natively into the maps the source reads. The ledger carries every
    coefficient, the two evaluated maps and the trial's weights, and a second run of
    the same prescribed, fixed-order march agrees within the same bound.
    """
    simulation, problem = _moving_grin()
    tables = problem.transform.modal_tables
    assert tables is not None
    assert tables.synthesis is not None
    intervals, n_deg = tables.synthesis.shape[:2]
    assert intervals * n_deg > 1
    expected, _ = _moving_run(NumpyRefBackend, simulation, problem)
    actual, ledger = _moving_run(NativeBackend, simulation, problem)
    again, _ = _moving_run(NativeBackend, simulation, problem)
    small = np.asarray(problem.initial.field0 * 1e-3, dtype=np.complex128)
    linear = 1e3 * _moving_run(NumpyRefBackend, simulation, problem, small)[0]
    scale = float(np.max(np.abs(problem.initial.field0)))
    assert float(np.max(np.abs(expected - linear))) / scale / STEPS > 1e-4
    bound = _moving_bound(problem, expected, linear)
    assert float(np.max(np.abs(actual - expected))) / scale <= bound
    assert float(np.max(np.abs(again - actual))) / scale <= bound
    one = int(tables.synthesis[0, 0].nbytes)
    stages = problem.stepper.tableau.n_stages
    assert ledger["general_synthesis"] == intervals * n_deg * one
    assert ledger["general_projection"] == intervals * n_deg * one
    assert ledger["general_stage_maps"] == 2 * one
    assert ledger["general_map_weights"] == 8 * stages * intervals * n_deg


def test_moving_maps_without_their_connection_are_refused_by_name() -> None:
    """The shared admission list admits moving maps only beside their connection.

    Solprop's one list (``refuse_general_route``, run under this backend's name)
    refuses moving maps carried with no coupled flow -- modes that change shape and
    exchange no power -- as ``numpy_ref`` does. The maps are spoiled to two
    coefficients on a copy, set with ``object.__setattr__`` so that solprop's own
    validation is not what refuses them.
    """
    case = CASES["strand_real"]()
    problem = case.problem
    tables = problem.transform.modal_tables
    assert tables is not None
    assert tables.synthesis is not None
    moving = dataclasses.replace(tables)
    object.__setattr__(moving, "synthesis", np.repeat(tables.synthesis, 2, axis=1))
    transform = dataclasses.replace(problem.transform)
    object.__setattr__(transform, "modal_tables", moving)
    spoiled = dataclasses.replace(problem)
    object.__setattr__(spoiled, "transform", transform)
    with pytest.raises(
        BackendError, match=r"native executes a moving guide.s maps.*no coupled flow"
    ):
        NativeBackend(spoiled, grid=case.simulation.grid)
