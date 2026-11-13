# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The coupled-flow parity cases: the qualification capillary, its walls and tapers.

Shared by the coupled-flow tests and the catalogue's rows
(``tests/backends/flow_rows.py``), so that no row imports a test module:

- the coherent-wall qualification capillary of
  ``tests/driver/test_coherent_capillary.py`` (:func:`capillary`, :func:`varying` and
  their grid, collar, taper and vector modes);
- a general guide launching one channel, ten steps over the millimetre
  (:func:`guide_simulation`, ``tests/driver/test_split_block.py``'s);
- the strong coupled-flow cases of ``tests/driver/test_native_coupled_flow.py``
  (:data:`CASES`), each a :class:`Case`, their ten-step run (:func:`propagate`) and the
  parity bound written before any measurement (:func:`parity_bound`), relative to the
  launch's peak.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pytest

from flow_parity import formation_bound, radial_family
from legacy_geometry import LEGACY_GRID
from modal_provider_cases import (
    GENERAL_CHI3,
    GENERAL_Z_END,
    SPLIT_DESCRIPTORS,
    general_grid,
    make_provider,
    split_basis,
)
from sessions.runs import bare, final_field
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import Start
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import GasFlowWarning
from solprop.grid import FieldGrid, TimeGrid
from solprop.modes import CoherentWall, MarcatiliMode
from solprop.modes.guide import ModalGuide
from solprop.modes.material import MaterialKerr
from solprop.modes.protocol import Mode
from solprop.physdata.mixtures import Pressure
from solprop.profiles import PiecewiseLinear, ZProfile
from solprop.stepper import FixedSchedule
from solprop.translate.problem import PropagationProblem
from solprop.translate.run_request import RunRequest

# --- the qualification capillary ------------------------------------------------------

RADIUS = 50e-6
"""Core radius of the qualification case, m.

Small enough that a millimetre is a measurable loss -- the highest retained channel
attenuates by several per cent over it -- and wide enough that the validity ratios stay
inside the default limit over the whole band.
"""


LENGTH = 2e-4
"""Length of the qualification case, m: a fifth of a millimetre."""


STEPS = 10
"""Steps of the fixed schedule, over a fifth of a millimetre."""


GRID = FieldGrid.from_limits(LENGTH, 800e-9, (400e-9, 1.6e-6), 300e-15, **LEGACY_GRID)
"""One grid for every case here, so that a launch can be built against it.

The former geometry: the records here were made on it, and nothing here is about
the temporal window.
"""


WEAK_COLLAR = absorber(GRID, boundary_length=1.0e3)
"""A collar five million times weaker than the default one.

The default collar is an absorber, and an absorber in the **nonlinear** scheme is an
explicit term whose stability caps the step at about 5 micrometres on this grid
(see docs/dev/decisions.md); it also eats a continuous wave, which is what a single
excited bin of a field grid is, so an absolute statement about the flow could not be
made beside it. This one is present -- the masks and the band limit still are what they
are -- and its rate over a fifth of a millimetre is below 2e-7, far under every bound
taken here. It says so, and the module filters that warning: a collar too weak to stop a
packet walking off the window is exactly what is wanted where nothing walks. The
collar itself is qualified in ``tests/test_boundaries.py``."""


def capillary(
    *,
    wall=None,
    modes: int | Sequence[Mode] = 3,
    pressure: Pressure = 1.0,
    radius: float | ZProfile = RADIUS,
    **overrides,
):
    """Return a small three-channel capillary simulation, with or without a wall.

    A fixed schedule, so that two runs whose operators differ in the last bits still
    land on the same positions: an adaptive controller would otherwise take different
    steps and agree only to its own tolerance, which is orders above the rounding a
    flow identity is about. Its length is set by the temporal collar, which is
    explicit in the nonlinear scheme and stable here only below 5.2 micrometres
    (see docs/dev/decisions.md); :data:`STEPS` steps of a fifth of a millimetre clear
    it.
    """
    arguments = {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-9,
        "modes": modes,
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "plasma": False,
        "raman": False,
    } | overrides
    return Simulation.from_capillary(
        radius,
        LENGTH,
        "He",
        pressure,
        wall=wall,
        **{"noise": None, **arguments},
    )


SMOOTH_STAGES = (0.0, 6e-5, 1.4e-4, 2e-4)
"""Where a taper of this qualification case starts and stops varying, m.

The stages land on multiples of `LENGTH / STEPS`, so the fixed schedule can contain
every hard kink of the radius, which `PropagationProblem.validate` requires of any
schedule (`_check_schedule`).
"""


NARROW = 40e-6
"""The taper's narrow end, m: a fifth off :data:`RADIUS` over a fifth of a millimetre.

A 50 to 40 micrometre contraction has the shape of a 125 to 62.5 micrometre taper --
straight, transition, straight -- at the radius and length a short test allows.
"""


def taper(stages=SMOOTH_STAGES, ends=(RADIUS, NARROW)):
    """Return the straight-transition-straight radius profile of the varying cases."""
    return PiecewiseLinear.from_stages(stages, [ends[0], ends[0], ends[1], ends[1]])


def varying(
    *,
    wall=None,
    radius=None,
    pressure: Pressure = 1.0,
    modes: int | Sequence[Mode] = 3,
    **overrides,
):
    """Return the varying-guide qualification case, with or without a wall."""
    arguments = {
        "wavelength": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": 1e-9,
        "modes": modes,
        "n_save": 2,
        "backend": "numpy_ref",
        "dz_max": LENGTH / STEPS,
        "dz_init": LENGTH / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        "plasma": False,
        "raman": False,
    } | overrides
    return Simulation.from_capillary(
        taper() if radius is None else radius,
        LENGTH,
        "He",
        pressure,
        wall=wall,
        **{"noise": None, **arguments},
    )


VECTOR_WALL = CoherentWall(impedance="te_tm")
"""The declaration under test here: the coupled azimuthal block."""


def vector_modes(
    counts=(2, 2), *, radius: float | ZProfile = RADIUS, orders=(1, -1), **kwargs
):
    """Return one azimuthal order of the qualification capillary."""
    return [
        MarcatiliMode(radius, "He", 1.0, n=order, m=index + 1, **kwargs)
        for order, count in zip(orders, counts, strict=True)
        for index in range(count)
    ]


# --- a general guide -----------------------------------------------------------------

GUIDE_PULSE = {
    "wavelength": 835e-9,
    "fwhm": 20e-15,
    "energy": 1e-12,
    "shape": "gauss",
    "backend": "numpy_ref",
}


def guide_simulation(guide, grid, *, channel=0, length=GENERAL_Z_END, **overrides):
    """Return a simulation of one guide launching one channel."""
    return Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grid,
        channel=channel,
        save_planes=[length],
        dz_max=length / STEPS,
        dz_init=length / STEPS,
        **({"noise": None, **GUIDE_PULSE | overrides}),
    )


# --- the strong coupled-flow cases ---------------------------------------------------

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
    """One coupled-flow problem, its run's request, its grid and its length."""

    problem: PropagationProblem
    run_request: RunRequest
    grid: TimeGrid
    length: float


def _capillary_case(simulation: Simulation) -> Case:
    """Wrap a capillary simulation's problem."""
    return Case(
        simulation.translate(), simulation._run_request(), simulation.grid, LENGTH
    )


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
    simulation = guide_simulation(guide, grid, energy=2e-9)
    return Case(
        simulation.translate(),
        simulation._run_request(),
        simulation.grid,
        GENERAL_Z_END,
    )


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
    assert math.isclose(case.length, float(case.run_request.stepper.z_end))
    # Through the backend's session, which closes it (SEP 47 phase 5).
    return final_field(
        backend(problem, case.run_request, grid=case.grid),
        problem,
        bare(case.run_request),
        Start(
            field=problem.initial.field0 if field0 is None else field0,
            z=float(problem.initial.z),
        ),
        ended=True,
        owns=True,
    )


def parity_bound(
    case: Case,
    expected: np.ndarray,
    linear: np.ndarray,
    formation: float | None = None,
) -> float:
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
    `formation`, when given, replaces ``eps_form`` (a run with bins off the
    eigenbasis route: ``flow_parity.adams_formation_bound``).
    """
    problem = case.problem
    flow = problem.linear.coupled_flow
    assert flow is not None
    eps = float(np.finfo(float).eps)
    channels, _, _ = problem.shape
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
    if formation is None:
        formation = formation_bound(problem, case.run_request, dz)
    carry = 2 * channels * kappa * eps + kappa * formation
    source = (
        2 * 512 * eps * math.log2(problem.grids.n_to)
        + (channels + 84 + 3 * nodes) * eps
    )
    stages = case.run_request.stepper.tableau.n_stages
    per_step = 16 * eps + 2 * carry
    per_stage = rho * (source + 2 * carry) + residual * (terms + 2 * channels) * eps
    growth = (1 + 3 * rho + residual) ** STEPS
    return STEPS * (per_step + stages * per_stage) * growth * scale


def reference_and_linear(case: Case) -> tuple[np.ndarray, np.ndarray]:
    """Return the reference's final state and the launch's linear propagation."""
    expected = propagate(case, NumpyRefBackend)
    linear = 1e3 * propagate(case, NumpyRefBackend, case.problem.initial.field0 * 1e-3)
    return expected, linear
