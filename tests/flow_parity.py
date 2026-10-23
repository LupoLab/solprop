# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The compiled coupled flow against the reference: cases and bounds.

Shared by ``tests/backends/test_numba_coupled_flow.py`` and the coupled-flow benchmark,
so that the bound a test asserts and the bound the benchmark records are one function.

**The bound, written before any measurement.** Both backends take the Schur factors from
the same host function on the same inputs (asserted). The compiled backends
form the **actions** ``U`` and ``U^-1`` in their own kernels -- from the eigenbasis of
the triangular factor, or by the reference's scaling and squaring where that basis is
ill-conditioned -- so the actions differ from the reference's by the *formation bound*
below, and the arithmetic around them by numpy's ``einsum`` and ``tensordot`` against a
sequential loop, and the complex product numpy fuses into multiply-adds on this host.

*The formation bound*, relative to ``||U||_2``, each backend's action against the exact
``Z exp(l T) Z^H`` of the shared factors, the two added.
The reference's scaling and squaring of ``A = l T`` with ``s`` squarings, ``a =
||A||_2 / 2^s``, ``gamma = sqrt(2) (n + 2) u`` for one complex product of length
``n`` and ``|| |X| ||_2 <= sqrt(n) ||X||_2``: Horner's sixteen steps from the
identity leave ``rho_0 = 16 u + ((gamma + 2 u) n + u) a e^a H_16`` of ``e^a`` (with
the series' tail ``a^17 e^a / 17!``); each squaring doubles it and adds ``n gamma``,
so ``rho_s = 2^s rho_0 + (2^s - 1) n gamma`` of ``e^||A||``; relative to ``||U|| >=
e^-||A||`` and with the back-transform's two products,

``eps_S = e^(2 ||A||) rho_s + 2 n gamma``.

The eigenbasis route ``W exp(l Lambda) Y``: each entry is a sum of ``n`` terms of two
complex products and an exponential accurate to ``(5.83 + 2 max |l lambda|) u``, so it
is within ``(n + 11 + 2 a_max) u |W||E||Y|``, normwise ``kappa (n + 11 + 2 a_max) u
||U||`` with ``kappa = || |W| ||_2 || |Y| ||_2`` (the eigenvalues are ``U``'s, so
``max |e_k| <= ||U||``). ``W`` and ``Y`` are themselves inexact: with ``r = ||W Y -
I||_2`` and ``d = ||W Lambda Y - Z T Z^H||_2``, measured, ``W E Y = e^(l G') (I + R)``
with ``||G' - G|| <= d' = (d + ||T|| r) / (1 - r)``, and ``||e^(l G') - e^(l G)|| <=
l d' C e^(l d' C) ||U||`` (Fréchet, Dyson/Gronwall, with the declared ``C`` bounding
``||e^(s l G)|| ||e^((1-s) l G)|| / ||U||``), so

``eps_E = kappa (n + 11 + 2 a_max) u + r (1 + drift) + drift``, ``drift = l d' C
e^(l d' C)``,

each of ``r`` and ``d`` taken with the rounding of its own measurement. A bin the
backend forms by scaling and squaring is bounded by ``eps_S`` at the reference's
squaring count, which is at least its own.

*One application.* A complex dot product of length ``n`` is within
``gamma = sqrt(2) (n + 2) u`` of exact, elementwise relative to ``|U||x|`` (Higham,
section 3.6, with ``sqrt(2) gamma_2`` for each complex product), so two backends are
within twice it. Normwise, ``|| |U||x| || <= sqrt(n) ||U|| ||x||`` and
``||x|| <= ||U^-1|| ||U x||``, so relative to the result

``eps_apply = 2 sqrt(2) (n + 2) sqrt(n) C u + C eps_form``,

with ``C`` the certificate's declared ``norm(U) norm(U^-1)``, which bounds the
measured one per step (the backend refuses otherwise), and ``eps_form`` the largest
formation bound over the sections, bins, stage lengths and both directions.

*The explicit residual.* ``R_s = sum_t w_t T_t - G_s`` over the ``n_deg`` nonzero
weights, then applied: within ``2 sqrt(2) (n_deg + n + 3) sqrt(n) G u`` of each other
relative to the lab-frame input, where ``G = n_deg max ||T_t|| + max ||G_s||`` in 1/m
majorises ``sum |w_t| ||T_t|| + ||G_s||`` (a Chebyshev weight is at most one in
modulus). It enters a step through the solution weights, ``sum |b_i| = 1.645``, times
``dz``, and is carried back by an action of conditioning ``C``.

*One step.* The accept pass applies ``U`` to the state (``eps_apply``) and to the
solution sum (``eps_apply`` times the step's derivative scale). Each stage applies
``U`` to its input and ``U^-1`` to its source, both perturbations reaching the step
through the solution weights and scaled by ``phi = dz (3 ||N(a)|| / ||a|| + K)``, the
Kerr source's Lipschitz scale plus the declared residual norm ``K``. The diagonal half
and the nonlinear source are the uncoupled route's own arithmetic, 32 u per step between
the two backends (the ladder's 16 eps, ``tests/backends/test_numba_cpu.py``), plus
``1.645 (Phi / S) eps_src`` for the source, with ``eps_src`` the general route's bound
(``general_parity.source_bound``) with its measured ``kappa`` on the general route, and
the same count with ``kappa = 1`` on the capillary's fixed quadrature, where ``Phi`` is
below 1e-3 and the term below 1e-17 whatever ``kappa`` is short of ten thousand:
``eps_step = 32 u + 1.645 (Phi / S) eps_src + eps_apply (1 + 3.29 C phi)``
``           + 1.645 dz eps_R C``.

*Ten steps.* Each step's departure is carried to the end by at most ``C`` per later
step and by ``exp(Phi)`` through the nonlinear half, and is relative to that step's
state, which the wall attenuates, so relative to the final state

``B_10 = C^S exp(Phi) (||a_0|| / ||a_L||) S eps_step``.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np

from general_parity import (
    CONTRACTION,
    FFT_CONSTANT,
    RK_WEIGHTS,
    STEP_CONSTANT,
    UNIT,
    conditioning,
    moving_count,
    source_bound,
)
from modal_provider_cases import (
    GENERAL_CHI3,
    SPLIT_DESCRIPTORS,
    general_grid,
    make_provider,
    split_basis,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import (
    _TAYLOR_ORDER,
    _TAYLOR_SCALE,
    _coupled_flow_maps,
    _general_maps,
)
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import GrazingIncidenceWarning, MaterialRangeWarning
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall, MarcatiliMode
from solprop.modes.guide import ModalGuide
from solprop.modes.material import MaterialKerr
from solprop.profiles import PiecewiseLinear
from solprop.stepper import FixedSchedule
from solprop.translate.coupled import schur_eigenbasis
from solprop.translate.problem import KernelProblem
from solprop.translate.stage import scalars

RADIUS = 50e-6
"""Core radius of the capillary cases, m: the coherent-wall qualification bore."""

NARROW = 40e-6
"""The taper's narrow end, m: the taper-coupling qualification taper."""

LENGTH = 2e-4
"""Length of the capillary cases, m: a fifth of the general cases' millimetre."""

STEPS = 10
"""Fixed steps every case takes."""

ENERGY = 1e-4
"""Pulse energy of the capillary cases, J.

Ten thousand times the coherent-wall qualification's, so that the Kerr half of the step
is exercised rather than a rounding of it: the launch's ``Phi`` is then of order 1e-4,
and a run with the nonlinear half removed departs from the full one by far more than the
bound.
"""

GRID = FieldGrid.from_limits(LENGTH, 800e-9, (400e-9, 1.6e-6), 300e-15)
"""The capillary cases' grid, the coherent-wall qualification's."""

WEAK_COLLAR = absorber(GRID, boundary_length=1.0e3)
"""A weak collar: present, so the in-stage absorber rides every stage."""

CASES = ("scalar", "te_tm", "taper", "gradient", "split", "te", "tm", "lossless")
"""Every parity case, in the order the report lists them.

``"te"`` and ``"tm"`` are the TE/TM wall beside a ``TE_0m`` or a ``TM_0m`` pair, whose
transform resolves E_x and E_y. ``"lossless"`` is the taper coupling without a wall: the
qualification taper with lossless modes, each channel's own propagation constant on the
diagonal."""

QUIET = (MaterialRangeWarning, GrazingIncidenceWarning)
"""The warnings the qualification cases raise on purpose.

``tests/driver/test_coherent_capillary.py`` says why.
"""


def capillary_case(*, wall, radius=RADIUS, pressure=1.0, modes=3, **overrides):
    """Return one capillary case on the ten-step schedule."""
    arguments = {
        "lambda0": 800e-9,
        "grid": GRID,
        "absorber": WEAK_COLLAR,
        "fwhm": 30e-15,
        "energy": ENERGY,
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
        **arguments,
    )


def vector_modes():
    """Return the two radial families of azimuthal order one, two of each."""
    return [
        MarcatiliMode(RADIUS, "He", 1.0, n=order, m=index + 1)
        for order in (1, -1)
        for index in range(2)
    ]


def radial_family(kind):
    """Return ``TE_01``, ``TE_02`` or ``TM_01``, ``TM_02``: E_x and E_y each."""
    return [MarcatiliMode(RADIUS, "He", 1.0, kind=kind, n=0, m=m) for m in (1, 2)]


def taper():
    """Return the qualification straight-transition-straight radius profile."""
    return PiecewiseLinear.from_stages(
        (0.0, 6e-5, 1.4e-4, 2e-4), [RADIUS, RADIUS, NARROW, NARROW]
    )


def split_simulation(**overrides):
    """Return the rotated split pair through the entry point, ten steps.

    The split guide (``modal_provider_cases.split_guide``) with the general parity
    cases' Kerr core, so that the nonlinear half runs beside the flow. With a collar as
    weak as the capillary cases': the entry point's default one is unstable in the
    explicit scheme at 100 um steps (the general parity cases take 2 um steps for it),
    and ten steps over the millimetre are what lets the pair beat.
    """
    grid = general_grid()
    length = 1e-3
    arguments = {
        "absorber": absorber(grid, boundary_length=1.0e3),
        "lambda0": 835e-9,
        "fwhm": 20e-15,
        "energy": 2e-9,
        "pulseshape": "gauss",
        "backend": "numpy_ref",
        "save_planes": [length],
        "dz_max": length / STEPS,
        "dz_init": length / STEPS,
        "schedule": FixedSchedule(np.full(STEPS, length / STEPS)),
    } | overrides
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
    return Simulation.from_waveguide(
        (),
        length,
        provider=guide,
        grid=grid,
        channel=0,
        **arguments,
    )


def build(name, **overrides):
    """Return the simulation of one parity case, any of its keywords overridden."""
    with warnings.catch_warnings():
        for category in QUIET:
            warnings.simplefilter("ignore", category)
        if name == "split":
            return split_simulation(**overrides)
        defaults = {
            "scalar": {},
            "te_tm": {"wall": CoherentWall(impedance="te_tm"), "modes": vector_modes()},
            "taper": {"radius": taper()},
            "gradient": {"pressure": (1.0, 0.1)},
            "te": {
                "wall": CoherentWall(impedance="te_tm"),
                "modes": radial_family("TE"),
            },
            "tm": {
                "wall": CoherentWall(impedance="te_tm"),
                "modes": radial_family("TM"),
            },
            "lossless": {
                "wall": None,
                "radius": taper(),
                "loss": False,
                "cladding": 1.45,
            },
        }[name]
        return capillary_case(**({"wall": CoherentWall()} | defaults | overrides))


@dataclass(frozen=True)
class FlowScales:
    """What the bound reads off one case.

    Attributes
    ----------
    n_modes, n_deg : int
        Channels, and Chebyshev degrees along z of the flow's terms.
    condition : float
        The largest declared ``norm(U) norm(U^-1)``.
    residual : float
        The largest declared residual norm ``K``, 1/m.
    generator : float
        ``n_deg max ||T_t|| + max ||G_s||``, 1/m; zero with no residual.
    phi : float
        The launch's ``L ||N(a)|| / ||a||``.
    dz : float
        The fixed step, m.
    source : float
        The per-stage source bound ``eps_src``.
    attenuation : float
        ``||a_0|| / ||a_L||`` of the reference run.
    formation : float
        The largest formation bound ``eps_form`` of the flow's actions at the step
        the case takes.
    """

    n_modes: int
    n_deg: int
    condition: float
    residual: float
    generator: float
    phi: float
    dz: float
    source: float
    attenuation: float
    formation: float = 0.0


HARMONIC_16 = sum(1.0 / term for term in range(1, _TAYLOR_ORDER + 1))
"""``H_16``, the sum of the Horner steps' ``1 / k``."""


def product_rounding(n_modes):
    """Return ``gamma = sqrt(2) (n + 2) u``, one complex product of length ``n``."""
    return math.sqrt(2.0) * (n_modes + 2) * UNIT


def reference_formation_bound(n_modes, norm, squarings):
    """Return ``eps_S``: scaling and squaring's action against the exact one.

    Parameters are the channel count, ``||l T||_2`` (any upper bound) and the number
    of squarings taken; the result is relative to ``||U||_2``.
    """
    gamma = product_rounding(n_modes)
    scaled = norm / 2.0**squarings
    tail = scaled ** (_TAYLOR_ORDER + 1) * math.exp(scaled) / math.factorial(17)
    horner = (
        16.0 * UNIT
        + ((gamma + 2.0 * UNIT) * n_modes + UNIT)
        * scaled
        * math.exp(scaled)
        * HARMONIC_16
    )
    squared = (
        2.0**squarings * (horner + tail) + (2.0**squarings - 1.0) * n_modes * gamma
    )
    return math.exp(2.0 * norm) * squared + 2.0 * n_modes * gamma


def eigenbasis_formation_bound(n_modes, spread, condition, residual, drift):
    """Return ``eps_E``: the eigenbasis route's action against the exact one.

    ``spread`` is ``max |l lambda|``, ``condition`` the basis's ``kappa``,
    ``residual`` the measured ``||W Y - I||`` and ``drift`` the Fréchet term
    ``l d' C e^(l d' C)``; any of them may be arrays over the bins.
    """
    return (
        condition * (n_modes + 11 + 2.0 * spread) * UNIT
        + residual * (1.0 + drift)
        + drift
    )


def section_formation_bounds(vectors, triangular, lengths, certificate, route=None):
    """Return each bin's formation bound, reference's plus compiled, per length.

    ``vectors`` and ``triangular`` are one section's Schur factors over its bins,
    ``lengths`` the positive stage lengths in m and ``certificate`` the declared
    ``C`` (a float or one per bin). Each bin is bounded by the route the compiled
    backends take for it (:func:`schur_eigenbasis`'s) plus the reference's scaling and
    squaring, at the reference's squaring count for that length (the stack's largest
    row-sum norm); ``route``, when given, replaces the basis's own (a test forcing
    every bin to scaling and squaring). Returns an array of shape
    ``(n_lengths, n_bins)``.
    """
    n_modes = triangular.shape[-1]
    gamma = product_rounding(n_modes)
    left, right, eigenvalues, own = schur_eigenbasis(vectors, triangular)
    route = own if route is None else route
    condition = np.linalg.norm(abs(left), ord=2, axis=(-2, -1)) * np.linalg.norm(
        abs(right), ord=2, axis=(-2, -1)
    )
    residual = np.linalg.norm(left @ right - np.eye(n_modes), ord=2, axis=(-2, -1))
    residual = residual + gamma * condition
    size = np.linalg.norm(triangular, axis=(-2, -1))
    recomposed = vectors @ triangular @ np.conj(np.swapaxes(vectors, -1, -2))
    defect = np.linalg.norm(
        (left * eigenvalues[:, None, :]) @ right - recomposed, ord=2, axis=(-2, -1)
    )
    defect = (
        defect
        + (gamma + 3.0 * UNIT) * condition * np.max(abs(eigenvalues), axis=-1)
        + 2.0 * n_modes * gamma * size
    )
    # A bin the basis does not cover has zero factors and no eigenbasis bound.
    defect = np.where(
        own, (defect + size * residual) / np.where(own, 1.0 - residual, 1.0), np.inf
    )
    rows = float(np.max(np.sum(abs(triangular), axis=-1), initial=0.0))
    bounds = []
    for length in lengths:
        largest = length * rows
        squarings = (
            0
            if largest <= _TAYLOR_SCALE
            else math.ceil(math.log2(largest / _TAYLOR_SCALE))
        )
        reference = np.array(
            [
                reference_formation_bound(n_modes, float(one), squarings)
                for one in length * size
            ]
        )
        with np.errstate(invalid="ignore"):
            growth = length * defect * certificate
            drift = growth * np.exp(growth)
        eigen = eigenbasis_formation_bound(
            n_modes,
            length * np.max(abs(eigenvalues), axis=-1),
            condition,
            residual,
            drift,
        )
        bounds.append(reference + np.where(route, eigen, reference))
    return np.array(bounds)


def formation_bound(problem: KernelProblem, dz: float) -> float:
    """Return ``eps_form`` of a flow at steps up to ``dz``, the largest over everything.

    Every section, active bin, distinct stage fraction and direction
    (:func:`section_formation_bounds`, with the declared certificate's largest row).
    """
    held = _coupled_flow_maps(problem)
    assert held is not None
    certificate = float(np.max(held.condition_bound))
    fractions = {float(node) for node in problem.stepper.tableau.nodes} | {1.0}
    lengths = [fraction * dz for fraction in sorted(fractions) if fraction]
    return max(
        float(
            np.max(section_formation_bounds(vectors, triangular, lengths, certificate))
        )
        for vectors, triangular in zip(held.vectors, held.triangular, strict=True)
    )


def apply_bound(n_modes, condition, formation=0.0):
    """Return ``eps_apply``, two backends' departure in one flow action's product."""
    return (
        2.0 * math.sqrt(2.0) * (n_modes + 2) * math.sqrt(n_modes) * condition * UNIT
        + condition * formation
    )


def residual_bound(n_modes, n_deg, generator):
    """Return ``eps_R``, two backends' departure in a residual's product, 1/m."""
    return (
        2.0
        * math.sqrt(2.0)
        * (n_deg + n_modes + 3)
        * math.sqrt(n_modes)
        * generator
        * UNIT
    )


def ten_step_bound(scales: FlowScales) -> float:
    """Return ``B_10``, the relative bound on the two backends' final states."""
    step_phi = scales.dz * (3.0 * scales.phi / (STEPS * scales.dz) + scales.residual)
    per_step = (
        STEP_CONSTANT * UNIT
        + RK_WEIGHTS * (scales.phi / STEPS) * scales.source
        + apply_bound(scales.n_modes, scales.condition, scales.formation)
        * (1.0 + 2.0 * RK_WEIGHTS * scales.condition * step_phi)
        + RK_WEIGHTS
        * scales.dz
        * residual_bound(scales.n_modes, scales.n_deg, scales.generator)
        * scales.condition
    )
    return (
        scales.condition**STEPS
        * math.exp(scales.phi)
        * scales.attenuation
        * STEPS
        * per_step
    )


def flow_scales(problem: KernelProblem, grid, reference) -> FlowScales:
    """Read the bound's scales off a problem and its reference run's final state."""
    flow = problem.linear.coupled_flow
    assert flow is not None and flow.generator is not None
    active = np.any(flow.generator != 0.0, axis=(0, 1, 2, 3))
    blocks = np.moveaxis(flow.generator[..., active], -1, 2)
    norms = np.linalg.norm(blocks, ord=2, axis=(-2, -1))
    residual = float(np.max(flow.residual_norm_bound))
    generator = 0.0
    if flow.section_methods != ("constant_schur",):
        generator = flow.n_deg * float(np.max(norms)) + float(np.max(norms[:, 0]))
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        state = problem.initial.field0
        bundle = backend._stage_scalars(0.0, 0.0)
        source = backend._nonlinear(state, bundle, np.ones_like(state))
    finally:
        backend.close()
    schedule = problem.stepper.schedule
    assert schedule is not None
    length = float(np.sum(schedule.steps))
    phi = length * float(np.linalg.norm(source) / np.linalg.norm(state))
    maps = _general_maps(problem)
    if maps is not None:
        # The general route: its source bound with the measured conditioning, and a
        # moving guide's per-stage map combination, the conditioning
        # taken on the maps at the launch.
        per_stage = source_bound(
            maps.at(bundle.transform_weights),
            grid.n_to,
            conditioning(
                maps.at(bundle.transform_weights), problem.initial.field0[0], grid
            ),
            moving_count(maps, stage_weights(problem)) if maps.moving else 0.0,
        )
    else:
        per_stage = (
            2.0
            * UNIT
            * (
                flow.n_modes
                + 2 * FFT_CONSTANT * math.log2(grid.n_to)
                + CONTRACTION
                + 3 * problem.transform.n_nodes
                + 2
            )
        )
    return FlowScales(
        n_modes=flow.n_modes,
        n_deg=flow.n_deg,
        condition=float(np.max(flow.inverse_condition_bound)),
        residual=residual,
        generator=generator,
        phi=phi,
        dz=float(np.max(schedule.steps)),
        source=per_stage,
        attenuation=float(np.linalg.norm(state) / np.linalg.norm(reference)),
        formation=formation_bound(problem, float(np.max(schedule.steps))),
    )


def stage_weights(problem: KernelProblem):
    """Return the ``transform_weights`` of every stage of a fixed schedule's run."""
    schedule = problem.stepper.schedule
    assert schedule is not None
    anchors = np.concatenate(([0.0], np.cumsum(schedule.steps)[:-1]))
    return [
        scalars(problem, float(anchor), float(anchor + node * step)).transform_weights
        for anchor, step in zip(anchors, schedule.steps, strict=True)
        for node in problem.stepper.tableau.nodes
    ]


def departure(got, expected):
    """Return the relative L2 departure of two states."""
    return float(np.linalg.norm(got - expected) / np.linalg.norm(expected))


def run(backend_class, problem, grid):
    """Return the final state of the fixed schedule on one backend."""
    backend = backend_class(problem, grid=grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        schedule = problem.stepper.schedule
        assert schedule is not None
        backend.propagate_to(float(np.sum(schedule.steps)))
        return backend.observe().require_field().copy()
    finally:
        backend.close()
