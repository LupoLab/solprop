# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The moving solid guides: their tapers, ten-step cases and parity bound.

Shared by the moving-guide tests and the catalogue's rows
(``tests/backends/flow_rows.py``), so that no row imports a test module:

- ``tests/driver/test_moving_guide.py``'s kinked graded-index taper and 10 % step-index
  taper, their grids and pulse, and a guide's simulation on a fixed schedule
  (:func:`moving_simulation`);
- ``tests/driver/test_native_moving_guide.py``'s three Kerr-cored cases (:data:`CASES`),
  each a ``coupled_flow_cases.Case`` of ten fixed steps, and their parity bound,
  written before any measurement: the coupled flow's plus the maps' combination
  (:func:`moving_parity_bound`).
"""

import dataclasses
import warnings

import numpy as np

from coupled_flow_cases import Case, parity_bound
from modal_provider_cases import GENERAL_CHI3, grin_grid, grin_provider, make_provider
from solprop.driver import Simulation
from solprop.grid import FieldGrid
from solprop.modes.guide import ModalGuide, circular_pair, provider_channels
from solprop.modes.material import MaterialKerr
from solprop.profiles import Linear, PiecewiseLinear
from solprop.stepper import FixedSchedule
from solprop.translate.problem import PropagationProblem

# --- the tapers ---------------------------------------------------------------------

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


def moving_simulation(guide, grid, *, steps=STEPS, length=LENGTH, **overrides):
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


STEP_RADIUS = Linear(4e-6, 3.6e-6, LENGTH)
"""A 10 % step-index taper: two HE_1m channels bound over the whole active band."""


def step_grid():
    """Return the 835 nm grid the step-index cases propagate on."""
    return FieldGrid.from_limits(LENGTH, 835e-9, (815e-9, 860e-9), 60e-15)


# --- the Kerr-cored cases ------------------------------------------------------------

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


def ten_step_simulation(guide, grid, length=LENGTH, **pulse) -> Simulation:
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


def as_case(simulation: Simulation, length=LENGTH) -> Case:
    """Wrap a simulation's problem as the coupled-flow parity harness's case."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        problem = simulation.translate()
    return Case(problem, simulation._run_request(), simulation.grid, length)


def grin_kinked() -> Case:
    """The kinked graded-index taper with a Kerr core, two sections of degree four."""
    grid = grin_grid()
    guide = grin_taper(kinked_radius(), kerr=True)
    return as_case(
        ten_step_simulation(guide, grid, **(GRIN_PULSE | {"energy": GRIN_ENERGY}))
    )


def step_guide_with_kerr(descriptors, *, basis_block=None, blocks=()) -> ModalGuide:
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
    guide = step_guide_with_kerr((("HE", 1, 1, "even"), ("HE", 1, 2, "even")))
    return as_case(
        ten_step_simulation(guide, step_grid(), STEP_LENGTH, **STEP_PULSE), STEP_LENGTH
    )


def complex_basis() -> Case:
    """The circular HE11 pair beside the HE12 parity pair, a Kerr core."""
    descriptors = (
        ("HE", 1, 1, "even"),
        ("HE", 1, 1, "odd"),
        ("HE", 1, 2, "even"),
        ("HE", 1, 2, "odd"),
    )
    guide = step_guide_with_kerr(
        descriptors, basis_block=(0, 1), blocks=((0, 1), (2, 3))
    )
    return as_case(
        ten_step_simulation(guide, step_grid(), STEP_LENGTH, **STEP_PULSE), STEP_LENGTH
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


def combination_bound(problem: PropagationProblem) -> float:
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
    stages = case.run_request.stepper.tableau.n_stages
    combination = STEPS * stages * rho * combination_bound(problem) * growth * scale
    return parity_bound(case, expected, linear) + combination
