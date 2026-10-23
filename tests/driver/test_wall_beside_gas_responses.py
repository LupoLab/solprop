# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A coherent wall beside the fill's own plasma and Raman responses.

``prop_capillary`` turns the photoionisation and the molecular Raman response on by
default (``plasma=None`` and ``raman=None`` take ``not envelope`` and "every species
with data"), while the other coherent-wall tests pass ``plasma=False,
raman=False``, so the combination the entry point gives a user by default would
otherwise run on no backend in the suite. Here it runs, through the
entry point with those two arguments **omitted**, on nitrogen -- a molecule with both
optmat's Raman response and its molecular rate -- for the scalar wall, the TE/TM wall
and a taper:

* the compiled backends are ``numpy_ref``'s within a ten-step bound written first
  (:func:`parity_bound`), which extends the flow bound of ``tests/flow_parity.py`` by
  the Raman and plasma sources' own rounding;
* the plasma and the Raman response each move the answer by more than that bound, so
  the parity is about a run in which they act;
* the record names the three responses that ran.

The launch, 30 uJ in 30 fs through a 50 um bore of nitrogen at one bar, was chosen
before any parity measurement from the (then default) MO-ADK rate alone: its peak
field, about 1.8e10 V/m on axis, ionises about 1e-6 of the gas over the pulse, a
plasma phase of order 1e-5 rad over the fifth of a millimetre -- far above rounding
and far below the strong-field regime in which a launch's rounding is amplified by
orders of magnitude. The Kerr phase is a few milliradians.

**Ten fixed steps over a fifth of a millimetre**; nothing here propagates further.
"""

import math
import warnings
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from bounds import scan_bound
from flow_parity import (
    CONTRACTION,
    FFT_CONSTANT,
    GRID,
    LENGTH,
    QUIET,
    STEPS,
    UNIT,
    WEAK_COLLAR,
    departure,
    flow_scales,
    taper,
    ten_step_bound,
    vector_modes,
)
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.registry import available
from solprop.driver import Simulation, prop_capillary
from solprop.modes import CoherentWall
from solprop.nonlinear.optmat_raman import OptmatRamanResponse
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.output.memory import MemoryOutput
from solprop.stepper import FixedSchedule
from solprop.translate.ionisation import RateTable

pytestmark = [
    pytest.mark.filterwarnings(f"ignore::{category.__module__}.{category.__name__}")
    for category in QUIET
] + [
    pytest.mark.filterwarnings("ignore:the temporal absorber attenuates"),
]
"""The qualification cases' deliberate warnings (``flow_parity.QUIET``)."""

GAS = "N2"
"""The fill: a molecule with optmat's molecular rate and Raman response."""

ENERGY = 30e-6
"""Launch energy, J (the module docstring says how it was chosen)."""

CASES = ("scalar", "te_tm", "taper")
"""The wall cases: constant scalar, constant TE/TM, and the varying scalar taper."""

COMPILED = {
    # Per backend: one transform's rounding in eps per log2(n_to), and the
    # synthesis, tensor and projection's contraction count in eps. numba_cpu's are
    # flow_parity's (FFTW against pocketfft); Kokkos's are its approved transform
    # envelope and contraction count (tests/driver/test_native_coupled_flow).
    "numba_cpu": (FFT_CONSTANT, CONTRACTION),
    "native": (512.0, 84),
}
"""The compiled backends and their per-transform and contraction constants."""


def arguments(name, **overrides):
    """Return one case's keyword arguments, with ``plasma`` and ``raman`` omitted."""
    wall, modes, radius = {
        "scalar": (CoherentWall(), 3, 50e-6),
        "te_tm": (CoherentWall(impedance="te_tm"), vector_modes(), 50e-6),
        "taper": (CoherentWall(), 3, taper()),
    }[name]
    return {
        "radius": radius,
        "wall": wall,
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
    } | overrides


def simulation(name, **overrides):
    """Return one case as a simulation object, for the bound's scales."""
    keywords = arguments(name, **overrides)
    radius = keywords.pop("radius")
    with warnings.catch_warnings():
        for category in QUIET:
            warnings.simplefilter("ignore", category)
        return Simulation.from_capillary(radius, LENGTH, GAS, 1.0, **keywords)


def entry_point(name, **overrides) -> MemoryOutput:
    """Run one case through ``prop_capillary`` and return the handler."""
    keywords = arguments(name, **overrides)
    radius = keywords.pop("radius")
    return cast("MemoryOutput", prop_capillary(radius, LENGTH, GAS, 1.0, **keywords))


def launch_source(problem, grid) -> tuple[np.ndarray, float]:
    """Return the reference's source ``N(a_0)`` at the launch, and ``||a_0||``."""
    backend = NumpyRefBackend(problem, grid=grid)
    try:
        state = problem.initial.field0
        bundle = backend._stage_scalars(0.0, 0.0)
        source = backend._nonlinear(state, bundle, np.ones_like(state))
        return np.array(source), float(np.linalg.norm(state))
    finally:
        backend.close()


def without(name, switch) -> np.ndarray:
    """Return the launch source of one case with one response switched off."""
    reduced = simulation(name, **{switch: False})
    return launch_source(reduced.translate(), reduced.grid)[0]


def rate_sensitivity(table: RateTable) -> tuple[float, float]:
    """Return the table's largest ``|d ln W / d ln E|`` and largest ``|ln W|``.

    The derivative of each interval's cubic is a quadratic in ``t``, largest at an end
    of ``[0, d_log_field]`` or at its vertex, so the maximum is exact up to rounding.
    """
    c0, c1, c2, _ = table.coefficients
    step = table.d_log_field
    candidates = [np.abs(c2), np.abs(3 * c0 * step**2 + 2 * c1 * step + c2)]
    with np.errstate(divide="ignore", invalid="ignore"):
        vertex = np.where(c0 != 0.0, -c1 / (3 * c0), 0.0)
    inside = (vertex > 0.0) & (vertex < step)
    candidates.append(
        np.where(inside, np.abs(3 * c0 * vertex**2 + 2 * c1 * vertex + c2), 0.0)
    )
    return float(np.max(candidates)), float(np.max(np.abs(table.log_rate)))


def shares(name, problem, grid) -> tuple[float, float, float]:
    """Return ``Phi_r`` of Kerr, Raman and plasma at the launch of one case.

    The source is a sum over the responses, so removing one and subtracting isolates
    it; each share is its norm over the state's, times the length.
    """
    total, norm = launch_source(problem, grid)
    plasma_source = total - without(name, "plasma")
    raman_source = total - without(name, "raman")
    kerr_source = total - plasma_source - raman_source
    kerr, raman, plasma = (
        LENGTH * float(np.linalg.norm(one)) / norm
        for one in (kerr_source, raman_source, plasma_source)
    )
    return kerr, raman, plasma


def lipschitz_scale(name) -> float:
    """Return ``sum_r k_r Phi_r`` of one case: the source's growth scale over the run.

    ``k = 3`` for Kerr and Raman and ``1 + S`` for the plasma, as
    :func:`parity_bound` states.
    """
    built = simulation(name)
    problem = built.translate()
    plasma_leaf = problem.responses.plasma_coefficients
    assert plasma_leaf is not None
    sensitivity, _ = rate_sensitivity(plasma_leaf.rate)
    phi_kerr, phi_raman, phi_plasma = shares(name, problem, built.grid)
    return 3 * phi_kerr + 3 * phi_raman + (1 + sensitivity) * phi_plasma


def parity_bound(name, backend, reference) -> float:
    """Return the relative bound on a compiled backend's final state, ten steps.

    **Stated before any measurement.** ``tests/flow_parity.py``'s ``B_10`` -- the
    flow's actions, the residual, the diagonal half's ``32 u`` per step and the
    source carried through ``C``, ``exp(Phi)`` and the attenuation -- with the source
    split by response, each response ``r`` entering with its own share ``Phi_r`` (its
    source's norm over the state's, times the length; :func:`without`), its own
    per-stage rounding ``eps_r`` relative to that source, and its own Lipschitz
    factor ``k_r``:

    * **Kerr**, ``k = 3``: flow_parity's fixed-quadrature count,
      ``eps (n + 2 F log2 n_to + K + 3 Q + 2)`` with the backend's transform constant
      ``F`` and contraction ``K`` (:data:`COMPILED`).
    * **Raman**, ``k = 3``: the Kerr count plus the delayed response's convolution on
      the doubled axis, ``eps (2 F log2 (2 n_to) + K)``, as the general source bound
      counts a delayed response (``tests/general_parity.py``).
    * **Plasma**, ``k = 1 + S``: the rate is ``exp`` of a cubic in ``ln |E|``, so a
      relative error ``d`` in a node's field moves it by ``S d`` with ``S`` the
      table's largest ``|d ln W / d ln E|`` (:func:`rate_sensitivity`, exact on the
      cubics), and the cubic's own rounding is ``4 eps |ln W|`` absolute in ``ln W``.
      A node field is synthesised to ``eps (n + F log2 n_to)`` relative to the
      largest one; the rate, and so the fraction and the loss current, are carried by
      fields within a factor two of the largest (halving the field divides the rate
      by ``exp(S / 2)`` at least there), so the pointwise relative error on them is
      twice that, and ``ln |E|`` adds ``2 eps`` more. The fraction's and the currents'
      running integrals are the fixed-geometry scan chain,
      ``192 eps log2 n_to`` (``tests/bounds.py``); the currents'
      synthesis and projection are the Kerr count again:
      ``eps [2 S (n + F log2 n_to + 2) + 4 max |ln W| + 2] + 192 eps log2 n_to``
      plus the Kerr count.

    ``B_10`` is then taken with ``3 Phi_eff = sum_r k_r Phi_r`` (its Lipschitz scale
    and its ``exp(Phi)``) and ``Phi_eff eps_src = sum_r Phi_r eps_r``, which is
    ``B_10``'s per-step source term exactly.
    """
    built = simulation(name)
    problem = built.translate()
    grid = built.grid
    transform, contraction = COMPILED[backend]
    eps = 2.0 * UNIT
    flow = problem.linear.coupled_flow
    assert flow is not None
    n_modes = flow.n_modes
    nodes = problem.transform.n_nodes
    log_n = math.log2(grid.n_to)
    kerr = eps * (n_modes + 2 * transform * log_n + contraction + 3 * nodes + 2)
    raman = kerr + eps * (2 * transform * math.log2(2 * grid.n_to) + contraction)
    plasma_leaf = problem.responses.plasma_coefficients
    assert plasma_leaf is not None
    sensitivity, largest_log = rate_sensitivity(plasma_leaf.rate)
    plasma = (
        kerr
        + eps
        * (2 * sensitivity * (n_modes + transform * log_n + 2) + 4 * largest_log + 2)
        + scan_bound(grid.n_to)
    )
    phi_kerr, phi_raman, phi_plasma = shares(name, problem, grid)
    lipschitz = 3 * phi_kerr + 3 * phi_raman + (1 + sensitivity) * phi_plasma
    phi_eff = lipschitz / 3
    source = (phi_kerr * kerr + phi_raman * raman + phi_plasma * plasma) / phi_eff
    scales = flow_scales(problem, grid, reference)
    return ten_step_bound(replace(scales, phi=phi_eff, source=source))


@pytest.fixture(scope="module")
def reference():
    """Each case's ``numpy_ref`` run through the entry point, once for the module."""
    runs = {}

    def get(name):
        if name not in runs:
            runs[name] = entry_point(name)
        return runs[name]

    return get


def last_field(output: MemoryOutput) -> np.ndarray:
    """Return the last plane's field."""
    field = output.planes()[-1].field
    assert field is not None
    return field


def backend_parameters():
    """Return the (case, backend) pairs, all but the scalar wall's in the slow tier."""
    pairs = []
    for name in CASES:
        for backend in COMPILED:
            marks = [pytest.mark.medium if name == "scalar" else pytest.mark.slow]
            if backend not in available():
                marks.append(pytest.mark.skip(reason=f"{backend} is not installed"))
            pairs.append(
                pytest.param(name, backend, marks=marks, id=f"{name}-{backend}")
            )
    return pairs


@pytest.mark.parametrize(("name", "backend"), backend_parameters())
def test_the_default_responses_beside_a_wall_run_on_the_compiled_backends(
    reference, name, backend, record_property
):
    """Ten steps through ``prop_capillary`` with the default plasma and Raman.

    The compiled backend's last plane is the reference's within :func:`parity_bound`,
    derived and stated before the measurement; the energy statistic within the same
    relative bound; and the two records carry the same flow entry.
    """
    expected = reference(name)
    got = entry_point(name, backend=backend)
    bound = parity_bound(name, backend, last_field(expected))
    measured = departure(last_field(got), last_field(expected))
    record_property("bound", bound)
    record_property("measured", measured)
    assert bound > 0.0
    assert measured <= bound
    energies = [
        float(np.sum(output.planes()[-1].stats["energy"])) for output in (expected, got)
    ]
    assert energies[1] == pytest.approx(energies[0], rel=bound, abs=0.0)
    records = [output.metadata for output in (expected, got)]
    assert all(one is not None and one.modal is not None for one in records)
    assert records[0].modal["flow"] == records[1].modal["flow"]  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("scalar", marks=pytest.mark.medium),
        pytest.param("taper", marks=pytest.mark.slow),
    ],
)
def test_the_plasma_and_the_raman_response_each_move_the_answer(
    reference, name, record_property
):
    """Each default response changes the run by more than the parity bound.

    Bound, stated first: the same case with ``plasma=False``, and with
    ``raman=False``, departs from the default run by **more than the larger of the two
    compiled backends' parity bounds** -- otherwise the parity above would say
    nothing about the response it claims to run. (The module docstring's estimate is
    a plasma phase of order 1e-5 rad, the Raman one comparable to the Kerr phase.)
    """
    expected = last_field(reference(name))
    bound = max(parity_bound(name, backend, expected) for backend in COMPILED)
    for switch in ("plasma", "raman"):
        changed = departure(last_field(entry_point(name, **{switch: False})), expected)
        record_property(switch, changed)
        assert changed > bound, switch
    record_property("bound", bound)


def test_the_record_names_the_responses_that_ran(reference):
    """The record lists Kerr, the plasma and the Raman response, by type.

    ``prop_capillary`` was given neither ``plasma`` nor ``raman``; what it turned on
    is what the record says, the plasma with the rate it chose for a molecule.
    """
    metadata = reference("scalar").metadata
    assert metadata is not None
    responses = metadata.configuration["responses"]
    kinds = [type(one) for one in responses]  # type: ignore[union-attr]
    # optmat's Raman response and rate since SEP 3's Phase 4a (solprop's own
    # RamanResponse and isotropic MO-ADK before, deleted at Phase 4b).
    assert kinds == [KerrResponse, PlasmaResponse, OptmatRamanResponse]
    plasma = responses[1]  # type: ignore[index]
    assert type(plasma.rate).__name__ == "OptmatRate"  # type: ignore[union-attr]
    assert metadata.modal is not None and "flow" in metadata.modal
