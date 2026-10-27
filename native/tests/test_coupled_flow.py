# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""solprop's coupled flow on the native backend, against ``numpy_ref``.

The engine carries each stage input forward and
its source back by the step's actions, adds a frozen-residual flow's explicit
residual, and carries the accepted step by the whole action
(``src/coupled_flow.cpp``); it also forms the actions, from each
section's eigenbasis or by the reference's scaling and squaring, and measures the
whole step's conditioning, which the adapter checks against the certificate. Two
capillary cases are propagated ten fixed steps on both backends: a scalar coherent
wall (one constant section) and a taper under a pressure gradient (a frozen residual
per section). The four-case comparison, the
thread-count probe and the entry-point doors are solprop's
(``tests/driver/test_native_coupled_flow.py``).
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from collections.abc import Iterator
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.numpy_ref.backend import (
    _coupled_flow_maps,  # pyright: ignore[reportPrivateUsage]
    _flow_action,  # pyright: ignore[reportPrivateUsage]
)
from solprop.boundaries import absorber
from solprop.driver import Simulation
from solprop.exceptions import BackendError
from solprop.grid import FieldGrid
from solprop.modes import CoherentWall
from solprop.profiles import PiecewiseLinear
from solprop.stepper import FixedSchedule
from solprop.translate.problem import KernelProblem

from solprop_native import NativeBackend, NativeBatchBackend
from solprop_native.backend import (
    _flow_payload,  # pyright: ignore[reportPrivateUsage]
    _host_flow,  # pyright: ignore[reportPrivateUsage]
)

type ComplexArray = npt.NDArray[np.complex128]
type RealArray = npt.NDArray[np.float64]


def _norms(stack: ComplexArray | RealArray, *, spectral: bool = True) -> RealArray:
    """Return the 2-norm (or Frobenius norm) of each matrix of a stack."""
    if spectral:
        return np.asarray(np.linalg.norm(stack, ord=2, axis=(-2, -1)), dtype=np.float64)
    return np.asarray(np.linalg.norm(stack, axis=(-2, -1)), dtype=np.float64)


RADIUS = 50e-6
"""Bore radius of solprop's coherent-wall qualification case, m."""
LENGTH = 2e-4
"""Length of the ten-step window, m."""
STEPS = 10
GRID = FieldGrid.from_limits(LENGTH, 800e-9, (400e-9, 1.6e-6), 300e-15)
EPS = float(np.finfo(np.float64).eps)


@pytest.fixture(autouse=True)
def silence_case_warnings() -> Iterator[None]:
    """Silence the case's deliberate range and grazing-incidence warnings.

    The band reaches past helium's fitted range and a fifth of a millimetre of a
    50 um bore sits near the quasi-static limit, on purpose: that is what makes the
    wall loss measurable in ten steps (solprop's own module filters the same two).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def simulation(*, radius: float | PiecewiseLinear, wall: CoherentWall) -> Simulation:
    """Return the ten-step capillary case with a strong launch at ten bar."""
    return Simulation.from_capillary(
        radius,
        LENGTH,
        "He",
        10.0 if isinstance(radius, float) else (10.0, 1.0),
        wall=wall,
        lambda0=800e-9,
        grid=GRID,
        absorber=absorber(GRID, boundary_length=1.0e3),
        fwhm=30e-15,
        energy=1e-3,
        modes=3,
        n_save=2,
        backend="numpy_ref",
        dz_max=LENGTH / STEPS,
        dz_init=LENGTH / STEPS,
        schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        plasma=False,
        raman=False,
        noise=None,
    )


def taper() -> PiecewiseLinear:
    """Return solprop's straight-transition-straight 50 to 40 um bore."""
    stages = (0.0, 6e-5, 1.4e-4, 2e-4)
    return PiecewiseLinear.from_stages(stages, [RADIUS, RADIUS, 40e-6, 40e-6])


CASES = {
    "scalar_wall": lambda: simulation(radius=RADIUS, wall=CoherentWall()),
    "taper_gradient": lambda: simulation(radius=taper(), wall=CoherentWall()),
}


def run(
    backend: type[NumpyRefBackend] | type[NativeBackend],
    problem: KernelProblem,
    grid: FieldGrid,
    field0: ComplexArray | None = None,
) -> ComplexArray:
    """Propagate one problem to the end of the window and return the state."""
    driver = backend(problem, grid=grid)
    try:
        driver.set_initial(problem.initial.field0 if field0 is None else field0, None)
        driver.propagate_to(LENGTH)
        return np.asarray(driver.observe().require_field(), dtype=np.complex128).copy()
    finally:
        driver.close()


def _formation_bound(problem: KernelProblem, dz: float) -> float:
    """Return the actions' formation bound, relative to each action; before any run.

    solprop's derivation (``tests/flow_parity.py``), restated: each
    backend's action against the exact ``Z exp(l T) Z^H``, the two added. The
    reference's scaling and squaring of ``A = l T`` with ``s`` squarings is within
    ``e^(2 ||A||) (2^s (rho_0 + tail) + (2^s - 1) n g) + 2 n g`` with ``g = sqrt(2)
    (n + 2) u``, ``a = ||A|| / 2^s`` and ``rho_0 = 16 u + ((g + 2 u) n + u) a e^a
    H_16``; the eigenbasis route within ``kappa (n + 11 + 2 a_max) u + r (1 + drift)
    + drift``, with ``kappa = || |W| || || |Y| ||``, the measured ``r = ||W Y - I||``
    and ``d = ||W Lambda Y - Z T Z^H||`` each with its own rounding, and ``drift = l
    d' C e^(l d')``, ``d' = (d + ||T|| r) / (1 - r)``.
    """
    flow = _host_flow(problem)
    assert flow is not None
    modes = int(flow.vectors.shape[-1])
    gamma = math.sqrt(2.0) * (modes + 2) * EPS / 2
    unit = EPS / 2
    certificate = float(np.max(flow.condition_bound))
    harmonic = sum(1.0 / term for term in range(1, 17))
    fractions = {float(node) for node in problem.stepper.tableau.nodes} | {1.0}
    worst = 0.0
    for section in range(flow.vectors.shape[0]):
        vectors: ComplexArray = flow.vectors[section]
        triangular: ComplexArray = flow.triangular[section]
        left: ComplexArray = flow.left[section]
        right: ComplexArray = flow.right[section]
        eigenvalues: ComplexArray = flow.eigenvalues[section]
        largest: RealArray = np.asarray(np.max(abs(eigenvalues), axis=-1), np.float64)
        condition = _norms(np.abs(left)) * _norms(np.abs(right))
        residual = _norms(left @ right - np.eye(modes)) + gamma * condition
        size = _norms(triangular, spectral=False)
        recomposed = vectors @ triangular @ np.conj(np.swapaxes(vectors, -1, -2))
        defect = _norms((left * eigenvalues[:, None, :]) @ right - recomposed)
        defect = (
            defect + (gamma + 3 * unit) * condition * largest + 2 * modes * gamma * size
        )
        defect = (defect + size * residual) / (1.0 - residual)
        rows = float(np.max(np.sum(np.abs(triangular), axis=-1)))
        for fraction in sorted(fractions):
            if fraction == 0.0:
                continue
            length = fraction * dz
            squarings = max(0, math.ceil(math.log2(max(length * rows, 1e-300) / 0.5)))
            norm: RealArray = length * size
            scaled: RealArray = norm / 2.0**squarings
            horner: RealArray = (
                16 * unit
                + ((gamma + 2 * unit) * modes + unit)
                * scaled
                * np.exp(scaled)
                * harmonic
            )
            tail: RealArray = scaled**17 * np.exp(scaled) / math.factorial(17)
            reference: RealArray = (
                np.exp(2 * norm)
                * (
                    2.0**squarings * (horner + tail)
                    + (2.0**squarings - 1) * modes * gamma
                )
                + 2 * modes * gamma
            )
            drift: RealArray = length * defect * certificate * np.exp(length * defect)
            eigen: RealArray = (
                condition * (modes + 11 + 2 * length * largest) * unit
                + residual * (1 + drift)
                + drift
            )
            assert np.all(flow.route)
            worst = max(worst, float(np.max(reference + eigen)))
    return worst


def _parity_bound(
    problem: KernelProblem, expected: ComplexArray, linear: ComplexArray
) -> float:
    """Return the ten-step bound relative to the launch; fixed before the first run.

    solprop's derivation (``tests/driver/test_native_coupled_flow.py::parity_bound``):
    the actions are within the formation bound of the reference's
    (:func:`_formation_bound`), so each carry is within ``2 n kappa u + kappa
    eps_form``; per step ``16 u`` of shared linear work and the whole step's two
    carries; per stage the
    source's rounding -- two transforms at the approved ``512 u log2 n_to`` and
    ``(n + 84 + 3 Q) u`` -- and the two stage carries, on the nonlinear increment
    ``rho``; per stage the residual's share ``r = dz max ||R||`` rounded in
    ``(T + 2 n) u``; ten steps amplified by ``1 + 3 rho + r`` each.
    """
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    _, channels, _, _ = problem.shape
    kappa = float(np.max(flow.inverse_condition_bound))
    scale = float(np.max(np.abs(problem.initial.field0)))
    rho = float(np.max(np.abs(expected - linear))) / scale / STEPS
    residual = float(np.max(flow.residual_norm_bound)) * LENGTH / STEPS
    terms = int(flow.generator.shape[0] * flow.generator.shape[1])
    carry = 2 * channels * kappa * EPS + kappa * _formation_bound(
        problem, LENGTH / STEPS
    )
    source = (
        2 * 512 * EPS * math.log2(problem.grids.n_to)
        + (channels + 84 + 3 * problem.transform.n_nodes) * EPS
    )
    stages = problem.stepper.tableau.n_stages
    per_step = 16 * EPS + 2 * carry
    per_stage = rho * (source + 2 * carry) + residual * (terms + 2 * channels) * EPS
    return STEPS * (per_step + stages * per_stage) * (1 + 3 * rho + residual) ** STEPS


@pytest.mark.parametrize("name", sorted(CASES))
def test_ten_steps_of_the_flow_are_the_reference_within_the_bound(name: str) -> None:
    case = CASES[name]()
    problem = case.translate()
    grid = cast(FieldGrid, case.grid)
    expected = run(NumpyRefBackend, problem, grid)
    actual = run(NativeBackend, problem, grid)
    small = np.asarray(problem.initial.field0 * 1e-3, dtype=np.complex128)
    linear = 1e3 * run(NumpyRefBackend, problem, grid, small)
    scale = float(np.max(np.abs(problem.initial.field0)))
    bound = _parity_bound(problem, expected, linear)
    assert float(np.max(np.abs(expected - linear))) / scale / STEPS > 1e-4
    assert float(np.max(np.abs(actual - expected))) / scale <= bound


def test_cf4_and_a_batch_are_refused_by_name() -> None:
    case = simulation(radius=taper(), wall=CoherentWall(method="cf4"))
    with pytest.raises(BackendError, match=r"native does not execute a 'cf4' section"):
        NativeBackend(case.translate(), grid=case.grid)
    constant = CASES["scalar_wall"]()
    problem = constant.translate()
    members = [NativeBackend(problem, grid=constant.grid) for _ in range(2)]
    try:
        for member in members:
            member.set_initial(problem.initial.field0, None)
        refusal = r"does not execute linear\.coupled_flow"
        with pytest.raises(BackendError, match=refusal):
            NativeBatchBackend(members)
    finally:
        for member in members:
            member.close()


def test_a_step_beyond_the_certificate_is_refused_by_name() -> None:
    # The certificate guards the Runge-Kutta march's inverse flow action; solprop's
    # default exponential formulation acts forward only and does not measure it.
    case = CASES["scalar_wall"]()
    problem = case.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    shrunk = dataclasses.replace(
        flow, inverse_condition_bound=np.ones_like(flow.inverse_condition_bound)
    )
    spoiled = dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, coupled_flow=shrunk),
        stepper=dataclasses.replace(problem.stepper, integrator=None),
    )
    backend = NativeBackend(spoiled, grid=case.grid)
    try:
        backend.set_initial(spoiled.initial.field0, None)
        with pytest.raises(BackendError, match="native measured a coupled flow"):
            backend.propagate_to(LENGTH)
    finally:
        backend.close()


def test_the_flow_ledger_counts_its_views() -> None:
    case = CASES["taper_gradient"]()
    problem = case.translate()
    flow = problem.linear.coupled_flow
    assert flow is not None
    assert flow.generator is not None
    backend = NativeBackend(problem, grid=case.grid)
    try:
        ledger = backend.memory_ledger()
    finally:
        backend.close()
    _, channels, _, bins = problem.shape
    active = int(np.count_nonzero(np.any(flow.generator != 0.0, axis=(0, 1, 2, 3))))
    block = active * channels * channels * 16
    stages = problem.stepper.tableau.n_stages
    assert ledger["flow_forward"] == ledger["flow_inverse"] == (stages + 1) * block
    assert ledger["flow_frozen"] == len(flow.section_methods) * block
    assert ledger["flow_lab"] == channels * bins * 16
    assert ledger["flow_slot"] == bins * 8


def test_a_length_needing_too_many_squarings_is_refused_by_name() -> None:
    """solprop's squaring limit, on the section's row-sum norm, before the engine runs.

    A refused length leaves no key behind, so the next trial forms its actions.
    """
    case = CASES["scalar_wall"]()
    problem = case.translate()
    backend = NativeBackend(problem, grid=case.grid)
    try:
        backend.set_initial(problem.initial.field0, None)
        whole = backend._stage_scalars(  # pyright: ignore[reportPrivateUsage]
            0.0, LENGTH / STEPS
        )
        with pytest.raises(BackendError, match="native cannot take a coupled flow"):
            backend._flow_trial(  # pyright: ignore[reportPrivateUsage]
                1e30, whole, [whole]
            )
        assert backend._flow_key is None  # pyright: ignore[reportPrivateUsage]
        backend.propagate_to(LENGTH)
    finally:
        backend.close()


def test_the_engine_forms_the_reference_s_actions_within_the_formation_bound() -> None:
    """Every stage's native pair against solprop's reference, bound first.

    The bound is :func:`_formation_bound`, relative to each action; the measured
    conditioning is the reference's singular-value product within ``16 n eps``, half
    the certificate's own allowance.
    """
    case = CASES["taper_gradient"]()
    problem = case.translate()
    reference = _coupled_flow_maps(problem)
    assert reference is not None
    dz = LENGTH / STEPS
    bound = _formation_bound(problem, dz)
    fractions = np.array(
        [*(float(node) for node in problem.stepper.tableau.nodes), 1.0],
        dtype=np.float64,
    )
    backend = NativeBackend(problem, grid=case.grid)
    try:
        native = backend._native_state  # pyright: ignore[reportPrivateUsage]
        shape = (fractions.size, -1, reference.n_modes, reference.n_modes)
        for section in range(reference.frozen.shape[0]):
            measured = native.form_flow_actions(section, dz, fractions)[0]
            forward = native.flow_actions_to_host(False).reshape(shape)
            inverse = native.flow_actions_to_host(True).reshape(shape)
            for slot, fraction in enumerate(fractions):
                if fraction == 0.0:
                    continue
                expected = _flow_action(reference, section, float(fraction) * dz)
                for got, want in zip(
                    (forward[slot], inverse[slot]), expected, strict=True
                ):
                    error = _norms(got - want) / _norms(want)
                    assert float(np.max(error)) <= bound
            pair = _flow_action(reference, section, dz)
            whole = _norms(pair[0]) * _norms(pair[1])
            assert (
                float(np.max(abs(measured / whole - 1.0)))
                <= 16 * reference.n_modes * EPS
            )
    finally:
        backend.close()


def test_the_residual_s_data_are_zero_exactly_outside_the_requested_band() -> None:
    """The collar's residual is an exact zero; the scored bins' data are unchanged.

    As in solprop's reference, the explicit residual is applied only on the requested
    band (``grids.inband_mask``), the frozen flow alone carrying the spectral collar,
    where the wall model is outside its validity, whatever the error norm scores. The
    adapter zeroes the frozen blocks and terms it uploads for the residual there, so
    the engine's sum is ``+0`` and adds only a signed zero;
    everywhere else the data are exact copies of the host flow's, and a problem whose
    every bin is scored uploads exactly what it did before. All of this is a
    structural contract -- hard-mask zeros and lossless copies -- so it stays exact.
    """
    problem = CASES["taper_gradient"]().translate()
    flow = _host_flow(problem)
    assert flow is not None
    assert flow.residual_terms is not None
    scored = problem.grids.inband_mask[flow.active]
    assert np.any(~scored)
    assert np.any(scored)
    payload = _flow_payload(flow)
    frozen = cast("ComplexArray", payload["flow_frozen"])
    terms = cast("ComplexArray", payload["flow_residual_terms"])
    assert np.all(frozen[:, ~scored] == 0.0)
    assert np.all(terms[:, ~scored] == 0.0)
    assert np.array_equal(frozen[:, scored], flow.frozen[:, scored])
    assert np.array_equal(terms[:, scored], flow.residual_terms[:, scored])
    everywhere = dataclasses.replace(
        problem,
        grids=dataclasses.replace(
            problem.grids, inband_mask=np.ones_like(problem.grids.inband_mask)
        ),
    )
    full = _host_flow(everywhere)
    assert full is not None
    unmasked = _flow_payload(full)
    assert np.array_equal(cast("ComplexArray", unmasked["flow_frozen"]), flow.frozen)
    assert np.array_equal(
        cast("ComplexArray", unmasked["flow_residual_terms"]), flow.residual_terms
    )
