# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Assemble short scalar split-collar witnesses from public physical inputs."""

import math

import numpy as np

from solprop.boundaries import Absorber
from solprop.constants import C
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.monitor import BoundaryCheck, BoundaryMonitor
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import Translation, assemble


def split_case(
    kind="envelope",
    method="rk",
    *,
    steps=8,
    rate="varying",
    kerr=False,
    observers=(),
):
    """Return an eight-micrometre run with an independently sampled power-loss rate."""
    length, wavelength = 8e-6, 800e-9
    grid_type = FieldGrid if kind == "field" else EnvelopeGrid
    grid = grid_type.from_limits(length, wavelength, (500e-9, 1.2e-6), 200e-15)
    omega = 2 * np.pi * C / wavelength
    mode = SpecifiedMode.from_taylor([omega / C, 1 / C], omega, aeff=1e-8)
    initial = GaussianPulse(wavelength=wavelength, fwhm=20e-15, energy=1e-5).build(grid)

    def loss(times):
        if rate == "zero":
            return np.zeros_like(times)
        if rate == "uniform":
            return np.full_like(times, 0.2 / length)
        # The nonuniform interior loss is intentional: a short run must visibly
        # exercise the split even though this centred pulse never reaches an edge.
        return (0.1 + 0.2 * np.cos(times / 40e-15) ** 2) / length

    boundary = Absorber(
        np.zeros_like(grid.omega),
        loss(grid.t),
        loss(grid.t_fine),
        grid.active_mask,
        length,
    )
    integrator = (
        Multistep(order=3, variable_order=False)
        if method == "adams"
        else LowStorageRungeKutta()
        if method == "low_storage"
        else None
    )
    case = assemble(
        grid,
        mode,
        [KerrResponse(chi3=1e-22, thg=False)] if kerr else [],
        initial,
        density=1.0,
        controller=PIController(rtol=1e-10, atol=1e-16),
        save_planes=[length / 2, length] if steps > 1 else [length],
        dz_init=length / steps,
        dz_max=length / steps,
        schedule=FixedSchedule.uniform(0.0, length, length / steps),
        integrator=integrator,
        absorber=boundary,
        absorber_scheme="split_step",
        stats_spec=StatsSpec(
            names=("energy",) if "statistics" in observers else (),
            periodic=(),
            period=1,
        ),
    )
    if "monitor" in observers:
        watcher = BoundaryMonitor.build(
            grid,
            case.problem.responses.temporal_absorber,
            BoundaryCheck(budget=1.0),
            split=case.problem.flags.has_temporal_absorber,
        )
        case = Translation(case.problem, case.request.for_run(monitor=watcher))
    return case, grid, mode


def linear_split_rounding_bound(case):
    """Bound two loss-only paths using a declared CPU FFT forward-error envelope."""
    problem, request = case.problem, case.request
    assert request.replay is not None
    assert problem.responses.kerr_coefficients is None
    assert not problem.responses.raman_terms and not problem.responses.plasma_channels
    operator = problem.linear.phi_constant
    assert operator is not None and np.all(np.real(operator) <= 0)
    assert problem.responses.absorber_scheme == "split_step"
    size = problem.grids.n_t
    unit = np.finfo(float).eps / 2

    def gamma(count):
        assert count * unit < 1
        return count * unit / (1 - count * unit)

    # Each full-spectrum flow and nonnegative time multiplier is an l2 contraction.
    # One-sided real spectra use their Hermitian extension: sqrt(2) bounds the
    # conversion between its norm and the stored norm, once at the initial input.
    # Per FFT the CPU monitor hypothesis bounds each component by gamma(32*n)
    # times the input l1 norm. Converting to a relative l2 bound costs sqrt(n):
    # sqrt(n)*gamma(32*n) <= gamma(ceil(32*n*sqrt(n))). Two such transforms,
    # exponent formation/evaluation,
    # complex multiplication and the source-free method's stage combinations enter
    # each step. The total gamma absorbs their small multiplicative error growth.
    maximum_phase = float(np.max(np.abs(operator))) * max(request.replay.schedule.steps)
    count = len(request.replay.schedule.steps) * (
        2 * math.ceil(32 * size * math.sqrt(size))
        + 128 * (len(request.integration.tableau.nodes) + 1) ** 2
    )
    relative = (
        gamma(count) + gamma(32) * len(request.replay.schedule.steps) * maximum_phase
    )
    initial = np.linalg.norm(problem.initial.field0.ravel())
    bound = 2 * np.sqrt(2) * relative * initial
    assert np.isfinite(bound) and bound > 0
    return bound
