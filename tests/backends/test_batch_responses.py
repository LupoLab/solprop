# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check batch isolation for the tiny modal, Raman and plasma response problems.

The three problem builders below are backend-agnostic
:class:`~solprop.translate.problem.KernelProblem` constructions, small enough
(``n_to <= 64``, eight fixed steps) that a batch and its members can be compared
directly on every backend the batch runner supports, within a rounding budget.
"""

import dataclasses
import math
from contextlib import ExitStack

import numpy as np
import pytest

from backends import test_batch_lockstep as batch_tests
from bounds import EPS, gamma
from former_raman import FORMER_H2, former_response
from rate_oracles import AdkOracle, adk_oracle
from solprop.backends import select, select_batch
from solprop.constants import AU_ENERGY
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import MarcatiliMode, transverse_quadrature
from solprop.noise import modified_noise
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.pulses import GaussianPulse
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.problem import assemble


@pytest.fixture(params=["field", "envelope"])
def modal_case(request):
    # Same vacuum-index Kerr construction as test_polarised_runs, shortened only
    # in its temporal unit-test grid. No gas-fit extrapolation or hidden warning.
    grid_type = FieldGrid if request.param == "field" else EnvelopeGrid
    grid = grid_type.from_limits(0.002, 800e-9, (300e-9, 2e-6), 8e-15)
    modes = [MarcatiliMode(125e-6, m=m, cladding=1.45, loss=False) for m in (1, 2)]
    pulse = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=2e-6).build(grid)
    return grid, modes, pulse


def modal_problem(case, route):
    """Build the two-mode Kerr problem on one of the three component routes."""
    grid, modes, pulse = case
    initial = np.repeat(pulse, 2, axis=0) / math.sqrt(2)
    if route != "scalar":
        initial = np.concatenate(
            [
                math.cos(math.pi / 5) * initial,
                math.sin(math.pi / 5) * np.exp(1j * math.pi / 3) * initial,
            ],
            axis=1,
        )
        if route == "rows":
            modes = [row for mode in modes for row in (mode, mode.rotated(math.pi / 2))]
            initial = initial.reshape(4, 1, grid.n_omega)
    problem = assemble(
        grid,
        modes,
        [KerrResponse(chi3=1e-26, thg=grid.kind == "field")],
        initial,
        controller=PIController(rtol=1e-9, atol=1e-15),
        density=1.0,
        save_planes=[8e-4],
        dz_init=1e-4,
        dz_max=1e-4,
        schedule=FixedSchedule.uniform(0.0, 8e-4, 1e-4),
        quadrature=transverse_quadrature(modes, n_r=4, radial_rule="gauss"),
    )
    return dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=3)
    )


def scalar_raman_problem(kind, with_plasma):
    """Build the two-mode damped-oscillator Raman problem, optionally with plasma."""
    grid_type = FieldGrid if kind == "field" else EnvelopeGrid
    # Plasma's two current streams require a sampled, certified temporal collar.
    window = 400e-15 if with_plasma and kind == "field" else 8e-15
    grid = grid_type.from_limits(0.002, 800e-9, (300e-9, 2e-6), window)
    modes = [MarcatiliMode(125e-6, m=m, cladding=1.45, loss=False) for m in (1, 2)]
    pulse = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=200e-6).build(grid)
    initial = np.repeat(pulse, 2, axis=0) / math.sqrt(2)
    responses = [
        KerrResponse(chi3=2e-52, thg=kind == "field"),
        dataclasses.replace(former_response(FORMER_H2), thg=kind == "field"),
    ]
    if with_plasma:
        # The potential-only ADK form solprop's former ``ADK("H2")`` took, on the
        # former registry's potential, 0.5669 hartree (dc4ca0d): the batch compares
        # with its members, so any fixed rate serves.
        rate = AdkOracle("H2", 0.5669 * AU_ENERGY, 0, 1.0, hydrogenic=True)
        responses.append(PlasmaResponse("H2", rate))
    problem = assemble(
        grid,
        modes,
        responses,
        initial,
        density=1e24,
        ionisation_field_range=(1e8, 8e10) if with_plasma else None,
        controller=PIController(rtol=1e-9, atol=1e-15),
        save_planes=[8e-7],
        dz_init=1e-7,
        dz_max=1e-7,
        schedule=FixedSchedule.uniform(0.0, 8e-7, 1e-7),
        quadrature=transverse_quadrature(modes, n_r=4, radial_rule="gauss"),
    )
    return grid, dataclasses.replace(
        problem, transform=dataclasses.replace(problem.transform, node_chunk=3)
    )


def scalar_plasma_problem(mode_count, preionisation, with_kerr):
    """Build the argon plasma problem with per-species statistics to compare."""
    grid = FieldGrid.from_limits(0.002, 800e-9, (300e-9, 2e-6), 400e-15)
    modes = [
        MarcatiliMode(125e-6, m=m, cladding=1.45, loss=False)
        for m in range(1, mode_count + 1)
    ]
    pulse = GaussianPulse(wavelength=800e-9, fwhm=15e-15, energy=200e-6).build(grid)
    initial = np.repeat(pulse, mode_count, axis=0) / math.sqrt(mode_count)
    responses: list[PlasmaResponse | KerrResponse] = [
        PlasmaResponse("Ar", adk_oracle("Ar"), preionisation=preionisation)
    ]
    if with_kerr:
        responses.append(KerrResponse(chi3=2e-52, thg=True))
    problem = assemble(
        grid,
        modes[0] if mode_count == 1 else modes,
        responses,
        initial,
        controller=PIController(rtol=1e-9, atol=1e-15),
        density=1e24,
        ionisation_field_range=(1e8, 8e10),
        save_planes=[8e-7],
        dz_init=1e-7,
        dz_max=1e-7,
        schedule=FixedSchedule.uniform(0.0, 8e-7, 1e-7),
        quadrature=None
        if mode_count == 1
        else transverse_quadrature(modes, n_r=4, radial_rule="gauss"),
    )
    if mode_count > 1:
        problem = dataclasses.replace(
            problem, transform=dataclasses.replace(problem.transform, node_chunk=3)
        )
    return grid, problem


def compare(problem, grid, backend, *, factory=select, spec=None):
    if spec is None:
        spec = StatsSpec(names=("energy", "peak_power"), periodic=("energy",), period=1)
    problem = dataclasses.replace(problem, stats_spec=spec)
    # A member and its solo run take one method on one fixed schedule: two steps of
    # seven evaluations, each a transform's 32 eps a level and the plasma column's
    # sums, gamma(12 n_to) (the band-rule ladder's count with a single tap).
    n_to = problem.grids.n_to
    bound = 2 * 7 * (32 * EPS * math.ceil(math.log2(n_to)) + gamma(12 * n_to))
    rtol = problem.stepper.controller.rtol
    with ExitStack() as cleanup:
        members, independent = [], []
        for index in range(2):
            draw = modified_noise(
                grid,
                np.random.default_rng(313 + index),
                n_modes=problem.shape[1],
                n_pol=problem.shape[2],
            )
            initial = dataclasses.replace(
                problem.initial,
                field0=problem.initial.field0 * (1.0 + 0.125 * index),
                noise0=draw.spectrum[None],
                noise_model=draw.model,
            )
            one = dataclasses.replace(
                problem,
                initial=initial,
                flags=dataclasses.replace(problem.flags, has_noise=True),
            )
            for destination in (members, independent):
                owner = factory(backend, one, grid=grid)
                cleanup.callback(owner.close)
                owner.set_initial(initial.field0, draw)
                destination.append(owner)
        runner = select_batch(members)
        cleanup.callback(runner.close)
        for target in problem.stepper.schedule.edges[1:3]:
            runner.propagate_to(float(target))
            statistics = runner.stats(spec)
            for index, (actual, owner) in enumerate(
                zip(runner.observe_members(), independent, strict=True)
            ):
                owner.propagate_to(float(target))
                expected = owner.observe()
                batch_tests.assert_observations_agree(actual, expected, rtol, bound)
                for name, value in owner.stats(spec).items():
                    np.testing.assert_allclose(
                        statistics[name][index],
                        value[0],
                        rtol=batch_tests.statistics_bound(expected.field, bound),
                        err_msg=name,
                    )
        runner.close()
        assert runner.memory_ledger() == {}


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
@pytest.mark.parametrize("route", ["scalar", "product", "rows"])
def test_modal_batches_preserve_members(modal_case, backend, route):
    if backend == "native":
        pytest.importorskip("solprop_native")
    compare(modal_problem(modal_case, route), modal_case[0], backend)


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
@pytest.mark.parametrize(
    ("kind", "plasma"), [("envelope", False), ("field", False), ("field", True)]
)
def test_raman_and_plasma_batches_preserve_members(backend, kind, plasma):
    if backend == "native":
        pytest.importorskip("solprop_native")
    grid, problem = scalar_raman_problem(kind, plasma)
    compare(problem, grid, backend)


@pytest.mark.parametrize("backend", ["numpy_ref", "numba_cpu", "native"])
def test_plasma_batches_preserve_species_statistics(backend):
    if backend == "native":
        pytest.importorskip("solprop_native")
    grid, problem = scalar_plasma_problem(2, 0.125, True)
    compare(
        problem,
        grid,
        backend,
        spec=StatsSpec(
            names=("electron_density", "species_ionisation"),
            periodic=("electron_density", "species_ionisation"),
            period=1,
        ),
    )
