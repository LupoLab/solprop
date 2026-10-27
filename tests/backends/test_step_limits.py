# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""SEP 14 phase 1: the step limits by route, the start-up rule and their record.

The cumulative table makes a diagonal flow exact across its intervals, so the interval
cap is off there and on for a coupled flow or a reject-free controller; the start-up
rule takes the step after the first from the first step's own estimate (ADR-038).
"""

import dataclasses
import math
from functools import partial
from typing import cast

import numpy as np
import pytest

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import simulation as simulation_module
from solprop.driver.simulation import Simulation
from solprop.gasflow import steady_fill
from solprop.output.schema import StepLimits, StepperRecord, _hashed_stepper
from solprop.stepper import DOPRI5, Multistep, PIController
from solprop.stepper.controller import START_GROWTH
from solprop.translate.chebyshev import build_from_paths
from solprop.translate.coupled import constant_schur_flow
from solprop.translate.problem import _stepper_settings, assemble

LENGTH = 0.05


def history(problem, grid, *, z_start=0.0):
    """Run a problem to its last plane; return the accepted starts and lengths."""
    backend = NumpyRefBackend(problem, grid=grid)
    backend.set_initial(problem.initial.field0, None, z=z_start)
    starts, lengths = [], []
    for plane in problem.stepper.save_planes:
        if plane <= z_start:
            continue
        backend.propagate_to(float(plane))
        steps = backend.observe().step_history
        # The history records where each attempt ended.
        starts += list((steps.z - steps.dz)[steps.accepted])
        lengths += list(steps.dz[steps.accepted])
    backend.close()
    return np.array(starts), np.array(lengths)


def with_limits(problem, **limits):
    """Return the problem with its stepper settings changed."""
    return dataclasses.replace(
        problem, stepper=dataclasses.replace(problem.stepper, **limits)
    )


def test_the_diagonal_route_drops_the_interval_cap(narrow_grid, soliton_problem):
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    assert problem.stepper.interval_cap is False
    # An explicit dz_init is honoured as before; the interfaces ask for the rule.
    assert problem.stepper.auto_init is False
    ruled = soliton_problem(narrow_grid, 1, save_planes=[LENGTH], auto_init=True)
    assert ruled.stepper.auto_init is True
    capped = soliton_problem(narrow_grid, 1, save_planes=[LENGTH], interval_cap=True)
    assert capped.stepper.interval_cap is True


def test_a_reject_free_controller_keeps_the_version_16_limits(
    narrow_grid, soliton_problem
):
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH],
        controller=PIController(rtol=1e-8, atol=1e-18, reject_free=True),
        auto_init=True,
    )
    assert problem.stepper.interval_cap is True
    assert problem.stepper.auto_init is False


def test_a_coupled_flow_keeps_the_version_16_limits(narrow_grid, soliton_problem):
    problem = soliton_problem(narrow_grid, 1, save_planes=[LENGTH])
    # A zero constant-Schur flow on the envelope's frequencies, sorted as a flow's are:
    # the policy reads only that a coupled flow is present.
    flow = constant_schur_flow(
        np.sort(problem.grids.omega),
        problem.linear.phi_breakpoints[[0, -1]],
        np.zeros((1, 1, problem.grids.omega.size), dtype=complex),
        dz_max=problem.stepper.dz_max,
        condition_limit=1e6,
        gauge_id="test",
        certificate_id="constant-schur",
    )
    linear = dataclasses.replace(problem.linear, coupled_flow=flow)
    fields = {
        name: getattr(problem.stepper, name)
        for name in (
            "tableau",
            "dz_min",
            "dz_max",
            "dz_init",
            "stretch",
            "schedule",
            "save_planes",
            "propagator_mode",
        )
    }
    settings = _stepper_settings(
        linear,
        controller=problem.stepper.controller,
        interval_cap=None,
        auto_init=True,
        **fields,
    )
    assert settings.interval_cap is True
    assert settings.auto_init is False


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_without_the_cap_a_step_outgrows_the_table_s_intervals():
    # SEP 13's canonical fill, choked at both vacuum ends: its table bisects to the
    # width floor there, and the cap held every step to those micrometre intervals. A
    # weak pulse on a coarse grid keeps the run short; the table is the fill's.
    fill = steady_fill("He", [0.0, 2.1, 4.2], [0.0, 0.2, 0.0], radius=256e-6)
    simulation = Simulation.from_capillary(
        256e-6,
        4.2,
        "He",
        fill,
        lambda0=800e-9,
        fwhm=10e-15,
        energy=1e-6,
        trange=60e-15,
        wavelength_limits=(400e-9, 1.6e-6),
        rtol=1e-3,
        n_save=2,
        noise=None,
        backend="numpy_ref",
    )
    problem = simulation.translate()
    assert problem.stepper.interval_cap is False
    assert problem.stepper.auto_init is True
    edges = problem.linear.phi_breakpoints
    # The default Adams march carries its own cap, off unless asked for.
    method = problem.stepper.integrator
    assert isinstance(method, Multistep) and method.interval_cap is False
    capped = with_limits(
        problem, integrator=dataclasses.replace(method, interval_cap=True)
    )

    def widest(problem):
        # Each accepted step against the width of the interval it started in.
        starts, lengths = history(problem, simulation.grid)
        # A position accumulates z + dz and can sit an ulp short of an edge it landed
        # on; the driver's landing guard counts eight ulp as reached, and so does this.
        reached = starts + 8 * np.spacing(starts)
        index = np.searchsorted(edges, reached, side="right") - 1
        return float(np.max(lengths / np.diff(edges)[index])), lengths.size

    capped_ratio, capped_steps = widest(capped)
    free_ratio, free_steps = widest(problem)
    # The cap is the containing interval's width, stretched by at most 1.1 onto a
    # landing; without it the steps outgrow the fine intervals at the exit, as far as
    # the history's bounded growth from one step to the next lets them. Measured:
    # 1.0 and 3.3 times the interval, 101 and 93 steps.
    assert capped_ratio <= problem.stepper.stretch
    assert free_ratio > 2.0
    assert free_steps < capped_steps


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_coarser_table_moves_an_uncapped_run_far_less_than_its_tolerance(
    monkeypatch,
):
    # SEP 14 section 4.1's independent table refinement: with steps crossing the
    # operator table's intervals, the table's own error must stay hidden below the
    # run's. A table built to 1e-9 instead of 1e-12 moves the exit field by 3.3e-10 at
    # rtol=1e-6 (1.6e-7 at 1e-3): bounded at 1e-8, a hundredth of the tolerance.
    def exit_field(tol):
        monkeypatch.setattr(simulation_module, "assemble", partial(assemble, tol=tol))
        fill = steady_fill("He", [0.0, 2.1, 4.2], [0.0, 0.2, 0.0], radius=256e-6)
        output = Simulation.from_capillary(
            256e-6,
            4.2,
            "He",
            fill,
            lambda0=800e-9,
            fwhm=10e-15,
            energy=1e-4,
            trange=60e-15,
            wavelength_limits=(400e-9, 1.6e-6),
            rtol=1e-6,
            n_save=2,
            noise=None,
            backend="numpy_ref",
        ).run()
        return output.fields(convention="fft")[-1]

    fine, coarse = exit_field(1e-12), exit_field(1e-9)
    assert np.linalg.norm(coarse - fine) / np.linalg.norm(fine) < 1e-8


def test_a_manufactured_density_integrates_exactly_across_intervals():
    # SEP 14 section 4.1's manufactured witness: a density of a uniform-bore steady
    # fill, rho(z) = sqrt(a + b z), drives an operator linear in it, whose cumulative
    # integral is analytic. An uncapped step spans many of the table's intervals, so
    # the table must give Phi(z2) - Phi(z1) across them to its tolerance, not only
    # within one. Measured 8.4e-14 relative over pairs spanning up to 9 of 18 intervals.
    # bar^2 and bar^2/m: 0.2 bar falling to 1e-4 bar at 4.2 m, near choking.
    a, b = 4e-2, -(4e-2 - 1e-8) / 4.2
    slope = np.array([1.0, -2.5, 0.75])  # 1/m per unit density, three frequency bins
    table = build_from_paths(
        [lambda z: np.sqrt(a + b * z)],
        lambda z, rho: rho[:, None] * slope[None, :],
        0.0,
        4.2,
    ).integral()

    def exact(z):
        return 2.0 / (3.0 * b) * ((a + b * z) ** 1.5 - a**1.5) * slope

    edges = table.breakpoints
    assert edges.size > 10
    rng = np.random.default_rng(11)
    spans = []
    for _ in range(200):
        z1, z2 = np.sort(rng.uniform(0.0, 4.2, 2))
        found = table(z2) - table(z1)
        expected = exact(z2) - exact(z1)
        scale = np.max(np.abs(expected))
        assert np.max(np.abs(found - expected)) <= 1e-12 * max(scale, 1e-3)
        spans.append(np.count_nonzero((edges > z1) & (edges < z2)))
    assert max(spans) > 5


def test_the_start_up_rule_jumps_from_the_first_step(narrow_grid, soliton_problem):
    dz_init = 1e-6
    common = {"save_planes": [LENGTH], "dz_init": dz_init}
    ruled = soliton_problem(narrow_grid, 1, auto_init=True, **common)
    grown = soliton_problem(narrow_grid, 1, **common)
    _, ruled_steps = history(ruled, narrow_grid)
    _, grown_steps = history(grown, narrow_grid)
    assert ruled_steps[0] == grown_steps[0] == dz_init
    # The second step is the deadbeat one, far beyond the 1.1 the cap allows.
    assert ruled_steps[1] > 10 * dz_init
    assert ruled_steps[1] <= START_GROWTH * dz_init * (1 + 1e-12)
    assert grown_steps[1] == pytest.approx(dz_init)
    assert ruled_steps.size < grown_steps.size


def test_a_zero_source_takes_the_largest_start(narrow_grid, linear_problem):
    dz_init = 1e-6
    problem = linear_problem(
        narrow_grid, save_planes=[LENGTH], dz_init=dz_init, auto_init=True
    )
    _, steps = history(problem, narrow_grid)
    # No source, no estimate: the rule grows by its cap, START_GROWTH.
    assert steps[1] == pytest.approx(START_GROWTH * dz_init, rel=1e-12)


def test_a_resumed_run_takes_the_rule_again(narrow_grid, soliton_problem):
    dz_init = 1e-6
    problem = soliton_problem(
        narrow_grid,
        1,
        save_planes=[LENGTH / 2, LENGTH],
        dz_init=dz_init,
        auto_init=True,
    )
    starts, steps = history(problem, narrow_grid, z_start=LENGTH / 2)
    assert starts[0] == LENGTH / 2
    assert steps[0] == dz_init
    assert steps[1] > 10 * dz_init


def test_the_start_factor_is_bounded():
    control = PIController(rtol=1e-6, atol=0.0)
    # A step at tolerance keeps the safety margin; a tiny estimate is capped.
    assert control.start_factor(1.0) == pytest.approx(control.safety)
    assert control.start_factor(1e-40) == START_GROWTH
    assert control.start_factor(math.nan) == control.min_factor
    assert control.start_factor(1e6) == control.min_factor


def test_the_limits_enter_the_resume_key_at_schema_ten():
    control = PIController(rtol=1e-6, atol=0.0)
    old = StepperRecord.from_settings(
        DOPRI5,
        control,
        limits=StepLimits(),
    )
    new = StepperRecord.from_settings(
        DOPRI5,
        control,
        limits=StepLimits(interval_cap=False, auto_init=True),
    )
    # Below schema 10 the two records are the same key: neither field is in it.
    assert _hashed_stepper(old, 9) == _hashed_stepper(new, 9)
    assert _hashed_stepper(old, 10) != _hashed_stepper(new, 10)
    hashed = cast("dict[str, dict[str, object]]", _hashed_stepper(new, 9))
    assert "interval_cap" not in hashed["limits"]
