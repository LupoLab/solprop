# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The current rules on compiled backends against the reference (SEPs 18 and 27).

Each case is an existing field-grid plasma problem on one route -- mode-averaged,
multimode, a two-population mixture, radial, Cartesian, the line and the hybrid planar
kind -- under both running-integral rules. The compiled backend's field after one,
eight and ten fixed steps, and its plasma statistics and run
diagnostics, are compared with ``numpy_ref``'s under a bound derived like the other
equivalence ladders': the transforms' depth plus the plasma column's sums, which the
band rule widens by its ``2K + 1`` taps per integrand.
"""

import dataclasses
import math
from typing import cast

import numpy as np
import pytest

from backends.planar_cases import planar_case
from backends.test_radial_nonlinear import LENGTH, QuadraticRate, make_problem
from bounds import gamma, ladder_bound
from driver.test_prop_capillary import SHORT, build
from optmat_parity import INTEGRATION_AGREEMENT
from sessions.runs import run as session_run
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.backends.session import ExecutionOptions
from solprop.grid import FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.nonlinear.responses import PlasmaResponse
from solprop.profiles import Constant
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import PLASMA_INTEGRATION_STATISTICS, StatsSpec
from solprop.stepper import FixedSchedule
from solprop.translate.ionisation import build_rate_table
from solprop.translate.linop import profile_table
from solprop.translate.plasma_integration import design_running_integral

pytestmark = [
    # The planar family's material and collar notes, as in its own parity tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    pytest.mark.filterwarnings(
        "ignore:the temporal absorber attenuates a packet walking off"
        ":solprop.exceptions.SolpropWarning"
    ),
]

STEPS = 2
"""Fixed steps of every case."""

COLD_MIXTURE_DENSITY = 1e-6
"""Relative agreement of the cold-start mixture's electron density.

Without pre-ionisation the mixture's argon barely ionises, so its share of the density
rests on a near-threshold rate that platform-level rounding moves: after one step NumPy
on Linux x86-64 differs from both compiled backends by 4.4e-8 (they agree exactly with
each other, and everything agrees on Apple silicon and after eight steps). The bound
keeps a factor of about 20 over that measurement.
"""

STATISTICS = (
    "electron_density",
    "peak_ionisation_rate",
    *PLASMA_INTEGRATION_STATISTICS,
)


def banded(problem, grid):
    """Return `problem` with its plasma integrated by the grid's band rule."""
    rule = design_running_integral(grid, "band")
    responses = dataclasses.replace(problem.responses, plasma_integral=rule)
    return dataclasses.replace(problem, responses=responses)


def spatial(kind):
    """Return ``(grid, problem, run_request)``: argon on a spatial route."""
    # SEP 27's steady-current certificate requires resolved temporal tapers.
    # The former 150 fs window had only 30 taper samples and residual 9.8e-9.
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 200e-15)
    space = {
        "radial": RadialGrid(6e-4, 16, collar_fraction=0.0),
        "free": FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0),
        "line": LineGrid(4e-4, 24, collar_fraction=0.0),
    }[kind]
    energy = 1e-3 if kind == "line" else 1e-7
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy),
        80e-6,
        jones=(1, 0.5j),
    )
    responses = [PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10)))]
    problem, run_request = make_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        responses,
        steps=STEPS,
        density=1e22,
    )
    return grid, problem, run_request


def modal(kind):
    """Return ``(grid, problem, run_request)``: a capillary, two fixed steps."""
    # `build` forwards any fill, a mapping of components among them.
    gas = cast("str", {"He": 0.5, "Ar": 0.5}) if kind == "mixture" else "He"
    simulation = build(
        gas=gas,
        modes=2 if kind == "multimode" else 1,
        # Without the harmonic the Kerr term reads the analytic signal, and the
        # plasma its real part.
        thg=kind != "analytic",
        plasma_integration="band",
        schedule=FixedSchedule.uniform(0.0, SHORT, SHORT / STEPS),
        absorber_scheme="split_step",
        n_save=2,
    )
    return simulation.grid, simulation.translate(), simulation._run_request()


def planar():
    """Return ``(grid, problem, run_request)``: the hybrid planar field-plasma case."""
    grid, _, problem, run_request = planar_case("field-plasma", steps=STEPS)
    return grid, problem, run_request


CASES = (
    "mode_averaged",
    "analytic",
    "multimode",
    "mixture",
    "radial",
    "free",
    "line",
    "planar",
)


CHUNKS = {"free": 7, "line": 7, "planar": 5}
"""Exact node chunks that leave a remainder: the band column meets a partial block."""


def options_for(name):
    """Return the execution options a case runs with: its remainder chunk, if any."""
    chunk = CHUNKS.get(name)
    return None if chunk is None else ExecutionOptions(node_chunk=chunk)


def case(name):
    """Return the banded ``(grid, problem, run_request)`` of one case."""
    if name in {"mode_averaged", "analytic", "multimode", "mixture"}:
        grid, problem, run_request = modal(name)
    elif name == "planar":
        grid, problem, run_request = planar()
    else:
        grid, problem, run_request = spatial(name)
    problem = banded(problem, grid)
    return grid, problem, run_request.for_run(stats=StatsSpec(names=STATISTICS))


def march(implementation, problem, run_request, grid, *, steps, options=None):
    """Return the field and the statistics after the problem's fixed steps, run
    through the backend's own session."""
    backend = implementation(problem, run_request, grid=grid, options=options)
    ran = session_run(backend, problem, run_request, owns=True)
    assert ran.final.z == float(run_request.stepper.save_planes[-1])
    assert ran.attempts == steps
    return ran.final.field, ran.final.stats


def bound(problem, grid, backend, steps):
    """The equivalence ladder's bound, with the band rule's wider sums."""
    width = problem.responses.plasma_integral.taps.size * 2 - 1
    per_call = 32 * np.finfo(float).eps * math.ceil(math.log2(grid.n_to)) + gamma(
        12 * width * grid.n_to
    )
    evaluations = max(1, 7 * steps)
    return ladder_bound(backend, evaluations * per_call, grid.n_to, evaluations)


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("rule", ["band", "trapezoid"])
@pytest.mark.parametrize("steps", [1, 8, 10])
@pytest.mark.parametrize("preionisation", [0.0, 0.1])
def test_current_rules_match_the_reference_on_every_route(
    name, rule, steps, preionisation, spatial_equivalence_backend_class
):
    implementation = spatial_equivalence_backend_class
    grid, problem, run_request = case(name)
    original_end = float(run_request.stepper.save_planes[-1])
    # The original 0.1 m planar fixture needs 80 Adams steps. This parity ladder
    # keeps 1/8/10 steps over the approved 1e-7 m window: the fully ionised
    # cold-plasma phase rate is at most 2.765e5 /m, giving <0.028 rad in total.
    # The independent planar accuracy/convergence studies keep their own lengths.
    end = 1e-7 if name == "planar" else original_end
    # The weak helium launch has negligible new population. Retain it as an
    # edge case and also seed the existing ten-percent pre-ionisation fixture
    # on every route, so the free-current propagation cannot pass on zeros.
    # Response tables share the translated Phi domain, including its end margin.
    initial = profile_table(Constant(preionisation), *problem.linear.z_span)
    problem = dataclasses.replace(
        problem,
        responses=dataclasses.replace(
            problem.responses,
            plasma_integral=design_running_integral(grid, rule),
            plasma_channels=tuple(
                dataclasses.replace(channel, preionisation=initial)
                for channel in problem.responses.plasma_channels
            ),
        ),
    )
    run_request = run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            save_planes=np.asarray(run_request.stepper.save_planes)
            * (end / original_end),
            schedule=FixedSchedule.uniform(0.0, end, end / steps),
        )
    )
    # The existing ladder counts the selected rule's tap width and seven source
    # evaluations per step; the same bound covers these requested march lengths.
    limit = bound(problem, grid, implementation.name, steps)
    assert limit > 0
    options = options_for(name)
    expected, expected_stats = march(
        NumpyRefBackend, problem, run_request, grid, steps=steps, options=options
    )
    actual, actual_stats = march(
        implementation, problem, run_request, grid, steps=steps, options=options
    )
    if preionisation:
        assert np.max(expected_stats["electron_density"]) > 0
    scale = float(np.max(np.abs(expected)))
    assert float(np.max(np.abs(actual - expected))) <= limit * scale
    # The statistics read the same plane through the same rule. The diagnostics are
    # held absolutely, to a thousandth of their strictest threshold (the mixture's
    # argon barely ionises, and its work defect is rounding noise that one ulp of a
    # platform's exponential moves by tens of per cent: `INTEGRATION_AGREEMENT`).
    for statistic in STATISTICS:
        want = np.asarray(expected_stats[statistic], dtype=float)
        got = np.asarray(actual_stats[statistic], dtype=float)
        if statistic in PLASMA_INTEGRATION_STATISTICS:
            assert np.allclose(got, want, rtol=0.0, atol=INTEGRATION_AGREEMENT), (
                statistic
            )
        else:
            rtol = 1e3 * limit
            if (
                name == "mixture"
                and not preionisation
                and statistic == "electron_density"
            ):
                rtol = max(rtol, COLD_MIXTURE_DENSITY)
            assert np.allclose(got, want, rtol=rtol, atol=0.0), statistic
