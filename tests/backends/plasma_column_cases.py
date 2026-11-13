# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The field plasma column's cases on every route, and their ladder bound (SEPs 18, 27).

Each case is an existing field-grid plasma problem on one route -- mode-averaged (and
its analytic signal), multimode, a two-population mixture, radial, Cartesian, the line
and the hybrid planar kind -- banded by default. :func:`column_case` sets the
running-integral rule, a pre-ionisation and a fixed schedule; :func:`bound` is the
equivalence ladders' bound with the band rule's wider sums. Their rows of the case
catalogue are ``tests/backends/plasma_rows.py``.
"""

import dataclasses
import math
from typing import cast

import numpy as np

from backends.planar_cases import planar_case
from backends.test_radial_nonlinear import LENGTH, QuadraticRate, make_problem
from bounds import gamma, ladder_bound
from driver.test_prop_capillary import SHORT, build
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

STEPS = 2
"""Fixed steps of every case."""

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


PLANAR_WINDOW = 1e-7
"""The planar case's marched length, m.

The original 0.1 m planar fixture needs 80 Adams steps. A march of the parity ladder
keeps its fixed steps over this approved window: the fully ionised cold-plasma phase
rate is at most 2.765e5 /m, giving <0.028 rad in total. The independent planar
accuracy and convergence studies keep their own lengths."""


def column_case(name, rule, *, preionisation=None, steps=None):
    """Return ``(grid, problem, run_request)``: one case under a running-integral rule.

    `rule` is ``"band"`` or ``"trapezoid"``. `preionisation` is the ionised fraction
    every population starts with (the case's own, none, by default); `steps` marches
    the case in that many fixed steps (its own schedule by default), over
    :data:`PLANAR_WINDOW` on the planar route.
    """
    grid, problem, run_request = case(name)
    responses = dataclasses.replace(
        problem.responses, plasma_integral=design_running_integral(grid, rule)
    )
    if preionisation is not None:
        # Response tables share the translated Phi domain, including its end margin.
        initial = profile_table(Constant(preionisation), *problem.linear.z_span)
        responses = dataclasses.replace(
            responses,
            plasma_channels=tuple(
                dataclasses.replace(channel, preionisation=initial)
                for channel in responses.plasma_channels
            ),
        )
    problem = dataclasses.replace(problem, responses=responses)
    if steps is not None:
        original_end = float(run_request.stepper.save_planes[-1])
        end = PLANAR_WINDOW if name == "planar" else original_end
        run_request = run_request.with_stepper(
            dataclasses.replace(
                run_request.stepper,
                save_planes=np.asarray(run_request.stepper.save_planes)
                * (end / original_end),
                schedule=FixedSchedule.uniform(0.0, end, end / steps),
            )
        )
    return grid, problem, run_request


def bound(problem, grid, backend, steps):
    """The equivalence ladder's bound, with the band rule's wider sums.

    It counts the selected rule's tap width and seven source evaluations per step (one
    for ``steps=0``, a single evaluation): the transforms' depth plus the plasma
    column's sums, which the band rule widens by its ``2K + 1`` taps per integrand.
    """
    width = problem.responses.plasma_integral.taps.size * 2 - 1
    per_call = 32 * np.finfo(float).eps * math.ceil(math.log2(grid.n_to)) + gamma(
        12 * width * grid.n_to
    )
    evaluations = max(1, 7 * steps)
    return ladder_bound(backend, evaluations * per_call, grid.n_to, evaluations)
