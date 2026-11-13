# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The free-space routes' nonlinear ladders: their rows of the case catalogue.

**What is compared.** Each free-space route on native against ``numpy_ref``, with one
response family at a time (Kerr on a field grid's analytic signal and on an envelope,
argon's plasma on a field grid under an analytic quadratic rate, an envelope's Raman
oscillator), on the cases the retired ladders built:

- the Cartesian plane (:func:`cartesian_case`, once ``test_free_ladders.py``'s): 9 x 12
  columns in node chunks of seven, which leave a remainder chunk, and the analytic
  family's one plasma channel beside Kerr (:func:`analytic_plasma_case`), whose plasma
  statistics are compared at the launch;
- the line, ``n_y = 1`` (:func:`line_case`, once ``test_line_free.py``'s): 24 points in
  the same chunks, and a Kerr family whose beam's wings sit in a 0.3 collar;
- the radial Hankel route (:func:`radial_case`, once ``test_radial_nonlinear.py``'s):
  16 nodes;
- the Cartesian plane carrying SEP 40's safety ramp over ten steps
  (:func:`ramped_case`, once ``test_kperp_collar.py``'s);
- the radial collar's in-stage term beside a temporal rate ramped across the samples
  (:func:`ramped_rate_case`, once ``test_radial_absorber_ramp.py``'s).

A ladder compares one nonlinear evaluation (``derivative``), the statistics the
session observes at the launch (``launch``, each statistic one by one), one step over
the whole length (``one_step``, or ``steps`` where the step's statistics are read) and
the case's fixed schedule (``steps``: eight steps, and ten on the line) with every
statistic of its specification at the end. One step over the whole length is not the
first of eight, so a ladder's rungs are two rows (three on the line). Every comparison
is peak-normalised (``metric="peak"``) but the radial statistics, which are held
relative at every element, as their old test held them.

**The bounds are the old tests', fixed before native runs.** A field's is a rounding
budget per nonlinear evaluation, seven a step, added (``bounds.ladder_bound``, which
takes the device envelope on a device build): ``32 eps`` per FFT level of the temporal
and transverse transforms (:func:`free_amplitude_bound`) or ``16 eps`` per temporal
level and two ``gamma(n_r)`` for the Hankel pair (:func:`radial_amplitude`), and a
plasma scan's ``gamma(12 n_to)``. A statistic's is twice its field's (the square of a
field) and its reductions (:func:`plane_statistics`, :func:`radial_statistics`). The
ramped rate's is an absolute rounding count over the dot products and transforms of
each bin (:func:`ramped_rate_bound`), relative to the reference's peak.

**The guard.** Each row's term is switched off on the translated problem: the Kerr
coefficient, every plasma population's density or every Raman leaf's rows zeroed
(``optmat_rows.without_kerr``, ``without_plasma``, ``without_raman``), the safety
ramp's rate (:func:`without_ramp`) or the ramped temporal rate
(:func:`without_ramped_rate`). Each guarded rung's reference must then move by more
than ``parity_ladder.DETECTION`` (a thousand) of that rung's bounds. The statistics at
the launch read the launch, which no term has touched, so a row's ``launch`` rung is
unguarded and its field rungs carry the guard.

**Records.** The Cartesian and radial ladders write the accuracy page's records
(``docs/dev/accuracy.md``), as their old tests did: a Cartesian field reading is a rung
of the backend equivalence record and of the spatial accuracy record's ladder series,
and each statistic is a series of its own (:func:`cartesian_record`); the radial
ladder's field and statistics are spatial accuracy series (:func:`radial_record`).

**How to read a row.** Every row is a :func:`free_row` call, for example::

    free_row(
        name=f"cartesian {family} eight steps",
        route="cartesian",
        stepper="rk",
        response=RESPONSES[family],
        features="eight fixed steps, remainder chunk, every statistic",
        why=f"On the Cartesian plane the {family} family's eight fixed steps ...",
        build=partial(built_cartesian, family, steps=8),
        switch_off=SWITCH_OFFS[family],
        rungs={"steps": partial(free_amplitude, family, CARTESIAN, 8)},
        every_statistic=held_as_the_plane(family, CARTESIAN, 8),
        record=partial(cartesian_record, family, 8),
    )

``name`` makes the test id (``free cartesian field_kerr eight steps``); ``route``,
``stepper``, ``response`` and ``features`` are the axes the row covers. ``build`` makes
the problem: here the Cartesian case of one family on a schedule of eight steps.
``rungs`` names what is compared, each with its bound: here the field after the eight
steps, within eight steps' budget; every statistic is then held to its plane's bound.
``switch_off`` removes the row's term for the guard, and ``record`` writes the
readings. ``why`` is the claim a failure quotes.
"""

import dataclasses
import functools
import math
from collections.abc import Callable
from functools import partial
from typing import Any

import numpy as np

from backends.optmat_rows import without_kerr, without_plasma, without_raman
from backends.radial_absorber_cases import problem_for, radial_absorber_case
from backends.spatial_cases import CHI3, LENGTH, QuadraticRate, make_problem
from bounds import gamma, ladder_bound
from legacy_geometry import LEGACY_GRID, bare_absorber
from parity_ladder import Built, Reading, Row, Statistic, reference
from solprop.backends.numpy_ref import NumpyRefBackend, passes
from solprop.backends.session import ExecutionOptions
from solprop.boundaries import SpatialAbsorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController, RungeKutta
from solprop.translate.ionisation import build_rate_table
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import flux_of
from solprop.translate.run_request import RunRequest
from spatial_records import record_equivalence, record_metric

EPS = float(np.finfo(float).eps)
"""The machine epsilon."""

FAMILIES = ("field_kerr", "envelope_kerr", "field_plasma", "envelope_raman")
"""The response families of every free-space ladder."""

LINE_FAMILIES = (*FAMILIES, "field_kerr_collar")
"""The line's families: also the Kerr field family with its beam's wings in a
collar."""

RESPONSES = {
    "field_kerr": "kerr",
    "envelope_kerr": "kerr",
    "field_plasma": "plasma",
    "envelope_raman": "raman",
    "field_kerr_collar": "kerr",
}
"""Each family's response axis."""

SWITCH_OFFS: dict[str, Callable[[PropagationProblem], PropagationProblem]] = {
    "field_kerr": without_kerr,
    "envelope_kerr": without_kerr,
    "field_plasma": without_plasma,
    "envelope_raman": without_raman,
    "field_kerr_collar": without_kerr,
}
"""Each family's term, switched off."""

REMAINDER = ExecutionOptions(node_chunk=7)
"""Seven-column node chunks, which leave a remainder chunk on the plane and the
line."""

CARTESIAN = FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0)
"""The Cartesian ladders' plane, 9 x 12 columns."""

PLASMA_NAMES = ("electron_density", "peak_ionisation_rate")
"""The plasma family's statistics."""

PLANE_STATISTICS = ("energy", "peak_intensity", "on_axis_intensity", "on_axis_spectrum")
"""The Cartesian ladders' statistics; the plasma family adds :data:`PLASMA_NAMES`."""

LINE_STATISTICS = (*PLANE_STATISTICS, "beam_radius", "fwhm_r")
"""The line's statistics: the plane's and the beam's widths."""

RADIAL_STATISTICS = ("energy", "on_axis_intensity", "peak_intensity")
"""The radial ladder's statistics."""

# --- the cases ------------------------------------------------------------------------


def family_responses(family: str) -> tuple[list[Any], float]:
    """Return a family's responses and the density that scales them.

    Argon's plasma under the analytic quadratic rate at 1e22 per cubic metre, an
    envelope's Raman oscillator, or Kerr without its harmonic (on a field grid, the
    analytic family).
    """
    if family.endswith("plasma"):
        table = build_rate_table(QuadraticRate(), (1e3, 1e10))
        return [PlasmaResponse("Ar", table)], 1e22
    if family.endswith("raman"):
        oscillator = DampedOscillator(1e-8, 2e14, undamped_dephasing())
        return [RamanResponse((oscillator,), thg=False)], 1.0
    return [KerrResponse(CHI3, thg=False)], 1.0


def _constructor(family: str) -> Callable[..., Any]:
    """Return a family's grid constructor: a field grid's, or an envelope's."""
    if family.startswith("field"):
        return FieldGrid.from_limits
    return partial(EnvelopeGrid.from_limits, thg=False)


def _assembled(
    family: str, grid: Any, space: Any, energy: float
) -> tuple[PropagationProblem, RunRequest]:
    """Return a family's beam assembled on eight steps, observing its plasma rows.

    The ladders' 30 fs beam of 80 um waist, with a quarter-wave second polarisation but
    on the Raman family.
    """
    responses, density = family_responses(family)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=energy),
        80e-6,
        jones=(1,) if family.endswith("raman") else (1, 0.5j),
    )
    problem, run_request = make_problem(
        grid,
        space,
        pulse.build(grid, space, medium=SpatialMedium()),
        responses,
        density=density,
    )
    names = PLASMA_NAMES if family.endswith("plasma") else ()
    return problem, run_request.for_run(stats=StatsSpec(names=names, periodic=names))


def cartesian_case(family: str) -> tuple[Any, FreeGrid, PropagationProblem, Any]:
    """Return the Cartesian ladders' ``(grid, space, problem, run_request)``.

    A 100 nJ beam on :data:`CARTESIAN` on eight steps of
    ``test_radial_nonlinear.LENGTH``; the plasma family observes its statistics
    periodically.
    """
    # Resolve the held-current collar while retaining the short non-plasma grids.
    window = 400e-15 if family.endswith("plasma") else 150e-15
    # The former geometry, which the stored records' bounds were derived on.
    grid = _constructor(family)(LENGTH, 800e-9, (600e-9, 1.2e-6), window, **LEGACY_GRID)
    problem, run_request = _assembled(family, grid, CARTESIAN, 1e-7)
    return grid, CARTESIAN, problem, run_request


def line_space(family: str) -> LineGrid:
    """Return a family's line: 24 points, the collar family's in a 0.3 collar.

    The collar family puts the beam's wings in the collar, so that the window and the
    per-column scaling are exercised along x; the others have no collar.
    """
    collar = family.endswith("collar")
    return LineGrid(
        3e-4 if collar else 4e-4, 24, collar_fraction=0.3 if collar else 0.0
    )


def line_case(family: str) -> tuple[Any, LineGrid, PropagationProblem, Any]:
    """Return the line ladders' ``(grid, space, problem, run_request)``.

    A millijoule per metre of y, which over the plane's 80 um beam is about its peak
    intensity, on :func:`line_space`, eight steps; the plasma family observes its
    statistics periodically.
    """
    # Resolve the held-current collar while preserving the non-plasma fixtures.
    window = 200e-15 if family == "field_plasma" else 150e-15
    grid = _constructor(family)(LENGTH, 800e-9, (600e-9, 1.2e-6), window)
    space = line_space(family)
    problem, run_request = _assembled(family, grid, space, 1e-3)
    return grid, space, problem, run_request


def radial_case(family: str) -> tuple[Any, RadialGrid, np.ndarray, list[Any], float]:
    """Return the radial ladder's ``(grid, space, launch, responses, density)``.

    The plane's 100 nJ beam on 16 Hankel nodes over 0.6 mm, not yet assembled: a
    ladder's rows assemble it on their own schedules.
    """
    # Resolve the held-current collar while preserving the non-plasma fixtures.
    window = 200e-15 if family == "field_plasma" else 150e-15
    grid = _constructor(family)(LENGTH, 800e-9, (600e-9, 1.2e-6), window)
    space = RadialGrid(6e-4, 16, collar_fraction=0.0)
    jones = (1 + 0j,) if family.endswith("raman") else (1 + 0j, 0.5j)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), 80e-6, jones=jones
    )
    responses, density = family_responses(family)
    start = pulse.build(grid, space, medium=SpatialMedium())
    return grid, space, start, responses, density


def with_steps(run_request: RunRequest, steps: int) -> RunRequest:
    """Return the request on `steps` uniform steps over the case's length."""
    schedule = run_request.stepper.schedule
    assert schedule is not None
    return run_request.with_stepper(
        dataclasses.replace(
            run_request.stepper,
            schedule=type(schedule).uniform(0, LENGTH, LENGTH / steps),
        )
    )


def _observing(run_request: RunRequest, names: tuple[str, ...]) -> RunRequest:
    """Return the request observing the statistics `names`, none periodically."""
    return run_request.for_run(stats=StatsSpec(names=names))


def analytic_plasma_case() -> Built:
    """Return argon's plasma beside Kerr without its harmonic on the Cartesian plane.

    Kerr without its harmonic puts a field grid on the analytic family, whose
    statistics column must pack the analytic signal as the step path does; the plasma
    ladder above has no Kerr term and so never compiles this family. The request
    observes the one channel's electron density, its peak rate and the density per
    species.
    """
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (600e-9, 1.2e-6), 400e-15)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7),
        80e-6,
        jones=(1, 0.5j),
    )
    responses = [
        KerrResponse(CHI3, thg=False),
        PlasmaResponse("Ar", build_rate_table(QuadraticRate(), (1e3, 1e10))),
    ]
    problem, run_request = make_problem(
        grid,
        CARTESIAN,
        pulse.build(grid, CARTESIAN, medium=SpatialMedium()),
        responses,
        density=1e22,
    )
    names = (*PLASMA_NAMES, "species_electron_density")
    return Built(problem, _observing(run_request, names), grid, options=REMAINDER)


def analytic_plasma(problem: PropagationProblem) -> bool:
    """Return whether the problem is one plasma channel on the analytic family."""
    kerr = problem.responses.kerr_coefficients
    return (
        problem.flags.has_plasma
        and problem.flags.n_channels == 1
        and kerr is not None
        and kerr.analytic_signal
    )


def built_cartesian(family: str, *, steps: int) -> Built:
    """Return the Cartesian case on `steps` steps, observing the plane's statistics."""
    grid, _, problem, run_request = cartesian_case(family)
    if steps != 8:
        run_request = with_steps(run_request, steps)
    names = PLANE_STATISTICS + (PLASMA_NAMES if family.endswith("plasma") else ())
    return Built(problem, _observing(run_request, names), grid, options=REMAINDER)


def built_line(family: str, *, steps: int, statistics: bool = True) -> Built:
    """Return the line case on `steps` steps, observing the line's statistics."""
    grid, _, problem, run_request = line_case(family)
    if steps != 8:
        run_request = with_steps(run_request, steps)
    if statistics:
        names = LINE_STATISTICS + (PLASMA_NAMES if family.endswith("plasma") else ())
        run_request = _observing(run_request, names)
    return Built(problem, run_request, grid, options=REMAINDER)


def built_radial(family: str, *, steps: int) -> Built:
    """Return the radial case assembled on `steps` steps, observing its statistics."""
    grid, space, start, responses, density = radial_case(family)
    problem, run_request = make_problem(
        grid, space, start, responses, steps=steps, density=density
    )
    return Built(problem, _observing(run_request, RADIAL_STATISTICS), grid)


# --- the bounds -----------------------------------------------------------------------


def free_amplitude_bound(
    family: str, grid: Any, space: Any, backend: str, steps: int
) -> float:
    """Return the plane's (and the line's) amplitude bound after `steps` steps.

    ``32 eps`` per FFT level of the temporal and both transverse transforms (the line
    counts ``log2 n_y = 0``) and a plasma scan's ``gamma(12 n_to)``, per evaluation;
    seven evaluations a step, one at no steps.
    """
    depth = (
        math.ceil(math.log2(grid.n_to))
        + math.ceil(math.log2(space.n_x))
        + math.ceil(math.log2(space.n_y))
    )
    per_call = 32 * np.finfo(float).eps * depth
    if family.endswith("plasma"):
        per_call += gamma(12 * grid.n_to)
    evaluations = max(1, 7 * steps)
    return ladder_bound(backend, evaluations * per_call, grid.n_to, evaluations)


def free_amplitude(family: str, space: Any, steps: int, built: Built) -> float:
    """Bound a row's field after `steps` steps (:func:`free_amplitude_bound`)."""
    return free_amplitude_bound(family, built.grid, space, "native", steps)


def plane_statistics(family: str, space: Any, steps: int, built: Built) -> float:
    """Bound the plane's statistics after `steps` steps.

    Twice the amplitude bound, for squared fields, plus the column and temporal
    reductions, ``gamma(128 n_columns + 32 n_to)``.
    """
    bound = 2 * free_amplitude(family, space, steps, built)
    return bound + gamma(128 * space.n_columns + 32 * built.grid.n_to)


def fresh_statistics(built: Built) -> float:
    """Bound statistics read afresh: the transforms and the two reductions,
    ``gamma(128 n_columns + 32 n_to)``, no step ladder."""
    return gamma(128 * CARTESIAN.n_columns + 32 * built.grid.n_to)


def held_afresh(built: Built, name: str, field: float) -> Statistic:
    """Hold every statistic read afresh to :func:`fresh_statistics`."""
    return Statistic(fresh_statistics(built))


def held_as_the_plane(family: str, space: Any, steps: int) -> Any:
    """Return the rule holding every statistic to :func:`plane_statistics`."""

    def rule(built: Built, name: str, field: float) -> Statistic:
        return Statistic(plane_statistics(family, space, steps, built))

    return rule


def radial_amplitude(family: str, steps: int, built: Built) -> float:
    """Return the radial ladder's amplitude bound after `steps` steps.

    The modal ladders' CPU FFT-depth envelope, ``16 eps`` per temporal level, and two
    length-``n_r`` radial reductions per evaluation; a plasma scan adds
    ``gamma(12 n_to)``; seven evaluations a step, one at no steps.
    """
    grid = built.grid
    n_r = built.problem.shape[0]
    evaluations = max(1, 7 * steps)
    per_call = 16 * np.finfo(float).eps * math.ceil(math.log2(grid.n_to)) + 2 * gamma(
        n_r
    )
    if family.endswith("plasma"):
        per_call += gamma(12 * grid.n_to)
    return ladder_bound("native", evaluations * per_call, grid.n_to, evaluations)


def radial_statistics(family: str, steps: int) -> Any:
    """Return the rule holding the radial statistics, relative at every element.

    Squared moduli double the amplitude bound; the statistics' transforms and positive
    reductions add ``gamma(128 n_r + 32 ceil(log2 n_t) + 8 n_t)``.
    """

    def rule(built: Built, name: str, field: float) -> Statistic:
        grid = built.grid
        n_r = built.problem.shape[0]
        bound = 2 * radial_amplitude(family, steps, built) + gamma(
            128 * n_r + 32 * int(np.ceil(np.log2(grid.n_t))) + 8 * grid.n_t
        )
        return Statistic(bound, elementwise=True)

    return rule


# --- the safety ramp and the ramped rate ----------------------------------------------

RAMP_STEPS = 10
"""The safety ramp's run: ten fixed steps."""

RAMP_SPACE = FreeGrid(3e-4, 9, 4e-4, 12, collar_fraction=0.1)
"""The safety ramp's plane: the Cartesian ladders' with a tenth of a collar."""


def ramped_case() -> Built:
    """Return the ten-step Cartesian Kerr beam carrying SEP 40's safety ramp.

    A 3 uJ beam of 60 um waist on :data:`RAMP_SPACE`, the ramp designed over the outer
    three tenths of the transverse band (``safety_fraction=0.3``) and no temporal
    collar (the former grid), over a hundred micrometres in ten steps.
    """
    length = 1e-4
    steps = RAMP_STEPS
    # The former grid and no temporal collar: the subject is the transverse ramp.
    grid = EnvelopeGrid.from_limits(
        length, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False, **LEGACY_GRID
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-6), 6e-5
    )
    simulation = Simulation.from_free(
        grid,
        RAMP_SPACE,
        SpatialMedium(),
        [KerrResponse(1e-22, thg=False)],
        pulse,
        settings=SimulationSettings(
            boundary_check=None,
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[length],
            dz_max=length / steps,
            dz_init=length / steps,
            schedule=FixedSchedule.uniform(
                0.0, length, length / steps, save_planes=[length]
            ),
            integrator=RungeKutta(),
            backend="numpy_ref",
            spatial_absorber=SpatialAbsorber(real_space=False, safety_fraction=0.3),
            evanescent_rate=5e3,
            absorber=bare_absorber(grid),
        ),
        noise=None,
    )
    return Built(simulation.translate(), simulation._run_request(), grid)


def ramps(problem: PropagationProblem) -> bool:
    """Return whether some column of the problem carries the safety ramp's rate."""
    separable = problem.linear.separable
    return (
        separable is not None
        and separable.kperp_rate is not None
        and bool(np.any(separable.kperp_rate > 0))
    )


def without_ramp(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with the safety ramp's rate zeroed wherever it is carried.

    The separable operator holds it, and the responses and the flux plan hold the same
    operator.
    """
    separable = problem.linear.separable
    assert separable is not None and separable.kperp_rate is not None
    bare = dataclasses.replace(
        separable, kperp_rate=np.zeros_like(separable.kperp_rate)
    )
    flux = flux_of(problem.route)
    assert flux is not None
    return dataclasses.replace(
        problem,
        linear=dataclasses.replace(problem.linear, separable=bare),
        responses=dataclasses.replace(problem.responses, separable=bare),
        route=dataclasses.replace(
            problem.route, flux=dataclasses.replace(flux, base=bare)
        ),
    )


RAMPED_RATE = (0.0, 2000.0)
"""The ramped temporal rate's ends across the fine samples, 1/m."""


def ramped_rate_case(family: str) -> Built:
    """Return the radial collar's case with a temporal rate ramped across the samples.

    The case of ``test_radial_absorber.py`` (SEP 40's real-space collar on 24 nodes)
    on eight steps, with a rate rising linearly from nothing to 2000 1/m across the
    fine samples beside zeros on the coarse axis, so that a rate read at the wrong
    sample changes the stage.
    """
    case = radial_absorber_case(family)
    grid = case[0]
    base, run_request = problem_for(case, 8)
    rate = np.linspace(*RAMPED_RATE, grid.n_to)
    rate.setflags(write=False)
    # Beside zeros on the coarse axis: the case's zero-rate collar translates to no
    # temporal absorber at all.
    coarse = np.zeros(grid.n_t)
    coarse.setflags(write=False)
    problem = dataclasses.replace(
        base,
        responses=dataclasses.replace(
            base.responses, temporal_absorber=coarse, temporal_absorber_fine=rate
        ),
    )
    return Built(problem, run_request, grid)


def without_ramped_rate(problem: PropagationProblem) -> PropagationProblem:
    """Return the problem with the ramped rate zeroed, beside the coarse zeros."""
    rate = problem.responses.temporal_absorber_fine
    assert rate is not None
    zeros = np.zeros_like(rate)
    zeros.setflags(write=False)
    return dataclasses.replace(
        problem,
        responses=dataclasses.replace(problem.responses, temporal_absorber_fine=zeros),
    )


def ramped_rate_bound(built: Built) -> float:
    """Bound the first stage beside the ramped rate, relative to the reference's peak.

    The retired absolute bound: each time sample is an ``n_r``-term dot product of the
    reference's composed collar matrix plus the rate's term, so it differs by
    ``gamma(n_r + 1)`` of its absolute summands; a bin sums ``n_to`` of those, and an
    FFT adds about ``log2 n_to`` roundings of the same scale; twice that, for the two
    transforms. Divided by the peak of the reference's derivative, as the row
    compares.
    """
    problem = built.problem
    grid = built.grid
    backend = NumpyRefBackend(problem, built.run_request, grid=grid)
    try:
        matrix = backend._spatial_absorber.copy()
    finally:
        backend.close()
    rate = problem.responses.temporal_absorber_fine
    assert rate is not None
    samples = passes.to_time(
        problem.initial.field0,
        problem.grids.crop_index,
        grid.n_to,
        kind=problem.grid_kind,
    )
    n_r = problem.shape[0]
    rows = samples.reshape(n_r, -1)
    absolute = np.abs(matrix) @ np.abs(rows) + (0.5 * rate * np.abs(samples)).reshape(
        n_r, -1
    )
    n_to = grid.n_to
    operations = n_r + 1 + 4 * int(np.log2(n_to))
    bound = 2 * operations * EPS * n_to * float(np.max(absolute))
    return bound / float(np.max(np.abs(reference(built, "derivative"))))


# --- the records ----------------------------------------------------------------------

AXES: dict[str, int | str] = {"derivative": "N", "launch": 0, "one_step": 1}
"""Each rung's axis in the records; a ladder's ``steps`` rung is its count of steps."""


def recorded_family(family: str) -> str:
    """Distinguish the held-current fixture and its resolved collar from history."""
    return "field_plasma held current, 400 fs" if family == "field_plasma" else family


def cartesian_record(
    family: str, steps: int, request: Any, readings: list[Reading]
) -> None:
    """Write a Cartesian row's readings as the old ladders wrote them.

    A field reading is a rung of the backend equivalence record and of the spatial
    accuracy record's ladder series; a statistic is a spatial accuracy series of its
    own. The launch rung's reading over every statistic at once is none.
    """
    label = recorded_family(family)
    axes = AXES | {"steps": steps}
    for reading in readings:
        rung, _, name = reading.rung.partition(" ")
        measured, bound = reading.measured, reading.bound
        if name:
            series = f"Cartesian {label} {name} (relative)"
            record_metric(request, series, axes[rung], measured, bound, "native")
        elif rung != "launch":
            case = ("free", label)
            record_equivalence(request, case, "native", axes[rung], measured, bound)
            series = f"Cartesian {label} ladder (relative)"
            record_metric(request, series, axes[rung], measured, bound, "native")


def radial_record(
    family: str, steps: int, request: Any, readings: list[Reading]
) -> None:
    """Write a radial row's field and statistics as spatial accuracy series."""
    axes = AXES | {"steps": steps}
    for reading in readings:
        rung, _, name = reading.rung.partition(" ")
        series = f"nonlinear ladder {family}{f' {name}' if name else ''} (relative)"
        record_metric(
            request, series, axes[rung], reading.measured, reading.bound, "native"
        )


# --- the rows -------------------------------------------------------------------------

LAUNCH_IS_UNTOUCHED = (
    "The statistics at the launch read the launch, which no response has touched: "
    "switching the term off cannot move them. Kept as the old test's comparison of the "
    "statistics the session reads at the launch; the field rungs carry the guard."
)
"""Why a row's ``launch`` rung is unguarded."""


def free_row(
    *,
    name: str,
    route: str,
    stepper: str = "none",
    response: str,
    features: str,
    build: Callable[[], Built],
    **fields: Any,
) -> Row:
    """Return a free-space ladder's row, peak-normalised, medium when it marches.

    Its id is ``free {name}``; `fields` are the rest of :class:`parity_ladder.Row`.
    """
    marches = {"one_step", "steps", "run"} & set(fields["rungs"])
    return Row(
        id=f"free {name}",
        axes={
            "route": route,
            "stepper": stepper,
            "response": response,
            "features": features,
        },
        build=build,
        medium=bool(marches),
        metric="peak",
        **fields,
    )


def cartesian_rows() -> list[Row]:
    """Return the Cartesian ladders' rows: the launch and one step, and eight steps."""
    rows = []
    for family in FAMILIES:
        rows += [
            free_row(
                name=f"cartesian {family} one step",
                route="cartesian",
                stepper="rk",
                response=RESPONSES[family],
                features="one evaluation, launch statistics, one step, remainder chunk",
                why=f"On the Cartesian plane the {family} family's first stage, the "
                "statistics the session reads at the launch and one step over the "
                "whole length are the reference's.",
                build=partial(built_cartesian, family, steps=1),
                switch_off=SWITCH_OFFS[family],
                rungs={
                    "derivative": partial(free_amplitude, family, CARTESIAN, 0),
                    "launch": partial(plane_statistics, family, CARTESIAN, 0),
                    "one_step": partial(free_amplitude, family, CARTESIAN, 1),
                },
                every_statistic=held_as_the_plane(family, CARTESIAN, 0),
                unguarded={"launch": LAUNCH_IS_UNTOUCHED},
                record=partial(cartesian_record, family, 1),
            ),
            free_row(
                name=f"cartesian {family} eight steps",
                route="cartesian",
                stepper="rk",
                response=RESPONSES[family],
                features="eight fixed steps, remainder chunk, every statistic",
                why=f"On the Cartesian plane the {family} family's eight fixed steps "
                "are the reference's, and so is every statistic at their end.",
                build=partial(built_cartesian, family, steps=8),
                switch_off=SWITCH_OFFS[family],
                rungs={"steps": partial(free_amplitude, family, CARTESIAN, 8)},
                every_statistic=held_as_the_plane(family, CARTESIAN, 8),
                record=partial(cartesian_record, family, 8),
            ),
        ]
    rows.append(
        free_row(
            name="cartesian analytic family plasma statistics at the launch",
            route="cartesian",
            response="kerr, plasma",
            features="analytic family (Kerr without its harmonic on a field grid), "
            "one plasma channel, launch statistics",
            why="On the analytic family native's statistics column packs the analytic "
            "signal as the step path does: the electron density, the peak ionisation "
            "rate and the density per species at the launch are the reference's.",
            bound_note="Statistics read afresh, gamma(128 n_columns + 32 n_to), no "
            "step ladder. Packing the real projection instead halved the field and "
            "quartered the peak rate.",
            build=analytic_plasma_case,
            switch_off=without_plasma,
            expect={"problem": analytic_plasma},
            rungs={"launch": fresh_statistics},
            every_statistic=held_afresh,
        )
    )
    return rows


def line_rows() -> list[Row]:
    """Return the line's rows: the launch and one step, ten steps, and eight steps."""
    rows = []
    for family in LINE_FAMILIES:
        space = line_space(family)
        rows += [
            free_row(
                name=f"line {family} one step",
                route="line",
                stepper="rk",
                response=RESPONSES[family],
                features="n_y = 1, one evaluation, launch statistics, one step",
                why=f"On the line the {family} family's first stage, the statistics "
                "the session reads at the launch and one step over the whole length "
                "are the reference's.",
                build=partial(built_line, family, steps=1),
                switch_off=SWITCH_OFFS[family],
                rungs={
                    "derivative": partial(free_amplitude, family, space, 0),
                    "launch": partial(plane_statistics, family, space, 0),
                    "one_step": partial(free_amplitude, family, space, 1),
                },
                every_statistic=held_as_the_plane(family, space, 0),
                unguarded={"launch": LAUNCH_IS_UNTOUCHED},
            ),
            free_row(
                name=f"line {family} ten steps",
                route="line",
                stepper="rk",
                response=RESPONSES[family],
                features="n_y = 1, ten fixed steps",
                why=f"On the line the {family} family's ten fixed steps are the "
                "reference's.",
                build=partial(built_line, family, steps=10, statistics=False),
                switch_off=SWITCH_OFFS[family],
                rungs={"steps": partial(free_amplitude, family, space, 10)},
            ),
            free_row(
                name=f"line {family} eight steps",
                route="line",
                stepper="rk",
                response=RESPONSES[family],
                features="n_y = 1, eight fixed steps, every statistic",
                why=f"On the line the {family} family's eight fixed steps are the "
                "reference's, and so is every statistic at their end.",
                bound_note="The statistics keep the old test's bound, which counted "
                "ten steps (its parameter) for the case's own schedule of eight: ten "
                "steps' budget on an eight-step march (#807). The field is new, as the "
                "old test compared only the statistics; it is held to eight steps' "
                "budget.",
                build=partial(built_line, family, steps=8),
                switch_off=SWITCH_OFFS[family],
                rungs={"steps": partial(free_amplitude, family, space, 8)},
                every_statistic=held_as_the_plane(family, space, 10),
            ),
        ]
    return rows


def radial_rows() -> list[Row]:
    """Return the radial ladder's rows: one step, and eight steps."""
    rows = []
    for family in FAMILIES:
        rows += [
            free_row(
                name=f"radial {family} one step",
                route="radial",
                stepper="rk",
                response=RESPONSES[family],
                features="one evaluation, one step, every statistic",
                why=f"On the radial route the {family} family's first stage and one "
                "step over the whole length are the reference's, and so is every "
                "statistic at the step's end.",
                build=partial(built_radial, family, steps=1),
                switch_off=SWITCH_OFFS[family],
                rungs={
                    "derivative": partial(radial_amplitude, family, 0),
                    "steps": partial(radial_amplitude, family, 1),
                },
                every_statistic=radial_statistics(family, 1),
                record=partial(radial_record, family, 1),
            ),
            free_row(
                name=f"radial {family} eight steps",
                route="radial",
                stepper="rk",
                response=RESPONSES[family],
                features="eight fixed steps, every statistic",
                why=f"On the radial route the {family} family's eight fixed steps are "
                "the reference's, and so is every statistic at their end.",
                build=partial(built_radial, family, steps=8),
                switch_off=SWITCH_OFFS[family],
                rungs={"steps": partial(radial_amplitude, family, 8)},
                every_statistic=radial_statistics(family, 8),
                record=partial(radial_record, family, 8),
            ),
        ]
    return rows


def ramp_rows() -> list[Row]:
    """Return the safety ramp's run and the radial collar's ramped rate."""
    rows = [
        free_row(
            name="cartesian safety ramp ten steps",
            route="cartesian",
            stepper="rk",
            response="kerr",
            features="SEP 40 safety ramp (k-perp rate), ten fixed steps",
            why="A Cartesian Kerr run carrying the safety ramp's k-perp rate is the "
            "reference's after ten fixed steps.",
            build=ramped_case,
            switch_off=without_ramp,
            expect={"problem": ramps},
            rungs={
                "steps": partial(free_amplitude, "envelope", RAMP_SPACE, RAMP_STEPS)
            },
        )
    ]
    for family in ("field_real", "field_analytic", "envelope"):
        rows.append(
            free_row(
                name=f"radial collar ramped rate {family}",
                route="radial",
                response="kerr" if family == "field_analytic" else "none",
                features="SEP 40 real-space collar, a temporal rate ramped across the "
                "fine samples",
                why="Beside the radial collar's composed matrix native reads a "
                "temporal rate ramped across the fine samples sample by sample, as the "
                "reference does: its first stage is the reference's.",
                bound_note="The retired absolute bound, 2 (n_r + 1 + 4 log2 n_to) "
                "eps n_to times the largest absolute summand, relative to the "
                "reference's peak. The old test's docstring said the ramp had to move "
                "the answer by more than the bound, which nothing asserted: the row's "
                "guard asserts it.",
                build=partial(ramped_rate_case, family),
                switch_off=without_ramped_rate,
                rungs={"derivative": ramped_rate_bound},
            )
        )
    return rows


@functools.cache
def rows() -> tuple[Row, ...]:
    """Return every row of the free-space ladders."""
    return (*cartesian_rows(), *line_rows(), *radial_rows(), *ramp_rows())
