# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The optmat cases on a backend against the reference: the ladder and its bounds.

Shared by ``tests/backends/test_optmat_backend_parity.py`` and the cost study of
SEP 3 (``docs/dev/seps/archive/sep-0003-optmat-materials.md``), so that the bound
a test asserts and the bound the report records are one function.

**The ladder** (``AGENTS.md``, backends): one linear application, one nonlinear
evaluation, one step, eight steps, each statistic, and a complete short run. The
linear application is one step of the launch scaled by :data:`LINEAR_SCALE`, so that
every pass of the step runs and the cubic and delayed terms are below ``1e-40`` of the
linear one: what the rung compares is the operator built from optmat's indices. The
complete run is the driver's own, ``Simulation.run``, through the output layer, with
its saved planes, per-plane statistics and periodic rows.

**The bound, written before any measurement.** One stage of a capillary case is the
pipeline of ``tests/backends/test_vector_raman_routes.py`` -- transforms of length
``n_to``, the node synthesis and projection over ``n_nodes`` nodes, the plasma's
cumulative scan along ``n_to`` -- whose rounding that suite bounds at
``256 eps`` per fine sample and node. The Raman kernel adds a sum of ``n_rows`` damped
rows at every sample of its doubled axis, one multiply and one add per row, relative
to the sum of the rows' magnitudes: ``n_rows`` more roundings per sample. So one
stage's two backends are within

``e_stage = 256 eps (n_to + n_nodes + n_rows)``

of each other, peak-normalised. A rung of ``k`` stages accumulates ``k e_stage``,
carried by at most ``exp(Phi_k)`` through the nonlinear half, with
``Phi_k = z_k ||N(a_0)|| / ||a_0||`` measured on the reference at the launch, the
growth factor of ``tests/general_parity.py``. A step is seven stages.

A statistic is a functional of the field. Every statistic quadratic in it (energy,
powers, intensities, widths read off a power trace) moves by at most
:data:`FIELD_STATISTIC` times the field's relative deviation: two for the square, and
two again for a width's crossing of a half maximum. A plasma statistic moves through
the rate: by ``2 S`` with ``S = d ln W / d ln F`` taken where the rate is a thousandth
of the run's peak rate (:func:`bounds.rate_log_sensitivity`), which bounds
``S`` over the part of the pulse that ionises, since ``S`` falls as the field rises.
"""

import copy
import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from bounds import rate_log_sensitivity
from optmat_parity_cases import Case, build_case
from solprop import stats
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import PlasmaResponse

EPS = float(np.finfo(float).eps)
"""Machine epsilon of float64."""

SAMPLE_CONSTANT = 256
"""Roundings per fine sample and node of one stage (``test_vector_raman_routes``)."""

STAGES = 7
"""Stages of one Dormand-Prince step."""

LINEAR_SCALE = 1e-20
"""Factor on the launch of the linear rung: the cubic terms are ``1e-40`` of it."""

FIELD_STATISTIC = 4.0
"""Relative sensitivity of a field statistic to the field's relative deviation."""

PLASMA_STATISTICS = frozenset(
    {
        "electron_density",
        "peak_ionisation_rate",
        "ionisation",
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_ionisation_energy",
    }
)
"""Statistics that move through an ionisation rate."""

PLASMA_INTEGRATION_STATISTICS = frozenset(stats.PLASMA_INTEGRATION_STATISTICS)
"""SEP 18's run diagnostics, compared absolutely (:data:`INTEGRATION_AGREEMENT`)."""

RUNGS = ("linear", "nonlinear", "one_step", "eight_steps", "statistics", "run")
"""Every rung, in the order the report lists them."""


@dataclass(frozen=True)
class Ladder:
    """What one backend produced on one case, rung by rung.

    Attributes
    ----------
    linear : numpy.ndarray
        The state after one step of the launch scaled by :data:`LINEAR_SCALE`.
    nonlinear : numpy.ndarray
        The first stage's derivative at the launch, ``N(a_0)``.
    one_step, eight_steps : numpy.ndarray
        The state after one and after eight fixed steps.
    statistics : dict
        Every statistic of the problem's spec after eight steps.
    run_fields : numpy.ndarray
        The complete run's saved planes, optics convention.
    run_statistics : dict
        The complete run's per-plane statistics and periodic rows, by name.
    run_steps : int
        Accepted steps of the complete run.
    """

    linear: np.ndarray
    nonlinear: np.ndarray
    one_step: np.ndarray
    eight_steps: np.ndarray
    statistics: dict[str, np.ndarray]
    run_fields: np.ndarray
    run_statistics: dict[str, np.ndarray]
    run_steps: int


def _backend(backend_class, case: Case, problem=None):
    """Construct a backend on a case with the mode set and profiles the driver gives."""
    problem = case.problem if problem is None else problem
    options: dict[str, Any] = {"grid": case.grid}
    if case.modes is not None:
        options["modes"] = case.modes
    if case.simulation is not None and case.simulation.profiles:
        options["profiles"] = case.simulation.profiles
    return backend_class(problem, **options)


def derivative(backend_class, case: Case) -> np.ndarray:
    """Return the first stage's derivative at the launch, ``(n_modes, n_pol, n_omega)``.

    The probe of ``tests/backends/test_radial_nonlinear.py``: the reference's own
    nonlinear pass, and a compiled backend's first stage of a step of zero length,
    which is the same quantity in the interaction picture at its origin.
    """
    problem = case.problem
    backend = _backend(backend_class, case)
    try:
        backend.set_initial(problem.initial.field0, None)
        if isinstance(backend, NumpyRefBackend):
            backend._hoist(0.0)
            return backend._nonlinear(
                backend._require_state(),
                backend._stage_scalars(0.0, 0.0),
                np.ones(problem.shape, complex),
            )[0]
        backend._trial_step(0.0, 0.0)
        values = (
            backend._native_state.stage_to_host(0)
            if backend.name == "native"
            else backend._stages[0].copy()
        )
        return values.reshape(problem.shape[1:])
    finally:
        backend.close()


def _march(backend_class, case: Case, field, edges):
    """Return the states at each schedule edge given, and the statistics at the last."""
    backend = _backend(backend_class, case)
    try:
        backend.set_initial(field, None)
        states = []
        for edge in edges:
            backend.propagate_to(float(edge))
            states.append(backend.observe().require_field()[0].copy())
        statistics = backend.stats(case.problem.stats_spec)
        return states, {name: np.asarray(v) for name, v in statistics.items()}
    finally:
        backend.close()


def _complete_run(backend_name: str, case: Case):
    """Return the complete run's planes, statistics and accepted steps.

    Through the driver for a driver's case -- a shallow copy whose settings name the
    backend, so the translation is shared -- and at the seam for an assembled one.
    """
    if case.simulation is None:
        from solprop.backends.registry import select  # noqa: PLC0415

        problem = case.problem
        backend = select(backend_name, problem, grid=case.grid, modes=case.modes)
        try:
            backend.set_initial(problem.initial.field0, None)
            planes, statistics, steps = [], {}, 0
            for plane in problem.stepper.save_planes:
                steps += backend.propagate_to(float(plane)).n_steps
                planes.append(backend.observe().require_field()[0].copy())
                for name, value in backend.stats(problem.stats_spec).items():
                    statistics.setdefault(name, []).append(np.asarray(value))
            return (
                np.array(planes),
                {name: np.array(values) for name, values in statistics.items()},
                steps,
            )
        finally:
            backend.close()
    simulation = copy.copy(case.simulation)
    simulation.settings = dataclasses.replace(
        case.simulation.settings, backend=backend_name
    )
    handler = simulation.run(progress="none")
    statistics = {
        name: np.asarray(handler[f"stats/{name}"]) for name in handler.stat_names
    }
    statistics |= {
        f"rows/{name}": np.asarray(handler[f"step_stats/{name}"])
        for name in handler.step_stat_names
    }
    accepted = np.asarray(handler["step_history/accepted"])
    return handler.fields("optics"), statistics, int(np.count_nonzero(accepted))


def ladder(backend_class, case: Case) -> Ladder:
    """Run every rung of the ladder on one backend."""
    problem = case.problem
    schedule = problem.stepper.schedule
    assert schedule is not None
    edges = schedule.edges
    (linear,), _ = _march(
        backend_class, case, problem.initial.field0 * LINEAR_SCALE, [edges[1]]
    )
    (one, eight), statistics = _march(
        backend_class, case, problem.initial.field0, [edges[1], edges[8]]
    )
    fields, run_statistics, steps = _complete_run(backend_class.name, case)
    return Ladder(
        linear=linear,
        nonlinear=derivative(backend_class, case),
        one_step=one,
        eight_steps=eight,
        statistics=statistics,
        run_fields=fields,
        run_statistics=run_statistics,
        run_steps=steps,
    )


def deviation(got, expected) -> float:
    """Return ``max |got - expected| / max |expected|``, NaN where both are NaN.

    A NaN in one and not the other is an infinite deviation.
    """
    got, expected = np.asarray(got), np.asarray(expected)
    missing = np.isnan(got) | np.isnan(expected)
    if np.any(np.isnan(got) != np.isnan(expected)):
        return math.inf
    scale = float(np.max(np.abs(np.where(missing, 0.0, expected)), initial=0.0))
    if scale == 0.0:
        return 0.0 if np.all(np.where(missing, 0.0, got) == 0.0) else math.inf
    return float(np.max(np.abs(np.where(missing, 0.0, got - expected))) / scale)


def stage_bound(case: Case) -> float:
    """Return ``e_stage = 256 eps (n_to + n_nodes + n_rows)``."""
    problem = case.problem
    return (
        SAMPLE_CONSTANT
        * EPS
        * (problem.grids.n_to + problem.transform.n_nodes + case.n_rows)
    )


def growth(case: Case, reference: Ladder, length: float) -> float:
    """Return ``exp(Phi)``, ``Phi = length ||N(a_0)|| / ||a_0||`` on the reference."""
    launch = case.problem.initial.field0[0]
    phi = length * float(np.linalg.norm(reference.nonlinear) / np.linalg.norm(launch))
    return math.exp(phi)


def rate_sensitivity(case: Case, peak_rate: float) -> float:
    """Return the largest ``d ln W / d ln F`` of the case's rates where they ionise.

    Taken where each rate is a thousandth of the run's peak rate; one when the case
    has no plasma or did not ionise.
    """
    if case.simulation is None or not (peak_rate > 0.0):
        return 1.0
    rates = [
        response.rate
        for response in case.simulation.responses
        if isinstance(response, PlasmaResponse | EnvelopePlasmaResponse)
    ]
    sensitivities = [1.0]
    for rate in rates:
        # optmat's rate declares its threshold and maximum field (SEP 3, Phase 1d).
        assert isinstance(rate, OptmatRate)
        low, high = float(rate.threshold), float(rate.maximum_field)
        target = 1e-3 * peak_rate
        if float(rate(high)) <= target or float(rate(low)) >= target:
            continue
        sensitivities.append(rate_log_sensitivity(rate, target, (low, high)))
    return max(sensitivities)


def bounds(case: Case, reference: Ladder) -> dict[str, float]:
    """Return every rung's field bound, from the case and the reference's launch."""
    schedule = case.problem.stepper.schedule
    assert schedule is not None
    edges = schedule.edges
    stage = stage_bound(case)
    length = float(edges[-1])
    return {
        "linear": STAGES * stage,
        "nonlinear": stage,
        "one_step": STAGES * stage * growth(case, reference, float(edges[1])),
        "eight_steps": 8 * STAGES * stage * growth(case, reference, float(edges[8])),
        "run": len(schedule.steps) * STAGES * stage * growth(case, reference, length),
    }


def statistic_bound(case: Case, reference: Ladder, name: str, field: float) -> float:
    """Return one statistic's bound from its field rung's bound ``field``."""
    base = name.removeprefix("rows/")
    if base in PLASMA_INTEGRATION_STATISTICS:
        return INTEGRATION_AGREEMENT
    if base in PLASMA_STATISTICS:
        peaks = [
            float(np.nanmax(values))
            for key, values in reference.run_statistics.items()
            if key.removeprefix("rows/").endswith("peak_ionisation_rate")
            and np.size(values)
        ]
        return 2.0 * rate_sensitivity(case, max(peaks, default=0.0)) * field
    return FIELD_STATISTIC * field


def comparisons(
    case: Case, reference: Ladder, got: Ladder
) -> list[tuple[str, float, float]]:
    """Return ``(rung, measured, bound)`` for every field rung and every statistic.

    Raises ``AssertionError`` if the two ladders disagree on what they hold -- the
    statistics' names, the planes' shapes or the run's accepted steps -- or if the
    reference is not finite and non-zero, so that a backend returning zeros cannot
    pass a peak-normalised comparison.
    """
    assert got.run_steps == reference.run_steps
    assert set(got.statistics) == set(reference.statistics)
    assert set(got.run_statistics) == set(reference.run_statistics)
    field_bounds = bounds(case, reference)
    rows = []
    for rung in ("linear", "nonlinear", "one_step", "eight_steps"):
        expected = getattr(reference, rung)
        assert np.all(np.isfinite(expected)) and np.max(np.abs(expected)) > 0.0
        rows.append((rung, deviation(getattr(got, rung), expected), field_bounds[rung]))
    assert got.run_fields.shape == reference.run_fields.shape
    assert np.max(np.abs(reference.run_fields)) > 0.0
    rows.append(
        ("run", deviation(got.run_fields, reference.run_fields), field_bounds["run"])
    )
    for name, expected in reference.statistics.items():
        bound = statistic_bound(case, reference, name, field_bounds["eight_steps"])
        rows.append(
            (f"statistic {name}", compare(name, got.statistics[name], expected), bound)
        )
    for name, expected in reference.run_statistics.items():
        bound = statistic_bound(case, reference, name, field_bounds["run"])
        rows.append(
            (f"run {name}", compare(name, got.run_statistics[name], expected), bound)
        )
    return rows


INTEGRATION_AGREEMENT = 1e-7
"""Absolute agreement of SEP 18's run diagnostics across backends: a thousandth of the
smallest level at which one would mean anything (a work defect of 1e-4; an undershoot
matters from about 1e-2).

They are dimensionless ratios, and the work defect is a difference of nearly equal sums,
so a relative deviation measures nothing. Nor can they be held to the sums' rounding:
where a channel barely ionises, the fraction ``1 - (1 - p) exp(-S)`` carries an absolute
rounding of about 1e-16, so the work defect of a peak fraction of 1e-9 is noise of order
1e-8 that moves by tens of per cent with one ulp of ``exp``, and the platforms'
``exp`` differ by ulps."""


def compare(name: str, got, expected) -> float:
    """Return one statistic's deviation: relative, or absolute for the diagnostics."""
    if name.removeprefix("rows/") in PLASMA_INTEGRATION_STATISTICS:
        return float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
    return deviation(got, expected)


@functools.cache
def reference(name: str) -> Ladder:
    """Return the reference's ladder on one case, once per process."""
    return ladder(NumpyRefBackend, build_case(name))


def check_ladder(name: str, backend_class) -> list[tuple[str, float, float]]:
    """Assert every rung of one case on one backend inside its bound.

    Returns the rows, so that a caller can record them.
    """
    case = build_case(name)
    rows = comparisons(case, reference(name), ladder(backend_class, case))
    outside = [row for row in rows if not 0.0 <= row[1] < row[2]]
    assert not outside, f"{backend_class.name} on {name}: {outside}"
    return rows
