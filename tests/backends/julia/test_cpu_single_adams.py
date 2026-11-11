# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Apply SEP33's exact Adams-versus-RK rule on preselected short CPU32 windows."""

import json
import os
from dataclasses import dataclass, replace
from functools import cache
from typing import cast

import numpy as np
import pytest

from backends.julia.test_cpu_single import (
    _band_partition_budget,
    _bands,
    _method,
    _statistics,
)
from sessions.executors import IDENTITY, Started, drive
from solprop.backends.events import Outcome
from solprop.backends.julia.backend import JuliaBackend
from solprop.backends.registry import select
from solprop.backends.session import ExecutionOptions, Start
from solprop.driver import Simulation
from solprop.stats import StatsSpec
from solprop.stepper import RungeKutta
from solprop.translate.precision import SINGLE
from solprop.translate.problem import Translation

pytestmark = [
    pytest.mark.medium,
    pytest.mark.skipif(
        os.environ.get("SOLPROP_TEST_JULIA") != "1",
        reason="the packaged Julia runtime lane requires SOLPROP_TEST_JULIA=1",
    ),
]

WINDOW = 100e-6
STEP = WINDOW / 8
LIMITS = {"field": 1e-3, "bands": 1e-2, "energy": 1e-4, "peak": 1e-3, "fraction": 1e-3}
REFERENCE_PAIRS = ((1e-9, 1e-10), (1e-10, 1e-11), (1e-11, 1e-12))


@dataclass(frozen=True)
class _Reading:
    field: np.ndarray | None
    values: dict[str, np.ndarray]
    work: dict[str, int]
    outcome: str
    error: str
    population_present: bool


@cache
def _scientific_case(name):
    assert name in {"helium", "argon_field", "argon_envelope"}
    helium = name == "helium"
    simulation = Simulation.from_capillary(
        125e-6 if helium else 13e-6,
        1.5 if helium else 0.2,
        "He" if helium else "Ar",
        (0.8, 0.0) if helium else 1.7,
        wavelength=800e-9,
        wavelength_limits=(90e-9 if helium else 300e-9, 4e-6),
        trange=400e-15,
        fwhm=7.5e-15 if helium else 30e-15,
        energy=275e-6 if helium else 3e-6,
        modes=2 if helium else 1,
        envelope=name == "argon_envelope",
        loss=helium,
        # Envelope plasma is opt-in; this is the declared plasma configuration,
        # not the constructor's plasma-free default envelope.
        plasma=True,
        noise=None,
        backend="numpy_ref",
        integrator=RungeKutta(),
        rtol=1e-10,
        dz_init=STEP,
        dz_max=STEP,
        save_planes=[WINDOW],
        stats=StatsSpec(names=("energy", "peak_power", "electron_density", "density")),
    )
    case = Translation(simulation.translate(), simulation._run_request())
    assert case.request.replay is None and not case.request.observation.dense
    assert not case.problem.flags.has_noise
    assert case.problem.flags.has_plasma
    return simulation, case


@cache
def _run(name, method, rtol, single):
    simulation, original = _scientific_case(name)
    request = _method(original.request, method)
    request = replace(
        request,
        error_control=replace(
            request.error_control,
            controller=replace(request.error_control.controller, rtol=rtol),
        ),
    )
    problem = (
        replace(original.problem, precision=SINGLE) if single else original.problem
    )
    case = Translation(problem, request)
    if single:
        setup_id = os.environ.get("SOLPROP_TEST_JULIA_SETUP_ID")
        assert setup_id, "the enabled Julia lane must publish a managed setup identity"
        owner = JuliaBackend(
            problem,
            request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
            options=ExecutionOptions(
                node_chunk=8, backend_options={"setup_id": setup_id}
            ),
        )
    else:
        owner = select(
            "numpy_ref",
            problem,
            request,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
    before = problem.initial.field0.tobytes()
    try:
        with owner.prepare(problem, request).start(Start.entrance(problem)) as session:
            result = drive(
                Started(session), export=IDENTITY, work_limit=8, deadline=60.0
            )
            record = session.execution_record()
            work = {
                "attempts": result.status.attempts,
                "accepted": result.status.attempts - result.status.rejected,
                "rejected": result.status.rejected,
                "source_evaluations": cast("int", record["source_evaluations"]),
            }
            if single:
                assert record["precision"] == "single" and record["wide"] == "binary64"
            assert problem.initial.field0.tobytes() == before
            terminal = result.batches[-1].terminal
            assert terminal is not None
            values = {}
            if result.status.outcome is Outcome.COMPLETED:
                rows = _statistics(result)
                fraction = np.empty(0)
                if problem.flags.has_plasma:
                    density = np.asarray(rows["density"])
                    assert np.all(density > 0.0)
                    fraction = np.atleast_1d(rows["electron_density"] / density)
                values = {
                    "bands": _bands(case, simulation.grid, result.fields[-1], WINDOW),
                    "energy": np.atleast_1d(np.sum(rows["energy"])),
                    "peak": np.asarray(rows["peak_power"]).ravel(),
                    "fraction": fraction,
                }
                # Canonical binary64 positive reductions share the existing
                # operation-count allowance. The actual binary32 statistics use
                # SEP33's unchanged physical-energy envelope instead.
                budget = 1e-4 if single else _band_partition_budget(result.fields[-1])
                np.testing.assert_allclose(
                    np.sum(values["bands"]), values["energy"][0], rtol=budget, atol=0.0
                )
                assert all(np.all(np.isfinite(value)) for value in values.values())
                assert (
                    np.count_nonzero(result.fields[-1][..., ~problem.grids.active_mask])
                    == 0
                )
            return _Reading(
                result.fields[-1] if result.fields else None,
                values,
                work,
                terminal.outcome.value,
                "" if terminal.error is None else str(terminal.error),
                problem.flags.has_plasma,
            )
    finally:
        owner.close()


def _relative(actual, reference):
    error = np.abs(actual - reference)
    result = np.where(error == 0, 0.0, np.inf)
    # No denominator floor: a nonzero result against a true zero has infinite error.
    np.divide(error, np.abs(reference), out=result, where=reference != 0)
    return result


def _errors(actual, reference):
    assert actual.outcome == reference.outcome == Outcome.COMPLETED.value
    assert actual.field is not None and reference.field is not None
    norm = float(np.linalg.norm(reference.field.ravel()))
    assert norm > 0.0
    errors = {
        "field": np.atleast_1d(
            np.linalg.norm((actual.field - reference.field).ravel()) / norm
        )
    }
    for name, expected in reference.values.items():
        selected = (
            expected >= 1e-4 * np.max(expected)
            if name == "bands"
            else expected > 1e-6
            if name == "fraction"
            else np.ones(expected.shape, dtype=bool)
        )
        errors[name] = _relative(actual.values[name][selected], expected[selected])
    return errors


def _json_errors(errors):
    return {
        name: [float(item) if np.isfinite(item) else str(item) for item in values]
        for name, values in errors.items()
    }


@cache
def _reference(name):
    history = []
    for previous_rtol, reference_rtol in REFERENCE_PAIRS:
        previous = _run(name, "dopri5", previous_rtol, False)
        reference = _run(name, "dopri5", reference_rtol, False)
        assert previous.outcome == reference.outcome == Outcome.COMPLETED.value, (
            previous.error,
            reference.error,
            previous.work,
            reference.work,
        )
        errors = _errors(previous, reference)
        history.append(
            {
                "tolerances": [previous_rtol, reference_rtol],
                "errors": _json_errors(errors),
                "previous_work": previous.work,
                "reference_work": reference.work,
            }
        )
        if all(np.all(values <= 0.1 * LIMITS[key]) for key, values in errors.items()):
            return reference, history
    pytest.fail(f"binary64 reference did not settle: {json.dumps(history)}")


@pytest.mark.parametrize("name", ["helium", "argon_field", "argon_envelope"])
@pytest.mark.parametrize("rtol", [1e-6, 1e-5])
def test_binary32_adams_meets_twice_rk_error_and_fixed_envelopes(
    name, rtol, record_property
):
    reference, history = _reference(name)
    record_property("reference_refinements", json.dumps(history, sort_keys=True))
    record_property("julia_setup_id", os.environ["SOLPROP_TEST_JULIA_SETUP_ID"])
    rk = _run(name, "dopri5", rtol, True)
    adams = _run(name, "adams", rtol, True)
    work = {"rk": rk.work, "adams": adams.work}
    record_property("actual_work", json.dumps(work, sort_keys=True))
    record_property(
        "outcomes",
        json.dumps(
            {"rk": [rk.outcome, rk.error], "adams": [adams.outcome, adams.error]},
            sort_keys=True,
        ),
    )
    assert rk.outcome == adams.outcome == Outcome.COMPLETED.value, (
        rk.error,
        adams.error,
        work,
    )
    rk_errors, adams_errors = _errors(rk, reference), _errors(adams, reference)
    fraction = reference.values["fraction"]
    report = {
        "case": name,
        "window_m": WINDOW,
        "rtol": rtol,
        "reference_refinements": history,
        "actual_work": work,
        "rk_errors": _json_errors(rk_errors),
        "adams_errors": _json_errors(adams_errors),
        "reference_fraction": fraction.tolist(),
        "reference_values": {
            key: values.tolist() for key, values in reference.values.items()
        },
        "scored_band_indices": np.flatnonzero(
            reference.values["bands"] >= 1e-4 * np.max(reference.values["bands"])
        ).tolist(),
        "population_present": reference.population_present,
        "unscored_fraction": (fraction <= 1e-6).tolist(),
        "fraction_rule": "score reference >1e-6; no denominator floor",
    }
    record_property(
        "cpu32_adams_admission", json.dumps(report, sort_keys=True, allow_nan=False)
    )
    failures = []
    for metric, rk_error in rk_errors.items():
        adams_error = adams_errors[metric]
        for component, (r_error, a_error) in enumerate(
            zip(rk_error, adams_error, strict=True)
        ):
            if (
                r_error > LIMITS[metric]
                or a_error > LIMITS[metric]
                or a_error > 2 * r_error
            ):
                failures.append(
                    {
                        "metric": metric,
                        "component": component,
                        "rk_error": float(r_error),
                        "adams_error": float(a_error),
                        "limit": LIMITS[metric],
                        "twice_rk": float(2 * r_error),
                    }
                )
    assert rk.work["accepted"] >= 8 and adams.work["accepted"] >= 8
    assert not failures, json.dumps(failures, indent=2)
