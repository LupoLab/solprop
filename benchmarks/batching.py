# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure the real fixed-schedule batch route with independent member evidence."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import cast

import numpy as np

from benchmarks.builders import fixed_schedule
from benchmarks.exercises import fiber
from benchmarks.records import Record, SuiteError, identity, metadata_value
from benchmarks.references import field_errors
from benchmarks.registry import CASES
from benchmarks.worker import (
    Diagnostics,
    ResetWitness,
    accuracy,
    emit,
    observed_runtime,
    reset_columns,
    reset_differences,
    resolved,
    source_calls,
    timer_resolution,
)
from solprop.backends import select, select_batch
from solprop.backends.batch import LockstepRunner
from solprop.backends.driver import StepDriver
from solprop.backends.protocol import Backend, BatchBackend
from solprop.driver import Simulation
from solprop.driver.executors.pool import peak_rss
from solprop.driver.run import _power_observation, plane_from_observation
from solprop.output import MemoryOutput
from solprop.output.schema import serialise
from solprop.stepper import PIController, RungeKutta
from solprop.translate.problem import KernelProblem

__all__ = ["BatchCase", "construct", "execute_batch"]


@dataclass(frozen=True)
class BatchCase:
    """Retain independently translated members and their shared batch controller."""

    simulations: tuple[Simulation, ...]
    problems: tuple[KernelProblem, ...]
    members: tuple[Backend, ...]
    engine: BatchBackend

    def reset(self) -> None:
        """Reset members and initialise the base march without recreating buffers.

        The batch seam accepts independently loaded member states. Its common
        controller can be reinitialised with those same owners after their ordinary
        ``set_initial`` calls. Calling only the base initializer preserves Numba's
        fused buffers and the native backend's resident descriptors; backend-specific
        constructors, FFT planning and translation are not repeated.
        """
        for member, problem in zip(self.members, self.problems, strict=True):
            member.set_initial(problem.initial.field0, None)
        if not isinstance(self.engine, LockstepRunner):
            raise SuiteError("batch reset requires the known CPU lockstep controller")
        LockstepRunner.__init__(
            self.engine, [cast(StepDriver[object], member) for member in self.members]
        )

    def observe(self) -> tuple[MemoryOutput, ...]:
        """Convert every member's own field, statistics and history separately."""
        outputs = []
        for simulation, problem, member, observation in zip(
            self.simulations,
            self.problems,
            self.members,
            self.engine.observe_members(),
            strict=True,
        ):
            output = MemoryOutput()
            output.open(simulation.metadata())
            output.save(
                plane_from_observation(
                    _power_observation(problem, observation),
                    member.stats(problem.stats_spec),
                )
            )
            output.close()
            outputs.append(output)
        return tuple(outputs)

    def measure(self, stop: float) -> Record:
        """Time propagation only and retain actual source counts for every member."""
        before = [source_calls(member) for member in self.members]
        started = time.perf_counter_ns()
        summary = self.engine.propagate_to(stop)
        seconds = (time.perf_counter_ns() - started) * 1e-9
        if summary.z != stop:
            raise SuiteError("batch did not reach the declared endpoint")
        calls = [
            source_calls(member) - initial
            for member, initial in zip(self.members, before, strict=True)
        ]
        return {
            "seconds": seconds,
            "accepted": summary.n_steps - summary.n_rejected,
            "rejected": summary.n_rejected,
            "source_evaluations": sum(calls),
            "member_source_evaluations": calls,
            "endpoint": stop,
            "observation_seconds": None,
        }


def construct(configuration: Record) -> BatchCase:
    """Translate two distinct GNLSE members and select the actual batch backend."""
    simulations, problems, members = [], [], []
    try:
        for index in range(2):
            simulation = fiber(index, collar=False)
            simulation.settings = replace(
                simulation.settings,
                controller=PIController(
                    rtol=configuration["rtol"], atol=configuration["atol"]
                ),
                integrator=RungeKutta(),
                schedule=fixed_schedule(CASES["batch"], configuration),
                save_planes=[float(configuration["stop"])],
                backend=configuration["backend"],
                fields="none",
            )
            problem = simulation.translate()
            member = select(
                configuration["backend"],
                problem,
                grid=simulation.grid,
                modes=simulation.modes,
                profiles=simulation.profiles,
            )
            members.append(member)
            member.set_initial(problem.initial.field0, None)
            simulations.append(simulation)
            problems.append(problem)
        engine = select_batch(members)
        return BatchCase(tuple(simulations), tuple(problems), tuple(members), engine)
    except BaseException:
        for member in members:
            member.close()
        raise


def execute_batch(
    request: Record, diagnostics: Diagnostics, provenance: Record, import_seconds: float
) -> Record:
    """Use the ordinary receipt protocol for one fixed two-member batch sample."""
    configuration = request["configuration"]
    diagnostics.phase = "construction"
    started = time.perf_counter()
    case = construct(configuration)
    construction_seconds = time.perf_counter() - started
    try:
        diagnostics.phase = "first-window"
        first = case.measure(float(configuration["stop"]))
        emit("first-window", seconds=first["seconds"])
        fresh = case.observe() if request.get("verify_reset") else None
        repetitions = []
        for _ in range(request.get("repetitions", 1)):
            diagnostics.phase = "reset"
            case.reset()
            diagnostics.phase = "measurement"
            repetitions.append(case.measure(float(configuration["stop"])))
        peak = peak_rss()
        emit("measurement", peak_rss_bytes=peak)
        diagnostics.phase = "validation"
        started = time.perf_counter()
        outputs = case.observe()
        repetitions[-1]["observation_seconds"] = time.perf_counter() - started
        # One endpoint per member; the leading axis scores members separately,
        # so a bright member cannot hide an inaccurate weaker member.
        fields = np.concatenate([output.fields(convention="fft") for output in outputs])
        check = accuracy(fields, configuration, request)
        overridden = check["status"] == "invalid" and request.get(
            "allow_accuracy_failure", False
        )
        reset = None
        if fresh is not None:
            truth = np.concatenate(
                [output.fields(convention="fft") for output in fresh]
            )
            reset = {
                "plane_errors": field_errors(fields, truth),
                "history_statistics_noise": {
                    str(index): reset_differences(
                        ResetWitness(old.fields(convention="fft"), reset_columns(old)),
                        new,
                    )
                    for index, (old, new) in enumerate(zip(fresh, outputs, strict=True))
                },
                "fresh_counts": first,
                "reset_counts": repetitions,
            }
        if request.get("save_fields"):
            np.save(request["save_fields"], fields, allow_pickle=False)
            np.save(
                f"{request['save_fields']}.band.npy",
                case.problems[0].grids.inband_mask,
                allow_pickle=False,
            )
            np.save(
                f"{request['save_fields']}.omega.npy",
                case.problems[0].grids.omega,
                allow_pickle=False,
            )
        resolution = {
            "configuration": configuration,
            "field_axis": "member, mode, polarization, frequency",
            "members": [
                resolved(simulation, problem, configuration)
                for simulation, problem in zip(
                    case.simulations, case.problems, strict=True
                )
            ],
        }
        return {
            "status": "invalid"
            if check["status"] == "invalid" and not overridden
            else "ok",
            "accuracy_overridden": overridden,
            "accuracy": check,
            "seconds": sum(item["seconds"] for item in repetitions) / len(repetitions),
            "repetitions": repetitions,
            "peak_rss_bytes": peak,
            "reset_evidence": reset,
            "pilot_edges": [],
            "timer_resolution": timer_resolution(),
            "resolution": resolution,
            "resolution_hash": identity(resolution),
            "provenance": observed_runtime(provenance),
            "metadata": metadata_value(
                serialise(
                    {
                        **case.engine.metadata(),
                        "members": case.engine.member_metadata(),
                        "source_evaluations": sum(
                            source_calls(member) for member in case.members
                        ),
                    }
                )
            ),
            "ledger": case.engine.memory_ledger(),
            "first_window_seconds": first["seconds"],
            "import_seconds": import_seconds,
            "construction_seconds": construction_seconds,
            "backend_seconds": None,
            "backend_seconds_reason": "included in member construction",
        }
    finally:
        case.engine.close()
