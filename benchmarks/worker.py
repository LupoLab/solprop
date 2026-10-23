# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute one configuration in a fresh process with receipt-delimited measurement."""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import logging
import re
import statistics
import sys
import time
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.records import (
    Record,
    SuiteError,
    canonical,
    file_hash,
    identity,
    metadata_value,
    read_record,
)
from benchmarks.registry import CASES
from benchmarks.telemetry import process_binding

__all__ = [
    "Diagnostics",
    "ResetWitness",
    "accuracy",
    "emit",
    "execute",
    "execute_single",
    "main",
    "measure",
    "observed_runtime",
    "prepare",
    "reset_columns",
    "reset_differences",
    "reset_reference",
    "resolved",
    "retained_fields",
    "runtime",
    "save_plane",
    "source_calls",
    "timer_resolution",
    "validation_artifacts",
]

if TYPE_CHECKING:
    import numpy as np

    from solprop.backends.protocol import Backend
    from solprop.driver import Simulation
    from solprop.output import MemoryOutput
    from solprop.translate.problem import KernelProblem


@dataclasses.dataclass(frozen=True)
class ResetWitness:
    """Keep fresh fields and numerical history only in a qualification worker."""

    fields: np.ndarray
    columns: dict[str, np.ndarray]


def retained_fields(output: MemoryOutput) -> np.ndarray:
    """Read the fields actually saved without fabricating statistics-only planes."""
    import numpy as np  # noqa: PLC0415 - worker-only array access

    if output.retention.fields == "all":
        return output.fields(convention="fft")
    indices = output["field_planes"]
    if len(indices) == 0:
        raise SuiteError("validation needs at least one retained field")
    return np.stack([output.plane_field(int(index), "fft") for index in indices])


def reset_columns(output: MemoryOutput) -> dict[str, np.ndarray]:
    """Copy observed decisions, controller history, statistics and noise state."""
    names = [
        "step_history/z",
        "step_history/h",
        "step_history/norm",
        "step_history/accepted",
        *(f"stats/{name}" for name in output.stat_names),
    ]
    if output.step_stat_names:
        names.extend(
            [
                "step_stats/z",
                "step_stats/step",
                *(f"step_stats/{name}" for name in output.step_stat_names),
            ]
        )
    columns = {name: output[name].copy() for name in names}
    if output.noise_phase is not None:
        columns["noise_phase"] = output.noise_phase.copy()
    return columns


def reset_differences(witness: ResetWitness, output: MemoryOutput) -> Record:
    """Report numerical history differences without imposing bitwise agreement."""
    import numpy as np  # noqa: PLC0415 - qualification only

    found = reset_columns(output)
    result: Record = {"same_columns": found.keys() == witness.columns.keys()}
    for name, expected in witness.columns.items():
        actual = found.get(name)
        if actual is None or actual.shape != expected.shape:
            result[name] = {"same_shape": False}
        elif expected.dtype == np.bool_:
            result[name] = {"same_decisions": bool(np.array_equal(actual, expected))}
        else:
            # Empty modal columns have no centroid; a CW has no FWHM. These
            # statistics deliberately use NaN (stats.py), unlike state/history.
            # Compare their defined entries and retain the undefined masks
            # explicitly, without allowing a NaN into the JSON metric itself.
            undefined = np.isnan(expected)
            found_undefined = np.isnan(actual)
            if (
                np.isinf(expected).any()
                or np.isinf(actual).any()
                or (
                    not (
                        name.startswith(("stats/", "step_stats/"))
                        and name not in ("step_stats/z", "step_stats/step")
                    )
                    and (undefined.any() or found_undefined.any())
                )
            ):
                raise SuiteError(f"non-finite reset history or statistic: {name}")
            defined = ~(undefined | found_undefined)
            difference = (
                float(np.linalg.norm(actual[defined] - expected[defined]))
                if defined.any()
                else None
            )
            norm = float(np.linalg.norm(expected[defined]))
            result[name] = {
                "same_shape": True,
                "same_undefined": bool(np.array_equal(undefined, found_undefined)),
                "reference_undefined_count": int(np.count_nonzero(undefined)),
                "actual_undefined_count": int(np.count_nonzero(found_undefined)),
                "compared_finite_count": int(np.count_nonzero(defined)),
                "absolute_l2": difference,
                "relative_l2": difference / norm
                if norm and difference is not None
                else None,
                "relative_unavailable": "no jointly defined statistic entries"
                if difference is None
                else "zero reference norm"
                if norm == 0
                else None,
            }
    return result


def emit(kind: str, **values: object) -> None:
    """Flush a protocol event; the parent timestamps its receipt."""
    print(canonical({"event": kind, **values}), flush=True)


def runtime(backend: str, *, inventory: bool = False, coverage: bool = False) -> Record:
    """Import in the required order and identify the actual loaded packages."""
    info: Record = {}
    if backend == "native":
        # solprop_native must be imported before pyFFTW.
        import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        build = dataclasses.asdict(solprop_native.build_info())
        if build.get("coverage_instrumented", False) and not coverage:
            raise SuiteError("instrumented native builds cannot collect timing samples")
        if coverage and not build.get("coverage_instrumented", False):
            raise SuiteError("native coverage requires an instrumented native build")
        if build.get("execution_space") not in ("OpenMP", "Threads", "Serial"):
            raise SuiteError("this suite is qualified only for native CPU builds")
        extension = Path(importlib.import_module("solprop_native._core").__file__ or "")
        info.update(
            build_info=build,
            extension_hash=file_hash(extension),
            extension_path=str(extension),
            wheel_hash=None,
            wheel_hash_reason="original wheel unavailable from installed distribution",
        )
    from threadpoolctl import threadpool_info  # noqa: PLC0415

    import solprop  # noqa: PLC0415
    from benchmarks.system import git_receipt  # noqa: PLC0415

    packages = {}
    for name in (
        "solprop",
        "numpy",
        "scipy",
        "optmat",
        "numba",
        "pyfftw",
        "solprop-native",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source = Path(solprop.__file__).resolve().parents[2]
    info.update(
        packages=packages,
        solprop_path=str(Path(solprop.__file__).resolve()),
        source=git_receipt(source) if inventory else None,
        source_reason=None if inventory else "recorded once in collection inventory",
        libraries=threadpool_info(),
        filesystem_cache="unknown",
        numba_cache="uncontrolled-existing",
        python=sys.version,
        python_executable=sys.executable,
    )
    return info


def observed_runtime(provenance: Record) -> Record:
    """Observe lazily loaded library pools after timing and peak RSS collection."""
    from threadpoolctl import threadpool_info  # noqa: PLC0415

    return {**provenance, "libraries": threadpool_info()}


class Diagnostics(logging.Handler):
    """Aggregate emitted warnings and logs without suppressing their occurrences."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.phase = "import"
        self.case: str | None = None
        self.rows: dict[tuple[str, str, str], int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        key = (self.phase, f"logging.{record.levelname}", record.getMessage())
        self.rows[key] = self.rows.get(key, 0) + 1

    @contextmanager
    def capture(self) -> Iterator[None]:
        logger = logging.getLogger("solprop")
        logger.addHandler(self)
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            previous = warnings.showwarning

            def record(
                message: Warning | str,
                category: type[Warning],
                *_position: object,
                **_context: object,
            ) -> None:
                key = (self.phase, category.__name__, str(message))
                self.rows[key] = self.rows.get(key, 0) + 1

            warnings.showwarning = record
            try:
                yield
            finally:
                warnings.showwarning = previous
                logger.removeHandler(self)

    def records(self) -> list[Record]:
        """Return all occurrences; qualification reviews expected patterns."""
        return [
            {
                "phase": phase,
                "category": category,
                "message": message,
                "count": count,
                "expected": self.case is not None
                and any(
                    rule.phase == phase
                    and rule.category == category
                    and re.fullmatch(rule.pattern, message) is not None
                    for rule in CASES[self.case].warnings
                ),
            }
            for (phase, category, message), count in self.rows.items()
        ]


def source_calls(engine: Backend) -> int:
    """Read physical/logical evaluation ledgers without step-count inference."""
    metadata = engine.metadata()
    value = metadata.get("source_evaluations")
    if isinstance(value, int):
        return value
    ledger = metadata.get("response_calls")
    if isinstance(ledger, dict):
        return sum(int(ledger[key]) for key in ("startup", "predictor", "corrected"))
    raise SuiteError(
        f"{engine.name} has no source evaluation counter; rebuild the backend"
    )


def prepare(
    simulation: Simulation,
    problem: KernelProblem,
    engine: Backend,
    configuration: Record,
) -> MemoryOutput:
    """Restore launch and method/controller history through the declared lead-in."""
    from solprop.output import MemoryOutput  # noqa: PLC0415
    from solprop.output.schema import FieldRetention  # noqa: PLC0415

    engine.set_initial(problem.initial.field0, None)
    if configuration["start"]:
        engine.propagate_to(float(configuration["start"]))
        engine.observe(field=False)
    output = MemoryOutput()
    if configuration["fields"] == "planes":
        output.retain(
            FieldRetention("planes", (int(configuration["observations"]) - 1,)),
            problem.shape[1:],
        )
    output.open(simulation.metadata())
    output.reserve(int(configuration["observations"]))
    return output


def save_plane(
    engine: Backend, problem: KernelProblem, output: MemoryOutput, *, keep: bool = True
) -> None:
    """Use the ordinary amplitude and saved-plane conversions."""
    from solprop.driver.run import (  # noqa: PLC0415
        _power_observation,
        plane_from_observation,
    )

    observed = _power_observation(problem, engine.observe(field=keep))
    output.save(plane_from_observation(observed, engine.stats(problem.stats_spec)))


def measure(
    engine: Backend, problem: KernelProblem, configuration: Record, output: MemoryOutput
) -> Record:
    """Time the declared work, preserving every repetition's actual counters."""
    import numpy as np  # noqa: PLC0415

    adaptive = configuration["track"].startswith("adaptive-")
    planes = np.linspace(
        configuration["start"],
        configuration["stop"],
        int(configuration["observations"]) + 1,
    )[1:]
    before = source_calls(engine)
    steps = rejected = 0
    started = time.perf_counter_ns()
    for index, plane in enumerate(planes):
        summary = engine.propagate_to(float(plane))
        steps += summary.n_steps
        rejected += summary.n_rejected
        if summary.z != float(plane):
            raise SuiteError("backend did not reach the declared endpoint")
        if adaptive:
            save_plane(
                engine,
                problem,
                output,
                keep=configuration["fields"] != "planes" or index == len(planes) - 1,
            )
    seconds = (time.perf_counter_ns() - started) * 1e-9
    return {
        "seconds": seconds,
        "accepted": steps - rejected,
        "rejected": rejected,
        "source_evaluations": source_calls(engine) - before,
        "endpoint": float(planes[-1]),
        "observation_seconds": None,
    }


def resolved(
    simulation: Simulation, problem: KernelProblem, configuration: Record
) -> Record:
    """Record actual method defaults and material/initial-state identities."""

    from solprop.output.schema import serialise  # noqa: PLC0415

    return metadata_value(
        {
            "configuration": configuration,
            "simulation": serialise(simulation.metadata().configuration),
            "integrator": serialise(problem.stepper.integrator),
            "controller": serialise(problem.stepper.controller),
            "shape": list(problem.shape),
            "n_t": problem.grids.n_t,
            "n_to": problem.grids.n_to,
            "statistics": serialise(problem.stats_spec),
            "measurement_planes": configuration["observations"],
            "fields": configuration["fields"],
        }
    )


def accuracy(fields: np.ndarray, configuration: Record, request: Record) -> Record:
    """Check finiteness and a frozen independent reference after the RSS receipt."""
    import numpy as np  # noqa: PLC0415

    if not np.all(np.isfinite(fields)):
        return {"status": "invalid", "reason": "non-finite output"}
    previous = request.get("prior_validation")
    if previous is not None:
        if list(fields.shape) != previous["field_shape"]:
            return {"status": "invalid", "reason": "validated output shape changed"}
        return dict(previous)
    reference = request.get("reference")
    if not reference:
        return {"status": "not-qualified", "reason": "no frozen qualified reference"}
    from benchmarks.references import check  # noqa: PLC0415

    try:
        return {
            **check(fields, configuration, Path(reference)),
            "field_shape": list(fields.shape),
            "validation": "current sample",
        }
    except (SuiteError, OSError, ValueError) as error:
        return {"status": "invalid", "reason": str(error)}


def timer_resolution() -> float:
    """Measure clock increments and call overhead without a minimum time loop."""
    ticks = [time.perf_counter_ns() for _ in range(1001)]
    differences = [right - left for left, right in pairwise(ticks) if right > left]
    return max(
        time.get_clock_info("perf_counter").resolution,
        statistics.median(differences) * 1e-9,
    )


def validation_artifacts(
    fields: np.ndarray,
    fresh_fields: ResetWitness | None,
    output: MemoryOutput,
    problem: KernelProblem,
    request: Record,
    measurements: tuple[Record, list[Record]],
) -> Record:
    """Save qualification arrays and reset evidence after the measurement receipt."""
    import numpy as np  # noqa: PLC0415

    from benchmarks.references import field_errors  # noqa: PLC0415

    first, repetitions = measurements
    if request.get("save_fields"):
        np.save(request["save_fields"], fields, allow_pickle=False)
        np.save(
            f"{request['save_fields']}.band.npy",
            problem.grids.inband_mask,
            allow_pickle=False,
        )
        np.save(
            f"{request['save_fields']}.omega.npy",
            problem.grids.omega,
            allow_pickle=False,
        )
    reset = None
    if fresh_fields is not None:
        keys = ("accepted", "rejected", "source_evaluations")
        reset = {
            "plane_errors": field_errors(fields, fresh_fields.fields),
            "history_statistics_noise": reset_differences(fresh_fields, output),
            "fresh_counts": {key: first[key] for key in keys},
            "reset_counts": [{key: item[key] for key in keys} for item in repetitions],
        }
    edges = [
        0.0,
        *(
            float(z + h)
            for z, h, accepted in zip(
                output["step_history/z"],
                output["step_history/h"],
                output["step_history/accepted"],
                strict=True,
            )
            if accepted
        ),
    ]
    return {"reset_evidence": reset, "pilot_edges": edges[:11]}


def reset_reference(
    engine: Backend, problem: KernelProblem, output: MemoryOutput, request: Record
) -> ResetWitness | None:
    """Capture the fresh construction result only during separate qualification."""
    if not request.get("verify_reset"):
        return None
    if not request["configuration"]["track"].startswith("adaptive-"):
        save_plane(engine, problem, output)
    return ResetWitness(retained_fields(output).copy(), reset_columns(output))


def execute(request: Record, diagnostics: Diagnostics) -> Record:
    """Run one complete fresh-process sample and return its validated receipt."""
    configuration = request["configuration"]
    diagnostics.case = configuration["case"]
    started = time.perf_counter()
    provenance = runtime(
        configuration["backend"],
        inventory=request.get("inventory", False),
        coverage=request.get("coverage", False),
    )
    import_seconds = time.perf_counter() - started
    if request.get("inventory"):
        return {"status": "ok", "provenance": provenance}
    if request.get("coverage"):
        from benchmarks.coverage_worker import execute_coverage  # noqa: PLC0415

        return execute_coverage(request, diagnostics, provenance)
    if configuration["case"] == "batch":
        from benchmarks.batching import execute_batch  # noqa: PLC0415

        return execute_batch(request, diagnostics, provenance, import_seconds)
    return execute_single(request, diagnostics, provenance, import_seconds)


def execute_single(
    request: Record, diagnostics: Diagnostics, provenance: Record, import_seconds: float
) -> Record:
    """Measure one singleton backend after the common runtime inventory."""
    configuration = request["configuration"]
    from benchmarks.builders import build  # noqa: PLC0415
    from solprop.backends import select  # noqa: PLC0415
    from solprop.driver.executors.pool import peak_rss  # noqa: PLC0415
    from solprop.output.schema import serialise  # noqa: PLC0415

    diagnostics.phase = "construction"
    started = time.perf_counter()
    simulation = build(configuration)
    problem = simulation.translate()
    resolution = resolved(simulation, problem, configuration)
    construction_seconds = time.perf_counter() - started
    started = time.perf_counter()
    engine = select(
        configuration["backend"],
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    backend_seconds = time.perf_counter() - started
    repetitions: list[Record] = []
    try:
        diagnostics.phase = "first-window"
        output = prepare(simulation, problem, engine, configuration)
        first = measure(engine, problem, configuration, output)
        emit("first-window", seconds=first["seconds"])
        fresh_fields = reset_reference(engine, problem, output, request)
        output.close()
        for _ in range(request.get("repetitions", 1)):
            output.close()
            del output
            diagnostics.phase = "reset"
            output = prepare(simulation, problem, engine, configuration)
            diagnostics.phase = "measurement"
            repetitions.append(measure(engine, problem, configuration, output))
        peak = peak_rss()
        emit("measurement", peak_rss_bytes=peak)
        diagnostics.phase = "validation"
        if not configuration["track"].startswith("adaptive-"):
            # The fixed track's observation is separate, and cannot increase the
            # recorded measurement peak or the propagation timer.
            started = time.perf_counter()
            save_plane(engine, problem, output)
            repetitions[-1]["observation_seconds"] = time.perf_counter() - started
            output.close()
        fields = retained_fields(output)
        check = accuracy(fields, configuration, request)
        overridden = check["status"] == "invalid" and request.get(
            "allow_accuracy_failure", False
        )
        artifacts = validation_artifacts(
            fields, fresh_fields, output, problem, request, (first, repetitions)
        )
        output.close()
        return {
            "status": "invalid"
            if check["status"] == "invalid" and not overridden
            else "ok",
            "accuracy_overridden": overridden,
            "seconds": sum(item["seconds"] for item in repetitions) / len(repetitions),
            "repetitions": repetitions,
            "peak_rss_bytes": peak,
            "accuracy": check,
            **artifacts,
            "timer_resolution": timer_resolution(),
            "resolution": resolution,
            "resolution_hash": identity(resolution),
            "provenance": observed_runtime(provenance),
            "metadata": metadata_value(serialise(engine.metadata())),
            "ledger": engine.memory_ledger(),
            "first_window_seconds": first["seconds"],
            "import_seconds": import_seconds,
            "construction_seconds": construction_seconds,
            "backend_seconds": backend_seconds,
            "thread_diagnostics": process_binding()
            if request.get("thread_diagnostics")
            else None,
        }
    finally:
        engine.close()


def main() -> int:
    """Execute the JSON worker request passed by the controller."""
    diagnostics = Diagnostics()
    try:
        request = read_record(Path(sys.argv[1]))
        with diagnostics.capture():
            result = execute(request, diagnostics)
        result["diagnostics"] = diagnostics.records()
        emit("result", result=result)
        return 0 if result.get("status", "ok") == "ok" else 1
    except Exception as error:  # worker boundary: preserve a failed-attempt receipt
        traceback.print_exc(file=sys.stderr)
        emit(
            "result",
            result={
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "diagnostics": diagnostics.records(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
