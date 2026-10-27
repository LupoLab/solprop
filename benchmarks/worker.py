# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time complete runs of one configuration in a fresh process (SEP 38).

The controller sets the thread pools in the environment and passes a JSON request:
one configuration, the number of timed repeats and, optionally, where to save the
final field. The worker builds and translates the case and constructs its backend,
runs an untimed warm-up over the first part of the guide, then times complete runs
through the driver's own loop, each restarted from the translated launch. It prints
one JSON event per line; the last is the receipt.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import logging
import sys
import time
import traceback
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.records import (
    Record,
    SuiteError,
    canonical,
    file_hash,
    metadata_value,
    read_record,
)

__all__ = [
    "HOST_SPACES",
    "Diagnostics",
    "emit",
    "integrator_label",
    "main",
    "measure",
    "runtime",
    "source_calls",
    "timed_run",
]

if TYPE_CHECKING:
    import numpy as np

    from solprop.backends.protocol import Backend
    from solprop.driver import Simulation
    from solprop.output.schema import FieldRetention
    from solprop.translate.problem import KernelProblem


def emit(kind: str, **values: object) -> None:
    """Flush a protocol event; the parent reads it after the worker exits."""
    print(canonical({"event": kind, **values}), flush=True)


HOST_SPACES = ("OpenMP", "Threads", "Serial")
"""Kokkos execution spaces of a native CPU build; any other is an accelerator's."""


def runtime(backend: str, device: bool = False) -> Record:
    """Import in the required order and identify the loaded packages and pools.

    A `device` run needs a native accelerator build with a visible device; any
    other run needs a CPU build.
    """
    info: Record = {}
    if device and backend != "native":
        raise SuiteError("only the native backend runs on a device")
    if backend == "native":
        # solprop_native must be imported before pyFFTW.
        import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

        build = dataclasses.asdict(solprop_native.build_info())
        if build.get("coverage_instrumented", False):
            raise SuiteError("an instrumented native build cannot be timed")
        on_host = build.get("execution_space") in HOST_SPACES
        if device and (on_host or not build.get("device_count")):
            raise SuiteError("a device run needs a native GPU build and a visible GPU")
        if not device and not on_host:
            raise SuiteError("a native GPU build times device runs only (--device)")
        extension = Path(importlib.import_module("solprop_native._core").__file__ or "")
        info.update(build_info=build, extension_hash=file_hash(extension))
    from threadpoolctl import threadpool_info  # noqa: PLC0415

    import solprop  # noqa: PLC0415

    packages = {}
    for name in ("solprop", "numpy", "scipy", "optmat", "numba", "pyfftw"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info.update(
        packages=packages,
        solprop_path=str(Path(solprop.__file__).resolve()),
        libraries=threadpool_info(),
        python=sys.version,
        python_executable=sys.executable,
    )
    return info


class Diagnostics(logging.Handler):
    """Count emitted warnings and logs by phase without suppressing them."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.phase = "import"
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
        """Return every distinct occurrence with its count."""
        return [
            {"phase": phase, "category": category, "message": message, "count": count}
            for (phase, category, message), count in self.rows.items()
        ]


def source_calls(engine: Backend) -> int | None:
    """Read the backend's source-evaluation ledger, or ``None`` if it has none."""
    metadata = engine.metadata()
    value = metadata.get("source_evaluations")
    if isinstance(value, int):
        return value
    ledger = metadata.get("response_calls")
    if isinstance(ledger, dict):
        return sum(int(ledger[key]) for key in ("startup", "predictor", "corrected"))
    return None


def integrator_label(integrator: object) -> str:
    """Name an integrator with the option that distinguishes its variants.

    A translated problem holds ``None`` for the interaction-picture DOPRI5 march.
    """
    if integrator is None:
        return "RungeKutta"
    name = type(integrator).__name__
    for option in ("formulation", "scheme"):
        value = getattr(integrator, option, None)
        if isinstance(value, str):
            return f"{name}({value})"
    return name


def timed_run(
    simulation: Simulation,
    problem: KernelProblem,
    engine: Backend,
    retention: FieldRetention,
) -> tuple[Record, np.ndarray]:
    """Time one complete run from the launch; return its counts and final field."""
    import numpy as np  # noqa: PLC0415

    from solprop.driver.run import run_problem  # noqa: PLC0415
    from solprop.output import MemoryOutput  # noqa: PLC0415

    engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
    output = MemoryOutput()
    before = source_calls(engine)
    started = time.perf_counter_ns()
    run_problem(
        problem,
        engine,
        output,
        metadata=simulation.metadata(),
        stats_spec=problem.stats_spec,
        progress="none",
        resume_history=simulation.settings.resume_history,
        fields=retention.fields,
        field_planes=retention.field_planes,
    )
    seconds = (time.perf_counter_ns() - started) * 1e-9
    after = source_calls(engine)
    accepted = np.asarray(output["step_history/accepted"], dtype=bool)
    record = {
        "seconds": seconds,
        "accepted": int(accepted.sum()),
        "rejected": int((~accepted).sum()),
        "source_evaluations": None
        if before is None or after is None
        else after - before,
        "z": float(output["z"][-1]),
    }
    return record, output.plane_field(output.n_saved - 1, "fft")


def _device_free(configuration: Record) -> int | None:
    """Read the accelerator's free bytes on a device run, else ``None``."""
    if not configuration.get("device"):
        return None
    import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    return solprop_native.available_device_memory()


def measure(configuration: Record, repeats: int, fields: Path | None) -> Record:
    """Build once, warm up once, then time ``repeats`` complete runs.

    Each run restarts the backend from the translated launch and goes through
    :func:`solprop.driver.run.run_problem`, the loop of ``Simulation.run``, so it
    includes the plane observations and statistics. Construction, translation and
    backend construction are timed separately.
    """
    import numpy as np  # noqa: PLC0415

    from benchmarks.builders import build  # noqa: PLC0415
    from solprop.backends import select  # noqa: PLC0415
    from solprop.driver.executors.pool import peak_rss  # noqa: PLC0415
    from solprop.driver.run import field_retention  # noqa: PLC0415
    from solprop.output.schema import serialise  # noqa: PLC0415

    free_before = _device_free(configuration)
    started = time.perf_counter()
    simulation = build(configuration)
    problem = simulation.translate()
    construction = time.perf_counter() - started
    started = time.perf_counter()
    engine = select(
        configuration["backend"],
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    backend_seconds = time.perf_counter() - started
    retention = field_retention(
        problem, simulation.settings.fields, simulation.settings.field_planes
    )
    controller = problem.stepper.controller
    try:
        started = time.perf_counter()
        engine.set_initial(problem.initial.field0, None, z=problem.initial.z)
        if configuration["warmup"]:
            engine.propagate_to(configuration["warmup"] * configuration["stop"])
        # Observe once, so that the observation path's first use is warm too.
        engine.observe(field=True)
        engine.stats(problem.stats_spec)
        warmup = time.perf_counter() - started
        free_warm = _device_free(configuration)
        repetitions = []
        first = final = None
        difference = 0.0
        for _ in range(repeats):
            repetition, final = timed_run(simulation, problem, engine, retention)
            repetitions.append(repetition)
            # Every repeat restarts from the same launch and seed, so its final
            # field may differ from the first repeat's by rounding only.
            if first is None:
                first = final
            else:
                scale = float(np.linalg.norm(first)) or 1.0
                difference = max(
                    difference, float(np.linalg.norm(final - first)) / scale
                )
        assert first is not None
        final = first if final is None else final
        if fields is not None:
            fields.parent.mkdir(parents=True, exist_ok=True)
            np.save(fields, final, allow_pickle=False)
        return {
            "construction_seconds": construction,
            "backend_seconds": backend_seconds,
            "warmup_seconds": warmup,
            "repetitions": repetitions,
            "peak_rss_bytes": peak_rss(),
            # The run's workspace stays allocated, so the drop in free memory from
            # before construction to after the warm-up is what the run holds.
            "device_bytes": None
            if free_before is None or free_warm is None
            else free_before - free_warm,
            "finite": bool(np.all(np.isfinite(final))),
            "repeat_difference": difference,
            "integrator": integrator_label(problem.stepper.integrator),
            "rtol": controller.rtol,
            "atol": controller.atol,
            "noise": simulation.noise_resolution.resolved
            if simulation.noise_resolution is not None
            else None,
            "shape": list(problem.shape),
            "resolution": metadata_value(
                {
                    "integrator": serialise(problem.stepper.integrator),
                    "controller": serialise(controller),
                }
            ),
        }
    finally:
        engine.close()


def main() -> int:
    """Execute the JSON request passed by the controller."""
    diagnostics = Diagnostics()
    try:
        request = read_record(Path(sys.argv[1]))
        configuration = request["configuration"]
        with diagnostics.capture():
            emit(
                "provenance",
                provenance=runtime(
                    configuration["backend"], bool(configuration.get("device"))
                ),
            )
            diagnostics.phase = "measurement"
            fields = request.get("fields")
            result = measure(
                configuration,
                int(request["repeats"]),
                None if fields is None else Path(fields),
            )
        emit("result", status="ok", diagnostics=diagnostics.records(), **result)
    except Exception as error:  # worker boundary: the controller records the failure
        traceback.print_exc(file=sys.stderr)
        emit(
            "result",
            status="failed",
            error=f"{type(error).__name__}: {error}",
            diagnostics=diagnostics.records(),
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
