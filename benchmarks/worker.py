# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time complete runs of one configuration in a fresh process (SEP 38).

The controller sets the thread pools in the environment and passes a JSON request:
one configuration, the number of timed repeats and, optionally, where to save the
final field and whether to count. The worker builds and translates the case and
constructs its backend, runs an untimed warm-up over the first part of the guide, then
times complete runs through the driver's own loop, each restarted from the translated
launch with the boundary monitor ``Simulation.run`` attaches, or the case's workload of
repeated inputs, a scan or a group (``benchmarks.registry.Case``). Asked to count, it
then makes one more repetition with the native adapter's structural counters on
(SEP 47 section 7.3), reported apart so that counting costs no timed repetition
anything. It prints one JSON event per line; the last is the receipt.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import logging
import shutil
import sys
import tempfile
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
    "SNAPSHOT_CAPACITY",
    "Diagnostics",
    "count",
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
    from solprop.monitor import BoundaryMonitor
    from solprop.output.schema import FieldRetention, Output
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
        # solprop_native must be imported before any other OpenMP runtime.
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
    for name in ("solprop", "numpy", "scipy", "optmat"):
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


SNAPSHOT_CAPACITY = 100_000
"""Snapshot generations a repetition's store holds: every one a case asks for."""


class _Loaded:
    """Note when a run's session finishes loading its backend, and the counts then.

    The run's session loads the launch inside ``run_problem`` (SEP 47), and the load
    resets the backend's counters and masks and uploads the state. The timed window
    and the counted evaluations start where they always did: once the state is
    loaded, before the output opens and the launch plane is saved. Within the block
    the backend's ``set_initial`` is wrapped to read the clock and the counter as it
    returns; it is restored on exit.

    Attributes
    ----------
    ended : int
        ``time.perf_counter_ns()`` when the load returned.
    calls : int or None
        The backend's source evaluations just after the load (:func:`source_calls`).
    """

    def __init__(self, engine: Backend) -> None:
        self._engine = engine
        self.ended = 0
        self.calls: int | None = None

    def _load(self, *args: object, **keywords: object) -> None:
        type(self._engine).set_initial(self._engine, *args, **keywords)  # pyright: ignore[reportArgumentType]
        self.calls = source_calls(self._engine)
        self.ended = time.perf_counter_ns()

    def __enter__(self) -> _Loaded:
        self._engine.set_initial = self._load  # type: ignore[method-assign]  # pyright: ignore[reportAttributeAccessIssue]
        return self

    def __exit__(self, kind: type[BaseException] | None, *rest: object) -> None:
        del self._engine.set_initial  # pyright: ignore[reportAttributeAccessIssue]
        if kind is None and not self.ended:
            msg = "the run never loaded its backend, so nothing was timed"
            raise SuiteError(msg)


def timed_run(  # noqa: PLR0913 - the run and how it is observed
    simulation: Simulation,
    problem: KernelProblem,
    engine: Backend,
    retention: FieldRetention,
    monitor: BoundaryMonitor | None = None,
    *,
    output: str = "memory",
    scratch: Path | None = None,
    snapshots: int | None = None,
) -> tuple[Record, np.ndarray | None]:
    """Time one complete run from the launch; return its counts and final field.

    `monitor` is a fresh boundary monitor, attached as ``Simulation.run`` attaches
    one; ``None`` runs without the boundary checks. `output` is ``"memory"`` or
    ``"hdf5"``, a file in `scratch`, read back after the timed window; `snapshots`
    attaches a ``SnapshotController`` with its store in `scratch`, asked after every
    `snapshots`-th accepted step (``0``: never). The final field is ``None`` when the
    last plane keeps none.
    """
    import numpy as np  # noqa: PLC0415

    from solprop.backends.session import Start  # noqa: PLC0415
    from solprop.driver.run import run_problem  # noqa: PLC0415
    from solprop.output import MemoryOutput, SnapshotController  # noqa: PLC0415
    from solprop.output.hdf5 import HDF5Output, open_file  # noqa: PLC0415

    if (output == "hdf5" or snapshots is not None) and scratch is None:
        raise SuiteError("a run to HDF5 or with snapshots needs a scratch directory")
    path = None if scratch is None else scratch / "run.h5"
    handler: Output = (
        HDF5Output(path) if output == "hdf5" and path is not None else MemoryOutput()
    )
    controller = progress = None
    if snapshots is not None and scratch is not None:
        from benchmarks.sessions import Requests  # noqa: PLC0415

        controller = SnapshotController(
            scratch / "snapshots", max_snapshots=SNAPSHOT_CAPACITY
        )
        progress = Requests(controller, snapshots)
    loaded = _Loaded(engine)
    with loaded:
        run_problem(
            problem,
            engine,
            handler,
            start=Start.entrance(problem),
            metadata=simulation.metadata(),
            stats_spec=problem.stats_spec,
            progress="none" if progress is None else progress,
            resume_history=simulation.settings.resume_history,
            fields=retention.fields,
            field_planes=retention.field_planes,
            snapshots=controller,
            boundary_monitor=monitor,
        )
    seconds = (time.perf_counter_ns() - loaded.ended) * 1e-9
    before, after = loaded.calls, source_calls(engine)
    if isinstance(handler, HDF5Output) and path is not None:
        handler = open_file(path)
    try:
        accepted = np.asarray(handler["step_history/accepted"], dtype=bool)
        record: Record = {
            "seconds": seconds,
            "accepted": int(accepted.sum()),
            "rejected": int((~accepted).sum()),
            "source_evaluations": None
            if before is None or after is None
            else after - before,
            "z": float(handler["z"][-1]),
            "planes": handler.n_saved,
        }
        if controller is not None and scratch is not None:
            record["snapshots"] = sum(
                1 for _ in (scratch / "snapshots").glob("snapshot-*")
            )
        final = (
            None
            if retention.fields == "none"
            else handler.plane_field(handler.n_saved - 1, "fft")
        )
        record["finite"] = (
            bool(np.all(np.isfinite(final)))
            if final is not None
            else _finite_statistics(handler)
        )
    finally:
        handler.close()
    return record, final


def _finite_statistics(handler: Output) -> bool | None:
    """Whether a run's last plane's statistics are finite; ``None`` with none."""
    import numpy as np  # noqa: PLC0415

    values = [np.asarray(handler[f"stats/{name}"])[-1] for name in handler.stat_names]
    if not values:
        return None
    return all(bool(np.all(np.isfinite(value))) for value in values)


def count(backend: str) -> object | None:
    """Turn on the native adapter's structural counters, if this build has them.

    Returns the counters to pass to :func:`counted`, or ``None`` on another backend
    or a build without them (one from before SEP 47's counters).
    """
    if backend != "native":
        return None
    try:
        counters = importlib.import_module("solprop_native._counters")
    except ImportError:
        return None
    return counters.enable()


def counted(active: object | None) -> Record | None:
    """Turn the counters off and return their record, or ``None`` if none ran."""
    if active is None:
        return None
    importlib.import_module("solprop_native._counters").disable()
    return active.record()  # pyright: ignore[reportAttributeAccessIssue] - the build's own counters


def _device_free(configuration: Record) -> int | None:
    """Read the accelerator's free bytes on a device run, else ``None``."""
    if not configuration.get("device"):
        return None
    import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    return solprop_native.available_device_memory()


def _repetition(
    configuration: Record,
    simulation: Simulation,
    problem: KernelProblem,
    engine: Backend | None,
    retention: FieldRetention | None,
    scratch: Path,
) -> tuple[Record, np.ndarray | None]:
    """Time one repetition of the case's workload in a fresh `scratch` directory."""
    from benchmarks.registry import CASES  # noqa: PLC0415

    case = CASES[configuration["case"]]
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        if case.workload == "run":
            if engine is None or retention is None:
                raise SuiteError("a single run needs its backend and retention")
            # A user's run checks the boundaries as it goes, so each repeat does too,
            # with a monitor of its own: Simulation.run builds one per run, and a
            # monitor's record accumulates over the run it watches.
            return timed_run(
                simulation,
                problem,
                engine,
                retention,
                simulation._boundary_monitor(),
                output=case.output,
                scratch=scratch,
                snapshots=case.snapshots,
            )
        from benchmarks.sessions import repetition  # noqa: PLC0415

        started = time.perf_counter_ns()
        record, final = repetition(
            case.workload, simulation, scratch, configuration["fields"]
        )
        seconds = (time.perf_counter_ns() - started) * 1e-9
        return {"seconds": seconds, "source_evaluations": None, **record}, final
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _counted(
    configuration: Record,
    simulation: Simulation,
    problem: KernelProblem,
    engine: Backend | None,
    retention: FieldRetention | None,
    root: Path,
) -> Record:
    """Make one more repetition with the structural counters on; return its record."""
    active = count(configuration["backend"])
    try:
        record, _ = _repetition(
            configuration, simulation, problem, engine, retention, root / "counted"
        )
    finally:
        found = counted(active)
    return {**record, "counters": found}


def _warm(
    configuration: Record,
    simulation: Simulation,
    problem: KernelProblem,
    engine: Backend | None,
    retention: FieldRetention | None,
    root: Path,
) -> float:
    """Warm the case up, untimed, and return how long that took.

    A single run warms its backend over the first part of the guide and observes
    once; another workload, which builds its own backends, runs one whole repetition.
    """
    started = time.perf_counter()
    if engine is None:
        _repetition(configuration, simulation, problem, engine, retention, root)
    else:
        warm_up(engine, problem, configuration["warmup"] * configuration["stop"])
    return time.perf_counter() - started


def warm_up(engine: Backend, problem: KernelProblem, until: float) -> None:
    """Warm a backend up, untimed, through a session from the problem's launch.

    The session observes the launch plane, with its statistics and its field, so that
    the observation path's first use is warm too, and propagates until its first
    bounded yield or plane at or past `until`, m: a little past it, by at most one
    yield's work. The backend stays open for the timed runs, each of which starts
    from the launch again.
    """
    from solprop.backends.events import ObservationReady  # noqa: PLC0415
    from solprop.backends.export import ExportRecipe  # noqa: PLC0415
    from solprop.backends.session import Start  # noqa: PLC0415
    from solprop.backends.sessions import start_session  # noqa: PLC0415
    from solprop.translate.run_request import describe_run  # noqa: PLC0415

    session = start_session(
        engine, problem, describe_run(problem), Start.entrance(problem), owns=False
    )
    try:
        while not session.finished:
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady):
                    session.export(event.ticket, ExportRecipe(power=True, optics=False))
            session.acknowledge(batch.receipt)
            if session.status().z >= until:
                break
    finally:
        session.close()


def _repeat_difference(finals: list[np.ndarray]) -> float | None:
    """The largest relative L2 difference of the repeats' final fields from the first.

    Every repeat restarts from the same launch and seed, so its final field may
    differ from the first repeat's by rounding only. ``None`` when no repeat kept a
    final field, so that nothing was compared.
    """
    import numpy as np  # noqa: PLC0415

    if not finals:
        return None
    scale = float(np.linalg.norm(finals[0])) or 1.0
    return max(float(np.linalg.norm(final - finals[0])) / scale for final in finals)


def measure(
    configuration: Record,
    repeats: int,
    fields: Path | None,
    *,
    count_events: bool = False,
    scratch: Path | None = None,
) -> Record:
    """Build once, warm up once, then time ``repeats`` complete runs.

    Each run restarts the backend from the translated launch and goes through
    :func:`solprop.driver.run.run_problem`, the loop of ``Simulation.run``, so it
    includes the plane observations and statistics; a case with another workload
    times that instead (``benchmarks.sessions.repetition``), which constructs its own
    backends, so none is built here and its warm-up is one whole repetition.
    Construction, translation and backend construction are timed separately. With
    `count_events`, one more repetition runs with the structural counters on and is
    reported as ``counted``. Files go to a fresh directory under `scratch` per
    repetition, the system's temporary directory by default.

    ``finite`` is the last repetition's: its final field, or where it keeps none, its
    last plane's statistics; ``None`` when neither was available. ``repeat_difference``
    is ``None`` when no repetition kept a final field.
    """
    import numpy as np  # noqa: PLC0415

    from benchmarks.builders import build  # noqa: PLC0415
    from benchmarks.registry import CASES  # noqa: PLC0415
    from solprop.backends import select  # noqa: PLC0415
    from solprop.driver.executors.pool import peak_rss  # noqa: PLC0415
    from solprop.driver.run import field_retention  # noqa: PLC0415
    from solprop.output.schema import serialise  # noqa: PLC0415

    free_before = _device_free(configuration)
    started = time.perf_counter()
    simulation = build(configuration)
    problem = simulation.translate()
    construction = time.perf_counter() - started
    single = CASES[configuration["case"]].workload == "run"
    started = time.perf_counter()
    engine = (
        select(
            configuration["backend"],
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
            # SOLPROP_BENCH_NODE_BUDGET reaches the backend through its options.
            options=simulation.settings.execution_options(),
        )
        if single
        else None
    )
    backend_seconds = time.perf_counter() - started
    retention = (
        field_retention(
            problem, simulation.settings.fields, simulation.settings.field_planes
        )
        if single
        else None
    )
    controller = problem.stepper.controller
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="solprop-bench-", dir=scratch))
    try:
        warmup = _warm(
            configuration, simulation, problem, engine, retention, root / "warm"
        )
        free_warm = _device_free(configuration)
        repetitions = []
        finals = []
        final = None
        for index in range(repeats):
            repetition, final = _repetition(
                configuration,
                simulation,
                problem,
                engine,
                retention,
                root / f"r{index}",
            )
            repetitions.append(repetition)
            if final is not None:
                finals.append(final)
        counters = (
            _counted(configuration, simulation, problem, engine, retention, root)
            if count_events
            else None
        )
        last = repetitions[-1] if repetitions else {}
        if fields is not None and final is not None:
            fields.parent.mkdir(parents=True, exist_ok=True)
            np.save(fields, final, allow_pickle=False)
        return {
            "calibration": None
            if engine is None
            else calibration_label(engine.metadata()),
            "construction_seconds": construction,
            # Another workload constructs its backends inside each repetition.
            "backend_seconds": backend_seconds,
            "warmup_seconds": warmup,
            "repetitions": repetitions,
            "peak_rss_bytes": peak_rss(),
            # The run's workspace stays allocated, so the drop in free memory from
            # before construction to after the warm-up is what the run holds.
            "device_bytes": None
            if free_before is None or free_warm is None
            else free_before - free_warm,
            "finite": last.get("finite"),
            "counted": counters,
            "repeat_difference": _repeat_difference(finals),
            "integrator": integrator_label(problem.stepper.integrator),
            "precision": configuration.get("precision", "double"),
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
        shutil.rmtree(root, ignore_errors=True)
        if engine is not None:
            engine.close()


def calibration_label(metadata: Record) -> str | None:
    """Name the machine profile a run applied: ``built-in``, or its level and
    generation, with ``+refinement`` where a stage-2 refinement applied (SEP 46)."""
    found = metadata.get("calibration")
    if not isinstance(found, dict):
        return None
    profile = found.get("profile")
    label = (
        f"{profile['level']}:{profile['generation']}"
        if isinstance(profile, dict)
        else "built-in"
    )
    source = str(found.get("workspace_source", ""))
    return label + ("+refinement" if source.startswith("refinement") else "")


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
                count_events=bool(request.get("counters")),
                # Beside the request, in the run's bundle: one disk for both builds.
                scratch=Path(sys.argv[1]).resolve().parent,
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
