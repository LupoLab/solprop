# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collect serial fresh-process attempts and retain every completed receipt."""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from benchmarks.comparison import aa_control, compare, completed, needs_more
from benchmarks.records import (
    Record,
    SuiteError,
    atomic_record,
    attempts,
    export_tables,
    file_hash,
    identity,
    now,
    read_record,
    run_lock,
)
from benchmarks.system import (
    admission,
    counters,
    git_receipt,
    guard,
    interval,
    thread_environment,
)

__all__ = [
    "ROOT",
    "Attempt",
    "Collection",
    "Monitor",
    "calibrate",
    "calibration",
    "collect",
    "collect_round",
    "environment_identity",
    "freeze_harness",
    "freeze_manifest",
    "inventory",
    "launch",
    "ordered",
    "persist",
    "prepared_python",
    "prior_validation",
    "reference_for",
    "stop_tree",
]

ROOT = Path(__file__).resolve().parents[1]


@dataclasses.dataclass(frozen=True)
class Collection:
    """Freeze execution limits and prepared interpreters for an entire collection."""

    output: Path
    baseline_python: str | None = None
    candidate_python: str = sys.executable
    seed: int = 30
    timeout: float = 1800.0
    memory_limit: int | None = None
    force: bool = False
    allow_unestimated: bool = False
    resume: bool = False
    references: Path = Path(".benchmarks/suite/references")
    allow_accuracy_failure: bool = False

    def __post_init__(self) -> None:
        """Keep executable paths valid when a worker changes its directory."""
        for name in ("baseline_python", "candidate_python"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, prepared_python(value))


def prepared_python(value: str) -> str:
    """Locate an interpreter without resolving a virtual environment's symlink."""
    found = shutil.which(value)
    if found is None:
        raise SuiteError(f"prepared Python executable does not exist: {value}")
    return str(Path(found).absolute())


@dataclasses.dataclass(frozen=True)
class Attempt:
    """Identify one immutable attempt within a balanced round."""

    arm: str
    round_index: int
    repetitions: int = 1
    phase: str = "measurement"


@dataclasses.dataclass
class Monitor:
    """Accumulate parent observations while its one worker remains alive."""

    started: float
    allocation: tuple[int, ...]
    startup: float | None = None
    result: Record | None = None
    own_cpu: float | None = None
    sampled_peak: int | None = None
    samples: int = 0
    missed_samples: int = 0
    measurement_peak: int | None = None
    measurement_seen: bool = False
    measurement_system: Record | None = None
    measurement_cpu: float | None = None
    exceeded: str | None = None

    def guard_receipt(self, before: Record) -> Record:
        """Score load through measurement, retaining missing worker CPU confidence."""
        during = interval(
            before,
            self.measurement_system or counters(),
            self.allocation,
            self.measurement_cpu
            if self.measurement_cpu is not None
            else self.own_cpu or 0.0,
        )
        if self.measurement_cpu is None:
            during.update(
                confidence="unknown",
                quiet=False,
                reason="worker CPU usage unavailable at the measurement boundary",
            )
        return during

    def event(self, received: float, line: str) -> None:
        """Accept only declared protocol events; keep other stdout as diagnostics."""
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return
        if event.get("event") == "first-window":
            self.startup = received - self.started
        elif event.get("event") == "measurement":
            self.measurement_peak = event.get("peak_rss_bytes")
            self.measurement_seen = True
            self.measurement_system = counters()
            self.measurement_cpu = self.own_cpu
        elif event.get("event") == "result":
            self.result = event["result"]

    def sample(
        self, owner: psutil.Process | None, limits: tuple[float, int | None]
    ) -> None:
        """Sample only the owned process, without inventing unavailable counters."""
        if owner is not None:
            try:
                times = owner.cpu_times()
                self.own_cpu = times.user + times.system
                self.sampled_peak = max(self.sampled_peak or 0, owner.memory_info().rss)
                self.samples += 1
            except psutil.Error:
                self.missed_samples += 1
        else:
            self.missed_samples += 1
        if time.monotonic() - self.started > limits[0]:
            self.exceeded = "timeout"
        if (
            limits[1] is not None
            and self.sampled_peak is not None
            and self.sampled_peak > limits[1]
        ):
            self.exceeded = "memory-limit"


def stop_tree(process: subprocess.Popen[str], born: float | None) -> None:
    """Terminate only processes belonging to this attempt, checking PID reuse."""
    try:
        owner = psutil.Process(process.pid)
        if born is not None and owner.create_time() != born:
            return
        try:
            children = owner.children(recursive=True)
        except psutil.Error, OSError:
            # Restricted process inventories can still signal the directly
            # owned child. Never substitute a machine-wide process-name kill.
            children = []
        for child in reversed(children):
            with contextlib.suppress(psutil.NoSuchProcess):
                child.terminate()
        owner.terminate()
        _, alive = psutil.wait_procs([*children, owner], timeout=2)
        for child in alive:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
    except psutil.NoSuchProcess:
        pass


def launch(
    python: str,
    request: Record,
    environment: dict[str, str],
    directory: Path,
    limits: tuple[float, int | None],
    allocation: tuple[int, ...],
) -> Record:
    """Launch a worker, timestamp receipts and enforce limits from its parent."""
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    atomic_record(request_path, request, immutable=True)
    events: queue.Queue[tuple[float, str]] = queue.Queue()
    harness_root = Path(request.get("harness_root", ROOT)).resolve()
    environment = {**environment, "PYTHONPATH": str(harness_root)}
    started, controller_cpu = time.monotonic(), time.process_time()
    before = counters()
    monitor = Monitor(started, allocation)
    with (
        (directory / "stderr.log").open("w", encoding="utf-8") as errors,
        (directory / "stdout.log").open("w", encoding="utf-8") as output,
    ):
        process = subprocess.Popen(
            [python, "-m", "benchmarks.worker", str(request_path.resolve())],
            cwd=harness_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            encoding="utf-8",
            start_new_session=os.name != "nt",
        )
        owner: psutil.Process | None = None
        try:
            owner = psutil.Process(process.pid)
            born = owner.create_time()
        except psutil.NoSuchProcess:
            born = None

        def read_events() -> None:
            assert process.stdout is not None
            with process.stdout as stream:
                for line in stream:
                    events.put((time.monotonic(), line))

        reader = threading.Thread(target=read_events, daemon=True)
        reader.start()
        while process.poll() is None or reader.is_alive() or not events.empty():
            try:
                received, line = events.get(timeout=0.1)
                output.write(line)
                output.flush()
                monitor.sample(owner, limits)
                monitor.event(received, line)
            except queue.Empty:
                pass
            if process.poll() is None:
                monitor.sample(owner, limits)
            if monitor.exceeded:
                stop_tree(process, born)
                break
        reader.join(timeout=2)
        returncode = process.wait(timeout=5)
    during = monitor.guard_receipt(before)
    result = monitor.result
    if monitor.exceeded:
        result = {
            "status": monitor.exceeded,
            "error": f"attempt exceeded its {monitor.exceeded}",
        }
    elif result is None:
        result = {
            "status": "failed",
            "error": f"missing result receipt; exit {returncode}",
        }
    elif returncode != 0 and result.get("status") == "ok":
        result = {
            **result,
            "status": "failed",
            "error": f"worker exited {returncode} after receipt",
        }
    elif (
        result.get("status") == "ok"
        and not request.get("inventory")
        and (monitor.startup is None or not monitor.measurement_seen)
    ):
        result = {**result, "status": "failed", "error": "missing boundary receipt"}
    return {
        **result,
        "startup_seconds": monitor.startup,
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - started,
        "controller_cpu_seconds": time.process_time() - controller_cpu,
        "sampled_peak_bytes": monitor.sampled_peak,
        "sampled_peak_reason": None
        if monitor.sampled_peak is not None
        else "no successful process RSS poll",
        "measurement_peak_bytes": monitor.measurement_peak,
        "process": {
            "pid": process.pid,
            "created": born,
            "cpu_seconds": monitor.own_cpu,
            "cpu_reason": None
            if monitor.own_cpu is not None
            else "process CPU counters unavailable",
            "rss_samples": monitor.samples,
            "missed_samples": monitor.missed_samples,
            "poll_interval_seconds": 0.1,
        },
        "during_guard": during,
    }


def ordered(
    rows: list[Record], seed: int, round_index: int, paired: bool
) -> list[tuple[Record, tuple[str, ...]]]:
    """Shuffle each balanced round while keeping reversed A/B pairs adjacent."""
    shuffled = list(rows)
    random.Random(seed + round_index).shuffle(shuffled)
    arms = (
        ("baseline", "candidate") if round_index % 2 == 0 else ("candidate", "baseline")
    )
    return [(row, arms if paired else ("candidate",)) for row in shuffled]


def calibration(times: list[float], resolution: float) -> tuple[int, bool]:
    """Freeze a common repetition count from paired pilots, without a time floor."""
    target = max(0.1, 1000 * resolution)
    proposed = max(1, math.ceil(target / min(times)))
    return min(10, proposed), proposed > 10


def reference_for(directory: Path, configuration: Record) -> str | None:
    """Find a qualification receipt for this exact measurement identity."""
    path = directory / f"{configuration['case']}-{configuration['track']}.json"
    return str(path.resolve()) if path.exists() else None


def prior_validation(
    previous: list[Record], configuration_id: str, arm: str, reference: str | None
) -> Record | None:
    """Reuse a successful check only within the same arm and reference revision."""
    if reference is None:
        return None
    try:
        reference_id = identity(read_record(Path(reference)))
    except OSError, ValueError:
        return None
    for sample in reversed(previous):
        if sample["configuration_id"] != configuration_id or sample["arm"] != arm:
            continue
        check = sample.get("accuracy", {})
        if (
            sample["status"] != "ok"
            or sample.get("accuracy_overridden")
            or check.get("status") != "passed"
            or check.get("reference_id") != reference_id
            or not check.get("field_shape")
        ):
            return None
        return {
            "status": "passed",
            "reference_id": reference_id,
            "qualified_for_adoption": check.get("qualified_for_adoption", False),
            "reference_revision": check.get("reference_revision", 1),
            "policy_revision": check.get("policy_revision", 1),
            "field_shape": check["field_shape"],
            "validation_sample_id": check.get(
                "validation_sample_id", sample["sample_id"]
            ),
            "validation": "prior sample; current shape and finiteness checked",
        }
    return None


def _attempt(
    row: Record,
    attempt: Attempt,
    options: Collection,
    plan: Record,
    guard_receipt: Record,
) -> Record:
    previous = attempts(options.output)
    attempt_id = max((item["attempt_id"] for item in previous), default=-1) + 1
    # A killed collector can leave worker files without a completed shard.
    # Keep those diagnostics intact and reserve a fresh directory on resume.
    while (options.output / "raw" / f"{attempt_id:06d}").exists():
        attempt_id += 1
    configuration = row["configuration"]
    arm, round_index = attempt.arm, attempt.round_index
    sample_id = identity([row["id"], arm, round_index, attempt.phase])
    base = {
        "attempt_id": attempt_id,
        "sample_id": sample_id,
        "configuration_id": row["id"],
        "arm": arm,
        "round": round_index,
        "phase": attempt.phase,
        "created": now(),
        "guard": guard_receipt,
    }
    result: Record
    if row["status"] != "ready":
        result = {"status": "excluded", "error": row["reason"]}
    else:
        estimate_invalid = any(
            item.get("estimator_defect")
            for item in previous
            if item["configuration_id"] == row["id"]
        )
        admit = admission(
            None if estimate_invalid else row["estimate_bytes"],
            options.memory_limit,
            allow_unestimated=options.allow_unestimated,
        )
        if not admit["admitted"]:
            result = {"status": "refused", "admission": admit}
        else:
            profile = plan["profile"]
            environment = thread_environment(
                configuration["threads"], profile["policy"], dict(os.environ)
            )
            reference = reference_for(options.references, configuration)
            request = {
                "configuration": configuration,
                "repetitions": attempt.repetitions,
                "reference": reference,
                "prior_validation": prior_validation(
                    previous, row["id"], arm, reference
                ),
                "harness_root": str((options.output / "harness").resolve()),
                "allow_accuracy_failure": options.allow_accuracy_failure,
                "thread_diagnostics": plan.get("thread_diagnostics", False),
            }
            python = (
                options.baseline_python
                if arm == "baseline"
                else options.candidate_python
            )
            assert python is not None
            try:
                result = launch(
                    python,
                    request,
                    environment,
                    options.output / "raw" / f"{attempt_id:06d}",
                    (options.timeout, admit["limit_bytes"]),
                    tuple(profile["allocation"]),
                )
            except (OSError, SuiteError) as error:
                result = {"status": "failed", "error": f"worker launch: {error}"}
            if result["status"] == "ok" and environment_identity(
                result["provenance"]
            ) != (environment_identity(plan["environments"][arm])):
                result.update(
                    status="failed", error="worker environment identity changed"
                )
            result["admission"] = admit
            result["incoming_threads"] = {
                key: value
                for key, value in os.environ.items()
                if key in environment
                and (key.endswith("THREADS") or key.startswith("SOLPROP_FFT"))
            }
            result["effective_threads"] = {
                key: value
                for key, value in environment.items()
                if key.endswith("THREADS") or key.startswith("SOLPROP_FFT")
            }
            peak = result.get("peak_rss_bytes")
            result["estimator_defect"] = (
                peak is not None
                and row["estimate_bytes"] is not None
                and peak > row["estimate_bytes"]
            )
    result["clean"] = bool(
        guard_receipt.get("quiet")
        and result.get("during_guard", {}).get("quiet")
        and not options.force
    )
    result["warning_unreviewed"] = any(
        not item["expected"] for item in result.get("diagnostics", [])
    )
    return {**base, **result}


def freeze_harness(directory: Path) -> Record:
    """Copy the small harness so later workers cannot observe mid-run edits."""
    files = {}
    for source in sorted((ROOT / "benchmarks").glob("*.py")):
        target = directory / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_hash(target) != file_hash(source):
                raise SuiteError("stored harness differs from current source")
        else:
            target.write_bytes(source.read_bytes())
        files[str(source.relative_to(ROOT))] = file_hash(target)
    return files


def persist(options: Collection, result: Record) -> None:
    """Commit one immutable shard before refreshing derived views."""
    atomic_record(
        options.output / "attempts" / f"{result['attempt_id']:06d}.json",
        result,
        immutable=True,
    )
    export_tables(options.output)


def inventory(plan: Record, options: Collection) -> Record:
    """Identify prepared environments once, before any timed samples."""
    result: Record = {}
    for arm, python in (
        ("baseline", options.baseline_python),
        ("candidate", options.candidate_python),
    ):
        if python is None:
            continue
        profile = plan["profile"]
        record = launch(
            python,
            {
                "configuration": plan["rows"][0]["configuration"],
                "inventory": True,
                "harness_root": str((options.output / "harness").resolve()),
            },
            thread_environment(
                profile["production"], profile["policy"], dict(os.environ)
            ),
            options.output / "inventories" / f"{arm}-{time.time_ns()}",
            (options.timeout, options.memory_limit),
            tuple(profile["allocation"]),
        )
        if record["status"] != "ok":
            raise SuiteError(f"{arm} inventory failed: {record.get('error')}")
        result[arm] = record["provenance"]
    return result


def environment_identity(provenance: Record) -> Record:
    """Separate source/build identity from runtime telemetry and workload identity."""
    result = {
        key: provenance.get(key)
        for key in (
            "packages",
            "solprop_path",
            "python_executable",
            "extension_hash",
            "build_info",
        )
    }
    if result["build_info"] is not None:
        # BuildInfo also contains process-local pool widths and policies. Inventory
        # runs at production width; a one-thread sample, whose pools are all set
        # explicitly, must still identify the same extension. Keep its actual
        # widths in the unmodified sample receipt.
        runtime_widths = {
            "explicit_thread_pools",
            "blas_thread_policy",
            "requested_threads",
            "runtime_threads",
            "thread_count",
            "fftw_threads",
            "raman_fft_threads",
            "raman_tail_fft_threads",
            "blas_threads",
        }
        result["build_info"] = {
            key: value
            for key, value in result["build_info"].items()
            if key not in runtime_widths
        }
    return result


def freeze_manifest(plan: Record, options: Collection) -> Record:
    """Validate resume against the live harness and actual imported environments."""
    selected_options = dataclasses.asdict(options)
    for key in ("output", "references"):
        selected_options[key] = str(selected_options[key].resolve())
    selected_options.pop("resume")
    plan = {
        **plan,
        "options": selected_options,
        "exploratory": (
            options.force or options.allow_unestimated or options.allow_accuracy_failure
        ),
        "harness": git_receipt(ROOT),
        "harness_files": freeze_harness(options.output / "harness"),
        "environments": inventory(plan, options),
    }
    manifest_path = options.output / "manifest.json"
    if manifest_path.exists():
        if not options.resume:
            raise SuiteError("output already exists; select --resume or a new output")
        saved = read_record(manifest_path)
        if saved["plan_hash"] != identity(plan):
            raise SuiteError(
                "resume plan, profile, environment or harness identity differs"
            )
    else:
        if options.resume:
            raise SuiteError("cannot resume a run without a manifest")
        atomic_record(
            manifest_path,
            {"created": now(), "plan": plan, "plan_hash": identity(plan)},
            immutable=True,
        )
    return plan


def calibrate(plan: Record, options: Collection) -> Record:
    """Persist pilots and freeze common repetition counts before standard rounds."""
    path = options.output / "repetitions.json"
    counts = read_record(path) if path.exists() else {}
    paired = options.baseline_python is not None
    for row, arms in ordered(plan["rows"], options.seed, -1, paired):
        if row["id"] in counts:
            continue
        count, capped = 1, False
        if plan["mode"] == "standard" and row["status"] == "ready":
            pilots = []
            for arm in arms:
                previous = [
                    item
                    for item in attempts(options.output)
                    if item.get("phase") == "pilot"
                    and item["configuration_id"] == row["id"]
                    and item["arm"] == arm
                    and item["status"] == "ok"
                ]
                if previous:
                    pilots.append(previous[-1])
                    continue
                configuration = row["configuration"]
                print(
                    f"pilot: {configuration['case']} {configuration['track']} "
                    f"{configuration['threads']} threads {arm}",
                    flush=True,
                )
                receipt = guard(
                    tuple(plan["profile"]["allocation"]),
                    standard=True,
                    force=options.force,
                )
                result = _attempt(
                    row, Attempt(arm, -1, phase="pilot"), options, plan, receipt
                )
                persist(options, result)
                if result["status"] == "ok":
                    pilots.append(result)
            if len(pilots) != len(arms):
                raise SuiteError(
                    "calibration failed; attempts are retained; resume to retry "
                    "before collecting measurements"
                )
            count, capped = calibration(
                [item["seconds"] for item in pilots],
                max(item["timer_resolution"] for item in pilots),
            )
        counts[row["id"]] = {"count": count, "capped": capped}
        atomic_record(path, counts)
    return counts


def collect_round(
    plan: Record, options: Collection, round_index: int, counts: Record
) -> None:
    """Run one balanced round with at most one whole-pair guard deferral."""
    successful = {
        row["sample_id"]
        for row in completed(attempts(options.output))
        if row["status"] == "ok"
    }
    jobs = [
        (row, arms, False)
        for row, arms in ordered(
            plan["rows"], options.seed, round_index, options.baseline_python is not None
        )
    ]
    while jobs:
        row, arms, deferred = jobs.pop(0)
        remaining = [
            arm
            for arm in arms
            if identity([row["id"], arm, round_index, "measurement"]) not in successful
        ]
        if not remaining:
            continue
        guard_receipt = guard(
            tuple(plan["profile"]["allocation"]),
            standard=plan["mode"] == "standard",
            force=options.force,
        )
        if (
            plan["mode"] == "standard"
            and not guard_receipt["quiet"]
            and not deferred
            and not options.force
        ):
            jobs.append((row, arms, True))
            continue
        for arm in remaining:
            result = _attempt(
                row,
                Attempt(arm, round_index, counts[row["id"]]["count"]),
                options,
                plan,
                guard_receipt,
            )
            persist(options, result)
            configuration = row["configuration"]
            print(
                f"round {round_index + 1}: {configuration['case']} "
                f"{configuration['track']} {configuration['threads']} threads "
                f"{arm}: {result['status']}",
                flush=True,
            )


def collect(plan: Record, options: Collection) -> Record:
    """Collect/resume a frozen plan with durable failure rows and bounded retries."""
    started = time.perf_counter()
    if not plan["rows"]:
        raise SuiteError("selection is empty")
    paired = options.baseline_python is not None
    configuration_ids = [row["id"] for row in plan["rows"]]
    with run_lock(options.output):
        plan = freeze_manifest(plan, options)
        counts = calibrate(plan, options)
        for round_index in range(plan["maximum_rounds"]):
            rows = attempts(options.output)
            if round_index >= plan["rounds"] and not needs_more(
                rows,
                configuration_ids,
                ("baseline", "candidate") if paired else ("candidate",),
            ):
                break
            collect_round(plan, options, round_index, counts)
            # Explicit resume retries a failure; automatic extra rounds address
            # dispersion and contention only, never silently discard a crash.
            if any(
                row["status"] != "ok" for row in completed(attempts(options.output))
            ):
                break
        rows = attempts(options.output)
        result: Record = (
            compare(plan, rows)
            if paired
            else {"status": "collected", "adoption": "requires comparison"}
        )
        selected = completed(rows)
        result["accuracy_failures"] = sum(
            row.get("accuracy", {}).get("status") == "invalid" for row in selected
        )
        result["accuracy_override"] = options.allow_accuracy_failure
        result["collection_seconds"] = time.perf_counter() - started
        if options.allow_accuracy_failure:
            result.update(status="exploratory", adoption="accuracy override enabled")
        result["execution_ok"] = bool(selected) and all(
            row["status"] == "ok" for row in selected
        )
        atomic_record(options.output / "comparison.json", result)
        if paired and options.baseline_python == options.candidate_python:
            atomic_record(options.output / "aa.json", aa_control(plan, rows))
        export_tables(options.output)
        return result
