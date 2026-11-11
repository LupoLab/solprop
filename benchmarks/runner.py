# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Plan configurations and launch one worker per configuration (SEP 38).

Shared by ``track`` and ``ab``. Nothing here imports a scientific library.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from benchmarks.records import Record, SuiteError, atomic_record, canonical, file_hash
from benchmarks.registry import Case
from benchmarks.system import POOL_VARIABLES, widths

__all__ = [
    "ROOT",
    "estimate",
    "freeze_harness",
    "launch_worker",
    "plan_rows",
    "prepared_python",
    "receipt",
    "show_plan",
]

ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION_SECONDS = 10.0
"""A rough per-configuration allowance for import, construction and translation."""


def prepared_python(value: str) -> str:
    """Locate an interpreter without resolving a virtual environment's symlink."""
    found = shutil.which(value)
    if found is None:
        raise SuiteError(f"Python executable does not exist: {value}")
    return str(Path(found).absolute())


def freeze_harness(directory: Path) -> Record:
    """Copy the harness so that every worker of a run uses the same benchmark code."""
    files = {}
    for source in sorted((ROOT / "benchmarks").glob("*.py")):
        target = directory / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
        elif file_hash(target) != file_hash(source):
            raise SuiteError("stored harness differs from the current source")
        files[str(source.relative_to(ROOT))] = file_hash(target)
    return files


# The suppression below is for PLR0913: the keyword options are the shared
# command-line selection, passed through unchanged.
def plan_rows(
    cases: tuple[Case, ...],
    steppers: tuple[str, ...],
    threads: str | None,
    production: int,
    *,
    backend: str = "native",
    fraction: float = 1.0,
    device: bool = False,
    noise: str = "case",
    fft_size: str = "case",
) -> tuple[list[Record], list[str]]:
    """Resolve configurations in run order; explain every one left out.

    Without `threads`, each case runs at its declared widths; with it, every case
    runs at the widths it names. On a `device`, each case runs once, with the
    production width of host threads, and its CPU cost does not apply.
    """
    rows: list[Record] = []
    skipped: list[str] = []
    for case in cases:
        declared = "production" if device else ",".join(case.widths)
        for width in widths(threads or declared, production):
            for stepper in steppers:
                if stepper not in case.steppers:
                    skipped.append(
                        f"{case.name} {stepper}: "
                        + (
                            "a fixed-step case runs its own schedule only"
                            if case.step is not None
                            else "the case runs its own integrator only"
                        )
                    )
                    continue
                rows.append(
                    {
                        "id": f"{case.name}.{stepper}."
                        + ("device" if device else f"t{width}"),
                        "configuration": case.configuration(
                            stepper,
                            width,
                            backend,
                            fraction,
                            device=device,
                            noise=noise,
                            fft_size=fft_size,
                        ),
                        "cost": None if device else case.cost[0 if width == 1 else 1],
                    }
                )
    rows.sort(key=lambda row: row["configuration"]["threads"])
    return rows, skipped


def estimate(rows: list[Record], runs: int) -> float | None:
    """Estimate seconds for `runs` complete runs per row, or ``None`` if unknown."""
    costs = [row["cost"] for row in rows]
    if any(cost is None for cost in costs):
        return None
    return sum(CONSTRUCTION_SECONDS + cost * (runs + 0.02) for cost in costs)


def show_plan(rows: list[Record], skipped: list[str], runs: int) -> None:
    """Print the configurations and the estimated time, labelled as an estimate."""
    print(f"{'case':26} {'stepper':22} {'threads':>7} {'est. s/run':>10}")
    for row in rows:
        values = row["configuration"]
        cost = "?" if row["cost"] is None else f"{row['cost']:.0f}"
        print(
            f"{values['case']:26} {values['stepper']:22} "
            f"{'device' if values.get('device') else values['threads']:>7} "
            f"{cost:>10}"
        )
    for reason in skipped:
        print(f"skipped {reason}")
    total = estimate(rows, runs)
    print(
        f"{len(rows)} configurations, {runs} timed run(s) each; estimated "
        + ("time unknown" if total is None else f"{total / 60:.0f} min")
        + " (complete-run costs on 16 cores of an AMD EPYC 7B13; other machines differ)"
    )


CALIBRATION_LEVELS = ("thorough", "standard", "quick", "none")
"""``--calibrate``'s levels: how each build is calibrated before a series (SEP 46)."""


def worker_calibration(environ: dict[str, str]) -> dict[str, str]:
    """Return a worker's ``SOLPROP_CALIBRATION``: ``use``, or ``off`` if asked.

    A worker applies the profile the series was calibrated with and never measures
    inside a timed window (SEP 46 section 6.13); ``off`` holds the built-in table for
    an experiment that must keep the settings fixed while the code changes.
    """
    chosen = environ.get("SOLPROP_CALIBRATION", "").strip().lower()
    return {"SOLPROP_CALIBRATION": "off" if chosen == "off" else "use"}


_LEVEL_FLAGS: dict[str, tuple[str, ...]] = {
    "thorough": ("--thorough",),
    "standard": (),
    "quick": ("--quick",),
}
"""The ``solprop calibrate`` flags of each level: ``standard`` is its default."""


def calibrate_builds(
    pythons: dict[str, str],
    level: str,
    output: Path,
    environments: dict[str, dict[str, str]] | None = None,
) -> Record:
    """Calibrate each distinct build at `level` before a series (SEP 46 section 6.13).

    Each interpreter runs ``python -m solprop calibrate`` at the level once (no flag for
    ``standard``, the command's default; ``--quick`` or ``--thorough``), with its arm's
    own variables, so that an A/B's two builds each get their own profile, as two
    releases would for a user; its report is kept under ``calibration/``. ``none``
    calibrates nothing and the series runs on whatever profiles exist; an arm under
    ``SOLPROP_CALIBRATION=off`` runs the built-in table and calibrates nothing either.
    """
    if level == "none":
        return {}
    import os  # noqa: PLC0415

    done: dict[str, Record] = {}
    found: Record = {}
    for arm, python in pythons.items():
        if python in done:
            found[arm] = done[python]
            continue
        environment = dict(os.environ) | (environments or {}).get(arm, {})
        if worker_calibration(environment)["SOLPROP_CALIBRATION"] == "off":
            found[arm] = {"level": "off"}
            continue
        completed = subprocess.run(
            [python, "-m", "solprop", "calibrate", *_LEVEL_FLAGS[level]],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        report = output / "calibration" / f"{arm}.txt"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        done[python] = {
            "level": level,
            "status": completed.returncode,
            "summary": completed.stdout.splitlines()[:1],
        }
        found[arm] = done[python]
        print(
            f"calibrated {arm} ({level}): {' '.join(completed.stdout.splitlines()[:1])}"
        )
    return found


def launch_worker(  # noqa: PLR0917 - called positionally
    python: str,
    request: Record,
    environment: dict[str, str],
    directory: Path,
    timeout: float,
    harness: Path = ROOT,
) -> tuple[list[Record], float]:
    """Run one worker to completion; return its events and its wall time."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "request.json"
    atomic_record(path, request, immutable=True)
    atomic_record(
        directory / "environment.json",
        {
            key: value
            for key, value in environment.items()
            if key in POOL_VARIABLES or key.startswith(("SOLPROP_", "OMP_", "KMP_"))
        },
        immutable=True,
    )
    started = time.perf_counter()
    with (
        (directory / "stdout.log").open("w", encoding="utf-8") as stdout,
        (directory / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        try:
            subprocess.run(
                [python, "-m", "benchmarks.worker", str(path.resolve())],
                cwd=harness,
                env={**environment, "PYTHONPATH": str(harness)},
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            stdout.write(
                canonical(
                    {"event": "result", "status": "failed", "error": "worker timeout"}
                )
                + "\n"
            )
    events = []
    for line in (directory / "stdout.log").read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "event" in event:
            events.append(event)
    return events, time.perf_counter() - started


def receipt(events: list[Record]) -> Record:
    """Return the worker's result, or a failure saying why there is none."""
    result = next((e for e in reversed(events) if e["event"] == "result"), None)
    if result is None:
        return {"status": "failed", "error": "no receipt from the worker"}
    provenance = next(
        (e["provenance"] for e in events if e["event"] == "provenance"), None
    )
    return {
        **{key: value for key, value in result.items() if key != "event"},
        "provenance": provenance,
    }
