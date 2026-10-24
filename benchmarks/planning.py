# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Expand an immutable measurement matrix without constructing a simulation."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from benchmarks.records import (
    SCHEMA_VERSION,
    SUITE_VERSION,
    Record,
    SuiteError,
    identity,
    read_record,
)
from benchmarks.references import measurement_identity
from benchmarks.registry import ADAPTIVE, FIXED, KERR, TRACKS, Case
from benchmarks.system import Profile, topology
from benchmarks.telemetry import allocation_details, hardware_details

__all__ = ["Selection", "estimate", "matrix", "reference_status", "widths"]


def reference_status(directory: Path | None, configuration: Record) -> str:
    """Inspect receipt metadata without loading arrays or constructing a case."""
    if directory is None or "case_hash" not in configuration:
        return "not-qualified"
    path = directory / f"{configuration['case']}-{configuration['track']}.json"
    if not path.exists():
        return "not-qualified"
    try:
        receipt = read_record(path)
    except OSError, ValueError:
        return "invalid receipt"
    if (
        receipt.get("revision") != 1
        or receipt.get("case_hash") != configuration["case_hash"]
        or receipt.get("measurement_hash") != measurement_identity(configuration)
    ):
        return "incompatible receipt"
    return (
        "reviewed receipt; arrays checked by worker"
        if receipt.get("qualified_for_adoption")
        else "available receipt; review pending"
    )


def widths(value: str | None, profile: Profile, mode: str) -> tuple[int, ...]:
    """Resolve positive thread budgets without silently clamping them."""
    tokens = (value or ("production" if mode == "quick" else "1,production")).split(",")
    try:
        result = tuple(
            dict.fromkeys(
                profile.production if item == "production" else int(item)
                for item in tokens
            )
        )
    except ValueError as error:
        raise SuiteError("threads must be positive integers or production") from error
    if not result or any(item < 1 or item > len(profile.allocation) for item in result):
        raise SuiteError(
            "thread width exceeds the profile allocation or is not positive"
        )
    return result


def estimate(case: Case, track: str) -> int | None:
    """Bound core construction conservatively until empirical receipts supersede it.

    Cartesian RK4IP allowance: 24 complex field volumes plus 4 GiB for libraries,
    FFT plans, tables and transient copies. SEP 28's retained low-storage ledger
    holds five field volumes; the additional nineteen cover launch, translation,
    NumPy temporaries, output conversion and qualification copies. This does not
    admit an Adams history at the same size.
    Other core cases reserve 8 GiB, above their documented array inventories.
    These are explicitly provisional admission budgets, not measured peak RSS.
    """
    if case.estimate_bytes is not None:
        return case.estimate_bytes
    if case.builder == "pending":
        return None
    if case.builder == "kerr":
        if track != KERR:
            return None
        return 24 * 16 * case.dimensions[0] ** 2 * case.dimensions[-1] + 4 * 2**30
    if case.name.startswith("cartesian_filament"):
        # Full field state plus RK registers or Adams state/source histories,
        # propagator tables, launch/output copies and FFT temporaries. Reserve
        # 96 volumes for adaptive history and 48 for fixed RK, plus runtime and
        # node-chunk work. Empirical peaks still validate this provisional cap.
        volumes = 96 if track == ADAPTIVE[0] else 48
        bins = case.dimensions[-2] // 2 + 1
        return volumes * 16 * case.dimensions[0] ** 2 * bins + 8 * 2**30
    return 8 * 2**30


@dataclasses.dataclass(frozen=True)
class Selection:
    """Freeze matrix selection separately from machine and case identities."""

    mode: str = "quick"
    backend: str = "native"
    threads: str | None = None
    tracks: tuple[str, ...] = ()
    receipts: Path | None = None
    diagnostic: bool = False


def matrix(cases: tuple[Case, ...], profile: Profile, selection: Selection) -> Record:
    """Resolve selections and exclusions into a persistable collection plan."""
    mode, backend = selection.mode, selection.backend
    if mode not in ("quick", "standard") or backend not in (
        "numpy_ref",
        "numba_cpu",
        "native",
        "julia",
    ):
        raise SuiteError("unknown mode or backend")
    if any(track not in TRACKS for track in selection.tracks):
        raise SuiteError("unknown track")
    rows: list[Record] = []
    for case in cases:
        chosen = selection.tracks or (
            (
                tuple(
                    track for track in case.tracks if track in ADAPTIVE or track == KERR
                )
                or case.tracks
            )
            if mode == "quick"
            else case.tracks
        )
        for track in dict.fromkeys(chosen):
            for width in widths(selection.threads, profile, mode):
                reason = case.pending_reason
                if track not in case.tracks:
                    reason = "track not admitted by this case"
                if (
                    case.name == "kerr_beam_768"
                    and width == 1
                    and not selection.diagnostic
                ):
                    reason = "one thread requires explicit diagnostic selection"
                schedule: tuple[float, ...] = ()
                if track == FIXED and case.step is None:
                    if track in case.tracks:
                        schedule = tuple(case.measurement(track)["schedule"])
                    schedule_path = (
                        None
                        if selection.receipts is None
                        else selection.receipts / f"{case.name}-schedule.json"
                    )
                    if schedule_path is not None and schedule_path.exists():
                        receipt = read_record(schedule_path)
                        if receipt["case_hash"] == case.digest:
                            schedule = tuple(receipt["edges"])
                    if not schedule:
                        reason = "fixed schedule pending separate qualification"
                values: Record = (
                    case.measurement(track, schedule)
                    if track in case.tracks
                    else {"case": case.name, "track": track}
                )
                values.update(
                    backend=backend,
                    threads=width,
                    policy=profile.policy,
                    profile_hash=profile.digest,
                )
                key = identity(values)
                rows.append(
                    {
                        "id": key,
                        "configuration": values,
                        "status": "excluded" if reason else "ready",
                        "reason": reason,
                        "estimate_bytes": estimate(case, track),
                        "estimate_status": "provisional",
                        "reference": reference_status(selection.receipts, values),
                        "pilot_seconds": None,
                        "pilot_reason": "no compatible pilot receipt",
                    }
                )
    return {
        "schema_version": SCHEMA_VERSION,
        "suite_version": SUITE_VERSION,
        "mode": mode,
        "profile": dataclasses.asdict(profile),
        "system": {**topology(), **hardware_details(profile.allocation)},
        "allocation": allocation_details(),
        "rows": rows,
        "rounds": 1 if mode == "quick" else 5,
        "maximum_rounds": 1 if mode == "quick" else 11,
        "noise_extension_cost": "unknown until compatible pilots exist",
    }
