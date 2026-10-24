# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Measure several configurations of one build in one process for a fast A/B."""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from benchmarks.records import Record, metadata_value, read_record
from benchmarks.worker import (
    Diagnostics,
    emit,
    measure,
    prepare,
    retained_fields,
    runtime,
    save_plane,
)

__all__ = ["execute", "main", "measure_configuration"]


def measure_configuration(
    configuration: Record, repeats: int, fields: Path | None
) -> Record:
    """Construct once, warm up once, then time ``repeats`` resets of the window.

    Each repetition restores the initial field and the declared lead-in outside
    the timer, exactly as a standard sample's warm repetitions do. The retained
    fields of the last repetition are saved to ``fields`` when it is given.
    """
    import numpy as np  # noqa: PLC0415

    from benchmarks.builders import build  # noqa: PLC0415
    from solprop.backends import select  # noqa: PLC0415

    started = time.perf_counter()
    simulation = build(configuration)
    problem = simulation.translate()
    engine = select(
        configuration["backend"],
        problem,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    construction = time.perf_counter() - started
    try:
        prepare_started = time.perf_counter()
        output = prepare(simulation, problem, engine, configuration)
        first_prepare = time.perf_counter() - prepare_started
        first = measure(engine, problem, configuration, output)
        output.close()
        repetitions = []
        reset_preparations = []
        for _ in range(repeats):
            prepare_started = time.perf_counter()
            output = prepare(simulation, problem, engine, configuration)
            reset_preparations.append(time.perf_counter() - prepare_started)
            repetitions.append(measure(engine, problem, configuration, output))
            if len(repetitions) < repeats:
                output.close()
        observation_started = time.perf_counter()
        if not configuration["track"].startswith("adaptive-"):
            # Fixed tracks observe only after timing, as in a standard sample.
            save_plane(engine, problem, output)
        final_observation = time.perf_counter() - observation_started
        export_started = time.perf_counter()
        result = retained_fields(output)
        output.close()
        if fields is not None:
            fields.parent.mkdir(parents=True, exist_ok=True)
            np.save(fields, result, allow_pickle=False)
        field_export = time.perf_counter() - export_started
        return {
            "construction_seconds": construction,
            "first_prepare_seconds": first_prepare,
            "first_window_seconds": first["seconds"],
            "reset_prepare_seconds": reset_preparations,
            "repetitions": repetitions,
            "final_observation_seconds": final_observation,
            "field_export_seconds": field_export,
            "finite": bool(np.all(np.isfinite(result))),
            "backend_metadata": metadata_value(engine.metadata()),
            "owned_bytes": sum(engine.memory_ledger().values()),
        }
    finally:
        engine.close()


def execute(request: Record, diagnostics: Diagnostics) -> None:
    """Emit the runtime identity, then one receipt per configuration as it ends."""
    rows = request["rows"]
    provenance = runtime(rows[0]["configuration"]["backend"])
    emit("provenance", provenance=provenance)
    fields = request.get("fields")
    for row in rows:
        diagnostics.case = row["configuration"]["case"]
        diagnostics.phase = "measurement"
        try:
            result = measure_configuration(
                row["configuration"],
                request["repeats"],
                None if fields is None else Path(fields) / f"{row['id']}.npy",
            )
            emit("row", id=row["id"], status="ok", **result)
        except Exception as error:  # one failed configuration keeps the others
            traceback.print_exc(file=sys.stderr)
            emit("row", id=row["id"], status="failed", error=f"{error!r}")


def main() -> int:
    """Execute the JSON request passed by the A/B controller."""
    diagnostics = Diagnostics()
    try:
        request = read_record(Path(sys.argv[1]))
        with diagnostics.capture():
            execute(request, diagnostics)
    except Exception as error:  # worker boundary: the controller records the failure
        traceback.print_exc(file=sys.stderr)
        emit("failed", error=f"{type(error).__name__}: {error}")
        return 1
    emit("diagnostics", diagnostics=diagnostics.records())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
