# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute one declared window for coverage, without warmups or repetitions."""

from benchmarks.records import Record, SuiteError, metadata_value
from benchmarks.worker import (
    Diagnostics,
    accuracy,
    emit,
    measure,
    prepare,
    retained_fields,
    save_plane,
)

__all__ = ["execute_coverage"]


def execute_coverage(
    request: Record, diagnostics: Diagnostics, provenance: Record
) -> Record:
    """Run construction, lead-in, one window and its declared observations."""
    import numpy as np  # noqa: PLC0415 - pre-import environment already installed

    from benchmarks.builders import build  # noqa: PLC0415
    from solprop.backends import select  # noqa: PLC0415
    from solprop.driver.executors.pool import peak_rss  # noqa: PLC0415
    from solprop.output.schema import serialise  # noqa: PLC0415

    if provenance.get("extension_hash") != request["extension_hash"]:
        raise SuiteError("coverage worker imported a different native binary")
    configuration = request["configuration"]
    diagnostics.phase = "construction"
    if configuration["case"] == "batch":
        from benchmarks.batching import construct  # noqa: PLC0415

        case = construct(configuration)
        try:
            diagnostics.phase = "measurement"
            work = case.measure(float(configuration["stop"]))
            outputs = case.observe()
            fields = np.concatenate([retained_fields(output) for output in outputs])
            metadata = case.engine.metadata()
        finally:
            case.engine.close()
    else:
        simulation = build(configuration)
        problem = simulation.translate()
        engine = select(
            "native",
            problem,
            grid=simulation.grid,
            modes=simulation.modes,
            profiles=simulation.profiles,
        )
        try:
            output = prepare(simulation, problem, engine, configuration)
            diagnostics.phase = "measurement"
            work = measure(engine, problem, configuration, output)
            if not configuration["track"].startswith("adaptive-"):
                save_plane(engine, problem, output)
            fields = retained_fields(output)
            output.close()
            metadata = engine.metadata()
        finally:
            engine.close()
    # The common parent needs completion and memory boundaries. None of these
    # diagnostic wall times become a warm sample or enter timing comparisons.
    emit("first-window")
    emit("measurement", peak_rss_bytes=peak_rss())
    check = accuracy(fields, configuration, request)
    return {
        "purpose": "coverage",
        "status": "invalid" if check["status"] == "invalid" else "ok",
        "provenance": provenance,
        "accuracy": check,
        "work": {k: v for k, v in work.items() if not k.endswith("seconds")},
        "metadata": metadata_value(serialise(metadata)),
    }
