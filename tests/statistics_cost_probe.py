# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the statistics cost assertion with production imports and no test-only hooks."""

import importlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path


def main():
    # The imported witness fixture also uses the repository's private benchmark
    # package. A script invocation otherwise exposes only tests/ on sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    # Match conftest's macOS runtime ordering while keeping production imports.
    if importlib.util.find_spec("solprop_native") is not None:
        importlib.import_module("solprop_native")
    case = importlib.import_module("driver.test_prop_capillary")
    statistics = importlib.import_module("solprop.stats")
    assert not hasattr(statistics.collect, "__wrapped__"), "unexpected test type hook"
    records = {
        "statistics_runtime_typechecking": False,
        "statistics_cost_instrumented": False,
        "load_before": os.getloadavg() if hasattr(os, "getloadavg") else None,
        "selection": sys.argv[2],
    }
    try:
        fixture = inspect.unwrap(case.reduced_m2_channel)()
        records["runs"] = case.statistics_channel_measurement(fixture, sys.argv[2])
    finally:
        records["load_after"] = os.getloadavg() if hasattr(os, "getloadavg") else None
        Path(sys.argv[1]).write_text(json.dumps(records, indent=2) + "\n")
        sys.stdout.write(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
