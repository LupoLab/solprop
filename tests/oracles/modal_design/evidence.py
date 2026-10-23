# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Record immutable, source-identified research receipts."""

import argparse
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    sources = {
        name: subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)
        for name, args in (
            ("head", ["rev-parse", "HEAD"]),
            ("status", ["status", "--porcelain"]),
            ("diff", ["diff", "--stat"]),
        )
    }
    specification = importlib.util.find_spec("solprop")
    if specification is None or specification.origin is None:
        raise RuntimeError("the paired solprop import is unavailable")
    imported_path = Path(specification.origin).resolve()
    if not imported_path.is_relative_to(ROOT / "src"):
        raise RuntimeError(f"solprop import is outside this checkout: {imported_path}")
    return {
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "load_average": os.getloadavg(),
        "logical_cpus": os.cpu_count(),
        "command": sys.argv,
        "sources": sources,
        "solprop_import": str(imported_path),
        "scripts": {
            str(p.relative_to(ROOT)): digest(p)
            for p in Path(__file__).parent.glob("*.py")
        },
        "threads": {
            key: value
            for key, value in os.environ.items()
            if "THREAD" in key or key.startswith("OMP_")
        },
    }


def serialise(value):
    if isinstance(value, np.ndarray):
        return (
            value.tolist()
            if not np.iscomplexobj(value)
            else {"real": value.real.tolist(), "imag": value.imag.tolist()}
        )
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"Cannot serialise {type(value)}")


def run(study):
    parser = argparse.ArgumentParser(description=study.__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_suffix(".sources")
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    receipt = {"provenance": provenance(), "full": args.full}
    receipt["warnings"] = []
    original_warning = warnings.showwarning

    def record_warning(message, category, filename, lineno, file=None, line=None):
        receipt["warnings"].append(
            {
                "category": category.__name__,
                "message": str(message),
                "file": filename,
                "line": lineno,
            }
        )
        original_warning(message, category, filename, lineno, file, line)

    warnings.showwarning = record_warning
    start = time.perf_counter()
    failed = False
    try:
        receipt["results"] = study(args.full)
        receipt["status"] = "completed"
    except Exception:  # preserve an unsuccessful research run, then fail the command
        receipt["status"] = "failed"
        receipt["traceback"] = traceback.format_exc()
        failed = True
    receipt["seconds"] = time.perf_counter() - start
    receipt["load_average_after"] = os.getloadavg()
    warnings.showwarning = original_warning
    import resource  # noqa: PLC0415 - POSIX only, and this receipt is only written there

    receipt["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )
    nonfinite = []

    def clean(value, path=""):
        if isinstance(value, dict):
            return {key: clean(item, f"{path}/{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item, f"{path}/{index}") for index, item in enumerate(value)]
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite.append(path)
            return {"nonfinite": str(value)}
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return clean(serialise(value), path)

    encoded = clean(receipt)
    encoded["nonfinite_paths"] = nonfinite
    args.output.write_text(json.dumps(encoded, indent=2, allow_nan=False) + "\n")
    print(f"{receipt['status']}: {args.output} ({receipt['seconds']:.3f} s)")
    if failed:
        print(receipt["traceback"], file=sys.stderr)
        raise SystemExit(1)
