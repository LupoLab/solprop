# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Convergence records: what the ladders measured, stored beside the witnesses.

Every convergence study on ``docs/dev/accuracy.md`` writes one small JSON file under
``tests/data/convergence/``. The file is committed like a witness and is the ladder's
regression witness: on every ordinary run the study recomputes its numbers as it always
did and compares them with the stored ones, and the figures on the accuracy page are
drawn from the same files by ``docs/dev/accuracy_figures.py``.

A record is written **only when asked** -- ``SOLPROP_WRITE_RECORDS=1``, serially, from a
tree that is clean outside ``tests/data/convergence/`` -- so no lane ever rewrites one.
``docs/dev/accuracy.md`` section 9 carries the commands.

Rows are merged rather than replaced, because several studies are measured by more than
one test: the backend-equivalence ladder is six functions in one file plus one in
another, and the M2 order study's four switches are split across three tests in two
tiers. Merging is also why provenance is stamped **per row**: a default-lane write run
must not restamp rows that a slow-lane run measured at another commit.
"""

import dataclasses
import json
import math
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import threadpoolctl

from distribution import original_nodeid

REPO = Path(__file__).resolve().parents[1]
RECORDS = Path(__file__).resolve().parent / "data" / "convergence"
RELATIVE = "tests/data/convergence"

SCHEMA_VERSION = 1
WRITE_ENV = "SOLPROP_WRITE_RECORDS"

# The pools and plan flags that change the last digits of a propagation, recorded with
# every write so that a row measured under `poe test-slow`'s pins is not silently
# compared against one taken with a many-threaded BLAS.
THREAD_VARIABLES = (
    "NUMBA_NUM_THREADS",
    "SOLPROP_THREADS",
    "SOLPROP_NUMBA_THREADS",
    "SOLPROP_OVERSUBSCRIBE",
    "SOLPROP_NATIVE_THREADS",
    "SOLPROP_FFTW_THREADS",
    "SOLPROP_TRANSVERSE_FFTW_THREADS",
    "SOLPROP_BLAS_THREADS",
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OMP_WAIT_POLICY",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)

_WRITTEN: set[Path] = set()


class RecordError(RuntimeError):
    """A record could not be written, or a study disagrees with its stored record."""


@dataclasses.dataclass(frozen=True)
class Column:
    """One measured quantity of a study, with the bound it is compared against.

    ``rtol`` of ``None`` means the column is recorded but not compared -- an accepted
    step count, which is not reproducible across machines, or a quantity whose only
    claim is the ``bound`` its own test asserts. ``atol`` is what makes a comparison
    survive a round-off-limited tail, where a relative tolerance means nothing.

    A column's value may be ``None`` on a rung where the quantity does not exist -- the
    coarsest rung of a ladder has no observed order, because an order is a property of
    a pair. It is then ``null`` in the file, and a rung that gains or loses one is a
    failure like any other change.
    """

    name: str
    unit: str
    why: str
    bound: float | None = None
    rtol: float | None = None
    atol: float = 0.0

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Study:
    """What a record is of: its name, the case it was run on and what varies.

    Everything here is compared against the stored record on every ordinary run, so a
    figure's caption cannot come to describe a case nobody ran. It is a frozen
    dataclass rather than five keyword arguments because the study's identity is one
    thing and is usually shared by several tests that each write part of it.
    """

    name: str
    title: str
    case: dict[str, Any]
    axis: dict[str, str]
    columns: tuple[Column, ...]
    luna: dict[str, Any] | None = None

    def skeleton(self) -> dict[str, Any]:
        """Return an empty record of this study."""
        return {
            "schema": SCHEMA_VERSION,
            "study": self.name,
            "title": self.title,
            "case": self.case,
            "axis": self.axis,
            "columns": [column.as_json() for column in self.columns],
            "luna": self.luna,
            "writes": [],
            "rows": [],
        }


# --------------------------------------------------------------- the write gate


def writing() -> bool:
    """Return whether this session was asked to write records."""
    return os.environ.get(WRITE_ENV) == "1"


def _git(*arguments: str) -> str | None:
    """Return the output of a read-only git command in this checkout, or None."""
    try:
        finished = subprocess.run(
            ["git", "-C", str(REPO), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return finished.stdout.strip() if finished.returncode == 0 else None


def dirty_paths() -> list[str] | None:
    """Return the paths that make the tree unfit to write a record from.

    The record files themselves are excluded: a write run modifies them by definition,
    and after the first commit they are tracked, so `GitProvenance.detect`'s
    tracked-only notion of "dirty" would refuse every write after the first one.
    Untracked files elsewhere *do* count -- an untracked module under ``src/`` is
    exactly the thing that silently changes what a ladder measures.
    """
    status = _git("status", "--porcelain", "--", ".", f":(exclude){RELATIVE}")
    if status is None:
        return None
    return [line.strip() for line in status.splitlines() if line.strip()]


def _require_writable() -> None:
    """Raise unless this process may write records."""
    if os.environ.get("CI"):
        msg = f"{WRITE_ENV} must not be set in CI: a lane would rewrite its own witness"
        raise RecordError(msg)
    if os.environ.get("PYTEST_XDIST_WORKER"):
        msg = (
            "records cannot be written under xdist: several workers would race on one "
            "file. Regenerate serially -- docs/dev/accuracy.md section 9"
        )
        raise RecordError(msg)
    dirty = dirty_paths()
    if dirty is None:
        msg = f"no git checkout at {REPO}: a record without a commit is not a record"
        raise RecordError(msg)
    if dirty:
        listed = ", ".join(dirty[:10])
        msg = f"the tree is not clean, so a record would not match its commit: {listed}"
        raise RecordError(msg)


# ------------------------------------------------------------- the write stamp


def _blas() -> str | None:
    """Return the BLAS numpy is linked against, through threadpoolctl."""
    pools = [
        pool
        for pool in threadpoolctl.threadpool_info()
        if pool.get("user_api") == "blas"
    ]
    if not pools:
        return None
    first = pools[0]
    return f"{first.get('internal_api')} {first.get('version')}"


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:  # pragma: no cover - all are dependencies
        return None


def machine_id() -> str:
    """Return a filesystem-safe machine identifier: operating system and architecture.

    The host name is deliberately left out, so committed records name no machine.
    """
    system = platform.system().lower() or "unknown"
    return re.sub(r"[^a-z0-9]+", "-", f"{system}-{platform.machine().lower()}").strip(
        "-"
    )


def _write_stamp(backend: str | None) -> dict[str, Any]:
    """Describe the run that produced a row: commit, machine and the pools it used."""
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "machine_id": machine_id(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "backend": backend,
        "blas": _blas(),
        "packages": {name: _version(name) for name in ("numpy", "scipy", "numba")},
        "fftw_planner": os.environ.get("SOLPROP_FFTW_PLAN"),
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


# -------------------------------------------------------------- reading a file


def path_of(study: str) -> Path:
    """Return the file a study's record lives in."""
    return RECORDS / f"{study}.json"


def relative(path: Path) -> Path:
    """Return a path relative to the repository, or the path itself if it is outside."""
    try:
        return path.relative_to(REPO)
    except ValueError:
        return path


def load(study: str) -> dict[str, Any]:
    """Return a stored record, raising when it does not exist."""
    path = path_of(study)
    if not path.exists():
        msg = (
            f"no convergence record at {relative(path)}. Write it with "
            f"{WRITE_ENV}=1 -- docs/dev/accuracy.md section 9"
        )
        raise RecordError(msg)
    stored = json.loads(path.read_text())
    keys = [_key(row) for row in stored["rows"]]
    if len(set(keys)) != len(keys):
        msg = f"{path.name} has duplicate (series, axis) rows: {keys}"
        raise RecordError(msg)
    return stored


def studies() -> list[str]:
    """Return every stored study, by name."""
    return sorted(path.stem for path in RECORDS.glob("*.json"))


# ------------------------------------------------------------------ the merge


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row["series"]), f"{row['axis']!r}")


def _sort_key(row: dict[str, Any]) -> tuple[str, int, Any]:
    axis = row["axis"]
    return (str(row["series"]), 0 if isinstance(axis, int) else 1, axis)


def _validate(rows: list[dict[str, Any]], columns: tuple[Column, ...]) -> None:
    """Reject a row set that could not be merged, compared or plotted."""
    if not rows:
        msg = "a record call with no rows would pass vacuously"
        raise RecordError(msg)
    names = {column.name for column in columns}
    for row in rows:
        if not isinstance(row.get("axis"), int | str):
            msg = (
                f"axis {row.get('axis')!r} is not an int or a str; a float axis makes "
                "two rows that differ in the last bit rather than one"
            )
            raise RecordError(msg)
        metrics = set(row) - {"series", "axis"}
        if metrics != names:
            msg = f"row {_key(row)} carries {sorted(metrics)}, not {sorted(names)}"
            raise RecordError(msg)
        for name in metrics:
            if row[name] is None:
                continue
            value = float(row[name])
            if not math.isfinite(value):
                msg = f"row {_key(row)} has a non-finite {name}: {value}"
                raise RecordError(msg)


def _merged(
    stored: dict[str, Any] | None,
    skeleton: dict[str, Any],
    rows: list[dict[str, Any]],
    stamp: dict[str, Any],
) -> dict[str, Any]:
    """Return the record with `rows` merged in, each carrying this run's stamp."""
    record = dict(stored) if stored else skeleton
    carried = ("schema", "study", "title", "case", "axis", "luna")
    record |= {key: skeleton[key] for key in carried}
    record["columns"] = skeleton["columns"]
    writes: list[dict[str, Any]] = list(record.get("writes", []))
    if stamp in writes:
        index = writes.index(stamp)
    else:
        writes.append(stamp)
        index = len(writes) - 1
    fresh = {_key(row): dict(row) | {"write": index} for row in rows}
    kept = [row for row in record.get("rows", []) if _key(row) not in fresh]
    record["rows"] = sorted([*kept, *fresh.values()], key=_sort_key)
    used = sorted({int(row["write"]) for row in record["rows"]})
    record["writes"] = [writes[position] for position in used]
    remap = {old: new for new, old in enumerate(used)}
    for row in record["rows"]:
        row["write"] = remap[int(row["write"])]
    return record


def _compare(
    stored: dict[str, Any],
    rows: list[dict[str, Any]],
    columns: tuple[Column, ...],
) -> None:
    """Assert every fresh row equals its stored one, column by column."""
    by_key = {_key(row): row for row in stored["rows"]}
    for row in rows:
        key = _key(row)
        if key not in by_key:
            msg = (
                f"{stored['study']}: no stored row for {key}. The study moved; "
                f"regenerate with {WRITE_ENV}=1 -- docs/dev/accuracy.md section 9"
            )
            raise RecordError(msg)
        was = by_key[key]
        for column in columns:
            if column.rtol is None:
                continue
            fresh, kept = row[column.name], was[column.name]
            if (fresh is None) != (kept is None):
                msg = (
                    f"{stored['study']} {key} {column.name}: {fresh} against the "
                    f"stored {kept}; a rung gained or lost the quantity"
                )
                raise RecordError(msg)
            if fresh is None:
                continue
            fresh, kept = float(fresh), float(kept)
            if not math.isclose(fresh, kept, rel_tol=column.rtol, abs_tol=column.atol):
                msg = (
                    f"{stored['study']} {key} {column.name}: {fresh:.6g} against the "
                    f"stored {kept:.6g}, outside rtol {column.rtol:g} atol "
                    f"{column.atol:g} ({column.why})"
                )
                raise RecordError(msg)


def _same_shape(stored: dict[str, Any], skeleton: dict[str, Any]) -> None:
    """Assert the study still describes the case and the columns it was recorded for."""
    for field in ("study", "title", "case", "axis", "columns", "luna"):
        if stored.get(field) != skeleton[field]:
            msg = (
                f"{skeleton['study']}: the stored {field} is not the one this study "
                f"now describes. Regenerate with {WRITE_ENV}=1"
            )
            raise RecordError(msg)


# ------------------------------------------------------------------ the caller


def record(
    study: Study,
    *,
    request: Any,
    rows: list[dict[str, Any]],
    backend: str | None = None,
) -> dict[str, Any]:
    """Write this study's rows, or compare them with the stored record.

    Parameters
    ----------
    study
        The study these rows belong to. The file is
        ``tests/data/convergence/<study.name>.json``.
    request
        The pytest ``request`` fixture. Its original collected ID, without xdist's
        scheduling suffix, is stamped on every row written, so a renamed or deleted
        writer is caught by the closure test rather
        than leaving behind a row nothing produces any more.
    rows
        ``{"series": str, "axis": int | str, <metric>: float}``, one per rung.
    backend
        The backend the rows were measured on, recorded with the run's pools.

    Returns
    -------
    dict
        The stored record, so a caller may read back rows it did not measure itself.
    """
    _validate(rows, study.columns)
    skeleton = study.skeleton()
    stamped = [dict(row) | {"series": str(row["series"])} for row in rows]
    if not writing():
        stored = load(study.name)
        _same_shape(stored, skeleton)
        _compare(stored, stamped, study.columns)
        return stored

    _require_writable()
    path = path_of(study.name)
    stored = json.loads(path.read_text()) if path.exists() else None
    merged = _merged(stored, skeleton, stamped, _write_stamp(backend))
    written = {_key(row) for row in stamped}
    for row in merged["rows"]:
        if _key(row) in written:
            row["written_by"] = original_nodeid(request.node)
    RECORDS.mkdir(parents=True, exist_ok=True)
    text = json.dumps(merged, indent=1, sort_keys=True, allow_nan=False)
    path.write_text(text + "\n")
    _WRITTEN.add(path)
    return merged


def written_paths() -> list[Path]:
    """Return the record files this process wrote, for the terminal summary."""
    return sorted(_WRITTEN)
