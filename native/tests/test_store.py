# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The per-user calibration store (solprop SEP 46 sections 5 and 6.8)."""

from __future__ import annotations

import json
import multiprocessing
import multiprocessing.synchronize
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from solprop_native import _store
from solprop_native._store import (
    STALE_CLAIM_SECONDS,
    Generation,
    Row,
    Store,
    StoreWarning,
    cache_directory,
    lookup,
    mode,
    search_path,
)

ROW = Row(
    allocation={"cores": 4},
    level="standard",
    process={"threads": 4, "wait": "spin"},
    workspace={"stage_threshold": 2048},
    fitted_under={"threads": 4, "wait": "spin"},
    evidence={"geomean": 0.93},
)


def _generation(name: str = "g1", fingerprint: str = "abc") -> Generation:
    return Generation(
        fingerprint=fingerprint,
        name=name,
        created="2026-11-08T12:00:00Z",
        rows=(ROW,),
        wisdom={"rows:c2c:4096": "(fftw-3.3.11 fftw_wisdom)"},
    )


@pytest.mark.parametrize(
    ("system", "environ", "expected"),
    [
        ("Linux", {"HOME": "/home/u"}, "/home/u/.cache/solprop"),
        ("Linux", {"HOME": "/home/u", "XDG_CACHE_HOME": "/x"}, "/x/solprop"),
        ("Darwin", {"HOME": "/Users/u"}, "/Users/u/Library/Caches/solprop"),
        ("Windows", {"HOME": "/h", "LOCALAPPDATA": "/L"}, "/L/solprop/Cache"),
        ("Windows", {"HOME": "/h"}, "/h/AppData/Local/solprop/Cache"),
        ("Linux", {"HOME": "/h", "SOLPROP_CACHE_DIR": "/c"}, "/c"),
        ("Darwin", {"HOME": "/h", "SOLPROP_CACHE_DIR": "/c"}, "/c"),
    ],
)
def test_the_store_lives_in_the_platforms_cache_directory(
    system: str, environ: dict[str, str], expected: str
) -> None:
    assert cache_directory(environ, system) == Path(expected)


def test_the_mode_and_search_path_are_read_from_the_environment() -> None:
    assert mode({}) == "auto"
    assert mode({"SOLPROP_CALIBRATION": " Use "}) == "use"
    assert mode({"SOLPROP_CALIBRATION": "off"}) == "off"
    with pytest.raises(ValueError, match="auto, use, off"):
        mode({"SOLPROP_CALIBRATION": "always"})
    joined = os.pathsep.join(["/a", "", "/b"])
    assert search_path({"SOLPROP_CALIBRATION_PATH": joined}) == (Path("/a"), Path("/b"))
    assert search_path({}) == ()


def test_a_published_generation_is_read_back_whole(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert store.current("abc") is None
    path = store.publish(_generation())
    read = store.current("abc")
    assert read is not None
    assert read.source == path
    assert read.rows == (ROW,)
    assert read.wisdom == {"rows:c2c:4096": "(fftw-3.3.11 fftw_wisdom)"}
    # A later generation replaces it; the earlier file stays until cleared.
    store.publish(_generation("g2"))
    newer = store.current("abc")
    assert newer is not None
    assert newer.name == "g2"
    assert path.exists()


def test_a_failed_publication_leaves_the_previous_generation_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path)
    store.publish(_generation())
    original = Path.replace

    def refuse(self: Path, target: Path) -> Path:
        if Path(target).name == "current":
            raise OSError("disk full")
        return original(self, target)

    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(OSError, match="disk full"):
        store.publish(_generation("g2"))
    monkeypatch.undo()
    current = store.current("abc")
    assert current is not None
    assert current.name == "g1"
    # No temporary file is left behind.
    assert not [p for p in store.directory("abc").iterdir() if p.name.startswith(".")]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("{not json", "ignoring the calibration profile"),
        (json.dumps({"format": 99}), "generation format 99"),
        (
            json.dumps(_generation().record() | {"fingerprint": "other"}),
            "names another machine",
        ),
        (
            json.dumps(_generation().record() | {"rows": [{"level": "never"}]}),
            "a row's level",
        ),
    ],
)
def test_a_corrupt_or_foreign_generation_is_ignored_with_a_warning(
    tmp_path: Path, contents: str, message: str
) -> None:
    store = Store(tmp_path)
    store.publish(_generation())
    (store.directory("abc") / "g1.json").write_text(contents, encoding="utf-8")
    _store._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    with pytest.warns(StoreWarning, match=message):
        assert store.current("abc") is None


def test_a_current_naming_a_missing_generation_is_ignored(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.directory("abc").mkdir(parents=True)
    (store.directory("abc") / "current").write_text("gone\n", encoding="utf-8")
    _store._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    with pytest.warns(StoreWarning, match="gone.json"):
        assert store.current("abc") is None


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"format": 1, "fingerprint": 1, "name": "g", "created": "t", "rows": []},
        {"format": 1, "fingerprint": "f", "name": "g", "created": "t", "rows": {}},
        {
            "format": 1,
            "fingerprint": "f",
            "name": "g",
            "created": "t",
            "rows": [],
            "wisdom": {"k": 1},
        },
        {1: 2},
    ],
)
def test_a_malformed_generation_is_refused(data: object) -> None:
    with pytest.raises(ValueError, match=r"generation|mapping|wisdom|rows"):
        Generation.read(data)


def test_clearing_removes_generations_but_not_a_live_claim(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.publish(_generation())
    store.publish(_generation("g2", fingerprint="def"))
    claim = store.claim("abc", "cls", time.time() + 60)
    assert claim is not None
    assert store.clear("abc") == 2
    assert store.current("abc") is None
    assert store.current("def") is not None
    assert claim.path.exists()
    assert store.clear() == 2
    assert store.current("def") is None


def test_clearing_forgets_the_machines_refinements(tmp_path: Path) -> None:
    # A refinement fitted over the built-in table would otherwise still apply.
    store = Store(tmp_path)
    store.publish(_generation())
    store.publish_refinement("abc", "d1", {"base": {}, "record": {}})
    store.publish_refinement("def", "d2", {"base": {}, "record": {}})
    assert store.clear("abc") == 3  # its current, its generation and its refinement
    assert store.refinement("abc", "d1") is None
    assert store.refinement("def", "d2") is not None
    assert store.clear() == 1
    assert store.refinement("def", "d2") is None


def test_a_claim_is_exclusive_until_released(tmp_path: Path) -> None:
    store = Store(tmp_path)
    first = store.claim("abc", "cls", time.time() + 60)
    assert first is not None
    assert store.claim("abc", "cls", time.time() + 60) is None
    # Another allocation class is a separate claim.
    other = store.claim("abc", "other", time.time() + 60)
    assert other is not None
    holder = store.holder("abc", "cls")
    assert holder is not None
    assert (holder.host, holder.pid) == (socket.gethostname(), os.getpid())
    first.release()
    assert store.holder("abc", "cls") is None
    assert store.claim("abc", "cls", time.time() + 60) is not None
    other.release()


def test_a_claim_past_its_deadline_or_whose_owner_died_is_stale(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path)
    path = store.directory("abc") / "claim-cls"
    store.directory("abc").mkdir(parents=True)
    overdue = time.time() - STALE_CLAIM_SECONDS - 1
    path.write_text(
        json.dumps({"host": "elsewhere", "pid": 1, "deadline": overdue}),
        encoding="utf-8",
    )
    assert store.claim("abc", "cls", time.time() + 60) is not None
    # A process that has exited on this host holds nothing.
    gone = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
        check=True,
    )
    path.write_text(
        json.dumps(
            {
                "host": socket.gethostname(),
                "pid": int(gone.stdout),
                "deadline": time.time() + 600,
            }
        ),
        encoding="utf-8",
    )
    assert store.holder("abc", "cls") is None
    # Another host's live claim is respected until its deadline.
    path.write_text(
        json.dumps({"host": "elsewhere", "pid": 1, "deadline": time.time() + 600}),
        encoding="utf-8",
    )
    assert store.claim("abc", "cls", time.time() + 60) is None
    # A claim still being written is fresh; an old unreadable one is stale.
    path.write_text("", encoding="utf-8")
    assert store.claim("abc", "cls", time.time() + 60) is None
    old = time.time() - 2 * STALE_CLAIM_SECONDS
    os.utime(path, (old, old))
    assert store.claim("abc", "cls", time.time() + 60) is not None


def _race(
    root: str,
    start: multiprocessing.synchronize.Event,
    results: multiprocessing.Queue[bool],
    done: multiprocessing.synchronize.Event,
) -> None:
    start.wait(60)
    results.put(Store(Path(root)).claim("abc", "cls", time.time() + 60) is not None)
    # Hold on until every process has tried: a winner that exits leaves a stale claim.
    done.wait(60)


def test_concurrent_claims_admit_exactly_one(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start, done = context.Event(), context.Event()
    results: multiprocessing.Queue[bool] = context.Queue()
    workers = [
        context.Process(target=_race, args=(str(tmp_path), start, results, done))
        for _ in range(6)
    ]
    for worker in workers:
        worker.start()
    start.set()
    won = [results.get(timeout=60) for _ in workers]
    done.set()
    for worker in workers:
        worker.join(timeout=60)
    assert won.count(True) == 1


def test_inconclusive_attempts_are_remembered(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert store.inconclusive_since("abc", "cls") is None
    store.record_inconclusive("abc", "cls", when=1000.0)
    store.record_inconclusive("abc", "other", when=2000.0)
    assert store.inconclusive_since("abc", "cls") == 1000.0
    assert store.inconclusive_since("abc", "other") == 2000.0
    (store.directory("abc") / "attempts.json").write_text("[]", encoding="utf-8")
    assert store.inconclusive_since("abc", "cls") is None
    store.record_inconclusive("abc", "cls", when=3000.0)
    assert store.inconclusive_since("abc", "cls") == 3000.0


def test_lookup_reads_the_users_store_then_read_only_directories(
    tmp_path: Path,
) -> None:
    mine, shared = Store(tmp_path / "mine"), Store(tmp_path / "shared")
    shared.publish(_generation("shared"))
    found = lookup("abc", [mine, shared])
    assert found is not None
    assert found.name == "shared"
    mine.publish(_generation("mine"))
    found = lookup("abc", [mine, shared])
    assert found is not None
    assert found.name == "mine"
    assert lookup("missing", [mine, shared]) is None
    environ = {
        "SOLPROP_CACHE_DIR": str(tmp_path / "mine"),
        "SOLPROP_CALIBRATION_PATH": str(tmp_path / "shared"),
    }
    assert [store.root for store in _store.stores(environ)] == [
        tmp_path / "mine",
        tmp_path / "shared",
    ]


def test_generation_names_are_unique_and_sortable() -> None:
    names = {_store.generation_name() for _ in range(20)}
    assert len(names) == 20
    assert all(name.startswith("g20") for name in names)


def test_refinements_are_kept_by_descriptor(tmp_path: Path) -> None:
    mine, shared = Store(tmp_path / "mine"), Store(tmp_path / "shared")
    assert mine.refinement("abc", "d1") is None
    path = shared.publish_refinement("abc", "d1", {"workspace": {"medium_threads": 2}})
    assert path.name == "d1.json"
    found = _store.refinement("abc", "d1", [mine, shared])
    assert found is not None
    assert found["workspace"] == {"medium_threads": 2}
    assert found["descriptor"] == "d1"
    assert _store.refinement("abc", "d2", [mine, shared]) is None
    # A file naming another descriptor, or unreadable, is skipped with a warning.
    path.write_text(json.dumps({"descriptor": "other"}), encoding="utf-8")
    _store._WARNED.clear()  # pyright: ignore[reportPrivateUsage]
    with pytest.warns(StoreWarning, match="another descriptor"):
        assert shared.refinement("abc", "d1") is None
    path.write_text("{", encoding="utf-8")
    with pytest.warns(StoreWarning, match="ignoring the refinement"):
        assert shared.refinement("abc", "d1") is None
