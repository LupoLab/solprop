# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Test session configuration.

Shape and dtype annotations are checked at runtime during tests only; production imports
are unaffected. The jaxtyping hook is installed for the whole session rather than as a
context manager, so every ``solprop`` module imported later, by a test module or
transitively, is instrumented with beartype; nothing has to be listed here. Under xdist
each worker is its own process and loads this file before it loads ``solprop``, so every
worker installs its own hook (``docs/dev/testing.md`` section 2).
"""

import importlib
import importlib.util
import os
import sys

import pytest
from hypothesis import settings
from jaxtyping import install_import_hook

from runtime_types import TYPECHECKED

# Kept as a module attribute so the hook lives for the whole session.
# beartype is configured with the PEP 484 numeric tower, so an ``int`` is accepted
# where a ``float`` is annotated and a ``float`` where a ``complex`` is, as static
# checkers assume. `SOLPROP_TEST_TYPECHECK=0` leaves the hook out (runtime_types.py,
# ADR-053): CI's coverage lane checks, its other lanes do not.
JAXTYPING_HOOK = (
    install_import_hook(
        "solprop",
        "beartype.beartype(conf=beartype.BeartypeConf(is_pep484_tower=True))",
    )
    if TYPECHECKED
    else None
)

# Hypothesis's per-example deadline is a wall-clock limit, which a parallel lane on a
# shared node exceeds without anything being wrong: the first example of a property
# also pays the import and JIT costs. Examples are bounded by their count instead.
settings.register_profile("solprop", deadline=None)
settings.load_profile("solprop")

# Any driver test can select the installed backend, even without collecting
# tests/backends. On macOS Kokkos must load before pyFFTW's private OpenMP runtime.
# A present but broken wheel fails here; only an absent optional wheel is omitted.
if importlib.util.find_spec("solprop_native") is not None:
    importlib.import_module("solprop_native")

import convergence_records  # noqa: E402 - `tests/` reaches sys.path with this file
import shards  # noqa: E402 - test helper beside conftest
import solprop  # noqa: E402, F401 - imported after the hook so the package is checked too
from distribution import remember_nodeid  # noqa: E402 - test helper beside conftest
from solprop.backends import registry  # noqa: E402 - needs the hook above

# The lanes without the native package run on `backend="auto"`, and its fallback
# warning fires once per process: under `filterwarnings = ["error"]` it would fail
# whichever test came first on each worker. The warning's own tests reset this flag.
registry._warned_native_fallback = True

# ------------------------------------------------------------------------- the tiers
#
# Which tier a test is in is a property of the collected item, so it is recorded from
# the item rather than re-derived by reading the source. ``tryfirst`` is what makes the
# record complete: pytest's own mark plugin deselects by ``-m`` inside this same hook,
# so a hook running after it would see only the lane being run, and the check in
# ``tests/test_tiers.py`` would quietly stop covering the tests that lane excludes.
# The same ordering places default worker groups before xdist's non-wrapper hook
# appends group suffixes. Tier and convergence-record identities stay unsuffixed.
TIER_MARKERS = ("medium", "slow", "gpu", "benchmark")

_COLLECTED_TIERS: dict[str, tuple[str, ...]] = {}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Record original tier identities, then assign groups before xdist names them."""
    _COLLECTED_TIERS.clear()
    for item in items:
        remember_nodeid(item)
        _COLLECTED_TIERS[item.nodeid] = tuple(
            name for name in TIER_MARKERS if item.get_closest_marker(name) is not None
        )
    # Workers reset ``dist`` to "no" and retain the selected strategy in
    # ``loadgroup``; checking only the controller option would split fixtures.
    if config.getoption("loadgroup", default=False) or (
        config.getoption("dist", default="no") == "loadgroup"
    ):
        _group_items_by_module(items)


SHARED_SCOPES = ("module", "class", "package")
"""Fixture scopes built once for several tests, which a worker group keeps together."""


def _group_items_by_module(items):
    """Keep shared fixtures together; spread tests that share none.

    A test that uses a module-, class- or package-scoped fixture is grouped with the
    rest of its module, or, when that fixture is parametrised, with the tests of the
    same parameter, so each instance is built once on one worker. A test that shares
    none goes wherever a worker is free: grouping it by module only queued the heavy
    modules behind one worker. An explicit ``xdist_group`` is kept as declared; an item
    the collector gives no fixture information (a doctest) keeps its module's group.
    """
    for item in items:
        if item.get_closest_marker("xdist_group") is not None:
            continue
        shared = _shared_fixtures(item)
        if shared is not None and not shared:
            continue
        module = item.nodeid.split("::", 1)[0]
        callspec = getattr(item, "callspec", None)
        instances = sorted(
            f"{name}={callspec.indices[name]}"
            for name in shared or ()
            if callspec is not None and name in callspec.indices
        )
        suffix = "" if not instances else ":" + ",".join(instances)
        item.add_marker(pytest.mark.xdist_group(name=f"module:{module}{suffix}"))


def _shared_fixtures(item):
    """Return the names of an item's shared-scope fixtures, or ``None`` if unknown.

    Parameters
    ----------
    item : pytest.Item
        The collected test.

    Returns
    -------
    list of str or None
        The fixtures of :data:`SHARED_SCOPES` it requests, directly or through others.
    """
    info = getattr(item, "_fixtureinfo", None)
    if info is None:
        return None
    return [
        name
        for name, definitions in info.name2fixturedefs.items()
        if definitions and definitions[-1].scope in SHARED_SCOPES
    ]


# ------------------------------------------------------------------------ the shards
#
# CI splits a lane across several jobs: `SOLPROP_TEST_SHARD=i/n` keeps the modules of
# shard i of n and deselects the rest. Whole modules move, so a module fixture is built
# in one job only. The modules are dealt longest first to the shard with the least work
# so far, by the seconds in tests/data/shard_weights.json (`tests/shards.py` writes it
# from JUnit reports); a module missing from it counts as the median module. Every
# worker of every shard deals the same collection the same way, which is what lets
# xdist's workers agree on it.
SHARD_VARIABLE = "SOLPROP_TEST_SHARD"


@pytest.hookimpl(trylast=True, specname="pytest_collection_modifyitems")
def pytest_collection_modifyitems_shards(config, items):
    """Deselect the modules another shard runs, when ``SOLPROP_TEST_SHARD`` is set."""
    value = os.environ.get(SHARD_VARIABLE, "").strip()
    if not value:
        return
    index, count = shards.parse(value)
    keep = shards.deal(items, count)[index]
    kept = [item for item in items if shards.module_of(item.nodeid) in keep]
    dropped = [item for item in items if shards.module_of(item.nodeid) not in keep]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
    items[:] = kept


def pytest_collection_finish(session):
    """Undo the session namespace's import side effects before any item runs.

    ``--doctest-modules`` imports every module under ``src`` at collection, and
    ``solprop.sol`` configures logging and a warning filter when it is imported: that is
    its purpose. The suite tests the library's defaults, so they are restored here with
    the module's own public ``end_session``; nothing in the library knows about the test
    runner. A hook rather than a fixture, because a fixture of this file would not reach
    the doctests collected from ``src``. ``tests/test_sol.py`` exercises the side
    effects themselves, in subprocesses.
    """
    module = sys.modules.get("solprop.sol")
    if module is not None:
        module.end_session()


@pytest.fixture(scope="session")
def collected_tiers():
    """Return ``nodeid -> tier markers`` for every item collected this session."""
    return dict(_COLLECTED_TIERS)


# ------------------------------------------------------------ recorded properties
#
# Under pytest-xdist a recorded property crosses to the controller through execnet,
# which serialises builtin types only and stops the whole session on a numpy scalar.
# Tests record what they measured, which is usually a numpy scalar, so the conversion
# lives here once rather than in every call.


def _plain(value):
    """Return `value` with numpy scalars and arrays replaced by builtin types."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


@pytest.fixture
def record_property(record_property):
    """Record a property after converting numpy values, so the lanes run under xdist."""

    def record(name, value):
        record_property(name, _plain(value))

    return record


LIFTED_THREADS = 4
"""The budget of a test that compares thread counts: the most any of them asks for."""


@pytest.fixture
def thread_budget_lifted(monkeypatch):
    """Lift the lanes' one-thread budget for a test that compares thread counts.

    The lanes set ``SOLPROP_THREADS=1`` per xdist worker (ADR-052); a test that asks for
    several threads on purpose takes this fixture, and its budget is then
    :data:`LIFTED_THREADS` (or the CPUs, where fewer), not the whole machine, so a pool
    it leaves to its automatic size does not open a thread per core beside the other
    workers.
    """
    monkeypatch.setenv("SOLPROP_THREADS", str(LIFTED_THREADS))


# ------------------------------------------------------- the convergence records
#
# A write run must never be silent: `SOLPROP_WRITE_RECORDS=1` changes files that are
# committed, so the session says which ones at the end. `pytest.warns` is not usable
# for this -- `filterwarnings = ["error"]` would turn the announcement into a failure.


def pytest_terminal_summary(terminalreporter):
    """Name every convergence record this session wrote."""
    written = convergence_records.written_paths()
    if not written:
        return
    terminalreporter.write_line("")
    terminalreporter.write_line(f"convergence records written ({len(written)}):")
    for path in written:
        terminalreporter.write_line(f"  {convergence_records.relative(path)}")
