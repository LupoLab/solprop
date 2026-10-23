# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The process's CPU grant survives an OpenMP runtime's binding of the main thread."""

import pytest

from solprop import _process


@pytest.fixture
def granted(monkeypatch):
    """A process granted CPUs 0-17 at import."""
    monkeypatch.setattr(_process, "_AT_IMPORT", tuple(range(18)))


def _affinity(monkeypatch, cpus):
    monkeypatch.setattr(
        _process.os, "sched_getaffinity", lambda pid: set(cpus), raising=False
    )


@pytest.mark.parametrize("binding", ["close", "spread", "TRUE"])
def test_a_main_thread_bound_by_openmp_still_sees_the_grant(
    monkeypatch, granted, binding
):
    # libgomp with OMP_PROC_BIND binds the main thread to its first place.
    monkeypatch.setenv("OMP_PROC_BIND", binding)
    _affinity(monkeypatch, {0})
    assert _process.allowed_cpus() == tuple(range(18))


@pytest.mark.parametrize("binding", [None, "false"])
def test_without_a_binding_the_current_affinity_is_believed(
    monkeypatch, granted, binding
):
    if binding is None:
        monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    else:
        monkeypatch.setenv("OMP_PROC_BIND", binding)
    _affinity(monkeypatch, {3})
    assert _process.allowed_cpus() == (3,)


def test_a_deliberate_narrowing_to_several_cpus_is_believed(monkeypatch, granted):
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    _affinity(monkeypatch, {2, 4})
    assert _process.allowed_cpus() == (2, 4)


def test_a_cpu_outside_the_grant_is_believed(monkeypatch, granted):
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    _affinity(monkeypatch, {40})
    assert _process.allowed_cpus() == (40,)


def test_without_an_affinity_call_the_usable_count_is_used(monkeypatch):
    # macOS has no sched_getaffinity.
    monkeypatch.delattr(_process.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(_process.os, "process_cpu_count", lambda: 3)
    assert _process._query() == (0, 1, 2)


def test_a_pool_started_from_a_bound_thread_gets_the_grant(monkeypatch, granted):
    # Workers inherit the creating thread's affinity: the binding is lifted while a
    # pool starts and put back after.
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    current = {0}
    calls = []

    def set_affinity(pid, cpus):
        calls.append(tuple(sorted(cpus)))
        current.clear()
        current.update(cpus)

    monkeypatch.setattr(
        _process.os, "sched_getaffinity", lambda pid: set(current), raising=False
    )
    monkeypatch.setattr(_process.os, "sched_setaffinity", set_affinity, raising=False)
    with _process.unbound():
        inside = tuple(sorted(current))
    assert inside == tuple(range(18))
    assert calls == [tuple(range(18)), (0,)]
    assert current == {0}


def test_an_unbound_thread_is_left_alone(monkeypatch, granted):
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    _affinity(monkeypatch, {0})

    def refuse(pid, cpus):
        raise AssertionError("an unbound thread's affinity was changed")

    monkeypatch.setattr(_process.os, "sched_setaffinity", refuse, raising=False)
    with _process.unbound():
        pass


def test_a_grant_recorded_before_native_bound_the_main_thread_is_taken(monkeypatch):
    # solprop_native imported first records the process's CPUs before its OpenMP
    # runtime binds the main thread to one of them.
    monkeypatch.setitem(
        _process.sys.modules,
        "solprop_native",
        type("Kokkos", (), {"PROCESS_CPUS": tuple(range(18))}),
    )
    _affinity(monkeypatch, {0})
    assert _process._at_import() == tuple(range(18))


@pytest.mark.parametrize("recorded", [(), (5, 6), (1, 2, 3)])
def test_an_earlier_record_that_is_not_a_wider_grant_is_ignored(monkeypatch, recorded):
    monkeypatch.setitem(
        _process.sys.modules,
        "solprop_native",
        type("Kokkos", (), {"PROCESS_CPUS": recorded}),
    )
    _affinity(monkeypatch, {0, 1})
    assert _process._at_import() == (0, 1)


def test_numba_is_sized_from_the_grant_after_an_early_binding(monkeypatch):
    monkeypatch.setattr(_process, "_AT_IMPORT", tuple(range(18)))
    monkeypatch.delitem(_process.sys.modules, "numba", raising=False)
    # Recorded first, so that undo removes what the call writes.
    monkeypatch.setenv("NUMBA_NUM_THREADS", "1")
    monkeypatch.delenv("NUMBA_NUM_THREADS")
    _affinity(monkeypatch, {0})
    _process._size_numba()
    assert _process.os.environ["NUMBA_NUM_THREADS"] == "18"


@pytest.mark.parametrize("given", ["numba", "environment", "unbound"])
def test_numba_is_left_alone_otherwise(monkeypatch, given):
    monkeypatch.setattr(_process, "_AT_IMPORT", tuple(range(18)))
    monkeypatch.setenv("NUMBA_NUM_THREADS", "5")
    if given != "environment":
        monkeypatch.delenv("NUMBA_NUM_THREADS")
    if given == "numba":
        monkeypatch.setitem(_process.sys.modules, "numba", object())
    _affinity(monkeypatch, set(range(18)) if given == "unbound" else {0})
    _process._size_numba()
    expected = "5" if given == "environment" else None
    assert _process.os.environ.get("NUMBA_NUM_THREADS") == expected


def test_a_collapsed_main_thread_is_released_for_good(monkeypatch, granted):
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    _affinity(monkeypatch, {0})
    calls = []
    monkeypatch.setattr(
        _process.os,
        "sched_setaffinity",
        lambda pid, cpus: calls.append(cpus),
        raising=False,
    )
    assert _process.release()
    assert calls == [tuple(range(18))]


def test_an_uncollapsed_thread_is_left_alone(monkeypatch, granted):
    monkeypatch.delenv("OMP_PROC_BIND", raising=False)
    _affinity(monkeypatch, set(range(18)))
    monkeypatch.setattr(
        _process.os,
        "sched_setaffinity",
        lambda pid, cpus: pytest.fail("moved"),
        raising=False,
    )
    assert not _process.release()
