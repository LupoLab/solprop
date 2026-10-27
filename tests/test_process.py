# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The process's CPU grant survives an OpenMP runtime's binding of the main thread."""

import importlib.util
import sys
from pathlib import Path

import pytest

from solprop import _process

NATIVE_BUDGET = (
    Path(__file__).parents[1] / "native" / "src" / "solprop_native" / "_budget.py"
)


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


def _cgroup(root, line, groups):
    """Write ``/proc/self/cgroup`` and each group's control files under ``root``."""
    (root / "proc/self").mkdir(parents=True)
    (root / "proc/self/cgroup").write_text(line + "\n")
    for directory, files in groups.items():
        group = root / "sys/fs/cgroup" / directory
        group.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            (group / name).write_text(text)
    return root


# (``/proc/self/cgroup``, {group: {file: text}}, whole CPUs) for both readers.
CGROUP_CASES = [
    # Version 2: a parent's quota binds its child, and 1.5 CPUs is one thread.
    (
        "0::/a/b",
        {"a": {"cpu.max": "150000 100000"}, "a/b": {"cpu.max": "max 100000"}},
        1,
    ),
    ("0::/a/b", {"a/b": {"cpu.max": "400000 100000"}}, 4),
    ("0::/a", {"a": {"cpu.max": "max 100000"}}, None),
    # A container without its own namespace: the host path is absent below the
    # mount, and the mount's own group carries the quota.
    ("0::/docker/abc", {".": {"cpu.max": "250000 100000"}}, 2),
    # Version 1, with the controller's mount named for both controllers.
    (
        "4:cpu,cpuacct:/slice\n3:memory:/slice",
        {
            "cpu,cpuacct/slice": {
                "cpu.cfs_quota_us": "300000",
                "cpu.cfs_period_us": "100000",
            }
        },
        3,
    ),
    (
        "4:cpu,cpuacct:/slice",
        {
            "cpu,cpuacct/slice": {
                "cpu.cfs_quota_us": "-1",
                "cpu.cfs_period_us": "100000",
            }
        },
        None,
    ),
    # A line that is not hierarchy:controllers:path is skipped, not fatal.
    ("garbled\n0::/a", {"a": {"cpu.max": "200000 100000"}}, 2),
    # A quota below one CPU still runs one thread.
    ("0::/", {".": {"cpu.max": "50000 100000"}}, 1),
    ("0::/", {".": {"cpu.max": "garbled"}}, None),
]


@pytest.mark.parametrize(("line", "groups", "expected"), CGROUP_CASES)
def test_the_cgroup_quota_is_the_tightest_on_the_path_rounded_down(
    tmp_path, line, groups, expected
):
    assert _process.cgroup_cpus(_cgroup(tmp_path, line, groups)) == expected


def test_no_control_groups_means_no_quota(tmp_path):
    assert _process.cgroup_cpus(tmp_path) is None


@pytest.mark.parametrize(
    ("cpus", "quota", "variable", "expected"),
    [
        (8, None, None, _process.ThreadBudget(8, "affinity")),
        (8, 3, None, _process.ThreadBudget(3, "cgroup quota")),
        (8, 3, "2", _process.ThreadBudget(2, "SOLPROP_THREADS")),
        (8, None, "64", _process.ThreadBudget(8, "affinity")),
        (4, 4, "4", _process.ThreadBudget(4, "affinity")),
    ],
)
def test_the_budget_is_the_tightest_limit(monkeypatch, cpus, quota, variable, expected):
    monkeypatch.setattr(_process, "allowed_cpus", lambda: tuple(range(cpus)))
    monkeypatch.setattr(_process, "physical_cores", lambda cpus: None)
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: quota)
    if variable is None:
        monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    else:
        monkeypatch.setenv("SOLPROP_THREADS", variable)
    budget = _process.thread_budget()
    assert (budget.threads, budget.source) == (expected.threads, expected.source)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_an_invalid_solprop_threads_is_refused(monkeypatch, value):
    monkeypatch.setenv("SOLPROP_THREADS", value)
    with pytest.raises(ValueError, match="SOLPROP_THREADS"):
        _process.thread_budget()


def test_a_blank_solprop_threads_is_unset(monkeypatch):
    monkeypatch.setenv("SOLPROP_THREADS", " ")
    assert _process.thread_count("SOLPROP_THREADS") is None


@pytest.fixture
def native_budget(monkeypatch):
    """The native package's copy of the reader, loaded without starting its core."""
    spec = importlib.util.spec_from_file_location("_native_budget", NATIVE_BUDGET)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_native_budget", module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("line", "groups", "expected"), CGROUP_CASES)
def test_the_native_copy_reads_control_groups_alike(
    tmp_path, native_budget, line, groups, expected
):
    root = _cgroup(tmp_path, line, groups)
    assert native_budget.cgroup_cpus(root) == _process.cgroup_cpus(root) == expected


@pytest.mark.parametrize("cores", [None, 6])
@pytest.mark.parametrize("quota", [None, 3])
@pytest.mark.parametrize("variable", [None, "2", "64"])
@pytest.mark.parametrize("oversubscribe", [None, "0", "1"])
def test_the_native_copy_sets_the_same_budget(
    monkeypatch, native_budget, cores, quota, variable, oversubscribe
):
    """Both backends read the same budget, and clamp an explicit count alike.

    So ``thread_budget`` means the same in every backend's metadata.
    """
    monkeypatch.setattr(_process, "allowed_cpus", lambda: tuple(range(8)))
    monkeypatch.setattr(_process, "physical_cores", lambda cpus: cores)
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: quota)
    monkeypatch.setattr(native_budget, "cgroup_cpus", lambda: quota)
    monkeypatch.delenv("SOLPROP_NATIVE_THREADS", raising=False)
    for name, value in (
        ("SOLPROP_THREADS", variable),
        ("SOLPROP_OVERSUBSCRIBE", oversubscribe),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    ours, theirs = _process.thread_budget(), native_budget.thread_budget(8, cores)
    fields = ("threads", "source", "oversubscribe", "requested")
    assert [getattr(theirs, f) for f in fields] == [getattr(ours, f) for f in fields]
    for count in (1, 4, 16):
        assert theirs.explicit(count) == ours.explicit(count)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "many"])
def test_the_native_copy_refuses_the_same_counts(monkeypatch, native_budget, value):
    monkeypatch.setenv("SOLPROP_THREADS", value)
    with pytest.raises(ValueError, match="SOLPROP_THREADS") as ours:
        _process.thread_count("SOLPROP_THREADS")
    with pytest.raises(ValueError, match="SOLPROP_THREADS") as theirs:
        native_budget.thread_count("SOLPROP_THREADS")
    assert str(ours.value) == str(theirs.value)


@pytest.mark.parametrize(
    ("native", "oversubscribe", "expected"),
    [(None, None, 4), ("8", None, 4), ("2", None, 2), ("16", "1", 16)],
)
def test_the_native_count_narrows_unless_oversubscribing(
    monkeypatch, native_budget, native, oversubscribe, expected
):
    """The Kokkos pool is the native count within the budget, which stays the CPUs."""
    monkeypatch.setattr(native_budget, "cgroup_cpus", lambda: None)
    monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    for name, value in (
        ("SOLPROP_NATIVE_THREADS", native),
        ("SOLPROP_OVERSUBSCRIBE", oversubscribe),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)
    budget = native_budget.thread_budget(4)
    assert (budget.threads, budget.source) == (4, "affinity")
    assert native_budget.native_threads(budget) == expected


def test_oversubscribing_never_passes_solprop_threads(monkeypatch, native_budget):
    monkeypatch.setattr(native_budget, "cgroup_cpus", lambda: None)
    monkeypatch.setenv("SOLPROP_OVERSUBSCRIBE", "1")
    monkeypatch.setenv("SOLPROP_THREADS", "3")
    monkeypatch.setenv("SOLPROP_NATIVE_THREADS", "16")
    budget = native_budget.thread_budget(4)
    assert (budget.threads, budget.source) == (3, "SOLPROP_THREADS")
    assert native_budget.native_threads(budget) == 3
    assert (
        _process.ThreadBudget(4, "affinity", oversubscribe=True, requested=3).explicit(
            16
        )
        == 3
    )


@pytest.mark.parametrize("module", ["solprop", "native"])
def test_the_processs_own_quota_is_read_once(monkeypatch, native_budget, module):
    reader = _process if module == "solprop" else native_budget
    reads = []

    def read(root):
        reads.append(root)
        return 2

    monkeypatch.setattr(reader, "_read_cgroup_cpus", read)
    reader._own_cgroup_cpus.cache_clear()
    try:
        assert reader.cgroup_cpus() == reader.cgroup_cpus() == 2
        assert reads == [Path("/")]
    finally:
        reader._own_cgroup_cpus.cache_clear()


@pytest.mark.parametrize("value", ["yes", "2"])
def test_oversubscribe_is_zero_or_one(monkeypatch, native_budget, value):
    monkeypatch.setenv("SOLPROP_OVERSUBSCRIBE", value)
    with pytest.raises(ValueError, match="SOLPROP_OVERSUBSCRIBE"):
        _process.oversubscribing()
    with pytest.raises(ValueError, match="SOLPROP_OVERSUBSCRIBE"):
        native_budget.oversubscribing()


def _topology(root, cpus):
    """Write a sysfs CPU tree: ``cpus`` maps each CPU to its (package, core)."""
    for cpu, (package, core) in cpus.items():
        directory = root / f"cpu{cpu}" / "topology"
        directory.mkdir(parents=True)
        (directory / "physical_package_id").write_text(f"{package}\n")
        (directory / "core_id").write_text(f"{core}\n")
    return root


# Two packages of two cores, two hardware threads each: CPUs 0-3 are the first thread
# of each core and 4-7 the second, as Linux numbers them.
SMT = {cpu: (cpu % 4 // 2, cpu % 2) for cpu in range(8)}


@pytest.mark.parametrize(
    ("cpus", "expected"),
    [
        (range(8), 4),
        (range(4), 4),
        ((0, 4), 1),
        ((0, 1, 4, 5), 2),
        ((0, 2), 2),  # core 0 of each package: the same core number, two cores
        ((0, 9), None),  # a CPU with no topology: unknown, so no limit
    ],
)
def test_physical_cores_count_package_and_core_pairs(
    tmp_path, native_budget, monkeypatch, cpus, expected
):
    root = _topology(tmp_path, SMT)
    for module in (_process, native_budget):
        monkeypatch.setattr(module.sys, "platform", "linux")
        assert module.physical_cores(tuple(cpus), root) == expected


def test_windows_cores_count_the_masks_holding_the_cpus(native_budget):
    masks = [0b0001_0001, 0b0010_0010, 0b0100_0100, 0b1000_1000]
    for module in (_process, native_budget):
        assert module.cores_in_masks(masks, range(8)) == 4
        assert module.cores_in_masks(masks, (0, 4, 1)) == 2
        assert module.cores_in_masks(masks, ()) == 0


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        (None, _process.ThreadBudget(4, "physical cores")),
        # Asking for threads lifts the core limit, not the CPUs' or the quota's.
        ("6", _process.ThreadBudget(6, "SOLPROP_THREADS")),
        ("64", _process.ThreadBudget(8, "affinity")),
    ],
)
def test_the_budget_counts_physical_cores_unless_threads_are_asked_for(
    monkeypatch, variable, expected
):
    monkeypatch.setattr(_process, "allowed_cpus", lambda: tuple(range(8)))
    monkeypatch.setattr(_process, "physical_cores", lambda cpus: 4)
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: None)
    if variable is None:
        monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    else:
        monkeypatch.setenv("SOLPROP_THREADS", variable)
    budget = _process.thread_budget()
    assert (budget.threads, budget.source) == (expected.threads, expected.source)


@pytest.fixture
def native_wait():
    """The native package's wait-policy module, loaded without starting its core."""
    path = NATIVE_BUDGET.with_name("_wait.py")
    spec = importlib.util.spec_from_file_location("_native_wait", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_threads_sleep_at_once(native_wait):
    environ: dict[str, str] = {}
    assert native_wait.wait_policy(environ) == "passive"
    assert environ == {"OMP_WAIT_POLICY": "passive"}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OMP_WAIT_POLICY", "active"),
        ("GOMP_SPINCOUNT", "10000"),
        ("KMP_BLOCKTIME", "200"),
    ],
)
def test_a_wait_policy_the_user_chose_is_kept(native_wait, name, value):
    environ = {name: value}
    assert native_wait.wait_policy(environ) == f"environment: {name}={value}"
    assert environ == {name: value}
    # A blank variable is unset.
    assert native_wait.wait_policy({name: " "}) == "passive"


@pytest.mark.parametrize(("platform", "expected"), [("win32", 3), ("darwin", None)])
def test_physical_cores_ask_windows_and_trust_other_platforms(
    native_budget, monkeypatch, platform, expected
):
    """Windows asks its processor table; Apple silicon has one thread per core."""
    for module in (_process, native_budget):
        monkeypatch.setattr(module.sys, "platform", platform)
        monkeypatch.setattr(module, "_windows_cores", lambda cpus: 3)
        assert module.physical_cores((0, 1, 2, 3)) == expected
