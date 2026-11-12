# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The process's CPU grant survives an OpenMP runtime's binding of the main thread."""

import importlib.util
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from solprop import _process


@pytest.fixture(autouse=True)
def _one_kind_of_core(monkeypatch):
    """Fake topologies stand alone: this machine's kinds of core do not enter them."""
    monkeypatch.setattr(_process, "top_tier_cores", lambda: None)


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
    monkeypatch.setattr(_process, "physical_cores", lambda cpus: 2)
    _affinity(monkeypatch, {2, 4})
    assert _process.allowed_cpus() == (2, 4)


def test_a_main_thread_bound_to_one_core_place_still_sees_the_grant(
    monkeypatch, granted
):
    # With OMP_PLACES=cores, libgomp binds the main thread to its first core's
    # hardware threads: two CPUs on a machine with two threads per core.
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    seen = []

    def one_core(cpus):
        seen.append(tuple(cpus))
        return 1

    monkeypatch.setattr(_process, "physical_cores", one_core)
    _affinity(monkeypatch, {0, 9})
    assert _process.allowed_cpus() == tuple(range(18))
    assert seen == [(0, 9)]


def test_a_core_place_outside_the_grant_is_believed(monkeypatch, granted):
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    monkeypatch.setattr(_process, "physical_cores", lambda cpus: 1)
    _affinity(monkeypatch, {17, 40})
    assert _process.allowed_cpus() == (17, 40)


def test_a_cpu_outside_the_grant_is_believed(monkeypatch, granted):
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    _affinity(monkeypatch, {40})
    assert _process.allowed_cpus() == (40,)


def test_without_an_affinity_call_the_usable_count_is_used(monkeypatch):
    # macOS has no sched_getaffinity.
    monkeypatch.delattr(_process.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(_process.os, "process_cpu_count", lambda: 3)
    assert _process._query() == (0, 1, 2)


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


def test_the_processs_own_quota_is_read_once(monkeypatch):
    reader = _process
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
def test_oversubscribe_is_zero_or_one(monkeypatch, value):
    monkeypatch.setenv("SOLPROP_OVERSUBSCRIBE", value)
    with pytest.raises(ValueError, match="SOLPROP_OVERSUBSCRIBE"):
        _process.oversubscribing()


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
    tmp_path, monkeypatch, cpus, expected
):
    root = _topology(tmp_path, SMT)
    monkeypatch.setattr(_process.sys, "platform", "linux")
    assert _process.physical_cores(tuple(cpus), root) == expected


def test_windows_cores_count_the_masks_holding_the_cpus():
    masks = [0b0001_0001, 0b0010_0010, 0b0100_0100, 0b1000_1000]
    assert _process.cores_in_masks(masks, range(8)) == 4
    assert _process.cores_in_masks(masks, (0, 4, 1)) == 2
    assert _process.cores_in_masks(masks, ()) == 0


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
def native_wait(monkeypatch):
    """The native package's wait-policy module, loaded without starting its core."""
    path = Path(__file__).parents[1] / "native" / "src" / "solprop_native" / "_wait.py"
    spec = importlib.util.spec_from_file_location("_native_wait", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "_native_wait", module)
    spec.loader.exec_module(module)
    return module


def test_native_threads_spin_briefly_by_default(native_wait):
    """GNU's runtime spins 10,000 times; LLVM's sleeps outside native calls."""
    environ: dict[str, str] = {}
    chosen = native_wait.wait_policy(environ, threads=6, top_tier=None)
    assert chosen == native_wait.WaitPolicy("spin", native_wait.SPIN_MILLISECONDS)
    assert environ == {
        "GOMP_SPINCOUNT": "10000",
        "KMP_BLOCKTIME": "0",
        native_wait.MARKER: "spin",
    }


def test_native_threads_sleep_at_once_on_a_team_wider_than_the_fastest_cores(
    native_wait,
):
    environ: dict[str, str] = {}
    chosen = native_wait.wait_policy(environ, threads=18, top_tier=6)
    assert chosen == native_wait.WaitPolicy("passive", 0)
    assert environ == {"OMP_WAIT_POLICY": "passive", native_wait.MARKER: "passive"}
    # Within the fastest cores the team spins.
    assert native_wait.wait_policy({}, threads=6, top_tier=6) == (
        native_wait.WaitPolicy("spin", native_wait.SPIN_MILLISECONDS)
    )


def test_a_child_process_recomputes_an_inherited_default(native_wait):
    """Inherited settings carry the marker, so a child is not held to its parent's."""
    environ: dict[str, str] = {}
    native_wait.wait_policy(environ, threads=18, top_tier=6)
    chosen = native_wait.wait_policy(environ, threads=4, top_tier=6)
    assert chosen.description == "spin"
    assert "OMP_WAIT_POLICY" not in environ


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
    chosen = native_wait.wait_policy(environ, threads=6)
    assert chosen == native_wait.WaitPolicy(f"environment: {name}={value}", 0)
    assert environ == {name: value}
    # A blank variable is unset.
    assert native_wait.wait_policy({name: " "}).description == "spin"
    # A user's value beside a stale marker is still the user's.
    environ = {name: value, native_wait.MARKER: "spin"}
    assert native_wait.wait_policy(environ).description.startswith("environment")


@pytest.mark.parametrize(("platform", "expected"), [("win32", 3), ("darwin", None)])
def test_physical_cores_ask_windows_and_trust_other_platforms(
    monkeypatch, platform, expected
):
    """Windows asks its processor table; Apple silicon has one thread per core."""
    monkeypatch.setattr(_process.sys, "platform", platform)
    monkeypatch.setattr(_process, "_windows_cores", lambda cpus: 3)
    assert _process.physical_cores((0, 1, 2, 3)) == expected


@pytest.mark.parametrize(
    ("system", "machine", "expected"),
    [
        ("win32", "ARM64", True),
        ("win32", "AMD64", False),
        ("linux", "aarch64", False),
    ],
)
def test_only_windows_on_arm_sizes_blas_after_load(
    monkeypatch, system, machine, expected
):
    monkeypatch.setattr(_process.sys, "platform", system)
    monkeypatch.setattr(_process.platform, "machine", lambda: machine)
    assert _process.sizes_blas_after_load() is expected


@pytest.mark.parametrize(
    ("value", "warned"),
    [("1", True), ("2", True), ("3", True), ("4", False), ("0", False), ("x", False)],
)
def test_a_small_load_time_blas_width_warns_on_windows_on_arm(
    monkeypatch, value, warned
):
    monkeypatch.setattr(_process, "sizes_blas_after_load", lambda: True)
    monkeypatch.delenv("GOTO_NUM_THREADS", raising=False)
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _process._warn_of_load_time_blas_width()
    assert bool(caught) is warned
    if warned:
        assert f"OMP_NUM_THREADS={value}" in str(caught[0].message)


def test_a_small_load_time_blas_width_is_fine_elsewhere(monkeypatch):
    monkeypatch.setattr(_process, "sizes_blas_after_load", lambda: False)
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _process._warn_of_load_time_blas_width()


def test_parallel_rows_covers_each_row_once_in_threads_when_it_pays(monkeypatch):
    monkeypatch.setattr(_process, "PARALLEL_BYTES", 100)
    monkeypatch.setattr(_process, "thread_budget", lambda: SimpleNamespace(threads=3))
    slices = _process.parallel_rows(10, 100, lambda part: part)
    assert slices == [slice(0, 3), slice(3, 6), slice(6, 10)]
    # Fewer rows than threads: a thread per row, and none left idle.
    assert _process.parallel_rows(2, 100, lambda part: part) == [
        slice(0, 1),
        slice(1, 2),
    ]


def test_a_small_pass_reads_no_budget(monkeypatch):
    """Below the threshold no pass threads, so the topology is not walked (#556)."""

    def unread():
        raise AssertionError("a small pass read the thread budget")

    monkeypatch.setattr(_process, "PARALLEL_BYTES", 100)
    monkeypatch.setattr(_process, "thread_budget", unread)
    assert _process.parallel_rows(10, 99, lambda part: part) == [slice(0, 10)]
    assert _process.parallel_rows(1, 100, lambda part: part) == [slice(0, 1)]


def test_parallel_rows_takes_one_call_when_threads_would_not_pay(monkeypatch):
    monkeypatch.setattr(_process, "PARALLEL_BYTES", 100)
    monkeypatch.setattr(_process, "thread_budget", lambda: SimpleNamespace(threads=3))
    assert _process.parallel_rows(10, 99, lambda part: part) == [slice(0, 10)]
    monkeypatch.setattr(_process, "thread_budget", lambda: SimpleNamespace(threads=1))
    assert _process.parallel_rows(10, 100, lambda part: part) == [slice(0, 10)]


def _perf_levels(levels, top):
    """A ``sysctl`` reader for a chip with ``levels`` kinds of core, ``top`` fastest."""
    values = {"hw.nperflevels": levels, "hw.perflevel0.physicalcpu": top}
    return values.get


@pytest.mark.parametrize(
    ("platform", "levels", "top", "expected"),
    [
        ("darwin", 2, 6, 6),  # an Apple M5 Max: 6 top-tier and 12 second-tier cores
        ("darwin", 1, 8, None),  # one kind of core: no limit
        ("darwin", None, None, None),  # an older kernel without performance levels
        ("linux", 2, 6, None),  # not read elsewhere
    ],
)
def test_top_tier_cores_read_the_fastest_performance_level(
    monkeypatch, platform, levels, top, expected
):
    monkeypatch.undo()
    monkeypatch.setattr(_process.sys, "platform", platform)
    monkeypatch.setattr(_process, "sysctl_int", _perf_levels(levels, top))
    reader = getattr(_process.top_tier_cores, "__wrapped__", _process.top_tier_cores)
    assert reader() == expected


@pytest.mark.parametrize(
    ("variable", "expected"),
    [
        (None, ("top-tier cores", 6)),
        # Asking for threads lifts the top tier too: the second tier stays reachable.
        ("12", ("SOLPROP_THREADS", 12)),
    ],
)
def test_the_budget_keeps_to_the_top_tier_unless_threads_are_asked_for(
    monkeypatch, variable, expected
):
    monkeypatch.setattr(_process, "allowed_cpus", lambda: tuple(range(18)))
    monkeypatch.setattr(_process, "physical_cores", lambda cpus: None)
    monkeypatch.setattr(_process, "cgroup_cpus", lambda: None)
    monkeypatch.setattr(_process, "top_tier_cores", lambda: 6)
    if variable is None:
        monkeypatch.delenv("SOLPROP_THREADS", raising=False)
    else:
        monkeypatch.setenv("SOLPROP_THREADS", variable)
    budget = _process.thread_budget()
    assert (budget.source, budget.threads) == expected


def test_a_child_takes_the_blas_idle_defaults_unless_it_has_its_own():
    assert _process.child_environment({"A": "1"}) == {
        "A": "1",
        **_process.BLAS_IDLE_DEFAULTS,
    }
    own = dict.fromkeys(_process.BLAS_IDLE_DEFAULTS, "28")
    assert _process.child_environment(own) == own


def test_a_spawn_environment_is_the_child_s_and_the_parent_s_comes_back(monkeypatch):
    monkeypatch.setenv("KEPT", "parent")
    monkeypatch.setenv("GONE", "parent")
    monkeypatch.delenv("ADDED", raising=False)
    for name in _process.BLAS_IDLE_DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    before = dict(os.environ)
    with _process.spawn_environment({"ADDED": "child", "KEPT": "child"}, ["GONE"]):
        assert os.environ["ADDED"] == os.environ["KEPT"] == "child"
        assert "GONE" not in os.environ
        for name, value in _process.BLAS_IDLE_DEFAULTS.items():
            assert os.environ[name] == value
    assert dict(os.environ) == before
    # Restored when the start fails too.
    with pytest.raises(RuntimeError), _process.spawn_environment({"ADDED": "child"}):
        raise RuntimeError
    assert dict(os.environ) == before


# --------------------------------------------------------------------- process start


def test_a_linux_stamp_counts_fields_past_a_name_with_parentheses(
    tmp_path, monkeypatch
):
    path = tmp_path / "stat"
    path.write_text(
        "123 (command (with) brackets) " + " ".join(str(i) for i in range(30)),
        encoding="utf-8",
    )
    monkeypatch.setattr(_process, "Path", lambda name: path)
    assert _process._proc_stamp(123) == "19"


@pytest.mark.parametrize("error", [FileNotFoundError, ProcessLookupError])
def test_a_process_that_exits_while_its_stamp_is_read_has_none(monkeypatch, error):
    """The process is gone by the time `/proc/<pid>/stat` is read."""

    class Vanishing:
        def read_text(self, encoding):
            raise error(2, "No such file or directory")

    monkeypatch.setattr(_process, "Path", lambda name: Vanishing())
    assert _process._proc_stamp(123) is None


def test_a_stamp_read_that_fails_otherwise_raises(monkeypatch):
    # Only a vanished process means no identity; a refused read is an error.
    class Refused:
        def read_text(self, encoding):
            raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(_process, "Path", lambda name: Refused())
    with pytest.raises(PermissionError):
        _process._proc_stamp(123)


@pytest.mark.parametrize(
    ("printed", "stamp"),
    [("Tue Sep 23 13:44:28 2026\n", "Tue Sep 23 13:44:28 2026"), ("", None)],
)
def test_a_stamp_without_proc_is_ps_s_start_time(monkeypatch, printed, stamp):
    """Off Linux (macOS) the stamp is `ps`'s start time, or None if it has none."""
    calls = []

    def ps(arguments, **options):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, stdout=printed, stderr="")

    monkeypatch.setattr(subprocess, "run", ps)
    assert _process._ps_stamp(123) == stamp
    assert calls == [["ps", "-p", "123", "-o", "lstart="]]


@pytest.mark.skipif(sys.platform == "win32", reason="the stamp of a POSIX process")
def test_this_process_has_a_stamp_and_a_missing_one_has_none():
    assert _process.process_stamp(os.getpid())
    assert _process.process_stamp(2**22 + 7) is None
