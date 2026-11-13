# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CPUs this process may run on, as the scheduler granted them.

Linux reports affinity per thread. An OpenMP runtime started with ``OMP_PROC_BIND``
set -- GNU libgomp, which Kokkos's builds load -- binds the **main** thread to the first
of its places when it starts, so after the first parallel region
``os.sched_getaffinity(0)`` names one CPU although the process was granted many. A
thread pool sized from that answer runs on one core (measured: an 18-core allocation
gave a pool sized this way one thread). The affinity is therefore read once, when
``solprop`` is imported, before any of its runtimes start, and the later answer is
believed only when it is not that collapse.

It also says when a process started (:func:`process_stamp`), which tells a process
from a later one given its number: the scan queue's owners and the front end's runs
both need it, and this module, which imports nothing from the rest of the package,
sits below both.
"""

import contextlib
import functools
import itertools
import os
import platform
import shlex
import sys
import threading
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# NumPy's OpenBLAS on Windows on Arm crashes or hangs in small linear solves when one of
# these sizes it at load. threadpoolctl reaches the same widths safely once it's loaded.
LOAD_TIME_BLAS_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
)

BLAS_IDLE_DEFAULTS: Mapping[str, str] = {"OPENBLAS_THREAD_TIMEOUT": "20"}
"""Variables a Python process solprop starts takes, unless they are already set.

NumPy's and SciPy's pthreads OpenBLAS each start a pool of every CPU when they load,
whose threads then spin for ``2**28`` cycles, about 0.1 s, before sleeping: on a
72-CPU machine importing the driver used 13.8 CPU seconds in 0.76 s of wall time,
and 0.75 in 0.67 at ``2**20`` cycles (solprop issue #383). This changes no width, so
it is safe where a load-time width is not (:data:`LOAD_TIME_BLAS_VARIABLES`). At
``2**20`` cycles, about 0.4 ms, back-to-back products of 300 x 300 matrices ran as
fast as at the default, and 11 % slower at ``2**4``. Only a process started fresh
can take it: OpenBLAS reads it once, when it loads."""

THREAD_VARIABLES = (
    "SOLPROP_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
"""The variables that size a process's thread pools: solprop's budget, then the OpenMP
and BLAS widths a scan's worker processes are started with."""

_BLAS_LOCK = threading.RLock()
_BLAS_USERS: set[object] = set()
_BLAS_LIMITS = contextlib.ExitStack()


def child_environment(base: Mapping[str, str]) -> dict[str, str]:
    """Return the environment of a Python process solprop starts: `base`, BLAS idle.

    Adds :data:`BLAS_IDLE_DEFAULTS` where `base` sets none of them, so a user's own
    value wins.

    - ``base``: the environment the process would otherwise take.
    """
    return {**BLAS_IDLE_DEFAULTS, **base}


def batch_script(
    directives: Mapping[str, str | None],
    body: Sequence[str],
    *,
    site: Sequence[str] = (),
) -> str:
    """Return a Slurm batch script whose Python is a process solprop starts.

    The ``#SBATCH`` directives come first, each value quoted for the shell and one
    without a value left out, then the `site`'s own lines (further directives, module
    loads), then :data:`BLAS_IDLE_DEFAULTS` where the job's environment sets none of
    them (solprop issue #383), then `body`.
    """
    lines = [
        "#!/usr/bin/env bash",
        *(
            f"#SBATCH --{key}={shlex.quote(value)}"
            for key, value in directives.items()
            if value
        ),
        *site,
        *(
            f"export {name}=${{{name}:-{value}}}"
            for name, value in BLAS_IDLE_DEFAULTS.items()
        ),
        *body,
    ]
    return "\n".join(lines) + "\n"


_ENVIRONMENT_LOCK = threading.Lock()


@contextlib.contextmanager
def spawn_environment(
    overrides: Mapping[str, str] | None = None, removed: Sequence[str] = ()
) -> Iterator[None]:
    """Hold this process's environment as a spawned child should take it.

    ``multiprocessing``'s spawn reads the environment when a process starts, before
    the child replays ``__main__`` or imports its target; start it inside this block.
    The child takes :func:`child_environment` of this one with `overrides` set and
    `removed` unset, and the parent's own values come back on exit. One lock serves
    every caller, so two starts never see each other's variables.

    - ``removed``: variables the child must not have.
    """
    with _ENVIRONMENT_LOCK:
        target = child_environment({**os.environ, **(overrides or {})})
        for key in removed:
            target.pop(key, None)
        touched = {key for key in target if os.environ.get(key) != target[key]}
        touched |= {key for key in removed if key in os.environ}
        previous = {key: os.environ.get(key) for key in touched}
        try:
            for key in touched:
                if key in target:
                    os.environ[key] = target[key]
                else:
                    os.environ.pop(key, None)
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def sizes_blas_after_load() -> bool:
    """Say whether BLAS must be sized after it loads, not by the environment."""
    return sys.platform == "win32" and platform.machine() == "ARM64"


def _warn_of_load_time_blas_width() -> None:
    """Warn of a width that NumPy's linear solves can't survive on this platform."""
    if not sizes_blas_after_load():
        return
    for name in LOAD_TIME_BLAS_VARIABLES:
        value = os.environ.get(name, "").strip()
        # Measured: 1 and 2 fail at sizes 23 to 31, and 3 at size 31.
        if value.isdecimal() and 1 <= int(value) <= 3:
            warnings.warn(
                f"{name}={value} can crash or hang NumPy's linear solves on Windows "
                "on Arm, so unset it: solprop limits these libraries itself",
                RuntimeWarning,
                stacklevel=2,
            )


_warn_of_load_time_blas_width()


@dataclass(frozen=True)
class _BlasPool:
    """Keep one library's original width and its live owners' requests."""

    # threadpoolctl has no stubs; dynamic library controllers stop at this boundary.
    library: Any
    original: int
    requests: dict[object, int]


_BLAS_POOLS: dict[str, _BlasPool] = {}


@dataclass(frozen=True)
class _BlasLease(contextlib.AbstractContextManager[None]):
    """Release one owner's requests without disturbing the remaining owners."""

    owner: object
    paths: tuple[str, ...]

    def __exit__(self, *_exception: object) -> None:
        """Release the owner without suppressing a context's exception."""
        self.restore_original_limits()

    def restore_original_limits(self) -> None:
        """Restore the remaining limit, or the original width after last release."""
        with _BLAS_LOCK:
            for path in self.paths:
                pool = _BLAS_POOLS.get(path)
                if pool is None or self.owner not in pool.requests:
                    continue
                remaining = [
                    width
                    for owner, width in pool.requests.items()
                    if owner is not self.owner
                ]
                pool.library.set_num_threads(
                    min(remaining) if remaining else pool.original
                )
                # A failed setter leaves this owner intact so release can retry.
                del pool.requests[self.owner]
                if not pool.requests:
                    del _BLAS_POOLS[path]


def acquire_host_blas(libraries: Sequence[Any], threads: int) -> _BlasLease:
    """Borrow host BLAS libraries at the narrowest of their active requests.

    Construction and backend lifetimes can cross in either order. Independent
    threadpoolctl limiters capture each other's temporary settings and restore
    them too early; this process-owned lease saves the original width once.
    Native's adapter supplies this hook explicitly, keeping standalone native
    imports independent of solprop.
    """
    owner = object()
    paths = tuple(os.path.realpath(library.filepath) for library in libraries)
    lease = _BlasLease(owner, paths)
    with _BLAS_LOCK:
        try:
            for library, path in zip(libraries, paths, strict=True):
                if path not in _BLAS_POOLS:
                    _BLAS_POOLS[path] = _BlasPool(library, library.num_threads, {})
                pool = _BLAS_POOLS[path]
                pool.requests[owner] = threads
                library.set_num_threads(min(pool.requests.values()))
        except BaseException:
            lease.restore_original_limits()
            raise
    return lease


@contextlib.contextmanager
def host_blas() -> Iterator[None]:
    """Hold host construction's BLAS pools within the process's thread budget.

    Small mode solves and coupling products otherwise start every worker in
    NumPy's and SciPy's independent BLAS libraries. The scope overrides their
    vendor defaults with one thread, or ``SOLPROP_BLAS_THREADS`` under ADR-052,
    then restores the previous limits, including a live backend's own hold.
    The native backend's BLAS keeps its separately fitted width.

    Nested and overlapping construction share one library walk and the first
    scope's limit. The last scope restores it: threadpoolctl changes process-wide
    pools, so one Python thread must not restore them while another is using
    them. Importing this module starts no library walk or numerical runtime.
    """
    with _BLAS_LOCK:
        if not _BLAS_USERS:
            # Delayed: _process is imported before NumPy and the native runtime.
            import threadpoolctl  # noqa: PLC0415

            requested = thread_count("SOLPROP_BLAS_THREADS")
            threads = thread_budget().explicit(1 if requested is None else requested)
            native = sys.modules.get("solprop_native")
            build_info = getattr(native, "build_info", None)
            own = (
                ""
                if build_info is None
                else os.path.realpath(build_info().blas_library)
            )
            controller = threadpoolctl.ThreadpoolController()
            libraries = controller.select(user_api="blas")
            paths = [
                library.filepath
                for library in libraries.lib_controllers
                if os.path.realpath(library.filepath) != own
            ]
            # An empty selection must not limit the native BLAS on its own.
            if paths:
                _BLAS_LIMITS.enter_context(
                    acquire_host_blas(
                        controller.select(filepath=paths).lib_controllers, threads
                    )
                )
        owner = object()
        _BLAS_USERS.add(owner)
    try:
        yield
    finally:
        with _BLAS_LOCK:
            _BLAS_USERS.remove(owner)
            if not _BLAS_USERS:
                _BLAS_LIMITS.close()


def _query() -> tuple[int, ...]:
    """Return the calling thread's CPUs: its affinity, or the usable count's range."""
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.process_cpu_count() or 1))


def _at_import() -> tuple[int, ...]:
    """Return the process's CPUs as first seen, before an OpenMP runtime bound it.

    ``solprop_native``, when imported before ``solprop``, starts its OpenMP runtime
    first; it records the process's CPUs just before (``PROCESS_CPUS``), and that wider
    set is the grant when this thread's answer is one of its CPUs.
    """
    own = _query()
    earlier = getattr(sys.modules.get("solprop_native"), "PROCESS_CPUS", ())
    if len(earlier) > len(own) and set(own) <= set(earlier):
        return tuple(earlier)
    return own


_AT_IMPORT = _at_import()


def allowed_cpus() -> tuple[int, ...]:
    """Return the CPUs this process may run on, undoing an OpenMP main-thread binding.

    The current affinity, unless ``OMP_PROC_BIND`` is set and the current answer is a
    single CPU, or the hardware threads of a single core, of the wider set the process
    had when ``solprop`` was imported: that is the runtime's binding of the main thread
    to its first place (``OMP_PLACES=threads`` or ``cores``) and not the scheduler's
    grant, so the import-time set is returned. A deliberate narrowing to several cores,
    or any narrowing without ``OMP_PROC_BIND``, is believed.

    - Returns: sorted CPU indices, at least one.
    """
    current = _query()
    binding = os.environ.get("OMP_PROC_BIND", "").strip().lower()
    if (
        binding not in {"", "false"}
        and len(_AT_IMPORT) > len(current)
        and set(current) <= set(_AT_IMPORT)
        and (len(current) == 1 or physical_cores(current) == 1)
    ):
        return _AT_IMPORT
    return current


@dataclass(frozen=True)
class ThreadBudget:
    """The threads a process may run at once, and the limit that set it.

    Attributes
    ----------
    threads : int
        The budget, at least one.
    source : str
        The tightest limit: ``"affinity"``, ``"cgroup quota"`` or the name of the
        environment variable that set it.
    oversubscribe : bool
        Whether ``SOLPROP_OVERSUBSCRIBE`` lets an explicit count exceed the CPUs and
        the quota (see :meth:`explicit`).
    requested : int or None
        ``SOLPROP_THREADS``, which nothing exceeds, or ``None`` when it is unset.
    """

    threads: int
    source: str
    oversubscribe: bool = False
    requested: int | None = None

    def explicit(self, count: int) -> int:
        """Clamp a count someone asked for to what this budget allows.

        Normally the budget itself. With ``SOLPROP_OVERSUBSCRIBE=1`` only
        ``SOLPROP_THREADS`` holds, so a test can run more threads than the machine
        has CPUs on purpose (a thread-count invariance test on a small runner).

        Parameters
        ----------
        count : int
            The positive count asked for.

        Returns
        -------
        int
            The count allowed.

        Examples
        --------
        >>> from solprop._process import ThreadBudget
        >>> ThreadBudget(4, "affinity").explicit(8)
        4
        >>> ThreadBudget(4, "affinity", oversubscribe=True).explicit(8)
        8
        """
        if not self.oversubscribe:
            return min(count, self.threads)
        return count if self.requested is None else min(count, self.requested)


def _quota(directory: Path) -> int | None:
    """Return one control group's CPU quota in whole CPUs, or ``None`` for none.

    - Returns: the quota divided by its period, rounded down and at least one.
    """
    try:
        limit = directory / "cpu.max"
        if limit.is_file():
            value, period = limit.read_text(encoding="utf-8").split()
            return None if value == "max" else max(1, int(value) // int(period))
        quota, period = directory / "cpu.cfs_quota_us", directory / "cpu.cfs_period_us"
        if quota.is_file() and period.is_file():
            value = int(quota.read_text(encoding="utf-8"))
            if value > 0:
                return max(1, value // int(period.read_text(encoding="utf-8")))
    except OSError, ValueError:
        return None
    return None


def cgroup_cpus(root: Path | None = None) -> int | None:
    """Return the tightest CPU quota on this process's control-group path.

    A quota limits CPU *time*, not which CPUs: a team of threads that exceeds it is
    throttled for the rest of each period, and a team that waits on itself at every
    step stalls whole. The quota is therefore rounded **down** -- 1.5 CPUs is one
    thread. Every group from the process's own up to the hierarchy's root is read,
    for control-group versions 1 and 2, because a parent's quota binds its children.

    This process's own quota (``root`` unset) is read once and kept: every backend
    build and every scan admission asks, and a running process's groups do not
    change under it.

    - Returns: whole CPUs, at least one, or ``None`` where no quota is set or none can
      be read (not Linux, or no control groups).
    """
    return _own_cgroup_cpus() if root is None else _read_cgroup_cpus(root)


@functools.cache
def _own_cgroup_cpus() -> int | None:
    """Return this process's CPU quota, read once."""
    return _read_cgroup_cpus(Path("/"))


def _read_cgroup_cpus(root: Path) -> int | None:
    """Read the tightest CPU quota on this process's path under a filesystem root."""
    try:
        lines = (root / "proc/self/cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    quotas = [
        quota
        for directory in _cgroup_path(root, lines, "cpu")
        if (quota := _quota(directory)) is not None
    ]
    return min(quotas) if quotas else None


def _cgroup_path(root: Path, lines: Sequence[str], controller: str) -> Iterator[Path]:
    """Yield a controller's group directories from the process's own up to the mount.

    - ``lines``: ``/proc/self/cgroup``, one ``hierarchy:controllers:path`` per line;
      a line of another form is skipped. Version 2's unified line serves every
      controller, version 1's only its own.
    """
    mounts = root / "sys/fs/cgroup"
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) != 3:
            continue
        controllers, path = fields[1], Path(fields[2].lstrip("/"))
        if not controllers:
            base = mounts
        elif controller in controllers.split(","):
            base = mounts / controllers
            base = base if base.is_dir() else mounts / controller
        else:
            continue
        # A container without its own cgroup namespace names the host's path, which
        # its /sys/fs/cgroup does not have; the parents end at the mount, its own group.
        for directory in (path, *path.parents):
            yield base / directory


def cgroup_memory(root: Path | None = None) -> int | None:
    """Return the tightest memory limit on this process's control-group path.

    Every group from the process's own up to the hierarchy's root is read --
    ``memory.max`` (version 2) and ``memory.limit_in_bytes`` (version 1) -- because a
    parent's limit binds its children, and on a shared node the root group is often
    unlimited while the process's own is not. Unlike :func:`cgroup_cpus` it is read
    afresh at each call, and a limit file that exists but cannot be read or parsed is
    an error, not "no limit": admitting workers against the whole machine is how the
    group's limit kills them.

    - ``root``: filesystem root holding ``proc/self/cgroup`` and ``sys/fs/cgroup``;
      ``/`` when unset.
    - Returns: bytes, positive, or ``None`` where no limit is set (``max``, or zero)
      or the process has no control groups (not Linux).
    - Raises ``OSError`` when a file that exists cannot be read and ``ValueError`` when
      a limit is neither ``max`` nor a whole number of bytes; both name the file.
    """
    root = Path("/") if root is None else root
    record = root / "proc/self/cgroup"
    if not record.exists():
        return None
    limits: list[int] = []
    for directory in _cgroup_path(
        root, record.read_text(encoding="utf-8").splitlines(), "memory"
    ):
        for name in ("memory.max", "memory.limit_in_bytes"):
            file = directory / name
            if not file.exists():
                continue
            value = file.read_text(encoding="utf-8").strip()
            if value == "max":
                continue
            if not value.isdecimal():
                raise ValueError(f"{file} holds {value!r}, not a memory limit in bytes")
            if int(value) > 0:
                limits.append(int(value))
    return min(limits) if limits else None


def cores_in_masks(masks: Sequence[int], cpus: Sequence[int]) -> int:
    """Count the cores, given as CPU bitmasks, that hold any of ``cpus``.

    - ``masks``: one bitmask per physical core: bit ``n`` set when CPU ``n`` is one of
      its hardware threads.
    - Returns: the number of cores holding at least one of ``cpus``.
    """
    wanted = sum(1 << cpu for cpu in cpus)
    return sum(1 for mask in masks if mask & wanted)


def _linux_cores(cpus: Sequence[int], root: Path) -> int | None:
    """Count the distinct (package, core) pairs of ``cpus`` in a sysfs CPU tree.

    - Returns: the number of physical cores, or ``None`` if any CPU's topology is
      unreadable.
    """
    pairs: set[tuple[str, str]] = set()
    for cpu in cpus:
        topology = root / f"cpu{cpu}" / "topology"
        try:
            pairs.add(
                (
                    (topology / "physical_package_id").read_text(encoding="utf-8"),
                    (topology / "core_id").read_text(encoding="utf-8"),
                )
            )
        except OSError:
            return None
    return len(pairs) or None


def _windows_cores(cpus: Sequence[int]) -> int | None:  # pragma: no cover - Windows
    """Count the physical cores holding ``cpus`` from Windows' processor table.

    - Returns: the number of physical cores, or ``None`` if the table cannot be read.
    """
    import ctypes  # noqa: PLC0415

    class _Information(ctypes.Structure):
        """One ``SYSTEM_LOGICAL_PROCESSOR_INFORMATION`` entry.

        The mask, the relationship and a 16-byte union aligned to 8.
        """

        _fields_ = (
            ("mask", ctypes.c_size_t),
            ("relationship", ctypes.c_int),
            ("union", ctypes.c_ulonglong * 2),
        )

    try:
        # One calling convention on 64-bit Windows, so CDLL reaches kernel32 and is
        # typed on every platform, unlike ``windll``.
        query = ctypes.CDLL("kernel32").GetLogicalProcessorInformation
        size = ctypes.c_ulong(0)
        query(None, ctypes.byref(size))
        table = (_Information * (size.value // ctypes.sizeof(_Information)))()
        if not query(table, ctypes.byref(size)):
            return None
    except AttributeError, OSError:
        return None
    # Relationship 0 is RelationProcessorCore: one entry per physical core.
    masks = [int(entry.mask) for entry in table if entry.relationship == 0]
    return cores_in_masks(masks, cpus) or None


def physical_cores(
    cpus: Sequence[int], root: Path = Path("/sys/devices/system/cpu")
) -> int | None:
    """Count the physical cores that hold ``cpus``, or ``None`` where unknown.

    Two hardware threads of one core share its arithmetic units, and solprop's kernels
    keep those busy, so a second thread per core adds contention, not speed. Linux
    reads each CPU's package and core from ``root``; Windows asks the kernel. macOS on
    Apple silicon has one thread per core, so the CPUs are the cores.

    - Returns: the number of physical cores, or ``None`` on other platforms or when the
      topology cannot be read.
    """
    if sys.platform.startswith("linux"):
        return _linux_cores(cpus, root)
    if sys.platform == "win32":
        return _windows_cores(cpus)
    return None


def _sysctl_int(name: str) -> int | None:  # pragma: no cover - macOS
    """Read an integer ``sysctl`` value by name on macOS, or ``None``.

    - Returns: its value, or ``None`` if the kernel does not have it.
    """
    import ctypes  # noqa: PLC0415

    value = ctypes.c_int(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    try:
        found = ctypes.CDLL(None).sysctlbyname(
            name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0
        )
    except AttributeError, OSError:
        return None
    return value.value if found == 0 else None


@functools.cache
def top_tier_cores() -> int | None:
    """Count the cores of a chip's fastest kind, where it has more than one kind.

    A parallel region splits its work evenly and waits for its slowest thread, so a
    team that spans a chip's slower cores runs at their pace: on an Apple M5 Max (6
    top-tier and 12 second-tier cores) five of six cases took 1.17 to 1.49 times as
    long at 18 threads as at 6 (solprop issue #333). macOS reports the kinds as
    performance levels, level 0 the fastest. Other platforms are not read here.

    - Returns: the cores of performance level 0 on a macOS chip with several levels,
      else ``None``.
    """
    if sys.platform != "darwin":
        return None
    levels = _sysctl_int("hw.nperflevels")
    if levels is None or levels < 2:
        return None
    cores = _sysctl_int("hw.perflevel0.physicalcpu")
    return cores if cores is not None and cores > 0 else None


def thread_count(name: str) -> int | None:
    """Read a positive thread count from the environment.

    - Returns: the count, or ``None`` when the variable is unset or blank.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        return None
    try:
        count = int(value)
    except ValueError as error:
        msg = f"{name} must be a thread count, and is {value!r}"
        raise ValueError(msg) from error
    if count < 1:
        msg = f"{name} must be positive, and is {value!r}"
        raise ValueError(msg)
    return count


PARALLEL_BYTES = 2**26
"""The fewest bytes a host pass over a state threads (:func:`parallel_rows`)."""


def parallel_rows[T](n_rows: int, nbytes: int, work: Callable[[slice], T]) -> list[T]:
    """Run `work` over slices covering `n_rows` rows, in threads if it is worth it.

    For a NumPy pass over a state too large for one thread's page faults and memory
    bandwidth: ufuncs release the GIL, and a state's rows are independent.

    - ``n_rows``: the rows to cover; positive.
    - ``nbytes``: the bytes the pass reads. Below :data:`PARALLEL_BYTES` one call takes
      every row.
    - ``work``: called with each slice of rows; it must touch nothing the others do.
    """
    # A small pass never threads, so it reads no budget: the topology walk behind it
    # cost 60 ms of a 0.36 s window in host statistics (solprop issue #556).
    if nbytes < PARALLEL_BYTES or n_rows <= 1:
        return [work(slice(0, n_rows))]
    workers = min(thread_budget().threads, n_rows)
    if workers <= 1:
        return [work(slice(0, n_rows))]
    edges = [n_rows * index // workers for index in range(workers + 1)]
    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(work, (slice(a, b) for a, b in itertools.pairwise(edges))))


def thread_budget() -> ThreadBudget:
    """Return how many threads this process may run at once (ADR-052).

    The smallest of the CPUs it may run on (:func:`allowed_cpus`: ``taskset``, a batch
    scheduler's allocation), the physical cores that hold them
    (:func:`physical_cores`) and, on a chip with cores of several kinds, those of the
    fastest kind (:func:`top_tier_cores`), both unless ``SOLPROP_THREADS`` is set, its
    control groups'
    CPU quota (:func:`cgroup_cpus`) and ``SOLPROP_THREADS``. Every pool solprop sizes
    is held within it, and so is any count asked for explicitly unless
    ``SOLPROP_OVERSUBSCRIBE=1``
    (:meth:`ThreadBudget.explicit`); the pools take turns, so each may use the whole
    budget.

    Returns
    -------
    ThreadBudget
        The budget and the limit that set it; on a tie, the earlier in that order.

    Raises
    ------
    ValueError
        If ``SOLPROP_THREADS`` is not a positive thread count, or
        ``SOLPROP_OVERSUBSCRIBE`` is neither ``0`` nor ``1``.

    Examples
    --------
    >>> from solprop._process import thread_budget
    >>> thread_budget().threads >= 1
    True
    """
    cpus = allowed_cpus()
    limits = [(len(cpus), "affinity")]
    requested = thread_count("SOLPROP_THREADS")
    if requested is None and (cores := physical_cores(cpus)) is not None:
        limits.append((cores, "physical cores"))
    if requested is None and (top := top_tier_cores()) is not None:
        limits.append((top, "top-tier cores"))
    quota = cgroup_cpus()
    if quota is not None:
        limits.append((quota, "cgroup quota"))
    if requested is not None:
        limits.append((requested, "SOLPROP_THREADS"))
    threads, source = min(limits, key=lambda limit: limit[0])
    return ThreadBudget(
        threads=max(1, threads),
        source=source,
        oversubscribe=oversubscribing(),
        requested=requested,
    )


def oversubscribing() -> bool:
    """Return whether ``SOLPROP_OVERSUBSCRIBE`` lets explicit counts pass the CPUs."""
    value = os.environ.get("SOLPROP_OVERSUBSCRIBE", "").strip()
    if value not in {"", "0", "1"}:
        msg = f"SOLPROP_OVERSUBSCRIBE must be 0 or 1, and is {value!r}"
        raise ValueError(msg)
    return value == "1"


def process_stamp(pid: int) -> str | None:
    """Return when a process started, as its operating system states it.

    A process number is reused once its process has gone; the start time tells a
    run's own process from a later one with its number.

    - Returns: its start, in the system's own terms; ``None`` where the system does not
      say or the process does not exist.
    """
    if sys.platform == "linux":
        return _proc_stamp(pid)
    if sys.platform == "win32":  # pragma: no cover - Windows only
        return _windows_stamp(pid)
    return _ps_stamp(pid)  # pragma: no cover - macOS and other POSIX: not the lanes


def _proc_stamp(pid: int) -> str | None:
    """Return a process's start as Linux's ``/proc`` gives it, in clock ticks.

    - Returns: field 22 of ``/proc/<pid>/stat``; ``None`` when the process does not
      exist, including when it exits while the file is read.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError, ProcessLookupError:
        return None
    # The command name, field 2, is in parentheses and may itself hold them, so the
    # fields are counted from the last closing one.
    return stat.rsplit(")", 1)[1].split()[19]


def _ps_stamp(pid: int) -> str | None:  # pragma: no cover - not Linux, see above
    """Return a process's start as ``ps`` gives it, off Linux.

    - Returns: its start; ``None`` when it does not exist.
    """
    import subprocess  # noqa: PLC0415 - only off Linux

    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def _windows_stamp(pid: int) -> str | None:  # pragma: no cover - Windows only
    """Return a Windows process's creation time, or ``None`` when it has gone.

    - Returns: its creation time in 100 ns units, as text.
    """
    import ctypes  # noqa: PLC0415 - Windows only
    from ctypes import wintypes  # noqa: PLC0415 - Windows only

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    try:
        code = wintypes.DWORD()
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return None
        if code.value != 259:  # STILL_ACTIVE
            return None
        times = [wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
            return None
        created = times[0]
        return str((created.dwHighDateTime << 32) | created.dwLowDateTime)
    finally:
        kernel.CloseHandle(handle)
