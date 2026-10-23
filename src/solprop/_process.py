# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CPUs this process may run on, as the scheduler granted them.

Linux reports affinity per thread. An OpenMP runtime started with ``OMP_PROC_BIND``
set -- GNU libgomp, which pyFFTW's and Kokkos's builds load -- binds the **main**
thread to the first of its places when it starts, so after the first parallel region
``os.sched_getaffinity(0)`` names one CPU although the process was granted many. A
thread pool sized from that answer runs on one core (measured: an 18-core allocation
gave the Numba kernels one thread). The affinity is therefore read once, when
``solprop`` is imported, before any of its runtimes start, and the later answer is
believed only when it is not that collapse.

Threads inherit their creator's affinity, so a pool started from the bound main thread
puts every worker on that one CPU: eighteen Numba workers on one core ran 3.7 times
slower than one thread (measured). :func:`unbound` lifts the binding while a pool is
started and puts it back after.
"""

import contextlib
import os
import sys
from collections.abc import Iterator


def _query() -> tuple[int, ...]:
    """Return the calling thread's CPUs: its affinity, or the usable count's range.

    Returns
    -------
    tuple of int
        The CPU indices, sorted.
    """
    if hasattr(os, "sched_getaffinity"):
        return tuple(sorted(os.sched_getaffinity(0)))
    return tuple(range(os.process_cpu_count() or 1))


def _at_import() -> tuple[int, ...]:
    """Return the process's CPUs as first seen, before an OpenMP runtime bound it.

    ``solprop_native``, when imported before ``solprop``, starts its OpenMP runtime
    first; it records the process's CPUs just before (``PROCESS_CPUS``), and that wider
    set is the grant when this thread's answer is one of its CPUs.

    Returns
    -------
    tuple of int
        Sorted CPU indices.
    """
    own = _query()
    earlier = getattr(sys.modules.get("solprop_native"), "PROCESS_CPUS", ())
    if len(earlier) > len(own) and set(own) <= set(earlier):
        return tuple(earlier)
    return own


_AT_IMPORT = _at_import()


def _size_numba() -> None:
    """Name the grant's size to Numba when an OpenMP binding would shrink its pool.

    Numba sizes its pool from the importing thread's affinity, once. When an OpenMP
    runtime has already bound this thread to one CPU (``solprop_native`` imported
    first), that pool would hold one thread. ``NUMBA_NUM_THREADS`` is set to the
    grant's size first, unless Numba is already imported or the user set it.
    """
    if (
        len(_AT_IMPORT) > len(_query())
        and "numba" not in sys.modules
        and "NUMBA_NUM_THREADS" not in os.environ
    ):
        os.environ["NUMBA_NUM_THREADS"] = str(len(_AT_IMPORT))


_size_numba()


def allowed_cpus() -> tuple[int, ...]:
    """Return the CPUs this process may run on, undoing an OpenMP main-thread binding.

    The current affinity, unless ``OMP_PROC_BIND`` is set and the current answer is a
    single CPU of the wider set the process had when ``solprop`` was imported: that is
    the runtime's binding of the main thread and not the scheduler's grant, so the
    import-time set is returned. A deliberate narrowing to several CPUs, or any
    narrowing without ``OMP_PROC_BIND``, is believed.

    Returns
    -------
    tuple of int
        Sorted CPU indices, at least one.

    Examples
    --------
    >>> from solprop._process import allowed_cpus
    >>> len(allowed_cpus()) >= 1
    True
    """
    current = _query()
    binding = os.environ.get("OMP_PROC_BIND", "").strip().lower()
    if (
        binding not in {"", "false"}
        and len(current) == 1
        and len(_AT_IMPORT) > 1
        and set(current) <= set(_AT_IMPORT)
    ):
        return _AT_IMPORT
    return current


def _collapsed() -> bool:
    """Return whether the main thread's affinity is an OpenMP binding's single CPU.

    Returns
    -------
    bool
        True when :func:`allowed_cpus` differs from the thread's current affinity.
    """
    return allowed_cpus() != _query()


@contextlib.contextmanager
def unbound() -> Iterator[None]:
    """Give the calling thread the process's CPUs while a thread pool is started.

    Workers created inside the block inherit every CPU the process was granted, not
    the one CPU an OpenMP runtime bound the main thread to; the binding is restored on
    leaving, so the runtime's own placement of the main thread is unchanged. Nothing
    happens where the affinity cannot be set or was not collapsed.

    Yields
    ------
    None
        Inside the block the calling thread may run on the whole grant.

    Examples
    --------
    >>> from solprop._process import unbound
    >>> with unbound():
    ...     pass
    """
    if not hasattr(os, "sched_setaffinity") or not _collapsed():
        yield
        return
    bound = _query()
    os.sched_setaffinity(0, _AT_IMPORT)
    try:
        yield
    finally:
        os.sched_setaffinity(0, bound)


def release() -> bool:
    """Give the calling thread the process's CPUs for good, undoing an OpenMP binding.

    For ``numba_cpu``: FFTW's and Numba's threads start whenever a plan first runs or a
    pool first grows, long after construction, and each inherits the thread that starts
    it. When the native backend's OpenMP runtime, loaded first under ``OMP_PROC_BIND``,
    has bound the main thread to one CPU, every such thread would share it: the
    comb-to-continuum window took 45 s against 5.4 s on 64 Zen 5 cores (SEP 28).
    The native backend's own threads keep their placement; only the main thread moves.

    Returns
    -------
    bool
        Whether the affinity was changed.

    Examples
    --------
    >>> from solprop._process import release
    >>> isinstance(release(), bool)
    True
    """
    if not hasattr(os, "sched_setaffinity") or not _collapsed():
        return False
    os.sched_setaffinity(0, _AT_IMPORT)
    return True
