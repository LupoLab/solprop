# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""FFTW plans, wisdom, and the five thread pools the backend owns.

Two things live here because they are the same decision. The transforms are planned by
FFTW on aligned buffers that never move, and FFTW's own pool is one of five that share
the machine: the pool configuration alone was measured to move **one step by a
factor of nine** (``docs/dev/backends.md`` section 3.4). Three pools of N threads
on N cores is 3N runnable threads and the idle ones spin, so a backend that sets its
own pool and inherits the others is not configured, it is lucky.

**Planning is destructive.** ``FFTW_MEASURE`` writes over its input while it searches
for a codelet, so every plan is built at construction, before a buffer holds a state.
A plan built later, on a live state, would silently replace it with noise.

**No normalisation.** ``pyfftw.FFTW.execute`` applies none, and this backend wants none:
the ``1/N`` of an inverse transform is folded into the contract's own scale factors
(``kernel-contract.md`` section 4), exactly as the reference's ``numpy.fft`` calls have
it folded in.
"""

import ctypes
import logging
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import threadpoolctl
from numba import get_num_threads, set_num_threads
from numba.core import config as numba_config

from solprop._process import ThreadBudget, thread_budget, unbound
from solprop.backends.numba_cpu import openmp
from solprop.backends.numba_cpu.openmp import pyfftw
from solprop.exceptions import BackendError

__all__ = [
    "BLAS_MAX_THREADS",
    "BLAS_THREAD_ELEMENTS",
    "FFTW_MAX_THREADS",
    "FFTW_THREAD_ELEMENTS",
    "FFTW_WIDE_ELEMENTS",
    "MODAL_REGIME_ELEMENTS",
    "PLAN_FLAGS",
    "REGIME_ELEMENTS",
    "ColumnFamily",
    "GeneralPlans",
    "ThreadPools",
    "TransformPlans",
    "TransversePlans",
    "apply_pools",
    "build_general_plans",
    "build_plans",
    "build_transverse_plans",
    "choose_pools",
    "export_wisdom",
    "import_wisdom",
    "release_pools",
    "wisdom_digest",
]

logger = logging.getLogger(__name__)

ColumnFamily = Literal["field_real", "field_analytic", "envelope"]
"""Which of the three response columns a problem compiles to.

``field_real``
    A field grid that radiates the third harmonic: the samples are real and the Kerr
    term is the full cubic ``c rho (sum_q E_q^2) E_p``.
``field_analytic``
    A field grid that does not: the column works on the analytic signal of the real
    field and takes the real part of the polarisation on the way out. The analytic
    signal is **not** a second transform here -- it is the complex inverse of the same
    one-sided spectrum with the positive bins doubled, which the reference forms
    separately once per node chunk.
``envelope``
    An envelope grid: the samples are complex throughout, the harmonic arrives as a
    carrier when the grid radiates it. Envelope plasma carries two independent
    current streams through its own forward transforms before spectral assembly.
"""

PLAN_FLAGS: Mapping[str, str] = {
    "measure": "FFTW_MEASURE",
    "estimate": "FFTW_ESTIMATE",
    "patient": "FFTW_PATIENT",
}
"""The planning effort, by the name a run asks for it by.

``measure`` is the default: a plan executed millions of times gains from codelet
selection. It must be built on a quiet machine, though -- ``MEASURE`` has been measured
planning 2.4 times *slower* than ``ESTIMATE`` on a busy host, because the search itself
is timed against the load (``docs/dev/backends.md`` section 3.3).
"""

REGIME_ELEMENTS = 1_000_000
"""Element count above which a single problem is stepped in the large regime.

``n_to * n_modes * n_pol * n_nodes`` from ``docs/dev/backends.md`` section 3.4: below
this the kernels run serially and FFTW takes one thread, because the pools cost more
to wake than the work saves; above it the response pass runs ``prange`` over node
chunks and the batched plans take FFTW's pool.
"""

MODAL_REGIME_ELEMENTS = 131_072
"""Kernel-regime cutoff for fibre transforms with multiple physical nodes.

The reduced M2 and CtC trials benefit from node parallelism below a million-element
cutoff. Re-measured after SEP 25's multipass plasma column: the resolved argon
blue-shift window (one mode, 33 nodes, 4096 samples, 135 168 elements) took 3.35 ms a
Runge-Kutta trial in the parallel regime against 7.68 ms serially at six threads on
Apple silicon, so the power-of-two bracket now lies below it. Single-node and spatial
problems keep :data:`REGIME_ELEMENTS`. FFTW and BLAS retain independent size rules.
"""

FFTW_THREAD_ELEMENTS = 65_536
"""Elements in one batched transform at or above which FFTW is given more than one
thread.

``n_columns * n_to`` for the modal pair, and the same count over the doubled grid for
the Raman pair. **Measured on two machines**, an Apple-silicon laptop (numpy linked
against Accelerate) and an 18-core x86-64 server (OpenBLAS), as the smallest size at
which two threads repay what they cost -- which is the only defensible place for it,
since below that they are a straight loss:

::

    elements                        laptop, 2 threads   server, 2 threads
    8 192                                       0.49x               0.81x
    16 384                                      0.78x               0.63x
    32 768  (the M2 modal pair)                 1.06x               1.37x
    65 536  <- the threshold                    1.52x               1.58x
    393 216  (the CtC modal pair)               1.89x               1.81x


The two FFTW pools break even at the same count, 32 768, and the threshold sits an
octave **above** it. That is a second measurement, not caution. Timing the M2 case end
to end on this backend with the pool forced to one and left to the rule, interleaved so
that a drifting load cancels, the transform-threaded arm is **slower**: 12.89 and 11.48
s at one thread against 14.66 and 13.36 s at four, on a case whose modal pair is
exactly 32 768 elements. A tight loop over one plan amortises the pool's wake-up; a
step pays it on every call, so break-even in the sweep is a loss in a run. The
threshold belongs where the sweep shows a gain worth having on both machines -- 1.52x
and 1.58x at 65 536 -- and the CtC case, at 393 216, is far above it either way.

A placeholder of 1e6 is fifteen times too high: with it the CtC case ran its
393 216-element transform serially and took 5710 s at 132 % of one core.
"""

FFTW_WIDE_ELEMENTS = 1 << 20
"""Elements in one batched transform at or above which FFTW takes every allowed thread.

:data:`FFTW_MAX_THREADS` was measured on transforms of up to 393 216 elements. A
free-space evaluation's transforms are far larger: the transverse pair moves a whole
field volume, ``n_x n_y n_omega n_pol`` elements (88 million on a 384-square grid of 600
bins), and a node chunk of the spatial default budget gives a temporal batch of about
three million. Both scale with the pool (SEP 28, the 384-square Kerr march):

::

    pool                         x86-64, 64 cores (Linux)  Apple silicon, 18 cores
    transverse 4, temporal 4                         --                   4.75 s/step
    transverse all, temporal 4              2.63 s/step                   2.26 s/step
    transverse all, temporal all            0.77 s/step                   2.02 s/step
    transverse all, temporal 1              3.31 s/step                            --

A transform this long also makes a pool's wake-up negligible.
"""

FFTW_MAX_THREADS = 4
"""Threads a batched transform is given once it is over the threshold.

Not the core count, which is what the pool used to get and is worse almost everywhere:
on the CtC modal pair the Apple-silicon laptop reads 1.89x at two threads, **2.64x at
four**, 2.31x at eight and 1.69x at eighteen, and the x86-64 server 1.81x, 2.33x, 1.66x
and 1.74x. Four is at or near the best on every size the M2 and CtC cases span, on both
machines.

Above about half a million elements more threads do start to pay again -- the laptop
reads 4.30x at eight on a 786 432-element pair against 2.62x at four -- but both cases
are below that and a size-dependent count is not measured finely enough to set here.
"""

BLAS_THREAD_ELEMENTS = 262_144
"""Elements in the synthesis GEMM at or above which the BLAS keeps its own pool.

``n_rows * n_to``. **Set from the x86-64 server alone, and that is not a shortcut**:
this is the one threshold the two machines cannot both inform, because numpy on the
Apple-silicon laptop is linked against Accelerate, which exposes no runtime thread count
at all. The laptop's own measurement is what establishes that -- 0.98x, 0.98x, 0.99x,
0.99x, 0.99x, 1.01x, 1.02x, 0.89x across a 160-fold range of sizes, which is not a
threading curve but a flat line with noise.

On the server, where OpenBLAS answers to `threadpoolctl`, two threads first repay at the
M2 case's own GEMM shape:

::

    elements                                  server, 2 threads
    135 168                                               0.99x
    270 336   (33 x 4 @ 4 x 8192)                         1.22x
    540 672                                               1.35x
    2 686 976 (41 x 6 @ 6 x 65 536)                       1.45x


The threshold is the round number below that crossover. The bracket of
``docs/dev/backends.md`` section 3.4 -- one thread wins on RDW, loses on CtC -- is
confirmed and located.
"""

BLAS_MAX_THREADS = 2
"""Threads the synthesis GEMM is given once it is over the threshold.

Two, and the machine's own count is far from harmless: on the CtC GEMM the server reads
**1.45x at two threads**, 1.22x at four, 1.23x at eight, 1.19x at eighteen, 1.05x at
thirty-six and 1.01x at seventy-two. The GEMM is pathologically skinny -- `k` is the
mode count, four or six -- so there is very little for a large pool to divide and a
great deal of synchronisation to pay for it.

A GEMM twice the size of the CtC case's does start to want more (1.83x at four against
1.60x at two on a 5.4-million-element shape), and that is where a size-dependent count
would earn itself; neither the M2 nor the CtC case reaches it.
"""


@dataclass(frozen=True)
class ThreadPools:
    """The four modal pools, as this backend set them.

    The transverse FFTW pool of a Cartesian run is chosen beside them by
    :func:`build_transverse_plans`.

    Attributes
    ----------
    regime : str
        ``"small"`` or ``"large"``, from :data:`REGIME_ELEMENTS`.
    numba : int
        Threads the fused kernels may use.
    fftw : int
        Threads each batched plan was built with.
    blas : int
        Threads the BLAS was limited to, or its own count where it could not be set.
    blas_mechanism : str
        How the BLAS pool was reached: the library threadpoolctl found and limited, or
        ``"uncontrolled"`` when there is none it can set. Apple's Accelerate is the
        latter: it dispatches through Grand Central Dispatch and exposes no
        ``set_num_threads``, so it is left to its own size gating and said to be.
    numba_openmp : str
        How Numba's threads were started
        (:func:`solprop.backends.numba_cpu.openmp.numba_launch`): ``"unset"`` until
        :func:`apply_pools` starts them.
    thread_budget : int
        The process's thread budget, which every pool stays within (ADR-052); zero
        where it was not read.
    thread_budget_source : str
        The limit that set it (:class:`solprop._process.ThreadBudget`).
    """

    regime: str
    numba: int
    fftw: int
    blas: int
    blas_mechanism: str
    numba_openmp: str = "unset"
    thread_budget: int = 0
    thread_budget_source: str = "unset"

    def as_metadata(self) -> dict[str, object]:
        """Return the pools as the run's metadata records them.

        Returns
        -------
        dict
            One entry per pool plus the regime, all JSON-representable.
        """
        return {
            "regime": self.regime,
            "numba_threads": self.numba,
            "fftw_threads": self.fftw,
            "blas_threads": self.blas,
            "blas_mechanism": self.blas_mechanism,
            "numba_openmp": self.numba_openmp,
            "thread_budget": self.thread_budget,
            "thread_budget_source": self.thread_budget_source,
        }


def _thread_budget() -> ThreadBudget:
    """Return the process's thread budget, which every pool here stays within.

    ``os.cpu_count`` reports the machine; a process is held below it by its affinity
    mask, its control groups' CPU quota and ``SOLPROP_THREADS``
    (:func:`solprop._process.thread_budget`, ADR-052), and a pool sized from the
    machine inside a half-core container spends its life being descheduled.

    Returns
    -------
    solprop._process.ThreadBudget
        The budget and the limit that set it.

    Raises
    ------
    BackendError
        If ``SOLPROP_THREADS`` is set to something that is not a positive thread count.
    """
    try:
        return thread_budget()
    except ValueError as error:
        raise BackendError(str(error)) from error


def choose_pools(  # noqa: PLR0913 - pool sizes, cutoff and explicit overrides
    elements: int,
    gemm_elements: int,
    transform_elements: int,
    *,
    threads: int | None = None,
    fftw_threads: int | None = None,
    blas_threads: int | None = None,
    regime: str | None = None,
    regime_elements: int | None = None,
    spatial: bool = False,
) -> ThreadPools:
    """Choose the four modal pools from the problem's size, with explicit overrides.

    Parameters
    ----------
    elements : int
        ``n_to * n_modes * n_pol * n_nodes``, which selects the regime.
    gemm_elements : int
        ``n_rows * n_to`` of the synthesis GEMM, which decides the BLAS pool.
    transform_elements : int
        ``n_columns * n_to`` of the batched modal transform, which decides FFTW's.
    threads : int, optional
        Override for the kernels' pool. ``None`` takes the regime's choice. Either way
        the pool is clamped to the process's thread budget
        (:meth:`solprop._process.ThreadBudget.explicit`) and to Numba's own
        ``NUMBA_NUM_THREADS``.
    fftw_threads : int, optional
        Override for FFTW's pool.
    blas_threads : int, optional
        Override for the BLAS pool.
    regime : {'small', 'large'}, optional
        Override for the regime itself. ``None`` reads it off ``elements`` against
        the selected cutoff, which is what a run should normally do; forcing it is
        for a measurement that wants a problem below the threshold stepped as a large
        one -- the node loop becomes a ``prange`` and every pool takes its threads, so
        a case that would otherwise be serial can be measured against thread count.
    regime_elements : int, optional
        Positive element cutoff for the automatic kernel regime. ``None`` uses
        :data:`REGIME_ELEMENTS`; modal problems with multiple physical nodes use
        :data:`MODAL_REGIME_ELEMENTS`. An explicit ``regime`` still takes precedence.
    spatial : bool, optional
        Whether the transform is a free-space or planar evaluation's (its transverse
        pair or its chunk's temporal batch), which takes every thread from
        :data:`FFTW_WIDE_ELEMENTS`. Modal and Raman transforms keep
        :data:`FFTW_MAX_THREADS`, the rule measured for them; the wider pool was
        measured on spatial transforms only (SEP 28).

    Returns
    -------
    ThreadPools
        What was chosen, before it is applied.

    Raises
    ------
    BackendError
        If a thread count or cutoff is not positive, or the regime is unknown.

    Examples
    --------
    >>> from solprop.backends.numba_cpu.fft import choose_pools
    >>> choose_pools(1000, 1000, 1000, threads=4).regime
    'small'
    """
    for name, value in (
        ("threads", threads),
        ("fftw_threads", fftw_threads),
        ("blas_threads", blas_threads),
    ):
        if value is not None and value < 1:
            msg = f"{name} must be a positive thread count, got {value}"
            raise BackendError(msg)
    if regime is not None and regime not in {"small", "large"}:
        msg = f"regime must be 'small' or 'large', got {regime!r}"
        raise BackendError(msg)
    cutoff = REGIME_ELEMENTS if regime_elements is None else regime_elements
    if cutoff < 1:
        msg = f"regime_elements must be a positive element count, got {cutoff}"
        raise BackendError(msg)
    budget = _thread_budget()
    cap = budget.threads
    # Numba's pool is fixed at import by NUMBA_NUM_THREADS and set_num_threads refuses
    # more, so the kernels are held to it as well; FFTW and the BLAS have no such limit.
    ceiling: int = numba_config.NUMBA_NUM_THREADS  # pyright: ignore[reportAttributeAccessIssue] - set at import
    large = elements > cutoff if regime is None else regime == "large"
    if threads is not None:
        kernels = min(budget.explicit(threads), ceiling)
    else:
        kernels = min(cap, ceiling) if large else 1
    # The regime decides the *kernels'* pool, because that is what it is about: whether
    # the node loop is a prange. The transforms and the GEMMs are governed by their own
    # measured thresholds and not by the regime, since a transform of 393 216 elements
    # repays two threads whatever the rest of the problem looks like -- and the CtC
    # case, which is well inside the large regime, is exactly where the old
    # `large and ...` conjunction still left it serial.
    if fftw_threads is not None:
        fftw = budget.explicit(fftw_threads)
    else:
        fftw = (
            cap
            if spatial and transform_elements >= FFTW_WIDE_ELEMENTS
            else min(cap, FFTW_MAX_THREADS)
            if transform_elements >= FFTW_THREAD_ELEMENTS
            else 1
        )
    if blas_threads is not None:
        blas = budget.explicit(blas_threads)
    else:
        blas = (
            min(cap, BLAS_MAX_THREADS) if gemm_elements >= BLAS_THREAD_ELEMENTS else 1
        )
    return ThreadPools(
        regime="large" if large else "small",
        numba=kernels,
        fftw=fftw,
        blas=blas,
        blas_mechanism="unset",
        thread_budget=budget.threads,
        thread_budget_source=budget.source,
    )


def apply_pools(
    pools: ThreadPools, *, controller: Any = None
) -> tuple[ThreadPools, Any]:
    """Set the kernel and BLAS pools, and say what was actually reached.

    FFTW's pool is not set here: it is fixed when a plan is built, so it travels with
    the plan rather than with the process.

    Parameters
    ----------
    pools : ThreadPools
        What :func:`choose_pools` decided.
    controller : threadpoolctl.ThreadpoolController, optional
        The library walk to use. A backend passes none and gets the real one; a test
        passes its own, because **which** branch is taken here is a property of the
        machine -- Accelerate has no controllable BLAS and OpenBLAS has -- and both must
        be exercised wherever the suite runs.

    Returns
    -------
    pools : ThreadPools
        The same, with ``blas_mechanism`` filled in from what was found.
    limiter : object or None
        The live threadpoolctl limit, to be released when the backend closes, or
        ``None`` where there was no controllable library.
    """
    # Numba starts its workers here; they inherit this thread's affinity, which an
    # OpenMP runtime may have bound to one CPU (solprop._process).
    with unbound(), openmp.numba_launch() as launch:
        set_num_threads(pools.numba)
    pools = replace(pools, numba_openmp=launch)
    walk = threadpoolctl.ThreadpoolController() if controller is None else controller
    libraries = walk.select(user_api="blas")
    if not libraries.lib_controllers:
        # Apple's Accelerate is the case that matters here: numpy links it on macOS, it
        # threads through Grand Central Dispatch, and it exposes no set_num_threads at
        # all. Saying so is the honest record; pretending the pool was set is not.
        uncontrolled = replace(pools, blas_mechanism="uncontrolled")
        _log_pools(uncontrolled)
        return uncontrolled, None
    limiter = libraries.limit(limits=pools.blas)
    limiter.__enter__()
    names = ",".join(sorted({one.internal_api for one in libraries.lib_controllers}))
    reached = replace(pools, numba=get_num_threads(), blas_mechanism=names)
    _log_pools(reached)
    return reached, limiter


def _log_pools(pools: ThreadPools) -> None:
    """Say what each pool got, on every path out of :func:`apply_pools`.

    INFO, not DEBUG: a case just below a threshold runs serially and silently, and
    one CtC run lost 5710 s to exactly that. And on *every* path, because logging only
    where the BLAS is controllable would never print on a numpy linked against
    Accelerate.

    Parameters
    ----------
    pools : ThreadPools
        What was reached.
    """
    logger.info(
        "numba_cpu: %s regime, %d kernel threads, %d FFTW, %d BLAS (%s)",
        pools.regime,
        pools.numba,
        pools.fftw,
        pools.blas,
        pools.blas_mechanism,
    )


def release_pools(limiter: Any) -> None:
    """Give the BLAS pool back, if this backend took it.

    Parameters
    ----------
    limiter : object or None
        Whatever :func:`apply_pools` returned.
    """
    if limiter is not None:
        limiter.__exit__(None, None, None)


def import_wisdom(path: str | os.PathLike[str] | None = None) -> bool:
    """Load FFTW wisdom from a file, if one is named and present.

    Explicit, never implicit: a plan built from someone else's wisdom is a plan whose
    provenance is not in the run's metadata. ``None`` reads the ``SOLPROP_FFTW_WISDOM``
    environment variable, and an absent or unreadable file is not an error -- planning
    simply starts from nothing.

    Parameters
    ----------
    path : str or os.PathLike, optional
        The wisdom file. ``None`` takes ``SOLPROP_FFTW_WISDOM``.

    Returns
    -------
    bool
        Whether wisdom was loaded.
    """
    named = os.environ.get("SOLPROP_FFTW_WISDOM") if path is None else str(path)
    if not named or not Path(named).is_file():
        return False
    stored = Path(named).read_bytes().split(b"\0\0")
    if len(stored) != 3:
        logger.warning("numba_cpu: %s is not FFTW wisdom; ignoring it", named)
        return False
    return any(pyfftw.import_wisdom(tuple(stored)))


def export_wisdom(path: str | os.PathLike[str] | None = None) -> bool:
    """Write FFTW's accumulated wisdom to a file, if one is named.

    Parameters
    ----------
    path : str or os.PathLike, optional
        The wisdom file. ``None`` takes ``SOLPROP_FFTW_WISDOM``.

    Returns
    -------
    bool
        Whether wisdom was written.
    """
    named = os.environ.get("SOLPROP_FFTW_WISDOM") if path is None else str(path)
    if not named:
        return False
    Path(named).parent.mkdir(parents=True, exist_ok=True)
    Path(named).write_bytes(b"\0\0".join(pyfftw.export_wisdom()))
    return True


def wisdom_digest() -> str:
    """Return a short digest of the wisdom in force, for the run's metadata.

    Two runs whose plans were chosen from different wisdom are not the same execution,
    and the digest is what makes that visible in a ledger entry.

    Returns
    -------
    str
        The first sixteen hex characters of a BLAKE2b over the exported wisdom.
    """
    import hashlib  # noqa: PLC0415 - only ever wanted when a run is being recorded

    digest = hashlib.blake2b(b"\0\0".join(pyfftw.export_wisdom()), digest_size=8)
    return digest.hexdigest()


@dataclass
class TransformPlans:
    """Every plan one problem needs, and the aligned buffers they are bound to.

    A plan owns its buffers: ``pyfftw.FFTW.execute`` reads and writes the arrays the
    plan was built on and nothing else, so the stage driver fills these in place rather
    than passing arrays in.

    Attributes
    ----------
    family : str
        Which column the plans were built for.
    modal_spectrum : numpy.ndarray
        The padded fine spectrum the inverse transform reads, complex.
    modal_time : numpy.ndarray
        The modal field on the fine time grid, real for ``field_real`` and complex
        otherwise.
    projected : numpy.ndarray
        The modal polarisation on the fine time grid, real except on an envelope grid.
    projected_spectrum : numpy.ndarray
        Its one-sided (or full, on an envelope grid) spectrum, complex.
    to_time : pyfftw.FFTW
        Padded spectrum to modal time.
    to_frequency : pyfftw.FFTW
        Modal polarisation back to the fine spectrum.
    coarse_spectrum, coarse_time : numpy.ndarray
        The absorber's buffers on the coarse grid.
    absorber_to_time, absorber_to_frequency : pyfftw.FFTW
        The absorber's pair, or ``None`` when the problem has no absorber.
    in_stage_time, in_stage_spectrum : numpy.ndarray or None
        The in-stage absorber's term on the fine time grid (real except on an
        envelope grid) and its spectrum, a second output beside the polarisation's
        because both spectra are read by the same stage's finish.
    in_stage_to_frequency : pyfftw.FFTW or None
        The in-stage term's forward transform, or ``None`` when no in-stage absorber
        is on.
    coarse_projected, coarse_projected_spectrum : numpy.ndarray or None
        The embedded coarse rule's projection and its spectrum, for the quadrature
        alarm. The fine rule's spectrum is already in hand from pass 6 of the last
        stage, so the alarm costs one forward transform where the reference pays two.
    coarse_to_frequency : pyfftw.FFTW or None
        The alarm's transform, or ``None`` when the rule has no embedded pair.
    drive, drive_spectrum, convolved : numpy.ndarray or None
        The Raman buffers on the doubled grid, one set for a whole node chunk.
    raman_to_frequency, raman_to_time : pyfftw.FFTW or None
        The Raman pair on the doubled grid.
    raman_tail_to_frequency, raman_tail_to_time : pyfftw.FFTW or None
        A partial node chunk's pair, bound to prefix views of the same Raman buffers.
        The pair adds no owned array and executes sequentially with the full pair.
    planning_seconds : float
        What planning cost, for the ledger.
    """

    family: str
    modal_spectrum: Any
    modal_time: Any
    projected: Any
    projected_spectrum: Any
    to_time: Any
    to_frequency: Any
    coarse_spectrum: Any = None
    coarse_time: Any = None
    absorber_to_time: Any = None
    absorber_to_frequency: Any = None
    in_stage_time: Any = None
    in_stage_spectrum: Any = None
    in_stage_to_frequency: Any = None
    coarse_projected: Any = None
    coarse_projected_spectrum: Any = None
    coarse_to_frequency: Any = None
    drive: Any = None
    drive_spectrum: Any = None
    convolved: Any = None
    raman_to_frequency: Any = None
    raman_to_time: Any = None
    raman_tail_to_frequency: Any = None
    raman_tail_to_time: Any = None
    planning_seconds: float = 0.0
    buffer_bytes: dict[str, int] = field(default_factory=dict)

    def ledger(self) -> dict[str, int]:
        """Return the bytes each plan buffer holds, by name.

        Returns
        -------
        dict
            Buffer name to bytes, for
            :meth:`solprop.backends.protocol.Backend.memory_ledger`.
        """
        return dict(self.buffer_bytes)


class _MallocStatistics(ctypes.Structure):
    """Mirror the macOS allocator statistics ABI without per-query class allocation."""

    _fields_ = [
        ("blocks_in_use", ctypes.c_uint),
        ("size_in_use", ctypes.c_size_t),
        ("max_size_in_use", ctypes.c_size_t),
        ("size_allocated", ctypes.c_size_t),
    ]


class _MallInfo(ctypes.Structure):
    """Mirror the glibc mallinfo2 ABI, including heap and separately mapped storage."""

    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


def _heap_bytes() -> int:
    """Read allocator-owned bytes around planning on the qualification platforms.

    The process-wide net growth includes FFTW's opaque work areas and plan
    objects. It is a measurement, not an allocation-size or peak-memory bound.

    Returns
    -------
    int
        Current process allocator bytes in use, including large mapped blocks; zero
        where neither interface exists (Windows, musl, glibc before 2.33), so that the
        ledger row measures nothing there and the backend still constructs.
    """
    import sys  # noqa: PLC0415 - platform-specific allocator ABI

    if sys.platform == "win32":
        # ctypes.CDLL(None) raises TypeError there: Windows has no process-wide handle.
        return 0
    allocator = ctypes.CDLL(None)
    if sys.platform == "darwin":
        result = _MallocStatistics()
        allocator.malloc_zone_statistics(None, ctypes.byref(result))
        return int(result.size_in_use)
    if sys.platform.startswith("linux") and hasattr(allocator, "mallinfo2"):
        allocator.mallinfo2.restype = _MallInfo
        result = allocator.mallinfo2()
        return int(result.uordblks + result.hblkhd)
    return 0


def _plan(source: Any, target: Any, *, forward: bool, flag: str, threads: int) -> Any:
    """Build one FFTW plan over the last axis.

    Parameters
    ----------
    source, target : numpy.ndarray
        Aligned buffers, allocated by :func:`pyfftw.empty_aligned`.
    forward : bool
        Whether the transform runs time to frequency.
    flag : str
        The FFTW planning flag.
    threads : int
        FFTW's pool for this plan.

    Returns
    -------
    pyfftw.FFTW
        The plan, bound to those buffers.
    """
    return pyfftw.FFTW(
        source,
        target,
        axes=(-1,),
        direction="FFTW_FORWARD" if forward else "FFTW_BACKWARD",
        flags=(flag,),
        threads=threads,
    )


def _plan_raman_tail(
    plans: TransformPlans, count: int, flag: str, threads: int
) -> None:
    """Bind the optional Raman tail pair to prefix views of its owned arrays.

    Parameters
    ----------
    plans : TransformPlans
        Owner of the full Raman arrays and the new prefix plans.
    count : int
        Validated tail width, or zero to leave the pair absent.
    flag : str
        Resolved FFTW planning flag.
    threads : int
        Positive thread count for the tail pair.
    """
    if count:
        drive, spectrum, convolved = plans.drive, plans.drive_spectrum, plans.convolved
        assert drive is not None and spectrum is not None and convolved is not None
        plans.raman_tail_to_frequency = _plan(
            drive[:count], spectrum[:count], forward=True, flag=flag, threads=threads
        )
        plans.raman_tail_to_time = _plan(
            spectrum[:count],
            convolved[:count],
            forward=False,
            flag=flag,
            threads=threads,
        )


def _forward_buffers(
    n_columns: int, n_to: int, *, complex_time: bool
) -> tuple[Any, Any]:
    """Allocate the aligned pair a forward transform of fine columns is planned on.

    Parameters
    ----------
    n_columns : int
        Columns transformed together.
    n_to : int
        Fine time samples.
    complex_time : bool
        Whether the samples are complex (an envelope grid) rather than real.

    Returns
    -------
    tuple of numpy.ndarray
        The time samples, ``(n_columns, n_to)``, and their one-sided (or, for complex
        samples, full) spectrum, complex.
    """
    n_omega_fine = n_to if complex_time else n_to // 2 + 1
    return (
        pyfftw.empty_aligned(
            (n_columns, n_to), "complex128" if complex_time else "float64"
        ),
        pyfftw.empty_aligned((n_columns, n_omega_fine), "complex128"),
    )


def build_plans(  # noqa: PLR0913 - one plan set has this many dimensions
    family: ColumnFamily,
    n_columns: int,
    n_to: int,
    n_t: int,
    *,
    flag: str = "measure",
    threads: int = 1,
    raman_threads: int | None = None,
    chunk_columns: int = 0,
    tail_columns: int = 0,
    raman_tail_threads: int | None = None,
    absorber: bool = False,
    in_stage_absorber: bool = False,
    embedded: bool = False,
) -> TransformPlans:
    """Allocate the aligned buffers and plan every transform on them.

    Parameters
    ----------
    family : str
        Which response column the problem compiles to; see :data:`ColumnFamily`.
    n_columns : int
        Modal columns transformed together, ``n_modes * n_pol``.
    n_to : int
        Fine time samples.
    n_t : int
        Coarse time samples, which the temporal absorber works on.
    flag : str, optional
        A key of :data:`PLAN_FLAGS`. ``"measure"`` by default.
    threads : int, optional
        FFTW's pool for the modal, embedded and absorber plans, and the Raman pair
        when ``raman_threads`` is not supplied.
    raman_threads : int, optional
        Positive FFTW thread count for the doubled-grid Raman pair. ``None`` uses
        ``threads``; a backend sizes this pair from its own batched operand.
    chunk_columns : int, optional
        Columns in one node chunk. Non-zero adds the Raman pair on the **doubled** grid,
        one pair for the whole chunk rather than one per column
        (``docs/dev/backends.md`` section 3.3).
    tail_columns : int, optional
        Columns in a partial Raman chunk, or zero when there is no tail. A positive
        count must be smaller than ``chunk_columns``. The persistent tail pair uses
        prefix views of the existing Raman arrays, without extra owned storage.
    raman_tail_threads : int, optional
        Positive FFTW thread count for the tail pair. ``None`` uses the full Raman
        pair's count; a backend sizes the tail from its own doubled-grid batch.
    absorber : bool, optional
        Whether to plan the coarse-grid pair the temporal absorber needs.
    in_stage_absorber : bool, optional
        Whether to plan the fine-grid forward transform of ADR-025's in-stage
        absorber term, on buffers of its own.
    embedded : bool, optional
        Whether to plan the extra forward transform the quadrature alarm needs, which
        only a rule with an embedded coarse pair has.

    Returns
    -------
    TransformPlans
        The plans, their buffers, and what planning cost.

    Raises
    ------
    BackendError
        If ``flag`` is not one of :data:`PLAN_FLAGS`, an explicit Raman thread count
        is not positive, or the tail width is outside ``[0, chunk_columns)`` except
        that both widths may be zero when there is no Raman pair.
    """
    if flag not in PLAN_FLAGS:
        usable = ", ".join(sorted(PLAN_FLAGS))
        msg = f"{flag!r} is not a planning flag; use one of: {usable}"
        raise BackendError(msg)
    if raman_threads is not None and raman_threads < 1:
        msg = f"raman_threads must be a positive thread count, got {raman_threads}"
        raise BackendError(msg)
    if raman_tail_threads is not None and raman_tail_threads < 1:
        msg = (
            "raman_tail_threads must be a positive thread count, "
            f"got {raman_tail_threads}"
        )
        raise BackendError(msg)
    if tail_columns < 0 or (tail_columns and tail_columns >= chunk_columns):
        msg = "tail_columns must be zero or smaller than a positive chunk_columns"
        raise BackendError(msg)
    raman_pool = threads if raman_threads is None else raman_threads
    tail_pool = raman_pool if raman_tail_threads is None else raman_tail_threads
    chosen = PLAN_FLAGS[flag]
    real_field = family == "field_real"
    complex_time = family == "envelope"
    n_omega_fine = n_to if complex_time else n_to // 2 + 1
    time_dtype = "float64" if real_field else "complex128"

    modal_spectrum = pyfftw.empty_aligned((n_columns, n_omega_fine), "complex128")
    if family == "field_analytic":
        # The analytic signal is the complex inverse of the same one-sided spectrum with
        # its positive bins doubled, so the *input* buffer is a full-length complex
        # array rather than the half spectrum. That is what removes the reference's
        # separate Hilbert transform per node chunk rather than merely moving it.
        modal_spectrum = pyfftw.empty_aligned((n_columns, n_to), "complex128")
    modal_time = pyfftw.empty_aligned((n_columns, n_to), time_dtype)
    projected, projected_spectrum = _forward_buffers(
        n_columns, n_to, complex_time=complex_time
    )
    coarse_spectrum = coarse_time = None
    drive = drive_spectrum = convolved = None
    coarse_projected, coarse_projected_spectrum = (
        _forward_buffers(n_columns, n_to, complex_time=complex_time)
        if embedded
        else (None, None)
    )
    in_stage_time, in_stage_spectrum = (
        _forward_buffers(n_columns, n_to, complex_time=complex_time)
        if in_stage_absorber
        else (None, None)
    )
    if absorber:
        coarse_spectrum = pyfftw.empty_aligned(
            (n_columns, n_t if complex_time else n_t // 2 + 1), "complex128"
        )
        coarse_time = pyfftw.empty_aligned(
            (n_columns, n_t), "complex128" if complex_time else "float64"
        )
    if chunk_columns:
        drive = pyfftw.empty_aligned((chunk_columns, 2 * n_to), "float64")
        drive_spectrum = pyfftw.empty_aligned((chunk_columns, n_to + 1), "complex128")
        convolved = pyfftw.empty_aligned((chunk_columns, 2 * n_to), "float64")

    heap_before = _heap_bytes()
    started = time.perf_counter()
    plans = TransformPlans(
        family=family,
        modal_spectrum=modal_spectrum,
        modal_time=modal_time,
        projected=projected,
        projected_spectrum=projected_spectrum,
        to_time=_plan(
            modal_spectrum, modal_time, forward=False, flag=chosen, threads=threads
        ),
        to_frequency=_plan(
            projected,
            projected_spectrum,
            forward=True,
            flag=chosen,
            threads=threads,
        ),
        coarse_projected=coarse_projected,
        coarse_projected_spectrum=coarse_projected_spectrum,
        coarse_to_frequency=None
        if coarse_projected is None
        else _plan(
            coarse_projected,
            coarse_projected_spectrum,
            forward=True,
            flag=chosen,
            threads=threads,
        ),
        coarse_spectrum=coarse_spectrum,
        coarse_time=coarse_time,
        absorber_to_time=None
        if coarse_spectrum is None
        else _plan(
            coarse_spectrum, coarse_time, forward=False, flag=chosen, threads=threads
        ),
        absorber_to_frequency=None
        if coarse_time is None
        else _plan(
            coarse_time, coarse_spectrum, forward=True, flag=chosen, threads=threads
        ),
        in_stage_time=in_stage_time,
        in_stage_spectrum=in_stage_spectrum,
        in_stage_to_frequency=None
        if in_stage_time is None
        else _plan(
            in_stage_time, in_stage_spectrum, forward=True, flag=chosen, threads=threads
        ),
        drive=drive,
        drive_spectrum=drive_spectrum,
        convolved=convolved,
        raman_to_frequency=None
        if drive is None
        else _plan(
            drive, drive_spectrum, forward=True, flag=chosen, threads=raman_pool
        ),
        raman_to_time=None
        if convolved is None
        else _plan(
            drive_spectrum, convolved, forward=False, flag=chosen, threads=raman_pool
        ),
    )
    _plan_raman_tail(plans, tail_columns, chosen, tail_pool)
    if drive is not None:
        # Planning may overwrite inputs. Establish padding only after every full
        # and tail plan is built; subsequent r2c executions preserve this input.
        drive[:] = 0.0
    plan_heap_growth = max(0, _heap_bytes() - heap_before)
    plans.planning_seconds = time.perf_counter() - started
    # Every buffer is named even when it is empty, which is the reference's convention
    # and what lets one budget test read either backend's ledger.
    plans.buffer_bytes = {
        name: 0 if buffer is None else int(np.asarray(buffer).nbytes)
        for name, buffer in (
            ("fine_spectrum", modal_spectrum),
            ("modal_time", modal_time),
            ("projected", projected),
            ("projected_spectrum", projected_spectrum),
            ("absorber_spectrum", coarse_spectrum),
            ("absorber_time", coarse_time),
            ("in_stage_absorber_time", in_stage_time),
            ("in_stage_absorber_spectrum", in_stage_spectrum),
            ("projected_coarse", coarse_projected),
            ("projected_coarse_spectrum", coarse_projected_spectrum),
            ("node_raman_drive", drive),
            ("node_raman_spectrum", drive_spectrum),
            ("node_raman_convolved", convolved),
        )
    }
    plans.buffer_bytes["fft_plan_heap_growth"] = plan_heap_growth
    return plans


@dataclass
class GeneralPlans:
    """The general modal route's plans and the aligned buffers they are bound to.

    The fourth plan family: the general route synthesises in
    the frequency domain and transforms **three Cartesian components per node**, a
    complex inverse of the doubled one-sided spectrum that is the analytic signal,
    where the three column families above transform modal columns once per stage.
    Every buffer is a node chunk wide, rows ``3 q + c``; the delayed response adds a
    pair on the doubled axis, rows ``6 q + e`` over the six stored pair histories.
    A partial last chunk has its own plans bound to prefix views of the same
    buffers, so it transforms its own rows and no more, and owns nothing.

    Attributes
    ----------
    spectrum : numpy.ndarray
        The synthesised fine spectrum, complex, ``(3 chunk, n_to)``.
    signal : numpy.ndarray
        Its inverse, the analytic signal at the chunk's nodes, complex.
    polarisation : numpy.ndarray
        The real polarisation on the fine time grid, float64, ``(3 chunk, n_to)``.
    polarisation_spectrum : numpy.ndarray
        Its one-sided spectrum, complex, ``(3 chunk, n_to // 2 + 1)``.
    to_time, to_frequency : pyfftw.FFTW
        The chunk's pair.
    tail_to_time, tail_to_frequency : pyfftw.FFTW or None
        A partial last chunk's pair, or ``None`` when the chunk divides the nodes.
    drive, drive_spectrum, product, history : numpy.ndarray or None
        The delayed response's buffers on the doubled axis, ``(6 chunk, 2 n_to)``
        real, ``(6 chunk, n_to + 1)`` complex twice and ``(6 chunk, 2 n_to)`` real;
        ``None`` when no region declares a delayed response. The upper half of
        `drive` is zero and stays zero: the real-to-complex transform preserves its
        input and nothing else writes there.
    drive_to_frequency, history_to_time : pyfftw.FFTW or None
        The doubled pair, ``None`` with the buffers.
    tail_drive_to_frequency, tail_history_to_time : pyfftw.FFTW or None
        Their partial-chunk pair.
    planning_seconds : float
        What planning cost.
    buffer_bytes : dict
        Bytes each named buffer holds, and the allocator's growth over planning.
    """

    spectrum: Any
    signal: Any
    polarisation: Any
    polarisation_spectrum: Any
    to_time: Any
    to_frequency: Any
    tail_to_time: Any = None
    tail_to_frequency: Any = None
    drive: Any = None
    drive_spectrum: Any = None
    product: Any = None
    history: Any = None
    drive_to_frequency: Any = None
    history_to_time: Any = None
    tail_drive_to_frequency: Any = None
    tail_history_to_time: Any = None
    planning_seconds: float = 0.0
    buffer_bytes: dict[str, int] = field(default_factory=dict)


def _plan_pair(
    backward: tuple[Any, Any],
    forward: tuple[Any, Any],
    count: int,
    flag: str,
    threads: int,
) -> tuple[Any, Any]:
    """Plan one backward and one forward transform over the leading rows.

    Parameters
    ----------
    backward : tuple
        The backward transform's source and target buffers.
    forward : tuple
        The forward transform's source and target buffers.
    count : int
        How many leading rows the pair transforms, positive.
    flag : str
        The resolved FFTW planning flag.
    threads : int
        FFTW's pool for both plans.

    Returns
    -------
    tuple
        The backward plan and the forward plan, bound to prefix views.
    """
    return (
        _plan(
            backward[0][:count],
            backward[1][:count],
            forward=False,
            flag=flag,
            threads=threads,
        ),
        _plan(
            forward[0][:count],
            forward[1][:count],
            forward=True,
            flag=flag,
            threads=threads,
        ),
    )


def build_general_plans(
    chunk_nodes: int,
    tail_nodes: int,
    n_to: int,
    *,
    delayed: bool,
    flag: str = "measure",
    threads: int = 1,
) -> GeneralPlans:
    """Allocate the general route's aligned buffers and plan every transform on them.

    Planning is destructive, so every buffer is zeroed after the last plan is built
    and before a state ever reaches one.

    Parameters
    ----------
    chunk_nodes : int
        Nodes a chunk holds, positive.
    tail_nodes : int
        Nodes of a partial last chunk, or zero when the chunk divides the node count.
        Smaller than `chunk_nodes`.
    n_to : int
        Fine time samples, positive and even.
    delayed : bool
        Whether to add the delayed response's pair on the doubled axis.
    flag : str, optional
        A key of :data:`PLAN_FLAGS`. ``"measure"`` by default.
    threads : int, optional
        FFTW's pool for every plan of the family, positive.

    Returns
    -------
    GeneralPlans
        The plans, their buffers, and what planning cost.

    Raises
    ------
    BackendError
        If ``flag`` is not one of :data:`PLAN_FLAGS` or the tail is not smaller than
        the chunk.
    """
    if flag not in PLAN_FLAGS:
        usable = ", ".join(sorted(PLAN_FLAGS))
        msg = f"{flag!r} is not a planning flag; use one of: {usable}"
        raise BackendError(msg)
    if not 0 <= tail_nodes < chunk_nodes:
        msg = (
            f"the general route's tail of {tail_nodes} node(s) must be smaller than "
            f"its chunk of {chunk_nodes}"
        )
        raise BackendError(msg)
    chosen = PLAN_FLAGS[flag]
    rows, tail = 3 * chunk_nodes, 3 * tail_nodes
    spectrum = pyfftw.empty_aligned((rows, n_to), "complex128")
    field_time = pyfftw.empty_aligned((rows, n_to), "complex128")
    polarisation = pyfftw.empty_aligned((rows, n_to), "float64")
    polarisation_spectrum = pyfftw.empty_aligned((rows, n_to // 2 + 1), "complex128")
    heap_before = _heap_bytes()
    started = time.perf_counter()

    to_time, to_frequency = _plan_pair(
        (spectrum, field_time),
        (polarisation, polarisation_spectrum),
        rows,
        chosen,
        threads,
    )
    plans = GeneralPlans(
        spectrum=spectrum,
        signal=field_time,
        polarisation=polarisation,
        polarisation_spectrum=polarisation_spectrum,
        to_time=to_time,
        to_frequency=to_frequency,
    )
    if tail:
        plans.tail_to_time, plans.tail_to_frequency = _plan_pair(
            (spectrum, field_time),
            (polarisation, polarisation_spectrum),
            tail,
            chosen,
            threads,
        )
    if delayed:
        plans.drive = pyfftw.empty_aligned((2 * rows, 2 * n_to), "float64")
        plans.drive_spectrum = pyfftw.empty_aligned((2 * rows, n_to + 1), "complex128")
        plans.product = pyfftw.empty_aligned((2 * rows, n_to + 1), "complex128")
        plans.history = pyfftw.empty_aligned((2 * rows, 2 * n_to), "float64")
        plans.history_to_time, plans.drive_to_frequency = _plan_pair(
            (plans.product, plans.history),
            (plans.drive, plans.drive_spectrum),
            2 * rows,
            chosen,
            threads,
        )
        if tail:
            plans.tail_history_to_time, plans.tail_drive_to_frequency = _plan_pair(
                (plans.product, plans.history),
                (plans.drive, plans.drive_spectrum),
                2 * tail,
                chosen,
                threads,
            )
    buffers = (
        ("general_node_spectrum", spectrum),
        ("general_node_field", field_time),
        ("general_node_polarisation", polarisation),
        ("general_node_source", polarisation_spectrum),
        ("general_node_drive", plans.drive),
        ("general_node_drive_spectrum", plans.drive_spectrum),
        ("general_node_product", plans.product),
        ("general_node_histories", plans.history),
    )
    for _, buffer in buffers:
        if buffer is not None:
            buffer[:] = 0.0
    plans.planning_seconds = time.perf_counter() - started
    plans.buffer_bytes = {
        name: 0 if buffer is None else int(buffer.nbytes) for name, buffer in buffers
    }
    plans.buffer_bytes["general_fft_plan_heap_growth"] = max(
        0, _heap_bytes() - heap_before
    )
    return plans


@dataclass(frozen=True)
class TransversePlans:
    """Own one Cartesian scratch and its persistent in-place FFTW pair.

    Attributes
    ----------
    spectrum : numpy.ndarray
        Complex scratch shaped (n_y, n_x, n_pol, n_omega), with x fastest
        after flattening the first two axes.
    forward, inverse : pyfftw.FFTW
        Unnormalised transverse plans over the first two axes.
    plan_heap_growth : int
        Process allocator net growth during planning in bytes, including opaque FFTW
        work areas. A measured count, not a portable bound or allocator high-water mark.
    planning_seconds : float
        Time spent constructing the pair, in seconds.
    """

    spectrum: Any
    forward: Any
    inverse: Any
    plan_heap_growth: int
    planning_seconds: float


def build_transverse_plans(
    shape: tuple[int, int, int, int],
    *,
    flag: str,
    threads: int,
    spatial_contiguous: bool = False,
) -> TransversePlans:
    """Plan an in-place complex transverse FFT pair before loading any state.

    Parameters
    ----------
    shape : tuple of int
        Scratch dimensions (n_y, n_x, n_pol, n_omega), all positive.
    flag : str
        Existing FFTW planning choice: estimate, measure or patient.
    threads : int
        FFTW worker count, positive.
    spatial_contiguous : bool, optional
        Store spatial/polarisation columns contiguously behind the public view.
        Envelope transforms benefit from this layout; field columns retain their
        contiguous temporal-frequency axis. Both use exactly one owning scratch.

    Returns
    -------
    TransversePlans
        Persistent pair sharing exactly one full coarse spectral scratch.

    Raises
    ------
    BackendError
        If the planning flag is unsupported.
    """
    if flag not in PLAN_FLAGS:
        msg = f"{flag!r} is not a planning flag"
        raise BackendError(msg)
    started = time.perf_counter()
    if spatial_contiguous:
        n_y, n_x, n_pol, n_omega = shape
        storage = pyfftw.empty_aligned((n_omega, n_y, n_x, n_pol), "complex128")
        spectrum = storage.transpose(1, 2, 3, 0)
    else:
        spectrum = pyfftw.empty_aligned(shape, "complex128")
    heap_before = _heap_bytes()
    forward = pyfftw.FFTW(
        spectrum,
        spectrum,
        axes=(0, 1),
        direction="FFTW_FORWARD",
        flags=(PLAN_FLAGS[flag],),
        threads=threads,
    )
    inverse = pyfftw.FFTW(
        spectrum,
        spectrum,
        axes=(0, 1),
        direction="FFTW_BACKWARD",
        flags=(PLAN_FLAGS[flag],),
        threads=threads,
    )
    return TransversePlans(
        spectrum,
        forward,
        inverse,
        max(0, _heap_bytes() - heap_before),
        time.perf_counter() - started,
    )
