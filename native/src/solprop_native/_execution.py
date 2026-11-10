# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A workspace's CPU execution settings and the built-in table (solprop SEP 46).

Every native workspace chooses the width of each CPU operation from an
:class:`ExecutionRecord` copied into it when it is built (SEP 46 section 5's *workspace
settings*). Native code holds no values for them: they come from this module's
built-in table, from a machine calibration's profile, or from both, and the record a
workspace applied is read back into a run's metadata.

This module is standard library only and loads nothing of the package, so solprop's
calibration can read it by path without starting the native runtime.

Work is counted per operation class: elements for stages and copies, active columns *
fine samples for responses, rows * inner * samples for dense products, active bins *
modes^3 for matrix functions, elements * history rows for the exponential Adams
trial's passes, and actual real-side batch samples for FFT and Raman plans (tails
included). Below an operation's threshold it runs on one thread, below its full size
on the medium width, and from it on its pool's capacity; a batch of transforms whose
row length is known threads instead by its rows (``ExecutionRecord.fft_thread_elements``
and ``fft_long_row``, solprop issue #587). solprop's
``docs/dev/benchmarks.md``, section "Kokkos CPU operation-size threading", records the
windows and sweeps behind each built-in value; ``docs/dev/cpu-performance.md`` gives
the table's provenance.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

__all__ = [
    "OPERATIONS",
    "SPIN_DIVISOR",
    "SPIN_MAX_TEAM",
    "ExecutionRecord",
    "builtin",
]

OPERATIONS = (
    "stage",
    "response",
    "dense",
    "fft",
    "raman",
    "copy",
    "flow",
    "modal",
    "adams",
)
"""The CPU operation classes, in the native core's order (``CpuOperation``)."""

SPIN_DIVISOR = 16
"""What the built-in thresholds are divided by on Apple silicon's spinning top tier.

The thresholds were swept with idle threads asleep, a fork and join of 16 threads
costing 100-200 us. On Apple silicon, while LLVM's runtime spins within native calls
(solprop ADR-071) on a team of at most :data:`SPIN_MAX_TEAM` threads, a region costs
about a microsecond. On an Apple M5 Max at six threads, on top of the spin, a sixteenth
ran the Raman GNLSE in 0.44 of its window, Raman hydrogen in 0.89, comb-to-continuum in
0.92 and the tracking cases in 0.87 on geometric mean, none slower; its top-tier cluster
shares one L2 cache. On two Intel Xeon Gold 6240, whose cores share only the L3, a
sixteenth ran the averaged argon and hydrogen 1.4-1.7 times slower at 4, 8, 18 and 36
threads with either runtime, and a quarter lost on 16 cores of an AMD EPYC 7763
(ADR-073). The grain belongs to the machine, which is what a calibration measures.
"""

SPIN_MAX_TEAM = 8
"""The widest team on which :data:`SPIN_DIVISOR` applies."""


@dataclass(frozen=True)
class ExecutionRecord:
    """One workspace's CPU execution settings, as the native core reads them.

    Attributes
    ----------
    stage_threshold, response_threshold, dense_threshold, fft_threshold,
    raman_threshold, copy_threshold, flow_threshold, modal_threshold,
    adams_threshold : int
        Each operation class's work below which it runs on one thread. The FFT
        class's is where a modal or spatial FFTW plan starts to thread.
    stage_full, response_full, dense_full, fft_full, raman_full, copy_full, flow_full,
    modal_full, adams_full : int
        Each class's work from which it takes its pool's capacity, at least its
        threshold; between the two it takes ``medium_threads``.
    response_full_samples, response_full_columns : int
        A response of this many fine samples, or of this many columns, takes the
        full pool at any size above its threshold.
    medium_threads : int
        The medium width.
    compact_threads : int
        The team of a compact problem (one column on one node with fewer than
        ``compact_max_fine_samples`` fine samples): its kernels run on at most this
        many threads and its threaded transforms on that whole team.
    compact_max_fine_samples : int
        The fine samples below which a one-column problem is compact.
    stable_kernel_team : bool
        Whether a medium-width kernel takes the full width once its pool holds a
        full-width instance.
    fft_thread_elements, fft_long_row : int
        A transform batch from ``fft_threshold`` up to ``fft_full`` takes a thread per
        ``fft_thread_elements`` samples and at most one a row, or ``medium_threads``
        if more when its rows are at least ``fft_long_row`` samples long (solprop
        issue #587); from ``fft_full`` its pool's capacity.
    """

    stage_threshold: int
    stage_full: int
    response_threshold: int
    response_full: int
    dense_threshold: int
    dense_full: int
    fft_threshold: int
    fft_full: int
    raman_threshold: int
    raman_full: int
    copy_threshold: int
    copy_full: int
    flow_threshold: int
    flow_full: int
    modal_threshold: int
    modal_full: int
    adams_threshold: int
    adams_full: int
    response_full_samples: int
    response_full_columns: int
    medium_threads: int
    compact_threads: int
    compact_max_fine_samples: int
    stable_kernel_team: bool
    fft_thread_elements: int
    fft_long_row: int

    def __post_init__(self) -> None:
        """Refuse a record the native core would refuse."""
        for name in OPERATIONS:
            threshold = getattr(self, f"{name}_threshold")
            full = getattr(self, f"{name}_full")
            if not _count(threshold) or not _count(full) or full < threshold:
                raise ValueError(
                    f"{name}: a threshold and full size must be positive integers, "
                    f"the full size at least the threshold; got {threshold!r} and "
                    f"{full!r}"
                )
        for name in (
            "response_full_samples",
            "response_full_columns",
            "medium_threads",
            "compact_threads",
            "compact_max_fine_samples",
            "fft_thread_elements",
            "fft_long_row",
        ):
            if not _count(getattr(self, name)):
                raise ValueError(
                    f"{name} must be a positive integer, got {getattr(self, name)!r}"
                )
        # A record read from JSON may hold anything here.
        if not isinstance(cast(object, self.stable_kernel_team), bool):
            raise ValueError(
                f"stable_kernel_team must be a bool, got {self.stable_kernel_team!r}"
            )

    def native(self) -> dict[str, int | bool]:
        """Return the record as the native core's ``cpu_execution`` field reads it."""
        return dataclasses.asdict(self)

    @classmethod
    def from_native(cls, values: Mapping[str, object]) -> ExecutionRecord:
        """Read a record written by :meth:`native` or read back from a state.

        Raises
        ------
        ValueError
            When a field is missing, unknown or out of range.
        """
        names = {field.name for field in dataclasses.fields(cls)}
        if set(values) != names:
            missing = sorted(names - set(values))
            unknown = sorted(set(values) - names)
            raise ValueError(
                f"a CPU execution record's fields differ: missing {missing}, "
                f"unknown {unknown}"
            )
        return cls(**values)  # pyright: ignore[reportArgumentType] - checked above

    def updated(self, values: Mapping[str, object]) -> ExecutionRecord:
        """Return the record with `values`' fields replaced, validated.

        A profile row holds only the settings it measured; the rest stay as they are.

        Raises
        ------
        ValueError
            When `values` names a field the record does not hold, or a value is out
            of range.
        """
        names = {field.name for field in dataclasses.fields(self)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"unknown CPU execution settings {unknown}")
        return dataclasses.replace(self, **values)  # pyright: ignore[reportArgumentType] - checked above

    def with_fft_start(self, start: int) -> ExecutionRecord:
        """Return the record with a user's FFT start, which no profile overrides.

        Its full size stays where it is, or moves up to the start.
        """
        return dataclasses.replace(
            self, fft_threshold=start, fft_full=max(start, self.fft_full)
        )

    def differences(self, other: ExecutionRecord) -> dict[str, tuple[object, object]]:
        """Return each field whose value differs from `other`'s, as (this, other)."""
        mine, theirs = self.native(), other.native()
        return {
            name: (value, theirs[name])
            for name, value in mine.items()
            if value != theirs[name]
        }


def _count(value: object) -> bool:
    """Whether `value` is a positive integer and not a bool."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def builtin(
    system: str,
    *,
    spin_milliseconds: int = 0,
    width: int = 1,
    fft_threshold: int | None = None,
) -> ExecutionRecord:
    """Return the built-in table's record for a process.

    The table is keyed by operating system (`system`, as :func:`platform.system`
    names it): macOS takes its own stage and FFT starts and no stable kernel team;
    every other system takes the x86 Linux values, which were not measured on Windows
    or Linux aarch64 (solprop issue #328). On macOS a process whose LLVM runtime spins
    within native calls (`spin_milliseconds` > 0) on a team of at most
    :data:`SPIN_MAX_TEAM` threads (`width`) divides every threshold and full size by
    :data:`SPIN_DIVISOR` (solprop ADR-073). `fft_threshold`, from
    ``SOLPROP_NATIVE_THREADED_FFT_MIN_SAMPLES``, replaces the FFT start and is never
    divided.

    Parameters
    ----------
    system : str
        ``"Darwin"``, ``"Linux"``, ``"Windows"`` or another :func:`platform.system`
        name.
    spin_milliseconds : int
        LLVM's block time inside native calls, zero when the runtime does not spin
        there.
    width : int
        The runtime's thread count.
    fft_threshold : int or None
        The user's FFT start, or ``None``.

    Returns
    -------
    ExecutionRecord
        The record a workspace of this process takes without a profile.

    Examples
    --------
    >>> builtin("Linux").stage_threshold
    4096
    >>> builtin("Darwin", spin_milliseconds=1, width=6).stage_threshold
    512
    """
    apple = system == "Darwin"
    # The stage boundary is platform-dependent. At 8192, just under the calibrated
    # 8196-element RDW stage, the same case on a 2-3-5-smooth grid (n_t 4000, 8004
    # elements) ran its stage entry serially on Linux: 708 us against 59 us at full
    # width on an 18-core Cascade Lake socket, 1.8 times the whole evaluation of n_t
    # 4096. At 4096 (with the FFT start at 16384) n_t 4000 costs no more than 4096
    # there and on Zen 3; at 1024 a 1351-element SSFS stage regressed. On an Apple
    # arm64 laptop the serial stage is the faster one at 8004 elements (and 4096's
    # threaded one is already dearer than 4000's serial), so macOS keeps 8192. An
    # 8196-element RDW stage needs the full pool in the server calibration; putting its
    # reductions on a separate width loses that gain, so the stage's full size is its
    # threshold.
    stage = 8192 if apple else 4096
    # The FFT start. On Linux at 32768 the four-mode RDW's 32000-sample coarse batch
    # (n_t 4000) ran serially, its transforms 2.5-3.2 times slower on an 18-core
    # Cascade Lake socket than the 32768-sample batch's four-thread ones. Above the
    # start a batch threads by its shape (solprop issue #587): FFTW divides its rows
    # among the threads and splits a row only where it is long. Timed alone, batched
    # real transforms of 1-7 rows ran fastest at about a thread a row on an Intel
    # Xeon Gold 6240, 16 cores of an AMD EPYC 7763 and an Apple M5 Max, the full
    # width 1.1-1.6 times slower; many rows of a few thousand samples (128 x 576,
    # 16 x 6144) wanted 8-18 threads, where four threads were 2-3 times slower; rows
    # gained from being split, at about four threads, from 16384 samples on Linux
    # (4 x 3072 also gained 1.3-2.9 times from three threads, so the start is 8192).
    # On the M5 Max's spinning top tier the start, thread grain and long row are a
    # sixteenth of the values below: rows split from 4608 samples, at four threads.
    fft = 32768 if apple else 8192
    fft_thread_elements = 32768 if apple else 4096
    fft_long_row = 65536 if apple else 16384
    record = ExecutionRecord(
        stage_threshold=stage,
        stage_full=stage,
        # Responses keep four threads for many short columns (33 x 4096), while long
        # 65536-sample columns take the full pool without threading small copy
        # kernels. Wider chunks also need the full pool: switching between a
        # four-thread response and a full-width FFT regressed the Cartesian filament on
        # the server. The column boundary keeps the measured 33-column regime and
        # includes 44-column tails.
        response_threshold=32768,
        response_full=196608,
        # Native products stay serial below the conservative one-million-work
        # crossover: 2 x 2 x 8192 favours serial on both hosts, whereas 33 x 4 x 8192
        # favours the full pool. Intermediate widths did not establish a
        # complete-window benefit.
        dense_threshold=1048576,
        dense_full=1048576,
        fft_threshold=fft,
        # Every batch this large takes the full pool, whatever its shape.
        fft_full=4194304,
        raman_threshold=65536,
        raman_full=262144,
        # An 8192-element temporal copy must stay serial: threading it changed
        # averaged helium from 0.74 to 1.27 s in the server calibration.
        copy_threshold=65536,
        copy_full=65536,
        # The 38691-work scalar-wall flow needs full width: its serial block functions
        # took 2.39 s against 0.13 s with the full pool in the server window profile.
        flow_threshold=32768,
        flow_full=32768,
        # General-modal synthesis and projection have dedicated team kernels, with
        # work active nodes * components * spectral samples * modes. Their measured
        # full-width threshold stays separate: serialising them regressed complete
        # windows on both hosts, despite the generic small-GEMM crossover above.
        modal_threshold=65536,
        modal_full=65536,
        # The exponential Adams trial's two passes cost about ten stage entries an
        # element, so they thread on fields the stage keeps serial. Swept on
        # single-column capillaries of 401 to 10241 elements with the passes forced to
        # 1-16 threads of a 16-thread process, idle threads sleeping: on 16 cores of a
        # dual EPYC 7643 four threads lost at 4487 (0.8x) and gained 1.1-2.1x from 5607
        # to 71687, while 16 lost to four on helium up to 17927 and to one at 8967; an
        # 18-core Apple M5 Max gained at four from 4487 and at 16 over four only at
        # 71687.
        adams_threshold=8192,
        adams_full=65536,
        response_full_samples=65536,
        response_full_columns=40,
        medium_threads=4,
        # A compact problem's stage of a few thousand to 16384 bins otherwise woke the
        # whole pool for one short pass between serial transforms: on 16 cores of an
        # Intel Xeon Gold 6240 the main thread then ran at about 2.7 instead of 3.5
        # GHz, its arrays came back from other cores' caches, and gnlse_raman_shock
        # took as long on 16 threads as on one (docs/dev/benchmarks.md, "Compact CPU
        # teams").
        compact_threads=8,
        compact_max_fine_samples=65536,
        # GNU OpenMP ends surplus threads when a team narrows, and on Windows, with
        # LLVM's libomp, the wider team still ran h2_raman_averaged 1.1 times faster at
        # eight threads. Not on macOS: on an Apple M5 Max, whose 18 cores are of two
        # kinds, the promoted passes took argon_blueshift from 3.0 to 3.2 s at 18
        # threads.
        stable_kernel_team=not apple,
        fft_thread_elements=fft_thread_elements,
        fft_long_row=fft_long_row,
    )
    if apple and spin_milliseconds > 0 and width <= SPIN_MAX_TEAM:
        record = _divided(record, SPIN_DIVISOR)
    if fft_threshold is not None:
        record = dataclasses.replace(
            record,
            fft_threshold=fft_threshold,
            fft_full=max(fft_threshold, builtin(system).fft_full),
        )
    return record


def _divided(record: ExecutionRecord, divisor: int) -> ExecutionRecord:
    """Divide each threshold and full size; each full stays at least its threshold."""
    values: dict[str, int] = {}
    for name in OPERATIONS:
        threshold = max(1, getattr(record, f"{name}_threshold") // divisor)
        values[f"{name}_threshold"] = threshold
        values[f"{name}_full"] = max(
            threshold, getattr(record, f"{name}_full") // divisor
        )
    for name in ("fft_thread_elements", "fft_long_row"):
        values[name] = max(1, getattr(record, name) // divisor)
    return dataclasses.replace(record, **values)
