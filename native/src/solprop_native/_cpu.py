# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The processor check a distribution build makes before it loads its extension.

A distribution build is compiled for a CPU baseline (``build_info().cpu_baseline``)
and records it in ``_cpu_baseline.txt`` beside the extension. On an older processor
the extension would stop at its first unsupported instruction, so the package reads
that record and refuses to import, with :class:`CpuBaselineError` naming the missing
features. The processor is asked in the way each wheel platform offers:

- **Linux x86-64**: the ``flags`` line of ``/proc/cpuinfo``.
- **Windows x86-64**: ``IsProcessorFeaturePresent`` in ``kernel32``, through
  :mod:`ctypes`. Windows reports SSSE3, SSE4.1, SSE4.2, CMPXCHG16B, XSAVE, AVX, AVX2
  and AVX-512F, and not POPCNT, LAHF/SAHF, BMI1, FMA, F16C, MOVBE or LZCNT; its BMI2
  constant exists only from Windows 11 24H2 and reads as absent before it. So AVX2
  stands for the rest of x86-64-v3: every processor with AVX2 (Intel from Haswell,
  AMD from Excavator) also has BMI1, BMI2, FMA, F16C, MOVBE and LZCNT. SSE4.2 stands
  for POPCNT, 64-bit Windows itself requires LAHF/SAHF, and AVX-512F stands for the
  rest of x86-64-v4. Windows reports SSSE3 to AVX-512F from Windows 10 version 2004;
  an older Windows reads them as absent.
- **Linux aarch64**: the ``Features`` line of ``/proc/cpuinfo``, against what
  ``-mcpu=neoverse-n1`` lets GCC (the Linux wheel's compiler) emit: Armv8.2-A with its
  mandatory CRC32, LSE atomics and rounding doubling multiply, plus the N1's
  half-precision arithmetic, dot product and RCpc loads.
- **macOS arm64**: nothing is checked; every Apple silicon Mac is at least an M1.

Where the processor cannot be asked (no ``/proc/cpuinfo``, another platform), or the
record names a baseline this module does not know (a source build's own ``-march``),
nothing is checked. A missing record is a local or editable build, compiled for the
machine it runs on.

The module is pure Python and imports nothing from the package, so solprop's
automatic backend selection loads it by path and runs the same check without
importing the package or starting its runtime (``solprop.backends.registry``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "BASELINE_RECORD",
    "BUILD_GUIDE",
    "CPUINFO",
    "REQUIRED_FEATURES",
    "CpuBaselineError",
    "baseline_shortfall",
    "check_cpu_baseline",
    "cpuinfo_flags",
    "kernel32_feature_query",
    "missing_features",
    "windows_features",
]

BASELINE_RECORD = "_cpu_baseline.txt"
CPUINFO = Path("/proc/cpuinfo")
BUILD_GUIDE = (
    "https://solprop.readthedocs.io/en/latest/guide/performance/building-native.html"
)

# The x86-64 psABI microarchitecture levels, as Linux names the CPUID features in
# /proc/cpuinfo (``abm`` is LZCNT).
_X86_64_V2 = frozenset({"cx16", "lahf_lm", "popcnt", "sse4_1", "sse4_2", "ssse3"})
_X86_64_V3 = _X86_64_V2 | {
    "abm",
    "avx",
    "avx2",
    "bmi1",
    "bmi2",
    "f16c",
    "fma",
    "movbe",
    "xsave",
}
_X86_64_V4 = _X86_64_V3 | {"avx512bw", "avx512cd", "avx512dq", "avx512f", "avx512vl"}
# What GCC's -mcpu=neoverse-n1 enables (its predefined __ARM_FEATURE_* macros: NEON,
# ATOMICS, CRC32, QRDMX, FP16 scalar and vector arithmetic, DOTPROD, RCPC), as the
# Linux kernel names those hardware capabilities in /proc/cpuinfo.
_NEOVERSE_N1 = frozenset(
    {
        "asimd",
        "asimddp",
        "asimdhp",
        "asimdrdm",
        "atomics",
        "crc32",
        "fp",
        "fphp",
        "lrcpc",
    }
)
REQUIRED_FEATURES: dict[str, frozenset[str]] = {
    "x86-64-v2": _X86_64_V2,
    "x86-64-v3": _X86_64_V3,
    "x86-64-v4": _X86_64_V4,
    "neoverse-n1": _NEOVERSE_N1,
}
"""The processor features each checked baseline requires, in Linux's spelling."""

# kernel32's IsProcessorFeaturePresent constants (winnt.h, as Microsoft's reference
# page for the function lists them), by the Linux name of the feature each reports.
_PF_FEATURES: dict[str, int] = {
    "cx16": 14,  # PF_COMPARE_EXCHANGE128
    "xsave": 17,  # PF_XSAVE_ENABLED
    "ssse3": 36,  # PF_SSSE3_INSTRUCTIONS_AVAILABLE
    "sse4_1": 37,  # PF_SSE4_1_INSTRUCTIONS_AVAILABLE
    "sse4_2": 38,  # PF_SSE4_2_INSTRUCTIONS_AVAILABLE
    "avx": 39,  # PF_AVX_INSTRUCTIONS_AVAILABLE
    "avx2": 40,  # PF_AVX2_INSTRUCTIONS_AVAILABLE
    "avx512f": 41,  # PF_AVX512F_INSTRUCTIONS_AVAILABLE
}
# Features Windows does not report, each stood for by one it does (module docstring).
_WINDOWS_IMPLIED: dict[str, frozenset[str]] = {
    "sse4_2": frozenset({"popcnt"}),
    "avx2": frozenset({"abm", "bmi1", "bmi2", "f16c", "fma", "movbe"}),
    "avx512f": frozenset({"avx512bw", "avx512cd", "avx512dq", "avx512vl"}),
}
# LAHF/SAHF in 64-bit mode: required by 64-bit Windows 8.1 and later.
_WINDOWS_ALWAYS = frozenset({"lahf_lm"})


class CpuBaselineError(ImportError):
    """The processor is below the CPU baseline this build was compiled for.

    Parameters
    ----------
    baseline
        The build's recorded CPU baseline, for example ``"x86-64-v3"``.
    missing
        The required features the processor lacks.

    Attributes
    ----------
    baseline : str
        The recorded CPU baseline.
    missing_features : tuple[str, ...]
        The missing features in Linux's spelling, sorted.
    """

    def __init__(self, baseline: str, missing: Iterable[str]) -> None:
        self.baseline = baseline
        self.missing_features = tuple(sorted(missing))
        super().__init__(
            f"this solprop-native build requires an {baseline} processor, and this "
            f"one lacks {', '.join(self.missing_features)}; build solprop-native from "
            f"source on this machine instead ({BUILD_GUIDE})"
        )


def missing_features(baseline: str, flags: Iterable[str]) -> list[str]:
    """Return the features ``baseline`` requires that ``flags`` lacks, sorted.

    Parameters
    ----------
    baseline
        The recorded CPU baseline, for example ``"x86-64-v3"``.
    flags
        The processor's feature names as ``/proc/cpuinfo`` spells them.

    Returns
    -------
    list[str]
        Missing features; empty when all are present or the baseline is not one this
        module checks.
    """
    required = REQUIRED_FEATURES.get(baseline)
    if required is None:
        return []
    return sorted(required - set(flags))


def cpuinfo_flags(cpuinfo: str, key: str = "flags") -> frozenset[str] | None:
    """Return the first processor's feature names from ``/proc/cpuinfo`` text.

    Parameters
    ----------
    cpuinfo
        The file's contents.
    key
        The line naming them: ``"flags"`` on x86, ``"Features"`` on Arm.

    Returns
    -------
    frozenset[str] | None
        The names, or ``None`` when the text has no such line.
    """
    for line in cpuinfo.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() == key:
            return frozenset(value.split())
    return None


def windows_features(is_present: Callable[[int], object]) -> frozenset[str]:
    """Return the x86-64 features Windows reports, with those they stand for.

    Parameters
    ----------
    is_present
        ``kernel32.IsProcessorFeaturePresent``, or a stand-in: takes a ``PF_*``
        constant and returns a true value when the feature is present.

    Returns
    -------
    frozenset[str]
        The features in Linux's spelling, including those Windows does not report but
        a reported one stands for (module docstring).
    """
    present = {name for name, constant in _PF_FEATURES.items() if is_present(constant)}
    for marker, implied in _WINDOWS_IMPLIED.items():
        if marker in present:
            present |= implied
    return frozenset(present | _WINDOWS_ALWAYS)


def kernel32_feature_query() -> Callable[[int], object]:
    """Return ``kernel32.IsProcessorFeaturePresent``; Windows only.

    Returns
    -------
    collections.abc.Callable
        The function, typed to take a ``DWORD`` and return a ``BOOL``.
    """
    import ctypes  # only a Windows check asks kernel32

    kernel32 = ctypes.WinDLL("kernel32")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    query = kernel32.IsProcessorFeaturePresent  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    query.argtypes = [ctypes.c_uint32]  # pyright: ignore[reportUnknownMemberType]
    query.restype = ctypes.c_int  # pyright: ignore[reportUnknownMemberType]
    return query  # pyright: ignore[reportUnknownVariableType]


def _processor_features(
    baseline: str,
    platform: str,
    cpuinfo: Path,
    feature_query: Callable[[], Callable[[int], object]],
) -> frozenset[str] | None:
    """Return the processor's features as a baseline's check reads them, or ``None``.

    ``None`` means the processor cannot be asked here.
    """
    x86 = baseline.startswith("x86-64-")
    if x86 and platform == "win32":
        return windows_features(feature_query())
    if platform != "linux":
        return None
    try:
        text = cpuinfo.read_text(encoding="utf-8")
    except OSError:
        return None
    return cpuinfo_flags(text, "flags" if x86 else "Features")


def baseline_shortfall(
    package_dir: Path,
    *,
    platform: str | None = None,
    cpuinfo: Path = CPUINFO,
    feature_query: Callable[[], Callable[[int], object]] = kernel32_feature_query,
) -> CpuBaselineError | None:
    """Return the refusal a processor below this build's baseline earns, or ``None``.

    Parameters
    ----------
    package_dir
        The installed package directory, holding the baseline record.
    platform
        ``sys.platform`` (the default), or a stand-in for tests.
    cpuinfo
        The processor description read on Linux.
    feature_query
        Returns Windows's ``IsProcessorFeaturePresent``; a stand-in for tests.

    Returns
    -------
    CpuBaselineError | None
        The error naming the missing features, or ``None`` when the processor meets
        the baseline or cannot be checked.
    """
    try:
        baseline = (package_dir / BASELINE_RECORD).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if baseline not in REQUIRED_FEATURES:
        return None
    flags = _processor_features(
        baseline, sys.platform if platform is None else platform, cpuinfo, feature_query
    )
    if flags is None:
        return None
    missing = missing_features(baseline, flags)
    return CpuBaselineError(baseline, missing) if missing else None


def check_cpu_baseline(package_dir: Path, cpuinfo: Path = CPUINFO) -> None:
    """Refuse to continue on a processor below this build's CPU baseline.

    Parameters
    ----------
    package_dir
        The installed package directory, holding the baseline record.
    cpuinfo
        The processor description read on Linux, ``/proc/cpuinfo``.

    Raises
    ------
    CpuBaselineError
        If the processor lacks a feature of the recorded baseline.
    """
    error = baseline_shortfall(package_dir, cpuinfo=cpuinfo)
    if error is not None:
        raise error
