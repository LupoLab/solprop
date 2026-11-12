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
- **Windows on Arm**: ``IsProcessorFeaturePresent`` again, for the wheel's ``armv8.2-a``
  floor. The core keeps the compiler's generic Armv8-A baseline, but it statically links
  OpenBLAS built with ``TARGET=NEOVERSEN1``, which compiles with ``-march=armv8.2-a``.
  Windows reports CRC32 and the LSE atomics that Armv8.1 and Armv8.2 make mandatory, and
  has no query for the rest of Armv8.2, so those two stand for it.
- **macOS arm64**: nothing is checked; every Apple silicon Mac is at least an M1.

Where ``/proc/cpuinfo`` cannot be read or has no features line, an x86-64 build asks
NumPy's own ``cpuid`` check, ``__cpu_features__``, a private name that a test holds in
place. If neither answers, the build refuses to load with
:class:`CpuFeaturesUnknownError` rather than risk an illegal instruction with no Python
error, and ``SOLPROP_NATIVE_CPU_VARIANT`` is the deliberate way past that, on the user's
word about their processor. NumPy's names cover only part of Neoverse N1's list, so
Linux aarch64 can't ask it. A record naming a baseline this module does not know
(a source build's own ``-march``) is not checked, and a missing record is a local or
editable build, compiled for the machine it runs on.

A wheel may carry its extension compiled for more than one baseline: the x86-64 wheels
hold an x86-64-v3 and an x86-64-v2 build. ``_cpu_baseline.txt`` then names the lowest,
the wheel's floor, and ``_cpu_variants.txt`` lists each build, best first, as a baseline
and a module stem. :func:`select_variant` takes the first the processor meets and logs,
at INFO, which features put a better build out of reach;
``SOLPROP_NATIVE_CPU_VARIANT`` names a lower one, for tests and comparisons.

The module is pure Python and imports nothing from the package, so solprop's
automatic backend selection loads it by path and runs the same check without
importing the package or starting its runtime (``solprop.backends.registry``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

__all__ = [
    "BASELINE_RECORD",
    "BUILD_GUIDE",
    "CPUINFO",
    "ISSUES",
    "REQUIRED_FEATURES",
    "VARIANTS_RECORD",
    "VARIANT_SETTING",
    "CpuBaselineError",
    "CpuFeaturesUnknownError",
    "baseline_shortfall",
    "check_cpu_baseline",
    "cpuinfo_flags",
    "kernel32_feature_query",
    "missing_features",
    "select_variant",
    "windows_arm_features",
    "windows_features",
]

_LOGGER = logging.getLogger(__name__)

BASELINE_RECORD = "_cpu_baseline.txt"
VARIANTS_RECORD = "_cpu_variants.txt"
VARIANT_SETTING = "SOLPROP_NATIVE_CPU_VARIANT"
CPUINFO = Path("/proc/cpuinfo")
BUILD_GUIDE = (
    "https://solprop.readthedocs.io/en/stable/guide/performance/building-native.html"
)
ISSUES = "https://github.com/LupoLab/solprop/issues"

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
# What Windows can report of the Armv8.2-A that the linked OpenBLAS is compiled for.
_ARMV8_2_A = frozenset({"atomics", "crc32"})
REQUIRED_FEATURES: dict[str, frozenset[str]] = {
    "x86-64-v2": _X86_64_V2,
    "x86-64-v3": _X86_64_V3,
    "x86-64-v4": _X86_64_V4,
    "neoverse-n1": _NEOVERSE_N1,
    "armv8.2-a": _ARMV8_2_A,
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
# The same, for Windows on Arm.
_PF_ARM_FEATURES: dict[str, int] = {
    "crc32": 31,  # PF_ARM_V8_CRC32_INSTRUCTIONS_AVAILABLE
    "atomics": 34,  # PF_ARM_V81_ATOMIC_INSTRUCTIONS_AVAILABLE
}
# NumPy's names for the x86-64 features, as ``__cpu_features__`` reports them. NumPy
# sets AVX only when the operating system saves its state through XSAVE, so AVX stands
# for it.
_NUMPY_FEATURES: dict[str, str] = {
    "abm": "LZCNT",
    "avx": "AVX",
    "avx2": "AVX2",
    "avx512bw": "AVX512BW",
    "avx512cd": "AVX512CD",
    "avx512dq": "AVX512DQ",
    "avx512f": "AVX512F",
    "avx512vl": "AVX512VL",
    "bmi1": "BMI",
    "bmi2": "BMI2",
    "cx16": "CX16",
    "f16c": "F16C",
    "fma": "FMA3",
    "lahf_lm": "LAHF",
    "movbe": "MOVBE",
    "popcnt": "POPCNT",
    "sse4_1": "SSE41",
    "sse4_2": "SSE42",
    "ssse3": "SSSE3",
    "xsave": "AVX",
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
            f"source on this machine instead ({BUILD_GUIDE}), or report this processor "
            f"({ISSUES})"
        )


class CpuFeaturesUnknownError(ImportError):
    """This processor's features could not be established, so the build isn't loaded.

    - ``variants``: the builds that :data:`VARIANT_SETTING` may name; empty for a single
      build.

    Attributes
    ----------
    baseline : str
        The recorded CPU baseline.
    """

    def __init__(self, baseline: str, variants: Iterable[str] = ()) -> None:
        self.baseline = baseline
        names = ", ".join(variants)
        named = (
            f"name a build this processor runs with {VARIANT_SETTING} ({names}), "
            if names
            else ""
        )
        super().__init__(
            f"cannot establish this processor's features, so the solprop-native build "
            f"with CPU baseline {baseline} is not loaded; {named}build solprop-native "
            f"from source on this machine ({BUILD_GUIDE}), or report this system "
            f"({ISSUES})"
        )


def missing_features(baseline: str, flags: Iterable[str]) -> list[str]:
    """Return the features ``baseline`` requires that ``flags`` lacks, sorted.

    - Returns: missing features; empty when all are present or the baseline is not one
      this module checks.
    """
    required = REQUIRED_FEATURES.get(baseline)
    if required is None:
        return []
    return sorted(required - set(flags))


def cpuinfo_flags(cpuinfo: str, key: str = "flags") -> frozenset[str] | None:
    """Return the first processor's feature names from ``/proc/cpuinfo`` text.

    - ``key``: the line naming them: ``"flags"`` on x86, ``"Features"`` on Arm.
    - Returns: the names, or ``None`` when the text has no such line.
    """
    for line in cpuinfo.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip() == key:
            return frozenset(value.split())
    return None


def windows_features(is_present: Callable[[int], object]) -> frozenset[str]:
    """Return the x86-64 features Windows reports, with those they stand for.

    - ``is_present``: ``kernel32.IsProcessorFeaturePresent``, or a stand-in: takes a
      ``PF_*`` constant and returns a true value when the feature is present.
    """
    present = {name for name, constant in _PF_FEATURES.items() if is_present(constant)}
    for marker, implied in _WINDOWS_IMPLIED.items():
        if marker in present:
            present |= implied
    return frozenset(present | _WINDOWS_ALWAYS)


def windows_arm_features(is_present: Callable[[int], object]) -> frozenset[str]:
    """Return what Windows on Arm reports of the features this module knows."""
    return frozenset(
        name for name, constant in _PF_ARM_FEATURES.items() if is_present(constant)
    )


def numpy_features() -> frozenset[str] | None:
    """Return NumPy's view of the x86-64 features, or ``None`` if it gives none.

    - Returns: the features in Linux's spelling; ``None`` when NumPy no longer has the
      private name, or lacks one of the features asked for.
    """
    try:
        from numpy._core._multiarray_umath import (
            __cpu_features__ as present,  # pyright: ignore[reportUnknownVariableType]
        )
    except ImportError:
        return None
    if not all(key in present for key in _NUMPY_FEATURES.values()):
        return None
    return frozenset(name for name, key in _NUMPY_FEATURES.items() if present[key])


def kernel32_feature_query() -> Callable[[int], object]:
    """Return ``kernel32.IsProcessorFeaturePresent``; Windows only."""
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
    numpy_query: Callable[[], frozenset[str] | None],
) -> frozenset[str] | None:
    """Return the processor's features as a baseline's check reads them, or ``None``.

    ``None`` means nothing could establish them (module docstring).
    """
    x86 = baseline.startswith("x86-64-")
    if platform == "win32":
        query = feature_query()
        return windows_features(query) if x86 else windows_arm_features(query)
    flags = None
    if platform == "linux":
        try:
            text = cpuinfo.read_text(encoding="utf-8")
        except OSError:
            text = ""
        flags = cpuinfo_flags(text, "flags" if x86 else "Features")
    if flags:
        return flags
    return numpy_query() if x86 else None


def baseline_shortfall(
    package_dir: Path,
    *,
    platform: str | None = None,
    cpuinfo: Path = CPUINFO,
    feature_query: Callable[[], Callable[[int], object]] = kernel32_feature_query,
    numpy_query: Callable[[], frozenset[str] | None] = numpy_features,
) -> CpuBaselineError | CpuFeaturesUnknownError | None:
    """Return the refusal a processor below this build's baseline earns, or ``None``.

    - Returns: the error naming the missing features, or the error saying they couldn't
      be established; ``None`` when the processor meets the baseline, this module
      doesn't check the baseline, or :data:`VARIANT_SETTING` has named a build of an
      unknown processor.
    """
    try:
        baseline = (package_dir / BASELINE_RECORD).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if baseline not in REQUIRED_FEATURES:
        return None
    flags = _processor_features(
        baseline,
        sys.platform if platform is None else platform,
        cpuinfo,
        feature_query,
        numpy_query,
    )
    if flags is None:
        names = [name for name, _ in read_variants(package_dir)]
        if os.environ.get(VARIANT_SETTING, "").strip() in names:
            return None
        return CpuFeaturesUnknownError(baseline, names)
    missing = missing_features(baseline, flags)
    return CpuBaselineError(baseline, missing) if missing else None


def check_cpu_baseline(package_dir: Path, cpuinfo: Path = CPUINFO) -> None:
    """Refuse to continue on a processor below this build's CPU baseline."""
    error = baseline_shortfall(package_dir, cpuinfo=cpuinfo)
    if error is not None:
        raise error


def read_variants(package_dir: Path) -> tuple[tuple[str, str], ...]:
    """Return a wheel's builds, best first, as ``(baseline, module stem)`` pairs.

    - Returns: the builds; empty when the package holds only its one extension.
    """
    try:
        text = (package_dir / VARIANTS_RECORD).read_text(encoding="utf-8")
    except OSError:
        return ()
    pairs = (line.split() for line in text.splitlines() if line.strip())
    return tuple((baseline, stem) for baseline, stem in pairs)


def select_variant(
    package_dir: Path,
    *,
    requested: str | None = None,
    platform: str | None = None,
    cpuinfo: Path = CPUINFO,
    feature_query: Callable[[], Callable[[int], object]] = kernel32_feature_query,
    numpy_query: Callable[[], frozenset[str] | None] = numpy_features,
) -> str:
    """Return the module stem of the build to load on this processor.

    - ``requested``: a baseline to run instead, which must be listed and no better than
      the one chosen; the default reads ``SOLPROP_NATIVE_CPU_VARIANT``.
    - Returns: the stem; ``"_core"`` for a package with one extension.
    """
    variants = read_variants(package_dir)
    request = os.environ.get(VARIANT_SETTING, "") if requested is None else requested
    request = request.strip()
    if not variants:
        if request:
            raise ValueError(
                f"{VARIANT_SETTING}={request!r}: this build holds one extension"
            )
        return "_core"
    names = [baseline for baseline, _ in variants]
    if request and request not in names:
        raise ValueError(
            f"{VARIANT_SETTING}={request!r}: this wheel holds {', '.join(names)}"
        )
    system = sys.platform if platform is None else platform
    best = len(variants) - 1
    out_of_reach: list[tuple[str, list[str]]] = []
    for index, (baseline, _) in enumerate(variants):
        flags = _processor_features(
            baseline, system, cpuinfo, feature_query, numpy_query
        )
        if flags is None:
            if request:
                return variants[names.index(request)][1]
            raise CpuFeaturesUnknownError(baseline, names)
        missing = missing_features(baseline, flags)
        if not missing:
            best = index
            break
        out_of_reach.append((baseline, missing))
    if not request:
        for baseline, missing in out_of_reach[:best]:
            _LOGGER.info(
                "loading the %s build, which runs longer: this processor reports "
                "no %s, which the %s build needs",
                names[best],
                ", ".join(missing),
                baseline,
            )
        return variants[best][1]
    chosen = names.index(request)
    if chosen < best:
        raise ValueError(
            f"{VARIANT_SETTING}={request!r}: this processor runs only "
            f"{', '.join(names[best:])}"
        )
    return variants[chosen][1]
