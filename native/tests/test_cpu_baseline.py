# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The processor check a distribution build makes before loading its extension."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from pathlib import Path

import pytest

import solprop_native as native
from solprop_native import _cpu  # pyright: ignore[reportPrivateUsage]

# Abbreviated /proc/cpuinfo entries: a Haswell-class (x86-64-v3) processor, and a
# Sandy Bridge-class one with AVX but neither AVX2 nor FMA (x86-64-v2 only).
V3_FLAGS = (
    "fpu cx8 cmov sse sse2 ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt xsave avx f16c "
    "lahf_lm abm bmi1 avx2 bmi2"
)
V2_FLAGS = "fpu cx8 cmov sse sse2 ssse3 cx16 sse4_1 sse4_2 popcnt xsave avx lahf_lm"
# Linux aarch64 Features lines: a Neoverse N1 (Graviton2), and a Cortex-A72
# (Armv8.0-A: Graviton1, Raspberry Pi 4).
N1_FEATURES = (
    "fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm "
    "lrcpc dcpop asimddp ssbs"
)
A72_FEATURES = "fp asimd evtstrm aes pmull sha1 sha2 crc32 cpuid"
# Windows PF_* constants (IsProcessorFeaturePresent) of a Sandy Bridge and a Haswell.
PF_SANDY_BRIDGE = frozenset({14, 17, 36, 37, 38, 39})
PF_HASWELL = PF_SANDY_BRIDGE | {40}


def cpuinfo(flags: str, key: str = "flags") -> str:
    return (
        "processor\t: 0\nvendor_id\t: GenuineIntel\n"
        f"{key}\t\t: {flags}\n\nprocessor\t: 1\n"
    )


def test_levels_nest_and_name_the_psabi_features() -> None:
    levels = _cpu.REQUIRED_FEATURES
    assert levels["x86-64-v2"] < levels["x86-64-v3"] < levels["x86-64-v4"]
    assert {"avx2", "fma", "bmi2"} <= levels["x86-64-v3"] - levels["x86-64-v2"]
    assert "avx512f" in levels["x86-64-v4"] - levels["x86-64-v3"]


def test_cpuinfo_flags_reads_the_first_processor() -> None:
    assert _cpu.cpuinfo_flags(cpuinfo("sse2 avx2")) == {"sse2", "avx2"}
    assert _cpu.cpuinfo_flags("processor\t: 0\nFeatures\t: fp asimd\n") is None
    assert _cpu.cpuinfo_flags("processor\t: 0\nFeatures\t: fp asimd\n", "Features") == {
        "fp",
        "asimd",
    }


def test_missing_features_names_what_the_baseline_needs() -> None:
    v3 = _cpu.cpuinfo_flags(cpuinfo(V3_FLAGS))
    v2 = _cpu.cpuinfo_flags(cpuinfo(V2_FLAGS))
    assert v3 is not None
    assert v2 is not None
    assert _cpu.missing_features("x86-64-v3", v3) == []
    assert _cpu.missing_features("x86-64-v3", v2) == [
        "abm",
        "avx2",
        "bmi1",
        "bmi2",
        "f16c",
        "fma",
        "movbe",
    ]
    assert _cpu.missing_features("x86-64-v2", v2) == []
    # apple-m1 (every Apple silicon Mac) and "none" are not checked.
    assert _cpu.missing_features("apple-m1", v2) == []
    assert _cpu.missing_features("none", v2) == []


def write_record(tmp_path: Path, baseline: str) -> Path:
    package = tmp_path / "solprop_native"
    package.mkdir(parents=True)
    (package / _cpu.BASELINE_RECORD).write_text(f"{baseline}\n", encoding="utf-8")
    return package


def write_build(
    tmp_path: Path, baseline: str, flags: str, key: str = "flags"
) -> tuple[Path, Path]:
    package = write_record(tmp_path, baseline)
    info = tmp_path / "cpuinfo"
    info.write_text(cpuinfo(flags, key), encoding="utf-8")
    return package, info


def test_an_older_processor_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_cpu.sys, "platform", "linux")
    package, info = write_build(tmp_path, "x86-64-v3", V2_FLAGS)
    with pytest.raises(
        _cpu.CpuBaselineError, match=r"x86-64-v3 processor.*lacks abm, avx2"
    ) as refused:
        _cpu.check_cpu_baseline(package, info)
    assert isinstance(refused.value, ImportError)
    assert refused.value.baseline == "x86-64-v3"
    assert refused.value.missing_features == (
        "abm",
        "avx2",
        "bmi1",
        "bmi2",
        "f16c",
        "fma",
        "movbe",
    )
    assert _cpu.BUILD_GUIDE in str(refused.value)


def test_a_capable_or_unverifiable_processor_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_cpu.sys, "platform", "linux")
    package, info = write_build(tmp_path, "x86-64-v3", V3_FLAGS)
    _cpu.check_cpu_baseline(package, info)
    # No cpuinfo (macOS), or no baseline record (a local or editable build).
    _cpu.check_cpu_baseline(package, tmp_path / "absent")
    _cpu.check_cpu_baseline(tmp_path, info)


def test_linux_x86_is_checked_only_on_linux(tmp_path: Path) -> None:
    package, info = write_build(tmp_path, "x86-64-v3", V2_FLAGS)
    error = _cpu.baseline_shortfall(package, platform="linux", cpuinfo=info)
    assert error is not None
    assert "avx2" in error.missing_features
    assert _cpu.baseline_shortfall(package, platform="darwin", cpuinfo=info) is None


def test_linux_aarch64_needs_the_neoverse_n1_features(tmp_path: Path) -> None:
    package, info = write_build(tmp_path, "neoverse-n1", A72_FEATURES, "Features")
    error = _cpu.baseline_shortfall(package, platform="linux", cpuinfo=info)
    assert error is not None
    # Armv8.0 lacks the Armv8.1 atomics and RDM and the N1's Armv8.2 extensions.
    assert error.missing_features == (
        "asimddp",
        "asimdhp",
        "asimdrdm",
        "atomics",
        "fphp",
        "lrcpc",
    )
    assert str(error).startswith(
        "this solprop-native build requires an neoverse-n1 processor, and this one "
        "lacks asimddp, asimdhp, asimdrdm, atomics, fphp, lrcpc"
    )
    info.write_text(cpuinfo(N1_FEATURES, "Features"), encoding="utf-8")
    assert _cpu.baseline_shortfall(package, platform="linux", cpuinfo=info) is None
    # The N1's features are exactly what its compiler target enables.
    assert _cpu.REQUIRED_FEATURES["neoverse-n1"] <= set(N1_FEATURES.split())


def test_apple_silicon_and_unknown_baselines_are_not_checked(tmp_path: Path) -> None:
    def ask_nothing() -> Callable[[int], object]:
        raise AssertionError("the processor was asked")

    package = write_record(tmp_path / "m1", "apple-m1")
    assert _cpu.baseline_shortfall(package, platform="darwin") is None
    package = write_record(tmp_path / "haswell", "haswell")
    assert (
        _cpu.baseline_shortfall(package, platform="win32", feature_query=ask_nothing)
        is None
    )


@pytest.mark.parametrize(
    ("baseline", "present", "missing"),
    [
        ("x86-64-v3", PF_HASWELL, ()),
        (
            "x86-64-v3",
            PF_SANDY_BRIDGE,
            ("abm", "avx2", "bmi1", "bmi2", "f16c", "fma", "movbe"),
        ),
        ("x86-64-v2", PF_SANDY_BRIDGE, ()),
        # A Core 2 (Penryn) has SSSE3, SSE4.1 and CMPXCHG16B, not SSE4.2.
        ("x86-64-v2", frozenset({14, 36, 37}), ("popcnt", "sse4_2")),
        (
            "x86-64-v4",
            PF_HASWELL,
            ("avx512bw", "avx512cd", "avx512dq", "avx512f", "avx512vl"),
        ),
        ("x86-64-v4", PF_HASWELL | {41}, ()),
    ],
)
def test_windows_asks_kernel32(
    tmp_path: Path, baseline: str, present: frozenset[int], missing: tuple[str, ...]
) -> None:
    package = write_record(tmp_path, baseline)

    def is_present(constant: int) -> int:
        return int(constant in present)

    def query() -> Callable[[int], object]:
        return is_present

    error = _cpu.baseline_shortfall(package, platform="win32", feature_query=query)
    if not missing:
        assert error is None
        return
    assert error is not None
    assert error.missing_features == missing
    assert f"requires an {baseline} processor" in str(error)
    assert f"lacks {', '.join(missing)}" in str(error)


def test_the_windows_query_is_kernel32s(monkeypatch: pytest.MonkeyPatch) -> None:
    class Function:
        def __call__(self, constant: int) -> int:
            return int(constant == 40)

    class Kernel32:
        IsProcessorFeaturePresent = Function()

    opened: list[str] = []

    def win_dll(name: str) -> Kernel32:
        opened.append(name)
        return Kernel32()

    monkeypatch.setattr(ctypes, "WinDLL", win_dll, raising=False)
    query = _cpu.kernel32_feature_query()
    assert opened == ["kernel32"]
    assert query(40) == 1
    assert query(39) == 0
    assert vars(query)["restype"] is ctypes.c_int
    assert vars(query)["argtypes"] == [ctypes.c_uint32]
    assert _cpu.windows_features(query) >= {"avx2", "fma", "bmi2", "lahf_lm"}


def test_an_installed_record_matches_the_build() -> None:
    # Editable installs keep the record in the build tree; wheels install it.
    record = Path(native.__file__).parent / _cpu.BASELINE_RECORD
    if not record.exists():
        pytest.skip("no installed CPU-baseline record (editable install)")
    assert (
        record.read_text(encoding="utf-8").strip() == native.build_info().cpu_baseline
    )


def variants_package(tmp_path: Path, floor: str = "x86-64-v2") -> Path:
    (tmp_path / _cpu.BASELINE_RECORD).write_text(floor + "\n", encoding="utf-8")
    (tmp_path / _cpu.VARIANTS_RECORD).write_text(
        "x86-64-v3 _core\nx86-64-v2 _core_v2\n", encoding="utf-8"
    )
    return tmp_path


@pytest.mark.parametrize(
    ("flags", "stem"), [(V3_FLAGS, "_core"), (V2_FLAGS, "_core_v2")]
)
def test_a_processor_gets_the_best_build_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flags: str, stem: str
) -> None:
    monkeypatch.delenv(_cpu.VARIANT_SETTING, raising=False)
    info = tmp_path / "cpuinfo"
    info.write_text(cpuinfo(flags), encoding="utf-8")
    package = variants_package(tmp_path)
    assert _cpu.select_variant(package, platform="linux", cpuinfo=info) == stem


def test_windows_chooses_from_kernel32(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_cpu.VARIANT_SETTING, raising=False)
    package = variants_package(tmp_path)
    for present, stem in ((PF_HASWELL, "_core"), (PF_SANDY_BRIDGE, "_core_v2")):

        def query(present: frozenset[int] = present) -> Callable[[int], object]:
            return lambda constant: constant in present

        assert _cpu.select_variant(package, platform="win32", feature_query=query) == stem


def test_the_setting_may_only_ask_for_a_lower_build(tmp_path: Path) -> None:
    info = tmp_path / "cpuinfo"
    package = variants_package(tmp_path)
    info.write_text(cpuinfo(V3_FLAGS), encoding="utf-8")
    assert (
        _cpu.select_variant(
            package, requested="x86-64-v2", platform="linux", cpuinfo=info
        )
        == "_core_v2"
    )
    info.write_text(cpuinfo(V2_FLAGS), encoding="utf-8")
    with pytest.raises(ValueError, match="runs only x86-64-v2"):
        _cpu.select_variant(
            package, requested="x86-64-v3", platform="linux", cpuinfo=info
        )
    with pytest.raises(ValueError, match="holds x86-64-v3, x86-64-v2"):
        _cpu.select_variant(
            package, requested="x86-64-v4", platform="linux", cpuinfo=info
        )


def test_a_package_with_one_extension_has_no_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_cpu.VARIANT_SETTING, raising=False)
    assert _cpu.select_variant(tmp_path) == "_core"
    with pytest.raises(ValueError, match="holds one extension"):
        _cpu.select_variant(tmp_path, requested="x86-64-v2")

