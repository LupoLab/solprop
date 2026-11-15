# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CUDA build check that fails on a device lambda two files instantiate (#815)."""

import importlib.util
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "native" / "ci" / "check_lambda_tags.py"
spec = importlib.util.spec_from_file_location("check_lambda_tags", SCRIPT)
assert spec is not None and spec.loader is not None
check_lambda_tags = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_lambda_tags)


def _helper(function: str, slot: str = "fp_copier") -> str:
    """Return `nm --demangle --format=posix`'s line for a lambda's helper static."""
    return (
        "(anonymous namespace)::__nv_hdl_helper<__nv_dl_tag<void (*)(Kokkos::Cuda "
        "const&, Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::CudaSpace, "
        f"Kokkos::MemoryTraits<1u> > const&, double), &(void {function}<Kokkos::Cuda, "
        "Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::CudaSpace, "
        "Kokkos::MemoryTraits<1u> > >(Kokkos::Cuda const&, Kokkos::View<double**, "
        "Kokkos::LayoutRight, Kokkos::CudaSpace, Kokkos::MemoryTraits<1u> > const&, "
        f"double)), 1u>, void, unsigned long>::{slot} b 0000000000000018 "
        "0000000000000008\n"
    )


GEMM = _helper("solprop_native::device_dense_gemm")
GATHER = _helper("solprop_native::GeneralWorkspace::synthesise_retained")
NORMALIZE = _helper("KokkosFFT::Impl::normalize_impl")
OTHER = (
    "solprop_native::Problem::validate() const T 0000000000001000 0000000000000040\n"
)


def _objects(tmp_path, monkeypatch, listings):
    """Return object files whose `nm` listings are `listings`, in order."""
    by_path = {}
    for index, listing in enumerate(listings):
        path = tmp_path / f"unit{index}.cpp.o"
        path.write_bytes(b"")
        by_path[str(path)] = listing

    def nm(command, **_):
        return subprocess.CompletedProcess(command, 0, stdout=by_path[command[-1]])

    monkeypatch.setattr(check_lambda_tags.subprocess, "run", nm)
    return list(by_path)


def test_each_file_s_own_lambdas_pass(tmp_path, monkeypatch, capsys):
    objects = _objects(tmp_path, monkeypatch, [GEMM + OTHER, GATHER])
    assert check_lambda_tags.main(objects) == 0
    assert (
        "2 extended lambda helpers in 2 files, none shared" in capsys.readouterr().out
    )


def test_a_lambda_two_files_hold_fails_and_is_named(tmp_path, monkeypatch, capsys):
    # The radial route's device GEMM, instantiated alike in two files, called a null
    # copy function from the file whose dispatch the link kept (#815).
    objects = _objects(tmp_path, monkeypatch, [GEMM, GATHER + GEMM])
    assert check_lambda_tags.main(objects) == 1
    error = capsys.readouterr().err
    assert "lambda in several files (unit0.cpp.o, unit1.cpp.o)" in error
    assert "solprop_native::device_dense_gemm" in error


def test_an_allowed_library_lambda_is_reported_not_failed(
    tmp_path, monkeypatch, capsys
):
    objects = _objects(tmp_path, monkeypatch, [NORMALIZE + GEMM, NORMALIZE])
    assert check_lambda_tags.main(objects) == 0
    assert "allowed in several files" in capsys.readouterr().out


def test_the_other_slots_are_not_counted_twice(tmp_path, monkeypatch):
    # A lambda's call and delete static members name the same tag as its copy member.
    listing = GEMM + _helper("solprop_native::device_dense_gemm", "fp_caller")
    objects = _objects(tmp_path, monkeypatch, [listing])
    assert check_lambda_tags.main(objects) == 0


def test_finding_no_helper_fails(tmp_path, monkeypatch, capsys):
    # A change in nvcc's naming must not make the check pass on nothing.
    objects = _objects(tmp_path, monkeypatch, [OTHER])
    assert check_lambda_tags.main(objects) == 1
    assert "no extended lambda helper found" in capsys.readouterr().err


def test_naming_no_file_is_a_usage_error():
    assert check_lambda_tags.main([]) == 2
