# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``check_build``'s refusals of an inconsistent build record.

Each case is a consistent record with one rule broken, read through the seam
``check_build`` reads its build from (``_build.build_info``), so that a refusal left
out lets its case through. The checkout's digest is a separate refusal, tested in
``test_api_contract.py``; here it reads as an installed wheel's, which has none.
"""

from __future__ import annotations

import dataclasses

import pytest
from solprop.exceptions import BackendError

import solprop_native as native
from solprop_native import _build  # pyright: ignore[reportPrivateUsage]

# Consistent records of each kind: a CPU build with a vendor BLAS, the reference and
# distribution profiles, and a CUDA and a HIP build with their device BLAS.
CPU: dict[str, object] = {
    "fast_math_enabled": False,
    "openmp_enabled": True,
    "execution_space": "OpenMP",
    "device_backend": "none",
    "device_count": 0,
    "gemm_provider": "HYBRID",
    "tpl_blas_enabled": True,
    "blas_backend": "OpenBLAS",
    "build_profile": "local",
    "native_arch_enabled": False,
    "cpu_baseline": "none",
}
REFERENCE = CPU | {
    "build_profile": "reference",
    "tpl_blas_enabled": False,
    "blas_backend": "none",
}
DISTRIBUTION = CPU | {"build_profile": "distribution", "cpu_baseline": "x86-64-v3"}
CUDA = CPU | {
    "openmp_enabled": False,
    "execution_space": "Cuda",
    "device_backend": "cuda",
    "device_count": 1,
    "blas_backend": "cuBLAS",
}
HIP = CUDA | {
    "execution_space": "HIP",
    "device_backend": "hip",
    "blas_backend": "rocBLAS",
}

ACCEPTED = [
    pytest.param(CPU, id="local CPU"),
    pytest.param(CPU | {"native_arch_enabled": True}, id="local native CPU"),
    pytest.param(
        CPU | {"tpl_blas_enabled": False, "blas_backend": "none"}, id="no BLAS"
    ),
    pytest.param(REFERENCE, id="reference"),
    pytest.param(DISTRIBUTION, id="distribution"),
    pytest.param(CUDA, id="CUDA"),
    pytest.param(HIP, id="HIP"),
    pytest.param(
        CUDA | {"tpl_blas_enabled": False, "blas_backend": "none"},
        id="CUDA without a BLAS",
    ),
]

REFUSED = [
    pytest.param(
        CPU | {"fast_math_enabled": True},
        "solprop-native was built with forbidden fast-math",
        id="fast-math",
    ),
    pytest.param(
        CUDA | {"openmp_enabled": True},
        "the build enables OpenMP but its Kokkos execution space is 'Cuda'",
        id="OpenMP off its space",
    ),
    pytest.param(
        CPU | {"openmp_enabled": False, "execution_space": "Serial"},
        "the 'none' device backend is inconsistent with execution space 'Serial'",
        id="serial CPU build",
    ),
    pytest.param(
        CUDA | {"execution_space": "HIP"},
        "the 'cuda' device backend is inconsistent with execution space 'HIP'",
        id="CUDA on the HIP space",
    ),
    pytest.param(
        CPU | {"device_backend": "sycl"},
        "the 'sycl' device backend is inconsistent with execution space 'OpenMP'",
        id="unknown device backend",
    ),
    pytest.param(
        CUDA | {"device_count": 0},
        "the cuda build has no visible accelerator device",
        id="no visible device",
    ),
    pytest.param(
        CPU | {"gemm_provider": "CUSTOM"},
        "unknown compiled GEMM provider 'CUSTOM'",
        id="unknown GEMM provider",
    ),
    pytest.param(
        CPU | {"blas_backend": "none"},
        "the build enables a BLAS TPL but identifies no provider",
        id="BLAS TPL with no provider",
    ),
    pytest.param(
        CPU | {"build_profile": "native"},
        "unknown build profile 'native'",
        id="unknown profile",
    ),
    pytest.param(
        REFERENCE | {"tpl_blas_enabled": True, "blas_backend": "OpenBLAS"},
        "the reference build links a vendor BLAS or machine-specific code",
        id="reference with a vendor BLAS",
    ),
    pytest.param(
        REFERENCE | {"native_arch_enabled": True},
        "the reference build links a vendor BLAS or machine-specific code",
        id="reference with native code",
    ),
    pytest.param(
        REFERENCE | {"cpu_baseline": "x86-64-v3"},
        "the reference build links a vendor BLAS or machine-specific code",
        id="reference with a CPU baseline",
    ),
    pytest.param(
        DISTRIBUTION | {"native_arch_enabled": True},
        "the distribution build carries machine-native code or no BLAS",
        id="distribution with native code",
    ),
    pytest.param(
        DISTRIBUTION | {"tpl_blas_enabled": False, "blas_backend": "none"},
        "the distribution build carries machine-native code or no BLAS",
        id="distribution without a BLAS",
    ),
    pytest.param(
        CUDA | {"blas_backend": "OpenBLAS"},
        "the selected BLAS provider cannot access this accelerator's memory",
        id="CUDA with a host BLAS",
    ),
    pytest.param(
        HIP | {"blas_backend": "cuBLAS"},
        "the selected BLAS provider cannot access this accelerator's memory",
        id="HIP with another device's BLAS",
    ),
]


@pytest.fixture(autouse=True)
def no_checkout_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare no checkout's sources, as in an installed wheel."""
    monkeypatch.setattr(_build, "_checkout_digest", lambda: None)


def built(monkeypatch: pytest.MonkeyPatch, fields: dict[str, object]) -> None:
    """Make ``check_build`` read this build's record with `fields` in place."""
    info = dataclasses.replace(native.build_info(), **fields)
    monkeypatch.setattr(_build, "build_info", lambda: info)


@pytest.mark.parametrize("fields", ACCEPTED)
def test_a_consistent_build_is_accepted(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, object]
) -> None:
    built(monkeypatch, fields)
    assert native.check_build() is None


@pytest.mark.parametrize(("fields", "message"), REFUSED)
def test_an_inconsistent_build_is_refused_naming_what_is_wrong(
    monkeypatch: pytest.MonkeyPatch, fields: dict[str, object], message: str
) -> None:
    built(monkeypatch, fields)
    with pytest.raises(BackendError) as refusal:
        native.check_build()
    assert str(refusal.value) == message
