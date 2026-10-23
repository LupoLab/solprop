# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The floating-point licence of ``solprop._numba_cache.njit`` (SEP 25 section 5).

Contraction is on for every kernel; reassociation only where a kernel asks for it; a
protected helper compiles strict and is called, never inlined; anything else is
refused. The licence is part of the disk-cache key, and once, on the machine's own
instruction set, the disassembly shows a fused multiply-add in an ordinary kernel and
none in a strict helper.
"""

import importlib
import platform
import re
import sys
import textwrap

import pytest
from numba.core.dispatcher import Dispatcher

from solprop._numba_cache import CONTRACT, REASSOCIATE, licence, njit
from solprop.backends.numba_cpu import multistep
from solprop.backends.numba_cpu.kernels import HOST

FUSED = re.compile(r"^\s+v?f(n)?m(add|sub)\w*\s", re.MULTILINE)
"""A fused multiply-add on aarch64 (``fmadd``, ``fnmsub``, ...) or x86 (``vfmadd231sd``,
...)."""

PROTECTED = (
    HOST.fraction_at,
    HOST.envelope_population,
    HOST.envelope_axes,
    HOST.envelope_moments,
    HOST.mask_plane_radius,
    multistep._weights,
    multistep._phi,
    *(
        getattr(multistep.make_passes(parallel=parallel), name)
        for parallel in (False, True)
        for name in ("correct", "exponential_correct", "carry")
    ),
    multistep._coupled_correct,
    multistep._carry_flows,
)
"""The protected helpers of the ``numba_cpu`` backend (SEP 25 section 4.2)."""


def test_the_licence_accepts_the_two_flag_sets_and_strict():
    assert licence() == CONTRACT == frozenset({"contract"})
    assert licence({"reassoc", "contract"}) == REASSOCIATE
    assert licence(frozenset(REASSOCIATE)) == REASSOCIATE
    assert licence(strict=True) == frozenset()


@pytest.mark.parametrize(
    "flags",
    [
        True,
        False,
        "contract",
        {"fast"},
        {"reassoc"},
        {"contract", "nnan"},
        {"contract", "reassoc", "ninf"},
        {"contract", "afn"},
        {"contract", "arcp"},
        {"contract", "nsz"},
        ["contract"],
    ],
)
def test_the_licence_refuses_every_other_flag(flags):
    with pytest.raises(ValueError, match="outside the floating-point licence"):
        njit(fastmath=flags)


def test_a_strict_helper_takes_no_flags_and_is_not_inlined():
    with pytest.raises(ValueError, match="one or the other"):
        njit(strict=True, fastmath={"contract"})
    with pytest.raises(ValueError, match="compiled separately and called"):
        njit(strict=True, inline="always")


def test_each_compilation_carries_its_licence():
    def mad(a, b, c):
        return a * b + c

    default = njit()(mad)
    reassociated = njit(fastmath={"contract", "reassoc"})(mad)
    strict = njit(strict=True)(mad)
    assert default.targetoptions["fastmath"] == {"contract"}
    assert reassociated.targetoptions["fastmath"] == {"contract", "reassoc"}
    assert strict.targetoptions["fastmath"] is False


@pytest.mark.parametrize(
    "helper",
    PROTECTED,
    ids=lambda one: (
        f"{one.py_func.__name__}-{one.targetoptions.get('parallel', False)}"
    ),
)
def test_every_protected_helper_is_strict_and_called(helper):
    # Structural: inlined, a helper would compile under its caller's flags.
    assert isinstance(helper, Dispatcher)
    assert helper.targetoptions["fastmath"] is False
    assert helper.targetoptions.get("inline", "never") == "never"


def test_an_inlined_body_takes_contraction():
    assert HOST.accumulate.targetoptions["fastmath"] == {"contract"}
    assert HOST.accumulate.targetoptions["inline"] == "always"


def test_the_licence_is_part_of_the_cache_key(tmp_path, monkeypatch):
    """Two compilations under different flags keep two entries, each read back."""
    source = tmp_path / "licensed_probe.py"
    source.write_text(
        textwrap.dedent(
            """
            def mad(a, b, c):
                return a * b + c
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("licensed_probe")
    try:
        args = (1.5, 2.5, 3.5)
        contracted = njit(cache=True)(module.mad)
        reassociated = njit(cache=True, fastmath={"contract", "reassoc"})(module.mad)
        assert contracted(*args) == reassociated(*args) == 7.25
        files = sorted(tmp_path.rglob("licensed_probe.mad-*.nbc"))
        assert len(files) == 2
        # A fresh dispatcher under each licence loads its own entry, not the other's.
        for flags in (None, {"contract", "reassoc"}):
            again = njit(cache=True, fastmath=flags)(module.mad)
            assert again(*args) == 7.25
            assert sum(again.stats.cache_hits.values()) == 1
            assert again.targetoptions["fastmath"] == licence(flags)
        strict = njit(cache=True, strict=True)(module.mad)
        assert strict(*args) == 7.25
        assert sum(strict.stats.cache_hits.values()) == 0
        assert len(sorted(tmp_path.rglob("licensed_probe.mad-*.nbc"))) == 3
    finally:
        sys.modules.pop("licensed_probe", None)


def _assembly(dispatcher, *args) -> str:
    """Compile `dispatcher` for `args` and return its machine code."""
    dispatcher(*args)
    return dispatcher.inspect_asm(dispatcher.signatures[-1])


@pytest.mark.skipif(
    platform.machine().lower() not in {"arm64", "aarch64", "x86_64", "amd64"},
    reason="the fused-instruction pattern is written for aarch64 and x86",
)
def test_contraction_shows_in_kernels_and_not_in_strict_helpers():
    """Once, in the disassembly: SEP 25 checks the flags take effect."""

    def mad(a, b, c):
        return a * b + c

    probe = _assembly(njit()(mad), 1.0, 2.0, 3.0)
    if not FUSED.search(probe):
        pytest.skip("this CPU has no fused multiply-add instruction")
    assert not FUSED.search(_assembly(njit(strict=True)(mad), 1.0, 2.0, 3.0))
    # The production bodies: an ordinary one contracts, the protected ones do not.
    assert FUSED.search(_assembly(HOST.accumulate, 1.0, 2.0, 3.0))
    assert not FUSED.search(_assembly(HOST.fraction_at, 0.25, 0.1))
    assert not FUSED.search(_assembly(HOST.envelope_axes, 1.0 + 2.0j, 0.5 - 1.0j))
    # Not the phi-functions here: their machine code links numba's own complex
    # ``exp``, whose compiled helper need not carry the caller's flags, so the
    # disassembly does not show the phi-function's own arithmetic alone; the library
    # function keeps its own error bound and is outside the licence.
