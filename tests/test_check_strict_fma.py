# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The wheel check that fails on a fused multiply-add in a strict file (#135)."""

import importlib.util
import subprocess
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "native" / "ci" / "check_strict_fma.py"
spec = importlib.util.spec_from_file_location("check_strict_fma", SCRIPT)
assert spec is not None and spec.loader is not None
check_strict_fma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_strict_fma)

STRICT = """\
0000000000001100 <solprop_native::march::factor(double)>:
    1100:\tvmulsd %xmm1,%xmm0,%xmm0
    1104:\tvaddsd %xmm2,%xmm0,%xmm0
0000000000001200 <solprop_native::rk::score(double const*, unsigned long)>:
    1200:\tfmul\td0, d1, d2
"""
CONTRACTING = """\
0000000000001300 <solprop_native::general::transform(double)>:
    1300:\tvfmadd231sd %xmm1,%xmm2,%xmm0
"""
FUSED = """\
0000000000001400 <solprop_native::watch::check(double)>:
    1400:\tvfmadd213sd %xmm1,%xmm2,%xmm0
    1404:\tfmadd\td0, d1, d2, d3
"""


@pytest.mark.parametrize(
    "line",
    [
        "\tvfmadd231sd %xmm1,%xmm2,%xmm0",
        "\tvfnmsub132pd %ymm1,%ymm2,%ymm0",
        "\tvfmaddsub213pd %zmm1,%zmm2,%zmm0",
        "\tfmadd\td0, d1, d2, d3",
        "\tfnmsub\td0, d1, d2, d3",
        "\tfmla\tv0.2d, v1.2d, v2.2d",
    ],
)
def test_each_fused_form_is_recognised(line):
    assert check_strict_fma.FUSED.search(line)


@pytest.mark.parametrize(
    "line",
    ["\tvmulsd %xmm1,%xmm0,%xmm0", "\tvaddpd %ymm1,%ymm2,%ymm0", "\tfmul\td0, d1, d2"],
)
def test_a_separate_product_or_sum_is_not(line):
    assert not check_strict_fma.FUSED.search(line)


def _wheel(tmp_path, monkeypatch, listing):
    wheel = tmp_path / "solprop_native-0-cp314-abi3-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("solprop_native/_core.abi3.so", b"")

    def objdump(command, **_):
        return subprocess.CompletedProcess(command, 0, stdout=listing, stderr="")

    monkeypatch.setattr(check_strict_fma.subprocess, "run", objdump)
    return str(wheel)


def test_strict_functions_without_a_fused_instruction_pass(
    tmp_path, monkeypatch, capsys
):
    # A contracting file's fused multiply-add is allowed; only the strict files count.
    wheel = _wheel(tmp_path, monkeypatch, STRICT + CONTRACTING)
    assert check_strict_fma.main([wheel]) == 0
    assert "2 strict functions, none fused" in capsys.readouterr().out


def test_a_fused_strict_function_fails_and_is_named(tmp_path, monkeypatch, capsys):
    wheel = _wheel(tmp_path, monkeypatch, STRICT + FUSED)
    assert check_strict_fma.main([wheel]) == 1
    assert "2 fused multiply-add(s) in" in capsys.readouterr().err


def test_finding_no_strict_function_fails(tmp_path, monkeypatch, capsys):
    # A change in symbol names or visibility must not make the check pass on nothing.
    wheel = _wheel(tmp_path, monkeypatch, CONTRACTING)
    assert check_strict_fma.main([wheel]) == 1
    assert "no strict function found" in capsys.readouterr().err


def test_the_native_sessions_unit_is_strict(tmp_path, monkeypatch, capsys):
    # SEP 47: the session's scheduling and export are compiled strict as well.
    session = (
        "0000000000001500 <solprop_native::session::apply_recipe(double const*)>:\n"
        "    1500:\tvfmadd231pd %ymm1,%ymm2,%ymm0\n"
    )
    wheel = _wheel(tmp_path, monkeypatch, STRICT + session)
    assert check_strict_fma.main([wheel]) == 1
    assert "solprop_native::session::apply_recipe" in capsys.readouterr().err
