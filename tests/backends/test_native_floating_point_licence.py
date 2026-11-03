# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Native's floating-point licence on its production helpers (SEP 25 section 4.2).

``native/tests/native/test_fp.cpp`` proves that ``SOLPROP_NATIVE_STRICT_BLOCK`` works,
on functions local to the test. Nothing there fails if the block is removed from a
production helper, or if a build's flags change. Three checks close that:

* **The list.** Each helper the licence names carries the strict attribute and starts
  its body with the strict block, read from the source. The list is written out here
  and not derived from the source, so that removing a block fails.
* **The flags.** No build carries a flag SEP 25 refuses, from the build's own record
  and from ``CMakeLists.txt``.
* **The behaviour.** One production helper reachable from Python on chosen operands:
  the phi-functions' series. A strict series rounds every product and sum on its own,
  which Python's floats do too, so its moments are the bits of the same operations
  written out here (an exact structural contract, not a numerical comparison), on any
  host. A build with a fused multiply-add and no block there departs.

The Numba test read each helper's compiled flags and its disassembly; a native wheel is
stripped, and GCC's attribute leaves nothing else to read, so the source text stands in
for the former. Only the phi-functions are held by behaviour: the fraction, the
defects and the phase sums are read by the source check alone, which sees the block
and not its effect.
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

from solprop.grid import EnvelopeGrid

NATIVE = Path(__file__).resolve().parents[2] / "native"

STRICT_ATTRIBUTE = "SOLPROP_NATIVE_STRICT_FUNCTION"
STRICT_BLOCK = "SOLPROP_NATIVE_STRICT_BLOCK"

ENVELOPE_NOT_STRICT = pytest.mark.xfail(
    strict=True,
    reason=(
        "SEP 43 inventory: the shared Numba body was protected from contraction "
        "(bodies.py @protect, test_numba_licence's PROTECTED), and native's is not; "
        "SEP 25 section 4.2 does not name it and fp.hpp's list omits it, so this "
        "wants a decision: protect it, or take it off the licence"
    ),
)

PROTECTED = [
    # The ionised fraction (Numba's fraction_at, envelope_population).
    ("src/general.cpp", "decayed_fraction"),
    ("src/general.cpp", "decayed_fractions"),
    ("src/general.cpp", "envelope_fraction"),
    # The small-argument phi-functions (multistep._phi, _weights).
    ("include/solprop_native/adams_phi.hpp", "adams_phi_fixed"),
    ("include/solprop_native/adams_phi.hpp", "adams_lanes_classify"),
    ("include/solprop_native/adams_phi.hpp", "adams_lanes_phi"),
    ("include/solprop_native/adams_phi.hpp", "adams_lanes_large_exponential"),
    ("include/solprop_native/lagrange_basis.hpp", "basis_rows"),
    ("include/solprop_native/lagrange_basis.hpp", "trial_tables"),
    # The stable phase increments.
    ("src/general.cpp", "phase_sum"),
    ("src/general.cpp", "phase_block"),
    ("src/general.cpp", "adams_phase_sum"),
    ("src/general.cpp", "adams_phase_block"),
    ("src/general.cpp", "adams_lanes_phase"),
    # The multistep defects and compensation (correct, exponential_correct, carry,
    # _coupled_correct, _carry_flows).
    ("src/general.cpp", "compensation_residual"),
    ("src/general.cpp", "adams_predicted_compensation"),
    ("src/general.cpp", "adams_corrected_compensation"),
    ("src/general.cpp", "adams_observed_compensation"),
    ("src/general.cpp", "adams_lanes_predicted_compensation"),
    ("src/general.cpp", "adams_lanes_corrected_compensation"),
    ("src/general.cpp", "adams_lanes_defects"),
    ("src/general.cpp", "adams_lanes_rate"),
    ("src/general.cpp", "adams_defect_block"),
    # The far field's plane radius (mask_plane_radius).
    ("include/solprop_native/far_field.hpp", "far_field_radius"),
    # The ellipse's axes and its cycle moments.
    pytest.param(
        "include/solprop_native/envelope_plasma.hpp",
        "envelope_axes",
        marks=ENVELOPE_NOT_STRICT,
    ),
    pytest.param(
        "include/solprop_native/envelope_plasma.hpp",
        "envelope_moments",
        marks=ENVELOPE_NOT_STRICT,
    ),
]
"""Each helper the licence protects, and the file that defines it."""


def definition(text, name):
    """Return what precedes a function's definition, and its body's start."""
    for match in re.finditer(rf"\b{re.escape(name)}\s*\(", text):
        # The first parenthesis closing at depth zero, then a brace and not a semicolon.
        depth, index = 0, match.end() - 1
        while True:
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    break
            index += 1
        following = re.match(r"(\s*(const|noexcept))*\s*\{", text[index + 1 :])
        if following is None:
            continue
        # A call in a body is preceded by its statement, not by a declaration.
        before = text[: match.start()]
        start = max(before.rfind(";"), before.rfind("}"), before.rfind("{"))
        return before[start + 1 :], text[index + 1 + following.end() :]
    raise AssertionError(f"no definition of {name}")


@pytest.mark.parametrize(("path", "name"), PROTECTED)
def test_every_protected_helper_is_strict(path, name):
    source = NATIVE / path
    if not source.exists():
        pytest.skip("the native sources are not in this tree")
    declaration, body = definition(source.read_text(encoding="utf-8"), name)
    assert STRICT_ATTRIBUTE in declaration
    assert body.lstrip().startswith(STRICT_BLOCK)


def test_the_phase_lambdas_of_the_core_are_strict():
    """The compact state's stage phases are lambdas, each carrying the attribute."""
    source = NATIVE / "src" / "core.cpp"
    if not source.exists():
        pytest.skip("the native sources are not in this tree")
    text = source.read_text(encoding="utf-8")
    for name in ("whole_phase", "stage_phase"):
        pattern = rf"{name} = \[&\]\(\) {STRICT_ATTRIBUTE} \{{\s*{STRICT_BLOCK}"
        assert re.search(pattern, text), name


def test_a_protected_marker_still_does_something():
    """The macro names its two mechanisms: the list cannot see an empty one."""
    source = NATIVE / "include" / "solprop_native" / "fp.hpp"
    if not source.exists():
        pytest.skip("the native sources are not in this tree")
    text = source.read_text(encoding="utf-8")
    assert f'#define {STRICT_BLOCK} _Pragma("clang fp contract(off)")' in text
    assert (
        f'#define {STRICT_ATTRIBUTE} __attribute__((optimize("fp-contract=off")))'
        in text
    )


REFUSED = (
    "-ffast-math",
    "-Ofast",
    "-funsafe-math-optimizations",
    "-fassociative-math",
    "-freciprocal-math",
    "-ffinite-math-only",
    "-fno-signed-zeros",
    "-fno-trapping-math",
    "-fapprox-func",
    "-fp-model=fast",
)
"""Flags SEP 25 section 4.2 item 4 and section 5 refuse."""


def test_the_build_system_asks_for_none_of_the_refused_flags():
    source = NATIVE / "CMakeLists.txt"
    if not source.exists():
        pytest.skip("the native sources are not in this tree")
    # Comments may name a flag to say why it is absent.
    lines = source.read_text(encoding="utf-8").splitlines()
    text = "\n".join(line.split("#", 1)[0] for line in lines)
    for flag in REFUSED:
        assert flag not in text, flag
    assert "-fno-fast-math" in text
    assert "-ffp-contract=fast" in text
    # The Adams march's decisions round as Python's do, on every compiler.
    assert re.search(
        r"set_source_files_properties\(src/adams_march\.cpp\s+"
        r"PROPERTIES COMPILE_OPTIONS"
        r'\s+"-ffp-contract=off"\)',
        text,
    )


def test_the_installed_build_records_none_of_the_refused_flags():
    native = pytest.importorskip("solprop_native")
    info = native.build_info()
    assert info.fast_math_enabled is False
    assert "no fast-math" in info.optimization_policy
    for flag in REFUSED:
        assert flag not in info.compiler_flags, flag


def _strict_mul(a, b):
    return complex(a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real)


def _strict_norm(a):
    return a.real * a.real + a.imag * a.imag


def _series(x, order):
    """Return ``phi_1(x) .. phi_order(x)`` by ``adams_phi``'s operations, small ``x``.

    Spelled as ``fp.hpp``'s strict helpers: a Python float rounds every operation, and
    never fuses one. At ``|x| <= 1/2`` there are no squarings, so nothing here depends
    on the library's ``exp``, which enters at ``phi_0`` alone.
    """
    reciprocal = [1.0 / math.factorial(k) for k in range(order + 1)]
    stop = (0.25 * np.finfo(float).eps) ** 2
    term = complex(reciprocal[order])
    total = term
    for m in range(1, 24):
        product = _strict_mul(term, x)
        divisor = 1.0 / (m + order)
        term = complex(divisor * product.real, divisor * product.imag)
        total = complex(total.real + term.real, total.imag + term.imag)
        if _strict_norm(term) <= stop * _strict_norm(total):
            break
    out = [0j] * (order + 1)
    out[order] = total
    for k in range(order - 1, 0, -1):
        product = _strict_mul(x, out[k + 1])
        out[k] = complex(reciprocal[k] + product.real, product.imag)
    return out[1:]


def test_the_small_argument_series_rounds_every_operation():
    """Native's moments are the bits of a series with no fused operation in it."""
    native = pytest.importorskip("solprop_native")
    from backends.test_native_phi_moments import read_moments  # noqa: PLC0415

    grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    rng = np.random.default_rng(2026)
    size = 512
    radius = 0.5 * np.sqrt(rng.random(size))
    values = radius * np.exp(2j * np.pi * rng.random(size))
    moments = read_moments(native, grid, values)
    order = moments.shape[1]
    # Random operands: a contraction that changed nothing on all of them is no fusion.
    for index, x in enumerate(values):
        expected = [
            math.factorial(k) * one for k, one in enumerate(_series(complex(x), order))
        ]
        assert moments[index].tolist() == expected, x
