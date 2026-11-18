# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""No comparison in the test trees takes a default absolute tolerance on SI values.

``pytest.approx`` and NumPy's ``allclose``/``isclose`` keep an absolute tolerance beside
the relative one (1e-12 and 1e-8), and many of the suite's quantities are far smaller:
times of 1e-15 s, nonlinear indices of 1e-25 m^2/W, energies of 1e-16 J. Under such a
default the comparison accepts any result. ``tests/conftest.py`` gives
``pytest.approx`` no absolute tolerance when a test gives ``rel=`` alone or compares
with a nonzero number bare, and every NumPy comparison states its ``atol``
(``docs/dev/testing.md`` section 3).
``numpy.testing.assert_allclose`` and ``math.isclose`` default to no absolute tolerance
and need no rule.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREES = ("tests", "native/tests", "gui/tests")
"""The test trees, relative to the repository root."""

NUMPY_NAMES = {"np", "numpy"}
COMPARISONS = {"allclose", "isclose"}
ATOL_POSITION = 3
"""``allclose(a, b, rtol, atol)``: a fourth positional argument is the ``atol``."""


def numpy_comparisons() -> tuple[int, list[str]]:
    """Return how many NumPy comparisons the test trees hold, and those with no atol."""
    seen = 0
    missing = []
    for tree in TREES:
        for path in sorted((ROOT / tree).rglob("*.py")):
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(module):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in COMPARISONS
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in NUMPY_NAMES
                ):
                    continue
                seen += 1
                keywords = {keyword.arg for keyword in node.keywords}
                if "atol" not in keywords and len(node.args) <= ATOL_POSITION:
                    where = path.relative_to(ROOT).as_posix()
                    missing.append(f"{where}:{node.lineno}: {ast.unparse(node.func)}")
    return seen, missing


def test_every_numpy_comparison_states_its_absolute_tolerance():
    seen, missing = numpy_comparisons()
    # The scan finds the comparisons it is about; an empty scan would pass vacuously.
    assert seen > 50
    assert not missing, (
        "np.allclose/np.isclose default to atol=1e-8, which accepts any result on "
        "small SI values; give each an explicit atol (0.0 for a relative comparison):\n"
        + "\n".join(missing)
    )


def test_a_relative_approx_has_no_absolute_tolerance():
    # The helium n2 of 3.6e-25 m^2/W at 2 %: pytest's own default absolute tolerance of
    # 1e-12 would accept 5.45e-25, and any other value of that size.
    assert pytest.approx(3.6e-25, rel=0.02) == 3.6e-25 * 1.019
    assert pytest.approx(3.6e-25, rel=0.02) != 5.45e-25
    # A bare approx of a nonzero number keeps pytest's relative 1e-6 and nothing else.
    assert pytest.approx(1e-13) == 1e-13 * (1 + 5e-7)
    assert pytest.approx(1e-13) != -1e-13
    # A bare approx of zero keeps pytest's defaults, and an absolute tolerance the test
    # gives is kept.
    assert pytest.approx(0.0) == 1e-13
    assert pytest.approx(0.0, rel=1e-6, abs=1e-20) == 1e-21
