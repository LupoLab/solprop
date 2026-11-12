# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The size report counts lines, private docstrings, imports and clones as stated."""

from pathlib import Path

from tools import size_report

SOURCE = '''"""A public module."""

import os  # noqa: F401, E501 - two codes

__all__ = [
    "public",
]


def public(x):
    """Return x.

    Parameters
    ----------
    x : float
        A value.
    """

    # a comment
    def _inner(y):
        """Return y.

        Returns
        -------
        float
            The value.
        """
        return y

    def shown(y):
        """Return y: nested, but public by numpydoc's rule."""
        return y

    return shown(_inner(x))


def _private(x):
    """Return x, sectioned.

    Parameters
    ----------
    x : float
        A value.
    """
    return x


class _Hidden:
    """One line."""

    def method(self):
        """Return nothing."""
'''


def _measured(tmp_path: Path) -> tuple[size_report._Package, size_report._Report]:
    path = tmp_path / "mod.py"
    path.write_text(SOURCE, encoding="utf-8")
    package, report = size_report._Package(), size_report._Report()
    size_report._measure_file(size_report._read(path), package, report, "mod.py")
    return package, report


def test_lines_split_into_code_docstrings_comments_and_blanks(tmp_path):
    package, _ = _measured(tmp_path)
    lines = SOURCE.splitlines()
    assert package.total == len(lines)
    assert package.blank == sum(1 for line in lines if not line.strip()) - 3
    assert package.comment == 1  # the noqa shares its line with code
    # Public by numpydoc's rule: the module, `public` and `public.shown` (1 + 7 + 1).
    # Private: `public._inner`, `_private`, `_Hidden`, `_Hidden.method` (7 + 7 + 1 + 1).
    assert package.public_doc == 9
    assert package.private_doc == 16
    total = package.code + package.public_doc + package.private_doc
    assert total + package.comment + package.blank == package.total


def test_sectioned_private_docstrings_and_long_docstrings_are_counted(tmp_path):
    package, _ = _measured(tmp_path)
    assert package.sectioned_private == 2  # _inner and _private; public is not counted
    # _inner, _private and method have docstrings longer than their bodies.
    assert package.doc_over_body == 3


def test_noqa_codes_and_all_lines_are_counted(tmp_path):
    package, report = _measured(tmp_path)
    assert report.noqa == {"F401": 1, "E501": 1}
    assert package.all_lines == 3
    names = [name for name, _ in report.largest_functions]
    assert "mod.py:mod.public._inner" in names
    assert "mod.py:mod._Hidden" not in names  # functions only


def test_private_imports_across_packages_only(tmp_path):
    root = tmp_path / "pkg"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "_shared.py").write_text("", encoding="utf-8")
    path = root / "a" / "m.py"
    path.write_text(
        "from pkg.b.x import _hidden, shown\n"
        "from pkg.b._y import shown_too\n"
        "from pkg.a.z import _same_package\n"
        "from pkg._shared import public, _private\n"
        "from pkg import _shared, b\n"
        "import pkg.b._z\n"
        "import pkg.b.w, pkg.a._v\n",
        encoding="utf-8",
    )
    found = size_report._private_imports(size_report._read(path), root, "m.py")
    assert sorted(entry.split(" ")[1] for entry in found) == [
        "pkg._shared",
        "pkg._shared._private",
        "pkg.b._y.shown_too",
        "pkg.b._z",
        "pkg.b.x._hidden",
    ]


def test_clones_find_a_copied_window_and_its_pair():
    tokens = [f"t{i}" for i in range(size_report._WINDOW)]
    unique = [f"u{i}" for i in range(size_report._WINDOW)]
    percent, pairs = size_report._clones({"a": tokens, "b": list(tokens), "c": unique})
    assert percent == 100.0 * 2 / 3
    assert pairs == [("a", "b", 1)]


def _budget(**allow: int) -> dict:
    return {
        "private_docstrings": {"max_sectioned": 2},
        "modules": {
            "max_lines": 10,
            "headroom_percent": 2,
            "headroom_lines": 5,
            "allow": {f"src/solprop/{k}.py": v for k, v in allow.items()},
        },
        "private_imports": {"max_cross_package": 0},
    }


def test_check_reports_each_exceeded_cap():
    report = size_report._Report()
    report.packages["driver"] = size_report._Package(sectioned_private=3)
    report.module_lines = {"src/solprop/big.py": 11, "src/solprop/old.py": 21}
    report.private_imports = ["src/solprop/a.py:1 solprop.b._c"]
    budget = _budget(old=20)
    failures = size_report._check(report, budget)
    assert len(failures) == 4
    assert "src/solprop/big.py has 11 lines (cap 10)" in failures
    assert "src/solprop/old.py has 21 lines (cap 20)" in failures
    budget = _budget(old=21)
    budget["private_docstrings"]["max_sectioned"] = 3
    budget["modules"]["max_lines"] = 11
    budget["private_imports"]["max_cross_package"] = 1
    assert size_report._check(report, budget) == []


def test_check_refuses_stale_allow_entries():
    report = size_report._Report()
    report.module_lines = {"src/solprop/old.py": 300, "src/solprop/small.py": 8}
    # The headroom of a 300-line module is max(5, 2 % of 300) = 6 lines; an entry goes
    # stale more than three headrooms (18 lines) above the module's size.
    assert size_report._check(report, _budget(old=318)) == []
    failures = size_report._check(report, _budget(old=319, small=12, gone=20))
    assert failures == [
        "src/solprop/old.py has 300 lines, so its allow entry 319 is stale: "
        "set it to 306",
        "src/solprop/small.py is under 10 lines: remove its allow entry",
        "src/solprop/gone.py is allowed 20 lines but no longer exists",
    ]
