# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The job that fails on a skip nobody expected (SEP 43, section 7)."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "native" / "ci" / "check_skips.py"
spec = importlib.util.spec_from_file_location("check_skips", SCRIPT)
assert spec is not None and spec.loader is not None
check_skips = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_skips)

CASE = """<testcase classname="tests.test_x" name="test_{name}">{body}</testcase>"""


def junit(tmp_path, *cases):
    path = tmp_path / "junit.xml"
    body = "".join(CASE.format(name=name, body=inner) for name, inner in cases)
    path.write_text(
        f'<testsuites><testsuite name="pytest">{body}</testsuite></testsuites>'
    )
    return path


def skip(reason):
    return f'<skipped type="pytest.skip" message="{reason}"/>'


def test_an_expected_reason_and_a_pass_are_clean(tmp_path, capsys):
    path = junit(
        tmp_path,
        ("a", ""),
        ("b", skip("Pyroraptor is not installed")),
    )
    assert check_skips.main([str(path)]) == 0
    assert "0 unexpected" in capsys.readouterr().out


def test_a_commit_hash_varies_and_nothing_else_does(tmp_path):
    good = skip("93e8ee6a2376 is not in this checkout (squashed or rebased)")
    bad = skip("93e8ee6a is not in this checkout (squashed or rebased)")
    path = junit(tmp_path, ("good", good), ("bad", bad))
    assert [name for name, _ in check_skips.unexpected_skips(path)] == [
        "tests.test_x::test_bad"
    ]


def test_a_skip_for_a_missing_native_fails_and_names_the_test(tmp_path, capsys):
    path = junit(tmp_path, ("lost", skip("solprop-native is not installed")))
    assert check_skips.main([str(path)]) == 1
    out = capsys.readouterr().out
    assert "tests.test_x::test_lost" in out
    assert "solprop-native is not installed" in out


def test_pytest_s_prefix_doesn_t_hide_a_reason(tmp_path):
    path = junit(tmp_path, ("a", skip("Skipped: POSIX SIGKILL")))
    assert check_skips.unexpected_skips(path) == []


def test_an_expected_failure_isn_t_a_skip(tmp_path):
    xfail = '<skipped type="pytest.xfail" message="anything at all"/>'
    assert check_skips.unexpected_skips(junit(tmp_path, ("a", xfail))) == []


def test_no_files_is_a_usage_error(capsys):
    assert check_skips.main([]) == 2
    assert "at least one" in capsys.readouterr().err
