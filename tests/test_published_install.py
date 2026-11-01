# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise failure evidence and isolation in the published-install check."""

import argparse
import json
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/published_install.py"


@pytest.fixture
def check_script():
    return runpy.run_path(str(SCRIPT))


@pytest.mark.parametrize("version", ["0.1.0", "v0.1.0", "0.2.0rc1"])
def test_published_version_is_an_exact_pin(check_script, version):
    assert check_script["_version"](version) == version.removeprefix("v")


@pytest.mark.parametrize(
    "version", ["", "../0.1.0", "0.1.0; python_version < '3'", "https://x"]
)
def test_published_version_refuses_requirement_expressions(check_script, version):
    with pytest.raises(argparse.ArgumentTypeError, match="one published version"):
        check_script["_version"](version)


def test_failed_command_keeps_its_output_and_fails(check_script, tmp_path):
    (tmp_path / "evidence").mkdir()
    with pytest.raises(subprocess.CalledProcessError) as error:
        check_script["_run"](
            [
                sys.executable,
                "-I",
                "-c",
                "print('install failed'); raise SystemExit(7)",
            ],
            tmp_path,
            "pip-install",
        )
    assert error.value.returncode == 7
    assert (tmp_path / "evidence/pip-install.log").read_text(encoding="utf-8") == (
        "install failed\n"
    )


def test_install_failure_records_platform_and_job_summary(
    check_script, monkeypatch, tmp_path
):
    def fail_environment(_builder, _path):
        raise OSError("could not create the environment")

    monkeypatch.setattr(check_script["venv"].EnvBuilder, "create", fail_environment)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    directory = tmp_path / "install"
    with pytest.raises(OSError, match="could not create"):
        check_script["_check"]("0.1.0", directory, readme_example=False)
    record = json.loads(
        (directory / "evidence/result.json").read_text(encoding="utf-8")
    )
    assert record["status"] == "failed"
    assert record["requested_version"] == "0.1.0"
    assert record["platform"]
    assert record["machine"]
    assert record["elapsed_seconds"] >= 0
    assert "could not create the environment" in summary.read_text(encoding="utf-8")


def test_existing_environment_is_never_reused(check_script, tmp_path):
    with pytest.raises(FileExistsError):
        check_script["_check"]("0.1.0", tmp_path, readme_example=False)


def test_readme_check_requires_the_tagged_example(check_script, monkeypatch, tmp_path):
    (tmp_path / "evidence").mkdir()
    readme = "# Release\n\n```python\nraise SystemExit(0)\n```\n"
    calls = []

    def read_tag(command, directory, name):
        calls.append((command, directory, name))
        return readme

    monkeypatch.setitem(check_script["_readme_example"].__globals__, "_run", read_tag)
    example = check_script["_readme_example"]("0.1.0", tmp_path)
    assert calls[0][0][-2:] == ["show", "v0.1.0:README.md"]
    assert example.read_text(encoding="utf-8") == "raise SystemExit(0)\n"
    readme = "# No example"
    with pytest.raises(ValueError, match="no Python example"):
        check_script["_readme_example"]("0.1.0", tmp_path)
