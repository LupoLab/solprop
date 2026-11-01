# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check a published release in a fresh pip environment outside the checkout."""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import subprocess
import sys
import time
import venv
from pathlib import Path

_CHECK = """
import importlib.metadata as metadata
import json
import sys

import solprop
import solprop_native
from solprop.backends.registry import resolve_name

versions = {name: metadata.version(name) for name in
            ("solprop", "solprop-native", "optmat", "numpy", "scipy", "pip")}
expected = sys.argv[1]
assert versions["solprop"] == expected, versions
assert versions["solprop-native"] == expected, versions
assert solprop.__version__ == expected, solprop.__version__
assert solprop_native.__version__ == expected, solprop_native.__version__
backend = resolve_name("auto")
assert backend == "native", backend
print(json.dumps({"versions": versions, "backend": backend}, indent=2))
"""


def _version(value: str) -> str:
    version = value.removeprefix("v")
    # Accept release and prerelease spellings, but never requirement syntax or URLs.
    if not re.fullmatch(r"[0-9][0-9A-Za-z.!+_-]*", version):
        raise argparse.ArgumentTypeError(
            "expected one published version, such as 0.1.0"
        )
    return version


def _run(command: list[str], directory: Path, name: str) -> str:
    result = subprocess.run(
        command,
        cwd=directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    (directory / "evidence" / f"{name}.log").write_text(result.stdout, encoding="utf-8")
    sys.stdout.write(result.stdout)
    result.check_returncode()
    return result.stdout


def _readme_example(version: str, directory: Path) -> Path:
    checkout = Path(__file__).resolve().parents[2]
    readme = _run(
        ["git", "-C", str(checkout), "show", f"v{version}:README.md"],
        directory,
        "release-readme",
    )
    match = re.search(r"^```python\s*\n(.*?)^```\s*$", readme, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"v{version}'s README has no Python example")
    source = match[1]
    ast.parse(source)
    example = directory / "evidence" / "readme_example.py"
    example.write_text(source, encoding="utf-8")
    return example


def _check(version: str, directory: Path, *, readme_example: bool) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    evidence = directory / "evidence"
    evidence.mkdir()
    started = time.perf_counter()
    record: dict[str, object] = {
        "requested_version": version,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "readme_example": readme_example,
        "status": "failed",
    }
    try:
        if sys.version_info[:2] != (3, 14):
            raise RuntimeError("the published-install check requires CPython 3.14")
        environment = directory / "environment"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        command = [str(python), "-I", "-m", "pip", "--isolated"]
        _run(
            [
                *command,
                "install",
                "--index-url",
                "https://pypi.org/simple",
                "--no-cache-dir",
                "--report",
                str(evidence / "install-report.json"),
                f"solprop[native]=={version}",
            ],
            directory,
            "pip-install",
        )
        _run([*command, "check"], directory, "pip-check")
        _run([*command, "freeze", "--all"], directory, "pip-freeze")
        installed = _run(
            [str(python), "-I", "-c", _CHECK, version], directory, "backend"
        )
        record["installed"] = json.loads(installed)
        if readme_example:
            example = _readme_example(version, directory)
            os.environ["MPLBACKEND"] = "Agg"
            example_started = time.perf_counter()
            _run([str(python), "-I", str(example)], directory, "readme-example")
            record["readme_seconds"] = time.perf_counter() - example_started
        record["status"] = "passed"
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        record["error"] = str(error)
        raise
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        result = json.dumps(record, indent=2) + "\n"
        (evidence / "result.json").write_text(result, encoding="utf-8")
        summary = f"## Published install: {record['status']}\n\n```json\n{result}```\n"
        (evidence / "summary.md").write_text(summary, encoding="utf-8")
        if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
            with Path(summary_path).open("a", encoding="utf-8") as stream:
                stream.write(summary)


def main() -> None:
    """Install the requested release and retain its qualification evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=_version, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--readme-example", action="store_true")
    arguments = parser.parse_args()
    _check(
        arguments.version,
        arguments.directory.resolve(),
        readme_example=arguments.readme_example,
    )


if __name__ == "__main__":
    main()
