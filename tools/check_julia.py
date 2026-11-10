# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run the separate, pinned Julia CI checks without invoking the full Python gate."""

import argparse
import importlib.metadata
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Sequence
from pathlib import Path
from xml.etree import ElementTree

__all__ = ["main"]

_ROOT = Path(__file__).resolve().parents[1]
_CHECKS = ("metadata", "setup", "format", "core", "quality", "python")
_RUNTIME_TESTS = (
    "tests/backends/julia/test_runtime_subprocess.py",
    "tests/backends/julia/runtime_probe.py",
)


def _require_files(root: Path, names: Sequence[str]) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise ValueError("missing mandatory Julia check inputs: " + ", ".join(missing))


def _pins(root: Path) -> dict[str, str]:
    metadata = root / "julia/environments/versions.toml"
    values = tomllib.loads(metadata.read_text(encoding="utf-8"))
    pins = {}
    for name in ("julia", "PythonCall", "Aqua", "JET", "Runic"):
        value = values.get(name)
        if not isinstance(value, str) or re.fullmatch(r"\d+\.\d+\.\d+", value) is None:
            raise ValueError(f"{metadata}: {name} must pin one stable release")
        pins[name] = value
    for environment, packages in (
        ("cpu", ("PythonCall",)),
        ("quality", ("Aqua", "JET")),
        ("formatter", ("Runic",)),
    ):
        path = metadata.parent / environment / "Manifest.toml"
        manifest = tomllib.loads(path.read_text(encoding="utf-8"))
        if manifest.get("julia_version") != pins["julia"]:
            raise ValueError(f"{path}: the manifest Julia version differs from its pin")
        for package in packages:
            entries = manifest.get("deps", {}).get(package, [])
            if len(entries) != 1 or entries[0].get("version") != pins[package]:
                raise ValueError(f"{path}: {package} differs from its committed pin")
    return pins


def _publish(variable: str, content: str) -> None:
    destination = os.environ.get(variable)
    if destination:
        with Path(destination).open("a", encoding="utf-8") as stream:
            stream.write(content)
    sys.stdout.write(content)


def _environment() -> dict[str, str]:
    depot = os.environ.get("SOLPROP_JULIA_DEPOT")
    if not depot:
        raise ValueError("set SOLPROP_JULIA_DEPOT to the private Julia CI depot")
    return {
        **os.environ,
        "JULIA_DEPOT_PATH": str(Path(depot).resolve()),
        "JULIA_LOAD_PATH": "@:@stdlib",
        "JULIA_CPU_TARGET": "generic",
        "JULIA_REVISE_POLL": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _julia() -> str:
    executable = shutil.which("julia")
    if executable is None:
        raise ValueError("the pinned Julia executable must be on PATH")
    return str(Path(executable).resolve())


def _run(root: Path, command: Sequence[str], *, group: str | None = None) -> None:
    environment = _environment()
    if group is not None:
        environment["GROUP"] = group
    sys.stdout.write(shlex.join(command) + "\n")
    sys.stdout.flush()
    subprocess.run(command, cwd=root, env=environment, check=True)


def _command(root: Path, environment: str, *arguments: str) -> tuple[str, ...]:
    return (
        _julia(),
        "--startup-file=no",
        "--history-file=no",
        "--check-bounds=yes",
        f"--project={root / 'julia/environments' / environment}",
        *arguments,
    )


def _check_runtime(root: Path) -> None:
    pins = _pins(root)
    result = subprocess.run(
        [_julia(), "--startup-file=no", "--history-file=no", "-e", "print(VERSION)"],
        cwd=root,
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != pins["julia"]:
        raise ValueError("CI must use the exact committed Julia release")
    if importlib.metadata.version("juliacall") != pins["PythonCall"]:
        raise ValueError("the installed juliacall differs from the PythonCall pin")


def _setup(root: Path) -> None:
    # Keep metadata and the standalone Julia checks usable before solprop is installed.
    from solprop.backends.julia import (  # noqa: PLC0415 - metadata precedes installation
        _platforms,
        _source,
    )
    from solprop.backends.julia._setup import (  # noqa: PLC0415 - metadata precedes installation
        select_setup,
        setup,
    )

    pins = _pins(root)
    if (
        pins["julia"],
        pins["PythonCall"],
    ) != (_platforms.JULIA_VERSION, _platforms.BRIDGE_VERSION):
        raise ValueError(
            "managed setup's Julia or bridge pin differs from versions.toml"
        )
    if _source.source_bundle().resolve() != root / "julia":
        raise ValueError(
            "the installed solprop source differs from the Julia CI checkout"
        )
    _check_runtime(root)
    record = setup(julia=Path(_julia()), cpu_target="generic")
    select_setup(setup_id=record.setup_id, device="cpu")
    _publish("GITHUB_ENV", f"SOLPROP_TEST_JULIA_SETUP_ID={record.setup_id}\n")


def _instantiate(root: Path, environment: str) -> None:
    directory = root / "julia/environments" / environment
    inputs = {
        name: (directory / name).read_bytes()
        for name in ("Project.toml", "Manifest.toml")
    }
    _run(
        root,
        _command(
            root,
            environment,
            "-e",
            "using Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile()",
        ),
    )
    for name, expected in inputs.items():
        if (directory / name).read_bytes() != expected:
            raise ValueError(f"instantiation changed pinned {environment}/{name}")


def _format(root: Path) -> None:
    _instantiate(root, "formatter")
    _run(root, _command(root, "formatter", "-m", "Runic", "--check", "--diff", "julia"))


def _core(root: Path) -> None:
    _require_files(root, ("julia/test/runtests.jl",))
    # Pkg 1.13 does not forward the parent's bounds option to its test subprocess.
    # Name it in julia_args as well as in the parent command.
    _run(
        root,
        _command(
            root,
            "cpu",
            "-e",
            'using Pkg; Pkg.test("SolpropJulia"; allow_reresolve=false, '
            'julia_args=["--check-bounds=yes"])',
        ),
        group="Core",
    )


def _quality(root: Path) -> None:
    _require_files(root, ("julia/test/runtests.jl", "julia/test/quality.jl"))
    _instantiate(root, "quality")
    _run(
        root,
        _command(
            root,
            "quality",
            "-e",
            'import JET; @assert JET.JET_AVAILABLE; include("julia/test/runtests.jl")',
        ),
        group="Quality",
    )


def _python(root: Path) -> None:
    _require_files(root, _RUNTIME_TESTS)
    if os.environ.get("SOLPROP_TEST_JULIA") != "1":
        raise ValueError("the Julia lane requires SOLPROP_TEST_JULIA=1")
    if not os.environ.get("SOLPROP_TEST_JULIA_SETUP_ID"):
        raise ValueError(
            "the Julia lane requires a published SOLPROP_TEST_JULIA_SETUP_ID"
        )
    # The directory includes each future conformance module without maintaining a
    # second list. Fresh-runtime tests create their own child processes and never
    # initialize Julia in the pytest process, which also runs isolated adapter mocks.
    with tempfile.TemporaryDirectory(prefix="solprop-julia-ci-") as directory:
        report = Path(directory) / "results.xml"
        _run(
            root,
            (
                sys.executable,
                "-m",
                "pytest",
                "tests/backends/julia",
                f"--junitxml={report}",
            ),
        )
        _verify_runtime_report(report)


def _verify_runtime_report(report: Path) -> None:
    cases = [
        case
        for case in ElementTree.parse(report).iter("testcase")
        if "test_runtime_subprocess" in case.get("classname", "").split(".")
    ]
    if not cases:
        raise ValueError(
            "the mandatory runtime subprocess suite produced no test results"
        )
    for case in cases:
        if any(
            case.find(outcome) is not None
            for outcome in ("skipped", "failure", "error")
        ):
            raise ValueError(
                f"mandatory runtime subprocess test did not pass: {case.get('name')}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Run one mandatory Julia CI phase.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command arguments, or the process arguments by default.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=_CHECKS)
    parser.add_argument("--root", type=Path, default=_ROOT)
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        pins = _pins(root)
        if arguments.check == "metadata":
            _publish(
                "GITHUB_OUTPUT",
                "".join(f"{key}={value}\n" for key, value in pins.items()),
            )
            return
        if arguments.check == "setup":
            _setup(root)
            return
        _check_runtime(root)
        checks = {
            "format": _format,
            "core": _core,
            "quality": _quality,
            "python": _python,
        }
        checks[arguments.check](root)
    except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
