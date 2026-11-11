# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Qualify the mandatory Julia CI commands without starting a Julia runtime."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from tools import check_julia

from solprop.backends.julia import _platforms, _setup, _source


@pytest.fixture
def pinned_tree(tmp_path):
    environments = tmp_path / "julia/environments"
    environments.mkdir(parents=True)
    versions = {
        "julia": "1.13.3",
        "PythonCall": "0.9.37",
        "Aqua": "0.8.18",
        "JET": "0.13.1",
        "Runic": "1.13.0",
    }
    (environments / "versions.toml").write_text(
        "".join(f'{name} = "{version}"\n' for name, version in versions.items()),
        encoding="utf-8",
    )
    for environment, packages in (
        ("cpu", ("PythonCall",)),
        ("quality", ("Aqua", "JET")),
        ("formatter", ("Runic",)),
    ):
        directory = environments / environment
        directory.mkdir()
        (directory / "Project.toml").write_text("[deps]\n", encoding="utf-8")
        (directory / "Manifest.toml").write_text(
            'julia_version = "1.13.3"\n'
            + "".join(
                f'[[deps.{package}]]\nversion = "{versions[package]}"\n'
                for package in packages
            ),
            encoding="utf-8",
        )
    return tmp_path


@pytest.fixture
def private_depot(tmp_path, monkeypatch):
    directory = tmp_path / "depot"
    monkeypatch.setenv("SOLPROP_JULIA_DEPOT", str(directory))
    monkeypatch.setattr(check_julia, "_julia", lambda: str(tmp_path / "bin/julia"))
    return directory


def test_metadata_publishes_the_committed_versions(
    pinned_tree, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "github-output"
    output.write_text("existing=value\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    check_julia.main(["metadata", "--root", str(pinned_tree)])
    published = capsys.readouterr().out
    assert published == (
        "julia=1.13.3\nPythonCall=0.9.37\nAqua=0.8.18\nJET=0.13.1\nRunic=1.13.0\n"
    )
    assert output.read_text(encoding="utf-8") == "existing=value\n" + published


@pytest.mark.parametrize(
    "release", ['"1.13"', '"nightly"', "13", '"1.13.3\\nother=value"']
)
def test_metadata_refuses_nonexact_pins(pinned_tree, release):
    path = pinned_tree / "julia/environments/versions.toml"
    path.write_text(f"julia = {release}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="one stable release"):
        check_julia._pins(pinned_tree)


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("cpu", 'julia_version = "1.13.2"\n'),
        ("cpu", 'julia_version = "1.13.3"\n'),
        ("quality", 'julia_version = "1.13.3"\n[[deps.Aqua]]\nversion = "0.8.17"\n'),
        ("formatter", 'julia_version = "1.13.3"\n[[deps.Runic]]\nversion = "1.12.0"\n'),
    ],
)
def test_metadata_refuses_divergent_manifest_pins(pinned_tree, name, replacement):
    (pinned_tree / "julia/environments" / name / "Manifest.toml").write_text(
        replacement, encoding="utf-8"
    )
    with pytest.raises(ValueError, match="differs from"):
        check_julia._pins(pinned_tree)


def test_metadata_errors_name_the_bad_input(pinned_tree, capsys):
    (pinned_tree / "julia/environments/versions.toml").unlink()
    with pytest.raises(SystemExit, match="2"):
        check_julia.main(["metadata", "--root", str(pinned_tree)])
    assert "versions.toml" in capsys.readouterr().err


def test_private_depot_is_mandatory(monkeypatch):
    monkeypatch.delenv("SOLPROP_JULIA_DEPOT", raising=False)
    with pytest.raises(ValueError, match="private Julia CI depot"):
        check_julia._environment()


def test_commands_isolate_julia_configuration(private_depot, tmp_path, monkeypatch):
    monkeypatch.setenv("JULIA_DEPOT_PATH", "foreign-depot:")
    monkeypatch.setenv("JULIA_LOAD_PATH", "foreign-project:@v1.13:@stdlib")
    monkeypatch.setenv("JULIA_CPU_TARGET", "native")
    calls = []
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    check_julia._run(tmp_path, ("julia", "-e", "exit()"), group="Core")
    arguments, options = calls[0]
    assert arguments == (("julia", "-e", "exit()"),)
    assert options["cwd"] == tmp_path
    assert options["check"] is True
    environment = options["env"]
    assert environment["GROUP"] == "Core"
    assert environment["JULIA_DEPOT_PATH"] == str(private_depot)
    assert environment["JULIA_LOAD_PATH"] == "@:@stdlib"
    assert environment["JULIA_CPU_TARGET"] == "generic"
    assert environment["JULIA_REVISE_POLL"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert os.environ["JULIA_CPU_TARGET"] == "native"


@pytest.mark.parametrize(
    ("actual", "bridge", "message"),
    [
        ("1.13.2", "0.9.37", "exact committed Julia"),
        ("1.13.3", "0.9.36", "PythonCall pin"),
    ],
)
def test_runtime_versions_must_match_ci_pins(
    pinned_tree, private_depot, monkeypatch, actual, bridge, message
):
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=actual)
    )
    monkeypatch.setattr(check_julia.importlib.metadata, "version", lambda name: bridge)
    with pytest.raises(ValueError, match=message):
        check_julia._check_runtime(pinned_tree)


def test_missing_julia_is_actionable(monkeypatch):
    monkeypatch.setattr(check_julia.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="pinned Julia executable"):
        check_julia._julia()


def test_setup_publishes_only_a_verified_identity(
    pinned_tree, private_depot, monkeypatch, capsys
):
    calls = []
    monkeypatch.setattr(_source, "source_bundle", lambda: pinned_tree / "julia")
    monkeypatch.setattr(check_julia, "_check_runtime", lambda root: None)
    monkeypatch.setattr(
        _setup,
        "setup",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(setup_id="a" * 64),
    )
    monkeypatch.setattr(_setup, "select_setup", lambda **kwargs: calls.append(kwargs))
    output = pinned_tree / "github-env"
    monkeypatch.setenv("GITHUB_ENV", str(output))
    check_julia.main(["setup", "--root", str(pinned_tree)])
    assert calls == [
        {"julia": Path(check_julia._julia()), "cpu_target": "generic"},
        {"setup_id": "a" * 64, "device": "cpu"},
    ]
    expected = f"SOLPROP_TEST_JULIA_SETUP_ID={'a' * 64}\n"
    assert output.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out == expected


def test_setup_refuses_a_different_managed_pin(pinned_tree, monkeypatch):
    monkeypatch.setattr(_platforms, "JULIA_VERSION", "1.14.0")
    with pytest.raises(ValueError, match="managed setup's Julia or bridge pin"):
        check_julia._setup(pinned_tree)


def test_setup_refuses_another_checkout(pinned_tree, monkeypatch):
    monkeypatch.setattr(_source, "source_bundle", lambda: pinned_tree / "foreign/julia")
    with pytest.raises(ValueError, match="source differs from the Julia CI checkout"):
        check_julia._setup(pinned_tree)


@pytest.mark.parametrize("name", check_julia._RUNTIME_TESTS)
def test_python_phase_refuses_missing_mandatory_probes(pinned_tree, name):
    for required in check_julia._RUNTIME_TESTS:
        path = pinned_tree / required
        path.parent.mkdir(parents=True, exist_ok=True)
        if required != name:
            path.touch()
    with pytest.raises(ValueError, match=name):
        check_julia._python(pinned_tree)


@pytest.mark.parametrize(
    "variable", ["SOLPROP_TEST_JULIA", "SOLPROP_TEST_JULIA_SETUP_ID"]
)
def test_python_phase_refuses_disabled_runtime_tests(
    pinned_tree, monkeypatch, variable
):
    for name in check_julia._RUNTIME_TESTS:
        path = pinned_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("SOLPROP_TEST_JULIA", "1")
    monkeypatch.setenv("SOLPROP_TEST_JULIA_SETUP_ID", "a" * 64)
    monkeypatch.delenv(variable)
    with pytest.raises(ValueError, match=variable):
        check_julia._python(pinned_tree)


def test_python_phase_runs_the_whole_adapter_directory(pinned_tree, monkeypatch):
    for name in check_julia._RUNTIME_TESTS:
        path = pinned_tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setenv("SOLPROP_TEST_JULIA", "1")
    monkeypatch.setenv("SOLPROP_TEST_JULIA_SETUP_ID", "a" * 64)
    calls = []

    def run(root, command):
        calls.append(command)
        Path(command[-1].removeprefix("--junitxml=")).write_text(
            "<testsuite>"
            + "".join(
                f'<testcase classname="tests.backends.julia.{module}" '
                'name="test_real_engine"/>'
                for module in (
                    "test_runtime_subprocess",
                    "test_control_oracle",
                    "test_cpu_session",
                    "test_cpu_history",
                    "test_cpu_observations",
                    "test_cpu_monitor",
                    "test_cpu_group",
                    "test_cpu_dense",
                    "test_cpu_noise",
                    "test_cpu_split",
                    "test_cpu_spatial",
                    "test_cpu_transverse",
                    "test_cpu_probe",
                    "test_cpu_fft",
                    "test_cpu_provider",
                    "test_cpu_material_routes",
                    "test_transfer_aliases",
                )
            )
            + "</testsuite>",
            encoding="utf-8",
        )
        report = Path(command[-1].removeprefix("--junitxml="))
        text = report.read_text(encoding="utf-8")
        report.write_text(
            text.replace(
                "</testsuite>",
                "".join(
                    '<testcase classname="tests.sessions.test_conformance" '
                    f'name="{name}[julia-cpu]"/>'
                    for name in check_julia._CONFORMANCE_REQUIRED
                )
                + "</testsuite>",
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(check_julia, "_run", run)
    check_julia._python(pinned_tree)
    assert calls[0][:4] == (sys.executable, "-m", "pytest", "tests/backends/julia")
    assert calls[0][4:7] == ("tests/sessions/test_conformance.py", "-k", "julia")
    assert calls[0][7] == "--durations=20"
    assert calls[0][8].startswith("--junitxml=")


@pytest.mark.parametrize("result", ("skipped", "failure", "error"))
def test_runtime_evidence_cannot_skip_or_fail(tmp_path, result):
    report = tmp_path / "results.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.backends.julia.test_runtime_subprocess" '
        f'name="test_cpu_runtime"><{result}/></testcase></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="did not pass: test_cpu_runtime"):
        check_julia._verify_runtime_report(report)


def test_runtime_evidence_cannot_be_satisfied_by_mock_tests(tmp_path):
    report = tmp_path / "results.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.backends.julia.test_runtime" '
        'name="test_mocked_runtime"/></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="produced no test results"):
        check_julia._verify_runtime_report(report)


@pytest.mark.parametrize("result", ["failure", "error", "skipped"])
def test_conformance_evidence_refuses_an_unexpected_nonpass(result):
    cases = [
        ElementTree.fromstring(
            '<testcase classname="tests.sessions.test_conformance" '
            f'name="{name}[julia-cpu]"/>'
        )
        for name in check_julia._CONFORMANCE_REQUIRED
    ]
    cases[-1].append(ElementTree.Element(result, message="unexpected missing runtime"))
    with pytest.raises(ValueError, match=r"did not pass|unexpected Julia"):
        check_julia._verify_conformance_report(cases)


def test_conformance_evidence_requires_the_core_contract_cases():
    with pytest.raises(ValueError, match="missing passed Julia conformance"):
        check_julia._verify_conformance_report([])


def test_conformance_evidence_accepts_only_declared_bitwise_skips():
    cases = [
        ElementTree.fromstring(
            '<testcase classname="tests.sessions.test_conformance" '
            f'name="{name}[julia-cpu]"/>'
        )
        for name in check_julia._CONFORMANCE_REQUIRED
    ]
    for message in (
        "the julia-cpu executor promises numerical error bounds, "
        "not bitwise repetition",
    ):
        case = ElementTree.Element(
            "testcase",
            classname="tests.sessions.test_conformance",
            name="optional[julia-cpu]",
        )
        case.append(ElementTree.Element("skipped", message="Skipped: " + message))
        cases.append(case)
    check_julia._verify_conformance_report(cases)


def test_instantiation_refuses_manifest_changes(pinned_tree, monkeypatch):
    manifest = pinned_tree / "julia/environments/formatter/Manifest.toml"
    monkeypatch.setattr(check_julia, "_julia", lambda: "julia")
    monkeypatch.setattr(
        check_julia,
        "_run",
        lambda root, command: manifest.write_text("changed\n", encoding="utf-8"),
    )
    with pytest.raises(
        ValueError, match="instantiation changed pinned formatter/Manifest"
    ):
        check_julia._instantiate(pinned_tree, "formatter")


@pytest.mark.parametrize(
    ("check", "environment", "group"),
    [
        ("format", "formatter", None),
        ("core", "cpu", "Core"),
        ("quality", "quality", "Quality"),
    ],
)
def test_julia_phases_use_pinned_projects_and_bounds_checks(
    pinned_tree, monkeypatch, check, environment, group
):
    test_directory = pinned_tree / "julia/test"
    test_directory.mkdir()
    for name in ("runtests.jl", "quality.jl"):
        (test_directory / name).touch()
    calls = []
    monkeypatch.setattr(check_julia, "_julia", lambda: "julia")
    monkeypatch.setattr(check_julia, "_check_runtime", lambda root: None)
    monkeypatch.setattr(
        check_julia,
        "_run",
        lambda root, command, **kwargs: calls.append((command, kwargs)),
    )
    check_julia.main([check, "--root", str(pinned_tree)])
    command, options = calls[-1]
    assert command[:4] == (
        "julia",
        "--startup-file=no",
        "--history-file=no",
        "--check-bounds=yes",
    )
    assert command[4] == f"--project={pinned_tree / 'julia/environments' / environment}"
    assert options.get("group") == group
    if check == "format":
        assert command[5:] == ("-m", "Runic", "--check", "--diff", "julia")
    elif check == "core":
        assert 'Pkg.test("SolpropJulia"; allow_reresolve=false,' in command[-1]
        assert 'julia_args=["--check-bounds=yes"]' in command[-1]
    else:
        assert "@assert JET.JET_AVAILABLE" in command[-1]
        assert 'include("julia/test/runtests.jl")' in command[-1]
    if check != "core":
        assert "Pkg.instantiate(; allow_autoprecomp=false)" in calls[0][0][-1]


def test_command_failure_is_not_suppressed(pinned_tree, private_depot, monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])

    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        check_julia._run(pinned_tree, ("julia", "-e", "error()"))
    assert caught.value.returncode == 7


@pytest.mark.skipif(
    sys.platform == "win32", reason="CI change detection runs in Linux Bash"
)
@pytest.mark.parametrize("pipefail", [False, True])
def test_large_change_list_still_selects_julia_ci(tmp_path, pipefail):
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
    ).read_text(encoding="utf-8")
    start = workflow.index("          julia=true\n")
    last = '          echo "julia=$julia" | tee -a "$GITHUB_OUTPUT"'
    end = workflow.index(last, start) + len(last)
    script = textwrap.dedent(workflow[start:end])
    (tmp_path / "changed.txt").write_text(
        "julia/src/session.jl\n"
        + "".join(f"src/other/{index:08d}.py\n" for index in range(100_000)),
        encoding="utf-8",
    )
    output = tmp_path / "github-output"
    shell = ["bash", "-e"]
    if pipefail:
        shell.extend(("-o", "pipefail"))
    subprocess.run(
        [*shell, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
    )
    assert output.read_text(encoding="utf-8") == "julia=true\n"
