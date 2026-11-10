# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import tarfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime_types import unchecked
from solprop.backends.julia import (
    _artifacts,
    _identity,
    _platforms,
    _setup,
    _source,
    select_setup,
    setup,
    status,
)
from solprop.backends.julia import command as julia_command
from solprop.exceptions import BackendError

_UUID = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
_TREE = "1" * 40


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _bundle(path):
    _write(path / "Project.toml", 'name = "SolpropJulia"\n')
    _write(path / "bridge.toml", 'engine = "SolpropJulia"\n')
    _write(path / "src" / "SolpropJulia.jl", "module SolpropJulia\nend\n")
    _write(path / "ext" / "PythonCallExt.jl", "module PythonCallExt\nend\n")
    _write(path / "scripts" / "instantiate.jl", "# fixture installer\n")
    _write(path / "environments" / "versions.toml", "# fixture resolution record\n")
    for device in _platforms.DEVICES:
        project = path / "environments" / device
        _write(project / "Project.toml", f'[deps]\nPythonCall = "{_UUID}"\n')
        _write(
            project / "Manifest.toml",
            'manifest_format = "2.0"\n[[deps.PythonCall]]\n'
            f'uuid = "{_UUID}"\nversion = "{_platforms.BRIDGE_VERSION}"\n'
            f'git-tree-sha1 = "{"0" * 40}"\n',
        )
        if device != "cpu":
            vendor = {"cuda": "CUDA", "rocm": "AMDGPU", "metal": "Metal"}[device]
            with (project / "Manifest.toml").open("a", encoding="utf-8") as stream:
                stream.write(f'[[deps.{vendor}]]\nversion = "1.2.3"\n')
    _write(path / "environments" / "cuda" / "runtime.toml", 'cuda_runtime = "13.5"\n')
    return path


def _inventory(record, project, projects, output):
    assert next(name for name, _ in projects) == "cpu"
    _write(
        record.depot / "packages" / "PythonCall" / "fixture" / "Project.toml",
        'name = "PythonCall"\n',
    )
    _write(record.depot / "artifacts" / _TREE / "lib" / "library", "fixture artifact")
    output.write_text(
        f'''schema_version = 1
complete = true
project = "{project.name}"
platform = "fixture-platform"
project_sha256 = "{_source.file_digest(project / "Project.toml")}"
manifest_sha256 = "{_source.file_digest(project / "Manifest.toml")}"
package_count = 1
artifact_selection_count = 1
artifact_verified_count = 1

[[packages]]
name = "PythonCall"
uuid = "{_UUID}"
path = "packages/PythonCall/fixture"
selector = "hook"
selected_artifacts = ["fixture"]

[[artifacts]]
name = "fixture"
package_uuid = "{_UUID}"
tree_hash = "{_TREE}"
path = "artifacts/{_TREE}"
files = ["lib/library"]
''',
        encoding="utf-8",
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "origin" / "julia")
    binary = tmp_path / "runtime" / "bin" / "julia"
    _write(binary, "fixture Julia executable")
    depot = tmp_path / "depot"
    monkeypatch.setenv("SOLPROP_JULIA_DEPOT", str(depot))
    monkeypatch.setattr(_source, "source_bundle", lambda: bundle)
    monkeypatch.setattr(_setup, "_bridge_version", lambda: _platforms.BRIDGE_VERSION)
    monkeypatch.setattr(_platforms, "host_platform", lambda: "linux-x86_64")
    runtime = _setup._Runtime(
        binary, _platforms.JULIA_VERSION, _source.file_digest(binary), "fixture-cpu"
    )
    monkeypatch.setattr(_setup, "_runtime", lambda *args: runtime)
    monkeypatch.setattr(_setup, "_instantiate", _inventory)
    return SimpleNamespace(bundle=bundle, binary=binary, depot=depot, runtime=runtime)


def test_setup_publishes_complete_immutable_stack_and_reuses_it(prepared, monkeypatch):
    record = setup()
    assert record == select_setup()
    assert record.engine.is_dir()
    assert record.engine != prepared.bundle
    assert tuple(name for name, _ in record.projects) == ("cpu",)
    assert record.inventories[0][1].is_file()
    assert status()[0].compatible
    assert status()[0].tested_runtime

    def unexpected(*args, **kwargs):
        pytest.fail("idempotent setup instantiated packages again")

    monkeypatch.setattr(_setup, "_instantiate", unexpected)
    assert setup() == record


def test_selection_never_loads_runtime_or_network(prepared, monkeypatch):
    record = setup()

    def unexpected(*args, **kwargs):
        pytest.fail("availability attempted a runtime or network operation")

    monkeypatch.setattr(subprocess, "run", unexpected)
    monkeypatch.setattr(_setup.urllib.request, "urlopen", unexpected)
    assert select_setup() == record
    assert status()[0].compatible


def test_setup_survives_removing_origin_and_selects_same_installed_sources(
    prepared, monkeypatch
):
    record = setup()
    other_installation = (
        prepared.bundle.parent.parent / "second-installation" / "_bundle"
    )
    shutil.copytree(prepared.bundle, other_installation)
    shutil.rmtree(prepared.bundle.parent)
    monkeypatch.setattr(_source, "source_bundle", lambda: other_installation)
    assert select_setup() == record


def test_source_change_creates_distinct_identity_and_refuses_old(prepared):
    first = setup()
    _write(
        prepared.bundle / "src" / "SolpropJulia.jl", "module SolpropJulia\nx=1\nend\n"
    )
    with pytest.raises(BackendError, match="sources changed"):
        select_setup(setup_id=first.setup_id)
    second = setup()
    assert second.setup_id != first.setup_id
    assert second.engine_digest != first.engine_digest
    assert select_setup() == second
    assert first.engine.is_dir()


def test_preferences_and_installed_device_set_are_identity(prepared):
    first = setup(devices=("cuda",), cuda_runtime="12.8")
    second = setup(devices=("cuda",), cuda_runtime="13.5")
    cpu = setup()
    assert len({first.setup_id, second.setup_id, cpu.setup_id}) == 3
    with pytest.raises(BackendError, match="several compatible setups"):
        select_setup()
    assert select_setup(setup_id=first.setup_id, device="cuda") == first
    preferences = (dict(first.projects)["cuda"] / "LocalPreferences.toml").read_text(
        encoding="utf-8"
    )
    assert preferences.count('version = "12.8"') == 2
    assert "[CUDA_Runtime_jll]" in preferences
    assert "[CUDA_Compiler_jll]" in preferences
    assert "12.8" in preferences


def test_canonical_devices_and_default_cuda_runtime(prepared):
    record = setup(devices=("cuda", "cpu", "cuda"))
    assert tuple(name for name, _ in record.projects) == ("cpu", "cuda")
    assert record.cuda_runtime == "13.5"
    assert setup(devices=("cpu", "cuda")) == record


def test_local_cuda_preference_does_not_guess_runtime(prepared):
    record = setup(devices=("cuda",), cuda_local=True)
    assert record.cuda_runtime is None
    assert "local = true" in (
        dict(record.projects)["cuda"] / "LocalPreferences.toml"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"devices": "cuda"}, "sequence"),
        ({"devices": ["vulkan"]}, "sequence"),
        ({"cuda_local": True}, "require --device"),
        ({"cuda_runtime": "12.8"}, "require --device"),
        ({"devices": ["cuda"], "cuda_runtime": "latest"}, "release"),
        (
            {"devices": ["cuda"], "cuda_runtime": "12.8", "cuda_local": True},
            "alternatives",
        ),
        ({"cpu_target": ""}, "CPU target"),
        ({"cpu_target": "native\ninvalid"}, "CPU target"),
        ({"devices": ["metal"]}, "Apple silicon"),
    ],
)
def test_invalid_setup_inputs_refuse_before_creation(prepared, kwargs, message):
    with pytest.raises(BackendError, match=message):
        setup(**kwargs)
    assert not prepared.depot.exists()


def test_missing_default_cuda_runtime_refuses_without_guess(prepared):
    (prepared.bundle / "environments" / "cuda" / "runtime.toml").unlink()
    with pytest.raises(BackendError, match="recorded default"):
        setup(devices=("cuda",))
    assert setup(devices=("cuda",), cuda_runtime="12.8").cuda_runtime == "12.8"


def test_manifest_bridge_mismatch_refuses_before_creation(prepared):
    manifest = prepared.bundle / "environments" / "cpu" / "Manifest.toml"
    _write(
        manifest,
        manifest.read_text(encoding="utf-8").replace(
            _platforms.BRIDGE_VERSION, "0.9.1"
        ),
    )
    with pytest.raises(BackendError, match="pins PythonCall"):
        setup()
    assert not prepared.depot.exists()


def test_setup_interruption_leaves_no_publication_and_releases_lock(
    prepared, monkeypatch
):
    def interrupt(*args):
        assert status() == ()
        raise KeyboardInterrupt

    monkeypatch.setattr(_setup, "_instantiate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        setup()
    assert status() == ()
    assert list((prepared.depot / "staging").iterdir()) == []
    monkeypatch.setattr(_setup, "_instantiate", _inventory)
    assert setup() == select_setup()


def test_concurrent_setups_publish_once(prepared, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def installer(*args):
        calls.append(args)
        entered.set()
        assert release.wait(10)
        _inventory(*args)

    monkeypatch.setattr(_setup, "_instantiate", installer)
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(setup)
        assert entered.wait(10)
        second = workers.submit(setup)
        assert status() == ()
        release.set()
        assert first.result(timeout=10) == second.result(timeout=10)
    assert len(calls) == 1
    assert len(status()) == 1


def test_status_survives_disappearing_installer_staging_files(prepared, monkeypatch):
    record = setup()
    transient = prepared.depot / "staging" / "removed-by-installer"
    transient.write_bytes(b"temporary staging data")
    original = Path.lstat

    def removed_during_accounting(path, *args, **kwargs):
        if path == transient:
            transient.unlink()
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", removed_during_accounting)
    assert status()[0].compatible
    assert select_setup() == record


def test_cpu_selection_checks_every_installed_environment(prepared):
    record = setup(devices=("cuda",))
    (dict(record.projects)["cuda"] / "LocalPreferences.toml").write_text(
        "unrecorded = true\n", encoding="utf-8"
    )
    with pytest.raises(BackendError, match="CUDA preferences changed"):
        select_setup(setup_id=record.setup_id, device="cpu")


@pytest.mark.parametrize(
    "missing", ["artifact", "file", "package", "runtime", "inventory"]
)
def test_missing_installed_content_refuses(prepared, missing):
    record = setup()
    if missing == "artifact":
        shutil.rmtree(prepared.depot / "artifacts" / _TREE)
    elif missing == "file":
        (prepared.depot / "artifacts" / _TREE / "lib" / "library").unlink()
    elif missing == "package":
        shutil.rmtree(prepared.depot / "packages" / "PythonCall")
    elif missing == "runtime":
        prepared.binary.unlink()
    else:
        record.inventories[0][1].unlink()
    assert not status()[0].compatible
    with pytest.raises(BackendError, match="cannot run"):
        select_setup(setup_id=record.setup_id)


def test_overrides_refused_even_with_existing_artifacts(prepared):
    record = setup()
    _write(prepared.depot / "artifacts" / "Overrides.toml", "")
    with pytest.raises(BackendError, match=r"Overrides\.toml"):
        select_setup(setup_id=record.setup_id)
    with pytest.raises(BackendError, match=r"Overrides\.toml"):
        setup()


def test_altered_record_fails_identity_check(prepared):
    record = setup()
    path = prepared.depot / "setups" / record.setup_id / "setup.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cpu_target"] = "generic"
    _write(path, json.dumps(data))
    assert status()[0].record is None
    assert "identity" in status()[0].reason


def test_unknown_and_missing_device_refuse(prepared):
    with pytest.raises(BackendError, match="no usable cpu setup"):
        select_setup()
    record = setup()
    with pytest.raises(BackendError, match="no device"):
        select_setup(device="vulkan")
    with pytest.raises(BackendError, match="does not exist"):
        select_setup(setup_id="absent")
    with pytest.raises(BackendError, match="no cuda environment"):
        select_setup(setup_id=record.setup_id, device="cuda")


def test_same_minor_runtime_is_marked_untested(prepared, monkeypatch):
    replacement = _setup._Runtime(
        prepared.binary, "1.13.4", _source.file_digest(prepared.binary), "fixture-cpu"
    )
    monkeypatch.setattr(_setup, "_runtime", lambda *args: replacement)
    record = setup()
    assert select_setup() == record
    assert not status()[0].tested_runtime


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("complete = true", "complete = false", "not completed"),
        ("artifact_verified_count = 1", "artifact_verified_count = 0", "counts"),
        (
            'selected_artifacts = ["fixture"]',
            'selected_artifacts = ["omitted"]',
            "omits",
        ),
        ('files = ["lib/library"]', 'files = ["../escape"]', "unsafe"),
        ('selector = "hook"', 'selector = "unknown"', "selector"),
        ('path = "packages/PythonCall/fixture"', 'path = "/outside"', "unsafe"),
    ],
)
def test_inventory_requires_complete_selected_artifacts(prepared, old, new, message):
    record = setup()
    path = record.inventories[0][1]
    _write(path, path.read_text(encoding="utf-8").replace(old, new))
    with pytest.raises(BackendError, match=message):
        _artifacts.check_inventory(path, dict(record.projects)["cpu"], record.depot)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version = 1", "schema_version = true", "not completed"),
        ('project = "cpu"', 'project = "cuda"', "another environment"),
        (
            'selected_artifacts = ["fixture"]',
            'selected_artifacts = ["fixture", "fixture"]',
            "repeats an artifact",
        ),
        ('selector = "hook"', 'selector = "none"', "selector"),
        (f'tree_hash = "{_TREE}"', 'tree_hash = "invalid"', "tree hash"),
        (
            f'path = "artifacts/{_TREE}"',
            'path = "artifacts/elsewhere"',
            "inconsistent location",
        ),
        (
            'files = ["lib/library"]',
            'files = ["lib/library", "lib/library"]',
            "repeats inventoried files",
        ),
        (
            'name = "PythonCall"',
            'name = "MissingDependency"',
            "omits manifest packages",
        ),
        ('platform = "fixture-platform"', 'platform = ""', "invalid platform"),
        ("artifact_verified_count = 1", "artifact_verified_count = true", "counts"),
    ],
)
def test_malformed_inventory_cannot_claim_completeness(prepared, old, new, message):
    record = setup()
    path = record.inventories[0][1]
    _write(path, path.read_text(encoding="utf-8").replace(old, new))
    with pytest.raises(BackendError, match=message):
        _artifacts.check_inventory(path, dict(record.projects)["cpu"], record.depot)


@pytest.mark.parametrize("kind", ["packages", "artifacts"])
def test_repeated_inventory_identity_is_rejected(prepared, kind):
    record = setup()
    path = record.inventories[0][1]
    content = path.read_text(encoding="utf-8")
    addition = content.split(f"[[{kind}]]", 1)[1].split("[[", 1)[0]
    _write(path, content + f"\n[[{kind}]]" + addition)
    with pytest.raises(BackendError, match="repeats"):
        _artifacts.check_inventory(path, dict(record.projects)["cpu"], record.depot)


def test_inventory_binds_exact_project_and_manifest_bytes(prepared):
    record = setup()
    project = dict(record.projects)["cpu"]
    with (project / "Project.toml").open("a", encoding="utf-8") as stream:
        stream.write("# modified\n")
    with pytest.raises(BackendError, match=r"Project\.toml changed"):
        _artifacts.check_inventory(record.inventories[0][1], project, record.depot)


def test_artifact_path_must_remain_a_directory(prepared):
    setup()
    location = prepared.depot / "artifacts" / _TREE
    shutil.rmtree(location)
    location.write_bytes(b"not a directory")
    with pytest.raises(BackendError, match="not a directory"):
        select_setup()


def test_archive_sha256_and_safe_tar_extraction(tmp_path, monkeypatch):
    content = b"archive bytes"
    monkeypatch.setattr(
        _setup.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(content)
    )
    archive = tmp_path / "download.tar.gz"
    with pytest.raises(BackendError, match="SHA-256"):
        _setup._download(
            _platforms.Release("https://example.test/julia", "0" * 64), archive
        )
    _setup._download(
        _platforms.Release("https://example.test/julia", _source.file_digest(archive)),
        archive,
    )
    unsafe = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe, "w:gz") as bundle:
        item = tarfile.TarInfo("../outside")
        item.size = 1
        bundle.addfile(item, io.BytesIO(b"x"))
    destination = tmp_path / "extract"
    destination.mkdir()
    with pytest.raises(tarfile.FilterError):
        _setup._extract(unsafe, destination)
    assert not (tmp_path / "outside").exists()


def test_zip_extraction_refuses_escaping_paths(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../outside", "x")
    with pytest.raises(BackendError, match="unsafe"):
        _setup._extract(archive, tmp_path / "extract")
    assert not (tmp_path / "outside").exists()


def test_runtime_probe_pins_actual_executable_and_minor(tmp_path, monkeypatch):
    binary = tmp_path / "julia"
    _write(binary, "runtime")
    reply = SimpleNamespace(
        returncode=0,
        stdout=f"{_platforms.JULIA_VERSION}\n{binary}\nfixture\n",
        stderr="",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: reply)
    result = _setup._probe(tmp_path / "launcher")
    assert result.executable == binary
    assert result.binary_digest == _source.file_digest(binary)
    reply.stdout = f"1.12.6\n{binary}\nfixture"
    with pytest.raises(BackendError, match="needs Julia"):
        _setup._probe(binary)


def test_commands_parse_repeated_devices_and_report_status(prepared, capsys):
    parser = argparse.ArgumentParser()
    julia_command.add_arguments(parser)
    args = parser.parse_args(
        ["setup", "--device", "cuda", "--device", "cpu", "--cuda-runtime", "12.8"]
    )
    assert julia_command.run(args) == 0
    output = capsys.readouterr().out
    assert "Prepared Julia setup" in output
    assert "devices cpu, cuda" in output
    assert julia_command.run(parser.parse_args(["status"])) == 0
    assert "Julia depot" in capsys.readouterr().out


def test_empty_status_command_has_actionable_setup_command(prepared):
    assert "Run solprop julia setup" in julia_command.describe()


def test_bridge_availability_uses_metadata_without_import(monkeypatch):
    monkeypatch.setattr(_setup.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(BackendError, match="extra"):
        _setup._bridge_version()
    monkeypatch.setattr(_setup.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(_setup.importlib.metadata, "version", lambda name: "0.1.0")
    with pytest.raises(BackendError, match="juliacall=="):
        _setup._bridge_version()
    monkeypatch.setattr(
        _setup.importlib.metadata, "version", lambda name: _platforms.BRIDGE_VERSION
    )
    assert _setup._bridge_version() == _platforms.BRIDGE_VERSION


def test_identity_does_not_depend_on_dictionary_insertion_order():
    assert _identity.identity({"first": 1, "second": 2}) == _identity.identity(
        {"second": 2, "first": 1}
    )


@pytest.mark.parametrize("missing", ["artifact", "package"])
def test_explicit_setup_restores_only_wholly_missing_shared_trees(prepared, missing):
    record = setup()
    root = prepared.depot / "setups" / record.setup_id
    before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    if missing == "artifact":
        shutil.rmtree(prepared.depot / "artifacts" / _TREE)
    else:
        shutil.rmtree(prepared.depot / "packages" / "PythonCall")
    assert not status()[0].compatible
    assert setup() == record
    assert select_setup() == record
    after = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("missing", ["artifact-file", "package-project"])
def test_explicit_setup_does_not_patch_corrupt_shared_trees(
    prepared, missing, monkeypatch
):
    setup()
    if missing == "artifact-file":
        (prepared.depot / "artifacts" / _TREE / "lib" / "library").unlink()
    else:
        (
            prepared.depot / "packages" / "PythonCall" / "fixture" / "Project.toml"
        ).unlink()
    monkeypatch.setattr(
        _setup,
        "_instantiate",
        lambda *args: pytest.fail("modified an existing shared tree"),
    )
    with pytest.raises(BackendError, match="remove"):
        setup()


def test_repair_requires_the_original_inventory_identity(prepared, monkeypatch):
    record = setup()
    shutil.rmtree(prepared.depot / "artifacts" / _TREE)

    def changed_inventory(*args):
        _inventory(*args)
        output = args[-1]
        with output.open("a", encoding="utf-8") as stream:
            stream.write("# changed inventory\n")

    monkeypatch.setattr(_setup, "_instantiate", changed_inventory)
    with pytest.raises(BackendError, match="differs from its immutable"):
        setup()
    assert _identity.read_record(prepared.depot / "setups" / record.setup_id) == record


def test_setup_never_repairs_edited_owned_metadata(prepared, monkeypatch):
    record = setup(devices=("cuda",))
    path = dict(record.projects)["cuda"] / "LocalPreferences.toml"
    _write(path, "modified = true\n")
    monkeypatch.setattr(
        _setup, "_instantiate", lambda *args: pytest.fail("rewrote an immutable setup")
    )
    with pytest.raises(BackendError, match="preferences changed"):
        setup(devices=("cuda",))
    assert path.read_text(encoding="utf-8") == "modified = true\n"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": 2}, "unknown record format"),
        ({"complete": False}, "incomplete"),
        ({"devices": ["cpu", "cuda"]}, "inconsistent environment"),
        ({"cuda_runtime": 42}, "CUDA preferences"),
        ({"cuda_local": "false"}, "CUDA preferences"),
        ({"elapsed_seconds": -1}, "elapsed time"),
        ({"elapsed_seconds": "1"}, "elapsed time"),
        ({"elapsed_seconds": True}, "elapsed time"),
        ({"cpu_target": None}, "invalid cpu_target"),
        ({"projects": {"cpu": "elsewhere/cpu"}}, "owned-directory layout"),
        ({"engine": "elsewhere"}, "owned-directory layout"),
    ],
)
def test_malformed_publications_remain_unavailable(prepared, updates, message):
    record = setup()
    path = prepared.depot / "setups" / record.setup_id / "setup.json"
    content = json.loads(path.read_text(encoding="utf-8"))
    content.update(updates)
    _write(path, json.dumps(content))
    unavailable = status()[0]
    assert not unavailable.compatible
    assert message in unavailable.reason
    assert "unavailable" in julia_command.describe()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"projects": ()}, "installed-device stack"),
        ({"projects": (("cuda", Path("cuda")),)}, "installed-device stack"),
        ({"platform": "win32-x86_64"}, "another platform"),
        ({"depot": Path("elsewhere")}, "another platform or depot"),
        ({"bridge_version": "0.0.1"}, "needs JuliaCall"),
        ({"julia_version": "1.12.6"}, "another Julia minor"),
        ({"julia_binary_digest": "0" * 64}, "executable changed"),
        ({"bundle_digest": "0" * 64}, "templates changed"),
        ({"inventories": ()}, "lacks an inventory"),
    ],
)
def test_record_compatibility_checks_the_entire_execution_identity(
    prepared, changes, message
):
    record = setup()
    with pytest.raises(BackendError, match=message):
        _setup._check_record(
            replace(record, **changes),
            prepared.bundle,
            prepared.depot,
            _platforms.BRIDGE_VERSION,
            record.platform,
        )


@pytest.mark.parametrize("location", ["engine", "environments", "cpu"])
def test_unrecorded_preferences_are_refused(prepared, location):
    record = setup()
    parent = {
        "engine": record.engine,
        "environments": record.engine / "environments",
        "cpu": dict(record.projects)["cpu"],
    }[location]
    _write(parent / "LocalPreferences.toml", "unrecorded = true\n")
    with pytest.raises(BackendError, match="unrecorded"):
        select_setup()


@pytest.mark.parametrize("location", ["engine", "environments", "cpu"])
@pytest.mark.parametrize(
    "alias",
    [
        "JuliaProject.toml",
        "JuliaManifest.toml",
        "Manifest-v1.13.toml",
        "JuliaManifest-v1.13.toml",
        "JuliaLocalPreferences.toml",
    ],
)
def test_julia_loader_aliases_cannot_override_owned_inputs(prepared, location, alias):
    record = setup()
    root = {
        "engine": record.engine,
        "environments": record.engine / "environments",
        "cpu": dict(record.projects)["cpu"],
    }[location]
    _write(root / alias, "unrecorded = true\n")
    with pytest.raises(BackendError, match="unrecorded loader alias"):
        select_setup()


@pytest.mark.skipif(
    _setup.sys.platform == "win32", reason="creating symlinks needs Windows privileges"
)
@pytest.mark.parametrize(
    "name", ["LocalPreferences.toml", "JuliaLocalPreferences.toml"]
)
def test_dangling_preference_aliases_are_refused(prepared, name):
    record = setup()
    project = dict(record.projects)["cpu"]
    (project / name).symlink_to(project / "absent")
    with pytest.raises(BackendError, match="unrecorded"):
        select_setup()


@pytest.mark.parametrize("filename", ["Project.toml", "JuliaProject.toml"])
def test_ancestor_workspace_cannot_redirect_a_published_environment(prepared, filename):
    record = setup()
    project = dict(record.projects)["cpu"]
    relative = project.relative_to(prepared.depot).as_posix()
    _write(prepared.depot / filename, f'[workspace]\nprojects = ["{relative}"]\n')
    with pytest.raises(BackendError, match="redirected by ancestor workspace"):
        select_setup()


@pytest.mark.parametrize("members", ['["unrelated"]', '"unrelated"'])
def test_unrelated_ancestor_workspace_is_allowed(prepared, members):
    record = setup()
    _write(prepared.depot / "Project.toml", f"[workspace]\nprojects = {members}\n")
    assert select_setup() == record


def test_malformed_ancestor_workspace_refuses_before_startup(prepared):
    setup()
    _write(prepared.depot / "Project.toml", "[workspace]\nprojects = [42]\n")
    with pytest.raises(BackendError, match="ancestor workspace has invalid projects"):
        select_setup()


def test_workspace_search_honors_julias_home_boundary(prepared, monkeypatch):
    record = setup()
    monkeypatch.setattr(Path, "home", lambda: prepared.depot)
    project = dict(record.projects)["cpu"]
    outside = prepared.depot.parent / "Project.toml"
    relative = project.relative_to(outside.parent).as_posix()
    _write(outside, f'[workspace]\nprojects = ["{relative}"]\n')
    assert select_setup() == record


@pytest.mark.skipif(
    _setup.sys.platform == "win32",
    reason="creating symlinks needs Windows privileges",
)
def test_workspace_search_uses_julias_lexical_symlink_home(prepared, monkeypatch):
    record = setup()
    alias = prepared.depot.parent / "home-alias"
    alias.symlink_to(prepared.depot, target_is_directory=True)
    monkeypatch.setattr(Path, "home", lambda: alias)
    project = dict(record.projects)["cpu"]
    ancestor = prepared.depot.parent / "Project.toml"
    relative = project.relative_to(ancestor.parent).as_posix()
    _write(ancestor, f'[workspace]\nprojects = ["{relative}"]\n')
    with pytest.raises(BackendError, match="redirected by ancestor workspace"):
        select_setup()


def test_changed_retained_sources_and_templates_are_refused(prepared):
    record = setup()
    source = record.engine / "src" / "SolpropJulia.jl"
    original = source.read_text(encoding="utf-8")
    _write(source, "changed Julia source")
    with pytest.raises(BackendError, match="engine sources changed"):
        select_setup()
    _write(source, original)
    _write(record.engine / "scripts" / "instantiate.jl", "changed installer")
    with pytest.raises(BackendError, match="templates changed"):
        select_setup()


def test_changed_inventory_is_refused_before_reading_it(prepared):
    record = setup()
    _write(record.inventories[0][1], "invalid TOML!")
    with pytest.raises(BackendError, match="inventory changed"):
        select_setup()


def test_vendor_prerequisites_are_rechecked_for_selection(prepared, monkeypatch):
    record = setup(devices=("cuda",))
    monkeypatch.setattr(
        _platforms,
        "device_refusal",
        lambda device, host: "missing GPU prerequisite" if device == "cuda" else None,
    )
    assert status()[0].compatible
    assert status()[0].prerequisites == (
        ("cpu", None),
        ("cuda", "missing GPU prerequisite"),
    )
    assert select_setup(device="cpu") == record
    with pytest.raises(BackendError, match="missing GPU prerequisite"):
        select_setup(device="cuda")


def test_status_reports_engine_and_vendor_versions(prepared):
    setup(devices=("cuda",))
    assert status()[0].package_versions == (
        ("SolpropJulia", "unversioned source"),
        ("CUDA", "1.2.3"),
    )
    assert "Julia checks and recompiles" in julia_command.describe()


def test_missing_vendor_package_is_reported_from_retained_manifest(prepared):
    path = prepared.bundle / "environments" / "cuda" / "Manifest.toml"
    _write(
        path,
        path.read_text(encoding="utf-8").replace("[[deps.CUDA]]", "[[deps.Unrelated]]"),
    )
    setup(devices=("cuda",))
    assert "must pin one CUDA" in status()[0].reason


def test_setup_wraps_filesystem_and_runtime_timeout_errors(prepared, monkeypatch):
    def failed(*args):
        raise subprocess.TimeoutExpired("julia", 30)

    monkeypatch.setattr(_setup, "_runtime", failed)
    with pytest.raises(BackendError, match="setup failed"):
        setup()
    (prepared.bundle / "Project.toml").unlink()
    monkeypatch.setattr(_setup, "_runtime", lambda *args: prepared.runtime)
    with pytest.raises(BackendError, match="setup failed"):
        setup()


def test_source_change_while_copying_is_refused(prepared, monkeypatch):
    original_copy = shutil.copyfile

    def change_during_copy(source, destination):
        result = original_copy(source, destination)
        if source.name == "SolpropJulia.jl":
            _write(destination, "changed source")
        return result

    monkeypatch.setattr(shutil, "copyfile", change_during_copy)
    with pytest.raises(BackendError, match="sources changed while setup"):
        setup()
    assert status() == ()


def test_manifest_requires_one_paired_bridge(prepared):
    path = prepared.bundle / "environments" / "cpu" / "Manifest.toml"
    _write(path, "[deps]\n")
    with pytest.raises(BackendError, match="exactly one PythonCall"):
        setup()


def test_missing_bridge_package_metadata_is_actionable(monkeypatch):
    monkeypatch.setattr(_setup.importlib.util, "find_spec", lambda name: object())

    def missing(name):
        raise _setup.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_setup.importlib.metadata, "version", missing)
    with pytest.raises(BackendError, match="no package metadata"):
        _setup._bridge_version()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cuda_local": "yes"},
        {"cpu_target": 4},
        {"devices": ("cuda",), "cuda_runtime": 13.4},
    ],
)
def test_malformed_request_values_are_named_in_production(prepared, kwargs):
    arguments = {
        "devices": ("cpu",),
        "cuda_runtime": None,
        "cuda_local": False,
        "cpu_target": "native",
        "bundle": prepared.bundle,
    } | kwargs
    with pytest.raises(BackendError):
        unchecked(_setup._request)(**arguments)


def test_installer_environment_has_only_owned_projects_and_depot(prepared, monkeypatch):
    record = setup(devices=("cuda",))
    project = dict(record.projects)["cuda"]
    calls = []

    def invoke(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", invoke)
    # The fixture replaces only the installer, so retrieve the original implementation.
    _REAL_INSTANTIATE(record, project, record.projects, project / "inventory.toml")
    command, options = calls[0]
    assert command[0] == str(record.julia_executable)
    assert f"--project={project}" in command
    assert command[-2] == str(record.engine / "scripts" / "instantiate.jl")
    environment = options["env"]
    assert environment["JULIA_DEPOT_PATH"] == str(record.depot)
    assert environment["JULIA_LOAD_PATH"].split(_setup.os.pathsep) == [
        str(path) for _, path in record.projects
    ] + ["@stdlib"]
    assert environment["JULIA_CONDAPKG_BACKEND"] == "Null"
    assert environment["JULIA_PYTHONCALL_EXE"] == _setup.sys.executable
    assert environment["JULIA_PKG_PRECOMPILE_AUTO"] == "0"
    assert environment["JULIA_CPU_TARGET"] == "native"
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7)
    )
    with pytest.raises(BackendError, match="exit status 7"):
        _REAL_INSTANTIATE(record, project, record.projects, project / "inventory.toml")


_REAL_INSTANTIATE = _setup._instantiate
