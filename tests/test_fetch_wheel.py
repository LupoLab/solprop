# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Which CI artifact ``poe fetch-wheel`` installs for a checkout."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "native" / "scripts" / "fetch_wheel.py"
spec = importlib.util.spec_from_file_location("fetch_wheel", SCRIPT)
assert spec is not None and spec.loader is not None
fetch_wheel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch_wheel)


def artifact(sha, created, *, expired=False):
    return {
        "expired": expired,
        "created_at": created,
        "workflow_run": {"head_sha": sha, "id": 1},
    }


def test_the_newest_artifact_in_the_history_wins():
    artifacts = [artifact("old", "2026-11-01"), artifact("new", "2026-11-02")]
    assert fetch_wheel.newest_in_history(artifacts, lambda _: True) is artifacts[1]


def test_a_commit_outside_the_history_is_passed_over():
    artifacts = [artifact("mine", "2026-11-01"), artifact("other_pr", "2026-11-02")]
    assert (
        fetch_wheel.newest_in_history(artifacts, lambda sha: sha == "mine")
        is artifacts[0]
    )


def test_an_expired_artifact_is_passed_over():
    artifacts = [
        artifact("old", "2026-11-01"),
        artifact("new", "2026-11-02", expired=True),
    ]
    assert fetch_wheel.newest_in_history(artifacts, lambda _: True) is artifacts[0]


def test_no_match_gives_none():
    assert (
        fetch_wheel.newest_in_history([artifact("x", "2026-11-01")], lambda _: False)
        is None
    )


REPOSITORY = Path(__file__).parent.parent


def test_the_modules_a_wheel_depends_on_come_from_its_imports():
    paths = fetch_wheel.imported_modules(REPOSITORY)
    assert "src/solprop/translate/stage.py" in paths
    assert "src/solprop/exceptions.py" in paths
    assert all((REPOSITORY / path).exists() for path in paths)


def git(root, *arguments):
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *arguments],
        check=True,
        cwd=root,
        capture_output=True,
    )


@pytest.fixture
def tree(tmp_path):
    """A small repository in which native imports one solprop module of two."""
    for path, text in {
        "native/src/solprop_native/backend.py": "from solprop.seam import x\n",
        "src/solprop/seam.py": "x = 1\n",
        "src/solprop/other.py": "y = 1\n",
    }.items():
        file = tmp_path / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "built")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return tmp_path, sha


def commit(root):
    git(root, "add", ".")
    git(root, "commit", "-qm", "later")


def test_a_python_only_commit_outside_the_seam_keeps_the_wheel(tree):
    root, sha = tree
    (root / "src/solprop/other.py").write_text("y = 2\n", encoding="utf-8")
    commit(root)
    assert fetch_wheel.difference_from(sha, root) == ""


def test_a_seam_change_refuses_the_wheel(tree):
    root, sha = tree
    (root / "src/solprop/seam.py").write_text("x = 2\n", encoding="utf-8")
    commit(root)
    assert "src/solprop/seam.py" in fetch_wheel.difference_from(sha, root)


def test_a_native_change_refuses_the_wheel(tree):
    root, sha = tree
    (root / "native/src/solprop_native/more.py").write_text("z = 1\n", encoding="utf-8")
    commit(root)
    assert "native/src/solprop_native/more.py" in fetch_wheel.difference_from(sha, root)


def test_an_uncommitted_or_untracked_change_refuses_the_wheel(tree):
    root, sha = tree
    (root / "src/solprop/seam.py").write_text("x = 3\n", encoding="utf-8")
    assert "src/solprop/seam.py" in fetch_wheel.difference_from(sha, root)
    git(root, "checkout", "--", "src/solprop/seam.py")
    (root / "native/new.txt").write_text("new\n", encoding="utf-8")
    assert "native/new.txt" in fetch_wheel.difference_from(sha, root)


def test_an_expired_artifact_leaves_nothing_to_install(monkeypatch, capsys):
    monkeypatch.setattr(fetch_wheel.platform, "system", lambda: "Linux")
    monkeypatch.setattr(fetch_wheel.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        fetch_wheel,
        "_artifacts",
        lambda name: [artifact("a", "2026-11-01T00:00:00Z", expired=True)],
    )
    monkeypatch.setattr(fetch_wheel, "_is_ancestor", lambda sha: True)
    assert fetch_wheel.main() == 1
    assert "tools/setup_light.py" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("system", "machine", "wheel"),
    [
        ("Linux", "x86_64", "linux-x86_64"),
        ("Darwin", "arm64", "macos-arm64"),
        ("Windows", "AMD64", "windows-amd64"),
        ("Windows", "ARM64", "windows-arm64"),
    ],
)
def test_each_platform_asks_for_the_wheel_ci_builds_for_it(
    monkeypatch, capsys, system, machine, wheel
):
    monkeypatch.setattr(fetch_wheel.platform, "system", lambda: system)
    monkeypatch.setattr(fetch_wheel.platform, "machine", lambda: machine)
    monkeypatch.setattr(fetch_wheel, "_artifacts", lambda name: [])
    assert fetch_wheel.main() == 1
    assert f"native-wheel-{wheel} " in capsys.readouterr().err
