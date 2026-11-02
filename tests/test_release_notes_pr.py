# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Exercise the rolling PR against isolated local Git repositories, without a token."""

import json

import pytest
from tests.test_release_notes import CHANGELOG
from tools import update_release_notes as updater


@pytest.fixture
def repository(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    updater._git("init", "--bare", str(remote))
    updater._git("init", "-b", "main")
    updater._git("config", "user.name", "Release test")
    updater._git("config", "user.email", "release@example.org")
    updater._git("config", "commit.gpgsign", "false")
    updater._git("config", "core.autocrlf", "false")
    updater._git("remote", "add", "origin", str(remote))
    (checkout / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (checkout / "CITATION.cff").write_text("version: 0.1.0\n", encoding="utf-8")
    (checkout / "changelog.d").mkdir()
    (checkout / "changelog.d" / "README.md").write_text("Notes\n", encoding="utf-8")
    (checkout / "changelog.d" / "one.fixed.md").write_text(
        "Correction.\n", encoding="utf-8"
    )
    _save_main()
    return checkout


def _save_main():
    updater._git("add", ".")
    updater._git("commit", "-m", "Implementation change")
    updater._git("push", "origin", "main")
    return updater._git("rev-parse", "HEAD")


def test_refresh_tracks_main_and_preserves_history_and_checkout(repository):
    base = updater._git("rev-parse", "HEAD")
    first = updater._commit(base, None)
    assert first is not None
    assert updater._git("rev-list", "--parents", "-1", first).split() == [first, base]
    assert updater._commit(base, first) is None
    assert updater._show(first, "CHANGELOG.md").count("- Correction.") == 1
    assert (
        updater._git("ls-tree", "--name-only", first, "changelog.d/one.fixed.md") == ""
    )
    # A corrected old fragment, a new note and new code all come from current main.
    (repository / "changelog.d" / "one.fixed.md").write_text(
        "Corrected text.\n", encoding="utf-8"
    )
    (repository / "changelog.d" / "two.added.md").write_text(
        "New feature.\n", encoding="utf-8"
    )
    (repository / "code.py").write_text("answer = 42\n", encoding="utf-8")
    latest = _save_main()
    second = updater._commit(latest, first)
    assert second is not None
    assert updater._git("rev-list", "--parents", "-1", second).split() == [
        second,
        first,
        latest,
    ]
    notes = updater._show(second, "CHANGELOG.md")
    assert notes.count("Corrected text.") == 1
    assert "Correction." not in notes
    assert notes.count("New feature.") == 1
    assert (
        notes[notes.index("## [0.1.0]") :] == CHANGELOG[CHANGELOG.index("## [0.1.0]") :]
    )
    assert updater._show(second, "code.py") == "answer = 42\n"
    assert updater._git("status", "--porcelain") == ""
    assert updater._git("rev-parse", "HEAD") == latest
    # Both pushes succeed without force, including the update of an existing branch.
    updater._git("push", "origin", f"{first}:refs/heads/release")
    updater._git("push", "origin", f"{second}:refs/heads/release")


@pytest.mark.parametrize("preserve_message", [False, True])
def test_manual_branch_edits_are_not_overwritten(repository, preserve_message):
    base = updater._git("rev-parse", "HEAD")
    generated = updater._commit(base, None)
    assert generated is not None
    # Even a human commit that preserves the generator's trailer must match its tree.
    message = (
        updater._git("show", "-s", "--format=%B", generated)
        if preserve_message
        else "Prepare release"
    )
    manual = updater._run(
        "git",
        "commit-tree",
        updater._git("rev-parse", "HEAD^{tree}"),
        "-p",
        generated,
        input_text=message,
    )
    with pytest.raises(ValueError, match="manual edits"):
        updater._commit(base, manual)
    assert updater._git("status", "--porcelain") == ""


@pytest.mark.parametrize(
    "state,draft", [("CLOSED", True), ("MERGED", False), ("OPEN", False)]
)
def test_closed_or_ready_pr_pauses_before_pushing(
    repository, monkeypatch, state, draft
):
    original_run = updater._run

    def run(*args, input_text=None):
        if args[:3] == ("gh", "pr", "list"):
            return json.dumps([{"number": 1, "state": state, "isDraft": draft}])
        assert "push" not in args
        return original_run(*args, input_text=input_text)

    monkeypatch.setattr(updater, "_run", run)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    updater.main()


def test_create_retry_and_refresh_keep_one_draft(repository, monkeypatch):
    original_run = updater._run
    pulls = []
    created = []

    def run(*args, input_text=None):
        if args[:3] == ("gh", "pr", "list"):
            return json.dumps(pulls)
        if args[:3] == ("gh", "pr", "create"):
            assert "--draft" in args
            assert "--body-file" in args
            created.append(input_text)
            return "https://example.org/pull/1"
        return original_run(*args, input_text=input_text)

    monkeypatch.setattr(updater, "_run", run)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    # Retry after a pushed branch but failed PR creation must still create the PR.
    base = updater._git("rev-parse", "HEAD")
    commit = updater._commit(base, None)
    assert commit is not None
    updater._git("push", "origin", f"{commit}:refs/heads/chore/release-after-0.1.0")
    updater.main()
    assert len(created) == 1
    pulls.append({"number": 1, "state": "OPEN", "isDraft": True})
    updater.main()
    assert len(created) == 1
    (repository / "changelog.d" / "two.fixed.md").write_text(
        "Another correction.\n", encoding="utf-8"
    )
    _save_main()
    updater.main()
    head = updater._git(
        "ls-remote", "--heads", "origin", "chore/release-after-0.1.0"
    ).split()[0]
    assert updater._show(head, "CHANGELOG.md").count("Another correction.") == 1
    assert len(created) == 1


def test_no_pending_notes_creates_no_commit(repository):
    (repository / "changelog.d" / "one.fixed.md").unlink()
    base = _save_main()
    assert updater._commit(base, None) is None
