# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Release fragments collect once without changing any released history."""

from pathlib import Path

import pytest
from tools import release_notes

CHANGELOG = """# Changelog

## Unreleased

### Fixed

- An existing correction.

## [0.1.0] - 2026-11-01

### Added

- Released prose, with deliberate trailing spaces.

[0.1.0]: https://example.org/release
""".replace("trailing spaces.", "trailing spaces.  ")


def test_groups_notes_and_preserves_released_history():
    files = {
        "README.md": "Instructions, not a note.",
        "second.fixed.md": "Another correction.\nWith a continuation.",
        "first.fixed.md": "The first correction.",
        "interface.breaking.md": "Rename a keyword.\n\nUse its new name.",
        "feature.added.md": "A new feature.",
    }
    result = release_notes._render(CHANGELOG, release_notes._fragments(files))
    assert (
        result[result.index("## [0.1.0]") :]
        == CHANGELOG[CHANGELOG.index("## [0.1.0]") :]
    )
    assert result.count("### Fixed") == 1
    assert "- Rename a keyword.\n\n  Use its new name." in result
    assert result.index("### Added") < result.index("### Changed (breaking)")
    assert result.index("existing correction") < result.index("first correction")
    assert result.index("first correction") < result.index("Another correction")
    assert "Instructions, not a note" not in result


@pytest.mark.parametrize(
    "name,text",
    [
        ("wrong.md", "A correction."),
        ("note.unknown.md", "A correction."),
        ("../note.fixed.md", "A correction."),
        ("note.fixed.md", "  \n"),
        ("note.fixed.md", "# A heading"),
        ("note.fixed.md", "A correction.\n<<<<<<< branch"),
        ("note.fixed.md", "- A bullet"),
    ],
)
def test_rejects_invalid_fragments(name, text):
    with pytest.raises(ValueError):
        release_notes._fragments({name: text})


@pytest.mark.parametrize(
    "text",
    [
        "# No Unreleased section\n",
        "## Unreleased\n\n## Unreleased\n",
        "## Unreleased\n\n### Unknown\n",
        "## Unreleased\n\n### Fixed\n\n### Fixed\n",
    ],
)
def test_rejects_ambiguous_changelog(text):
    with pytest.raises(ValueError):
        release_notes._render(text, ())


def test_collect_is_idempotent_and_check_and_preview_do_not_write(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "CHANGELOG.md"
    path.write_text(CHANGELOG, encoding="utf-8")
    folder = tmp_path / "changelog.d"
    folder.mkdir()
    note = folder / "one.fixed.md"
    note.write_text("A new correction.\n", encoding="utf-8")
    instructions = folder / "README.md"
    instructions.write_text("Instructions\n", encoding="utf-8")
    for command in ("check", "preview"):
        monkeypatch.setattr("sys.argv", ["release_notes", command])
        release_notes.main()
        assert path.read_text(encoding="utf-8") == CHANGELOG
        assert note.exists()
    assert "A new correction." in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["release_notes", "collect"])
    release_notes.main()
    collected = path.read_bytes()
    assert not note.exists()
    assert instructions.exists()
    release_notes.main()
    assert path.read_bytes() == collected
    assert collected.count(b"A new correction.") == 1


def test_invalid_fragment_leaves_all_inputs_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "changelog.d" / "invalid.md").write_text("Text", encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.md")}
    monkeypatch.setattr("sys.argv", ["release_notes", "collect"])
    with pytest.raises(SystemExit, match="2"):
        release_notes.main()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.md")} == before


def test_current_repository_notes_validate():
    root = Path(__file__).resolve().parents[1]
    release_notes._render(
        (root / "CHANGELOG.md").read_text(encoding="utf-8"),
        release_notes._load(root),
    )
