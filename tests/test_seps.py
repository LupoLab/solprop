# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep the enhancement proposals consistent with their index, folders and roadmap.

Each SEP under ``docs/dev/seps/`` carries its status in its header table, and the
status decides where the SEP lives (the open folder or ``archive/``), how the index
lists it and whether the roadmap links it (SEP 0, ``docs/dev/seps/sep-0000-process.md``,
section 4). These tests check that the four agree, that every header is complete, that
the review status agrees with the recorded reviewers and that nothing is accepted
without a human and an AI review (SEP 0 section 6, *Review*), and that no SEP refers to
local files or to the private development archive: a SEP stands alone (SEP 0 section 7).
What a SEP says is for review; only its structure is checked here.
"""

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
SEPS = DOCS / "dev" / "seps"
ARCHIVE = SEPS / "archive"
ROADMAP = DOCS / "dev" / "roadmap.md"

FIELDS = (
    "SEP",
    "Title",
    "Author",
    "Drafting agent",
    "Status",
    "Review status",
    "Reviewers",
    "Type",
    "Created",
    "Resolution",
    "Requires",
    "Supersedes",
    "Superseded-by",
    "Implementation",
    "Manual pages",
)
OPEN_STATUSES = frozenset({"Draft", "Accepted", "In progress", "Deferred", "Active"})
ARCHIVED_STATUSES = frozenset({"Final", "Rejected", "Withdrawn", "Superseded"})
TYPES = frozenset({"Feature", "Change", "Process"})
REVIEW_STATUSES = frozenset({"Not reviewed", "In review", "Reviewed"})
# Statuses a proposal can only reach after review: acceptance and everything after it.
REVIEWED_STATUSES = frozenset({"Accepted", "In progress", "Final"})
PARTS = (
    "## Part I — Proposal",
    "## Part II — Implementation record",
    "## Part III — History",
)

FILENAME = re.compile(r"sep-(\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md")
TITLE = re.compile(r"\A# SEP (\S+) — (.+)$", re.MULTILINE)
FIELD = re.compile(r"^\| \*\*(.+?)\*\* \| (.*?) \|$", re.MULTILINE)
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
REVIEWER = re.compile(r"(.+?) \((human|AI)\)")
INDEX_ROW = re.compile(
    r"^\| \[(\d+)\]\(((?:archive/)?sep-[^)]+\.md)\) \| (.+?) \| (.+?) \| (.+?) \|$",
    re.MULTILINE,
)
ROADMAP_LINK = re.compile(r"\(seps/((?:archive/)?sep-\d{4}-[^)#]+\.md)")
# Absolute or home-relative paths, and links to development records that are not part of
# the public tree (an old repository-root dev/ directory).
LOCAL_REFERENCE = re.compile(
    r"/Users/|/home/|file://|~/|[A-Za-z]:\\|(?<![\w/])dev/(?:briefs|reports)/|(?:\.\./){3}dev/"
)


def header(page: Path) -> dict[str, str]:
    """Return the fields of a SEP's header table."""
    return dict(FIELD.findall(page.read_text()))


def reviewers(fields: dict[str, str]) -> list[tuple[str, str]]:
    """Return a header's reviewers as ``(name, kind)`` pairs; kind is human or AI."""
    if fields["Reviewers"] == "—":
        return []
    entries = [entry.strip() for entry in fields["Reviewers"].split(";")]
    matches = [REVIEWER.fullmatch(entry) for entry in entries]
    assert all(matches), f"write each as 'Name (human)' or 'Model (AI)': {entries}"
    return [(match[1], match[2]) for match in matches if match]


def relative(page: Path) -> str:
    """Return a SEP's path relative to ``docs/dev/seps``, as the index links it."""
    return page.relative_to(SEPS).as_posix()


SEP_FILES = sorted([*SEPS.glob("sep-*.md"), *ARCHIVE.glob("sep-*.md")])
IDS = [relative(page) for page in SEP_FILES]


def test_there_is_a_process_sep():
    assert SEPS / "sep-0000-process.md" in SEP_FILES


@pytest.mark.parametrize("page", SEP_FILES, ids=IDS)
def test_sep_filename_title_and_header_agree(page):
    match = FILENAME.fullmatch(page.name)
    assert match, f"{page.name} is not named sep-NNNN-short-slug.md"
    title = TITLE.search(page.read_text())
    assert title, f"{page.name} does not start with '# SEP N — Title'"
    fields = header(page)
    assert int(title[1]) == int(match[1]) == int(fields["SEP"])
    assert fields["Title"] == title[2]


@pytest.mark.parametrize("page", SEP_FILES, ids=IDS)
def test_sep_header_is_complete(page):
    fields = header(page)
    assert [field for field in FIELDS if field not in fields] == []
    assert fields["Status"] in OPEN_STATUSES | ARCHIVED_STATUSES
    assert fields["Type"] in TYPES
    assert DATE.fullmatch(fields["Created"])
    if fields["Status"] == "Superseded":
        assert fields["Superseded-by"] != "—"
    if fields["Status"] == "Active":
        assert fields["Type"] == "Process"
    assert fields["Author"] not in ("", "—"), "every SEP has a human author"
    assert fields["Drafting agent"] != ""
    assert fields["Review status"] in REVIEW_STATUSES


@pytest.mark.parametrize("page", SEP_FILES, ids=IDS)
def test_the_review_status_agrees_with_the_reviewers(page):
    fields = header(page)
    kinds = {kind for _, kind in reviewers(fields)}
    expected = (
        "Not reviewed"
        if not kinds
        else "Reviewed"
        if kinds == {"human", "AI"}
        else "In review"
    )
    assert fields["Review status"] == expected


@pytest.mark.parametrize("page", SEP_FILES, ids=IDS)
def test_nothing_is_accepted_without_a_human_and_an_ai_review(page):
    fields = header(page)
    if fields["Status"] in REVIEWED_STATUSES:
        assert fields["Review status"] == "Reviewed", (
            f"a {fields['Status']} SEP needs a human and a frontier AI review"
        )


@pytest.mark.parametrize("page", SEP_FILES, ids=IDS)
def test_sep_lives_in_the_folder_its_status_names(page):
    status = header(page)["Status"]
    expected = ARCHIVE if status in ARCHIVED_STATUSES else SEPS
    assert page.parent == expected, f"a {status} SEP belongs in {expected.name}/"


@pytest.mark.parametrize("page", SEP_FILES, ids=IDS)
def test_a_proposal_has_its_three_parts_in_order(page):
    if header(page)["Type"] == "Process":
        pytest.skip("a process SEP is structured by its subject")
    text = page.read_text()
    positions = [text.find(part) for part in PARTS]
    assert -1 not in positions, f"missing one of {PARTS}"
    assert positions == sorted(positions)


def test_sep_numbers_are_unique():
    numbers = [int(header(page)["SEP"]) for page in SEP_FILES]
    assert len(numbers) == len(set(numbers))


def test_the_index_lists_every_sep_with_its_header():
    rows = {
        path: (int(number), title, status, kind)
        for number, path, title, status, kind in INDEX_ROW.findall(
            (SEPS / "index.md").read_text()
        )
    }
    assert sorted(rows) == sorted(IDS)
    for page in SEP_FILES:
        fields = header(page)
        assert rows[relative(page)] == (
            int(fields["SEP"]),
            fields["Title"],
            fields["Status"],
            fields["Type"],
        ), relative(page)


@pytest.mark.parametrize(
    ("section", "statuses"),
    [
        ("Accepted enhancement proposals", {"Accepted", "In progress"}),
        ("Draft enhancement proposals", {"Draft"}),
    ],
)
def test_the_roadmap_links_exactly_the_seps_with_each_status(section, statuses):
    text = ROADMAP.read_text()
    accepted = text.index("\n## Accepted enhancement proposals\n")
    drafts = text.index("\n## Draft enhancement proposals\n")
    assert accepted < drafts
    start = text.index(f"\n## {section}\n")
    end = text.find("\n## ", start + 1)
    linked = set(ROADMAP_LINK.findall(text[start:end]))
    expected = {
        relative(page) for page in SEP_FILES if header(page)["Status"] in statuses
    }
    assert linked == expected


@pytest.mark.parametrize(
    "page",
    [*SEP_FILES, SEPS / "index.md", SEPS / "template.md", ARCHIVE / "index.md"],
    ids=[*IDS, "index.md", "template.md", "archive/index.md"],
)
def test_a_sep_page_refers_to_no_local_files(page):
    assert LOCAL_REFERENCE.findall(page.read_text()) == []


def test_the_template_carries_every_header_field_and_part():
    template = SEPS / "template.md"
    text = template.read_text()
    assert [field for field in FIELDS if field not in header(template)] == []
    assert [part for part in PARTS if part not in text] == []
    assert header(template)["Status"] == "Draft"
    assert header(template)["Review status"] == "Not reviewed"
