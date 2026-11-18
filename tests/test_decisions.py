# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep the decision log's amendments visible from both records.

``docs/dev/decisions.md`` says that where a later record changes part of an earlier
one, both say so. When a record's status says it amends or supersedes another, the
earlier record's status, or its ``Now:`` line, must name the later one, so a reader of
the earlier record learns it no longer stands alone.
"""

import re
from pathlib import Path

DECISIONS = Path(__file__).resolve().parents[1] / "docs" / "dev" / "decisions.md"
RECORD = re.compile(r"^## (ADR-\d{3}) — ", re.MULTILINE)
# The later record's own claim: "amends ADR-033", "supersedes the interval cap of
# ADR-004 ...", up to the clause's end.
CLAIM = re.compile(r"\b(?:amends|supersedes)\b([^;]*)")


def _field(body: str, name: str) -> str:
    """Return a record's bullet `name` as one line, or an empty string."""
    found = re.search(
        rf"^- \*\*{name}:\*\*(.*?)(?=^- \*\*|\Z)", body, flags=re.MULTILINE | re.DOTALL
    )
    return " ".join(found.group(1).split()) if found else ""


def _records() -> dict[str, str]:
    """Return each record's body by its number."""
    parts = RECORD.split(DECISIONS.read_text(encoding="utf-8"))
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def _claims(records: dict[str, str]) -> list[tuple[str, str]]:
    """Return ``(later, earlier)`` for every amendment a record's status claims."""
    found = []
    for later, body in records.items():
        for claim in CLAIM.finditer(_field(body, "Status")):
            clause = claim.group(1).split(". ")[0]
            found += [
                (later, earlier)
                for earlier in re.findall(r"ADR-\d{3}", clause)
                if earlier != later
            ]
    return found


def test_an_amended_record_names_the_record_that_amends_it():
    records = _records()
    claims = _claims(records)
    # The log holds more than twenty such claims; finding none would mean the parser,
    # not the log, had changed.
    assert len(claims) > 20, claims
    unknown = sorted({earlier for _, earlier in claims} - records.keys())
    assert not unknown, f"amendments of records the log does not hold: {unknown}"
    missing = [
        f"{later} amends or supersedes {earlier}, whose status does not name it"
        for later, earlier in claims
        if later
        not in _field(records[earlier], "Status") + _field(records[earlier], "Now")
    ]
    assert not missing, "\n".join(missing)
