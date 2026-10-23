# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Execute the code of the user guide, so that every snippet there runs as written.

Each page under ``docs/guide/`` is run as one script: its ``python`` fenced blocks, in
order, in one namespace, so a snippet may build on the one before it on the same page
and on nothing else. A block fenced as anything other than ``python`` (``sh``,
``text``, ``{code-block} python``) is illustration and is not run. The snippets are
written to finish in seconds each -- ten steps or a millimetre -- but a page is a few
of them, about 35 s for the guide, which is more than the medium tier's whole budget
(``docs/dev/testing.md`` section 2): running them is the slow lane's, and the default
lane compiles every page, which catches a syntax error or a page with no code.

Warnings are not what these tests are about: a snippet shows the ordinary use of an
entry point, and several of them (a gas index asked for beyond its fitted range, a
coherent wall near its quasi-static limit) warn by design, as the guide says. They
are ignored here rather than turned into errors.
"""

import re
import warnings
from pathlib import Path

import matplotlib as mpl
import pytest

DOCS = Path(__file__).resolve().parents[1] / "docs"
GUIDE = DOCS / "guide"
FENCE = re.compile(r"^```python\n(.*?)^```", re.DOTALL | re.MULTILINE)


def guide_code(page: Path) -> str:
    """Return the page's ``python`` blocks, joined in order."""
    return "\n\n".join(FENCE.findall(page.read_text()))


PAGES = sorted(
    page
    for page in (*GUIDE.rglob("*.md"), DOCS / "introduction" / "installation.md")
    if guide_code(page)
)
PAGE_IDS = [str(page.relative_to(DOCS)) for page in PAGES]


def test_the_guide_has_code_to_run():
    assert len(PAGES) >= 4


@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_every_page_of_code_compiles(page):
    assert compile(guide_code(page), str(page), "exec") is not None


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES, ids=PAGE_IDS)
def test_every_snippet_on_the_page_runs(page, tmp_path, monkeypatch):
    mpl.use("Agg")
    monkeypatch.chdir(tmp_path)  # a snippet that writes a file writes it here
    code = compile(guide_code(page), str(page), "exec")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(code, {"__name__": "__guide__"})
