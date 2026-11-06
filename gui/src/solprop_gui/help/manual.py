# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The front end's chapter of the user guide, carried with the window.

The chapter is written once, in the manual (``docs/guide/front-end/``). The wheel
carries a copy of its pages and screenshots in ``solprop_gui/help/guide/``
(``gui/hatch_build.py``), so the window's help works offline; a checkout reads them
where they are. The pages are MyST, which Qt's Markdown does not know, so they are shown
through :func:`to_markdown`: its cross-references become plain code, its table of
contents goes, and a link to another page of the manual becomes the online address of
that page, while a link within the chapter stays one the help window follows.
"""

import posixpath
import re
from pathlib import Path
from typing import Final

from solprop_gui.help.guide import MANUAL, guide_url

__all__ = ["CHAPTER", "chapter_folder", "page_title", "pages", "to_markdown"]

CHAPTER: Final = "guide/front-end"
"""The chapter's place in the manual."""

_HERE: Final = Path(__file__).resolve().parent

_ROLE: Final = re.compile(r"\{py:\w+\}`(~?)([^`]+)`")
"""A cross-reference to the API, as ``{py:func}`~solprop.processing.get_power```."""

_FENCE: Final = re.compile(r"^```\{toctree\}.*?^```\n?", re.MULTILINE | re.DOTALL)
"""A table of contents."""

_LINK: Final = re.compile(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)")
"""A link that is not an image."""


def chapter_folder(
    packaged: Path = _HERE / "guide",
    checkout: Path = _HERE.parents[3] / "docs" / "guide" / "front-end",
) -> Path | None:
    """Return the folder that holds the chapter's pages.

    Parameters
    ----------
    packaged : pathlib.Path, optional
        Where a wheel puts them.
    checkout : pathlib.Path, optional
        Where a checkout of the repository has them.

    Returns
    -------
    pathlib.Path or None
        The first that has the chapter's index; ``None`` when neither does.
    """
    for folder in (packaged, checkout):
        if (folder / "index.md").is_file():
            return folder
    return None


def pages(folder: Path) -> list[str]:
    r"""Return the chapter's pages in the order its table of contents gives.

    Parameters
    ----------
    folder : pathlib.Path
        The chapter.

    Returns
    -------
    list of str
        ``index`` first, then each page the index lists, by name without ``.md``.

    Examples
    --------
    >>> import tempfile
    >>> folder = Path(tempfile.mkdtemp())
    >>> _ = (folder / "index.md").write_text(
    ...     "# Hi\n\n```{toctree}\n:hidden:\n\nfirst\nsecond\n```\n")
    >>> pages(folder)
    ['index', 'first', 'second']
    """
    found = ["index"]
    index = (folder / "index.md").read_text(encoding="utf-8")
    for block in _FENCE.findall(index):
        for line in block.splitlines()[1:]:
            name = line.strip()
            if name and not name.startswith((":", "```")) and name not in found:
                found.append(name)
    return found


def page_title(text: str) -> str:
    r"""Return a page's title: its first heading.

    Parameters
    ----------
    text : str
        The page.

    Returns
    -------
    str
        The heading, or nothing.

    Examples
    --------
    >>> page_title("Some words\n# A first simulation\n## More")
    'A first simulation'
    >>> page_title("no heading")
    ''
    """
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return ""


def _role(match: re.Match[str]) -> str:
    """Return a cross-reference as the code it names.

    Parameters
    ----------
    match : re.Match
        The reference.

    Returns
    -------
    str
        The name in backticks; its last part only where MyST would show only that.
    """
    short, target = match.group(1), match.group(2)
    return f"`{target.rsplit('.', 1)[-1] if short else target}`"


def _link(match: re.Match[str], page: str, manual: str) -> str:
    """Return a link as the help window can follow it.

    Parameters
    ----------
    match : re.Match
        The link.
    page : str
        The page it is on, by name without ``.md``.
    manual : str
        The online manual's root, ending in ``/``.

    Returns
    -------
    str
        The link: unchanged when it is on the web or within the chapter, otherwise
        to that page of the online manual.
    """
    text, target = match.group(1), match.group(2)
    if "://" in target or target.startswith(("#", "mailto:")):
        return match.group(0)
    path, _, section = target.partition("#")
    if "/" not in path:
        return match.group(0)
    folder = posixpath.join(CHAPTER, posixpath.dirname(page))
    resolved = posixpath.normpath(posixpath.join(folder, path))
    anchor = f"{resolved}#{section}" if section else resolved
    return f"[{text}]({guide_url(anchor, manual)})"


def to_markdown(text: str, page: str = "index", manual: str = MANUAL) -> str:
    """Return a page of the chapter as Markdown that Qt shows.

    Parameters
    ----------
    text : str
        The page's MyST.
    page : str, optional
        Its name, without ``.md``.
    manual : str, optional
        The online manual's root, ending in ``/``.

    Returns
    -------
    str
        The page, its cross-references as code, its table of contents gone and its
        links to the rest of the manual online.

    Examples
    --------
    >>> to_markdown("See {py:func}`~solprop.processing.fields.get_power`.")
    'See `get_power`.'
    >>> shown = to_markdown("[classes](classes.md), [pins](../setup/recipes.md#pins)")
    >>> shown.split(", ")[0]
    '[classes](classes.md)'
    >>> shown.split(", ")[1].removeprefix("[pins](" + MANUAL)
    'guide/setup/recipes.html#pins)'
    """
    text = _FENCE.sub("", text)
    text = _ROLE.sub(_role, text)
    return _LINK.sub(lambda match: _link(match, page, manual), text)
