# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The user guide's chapter on this window, read in a window of its own.

The pages are the manual's own (:mod:`solprop_gui.help.manual`), carried with the
front end, so they read offline. A link within the chapter turns the page; a link to
the rest of the manual, or to the web, opens in the browser. Screenshots are drawn at
the width of the page, never wider than they were taken.
"""

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import (
    QDesktopServices,
    QImageReader,
    QResizeEvent,
    QTextCursor,
    QTextImageFormat,
)
from PySide6.QtWidgets import (
    QHBoxLayout,
    QListWidget,
    QTextBrowser,
    QWidget,
)

from solprop_gui.help.manual import chapter_folder, page_title, pages, to_markdown
from solprop_gui.ui.identity import LINKS, app_icon
from solprop_gui.ui.theme import STYLE

__all__ = ["HelpWindow"]

_MARGIN = 80
"""The space a screenshot leaves at the page's sides, pixels."""


class _Page(QTextBrowser):
    """A page that fits its pictures to its width.

    Attributes
    ----------
    folder : pathlib.Path or None
        Where the pictures' paths start.
    """

    folder: Path | None = None

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802  # pyright: ignore[reportIncompatibleMethodOverride]
        """Fit the pictures again as the page changes width.

        Parameters
        ----------
        event : PySide6.QtGui.QResizeEvent
            The resize.
        """
        super().resizeEvent(event)
        self.fit_pictures()

    def fit_pictures(self) -> None:
        """Draw each picture at the page's width, or its own if narrower."""
        width = max(self.viewport().width() - _MARGIN, 200)
        found: list[tuple[int, int, QTextImageFormat]] = []
        block = self.document().begin()
        while block.isValid():
            fragments = block.begin()
            while not fragments.atEnd():
                fragment = fragments.fragment()
                if fragment.isValid() and fragment.charFormat().isImageFormat():
                    found.append(
                        (
                            fragment.position(),
                            fragment.length(),
                            fragment.charFormat().toImageFormat(),
                        )
                    )
                fragments += 1
            block = block.next()
        cursor = QTextCursor(self.document())
        for position, length, image in found:
            own = (
                QImageReader(str(self.folder / image.name())).size().width()
                if self.folder is not None
                else 0
            )
            image.setWidth(float(min(width, own) if own > 0 else width))
            cursor.setPosition(position)
            cursor.setPosition(position + length, QTextCursor.MoveMode.KeepAnchor)
            cursor.setCharFormat(image)


class HelpWindow(QWidget):
    """The guide's pages, listed on the left and read on the right.

    Parameters
    ----------
    folder : pathlib.Path or None, optional
        The chapter; ``None`` finds the one carried with the window.
    parent : PySide6.QtWidgets.QWidget or None, optional
        The owner.

    Attributes
    ----------
    names : list of str
        The pages, in the chapter's order.
    contents : PySide6.QtWidgets.QListWidget
        Their titles.
    page : PySide6.QtWidgets.QTextBrowser
        The page shown.
    current : str or None
        Its name.
    """

    def __init__(
        self, folder: Path | None = None, parent: QWidget | None = None
    ) -> None:
        """List the pages and show the first."""
        super().__init__(parent)
        self.setWindowTitle("solprop: user guide")
        self.setWindowIcon(app_icon())
        self.setObjectName("page")
        self.setStyleSheet(STYLE)
        self.folder = folder or chapter_folder()
        self.names = pages(self.folder) if self.folder is not None else []
        self.current: str | None = None
        self.contents = QListWidget()
        self.contents.setObjectName("help-contents")
        self.contents.setFixedWidth(220)
        self.contents.setSpacing(3)
        self.page = _Page()
        self.page.folder = self.folder
        self.page.setObjectName("help-page")
        self.page.setOpenLinks(False)
        self.page.document().setDocumentMargin(28)
        self.page.anchorClicked.connect(self.follow)
        for name in self.names:
            title = page_title(self._read(name)) or name
            self.contents.addItem(title)
        self.contents.currentRowChanged.connect(
            lambda row: self.show_page(self.names[row]) if row >= 0 else None
        )
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        row.addWidget(self.contents)
        row.addWidget(self.page, 1)
        self.resize(1080, 760)
        if self.names:
            self.contents.setCurrentRow(0)
        else:
            self.page.setMarkdown(
                "# The guide is not installed\n\nIt is online: "
                f"[the desktop front end]({LINKS['guide']})."
            )

    def _read(self, name: str) -> str:
        """Return a page's text.

        Parameters
        ----------
        name : str
            The page.

        Returns
        -------
        str
            Its MyST.
        """
        assert self.folder is not None  # pages are listed only with a folder
        return (self.folder / f"{name}.md").read_text(encoding="utf-8")

    def show_page(self, name: str, section: str = "") -> None:
        """Show a page, at a section.

        Parameters
        ----------
        name : str
            The page.
        section : str, optional
            The heading's anchor to scroll to.
        """
        if name not in self.names:
            return
        if name != self.current:
            self.current = name
            document = self.page.document()
            document.setBaseUrl(QUrl.fromLocalFile(f"{self.folder}/"))
            self.page.setMarkdown(to_markdown(self._read(name), name))
            self.page.fit_pictures()
            self.contents.blockSignals(True)
            self.contents.setCurrentRow(self.names.index(name))
            self.contents.blockSignals(False)
        if section:
            self.page.scrollToAnchor(section)
        else:
            self.page.verticalScrollBar().setValue(0)

    def follow(self, url: QUrl) -> None:
        """Follow a link: a page of the chapter here, anything else in the browser.

        Parameters
        ----------
        url : PySide6.QtCore.QUrl
            The link.
        """
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        target = url.toString()
        path, _, section = target.partition("#")
        name = Path(path).stem if path else (self.current or "")
        self.show_page(name, section)
