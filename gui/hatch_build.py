# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin ``solprop`` to this release's own version, and carry the user guide's chapter.

solprop and solprop-gui are released together from one repository at one version, and
solprop's ``gui`` extra requires exactly this package's version; the reverse pin keeps a
``pip install solprop-gui`` from pairing the front end with another solprop. The version
is the repository's (hatch-vcs), known only when the metadata is built, so the
dependencies are written here rather than in ``pyproject.toml``.

The window's help shows the manual's chapter on it (``docs/guide/front-end/``), which
lies outside this project. The build hook copies its pages and screenshots into the
wheel as ``solprop_gui/help/guide/``, and into the sdist as ``guide/``, from which a
wheel built from the sdist takes them.
"""

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface

GUIDE_SUFFIXES = (".md", ".png")
"""What of the chapter is carried: its pages and its screenshots, not its script."""


class SolpropPinHook(MetadataHookInterface):
    """Write the runtime dependencies, with ``solprop`` at this version."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict[str, Any]) -> None:
        """Write the dependencies into the project's core metadata.

        Parameters
        ----------
        metadata : dict
            The project's core metadata, with its version already resolved.
        """
        metadata["dependencies"] = [
            f"solprop=={metadata['version']}",
            # 6.10.1: the first release whose wheels admit CPython 3.14 (6.10.0 declares
            # <3.14). Essentials holds the Qt modules the front end uses, about a third
            # of the full PySide6.
            "PySide6-Essentials>=6.10.1",
            # 0.14: the first release made after Qt 6.10 and CPython 3.14; 0.13.7
            # predates both and has not been tried. Lower it only with a lane that
            # tests it.
            "pyqtgraph>=0.14",
        ]


class GuideHook(BuildHookInterface):  # pyright: ignore[reportMissingTypeArgument]
    """Carry the user guide's chapter on the window with the package."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Add the chapter's files to the build.

        Parameters
        ----------
        version : str
            The build's version, unused.
        build_data : dict
            The build's data, whose ``force_include`` maps a file to its place.

        Raises
        ------
        FileNotFoundError
            If neither the repository's chapter nor an sdist's copy is there.
        """
        del version
        root = Path(self.root)
        chapter = root.parent / "docs" / "guide" / "front-end"
        if not (chapter / "index.md").is_file():
            chapter = root / "guide"  # an sdist's copy
        if not (chapter / "index.md").is_file():
            msg = f"the user guide's chapter is missing: {chapter}"
            raise FileNotFoundError(msg)
        place = "solprop_gui/help/guide" if self.target_name == "wheel" else "guide"
        for path in sorted(chapter.rglob("*")):
            if path.is_file() and path.suffix in GUIDE_SUFFIXES:
                relative = path.relative_to(chapter).as_posix()
                build_data["force_include"][str(path)] = f"{place}/{relative}"
