# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin ``solprop`` to this release's own version.

solprop and solprop-gui are released together from one repository at one version, and
solprop's ``gui`` extra requires exactly this package's version; the reverse pin keeps a
``pip install solprop-gui`` from pairing the front end with another solprop. The version
is the repository's (hatch-vcs), known only when the metadata is built, so the
dependencies are written here rather than in ``pyproject.toml``.
"""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


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
