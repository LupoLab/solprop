# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Declare the native release dependency and the experimental Julia bridge.

solprop and solprop-native are released together from one repository at one version
(SEP 29), so ``solprop[native]`` requires exactly that version. The version is the
repository's (hatch-vcs), known only when the metadata is built, so the extra is
written here rather than in ``pyproject.toml``. The source-only Julia experiment
adds a JuliaCall pin matching its committed PythonCall environment.
"""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class NativeExtraHook(MetadataHookInterface):
    """Add the versioned native extra and the pinned experimental Julia extra."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict[str, Any]) -> None:
        """Write the pinned extra into the project's core metadata.

        Parameters
        ----------
        metadata : dict
            The project's core metadata, with its version already resolved.
        """
        extras = metadata.setdefault("optional-dependencies", {})
        extras["native"] = [f"solprop-native=={metadata['version']}"]
        # Matches the prototype's resolved PythonCall bridge in julia/Manifest.toml.
        extras["julia"] = ["juliacall==0.9.37"]
