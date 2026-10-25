# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin the ``native`` extra to this release's own ``solprop-native``.

solprop and solprop-native are released together from one repository at one version
(SEP 29), so ``solprop[native]`` requires exactly that version. The version is the
repository's (hatch-vcs), known only when the metadata is built, so the extra is
written here rather than in ``pyproject.toml``.
"""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class NativeExtraHook(MetadataHookInterface):
    """Add the ``native`` and ``mlx`` extras to the optional dependencies."""

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
        # The Apple-GPU backend (SEP 33), on macOS on Apple silicon only; the pin is
        # raised only with the GPU lane and the arithmetic probes run (Q9).
        apple_silicon = 'sys_platform == "darwin" and platform_machine == "arm64"'
        extras["mlx"] = [f"mlx>=0.32.4,<0.33; {apple_silicon}"]
