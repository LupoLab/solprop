# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin ``solprop-native`` to this release's own version in the built metadata.

solprop and solprop-native are released together from one repository at one version
(SEP 29), so solprop requires exactly that version (SEP 43). The version is the
repository's (hatch-vcs), known only when the metadata is built, so ``pyproject.toml``
lists the bare name, which uv resolves to the workspace member, and the pin is written
here. This rewrites a statically listed field, which hatchling allows.
"""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class NativePinHook(MetadataHookInterface):
    """Pin the native dependency and add the ``mlx`` extra."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict[str, Any]) -> None:
        """Write the pin and the extra into the project's core metadata.

        Parameters
        ----------
        metadata : dict
            The project's core metadata, with its version already resolved.
        """
        pinned = f"solprop-native=={metadata['version']}"
        metadata["dependencies"] = [
            pinned if requirement == "solprop-native" else requirement
            for requirement in metadata["dependencies"]
        ]
        extras = metadata.setdefault("optional-dependencies", {})
        # The Apple-GPU backend (SEP 33), on macOS on Apple silicon only; the pin is
        # raised only with the GPU lane and the arithmetic probes run (Q9).
        apple_silicon = 'sys_platform == "darwin" and platform_machine == "arm64"'
        extras["mlx"] = [f"mlx>=0.32.4,<0.33; {apple_silicon}"]
