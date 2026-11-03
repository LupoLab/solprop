# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin ``solprop-native`` and the ``gui`` extra to this release's own distributions.

solprop, solprop-native and solprop-gui are released together from one repository at
one version (SEP 29), so solprop requires exactly that native version (SEP 43), and
``solprop[gui]`` that front end. The version is the repository's (hatch-vcs), known only
when the metadata is built, so ``pyproject.toml`` lists the bare native name, which uv
resolves to the workspace member, and the pins are written here. This rewrites a
statically listed field, which hatchling allows.
"""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class NativePinHook(MetadataHookInterface):
    """Pin the native dependency and add the ``gui`` and ``mlx`` extras."""

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
        # The desktop front end, from gui/ at this same version.
        extras["gui"] = [f"solprop-gui=={metadata['version']}"]
        # The Apple-GPU backend (SEP 33), on macOS on Apple silicon only; the pin is
        # raised only with the GPU lane and the arithmetic probes run (Q9).
        apple_silicon = 'sys_platform == "darwin" and platform_machine == "arm64"'
        extras["mlx"] = [f"mlx>=0.32.4,<0.33; {apple_silicon}"]
