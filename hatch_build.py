# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin sibling distributions and bundle the optional Julia engine.

solprop, solprop-native and solprop-gui are released together from one repository at
one version (SEP 29), so solprop requires exactly that native version (SEP 43), and
``solprop[gui]`` that front end. The version is the repository's (hatch-vcs), known only
when the metadata is built, so the runtime dependencies are written here. They are
dynamic rather than a pin over a static list: a wheel built from the sdist reads
dynamic fields from its metadata without rerunning the metadata hook. The separate
build hook includes the Julia package inputs and their engine digest in each wheel.
"""

import runpy
import tempfile
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from hatchling.metadata.plugin.interface import MetadataHookInterface


class NativePinHook(MetadataHookInterface):
    """Write the dependencies, with native pinned, and the optional extras."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict[str, Any]) -> None:
        """Write the dependencies and extras into the project's core metadata.

        Parameters
        ----------
        metadata : dict
            The project's core metadata, with its version already resolved.
        """
        # Each floor is the oldest release that has CPython 3.14 wheels for Linux, macOS
        # and Windows and provides the APIs solprop uses, unless a comment names a
        # stricter reason; the `oldest` CI job tests them (docs/dev/dependencies.md).
        metadata["dependencies"] = [
            # The backend solprop runs on (SEP 43), with no extra and no markers. uv
            # resolves it to the workspace member; a platform with no wheel gets the
            # installer's resolution error.
            f"solprop-native=={metadata['version']}",
            # 2.5: under 2.3 and 2.4 the free-space reference runs produce NaN (cause
            # not yet identified; docs/dev/dependencies.md).
            "numpy>=2.5",
            # optmat's floor: its stored fits reproduce bit for bit from SciPy 1.18.
            "scipy>=1.18",
            # 3.10.7: earlier releases call pyparsing APIs that current pyparsing
            # deprecates.
            "matplotlib>=3.10.7",
            "h5py>=3.15",
            # Every built-in material model (SEP 3): gases with their equation of state
            # and transport, solids, Raman responses and ionisation rates.
            "optmat[gas]>=0.9.1",
            # The native backend limits the other BLAS pools in its process through this
            # (solprop_native/_host_blas.py): no BLAS exposes a runtime thread count
            # except through the library walk this does.
            "threadpoolctl>=3.1",
            # Array annotations are evaluated at import time in every module (AGENTS.md,
            # Typing), so the annotation package is a runtime dependency; beartype (the
            # checker) is dev-only. 0.3.11: the oldest release the `oldest` CI job has
            # tested.
            "jaxtyping>=0.3.11",
            "tqdm>=4.67",
            # Writes recipes (SEP 41); 1.0, the first release, already writes every form
            # a recipe uses (nested tables, arrays of inline tables, quoted keys) back
            # through tomllib.
            "tomli-w>=1.0",
        ]
        extras = metadata.setdefault("optional-dependencies", {})
        # The desktop front end, from gui/ at this same version.
        extras["gui"] = [f"solprop-gui=={metadata['version']}"]
        # PythonCall in the prepared Julia environments must match exactly. The
        # base distribution never imports or installs this optional bridge.
        extras["julia"] = ["juliacall==0.9.37"]


class JuliaBundleHook(BuildHookInterface):
    """Ship the engine and its build-time digest without importing solprop."""

    PLUGIN_NAME = "custom"
    _temporary: tempfile.TemporaryDirectory[str] | None = None

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Add the Julia sources, environment inputs and exact digest to a wheel.

        Parameters
        ----------
        version : str
            Hatch's build version, including ``editable``.
        build_data : dict
            Hatch's mutable build configuration.
        """
        del version
        if self.target_name != "wheel":
            return
        root = Path(self.root)
        bundle = root / "julia"
        source = runpy.run_path(str(root / "src/solprop/backends/julia/_source.py"))
        digest = source["engine_digest"](bundle)
        self._temporary = tempfile.TemporaryDirectory(prefix="solprop-julia-build-")
        digest_file = Path(self._temporary.name) / "_engine_digest.txt"
        digest_file.write_text(digest + "\n", encoding="utf-8")
        target = "solprop/backends/julia"
        force_include = build_data.setdefault("force_include", {})
        force_include[str(digest_file)] = f"{target}/_engine_digest.txt"
        # An explicit source inventory excludes developer records, caches and
        # local evidence, including when building from a worktree or an sdist.
        for path in sorted(source["bundle_files"](bundle)):
            relative = path.relative_to(bundle).as_posix()
            force_include[str(path)] = f"{target}/_bundle/{relative}"

    def finalize(
        self, version: str, build_data: dict[str, Any], artifact_path: str
    ) -> None:
        """Remove only the temporary digest after Hatch has copied it.

        Parameters
        ----------
        version : str
            Hatch's build version.
        build_data : dict
            The completed build configuration.
        artifact_path : str
            The resulting wheel or source archive.
        """
        del version, build_data, artifact_path
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None
