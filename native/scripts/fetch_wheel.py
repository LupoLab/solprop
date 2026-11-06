# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Install the wheel CI built for this checkout, for a contributor without a compiler.

``poe fetch-wheel`` finds the newest ``native-wheel-*`` artifact whose commit is in
this checkout's history, and installs it without its dependencies if it fits this tree:
nothing in ``native/`` or in the solprop modules ``solprop_native`` imports may differ
from that commit, committed or not (SEP 43 section 6.7). It needs the ``gh`` command,
logged in. The wheel keeps its own version and build record, so ``uv pip check``
reports the pin as unmet, deliberately. Run solprop with ``uv run --no-sync``
afterwards, because a sync builds native in its place.
"""

import ast
import json
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

REPOSITORY = "LupoLab/solprop"
WHEEL_IDS = {
    ("Linux", "x86_64"): "linux-x86_64",
    ("Darwin", "arm64"): "macos-arm64",
    ("Windows", "AMD64"): "windows-amd64",
    ("Windows", "ARM64"): "windows-arm64",
}


ALTERNATIVES = (
    "Build native with `uv sync`, or set up without it with "
    "`uv run --no-project tools/setup_light.py`."
)


def newest_in_history(
    artifacts: Iterable[dict[str, Any]], in_history: Callable[[str], bool]
) -> dict[str, Any] | None:
    """Return the newest unexpired artifact whose commit passes ``in_history``."""
    live = (artifact for artifact in artifacts if not artifact["expired"])
    newest_first = sorted(
        live, key=lambda artifact: artifact["created_at"], reverse=True
    )
    return next(
        (a for a in newest_first if in_history(a["workflow_run"]["head_sha"])),
        None,
    )


def _is_ancestor(sha: str) -> bool:
    command = ["git", "merge-base", "--is-ancestor", sha, "HEAD"]
    return subprocess.run(command, capture_output=True, check=False).returncode == 0


def imported_modules(root: Path) -> list[str]:
    """Return the paths of the solprop modules that ``solprop_native`` imports.

    Parameters
    ----------
    root : pathlib.Path
        The repository root.

    Returns
    -------
    list of str
        Paths relative to ``root``, sorted, derived from the imports so that the list
        can't go stale.
    """
    names: set[str] = set()
    for source in (root / "native" / "src" / "solprop_native").glob("*.py"):
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
            elif isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
    paths: set[str] = set()
    for name in names:
        if name == "solprop" or not name.startswith("solprop."):
            continue
        path = Path("src", *name.split("."))
        module = path.with_suffix(".py")
        paths.add((module if (root / module).exists() else path).as_posix())
    return sorted(paths)


def difference_from(sha: str, root: Path) -> str:
    """Return what differs from ``sha`` where a wheel depends on the tree, if anything.

    Parameters
    ----------
    sha : str
        The commit the wheel was built from.
    root : pathlib.Path
        The repository root.

    Returns
    -------
    str
        The files that differ, including uncommitted and untracked ones; empty if none.
    """
    paths = ["native", *imported_modules(root)]
    changed = subprocess.run(
        ["git", "diff", "--name-only", sha, "--", *paths],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *paths],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    ).stdout
    return changed + untracked


def _artifacts(name: str) -> list[dict[str, Any]]:
    url = f"repos/{REPOSITORY}/actions/artifacts?name={name}&per_page=100"
    output = subprocess.run(
        ["gh", "api", "--paginate", "--slurp", url],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [artifact for page in json.loads(output) for artifact in page["artifacts"]]


def main() -> int:
    """Find, download and install the wheel; return the process exit code."""
    wheel_id = WHEEL_IDS.get((platform.system(), platform.machine()))
    if wheel_id is None:
        print(
            f"CI builds no wheel for {platform.system()} {platform.machine()}",
            file=sys.stderr,
        )
        return 1
    name = f"native-wheel-{wheel_id}"
    artifact = newest_in_history(_artifacts(name), _is_ancestor)
    if artifact is None:
        print(
            f"no unexpired {name} artifact from a commit in this history. "
            f"{ALTERNATIVES}",
            file=sys.stderr,
        )
        return 1
    sha = artifact["workflow_run"]["head_sha"]
    print(f"{name} from {sha[:10]}, built {artifact['created_at']}")
    difference = difference_from(sha, Path.cwd())
    if difference:
        print(
            f"refusing that wheel: these files differ from {sha[:10]}, and a wheel "
            f"depends on them:\n{difference}{ALTERNATIVES}",
            file=sys.stderr,
        )
        return 1
    with tempfile.TemporaryDirectory() as directory:
        run_id = str(artifact["workflow_run"]["id"])
        download = [
            "gh",
            "run",
            "download",
            run_id,
            "-R",
            REPOSITORY,
            "-n",
            name,
            "-D",
            directory,
        ]
        subprocess.run(download, check=True)
        wheels = [str(wheel) for wheel in Path(directory).glob("*.whl")]
        subprocess.run(["uv", "pip", "install", "--no-deps", *wheels], check=True)
    if platform.system() == "Linux":
        print(
            "this wheel links the system FFTW: install libfftw3-dev "
            "or your distribution's FFTW"
        )
    print("run solprop with `uv run --no-sync`: a sync builds native in its place")
    return 0


if __name__ == "__main__":
    sys.exit(main())
