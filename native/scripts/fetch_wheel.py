# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Install the wheel CI built for this checkout, for a contributor without a compiler.

``poe fetch-wheel`` finds the newest ``native-wheel-*`` artifact whose commit is in
this checkout's history and installs it without its dependencies. It needs the ``gh``
command, logged in. Run solprop with ``uv run --no-sync`` afterwards, because a sync
removes the wheel.
"""

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
}


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


def _native_sources_changed_since(sha: str) -> bool:
    command = [
        "git",
        "diff",
        "--quiet",
        sha,
        "HEAD",
        "--",
        "native/src",
        "native/cmake",
        "native/CMakeLists.txt",
    ]
    return subprocess.run(command, check=False).returncode != 0


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
        print(f"no {name} artifact from a commit in this history", file=sys.stderr)
        return 1
    sha = artifact["workflow_run"]["head_sha"]
    print(f"{name} from {sha[:10]}, built {artifact['created_at']}")
    if _native_sources_changed_since(sha):
        print(
            "warning: native/ has changed since that wheel was built", file=sys.stderr
        )
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
    print("run solprop with `uv run --no-sync`: a sync removes the wheel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
