# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CI change filter that decides whether the device compilation lanes run."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/ci.yml"
LAST = '          echo "device=$device" | tee -a "$GITHUB_OUTPUT"'

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the CI change filter runs in Linux Bash"
)


def _device(directory: Path, changed: str | None, *, pipefail: bool) -> str:
    """Run the workflow's device filter on a change list; return what it writes."""
    workflow = WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("          device=true\n")
    end = workflow.index(LAST, start) + len(LAST)
    if changed is not None:
        (directory / "changed.txt").write_text(changed, encoding="utf-8")
    output = directory / "github-output"
    shell = ["bash", "-e", *(("-o", "pipefail") if pipefail else ())]
    subprocess.run(
        [*shell, "-c", textwrap.dedent(workflow[start:end])],
        cwd=directory,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        check=True,
        capture_output=True,
    )
    return output.read_text(encoding="utf-8")


@pytest.mark.parametrize("pipefail", [False, True])
def test_device_filter_selects_native_change_first_in_large_list(tmp_path, pipefail):
    # The match comes first and the list is long enough that a reader stopping at it
    # leaves the upstream grep writing into a closed pipe (SIGPIPE).
    changed = "native/src/core.cpp\n" + "".join(
        f"src/other/{index:08d}.py\n" for index in range(100_000)
    )
    assert _device(tmp_path, changed, pipefail=pipefail) == "device=true\n"


@pytest.mark.parametrize("pipefail", [False, True])
@pytest.mark.parametrize(
    "changed,expected",
    [
        ("native/src/core.cpp\n", "true"),
        ("docs/index.md\n.github/workflows/ci.yml\n", "true"),
        (None, "true"),
        ("native/README.md\ndocs/dev/testing.md\n", "false"),
        ("src/solprop/grid.py\ngui/app.py\n", "false"),
    ],
)
def test_device_filter_classifies_change_lists(tmp_path, changed, expected, pipefail):
    # None is a push whose base is unknown: no list to read, so a device build may
    # have changed.
    assert _device(tmp_path, changed, pipefail=pipefail) == f"device={expected}\n"
