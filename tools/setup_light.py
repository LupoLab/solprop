# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Set up a checkout with no native package and no compiler (SEP 43 section 6.7).

For Python and documentation contributors::

    uv run --no-project tools/setup_light.py

A plain ``uv sync`` builds solprop-native, which needs a C++ toolchain. This installs
everything else. It is a development workflow, not a way to run solprop in production:
``backend="auto"`` refuses here, and a change to ``native/`` still needs a native build.
"""

import subprocess
import sys

COMMAND = ("uv", "sync", "--locked", "--no-install-package", "solprop-native")

NEXT_STEPS = """\
The environment is in .venv, with no native package.

uv can't be told to leave it out for good: a plain `uv sync` or `uv run` would build it.
So activate the environment and run tools directly, or set UV_NO_SYNC=1 for `uv run`:

    export UV_NO_SYNC=1
    uv run poe lint
    uv run poe types
    uv run poe docs
    SOLPROP_TEST_BACKEND=numpy_ref uv run poe test

backend="auto" refuses here, so scripts name backend="numpy_ref". Tests take
SOLPROP_TEST_BACKEND=numpy_ref, and tests that need native skip. A change to native/
needs a native build and its qualification: run `uv sync` with a C++20 compiler.
"""


def main() -> int:
    """Install the environment and say how to continue; return the exit code."""
    status = subprocess.run(COMMAND, check=False).returncode
    if status == 0:
        print(NEXT_STEPS)
    return status


if __name__ == "__main__":
    sys.exit(main())
