# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Install solprop for a wheel's test, leaving out the wheel it would then require.

solprop requires solprop-native at its own version (SEP 43), and while a wheel is being
tested, that wheel is on no index: cibuildwheel installs it next. So this installs
solprop with no dependencies, then its dependencies but for that one::

    python native/ci/install_solprop.py path/to/solprop
"""

import subprocess
import sys
import tomllib
from pathlib import Path


def main() -> None:
    """Install the project named on the command line."""
    project = Path(sys.argv[1])
    metadata = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = [
        requirement
        for requirement in metadata["project"]["dependencies"]
        if requirement != "solprop-native"
    ]
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run([*pip, "--no-deps", str(project)], check=True)
    subprocess.run([*pip, *requirements], check=True)


if __name__ == "__main__":
    main()
