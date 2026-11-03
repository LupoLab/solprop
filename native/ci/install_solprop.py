# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Install solprop for a wheel's test, leaving out the wheel it would then require.

solprop requires solprop-native at its own version (SEP 43), and while a wheel is being
tested, that wheel is on no index: cibuildwheel installs it next. So this installs
solprop with no dependencies, then its dependencies but for that one::

    python native/ci/install_solprop.py path/to/solprop
"""

import importlib.metadata
import subprocess
import sys


def main() -> None:
    """Install the project named on the command line."""
    pip = [sys.executable, "-m", "pip", "install"]
    subprocess.run([*pip, "--no-deps", sys.argv[1]], check=True)
    requirements = [
        requirement
        for requirement in importlib.metadata.requires("solprop") or []
        if not requirement.startswith("solprop-native")
        and "extra ==" not in requirement
    ]
    subprocess.run([*pip, *requirements], check=True)


if __name__ == "__main__":
    main()
