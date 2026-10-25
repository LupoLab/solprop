# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collection of the package's doctests (``testpaths`` includes ``src``).

The ``mlx`` backend (SEP 33) imports MLX, which installs on macOS on Apple silicon
only; elsewhere its modules are not collected.
"""

import importlib.util

collect_ignore_glob = (
    [] if importlib.util.find_spec("mlx") else ["solprop/backends/mlx_gpu/*"]
)
