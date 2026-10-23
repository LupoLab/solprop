# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Register the independent modal-design oracles under ``tests/oracles/modal_design``.

The oracles are research prototypes written apart from the production solver: vector
and graded-index mode solvers and a condensed taper model. They import each other as
``modal_design.<module>``, so the directory is registered as that package by path
before a test imports one of them.
"""

import importlib.util
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent / "oracles" / "modal_design"


def register_modal_oracles() -> None:
    """Register ``tests/oracles/modal_design`` as the package ``modal_design``."""
    if "modal_design" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(
        "modal_design",
        PACKAGE / "__init__.py",
        submodule_search_locations=[str(PACKAGE)],
    )
    if spec is None or spec.loader is None:
        msg = f"cannot load the oracle package from {PACKAGE}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["modal_design"] = module
    spec.loader.exec_module(module)
