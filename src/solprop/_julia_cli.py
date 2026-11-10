# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Register Julia management arguments without importing an execution backend."""

import argparse
from pathlib import Path

__all__ = ["add_arguments"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add Julia setup and status subcommands to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser for ``solprop julia``.
    """
    actions = parser.add_subparsers(dest="julia_action", required=True)
    install = actions.add_parser(
        "setup", help="install and verify an optional Julia stack"
    )
    install.add_argument(
        "--device",
        action="append",
        choices=("cpu", "cuda", "rocm", "metal"),
        default=[],
    )
    install.add_argument("--julia", type=Path, help="use an installed Julia executable")
    install.add_argument(
        "--cpu-target", default="native", help="Julia target for compiled code"
    )
    install.add_argument(
        "--cuda-runtime", help="CUDA runtime release to prepare without a GPU"
    )
    install.add_argument(
        "--cuda-local", action="store_true", help="use the system CUDA toolkit"
    )
    actions.add_parser("status", help="inspect published setups without starting Julia")
