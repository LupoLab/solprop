# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Provide the argument parser and implementation of ``solprop julia``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from solprop.backends.julia import setup, status
from solprop.backends.julia._setup import depot_directory

__all__ = ["add_arguments", "describe", "run"]


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


def describe() -> str:
    """Describe every published setup and its current compatibility.

    Returns
    -------
    str
        Human-readable status, including why an installation cannot run.
    """
    lines = [f"Julia depot: {depot_directory()}"]
    found = status()
    if not found:
        lines.append("No published Julia setup. Run solprop julia setup.")
    for one in found:
        lines.append(
            f"{one.setup_id}: {'available' if one.compatible else 'unavailable'}"
        )
        if one.record is not None:
            record = one.record
            qualified = "pinned" if one.tested_runtime else "untested patch release"
            lines.extend(
                (
                    f"  Julia {record.julia_version} ({qualified}); "
                    f"bridge {record.bridge_version}",
                    f"  engine {record.engine_digest}",
                    f"  devices {', '.join(name for name, _ in record.projects)}; "
                    f"CPU target {record.cpu_target} (prepared on {record.cpu_name})",
                    f"  CUDA runtime {record.cuda_runtime or 'system/default'}; "
                    f"local {record.cuda_local}",
                    f"  depot {one.depot_bytes} bytes",
                )
            )
        if one.reason:
            lines.append(f"  {one.reason}")
        lines.extend(f"  {name} {version}" for name, version in one.package_versions)
        if one.record is not None:
            lines.append("  Julia checks and recompiles incompatible caches at load.")
        lines.extend(
            f"  {device}: {reason or 'prerequisites present'}"
            for device, reason in one.prerequisites
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> int:
    """Run the chosen Julia management command.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments produced by :func:`add_arguments`.

    Returns
    -------
    int
        Zero after success. Setup failures propagate as ``BackendError``.
    """
    if args.julia_action == "status":
        sys.stdout.write(describe())
    else:
        record = setup(
            devices=args.device,
            julia=args.julia,
            cuda_runtime=args.cuda_runtime,
            cuda_local=args.cuda_local,
            cpu_target=args.cpu_target,
        )
        sys.stdout.write(
            f"Prepared Julia setup {record.setup_id} "
            f"in {record.elapsed_seconds:.2f} s.\n"
        )
        sys.stdout.write(describe())
    return 0
