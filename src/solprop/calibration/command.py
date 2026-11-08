# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""``solprop calibrate``: measure, show, export, import or forget a profile (SEP 46).

The command line's parser (``solprop.cli``) hands this module its arguments. Nothing
here loads the native extension.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from solprop import calibration
from solprop.calibration import _native

__all__ = ["add_arguments", "describe", "run"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``solprop calibrate``'s options to `parser`.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The subcommand's parser.
    """
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--quick",
        dest="level",
        action="store_const",
        const="quick",
        help="about 20 s: the waiting policy and the team at reduced depth",
    )
    action.add_argument(
        "--thorough",
        dest="level",
        action="store_const",
        const="thorough",
        help="about five minutes: more rounds and longer windows",
    )
    action.add_argument(
        "--show",
        action="store_true",
        help="show the profile this allocation would use, and its evidence",
    )
    action.add_argument(
        "--export",
        metavar="FILE",
        type=Path,
        help="write this machine and build's profile to FILE",
    )
    action.add_argument(
        "--import",
        dest="import_",
        metavar="FILE",
        type=Path,
        help="publish an exported profile in your store",
    )
    action.add_argument(
        "--clear",
        action="store_true",
        help="forget this machine and build's profiles",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="SECONDS",
        help="the calibration's deadline, the level's when omitted",
    )
    parser.set_defaults(level="standard")


def _value(value: object) -> str:
    """Format one setting for the description.

    Parameters
    ----------
    value : object
        A setting's value.

    Returns
    -------
    str
        ``yes`` or ``no`` for a flag, else the value as text.
    """
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _summary(record: Mapping[str, object]) -> str:
    """Describe an allocation class in a line.

    Parameters
    ----------
    record : Mapping[str, object]
        An allocation class's record.

    Returns
    -------
    str
        Its cores, kinds, spans, SMT and quota.
    """
    cores = record.get("cores")
    kinds = record.get("kinds")
    parts = [f"{cores} cores"]
    if isinstance(kinds, list) and len(kinds) > 1:  # pyright: ignore[reportUnknownArgumentType]
        parts.append("of kinds " + " + ".join(str(kind) for kind in kinds))  # pyright: ignore[reportUnknownVariableType]
    for name, label in (
        ("sockets", "sockets"),
        ("numa_nodes", "NUMA nodes"),
        ("l3_groups", "L3 groups"),
    ):
        if record.get(name, 1) != 1:
            parts.append(f"{record.get(name)} {label}")
    if record.get("smt"):
        parts.append("SMT")
    if record.get("quota") is not None:
        parts.append(f"quota {record.get('quota')}")
    return ", ".join(parts)


def _evidence(evidence: Mapping[str, object]) -> list[str]:
    """Summarise a row's evidence: why, the noise, each verdict, the validation.

    Parameters
    ----------
    evidence : Mapping[str, object]
        A row's evidence.

    Returns
    -------
    list of str
        One line each.
    """
    lines: list[str] = []
    if "reason" in evidence:
        lines.append(f"decided:      {evidence['reason']}")
    if "noise" in evidence:
        lines.append(f"noise:        {_value(evidence['noise'])} (A/A)")
    verdicts = evidence.get("verdicts")
    if isinstance(verdicts, dict):
        for name, verdict in cast(dict[str, dict[str, object]], verdicts).items():
            score = verdict.get("score")
            shown = f"{score:.3f}" if isinstance(score, float) else "-"
            lines.append(f"candidate:    {name} {shown} ({verdict.get('reason')})")
    validation = evidence.get("validation")
    if isinstance(validation, dict):
        lines.append(
            "validation:   "
            + str(cast(dict[str, object], validation).get("reason", ""))
        )
    return lines


def describe(environ: Mapping[str, str] | None = None) -> str:
    """Return ``solprop calibrate --show``'s text.

    Parameters
    ----------
    environ : Mapping[str, str] or None, optional
        The environment; the process's when ``None``.

    Returns
    -------
    str
        The store, the machine, the build, the allocation, and the profile's
        settings beside the built-in table's, or why none applies.
    """
    found = calibration.choice(environ)
    lines = [f"store:        {calibration.store_directory(environ)}"]
    for root in _native.store.search_path(os.environ if environ is None else environ):
        lines.append(f"read-only:    {root}")
    lines.append(f"mode:         SOLPROP_CALIBRATION={found.mode}")
    if found.fingerprint is None or found.allocation is None:
        lines.append("profile:      none (the built-in table applies)")
        return "\n".join(lines) + "\n"
    machine = found.fingerprint.machine
    build = found.fingerprint.build
    record = build.get("record")
    details = (
        [
            str(record.get(key))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            for key in ("compiler_id", "compiler_version", "openmp_libraries")
        ]
        if isinstance(record, dict)
        else ["no build record"]
    )
    lines += [
        f"machine:      {machine.get('model')} ({machine.get('system')} "
        f"{machine.get('architecture')})",
        f"build:        solprop-native {build.get('version')}, {' '.join(details)}",
        f"fingerprint:  {found.fingerprint.digest}",
        f"allocation:   {_summary(found.allocation.record())}",
    ]
    if found.generation is None or found.found is None:
        lines.append(f"profile:      none: {found.note} (the built-in table applies)")
        return "\n".join(lines) + "\n"
    row = found.found.row
    scope = (
        "this allocation"
        if found.found.exact
        else f"nearest smaller allocation ({_summary(row.allocation)}): workspace "
        "settings only"
    )
    lines += [
        f"profile:      {row.level}, generation {found.generation.name}, "
        f"{found.generation.created}",
        f"              measured on {scope}",
        f"file:         {found.generation.source}",
    ]
    if row.process:
        lines.append(
            "process:      "
            + ", ".join(
                f"{name} {_value(value)}" for name, value in row.process.items()
            )
        )
    table = _native.execution.builtin(platform.system()).native()
    for name, value in row.workspace.items():
        lines.append(
            f"workspace:    {name} {_value(value)} (built-in {_value(table.get(name))})"
        )
    if row.fitted_under:
        fitted = ", ".join(f"{k} {_value(v)}" for k, v in row.fitted_under.items())
        lines.append(f"fitted under: {fitted}")
    lines += _evidence(row.evidence)
    return "\n".join(lines) + "\n"


def run(arguments: argparse.Namespace) -> int:
    """Carry out ``solprop calibrate``.

    Parameters
    ----------
    arguments : argparse.Namespace
        The parsed options of :func:`add_arguments`.

    Returns
    -------
    int
        The exit status: zero on success.

    Raises
    ------
    CalibrationError
        When there is nothing to export, a file cannot be imported, or no action
        was asked for.
    """
    if arguments.show:
        sys.stdout.write(describe())
        return 0
    if arguments.export is not None:
        source = calibration.export_profile(arguments.export)
        sys.stdout.write(f"exported {source} to {arguments.export}\n")
        return 0
    if arguments.import_ is not None:
        applies = calibration.import_profile(arguments.import_)
        note = (
            "it applies to this machine and build"
            if applies
            else "it is for another machine or build, and does not apply here"
        )
        sys.stdout.write(f"imported {arguments.import_}: {note}\n")
        return 0
    if arguments.clear:
        removed = calibration.clear()
        sys.stdout.write(
            f"removed {removed} file(s) from {calibration.store_directory()}\n"
        )
        return 0
    from solprop.calibration.controller import calibrate  # noqa: PLC0415 - heavy

    report = calibrate(
        arguments.level,
        arguments.budget,
        progress=lambda line: sys.stderr.write(f"solprop calibrate: {line}\n"),
    )
    sys.stdout.write(report.describe())
    return 0 if report.status == "published" else 1
