# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``solprop`` command line: run, check and export recipes (SEP 41, S1).

``solprop run recipe.toml -o out.h5`` builds and runs a recipe, ``solprop check``
reports what stops it building, ``solprop export --python`` prints the equivalent
script, and ``solprop scan`` is ``python -m solprop.scan``. The same commands run as
``python -m solprop``. The manual page is ``docs/guide/setup/recipes.md``.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from solprop.exceptions import SolpropError

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    """Return the command line's parser.

    Returns
    -------
    argparse.ArgumentParser
        The parser, one subcommand per action.
    """
    parser = argparse.ArgumentParser(
        prog="solprop",
        description="Run, check and export solprop recipes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="build and run a recipe")
    run.add_argument("recipe", type=Path, help="the recipe file")
    run.add_argument(
        "-o", "--output", type=Path, required=True, help="the HDF5 file to write"
    )
    run.add_argument(
        "--progress",
        choices=("auto", "bar", "log", "none"),
        default="auto",
        help="how the run reports its progress (default: auto)",
    )
    run.add_argument(
        "--resume", action="store_true", help="continue the output file if it exists"
    )
    check = commands.add_parser("check", help="report what stops a recipe building")
    check.add_argument("recipe", type=Path, help="the recipe file")
    export = commands.add_parser("export", help="print a recipe as another form")
    export.add_argument("recipe", type=Path, help="the recipe file")
    export.add_argument(
        "--python", action="store_true", required=True, help="print a Python script"
    )
    export.add_argument(
        "-o", "--output", default=None, help="the file the script's run writes"
    )
    commands.add_parser(
        "scan", help="run a scan script (python -m solprop.scan)", add_help=False
    )
    return parser


def _run(recipe_path: Path, output: Path, progress: str, *, resume: bool) -> int:
    """Build and run a recipe.

    Parameters
    ----------
    recipe_path : pathlib.Path
        The recipe file.
    output : pathlib.Path
        The HDF5 file.
    progress : str
        The progress choice.
    resume : bool
        Whether to continue the file.

    Returns
    -------
    int
        Zero.
    """
    from solprop.recipe import Recipe  # noqa: PLC0415 - imports the physics

    Recipe.load(recipe_path).run(output, progress=progress, resume=resume)
    return 0


def _check(recipe_path: Path) -> int:
    """Print what stops a recipe building, or makes it warn.

    Parameters
    ----------
    recipe_path : pathlib.Path
        The recipe file.

    Returns
    -------
    int
        Zero when the recipe builds without an error, one otherwise.
    """
    from solprop.recipe import Recipe  # noqa: PLC0415 - imports the physics

    issues = Recipe.load(recipe_path).check()
    for issue in issues:
        where = (
            ""
            if issue.parameter is None
            else f"{', '.join(issue.parameter)}: "
            if isinstance(issue.parameter, tuple)
            else f"{issue.parameter}: "
        )
        sys.stdout.write(f"{issue.severity}: {where}{issue.message}\n")
    if not issues:
        sys.stdout.write(f"{recipe_path}: builds\n")
    return 1 if any(issue.severity == "error" for issue in issues) else 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``solprop`` command line.

    Parameters
    ----------
    argv : sequence of str or None, optional
        The arguments, or ``None`` for the process's own.

    Returns
    -------
    int
        The exit status: zero on success, one when a recipe has an error, two for a
        refused recipe or arguments.

    Examples
    --------
    >>> from solprop.cli import main
    >>> main(["check", "missing.toml"])
    2
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "scan":
        from solprop.scan import main as scan  # noqa: PLC0415 - its own parser

        return scan(arguments[1:])
    args = _parser().parse_args(arguments)
    try:
        if args.command == "run":
            return _run(args.recipe, args.output, args.progress, resume=args.resume)
        if args.command == "check":
            return _check(args.recipe)
        from solprop.recipe import Recipe  # noqa: PLC0415 - imports the physics

        sys.stdout.write(Recipe.load(args.recipe).to_python(args.output))
    except (SolpropError, OSError) as error:
        sys.stderr.write(f"solprop: {error}\n")
        return 2
    return 0
