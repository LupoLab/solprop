# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``solprop`` command line: run, check, export and resume recipes (SEP 41, S1).

``solprop run recipe.toml -o out.h5`` builds and runs a recipe, ``solprop resume
out.h5`` continues it from the file alone, ``solprop check``
reports what stops it building, ``solprop export --python`` prints the equivalent
script, and ``solprop scan`` is ``python -m solprop.scan``. The same commands run as
``python -m solprop``. The manual page is ``docs/guide/setup/recipes.md``.
"""

import argparse
import signal
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import TYPE_CHECKING, Literal, cast

from solprop.exceptions import SolpropError

if TYPE_CHECKING:
    from solprop.log import ProgressChoice

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
    resume = commands.add_parser(
        "resume", help="continue a run from its output file and the recipe it records"
    )
    resume.add_argument("output", type=Path, help="the HDF5 file to continue")
    how = resume.add_mutually_exclusive_group()
    how.add_argument(
        "--interpolated",
        action="store_true",
        help="continue from a last plane interpolated inside a step",
    )
    how.add_argument(
        "--force",
        action="store_true",
        help="continue across changed physics while the layout agrees",
    )
    resume.add_argument(
        "--memory-budget",
        default=None,
        help="the run's memory budget, such as '16 GiB' (default: four fifths of "
        "the machine, refused beyond)",
    )
    resume.add_argument(
        "--progress",
        choices=("auto", "bar", "log", "none"),
        default="auto",
        help="how the run reports its progress (default: auto)",
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


def _stop_signals() -> tuple[signal.Signals, ...]:
    """Return the signals that stop a run from the command line.

    Returns
    -------
    tuple of signal.Signals
        SIGINT and SIGTERM, and on Windows SIGBREAK, which ``CTRL_BREAK_EVENT``
        sends.
    """
    names = ("SIGINT", "SIGTERM", "SIGBREAK")
    return tuple(getattr(signal, name) for name in names if hasattr(signal, name))


@contextmanager
def _stopping() -> Iterator[tuple[threading.Event, list[int]]]:
    """Map the stop signals onto a run's flag for the block's duration.

    The first signal sets the flag (SEP 41, S17), so the run writes its state and
    returns; a second, while it stops, interrupts at once.

    Yields
    ------
    tuple
        The flag, and the list the signals received are appended to.
    """
    flag = threading.Event()
    received: list[int] = []

    def stop(number: int, frame: FrameType | None) -> None:
        """Set the run's flag on the first signal; interrupt on the second.

        Parameters
        ----------
        number : int
            The signal.
        frame : types.FrameType or None
            The interrupted frame, unused.

        Raises
        ------
        KeyboardInterrupt
            On a second signal, while the run is still stopping.
        """
        del frame
        if flag.is_set():
            raise KeyboardInterrupt
        received.append(number)
        flag.set()

    previous = {number: signal.signal(number, stop) for number in _stop_signals()}
    try:
        yield flag, received
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _stopped(received: list[int], output: Path) -> int:
    """Report how a run ended and return the exit status.

    Parameters
    ----------
    received : list of int
        The signals that stopped it; empty when it completed.
    output : pathlib.Path
        The file it wrote.

    Returns
    -------
    int
        Zero, or ``128`` plus the first signal's number.
    """
    if not received:
        return 0
    sys.stderr.write(
        f"solprop: stopped by {signal.Signals(received[0]).name}; the state is "
        f"written to {output}, and solprop resume {output} continues it\n"
    )
    return 128 + received[0]


def _run(recipe_path: Path, output: Path, progress: str, *, resume: bool) -> int:
    """Build and run a recipe, stopping it cleanly on a signal.

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
        Zero when the run completes, and ``128`` plus the signal's number when a
        signal stopped it.
    """
    from solprop.recipe import Recipe  # noqa: PLC0415 - imports the physics

    recipe = Recipe.load(recipe_path)
    with _stopping() as (flag, received):
        recipe.run(output, progress=progress, resume=resume, stop=flag)
    return _stopped(received, output)


def _resume(
    output: Path, mode: bool | str, memory_budget: str | None, progress: str
) -> int:
    """Continue a run from its output file alone (SEP 41, S2).

    Parameters
    ----------
    output : pathlib.Path
        The HDF5 file.
    mode : bool or str
        ``True``, ``"interpolated"`` or ``"force"``.
    memory_budget : str or None
        The caller's memory budget, or ``None`` for the machine's default as a
        refusal.
    progress : str
        The progress choice.

    Returns
    -------
    int
        As :func:`_run`'s.
    """
    from solprop.recipe import resume_file  # noqa: PLC0415 - imports the physics

    with _stopping() as (flag, received):
        resume_file(
            output,
            cast("bool | Literal['interpolated', 'force']", mode),
            memory_budget=memory_budget,
            progress=cast("ProgressChoice", progress),
            stop=flag,
        )
    return _stopped(received, output)


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
        if args.command == "resume":
            mode = (
                "force" if args.force else "interpolated" if args.interpolated else True
            )
            return _resume(args.output, mode, args.memory_budget, args.progress)
        from solprop.recipe import Recipe  # noqa: PLC0415 - imports the physics

        sys.stdout.write(Recipe.load(args.recipe).to_python(args.output))
    except (SolpropError, OSError) as error:
        sys.stderr.write(f"solprop: {error}\n")
        return 2
    return 0
