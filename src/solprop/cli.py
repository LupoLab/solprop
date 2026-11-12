# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``solprop`` command line: run, check, export and resume recipes (SEP 41, S1).

``solprop run recipe.toml -o out.h5`` builds and runs a recipe, ``solprop resume
out.h5`` continues it from the file alone, ``solprop check``
reports what stops it building, ``solprop export --python`` prints the equivalent
script, ``solprop scan`` is ``python -m solprop.scan``, and ``solprop calibrate``
measures, shows or forgets the native backend's machine profile (SEP 46). The same
commands run as ``python -m solprop``. The manual page is
``docs/guide/setup/recipes.md``.
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
    """Return the command line's parser."""
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
    resume.add_argument(
        "--interpolated",
        action="store_true",
        help="continue from a last plane interpolated inside a step",
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
    calibrate = commands.add_parser(
        "calibrate",
        help="calibrate the native backend to this machine (SEP 46)",
        description="Measure this machine's native execution settings, or show, "
        "export, import or forget its profile.",
    )
    from solprop.calibration.command import add_arguments  # noqa: PLC0415 - lazy

    add_arguments(calibrate)
    return parser


def _stop_signals() -> tuple[signal.Signals, ...]:
    """Return the signals that stop a run from the command line.

    - Returns: SIGINT and SIGTERM, and on Windows SIGBREAK, which ``CTRL_BREAK_EVENT``
      sends.
    """
    names = ("SIGINT", "SIGTERM", "SIGBREAK")
    return tuple(getattr(signal, name) for name in names if hasattr(signal, name))


@contextmanager
def _stopping() -> Iterator[tuple[threading.Event, list[int]]]:
    """Map the stop signals onto a run's flag for the block's duration.

    The first signal sets the flag (SEP 41, S17), so the run writes its state and
    returns; a second, while it stops, interrupts at once.

    - Yields: the flag, and the list the signals received are appended to.
    """
    flag = threading.Event()
    received: list[int] = []

    def stop(number: int, frame: FrameType | None) -> None:
        """Set the run's flag on the first signal; interrupt on the second."""
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

    - ``received``: the signals that stopped it; empty when it completed.
    - Returns: zero, or ``128`` plus the first signal's number. A signal that came too
      late to stop anything, since the file says the run completed, gives zero too.
    """
    if not received:
        return 0
    from solprop.exceptions import OutputError  # noqa: PLC0415 - imports the physics
    from solprop.output.hdf5 import open_file  # noqa: PLC0415 - imports the physics

    # A signal during the build can stop a run before it has written anything.
    try:
        with open_file(output) as stored:
            outcome = stored.outcome
    except OSError, OutputError:
        outcome = None
    complete = outcome is not None and outcome.status == "complete"
    name = signal.Signals(received[0]).name
    if complete:
        sys.stderr.write(
            f"solprop: {name} arrived too late to stop the run; {output} is complete\n"
        )
        return 0
    sys.stderr.write(
        f"solprop: stopped by {name}; the state is "
        f"written to {output}, and solprop resume {output} continues it\n"
    )
    return 128 + received[0]


def _run(recipe_path: Path, output: Path, progress: str, *, resume: bool) -> int:
    """Build and run a recipe, stopping it cleanly on a signal.

    - Returns: zero when the run completes, and ``128`` plus the signal's number when a
      signal stopped it.
    """
    from solprop.recipe import Recipe  # noqa: PLC0415 - imports the physics

    recipe = Recipe.load(recipe_path)
    with _stopping() as (flag, received):
        recipe.run(output, progress=progress, resume=resume, stop=flag)
    return _stopped(received, output)


def _resume(output: Path, mode: bool | str, progress: str) -> int:
    """Continue a run from its output file alone (SEP 41, S2).

    - ``mode``: ``True`` or ``"interpolated"``.
    """
    from solprop.recipe import resume_file  # noqa: PLC0415 - imports the physics

    with _stopping() as (flag, received):
        resume_file(
            output,
            cast("bool | Literal['interpolated']", mode),
            progress=cast("ProgressChoice", progress),
            stop=flag,
        )
    return _stopped(received, output)


def _check(recipe_path: Path) -> int:
    """Print what stops a recipe building, or makes it warn.

    - Returns: zero when the recipe builds without an error, one otherwise.
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
        return _command(args)
    except (SolpropError, OSError) as error:
        sys.stderr.write(f"solprop: {error}\n")
        return 2


def _command(args: argparse.Namespace) -> int:
    """Carry out one parsed command other than ``scan``."""
    if args.command == "run":
        return _run(args.recipe, args.output, args.progress, resume=args.resume)
    if args.command == "check":
        return _check(args.recipe)
    if args.command == "resume":
        mode = "interpolated" if args.interpolated else True
        return _resume(args.output, mode, args.progress)
    if args.command == "calibrate":
        from solprop.calibration.command import run  # noqa: PLC0415 - reads the store

        return run(args)
    from solprop.recipe import Recipe  # noqa: PLC0415 - imports the physics

    sys.stdout.write(Recipe.load(args.recipe).to_python(args.output))
    return 0
