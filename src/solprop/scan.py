# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Launch an explicitly selected scan script without evaluating CLI expressions.

``python -m solprop.scan script.py [options]`` (and ``solprop scan``): :func:`main`
parses the options into one execution policy (``_policy``: ``Local``, ``Pool``,
``Queue`` or ``Slurm``), sets it as the default executor of the scan script, which it
then runs as ``__main__``; a :func:`~solprop.driver.scan.run_scan` with an explicit
``executor`` still takes precedence. ``--emit-slurm`` writes the Slurm array script
instead, and ``python -m solprop.scan collect QUEUE --output OUT.h5`` collects a durable
queue into a scan file. :func:`current_executor` reads the policy in force. No option
is evaluated as Python. The manual page is ``docs/guide/workflows/scans.md``.
"""

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from solprop.driver._scan_context import _EXECUTOR, current_executor
from solprop.driver.executors import Executor, Local, Pool, Queue, Slurm
from solprop.driver.executors.queue import collect

__all__ = ["current_executor", "main"]

_DESCRIPTION = (
    "Launch an explicitly selected scan script without evaluating CLI expressions."
)

# The collect mode is dispatched before the script parser sees its arguments, so
# argparse cannot advertise it on its own: `collect` would otherwise be read as a
# script path. Both modes name it from here so the two descriptions cannot drift.
_COLLECT_HELP = (
    "collection mode:\n"
    "  python -m solprop.scan collect QUEUE --output OUT.h5\n"
    "    Collect the durable artifacts of the shared queue directory QUEUE into\n"
    "    the standalone scan file OUT.h5. Idempotent: an already collected point\n"
    "    is left untouched and a second run rewrites nothing."
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a scan script under an explicit CLI policy.

    Parameters
    ----------
    argv : sequence of str or None, optional
        CLI arguments, or None to parse process arguments.

    Returns
    -------
    int
        Zero after a successful script execution.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "collect":
        parser = argparse.ArgumentParser(
            prog="python -m solprop.scan collect",
            description="Collect durable scan artifacts",
            epilog=_COLLECT_HELP,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.add_argument("queue", type=Path, help="the shared queue directory")
        parser.add_argument(
            "--output", "-o", type=Path, required=True, help="the scan file to write"
        )
        args = parser.parse_args(arguments[1:])
        collect(args.queue, args.output)
        return 0
    parser = _parser()
    args = parser.parse_args(arguments)
    executor = _policy(args, parser)
    if args.emit_slurm is not None:
        if not isinstance(executor, Slurm):
            parser.error("--emit-slurm requires a Slurm policy")
        args.emit_slurm.write_text(executor.render(), encoding="utf-8")
        return 0
    token = _EXECUTOR.set(executor)
    previous = sys.argv
    sys.argv = [str(args.script.resolve())]
    try:
        runpy.run_path(str(args.script.resolve()), run_name="__main__")
    finally:
        sys.argv = previous
        _EXECUTOR.reset(token)
    return 0


def _parser() -> argparse.ArgumentParser:
    """Declare CLI modes without interpreting values as Python expressions."""
    parser = argparse.ArgumentParser(
        description=_DESCRIPTION,
        epilog=_COLLECT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("script", type=Path, help="the scan script to run")
    parser.add_argument(
        "--local",
        action="store_true",
        help="run every point in this process, one after another",
    )
    parser.add_argument(
        "--queue", type=Path, help="the shared queue directory to claim points from"
    )
    parser.add_argument(
        "--chunk",
        help="claim only chunk k of n of the points, zero-based, written k/n",
    )
    parser.add_argument(
        "--submit", action="store_true", help="submit the scan as a Slurm array"
    )
    parser.add_argument(
        "--emit-slurm",
        type=Path,
        help="write the Slurm array script to this file instead of submitting it",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="the most worker processes (default: as the CPUs and memory admit)",
    )
    parser.add_argument(
        "--threads", type=int, default=1, help="threads per worker (default: 1)"
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        help="the memory the workers may use, in bytes (default: the allocation's)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="start a fresh interpreter for every point",
    )
    parser.add_argument(
        "--stale-age",
        type=float,
        default=600.0,
        help="seconds after which a remote claim on a point lapses (default: 600)",
    )
    parser.add_argument(
        "--ntasks", type=int, default=1, help="Slurm array tasks (default: 1)"
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="CPUs allocated to each Slurm task (default: 1)",
    )
    parser.add_argument(
        "--mem",
        default="4G",
        help="memory for each Slurm task, MiB without a K/M/G/T suffix (default: 4G)",
    )
    parser.add_argument(
        "--time",
        default="01:00:00",
        help="the Slurm wall-clock limit of each task (default: 01:00:00)",
    )
    parser.add_argument("--partition", help="the Slurm partition")
    parser.add_argument(
        "--gres", help="Slurm generic resources for each task, for example gpu:1"
    )
    parser.add_argument(
        "--mode",
        choices=("queue", "chunk"),
        default="queue",
        help="Slurm tasks claim points from a shared queue, or each runs a disjoint "
        "chunk (default: queue)",
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=0,
        help="controllers per Slurm task; 0 runs one (default: 0)",
    )
    return parser


def _policy(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Executor:
    """Resolve one execution default while retaining explicit script precedence."""
    slurm = args.submit or args.emit_slurm is not None
    worker_controls = (
        args.workers is not None
        or args.memory_limit is not None
        or args.threads != 1
        or args.fresh
    )
    if args.local and (args.queue is not None or args.chunk is not None or slurm):
        parser.error("--local cannot be combined with queue, chunk or Slurm modes")
    if args.local and worker_controls:
        parser.error("--local cannot use process-pool resource controls")
    if slurm:
        if args.memory_limit is not None:
            parser.error("use --mem for a Slurm allocation")
        if args.chunk is not None:
            parser.error("use --mode chunk and --ntasks for a Slurm submission")
        return Slurm(
            args.script,
            ntasks=args.ntasks,
            cpus_per_task=args.cpus_per_task,
            mem=args.mem,
            time=args.time,
            partition=args.partition,
            gres=args.gres,
            mode=args.mode,
            instances=args.instances,
            queue_path=args.queue,
            workers=args.workers,
            threads=args.threads,
            fresh=args.fresh,
        )
    chunk = None
    if args.chunk is not None:
        try:
            k, n = args.chunk.split("/")
            chunk = (int(k), int(n))
        except ValueError as error:
            parser.error(f"chunk must be zero-based k/n: {error}")
    if args.queue is not None or chunk is not None:
        return Queue(
            args.queue,
            stale_age=args.stale_age,
            pool=Pool(
                workers=args.workers,
                memory_limit=args.memory_limit,
                threads=args.threads,
                fresh=args.fresh,
            ),
            chunk=chunk,
        )
    return (
        Pool(
            workers=args.workers,
            memory_limit=args.memory_limit,
            threads=args.threads,
            fresh=args.fresh,
        )
        if worker_controls
        else Local()
    )


if __name__ == "__main__":
    raise SystemExit(main())
