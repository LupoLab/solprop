# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Launch an explicitly selected scan script without evaluating CLI expressions."""

import argparse
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from solprop.driver._scan_context import _EXECUTOR, current_executor
from solprop.driver.executors import Executor, Local, Pool, Queue, Slurm
from solprop.driver.executors.queue import collect

__all__ = ["current_executor", "main"]

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
        parser.add_argument("queue", type=Path)
        parser.add_argument("--output", "-o", type=Path, required=True)
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
    """Declare CLI modes without interpreting values as Python expressions.

    Returns
    -------
    argparse.ArgumentParser
        Parser for safe execution and Slurm resource controls.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=_COLLECT_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("script", type=Path)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--chunk")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--emit-slurm", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--memory-limit", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--stale-age", type=float, default=600.0)
    parser.add_argument("--ntasks", type=int, default=1)
    parser.add_argument("--cpus-per-task", type=int, default=1)
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--time", default="01:00:00")
    parser.add_argument("--partition")
    parser.add_argument("--gres")
    parser.add_argument("--mode", choices=("queue", "chunk"), default="queue")
    parser.add_argument("--instances", type=int, default=0)
    return parser


def _policy(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Executor:
    """Resolve one execution default while retaining explicit script precedence.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line execution and resource controls.
    parser : argparse.ArgumentParser
        Argument parser used to report incompatible controls.

    Returns
    -------
    Executor
        CLI default, overridden by explicit script executors.
    """
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
