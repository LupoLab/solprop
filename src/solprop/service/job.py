# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run with no window: ``python -m solprop.service.job FOLDER`` (design 13.4).

Every run is this process, and needs nothing but its folder. It reads the run's spec
from ``spec.json`` (:data:`~solprop.service.worker.SPEC`), runs it with its record
(:mod:`solprop.service.record`), and exits. It is how a run starts directly on a
machine, as a detached process, and inside a Slurm job; it outlives whoever started it,
and anyone who can read the folder can follow it.

``--slurm`` says the process is the Slurm job solprop submitted for the run (its batch
script, :func:`~solprop.service.launch.slurm_script`, passes it), so the run records
that job as its own. Without it the run is a direct one, even inside a Slurm
allocation: the allocation is not the run's to cancel.
"""

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from solprop.service.messages import ServiceError
from solprop.service.worker import SPEC, spec_from_value

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the run a folder's ``spec.json`` describes, with its record.

    Parameters
    ----------
    argv : Sequence of str or None, optional
        The folder, after ``--slurm`` for the Slurm job solprop submitted for the run;
        ``None`` reads the command line.

    Returns
    -------
    int
        The exit status: 0 when the run ended (finished, stopped or failed, as its
        record says), 2 when the folder holds no spec.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    slurm = arguments[:1] == ["--slurm"]
    if slurm:
        del arguments[0]
    if len(arguments) != 1:
        usage = "usage: python -m solprop.service.job [--slurm] FOLDER"
        print(usage, file=sys.stderr)  # noqa: T201
        return 2
    folder = Path(arguments[0]).resolve()
    try:
        value = json.loads((folder / SPEC).read_text(encoding="utf-8"))
        spec = spec_from_value(value)
    except (OSError, ValueError, ServiceError) as error:
        print(f"no run to start in {folder}: {error}", file=sys.stderr)  # noqa: T201
        return 2
    spec = replace(spec, folder=str(folder), launcher="slurm" if slurm else "direct")
    if spec.threads is not None:
        os.environ["SOLPROP_THREADS"] = str(spec.threads)
    from solprop.service.run import execute  # noqa: PLC0415 - after the threads

    execute(spec)
    return 0


if __name__ == "__main__":  # pragma: no cover - the command itself
    raise SystemExit(main())
