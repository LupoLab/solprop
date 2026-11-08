# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run with no window: ``python -m solprop.service.job FOLDER`` (design 13.4).

What a window's worker does over pipes, this does from a folder alone. It reads the
run's spec from ``spec.json`` (:data:`~solprop.service.worker.SPEC`), runs it with its
record (:mod:`solprop.service.record`), and exits. It is how a run starts directly on a
machine, as a detached process, and inside a Slurm job; it outlives whoever started it,
and anyone who can read the folder can follow it.
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
        The folder, alone; ``None`` reads the command line.

    Returns
    -------
    int
        The exit status: 0 when the run ended (finished, stopped or failed, as its
        record says), 2 when the folder holds no spec.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("usage: python -m solprop.service.job FOLDER", file=sys.stderr)  # noqa: T201
        return 2
    folder = Path(arguments[0]).resolve()
    try:
        value = json.loads((folder / SPEC).read_text(encoding="utf-8"))
        spec = spec_from_value(value)
    except (OSError, ValueError, ServiceError) as error:
        print(f"no run to start in {folder}: {error}", file=sys.stderr)  # noqa: T201
        return 2
    launcher = "slurm" if os.environ.get("SLURM_JOB_ID") else "direct"
    spec = replace(spec, folder=str(folder), launcher=launcher)
    if spec.threads is not None:
        os.environ["SOLPROP_THREADS"] = str(spec.threads)
    from solprop.service.run import execute  # noqa: PLC0415 - after the threads

    execute(spec)
    return 0


if __name__ == "__main__":  # pragma: no cover - the command itself
    raise SystemExit(main())
