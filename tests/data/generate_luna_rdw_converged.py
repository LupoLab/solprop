# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the full M2 case from Luna at a converged step tolerance.

Run once, by hand, on a machine with Julia and Luna; it takes about forty minutes:

.. code-block:: bash

    uv run python tests/data/generate_luna_rdw_converged.py

It writes ``tests/data/luna_rdw_full_converged.h5`` -- the same trimmed shape as
``luna_rdw_full.h5``, the exit spectrum and Luna's own per-mode statistics at every
accepted step -- and ``luna_rdw_full_converged_provenance.json`` beside it.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout, and it runs in Julia's default environment rather than
against the checkout's own project, so no manifest there is touched.

**Why this exists.** Every published Luna result carries ``rtol = 1e-6`` because its
interface cannot set a tolerance at all, so the M2 regression has always compared
solprop *converged* against Luna *unconverged*. A higher-mode disagreement there could
be either code, and the test could only record the number rather than bound it. This is
the other half: the same case at ``rtol = 1e-9``, through ``prop_capillary_args`` and
``Luna.run`` because that is the only route that can set one, against which solprop's
higher modes can be bounded at the percent level measured on the switches ladder --
while the regression against production Luna stays exactly as it is, because that is
what a user of Luna actually gets.
"""

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import h5py
from generate_luna_rdw import (  # type: ignore[import-not-found]
    ENERGY,
    FULL,
    FULL_PLANES,
    LAMBDA0,
    LUNA,
    TAU_FWHM,
    decode,
    digest,
    git,
    store,
)

from solprop.constants import ROOM_TEMPERATURE

HERE = Path(__file__).parent
SIDECAR = HERE / "luna_rdw_converged.jl"
WITNESS = HERE / "luna_rdw_full_converged.h5"
PROVENANCE = HERE / "luna_rdw_full_converged_provenance.json"

#: Two and a half decades tighter than the tolerance every Luna result carries.
RTOL = 1e-9
ATOL = 1e-10

#: The statistics kept. Luna writes seven of them at every accepted step, and at
#: 117 486 steps that is 26 MB of arrays no test reads. The energies are what the
#: higher-mode comparison is made of and ``z`` is what places them, so those two stay
#: -- at every step, which is finer than the 201 save planes -- and the rest is
#: dropped, the same judgement the witness generator makes about Luna's source code
#: and resume cache.
KEEP_STATS = ("stats/z", "stats/energy")


def julia_program(output_path: Path) -> str:
    """Return the sidecar with its constants prepended.

    Parameters
    ----------
    output_path : pathlib.Path
        Where Luna is to write its own output file.

    Returns
    -------
    str
        The program to hand to ``julia -e``.
    """
    constants = [
        "const GAS = :He",
        f"const LAMBDA0 = {LAMBDA0!r}",
        f"const TAU_FWHM = {TAU_FWHM!r}",
        f"const ENERGY = {ENERGY!r}",
        f"const TEMPERATURE = {ROOM_TEMPERATURE!r}",
        f"const RADIUS = {FULL['radius']!r}",
        f"const LENGTH = {FULL['length']!r}",
        f"const P_IN = {FULL['p_in']!r}",
        f"const P_OUT = {FULL['p_out']!r}",
        f"const MODES = {FULL['modes']!r}",
        f"const LAMBDA_MIN = {FULL['lambda_min']!r}",
        f"const LAMBDA_MAX = {FULL['lambda_max']!r}",
        f"const TRANGE = {FULL['trange']!r}",
        f"const N_SAVE = {FULL['n_save']!r}",
        f"const RTOL = {RTOL!r}",
        f"const ATOL = {ATOL!r}",
        f'const OUTPUT_PATH = raw"{output_path}"',
    ]
    return "\n".join(constants) + "\n" + SIDECAR.read_text()


def main() -> int:
    """Run Luna once at the converged tolerance and write the witness.

    Returns
    -------
    int
        A process exit code.
    """
    parser = argparse.ArgumentParser(description="The converged M2 witness.")
    parser.add_argument(
        "--runtime",
        type=float,
        default=None,
        help="trim the raw file already on disk, recording this wall time for it",
    )
    arguments = parser.parse_args()

    raw = HERE / "luna_rdw_full_converged_raw.h5"
    WITNESS.unlink(missing_ok=True)
    # A thirty-five minute run is not repeated because of a bug in what happens after
    # it, so a raw file already on disk is trimmed rather than regenerated.
    if arguments.runtime is not None and raw.exists():
        emitted = {"RUNTIME": [arguments.runtime], "THREADS": [1.0, 1.0]}
    else:
        raw.unlink(missing_ok=True)
        result = subprocess.run(
            ["julia", "--startup-file=no", "-e", julia_program(raw)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stdout[-4000:])
            sys.stderr.write(result.stderr[-4000:])
            return result.returncode
        emitted = decode(result.stdout)
        if "CONSTANTS" not in emitted:
            sys.stderr.write("Luna emitted nothing; is the sidecar in place?\n")
            return 1

    # The accepted-step count comes from the file Luna wrote, which is where time_luna
    # reads every case's expected count from too.
    with h5py.File(raw, "r") as written:
        steps = int(cast("h5py.Dataset", written["stats/z"]).shape[0])

    store(raw, WITNESS, FULL_PLANES)
    _drop_unread_statistics(WITNESS)
    raw_size = raw.stat().st_size
    raw.unlink()

    payload: dict[str, Any] = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "luna_path": str(LUNA),
        "luna_commit": git("rev-parse", "HEAD"),
        "luna_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "luna_describe": git("log", "-1", "--format=%h %s"),
        "julia_version": subprocess.run(
            ["julia", "--startup-file=no", "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "julia_threads": int(emitted["THREADS"][0]),
        "blas_threads": int(emitted["THREADS"][1]),
        "temperature_k": ROOM_TEMPERATURE,
        "shotnoise": False,
        "lambda0": LAMBDA0,
        "tau_fwhm": TAU_FWHM,
        "energy": ENERGY,
        "full": FULL,
        "rtol": RTOL,
        "atol": ATOL,
        "accepted_steps": steps,
        "runtime_s": emitted["RUNTIME"][0],
        "raw_bytes": raw_size,
        "stored_bytes": WITNESS.stat().st_size,
        "sha256": digest(WITNESS),
    }
    PROVENANCE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(
        f"{WITNESS.name}: {steps} accepted steps in {payload['runtime_s']:.0f} s, "
        f"{payload['stored_bytes'] / 1e6:.1f} MB stored of {raw_size / 1e6:.1f} MB\n"
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    raise SystemExit(main())


def _drop_unread_statistics(path: Path) -> None:
    """Rewrite a stored witness keeping only the statistics a test reads.

    :func:`generate_luna_rdw.store` drops Luna's source code and resume cache and keeps
    everything else, which is right for a witness whose statistics run over 4365 steps.
    This one runs over 117 486, where the five statistics nothing reads are 19 MB.

    Parameters
    ----------
    path : pathlib.Path
        The stored witness, rewritten in place.
    """
    trimmed = path.with_suffix(".trimmed.h5")
    with h5py.File(path, "r") as source, h5py.File(trimmed, "w") as target:

        def copy(name: str, node: h5py.Group | h5py.Dataset) -> None:
            """Copy one node unless it is an unread statistic."""
            if name.startswith("stats/") and name not in KEEP_STATS:
                return
            if isinstance(node, h5py.Group):
                target.require_group(name)
                return
            values = node[()]
            if getattr(values, "ndim", 0) == 0:
                target.create_dataset(name, data=values)
            else:
                target.create_dataset(name, data=values, compression="gzip")

        source.visititems(copy)
        for key, value in source.attrs.items():
            target.attrs[key] = value
        target.attrs["stats_kept"] = json.dumps(list(KEEP_STATS))
    trimmed.replace(path)
