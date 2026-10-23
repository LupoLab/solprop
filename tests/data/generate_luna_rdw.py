# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna RDW results the M2 regression compares against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_rdw.py

It writes ``tests/data/luna_rdw_reduced.h5`` (every plane), ``luna_rdw_full.h5`` (the
planes the benchmark reads), ``luna_rdw_media.json`` (the two media the comparison
replaces on the solprop side, and the density path, sampled on each run's own axis) and
``luna_rdw.json`` (provenance). The test suite reads only those files: it never runs
Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout, and it runs in Julia's default environment rather than
against the checkout's own project, so no manifest there is touched.

The stored results are regression witnesses, not truth. Four things about them matter:

* **Two cases.** The *reduced* one is the M2 physics on a smaller fibre, a narrower
  band, two modes and a shorter window, so that a solprop run of it costs about twelve
  seconds and can sit in the slow lane; the *full* one is Luna's published example
  verbatim except for the two departures below.
* **The temperature is given to Luna, not taken from it.** Luna's ``roomtemp`` is
  293.15 K and solprop's ``ROOM_TEMPERATURE`` is 294.0 K, a 0.29 % difference in
  density that feeds straight into both the dispersion and the nonlinearity. Passing
  solprop's value removes it from the comparison, as ``luna_modes.json`` already does.
* **The shot noise is off.** Luna adds a modified-noise field by default and solprop
  adds none unless asked; a comparison of two different noise realisations would
  measure the noise.
* **The media are emitted, not assumed.** Luna's helium susceptibility is the Ermolov
  expansion divided by the 1 bar density although the fit is referenced to 1 atm
  and its cladding is an unreferenced 400-row table. Both are sampled here so that the
  comparison can give the solprop modes Luna's own refractive data and isolate the
  solver; the package defaults stay the cited ones.

The full witness is **trimmed**: 201 planes of four modes on 2049 bins is 27 MB of
complex doubles, and the only test that reads it looks at the last plane and the
statistics. What is dropped and how to regenerate it is recorded in the provenance.
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from solprop.constants import ROOM_TEMPERATURE

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
HERE = Path(__file__).parent
MEDIA = HERE / "luna_rdw_media.json"
PROVENANCE = HERE / "luna_rdw.json"

LAMBDA0 = 800e-9
TAU_FWHM = 7.5e-15
ENERGY = 275e-6

# Luna's published example, verbatim.
FULL = {
    "radius": 125e-6,
    "length": 1.5,
    "p_in": 0.8,
    "p_out": 0.0,
    "modes": 4,
    "lambda_min": 90e-9,
    "lambda_max": 4e-6,
    "trange": 400e-15,
    "n_save": 201,
}

# The reduced case: a shorter fibre at a higher entrance pressure, so that the pulse
# still self-compresses far enough to shed a dispersive wave, on half the modes and a
# narrower, shorter window. Chosen by measurement, not by taste -- solprop's run takes
# about twelve seconds and the 100-140 nm band carries 1.6 % of the input energy
# against 1.8 % for the full case, so the quantity the regression is built on is the
# same size in both.
REDUCED = {
    "radius": 125e-6,
    "length": 1.0,
    "p_in": 1.2,
    "p_out": 0.0,
    "modes": 2,
    "lambda_min": 100e-9,
    "lambda_max": 1.5e-6,
    "trange": 200e-15,
    "n_save": 51,
}

# Luna's name for the stored spectra, and what the trimmed full witness keeps.
FIELD = "E\u03c9"
FULL_PLANES = (0, -1)

# `Luna.Utils._native_fftw_threads[]`, encoded numerically in the Julia program because
# its `emit` is numbers-only.
FFTW_POOL_NAMES = {0: "none", 1: "native", 2: "julia-task", 3: "undetermined"}


def git(*arguments: str) -> str:
    """Run a read-only git command in the Luna checkout and return its output.

    Parameters
    ----------
    *arguments : str
        Arguments after ``git``.

    Returns
    -------
    str
        Standard output, stripped.
    """
    return subprocess.run(
        ["git", "-C", str(LUNA), *arguments],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def julia_program(reduced_path: Path, full_path: Path) -> str:
    """Build the Julia program, with the two cases interpolated as literals.

    Parameters
    ----------
    reduced_path : pathlib.Path
        Where Luna writes the reduced case.
    full_path : pathlib.Path
        Where Luna writes the full case.

    Returns
    -------
    str
        The complete program: a header of constants and the sidecar source.
    """
    lines = [
        "const GAS = :He",
        f"const LAMBDA0 = {LAMBDA0!r}",
        f"const TAU_FWHM = {TAU_FWHM!r}",
        f"const ENERGY = {ENERGY!r}",
        f"const TEMPERATURE = {ROOM_TEMPERATURE!r}",
        f'const REDUCED_PATH = raw"{reduced_path}"',
        f'const FULL_PATH = raw"{full_path}"',
    ]
    for name, case in (("REDUCED", REDUCED), ("FULL", FULL)):
        lines += [
            f"const {name}_{key.upper()} = {value!r}" for key, value in case.items()
        ]
    header = "\n".join(lines) + "\n"
    return header + (HERE / "luna_rdw.jl").read_text()


def decode(output: str) -> dict[str, list[float]]:
    """Parse the Julia program's tagged output into arrays.

    Parameters
    ----------
    output : str
        Standard output of the Julia program.

    Returns
    -------
    dict
        Tag to the list of values it carried.
    """
    emitted: dict[str, list[float]] = {}
    for line in output.splitlines():
        if "\t" not in line:
            continue
        tag, values = line.split("\t", 1)
        emitted[tag] = [float(value) for value in values.split(",")]
    return emitted


def store(source: Path, target: Path, planes: tuple[int, ...] | None) -> None:
    """Copy a Luna output file, keeping the planes a test reads and nothing spare.

    Luna writes its own source code into every file it produces -- 0.8 MB of it -- and
    a cache copy of the whole state for its resume, neither of which a witness needs.
    The field's plane axis is the first one as h5py reads it, Julia's arrays being
    column-major.

    Parameters
    ----------
    source : pathlib.Path
        The file Luna wrote.
    target : pathlib.Path
        Where to write the stored copy.
    planes : tuple of int or None
        Indices into the plane axis to keep, in the order given, or ``None`` for all.
    """
    dropped = ("meta/sourcecode", "meta/cache", "meta/cachehash")
    with h5py.File(source, "r") as original, h5py.File(target, "w") as stored:

        def copy(name: str, node: h5py.Group | h5py.Dataset) -> None:
            """Copy one node, subsetting the field along its plane axis."""
            if name.startswith(dropped):
                return
            if isinstance(node, h5py.Group):
                stored.require_group(name)
                return
            values = np.asarray(node[()])
            if name == FIELD and planes is not None:
                values = values[list(planes)]
            if values.ndim == 0:
                stored.create_dataset(name, data=values)
            else:
                stored.create_dataset(name, data=values, compression="gzip")

        original.visititems(copy)
        for key, value in original.attrs.items():
            stored.attrs[key] = value
        stored.attrs["stored_planes"] = json.dumps(
            "all" if planes is None else list(planes)
        )


def digest(path: Path) -> str:
    """Return the SHA-256 of a file, so a witness can be told from a copy.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        Its digest, hex.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Run Luna twice and write the witness files.

    Returns
    -------
    int
        A process exit code.
    """
    reduced_path = HERE / "luna_rdw_reduced.h5"
    full_path = HERE / "luna_rdw_full.h5"
    raw_reduced = HERE / "luna_rdw_reduced_raw.h5"
    raw_full = HERE / "luna_rdw_full_raw.h5"
    for path in (reduced_path, full_path, raw_reduced, raw_full):
        path.unlink(missing_ok=True)

    program = julia_program(raw_reduced, raw_full)
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", program],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stdout[-4000:])
        sys.stderr.write(result.stderr)
        return result.returncode
    emitted = decode(result.stdout)
    if "CONSTANTS" not in emitted:
        sys.stderr.write("Luna emitted nothing; is the sidecar program in place?\n")
        return 1

    store(raw_reduced, reduced_path, None)
    store(raw_full, full_path, FULL_PLANES)
    raw_sizes = {
        reduced_path.name: raw_reduced.stat().st_size,
        full_path.name: raw_full.stat().st_size,
    }
    raw_reduced.unlink()
    raw_full.unlink()

    MEDIA.write_text(json.dumps(emitted, indent=1, sort_keys=True) + "\n")
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
        "temperature_k": ROOM_TEMPERATURE,
        "shotnoise": False,
        "lambda0": LAMBDA0,
        "tau_fwhm": TAU_FWHM,
        "energy": ENERGY,
        "reduced": REDUCED,
        "full": FULL,
        "runtime_s": {
            "reduced": emitted["REDUCED_RUNTIME"][0],
            "full": emitted["FULL_RUNTIME"][0],
        },
        "julia_threads": int(emitted["THREADS"][0]),
        "blas_threads": int(emitted["THREADS"][1]),
        "fftw_threads": int(emitted["THREADS"][2]),
        "fftw_pool": FFTW_POOL_NAMES[int(emitted["THREADS"][3])],
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in (reduced_path, full_path, MEDIA)
        },
        "full_trimmed_to_planes": list(FULL_PLANES),
        "as_luna_wrote_them_bytes": raw_sizes,
        "note": (
            "Regression witnesses, not truth. Generated at solprop's room temperature "
            "of 294.0 K rather than Luna's 293.15 K, and with the shot noise off; both "
            "departures are so that the comparison measures the solver. Luna's helium "
            "susceptibility and cladding index are emitted in luna_rdw_media.json "
            "because the package defaults differ from them by design. The full "
            "case's field is trimmed "
            "to the planes listed above; regenerate with "
            "tests/data/generate_luna_rdw.py. julia_threads, blas_threads, "
            "fftw_threads and fftw_pool are read inside the Julia program rather "
            "than assumed."
        ),
    }
    PROVENANCE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(f"wrote {reduced_path.name}, {full_path.name}, {MEDIA.name}\n")
    sys.stdout.write(json.dumps(payload["files"], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
