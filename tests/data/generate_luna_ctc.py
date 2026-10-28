# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna CtC result the reduced CtC case is compared against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_ctc.py

It writes ``tests/data/luna_ctc_reduced.h5`` (every plane), ``luna_ctc_media.json`` (the
media the comparison replaces on the solprop side, the density, and Luna's own molecular
ionisation rate) and ``luna_ctc.json`` (provenance). The test suite reads only those
files: it never runs Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout, and it runs in Julia's default environment (or the one
``JULIA_PROJECT`` names) rather than against the checkout's own project, so no manifest
there is touched. ``LUNA_DIR`` names the checkout whose commit is recorded.

The stored result is a regression witness for the solver, not truth: the reduced case
tests the speed and accuracy of CtC-type propagation -- hydrogen's Raman response on the
doubled grid, the molecular plasma, a field grid -- over a short length. It is
noise-free, so it forms no continuum (below); seeded with shot noise, solprop's own run
of the case forms the Raman cascade, which a test checks without Luna. Four things about
the witness matter:

* **One case.** 16 um radius, hydrogen at 40 bar, the fundamental HE_11 mode alone,
  400 fs and 10 uJ at 515 nm on a 2 ps, 250 nm-2 um field grid over 0.1 m with no loss.
  Luna turns the Raman response on by itself for a molecular gas and the plasma on for a
  real field. A full-shape case (six modes, 4 ps, 170 nm-4 um) was stored beside it
  until 2026-10-24 and retired with its comparison; the full case is a solprop-only
  benchmark.
* **The temperature is given to Luna, not taken from it.** The case runs at 294.0 K
  (``TEMPERATURE`` below), solprop's room temperature when the witnesses were first
  made, and the tests pass the same value to solprop from the provenance. A 0.29 %
  difference in density feeds the dispersion, the nonlinearity and -- for a molecular
  fill -- the rotational populations of the Raman response, so one temperature on both
  sides removes it from the comparison, as the RDW witness already does.
* **The shot noise is off.** Luna adds a modified-noise field by default and solprop
  adds none unless asked; a comparison of two different noise realisations would
  measure the noise.
* **The media are emitted, not assumed.** Hydrogen's susceptibility, its
  hyperpolarisability, the cladding index, Luna's own Raman response and Luna's own
  molecular ionisation rate are sampled on the run's own axis, so that the comparison
  can give solprop Luna's material data and isolate the solver; the package defaults
  stay the cited ones. ``Ionisation.IonRateADK(:H2)`` takes the potential alone -- it
  never calls ``PhysData.quantum_numbers``, which throws for hydrogen -- so it is the
  potential-only form with ``Z = 1``, ``l = 0`` and an occupancy of two; the tests use
  it through the ADK oracle of ``tests/rate_oracles.py`` with Luna's recorded
  potential.

The statistics are stored on a stride when the case takes more accepted steps than the
RDW witness's four thousand; the provenance records the stride.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
HERE = Path(__file__).parent
WITNESS = HERE / "luna_ctc_reduced.h5"
MEDIA = HERE / "luna_ctc_media.json"
PROVENANCE = HERE / "luna_ctc.json"

# solprop's room temperature when the witnesses were first made.
# ``solprop.constants.ROOM_TEMPERATURE`` has since become Luna's 293.15 K; the witness
# keeps 294.0 K, which the tests read from the provenance.
TEMPERATURE = 294.0

LAMBDA0 = 515e-9
TAU_FWHM = 400e-15
ENERGY = 10e-6

# The reduced case, chosen by measurement and not by taste:
#
# * **0.1 m.** Long enough for many steps of the Raman and plasma sources at a cost of
#   about a minute in each code, and short enough for the suite. Noise-free and with
#   the pump clear of the window edges, neither code broadens the exit spectrum beyond
#   about 485-544 nm at a ten-thousandth of its peak over this length: nothing seeds the
#   Stokes band. With shot noise as the seed, solprop's run forms the cascade and spans
#   about 310-1610 nm; that is checked without Luna, whose noise is its own.
# * **One mode.** The case measures CtC-type propagation, not intermodal coupling; the
#   multimode routes are covered by the RDW witnesses and the full-shape benchmark.
# * **A narrower band**, 250 nm-2 um rather than the full shape's 170 nm-4 um. The floor
#   drops the third-harmonic collar at 172 nm; this is what takes the grid from 32768
#   samples to 8192.
# * **2 ps in time**, five times the 400 fs pulse. A 1 ps window was used until
#   2026-10-24 and is too narrow: its realised edge is +-569 fs, where the pump still
#   has 3.6e-3 of its peak intensity, and its temporal collar starts at +-500 fs, where
#   it has 1.3e-2. Clipping the launched pump seeds stimulated Raman scattering, which
#   this noise-free case amplifies to saturation, so a 1 ps run measured each code's
#   boundary rather than the physics: about a tenth of the energy ended in Stokes
#   light that a wider noise-free window does not produce, and the continuum the case
#   was once described as forming grew from that seed. With 2 ps the collar starts at
#   +-1 ps, where the pump is at 3e-8 of its peak, and the exit field agrees with a
#   4 ps run to 1e-5.
#   Both windows are far shorter than hydrogen's rotational coherence, so neither
#   resolves it and the truncation of the response is the same in both.
REDUCED = {
    "radius": 16e-6,
    "length": 0.10,
    "pressure": 40.0,
    "modes": 1,
    "lambda_min": 250e-9,
    "lambda_max": 2e-6,
    "trange": 2e-12,
    "n_save": 21,
}

# The field range Luna's hydrogen ADK rate is sampled over, V/m: from below its own
# threshold (Luna reports 5.59e8 V/m, where the rate first exceeds a double's smallest
# normal) to ten times the peak field the case reaches, so that the comparison
# covers the cut-off, the physical range and the tail above it.
ADK_FIELD_MIN = 1e8
ADK_FIELD_MAX = 2e11
ADK_SAMPLES = 257


# Statistics longer than this are stored on a stride. Luna records one sample per
# accepted step, and the RDW witness was stored whole at four thousand of them.
MAX_STATS_SAMPLES = 8192

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


def julia_program(path: Path) -> str:
    """Build the Julia program, with the case interpolated as literals.

    Parameters
    ----------
    path : pathlib.Path
        Where Luna writes the case.

    Returns
    -------
    str
        The complete program: a header of constants and the sidecar source.
    """
    lines = [
        "const GAS = :H2",
        f"const LAMBDA0 = {LAMBDA0!r}",
        f"const TAU_FWHM = {TAU_FWHM!r}",
        f"const ENERGY = {ENERGY!r}",
        f"const TEMPERATURE = {TEMPERATURE!r}",
        f"const ADK_FIELD_MIN = {ADK_FIELD_MIN!r}",
        f"const ADK_FIELD_MAX = {ADK_FIELD_MAX!r}",
        f"const ADK_SAMPLES = {ADK_SAMPLES!r}",
        f'const REDUCED_PATH = raw"{path}"',
    ]
    lines += [
        f"const REDUCED_{key.upper()} = {value!r}" for key, value in REDUCED.items()
    ]
    header = "\n".join(lines) + "\n"
    return header + (HERE / "luna_ctc.jl").read_text()


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


def store(source: Path, target: Path) -> int:
    """Copy a Luna output file, keeping what a test reads and nothing spare.

    Luna writes its own source code into every file it produces -- 0.8 MB of it -- and
    a cache copy of the whole state for its resume, neither of which a witness needs.
    The field's plane axis is the first one as h5py reads it, Julia's arrays being
    column-major. A statistics record longer than `MAX_STATS_SAMPLES` is stored on a
    stride, since no test reads a statistic sample by sample.

    Parameters
    ----------
    source : pathlib.Path
        The file Luna wrote.
    target : pathlib.Path
        Where to write the stored copy.

    Returns
    -------
    int
        The stride the statistics were stored on; one if they were stored whole.
    """
    dropped = ("meta/sourcecode", "meta/cache", "meta/cachehash")
    with h5py.File(source, "r") as original, h5py.File(target, "w") as stored:
        steps = len(np.asarray(cast("h5py.Dataset", original["stats/z"])[()]))
        stride = max(1, -(-steps // MAX_STATS_SAMPLES))

        def copy(name: str, node: h5py.Group | h5py.Dataset) -> None:
            """Copy one node, subsetting the statistics."""
            if name.startswith(dropped):
                return
            if isinstance(node, h5py.Group):
                stored.require_group(name)
                return
            values = np.asarray(node[()])
            if name.startswith("stats/") and values.ndim > 0 and stride > 1:
                values = values[::stride]
            if values.ndim == 0:
                stored.create_dataset(name, data=values)
            else:
                stored.create_dataset(name, data=values, compression="gzip")

        original.visititems(copy)
        for key, value in original.attrs.items():
            stored.attrs[key] = value
        stored.attrs["stats_stride"] = stride
        stored.attrs["stats_samples_as_luna_wrote_them"] = steps
    return stride


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


def luna_branch() -> str:
    """Return the branch the Luna commit is on, also from a detached clone.

    Returns
    -------
    str
        The checked-out branch, or else the first branch, local or remote, that
        contains the commit, without its remote's name.
    """
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != "HEAD":
        return branch
    names = git("branch", "--all", "--contains", "HEAD", "--format=%(refname:short)")
    # A detached checkout lists itself first, in parentheses.
    branches = [name for name in names.splitlines() if not name.startswith("(")]
    return branches[0].split("/", 1)[-1]


def luna_path() -> str:
    """Return the Luna checkout's location for the record, without a local path.

    Returns
    -------
    str
        The path relative to the home directory, or a description when it lies
        elsewhere (an exported clone of a recorded commit, for example).
    """
    home = Path.home()
    if LUNA.is_relative_to(home):
        return str(Path("~") / LUNA.relative_to(home))
    return "an exported clone of the Luna repository at the recorded commit"


def main() -> int:
    """Run Luna and write the witness files.

    Returns
    -------
    int
        A process exit code.
    """
    argparse.ArgumentParser(
        description="Generate the stored Luna CtC witness."
    ).parse_args()
    raw = HERE / "luna_ctc_reduced_raw.h5"
    for path in (WITNESS, raw):
        path.unlink(missing_ok=True)
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", julia_program(raw)],
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
    stride = store(raw, WITNESS)
    raw_size = raw.stat().st_size
    raw.unlink()
    threads = emitted.pop("THREADS")
    MEDIA.write_text(json.dumps(emitted, indent=1, sort_keys=True) + "\n")
    payload: dict[str, Any] = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "luna_path": luna_path(),
        "luna_commit": git("rev-parse", "HEAD"),
        "luna_branch": luna_branch(),
        "luna_describe": git("log", "-1", "--format=%h %s"),
        "julia_version": subprocess.run(
            ["julia", "--startup-file=no", "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        "julia_threads": int(threads[0]),
        "blas_threads": int(threads[1]),
        "fftw_threads": int(threads[2]),
        "fftw_pool": FFTW_POOL_NAMES[int(threads[3])],
        "temperature_k": TEMPERATURE,
        "shotnoise": False,
        "lambda0": LAMBDA0,
        "tau_fwhm": TAU_FWHM,
        "energy": ENERGY,
        "reduced": REDUCED,
        "runtime_s": emitted["REDUCED_RUNTIME"][0],
        "stats_stride": stride,
        "as_luna_wrote_it_bytes": raw_size,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in (WITNESS, MEDIA)
        },
        "note": (
            "A noise-free regression witness for the solver, not truth; it forms no "
            "continuum, since nothing seeds the Stokes band. "
            "Generated at 294.0 K rather than Luna's 293.15 K, and with the shot "
            "noise off, so that the comparison measures the solver. Luna turns the "
            "Raman response on by itself for hydrogen and the plasma on for a real "
            "field, so neither is passed. Hydrogen's susceptibility, its "
            "hyperpolarisability, the cladding index, Luna's Raman response and "
            "Luna's own molecular ADK rate are emitted in luna_ctc_media.json because "
            "the package defaults differ from them by design. Regenerate with "
            "tests/data/generate_luna_ctc.py. The thread counts and the FFTW pool are "
            "read inside the Julia program rather than assumed."
        ),
    }
    PROVENANCE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(json.dumps(payload["files"], indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
