# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna CtC results the M4 problem is compared against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_ctc.py

It writes ``tests/data/luna_ctc_reduced.h5`` (every plane), ``luna_ctc_full.h5`` (the
planes the benchmark reads), ``luna_ctc_media.json`` (the media the comparison replaces
on the solprop side, the density, and Luna's own molecular ionisation rate) and
``luna_ctc.json`` (provenance). The test suite reads only those files: it never runs
Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout, and it runs in Julia's default environment rather than
against the checkout's own project, so no manifest there is touched.

The stored results are regression witnesses, not truth. Five things about them matter:

* **Two cases.** The *full* one is the comb-to-continuum supercontinuum target --
  16 um radius, hydrogen at 40 bar, six HE_1m modes, 400 fs and 10 uJ at 515 nm on a
  4 ps field grid with no loss -- and Luna turns the Raman response on by itself for a
  molecular gas and the plasma on for a real field. Its length is **0.2 m and not the
  1.5 m of the script the target came from**: the length was arbitrary in that script,
  it is the one parameter that changes nothing else about the case -- the grid is still
  32768 samples on 16385 bins, the mode set is still six, the node budget is still the
  45 MB `docs/dev/architecture.md` section 8 sizes -- and 1.5 m costs the better part of
  a day in each code. The continuum is fully formed by 0.2 m; it is not asymptotic
  there, the red shoulder still growing with distance (measured on Luna at the reduced
  window: 3.5 % of the total in 900-1300 nm at 0.1 m, 11.8 % at 0.4 m). The *reduced*
  one is the same case on half the fibre, two modes and a narrower, shorter window;
  every reduction is justified where ``REDUCED`` is defined below.
* **The temperature is given to Luna, not taken from it.** Luna's ``roomtemp`` is
  293.15 K and solprop's ``ROOM_TEMPERATURE`` is 294.0 K, a 0.29 % difference in
  density that feeds the dispersion, the nonlinearity and -- for a molecular fill --
  the rotational populations of the Raman response. Passing solprop's value removes it
  from the comparison, as the RDW witness already does.
* **The shot noise is off.** Luna adds a modified-noise field by default and solprop
  adds none unless asked; a comparison of two different noise realisations would
  measure the noise.
* **The media are emitted, not assumed.** Hydrogen's susceptibility, its
  hyperpolarisability, the cladding index and Luna's own Raman response are sampled on
  each run's own axis, so that the comparison can give the solprop modes Luna's
  refractive data and isolate the solver; the package defaults stay the cited ones.
* **Luna's molecular ionisation rate is emitted too.** ``Ionisation.IonRateADK(:H2)``
  takes the potential alone -- it never calls ``PhysData.quantum_numbers``, which throws
  for hydrogen -- so it is the potential-only form with ``Z = 1``, ``l = 0`` and an
  occupancy of two. Sampling it across the field range this case reaches is what pins
  the rate the witness is compared on (solprop's own molecular ADK when this was
  generated; since SEP 3, Phase 4b the test-only ADK oracle of ``tests/rate_oracles.py``
  with Luna's recorded potential) to Luna's numbers.

The full witness is **trimmed**: 201 planes of six modes on 16 385 bins is 320 MB of
complex doubles, and the only test that reads it looks at the last plane and the
statistics. The statistics are subsampled too, on a stride, when a case takes more
accepted steps than the RDW witness's four thousand. What is dropped and how to
regenerate it is recorded in the provenance.
"""

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

from solprop.constants import ROOM_TEMPERATURE

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
HERE = Path(__file__).parent
MEDIA = HERE / "luna_ctc_media.json"
PROVENANCE = HERE / "luna_ctc.json"

LAMBDA0 = 515e-9
TAU_FWHM = 400e-15
ENERGY = 10e-6

# The target: prop_capillary(16e-6, L, :H2, 40.0; λ0=515e-9,
# pulses=Pulses.GaussPulse(λ0, τfwhm=400e-15, energy=10e-6), λlims=(170e-9, 4e-6),
# trange=4e-12, loss=false, modes=6). The realised grid is n_t 32768 on 16385 bins,
# with a fine grid of 65536 -- the case `docs/dev/architecture.md` section 8 sizes.
# L is 0.2 m rather than the 1.5 m of the script this came from; see the module
# docstring for why, and the provenance file records it.
FULL = {
    "radius": 16e-6,
    "length": 0.2,
    "pressure": 40.0,
    "modes": 6,
    "lambda_min": 170e-9,
    "lambda_max": 4e-6,
    "trange": 4e-12,
    "n_save": 201,
}

# The reduced case, chosen by measurement and not by taste. Measured on Luna at
# solprop's room temperature with the shot noise off:
#
# * **Half the fibre**, 0.10 m rather than 0.2 m. The supercontinuum is already
#   there: the exit spectrum spans 283-1517 nm at a ten-thousandth of its peak, which is
#   2.4 octaves about the 515 nm pump, and the two shoulders the case is run for carry
#   2.8 % (330-430 nm) and 3.5 % (900-1300 nm) of the total. Beyond 0.1 m the shoulders
#   grow but nothing new appears. Length is the smaller part of what this case saves --
#   the grid and the mode count are the rest, and together they are a factor of twenty.
# * **Two modes** rather than six, and two rather than three because of the walk-off.
#   A mode set shares one co-moving frame, so mode p drifts against the first
#   by the integral of its own group delay: for this capillary that is 1.08, 3.02, 5.82,
#   9.50 and 14.06 ps per metre for modes two to six. Over this case's 0.1 m the second
#   mode walks 0.108 ps against a realised window of 1.14 ps and stays well inside it,
#   so its exit content is physics; a third mode would walk 0.302 ps, a quarter of the
#   window, and start meeting the temporal collar, which is each code's own and not a
#   thing to compare. The second mode carries 24 % of the fundamental's energy at the
#   exit in Luna, so the intermodal drive is not a perturbation either. Six modes is
#   what the full case is for.
# * **A narrower window**, 250 nm-2 um rather than 170 nm-4 um. The floor drops the
#   third-harmonic collar at 172 nm and keeps the whole octave and the plasma blue
#   shift, which reach 283 nm; the ceiling is well beyond the 1517 nm red edge. This is
#   what takes the grid from 32768 samples to 4096.
# * **A quarter of the window in time**, 1 ps rather than 4 ps. That is two and a half
#   times the 400 fs pulse and holds its dispersive spread at this length; both windows
#   are far shorter than hydrogen's rotational coherence either way, so neither
#   resolves it and the truncation is the same in both.
#
# The plasma still ionises -- peak electron density 3.0e21 m^-3, at a peak field of
# 2.1e10 V/m -- so the molecular ADK rate is exercised, which is the other half of what
# this witness is for.
REDUCED = {
    "radius": 16e-6,
    "length": 0.10,
    "pressure": 40.0,
    "modes": 2,
    "lambda_min": 250e-9,
    "lambda_max": 2e-6,
    "trange": 1e-12,
    "n_save": 21,
}

# The field range Luna's hydrogen ADK rate is sampled over, V/m: from below its own
# threshold (Luna reports 5.59e8 V/m, where the rate first exceeds a double's smallest
# normal) to ten times the peak field the reduced case reaches, so that the comparison
# covers the cut-off, the physical range and the tail above it.
ADK_FIELD_MIN = 1e8
ADK_FIELD_MAX = 2e11
ADK_SAMPLES = 257

# Luna's name for the stored spectra, and what the trimmed full witness keeps.
FIELD = "Eω"
FULL_PLANES = (0, -1)

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
        "const GAS = :H2",
        f"const LAMBDA0 = {LAMBDA0!r}",
        f"const TAU_FWHM = {TAU_FWHM!r}",
        f"const ENERGY = {ENERGY!r}",
        f"const TEMPERATURE = {ROOM_TEMPERATURE!r}",
        f"const ADK_FIELD_MIN = {ADK_FIELD_MIN!r}",
        f"const ADK_FIELD_MAX = {ADK_FIELD_MAX!r}",
        f"const ADK_SAMPLES = {ADK_SAMPLES!r}",
        f'const REDUCED_PATH = raw"{reduced_path}"',
        f'const FULL_PATH = raw"{full_path}"',
    ]
    for name, case in (("REDUCED", REDUCED), ("FULL", FULL)):
        lines += [
            f"const {name}_{key.upper()} = {value!r}" for key, value in case.items()
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


def store(source: Path, target: Path, planes: tuple[int, ...] | None) -> int:
    """Copy a Luna output file, keeping the planes a test reads and nothing spare.

    Luna writes its own source code into every file it produces -- 0.8 MB of it -- and
    a cache copy of the whole state for its resume, neither of which a witness needs.
    The field's plane axis is the first one as h5py reads it, Julia's arrays being
    column-major. A statistics record longer than `MAX_STATS_SAMPLES` is stored on a
    stride: this case accepts tens of thousands of steps, and no test reads a
    statistic sample by sample.

    Parameters
    ----------
    source : pathlib.Path
        The file Luna wrote.
    target : pathlib.Path
        Where to write the stored copy.
    planes : tuple of int or None
        Indices into the plane axis to keep, in the order given, or ``None`` for all.

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
            """Copy one node, subsetting the field and the statistics."""
            if name.startswith(dropped):
                return
            if isinstance(node, h5py.Group):
                stored.require_group(name)
                return
            values = np.asarray(node[()])
            if name == FIELD and planes is not None:
                values = values[list(planes)]
            if name.startswith("stats/") and values.ndim > 0 and stride > 1:
                values = values[::stride]
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


def main() -> int:
    """Run Luna twice and write the witness files.

    Returns
    -------
    int
        A process exit code.
    """
    reduced_path = HERE / "luna_ctc_reduced.h5"
    full_path = HERE / "luna_ctc_full.h5"
    raw_reduced = HERE / "luna_ctc_reduced_raw.h5"
    raw_full = HERE / "luna_ctc_full_raw.h5"
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

    strides = {
        reduced_path.name: store(raw_reduced, reduced_path, None),
        full_path.name: store(raw_full, full_path, FULL_PLANES),
    }
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
        "stats_stride": strides,
        "as_luna_wrote_them_bytes": raw_sizes,
        "note": (
            "Regression witnesses, not truth. Generated at solprop's room temperature "
            "of 294.0 K rather than Luna's 293.15 K, and with the shot noise off; both "
            "departures are so that the comparison measures the solver. Luna turns the "
            "Raman response on by itself for hydrogen and the plasma on for a real "
            "field, so neither is passed. Hydrogen's susceptibility, its "
            "hyperpolarisability, the cladding index, Luna's Raman response and Luna's "
            "own molecular ADK rate are emitted in luna_ctc_media.json because the "
            "package defaults differ from them by design. The full case's field is "
            "trimmed to the "
            "planes listed above and both files' statistics are stored on the stride "
            "listed above; regenerate with tests/data/generate_luna_ctc.py. The full "
            "case is 0.2 m, not the 1.5 m of the script the target came from: that "
            "length was arbitrary and is the one parameter that changes nothing else "
            "about the case. The continuum is formed by 0.2 m but still evolving, the "
            "red shoulder growing with distance, so this is a regression witness and a "
            "benchmark rather than a converged endpoint. julia_threads, blas_threads, "
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
