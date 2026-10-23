# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna values that the linear and nonlinear tests compare against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_linear.py

It writes ``tests/data/luna_linear.json`` with a provenance block naming the Luna
commit, branch and Julia version. The test suite only reads that file: it never runs
Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout.

The stored numbers are regression witnesses, not truth. Where solprop deliberately
differs -- the whole of gamma acting as Kerr when Raman is off, the analytic rather than
differenced group delay, the loss policy -- ``tests/test_luna_linear.py`` names the
difference and asserts its size, and a test fails if a stored key is neither classified
as faithful nor as divergent.

Unlike ``generate_luna_reference.py`` this script asks one fixed question rather than a
data-driven list of them, so the parameters are interpolated into the Julia source
instead of being streamed over a line protocol. They are written with ``repr``, which
round-trips a double exactly and which Julia's parser reads back to the same bits, so
both sides work from identical numbers.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from solprop.constants import C

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_linear.json")

# A helium-filled capillary of the kind the RDW target simulation uses, on a grid small
# enough to store: the point is the operator's algebra, not the size of the problem.
CAPILLARY = {
    "core_radius": 125e-6,
    "gas": "He",
    "pressure": 0.8,
    "length": 0.15,
    "wavelength": 800e-9,
    "wavelength_limits": (200e-9, 3e-6),
    "trange": 60e-15,
}

# A silica-like GNLSE fibre: 1.45 phase index, 1.47 group index, 20 fs^2/m of
# group-delay dispersion and a little third order, with 0.2 dB/m of loss.
#
# The window reaches further into the blue than a GNLSE run normally would. That is on
# purpose: solprop's envelope grid measures its relative-frequency span symmetrically
# while Luna measures only the upper side (`EnvelopeGrid.from_limits`), so the two
# constructions agree only when the upper side is the wider one. Choosing a window
# where they do keeps this witness about the linear operator rather than about the
# grid rule, and `test_luna_linear.py` asserts the two axes still coincide.
GNLSE = {
    "wavelength": 1030e-9,
    "wavelength_limits": (350e-9, 2e-6),
    "trange": 300e-15,
    "length": 0.1,
    "aeff": 8e-11,
    "loss_db_per_m": 0.2,
}

# (gamma [1/(W m)], A_eff [m^2], f_R) for the coefficient mapping of
# Interface.jl:1071-1075.
MAPPINGS = [(0.011, 1.0, 0.18), (0.11, 8e-11, 0.0), (1.3, 1.0, 0.5)]

KERR = {"gamma3": 1e-60, "density": 2.5e25, "omega0": 1.83e15, "n_samples": 24}


def taylor_coefficients() -> list[float]:
    """Return the Taylor coefficients of the GNLSE fibre's propagation constant.

    Returns
    -------
    list of float
        ``beta_0, beta_1, beta_2, beta_3`` in s^n/m.
    """
    omega0 = 2.0 * np.pi * C / GNLSE["wavelength"]
    return [1.45 * omega0 / C, 1.47 / C, 2.0e-26, -1.0e-41]


def kerr_signals() -> dict[str, list[float]]:
    """Return the deterministic test signals the Kerr responses are evaluated on.

    The real signals are band-limited: their Nyquist bin is zero, so Luna's analytic
    signal (which drops that bin) and :func:`solprop.fourier.analytic_signal` (which
    keeps it, as :func:`scipy.signal.hilbert` does) agree exactly and the comparison
    tests the physics rather than that documented difference.

    Returns
    -------
    dict
        Time axis and the real and complex, scalar and two-component signals.
    """
    n = KERR["n_samples"]
    rng = np.random.default_rng(20260906)
    spectrum = rng.normal(size=(4, n // 2 + 1)) + 1j * rng.normal(size=(4, n // 2 + 1))
    spectrum[:, 0] = spectrum[:, 0].real
    spectrum[:, -1] = 0.0  # kill the Nyquist bin, where the two conventions differ
    real = np.fft.irfft(spectrum, n=n, axis=-1) * 1e8
    complex_parts = rng.normal(size=(4, n)) * 1e8
    dt = 2.0 * np.pi / (KERR["omega0"] * 8.0)
    return {
        "t": ((np.arange(n) - n / 2) * dt).tolist(),
        "field_x": real[0].tolist(),
        "field_y": real[1].tolist(),
        "envelope_x_re": complex_parts[0].tolist(),
        "envelope_x_im": complex_parts[1].tolist(),
        "envelope_y_re": complex_parts[2].tolist(),
        "envelope_y_im": complex_parts[3].tolist(),
    }


def julia_vector(values: list[float] | tuple[float, ...]) -> str:
    """Render a sequence of floats as a Julia vector literal, bit for bit.

    Parameters
    ----------
    values : sequence of float
        The numbers.

    Returns
    -------
    str
        A ``[a, b, c]`` literal whose elements parse to the same doubles.
    """
    return "[" + ", ".join(repr(float(value)) for value in values) + "]"


def julia_program(signals: dict[str, list[float]]) -> str:
    """Build the Julia program, with the parameters interpolated as literals.

    Parameters
    ----------
    signals : dict
        The output of :func:`kerr_signals`.

    Returns
    -------
    str
        The program to run under ``julia --startup-file=no``.
    """
    betas = taylor_coefficients()
    header = f"""
const CORE_RADIUS = {CAPILLARY["core_radius"]!r}
const GAS = :{CAPILLARY["gas"]}
const PRESSURE = {CAPILLARY["pressure"]!r}
const CAP_LENGTH = {CAPILLARY["length"]!r}
const CAP_LAMBDA0 = {CAPILLARY["wavelength"]!r}
const CAP_LIMS = ({CAPILLARY["wavelength_limits"][0]!r}, \
{CAPILLARY["wavelength_limits"][1]!r})
const CAP_TRANGE = {CAPILLARY["trange"]!r}

const GN_LAMBDA0 = {GNLSE["wavelength"]!r}
const GN_LIMS = ({GNLSE["wavelength_limits"][0]!r}, {GNLSE["wavelength_limits"][1]!r})
const GN_TRANGE = {GNLSE["trange"]!r}
const GN_LENGTH = {GNLSE["length"]!r}
const GN_AEFF = {GNLSE["aeff"]!r}
const GN_LOSS_DB = {GNLSE["loss_db_per_m"]!r}
const BETAS = {julia_vector(betas)}

const MAPPINGS = {julia_vector([v for row in MAPPINGS for v in row])}

const GAMMA3 = {KERR["gamma3"]!r}
const DENSITY = {KERR["density"]!r}
const KERR_OMEGA0 = {KERR["omega0"]!r}
const TAXIS = {julia_vector(signals["t"])}
const FIELD_X = {julia_vector(signals["field_x"])}
const FIELD_Y = {julia_vector(signals["field_y"])}
const ENV_X = {julia_vector(signals["envelope_x_re"])} .+ \
1im .* {julia_vector(signals["envelope_x_im"])}
const ENV_Y = {julia_vector(signals["envelope_y_re"])} .+ \
1im .* {julia_vector(signals["envelope_y_im"])}
"""
    return header + Path(__file__).with_name("luna_linear.jl").read_text()


def git(*args: str) -> str:
    """Run a read-only git command in the Luna checkout and return its output.

    Parameters
    ----------
    *args : str
        Arguments after ``git``.

    Returns
    -------
    str
        Standard output, stripped.
    """
    return subprocess.run(
        ["git", "-C", str(LUNA), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def decode(text: str) -> dict[str, list[float]]:
    """Decode the Julia program's output.

    Parameters
    ----------
    text : str
        Standard output of the Julia program.

    Returns
    -------
    dict
        One entry per tag, each a list of floats.

    Raises
    ------
    RuntimeError
        If Luna produced nothing, or repeated a tag.
    """
    sections: dict[str, list[float]] = {}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        tag, values = line.split("\t", 1)
        if tag in sections:
            msg = f"Luna emitted {tag!r} twice"
            raise RuntimeError(msg)
        sections[tag] = [float(value) for value in values.split(",")]
    if not sections:
        msg = "Luna produced no output"
        raise RuntimeError(msg)
    return sections


def main() -> int:
    """Run Luna and write the reference file.

    Returns
    -------
    int
        A process exit code.
    """
    signals = kerr_signals()
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", julia_program(signals)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode

    payload: dict[str, Any] = {
        "provenance": {
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
            "note": (
                "Regression witnesses, not truth. Generated by "
                "tests/data/generate_luna_linear.py; see that file and "
                "tests/test_luna_linear.py for the documented differences."
            ),
        },
        "parameters": {
            "capillary": CAPILLARY,
            "gnlse": {**GNLSE, "betas": taylor_coefficients()},
            "mappings": MAPPINGS,
            "kerr": KERR,
        },
        "signals": signals,
        "luna": decode(result.stdout),
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    sys.stdout.write(
        f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1024:.0f} kB) from Luna "
        f"{payload['provenance']['luna_commit']}\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
