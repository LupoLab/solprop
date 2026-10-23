# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the stored Luna Raman responses that the Raman tests compare against.

Run once, by hand, on a machine with Julia and Luna:

.. code-block:: bash

    uv run python tests/data/generate_luna_raman.py

It writes ``tests/data/luna_raman.json`` with a provenance block naming the Luna
commit, branch and Julia version. The test suite only reads that file: it never runs
Julia, so the suite stays offline and machine-independent.

Luna is read-only reference material. This script calls it and reads its output; it
never modifies the checkout.

The stored numbers are regression witnesses, not truth. Four things about them matter:

* **Luna's ``h`` is this package's times ``eps0``.** Luna's polarisation expression
  carries no vacuum permittivity outside the integral (``Nonlinear.jl:587``) and hides
  it inside every scale factor as ``(4 pi eps0)^2``, so Luna's response is quadratic in
  ``eps0`` where solprop's is linear. The comparison divides by Luna's own ``eps0``.
* **Luna's constants are CODATA 2014 and solprop's are CODATA 2022.** The script emits
  Luna's, so the test can rescale by the exact power of each where the dependence is a
  monomial. Two shifts do not cancel that way: ``m_u`` (1.7e-8) enters the vibrational
  scale factor as ``1/mu``, which *is* a monomial and is rescaled; ``hbar/k_B``
  (3.3e-7) enters the rotational populations through ``exp(-E_J/(k_B T))``, which is
  not, and sets a floor of about 1e-6 on any rotational comparison, so a 1e-10 bound
  is unattainable there.
* **The density is given in amagat, not in pascals.** The dephasing depends on the
  density only through ``rho/amg``, and ``amg = atm/(k_B * 273.15)`` differs between the
  two constant sets. Passing a multiple of each side's own amagat makes the dephasing
  time identical on both sides and takes the equation of state -- CoolProp against
  pyfluids -- out of the comparison entirely.
* **Luna is asked for its untapered response.** ``CombinedRamanResponse`` has two call
  signatures: the in-place array one applies a Planck taper flat to 70 % of the grid,
  and the scalar one does not (``Raman.jl:256-264``). solprop deliberately replaces
  that taper with one shaped by the grid's temporal collar, so
  the witness uses the scalar path and the taper is tested separately.
"""

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
OUTPUT = Path(__file__).with_name("luna_raman.json")

TEMPERATURE = 294.0
"""Temperature the rotational populations are built at, K.

Passed explicitly: Luna's ``roomtemp`` is 293.15 K and solprop's default is 294.0 K,
and a 0.3 % temperature difference moves every population.
"""

AMAGAT_MULTIPLE = 0.917
"""Density, in multiples of each side's own amagat: about one bar at 294 K."""

MIN_J = 0
MAX_J = 50

# Three axes, so that both the oscillation and the decay are compared. The first
# resolves the fastest line, the second a vibrational coherence of a few picoseconds,
# the third a rotational one of nanoseconds. Aliasing on the coarse axes does not
# matter: the comparison is pointwise at times both sides are given.
TIME_AXES: dict[str, tuple[int, float]] = {
    "fast": (512, 2e-15),
    "medium": (256, 1e-13),
    "slow": (256, 2e-11),
}

# gas, whether to include the rotation, whether to include the vibration.
CASES: dict[str, tuple[str, bool, bool]] = {
    "h2-rotation": ("H2", True, False),
    "h2-vibration": ("H2", False, True),
    "h2-both": ("H2", True, True),
    "n2-rotation": ("N2", True, False),
    "n2-vibration": ("N2", False, True),
    "d2-rotation": ("D2", True, False),
    "ch4-vibration": ("CH4", False, True),
    "sf6-vibration": ("SF6", False, True),
}

# The normalised single damped oscillator, whose scale factor is a pure number: no
# physical constant enters it, so it compares at round-off whatever the CODATA set.
SDO = (1.0, 2.0e14, 1.0e-13)

JULIA_PROGRAM = r"""
# A plain text protocol rather than JSON.jl: this must run against whatever Julia
# environment the machine has, without installing anything into it.
#   in :  C<TAB>key
#         H<TAB>key<TAB>gas<TAB>rotation<TAB>vibration<TAB>temp<TAB>amagat<TAB>t1,t2,...
#         L<TAB>key<TAB>gas<TAB>temp<TAB>minJ<TAB>maxJ
#         V<TAB>key<TAB>gas
#         S<TAB>key<TAB>K<TAB>Omega<TAB>tau2
#         Q<TAB>key
#   out:  R<TAB>key<TAB>comma separated values
using Luna
import Luna.PhysData: raman_parameters, ε_0, ħ, k_B, c, m_u, amg, atm
import Luna.Raman

flag(s) = s == "1"
emit(key, values) = println("R\t", key, "\t", join(string.(values), ","))

# A two-point grid: the scalar call signature ignores it, and it is the untapered one.
const DUMMY_T = [0.0, 1e-15]

for line in eachline(stdin)
    isempty(strip(line)) && continue
    f = split(line, '\t')
    kind, key = f[1], f[2]
    if kind == "C"
        emit(key, [ε_0, ħ, k_B, c, m_u, amg, atm])
    elseif kind == "H"
        sym = Symbol(f[3])
        temp = parse(Float64, f[6])
        rho = parse(Float64, f[7]) * amg
        t = parse.(Float64, split(f[8], ','))
        R = Raman.molecular_raman_response(DUMMY_T, raman_parameters(sym);
                                           rotation=flag(f[4]),
                                           vibration=flag(f[5]),
                                           temp=temp)
        emit(key, [R(ti, rho) for ti in t])
    elseif kind == "L"
        sym = Symbol(f[3])
        rp = raman_parameters(sym)
        temp = parse(Float64, f[4])
        minJ = parse(Int, f[5])
        maxJ = parse(Int, f[6])
        # The level ladder, recomputed here because Luna keeps only the lines.
        J = minJ:maxJ
        EJ = @. 2*pi*ħ*c*(rp.B*J*(J + 1) - rp.D*(J*(J + 1))^2)
        mJ = findfirst(x -> x < 0.0, diff(EJ))
        if !isnothing(mJ)
            J = J[1:mJ]
            EJ = EJ[1:mJ]
        end
        DJ = map(x -> isodd(x) ? rp.qJodd : rp.qJeven, J)
        pop = @. DJ*(2*J + 1)*exp(-EJ/(k_B*temp))
        pop ./= sum(pop)
        emit(key * "-J", collect(J))
        emit(key * "-EJ", EJ)
        emit(key * "-POP", pop)
        # The lines themselves, straight out of Luna's constructor.
        hr = Raman.raman_response(DUMMY_T, sym; vibration=false, temp=temp,
                                  minJ=minJ, maxJ=maxJ).Rs[1]
        emit(key * "-OMEGA", [Ri.Ω for Ri in hr.Rs])
        emit(key * "-K", [Ri.K for Ri in hr.Rs])
    elseif kind == "V"
        sym = Symbol(f[3])
        rp = raman_parameters(sym)
        hv = Raman.raman_response(DUMMY_T, sym; rotation=false).Rs[1]
        emit(key, [hv.K, hv.Ω])
    elseif kind == "S"
        K = parse(Float64, f[3])
        Om = parse(Float64, f[4])
        tau2 = parse(Float64, f[5])
        sdo = Raman.RamanRespNormedSingleDampedOscillator(K, Om, tau2)
        emit(key, [sdo.K])
    elseif kind == "Q"
        R = Raman.raman_response(collect(range(0, stop=1e-12, length=1024)), :SiO2)
        emit(key, [R.scale])
    end
end
"""


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


def time_axis(name: str) -> list[float]:
    """Return one of the delay axes the responses are sampled on.

    Parameters
    ----------
    name : str
        A key of :data:`TIME_AXES`.

    Returns
    -------
    list of float
        Delays in s, starting at zero.
    """
    count, spacing = TIME_AXES[name]
    return [index * spacing for index in range(count)]


def build_query() -> list[str]:
    """Assemble every line to send to the Julia program.

    Returns
    -------
    list of str
        The tab-separated request lines, in the order the output is expected.
    """
    lines = [
        "C\tCONSTANTS",
        "Q\tSILICA_SCALE",
        f"S\tSDO\t{SDO[0]!r}\t{SDO[1]!r}\t{SDO[2]!r}",
    ]
    for gas in sorted({case[0] for case in CASES.values()}):
        lines.append(f"V\tVIB-{gas}\t{gas}")
    for gas in ("H2", "N2", "D2"):
        lines.append(f"L\tLADDER-{gas}\t{gas}\t{TEMPERATURE!r}\t{MIN_J}\t{MAX_J}")
    for key, (gas, rotation, vibration) in CASES.items():
        for axis in TIME_AXES:
            times = ",".join(repr(value) for value in time_axis(axis))
            lines.append(
                f"H\t{key}-{axis}\t{gas}\t{int(rotation)}\t{int(vibration)}\t"
                f"{TEMPERATURE!r}\t{AMAGAT_MULTIPLE!r}\t{times}"
            )
    return lines


def decode(output: str) -> dict[str, list[float]]:
    """Parse the Julia program's output back into arrays.

    Parameters
    ----------
    output : str
        Standard output of the Julia program.

    Returns
    -------
    dict
        Emitted key to its list of values.

    Raises
    ------
    ValueError
        If a key is emitted twice, or nothing is emitted at all.
    """
    values: dict[str, list[float]] = {}
    for line in output.splitlines():
        if not line.startswith("R\t"):
            continue
        _, key, payload = line.split("\t")
        if key in values:
            msg = f"Luna emitted {key} twice"
            raise ValueError(msg)
        values[key] = [float(entry) for entry in payload.split(",")]
    if not values:
        msg = "Luna emitted nothing"
        raise ValueError(msg)
    return values


def expected_keys() -> set[str]:
    """Return every key the Julia program is expected to emit.

    Returns
    -------
    set of str
        Keys, for a completeness check before anything is written.
    """
    keys = {"CONSTANTS", "SDO", "SILICA_SCALE"}
    keys |= {f"VIB-{gas}" for gas, _, _ in CASES.values()}
    for gas in ("H2", "N2", "D2"):
        keys |= {f"LADDER-{gas}-{part}" for part in ("J", "EJ", "POP", "OMEGA", "K")}
    keys |= {f"{key}-{axis}" for key in CASES for axis in TIME_AXES}
    return keys


def main() -> int:
    """Run Luna and write the reference file.

    Returns
    -------
    int
        A process exit code.
    """
    result = subprocess.run(
        ["julia", "--startup-file=no", "-e", JULIA_PROGRAM],
        input="\n".join(build_query()) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    values = decode(result.stdout)
    missing = expected_keys() - set(values)
    if missing:
        sys.stderr.write(f"Luna returned nothing for {sorted(missing)}\n")
        return 1

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
            "temperature_k": TEMPERATURE,
            "amagat_multiple": AMAGAT_MULTIPLE,
            "min_j": MIN_J,
            "max_j": MAX_J,
            "normalised_oscillator": list(SDO),
            "note": (
                "Regression witnesses, not truth. Luna's h is solprop's times eps0 "
                "and its constants are CODATA 2014; the density is given in amagat so "
                "that the dephasing time is identical on both sides, and the "
                "responses are Luna's untapered scalar path. Generated by "
                "tests/data/generate_luna_raman.py; see that file and "
                "tests/nonlinear/test_luna_raman.py for the differences."
            ),
        },
        "cases": {
            key: {"gas": gas, "rotation": rotation, "vibration": vibration}
            for key, (gas, rotation, vibration) in CASES.items()
        },
        "time_axes": {name: time_axis(name) for name in TIME_AXES},
        "luna": values,
    }
    OUTPUT.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    written = payload["provenance"]["luna_commit"]
    size = OUTPUT.stat().st_size / 1024
    sys.stdout.write(f"wrote {OUTPUT} ({size:.0f} kB) from Luna {written}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
