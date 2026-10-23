# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Generate the Vincetti fits' FEM regression witness from the pyJCMwave library.

Run with ``uv run --no-sync python tests/data/generate_jcmwave_vincetti.py``.

It reads two entries of the reference implementation pyJCMwave's library of
finite-element mode solutions, ``library/ARF_D30.0_dD0.680_t180_N6_vacuum`` and
``library/ARF_D30.0_dD0.680_t180_N6_Ar10bar`` under a pyJCMwave results
directory named by ``JCMWAVE_DATA`` (``meta.json`` and ``data.csv``: JCMsuite
solutions of the fundamental mode of a 30 um core, 180 nm wall, six-tube fibre,
evacuated and with argon at 10 bar and 293 K), and evaluates the run's own materials
through ``pyjcmwave/materials.py`` loaded by file path, so that neither the package's
JCMsuite setup nor its solver is imported: the tabulated silica index (``lookup``,
absorption off, as the metadata declares) and the argon index from its three-pole
fit and ideal-gas density. Nothing in pyJCMwave is written to. The rows are stored as
read; the geometry follows pyJCMwave's ``from_uebel`` convention (core diameter,
inner-to-core diameter ratio, wall thickness); the vacuum wall resonances that split
the rows into near and away sets, the 3 % convention of the FEM comparison report,
are the self-consistent solutions of ``lambda = 2 t sqrt(n_t(lambda)^2 - 1) / q``
on the same silica table. The SHA-256 of every file read is recorded. Stored FEM
results are a regression witness for a t/r = 0.017 fibre outside the fits' 0.05-0.2
validation range, not an accuracy claim.
"""

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("JCMWAVE_DATA", "pyJCMwave"))
ENTRIES = {
    "vacuum": "ARF_D30.0_dD0.680_t180_N6_vacuum",
    "argon": "ARF_D30.0_dD0.680_t180_N6_Ar10bar",
}
MATERIALS = ROOT / "pyjcmwave" / "materials.py"
SILICA_TABLES = [
    ROOT / "pyjcmwave" / "data" / name for name in ("silica_n.csv", "silica_k.csv")
]
OUTPUT = Path(__file__).with_name("jcmwave_vincetti.json")
COLUMNS = [
    "wavelength_m",
    "gas_index",
    "glass_index",
    "fem_neff_real",
    "fem_neff_imag",
    "fem_loss_db_per_m",
    "fem_area_m2",
]
RESONANCE_ORDERS = 2


def load_materials():
    """Load pyJCMwave's materials module by path, without its package setup."""
    sys.dont_write_bytecode = True
    specification = importlib.util.spec_from_file_location(
        "pyjcmwave_materials", MATERIALS
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def read_entry(name):
    """Return the metadata and the rows of one library entry, sorted by wavelength."""
    directory = ROOT / "library" / name
    metadata = json.loads((directory / "meta.json").read_text())
    with (directory / "data.csv").open(newline="") as handle:
        rows = sorted(
            csv.DictReader(handle), key=lambda row: float(row["wavelength_m"])
        )
    return metadata, rows


def geometry_of(metadata):
    """Return the Luna-convention geometry of an Uebel ``d/D`` entry."""
    core_radius = metadata["core_diameter"] / 2
    tube_radius = (
        metadata["d_over_D"] * metadata["core_diameter"] / 2
        + metadata["wall_thickness"]
    )
    n_tubes = metadata["n_resonators"]
    gap = 2 * ((core_radius + tube_radius) * math.sin(math.pi / n_tubes) - tube_radius)
    return {
        "core_radius_m": core_radius,
        "wall_thickness_m": metadata["wall_thickness"],
        "tube_radius_m": tube_radius,
        "n_tubes": n_tubes,
        "gap_m": gap,
    }


def resonance_wavelengths(materials, wall_thickness, silica_model):
    """Return the vacuum wall resonances ``q = 1 .. RESONANCE_ORDERS`` in m."""
    out = []
    for order in range(1, RESONANCE_ORDERS + 1):
        wavelength = 2 * wall_thickness * math.sqrt(1.45**2 - 1) / order
        for _ in range(100):
            index = float(
                np.real(materials.silica_index(wavelength, model=silica_model))
            )
            updated = 2 * wall_thickness * math.sqrt(index**2 - 1) / order
            if abs(updated - wavelength) < 1e-16:
                break
            wavelength = updated
        out.append(wavelength)
    return out


def main():
    """Write the FEM witness and its provenance."""
    materials = load_materials()
    fills = {}
    files = [MATERIALS, *SILICA_TABLES]
    geometry = None
    resonances = None
    for fill, name in ENTRIES.items():
        metadata, rows = read_entry(name)
        files += [
            ROOT / "library" / name / "meta.json",
            ROOT / "library" / name / "data.csv",
        ]
        wavelength = np.array([float(row["wavelength_m"]) for row in rows])
        assert len(np.unique(wavelength)) == len(wavelength)
        glass = np.real(
            materials.silica_index(wavelength, model=metadata["silica_model"])
        )
        if metadata["gas"] is None:
            gas = np.ones_like(wavelength)
        else:
            gas = materials.gas_index(
                metadata["gas"],
                wavelength,
                metadata["pressure_bar"],
                metadata["temperature"],
            )
        if geometry is None:
            geometry = geometry_of(metadata)
            resonances = resonance_wavelengths(
                materials, metadata["wall_thickness"], metadata["silica_model"]
            )
        assert geometry == geometry_of(metadata)
        table = np.column_stack(
            [
                wavelength,
                gas,
                glass,
                [float(row["neff_real"]) for row in rows],
                [float(row["neff_imag"]) for row in rows],
                [float(row["loss_db_per_m"]) for row in rows],
                [float(row["aeff_m2"]) for row in rows],
            ]
        )
        assert np.all(np.isfinite(table)) and np.all(table[:, 5] > 0)
        fills[fill] = {
            "entry": name,
            "metadata": metadata,
            "gas": metadata["gas"],
            "pressure_bar": metadata["pressure_bar"],
            "temperature_k": metadata["temperature"],
            "columns": COLUMNS,
            "rows": table.tolist(),
            "presets": [row["preset"] for row in rows],
        }
    OUTPUT.write_text(
        json.dumps(
            {
                "provenance": {
                    "generated": datetime.now(UTC).isoformat(),
                    "generator_sha256": hashlib.sha256(
                        Path(__file__).read_bytes()
                    ).hexdigest(),
                    "library_root": str(ROOT),
                    "source_sha256": {
                        str(path.relative_to(ROOT)): hashlib.sha256(
                            path.read_bytes()
                        ).hexdigest()
                        for path in files
                    },
                    "note": (
                        "JCMsuite finite-element solutions of the fundamental mode, "
                        "read from the reference implementation pyJCMwave's library; "
                        "a regression witness on a t/r = 0.017 fibre outside the "
                        "fits' validated wall ratio, not an accuracy claim. The "
                        "resonances split the rows by the report's descriptive 3 % "
                        "convention."
                    ),
                },
                "geometry": geometry,
                "resonance_wavelengths_m": resonances,
                "fills": fills,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
