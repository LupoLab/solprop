# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Copy Luna's chirped-mirror data files into the package, each behind a header.

Run with ``uv run python tests/data/copy_luna_mirror_data.py``. The Luna checkout is
read only: the files are read, never touched. Every number is copied as it stands; the
only changes are a ``#`` provenance header, the removal of a byte-order mark and the
normalisation of line endings. Where the data came from is **not recorded in Luna**
(``PhysData.jl`` ``lookup_mirror`` carries column comments and nothing else), and the
headers say so rather than guess. The tables' origin, units and columns have not been
independently confirmed.
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

LUNA = Path(os.environ.get("LUNA_DIR", Path.home() / ".julia" / "dev" / "Luna"))
SOURCE = LUNA / "src/data"
OUTPUT = Path(__file__).parents[2] / "src/solprop/physdata/data/mirrors"

FILES = {
    "PC70_R.csv": (
        "PC70",
        "wavelength (nm), power reflectivity in percent at 5 and at 19 degrees, "
        "p-polarised by the column names",
    ),
    "PC70_GDD.csv": (
        "PC70",
        "wavelength (nm), group-delay dispersion in fs^2 at 5 and at 19 degrees; Luna "
        "uses the mean of the two as the GDD per reflection",
    ),
    "HD59_GDD.dat": (
        "HD59",
        "wavelength (nm), group-delay dispersion in fs^2; the two header lines read "
        "'Simulated' and 'actual'; there is no reflectivity data and Luna takes unit "
        "amplitude",
    ),
    "PC147.txt": (
        "PC147",
        "sixteen tab-separated columns; Luna reads the first four: wavelength (nm), "
        "'Rp sum' power reflectivity in percent, wavelength, 'GDDrp sum' in fs^2, "
        "commented there as the average per mirror of a complementary pair",
    ),
    "PC1611.txt": (
        "PC1611",
        "wavelength (nm), 'Rp 7 deg' power reflectivity in percent, wavelength, "
        "'GDDrp 7 deg' in fs^2",
    ),
    "PC1821.txt": (
        "PC1821",
        "wavelength (nm), 'Rp' power reflectivity in percent, wavelength, 'GDDrs' in "
        "fs^2: a p-polarised reflectivity beside an s-polarised GDD, by the column "
        "names",
    ),
    "HD120.csv": (
        "HD120",
        "wavelength (nm), power reflectivity in percent (P-Pol, 5 deg), wavelength, "
        "GDD in fs^2 (P-Pol, 5 deg)",
    ),
    "UCxx-15FS_R.csv": (
        "ThorlabsUMC",
        "wavelength (nm), power reflectivity in percent for p and s polarisation; "
        "Luna uses s",
    ),
    "UCxx-15FS_GD.csv": (
        "ThorlabsUMC",
        "wavelength (nm), group DELAY in fs for p and s polarisation (Luna's comment "
        "says fs^2; its code integrates once, as for a delay); Luna uses s",
    ),
}


def luna_commit() -> str:
    """Return the commit of the Luna checkout, read without changing it."""
    result = subprocess.run(
        ["git", "-C", str(LUNA), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    """Write the nine files with their headers."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    commit = luna_commit()
    for name, (mirror, columns) in FILES.items():
        raw = (SOURCE / name).read_bytes()
        text = raw.decode("utf-8-sig").replace("\r\n", "\n")
        if not text.endswith("\n"):
            text += "\n"
        header = [
            f"# mirror: {mirror} (solprop.optics.ChirpedMirror.lookup)",
            f"# source: Luna src/data/{name} at commit {commit}",
            f"# source-sha256: {hashlib.sha256(raw).hexdigest()}",
            f"# columns: {columns}",
            "# origin: NOT RECORDED in Luna: no vendor, part number, data sheet, "
            "measurement or citation is named in PhysData.jl or beside the file",
            "# changes: this header added; byte-order mark removed; line endings "
            "normalised; every number as in the source",
            "# generated-by: tests/data/copy_luna_mirror_data.py",
        ]
        (OUTPUT / name).write_text("\n".join(header) + "\n" + text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
