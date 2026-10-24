# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Install the prototype's checksummed private Julia environment."""

import hashlib
import os
import platform
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

VERSION = "1.13.2"
# Digests published by Julia's official versions.json for these release archives.
RELEASES = {
    ("darwin", "arm64"): (
        "mac/aarch64",
        "macaarch64",
        "63309775ab65021036b0b4a069e82ae004a3f2b551fc10ceb40ddb2203200387",
    ),
    ("linux", "x86_64"): (
        "linux/x64",
        "linux-x86_64",
        "2ecdfc9890ab0a565fdd92404c915ba2a04a933dbd5d292cc224d13819d17d07",
    ),
}


def main() -> None:
    """Prepare the isolated runtime and instantiate the committed Julia manifest."""
    release = RELEASES.get((sys.platform, platform.machine()))
    if release is None:
        raise RuntimeError("bootstrap supports Apple Silicon and Linux x86_64")
    directory, suffix, expected_digest = release
    archive_name = f"julia-{VERSION}-{suffix}.tar.gz"
    url = f"https://julialang-s3.julialang.org/bin/{directory}/1.13/{archive_name}"
    root = Path(__file__).resolve().parents[1]
    private = root / ".julia-runtime"
    private.mkdir(exist_ok=True)
    executable = private / f"julia-{VERSION}/bin/julia"
    if not executable.is_file():
        archive = private / archive_name
        if not archive.is_file():
            urllib.request.urlretrieve(url, archive)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected_digest:
            raise RuntimeError("Julia archive SHA256 does not match the release")
        with tarfile.open(archive) as bundle:
            bundle.extractall(private, filter="data")
    env = os.environ | {
        "JULIA_DEPOT_PATH": str(root / ".julia-depot"),
        "JULIA_LOAD_PATH": "@:@stdlib",
        "JULIA_CONDAPKG_BACKEND": "Null",
        "JULIA_PYTHONCALL_EXE": sys.executable,
    }
    subprocess.run(
        [
            str(executable),
            "--startup-file=no",
            f"--project={root / 'julia'}",
            "-e",
            "using Pkg; Pkg.instantiate(); Pkg.precompile()",
        ],
        env=env,
        check=True,
    )
    sys.stdout.write(f"Prepared private Julia {VERSION} and julia/Manifest.toml\n")


if __name__ == "__main__":
    main()
