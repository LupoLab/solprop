# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run records stay what they were when they were pinned (SEP 22 section 7, SEP 42).

Every field of a recorded dataclass is recorded, at its default too, so a change that
moves one of these records is deliberate. Two pins hold them.

**Frozen records** (SEP 42 section 7): each case's whole record, as a release wrote it,
is stored beside this test and never regenerated after that release is tagged. Each must
compare as continuing with the current build (:meth:`RunMetadata.compare`), on every CI
platform, which is the cross-platform check of the resume comparison; and a recorded
field a later change adds or removes fails until it has a neutral value
(:data:`solprop.output.compatibility.NEUTRAL_VALUES`), which is what enforces that rule.
An optmat data revision moves a model's digest and the arrays built from its data; a
case whose digests moved is then compared by structure alone, and logs them. A case
with no optmat data, such as the GNLSE one, is always compared in full. Each release's
set is a folder of ``frozen/``. The first is 0.2.0's: until that release is tagged it
is refreshed with :func:`write_frozen`, and afterwards each release adds its own folder
from a checkout of its tag (``docs/dev/releasing.md``). A folder goes, with the neutral
values that name its release, once the next minor release after it is out (ADR-063).

**Pinned skeletons.** The records were
first pinned before five later changes, which the projection in :func:`split` takes
back out, so that the pins still hold everything else. Each is pinned as
SEP 25 pins a record: its structure -- keys, strings, integers, flags and where every
number sits -- by an exact SHA-256 (identity metadata, class P), and its numbers, stored
beside this test, to ``RECORD_RTOL`` (``tests/record_numbers.py``). What an optmat
release writes into a record is neutralised first, as for the printed run record
(``tests/test_display.py``).
"""

import dataclasses
import gzip
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from modal_provider_cases import general_grid, general_guide
from record_numbers import (
    assert_record_numbers,
    record_skeleton,
    skeleton_hash,
    without_sep21_absorber,
)
from solprop.boundaries import absorber, custom_absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.output import compatibility
from solprop.output.schema import SCHEMA_VERSION, RunMetadata, serialise
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import ApertureSet, StatsSpec
from solprop.stepper import Multistep, PIController
from test_display import without_optmat_release

STORED = Path(__file__).with_suffix("")
FROZEN = STORED / "frozen"
FROZEN_RELEASES = tuple(sorted(path.name for path in FROZEN.iterdir() if path.is_dir()))
"""The releases whose records are frozen: one folder each."""

FROZEN_CASES = (
    "capillary",
    "cartesian",
    "cartesian_absorber",
    "custom_absorber",
    "modal",
    "radial",
)
"""The cases whose whole records are frozen: the pinned ones and an optmat capillary."""

logger = logging.getLogger(__name__)

SKELETONS = {
    "modal": "77a845cabf424c75bce9080a2ce2464e2d399b8709502303d842733d411ee36a",
    "radial": "747f15eb378a23d77da9807c04141fdd8a0b803e1dadf2f38f540ce78e2d171b",
    "cartesian": "c568de28b432c181884e8dffa90961a1381452e755d33ff7af3e709f419afd46",
    "cartesian_absorber": (
        "ff33b91195c081c33e5e3bcfd6d0541790da91f62e69c582d6e3482ab6ec57c3"
    ),
    "custom_absorber": (
        "ff33b91195c081c33e5e3bcfd6d0541790da91f62e69c582d6e3482ab6ec57c3"
    ),
}
"""The structure of each record, as :func:`split` projects it."""


def spatial(kind: str) -> Simulation:
    """Return a small spatial run of the kind named, built from its public pieces."""
    grid = EnvelopeGrid.from_limits(
        1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False, **LEGACY_GRID
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 60e-6
    )
    boundary = {
        "radial": bare_absorber(grid),
        "cartesian": bare_absorber(grid),
        "cartesian_absorber": absorber(grid),
        "custom_absorber": custom_absorber(grid, 0.0),
    }[kind]
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[5e-4, 1e-3],
        dz_max=1e-3,
        dz_init=1e-4,
        absorber=boundary,
        backend="numpy_ref",
        # The records were taken on the Adams march, the free-space default before
        # SEP 37 made it the Runge-Kutta one.
        integrator=Multistep(),
    )
    responses = [KerrResponse(1e-22, thg=False)]
    if kind == "radial":
        return Simulation.from_radial(
            grid,
            RadialGrid(300e-6, 16, window_factor=0.2),
            SpatialMedium(),
            responses,
            pulse,
            settings=settings,
            noise=None,
        )
    return Simulation.from_free(
        grid,
        FreeGrid(3e-4, 8, 4e-4, 6, window_factor=0.2),
        SpatialMedium(),
        responses,
        pulse,
        settings=settings,
        noise=None,
    )


def capillary() -> Simulation:
    """Return a short argon capillary with plasma: optmat models and a quadrature."""
    grid = FieldGrid.from_limits(4e-4, 800e-9, (300e-9, 3e-6), 100e-15)
    return Simulation.from_capillary(
        50e-6,
        4e-4,
        "Ar",
        1.0,
        wavelength=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0),
        fwhm=30e-15,
        energy=1e-6,
        n_r=8,
        n_save=3,
        backend="numpy_ref",
        plasma=True,
        noise=None,
        rtol=1e-6,
    )


def build(kind: str) -> Simulation:
    if kind == "capillary":
        return capillary()
    if kind == "modal":
        return Simulation.from_gnlse(
            0.11,
            0.01,
            [0.0, 0.0, -1.183e-26],
            wavelength=835e-9,
            grid=EnvelopeGrid.from_limits(
                0.01, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
            ),
            width=100e-15,
            peak_power=9.8,
            shape="sech",
            n_save=3,
            backend="numpy_ref",
            absorber=False,
            noise=None,
            # The record was taken under the GNLSE defaults before SEP 37.
            integrator=Multistep(),
            rtol=1e-8,
        )
    return spatial(kind)


def split(kind: str):
    metadata = build(kind).metadata()
    assert metadata.schema_version == SCHEMA_VERSION == 2
    # The projection works on the hashed record, since a record without the later
    # additions is not one a run can hold.
    hashed: dict[str, Any] = metadata.hashed()
    # Output schema 2 adds a field only to `DataPulse`, which none of these runs holds;
    # the projection takes the number back, and everything else must be unchanged.
    hashed["schema_version"] = 1
    # SEP 34 adds the plane measure to the spatial record; the projection leaves it
    # out, and everything else must be unchanged.
    if hashed["spatial"] is not None:
        hashed["spatial"] = {
            key: value
            for key, value in hashed["spatial"].items()
            if key not in ("flux_model", "flux_plan")
        }
    # SEP 33 records every run's precision in its configuration; the projection
    # leaves it out too.
    hashed["configuration"] = {
        key: value
        for key, value in hashed["configuration"].items()
        if key != "precision"
    }
    # SEP 35 names the carrier frame in every Multistep's representation, at its
    # default too (no output layout change); the projection leaves the default out.
    method = hashed["stepper"]["method"]
    assert isinstance(method, str)
    hashed["stepper"] = {
        **hashed["stepper"],
        "method": method.replace(", carrier_frame='off'", ""),
    }
    # SEP 21 records the absorber's spatial flag, and the runs that declared none
    # record the bare boundaries that stand for it; the projection takes both back.
    hashed = without_sep21_absorber(
        hashed,
        bare=kind in ("modal", "radial", "cartesian"),
        spatial=kind != "modal",
    )
    return record_skeleton(without_optmat_release(hashed))


@pytest.mark.parametrize("kind", sorted(SKELETONS))
def test_a_pinned_record_is_what_it_was(kind):
    skeleton, numbers, floors = split(kind)
    assert skeleton_hash(skeleton) == SKELETONS[kind], kind
    assert_record_numbers(numbers, floors, STORED / f"{kind}.npz")


def test_a_custom_absorber_at_its_defaults_records_the_collar():
    # A zero user rate returns the collar field for field (SEP 22 section 5.4): the
    # two runs are one run, with one hash.
    assert (
        build("custom_absorber").metadata().configuration_hash()
        == build("cartesian_absorber").metadata().configuration_hash()
    )


def test_the_far_field_tables_enter_the_record_at_their_defaults_too():
    # SEP 23 section 9: a spec records both tables, empty ones included; an empty
    # region table is the default and records the same, and one with tables differs.
    plain = serialise(StatsSpec(names=("energy",)))
    assert isinstance(plain, dict)
    assert plain["apertures"] is None and "spectral_regions" in plain
    assert (
        serialise(StatsSpec(names=("energy",), spectral_regions=np.empty((0, 4))))
        == plain
    )
    holes = ApertureSet(centres=[[0.0, 0.0]], diameters=[1e-3], distances=[0.25])
    tables = serialise(
        StatsSpec(
            names=("energy", "aperture_spectrum"),
            apertures=holes,
            spectral_regions=np.array([[0.0, 1.0, -1.0, 1.0]]),
        )
    )
    assert isinstance(tables, dict)
    assert "apertures" in tables and "spectral_regions" in tables
    digests = []
    for spec in (
        StatsSpec(names=("energy",)),
        StatsSpec(names=("energy",), spectral_regions=np.empty((0, 4))),
        StatsSpec(names=("energy",), apertures=holes),
    ):
        run = spatial("cartesian")
        run.settings = dataclasses.replace(run.settings, stats=spec)
        digests.append(run.metadata().configuration_hash())
    assert digests[0] == digests[1] != digests[2]


def write_stored() -> None:  # pragma: no cover - run by hand to set the records
    """Print the skeleton hashes and write the numbers, for a deliberate change."""
    STORED.mkdir(exist_ok=True)
    for kind in sorted(SKELETONS):
        skeleton, numbers, _ = split(kind)
        np.savez(STORED / f"{kind}.npz", numbers=np.asarray(numbers, dtype=float))
        print(f'    "{kind}": "{skeleton_hash(skeleton)}",')  # noqa: T201


# ------------------------------------------------------------------ frozen records


def frozen(release: str, kind: str) -> RunMetadata:
    """Return the frozen record of one case, as the release wrote it."""
    path = FROZEN / release / f"{kind}.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stored:
        return RunMetadata.from_json(stored.read())


def digests(metadata: RunMetadata) -> set[str]:
    """Return the digests of the optmat models a record names."""
    materials = metadata.hashed()["materials"]
    if not isinstance(materials, dict):
        return set()
    records = materials.get("records")
    assert isinstance(records, dict)
    return {
        str(cast("dict[str, object]", record)["digest"]) for record in records.values()
    }


@pytest.mark.parametrize("kind", FROZEN_CASES)
@pytest.mark.parametrize("release", FROZEN_RELEASES)
def test_a_frozen_record_continues_with_this_build(release, kind):
    stored = frozen(release, kind)
    current = build(kind).metadata()
    comparison = stored.compare(current)
    changes = (*comparison.layout, *comparison.refused, *comparison.numerics)
    absent = [compatibility.dotted(change.path) for change in changes if change.absent]
    assert not absent, (
        f"{kind}: these recorded paths are new or gone; each needs a neutral value in "
        f"solprop.output.compatibility.NEUTRAL_VALUES, or a frozen record refreshed "
        f"before the release is tagged: {absent}"
    )
    moved = digests(stored) ^ digests(current)
    if moved:
        # An optmat data revision moves the arrays built from its data too: the
        # structure must still agree, and the numbers are optmat's business.
        logger.warning("%s: optmat digests moved: %s", kind, sorted(moved))
        assert not comparison.layout, comparison.summary()
        assert all(
            not isinstance(change.stored, tuple) or change.stored == change.requested
            for change in changes
        ), comparison.summary()
        return
    assert comparison.continues, f"{release} {kind}: {comparison.summary()}"


def route_records() -> list[dict[str, Any]]:
    """Return the hashed records of every route the table test reads."""
    records = [build(kind).metadata().hashed() for kind in FROZEN_CASES]
    gnlse = Simulation.from_gnlse(
        0.11,
        0.01,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=1e-12,
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        n_save=2,
        backend="numpy_ref",
    )
    records.append(gnlse.metadata().hashed())
    grid = general_grid()
    general = Simulation.from_waveguide(
        (),
        1e-4,
        provider=general_guide(grid=grid),
        grid=grid,
        fwhm=30e-15,
        energy=1e-9,
        wavelength=800e-9,
        n_save=2,
        backend="numpy_ref",
        noise=None,
    )
    records.append(general.metadata().hashed())
    return records


def test_every_row_of_the_class_table_matches_a_recorded_path():
    # The table is data, reviewed like code: a row no route records is stale.
    paths = [
        path
        for record in route_records()
        for path in compatibility.record_paths(record)
    ]
    stale = [
        row.pattern
        for row in compatibility.CLASS_TABLE
        if not any(row.covers(path) for path in paths)
    ]
    assert not stale, stale


def write_frozen(release: str) -> None:  # pragma: no cover - run by hand at a release
    """Write each case's whole record, as this build writes it, under `release`.

    Run from a checkout of the release's tag (``docs/dev/releasing.md``); 0.2.0's set is
    refreshed from the tree until that release is tagged.
    """
    folder = FROZEN / release
    folder.mkdir(parents=True, exist_ok=True)
    for kind in FROZEN_CASES:
        with gzip.open(folder / f"{kind}.json.gz", "wt", encoding="utf-8") as out:
            out.write(build(kind).metadata().to_json())
