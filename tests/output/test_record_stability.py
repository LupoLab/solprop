# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run records stay what they were when they were pinned (SEP 22 section 7).

Every field of a recorded dataclass is recorded, at its default too, so a change that
moves one of these records is deliberate and bumps the output schema. The records were
first pinned before four later additions, which the projection in :func:`split` takes
back out, so that the pins still hold everything else. Each is pinned as
SEP 25 pins a record: its structure -- keys, strings, integers, flags and where every
number sits -- by an exact SHA-256 (identity metadata, class P), and its numbers, stored
beside this test, to ``RECORD_RTOL`` (``tests/record_numbers.py``). What an optmat
release writes into a record is neutralised first, as for the printed run record
(``tests/test_display.py``).
"""

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from record_numbers import (
    assert_record_numbers,
    record_skeleton,
    skeleton_hash,
    without_sep21_absorber,
)
from solprop.boundaries import absorber, custom_absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.output.schema import SCHEMA_VERSION, serialise
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import ApertureSet, StatsSpec
from solprop.stepper import Multistep, PIController
from test_display import without_optmat_release

STORED = Path(__file__).with_suffix("")

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


def build(kind: str) -> Simulation:
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
    assert metadata.schema_version == SCHEMA_VERSION == 1
    # The projection works on the hashed record, since a record without the later
    # additions is not one a run can hold.
    hashed: dict[str, Any] = metadata.hashed()
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
