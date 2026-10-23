# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run records stay what they were when output schema 21 was set (SEP 22 section 7).

Every field of a recorded dataclass is recorded, at its default too, so a change that
moves one of these records is deliberate and bumps the output schema. Each is pinned as
SEP 25 pins a record: its structure -- keys, strings, integers, flags and where every
number sits -- by an exact SHA-256 (identity metadata, class P), and its numbers, stored
beside this test, to ``RECORD_RTOL`` (``tests/record_numbers.py``). What an optmat
release writes into a record is neutralised first, as for the printed run record
(``tests/test_display.py``).
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from record_numbers import assert_record_numbers, record_skeleton, skeleton_hash
from solprop.boundaries import absorber, custom_absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid, RadialGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.output.schema import SCHEMA_VERSION, serialise
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import ApertureSet, StatsSpec
from solprop.stepper import PIController
from test_display import without_optmat_release

STORED = Path(__file__).with_suffix("")

SKELETONS = {
    "modal": "f81a10cfb4969e578ed09ac42255ab3003295475850c16f4204309a02e0d2db1",
    "radial": "fa2e78353328afe69b47f800782cecb3d660baef42b1fd630b7a4c7feac17236",
    "cartesian": "decacacf83d15b89d6b1497fe8fa889f160b1caba5a4b1def2ae2701536d6532",
    "cartesian_absorber": (
        "6a3b572aac3b8cb755e272dd89dafaa0e87dc6c4d80656059d434d5ab59f9430"
    ),
    "custom_absorber": (
        "6a3b572aac3b8cb755e272dd89dafaa0e87dc6c4d80656059d434d5ab59f9430"
    ),
}
"""The structure of each record at output schema 21."""


def spatial(kind: str) -> Simulation:
    """Return a small spatial run of the kind named, built from its public pieces."""
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 60e-6
    )
    boundary = {
        "radial": None,
        "cartesian": None,
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
        )
    return Simulation.from_free(
        grid,
        FreeGrid(3e-4, 8, 4e-4, 6, window_factor=0.2),
        SpatialMedium(),
        responses,
        pulse,
        settings=settings,
    )


def build(kind: str) -> Simulation:
    if kind == "modal":
        return Simulation.from_gnlse(
            0.11,
            0.01,
            [0.0, 0.0, -1.183e-26],
            lambda0=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=1e-12,
            width=100e-15,
            peak_power=9.8,
            pulseshape="sech",
            n_save=3,
            backend="numpy_ref",
        )
    return spatial(kind)


def split(kind: str):
    metadata = build(kind).metadata()
    assert metadata.schema_version == SCHEMA_VERSION == 22
    # SEP 27 changes the global resume boundary, not these non-plasma records.
    historical = dataclasses.replace(metadata, schema_version=21)
    return record_skeleton(without_optmat_release(historical.hashed()))


@pytest.mark.parametrize("kind", sorted(SKELETONS))
def test_a_record_of_schema_21_is_what_it_was(kind):
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
