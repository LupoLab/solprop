# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A point's noise seed on a simulation derived from, or shared with, other points."""

from functools import cache
from typing import cast

import h5py
import numpy as np
import pytest

from legacy_geometry import LEGACY_GRID, bare_absorber
from march_bounds import FIXED_ORDER, assert_points_matched
from solprop.driver import Simulation, SimulationSettings
from solprop.driver._scan_model import name_seed
from solprop.driver.executors import Batch, Local, Pool, pool
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import OutputError
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.output.hdf5 import _dataset
from solprop.physdata import Mixture
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, PIController, RungeKutta

# These short fixtures launch into tight windows; the boundary is not what is tested.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")

LENGTH = 0.002
"""Metres of capillary, in the few steps its collar's stability allows."""

FIBRE = 0.001
"""Metres of fibre, in four prescribed steps."""

ROUNDING = 8 * np.finfo(float).eps
"""Relative bound on one array divided by the root of one weight, computed two ways."""


@pytest.fixture(autouse=True)
def _controller_without_history(monkeypatch):
    """Size the pool from the scan, not from this test process's past."""
    monkeypatch.setattr(pool, "peak_rss", lambda: 0)


def capillary(**overrides):
    """Return the scans guide's base on the reference backend, a few steps long."""
    return Simulation.from_capillary(
        125e-6,
        LENGTH,
        Mixture.fill({"He": 1.0}, 1.0),
        wavelength=800e-9,
        wavelength_limits=(200e-9, 1.5e-6),
        trange=200e-15,
        fwhm=7.5e-15,
        energy=200e-6,
        modes=1,
        raman=False,
        plasma=False,
        thg=False,
        n_save=2,
        backend="numpy_ref",
        # Short enough for the temporal collar's explicit stability limit.
        integrator=RungeKutta(),
        schedule=FixedSchedule(np.full(40, LENGTH / 40)),
        **overrides,
    )


@cache
def base():
    return capillary()


def recipe(point):
    """Derive a point as the guide does: one base per process, a new fill and input."""
    mixture = Mixture.fill({"He": 1.0}, float(point.values["pressure"]))
    pulse = GaussianPulse(wavelength=800e-9, fwhm=7.5e-15, energy=200e-6)
    return base().with_mixture(mixture).with_input(pulse)


def fibre(**overrides):
    """Return a fibre whose batch members and solo runs are one discrete march."""
    # The former grid without a collar, which a short fixed schedule is refused beside.
    grid = EnvelopeGrid.from_limits(FIBRE, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID)
    return Simulation.from_gnlse(
        0.11,
        FIBRE,
        [0.0, 0.0, -1.183e-26],
        wavelength=835e-9,
        grid=grid,
        absorber=bare_absorber(grid),
        width=100e-15,
        peak_power=9.8,
        shape="sech",
        save_planes=[FIBRE / 2, FIBRE],
        backend="numpy_ref",
        integrator=FIXED_ORDER,
        schedule=FixedSchedule.uniform(
            0.0, FIBRE, FIBRE / 4, save_planes=[FIBRE / 2, FIBRE]
        ),
        **overrides,
    )


@cache
def one_object():
    return fibre()


def shared(point):
    """Return one object for every point, as a builder may."""
    return one_object()


def radial(medium=None, **overrides):
    """Return a small radial run in helium, by default at one bar."""
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    return Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 8, window_factor=0.0),
        SpatialMedium(Mixture.fill({"He": 1.0}, 1.0)) if medium is None else medium,
        [],
        SpatialPulse(GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-9), 80e-6),
        settings=SimulationSettings(
            controller=PIController(rtol=1e-3, atol=1e-6),
            save_planes=[1e-3],
            dz_max=1e-3,
            dz_init=1e-4,
            backend="numpy_ref",
        ),
        **overrides,
    )


def drawn(simulation):
    noise = simulation.noise
    assert noise is not None
    return noise


def recorded_seeds(path):
    with h5py.File(path) as file:
        return [int(seed) for seed in _dataset(file, "points/noise_seed").asstr()[...]]


def assert_same_noise(simulation, reference):
    """Assert a run carries the realisation and record of one built with its seed."""
    assert drawn(simulation).seed == drawn(reference).seed
    # One draw from one seed: a structural contract, not a computation.
    assert np.array_equal(drawn(simulation).spectrum, drawn(reference).spectrum)
    assert (
        simulation.metadata().configuration_hash()
        == reference.metadata().configuration_hash()
    )
    expected = reference.translate().initial.noise0
    actual = simulation.translate().initial.noise0
    assert np.max(np.abs(actual - expected)) <= ROUNDING * np.max(np.abs(expected))


TWO = [0.8, 1.2]
FOUR = [0.8, 1.0, 1.2, 1.4]


def test_the_recipe_records_a_seed_per_point_and_a_rerun_skips_them(tmp_path):
    scan = Scan("recipe", pressure=TWO)
    path = tmp_path / "recipe.h5"
    # In workers: each `base` there has its own operating-system draw, and this
    # process's has another, so the rerun below is a rerun in a new process.
    first = run_scan(
        scan, recipe, executor=Pool(workers=2), output=path, progress="none"
    )
    assert first.status.tolist() == ["done", "done"]
    assert recorded_seeds(path) == [name_seed("recipe", index) for index in range(2)]
    base.cache_clear()
    again = run_scan(scan, recipe, executor=Local(), output=path, progress="none")
    assert again.status.tolist() == ["done", "done"]


def test_the_recipe_seeds_do_not_depend_on_the_execution(tmp_path):
    scan = Scan("order", pressure=FOUR)
    local = tmp_path / "local.h5"
    run_scan(scan, recipe, executor=Local(), output=local, progress="none")
    seeds = recorded_seeds(local)
    assert len(set(seeds)) == 4
    in_pool = tmp_path / "pool.h5"
    run_scan(scan, recipe, executor=Pool(workers=2), output=in_pool, progress="none")
    assert recorded_seeds(in_pool) == seeds
    for k in range(2):
        chunk = scan.chunk(k, 2)
        out = tmp_path / f"chunk{k}.h5"
        run_scan(chunk, recipe, executor=Local(), output=out, progress="none")
        with h5py.File(out) as file:
            stored = _dataset(file, "points/noise_seed").asstr()[...]
            done = [one for one in stored if one]
        assert [int(one) for one in done] == [
            seeds[point.index] for point in chunk.points()
        ]


def test_the_recipe_takes_the_base_seed(tmp_path):
    scan = Scan("seeded", base_seed=42, pressure=TWO)
    path = tmp_path / "seeded.h5"
    result = run_scan(scan, recipe, executor=Local(), output=path, progress="none")
    assert result.status.tolist() == ["done", "done"]
    assert recorded_seeds(path) == [point.seed for point in scan.points()]


def test_one_object_under_batch_matches_local_member_for_member(tmp_path):
    scan = Scan("trials", trial=[0, 1, 2])
    one_object.cache_clear()
    reference = run_scan(
        scan,
        shared,
        executor=Local(),
        output=tmp_path / "local.h5",
        fields="all",
        progress="none",
    )
    one_object.cache_clear()
    result = run_scan(
        scan,
        shared,
        executor=Batch(3),
        output=tmp_path / "batch.h5",
        fields="all",
        progress="none",
    )
    assert result.status.tolist() == ["done"] * 3
    launches = [
        cast("np.ndarray", result.point(index)["fields/values"])[0]
        for index in range(3)
    ]
    assert not np.array_equal(launches[0], launches[1])
    assert_points_matched(result, reference, scan, shared, what="batch member")
    assert recorded_seeds(tmp_path / "batch.h5") == recorded_seeds(
        tmp_path / "local.h5"
    )


def test_a_scan_leaves_the_builders_object_as_it_is(tmp_path):
    one_object.cache_clear()
    built = one_object()
    noise, problem, metadata = built.noise, built.translate(), built.metadata()
    run_scan(
        Scan("left", trial=[0, 1]),
        shared,
        executor=Local(),
        output=tmp_path / "left.h5",
        progress="none",
    )
    assert built.noise is noise
    assert built.translate() is problem
    assert built.metadata() is metadata


def test_a_reseeded_copy_has_its_own_noise_log_and_shares_the_translation():
    built = fibre()
    copy = built._reseeded(7)
    assert copy is not built
    assert drawn(copy).seed == 7
    assert drawn(built).seed != 7
    assert copy._material_log is not built._material_log
    # Translated once, on the object the points share.
    assert copy.translate().linear is built.translate().linear
    assert copy._reseeded(7) is copy


def test_a_run_with_nothing_to_draw_is_returned_as_it_is():
    seeded = fibre(noise="input", seed=3)
    assert seeded._reseeded(7) is seeded
    quiet = fibre(noise=None)
    assert quiet._reseeded(7) is quiet


@pytest.mark.parametrize(
    "build", [fibre, capillary, radial], ids=["gnlse", "capillary", "radial"]
)
def test_a_reseed_replaces_only_the_noise_of_a_translation(build):
    simulation = build()
    original = simulation.translate()
    assert simulation._reseed(11)
    problem = simulation.translate()
    assert problem is not original
    assert problem.linear is original.linear
    assert problem.transform is original.transform
    assert problem.responses is original.responses
    assert problem.initial.field0 is original.initial.field0
    assert_same_noise(simulation, build(noise="input", seed=11))


def test_a_reseed_before_translation_translates_the_new_draw():
    simulation = fibre()
    assert simulation._reseed(11)
    assert_same_noise(simulation, fibre(noise="input", seed=11))


@pytest.mark.parametrize(
    "derive",
    [
        lambda one: one.with_input(one.initial),
        lambda one: one.with_mixture(Mixture.fill({"He": 1.0}, 1.0)),
    ],
    ids=["with_input", "with_mixture"],
)
def test_a_derived_run_carries_the_redraw_on_its_shared_translation(derive):
    parent = capillary()
    derived = derive(parent)
    assert drawn(derived).seed == drawn(parent).seed
    linear = derived.translate().linear
    assert derived._reseed(11)
    assert derived.translate().linear is linear
    assert drawn(derived).seed == 11
    assert drawn(parent).seed != 11
    assert_same_noise(derived, derive(capillary(noise="input", seed=11)))


def test_explicit_noise_and_no_noise_are_not_carried():
    parent = fibre()
    field = parent.translate().initial.field0[0]
    replaced = parent.with_input(field, noise=fibre(noise="input", seed=5).noise)
    assert not replaced._reseed(11)
    assert drawn(replaced).seed == 5
    removed = parent.with_input(field, noise=None)
    assert not removed._reseed(11)
    assert removed.noise is None


def test_a_seeded_base_stays_seeded_through_its_derivations():
    parent = capillary(noise="input", seed=5)
    derived = parent.with_mixture(Mixture.fill({"He": 1.0}, 1.2)).with_input(
        parent.initial
    )
    assert derived.noise is parent.noise
    assert not derived._reseed(11)


def test_a_seeded_radial_update_shares_its_realisation():
    parent = radial(noise="input", seed=5)
    updated = parent.with_mixture(Mixture.fill({"He": 1.0}, 2.0))
    assert updated.noise is parent.noise


def test_a_spatial_update_draws_on_the_new_mediums_flux():
    filled = Mixture.fill({"He": 1.0}, 2.0)
    parent = radial()
    updated = parent.with_mixture(filled)
    # At once and from the same seed: what it carries is what its redraw gives, or a
    # resume drawing from the file's seed would disagree with the file.
    assert drawn(updated).seed == drawn(parent).seed
    assert np.array_equal(
        drawn(updated).spectrum,
        drawn(
            radial(SpatialMedium(filled), noise="input", seed=drawn(parent).seed)
        ).spectrum,
    )
    assert not np.array_equal(drawn(updated).spectrum, drawn(parent).spectrum)
    assert updated._reseed(11)
    assert_same_noise(updated, radial(SpatialMedium(filled), noise="input", seed=11))


def test_a_derived_run_resumes_from_its_file_after_a_new_draw(tmp_path):
    path = tmp_path / "run.h5"
    field = fibre().translate().initial.field0[0]
    first = fibre().with_input(field)
    first.run(path, progress="none")
    seed = drawn(first).seed
    # A new draw: the same call in a new process.
    again = fibre().with_input(field)
    assert drawn(again).seed != seed
    again.run(path, resume=True, progress="none")
    assert drawn(again).seed == seed


def test_an_explicit_realisation_is_still_refused_by_a_resume(tmp_path):
    path = tmp_path / "run.h5"
    parent = fibre()
    field = parent.translate().initial.field0[0]
    parent.with_input(field).run(path, progress="none")
    other = parent.with_input(field, noise=fibre(noise="input", seed=5).noise)
    with pytest.raises(OutputError, match=r"noise\.seed"):
        other.run(path, resume=True, progress="none")
