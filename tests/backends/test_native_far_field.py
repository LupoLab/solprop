# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""native CPU's native far-field pass (SEP 23 section 6; section 7 items 3, 10, 11).

The cases of ``test_numba_far_field_kernels.py`` on the native backend: the pass
against ``numpy_ref``'s :func:`solprop.spatial.aperture_statistics` and against
``numba_cpu``'s pass, on field and envelope grids, one and two polarisations, a plane
and a line of three blocks, every profile and weighting; the ``W = 1`` identities;
hard-edge membership at an edge; thread counts 1, 2 and 4 (one process each, since
the runtime reads its thread count once); the native ledger against the estimate, to
the byte; a run through the driver; and the refusals. Every comparison of two separate
computations is held to the rounding bound of that module, fixed before measuring
(SEP 25): the native pass sums each band's columns, and then the bands, in the
``numba_cpu`` pass's order, but its compiler may contract and vectorise otherwise.
"""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from backends.matched import native_thread_fields
from backends.test_numba_far_field_kernels import (
    DISTANCE,
    FINE,
    WITH_AXIS,
    assert_matches,
    bound_for,
    fine_space,
    holes,
    magnitudes,
    random_field,
    reference,
    spec_of,
    time_grid,
)
from bounds import assert_within
from solprop.backends import select
from solprop.backends.memory import estimate_memory, far_field_memory
from solprop.backends.numba_cpu.spatial import FarFieldPass
from solprop.backends.spatial import AXIS_SPECTRUM_ONLY, far_field_partials
from solprop.boundaries import TransverseCollar, absorber
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError
from solprop.grid import FreeGrid, LineGrid
from solprop.output import MemoryOutput
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import FAR_FIELD_STATISTICS, SpatialMedium, mask_plane_radius
from solprop.stats import ApertureSet, StatsSpec
from solprop.stepper import PIController, RungeKutta
from solprop.translate.spatial import aperture_tables

solprop_native = pytest.importorskip("solprop_native")
NativeBackend = solprop_native.NativeBackend

PROBE = Path(__file__).with_name("native_far_field_thread_probe.py")


def problem_of(grid, space, field, spec, *, medium=None, collar=None):
    """The kernel problem of ``test_numba_far_field_kernels.problem_of`` for Kokkos."""
    simulation = Simulation.from_free(
        grid,
        space,
        SpatialMedium(1 + 1e-6j) if medium is None else medium,
        [],
        field,
        settings=SimulationSettings(
            controller=PIController(rtol=1e-6, atol=1e-14),
            save_planes=[1e-4],
            dz_max=1e-4,
            dz_init=1e-4,
            absorber=absorber(grid, temporal=False),
            backend="native",
            integrator=RungeKutta(),
            stats=spec,
            transverse_absorber=collar,
        ),
    )
    return simulation.translate()


def held_state(owner, problem):
    """The state a native backend holds, one member, copied to the host."""
    native = owner._native_state
    return native.field_to_host().reshape(problem.shape)[0]


def native_stats(problem, grid, field, names):
    """The native backend's statistics of `field`, without the batch axis.

    Returns them with the state the backend holds, which loading masks to the
    problem's band: the state every other backend is compared on.
    """
    owner = NativeBackend(problem, grid=grid)
    try:
        owner.set_initial(np.ascontiguousarray(field[None]), None)
        spec = dataclasses.replace(problem.stats_spec, names=tuple(names), periodic=())
        values = owner.stats(spec)
        held = held_state(owner, problem)
    finally:
        owner.close()
    return {name: value[0] for name, value in values.items()}, held


def case(kind, n_pol, profile, weighting):
    """One parity case: its grid, problem, field and tables."""
    grid = time_grid(kind)
    space = fine_space()
    spec = spec_of(holes(profile, weighting, size=FINE))
    field = random_field(space, n_pol, grid)
    problem = problem_of(
        grid, space, field, spec, collar=TransverseCollar(4e3, fraction=0.5)
    )
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    return grid, problem, field, tables


CASES = {
    "envelope tanh": ("envelope", 1, "tanh", "amplitude"),
    "envelope rcos quadrature": ("envelope", 2, "rcos", "quadrature"),
    "field supergauss": ("field", 1, "supergauss", "amplitude"),
    "field hard": ("field", 2, "hard", "amplitude"),
}


# ------------------------------------------------------- item 10: three backends


@pytest.mark.parametrize("name", list(CASES))
def test_the_native_pass_matches_the_reference_and_numba_cpu(name):
    grid, problem, field, tables = case(*CASES[name])
    values, field = native_stats(problem, grid, field, WITH_AXIS)
    assert set(values) == set(WITH_AXIS)
    absolute = magnitudes(WITH_AXIS, grid, problem, field, tables)
    assert_matches(
        values, reference(WITH_AXIS, grid, problem, field, tables), absolute, problem
    )
    # numba_cpu's pass is within the same bound of the exact sums, so the two are
    # within twice it of each other.
    numba, _ = FarFieldPass(problem, parallel=False)(WITH_AXIS, grid, field, tables)
    for one in WITH_AXIS:
        assert_within(
            values[one],
            numba[one],
            2 * bound_for(absolute, problem, one),
            elementwise=True,
            what=one,
        )
    # Every channel but the evanescent one is live on this fixture (an evanescent
    # column carries no axial flux, SEP 34), and every aperture collects.
    assert np.all(values["absorbed_power"][[0, 1, 3]] > 0)
    assert values["absorbed_power"][2] == 0.0
    assert np.all(np.max(values["aperture_spectrum"], axis=1) > 0)


def test_the_native_pass_matches_the_reference_on_a_line():
    grid = time_grid("envelope")
    space = LineGrid(4e-4, 130, window_factor=0)  # three blocks of columns
    apertures = ApertureSet(
        centres=[[0.0, 0.0], [1.2e-3, 0.0]],
        diameters=[1e-3, 6e-4],
        distances=[DISTANCE, 0.5],
        edges=[5e-5, 5e-5],
        profile="rcos",
        weighting="quadrature",
    )
    regions = np.array([[-np.inf, 0.0, -1.0, 1.0], [0.0, 2e4, -np.inf, np.inf]])
    spec = spec_of(apertures, regions=regions)
    field = random_field(space, 1, grid, seed=11)
    problem = problem_of(grid, space, field, spec)
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    assert far_field_partials(problem)[1] == 3
    values, field = native_stats(problem, grid, field, WITH_AXIS)
    assert_matches(
        values,
        reference(WITH_AXIS, grid, problem, field, tables),
        magnitudes(WITH_AXIS, grid, problem, field, tables),
        problem,
    )


def test_the_native_operator_slots_follow_gain_and_zero_frequency():
    grid = time_grid("field")
    space = FreeGrid(3e-4, 8, 3e-4, 8, window_factor=0)
    spec = spec_of(holes("hard"))
    field = random_field(space, 1, grid)
    field[..., 0] = 1e3  # content at zero frequency
    problem = problem_of(grid, space, field, spec, medium=SpatialMedium(1 - 1e-5j))
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    values, field = native_stats(problem, grid, field, WITH_AXIS)
    expected = reference(WITH_AXIS, grid, problem, field, tables)
    assert values["absorbed_power"][3] < 0
    assert values["beam_spectrum"][0] == 0.0
    assert np.all(values["aperture_spectrum"][:, 0] == 0.0)
    assert np.all(values["aperture_on_axis_spectrum"][:, 0] == 0.0)
    absolute = magnitudes(WITH_AXIS, grid, problem, field, tables)
    names = tuple(name for name in WITH_AXIS if name != "absorbed_power")
    assert_matches(values, expected, absolute, problem, names)
    # The material channel's terms are signed: its bound is that of the magnitudes.
    assert_within(
        values["absorbed_power"],
        expected["absorbed_power"],
        np.max(bound_for(absolute, problem, "absorbed_power")),
        elementwise=True,
    )


def test_a_partial_native_request_packs_only_its_statistics():
    grid = time_grid("envelope")
    space = fine_space()
    names = ("beam_spectrum", "aperture_on_axis_spectrum")
    spec = spec_of(holes("supergauss", size=FINE), names, regions=np.empty((0, 4)))
    field = random_field(space, 1, grid)
    problem = problem_of(grid, space, field, spec)
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    values, field = native_stats(problem, grid, field, names)
    assert set(values) == set(names)
    assert_matches(
        values,
        reference(names, grid, problem, field, tables),
        magnitudes(names, grid, problem, field, tables),
        problem,
        names,
    )


# ----------------------------------------------------------------- item 3: W = 1


def test_a_native_hole_enclosing_every_image_is_the_beam_and_the_axis():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 3e-4, 12, window_factor=0)
    k_max = np.hypot(np.max(np.abs(space.k_x)), np.max(np.abs(space.k_y)))
    reach = DISTANCE * 299792458.0 * k_max / float(np.min(grid.omega))
    apertures = ApertureSet(
        centres=[[0.0, 0.0]], diameters=[4 * reach], distances=[DISTANCE]
    )
    spec = StatsSpec(
        names=("aperture_spectrum", "aperture_on_axis_spectrum", "beam_spectrum"),
        apertures=apertures,
    )
    field = random_field(space, 2, grid)
    problem = problem_of(grid, space, field, spec)
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    names = (*spec.names, "on_axis_spectrum")
    values, field = native_stats(problem, grid, field, names)
    absolute = magnitudes(names, grid, problem, field, tables)
    assert_within(
        values["aperture_spectrum"][0],
        values["beam_spectrum"],
        bound_for(absolute, problem, "beam_spectrum"),
        elementwise=True,
    )
    assert_within(
        values["aperture_on_axis_spectrum"][0],
        values["on_axis_spectrum"],
        bound_for(absolute, problem, "on_axis_spectrum"),
        elementwise=True,
    )


def test_a_column_at_a_hard_edge_is_decided_natively_as_the_reference_decides():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 3e-4, 10, window_factor=0)
    column, bin_ = 29, grid.n_omega // 2
    tables = aperture_tables(
        spec_of(holes("hard"), ("aperture_spectrum",)), grid, space
    )
    assert tables is not None
    centre = (2e-4, -1e-4)
    radius = float(
        mask_plane_radius(
            tables.scale[0, bin_ : bin_ + 1],
            tables.k_x[column % space.n_x : column % space.n_x + 1],
            tables.k_y[column // space.n_x : column // space.n_x + 1],
            centre,
        )[0, 0]
    )
    field = np.zeros((space.n_columns, 1, grid.n_omega), complex)
    field[column, 0, bin_] = 1e3
    for diameter in (
        2 * radius,
        np.nextafter(2 * radius, 0),
        np.nextafter(2 * radius, 1),
    ):
        apertures = ApertureSet(
            centres=[list(centre)], diameters=[diameter], distances=[DISTANCE]
        )
        spec = StatsSpec(names=("aperture_spectrum",), apertures=apertures)
        problem = problem_of(grid, space, field, spec)
        values, field = native_stats(problem, grid, field, spec.names)
        inside = diameter / 2 >= radius
        # Structural: the membership is exact, so the one term is in or out.
        assert (values["aperture_spectrum"][0, bin_] > 0) == inside


# ------------------------------------------------ item 11: thread counts, bands


def thread_case():
    grid = time_grid("envelope")
    space = fine_space()
    spec = spec_of(holes("tanh", shared=True, size=FINE))
    field = random_field(space, 2, grid, seed=5)
    problem = problem_of(grid, space, field, spec)
    return grid, problem, field


def test_the_native_pass_does_not_depend_on_the_thread_count(tmp_path):
    grid, problem, field = thread_case()
    tables = aperture_tables(
        problem.stats_spec, problem.grids, problem.transform.geometry
    )
    _, field = native_stats(problem, grid, field, ("beam_spectrum",))
    absolute = magnitudes(WITH_AXIS, grid, problem, field, tables)
    results = {
        threads: native_thread_fields(PROBE, threads, WITH_AXIS, tmp_path)
        for threads in (1, 2, 4)
    }
    for threads in (2, 4):
        # The bands and every sum's order are fixed by the shapes: within the
        # rounding bound (not asserted to the bit).
        assert_matches(results[threads], results[1], absolute, problem)
    assert_matches(
        results[1],
        reference(WITH_AXIS, grid, problem, field, tables),
        absolute,
        problem,
    )


def test_the_native_partials_are_held_and_estimated_to_the_byte():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 2.5e-4, 10, window_factor=0.1)
    spec = spec_of(holes("tanh"), WITH_AXIS)
    field = random_field(space, 2, grid)
    problem = problem_of(grid, space, field, spec)
    n_slots, n_bands = far_field_partials(problem)
    rows = far_field_memory(problem, "native")
    assert rows == {"far_field_partials": 8 * n_bands * n_slots * grid.n_omega}
    owner = select("native", problem, grid=grid)
    try:
        ledger = owner.memory_ledger()
        assert ledger["far_field_partials"] == rows["far_field_partials"]
        # The axes, then the tables and packed output once a call has run.
        assert ledger["far_field_tables"] == 8 * (space.n_x + space.n_y)
        owner.set_initial(problem.initial.field0, None)
        owner.stats(problem.stats_spec)
        # Three apertures' rows and four boxes; twelve spectral rows (the beam, four
        # regions, three collected, three through the holes, the axis) and the four
        # absorbed powers.
        tables = 3 * (6 + grid.n_omega) + 4 * 4
        packed = 12 * grid.n_omega + 4
        ledger = owner.memory_ledger()
        assert ledger["far_field_tables"] == 8 * (
            space.n_x + space.n_y + tables + packed
        )
        assert ledger["far_field_partials"] == rows["far_field_partials"]
    finally:
        owner.close()
    assert (
        estimate_memory(problem, "native").groups["base"]
        == estimate_memory(problem, "numpy_ref").groups["base"]
        + rows["far_field_partials"]
    )


def test_on_axis_intensity_takes_no_native_slot():
    # Kokkos forms on_axis_intensity in its own spatial pass, so its partials, and the
    # estimate's row, leave the W = 1 slot out; numba_cpu's pass reads that slot.
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 12, 2.5e-4, 10, window_factor=0.1)
    field = random_field(space, 2, grid)
    spec = StatsSpec(names=("beam_spectrum", "on_axis_intensity"))
    problem = problem_of(grid, space, field, spec)
    n_slots, n_bands = far_field_partials(problem, axis_folded=AXIS_SPECTRUM_ONLY)
    assert n_slots == 1
    # The axis amplitude and its closure, 4 n_pol slots.
    assert far_field_partials(problem)[0] == 1 + 4 * 2
    rows = far_field_memory(problem, "native")
    assert rows == {"far_field_partials": 8 * n_bands * grid.n_omega}
    assert far_field_memory(problem, "numba_cpu") == {
        "statistics_partials": 8 * (n_bands + 1) * 9 * grid.n_omega
    }
    owner = select("native", problem, grid=grid)
    try:
        owner.set_initial(problem.initial.field0, None)
        values = owner.stats(problem.stats_spec)
        assert owner.memory_ledger()["far_field_partials"] == rows["far_field_partials"]
    finally:
        owner.close()
    assert values["on_axis_intensity"].shape == (1,)
    assert values["beam_spectrum"].shape == (1, grid.n_omega)


def test_declared_apertures_leave_a_request_without_them_unchanged():
    # A periodic beam or region sample carries the run's apertures; the native pass
    # gates its aperture work on the layout's aperture slot, and the values are the
    # reference's within the bound.
    grid, problem, field, tables = case("envelope", 2, "rcos", "quadrature")
    names = ("beam_spectrum", "region_spectrum", "absorbed_power")
    values, held = native_stats(problem, grid, field, names)
    assert_matches(
        values,
        reference(names, grid, problem, held, tables),
        magnitudes(names, grid, problem, held, tables),
        problem,
        names,
    )


def test_a_native_request_beyond_the_held_partials_grows_them():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 8, 3e-4, 8, window_factor=0)
    field = random_field(space, 1, grid)
    bare = problem_of(grid, space, field, StatsSpec(names=("energy",)))
    assert far_field_partials(bare) == (0, 0)
    assert far_field_memory(bare, "native") == {}
    owner = NativeBackend(bare, grid=grid)
    try:
        assert owner.memory_ledger()["far_field_partials"] == 0
        owner.set_initial(np.ascontiguousarray(field[None]), None)
        spec = spec_of(holes("rcos"))
        values = owner.stats(spec)
        assert owner.memory_ledger()["far_field_partials"] > 0
        field = held_state(owner, bare)
    finally:
        owner.close()
    tables = aperture_tables(spec, grid, space)
    assert_matches(
        {name: value[0] for name, value in values.items()},
        reference(spec.names, grid, bare, field, tables),
        magnitudes(spec.names, grid, bare, field, tables),
        bare,
        spec.names,
    )


# ------------------------------------------------------ runs, folds and refusals


def test_a_native_run_forms_the_statistics_at_planes_and_between_them():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 16, 3e-4, 16, window_factor=0.1)
    launch = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 6e-5
    ).build(grid, space, medium=SpatialMedium(1 + 1e-6j))
    names = (*FAR_FIELD_STATISTICS, "energy", "on_axis_spectrum", "on_axis_intensity")
    spec = dataclasses.replace(
        spec_of(holes("tanh"), names), periodic=("absorbed_power",), period=1
    )

    def simulation_on(backend):
        return Simulation.from_free(
            grid,
            space,
            SpatialMedium(1 + 1e-6j),
            [],
            launch,
            settings=SimulationSettings(
                controller=PIController(rtol=1e-6, atol=1e-14),
                save_planes=[1e-4, 2e-4],
                dz_max=1e-4,
                dz_init=1e-4,
                absorber=absorber(grid, temporal=False),
                backend=backend,
                integrator=RungeKutta(),
                stats=spec,
            ),
        )

    outputs = {
        backend: simulation_on(backend).run(MemoryOutput())
        for backend in ("native", "numba_cpu")
    }
    native, numba = outputs["native"], outputs["numba_cpu"]
    simulation = simulation_on("native")
    # The entrance plane and two saved planes.
    assert native["stats/aperture_spectrum"].shape == (3, 3, grid.n_omega)
    assert native["stats/region_spectrum"].shape == (3, 4, grid.n_omega)
    assert native["stats/absorbed_power"].shape == (3, 4)
    # The periodic channel's rows, one per accepted step on each backend.
    rows = np.asarray(native["step_stats/absorbed_power"])
    assert rows.shape[1:] == (4,)
    np.testing.assert_allclose(
        rows,
        np.asarray(numba["step_stats/absorbed_power"]),
        atol=1e-9 * float(np.max(np.abs(rows))),
    )
    # The entrance plane: one state, two passes, within the rounding bound of its
    # magnitudes (a Gaussian's terms are positive but for the amplitudes' phases).
    problem = simulation.translate()
    field = problem.initial.field0[0]
    tables = aperture_tables(spec, problem.grids, problem.transform.geometry)
    absolute = magnitudes(WITH_AXIS, grid, problem, field, tables)
    for name in WITH_AXIS:
        assert_within(
            native[f"stats/{name}"][0],
            numba[f"stats/{name}"][0],
            2 * bound_for(absolute, problem, name),
            elementwise=True,
            what=name,
        )
    # Two steps later the backends' fields differ by their steps' rounding, far
    # below the controller's tolerance: the statistics within 1e-9 of their scale.
    for name in WITH_AXIS:
        scale = float(np.max(np.abs(numba[f"stats/{name}"][-1])))
        np.testing.assert_allclose(
            native[f"stats/{name}"][-1], numba[f"stats/{name}"][-1], atol=1e-9 * scale
        )


def test_the_native_refusals_name_what_is_missing():
    grid = time_grid("envelope")
    space = FreeGrid(3e-4, 8, 3e-4, 8, window_factor=0)
    field = random_field(space, 1, grid)
    problem = problem_of(grid, space, field, spec_of(holes("tanh")))
    owner = NativeBackend(problem, grid=grid)
    try:
        owner.set_initial(np.ascontiguousarray(field[None]), None)
        native = owner._native_state
        apertures = np.zeros((1, 6 + grid.n_omega))
        regions = np.empty((0, 4))
        with pytest.raises(ValueError, match="flags"):
            native.aperture_statistics(16, apertures, 0, 2, regions, 1.0)
        with pytest.raises(ValueError, match="profile"):
            native.aperture_statistics(512, apertures, 7, 2, regions, 1.0)
        with pytest.raises(ValueError, match="rows"):
            native.aperture_statistics(512, apertures[:, :-1], 0, 2, regions, 1.0)
        with pytest.raises(ValueError, match="regions"):
            native.aperture_statistics(256, apertures, 0, 2, np.empty((1, 3)), 1.0)
        with pytest.raises(ValueError, match="NaN"):
            native.aperture_statistics(
                256, apertures, 0, 2, np.full((1, 4), np.nan), 1.0
            )
        with pytest.raises(TypeError):
            native.aperture_statistics(
                512, apertures.astype(np.float32), 0, 2, regions, 1.0
            )
        # The far-field statistics without tables formed for them.
        with pytest.raises(BackendError, match="tables"):
            owner._far_field_statistics({"beam_spectrum"}, StatsSpec(names=("energy",)))
    finally:
        owner.close()
