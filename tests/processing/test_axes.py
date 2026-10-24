# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import dataclasses

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from runtime_types import requires_typecheck
from solprop.exceptions import GridError, ModeError, OutputError, ProcessingError
from solprop.grid import EnvelopeGrid, FieldGrid, TimeGrid
from solprop.output import MemoryOutput
from solprop.output.schema import Convention, FieldRetention, GridRecord
from solprop.processing import axes
from solprop.processing._examples import example_grid, example_output, gvd_phase


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_grid_from_record_rebuilds_every_axis_exactly(kind, synthetic):
    grid = synthetic.build_grid(kind)
    rebuilt = axes.grid_from_record(GridRecord.from_grid(grid))
    # The axes are functions of the recorded scalars alone, evaluated by the same
    # expressions, so equality is exact rather than approximate.
    assert rebuilt.kind == grid.kind
    for name in ("t", "omega", "t_fine", "omega_fine"):
        assert np.array_equal(getattr(rebuilt, name), getattr(grid, name)), name
    assert rebuilt.dt == grid.dt
    assert rebuilt.dt_fine == grid.dt_fine
    assert rebuilt.omega0 == grid.omega0


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_grid_from_record_keeps_the_spectral_collar(kind, synthetic):
    grid = synthetic.build_grid(kind)
    rebuilt = axes.grid_from_record(GridRecord.from_grid(grid))
    assert rebuilt.omega_collar == grid.omega_collar
    assert np.array_equal(rebuilt.active_mask, grid.active_mask)
    assert np.array_equal(rebuilt.inband_mask, grid.inband_mask)


def test_grid_from_record_refuses_an_inconsistent_record(synthetic):
    grid = synthetic.build_grid("envelope")
    record = GridRecord.from_grid(grid)
    broken = GridRecord(**{**record.__dict__, "ffac": 1.0})
    with pytest.raises(GridError, match="ffac"):
        axes.grid_from_record(broken)


def test_grid_of_reads_the_record_of_a_run(output):
    assert axes.grid_of(output).n_omega == output["field"].shape[-1]


def test_grid_of_refuses_a_handler_with_no_metadata():
    with pytest.raises(OutputError, match="no metadata"):
        axes.grid_of(MemoryOutput())


@pytest.mark.parametrize("kind", ["envelope", "field"])
def test_spectral_order_sorts_the_axis(kind, synthetic):
    grid = synthetic.build_grid(kind)
    order = axes.spectral_order(grid)
    assert np.all(np.diff(grid.omega[order]) > 0.0)
    assert sorted(order.tolist()) == list(range(grid.n_omega))


def test_spectral_order_is_the_identity_on_a_field_grid(synthetic):
    grid = synthetic.build_grid("field")
    assert np.array_equal(axes.spectral_order(grid), np.arange(grid.n_omega))


def test_band_mask_selects_the_active_band(grid):
    assert np.array_equal(axes.band_mask(grid, "active"), grid.active_mask)
    assert axes.band_mask(grid, "all").all()


def test_check_choice_refuses_an_unknown_keyword():
    with pytest.raises(ProcessingError, match="band must be one of"):
        axes._check_choice("middle", axes.BANDS, "band")
    with pytest.raises(ProcessingError, match="axis must be one of"):
        axes._check_choice("energy", axes.SPECTRAL_AXES, "axis")


def test_spectral_axis_carries_the_jacobian_of_each_abscissa():
    omega = np.array([2.0e15, 3.0e15])
    x, jacobian = axes.spectral_axis(omega, "omega")
    assert np.array_equal(x, omega)
    assert np.array_equal(jacobian, np.ones_like(omega))
    x, jacobian = axes.spectral_axis(omega, "f")
    assert np.allclose(x, omega / (2 * np.pi), rtol=1e-15)
    assert np.allclose(jacobian, 2 * np.pi, rtol=1e-15)
    x, jacobian = axes.spectral_axis(omega, "wavelength")
    # The wavelength Jacobian is 2 pi c / lambda^2, which is omega^2 / (2 pi c); the two
    # forms agree to round-off, which is what this checks.
    assert np.allclose(jacobian, 2 * np.pi * 299792458.0 / x**2, rtol=1e-14)


def test_spectral_axis_refuses_a_wavelength_at_zero_frequency():
    with pytest.raises(ProcessingError, match="not positive"):
        axes.spectral_axis(np.array([0.0, 1e15]), "wavelength")


def test_layout_reports_the_shape_of_a_run(synthetic):
    output = synthetic.build_run("envelope", n_modes=2, n_pol=2)
    n_z, n_modes, n_pol, n_omega = axes.layout(output)
    assert (n_z, n_modes, n_pol) == (5, 2, 2)
    assert n_omega == axes.grid_of(output).n_omega


def test_nearest_plane_needs_a_saved_plane():
    with pytest.raises(OutputError, match="no saved plane"):
        axes._nearest_plane(MemoryOutput(), 0.0)


def test_plane_field_refuses_a_plane_and_a_position_together(output):
    with pytest.raises(OutputError, match="not both"):
        axes._plane_field(output, 0, 0.0, convention="fft")


def test_plane_field_refuses_an_index_out_of_range(output):
    with pytest.raises(OutputError, match="out of range"):
        axes._plane_field(output, 9, None, convention="fft")


def test_plane_field_counts_a_negative_index_from_the_end(output):
    last = axes._plane_field(output, -1, None, convention="fft")
    assert np.array_equal(last, axes._plane_field(output, 4, None, convention="fft"))


def test_plane_field_selects_by_position(output):
    by_z = axes._plane_field(output, None, 0.049, convention="fft")
    assert np.array_equal(by_z, axes._plane_field(output, 2, None, convention="fft"))


class _ProtocolOnly:
    """An output that is not a plane store: the ``Output`` protocol alone, as a
    user's own handler implements it, reading through a store it wraps."""

    def __init__(self, stored: MemoryOutput) -> None:
        self.stored = stored

    def open(self, metadata, *, force=False):
        self.stored.open(metadata, force=force)

    def save(self, plane):
        self.stored.save(plane)

    def close(self):
        self.stored.close()

    @property
    def metadata(self):
        return self.stored.metadata

    @property
    def n_saved(self):
        return self.stored.n_saved

    @property
    def stat_names(self):
        return self.stored.stat_names

    @property
    def step_stat_names(self):
        return self.stored.step_stat_names

    def __getitem__(self, key):
        return self.stored[key]

    def __contains__(self, key):
        return key in self.stored

    def fields(self, convention: Convention = "optics"):
        return self.stored.fields(convention)


def keeping_planes(memory: MemoryOutput, kept: tuple[int, ...]) -> MemoryOutput:
    """Return the run stored again keeping the field on the planes `kept` alone."""
    planes = memory.planes()
    retention = FieldRetention("planes", kept).resolved(len(planes))
    first = planes[0].field
    assert first is not None and memory.metadata is not None
    partial = MemoryOutput()
    partial.open(memory.metadata)
    partial.retain(retention, first.shape)
    for index, plane in enumerate(planes):
        partial.save(
            plane
            if retention.keeps(index)
            else dataclasses.replace(plane, field=None, noise_phase=None)
        )
    partial.close()
    return partial


def test_a_handler_that_is_not_a_store_reads_its_planes_by_index_and_position(memory):
    view = _ProtocolOnly(memory)
    # The same stored rows read through either path: exact.
    for index in (1, -1):
        assert np.array_equal(
            axes._plane_field(view, index, None, convention="fft"),
            axes._plane_field(memory, index, None, convention="fft"),
        )
    assert np.array_equal(
        axes._plane_field(view, None, 0.049, convention="fft"),
        axes._plane_field(memory, 2, None, convention="fft"),
    )
    with pytest.raises(OutputError, match="out of range"):
        axes._plane_field(view, 9, None, convention="fft")
    assert np.array_equal(
        axes._plane_field(view, None, None, convention="fft"), memory.fields("fft")
    )


def test_a_run_keeping_some_fields_answers_for_those_planes_alone(memory):
    partial = keeping_planes(memory, (0, -1))
    # A store reads one plane alone: the kept ones answer, exactly as stored ...
    for index in (0, -1):
        assert np.array_equal(
            axes._plane_field(partial, index, None, convention="fft"),
            axes._plane_field(memory, index, None, convention="fft"),
        )
    # ... and another plane is refused by name, by index or by position.
    with pytest.raises(OutputError, match="fields='planes'"):
        axes._plane_field(partial, 2, None, convention="fft")
    with pytest.raises(OutputError, match="fields='planes'"):
        axes._plane_field(partial, None, 0.049, convention="fft")
    # A handler that reads only whole runs cannot pick out the kept planes.
    with pytest.raises(OutputError, match="fields='planes'"):
        axes._plane_field(_ProtocolOnly(partial), 0, None, convention="fft")


@pytest.mark.parametrize(
    ("modes", "expected"),
    [(None, (2, 2, 8)), (0, (2, 8)), ([0, 1], (2, 2, 8)), ("sum", (2, 8))],
)
def test_apply_modes_selects_and_reduces(modes, expected):
    values = np.ones((2, 2, 8), dtype=complex)
    selected, kept = axes._apply_modes(values, modes, allow_sum=True)
    assert selected.shape == expected
    assert kept is (modes is None or isinstance(modes, list))


def test_apply_modes_refuses_a_sum_of_complex_amplitudes():
    with pytest.raises(ModeError, match="meaningless"):
        axes._apply_modes(np.ones((2, 1, 4), dtype=complex), "sum", allow_sum=False)


@pytest.mark.parametrize("modes", ["all", 1.5, ["a"], object()])
def test_check_mode_selection_refuses_an_unknown_selection(modes):
    with pytest.raises(ModeError, match="must be an index"):
        axes._check_mode_selection(modes)


@pytest.mark.parametrize("modes", [None, 0, [0, 1], "sum"])
def test_check_mode_selection_accepts_every_documented_form(modes):
    assert axes._check_mode_selection(modes) is None


@pytest.mark.parametrize("modes", [3, -3, [0, 2]])
def test_apply_modes_refuses_an_index_out_of_range(modes):
    with pytest.raises(ModeError, match="out of range"):
        axes._apply_modes(np.ones((2, 1, 4), dtype=complex), modes, allow_sum=True)


@pytest.mark.parametrize(
    ("shape", "mode_axis", "expected"),
    [
        ((3, 1, 1, 8), True, (3, 8)),
        ((3, 2, 1, 8), True, (3, 2, 8)),
        ((3, 1, 2, 8), True, (3, 2, 8)),
        ((3, 2, 2, 8), True, (3, 2, 2, 8)),
        ((3, 1, 8), False, (3, 8)),
    ],
)
def test_squeeze_drops_only_trivial_axes(shape, mode_axis, expected):
    values = np.ones(shape, dtype=complex)
    assert axes._squeeze(values, mode_axis=mode_axis).shape == expected


def test_example_helpers_build_a_runnable_synthetic_run():
    assert example_grid("field").kind == "field"
    assert np.array_equal(
        gvd_phase(example_grid(), 1e-27, 0.0), np.ones(example_grid().n_omega)
    )
    plain = example_output(with_stats=False)
    assert plain.stat_names == ()
    assert example_output(n_modes=2)["field"].shape[1] == 2


@requires_typecheck
def test_grid_from_record_is_shape_checked():
    with pytest.raises(TypeCheckError):
        axes.spectral_axis(np.array([1, 2]), "f")


def test_the_temporal_collar_is_recorded_and_rebuilt_exactly(grid):
    rebuilt = axes.grid_from_record(GridRecord.from_grid(grid))
    # The record carries the collar's inner edges, so the rebuilt collar is the
    # original's bit for bit, not a placeholder at a default fraction.
    assert rebuilt.t_collar == grid.t_collar
    assert np.array_equal(rebuilt.t_window, grid.t_window)


def _smooth_grids() -> list[TimeGrid]:
    """Return 2-3-5-smooth field and envelope grids with fine/coarse ratios 1 to 8."""
    rng = np.random.default_rng(118)
    grids: list[TimeGrid] = [
        cls.from_limits(1.0, 800e-9, (400e-9, 2e-6), trange, fft_size="smooth")
        for cls in (FieldGrid, EnvelopeGrid)
        for trange in (0.4e-12, 1.0e-12, 2.0e-12)
    ]
    for cls in (FieldGrid, EnvelopeGrid):
        for _ in range(60):
            trange = float(rng.uniform(0.2e-12, 2e-12))
            dt = float(rng.uniform(0.1e-15, 1.0e-15))
            grids.append(
                cls.from_limits(
                    1.0, 800e-9, (600e-9, 1.2e-6), trange, dt=dt, fft_size="smooth"
                )
            )
    return grids


def test_grid_from_record_is_bitwise_on_smooth_grids():
    # d_omega must be the expression the axes were built with, 2 pi/(n_to
    # dt_fine). The rounded dt = dt_fine (n_to // n_t) makes 2 pi/(n_t dt) differ from
    # it in the last place on some non-power-of-two grids, which the witness below
    # checks this set contains.
    grids = _smooth_grids()
    ratios = {grid.n_to // grid.n_t for grid in grids}
    assert {1, 2, 3, 4, 5, 6, 8} <= ratios
    assert any(2.0 * np.pi / (grid.n_t * grid.dt) != grid.d_omega for grid in grids)
    for grid in grids:
        rebuilt = axes.grid_from_record(GridRecord.from_grid(grid))
        for name in ("t", "omega", "t_fine", "omega_fine"):
            assert np.array_equal(getattr(rebuilt, name), getattr(grid, name)), (
                grid.kind,
                grid.n_t,
                grid.n_to,
                name,
            )
