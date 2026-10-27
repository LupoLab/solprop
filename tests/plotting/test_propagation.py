# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import matplotlib.pyplot as plt
import numpy as np
import pytest

from solprop.exceptions import ModeError, ProcessingError
from solprop.plotting import propagation, style
from solprop.processing._examples import example_output
from solprop.processing.axes import _check_choice


def test_prop_2d_draws_one_figure_of_two_panels_for_one_mode(single):
    figures = propagation.prop_2d(single)
    assert len(figures) == 1
    # Two maps and the colour bar of each.
    assert len(figures[0].axes) == 4
    spectral, temporal = figures[0].axes[0], figures[0].axes[1]
    assert spectral.get_xlabel() == "Frequency (PHz)"
    assert spectral.get_ylabel() == "Distance (cm)"
    assert temporal.get_xlabel() == "Time (fs)"


def test_prop_2d_adds_a_modal_sum(multimode):
    figures = propagation.prop_2d(multimode)
    assert len(figures) == 4
    assert figures[-1].get_suptitle() == "Propagation (sum)"


def test_prop_2d_draws_only_the_sum_when_asked(multimode):
    assert len(propagation.prop_2d(multimode, modes="sum")) == 1


def test_prop_2d_draws_only_one_mode_when_asked(multimode):
    figures = propagation.prop_2d(multimode, modes=1)
    assert len(figures) == 1
    assert figures[0].get_suptitle() == "Propagation (mode 2)"


def test_prop_2d_draws_a_chosen_set_of_modes(multimode):
    figures = propagation.prop_2d(multimode, modes=[0, 2])
    assert [figure.get_suptitle() for figure in figures] == [
        "Propagation (mode 1)",
        "Propagation (mode 3)",
        "Propagation (sum)",
    ]


@pytest.mark.parametrize(
    ("axis", "label"),
    [
        ("wavelength", "Wavelength (nm)"),
        ("f", "Frequency (PHz)"),
        ("omega", "Angular frequency (rad/fs)"),
    ],
)
def test_prop_2d_labels_each_spectral_axis(single, axis, label):
    figures = propagation.prop_2d(single, axis)
    assert figures[0].axes[0].get_xlabel() == label


def test_prop_2d_takes_a_frequency_range_in_hertz(single):
    figures = propagation.prop_2d(single, "f", spec_range=(0.0, 2e15))
    assert figures[0].axes[0].get_xlim() == pytest.approx((0.0, 2.0))


def test_prop_2d_shows_the_default_wavelength_range_on_any_axis(single):
    figures = propagation.prop_2d(single, "f")
    expected = style.wavelength_to_axis("f", propagation.DEFAULT_WAVELENGTH_RANGE)
    assert figures[0].axes[0].get_xlim() == pytest.approx(
        (expected[0] * 1e-15, expected[1] * 1e-15)
    )


def test_prop_2d_puts_the_peak_at_zero_decibels(single):
    figures = propagation.prop_2d(single)
    mesh = figures[0].axes[0].collections[0]
    assert float(np.max(np.asarray(mesh.get_array()))) == pytest.approx(0.0, abs=1e-12)
    assert mesh.get_clim() == (-60.0, 0.0)


def test_prop_2d_honours_the_decibel_floor(single):
    figures = propagation.prop_2d(single, db_min=-30.0)
    assert figures[0].axes[0].collections[0].get_clim() == (-30.0, 0.0)


def test_prop_2d_accepts_a_resolution_a_bandpass_and_a_colour_map(single):
    figures = propagation.prop_2d(
        single,
        "wavelength",
        resolution=30e-9,
        bandpass=(700e-9, 900e-9),
        cmap=style.cmap_white(),
    )
    assert len(figures) == 1


def test_prop_2d_reads_a_file_handler_the_same_way(single, on_file):
    memory = propagation.prop_2d(single)[0].axes[0].collections[0].get_array()
    from_file = propagation.prop_2d(on_file)[0].axes[0].collections[0].get_array()
    assert np.array_equal(np.asarray(memory), np.asarray(from_file))


def test_time_1d_draws_the_end_of_the_run_by_default(single):
    figure = propagation.time_1d(single)
    (line,) = figure.axes[0].lines
    assert "10.00 cm" in str(line.get_label())


def test_time_1d_draws_one_line_per_position(single):
    figure = propagation.time_1d(single, [0.0, 0.05, 0.1])
    assert len(figure.axes[0].lines) == 3
    assert figure.axes[0].get_xlabel() == "Time (fs)"


def test_time_1d_draws_one_line_per_mode_and_position(multimode):
    figure = propagation.time_1d(multimode, [0.0, 0.1])
    assert len(figure.axes[0].lines) == 6
    assert "mode 2" in str(figure.axes[0].lines[1].get_label())


@pytest.mark.parametrize(
    ("quantity", "ordinate"),
    [
        ("power", "Power"),
        ("field", "Field"),
        ("field_squared", "Power"),
    ],
)
def test_time_1d_draws_each_quantity(single, quantity, ordinate):
    figure = propagation.time_1d(single, 0.0, quantity=quantity)
    assert figure.axes[0].get_ylabel().startswith(ordinate)


def test_time_1d_reports_the_width_in_the_legend(single):
    with_width = propagation.time_1d(single, 0.0)
    without = propagation.time_1d(single, 0.0, widths=False)
    assert "fs]" in str(with_width.axes[0].lines[0].get_label())
    assert "fs]" not in str(without.axes[0].lines[0].get_label())


def test_time_1d_draws_into_a_given_axis(single):
    figure, ax = plt.subplots()
    assert propagation.time_1d(single, 0.0, ax=ax) is figure


def test_time_1d_refuses_an_unknown_quantity():
    with pytest.raises(ProcessingError, match="quantity must be one of"):
        _check_choice("intensity", propagation.SLICE_QUANTITIES, "quantity")


def test_time_1d_sums_the_modes_when_asked(multimode):
    figure = propagation.time_1d(multimode, 0.0, modes="sum")
    assert len(figure.axes[0].lines) == 1


def test_time_1d_refuses_a_modal_sum_of_a_field(multimode):
    with pytest.raises(ModeError, match="meaningless"):
        propagation.time_1d(multimode, 0.0, quantity="field", modes="sum")


def test_spec_1d_uses_a_logarithmic_ordinate_by_default(single):
    figure = propagation.spec_1d(single, [0.0, 0.1])
    ax = figure.axes[0]
    assert ax.get_yscale() == "log"
    assert ax.get_xlabel() == "Wavelength (nm)"
    assert ax.get_ylabel() == "Spectral energy density (J/m)"
    assert len(ax.lines) == 2


def test_spec_1d_can_use_a_linear_ordinate(single):
    assert propagation.spec_1d(single, 0.0, log=False).axes[0].get_yscale() == "linear"


@pytest.mark.parametrize("axis", ["wavelength", "f", "omega"])
def test_spec_1d_draws_each_spectral_axis(single, axis):
    figure = propagation.spec_1d(single, 0.0, axis)
    low, high = figure.axes[0].get_xlim()
    assert low < high


def test_spec_1d_takes_an_angular_frequency_range_in_rad_per_second(single):
    figure = propagation.spec_1d(single, 0.0, "omega", spec_range=(1e15, 4e15))
    assert figure.axes[0].get_xlim() == pytest.approx((1.0, 4.0))


def test_spec_1d_accepts_a_resolution_and_an_axis(single):
    figure, ax = plt.subplots()
    assert propagation.spec_1d(single, 0.0, resolution=20e-9, ax=ax) is figure


def test_spec_1d_names_the_modes(multimode):
    figure = propagation.spec_1d(multimode, 0.0)
    assert "mode 3" in str(figure.axes[0].lines[2].get_label())


def test_energy_draws_a_conversion_efficiency_axis(single):
    figure = propagation.energy(single)
    left, right = figure.axes
    assert left.get_ylabel() == "Energy (uJ)"
    assert right.get_ylabel() == "Conversion efficiency (%)"
    # The example run conserves energy, so the efficiency ends where it started.
    values = np.asarray(left.lines[0].get_ydata(), dtype=float)
    assert values[-1] == pytest.approx(values[0], rel=1e-9)


def test_energy_scales_the_efficiency_by_the_input(single):
    figure = propagation.energy(single)
    left, right = figure.axes
    low, high = left.get_ylim()
    ratio = right.get_ylim()[1] / high
    assert ratio == pytest.approx(right.get_ylim()[0] / low if low else ratio, rel=1e-9)
    assert right.get_ylim()[1] > 100.0 * 0.9


def test_energy_draws_one_line_per_mode(multimode):
    figure = propagation.energy(multimode)
    assert len(figure.axes[0].lines) == 3
    assert figure.axes[0].get_legend() is not None


def test_energy_accepts_a_bandpass_and_an_axis(single):
    figure, ax = plt.subplots()
    assert propagation.energy(single, bandpass=(700e-9, 900e-9), ax=ax) is figure


def test_the_dash_patterns_repeat_beyond_the_fifth_mode():
    assert propagation._dashes(0) == propagation._dashes(5)
    assert propagation._dashes(1) != propagation._dashes(0)


def test_a_scalar_position_and_a_sequence_agree(single):
    assert propagation._plane_indices(single, 0.05) == propagation._plane_indices(
        single, [0.05]
    )
    assert propagation._plane_indices(single, None) == [single.n_saved - 1]


def test_a_two_mode_run_labels_its_sum():
    output = example_output(n_modes=2)
    assert propagation._drawn_labels(output, 2, None) == ["mode 1", "mode 2"]
    assert propagation._drawn_labels(output, 1, "sum") == ["sum"]
    assert propagation._drawn_labels(output, 1, 1) == ["mode 2"]
