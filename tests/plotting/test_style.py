# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import math

import matplotlib.pyplot as plt
import numpy as np
import pytest

from solprop.exceptions import ProcessingError
from solprop.output import MemoryOutput
from solprop.output.schema import GridRecord, ModeRecord, RunMetadata, StepperRecord
from solprop.plotting import style
from solprop.processing._examples import example_grid, example_output
from solprop.processing.axes import _check_choice


def test_cmap_white_starts_at_white():
    colours = style.cmap_white()
    assert [float(one) for one in colours(0.0)] == [1.0, 1.0, 1.0, 1.0]
    assert [float(one) for one in colours(1.0)] != [1.0, 1.0, 1.0, 1.0]


def test_cmap_colours_returns_distinct_colours():
    colours = style.cmap_colours(4)
    assert len(colours) == 4
    assert len({tuple(one) for one in colours}) == 4


def test_cmap_colours_needs_at_least_one():
    with pytest.raises(ProcessingError, match="at least one colour"):
        style.cmap_colours(0)


@pytest.mark.parametrize(("count", "portrait"), [(1, True), (5, True), (5, False)])
def test_subplot_grid_makes_exactly_the_axes_asked_for(count, portrait):
    figure, axes = style.subplot_grid(count, portrait=portrait)
    assert len(axes) == count
    assert len(figure.axes) == count
    plt.close(figure)


def test_subplot_grid_has_more_rows_than_columns_in_portrait():
    def geometry(**kwargs):
        spec = style.subplot_grid(6, **kwargs)[1][0].get_subplotspec()
        assert spec is not None
        return spec.get_gridspec().get_geometry()

    assert geometry() == (3, 2)
    assert geometry(portrait=False) == (2, 3)


def test_subplot_grid_needs_at_least_one_axis():
    with pytest.raises(ProcessingError, match="at least one axis"):
        style.subplot_grid(0)


@pytest.mark.parametrize(
    ("peak", "unit"),
    [(2.5e3, "kW"), (2.5e6, "MW"), (2.5e9, "GW"), (2.5e12, "TW"), (2.5e15, "PW")],
)
def test_power_unit_steps_by_thousands(peak, unit):
    factor, label = style.power_unit(np.array([peak]))
    assert label == unit
    assert 1.0 <= peak * factor < 1000.0


@pytest.mark.parametrize("peak", [0.0, -1.0, 1e-3, 1e20])
def test_power_unit_clamps_to_the_ends_of_the_scale(peak):
    factor, label = style.power_unit(np.array([peak]))
    assert label in style.POWER_UNITS
    assert factor > 0.0


def test_power_unit_needs_a_value():
    with pytest.raises(ProcessingError, match="empty array"):
        style.power_unit(np.array([]))


@pytest.mark.parametrize(
    ("axis", "label"),
    [
        ("wavelength", "Wavelength (nm)"),
        ("f", "Frequency (PHz)"),
        ("omega", "Angular frequency (rad/fs)"),
    ],
)
def test_spectral_display_labels_each_abscissa(axis, label):
    factor, shown = style.spectral_display(axis)
    assert shown == label
    assert factor > 0.0


@pytest.mark.parametrize("axis", ["wavelength", "f", "omega"])
def test_spectral_limits_increase(axis):
    low, high = style.spectral_limits(axis, (400e-9, 2e-6))
    assert low < high


def test_spectral_limits_of_a_wavelength_axis_are_the_range():
    assert style.spectral_limits("wavelength", (2e-6, 400e-9)) == (400.0, 2000.0)


@pytest.mark.parametrize("corner", ["ul", "ur", "ll", "lr"])
def test_corner_text_places_a_label_in_each_corner(corner):
    figure, ax = plt.subplots()
    style.corner_text(ax, "a)", corner)
    assert ax.texts[0].get_text() == "a)"
    plt.close(figure)


def test_corner_text_refuses_an_unknown_corner():
    with pytest.raises(ProcessingError, match="corner must be one of"):
        _check_choice("middle", style.CORNERS, "corner")


def test_fwhm_arrows_marks_the_width():
    figure, ax = plt.subplots()
    t = np.linspace(-50.0, 50.0, 2001)
    width = style.fwhm_arrows(ax, t, np.exp(-((t / 10.0) ** 2)))
    # exp(-(t/10)^2) is half its peak at t = +/- 10 sqrt(ln 2), so 16.65 fs wide.
    assert width == pytest.approx(16.651, abs=1e-3)
    # Two arrows and their label; the arrows are annotations with no text of their own.
    assert len(ax.texts) == 3
    assert "16.65 fs" in ax.texts[-1].get_text()
    plt.close(figure)


def test_fwhm_arrows_draws_nothing_without_a_width():
    figure, ax = plt.subplots()
    t = np.linspace(-50.0, 50.0, 101)
    assert math.isnan(style.fwhm_arrows(ax, t, np.ones_like(t)))
    assert len(ax.texts) == 0
    plt.close(figure)


def test_mode_labels_fall_back_to_an_index():
    assert style.mode_labels(example_output(n_modes=2), 2) == ["mode 1", "mode 2"]


def test_mode_labels_are_read_from_the_mode_records():
    output = _with_modes([{"kind": "HE", "n": 1, "m": 1, "phi": 0.0}])
    assert style.mode_labels(output, 1) == ["HE11"]


def test_mode_labels_name_the_polarisation_when_the_set_mixes_angles():
    output = _with_modes(
        [
            {"kind": "HE", "n": 1, "m": 1, "phi": 0.0},
            {"kind": "HE", "n": 1, "m": 1, "phi": 0.5 * math.pi},
            {"kind": "HE", "n": 1, "m": 2, "phi": 0.25 * math.pi},
            {"kind": "HE", "n": 1, "m": 2},
        ]
    )
    assert style.mode_labels(output, 4) == [
        "HE11 (x)",
        "HE11 (y)",
        "HE12 (0.25pi)",
        "HE12 (?)",
    ]


def test_mode_labels_of_a_handler_without_metadata():
    assert style.mode_labels(MemoryOutput(), 2) == ["mode 1", "mode 2"]


def _with_modes(configurations):
    grid = example_grid()
    metadata = RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=tuple(
            ModeRecord(
                kind="solprop.modes.capillary.MarcatiliMode",
                effective_area=1e-8,
                configuration=configuration,
            )
            for configuration in configurations
        ),
        simulation_kind="multimode",
        configuration={},
        stepper=StepperRecord(tableau="DOPRI5", controller={}),
    )
    output = MemoryOutput()
    output.open(metadata)
    return output
