# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""How numbers are shown: the ends a doctest does not reach."""

import numpy as np

from solprop_gui.model.display import (
    beam_reach,
    decibels,
    even_rows,
    light_range,
    mirrored_rows,
    pulse_window,
    scaled_unit,
    steady_limits,
    wants_log,
)


def test_no_light_reads_as_the_floor_and_the_whole_window() -> None:
    assert (decibels(np.zeros(3)) == -300.0).all()
    t = np.linspace(-50.0, 50.0, 11)
    assert pulse_window(t, np.zeros(11)) == 50.0
    assert pulse_window(np.zeros(0), np.zeros(0)) == 1.0


def test_a_pulse_window_reads_every_plane() -> None:
    t = np.linspace(-50.0, 50.0, 101)
    rows = np.vstack((np.exp(-(t**2)), np.exp(-((t - 20.0) ** 2))))
    assert pulse_window(t, rows) > 20.0


def test_a_pure_number_near_one_takes_no_power_of_ten() -> None:
    assert scaled_unit(np.array([3.0]), "1/m") == ("1/m", 1.0)


def test_uneven_planes_and_none() -> None:
    assert even_rows(np.zeros(0))[0].size == 0
    assert even_rows(np.zeros(0))[1:] == (0.0, 1.0)
    assert wants_log(np.array([np.nan])) is False
    assert steady_limits([np.nan]) == (0.0, 1.0)


def test_a_beam_with_no_light_shows_all_its_radius_and_rows_mirror() -> None:
    r = np.array([1.0, 2.0])
    assert beam_reach(r, np.zeros((0, 2))) == 2.0
    y, rows = mirrored_rows(r, np.array([[4.0, 2.0]]), 2.0, 5)
    assert y.size == 5
    assert rows.shape == (1, 5)


def test_a_power_of_ten_reads_downwards_too() -> None:
    label, factor = scaled_unit(np.array([3e-5]), "1/m")
    assert label == "10⁻⁵ 1/m"
    assert abs(3e-5 * factor - 3.0) < 1e-9


def test_the_light_on_an_axis_is_where_any_row_holds_it() -> None:
    x = np.arange(10.0)
    rows = np.zeros((2, 10))
    rows[0, 3] = 1.0
    rows[1, 6] = 1.0
    assert light_range(x, rows) == (3.0, 6.0)
    assert light_range(x, rows, margin=0.5) == (1.5, 7.5)
    assert light_range(x, rows, margin=10.0) == (0.0, 9.0)
    assert light_range(x, np.zeros((2, 10))) == (0.0, 9.0)
