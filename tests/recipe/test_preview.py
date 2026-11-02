# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Previews: the mode set and the input pulse, before the run (SEP 41, S18)."""

import warnings

import numpy as np
import pytest

from solprop.exceptions import ToolsError
from solprop.modes.base import beta1, beta2, beta3, zdw
from solprop.recipe import Recipe, RecipeError
from solprop.recipe.preview import _phase_matching, _pump
from solprop.tools import (
    _fill_n2,
    lambda_rdw_all,
    nonlinear_coefficient,
    phase_mismatch,
    pulse_peak_power,
)


@pytest.fixture(autouse=True)
def _quiet():
    """Ignore the warnings the examples' builds give by design."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def test_the_mode_preview_is_the_run_s_mode_set_at_the_launch():
    recipe = Recipe.example("rdw")
    preview = recipe.modes()
    simulation = recipe.build()
    assert preview.z == 0.0
    assert [mode.label for mode in preview.modes] == [
        mode.channel_label.display_label for mode in simulation.modes
    ]
    first = simulation.modes[0]
    assert np.array_equal(preview.modes[0].beta2, beta2(first, preview.omega))
    limits = (
        2 * np.pi * 299_792_458.0 / preview.omega[-1],
        2 * np.pi * 299_792_458.0 / preview.omega[0],
    )
    assert preview.modes[0].zdw == pytest.approx(zdw(first, wavelength_limits=limits))
    assert np.all(np.diff(preview.omega) > 0)
    assert preview.modes[0].aeff > preview.modes[1].aeff


def test_the_mode_preview_carries_the_index_the_dispersion_and_gamma():
    recipe = Recipe.example("rdw")
    preview = recipe.modes()
    simulation = recipe.build()
    first, omega = simulation.modes[0], preview.omega
    mode = preview.modes[0]
    assert np.array_equal(mode.beta1, beta1(first, omega))
    assert np.array_equal(mode.beta3, beta3(first, omega))
    assert np.array_equal(mode.neff_minus_one, np.real(first.neff_minus_one(omega)))
    index = mode.beta * 299_792_458.0 / omega  # beta = omega Re(n_eff) / c
    assert mode.neff_minus_one == pytest.approx(index - 1.0, abs=1e-12)
    carrier = simulation.grid.reference_wavelength
    n2, _ = _fill_n2(getattr(first, "mixture"), carrier)  # noqa: B009 - not on Mode
    wavelengths = 2 * np.pi * 299_792_458.0 / omega[[0, -1]]
    expected = [nonlinear_coefficient(first, n2, w) for w in wavelengths]
    assert mode.gamma is not None
    assert mode.gamma[[0, -1]] == pytest.approx(expected, rel=1e-12)


def test_a_mode_without_a_fill_or_kerr_data_has_no_gamma(monkeypatch):
    assert all(
        mode.gamma is None for mode in Recipe.example("planar_hybrid").modes().modes
    )

    def refuse(*_args):
        raise ToolsError("no Kerr data")

    monkeypatch.setattr("solprop.tools._fill_n2", refuse)
    linear = Recipe.example("rdw").modes()
    assert linear.modes[0].gamma is None
    # Without the fill's Kerr data the phase matching is the linear condition.
    assert not linear.nonlinear and linear.modes[0].phase_mismatch is not None


def test_the_mode_preview_is_taken_where_asked():
    recipe = Recipe.example("rdw")  # a pressure gradient: the modes change along z
    entrance, exit_ = recipe.modes(), recipe.modes(z=1.5)
    assert exit_.z == 1.5
    assert not np.array_equal(entrance.modes[0].beta, exit_.modes[0].beta)


def test_a_route_without_modes_has_no_mode_preview():
    with pytest.raises(RecipeError, match="builds no mode set"):
        Recipe.example("soliton").modes()


@pytest.mark.parametrize(
    ("name", "energy", "peak_power"),
    [("rdw", 275e-6, None), ("soliton", None, 96.79090909090908)],
)
def test_the_input_preview_carries_the_recipe_s_pulse(name, energy, peak_power):
    preview = Recipe.example(name).preview_input()
    if energy is not None:
        assert preview.energy == pytest.approx(energy, rel=1e-4)
    if peak_power is not None:
        assert preview.peak_power == pytest.approx(peak_power, rel=1e-6)
    assert preview.time.shape == preview.power.shape


def test_the_input_preview_has_no_noise_and_draws_nothing():
    recipe = Recipe.example("rdw")
    first, second = recipe.preview_input(), recipe.preview_input()
    assert np.array_equal(first.power, second.power)
    document = recipe.to_dict()
    document["build"]["noise"] = "none"
    document["build"].pop("seed", None)
    quiet = Recipe.from_dict(document).preview_input()
    assert np.array_equal(first.spectrum, quiet.spectrum)


def test_a_preview_solves_the_pins_first():
    document = Recipe.example("soliton_self_compression").to_dict()
    del document["build"]["energy"]
    document["pins"] = {"energy": {"target": "soliton_order", "order": 4}}
    recipe = Recipe.from_dict(document)
    solved = recipe.resolve().get("energy")
    assert isinstance(solved, float)
    assert recipe.preview_input().energy == pytest.approx(solved, rel=1e-4)


def test_a_mode_without_a_zero_in_the_band_has_none():
    document = Recipe.example("rdw").to_dict()
    document["build"]["wavelength_limits"] = ["700 nm", "900 nm"]
    preview = Recipe.from_dict(document).modes()
    assert preview.modes[0].zdw is None


def test_each_mode_is_phase_matched_to_the_launch_soliton():
    recipe = Recipe.example("rdw")
    preview = recipe.modes()
    simulation = recipe.build()
    keywords = recipe.resolve().keywords()
    soliton, omega, pump = simulation.modes[0], preview.omega, 800e-9
    energy, fwhm = keywords["energy"], keywords["fwhm"]
    assert isinstance(energy, float) and isinstance(fwhm, float)
    assert preview.pump == pump and preview.nonlinear
    n2, _ = _fill_n2(getattr(soliton, "mixture"), pump)  # noqa: B009 - not on Mode
    pulse = {
        "peak_power": pulse_peak_power(energy, fwhm, "gauss"),
        "gamma": nonlinear_coefficient(soliton, n2, pump),
        "fwhm": fwhm,
        "shape": "gauss",  # the route's default
    }
    window = (2 * np.pi * 299_792_458.0 / omega[-1], 0.9 * pump)
    for built, mode in zip(simulation.modes, preview.modes, strict=True):
        expected = phase_mismatch(built, omega, pump, soliton_mode=soliton, **pulse)
        assert mode.phase_mismatch == pytest.approx(expected, rel=1e-6)
        roots = lambda_rdw_all(
            built, pump, soliton_mode=soliton, wavelength_limits=window, **pulse
        )
        assert mode.rdw == pytest.approx(roots, rel=1e-9)
    # The intermodal wave in the second mode is bluer than the launch mode's own.
    assert 0 < preview.modes[1].rdw[0] < preview.modes[0].rdw[0] < pump


def test_a_band_short_of_the_window_has_no_phase_matched_wave():
    document = Recipe.example("rdw").to_dict()
    document["build"]["wavelength_limits"] = ["750 nm", "900 nm"]
    mode = Recipe.from_dict(document).modes().modes[0]
    assert mode.rdw == () and mode.phase_mismatch is not None


def test_without_a_pump_there_is_no_phase_matching(monkeypatch):
    mode = Recipe.example("rdw").build().modes[0]
    assert _pump({"wavelength": None}, mode) == (None, {})
    assert _pump({"wavelength": 8e-7}, mode) == (8e-7, {})  # no width: linear
    omega = np.linspace(2e15, 4e15, 8)
    nothing = (None, ())
    assert (
        _phase_matching(
            mode, mode, omega, z=0.0, pump=None, pulse={}, limits=(1e-7, 1e-6)
        )
        == nothing
    )

    def refuse(*_args, **_kwargs):
        raise ToolsError("outside the band")

    monkeypatch.setattr("solprop.tools.phase_mismatch", refuse)
    assert (
        _phase_matching(
            mode, mode, omega, z=0.0, pump=8e-7, pulse={}, limits=(1e-7, 1e-6)
        )
        == nothing
    )
