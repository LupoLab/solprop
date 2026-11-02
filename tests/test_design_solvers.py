# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The general design solver, its targets and its roots, and the radius of a zero."""

import math
from itertools import pairwise

import numpy as np
import pytest

from solprop import tools
from solprop.constants import C
from solprop.exceptions import ToolsError
from solprop.modes import MarcatiliMode, base
from solprop.modes.antiresonant import TubeLattice, VincettiMode
from solprop.nonlinear.gnlse import SpecifiedMode
from solprop.physdata import MaterialSelection
from solprop.pulses import GaussianPulse, SechPulse
from solprop.tools import (
    RDW,
    ZDW,
    DesignRoots,
    SolitonOrder,
    radius_zdw,
    solve_parameter,
)

PUMP = 800e-9
OMEGA0 = 2.0 * math.pi * C / PUMP
# A constant cladding keeps every index request inside a model's range, so the suite's
# warnings-are-errors rule needs no filter here.
GLASS = 1.45


def capillary(gas, radius=125e-6):
    """The capillary pressure_zdw and pressure_rdw re-evaluate at each pressure."""
    return tools._design_capillary(
        radius,
        gas,
        temperature=293.15,
        m=1,
        model="full",
        cladding=GLASS,
        lattice=None,
        materials=MaterialSelection(),
    )


def pressure_builder(gas):
    """The Marcatili builder over pressure, with the fill's Kerr coefficient."""
    fill = capillary(gas)

    def build(pressure):
        mode = tools._capillary_at(fill, pressure)
        assert mode.mixture is not None  # _design_capillary builds on a mixture.
        n2 = tools._fill_n2(mode.mixture, PUMP)[0]
        return mode, tools.nonlinear_coefficient(mode, n2, PUMP)

    return build


def sech(**keywords):
    return SechPulse(wavelength=PUMP, fwhm=10e-15, **keywords)


# ------------------------------------------------------- the capillary functions


@pytest.mark.parametrize(
    "pulse",
    [
        {},
        {"peak_power": 2e9, "fwhm": 10e-15},
        {"peak_power": 2e10, "nonlinear_phase": "supplied_power"},
    ],
    ids=["linear", "largest", "supplied_power"],
)
def test_the_rdw_target_reproduces_pressure_rdw(pulse):
    expected = tools.pressure_rdw(125e-6, "He", 200e-9, PUMP, cladding=GLASS, **pulse)
    roots = solve_parameter(
        pressure_builder("He"), (1e-6, 100.0), RDW(PUMP, 200e-9, **pulse)
    )
    # The same residual on the same samples, refined to the same tolerance: the same
    # root, to rounding. A linear target takes the builder's coefficient and ignores it.
    assert roots.roots == pytest.approx((expected,), rel=1e-12)
    assert roots.search == (1e-6, 100.0)
    # 258 samples less the two shared ends, the refinement, and the answer once more.
    assert 256 < roots.builds < 300


def test_the_zdw_target_reproduces_pressure_zdw_where_it_has_one_root():
    expected = tools.pressure_zdw(125e-6, "Ar", 500e-9, cladding=GLASS)
    fill = capillary("Ar")
    roots = solve_parameter(
        lambda pressure: tools._capillary_at(fill, pressure), (1e-6, 100.0), ZDW(500e-9)
    )
    assert len(roots) == 1
    # pressure_zdw brackets once and stops at xtol = 2e-12 bar, this search refines
    # one sampled bracket: both roots sit inside the band in which beta2, a
    # Richardson difference good to 5e-9 (modes.base), is noise, which its slope in
    # pressure carries to the root as a few parts in 1e8.
    assert roots.pick() == pytest.approx(expected, rel=1e-6)


def test_radius_zdw_inverts_the_modes_zero_dispersion_wavelength():
    radius = radius_zdw("He", 1.0, 400e-9, cladding=GLASS)
    mode = MarcatiliMode(radius, "He", 1.0, cladding=GLASS)
    # Brent stops at 2e-12 of the radius; beta2 is a Richardson difference good to
    # 5e-9 (modes.base), which both zeros inherit through slopes of order one.
    assert base.zdw(mode, wavelength_limits=(200e-9, 990e-9)) == pytest.approx(
        400e-9, rel=1e-6
    )


def test_radius_zdw_refuses_bad_limits_and_a_missing_root():
    with pytest.raises(ToolsError, match="radius_limits"):
        radius_zdw("He", 1.0, 400e-9, radius_limits=(1e-4, 1e-5), cladding=GLASS)
    with pytest.raises(ToolsError, match="pressure must be finite and positive"):
        radius_zdw("He", 0.0, 400e-9, cladding=GLASS)
    with pytest.raises(ToolsError, match=r"root index 1 is out of range: 1 root, at"):
        radius_zdw("He", 1.0, 400e-9, cladding=GLASS, root=1)


# ------------------------------------------------------------ the soliton order


def test_a_soliton_order_has_a_root_on_each_side_of_the_zero_dispersion_pressure():
    target = SolitonOrder(4.0, sech(energy=20e-6))
    build = pressure_builder("Ar")
    roots = solve_parameter(build, (0.01, 5.0), target)
    zero = tools.pressure_zdw(125e-6, "Ar", PUMP, cladding=GLASS)
    # The order diverges at the zero, and falls on both sides of it: in normal
    # dispersion it is the conventional measure, which settles above 3.2 here.
    assert len(roots) == 2
    assert roots.pick("lowest") < zero < roots.pick("highest")
    assert roots.pick(0) == roots.pick("lowest")
    assert roots.pick(-1) == roots.pick("highest")
    for pressure in roots:
        mode, gamma = build(pressure)
        dispersion = float(base.beta2(mode, np.array([OMEGA0]))[0])
        order = tools.soliton_order(target.peak_power, 10e-15, gamma, dispersion)
        # The root is found to 2e-12 of the pressure, but the order inherits the
        # Richardson noise of beta2 (5e-9, modes.base) at half its relative size.
        assert order == pytest.approx(4.0, rel=1e-7)


def test_the_pulse_for_an_order_is_closed_form():
    mode = MarcatiliMode(125e-6, "He", 1.0, cladding=GLASS)
    params = tools.soliton_params(
        mode, "He", 1.0, energy=1e-6, fwhm=10e-15, wavelength=PUMP
    )
    for shape in ("sech", "gauss"):
        power = tools.soliton_peak_power(3.0, 10e-15, params.gamma, params.beta2, shape)
        energy = tools.pulse_energy(power, 10e-15, shape)
        designed = tools.soliton_params(
            mode, "He", 1.0, energy=energy, fwhm=10e-15, wavelength=PUMP, shape=shape
        )
        # N^2 is linear in the peak power, and the energy in it: rounding only.
        assert designed.soliton_order == pytest.approx(3.0, rel=1e-12)
        assert tools.pulse_peak_power(energy, 10e-15, shape) == pytest.approx(
            power, rel=1e-15
        )


@pytest.mark.parametrize(
    "pulse, shape, peak_power",
    [
        (sech(peak_power=1e9), "sech", 1e9),
        (
            SechPulse(wavelength=PUMP, width=10e-15 / tools._SECH_WIDTH, energy=1e-5),
            "sech",
            tools.pulse_peak_power(1e-5, 10e-15, "sech"),
        ),
        (
            GaussianPulse(
                wavelength=PUMP, fwhm=10e-15, energy=1e-5, phase=[0.3, 1e-15]
            ),
            "gauss",
            tools.pulse_peak_power(1e-5, 10e-15, "gauss"),
        ),
    ],
)
def test_a_soliton_order_reads_its_pulse(pulse, shape, peak_power):
    target = SolitonOrder(2.0, pulse)
    assert target.shape == shape
    # A natural width converts to the FWHM by one multiplication.
    assert target.fwhm == pytest.approx(10e-15, rel=1e-15)
    assert target.peak_power == pytest.approx(peak_power, rel=1e-15)


# ------------------------------------------------------------------ the search


def taylor(value, pole=2.53):
    """A mode whose dispersion at the pump changes sign at 1.7 and through a pole."""
    return SpecifiedMode.from_taylor(
        [0.0, 0.0, 1e-27 * (value - 1.7) / (value - pole)], OMEGA0
    )


def test_a_sign_change_through_a_pole_is_not_a_root():
    roots = solve_parameter(taylor, (0.0, 4.0), ZDW(PUMP))
    # A sign change at 1.7 and another through the pole at 2.53; Brent converges onto
    # both, and only the first ends on a small residual. The root is a Richardson
    # difference of an exact quadratic, good to rounding of beta1.
    assert roots.roots == pytest.approx((1.7,), rel=1e-9)
    # A search from zero samples linearly only: 129 values.
    assert 129 < roots.builds < 200


def test_a_search_without_a_root_reports_the_attainable_range():
    target = SolitonOrder(30.0, sech(energy=50e-6))
    roots = solve_parameter(pressure_builder("He"), (0.1, 2.0), target)
    assert len(roots) == 0
    assert roots.quantity == "soliton order"
    assert roots.attainable is not None
    low, high = roots.attainable
    assert 0.0 < low < high < 30.0
    with pytest.raises(
        ToolsError,
        match=r"between 0.1 and 2 gives a soliton order of 30: across it the "
        r"soliton order spans",
    ):
        roots.pick()


def test_an_unreachable_dispersive_wave_reports_its_wavelengths():
    roots = solve_parameter(pressure_builder("He"), (1.0, 2.0), RDW(PUMP, 120e-9))
    assert roots.roots == ()
    assert roots.attainable is not None
    # The capillary's dispersive wave from 800 nm lies near 200 nm at a bar or two.
    assert 150e-9 < roots.attainable[0] < roots.attainable[1] < 350e-9
    with pytest.raises(ToolsError, match="phase-matched wavelength nearest the target"):
        roots.pick("highest")


@pytest.mark.parametrize(
    "gas, target",
    [
        # Argon at a bar or two is normal at 800 nm (its zero is at 0.46 bar), where
        # the largest constituent does not exist.
        ("Ar", RDW(PUMP, 200e-9, peak_power=2e9, fwhm=10e-15)),
        # Helium is anomalous at 800 nm, and has no phase-matched wave on the red side.
        ("He", RDW(PUMP, 2e-6)),
    ],
)
def test_a_target_undefined_everywhere_says_so(gas, target):
    roots = solve_parameter(pressure_builder(gas), (1.0, 2.0), target)
    assert roots.roots == ()
    assert roots.attainable is None
    with pytest.raises(ToolsError, match="undefined across it"):
        roots.pick()


def test_an_index_model_short_of_the_window_leaves_the_range_undefined(monkeypatch):
    def short(*arguments, **keywords):
        raise ToolsError("the phase mismatch is not finite")

    monkeypatch.setattr(tools, "lambda_rdw_all", short)
    roots = solve_parameter(pressure_builder("He"), (1.0, 2.0), RDW(PUMP, 120e-9))
    assert roots.attainable is None


@pytest.mark.medium
def test_an_unreachable_zero_reports_the_zeros_it_reaches():
    fill = capillary("He")
    roots = solve_parameter(
        lambda pressure: tools._capillary_at(fill, pressure), (0.1, 2.0), ZDW(250e-9)
    )
    assert roots.attainable is not None
    # The helium capillary's zero moves from the blue towards 560 nm as the pressure
    # rises, never reaching 250 nm.
    assert 250e-9 < roots.attainable[0] < roots.attainable[1] < 600e-9
    # 256 samples, then 32 rebuilt for the report: 17 linear and 17 logarithmic
    # values, sharing the two ends.
    assert roots.builds == 256 + 32


# ---------------------------------------------------------------- the resonance


def resonant(value):
    """A mode whose dispersion at the pump is normal across a resonance at 2."""
    detuning = (value - 2.0) / 0.2
    return SpecifiedMode.from_taylor(
        [0.0, 0.0, 1e-27 * (3.0 / (1.0 + detuning**2) - 1.0)], OMEGA0
    )


def test_a_resonance_gives_both_of_its_roots():
    roots = solve_parameter(resonant, (1.0, 3.0), ZDW(PUMP))
    # The dispersion vanishes where 3 / (1 + x^2) = 1, x = +-sqrt(2), at
    # 2 -+ 0.2 sqrt(2). The Richardson difference of an exact quadratic is good to
    # the rounding of beta1, which the slope carries to the roots.
    expected = (2.0 - 0.2 * math.sqrt(2.0), 2.0 + 0.2 * math.sqrt(2.0))
    assert roots.roots == pytest.approx(expected, rel=1e-9)


@pytest.mark.medium
def test_a_wall_resonance_gives_a_zero_on_each_side():
    radius = 15e-6
    fill = tools._resolve_fill("He", 1.0, 293.15, MaterialSelection())

    def build(wall):
        # Two terms of the fits keep the first resonance and cost a quarter of eight.
        lattice = TubeLattice(wall, 0.6 * radius + wall, 6, n_terms=2)
        return VincettiMode(
            radius, None, 0.0, mixture=fill, lattice=lattice, cladding=GLASS
        )

    roots = solve_parameter(build, (250e-9, 500e-9), ZDW(PUMP))
    # The first resonance of an 800 nm wave in a wall of index 1.45 is at
    # t = 800 nm / (2 sqrt(1.45^2 - 1)) = 381 nm. The dispersion is anomalous far
    # from it on both sides and normal beside it, so its zeros come in pairs: one on
    # each side, and further ones inside the resonance band.
    resonance = PUMP / (2.0 * math.sqrt(GLASS**2 - 1.0))
    assert len(roots) % 2 == 0
    assert roots.pick("lowest") < resonance < roots.pick("highest")
    for wall in (roots.pick("lowest"), roots.pick("highest")):
        # Brent stops at 2e-12 of the wall; beta2 is good to 5e-9 (modes.base).
        assert tools._zdw_near(build(wall), OMEGA0, 0.0) == pytest.approx(
            PUMP, rel=1e-6
        )
    # Between consecutive roots the sign alternates: none was dropped between two.
    ends = (250e-9, *roots.roots, 500e-9)
    signs = [
        np.sign(base.beta2(build(0.5 * (a + b)), np.array([OMEGA0]))[0])
        for a, b in pairwise(ends)
    ]
    assert signs[0] == signs[-1] == -1.0
    assert all(a == -b for a, b in pairwise(signs))


# ---------------------------------------------------------------- the refusals


def test_design_roots_pick_and_iterate():
    roots = DesignRoots((1.0, 2.0, 3.0), (0.5, 4.0), ZDW(PUMP), 260, "")
    assert list(roots) == [1.0, 2.0, 3.0]
    assert len(roots) == 3
    assert roots.pick() == 1.0
    assert roots.pick("highest") == 3.0
    assert roots.pick(1) == 2.0
    assert roots.pick(-3) == 1.0
    with pytest.raises(
        ToolsError, match=r"root index 3 is out of range: 3 roots, at 1, 2, 3"
    ):
        roots.pick(3)
    with pytest.raises(ToolsError, match="out of range"):
        roots.pick(-4)
    with pytest.raises(ToolsError, match="'lowest', 'highest' or an index"):
        roots.pick(True)


@pytest.mark.parametrize(
    "build_target, message",
    [
        (
            (taylor, SolitonOrder(2.0, sech(energy=1e-6))),
            "needs the fill's nonlinear coefficient",
        ),
        ((lambda value: (taylor(value), 1e-3, 0.0), ZDW(PUMP)), "a tuple of 3"),
        ((lambda value: 1.0, ZDW(PUMP)), "returned float, not a mode"),
        (
            (lambda value: (taylor(value), -1.0), ZDW(PUMP)),
            "gamma must be finite and positive",
        ),
    ],
)
def test_a_builder_must_return_a_mode_and_a_coefficient_where_needed(
    build_target, message
):
    build, target = build_target
    with pytest.raises(ToolsError, match=message):
        solve_parameter(build, (1.0, 2.0), target)


@pytest.mark.parametrize(
    "search, z, message",
    [
        ((2.0, 1.0), 0.0, "search must be finite and ordered"),
        ((math.nan, 1.0), 0.0, "search must be finite and ordered"),
        ((1.0, 2.0), math.inf, "z must be finite"),
    ],
)
def test_a_search_range_and_position_are_validated(search, z, message):
    with pytest.raises(ToolsError, match=message):
        solve_parameter(taylor, search, ZDW(PUMP), z=z)


@pytest.mark.parametrize(
    "make, message",
    [
        (lambda: ZDW(-1e-6), "wavelength must be finite and positive"),
        (lambda: ZDW(math.nan), "wavelength must be finite and positive"),
        (lambda: RDW(PUMP, PUMP), "cannot be at the pump"),
        (lambda: RDW(0.0, 200e-9), "pump must be finite and positive"),
        (lambda: RDW(PUMP, 200e-9, peak_power=1e9), "largest constituent needs fwhm"),
        (lambda: RDW(PUMP, 200e-9, fwhm=1e-14), "fwhm requires peak_power"),
        (
            lambda: SolitonOrder(0.0, sech(energy=1e-6)),
            "order must be finite and positive",
        ),
        (
            lambda: SolitonOrder(
                1.0, GaussianPulse(wavelength=PUMP, fwhm=1e-14, energy=1e-6, m=2)
            ),
            "super-Gaussian",
        ),
        (
            lambda: SolitonOrder(1.0, sech(energy=1e-6, phase=[0.0, 0.0, 1e-30])),
            "unchirped",
        ),
    ],
)
def test_targets_refuse_what_they_do_not_define(make, message):
    with pytest.raises(ToolsError, match=message):
        make()


def test_the_inverse_helpers_refuse_bad_numbers():
    with pytest.raises(ToolsError, match="order must be finite and positive"):
        tools.soliton_peak_power(-1.0, 10e-15, 1e-9, -1e-30)
    with pytest.raises(ToolsError, match="beta2 must be finite and non-zero"):
        tools.soliton_peak_power(1.0, 10e-15, 1e-9, 0.0)
    with pytest.raises(ToolsError, match="peak_power must be finite and positive"):
        tools.pulse_energy(0.0, 10e-15)
