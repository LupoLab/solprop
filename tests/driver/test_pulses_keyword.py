# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Multi-colour inputs: the ``pulses=`` keyword and :class:`solprop.pulses.PulseSum`."""

import numpy as np
import pytest
from jaxtyping import TypeCheckError

from march_bounds import FIXED_ORDER, assert_matched, simulation_bound
from solprop import fourier
from solprop.driver import Simulation, prop_capillary, prop_gnlse
from solprop.exceptions import DriverError, PulseError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.pulses import GaussianPulse, PulseSum, SechPulse
from solprop.stepper import FixedSchedule

PUMP = GaussianPulse(wavelength=267e-9, fwhm=35e-15, energy=95e-6)
SEED = GaussianPulse(wavelength=800e-9, fwhm=35e-15, energy=45e-6)


@pytest.fixture(scope="module")
def field_grid():
    return FieldGrid.from_limits(1.0, 800e-9, (150e-9, 3e-6), 1e-12)


def test_sum_is_the_superposition_of_its_members(field_grid):
    total = PulseSum((PUMP, SEED)).build(field_grid)
    expected = PUMP.build(field_grid) + SEED.build(field_grid)
    # One addition per bin: the two spectra are summed exactly as the test sums them.
    np.testing.assert_array_equal(total, expected)


def test_energies_of_separated_members_add(field_grid):
    spectrum = PulseSum((PUMP, SEED)).build(field_grid)
    energy = fourier.energy_omega(spectrum[0, 0], field_grid.dt, kind=field_grid.kind)
    # The members barely overlap in frequency (267 and 800 nm, 35 fs each), so the
    # cross term is below 1e-12 of the total and the energies add to rounding.
    assert energy == pytest.approx(95e-6 + 45e-6, rel=1e-9)


def test_overlapping_members_interfere(field_grid):
    # Two identical in-phase pulses: twice the field, four times one pulse's energy.
    one = GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6)
    spectrum = PulseSum((one, one)).build(field_grid)
    energy = fourier.energy_omega(spectrum[0, 0], field_grid.dt, kind=field_grid.kind)
    assert energy == pytest.approx(4e-6, rel=1e-12)


def test_a_sum_builds_on_an_envelope_grid():
    grid = EnvelopeGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 2e-12)
    delayed = SechPulse(wavelength=900e-9, fwhm=50e-15, energy=1e-7, t0=200e-15)
    spectrum = PulseSum(
        (GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-7), delayed)
    ).build(grid)
    energy = fourier.energy_omega(spectrum[0, 0], grid.dt, kind=grid.kind)
    # Separated by 200 fs and 100 nm: overlap negligible at this tolerance.
    assert energy == pytest.approx(2e-7, rel=1e-6)


def test_an_empty_sum_is_refused():
    with pytest.raises(PulseError, match="at least one pulse"):
        PulseSum(())


def test_a_member_that_cannot_build_is_refused():
    # The runtime type check samples one random tuple element, so it catches the float
    # first on some runs; otherwise PulseSum's own check must name the member.
    with pytest.raises((PulseError, TypeCheckError)) as caught:
        PulseSum((PUMP, 800e-9))  # pyright: ignore[reportArgumentType]
    if caught.type is PulseError:
        caught.match(r"member 1 .* no build\(grid\) method")


class _UncallableBuild:
    """Passes the Pulse protocol check (it has a build attribute) but cannot build."""

    build = "not a method"  # None would fail the protocol check itself


def test_a_member_whose_build_is_not_callable_is_refused():
    with pytest.raises(PulseError, match=r"member 1 .* no build\(grid\) method"):
        PulseSum((PUMP, _UncallableBuild()))  # pyright: ignore[reportArgumentType]


def test_members_of_different_shapes_are_refused(field_grid):
    class Wide:
        def build(self, grid):
            return np.zeros((2, 1, grid.n_omega), dtype=np.complex128)

    with pytest.raises(PulseError, match="different shapes"):
        PulseSum((PUMP, Wide())).build(field_grid)  # pyright: ignore[reportArgumentType]


def test_sum_prints_its_members():
    text = repr(PulseSum((PUMP, SEED)))
    assert text.startswith("PulseSum(GaussianPulse(")
    assert "267 nm" in text and "800 nm" in text
    assert "pulse 1" in str(PulseSum((PUMP, SEED)))


GNLSE = {
    "gamma": 0.11,
    "length": 1e-4,
    "betas": [0.0, 0.0, -1.1830e-26],
    "lambda0": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
    "n_save": 2,
    "progress": "none",
    "backend": "numpy_ref",
}


def test_one_pulse_through_pulses_equals_the_scalar_keywords():
    # On a prescribed schedule at a fixed order, so that the two runs are one march,
    # and without the collar, which refuses ten fixed steps.
    prescribed = {
        "schedule": FixedSchedule.uniform(0.0, 1e-4, 1e-5, save_planes=[]),
        "integrator": FIXED_ORDER,
        "absorber": False,
    }
    scalar = prop_gnlse(
        **{"noise": None, **GNLSE},
        width=28.4e-15,
        peak_power=100.0,
        pulseshape="sech",
        shock=False,
        **prescribed,
    )
    listed = prop_gnlse(
        **{"noise": None, **GNLSE},
        pulses=[SechPulse(wavelength=835e-9, width=28.4e-15, peak_power=100.0)],
        shock=False,
        **prescribed,
    )
    # The same pulse built on the same grid; the sum of one member is that member,
    # so the launches are equal exactly, and the run is then the same march, within
    # its rounding budget (SEP 25).
    np.testing.assert_array_equal(listed["field"][0], scalar["field"][0])
    settings = {key: value for key, value in GNLSE.items() if key != "progress"}
    simulation = Simulation.from_gnlse(
        settings.pop("gamma"),
        settings.pop("length"),
        settings.pop("betas"),
        width=28.4e-15,
        peak_power=100.0,
        pulseshape="sech",
        shock=False,
        **{"noise": None, **settings},
        **prescribed,
    )
    assert_matched(listed["field"], scalar["field"], simulation_bound(simulation))


@pytest.mark.parametrize(
    "extra",
    [
        {"fwhm": 30e-15},
        {"energy": 1e-6},
        {"peak_power": 1.0},
        {"width": 10e-15},
        {"phases": [0.0, 0.0, 1e-30]},
        {"pulseshape": "sech"},
    ],
)
def test_pulses_beside_a_single_pulse_keyword_is_refused(extra):
    with pytest.raises(DriverError, match="leave the single-pulse keywords unset"):
        Simulation.from_gnlse(
            0.11,
            1e-4,
            [0.0, 0.0, -1.1830e-26],
            lambda0=835e-9,
            wavelength_limits=(500e-9, 2e-6),
            trange=1e-12,
            pulses=[PUMP],
            **{"noise": None, **extra},
        )


@pytest.mark.medium
# The helium index is extrapolated past its fitted range at the grid's long end.
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_two_colour_capillary_run_records_both_pulses():
    run = prop_capillary(
        75e-6,
        1e-3,
        "He",
        0.5,
        lambda0=267e-9,
        pulses=[PUMP, SEED],
        wavelength_limits=(150e-9, 3e-6),
        trange=400e-15,
        n_save=2,
        progress="none",
        backend="numpy_ref",
        noise=None,
    )
    assert run.metadata is not None
    recorded = run.metadata.configuration["pulse"]
    assert isinstance(recorded, PulseSum)
    assert recorded.pulses == (PUMP, SEED)
    start = float(np.sum(run["stats/energy"][0]))
    # A millimetre of low-pressure helium keeps nearly all of both pulses.
    assert start == pytest.approx(95e-6 + 45e-6, rel=1e-3)
