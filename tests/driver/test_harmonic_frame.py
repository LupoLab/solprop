# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The frame an envelope grid's third harmonic asks for, chosen by one rule.

An envelope grid keeps ``beta_0``, and measures its frame delay from zero frequency,
exactly when an electronic response radiates the harmonic
(:func:`solprop.translate.responses.declares_harmonic_carrier`). Every route chooses its
frame by that rule before anything compiles; ``assemble`` checks it against the
compiled carrier, and refuses a frame given with ``subtract_beta0=True`` beside it. A
Raman response's ``thg`` is ignored on an envelope grid, so it keeps nothing.
"""

import dataclasses

import numpy as np
import pytest

from solprop.backends.exponential import carrier_reference
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import TranslationError
from solprop.grid import EnvelopeGrid, FieldGrid, RadialGrid
from solprop.linear import Frame
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear.raman import DampedOscillator, RamanResponse, undamped_dephasing
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import Multistep, PIController
from solprop.translate.responses import declares_harmonic_carrier

# The hydrogen index fit does not reach the window's edges; those bins are not the ones
# this module reads.
pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

LENGTH = 0.05
WAVELENGTH = 800e-9
LIMITS = (400e-9, 2e-6)
WINDOW = 400e-15
EPS = np.finfo(float).eps


def envelope_grid(*, thg: bool) -> EnvelopeGrid:
    """Return the envelope grid of the capillary cases, oversampled for a harmonic."""
    return EnvelopeGrid.from_limits(LENGTH, WAVELENGTH, LIMITS, WINDOW, thg=thg)


def capillary(*, thg: bool, raman: bool, **changes) -> Simulation:
    """Return a hydrogen capillary on an envelope grid, without plasma or noise."""
    return Simulation.from_capillary(
        100e-6,
        LENGTH,
        "H2",
        5.0,
        grid=envelope_grid(thg=thg),
        raman=raman,
        thg=thg,
        wavelength=WAVELENGTH,
        fwhm=30e-15,
        energy=5e-6,
        n_save=2,
        noise=None,
        backend="numpy_ref",
        plasma=False,
        **changes,
    )


def frame_origin_of(simulation: Simulation) -> object:
    """Return the frequency origin a run's plane-reference record gives its frame."""
    record = simulation.metadata().reference
    assert record is not None
    frame = record["frame"]
    assert isinstance(frame, dict)
    return frame["frequency_origin"]


def oscillator_raman(*, thg: bool) -> RamanResponse:
    """Return a one-line Raman response; only its ``thg`` flag matters here."""
    return RamanResponse((DampedOscillator(1e-8, 2e14, undamped_dephasing()),), thg=thg)


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        ([KerrResponse(1.0, thg=True)], True),
        ([KerrResponse(1.0, thg=False)], False),
        ([KerrResponse(1.0, thg=False), oscillator_raman(thg=True)], False),
        ([oscillator_raman(thg=True)], False),
        ([], False),
    ],
)
def test_declares_harmonic_carrier_reads_the_electronic_responses(responses, expected):
    assert declares_harmonic_carrier(responses, envelope_grid(thg=True)) is expected


def test_declares_harmonic_carrier_is_false_on_a_field_grid():
    # A field grid keeps beta_0 whatever it declares and compiles no separate carrier.
    grid = FieldGrid.from_limits(LENGTH, WAVELENGTH, LIMITS, WINDOW)
    assert not declares_harmonic_carrier([KerrResponse(1.0, thg=True)], grid)


def test_a_beta0_removing_frame_beside_the_harmonic_is_refused():
    simulation = capillary(thg=True, raman=False)
    simulation.frame = Frame(simulation.grid.omega0, subtract_beta0=True)
    with pytest.raises(TranslationError, match="subtract_beta0=True"):
        simulation.translate()


def test_a_beta0_keeping_frame_beside_the_harmonic_is_taken():
    simulation = capillary(thg=True, raman=False)
    simulation.frame = Frame(simulation.grid.omega0, subtract_beta0=False)
    problem = simulation.translate()
    assert problem.responses.harmonic_carrier
    assert problem.grids.carrier_2w0 is not None


def test_a_raman_harmonic_flag_does_not_keep_beta0():
    scalar = Multistep(carrier_frame="scalar")
    base = capillary(thg=False, raman=True, integrator=scalar)
    kerr, raman = base.responses
    assert isinstance(raman, RamanResponse)
    simulation = Simulation(
        base.grid,
        base.modes,
        (kerr, dataclasses.replace(raman, thg=True)),
        base._field0,
        settings=base.settings,
    )
    problem = simulation.translate()
    assert not problem.responses.harmonic_carrier
    grids = problem.grids
    at = int(np.argmin(np.abs(grids.omega - grids.omega0)))
    span = 1e-3
    rate = (
        np.imag(
            np.ravel(
                problem.linear.polynomial(span) - problem.linear.polynomial(0.0)
            ).reshape(-1, grids.omega.size)[0, at]
        )
        / span
    )
    # Removing beta_0 makes the operator vanish at the carrier, up to the rounding of
    # the cancellation of beta itself, about omega0 / c in a dilute gas; the margin
    # covers the table's interpolation, well below the 193 rad/m that keeping beta_0
    # leaves.
    assert abs(rate) <= 1e3 * EPS * grids.omega0 / C
    reference = carrier_reference(problem, scalar)
    assert reference is not None
    assert "removes beta_0" in reference.reason
    # The run record (SEP 31) measures the frame's delay from the carrier, as the
    # operator does.
    assert frame_origin_of(simulation) == grids.omega0


def test_the_record_measures_a_harmonic_frame_from_zero_frequency():
    simulation = capillary(thg=True, raman=False)
    assert simulation.translate().responses.harmonic_carrier
    assert frame_origin_of(simulation) == 0.0


def spatial_settings(grid: EnvelopeGrid) -> SimulationSettings:
    """Return settings for a construction that is never run."""
    return SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
    )


def beam() -> GaussianPulse:
    """Return the temporal pulse of the spatial constructions."""
    return GaussianPulse(wavelength=WAVELENGTH, fwhm=30e-15, energy=3e-9)


def radial(grid, responses):
    """Return a radial simulation, choosing its frame in ``_from_spatial``."""
    return Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 32),
        SpatialMedium("Ar", 1.0),
        responses,
        SpatialPulse(beam(), 60e-6),
        settings=spatial_settings(grid),
        noise=None,
    )


def planar(grid, responses):
    """Return a planar simulation, choosing its frame in ``from_planar``."""
    half_gap = 63.5e-6
    return Simulation.from_planar(
        grid,
        PlanarGrid([SlabMode(half_gap, "Ar", 1.0, order=1)], 1e-3, 8),
        responses,
        PlanarPulse(beam(), 400e-6, waist_x=0.735 * half_gap),
        settings=spatial_settings(grid),
    )


def spatial_pulse(grid, responses):
    """Return a simulation launching a SpatialPulse; ``__init__`` chooses its frame."""
    space = RadialGrid(6e-4, 32)
    return Simulation(
        grid,
        [],
        responses,
        SpatialPulse(beam(), 60e-6),
        settings=spatial_settings(grid),
        spatial_grid=space,
        medium=SpatialMedium("Ar", 1.0),
    )


@pytest.mark.parametrize("route", [radial, planar, spatial_pulse])
@pytest.mark.parametrize(
    ("responses", "removes_beta0"),
    [
        ([KerrResponse(1.0, thg=True)], False),
        ([KerrResponse(1.0, thg=False), oscillator_raman(thg=True)], True),
    ],
)
def test_the_spatial_routes_choose_the_frame_by_the_same_rule(
    route, responses, removes_beta0
):
    # Only the frame chosen at construction is read; nothing is translated.
    simulation = route(envelope_grid(thg=True), responses)
    assert simulation.frame is not None
    assert simulation.frame.subtract_beta0 is removes_beta0
