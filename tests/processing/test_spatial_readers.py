# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The readers that take a column for a mode refuse a spatial file by name.

A radial, Cartesian, line or planar file stores fields in V m or V/m, or slab
amplitudes, in its columns. The legacy energy, power and density readers and the plots
drawn from them would read those columns as modal amplitudes in sqrt(W): a launch of
3 nJ read back as 4.8e-15 J. Each refuses instead, names the spatial functional to use,
and leaves the run's statistics and every modal file as they were.
"""

import numpy as np
import pytest

from solprop import capabilities
from solprop._host import ObserverSpend
from solprop.driver import Simulation, SimulationSettings
from solprop.driver.plane_view import PlaneView
from solprop.exceptions import OutputError, ProcessingError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.output.capability import COLUMN_READINGS, reading_refusal
from solprop.plotting import plot_energy, prop_2d, spec_1d, time_1d
from solprop.processing import (
    arrival_time,
    coherence_time,
    energy,
    field_autocorrelation,
    field_plane,
    fwhm_f,
    fwhm_t,
    get_field,
    get_phase,
    get_power,
    get_spectral_density,
    get_spectrum,
    intensity_autocorrelation,
    peak_power,
    time_bandwidth,
)
from solprop.processing._examples import example_general_output, example_output
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    on_axis_spectrum,
    planar_mode_energy,
    spatial_energy,
    spatial_power,
)
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, PIController

# Coarse boxes and short windows, kept small for speed, hold more than a millionth of
# the energy at their edges; the boundary monitor's warning is not what this tests.
pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

ENERGY = 3e-9
HALF_GAP = 63.5e-6

READERS = {
    "energy": energy,
    "peak_power": peak_power,
    "get_power": get_power,
    "get_spectral_density": get_spectral_density,
    "fwhm_t": fwhm_t,
    "fwhm_f": fwhm_f,
    "time_bandwidth": time_bandwidth,
    "arrival_time": arrival_time,
    "field_autocorrelation": field_autocorrelation,
    "intensity_autocorrelation": intensity_autocorrelation,
    "coherence_time": coherence_time,
    "prop_2d": prop_2d,
    "time_1d": time_1d,
    "spec_1d": spec_1d,
    "plot_energy": plot_energy,
}
"""Every reader of :data:`COLUMN_READINGS`, by the name its refusal gives."""

KINDS = {
    "radial": "radial",
    "cartesian": "Cartesian",
    "line": "line",
    "planar": "planar",
}
"""How the refusal names each route's run."""

SPECTRAL = ("get_spectral_density", "fwhm_f", "time_bandwidth", "coherence_time")
"""Readers whose advice is the route's spectral functional."""


def _settings(length: float) -> SimulationSettings:
    steps = 2
    return SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[length],
        dz_max=length / steps,
        dz_init=length / steps,
        schedule=FixedSchedule(np.full(steps, length / steps)),
        backend="numpy_ref",
        stats=StatsSpec(names=("energy",)),
        # The coarse box launches into the safety ramp's band (SEP 40 section 5.3).
        spatial_absorber=False,
    )


def _simulation(route: str) -> Simulation:
    length = 1e-3
    if route == "planar":
        grid = EnvelopeGrid.from_limits(length, 800e-9, (650e-9, 1000e-9), 300e-15)
        space = PlanarGrid([SlabMode(HALF_GAP, "Ar", 1.0, order=1)], 1e-3, 8)
        pulse = PlanarPulse(
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY),
            400e-6,
            waist_x=0.735 * HALF_GAP,
        )
        settings = _settings(length)
        return Simulation.from_planar(grid, space, [], pulse, settings=settings)
    grid = FieldGrid.from_limits(length, 800e-9, (600e-9, 1.2e-6), 400e-15)
    space = {
        "radial": RadialGrid(6e-4, 16),
        "cartesian": FreeGrid(4e-4, 8, 4e-4, 8),
        "line": LineGrid(4e-4, 16, collar_fraction=0.0),
    }[route]
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=ENERGY), 6e-5
    )
    build = Simulation.from_radial if route == "radial" else Simulation.from_free
    return build(
        grid,
        space,
        SpatialMedium("Ar", 1.0),
        [],
        pulse,
        settings=_settings(length),
        noise=None,
    )


@pytest.fixture(scope="module", params=list(KINDS))
def spatial_run(request):
    simulation = _simulation(request.param)
    return request.param, simulation, simulation.run(progress="none")


@pytest.mark.parametrize("name", list(READERS))
def test_each_column_reader_refuses_a_spatial_file_by_name(spatial_run, name):
    route, _, output = spatial_run
    with pytest.raises(OutputError) as caught:
        READERS[name](output)
    message = str(caught.value)
    assert message.startswith(
        f"{name} measures modal amplitudes and this is a {KINDS[route]} run, whose "
        "columns are not: "
    )
    assert message == reading_refusal(output.metadata, name, recorded=output.stat_names)
    # The run recorded its energy and no other beam statistic, and the refusal points
    # to that one alone.
    assert "the run's own beam statistics are energy." in message
    assert "peak_intensity" not in message


def test_every_column_reader_is_tested():
    assert set(READERS) == set(COLUMN_READINGS)


def test_the_stored_columns_and_the_statistics_still_read(spatial_run):
    route, simulation, output = spatial_run
    stored = output.fields("fft")
    n_columns = stored.shape[1]
    omega, spectrum = get_spectrum(output, plane=-1)
    assert spectrum.shape[-1] == omega.size
    assert get_phase(output, plane=-1)[1].shape == spectrum.shape
    assert get_field(output, plane=-1)[1].shape[0] == n_columns
    # The run's own energy is the spatial functional of the stored plane, and the
    # launch's at the entrance: unchanged by the refusal of the readers.
    space, medium = simulation.spatial_grid, simulation.medium
    flux = SpatialFlux.resolve(simulation.grid, space, medium, z=float(output["z"][-1]))
    recorded = float(np.ravel(output["stats/energy"])[-1])
    # The same functional of the same stored array: equal to a few rounding errors of
    # its sums (2e-16 measured), far inside the bound.
    assert recorded == pytest.approx(spatial_energy(stored[-1], flux), rel=1e-12)
    if route != "planar":
        # Linear propagation in lossless argon is a phase per column, so the energy
        # moves only by the rounding of its Parseval sums (6e-16 measured). The planar
        # launch couples part of the beam into no slab mode, and is left out.
        assert recorded == pytest.approx(ENERGY, rel=1e-12)


def test_a_plane_view_of_a_spatial_run_refuses_its_power_and_density(spatial_run):
    _, _, output = spatial_run
    plane = next(iter(output.planes()))
    view = PlaneView(plane, 0, output.metadata, None, ObserverSpend())
    try:
        with pytest.raises(OutputError, match=r"^get_power measures modal amplitudes"):
            view.power()
        with pytest.raises(
            OutputError, match=r"^get_spectral_density measures modal amplitudes"
        ):
            view.spectral_density()
        assert view.spectrum()[1].shape[0] == output.fields().shape[1]
    finally:
        view._close()


def test_the_advice_names_only_what_works_on_the_route(spatial_run):
    route, simulation, output = spatial_run
    stored = output.fields("fft")[-1]
    flux = SpatialFlux.resolve(
        simulation.grid,
        simulation.spatial_grid,
        simulation.medium,
        z=float(output["z"][-1]),
    )
    recorded = float(np.ravel(output["stats/energy"])[-1])
    advice = {name: reading_refusal(output.metadata, name) or "" for name in READERS}
    # What every route is pointed to, called on its saved plane.
    assert spatial_power(stored, flux).shape == (simulation.grid.n_t,)
    assert "solprop.spatial.spatial_power" in advice["get_power"]
    assert "get_field reads a column's field" in advice["time_1d"]
    assert get_field(output, plane=-1)[1].shape[0] == stored.shape[0]
    if route == "planar":
        assert "planar_mode_energy" in advice["energy"]
        # The slab modes' energies sum to the beam's: two sums of the same terms in a
        # different order, a few rounding errors apart (bound 1e-12).
        modal = planar_mode_energy(stored, flux)
        assert float(np.sum(modal)) == pytest.approx(recorded, rel=1e-12)
        for name in SPECTRAL:
            assert (
                "no spatial functional gives a planar run's spectrum yet"
                in (advice[name])
            )
            assert "field_plane" not in advice[name]
            assert "on_axis_spectrum" not in advice[name]
        with pytest.raises(ProcessingError, match="planar planes are not exchanged"):
            field_plane(output)
        return
    assert "planar_mode_energy" not in advice["energy"]
    for name in SPECTRAL:
        assert "field_plane" in advice[name] and "on_axis_spectrum" in advice[name]
    # The plane's energy per bin sums to the beam's energy: the same weighted terms
    # accumulated per bin rather than in total, a few rounding errors apart.
    per_bin = field_plane(output).spectral_energy()
    assert float(np.sum(per_bin)) == pytest.approx(recorded, rel=1e-12)
    assert on_axis_spectrum(stored, flux).shape == (simulation.grid.n_omega,)


def test_the_file_statement_keeps_the_statistic_and_states_the_readers(spatial_run):
    _, _, output = spatial_run
    statement = capabilities.output(output)
    assert statement.basis == "spatial"
    # The energy statistic is the beam's physical flux, and stays admitted.
    assert statement.admits("energy") and statement.admits("peak_intensity")
    assert set(statement.refused_readers) == set(COLUMN_READINGS)
    for name, text in statement.refused_readers.items():
        assert text == reading_refusal(
            output.metadata, name, recorded=output.stat_names
        )


def test_a_spatial_spectrum_refuses_the_floor_without_pointing_to_a_refused_reader(
    spatial_run,
):
    _, _, output = spatial_run
    with pytest.raises(ProcessingError) as caught:
        get_spectrum(output, noise_floor="subtract")
    message = str(caught.value)
    assert "a spatial file has no reader of the floor-subtracted density" in message
    assert "ask get_spectral_density" not in message


@pytest.mark.parametrize("build", [example_output, example_general_output])
def test_a_modal_file_keeps_its_column_readers(build):
    output = build()
    metadata = output.metadata
    assert metadata is not None
    # The general modal basis keeps refusing its two legacy metrics and the plot of
    # the energy, as before, and admits every other reader; the orthogonal basis
    # admits all of them.
    refused = {
        name for name in COLUMN_READINGS if reading_refusal(metadata, name) is not None
    }
    general = {"energy", "peak_power", "plot_energy"}
    assert refused == (set() if build is example_output else general)
    assert set(capabilities.output(output).refused_readers) == refused
    if build is example_general_output:
        with pytest.raises(OutputError, match=r"^energy is unavailable"):
            plot_energy(output)
    t, power = get_power(output, plane=0)
    assert power.shape[-1] == t.size
    assert np.all(fwhm_t(output, plane=0) > 0.0)
