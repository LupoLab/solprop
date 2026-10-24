# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The hybrid planar front end: ``prop_planar`` and ``Simulation.from_planar``.

Every run is at most ten fixed steps of an argon-filled gap on ``numpy_ref``;
``numba_cpu`` (``tests/backends/test_numba_planar.py``) and Kokkos
(``tests/backends/test_planar_native.py``) execute it as well.
Energies are in J: the geometry integrates over y.
"""

import json
import math
import warnings

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from solprop import fourier
from solprop.boundaries import absorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings, prop_planar
from solprop.driver.executors import Local
from solprop.driver.scan import Scan, ScanPoint, run_scan
from solprop.exceptions import (
    DriverError,
    GridError,
    MaterialRangeWarning,
    NoiseError,
    ProcessingError,
    PulseError,
    TranslationError,
)
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid
from solprop.linear import Frame
from solprop.modes import PlanarGrid, SlabMode, gaussian_coupling
from solprop.noise import spatial_noise
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse
from solprop.output import open_file
from solprop.physdata import Mixture, gamma3
from solprop.processing.scans import (
    energy_in_band,
    energy_in_window,
    field_at_plane,
    mode_energies,
    peak_power,
    spectrum_in_band,
)
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    beam_radius,
    critical_power,
    fwhm_r,
    on_axis_intensity,
    on_axis_spectrum,
    planar_fluence,
    planar_intensity,
    planar_mode_energy,
    spatial_energy,
    spatial_fluence,
    spatial_intensity,
)
from solprop.stats import analytic_field, default_spec
from solprop.stepper import PIController
from solprop.stepper.schedule import FixedSchedule

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
]

LENGTH = 0.1
HALF_GAP = 63.5e-6
GRID = EnvelopeGrid.from_limits(LENGTH, 800e-9, (650e-9, 1000e-9), 600e-15)


FILL = SpatialMedium("Ar", 1.0)
"""The slab modes' own fill, which measures their plane (SEP 34)."""


def measure(space, medium=FILL):
    """Return the plane measure of a planar or free-space grid at the entrance."""
    return SpatialFlux.resolve(GRID, space, medium)


def slab_modes(orders=(1, 3, 5), **fill):
    return [SlabMode(HALF_GAP, "Ar", 1.0, order=p, **fill) for p in orders]


def settings(backend="numpy_ref", steps=10, **changes):
    return SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[LENGTH],
        dz_max=LENGTH / steps,
        dz_init=LENGTH / steps,
        schedule=FixedSchedule(np.full(steps, LENGTH / steps)),
        backend=backend,
        **changes,
    )


def pulse(energy=1e-3):
    return PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=energy),
        400e-6,
        waist_x=0.735 * HALF_GAP,
    )


def kerr():
    # Per particle: the run scales it by the fill's number density. optmat's argon
    # coefficient at the 800 nm carrier (a KerrResponse is the caller's, SEP 3 4b).
    return KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)


@pytest.mark.medium
def test_prop_planar_runs_a_launch_and_records_the_slab_geometry(tmp_path):
    """Ten Kerr steps through prop_planar into HDF5: record, units, statistics."""
    space = PlanarGrid(slab_modes(), 1.5e-3, 32)
    path = tmp_path / "planar.h5"
    prop_planar(
        GRID,
        space,
        [kerr()],
        pulse(),
        settings=settings(stats=default_spec("free")),
        output=path,
        progress="none",
    )
    with open_file(path) as output:
        check_the_record(output, space)
    with h5py.File(path, "r") as file:
        assert file.attrs["simulation_kind"] == "slab"
        assert file.attrs["field_units"] == "sqrt(W/m)"
        assert file.attrs["transverse_spectral_axis"] == "p,k_y (p fastest)"


def launched_energy(space, energy=1e-3):
    """Return the slab modes' share of the launch's physical energy, independently.

    The launch gives the incident astigmatic Gaussian ``energy`` of axial flux in the
    fill: each y column and bin weighted by the bulk admittance averaged over the
    beam's k_x spectrum ``exp(-k_x^2 w_x^2 / 2)``. Mode p takes the coupled share
    of the amplitude and is measured by its own admittance ``c Re(q)/omega``,
    ``q = sqrt((omega n_eff/c)^2 - k_y^2)`` (SEP 34 section 4.3). The k_x average is
    a 200-node Gauss-Hermite rule, converged far below rounding for an integrand
    analytic out to the cutoff at k_x ~ k.
    """
    launch = pulse(energy)
    omega = GRID.omega
    live = GRID.active_mask & (omega > 0)
    temporal = launch.temporal.build(GRID)[0, 0]
    gamma = fourier.energy_weights(GRID.n_omega, GRID.dt, kind=GRID.kind)
    profile = np.fft.fft(np.exp(-((space.y / launch.waist) ** 2)))
    density = np.abs(profile[:, None]) ** 2 * (gamma * np.abs(temporal) ** 2)[None, :]
    density = density[:, live]
    k_y = space.k_y[:, None]
    wave = omega[live][None, :] / C

    def weight(square):
        root = np.sqrt(square.astype(complex))
        return np.where(np.real(square) > 0, np.real(root) / wave, 0.0)

    index = FILL.index(omega[live])[None, :]
    nodes, weights = np.polynomial.hermite.hermgauss(200)
    incident = sum(
        w
        * weight(
            (wave * index) ** 2 - k_y**2 - (math.sqrt(2) * x / launch.waist_x) ** 2
        )
        for x, w in zip(nodes, weights, strict=True)
    ) / math.sqrt(math.pi)
    reference = float(np.sum(density * incident))
    shares = gaussian_coupling(space.modes, launch.waist_x)
    modal = [
        float(
            np.sum(
                density * weight((wave * mode.neff(omega[live])[None, :]) ** 2 - k_y**2)
            )
        )
        for mode in space.modes
    ]
    return energy * float(np.sum(shares * np.asarray(modal))) / reference


def check_the_record(output, space):
    """Check a planar run's record, units, energies and statistics."""
    metadata = output.metadata
    assert metadata is not None
    assert metadata.simulation_kind == "slab"
    assert metadata.transverse_measure == "area"
    spatial = metadata.spatial
    assert spatial is not None
    assert spatial["kind"] == "slab"
    assert tuple(spatial["spectral_axis_names"]) == ("k_y",)
    assert spatial["column_order"] == "p + n_p*iy"
    assert spatial["stored_field_units"] == "sqrt(W/m)"
    fields = output.fields("fft")
    assert fields.shape == (2, space.n_columns, 1, GRID.n_omega)
    energy = np.ravel(output["stats/energy"])
    assert_allclose(energy[0], launched_energy(space), rtol=1e-12)
    # TE1 carries 98.8 % and loses 0.149 /m of field: the energy falls, a little.
    assert 0.95 * energy[0] < energy[-1] < energy[0]
    per_mode = planar_mode_energy(fields[-1], measure(space))
    assert_allclose(float(np.sum(per_mode)), energy[-1], rtol=1e-12)
    assert per_mode[0] > 100 * per_mode[1]
    fluence = planar_fluence(fields[0], measure(space))
    assert fluence.shape == (space.n_nodes,)
    # The peak fluence is on the mid-plane row, at the beam centre.
    assert int(np.argmax(fluence)) // space.x.size == space.n_y // 2
    for name in ("peak_intensity", "on_axis_intensity", "beam_radius", "fwhm_r"):
        assert np.all(np.isfinite(output[f"stats/{name}"]))
    # A centred launch: the axis is the peak, so the two intensities agree to the
    # temporal sampling of the peak (the axis is reconstructed, not a node).
    assert_allclose(
        np.ravel(output["stats/on_axis_intensity"]),
        np.ravel(output["stats/peak_intensity"]),
        rtol=1e-2,
    )


def test_the_simulation_builds_the_fills_medium_and_frame():
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    simulation = Simulation.from_planar(GRID, space, [], pulse(), settings=settings())
    assert isinstance(simulation.medium, SpatialMedium)
    assert_allclose(
        simulation.medium.density(np.array([0.0])),
        SlabMode(HALF_GAP, "Ar", 1.0).density(np.array([0.0])),
        rtol=1e-15,
    )
    assert simulation.frame is not None
    assert "PlanarGrid" in repr(simulation)
    problem = simulation.translate()
    assert problem.transform.geometry is not None
    assert problem.transform.geometry.kind == "slab"
    vacuum = PlanarGrid([SlabMode(HALF_GAP)], 1e-3, 8)
    empty = Simulation.from_planar(GRID, vacuum, [], pulse(), settings=settings())
    assert empty.medium is not None
    assert empty.medium.material == 1.0


def test_a_planar_run_has_no_input_port():
    """Slab columns per y frequency are not exchanged in this release (SEP 31)."""
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    simulation = Simulation.from_planar(GRID, space, [], pulse(), settings=settings())
    with pytest.raises(DriverError, match="planar planes are not exchanged"):
        simulation.input_port()


def test_a_planar_run_binds_an_optmat_rate_to_the_inputs_peak_field():
    """SEP 3, Q14 on the planar route, as on the free-space routes."""
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    rate = OptmatRate("Ar", 800e-9)
    assert rate.reference_field is None
    simulation = Simulation.from_planar(
        GRID, space, [EnvelopePlasmaResponse("Ar", rate)], pulse(), settings=settings()
    )
    (response,) = simulation.responses
    assert isinstance(response, EnvelopePlasmaResponse)
    bound = response.rate
    assert isinstance(bound, OptmatRate)
    # The rate is driven by the physical field, not the axial flux (SEP 34): the
    # largest analytic |E| at any node and time, the slab modes synthesised in x
    # over the inverse y transform of the stored (p, k_y) amplitudes.
    field0 = simulation._field0
    rows = np.fft.ifft(field0[:, 0].reshape(space.n_y, space.n_p, -1), axis=0)
    physical = np.einsum("xp,ypw->yxw", space.synthesis, rows).reshape(-1, GRID.n_omega)
    peak = float(np.max(np.abs(analytic_field(GRID, physical[:, None, :]))))
    assert bound.reference_field == pytest.approx(peak, rel=1e-13)
    assert bound.ionisation_potential == float(
        bound.optmat_model.energy_per_event(bound.effective_reference_field)
    )
    explicit = rate.bound(1e10)
    kept = Simulation.from_planar(
        GRID,
        space,
        [EnvelopePlasmaResponse("Ar", explicit)],
        pulse(),
        settings=settings(),
    )
    (held,) = kept.responses
    assert isinstance(held, EnvelopePlasmaResponse)
    assert held.rate is explicit


@pytest.mark.parametrize("bind", [False, True])
def test_a_planar_input_above_a_rates_maximum_field_warns(bind):
    # Bound or not, the rate checks the input: 0.1 J here peaks at 1.5e11 V/m, above
    # optmat's argon maximum field of 8.6e10 V/m.
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    rate = OptmatRate("Ar", 800e-9)
    rate = rate.bound(1e10) if bind else rate
    with pytest.warns(MaterialRangeWarning, match="exceeds the maximum field"):
        Simulation.from_planar(
            GRID,
            space,
            [EnvelopePlasmaResponse("Ar", rate)],
            pulse(energy=0.1),
            settings=settings(),
        )


def test_the_planar_intensity_integrates_to_the_fluence_and_the_energy():
    space = PlanarGrid(slab_modes(), 1e-3, 8)
    field = pulse().build(GRID, space, medium=FILL)
    flux = measure(space)
    intensity = planar_intensity(field, flux)
    assert intensity.shape == (space.x.size * space.n_y, GRID.n_t)
    # Only the summation's rounding separates them: the slab rule integrates every
    # product of the modes exactly, so Higham's gamma of the longest sum bounds both.
    terms = intensity.size
    bound = terms * np.finfo(float).eps
    assert 0 < bound < 1e-8
    assert_allclose(
        GRID.dt * np.sum(intensity, axis=-1),
        planar_fluence(field, flux),
        rtol=bound,
        atol=0.0,
    )
    rows = intensity.reshape(space.n_y, space.x.size, GRID.n_t)
    energy = GRID.dt * space.dy * np.einsum("yxt,x->", rows, space.x_weights)
    assert_allclose(
        energy, np.sum(planar_mode_energy(field, flux)), rtol=bound, atol=0.0
    )
    with pytest.raises(GridError, match="planar spectrum"):
        planar_intensity(field[:-1], flux)


def test_a_gas_path_and_a_mixture_are_refused_by_name():
    varying = [SlabMode(HALF_GAP, "Ar", Linear(1.0, 0.5, LENGTH))]
    with pytest.raises(DriverError, match="requires constant media"):
        Simulation.from_planar(
            GRID, PlanarGrid(varying, 1e-3, 8), [], pulse(), settings=settings()
        )
    mixed = [SlabMode(HALF_GAP, mixture=Mixture.fill({"Ar": 0.5, "He": 0.5}, 1.0))]
    with pytest.raises(DriverError, match="single-gas fill"):
        Simulation.from_planar(
            GRID, PlanarGrid(mixed, 1e-3, 8), [], pulse(), settings=settings()
        )


def test_an_input_that_is_not_a_planar_launch_is_refused_by_name():
    space = PlanarGrid(slab_modes((1, 3)), 1e-3, 8)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3)
    with pytest.raises(DriverError, match="launches a PlanarPulse"):
        Simulation.from_planar(GRID, space, [], temporal, settings=settings())
    with pytest.raises(DriverError, match="shape"):
        Simulation.from_planar(
            GRID,
            space,
            [],
            np.zeros((3, 1, GRID.n_omega), complex),
            settings=settings(),
        )
    with pytest.raises(DriverError, match="launches a PlanarPulse"):
        Simulation.from_planar(
            GRID,
            space,
            [],
            SpatialPulse(temporal, 1e-4),  # pyright: ignore[reportArgumentType] - refused
            settings=settings(),
        )
    with pytest.raises(PulseError, match="use PlanarPulse"):
        SpatialPulse(temporal, 1e-4).build(GRID, space, medium=FILL)
    with pytest.raises(PulseError, match="launches into a PlanarGrid"):
        pulse().build(GRID, FreeGrid(1e-3, 4, 1e-3, 4), medium=FILL)


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({"propz": 1e-3}, "propz must be 0"),
        ({"jones": (1.0, 1.0j)}, "jones takes one entry"),
        ({"offset_x": math.inf}, "offset_x must be finite"),
        ({"waist_x": 0.0}, "waist_x must be a finite positive number"),
    ],
)
def test_a_planar_pulse_refuses_what_it_does_not_carry(keywords, match):
    arguments = {"waist_x": 0.735 * HALF_GAP} | keywords
    temporal = GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3)
    with pytest.raises(PulseError, match=match):
        PlanarPulse(temporal, 400e-6, **arguments)


def test_a_planar_pulse_prints_its_two_waists():
    launch = pulse()
    assert repr(launch).startswith("PlanarPulse(waist_x=")
    assert "waist_y=400 um" in repr(launch)
    assert "astigmatic Gaussian" in str(launch)


def test_the_critical_power_is_the_material_quantity_on_a_planar_grid():
    """Arnold 2009 quote a planar compressor against it; a line still refuses."""
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    index = float(np.real(FILL.index(np.array([GRID.omega0]))[0]))
    assert critical_power(800e-9, 1e-25, index=index, space=space) == critical_power(
        800e-9, 1e-25, index=index
    )
    with pytest.raises(GridError, match="does not apply to a LineGrid"):
        critical_power(800e-9, 1e-25, index=index, space=LineGrid(1e-3, 8))


def test_a_medium_that_is_not_the_modes_fill_is_refused_by_name():
    """The ordinary constructor: helium beside argon-filled modes would respond as He.

    The linear part would be right (the modes' kappa**2 cancels the medium's index),
    and the responses scaled by the wrong density; the translation refuses it.
    """
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    field0 = np.zeros((space.n_columns, 1, GRID.n_omega), complex)
    for medium in (SpatialMedium("He", 3.0), SpatialMedium("Ar", 1.1), SpatialMedium()):
        simulation = Simulation(
            GRID,
            (),
            [kerr()],
            field0,
            settings=settings(),
            spatial_grid=space,
            medium=medium,
        )
        with pytest.raises(TranslationError, match="slab modes' own fill"):
            simulation.translate()
    own = Simulation(
        GRID,
        (),
        [kerr()],
        field0,
        settings=settings(),
        spatial_grid=space,
        medium=SpatialMedium("Ar", 1.0),
    )
    assert own.translate().transform.geometry is not None
    vacuum = PlanarGrid([SlabMode(HALF_GAP)], 1e-3, 8)
    with pytest.raises(TranslationError, match="slab modes' own fill"):
        Simulation(
            GRID,
            (),
            [],
            field0,
            settings=settings(),
            spatial_grid=vacuum,
            medium=SpatialMedium("Ar", 1.0),
        ).translate()


@pytest.mark.parametrize(
    "helper", [spatial_fluence, spatial_intensity, beam_radius, fwhm_r]
)
def test_the_field_helpers_refuse_a_planar_grid_by_name(helper):
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    spectrum = np.zeros((space.n_columns, 1, GRID.n_omega), complex)
    with pytest.raises(GridError, match="planar_fluence or planar_intensity"):
        helper(spectrum, measure(space))


def test_the_integrated_and_axis_helpers_read_a_planar_plane():
    # The energy and the axis statistics are defined on a planar plane too: the
    # energy is the sum of the slab metric's per-mode energies (SEP 34 section 4.3).
    space = PlanarGrid(slab_modes((1, 3)), 1e-3, 8)
    field = pulse().build(GRID, space, medium=FILL)
    flux = measure(space)
    # Two orders of one sum of n positive terms: 2 gamma_n.
    bound = 2 * space.n_columns * GRID.n_omega * np.finfo(float).eps
    assert_allclose(
        spatial_energy(field, flux),
        np.sum(planar_mode_energy(field, flux)),
        rtol=bound,
        atol=0.0,
    )
    assert on_axis_intensity(field, flux) > 0
    assert np.all(np.isfinite(on_axis_spectrum(field, flux)))


def test_noise_is_refused_on_a_planar_grid():
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    with pytest.raises(NoiseError, match="PlanarGrid has no declared photon measure"):
        spatial_noise(measure(space), np.random.default_rng(49))
    draw = spatial_noise(
        measure(FreeGrid(1e-3, 2, 1e-3, 8), SpatialMedium()), np.random.default_rng(49)
    )
    field0 = np.zeros((space.n_columns, 1, GRID.n_omega), complex)
    simulation = Simulation(
        GRID,
        (),
        [],
        field0,
        settings=settings(),
        noise=draw,
        spatial_grid=space,
        medium=SpatialMedium("Ar", 1.0),
    )
    with pytest.raises(TranslationError, match="planar grid carries no noise"):
        simulation.translate()


def test_the_x_rule_warning_names_the_grids_own_keyword():
    with pytest.warns(
        UserWarning, match=r"PlanarGrid n_x = 12 .* Use n_x = 36"
    ) as caught:
        PlanarGrid(slab_modes(), 1e-3, 4, n_x=12)
    assert not any("n_r =" in str(one.message) for one in caught)


@pytest.mark.medium
def test_a_planar_scan_records_its_field_units_and_refuses_modal_reductions(
    tmp_path,
):
    """Two points; the stored columns are slab amplitudes per y frequency."""
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)

    def build(point):
        return Simulation.from_planar(
            GRID,
            space,
            [],
            pulse(energy=float(point.values["energy"])),
            settings=settings(steps=2),
        )

    path = tmp_path / "scan.h5"
    run_scan(
        Scan("planar", energy=[1e-3, 2e-3]),
        build,
        executor=Local(),
        output=path,
        reductions=[field_at_plane()],
        fields="all",
        progress="none",
    )
    found = []
    with h5py.File(path, "r") as file:
        file.visititems(
            lambda name, item: (
                found.append(json.loads(item.attrs["array_units"]))
                if "array_units" in item.attrs
                else None
            )
        )
    assert len(found) == 2
    for units in found:
        assert units["fields/values"] == "sqrt(W/m)"
        assert units["reduced/field_at_plane"] == "sqrt(W/m)"
    output = build(ScanPoint(0, {"energy": 1e-3}, (0,))).run(progress="none")
    for reduction in (
        energy_in_band(600e-9, 1100e-9),
        energy_in_window(-1e-12, 1e-12),
        peak_power(),
        mode_energies(),
        spectrum_in_band(600e-9, 1100e-9),
    ):
        with pytest.raises(ProcessingError, match="measures modal amplitudes"):
            reduction.evaluate(output, -1)  # pyright: ignore[reportAttributeAccessIssue] - a built-in's method


def test_the_planar_statistics_refuse_a_wrong_spectrum():
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    with pytest.raises(GridError, match="planar spectrum"):
        planar_mode_energy(np.zeros((3, 1, GRID.n_omega), complex), measure(space))


@pytest.mark.medium
def test_the_y_collar_absorbs_what_reaches_it_when_an_absorber_is_declared():
    """The Cartesian collar, applied (unlike the Luna planar branch's y window).

    A Gaussian in y centred in the collar: with an absorber declared the in-stage
    rate removes energy the linear flow otherwise keeps, save the modal loss; the
    same run without one keeps it. Both schemes of the temporal absorber run. The
    spatial collar is always in-stage: the default Adams history steps it only
    below 1.4 mm here, so the runs take eighty steps of 1.25 mm.
    """
    space = PlanarGrid(slab_modes((1,)), 0.5e-3, 32, window_factor=0.5)
    profile = np.exp(-(((space.y - 0.62e-3) / 60e-6) ** 2))
    temporal = GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3).build(GRID)
    field0 = (np.fft.fft(profile)[:, None, None] * temporal[0][None]).astype(complex)
    energies = {}
    collar = absorber(GRID, boundary_length=0.2)
    for name, declared, scheme in (
        ("none", None, "nonlinear"),
        ("nonlinear", collar, "nonlinear"),
        ("split_step", collar, "split_step"),
    ):
        output = Simulation.from_planar(
            GRID,
            space,
            [],
            field0,
            settings=settings(
                steps=80,
                stats=default_spec("free"),
                absorber=declared,
                absorber_scheme=scheme,
            ),
        ).run(progress="none")
        energy = np.ravel(output["stats/energy"])
        energies[name] = energy[-1] / energy[0]
    loss = math.exp(-2 * 0.149 * LENGTH)
    assert energies["none"] == pytest.approx(loss, rel=1e-3)
    assert energies["nonlinear"] < 0.5 * loss
    assert energies["split_step"] < 0.5 * loss


@pytest.mark.parametrize(
    ("shape", "match"),
    [("two_columns", "scalar single-column"), ("nonfinite", "nonfinite")],
)
def test_a_planar_pulse_checks_the_temporal_pulse_result(shape, match):
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)

    class Temporal:
        def build(self, grid):
            if shape == "two_columns":
                return np.zeros((2, 1, grid.n_omega), complex)
            return np.full((1, 1, grid.n_omega), complex(np.inf))

    launch = PlanarPulse(Temporal(), 4e-4, waist_x=0.735 * HALF_GAP)  # pyright: ignore[reportArgumentType] - a stand-in pulse
    with warnings.catch_warnings():
        # inf times a zero Fourier coefficient is the NaN the check is for.
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(PulseError, match=match):
            launch.build(GRID, space, medium=FILL)


def test_a_beam_narrower_than_the_y_samples_is_unresolved():
    # An odd count places no sample at y = 0, so a needle beam falls between them.
    space = PlanarGrid(slab_modes((1,)), 1e-3, 3)
    launch = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        1e-9,
        waist_x=0.735 * HALF_GAP,
    )
    with pytest.raises(PulseError, match="unresolved"):
        launch.build(GRID, space, medium=FILL)


def test_an_explicit_frame_is_kept():
    space = PlanarGrid(slab_modes((1,)), 1e-3, 8)
    frame = Frame(GRID.omega0, subtract_beta0=True, beta0=0.0, beta1=0.0)
    simulation = Simulation.from_planar(
        GRID, space, [], pulse(), settings=settings(), frame=frame
    )
    assert simulation.frame is frame
    separable = simulation.translate().linear.separable
    assert separable is not None
    assert not np.any(separable.frame)
