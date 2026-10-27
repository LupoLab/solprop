# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Cartesian public inputs, constant flow and output metadata."""

import dataclasses
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_allclose

from flux_oracles import admittance
from hankel_oracles import gamma
from solprop.backends.registry import available
from solprop.constants import EPS0, HBAR, C
from solprop.driver import Simulation, SimulationSettings, prop_free
from solprop.exceptions import DriverError, GridError, OutputError, TranslationError
from solprop.fourier import spectral_density
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid
from solprop.linear import Frame
from solprop.modes import SlabMode
from solprop.noise import spatial_noise
from solprop.nonlinear.responses import KerrResponse
from solprop.output import check_cache
from solprop.plotting.style import per_width
from solprop.processing.axes import transverse_units
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import (
    SpatialFlux,
    SpatialMedium,
    critical_power,
    spatial_energy,
    spatial_intensity,
)
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta
from solprop.translate.linop import delta_phi
from spatial_records import record_metric


def _vacuum(grid, space):
    """Return the vacuum measure of the plane at z = 0 (SEP 34)."""
    return SpatialFlux.resolve(grid, space, SpatialMedium())


EPS = np.finfo(float).eps


@pytest.fixture(params=["field", "envelope"])
def free_case(request):
    constructor = (
        FieldGrid.from_limits
        if request.param == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    grid = constructor(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = FreeGrid(3e-4, 12, 4e-4, 16, window_factor=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 80e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        backend="numpy_ref",
    )
    return grid, space, pulse, settings


def test_free_interface_preserves_an_explicit_frame(free_case):
    grid, space, pulse, settings = free_case
    frame = dataclasses.replace(Frame.comoving(grid), beta0=0.0, beta1=0.0)
    simulation = Simulation.from_free(
        grid,
        space,
        SpatialMedium(),
        [],
        pulse,
        settings=settings,
        frame=frame,
        noise=None,
    )
    assert simulation.frame is frame
    problem = simulation.translate()
    assert problem.linear.separable is not None
    assert not np.any(problem.linear.separable.frame)


@pytest.mark.parametrize("backend", available())
def test_free_interface_separable_flow_and_metadata(free_case, request, backend):
    grid, space, pulse, settings = free_case
    settings = dataclasses.replace(settings, backend=backend)
    simulation = Simulation.from_free(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    )
    problem = simulation.translate()
    assert problem is simulation.translate()
    assert problem.transform.geometry is not None
    assert problem.transform.geometry.kind == "free"
    assert problem.linear.separable is problem.responses.separable
    assert problem.linear.phi_coefficients is None
    assert problem.responses.prefactor is None
    assert problem.transform.synthesis.size == 0
    metadata = simulation.metadata()
    assert metadata.simulation_kind == "free"
    assert metadata.spatial is not None
    assert metadata.spatial["stored_field_units"] == "V/m"
    assert metadata.spatial["spectral_axis_names"] == ("k_x", "k_y")
    # Exact constant flow is two exponentials and one complex product. Energy
    # additionally sums N columns and the temporal frequency bins.
    field_bound = gamma(64)
    energy_bound = gamma(32 * (space.n_columns + grid.n_omega))
    output = prop_free(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    )
    start, finish = output.fields("fft")
    expected = start * np.exp(delta_phi(problem.linear, 0, grid.length))[0]
    defect = np.max(np.abs(finish - expected)) / np.max(np.abs(start))
    assert defect < field_bound
    energy_error = abs(
        spatial_energy(finish, _vacuum(grid, space))
        / spatial_energy(start, _vacuum(grid, space))
        - 1
    )
    assert energy_error < energy_bound
    record_metric(
        request,
        "Cartesian exact linear flow (relative)",
        grid.kind,
        defect,
        field_bound,
        backend,
    )
    record_metric(
        request,
        "Cartesian linear energy (relative)",
        grid.kind,
        energy_error,
        energy_bound,
        backend,
    )
    assert_allclose(problem.transform.geometry.on_axis, space.on_axis, rtol=0, atol=0)


@pytest.mark.parametrize("model", ["input", "modified"])
@pytest.mark.parametrize("n_pol", [1, 2])
def test_spatial_noise_has_one_photon_in_every_column_bin(
    free_case, model, n_pol, request
):
    grid, space, _, _ = free_case
    # sqrt normalization, complex phase, squared modulus and density multiplication
    # involve fewer than 32 rounded operations per bin; there is no long reduction.
    bound = 32 * EPS / (1 - 32 * EPS)
    draw = spatial_noise(
        _vacuum(grid, space), np.random.default_rng(313), model=model, n_pol=n_pol
    )
    # A bin's photon is measured in the entrance metric, its real admittance (the
    # vacuum obliquity) times the column's area weight; a bin that is no forward
    # power port carries none (SEP 34 section 4.3).
    eta = admittance(grid, space.kperp_sq)
    measured = (
        0.5
        * EPS0
        * C
        * space.spectral_weights[:, None, None]
        * eta[:, None, :]
        * spectral_density(draw.spectrum, grid.dt, kind=grid.kind)
        * grid.d_omega
    )
    wanted = np.broadcast_to(
        (HBAR * grid.omega * (eta > 0))[:, None, :], measured.shape
    )
    assert_allclose(measured, wanted, rtol=bound, atol=0)
    errors = np.divide(
        np.abs(measured - wanted), wanted, out=np.zeros_like(wanted), where=wanted > 0
    )
    record_metric(
        request,
        "Cartesian spatial noise photons per bin (relative)",
        f"{grid.kind}, {model}, p={n_pol}",
        float(np.max(errors)),
        bound,
    )
    other = spatial_noise(
        _vacuum(grid, space),
        np.random.default_rng(313),
        model="modified" if model == "input" else "input",
        n_pol=n_pol,
    )
    np.testing.assert_array_equal(draw.spectrum, other.spectrum)
    assert draw.seed == 313 and not draw.spectrum.flags.writeable


@pytest.mark.parametrize("model", ["input", "modified"])
@pytest.mark.parametrize("backend", available())
def test_free_simulation_records_noise_model_and_seed(
    free_case, model, backend, request
):
    grid, space, pulse, settings = free_case
    settings = dataclasses.replace(settings, backend=backend)
    simulation = Simulation.from_free(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=model, seed=33
    )
    assert simulation.noise is not None
    assert simulation.noise.seed == 33
    assert simulation.translate().initial.noise_model == model
    noise = simulation.metadata().noise
    assert noise is not None and noise.seed == 33
    # With no nonlinear source, both noise models obey the exact linear flow
    # of their saved entrance. The phase subtraction and complex exponential
    # cost fewer than 64 rounded operations per optical phase magnitude.
    phase = delta_phi(simulation.translate().linear, 0.0, grid.length)[0]
    bound = 64 * EPS * (1 + float(np.max(np.abs(phase))))
    fields = simulation.run().fields("fft")
    expected = np.exp(phase) * fields[0]
    assert_allclose(fields[-1], expected, rtol=bound, atol=0)
    errors = np.divide(
        np.abs(fields[-1] - expected),
        np.abs(expected),
        out=np.zeros(expected.shape),
        where=np.abs(expected) > 0,
    )
    record_metric(
        request,
        "Cartesian spatial noise linear flow (relative)",
        f"{grid.kind}, {model}",
        float(np.max(errors)),
        bound,
        backend=backend,
    )


def test_spatial_resume_refuses_geometry_changes(free_case, tmp_path):
    grid, space, pulse, settings = free_case
    settings = dataclasses.replace(settings, backend="numpy_ref")
    simulation = Simulation.from_free(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    )
    path = tmp_path / "free.h5"
    simulation.run(path)
    assert check_cache(path, simulation.metadata())
    simulation.run(path, resume=True)
    changed = Simulation.from_free(
        grid,
        dataclasses.replace(space, radius_x=space.radius_x * 2),
        SpatialMedium(),
        [],
        pulse,
        settings=settings,
        noise=None,
    )
    with pytest.raises(OutputError, match="spatial"):
        changed.run(path, resume=True)


def test_cartesian_refuses_varying_media_and_tabulation(free_case):
    grid, space, pulse, settings = free_case
    with pytest.raises(DriverError, match="constant"):
        Simulation.from_free(
            grid,
            space,
            SpatialMedium("Ar", pressure=Linear(1.0, 2.0, grid.length)),
            [],
            pulse,
            settings=settings,
            noise=None,
        )
    with pytest.raises(TranslationError, match="tabulate"):
        Simulation.from_free(
            grid,
            space,
            SpatialMedium(),
            [],
            pulse,
            settings=dataclasses.replace(settings, propagator_mode="tabulate"),
            noise=None,
        ).translate()


@pytest.mark.parametrize("imaginary_index", [-1e-5, 1e-5])
@pytest.mark.parametrize("backend", available())
def test_cartesian_material_loss_gain_and_evanescence(
    free_case, imaginary_index, backend, request
):
    grid, _, _, settings = free_case
    space = FreeGrid(3e-7, 4, 4e-7, 6, window_factor=0)
    rng = np.random.default_rng(317)
    initial = (
        rng.normal(size=(24, 2, grid.n_omega))
        + 1j * rng.normal(size=(24, 2, grid.n_omega))
    ) * grid.active_mask
    index = 1.2 + 1j * imaginary_index
    medium = SpatialMedium(index)
    # Constant n gives beta1=Re(n)/c exactly in the physical definition. This
    # independent expression includes both propagating loss/gain and the
    # attenuation-clamped evanescent branch before the backend is constructed.
    square = (index * grid.omega[None, :] / C) ** 2 - space.kperp_sq[:, None]
    root = np.sqrt(square)
    propagating = square.real > 0
    attenuation = np.where(propagating, root.imag, np.minimum(np.abs(root.imag), 200.0))
    rate = (
        -attenuation - 1j * (root.real - index.real * grid.omega[None, :] / C)
    ) * grid.active_mask
    bound = gamma(128) * (1 + grid.length * np.max(np.abs(rate)))
    expected = initial * np.exp(grid.length * rate[:, None, :])
    assert np.any(propagating[:, grid.active_mask]) and np.any(
        ~propagating[:, grid.active_mask]
    )
    simulation = Simulation.from_free(
        grid,
        space,
        medium,
        [],
        initial,
        settings=dataclasses.replace(settings, backend=backend),
        noise=None,
    )
    actual = simulation.run().fields("fft")[-1]
    measured = np.max(np.abs(actual - expected)) / np.max(np.abs(initial))
    assert measured < bound
    record_metric(
        request,
        "Cartesian complex-index exact flow (relative)",
        f"{grid.kind}, Im(n)={imaginary_index}",
        measured,
        bound,
        backend,
    )


def test_cartesian_resume_distinguishes_legacy_cutoff_arithmetic(free_case):
    grid, space, pulse, settings = free_case
    simulation = Simulation.from_free(
        grid, space, SpatialMedium(), [], pulse, settings=settings, noise=None
    )
    current = simulation.metadata()
    assert current.schema_version == 25
    assert current.configuration["spatial_arithmetic"] == "split-square-v1"
    legacy_configuration = dict(current.configuration)
    del legacy_configuration["spatial_arithmetic"]
    legacy = dataclasses.replace(current, configuration=legacy_configuration)
    assert legacy.configuration_hash() != current.configuration_hash()


# --- the line (x-only) free space, uniform in y ---------------------------------------


LINE_WAVELENGTH = 800e-9
LINE_K = 2 * np.pi / LINE_WAVELENGTH


def _line_settings(length, planes, *, steps=None):
    return SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=planes,
        dz_max=length,
        dz_init=length if steps is None else length / steps,
        schedule=None
        if steps is None
        else FixedSchedule.uniform(0.0, length, length / steps),
        # A few fixed steps measure a one-step method; an Adams history would start
        # them at order one.
        integrator=Multistep() if steps is None else RungeKutta(),
        backend="numpy_ref",
    )


def test_line_gaussian_diffracts_with_the_one_dimensional_closed_forms():
    """Width, amplitude and Gouy phase of a line beam against (1+1)-D paraxial optics.

    At the carrier bin the stored spectrum is a monochromatic beam propagated by the
    exact operator, whose frame subtracts exactly ``k`` there. A Gaussian of waist
    ``w0`` then has ``w(z) = w0 sqrt(1 + (z/z_R)^2)``, an axial amplitude falling as
    ``(1 + (z/z_R)^2)^(-1/4)`` -- not the plane's ``^(-1/2)`` -- and the Gouy phase
    ``arctan(z/z_R)/2``, half the plane's; in the internal FFT convention it is
    ``+arctan/2`` (the optics phase conjugated).
    """
    waist = 100e-6
    rayleigh = LINE_K * waist**2 / 2
    grid = EnvelopeGrid.from_limits(
        2 * rayleigh, LINE_WAVELENGTH, (780e-9, 820e-9), 2e-12, thg=False
    )
    assert grid.omega[0] == grid.omega0
    space = LineGrid(1.5e-3, 256, window_factor=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=LINE_WAVELENGTH, fwhm=300e-15, energy=1e-3), waist
    )
    planes = [rayleigh / 2, rayleigh, 2 * rayleigh]
    output = prop_free(
        grid,
        space,
        SpatialMedium(),
        [],
        pulse,
        settings=_line_settings(2 * rayleigh, planes),
        noise=None,
    )
    assert output.metadata is not None
    assert output.metadata.transverse_measure == "per_unit_width"
    fields = output.fields("fft")
    centre = int(np.argmin(np.abs(space.x)))
    assert space.x[centre] == 0
    entrance = np.fft.ifft(fields[0][:, 0, 0])
    # The first non-paraxial term of the exact operator is O((k w0)^-2) = 1.6e-6
    # relative in width, amplitude and phase; the bound is four times that. The
    # grid's aliasing and wrap-around are below 1e-30 at this spacing and extent.
    bound = 4 / (LINE_K * waist) ** 2
    for z, field in zip(planes, fields[1:], strict=True):
        beam = np.fft.ifft(field[:, 0, 0])
        intensity = np.abs(beam) ** 2
        width = 2 * np.sqrt(np.sum(intensity * space.x**2) / np.sum(intensity))
        spread = 1 + (z / rayleigh) ** 2
        assert abs(width / (waist * np.sqrt(spread)) - 1) < bound
        amplitude = abs(beam[centre]) / abs(entrance[centre])
        assert abs(amplitude / spread**-0.25 - 1) < bound
        gouy = np.angle(beam[centre] / entrance[centre])
        assert abs(gouy - 0.5 * np.arctan(z / rayleigh)) < bound
        # The measure is per metre of y and the flow is lossless.
        energy_error = abs(
            spatial_energy(field, _vacuum(grid, space))
            / spatial_energy(fields[0], _vacuum(grid, space))
            - 1
        )
        assert energy_error < gamma(32 * (space.n_columns + grid.n_omega))


def test_line_focal_witness_of_pr_416():
    """Launch from ``propz = -L`` and recover the waist at the focus (Luna PR #416).

    Luna's ``test_freespace.jl`` compares the focal ``I(x)`` with a Gaussian at 1e-3.
    Here the pre-propagation and the run are the same exact operator, so the focus
    is also compared with the beam built directly at its waist, to rounding.
    """
    waist = 100e-6
    length = LINE_K * waist**2 / 2
    grid = EnvelopeGrid.from_limits(
        length, LINE_WAVELENGTH, (780e-9, 820e-9), 2e-12, thg=False
    )
    space = LineGrid(1.5e-3, 256, window_factor=0)
    temporal = GaussianPulse(wavelength=LINE_WAVELENGTH, fwhm=300e-15, energy=1e-3)
    output = prop_free(
        grid,
        space,
        SpatialMedium(),
        [],
        SpatialPulse(temporal, waist, propz=-length),
        settings=_line_settings(length, [length]),
        noise=None,
    )
    launched, focus = output.fields("fft")
    intensity = spatial_intensity(focus, _vacuum(grid, space))
    peak = int(np.argmax(np.max(intensity, axis=0)))
    profile = intensity[:, peak] / np.max(intensity[:, peak])
    gaussian = np.exp(-2 * space.x**2 / waist**2)
    assert np.max(np.abs(profile - gaussian)) < 1e-3
    launch = spatial_intensity(launched, _vacuum(grid, space))[:, peak]
    assert np.max(np.abs(launch / np.max(launch) - gaussian)) > 0.1
    direct = SpatialPulse(temporal, waist).build(grid, space, medium=SpatialMedium())
    # Launch and run apply exp(-D L) and exp(+D L) with |D| L below k L at the
    # band's top: 64 rounded operations on a phase of that size bound the rest.
    phase = float(np.max(grid.omega)) / C * length
    assert np.max(np.abs(focus - direct)) < 64 * EPS * phase * np.max(np.abs(direct))


def test_line_equals_a_plane_uniform_in_y():
    """A Kerr run on the line is the plane's run of a field uniform in y.

    A field uniform in y lives in the plane's ``k_y = 0`` columns alone, as
    ``n_y`` times the line's spectrum; the nonlinear response of equal rows is
    uniform too, so the other columns stay empty. The plane's energy is the line's
    per metre of y times its y period.
    """
    length = 1e-2
    grid = EnvelopeGrid.from_limits(
        length, LINE_WAVELENGTH, (700e-9, 950e-9), 400e-15, thg=False
    )
    line = LineGrid(1e-3, 64, window_factor=0)
    plane = FreeGrid(1e-3, 64, 2e-4, 4, window_factor=0)
    start = SpatialPulse(
        GaussianPulse(wavelength=LINE_WAVELENGTH, fwhm=50e-15, energy=5e-2), 150e-6
    ).build(grid, line, medium=SpatialMedium())
    uniform = np.zeros((plane.n_columns, 1, grid.n_omega), complex)
    uniform[: plane.n_x] = plane.n_y * start
    responses = [KerrResponse(3e-26, thg=False)]
    settings = _line_settings(length, [length], steps=10)
    along_line = prop_free(
        grid, line, SpatialMedium(), responses, start, settings=settings, noise=None
    )
    on_plane = prop_free(
        grid, plane, SpatialMedium(), responses, uniform, settings=settings, noise=None
    )
    finish = along_line.fields("fft")[-1]
    assert np.max(np.abs(finish - start)) > 0.01 * np.max(np.abs(start))
    plane_finish = on_plane.fields("fft")[-1]
    # The two runs share every operation but the y transform of equal rows; 70
    # nonlinear evaluations of the Cartesian FFT depth bound them (the ladder's).
    depth = int(np.ceil(np.log2(grid.n_to))) + 6 + 2
    bound = 70 * 32 * EPS * depth * np.max(np.abs(finish))
    assert np.max(np.abs(plane_finish[: plane.n_x] / plane.n_y - finish)) < bound
    assert np.max(np.abs(plane_finish[plane.n_x :])) < bound * plane.n_y
    period = plane.n_y * 2 * (plane.radius_y / plane.n_y)
    assert_allclose(
        spatial_energy(plane_finish, _vacuum(grid, plane)),
        period * spatial_energy(finish, _vacuum(grid, line)),
        rtol=gamma(64 * (plane.n_columns + grid.n_omega)),
    )


@pytest.mark.medium
@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_line_kerr_soliton_holds_at_its_power_per_metre(fraction):
    """The (1+1)-D spatial soliton ``sech(x/x0)`` at the temporal peak of a long pulse.

    ``i A_z + A_xx/(2k) + g|A|^2 A = 0`` with ``g = 3 k chi3/(8 n^2)`` is solved by
    ``A0 sech(x/x0)`` when ``A0^2 = 8/(3 k0^2 chi3 x0^2)``: a power per metre of y
    ``P' = 8 eps0 c/(3 k0^2 chi3 x0)``, derived from ``g``. In vacuum, with a 1 ps
    pulse, every time slice evolves on its own, so the peak slice holds its shape
    over three diffraction lengths ``k x0^2``; at half the power it diffracts.
    """
    chi3, x0 = 3e-26, 50e-6
    length = 3 * LINE_K * x0**2
    power = fraction * 8 * EPS0 * C / (3 * LINE_K**2 * chi3 * x0)
    grid = EnvelopeGrid.from_limits(
        length, LINE_WAVELENGTH, (760e-9, 845e-9), 6e-12, thg=False
    )
    space = LineGrid(20 * x0, 256, window_factor=0)
    temporal = GaussianPulse(
        wavelength=LINE_WAVELENGTH, fwhm=1e-12, peak_power=power
    ).build(grid)
    profile = 1 / np.cosh(space.x / x0)
    spectrum = np.fft.fft(profile)
    norm = 0.5 * EPS0 * C * np.sum(space.spectral_weights * np.abs(spectrum) ** 2)
    start = spectrum[:, None, None] / np.sqrt(norm) * temporal
    launched = spatial_intensity(start, _vacuum(grid, space))
    first = int(np.argmax(np.max(launched, axis=0)))
    # The launched peak slice carries P' per metre of y in the paraxial measure
    # eps0 c |E|^2 / 2. Its axial flux is lower by the mean obliquity: for
    # sech(x/x0), <k_x^2> = 1/(3 x0^2), so the flux is P' (1 - 1/(6 k0^2 x0^2)),
    # 1.1e-6 below it. The next order, (k0 x0)^-4 ~ 4e-11, and the obliquity's
    # change across the 1 ps pulse's 1e-3 relative bandwidth, ~2e-9, fit 1e-8.
    assert_allclose(
        np.sum(space.spatial_weights * launched[:, first]),
        power * (1 - 1 / (6 * LINE_K**2 * x0**2)),
        rtol=1e-8,
    )
    output = prop_free(
        grid,
        space,
        SpatialMedium(),
        [KerrResponse(chi3, thg=False)],
        start,
        settings=_line_settings(length, [length], steps=10),
        noise=None,
    )
    final = spatial_intensity(output.fields("fft")[-1], _vacuum(grid, space))
    last = int(np.argmax(np.max(final, axis=0)))
    shape = final[:, last] / np.max(launched[:, first])
    deviation = float(np.max(np.abs(shape - profile**2)))
    if fraction == 1.0:
        # Ten fixed steps of the embedded fourth-order pair: twenty steps measure
        # 1.1e-5 and the local error is fifth order, so ten are about 3.5e-4. The
        # shock term adds O(phi_NL/(omega0 T)) ~ 1e-3 * 3 / 2400 ~ 1e-6. The bound
        # is 2e-3; a diffracting beam of this width departs by order one.
        assert deviation < 2e-3
    else:
        assert deviation > 0.1


def test_line_run_is_recorded_and_labelled_per_metre_of_y(tmp_path):
    grid = EnvelopeGrid.from_limits(1e-3, LINE_WAVELENGTH, (600e-9, 1.2e-6), 150e-15)
    line = LineGrid(3e-4, 16, window_factor=0.1)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=LINE_WAVELENGTH, fwhm=30e-15, energy=1e-3), 80e-6
    )
    settings = _line_settings(1e-3, [1e-3])
    simulation = Simulation.from_free(
        grid, line, SpatialMedium(), [], pulse, settings=settings, noise=None
    )
    metadata = simulation.metadata()
    assert metadata.simulation_kind == "free"
    assert metadata.transverse_measure == "per_unit_width"
    assert metadata.spatial is not None
    assert metadata.spatial["shape"] == (16, 1)
    assert metadata.spatial["stored_field_units"] == "V/m"
    output = simulation.run()
    assert transverse_units(output).energy == "J/m"
    assert per_width(output) == "/m"
    # The launched energy is the pulse's, per metre of y.
    assert_allclose(
        spatial_energy(output.fields("fft")[0], _vacuum(grid, line)),
        1e-3,
        rtol=gamma(64 * (line.n_columns + grid.n_omega)),
    )
    plane = FreeGrid(3e-4, 16, 3e-4, 16, window_factor=0.1)
    area = Simulation.from_free(
        grid, plane, SpatialMedium(), [], pulse, settings=settings, noise=None
    )
    assert area.metadata().transverse_measure == "area"
    assert "transverse_measure" not in area.metadata().configuration
    # A line and a plane are different runs: a resume across them is refused.
    simulation.run(tmp_path / "line.h5")
    with pytest.raises(OutputError, match="spatial"):
        area.run(tmp_path / "line.h5", resume=True)


@pytest.mark.parametrize("model", ["input", "modified"])
def test_line_noise_is_one_photon_per_column_per_metre_of_y(model):
    grid = EnvelopeGrid.from_limits(1e-3, LINE_WAVELENGTH, (600e-9, 1.2e-6), 150e-15)
    line = LineGrid(3e-4, 12, window_factor=0)
    draw = spatial_noise(_vacuum(grid, line), np.random.default_rng(49), model=model)
    # The plane's bound: fewer than 32 rounded operations per bin, no long sum.
    bound = 32 * EPS / (1 - 32 * EPS)
    eta = admittance(grid, line.kperp_sq)
    measured = (
        0.5
        * EPS0
        * C
        * line.spectral_weights[:, None, None]
        * eta[:, None, :]
        * spectral_density(draw.spectrum, grid.dt, kind=grid.kind)
        * grid.d_omega
    )
    wanted = np.broadcast_to(
        (HBAR * grid.omega * (eta > 0))[:, None, :], measured.shape
    )
    assert_allclose(measured, wanted, rtol=bound, atol=0)


def test_line_refusals_name_the_line():
    grid = EnvelopeGrid.from_limits(1e-3, LINE_WAVELENGTH, (600e-9, 1.2e-6), 150e-15)
    line = LineGrid(3e-4, 16)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=LINE_WAVELENGTH, fwhm=30e-15, energy=1e-3), 80e-6
    )
    settings = _line_settings(1e-3, [1e-3])
    with pytest.raises(GridError, match="LineGrid"):
        critical_power(LINE_WAVELENGTH, 3e-26, index=1.0, space=line)
    plane = critical_power(
        LINE_WAVELENGTH, 3e-26, index=1.0, space=FreeGrid(1e-3, 8, 1e-3, 8)
    )
    assert plane == critical_power(LINE_WAVELENGTH, 3e-26, index=1.0)
    # Constant media only, as on the plane: a gas path is refused.
    with pytest.raises(DriverError, match="constant"):
        Simulation.from_free(
            grid,
            line,
            SpatialMedium("Ar", pressure=Linear(1.0, 2.0, 1e-3)),
            [],
            pulse,
            settings=settings,
            noise=None,
        )
    # A line is a free-space grid: modes uniform in y go through prop_waveguide.
    with pytest.raises(DriverError, match="empty mode sequence"):
        Simulation(
            grid,
            [SlabMode(50e-6, "Ar", 1.0)],
            [],
            pulse,
            settings=settings,
            spatial_grid=line,
            medium=SpatialMedium(),
        )


def test_a_line_is_recorded_per_metre_of_y_without_from_free_s_key():
    """The measure follows the problem's geometry, not a key only from_free writes."""
    grid = EnvelopeGrid.from_limits(1e-3, LINE_WAVELENGTH, (600e-9, 1.2e-6), 150e-15)
    line = LineGrid(3e-4, 16, window_factor=0.1)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=LINE_WAVELENGTH, fwhm=30e-15, energy=1e-3), 80e-6
    )
    simulation = Simulation.from_free(
        grid,
        line,
        SpatialMedium(),
        [],
        pulse,
        settings=_line_settings(1e-3, [1e-3]),
        noise=None,
    )
    simulation.configuration.pop("transverse_measure")
    assert "transverse_measure" not in simulation.configuration
    assert simulation.metadata().transverse_measure == "per_unit_width"
