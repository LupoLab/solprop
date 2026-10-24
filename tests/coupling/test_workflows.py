# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The eight compositions of SEP 31 section 2, each as a short integrated run.

Each composition joins ordinary simulations through field planes: extraction
(``field_plane``), transport, coupling (``couple``) and a launch (``with_input``)
into a receiving run of about ten steps. Every test asserts the energy ledger of
what it composed -- the incident, captured, uncaptured and launched energies and the
transport ledger -- that the reported quantities are the ones the model defines, and
that no energy is restored: a launch never exceeds its incident energy beyond the
coupling's reported defects, and the receiving run's entrance carries exactly the
launch. The mandatory field-grid Kerr/plasma/THG case checks that the generated
third-harmonic band is transferred with its energy and its reference.

Several compositions pair different backends: the source on ``numpy_ref`` or
``numba_cpu``, the receiver on the native CPU backend when it is installed (else
``numba_cpu``). The physics of each piece is qualified by its own module's tests;
these check the seams between them.
"""

import math
from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import j0, jn_zeros, k0

from solprop import fourier
from solprop.backends.registry import available
from solprop.constants import C
from solprop.coupling import (
    STEP_INDEX_BEAM_ENVELOPE,
    Alignment,
    Aperture,
    CouplingModel,
    CouplingReport,
    FieldPlane,
    FreeSpace,
    Outcouple,
    PlaneGeometry,
    SpectralTransfer,
    ThinLens,
    couple,
    transport,
)
from solprop.coupling.transport import TransportReport
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import CouplingError
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, RadialGrid
from solprop.modes import StepIndexMode
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.physdata import gamma3
from solprop.processing import field_plane
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController

pytestmark = pytest.mark.medium

RECEIVER = "native" if "native" in available() else "numba_cpu"
"""The receiving runs' backend in the mixed-backend compositions."""

PRESSURE = 3.0
ARGON = SpatialMedium("Ar", PRESSURE)
SCALAR = CouplingModel("scalar_paraxial", model_error=0.05)
ENVELOPE_GRID = {
    "lambda0": 800e-9,
    "wavelength_limits": (700e-9, 920e-9),
    "trange": 200e-15,
}
LAUNCH_ROUNDING = 1e-12
"""Relative agreement of a run's entrance-plane energy with the launch's.

The entrance plane stores the launched coefficients; the two energies are the
same weighted sum in another order, so they agree to its rounding (measured 3e-16).
"""


def capillary(
    radius=75e-6,
    modes=4,
    *,
    length=1e-3,
    energy=1e-6,
    backend="numpy_ref",
    **changes,
):
    """Return a short argon capillary on the tests' envelope grid."""
    arguments = {
        "modes": modes,
        "n_save": 2,
        "noise": None,
        "envelope": True,
        "energy": energy,
        "fwhm": 30e-15,
        "plasma": False,
        "backend": backend,
        **ENVELOPE_GRID,
        **changes,
    }
    return Simulation.from_capillary(radius, length, "Ar", PRESSURE, **arguments)


def assert_coupled(report: CouplingReport, received=None) -> None:
    """Assert a scalar coupling's ledger, and the receiving run's entrance.

    The scalar model's captured energy lives in a common positive metric, so it is
    reported with its uncaptured complement, which is never clipped: the
    orthogonal-residual identity makes both nonnegative per bin, and they sum to
    the incident energy. The launch is the captured coordinates in the target's
    native metric; it may differ from the captured energy only by the reported
    target-metric defect, never by a renormalisation to the incident energy.
    """
    assert report.model == "scalar_paraxial"
    assert report.captured is not None and report.captured_reason == ""
    uncaptured = report.uncaptured
    assert uncaptured is not None
    scale = float(np.max(report.incident))
    # The residual identity holds to the fit's rounding per bin.
    assert np.all(uncaptured >= -1e-12 * scale)
    captured, missing = report.captured_energy, report.uncaptured_energy
    assert captured is not None and missing is not None
    assert captured + missing == pytest.approx(report.incident_energy, rel=1e-12)
    assert report.model_error is not None and report.budget <= report.model_error
    target_metric = next(item for item in report.items if item.name == "target metric")
    assert target_metric.value is not None
    # Nothing restored: the launch is the captured energy up to the reported
    # metric defect between the common measure and the target's native one.
    assert report.launched_energy <= captured * (1 + target_metric.value) + 1e-30
    assert abs(report.launched_energy - captured) <= target_metric.value * captured
    if received is not None:
        entrance = field_plane(received, 0)
        assert entrance.energy() == pytest.approx(
            report.launched_energy, rel=LAUNCH_ROUNDING
        )


def assert_transported(report: TransportReport) -> None:
    """Assert a transport ledger: passive, closed and with nothing restored."""
    # A lossless element such as FreeSpace keeps the energy to rounding.
    assert report.energy_out <= report.energy_in * (1 + 1e-12)
    # Each element books its change as loss, removal or signed approximation
    # error; the chain closes to the rounding of those sums.
    assert abs(report.ledger_residual) <= 1e-12 * report.energy_in
    # The signed approximation error is booked as energy gained, so the energy
    # change is the losses and removals less it.
    accounted = report.specified_loss + report.removed - report.approximation_error
    assert report.energy_in - report.energy_out == pytest.approx(
        accounted, rel=1e-9, abs=1e-12 * report.energy_in
    )


def band_energy(spectrum, omega, low, high):
    """Return the energy of the bins with vacuum wavelength in [low, high], J."""
    wavelength = 2 * math.pi * C / np.where(omega > 0, omega, np.inf)
    return float(np.sum(spectrum[(wavelength >= low) & (wavelength <= high)]))


# ---------------------------------------------------------------- (a) capillaries


@pytest.fixture(scope="module")
def capillary_exit():
    """Ten steps of a 75 um argon capillary, read with its simulation."""
    source = capillary(modes=3)
    run = source.run(progress="none")
    return source, field_plane(run, source=source)


def test_a_capillary_exit_into_a_wider_capillary_two_ways(capillary_exit):
    """(a) Direct overlap, and outcouple -> 2 mm of argon -> recouple.

    Source on ``numpy_ref``, receiver on the native CPU backend. The direct overlap
    and the synthesise-then-project route agree to the synthesis window's error
    with no gap (the coupling module holds that); here a 2 mm argon gap between
    the exit and the wider entrance lowers the capture (0.99962 direct, 0.99853
    through the gap, measured), and every step books its energy.
    """
    source, plane = capillary_exit
    wider = capillary(radius=100e-6, modes=6, backend=RECEIVER)
    port = wider.input_port()
    direct = couple(plane, port, model=SCALAR)
    assert_coupled(direct.report)
    # The incident energy in M0 is the exit's modal energy to the source-metric
    # item (core index against effective index).
    source_metric = next(
        item for item in direct.report.items if item.name == "source metric"
    )
    assert source_metric.value is not None
    assert direct.report.incident_energy == pytest.approx(
        plane.energy(), rel=2 * source_metric.value
    )
    synthesised = Outcouple(RadialGrid(400e-6, 256), medium=ARGON).apply(plane)
    moved, ledger = transport(
        synthesised.plane,
        [FreeSpace(2e-3, ARGON)],
        frame=source.frame,
        geometry=port.geometry,
    )
    assert_transported(ledger)
    assert moved.reference.delay > plane.reference.delay
    through = couple(moved, port, model=SCALAR)
    received = wider.with_input(through.launch).run(progress="none")
    assert_coupled(through.report, received)
    direct_share, gap_share = direct.report.efficiency, through.report.efficiency
    assert direct_share is not None and gap_share is not None
    assert 0.99 < gap_share < direct_share < 1.0
    # The launched run continues the plane's absolute time and carrier phase.
    entrance = field_plane(received, 0)
    assert entrance.reference == moved.reference
    assert received.metadata is not None
    record = cast("Mapping[str, object]", received.metadata.configuration["coupling"])
    assert record["model"] == "scalar_paraxial"


# ------------------------------------------------- (b) a nonlinear radial beam


def test_a_kerr_prepared_radial_beam_into_a_capillary():
    """(b) A Kerr gas gap on a radial grid, its entrance plane, couple, capillary.

    5 mm of 3 bar argon on an envelope grid, the beam 37.5 um (w/a = 0.5), run on
    ``numba_cpu``; the capillary receives on the native CPU backend. The same gap
    without Kerr is the control: Kerr changes the fundamental's share (0.8715
    linear, 0.8805 with Kerr, measured), so the prepared field, not a fitted
    Gaussian, is what the capillary receives.
    """
    gap = 5e-3
    grid = EnvelopeGrid.from_limits(gap, 800e-9, (650e-9, 1000e-9), 300e-15, thg=False)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=10e-6), 37.5e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-12),
        save_planes=[gap],
        dz_max=gap / 10,
        dz_init=gap / 10,
        backend="numba_cpu",
    )
    receiver = Simulation.from_capillary(
        75e-6,
        2e-3,
        "Ar",
        PRESSURE,
        lambda0=800e-9,
        grid=grid,
        modes=4,
        plasma=False,
        thg=False,
        noise=None,
        energy=1e-9,
        fwhm=30e-15,
        n_save=2,
        backend=RECEIVER,
    )
    port = receiver.input_port()
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    fundamental = {}
    for name, responses in (
        ("linear", []),
        ("kerr", [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)]),
    ):
        gas = Simulation.from_radial(
            grid, RadialGrid(4e-4, 96), ARGON, responses, pulse, settings=settings
        )
        plane = field_plane(gas.run(progress="none"), source=gas)
        coupled = couple(plane, port, model=SCALAR)
        received = receiver.with_input(coupled.launch).run(progress="none")
        assert_coupled(coupled.report, received)
        # The radial plane's own energy is the incident energy to the
        # source-metric item: M0 against the axial flux.
        assert coupled.report.incident_energy == pytest.approx(plane.energy(), rel=1e-3)
        launched = np.sum(weights * np.abs(coupled.launch.samples[:, 0, :]) ** 2, 1)
        fundamental[name] = launched[0] / np.sum(launched)
    assert fundamental["linear"] == pytest.approx(0.8715, abs=1e-3)
    assert fundamental["kerr"] - fundamental["linear"] > 5e-3


# ------------------------------------------- the mandatory Kerr/plasma/THG case


def test_a_field_grid_kerr_plasma_thg_entrance_is_transferred_with_its_harmonic():
    """A radial field-grid gap with Kerr, plasma and THG, then a capillary.

    2 mm of 1 bar argon, a 40 uJ, 30 fs, 30 um beam (about 9e13 W/cm^2 on axis) on
    a field grid whose band reaches 200 nm, so the third harmonic (near 267 nm) is
    generated in the gap; the gap runs on ``numpy_ref``, the capillary on the native
    CPU backend. The plasma takes 1.6e-3 of the energy (the Kerr-only gap keeps it
    to 1e-8, measured), so the ionisation response acts.

    The harmonic band, 230-300 nm, carries 82 nJ at the entrance plane. Its
    incident energy in the report is the plane's own band energy to the
    source-metric item; the capillary captures 0.998 of it and the run's entrance
    stores exactly the launch in that band. Its reference is transferred: the
    capillary's entrance plane has the source plane's reference, and the harmonic's
    energy-weighted arrival time in the launched modes is the source plane's to an
    attosecond (measured 131 as and 132 as from the window centre).
    """
    gap = 2e-3
    grid = FieldGrid.from_limits(gap, 800e-9, (200e-9, 1.6e-6), 150e-15)
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[gap],
        dz_max=gap / 10,
        dz_init=gap / 10,
        backend="numpy_ref",
        ionisation_field_range=(1e7, 1e11),
    )
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=40e-6), 30e-6
    )
    responses = [
        KerrResponse(gamma3("Ar", wavelength=800e-9), thg=True),
        PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)),
    ]
    medium = SpatialMedium("Ar", 1.0)
    gas = Simulation.from_radial(
        grid, RadialGrid(3e-4, 64), medium, responses, pulse, settings=settings
    )
    run = gas.run(progress="none")
    energy = np.asarray(run["stats/energy"])
    absorbed = 1 - energy[-1] / energy[0]
    assert 1e-3 < absorbed < 3e-3
    plane = field_plane(run, source=gas)
    start = field_plane(run, 0, source=gas)
    harmonic = (230e-9, 300e-9)
    generated = band_energy(plane.spectral_energy(), plane.omega, *harmonic)
    assert generated > 1e3 * band_energy(
        start.spectral_energy(), start.omega, *harmonic
    )
    assert 5e-8 < generated < 1.2e-7
    receiver = Simulation.from_capillary(
        75e-6,
        1e-3,
        "Ar",
        1.0,
        lambda0=800e-9,
        grid=grid,
        modes=4,
        plasma=False,
        noise=None,
        energy=1e-9,
        fwhm=30e-15,
        n_save=2,
        backend=RECEIVER,
    )
    coupled = couple(plane, receiver.input_port(), model=SCALAR)
    report = coupled.report
    received = receiver.with_input(coupled.launch).run(progress="none")
    assert_coupled(report, received)
    incident = band_energy(report.incident, plane.omega, *harmonic)
    assert report.captured is not None
    captured = band_energy(report.captured, plane.omega, *harmonic)
    launched = band_energy(report.launched, plane.omega, *harmonic)
    # M0 against the axial flux: the harmonic's obliquity is small (5e-6, measured).
    assert incident == pytest.approx(generated, rel=1e-4)
    assert 0.99 < captured / incident <= 1.0
    entrance = field_plane(received, 0)
    stored = band_energy(entrance.spectral_energy(), entrance.omega, *harmonic)
    assert stored == pytest.approx(launched, rel=LAUNCH_ROUNDING)
    assert entrance.reference == plane.reference
    # The harmonic band's energy-weighted arrival time, from the spectral phase
    # slope: in the optics convention t = sum Im(conj(S) dS/domega) / sum |S|^2.
    wavelength = 2 * math.pi * C / np.where(plane.omega > 0, plane.omega, np.inf)
    band = (wavelength >= harmonic[0]) & (wavelength <= harmonic[1])

    def arrival(samples, weights):
        values = samples[:, 0, band]
        slope = np.gradient(values, plane.omega[band], axis=-1)
        return float(
            np.sum(weights[:, None] * np.imag(np.conj(values) * slope))
            / np.sum(weights[:, None] * np.abs(values) ** 2)
        )

    space = plane.basis.space
    assert isinstance(space, RadialGrid)
    source_time = arrival(plane.samples, np.asarray(space.spectral_weights))
    launched_time = arrival(coupled.launch.samples, np.ones(4))
    # The coupling is a real, frequency-independent projection per mode: the
    # band's timing changes only by the different radial weighting of the
    # projection (measured 1.1 as of a 30 fs pulse).
    assert abs(launched_time - source_time) < 1e-17
    # The harmonic propagates on in the capillary (it is still generated there).
    exit_plane = field_plane(received)
    assert band_energy(exit_plane.spectral_energy(), exit_plane.omega, *harmonic) > 0


# ------------------------------------------------------------ (c) solid fibres


def lp01(radius, aperture, wavelength):
    """Return the weak-guidance LP01 profile function and its U, W."""
    v = 2 * math.pi / wavelength * radius * aperture
    from scipy.optimize import brentq  # noqa: PLC0415 - only this oracle needs it
    from scipy.special import j1, k1  # noqa: PLC0415

    u = brentq(
        lambda u: (
            u * j1(u) / j0(u)
            - math.sqrt(v**2 - u**2)
            * k1(math.sqrt(v**2 - u**2))
            / k0(math.sqrt(v**2 - u**2))
        ),
        0.1,
        min(v, 2.4048) - 1e-9,
    )
    w = math.sqrt(v**2 - u**2)

    def profile(r):
        inside = j0(u * r / radius)
        outside = j0(u) * k0(w * r / radius) / k0(w)
        return np.where(r <= radius, inside, outside)

    return profile


def test_a_solid_fibre_into_another_fibre():
    """(c) The admitted weak-guidance pair: a step-index fibre into a wider one.

    Contrast 0.005, core radii 2.18 and 2.57 um at 1064 nm, 1 mm each on the GNLSE
    route. The captured share is the LP01 overlap squared, here by an independent
    quadrature of the weak-guidance profiles (0.9947 launched, measured).
    """
    aperture = 1.45 * math.sqrt(1 - 0.99)

    def fibre(radius, backend):
        mode = StepIndexMode(
            radius,
            core=1.45,
            cladding=1.45 * math.sqrt(0.99),
            profile_wavelength=1064e-9,
        )
        return Simulation.from_gnlse(
            None,
            1e-3,
            fibre=mode,
            n2=2.6e-20,
            lambda0=1064e-9,
            wavelength_limits=(900e-9, 1300e-9),
            trange=400e-15,
            fwhm=100e-15,
            energy=1e-9,
            n_save=2,
            backend=backend,
        )

    source = fibre(2.18e-6, "numba_cpu")
    target = fibre(2.57e-6, RECEIVER)
    plane = field_plane(source.run(progress="none"), source=source)
    model = CouplingModel("scalar_paraxial", model_error=0.1)
    coupled = couple(plane, target.input_port(), model=model)
    received = target.with_input(coupled.launch).run(progress="none")
    report = coupled.report
    assert report.captured is not None
    first = lp01(2.18e-6, aperture, 1064e-9)
    second = lp01(2.57e-6, aperture, 1064e-9)

    # The integrands are ~1e-12 m^2: quad's default absolute tolerance would
    # swamp them, so only a relative one is set.
    rule = {"points": [2.18e-6, 2.57e-6], "limit": 400, "epsabs": 0, "epsrel": 1e-13}

    def norm(profile):
        return quad(lambda r: profile(r) ** 2 * r, 0, 30e-6, **rule)[0]

    overlap = quad(lambda r: first(r) * second(r) * r, 0, 30e-6, **rule)[0]
    expected = overlap**2 / (norm(first) * norm(second))
    # Two quadratures of one integral; the model's own is closed form (1e-10 here).
    captured = report.captured_energy
    assert captured is not None
    assert captured / report.incident_energy == pytest.approx(expected, rel=1e-8)
    model_budget = report.model_error
    assert model_budget is not None and report.budget <= model_budget
    uncaptured = report.uncaptured_energy
    assert uncaptured is not None and uncaptured > 0
    entrance = field_plane(received, 0)
    assert entrance.energy() == pytest.approx(
        report.launched_energy, rel=LAUNCH_ROUNDING
    )
    assert report.launched_energy < report.incident_energy


def test_a_gaussian_beam_into_a_solid_fibre():
    """(c') A Gaussian beam into a weakly guiding step-index fibre, then ten steps.

    Contrast 0.005 and V 2.2 at 1064 nm (core 2.57 um), the beam built in the core
    index at a 2.6 um waist. The captured share is the Gaussian--LP01 overlap
    squared, here by an independent quadrature, and the beam-into-fibre envelope
    is counted at its contrast-scaled allowance (plus the blue bins where V
    leaves its range).
    """
    radius = 2.57e-6
    aperture = 1.45 * math.sqrt(1 - 0.99)
    mode = StepIndexMode(
        radius, core=1.45, cladding=1.45 * math.sqrt(0.99), profile_wavelength=1064e-9
    )
    target = Simulation.from_gnlse(
        None,
        1e-3,
        fibre=mode,
        n2=2.6e-20,
        lambda0=1064e-9,
        wavelength_limits=(900e-9, 1300e-9),
        trange=400e-15,
        fwhm=100e-15,
        energy=1e-9,
        n_save=2,
        backend=RECEIVER,
    )
    port = target.input_port()
    waist = 2.6e-6
    beam = SpatialPulse(
        GaussianPulse(wavelength=1064e-9, fwhm=100e-15, energy=1e-9), waist
    )
    incident = FieldPlane.from_pulse(
        beam,
        port.grid,
        PlaneGeometry(0.0),
        space=RadialGrid(40 * radius, 256),
        index=np.asarray(port.reference_index),
    )
    coupled = couple(
        incident, port, model=CouplingModel("scalar_paraxial", model_error=0.1)
    )
    received = target.with_input(coupled.launch).run(progress="none")
    report = coupled.report
    assert_coupled(report, received)
    profile = lp01(radius, aperture, 1064e-9)
    rule = {"points": [radius], "limit": 400, "epsabs": 0, "epsrel": 1e-13}
    overlap = quad(
        lambda r: profile(r) * np.exp(-((r / waist) ** 2)) * r, 0, 40e-6, **rule
    )[0]
    norm = quad(lambda r: profile(r) ** 2 * r, 0, 40e-6, **rule)[0]
    expected = overlap**2 / (norm * waist**2 / 4)
    captured = report.captured_energy
    assert captured is not None
    # Two quadratures of one integral, the model's through the radial plane's
    # spectral representation of the sampled beam (its numerical item).
    assert captured / report.incident_energy == pytest.approx(expected, rel=1e-6)
    envelope = next(item for item in report.items if item.name == "envelope")
    assert envelope.value is not None
    # The contrast-scaled allowance; the blue bins beyond V 2.4 carry ~1e-20 of
    # this narrow pulse.
    assert envelope.value == pytest.approx(
        STEP_INDEX_BEAM_ENVELOPE.allowance({"contrast": 0.005}), rel=1e-9
    )
    assert envelope.note.startswith(STEP_INDEX_BEAM_ENVELOPE.identity)


# -------------------------------------------- (d) a Gaussian, (e) an analytic field


@pytest.fixture(scope="module")
def receiving_capillary():
    """A 75 um argon capillary with four HE1m modes, and its port."""
    simulation = capillary()
    return simulation, simulation.input_port()


def test_a_gaussian_beam_into_a_capillary(receiving_capillary):
    """(d) A Gaussian beam at w/a = 0.5 into four HE1m modes, then ten steps.

    The modal shares are the Gaussian--Bessel overlaps (0.906 and 0.092 of the
    incident energy); the launch carries what the modes capture, not the
    requested microjoule.
    """
    simulation, port = receiving_capillary
    beam = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 37.5e-6
    )
    incident = FieldPlane.from_pulse(
        beam, port.grid, PlaneGeometry(0.0), space=RadialGrid(400e-6, 256), medium=ARGON
    )
    coupled = couple(incident, port, model=SCALAR)
    received = simulation.with_input(coupled.launch).run(progress="none")
    assert_coupled(coupled.report, received)
    # The beam's requested energy is its axial flux; the report's incident energy
    # is M0 over the whole plane: they differ by the beam's obliquity.
    assert coupled.report.incident_energy == pytest.approx(1e-6, rel=1e-4)
    grid = port.grid
    weights = fourier.energy_weights(grid.n_omega, grid.dt, kind=grid.kind)
    launched = np.sum(weights * np.abs(coupled.launch.samples[:, 0, :]) ** 2, axis=1)
    shares = launched / coupled.report.incident_energy
    assert shares[0] == pytest.approx(0.906, abs=1e-3)
    assert shares[1] == pytest.approx(0.092, abs=1e-3)
    assert coupled.report.launched_energy < 1e-6


def test_an_analytic_field_with_a_chromatic_waist_into_a_capillary(
    receiving_capillary,
):
    """(e) A nonseparable analytic field, waist 0.6a at 800 nm scaling as 1/omega.

    Sampled by ``FieldPlane.from_analytic``; the captured share differs from bin
    to bin, and at each bin equals the four-mode Gaussian--Bessel overlap at that
    bin's waist, by an independent Gauss--Legendre integral.
    """
    simulation, port = receiving_capillary
    grid = port.grid
    radius = 75e-6
    waist = 0.6 * radius
    width = 30e-15 / (2 * math.sqrt(math.log(2)))

    def field(r, omega):
        local = waist * grid.omega0 / omega
        spectral = 1e-9 * np.exp(-(((omega - grid.omega0) * width) ** 2) / 2)
        return np.exp(-((r[:, None] / local[None, :]) ** 2)) * spectral[None, :]

    plane = FieldPlane.from_analytic(
        grid, PlaneGeometry(0.0), field, space=RadialGrid(400e-6, 256), medium=ARGON
    )
    assert plane.analytic is not None
    coupled = couple(plane, port, model=SCALAR)
    received = simulation.with_input(coupled.launch).run(progress="none")
    assert_coupled(coupled.report, received)
    numerical = next(item for item in coupled.report.items if item.name == "numerical")
    assert numerical.value is not None
    assert numerical.value >= plane.analytic.error
    nodes, weights = np.polynomial.legendre.leggauss(400)
    r = 0.5 * radius * (nodes + 1)
    area = 0.5 * radius * weights * r
    report = coupled.report
    assert report.captured is not None
    peak = int(np.argmax(report.incident))
    shares = []
    for k in (peak - 6, peak, peak + 6):
        local = waist * grid.omega0 / grid.omega[k]
        gaussian = np.exp(-((r / local) ** 2))
        expected = sum(
            np.sum(area * gaussian * j0(u * r / radius)) ** 2
            / (local**2 / 4 * np.sum(area * j0(u * r / radius) ** 2))
            for u in jn_zeros(0, 4)
        )
        share = report.captured[k] / report.incident[k]
        # Two quadratures of the same overlaps: rounding (measured 3e-14).
        assert share == pytest.approx(expected, abs=1e-10)
        shares.append(share)
    assert shares[0] != pytest.approx(shares[2], abs=1e-6)


# ------------------------------------------- (f) a Cartesian beam, no symmetry


def test_a_displaced_tilted_cartesian_beam_into_a_capillary(receiving_capillary):
    """(f) A Cartesian beam displaced by 15 um and tilted by 2 mrad, then coupled.

    The alignment is a transport element on a Cartesian plane; on a radial plane it
    is refused, since it would destroy the symmetry the grid imposes. The aligned
    beam couples less into the HE1m modes (0.792 against 0.9994 centred, measured),
    and its loss is the reported uncaptured energy, not restored. The same beam
    built directly displaced and tilted couples identically to the plane's
    representation error.
    """
    simulation, port = receiving_capillary
    grid = port.grid
    beam = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6), 37.5e-6
    )
    space = FreeGrid(300e-6, 128, 300e-6, 128)
    centred = FieldPlane.from_pulse(
        beam, grid, PlaneGeometry(0.0), space=space, medium=ARGON
    )
    alignment = Alignment(displacement=(15e-6, 0.0), tilt=(2e-3, 0.0), model_error=0.05)
    aligned, ledger = transport(centred, [alignment])
    assert_transported(ledger)
    coupled = couple(aligned, port, model=SCALAR)
    received = simulation.with_input(coupled.launch).run(progress="none")
    assert_coupled(coupled.report, received)
    reference = couple(centred, port, model=SCALAR).report.efficiency
    efficiency = coupled.report.efficiency
    assert reference is not None and efficiency is not None
    assert efficiency == pytest.approx(0.792, abs=2e-3)
    assert reference - efficiency > 0.2
    uncaptured = coupled.report.uncaptured_energy
    assert uncaptured is not None
    assert uncaptured > 0.2 * coupled.report.incident_energy
    width = 30e-15 / (2 * math.sqrt(math.log(2)))
    index = float(np.max(np.asarray(port.reference_index)))

    def direct(x, y, omega):
        k = index * omega / C
        envelope = np.exp(-(((omega - grid.omega0) * width) ** 2) / 2)
        spatial = np.exp(-((x - 15e-6) ** 2 + y**2) / 37.5e-6**2)
        return spatial[:, None] * np.exp(1j * np.outer(x, k) * 2e-3) * envelope[None, :]

    built = FieldPlane.from_analytic(
        grid, PlaneGeometry(0.0), direct, space=space, medium=ARGON
    )
    rebuilt = couple(built, port, model=SCALAR).report.efficiency
    assert rebuilt is not None
    # The tilt's k uses each bin's own index in the element and the carrier's
    # here; the shares agree to the plane's sampling and that difference.
    assert rebuilt == pytest.approx(efficiency, abs=2e-3)
    radial = FieldPlane.from_pulse(
        beam, grid, PlaneGeometry(0.0), space=RadialGrid(400e-6, 128), medium=ARGON
    )
    with pytest.raises(CouplingError, match="cylindrical symmetry"):
        transport(radial, [Alignment(displacement=(15e-6, 0.0))])


# ------------------------------------------------------- (g) a distant plane


def test_a_capillary_exit_at_a_distant_free_space_plane(capillary_exit):
    """(g) Outcouple, then 0.1, 0.2 and 0.4 m of argon onto scaled output planes.

    The Fresnel output reaches any observation window. Each output grid is twice
    the last on a twice longer path, so its nodes sit at the same angles: the
    intensity pattern then converges to the Fraunhofer limit, the mismatch between
    successive distances falling as the Fresnel number does. Energy is booked:
    the far planes carry the exit's energy less the Fresnel route's reported
    paraxial approximation error (2.1e-4 of it, measured).
    """
    _, plane = capillary_exit
    synthesised = Outcouple(RadialGrid(400e-6, 256), medium=ARGON).apply(plane)
    centre = int(np.argmax(synthesised.synthesised))
    patterns = []
    for distance in (0.1, 0.2, 0.4):
        far, ledger = transport(
            synthesised.plane,
            [
                FreeSpace(
                    distance,
                    ARGON,
                    output=RadialGrid(30e-3 * distance, 256),
                    output_method="fresnel",
                )
            ],
        )
        assert_transported(ledger)
        assert far.geometry.z == pytest.approx(plane.geometry.z + distance)
        assert abs(ledger.approximation_error) < 1e-3 * ledger.energy_in
        assert far.energy() == pytest.approx(ledger.energy_out, rel=1e-12)
        intensity = np.abs(far.node_spectra()[:, 0, centre]) ** 2
        patterns.append(distance**2 * intensity)
    first = np.max(np.abs(patterns[1] - patterns[0])) / np.max(patterns[1])
    second = np.max(np.abs(patterns[2] - patterns[1])) / np.max(patterns[2])
    assert second < 0.6 * first


# ----------------------------------- (h) SPM, a compressor, focusing, a capillary


def test_spm_then_a_compressor_and_a_4f_relay_into_a_second_capillary():
    """(h) Capillary SPM -> compressor -> 4f relay with a Fourier-plane aperture.

    20 mm of 3 bar argon at 100 uJ broadens the spectrum about sevenfold (measured
    6.9 in rms bandwidth) with positive chirp. The exit is synthesised, compressed
    by -50 fs^2 (a ``SpectralTransfer`` of an optics ``Chain``), relayed 1:1 by two
    20 mm lenses with a 300 um tanh aperture in the Fourier plane, and coupled into
    an identical capillary, which runs ten steps. The compressor is phase-only, so
    its ledger entry is zero, and it raises the launched fundamental's peak power
    (by 2.05, measured); the aperture's loss is booked as specified loss, the
    windows' angular removals as removed energy.
    """
    grid = {
        "lambda0": 800e-9,
        "wavelength_limits": (600e-9, 1100e-9),
        "trange": 300e-15,
    }
    first = capillary(length=0.02, energy=100e-6, **grid)
    run = first.run(progress="none")
    start, plane = field_plane(run, 0), field_plane(run, source=first)

    def bandwidth(samples):
        power = np.sum(np.abs(samples) ** 2, axis=(0, 1))
        omega = plane.omega
        mean = np.sum(omega * power) / np.sum(power)
        return math.sqrt(np.sum((omega - mean) ** 2 * power) / np.sum(power))

    assert bandwidth(plane.samples) > 5 * bandwidth(start.samples)
    second = capillary(energy=1e-9, backend=RECEIVER, **grid)
    port = second.input_port()
    synthesised = Outcouple(RadialGrid(1.5e-3, 512), medium=ARGON).apply(plane)
    focal = 0.02

    def gap():
        return FreeSpace(focal, ARGON, beyond_window="remove")

    relay = [
        gap(),
        ThinLens(focal, model_error=0.05),
        gap(),
        Aperture(300e-6, model_error=0.05, edge=20e-6, kind="tanh"),
        gap(),
        ThinLens(focal, model_error=0.05),
        gap(),
    ]
    peaks = {}
    for name, gdd in (("uncompressed", 0.0), ("compressed", -50e-30)):
        from solprop.optics import Chain, TaylorPhase  # noqa: PLC0415

        compressor = SpectralTransfer(Chain((TaylorPhase((0.0, 0.0, gdd), 800e-9),)))
        relayed, ledger = transport(
            synthesised.plane,
            [compressor, *relay],
            frame=first.frame,
            geometry=port.geometry,
        )
        assert_transported(ledger)
        assert ledger.elements[0].energy_in == pytest.approx(
            ledger.elements[0].energy_out, rel=1e-12
        )
        assert relayed.reference.calibrated == ("phase", "delay")
        aperture = ledger.elements[4]
        assert aperture.specified_loss > 0
        assert ledger.specified_loss == pytest.approx(
            aperture.specified_loss, rel=1e-12
        )
        coupled = couple(relayed, port, model=SCALAR)
        assert_coupled(coupled.report)
        efficiency = coupled.report.efficiency
        assert efficiency is not None and efficiency > 0.99
        field = np.fft.fft(coupled.launch.samples[0, 0, :])
        peaks[name] = float(np.max(np.abs(field) ** 2))
        if name == "compressed":
            received = second.with_input(coupled.launch).run(progress="none")
            assert_coupled(coupled.report, received)
            assert received.metadata is not None
            record = cast(
                "Mapping[str, object]", received.metadata.configuration["coupling"]
            )
            assert record["model"] == "scalar_paraxial"
    assert peaks["compressed"] > 1.8 * peaks["uncompressed"]
