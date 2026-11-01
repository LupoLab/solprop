# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A protected weak signal at an angle (SEP 40 sections 4.4 and 7.5, Part II's witness).

Part II's witness is a four-wave-mixing sideband in fused silica on the line route at
256 samples: a pump on axis at 800 nm and a seed at 727.5 nm tilted by 1.5e5 rad/m
phase-match an idler at 888.6 nm and the opposite tilt, about 1e-6 of the pump. Its
nonlinear run, with the reference four times wider that it needs, is too heavy for the
test tiers on ``numpy_ref``; this is its linear equivalent. A weak packet at the
sideband's wavelength and transverse wavenumber, 1e-3 of the energy, is launched beside
a strong main beam on axis, and propagated linearly, so what reaches the collar is the
same light at the same angle, without the generation.

Set-up, as Part II's: ``LineGrid(500 um, 256, collar_fraction=0.7)``, a collar 350 um
wide per face in a periodic half-width of 850 um; the requested band 690-940 nm with a
1.8 ps window (1024 bins); 10 cm; noise off. The main beam has a 1/e field half-width
of 100 um and 150 fs at 800 nm, the packet 80 um and 150 fs at 888.6 nm. Bare
reference: a line four times wider at the same spacing, with no transverse boundary,
whose nodes contain the design's.

solprop propagates as ``exp(-i k_z z)``, so the packet ``exp(+i kappa x)`` travels
towards -x; what the collar transmits wraps and re-enters the aperture from +x still
travelling towards -x, at +kappa in the aperture's far field, and what it reflects
appears at -kappa. The returned energy is the error against the reference in the
sideband's band, in the aperture's far field at ``0.4 kappa < k_x < 1.6 kappa``, over
the packet's energy. The design's report, section 4.4, is ``exp(-2 Gamma)`` with the
exact symbol at each (kappa, omega), weighted by the packet's far-field spectrum, which
a linear flow leaves unchanged.
"""

import math
import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from solprop.boundaries import SpatialAbsorber, SpatialAbsorberDesign
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import EnvelopeGrid, LineGrid
from solprop.monitor import BoundaryCheck
from solprop.output.schema import PlaneStore
from solprop.physdata.glasses import fused_silica
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialMedium
from solprop.spatial_absorber import collar_attenuation
from solprop.stepper import Multistep, PIController

C = 299792458.0
LENGTH = 0.1
RADIUS, FRACTION, SAMPLES = 500e-6, 0.7, 256
KAPPA = 1.5e5
"""The sideband's transverse wavenumber, rad/m."""
SIDEBAND = 888.6e-9
SHARE = 1e-3
"""The packet's energy over the main beam's."""
BAND = (870e-9, 910e-9)
"""The sideband's band, which holds none of the main beam."""
PROTECTED = math.asin(2e5 * 940e-9 / (2 * math.pi))
"""The angle whose band edge at the reddest requested wavelength is Part II's
protected design's, 2e5 rad/m."""
OUTSIDE = 7.5e4
"""Part II's band edge that declares the sideband outside: half its wavenumber."""
PADDED = 8192


def time_grid() -> EnvelopeGrid:
    """Return the envelope grid: 690-940 nm requested, a 1.8 ps window."""
    return EnvelopeGrid.from_limits(LENGTH, 800e-9, (690e-9, 940e-9), 1.8e-12)


def packet_columns(space: LineGrid) -> np.ndarray:
    """Return the packet's transverse profile on `space`'s spectral columns."""
    x = np.asarray(space.x)
    return math.sqrt(SHARE) * np.fft.fft(np.exp(-((x / 80e-6) ** 2) + 1j * KAPPA * x))


def spectra(grid: EnvelopeGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return the main beam's and the packet's spectra on `grid`."""
    main, side = (
        GaussianPulse(wavelength=wavelength, fwhm=150e-15, energy=1e-9).build(grid)[
            0, 0
        ]
        for wavelength in (800e-9, SIDEBAND)
    )
    return main, side


def run(
    space: LineGrid,
    specification: SpatialAbsorber | bool,
    check: BoundaryCheck | None = BoundaryCheck(),  # noqa: B008 - immutable
) -> tuple[Simulation, PlaneStore]:
    """Run the main beam and the packet on `space` linearly to L.

    ``Multistep`` at ``rtol = 1e-6``: the returned light is about 1e-13 of the launch
    in energy, and ``rtol = 1e-8`` changes it by under 1 %.
    """
    grid = time_grid()
    main, side = spectra(grid)
    x = np.asarray(space.x)
    beam = np.fft.fft(np.exp(-((x / 100e-6) ** 2)))
    launch = (
        beam[:, None, None] * main[None, None, :]
        + packet_columns(space)[:, None, None] * side[None, None, :]
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=0.0),
        save_planes=[LENGTH],
        dz_max=LENGTH,
        dz_init=LENGTH * 1e-4,
        integrator=Multistep(),
        backend="numpy_ref",
        spatial_absorber=specification,  # pyright: ignore[reportArgumentType]
        boundary_check=check,
    )
    simulation = Simulation.from_free(
        grid,
        space,
        SpatialMedium(fused_silica()),
        [],
        launch,
        settings=settings,
        noise=None,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        output = simulation.run(progress="none")
    assert isinstance(output, PlaneStore)
    return simulation, output


def design_grid() -> LineGrid:
    """Return the design's line: 500 um requested, 350 um of collar room per face."""
    return LineGrid(RADIUS, SAMPLES, collar_fraction=FRACTION)


def sideband_bins(grid: EnvelopeGrid) -> np.ndarray:
    """Return the bins in the sideband's band."""
    wavelength = 2 * math.pi * C / np.asarray(grid.omega)
    return (wavelength >= BAND[0]) & (wavelength <= BAND[1])


def aperture_field(output: PlaneStore, space: LineGrid) -> np.ndarray:
    """Return the field at L on `space`'s nodes, sideband bins only."""
    field = np.fft.ifft(np.asarray(output["field"][-1])[:, 0, :], axis=0)
    return field[:, sideband_bins(time_grid())]


@dataclass(frozen=True)
class Reference:
    """The bare reference on the design's nodes, and the packet's energy.

    Attributes
    ----------
    field : numpy.ndarray
        The reference's sideband bins at L on the design's nodes.
    energy : float
        The packet's launched energy, J/m.
    """

    field: np.ndarray
    energy: float


@pytest.fixture(scope="module")
def reference() -> Reference:
    """Return the bare reference, four times wider at the same spacing."""
    space = design_grid()
    wide = LineGrid(4 * RADIUS * (1 + FRACTION), 4 * SAMPLES)
    first = (wide.n_x - space.n_x) // 2
    np.testing.assert_array_equal(
        np.asarray(wide.x)[first : first + space.n_x], np.asarray(space.x)
    )
    _, output = run(wide, False)
    field = aperture_field(output, wide)[first : first + space.n_x]
    _, side = spectra(time_grid())
    dx = float(space.x[1] - space.x[0])
    profile = np.fft.ifft(packet_columns(space))
    energy = float(np.sum(np.abs(profile) ** 2) * np.sum(np.abs(side) ** 2)) * dx
    return Reference(field, energy)


def returned(output: PlaneStore, reference: Reference) -> tuple[float, float]:
    """Return the energy returned into the aperture at L, over the packet's.

    Returns
    -------
    tuple of float
        The transmitted return, at +kappa in the far field, and the reflection, at
        -kappa.
    """
    space = design_grid()
    aperture = np.abs(np.asarray(space.x)) <= RADIUS
    error = (aperture_field(output, space) - reference.field)[aperture]
    padded = np.zeros((PADDED, error.shape[1]), dtype=complex)
    padded[: error.shape[0]] = error
    dx = float(space.x[1] - space.x[0])
    far = np.abs(np.fft.fft(padded, axis=0)) ** 2 * dx / PADDED
    k_x = 2 * math.pi * np.fft.fftfreq(PADDED, dx)
    back = (k_x > 0.4 * KAPPA) & (k_x < 1.6 * KAPPA)
    reflected = (k_x < -0.4 * KAPPA) & (k_x > -1.6 * KAPPA)
    return (
        float(np.sum(far[back])) / reference.energy,
        float(np.sum(far[reflected])) / reference.energy,
    )


def report(design: SpatialAbsorberDesign) -> float:
    """Return section 4.4's report for the packet on `design`.

    ``exp(-2 Gamma)`` with the exact symbol at each column and frequency, at the
    design's peak rate and width, weighted by the packet's far-field spectrum.
    """
    grid = time_grid()
    space = design_grid()
    bins = sideband_bins(grid)
    omega = np.asarray(grid.omega)[bins]
    k = np.real(np.asarray(SpatialMedium(fused_silica()).index(omega, 0.0))) * omega / C
    k_x = np.asarray(space.k_x)
    columns = (k_x > 0.4 * KAPPA) & (k_x < 1.6 * KAPPA)
    _, side = spectra(grid)
    weights = (np.abs(packet_columns(space)[columns]) ** 2)[:, None] * (
        np.abs(side[bins]) ** 2
    )[None, :]
    kappa, wavenumber = np.broadcast_arrays(k_x[columns][:, None], k[None, :])
    gamma = collar_attenuation(kappa, wavenumber, design.alpha0, design.width[0])
    return float(np.sum(weights * np.exp(-2 * gamma)) / np.sum(weights))


def transverse(output: PlaneStore) -> dict[str, Any]:
    """Return a finished run's transverse edge record."""
    assert output.outcome is not None
    monitor = output.outcome.execution["boundary_monitor"]
    assert isinstance(monitor, dict)
    return cast("dict[str, Any]", monitor["transverse"])


@dataclass(frozen=True)
class Collared:
    """A run on a requested collar.

    Attributes
    ----------
    design : SpatialAbsorberDesign
        Its resolved design.
    output : PlaneStore
        The run.
    """

    design: SpatialAbsorberDesign
    output: PlaneStore


def collared(specification: SpatialAbsorber) -> Collared:
    """Run on the design's line with a requested collar."""
    simulation, output = run(design_grid(), specification)
    design = simulation.translate().transverse
    assert design is not None
    assert design.has_collar
    return Collared(design, output)


@pytest.fixture(scope="module")
def outside() -> Collared:
    """Return the run whose band is declared below the sideband."""
    return collared(SpatialAbsorber(kappa=OUTSIDE))


@pytest.mark.medium
def test_a_protected_sideband_returns_what_the_design_reports(
    reference: Reference,
) -> None:
    # Part II: returned energy over the report 0.96, +-10 % (measured 1-4 %, doubled
    # for the grid). The declared angle covers the sideband at every frequency of its
    # band, and the run reads clear: what returns is about 2e-13 of the launch.
    protected = collared(SpatialAbsorber(angle=PROTECTED))
    reddest = 2 * math.pi * math.sin(PROTECTED) / BAND[1]
    assert reddest > KAPPA
    back, reflected = returned(protected.output, reference)
    assert back / report(protected.design) == pytest.approx(0.96, rel=0.1, abs=0.0)
    assert reflected < back
    assert protected.design.check["status"] == "passed"
    assert transverse(protected.output)["transverse_status"] == "clear"


@pytest.mark.medium
def test_a_sideband_declared_outside_the_band_returns_what_the_design_reports(
    reference: Reference, outside: Collared
) -> None:
    # Part II: 1.006, +-10 %. The sideband lies in the intermediate band, between the
    # protected edge and the safety ramp, and returns about 1e-4 of itself.
    design = outside.design
    assert design.kappa_p[0] < KAPPA < design.kappa_s[0]
    back, _ = returned(outside.output, reference)
    assert back / report(design) == pytest.approx(1.006, rel=0.1, abs=0.0)
    assert transverse(outside.output)["transverse_status"] == "clear"


@pytest.mark.medium
def test_the_design_s_own_intermediate_report_predicts_the_return(
    reference: Reference, outside: Collared
) -> None:
    # Section 4.4: the report predicts the returned energy to 1-4 %. The packet is
    # the launch's only content in the intermediate band, so the design's record, a
    # share of the whole launch, is the packet's return times the launch's share in
    # that band, read from the launch itself (the packet's far wings lie outside it).
    back, _ = returned(outside.output, reference)
    space = design_grid()
    grid = time_grid()
    design = outside.design
    launch = np.asarray(outside.output["field"][0])
    band = grid.inband_mask & grid.active_mask
    power = np.sum(np.abs(launch[..., band]) ** 2, axis=(1, 2))
    magnitude = np.abs(np.asarray(space.k_x))
    between = (magnitude > OUTSIDE) & (magnitude < design.kappa_s[0])
    share = float(np.sum(power[between])) / float(np.sum(power))
    notes = design.notes
    assert notes["intermediate_return"] == pytest.approx(back * share, rel=0.1, abs=0.0)


def test_a_wrapped_sideband_is_wrong_by_its_own_field(reference: Reference) -> None:
    # Part II: wrong by 0.97 of its own field, asserted above 0.5 (any wrap). The
    # same line with its collar room inert: the packet leaves through one face and
    # comes back through the other. The edge's budget is relaxed so that the run
    # reaches L; time keeps its own.
    space = design_grid()
    _, output = run(space, False, BoundaryCheck(transverse_budget=0.1))
    aperture = np.abs(np.asarray(space.x)) <= RADIUS
    error = (aperture_field(output, space) - reference.field)[aperture]
    dx = float(space.x[1] - space.x[0])
    assert math.sqrt(float(np.sum(np.abs(error) ** 2)) * dx / reference.energy) > 0.5


def test_a_wrapped_seed_this_strong_is_stopped() -> None:
    # At 1e-3 of the launch the packet is visible to the launch-normalised edge,
    # which stops the run as the packet crosses the wrap; Part II's sideband, about
    # 1e-6 of the pump, is not, and the collar's design is what protects it.
    with pytest.raises(BoundaryError, match="reached the transverse edge"):
        run(design_grid(), False)
