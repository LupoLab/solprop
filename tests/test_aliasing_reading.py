# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The aliasing reading of the Cartesian chapter: the energy's share past ``k_max / 2``.

The case is a fringe pattern at ``k = +-5`` of the box's fundamental, under envelope
Kerr. Its cubic product at ``+-15`` is real on 32 samples and folds to ``-+1`` on 16.
Every wavenumber is far below the medium's, because the admittance in both spectra is
zero on an evanescent bin and would otherwise decide the reading.

**The size of what moves the reading.** Writing the field as ``A(t) 2 cos(kx)``,
``|E|^2 E`` is ``|A|^2 A (2 cos(3kx) + 6 cos(kx))``, so the product's amplitude grows
as ``gamma |A|^2 A z`` with ``gamma = 3 omega chi3 / (8 n c)``. A Gaussian's ``|A|^6``
over its ``|A|^2`` integrates to ``E0^4 / sqrt(3)``, so the product's share of the
energy is

    ``(gamma E0^2 L)^2 / sqrt(3)``.

That share is the distance from the reading's ideal 1 or 0 at the exit, so it sets the
tolerance, derived before anything was compared. Its 5 % covers what the expression
leaves out: the phase mismatch ``15^2 k_1^2 L / (2 n k_0)``, about 0.1 rad, and the
shock term, of order ``1 / (omega tau)``.
"""

import math

import numpy as np
import pytest

from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid
from solprop.nonlinear.responses import KerrResponse
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import PIController, RungeKutta

LENGTH = 1e-3
HALF_WIDTH = 1e-3
INDEX = 1.45
CHI3 = 2.2e-22
PEAK_FIELD = 2e8
GRID = EnvelopeGrid.from_limits(LENGTH, 800e-9, (600e-9, 1200e-9), 300e-15, thg=False)
FUNDAMENTAL = np.pi / HALF_WIDTH
GAMMA = 3 * GRID.omega0 * CHI3 / (8 * INDEX * C)
PRODUCT_SHARE = (GAMMA * PEAK_FIELD**2 * LENGTH) ** 2 / math.sqrt(3)
ROUNDING = 64 * np.finfo(float).eps  # a reading of a few sums, at the launch


def inner_box(space):
    """The chapter's box: half a sample short of ``k_max / 2``, whose column folds."""
    half_x = 0.5 * np.abs(space.k_x).max() - 0.5 * FUNDAMENTAL
    if space.n_y == 1:
        return np.array([[-half_x, half_x, -np.inf, np.inf]])
    half_y = 0.5 * np.abs(space.k_y).max() - 0.5 * FUNDAMENTAL
    return np.array([[-half_x, half_x, -half_y, half_y]])


def fringes(route, samples, mode=5):
    """The grid of a route, and its transverse pattern in physical space."""
    if route == "line":
        space = LineGrid(HALF_WIDTH, samples)
        pattern = 2 * np.cos(mode * FUNDAMENTAL * np.asarray(space.x))
        return space, np.fft.fft(pattern), samples
    nx, ny = (samples, 4) if route == "x" else (4, samples)
    space = FreeGrid(HALF_WIDTH, nx, HALF_WIDTH, ny)
    x, y = np.meshgrid(space.x, space.y)
    pattern = 2 * np.cos(mode * FUNDAMENTAL * (x if route == "x" else y))
    return space, np.fft.fft2(pattern).reshape(-1), samples


def reading_of(route, samples, *, backend="numpy_ref", chi3=CHI3, mode=5):
    """Run the fringes; return the reading per plane and the exit's column energies."""
    space, columns, _ = fringes(route, samples, mode)
    temporal = GaussianPulse(wavelength=800e-9, fwhm=60e-15, energy=1e-9).build(GRID)
    temporal = PEAK_FIELD * temporal[0, 0] / np.abs(np.fft.ifft(temporal[0, 0])).max()
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-16),
        save_planes=[LENGTH / 2, LENGTH],
        dz_max=LENGTH / 8,
        dz_init=LENGTH / 16,
        integrator=RungeKutta(),
        backend=backend,
        # The ramp would damp +-15 of 16 and not +-1, which are compared below.
        spatial_absorber=False,
        boundary_check=None,
        stats=StatsSpec(
            names=("beam_spectrum", "region_spectrum"),
            spectral_regions=inner_box(space),
        ),
    )
    output = Simulation.from_free(
        GRID,
        space,
        SpatialMedium(INDEX),
        [KerrResponse(chi3, thg=False)],
        columns[:, None, None] * temporal[None, None, :],
        settings=settings,
        noise=None,
    ).run(progress="none")
    beam = np.asarray(output["stats/beam_spectrum"]).sum(axis=-1)
    inside = np.asarray(output["stats/region_spectrum"])[:, 0].sum(axis=-1)
    assert np.all(beam > 0)  # no evanescent bin decides the reading
    return 1.0 - inside / beam, np.sum(np.abs(output["field"][-1]) ** 2, axis=(1, 2))


def share_at(energies, route, samples, mode):
    """The share of the exit's energy in columns ``+-mode`` along the route's axis."""
    stride = 4 if route == "y" else 1
    columns = (mode % samples) * stride, (-mode % samples) * stride
    return float(sum(energies[column] for column in columns) / np.sum(energies))


ROUTES = ["x", "y", "line"]


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
@pytest.mark.parametrize("route", ROUTES)
def test_the_reading_is_one_under_the_rule_s_limit_and_falls_under_refinement(
    route, backend
):
    if backend == "native":
        pytest.importorskip("solprop_native")
    coarse, _ = reading_of(route, 16, backend=backend)
    fine, _ = reading_of(route, 32, backend=backend)
    assert coarse[0] == pytest.approx(1.0, abs=ROUNDING)
    assert fine[0] == pytest.approx(0.0, abs=ROUNDING)
    assert 1.0 - coarse[-1] == pytest.approx(PRODUCT_SHARE, rel=0.05)
    assert fine[-1] == pytest.approx(PRODUCT_SHARE, rel=0.05)


@pytest.mark.parametrize("route", ROUTES)
def test_what_it_warns_of_is_there_on_the_coarse_grid_and_not_on_the_fine_one(route):
    """The fold itself, read from the columns: a one in the reading is not a proxy for
    nothing on this case, and the energy of the fold is the real product's."""
    _, coarse = reading_of(route, 16)
    _, fine = reading_of(route, 32)
    folded = share_at(coarse, route, 16, 1)
    real = share_at(fine, route, 32, 15)
    assert folded == pytest.approx(PRODUCT_SHARE, rel=0.05)
    assert real == pytest.approx(folded, rel=0.05)
    # On the fine grid, ``+-1`` is reached only through a further two orders: 25 folds
    # to -7, and so on. The square of the share bounds that.
    assert share_at(fine, route, 32, 1) < PRODUCT_SHARE**2


@pytest.mark.parametrize("route", ROUTES)
def test_a_field_exactly_at_the_limit_reads_as_past_it(route):
    """``+-4`` of 16: the product at ``+-12`` folds to ``-+4``, onto the band's edge."""
    reading, _ = reading_of(route, 16, mode=4)
    assert reading[0] == pytest.approx(1.0, abs=ROUNDING)


def test_the_source_near_the_cutoff_would_read_zero_on_this_case():
    """Why the chapter reads the field past ``k_max / 2``, not the source at cutoff."""
    samples = 16
    x = 2 * np.pi * np.arange(samples) / samples
    field = np.exp(5j * x) + np.exp(-5j * x)
    source = np.fft.fft(np.abs(field) ** 2 * field) / samples
    modes = np.fft.fftfreq(samples, 1 / samples)
    amplitudes = {
        int(mode): abs(value) for mode, value in zip(modes, source, strict=True)
    }
    assert amplitudes[1] == pytest.approx(1.0)
    assert amplitudes[-1] == pytest.approx(1.0)
    assert amplitudes[5] == pytest.approx(3.0)
    near_cutoff = np.abs(modes) >= 0.75 * samples / 2
    assert np.sum(np.abs(source[near_cutoff]) ** 2) < 1e-25
    # While the field past the half band reads all of it.
    spectrum = np.fft.fft(field) / samples
    past = np.abs(modes) > samples / 4
    assert np.sum(np.abs(spectrum[past]) ** 2) == pytest.approx(
        np.sum(np.abs(spectrum) ** 2)
    )


@pytest.mark.parametrize(
    ("order", "fraction"), [(3, 1 / 2), (2, 2 / 3)], ids=["cubic", "quadratic"]
)
def test_a_product_folds_in_band_past_its_own_limit(order, fraction):
    """``K >= 2 k_max / (p + 1)``: band edges either side of it, on integer modes."""
    samples = 1024
    k_max = samples // 2
    limit = fraction * k_max
    edges = [(math.floor(limit) - 1, False), (math.ceil(limit) + 1, True)]
    if limit.is_integer():
        edges.append((int(limit), True))  # exactly at it, the image is the band's edge
    for band, lands in edges:
        x = 2 * np.pi * np.arange(samples) / samples
        # The widest product of the edge with itself.
        field = np.exp(1j * band * x)
        product = field**order
        image = int(
            np.fft.fftfreq(samples, 1 / samples)[np.argmax(np.abs(np.fft.fft(product)))]
        )
        assert (abs(image) <= band) == lands
