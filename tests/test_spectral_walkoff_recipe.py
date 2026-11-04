# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The spectral walk-off absorber of the boundaries guide, and its per-band check.

The recipe is the caller's code on purpose, so these tests hold the physics the guide
states: where light wraps on a periodic window, what the strength removes, and that the
loss read per band from saved planes is the loss. Every run is linear and on a short
slab of glass, where the operator is exact, so what sets a tolerance is the controller's
``rtol``, a stated quadrature error, or a packet's own wings.
"""

import math

import numpy as np
import pytest

from solprop.boundaries import absorber, custom_absorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import FieldGrid, LineGrid
from solprop.physdata.glasses import fused_silica
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import PIController, RungeKutta
from solprop.translate.masks import temporal_rates

LENGTH = 40e-6
PROTECTED = 25e-15
SURVIVAL = 1e-6
MEDIUM = SpatialMedium(fused_silica())
SPACE = LineGrid(4e-3, 16)


def built_grid(length=LENGTH):
    """A window of about 128 fs: the requested 120 fs, with no room for a collar."""
    return FieldGrid.from_limits(
        length, 800e-9, (150e-9, 2e-6), 120e-15, collar_fraction=0
    )


GRID = built_grid()
WINDOW = GRID.n_t * GRID.dt
ARRIVAL = WINDOW - 2 * PROTECTED
ONSET = ARRIVAL - PROTECTED


def beta1(omega):
    index = np.real(MEDIUM.index(omega))
    return (index + omega * np.real(MEDIUM.dindex_domega(omega))) / C


def delays(grid=GRID, length=LENGTH):
    """``dtau(omega)`` of the guide's recipe, on the active bins."""
    delay = np.zeros(grid.n_omega)
    active = grid.active_mask
    frame = beta1(np.array([grid.omega0]))
    delay[active] = np.abs(beta1(grid.omega[active]) - frame) * length
    return delay


def recipe_rate(full, delay=None):
    """The guide's rate on ``GRID``: a rise over the ``h`` before ``W - 2h``, then full."""
    delay = delays() if delay is None else delay
    rise = np.clip((delay - ONSET) / PROTECTED, 0, 1)
    return full * 0.5 * (1 - np.cos(np.pi * rise))


def carrier_at(delay):
    """The wavelength on the blue side of the frame whose walk-off is ``delay``."""
    walked = delays()
    blue = np.flatnonzero(GRID.inband_mask & (GRID.omega > GRID.omega0))
    return 2 * np.pi * C / GRID.omega[blue[np.argmin(np.abs(walked[blue] - delay))]]


def launch(pulses, grid=GRID):
    """A wide beam carrying the sum of ``(wavelength, energy, t0)`` packets."""
    x = np.asarray(SPACE.x)
    beam = np.fft.fft(np.exp(-((x / 1e-3) ** 2)))
    packets = [
        GaussianPulse(wavelength=wavelength, fwhm=20e-15, energy=energy, t0=t0).build(
            grid
        )[0, 0]
        for wavelength, energy, t0 in pulses
    ]
    spectrum = np.sum(packets, axis=0)
    return beam[:, None, None] * spectrum[None, None, :]


def run(collar, field, *, grid=GRID, planes=32, backend="numpy_ref"):
    settings = SimulationSettings(
        controller=PIController(rtol=1e-7, atol=1e-14),
        save_planes=np.linspace(grid.length / planes, grid.length, planes),
        dz_max=grid.length / planes,
        dz_init=grid.length / planes,
        integrator=RungeKutta(),
        backend=backend,
        absorber=collar,
        boundary_check=None,
        stats=StatsSpec(names=("energy", "beam_spectrum")),
    )
    simulation = Simulation.from_free(
        grid, SPACE, MEDIUM, [], field, settings=settings, noise=None
    )
    return simulation.run(progress="none")


def spectral_only(rate=None, grid=GRID):
    collar = absorber(grid, temporal=False)
    return collar if rate is None else custom_absorber(grid, rate, collar=collar)


def protected_share(output, reference):
    """The field's energy inside ``|t| <= h`` over a reference's whole energy."""
    spectrum = np.asarray(output["field"][-1])[0, 0]
    # A saved field is in the optics convention, so its conjugate is the raw DFT.
    field = np.fft.irfft(np.conj(spectrum), n=GRID.n_t)
    times = np.fft.fftfreq(GRID.n_t, 1 / GRID.n_t) * GRID.dt
    whole = np.fft.irfft(np.conj(np.asarray(reference["field"][-1])[0, 0]), n=GRID.n_t)
    return float(np.sum(field[np.abs(times) <= PROTECTED] ** 2) / np.sum(whole**2))


def carried_energy(output, grid=GRID):
    """The energy in the spectrum at each plane; the end bins hold nothing."""
    return np.sum(np.asarray(output["stats/beam_spectrum"]), axis=-1) * grid.d_omega


@pytest.mark.parametrize(
    ("t0_over_h", "walk_over_h", "lowest", "highest"),
    [
        (0.0, -2.0, 0.0, 0.01),
        (-0.8, -1.5, 0.0, 0.01),
        (0.8, -1.5, 0.5, 1.0),
        (0.0, -0.5, 0.5, 1.0),
        (1.0, -2.0, 0.2, 0.8),
    ],
    ids=["short walk", "launched early", "launched late", "centre", "the onset"],
)
def test_light_lands_on_the_protected_region_where_the_geometry_says(
    t0_over_h, walk_over_h, lowest, highest
):
    """A packet launched at ``t0`` lands at ``t0 + dtau - W``, so the worst launch, at
    ``t0 = +h``, reaches the region's far edge at ``dtau = W - 2h``.

    The same walk-off, ``W - 1.5 h``, misses the region from one end of the scan and
    lands on it from the other, which is why the onset is the worst launch's. At the
    onset itself the packet straddles the edge, so about half of it is inside. No rate
    acts in these runs. The packets' 20 fs and the bins' spacing blur that edge, so the
    gates are coarse and the other cases sit at least a packet's width from it.
    """
    delay = WINDOW + walk_over_h * PROTECTED
    field = launch([(carrier_at(delay), 1e-9, t0_over_h * PROTECTED)])
    free = run(spectral_only(), field, planes=1)
    assert lowest <= protected_share(free, free) <= highest


def test_the_recipe_removes_what_would_wrap_and_touches_what_would_not():
    """Bins below the onset are as they were; a bin at full rate leaves ``f``."""
    full = math.log(1 / SURVIVAL) / LENGTH
    rate = recipe_rate(full)
    walked = delays()
    assert np.all(rate[walked <= ONSET] == 0.0)
    assert np.all(rate[walked >= ARRIVAL] == full)

    late = (carrier_at(ARRIVAL + 0.5 * PROTECTED), 1e-9, 0.8 * PROTECTED)
    safe = (carrier_at(0.5 * ONSET), 1e-9, 0.8 * PROTECTED)
    field = launch([safe, late])
    free = run(spectral_only(), field, planes=1)
    kept = run(spectral_only(rate), field, planes=1)
    below = (rate == 0.0) & GRID.active_mask
    free_spectrum = np.asarray(free["field"][-1])[0, 0]
    kept_spectrum = np.asarray(kept["field"][-1])[0, 0]
    # Both runs apply one exact operator to one launch, one step each: rounding apart.
    np.testing.assert_allclose(
        kept_spectrum[below],
        free_spectrum[below],
        rtol=0,
        atol=1e-12 * np.max(np.abs(free_spectrum)),
    )
    at_full = rate == full
    np.testing.assert_allclose(
        np.abs(kept_spectrum[at_full]) ** 2,
        SURVIVAL * np.abs(free_spectrum[at_full]) ** 2,
        rtol=1e-9,
        atol=1e-12 * SURVIVAL * np.max(np.abs(free_spectrum)) ** 2,
    )

    # And so the wrapped packet leaves the region: the free run has it there, of the
    # sum's energy, and the absorbed one holds only the safe packet's wing and the
    # edge's spreading of ``|beta_2| L / h``, three decades down.
    only_late = launch([late])
    free_late = run(spectral_only(), only_late, planes=1)
    kept_late = run(spectral_only(rate), only_late, planes=1)
    assert protected_share(free_late, free_late) > 0.5
    assert protected_share(kept_late, free_late) < 1e-3


def test_light_just_past_the_arrival_does_not_get_through():
    """What lands just inside the region is the case a rise must not cover.

    A packet launched at ``+h`` with a walk-off of ``W - 1.85 h`` lands at ``-0.85 h``.
    With the rise starting at ``W - 2h`` instead of ending there, about 15 % of the free
    run's energy stayed in the region. The guarantee is ``f``, whatever the packet.
    """
    full = math.log(1 / SURVIVAL) / LENGTH * delays()[GRID.inband_mask].max() / ARRIVAL
    field = launch([(carrier_at(ARRIVAL + 0.15 * PROTECTED), 1e-9, PROTECTED)])
    free = run(spectral_only(), field, planes=1)
    kept = run(spectral_only(recipe_rate(full)), field, planes=1)
    assert protected_share(free, free) > 0.5
    assert protected_share(kept, free) < SURVIVAL


def test_a_recipe_collar_compiles_no_temporal_pass():
    collar = spectral_only(recipe_rate(1e5))
    assert temporal_rates(collar) == (None, None)
    assert collar.temporal is False


def test_the_strength_holds_for_light_born_part_way():
    """Light of walk-off ``dtau`` born ``(W - 2h)/dtau`` from the exit just reaches the
    region, having lost that share of the exponent.

    The guide's ``alpha_f`` is sized so that this light still leaves ``f``. It is run as
    the same glass over only that distance, with the whole length's rate.
    """
    walk = ARRIVAL + 0.5 * PROTECTED
    full = math.log(1 / SURVIVAL) / LENGTH * walk / ARRIVAL
    walked = delays()
    wavelength = carrier_at(walk)
    bin_index = np.argmin(np.abs(GRID.omega - 2 * np.pi * C / wavelength))
    assert walked[bin_index] == pytest.approx(walk, rel=0.02)
    distance = LENGTH * ARRIVAL / walked[bin_index]
    grid = built_grid(distance)
    assert grid.n_t == GRID.n_t
    rate = recipe_rate(full, walked)
    field = launch([(wavelength, 1e-9, 0.0)], grid)
    collar = absorber(grid, temporal=False)
    free = run(collar, field, grid=grid, planes=1)
    kept = run(custom_absorber(grid, rate, collar=collar), field, grid=grid, planes=1)
    ratio = (
        np.abs(np.asarray(kept["field"][-1])[0, 0, bin_index]) ** 2
        / np.abs(np.asarray(free["field"][-1])[0, 0, bin_index]) ** 2
    )
    assert ratio == pytest.approx(SURVIVAL ** (walk / walked[bin_index]), rel=1e-6)


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
def test_the_per_band_loss_read_from_planes_is_the_loss(backend):
    """``alpha S`` integrated over saved planes against the exact exponential.

    Fused silica absorbs here, so a bin falls at ``alpha + alpha_m`` and the rate takes
    its share of that. The trapezoid's relative error ``(a dz)^2 / 12`` at the bin's
    whole rate ``a`` is the bound, computed and not quoted, and rounding of order
    ``1e-9`` of the bin sets its floor. The same sum over every bin, with the glass's
    share, closes the energy balance.
    """
    if backend == "native":
        pytest.importorskip("solprop_native")
    planes = 32
    rate = recipe_rate(math.log(1 / SURVIVAL) / LENGTH)
    wavelength = carrier_at(ONSET + 0.25 * PROTECTED)
    output = run(
        spectral_only(rate),
        launch([(wavelength, 1e-9, 0.0)]),
        planes=planes,
        backend=backend,
    )
    spectrum = np.asarray(output["stats/beam_spectrum"])
    z = np.asarray(output["z"])
    assert spectrum.shape == (planes + 1, GRID.n_omega)
    glass = np.zeros(GRID.n_omega)
    active = GRID.active_mask
    glass[active] = (
        2 * GRID.omega[active] * np.imag(MEDIUM.index(GRID.omega[active])) / C
    )
    whole = rate + glass
    weak = (spectrum[0] > 1e-3 * spectrum[0].max()) & (rate > 0.0)
    assert np.count_nonzero(weak) > 3
    lost = np.trapezoid(rate * spectrum, z, axis=0)
    exact = (
        spectrum[0, weak]
        * (rate[weak] / whole[weak])
        * -np.expm1(-whole[weak] * LENGTH)
    )
    bound = (whole[weak] * LENGTH / planes) ** 2 / 12 + 1e-9
    np.testing.assert_array_less(np.abs(lost[weak] - exact), 2 * bound * exact)
    energy = carried_energy(output)
    taken = np.trapezoid(whole * spectrum, z, axis=0)
    # The bins carrying the packet are weak, so the sum carries the same bound.
    assert np.sum(taken) * GRID.d_omega == pytest.approx(
        energy[0] - energy[-1], rel=2 * float(bound.max())
    )


def test_a_band_can_lose_most_of_itself_while_the_run_s_energy_barely_moves():
    """Why the check is per band: the total reads ``1e-5`` while the band reads tens of
    per cent."""
    rate = recipe_rate(math.log(1 / SURVIVAL) / LENGTH)
    weak = carrier_at(ONSET + 0.5 * PROTECTED)
    field = launch([(800e-9, 1e-9, 0.0), (weak, 1e-14, 0.0)])
    output = run(spectral_only(rate), field)
    spectrum = np.asarray(output["stats/beam_spectrum"])
    z = np.asarray(output["z"])
    energy = carried_energy(output)
    lost = np.asarray(np.trapezoid(rate * spectrum, z, axis=0))
    band = rate > 0.0
    share = lost[band].sum() / (lost[band].sum() + spectrum[-1, band].sum())
    assert (energy[0] - energy[-1]) / energy[0] < 2e-5
    assert share > 0.5


def test_the_user_s_rate_and_the_collar_s_split_on_the_host():
    """The collar's rate and the user's are one array in the operator; the caller, who
    holds its own, splits the per-bin product. The collar acts past the band."""
    rate = recipe_rate(math.log(1 / SURVIVAL) / LENGTH)
    boundary = spectral_only(rate)
    collar_rate = boundary.spectral_rate - rate
    plain = absorber(GRID, temporal=False).spectral_rate
    # One subtraction from the sum: a few ulps of the sum, not of the difference.
    assert np.all(
        np.abs(collar_rate - plain) <= 4 * np.finfo(float).eps * boundary.spectral_rate
    )
    output = run(boundary, launch([(160e-9, 1e-9, 0.0), (800e-9, 1e-9, 0.0)]))
    spectrum = np.asarray(output["stats/beam_spectrum"])
    z = np.asarray(output["z"])
    user = np.trapezoid(rate * spectrum, z, axis=0)
    collar = np.trapezoid(collar_rate * spectrum, z, axis=0)
    assert np.sum(user) > 0.0
    # Where only one of the two rates acts, its product is that bin's whole reading.
    only_user = (rate > 0.0) & (plain == 0.0)
    only_collar = (plain > 0.0) & (rate == 0.0)
    assert np.any(only_user)
    assert np.any(only_collar)
    assert np.all(collar[only_user] == 0.0)
    assert np.all(user[only_collar] == 0.0)
