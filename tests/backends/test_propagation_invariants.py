# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Time reversal and photon-number conservation, per geometry family, on each backend.

Both are invariants of the lossless, noise-free problem without an absorber
(``docs/dev/testing.md`` section 1), checked here on short runs of every family: the
mode-averaged GNLSE, the multimode capillary, radial, Cartesian and line free space
and the hybrid planar guide.

**Time reversal.** With real ``beta(omega)`` and an instantaneous response, the
propagation equation is unchanged by ``z -> L - z`` together with ``t -> -t`` and
complex conjugation: if ``a(z)`` solves it, so does ``conj(a(L - z))``. In the stored
internal spectrum, ``t -> -t`` with conjugation is the conjugation of every bin,
whatever the dispersion (odd orders included), on field and envelope grids alike. So
launching ``conj(a(L))`` into the *same* problem returns ``conj(a(0))`` after ``L``.
solprop has no negative length, but the GNLSE can also be reversed without the
conjugation, by negating ``beta`` and ``gamma`` together; negating only one of them is
the defect a reversal test must see. A delayed (Raman) response is causal, and its
reversal would need the anti-causal one, so a Raman run must fail the same bound.

**Photon number.** With the ``omega`` prefactor (the shock term) an instantaneous Kerr
response conserves the energy and the photon number
``sum_k S_k d_omega / (hbar omega_k)``; a Raman response moves photons to the red, so
it keeps the photon number and costs energy. The measure is the library's energy per
bin: :func:`solprop.stats.spectral_flux` on a modal route and
:func:`solprop.spatial.spatial_energy_spectrum`, the axial Poynting flux, on a spatial
one.

**The bounds.** An accepted step's embedded error estimate is at most ``rtol`` times
the state's norm (the weak Euclidean norm with ``atol = 0``, ``docs/dev/stepper.md``
section 3), and the estimate bounds the step's own error: the stated assumption of
both bounds.

- A quadratic measure ``Q(a) = sum_i w_i |a_i|^2`` that the exact flow conserves
  drifts only by what each step's local error ``e_j`` changes it by, with no growth
  from step to step. By Cauchy-Schwarz on ``sum_i w_i conj(a_i) e_i``,
  ``|dQ| / Q <= 2 kappa sum_j ||e_j|| / ||a||`` with
  ``kappa^2 = max_i w_i ||a||^2 / Q(a)``, which converts the controller's norm into
  the measure's; it is computed from the measure itself, at the launch, over the bins
  the controller scores. Hence ``2 kappa S rtol`` for ``S`` accepted steps, plus the
  same factor on the march's rounding (:func:`march_bounds.march_bound`).
- A reversal adds the two legs' global errors. Each step's local error is carried by
  at most ``exp(Phi)``, ``Phi`` the launch's nonlinear phase
  (:func:`march_bounds.measured_scales`, floored at a tenth as
  :func:`march_bounds.simulation_bound` floors it), and the return leg carries the
  forward leg's error by ``exp(Phi)`` again: ``exp(Phi) (exp(Phi) S_f + S_b) rtol``,
  plus the march's rounding.

Both take :data:`march_bounds.MARGIN`. The local estimate is pessimistic for these
high-order steppers, so the measured departures sit far inside the bounds; the
defects below land 47 to 190000 times outside them.

Medium tier: each case is a real run of six to forty steps on a small grid.
"""

import dataclasses
import functools
import math
from collections.abc import Callable

import numpy as np
import pytest

from bounds import assert_within, higham_gamma
from capability_routes import CAPILLARY_GRID, LENGTH, SPATIAL_GRID, STEP
from march_bounds import MARGIN, march_bound, measured_scales
from solprop import fourier
from solprop.boundaries import no_absorber
from solprop.constants import HBAR
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid, LineGrid, RadialGrid
from solprop.modes import PlanarGrid, SlabMode
from solprop.noise import photons_per_bin
from solprop.nonlinear.raman import response_for
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata import MaterialSelection, gamma3
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialFlux, SpatialMedium, spatial_energy_spectrum
from solprop.stats import StatsSpec, spectral_flux
from solprop.stepper import PIController

pytestmark = pytest.mark.medium

BACKENDS = ("native", "numpy_ref")

REVERSAL_RTOL = 1e-8
"""The reversal runs' tolerance: tens of steps a leg on the GNLSE, a few elsewhere."""

CONSERVATION_RTOL = 1e-10
"""The conservation runs' tolerance, two decades below the reversal's so that the
bound sits well under what a perturbed shock term or a loss moves the measure by."""

GAS = "N2"
"""The fill of every gas case: a molecule, so that each geometry carries its Raman."""

WAVELENGTH = 800e-9

# The GNLSE case: the third-order soliton of test_numpy_ref_raman.py's photon test over
# a twentieth of a metre of the fibre of tests/backends/conftest.py, with a third-order
# dispersion so that the reversal is seen to undo odd orders as well.
GNLSE_LENGTH = 0.05
GNLSE_BETA2 = -1.183e-26
GNLSE_BETA3 = 1e-41
GNLSE_GAMMA = 0.11
GNLSE_WIDTH = 100e-15
GNLSE_PEAK = 9 * abs(GNLSE_BETA2) / (GNLSE_GAMMA * GNLSE_WIDTH**2)


# ------------------------------------------------------------------------ the families


def _gas_raman():
    """Return the fill's molecular Raman response, without the third harmonic."""
    selection = MaterialSelection(carrier_wavelength=WAVELENGTH)
    return response_for(GAS, thg=False, selection=selection)


def _gas_responses(raman: bool) -> list[object]:
    """Return the fill's Kerr response, and its Raman response if asked for."""
    kerr = KerrResponse(gamma3(GAS, wavelength=WAVELENGTH), thg=False)
    return [kerr, _gas_raman()] if raman else [kerr]


def _spatial_settings() -> SimulationSettings:
    """Return the spatial routes' steps, as tests/capability_routes.py sets them."""
    return SimulationSettings(
        save_planes=[LENGTH],
        dz_max=STEP,
        dz_init=STEP,
        # A beam far inside these small transverse grids: the real-space collar would
        # be a loss, and its design refuses grids this coarse (capability_routes.py).
        spatial_absorber=False,
    )


def gnlse(raman: bool, *, sign: float = 1.0, beta_sign: float = 1.0) -> Simulation:
    """Return the GNLSE soliton; `sign` negates beta and gamma, `beta_sign` beta."""
    beta = sign * beta_sign
    grid = EnvelopeGrid.from_limits(GNLSE_LENGTH, 835e-9, (400e-9, 3e-6), 4e-12)
    return Simulation.from_gnlse(
        sign * GNLSE_GAMMA,
        GNLSE_LENGTH,
        [0.0, 0.0, beta * GNLSE_BETA2, beta * GNLSE_BETA3],
        wavelength=835e-9,
        grid=grid,
        width=GNLSE_WIDTH,
        peak_power=GNLSE_PEAK,
        shape="sech",
        shock=True,
        raman=raman,
        noise=None,
        save_planes=[GNLSE_LENGTH],
        absorber=False,
    )


def capillary(raman: bool) -> Simulation:
    """Return three lossless HE1m modes of a nitrogen capillary on a field grid."""
    return Simulation.from_capillary(
        100e-6,
        LENGTH,
        GAS,
        1.0,
        wavelength=WAVELENGTH,
        grid=CAPILLARY_GRID,
        fwhm=30e-15,
        energy=1e-5,
        modes=3,
        loss=False,
        plasma=False,
        raman=raman,
        absorber=False,
        noise=None,
        save_planes=[LENGTH],
        dz_max=STEP,
        dz_init=STEP,
    )


def _free(
    space: RadialGrid | FreeGrid,
    raman: bool,
    energy: float,
    *,
    lossy: bool = False,
) -> Simulation:
    """Return an 80 um nitrogen beam on a free-space grid.

    `lossy` replaces the gas by a constant index near the gas's, with an imaginary part
    of ``1e-9``: a power loss of ``2 omega Im(n) / c``, ``0.016 /m`` at the carrier,
    beside the gas's own Kerr susceptibility at 1 bar (absolute, as a constant index
    takes it).
    """
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=30e-15, energy=energy)
    medium = SpatialMedium(GAS, 1.0)
    responses = _gas_responses(raman)
    if lossy:
        chi3 = gamma3(GAS, wavelength=WAVELENGTH) * float(
            medium.density(np.zeros(1))[0]
        )
        medium = SpatialMedium(complex(1.0003, 1e-9))
        responses = [KerrResponse(chi3, thg=False)]
    return Simulation.from_free(
        SPATIAL_GRID,
        space,
        medium,
        responses,
        SpatialPulse(pulse, 80e-6),
        settings=_spatial_settings(),
        absorber=False,
        noise=None,
    )


def radial(raman: bool, *, lossy: bool = False) -> Simulation:
    """Return the radial beam: 30 uJ, 32 Hankel nodes."""
    return _free(RadialGrid(6e-4, 32), raman, 3e-5, lossy=lossy)


def cartesian(raman: bool, *, lossy: bool = False) -> Simulation:
    """Return the Cartesian beam: 30 uJ on a 12 x 16 plane."""
    space = FreeGrid(3e-4, 12, 4e-4, 16, collar_fraction=0)
    return _free(space, raman, 3e-5, lossy=lossy)


def line(raman: bool, *, lossy: bool = False) -> Simulation:
    """Return the line beam: 0.3 J per metre of y, about the radial beam's intensity."""
    return _free(LineGrid(3e-4, 16, collar_fraction=0), raman, 0.3, lossy=lossy)


HALF_GAP = 63.5e-6


def planar(raman: bool, *, lossy: bool = False) -> Simulation:
    """Return the hybrid planar guide: two lossless slab modes of a nitrogen gap.

    `lossy` gives the modes a hundredth of their modelled wall loss.
    """
    grid = EnvelopeGrid.from_limits(LENGTH, WAVELENGTH, (650e-9, 1000e-9), 600e-15)
    loss = 0.01 if lossy else False
    modes = [SlabMode(HALF_GAP, GAS, 1.0, order=p, loss=loss) for p in (1, 3)]
    pulse = PlanarPulse(
        GaussianPulse(wavelength=WAVELENGTH, fwhm=100e-15, energy=1e-3),
        400e-6,
        waist_x=0.735 * HALF_GAP,
    )
    return Simulation.from_planar(
        grid,
        PlanarGrid(modes, 1e-3, 8),
        _gas_responses(raman),
        pulse,
        settings=_spatial_settings(),
        absorber=False,
    )


FAMILIES: dict[str, Callable[[bool], Simulation]] = {
    "gnlse": gnlse,
    "capillary": capillary,
    "radial": radial,
    "cartesian": cartesian,
    "line": line,
    "planar": planar,
}
"""Each geometry family's lossless case, by whether it carries the Raman response."""

LOSSY: dict[str, Callable[[], Simulation]] = {
    "radial": functools.partial(radial, False, lossy=True),
    "cartesian": functools.partial(cartesian, False, lossy=True),
    "line": functools.partial(line, False, lossy=True),
    "planar": functools.partial(planar, False, lossy=True),
}
"""Each spatial family with a small loss and the Kerr response alone."""


def controlled(
    simulation: Simulation,
    backend: str,
    rtol: float,
    *,
    shock: bool = True,
) -> Simulation:
    """Return `simulation` on `backend` at `rtol`, unabsorbed, recording its energy.

    ``atol = 0`` makes the controller's scale ``rtol ||a||`` exactly, which is what the
    bounds assume; `shock` False freezes the prefactor's ``omega`` at the carrier.
    """
    simulation.settings = dataclasses.replace(
        simulation.settings,
        backend=backend,
        controller=PIController(rtol=rtol, atol=0.0),
        absorber=no_absorber(simulation.grid),
        shock=shock,
        stats=StatsSpec(names=("energy",)),
    )
    return simulation


# ------------------------------------------------------------------------- the measure


def internal(simulation: Simulation, field: np.ndarray) -> np.ndarray:
    """Return a saved optics-convention plane as the internal spectrum it was."""
    return fourier.spectrum_to_internal(field, kind=simulation.grid.kind)


def energy_bins(simulation: Simulation) -> Callable[[np.ndarray], np.ndarray]:
    """Return the library's energy per frequency bin of an internal spectrum, in J."""
    grid = simulation.grid
    space = simulation.spatial_grid
    medium = simulation.medium
    if space is None or medium is None:
        return lambda spectrum: spectral_flux(grid, spectrum) * grid.d_omega
    # On a planar grid the medium is the slab modes' own fill, as the measure takes it.
    flux = SpatialFlux.resolve(grid, space, medium)
    return lambda spectrum: spatial_energy_spectrum(spectrum, flux)


def photon_weights(grid) -> np.ndarray:
    """Return ``1 / (hbar omega)`` per bin, zero at and below zero frequency."""
    weights = np.zeros(grid.n_omega)
    positive = grid.omega > 0.0
    weights[positive] = 1.0 / (HBAR * grid.omega[positive])
    return weights


def spread(
    bins: Callable[[np.ndarray], np.ndarray],
    weights: np.ndarray,
    spectrum: np.ndarray,
    scored: np.ndarray,
) -> float:
    """Return ``kappa``: ``sqrt(max_i w_i ||a||^2 / Q(a))`` over the scored bins.

    The measure is diagonal in the stored spectrum, so each column's element weights
    are its value on that column alone with every scored bin set to one.
    """
    largest = 0.0
    for column in range(spectrum.shape[0]):
        probe = np.zeros_like(spectrum)
        probe[column, :, scored] = 1.0
        per_element = weights * bins(probe) / spectrum.shape[1]
        largest = max(largest, float(np.max(per_element[scored])))
    total = float(np.sum(weights * bins(spectrum)))
    return math.sqrt(largest * float(np.sum(np.abs(spectrum) ** 2)) / total)


def steps_taken(output) -> int:
    """Return the run's accepted steps."""
    return int(np.count_nonzero(output["step_history/accepted"]))


def rounding(simulation: Simulation, steps: int, phi: float) -> float:
    """Return the march's rounding over `steps`, relative to the state, unmargined."""
    return march_bound(
        simulation.translate(),
        simulation._run_request(),
        steps,
        phi=phi,
        margin=1.0,
    )


def launch_phase(simulation: Simulation) -> float:
    """Return the launch's nonlinear phase, floored as simulation_bound floors it."""
    phi, _ = measured_scales(simulation)
    return max(phi, 0.1)


# ---------------------------------------------------------------------- time reversal


def reversal_bound(simulation: Simulation, forward: int, back: int) -> float:
    """Return the round trip's relative bound: two legs' errors and their rounding."""
    phi = launch_phase(simulation)
    growth = math.exp(phi)
    tolerance = growth * (growth * forward + back) * REVERSAL_RTOL
    return MARGIN * (tolerance + growth * rounding(simulation, forward + back, phi))


REVERSED = ("gnlse", "capillary", "radial", "planar")
"""The families the reversal runs on: one of each grid kind and transverse measure.

Cartesian and line free space reverse alike (``4e-16`` against ``6e-7``, their Raman
runs ``1.8e-4`` and ``2.3e-4``), and are left to the conservation table for cost."""


def round_trip(
    family: Callable[[bool], Simulation], backend: str, *, raman: bool = False
) -> tuple[float, float]:
    """Return a conjugated round trip's departure and its bound.

    The departure is ``||a_back - conj(a(0))|| / ||a(0)||``, normwise over the state.
    """
    simulation = controlled(family(raman), backend, REVERSAL_RTOL)
    there = simulation.run()
    start = internal(simulation, there["field"][0])
    end = internal(simulation, there["field"][-1])
    back = simulation.with_input(np.conj(end), noise=None).run()
    returned = internal(simulation, back["field"][-1])
    bound = reversal_bound(simulation, steps_taken(there), steps_taken(back))
    departure = float(np.linalg.norm(returned - np.conj(start)) / np.linalg.norm(start))
    return departure, bound


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("family", REVERSED)
def test_a_conjugated_launch_retraces_the_run(family, backend):
    """Launching ``conj(a(L))`` into the same problem returns ``conj(a(0))``.

    The reversal of an instantaneous response on an envelope grid (the GNLSE, with
    second- and third-order dispersion and the shock term), a modal field grid, a
    spatial field grid and the planar envelope. The Raman run of the same family
    must fail the same bound, because the causal response has no time-reversed twin.
    Measured, alike on both backends: the Kerr round trips depart by ``4e-14``
    (capillary) to ``2.9e-11`` (GNLSE), at most ``7e-6`` of their bounds of ``6e-7``
    to ``4e-6``; the Raman ones by ``7.6e-5`` (capillary, 136 times its bound) to
    ``2.0e-2`` (GNLSE, 5200 times).
    """
    departure, bound = round_trip(FAMILIES[family], backend)
    assert departure < bound, f"{family}: departure {departure:.3e} > {bound:.3e}"
    raman, raman_bound = round_trip(FAMILIES[family], backend, raman=True)
    assert raman > raman_bound, f"{family}: the Raman run reversed ({raman:.3e})"


@pytest.mark.parametrize("backend", BACKENDS)
def test_negating_beta_and_gamma_together_reverses_the_gnlse(backend):
    """The GNLSE with ``-beta`` and ``-gamma`` takes ``a(L)`` back to ``a(0)``.

    ``z -> -z`` without the conjugation. Negating ``beta`` alone leaves the Kerr
    term's sign unreversed and must fail the bound. Measured on both backends: the
    reversal departs by ``2.9e-11`` and the defect by ``0.75``, against ``3.9e-6``
    (negating ``gamma`` alone departs by ``5.3e-2``).
    """
    forward = controlled(gnlse(False), backend, REVERSAL_RTOL)
    there = forward.run()
    start = internal(forward, there["field"][0])
    end = internal(forward, there["field"][-1])
    for sign, beta_sign, reverses in ((-1.0, 1.0, True), (1.0, -1.0, False)):
        reverse = controlled(
            gnlse(False, sign=sign, beta_sign=beta_sign), backend, REVERSAL_RTOL
        )
        back = reverse.with_input(end, noise=None).run()
        returned = internal(reverse, back["field"][-1])
        departure = float(np.linalg.norm(returned - start) / np.linalg.norm(start))
        bound = reversal_bound(forward, steps_taken(there), steps_taken(back))
        assert (departure < bound) == reverses, (sign, beta_sign, departure, bound)


# -------------------------------------------------------------------- photon number


def conservation(
    simulation: Simulation, backend: str, *, shock: bool = True
) -> dict[str, float]:
    """Run `simulation` and return its photon and energy drifts with their bounds.

    Each bound is ``MARGIN 2 kappa (S rtol + march rounding)``, with ``kappa`` the
    measure's own weight spread at the launch (the module's docstring).
    """
    simulation = controlled(simulation, backend, CONSERVATION_RTOL, shock=shock)
    output = simulation.run()
    grid = simulation.grid
    bins = energy_bins(simulation)
    first = internal(simulation, output["field"][0])
    last = internal(simulation, output["field"][-1])
    steps = steps_taken(output)
    scored = np.asarray(simulation.translate().grids.norm_mask)
    march = rounding(simulation, steps, launch_phase(simulation))
    result = {"steps": float(steps)}
    for name, weights in (
        ("photons", photon_weights(grid)),
        ("energy", np.ones(grid.n_omega)),
    ):
        before = float(np.sum(weights * bins(first)))
        after = float(np.sum(weights * bins(last)))
        kappa = spread(bins, weights, first, scored)
        result[name] = after / before - 1.0
        result[f"{name} bound"] = (
            MARGIN * 2.0 * kappa * (steps * CONSERVATION_RTOL + march)
        )
    # The run's own energy statistic, formed by the backend, is the same measure: a
    # sum of n_columns n_omega positive terms either way, in another order and with
    # the admittance computed apart, which Higham's gamma of the term count bounds
    # with the margin. Measured at most 7e-16 relative, under 1e-3 of the bound.
    recorded = np.sum(np.asarray(output["stats/energy"]).reshape(2, -1), axis=-1)
    for plane, spectrum in ((0, first), (1, last)):
        assert_within(
            float(np.sum(bins(spectrum))),
            float(recorded[plane]),
            MARGIN * higham_gamma(first.size),
            what="the energy statistic",
        )
    return result


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("raman", [False, True], ids=["kerr", "raman"])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_photon_number_is_conserved_and_raman_alone_costs_energy(
    family, raman, backend
):
    """The photon number holds to the step bound on every family, Raman on or off.

    The energy holds with it for the instantaneous response, and a Raman response
    costs energy far outside the same bound, so that the photon number holding is a
    statement about the Raman term and not about a run in which nothing moved.
    Measured, alike on both backends: the photon drifts are at most ``3e-13``, under
    ``3e-5`` of their bounds of ``8e-9`` to ``9e-8``; the Raman energy costs are
    ``1.1e-6`` (capillary, 160 times its bound) to ``4.5e-5`` (GNLSE, 1400 times).
    """
    drifts = conservation(FAMILIES[family](raman), backend)
    assert abs(drifts["photons"]) < drifts["photons bound"], drifts
    if raman:
        assert drifts["energy"] < -drifts["energy bound"], drifts
    else:
        assert abs(drifts["energy"]) < drifts["energy bound"], drifts


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("family", ["gnlse", "capillary"])
def test_freezing_the_shock_term_breaks_photon_number_conservation(family, backend):
    """Without the prefactor's ``omega`` the photon number moves outside the bound.

    With the shock term frozen at the carrier the equation conserves the energy
    instead, so a Raman run's photon number drifts by about what its energy drifted
    by with the term in place. Measured on both backends: the GNLSE by ``4.6e-5``
    (540 times the bound) and the capillary by ``1.1e-6`` (79 times).
    """
    drifts = conservation(FAMILIES[family](True), backend, shock=False)
    assert abs(drifts["photons"]) > drifts["photons bound"], drifts
    assert abs(drifts["energy"]) < drifts["energy bound"], drifts


@pytest.mark.parametrize("family", list(LOSSY))
def test_a_small_loss_breaks_photon_number_conservation(family):
    """A loss of a few parts in a million is seen on every spatial measure.

    The spatial routes take no frozen shock term (their normalisation is the
    material's own), so their non-vacuity is a loss: a constant complex index of
    imaginary part ``1e-9`` on the free-space grids, whose photon loss is
    ``2 omega Im(n) L / c = 3.14e-6`` at the carrier, and a hundredth of the wall
    loss on the slab modes. Measured, alike on both backends: ``-3.14e-6`` (radial,
    65 times its bound; Cartesian and line, 380 times) and ``-6.0e-7`` (planar, 47
    times). The sensitivity is the measure's and the bound's, so ``native`` alone.
    """
    drifts = conservation(LOSSY[family](), "native")
    assert drifts["photons"] < -drifts["photons bound"], drifts


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_photon_measure_agrees_with_the_shot_noise_definition(backend):
    """Three readings of one GNLSE run's photon number agree to rounding.

    The flux-per-bin measure above against :func:`solprop.noise.photons_per_bin`, the
    definition the shot-noise floor is drawn from, absolutely; and against the plain
    ``sum |A|^2 / omega`` of ``test_numpy_ref_raman.py`` as ratios to the launch.
    """
    simulation = controlled(gnlse(True), backend, CONSERVATION_RTOL)
    output = simulation.run()
    grid = simulation.grid
    bins = energy_bins(simulation)
    weights = photon_weights(grid)
    positive = grid.omega > 0.0
    flux, plain = [], []
    # Each reading is a sum of n_omega positive terms of a few products and quotients
    # each; two such sums, or a ratio of two, differ by Higham's gamma of the bin count
    # twice. Measured: photons_per_bin exactly (it divides the same spectral density),
    # the plain sum to 4e-16 of the bound's 4.6e-12.
    bound = MARGIN * 2 * higham_gamma(grid.n_omega + 4)
    for plane in (0, -1):
        spectrum = internal(simulation, output["field"][plane])
        flux.append(float(np.sum(weights * bins(spectrum))))
        assert_within(
            float(np.sum(photons_per_bin(grid, spectrum))),
            flux[-1],
            bound,
            what="photons_per_bin",
        )
        plain.append(
            float(np.sum(np.abs(spectrum[..., positive]) ** 2 / grid.omega[positive]))
        )
    assert_within(plain[1] / plain[0], flux[1] / flux[0], bound, what="sum |A|^2/omega")
