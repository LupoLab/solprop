# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The floor-subtracted spectral density of an input-noise run (SEP 37, section 5.5).

The input draw has a fixed modulus, one photon per active bin and channel, so removing
its energy is exact for the one realisation. What remains of a noisy launch is the
pulse's own density plus the interference of the pulse with the seed, which is as large
as the seed and does not vanish for one draw; where the linear operator only changes the
phase, the remainder of a propagated run is the modified model's corrected readout
``|psi + N|^2 - |N|^2`` of the same draw. Every run here is a lossless GNLSE fibre with
Kerr and Raman, no absorber, stepped by the same fixed Runge-Kutta schedule, so the
two models' runs are the same discrete computation and agree to rounding.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from solprop.driver import Simulation
from solprop.exceptions import ProcessingError
from solprop.output.schema import Output
from solprop.plotting.propagation import prop_2d, spec_1d
from solprop.processing import get_spectral_density, get_spectrum
from solprop.processing.axes import _check_choice, grid_of
from solprop.processing.fields import NOISE_FLOORS, _fft_norm
from solprop.stepper import FixedSchedule, RungeKutta

LENGTH = 0.02
"""Fibre length, m: about 4.4 rad of nonlinear phase at the launch's peak power."""

STEPS = 40
"""Fixed steps of every run."""

SEED = 26
"""Seed of the draw both noisy runs share."""

READOUT_ROUNDING = 64 * STEPS * np.finfo(float).eps
"""Rounding budget of the input/modified equivalence, relative to the density's peak.

Dimensionless, fixed before the comparison: both runs evaluate the same stages on the
same arguments (the interaction picture carries the unitary draw as a constant), so
they differ only by the order in which the seed is added, a few roundings per stage of
the peak density, accumulated over :data:`STEPS` steps. Measured 1.2e-15.
"""

LAUNCH_ROUNDING = 64 * np.finfo(float).eps
"""Rounding budget of the unpropagated identity, relative to the density's peak."""


def _fibre(noise: str | None) -> Simulation:
    """Return a 50 fs sech soliton-order launch in a lossless Kerr and Raman fibre."""
    return Simulation.from_gnlse(
        0.11,
        LENGTH,
        [0.0, 0.0, -1.183e-26, 8.1e-41],
        lambda0=835e-9,
        wavelength_limits=(500e-9, 2e-6),
        trange=2e-12,
        width=50e-15,
        peak_power=2e3,
        pulseshape="sech",
        raman=True,
        noise=noise,  # pyright: ignore[reportArgumentType]
        seed=None if noise is None else SEED,
        schedule=FixedSchedule(np.full(STEPS, LENGTH / STEPS)),
        dz_max=LENGTH / STEPS,
        dz_init=LENGTH / STEPS,
        integrator=RungeKutta(),
        n_save=2,
        backend="numpy_ref",
    )


@pytest.fixture(scope="module")
def runs() -> dict[str, Output]:
    """The three runs every test here reads: noiseless, input and modified."""
    return {
        "none": _fibre(None).run(progress="none"),
        "input": _fibre("input").run(progress="none"),
        "modified": _fibre("modified").run(progress="none"),
    }


def _ascending(output: Output, values: np.ndarray) -> np.ndarray:
    """Return a per-bin array in the ascending order the accessors use."""
    return values[..., np.argsort(grid_of(output).omega)]


def test_subtracting_the_floor_of_a_launch_leaves_the_pulse_and_its_interference(runs):
    """Exactly one photon per bin is removed from the unpropagated noisy launch.

    The remainder is the pulse's own density plus ``2 Re(conj(psi) N)``, the one term
    a single realisation cannot remove; outside the pulse it is zero to rounding,
    where the saved density is the floor alone.
    """
    noisy, clean = runs["input"], runs["none"]
    grid = grid_of(noisy)
    _, subtracted = get_spectral_density(
        noisy, "omega", plane=0, band="all", noise_floor="subtract"
    )
    _, pulse = get_spectral_density(clean, "omega", plane=0, band="all")
    _, kept = get_spectral_density(noisy, "omega", plane=0, band="all")
    norm = _fft_norm(grid)
    launch = np.asarray(clean.fields("optics")[0])[0, 0] * norm
    seed = np.asarray(noisy.fields("optics")[0])[0, 0] * norm - launch
    interference = _ascending(noisy, 2.0 * np.real(np.conj(launch) * seed))
    peak = float(np.max(pulse))
    np.testing.assert_allclose(
        subtracted, pulse + interference, rtol=0.0, atol=LAUNCH_ROUNDING * peak
    )
    # The removed energy is exactly the floor: kept less subtracted is the seed's.
    floor = _ascending(noisy, abs(seed) ** 2)
    np.testing.assert_allclose(
        kept - subtracted, floor, rtol=0.0, atol=1e-3 * floor.max()
    )
    # Far from the pulse the launch is nothing but the seed, so nothing is left.
    empty = pulse < 1e-30 * peak
    assert np.count_nonzero(empty) > 0
    assert float(np.max(abs(subtracted[empty]))) <= 1e-9 * float(floor.max())


def test_the_input_run_minus_its_floor_is_the_modified_run_s_readout(runs):
    """Lossless, the subtracted input run is ``|psi + N|^2 - |N|^2`` of the same draw.

    The modified run saves ``psi`` without its draw; the draw at the plane is the
    stored realisation advanced by the saved ``noise_phase``, both in the internal
    convention, which is the stored field's in the FFT convention.
    """
    noisy, modified = runs["input"], runs["modified"]
    grid = grid_of(noisy)
    _, subtracted = get_spectral_density(
        noisy, "omega", plane=-1, band="all", noise_floor="subtract"
    )
    record = modified.metadata.noise
    phase = modified.noise_phase
    assert record is not None and phase is not None
    norm = _fft_norm(grid)
    psi = np.asarray(modified.fields("fft")[-1])
    draw = np.asarray(record.spectrum) * np.exp(1j * np.asarray(phase[-1]))
    readout = np.abs((psi + draw) * norm) ** 2 - np.abs(record.spectrum * norm) ** 2
    readout = _ascending(noisy, readout[0, 0])
    peak = float(np.max(abs(subtracted)))
    np.testing.assert_allclose(
        subtracted, readout, rtol=0.0, atol=READOUT_ROUNDING * peak
    )
    # The floor-subtracted density is signed: the interference leaves negative bins.
    assert np.any(subtracted < 0.0)


def test_the_floor_is_refused_without_a_seed_in_the_saved_field(runs):
    """A noiseless run has no floor; a modified run's saved field holds no seed."""
    with pytest.raises(ProcessingError, match="carries no noise"):
        get_spectral_density(runs["none"], noise_floor="subtract")
    with pytest.raises(ProcessingError, match="'modified' run's saved field holds"):
        get_spectral_density(runs["modified"], noise_floor="subtract")
    # An unknown name is refused by the same door in production, where no runtime
    # type checker stands in front of the keyword.
    with pytest.raises(ProcessingError, match="noise_floor must be one of"):
        _check_choice("remove", NOISE_FLOORS, "noise_floor")


def test_a_spectrum_refuses_the_floor_and_names_the_density(runs):
    """An amplitude has no part that is the floor's power; only the density has it."""
    omega, spectrum = get_spectrum(runs["input"], plane=-1, noise_floor="keep")
    assert spectrum.shape == omega.shape
    with pytest.raises(ProcessingError, match="ask get_spectral_density"):
        get_spectrum(runs["input"], plane=-1, noise_floor="subtract")


def test_the_plots_draw_the_subtracted_density_and_mask_its_gaps_on_a_log_axis(runs):
    """A logarithmic slice leaves the non-positive bins out; the map takes them too."""
    noisy = runs["input"]
    figure = spec_1d(
        noisy, LENGTH, "omega", spec_range=(1e15, 4e15), noise_floor="subtract"
    )
    try:
        (line,) = figure.axes[0].get_lines()
        drawn = np.asarray(line.get_ydata(), dtype=float)
        _, density = get_spectral_density(
            noisy, "omega", plane=-1, spec_range=(1e15, 4e15), noise_floor="subtract"
        )
        assert np.any(density <= 0.0)
        assert np.array_equal(np.isnan(drawn), density <= 0.0)
    finally:
        plt.close(figure)
    figures = prop_2d(noisy, "omega", spec_range=(1e15, 4e15), noise_floor="subtract")
    try:
        assert len(figures) == 1
    finally:
        plt.close("all")
