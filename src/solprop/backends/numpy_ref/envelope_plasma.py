# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Evaluate compiled envelope plasma as separate current-source streams."""

import numpy as np
from jaxtyping import Complex, Float

from solprop.translate.envelope_plasma import (
    CycleIntegralData,
    EnvelopePlasmaAssembly,
    cycle_moments,
)
from solprop.translate.stage import StageScalars

__all__ = ["envelope_populations", "envelope_sources"]


def envelope_sources(
    field_t: Complex[np.ndarray, "n_nodes n_pol n_to"],
    assembly: EnvelopePlasmaAssembly,
    scalars: StageScalars,
    dt: float,
    window: Float[np.ndarray, " n_to"],
) -> tuple[
    Complex[np.ndarray, "n_nodes n_pol n_to"],
    Complex[np.ndarray, "n_nodes n_pol n_to"],
]:
    """Form two windowed sources from independently recomputed channel populations.

    Parameters
    ----------
    field_t : numpy.ndarray
        Physical complex V/m envelope, shape ``(n_nodes, n_pol, n_to)``.
    assembly : EnvelopePlasmaAssembly
        Labelled channels and validated cycle data.
    scalars : StageScalars
        Density and pre-ionisation at this RHS position in channel order.
    dt : float
        Fine time spacing in s.
    window : numpy.ndarray
        Material-source collar, shape ``(n_to,)``.

    Returns
    -------
    tuple of numpy.ndarray
        Drude drive in A m^-2 s^-1 and loss current in A m^-2, each with the
        field's shape. Projection and temporal FFT follow this function.
    """
    drive = np.zeros_like(field_t)
    loss = np.zeros_like(field_t)
    for channel, density, initial in zip(
        assembly.channels,
        scalars.plasma_densities,
        scalars.plasma_preionisation,
        strict=True,
    ):
        for node, field in enumerate(field_t):
            rate, current = _cycle_quantities(field, channel.cycle, loss=True)
            fraction, survival = _population(rate, initial, dt)
            drive[node] += (channel.charge_mass_ratio * density * fraction) * field
            loss[node] += (channel.ionisation_potential * density * survival) * current
    return drive * window, loss * window


def _cycle_quantities(
    field: Complex[np.ndarray, "n_pol n_to"], cycle: CycleIntegralData, *, loss: bool
) -> tuple[Float[np.ndarray, " n_to"], Complex[np.ndarray, "n_pol n_to"]]:
    """Evaluate rate and optional directional loss by independent SVD axes.

    Parameters
    ----------
    field : numpy.ndarray
        Physical V/m envelope, shape (n_pol, n_to).
    cycle : CycleIntegralData
        Validated instantaneous rate and quadrature rules.
    loss : bool
        Whether to form the directional loss moments beside the averaged rate.

    Returns
    -------
    tuple of numpy.ndarray
        Rate in s^-1 and loss moment per unit ionisation work, with shapes
        (n_to,) and (n_pol, n_to).
    """
    rate = np.zeros(field.shape[1])
    current = np.zeros_like(field)
    for sample in range(field.shape[1]):
        a = field[:, sample]
        scale = float(np.max(np.abs(a)))
        if scale == 0:
            continue
        matrix = np.column_stack((np.real(a) / scale, np.imag(a) / scale))
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        major = float(singular[0]) * scale
        ratio = (
            0.0 if len(singular) == 1 else min(1.0, float(singular[1] / singular[0]))
        )
        moments = cycle_moments(major, ratio, cycle)
        rate[sample] = moments[0] + ratio**2 * moments[1]
        if not loss:
            continue
        current[:, sample] = (
            (2 * (moments[0] / major)) * left[:, 0] * (right[0, 0] + 1j * right[0, 1])
        )
        if len(singular) == 2:
            current[:, sample] += (
                (2 * (ratio * moments[1] / major))
                * left[:, 1]
                * (right[1, 0] + 1j * right[1, 1])
            )
    return rate, current


def _population(
    rate: Float[np.ndarray, " n_to"], initial: float, dt: float
) -> tuple[Float[np.ndarray, " n_to"], Float[np.ndarray, " n_to"]]:
    """Retain survival separately from a cancellation-free population scan.

    Parameters
    ----------
    rate : numpy.ndarray
        Cycle-averaged rates in s^-1, shape (n_to,).
    initial : float
        Initial ionised fraction in [0, 1].
    dt : float
        Fine time spacing in s.

    Returns
    -------
    tuple of numpy.ndarray
        Ionised fraction and surviving neutral fraction, each shape (n_to,).
    """
    with np.errstate(over="ignore"):
        increments = (0.5 * rate[:-1] + 0.5 * rate[1:]) * dt
        exposure = np.concatenate((np.zeros(1), np.cumsum(increments)))
    survival = (1 - initial) * np.exp(-exposure)
    fraction = initial + (1 - initial) * -np.expm1(-exposure)
    return fraction, survival


def envelope_populations(
    field_t: Complex[np.ndarray, "n_nodes n_pol n_to"],
    assembly: EnvelopePlasmaAssembly,
    initials: tuple[float, ...],
    dt: float,
) -> tuple[
    Float[np.ndarray, "n_channels n_nodes"], Float[np.ndarray, "n_channels n_nodes"]
]:
    """Recompute each node's final population and peak cycle-averaged rate.

    Parameters
    ----------
    field_t : numpy.ndarray
        Physical V/m envelopes, shape (n_nodes, n_pol, n_to).
    assembly : EnvelopePlasmaAssembly
        Compiled channels in population order.
    initials : tuple of float
        Initial fractions at this position in channel order.
    dt : float
        Fine time spacing in s.

    Returns
    -------
    tuple of numpy.ndarray
        Final fractions and peak cycle-averaged rates in s^-1, both shaped
        (n_channels, n_nodes). Every call scans the supplied field afresh.
    """
    fractions = np.empty((len(assembly.channels), len(field_t)))
    peaks = np.empty_like(fractions)
    for index, (channel, initial) in enumerate(
        zip(assembly.channels, initials, strict=True)
    ):
        for node, field in enumerate(field_t):
            rate, _ = _cycle_quantities(field, channel.cycle, loss=False)
            fraction, _ = _population(rate, initial, dt)
            fractions[index, node] = fraction[-1]
            peaks[index, node] = np.max(rate)
    return fractions, peaks
