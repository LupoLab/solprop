# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Map backend channel reductions onto the mixture's ordered species axis."""

from collections.abc import Iterable

import numpy as np
from jaxtyping import Float

from solprop.exceptions import BackendError
from solprop.stats import ON_AXIS_PLASMA_STATISTICS, species_statistics
from solprop.translate.problem import PropagationProblem

__all__ = [
    "axis_node",
    "axis_refusal",
    "axis_row_matrix",
    "axis_statistics",
    "population_statistics",
]


def population_statistics(
    problem: PropagationProblem,
    z: float,
    channel_fractions: Float[np.ndarray, "n_channels n_nodes"],
    channel_peaks: Float[np.ndarray, "n_peak_channels n_peak_nodes"],
) -> dict[str, np.ndarray]:
    """Reduce channel data without removing non-ionising species from the result.

    Parameters
    ----------
    problem : PropagationProblem
        Species labels, density tables, potentials and transverse measure.
    z : float
        Plane position in m.
    channel_fractions : numpy.ndarray
        Final ionised fractions, shape (n_channels, n_nodes).
    channel_peaks : numpy.ndarray
        Largest rate in s^-1 at each channel and node; cycle-averaged on an
        envelope grid.

    Returns
    -------
    dict
        Species statistics and the two total plasma statistics.

    Raises
    ------
    BackendError
        If the channel extents disagree or a resolved modal transform has no weights.
    """
    responses, transform = problem.responses, problem.transform
    expected = (len(responses.population_channels), transform.n_nodes)
    if channel_fractions.shape != expected or channel_peaks.shape != expected:
        raise BackendError(
            f"population statistics require channel arrays of shape {expected}"
        )
    densities = np.array([float(table(z)) for table in responses.species_densities])
    fractions = np.zeros((responses.n_species, transform.n_nodes))
    peaks = np.zeros(responses.n_species)
    potentials = np.zeros(responses.n_species)
    for channel, fraction, peak in zip(
        responses.population_channels, channel_fractions, channel_peaks, strict=True
    ):
        index = responses.labels.index(channel.label)
        fractions[index] = fraction
        peaks[index] = np.max(peak)
        potentials[index] = channel.ionisation_potential
    if transform.geometry is not None:
        weights = transform.geometry.spatial_weights
    elif transform.mode_averaged:
        weights = np.array([float(transform.aeff(z))])
    elif transform.node_weights is not None:
        taper = (
            1.0 if transform.taper_scale is None else float(transform.taper_scale(z))
        )
        weights = transform.node_weights / taper**2
    else:
        raise BackendError(
            "species statistics require physical transverse node weights"
        )
    result = species_statistics(densities, fractions, peaks, potentials, weights)
    if (
        responses.envelope_plasma is not None
        and len(responses.population_channels) == 1
    ):
        # Preserve the single-population statistic's node selection on both grids.
        densest = int(np.argmax(channel_fractions[0]))
        result["peak_ionisation_rate"] = np.asarray(channel_peaks[0, densest])
    return result


def axis_statistics(
    problem: PropagationProblem,
    z: float,
    channel_fractions: Float[np.ndarray, " n_channels"],
    channel_peaks: Float[np.ndarray, " n_peak_channels"],
) -> dict[str, np.ndarray]:
    """Reduce the axis row's channel data to the four on-axis statistics (SEP 19).

    Parameters
    ----------
    problem : PropagationProblem
        Species labels, density tables and channels.
    z : float
        Plane position in m.
    channel_fractions : numpy.ndarray
        Final ionised fraction on the axis per channel, shape (n_channels,).
    channel_peaks : numpy.ndarray
        Largest rate in s^-1 on the axis per channel; cycle-averaged on an envelope
        grid.

    Returns
    -------
    dict
        ``on_axis_electron_density`` and ``on_axis_peak_ionisation_rate``, and their
        species forms in the mixture's order, non-ionising species at zero and the
        total the sum of the species densities.

    Raises
    ------
    BackendError
        If the channel extents disagree or a value is outside its domain.
    """
    responses = problem.responses
    expected = (len(responses.population_channels),)
    if channel_fractions.shape != expected or channel_peaks.shape != expected:
        raise BackendError(
            f"on-axis statistics require channel arrays of shape {expected}"
        )
    if (
        not np.all(np.isfinite(channel_fractions))
        or not np.all(np.isfinite(channel_peaks))
        or np.any(channel_fractions < 0)
        or np.any(channel_fractions > 1)
        or np.any(channel_peaks < 0)
    ):
        raise BackendError(
            "on-axis statistics require finite rates and fractions in [0, 1]"
        )
    densities = np.array([float(table(z)) for table in responses.species_densities])
    electrons = np.zeros(responses.n_species)
    peaks = np.zeros(responses.n_species)
    for channel, fraction, peak in zip(
        responses.population_channels, channel_fractions, channel_peaks, strict=True
    ):
        index = responses.labels.index(channel.label)
        electrons[index] = densities[index] * fraction
        peaks[index] = peak
    total = 0.0
    for value in electrons:
        total += value
    return {
        "on_axis_electron_density": np.asarray(total),
        "on_axis_peak_ionisation_rate": np.asarray(np.max(peaks)),
        "on_axis_species_electron_density": electrons,
        "on_axis_species_peak_ionisation_rate": peaks,
    }


def axis_refusal(problem: PropagationProblem, names: Iterable[str]) -> str | None:
    """Return why this problem cannot give the requested on-axis statistics.

    Parameters
    ----------
    problem : PropagationProblem
        The translated problem.
    names : iterable of str
        The requested statistic names.

    Returns
    -------
    str or None
        The refusal, naming the statistics and the route, or ``None``.
    """
    asked = sorted(set(names).intersection(ON_AXIS_PLASMA_STATISTICS))
    if not asked:
        return None
    transform = problem.transform
    if transform.kind != "modal" or transform.modal_tables is not None:
        route = (
            "a general modal guide"
            if transform.modal_tables is not None
            else "the mode-averaged route"
            if transform.kind == "mode_averaged"
            else f"the {transform.kind} spatial route"
        )
        return (
            f"{asked} are defined on resolved modal routes, where the axis field is "
            f"synthesised from the mode profiles; {route} is not qualified for them. "
            "Use electron_density and the species statistics"
        )
    if not problem.responses.population_channels:
        return f"{asked} need a compiled plasma response, and this problem has none"
    return None


def axis_row_matrix(
    problem: PropagationProblem, axis: Float[np.ndarray, "2 n_modes"]
) -> Float[np.ndarray, "n_rows n_modes"]:
    """Return the matrix that synthesises the axis row from the modal state.

    With both Cartesian components of the mode profiles, as for ``peak_intensity``;
    when the state carries the components itself, the profile's magnitude on the axis.

    Parameters
    ----------
    problem : PropagationProblem
        The translated problem.
    axis : numpy.ndarray
        The mode set's synthesis on the axis at this position, taper included, rows x
        and y (``HostDriver._on_axis_synthesis``).

    Returns
    -------
    numpy.ndarray
        The rows, carrying the pad scale: x and y on a scalar state, or the one
        magnitude row on a state with two polarisations.
    """
    scaled = axis * problem.grids.pad_scale
    if problem.shape[1] == 1:
        return scaled
    return np.hypot(scaled[0], scaled[1])[None]


def axis_node(
    problem: PropagationProblem, axis: Float[np.ndarray, "2 n_modes"]
) -> int | None:
    """Return the quadrature node on the axis, or ``None`` when there is none.

    A Gauss-Radau field-scale rule (SEP 19) has one: its synthesis row is the mode
    profile at ``r = 0``, the on-axis synthesis itself, so its own node scan supplies
    the axis statistics without a further one. A transform that does not carry every
    component the modes have on the axis has none: its node scan lacks that row.

    Parameters
    ----------
    problem : PropagationProblem
        The translated problem.
    axis : numpy.ndarray
        The on-axis synthesis at the reference position, rows x and y.

    Returns
    -------
    int or None
        The node index.
    """
    transform = problem.transform
    if transform.kind != "modal" or transform.modal_tables is not None:
        return None
    n_pol = transform.n_pol
    if np.any(axis[n_pol:]):
        return None
    rows = transform.synthesis.reshape(transform.n_nodes, n_pol, -1)
    scaled = axis[:n_pol] * problem.grids.pad_scale
    for index, row in enumerate(rows):
        if np.allclose(row, scaled, rtol=1e-14, atol=0.0):
            return index
    return None
