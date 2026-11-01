# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""First-order coherence across an ensemble of noise realisations.

The quantity that says whether a supercontinuum is reproducible from shot to shot: the
modulus of the normalised cross-correlation of the spectra of independent realisations,
at each frequency (J. M. Dudley and S. Coen, Optics Letters **27**, 1180 (2002),
doi:10.1364/OL.27.001180).

The definition here is the unbiased one over distinct pairs, which is one for identical
realisations and zero, in the mean, for independent ones. A sum over every ordered pair,
the diagonal ``i == j`` included, would carry the self-terms in its numerator and tend
to ``1/N`` rather than to zero for a fully incoherent ensemble.
"""

from collections.abc import Sequence

import numpy as np
from jaxtyping import Complex, Float

from solprop.exceptions import OutputError
from solprop.output.schema import Output
from solprop.processing.axes import Band, ModeSelection
from solprop.processing.fields import get_spectrum

__all__ = ["band_coherence", "g12", "g12_of", "spectral_bands"]


def g12(
    spectra: Complex[np.ndarray, "n_realisations *rest n_x"],
) -> Float[np.ndarray, "*rest n_x"]:
    r"""Return the first-order coherence of an ensemble of complex spectra.

    .. math::

        g_{12}(\omega) = \frac{\left| \sum_{i \neq j}
            \tilde E_i^{*}(\omega) \tilde E_j(\omega) \right|}
            {(N - 1) \sum_i \left| \tilde E_i(\omega) \right|^2}

    evaluated with the identity
    :math:`\sum_{i \neq j} \tilde E_i^{*} \tilde E_j =
    \left| \sum_i \tilde E_i \right|^2 - \sum_i \left| \tilde E_i \right|^2`,
    which costs one pass over the ensemble rather than :math:`N^2` slices.
    Identical realisations give exactly one; independent ones give zero in the mean,
    because the self-terms the identity removes are the only ones that survive the
    average.

    Parameters
    ----------
    spectra : numpy.ndarray
        Complex spectra of the ensemble, shape ``(n_realisations, ..., n_x)``, with
        at least two realisations along the leading axis.

    Returns
    -------
    numpy.ndarray
        The coherence, shape ``(...)``, between 0 and 1. A frequency at which every
        realisation is empty gives ``nan``.

    Raises
    ------
    solprop.exceptions.OutputError
        If fewer than two realisations are given.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing.coherence import g12
    >>> one = np.array([1.0 + 0.0j, 2.0 + 0.0j])
    >>> g12(np.stack([one, one])).round(12).tolist()
    [1.0, 1.0]
    """
    if spectra.shape[0] < 2:
        msg = (
            "the first-order coherence needs at least two realisations, got "
            f"{spectra.shape[0]}"
        )
        raise OutputError(msg)
    n_realisations = spectra.shape[0]
    total = np.abs(np.sum(spectra, axis=0)) ** 2
    individual = np.sum(np.abs(spectra) ** 2, axis=0)
    # The difference is a sum of conjugate pairs and so real and non-negative in exact
    # arithmetic; cancellation can take it a few ulp below zero when the ensemble is
    # incoherent, which the modulus of the definition removes.
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(total - individual) / ((n_realisations - 1) * individual)


def g12_of(
    outputs: Sequence[Output],
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    band: Band = "active",
) -> tuple[Float[np.ndarray, " n_x"], Float[np.ndarray, "*shape n_x"]]:
    """Return the first-order coherence of an ensemble of saved runs.

    Every run must have been computed on the same grid and saved at the same planes; the
    spectra are read through :func:`solprop.processing.fields.get_spectrum`, so the
    ensemble may mix memory and HDF5 handlers.

    Parameters
    ----------
    outputs : sequence of solprop.output.schema.Output
        The ensemble, at least two opened handlers.
    plane : int or None
        Index of the plane wanted, negative from the end; with neither ``plane`` nor
        ``z``, every plane is used and the result carries a leading ``n_z`` axis.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report; ``'sum'`` is refused, as it is for any complex amplitude.
    band : {'active', 'all'}
        Which part of the stored axis to use.

    Returns
    -------
    omega : numpy.ndarray
        Angular frequency, shape ``(n_x,)``, rad/s, ascending.
    coherence : numpy.ndarray
        The coherence, between 0 and 1, and ``nan`` at a frequency where every run of
        the ensemble is empty.

    Raises
    ------
    solprop.exceptions.OutputError
        If fewer than two runs are given, or their spectra do not have the same shape.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.coherence import g12_of
    >>> ensemble = [example_output(), example_output()]
    >>> omega, coherence = g12_of(ensemble, plane=-1)
    >>> bool(np.all(coherence[np.isfinite(coherence)] == 1.0))
    True
    """
    if len(outputs) < 2:
        msg = f"the first-order coherence needs at least two runs, got {len(outputs)}"
        raise OutputError(msg)
    gathered = [
        get_spectrum(output, plane=plane, z=z, modes=modes, band=band)
        for output in outputs
    ]
    shapes = {values.shape for _, values in gathered}
    if len(shapes) > 1:
        msg = f"every run of an ensemble must have the same shape, got {sorted(shapes)}"
        raise OutputError(msg)
    return gathered[0][0], g12(np.stack([values for _, values in gathered]))


def spectral_bands(
    weights: Float[np.ndarray, " n_x"], n_bands: int = 32
) -> list[np.ndarray]:
    """Return the bin indices of equal bands across a spectrum's occupied span.

    SEP 33's spectral metric: the span from the first to the last bin with non-zero
    weight, cut into `n_bands` bands of as nearly equal width as the bins allow.

    Parameters
    ----------
    weights : numpy.ndarray
        Non-negative spectral weight per bin, shape ``(n_x,)``, zero outside the
        band of interest (a mean spectrum restricted to the active band).
    n_bands : int, optional
        How many bands, 32 by default.

    Returns
    -------
    list of numpy.ndarray
        The bins of each band, in order; empty when every weight is zero.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing.coherence import spectral_bands
    >>> [band.tolist() for band in spectral_bands(np.array([0.0, 1, 1, 1, 1, 0]), 2)]
    [[1, 2], [3, 4]]
    """
    occupied = np.flatnonzero(np.asarray(weights) > 0)
    if occupied.size == 0:
        return []
    span = np.arange(occupied[0], occupied[-1] + 1)
    return list(np.array_split(span, n_bands))


def band_coherence(
    spectra: Complex[np.ndarray, "n_realisations *rest n_x"],
    reference: Complex[np.ndarray, "n_reference *rest n_x"],
    *,
    n_bands: int = 32,
    db_min: float = -30.0,
) -> tuple[Float[np.ndarray, " n_bands"], Float[np.ndarray, " n_bands"]]:
    r"""Return an ensemble's coherence averaged over the bands of a reference spectrum.

    The coherence :func:`g12` of `spectra` per bin, averaged over each band's bins
    weighted by the `reference` ensemble's mean spectrum,
    :math:`\sum_b \bar S_b g_{12,b} / \sum_b \bar S_b`, in the `n_bands` equal bands of
    :func:`spectral_bands` across the reference mean spectrum's occupied span. A band
    whose mean spectral energy is below `db_min` of the largest band's is not
    scored. The reference selects the bands and weighs them, so two ensembles judged
    against one reference (for example two precisions or two solvers with the same
    seeds) are averaged over the same bins with the same weights. The reference's own
    band coherence is ``band_coherence(reference, reference)``.

    Parameters
    ----------
    spectra : numpy.ndarray
        Complex spectra of the ensemble judged, shape ``(n_realisations, ..., n_x)``,
        at least two realisations.
    reference : numpy.ndarray
        Complex spectra of the reference ensemble, shape ``(n_reference, ..., n_x)``
        with the same trailing shape. Its mean of :math:`|\tilde E|^2` over the
        realisations, summed over the axes between, is the weight.
    n_bands : int, optional
        How many bands, 32 by default.
    db_min : float, optional
        Bands below this level of the largest band's energy are not scored, dB;
        -30 by default.

    Returns
    -------
    coherence : numpy.ndarray
        The weighted mean coherence of each band, shape ``(n_bands,)``; ``nan`` for a
        band not scored.
    energy : numpy.ndarray
        The reference mean spectrum's energy in each band, shape ``(n_bands,)``.

    Raises
    ------
    solprop.exceptions.OutputError
        If the trailing shapes differ, or `spectra` has fewer than two realisations.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing.coherence import band_coherence
    >>> one = np.ones(64, dtype=complex)
    >>> coherence, energy = band_coherence(np.stack([one, one]), np.stack([one, one]))
    >>> bool(np.allclose(coherence, 1.0)), energy.shape
    (True, (32,))
    """
    if (
        spectra.shape[1:] != reference.shape[1:]
    ):  # pragma: no cover - refused by the annotation where it is checked
        msg = (
            f"the ensemble's spectra have shape {spectra.shape[1:]}, the reference's "
            f"{reference.shape[1:]}"
        )
        raise OutputError(msg)
    coherence = g12(spectra)
    weight = np.mean(np.abs(reference) ** 2, axis=0)
    weight = weight.reshape(-1, weight.shape[-1])
    coherence = coherence.reshape(-1, coherence.shape[-1])
    bands = spectral_bands(weight.sum(axis=0), n_bands)
    result = np.full(n_bands, np.nan)
    if not bands:
        return result, np.zeros(n_bands)
    energy = np.array([weight[:, band].sum() for band in bands])
    kept = energy >= energy.max() * 10.0 ** (db_min / 10.0)
    for index, band in enumerate(bands):
        if not kept[index]:
            continue
        weights = weight[:, band]
        values = coherence[:, band]
        # An empty bin has no coherence (nan) and no weight: it adds nothing.
        finite = np.isfinite(values) & (weights > 0)
        result[index] = float(
            np.sum(weights[finite] * values[finite]) / np.sum(weights[finite])
        )
    return result, energy
