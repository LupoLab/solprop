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

__all__ = ["g12", "g12_of"]


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
