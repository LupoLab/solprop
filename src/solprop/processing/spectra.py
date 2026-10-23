# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Spectral post-processing: instrument resolution and explicit bandpass windows.

Two operations that belong to the spectrum rather than to the accessor that fetched it:
smoothing a spectral energy density to the resolution of a real instrument, and building
the Planck-tapered window of a wavelength bandpass.

Neither is ever applied on its own. A window is a keyword, it is a wavelength range, and
the array it produces is returned to the caller to multiply; no window adjusts its own
limits as a side channel.
"""

import numpy as np
from jaxtyping import Float

from solprop.constants import C
from solprop.exceptions import ProcessingError
from solprop.grid import TimeGrid, planck_taper

__all__ = [
    "DEFAULT_BANDPASS_BINS",
    "DEFAULT_BANDPASS_FRACTION",
    "Bandpass",
    "bandpass",
    "specres",
]

Bandpass = tuple[float, float] | Float[np.ndarray, " n_omega"]
"""A wavelength range to taper between, or a ready-made window on the stored axis."""

DEFAULT_BANDPASS_BINS = 64
"""Widest default bandpass edge, in frequency bins (Luna's ``winwidth = :auto``)."""

DEFAULT_BANDPASS_FRACTION = 0.25
"""Widest default bandpass edge as a fraction of the pass band's angular width.

Each Planck edge is antisymmetric about its midpoint, so it transmits half its width:
a window with edges ``f`` times the band's width has an integrated width ``1 + f``
times the band's. A quarter keeps that excess to 25 % while each edge still spans a
quarter of the band, so the window stays smooth on the band's own scale; and it leaves
every band wider than ``64 / 0.25 = 256`` bins on the 64-bin edge it always had.
"""

# Widening the kernel until the tail falls below 1e-8 of the peak converges, for a
# Gaussian of this width, to three resolution elements; the truncated tail is
# then at the 1e-11 level and the discrete kernel still sums to one to round-off.
_KERNEL_SPAN = 3.0

# A Gaussian's full width at half maximum in units of its standard deviation.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def bandpass(
    grid: TimeGrid,
    wavelength_limits: tuple[float, float],
    *,
    width: float | None = None,
) -> Float[np.ndarray, " n_omega"]:
    """Return a Planck-tapered bandpass window on the stored frequency axis.

    The window is one inside the wavelength range and zero
    outside it, joined on each side by a Planck taper of the given width in angular
    frequency. The default edge is the narrower of :data:`DEFAULT_BANDPASS_BINS` (64)
    bin spacings and :data:`DEFAULT_BANDPASS_FRACTION` (a quarter) of the pass band's
    angular width, so a narrow band on a short time window is not swamped by edges
    wider than itself. It is returned rather than applied, because nothing in
    :mod:`solprop.processing` windows a spectrum unless it was asked to.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid whose ``omega`` axis the window is built on.
    wavelength_limits : tuple of float
        The pass band, m, in either order.
    width : float or None
        Width of each taper edge, rad/s; ``None`` gives the smaller of 64 bin spacings
        and a quarter of the pass band's angular width.

    Returns
    -------
    numpy.ndarray
        Window values in ``[0, 1]``, shape ``(n_omega,)``, in storage order.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If a limit is not positive, or the width is not positive -- including the
        default width of an empty pass band (equal limits).

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_grid
    >>> from solprop.processing.spectra import bandpass
    >>> grid = example_grid()
    >>> window = bandpass(grid, (700e-9, 900e-9))
    >>> float(window.max()), float(window.min())
    (1.0, 0.0)
    """
    if min(wavelength_limits) <= 0.0:
        msg = f"wavelength limits must be positive, got {wavelength_limits!r}"
        raise ProcessingError(msg)
    edges = sorted(2.0 * np.pi * C / limit for limit in wavelength_limits)
    if width is None:
        width = min(
            DEFAULT_BANDPASS_BINS * grid.d_omega,
            DEFAULT_BANDPASS_FRACTION * (edges[1] - edges[0]),
        )
    if width <= 0.0:
        msg = f"the taper width must be positive, got {width!r}"
        raise ProcessingError(msg)
    return planck_taper(
        grid.omega, edges[0] - width, edges[0], edges[1], edges[1] + width
    )


def specres(
    x: Float[np.ndarray, " n_x"],
    density: Float[np.ndarray, "*rest n_x"],
    resolution: float,
    *,
    weights: Float[np.ndarray, " n_x"] | None = None,
    spec_range: tuple[float, float] | None = None,
    samples: int = 10,
) -> tuple[Float[np.ndarray, " n_out"], Float[np.ndarray, "*rest n_out"]]:
    """Smooth a spectral density to an instrument resolution on a uniform axis.

    The output axis is uniform with ``samples`` points
    per resolution element, and each input bin is spread over it by a unit-area Gaussian
    whose **full width at half maximum** is ``resolution``, weighted by the width of the
    bin it came from:

    ``S_out(x_j) = sum_k S(x_k) w_k G(x_j - x_k)``.

    Because the kernel integrates to one, the integral of the density -- the energy --
    is preserved to round-off wherever the signal is well inside the range. The input
    axis need not be uniform, which is what makes this work directly on a wavelength
    axis derived from a uniform frequency grid: pass the bin widths as ``weights``.

    Non-positive output values are not replaced with the smallest positive one to keep
    a logarithmic plot away from zero: that would change the data.

    Parameters
    ----------
    x : numpy.ndarray
        Input abscissa, shape ``(n_x,)``, strictly increasing.
    density : numpy.ndarray
        Spectral density on that abscissa, shape ``(..., n_x)``, per unit of ``x``.
    resolution : float
        Instrument resolution as a full width at half maximum, in the units of ``x``.
    weights : numpy.ndarray or None
        Integration measure of each input bin, in the units of ``x``; ``None`` uses the
        central differences of ``x``.
    spec_range : tuple of float or None
        Limits of the output axis, clipped to the range of ``x``; ``None`` uses all
        of it.
    samples : int
        Output points per resolution element, at least one.

    Returns
    -------
    x_out : numpy.ndarray
        Uniform output abscissa, shape ``(n_out,)``.
    density_out : numpy.ndarray
        The smoothed density, shape ``(..., n_out)``.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the resolution or the sampling is not positive, if ``x`` does not increase,
        or if the requested range does not overlap the data.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing.spectra import specres
    >>> x = np.linspace(400e-9, 1200e-9, 2001)
    >>> density = np.exp(-0.5 * ((x - 800e-9) / 20e-9) ** 2)
    >>> smooth_x, smooth = specres(x, density, 50e-9)
    >>> bool(smooth.max() < density.max())
    True
    """
    x = np.asarray(x, dtype=float)
    if resolution <= 0.0:
        msg = f"the resolution must be positive, got {resolution!r}"
        raise ProcessingError(msg)
    if samples < 1:
        msg = f"samples must be at least 1, got {samples}"
        raise ProcessingError(msg)
    if x.size < 2 or np.any(np.diff(x) <= 0.0):
        msg = "the abscissa of a resolution smoothing must be strictly increasing"
        raise ProcessingError(msg)
    measure = np.abs(np.gradient(x)) if weights is None else np.asarray(weights, float)
    low, high = _output_limits(x, spec_range)
    n_out = max(int(np.ceil((high - low) / resolution * samples)), 2)
    x_out = np.linspace(low, high, n_out)
    step = (high - low) / (n_out - 1)
    sigma = resolution * _FWHM_TO_SIGMA

    # The output axis is uniform, so every input bin reaches a fixed number of output
    # bins: scatter each into that footprint instead of forming an (n_x, n_out) kernel.
    half = int(np.ceil(_KERNEL_SPAN * resolution / step))
    centre = np.rint((x - low) / step).astype(int)
    flat = density.reshape(-1, x.size).T * measure[:, np.newaxis]
    out = np.zeros((n_out, flat.shape[1]))
    for offset in range(-half, half + 1):
        column = centre + offset
        inside = (column >= 0) & (column < n_out)
        distance = low + column[inside] * step - x[inside]
        kernel = np.exp(-0.5 * (distance / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        np.add.at(out, column[inside], flat[inside] * kernel[:, np.newaxis])
    return x_out, out.T.reshape(*density.shape[:-1], n_out)


def _output_limits(
    x: Float[np.ndarray, " n_x"], spec_range: tuple[float, float] | None
) -> tuple[float, float]:
    """Return the limits of the smoothed axis, clipped to the data.

    Parameters
    ----------
    x : numpy.ndarray
        Input abscissa, increasing.
    spec_range : tuple of float or None
        Requested limits, in either order, or ``None`` for the whole axis.

    Returns
    -------
    tuple of float
        The lower and upper limits.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the requested range does not overlap the data.
    """
    if spec_range is None:
        return float(x[0]), float(x[-1])
    low = max(float(min(spec_range)), float(x[0]))
    high = min(float(max(spec_range)), float(x[-1]))
    if high <= low:
        msg = (
            f"the requested range {spec_range!r} does not overlap the data, which "
            f"spans {float(x[0])!r} to {float(x[-1])!r}"
        )
        raise ProcessingError(msg)
    return low, high


def _as_window(grid: TimeGrid, spec: Bandpass) -> Float[np.ndarray, " n_omega"]:
    """Return the window an accessor's ``bandpass`` keyword asks for.

    A pair of wavelengths is turned into a Planck taper of the default width; an array
    is taken as it stands, which is how a caller uses a sharper edge than the default,
    a different profile, or a window it built once and reuses.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid whose ``omega`` axis the window lives on.
    spec : tuple of float or numpy.ndarray
        The pass band, m, or the window itself, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The window, shape ``(n_omega,)``, in storage order.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If an array is offered whose length is not the grid's.
    """
    if isinstance(spec, np.ndarray):
        if spec.shape != (grid.n_omega,):
            msg = (
                f"a bandpass window must have one value per stored bin, got shape "
                f"{spec.shape} for a grid of {grid.n_omega}"
            )
            raise ProcessingError(msg)
        return spec
    return bandpass(grid, spec)
