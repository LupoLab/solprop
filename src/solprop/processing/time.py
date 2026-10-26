# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pulse metrics: energy, peak power, durations, arrival time and autocorrelations.

The metrics follow two rules. Where :mod:`solprop.stats`
already owns a definition -- the energy, the peak power, the two temporal widths -- this
module
**calls it**, so a metric read off a saved run and the same metric recorded during the
run are the same number to the bit. Where it does not -- a spectral width, an arrival
time, an autocorrelation -- the definition lives here, on the general
half-maximum and moment helpers below.

Polarisation components are summed wherever the quantity is quadratic in the field, as
they are in :mod:`solprop.stats`: they are two components of one field and their powers
add at each instant. Modes are not summed unless asked: ``modes=None`` gives one number
per mode, ``modes='sum'`` the number of the total.

**What the stored basis admits is checked here too.** :func:`energy` and
:func:`peak_power` are the two metrics of this module whose meaning depends on the mode
axis being an orthogonal basis whose coordinate powers add, so on a general modal file
they are refused by name, with the same text the run loop and
:mod:`solprop.processing.modal` use (:mod:`solprop.output.capability`). The widths, the
spectral metrics and the autocorrelations are shape functionals of the same trace and
are admitted on every basis.
"""

from collections.abc import Callable
from typing import Literal

import numpy as np
from jaxtyping import Float, Inexact

from solprop import stats
from solprop.exceptions import OutputError
from solprop.grid import TimeGrid
from solprop.output.capability import capability_of, refusal_for, withheld_from
from solprop.output.schema import Output
from solprop.processing import spectra
from solprop.processing.axes import (
    ModeSelection,
    _apply_modes,
    _check_choice,
    _check_mode_index,
    _check_mode_selection,
    _plane_field,
    _squeeze,
    grid_of,
)
from solprop.processing.fields import (
    _plane_amplitude,
    get_power,
    get_spectral_density,
)

__all__ = [
    "ARRIVAL_METHODS",
    "ArrivalMethod",
    "arrival_time",
    "coherence_time",
    "energy",
    "field_autocorrelation",
    "fwhm",
    "fwhm_f",
    "fwhm_t",
    "half_max_range",
    "intensity_autocorrelation",
    "moment",
    "peak_power",
    "time_bandwidth",
]

ArrivalMethod = Literal["moment", "peak"]
"""Whether an arrival time is the first moment of the power or the time of its peak."""

ARRIVAL_METHODS: tuple[ArrivalMethod, ...] = ("moment", "peak")
"""The two definitions of an arrival time."""


# --- general trace helpers -----------------------------------------------------------


def moment(
    x: Float[np.ndarray, " n"], y: Float[np.ndarray, " n"], order: int = 1
) -> float:
    """Return a normalised moment of a trace, ``sum(x**order y) / sum(y)``.

    The sample spacing cancels between numerator and
    denominator, so none is applied and the answer is exact on a uniform axis.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa, shape ``(n,)``.
    y : numpy.ndarray
        Weight, shape ``(n,)``, non-negative for the result to be a centroid.
    order : int
        Which moment, one for the centroid.

    Returns
    -------
    float
        The moment, in the units of ``x`` raised to ``order``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing.time import moment
    >>> moment(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 1.0]))
    1.5
    """
    return float(np.sum(x**order * y) / np.sum(y))


def half_max_range(
    x: Float[np.ndarray, " n"], y: Float[np.ndarray, " n"], *, outer: bool = False
) -> tuple[float, float]:
    """Return the two points at which a trace crosses half its maximum.

    The level is half the global peak with no baseline subtraction, and the crossing is
    interpolated linearly between the two samples that bracket it, in the stable form
    ``x1 + (level - y1)(x2 - x1)/(y2 - y1)``. ``outer=False`` walks outwards from the
    peak and stops at the first sample below the level on either side, so structure
    further out does not widen the answer; ``outer=True`` takes the first and last
    sample above the level anywhere. These are the two conventions
    :func:`solprop.stats.fwhm_t` and :func:`solprop.stats.fwhm_t_outer` use for the
    temporal power, here for any trace, and the same code: both widths are measured by
    the one implementation this function exposes.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa, shape ``(n,)``, increasing.
    y : numpy.ndarray
        The trace, shape ``(n,)``.
    outer : bool
        Whether to take the outermost pair of crossings.

    Returns
    -------
    tuple of float
        The lower and upper crossings, both ``nan`` when the trace carries nothing, does
        not fall to half its maximum on both sides, or reaches the end of the axis still
        above it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing.time import half_max_range
    >>> x = np.linspace(-4.0, 4.0, 801)
    >>> low, high = half_max_range(x, np.exp(-(x**2)))
    >>> bool(np.isclose(high - low, 2 * np.sqrt(np.log(2)), atol=1e-4))
    True
    """
    return stats._half_max_range(x, y, outer=outer)


def fwhm(
    x: Float[np.ndarray, " n"], y: Float[np.ndarray, " n"], *, outer: bool = False
) -> float:
    """Return the full width of a trace at half its maximum.

    The distance between the two crossings of :func:`half_max_range`: those nearest the
    peak, or with ``outer`` the first and last anywhere.

    Parameters
    ----------
    x : numpy.ndarray
        Abscissa, shape ``(n,)``, increasing.
    y : numpy.ndarray
        The trace, shape ``(n,)``.
    outer : bool
        Whether to span every satellite rather than the central feature alone.

    Returns
    -------
    float
        The width in the units of ``x``, or ``nan`` where it is undefined.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing.time import fwhm
    >>> x = np.linspace(-4.0, 4.0, 801)
    >>> bool(np.isclose(fwhm(x, np.exp(-(x**2))), 1.6651, atol=1e-4))
    True
    """
    low, high = half_max_range(x, y, outer=outer)
    return high - low


# --- metrics of a saved run ----------------------------------------------------------


def energy(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
) -> Float[np.ndarray, "*shape"]:
    """Return the pulse energy of a plane, or of every plane, in joules.

    :func:`solprop.stats.energy`, which is the exact discrete functional of
    ``conventions.md`` section 4, summed over polarisation.

    **A general modal file is refused by name**, as it is at the run loop, in both
    CPU backends and in :mod:`solprop.processing.modal`: a coordinate energy is not
    additive on a three-component basis, and
    :func:`solprop.processing.modal.physical_energy` is that basis's own energy
    (:mod:`solprop.output.capability`). Every older file, and every schema-7 run on
    an ordinary basis, is unaffected.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end; with neither ``plane`` nor
        ``z``, every plane is returned with a leading ``n_z`` axis.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report; ``'sum'`` adds their energies.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the measurement, or a ready-made
        window on the stored axis.

    Returns
    -------
    numpy.ndarray
        The energy, J, with the plane and mode axes the selection leaves.

    Raises
    ------
    solprop.exceptions.OutputError
        If the stored run is a general modal one, which does not define this name.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import energy
    >>> values = energy(example_output())
    >>> bool(np.allclose(values, values[0], rtol=1e-12))
    True
    """
    _refuse_by_basis(output, "energy")
    grid, planes, has_z = _planes(output, plane, z, bandpass)
    per_mode = np.stack([np.sum(stats.energy(grid, one), axis=-1) for one in planes])
    return _finish(_take_modes(per_mode, modes), has_z)


def peak_power(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
    oversampling: int = 1,
) -> Float[np.ndarray, "*shape"]:
    """Return the peak power of a plane, or of every plane, in watts.

    :func:`solprop.stats.peak_power` per mode and
    :func:`solprop.stats.peak_power_total` for ``modes='sum'``, so that a modal sum is
    the peak of the total power and not the largest of the per-mode peaks: modes that
    peak at different times never carry their combined power at any one instant.
    Above unit ``oversampling`` the maximum is taken on the
    resampled trace instead, which finds a peak that falls between two samples.

    The measured quantity is the squared modulus of the analytic true-power envelope,
    ``max_t |b(t)|^2``, because a saved plane is what this is given and a file carries
    no mode. The value the run itself recorded is the analytic Poynting flux
    ``Re[a(t) conj(a_w(t))]`` of ``docs/guide/results/power-and-intensity.md``, which
    needs the
    plane's index; it is in ``stats/peak_power`` and is the number to quote. The two
    differ at second order in the variation of ``Re n_eff`` across the spectrum -- of
    order 1e-7 of the peak for a gas-filled capillary, 7.95e-8 on the reduced
    comb-to-continuum witness -- and their time integrals agree exactly by Parseval,
    so the energy does not move at all.

    **A general modal file is refused by name**, as it is at the run loop, in both CPU
    backends and in :mod:`solprop.processing.modal`: ``modes='sum'`` here is
    :func:`solprop.stats.peak_power_total`, which promises a sum of coordinate powers
    that a three-component basis does not make, and
    :func:`solprop.processing.modal.reconstructed_flux` is that basis's own flux
    (:mod:`solprop.output.capability`). Every older file, and every schema-7 run on an
    ordinary basis, is unaffected.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the measurement, or a ready-made
        window on the stored axis.
    oversampling : int
        Factor by which to resample the time axis before taking the maximum.

    Returns
    -------
    numpy.ndarray
        The peak power, W.

    Raises
    ------
    solprop.exceptions.OutputError
        If the stored run is a general modal one, which does not define this name.

    Examples
    --------
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import peak_power
    >>> float(peak_power(example_output(), plane=0)) > 0.0
    True
    """
    _refuse_by_basis(output, "peak_power")
    if oversampling > 1:
        _, power = get_power(
            output,
            plane=plane,
            z=z,
            modes=modes,
            oversampling=oversampling,
            bandpass=bandpass,
        )
        return np.asarray(np.max(power, axis=-1))
    grid, planes, has_z = _planes(output, plane, z, bandpass)
    if modes == "sum":
        total = np.stack([stats.peak_power_total(grid, one) for one in planes])
        return _finish(total, has_z)
    per_mode = np.stack([stats.peak_power(grid, one) for one in planes])
    return _finish(_take_modes(per_mode, modes), has_z)


def fwhm_t(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
    outer: bool = False,
) -> Float[np.ndarray, "*shape"]:
    """Return the temporal width at half maximum of a plane, or of every plane.

    :func:`solprop.stats.fwhm_t`, the width of the central feature of the total power,
    or :func:`solprop.stats.fwhm_t_outer` with ``outer=True``, the total extent above
    half maximum. A trace that never falls to half its maximum has no width and gives
    ``nan``.

    The trace measured here is ``|b(t)|^2``, the squared modulus of the analytic
    true-power envelope, because a saved plane carries no mode index. The run's own
    ``stats/fwhm_t`` is the width of the analytic Poynting flux
    ``Re[a(t) conj(a_w(t))]`` (``docs/guide/results/power-and-intensity.md``). The two
    traces
    differ at second order in the variation of ``Re n_eff`` across the spectrum, of
    order 1e-7 of the peak for a gas-filled capillary (7.95e-8 on the reduced
    comb-to-continuum witness), so the widths differ far below the sampling.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report; ``'sum'`` measures the width of their total power.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the measurement, or a ready-made
        window on the stored axis.
    outer : bool
        Whether to span every satellite rather than the central feature alone.

    Returns
    -------
    numpy.ndarray
        The width, s.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import fwhm_t
    >>> widths = fwhm_t(example_output())
    >>> bool(np.isclose(widths[-1] / widths[0], 2.0, rtol=1e-3))
    True
    """
    measure = stats.fwhm_t_outer if outer else stats.fwhm_t
    grid, planes, has_z = _planes(output, plane, z, bandpass)
    groups = _mode_groups(planes[0].shape[0], modes)
    widths = np.asarray(
        [[float(measure(grid, one[group])) for group in groups] for one in planes]
    )
    return _finish(_squeeze_modes(widths), has_z)


def fwhm_f(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    outer: bool = False,
) -> Float[np.ndarray, "*shape"]:
    """Return the spectral width at half maximum of a plane, in hertz.

    The width of the spectral energy density on the frequency axis, which is the
    quantity a time-bandwidth product needs.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report; ``'sum'`` adds their densities first.
    outer : bool
        Whether to span every satellite rather than the central feature alone.

    Returns
    -------
    numpy.ndarray
        The width, Hz.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import fwhm_f
    >>> bool(np.all(fwhm_f(example_output()) > 0.0))
    True
    """
    frequency, density = get_spectral_density(
        output, "f", plane=plane, z=z, modes=modes
    )
    return _along_last(density, lambda trace: fwhm(frequency, trace, outer=outer))


def time_bandwidth(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
) -> Float[np.ndarray, "*shape"]:
    """Return the time-bandwidth product of a plane, or of every plane.

    The product of the temporal and spectral widths at half maximum, which is
    ``2 ln 2 / pi = 0.4413`` for a transform-limited Gaussian and grows with any
    residual phase. One mode selection reaches both widths, so a modal sum does not
    silently measure the modes separately.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the temporal width, or a ready-made
        window on the stored axis.

    Returns
    -------
    numpy.ndarray
        The product, dimensionless.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import time_bandwidth
    >>> product = time_bandwidth(example_output(), plane=0)
    >>> bool(np.isclose(product, 2 * np.log(2) / np.pi, rtol=1e-2))
    True
    """
    widths = fwhm_t(output, plane=plane, z=z, modes=modes, bandpass=bandpass)
    # Multiplying two zero-dimensional arrays gives a numpy scalar; every metric
    # here returns an array, as the statistics catalogue does.
    return np.asarray(widths * fwhm_f(output, plane=plane, z=z, modes=modes))


def arrival_time(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
    method: ArrivalMethod = "moment",
) -> Float[np.ndarray, "*shape"]:
    """Return when a pulse arrives, in seconds on the grid's own time axis.

    ``'moment'`` is the first moment of the instantaneous power, which weighs the whole
    trace; ``'peak'`` is the time of its maximum, which does not. A bandpass makes this
    the arrival time of one spectral region, which is how a walk-off is measured.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the measurement, or a ready-made
        window on the stored axis.
    method : {'moment', 'peak'}
        Which definition to use.

    Returns
    -------
    numpy.ndarray
        The arrival time, s, positive for a pulse later than the window centre.

    Raises
    ------
    solprop.exceptions.GridError
        If ``method`` is not one of :data:`ARRIVAL_METHODS`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import arrival_time
    >>> bool(np.all(np.abs(arrival_time(example_output())) < 1e-16))
    True
    """
    _check_choice(method, ARRIVAL_METHODS, "method")
    t, power = get_power(output, plane=plane, z=z, modes=modes, bandpass=bandpass)
    if method == "peak":
        return _along_last(power, lambda trace: float(t[np.argmax(trace)]))
    return _along_last(power, lambda trace: moment(t, trace))


def field_autocorrelation(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
) -> tuple[Float[np.ndarray, " n_t"], Inexact[np.ndarray, "*shape n_t"]]:
    """Return the field autocorrelation of a plane, in joules.

    ``A(tau) = integral E_a*(t) E_a(t + tau) dt`` by the Wiener-Khinchin theorem, so
    that its value at zero delay is the pulse energy and its width is the coherence
    time. It carries the sample spacing rather than being a bare sample sum, which is
    what gives it units and makes the zero-delay identity exact.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report; ``'sum'`` adds their autocorrelations.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the transform, or a ready-made window
        on the stored axis.

    Returns
    -------
    delay : numpy.ndarray
        Delay axis, shape ``(n_t,)``, s, centred on zero.
    correlation : numpy.ndarray
        The autocorrelation, J.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import energy, field_autocorrelation
    >>> delay, correlation = field_autocorrelation(example_output(), plane=0)
    >>> zero = correlation[delay.size // 2]
    >>> bool(np.isclose(zero.real, energy(example_output(), plane=0), rtol=1e-12))
    True
    """
    _, t, amplitude = _plane_amplitude(
        output, plane=plane, z=z, oversampling=1, bandpass=bandpass
    )
    dt = float(t[1] - t[0])
    # Polarisation components are summed, as they are in every quadratic quantity, so
    # that the value at zero delay is the pulse energy of the whole field.
    correlation = np.sum(_wiener_khinchin(amplitude), axis=-2, keepdims=True)
    correlation, kept = _apply_modes(correlation, modes, allow_sum=True)
    return t, dt * _squeeze(correlation, mode_axis=kept)


def intensity_autocorrelation(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
) -> tuple[Float[np.ndarray, " n_t"], Float[np.ndarray, "*shape n_t"]]:
    """Return the intensity autocorrelation of a plane, in watts squared times seconds.

    ``integral P(t) P(t + tau) dt``, the quantity an autocorrelator measures, formed by
    the same circular correlation as :func:`field_autocorrelation` on the instantaneous
    power.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the transform, or a ready-made window
        on the stored axis.

    Returns
    -------
    delay : numpy.ndarray
        Delay axis, shape ``(n_t,)``, s, centred on zero.
    correlation : numpy.ndarray
        The autocorrelation, W^2 s.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import intensity_autocorrelation
    >>> delay, correlation = intensity_autocorrelation(example_output(), plane=0)
    >>> bool(np.argmax(correlation) == delay.size // 2)
    True
    """
    t, power = get_power(output, plane=plane, z=z, modes=modes, bandpass=bandpass)
    dt = float(t[1] - t[0])
    return t, dt * np.real(_wiener_khinchin(power))


def coherence_time(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    bandpass: spectra.Bandpass | None = None,
) -> Float[np.ndarray, "*shape"]:
    """Return the coherence time of a plane, in seconds.

    The width at half maximum of the squared modulus of the field autocorrelation, which
    for a Gaussian spectrum of angular width ``d_omega`` is ``2 sqrt(ln 2) / d_omega``.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to report.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the transform, or a ready-made window
        on the stored axis.

    Returns
    -------
    numpy.ndarray
        The coherence time, s.

    Examples
    --------
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.time import coherence_time
    >>> float(coherence_time(example_output(), plane=0)) > 0.0
    True
    """
    delay, correlation = field_autocorrelation(
        output, plane=plane, z=z, modes=modes, bandpass=bandpass
    )
    squared = np.abs(correlation) ** 2
    return _along_last(squared, lambda trace: fwhm(delay, trace))


# --- private helpers -----------------------------------------------------------------


def _wiener_khinchin(
    values: Inexact[np.ndarray, "*rest n_t"],
) -> Inexact[np.ndarray, "*rest n_t"]:
    """Return the circular autocorrelation of a trace, as a sample sum.

    Parameters
    ----------
    values : numpy.ndarray
        The trace, shape ``(..., n_t)``.

    Returns
    -------
    numpy.ndarray
        ``sum_t conj(v(t)) v(t + tau)``, shape ``(..., n_t)``, with zero delay at the
        centre of the axis.
    """
    spectrum = np.fft.fft(values, axis=-1)
    return np.fft.fftshift(np.fft.ifft(np.abs(spectrum) ** 2, axis=-1), axes=-1)


def _refuse_by_basis(output: Output, name: str) -> None:
    """Refuse a legacy metric the stored basis does not define, by name.

    The whitelist's fourth door (:mod:`solprop.output.capability`). The accessors of
    this module were written when every run's mode axis was an orthogonal one- or
    two-component basis whose coordinate powers add up to the physical power; a
    general modal file is not that, and ``peak_power(modes='sum')`` is
    :func:`solprop.stats.peak_power_total`, which the run loop, both CPU backends and
    :mod:`solprop.processing.modal` all refuse on that basis. Without this check the
    same number came back through a different import.

    A file written before output schema 7 carries no identity and is the ordinary
    modal run it has always been, so nothing changes for it; nor for a schema-7 run
    on an ordinary basis, which is every gas-filled run.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    name : str
        The accessor being asked for, which the refusal names.

    Raises
    ------
    solprop.exceptions.OutputError
        If the stored basis does not admit `name`. It is an
        :class:`~solprop.exceptions.OutputError` and not a
        :class:`~solprop.exceptions.ProcessingError` because it is a statement about
        the record, and because one gate raising two classes would make the refusal
        harder to catch than to cause: :mod:`solprop.processing.modal` raises the
        same class for the same text.
    """
    metadata = output.metadata
    if metadata is None:
        return
    refusal = refusal_for(
        capability_of(metadata), [name], withheld=withheld_from(metadata)
    )
    if refusal is not None:
        raise OutputError(refusal)


def _planes(
    output: Output,
    plane: int | None,
    z: float | None,
    bandpass: spectra.Bandpass | None,
) -> tuple[TimeGrid, list[np.ndarray], bool]:
    """Return the grid, the selected planes and whether a plane axis was asked for.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Plane index, negative from the end, or ``None``.
    z : float or None
        Plane position, m, or ``None``.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied to every plane, or a ready-made window on
        the stored axis.

    Returns
    -------
    grid : solprop.grid.TimeGrid
        The rebuilt grid.
    planes : list of numpy.ndarray
        One ``(n_modes, n_pol, n_omega)`` array per selected plane.
    has_z : bool
        Whether the caller asked for every plane.
    """
    grid = grid_of(output)
    values = _plane_field(output, plane, z, convention="fft")
    if bandpass is not None:
        values = values * spectra._as_window(grid, bandpass)
    if values.ndim == 4:
        return grid, list(values), True
    return grid, [values], False


def _mode_groups(n_modes: int, modes: ModeSelection) -> list[list[int]]:
    """Return the groups of modes a metric is evaluated over, one result each.

    Parameters
    ----------
    n_modes : int
        Number of propagated modes.
    modes : int, sequence of int, 'sum' or None
        The selection.

    Returns
    -------
    list of list of int
        Index lists; a single list of every mode for ``'sum'``.

    Raises
    ------
    solprop.exceptions.ModeError
        If an index is out of range or the selection is not recognised.
    """
    _check_mode_selection(modes)
    if modes is None:
        return [[index] for index in range(n_modes)]
    if isinstance(modes, str):
        return [list(range(n_modes))]
    if isinstance(modes, int):
        _check_mode_index(modes, n_modes)
        return [[modes]]
    for one in modes:
        _check_mode_index(one, n_modes)
    return [[one] for one in modes]


def _take_modes(
    values: Float[np.ndarray, "n_z n_modes"], modes: ModeSelection
) -> Float[np.ndarray, "*shape"]:
    """Select or sum the trailing mode axis of an already-additive quantity.

    Parameters
    ----------
    values : numpy.ndarray
        One value per plane and mode, shape ``(n_z, n_modes)``.
    modes : int, sequence of int, 'sum' or None
        The selection.

    Returns
    -------
    numpy.ndarray
        The selection, with a trivial mode axis squeezed away.
    """
    if modes == "sum":
        return np.asarray(np.sum(values, axis=-1))
    groups = _mode_groups(values.shape[-1], modes)
    return _squeeze_modes(values[:, [group[0] for group in groups]])


def _squeeze_modes(
    values: Float[np.ndarray, "n_z n_modes"],
) -> Float[np.ndarray, "*shape"]:
    """Drop the mode axis of a per-mode result when the run has one mode.

    Parameters
    ----------
    values : numpy.ndarray
        One value per plane and selected mode, shape ``(n_z, n_selected)``.

    Returns
    -------
    numpy.ndarray
        The same values without a trailing axis of length one.
    """
    return values[:, 0] if values.shape[-1] == 1 else values


def _finish(
    values: Float[np.ndarray, "*rest"], has_z: bool
) -> Float[np.ndarray, "*shape"]:
    """Drop the plane axis when a single plane was selected.

    Parameters
    ----------
    values : numpy.ndarray
        Results with a leading plane axis.
    has_z : bool
        Whether the caller asked for every plane.

    Returns
    -------
    numpy.ndarray
        The results, without the plane axis when only one plane was asked for.
    """
    # np.asarray keeps a single-plane scalar a zero-dimensional array, as every
    # scalar statistic in solprop.stats is.
    return values if has_z else np.asarray(values[0])


def _along_last(
    values: Float[np.ndarray, "*rest n"], measure: Callable[[np.ndarray], float]
) -> Float[np.ndarray, "*shape"]:
    """Apply a scalar measurement to every trace along the last axis.

    Parameters
    ----------
    values : numpy.ndarray
        Traces, shape ``(..., n)``.
    measure : callable
        A function of one trace returning a float.

    Returns
    -------
    numpy.ndarray
        The results, shape ``(...)``, zero-dimensional for a single trace.
    """
    flat = values.reshape(-1, values.shape[-1])
    out = np.asarray([measure(trace) for trace in flat])
    return out.reshape(values.shape[:-1])
