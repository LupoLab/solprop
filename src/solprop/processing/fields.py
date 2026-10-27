# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Read fields, spectra and spectral densities out of a saved run.

The field, spectrum and density accessors, in solprop's conventions. Each one takes an
:class:`~solprop.output.schema.Output` -- a memory handler or an HDF5 handler, through
the same protocol -- selects a saved plane and a set of modes, and returns an abscissa
and an array whose trivial mode and polarisation axes have been squeezed away.

Three rules hold throughout.

**Convention.** Arrays come back in the optics convention by default and in the FFT
convention on request. For a spectrum the difference is the conjugation *and*
the phase origin, both of which the output layer applies; for a time-domain array it is
the conjugation alone. A real field, an energy and a density are convention-free.

**Normalisation.** :func:`get_spectrum` carries the spectral normalisation, so that
``abs(spectrum) ** 2`` is the spectral energy density in J/(rad/s) --
:func:`solprop.fourier.spectral_density` of the same plane, to round-off. The stored
array itself, the raw forward DFT, is ``output.fields()``.

**Band.** The default is the grid's active band, on an ascending
frequency axis. Pass ``band="all"`` for every stored bin, remembering that a field grid
then includes the zero-frequency bin, where a wavelength is undefined.

No window is ever applied implicitly. ``bandpass`` is a keyword, it is a wavelength
range, and nothing happens without it.

**The noise floor.** A run with input shot noise saves a field that carries its seed,
one photon per active bin and channel. ``noise_floor="subtract"`` on
:func:`get_spectral_density` removes exactly that seed's energy, bin by bin and channel
by channel, and returns the signed remainder (SEP 37, section 5.5).
"""

from typing import Literal

import numpy as np
from jaxtyping import Complex, Float, Inexact

from solprop import fourier
from solprop.exceptions import ProcessingError
from solprop.grid import TimeGrid
from solprop.optics import TransferFunction, stored_factor
from solprop.output.schema import Convention, Output
from solprop.processing import spectra
from solprop.processing.axes import (
    Band,
    ModeSelection,
    SpectralAxis,
    _apply_modes,
    _check_choice,
    _keep_index,
    _nearest_plane,
    _plane_field,
    _squeeze,
    grid_of,
    spectral_axis,
)

__all__ = [
    "FIELD_KINDS",
    "NOISE_FLOORS",
    "FieldKind",
    "ModeSelection",
    "NoiseFloor",
    "get_field",
    "get_phase",
    "get_power",
    "get_spectral_density",
    "get_spectrum",
    "nearest_plane",
]

FieldKind = Literal["analytic", "real", "envelope"]
"""Which time-domain representation :func:`get_field` returns."""

FIELD_KINDS: tuple[FieldKind, ...] = ("analytic", "real", "envelope")
"""The analytic signal, the real field, and the analytic signal without its carrier."""

NoiseFloor = Literal["keep", "subtract"]
"""Whether a spectral density keeps or removes the input noise's one-photon seed."""

NOISE_FLOORS: tuple[NoiseFloor, ...] = ("keep", "subtract")
"""The saved density as it stands, and the density less the seed's energy."""


def nearest_plane(output: Output, z: float) -> int:
    """Return the index of the saved plane nearest a position.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler holding at least one plane.
    z : float
        Position along the propagation, m.

    Returns
    -------
    int
        Index into the saved planes.

    Raises
    ------
    solprop.exceptions.OutputError
        If no plane has been saved.

    Examples
    --------
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.fields import nearest_plane
    >>> nearest_plane(example_output(), 0.049)
    2
    """
    return _nearest_plane(output, z)


def get_spectrum(  # noqa: PLR0913 - a plane, a mode set, a band, a convention, a floor
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    band: Band = "active",
    convention: Convention = "optics",
    noise_floor: NoiseFloor = "keep",
) -> tuple[Float[np.ndarray, " n_x"], Complex[np.ndarray, "*shape n_x"]]:
    """Return the complex spectrum of one plane, or of every plane.

    The stored raw forward DFT is scaled by the spectral normalisation --
    ``2 dt / sqrt(2 pi)`` on a field grid and ``dt / sqrt(2 pi)`` on an envelope grid --
    so that the modulus squared is the spectral energy density in J/(rad/s)
    (``conventions.md`` section 4). The axis is ascending and, by default, the active
    band.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end. With ``z``, at most one of the
        two; with neither, every plane is returned and the result carries a leading
        ``n_z`` axis.
    z : float or None
        Position of the plane wanted, m; the nearest saved plane is used.
    modes : int, sequence of int, 'sum' or None
        Which modes to return. ``'sum'`` is refused here: modal amplitudes are complex
        and adding them is meaningless away from a point in space.
    band : {'active', 'all'}
        Which part of the stored axis to return.
    convention : {'optics', 'fft'}
        Sign convention of the returned spectrum.
    noise_floor : {'keep'}
        Only ``'keep'``. The noise floor is a power, one photon per bin, and an
        amplitude has no part that is that power: the saved amplitude is the pulse and
        the seed added coherently, and their interference is as large as the seed
        itself, so no amplitude has the floor-subtracted density as its modulus
        squared, which can be negative. ``'subtract'`` is refused and named;
        :func:`get_spectral_density` subtracts it.

    Returns
    -------
    omega : numpy.ndarray
        Angular frequency, shape ``(n_x,)``, rad/s, ascending.
    spectrum : numpy.ndarray
        The spectrum, in the square root of J/(rad/s), with the leading ``n_z`` axis
        when no plane was selected and trivial mode and polarisation axes squeezed.

    Raises
    ------
    solprop.exceptions.ModeError
        If ``modes='sum'``.
    solprop.exceptions.ProcessingError
        If ``noise_floor`` is not ``'keep'``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.fields import get_spectrum
    >>> omega, spectrum = get_spectrum(example_output(), plane=-1)
    >>> spectrum.shape == omega.shape
    True
    """
    _check_choice(noise_floor, NOISE_FLOORS, "noise_floor")
    if noise_floor != "keep":
        msg = (
            "noise_floor='subtract' removes a power, one photon per bin, and a "
            "spectrum is an amplitude, whose modulus squared cannot be negative; ask "
            "get_spectral_density for the floor-subtracted density"
        )
        raise ProcessingError(msg)
    grid = grid_of(output)
    values = _plane_field(output, plane, z, convention=convention)
    values = values * _fft_norm(grid)
    values, kept = _apply_modes(values, modes, allow_sum=False)
    keep = _keep_index(grid, band)
    return grid.omega[keep], _squeeze(values[..., keep], mode_axis=kept)


def get_spectral_density(  # noqa: PLR0913 - the abscissa and the smoothing on top
    output: Output,
    axis: SpectralAxis = "wavelength",
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    band: Band = "active",
    spec_range: tuple[float, float] | None = None,
    resolution: float | None = None,
    noise_floor: NoiseFloor = "keep",
) -> tuple[Float[np.ndarray, " n_x"], Float[np.ndarray, "*shape n_x"]]:
    """Return the spectral energy density of one plane, or of every plane.

    The density is per unit of the abscissa asked for: J/(rad/s),
    J/Hz or J/m, with the Jacobian of :func:`solprop.processing.axes.spectral_axis`
    applied. The wavelength abscissa is returned ascending, so the density is reversed
    with it. Polarisation components are summed, because they are two components of one
    field and their densities add.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    axis : {'wavelength', 'f', 'omega'}
        The abscissa, in m, Hz or rad/s.
    plane : int or None
        Index of the plane wanted, negative from the end; see :func:`get_spectrum`.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to return; ``'sum'`` adds their densities.
    band : {'active', 'all'}
        Which part of the stored axis to use.
    spec_range : tuple of float or None
        Limits on the abscissa, in its own units, applied after the change of variable.
    resolution : float or None
        Instrument resolution, in the units of the abscissa, as a full width at half
        maximum. When given, the density is resampled onto a uniform axis and smoothed
        by :func:`solprop.processing.spectra.specres`.
    noise_floor : {'keep', 'subtract'}
        ``'keep'`` returns the saved density as it stands. ``'subtract'`` removes the
        input noise's seed: the energy of the stored realisation, which is one photon
        per active bin and channel in the route's own photon measure (the one
        :func:`solprop.noise.photons_per_bin` inverts), before the polarisation
        components are summed and any mode selection or smoothing. The draw has a
        fixed modulus, so the subtraction is exact for the one realisation; where the
        linear operator only changes the phase, it equals the modified model's
        corrected readout ``|psi + N|^2 - |N|^2`` of the same draw. It is not a photon
        estimator under loss, where the seed has decayed and more is removed than
        remains. Values near the floor can be negative, by the interference of the
        pulse with the seed and by its decay, and are returned signed. Refused for a
        run without noise and for the ``'modified'`` model, whose saved field holds no
        seed.

    Returns
    -------
    x : numpy.ndarray
        The abscissa, shape ``(n_x,)``, ascending.
    density : numpy.ndarray
        The spectral energy density, J per unit of ``x``; signed under
        ``noise_floor='subtract'``.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If ``noise_floor`` is not one of :data:`NOISE_FLOORS`, or is ``'subtract'``
        for a run without input noise.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.fields import get_spectral_density
    >>> wavelength, density = get_spectral_density(example_output(), plane=0)
    >>> bool(np.all(np.diff(wavelength) > 0))
    True
    """
    _check_choice(noise_floor, NOISE_FLOORS, "noise_floor")
    grid = grid_of(output)
    values = _plane_field(output, plane, z, convention="optics")
    density = np.abs(values * _fft_norm(grid)) ** 2
    if noise_floor == "subtract":
        density = density - _seed_density(output, grid)
    density = np.sum(density, axis=-2, keepdims=True)
    density, kept = _apply_modes(density, modes, allow_sum=True)
    keep = _keep_index(grid, band)
    omega = grid.omega[keep]
    x, jacobian = spectral_axis(omega, axis)
    density = _squeeze(density[..., keep] * jacobian, mode_axis=kept)
    if axis == "wavelength":
        # The wavelength abscissa decreases where omega increases; both are reversed so
        # that every accessor hands back an ascending axis.
        x, density = x[::-1], density[..., ::-1]
    if resolution is not None:
        # Integrating in omega with a window evaluated on the target axis is the same
        # sum as the target-axis integral with the measure d omega / |d omega / dx|.
        weights = np.abs(grid.d_omega / jacobian)
        if axis == "wavelength":
            weights = weights[::-1]
        return spectra.specres(
            x, density, resolution, weights=weights, spec_range=spec_range
        )
    if spec_range is not None:
        inside = (x >= min(spec_range)) & (x <= max(spec_range))
        return x[inside], density[..., inside]
    return x, density


def get_field(  # noqa: PLR0913 - a plane, a mode set, a representation and a window
    output: Output,
    *,
    kind: FieldKind = "analytic",
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    oversampling: int = 1,
    bandpass: spectra.Bandpass | None = None,
    convention: Convention = "optics",
    transfer: TransferFunction | None = None,
    transform_limit: bool = False,
) -> tuple[Float[np.ndarray, " n_t"], Inexact[np.ndarray, "*shape n_t"]]:
    """Return the time-domain field of one plane, or of every plane.

    On either grid and in three representations: the analytic signal
    ``E_a(t)``, whose modulus squared is the instantaneous power in watts; the real
    field ``Re E_a``; and the envelope ``A(t) = E_a(t) exp(-i omega0 t)``. On an
    envelope grid the stored state already is the envelope, and on a field grid the
    analytic signal is formed by :func:`solprop.fourier.analytic_signal`; one definition
    serves both.

    Oversampling zero-pads the spectrum, so the returned time axis is the grid's own
    fine axis: centred on zero, of length ``oversampling * n_t``.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    kind : {'analytic', 'real', 'envelope'}
        Which representation to return.
    plane : int or None
        Index of the plane wanted, negative from the end; see :func:`get_spectrum`.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to return. ``'sum'`` is refused: modal amplitudes are complex.
    oversampling : int
        Factor by which to resample the time axis, at least 1.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit Planck-tapered window applied to the spectrum before the
        transform, or a ready-made window on the stored axis.
    convention : {'optics', 'fft'}
        Sign convention of the complex representations; ``'real'`` is convention-free.
    transfer : TransferFunction or None
        A linear optical element of :mod:`solprop.optics` applied to the spectrum before
        the transform -- a compressor, a window, a stretch of air -- in the optics
        convention.
    transform_limit : bool
        Whether to discard the spectral phase first, giving the shortest pulse the
        spectrum supports, centred on ``t = 0``. Applied after
        ``bandpass`` and before ``transfer``.

    Returns
    -------
    t : numpy.ndarray
        Time axis, shape ``(n_t,)`` or ``(oversampling n_t,)``, s, centred on zero.
    field : numpy.ndarray
        The field, in the square root of watts, with trivial axes squeezed; real for
        ``kind='real'`` and complex otherwise.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If ``kind`` is unknown or ``oversampling`` is below one.
    solprop.exceptions.ModeError
        If ``modes='sum'``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.fields import get_field
    >>> t, field = get_field(example_output(), plane=0, kind="real")
    >>> bool(np.isrealobj(field))
    True
    """
    _check_choice(kind, FIELD_KINDS, "kind")
    grid, t, amplitude = _plane_amplitude(
        output,
        plane=plane,
        z=z,
        oversampling=oversampling,
        bandpass=bandpass,
        transfer=transfer,
        transform_limit=transform_limit,
    )
    amplitude, kept = _apply_modes(amplitude, modes, allow_sum=False)
    # The stored envelope has its carrier removed and the stored real field does not, so
    # exactly one of the two representations needs the phasor, whichever grid this is.
    if (grid.kind == "envelope") != (kind == "envelope"):
        sign = 1.0 if grid.kind == "envelope" else -1.0
        amplitude = amplitude * np.exp(sign * 1j * grid.omega0 * t)
    if kind == "real":
        return t, _squeeze(np.real(amplitude), mode_axis=kept)
    if convention == "optics":
        amplitude = fourier.to_optics(amplitude)
    return t, _squeeze(amplitude, mode_axis=kept)


def get_power(  # noqa: PLR0913 - a plane, a mode set, a window and an optical element
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    oversampling: int = 1,
    bandpass: spectra.Bandpass | None = None,
    transfer: TransferFunction | None = None,
    transform_limit: bool = False,
) -> tuple[Float[np.ndarray, " n_t"], Float[np.ndarray, "*shape n_t"]]:
    """Return the instantaneous power of one plane, or of every plane.

    ``P(t) = sum_pol abs(E_a(t)) ** 2`` in watts, the quantity every temporal metric and
    every temporal plot is built on. Polarisation components are summed because they are
    two components of one field whose powers add at each instant, which is the reduction
    :func:`solprop.stats.peak_power` uses.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end; see :func:`get_spectrum`.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to return; ``'sum'`` adds their powers.
    oversampling : int
        Factor by which to resample the time axis, at least 1.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the transform, or a ready-made window
        on the stored axis.
    transfer : TransferFunction or None
        A linear optical element of :mod:`solprop.optics` applied to the spectrum before
        the transform -- a compressor, a window, a stretch of air -- in the optics
        convention.
    transform_limit : bool
        Whether to discard the spectral phase first, giving the shortest pulse the
        spectrum supports, centred on ``t = 0``. Applied after
        ``bandpass`` and before ``transfer``.

    Returns
    -------
    t : numpy.ndarray
        Time axis, shape ``(n_t,)`` or ``(oversampling n_t,)``, s.
    power : numpy.ndarray
        Instantaneous power, W, with trivial axes squeezed.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.fields import get_power
    >>> t, power = get_power(example_output(), plane=0)
    >>> bool(np.all(power >= 0.0))
    True
    """
    _, t, amplitude = _plane_amplitude(
        output,
        plane=plane,
        z=z,
        oversampling=oversampling,
        bandpass=bandpass,
        transfer=transfer,
        transform_limit=transform_limit,
    )
    power = np.sum(np.abs(amplitude) ** 2, axis=-2, keepdims=True)
    power, kept = _apply_modes(power, modes, allow_sum=True)
    return t, _squeeze(power, mode_axis=kept)


def get_phase(
    output: Output,
    *,
    plane: int | None = None,
    z: float | None = None,
    modes: ModeSelection = None,
    band: Band = "active",
) -> tuple[Float[np.ndarray, " n_x"], Float[np.ndarray, "*shape n_x"]]:
    """Return the unwrapped spectral phase of one plane, or of every plane.

    The phase of the optics-convention spectrum, unwrapped along the frequency axis.
    The optics convention has already moved the phase origin from the first time sample
    to the centre of the window (``conventions.md`` section 2), so no linear ramp
    ``omega T / 2`` is subtracted and a flat phase means a transform-limited pulse.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Index of the plane wanted, negative from the end; see :func:`get_spectrum`.
    z : float or None
        Position of the plane wanted, m.
    modes : int, sequence of int, 'sum' or None
        Which modes to return. ``'sum'`` is refused: a phase is not additive.
    band : {'active', 'all'}
        Which part of the stored axis to use.

    Returns
    -------
    omega : numpy.ndarray
        Angular frequency, shape ``(n_x,)``, rad/s, ascending.
    phase : numpy.ndarray
        Unwrapped spectral phase, rad, with trivial axes squeezed.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.fields import get_phase
    >>> omega, phase = get_phase(example_output(), plane=0)
    >>> bool(np.all(np.isfinite(phase)))
    True
    """
    omega, spectrum = get_spectrum(
        output, plane=plane, z=z, modes=modes, band=band, convention="optics"
    )
    return omega, np.unwrap(np.angle(spectrum), axis=-1)


def _seed_density(
    output: Output, grid: TimeGrid
) -> Float[np.ndarray, "n_modes n_pol n_omega"]:
    """Return the input seed's spectral energy density, per channel and component.

    The stored realisation is the draw the launch was given, in the saved field's own
    units, so its density is the floor in the route's own photon measure:
    ``hbar omega / d_omega`` in every active bin of a modal route in its own basis, the
    redistributed share of a declared unitary basis, and the admittance-weighted
    column measure of a spatial one. A modulus is convention-free, so the record's
    internal convention needs no conjugation.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    grid : solprop.grid.TimeGrid
        The run's grid.

    Returns
    -------
    numpy.ndarray
        The density in J/(rad/s), shape ``(n_modes, n_pol, n_omega)``.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the run carries no noise record, or one of the ``'modified'`` model.
    """
    metadata = output.metadata
    record = None if metadata is None else metadata.noise
    if record is None or record.model not in ("input", "modified"):
        msg = (
            "noise_floor='subtract' removes the input noise's seed, and this run "
            "carries no noise: its density has no floor to subtract"
        )
        raise ProcessingError(msg)
    if record.model == "modified":
        msg = (
            "noise_floor='subtract' removes the seed a saved field carries, and a "
            "'modified' run's saved field holds none: its draw is kept apart from the "
            "field and never saved in it"
        )
        raise ProcessingError(msg)
    return np.abs(np.asarray(record.spectrum) * _fft_norm(grid)) ** 2


def _fft_norm(grid: TimeGrid) -> float:
    """Return the factor that turns a stored spectrum into a spectral amplitude.

    The square root of the prefactor of :func:`solprop.fourier.spectral_density`, so
    that the modulus squared of the scaled spectrum is the energy density in
    J/(rad/s).

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.

    Returns
    -------
    float
        ``dt sqrt(2 / pi)`` on a field grid, ``dt / sqrt(2 pi)`` on an envelope grid.
    """
    if grid.kind == "field":
        return grid.dt * np.sqrt(2.0 / np.pi)
    return grid.dt / np.sqrt(2.0 * np.pi)


def _to_time(
    grid: TimeGrid,
    stored: Complex[np.ndarray, "*rest n_omega"],
    *,
    oversampling: int,
    bandpass: spectra.Bandpass | None,
    transfer: TransferFunction | None = None,
    transform_limit: bool = False,
) -> tuple[Float[np.ndarray, " n_t"], Complex[np.ndarray, "*rest n_t"]]:
    """Return the time axis and the amplitude whose modulus squared is the power.

    That amplitude is the analytic signal on a field grid and the carrier-free envelope
    on an envelope grid -- the same pair :func:`solprop.stats` works with, so that a
    power computed here is bit-identical to a power computed there. Restoring the
    carrier is the caller's business, because doing it unconditionally would cost a
    complex multiply and the last bit of every power.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid the spectrum lives on.
    stored : numpy.ndarray
        Stored spectrum, internal convention, shape ``(..., n_omega)``.
    oversampling : int
        Factor by which to resample the time axis, at least 1.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the transform, or a ready-made window
        on the stored axis.
    transfer : TransferFunction or None
        An optics-convention element applied to the spectrum, after the band pass and
        the transform limit.
    transform_limit : bool
        Whether to discard the spectral phase first.

    Returns
    -------
    t : numpy.ndarray
        Time axis, s, centred on zero.
    amplitude : numpy.ndarray
        The complex amplitude, internal convention, in the square root of watts.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If ``oversampling`` is below one.
    """
    if oversampling < 1:
        msg = f"oversampling must be at least 1, got {oversampling}"
        raise ProcessingError(msg)
    if bandpass is not None:
        stored = stored * spectra._as_window(grid, bandpass)
    if transform_limit:
        # Zero spectral phase about the window centre, which in a stored spectrum is
        # the alternating sign of the half-window origin shift (conventions section 2).
        stored = fourier.spectrum_to_internal(
            np.abs(stored).astype(np.complex128), kind=grid.kind
        )
    if transfer is not None:
        stored = stored * stored_factor(grid, transfer)
    n_t = oversampling * grid.n_t
    if oversampling > 1:
        stored = fourier.oversample(stored, n_t, kind=grid.kind)
    samples = fourier.inverse(stored, kind=grid.kind)
    amplitude = fourier.analytic_signal(samples) if grid.kind == "field" else samples
    t = (np.arange(n_t, dtype=float) - n_t / 2) * (grid.dt / oversampling)
    return t, amplitude


def _plane_amplitude(  # noqa: PLR0913 - the plane and everything done to its spectrum
    output: Output,
    *,
    plane: int | None,
    z: float | None,
    oversampling: int,
    bandpass: spectra.Bandpass | None,
    transfer: TransferFunction | None = None,
    transform_limit: bool = False,
) -> tuple[TimeGrid, Float[np.ndarray, " n_t"], Complex[np.ndarray, "*planes n_t"]]:
    """Return the grid, the time axis and the amplitude of the selected planes.

    The one path from a saved run to the time domain, so that a power, a field and an
    autocorrelation all come from the same transform with the same normalisation.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int or None
        Plane index, negative from the end, or ``None`` for every plane.
    z : float or None
        Plane position, m, or ``None``.
    oversampling : int
        Factor by which to resample the time axis, at least 1.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Wavelength limits, m, or a :class:`~solprop.processing.spectra.Passband` on any
        axis, of an explicit window applied before the transform, or a ready-made window
        on the stored axis.
    transfer : TransferFunction or None
        An optics-convention element applied to the spectrum.
    transform_limit : bool
        Whether to discard the spectral phase first.

    Returns
    -------
    grid : solprop.grid.TimeGrid
        The rebuilt grid.
    t : numpy.ndarray
        Time axis, s, centred on zero.
    amplitude : numpy.ndarray
        Complex amplitude with every axis kept, shape ``(..., n_modes, n_pol, n_t)``, in
        the internal convention and the square root of watts.
    """
    grid = grid_of(output)
    stored = _plane_field(output, plane, z, convention="fft")
    t, amplitude = _to_time(
        grid,
        stored,
        oversampling=oversampling,
        bandpass=bandpass,
        transfer=transfer,
        transform_limit=transform_limit,
    )
    return grid, t, amplitude
