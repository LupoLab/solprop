# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fourier conventions, transforms, oversampling and energy functionals.

This module owns the sign convention of ``docs/dev/conventions.md`` §2 and the discrete
energy functionals of §4. Every array is a plain NumPy array and the frequency or time
axis is always the **last** one, because the kernel state is
``(n_batch, n_modes, n_pol, n_omega)`` with that axis contiguous (§6).

Two Fourier conventions are in play. User-facing spectra and phases use the **optics**
convention ``E(omega) = int E(t) exp(+i omega t) dt``; everything stored internally uses
the **FFT** convention that NumPy and FFTW implement directly. The boundary between them
is a conjugation, and :func:`to_optics` and :func:`to_internal` are the only sanctioned
place where it happens (``docs/dev/conventions.md`` §2). For a **spectrum** the boundary
also moves the phase origin to the window centre, which is a factor ``(-1)**k`` on the
storage index; :func:`spectrum_to_optics` and :func:`spectrum_to_internal` do both
steps.

Stored spectra are **raw forward DFTs**: no ``1/N`` and no factor of the sample spacing.
The transform is taken over the sample array as it stands, so the phase origin is the
first sample rather than ``t = 0``; the sample spacing enters only where a physical
density does, in :func:`energy_omega` and :func:`spectral_density`.

**The layout is always passed in, never inferred.** A field grid stores the one-sided
spectrum of a real field (``n_omega = n_t/2 + 1``) and an envelope grid the full
FFT-order spectrum of a complex envelope (``n_omega = n_t``). Neither dtype nor array
length identifies the two reliably — a real-valued envelope is an ordinary unchirped
pulse, and a real field cast to complex is indistinguishable from an analytic signal —
so every function that needs to know takes an explicit ``kind`` and checks the array
against it.
"""

from typing import Literal

import numpy as np
from jaxtyping import Complex, Float, Inexact, Num
from scipy import fft as _fft
from scipy import signal as _signal

from solprop.exceptions import GridError

__all__ = [
    "GridKind",
    "analytic_signal",
    "crop",
    "crop_index",
    "crop_scale",
    "energy_omega",
    "energy_t",
    "energy_weights",
    "forward",
    "inverse",
    "n_omega_for",
    "oversample",
    "pad_scale",
    "power_amplitude",
    "spectral_density",
    "spectrum_to_internal",
    "spectrum_to_optics",
    "to_internal",
    "to_optics",
]

GridKind = Literal["field", "envelope"]
"""Which of the two spectral layouts an array uses."""

# Four is the smallest number of time samples on which the two layouts differ in length,
# so it is the smallest any grid or transform helper accepts.
MIN_N_T = 4


def _check_kind(kind: str) -> None:
    """Reject a layout name that is neither ``'field'`` nor ``'envelope'``.

    The parameter is a plain string rather than :data:`GridKind` on purpose: this is the
    run-time guard for callers that ignore the annotation, and the test session's
    ``beartype`` hook would otherwise reject the bad value before this check is reached.

    Parameters
    ----------
    kind : str
        Layout name to check.

    Raises
    ------
    GridError
        If ``kind`` is not one of the two layout names.
    """
    if kind not in ("field", "envelope"):
        msg = f"kind must be 'field' or 'envelope', got {kind!r}"
        raise GridError(msg)


def _n_t_from_n_omega(n_omega: int, kind: GridKind) -> int:
    """Return the number of time samples a spectrum of ``n_omega`` bins came from.

    Parameters
    ----------
    n_omega : int
        Number of spectral bins.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    int
        The corresponding number of time samples, ``n_t``.

    Raises
    ------
    GridError
        If the bin count is too small to be a valid grid.
    """
    _check_kind(kind)
    n_t = 2 * (n_omega - 1) if kind == "field" else n_omega
    if n_t < MIN_N_T or n_t % 2:
        msg = (
            f"a {kind} spectrum of {n_omega} bins implies n_t = {n_t}, which is not an "
            f"even integer of at least {MIN_N_T}"
        )
        raise GridError(msg)
    return n_t


def n_omega_for(n_t: int, kind: GridKind) -> int:
    """Return the number of spectral bins that ``n_t`` time samples produce.

    This is the layout rule in one place: a field grid stores the one-sided spectrum of
    a real signal, an envelope grid the full spectrum of a complex one.

    Parameters
    ----------
    n_t : int
        Number of time samples, even and at least 4.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    int
        The corresponding number of spectral bins, ``n_omega``.

    Raises
    ------
    GridError
        If ``n_t`` is odd or smaller than 4.

    Examples
    --------
    >>> from solprop.fourier import n_omega_for
    >>> n_omega_for(8, kind="field"), n_omega_for(8, kind="envelope")
    (5, 8)
    """
    _check_kind(kind)
    if n_t < MIN_N_T or n_t % 2:
        msg = f"n_t must be an even integer of at least {MIN_N_T}, got {n_t}"
        raise GridError(msg)
    return n_t // 2 + 1 if kind == "field" else n_t


def _check_time_dtype(x: np.ndarray, kind: GridKind) -> None:
    """Check that time samples have the dtype their layout requires.

    Parameters
    ----------
    x : numpy.ndarray
        Time samples to check.
    kind : {'field', 'envelope'}
        Layout the samples are claimed to belong to.

    Raises
    ------
    GridError
        If a field array is complex or an envelope array is real.
    """
    _check_kind(kind)
    if kind == "field" and np.iscomplexobj(x):
        msg = (
            "a field grid carries the real electric field; got a complex array. Pass "
            "kind='envelope' for an analytic signal, or take its real part."
        )
        raise GridError(msg)
    if kind == "envelope" and not np.iscomplexobj(x):
        msg = (
            "an envelope grid carries a complex analytic signal; got a real array. "
            "Pass kind='field' for a real electric field, or cast to complex."
        )
        raise GridError(msg)


def _check_spectrum(x_omega: np.ndarray, kind: GridKind) -> int:
    """Check a spectrum's trailing axis against its layout and return ``n_t``.

    Parameters
    ----------
    x_omega : numpy.ndarray
        Spectrum whose last axis is the frequency axis.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    int
        The number of time samples the spectrum corresponds to.

    Raises
    ------
    GridError
        If the trailing axis is not a valid length for the layout.
    """
    return _n_t_from_n_omega(np.shape(x_omega)[-1], kind)


def forward(
    x: Inexact[np.ndarray, "*batch n_t"], *, kind: GridKind
) -> Complex[np.ndarray, "*batch n_omega"]:
    """Transform time samples to the stored spectrum with the raw forward DFT.

    A field grid uses a real-to-complex DFT and yields the one-sided spectrum of
    ``n_t // 2 + 1`` bins; an envelope grid uses a complex-to-complex DFT and yields the
    full spectrum of ``n_t`` bins in FFT order. No ``1/N`` and no sample spacing are
    applied (conventions.md §2), so this is the FFT-convention spectrum, not the
    optics-convention one.

    Parameters
    ----------
    x : numpy.ndarray
        Time samples, shape ``(..., n_t)``, in the square root of watts. Real for a
        field grid, complex for an envelope grid.
    kind : {'field', 'envelope'}
        Spectral layout to produce; the dtype of ``x`` is checked against it.

    Returns
    -------
    numpy.ndarray
        Complex spectrum, shape ``(..., n_omega)``.

    Raises
    ------
    GridError
        If ``kind`` is unknown, or the dtype of ``x`` does not match it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import forward
    >>> forward(np.ones(8), kind="field").shape
    (5,)
    >>> forward(np.ones(8, dtype=complex), kind="envelope").shape
    (8,)
    """
    values = np.asarray(x)
    _check_time_dtype(values, kind)
    n_omega_for(values.shape[-1], kind)
    if kind == "field":
        return _fft.rfft(values, axis=-1)
    return _fft.fft(values, axis=-1)


def inverse(
    x_omega: Complex[np.ndarray, "*batch n_omega"], *, kind: GridKind
) -> Inexact[np.ndarray, "*batch n_t"]:
    """Transform a stored spectrum back to time samples.

    The exact inverse of :func:`forward`, carrying NumPy's ``1/N``. The number of time
    samples follows from the layout, so it is not passed separately.

    Parameters
    ----------
    x_omega : numpy.ndarray
        Complex spectrum, shape ``(..., n_omega)``.
    kind : {'field', 'envelope'}
        Spectral layout of ``x_omega``.

    Returns
    -------
    numpy.ndarray
        Time samples, shape ``(..., n_t)``: real for a field grid, complex for an
        envelope grid.

    Raises
    ------
    GridError
        If ``kind`` is unknown, or the trailing axis is not a valid length for it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import forward, inverse
    >>> field = np.array([1.0, -2.0, 3.0, 0.5, 0.0, 1.5, -1.0, 2.0])
    >>> bool(np.allclose(inverse(forward(field, kind="field"), kind="field"), field))
    True
    """
    values = np.asarray(x_omega)
    n_t = _check_spectrum(values, kind)
    if kind == "field":
        return _fft.irfft(values, n=n_t, axis=-1)
    return _fft.ifft(values, n=n_t, axis=-1)


def analytic_signal(
    x: Inexact[np.ndarray, "*batch n_t"],
) -> Complex[np.ndarray, "*batch n_t"]:
    """Return the analytic signal of a real field.

    The analytic signal keeps the zero-frequency bin, doubles the positive-frequency
    bins, **keeps the Nyquist bin** and zeroes the negative-frequency bins, so that
    ``Re(analytic_signal(E))`` reproduces ``E`` to the round-off of the transform pair.
    This is :func:`scipy.signal.hilbert`'s convention.

    Zeroing the Nyquist bin instead would agree with this on any physically apodised
    grid, where the spectral collar has already driven that bin to zero, but only this
    convention makes the real part an identity, which
    ``docs/dev/conventions.md`` §4 relies on when it defines the field-grid state as
    ``E(t) = Re E_a(t)``, and only this convention makes :func:`energy_t` and
    :func:`energy_omega` an exact discrete Parseval pair.

    Parameters
    ----------
    x : numpy.ndarray
        Real field samples, shape ``(..., n_t)``.

    Returns
    -------
    numpy.ndarray
        Complex analytic signal, shape ``(..., n_t)``, whose modulus squared is the
        instantaneous power when the field is in the square root of watts.

    Raises
    ------
    GridError
        If ``x`` is complex; an envelope is already an analytic signal and must not be
        passed through a second time.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import analytic_signal
    >>> t = np.arange(16) / 16
    >>> envelope = analytic_signal(np.cos(2 * np.pi * 2 * t))
    >>> bool(np.allclose(envelope, np.exp(2j * np.pi * 2 * t)))
    True
    """
    values = np.asarray(x)
    _check_time_dtype(values, "field")
    return _signal.hilbert(values, axis=-1)


def power_amplitude(
    spectrum: Complex[np.ndarray, "... n_modes n_pol n_omega"],
    weight: Float[np.ndarray, "n_modes n_pol n_omega"],
    *,
    inverse: bool = False,
) -> Complex[np.ndarray, "... n_modes n_pol n_omega"]:
    """Convert the internal modal amplitude to true-power units, or back on resume.

    This is the amplitude boundary, beside the Fourier-sign boundary: saved modal
    amplitudes are true-power amplitudes on every route.
    It preserves the Fourier convention. The saved amplitude is ``b = sqrt(w) a``;
    its squared temporal modulus integrates to true energy. Broadband instantaneous
    Poynting flux uses the arithmetic, rather than geometric, two-frequency weight.

    Parameters
    ----------
    spectrum : numpy.ndarray
        Modal spectrum, shape ``(..., n_modes, n_pol, n_omega)``.
    weight : numpy.ndarray
        Positive real effective index, dimensionless, shape
        ``(n_modes, n_pol, n_omega)``; one for an index-one route.
    inverse : bool, optional
        Divide by the square root to recover internal units when resuming.

    Returns
    -------
    numpy.ndarray
        Converted spectrum in sqrt(W), with unchanged shape and phase convention.

    Raises
    ------
    GridError
        If a weight is not positive and finite.
    """
    if not np.all(np.isfinite(weight) & (weight > 0)):
        msg = "power weights must be positive and finite"
        raise GridError(msg)
    if np.all(weight == 1.0):
        return spectrum.copy()
    root = np.sqrt(weight)
    return spectrum / root if inverse else spectrum * root


def to_optics(x: Num[np.ndarray, "*shape"]) -> Num[np.ndarray, "*shape"]:
    """Convert an internal FFT-convention array to the optics convention.

    The two conventions differ only in the sign of the exponent, so for a real field's
    spectrum, and for an envelope in either domain, the conversion is exactly a complex
    conjugation, bin by bin. It is involutive, and :func:`to_optics` and
    :func:`to_internal` are the only sanctioned place in solprop where it happens;
    no other module conjugates anything.

    Parameters
    ----------
    x : numpy.ndarray
        Array in the internal FFT convention, of any shape.

    Returns
    -------
    numpy.ndarray
        The same array in the optics convention.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import to_internal, to_optics
    >>> spectrum = np.array([1 + 2j, 3 - 1j])
    >>> bool(np.array_equal(to_internal(to_optics(spectrum)), spectrum))
    True
    """
    return np.conj(np.asarray(x))


def to_internal(x: Num[np.ndarray, "*shape"]) -> Num[np.ndarray, "*shape"]:
    """Convert an optics-convention array to the internal FFT convention.

    The exact inverse of :func:`to_optics`, and equal to it: both are a complex
    conjugation. Two names exist so that the direction of travel is readable at each of
    the three boundary sites, and so that the boundary can be found by name.

    Parameters
    ----------
    x : numpy.ndarray
        Array in the optics convention, of any shape.

    Returns
    -------
    numpy.ndarray
        The same array in the internal FFT convention.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import to_internal
    >>> bool(np.array_equal(to_internal(np.array([1 + 2j])), np.array([1 - 2j])))
    True
    """
    return np.conj(np.asarray(x))


def _origin_shift(n_omega: int) -> Float[np.ndarray, " n_omega"]:
    """Return the half-window phase-origin factor as an array of alternating signs.

    Parameters
    ----------
    n_omega : int
        Number of spectral bins.

    Returns
    -------
    numpy.ndarray
        The values ``(-1)**k`` for storage index ``k``, shape ``(n_omega,)``.
    """
    shift = np.ones(n_omega)
    shift[1::2] = -1.0
    return shift


def spectrum_to_optics(
    x_omega: Num[np.ndarray, "*batch n_omega"], *, kind: GridKind
) -> Num[np.ndarray, "*batch n_omega"]:
    """Convert an internal spectrum to the optics convention about the window centre.

    This is the **spectrum** form of the convention boundary and does two things at
    once: the conjugation of :func:`to_optics`, and the move of the phase origin from
    the first time sample to ``t = 0`` at the centre of the window,
    ``E_optics[k] = (-1)**k * conj(E[k])`` (conventions.md §2). The second step is what
    makes a user-facing spectral phase mean what a spectrometer means by it: a
    transform-limited pulse sitting at the middle of the window has a flat optics phase,
    where its stored phase alternates in sign from bin to bin.

    The factor is ``(-1)**k`` on the storage index for both layouts. On a field grid
    ``omega_k n_t dt/2 = pi k``; on an envelope grid the transform's own variable is the
    relative frequency, whose bin ``k`` gives ``pi k`` as well because ``n_t`` is even.
    So ``kind`` is not used to choose the factor, only to check the array against its
    claimed layout.

    Use :func:`to_optics` instead for time-domain arrays, and wherever only the sign of
    the convention matters. A Taylor phase about a reference frequency and a pure delay
    both commute with the origin shift, so they may be applied to a stored spectrum
    directly; an *absolute* spectral phase — measured data, a group delay read as an
    arrival time — must come through here.

    Parameters
    ----------
    x_omega : numpy.ndarray
        Stored spectrum (raw forward DFT), shape ``(..., n_omega)``.
    kind : {'field', 'envelope'}
        Spectral layout of ``x_omega``.

    Returns
    -------
    numpy.ndarray
        The same spectrum in the optics convention, referenced to the window centre,
        shape ``(..., n_omega)``.

    Raises
    ------
    GridError
        If ``kind`` is unknown or the trailing axis is not a valid length for it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import spectrum_to_internal, spectrum_to_optics
    >>> spectrum = np.array([1 + 1j, 2 - 1j, 3 + 2j])
    >>> spectrum_to_optics(spectrum, kind="field").tolist()
    [(1-1j), (-2-1j), (3-2j)]
    >>> back = spectrum_to_internal(spectrum_to_optics(spectrum, kind="field"),
    ...                             kind="field")
    >>> bool(np.array_equal(back, spectrum))
    True
    """
    values = np.asarray(x_omega)
    _check_spectrum(values, kind)
    return _origin_shift(values.shape[-1]) * to_optics(values)


def spectrum_to_internal(
    x_omega: Num[np.ndarray, "*batch n_omega"], *, kind: GridKind
) -> Num[np.ndarray, "*batch n_omega"]:
    """Convert an optics-convention spectrum to the internal stored form.

    The exact inverse of :func:`spectrum_to_optics`, and equal to it: both conjugate and
    both multiply by ``(-1)**k``, and each operation is its own inverse. Two names exist
    so that the direction of travel is readable at each boundary site, exactly as for
    :func:`to_optics` and :func:`to_internal`.

    Parameters
    ----------
    x_omega : numpy.ndarray
        Optics-convention spectrum referenced to ``t = 0``, shape ``(..., n_omega)``.
    kind : {'field', 'envelope'}
        Spectral layout of ``x_omega``.

    Returns
    -------
    numpy.ndarray
        The stored spectrum, with its phase origin back at the first time sample,
        shape ``(..., n_omega)``.

    Raises
    ------
    GridError
        If ``kind`` is unknown or the trailing axis is not a valid length for it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import inverse, spectrum_to_internal
    >>> flat = np.ones(5, dtype=complex)
    >>> samples = inverse(spectrum_to_internal(flat, kind="field"), kind="field")
    >>> int(np.argmax(samples))
    4
    """
    values = np.asarray(x_omega)
    _check_spectrum(values, kind)
    return _origin_shift(values.shape[-1]) * to_internal(values)


def _check_sizes(n_t: int, n_to: int) -> None:
    """Check a coarse/fine size pair, naming the offending quantity.

    Parameters
    ----------
    n_t : int
        Coarse number of time samples.
    n_to : int
        Fine number of time samples.

    Raises
    ------
    GridError
        If either size is not an even integer of at least 4, if ``n_to`` is smaller than
        ``n_t``, or if it is not an integer multiple of ``n_t``.
    """
    for name, value in (("n_t", n_t), ("n_to", n_to)):
        if value < MIN_N_T or value % 2:
            msg = f"{name} must be an even integer of at least {MIN_N_T}, got {value}"
            raise GridError(msg)
    if n_to < n_t:
        msg = f"n_to must be at least n_t, got n_to={n_to} and n_t={n_t}"
        raise GridError(msg)
    if n_to % n_t:
        msg = (
            f"n_to must be an integer multiple of n_t so that the coarse time samples "
            f"coincide with fine ones, got n_to={n_to} and n_t={n_t}"
        )
        raise GridError(msg)


def crop_index(n_t: int, n_to: int, *, kind: GridKind) -> np.ndarray:
    """Return the positions of the coarse spectral bins inside the fine spectrum.

    On a field grid the coarse bins are the leading ``n_t // 2 + 1`` bins of the fine
    one-sided spectrum. On an envelope grid the spectrum is in FFT order, so the coarse
    bins are the leading and the trailing ``n_t // 2`` bins: the positive and the
    negative relative frequencies respectively.
    Because ``n_to`` is an integer multiple of ``n_t``, the halves never overlap.

    Parameters
    ----------
    n_t : int
        Coarse number of time samples. Even and at least 4.
    n_to : int
        Fine number of time samples. Even, at least ``n_t``, and an integer multiple of
        it, so that the coarse time samples coincide with fine ones.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        Index array of ``numpy.intp`` and length ``n_omega``, giving for each coarse bin
        its position in the fine spectrum.

    Raises
    ------
    GridError
        If ``kind`` is unknown or the sizes are inconsistent.

    Examples
    --------
    >>> from solprop.fourier import crop_index
    >>> crop_index(4, 8, kind="field").tolist()
    [0, 1, 2]
    >>> crop_index(4, 8, kind="envelope").tolist()
    [0, 1, 6, 7]
    """
    _check_sizes(n_t, n_to)
    n_omega = n_omega_for(n_t, kind)
    if kind == "field":
        return np.arange(n_omega, dtype=np.intp)
    half = n_t // 2
    return np.concatenate(
        (np.arange(half, dtype=np.intp), np.arange(n_to - half, n_to, dtype=np.intp))
    )


def pad_scale(n_t: int, n_to: int) -> float:
    """Return the factor applied when padding a coarse spectrum onto the fine grid.

    The two inverse transforms carry different ``1/N``, so a spectrum moved from the
    coarse grid to the fine one must be scaled by ``n_to / n_t`` for the two time-domain
    signals to agree at the coincident samples (conventions.md §3).

    Parameters
    ----------
    n_t : int
        Coarse number of time samples.
    n_to : int
        Fine number of time samples.

    Returns
    -------
    float
        The factor ``n_to / n_t``.

    Raises
    ------
    GridError
        If the sizes are inconsistent.

    Examples
    --------
    >>> from solprop.fourier import pad_scale
    >>> pad_scale(4096, 8192)
    2.0
    """
    _check_sizes(n_t, n_to)
    return n_to / n_t


def crop_scale(n_t: int, n_to: int) -> float:
    """Return the factor applied when cropping a fine spectrum back to the coarse grid.

    The exact inverse of :func:`pad_scale`, so that a pad followed by a crop is the
    identity on data band-limited to the coarse grid.

    Parameters
    ----------
    n_t : int
        Coarse number of time samples.
    n_to : int
        Fine number of time samples.

    Returns
    -------
    float
        The factor ``n_t / n_to``.

    Raises
    ------
    GridError
        If the sizes are inconsistent.

    Examples
    --------
    >>> from solprop.fourier import crop_scale, pad_scale
    >>> float(crop_scale(4096, 8192) * pad_scale(4096, 8192))
    1.0
    """
    _check_sizes(n_t, n_to)
    return n_t / n_to


def oversample(
    x_omega: Complex[np.ndarray, "*batch n_omega"], n_to: int, *, kind: GridKind
) -> Complex[np.ndarray, "*batch n_omega_fine"]:
    """Zero-pad a coarse spectrum onto the fine grid, with the scale factor applied.

    The fine grid resolves the harmonics a nonlinear response generates, so the response
    is evaluated there and the result cropped back. The factor :func:`pad_scale` makes
    the fine time samples equal the coarse ones wherever they coincide.

    On a field grid this preserves energy only when the coarse Nyquist bin is zero: that
    bin is self-conjugate on the coarse grid but becomes an ordinary interior bin on the
    fine one, so a non-zero value there gains a factor of two in amplitude. Every grid
    built by :mod:`solprop.grid` apodises it to zero, so the condition holds in use.

    Parameters
    ----------
    x_omega : numpy.ndarray
        Coarse spectrum, shape ``(..., n_omega)``.
    n_to : int
        Fine number of time samples.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        A new fine spectrum, shape ``(..., n_omega_fine)``, zero outside the coarse
        bins.

    Raises
    ------
    GridError
        If ``kind`` is unknown or the sizes are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import oversample
    >>> fine = oversample(np.ones(3, dtype=complex), 8, kind="field")
    >>> bool(np.allclose(fine, [2, 2, 2, 0, 0]))
    True
    """
    values = np.asarray(x_omega)
    n_t = _check_spectrum(values, kind)
    index = crop_index(n_t, n_to, kind=kind)
    n_omega_fine = n_omega_for(n_to, kind)
    fine = np.zeros((*values.shape[:-1], n_omega_fine), dtype=complex)
    fine[..., index] = values * pad_scale(n_t, n_to)
    return fine


def crop(
    x_omega_fine: Complex[np.ndarray, "*batch n_omega_fine"],
    n_t: int,
    *,
    kind: GridKind,
) -> Complex[np.ndarray, "*batch n_omega"]:
    """Crop a fine spectrum back to the coarse grid, with the scale factor applied.

    The inverse of :func:`oversample`, exact for data band-limited to the coarse grid.
    Content above the coarse band is discarded, which is the point: it is the aliasing
    the fine grid existed to keep out of the coarse one.

    Parameters
    ----------
    x_omega_fine : numpy.ndarray
        Fine spectrum, shape ``(..., n_omega_fine)``.
    n_t : int
        Coarse number of time samples.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        A new coarse spectrum, shape ``(..., n_omega)``.

    Raises
    ------
    GridError
        If ``kind`` is unknown or the sizes are inconsistent.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import crop, oversample
    >>> coarse = np.array([1 + 0j, 2 - 1j, 0.5 + 0j])
    >>> fine = oversample(coarse, 8, kind="field")
    >>> bool(np.allclose(crop(fine, 4, kind="field"), coarse))
    True
    """
    values = np.asarray(x_omega_fine)
    n_to = _check_spectrum(values, kind)
    index = crop_index(n_t, n_to, kind=kind)
    return values[..., index] * crop_scale(n_t, n_to)


def energy_t(
    x: Inexact[np.ndarray, "*batch n_t"], dt: float, *, kind: GridKind
) -> Float[np.ndarray, " *batch"]:
    """Return the pulse energy from time samples, in joules.

    One functional serves both grids, because the field grid's power is that of the
    analytic signal and the envelope grid's state *is* an analytic signal with the
    carrier removed (conventions.md §4): ``P(t) = |E_a(t)|²`` in watts for modal
    amplitudes in the square root of watts, and ``U = ∫ P dt``.

    The integral is the trapezoid rule, which on a periodic band-limited grid is
    spectrally accurate and reduces to ``dt`` times a plain sum. Simpson's rule would
    carry a fourth-order error that makes the time-domain and frequency-domain energies
    disagree with each other; here they agree to round-off.

    Parameters
    ----------
    x : numpy.ndarray
        Time samples, shape ``(..., n_t)``, in the square root of watts. Real for a
        field grid, complex for an envelope grid.
    dt : float
        Coarse sample spacing, s.
    kind : {'field', 'envelope'}
        Layout of ``x``; the dtype is checked against it.

    Returns
    -------
    numpy.ndarray
        Energy, J, with the time axis reduced away.

    Raises
    ------
    GridError
        If ``kind`` is unknown, or the dtype of ``x`` does not match it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import energy_t
    >>> float(energy_t(np.full(16, 2.0 + 0j), 0.25, kind="envelope"))
    16.0
    """
    values = np.asarray(x)
    _check_time_dtype(values, kind)
    n_omega_for(values.shape[-1], kind)
    envelope = values if kind == "envelope" else analytic_signal(values)
    # asarray keeps the contract uniform: a 1-D input reduces to a 0-d array rather than
    # to a NumPy scalar, so the return is always an ndarray of the batch shape.
    return np.asarray(dt * np.sum(np.abs(envelope) ** 2, axis=-1))


def energy_omega(
    x_omega: Complex[np.ndarray, "*batch n_omega"], dt: float, *, kind: GridKind
) -> Float[np.ndarray, " *batch"]:
    """Return the pulse energy from the stored spectrum, in joules.

    This is the exact discrete form of conventions.md §4's
    ``U = (2 δt²/π) ∫ |Ẽ|² dω`` (field) and ``U = (δt² δω / 2π) Σ |Ã|²`` (envelope
    grid), so it agrees with :func:`energy_t` to round-off rather than to a quadrature
    error. On the envelope grid it *is* §4's formula. On the field grid it differs from
    a trapezoid discretisation of §4 in two bins only: the analytic signal doubles every
    one-sided bin in amplitude, and therefore quadruples it in power, except the
    zero-frequency and Nyquist bins, which it leaves alone.

    Parameters
    ----------
    x_omega : numpy.ndarray
        Stored spectrum (raw forward DFT), shape ``(..., n_omega)``.
    dt : float
        Coarse sample spacing, s.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        Energy, J, with the frequency axis reduced away.

    Raises
    ------
    GridError
        If ``kind`` is unknown or the trailing axis is not a valid length for it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import energy_omega, energy_t, forward
    >>> field = np.random.default_rng(0).standard_normal(64)
    >>> u_t = energy_t(field, 1e-15, kind="field")
    >>> u_omega = energy_omega(forward(field, kind="field"), 1e-15, kind="field")
    >>> bool(np.isclose(u_t, u_omega, rtol=1e-13))
    True
    """
    values = np.asarray(x_omega)
    n_t = _check_spectrum(values, kind)
    power = np.abs(values) ** 2
    if kind == "envelope":
        return np.asarray((dt / n_t) * np.sum(power, axis=-1))
    # Weights [1, 4, 4, ..., 4, 1], not Simpson's [1, 4, 2, ..., 4, 1]/3: the 4 is the
    # analytic signal's amplitude doubling, which does not reach the zero-frequency and
    # Nyquist bins. With those weights the identity with energy_t is exact.
    interior = 4.0 * np.sum(power[..., 1:-1], axis=-1)
    return np.asarray((dt / n_t) * (power[..., 0] + interior + power[..., -1]))


def energy_weights(
    n_omega: int, dt: float, *, kind: GridKind
) -> Float[np.ndarray, " n_omega"]:
    """Return the weight of each stored bin's ``|X|^2`` in :func:`energy_omega`.

    ``energy_omega(x, dt) = sum(w * |x|^2)`` with these weights, so a sum over bins can
    be split into blocks of frequency and accumulated (SEP 22, the blockwise collar
    diagnostic), agreeing with :func:`energy_omega` to the rounding of the sum's order.

    Parameters
    ----------
    n_omega : int
        Number of stored bins.
    dt : float
        Coarse sample spacing, s.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        Weights in s per bin, shape ``(n_omega,)``: ``dt / n_t`` on the envelope grid,
        and ``[1, 4, ..., 4, 1] dt / n_t`` on the field grid.

    Raises
    ------
    GridError
        If ``kind`` is unknown or ``n_omega`` is not a valid length for it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.fourier import energy_omega, energy_weights, forward
    >>> spectrum = forward(np.random.default_rng(0).standard_normal(64), kind="field")
    >>> weights = energy_weights(33, 1e-15, kind="field")
    >>> weighted = np.sum(weights * np.abs(spectrum) ** 2)
    >>> bool(np.isclose(weighted, energy_omega(spectrum, 1e-15, kind="field")))
    True
    """
    n_t = _n_t_from_n_omega(n_omega, kind)
    if kind == "envelope":
        return np.full(n_omega, dt / n_t)
    weights = np.full(n_omega, 4.0 * dt / n_t)
    weights[0] = weights[-1] = dt / n_t
    return weights


def spectral_density(
    x_omega: Complex[np.ndarray, "*batch n_omega"], dt: float, *, kind: GridKind
) -> Float[np.ndarray, "*batch n_omega"]:
    """Return the spectral energy density of a stored spectrum, in J/(rad/s).

    ``S(ω) = dU/dω`` as fixed by conventions.md §4: ``(2 δt²/π)|Ẽ|²`` on the field grid
    and ``(δt²/2π)|Ã|²`` on the envelope grid. These are the prefactors that pair with
    the shot-noise amplitudes of §4, so that one photon per bin gives exactly
    ``S(ω_k) δω = ħ ω_k``. Convert to a density per wavelength with
    ``S(λ) = S(ω)·2πc/λ²``.

    On the field grid the zero-frequency and Nyquist bins are the one exception to
    ``δω Σ_k S_k = U``: the analytic signal does not double them, so summing this
    density over every bin overcounts those two, by ``3 δt (|Ẽ_0|² + |Ẽ_Nyq|²) / n_t``
    in total. Every interior bin agrees exactly, and both end bins are identically zero
    on any grid built by :mod:`solprop.grid`, whose spectral collar apodises them away.
    Use :func:`energy_omega` when an exact energy is wanted.

    Parameters
    ----------
    x_omega : numpy.ndarray
        Stored spectrum (raw forward DFT), shape ``(..., n_omega)``.
    dt : float
        Coarse sample spacing, s.
    kind : {'field', 'envelope'}
        Spectral layout.

    Returns
    -------
    numpy.ndarray
        Spectral energy density, J/(rad/s), of the same shape as ``x_omega``.

    Raises
    ------
    GridError
        If ``kind`` is unknown or the trailing axis is not a valid length for it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.constants import HBAR
    >>> from solprop.fourier import spectral_density
    >>> n_t, dt, bin_index = 64, 1e-16, 10
    >>> d_omega = 2 * np.pi / (n_t * dt)
    >>> omega = bin_index * d_omega
    >>> spectrum = np.zeros(n_t // 2 + 1, dtype=complex)
    >>> amplitude = np.sqrt(HBAR * omega / d_omega) * np.sqrt(2 * np.pi)
    >>> spectrum[bin_index] = amplitude / (2 * dt)
    >>> density = spectral_density(spectrum, dt, kind="field")
    >>> bool(np.isclose(density[bin_index] * d_omega / (HBAR * omega), 1.0))
    True
    """
    values = np.asarray(x_omega)
    _check_spectrum(values, kind)
    power = np.abs(values) ** 2
    if kind == "envelope":
        return dt**2 / (2.0 * np.pi) * power
    return 2.0 * dt**2 / np.pi * power
