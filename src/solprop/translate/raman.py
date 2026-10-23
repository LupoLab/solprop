# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Sampling the Raman response kernel onto the doubled grid a backend convolves on.

The physics layer (:mod:`solprop.nonlinear.raman`) says what the response *is*; this
module turns it into the arrays a kernel consumes. Three things happen here.

**The grid is doubled.** The convolution is evaluated as a product of transforms, and a
transform product is a *circular* convolution: without padding, the tail of the response
wraps back onto the head of the drive. A rotational coherence in hydrogen lives for
nanoseconds against a window of tens of picoseconds, so that wrapped tail is not a small
error. Writing the response and the drive into the first half of an array of twice the
length makes the product the linear convolution exactly. It costs one transform of
``2 n_to`` in place of one of ``n_to``.

**The tail is tapered, by the grid's own collar.** Whatever the window, a coherence that
outlives it has to be cut somewhere, and cutting it with a step puts a sinc ringing
across the whole kernel spectrum. The taper's falling edge has the width of the grid's
temporal collar, so the response is cut with the same profile, and over the same
fraction of the window, as everything else the grid absorbs. A coherence that
outlives the window is noted in the debug log. Padding prevents wraparound, but a
pump-probe delay entering the taper collar samples an attenuated response even when
both pulses lie inside the window. Extend the window until every relevant delay
precedes the collar, and check window convergence separately from sample spacing.

**The dephasing is applied per density.** This is the design decision the kernel
contract needs, and it is set out in ``docs/guide/media/raman-numerics.md``: a single
``raman_kernel``
array cannot be right, because the dephasing depends on the density, which varies
along a pressure gradient, and a molecular gas has *two* dephasing times -- its
rotational and vibrational coherences decay 3.5 times (H2) to 17 times (N2) apart at one
bar. So the expensive part, the undamped model responses, is sampled once per grid as
an ``(n_terms, 2 n_to)`` array (the contract's ``raman_kernel_undamped``), the cheap
part is one exponential per term at a decay rate that is a scalar path along ``z``
(``raman_decay_rates``, from :func:`decay_rate_profiles`, so that the contract carries
paths and arrays and no physics object), and the damped kernel spectrum is recomputed
and memoised whenever the rate changes: one real transform of length ``2 n_to`` per
change, against ``n_modes n_pol`` transforms of length ``n_to`` per stage.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from jaxtyping import Complex, Float, Num

from solprop.exceptions import TranslationError
from solprop.grid import TimeGrid, planck_taper
from solprop.nonlinear.raman import Dephasing, RamanModel, RamanResponse
from solprop.profiles import FromCallable, ZProfile

__all__ = [
    "MAX_DECAY_EXPONENT",
    "WINDOW_RESIDUAL_TOLERANCE",
    "RamanKernels",
    "build_raman_kernels",
    "decay_rate_profiles",
    "kernel_spectrum",
    "kernel_taper",
    "sample_kernel",
    "undamped_kernels",
]

logger = logging.getLogger(__name__)

WINDOW_RESIDUAL_TOLERANCE: Final = 1e-3
"""Amplitude a coherence may still have where the taper starts, before it is noted.

The taper multiplies whatever is left at that delay down to zero over the collar's
width. A coherence damped to a thousandth of its initial amplitude by then is cut
with a small absolute change; one that is still at a tenth outlives the window.
The taper changes responses at delays within its collar, and the kernel spectrum
resolves that line only to the window's own width. The debug log reports the residual
once per density; it does not certify window convergence.
"""

MAX_DECAY_EXPONENT: Final = 708.0
"""Largest ``rate * dt_fine`` a decay-rate path carries.

``exp(-708)`` is the last power of e above the smallest normal double, so a rate at
this cap damps a coherence to nothing within one sample. It is the finite stand-in for
the infinite rate of a Dicke-narrowed line at zero density, where the polarisation is
zero anyway; the same device as the limit of :class:`solprop.linear.LossPolicy`.
"""


def kernel_taper(grid: TimeGrid) -> Float[np.ndarray, " n_to"]:
    """Return the taper applied to the tail of the sampled response.

    A Planck taper (:func:`solprop.grid.planck_taper`) that is one from zero delay until
    the last ``t_collar.width_high`` of the response's span, then falls smoothly to zero
    at the final sample. The rising edge is placed below zero delay, so the taper is
    exactly one at ``t = 0``: the response must not be attenuated where it is strongest.

    Using the grid's own collar width rather than a fixed fraction of the window keeps
    one statement of "how much of the window is boundary" instead of two that can
    disagree.

    Parameters
    ----------
    grid : TimeGrid
        The grid whose fine time axis and temporal collar the kernel follows.

    Returns
    -------
    numpy.ndarray
        Taper values in ``[0, 1]``, shape ``(n_to,)``, one at the start and zero at the
        end.

    Raises
    ------
    TranslationError
        If the collar's falling edge is not narrower than the response's own span, so
        that there is no untapered region left.

    Examples
    --------
    >>> from solprop.grid import FieldGrid
    >>> from solprop.translate.raman import kernel_taper
    >>> grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 2e-12)
    >>> taper = kernel_taper(grid)
    >>> float(taper[0]), float(taper[-1])
    (1.0, 0.0)
    """
    delays = np.arange(grid.n_to) * grid.dt_fine
    span = float(delays[-1])
    width = grid.t_collar.width_high
    if not 0.0 < width < span:
        raise TranslationError(
            f"the temporal collar's falling edge is {width} s, which is not a taper "
            f"inside a response span of {span} s; widen the time window or narrow the "
            "collar"
        )
    # The rising edge sits entirely at negative delay, so the taper is one from the
    # first sample; planck_taper needs a non-degenerate edge there whatever the values.
    return planck_taper(delays, -2.0 * grid.dt_fine, -grid.dt_fine, span - width, span)


def undamped_kernels(
    response: RamanResponse, grid: TimeGrid
) -> Float[np.ndarray, "n_terms n_to_doubled"]:
    """Return each model's undamped response, causal and tapered, on the doubled grid.

    This is the expensive half of the kernel and the half that does not move with the
    density, so it is sampled once per grid and reused. The rows are the contract's
    ``raman_kernel_undamped``; :func:`decay_rate_profiles` gives the matching
    ``raman_decay_rates``.

    The overall fraction is **not** folded in here: it is applied once to the sum in
    :func:`sample_kernel`, so that the arithmetic matches
    :meth:`solprop.nonlinear.raman.RamanResponse.impulse` operation for operation.

    Parameters
    ----------
    response : RamanResponse
        The response whose models are sampled.
    grid : TimeGrid
        The grid supplying the fine time axis and the temporal collar.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_terms, 2 n_to)``, m^5 V^-2 s^-1. The first ``n_to`` columns hold the
        response from zero delay; the rest is the zero padding the linear convolution
        needs.

    Raises
    ------
    TranslationError
        If the collar leaves no untapered region.

    Examples
    --------
    >>> from solprop.grid import FieldGrid
    >>> from solprop.nonlinear.raman import response_for
    >>> from solprop.translate.raman import undamped_kernels
    >>> grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 2e-12)
    >>> kernels = undamped_kernels(response_for("SiO2"), grid)
    >>> kernels.shape == (1, 2 * grid.n_to)
    True
    """
    n_to = grid.n_to
    delays = np.arange(n_to) * grid.dt_fine
    taper = kernel_taper(grid)
    kernels = np.zeros((len(response.models), 2 * n_to))
    for row, model in enumerate(response.models):
        kernels[row, :n_to] = model.undamped(delays) * taper
    return kernels


def sample_kernel(
    response: RamanResponse, grid: TimeGrid, density: float
) -> Float[np.ndarray, " n_to_doubled"]:
    """Return the causal, tapered, damped response kernel at one density.

    On the untapered part this is bit for bit
    :meth:`solprop.nonlinear.raman.RamanResponse.impulse`, because the taper is exactly
    one there and the terms are summed in the same order.

    Parameters
    ----------
    response : RamanResponse
        The response to sample.
    grid : TimeGrid
        The grid supplying the fine time axis and the temporal collar.
    density : float
        Number density in m^-3, non-negative.

    Returns
    -------
    numpy.ndarray
        Shape ``(2 n_to,)``, m^5 V^-2 s^-1, zero in its second half.

    Raises
    ------
    TranslationError
        If the collar leaves no untapered region.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import FieldGrid
    >>> from solprop.nonlinear.raman import response_for
    >>> from solprop.translate.raman import sample_kernel
    >>> grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 2e-12)
    >>> kernel = sample_kernel(response_for("SiO2"), grid, 1.0)
    >>> bool(np.all(kernel[grid.n_to :] == 0.0))
    True
    """
    kernels = undamped_kernels(response, grid)
    return _damped(response, kernels, grid, density, check=True)


def _damped(
    response: RamanResponse,
    kernels: Float[np.ndarray, "n_terms n_to_doubled"],
    grid: TimeGrid,
    density: float,
    *,
    check: bool,
) -> Float[np.ndarray, " n_to_doubled"]:
    """Apply each term's dephasing to its undamped kernel and sum.

    Parameters
    ----------
    response : RamanResponse
        The response the kernels came from; supplies the dephasings and the fraction.
    kernels : numpy.ndarray
        The undamped, tapered rows, shape ``(n_terms, 2 n_to)``.
    grid : TimeGrid
        The grid supplying the fine time axis and the temporal collar.
    density : float
        Number density in m^-3, non-negative.
    check : bool
        Whether to check each damped term against the window it has to fit in, and
        note in the debug log the ones that outlive it.

    Returns
    -------
    numpy.ndarray
        The damped kernel, shape ``(2 n_to,)``.
    """
    n_to = grid.n_to
    delays = np.arange(n_to) * grid.dt_fine
    flat = _flat_samples(grid)
    total = np.zeros(2 * n_to)
    # At zero density the polarisation is identically zero whatever the kernel says,
    # and a purely collisional linewidth vanishes there, so every such term would be
    # noted for a truncation that changes nothing.
    checking = check and density > 0.0
    for row, model in enumerate(response.models):
        damped = kernels[row, :n_to] * model.dephasing.damping(delays, density)
        if checking:
            _check_window(response, model, damped[flat:], damped, density, row)
        total[:n_to] += damped
    # The fraction multiplies the sum, not each term, so that this matches
    # RamanResponse.impulse operation for operation and the two agree to the last bit.
    return response.scale * total


def _flat_samples(grid: TimeGrid) -> int:
    """Return the number of kernel samples the taper leaves alone.

    Parameters
    ----------
    grid : TimeGrid
        The grid supplying the fine time axis and the temporal collar.

    Returns
    -------
    int
        The index at which the taper starts to fall.
    """
    span = (grid.n_to - 1) * grid.dt_fine
    return int(
        np.searchsorted(
            np.arange(grid.n_to) * grid.dt_fine, span - grid.t_collar.width_high
        )
    )


def _check_window(
    response: RamanResponse,
    model: RamanModel,
    tail: Float[np.ndarray, " n_tail"],
    damped: Float[np.ndarray, " n_to"],
    density: float,
    row: int,
) -> None:
    """Note in the debug log a term that still has amplitude where the taper starts.

    A log entry rather than a warning, because nothing is wrong with the polarisation:
    the drive is zero outside the window, so a coherence that outlives it is only
    unresolved in the kernel spectrum, never mis-convolved. It would also fire for every
    molecular gas on every window shorter than its rotational coherence, which is every
    realistic window.

    The measure is the damped kernel itself rather than the dephasing time alone,
    because a glass carries its decay inside the model and its density-dependent
    dephasing is none: judging by the dephasing time would note every glass and
    nothing else. Reading it off the tapered row understates the residual slightly,
    since the taper has already begun; that makes the test conservative, and the taper
    is one at the sample the tail starts from.

    Parameters
    ----------
    response : RamanResponse
        The response, for the number of terms in the message.
    model : RamanModel
        The term itself, for its dephasing time.
    tail : numpy.ndarray
        The damped kernel over the tapered region, shape ``(n_tail,)``.
    damped : numpy.ndarray
        The damped kernel over the whole response span, shape ``(n_to,)``.
    density : float
        Number density in m^-3, non-negative.
    row : int
        Index of the term, for the message.
    """
    peak = float(np.max(np.abs(damped)))
    if peak == 0.0 or tail.size == 0:
        return
    residual = float(np.max(np.abs(tail))) / peak
    if residual <= WINDOW_RESIDUAL_TOLERANCE:
        return
    dephasing_time = model.dephasing.dephasing_time(density)
    needed = -dephasing_time * math.log(WINDOW_RESIDUAL_TOLERANCE)
    logger.debug(
        "Raman term %d of %d still has %.3g of its amplitude where the taper starts: "
        "its dephasing time is %.3g s at %.3g m^-3, longer than the untapered span. "
        "The polarisation inside the window is unaffected; the kernel spectrum "
        "resolves that line only to the window's width. An untapered span of %.3g s "
        "would hold it.",
        row + 1,
        len(response.models),
        residual,
        dephasing_time,
        density,
        needed,
    )


def kernel_spectrum(
    kernel: Num[np.ndarray, " n_to_doubled"], dt: float
) -> Complex[np.ndarray, " n_omega_doubled"]:
    """Return the kernel spectrum with the convolution's normalisation folded in.

    A backend then computes the convolved drive as
    ``irfft(kernel_spectrum * rfft(drive_padded))``: the sample spacing that turns the
    discrete sum into an integral is already here, and the inverse transform's ``1/N``
    supplies the rest. Folding it into the kernel rather than multiplying by ``dt``
    after the product means it is applied once per density change instead of once per
    column.

    The spectrum is in the **internal** FFT convention (``docs/dev/conventions.md``
    section 2) and must not be passed through :func:`solprop.fourier.to_optics`: the
    sign of its imaginary part is what distinguishes Raman gain from Raman loss, and
    conjugating it would silently swap the Stokes and anti-Stokes sides.

    Parameters
    ----------
    kernel : numpy.ndarray
        The sampled kernel on the doubled grid, shape ``(2 n_to,)``, real.
    dt : float
        Sample spacing of the fine time grid, s, positive.

    Returns
    -------
    numpy.ndarray
        The one-sided spectrum, shape ``(n_to + 1,)``, complex.

    Raises
    ------
    TranslationError
        If the sample spacing is not positive and finite, or the kernel is not real, or
        its length is odd so that it is not a doubled grid.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.translate.raman import kernel_spectrum
    >>> kernel = np.zeros(8)
    >>> kernel[1] = 1.0
    >>> kernel_spectrum(kernel, 2.0).shape
    (5,)
    """
    values = np.asarray(kernel)
    if not (dt > 0.0 and math.isfinite(dt)):
        raise TranslationError(
            f"a Raman kernel needs a positive, finite sample spacing in s, got {dt!r}"
        )
    if np.iscomplexobj(values) or values.size % 2:
        raise TranslationError(
            f"a Raman kernel is a real array of even length on the doubled grid, got "
            f"{values.size} samples of dtype {values.dtype}"
        )
    return np.fft.rfft(values) * dt


@dataclass
class _SpectrumMemo:
    """One damped kernel spectrum, remembered with the density it was built at.

    Mutable and private, held by a frozen :class:`RamanKernels`: the cache is not part
    of what defines a kernel, so two kernels that have transformed a different number
    of times are still the same kernel and compare equal.

    Attributes
    ----------
    density : float or None
        The density the stored spectrum was built at, or ``None`` when empty.
    spectrum : numpy.ndarray or None
        The stored spectrum, shape ``(n_omega_doubled,)``.
    transforms : int
        How many transforms this kernel has done, for diagnostics and tests.
    """

    density: float | None = None
    spectrum: Complex[np.ndarray, " n_omega_doubled"] | None = None
    transforms: int = 0


@dataclass(frozen=True)
class RamanKernels:
    """The Raman kernel of one response on one grid, with its spectrum cached on rho.

    The two arrays the kernel contract needs, kept together with the transform that
    joins them. :attr:`undamped` is sampled once; :meth:`spectrum` applies the
    dephasings, transforms, and remembers the result, so that a propagation at constant
    density transforms the kernel once and one along a pressure gradient transforms it
    once per density the stepper asks for.

    Only the most recent density is remembered, which is the right behaviour here: a
    gradient presents a new density at every stage, so a growing cache would hold
    thousands of spectra and hit none of them.

    Build one with :func:`build_raman_kernels`.

    Attributes
    ----------
    undamped : numpy.ndarray
        Each model's undamped, causal, tapered response, shape ``(n_terms, 2 n_to)``,
        m^5 V^-2 s^-1. The contract's ``raman_kernel_undamped``.
    response : RamanResponse
        The response the kernels came from; supplies the dephasings and the fraction.
    grid : TimeGrid
        The grid they were sampled on.

    Raises
    ------
    TranslationError
        If the kernel array does not match the response and the grid.

    Examples
    --------
    >>> from solprop.grid import FieldGrid
    >>> from solprop.nonlinear.raman import response_for
    >>> from solprop.translate.raman import build_raman_kernels
    >>> grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 2e-12)
    >>> kernels = build_raman_kernels(response_for("SiO2"), grid)
    >>> kernels.n_terms
    1
    >>> kernels.spectrum(1.0).shape == (grid.n_to + 1,)
    True
    >>> kernels.recomputations
    1
    """

    undamped: Float[np.ndarray, "n_terms n_to_doubled"]
    response: RamanResponse
    grid: TimeGrid
    _memo: _SpectrumMemo = field(
        default_factory=_SpectrumMemo, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Check that the kernel array matches the response and the grid.

        Raises
        ------
        TranslationError
            If the number of rows is not the number of models, or their length is not
            twice the grid's fine size.
        """
        rows, columns = self.undamped.shape
        if rows != len(self.response.models):
            raise TranslationError(
                f"the kernel has {rows} rows but the response has "
                f"{len(self.response.models)} models"
            )
        if columns != 2 * self.grid.n_to:
            raise TranslationError(
                f"the kernel is {columns} samples long but the doubled fine grid is "
                f"{2 * self.grid.n_to}"
            )

    @property
    def n_terms(self) -> int:
        """Return the number of independently damped terms.

        Returns
        -------
        int
            One for a glass or a vibration-only gas, two for a gas with both a
            rotational and a vibrational coherence.
        """
        return int(self.undamped.shape[0])

    @property
    def dephasings(self) -> tuple[Dephasing, ...]:
        """Return the dephasing of each term, in the order of :attr:`undamped`.

        Returns
        -------
        tuple of Dephasing
            One entry per row of the kernel; :meth:`decay_rates` turns them into the
            paths the contract carries.
        """
        return tuple(model.dephasing for model in self.response.models)

    def decay_rates(self, density: ZProfile) -> tuple[ZProfile, ...]:
        """Return the decay rate of each term along ``z``, for a density path.

        Parameters
        ----------
        density : solprop.profiles.ZProfile
            The number density along the waveguide, m^-3.

        Returns
        -------
        tuple of solprop.profiles.ZProfile
            The contract's ``raman_decay_rates``, one path per row of the kernel;
            see :func:`decay_rate_profiles`.

        Examples
        --------
        >>> from solprop.grid import FieldGrid
        >>> from solprop.nonlinear.raman import response_for
        >>> from solprop.physdata import Gas, MaterialSelection
        >>> from solprop.profiles import Constant
        >>> from solprop.translate.raman import build_raman_kernels
        >>> grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 2e-12)
        >>> selection = MaterialSelection(carrier_wavelength=800e-9)
        >>> response = response_for(Gas.N2, selection=selection)
        >>> kernels = build_raman_kernels(response, grid)
        >>> len(kernels.decay_rates(Constant(2.5e25)))
        72
        """
        return decay_rate_profiles(self.response, density, self.grid)

    @property
    def recomputations(self) -> int:
        """Return how many times the spectrum has been transformed.

        Returns
        -------
        int
            One per distinct density asked for in succession; repeated calls at the
            same density do not add to it.
        """
        return self._memo.transforms

    def spectrum(self, density: float) -> Complex[np.ndarray, " n_omega_doubled"]:
        """Return the damped kernel spectrum at a density, transforming if it changed.

        Parameters
        ----------
        density : float
            Number density in m^-3, non-negative.

        Returns
        -------
        numpy.ndarray
            The one-sided spectrum with the ``dt`` scaling folded in, shape
            ``(n_to + 1,)``, complex. Read-only, because it is the cached array itself.

        Raises
        ------
        MaterialError
            If the density is negative.

        Examples
        --------
        >>> from solprop.grid import FieldGrid
        >>> from solprop.nonlinear.raman import response_for
        >>> from solprop.translate.raman import build_raman_kernels
        >>> grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 2e-12)
        >>> kernels = build_raman_kernels(response_for("SiO2"), grid)
        >>> first = kernels.spectrum(1.0)
        >>> second = kernels.spectrum(1.0)
        >>> (first is second, kernels.recomputations)
        (True, 1)
        """
        cached = self._memo.spectrum
        if cached is not None and self._memo.density == density:
            return cached
        # The truncation note is logged here only for a density not seen last, so a
        # propagation at constant density notes it once rather than once per stage.
        kernel = _damped(self.response, self.undamped, self.grid, density, check=True)
        spectrum = kernel_spectrum(kernel, self.grid.dt_fine)
        spectrum.setflags(write=False)
        self._memo.density = density
        self._memo.spectrum = spectrum
        self._memo.transforms += 1
        return spectrum


def decay_rate_profiles(
    response: RamanResponse, density: ZProfile, grid: TimeGrid
) -> tuple[ZProfile, ...]:
    """Return the decay rate ``1/tau2`` of each term as a path along ``z``.

    The contract's ``raman_decay_rates``: the translation layer composes each term's
    :class:`solprop.nonlinear.raman.Dephasing` with the density path, so that what
    reaches a backend is a scalar path and an array per term, and no physics object.
    A backend damps row ``k`` of ``raman_kernel_undamped`` by ``exp(-rate_k(z) t)``.

    The paths are finite everywhere. A Dicke-narrowed line has an infinite rate at
    zero density; the path carries :data:`MAX_DECAY_EXPONENT` divided by the fine
    sample spacing instead, which damps the term to nothing within one sample -- the
    same answer :meth:`solprop.nonlinear.raman.Dephasing.damping` gives, and one the
    density multiplies by zero in any case. The breakpoints and the span are the
    density path's own: the rate can kink only where the density does.

    Parameters
    ----------
    response : RamanResponse
        The response whose terms are damped.
    density : solprop.profiles.ZProfile
        The number density along the waveguide, m^-3.
    grid : TimeGrid
        The grid whose fine sample spacing sets the cap.

    Returns
    -------
    tuple of solprop.profiles.ZProfile
        One path per term, in the order of the response's models, in s^-1.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import FieldGrid
    >>> from solprop.nonlinear.raman import response_for
    >>> from solprop.profiles import Constant
    >>> from solprop.translate.raman import decay_rate_profiles
    >>> grid = FieldGrid.from_limits(0.1, 800e-9, (200e-9, 4e-6), 2e-12)
    >>> (path,) = decay_rate_profiles(response_for("SiO2"), Constant(1.0), grid)
    >>> path(np.array([0.0, 0.05])).tolist()
    [0.0, 0.0]
    """
    cap = MAX_DECAY_EXPONENT / grid.dt_fine
    kinks = np.asarray(density.breakpoints, dtype=float)
    paths: list[ZProfile] = []
    for model in response.models:
        dephasing = model.dephasing

        def rate(
            z: Float[np.ndarray, " n_z"], dephasing: Dephasing = dephasing
        ) -> Float[np.ndarray, " n_z"]:
            """Return the capped decay rate at each position.

            Parameters
            ----------
            z : numpy.ndarray
                Positions in m, shape ``(n_z,)``.
            dephasing : Dephasing
                The term's dephasing, bound at definition time.

            Returns
            -------
            numpy.ndarray
                ``min(1/tau2(rho(z)), cap)`` in s^-1, shape ``(n_z,)``.
            """
            values = density(np.asarray(z, dtype=float))
            return np.array(
                [min(dephasing.decay_rate(float(rho)), cap) for rho in values]
            )

        paths.append(FromCallable(rate, density.z_span, kinks))
    return tuple(paths)


def build_raman_kernels(response: RamanResponse, grid: TimeGrid) -> RamanKernels:
    """Sample a Raman response onto a grid, ready for a backend to convolve with.

    Parameters
    ----------
    response : RamanResponse
        The response to sample.
    grid : TimeGrid
        The grid supplying the fine time axis and the temporal collar.

    Returns
    -------
    RamanKernels
        The undamped rows, the dephasings and the density-cached spectrum.

    Raises
    ------
    TranslationError
        If the collar leaves no untapered region.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.raman import response_for
    >>> from solprop.physdata import Gas, MaterialSelection
    >>> from solprop.translate.raman import build_raman_kernels
    >>> grid = EnvelopeGrid.from_limits(0.1, 800e-9, (400e-9, 2e-6), 2e-12)
    >>> selection = MaterialSelection(carrier_wavelength=800e-9)
    >>> response = response_for(Gas.N2, selection=selection)
    >>> build_raman_kernels(response, grid).n_terms
    72
    """
    kernels = undamped_kernels(response, grid)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Raman kernel: %d term(s) on %d doubled samples, dt %.3g s, taper over "
            "the last %.3g s",
            kernels.shape[0],
            kernels.shape[1],
            grid.dt_fine,
            grid.t_collar.width_high,
        )
    return RamanKernels(undamped=kernels, response=response, grid=grid)
