# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Grid data, masks, absorber rates and the noise field, in the kernel's layout.

Everything a backend needs to know about the frequency axis and the time window, read
from the grid and the absorber rather than recomputed: the two band masks, the spectral
and temporal windows, the oversampling crop index and its two scale factors, the
temporal absorber rates, and the shot-noise realisation in the canonical
state layout.

Section 2 of ``docs/dev/kernel-contract.md`` is the class :class:`Grids`, field for
field; the temporal rates belong to section 5 and the noise field to section 6, so those
are functions here and are stored by :mod:`solprop.translate.responses` and
:mod:`solprop.translate.problem`.

Nothing in this module computes physics. Every value is taken from
:class:`solprop.grid.TimeGrid`, :class:`solprop.boundaries.Absorber` or
:class:`solprop.noise.NoiseRealisation`, so a mask can never disagree with the grid that
produced it.
"""

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Complex, Float, Int

from solprop.boundaries import Absorber
from solprop.exceptions import TranslationError
from solprop.grid import TimeGrid
from solprop.noise import NoiseRealisation
from solprop.nonlinear.responses import thg_carrier

__all__ = [
    "DEFAULT_ABSORBER_SCHEME",
    "AbsorberScheme",
    "Grids",
    "build_masks",
    "noise_spectrum",
    "temporal_rates",
]

logger = logging.getLogger(__name__)

AbsorberScheme = Literal["split_step", "nonlinear"]
"""How the temporal absorber is applied (see ``docs/dev/decisions.md``).

``"split_step"`` multiplies the modal time-domain field by ``exp(-alpha(t) h / 2)`` once
per accepted step; ``"nonlinear"`` adds ``-alpha(t) E(t) / 2`` inside every stage
evaluation.

Both are exact *as a rate* in the sense that the factors telescope, but they are not
equally accurate, and a comparison on a soliton launched into its own temporal collar
measured the difference: the
split factor is a **Lie splitting** of a term that does not commute with the linear
flow, so it caps the observed order of the whole scheme at **one** wherever the collar
does any work, and the energy it removes drifts with the step count at the same order.
The in-stage term keeps the scheme's own order and its absorbed energy is step-count
independent to twelve digits. Because the split factor degrades the observed order,
the in-stage term is the default whatever it costs.
"""

DEFAULT_ABSORBER_SCHEME: AbsorberScheme = "nonlinear"
"""The default scheme, chosen on accuracy (see :data:`AbsorberScheme`)."""


@dataclass(frozen=True, eq=False)
class Grids:
    """The frequency axis, the masks and the oversampling data a kernel needs.

    Section 2 of the kernel contract. The arrays are read-only, C-contiguous copies, so
    a backend may hold them for the life of a run without defensive copying. Equality is
    by identity: comparing whole axes elementwise is never what a caller means.

    The shape annotations give each array its own axis name, because
    ``__post_init__`` is what enforces their agreement and that check must stay
    reachable in production, where no runtime type checker runs.

    Attributes
    ----------
    omega : numpy.ndarray
        Coarse angular frequency axis, rad/s, shape ``(n_omega,)``. Non-negative and
        increasing on a field grid, in FFT order about ``omega0`` on an envelope grid.
    omega0 : float
        Reference angular frequency, rad/s.
    dt : float
        Coarse sample spacing, s.
    dt_fine : float
        Fine sample spacing, s.
    n_t : int
        Coarse number of time samples.
    n_to : int
        Fine number of time samples.
    ffac : float
        Oversampling factor of the fine grid.
    crop_index : numpy.ndarray
        Position of each coarse bin in the fine spectrum, ``numpy.intp``, shape
        ``(n_omega,)``.
    pad_scale : float
        Factor applied when padding the coarse spectrum out, ``n_to / n_t``.
    crop_scale : float
        Factor applied when cropping back, ``n_t/n_to``.
    inband_mask : numpy.ndarray
        Boolean, shape ``(n_omega,)``: the bins inside the requested wavelength limits.
    active_mask : numpy.ndarray
        Boolean, shape ``(n_omega,)``: ``omega_min/2 < omega < 1.1 omega_max``. The
        linear operator and the prefactor are zero outside it and the
        hard band mask removes the state there once per accepted step.
    norm_mask : numpy.ndarray
        Boolean, shape ``(n_omega,)``: the bins the error norm scores, the
        controller's ``region`` -- ``active_mask`` by default, or ``inband_mask``.
    omega_window : numpy.ndarray
        The taper applied to the nonlinear polarisation in the spectral collar, shape
        ``(n_omega,)``: the grid's collar profile in ``[0, 1]`` when the polarisation
        taper is on, and all ones when it is off, which is the default. The rate
        absorber on the field is inside the linear operator either way; this taper is
        an option, off by default, because it sets a floor of the size of the field's
        content in the collar.
    t_window_fine : numpy.ndarray
        Temporal collar profile in ``[0, 1]``, shape ``(n_to,)``, applied inside the
        response pass.
    t_window_fine_derivative : numpy.ndarray
        Analytic time derivative of the fine collar, in s^-1, shape ``(n_to,)``.
        The field plasma uses it to hold the current at the collar (SEP 27).
    carrier_2w0 : numpy.ndarray or None
        ``exp(2 i omega_0 t)`` on the fine time axis, complex128, shape ``(n_to,)``,
        present exactly when a third-harmonic Kerr term is compiled on an envelope grid
        and ``None`` otherwise. That term sits an offset ``2 omega_0`` from the carrier
        the envelope is written against, so the response pass needs the factor; it is
        built here from :func:`solprop.nonlinear.responses.thg_carrier` and the grid's
        own fine time axis, because a backend that built it from ``omega0``, ``n_to``
        and ``dt_fine`` would be carrying a piece of physics
        (``docs/dev/architecture.md`` section 3).

    Raises
    ------
    TranslationError
        If an array length disagrees with ``n_omega`` or ``n_to``, or a size or scale is
        not positive.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.translate.masks import build_masks
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> masks = build_masks(grid)
    >>> masks.omega.shape == (grid.n_omega,), masks.pad_scale
    (True, 1.0)
    """

    omega: Float[np.ndarray, " n_omega"]
    omega0: float
    dt: float
    dt_fine: float
    n_t: int
    n_to: int
    ffac: float
    crop_index: Int[np.ndarray, " n_crop"]
    pad_scale: float
    crop_scale: float
    inband_mask: Bool[np.ndarray, " n_inband"]
    active_mask: Bool[np.ndarray, " n_active"]
    norm_mask: Bool[np.ndarray, " n_norm"]
    omega_window: Float[np.ndarray, " n_window"]
    t_window_fine: Float[np.ndarray, " n_to_window"]
    t_window_fine_derivative: Float[np.ndarray, " n_to_derivative"]
    carrier_2w0: Complex[np.ndarray, " n_to_carrier"] | None = None

    def __post_init__(self) -> None:
        """Adopt read-only copies and check the sizes the annotations cannot.

        Raises
        ------
        TranslationError
            Naming the offending field.
        """
        for name, dtype in (
            ("omega", np.float64),
            ("crop_index", np.intp),
            ("inband_mask", np.bool_),
            ("active_mask", np.bool_),
            ("norm_mask", np.bool_),
            ("omega_window", np.float64),
            ("t_window_fine", np.float64),
            ("t_window_fine_derivative", np.float64),
        ):
            values = np.ascontiguousarray(getattr(self, name), dtype=dtype)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.carrier_2w0 is not None:
            carrier = np.ascontiguousarray(self.carrier_2w0, dtype=np.complex128)
            carrier.setflags(write=False)
            object.__setattr__(self, "carrier_2w0", carrier)
        n_omega = self.omega.size
        for name in (
            "crop_index",
            "inband_mask",
            "active_mask",
            "norm_mask",
            "omega_window",
        ):
            if getattr(self, name).size != n_omega:
                msg = (
                    f"{name} has {getattr(self, name).size} entries, expected "
                    f"{n_omega} to match omega"
                )
                raise TranslationError(msg)
        for name in ("t_window_fine", "t_window_fine_derivative", "carrier_2w0"):
            fine = getattr(self, name)
            if fine is not None and fine.size != self.n_to:
                msg = (
                    f"{name} has {fine.size} entries, expected {self.n_to} to match "
                    "n_to"
                )
                raise TranslationError(msg)
        if min(self.n_t, self.n_to) < 1 or self.n_to % self.n_t:
            msg = (
                f"n_to ({self.n_to}) must be a positive multiple of n_t ({self.n_t}), "
                "so that the coarse time samples coincide with fine ones"
            )
            raise TranslationError(msg)
        if not (self.pad_scale > 0.0 and self.crop_scale > 0.0 and self.ffac >= 1.0):
            msg = (
                f"pad_scale ({self.pad_scale!r}), crop_scale ({self.crop_scale!r}) and "
                f"ffac ({self.ffac!r}) must be positive"
            )
            raise TranslationError(msg)

    @property
    def n_omega(self) -> int:
        """Return the number of coarse spectral bins.

        Returns
        -------
        int
            The length of ``omega``.

        Examples
        --------
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.translate.masks import build_masks
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
        >>> build_masks(grid).n_omega == grid.n_omega
        True
        """
        return int(self.omega.size)


def build_masks(
    grid: TimeGrid,
    *,
    polarisation_taper: bool = False,
    harmonic_carrier: bool = False,
    region: str = "active",
) -> Grids:
    """Collect a grid's axis, masks and oversampling data for the kernel contract.

    Every value is read from `grid`; nothing is recomputed, so the seam and the grid can
    never disagree. The one choice made here is whether the spectral collar tapers the
    nonlinear polarisation: off by default, because the rate absorber already acts on
    the field and the taper costs a systematic error of the size of the field's content
    in the collar (2.3e-6 on the recurrence of an N = 6 soliton against 2.7e-7
    without it).

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The field or envelope grid the run propagates on.
    polarisation_taper : bool, optional
        Whether ``omega_window`` carries the grid's collar profile or all ones.
        Default ``False``.
    harmonic_carrier : bool, optional
        Whether to carry ``exp(2 i omega_0 t)`` on the fine time axis, which an
        envelope grid's third-harmonic Kerr term needs and nothing else does. Default
        ``False``. :func:`solprop.translate.problem.assemble` passes the compiled Kerr
        term's own answer, so the field is present exactly when the term is.
    region : str, optional
        The bins the error norm scores, the controller's ``region``: ``"active"`` (the
        default) or ``"band"``.

    Returns
    -------
    Grids
        Section 2 of the kernel contract, with read-only arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import FieldGrid
    >>> from solprop.translate.masks import build_masks
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 4e-6), 200e-15)
    >>> masks = build_masks(grid)
    >>> bool(np.array_equal(masks.active_mask, grid.active_mask))
    True
    >>> masks.crop_index.flags.writeable
    False
    >>> bool(np.all(masks.omega_window == 1.0))
    True
    >>> taper = build_masks(grid, polarisation_taper=True).omega_window
    >>> bool(np.array_equal(taper, grid.omega_window))
    True
    >>> masks.carrier_2w0 is None
    True
    """
    window = (
        grid.omega_window if polarisation_taper else np.ones_like(grid.omega_window)
    )
    carrier = thg_carrier(grid.omega0, grid.t_fine) if harmonic_carrier else None
    masks = Grids(
        omega=grid.omega,
        omega0=grid.omega0,
        dt=grid.dt,
        dt_fine=grid.dt_fine,
        n_t=grid.n_t,
        n_to=grid.n_to,
        ffac=grid.ffac,
        crop_index=grid.crop_index,
        pad_scale=grid.pad_scale,
        crop_scale=grid.crop_scale,
        inband_mask=grid.inband_mask,
        active_mask=grid.active_mask,
        norm_mask=grid.active_mask if region == "active" else grid.inband_mask,
        omega_window=window,
        t_window_fine=grid.t_window_fine,
        t_window_fine_derivative=grid.t_collar.window_derivative(grid.t_fine),
        carrier_2w0=carrier,
    )
    logger.debug(
        "masks: n_omega %d, n_t %d, n_to %d (ffac %.3g), %d active bins, %d in band, "
        "harmonic carrier %s",
        masks.n_omega,
        masks.n_t,
        masks.n_to,
        masks.ffac,
        int(np.count_nonzero(masks.active_mask)),
        int(np.count_nonzero(masks.inband_mask)),
        "carried" if masks.carrier_2w0 is not None else "absent",
    )
    return masks


def temporal_rates(
    absorber: Absorber | None, *, scheme: str = DEFAULT_ABSORBER_SCHEME
) -> tuple[
    Float[np.ndarray, " n_t"] | None,
    Float[np.ndarray, " n_to"] | None,
]:
    """Return the coarse and fine temporal absorber rates the chosen scheme needs.

    The temporal collar cannot be folded into the cumulative linear operator, because it
    is not diagonal in ``omega``. It crosses the seam as the bounded profile
    of :func:`solprop.boundaries.temporal_collar_rate` on both time axes, and `scheme`
    records which of
    the two exact applications of the rate the backend is to use: the per-accepted-step
    factor ``exp(-alpha h / 2)`` on the coarse axis, or the term ``-alpha E / 2`` inside
    the nonlinear evaluation on the fine axis. Both rates are carried whichever is
    chosen, so switching schemes is not a retranslation.

    A collar whose temporal rates are all zero, however it was built (a spectral-only
    collar from ``absorber(grid, temporal=False)``, or a hand-built one), crosses the
    seam as **no** temporal absorber under either scheme (SEP 22): the kernels then
    compile no in-stage term and no split pass, recycle the first-same-as-last stage,
    and the scheme, still recorded, applies to nothing. Under the in-stage scheme the
    result is exactly the zero rate's, which only added zeros; under the split it
    differs by the transverse transform pair the zero factor cost, at rounding.

    Parameters
    ----------
    absorber : solprop.boundaries.Absorber or None
        The collar rates from :func:`solprop.boundaries.absorber`. ``None`` means no
        temporal absorber.
    scheme : {'split_step', 'nonlinear'}, optional
        Which application of the rate the backend will use. Default
        :data:`DEFAULT_ABSORBER_SCHEME`, ``'nonlinear'``.
        Annotated as a plain string rather than :data:`AbsorberScheme` for the reason
        given in ``solprop.fourier._check_kind``: this is the run-time guard, and
        the test session's ``beartype`` hook would otherwise reject a bad value first,
        leaving the guard unreachable.

    Returns
    -------
    tuple
        ``(coarse, fine)``: rates in 1/m of shapes ``(n_t,)`` and ``(n_to,)``, or
        ``(None, None)`` when `absorber` is ``None`` or its temporal rates are all
        zero.

    Raises
    ------
    TranslationError
        If `scheme` is not one of the two schemes.

    Examples
    --------
    >>> from solprop.boundaries import absorber
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.translate.masks import temporal_rates
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> coarse, fine = temporal_rates(absorber(grid))
    >>> coarse.shape == (grid.n_t,), fine.shape == (grid.n_to,)
    (True, True)
    >>> temporal_rates(None)
    (None, None)
    >>> temporal_rates(absorber(grid, temporal=False))
    (None, None)
    """
    if scheme not in ("split_step", "nonlinear"):
        msg = f"absorber scheme must be 'split_step' or 'nonlinear', got {scheme!r}"
        raise TranslationError(msg)
    if absorber is None:
        return None, None
    if not np.any(absorber.temporal_rate) and not np.any(absorber.temporal_rate_fine):
        logger.debug(
            "temporal absorber (%s): every rate is zero, so no temporal pass is "
            "compiled",
            scheme,
        )
        return None, None
    coarse = np.ascontiguousarray(absorber.temporal_rate, dtype=np.float64)
    fine = np.ascontiguousarray(absorber.temporal_rate_fine, dtype=np.float64)
    coarse.setflags(write=False)
    fine.setflags(write=False)
    logger.debug(
        "temporal absorber (%s): peak rate %.6e 1/m over %d coarse and %d fine samples",
        scheme,
        float(coarse.max(initial=0.0)),
        coarse.size,
        fine.size,
    )
    return coarse, fine


def noise_spectrum(
    noise: NoiseRealisation, shape: tuple[int, int, int, int]
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Put a shot-noise realisation into the canonical state layout.

    The realisation is drawn as ``(n_modes, n_pol, n_omega)``
    (:mod:`solprop.noise`); the kernel state carries a leading batch axis
    (``conventions.md`` section 6). A realisation with one mode or one polarisation is
    broadcast over the requested shape, so a batch of members shares one draw and stays
    reproducible from its seed.

    Parameters
    ----------
    noise : solprop.noise.NoiseRealisation
        The realisation, in the internal FFT convention.
    shape : tuple of int
        Target ``(n_batch, n_modes, n_pol, n_omega)``.

    Returns
    -------
    numpy.ndarray
        Read-only complex128 array of the requested shape.

    Raises
    ------
    TranslationError
        If the realisation cannot be broadcast onto `shape`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.noise import modified_noise
    >>> from solprop.translate.masks import noise_spectrum
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 4e-12)
    >>> draw = modified_noise(grid, np.random.default_rng(7))
    >>> noise_spectrum(draw, (1, 1, 1, grid.n_omega)).shape
    (1, 1, 1, 3072)
    """
    try:
        values = np.broadcast_to(noise.spectrum[np.newaxis], shape)
    except ValueError as error:
        msg = (
            f"noise0: a realisation of shape {noise.spectrum.shape} does not fit the "
            f"state shape {shape}"
        )
        raise TranslationError(msg) from error
    spectrum = np.ascontiguousarray(values, dtype=np.complex128)
    spectrum.setflags(write=False)
    return spectrum
