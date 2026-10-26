# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Compress the pulse of a saved run with linear optics, and say how well.

Each compressor searches the parameters of one kind of element of
:mod:`solprop.optics` -- a Taylor spectral phase, a thickness of material, reflections
from a chirped mirror with an optional wedge -- for the pulse of highest peak power,
and returns a :class:`Compression` holding what it found, the transfer function ready
to hand to :func:`~solprop.processing.fields.get_power` or
:func:`~solprop.plotting.propagation.time_1d`, and the pulse's duration and peak power
before, after, and at the transform limit.

The objective is

.. math:: J = \frac{\max_t P_{\rm limit}(t)}{\max_t P(t)} \ge 1,

with :math:`P` the instantaneous power summed over the selected modes and both
polarisation components. :math:`J = 1` is the transform limit and is the target.
Every search is deterministic: SciPy's bounded scalar minimiser after a coarse scan, or
its differential evolution with a fixed seed. A search that ends on a bound or fails is
reported in the result and logged; it never raises for that and never claims success.
"""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from jaxtyping import Complex, Float
from scipy.optimize import differential_evolution, minimize_scalar

from solprop._display import Displayed, engineering, line, phase_text
from solprop.constants import C
from solprop.exceptions import OpticsGeometryError, ProcessingError
from solprop.grid import TimeGrid
from solprop.optics import (
    Chain,
    ChirpedMirror,
    GratingPair,
    Material,
    Medium,
    PrismPair,
    TaylorPhase,
    TransferFunction,
)
from solprop.output.capability import capability_of
from solprop.output.schema import Output
from solprop.processing import spectra, time
from solprop.processing.axes import _keep_index, _plane_field, grid_of
from solprop.processing.fields import _fft_norm, _to_time

__all__ = [
    "CepOptimum",
    "Compression",
    "ModeChoice",
    "apply_transfer",
    "compress_gratings",
    "compress_material",
    "compress_mirrors",
    "compress_prisms",
    "compress_taylor",
    "optimise_cep",
    "transform_limit",
]

logger = logging.getLogger(__name__)

ModeChoice = int | Sequence[int] | None
"""Which modes a compressor works on: one, several, or ``None`` for all of them."""

_SCAN_POINTS = 65
"""Samples of the coarse scan that brackets a one-parameter search.

The objective of a structured pulse has several minima, and a bounded Brent search
started blind finds whichever is nearest its golden-section points. Sixty
five samples resolve a minimum wider than a thirtieth of the range, which for a
material thickness range chosen by a person is every minimum that matters.
"""

_CEP_SCAN = 64
"""Carrier-envelope phases sampled around the circle before the local refinement."""

_REFINEMENT = 1e-9
"""Tolerance of a one-parameter refinement, as a fraction of its bracket. The peak of a
pulse is quadratic in the residual chirp, so the objective is flat to 1e-18 there: past
what double precision resolves, for about thirty iterations of Brent's method."""

_BOUND_TOLERANCE = 1e-3
"""Fraction of a parameter's range within which an optimum counts as on the bound."""


@dataclass(frozen=True, eq=False)
class Compression(Displayed):
    """What a compressor found.

    Attributes
    ----------
    kind : str
        ``"taylor"``, ``"material"``, ``"mirrors"``, ``"gratings"`` or ``"prisms"``.
    transfer : TransferFunction
        The optimal element, for ``get_power(..., transfer=...)`` and
        ``time_1d(..., transfer=...)``.
    coefficients : tuple of float
        Taylor coefficients ``(0, 0, GDD, TOD, ...)`` in ``s**n``; empty otherwise.
    thickness : float
        Material thickness in m; NaN when no material is involved.
    reflections : int
        Number of mirror reflections; zero when no mirror is involved.
    fwhm_before : float
        Full width at half maximum of the power before compression, s.
    fwhm_after : float
        The same after it, s.
    fwhm_limit : float
        The same at the transform limit, s.
    peak_power_before : float
        Peak power before compression, W.
    peak_power_after : float
        Peak power after it, W. A mirror's reflectivity lowers it.
    peak_power_limit : float
        Peak power at the transform limit, W.
    quality : float
        ``peak_power_after / peak_power_limit``, at most one.
    evaluations : int
        Objective evaluations spent.
    success : bool
        Whether the optimiser converged **and** the optimum lies inside its bounds.
    message : str
        What the optimiser said, and which bound was reached if one was.
    table : tuple of tuple
        For mirrors: ``(reflections, thickness, quality)`` of every count tried.
    """

    kind: str
    transfer: TransferFunction
    coefficients: tuple[float, ...]
    thickness: float
    reflections: int
    fwhm_before: float
    fwhm_after: float
    fwhm_limit: float
    peak_power_before: float
    peak_power_after: float
    peak_power_limit: float
    quality: float
    evaluations: int
    success: bool
    message: str
    table: tuple[tuple[int, float, float], ...] = ()

    def _found_text(self) -> str:
        """Return what the search found, in the units it is quoted in.

        Returns
        -------
        str
            The Taylor coefficients, the thickness, or the reflections and thickness.
        """
        if self.kind == "taylor":
            return phase_text(self.coefficients)
        if isinstance(self.transfer, GratingPair):
            return f"normal separation {engineering(self.transfer.separation, 'm')}"
        if isinstance(self.transfer, PrismPair):
            geometry = self.transfer.geometry
            return (
                f"R {engineering(geometry.separation, 'm')}, "
                f"H {engineering(geometry.insertion, 'm')}"
            )
        parts = []
        if self.kind == "mirrors":
            parts.append(f"{self.reflections} reflections")
        if not math.isnan(self.thickness):
            parts.append(f"{engineering(self.thickness, 'm')} of material")
        return ", ".join(parts)

    def __repr__(self) -> str:
        """Return a one-line description: what was found and how good it is.

        Returns
        -------
        str
            For example ``Compression(taylor: GDD -250 fs^2, 38 fs to 30 fs,
            quality=0.999)``.
        """
        return line(
            "Compression",
            [
                ("", f"{self.kind}: {self._found_text()}"),
                (
                    "",
                    f"{engineering(self.fwhm_before, 's')} to "
                    f"{engineering(self.fwhm_after, 's')}",
                ),
                ("quality", f"{self.quality:.4g}"),
                ("success", self.success),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        rows: list[tuple[str, object]] = [
            ("found", self._found_text()),
            (
                "duration",
                f"{engineering(self.fwhm_before, 's')} before, "
                f"{engineering(self.fwhm_after, 's')} after, "
                f"{engineering(self.fwhm_limit, 's')} at the transform limit",
            ),
            (
                "peak power",
                f"{engineering(self.peak_power_before, 'W')} before, "
                f"{engineering(self.peak_power_after, 'W')} after, "
                f"{engineering(self.peak_power_limit, 'W')} at the limit",
            ),
            ("quality", f"{self.quality:.4g} of the transform-limited peak power"),
            (
                "search",
                f"{'converged' if self.success else 'NOT CLEAN'}: {self.message} "
                f"({self.evaluations} evaluations)",
            ),
        ]
        if isinstance(self.transfer, (GratingPair, PrismPair)):
            band = self.transfer.band
            label = (
                "all active frequencies"
                if band is None
                else f"{engineering(band[0], 'm')} to {engineering(band[1], 'm')}"
            )
            rows.insert(1, ("fixed working band", label))
            rows.insert(2, ("metrics", "selected pulse, including any extra bandpass"))
        if self.table:
            rows.append(
                (
                    "per count",
                    "\n".join(
                        f"{count}: quality {quality:.4g}"
                        + (
                            ""
                            if math.isnan(thickness)
                            else f", {engineering(thickness, 'm')}"
                        )
                        for count, thickness, quality in self.table
                    ),
                )
            )
        return f"Compression by {self.kind}", rows


@dataclass(frozen=True, eq=False)
class CepOptimum(Displayed):
    """The carrier-envelope phase that maximises the peak of the real field.

    Attributes
    ----------
    phase : float
        The phase in rad, in ``[-pi, pi]``, to **add** in the optics convention.
    transfer : TransferFunction
        ``TaylorPhase((phase,), wavelength)``.
    peak_field : float
        The largest value of the real field after the shift, in the square root of W.
    """

    phase: float
    transfer: TransferFunction
    peak_field: float

    def __repr__(self) -> str:
        """Return a one-line description: the phase to add and the field it gives.

        Returns
        -------
        str
            For example ``CepOptimum(phase=-1.3 rad, peak_field=1.5e+04 sqrt(W))``.
        """
        return line(
            "CepOptimum",
            [
                ("phase", f"{self.phase:.4g} rad"),
                ("peak_field", f"{self.peak_field:.4g} sqrt(W)"),
            ],
        )


class _Pulse:
    """One plane of a run, held ready for repeated trial compressions.

    Parameters
    ----------
    grid : solprop.grid.TimeGrid
        The grid.
    stored : numpy.ndarray
        Stored spectrum, internal convention, shape ``(n_modes, n_pol, n_omega)``,
        already band-passed.
    oversampling : int
        Time-domain resampling factor used for every power evaluated.
    """

    def __init__(
        self,
        grid: TimeGrid,
        stored: Complex[np.ndarray, "n_modes n_pol n_omega"],
        oversampling: int,
    ) -> None:
        self.grid = grid
        self.stored = stored
        self.oversampling = oversampling
        self.evaluations = 0
        self.t, self.limit = self._power(None, transform_limit=True)
        self.peak_limit = float(np.max(self.limit))
        if not self.peak_limit > 0.0:
            raise ProcessingError("the selected plane carries no power to compress")

    def _power(
        self, transfer: TransferFunction | None, *, transform_limit: bool = False
    ) -> tuple[Float[np.ndarray, " n_t"], Float[np.ndarray, " n_t"]]:
        """Return the time axis and the power summed over modes and polarisation.

        Parameters
        ----------
        transfer : TransferFunction or None
            Element applied first.
        transform_limit : bool
            Whether to discard the spectral phase first.

        Returns
        -------
        tuple of numpy.ndarray
            Time in s and power in W.
        """
        t, amplitude = _to_time(
            self.grid,
            self.stored,
            oversampling=self.oversampling,
            bandpass=None,
            transfer=transfer,
            transform_limit=transform_limit,
        )
        return t, np.sum(np.abs(amplitude) ** 2, axis=(0, 1))

    def power(self, transfer: TransferFunction | None) -> Float[np.ndarray, " n_t"]:
        """Return the summed power after an element.

        Parameters
        ----------
        transfer : TransferFunction or None
            The element; ``None`` leaves the pulse as it is.

        Returns
        -------
        numpy.ndarray
            Power in W on :attr:`t`.
        """
        return self._power(transfer)[1]

    def objective(self, transfer: TransferFunction) -> float:
        """Return the transform-limited peak power over the peak power after an element.

        Parameters
        ----------
        transfer : TransferFunction
            The trial element.

        Returns
        -------
        float
            At least one for a phase-only element, and one at the transform limit;
            infinite if the element leaves no power in the window.
        """
        self.evaluations += 1
        peak = float(np.max(self.power(transfer)))
        return self.peak_limit / peak if peak > 0.0 else math.inf

    def result(  # noqa: PLR0913 - the fields of the result that differ between compressors
        self,
        kind: str,
        transfer: TransferFunction,
        *,
        coefficients: tuple[float, ...] = (),
        thickness: float = math.nan,
        reflections: int = 0,
        success: bool,
        message: str,
        table: tuple[tuple[int, float, float], ...] = (),
    ) -> Compression:
        """Assemble a result, measuring the pulse before and after.

        Parameters
        ----------
        kind : str
            Which compressor.
        transfer : TransferFunction
            The optimal element.
        coefficients : tuple of float, optional
            Taylor coefficients found.
        thickness : float, optional
            Thickness found, m.
        reflections : int, optional
            Reflections found.
        success : bool
            Whether the search converged inside its bounds.
        message : str
            What to tell the user about the search.
        table : tuple of tuple, optional
            Per-count results of a mirror search.

        Returns
        -------
        Compression
            The result. A failed search is also logged at ``WARNING``.
        """
        before, after = self.power(None), self.power(transfer)
        if not success:
            logger.warning("%s compression did not converge cleanly: %s", kind, message)
        return Compression(
            kind=kind,
            transfer=transfer,
            coefficients=coefficients,
            thickness=thickness,
            reflections=reflections,
            fwhm_before=time.fwhm(self.t, before),
            fwhm_after=time.fwhm(self.t, after),
            fwhm_limit=time.fwhm(self.t, self.limit),
            peak_power_before=float(np.max(before)),
            peak_power_after=float(np.max(after)),
            peak_power_limit=self.peak_limit,
            quality=float(np.max(after)) / self.peak_limit,
            evaluations=self.evaluations,
            success=success,
            message=message,
            table=table,
        )


def _select_modes(
    stored: Complex[np.ndarray, "n_modes n_pol n_omega"], modes: ModeChoice
) -> Complex[np.ndarray, "n_selected n_pol n_omega"]:
    """Return the modes a compressor works on.

    Parameters
    ----------
    stored : numpy.ndarray
        One plane, shape ``(n_modes, n_pol, n_omega)``.
    modes : int, sequence of int or None
        The selection.

    Returns
    -------
    numpy.ndarray
        The selected modes, mode axis kept.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If an index is out of range.
    """
    if modes is None:
        return stored
    chosen = [modes] if isinstance(modes, int) else list(modes)
    if not chosen or any(
        not -stored.shape[0] <= index < stored.shape[0] for index in chosen
    ):
        raise ProcessingError(
            f"modes={modes!r} does not select from the {stored.shape[0]} modes saved"
        )
    return stored[chosen]


def _require_modal(output: Output) -> None:
    """Refuse an output whose amplitudes are not orthogonal modal powers.

    Every compressor's objective sums ``|a|^2`` over the selected modes and
    polarisations, which is the physical power only on an orthogonal modal basis
    (:func:`solprop.output.capability.capability_of`). A general modal or spatial
    run's coordinates are not that, so it is refused by name rather than optimised on
    a quantity that is not a power.

    Parameters
    ----------
    output : solprop.output.schema.Output
        An opened output handler.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the output carries no metadata, or declares a ``general_modal`` or
        ``spatial`` capability.
    """
    if output.metadata is None:
        raise ProcessingError("output has no metadata")
    capability = capability_of(output.metadata)
    if capability != "modal":
        raise ProcessingError(
            f"compression requires orthogonal modal power amplitudes, and this is a "
            f"{capability} output: its summed |a|^2 is not a physical power, so it "
            f"needs an explicit physical-flux objective"
        )


def _pulse_of(  # noqa: PLR0913 - the two input forms and the selection common to both
    output: Output | None,
    grid: TimeGrid | None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None,
    *,
    plane: int,
    modes: ModeChoice,
    bandpass: spectra.Bandpass | None,
    oversampling: int,
) -> _Pulse:
    """Return the pulse a compressor works on, from either input form.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        An opened output handler.
    grid : solprop.grid.TimeGrid or None
        The grid of a bare spectrum.
    spectrum : numpy.ndarray or None
        A stored spectrum in the internal convention, as a pulse's ``build(grid)``
        returns it: shape ``(n_omega,)`` or ``(n_modes, n_pol, n_omega)``.
    plane : int
        Plane of the output to take.
    modes : int, sequence of int or None
        Modes to work on.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Window applied to the spectrum first.
    oversampling : int
        Time-domain resampling factor.

    Returns
    -------
    _Pulse
        The plane, ready.

    Raises
    ------
    solprop.exceptions.ProcessingError
        Unless exactly one input form is given, or if the output is not an
        orthogonal modal one (:func:`_require_modal`).
    """
    mixed = "give an output, or a grid together with its stored spectrum, not both"
    if output is not None:
        if grid is not None or spectrum is not None:
            raise ProcessingError(mixed)
        _require_modal(output)
        working = grid_of(output)
        stored = _plane_field(output, plane, None, convention="fft")
    elif grid is not None and spectrum is not None:
        working = grid
        stored = np.asarray(spectrum, dtype=np.complex128)
        stored = stored.reshape((1, 1, -1)) if stored.ndim == 1 else stored
    else:
        raise ProcessingError(mixed)
    stored = _select_modes(stored, modes)
    if bandpass is not None:
        stored = stored * spectra._as_window(working, bandpass)
    return _Pulse(working, stored, oversampling)


def apply_transfer(
    transfer: TransferFunction,
    output: Output,
    *,
    plane: int = -1,
    modes: ModeChoice = None,
) -> tuple[Float[np.ndarray, " n_x"], Complex[np.ndarray, "*shape n_x"]]:
    """Return the spectrum of a plane after a linear optical element.

    Parameters
    ----------
    transfer : TransferFunction
        The element, from :mod:`solprop.optics`.
    output : solprop.output.schema.Output
        An opened output handler.
    plane : int, optional
        Index of the plane, negative from the end.
    modes : int, sequence of int or None, optional
        Which modes to return.

    Returns
    -------
    omega : numpy.ndarray
        Angular frequency of the active band, rad/s, ascending.
    spectrum : numpy.ndarray
        The spectral amplitude in the optics convention, in the square root of
        J/(rad/s), shape ``(n_modes, n_pol, n_x)``: the normalisation of
        :func:`~solprop.processing.fields.get_spectrum` with no axis squeezed.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.optics import TaylorPhase
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.compression import apply_transfer
    >>> element = TaylorPhase((0, 0, 1e-27), 800e-9)
    >>> omega, spectrum = apply_transfer(element, example_output())
    >>> spectrum.shape[-1] == omega.size
    True
    """
    grid = grid_of(output)
    stored = _select_modes(
        _plane_field(output, plane, None, convention="optics"), modes
    )
    keep = _keep_index(grid, "active")
    omega = grid.omega[keep]
    return omega, stored[..., keep] * _fft_norm(grid) * transfer(omega)


def transform_limit(  # noqa: PLR0913 - the two input forms and the selection
    output: Output | None = None,
    *,
    grid: TimeGrid | None = None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None = None,
    plane: int = -1,
    modes: ModeChoice = None,
    bandpass: spectra.Bandpass | None = None,
    oversampling: int = 4,
) -> tuple[float, float]:
    """Return the duration and peak power of a plane's transform-limited pulse.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        An opened output handler.
    grid : solprop.grid.TimeGrid or None
        With `spectrum`, the alternative to `output`.
    spectrum : numpy.ndarray or None
        A stored spectrum in the internal convention, as ``pulse.build(grid)`` returns.
    plane : int, optional
        Index of the plane, negative from the end.
    modes : int, sequence of int or None, optional
        Modes whose summed power is measured.
    bandpass : tuple of float, Passband, numpy.ndarray or None, optional
        Window applied to the spectrum first.
    oversampling : int, optional
        Time-domain resampling factor.

    Returns
    -------
    fwhm : float
        Full width at half maximum of the transform-limited power, s.
    peak_power : float
        Its peak power, W.

    Examples
    --------
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.compression import transform_limit
    >>> fwhm, peak = transform_limit(example_output())
    >>> fwhm > 0.0 and peak > 0.0
    True
    """
    pulse = _pulse_of(
        output,
        grid,
        spectrum,
        plane=plane,
        modes=modes,
        bandpass=bandpass,
        oversampling=oversampling,
    )
    return time.fwhm(pulse.t, pulse.limit), pulse.peak_limit


def _scan_then_refine(
    objective: Callable[[float], float], low: float, high: float
) -> tuple[float, bool, str]:
    """Minimise a function of one bounded parameter that may have several minima.

    Parameters
    ----------
    objective : callable
        The function.
    low : float
        Lower bound.
    high : float
        Upper bound, above `low`.

    Returns
    -------
    best : float
        The minimiser.
    success : bool
        Whether the refinement converged with the minimiser inside the bounds.
    message : str
        What happened, in words.
    """
    samples = np.linspace(low, high, _SCAN_POINTS)
    values = np.array([objective(float(sample)) for sample in samples])
    centre = int(np.argmin(values))
    left = float(samples[max(centre - 1, 0)])
    right = float(samples[min(centre + 1, _SCAN_POINTS - 1)])
    # SciPy's `xatol` is ABSOLUTE, in the parameter's own units, and defaults to 1e-5:
    # twenty orders above a group-delay dispersion in s^2 and ten microns of glass. Left
    # at its default the refinement returns its first golden-section point and calls
    # that a solution. It is set from the bracket instead.
    found = minimize_scalar(
        objective,
        bounds=(left, right),
        method="bounded",
        options={"xatol": _REFINEMENT * (right - left)},
    )
    best, success, message = float(found.x), bool(found.success), str(found.message)
    if found.fun > values[centre]:
        best = float(samples[centre])
        message = "the refinement did not improve on the scan; its best sample is kept"
    margin = _BOUND_TOLERANCE * (high - low)
    if best - low < margin or high - best < margin:
        side = "lower" if best - low < margin else "upper"
        return best, False, f"the optimum {best:.6g} lies on the {side} bound"
    return best, success, message


def _gaussian_width(
    t: Float[np.ndarray, " n_t"], power: Float[np.ndarray, " n_t"]
) -> float:
    """Return a Gaussian-equivalent natural width of a possibly structured pulse.

    The larger of the width read from the full width at half maximum and the one read
    from the root-mean-square duration, each converted as for a Gaussian. A clean pulse
    gives the same number twice; a pulse with a pedestal or satellites has a small
    half-maximum width and a large RMS one, and the search bounds built on this width
    must contain the chirp of the whole pulse, not of its spike.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis, s.
    power : numpy.ndarray
        Power on it, W.

    Returns
    -------
    float
        ``tau0`` of ``exp(-t**2 / tau0**2)`` in s.
    """
    from_fwhm = time.fwhm(t, power) / (2.0 * math.sqrt(math.log(2.0)))
    centre = time.moment(t, power)
    variance = time.moment(t - centre, power, 2)
    from_rms = math.sqrt(2.0 * max(variance, 0.0))
    return max(from_fwhm, from_rms)


def compress_taylor(  # noqa: PLR0913 - the input forms, the selection and the search
    output: Output | None = None,
    *,
    grid: TimeGrid | None = None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None = None,
    order: int = 2,
    wavelength: float | None = None,
    plane: int = -1,
    modes: ModeChoice = None,
    bandpass: spectra.Bandpass | None = None,
    bound_factor: float = 8.0,
    oversampling: int = 2,
    seed: int = 0,
) -> Compression:
    """Find the Taylor spectral phase that best compresses a pulse.

    Orders two to `order` are searched together; the carrier-envelope phase and the
    group delay are held at zero, since neither changes the peak power. The bounds are
    heuristic: with ``phi2 = tau_lim sqrt(abs(tau**2 - tau_lim**2))`` the chirp a
    Gaussian of the measured and transform-limited widths would carry, order ``n`` is
    searched within plus and minus ``bound_factor * phi2**(n/2)``. The search starts
    from the better of ``-phi2`` and ``+phi2``.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        An opened output handler.
    grid : solprop.grid.TimeGrid or None
        With `spectrum`, the alternative to `output`.
    spectrum : numpy.ndarray or None
        A stored spectrum in the internal convention, as ``pulse.build(grid)`` returns.
    order : int, optional
        Highest order searched, at least 2. Two is a one-parameter search; above it
        the search is SciPy's differential evolution, seeded, then polished.
    wavelength : float or None, optional
        Wavelength the phase is expanded about, m; the grid's reference by default.
    plane : int, optional
        Index of the plane, negative from the end.
    modes : int, sequence of int or None, optional
        Modes whose summed power is maximised.
    bandpass : tuple of float, Passband, numpy.ndarray or None, optional
        Window applied to the spectrum first.
    bound_factor : float, optional
        Width of the search box in units of the Gaussian estimate, > 0.
    oversampling : int, optional
        Time-domain resampling factor of every power evaluated.
    seed : int, optional
        Seed of the differential evolution; the same seed gives the same result.

    Returns
    -------
    Compression
        ``coefficients`` is ``(0, 0, GDD, ...)``, ready for ``TaylorPhase`` and for the
        ``phases=`` keyword of a run.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If `order` is below two, `bound_factor` is not positive, or the input forms
        are mixed.

    Examples
    --------
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.compression import compress_taylor
    >>> found = compress_taylor(example_output())
    >>> found.success and found.quality > 0.999
    True
    """
    if order < 2:
        raise ProcessingError(f"order must be at least 2, got {order}")
    if not bound_factor > 0.0:
        raise ProcessingError(f"bound_factor must be positive, got {bound_factor}")
    pulse = _pulse_of(
        output,
        grid,
        spectrum,
        plane=plane,
        modes=modes,
        bandpass=bandpass,
        oversampling=oversampling,
    )
    reference = pulse.grid.reference_wavelength if wavelength is None else wavelength
    width_limit = _gaussian_width(pulse.t, pulse.limit)
    width = _gaussian_width(pulse.t, pulse.power(None))
    chirp = width_limit * math.sqrt(abs(width**2 - width_limit**2))
    # A pulse already at its limit gives no chirp to scale the box with; the limit's
    # own width squared is then the natural unit of group-delay dispersion.
    scale = math.sqrt(max(chirp, width_limit**2))
    bounds = [bound_factor * scale**n for n in range(2, order + 1)]

    def objective(dispersion: Sequence[float] | np.ndarray) -> float:
        """Return the objective for trial dispersion coefficients.

        Parameters
        ----------
        dispersion : sequence of float
            Orders two and above.

        Returns
        -------
        float
            The transform-limited peak power over the trial's.
        """
        return pulse.objective(TaylorPhase((0.0, 0.0, *dispersion), reference))

    if order == 2:
        best, success, message = _scan_then_refine(
            lambda gdd: objective([gdd]), -bounds[0], bounds[0]
        )
        found = [best]
    else:
        # A narrow box may not contain the Gaussian estimate; start on its edge then.
        edge = min(chirp, bounds[0])
        start = min(
            ([-edge], [edge]),
            key=lambda guess: objective([*guess, *([0.0] * (order - 2))]),
        )
        result = differential_evolution(
            objective,
            [(-bound, bound) for bound in bounds],
            x0=[*start, *([0.0] * (order - 2))],
            seed=seed,
            polish=True,
            # Convergence of the population, not accuracy: the polish that follows is
            # what sets the digits. Tighter than this and a hard objective exhausts
            # `maxiter` and reports failure beside a good answer.
            tol=1e-8,
            atol=1e-10,
            maxiter=300,
        )
        found = [float(value) for value in result.x]
        success, message = bool(result.success), str(result.message)
        for n, (value, bound) in enumerate(zip(found, bounds, strict=True), start=2):
            if bound - abs(value) < _BOUND_TOLERANCE * 2.0 * bound:
                success, message = False, f"order {n} lies on its bound {bound:.4g}"
    coefficients = (0.0, 0.0, *found)
    return pulse.result(
        "taylor",
        TaylorPhase(coefficients, reference),
        coefficients=coefficients,
        success=success,
        message=message,
    )


def compress_material(  # noqa: PLR0913 - the input forms, the medium and the selection
    output: Output | None = None,
    *,
    medium: Medium,
    thickness_range: tuple[float, float],
    grid: TimeGrid | None = None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None = None,
    wavelength: float | None = None,
    pressure: float | None = None,
    plane: int = -1,
    modes: ModeChoice = None,
    bandpass: spectra.Bandpass | None = None,
    oversampling: int = 2,
) -> Compression:
    """Find the thickness of a material that best compresses a pulse.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        An opened output handler.
    medium : GlassIndex, Gas, str or Mixture
        What the material is; see :class:`solprop.optics.Material`.
    thickness_range : tuple of float
        Smallest and largest thickness in m. Negative values remove material: a
        compressor built with too much glass in it is corrected by taking some out.
    grid : solprop.grid.TimeGrid or None
        With `spectrum`, the alternative to `output`.
    spectrum : numpy.ndarray or None
        A stored spectrum in the internal convention, as ``pulse.build(grid)`` returns.
    wavelength : float or None, optional
        Wavelength at which the material's group delay is removed, m; the grid's
        reference by default.
    pressure : float or None, optional
        Pressure in bar, for a built-in gas.
    plane : int, optional
        Index of the plane, negative from the end.
    modes : int, sequence of int or None, optional
        Modes whose summed power is maximised.
    bandpass : tuple of float, Passband, numpy.ndarray or None, optional
        Window applied to the spectrum first.
    oversampling : int, optional
        Time-domain resampling factor of every power evaluated.

    Returns
    -------
    Compression
        ``thickness`` holds the optimum. The range is scanned coarsely before the
        bounded refinement, so the global optimum is found where a blind Brent search
        would find the nearest one.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If the thickness range is not ordered, or the input forms are mixed.

    Examples
    --------
    >>> from solprop.physdata import calcium_fluoride
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.compression import compress_material
    >>> window = calcium_fluoride(model="malitson_1963/sellmeier")
    >>> found = compress_material(
    ...     example_output(), medium=window, thickness_range=(-0.05, 0.05))
    >>> found.kind
    'material'
    """
    low, high = thickness_range
    if not low < high:
        raise ProcessingError(f"thickness_range must be ordered, got {thickness_range}")
    pulse = _pulse_of(
        output,
        grid,
        spectrum,
        plane=plane,
        modes=modes,
        bandpass=bandpass,
        oversampling=oversampling,
    )
    reference = pulse.grid.reference_wavelength if wavelength is None else wavelength

    def element(thickness: float) -> Material:
        """Return the material at one thickness.

        Parameters
        ----------
        thickness : float
            Thickness in m.

        Returns
        -------
        Material
            The element.
        """
        return Material(medium, thickness, reference, pressure=pressure)

    best, success, message = _scan_then_refine(
        lambda thickness: pulse.objective(element(thickness)), low, high
    )
    return pulse.result(
        "material", element(best), thickness=best, success=success, message=message
    )


def _behind_mirrors(
    pulse: _Pulse,
    mirrors: ChirpedMirror,
    glass: Medium,
    reference: float,
    thickness: float,
) -> float:
    """Return the objective for one insertion of material behind some reflections.

    Parameters
    ----------
    pulse : _Pulse
        The pulse being compressed.
    mirrors : ChirpedMirror
        The reflections.
    glass : Medium
        The material.
    reference : float
        Wavelength at which the material's group delay is removed, m.
    thickness : float
        Thickness in m.

    Returns
    -------
    float
        The transform-limited peak power over the trial's.
    """
    insertion = Material(glass, thickness, reference)
    return pulse.objective(Chain((mirrors, insertion)))


def compress_mirrors(  # noqa: PLR0913 - the input forms, the mirror, the wedge and the selection
    output: Output | None = None,
    *,
    mirror: ChirpedMirror | str,
    max_reflections: int,
    medium: Medium | None = None,
    thickness_range: tuple[float, float] | None = None,
    grid: TimeGrid | None = None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None = None,
    wavelength: float | None = None,
    plane: int = -1,
    modes: ModeChoice = None,
    bandpass: spectra.Bandpass | None = None,
    oversampling: int = 2,
) -> Compression:
    """Find the number of chirped-mirror reflections that best compresses a pulse.

    Every count from zero to `max_reflections` is tried. With a `medium`, its thickness
    is optimised for each count -- the mirrors-and-wedges compressor as it stands on a
    table, where the mirrors over-compensate in whole bounces and glass trims the rest.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        An opened output handler.
    mirror : ChirpedMirror or str
        The mirror, or the name of a shipped data set
        (:func:`solprop.optics.chirped_mirrors`).
    max_reflections : int
        Largest number of reflections tried, at least 1.
    medium : GlassIndex, Gas, str, Mixture or None, optional
        A material whose thickness is tuned for each count.
    thickness_range : tuple of float or None, optional
        Its smallest and largest thickness in m; required with `medium`.
    grid : solprop.grid.TimeGrid or None
        With `spectrum`, the alternative to `output`.
    spectrum : numpy.ndarray or None
        A stored spectrum in the internal convention, as ``pulse.build(grid)`` returns.
    wavelength : float or None, optional
        Wavelength at which the material's group delay is removed, m.
    plane : int, optional
        Index of the plane, negative from the end.
    modes : int, sequence of int or None, optional
        Modes whose summed power is maximised.
    bandpass : tuple of float, Passband, numpy.ndarray or None, optional
        Window applied to the spectrum first.
    oversampling : int, optional
        Time-domain resampling factor of every power evaluated.

    Returns
    -------
    Compression
        ``reflections`` and ``thickness`` hold the best pair and ``table`` every count
        tried. ``quality`` includes the mirror's reflectivity and the part of the
        spectrum outside its band, both of which lower the peak power.

    Raises
    ------
    solprop.exceptions.ProcessingError
        If `max_reflections` is below one, a medium is given without a range or a
        range without a medium, or the input forms are mixed.

    Examples
    --------
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.compression import compress_mirrors
    >>> found = compress_mirrors(example_output(), mirror="PC70", max_reflections=4)
    >>> 0 <= found.reflections <= 4
    True
    """
    if max_reflections < 1:
        raise ProcessingError(
            f"max_reflections must be at least 1, got {max_reflections}"
        )
    if (medium is None) != (thickness_range is None):
        raise ProcessingError("give medium and thickness_range together, or neither")
    pulse = _pulse_of(
        output,
        grid,
        spectrum,
        plane=plane,
        modes=modes,
        bandpass=bandpass,
        oversampling=oversampling,
    )
    reference = pulse.grid.reference_wavelength if wavelength is None else wavelength
    single = ChirpedMirror.lookup(mirror) if isinstance(mirror, str) else mirror
    # Both or neither, by the check above; paired so that the type checker knows it.
    wedge = (
        (medium, thickness_range)
        if medium is not None and thickness_range is not None
        else None
    )
    rows = []
    for count in range(max_reflections + 1):
        bounced = single.bounces(count)
        if wedge is None:
            rows.append((count, math.nan, pulse.objective(bounced), True, "scanned"))
            continue

        objective = partial(_behind_mirrors, pulse, bounced, wedge[0], reference)
        best, success, message = _scan_then_refine(objective, *wedge[1])
        rows.append((count, best, objective(best), success, message))
    count, thickness, _, success, message = min(rows, key=lambda row: row[2])
    chosen: TransferFunction = single.bounces(count)
    if wedge is not None:
        chosen = Chain((chosen, Material(wedge[0], thickness, reference)))
    if count == max_reflections:
        success, message = False, f"the optimum uses all {count} reflections allowed"
    return pulse.result(
        "mirrors",
        chosen,
        thickness=thickness,
        reflections=count,
        success=success,
        message=message,
        table=tuple((row[0], row[1], 1.0 / row[2]) for row in rows),
    )


def optimise_cep(
    output: Output | None = None,
    *,
    grid: TimeGrid | None = None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None = None,
    plane: int = -1,
    mode: int = 0,
    oversampling: int = 4,
) -> CepOptimum:
    """Find the carrier-envelope phase that maximises the peak of the real field.

    The phase is periodic, so the circle is scanned and the best sample refined in its
    own neighbourhood; Brent's method run across ``[-pi, pi]`` could not see an optimum
    at the ends.

    Both grids are served, an envelope's carrier being restored first. The phase
    returned is the optics-convention constant to **add**; it is the negative of the
    phase to add to an FFT-convention spectrum.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        An opened output handler.
    grid : solprop.grid.TimeGrid or None
        With `spectrum`, the alternative to `output`.
    spectrum : numpy.ndarray or None
        A stored spectrum in the internal convention, as ``pulse.build(grid)`` returns.
    plane : int, optional
        Index of the plane, negative from the end.
    mode : int, optional
        The one mode whose field is optimised: modal fields do not add away from a
        point in space. The first polarisation component is used.
    oversampling : int, optional
        Time-domain resampling factor; the field's peak lies between samples.

    Returns
    -------
    CepOptimum
        The phase to add, the element that adds it, and the peak field reached.

    Examples
    --------
    >>> from solprop.processing._examples import example_output
    >>> from solprop.processing.compression import optimise_cep
    >>> found = optimise_cep(example_output("field"))
    >>> -3.1416 < found.phase <= 3.1416
    True
    """
    pulse = _pulse_of(
        output,
        grid,
        spectrum,
        plane=plane,
        modes=mode,
        bandpass=None,
        oversampling=oversampling,
    )
    working = pulse.grid
    reference = working.reference_wavelength

    def peak(phase: float) -> float:
        """Return minus the largest value of the real field at one added phase.

        Parameters
        ----------
        phase : float
            Carrier-envelope phase added, rad.

        Returns
        -------
        float
            Minus the peak of the real field, in the square root of W.
        """
        t, amplitude = _to_time(
            working,
            pulse.stored,
            oversampling=oversampling,
            bandpass=None,
            transfer=TaylorPhase((phase,), reference),
        )
        if working.kind == "envelope":
            # The stored envelope has its carrier removed (internal convention).
            amplitude = amplitude * np.exp(1j * working.omega0 * t)
        return -float(np.max(np.real(amplitude[0, 0])))

    samples = np.linspace(-math.pi, math.pi, _CEP_SCAN, endpoint=False)
    start = float(samples[int(np.argmin([peak(float(sample)) for sample in samples]))])
    step = 2.0 * math.pi / _CEP_SCAN
    found = minimize_scalar(peak, bounds=(start - step, start + step), method="bounded")
    phase = math.remainder(float(found.x), 2.0 * math.pi)
    return CepOptimum(
        phase=phase,
        transfer=TaylorPhase((phase,), reference),
        peak_field=-float(found.fun),
    )


def _pair_range(bounds: tuple[float, float], name: str) -> tuple[float, float]:
    """Require explicit finite, ordered nonnegative motor travel in metres.

    Parameters
    ----------
    bounds : tuple of float
        Lower and upper stage coordinates in m.
    name : str
        Coordinate name for an error message.

    Returns
    -------
    tuple of float
        Validated lower and upper stage coordinates in m.
    """
    if (
        len(bounds) != 2
        or not all(math.isfinite(value) for value in bounds)
        or not 0.0 <= bounds[0] < bounds[1]
    ):
        raise ProcessingError(
            f"{name} must be finite, nonnegative and strictly ordered"
        )
    return bounds


def _pair_pulse(  # noqa: PLR0913 - the shared input forms and fixed selection
    output: Output | None,
    grid: TimeGrid | None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None,
    *,
    pair: GratingPair | PrismPair,
    plane: int,
    modes: ModeChoice,
    bandpass: spectra.Bandpass | None,
    oversampling: int,
) -> _Pulse:
    """Select the same pulse for every trial, before forming any power objective.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        Orthogonal modal saved output.
    grid : solprop.grid.TimeGrid or None
        Grid of a raw spectrum, alternative to output.
    spectrum : numpy.ndarray or None
        Stored internal spectrum, shape (n_omega,) or (n_modes, n_pol, n_omega).
    pair : GratingPair or PrismPair
        Fixed optical settings including the selected working band.
    plane : int
        Saved plane index, negative from the end.
    modes : int, sequence of int or None
        Modes whose orthogonal powers are summed.
    bandpass : tuple of float, Passband, numpy.ndarray or None
        Additional fixed spectral window.
    oversampling : int
        Positive temporal resampling factor.

    Returns
    -------
    _Pulse
        The selected pulse, with all metrics using the same selection.
    """
    if output is not None:
        _require_modal(output)
        if grid is not None or spectrum is not None:
            raise ProcessingError("give output or grid and spectrum, not both")
        working = grid_of(output)
    elif grid is not None and spectrum is not None:
        working = grid
        if (
            spectrum.ndim not in (1, 3)
            or spectrum.shape[-1] != grid.omega.size
            or spectrum.size == 0
            or not np.all(np.isfinite(spectrum))
        ):
            raise ProcessingError(
                "spectrum must be finite with shape (n_omega,) or "
                "(n_modes, n_pol, n_omega)"
            )
    else:
        raise ProcessingError("give output or grid and spectrum, not both")
    if (
        isinstance(oversampling, bool)
        or not isinstance(oversampling, int)
        or oversampling < 1
    ):
        raise ProcessingError("oversampling must be a positive integer")
    # Match stored_factor exactly: only active bins are selected. Out-of-active
    # collar residuals retain the existing identity response, even with a pair band.
    window = np.ones(working.omega.size)
    if pair.band is not None:
        low, high = pair.band
        active = working.active_mask
        window[active] = (working.omega[active] >= 2.0 * math.pi * C / high) & (
            working.omega[active] <= 2.0 * math.pi * C / low
        )
        if not np.any(window[active]):
            raise ProcessingError(
                "the fixed working band selects no active frequencies"
            )
    if bandpass is not None:
        window = window * spectra._as_window(working, bandpass)
    return _pulse_of(
        output,
        grid,
        spectrum,
        plane=plane,
        modes=modes,
        bandpass=window,
        oversampling=oversampling,
    )


def _pair_trial(
    pulse: _Pulse,
    element: Callable[[Sequence[float]], GratingPair | PrismPair],
    coordinates: Sequence[float],
) -> float:
    """Count every trial, including expected geometric failures only.

    Parameters
    ----------
    pulse : _Pulse
        Fixed selected pulse and objective evaluation counter.
    element : callable
        Map normalised stage coordinates to a physical pair.
    coordinates : sequence of float
        Normalised coordinates in the unit search interval or square.

    Returns
    -------
    float
        Transform-limited peak divided by trial peak, or infinity if infeasible.
    """
    pulse.evaluations += 1
    try:
        peak = float(np.max(pulse.power(element(coordinates))))
    except OpticsGeometryError:
        return math.inf
    return pulse.peak_limit / peak if peak > 0.0 else math.inf


def _feasible_scan(
    objective: Callable[[float], float],
) -> tuple[float, float, bool]:
    """Scan a unit interval and refine only brackets with feasible endpoints.

    Parameters
    ----------
    objective : callable
        Objective on normalised coordinates; infinity for infeasible trials.

    Returns
    -------
    tuple
        Best coordinate, objective value and convergence flag; infinity if all invalid.
    """
    samples = np.linspace(0.0, 1.0, _SCAN_POINTS)
    values = np.array([objective(float(sample)) for sample in samples])
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0, math.inf, False
    best = int(np.argmin(values))
    coordinate, value, success = float(samples[best]), float(values[best]), False
    for centre in np.flatnonzero(finite):
        left, right = max(int(centre) - 1, 0), min(int(centre) + 1, _SCAN_POINTS - 1)
        left = left if finite[left] else int(centre)
        right = right if finite[right] else int(centre)
        if left == right or values[centre] > min(values[left], values[right]):
            continue
        # At fixed angles and material, prism face/path distances are affine in R,H.
        # Their positive half-spaces form a convex feasible region; two feasible
        # endpoints therefore bracket no invalid geometry. Never bridge an invalid
        # scan sample. A very narrow feasible interval can still be missed by a scan.
        found = minimize_scalar(
            objective,
            bounds=(samples[left], samples[right]),
            method="bounded",
            options={"xatol": _REFINEMENT},
        )
        if int(centre) == best:
            success = bool(found.success)
        if float(found.fun) < value:
            coordinate, value, success = (
                float(found.x),
                float(found.fun),
                bool(found.success),
            )
    return coordinate, value, success


def _pair_edge(
    objective: Callable[[Sequence[float]], float],
    fixed_axis: int,
    edge: float,
    variable: float,
) -> float:
    """Evaluate one fixed edge of the normalised motor box.

    Parameters
    ----------
    objective : callable
        Objective on two normalised coordinates.
    fixed_axis : int
        Index of the coordinate held fixed on this edge, zero or one.
    edge : float
        Fixed normalised coordinate, zero or one.
    variable : float
        Free coordinate on the fixed boundary, in [0, 1].

    Returns
    -------
    float
        Objective on the fixed edge.
    """
    point = [edge, variable] if fixed_axis == 0 else [variable, edge]
    return objective(point)


def _pair_search(
    objective: Callable[[Sequence[float]], float], names: Sequence[str]
) -> tuple[tuple[float, ...], bool, str]:
    """Search one or two normalised stage coordinates including their boundaries.

    Parameters
    ----------
    objective : callable
        Objective on normalised coordinates; infinity for infeasible trials.
    names : sequence of str
        Names of the one or two fitted stage coordinates.

    Returns
    -------
    tuple
        Normalised coordinates, convergence flag and boundary/search message.
    """
    if len(names) == 1:
        point, value, success = _feasible_scan(lambda value: objective([value]))
        coordinates = (point,)
        algorithm = "feasible scan and bounded refinement"
    else:
        axis = np.linspace(0.0, 1.0, 17)
        scan = [(float(x), float(y)) for x in axis for y in axis]
        values = np.array([objective(point) for point in scan])
        feasible = np.flatnonzero(np.isfinite(values))
        if not feasible.size:
            raise ProcessingError(
                "no feasible compressor setting found in the search box"
            )
        best = int(np.argmin(values))
        coordinates, value, success = scan[best], float(values[best]), True
        # Deterministic dimensionless population, with feasible seeds even when most
        # of the motor box misses a prism. No worker pool or shared random state.
        population = np.random.default_rng(0).random((64, 2))
        seeds = feasible[np.argsort(values[feasible])[:16]]
        population[: len(seeds)] = np.array([scan[int(index)] for index in seeds])
        found = differential_evolution(
            lambda point: objective(point.tolist()),
            [(0.0, 1.0), (0.0, 1.0)],
            seed=0,
            init=population,
            tol=1e-9,
            atol=1e-11,
            maxiter=300,
            polish=False,
        )
        success = bool(found.success)
        if float(found.fun) < value:
            coordinates = (float(found.x[0]), float(found.x[1]))
            value, success = float(found.fun), bool(found.success)
        # DE does not sample edges exactly. Refine each edge separately and keep its
        # feasible optimum, including corners, so constrained optima are reported.
        for fixed_axis in (0, 1):
            for edge in (0.0, 1.0):
                point, trial, converged = _feasible_scan(
                    partial(_pair_edge, objective, fixed_axis, edge)
                )
                if trial < value:
                    coordinates = (edge, point) if fixed_axis == 0 else (point, edge)
                    value, success = trial, converged
        algorithm = "feasibility scan, seeded differential evolution and bounded edges"
    if not math.isfinite(value):
        raise ProcessingError("no feasible compressor setting found in the search box")
    for name, coordinate in zip(names, coordinates, strict=True):
        if coordinate < _BOUND_TOLERANCE or 1.0 - coordinate < _BOUND_TOLERANCE:
            side = "lower" if coordinate < _BOUND_TOLERANCE else "upper"
            return coordinates, False, f"{name} lies on its {side} bound; {algorithm}"
    message = f"{algorithm}; {'converged' if success else 'iteration limit reached'}"
    return coordinates, success, message


def compress_gratings(  # noqa: PLR0913 - physical settings, input forms and selection
    output: Output | None = None,
    *,
    pair: GratingPair,
    separation_range: tuple[float, float],
    grid: TimeGrid | None = None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None = None,
    plane: int = -1,
    modes: ModeChoice = None,
    bandpass: spectra.Bandpass | None = None,
    oversampling: int = 2,
) -> Compression:
    """Find the normal grating separation that maximises selected pulse peak power.

    Scan and refine the explicit motor range with every other pair setting fixed.
    A fixed pair band and any additional bandpass select the pulse before all metrics.
    An optimum on a bound returns useful settings with ``success=False``.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        Orthogonal modal output; alternative to `grid` and `spectrum` together.
    pair : GratingPair
        Fixed grating density, angles, order, reference, pass count and working band.
    separation_range : tuple of float
        Finite, nonnegative, strictly ascending normal separations in m.
    grid : solprop.grid.TimeGrid or None
        Grid for a raw stored spectrum.
    spectrum : numpy.ndarray or None
        Internal-convention orthogonal power amplitudes, shape ``(n_omega,)`` or
        ``(n_modes, n_pol, n_omega)``. Coordinate power must equal physical power.
    plane : int, optional
        Output plane index, negative from the end.
    modes : int, sequence of int or None, optional
        Modes whose powers are summed; all by default, including both polarisations.
    bandpass : tuple of float, Passband, numpy.ndarray or None, optional
        Additional fixed spectral window, as in the existing compressors.
    oversampling : int, optional
        Positive integer temporal resampling factor for every trial, two by default.

    Returns
    -------
    Compression
        Fitted :class:`~solprop.optics.GratingPair` in ``transfer``, selected-pulse
        metrics, evaluation count and boundary/convergence status.

    Raises
    ------
    ProcessingError
        If bounds, inputs or selection are invalid, the output is spatial or general
        modal, or no feasible setting is found by the scan.
    MaterialError
        If the optical or material inputs are invalid.
    """
    low, high = _pair_range(separation_range, "separation_range")
    pulse = _pair_pulse(
        output,
        grid,
        spectrum,
        pair=pair,
        plane=plane,
        modes=modes,
        bandpass=bandpass,
        oversampling=oversampling,
    )

    def element(coordinates: Sequence[float]) -> GratingPair:
        """Map a normalised separation to its physical grating pair.

        Parameters
        ----------
        coordinates : sequence of float
            Normalised coordinates in the unit search interval or square.

        Returns
        -------
        GratingPair or PrismPair
            Pair at the requested physical stage position.
        """
        return replace(pair, separation=low + (high - low) * coordinates[0])

    coordinates, success, message = _pair_search(
        lambda point: _pair_trial(pulse, element, point), ["separation"]
    )
    return pulse.result(
        "gratings", element(coordinates), success=success, message=message
    )


def compress_prisms(  # noqa: PLR0913 - physical settings, input forms and selection
    output: Output | None = None,
    *,
    pair: PrismPair,
    insertion_range: tuple[float, float] | None = None,
    separation_range: tuple[float, float] | None = None,
    grid: TimeGrid | None = None,
    spectrum: Complex[np.ndarray, "*shape n_omega"] | None = None,
    plane: int = -1,
    modes: ModeChoice = None,
    bandpass: spectra.Bandpass | None = None,
    oversampling: int = 2,
) -> Compression:
    """Tune second-prism insertion, separation, or both for maximum pulse peak power.

    Keep the first prism, incidence, apex angle, material, reference, pass count and
    working band fixed. Omit a coordinate range to hold that coordinate fixed. The
    search is deterministic; an interior result is not a global-optimality certificate.
    Narrow spectra may not uniquely determine both coordinates.

    Parameters
    ----------
    output : solprop.output.schema.Output or None
        Orthogonal modal output; alternative to `grid` and `spectrum` together.
    pair : PrismPair
        Initial feasible pair and the settings held fixed during the search.
    insertion_range : tuple of float or None, optional
        Strictly ascending finite nonnegative H bounds in m, perpendicular to the base.
    separation_range : tuple of float or None, optional
        Strictly ascending finite nonnegative R bounds in m, along the reference ray.
        At least one of the two ranges is required.
    grid : solprop.grid.TimeGrid or None
        Grid for a raw stored spectrum.
    spectrum : numpy.ndarray or None
        Internal-convention orthogonal power amplitudes, shape ``(n_omega,)`` or
        ``(n_modes, n_pol, n_omega)``. Coordinate power must equal physical power.
    plane : int, optional
        Output plane index, negative from the end.
    modes : int, sequence of int or None, optional
        Modes whose powers are summed; all by default, including both polarisations.
    bandpass : tuple of float, Passband, numpy.ndarray or None, optional
        Additional fixed spectral window, applied before all reported pulse metrics.
    oversampling : int, optional
        Positive integer temporal resampling factor for every trial, two by default.

    Returns
    -------
    Compression
        Fitted :class:`~solprop.optics.PrismPair` in ``transfer`` and selected-pulse
        metrics. A best setting on a bound has ``success=False``. Geometry failures
        count as infeasible trials; the selected band never changes with the trial.

    Raises
    ------
    ProcessingError
        If no range is supplied, bounds or inputs are invalid, the output is spatial
        or general modal, selection is empty, or no feasible setting is found.
    MaterialError
        If an optical input or material model is invalid; material errors propagate.
    """
    names, bounds = [], []
    for name, interval in (
        ("insertion", insertion_range),
        ("separation", separation_range),
    ):
        if interval is not None:
            names.append(name)
            bounds.append(_pair_range(interval, f"{name}_range"))
    if not names:
        raise ProcessingError("give at least one insertion_range or separation_range")
    pulse = _pair_pulse(
        output,
        grid,
        spectrum,
        pair=pair,
        plane=plane,
        modes=modes,
        bandpass=bandpass,
        oversampling=oversampling,
    )

    def element(coordinates: Sequence[float]) -> PrismPair:
        """Move only the requested second-prism stage coordinates.

        Parameters
        ----------
        coordinates : sequence of float
            Normalised coordinates in the unit search interval or square.

        Returns
        -------
        GratingPair or PrismPair
            Pair at the requested physical stage position.
        """
        settings = {
            name: low + (high - low) * coordinate
            for name, (low, high), coordinate in zip(
                names, bounds, coordinates, strict=True
            )
        }
        return replace(pair, geometry=replace(pair.geometry, **settings))

    coordinates, success, message = _pair_search(
        lambda point: _pair_trial(pulse, element, point), names
    )
    return pulse.result(
        "prisms", element(coordinates), success=success, message=message
    )
