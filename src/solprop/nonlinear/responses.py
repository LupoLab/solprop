# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Nonlinear response definitions and their declared traits.

A response says how a medium answers a field: the nonlinear polarisation ``P(t)`` it
produces from the field ``E(t)`` at a point, and the declared traits the translation
layer needs in order to compile a set of responses into one fused time-domain pass.
This module holds the Kerr and plasma responses, SEP 6's frequency-dependent electronic
response (:class:`DispersiveKerrResponse`), and the trait vocabulary the delayed Raman
response of :mod:`solprop.nonlinear.raman` extends.

The ``evaluate`` method here is the **NumPy reference**: the oracle every backend kernel
is tested against, written for clarity rather than for speed
(``docs/dev/architecture.md`` section 3). Units, signs and the analytic-signal
convention follow ``docs/dev/conventions.md`` sections 2 and 4, and the derivation of
every numerical factor is in ``docs/guide/media/nonlinear.md``.
"""

import logging
import math
from dataclasses import KW_ONLY, dataclass
from typing import Literal

import numpy as np
from jaxtyping import Bool, Complex, Float, Num

from solprop import fourier
from solprop.constants import CHARGE_MASS_RATIO, EPS0
from solprop.exceptions import GridError, MaterialError
from solprop.nonlinear.ionisation import (
    EvaluableRate,
    IntegrationPolicy,
    RunningIntegral,
    bind_rate_energy,
    ionisation_fraction,
)
from solprop.physdata import Gas, as_gas, ionisation_potential
from solprop.physdata.mixtures import check_preionisation
from solprop.physdata.nonlinear_data import DispersiveKerrModel
from solprop.physdata.species import Species
from solprop.profiles import ZProfile

__all__ = [
    "MAX_N_POL",
    "DispersiveKerrResponse",
    "DispersiveWeights",
    "Domain",
    "Evaluator",
    "KerrResponse",
    "PlasmaResponse",
    "Polarisation",
    "ResponseTraits",
    "Sector",
    "dispersive_cubic",
    "mixture_filter",
    "thg_carrier",
]

logger = logging.getLogger(__name__)

Domain = Literal["pointwise", "memory", "convolution", "spectral"]
"""How a response is evaluated: sample by sample, with memory along t, as a convolution,
or after the forward transform (``docs/dev/architecture.md`` section 6.6)."""

Polarisation = Literal["scalar", "vector_isotropic", "tensor"]
"""How a response treats the polarisation axis."""

Evaluator = Literal["symmetric", "polynomial"]
"""SEP 6's two dispersive evaluators: the symmetric factorisation of the Kerr diagonal,
and the gas dispersion polynomial itself."""

Sector = Literal["full", "conserving"]
"""Which quartets a dispersive cubic keeps: all of them, or the number-conserving ones
with two positive and two negative signed legs (SEP 6, Phase 0, decision Q4)."""

MAX_N_POL = 2
"""Largest polarisation axis a response accepts: ``(x, y)`` (``conventions.md`` 5)."""

_PREIONISATION_PATH = (
    "this plasma response carries its preionisation as a path along z, and the "
    "reference evaluation sees one plane; evaluate "
    "dataclasses.replace(response, preionisation=<the value at that plane>)"
)


@dataclass(frozen=True)
class ResponseTraits:
    """What the translation layer must know about a response to compile it.

    The four declared traits (see ``docs/dev/decisions.md``). They are data, not
    behaviour: the compiler reads them to choose a fused traversal, and a response that
    lies about them produces a wrong kernel rather than an error.

    Attributes
    ----------
    domain : {'pointwise', 'memory', 'convolution', 'spectral'}
        How the response is evaluated along the time axis.
    grids : tuple of {'field', 'envelope'}
        The spectral layouts the response is defined on.
    polarisation : {'scalar', 'vector_isotropic', 'tensor'}
        How the response couples the polarisation components.
    z_profiles : tuple of str
        Names of the scalar paths along z the response consumes, for example
        ``("density",)``. Empty for a response that does not vary with ``z``.

    Examples
    --------
    >>> from solprop.nonlinear.responses import ResponseTraits
    >>> traits = ResponseTraits("pointwise", ("field",), "scalar", ())
    >>> traits.domain, traits.grids
    ('pointwise', ('field',))
    """

    domain: Domain
    grids: tuple[fourier.GridKind, ...]
    polarisation: Polarisation
    z_profiles: tuple[str, ...]


def thg_carrier(
    omega0: float, t: Float[np.ndarray, " n_to"]
) -> Complex[np.ndarray, " n_to"]:
    """Return the carrier factor an envelope-grid third-harmonic term needs.

    On an envelope grid the state is ``A(t) = E_a(t) exp(-i omega0 t)``
    (``docs/dev/conventions.md`` section 4), so the part of the cubic response that
    radiates near ``3 omega0`` carries an explicit ``exp(2 i omega0 t)`` to place it at
    the right offset from the carrier. On a field grid nothing is needed: the harmonic
    is already in the cubic.

    The ``+2i`` sign follows from ``E_a = A exp(+i omega0 t)`` in the internal FFT
    convention and is derived in ``docs/guide/media/nonlinear.md``; it depends on the
    envelope convention, so it is derived here rather than transcribed.

    Parameters
    ----------
    omega0 : float
        Carrier angular frequency, rad/s, positive.
    t : numpy.ndarray
        Time samples of the fine grid, s, shape ``(n_to,)``.

    Returns
    -------
    numpy.ndarray
        ``exp(2 i omega0 t)``, complex, shape ``(n_to,)``.

    Raises
    ------
    GridError
        If ``omega0`` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.responses import thg_carrier
    >>> carrier = thg_carrier(1.0, np.array([0.0, np.pi / 2]))
    >>> bool(np.allclose(carrier, [1.0, -1.0]))
    True
    """
    if not omega0 > 0.0:
        msg = f"omega0 must be positive, got {omega0!r}"
        raise GridError(msg)
    return np.exp(2j * omega0 * np.asarray(t, dtype=float))


def _check_evaluate(
    values: np.ndarray,
    kind: str,
    carrier: np.ndarray | None,
    *,
    needs_carrier: bool,
    max_pol: int = MAX_N_POL,
) -> None:
    """Validate the arguments of :meth:`KerrResponse.evaluate`.

    ``kind`` is a plain string rather than :data:`solprop.fourier.GridKind` for the
    reason given in :func:`solprop.fourier._check_kind`: this is the run-time guard, and
    the test session's ``beartype`` hook would otherwise reject a bad value first.

    Parameters
    ----------
    values : numpy.ndarray
        The samples passed as ``field_t``.
    kind : str
        Layout name to check.
    carrier : numpy.ndarray or None
        The carrier passed by the caller, if any.
    needs_carrier : bool
        Whether this response and layout require one.
    max_pol : int, optional
        The most field components the response takes: two for a transverse field, three
        for the general modal route's full vector field. Default :data:`MAX_N_POL`.

    Raises
    ------
    GridError
        If the layout is unknown, the array shape or dtype does not match the layout,
        or the carrier is missing, unwanted or the wrong length.
    """
    if kind not in ("field", "envelope"):
        msg = f"kind must be 'field' or 'envelope', got {kind!r}"
        raise GridError(msg)
    if values.ndim != 2:
        msg = f"field_t must have shape (n_pol, n_to), got {values.shape}"
        raise GridError(msg)
    if not 1 <= values.shape[0] <= max_pol:
        allowed = "1 or 2" if max_pol == MAX_N_POL else f"between 1 and {max_pol}"
        msg = f"n_pol must be {allowed}, got {values.shape[0]}"
        raise GridError(msg)
    if kind == "field" and np.iscomplexobj(values):
        msg = "a field grid carries the real field; got a complex array"
        raise GridError(msg)
    if kind == "envelope" and not np.iscomplexobj(values):
        msg = "an envelope grid carries a complex envelope; got a real array"
        raise GridError(msg)
    if needs_carrier and carrier is None:
        msg = (
            "an envelope-grid Kerr response with thg=True needs the carrier "
            "exp(2i omega0 t); build it with thg_carrier"
        )
        raise GridError(msg)
    if carrier is None:
        return
    if not needs_carrier:
        msg = f"carrier is only used on an envelope grid with thg=True, not {kind!r}"
        raise GridError(msg)
    if carrier.shape != values.shape[-1:]:
        msg = f"carrier has shape {carrier.shape}, expected {values.shape[-1:]}"
        raise GridError(msg)


@dataclass(frozen=True)
class KerrResponse:
    """The instantaneous third-order (Kerr) polarisation of an isotropic medium.

    ``P = eps0 chi3 rho x (a cubic form of the field)``, on either grid, for one or two
    polarisation components. The cubic form differs between the grids because a field
    grid resolves the carrier and an envelope grid does not; the four cases and their
    numerical factors are derived in ``docs/guide/media/nonlinear.md``.

    ``chi3`` is the susceptibility **per unit of the density argument**, so a gas passes
    the per-particle hyperpolarisability with its number density in m^-3 and a fixed
    medium passes chi3 itself with a density of 1.0. This is the convention the kernel
    contract's ``density`` profile assumes.

    Attributes
    ----------
    chi3 : float
        Third-order susceptibility per unit density, m^2/V^2 when the density is
        dimensionless, m^5/V^2 when it is a number density in m^-3.
    thg : bool, optional
        Whether the response radiates the third harmonic. On a field grid ``True`` is
        the full cubic and ``False`` removes the harmonic; on an envelope grid ``True``
        adds the harmonic term, which needs a grid built with ``thg=True`` and the
        carrier from :func:`thg_carrier`. Default ``True``.
    species : str or None, optional
        Component label in a mixture; ``None`` retains the single-medium spelling.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.constants import EPS0
    >>> from solprop.nonlinear.responses import KerrResponse
    >>> kerr = KerrResponse(chi3=1.0, thg=True)
    >>> kerr.traits.domain
    'pointwise'
    >>> field = np.array([[1.0, 2.0]])
    >>> polarisation = kerr.evaluate(field, 1.0, kind="field")
    >>> bool(np.array_equal(polarisation, EPS0 * np.array([[1.0, 8.0]])))
    True
    """

    chi3: float
    thg: bool = True
    species: str | None = None

    @property
    def traits(self) -> ResponseTraits:
        """Return the declared traits of the Kerr response.

        Returns
        -------
        ResponseTraits
            Pointwise, both grids, vector-isotropic, consuming the density path.

        Examples
        --------
        >>> from solprop.nonlinear.responses import KerrResponse
        >>> KerrResponse(1.0).traits.z_profiles
        ('density',)
        """
        return ResponseTraits(
            domain="pointwise",
            grids=("field", "envelope"),
            polarisation="vector_isotropic",
            z_profiles=(
                "density" if self.species is None else f"density:{self.species}",
            ),
        )

    def evaluate(
        self,
        field_t: Num[np.ndarray, "n_pol n_to"],
        density: float,
        *,
        kind: fourier.GridKind,
        carrier: Complex[np.ndarray, " n_to"] | None = None,
    ) -> Num[np.ndarray, "n_pol n_to"]:
        """Return the Kerr polarisation on the fine time grid.

        The reference implementation, and therefore the oracle for every backend. It
        allocates and is not written for speed.

        Parameters
        ----------
        field_t : numpy.ndarray
            The field on the fine time grid, shape ``(n_pol, n_to)``: real ``E(t)`` in
            V/m on a field grid, complex ``A(t)`` in V/m on an envelope grid.
        density : float
            Density multiplying ``chi3``: a number density in m^-3 for a gas, 1.0 for a
            medium whose ``chi3`` is already absolute. Non-negative.
        kind : {'field', 'envelope'}
            The spectral layout the samples belong to.
        carrier : numpy.ndarray or None, optional
            ``exp(2 i omega0 t)`` on the fine grid, shape ``(n_to,)``; required when
            ``kind='envelope'`` and ``thg`` is true, rejected otherwise. Build it with
            :func:`thg_carrier`.

        Returns
        -------
        numpy.ndarray
            The polarisation in C/m^2, shape ``(n_pol, n_to)``, real on a field grid and
            complex on an envelope grid.

        Raises
        ------
        GridError
            If the layout is unknown, the array shape or dtype does not match it, or the
            carrier is missing, unwanted or the wrong length.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.constants import EPS0
        >>> from solprop.nonlinear.responses import KerrResponse
        >>> envelope = np.array([[2.0 + 0.0j]])
        >>> out = KerrResponse(chi3=1.0, thg=False).evaluate(
        ...     envelope, 1.0, kind="envelope"
        ... )
        >>> bool(np.allclose(out, 0.75 * EPS0 * 8.0))
        True
        """
        values = np.asarray(field_t)
        needs_carrier = kind == "envelope" and self.thg
        carrier_values = None if carrier is None else np.asarray(carrier)
        _check_evaluate(values, kind, carrier_values, needs_carrier=needs_carrier)
        if density < 0.0:
            msg = f"density must be non-negative, got {density!r}"
            raise GridError(msg)

        factor = density * EPS0 * self.chi3
        if kind == "field" and self.thg:
            # The full cubic already carries the third harmonic and the counter-
            # rotating terms; squares rather than moduli because E is real.
            return factor * _sum_of_squares(values) * values

        analytic = fourier.analytic_signal(values) if kind == "field" else values
        out = factor * _fundamental_cubic(analytic)
        # The validation above accepts a carrier exactly when one is needed, so this
        # test is `needs_carrier` and also narrows the optional away for the checker.
        if carrier_values is not None:
            harmonic = 0.25 * _sum_of_squares(analytic) * analytic
            out = out + factor * carrier_values * harmonic
        # A view of the real part is not contiguous; the seam and the backends expect a
        # plain array, so materialise it here rather than at every call site.
        return np.ascontiguousarray(out.real) if kind == "field" else out


@dataclass(frozen=True, eq=False)
class DispersiveWeights:
    """The spectral multipliers of one species' dispersive cubic on a fine grid.

    Every multiplier is a real, even function of the **absolute** angular frequency of
    each bin of the fine transform's spectrum: the one-sided ``rfft`` layout,
    ``(n_to // 2 + 1,)``, on a field grid, and the full FFT-order layout, ``(n_to,)``,
    on an envelope grid. The compiled bundle of the translation layer carries the same
    arrays, one row per species
    (:class:`solprop.translate.responses.DispersiveKerrTerms`).

    Attributes
    ----------
    evaluator : {'symmetric', 'polynomial'}
        Which evaluator the arrays belong to.
    positive : numpy.ndarray
        ``Pi_+``: the bins of strictly positive absolute frequency, bool. On a field
        grid the DC and the Nyquist bins are excluded.
    filter : numpy.ndarray
        The input multiplier, float64: the symmetric factor ``g`` (dimensionless), or
        the polynomial's bounded leg weight ``d`` in m^-2.
    output : numpy.ndarray or None
        The polynomial's two output multipliers, float64, shape
        ``(2, n_omega_fine)``: ``1 + a d + (b + b') d^2`` (dimensionless) and
        ``3 a + 6 b d`` (m^2). ``None`` for the symmetric evaluator, whose output
        multiplier is ``filter`` again.
    products : tuple of float
        The polynomial's constant weights of ``E^2 D^2 E`` and ``E (D E)^2``,
        ``3 (b + b')`` and ``6 b`` in m^4; ``(0, 0)`` for the symmetric evaluator.
    """

    evaluator: Evaluator
    positive: Bool[np.ndarray, " n_omega_fine"]
    filter: Float[np.ndarray, " n_omega_fine"]
    output: Float[np.ndarray, "2 n_omega_fine_out"] | None = None
    products: tuple[float, float] = (0.0, 0.0)


def _isotropic(
    x: Num[np.ndarray, "*nodes n_pol n_to"],
    y: Num[np.ndarray, "*nodes n_pol n_to"],
    z: Num[np.ndarray, "*nodes n_pol n_to"],
) -> Num[np.ndarray, "*nodes n_pol n_to"]:
    """Return the isotropic Kleinman contraction of three fields.

    ``T_ijkl x_j y_k z_l`` with ``T = (d_ij d_kl + d_ik d_jl + d_il d_jk) / 3``,
    contracted over the polarisation axis without conjugation; ``x y z`` for one
    component and ``(E . E) E`` when the three coincide (SEP 6, Phase 0, decision Q3).

    Parameters
    ----------
    x, y, z : numpy.ndarray
        Three fields of one shape, the polarisation axis second to last.

    Returns
    -------
    numpy.ndarray
        The contraction, of the same shape.
    """
    if x.shape[-2] == 1:
        return x * y * z
    xy = np.sum(x * y, axis=-2, keepdims=True)
    xz = np.sum(x * z, axis=-2, keepdims=True)
    yz = np.sum(y * z, axis=-2, keepdims=True)
    return (xy * z + xz * y + yz * x) / 3.0


def _filtered(
    values: Num[np.ndarray, "*nodes n_pol n_to"],
    multiplier: Float[np.ndarray, " n_omega_fine"],
    kind: str,
) -> Num[np.ndarray, "*nodes n_pol n_to"]:
    """Apply a spectral multiplier to a field on the fine time grid.

    Parameters
    ----------
    values : numpy.ndarray
        A real field (field grid) or a complex envelope (envelope grid).
    multiplier : numpy.ndarray
        Its multiplier on the transform's spectral layout.
    kind : {'field', 'envelope'}
        The layout.

    Returns
    -------
    numpy.ndarray
        The filtered field, of the input's dtype.
    """
    n_to = values.shape[-1]
    if kind == "field":
        return np.fft.irfft(np.fft.rfft(values, axis=-1) * multiplier, n=n_to, axis=-1)
    return np.fft.ifft(np.fft.fft(values, axis=-1) * multiplier, axis=-1)


def _analytic(
    values: Float[np.ndarray, "*nodes n_pol n_to"],
    positive: Bool[np.ndarray, " n_omega_fine"],
) -> Complex[np.ndarray, "*nodes n_pol n_to"]:
    """Return the analytic signal of a real field: ``Pi_+`` doubled, DC and Nyquist out.

    Parameters
    ----------
    values : numpy.ndarray
        A real field on the fine time grid.
    positive : numpy.ndarray
        The one-sided bins of strictly positive frequency.

    Returns
    -------
    numpy.ndarray
        ``u`` with ``Re u = E`` wherever the field has no DC or Nyquist content.
    """
    n_to = values.shape[-1]
    spectrum = np.zeros((*values.shape[:-1], n_to), dtype=np.complex128)
    spectrum[..., : positive.shape[0]] = 2.0 * np.fft.rfft(values, axis=-1) * positive
    return np.fft.ifft(spectrum, axis=-1)


def _product_spectrum(  # noqa: PLR0913 - the three fields of a product and its context
    x: Num[np.ndarray, "*nodes n_pol n_to"],
    y: Num[np.ndarray, "*nodes n_pol n_to"],
    z: Num[np.ndarray, "*nodes n_pol n_to"],
    positive: Bool[np.ndarray, " n_omega_fine"],
    *,
    kind: str,
    sector: str,
    carrier: Complex[np.ndarray, " n_to"] | None,
    window: Float[np.ndarray, " n_to"] | None = None,
) -> Complex[np.ndarray, "*nodes n_pol n_omega_fine"]:
    """Return the spectrum of one sector of the trilinear product of three fields.

    SEP 6, Phase 0: with analytic signals ``x, y, z`` of real fields ``X, Y, Z``, the
    analytic polarisation of ``X Y Z`` is ``(1/4) Pi_+ [x y z + (x y z* + x y* z +
    x* y z) + (x y* z* + x* y z* + x* y* z)]``, the groups being the quartets with
    three, two and one positive signed legs. The full cubic keeps all three, and on a
    field grid it is the real product itself; the number-conserving sector keeps the
    middle group. On an envelope frame ``A = u exp(-i omega0 t)`` the first group
    carries ``exp(+2 i omega0 t)`` and the third ``exp(-2 i omega0 t)``, and ``Pi_+``
    acts on absolute frequency.

    Parameters
    ----------
    x, y, z : numpy.ndarray
        The three fields on the fine time grid: real on a field grid, envelopes on an
        envelope grid.
    positive : numpy.ndarray
        ``Pi_+`` on the transform's spectral layout.
    kind : {'field', 'envelope'}
        The layout.
    sector : {'full', 'conserving'}
        Which quartets are kept.
    carrier : numpy.ndarray or None
        ``exp(2 i omega0 t)``, required for the full sector on an envelope grid.
    window : numpy.ndarray or None, optional
        The temporal collar, multiplying the product in time before its transform,
        ahead of ``Pi_+`` and every output multiplier; ``None`` is a unit collar.

    Returns
    -------
    numpy.ndarray
        The product's spectrum on the transform's layout: the one-sided spectrum of
        the real polarisation on a field grid, the envelope of the analytic
        polarisation on an envelope grid.
    """
    collar = 1.0 if window is None else window
    if kind == "field" and sector == "full":
        return np.fft.rfft(collar * _isotropic(x, y, z), axis=-1)
    if kind == "field":
        a, b, c = (_analytic(series, positive) for series in (x, y, z))
    else:
        a, b, c = x, y, z
    inner = (
        _isotropic(a, b, c.conj())
        + _isotropic(a, b.conj(), c)
        + _isotropic(a.conj(), b, c)
    )
    if kind == "field":
        # Re[(1/4) ifft(Pi_+ S)] is the inverse real transform of S/8 on the positive
        # bins: the one-sided spectrum counts each positive bin twice.
        n_omega = positive.shape[0]
        return np.fft.fft(collar * inner, axis=-1)[..., :n_omega] * positive / 8.0
    if sector == "full":
        assert carrier is not None
        up = _isotropic(a, b, c)
        down = (
            _isotropic(a, b.conj(), c.conj())
            + _isotropic(a.conj(), b, c.conj())
            + _isotropic(a.conj(), b.conj(), c)
        )
        inner = inner + carrier * up + carrier.conj() * down
    return np.fft.fft(collar * inner, axis=-1) * positive / 4.0


def dispersive_cubic(
    field_t: Num[np.ndarray, "*nodes n_pol n_to"],
    weights: DispersiveWeights,
    *,
    kind: str,
    sector: str,
    carrier: Complex[np.ndarray, " n_to"] | None = None,
    window: Float[np.ndarray, " n_to"] | None = None,
) -> Num[np.ndarray, "*nodes n_pol n_to"]:
    r"""Return SEP 6's dispersive cubic of a field, per unit ``eps0 gamma3_0 rho``.

    The forms Phase 0 derived and holds to an explicit triplet sum. The **symmetric**
    evaluator is one product of the filtered field, ``G[(G E)^3]``, with ``G`` the
    multiplier ``g``. The **polynomial** evaluator is the grouped sum

    .. math:: [1 + a\tilde d + (b + b')\tilde d^2]\,E^3
              + [3a + 6b\tilde d]\,E^2\tilde DE
              + 3(b + b')\,E^2\tilde D^2E + 6b\,E(\tilde DE)^2,

    each bracket an output multiplier, with ``D~`` the bounded leg weight. A product of
    three fields is the isotropic Kleinman contraction of ``_isotropic``, and the
    sector restricts it to the quartets kept (``_product_spectrum``). The output
    multipliers commute with ``Pi_+`` and with taking the real part, so every product's
    spectrum is weighted and summed before one inverse transform.

    The temporal collar ``w`` multiplies each product in time, before ``Pi_+`` and the
    output multipliers: ``G[w (G E)^3]``, and the polynomial's output multipliers act
    on the windowed products (decided by John Travers, 2026-10-08, SEP 6 Phase 2a,
    open question 1). With a unit collar this is the operator without one; the order
    matters only where the collar varies.

    The products are formed on the fine grid, which must hold the cubic's support for
    the result to be the convolution it stands for: the same oversampling rule as the
    constant cubic's (``conventions.md`` section 3).

    Parameters
    ----------
    field_t : numpy.ndarray
        The field on the fine time grid, shape ``(..., n_pol, n_to)``: real ``E(t)``
        on a field grid, the complex envelope ``A(t)`` on an envelope grid, in V/m.
    weights : DispersiveWeights
        The species' multipliers on the fine grid's spectral layout.
    kind : {'field', 'envelope'}
        The spectral layout.
    sector : {'full', 'conserving'}
        Which quartets are kept.
    carrier : numpy.ndarray or None, optional
        ``exp(2 i omega0 t)`` on the fine grid; required for the full sector on an
        envelope grid and ignored otherwise.
    window : numpy.ndarray or None, optional
        The temporal collar ``grids.t_window_fine`` in ``[0, 1]``, shape ``(n_to,)``;
        ``None`` is a unit collar.

    Returns
    -------
    numpy.ndarray
        The cubic in V^3/m^3 (times m^0: every multiplier is normalised by the static
        coefficient), real on a field grid and complex on an envelope grid, of the
        input's shape.
    """
    values = np.asarray(field_t)
    n_to = values.shape[-1]
    positive = weights.positive

    def product(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> Complex[np.ndarray, "*nodes n_pol n_omega_fine"]:
        """Return one product's spectrum in this call's sector and layout.

        Parameters
        ----------
        x, y, z : numpy.ndarray
            The three fields of the product.

        Returns
        -------
        numpy.ndarray
            Its spectrum (:func:`_product_spectrum`).
        """
        return _product_spectrum(
            x, y, z, positive, kind=kind, sector=sector, carrier=carrier, window=window
        )

    if weights.evaluator == "symmetric":
        filtered = _filtered(values, weights.filter, kind)
        spectrum = product(filtered, filtered, filtered) * weights.filter
    else:
        assert weights.output is not None
        first = _filtered(values, weights.filter, kind)
        second = _filtered(first, weights.filter, kind)
        spectrum = (
            product(values, values, values) * weights.output[0]
            + product(values, values, first) * weights.output[1]
            + weights.products[0] * product(values, values, second)
            + weights.products[1] * product(values, first, first)
        )
    if kind == "field":
        return np.fft.irfft(spectrum, n=n_to, axis=-1)
    return np.fft.ifft(spectrum, axis=-1)


def mixture_filter(
    filters: Float[np.ndarray, "n_species n_omega_fine"],
    factors: Float[np.ndarray, " n_species"],
) -> tuple[float, Float[np.ndarray, " n_omega_fine"]]:
    r"""Return the symmetric evaluator's filter of a gas mixture and its coefficient.

    SEP 6, section 6, adopted for the symmetric evaluator of a mixture (decided by John
    Travers, 2026-10-08, Phase 2b, open question 3): one filter from the mixture's own
    Kerr diagonal, :math:`\gamma_{K,\rm mix}(\nu) = \sum_s N_s\gamma_{K,s}(\tilde
    \nu_s)`, factorised exactly as section 4.4 factorises a pure gas's. With
    :math:`f_s = \epsilon_0\gamma_{3,s} N_s`, each species' static coefficient times its
    density, and :math:`\hat K_s = g_s^4 = \gamma_{K,s}(\tilde\nu_s)/\gamma_{3,s}`, its
    own Kerr diagonal under its own leg bound,

    .. math:: g_{\rm mix} = \Big[\sum_s f_s \hat K_s \Big/ \sum_s f_s\Big]^{1/4},
              \qquad P = \Big(\sum_s f_s\Big)\,G_{\rm mix}[w\,(G_{\rm mix}E)^3].

    On the Kerr diagonal the four legs give
    :math:`(\sum_s f_s) g_{\rm mix}^4 = \sum_s f_s\hat K_s = \epsilon_0\gamma_{K,\rm
    mix}`, the mixture's own degenerate coefficient. The reference is the mixture's
    static coefficient :math:`\sum_s f_s`, so :math:`g_{\rm mix}(0) = 1`; as in section
    4.4 any other positive reference :math:`\gamma_*` cancels, moving a factor between
    the coefficient and :math:`g^4`. Off the diagonal this approximates the exact
    per-species sum :math:`\sum_s f_s G_s[w (G_s E)^3]`, as Phase 2b measured; for one
    species it **is** that species' operator, which this function returns unchanged
    rather than recomputed.

    The filter depends on the composition :math:`f_s / \sum f_s`, not on the total
    density, so a backend rebuilds it wherever the composition changes along z. The
    densities are non-negative (the problem's domain check), so :math:`g_{\rm mix}^4` is
    a convex combination of the species' :math:`\hat K_s`; at a vacuum, where every
    :math:`f_s` is zero, the response is zero whatever the filter, which is then taken
    as one.

    Parameters
    ----------
    filters : numpy.ndarray
        Each species' own symmetric filter :math:`g_s` on one spectral layout,
        dimensionless, shape ``(n_species, n_omega_fine)``.
    factors : numpy.ndarray
        :math:`f_s` per species, in C/V^3 (:attr:`DispersiveKerrResponse.coefficient`
        times the number density in m^-3), non-negative.

    Returns
    -------
    tuple
        ``(sum_s f_s, g_mix)``: the mixture's static coefficient and its filter, shape
        ``(n_omega_fine,)``. One species returns its own factor and filter.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.responses import mixture_filter
    >>> filters = np.array([[1.0, 2.0], [1.0, 1.0]])
    >>> total, g = mixture_filter(filters, np.array([1.0, 1.0]))
    >>> total, float(g[0]), float(g[1] ** 4)
    (2.0, 1.0, 8.5)
    """
    rows = np.asarray(filters, dtype=float)
    weights = np.asarray(factors, dtype=float)
    if rows.shape[0] == 1:
        return float(weights[0]), rows[0]
    total = 0.0
    for weight in weights:
        total += float(weight)
    if not total > 0.0:
        return 0.0, np.ones(rows.shape[1])
    # Summed in species order and rooted as two square roots, as the compiled kernels
    # do: every operation is correctly rounded, so the backends agree to the last bit.
    quartic = np.zeros(rows.shape[1])
    for weight, row in zip(weights, rows, strict=True):
        square = row * row
        quartic = quartic + weight * (square * square)
    return total, np.sqrt(np.sqrt(quartic / total))


@dataclass(frozen=True)
class DispersiveKerrResponse:
    r"""The frequency-dependent electronic third-order response of a gas (SEP 6).

    One broadband polarisation from optmat's whole hyperpolarizability model of the gas,
    ``P = eps0 gamma3_0 rho x (a filtered cubic of the field)``, the kernel evaluated at
    the absolute frequencies of every quartet the cubic couples. No SPM, XPM, FWM or
    THG event is identified: the four-frequency rule assigns each its coefficient.

    Two evaluators (``docs/dev/seps/archive/sep-0006-dispersive-nonlinearity.md``,
    sections 4.4-4.5 and the Phase 0 report):

    ``"polynomial"``
        The gas dispersion polynomial itself,
        :math:`\gamma_0[1 + a S_2 + b S_2^2 + b' S_4]`, by its grouped FFT form:
        exact evaluation of the fit on the resolved grid.
    ``"symmetric"``
        The symmetric factorisation of the Kerr diagonal,
        :math:`\gamma_0\prod_j g(\omega_j)` with
        :math:`g = [\gamma_K(\tilde\nu)/\gamma_0]^{1/4}`: exact on the Kerr diagonal
        and to second order elsewhere, one filtered product.

    Every leg is held at the gas's leg bound :math:`\nu_c` (section 4.10): the weight
    :math:`\tilde d = \min((\omega/2\pi c)^2, \nu_c^2)`, real and even, so the lossless
    work identity holds and no frequency a field occupies is refused.

    ``thg`` selects the sector (Phase 0, decision Q4): ``True`` is the full cubic, all
    quartets, harmonic and inverse harmonic exchange included; ``False`` is the
    number-conserving sector, the quartets with two positive and two negative signed
    legs, with :math:`\Pi_+` on absolute frequency. Below an octave the latter equals
    the constant cubic's ``thg=False`` rule; beyond it, unlike that rule, it does no net
    work. Two polarisation components contract through the isotropic Kleinman tensor,
    the scalar kernel's named vector approximation (decision Q3).

    Attributes
    ----------
    model : solprop.physdata.nonlinear_data.DispersiveKerrModel
        optmat's model of the gas in SI, with its leg bound and band edges.
    evaluator : {'symmetric', 'polynomial'}
        Which evaluator.
    thg : bool, optional
        ``True`` for the full cubic, ``False`` for the number-conserving sector.
        Default ``True``.
    species : str or None, optional
        Component label in a mixture; ``None`` retains the single-medium spelling.

    Raises
    ------
    MaterialError
        If the evaluator is not one of the two.

    Examples
    --------
    >>> from solprop.nonlinear.responses import DispersiveKerrResponse
    >>> from solprop.physdata.nonlinear_data import OptmatKerr
    >>> model = OptmatKerr("Ar").dispersive_model()
    >>> response = DispersiveKerrResponse(model, "polynomial", thg=False)
    >>> response.sector, response.traits.domain
    ('conserving', 'convolution')
    """

    model: DispersiveKerrModel
    # A plain string rather than `Evaluator`, so that the run-time guard below is the
    # one that answers a wrong value, as for `fourier.GridKind` in `_check_evaluate`.
    evaluator: str
    thg: bool = True
    species: str | None = None

    def __post_init__(self) -> None:
        """Check the evaluator.

        Raises
        ------
        MaterialError
            If it is not ``"symmetric"`` or ``"polynomial"``.
        """
        if self.evaluator not in ("symmetric", "polynomial"):
            msg = (
                "a dispersive Kerr response's evaluator is 'symmetric' or "
                f"'polynomial', got {self.evaluator!r}"
            )
            raise MaterialError(msg)

    @property
    def sector(self) -> Sector:
        """Return which quartets the response keeps.

        Returns
        -------
        {'full', 'conserving'}
            ``"full"`` with ``thg``, ``"conserving"`` without.
        """
        return "full" if self.thg else "conserving"

    @property
    def traits(self) -> ResponseTraits:
        """Return the declared traits of the response.

        Returns
        -------
        ResponseTraits
            A convolution in time (spectral multipliers on the inputs and the output),
            both grids, vector-isotropic, consuming the density path.
        """
        return ResponseTraits(
            domain="convolution",
            grids=("field", "envelope"),
            polarisation="vector_isotropic",
            z_profiles=(
                "density" if self.species is None else f"density:{self.species}",
            ),
        )

    @property
    def coefficient(self) -> float:
        """Return ``eps0 gamma3_0``, the static coefficient per unit density.

        Returns
        -------
        float
            In C m^2/V^3 (m^5/V^2 times eps0), multiplying the number density.
        """
        return EPS0 * self.model.gamma3_static

    def weights(
        self, omega: Float[np.ndarray, " n_omega_fine"], *, kind: fourier.GridKind
    ) -> DispersiveWeights:
        """Return the multipliers on a fine grid's spectral layout.

        Parameters
        ----------
        omega : numpy.ndarray
            The **absolute** angular frequency of every bin, rad/s: ``k domega`` on a
            field grid's one-sided layout, ``omega0 + v`` in FFT order on an envelope
            grid.
        kind : {'field', 'envelope'}
            The layout.

        Returns
        -------
        DispersiveWeights
            The response's multipliers there, read-only.
        """
        absolute = np.asarray(omega, dtype=float)
        positive = absolute > 0.0
        if kind == "field":
            # Pi_+ as the triplet sum counts it: the Nyquist bin is a negative
            # frequency of the full transform.
            positive[-1] = False
        weight = self.model.leg_weight(absolute)
        model = self.model
        if self.evaluator == "symmetric":
            return DispersiveWeights(
                "symmetric",
                _frozen(positive),
                _frozen(model.kerr_ratio(weight) ** 0.25),
            )
        output = np.stack(
            [
                1.0 + model.a * weight + (model.b + model.b_prime) * weight**2,
                3.0 * model.a + 6.0 * model.b * weight,
            ]
        )
        return DispersiveWeights(
            "polynomial",
            _frozen(positive),
            _frozen(weight),
            _frozen(output),
            (3.0 * (model.b + model.b_prime), 6.0 * model.b),
        )

    def evaluate(
        self,
        field_t: Num[np.ndarray, "n_pol n_to"],
        density: float,
        *,
        kind: fourier.GridKind,
        omega: Float[np.ndarray, " n_omega_fine"],
        carrier: Complex[np.ndarray, " n_to"] | None = None,
        window: Float[np.ndarray, " n_to_window"] | None = None,
    ) -> Num[np.ndarray, "n_pol n_to"]:
        """Return the polarisation on the fine time grid: the reference evaluation.

        The oracle every backend is tested against, itself held to the triplet sum of
        SEP 6's Phase 0 (``tests/dispersive_kerr_oracle.py``). It allocates and is not
        written for speed.

        Parameters
        ----------
        field_t : numpy.ndarray
            The field on the fine time grid, shape ``(n_pol, n_to)``: real ``E(t)`` in
            V/m on a field grid, the complex envelope ``A(t)`` in V/m on an envelope
            grid. One, two or three components: three is the general modal route's
            full vector field, contracted through the same isotropic tensor.
        density : float
            Number density in m^-3, non-negative.
        kind : {'field', 'envelope'}
            The spectral layout the samples belong to.
        omega : numpy.ndarray
            The absolute angular frequency of every bin of the fine transform in
            rad/s: shape ``(n_to // 2 + 1,)`` on a field grid, ``(n_to,)`` in FFT
            order on an envelope grid (``TimeGrid.omega_fine``).
        carrier : numpy.ndarray or None, optional
            ``exp(2 i omega0 t)`` on the fine grid, shape ``(n_to,)``; required when
            ``kind='envelope'`` and ``thg`` is true, rejected otherwise
            (:func:`thg_carrier`).
        window : numpy.ndarray or None, optional
            The temporal collar in ``[0, 1]``, shape ``(n_to,)``: the grid's
            ``t_window_fine``, which a propagation applies. It multiplies each product
            in time before the output multipliers, ``G[w (G E)^3]``
            (:func:`dispersive_cubic`); ``None``, the default, is a unit collar.

        Returns
        -------
        numpy.ndarray
            The polarisation in C/m^2, shape ``(n_pol, n_to)``: real on a field grid,
            the envelope of the analytic polarisation on an envelope grid.

        Raises
        ------
        GridError
            If the layout is unknown, the arrays do not match it, the frequencies do
            not match the transform, the carrier is missing, unwanted or the wrong
            length, or the collar's length is not the field's.
        """
        values = np.asarray(field_t)
        needs_carrier = kind == "envelope" and self.thg
        carrier_values = None if carrier is None else np.asarray(carrier)
        _check_evaluate(
            values, kind, carrier_values, needs_carrier=needs_carrier, max_pol=3
        )
        if density < 0.0:
            msg = f"density must be non-negative, got {density!r}"
            raise GridError(msg)
        frequencies = np.asarray(omega, dtype=float)
        expected = fourier.n_omega_for(values.shape[-1], kind)
        if frequencies.shape != (expected,):
            msg = (
                f"omega has shape {frequencies.shape}; a {kind} grid of "
                f"{values.shape[-1]} samples has ({expected},) bins"
            )
            raise GridError(msg)
        collar = None if window is None else np.asarray(window, dtype=float)
        if collar is not None and collar.shape != (values.shape[-1],):
            msg = (
                f"window has shape {collar.shape}; the field has {values.shape[-1]} "
                "samples"
            )
            raise GridError(msg)
        cubic = dispersive_cubic(
            values,
            self.weights(frequencies, kind=kind),
            kind=kind,
            sector=self.sector,
            carrier=carrier_values,
            window=collar,
        )
        out = density * self.coefficient * cubic
        return np.ascontiguousarray(out.real) if kind == "field" else out


def _frozen(values: np.ndarray) -> np.ndarray:
    """Return a C-contiguous read-only copy of an array.

    Parameters
    ----------
    values : numpy.ndarray
        The array.

    Returns
    -------
    numpy.ndarray
        Its values, not writeable.
    """
    frozen = np.array(values, order="C")
    frozen.setflags(write=False)
    return frozen


def _check_plasma(values: np.ndarray, kind: str, dt: float) -> None:
    """Validate the arguments of :meth:`PlasmaResponse.evaluate`.

    The layout check comes first, so that asking for an envelope grid is answered with
    the reason a plasma response refuses it rather than with a complaint about the array
    being real. What follows is the shape and dtype half of :func:`_check_evaluate`,
    reused rather than repeated; its carrier arguments never apply here.

    Parameters
    ----------
    values : numpy.ndarray
        The samples passed as ``field_t``.
    kind : str
        Layout name to check.
    dt : float
        Sample spacing of the fine time grid, s.

    Raises
    ------
    GridError
        If the layout is not ``'field'``, the array shape or dtype does not match it, or
        the time step is not positive and finite.
    """
    if kind != "field":
        msg = (
            f"the plasma response is defined on a field grid only, got {kind!r}; the "
            "envelope form has to be re-derived rather than adapted"
        )
        raise GridError(msg)
    _check_evaluate(values, kind, None, needs_carrier=False)
    if not (math.isfinite(dt) and dt > 0.0):
        msg = f"dt must be positive and finite, got {dt!r}"
        raise GridError(msg)


@dataclass(frozen=True, eq=False)
class PlasmaResponse:
    """The polarisation of the free electrons a strong field tears from a gas.

    Two currents, both per unit density and both integrated along the time axis
    (Geissler et al. 1999). Write ``f(t)`` for the fraction of atoms ionised by time
    ``t`` and ``W`` for the ionisation rate at the instantaneous field:

    - the **free-electron current** ``J_free = (e^2/m_e) int f E dt``, the current of
      electrons already freed and being accelerated by the field. It is what lowers the
      refractive index and blue-shifts the pulse;
    - the **ionisation loss** ``J_loss = I_p W (1-f) E / |E|^2``, added pointwise. Its
      form is chosen so that ``E . J_loss = I_p W (1-f) = I_p df/dt`` exactly: the field
      pays one ionisation potential for every electron it frees.

    The polarisation is ``rho`` times the integral of their sum, so ``evaluate`` does
    three cumulative integrals in all. It is a **memory** response: the value at one
    sample depends on every earlier sample. Under the plain trapezoid it depends on none
    of the later ones; under the band rule (SEP 18) each integral reads ``K`` samples
    ahead, so the polarisation at one sample depends on samples up to ``3K`` later, a
    band-limited approximation to the causal response whose precursor is confined to
    those samples.

    Production forms the held-current source directly in frequency (SEP 27), using
    only the population and free-current integrals. This does not change the
    physical polarisation returned by ``evaluate``.

    ``integration`` selects the three physical-response integrals. ``'band'``
    filters each integrand with symmetric taps designed for the grid, so every
    integral is exact to a certified ``1e-5`` across the active band. The plain
    trapezoid is short by
    ``1 - (theta/2) cot(theta/2)`` per integral at ``theta = omega dt``, 11 % at the top
    of the default active band. ``'trapezoid'`` is that plain rule, bit for bit, and is
    what the frozen witness comparisons use. The taps belong to the grid, not to the
    response: a propagation designs them in translation, and a direct call to
    :meth:`evaluate` or :meth:`ionised_fraction` under the band rule is passed them as
    ``integral`` (:func:`solprop.translate.plasma_integration.design_running_integral`).

    The ionised fraction is ``f = 1 - (1 - f_0) exp(-int W dt)``, with the pre-ionised
    fraction inside the exponential. Adding it outside, ``f_0 + 1 - exp(-int W dt)``,
    would exceed one, make ``1 - f`` negative and break the energy bookkeeping above,
    because ``df/dt`` would then no longer be ``W (1 - f)``.

    ``preionisation`` is a number or a path along ``z``: the fraction ionised before the
    pulse arrives at each position. The trait declares it as a path because that is
    what the translation layer supplies -- a value per plane -- and the reference
    evaluation sees one plane at a time. Where the path varies, the oracle for a plane
    is :func:`dataclasses.replace` with that plane's value, which rebuilds and
    revalidates the response; the profile itself is never resolved here, and
    :meth:`evaluate` and :meth:`ionised_fraction` refuse a response that still carries
    one.

    Attributes
    ----------
    gas : Gas or str or Species
        The species, for its ionisation potential. Beside a rate that carries its own
        energy per event (an :class:`~solprop.nonlinear.optmat_rates.OptmatRate`),
        the species is stored with that energy as its potential
        (:func:`~solprop.nonlinear.ionisation.bind_rate_energy`, SEP 3, D8).
    rate : EvaluableRate
        The ionisation rate in s^-1 as a function of field amplitude in V/m: optmat's
        :class:`solprop.nonlinear.optmat_rates.OptmatRate`, a caller's rate, or the
        :class:`solprop.translate.ionisation.RateTable` a propagation actually uses. It
        must be exactly zero at zero field, which is what makes the loss term finite.
    preionisation : float or ZProfile, optional
        Fraction already ionised before the pulse, in ``[0, 1]``: a number, or a path
        along z in m giving that fraction at each position. A crude model: it says
        nothing about where those electrons came from or what they were doing.
    species : str or None, optional
        Component label, distinct from the chemical species when populations split.
    integration : {'band', 'trapezoid'}, optional
        The running-integral rule of the column (SEP 18): the band rule, the default,
        or the plain cumulative trapezoid.

    Raises
    ------
    MaterialError
        If the gas is unknown or has no ionisation potential, the pre-ionised fraction
        is neither a number nor a path or leaves ``[0, 1]``, the rate is not zero at
        zero field, it carries an energy per event for another species, or
        ``integration`` is not a known rule.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> from solprop.nonlinear.responses import PlasmaResponse
    >>> plasma = PlasmaResponse("Ar", OptmatRate("Ar", 800e-9))
    >>> plasma.traits.domain
    'memory'

    Evaluated directly, the band rule is handed its grid's design:

    >>> from solprop.grid import FieldGrid
    >>> from solprop.translate.plasma_integration import design_running_integral
    >>> grid = FieldGrid.from_limits(1.0, 800e-9, (200e-9, 2e-6), 40e-15)
    >>> t = (np.arange(grid.n_to) - grid.n_to // 2) * grid.dt_fine
    >>> field = 4e10 * np.exp(-((t / 8e-15) ** 2))
    >>> rule = design_running_integral(grid, "band")
    >>> fraction = plasma.ionised_fraction(
    ...     field[None, :], dt=grid.dt_fine, kind="field", integral=rule
    ... )
    >>> f"{fraction[-1]:.4f}"
    '0.3209'
    """

    gas: Gas | str | Species
    rate: EvaluableRate
    _: KW_ONLY
    preionisation: float | ZProfile = 0.0
    species: str | None = None
    integration: IntegrationPolicy = "band"

    def __post_init__(self) -> None:
        """Normalise the species and reject settings that cannot give a polarisation.

        Raises
        ------
        MaterialError
            If the gas is unknown or has no ionisation potential, the pre-ionised
            fraction is neither a number nor a path or leaves ``[0, 1]``, the rate
            is not zero at zero field, it carries an energy per event for another
            species, or the integration rule is unknown.
        """
        # The annotation refuses another rule first wherever the annotations are
        # checked at run time, as in the test suite.
        if self.integration not in {"band", "trapezoid"}:  # pragma: no cover
            msg = f"integration must be 'band' or 'trapezoid', got {self.integration!r}"
            raise MaterialError(msg)
        if not isinstance(self.gas, Species):
            object.__setattr__(self, "gas", as_gas(self.gas))
        # A rate that carries its own energy per event sets the energy the loss pays
        # (SEP 3, D8); any other rate leaves the species as given.
        object.__setattr__(self, "gas", bind_rate_energy(self.gas, self.rate))
        object.__setattr__(
            self, "preionisation", check_preionisation(self.preionisation)
        )
        # Touch the material data now, so that an unknown species fails at construction
        # rather than in the middle of a propagation.
        _ = self.potential
        # The loss term divides by the field, and is finite only because the rate is
        # identically zero where the field is too weak to ionise. A rate that is not
        # zero at zero field would put an infinity into the first sample of every pulse.
        if float(np.asarray(self.rate(0.0))) != 0.0:
            msg = (
                "the rate must be exactly zero at zero field; the ionisation loss term "
                "divides by the field and is finite only because it is"
            )
            raise MaterialError(msg)

    @property
    def potential(self) -> float:
        """Return the ionisation potential without inventing custom species data.

        Returns
        -------
        float
            First ionisation potential in J.

        Raises
        ------
        MaterialError
            If the named custom species has no ionisation data.
        """
        if isinstance(self.gas, Species):
            if self.gas.ionisation is None:
                raise MaterialError(f"{self.gas.name} has no ionisation data")
            return self.gas.ionisation.potential
        return float(ionisation_potential(self.gas))

    @property
    def traits(self) -> ResponseTraits:
        """Return the declared traits of the plasma response.

        Returns
        -------
        ResponseTraits
            With memory, field grid only, vector-isotropic, consuming the density and
            pre-ionisation paths.

        Examples
        --------
        >>> from solprop.nonlinear.optmat_rates import OptmatRate
        >>> from solprop.nonlinear.responses import PlasmaResponse
        >>> PlasmaResponse("Ar", OptmatRate("Ar", 800e-9)).traits.grids
        ('field',)
        """
        return ResponseTraits(
            domain="memory",
            grids=("field",),
            polarisation="vector_isotropic",
            z_profiles=("density", "preionisation")
            if self.species is None
            else (f"density:{self.species}", f"preionisation:{self.species}"),
        )

    def ionised_fraction(
        self,
        field_t: Num[np.ndarray, "n_pol n_to"],
        *,
        dt: float,
        kind: fourier.GridKind,
        integral: RunningIntegral | None = None,
    ) -> Float[np.ndarray, " n_to"]:
        """Return the fraction of atoms ionised, sample by sample.

        Multiply by the density for the electron density, which is what a run reports as
        a statistic. The rate is taken on the **magnitude** of the field, so an
        elliptically polarised pulse ionises according to how hard it pulls, not
        according to one component.

        Parameters
        ----------
        field_t : numpy.ndarray
            The real field on the fine time grid, V/m, shape ``(n_pol, n_to)``. It is
            annotated as any number and checked at run time, as in
            :meth:`KerrResponse.evaluate`, so that the guard is the authority on the
            dtype rather than the test session's type checker.
        dt : float
            Sample spacing of the fine time grid, s, positive.
        kind : {'field'}
            The spectral layout; only a field grid is defined.
        integral : RunningIntegral or None, optional
            The grid's running-integral rule, required under ``integration='band'``.

        Returns
        -------
        numpy.ndarray
            Fraction ionised, in ``[preionisation, 1]``, shape ``(n_to,)``,
            non-decreasing in time (under the band rule, to within the filter's
            undershoot).

        Raises
        ------
        GridError
            If the layout is not a field grid, the array shape or dtype does not match
            it, the time step is not positive, or the rate returns the wrong shape or a
            value that is not finite and non-negative; or if the integration rule is
            missing, is another rule than the response's, or was designed for another
            ``dt``.
        MaterialError
            If the preionisation is a path along z rather than one plane's value.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.optmat_rates import OptmatRate
        >>> from solprop.nonlinear.responses import PlasmaResponse
        >>> plasma = PlasmaResponse(
        ...     "Ar", OptmatRate("Ar", 800e-9), preionisation=0.25,
        ...     integration="trapezoid",
        ... )
        >>> field = np.full((1, 8), 1e6)
        >>> plasma.ionised_fraction(field, dt=1e-16, kind="field").tolist()
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
        """
        values = np.asarray(field_t)
        _check_plasma(values, kind, dt)
        return self._fraction(values, dt, self._rule(integral, dt))[2]

    def evaluate(
        self,
        field_t: Num[np.ndarray, "n_pol n_to"],
        density: float,
        *,
        dt: float,
        kind: fourier.GridKind,
        integral: RunningIntegral | None = None,
    ) -> Float[np.ndarray, "n_pol n_to"]:
        """Return the plasma polarisation on the fine time grid.

        Return the physical, ungated polarisation, including its post-pulse drift.
        This reference integrates the population, free current and polarisation in
        three separate passes. Field-grid propagation instead holds the same total
        current at the temporal boundary: it assembles ``DFT[g J]/(i omega) -
        DFT[g' J]/(i omega)**2`` on active bins (SEP 27). That assembly omits the
        final running integral; it does not change this physical-response oracle.

        Parameters
        ----------
        field_t : numpy.ndarray
            The real field on the fine time grid, V/m, shape ``(n_pol, n_to)``.
        density : float
            Number density of the gas in m^-3, non-negative. The rate and the fraction
            are per atom; this is what turns them into a polarisation.
        dt : float
            Sample spacing of the fine time grid, s, positive.
        kind : {'field'}
            The spectral layout; only a field grid is defined.
        integral : RunningIntegral or None, optional
            The grid's running-integral rule, required under ``integration='band'``.

        Returns
        -------
        numpy.ndarray
            The polarisation in C/m^2, shape ``(n_pol, n_to)``, real.

        Raises
        ------
        GridError
            If the layout is not a field grid, the array shape or dtype does not match
            it, the time step is not positive, the density is negative, or the rate
            returns the wrong shape or a value that is not finite and non-negative; or
            if the integration rule is missing, is another rule than the response's, or
            was designed for another ``dt``.
        MaterialError
            If the preionisation is a path along z rather than one plane's value.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.nonlinear.optmat_rates import OptmatRate
        >>> from solprop.nonlinear.responses import PlasmaResponse
        >>> t = np.linspace(-2e-14, 2e-14, 257)
        >>> dt = float(t[1] - t[0])
        >>> field = 4e10 * np.exp(-((t / 8e-15) ** 2)) * np.cos(2.35e15 * t)
        >>> plasma = PlasmaResponse(
        ...     "Ar", OptmatRate("Ar", 800e-9), integration="trapezoid"
        ... )
        >>> out = plasma.evaluate(field[None, :], 2.5e25, dt=dt, kind="field")
        >>> f"{out[0, -1]:.4e}"
        '8.9313e-07'
        """
        values = np.asarray(field_t)
        _check_plasma(values, kind, dt)
        if density < 0.0:
            msg = f"density must be non-negative, got {density!r}"
            raise GridError(msg)

        rule = self._rule(integral, dt)
        magnitude, rate, fraction = self._fraction(values, dt, rule)
        current = rule.integrate(fraction * CHARGE_MASS_RATIO * values, dt)
        current = current + self._loss(values, magnitude, rate, fraction)
        return density * rule.integrate(current, dt)

    def _rule(self, integral: RunningIntegral | None, dt: float) -> RunningIntegral:
        """Return the running-integral rule a direct evaluation uses.

        The response never designs taps itself and never looks at the field's spectrum:
        the band rule's taps belong to the grid, and a direct call is handed them.

        Parameters
        ----------
        integral : RunningIntegral or None
            The rule passed to the call.
        dt : float
            The call's sample spacing, s.

        Returns
        -------
        RunningIntegral
            `integral`, or the trapezoid when it is omitted under that rule.

        Raises
        ------
        GridError
            If the band rule has no `integral`, `integral` is another rule than
            ``integration``, or it was designed for another ``dt``.
        """
        if integral is None:
            if self.integration == "band":
                msg = (
                    "integration='band' needs the grid's running integral: pass "
                    "integral=design_running_integral(grid, 'band') from "
                    "solprop.translate.plasma_integration, or use "
                    "integration='trapezoid'"
                )
                raise GridError(msg)
            return RunningIntegral.trapezoid()
        if integral.policy != self.integration:
            msg = (
                f"the running integral is the {integral.policy!r} rule but this "
                f"response asks for integration={self.integration!r}"
            )
            raise GridError(msg)
        integral.check(dt)
        return integral

    def _magnitude(
        self, values: Float[np.ndarray, "n_pol n_to"]
    ) -> Float[np.ndarray, " n_to"]:
        """Return the field magnitude the rate is evaluated on.

        Through :func:`numpy.hypot` for two components rather than a sum of squares:
        the tunnelling exponent is tens to hundreds, so it turns a difference of one
        unit in the last place of ``|E|`` into a relative difference of 1e-13 in the
        rate, and the stored reference data are built with ``hypot`` too.

        Parameters
        ----------
        values : numpy.ndarray
            The real field, shape ``(n_pol, n_to)``.

        Returns
        -------
        numpy.ndarray
            ``|E|`` in V/m, shape ``(n_to,)``.
        """
        if values.shape[0] == 1:
            return np.abs(values[0])
        return np.hypot(values[0], values[1])

    def _fraction(
        self,
        values: Float[np.ndarray, "n_pol n_to"],
        dt: float,
        rule: RunningIntegral,
    ) -> tuple[
        Float[np.ndarray, " n_to"],
        Float[np.ndarray, " n_to"],
        Float[np.ndarray, " n_to"],
    ]:
        """Return the field magnitude, the rate and the fraction ionised.

        Parameters
        ----------
        values : numpy.ndarray
            The real field, shape ``(n_pol, n_to)``.
        dt : float
            Sample spacing of the fine time grid, s.
        rule : RunningIntegral
            The running-integral rule of the exposure.

        Returns
        -------
        tuple of numpy.ndarray
            ``|E|`` in V/m, the rate in s^-1 and the fraction ionised, each shape
            ``(n_to,)``.

        Raises
        ------
        GridError
            If the rate returns the wrong shape, or a value that is not finite and
            non-negative.
        MaterialError
            If the preionisation is a path along z rather than one plane's value.
        """
        if isinstance(self.preionisation, ZProfile):
            raise MaterialError(_PREIONISATION_PATH)
        magnitude = self._magnitude(values)
        rate = np.asarray(self.rate(magnitude), dtype=np.float64)
        if rate.shape != magnitude.shape:
            msg = (
                f"the rate returned shape {rate.shape} for {magnitude.shape} field "
                "amplitudes; it must be evaluated sample by sample"
            )
            raise GridError(msg)
        if not np.all(np.isfinite(rate)) or np.any(rate < 0.0):
            msg = "the rate must be finite and non-negative at every sample"
            raise GridError(msg)
        fraction = ionisation_fraction(
            self.rate, magnitude, dt, preionisation=self.preionisation, integral=rule
        )
        return magnitude, rate, fraction

    def _loss(
        self,
        values: Float[np.ndarray, "n_pol n_to"],
        magnitude: Float[np.ndarray, " n_to"],
        rate: Float[np.ndarray, " n_to"],
        fraction: Float[np.ndarray, " n_to"],
    ) -> Float[np.ndarray, "n_pol n_to"]:
        """Return the ionisation-loss current, per unit density.

        ``I_p W (1 - f) E / |E|^2``: the current whose work against the field is exactly
        the energy the electrons cost. For one component it is ``I_p W (1-f)/E``;
        for two it is that magnitude directed along the field, which is what makes the
        scalar and vector forms agree for a linearly polarised pulse.

        Written with ``where`` rather than a branch, so that the divide never happens
        where the field vanishes -- the whole point of the rate being identically zero
        below threshold, and the reason a clipped-but-non-zero table would be unusable.

        Parameters
        ----------
        values : numpy.ndarray
            The real field, shape ``(n_pol, n_to)``.
        magnitude : numpy.ndarray
            ``|E|``, shape ``(n_to,)``.
        rate : numpy.ndarray
            The ionisation rate in s^-1, shape ``(n_to,)``.
        fraction : numpy.ndarray
            The fraction ionised, shape ``(n_to,)``.

        Returns
        -------
        numpy.ndarray
            The loss current per unit density, shape ``(n_pol, n_to)``.
        """
        # The divisor rather than the field itself: |E| squared underflows to zero
        # below about 1e-162 V/m, and this is the condition the division needs.
        squared = magnitude**2
        prefactor = np.zeros_like(magnitude)
        np.divide(
            self.potential * rate * (1.0 - fraction),
            squared,
            out=prefactor,
            where=(rate > 0.0) & (squared > 0.0),
        )
        return prefactor * values


def _sum_of_squares(
    values: Num[np.ndarray, "n_pol n_to"],
) -> Num[np.ndarray, "1 n_to"]:
    """Return the sum of the squares over the polarisation axis, keeping the axis.

    Squares, not squared moduli: this is the combination that produces the third
    harmonic, and for a single component it is simply ``a**2``.

    Parameters
    ----------
    values : numpy.ndarray
        Samples, shape ``(n_pol, n_to)``.

    Returns
    -------
    numpy.ndarray
        ``sum_q values_q**2``, shape ``(1, n_to)``, broadcasting against the input.
    """
    return np.sum(values**2, axis=0, keepdims=True)


def _fundamental_cubic(
    analytic: Complex[np.ndarray, "n_pol n_to"],
) -> Complex[np.ndarray, "n_pol n_to"]:
    """Return the fundamental-band cubic of an isotropic medium, per component.

    ``P_p = (1/2) (sum_q |a_q|^2) a_p + (1/4) (sum_q a_q^2) conj(a_p)``: the content of
    the real cubic that oscillates as ``exp(i omega0 t)``. For one component it
    collapses to ``(3/4)|a|^2 a`` and for two it expands to
    ``(3/4)[(|a_x|^2 + (2/3)|a_y|^2) a_x + (1/3) conj(a_x) a_y^2]``; the algebra is in
    ``docs/guide/media/nonlinear.md``.

    Parameters
    ----------
    analytic : numpy.ndarray
        Analytic signal or envelope, complex, shape ``(n_pol, n_to)``.

    Returns
    -------
    numpy.ndarray
        The cubic, complex, shape ``(n_pol, n_to)``.
    """
    sum_abs = np.sum(np.abs(analytic) ** 2, axis=0, keepdims=True)
    conjugate_term = 0.25 * _sum_of_squares(analytic) * np.conj(analytic)
    return 0.5 * sum_abs * analytic + conjugate_term
