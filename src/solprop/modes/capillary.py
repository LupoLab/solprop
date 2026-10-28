# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Marcatili modes of a gas-filled hollow capillary, with analytic derivatives.

The model is Marcatili and Schmeltzer's (*Bell Syst. Tech. J.* **43**, 1783-1809, 1964)
asymptotic expansion for a hollow dielectric waveguide whose core radius is large
compared with the wavelength. With :math:`k = \omega/c`, :math:`u_{nm}` the Bessel zero
of the mode, :math:`\varepsilon_{\rm co}` the core permittivity and :math:`v_n` the
cladding term,

.. math::
   n_{\rm eff} = \sqrt{\varepsilon_{\rm co}
   - \left(\frac{u_{nm}}{ka}\right)^2\left(1 - \frac{i v_n}{ka}\right)^2}

is the ``"full"`` model, and expanding it to leading order in :math:`1/(ka)` gives the
``"reduced"`` one,

.. math::
   n_{\rm eff} = 1 + \frac{\varepsilon_{\rm co} - 1}{2}
   - \frac{c^2u_{nm}^2}{2\omega^2a^2}
   + i\,\frac{c^3u_{nm}^2v_n}{a^3\omega^3}.

A positive imaginary part is loss (``docs/dev/conventions.md`` section 5), so the
reduced model's loss term enters with a **plus** sign. Getting that backwards turns
loss into gain silently, so there is one arithmetic kernel here and a test that asserts
loss is loss.

Three properties of this implementation matter.

**Analytic frequency derivatives.** :meth:`MarcatiliMode.dneff_domega` differentiates
both models in closed form, chaining through the analytic derivative of the gas
susceptibility and of the cladding index. Recomputing ``beta_1`` every stage with a
seven-point finite-difference stencil instead would cost about seven full index
evaluations carrying 1e-10 of noise, which floors the cumulative operator at 1e-8 (see
``docs/dev/decisions.md``).

**No round trip through unity.** :meth:`MarcatiliMode.neff_minus_one` forms
``n_eff - 1`` directly -- ``x/(sqrt(1 + x) + 1)`` for the full model, term by term for
the reduced one -- rather than building ``1 + chi/2`` and subtracting one, which loses
about eight digits at the 1e-4 susceptibility of a gas fill.

**A factorisation the translation layer can use.** ``z`` enters only through the core
radius ``a(z)`` and the number density ``rho(z)``; everything frequency-dependent is
:class:`MarcatiliTerms`, sampled once. :meth:`MarcatiliMode.sampled_terms` and
:meth:`MarcatiliMode.neff_from_terms` expose that split, so the Chebyshev builder can
evaluate the operator at many ``z`` for two scalars each.

The manual page is ``docs/guide/waveguides/capillary-model.md``.
"""

import logging
import math
import operator
import warnings
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import KW_ONLY, dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar, Literal, cast

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact
from scipy.integrate import quad
from scipy.linalg import sqrtm
from scipy.special import jn_zeros, jv

from solprop._display import Displayed, engineering, line
from solprop.constants import EPS0, C
from solprop.exceptions import GrazingIncidenceWarning, ModeError, TranslationError
from solprop.modes.fill import (
    EVANESCENT_INDEX_FLOOR,
    GasFillMixin,
    LossSpec,
    _sqrt1p_minus_one,
)
from solprop.modes.protocol import (
    ChannelLabel,
    CrossSection,
    Mode,
    PolarisationComponent,
)
from solprop.physdata.gases import Gas
from solprop.physdata.glasses import GlassIndex, fused_silica
from solprop.physdata.mixtures import Mixture
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.profiles import (
    Constant,
    ZProfile,
    _temperature_text,
    as_profile,
    profile_identity,
)

__all__ = [
    "EVANESCENT_INDEX_FLOOR",
    "CoherentWall",
    "LossSpec",
    "MarcatiliMode",
    "MarcatiliModel",
    "MarcatiliTerms",
    "ModeKind",
    "VectorWallOperator",
    "WallFamily",
    "WallImpedance",
    "WallLimits",
    "WallOperator",
    "bessel_zero",
    "taper_connection",
    "taper_coupling_block",
    "vector_wall_operator",
    "wall_connection",
    "wall_operator",
    "wall_square_root_metric",
    "worst_wall_limits",
]

logger = logging.getLogger(__name__)

ModeKind = Literal["HE", "TE", "TM"]
"""Which family a capillary mode belongs to: hybrid, transverse electric or magnetic."""

MarcatiliModel = Literal["full", "reduced"]
"""Whether to use the square-root expression or its leading-order expansion."""

WallImpedance = Literal["mean", "te_tm"]
"""Whether a coherent wall condenses one mean impedance or separate TE and TM ones."""


def bessel_zero(order: int, index: int) -> float:
    """Return the ``index``-th positive zero of the Bessel function of the first kind.

    Parameters
    ----------
    order : int
        Order of the Bessel function. Negative orders are folded onto positive ones,
        since ``J_-p = (-1)^p J_p`` for integer ``p`` and the zeros coincide.
    index : int
        Which zero, counting from 1.

    Returns
    -------
    float
        The zero ``j_{order, index}``, dimensionless.

    Raises
    ------
    ModeError
        If `index` is not a positive integer.

    Examples
    --------
    >>> from solprop.modes.capillary import bessel_zero
    >>> f"{bessel_zero(0, 1):.10f}"
    '2.4048255577'
    >>> bessel_zero(-2, 1) == bessel_zero(2, 1)
    True
    """
    if index < 1:
        raise ModeError(f"the radial index must be a positive integer, got {index}")
    return float(jn_zeros(abs(order), index)[-1])


@dataclass(frozen=True, eq=False)
class MarcatiliTerms:
    """Everything in a Marcatili effective index that depends only on frequency.

    ``z`` enters a capillary mode through exactly two scalars, the core radius and the
    number density, so the frequency-dependent pieces can be sampled once and reused at
    every position (``docs/dev/architecture.md`` section 6.5).
    Building this is the expensive part: it calls the gas model and the cladding model.

    The frequency is typed as inexact rather than real on purpose. The effective index
    is a holomorphic function of ``omega`` -- powers, a square root and a rational
    cladding term -- so sampling these terms at a *complex* frequency lets the analytic
    derivative be checked against Cauchy's integral formula, which is free of the
    subtractive cancellation that limits any finite difference. Nothing in the public
    API does that; :meth:`MarcatiliMode.sampled_terms` requires a real, positive
    frequency.

    Attributes
    ----------
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``, > 0 in every use except
        the contour check described above.
    u_nm : float
        The Bessel zero of the mode, dimensionless.
    core_susceptibility : numpy.ndarray
        Per-particle susceptibility ``gamma(omega)`` in m^3, so that
        ``chi = gamma * rho``. Mixtures carry shape ``(n_species, n_omega)``;
        the existing single-gas spelling retains its frequency vector. Values are
        real for transparent built-in gases and complex for absorbing custom species.
    dcore_susceptibility_domega : numpy.ndarray
        Its analytic frequency derivative in m^3 s/rad.
    cladding_term : numpy.ndarray
        The complex ``v_n(omega)`` of the mode's family.
    dcladding_term_domega : numpy.ndarray
        Its analytic frequency derivative in s/rad.
    """

    omega: Inexact[np.ndarray, " n_omega"]
    u_nm: float
    core_susceptibility: Inexact[np.ndarray, "*species n_terms"]
    dcore_susceptibility_domega: Inexact[np.ndarray, "*species n_terms"]
    cladding_term: Complex[np.ndarray, " n_clad"]
    dcladding_term_domega: Complex[np.ndarray, " n_clad"]

    def __post_init__(self) -> None:
        """Check that every sampled array has one entry per frequency.

        Raises
        ------
        ModeError
            If the arrays disagree in shape.
        """
        shapes = {
            self.omega.shape,
            self.core_susceptibility.shape[-1:],
            self.dcore_susceptibility_domega.shape[-1:],
            self.cladding_term.shape,
            self.dcladding_term_domega.shape,
        }
        if (
            len(shapes) != 1
            or self.core_susceptibility.ndim not in (1, 2)
            or self.core_susceptibility.shape != self.dcore_susceptibility_domega.shape
        ):
            raise ModeError(f"sampled terms disagree in shape: {sorted(shapes)}")


def _cladding_term_and_derivative(
    permittivity: Complex[np.ndarray, " n_omega"],
    dpermittivity: Complex[np.ndarray, " n_omega"],
    kind: ModeKind,
) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
    """Return the Marcatili cladding term and its frequency derivative.

    With ``e`` the cladding permittivity the three families give
    ``v_HE = (e + 1) / (2 sqrt(e - 1))``, ``v_TE = 1 / sqrt(e - 1)`` and
    ``v_TM = e / sqrt(e - 1)``. The root is taken in the complex plane, so a cladding
    with ``Re e < 1`` -- a metal, or silica in the extreme ultraviolet -- is handled
    rather than producing a NaN.

    Parameters
    ----------
    permittivity : numpy.ndarray
        Complex relative permittivity of the cladding, shape ``(n_omega,)``.
    dpermittivity : numpy.ndarray
        Its analytic frequency derivative in s/rad, same shape.
    kind : str
        ``"HE"``, ``"TE"`` or ``"TM"``.

    Returns
    -------
    tuple of numpy.ndarray
        ``v_n(omega)`` and ``dv_n/domega``.
    """
    offset = np.asarray(permittivity - 1.0, dtype=complex)
    root = np.sqrt(offset)
    if kind == "HE":
        term = (permittivity + 1.0) / (2.0 * root)
        dterm_dpermittivity = (permittivity - 3.0) / (4.0 * offset * root)
    elif kind == "TE":
        term = 1.0 / root
        dterm_dpermittivity = -0.5 / (offset * root)
    else:
        term = permittivity / root
        dterm_dpermittivity = (0.5 * permittivity - 1.0) / (offset * root)
    return np.asarray(term), np.asarray(dterm_dpermittivity * dpermittivity)


@dataclass(frozen=True, eq=False)
class MarcatiliMode(GasFillMixin):
    """A Marcatili mode of a gas-filled hollow capillary.

    Departs from the ``omega >= 0`` of :class:`~solprop.modes.protocol.Mode` in one
    place: a hollow capillary has no mode at zero frequency -- the waveguide term
    ``(u c / (omega a))^2`` diverges -- so ``omega`` must be strictly positive here and
    a zero raises. The linear operator evaluates modes on the active band, which
    excludes the DC bin, so nothing in the package asks.

    Both ``radius`` and ``pressure`` accept a number or a
    :class:`~solprop.profiles.ZProfile`, which is what makes a taper and a pressure
    gradient ordinary cases rather than special ones.

    Attributes
    ----------
    radius : float or ZProfile
        Core radius in m, > 0. A number becomes a :class:`~solprop.profiles.Constant`.
    gas : Gas or str or None
        The fill species, or ``None`` for an evacuated capillary.
    pressure : float or ZProfile
        Absolute pressure in **bar**, >= 0 (``docs/dev/conventions.md`` section 1).
    kind : str
        ``"HE"`` (hybrid), ``"TE"`` or ``"TM"``. The transverse families require
        ``n == 0``; hybrids require ``n != 0``.
    n : int
        Azimuthal index. May be negative for a hybrid mode.
    m : int
        Radial index, >= 1: which zero of the Bessel function.
    phi : float
        Rotation of the mode about the axis in rad, positive from ``x`` towards ``y``.
        It turns the whole transverse pattern: the Cartesian components of a hybrid
        mode are ``(cos((n - 1) theta - n phi), -sin((n - 1) theta - n phi))`` times
        ``J_{n-1}(u r / a)``. For ``HE_1m`` the pattern is uniform, so ``phi`` *is*
        the polarisation angle measured from the ``x`` axis: ``phi = 0`` is
        x-polarised and ``phi = pi / 2`` is y-polarised. For ``n >= 2`` an oblique
        ``phi`` is a rotated pattern and not a polarisation angle -- the field points
        in different directions at different ``theta`` -- and the degenerate partner
        of ``phi = 0`` is the quarter turn ``phi = pi / (2 n)``, which
        :attr:`orthogonal_polarisation` returns.
    model : str
        ``"full"`` for the square root, ``"reduced"`` for its leading-order expansion.
    loss : bool or float or callable
        ``True`` for the model's own loss, ``False`` for none, a float to scale it, or
        a callable ``alpha(omega)`` giving an **absolute** power loss coefficient in
        1/m that replaces it. The float and the callable therefore mean different
        things, deliberately: the float is a fitting knob on the model, the callable is
        measured fibre loss. The model's own loss is never negative: where the square
        root lands on the branch that reads as gain -- a cladding index below unity,
        outside Marcatili's regime -- it is clamped at zero and the mode logs the fact
        once. A callable is the caller's and is not clamped. A callable that
        carries a ``derivative`` attribute gets an analytic ``d(Im n_eff)/d(omega)``;
        otherwise that one term is differenced, and the constructor says so.
    cladding : GlassIndex or complex or None
        The cladding index. ``None`` uses
        :func:`~solprop.physdata.glasses.fused_silica`; a plain complex number is a
        constant index, which is what a comparison against a published or symbolic
        result needs.
    temperature : float, ZProfile or None
        Temperature of the fill in K, > 0, or a path of it along z in m, which the
        density follows (:class:`~solprop.profiles.DensityPath`); a constant path is
        stored as its number. ``None`` (the default) is 293.15 K for a gas and the
        mixture's own beside `mixture`, which carries its temperature: a different one
        given beside it is refused by name. Resolved at construction, so the field
        is never ``None`` afterwards.
    susceptibility : GasSusceptibility or None
        The gas susceptibility per unit number density. ``None`` takes the built-in
        gas's, optmat's, which is what a run normally wants. Passing one is
        the gas-side counterpart of a constant or tabulated `cladding`: it replaces
        the refractive data while leaving the equation of state alone, so `gas` is
        still what sets the density along z. A comparison against a published result
        or another code whose gas model differs can give the mode the other model's
        susceptibility and isolate what it is actually comparing.
    mixture : Mixture or None
        Labelled populations with their own density paths. Supplies temperature and
        optical data; do not also pass gas, pressure or susceptibility.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode
    >>> mode = MarcatiliMode(125e-6, "He", 1.0, model="reduced")
    >>> omega = np.array([2.356e15])
    >>> f"{complex(mode.neff(omega)[0]):.9f}"
    '1.000028936+0.000000009j'
    >>> f"{float(mode.aeff()):.4e}"
    '2.3393e-08'
    """

    radius: float | ZProfile
    gas: Gas | str | None = None
    pressure: float | ZProfile = 0.0
    _: KW_ONLY
    kind: ModeKind = "HE"
    n: int = 1
    m: int = 1
    phi: float = 0.0
    model: MarcatiliModel = "full"
    loss: LossSpec = True
    cladding: GlassIndex | complex | None = None
    temperature: float | ZProfile | None = None
    susceptibility: GasSusceptibility | None = None
    mixture: Mixture | None = None
    u_nm: float = field(init=False, repr=False)
    core_radius: ZProfile = field(init=False, repr=False)
    density: ZProfile = field(init=False, repr=False)
    cladding_index: GlassIndex | complex = field(init=False, repr=False)
    _aeff_factor: float = field(init=False, repr=False)
    _gain_warned: bool = field(init=False, default=False, repr=False)
    # The mode a frozen copy was taken from, when the copy's frequency-only terms are
    # the origin's (:meth:`at_parameters`); shared by :meth:`sampled_terms` in a scope.
    _terms_origin: ClassVar["MarcatiliMode | None"] = None  # noqa: UP037 - the hook resolves class names eagerly

    def _check_arguments(self) -> None:
        """Check the constructor arguments against each other.

        Raises
        ------
        ModeError
            If the family and the azimuthal index disagree, or the model, the
            temperature or the loss specification is unusable.
        """
        if not math.isfinite(self.phi):
            raise ModeError(f"phi must be finite, got {self.phi!r}")
        if self.kind not in ("HE", "TE", "TM"):
            raise ModeError(f"kind must be 'HE', 'TE' or 'TM', got {self.kind!r}")
        if self.model not in ("full", "reduced"):
            raise ModeError(f"model must be 'full' or 'reduced', got {self.model!r}")
        if self.kind == "HE" and self.n == 0:
            raise ModeError("a hybrid HE mode needs a non-zero azimuthal index n")
        if self.kind in ("TE", "TM") and self.n != 0:
            raise ModeError(f"a {self.kind} mode needs n = 0, got n = {self.n}")
        self._check_fill_arguments()

    def __post_init__(self) -> None:
        """Validate the mode and build the derived quantities.

        Raises
        ------
        ModeError
            If the family and the azimuthal index disagree, the radial index is not
            positive, the model or the loss specification is unknown, the radius or
            or temperature is not positive.
        """
        self._check_arguments()
        # The transverse families sit on a zero of J_1; the hybrids on one of J_(n-1).
        order = 1 if self.kind in ("TE", "TM") else self.n - 1
        object.__setattr__(self, "u_nm", bessel_zero(order, self.m))
        core_radius = as_profile(self.radius)
        object.__setattr__(self, "core_radius", core_radius)
        self._resolve_fill()
        object.__setattr__(
            self,
            "cladding_index",
            fused_silica() if self.cladding is None else self.cladding,
        )
        object.__setattr__(self, "_aeff_factor", self._effective_area_factor())
        if not float(core_radius(np.array([0.0]))[0]) > 0.0:
            raise ModeError("the core radius must be positive")
        # Warned last, because the message renders the mode and so needs the derived
        # fields above.
        self._warn_loss_derivative()

    def __repr__(self) -> str:
        """Return a short description naming the mode, its fill and its model.

        Returns
        -------
        str
            For example ``MarcatiliMode(HE11, a=125.0 um, He, model=full, loss=True)``.
        """
        if self.mixture is not None:
            # A mode the run interfaces build carries its fill as a mixture and no
            # `gas`; before this branch it printed as evacuated.
            fill = "+".join(self.mixture.labels)
        else:
            fill = "evacuated" if self.gas is None else str(self.gas)
        radius = float(self.core_radius(np.array([0.0]))[0])
        return (
            f"MarcatiliMode({self.kind}{self.n}{self.m}, a={radius * 1e6:g} um, "
            f"{fill}, model={self.model}, loss={self.loss})"
        )

    def _area_rows(self) -> list[tuple[str, object]]:
        """Return the effective-area row of the printed block.

        Marcatili's area is a constant times the radius squared, so it costs nothing;
        a subclass whose area is frequency dependent or tabulated returns no row.

        Returns
        -------
        list of tuple
            ``(key, value)`` rows.
        """
        return [("effective area", f"{self.aeff() * 1e12:.4g} um^2")]

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        radius = float(self.core_radius(np.array([0.0]))[0])
        cladding = self.cladding_index
        glass = getattr(cladding, "name", None) or f"constant index {cladding}"
        return f"{type(self).__name__} {self.kind}{self.n}{self.m}", [
            ("core radius", engineering(radius, "m")),
            ("fill", self._fill_text()),
            (
                "temperature",
                _temperature_text(cast("float | ZProfile", self.temperature)),
            ),
            ("model", self.model),
            (
                "loss",
                self._loss_text(),
            ),
            ("cladding", glass),
            *self._area_rows(),
        ]

    # ------------------------------------------------------------------ traits

    @property
    def z_constant(self) -> bool:
        """Return whether nothing about the mode varies along z.

        Returns
        -------
        bool
            ``True`` when neither the core radius nor the density varies. The
            multimode transform cares only about the radius, which
            :attr:`radius_is_constant` reports separately.
        """
        return all(path.is_constant for path in self.parameter_paths.values())

    @property
    def radius_is_constant(self) -> bool:
        """Return whether the transverse profile is the same at every z.

        Returns
        -------
        bool
            ``True`` when the core radius does not vary, so the synthesis and
            projection matrices of the multimode transform may be built once. A
            pressure gradient changes the effective index but not the field profile.
        """
        return self.core_radius.is_constant

    @property
    def scale_invariant(self) -> bool:
        """Return whether the profile scales exactly with the core radius.

        Returns
        -------
        bool
            Always ``True``: the field depends on ``r`` only through ``r / a`` and the
            normalisation goes as ``a^2``, so a taper rescales the transform matrices
            by two scalars.
        """
        return True

    @property
    def azimuthal_order(self) -> int:
        """Return the highest azimuthal harmonic in the Cartesian field components.

        Returns
        -------
        int
            ``|n - 1|`` for a hybrid mode and 1 for a transverse one. The signed
            ``n - 1`` would be negative for ``HE_{-1,m}`` and would then understate
            the theta rule a cubic product needs.
        """
        return 1 if self.kind in ("TE", "TM") else abs(self.n - 1)

    @property
    def channel_label(self) -> ChannelLabel:
        """Return the family, the two indices and which polarisation partner this is.

        Returns
        -------
        ChannelLabel
            ``parity`` is ``None`` for a transverse family, which has no degenerate
            polarisation partner. For a hybrid mode it follows the one definition in
            :class:`~solprop.modes.protocol.ChannelLabel`: the ``even`` member is the
            ``cos((n - 1) theta)`` pattern about the x axis, which here is the
            **unrotated** mode ``phi = 0`` -- at that angle the transverse components
            are ``(cos((n - 1) theta), -sin((n - 1) theta))``, the same pattern the
            exact step-index ``even`` member carries, and for ``HE_1m`` it is the
            x-polarised state. The quarter turn ``phi = pi / (2 n)`` is therefore the
            ``odd`` member, y-polarised for ``HE_1m``. Any other rotation is a gauge
            rather than a parity: its parity is ``None`` and its display label carries
            the angle, so that two rotated copies of one mode are still told apart.

            A **negative** azimuthal index is the ``EH`` family, and the label says
            so: the label's own ``azimuthal_order`` is the non-negative total
            azimuthal order ``|n|`` and the family is ``"EH"``, which is how an exact
            :class:`~solprop.modes.stepindex.StepIndexMode` of the same physical
            channel labels itself, so an ``EH`` capillary mode can be put in a
            :class:`~solprop.modes.collection.ModeCollection`.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, "He", 1.0)
        >>> mode.channel_label.display_label, mode.principal_component()
        ('HE1,1:even', 'x')
        >>> partner = mode.orthogonal_polarisation
        >>> partner.channel_label.display_label, partner.principal_component()
        ('HE1,1:odd', 'y')
        >>> MarcatiliMode(125e-6, "He", 1.0, n=-1).channel_label.display_label
        'EH1,1:even'
        """
        parity: Literal["even", "odd"] | None = None
        if self.kind == "HE":
            # phi = 0 is the "even" pattern every family shares: the Cartesian
            # components are cos((n-1)theta) and -sin((n-1)theta), x-polarised for
            # HE_1m (ADR-031, docs/dev/decisions.md). Verified against the exact
            # step-index pair on axis.
            if self.phi == 0.0:
                parity = "even"
            elif self.phi == math.pi / (2 * self.n):
                parity = "odd"
        suffix = (
            f":{parity}"
            if parity is not None
            else ("" if self.phi == 0.0 else f":phi={self.phi:.6g}")
        )
        family = "EH" if self.n < 0 else self.kind
        return ChannelLabel(
            family,
            abs(self.n),
            self.m,
            parity,
            None,
            f"{family}{abs(self.n)},{self.m}{suffix}",
        )

    def principal_component(self, z: float = 0.0) -> PolarisationComponent | None:
        """Return the sole nonzero Cartesian component, or None for a vector profile.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m; the polarisation is independent of it.

        Returns
        -------
        {'x', 'y'} or None
            An axis-aligned HE_1m component, or ``None`` when both components are
            nonzero. ``phi = 0`` is x (ADR-031) and the quarter turn is y. Angles are
            classified exactly modulo pi, without a tolerance that would erase a small
            physical component.
        """
        del z
        if self.kind != "HE" or self.n != 1:
            return None
        angle = math.remainder(self.phi, math.pi)
        if angle == 0.0:
            return "x"
        if abs(angle) == math.pi / 2:
            return "y"
        return None

    def rotated(self, phi: float) -> "MarcatiliMode":  # noqa: UP037 - bound after the class body
        """Return the same mode turned about the axis.

        Parameters
        ----------
        phi : float
            The new rotation angle in rad, in the sense of :attr:`phi`: positive from
            ``x`` towards ``y``, so for ``HE_1m`` it is the polarisation angle
            measured from the ``x`` axis.

        Returns
        -------
        MarcatiliMode
            A copy with the new angle; everything else is unchanged.

        Examples
        --------
        >>> import math
        >>> from solprop.modes import MarcatiliMode
        >>> MarcatiliMode(125e-6, "He", 1.0).rotated(math.pi / 2).phi
        1.5707963267948966
        """
        return replace(self, phi=phi)

    @property
    def orthogonal_polarisation(self) -> "MarcatiliMode | None":  # noqa: UP037 - as above
        """Return the degenerate partner of a hybrid mode, or ``None``.

        Returns
        -------
        MarcatiliMode or None
            The same mode turned by ``pi / (2 n)``, which turns the pattern a further
            quarter turn from ``x`` towards ``y`` and so exchanges the two Cartesian
            components: for ``HE_11`` that is the y-polarised partner of the
            x-polarised default, and so the ``odd`` member of the pair
            (:attr:`channel_label`). ``None`` for a transverse family, whose
            members are not degenerate in polarisation -- ``TE_0m`` and ``TM_0m``
            are different modes, not two states of one.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> partner = MarcatiliMode(125e-6, "He", 1.0).orthogonal_polarisation
        >>> f"{partner.phi:.6f}"
        '1.570796'
        >>> MarcatiliMode(125e-6, kind="TE", n=0).orthogonal_polarisation is None
        True
        """
        if self.kind != "HE":
            return None
        return self.rotated(self.phi + math.pi / (2 * self.n))

    # ------------------------------------------------ the parameterised protocol

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the scalar paths that carry the whole z dependence.

        Returns
        -------
        collections.abc.Mapping
            Radius and the material paths, including constant ones. One component
            retains ``density``; a shared fill uses total ``pressure`` in bar;
            independent populations use ``density:<label>`` in m^-3. See
            :class:`solprop.modes.protocol.ParameterisedMode`.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> sorted(MarcatiliMode(125e-6, "He", 1.0).parameter_paths)
        ['core_radius', 'density']
        """
        paths: dict[str, ZProfile] = {"core_radius": self.core_radius}
        paths.update(self._fill_paths())
        return MappingProxyType(paths)

    def at_parameters(
        self,
        values: Mapping[str, float],
        # Quoted because beartype resolves the annotation while the class body is
        # still executing, when the bare class name is not yet bound.
    ) -> "MarcatiliMode":  # noqa: UP037
        """Return the same mode with the named paths frozen at the given values.

        The effective index reaches ``z`` only through the two paths, so replacing
        the two paths by constants gives a mode that is exactly this one at every
        position where the paths take those values. That is what lets
        :func:`solprop.linear.dispersion_operator` be evaluated at parameter values,
        and so what lets the cumulative operator be expanded in the parameters rather
        than tabulated along z (``docs/guide/numerics/linear-operator.md``).

        The copy shares the parent's derived frequency-only quantities -- the Bessel
        zero, the cladding index, the effective-area factor -- so freezing costs
        nothing per evaluation. Its `radius` field follows `core_radius`; its
        `pressure` field still records the parent's declaration, because the density is
        given directly here and the equation of state is not inverted. Nothing reads
        `pressure` after construction; `density` is the path that is evaluated.

        Parameters
        ----------
        values : collections.abc.Mapping
            Any subset of ``{"core_radius": a in m, "density": rho in m^-3}``. A name
            left out keeps its path.

        Returns
        -------
        MarcatiliMode
            A copy reporting ``z_constant``.

        Raises
        ------
        ModeError
            If a name is not a declared path, the core radius is not positive, or the
            density is negative.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.profiles import PressureGradient
        >>> fill = PressureGradient.two_point(1.0, 5.0, 1.0)
        >>> mode = MarcatiliMode(125e-6, "He", fill)
        >>> frozen = mode.at_parameters(
        ...     {
        ...         name: float(path(np.array([0.3]))[0])
        ...         for name, path in mode.parameter_paths.items()
        ...     }
        ... )
        >>> frozen.z_constant
        True
        >>> omega = np.array([2.356e15])
        >>> bool(np.array_equal(frozen.neff(omega), mode.neff(omega, 0.3)))
        True
        """
        unknown = sorted(set(values) - set(self.parameter_paths))
        if unknown:
            msg = (
                f"{unknown} is not a declared path of {self!r}; this mode declares "
                f"{sorted(self.parameter_paths)}"
            )
            raise ModeError(msg)
        frozen = cast(
            MarcatiliMode,
            self._frozen_fill(
                {name: value for name, value in values.items() if name != "core_radius"}
            ),
        )
        if "core_radius" in values:
            radius = float(values["core_radius"])
            if not radius > 0.0:
                msg = f"the core radius must be positive, got {radius} m"
                raise ModeError(msg)
            object.__setattr__(frozen, "radius", radius)
            object.__setattr__(frozen, "core_radius", Constant(radius))
        if _same_rows(frozen._fill_rows(), self._fill_rows()):
            # The freeze changed densities and the core radius, which do not enter the
            # frequency-only terms, and left every susceptibility row as it was: the
            # copy's terms are its origin's, the same function of the same models and
            # frequencies.
            object.__setattr__(frozen, "_terms_origin", self._terms_origin or self)
        return frozen

    # ------------------------------------------------- the frequency-only part

    def sampled_terms(self, omega: Float[np.ndarray, " n_omega"]) -> MarcatiliTerms:
        """Sample everything in the effective index that does not depend on z.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, shape ``(n_omega,)``, > 0.

        Returns
        -------
        MarcatiliTerms
            The gas susceptibility, the cladding term and their derivatives, ready to
            be combined with a radius and a density at any position.

        Raises
        ------
        ModeError
            If any frequency is not strictly positive or not finite.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> terms = MarcatiliMode(125e-6, "He", 1.0).sampled_terms(np.array([2.4e15]))
        >>> terms.cladding_term.shape
        (1,)
        """
        values = _checked_omega(omega)
        shared = _SHARED_TERMS.get()
        if shared is None:
            return self._sample_terms(values)
        # A frozen single-gas copy shares its origin's terms, sampled by the origin, so
        # the dilute-scaling check runs once, at the largest density along the fill.
        origin = self._terms_origin or self
        held = shared.setdefault(id(origin), [])
        for mode, terms in held:
            if mode is origin and np.array_equal(terms.omega, values):
                return terms
        terms = origin._sample_terms(values.copy())
        for array in (
            terms.omega,
            terms.core_susceptibility,
            terms.dcore_susceptibility_domega,
            terms.cladding_term,
            terms.dcladding_term_domega,
        ):
            array.flags.writeable = False
        held.append((origin, terms))
        return terms

    def _sample_terms(self, values: Float[np.ndarray, " n_omega"]) -> MarcatiliTerms:
        """Sample the gas and cladding models at validated frequencies.

        Parameters
        ----------
        values : numpy.ndarray
            Finite, strictly positive angular frequencies in rad/s.

        Returns
        -------
        MarcatiliTerms
            The frequency-only terms, as :meth:`sampled_terms` documents.
        """
        susceptibility, dsusceptibility = self._sampled_susceptibility(values)
        if isinstance(self.cladding_index, GlassIndex):
            permittivity = np.asarray(self.cladding_index.permittivity(values))
            dpermittivity = np.asarray(self.cladding_index.dpermittivity_domega(values))
        else:
            permittivity = np.full(values.shape, complex(self.cladding_index) ** 2)
            dpermittivity = np.zeros_like(permittivity)
        term, dterm = _cladding_term_and_derivative(
            permittivity, dpermittivity, self.kind
        )
        return MarcatiliTerms(
            omega=values,
            u_nm=self.u_nm,
            core_susceptibility=susceptibility,
            dcore_susceptibility_domega=dsusceptibility,
            cladding_term=term,
            dcladding_term_domega=dterm,
        )

    def _waveguide(
        self, terms: MarcatiliTerms, radius: float
    ) -> tuple[Inexact[np.ndarray, " n_omega"], Inexact[np.ndarray, " n_omega"]]:
        """Return the waveguide term of the full model and its frequency derivative.

        Parameters
        ----------
        terms : MarcatiliTerms
            The sampled frequency-dependent quantities.
        radius : float
            Core radius in m at the position of interest.

        Returns
        -------
        tuple of numpy.ndarray
            ``w = (u/(ka))^2 (1 - i v/(ka))^2`` and ``dw/domega``.
        """
        omega = terms.omega
        scale = C / (omega * radius)
        bracket = 1.0 - 1j * terms.cladding_term * scale
        squared = (terms.u_nm * scale) ** 2
        # d(scale)/d(omega) = -scale/omega, so the two pieces below are the product
        # rule on w = u^2 scale^2 bracket^2.
        dbracket = -1j * (
            terms.dcladding_term_domega * scale - terms.cladding_term * scale / omega
        )
        derivative = squared * (-2.0 * bracket**2 / omega + 2.0 * bracket * dbracket)
        return np.asarray(squared * bracket**2), np.asarray(derivative)

    def _model_values(
        self,
        terms: MarcatiliTerms,
        radius: float,
        density: float | Float[np.ndarray, " n_species"],
    ) -> tuple[Inexact[np.ndarray, " n_omega"], Inexact[np.ndarray, " n_omega"]]:
        """Return ``n_eff - 1`` and ``dn_eff/domega`` before the loss specification.

        Parameters
        ----------
        terms : MarcatiliTerms
            The sampled frequency-dependent quantities.
        radius : float
            Core radius in m.
        density : float or numpy.ndarray
            Number density in m^-3, scalar for the existing gas spelling or one
            entry per sampled mixture component.

        Returns
        -------
        tuple of numpy.ndarray
            The refractivity and its derivative, with the model's own loss included.
        """
        omega = terms.omega
        if terms.core_susceptibility.ndim == 1:
            chi = terms.core_susceptibility * density
            dchi = terms.dcore_susceptibility_domega * density
        else:
            weights = np.atleast_1d(density)
            if weights.size != terms.core_susceptibility.shape[0]:
                raise ModeError("one density is required for every sampled species")
            chi = terms.core_susceptibility[0] * weights[0]
            dchi = terms.dcore_susceptibility_domega[0] * weights[0]
            for row, weight in enumerate(weights[1:], start=1):
                chi = chi + terms.core_susceptibility[row] * weight
                dchi = dchi + terms.dcore_susceptibility_domega[row] * weight
        if self.model == "full":
            waveguide, dwaveguide = self._waveguide(terms, radius)
            # sqrt(1 + x) - 1 without the round trip through unity that costs
            # eight digits at a gas fill's susceptibility.
            refractivity = _sqrt1p_minus_one(chi - waveguide)
            derivative = (dchi - dwaveguide) / (2.0 * (1.0 + refractivity))
            return np.asarray(refractivity), np.asarray(derivative)
        scale = C / (omega * radius)
        squared = (terms.u_nm * scale) ** 2
        refractivity = 0.5 * chi - 0.5 * squared
        derivative = 0.5 * dchi + squared / omega
        if self.loss is not False:
            # Dropped rather than real-parted when the mode is declared lossless. It
            # matters because v_n is complex for an absorbing cladding, so `i v_n`
            # carries a *real* shift as well as the loss: taking the real
            # part of the lossy expression would keep that shift, and the lossless
            # reduced model would then differ from the polynomial it is meant to be.
            # The full model behaves the other way round -- there the lossless mode
            # takes the real part of the square root, complex v_n and all.
            refractivity = refractivity + 1j * squared * terms.cladding_term * scale
            derivative = derivative + 1j * squared * scale * (
                terms.dcladding_term_domega - 3.0 * terms.cladding_term / omega
            )
        return np.asarray(refractivity, dtype=complex), np.asarray(
            derivative, dtype=complex
        )

    def neff_from_terms(
        self,
        terms: MarcatiliTerms,
        radius: float,
        density: float | Float[np.ndarray, " n_species"],
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the effective index from pre-sampled terms and two scalars.

        Parameters
        ----------
        terms : MarcatiliTerms
            The sampled frequency-dependent quantities, from :meth:`sampled_terms`.
        radius : float
            Core radius in m at the position of interest, > 0.
        density : float
            Number density of the fill in m^-3, >= 0.

        Returns
        -------
        numpy.ndarray
            ``n_eff``, complex, shape ``(n_omega,)``; ``Im n_eff > 0`` is loss.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, "He", 1.0)
        >>> terms = mode.sampled_terms(np.array([2.356e15]))
        >>> value = mode.neff_from_terms(terms, 125e-6, 2.46e25)
        >>> bool(value.imag[0] > 0.0)
        True
        """
        return np.asarray(1.0 + self.refractivity_from_terms(terms, radius, density))

    def refractivity_from_terms(
        self,
        terms: MarcatiliTerms,
        radius: float,
        density: float | Float[np.ndarray, " n_species"],
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return ``n_eff - 1`` from pre-sampled terms and two scalars.

        Parameters
        ----------
        terms : MarcatiliTerms
            The sampled frequency-dependent quantities.
        radius : float
            Core radius in m, > 0.
        density : float
            Number density of the fill in m^-3, >= 0.

        Returns
        -------
        numpy.ndarray
            ``n_eff - 1``, complex, formed without cancellation.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, "He", 1.0, model="reduced")
        >>> terms = mode.sampled_terms(np.array([2.356e15]))
        >>> f"{complex(mode.refractivity_from_terms(terms, 125e-6, 2.46e25)[0]):.3e}"
        '2.881e-05+9.000e-09j'
        """
        refractivity, derivative = self._model_values(terms, radius, density)
        refractivity, derivative = self._apply_loss(
            terms.omega, refractivity, derivative
        )
        return self._floored(refractivity, derivative)[0]

    def dneff_domega_from_terms(
        self,
        terms: MarcatiliTerms,
        radius: float,
        density: float | Float[np.ndarray, " n_species"],
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the analytic ``dn_eff/domega`` from pre-sampled terms.

        Parameters
        ----------
        terms : MarcatiliTerms
            The sampled frequency-dependent quantities.
        radius : float
            Core radius in m, > 0.
        density : float
            Number density of the fill in m^-3, >= 0.

        Returns
        -------
        numpy.ndarray
            ``dn_eff/domega`` in s/rad, complex.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, "He", 1.0, model="reduced")
        >>> terms = mode.sampled_terms(np.array([2.356e15]))
        >>> f"{mode.dneff_domega_from_terms(terms, 125e-6, 2.46e25).real[0]:.4e}"
        '2.6420e-21'
        """
        refractivity, derivative = self._model_values(terms, radius, density)
        refractivity, derivative = self._apply_loss(
            terms.omega, refractivity, derivative
        )
        return self._floored(refractivity, derivative)[1]

    # -------------------------------------------------------- the Mode protocol

    def neff(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the complex effective index.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``n_eff(omega, z)``, complex, shape ``(n_omega,)``; ``Im n_eff > 0`` is
            loss.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, "He", 1.0)
        >>> bool(mode.neff(np.array([2.356e15]))[0].imag > 0.0)
        True
        """
        return np.asarray(1.0 + self.neff_minus_one(omega, z))

    def neff_minus_one(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return ``n_eff - 1`` formed without cancellation.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``n_eff - 1``, complex, exact to rounding for a gas-filled capillary.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, "He", 1.0)
        >>> f"{mode.neff_minus_one(np.array([2.356e15]))[0].real:.4e}"
        '2.8935e-05'
        """
        terms = self.sampled_terms(omega)
        return self.refractivity_from_terms(terms, *self._scalars(z))

    def core_index_minus_one(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the filled core's ``n_core - 1`` formed without cancellation.

        ``n_core = sqrt(1 + chi)`` with ``chi`` the fill's susceptibility at the
        position's density, and ``n_core - 1`` is formed as ``chi / (sqrt(1 + chi) +
        1)``, whose denominator is near two, so it carries the relative rounding of
        ``chi`` itself rather than the absolute rounding of unity. Its difference with
        :meth:`neff_minus_one` is ``n_core - n_eff`` to the rounding of two numbers of
        order ``chi``, where forming ``k - beta`` from the two wavenumbers would carry
        the rounding of ``k``.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``n_core - 1``, complex, shape ``(n_omega,)``; zero for an evacuated core.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, "He", 1.0)
        >>> omega = np.array([2.356e15])
        >>> core = mode.core_index_minus_one(omega)
        >>> bool(core.real[0] > mode.neff_minus_one(omega).real[0] > 0.0)
        True
        """
        terms = self.sampled_terms(omega)
        chi = _core_susceptibility(terms, self._scalars(z)[1])
        return _sqrt1p_minus_one(chi)

    def dneff_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the analytic frequency derivative of the effective index.

        Both models are differentiated in closed form, chaining through the analytic
        derivative of the gas susceptibility and of the cladding index (ADR-006). The
        only term that may be differenced is the imaginary part when the loss is given
        as a callable without its own derivative.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``dn_eff/domega`` in s/rad, complex, shape ``(n_omega,)``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, "He", 1.0)
        >>> f"{mode.dneff_domega(np.array([2.356e15]))[0].real:.4e}"
        '2.6423e-21'
        """
        terms = self.sampled_terms(omega)
        return self.dneff_domega_from_terms(terms, *self._scalars(z))

    def dneff_dz(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the analytic derivative of the effective index along the capillary.

        ``z`` reaches the index through exactly two scalars, the core radius and the
        fill's number density, so ``dn/dz = (dn/da) a' + sum_s (dn/drho_s) rho_s'``
        with the declared profiles' own analytic slopes. For the full model
        ``n^2 = 1 + chi - W`` with ``chi = sum_s gamma_s rho_s`` and
        ``W = (u s)^2 (1 - i v s)^2``, ``s = c/(omega a)``, so that
        ``dW/da = -(2 u^2 s^2/a)(1 - i v s)(1 - 2 i v s)`` and
        ``dn/dz = (sum_s gamma_s rho_s' - (dW/da) a')/(2 n)``; the reduced model
        differentiates its own polynomial. The loss specification acts on the
        imaginary part exactly as it does on the index: ``False`` drops it, a float
        scales it, a measured curve ``alpha(omega)`` does not depend on ``z`` and
        contributes nothing, where the no-gain clamp binds the imaginary part's slope
        is zero, and where the evanescent floor binds the whole slope is zero, as
        :meth:`dneff_domega`'s is. What the coherent wall
        route reads is the real part, the slope of the power weight
        ``w = Re n_eff``.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies, rad/s, shape ``(n_omega,)``, strictly positive.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``dn_eff/dz`` in 1/m, complex, shape ``(n_omega,)``.

        Raises
        ------
        ModeError
            If the mode is a subclass that computes its index another way -- a
            structured cladding or a tabulated index, whose ``z`` derivative is not
            this one -- or a declared profile reports no analytic derivative.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.profiles import Linear
        >>> mode = MarcatiliMode(Linear(50e-6, 40e-6, 1e-3), "He", 1.0)
        >>> bool(mode.dneff_dz(np.array([2.356e15]), 5e-4)[0].real < 0.0)
        True
        """
        if type(self) is not MarcatiliMode:
            msg = (
                f"dneff_dz differentiates the Marcatili index, and "
                f"{type(self).__name__} computes its index another way; it declares "
                "no analytic z derivative"
            )
            raise ModeError(msg)
        for name, path in (
            ("core radius", self.core_radius),
            ("density", self.density),
        ):
            if not path.derivative_is_analytic:
                msg = (
                    f"dneff_dz needs the {name} profile's analytic derivative, and "
                    f"{path!r} reports none: a difference of samples is not a slope"
                )
                raise ModeError(msg)
        terms = self.sampled_terms(omega)
        radius, density = self._scalars(z)
        position = np.array([float(z)])
        slope = float(self.core_radius.derivative(position)[0])
        rates: float | np.ndarray = (
            np.array(
                [
                    float(c.density.derivative(position)[0])
                    for c in self.mixture.components
                ]
            )
            if self.mixture is not None and len(self.mixture.components) > 1
            else float(self.density.derivative(position)[0])
        )
        dchi = _core_susceptibility(terms, rates)
        scale = C / (terms.omega * radius)
        squared = (terms.u_nm * scale) ** 2
        refractivity = self._model_values(terms, radius, density)[0]
        if self.model == "full":
            vs = terms.cladding_term * scale
            dwaveguide = -2.0 * squared / radius * (1.0 - 1j * vs) * (1.0 - 2j * vs)
            derivative = (dchi - dwaveguide * slope) / (2.0 * (1.0 + refractivity))
        else:
            # n - 1 = chi/2 - (u s)^2/2 + i u^2 s^3 v, with d(s^k)/da = -k s^k/a.
            derivative = 0.5 * dchi + squared / radius * slope
            if self.loss is not False:
                derivative = (
                    derivative
                    - 3j * squared * terms.cladding_term * scale / radius * slope
                )
        return self._loss_on_slope(refractivity, np.asarray(derivative, dtype=complex))

    def _loss_on_slope(
        self,
        refractivity: Complex[np.ndarray, " n_omega"],
        derivative: Complex[np.ndarray, " n_omega"],
    ) -> Complex[np.ndarray, " n_omega"]:
        """Apply the loss specification, the no-gain clamp and the floor to a slope.

        The ``z`` counterpart of ``_apply_loss`` and ``_floored``, zeroing the slope
        wherever they zero the frequency derivative.

        Parameters
        ----------
        refractivity : numpy.ndarray
            The model's ``n_eff - 1`` with its own loss, before the specification.
        derivative : numpy.ndarray
            Its ``z`` derivative in 1/m.

        Returns
        -------
        numpy.ndarray
            The ``z`` derivative of what :meth:`neff` returns, in 1/m.
        """
        specification = self.loss
        if callable(specification):
            # An absolute alpha(omega) replaces the imaginary part and has no z in it.
            slope = np.real(derivative) + 0j
        else:
            factor = float(specification)
            refractivity = np.real(refractivity) + 1j * factor * np.imag(refractivity)
            slope = np.real(derivative) + 1j * factor * np.imag(derivative)
            slope = np.where(np.imag(refractivity) < 0.0, np.real(slope), slope)
        if self.model == "full":
            slope = np.where(
                np.real(1.0 + refractivity) < EVANESCENT_INDEX_FLOOR, 0.0, slope
            )
        return np.asarray(slope, dtype=complex)

    def _scalars(
        self, z: float
    ) -> tuple[float, float | Float[np.ndarray, " n_species"]]:
        """Return the core radius and the number density at one position.

        Parameters
        ----------
        z : float
            Position along the waveguide, m.

        Returns
        -------
        tuple of float
            Core radius in m and number density in m^-3 -- the two scalars that carry
            the whole z dependence of the effective index.

        Raises
        ------
        ModeError
            If the core radius is not positive there.
        """
        position = np.array([float(z)])
        radius = float(self.core_radius(position)[0])
        if not radius > 0.0:
            raise ModeError(
                f"the core radius must be positive, got {radius} m at z = {z}"
            )
        if self.mixture is not None and len(self.mixture.components) > 1:
            return radius, self.mixture.density(position)[:, 0]
        return radius, float(self.density(position)[0])

    def aeff(self, z: float = 0.0) -> float:
        """Return the effective area of the mode.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        float
            ``A_eff = (int |e|^2 dA)^2 / int |e|^4 dA`` in m^2, in closed form: the
            transverse profile is a Bessel function of ``r / a``, so the whole radius
            dependence is the prefactor ``a^2``.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> f"{MarcatiliMode(75e-6, 'He', 1.0).aeff():.6e}"
        '8.421575e-09'
        """
        return self._scalars(z)[0] ** 2 * self._aeff_factor

    def normalisation(self, z: float = 0.0) -> float:
        """Return the power carried by the unnormalised transverse profile.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        float
            ``N = (1/2) sqrt(eps_0/mu_0) int |e|^2 dA`` in W, in closed form
            ``(pi/2) a^2 J_{n'}(u)^2 sqrt(eps_0/mu_0)`` with ``n' = n`` for a hybrid
            mode and 2 for a transverse one. The vacuum-normalised basis gives
            monochromatic power ``Re(n_eff) * abs(a)**2`` for internal amplitude ``a``
            (``docs/dev/conventions.md`` section 4).

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> f"{MarcatiliMode(125e-6, 'He', 1.0).normalisation():.6e}"
        '1.755864e-11'
        """
        radius = self._scalars(z)[0]
        order = self.n if self.kind == "HE" else 2
        return float(0.5 * math.pi * radius**2 * jv(order, self.u_nm) ** 2 * EPS0 * C)

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, float], tuple[float, float]]:
        """Return the cross-section parametrisation and the integration limits.

        Parameters
        ----------
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        tuple
            ``("polar", (0, 0), (a(z), 2 pi))`` in m and rad.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> MarcatiliMode(125e-6, 'He', 1.0).dimlimits()[0]
        'polar'
        """
        return "polar", (0.0, 0.0), (self._scalars(z)[0], 2.0 * math.pi)

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Return the transverse field components at points of the cross-section.

        Parameters
        ----------
        points : numpy.ndarray
            Shape ``(n_points, 2)``: ``(r, theta)`` in m and rad.
        z : float, optional
            Position along the waveguide, m.

        Returns
        -------
        numpy.ndarray
            ``(E_x, E_y)`` of the unnormalised profile, shape ``(n_points, 2)``, real.
            A hybrid mode has magnitude ``J_{n-1}(u r/a)`` independent of theta and
            azimuthal order ``|n - 1|`` in its Cartesian components; at ``phi = 0``
            those components are ``(cos((n-1) theta), -sin((n-1) theta))``, the
            ``even`` pattern of every family, so ``HE_{11}`` is uniformly polarised
            along ``x`` there.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.modes import MarcatiliMode
        >>> mode = MarcatiliMode(125e-6, 'He', 1.0)
        >>> mode.field(np.array([[0.0, 0.0]])).round(6)
        array([[1., 0.]])
        """
        radius = self._scalars(z)[0]
        coordinates = np.asarray(points, dtype=float)
        r, theta = coordinates[:, 0], coordinates[:, 1]
        if self.kind == "HE":
            profile = jv(self.n - 1, r * self.u_nm / radius)
            # Reduce the cylindrical-to-Cartesian rotation analytically. Otherwise
            # theta-dependent cancellation creates a spurious orthogonal component
            # even for a uniformly polarised HE_1m mode.
            # phi turns the whole pattern about the axis, from x towards y: the
            # cylindrical components are cos(n(theta - phi)) and -sin(n(theta - phi)),
            # which gives an HE_1m polarisation at the angle phi from x.
            angle = (self.n - 1) * theta - self.n * self.phi
            components = np.stack([np.cos(angle), -np.sin(angle)], axis=-1)
            principal = self.principal_component(z)
            if principal == "x":
                components = np.stack(
                    [np.full_like(theta, math.cos(self.phi)), np.zeros_like(theta)],
                    axis=-1,
                )
            elif principal == "y":
                components = np.stack(
                    [np.zeros_like(theta), np.full_like(theta, math.sin(self.phi))],
                    axis=-1,
                )
        else:
            profile = jv(1, r * self.u_nm / radius)
            components = (
                np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
                if self.kind == "TE"
                else np.stack([np.cos(theta), np.sin(theta)], axis=-1)
            )
        return np.asarray(profile[:, None] * components)

    def _effective_area_factor(self) -> float:
        """Return the radius-independent part of the effective area.

        Returns
        -------
        float
            ``A_eff / a^2``, the ratio of the squared second moment of the transverse
            profile to its fourth moment, evaluated once at construction.
        """
        order = self.n if self.kind == "HE" else 2
        denominator = quad(
            lambda xi: xi * jv(self.n - 1, self.u_nm * xi) ** 4, 0.0, 1.0
        )[0]
        return float(2.0 * math.pi * 0.25 * jv(order, self.u_nm) ** 4 / denominator)


_SHARED_TERMS: ContextVar[
    dict[int, list[tuple[MarcatiliMode, MarcatiliTerms]]] | None
] = ContextVar("solprop_capillary_shared_terms", default=None)
"""The frequency-only terms sampled so far inside :func:`_shared_sampling`, by mode.

Keyed by ``id(mode)`` with the mode itself held beside its terms, so an id cannot be
reused by another mode while the scope lasts; ``None`` outside a scope, where every
call samples afresh exactly as it always has."""


@contextmanager
def _shared_sampling() -> Iterator[None]:
    """Sample each mode's frequency-only terms once per frequency vector in a scope.

    :class:`MarcatiliTerms` do not depend on ``z``, yet a varying coherent wall reads
    ``neff``, ``dneff_dz`` and the block at about a hundred positions for each channel,
    and each read samples the gas and the cladding models again, which dominates the
    translation cost. Inside this scope
    :meth:`MarcatiliMode.sampled_terms` returns the terms it already built for the same
    mode and the same frequencies: the same function of the same inputs, so every array
    built from them is bitwise what it was. The shared
    arrays are made read-only, so a caller that wrote into one would fail loudly rather
    than change what the next caller reads.

    A scope opened inside another is the enclosing one.

    Yields
    ------
    None
        Nothing; the scope ends when the block exits.
    """
    if _SHARED_TERMS.get() is not None:
        # Nested in an enclosing scope, which keeps sharing its terms.
        yield
        return
    token = _SHARED_TERMS.set({})
    try:
        yield
    finally:
        _SHARED_TERMS.reset(token)


def _same_rows(first: tuple[object, ...], second: tuple[object, ...]) -> bool:
    """Return whether two fills' row descriptions name the same models and absences.

    Parameters
    ----------
    first, second : tuple
        :meth:`~solprop.modes.fill.GasFillMixin._fill_rows` of each fill.

    Returns
    -------
    bool
        True when the models are the same objects, in order, with equal absences.
    """
    if len(first) != len(second):
        return False
    for one, other in zip(first, second, strict=True):
        if isinstance(one, tuple) and isinstance(other, tuple):
            if one[0] is not other[0] or one[1] != other[1]:
                return False
        elif isinstance(one, tuple) or isinstance(other, tuple) or one is not other:
            return False
    return True


def _checked_omega(
    omega: Float[np.ndarray, " n_omega"],
) -> Float[np.ndarray, " n_omega"]:
    """Validate an angular-frequency argument for a capillary mode.

    Parameters
    ----------
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The same values as a float array.

    Raises
    ------
    ModeError
        If any value is not finite or not strictly positive.
    """
    values = np.asarray(omega, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ModeError("angular frequency must be finite")
    if np.any(values <= 0.0):
        raise ModeError(
            "a hollow capillary has no mode at zero frequency: the waveguide term "
            "(u c / (omega a))^2 diverges there. Evaluate on the active band of the "
            "grid, which excludes the DC bin."
        )
    return values


@dataclass(frozen=True, eq=False)
class CoherentWall(Displayed):
    r"""Declare the coherent wall block of a capillary mode set.

    The capillary route's diagonal operator gives every mode its own Marcatili
    attenuation and no coupling at all. The **coherent** wall instead condenses the
    outgoing Robin boundary condition onto the retained basis, which makes the wall a
    rank-one term that couples every mode to every other one with a definite phase.
    Declaring one on
    :meth:`solprop.driver.simulation.Simulation.from_capillary` replaces the diagonal
    Marcatili loss of the block's channels by that operator, exactly as
    :func:`wall_operator` builds it.

    Its physical content, in the optics convention of ``docs/dev/conventions.md``
    section 2 and with the carrier removed, is the paraxial generator

    .. math::
       G = D + H,\qquad D = -\frac{i}{2ka^2}\operatorname{diag}(u_m^2),
       \qquad H = g(a)\,hh^{T},

    with :math:`h_m = -\sqrt2\,\operatorname{sgn}J_1(u_m)\,u_m` the radial derivative of
    the orthonormal basis function at the wall, and
    :math:`g(a) = i/[2ka^2(2N - i k_{\rm eff}a)]`. The manual page is
    ``docs/guide/waveguides/coherent-wall.md``.

    **A guide that varies along z** -- a pressure gradient, a taper -- is carried by
    a frozen-residual split: the complete generator ``G_s = D(z_s) + H(z_s)`` is
    frozen at each section's midpoint inside the exponential flow, and its variation
    joins the HE1m connection

    .. math::
       C_{mn} = \frac{a'}{a}\frac{2s_ms_nu_mu_n}{u_m^2-u_n^2}\ (m\ne n),
       \qquad C_{nn} = 0

    in the **explicit residual**
    ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)``, which the interaction-picture
    Runge-Kutta source carries. Freezing is a rearrangement of the same equation and
    not a second-order approximation, because the residual is evaluated exactly;
    `sections` is what trades the residual's stiffness against the number of frozen
    blocks (:func:`solprop.translate.coupled.frozen_residual_flow`).

    **What it replaces, and what it therefore refuses.** The block is the run's loss
    model, so a capillary carrying one must declare ``loss=True``: ``loss=False``, a
    scale or a measured curve is refused by name rather than discarded. It is derived
    for the full capillary model, so ``model="reduced"`` is refused too. It needs the
    contiguous ``HE_11`` to ``HE_1N`` set, whose every channel declares the same radius
    and density profiles and whose profiles carry analytic derivatives, and a scalar
    run. It carries the default ``"input"`` noise, drawn in the channel basis at the
    entrance and propagated like any field, and refuses the ``"modified"`` model, whose
    transport through a nonunitary coupled flow is not measured. All three backends
    execute the ``linear.coupled_flow`` leaf it becomes (``method="cf4"`` on
    ``numpy_ref`` alone).

    **The vector boundary response** is ``impedance="te_tm"``. The scalar
    block above condenses one *impedance-averaged* Robin condition, which is what a
    linearly polarised ``HE_1m`` field sees on azimuthal average; the vector block
    condenses the two conditions the wall actually imposes -- the radial (TM-like)
    field sees ``k_o/n_r^2`` and the tangential (TE-like) field sees
    ``k_o`` -- and is therefore a block **per total azimuthal order**, coupling the two
    radial families of that order rather than one. See :func:`vector_wall_operator`;
    the averaged impedance is recovered from it as a limit rather than assumed.

    Attributes
    ----------
    impedance : str
        ``"mean"`` for the impedance-averaged block, ``"te_tm"`` for the coupled
        azimuthal block, which gives the wall's TE (tangential) and TM (radial) fields
        their own impedances. It names the **wall's** two impedances and not the state's
        polarisation axis: a ``"te_tm"`` run still carries one transverse component per
        channel. On a guide that varies along z -- a taper, a pressure gradient -- the
        vector block takes the same frozen or CF4 route as the scalar one, with a
        connection that is each radial family's own and nothing between the families
        (:func:`wall_connection`). Which families the run carries is the **mode set's**
        business and not this field's: a set retaining one hybrid family alone is a
        legitimate vector block with no partner to couple to, and is the scalar block to
        the last bit, because that family's Bessel order is zero and its diagonal
        impedance *is* the average. Declare the partner family -- the hybrids of
        azimuthal index ``-nu`` -- to have a conversion at all. The transverse order
        ``nu = 0`` is not like that: ``TE_0m`` and ``TM_0m`` carry their own impedances
        whether or not the other is retained.
    condition_limit : float
        Largest admitted ``norm(U) norm(U^-1)`` of one step's flow action,
        dimensionless and at least one. The inverse action is what the
        interaction-picture source is carried back by, so an unbounded inverse is an
        unbounded amplification of that source; the
        backend measures the product per step and refuses above this. The default
        1e4 admits a step that attenuates the most lossy retained mode by about
        nine e-foldings.
    validity_limit : float
        Largest admitted value of the three dimensionless validity ratios of
        :class:`WallLimits`. Above it the block is **refused** by name; above a tenth
        of it the construction warns, naming the ratio. The default 0.25 keeps the
        retained transverse scale well inside the paraxial and quasi-static regimes.
    sections : int
        How many equal sections each interval of the run's own linear-operator table
        is divided into, a positive integer. The **placement** is not this number's:
        the edges are the table's breakpoints, which already land on every declared
        kink of the radius and the density, subdivided `sections` times. One, the
        default, freezes one generator per declared interval; raising it reduces the
        residual's norm and the inverse amplification, and is not permission to
        discard the residual. It has no effect on a constant guide, whose residual is
        exactly zero and which takes the single-section constant-Schur flow whatever
        this says.
    method : str
        How a guide that varies along z is integrated. ``"frozen_residual"``, the
        default, is the frozen-residual split: each section's
        generator frozen inside the exponential and its variation in the explicit
        residual the Runge-Kutta source carries, fifth order. ``"cf4"`` integrates the
        **whole** section generator by the fourth-order commutator-free exponential,
        with no explicit residual and so no advisory step, but at most **fourth**
        order and with a flow error the embedded pair does not see
        (:func:`solprop.translate.coupled.cf4_flow`). It is admitted only behind two
        guards: the internal-growth bound of the CF4 factors against `growth_limit`
        and the pair
        conditioning against `condition_limit`, for either impedance. It is refused
        by name on a guide that does not vary at all, whose flow the constant-Schur
        action already integrates exactly.
    growth_limit : float or None
        Largest admitted internal-growth bound of a CF4 section, dimensionless and at
        least one; ``None`` takes
        :data:`solprop.translate.coupled.CF4_GROWTH_LIMIT`. Declaring it without
        ``method="cf4"`` is refused, because nothing would read it.
    fallback : str or None
        What a section the CF4 guards reject becomes: ``None`` refuses it by name,
        ``"frozen_residual"`` integrates it with the frozen-residual split and warns
        naming it.
        Declaring it without ``method="cf4"`` is refused.

    Raises
    ------
    ModeError
        If a limit or the section count is out of range, or if a method, a growth
        limit or a fallback is declared where nothing would honour it.

    Examples
    --------
    >>> from solprop.modes import CoherentWall
    >>> CoherentWall().impedance, CoherentWall(impedance="te_tm").impedance
    ('mean', 'te_tm')
    >>> repr(CoherentWall())[:43]
    'CoherentWall(impedance=mean, condition_limi'
    """

    condition_limit: float = 1.0e4
    validity_limit: float = 0.25
    sections: int = 1
    impedance: WallImpedance = "mean"
    method: Literal["frozen_residual", "cf4"] = "frozen_residual"
    growth_limit: float | None = None
    fallback: Literal["frozen_residual"] | None = None

    def __post_init__(self) -> None:
        """Check the wall impedance, both limits and the section count.

        Raises
        ------
        ModeError
            If the wall impedance is not one this package derives, if a limit is
            not finite or outside the range its meaning allows, or if the section
            count is not a positive integer.
        """
        if self.impedance not in ("mean", "te_tm"):
            msg = (
                "CoherentWall.impedance is 'mean' for the impedance-averaged block "
                "or 'te_tm' for the coupled azimuthal one with the wall's own TE and "
                f"TM impedances, got {self.impedance!r}"
            )
            raise ModeError(msg)
        try:
            # `operator.index` is the integer test, not `int(x) != x`: the latter
            # admits 2.0, which then translates, runs and only fails in `__repr__`,
            # where the count is formatted with "d". A numpy integer is still an
            # integer here, and a bool is excluded by name below.
            count = operator.index(self.sections)
        except TypeError:
            count = 0
        if isinstance(self.sections, bool) or count < 1:
            msg = (
                "CoherentWall.sections is how many equal parts each interval of the "
                "operator table is divided into: a positive integer, and a flow "
                f"carries at least one frozen section. Got {self.sections!r}"
            )
            raise ModeError(msg)
        if not math.isfinite(self.condition_limit) or self.condition_limit < 1.0:
            msg = (
                "CoherentWall.condition_limit is the conditioning of a flow action "
                f"and is at least one, got {self.condition_limit!r}"
            )
            raise ModeError(msg)
        if not 0.0 < self.validity_limit <= 1.0:
            msg = (
                "CoherentWall.validity_limit is a dimensionless ratio in (0, 1], got "
                f"{self.validity_limit!r}"
            )
            raise ModeError(msg)
        self._check_method()

    def _check_method(self) -> None:
        """Check the varying-section method and the two declarations only CF4 reads.

        Raises
        ------
        ModeError
            If the method is unknown, or if a growth limit or a fallback is declared
            without CF4 or out of range.
        """
        if self.method not in ("frozen_residual", "cf4"):
            msg = (
                "CoherentWall.method is 'frozen_residual' or 'cf4' for a guide that "
                f"varies along z, got {self.method!r}"
            )
            raise ModeError(msg)
        if self.method != "cf4" and (
            self.growth_limit is not None or self.fallback is not None
        ):
            msg = (
                "CoherentWall.growth_limit and fallback are the CF4 guards' and are "
                f"read only with method='cf4'; this declares method={self.method!r} "
                f"with growth_limit={self.growth_limit!r} and "
                f"fallback={self.fallback!r}"
            )
            raise ModeError(msg)
        if self.fallback not in (None, "frozen_residual"):
            msg = (
                "CoherentWall.fallback is None, which refuses a section the CF4 "
                "guards reject, or 'frozen_residual', which integrates it with the "
                f"frozen split; got {self.fallback!r}"
            )
            raise ModeError(msg)
        if self.growth_limit is not None and not (
            math.isfinite(self.growth_limit) and self.growth_limit >= 1.0
        ):
            msg = (
                "CoherentWall.growth_limit bounds the product of CF4's internal "
                "factors against the exact action's and is at least one, got "
                f"{self.growth_limit!r}"
            )
            raise ModeError(msg)

    def __repr__(self) -> str:
        """Return a one-line description.

        Returns
        -------
        str
            For example ``CoherentWall(impedance=mean, condition_limit=10000,
            validity_limit=0.25, sections=1)``; a CF4 wall adds its method, growth
            limit and fallback, which the default frozen split does not carry.
        """
        rows = [
            ("impedance", self.impedance),
            ("condition_limit", f"{self.condition_limit:g}"),
            ("validity_limit", f"{self.validity_limit:g}"),
            ("sections", f"{self.sections:d}"),
        ]
        if self.method == "cf4":
            limit = "default" if self.growth_limit is None else f"{self.growth_limit:g}"
            rows += [
                ("method", self.method),
                ("growth_limit", limit),
                ("fallback", str(self.fallback)),
            ]
        return line("CoherentWall", rows)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "CoherentWall: the condensed Robin wall of a capillary block", [
            ("wall impedance", self.impedance),
            ("condition limit", f"{self.condition_limit:g}"),
            ("validity limit", f"{self.validity_limit:g}"),
            ("sections per interval", f"{self.sections:d}"),
            ("varying method", self._method_text()),
            ("model", self._model_text()),
        ]

    def _model_text(self) -> str:
        """Return the printed one-line model the declaration builds.

        Returns
        -------
        str
            The wall impedance, the paraxial generator and the flow it takes.
        """
        wall = "TE/TM impedances" if self.impedance == "te_tm" else "mean impedance"
        if self.method == "cf4":
            return f"{wall}, paraxial, commutator-free per section"
        return f"{wall}, paraxial, frozen per section"

    def _method_text(self) -> str:
        """Return the printed varying-section method with the CF4 guards it declares.

        Returns
        -------
        str
            For example ``cf4, growth limit 2 (default), fallback frozen_residual``.
        """
        if self.method != "cf4":
            return self.method
        limit = "default" if self.growth_limit is None else f"{self.growth_limit:g}"
        return f"cf4, growth limit {limit}, fallback {self.fallback}"


@dataclass(frozen=True, eq=False)
class WallLimits(Displayed):
    r"""The dimensionless validity ratios of one coherent wall block.

    Each is evaluated with the **highest retained** mode over the band the run scores
    (``band`` below), which is where every approximation in it is worst there.
    None of them is a proof of Maxwell accuracy: they are the stated conditions under
    which the condensed paraxial model is the model it claims to be. In particular a
    converged paraxial Robin equation is not by itself Maxwell accuracy, and many
    retained basis components can exceed the in-core wavenumber without being
    propagating channels.

    Attributes
    ----------
    transverse_ratio : float
        ``u_N / (|k| a)`` with ``k`` the **in-core** wavenumber: how far the highest
        retained transverse scale is from paraxial. The paraxial generator's own
        error is a quarter of its fourth power relative to the carrier.
    vacuum_ratio : float
        ``u_N / (k_0 a)`` with ``k_0`` the vacuum wavenumber: the grazing-incidence
        ratio the Marcatili expansion is asymptotic in, which the cladding term is
        referred to.
    quasi_static : float
        ``2N / |a k_eff|``: the fraction of the eliminated boundary coordinate the
        condensation discards in the quasi-static wall limit.
        Raising ``N`` raises this, so it cannot by itself validate the condensation.
        A **vector** block takes the same ratio family by family and keeps the worst:
        ``(2N_f + p_f) sum_g |Z_fg| / a``, with ``p_f`` the family's Bessel order and
        ``Z`` its impedance matrix in m, which is the row sum the condensed
        ``S = (C - i a Z^-1)^-1`` departs from ``i Z/a`` by
        (:func:`vector_wall_operator`).
    band : tuple of float
        Lowest and highest angular frequency in rad/s the ratios were **taken** over.
        On a run this is the error norm's **scored band**
        (:attr:`solprop.grid.TimeGrid.inband_mask`, the requested wavelength limits),
        not the active band the operator is built on: a field grid's active band
        reaches about twice the longest requested wavelength, where every ratio (each
        proportional to the wavelength) is about twice its in-band value, and those
        collar bins are absorbed rather than scored.
        Without a scored mask, or with one that holds no evaluated bin, it is the
        whole evaluated band.
    n_modes : int
        How many channels the block carries, over every family in it.
    radius : float
        Core radius in m the ratios were taken at.

    Examples
    --------
    >>> from solprop.modes.capillary import WallLimits
    >>> limits = WallLimits(0.01, 0.01, 0.02, (1e15, 3e15), 4, 125e-6)
    >>> f"{limits.worst:.3f}"
    '0.020'
    """

    transverse_ratio: float
    vacuum_ratio: float
    quasi_static: float
    band: tuple[float, float]
    n_modes: int
    radius: float

    @property
    def worst(self) -> float:
        """Return the largest of the three ratios.

        Returns
        -------
        float
            The one a refusal or a warning is taken on.
        """
        return max(self.transverse_ratio, self.vacuum_ratio, self.quasi_static)

    def named(self) -> tuple[str, float]:
        """Return the name and the value of the ratio that is worst.

        Returns
        -------
        tuple
            The ratio's name as it is written here, and its value.

        Examples
        --------
        >>> from solprop.modes.capillary import WallLimits
        >>> WallLimits(0.3, 0.1, 0.2, (1e15, 3e15), 4, 125e-6).named()[0]
        'transverse_ratio'
        """
        rows = (
            ("transverse_ratio", self.transverse_ratio),
            ("vacuum_ratio", self.vacuum_ratio),
            ("quasi_static", self.quasi_static),
        )
        return max(rows, key=lambda row: row[1])

    def __repr__(self) -> str:
        """Return a one-line description.

        Returns
        -------
        str
            For example ``WallLimits(4 channels, worst transverse_ratio=0.0123)``.
        """
        name, value = self.named()
        return line(
            "WallLimits",
            [
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("worst", f"{name}={value:.4g}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return "WallLimits: where the condensed paraxial wall model stops being one", [
            ("channels", self.n_modes),
            ("radius", engineering(self.radius, "m")),
            ("u_N/(k a)", f"{self.transverse_ratio:.4g}"),
            ("u_N/(k_0 a)", f"{self.vacuum_ratio:.4g}"),
            ("2N/|a k_eff|", f"{self.quasi_static:.4g}"),
            (
                "band",
                f"{engineering(self.band[0], 'rad/s')} to "
                f"{engineering(self.band[1], 'rad/s')}",
            ),
        ]


@dataclass(frozen=True, eq=False)
class WallOperator(Displayed):
    r"""The condensed Robin wall generator of a contiguous ``HE_1m`` block.

    The output of :func:`wall_operator`: the dense paraxial generator in the **optics**
    convention and in 1/m, together with the factored pieces it was built from and the
    validity ratios it was checked against. Crossing to the internal FFT convention is
    the complex conjugate of the **whole** equation, ``c' = G c`` becoming
    ``c' = conj(G) c``, which is what the translation layer stores; it is not ``-G``,
    and a passive mode propagated with the wrong one grows
    (``docs/dev/conventions.md`` section 2, ADR-017).

    Attributes
    ----------
    omega : numpy.ndarray
        The angular frequencies in rad/s the block is tabulated on, shape
        ``(n_omega,)``; the whole grid, with the inactive bins left exactly zero.
    active : numpy.ndarray
        Where the operator was evaluated, bool, shape ``(n_omega,)``.
    generator : numpy.ndarray
        ``G = D + H`` in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``, in the
        optics convention with the carrier ``k`` removed.
    trace_vector : numpy.ndarray
        ``h_m``, dimensionless, shape ``(n_modes,)``: the radial derivative of the
        ``m``-th orthonormal basis function at the wall.
    coupling : numpy.ndarray
        ``g(a)`` in 1/m, complex, shape ``(n_omega,)``, zero off the active band.
    wavenumber : numpy.ndarray
        The in-core wavenumber ``k = n_core omega / c`` in 1/m, complex, shape
        ``(n_omega,)``, zero off the active band.
    impedance : numpy.ndarray
        ``k_eff = k_0 / v_n`` in 1/m, complex, shape ``(n_omega,)``, zero off the
        active band: the vacuum wavenumber over the **current** filled-core cladding
        term, which is what makes the isolated block reproduce this package's own
        capillary loss.
    limits : WallLimits
        The validity ratios over the active band.
    n_modes : int
        How many channels the block carries.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, wall_operator
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=index) for index in (1, 2)]
    >>> block = wall_operator(modes, np.array([2.356e15]))
    >>> block.generator.shape
    (2, 2, 1)
    >>> bool(block.generator[0, 0, 0].real < 0.0)
    True
    """

    omega: Float[np.ndarray, " n_omega_wall"]
    active: Bool[np.ndarray, " n_omega_wall_a"]
    generator: Complex[np.ndarray, "n_modes_wall_out n_modes_wall_in n_omega_wall_g"]
    trace_vector: Float[np.ndarray, " n_modes_wall_h"]
    coupling: Complex[np.ndarray, " n_omega_wall_g2"]
    wavenumber: Complex[np.ndarray, " n_omega_wall_k"]
    impedance: Complex[np.ndarray, " n_omega_wall_e"]
    limits: WallLimits

    @property
    def n_modes(self) -> int:
        """Return how many channels the block couples.

        Returns
        -------
        int
            The length of the trace vector.
        """
        return len(self.trace_vector)

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``WallOperator(2 channels, 1 frequency, worst 0.0123)``.
        """
        count = int(np.count_nonzero(self.active))
        return line(
            "WallOperator",
            [
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("", f"{count} frequenc{'y' if count == 1 else 'ies'}"),
                ("worst", f"{self.limits.worst:.4g}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        band = self.generator[:, :, self.active]
        loss = -np.real(np.einsum("mmw->mw", band)) if band.size else np.zeros((1, 1))
        return "WallOperator: the condensed Robin wall of an HE_1m block", [
            ("channels", self.n_modes),
            ("frequencies", f"{int(np.count_nonzero(self.active))} active"),
            ("largest field loss", f"{float(np.max(loss)):.4g} 1/m"),
            ("largest coupling", f"{float(np.max(abs(self.coupling))):.4g} 1/m"),
            ("limits", repr(self.limits)),
        ]


@dataclass(frozen=True)
class WallFamily:
    r"""One radial family of a vector coherent wall block.

    A vector block is one **total azimuthal order** :math:`\nu` of a capillary, and
    that order has at most two radial families, because the transverse field of an
    order-:math:`\nu` mode is a combination of the two circular components
    :math:`\hat e_\pm`, whose Cartesian angular harmonics are :math:`\nu\mp1` and
    whose radial functions are therefore :math:`J_{\nu-1}` and :math:`J_{\nu+1}`.
    Each family is a contiguous ``m = 1 .. N_f`` set of
    this package's own Marcatili channels.

    Attributes
    ----------
    label : str
        How the family prints: ``"HE_1m"``, ``"EH_1m"``, ``"TE_0m"``, ``"TM_0m"`` and
        the higher orders in the same spelling.
    azimuthal_order : int
        The channels' own :attr:`MarcatiliMode.n`. Within one sector the two families
        carry ``+nu`` and ``-nu``; ``nu = 0`` carries the two transverse kinds.
    bessel_order : int
        ``p``, the order of the Bessel function the family's radial basis is built
        from, ``>= 0``: ``|n - 1|`` for a hybrid family and ``1`` for a transverse
        one.
    n_channels : int
        How many radial orders of this family the block retains, ``>= 1``.
    angular : tuple of float
        The family's **angular content at the wall**, ``(radial, tangential)``, a unit
        two-vector: the fraction of its boundary field that is normal to the wall and
        the fraction tangential to it. ``(1, 0)`` for ``TM_0m`` and ``(0, 1)`` for
        ``TE_0m``, which is why those two see the wall's own two impedances; and
        ``(1, -1)/sqrt 2`` and ``(1, 1)/sqrt 2`` for the two hybrid families of an
        order ``nu >= 1``, which is why *those* see the impedance **average** on the
        diagonal and the impedance **difference** between them.
    radial_sign : float
        The sign of the family's **implemented** radial profile against ``J_p``,
        ``+1`` or ``-1``. :meth:`MarcatiliMode.field` builds a hybrid from
        ``jv(n - 1, .)``, and ``J_{-p} = (-1)^p J_p``, so the ``n = -nu`` family of an
        **even** order is implemented with the opposite sign to ``J_{nu+1}``. It is a
        gauge of the mode and cancels out of every diagonal quantity, but it is the
        sign of the **coupling** between the two families, so the wall's boundary
        traces carry it.

    Examples
    --------
    >>> from solprop.modes.capillary import WallFamily
    >>> family = WallFamily("TE_0m", 0, 1, 4, (0.0, 1.0), 1.0)
    >>> family.truncation
    9
    """

    label: str
    azimuthal_order: int
    bessel_order: int
    n_channels: int
    angular: tuple[float, float]
    radial_sign: float

    @property
    def truncation(self) -> int:
        """Return the family's boundary truncation constant ``2 N_f + p_f``.

        Returns
        -------
        int
            The constant the condensation's denominator carries for this family: the
            truncated boundary self term ``sum_m h_m^2/u_m^2 = 2 N_f`` of the retained
            orders plus the ``p_f`` of the exact recurrence
            ``q J_{p+1}(q) = (p - i eta) J_p(q)``. Dropping the ``p_f`` leaves a
            residue that does **not** fall with ``N``.
        """
        return 2 * self.n_channels + self.bessel_order


@dataclass(frozen=True, eq=False)
class VectorWallOperator(Displayed):
    r"""The condensed vector Robin wall generator of one azimuthal order.

    The output of :func:`vector_wall_operator`, and the vector counterpart of
    :class:`WallOperator`: the dense paraxial generator in the **optics** convention
    and in 1/m, the pieces it was factored from and the validity ratios it was checked
    against. Crossing to the internal FFT convention is the complex conjugate of the
    **whole** equation, exactly as for the scalar block.

    The wall term is **rank two, not rank one**: one boundary trace vector per family,
    weighted by the condensed ``2 x 2`` impedance of the order.

    Attributes
    ----------
    omega : numpy.ndarray
        The angular frequencies in rad/s the block is tabulated on, shape
        ``(n_omega,)``; the whole grid, with the inactive bins left exactly zero.
    active : numpy.ndarray
        Where the operator was evaluated, bool, shape ``(n_omega,)``.
    generator : numpy.ndarray
        ``G = D + H`` in 1/m, complex, shape ``(n_modes, n_modes, n_omega)``, in the
        optics convention with the carrier ``k`` removed.
    families : tuple of WallFamily
        The one or two radial families of this order, in the state's channel order.
    sector : int
        The total azimuthal order ``nu >= 0`` every channel of the block belongs to.
    family_traces : numpy.ndarray
        The boundary traces ``h_m`` arranged one column per family, dimensionless,
        shape ``(n_modes, n_families)``, exactly zero outside each family's own
        channels. The wall term is ``V W V^T`` with ``V`` this matrix.
    weight : numpy.ndarray
        ``W = i S/(2 k a^2)`` in 1/m, complex, shape
        ``(n_families, n_families, n_omega)``: the condensed impedance of the order,
        scaled by the paraxial reduction. Zero off the active band.
    wavenumber : numpy.ndarray
        The in-core wavenumber ``k = n_core omega / c`` in 1/m, complex, shape
        ``(n_omega,)``, zero off the active band.
    impedance : numpy.ndarray
        The wall's own two impedances in m, complex, shape ``(2, n_omega)``: the
        radial (TM-like) ``v_TM/k_0`` and the tangential (TE-like) ``v_TE/k_0``, built
        from this package's own Marcatili cladding terms. Their mean is the scalar
        block's ``1/k_eff`` and their half-difference is the anisotropy.
    limits : WallLimits
        The validity ratios over the active band.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, vector_wall_operator
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, kind=k, n=0) for k in ("TM", "TE")]
    >>> block = vector_wall_operator(modes, np.array([2.356e15]))
    >>> block.sector, block.n_families, block.generator.shape
    (0, 2, (2, 2, 1))
    >>> bool(abs(block.generator[0, 1, 0]) == 0.0)
    True
    """

    omega: Float[np.ndarray, " n_omega_vec"]
    active: Bool[np.ndarray, " n_omega_vec_a"]
    generator: Complex[np.ndarray, "n_modes_vec_out n_modes_vec_in n_omega_vec_g"]
    families: tuple[WallFamily, ...]
    sector: int
    family_traces: Float[np.ndarray, "n_modes_vec_h n_families_vec_h"]
    weight: Complex[np.ndarray, "n_families_w_out n_families_w_in n_omega_vec_w"]
    wavenumber: Complex[np.ndarray, " n_omega_vec_k"]
    impedance: Complex[np.ndarray, "n_wall_components n_omega_vec_z"]
    limits: WallLimits

    @property
    def n_modes(self) -> int:
        """Return how many channels the block couples.

        Returns
        -------
        int
            The total over its families.
        """
        return sum(family.n_channels for family in self.families)

    @property
    def n_families(self) -> int:
        """Return how many radial families the block carries.

        Returns
        -------
        int
            One or two.
        """
        return len(self.families)

    def __repr__(self) -> str:
        """Return a one-line description, never an array.

        Returns
        -------
        str
            For example ``VectorWallOperator(nu=1, HE_1m+EH_1m, 4 channels,
            1 frequency, worst 0.0123)``.
        """
        count = int(np.count_nonzero(self.active))
        return line(
            "VectorWallOperator",
            [
                ("nu", f"{self.sector}"),
                ("", "+".join(family.label for family in self.families)),
                ("", f"{self.n_modes} channel{'' if self.n_modes == 1 else 's'}"),
                ("", f"{count} frequenc{'y' if count == 1 else 'ies'}"),
                ("worst", f"{self.limits.worst:.4g}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        band = self.generator[:, :, self.active]
        loss = -np.real(np.einsum("mmw->mw", band)) if band.size else np.zeros((1, 1))
        anisotropy = self.impedance[0] - self.impedance[1]
        mean = self.impedance[0] + self.impedance[1]
        ratio = abs(anisotropy[self.active]) / abs(mean[self.active])
        return "VectorWallOperator: the condensed vector wall of one azimuthal order", [
            ("azimuthal order", self.sector),
            (
                "families",
                ", ".join(
                    f"{family.label} (J_{family.bessel_order}, {family.n_channels})"
                    for family in self.families
                ),
            ),
            ("channels", self.n_modes),
            ("frequencies", f"{int(np.count_nonzero(self.active))} active"),
            ("largest field loss", f"{float(np.max(loss)):.4g} 1/m"),
            ("wall anisotropy", f"{float(np.max(ratio)):.4g} of the mean impedance"),
            ("limits", repr(self.limits)),
        ]


def _wall_block_modes(
    modes: Sequence[Mode], *, wall: bool = True
) -> tuple[MarcatiliMode, ...]:
    """Check a mode set is one contiguous scalar ``HE_11`` to ``HE_1N`` block.

    The scalar regulariser condenses the boundary coordinate
    onto the *complete* set of ``J_0`` radial orders, so a sparse or mixed set is not a
    smaller block but a different operator: the ``2N`` self term counts the retained
    orders, and a missing order leaves its share of the boundary trace nowhere to go.
    The set is therefore rejected by name rather than promoted, because promoting it
    would silently change the channels the caller asked to propagate and save.

    Two of the mode's own settings are refused here rather than ignored, because the
    block does ignore them. **The wall is the loss model**: the condensed Robin
    boundary condition *is* where every channel's attenuation comes from, the
    assembly subtracts the whole ``HE_11`` wavenumber and re-adds the block's own
    diagonal, and :func:`wall_operator` builds its impedance from the cladding term
    whatever :attr:`MarcatiliMode.loss` says -- so ``loss=False``, a scale or a
    measured curve beside a wall would vanish without a word.
    And the condensation inverts the **full** model's transverse eigenvalue, so
    ``model="reduced"``, whose leading-order expansion of ``n_eff`` is a different
    carrier, would be overridden by the block it sits beside.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set of the run, in the state's channel order. It is typed as the
        protocol rather than as :class:`MarcatiliMode` so that a set carrying another
        kind of mode reaches the refusal below and is named, instead of failing the
        runtime type check the test session installs.
    wall : bool, optional
        Whether the block is a coherent wall's. The **taper coupling** alone
        (:func:`taper_connection`) reads the radial orders and the radius, never the
        loss or the capillary model, so ``False`` admits any loss and either model
        and keeps every other condition.

    Returns
    -------
    tuple of MarcatiliMode
        The same modes, as a tuple.

    Raises
    ------
    ModeError
        Naming what makes the set incompatible.
    """
    block = tuple(modes)
    if not block:
        raise ModeError("a coherent wall block needs at least one HE_1m mode")
    for index, mode in enumerate(block):
        if isinstance(mode, MarcatiliMode) and type(mode) is not MarcatiliMode:
            msg = (
                "a coherent wall block assumes one uniform dielectric wall at a "
                f"single radius, and channel {index} is a {type(mode).__name__}: a "
                "structured cladding's loss is its own resonances, not a Robin "
                "condition on a bare capillary, and condensing this basis would give "
                "the structure the bare capillary's wall. Leave wall=None and keep "
                "the guide's own loss (a structured wall is not implemented)"
            )
            raise ModeError(msg)
    labels = []
    for mode in block:
        if not isinstance(mode, MarcatiliMode):
            labels.append(type(mode).__name__)
        else:
            labels.append(f"{mode.kind}_{mode.n}{mode.m}")
    wanted = [f"HE_1{index + 1}" for index in range(len(block))]
    if labels != wanted:
        msg = (
            "a coherent wall block needs the contiguous Marcatili set HE_11 to "
            f"HE_1{len(block)} in that order, and this run carries "
            f"{', '.join(labels)}. Propagate the complete block, or leave wall=None "
            "and keep the uncoupled Marcatili loss"
        )
        raise ModeError(msg)
    _check_wall_channels(cast("tuple[MarcatiliMode, ...]", block), wall=wall)
    return cast("tuple[MarcatiliMode, ...]", block)


def _wall_channel_settings(mode: MarcatiliMode) -> tuple[tuple[str, object], ...]:
    """Return everything the wall block reads from one channel but its radial order.

    The operator is built from ``block[0]`` alone -- its radius, its density, its
    cladding term and its capillary model -- and the retained orders enter only
    through ``u_m``. Every other channel must therefore declare the *same* guide, or
    the block silently propagates the first channel's one. These are the settings
    compared: the two **profiles** by their identity rather than by a value at one
    position, because a block frozen per section reads them at every section midpoint
    and two tapers that agree at the entrance are two different guides everywhere
    else.

    Parameters
    ----------
    mode : MarcatiliMode
        One channel of the block.

    Returns
    -------
    tuple
        ``(name, value)`` pairs, comparable and printable.
    """
    return (
        ("phi", mode.phi),
        (
            "temperature",
            _profile_key(mode.temperature)
            if isinstance(mode.temperature, ZProfile)
            else mode.temperature,
        ),
        ("core radius", _profile_key(mode.core_radius)),
        ("density", _profile_key(mode.density)),
    )


def _profile_key(profile: ZProfile) -> str:
    """Return a comparable, printable identity of one declared z profile.

    Parameters
    ----------
    profile : solprop.profiles.ZProfile
        The declared path.

    Returns
    -------
    str
        The sorted identity mapping of :func:`solprop.profiles.profile_identity`,
        rendered so that two paths compare equal exactly when the identity does. A
        path that has no structural identity -- an arbitrary callable -- is keyed on
        the object itself, so two channels sharing one declared path agree and two
        that were handed separate callables are refused, which is the conservative
        answer for something this package cannot inspect.
    """
    try:
        identity = profile_identity(profile)
    except TranslationError:
        return f"a declared callable at 0x{id(profile):x}"
    return repr(sorted(identity.items()))


def _check_wall_channels(
    block: tuple[MarcatiliMode, ...], *, wall: bool = True
) -> None:
    """Refuse a block whose channels do not declare one guide, by name.

    Parameters
    ----------
    block : tuple of MarcatiliMode
        The contiguous ``HE_11`` to ``HE_1N`` set, in channel order.
    wall : bool, optional
        Whether the block is a coherent wall's, which is the loss model and needs the
        full capillary model. ``False`` -- the taper coupling alone -- skips those two
        conditions and names the taper coupling in the others.

    Raises
    ------
    ModeError
        Naming the channel and the loss or capillary model the block would override,
        the channel whose section varies, or the setting two channels disagree on.
    """
    owner = "a coherent wall block" if wall else "the taper coupling"
    for index, mode in enumerate(block):
        if wall and mode.model != "full":
            msg = (
                "a coherent wall block needs one capillary model, the full one, and "
                f"channel {index} declares model={mode.model!r}: the condensation "
                "inverts the full model's transverse eigenvalue, and the reduced "
                "model's leading-order carrier beside it would be overridden rather "
                "than propagated"
            )
            raise ModeError(msg)
        if wall and mode.loss is not True:
            msg = (
                "a coherent wall block is the loss model, and channel "
                f"{index} declares loss={mode.loss!r}: the condensed Robin boundary "
                "condition replaces the Marcatili attenuation of every channel, so a "
                "loss setting beside it would be discarded in silence. Declare "
                "loss=True with a wall, or leave wall=None and keep the uncoupled "
                "loss you asked for"
            )
            raise ModeError(msg)
        for name, path in (
            ("core radius", mode.core_radius),
            ("density", mode.density),
        ):
            if not path.derivative_is_analytic:
                msg = (
                    f"{owner} frozen per section differentiates the "
                    f"declared profiles, and channel {index}'s {name} reports no "
                    "analytic derivative: the taper connection is a'/a and a "
                    "difference of a sampled profile is noise, not a slope. Declare "
                    "a profile with an analytic derivative, or supply the slope"
                    + (
                        ""
                        if wall
                        else "; or pass taper_coupling=False for the adiabatic route"
                    )
                )
                raise ModeError(msg)
    reference = _wall_channel_settings(block[0])
    for index, mode in enumerate(block[1:], start=1):
        for (name, mine), (_, theirs) in zip(
            _wall_channel_settings(mode), reference, strict=True
        ):
            if mine != theirs:
                msg = (
                    f"{owner} is built from one guide and reads it from "
                    f"the lowest channel alone; channel {index} declares "
                    f"{name}={mine!r} against {theirs!r}. Every channel shares one "
                    "capillary, one fill and one polarisation angle, the "
                    "mean-impedance block and the TE/TM one (impedance='te_tm') "
                    "alike"
                )
                raise ModeError(msg)


def _check_wall_guide(
    block: tuple[MarcatiliMode, ...],
    band: Float[np.ndarray, " n_active"],
    *,
    probe: ModeKind | None = None,
) -> None:
    """Refuse a block whose channels sample different cladding or fill, by name.

    The declared settings are compared in :func:`_check_wall_channels`; this compares
    what the operator actually reads, which is the only decisive test, because two
    channels can name their cladding differently and mean the same glass. The
    cladding term and the per-particle susceptibility come out of the same code on
    the same frequencies, so equal guides give **bitwise** equal arrays and the
    comparison is exact.

    It is a probe of at most three frequencies rather than the whole band: sampling
    the gas and the cladding models is the expensive part of building a mode's terms
    (:class:`MarcatiliTerms`), and a cladding that agrees at the two ends and the
    middle of the band and differs between them is not a thing this package can
    build.

    Parameters
    ----------
    block : tuple of MarcatiliMode
        The contiguous ``HE_11`` to ``HE_1N`` set, in channel order.
    band : numpy.ndarray
        The active frequencies in rad/s, ascending, non-empty.
    probe : str or None, optional
        Which family's cladding term to compare the channels through. ``None``
        compares each channel's own, which is right when they share one family. A
        **vector** block does not: its families are the point of it, and their
        cladding terms differ by construction, so it compares every channel through
        one named kind, which is the same permittivity read the same way.

    Raises
    ------
    ModeError
        Naming the channel whose cladding or fill is not the lowest channel's.
    """
    if len(block) == 1:
        return
    frequencies = band[np.unique([0, band.size // 2, band.size - 1])]
    sampled = [
        (mode if probe is None else replace(mode, kind=probe, n=0, m=1)).sampled_terms(
            frequencies
        )
        for mode in block
    ]
    reference = sampled[0]
    for index, terms in enumerate(sampled[1:], start=1):
        for name, mine, theirs in (
            ("cladding", terms.cladding_term, reference.cladding_term),
            (
                "fill",
                terms.core_susceptibility,
                reference.core_susceptibility,
            ),
        ):
            if not np.array_equal(mine, theirs):
                msg = (
                    "a coherent wall block is built from one guide and reads it from "
                    f"the lowest channel alone; channel {index} samples a different "
                    f"{name} over the active band. Every channel of the block shares "
                    "one capillary and one fill"
                )
                raise ModeError(msg)


def _core_susceptibility(
    terms: MarcatiliTerms, density: float | Float[np.ndarray, " n_species"]
) -> Complex[np.ndarray, " n_omega"]:
    """Return the core susceptibility of a fill at one position.

    Parameters
    ----------
    terms : MarcatiliTerms
        The sampled frequency-dependent quantities.
    density : float or numpy.ndarray
        Number density in m^-3, scalar or one entry per sampled species.

    Returns
    -------
    numpy.ndarray
        ``chi(omega)``, complex, shape ``(n_omega,)``.
    """
    if terms.core_susceptibility.ndim == 1:
        return np.asarray(terms.core_susceptibility * density, dtype=complex)
    weights = np.atleast_1d(density)
    return np.asarray(
        np.tensordot(weights, terms.core_susceptibility, axes=(0, 0)), dtype=complex
    )


def wall_operator(
    modes: Sequence[Mode],
    omega: Float[np.ndarray, " n_omega"],
    *,
    z: float = 0.0,
    active: Bool[np.ndarray, " n_omega"] | None = None,
    scored: Bool[np.ndarray, " n_omega_scored"] | None = None,
    validity_limit: float = 0.25,
) -> WallOperator:
    r"""Build the condensed Robin wall generator of a contiguous ``HE_1m`` block.

    In outline (the manual page is ``docs/guide/waveguides/coherent-wall.md``), the
    orthonormal radial basis is :math:`\phi_m(x) = \sqrt2 J_0(u_m x)/|J_1(u_m)|` on
    :math:`x = r/a`, whose derivative at the wall is
    :math:`h_m = -\sqrt2\,\operatorname{sgn}J_1(u_m)\,u_m`. Truncating the
    Mittag-Leffler expansion of the exact outgoing root equation
    :math:`qJ_1(q) + i\eta J_0(q) = 0` at ``N`` orders and eliminating the boundary
    coordinate gives the condensed transverse operator
    :math:`L_N = -\operatorname{diag}(u_m^2) + hh^{T}/(2N - i\eta)`, and the paraxial
    reduction of :math:`M = k^2I + L_N/a^2` about the in-core carrier is
    :math:`G = i(M - k^2)/(2k) = D + H` in 1/m.

    Two normalisations are re-derived here rather than assumed. The **amplitude map**
    is the identity up to one mode-independent positive scale: this package's internal
    amplitude is ``a_m = E_m sqrt(N_m)`` with
    ``N_m = (pi/2) a^2 J_1(u_m)^2 eps_0 c`` (``docs/dev/conventions.md`` section 4),
    and both that normalisation and the orthonormal basis above carry the same
    ``|J_1(u_m)|``, so the two sets of coordinates differ by ``a sqrt(pi eps_0 c)``
    for every channel alike and the generator is unchanged by the map. The
    **impedance** is ``k_eff = k_0 / v_n``, the *vacuum* wavenumber over the cladding
    term this package's own capillary index already uses, because that is the
    combination whose ``N = 1`` limit is this package's Marcatili transverse
    eigenvalue; the carrier ``k`` of the paraxial reduction is the **in-core**
    wavenumber. Referring the impedance to the core instead is identical for a vacuum
    core and differs at the order of half the gas susceptibility for a filled one.

    Parameters
    ----------
    modes : sequence of Mode
        The contiguous block ``HE_11`` to ``HE_1N``, in the state's channel order,
        sharing one polarisation angle, one capillary and one constant section.
        Anything else is refused by name.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``; the whole grid.
    z : float, optional
        Position along the capillary in m the block is frozen at.
    active : numpy.ndarray or None, optional
        Bool mask of the bins to evaluate, shape ``(n_omega,)``. ``None`` takes every
        strictly positive frequency. An empty out-of-band bin is never evaluated: the
        wall formula divides by ``omega`` and its validity ratios are meaningless
        there, so the generator is left exactly zero.
    scored : numpy.ndarray or None, optional
        Bool mask of the bins the validity ratios are **taken** over, shape
        ``(n_omega,)``, intersected with the evaluated bins; a run passes the error
        norm's scored band (:attr:`solprop.grid.TimeGrid.inband_mask`), so the absorbed
        collar a field grid adds below the requested band, at up to twice the longest
        requested wavelength, neither refuses nor warns. The generator is still built on
        every evaluated bin. ``None``, or a mask that holds no evaluated bin, takes the
        ratios over every evaluated bin.
    validity_limit : float, optional
        Largest admitted validity ratio; above it the block is refused, above a tenth
        of it it warns. See :class:`CoherentWall`.

    Returns
    -------
    WallOperator
        The dense optics-convention generator, its factored pieces and its limits.

    Raises
    ------
    ModeError
        If the set is not one contiguous scalar block, if a channel declares a loss
        setting or the reduced model, if the active band is empty or holds a
        non-positive frequency, if the block is not passive on the branch the
        cladding term lands on, or if a validity ratio exceeds `validity_limit`.

    Warns
    -----
    GrazingIncidenceWarning
        When a validity ratio exceeds a tenth of `validity_limit`, naming it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, wall_operator
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=index) for index in (1, 2, 3)]
    >>> block = wall_operator(modes, np.array([2.356e15]))
    >>> block.n_modes, f"{block.limits.worst:.4f}"
    (3, '0.0090')
    """
    block = _wall_block_modes(modes)
    values = np.asarray(omega, dtype=float)
    if values.ndim != 1 or not values.size:
        raise ModeError("a coherent wall block needs a non-empty frequency vector")
    mask = (
        values > 0.0
        if active is None
        else np.asarray(active, dtype=bool) & (values > 0.0)
    )
    if not np.any(mask):
        msg = (
            "a coherent wall block has no active bin to evaluate: the mask leaves no "
            "strictly positive frequency, and a hollow capillary has no mode at zero"
        )
        raise ModeError(msg)
    count = len(block)
    first = block[0]
    band = values[mask]
    judged = _judged_bins(mask, scored)
    _check_wall_guide(block, band)
    terms = first.sampled_terms(band)
    radius, density = first._scalars(z)
    chi = _core_susceptibility(terms, density)
    vacuum = band / C
    wavenumber = np.sqrt(1.0 + chi) * vacuum
    impedance = vacuum / terms.cladding_term
    eta = radius * impedance
    zeros = np.array([mode.u_nm for mode in block], dtype=float)
    trace = -np.sqrt(2.0) * np.sign(jv(1, zeros)) * zeros
    # g(a) and D share the prefactor 1/(2 k a^2) of the paraxial reduction; the
    # condensed rank-one term carries the eliminated boundary denominator.
    prefactor = 1.0 / (2.0 * wavenumber * radius**2)
    coupling = 1j * prefactor / (2.0 * count - 1j * eta)
    diagonal = -1j * prefactor[None, :] * (zeros**2)[:, None]
    generator = coupling[None, None, :] * np.outer(trace, trace)[:, :, None]
    generator[np.arange(count), np.arange(count)] += diagonal
    limits = WallLimits(
        transverse_ratio=float(np.max(zeros[-1] / (abs(wavenumber[judged]) * radius))),
        vacuum_ratio=float(np.max(zeros[-1] / (vacuum[judged] * radius))),
        quasi_static=float(np.max(2.0 * count / abs(eta[judged]))),
        band=(float(band[judged][0]), float(band[judged][-1])),
        n_modes=count,
        radius=radius,
    )
    _check_wall_passivity(generator, wavenumber, radius, count)
    _check_wall_limits(limits, validity_limit)
    return WallOperator(
        omega=values,
        active=mask,
        generator=_scattered(generator, mask, (count, count)),
        trace_vector=trace,
        coupling=_scattered(coupling, mask, ()),
        wavenumber=_scattered(wavenumber, mask, ()),
        impedance=_scattered(impedance, mask, ()),
        limits=limits,
    )


def _vector_wall_families(
    block: tuple[MarcatiliMode, ...],
) -> tuple[int, tuple[WallFamily, ...]]:
    """Return the azimuthal order of a vector wall block and its radial families.

    The set must be **one** total azimuthal order, because the wall is rotationally
    symmetric and therefore exactly diagonal in that order, and each of its families
    must be the contiguous ``m = 1 .. N_f`` set the condensation truncates.
    Everything else is refused by name.

    Parameters
    ----------
    block : tuple of MarcatiliMode
        The channels in the state's order.

    Returns
    -------
    tuple
        The order ``nu`` and its families, in channel order.

    Raises
    ------
    ModeError
        Naming what makes the set more or less than one azimuthal order.
    """
    runs: list[list[MarcatiliMode]] = []
    for mode in block:
        if runs and (runs[-1][0].kind, runs[-1][0].n) == (mode.kind, mode.n):
            runs[-1].append(mode)
        else:
            runs.append([mode])
    labels = [f"{run[0].kind}_{run[0].n}m x {len(run)}" for run in runs]
    for run in runs:
        if [mode.m for mode in run] != list(range(1, len(run) + 1)):
            msg = (
                "every family of a vector coherent wall block is the contiguous "
                f"radial set m = 1 to N of its own azimuthal order, and this run "
                f"carries {', '.join(f'{mode.kind}_{mode.n}{mode.m}' for mode in run)}"
                ": the boundary self term counts the retained orders, so a missing "
                "one leaves its share of the trace nowhere to go"
            )
            raise ModeError(msg)
    if len(runs) > 2:
        msg = (
            "a vector coherent wall block is one total azimuthal order, which has at "
            f"most two radial families, and this run carries {len(runs)}: "
            f"{', '.join(labels)}. The wall is rotationally symmetric, so orders do "
            "not couple and each is a block of its own"
        )
        raise ModeError(msg)
    kinds = {run[0].kind for run in runs}
    if kinds <= {"TE", "TM"}:
        sector = 0
        families = tuple(
            WallFamily(
                label=f"{run[0].kind}_0m",
                azimuthal_order=0,
                bessel_order=1,
                n_channels=len(run),
                angular=(1.0, 0.0) if run[0].kind == "TM" else (0.0, 1.0),
                # Both transverse families are built from +J_1 (`MarcatiliMode.field`).
                radial_sign=1.0,
            )
            for run in runs
        )
        return sector, families
    if kinds != {"HE"}:
        msg = (
            "a vector coherent wall block is one total azimuthal order, and this run "
            f"mixes the transverse order nu = 0 with a hybrid one: {', '.join(labels)}"
            ". TE_0m and TM_0m are the nu = 0 block; the hybrids of order nu are its "
            "own block"
        )
        raise ModeError(msg)
    sector = abs(runs[0][0].n)
    if len(runs) == 2 and runs[1][0].n != -runs[0][0].n:
        msg = (
            f"the two hybrid families of the azimuthal order nu = {sector} are "
            f"n = {sector} (radial J_{abs(sector - 1)}) and n = {-sector} (radial "
            f"J_{sector + 1}), and this run carries {', '.join(labels)}: the wall "
            "couples those two and no others, so a block is one of them or both"
        )
        raise ModeError(msg)
    scale = 1.0 / math.sqrt(2.0)
    families = tuple(
        WallFamily(
            label=("HE" if run[0].n > 0 else "EH") + f"_{sector}m",
            azimuthal_order=run[0].n,
            bessel_order=abs(run[0].n - 1),
            n_channels=len(run),
            angular=(scale, -scale if run[0].n > 0 else scale),
            radial_sign=-1.0 if (run[0].n - 1) < 0 and (run[0].n - 1) % 2 else 1.0,
        )
        for run in runs
    )
    return sector, families


def _vector_wall_block(
    modes: Sequence[Mode],
) -> tuple[int, tuple[WallFamily, ...], tuple[MarcatiliMode, ...]]:
    """Check a mode set is one azimuthal order of a bare capillary, and split it.

    Parameters
    ----------
    modes : sequence of Mode
        The mode set of the run, in the state's channel order.

    Returns
    -------
    tuple
        The azimuthal order, its families and the same modes as a tuple.

    Raises
    ------
    ModeError
        Naming what makes the set incompatible.
    """
    block = tuple(modes)
    if not block:
        msg = "a vector coherent wall block needs at least one capillary mode"
        raise ModeError(msg)
    for index, mode in enumerate(block):
        if type(mode) is not MarcatiliMode:
            msg = (
                "a vector coherent wall block assumes one uniform dielectric wall at "
                f"a single radius, and channel {index} is a {type(mode).__name__}: a "
                "structured cladding's loss is its own resonances, not a Robin "
                "condition on a bare capillary. Leave wall=None and keep the guide's "
                "own loss"
            )
            raise ModeError(msg)
    checked = cast("tuple[MarcatiliMode, ...]", block)
    sector, families = _vector_wall_families(checked)
    _check_wall_channels(checked)
    return sector, families, checked


def _vector_impedances(
    mode: MarcatiliMode, band: Float[np.ndarray, " n_active"]
) -> Complex[np.ndarray, "n_wall_components n_active"]:
    """Return the wall's radial and tangential impedances in m.

    They are this package's **own** Marcatili cladding terms of the two transverse
    families over the vacuum wavenumber: ``v_TM/k_0`` for a field normal to the wall
    and ``v_TE/k_0`` for one tangential to it. Sampling them through a replacement
    mode rather than through :func:`_cladding_term_and_derivative` directly is
    deliberate: a constant, tabulated or mixed cladding then reaches the wall model by
    exactly the path any TE or TM channel of this guide would take.

    Parameters
    ----------
    mode : MarcatiliMode
        The channel the block reads its guide from.
    band : numpy.ndarray
        The active frequencies in rad/s, ascending, strictly positive.

    Returns
    -------
    numpy.ndarray
        The radial and tangential impedances in m, complex, shape ``(2, n_active)``.
    """
    vacuum = band / C
    return np.stack(
        [
            np.asarray(
                replace(mode, kind=kind, n=0, m=1).sampled_terms(band).cladding_term
            )
            / vacuum
            for kind in ("TM", "TE")
        ]
    )


def vector_wall_operator(
    modes: Sequence[Mode],
    omega: Float[np.ndarray, " n_omega"],
    *,
    z: float = 0.0,
    active: Bool[np.ndarray, " n_omega"] | None = None,
    scored: Bool[np.ndarray, " n_omega_scored"] | None = None,
    validity_limit: float = 0.25,
) -> VectorWallOperator:
    r"""Build the condensed **vector** Robin wall generator of one azimuthal order.

    In outline (the manual page is ``docs/guide/waveguides/vector-wall.md``), three
    steps.

    **The boundary response is a matrix, not a scalar.** At grazing incidence the wall
    imposes :math:`\partial_rE = ik_oE` on the field *tangential* to it and
    :math:`\partial_rE = i(k_o/n_r^2)E` on the field *normal* to it. In the
    near-Dirichlet regime the condensation is linear in the **impedance**, so what
    enters is :math:`Z = \operatorname{diag}(1/k_{\rm TM}, 1/k_{\rm TE})` in the
    :math:`(\hat r,\hat\theta)` components. Rotating it into the radial families of a
    total azimuthal order :math:`\nu` -- the angular quadrature
    :math:`Z_{fg} = \int_0^{2\pi}[Z_{rr}P_fP_g + Z_{\theta\theta}Q_fQ_g]\,d\theta`
    over the unit-normalised angular patterns of :attr:`WallFamily.angular` -- gives,
    for :math:`\nu \ge 1`,

    .. math::
       Z = \bar z\,I + \delta\,\sigma_x,\qquad
       \bar z = \tfrac12(1/k_{\rm TM} + 1/k_{\rm TE}),\qquad
       \delta = \tfrac12(1/k_{\rm TM} - 1/k_{\rm TE}),

    and, for :math:`\nu = 0`, the same matrix diagonalised: TM sees :math:`\bar z +
    \delta` and TE sees :math:`\bar z - \delta`, with no coupling at all. The mean
    :math:`\bar z` is exactly the scalar block's :math:`1/k_{\rm eff}`, because
    ``v_HE`` is the mean of ``v_TE`` and ``v_TM``; the **anisotropy** :math:`\delta` is
    what the scalar average discards, and it is ``(n_r^2 - 1)/(n_r^2 + 1)`` of the
    mean -- 35 % for silica, not a small correction.

    **The condensation is rank two.** Truncating the Mittag-Leffler expansion of the
    exact recurrence :math:`qJ_{p+1}(q) = (p - i\eta)J_p(q)` family by family gives
    :math:`T_p(\lambda) \simeq 2N_f + p_f + h_f^{T}(\lambda - D_f)^{-1}h_f`, and the
    exact two-family secular equation
    :math:`\det[\operatorname{diag}(T_{p_A}, T_{p_B}) - iaZ^{-1}] = 0` is **exactly**
    the secular equation of

    .. math::
       L_N = -\operatorname{diag}(u_m^2) + VSV^{T},\qquad
       S = (C - i\,aZ^{-1})^{-1},\qquad C = \operatorname{diag}(2N_f + p_f),

    with ``V`` one boundary trace column per family. The truncation constant carries
    the family's Bessel order ``p_f``: without it the condensed eigenvalues stop
    converging to the exact roots. The paraxial reduction of
    :math:`M = k^2I + L_N/a^2` is :math:`G = i(M - k^2)/(2k)` in 1/m, as for the
    scalar block.

    **What is new beyond two copies of the scalar block.** The scalar block is one
    family with one averaged impedance, and two copies of it for two polarisations are
    two copies of the same average. This block is per azimuthal order: it gives
    ``TE_0m`` and ``TM_0m`` their **own** impedances, which differ by ``n_r^2``, and it
    **couples** the ``J_{nu-1}`` and ``J_{nu+1}`` families of every order ``nu >= 1``
    through :math:`\delta`, which converts power between them. Each family in
    isolation still reduces to this package's own Marcatili loss for its own ``v_n``.

    Parameters
    ----------
    modes : sequence of Mode
        One total azimuthal order of one capillary, in the state's channel order:
        either ``TM_0m`` and ``TE_0m`` sets, or the hybrid ``n = +nu`` set followed by
        the ``n = -nu`` one, each contiguous from ``m = 1``. Either family may be
        retained alone, in which case the block carries no conversion and the report's
        single-family loss limits are what it is. Every channel declares the same
        guide, fill and polarisation angle; anything else is refused by name.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``; the whole grid.
    z : float, optional
        Position along the capillary in m the block is frozen at.
    active : numpy.ndarray or None, optional
        Bool mask of the bins to evaluate, shape ``(n_omega,)``. ``None`` takes every
        strictly positive frequency; an out-of-band bin is never evaluated.
    scored : numpy.ndarray or None, optional
        Bool mask of the bins the validity ratios are taken over, as for
        :func:`wall_operator`; ``None`` takes every evaluated bin.
    validity_limit : float, optional
        Largest admitted validity ratio; above it the block is refused, above a tenth
        of it it warns. See :class:`CoherentWall`.

    Returns
    -------
    VectorWallOperator
        The dense optics-convention generator, its factored pieces and its limits.

    Raises
    ------
    ModeError
        If the set is not one azimuthal order of a bare capillary, if a channel
        declares a loss setting or the reduced model, if the active band is empty, if
        the block is not passive on the branch the cladding terms land on, or if a
        validity ratio exceeds `validity_limit`.

    Warns
    -----
    GrazingIncidenceWarning
        When a validity ratio exceeds a tenth of `validity_limit`, naming it.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, vector_wall_operator
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, n=n, m=1) for n in (1, -1)]
    >>> block = vector_wall_operator(modes, np.array([2.356e15]))
    >>> block.sector, [family.label for family in block.families]
    (1, ['HE_1m', 'EH_1m'])
    >>> bool(abs(block.generator[0, 1, 0]) > 0.0)
    True
    """
    sector, families, block = _vector_wall_block(modes)
    values = np.asarray(omega, dtype=float)
    if values.ndim != 1 or not values.size:
        msg = "a vector coherent wall block needs a non-empty frequency vector"
        raise ModeError(msg)
    mask = (
        values > 0.0
        if active is None
        else np.asarray(active, dtype=bool) & (values > 0.0)
    )
    if not np.any(mask):
        msg = (
            "a vector coherent wall block has no active bin to evaluate: the mask "
            "leaves no strictly positive frequency, and a hollow capillary has no "
            "mode at zero"
        )
        raise ModeError(msg)
    first = block[0]
    band = values[mask]
    judged = _judged_bins(mask, scored)
    # The families' own cladding terms differ by construction, so the channels are
    # compared through one of them: what must agree is the permittivity, not v_n.
    _check_wall_guide(block, band, probe="TE")
    terms = first.sampled_terms(band)
    radius, density = first._scalars(z)
    chi = _core_susceptibility(terms, density)
    vacuum = band / C
    wavenumber = np.sqrt(1.0 + chi) * vacuum
    impedance = _vector_impedances(first, band)
    angular = np.array([family.angular for family in families], dtype=float)
    # Z_fg(omega) = sum_i A_fi A_gi z_i(omega): the wall's two impedances rotated into
    # the families' angular content, which is the whole vector boundary response.
    gram = np.einsum("fi,iw,gi->fgw", angular, impedance, angular)
    truncation = np.diag([float(family.truncation) for family in families])
    admittance = radius * np.linalg.inv(np.moveaxis(gram, -1, 0))
    condensed = np.linalg.inv(truncation[None] - 1j * admittance)
    zeros = np.array([mode.u_nm for mode in block], dtype=float)
    orders = np.repeat(
        [family.bessel_order for family in families],
        [family.n_channels for family in families],
    )
    traces = np.zeros((len(block), len(families)))
    columns = np.repeat(np.arange(len(families)), [f.n_channels for f in families])
    # The derivative at the wall of each channel's **implemented** orthonormal radial
    # function, which is `jv(n - 1, .)` and so carries the family's own sign.
    signs = np.repeat(
        [family.radial_sign for family in families],
        [family.n_channels for family in families],
    )
    traces[np.arange(len(block)), columns] = (
        -np.sqrt(2.0) * signs * np.sign(jv(orders + 1, zeros)) * zeros
    )
    prefactor = 1.0 / (2.0 * wavenumber * radius**2)
    weight = 1j * prefactor[None, None, :] * np.moveaxis(condensed, 0, -1)
    generator = np.einsum("mf,fgw,ng->mnw", traces, weight, traces)
    generator[np.arange(len(block)), np.arange(len(block))] += (
        -1j * prefactor[None, :] * (zeros**2)[:, None]
    )
    rows = np.max(np.sum(abs(gram[..., judged]), axis=1), axis=-1) / radius
    # The **highest retained** order over every family, which is not the last channel:
    # a block may retain more radial orders of the J_{nu-1} family than of the
    # J_{nu+1} one, and then the largest zero is not the one the channel order ends on.
    highest = float(np.max(zeros))
    limits = WallLimits(
        transverse_ratio=float(np.max(highest / (abs(wavenumber[judged]) * radius))),
        vacuum_ratio=float(np.max(highest / (vacuum[judged] * radius))),
        quasi_static=float(
            np.max(
                [
                    family.truncation * row
                    for family, row in zip(families, rows, strict=True)
                ]
            )
        ),
        band=(float(band[judged][0]), float(band[judged][-1])),
        n_modes=len(block),
        radius=radius,
    )
    _check_wall_passivity(generator, wavenumber, radius, len(block))
    _check_wall_limits(limits, validity_limit)
    return VectorWallOperator(
        omega=values,
        active=mask,
        generator=_scattered(generator, mask, (len(block), len(block))),
        families=families,
        sector=sector,
        family_traces=traces,
        weight=_scattered(weight, mask, (len(families), len(families))),
        wavenumber=_scattered(wavenumber, mask, ()),
        impedance=_scattered(impedance, mask, (2,)),
        limits=limits,
    )


def wall_connection(
    modes: Sequence[Mode], z: float = 0.0, *, impedance: WallImpedance = "mean"
) -> Float[np.ndarray, "n_modes_link_out n_modes_link_in"]:
    r"""Return the moving-basis connection of a wall block at one position.

    The basis of the condensed wall block is attached to the core radius, so a taper
    moves it. Writing the field as :math:`E = \sum_n c_n \psi_n(r; a(z))` with
    :math:`\psi_n = \sqrt2 J_p(u_n r/a)/(a|J_{p+1}(u_n)|)` orthonormal under
    :math:`\int_0^a \cdot\, r\,dr` at every ``z``, :math:`u_n` the zeros of
    :math:`J_p`, the coefficients pick up :math:`c' = (iB - C)c` with
    :math:`C_{mn} = \langle\psi_m|\partial_z\psi_n\rangle`:

    .. math::
       C_{mn} = \frac{a'}{a}\,\frac{2s_ms_nu_mu_n}{u_m^2-u_n^2}\quad(m\ne n),
       \qquad C_{nn} = 0,

    with :math:`s_m = \operatorname{sgn}J_{p+1}(u_m)` the sign of the basis slope at
    the wall, :math:`-J_p'(u_m) = J_{p+1}(u_m)`. For the ``HE_1m`` block ``p = 0``
    and this is the scalar block's formula. It is **real** -- so the optics/internal
    conjugation of ``docs/dev/conventions.md`` section 2 leaves it alone -- and
    **antisymmetric**, because swapping the indices flips the denominator alone, so
    the connection on its own is norm preserving and moves no power out of the
    retained set. The sign is checked by differentiating the implemented profile
    (``tests/modes/test_wall_operator.py``), not asserted.

    **The vector block** (``impedance="te_tm"``) is one
    total azimuthal order with at most two radial families, and its connection is
    **block diagonal**: each family carries the formula above with its **own** Bessel
    order ``p_f`` (:attr:`WallFamily.bessel_order`), zeros and signs, and the two
    families do not connect at all. The reason is exact rather than small: a taper
    moves the radius and nothing else, and every channel's field is its radial
    function of ``r/a`` times an angular pattern that does not depend on ``a``, so
    :math:`\langle\psi^A_m|\partial_z\psi^B_n\rangle` is a radial integral times
    the angular overlap of the two families' patterns, which is zero -- the two
    hybrid families of an order ``nu >= 1`` overlap as :math:`\cos 2\nu\theta`, and
    ``TE_0m`` and ``TM_0m`` are orthogonal at every point. A family's implemented
    sign (:attr:`WallFamily.radial_sign`) enters :math:`s_m s_n` squared and cancels.
    Both statements are measured on the implemented fields, not assumed.

    The whole matrix scales with ``a'(z)/a(z)``, and nothing in it depends on
    frequency: the basis is the same set of Bessel orders at every bin.

    Parameters
    ----------
    modes : sequence of Mode
        The block in the state's channel order: the contiguous ``HE_11`` to ``HE_1N``
        set for ``impedance="mean"``, one azimuthal order of a capillary for
        ``"te_tm"``. Refused by name on exactly the conditions :func:`wall_operator`
        and :func:`vector_wall_operator` refuse.
    z : float, optional
        Position along the capillary in m.
    impedance : str, optional
        Which block the connection is of, ``"mean"`` or ``"te_tm"``, as
        :attr:`CoherentWall.impedance`.

    Returns
    -------
    numpy.ndarray
        ``C`` in 1/m, real, shape ``(n_modes, n_modes)``, with a zero diagonal.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.capillary import MarcatiliMode, wall_connection
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 62.5e-6, 1e-2)
    >>> modes = [MarcatiliMode(taper, "He", 1.0, m=index) for index in (1, 2)]
    >>> link = wall_connection(modes, 5e-3)
    >>> bool(np.all(link == -link.T)), float(link[0, 0])
    (True, 0.0)
    >>> vector = [MarcatiliMode(taper, "He", 1.0, n=n, m=1) for n in (1, -1)]
    >>> float(wall_connection(vector, 5e-3, impedance="te_tm")[0, 1])
    0.0
    """
    if impedance == "mean":
        return _scalar_connection(_wall_block_modes(modes), z)
    _, grouped, block = _vector_wall_block(modes)
    families = tuple((family.bessel_order, family.n_channels) for family in grouped)
    first = block[0]
    radius = float(first.core_radius(np.array([float(z)]))[0])
    slope = float(first.core_radius.derivative(np.array([float(z)]))[0])
    link = np.zeros((len(block), len(block)), dtype=float)
    start = 0
    for order, count in families:
        stop = start + count
        zeros = np.array([mode.u_nm for mode in block[start:stop]], dtype=float)
        link[start:stop, start:stop] = _bessel_connection(zeros, order, slope / radius)
        start = stop
    return link


def taper_coupling_block(modes: Sequence[Mode]) -> tuple[MarcatiliMode, ...] | None:
    """Return the mode set the taper coupling is carried on, or ``None``.

    The taper coupling -- the moving-basis connection ``<psi_m|d_z psi_n>`` of
    :func:`taper_connection` in a coupled flow, with each channel's own propagation
    constant on the flow's diagonal -- is carried by default on a **multimode capillary
    whose radius varies along z**, lossless or not. That is a set of two or more plain
    :class:`MarcatiliMode` channels forming the contiguous ``HE_11`` to ``HE_1N`` block,
    whose core radius is not constant.
    Every other set is ``None``: one channel needs no connection; a constant bore --
    a pressure gradient included, which changes the effective index and not the
    ``J_0(u_m r/a)`` field -- has none; and a structured cladding (a Vincetti mode), a
    tabulated mode or a set of several azimuthal orders is not the block the
    connection is derived for, and keeps the adiabatic route with its
    :class:`~solprop.exceptions.AdiabaticTaperWarning`.

    Parameters
    ----------
    modes : sequence of Mode
        The run's mode set, in the state's channel order.

    Returns
    -------
    tuple of MarcatiliMode or None
        The block, or ``None`` when the coupling does not apply.

    Examples
    --------
    >>> from solprop.modes.capillary import MarcatiliMode, taper_coupling_block
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 62.5e-6, 1e-2)
    >>> modes = [MarcatiliMode(taper, "He", 1.0, m=m) for m in (1, 2)]
    >>> len(taper_coupling_block(modes))
    2
    >>> taper_coupling_block([MarcatiliMode(125e-6, "He", 1.0, m=m) for m in (1, 2)])
    >>> taper_coupling_block([MarcatiliMode(taper, "He", 1.0, m=m) for m in (1, 3)])
    """
    block = tuple(modes)
    if len(block) < 2 or any(type(mode) is not MarcatiliMode for mode in block):
        return None
    channels = cast("tuple[MarcatiliMode, ...]", block)
    labels = [(mode.kind, mode.n, mode.m) for mode in channels]
    if labels != [("HE", 1, index + 1) for index in range(len(block))]:
        return None
    if channels[0].radius_is_constant:
        return None
    return channels


def taper_connection(
    modes: Sequence[Mode], z: float = 0.0
) -> Float[np.ndarray, "n_modes_taper_out n_modes_taper_in"]:
    r"""Return the taper connection of an ``HE_1m`` block at one position.

    The connection :math:`C_{mn} = \langle\psi_m|\partial_z\psi_n\rangle` of the
    field-normalised ``J_0(u_m r/a)`` basis, exactly :func:`wall_connection`'s scalar
    block (``impedance="mean"``) -- real, antisymmetric, proportional to ``a'/a``
    -- without its two conditions that belong to the wall: it admits any loss and
    either capillary model, because the basis is the Dirichlet one whatever the
    modes' effective index says. It is what the **taper coupling** carries on a
    multimode capillary without a wall.

    Parameters
    ----------
    modes : sequence of Mode
        The contiguous ``HE_11`` to ``HE_1N`` set in state order, every channel
        declaring the same radius and density profiles, each with an analytic
        derivative.
    z : float, optional
        Position along the capillary in m.

    Returns
    -------
    numpy.ndarray
        ``C`` in 1/m, real, shape ``(n_modes, n_modes)``, with a zero diagonal.

    Raises
    ------
    ModeError
        If the set is not such a block.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.capillary import MarcatiliMode, taper_connection
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 62.5e-6, 1e-2)
    >>> modes = [MarcatiliMode(taper, "He", 1.0, m=m, loss=False) for m in (1, 2)]
    >>> link = taper_connection(modes, 5e-3)
    >>> bool(np.all(link == -link.T)), float(link[0, 0])
    (True, 0.0)
    """
    return _scalar_connection(_wall_block_modes(modes, wall=False), z)


def _scalar_connection(
    block: tuple[MarcatiliMode, ...], z: float
) -> Float[np.ndarray, "n_modes_scalar_out n_modes_scalar_in"]:
    """Return the connection of a checked ``HE_1m`` block at one position.

    Parameters
    ----------
    block : tuple of MarcatiliMode
        The checked block.
    z : float
        Position in m.

    Returns
    -------
    numpy.ndarray
        ``C`` in 1/m.
    """
    first = block[0]
    radius = float(first.core_radius(np.array([float(z)]))[0])
    slope = float(first.core_radius.derivative(np.array([float(z)]))[0])
    zeros = np.array([mode.u_nm for mode in block], dtype=float)
    return _bessel_connection(zeros, 0, slope / radius)


def _bessel_connection(
    zeros: Float[np.ndarray, " n_family"], order: int, rate: float
) -> Float[np.ndarray, "n_family_out n_family_in"]:
    """Return the connection of one radial family of Bessel order `order`.

    Parameters
    ----------
    zeros : numpy.ndarray
        The family's retained zeros ``u_m`` of ``J_order``, ascending.
    order : int
        The family's Bessel order ``p``, ``>= 0``.
    rate : float
        ``a'/a`` in 1/m.

    Returns
    -------
    numpy.ndarray
        ``C`` of the family in 1/m, real, antisymmetric, with a zero diagonal.
    """
    count = zeros.size
    signs = np.sign(jv(order + 1, zeros))
    squared = zeros**2
    # The diagonal of the difference is zero, so the reciprocal is taken where the
    # denominator is not: C_nn is zero by the derivation, not by a limit.
    difference = squared[:, None] - squared[None, :]
    off = ~np.eye(count, dtype=bool)
    link = np.zeros((count, count), dtype=float)
    weight = 2.0 * np.outer(signs * zeros, signs * zeros)
    link[off] = rate * weight[off] / difference[off]
    return link


def worst_wall_limits(  # noqa: PLR0913 - the block's own arguments and the scan
    modes: Sequence[Mode],
    omega: Float[np.ndarray, " n_omega"],
    *,
    positions: Sequence[float],
    active: Bool[np.ndarray, " n_omega"] | None = None,
    scored: Bool[np.ndarray, " n_omega_scored"] | None = None,
    validity_limit: float = 0.25,
    impedance: WallImpedance = "mean",
) -> WallLimits:
    """Return the validity ratios at the position where they are worst.

    A guide that varies along z has a different block at every section, and the three
    ratios of :class:`WallLimits` are not monotone in the declared parameters in
    general -- the radius, the fill's index and the cladding term all enter them. The
    scan is therefore over the positions the caller actually builds the flow from, and
    the **one** call that carries the caller's `validity_limit` is the one at the worst
    of them, so a run is refused or warned about once and names the geometry that did
    it rather than the entrance's.

    Parameters
    ----------
    modes : sequence of Mode
        The contiguous block ``HE_11`` to ``HE_1N``, in the state's channel order.
    omega : numpy.ndarray
        Angular frequencies in rad/s, shape ``(n_omega,)``; the whole grid.
    positions : sequence of float
        Positions in m to scan, non-empty.
    active : numpy.ndarray or None, optional
        Bool mask of the bins to evaluate, shape ``(n_omega,)``.
    scored : numpy.ndarray or None, optional
        Bool mask of the bins the ratios are taken over, shape ``(n_omega,)``
        (:func:`wall_operator`); ``None`` takes every evaluated bin.
    validity_limit : float, optional
        Largest admitted validity ratio, applied at the worst position alone.
    impedance : str, optional
        Which block to build, ``"mean"`` (:func:`wall_operator`) or ``"te_tm"``
        (:func:`vector_wall_operator`). The scan is the same either way.

    Returns
    -------
    WallLimits
        The ratios at that position.

    Raises
    ------
    ModeError
        If `positions` is empty, or on everything the block's own constructor refuses.

    Warns
    -----
    GrazingIncidenceWarning
        Once, from the worst position, when a ratio exceeds a tenth of the limit.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.capillary import MarcatiliMode, worst_wall_limits
    >>> from solprop.profiles import Linear
    >>> taper = Linear(125e-6, 100e-6, 1e-2)
    >>> modes = [MarcatiliMode(taper, "He", 1.0, m=index) for index in (1, 2)]
    >>> limits = worst_wall_limits(
    ...     modes, np.array([2.356e15]), positions=[0.0, 5e-3, 1e-2]
    ... )
    >>> f"{limits.radius * 1e6:.1f} um"
    '100.0 um'
    """
    scan = [float(value) for value in positions]
    if not scan:
        msg = "a coherent wall block needs at least one position to scan its limits"
        raise ModeError(msg)
    build = wall_operator if impedance == "mean" else vector_wall_operator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GrazingIncidenceWarning)
        ratios = [
            build(
                modes,
                omega,
                z=z,
                active=active,
                scored=scored,
                validity_limit=math.inf,
            ).limits.worst
            for z in scan
        ]
    return build(
        modes,
        omega,
        z=scan[int(np.argmax(ratios))],
        active=active,
        scored=scored,
        validity_limit=validity_limit,
    ).limits


def _judged_bins(
    mask: Bool[np.ndarray, " n_omega"],
    scored: Bool[np.ndarray, " n_omega_scored"] | None,
) -> Bool[np.ndarray, " n_band"]:
    """Return which evaluated bins a wall block's validity ratios are taken over.

    Parameters
    ----------
    mask : numpy.ndarray
        The evaluated bins, bool, shape ``(n_omega,)``, at least one true.
    scored : numpy.ndarray or None
        The scored bins, bool, shape ``(n_omega,)``; ``None`` takes every evaluated
        bin.

    Returns
    -------
    numpy.ndarray
        Bool over the evaluated bins alone, at least one true. A scored mask that
        holds **no** evaluated bin -- requested limits narrower than one bin, which
        a monochromatic study grid can declare -- takes every evaluated bin, the
        conservative choice: it never admits a block the whole band would refuse.

    Raises
    ------
    ModeError
        If `scored` does not match the frequency axis.
    """
    if scored is None:
        return np.ones(int(np.count_nonzero(mask)), dtype=bool)
    chosen = np.asarray(scored, dtype=bool)
    if chosen.shape != mask.shape:
        msg = (
            f"a coherent wall block's scored mask has shape {chosen.shape}, not the "
            f"frequency axis's {mask.shape}"
        )
        raise ModeError(msg)
    judged = chosen[mask]
    return judged if np.any(judged) else np.ones(judged.shape, dtype=bool)


def _scattered(
    values: np.ndarray,
    mask: Bool[np.ndarray, " n_omega"],
    leading: tuple[int, ...],
) -> np.ndarray:
    """Return active-band values placed back on the whole frequency axis.

    Parameters
    ----------
    values : numpy.ndarray
        The values on the active bins, with the frequency axis last.
    mask : numpy.ndarray
        Which bins those are, bool, shape ``(n_omega,)``.
    leading : tuple of int
        The shape in front of the frequency axis.

    Returns
    -------
    numpy.ndarray
        Complex, shape ``(*leading, n_omega)``, exactly zero off the mask.
    """
    out = np.zeros((*leading, len(mask)), dtype=complex)
    out[..., mask] = values
    return out


def _check_wall_passivity(
    generator: Complex[np.ndarray, "n_modes n_modes n_omega"],
    wavenumber: Complex[np.ndarray, " n_omega"],
    radius: float,
    count: int,
) -> None:
    """Refuse a wall block whose Hermitian part is not negative semidefinite.

    The Hermitian part of the paraxial generator is the rank-one
    ``-Re(eta) h h^T / (|k| a^2 (4N^2 + |eta|^2))`` for the scalar block and rank two
    for the vector one (one rank per family: ``(S - S^H)/2i = S Re(H) S^H``), so it is
    negative exactly when the impedance lands on the passive branch. A cladding whose
    permittivity puts the square root elsewhere would give a wall that supplies power,
    which :class:`MarcatiliMode` clamps to zero for a diagonal loss; a clamp is
    not available here, because zeroing one entry of a rank-one operator is not a
    boundary condition. The block is refused instead.

    **No cladding model this package ships can reach the refusal**, and the proof is
    worth writing down rather than leaving as an untested branch. With
    ``s = sqrt(e - 1)`` on the principal root, ``Re s = x >= 0`` and
    ``1/v_HE = 2s/(s^2 + 2)``, whose real part is
    ``2x(|s|^2 + 2)/|s^2 + 2|^2 >= 0``; equality holds only for a real ``e < 1``,
    where the wall is exactly lossless rather than active. The guard is therefore a
    defence against a **future** impedance that does not take that root (the vector
    block's two impedances take it, and the guard is shared with it) and
    against a sign slipped into this arithmetic, not against a cladding a caller can
    pass today.

    Parameters
    ----------
    generator : numpy.ndarray
        The optics generator on the active band, shape ``(n_modes, n_modes, n_omega)``.
    wavenumber : numpy.ndarray
        The in-core wavenumber on the active band, 1/m.
    radius : float
        Core radius in m.
    count : int
        Channel count, used in the refusal.

    Raises
    ------
    ModeError
        If any Hermitian eigenvalue is positive beyond the rounding of the block.
    """
    moved = np.moveaxis(generator, (0, 1), (-2, -1))
    hermitian = 0.5 * (moved + np.conj(np.swapaxes(moved, -1, -2)))
    largest = float(np.max(np.linalg.eigvalsh(hermitian)))
    scale = float(np.max(abs(generator)))
    budget = 64.0 * count * float(np.finfo(float).eps) * scale
    if largest > budget:
        rate = float(np.max(1.0 / (abs(wavenumber) * radius**2)))
        msg = (
            "the coherent wall block is not passive on the branch this cladding term "
            "lands on: the Hermitian part of the generator has eigenvalue "
            f"{largest:.4g} 1/m, above the {budget:.4g} 1/m rounding of a block "
            f"whose own scale is {rate:.4g} 1/m. A cladding outside Marcatili's "
            "regime has no outgoing-wall impedance to condense"
        )
        raise ModeError(msg)


def _check_wall_limits(limits: WallLimits, validity_limit: float) -> None:
    """Refuse or warn on the validity ratios of a wall block, by name.

    Parameters
    ----------
    limits : WallLimits
        The measured ratios.
    validity_limit : float
        Largest admitted ratio.

    Raises
    ------
    ModeError
        If the worst ratio exceeds `validity_limit`.

    Warns
    -----
    GrazingIncidenceWarning
        If it exceeds a tenth of `validity_limit`.
    """
    name, value = limits.named()
    if value > validity_limit:
        msg = (
            f"the coherent wall block's {name} is {value:.4g} over "
            f"{limits.band[0]:.4g} to {limits.band[1]:.4g} rad/s, above the "
            f"{validity_limit:.4g} this block admits: the condensed paraxial model is "
            "not valid there. Narrow the band, widen the core, or retain fewer "
            "channels"
        )
        raise ModeError(msg)
    if value > 0.1 * validity_limit:
        warnings.warn(
            f"the coherent wall block's {name} reaches {value:.4g}, within a factor "
            f"ten of the {validity_limit:.4g} limit; the condensed paraxial model is "
            "marginal over this band",
            GrazingIncidenceWarning,
            stacklevel=3,
        )


def wall_square_root_metric(
    generator: Complex[np.ndarray, "n_modes_sqrt_out n_modes_sqrt_in"],
    wavenumber: float,
) -> tuple[
    Complex[np.ndarray, "n_modes_b_out n_modes_b_in"],
    Complex[np.ndarray, "n_modes_w_out n_modes_w_in"],
]:
    r"""Return the square-root generator and its candidate flux metric, as a diagnostic.

    For a **constant, real, positive** carrier ``k``, the square-root model
    ``c' = iBc`` with ``B^2 = M = k^2 I - 2ikG`` satisfies

    .. math::
       \frac{d}{dz}\left(c^\dagger W c\right)
       = -c^\dagger\frac{M - M^\dagger}{2ik}c,\qquad
       W = \frac{B + B^\dagger}{2k},

    which is the flux balance of this scalar model. **Its domain is exactly that
    statement.** Production semantics are the paraxial generator, not this square root;
    ``W`` is positive semidefinite only when the Hermitian part of ``B`` is, which
    positive eigenvalues of ``B`` do not establish, and which the principal branch does
    not establish either; a complex ``k``, an evanescent sector, a varying cross-section
    -- which adds ``W'`` -- and a varying ``k`` all fall outside it. Use it to measure
    the identity, not to define an error norm or to transport noise.

    Parameters
    ----------
    generator : numpy.ndarray
        The optics-convention paraxial generator at one frequency in 1/m, complex,
        shape ``(n_modes, n_modes)``.
    wavenumber : float
        The carrier ``k`` in 1/m, real and positive.

    Returns
    -------
    tuple of numpy.ndarray
        ``B`` in 1/m on the principal branch, and the dimensionless ``W``.

    Raises
    ------
    ModeError
        If the carrier is not real, finite and positive.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes import MarcatiliMode, wall_operator
    >>> from solprop.modes.capillary import wall_square_root_metric
    >>> modes = [MarcatiliMode(125e-6, "He", 1.0, m=index) for index in (1, 2)]
    >>> block = wall_operator(modes, np.array([2.356e15]))
    >>> k = float(block.wavenumber[0].real)
    >>> root, metric = wall_square_root_metric(block.generator[:, :, 0], k)
    >>> bool(np.max(abs(root @ root - (k**2 * np.eye(2)
    ...      - 2j * k * block.generator[:, :, 0]))) < 1e-6 * k**2)
    True
    """
    carrier = float(wavenumber)
    if not math.isfinite(carrier) or carrier <= 0.0:
        msg = (
            "the square-root flux metric is derived for a constant real positive "
            f"carrier, got k = {wavenumber!r}"
        )
        raise ModeError(msg)
    block = np.asarray(generator, dtype=complex)
    identity = np.eye(block.shape[0], dtype=complex)
    squared = carrier**2 * identity - 2j * carrier * block
    root = np.asarray(sqrtm(squared), dtype=complex)
    return root, (root + np.conj(root.T)) / (2.0 * carrier)
