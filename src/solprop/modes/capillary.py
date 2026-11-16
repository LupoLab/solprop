# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Marcatili modes of a gas-filled hollow capillary, with analytic derivatives.

:class:`MarcatiliMode` is Marcatili and Schmeltzer's asymptotic mode of a hollow
dielectric tube much wider than the wavelength (*Bell Syst. Tech. J.* **43**,
1783-1809, 1964), in the ``"full"`` and ``"reduced"`` models (:data:`MarcatiliModel`)
and the HE, TE and TM families (:data:`ModeKind`). It forms ``n_eff - 1`` directly
rather than through unity, and differentiates both models in closed form in frequency
and along z. Position reaches the index only through the core radius ``a(z)`` and the
number density ``rho(z)``, so everything frequency-dependent is a
:class:`MarcatiliTerms`, sampled once (``sampled_terms``) and evaluated per position
(``neff_from_terms``); :func:`shared_sampling` lets a translation reading the same
modes at many z share that sampling. :func:`bessel_zero`,
:func:`cladding_term_and_derivative` and :func:`core_susceptibility_at` are the
model's ingredients.

A positive imaginary part of ``n_eff`` is loss (``docs/dev/conventions.md`` section
5), so the reduced model's loss term enters with a **plus** sign; one arithmetic kernel
serves every path. The model, the rationalised refractivity, the derivatives and the
factorisation are in ``docs/guide/waveguides/capillary-model.md``; the coherent wall
operator that couples these modes is :mod:`solprop.modes.wall`.
"""

import logging
import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import KW_ONLY, dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar, Literal, cast

import numpy as np
from jaxtyping import Complex, Float, Inexact
from scipy.integrate import quad
from scipy.special import jn_zeros, jv

from solprop._display import engineering
from solprop.constants import EPS0, C
from solprop.exceptions import ModeError
from solprop.modes.fill import (
    EVANESCENT_INDEX_FLOOR,
    GasFillMixin,
    LossFunction,
    _sqrt1p_minus_one,
)
from solprop.modes.protocol import (
    ChannelLabel,
    CrossSection,
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
)

__all__ = [
    "EVANESCENT_INDEX_FLOOR",
    "MarcatiliMode",
    "MarcatiliModel",
    "MarcatiliTerms",
    "ModeKind",
    "bessel_zero",
    "cladding_term_and_derivative",
    "core_susceptibility_at",
    "shared_sampling",
]

logger = logging.getLogger(__name__)

ModeKind = Literal["HE", "TE", "TM"]
"""Which family a capillary mode belongs to: hybrid, transverse electric or magnetic."""

MarcatiliModel = Literal["full", "reduced"]
"""Whether to use the square-root expression or its leading-order expansion."""


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


def cladding_term_and_derivative(
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
        The cladding's relative permittivity, complex, shape (n_omega,).
    dpermittivity : numpy.ndarray
        Its analytic frequency derivative in s/rad, same shape.
    kind : {"HE", "TE", "TM"}
        The mode family.

    Returns
    -------
    tuple of numpy.ndarray
        ``v_n(omega)``, dimensionless, and ``dv_n/domega`` in s/rad.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.capillary import cladding_term_and_derivative
    >>> term, _ = cladding_term_and_derivative(
    ...     np.array([5.0 + 0j]), np.array([0j]), "TE"
    ... )
    >>> term.tolist()
    [(0.5+0j)]
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

    Like every :class:`~solprop.modes.protocol.Mode`, it takes strictly positive
    frequencies: a hollow capillary has no mode at zero frequency -- the waveguide
    term ``(u c / (omega a))^2`` diverges -- so a zero raises. The linear operator
    evaluates modes on the active band, which excludes the DC bin.

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
    alpha : float or callable or None
        An **absolute** power attenuation coefficient in 1/m that replaces the
        model's loss: a finite non-negative number, or a callable ``alpha(omega)``
        taking and returning arrays of shape ``(n_omega,)`` -- measured fibre loss,
        say. Only ``Im n_eff = c alpha / (2 omega)`` is replaced; the real part, and
        with it ``beta`` and ``beta_1``, stays the model's. A callable that carries a
        ``derivative`` attribute gets an analytic ``d(Im n_eff)/d(omega)``; otherwise
        that one term is differenced, and the constructor says so. ``alpha`` is the
        caller's and is not clamped. ``None`` (the default) keeps the model's loss.
    loss_scale : float
        Multiplies the model's own loss: ``1.0`` (the default) is the model, ``0.0``
        lossless, anything else a fitting knob on the model. Finite and
        non-negative; refused beside an ``alpha``, which replaces the loss it would
        scale. The model's own loss is never negative: where the square root lands
        on the branch that reads as gain -- a cladding index below unity, outside
        Marcatili's regime -- it is clamped at zero and the mode logs the fact once.
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
    alpha: float | LossFunction | None = None
    loss_scale: float = 1.0
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
    _INDEX_IS_MARCATILI: ClassVar[bool] = True
    """Whether the index is the Marcatili one that :meth:`dneff_dz` differentiates.

    A subclass that computes its index another way -- a structured cladding, a
    tabulated index -- sets it ``False``.
    """
    # The mode a frozen copy was taken from, when the copy's frequency-only terms are
    # the origin's (:meth:`at_parameters`); shared by :meth:`sampled_terms` in a scope.
    _terms_origin: ClassVar["MarcatiliMode | None"] = None

    def _check_arguments(self) -> None:
        """Check the constructor arguments against each other."""
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
            positive, the model is unknown, ``alpha`` or ``loss_scale`` is invalid or
            both are given, or the radius or temperature is not positive.
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
            For example ``MarcatiliMode(HE11, a=125.0 um, He, model=full, loss=model)``.
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
            f"{fill}, model={self.model}, loss={self._loss_text()})"
        )

    def _area_rows(self) -> list[tuple[str, object]]:
        """Return the effective-area row of the printed block.

        Marcatili's area is a constant times the radius squared, so it costs nothing;
        a subclass whose area is frequency dependent or tabulated returns no row.

        - Returns: ``(key, value)`` rows.
        """
        return [("effective area", f"{self.aeff() * 1e12:.4g} um^2")]

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
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
    def z_constant(self) -> bool:  # numpydoc ignore=RT01
        """Whether nothing about the mode varies along z.

        ``True`` when neither the core radius nor the density varies. The
        multimode transform cares only about the radius, which
        :attr:`radius_is_constant` reports separately.
        """
        return all(path.is_constant for path in self.parameter_paths.values())

    @property
    def radius_is_constant(self) -> bool:  # numpydoc ignore=RT01
        """Whether the transverse profile is the same at every z.

        ``True`` when the core radius does not vary, so the synthesis and
        projection matrices of the multimode transform may be built once. A
        pressure gradient changes the effective index but not the field profile.
        """
        return self.core_radius.is_constant

    @property
    def scale_invariant(self) -> bool:  # numpydoc ignore=RT01
        """Whether the profile scales exactly with the core radius.

        Always ``True``: the field depends on ``r`` only through ``r / a`` and the
        normalisation goes as ``a^2``, so a taper rescales the transform matrices
        by two scalars.
        """
        return True

    @property
    def azimuthal_order(self) -> int:  # numpydoc ignore=RT01
        """Highest azimuthal harmonic in the Cartesian field components.

        ``|n - 1|`` for a hybrid mode and 1 for a transverse one. The signed
        ``n - 1`` would be negative for ``HE_{-1,m}`` and would then understate
        the theta rule a cubic product needs.
        """
        return 1 if self.kind in ("TE", "TM") else abs(self.n - 1)

    @property
    def channel_label(self) -> ChannelLabel:  # numpydoc ignore=RT01
        """Family, the two indices and which polarisation partner this is.

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

    def rotated(self, phi: float) -> "MarcatiliMode":
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
    def orthogonal_polarisation(self) -> "MarcatiliMode | None":  # numpydoc ignore=RT01
        """Degenerate partner of a hybrid mode, or ``None``.

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
    def parameter_paths(self) -> Mapping[str, ZProfile]:  # numpydoc ignore=RT01
        """Scalar paths that carry the whole z dependence.

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
    ) -> "MarcatiliMode":
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

        - ``values``: finite, strictly positive angular frequencies in rad/s.
        - Returns: the frequency-only terms, as :meth:`sampled_terms` documents.
        """
        susceptibility, dsusceptibility = self._sampled_susceptibility(values)
        if isinstance(self.cladding_index, GlassIndex):
            permittivity = np.asarray(self.cladding_index.permittivity(values))
            dpermittivity = np.asarray(self.cladding_index.dpermittivity_domega(values))
        else:
            permittivity = np.full(values.shape, complex(self.cladding_index) ** 2)
            dpermittivity = np.zeros_like(permittivity)
        term, dterm = cladding_term_and_derivative(
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

        - ``radius``: core radius in m at the position of interest.
        - Returns: ``w = (u/(ka))^2 (1 - i v/(ka))^2`` and ``dw/domega``.
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

        - ``radius``: core radius in m.
        - ``density``: number density in m^-3, scalar for the existing gas spelling or
          one entry per sampled mixture component.
        - Returns: the refractivity and its derivative, with the model's own loss
          included.
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
        if not self._lossless:
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
        chi = core_susceptibility_at(terms, self._scalars(z)[1])
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
        if not self._INDEX_IS_MARCATILI:
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
        dchi = core_susceptibility_at(terms, rates)
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
            if not self._lossless:
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
        """Apply the loss declaration, the no-gain clamp and the floor to a slope.

        The ``z`` counterpart of ``_apply_loss`` and ``_floored``, zeroing the slope
        wherever they zero the frequency derivative.

        - ``derivative``: its ``z`` derivative in 1/m.
        - Returns: the ``z`` derivative of what :meth:`neff` returns, in 1/m.
        """
        if self.alpha is not None:
            # An absolute alpha replaces the imaginary part and has no z in it.
            slope = np.real(derivative) + 0j
        else:
            factor = float(self.loss_scale)
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

        - ``z``: position along the waveguide, m.
        - Returns: core radius in m and number density in m^-3 -- the two scalars that
          carry the whole z dependence of the effective index.
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

        - Returns: ``A_eff / a^2``, the ratio of the squared second moment of the
          transverse profile to its fourth moment, evaluated once at construction.
        """
        order = self.n if self.kind == "HE" else 2
        denominator = quad(
            lambda xi: xi * jv(self.n - 1, self.u_nm * xi) ** 4, 0.0, 1.0
        )[0]
        return float(2.0 * math.pi * 0.25 * jv(order, self.u_nm) ** 4 / denominator)


_SHARED_TERMS: ContextVar[
    dict[int, list[tuple[MarcatiliMode, MarcatiliTerms]]] | None
] = ContextVar("solprop_capillary_shared_terms", default=None)
"""The frequency-only terms sampled so far inside :func:`shared_sampling`, by mode.

Keyed by ``id(mode)`` with the mode itself held beside its terms, so an id cannot be
reused by another mode while the scope lasts; ``None`` outside a scope, where every
call samples afresh exactly as it always has."""


@contextmanager
def shared_sampling() -> Iterator[None]:
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

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.modes.capillary import MarcatiliMode, shared_sampling
    >>> mode = MarcatiliMode(125e-6, "He", 1.0)
    >>> omega = np.array([2.0e15, 2.5e15])
    >>> with shared_sampling():
    ...     same = mode.sampled_terms(omega) is mode.sampled_terms(omega)
    >>> same
    True
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

    - Returns: True when the models are the same objects, in order, with equal absences.
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

    - ``omega``: angular frequencies in rad/s, shape ``(n_omega,)``.
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


def core_susceptibility_at(
    terms: MarcatiliTerms, density: float | Float[np.ndarray, " n_species"]
) -> Complex[np.ndarray, " n_omega"]:
    """Return the core susceptibility of a fill at one position.

    Parameters
    ----------
    terms : MarcatiliTerms
        A mode's frequency-only terms, whose per-particle susceptibility is one row
        or one row per sampled species.
    density : float or numpy.ndarray
        Number density in m^-3, scalar or one entry per sampled species.

    Returns
    -------
    numpy.ndarray
        The dimensionless susceptibility, complex, shape (n_omega,).
    """
    if terms.core_susceptibility.ndim == 1:
        return np.asarray(terms.core_susceptibility * density, dtype=complex)
    weights = np.atleast_1d(density)
    return np.asarray(
        np.tensordot(weights, terms.core_susceptibility, axes=(0, 0)), dtype=complex
    )
