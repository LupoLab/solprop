# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Model the hollow planar (slab) waveguide: two plates, a gas-filled gap, uniform y.

Two parallel dielectric or metal plates at :math:`x = \pm a` guide light in x and
leave it free in y. This module models the guide **uniform in y**: the field is a plane
wave in y, the mode is the one-wall-pair limit :math:`b \to \infty` of the rectangular
guide of :mod:`solprop.modes.rectangular`, and every integral over the cross-section
is an integral over x **per metre of y**. A run on these modes reports power in W/m
and energy in J/m (``transverse_measure = "per_unit_width"``); intensities stay W/m².

The model is Laakmann and Steier's eq. (5) [1]_ with :math:`b \to \infty`, which is
Miyagi, Hongo and Kawakami's eq. (19) [2]_ and Arnold et al.'s eqs. (2)-(3) [3]_:

.. math::

   \kappa_p = \frac{p\pi}{2a}\Big(1 - \frac{i\,q}{k a}\Big),\qquad
   n_{\rm eff}^2 = \epsilon_{\rm co} - \frac{\kappa_p^2}{k^2},\qquad
   q_{\rm TE} = \frac{1}{s},\quad
   q_{\rm TM} = \frac{\epsilon_{\rm cl}}{\epsilon_{\rm co}\, s},

with :math:`s = \sqrt{\epsilon_{\rm cl} - \epsilon_{\rm co}}` the filled-core
contrast and :math:`\epsilon_{\rm cl}/\epsilon_{\rm co}` the TM-like permittivity
ratio of a filled core, both shared with the rectangular guide rather than restated.
To first order the field attenuation is
:math:`\alpha_p = (1/a)(p\lambda/4a)^2\,\mathrm{Re}\,q`, so the loss goes as
:math:`p^2` and :math:`a^{-3}`, and TM/TE is :math:`\epsilon_{\rm cl}/\epsilon_{\rm co}`
for glass and about :math:`n^2 + \kappa^2` for a metal (Miyagi eq. 36). **TE** has its
electric field along y, parallel to the plates -- the low-loss polarisation every
planar compressor uses; **TM** has it along x, normal to them.

**What a slab uniform in y cannot show.** Whole-beam self-focusing, filamentation and
the modulational instability in the free direction set the energy limit of real
planar compressors (Arnold et al. 2009, stable to :math:`B \approx 15.5`). A field
uniform in y has no y structure to focus or break up, so none of that appears here.
The manual page is ``docs/guide/waveguides/planar.md`` (section "Hollow planar
waveguides").

References
----------
.. [1] K. D. Laakmann and W. H. Steier, Appl. Opt. 15, 1334-1340 (1976),
       doi:10.1364/AO.15.001334.
.. [2] M. Miyagi, A. Hongo and S. Kawakami, IEEE J. Quantum Electron. QE-19,
       136-145 (1983), doi:10.1109/JQE.1983.1071837.
.. [3] C. L. Arnold et al., New J. Phys. 12, 073015 (2010),
       doi:10.1088/1367-2630/12/7/073015.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from types import MappingProxyType
from typing import Final, Literal, cast

import numpy as np
from jaxtyping import Complex, Float

from solprop._display import Displayed, engineering, line
from solprop.constants import EPS0, C
from solprop.exceptions import GridError, ModeError
from solprop.grid import _periodic_axis
from solprop.modes.fill import LossSpec, _sqrt1p_minus_one
from solprop.modes.protocol import (
    ChannelLabel,
    CrossSection,
    Mode,
    PolarisationComponent,
    TransverseMeasure,
)
from solprop.modes.quadrature import (
    TransverseQuadrature,
    cartesian_nodes,
    mode_matrix,
    on_axis_rule,
    quadrature_rule,
    transverse_quadrature,
)
from solprop.modes.rectangular import _RectTerms, _transverse_term, _WallPairFill
from solprop.physdata.gases import Gas
from solprop.physdata.glasses import GlassIndex, fused_silica
from solprop.physdata.mixtures import Mixture
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.profiles import Constant, ZProfile, _temperature_text, as_profile

__all__ = [
    "PlanarGrid",
    "SlabMode",
    "SlabPolarisation",
    "gaussian_coupling",
    "gaussian_projection",
    "gaussian_to_per_unit_width",
    "per_unit_width_to_gaussian",
]

SlabPolarisation = Literal["te", "tm"]
"""Name a slab family: ``"te"`` has E along y, ``"tm"`` has E along x."""

_GAUSSIAN_WIDTH: Final = math.sqrt(math.pi / 2.0)
"""``int exp(-2 y^2 / w^2) dy / w``: the width a Gaussian of 1/e field radius w has."""

_COUPLING_NODES: Final = 256
"""Gauss nodes of the rule :func:`gaussian_coupling` projects on.

The integrand is a Gaussian times a trigonometric profile; 256 nodes reach the
rounding floor for every waist above ``a / 50`` and every order up to 20, which the
tests check against an adaptive integral. A narrower launch is not a slab launch.
"""


@dataclass(frozen=True, eq=False)
class SlabMode(_WallPairFill):
    """Describe one mode of a gas-filled hollow planar waveguide, uniform in y.

    The plates are at ``x = +-a`` and y is free. The profile is ``cos(p pi x / 2a)``
    for odd ``p`` and ``sin(p pi x / 2a)`` for even ``p`` inside the gap and zero
    outside; the effective index is Laakmann and Steier's one-wall-pair form with
    the filled-core contrast (module docstring). Everything the mode integrates is
    **per metre of y** (:attr:`transverse_measure`).

    Attributes
    ----------
    half_gap : float or ZProfile
        Half the plate separation, ``a``, in m, positive. A path along z is a
        scale-invariant taper: the profile is a function of ``x / a(z)``.
    gas : Gas or str or None
        Legacy gas declaration; None means vacuum unless a mixture is supplied.
    pressure : float or ZProfile
        Legacy gas pressure in bar, non-negative.
    order : int
        Mode order ``p >= 1``; odd orders are symmetric about the mid-plane, so a
        centred launch couples only to them.
    polarisation : {'te', 'tm'}
        ``"te"`` (the default) has the electric field along y, parallel to the
        plates, and the low loss; ``"tm"`` has it along x, normal to them, and a
        loss larger by the wall's ``eps_cl / eps_co`` for glass. The two are
        distinct modes with their own effective index.
    loss : bool or float or callable
        Shared hollow-mode policy: model loss, no loss, a non-negative model scale,
        or an absolute power attenuation in 1/m.
    cladding : GlassIndex or complex or None
        Plate index, positive imaginary part for absorption: a glass, a metal of
        :mod:`solprop.physdata.metals`, or a constant complex number. None selects
        fused silica.
    temperature : float, ZProfile or None
        Positive legacy fill temperature in K, or a path of it along z in m, which
        the density follows. ``None`` (the default) is 293.15 K for a gas and
        the mixture's own beside a mixture; a different one beside it is refused.
    susceptibility : GasSusceptibility or None
        Explicit legacy gas susceptibility; ``None`` takes the built-in gas's.
    mixture : Mixture or None
        Ordered material populations; excludes competing legacy fill arguments.
    width_x : ZProfile
        The resolved half-gap path in m. It carries the rectangular guide's name for
        its x half-width, which this is, so that the geometry-path walkers of
        :data:`solprop.modes.protocol.GEOMETRY_PATH_NAMES` find its kinks.
    density : ZProfile
        Total number-density path in m^-3.
    cladding_index : GlassIndex or complex
        Resolved plate material.
    model : {'full', 'reduced'}
        Fixed internally to full, selecting the shared evanescent floor.

    References
    ----------
    K. D. Laakmann and W. H. Steier, Applied Optics 15, 1334-1340 (1976),
    doi:10.1364/AO.15.001334, eq. (5) with ``b -> infinity``; C. L. Arnold et al.,
    New J. Phys. 12, 073015 (2010), eqs. (2)-(3). The filled-core factor
    ``eps_cl / (eps_co s)`` of the TM-like coefficient follows from grazing-incidence
    Fresnel reflection; it reduces to the published ``eps_cl / s`` for an evacuated
    core.

    Examples
    --------
    >>> from solprop.modes import SlabMode
    >>> mode = SlabMode(63.5e-6, cladding=1.45)
    >>> mode.channel_label.display_label, mode.principal_component()
    ('TE1', 'y')
    >>> f"{mode.aeff() * 1e6:.3f} um effective width"
    '84.667 um effective width'
    """

    half_gap: float | ZProfile
    gas: Gas | str | None = None
    pressure: float | ZProfile = 0.0
    _: KW_ONLY
    order: int = 1
    polarisation: SlabPolarisation = "te"
    loss: LossSpec = True
    cladding: GlassIndex | complex | None = None
    temperature: float | ZProfile | None = None
    susceptibility: GasSusceptibility | None = None
    mixture: Mixture | None = None
    width_x: ZProfile = field(init=False, repr=False)
    density: ZProfile = field(init=False, repr=False)
    cladding_index: GlassIndex | complex = field(init=False, repr=False)
    model: Literal["full", "reduced"] = field(init=False, default="full", repr=False)
    _gain_warned: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the mode and resolve the gap and material paths.

        Raises
        ------
        ModeError
            If the order, the polarisation, the gap or the fill declaration is
            invalid.
        """
        # The once-per-mode validity flag: an attribute rather than a field, so the
        # declared fields stay what they were, and a list so that the shallow copies
        # `at_parameters` makes share it (`_WallPairFill._warn_validity`).
        object.__setattr__(self, "_validity_warned", [False])
        # bool is an int in Python; True is not a mode order.
        if (
            not isinstance(self.order, int)
            or isinstance(self.order, bool)
            or self.order < 1
        ):
            raise ModeError(
                "a slab mode order must be an integer of at least 1, got "
                f"{self.order!r}"
            )
        if self.polarisation not in ("te", "tm"):
            raise ModeError(
                f"slab polarisation must be 'te' (E along y, parallel to the plates) "
                f"or 'tm' (E along x), got {self.polarisation!r}"
            )
        self._check_fill_arguments()
        object.__setattr__(self, "width_x", as_profile(self.half_gap))
        self._resolve_fill()
        object.__setattr__(
            self,
            "cladding_index",
            fused_silica() if self.cladding is None else self.cladding,
        )
        positions = np.unique(
            np.concatenate(
                (self.width_x.breakpoints, np.asarray(self.width_x.z_span), [0.0])
            )
        )
        for position in positions:
            if math.isfinite(float(position)):
                self._half_gap(float(position))
        self._warn_loss_derivative()

    def _half_gap(self, z: float) -> float:
        """Return the positive half-gap at one longitudinal position.

        Parameters
        ----------
        z : float
            Position in m.

        Returns
        -------
        float
            Half-gap ``a`` in m.

        Raises
        ------
        ModeError
            If the half-gap is non-positive or non-finite there.
        """
        a = float(self.width_x(np.asarray([z]))[0])
        if not (math.isfinite(a) and a > 0):
            raise ModeError(
                f"a slab half-gap must be positive and finite, got {a!r} at z = {z!r}"
            )
        return a

    @property
    def transverse_measure(self) -> TransverseMeasure:
        """Return what this mode's transverse integrals measure.

        Returns
        -------
        str
            ``"per_unit_width"``: the normalisation is a power per metre of y and
            :meth:`aeff` an effective width, so a run's powers are W/m and its
            energies J/m (:data:`solprop.modes.protocol.TransverseMeasure`).
        """
        return "per_unit_width"

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the gap and material paths carrying longitudinal variation.

        Returns
        -------
        collections.abc.Mapping
            The half-gap path plus the shared fill's density or pressure paths.
        """
        return MappingProxyType({"half_gap": self.width_x, **self._fill_paths()})

    def at_parameters(self, values: Mapping[str, float]) -> "SlabMode":  # noqa: UP037 - eager hook
        """Copy the mode with the supplied gap and fill paths frozen.

        Parameters
        ----------
        values : collections.abc.Mapping
            A subset of the declared paths, in m, m^-3 or shared pressure in bar.

        Returns
        -------
        SlabMode
            A copy retaining unspecified paths and the same wall model.

        Raises
        ------
        ModeError
            If a path is unknown or the half-gap is invalid.
        """
        unknown = set(values) - set(self.parameter_paths)
        if unknown:
            raise ModeError(f"unknown slab parameters: {sorted(unknown)}")
        frozen = cast(
            SlabMode,
            self._frozen_fill(
                {key: value for key, value in values.items() if key != "half_gap"}
            ),
        )
        if "half_gap" in values:
            object.__setattr__(frozen, "half_gap", float(values["half_gap"]))
            object.__setattr__(frozen, "width_x", Constant(float(values["half_gap"])))
        frozen._half_gap(0.0)
        return frozen

    @property
    def z_constant(self) -> bool:
        """Return whether the gap and every material path are constant.

        Returns
        -------
        bool
            True if all declared paths are constant.
        """
        return all(path.is_constant for path in self.parameter_paths.values())

    @property
    def radius_is_constant(self) -> bool:
        """Return the transform's geometry-constant trait for the slab.

        Returns
        -------
        bool
            True when the half-gap path is constant; the historical trait name also
            serves Cartesian and slab modes.
        """
        return self.width_x.is_constant

    @property
    def scale_invariant(self) -> bool:
        """Return whether one scalar rescaling represents the geometry along z.

        Returns
        -------
        bool
            Always True: one gap is the whole geometry, and the profile is a function
            of ``x / a(z)``, so a taper rescales the matrices by
            ``sqrt(a(0) / a(z))`` (:func:`solprop.modes.quadrature.taper_scale`).
        """
        return True

    @property
    def polarisation_degenerate(self) -> bool:
        """Return whether the two polarisations share one effective index.

        Returns
        -------
        bool
            False: TE and TM take different wall coefficients, so they are distinct
            modes and need a row each, as the rectangular partners do.
        """
        return False

    @property
    def azimuthal_order(self) -> int:
        """Return the mode order as the rule's resolution hint.

        Returns
        -------
        int
            ``p``. A slab has no azimuthal coordinate; the x rule's node bound is the
            measured :func:`solprop.modes.quadrature.cartesian_nodes` of it.
        """
        return self.order

    @property
    def channel_label(self) -> ChannelLabel:
        """Return the slab family and order.

        Returns
        -------
        ChannelLabel
            Family ``TE`` or ``TM``, no azimuthal order, the mode order ``p`` in the
            radial field -- the one index the record carries -- and ``TE1``,
            ``TM3`` and so on as the display label. The other polarisation is a
            distinct mode, not a parity partner, so ``parity`` stays ``None``.

        Examples
        --------
        >>> from solprop.modes import SlabMode
        >>> SlabMode(50e-6, order=3, polarisation="tm").channel_label.display_label
        'TM3'
        """
        family = self.polarisation.upper()
        return ChannelLabel(
            family, None, self.order, None, None, f"{family}{self.order}"
        )

    def principal_component(self, z: float = 0.0) -> PolarisationComponent:
        """Return the nonzero Cartesian electric-field component.

        Parameters
        ----------
        z : float, optional
            Position in m; the orientation is fixed.

        Returns
        -------
        str
            ``"y"`` for TE, ``"x"`` for TM.
        """
        self._half_gap(z)
        return "y" if self.polarisation == "te" else "x"

    @property
    def orthogonal_polarisation(self) -> "SlabMode":  # noqa: UP037 - eager hook
        """Return the same order with the other polarisation.

        Returns
        -------
        SlabMode
            TM for a TE mode and TE for a TM one: a distinct mode with its own
            effective index and its own operator row.
        """
        return replace(self, polarisation="tm" if self.polarisation == "te" else "te")

    def _model_values(
        self, terms: _RectTerms, a: float
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Evaluate the one-wall-pair reflection model before the loss policy.

        Parameters
        ----------
        terms : _RectTerms
            Core and wall samples, with holomorphic frequency continuation allowed.
        a : float
            Positive half-gap in m.

        Returns
        -------
        tuple of numpy.ndarray
            Refractivity and its frequency derivative in s/rad.

        Raises
        ------
        ModeError
            If the wall contrast vanishes or the model produces a non-finite result.
        """
        te, dte, tm, dtm = self._wall_coefficients(terms)
        # E parallel to the plates is the TE-like (s) reflection, E normal to them
        # the TM-like (p) one: the rectangular guide's y-polarised mode sees te on
        # its x walls, which is this TE mode in the limit b -> infinity.
        q, dq = (te, dte) if self.polarisation == "te" else (tm, dtm)
        decrement, ddecrement = _transverse_term(terms.omega, a, self.order, q, dq)
        argument = terms.chi - decrement
        # The rationalised sqrt(1 + argument) - 1, as the rectangular guide forms
        # it, for the large-gap limit where the refractivity is tiny.
        refractivity = _sqrt1p_minus_one(argument)
        derivative = (terms.dchi - ddecrement) / (2 * (1 + refractivity))
        if not (np.all(np.isfinite(refractivity)) and np.all(np.isfinite(derivative))):
            raise ModeError("the slab model produced non-finite values")
        return np.asarray(refractivity), np.asarray(derivative)

    def _values(
        self, omega: Float[np.ndarray, " n_omega"], z: float
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Apply the shared loss policy and floor to the local model.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive frequencies in rad/s.
        z : float
            Position in m.

        Returns
        -------
        tuple of numpy.ndarray
            Refractivity and its derivative in s/rad after loss and floor policies.
        """
        a = self._half_gap(z)
        terms = self._terms(omega, z)
        values, derivative = self._model_values(terms, a)
        self._warn_validity(terms, ((a, self.order, self.polarisation),))
        return self._floored(*self._apply_loss(omega, values, derivative))

    def neff(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the effective index with positive imaginary part for attenuation.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive frequencies in rad/s.
        z : float, optional
            Position in m.

        Returns
        -------
        numpy.ndarray
            Complex effective index, shape (n_omega,).
        """
        return 1 + self._values(omega, z)[0]

    def neff_minus_one(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the refractivity directly from the small square-root argument.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive frequencies in rad/s.
        z : float, optional
            Position in m.

        Returns
        -------
        numpy.ndarray
            Complex ``n_eff - 1``, without subtracting unity from the index.
        """
        return self._values(omega, z)[0]

    def dneff_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Complex[np.ndarray, " n_omega"]:
        """Return the analytic derivative including core and wall material slopes.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive frequencies in rad/s.
        z : float, optional
            Position in m.

        Returns
        -------
        numpy.ndarray
            Complex index derivative in s/rad; only a slope-free loss callable uses
            the shared numerical loss derivative.
        """
        return self._values(omega, z)[1]

    def aeff(self, z: float = 0.0) -> float:
        """Return the closed-form effective width, per unit width in y.

        The mode protocol's effective area, ``(int V^2)^2 / int V^4``, taken over x
        alone because the guide is uniform in y: an **effective width**, in m, the
        same ``4a/3`` for every order. It is what the mode-averaged normalisation
        and ``gamma' = n2 omega / (c A_eff)`` read, so a mode-averaged run's powers
        come out per metre of y without a second name for the same quantity.

        Parameters
        ----------
        z : float, optional
            Position in m.

        Returns
        -------
        float
            ``4 a / 3`` in m.
        """
        return 4.0 / 3.0 * self._half_gap(z)

    def normalisation(self, z: float = 0.0) -> float:
        """Return the power per unit width carried by the unnormalised profile.

        Parameters
        ----------
        z : float, optional
            Position in m.

        Returns
        -------
        float
            ``eps0 c a / 2`` in W/m for a unit field amplitude: the vacuum
            admittance times ``int V^2 dx = a``, halved.
        """
        return 0.5 * EPS0 * C * self._half_gap(z)

    def __repr__(self) -> str:
        """Return a one-line description: the mode, the gap and the fill.

        Returns
        -------
        str
            For example ``SlabMode(TE1, gap 127 um, Ar 1 bar, loss=True)``.
        """
        gap = 2.0 * float(self.width_x(np.array([0.0]))[0])
        return line(
            "SlabMode",
            [
                ("", self.channel_label.display_label),
                ("", f"gap {engineering(gap, 'm')}"),
                ("", self._fill_text()),
                ("loss", self._loss_text()),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        cladding = self.cladding_index
        return f"SlabMode {self.channel_label.display_label}", [
            ("gap", engineering(2.0 * float(self.width_x(np.array([0.0]))[0]), "m")),
            ("fill", self._fill_text()),
            (
                "temperature",
                _temperature_text(cast("float | ZProfile", self.temperature)),
            ),
            ("loss", self._loss_text()),
            ("plates", getattr(cladding, "name", None) or f"constant index {cladding}"),
            ("measure", "per unit width in y (W/m, J/m)"),
        ]

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, ...], tuple[float, ...]]:
        """Return the slab cross-section and its integration bounds in x.

        Parameters
        ----------
        z : float, optional
            Position in m.

        Returns
        -------
        tuple
            ``("slab", (-a,), (a,))``, in m: one coordinate, y being uniform.
        """
        a = self._half_gap(z)
        return "slab", (-a,), (a,)

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Return the unnormalised real Cartesian electric-field components.

        Parameters
        ----------
        points : numpy.ndarray
            ``(x, y)`` coordinates in m, shape (n_points, 2); y is ignored.
        z : float, optional
            Position in m.

        Returns
        -------
        numpy.ndarray
            ``(E_x, E_y)``, shape (n_points, 2): the profile on ``E_y`` for TE and on
            ``E_x`` for TM, zero outside the gap.

        Raises
        ------
        ModeError
            If the points are non-finite or have an invalid shape.
        """
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ModeError("slab points must be finite with shape (n_points, 2)")
        a = self._half_gap(z)
        phase = self.order * math.pi * points[:, 0] / (2 * a)
        amplitude = np.cos(phase) if self.order % 2 else np.sin(phase)
        amplitude = np.where(np.abs(points[:, 0]) <= a, amplitude, 0.0)
        zeros = np.zeros_like(amplitude)
        return np.column_stack(
            (zeros, amplitude) if self.polarisation == "te" else (amplitude, zeros)
        )


def gaussian_coupling(
    modes: Sequence[Mode],
    waist: float,
    *,
    offset: float = 0.0,
    component: PolarisationComponent | None = None,
    z: float = 0.0,
) -> Float[np.ndarray, " n_modes"]:
    """Return the share of a Gaussian beam's power each slab mode takes.

    The beam is ``exp(-((x - offset) / waist)^2)`` in x, uniform in y, with flat
    phase and one linear polarisation, `component`. It is projected onto the modes
    through the slab rule's dual basis (:func:`solprop.modes.quadrature.mode_matrix`),
    and each share is the modal power over the beam's **whole** power, inside the gap
    and out: what lands on the plates is lost to every mode, so the shares sum to
    less than one. A centred beam couples to odd orders only, and a mode polarised
    across the beam takes nothing.

    Parameters
    ----------
    modes : sequence of Mode
        Slab modes sharing one gap, non-empty.
    waist : float
        The beam's 1/e **field** radius in x, m, > 0.
    offset : float, optional
        The beam centre's displacement from the mid-plane, m.
    component : {'x', 'y'} or None, optional
        The beam's electric-field direction: ``"y"`` launches TE, ``"x"`` TM.
        ``None`` (the default) takes the set's own component, which is refused for
        a set carrying both, where one beam cannot be both polarisations.
    z : float, optional
        Position along the guide the gap is taken at, m.

    Returns
    -------
    numpy.ndarray
        Fraction of the beam's power per unit width in each mode, shape
        ``(n_modes,)``, in ``[0, 1]``; zero for a mode of the other polarisation.

    Raises
    ------
    ModeError
        If the set is empty, is not a slab set, `waist` is not positive, or
        `component` is ``None`` for a set with both TE and TM modes.

    Examples
    --------
    >>> from solprop.modes import SlabMode, gaussian_coupling
    >>> modes = [SlabMode(1.0, order=p) for p in (1, 2, 3)]
    >>> [f"{share:.4f}" for share in gaussian_coupling(modes, 0.735)]
    ['0.9882', '0.0000', '0.0008']
    >>> pair = [SlabMode(1.0), SlabMode(1.0, polarisation="tm")]
    >>> [f"{share:.4f}" for share in gaussian_coupling(pair, 0.735, component="y")]
    ['0.9882', '0.0000']
    """
    amplitudes = gaussian_projection(
        modes, waist, offset=offset, component=component, z=z
    )
    # Each amplitude is the modal share in sqrt(W/m) of a unit-peak beam; the
    # beam's own power per unit width is (eps0 c / 2) waist sqrt(pi / 2).
    return amplitudes**2 / (0.5 * EPS0 * C * waist * _GAUSSIAN_WIDTH)


def gaussian_projection(
    modes: Sequence[Mode],
    waist: float,
    *,
    offset: float = 0.0,
    component: PolarisationComponent | None = None,
    z: float = 0.0,
) -> Float[np.ndarray, " n_modes"]:
    """Project a unit-peak Gaussian beam in x onto slab modes, with signs.

    The beam is ``exp(-((x - offset) / waist)^2)`` in V/m, uniform in y, polarised
    along `component`. Each entry is the modal amplitude in sqrt(W/m) that beam
    launches into one mode: ``(eps0 c / 2) int E(x) e_p(x) dx`` over the gap, with
    ``e_p`` the mode's field normalised to 1 W/m, through the slab rule's dual basis
    (:func:`solprop.modes.quadrature.mode_matrix`) on a fine Gauss rule. Unlike
    :func:`gaussian_coupling` it keeps the sign each order takes, which a launch
    needs: an offset beam couples to even orders with the sign of the offset.

    Parameters
    ----------
    modes : sequence of Mode
        Slab modes sharing one gap, non-empty.
    waist : float
        The beam's 1/e **field** radius in x, m, > 0.
    offset : float, optional
        The beam centre's displacement from the mid-plane, m.
    component : {'x', 'y'} or None, optional
        The beam's electric-field direction: ``"y"`` launches TE, ``"x"`` TM.
        ``None`` (the default) takes the set's own component, which is refused for
        a set carrying both.
    z : float, optional
        Position along the guide the gap is taken at, m.

    Returns
    -------
    numpy.ndarray
        Modal amplitudes in sqrt(W/m) per V/m of beam peak, shape ``(n_modes,)``;
        zero for a mode polarised across the beam.

    Raises
    ------
    ModeError
        If the set is empty, is not a slab set, `waist` is not positive, or
        `component` is ``None`` for a set with both TE and TM modes.

    Examples
    --------
    >>> from solprop.modes import SlabMode, gaussian_projection
    >>> modes = [SlabMode(1.0, order=p) for p in (1, 2)]
    >>> above = gaussian_projection(modes, 0.735, offset=0.1)
    >>> below = gaussian_projection(modes, 0.735, offset=-0.1)
    >>> bool(above[1] > 0), bool(below[1] < 0), bool(above[0] == below[0])
    (True, True, True)
    """
    if not modes:
        raise ModeError("gaussian_coupling needs at least one slab mode")
    if not (math.isfinite(waist) and waist > 0):
        raise ModeError(f"the waist must be positive and finite, got {waist!r}")
    limits = modes[0].dimlimits(z)
    if limits[0] != "slab":
        raise ModeError(
            f"gaussian_coupling projects onto slab modes; {type(modes[0]).__name__} "
            f"has a {limits[0]} cross-section"
        )
    if component is None:
        own = {mode.principal_component(z) for mode in modes}
        if len(own) != 1 or None in own:
            raise ModeError(
                "gaussian_coupling was given TE and TM modes and no component=: one "
                "linearly polarised beam launches one family; pass component='y' "
                "(TE) or component='x' (TM)"
            )
        component = cast("PolarisationComponent", own.pop())
    rule = quadrature_rule("slab", full=False, n_r=_COUPLING_NODES)
    points = rule.nodes(limits)
    beam = np.exp(-(((points[:, 0] - offset) / waist) ** 2))
    amplitudes = np.empty(len(modes))
    for index, mode in enumerate(modes):
        projection = mode_matrix([mode], rule, z, components=(component,)).projection[0]
        # The projection is the dual basis (eps0 c / 2) int E e / sqrt(N): the
        # modal amplitude in sqrt(W/m) of a unit-peak beam polarised along
        # `component`, zero for a mode polarised across it.
        amplitudes[index] = float(projection @ beam)
    return amplitudes


def per_unit_width_to_gaussian(value: float, w_y: float) -> float:
    """Return the total of a per-unit-width quantity spread as a Gaussian in y.

    A run uniform in y reports power in W/m and energy in J/m: the value at the
    centre of a beam that is wide in y. For a beam ``exp(-(y / w_y)^2)`` in field --
    ``w_y`` the **1/e field radius**, the 1/e^2 intensity radius -- the total is
    ``value * w_y * sqrt(pi / 2)``, because ``int exp(-2 y^2 / w_y^2) dy`` is
    ``w_y sqrt(pi / 2)``. Power and energy convert alike.

    Parameters
    ----------
    value : float
        Power in W/m or energy in J/m at the beam centre.
    w_y : float
        1/e field radius in y, m, > 0.

    Returns
    -------
    float
        Total power in W or energy in J.

    Raises
    ------
    ModeError
        If `w_y` is not positive and finite.

    Examples
    --------
    >>> from solprop.modes import per_unit_width_to_gaussian
    >>> f"{per_unit_width_to_gaussian(1.0, 2e-3):.4e}"
    '2.5066e-03'
    """
    return float(value) * _checked_radius(w_y) * _GAUSSIAN_WIDTH


def gaussian_to_per_unit_width(value: float, w_y: float) -> float:
    """Return the per-unit-width value at the centre of a Gaussian beam in y.

    The inverse of :func:`per_unit_width_to_gaussian`: a total power in W or energy
    in J carried by ``exp(-(y / w_y)^2)`` in field, with ``w_y`` the **1/e field
    radius**, is ``value / (w_y sqrt(pi / 2))`` per metre of y at its centre -- the
    number to give a run uniform in y.

    Parameters
    ----------
    value : float
        Total power in W or energy in J.
    w_y : float
        1/e field radius in y, m, > 0.

    Returns
    -------
    float
        Power in W/m or energy in J/m at the centre.

    Raises
    ------
    ModeError
        If `w_y` is not positive and finite.

    Examples
    --------
    >>> from solprop.modes import gaussian_to_per_unit_width
    >>> f"{gaussian_to_per_unit_width(10e-3, 2e-3):.4f} J/m"
    '3.9894 J/m'
    """
    return float(value) / (_checked_radius(w_y) * _GAUSSIAN_WIDTH)


def _checked_radius(w_y: float) -> float:
    """Return a Gaussian radius after checking it.

    Parameters
    ----------
    w_y : float
        1/e field radius in m.

    Returns
    -------
    float
        The same radius.

    Raises
    ------
    ModeError
        If it is not positive and finite.
    """
    if not (math.isfinite(w_y) and w_y > 0):
        raise ModeError(f"the Gaussian radius w_y must be positive, got {w_y!r}")
    return float(w_y)


_MINIMUM_X_NODES: Final = 32
"""Floor on a planar grid's default x node count.

The same floor, and the same eight nodes per mode, as the modal route's
``solprop.driver.simulation.default_radial_order`` gives a slab set, so that the
hybrid transform and ``from_waveguide`` sample the plates on the same default rule;
a test holds the two equal.
"""

_X_NODES_PER_MODE: Final = 8
"""Default x nodes per slab mode; see :data:`_MINIMUM_X_NODES`."""


@dataclass(frozen=True, eq=False)
class PlanarGrid(Displayed):
    """Describe the hybrid planar transverse grid: slab modes in x, samples in y.

    A hollow planar guide resolved in both transverse directions:
    the field is ``E(x, y, t) = sum_p V_p(x) A_p(y, t)``, the slab modes ``V_p`` of
    one gap in x -- sampled on the slab rule -- and a periodic y axis with the
    Cartesian collar, exactly :class:`solprop.grid.FreeGrid`'s. The state's columns
    are ``(p, k_y)``: a modal amplitude per mode and raw y-DFT frequency, flattened
    ``p + n_p*iy`` with the mode index fastest, as a free grid has x fastest. The
    geometry integrates over y, so energies are in J and powers in W
    (``transverse_measure = "area"``), not per unit width as a slab uniform in y.
    ``docs/guide/waveguides/planar.md`` ("Resolving y") and
    ``docs/guide/setup/spatial-decompositions.md``
    describe it.

    Attributes
    ----------
    modes : tuple of SlabMode
        The slab channels, in state order: one gap, one polarisation family, a
        constant gap along z. Each carries the fill.
    radius_y : float
        Requested physical half-width in y, m, finite and positive.
    n_y : int
        Samples in y, an integer of at least 2.
    window_factor : float
        Added collar width divided by `radius_y`, in [0, 1): the periodic half-width
        is ``radius_y * (1 + window_factor)``; zero disables the collar. Default 0.1.
    n_x : int or None
        Slab-rule x nodes asked of
        :func:`solprop.modes.quadrature.transverse_quadrature`; ``None`` (the
        default) takes ``max(32, 8 n_p, 6h + 6)`` with ``h`` the highest order, the
        modal route's default for the same set.
    x_rule : {'gauss_kronrod', 'gauss'}
        The x rule's family: ``"gauss_kronrod"`` (the default, as the modal route)
        carries its embedded Gauss rule and rounds an even `n_x` up by one;
        ``"gauss"`` has no embedded rule.
    rule : solprop.modes.quadrature.TransverseQuadrature
        The slab rule in x.
    x : numpy.ndarray
        The x nodes in m, increasing, inside the gap, shape ``(n_x,)``.
    x_weights : numpy.ndarray
        Their quadrature weights in m, shape ``(n_x,)``.
    y, k_y : numpy.ndarray
        The centred y axis in m and its angular frequencies in rad/m in unshifted
        FFT order, shape ``(n_y,)``.
    window_y : numpy.ndarray
        The y collar, shape ``(n_y,)``.
    synthesis : numpy.ndarray
        Mode fields at the x nodes, each normalised to 1 W/m, in V/(m sqrt(W/m)),
        shape ``(n_x, n_p)``, without any temporal pad scale.
    projection, projection_coarse : numpy.ndarray
        The dual basis with the weights and ``eps0 c / 2`` folded in, and its
        embedded coarse-rule counterpart, shape ``(n_p, n_x)``.
    centre : numpy.ndarray
        Each mode's normalised field at the mid-plane ``x = 0``, shape ``(n_p,)``.

    Raises
    ------
    GridError
        If the y axis, the collar or `n_x` is invalid.
    ModeError
        If the set is empty, holds a mode that is not a :class:`SlabMode`, repeats a
        channel, mixes gaps, fills or polarisation families, or has a gap that
        varies along z: the hybrid transform carries a constant gap only.

    Warns
    -----
    UserWarning
        If `n_x` is below the measured ``6h + 6`` bound of the set's highest order,
        naming the count to use.

    Examples
    --------
    >>> from solprop.modes import PlanarGrid, SlabMode
    >>> modes = [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]
    >>> space = PlanarGrid(modes, 2e-3, 64)
    >>> space.n_columns, space.n_nodes, space.x.size
    (128, 2112, 33)
    >>> space.transverse_measure
    'area'
    """

    modes: Sequence[Mode]
    radius_y: float
    n_y: int
    _: KW_ONLY
    window_factor: float = 0.1
    n_x: int | None = None
    x_rule: Literal["gauss_kronrod", "gauss"] = "gauss_kronrod"
    rule: TransverseQuadrature = field(init=False, repr=False)
    x: Float[np.ndarray, " n_x"] = field(init=False, repr=False)
    x_weights: Float[np.ndarray, " n_x"] = field(init=False, repr=False)
    y: Float[np.ndarray, " n_y"] = field(init=False, repr=False)
    k_y: Float[np.ndarray, " n_y"] = field(init=False, repr=False)
    window_y: Float[np.ndarray, " n_y"] = field(init=False, repr=False)
    synthesis: Float[np.ndarray, "n_x n_p"] = field(init=False, repr=False)
    projection: Float[np.ndarray, "n_p n_x"] = field(init=False, repr=False)
    projection_coarse: Float[np.ndarray, "n_p n_x"] = field(init=False, repr=False)
    centre: Float[np.ndarray, " n_p"] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the set, then build the slab rule in x and the periodic y axis."""
        modes = tuple(self.modes)
        object.__setattr__(self, "modes", modes)
        _check_planar_modes(modes)
        if isinstance(self.n_y, bool) or not isinstance(self.n_y, int) or self.n_y < 2:
            raise GridError("PlanarGrid.n_y must be an integer of at least 2")
        if not math.isfinite(self.radius_y) or self.radius_y <= 0:
            raise GridError("PlanarGrid.radius_y must be finite and positive")
        if not math.isfinite(self.window_factor) or not 0 <= self.window_factor < 1:
            raise GridError("PlanarGrid.window_factor must be finite and lie in [0, 1)")
        highest = max(mode.azimuthal_order for mode in modes)
        needed = cartesian_nodes(highest)
        count = (
            max(_MINIMUM_X_NODES, _X_NODES_PER_MODE * len(modes), needed)
            if self.n_x is None
            else self.n_x
        )
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise GridError("PlanarGrid.n_x must be an integer of at least 2")
        if count < needed:
            warnings.warn(
                f"PlanarGrid n_x = {count} is below the measured bound 6 * {highest} "
                f"+ 6 for quartic products of slab modes of order up to {highest}; the "
                f"Kerr and Raman terms will alias. Use n_x = {needed} or more",
                UserWarning,
                stacklevel=3,
            )
        with warnings.catch_warnings():
            # The rule's own warning names its keyword, n_r; the one above names
            # the keyword this grid takes.
            warnings.filterwarnings(
                "ignore", message=r"n_r = \d+ is the x node count", category=UserWarning
            )
            rule = transverse_quadrature(modes, n_r=count, radial_rule=self.x_rule)
        limits = modes[0].dimlimits(0.0)
        component = cast("PolarisationComponent", modes[0].principal_component(0.0))
        matrices = mode_matrix(modes, rule, 0.0, components=(component,))
        centre = mode_matrix(modes, on_axis_rule("slab"), 0.0, components=(component,))
        y, k_y, window_y, _ = _periodic_axis(
            "PlanarGrid", self.radius_y, self.n_y, self.window_factor
        )
        values = {
            "x": np.ascontiguousarray(rule.nodes(limits)[:, 0]),
            "x_weights": np.ascontiguousarray(rule.weights(limits)),
            "y": y,
            "k_y": k_y,
            "window_y": window_y,
            "synthesis": np.ascontiguousarray(matrices.synthesis),
            "projection": np.ascontiguousarray(matrices.projection),
            "projection_coarse": np.ascontiguousarray(matrices.projection_coarse),
            "centre": np.ascontiguousarray(centre.synthesis[0]),
        }
        for name, array in values.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        object.__setattr__(self, "rule", rule)

    @property
    def n_p(self) -> int:
        """Return the number of slab channels.

        Returns
        -------
        int
            ``len(modes)``.
        """
        return len(self.modes)

    @property
    def n_columns(self) -> int:
        """Return the number of state columns, ``(p, k_y)`` pairs.

        Returns
        -------
        int
            ``n_p * n_y``.
        """
        return self.n_p * self.n_y

    @property
    def n_nodes(self) -> int:
        """Return the number of physical nodes, ``(x, y)`` pairs with x fastest.

        Returns
        -------
        int
            ``n_x * n_y`` with ``n_x = x.size``.
        """
        return int(self.x.size) * self.n_y

    @property
    def dy(self) -> float:
        """Return the y spacing.

        Returns
        -------
        float
            In m.
        """
        return float(self.y[1] - self.y[0])

    @property
    def half_gap(self) -> float:
        """Return the plates' half separation.

        Returns
        -------
        float
            ``a`` in m.
        """
        return float(self.modes[0].dimlimits(0.0)[2][0])

    @property
    def transverse_measure(self) -> TransverseMeasure:
        """Return what the grid's transverse integrals measure.

        Returns
        -------
        str
            ``"area"``: the grid integrates over y, so energies are J and powers W.
        """
        return "area"

    @property
    def spectral_weights(self) -> Float[np.ndarray, " n_columns"]:
        """Return the Parseval weight of each state column.

        Returns
        -------
        numpy.ndarray
            ``dy / n_y`` in m, shape ``(n_columns,)``: the energy of a state of
            modal amplitudes in sqrt(W/m) is ``sum w |A|^2`` integrated over time.
        """
        return np.full(self.n_columns, self.dy / self.n_y)

    @property
    def kperp_sq(self) -> Float[np.ndarray, " n_columns"]:
        """Return each column's ``k_y**2``.

        Returns
        -------
        numpy.ndarray
            In m^-2, shape ``(n_columns,)``, the mode index fastest.
        """
        return np.repeat(self.k_y**2, self.n_p)

    @property
    def on_axis(self) -> Complex[np.ndarray, " n_columns"]:
        """Return the weights that reconstruct the field at ``x = y = 0``.

        Returns
        -------
        numpy.ndarray
            ``V_p(0) exp(-i k_y y[0]) / n_y`` in V/(m sqrt(W/m)), complex, shape
            ``(n_columns,)``: the inverse y DFT at ``y = 0`` times each mode's
            mid-plane field.
        """
        phase = np.exp(-1j * self.k_y * self.y[0]) / self.n_y
        return (phase[:, None] * self.centre[None, :]).ravel()

    def __repr__(self) -> str:
        """Return a one-line description: the modes, the gap and the y axis.

        Returns
        -------
        str
            For example ``PlanarGrid(TE1..TE5 x 33, y=2 mm x 128)``.
        """
        labels = [mode.channel_label.display_label for mode in self.modes]
        return line(
            "PlanarGrid",
            [
                ("", f"{','.join(labels)} x {self.x.size}"),
                ("gap", engineering(2 * self.half_gap, "m")),
                ("y", f"{engineering(self.radius_y, 'm')} x {self.n_y}"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        labels = ", ".join(mode.channel_label.display_label for mode in self.modes)
        return "PlanarGrid: slab modes in x, periodic samples in y", [
            ("modes", labels),
            ("gap", engineering(2 * self.half_gap, "m")),
            ("x nodes", f"{self.x.size} (slab rule)"),
            (
                "half-width in y",
                f"{engineering(self.radius_y, 'm')}, {self.n_y} samples",
            ),
            ("spacing in y", engineering(self.dy, "m")),
            ("window factor", f"{self.window_factor:g}"),
            ("measure", "area: energy J, power W"),
        ]


def _check_planar_modes(modes: Sequence[Mode]) -> None:
    """Refuse, by name, a mode set the hybrid planar transform cannot carry.

    Parameters
    ----------
    modes : sequence of Mode
        The proposed channels.

    Raises
    ------
    ModeError
        Naming what is wrong: an empty set, a mode that is not a slab mode, a
        repeated channel, different gaps, TE and TM together, different fills, or a
        gap that varies along z.
    """
    if not modes:
        raise ModeError("a planar grid needs at least one slab mode")
    for mode in modes:
        if not isinstance(mode, SlabMode):
            raise ModeError(
                f"a planar grid resolves slab modes in x; {type(mode).__name__} "
                "is not a SlabMode"
            )
    labels = [mode.channel_label for mode in modes]
    if len(set(labels)) != len(labels):
        raise ModeError(
            "a planar grid lists each slab channel once; a repeated channel would "
            "be two state rows for one mode"
        )
    limits = {mode.dimlimits(0.0) for mode in modes}
    if len(limits) != 1:
        raise ModeError(
            f"every mode of a planar grid shares one gap; the set has {len(limits)}"
        )
    families = {cast("SlabMode", mode).polarisation for mode in modes}
    if len(families) != 1:
        raise ModeError(
            "a planar grid carries one polarisation family: its state is scalar, "
            "one field component per node, and TE (E along y) and TM (E along x) "
            "modes are different components; give TE or TM modes, not both"
        )
    first = cast("SlabMode", modes[0])
    for mode in cast("tuple[SlabMode, ...]", tuple(modes)):
        if not mode.radius_is_constant:
            raise ModeError(
                "a planar grid carries a constant gap: the hybrid transform's "
                "(p, k_y) operator is frozen along z, and a varying gap moves the "
                "slab modes; a moving-gap hybrid is not implemented. Use a constant "
                "half_gap, or run the taper uniform in y through prop_waveguide"
            )
        if (
            mode.gas != first.gas
            or mode.mixture is not first.mixture
            or not np.array_equal(
                mode.density(np.array([0.0])), first.density(np.array([0.0]))
            )
        ):
            raise ModeError(
                "every mode of a planar grid shares one fill: the gas, its pressure "
                "and its temperature set the medium the nonlinear responses see"
            )
