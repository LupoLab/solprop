# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Model gas-filled rectangular waveguides with dielectric or metallic walls.

The two wall pairs use the TE-like and TM-like reflection corrections of
Laakmann and Steier (1976), with the filled-core contrast ``eps_cl - eps_co`` and
the TM-like permittivity ratio ``eps_cl / eps_co``. Complex wall corrections contribute
to both phase and attenuation. This is an asymptotic model, not exact boundary
matching: squaring the first-order boundary factor keeps a real second-order term the
expansion does not control, and the normalisation carries the vacuum admittance where
the modal power strictly carries the filled core's, a choice of the order of the gas
susceptibility. The indices follow one convention: m runs along x and n along y.

One wall pair's decrement (``_transverse_term``) and the material sampling and
TE-like/TM-like reflection coefficients (``_WallPairFill``) are shared with
the hollow planar guide of :mod:`solprop.modes.slab`, which is the one-wall-pair
limit ``b -> infinity`` of this guide.
"""

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
from jaxtyping import Complex, Float, Inexact

from solprop._display import engineering, line
from solprop.constants import EPS0, C
from solprop.exceptions import GrazingIncidenceWarning, ModeError
from solprop.modes.fill import GasFillMixin, LossSpec, _sqrt1p_minus_one
from solprop.modes.protocol import ChannelLabel, CrossSection, PolarisationComponent
from solprop.physdata.gases import Gas
from solprop.physdata.glasses import GlassIndex, fused_silica
from solprop.physdata.mixtures import Mixture
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.profiles import Constant, ZProfile, _temperature_text, as_profile

__all__ = ["RectMode"]


_VALIDITY_RATIO = 0.1
"""Largest first-order ratio a wall-pair guide is evaluated at without a warning.

The model is first order in the grazing angle ``p lambda / 4a`` of the mode's plane
waves and in the wall correction ``lambda |q| / (2 pi a)`` (Laakmann and Steier's
conditions 1-2); the terms it neglects are second order in them, about 1 % of the
mode's decrement at 0.1 and growing quadratically beyond. A grazing angle of 0.1
rad is also well short of glass's Brewster grazing angle ``arctan(1 / n)``, 0.6 rad
for silica, past which the grazing reflection the model expands does not hold.
"""


@dataclass(frozen=True, eq=False)
class _RectTerms:
    """Carry the sampled material values used in the rectangular equation.

    Attributes
    ----------
    omega : numpy.ndarray
        Frequencies in rad/s, with complex continuation allowed for contour checks.
    chi, dchi : numpy.ndarray
        Filled-core susceptibility and its derivative in s/rad.
    wall, dwall : numpy.ndarray
        Relative wall permittivity and its derivative in s/rad.
    """

    omega: Inexact[np.ndarray, " n_omega"]
    chi: Complex[np.ndarray, " n_omega"]
    dchi: Complex[np.ndarray, " n_omega"]
    wall: Complex[np.ndarray, " n_omega"]
    dwall: Complex[np.ndarray, " n_omega"]


def _transverse_term(
    omega: Inexact[np.ndarray, " n_omega"],
    width: float,
    order: int,
    correction: Complex[np.ndarray, " n_omega"],
    slope: Complex[np.ndarray, " n_omega"],
) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
    """Return one wall pair's squared transverse wavenumber divided by k squared.

    Parameters
    ----------
    omega : numpy.ndarray
        Frequencies in rad/s; complex continuation is allowed.
    width : float
        Positive half-width in m.
    order : int
        Positive transverse mode index.
    correction : numpy.ndarray
        Dimensionless TE-like or TM-like reflection coefficient.
    slope : numpy.ndarray
        Frequency derivative of the correction, in s/rad.

    Returns
    -------
    tuple of numpy.ndarray
        Dimensionless decrement and its derivative in s/rad.
    """
    scale = C / (omega * width)
    leading = (order * math.pi * scale / 2) ** 2
    factor = 1 - 1j * scale * correction
    dfactor = -1j * scale * (slope - correction / omega)
    value = leading * factor**2
    return np.asarray(value), np.asarray(
        -2 * value / omega + 2 * leading * factor * dfactor
    )


class _WallPairFill(GasFillMixin):
    """Sample the fill and the walls and form the two grazing reflection coefficients.

    The material half of a hollow guide bounded by Laakmann-Steier wall pairs,
    shared by :class:`RectMode` (two pairs) and
    :class:`~solprop.modes.slab.SlabMode` (one pair), so that the filled-core
    contrast and the TM-like permittivity ratio are written once. Like
    :class:`~solprop.modes.fill.GasFillMixin` it contributes methods only; the
    concrete frozen dataclass declares ``cladding_index`` beside the fill fields.
    """

    cladding_index: GlassIndex | complex
    _validity_warned: list[bool]

    def _warn_validity(
        self, terms: _RectTerms, pairs: Sequence[tuple[float, int, Literal["te", "tm"]]]
    ) -> None:
        """Warn, once per mode, where the first-order model leaves its validity.

        Over the evaluated band, each wall pair's grazing angle ``p lambda / 4a`` and
        wall correction ``lambda |q| / (2 pi a)`` (vacuum wavelength) are compared
        with :data:`_VALIDITY_RATIO`. The flag is shared with the mode's frozen copies
        (:meth:`~solprop.modes.fill.GasFillMixin._frozen_fill` copies the reference),
        so a guide that varies along z warns once, not once per sampled position.

        Parameters
        ----------
        terms : _RectTerms
            Core and wall samples on the evaluated band, real frequencies.
        pairs : sequence of tuple
            One ``(half_width, order, reflection)`` per wall pair, half-width in m and
            reflection ``"te"`` (E parallel to that pair's walls) or ``"tm"``.

        Warns
        -----
        solprop.exceptions.GrazingIncidenceWarning
            The first time either ratio exceeds :data:`_VALIDITY_RATIO` on this mode,
            naming the ratio, its value and the wavelength.
        """
        if self._validity_warned[0]:
            return
        te, _, tm, _ = self._wall_coefficients(terms)
        wavelength = 2 * math.pi * C / np.real(terms.omega)
        worst = (0.0, "", 0.0)
        for width, order, reflection in pairs:
            q = te if reflection == "te" else tm
            for name, ratio in (
                ("p lambda / 4a", order * wavelength / (4 * width)),
                (
                    "lambda |q| / (2 pi a)",
                    wavelength * np.abs(q) / (2 * math.pi * width),
                ),
            ):
                index = int(np.argmax(ratio))
                if float(ratio[index]) > worst[0]:
                    worst = (float(ratio[index]), name, float(wavelength[index]))
        value, name, at = worst
        if value > _VALIDITY_RATIO:
            self._validity_warned[0] = True
            warnings.warn(
                f"{self!r}: the first-order wall-pair model reaches {name} = "
                f"{value:.3g} at {at * 1e9:.4g} nm, above the {_VALIDITY_RATIO} it is "
                "valid to; its neglected terms are second order in that ratio. "
                "Narrow the band towards shorter wavelengths, widen the guide, or "
                "keep lower orders",
                GrazingIncidenceWarning,
                stacklevel=4,
            )

    def _terms(self, omega: Float[np.ndarray, " n_omega"], z: float) -> _RectTerms:
        """Sample the core and wall permittivities and frequency slopes.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive real frequencies in rad/s.
        z : float
            Position in m at which to evaluate the fill.

        Returns
        -------
        _RectTerms
            Frequency-dependent quantities for the local filled guide.

        Raises
        ------
        ModeError
            If omega is empty, non-positive or non-finite.
        """
        if not omega.size or not np.all(np.isfinite(omega)) or np.any(omega <= 0):
            raise ModeError(
                "a wall-pair guide's omega must be positive, finite and non-empty"
            )
        susceptibility, dsusceptibility = self._sampled_susceptibility(omega)
        position = np.asarray([z])
        if susceptibility.ndim == 1:
            density = float(self.density(position)[0])
            chi, dchi = susceptibility * density, dsusceptibility * density
        else:
            assert self.mixture is not None
            densities = self.mixture.density(position)[:, 0]
            chi = susceptibility[0] * densities[0]
            dchi = dsusceptibility[0] * densities[0]
            for row, density in enumerate(densities[1:], 1):
                chi = chi + susceptibility[row] * density
                dchi = dchi + dsusceptibility[row] * density
        if isinstance(self.cladding_index, GlassIndex):
            wall = self.cladding_index.permittivity(omega)
            dwall = self.cladding_index.dpermittivity_domega(omega)
        else:
            wall = np.full(omega.shape, complex(self.cladding_index) ** 2)
            dwall = np.zeros_like(wall)
        return _RectTerms(
            omega,
            np.asarray(chi, dtype=complex),
            np.asarray(dchi, dtype=complex),
            np.asarray(wall),
            np.asarray(dwall),
        )

    def _wall_coefficients(
        self, terms: _RectTerms
    ) -> tuple[
        Complex[np.ndarray, " n_omega"],
        Complex[np.ndarray, " n_omega"],
        Complex[np.ndarray, " n_omega"],
        Complex[np.ndarray, " n_omega"],
    ]:
        """Return the TE-like and TM-like wall coefficients and their slopes.

        Parameters
        ----------
        terms : _RectTerms
            Core and wall samples, with holomorphic frequency continuation allowed.

        Returns
        -------
        tuple of numpy.ndarray
            ``q_TE = 1/s``, its derivative, ``q_TM = eps_cl/(eps_co s)`` and its
            derivative, with ``s`` the filled-core contrast; derivatives in s/rad.

        Raises
        ------
        ModeError
            If the wall contrast vanishes.
        """
        contrast, dcontrast = self._wall_contrast(terms)
        if np.any(contrast == 0):
            raise ModeError("the wall contrast of a wall-pair guide must be nonzero")
        core, dcore = self._core_permittivity(terms)
        root = np.sqrt(contrast)
        te = 1 / root
        dte = -dcontrast / (2 * contrast * root)
        # Grazing-incidence Fresnel coefficients give q_TE = 1/s and
        # q_TM = eps_cl/(eps_co s): the TM-like pair carries the permittivity
        # ratio, not the wall permittivity alone. For an evacuated core the two
        # spellings coincide, which is why the printed paper shows eps_cl/s.
        tm = terms.wall * te / core
        dtm = (terms.dwall * te + terms.wall * dte) / core - (
            terms.wall * te * dcore / core**2
        )
        return te, dte, tm, dtm

    def _wall_contrast(
        self, terms: _RectTerms
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Return the filled-core wall contrast and its derivative.

        Parameters
        ----------
        terms : _RectTerms
            Local material samples.

        Returns
        -------
        tuple of numpy.ndarray
            Relative permittivity contrast and its frequency slope in s/rad.
        """
        core, dcore = self._core_permittivity(terms)
        return terms.wall - core, terms.dwall - dcore

    def _core_permittivity(
        self, terms: _RectTerms
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Return the filled-core permittivity and its derivative.

        It enters the model twice, in the wall contrast and in the TM-like
        reflection coefficient's permittivity ratio, so both read it here and a
        witness subclass can restore the evacuated-core convention in one place.

        Parameters
        ----------
        terms : _RectTerms
            Local material samples.

        Returns
        -------
        tuple of numpy.ndarray
            Relative core permittivity and its frequency slope in s/rad.
        """
        return 1.0 + terms.chi, terms.dchi


@dataclass(frozen=True, eq=False)
class RectMode(_WallPairFill):
    """Describe one polarisation of a hollow rectangular waveguide mode.

    Attributes
    ----------
    half_width_x, half_width_y : float or ZProfile
        Positive half-widths a and b in m. A shared path object gives a uniform
        scale taper; independently varying paths are not scale invariant.
    gas : Gas or str or None
        Legacy gas declaration; None means vacuum unless a mixture is supplied.
    pressure : float or ZProfile
        Legacy gas pressure in bar, non-negative.
    n, m : int
        Positive mode indices along y and x respectively.
    polarisation : {'x', 'y'}
        Nonzero transverse electric-field component. Its partner is a distinct
        mode with a different index unless the geometry makes them degenerate.
    loss : bool or float or callable
        Shared hollow-mode policy: model loss, no loss, a non-negative model scale,
        or an absolute power attenuation in 1/m.
    cladding : GlassIndex or complex or None
        Wall index, positive imaginary part for absorption. None selects fused silica.
    temperature : float, ZProfile or None
        Positive legacy fill temperature in K, or a path of it along z in m, which
        the density follows. ``None`` (the default) is 293.15 K for a gas and the
        mixture's own beside a mixture; a different one beside it is refused.
    susceptibility : GasSusceptibility or None
        Explicit legacy gas susceptibility; ``None`` takes the built-in gas's.
    mixture : Mixture or None
        Ordered material populations; excludes competing legacy fill arguments.
    width_x, width_y : ZProfile
        Resolved geometry paths in m.
    density : ZProfile
        Total number-density path in m^-3.
    cladding_index : GlassIndex or complex
        Resolved wall material.
    model : {'full', 'reduced'}
        Fixed internally to full, selecting the shared evanescent floor.

    References
    ----------
    K. D. Laakmann and W. H. Steier, Applied Optics 15, 1334-1340 (1976),
    doi:10.1364/AO.15.001334. The filled-core factor ``eps_cl / (eps_co s)`` of the
    TM-like coefficient follows from grazing-incidence Fresnel reflection; it reduces to
    the published ``eps_cl / s`` for an evacuated core.

    Examples
    --------
    >>> mode = RectMode(50e-6, 12.5e-6, "Ar", 1.0)
    >>> mode.principal_component(), mode.orthogonal_polarisation.principal_component()
    ('x', 'y')
    >>> f"{mode.aeff():.6e}"
    '1.111111e-09'
    """

    half_width_x: float | ZProfile
    half_width_y: float | ZProfile
    gas: Gas | str | None = None
    pressure: float | ZProfile = 0.0
    _: KW_ONLY
    n: int = 1
    m: int = 1
    polarisation: Literal["x", "y"] = "x"
    loss: LossSpec = True
    cladding: GlassIndex | complex | None = None
    temperature: float | ZProfile | None = None
    susceptibility: GasSusceptibility | None = None
    mixture: Mixture | None = None
    width_x: ZProfile = field(init=False, repr=False)
    width_y: ZProfile = field(init=False, repr=False)
    density: ZProfile = field(init=False, repr=False)
    cladding_index: GlassIndex | complex = field(init=False, repr=False)
    model: Literal["full", "reduced"] = field(init=False, default="full", repr=False)
    _gain_warned: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the mode and resolve geometry and material paths.

        Raises
        ------
        ModeError
            If half-widths, indices, polarisation or fill declarations are invalid.
        """
        # The once-per-mode validity flag: an attribute rather than a field, so the
        # declared fields stay what they were, and a list so that the shallow copies
        # `at_parameters` makes share it (`_WallPairFill._warn_validity`).
        object.__setattr__(self, "_validity_warned", [False])
        if self.n < 1 or self.m < 1:
            raise ModeError("rectangular mode indices n and m must be positive")
        if self.polarisation not in ("x", "y"):
            raise ModeError("rectangular polarisation must be 'x' or 'y'")
        self._check_fill_arguments()
        object.__setattr__(self, "width_x", as_profile(self.half_width_x))
        object.__setattr__(self, "width_y", as_profile(self.half_width_y))
        self._resolve_fill()
        object.__setattr__(
            self,
            "cladding_index",
            fused_silica() if self.cladding is None else self.cladding,
        )
        for path in (self.width_x, self.width_y):
            positions = np.unique(
                np.concatenate((path.breakpoints, np.asarray(path.z_span), [0.0]))
            )
            for position in positions:
                if math.isfinite(float(position)):
                    self._widths(float(position))
        self._warn_loss_derivative()

    def _widths(self, z: float) -> tuple[float, float]:
        """Return the positive half-widths at one longitudinal position.

        Parameters
        ----------
        z : float
            Position in m.

        Returns
        -------
        tuple of float
            Half-widths a and b in m.

        Raises
        ------
        ModeError
            If either half-width is non-positive or non-finite.
        """
        position = np.asarray([z])
        a, b = float(self.width_x(position)[0]), float(self.width_y(position)[0])
        if not (math.isfinite(a) and math.isfinite(b) and a > 0 and b > 0):
            raise ModeError("rectangular half-widths must be positive and finite")
        return a, b

    @property
    def parameter_paths(self) -> Mapping[str, ZProfile]:
        """Return the geometry and material paths carrying longitudinal variation.

        Returns
        -------
        collections.abc.Mapping
            Half-width paths plus the shared fill's density or pressure paths.
        """
        return MappingProxyType(
            {
                "half_width_x": self.width_x,
                "half_width_y": self.width_y,
                **self._fill_paths(),
            }
        )

    def at_parameters(self, values: Mapping[str, float]) -> "RectMode":  # noqa: UP037 - eager hook
        """Copy the mode with the supplied geometry and fill paths frozen.

        Parameters
        ----------
        values : collections.abc.Mapping
            A subset of the declared paths, in m, m^-3 or shared pressure in bar.

        Returns
        -------
        RectMode
            A copy retaining unspecified paths and the same wall model.

        Raises
        ------
        ModeError
            If a path is unknown or a half-width is invalid.
        """
        unknown = set(values) - set(self.parameter_paths)
        if unknown:
            raise ModeError(f"unknown rectangular parameters: {sorted(unknown)}")
        geometry = {"half_width_x": "width_x", "half_width_y": "width_y"}
        frozen = cast(
            RectMode,
            self._frozen_fill(
                {key: value for key, value in values.items() if key not in geometry}
            ),
        )
        for name, path_name in geometry.items():
            if name in values:
                object.__setattr__(frozen, name, float(values[name]))
                object.__setattr__(frozen, path_name, Constant(float(values[name])))
        frozen._widths(0.0)
        return frozen

    @property
    def z_constant(self) -> bool:
        """Return whether every geometry and material path is constant.

        Returns
        -------
        bool
            True if all declared paths are constant.
        """
        return all(path.is_constant for path in self.parameter_paths.values())

    @property
    def radius_is_constant(self) -> bool:
        """Return the transform's geometry-constant trait for the rectangle.

        Returns
        -------
        bool
            True when both half-width paths are constant; the historical trait
            name also serves Cartesian modes.
        """
        return self.width_x.is_constant and self.width_y.is_constant

    @property
    def scale_invariant(self) -> bool:
        """Return whether one scalar rescaling represents the geometry.

        Returns
        -------
        bool
            True for constant half-widths or the same path object on both axes.
        """
        return self.radius_is_constant or self.width_x is self.width_y

    @property
    def polarisation_degenerate(self) -> bool:
        """Return whether the two polarisations share one effective index.

        Returns
        -------
        bool
            False: the x- and y-polarised modes of one (n, m) take the TE-like
            and TM-like coefficients on opposite wall pairs, so they are
            distinct modes and need a row each.
        """
        return False

    @property
    def azimuthal_order(self) -> int:
        """Return the larger Cartesian order as a geometry-resolution hint.

        Returns
        -------
        int
            Maximum of n and m. Cartesian Gauss rules integrate trigonometric products,
            not polynomials; their required node count is measured, not derived (see
            :func:`solprop.modes.quadrature.cartesian_nodes`).
        """
        return max(self.n, self.m)

    @property
    def channel_label(self) -> ChannelLabel:
        """Return the rectangular family, its two indices and its polarisation.

        Returns
        -------
        ChannelLabel
            Family ``rect``, with ``n`` and ``m`` in the azimuthal and radial fields
            -- a rectangular guide has neither, but they are the two indices the
            record carries -- and the polarisation in the display label. The other
            polarisation is a distinct mode with its own operator row, not a parity
            partner of this one, so ``parity`` stays ``None``.

        Examples
        --------
        >>> from solprop.modes import RectMode
        >>> RectMode(50e-6, 12.5e-6, "Ar", 1.0).channel_label.display_label
        'rect1,1:x'
        """
        return ChannelLabel(
            "rect",
            self.n,
            self.m,
            None,
            None,
            f"rect{self.n},{self.m}:{self.polarisation}",
        )

    def principal_component(self, z: float = 0.0) -> PolarisationComponent:
        """Return the nonzero Cartesian electric-field component.

        Parameters
        ----------
        z : float, optional
            Longitudinal position in m; orientation is fixed.

        Returns
        -------
        str
            The component, x or y.
        """
        self._widths(z)
        return self.polarisation

    @property
    def orthogonal_polarisation(self) -> "RectMode":  # noqa: UP037 - eager hook
        """Return the distinct mode with the other electric-field orientation.

        Returns
        -------
        RectMode
            Same indices, geometry and fill, with the other wall-pair assignment.
            It requires its own operator row; it is not a duplicated scalar index.
        """
        return replace(self, polarisation="y" if self.polarisation == "x" else "x")

    def _model_values(
        self, terms: _RectTerms, a: float, b: float
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Evaluate the complex reflection model before the shared loss policy.

        Parameters
        ----------
        terms : _RectTerms
            Core and wall samples, with holomorphic frequency continuation allowed.
        a, b : float
            Positive half-widths in m.

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
        x, dx = (tm, dtm) if self.polarisation == "x" else (te, dte)
        y, dy = (te, dte) if self.polarisation == "x" else (tm, dtm)
        first, dfirst = _transverse_term(terms.omega, a, self.m, x, dx)
        second, dsecond = _transverse_term(terms.omega, b, self.n, y, dy)
        argument = terms.chi - first - second
        # The rationalised sqrt(1 + argument) - 1 the Marcatili model shares: it
        # keeps every digit in the large-guide limit, where the argument is tiny.
        refractivity = _sqrt1p_minus_one(argument)
        derivative = (terms.dchi - dfirst - dsecond) / (2 * (1 + refractivity))
        if not (np.all(np.isfinite(refractivity)) and np.all(np.isfinite(derivative))):
            raise ModeError("rectangular model produced non-finite values")
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
        a, b = self._widths(z)
        terms = self._terms(omega, z)
        values, derivative = self._model_values(terms, a, b)
        # The x walls see the TM-like reflection when E is along x, normal to them.
        x_wall: Literal["te", "tm"] = "tm" if self.polarisation == "x" else "te"
        y_wall: Literal["te", "tm"] = "te" if x_wall == "tm" else "tm"
        self._warn_validity(terms, ((a, self.m, x_wall), (b, self.n, y_wall)))
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
        """Return refractivity directly from the small square-root argument.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive frequencies in rad/s.
        z : float, optional
            Position in m.

        Returns
        -------
        numpy.ndarray
            Complex n_eff - 1, without subtracting unity from the returned index.
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
            Complex index derivative in s/rad; only a slope-free loss callable
            uses the shared numerical loss derivative.
        """
        return self._values(omega, z)[1]

    def aeff(self, z: float = 0.0) -> float:
        """Return the closed-form effective area.

        Parameters
        ----------
        z : float, optional
            Position in m.

        Returns
        -------
        float
            16 a b / 9 in m², independent of mode index and fill.
        """
        a, b = self._widths(z)
        return 16 / 9 * a * b

    def normalisation(self, z: float = 0.0) -> float:
        """Return the power carried by the unnormalised transverse profile.

        Parameters
        ----------
        z : float, optional
            Position in m.

        Returns
        -------
        float
            Power eps0 c a b / 2 in W.
        """
        a, b = self._widths(z)
        return 0.5 * EPS0 * C * a * b

    def __repr__(self) -> str:
        """Return a one-line description: the mode, the aperture and the fill.

        Returns
        -------
        str
            For example ``RectMode(x11, 200 um x 400 um, Ar 1 bar, loss=True)``.
        """
        entrance = np.array([0.0])
        width = 2.0 * float(self.width_x(entrance)[0])
        height = 2.0 * float(self.width_y(entrance)[0])
        return line(
            "RectMode",
            [
                ("", f"{self.polarisation}{self.n}{self.m}"),
                ("", f"{engineering(width, 'm')} x {engineering(height, 'm')}"),
                ("", self._fill_text()),
                (
                    "loss",
                    self.loss
                    if isinstance(self.loss, (bool, int, float))
                    else "callable",
                ),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        entrance = np.array([0.0])
        cladding = self.cladding_index
        return f"RectMode {self.polarisation}{self.n}{self.m}", [
            (
                "full width in x",
                engineering(2.0 * float(self.width_x(entrance)[0]), "m"),
            ),
            (
                "full width in y",
                engineering(2.0 * float(self.width_y(entrance)[0]), "m"),
            ),
            ("fill", self._fill_text()),
            (
                "temperature",
                _temperature_text(cast("float | ZProfile", self.temperature)),
            ),
            (
                "loss",
                self._loss_text(),
            ),
            ("walls", getattr(cladding, "name", None) or f"constant index {cladding}"),
        ]

    def dimlimits(
        self, z: float = 0.0
    ) -> tuple[CrossSection, tuple[float, float], tuple[float, float]]:
        """Return the Cartesian cross-section and its integration bounds.

        Parameters
        ----------
        z : float, optional
            Position in m.

        Returns
        -------
        tuple
            Cartesian rectangle from (-a, -b) to (a, b), in m.
        """
        a, b = self._widths(z)
        return "cartesian", (-a, -b), (a, b)

    def field(
        self, points: Float[np.ndarray, "n_points 2"], z: float = 0.0
    ) -> Float[np.ndarray, "n_points 2"]:
        """Return the unnormalised real Cartesian electric-field components.

        Parameters
        ----------
        points : numpy.ndarray
            Cartesian (x,y) coordinates in m, shape (n_points, 2).
        z : float, optional
            Position in m.

        Returns
        -------
        numpy.ndarray
            (Ex,Ey), shape (n_points, 2); zero outside the hollow rectangle.

        Raises
        ------
        ModeError
            If points are non-finite or have an invalid shape.
        """
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ModeError(
                "rectangular points must be finite with shape (n_points, 2)"
            )
        a, b = self._widths(z)
        x, y = (
            self.m * math.pi * points[:, 0] / (2 * a),
            self.n * math.pi * points[:, 1] / (2 * b),
        )
        amplitude = (np.cos(x) if self.m % 2 else np.sin(x)) * (
            np.cos(y) if self.n % 2 else np.sin(y)
        )
        inside = (np.abs(points[:, 0]) <= a) & (np.abs(points[:, 1]) <= b)
        amplitude = np.where(inside, amplitude, 0.0)
        zeros = np.zeros_like(amplitude)
        return np.column_stack(
            (amplitude, zeros) if self.polarisation == "x" else (zeros, amplitude)
        )
