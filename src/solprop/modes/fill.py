# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Share gas-fill resolution, freezing and loss policy across hollow-waveguide modes.

This class contributes methods only. Concrete frozen dataclasses declare their
own fields, preserving the public Marcatili constructor and its arithmetic.
"""

import copy
import logging
from collections.abc import Callable, Mapping
from dataclasses import is_dataclass, replace
from types import MappingProxyType
from typing import Final, Literal, cast

import numpy as np
from jaxtyping import Complex, Float, Inexact

from solprop._display import Displayed
from solprop.constants import ROOM_TEMPERATURE, C
from solprop.exceptions import MaterialError, ModeError
from solprop.physdata import _optmat
from solprop.physdata.gases import Gas
from solprop.physdata.mixtures import (
    Mixture,
    _check_flow_weight,
    _dilute_scaling_checks,
    _warn_dilute_scaling,
)
from solprop.physdata.selection import DiluteScalingCheck
from solprop.physdata.sellmeier import GasSusceptibility, OptmatSusceptibility
from solprop.physdata.species import Species, default_species, fill_species
from solprop.profiles import (
    Constant,
    DensityPath,
    Sum,
    ZProfile,
    _checked_temperature,
    _same_temperature,
    _temperature_text,
    as_profile,
)

__all__ = ["GasFillMixin"]

logger = logging.getLogger(__name__)

_LossCurve = Callable[[Float[np.ndarray, " n_omega"]], Float[np.ndarray, " n_omega"]]

LossSpec = bool | float | _LossCurve
"""Specify the model loss of any mode that mixes in :class:`GasFillMixin`.

The vocabulary is documented on :class:`~solprop.modes.capillary.MarcatiliMode`,
which shares it with :class:`~solprop.modes.rectangular.RectMode`.
"""

EVANESCENT_INDEX_FLOOR: Final = 1e-3
"""Smallest real effective index the full model will return.

Far below the band the square root turns evanescent and ``Re n_eff`` runs to zero, which
makes the mode-averaged prefactor ``1 / n_eff`` blow up. The real part is floored at
1e-3, and the derivative is reported as zero where the floor binds because the floored
value is constant. The reduced model has no floor.
"""


def _sqrt1p_minus_one(
    argument: Inexact[np.ndarray, " n_omega"],
) -> Complex[np.ndarray, " n_omega"]:
    """Return ``sqrt(1 + argument) - 1`` in the exactly rationalised form.

    ``argument / (sqrt(1 + argument) + 1)`` keeps the small ``argument`` in the
    numerator over a denominator near two, so the result carries the relative rounding
    of ``argument`` rather than the absolute rounding of unity. The spelling
    ``expm1(log1p(x)/2)`` is not equivalent for a complex argument: NumPy's complex
    ``log1p`` forms ``1 + x`` and rounds there (1.3e-12 relative at ``1e-4 + 1e-7j``
    on NumPy 2.5).
    Every gas-filled guide forms its refractivity here.

    Parameters
    ----------
    argument : numpy.ndarray
        ``n_eff**2 - 1``, real or complex, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        ``sqrt(1 + argument) - 1``, complex, shape ``(n_omega,)``, on the principal
        branch.
    """
    value = np.asarray(argument, dtype=complex)
    return np.asarray(value / (np.sqrt(1.0 + value) + 1.0))


_FD_RELATIVE_STEP: Final = 1e-5
"""Relative step of the Richardson difference used for a loss curve with no slope."""


class GasFillMixin(Displayed):
    """Provide material and loss methods for frozen hollow-mode dataclasses.

    Subclasses declare gas, pressure, temperature, susceptibility, mixture,
    density and loss fields. The model attribute selects the evanescent floor:
    ``"full"`` applies it and ``"reduced"`` does not. No dataclass fields or
    constructor are supplied here.
    """

    gas: Gas | str | None
    pressure: float | ZProfile
    temperature: float | ZProfile | None
    susceptibility: GasSusceptibility | None
    mixture: Mixture | None
    density: ZProfile
    loss: LossSpec
    model: Literal["full", "reduced"]
    _gain_warned: bool

    def _loss_text(self) -> object:
        """Return the loss specification for printing.

        Returns
        -------
        object
            The flag or the factor as it was given, or ``"callable"`` for a loss curve.
        """
        return self.loss if isinstance(self.loss, (bool, int, float)) else "callable"

    def _fill_text(self) -> str:
        """Return the fill as a person would say it, for printing.

        Returns
        -------
        str
            ``"evacuated"``, ``"He 1 bar"`` or, for a mixture, its labels and its
            shared pressure; a gradient is quoted at ``z = 0``.
        """
        entrance = np.array([0.0])
        if self.mixture is not None:
            labels = "+".join(self.mixture.labels)
            shared = self.mixture.shared
            if shared is None:
                return f"{labels} (independent populations)"
            return f"{labels} {float(shared(entrance)[0]):g} bar"
        if self.gas is None:
            return "evacuated"
        return f"{self.gas} {float(as_profile(self.pressure)(entrance)[0]):g} bar"

    def _check_fill_arguments(self) -> None:
        """Check the fill declaration and loss specification.

        Raises
        ------
        ModeError
            If temperature, loss or a competing mixture declaration is invalid.
        """
        if self.temperature is None:
            # Not given: room temperature for a gas, the mixture's own beside one.
            temperature = (
                ROOM_TEMPERATURE if self.mixture is None else self.mixture.temperature
            )
        else:
            try:
                temperature = _checked_temperature(self.temperature, "the fill")
            except MaterialError as error:
                msg = f"the temperature must be positive and finite: {error}"
                raise ModeError(msg) from error
        if self.mixture is not None and not _same_temperature(
            temperature, self.mixture.temperature
        ):
            msg = (
                f"the mixture carries its own temperature "
                f"({_temperature_text(self.mixture.temperature)}), and "
                f"temperature={_temperature_text(temperature)} was given beside it: "
                "omit temperature=, or build the Mixture at the temperature you mean"
            )
            raise ModeError(msg)
        # A constant path is stored as its number, so the mode takes the
        # constant-temperature arithmetic bit for bit.
        object.__setattr__(self, "temperature", temperature)
        if not isinstance(self.loss, bool):
            if isinstance(self.loss, (float, int)):
                if self.loss < 0.0:
                    raise ModeError(
                        f"a loss scale must not be negative, got {self.loss}"
                    )
            elif not callable(self.loss):
                raise ModeError(
                    "loss must be a bool, a non-negative scale or a callable "
                    f"alpha(omega), got {type(self.loss).__name__}"
                )

        if self.mixture is not None and (
            self.gas is not None
            or self.pressure != 0.0
            or self.susceptibility is not None
        ):
            raise ModeError(
                "mixture carries its gas, pressure and susceptibility; "
                "omit those arguments"
            )

    def _resolve_fill(self) -> None:
        """Resolve the susceptibility and density paths declared by the subclass.

        A gas named is the built-in gas, optmat's
        (:func:`~solprop.physdata.species.default_species`): its susceptibility,
        unless one is given, and its equation of state for the density. The fill is
        carried as that single species in the density path: with an optmat law of
        the same gas given (a selection's, SEP 3, Q27), the species takes it, so the
        species the fill records is the one whose index acts, and its dilute-scaling
        check is measured on it.
        """
        if self.susceptibility is None and self.gas is not None:
            object.__setattr__(
                self, "susceptibility", default_species(self.gas).susceptibility
            )
        pressure = as_profile(self.pressure)
        temperature = cast("float | ZProfile", self.temperature)
        # The gas whose equation of state, and whose flow weight, the fill takes.
        material: Species | None = (
            None if self.gas is None else fill_species(self.gas, self.susceptibility)
        )
        if material is not None:
            try:
                _check_flow_weight(
                    pressure, temperature, material, f"the {self.gas} fill"
                )
            except MaterialError as error:
                raise ModeError(str(error)) from error
        object.__setattr__(
            self,
            "density",
            Constant(0.0, pressure.z_span)
            if material is None
            else DensityPath(material, pressure, temperature),
        )
        if self.mixture is not None:
            paths = tuple(c.density for c in self.mixture.components)
            object.__setattr__(
                self, "density", paths[0] if len(paths) == 1 else Sum(paths)
            )

    def _warn_loss_derivative(self) -> None:
        """Report when a callable loss requires a numerical derivative."""
        if not isinstance(self.loss, (bool, int, float)) and not hasattr(
            self.loss, "derivative"
        ):
            logger.warning(
                "the loss callable of %r carries no 'derivative' attribute, so the "
                "imaginary part of dneff_domega is a Richardson difference with a "
                "relative floor near 1e-10; the real part, which sets beta_1, stays "
                "analytic",
                self,
            )

    def absorption_by_species(
        self, omega: Float[np.ndarray, " n_omega"], z: float = 0.0
    ) -> Float[np.ndarray, "n_species n_omega"]:
        """Attribute material intensity loss to populations to first order in chi.

        Return ``rho_s(z) omega Im(gamma_s(omega)) / c`` in mixture order.
        This dilute-material attribution excludes wall leakage and the higher-order
        square-root and waveguide corrections in the mode's total attenuation.
        ``loss=False`` removes it; a numeric loss scales it. A callable replaces
        total loss and therefore supplies no species attribution.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s, within each species' support.
        z : float, optional
            Position in m at which to evaluate the component densities.

        Returns
        -------
        numpy.ndarray
            Non-negative intensity absorption in m^-1, shape
            ``(n_species, n_omega)``. A legacy gas or vacuum has one row.

        Raises
        ------
        ModeError
            If omega is non-positive or non-finite, or loss is a callable whose
            total attenuation cannot be assigned to individual species.
        """
        if np.any(omega <= 0) or not np.all(np.isfinite(omega)):
            raise ModeError("species absorption requires positive finite omega")
        if callable(self.loss):
            raise ModeError(
                "callable loss replaces total loss without species attribution"
            )
        scale = float(self.loss)
        position = np.asarray([z])
        if self.mixture is not None:
            result = np.zeros((len(self.mixture.components), omega.size))
            for index, component in enumerate(self.mixture.components):
                model = component.species.susceptibility
                density = float(component.density(position)[0])
                # A transparent law is real by construction; its zero
                # absorption does not require extrapolating a refractivity fit.
                if (
                    scale == 0
                    or density == 0
                    or (isinstance(model, GasSusceptibility) and not model.absorbing)
                ):
                    continue
                result[index] = (
                    scale * density * omega * np.imag(model.susceptibility(omega)) / C
                )
            return result
        if self.susceptibility is None:
            return np.zeros((1, omega.size))
        # The legacy gas slot carries a real GasSusceptibility. Absorbing supplied
        # optical tables belong to explicit Species components of a mixture.
        return np.zeros((1, omega.size))

    def _fill_paths(self) -> Mapping[str, ZProfile]:
        """Return the fill's scalar paths, including constant declarations.

        Returns
        -------
        collections.abc.Mapping
            Density, shared pressure or independently labelled density paths. A
            shared fill whose temperature varies along z is declared by its labelled
            densities: one pressure does not fix them, so it takes
            one parameter per population.
        """
        paths: dict[str, ZProfile] = {}
        if self.mixture is None or len(self.mixture.components) == 1:
            paths["density"] = self.density
        elif self.mixture.shared is not None and not isinstance(
            self.mixture.temperature, ZProfile
        ):
            paths["pressure"] = self.mixture.shared
        else:
            paths.update(
                {f"density:{c.label}": c.density for c in self.mixture.components}
            )
        return MappingProxyType(paths)

    def _frozen_fill(self, values: Mapping[str, float]) -> "GasFillMixin":  # noqa: UP037 - the hook resolves class names eagerly
        """Copy the mode with its named material paths frozen.

        Parameters
        ----------
        values : collections.abc.Mapping
            Fill parameter values in SI density or shared pressure in bar.

        Returns
        -------
        GasFillMixin
            A shallow copy sharing the frequency-only model quantities.

        Raises
        ------
        ModeError
            If a supplied density is negative.
        """
        frozen = copy.copy(self)
        if "density" in values:
            number_density = float(values["density"])
            if number_density < 0.0:
                msg = f"the number density must not be negative, got {number_density}"
                raise ModeError(msg)
            object.__setattr__(frozen, "density", Constant(number_density))
            if self.mixture is not None:
                object.__setattr__(
                    frozen,
                    "mixture",
                    self.mixture.with_densities(
                        {self.mixture.labels[0]: number_density}
                    ),
                )
        mixture_values = {
            name: value for name, value in values.items() if name != "density"
        }
        if self.mixture is not None and mixture_values:
            mixture = self.mixture.at_parameters(mixture_values)
            object.__setattr__(frozen, "mixture", mixture)
            object.__setattr__(
                frozen, "density", Sum(tuple(c.density for c in mixture.components))
            )
        return frozen

    def _sampled_susceptibility(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> tuple[Inexact[np.ndarray, "*shape"], Inexact[np.ndarray, "*shape"]]:
        """Sample the material susceptibility and its frequency derivative.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s, already checked by the mode.

        Returns
        -------
        tuple of numpy.ndarray
            Susceptibility in m³ and its derivative in m³ s/rad, one spectral row
            for a legacy gas and one row per population for an explicit mixture.

        Warns
        -----
        DiluteScalingWarning
            Once per gas, where an optmat-selected population's linear density
            scaling departs from optmat's finite-density law beyond its bound.
        """
        # One evaluation context: each material's range warning is raised once for
        # the susceptibility and its derivative together.
        with _optmat.material_context():
            sampled = self._sample_fill(omega)
            _warn_dilute_scaling(self._dilute_checks(omega))
        return sampled

    def _dilute_checks(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> tuple[DiluteScalingCheck, ...]:
        """Measure the dilute-scaling error of the fill's optmat-selected gases.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s of the sampled band.

        Returns
        -------
        tuple of DiluteScalingCheck
            One per optmat population (or the optmat law of a single gas) with a
            non-zero density, at its largest density along the fill.
        """
        if self.mixture is not None:
            return self.mixture.dilute_scaling_checks(omega)
        if (
            not isinstance(self.susceptibility, OptmatSusceptibility)
            or self.gas is None
        ):
            return ()
        model = self.susceptibility
        gas = Species(model.gas.value, model)
        temperature = cast("float | ZProfile", self.temperature)
        return _dilute_scaling_checks(
            [(gas.name, gas, self.density)], temperature, omega
        )

    @property
    def fill_species(self) -> Species | None:
        """Return the single species a fill without a mixture carries.

        Returns
        -------
        Species or None
            The species of the density path, whose equation of state gives the
            density and, for an optmat law, whose susceptibility is the fill's;
            ``None`` for vacuum or a mixture.
        """
        if self.mixture is not None or not isinstance(self.density, DensityPath):
            return None
        material = self.density.gas
        return material if isinstance(material, Species) else None

    def _sample_fill(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> tuple[Inexact[np.ndarray, "*shape"], Inexact[np.ndarray, "*shape"]]:
        """Sample the fill's susceptibility rows and their derivatives.

        Parameters
        ----------
        omega : numpy.ndarray
            Positive angular frequencies in rad/s, already checked by the mode.

        Returns
        -------
        tuple of numpy.ndarray
            As :meth:`_sampled_susceptibility`.
        """
        values = omega
        if self.mixture is not None:
            # A population that is identically absent needs no spectral data. Keep
            # its row so labels and density parameters retain their declared order.
            absent = self._absent_components()
            susceptibility = np.stack(
                [
                    np.zeros_like(values)
                    if zero
                    else c.species.susceptibility.susceptibility(values)
                    for c, zero in zip(self.mixture.components, absent, strict=True)
                ]
            )
            dsusceptibility = np.stack(
                [
                    np.zeros_like(values)
                    if zero
                    else c.species.susceptibility.dsusceptibility_domega(values)
                    for c, zero in zip(self.mixture.components, absent, strict=True)
                ]
            )
        elif self.susceptibility is None:
            susceptibility = np.zeros_like(values)
            dsusceptibility = np.zeros_like(values)
        else:
            susceptibility = np.asarray(self.susceptibility.susceptibility(values))
            dsusceptibility = np.asarray(
                self.susceptibility.dsusceptibility_domega(values)
            )
        return susceptibility, dsusceptibility

    def _absent_components(self) -> tuple[bool, ...]:
        """Return which mixture populations are identically absent.

        Returns
        -------
        tuple of bool
            Per component of the mixture, whether its density is the constant zero.
            Only a fill with a mixture asks.
        """
        assert self.mixture is not None
        return tuple(
            c.density.is_constant and float(c.density(np.array([0.0]))[0]) == 0.0
            for c in self.mixture.components
        )

    def _fill_rows(self) -> tuple[object, ...]:
        """Return what the sampled susceptibility rows are a function of, besides omega.

        Two fills with equal results here sample the same rows: the same models, in the
        same order, with the same populations replaced by zeros (:meth:`_sample_fill`).

        Returns
        -------
        tuple
            The susceptibility model of a fill without a mixture; otherwise each
            component's model and whether it is absent.
        """
        if self.mixture is None:
            return (self.susceptibility,)
        return tuple(
            zip(
                (c.species.susceptibility for c in self.mixture.components),
                self._absent_components(),
                strict=True,
            )
        )

    def _apply_loss(
        self,
        omega: Float[np.ndarray, " n_omega"],
        refractivity: Complex[np.ndarray, " n_omega"],
        derivative: Complex[np.ndarray, " n_omega"],
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Apply the loss specification to a model refractivity and its derivative.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.
        refractivity : numpy.ndarray
            ``n_eff - 1`` with the model's own loss.
        derivative : numpy.ndarray
            ``dn_eff/domega`` with the model's own loss.

        Returns
        -------
        tuple of numpy.ndarray
            The refractivity and derivative with the requested loss.
        """
        specification = self.loss
        if isinstance(specification, bool):
            if specification:
                return self._no_gain(refractivity, derivative)
            return (
                np.asarray(np.real(refractivity), dtype=complex),
                np.asarray(np.real(derivative), dtype=complex),
            )
        if isinstance(specification, (int, float)):
            scale = float(specification)
            return self._no_gain(
                np.asarray(np.real(refractivity) + 1j * scale * np.imag(refractivity)),
                np.asarray(np.real(derivative) + 1j * scale * np.imag(derivative)),
            )
        # An absolute power loss coefficient: alpha = 2 omega Im(n) / c, so
        # Im n = c alpha / (2 omega) and its derivative follows by the quotient rule.
        # Only the imaginary part is replaced; beta and beta_1 stay analytic.
        alpha = np.asarray(specification(omega), dtype=float)
        dalpha = self._loss_derivative(specification, omega)
        extinction = 0.5 * C * alpha / omega
        dextinction = 0.5 * C * (dalpha * omega - alpha) / omega**2
        return (
            np.asarray(np.real(refractivity) + 1j * extinction),
            np.asarray(np.real(derivative) + 1j * dextinction),
        )

    def _no_gain(
        self,
        refractivity: Complex[np.ndarray, " n_omega"],
        derivative: Complex[np.ndarray, " n_omega"],
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Clamp the model's own loss at zero: a passive waveguide cannot amplify.

        With a cladding index below unity -- a metal, or silica in the extreme
        ultraviolet, outside anything Marcatili's expansion describes -- the complex
        square root can land on the branch where ``Im n_eff < 0``, which reads as gain.
        It is clamped here on every path, the derivative is zeroed where the clamp
        binds, and the mode logs the fact once.

        Parameters
        ----------
        refractivity : numpy.ndarray
            ``n_eff - 1`` with the model's loss.
        derivative : numpy.ndarray
            ``dn_eff/domega`` with the model's loss.

        Returns
        -------
        tuple of numpy.ndarray
            The same, with a non-negative imaginary part.
        """
        gain = np.imag(refractivity) < 0.0
        if not np.any(gain):
            return refractivity, derivative
        if not self._gain_warned:
            logger.warning(
                "%s: the model's loss is negative (gain) at %d of %d frequencies, "
                "which happens when the cladding index falls below unity, outside "
                "the model's validity; the loss is clamped at zero there",
                self,
                int(np.count_nonzero(gain)),
                gain.size,
            )
            object.__setattr__(self, "_gain_warned", True)
        return (
            np.asarray(
                np.where(gain, np.real(refractivity), refractivity), dtype=complex
            ),
            np.asarray(np.where(gain, np.real(derivative), derivative), dtype=complex),
        )

    def _loss_derivative(
        self, curve: _LossCurve, omega: Float[np.ndarray, " n_omega"]
    ) -> Float[np.ndarray, " n_omega"]:
        """Return the frequency derivative of a loss curve supplied as a callable.

        Parameters
        ----------
        curve : callable
            The loss curve ``alpha(omega)`` in 1/m.
        omega : numpy.ndarray
            Angular frequencies in rad/s.

        Returns
        -------
        numpy.ndarray
            ``d(alpha)/d(omega)`` in 1/(m rad/s), analytic when the callable carries a
            ``derivative`` attribute and otherwise a Richardson-extrapolated central
            difference with a relative floor near 1e-10.
        """
        analytic = getattr(curve, "derivative", None)
        if callable(analytic):
            return np.asarray(analytic(omega), dtype=float)
        step = _FD_RELATIVE_STEP * omega
        wide = (np.asarray(curve(omega + step)) - np.asarray(curve(omega - step))) / (
            2.0 * step
        )
        half = step / 2.0
        narrow = (np.asarray(curve(omega + half)) - np.asarray(curve(omega - half))) / (
            2.0 * half
        )
        return np.asarray((4.0 * narrow - wide) / 3.0, dtype=float)

    def _floored(
        self,
        refractivity: Complex[np.ndarray, " n_omega"],
        derivative: Complex[np.ndarray, " n_omega"],
    ) -> tuple[Complex[np.ndarray, " n_omega"], Complex[np.ndarray, " n_omega"]]:
        """Apply the full model's evanescent floor to a refractivity and derivative.

        Parameters
        ----------
        refractivity : numpy.ndarray
            ``n_eff - 1``.
        derivative : numpy.ndarray
            ``dn_eff/domega``.

        Returns
        -------
        tuple of numpy.ndarray
            The same, with ``Re n_eff`` floored at :data:`EVANESCENT_INDEX_FLOOR` and
            the derivative set to zero where the floor binds.
        """
        if self.model != "full":
            return refractivity, derivative
        index = 1.0 + refractivity
        binds = np.real(index) < EVANESCENT_INDEX_FLOOR
        if not np.any(binds):
            return refractivity, derivative
        extinction = np.maximum(np.imag(index), 0.0) if self.loss is not False else 0.0
        floored = np.where(
            binds, EVANESCENT_INDEX_FLOOR + 1j * extinction - 1.0, refractivity
        )
        return (
            np.asarray(floored, dtype=complex),
            np.asarray(np.where(binds, 0.0, derivative), dtype=complex),
        )

    def with_mixture(self, mixture: Mixture) -> "GasFillMixin":  # noqa: UP037 - the hook resolves class names eagerly
        """Return the same mode geometry with the supplied ordered gas mixture.

        Parameters
        ----------
        mixture : solprop.physdata.mixtures.Mixture
            New populations, density paths and temperature: the mode's gas,
            pressure and temperature are replaced by the mixture's.

        Returns
        -------
        GasFillMixin
            A new instance of the same concrete dataclass.

        Raises
        ------
        ModeError
            If the concrete mode is not a dataclass.
        """
        if not is_dataclass(self):
            raise ModeError("GasFillMixin requires a concrete mode dataclass")
        return replace(
            self,
            gas=None,
            pressure=0.0,
            temperature=None,
            susceptibility=None,
            mixture=mixture,
        )
