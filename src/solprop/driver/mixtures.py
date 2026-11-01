# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Resolve material declarations and population selections at the driver boundary."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from jaxtyping import Float

from solprop.constants import ROOM_TEMPERATURE
from solprop.exceptions import DriverError
from solprop.gasflow import VacuumEnd
from solprop.nonlinear.ionisation import IntegrationPolicy, IonisationRate
from solprop.nonlinear.mixtures import responses_for
from solprop.nonlinear.optmat_rates import bind_reference_field
from solprop.nonlinear.raman import MolecularOptions
from solprop.physdata import Gas, Mixture, species
from solprop.physdata.mixtures import Pressure
from solprop.physdata.selection import MaterialSelection
from solprop.physdata.sellmeier import GasSusceptibility
from solprop.profiles import (
    FromCallable,
    Sum,
    ZProfile,
    _checked_temperature,
    _same_temperature,
    _temperature_text,
)

__all__ = [
    "ResponseSelection",
    "capillary_fill",
    "is_air",
    "mixture_profiles",
]


@dataclass(frozen=True)
class ResponseSelection:
    """Retain the public response choices when updating a capillary fill.

    Attributes
    ----------
    kerr : bool
        Whether to include instantaneous electronic responses.
    thg : bool
        Whether to retain third harmonics on the field grid.
    plasma : bool, IonisationRate or Mapping
        Global or per-label plasma selection.
    raman : bool, sequence of str or Mapping
        Global or per-label Raman term selection.
    wavelength : float
        Reference wavelength in m, the carrier of the default rates.
    envelope : bool
        Preserve the envelope plasma selection when replacing a fill.
    length : float or None
        The run's length in m, over which a temperature path's Raman reference
        temperature is averaged.
    selection : MaterialSelection or None
        The run's material selection, under which an optmat-resolved fill takes its
        Raman response from optmat and ``plasma=True`` takes optmat's rates.
    reference_field : float or None
        The input's peak field in V/m, at which an optmat rate's constant energy
        per event is taken; ``None`` leaves the rates unbound.
    field_range : tuple of float or None
        The field amplitudes in V/m the run tabulates its rates over.
    plasma_integration : {'band', 'trapezoid'} or None
        The field plasma column's running-integral rule (SEP 18); ``None`` takes the
        response's default.
    """

    kerr: bool
    thg: bool
    plasma: bool | IonisationRate | Mapping[str, bool | IonisationRate]
    raman: bool | Sequence[str] | Mapping[str, bool | MolecularOptions]
    wavelength: float
    envelope: bool = False
    length: float | None = None
    selection: MaterialSelection | None = None
    reference_field: float | None = None
    field_range: tuple[float, float] | None = None
    plasma_integration: IntegrationPolicy | None = None

    def build(self, mixture: Mixture) -> tuple[object, ...]:
        """Build the selected responses using the updated component declarations.

        Parameters
        ----------
        mixture : Mixture
            Ordered material, temperature and initial populations.

        Returns
        -------
        tuple
            Response declarations in mixture order, an optmat rate bound to the
            run's reference field.
        """
        responses = responses_for(
            mixture,
            kerr=self.kerr,
            thg=self.thg,
            plasma=self.plasma,
            raman=self.raman,
            wavelength=self.wavelength,
            envelope=self.envelope,
            length=self.length,
            selection=self.selection,
            plasma_integration=self.plasma_integration,
        )
        if self.reference_field is None:
            return responses
        return bind_reference_field(responses, self.reference_field, self.field_range)


def capillary_fill(  # noqa: PLR0913 - the public constructor's material options
    material: Gas | str | Mapping[str, float] | Mixture,
    pressure: Pressure | None,
    *,
    length: float,
    temperature: float | ZProfile | None,
    floor: float | None,
    susceptibility: GasSusceptibility | None,
    preionisation: float | ZProfile | Mapping[str, float | ZProfile] | None,
    radius: float | ZProfile | None = None,
    vacuum_end: VacuumEnd | None = None,
    selection: MaterialSelection | None = None,
) -> Mixture:
    """Resolve one material declaration and explicit initial populations.

    Parameters
    ----------
    material : Gas, str, Mapping or Mixture
        A built-in gas, mole fractions or a complete mixture declaration.
    pressure : float, sequence, ZProfile or None
        Total pressure in bar; None uses one bar for a built-in gas or fractions.
        A supplied Mixture requires None because it already carries its paths.
    length : float
        Capillary length in m, needed by endpoint pressure pairs.
    temperature : float, ZProfile or None
        Fill temperature in K, or a path of it along z in m; ``None`` is 293.15 K for a
        new fill. An existing mixture carries its own temperature: beside it this
        must be ``None`` or the mixture's own (the same number or path object).
    floor : float or None
        Pressure floor in bar for a new fill; incompatible with an existing mixture.
    susceptibility : GasSusceptibility or None
        Optical override for a single built-in gas only.
    preionisation : float, ZProfile, Mapping or None
        Initial ionised fractions in [0, 1], globally or per label, each a number or a
        path along z in m. None retains the mixture's declarations; omitted mapping
        labels retain their declarations.
    radius : float, ZProfile or None, optional
        Core radius in m of a circular hollow guide. Given, a pressure pair or
        (z, p) pair becomes the steady flow of :func:`solprop.gasflow.steady_fill`,
        which follows the bore and chokes a vacuum end; ``None`` (a rectangular
        guide, or a mode set without a core radius) keeps the uniform law and the
        numerical floor. A constant pressure or a supplied profile does not take it.
    vacuum_end : {'choked', 'floor'} or None, optional
        How an outer end below its choked pressure is treated; ``None`` chokes it
        where `radius` is given. Refused beside an existing mixture, which carries
        its own paths.
    selection : MaterialSelection or None, optional
        Resolve a built-in gas's name, a fraction mapping's names and ``"air"`` from
        optmat under this selection; ``None`` is ``MaterialSelection()``. An
        existing mixture is taken as it is.

    Returns
    -------
    Mixture
        Ordered species, density paths and initial fractions.

    Raises
    ------
    DriverError
        If pressure, temperature or optical arguments conflict, or a label is
        unknown.
    """
    if isinstance(material, Mixture):
        if temperature is not None and not _same_temperature(
            _checked_temperature(temperature, "the run"), material.temperature
        ):
            raise DriverError(
                "the Mixture carries its own temperature "
                f"({_temperature_text(material.temperature)}), and "
                f"temperature={temperature!r} was given beside it: omit temperature=, "
                "or build the Mixture at the temperature you mean",
                parameter=("gas", "temperature"),
            )
        if pressure is not None or floor is not None or vacuum_end is not None:
            given = {
                "pressure": pressure,
                "pressure_floor": floor,
                "vacuum_end": vacuum_end,
            }
            raise DriverError(
                "Mixture carries its pressure paths; omit pressure, pressure_floor "
                "and vacuum_end (a mixture of independent partial pressures is not "
                "one flow, and a shared fill was built with its own ends)",
                parameter=(
                    "gas",
                    *(name for name, value in given.items() if value is not None),
                ),
            )
        if susceptibility is not None:
            raise DriverError(
                "Mixture carries species susceptibilities; omit susceptibility",
                parameter=("gas", "susceptibility"),
            )
        mixture = material
    else:
        temperature = ROOM_TEMPERATURE if temperature is None else temperature
        # The bore shapes only a fill built here from end pressures; a supplied
        # profile is used as given, and the constructor warns if it is boreless.
        bore = radius if isinstance(pressure, Sequence) else None
        if isinstance(pressure, Sequence) and len(pressure) != 2:
            raise DriverError(
                "pressure must be a number, profile, (p_in, p_out) or (z, p) pair",
                parameter="pressure",
            )
        if isinstance(material, Mapping):
            if susceptibility is not None:
                raise DriverError(
                    "fraction mappings carry species optics; omit susceptibility",
                    parameter=("gas", "susceptibility"),
                )
            fractions = {
                species(name, selection=selection): fraction
                for name, fraction in material.items()
            }
        elif is_air(material):
            # The convenience the interface promises: "air" is a nitrogen/oxygen
            # mixture with separate channels, not the dispersion-only Gas.Air entry.
            if susceptibility is not None:
                raise DriverError(
                    '"air" is a mixture of nitrogen and oxygen and carries their '
                    "susceptibilities; omit susceptibility, or build the Mixture "
                    "yourself",
                    parameter=("gas", "susceptibility"),
                )
            mixture = Mixture.air(
                1.0 if pressure is None else pressure,
                temperature=temperature,
                argon=False,
                length=length,
                floor=floor,
                radius=bore,
                vacuum_end=vacuum_end,
                selection=selection,
            )
            return _with_initial_populations(mixture, preionisation)
        else:
            selected = species(material, selection=selection)
            if susceptibility is not None:
                selected = replace(selected, susceptibility=susceptibility)
            fractions = {selected: 1.0}
        mixture = Mixture.fill(
            fractions,
            1.0 if pressure is None else pressure,
            temperature=temperature,
            length=length,
            floor=floor,
            radius=bore,
            vacuum_end=vacuum_end,
        )
    return _with_initial_populations(mixture, preionisation)


def is_air(material: Gas | str) -> bool:
    """Return whether a built-in gas's name means air.

    Parameters
    ----------
    material : Gas or str
        The name given to the interface.

    Returns
    -------
    bool
        ``True`` for ``Gas.Air`` and for ``"air"`` in any case.
    """
    if isinstance(material, Gas):
        return material is Gas.Air
    return material.strip().lower() == "air"


def _with_initial_populations(
    mixture: Mixture,
    preionisation: float | ZProfile | Mapping[str, float | ZProfile] | None,
) -> Mixture:
    """Retain unspecified populations when overriding initial ionised fractions.

    Parameters
    ----------
    mixture : Mixture
        Existing populations.
    preionisation : float, ZProfile, Mapping or None
        Global or labelled initial fractions in [0, 1], each a number or a path along
        z in m.

    Returns
    -------
    Mixture
        Updated populations with every unspecified fraction retained.
    """
    if preionisation is None:
        return mixture
    if isinstance(preionisation, Mapping):
        unknown = set(preionisation) - set(mixture.labels)
        if unknown:
            paths = tuple(f'preionisation["{label}"]' for label in sorted(unknown))
            raise DriverError(
                f"unknown preionisation component labels: {sorted(unknown)}",
                parameter=paths[0] if len(paths) == 1 else paths,
            )
        values = preionisation
    else:
        values = dict.fromkeys(mixture.labels, preionisation)
    return replace(
        mixture,
        components=tuple(
            replace(
                component,
                preionisation=values.get(component.label, component.preionisation),
            )
            for component in mixture.components
        ),
    )


def mixture_profiles(
    mixture: Mixture, *, pressure: bool = False
) -> dict[str, ZProfile]:
    """Return the total density and available total-pressure statistic paths.

    Parameters
    ----------
    mixture : Mixture
        Ordered material populations.
    pressure : bool, optional
        For independent densities, include the sum of partial pressures obtained
        by inverting each species' equation of state at the declared temperature
        (at each position's own, for a temperature path).
        Used to preserve an existing pressure statistic after a material update.

    Returns
    -------
    dict
        Density in m^-3 and, for a shared fill, pressure in bar.
    """
    paths = tuple(component.density for component in mixture.components)
    result = {"density": paths[0] if len(paths) == 1 else Sum(paths)}
    if mixture.shared is not None:
        result["pressure"] = mixture.shared
    elif pressure:
        total = result["density"]
        result["pressure"] = FromCallable(
            _PartialPressureSum(mixture), total.z_span, total.breakpoints
        )
    return result


@dataclass(frozen=True)
class _PartialPressureSum:
    """Evaluate the declared partial-EOS pressure diagnostic, in bar."""

    mixture: Mixture

    def __call__(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Sum the EOS-inverted partial pressures at the requested positions.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m, shape (n_z,).

        Returns
        -------
        numpy.ndarray
            Total pressure diagnostic in bar, shape (n_z,).
        """
        temperature = self.mixture.temperature
        kelvin = (
            [float(value) for value in temperature(z)]
            if isinstance(temperature, ZProfile)
            else [temperature] * np.size(z)
        )
        return np.sum(
            [
                [
                    component.species.equation_of_state.pressure(float(value), local)
                    for value, local in zip(component.density(z), kelvin, strict=True)
                ]
                for component in self.mixture.components
            ],
            axis=0,
        )
