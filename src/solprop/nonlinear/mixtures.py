# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Select nonlinear responses for the labelled populations of a mixture."""

import itertools
import logging
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import numpy as np
from jaxtyping import Float
from scipy.integrate import quad

from solprop.exceptions import MaterialError, RamanTemperatureWarning
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.ionisation import IntegrationPolicy, IonisationRate, rate_species
from solprop.nonlinear.optmat_raman import mixture_raman
from solprop.nonlinear.optmat_rates import selected_rate
from solprop.nonlinear.raman import (
    DampedOscillator,
    RamanResponse,
    RotationalLines,
    response_for,
)
from solprop.nonlinear.responses import (
    DispersiveKerrResponse,
    KerrResponse,
    PlasmaResponse,
)
from solprop.physdata.mixtures import Component, Mixture
from solprop.physdata.raman_types import MolecularOptions
from solprop.physdata.selection import DISPERSIVE_POLICIES, MaterialSelection
from solprop.physdata.species import Species, built_in_population
from solprop.profiles import Sum, ZProfile, _probe_positions

__all__ = ["electronic_response", "raman_temperature", "responses_for"]

logger = logging.getLogger(__name__)

_PACKAGE_FILES: Final = (str(Path(__file__).resolve().parents[1]),)
"""Frames the temperature warning skips, so that it names the caller's line."""

_TEMPERATURE_SPREAD_PROBES: Final = 65
"""Evenly spaced positions over the run, beside the breakpoints, at which the warning
finds the spread of ``T(z)``."""


def _default_rate(
    component: Component,
    wavelength: float | None,
    selection: MaterialSelection,
) -> IonisationRate:
    """Select the rate ``plasma=True`` means for one component: optmat's.

    The rate model the selection names
    (:func:`solprop.nonlinear.optmat_rates.selected_rate`): no solprop formula, and a
    custom species is refused, since its rate must be given explicitly (SEP 3, D1).

    Parameters
    ----------
    component : Component
        Labelled species, density, initial ionisation and Raman overrides.
    wavelength : float or None
        Reference wavelength in m, the rate's carrier unless the selection names one.
    selection : MaterialSelection
        The run's material selection.

    Returns
    -------
    IonisationRate
        The default rate.
    """
    return selected_rate(
        component.species, selection, wavelength, label=component.label
    )


def _check_labels(labels: Sequence[str], mixture: Mixture, option: str) -> None:
    """Reject misspelled selections before building any response.

    Parameters
    ----------
    labels : Sequence of str
        Component labels explicitly selected by the caller.
    mixture : Mixture
        Material declaration supplying the valid labels.
    option : str
        Response option name used in a refusal diagnostic.
    """
    for label in labels:
        if label not in mixture.labels:
            raise MaterialError(f"unknown {option} component label {label!r}")


def _component_raman(
    component: Component,
    selected: bool | MolecularOptions,
    mixture: Mixture,
    *,
    thg: bool,
    explicit: bool,
    length: float | None = None,
) -> RamanResponse | None:
    """Select available terms by default and honour explicit per-term requests.

    Parameters
    ----------
    component : Component
        Labelled species, density, initial ionisation and Raman overrides.
    selected : bool or MolecularOptions
        Whether and which molecular terms to include.
    mixture : Mixture
        The mixture the component belongs to, whose temperature (or, for a
        temperature path, its reference, :func:`raman_temperature`) sets the
        rotational populations.
    thg : bool
        Whether the field-grid drive retains third harmonics.
    explicit : bool
        Whether absence of the selected response is an error.
    length : float or None, optional
        The run's length in m, over which a temperature path is averaged.

    Returns
    -------
    RamanResponse or None
        Selected component terms, or None for an allowed omission.
    """
    if selected is False:
        return None
    material = component.species
    data = material.raman
    options = selected if isinstance(selected, MolecularOptions) else component.raman
    required = options is not None and (
        options.rotation is True or options.vibration is True
    )
    if data is None or (data.rotational is None and data.vibrational is None):
        detail = "no Raman data" if data is None else data.missing
        if explicit or required:
            raise MaterialError(f"{component.label} ({material.name}): {detail}")
        logger.info("%s (%s): %s", component.label, material.name, detail)
        return None
    # The mixture's temperature reaches the rotational populations unless the
    # options set their own; a term flag left None takes the term when it exists.
    options = options or MolecularOptions()
    temperature = mixture.temperature
    if isinstance(temperature, ZProfile):
        # A temperature path reaches the populations as one reference temperature.
        rotating = data.rotational is not None and options.rotation is not False
        temperature = raman_temperature(
            mixture, component.label, options.temperature, length=length, warn=rotating
        )
    options = options.resolved(temperature)
    response = response_for(material, options, thg=thg)
    if component.dephasing is not None:
        terms = tuple(
            replace(term, model=replace(term.model, dephasing=component.dephasing))
            if isinstance(term.model, (RotationalLines, DampedOscillator))
            else term
            for term in response.tensor_terms
        )
        response = replace(response, tensor_terms=terms)
    return replace(response, species=component.label)


def raman_temperature(
    mixture: Mixture,
    label: str | Sequence[str],
    requested: float | None = None,
    *,
    length: float | None = None,
    warn: bool = True,
    widths: bool = False,
) -> float:
    r"""Return the temperature a component's rotational Raman populations take.

    The rotational populations are a Boltzmann ladder fixed per run: they cannot
    follow a temperature that varies along the fibre without new content at the
    kernel seam, which is not supported. A mixture whose temperature is a
    path therefore takes **one reference temperature**: `requested`
    (``MolecularOptions.temperature``) if given, otherwise the mean weighted by that
    **species' own** number density

    .. math:: \bar T_s = \frac{\int_0^L \rho_s\,T\,\mathrm{d}z}
       {\int_0^L \rho_s\,\mathrm{d}z}

    over the run ``[0, L]`` (`length`; without it, the species' density span, which
    must then be finite), by adaptive quadrature across the breakpoints of both. The
    populations are the species' own, so the positions where the species is are the
    ones that count: in independent populations one species can sit where the gas is
    cool and another where it is hot. The Raman linewidths are unchanged: they follow
    the density only. A constant temperature is returned as it is. The mean warns that
    the populations do not follow ``T(z)``; an explicit reference, the caller's own
    choice, logs the same message at INFO instead.

    An optmat response (SEP 3) is built for a mixture's whole composition, so it
    names several labels: the weight is then their **summed** density, and the
    reference fixes the widths' temperature laws as well as the populations
    (`widths`).

    Parameters
    ----------
    mixture : Mixture
        The fill.
    label : str or sequence of str
        The component whose populations these are, or the components of a
        composition, whose summed density is the weight.
    requested : float or None, optional
        An explicit reference temperature in K, > 0; ``None`` takes the mean.
    length : float or None, optional
        The run's length in m, > 0: the mean is taken over ``[0, length]``, never
        beyond the fibre. ``None`` takes the species' density span.
    warn : bool, optional
        Whether to warn when the temperature varies and the reference is the mean.
        Default ``True``.
    widths : bool, optional
        Whether the reference also fixes the linewidths' temperature laws, as for an
        optmat response; the message says so. Default ``False``.

    Returns
    -------
    float
        The reference temperature in K.

    Raises
    ------
    MaterialError
        If the label is unknown, or the mean is needed and there is no finite span to
        average over (an unbounded density and no `length`), or the species' density
        vanishes there.

    Warns
    -----
    RamanTemperatureWarning
        When the temperature varies, the reference is the mean and `warn` is set,
        naming the species, the reference, the spread of ``T(z)`` over the run, and
        that the populations do not follow it.

    Examples
    --------
    >>> import warnings
    >>> from solprop.nonlinear.mixtures import raman_temperature
    >>> from solprop.physdata import Mixture
    >>> from solprop.profiles import Linear
    >>> mixture = Mixture.fill({"N2": 1.0}, 1.0, temperature=Linear(290.0, 310.0, 1.0))
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")
    ...     f"{raman_temperature(mixture, 'N2', length=1.0):.3f}"
    '299.889'

    At a constant pressure the density goes as ``1/T`` (to the compressibility's
    variation), so the mean is nearly the logarithmic mean
    ``20 K / ln(310/290) = 299.889 K``.
    """
    temperature = mixture.temperature
    if not isinstance(temperature, ZProfile):
        return float(temperature if requested is None else requested)
    density, label = _weight(mixture, label)
    if length is not None:
        if not (math.isfinite(length) and length > 0.0):
            raise MaterialError(f"the run length must be positive, got {length} m")
        start, end = 0.0, float(length)
    else:
        start, end = density.z_span
    finite = math.isfinite(start) and math.isfinite(end) and end > start
    if not finite and requested is None:
        raise MaterialError(
            f"{label}: the rotational Raman populations need one reference "
            "temperature, and there is no finite run to average T(z) over (the "
            "density is unbounded along z): pass length=, or "
            "MolecularOptions(temperature=...)"
        )
    kinks = np.concatenate(
        [
            np.asarray(temperature.breakpoints, dtype=float),
            np.asarray(density.breakpoints, dtype=float),
        ]
    )
    if finite:
        inside = np.unique(
            np.concatenate(
                [
                    np.linspace(start, end, _TEMPERATURE_SPREAD_PROBES),
                    kinks[(kinks > start) & (kinks < end)],
                ]
            )
        )
    else:
        inside = _probe_positions(temperature)
    samples = np.asarray(temperature(inside), dtype=float)
    if requested is not None:
        reference = float(requested)
        how = "MolecularOptions.temperature"
    else:
        reference = _mean_temperature(density, temperature, start, end, inside)
        how = f"the mean of T(z) weighted by the density of {label} over the run"
    held = (
        "Raman populations and linewidth temperature laws"
        if widths
        else "rotational Raman populations"
    )
    message = (
        f"{label}: the gas temperature varies along z from "
        f"{float(np.min(samples)):.6g} K to {float(np.max(samples)):.6g} K, and "
        f"the {held} of {label} are computed at one "
        f"reference temperature, {reference:.6g} K ({how}); they do not follow "
        "T(z), which would need a change at the kernel seam"
    )
    if requested is not None:
        # The caller chose the reference knowing the path: say so, do not warn.
        logger.info(message)
    elif warn:
        warnings.warn(
            f"{message}. Pass MolecularOptions(temperature=...) to choose the "
            "reference",
            RamanTemperatureWarning,
            stacklevel=2,
            skip_file_prefixes=_PACKAGE_FILES,
        )
    return reference


def _weight(mixture: Mixture, label: str | Sequence[str]) -> tuple[ZProfile, str]:
    """Return the density a reference temperature is weighted by, and its name.

    Parameters
    ----------
    mixture : Mixture
        The fill.
    label : str or sequence of str
        One component, or the components of a composition.

    Returns
    -------
    tuple
        The component's density, or the components' summed density, in m^-3, and
        the label or the labels joined by ``+``.
    """
    if isinstance(label, str):
        return mixture[label].density, label
    paths = tuple(mixture[one].density for one in label)
    return (paths[0] if len(paths) == 1 else Sum(paths)), "+".join(label)


def _mean_temperature(
    density: ZProfile,
    temperature: ZProfile,
    start: float,
    end: float,
    kinks: Float[np.ndarray, " n_kinks"],
) -> float:
    """Return the mean of a temperature path weighted by one density over a span.

    Parameters
    ----------
    density : ZProfile
        The species' number density in m^-3.
    temperature : ZProfile
        The temperature path in K.
    start, end : float
        The finite span in m.
    kinks : numpy.ndarray
        Positions in m inside the span where the density or the temperature kinks.

    Returns
    -------
    float
        ``int rho T dz / int rho dz`` in K.

    Raises
    ------
    MaterialError
        If the density vanishes over the span.
    """
    edges = np.unique(np.concatenate([[start, end], kinks]))

    def weighted(z: float, power: int) -> float:
        """Return ``rho(z) T(z)^power`` at one position.

        Parameters
        ----------
        z : float
            Position in m.
        power : int
            0 for the density, 1 for the density times the temperature.

        Returns
        -------
        float
            The integrand.
        """
        position = np.array([z])
        value = float(density(position)[0])
        return value * float(temperature(position)[0]) ** power

    totals = [0.0, 0.0]
    for left, right in itertools.pairwise(edges):
        for power in (0, 1):
            part, _ = quad(
                weighted,
                left,
                right,
                args=(power,),
                epsabs=0.0,
                epsrel=1e-10,
                limit=200,
            )
            totals[power] += part
    if not totals[0] > 0.0:
        raise MaterialError(
            "the rotational Raman populations need a reference temperature, and the "
            "species' density vanishes over the run: pass "
            "MolecularOptions(temperature=...)"
        )
    return totals[1] / totals[0]


def _raman_choice(
    raman: bool | Sequence[str] | Mapping[str, bool | MolecularOptions], label: str
) -> bool | MolecularOptions:
    """Return what a Raman selection asks of one label.

    Parameters
    ----------
    raman : bool, sequence of str or Mapping
        :func:`responses_for`'s selection.
    label : str
        The component's label.

    Returns
    -------
    bool or MolecularOptions
        Whether, and which terms.
    """
    if isinstance(raman, Mapping):
        return raman.get(label, False)
    return raman if isinstance(raman, bool) else label in raman


def _composition_at(
    component: Component, mixture: Mixture, composition: RamanResponse | None
) -> RamanResponse | None:
    """Place a composition's one response at the label its leaf carries.

    Parameters
    ----------
    component : Component
        The component being visited.
    mixture : Mixture
        The fill.
    composition : RamanResponse or None
        The composition's response, if any.

    Returns
    -------
    RamanResponse or None
        The response at its leaf's label, and nothing at the others.
    """
    if composition is None:
        return None
    leaf = composition.species or mixture.labels[0]
    return composition if component.label == leaf else None


def electronic_response(
    material: Species,
    *,
    selection: MaterialSelection,
    wavelength: float | None,
    thg: bool,
    species: str | None = None,
) -> KerrResponse | DispersiveKerrResponse:
    """Build a species' electronic third-order response under a selection's policy.

    ``selection.electronic`` decides: the carrier policies (``"recommended"``,
    ``"carrier-kerr"``) give a :class:`~solprop.nonlinear.responses.KerrResponse` with
    the Kerr coefficient at the carrier, and the dispersive policies of SEP 6
    (:data:`~solprop.physdata.selection.DISPERSIVE_POLICIES`) a
    :class:`~solprop.nonlinear.responses.DispersiveKerrResponse` of optmat's whole
    hyperpolarizability model, with ``thg`` selecting its sector.

    Parameters
    ----------
    material : Species
        The population's species.
    selection : MaterialSelection
        The run's material selection.
    wavelength : float or None
        The carrier in m, at which a carrier policy evaluates the coefficient.
    thg : bool
        Retain the third harmonic: the full cubic, or for a dispersive policy its
        number-conserving sector when false.
    species : str or None, optional
        The component label the response is bound to.

    Returns
    -------
    KerrResponse or DispersiveKerrResponse
        The response.

    Raises
    ------
    MaterialError
        If the species has no Kerr data, or a dispersive policy meets a species
        without optmat's hyperpolarizability model (a supplied species, air), which
        is refused by name rather than given a borrowed curve.

    Examples
    --------
    >>> from solprop.nonlinear.mixtures import electronic_response
    >>> from solprop.physdata import MaterialSelection, species
    >>> chosen = MaterialSelection(electronic="gas-polynomial")
    >>> response = electronic_response(
    ...     species("Ar"), selection=chosen, wavelength=800e-9, thg=True
    ... )
    >>> type(response).__name__, response.evaluator
    ('DispersiveKerrResponse', 'polynomial')
    """
    evaluator = DISPERSIVE_POLICIES.get(selection.electronic)
    if evaluator is None:
        coefficient = material.kerr_coefficient(wavelength)
        if coefficient is None:
            raise MaterialError(f"{material.name} has no Kerr data")
        return KerrResponse(coefficient, thg=thg, species=species)
    if material.kerr is None:
        msg = (
            f"the electronic policy {selection.electronic!r} evaluates optmat's "
            f"hyperpolarizability model, and {material.name} carries none (a supplied "
            "species, or air, whose populations carry their own); give the species "
            "optmat's model or select 'carrier-kerr'"
        )
        raise MaterialError(msg)
    return DispersiveKerrResponse(
        material.kerr.dispersive_model(),
        evaluator,
        thg=thg,
        species=species,
    )


def responses_for(  # noqa: PLR0913 - independent public response selectors
    mixture: Mixture,
    *,
    kerr: bool = True,
    thg: bool | None = None,
    plasma: bool | IonisationRate | Mapping[str, bool | IonisationRate] = False,
    raman: bool | Sequence[str] | Mapping[str, bool | MolecularOptions] = False,
    wavelength: float | None = None,
    envelope: bool = False,
    length: float | None = None,
    selection: MaterialSelection | None = None,
    plasma_integration: IntegrationPolicy | None = None,
) -> tuple[
    KerrResponse
    | DispersiveKerrResponse
    | PlasmaResponse
    | EnvelopePlasmaResponse
    | RamanResponse,
    ...,
]:
    """Build responses bound to the ordered component labels of a mixture.

    Parameters
    ----------
    mixture : Mixture
        Species and density paths; component preionisation and Raman options apply.
    kerr : bool, optional
        Include every component's electronic third-order response: under the
        selection's ``electronic`` policy, the Kerr coefficient at the carrier or
        SEP 6's dispersive response (:func:`electronic_response`). Missing data is an
        error naming that species, including at zero density.
    thg : bool or None, optional
        Retain third harmonics; ``None`` follows the grid (off for envelopes).
    plasma : bool, IonisationRate or Mapping, optional
        ``True`` selects each built-in gas's rate from optmat under the selection
        (:class:`~solprop.nonlinear.optmat_rates.OptmatRate`); a custom species'
        rate must be given by label. A mapping selects rates by label; omitted
        labels are off.
        A bare rate must name the species to which it applies. Default ``False``.
    raman : bool, Sequence of str or Mapping, optional
        ``True`` includes every available term and logs omissions. A sequence selects
        labels; a mapping selects labels and optional molecular terms. Explicit
        unavailable selections raise an error. Default ``False``.
    wavelength : float or None, optional
        Reference wavelength in m, the carrier: required by the default rate and by
        a species whose Kerr coefficient is optmat's, which is evaluated there.
    envelope : bool, optional
        Build cycle-averaged envelope plasma from instantaneous isotropic rates.
        Default False retains the real-field response.
    length : float or None, optional
        The run's length in m. Under a temperature path the rotational Raman
        populations take a reference temperature averaged over ``[0, length]``
        (:func:`raman_temperature`); ``None`` averages over each species' density
        span, which must then be finite.
    selection : MaterialSelection or None, optional
        The run's material selection; ``None`` is ``MaterialSelection()``.
        ``plasma=True`` takes optmat's rate model for each component
        (:class:`~solprop.nonlinear.optmat_rates.OptmatRate`,
        ``selection.ionisation``), and refuses a custom species, whose rate must be
        given by label. The Raman response, too, is optmat's when the mixture's
        populations are built-in gases (SEP 3): one response for the whole
        composition, driven and damped by its summed density
        (:func:`solprop.nonlinear.optmat_raman.mixture_raman`), with the carrier
        `wavelength` unless the selection names one; a mixture of supplied species
        keeps their per-component responses.
    plasma_integration : {'band', 'trapezoid'} or None, optional
        The running-integral rule of every field plasma response (SEP 18); ``None``
        takes :class:`~solprop.nonlinear.responses.PlasmaResponse`'s default. One
        problem's channels share their current integrals, so one rule serves them
        all. Refused with envelope plasma.

    Returns
    -------
    tuple
        Declared responses in component order, with stable label bindings.

    Raises
    ------
    MaterialError
        If labels, rate bindings or required response data are unavailable, or a
        plasma integration rule is given beside envelope plasma.
    """
    selection = MaterialSelection() if selection is None else selection
    if envelope and plasma_integration is not None:
        msg = (
            f"plasma_integration={plasma_integration!r} applies to the field-grid "
            "plasma column; envelope plasma takes no running-integral rule"
        )
        raise MaterialError(msg)
    chosen: dict[str, IntegrationPolicy] = (
        {} if plasma_integration is None else {"integration": plasma_integration}
    )
    if isinstance(plasma, Mapping):
        _check_labels(tuple(plasma), mixture, "plasma")
    if not isinstance(raman, bool):
        _check_labels(tuple(raman), mixture, "Raman")
    harmonic = (not envelope) if thg is None else thg
    result: list[
        KerrResponse
        | DispersiveKerrResponse
        | PlasmaResponse
        | EnvelopePlasmaResponse
        | RamanResponse
    ] = []
    # A mixture of built-in gases takes one response for its whole composition
    # (SEP 3, D16); a caller's mixture of supplied species keeps the per-component
    # responses, taken as it was built.
    optmat_route = any(built_in_population(c.species) for c in mixture.components)
    composition = (
        mixture_raman(
            mixture,
            raman,
            selection=selection,
            thg=harmonic,
            wavelength=wavelength,
            length=length,
            explicit=not isinstance(raman, bool),
        )
        if optmat_route and raman is not False
        else None
    )
    for component in mixture.components:
        label = component.label
        material = component.species
        if kerr:
            # A carrier policy evaluates optmat's Kerr model at `wavelength`; a
            # dispersive one takes the whole model (SEP 6).
            try:
                result.append(
                    electronic_response(
                        material,
                        selection=selection,
                        wavelength=wavelength,
                        thg=harmonic,
                        species=label,
                    )
                )
            except MaterialError as error:
                raise MaterialError(f"{label} ({material.name}): {error}") from error
        selected = plasma.get(label, False) if isinstance(plasma, Mapping) else plasma
        if selected is not False:
            if envelope and harmonic:
                raise MaterialError("envelope plasma cannot be combined with THG")
            rate = (
                _default_rate(component, wavelength, selection)
                if selected is True
                else selected
            )
            if not isinstance(plasma, (bool, Mapping)):
                name = rate_species(rate)[0]
                if name != material.name:
                    raise MaterialError(
                        f"bare plasma rate names {name!r}, not "
                        f"{label} ({material.name})"
                    )
            result.append(
                EnvelopePlasmaResponse(
                    material, rate, preionisation=component.preionisation, species=label
                )
                if envelope
                else PlasmaResponse(
                    material,
                    rate,
                    preionisation=component.preionisation,
                    species=label,
                    **chosen,
                )
            )
        response = (
            _composition_at(component, mixture, composition)
            if optmat_route
            else _component_raman(
                component,
                _raman_choice(raman, label),
                mixture,
                thg=harmonic,
                explicit=not isinstance(raman, bool),
                length=length,
            )
        )
        if response is not None:
            result.append(response)
    if len(mixture.components) == 1:
        return tuple(replace(response, species=None) for response in result)
    return tuple(result)
