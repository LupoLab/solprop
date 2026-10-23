# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Compose ordered gas populations with their own number-density paths."""

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Final, cast

import numpy as np
from jaxtyping import Float, Num

from solprop._display import Displayed, line
from solprop.constants import ROOM_TEMPERATURE
from solprop.exceptions import DiluteScalingWarning, MaterialError
from solprop.physdata import _optmat
from solprop.physdata.gases import Gas
from solprop.physdata.raman_types import Dephasing, MolecularOptions
from solprop.physdata.selection import (
    DiluteScalingCheck,
    MaterialRecord,
    MaterialSelection,
)
from solprop.physdata.sellmeier import OptmatSusceptibility
from solprop.physdata.species import Species, species
from solprop.profiles import (
    Constant,
    DensityPath,
    PressureGradient,
    Scaled,
    Sum,
    ZProfile,
    _checked_temperature,
    _temperature_text,
    as_profile,
)

if TYPE_CHECKING:
    from solprop.gasflow import VacuumEnd

__all__ = [
    "DILUTE_SCALING_THRESHOLD",
    "Component",
    "Mixture",
    "check_preionisation",
]

Pressure = float | Sequence[float] | Sequence[Sequence[float]] | ZProfile

DILUTE_SCALING_THRESHOLD: Final = 1e-2
"""Relative error in ``n - 1`` of linear density scaling above which a fill warns.

Measured against optmat's Lorentz--Lorenz law with its refractivity virial: at 800 nm
and 293.15 K argon and nitrogen pass 1 % near 52 and 53 bar, krypton near 31 bar, xenon
near 17 bar and SF6 near 15 bar. The error is recorded in the run's manifest whatever
its size; the bound only decides the warning, until a finite-density law replaces the
dilute one (SEP 11).
"""

_AIR_COMPOSITIONS: Final = {True: "dry air (N2, O2, Ar)", False: "dry air (N2, O2)"}
"""optmat's named air compositions, by whether they carry argon."""


def _resolved(key: Gas | str | Species, selection: MaterialSelection | None) -> Species:
    """Return a supplied species as it is, or resolve a name under a selection.

    Parameters
    ----------
    key : Gas or str or Species
        A built-in gas's name or a supplied species.
    selection : MaterialSelection or None
        The selection for names; ``None`` is ``MaterialSelection()``.

    Returns
    -------
    Species
        The species. A supplied one is never matched to a built-in by its name.
    """
    if isinstance(key, Species):
        return key
    return species(key, selection=selection)


def _largest_density(
    path: ZProfile, temperature: float | ZProfile
) -> tuple[float, float]:
    """Return a density path's largest sampled value and the temperature there.

    Parameters
    ----------
    path : ZProfile
        Number density in m^-3 along z.
    temperature : float or ZProfile
        Temperature in K, or its path along z.

    Returns
    -------
    tuple of float
        The largest density in m^-3 at the path's breakpoints and evenly spaced
        probes (the same samples a :class:`Component` is validated at), and the
        temperature in K at that position.
    """
    start, end = path.z_span
    probes = np.unique(
        np.concatenate(
            (
                path.breakpoints,
                np.linspace(start, end, _PREIONISATION_PROBES)
                if math.isfinite(end)
                else np.array([start]),
            )
        )
    )
    values = np.asarray(path(probes), dtype=float)
    where = int(np.argmax(values))
    kelvin = (
        float(temperature(probes[where : where + 1])[0])
        if isinstance(temperature, ZProfile)
        else float(temperature)
    )
    return float(values[where]), kelvin


_DILUTE_SAMPLES: Final = 64
"""Frequencies of a band the dilute-scaling check evaluates, evenly chosen.

The error is of order ``N alpha / (3 eps0)`` and varies as slowly across a band as
the polarizability does, so a coarse selection finds its largest value; it keeps the
check, which runs at a fill's spectral sampling, cheap on a fine grid.
"""


def _dilute_scaling_checks(
    populations: Sequence[tuple[str, Species, ZProfile]],
    temperature: float | ZProfile,
    omega: Float[np.ndarray, " n_omega"],
) -> tuple[DiluteScalingCheck, ...]:
    """Measure the error of linear density scaling for optmat-selected populations.

    Parameters
    ----------
    populations : sequence of tuple
        ``(label, species, density path)`` of each population; only those whose
        linear law is an :class:`~solprop.physdata.sellmeier.OptmatSusceptibility`
        and whose density is not identically zero are checked.
    temperature : float or ZProfile
        Temperature in K, or its path along z.
    omega : numpy.ndarray
        Angular frequencies in rad/s of the band, > 0.

    Returns
    -------
    tuple of DiluteScalingCheck
        One per checked population, at its largest density; one optmat could not
        evaluate is returned unmeasured, with its reason.
    """
    values = np.asarray(omega, dtype=float).ravel()
    values = values[values > 0]
    if values.size > _DILUTE_SAMPLES:
        values = values[np.linspace(0, values.size - 1, _DILUTE_SAMPLES).astype(int)]
    checks = []
    for label, material, path in populations:
        model = material.susceptibility
        if not isinstance(model, OptmatSusceptibility):
            continue
        density, kelvin = _largest_density(path, temperature)
        if density == 0.0:
            continue
        # The check is a diagnostic: a refusal from optmat is recorded, as not
        # measured, and never fails the fill whose index it does not change.
        try:
            error, wavelength = model.dilute_scaling_error(density, kelvin, values)
            reason = ""
        except MaterialError as refusal:
            error, wavelength, reason = math.nan, math.nan, str(refusal)
        checks.append(
            DiluteScalingCheck(
                label,
                model.gas.value,
                density,
                kelvin,
                wavelength,
                error,
                DILUTE_SCALING_THRESHOLD,
                reason,
            )
        )
    return tuple(checks)


def _warn_dilute_scaling(checks: Sequence[DiluteScalingCheck]) -> None:
    """Warn once per gas and evaluation context where dilute scaling exceeds its bound.

    Parameters
    ----------
    checks : sequence of DiluteScalingCheck
        The measured checks.

    Warns
    -----
    DiluteScalingWarning
        For each gas whose error exceeds :data:`DILUTE_SCALING_THRESHOLD`.
    """
    for check in checks:
        if check.exceeded:
            _optmat.warn_once(
                ("dilute", check.material),
                f"{check.label}: linear density scaling departs from optmat's "
                f"finite-density index law by {check.relative_error:.3g} relative in "
                f"n - 1 at {check.density:.4g} m^-3, {check.temperature:g} K and "
                f"{check.wavelength * 1e9:.4g} nm, above {check.threshold:g}; the run "
                "keeps the dilute law, and records the error",
                DiluteScalingWarning,
            )


def _pressure_path(  # noqa: PLR0913 - the fill vocabulary and what the flow needs
    value: Pressure,
    length: float | None,
    floor: float | None,
    *,
    radius: float | ZProfile | None = None,
    vacuum_end: str | None = None,
    composition: Mapping[Species, float] | None = None,
    temperature: float | ZProfile = ROOM_TEMPERATURE,
) -> ZProfile:
    """Resolve the pressure vocabulary without introducing a driver dependency.

    Parameters
    ----------
    value : float or sequence or ZProfile
        Total pressure in bar.
    length : float or None
        Fibre length in m, required for endpoint pairs.
    floor : float or None
        Minimum endpoint pressure in bar.
    radius : float, ZProfile or None, optional
        Core radius in m. Given, an endpoint pair or stage table becomes the
        flow-aware fill of :func:`solprop.gasflow.steady_fill`; ``None`` keeps the
        uniform law and the numerical floor.
    vacuum_end : {'choked', 'floor'} or None, optional
        How an outer end below its choked pressure is treated; ``None`` chokes
        where a radius is given and floors otherwise.
    composition : Mapping or None, optional
        Species to mole fraction of the fill, whose transport data the flow uses.
    temperature : float or ZProfile, optional
        Temperature in K, validated, or a varying path of it along z: a gradient then
        follows the flow weight ``eta(T) T`` of the composition
        (:func:`solprop.gasflow.temperature_weight`) with or without a radius.

    Returns
    -------
    ZProfile
        The total-pressure profile in bar.
    """
    if vacuum_end not in (None, "choked", "floor"):
        raise MaterialError(
            f"vacuum_end must be 'choked' or 'floor', got {vacuum_end!r}"
        )
    if isinstance(value, (ZProfile, float, int)):
        return _endless_path(
            value, length, vacuum_end, radius, temperature, composition
        )
    if len(value) != 2:
        raise MaterialError(
            "pressure must be a number, profile, endpoint pair or (z, p) pair"
        )
    first, second = value
    if isinstance(first, (float, int)) and isinstance(second, (float, int)):
        if length is None:
            raise MaterialError("length is required for endpoint pressure pairs")
        if (
            radius is None
            and vacuum_end is None
            and not isinstance(temperature, ZProfile)
        ):
            return PressureGradient.two_point(length, first, second, floor=floor)
        positions = np.array([0.0, float(length)])
        pressures = np.array([float(first), float(second)])
    else:
        positions = np.asarray(first, dtype=float)
        pressures = np.asarray(second, dtype=float)
    if radius is None:
        if vacuum_end == "choked":
            raise MaterialError(
                "vacuum_end='choked' needs the core radius, which sets the choked "
                "pressure: pass radius=, or build the fill through a simulation "
                "constructor, which supplies the guide's"
            )
        if not isinstance(temperature, ZProfile):
            return PressureGradient.from_stages(positions, pressures, floor=floor)
    # The flow needs the gas and so imports this module: import it where it is used.
    from solprop.gasflow import (  # noqa: PLC0415 - gasflow imports mixtures
        steady_fill,
        temperature_weight,
    )

    if radius is None:
        return PressureGradient.from_stages(
            positions,
            pressures,
            floor=floor,
            weight=temperature_weight(
                dict(composition or {}), cast("ZProfile", temperature), positions
            ),
        )

    return steady_fill(
        dict(composition or {}),
        positions,
        pressures,
        radius=radius,
        temperature=temperature,
        vacuum_end=cast("VacuumEnd", "choked" if vacuum_end is None else vacuum_end),
        floor=floor,
    )


def _endless_path(
    value: float | ZProfile,
    length: float | None,
    vacuum_end: str | None,
    radius: float | ZProfile | None,
    temperature: float | ZProfile = ROOM_TEMPERATURE,
    composition: Mapping[Species, float] | None = None,
) -> ZProfile:
    """Return a constant pressure or a supplied profile, which have no end to choke.

    Parameters
    ----------
    value : float or ZProfile
        Total pressure in bar.
    length : float or None
        Fibre length in m, the span of a constant.
    vacuum_end : str or None
        Refused unless ``None``.
    radius : float, ZProfile or None
        The guide's core radius in m. A constant pressure ignores it; a varying
        pressure profile must already follow a varying radius.
    temperature : float or ZProfile, optional
        The validated temperature in K. A varying path beside a varying pressure
        profile needs the profile to carry its flow weight (:func:`_check_flow_weight`).
    composition : Mapping or None, optional
        Species to mole fraction of the fill, whose flow weight the profile must carry.

    Returns
    -------
    ZProfile
        The total-pressure profile in bar.

    Raises
    ------
    MaterialError
        If the pressure is invalid, `vacuum_end` is given, or a varying radius or
        temperature is given beside a varying profile built without it.
    """
    if isinstance(value, ZProfile):
        if vacuum_end is not None:
            raise MaterialError(
                f"vacuum_end={vacuum_end!r} applies to a fill built from end "
                "pressures, and a supplied pressure profile carries its own ends"
            )
        bore = None if radius is None else as_profile(radius)
        if (
            bore is not None
            and not bore.is_constant
            and not value.is_constant
            and getattr(value, "radius", None) is None
        ):
            # The radius would be silently dropped: the profile is used as given.
            raise MaterialError(
                "the pressure profile was built without the bore, and radius= "
                "varies along z: its p^2 is linear in z, whereas a taper's steady "
                "flow makes p^2 linear in the hydraulic resistance int a^-4 dz. Build "
                "it with PressureGradient.from_stages(..., radius=) or "
                "solprop.gasflow.steady_fill, or pass the end pressures as a pair "
                "beside radius="
            )
        _check_flow_weight(value, temperature, composition or {}, "the fill")
        return value
    if not math.isfinite(value) or value < 0:
        raise MaterialError("pressure must be finite and non-negative")
    if vacuum_end is not None:
        raise MaterialError(
            f"vacuum_end={vacuum_end!r} applies to a pressure gradient, and a "
            "constant pressure has no vacuum end"
        )
    return Constant(float(value), (0.0, math.inf if length is None else length))


_WEIGHT_MATCH: Final = 1e-5
"""Relative agreement at which a supplied flow weight counts as the declared one.

Compared after normalising both to their mean, since a constant factor cancels from
the fill: two viscosity laws of the same gas fitted over different temperature ranges
each reproduce the data to 1e-6 (``gasflow._VISCOSITY_TOLERANCE``), so they agree to a
few times that, while a different gas or temperature path differs by per cents.
"""


def _check_flow_weight(
    pressure: ZProfile,
    temperature: float | ZProfile,
    composition: object,
    owner: str,
) -> None:
    """Refuse a steady-flow fill that does not carry the declared flow weight.

    Under a temperature that varies along z a steady flow makes ``p^2`` linear in
    ``int eta(T) T a^-4 dz``, not ``int a^-4 dz``: a varying
    :class:`~solprop.profiles.PressureGradient` given beside it must carry that weight
    (``PressureGradient.weight``, built by :func:`solprop.gasflow.steady_fill` or
    :func:`solprop.gasflow.temperature_weight`) **for this gas and this temperature
    path**. Any other profile (a ``Linear``, a ``PiecewiseLinear``, a callable) makes
    no flow claim: it is the pressure the caller declares, and is taken as given.
    The weight is compared with the one the declaration implies, normalised to their
    means, at the fill's breakpoints and evenly spaced probes. One check for every door
    that takes a pressure profile beside a temperature.

    Parameters
    ----------
    pressure : ZProfile
        The pressure profile in bar; only a ``PressureGradient`` is checked.
    temperature : float or ZProfile
        The validated temperature in K; a number needs nothing.
    composition : Gas, str, Species or Mapping
        The gas of the flow, or species to mole fractions.
    owner : str
        What carries the profile, for the message.

    Raises
    ------
    MaterialError
        If the fill varies and carries no weight, a weight for another gas or
        temperature path, or the gas has no transport data to confirm it.
    """
    if (
        not isinstance(temperature, ZProfile)
        or not isinstance(pressure, PressureGradient)
        or pressure.is_constant
    ):
        return
    weight = getattr(pressure, "weight", None)
    build = (
        "Build it with solprop.gasflow.steady_fill(..., temperature=) for this gas "
        "and temperature, or pass the end pressures as a pair"
    )
    if not isinstance(weight, ZProfile):
        raise MaterialError(
            f"the pressure gradient of {owner} was built without the flow weight, and "
            "the temperature varies along z: its p^2 is linear in int a^-4 dz, whereas "
            "a steady flow at a varying temperature makes it linear in "
            f"int eta(T) T a^-4 dz. {build}"
        )
    # The flow needs the gas and so imports this module: import it where it is used.
    from solprop.gasflow import temperature_weight  # noqa: PLC0415 - see above

    edges = pressure.positions
    expected = temperature_weight(
        cast("Mapping[Species, float]", composition), temperature, edges
    )
    probes = np.unique(
        np.concatenate(
            [
                np.linspace(float(edges[0]), float(edges[-1]), _PREIONISATION_PROBES),
                np.asarray(pressure.breakpoints, dtype=float),
            ]
        )
    )
    given = np.asarray(weight(probes), dtype=float)
    implied = np.asarray(expected(probes), dtype=float)
    mismatch = np.max(np.abs(given / np.mean(given) / (implied / np.mean(implied)) - 1))
    if not mismatch <= _WEIGHT_MATCH:
        raise MaterialError(
            f"the pressure gradient of {owner} carries a flow weight for another gas "
            f"or temperature path: its eta(T) T differs from this declaration's by "
            f"{mismatch:.2g} relative (at most {_WEIGHT_MATCH:g}). {build}"
        )


def _fractions(values: Sequence[float]) -> tuple[float, ...]:
    """Validate a distribution without silently renormalising supplied fractions.

    Parameters
    ----------
    values : sequence of float
        Non-negative mole fractions summing to one.

    Returns
    -------
    tuple of float
        Validated fractions in the supplied order.
    """
    result = tuple(float(v) for v in values)
    if not result or any(not math.isfinite(v) or v < 0 for v in result):
        raise MaterialError("fractions must be finite, non-negative and non-empty")
    # A supplied distribution can incur one rounding per entry and its sum.
    if abs(math.fsum(result) - 1.0) > 2 * len(result) * np.finfo(float).eps:
        raise MaterialError(f"fractions must sum to one, got {math.fsum(result):.17g}")
    return result


_PREIONISATION_PROBES = 65
"""Evenly spaced positions a finite path is sampled at, beside its breakpoints.

The same count the density paths of a :class:`Component` are sampled at. The probe is
a guard at construction, not a proof. The translation layer then checks the compiled
table of every plasma channel at its endpoints and stationary points on every interval
(``KernelProblem._check_plasma``), so a table that leaves ``[0, 1]`` is refused; but a
feature narrower than the table resolves between its samples is not seen by either
check, and must be declared as breakpoints of the path.
"""


def check_preionisation(value: object, owner: str = "") -> float | ZProfile:
    """Validate an initial ionised fraction, a number or a path in ``[0, 1]``.

    A number is the fraction ionised before the pulse arrives, the same everywhere. A
    path is that fraction at each position along the fibre, still set before the pulse
    arrives there; it is sampled at its breakpoints and at evenly spaced probes over its
    span (at its start alone when the span is unbounded). A feature narrower than those
    samples is not seen: declare it with breakpoints.

    Parameters
    ----------
    value : object
        The fraction, dimensionless in ``[0, 1]``: a real number (a Python or NumPy
        scalar, or a zero-dimensional array), or a ``ZProfile`` of it along z in m.
    owner : str, optional
        What carries the value, for the message: a species or component label.

    Returns
    -------
    float or ZProfile
        The value itself when it is a Python number or a path; a NumPy scalar or a
        zero-dimensional array as the Python float it holds, which is what the
        translation layer takes.

    Raises
    ------
    MaterialError
        If `value` is a boolean, neither a real number nor a ``ZProfile``, not finite,
        or outside ``[0, 1]`` at any sampled position.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.physdata.mixtures import check_preionisation
    >>> from solprop.profiles import Linear
    >>> check_preionisation(0.1)
    0.1
    >>> check_preionisation(np.array(0.25))
    0.25
    >>> check_preionisation(Linear(0.0, 0.2, 1.0), "Ar")
    Linear(0 to 0.2 over 1 m)
    >>> check_preionisation(Linear(0.0, 1.5, 1.0), "Ar")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    solprop.exceptions.MaterialError: preionisation must be in [0, 1] for Ar at ...
    """
    where = f" for {owner}" if owner else ""
    if isinstance(value, ZProfile):
        start, end = value.z_span
        probes = np.unique(
            np.concatenate(
                (
                    np.asarray(value.breakpoints, dtype=float),
                    np.linspace(start, end, _PREIONISATION_PROBES)
                    if math.isfinite(end)
                    else np.array([start]),
                )
            )
        )
        samples = np.asarray(value(probes), dtype=float)
        bad = ~np.isfinite(samples) | (samples < 0.0) | (samples > 1.0)
        if np.any(bad):
            first = int(np.argmax(bad))
            msg = (
                f"preionisation must be in [0, 1]{where} at every position; the path "
                f"reaches {samples[first]:g} at z = {probes[first]:g} m"
            )
            raise MaterialError(msg)
        return value
    number = value
    if isinstance(value, np.ndarray) and value.ndim == 0:
        number = value[()]
    if isinstance(number, bool | np.bool_) or not isinstance(number, numbers.Real):
        msg = (
            f"preionisation{where} must be a number or a ZProfile, got "
            f"{type(value).__name__}"
        )
        raise MaterialError(msg)
    real = float(number)
    if not 0.0 <= real <= 1.0:
        msg = f"preionisation must be in [0, 1]{where}, got {value!r}"
        raise MaterialError(msg)
    # A Python number is kept as given (an int stays an int in the run record).
    return cast("float", number) if type(number) in (float, int) else real


def _preionisation_text(value: float | ZProfile) -> str:
    """Return a number as ``%g`` and a path by its own short ``repr``, for printing.

    Parameters
    ----------
    value : float or ZProfile
        An initial ionised fraction, or a path of it along z.

    Returns
    -------
    str
        For example ``'0.1'`` or ``'Linear(0 to 0.2 over 1 m)'``.
    """
    return repr(value) if isinstance(value, ZProfile) else f"{value:g}"


@dataclass(frozen=True)
class Component(Displayed):
    """Bind one labelled population to a species and a density path.

    Attributes
    ----------
    species : Species
        The population's material data.
    density : ZProfile
        Number density in m^-3 along z, with non-negative values.
    label : str
        Unique mixture label; an empty default takes the species' name.
    preionisation : float or ZProfile
        Initial ionised fraction, between zero and one inclusive: a number, or a path
        along z giving the fraction ionised before the pulse arrives at each position.
    dephasing : Dephasing or None
        Supplied partner-aware Raman linewidth, if available.
    raman : MolecularOptions or None
        Raman term selection; None selects every available term.
    """

    species: Species
    density: ZProfile
    label: str = ""
    preionisation: float | ZProfile = 0.0
    dephasing: Dephasing | None = None
    raman: MolecularOptions | None = None

    def __repr__(self) -> str:
        """Return a one-line description: the label, the species and what it carries.

        Returns
        -------
        str
            For example ``Component(N2, preionisation=0)``.
        """
        fields: list[tuple[str, object]] = [("", self.label or self.species.name)]
        if self.label and self.label != self.species.name:
            fields.append(("species", self.species.name))
        fields.append(("preionisation", _preionisation_text(self.preionisation)))
        return line("Component", fields)

    def __post_init__(self) -> None:
        """Validate the label, preionisation and sampled density domain."""
        if not self.label:
            object.__setattr__(self, "label", self.species.name)
        if not self.label.strip() or ":" in self.label:
            raise MaterialError(f"invalid component label {self.label!r}")
        object.__setattr__(
            self, "preionisation", check_preionisation(self.preionisation, self.label)
        )
        start, end = self.density.z_span
        probes = np.unique(
            np.concatenate(
                (
                    self.density.breakpoints,
                    np.linspace(start, end, 65)
                    if math.isfinite(end)
                    else np.array([start]),
                )
            )
        )
        self._checked_density(probes)

    def _checked_density(
        self, z: Float[np.ndarray, " n_z"]
    ) -> Float[np.ndarray, " n_z"]:
        """Reject negative or non-finite population densities at the requested points.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Non-negative finite densities in m^-3.
        """
        values = self.density(z)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise MaterialError(
                f"density for {self.label} must be finite and non-negative"
            )
        return values


@dataclass(frozen=True, eq=False)
class Mixture(Displayed):
    """Keep ordered populations and their shared fill, when present.

    Attributes
    ----------
    components : tuple of Component
        Non-empty ordered populations with unique labels.
    temperature : float or ZProfile
        Positive temperature in K, or a path of it along z, which every density path
        of a fill follows (:class:`solprop.profiles.DensityPath`); a constant path is
        stored as its value. A varying temperature makes a shared fill's populations
        independent parameters of the linear operator (``density:<label>``), since
        one pressure no longer fixes every density, and the rotational Raman
        populations take one reference temperature
        (:mod:`solprop.nonlinear.mixtures`).
    shared : ZProfile or None
        Shared total-pressure path in bar, or None for independent densities.
    fractions : tuple of float or None
        Mole fractions of a shared fill, in component order.
    composition : str or None
        The named optmat composition the fractions were taken from, such as
        ``"dry air (N2, O2)"``, which a run's material manifest records; ``None``
        for fractions given otherwise. Set by :meth:`air` under a selection; a
        mixture derived from it by :meth:`with_fill`, :meth:`with_densities` or
        :meth:`at_parameters` does not keep it.
    """

    components: tuple[Component, ...]
    temperature: float | ZProfile = ROOM_TEMPERATURE
    shared: ZProfile | None = None
    fractions: tuple[float, ...] | None = None
    # Carried by the mixture's own record (`serialise`) and the run's material
    # manifest, where its identity is hashed; left out of the generic dataclass
    # record, so that every mixture without one keeps the digest it had.
    composition: str | None = field(default=None, metadata={"record": False})

    def _pressure_text(self) -> str:
        """Return the shared pressure at the entrance, for printing.

        Returns
        -------
        str
            The pressure in bar, or a note that the populations are independent.
        """
        if self.shared is None:
            return "independent populations"
        return f"{float(self.shared(np.array([0.0]))[0]):.4g} bar"

    def __repr__(self) -> str:
        """Return a one-line description: the species, the pressure, the temperature.

        Returns
        -------
        str
            For example ``Mixture(N2+O2, 1 bar, 293.15 K)``.
        """
        return line(
            "Mixture",
            [
                ("", "+".join(self.labels)),
                ("", self._pressure_text()),
                ("", _temperature_text(self.temperature)),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        fractions = self.fractions or (None,) * len(self.components)
        rows: list[tuple[str, object]] = [
            ("total pressure", f"{self._pressure_text()} at the entrance"),
            ("temperature", _temperature_text(self.temperature)),
        ]
        for component, fraction in zip(self.components, fractions, strict=True):
            share = "" if fraction is None else f"mole fraction {fraction:.4g}, "
            rows.append(
                (
                    component.label,
                    f"{share}preionisation "
                    f"{_preionisation_text(component.preionisation)}",
                )
            )
        return f"Mixture: {'+'.join(self.labels)}", rows

    def __post_init__(self) -> None:
        """Validate the composition without changing its ordering."""
        object.__setattr__(self, "components", tuple(self.components))
        if not self.components:
            raise MaterialError("a mixture needs at least one component")
        if len(set(self.labels)) != len(self.labels):
            raise MaterialError(f"component labels must be unique: {self.labels}")
        object.__setattr__(
            self, "temperature", _checked_temperature(self.temperature, "the mixture")
        )
        if self.fractions is not None and (
            self.shared is None or len(self.fractions) != len(self.components)
        ):
            raise MaterialError(
                "fill fractions need a shared pressure and one entry per component"
            )
        if self.fractions is not None:
            object.__setattr__(self, "fractions", _fractions(self.fractions))
        if self.composition is not None and self.fractions is None:
            raise MaterialError(
                f"a named composition ({self.composition!r}) needs the fill fractions "
                "it gave"
            )

    @classmethod
    def fill[SpeciesKey: Gas | str | Species](  # noqa: PLR0913 - the fill vocabulary
        cls,
        fractions: Mapping[SpeciesKey, float],
        pressure: Pressure,
        *,
        temperature: float | ZProfile = ROOM_TEMPERATURE,
        floor: float | None = None,
        length: float | None = None,
        radius: float | ZProfile | None = None,
        vacuum_end: str | None = None,
        selection: MaterialSelection | None = None,
    ) -> "Mixture":  # noqa: UP037 - runtime annotations are evaluated during class creation
        """Construct a fill using each species' partial-pressure equation of state.

        Parameters
        ----------
        fractions : collections.abc.Mapping
            Species to non-negative mole fraction, summing to one.
        pressure : float or sequence or ZProfile
            Total pressure in bar; a scalar, endpoint pair, (z, p) pair or profile.
        temperature : float or ZProfile, optional
            Positive temperature in K, or a path of it along z in m. Every density
            follows it; a gradient built here follows the flow weight
            ``eta(T) T`` (:func:`solprop.gasflow.temperature_weight`), which needs
            the species' transport data; a varying pressure profile given beside a
            varying temperature must carry that weight.
        floor : float or None, optional
            Pressure-gradient floor in bar; None uses the standard floor.
        length : float or None, optional
            Positive length in m, required for endpoint pressure pairs.
        radius : float, ZProfile or None, optional
            Core radius in m of the guide the gas flows through. Given, an endpoint
            pair or (z, p) pair becomes the steady flow of the whole mixture
            (:func:`solprop.gasflow.steady_fill`, with the mixture's Wilke viscosity
            and sound speed): it follows the bore, chokes a vacuum end and checks the
            flow regime. ``None`` keeps the uniform law and the numerical floor; the
            simulation constructors supply the guide's radius.
        vacuum_end : {'choked', 'floor'} or None, optional
            How an outer end below its choked pressure is treated; ``None`` chokes
            it when `radius` is given. Refused beside a constant pressure or a
            profile, which have no end to choke.
        selection : MaterialSelection or None, optional
            Resolve the named species from optmat under this selection
            (:func:`~solprop.physdata.species`); ``None`` is
            ``MaterialSelection()``, optmat's recommended models. A supplied
            :class:`~solprop.physdata.species.Species` is taken as it is. Each
            population's density is still the pure gas's at its partial pressure.

        Returns
        -------
        Mixture
            The fill in mapping iteration order.

        Raises
        ------
        MaterialError
            If fractions, species or the pressure specification are invalid.
        """
        weights = _fractions(tuple(fractions.values()))
        materials = tuple(_resolved(key, selection) for key in fractions)
        kelvin = _checked_temperature(temperature, "the fill")
        # One evaluation context: a flow built here evaluates the viscosity many times.
        with _optmat.material_context():
            path = _pressure_path(
                pressure,
                length,
                floor,
                radius=radius,
                vacuum_end=vacuum_end,
                composition=dict(zip(materials, weights, strict=True)),
                temperature=kelvin,
            )
            components = tuple(
                Component(
                    material, DensityPath(material, Scaled(path, fraction), kelvin)
                )
                for material, fraction in zip(materials, weights, strict=True)
            )
        return cls(components, kelvin, path, weights)

    @classmethod
    def partial_pressures[SpeciesKey: Gas | str | Species](
        cls,
        pressures: Mapping[SpeciesKey, float | ZProfile],
        *,
        temperature: float | ZProfile = ROOM_TEMPERATURE,
        selection: MaterialSelection | None = None,
    ) -> "Mixture":  # noqa: UP037 - see fill
        """Construct populations from independent partial pressures.

        Parameters
        ----------
        pressures : collections.abc.Mapping
            Species to non-negative partial pressure in bar, scalar or profile.
        temperature : float or ZProfile, optional
            Positive temperature in K, or a path of it along z in m, which every
            density follows. Beside it a varying pressure profile must carry its
            species' flow weight (the equal mixture's for one path shared by all).
        selection : MaterialSelection or None, optional
            Resolve the named species from optmat, as in :meth:`fill`.

        Returns
        -------
        Mixture
            The populations in mapping order. Identical profile objects share a fill.
        """
        paths = tuple(as_profile(p) for p in pressures.values())
        kelvin = _checked_temperature(temperature, "the populations")
        materials = tuple(_resolved(key, selection) for key in pressures)
        shared = bool(paths) and all(p is paths[0] for p in paths)
        for material, path in zip(materials, paths, strict=True):
            # Independent populations are each their own flow; one path for every
            # species is the flow of the equal mixture.
            flow = (
                dict.fromkeys(materials, 1.0 / len(materials))
                if shared
                else {material: 1.0}
            )
            _check_flow_weight(path, kelvin, flow, f"the {material.name} population")
        components = tuple(
            Component(material, DensityPath(material, path, kelvin))
            for material, path in zip(materials, paths, strict=True)
        )
        if shared:
            count = len(paths)
            return cls(
                components,
                kelvin,
                Scaled(paths[0], float(count)),
                (1 / count,) * count,
            )
        return cls(components, kelvin)

    @classmethod
    def of(
        cls, *components: Component, temperature: float | ZProfile = ROOM_TEMPERATURE
    ) -> "Mixture":  # noqa: UP037 - see fill
        """Construct a mixture from labelled density populations.

        Parameters
        ----------
        *components : Component
            Ordered populations with unique labels.
        temperature : float or ZProfile, optional
            Positive temperature in K, or a path of it along z in m. The densities
            are given, so it reaches the run only through the Raman populations'
            reference temperature.

        Returns
        -------
        Mixture
            A mixture with independent density paths.
        """
        return cls(components, temperature)

    @classmethod
    def air(  # noqa: PLR0913 - the fill vocabulary of fill, with air's composition
        cls,
        pressure: Pressure,
        *,
        temperature: float | ZProfile = ROOM_TEMPERATURE,
        fractions: Mapping[Gas | str | Species, float] | None = None,
        argon: bool = True,
        length: float | None = None,
        floor: float | None = None,
        radius: float | ZProfile | None = None,
        vacuum_end: str | None = None,
        selection: MaterialSelection | None = None,
    ) -> "Mixture":  # noqa: UP037 - see fill
        """Construct dry air from nitrogen, oxygen and argon; omit carbon dioxide.

        Parameters
        ----------
        pressure : float or sequence or ZProfile
            Total pressure in bar, with the vocabulary of :meth:`fill`.
        temperature : float or ZProfile, optional
            Positive temperature in K, or a path of it along z, as in :meth:`fill`.
        fractions : collections.abc.Mapping or None, optional
            Mole fractions. ``None`` takes the named composition of the selection
            (below).
        argon : bool, optional
            Whether the default composition carries argon. ``False`` is the
            two-component convenience the capillary interface's ``"air"`` means:
            nitrogen and oxygen renormalised to each other, a declared approximation.
            Ignored when `fractions` is given.
        length : float or None, optional
            Positive length in m, required for endpoint pressure pairs.
        floor : float or None, optional
            Pressure-gradient floor in bar.
        radius : float, ZProfile or None, optional
            Core radius in m for a steady-flow gradient, as in :meth:`fill`.
        vacuum_end : {'choked', 'floor'} or None, optional
            How a vacuum end is treated, as in :meth:`fill`.
        selection : MaterialSelection or None, optional
            Resolve the species from optmat, as in :meth:`fill`. The default
            composition is optmat's named one: CIPM-2007's
            ``"dry air (N2, O2, Ar)"``, or with ``argon=False``
            ``"dry air (N2, O2)"``, each renormalised over what it keeps.

        Returns
        -------
        Mixture
            The resolved gas populations, not the dispersion-only Gas.Air entry.

        Examples
        --------
        >>> from solprop.physdata import Mixture
        >>> Mixture.air(1.0).labels
        ('N2', 'O2', 'Ar')
        >>> Mixture.air(1.0, argon=False).labels
        ('N2', 'O2')
        >>> f"{Mixture.air(1.0).fractions[0]:.6f}", Mixture.air(1.0).composition
        ('0.781184', 'dry air (N2, O2, Ar)')
        """
        selection = MaterialSelection() if selection is None else selection
        named_air = fractions is None
        chosen: Mapping[Gas | str | Species, float]
        if fractions is not None:
            chosen = fractions
        else:
            named = _optmat.composition_fractions(_AIR_COMPOSITIONS[argon])
            chosen = {gas: fraction for gas, fraction in named.items()}  # noqa: C416 - the key type widens
        mixture = cls.fill(
            chosen,
            pressure,
            temperature=temperature,
            length=length,
            floor=floor,
            radius=radius,
            vacuum_end=vacuum_end,
            selection=selection,
        )
        if named_air:
            return replace(mixture, composition=_AIR_COMPOSITIONS[argon])
        return mixture

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the records of the optmat models the populations carry, once each.

        Returns
        -------
        tuple of MaterialRecord
            In component order, a model shared by two populations once.
        """
        seen: dict[str, MaterialRecord] = {}
        for component in self.components:
            for entry in component.species.material_records:
                seen.setdefault(entry.digest, entry)
        return tuple(seen.values())

    def dilute_scaling_checks(
        self, omega: Float[np.ndarray, " n_omega"]
    ) -> tuple[DiluteScalingCheck, ...]:
        """Measure the error of linear density scaling of each optmat population.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s of the band, > 0.

        Returns
        -------
        tuple of DiluteScalingCheck
            One per optmat-selected population with a non-zero density, at its
            largest density along the fill.
        """
        return _dilute_scaling_checks(
            [(c.label, c.species, c.density) for c in self.components],
            self.temperature,
            omega,
        )

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the labels in mixture order.

        Returns
        -------
        tuple of str
            Unique population labels.
        """
        return tuple(c.label for c in self.components)

    def __getitem__(self, label: str) -> Component:
        """Look up a population by its label.

        Parameters
        ----------
        label : str
            Population label.

        Returns
        -------
        Component
            The labelled population.

        Raises
        ------
        MaterialError
            If the label is absent.
        """
        for component in self.components:
            if component.label == label:
                return component
        raise MaterialError(
            f"unknown mixture label {label!r}; available: {self.labels}"
        )

    def density(
        self, z: Float[np.ndarray, " n_z"]
    ) -> Float[np.ndarray, "n_species n_z"]:
        """Evaluate every component density in mixture order.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Non-negative densities in m^-3, shape (n_species, n_z).
        """
        return np.stack([c._checked_density(z) for c in self.components])

    def total_density(self, z: Float[np.ndarray, " n_z"]) -> Float[np.ndarray, " n_z"]:
        """Evaluate total number density.

        Parameters
        ----------
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Total density in m^-3, shape (n_z,).
        """
        values = self.density(z)
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result

    def _optical_sum(
        self,
        omega: Float[np.ndarray, " n_omega"],
        z: Float[np.ndarray, " n_z"],
        *,
        derivative: bool,
    ) -> Num[np.ndarray, "n_z n_omega"]:
        """Sum the active species in component order without promoting real data.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s.
        z : numpy.ndarray
            Positions in m.
        derivative : bool
            Select the frequency derivative instead of susceptibility.

        Returns
        -------
        numpy.ndarray
            Susceptibility or its derivative, in dimensionless units or s.
        """
        result = None
        for component, density in zip(self.components, self.density(z), strict=True):
            if np.all(density == 0):
                continue
            model = component.species.susceptibility
            sampled = (
                model.dsusceptibility_domega(omega)
                if derivative
                else model.susceptibility(omega)
            )
            term = sampled[np.newaxis, :] * density[:, np.newaxis]
            result = term if result is None else result + term
        return np.zeros((z.size, omega.size)) if result is None else result

    def susceptibility(
        self, omega: Float[np.ndarray, " n_omega"], z: Float[np.ndarray, " n_z"]
    ) -> Num[np.ndarray, "n_z n_omega"]:
        """Sum species susceptibilities before forming any refractive index.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, within each active species' support.
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Dimensionless susceptibility, shape (n_z, n_omega); complex if absorbing.
        """
        return self._optical_sum(omega, z, derivative=False)

    def dsusceptibility_domega(
        self, omega: Float[np.ndarray, " n_omega"], z: Float[np.ndarray, " n_z"]
    ) -> Num[np.ndarray, "n_z n_omega"]:
        """Sum the analytic species susceptibility derivatives.

        Parameters
        ----------
        omega : numpy.ndarray
            Angular frequencies in rad/s, within each active species' support.
        z : numpy.ndarray
            Positions in m.

        Returns
        -------
        numpy.ndarray
            Derivative in s, shape (n_z, n_omega).
        """
        return self._optical_sum(omega, z, derivative=True)

    def gamma3_path(self, wavelength: float | None = None) -> ZProfile:
        """Compose the density-weighted third-order susceptibility path.

        Parameters
        ----------
        wavelength : float or None, optional
            The vacuum carrier wavelength in m at which a species' optmat Kerr model
            is evaluated (:meth:`~solprop.physdata.species.Species.kerr_coefficient`);
            required when one has such a model, ignored by fixed coefficients.

        Returns
        -------
        ZProfile
            Sum of density times gamma3, in m^2/V^2.

        Raises
        ------
        MaterialError
            If any component lacks third-order response data, or an optmat model
            needs a carrier and none is given.
        """
        paths = []
        for component in self.components:
            coefficient = component.species.kerr_coefficient(wavelength)
            if coefficient is None:
                raise MaterialError(
                    f"no Kerr data for species {component.species.name}"
                )
            paths.append(Scaled(component.density, coefficient))
        return Sum(tuple(paths))

    def with_densities(self, mapping: Mapping[str, float | ZProfile]) -> "Mixture":  # noqa: UP037 - see fill
        """Replace labelled density paths while preserving component order and data.

        Parameters
        ----------
        mapping : collections.abc.Mapping
            Labels to density paths or constant densities in m^-3.

        Returns
        -------
        Mixture
            Updated independent profiles; omitted labels keep their paths.

        Raises
        ------
        MaterialError
            If a label is unknown or a supplied density is invalid.
        """
        for label in mapping:
            self[label]
        return Mixture.of(
            *(
                replace(c, density=as_profile(mapping[c.label]))
                if c.label in mapping
                else c
                for c in self.components
            ),
            temperature=self.temperature,
        )

    def with_fill(
        self,
        fractions: Mapping[str, float] | None = None,
        pressure: Pressure | None = None,
        *,
        length: float | None = None,
        floor: float | None = None,
        radius: float | ZProfile | None = None,
        vacuum_end: str | None = None,
    ) -> "Mixture":  # noqa: UP037 - see fill
        """Replace fill fractions or pressure without replacing species or labels.

        Parameters
        ----------
        fractions : collections.abc.Mapping or None, optional
            New mole fraction for every label; None retains the current fractions.
        pressure : float or sequence or ZProfile or None, optional
            New total pressure in bar; None retains the shared path.
        length : float or None, optional
            Length in m required for new endpoint pressures.
        floor : float or None, optional
            New pressure-gradient floor in bar.
        radius : float, ZProfile or None, optional
            Core radius in m for a new steady-flow gradient, as in :meth:`fill`.
        vacuum_end : {'choked', 'floor'} or None, optional
            How a new gradient's vacuum end is treated, as in :meth:`fill`.

        Returns
        -------
        Mixture
            Updated populations sharing the supplied total pressure.

        Raises
        ------
        MaterialError
            If fractions or pressure are missing for an independent mixture, or new
            fractions are given beside a retained fill whose choked end or flow
            weight was computed for the old composition.
        """
        if fractions is not None and set(fractions) != set(self.labels):
            raise MaterialError(f"fill fractions must name exactly {self.labels}")
        weights = (
            self.fractions
            if fractions is None
            else _fractions(tuple(fractions[label] for label in self.labels))
        )
        if pressure is None and (radius is not None or vacuum_end is not None):
            raise MaterialError(
                "radius and vacuum_end shape a new pressure fill; pass pressure= "
                "beside them, since the retained shared path carries its own"
            )
        if (
            pressure is None
            and fractions is not None
            and isinstance(self.shared, PressureGradient)
            and (any(self.shared.choked_ends) or self.shared.weight is not None)
        ):
            what = (
                "a choked end computed for the old composition's viscosity and sound "
                "speed"
                if any(self.shared.choked_ends)
                else "a flow weight eta(T) T computed for the old composition's "
                "viscosity"
            )
            raise MaterialError(
                f"the retained fill has {what}: pass pressure= (with radius= and "
                "length= as before) so that the steady flow is built again for the "
                "new fractions"
            )
        path = (
            self.shared
            if pressure is None or weights is None
            else _pressure_path(
                pressure,
                length,
                floor,
                radius=radius,
                vacuum_end=vacuum_end,
                composition={
                    c.species: weight
                    for c, weight in zip(self.components, weights, strict=True)
                },
                temperature=self.temperature,
            )
        )
        if weights is None or path is None:
            raise MaterialError(
                "with_fill needs fractions and pressure for independent profiles"
            )
        components = tuple(
            replace(
                c,
                density=DensityPath(c.species, Scaled(path, weight), self.temperature),
            )
            for c, weight in zip(self.components, weights, strict=True)
        )
        return Mixture(components, self.temperature, path, weights)

    def at_parameters(self, values: Mapping[str, float]) -> "Mixture":  # noqa: UP037 - see fill
        """Sample pressure or labelled densities without changing material data.

        Parameters
        ----------
        values : collections.abc.Mapping
            Total pressure under 'pressure', or densities under 'density:label'.

        Returns
        -------
        Mixture
            Constant sampled populations with the same labels and species.

        Raises
        ------
        MaterialError
            If a parameter is unknown or incompatible with the fill: a total
            pressure beside a temperature that varies along z, where one pressure
            does not fix the densities, is refused by name.
        """
        if "pressure" in values:
            if len(values) != 1 or self.shared is None:
                raise MaterialError(
                    "pressure sampling requires a shared fill and no density parameters"
                )
            if isinstance(self.temperature, ZProfile):
                raise MaterialError(
                    "pressure sampling needs one temperature, and this fill's "
                    f"temperature varies along z ({self.temperature!r}): sample its "
                    "populations by 'density:<label>' instead"
                )
            # Within its sampled range, keep each density spline's arithmetic.
            # Outside that range evaluate the equation of state: extrapolating a
            # constant fill's nearly coincident knots is catastrophically unstable.
            p = values["pressure"]
            if not math.isfinite(p) or p < 0:
                raise MaterialError("sampled pressure must be finite and non-negative")
            if self.fractions is None:
                raise MaterialError(
                    "pressure sampling requires declared fill fractions"
                )
            kelvin = self.temperature
            components = []
            for c, weight in zip(self.components, self.fractions, strict=True):
                spline = (
                    c.density._spline if isinstance(c.density, DensityPath) else None
                )
                rho = (
                    float(spline(p * weight))
                    if spline is not None and spline.x[0] <= p * weight <= spline.x[-1]
                    else c.species.equation_of_state.density(p * weight, kelvin)
                )
                components.append(
                    replace(
                        c, density=Constant(0.0 if p * weight == 0 else max(rho, 0.0))
                    )
                )
            return Mixture(tuple(components), self.temperature)
        mapping = {}
        for name, value in values.items():
            if not name.startswith("density:"):
                raise MaterialError(f"unknown mixture parameter {name!r}")
            mapping[name.removeprefix("density:")] = value
        return self.with_densities(mapping)

    def serialise(self) -> dict[str, object]:
        """Describe the composition and sampled profiles in explicit units.

        Returns
        -------
        dict
            Labels, species names, temperature in K (a number, or the sampled record
            of a path along z) and density samples in m^-3, and the named
            composition where there is one. Breakpoints accompany every path; no
            callable is claimed reconstructible.
        """
        record: dict[str, object] = {
            "temperature_K": _path_record(self.temperature)
            if isinstance(self.temperature, ZProfile)
            else self.temperature,
            "labels": list(self.labels),
            "fractions": self.fractions,
            "components": [
                {
                    "label": c.label,
                    "species": c.species.name,
                    "citation": c.species.citation,
                    "preionisation": c.preionisation
                    if not isinstance(c.preionisation, ZProfile)
                    else _path_record(c.preionisation),
                    "density": _path_record(c.density),
                }
                for c in self.components
            ],
            "shared_pressure": None
            if self.shared is None
            else _path_record(self.shared),
        }
        if self.composition is not None:
            record["composition"] = self.composition
        return record


def _path_record(path: ZProfile) -> dict[str, object]:
    """Preserve a profile's samples and declarations without serialising a callable.

    Parameters
    ----------
    path : ZProfile
        The profile, with its own value units and z in m.

    Returns
    -------
    dict
        Sampled values, positions in m and breakpoint declarations.
    """
    start, end = path.z_span
    positions = np.unique(
        np.concatenate(
            (
                path.breakpoints,
                np.linspace(start, end, 257)
                if math.isfinite(end)
                else np.array([start]),
            )
        )
    )
    return {
        "kind": type(path).__name__,
        "z_m": positions.tolist(),
        "values": path(positions).tolist(),
        "breakpoints_m": path.breakpoints.tolist(),
        "is_constant": path.is_constant,
        "derivative_is_analytic": path.derivative_is_analytic,
    }
