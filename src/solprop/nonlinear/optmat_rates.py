# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Ionisation rates from optmat's rate models, on solprop's rate protocol (SEP 3).

solprop keeps no ionisation-rate formula of its own for an optmat-backed run (SEP 3,
D1): :class:`OptmatRate` adapts one of optmat's rate models to the
:class:`~solprop.nonlinear.ionisation.IonisationRate` protocol, which the translation
layer tabulates into the existing :class:`~solprop.translate.ionisation.RateTable`
unchanged. An atom's rate comes from ``optmat.gas.ionisation_rates`` and a
molecule's from ``optmat.gas.molecular_ionisation_rates``; each refuses the other's
species. The adapter converts and records; the physics is optmat's.

What the adapter adds, and nothing else:

- a **threshold**: the field below which optmat's instantaneous rate is under a
  declared negligible rate (:data:`NEGLIGIBLE_RATE`), found by bisection in the
  logarithm of the field. Below it the rate is exactly zero, which the plasma loss
  term ``I W (1 - f) / E`` relies on;
- the **ceiling** :data:`~solprop.nonlinear.ionisation.MAX_RATE`, a numerical safety
  cap that no physical rate reaches;
- the **energy per event** the plasma loss pays (SEP 3, D8). optmat's rate-weighted
  energy ``I(E) = L(E) / W(E)`` moves with the field between the channels of a noble
  gas; solprop's kernel carries one energy per population, so the loss takes the
  constant ``I(E_ref)`` at a reference field -- the input's peak field, which the
  capillary driver binds (:func:`bind_reference_field`) -- clipped to the tabulated
  fields, and the largest relative departure of ``I(E)`` from it over the tabulated
  fields at which the rate is not negligible is recorded as its bound. A one-channel
  model, every molecular one among them, is exact. Without a reference field the
  energy is the weak-field limit, the lowest channel's;
- the **maximum field**: optmat warns at every evaluation above it, and a rate table
  routinely extends there. Above it optmat's instantaneous rate is outside its fit and
  turns over, falling towards zero within a decade of field, so the adapter holds the
  rate at its value at the maximum field and a table is built up to it
  (:func:`solprop.translate.ionisation.build_rate_table`). A direct evaluation above
  it is recorded in the run's material log rather than warned at every call;
  :meth:`OptmatRate.check_input` warns when the input's own peak exceeds it, and
  whether a run reached it is :func:`peak_field_report`'s to say, after the run.

The manual page is ``docs/guide/media/ionisation.md``, section "Rates from optmat".
"""

import dataclasses
import logging
import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import KW_ONLY, dataclass, field
from typing import Final, cast

import numpy as np
from jaxtyping import Float
from numpy.typing import ArrayLike

from solprop.exceptions import MaterialError, MaterialRangeWarning
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.ionisation import MAX_RATE, RateConvention
from solprop.nonlinear.responses import PlasmaResponse
from solprop.physdata import _optmat
from solprop.physdata.gases import Gas
from solprop.physdata.selection import MaterialRecord, MaterialSelection
from solprop.physdata.species import (
    Species,
    built_in_population,
    ionisation_potential,
)

__all__ = [
    "ENERGY_LOSS",
    "LINEAR_POLARISATION_TOLERANCE",
    "NEGLIGIBLE_RATE",
    "OptmatRate",
    "PeakFieldReport",
    "bind_reference_field",
    "peak_field_report",
    "selected_rate",
]

logger = logging.getLogger(__name__)

NEGLIGIBLE_RATE: Final[float] = 1e-30
"""Rate in s^-1 below which an optmat rate is taken as exactly zero.

Over a picosecond it ionises one atom in 1e42, and in a 30 ps window one in 3e40: no
statistic a run reports can see it. It is the same floor the envelope plasma's cycle
tables compare against (``solprop.translate.envelope_plasma``).
"""

ENERGY_LOSS: Final = "peak-weighted"
"""The plasma energy-loss convention: the rate-weighted energy at a reference field."""

LINEAR_POLARISATION_TOLERANCE: Final[float] = 1e-6
"""Circular share ``|S3| / S0`` at the peak below which a field counts as linear.

The analytic signal of a linearly polarised field has ``S3 = 0`` to rounding; a
millionth of circular polarisation changes no rate this package carries.
"""

_RECOMMENDED: Final = "recommended"
"""The selection name of optmat's recommended model, ``"default"`` in its registry."""

_BOUND_SAMPLES: Final = 257
"""Fields, geometric between the ends of the span, at which the energy bound is taken.

optmat's rate-weighted energy is monotone in the field over every model's span (it
moves from the lowest channel's energy towards the weighted mean of the channels), so
the extremes are at the ends, which are sampled; the interior samples guard a model
for which that is not so.
"""

_BISECTION_TOLERANCE: Final = 1e-12
"""Relative width in field at which the threshold and peak-field bisections stop."""

_TABLE_TOLERANCE: Final = 1e-5
"""Relative margin under the held rate at which a recorded peak counts as reaching it.

Ten times a rate table's default tolerance (``DEFAULT_RTOL``, 1e-6), against which the
table's held value can sit below the direct rate at the maximum field.
"""

_CEILING_ROUNDING: Final = 1e-12
"""Relative margin above the maximum field within which a field counts as at it.

A rate table's knots are ``exp`` of uniform logarithms, so its last one rounds to
within a few ulps (1e-16 relative) of the maximum field it was built to.
"""

_LOWEST_FIELD: Final = 1e-3
"""Field in V/m below which a threshold search gives up: no strong-field rate is
above a negligible value there, so a model that is is refused rather than tabulated."""


def _validate_positive(name: str, value: float) -> float:
    """Return a positive finite number, or refuse it by name.

    Parameters
    ----------
    name : str
        The argument, for the message.
    value : float
        The number.

    Returns
    -------
    float
        The number as a float.

    Raises
    ------
    MaterialError
        If it is not positive and finite.
    """
    if isinstance(value, bool) or not (math.isfinite(value) and value > 0):
        msg = f"OptmatRate: {name} must be positive and finite, got {value!r}"
        raise MaterialError(msg)
    return float(value)


def _is_rate_model(value: object) -> bool:
    """Return whether an object is an optmat rate model.

    Parameters
    ----------
    value : object
        Anything.

    Returns
    -------
    bool
        True for an object with an identity, an instantaneous rate, an energy per
        event, a maximum field and a wavelength.
    """
    return all(
        hasattr(value, name)
        for name in (
            "identity",
            "instantaneous_rate",
            "energy_per_event",
            "maximum_field",
            "wavelength",
        )
    )


def _resolve(name: str, wavelength: float, form: str) -> object:
    """Resolve a species name and a model name to optmat's rate model.

    Parameters
    ----------
    name : str
        The species: an atom of ``ionisation_rate_model`` or a molecule of
        ``molecular_rate_model``.
    wavelength : float
        Vacuum carrier wavelength in m.
    form : str
        ``"recommended"`` or a model name of the species' registry.

    Returns
    -------
    object
        The optmat model.

    Raises
    ------
    MaterialError
        If optmat has no rate for the species, or the form is not one of its
        family's, naming the family that serves it.
    """
    from optmat.exceptions import OptmatError  # noqa: PLC0415 - optmat is lazy
    from optmat.gas.ionisation_rates import (  # noqa: PLC0415 - optmat is lazy
        ionisation_rate_coverage,
        ionisation_rate_model,
    )
    from optmat.gas.molecular_ionisation_rates import (  # noqa: PLC0415 - as above
        molecular_rate_coverage,
        molecular_rate_model,
    )

    registry = "default" if form == _RECOMMENDED else form
    # Each family's model names and the gases they cover, read without building any
    # model: only the one asked for is built.
    atomic = ionisation_rate_coverage()
    molecular = molecular_rate_coverage()
    atoms = {gas.value for gas in atomic["default"]}
    molecules = {gas.value for gas in molecular["default"]}
    if name in atoms:
        family, models, other, build = (
            "atomic rates (ionisation_rate_model)",
            atomic,
            molecular,
            ionisation_rate_model,
        )
    elif name in molecules:
        family, models, other, build = (
            "molecular rates (molecular_rate_model)",
            molecular,
            atomic,
            molecular_rate_model,
        )
    else:
        msg = (
            f"optmat has no ionisation rate for {name!r}: its atoms are "
            f"{sorted(atoms)} and its molecules {sorted(molecules)}. A custom species "
            "takes OptmatRate(IonisationRateModel.from_parameters(...)) or "
            "OptmatRate(MolecularRateModel.from_parameters(...)), or a rate of its own"
        )
        raise MaterialError(msg)
    if registry not in models:
        served = (
            f"; {form!r} is a form of the other family, which refuses {name}"
            if registry in other
            else ""
        )
        msg = (
            f"{name}: {form!r} is not one of optmat's {family}, which are "
            f"{sorted(['recommended', *(m for m in models if m != 'default')])}{served}"
        )
        raise MaterialError(msg)
    try:
        return build(name, wavelength, registry)
    except OptmatError as error:
        msg = f"{name}: optmat refuses the rate model {form!r}: {error}"
        raise MaterialError(msg) from error


@contextmanager
def _quiet(*, reemit: bool) -> Iterator[list[warnings.WarningMessage]]:
    """Evaluate optmat with its range warnings captured, and pass on any other.

    Parameters
    ----------
    reemit : bool
        Whether the caller handles the captured range warnings itself (the list is
        returned) rather than dropping them.

    Yields
    ------
    list of warnings.WarningMessage
        optmat's ``ModelRangeWarning`` raised inside, for the caller.
    """
    from optmat.exceptions import ModelRangeWarning  # noqa: PLC0415 - optmat is lazy

    ranged: list[warnings.WarningMessage] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield ranged
    for item in caught:
        if issubclass(item.category, ModelRangeWarning):
            if reemit:
                ranged.append(item)
            continue
        warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)


def _bisect(
    above: Callable[[float], bool], low: float, high: float
) -> tuple[float, float]:
    """Narrow a bracket in the logarithm of the field.

    Parameters
    ----------
    above : callable
        True where the field is at or above the sought crossing.
    low, high : float
        A bracket in V/m, ``above(low)`` false and ``above(high)`` true.

    Returns
    -------
    tuple of float
        The narrowed bracket, ``high / low - 1`` below :data:`_BISECTION_TOLERANCE`.
    """
    # Each step halves the bracket's logarithmic width; from a few decades about
    # 55 steps reach the tolerance, far above the spacing of adjacent doubles.
    while high / low - 1.0 > _BISECTION_TOLERANCE:
        middle = math.sqrt(low * high)
        if above(middle):
            high = middle
        else:
            low = middle
    return low, high


@dataclass(frozen=True, eq=False)
class OptmatRate:
    r"""Take an ionisation rate from optmat, on solprop's rate protocol.

    Built from a species name and a carrier wavelength -- an atom through optmat's
    ``ionisation_rate_model``, a molecule through ``molecular_rate_model`` -- or from
    an optmat rate model taken as it is: one made by ``with_parameters`` (an
    override, status ``user``) or by ``from_parameters`` (a custom species). The
    rate is optmat's instantaneous rate of the field's magnitude,

    .. math::

        W(E) = \min\left(W_\text{optmat}(\min(|E|, E_m)), W_\text{max}\right)
        \quad (|E| \ge E_t), \qquad W(E) = 0 \quad (|E| < E_t),

    with the threshold :math:`E_t` the field at which optmat's rate reaches
    `negligible_rate` and :math:`E_m` the model's maximum field, above which the rate
    is held: optmat's rate is outside its fit there, and turns over. optmat's rates
    are for linear polarisation; along an elliptical field the instantaneous rate is a
    quasi-static approximation (SEP 3, D12).

    Attributes
    ----------
    source : str or object
        A species name optmat curates a rate for (``"Ar"``, ``"N2"``, a
        :class:`~solprop.physdata.gases.Gas`), or an optmat rate model.
    wavelength : float or None
        Vacuum carrier wavelength in m, positive and finite; required with a name.
        With a model it is the model's own, and a different one is refused.
    model : str, optional
        ``"recommended"`` (optmat's default model of the species), or a named
        literature form: ``"ppt_ilkov"``, ``"ppt_larochelle"``, ``"ppt_popruzhenko"``
        for an atom; ``"moadk_zhao_2010"``, ``"moadk_tong_2002"``,
        ``"moadk_zhao_2017"``, ``"ppt_talebpour"``, ``"ppt_talebpour_ilkov"``,
        ``"adk_hydrogenic"`` for a molecule. Only with a name.
    reference_field : float or None, optional
        Field amplitude in V/m at which the constant energy per event is taken: the
        input's peak, which the capillary driver binds. It is clipped to the span
        the energy is defined over. ``None`` takes the weak-field limit, the lowest
        channel's energy.
    field_range : tuple of float or None, optional
        The field amplitudes in V/m the run tabulates the rate over; the energy's
        bound is taken over its part above the threshold. ``None`` takes the span from
        the threshold to the model's maximum field.
    negligible_rate : float, optional
        Rate in s^-1 below which the rate is zero, :data:`NEGLIGIBLE_RATE` by default.
    maximum : float, optional
        Ceiling on the rate in s^-1.

    Raises
    ------
    MaterialError
        If optmat has no rate for the species or the form is not of its family, a
        model object is given with a model name or another wavelength, or a number
        is not positive and finite.

    Warns
    -----
    MaterialStatusWarning
        Once per run, for a ``provisional`` or ``untested`` model (SF6's).
    MaterialRangeWarning
        Once per run, translating optmat's warning for a carrier below the
        wavelengths its rates were compared at (350 nm).

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import OptmatRate
    >>> rate = OptmatRate("Ar", 800e-9)
    >>> rate.convention, float(rate(0.0)), float(rate(0.5 * rate.threshold))
    ('instantaneous-isotropic', 0.0, 0.0)
    >>> rate.material_records[0].kind
    'ionisation_rate'
    """

    source: object
    wavelength: float | None = None
    _: KW_ONLY
    model: str = _RECOMMENDED
    reference_field: float | None = None
    field_range: tuple[float, float] | None = None
    negligible_rate: float = NEGLIGIBLE_RATE
    maximum: float = MAX_RATE
    _resolved: object = field(init=False, repr=False, default=None)
    _record: MaterialRecord | None = field(init=False, repr=False, default=None)
    _threshold: float = field(init=False, repr=False, default=math.nan)
    _span: tuple[float, float] = field(init=False, repr=False, default=(0.0, 0.0))
    _reference: float | None = field(init=False, repr=False, default=None)
    _energy: float = field(init=False, repr=False, default=math.nan)
    _bound: float = field(init=False, repr=False, default=math.nan)

    def __post_init__(self) -> None:
        """Resolve the model, its threshold and the energy per event."""
        if isinstance(self.source, Gas):
            object.__setattr__(self, "source", self.source.value)
        _validate_positive("negligible_rate", self.negligible_rate)
        _validate_positive("maximum", self.maximum)
        if self.reference_field is not None:
            _validate_positive("reference_field", self.reference_field)
        if self.field_range is not None:
            low, high = (float(v) for v in self.field_range)
            if not (math.isfinite(high) and 0.0 < low < high):
                msg = (
                    "OptmatRate: field_range must be two finite amplitudes in V/m "
                    f"with 0 < low < high, got {self.field_range!r}"
                )
                raise MaterialError(msg)
            object.__setattr__(self, "field_range", (low, high))
        with _quiet(reemit=True) as ranged:
            resolved = self._resolve_model()
            object.__setattr__(self, "_resolved", resolved)
            object.__setattr__(self, "_record", _optmat.record(resolved))
            threshold = self._find_threshold()
        object.__setattr__(self, "_threshold", threshold)
        self._translate_warnings(ranged)
        _optmat.check_status(resolved, self.species_identity)
        self._bind_energy()

    def _resolve_model(self) -> object:
        """Return the optmat model a name resolves to, or the model given.

        Returns
        -------
        object
            The optmat rate model.

        Raises
        ------
        MaterialError
            As the class does.
        """
        if isinstance(self.source, str):
            if self.wavelength is None:
                msg = (
                    f"OptmatRate({self.source!r}) needs the carrier wavelength in m "
                    "its rate is evaluated at"
                )
                raise MaterialError(msg)
            wavelength = _validate_positive("wavelength", self.wavelength)
            return _resolve(self.source, wavelength, self.model)
        if not _is_rate_model(self.source):
            msg = (
                "OptmatRate takes a species name or an optmat rate model, got a "
                f"{type(self.source).__name__}"
            )
            raise MaterialError(msg)
        given = _optmat.untyped(self.source)
        if self.model != _RECOMMENDED:
            msg = (
                f"OptmatRate: model={self.model!r} names a form of optmat's registry; "
                "a model object is taken as it is"
            )
            raise MaterialError(msg)
        own = float(given.wavelength)
        if self.wavelength is not None and not math.isclose(
            float(self.wavelength), own, rel_tol=1e-12
        ):
            msg = (
                f"OptmatRate: the model is at {own:.6g} m and wavelength="
                f"{self.wavelength:.6g} m was given; move it with "
                "model.at_wavelength(...)"
            )
            raise MaterialError(msg)
        object.__setattr__(self, "wavelength", own)
        return self.source

    @property
    def optmat_model(self) -> _optmat.Untyped:
        """Return the optmat rate model the adapter evaluates.

        Returns
        -------
        Any
            The model, for direct comparison with its own methods. Typed loosely, as
            the bridge types every optmat object (``solprop.physdata._optmat``).
        """
        return self._model()

    def _model(self) -> _optmat.Untyped:
        """Return the model for attribute access.

        Returns
        -------
        Any
            The resolved optmat model.
        """
        return _optmat.untyped(self._resolved)

    def _raw(
        self, field_amplitude: Float[np.ndarray, " n_field"]
    ) -> Float[np.ndarray, " n_field"]:
        """Return optmat's instantaneous rate, its range warnings dropped.

        Parameters
        ----------
        field_amplitude : numpy.ndarray
            Non-negative fields in V/m, shape ``(n_field,)``.

        Returns
        -------
        numpy.ndarray
            Rates in s^-1, shape ``(n_field,)``.
        """
        with _quiet(reemit=False):
            return np.asarray(
                self._model().instantaneous_rate(field_amplitude), dtype=np.float64
            )

    def _find_threshold(self) -> float:
        """Return the field at which the rate reaches the negligible rate.

        Returns
        -------
        float
            The smallest field in V/m of the final bracket, at which the rate is at
            or above `negligible_rate`.

        Raises
        ------
        MaterialError
            If the rate is negligible even at the barrier-suppression field, or not
            negligible at any field the search reaches.
        """
        model = self._model()
        negligible = self.negligible_rate

        def above(value: float) -> bool:
            """Return whether the rate at a field is at least the negligible rate.

            Parameters
            ----------
            value : float
                Field in V/m.

            Returns
            -------
            bool
                The comparison.
            """
            return float(model.instantaneous_rate(value)) >= negligible

        high = float(model.barrier_suppression_field)
        if not above(high):
            msg = (
                f"{self.species_identity}: optmat's rate is below the negligible rate "
                f"{negligible:g} 1/s even at its barrier-suppression field "
                f"{high:.4g} V/m; there is nothing to tabulate"
            )
            raise MaterialError(msg)
        low = high
        while above(low):
            if low < _LOWEST_FIELD:
                msg = (
                    f"{self.species_identity}: optmat's rate is above the negligible "
                    f"rate {negligible:g} 1/s at {low:.3g} V/m; it has no threshold "
                    "field to tabulate from"
                )
                raise MaterialError(msg)
            high, low = low, low / 10.0
        return _bisect(above, low, high)[1]

    def _translate_warnings(self, ranged: Sequence[warnings.WarningMessage]) -> None:
        """Raise optmat's warnings at resolution once per run, and record them.

        At resolution the model is evaluated below its barrier-suppression field, so
        a range warning there is about the carrier (below 350 nm), not the field.

        Parameters
        ----------
        ranged : sequence of warnings.WarningMessage
            The range warnings optmat raised while the model was resolved.
        """
        record = self.material_records[0]
        # Each evaluation of the search repeats the same warning; one is enough.
        for text in dict.fromkeys(str(item.message) for item in ranged):
            _optmat.record_outside(self._resolved, self.species_identity, (text,))
            _optmat.warn_once(
                ("rate-range", record.digest, text),
                f"{self.species_identity}: optmat's {record.name}: {text}. It is used "
                "as selected, and recorded",
                MaterialRangeWarning,
            )

    def _energy_span(self) -> tuple[float, float]:
        """Return the fields the constant energy and its bound are defined over.

        Returns
        -------
        tuple of float
            The tabulated fields above the threshold and up to the maximum field,
            above which the rate is held (:meth:`__call__`); the threshold to the
            maximum field without a `field_range`. A span the threshold leaves empty
            is the threshold alone.
        """
        threshold = self._threshold
        ceiling = max(threshold, float(self._model().maximum_field))
        if self.field_range is None:
            return threshold, ceiling
        low, high = self.field_range
        return max(low, threshold), min(max(high, threshold), ceiling)

    def _energies(
        self, fields: Float[np.ndarray, " n_field"]
    ) -> Float[np.ndarray, " n_field"]:
        """Return optmat's rate-weighted energy per event, its range warnings dropped.

        Parameters
        ----------
        fields : numpy.ndarray
            Fields in V/m, shape ``(n_field,)``.

        Returns
        -------
        numpy.ndarray
            Energies in J, shape ``(n_field,)``.
        """
        with _quiet(reemit=False):
            return np.asarray(self._model().energy_per_event(fields), dtype=np.float64)

    def energy_per_event(
        self, field_amplitude: ArrayLike | float
    ) -> Float[np.ndarray, " *shape"]:
        """Return optmat's rate-weighted energy per event, ``L(E) / W(E)``.

        Parameters
        ----------
        field_amplitude : array_like or float
            Field amplitude in V/m; the sign is ignored.

        Returns
        -------
        numpy.ndarray
            Energy in J, with the shape of the field; the lowest channel's at zero
            field. Above the maximum field it is held at its value there, as the
            rate is.
        """
        values = np.asarray(np.abs(np.asarray(field_amplitude, dtype=np.float64)))
        held = np.minimum(np.atleast_1d(values).ravel(), self.maximum_field)
        return self._energies(held).reshape(values.shape)

    def _bind_energy(self) -> None:
        """Fix the constant energy per event and its bound over the span."""
        low, high = self._energy_span()
        object.__setattr__(self, "_span", (low, high))
        if self.reference_field is None:
            reference = None
            energy = float(self._energies(np.zeros(1))[0])
        else:
            reference = min(max(float(self.reference_field), low), high)
            energy = float(self._energies(np.array([reference]))[0])
        samples = (
            np.geomspace(low, high, _BOUND_SAMPLES) if high > low else np.array([low])
        )
        bound = float(np.max(np.abs(self._energies(samples) / energy - 1.0)))
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "_energy", energy)
        object.__setattr__(self, "_bound", bound)

    def bound(
        self, reference_field: float, field_range: tuple[float, float] | None = None
    ) -> "OptmatRate":  # noqa: UP037 - the hook resolves class names eagerly
        """Return the rate with its energy taken at a reference field.

        Parameters
        ----------
        reference_field : float
            Field amplitude in V/m, the input's peak.
        field_range : tuple of float or None, optional
            The field amplitudes in V/m the run tabulates; ``None`` keeps this rate's.

        Returns
        -------
        OptmatRate
            The same model, its energy per event at the reference field, clipped to
            the tabulated fields above the threshold, and its bound over them.

        Examples
        --------
        >>> from solprop.nonlinear.optmat_rates import OptmatRate
        >>> rate = OptmatRate("Ar", 800e-9).bound(2.7e10, (2.7e7, 8.1e11))
        >>> rate.effective_reference_field
        27000000000.0
        """
        return dataclasses.replace(
            self,
            reference_field=reference_field,
            field_range=self.field_range if field_range is None else field_range,
        )

    @property
    def threshold(self) -> float:
        """Return the field amplitude below which the rate is exactly zero.

        Returns
        -------
        float
            Field in V/m at which optmat's rate reaches `negligible_rate`, to a
            relative 1e-12 from above.
        """
        return self._threshold

    @property
    def citation(self) -> str:
        """Return the model's references and the adapter's settings.

        Returns
        -------
        str
            The model's citation as optmat gives it, its name and digest, the
            threshold and the energy convention.
        """
        record = self.material_records[0]
        reference = (
            "the weak-field limit"
            if self._reference is None
            else f"{self._reference:.4g} V/m"
        )
        return (
            f"{self._model().citation}; optmat {record.name} (digest "
            f"{record.digest[:12]}, {record.status}); zero below {self._threshold:.4g} "
            f"V/m, where it falls under {self.negligible_rate:g} 1/s; energy per event "
            f"{ENERGY_LOSS} at {reference}"
        )

    @property
    def parameters(self) -> tuple[tuple[str, float | int | bool | str], ...]:
        """Return every setting that changes the rate, name-sorted and hashable.

        Returns
        -------
        tuple
            The model's identity digest, the species, the carrier, the threshold and
            its negligible rate, the ceiling and the energy convention.
        """
        settings: dict[str, float | int | bool | str] = {
            "model": "optmat-rate",
            "digest": self.material_records[0].digest,
            "species": self.species_identity,
            "wavelength": float(self._model().wavelength),
            "threshold": self._threshold,
            "negligible_rate": self.negligible_rate,
            "maximum": self.maximum,
            "energy_loss": ENERGY_LOSS,
        }
        return tuple(sorted(settings.items()))

    @property
    def convention(self) -> RateConvention:
        """Return the optical-cycle convention of the rate.

        Returns
        -------
        str
            ``'instantaneous-isotropic'``: optmat's instantaneous rate of the field's
            magnitude, which an envelope averages over the polarisation ellipse.
        """
        return "instantaneous-isotropic"

    @property
    def species_identity(self) -> str:
        """Return the species the model describes.

        Returns
        -------
        str
            The gas's name, or a custom species' name as its model gives it.
        """
        gas = self._model().gas
        return str(getattr(gas, "value", gas))

    @property
    def ionisation_potential(self) -> float:
        """Return the constant energy per ionisation event the plasma loss pays.

        Returns
        -------
        float
            The rate-weighted energy at the reference field, in J, or the
            weak-field limit without one.
        """
        return self._energy

    @property
    def plasma_energy(self) -> float:
        """Return the energy per event a plasma response pays, the same constant.

        Returns
        -------
        float
            :attr:`ionisation_potential`, J; declaring it is what makes a response
            take it (:func:`solprop.nonlinear.ionisation.bind_rate_energy`).
        """
        return self._energy

    @property
    def energy_bound(self) -> float:
        """Return the bound of the constant energy over the span it is defined on.

        Returns
        -------
        float
            ``max |I(E) / I_ref - 1|`` over the tabulated fields above the threshold
            (:attr:`energy_span`), dimensionless; zero for a one-channel model.
        """
        return self._bound

    @property
    def energy_span(self) -> tuple[float, float]:
        """Return the fields the bound is taken over.

        Returns
        -------
        tuple of float
            Lowest and highest field in V/m.
        """
        return self._span

    @property
    def effective_reference_field(self) -> float | None:
        """Return the field the constant energy is taken at, after clipping.

        Returns
        -------
        float or None
            Field in V/m, or ``None`` for the weak-field limit.
        """
        return self._reference

    @property
    def energy_kind(self) -> str:
        """Return the kind of the energies the rate's exponent uses.

        Returns
        -------
        str
            ``'adiabatic'``, ``'vertical'`` or ``'not stated'``, as optmat states it.
        """
        return str(self._model().energy_kind)

    @property
    def status(self) -> str:
        """Return the model's optmat status.

        Returns
        -------
        str
            ``'validated'``, ``'provisional'``, ``'untested'``, ``'comparison'`` or
            ``'user'``.
        """
        return self.material_records[0].status

    @property
    def maximum_field(self) -> float:
        """Return the largest field at which optmat gives the rate without a warning.

        Returns
        -------
        float
            Field in V/m.
        """
        return float(self._model().maximum_field)

    @property
    def uncertainty_factor(self) -> float:
        """Return the factor within which optmat expects the rate to be right.

        Returns
        -------
        float
            Dimensionless; NaN for a literature form or an override, which state none.
        """
        return float(self._model().uncertainty_factor)

    @property
    def material_records(self) -> tuple[MaterialRecord, ...]:
        """Return the record of the optmat model.

        Returns
        -------
        tuple of MaterialRecord
            One record.
        """
        return (cast("MaterialRecord", self._record),)

    def keldysh(self, field_amplitude: float) -> float:
        """Return the model's Keldysh parameter at a field.

        Parameters
        ----------
        field_amplitude : float
            Field amplitude in V/m, positive.

        Returns
        -------
        float
            The Keldysh parameter of the model's lowest channel, dimensionless.
        """
        return float(self._model().keldysh(field_amplitude))

    def manifest_entry(self) -> dict[str, object]:
        """Return the rate's entry in a run's material manifest.

        Returns
        -------
        dict
            The species, model and digest, the threshold policy, the energy
            convention with its reference field, constant and bound, the maximum
            field, the uncertainty factor and the polarisation approximation.
        """
        record = self.material_records[0]
        uncertainty = self.uncertainty_factor
        return {
            "species": self.species_identity,
            "model": record.name,
            "digest": record.digest,
            "status": record.status,
            "wavelength": float(self._model().wavelength),
            "threshold": {
                "kind": "negligible-rate",
                "negligible_rate": self.negligible_rate,
                "field": self._threshold,
            },
            "energy_loss": {
                "convention": ENERGY_LOSS,
                "energy_kind": self.energy_kind,
                "reference_field": self._reference,
                "requested_reference_field": self.reference_field,
                "energy": self._energy,
                "bound": self._bound,
                "span": list(self._span),
            },
            "maximum_field": self.maximum_field,
            "uncertainty_factor": None if math.isnan(uncertainty) else uncertainty,
            "polarisation": "linear-polarisation rate; quasi-static along an "
            "elliptical field",
        }

    def check_input(self, peak_field: float, *, circular: float = 0.0) -> None:
        """Warn about an input outside what the rate is defined for, once per run.

        Parameters
        ----------
        peak_field : float
            The input's peak field amplitude in V/m.
        circular : float, optional
            The circular share ``|S3| / S0`` of the field at its peak, in ``[0, 1]``.

        Warns
        -----
        MaterialRangeWarning
            If the peak exceeds the model's maximum field, or the field is not
            linearly polarised (:data:`LINEAR_POLARISATION_TOLERANCE`) and the
            Keldysh parameter at the peak exceeds one, where elliptic and
            orbital-resolved theories depart from the quasi-static average (D12).
        """
        record = self.material_records[0]
        if peak_field > self.maximum_field:
            _optmat.warn_once(
                ("rate-maximum-field", record.digest),
                f"{self.species_identity}: the input's peak field {peak_field:.4g} V/m "
                f"exceeds the maximum field {self.maximum_field:.4g} V/m of optmat's "
                f"{record.name}, beyond which its static correction was not fitted",
                MaterialRangeWarning,
            )
        if circular > LINEAR_POLARISATION_TOLERANCE:
            gamma = self.keldysh(peak_field)
            if gamma > 1.0:
                _optmat.warn_once(
                    ("rate-polarisation", record.digest),
                    f"{self.species_identity}: the input is not linearly polarised "
                    f"(circular share {circular:.3g} at its peak) and its Keldysh "
                    f"parameter there is {gamma:.3g} > 1. optmat's {record.name} is a "
                    "linear-polarisation rate, used quasi-statically along the "
                    "polarisation ellipse, where elliptic and orbital-resolved "
                    "theories depart from it",
                    MaterialRangeWarning,
                )

    def __call__(
        self, field_amplitude: ArrayLike | float
    ) -> Float[np.ndarray, " *shape"]:
        """Return the ionisation rate for a field amplitude.

        Parameters
        ----------
        field_amplitude : array_like or float
            Field amplitude in V/m; the sign is ignored.

        Returns
        -------
        numpy.ndarray
            Rate per atom or molecule in s^-1, with the shape of the field: exactly
            zero below :attr:`threshold`, held at its value at :attr:`maximum_field`
            above that field, and never above :attr:`maximum`.

        Raises
        ------
        MaterialError
            If the field is not finite.
        """
        values = np.asarray(np.abs(np.asarray(field_amplitude, dtype=np.float64)))
        if not np.all(np.isfinite(values)):
            msg = "field amplitude must be finite; a NaN or infinity reached the rate"
            raise MaterialError(msg)
        flat = np.atleast_1d(values).ravel()
        rate = np.zeros_like(flat)
        active = flat >= self._threshold
        if bool(np.any(active)):
            ceiling = self.maximum_field
            fields = flat[active]
            # Above its maximum field optmat's static correction is outside its fit,
            # and its instantaneous rate there turns over and loses smoothness (it
            # falls towards zero within a decade). The rate is held at its value at
            # the maximum field instead, as a rate table holds its last knot.
            rate[active] = self._raw(np.minimum(fields, ceiling))
            # A table's last knot, exp(log(E_max)), can land an ulp above E_max: the
            # margin keeps that from being recorded as a field beyond it. Whether a
            # run reached the maximum field is the post-run diagnostic's to say.
            if float(np.max(fields)) > ceiling * (1.0 + _CEILING_ROUNDING):
                _optmat.record_outside(
                    self._resolved,
                    self.species_identity,
                    (
                        f"fields above its maximum field {ceiling:.4g} V/m, where the "
                        "rate is held at its value there",
                    ),
                )
        np.minimum(rate, self.maximum, out=rate)
        rate[~np.isfinite(rate)] = self.maximum
        return rate.reshape(values.shape)


def selected_rate(
    material: Species,
    selection: MaterialSelection,
    wavelength: float | None,
    *,
    label: str | None = None,
) -> OptmatRate:
    """Return the rate a selection means for a species: optmat's, never solprop's.

    Parameters
    ----------
    material : Species
        A built-in gas, carrying optmat's ionisation energy unchanged.
    selection : MaterialSelection
        The run's selection; its ``ionisation`` names the model (``"recommended"``,
        a literature form) or is an optmat rate model for this species.
    wavelength : float or None
        The run's reference wavelength in m; ``selection.carrier_wavelength`` takes
        precedence when given.
    label : str or None, optional
        The population, for messages.

    Returns
    -------
    OptmatRate
        The rate, unbound to a reference field.

    Raises
    ------
    MaterialError
        If the species is custom (not built in, or with its own ionisation data),
        which needs an explicit rate (D1), or there is no carrier, or a model object
        describes another species.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import selected_rate
    >>> from solprop.physdata import species
    >>> from solprop.physdata.selection import MaterialSelection
    >>> selection = MaterialSelection(ionisation="moadk_zhao_2010")
    >>> selected_rate(species("N2"), selection, 800e-9).status
    'comparison'
    """
    name = label or material.name
    if (
        not built_in_population(material)
        or material.name == Gas.Air.value
        or material.ionisation is None
        or material.ionisation.potential != ionisation_potential(material.name)
    ):
        msg = (
            f"{name} ({material.name}) is a custom species, and under a material "
            "selection its rate comes from optmat or from the caller (SEP 3, D1): "
            "give it explicitly, plasma={label: OptmatRate(IonisationRateModel"
            ".from_parameters(...))} or a rate of its own"
        )
        raise MaterialError(msg)
    carrier = (
        selection.carrier_wavelength
        if selection.carrier_wavelength is not None
        else wavelength
    )
    choice = selection.ionisation
    if isinstance(choice, str):
        if carrier is None:
            msg = f"{name}: the rate needs a carrier wavelength in m"
            raise MaterialError(msg)
        return OptmatRate(material.name, carrier, model=choice)
    rate = OptmatRate(choice, carrier)
    if rate.species_identity != material.name:
        msg = (
            f"{name}: the selected rate model describes {rate.species_identity}, not "
            f"{material.name}"
        )
        raise MaterialError(msg)
    return rate


def bind_reference_field(
    responses: Sequence[object],
    reference_field: float,
    field_range: tuple[float, float] | None,
) -> tuple[object, ...]:
    """Bind every unbound optmat rate of a run's plasma responses to its peak field.

    Parameters
    ----------
    responses : sequence
        The run's responses.
    reference_field : float
        The input's peak field amplitude in V/m.
    field_range : tuple of float or None
        The field amplitudes in V/m the run tabulates the rates over.

    Returns
    -------
    tuple
        The responses, each plasma response whose rate is an :class:`OptmatRate`
        without a reference field rebuilt with the bound rate (and so the bound
        energy); every other response as it was.
    """
    return tuple(
        dataclasses.replace(
            response, rate=response.rate.bound(reference_field, field_range)
        )
        if isinstance(response, PlasmaResponse | EnvelopePlasmaResponse)
        and isinstance(response.rate, OptmatRate)
        and response.rate.reference_field is None
        else response
        for response in responses
    )


@dataclass(frozen=True)
class PeakFieldReport:
    """Report the largest field one population's rate saw in a run.

    Attributes
    ----------
    label : str
        The population.
    species : str
        Its species.
    model : str
        optmat's model name.
    peak_rate : float
        The largest rate the run recorded at its save planes, s^-1.
    largest_field : float or None
        The field in V/m at which the rate is the peak rate; the maximum field, a
        lower bound, when the run reached it (``reached_maximum``); ``None`` when no
        ionisation was recorded.
    reached_maximum : bool
        Whether the run reached the model's maximum field, above which the rate is
        held, so that the largest field is at least that.
    maximum_field : float
        The model's maximum field, V/m.
    energy_at_largest : float or None
        optmat's rate-weighted energy at the largest field, J.
    constant_energy : float
        The constant energy the run's loss paid, J.
    departure : float or None
        ``energy_at_largest / constant_energy - 1``, dimensionless.
    """

    label: str
    species: str
    model: str
    peak_rate: float
    largest_field: float | None
    reached_maximum: bool
    maximum_field: float
    energy_at_largest: float | None
    constant_energy: float
    departure: float | None

    def to_dict(self) -> dict[str, object]:
        """Return the report as plain values, for a run's manifest.

        Returns
        -------
        dict
            Every field.
        """
        return dataclasses.asdict(self)


def peak_field_report(
    rate: OptmatRate,
    peak_rate: float,
    *,
    label: str,
    average: Callable[[float], float] | None = None,
) -> PeakFieldReport:
    """Return the largest field a population saw, from its recorded peak rate.

    The recorded rate rises with the field below the maximum field, so the field is
    its inverse there, found by bisection in the logarithm of the field. A field
    grid's statistic is the tabulated rate of ``|E|``, inverted through the rate
    itself. An envelope's is cycle-averaged, and is inverted through the average the
    run itself took: the caller passes it as `average`, the run's cycle integrals at
    linear polarisation (``solprop.translate.envelope_plasma.cycle_moments``), the
    quadrature its statistic comes from. optmat's own ``cycle_averaged_rate`` would
    not do: above the barrier-suppression field it departs from that average (by
    35 % for argon at its maximum field). The statistic is recorded at the save
    planes, so the field between them is not seen. A peak rate at the recorded rate
    of the maximum field, to the table's tolerance, means the run reached it: the
    rate is held above it, so how far the field went beyond is not known.

    Parameters
    ----------
    rate : OptmatRate
        The population's rate.
    peak_rate : float
        The largest rate recorded, s^-1, non-negative.
    label : str
        The population.
    average : callable or None, optional
        The recorded rate in s^-1 at an envelope amplitude in V/m, for an envelope
        run; ``None`` (a field grid) takes the rate itself.

    Returns
    -------
    PeakFieldReport
        The largest field, whether it reached the maximum field, and the
        rate-weighted energy there against the constant the loss paid.

    Examples
    --------
    >>> from solprop.nonlinear.optmat_rates import OptmatRate, peak_field_report
    >>> rate = OptmatRate("Ar", 800e-9)
    >>> report = peak_field_report(rate, float(rate(3e10)), label="Ar")
    >>> f"{report.largest_field:.6e}", report.reached_maximum
    ('3.000000e+10', False)
    """
    record = rate.material_records[0]

    def at(value: float) -> float:
        """Return the rate the statistic records at a field.

        Parameters
        ----------
        value : float
            Field in V/m.

        Returns
        -------
        float
            Rate in s^-1.
        """
        return float(rate(value)) if average is None else float(average(value))

    maximum = rate.maximum_field
    # The rate is held above the maximum field, so a peak at the rate there (to the
    # table's own tolerance) says the run reached it, and not by how much.
    held = at(maximum) * (1.0 - _TABLE_TOLERANCE)
    reached = peak_rate >= held > 0.0
    if not peak_rate > 0.0:
        largest = None
    elif reached:
        largest = maximum
    else:
        _, largest = _bisect(
            lambda value: at(value) >= peak_rate, rate.threshold, maximum
        )
    energy = None if largest is None else float(rate.energy_per_event(largest))
    return PeakFieldReport(
        label=label,
        species=rate.species_identity,
        model=record.name,
        peak_rate=float(peak_rate),
        largest_field=largest,
        reached_maximum=reached,
        maximum_field=maximum,
        energy_at_largest=energy,
        constant_energy=rate.ionisation_potential,
        departure=None if energy is None else energy / rate.ionisation_potential - 1.0,
    )
