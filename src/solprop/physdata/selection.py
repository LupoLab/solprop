# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Name the material models a run takes from optmat, and record the ones it resolved.

A :class:`MaterialSelection` is the small immutable choice a caller makes: which
linear, electronic, Raman and ionisation models, at which carrier, under which
approximation policies. ``"recommended"`` is optmat's recommended model of each
material; a model name selects another one from optmat's registry, and an optmat model
object is taken as it is. The selection holds names and policies, never data: the
models are resolved where a species or a solid is built.

A :class:`MaterialRecord` is what a resolved model leaves behind: optmat's identity of
it (name, quantity, material, content digest and status) in plain values, ready for a
run's manifest. Every optmat-backed object solprop builds exposes its records as
``material_records``.

A :class:`MaterialManifest` is a run's whole material record: the optmat version, every
resolved model's record, the species, compositions and solids with their axes, the
policies in force and what was measured about them. Its identities and policies enter
the run's resume key (output schema 8); see :doc:`/guide/results/output`.

This module imports nothing from optmat, so building a selection costs nothing; see
:doc:`/guide/media/materials` for the models, their statuses and the warnings they
raise.
"""

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol, cast, runtime_checkable

from solprop.exceptions import MaterialError

__all__ = [
    "DISPERSIVE_POLICIES",
    "ELECTRONIC_POLICIES",
    "ENERGY_LOSS_POLICIES",
    "RAMAN_POLICIES",
    "RAMAN_SHIFT_POLICIES",
    "STATUSES",
    "DiluteScalingCheck",
    "MaterialLog",
    "MaterialManifest",
    "MaterialRecord",
    "MaterialSelection",
    "ModelChoice",
    "ModelIdentityLike",
    "OptmatModel",
]

STATUSES: Final[tuple[str, ...]] = (
    "validated",
    "provisional",
    "untested",
    "comparison",
    "user",
)
"""optmat's statuses, from most to least curated; ``user`` marks a caller's model."""

ELECTRONIC_POLICIES: Final[tuple[str, ...]] = (
    "recommended",
    "carrier-kerr",
    "symmetric-kerr",
    "gas-polynomial",
)
"""The electronic-response policies.

``"recommended"`` and ``"carrier-kerr"`` name the Kerr coefficient at the carrier, one
number per species and run (SEP 3, section 4.3). The two dispersive policies of SEP 6
evaluate optmat's whole hyperpolarizability model at the absolute frequencies of every
quartet the cubic couples (:data:`DISPERSIVE_POLICIES` names their evaluators):
``"symmetric-kerr"`` factorises the Kerr diagonal symmetrically over the four legs, and
``"gas-polynomial"`` evaluates the gas dispersion polynomial itself.
"""

DISPERSIVE_POLICIES: Final[Mapping[str, str]] = {
    "symmetric-kerr": "symmetric",
    "gas-polynomial": "polynomial",
}
"""The dispersive electronic policies and the evaluator each selects (SEP 6)."""

ENERGY_LOSS_POLICIES: Final[tuple[str, ...]] = ("peak-weighted",)
"""The plasma energy-loss conventions: the rate-weighted energy at the peak field."""

RAMAN_POLICIES: Final[tuple[str, ...]] = ("recommended", "provisional")
"""The names ``MaterialSelection.raman`` takes, for a gas and for silica alike.

``"recommended"`` is the curated default in both contexts: optmat's response of a gas
fill (SEP 3, section 4.4), and optmat's default silica bundle on a fibre's GNLSE, or
its tensor bundle for a vector run (section 4.5). ``"provisional"`` is a gas policy: it
also admits optmat's provisional bands (N2O's, D13), and a silica run refuses it by
name. Beside these, a gas run takes an optmat response object and a silica run a bundle
name or object; each refuses the other's, by name.
"""

RAMAN_SHIFT_POLICIES: Final[tuple[str | None, ...]] = (None, "reference-density")
"""The density-shift approximations a Raman response may take; ``None`` takes none."""


@runtime_checkable
class ModelIdentityLike(Protocol):
    """Read the fields of an optmat ``ModelIdentity`` that a record keeps."""

    @property
    def name(self) -> str:
        """Return the model's name."""
        ...

    @property
    def kind(self) -> str:
        """Return the quantity the model gives."""
        ...

    @property
    def material(self) -> str:
        """Return the material it describes."""
        ...

    @property
    def digest(self) -> str:
        """Return the SHA-256 of its canonical record."""
        ...

    @property
    def optmat_version(self) -> str:
        """Return the optmat version that built it."""
        ...

    @property
    def status(self) -> str:
        """Return its status."""
        ...

    @property
    def status_reason(self) -> str:
        """Return why its status is what it is."""
        ...

    @property
    def modified(self) -> tuple[tuple[str, str], ...]:
        """Return the overrides applied."""
        ...

    @property
    def base(self) -> str:
        """Return the digest of the curated model an override started from."""
        ...

    @property
    def sources(self) -> tuple[str, ...]:
        """Return the source records the model rests on."""
        ...

    @property
    def reference_temperature(self) -> float | None:
        """Return the temperature in K a gas polarizability refers to, if declared."""
        ...


@runtime_checkable
class OptmatModel(Protocol):
    """Anything that carries an optmat identity: a model, a response or a bundle."""

    @property
    def identity(self) -> ModelIdentityLike:
        """Return the model's optmat ``ModelIdentity``."""
        ...


# A type checker cannot match optmat's models to `OptmatModel`: their `identity` is a
# `functools.cached_property`, which static protocol matching does not see through. The
# choice is therefore typed as any object and checked at run time, where `isinstance`
# against the protocol works.
type ModelChoice = str | object
"""A model choice: ``"recommended"``, an optmat model name, or an optmat model object
(anything with an ``identity``, an :class:`OptmatModel`), checked when resolved."""


def _choice(value: object, field: str) -> ModelChoice:
    """Validate one model choice: a non-empty name or an optmat model object.

    Parameters
    ----------
    value : object
        The choice as given.
    field : str
        The selection field, for the message.

    Returns
    -------
    ModelChoice
        The choice, a name stripped of surrounding space.

    Raises
    ------
    MaterialError
        If the choice is neither a non-empty string nor an object with an identity.
    """
    if isinstance(value, str):
        name = value.strip()
        if not name:
            msg = f"MaterialSelection.{field} must name a model, got an empty string"
            raise MaterialError(msg)
        return name
    if isinstance(value, OptmatModel):
        return value
    msg = (
        f"MaterialSelection.{field} must be 'recommended', an optmat model name or an "
        f"optmat model object, got {type(value).__name__}"
    )
    raise MaterialError(msg)


def _choice_record(value: ModelChoice) -> str | dict[str, object]:
    """Return a choice as plain values: its name, or a model object's record.

    Parameters
    ----------
    value : ModelChoice
        A validated choice.

    Returns
    -------
    str or dict
        The name, or the canonical record of the model's identity.
    """
    if isinstance(value, str):
        return value
    return MaterialRecord.from_identity(cast("OptmatModel", value).identity).to_dict()


@dataclass(frozen=True)
class MaterialSelection:
    """Choose the optmat models and approximation policies for a run's materials.

    Each model field takes ``"recommended"`` (optmat's recommended model of the
    material), an optmat model name, or an optmat model object, such as one made by
    ``with_parameters``. A name or an object is checked against the material it is
    used for when it is resolved, never matched to another material by name.
    ``MaterialSelection()`` is what an omitted selection means everywhere.

    Attributes
    ----------
    linear : str or OptmatModel
        The linear dispersion model of a gas: its per-particle polarizability, used
        at linear density scaling. The family also covers what a driver resolves
        beside it: the gas's equation of state and transport, the composition of
        air, and the default wall or slab cladding (fused silica).
    electronic : str
        The electronic third-order policy (:data:`ELECTRONIC_POLICIES`):
        ``"recommended"`` or ``"carrier-kerr"``, both the Kerr coefficient evaluated
        at the carrier; or a dispersive policy of SEP 6, ``"symmetric-kerr"`` or
        ``"gas-polynomial"``, which evaluates optmat's hyperpolarizability model at
        every quartet's absolute frequencies
        (:class:`~solprop.nonlinear.responses.DispersiveKerrResponse`).
    raman : str or OptmatModel
        The Raman response (:data:`RAMAN_POLICIES`): ``"recommended"``, the curated
        default of a gas fill and of fused silica alike; ``"provisional"``, which
        also admits a gas's provisional bands with a warning and is refused for
        silica; on a gas run an optmat response object, and on a silica run an
        optmat bundle name or object, each refused in the other context. The
        family covers the silica bundle of a GNLSE run (its ``f_R``) and of a
        guide's tensor response.
    ionisation : str or OptmatModel
        The ionisation rate model: ``"recommended"``, a literature form such as
        ``"ppt_ilkov"``, or a rate-model object.
    carrier_wavelength : float or None
        Vacuum carrier wavelength in m at which the carrier electronic coefficient,
        the Raman pump dispersion and the rate are evaluated; positive and finite, or
        ``None`` to take the run's grid reference. A dispersive electronic policy
        does not use it: its response is evaluated at absolute frequencies.
    energy_loss : str
        The plasma energy-loss convention; ``"peak-weighted"`` is the only one, the
        rate-weighted energy per event at the run's peak field.
    raman_shift : str or None
        ``None``, or ``"reference-density"``: the explicit approximation that holds a
        density-dependent Raman line shift at a reference density.

    Raises
    ------
    MaterialError
        If a field is of the wrong type, empty, or not one of its policies.

    Examples
    --------
    >>> from solprop.physdata.selection import MaterialSelection
    >>> selection = MaterialSelection(carrier_wavelength=800e-9)
    >>> selection.linear, selection.energy_loss
    ('recommended', 'peak-weighted')
    >>> MaterialSelection(energy_loss="mean")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    solprop.exceptions.MaterialError: MaterialSelection.energy_loss must be one of ...
    """

    linear: ModelChoice = "recommended"
    electronic: str = "recommended"
    raman: ModelChoice = "recommended"
    ionisation: ModelChoice = "recommended"
    carrier_wavelength: float | None = None
    energy_loss: str = "peak-weighted"
    raman_shift: str | None = None

    def __post_init__(self) -> None:
        """Validate every field without resolving any model."""
        for name in ("linear", "raman", "ionisation"):
            object.__setattr__(self, name, _choice(getattr(self, name), name))
        policies: tuple[tuple[str, object, tuple[str | None, ...]], ...] = (
            ("electronic", self.electronic, ELECTRONIC_POLICIES),
            ("energy_loss", self.energy_loss, ENERGY_LOSS_POLICIES),
            ("raman_shift", self.raman_shift, RAMAN_SHIFT_POLICIES),
        )
        for name, value, allowed in policies:
            if value not in allowed:
                msg = (
                    f"MaterialSelection.{name} must be one of {allowed}, got {value!r}"
                )
                raise MaterialError(msg)
        carrier = self.carrier_wavelength
        if carrier is not None and (
            isinstance(carrier, bool)
            or not isinstance(carrier, (int, float))
            or not math.isfinite(carrier)
            or carrier <= 0
        ):
            msg = (
                "MaterialSelection.carrier_wavelength must be a positive finite "
                f"wavelength in m or None, got {carrier!r}"
            )
            raise MaterialError(msg)

    def to_dict(self) -> dict[str, object]:
        """Return the selection as plain values, for a run's record.

        Returns
        -------
        dict
            Every field; a model object enters as its identity record.

        Examples
        --------
        >>> from solprop.physdata.selection import MaterialSelection
        >>> MaterialSelection().to_dict()["carrier_wavelength"] is None
        True
        """
        return {
            "linear": _choice_record(self.linear),
            "electronic": self.electronic,
            "raman": _choice_record(self.raman),
            "ionisation": _choice_record(self.ionisation),
            "carrier_wavelength": self.carrier_wavelength,
            "energy_loss": self.energy_loss,
            "raman_shift": self.raman_shift,
        }


@dataclass(frozen=True)
class MaterialRecord:
    """Record one resolved optmat model by its identity, in plain values.

    Attributes
    ----------
    name : str
        optmat's model name, such as ``"fit/Ar"`` or ``"eos/Ar"``.
    kind : str
        The quantity the model gives (``"gas_refractive"``, ``"equation_of_state"``,
        ``"transport"``, ``"solid"``, ...).
    material : str
        The gas, solid or composition the model describes.
    digest : str
        The SHA-256 of the model's canonical record, 64 hexadecimal digits: what
        identifies its numbers.
    status : str
        One of :data:`STATUSES`.
    status_reason : str
        Why the status is what it is, where the source says.
    optmat_version : str
        The optmat version that built the model; informational, not part of the
        digest.
    modified : tuple of tuple of str
        The overrides applied, ``(name, value)`` sorted by name; empty for a
        curated model.
    base : str
        The digest of the curated model an override started from, or empty.
    sources : tuple of str
        The source records the model rests on, as optmat names them.
    reference_temperature : float or None
        The temperature in K a gas polarizability refers to, where its source
        declares one.

    Raises
    ------
    MaterialError
        If the digest is not 64 hexadecimal digits or the status is unknown.
    """

    name: str
    kind: str
    material: str
    digest: str
    status: str
    status_reason: str
    optmat_version: str
    modified: tuple[tuple[str, str], ...] = ()
    base: str = ""
    sources: tuple[str, ...] = ()
    reference_temperature: float | None = None

    def __post_init__(self) -> None:
        """Validate the digest and the status."""
        if len(self.digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.digest
        ):
            msg = f"{self.name}: a digest is 64 hexadecimal digits, got {self.digest!r}"
            raise MaterialError(msg)
        if self.status not in STATUSES:
            msg = f"{self.name}: status {self.status!r} is not one of {STATUSES}"
            raise MaterialError(msg)
        object.__setattr__(
            self, "modified", tuple((str(k), str(v)) for k, v in self.modified)
        )
        object.__setattr__(self, "sources", tuple(str(v) for v in self.sources))

    @classmethod
    def from_identity(cls, identity: ModelIdentityLike) -> "MaterialRecord":  # noqa: UP037 - the hook resolves class names eagerly
        """Build a record from an optmat ``ModelIdentity``.

        Parameters
        ----------
        identity : optmat.identity.ModelIdentity
            The identity of a resolved model.

        Returns
        -------
        MaterialRecord
            Its name, kind, material, digest, status and override fields.
        """
        return cls(
            name=identity.name,
            kind=identity.kind,
            material=identity.material,
            digest=identity.digest,
            status=identity.status,
            status_reason=identity.status_reason,
            optmat_version=identity.optmat_version,
            modified=identity.modified,
            base=identity.base,
            sources=identity.sources,
            reference_temperature=identity.reference_temperature,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the record as canonical plain values.

        Returns
        -------
        dict
            Every field; ``modified`` as a list of ``[name, value]`` pairs and
            ``sources`` as a list.
        """
        return {
            "name": self.name,
            "kind": self.kind,
            "material": self.material,
            "digest": self.digest,
            "status": self.status,
            "status_reason": self.status_reason,
            "optmat_version": self.optmat_version,
            "modified": [list(pair) for pair in self.modified],
            "base": self.base,
            "sources": list(self.sources),
            "reference_temperature": self.reference_temperature,
        }


@dataclass(frozen=True)
class DiluteScalingCheck:
    """Record the error of linear density scaling for one gas at its largest density.

    solprop forms a gas's susceptibility as ``N alpha / eps0``, the dilute limit of
    optmat's Lorentz--Lorenz law with its refractivity virial. This is the relative
    difference of the two in ``n - 1`` at the density where it is largest.

    Attributes
    ----------
    label : str
        The population, or the gas, it was evaluated for.
    material : str
        The gas.
    density : float
        The largest number density of the population, m^-3.
    temperature : float
        Temperature in K at that density.
    wavelength : float
        Vacuum wavelength in m at which the error is largest over the band.
    relative_error : float
        ``(n - 1)_dilute / (n - 1)_optmat - 1``, dimensionless; negative where the
        dilute law is low. NaN when the check could not be measured.
    threshold : float
        The bound above which a warning is raised, relative.
    reason : str
        Empty for a measured check; otherwise why optmat could not evaluate it.
        An unmeasured check never fails the fill it belongs to.

    Examples
    --------
    >>> from solprop.physdata.selection import DiluteScalingCheck
    >>> check = DiluteScalingCheck("Ar", "Ar", 1.5e27, 293.15, 8e-7, -1.2e-2, 1e-2)
    >>> check.exceeded
    True
    """

    label: str
    material: str
    density: float
    temperature: float
    wavelength: float
    relative_error: float
    threshold: float
    reason: str = ""

    @property
    def measured(self) -> bool:
        """Return whether the error was measured.

        Returns
        -------
        bool
            True unless optmat refused the evaluation (see `reason`).
        """
        return not self.reason

    @property
    def exceeded(self) -> bool:
        """Return whether the error's magnitude is above the threshold.

        Returns
        -------
        bool
            ``abs(relative_error) > threshold`` for a measured check; False for one
            that was not measured.
        """
        return self.measured and abs(self.relative_error) > self.threshold

    def to_dict(self) -> Mapping[str, object]:
        """Return the check as plain values, for a run's record.

        Returns
        -------
        collections.abc.Mapping
            Every field; ``relative_error`` and ``wavelength`` are None for a check
            that was not measured.
        """
        return {
            "label": self.label,
            "material": self.material,
            "density": self.density,
            "temperature": self.temperature,
            "wavelength": self.wavelength if self.measured else None,
            "relative_error": self.relative_error if self.measured else None,
            "threshold": self.threshold,
            "measured": self.measured,
            "reason": self.reason,
        }


def _canonical(value: object) -> str:
    """Return plain values as canonical JSON, for ordering and deduplication.

    Parameters
    ----------
    value : object
        Plain values: mappings, sequences, strings, numbers, None.

    Returns
    -------
    str
        Sorted-key, compact JSON.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _plain(value: object) -> object:
    """Return a value with tuples as lists and mappings as dicts, recursively.

    Parameters
    ----------
    value : object
        Plain values, possibly nested in tuples and mappings.

    Returns
    -------
    object
        The same values in JSON's own containers.
    """
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _entries(values: Iterable[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    """Return entries as plain dicts, each once, in canonical order.

    Parameters
    ----------
    values : iterable of collections.abc.Mapping
        The entries, possibly repeated.

    Returns
    -------
    tuple of dict
        The distinct entries sorted by their canonical JSON.
    """
    distinct = {
        _canonical(_plain(entry)): cast("dict[str, object]", _plain(entry))
        for entry in values
    }
    return tuple(distinct[key] for key in sorted(distinct))


def _keyed(
    entries: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> dict[str, object]:
    """Key entries by the named fields, so a difference names the entry that moved.

    Parameters
    ----------
    entries : sequence of collections.abc.Mapping
        Distinct entries in canonical order.
    fields : sequence of str
        The fields whose values, joined by ``:``, make an entry's key. Entries
        that share a key take a numbered suffix, ``#1``, ``#2``, in their order.

    Returns
    -------
    dict
        The entries by key.
    """
    names = [
        ":".join(str(entry.get(name) or "-") for name in fields) for entry in entries
    ]
    keyed: dict[str, object] = {}
    counts: dict[str, int] = {}
    for name, entry in zip(names, entries, strict=True):
        if names.count(name) == 1:
            keyed[name] = entry
            continue
        counts[name] = counts.get(name, 0) + 1
        keyed[f"{name}#{counts[name]}"] = entry
    return keyed


def _stored_entries(
    document: Mapping[str, object], name: str
) -> list[Mapping[str, object]]:
    """Return a stored manifest field's entries, keyed or listed.

    Parameters
    ----------
    document : collections.abc.Mapping
        A stored manifest.
    name : str
        The field.

    Returns
    -------
    list of collections.abc.Mapping
        Its entries; none when it is absent.
    """
    stored = document.get(name) or {}
    items = stored.values() if isinstance(stored, Mapping) else stored
    return list(cast("Iterable[Mapping[str, object]]", items))


_SOLID_POLICY: Final = ("material", "axis", "model", "digest", "transparent")
"""The fields of a solid's entry that identify it; with its bound, they are hashed."""


@dataclass(frozen=True)
class MaterialManifest:
    """Record the optmat materials of one run: identities, policies and measurements.

    Built once per run from its mixtures, modes and solids
    (:func:`solprop.driver.material_output.material_manifest`) and stored in its
    :class:`~solprop.output.schema.RunMetadata`. A run with no optmat-backed object
    and no selection has the empty manifest, ``MaterialManifest()``.

    Two parts are distinguished. The **identities and policies** -- the optmat
    version, every model's record, the species, compositions and solids, and the
    policies -- are what was computed, and enter the resume key (:meth:`hashed`).
    The **measurements** -- the dilute-scaling errors, the loss a transparent solid
    dropped, the bands outside the models' qualified intervals, the citations of
    the other material objects and the carrier Kerr coefficients with their
    third-harmonic discrepancy -- say how good the approximations were, and are
    recorded beside it. The selection is recorded as it was given; what it resolved
    to is in the records, so naming a model ``"recommended"`` or by its name hashes
    alike.

    Attributes
    ----------
    optmat_version : str
        The installed optmat's full version string; empty for an empty manifest.
    selection : collections.abc.Mapping or None
        :meth:`MaterialSelection.to_dict` of the run's selection, or ``None``.
    records : tuple of MaterialRecord
        Every resolved model, each digest once, sorted by material, kind, name and
        digest.
    species : tuple of collections.abc.Mapping
        One entry per optmat-backed population: ``label``, ``species`` and the
        digests of its ``records``.
    compositions : tuple of collections.abc.Mapping
        The named optmat compositions a mixture was built from: ``name``, ``labels``
        and ``fractions``.
    solids : tuple of collections.abc.Mapping
        One entry per optmat solid: ``material``, ``axis`` (empty for an isotropic
        one), ``model``, ``digest``, ``transparent``, ``absorption_bound`` (1/m, for
        a transparent one) and the ``transparency`` measured in the run itself
        (:meth:`solprop.physdata.glasses.OptmatGlass.run_transparency`).
    policies : collections.abc.Mapping
        The approximation policies in force: ``linear_density_scaling`` (the gas
        law and its warning threshold), ``electronic`` and ``carrier_wavelength``
        as selected, ``electronic_response`` (the ``carrier-kerr`` law and the
        carrier it was evaluated at, when an optmat Kerr model is in the run; or
        a dispersive law, its evaluator, harmonic sector, tensor approximation,
        and each species' support-record digest and regions, when the run's
        electronic response is SEP 6's dispersive one),
        ``gnlse_raman`` (a GNLSE run's held-back fraction, where it came from and
        the silica bundle), ``raman_shift`` and ``raman_reference_temperature``,
        ``energy_loss``, and ``ionisation``: one entry per optmat rate, with its
        threshold kind, its reference field, its constant energy per event and that
        energy's bound
        (:meth:`solprop.nonlinear.optmat_rates.OptmatRate.manifest_entry`). A
        policy a later phase resolves is recorded as selected, or ``None``.
    dilute_scaling : tuple of collections.abc.Mapping
        :meth:`DiluteScalingCheck.to_dict` of each optmat population, measured once
        per run over the run's band.
    ranges : tuple of collections.abc.Mapping
        For each model evaluated outside its qualified intervals: ``material``,
        ``model``, ``digest`` and the ``outside`` phrases its range warning named.
    citations : tuple of collections.abc.Mapping
        The other material objects of the run (supplied or solprop's own):
        ``object`` (the class), ``name`` and ``citation``.
    electronic : tuple of collections.abc.Mapping
        Each optmat population's Kerr coefficient at the carrier, beside optmat's
        third-harmonic one and their ratio where that is within optmat's fitted
        data (:meth:`solprop.physdata.nonlinear_data.CarrierKerr.to_dict`, with the
        ``label``): what the single-coefficient ``carrier-kerr`` approximation
        leaves out. Under a dispersive policy, each species' dispersive model
        instead: its evaluator and sector, model and support digests, leg bound,
        band edges and the regions of the run's active band
        (:meth:`solprop.physdata.nonlinear_data.ElectronicRegions.to_dict`).
    peak_fields : tuple of collections.abc.Mapping
        The post-run diagnostic of each optmat ionisation rate: the largest field
        its population saw at the save planes, against the model's maximum field,
        and the rate-weighted energy there against the constant the loss paid
        (:class:`solprop.nonlinear.optmat_rates.PeakFieldReport`). Recorded once the
        run has run; empty before.
    raman : tuple of collections.abc.Mapping
        For each optmat Raman response: its labels, composition, reference
        temperature, pump wavelength, group and term counts, the densities declared
        to its range check, the composition's spread along the run, and a held line
        shift with its phase bound over the run's window (SEP 3, section 4.4).

    Examples
    --------
    >>> from solprop.physdata.selection import MaterialManifest
    >>> MaterialManifest().empty, MaterialManifest().hashed() is None
    (True, True)
    """

    optmat_version: str = ""
    selection: Mapping[str, object] | None = None
    records: tuple[MaterialRecord, ...] = ()
    species: tuple[Mapping[str, object], ...] = ()
    compositions: tuple[Mapping[str, object], ...] = ()
    solids: tuple[Mapping[str, object], ...] = ()
    policies: Mapping[str, object] | None = None
    dilute_scaling: tuple[Mapping[str, object], ...] = ()
    ranges: tuple[Mapping[str, object], ...] = ()
    citations: tuple[Mapping[str, object], ...] = ()
    electronic: tuple[Mapping[str, object], ...] = ()
    peak_fields: tuple[Mapping[str, object], ...] = ()
    raman: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        """Deduplicate every entry and put it in canonical order."""
        unique = {entry.digest: entry for entry in self.records}
        object.__setattr__(
            self,
            "records",
            tuple(
                sorted(
                    unique.values(),
                    key=lambda r: (r.material, r.kind, r.name, r.digest),
                )
            ),
        )
        for name in (
            "species",
            "compositions",
            "solids",
            "dilute_scaling",
            "ranges",
            "citations",
            "electronic",
            "peak_fields",
            "raman",
        ):
            object.__setattr__(self, name, _entries(getattr(self, name)))
        if self.selection is not None:
            object.__setattr__(
                self, "selection", cast("dict[str, object]", _plain(self.selection))
            )
        if self.policies is not None:
            object.__setattr__(
                self, "policies", cast("dict[str, object]", _plain(self.policies))
            )

    @property
    def empty(self) -> bool:
        """Return whether the run took nothing from optmat and selected nothing.

        Returns
        -------
        bool
            True for a run of solprop's own materials only.
        """
        return self.selection is None and not self.records

    def _solid_identity(self, solid: Mapping[str, object]) -> dict[str, object]:
        """Return the hashed part of a solid's entry: identity and policy.

        Parameters
        ----------
        solid : collections.abc.Mapping
            One entry of :attr:`solids`.

        Returns
        -------
        dict
            Its identifying fields, and the absorption bound of a transparent one.
        """
        identity = {name: solid.get(name) for name in _SOLID_POLICY}
        if solid.get("transparent"):
            identity["absorption_bound"] = solid.get("absorption_bound")
        return identity

    def hashed(self) -> dict[str, object] | None:
        """Return the identities and policies, which the resume key covers.

        Returns
        -------
        dict or None
            The optmat version, the records by ``material:kind:name``, the species
            by label, the compositions, the solids' identities and policies, and
            the policies; ``None`` for an empty manifest.

        Examples
        --------
        >>> from solprop.physdata.selection import MaterialManifest, MaterialSelection
        >>> selection = MaterialSelection().to_dict()
        >>> manifest = MaterialManifest("0.1.1", selection, policies={})
        >>> sorted(manifest.hashed())[:3]
        ['compositions', 'optmat_version', 'policies']
        """
        if self.empty:
            return None
        records = [record.to_dict() for record in self.records]
        return {
            "optmat_version": self.optmat_version,
            "records": _keyed(records, ("material", "kind", "name")),
            "species": _keyed(self.species, ("label",)),
            "compositions": [dict(entry) for entry in self.compositions],
            "solids": _keyed(
                [self._solid_identity(solid) for solid in self.solids],
                ("material", "axis", "model"),
            ),
            "policies": dict(self.policies or {}),
        }

    def to_dict(self) -> dict[str, object] | None:
        """Return the whole manifest as plain values, for a file.

        Returns
        -------
        dict or None
            :meth:`hashed`, with the selection, the solids' measurements, the
            dilute-scaling checks, the out-of-interval bands, the citations, the
            carrier Kerr coefficients, the post-run peak fields and the Raman
            responses' assemblies; ``None`` for an empty manifest.
        """
        document = self.hashed()
        if document is None:
            return None
        document["selection"] = None if self.selection is None else dict(self.selection)
        document["solids"] = _keyed(self.solids, ("material", "axis", "model"))
        document["dilute_scaling"] = [dict(entry) for entry in self.dilute_scaling]
        document["ranges"] = [dict(entry) for entry in self.ranges]
        document["citations"] = [dict(entry) for entry in self.citations]
        document["electronic"] = [dict(entry) for entry in self.electronic]
        document["peak_fields"] = [dict(entry) for entry in self.peak_fields]
        document["raman"] = [dict(entry) for entry in self.raman]
        return document

    @classmethod
    def from_dict(cls, document: Mapping[str, object] | None) -> "MaterialManifest":  # noqa: UP037 - the hook resolves class names eagerly
        """Rebuild a manifest from :meth:`to_dict`.

        Parameters
        ----------
        document : collections.abc.Mapping or None
            A stored manifest, or ``None`` for the empty one.

        Returns
        -------
        MaterialManifest
            The manifest, hashing as the one that was stored.

        Raises
        ------
        MaterialError
            If a stored record is malformed.

        Examples
        --------
        >>> from solprop.physdata.selection import MaterialManifest, MaterialSelection
        >>> selection = MaterialSelection().to_dict()
        >>> manifest = MaterialManifest("0.1.1", selection, policies={})
        >>> MaterialManifest.from_dict(manifest.to_dict()) == manifest
        True
        """
        if document is None:
            return cls()

        try:
            records = tuple(
                MaterialRecord(
                    **{
                        **entry,
                        "modified": tuple(
                            tuple(pair)
                            for pair in cast("list[list[str]]", entry["modified"])
                        ),
                        "sources": tuple(cast("list[str]", entry["sources"])),
                    }  # pyright: ignore[reportArgumentType] - checked by the record
                )
                for entry in _stored_entries(document, "records")
            )
        except (KeyError, TypeError) as error:
            msg = f"a stored material record is malformed: {error}"
            raise MaterialError(msg) from error
        selection = document.get("selection")
        policies = document.get("policies")
        return cls(
            optmat_version=str(document.get("optmat_version", "")),
            selection=cast("Mapping[str, object] | None", selection),
            records=records,
            species=tuple(_stored_entries(document, "species")),
            compositions=tuple(_stored_entries(document, "compositions")),
            solids=tuple(_stored_entries(document, "solids")),
            policies=cast("Mapping[str, object] | None", policies),
            dilute_scaling=tuple(_stored_entries(document, "dilute_scaling")),
            ranges=tuple(_stored_entries(document, "ranges")),
            citations=tuple(_stored_entries(document, "citations")),
            electronic=tuple(_stored_entries(document, "electronic")),
            peak_fields=tuple(_stored_entries(document, "peak_fields")),
            raman=tuple(_stored_entries(document, "raman")),
        )


@dataclass(eq=False)
class MaterialLog:
    """Keep what one evaluation context has warned about and found out of range.

    A run keeps one from its construction to its record, so that each material
    warning is raised once per run (``solprop.physdata._optmat.material_context``),
    and its manifest records the bands it found outside a model's intervals.

    Attributes
    ----------
    seen : set of tuple of str
        The keys of the warnings already raised.
    ranges : dict
        For each model evaluated outside its qualified intervals, keyed by its
        digest: the material, the model's name and the set of phrases naming the
        parts of the bands outside them.
    dropped : dict
        For each transparent solid index evaluated in the context, keyed by its
        model's digest and its absorption bound in 1/m: the largest extinction
        coefficient it dropped, the largest absorption coefficient in 1/m and that
        absorption's vacuum wavelength in m. The run's own, whatever the adapter
        dropped in other runs (SEP 3, Q28).
    """

    seen: set[tuple[str, ...]] = field(default_factory=set)
    ranges: dict[str, tuple[str, str, set[str]]] = field(default_factory=dict)
    dropped: dict[tuple[str, float], tuple[float, float, float]] = field(
        default_factory=dict
    )

    def record_dropped(
        self,
        key: tuple[str, float],
        extinction: float,
        absorption: float,
        wavelength: float,
    ) -> None:
        """Add the loss a transparent index dropped in one evaluation.

        Parameters
        ----------
        key : tuple
            The model's digest and the absorption bound in 1/m.
        extinction : float
            The largest ``k`` the evaluation dropped, dimensionless.
        absorption : float
            The largest ``4 pi k / lambda`` it dropped, 1/m.
        wavelength : float
            The vacuum wavelength in m of that absorption, or NaN.

        Examples
        --------
        >>> from solprop.physdata.selection import MaterialLog
        >>> log = MaterialLog()
        >>> log.record_dropped(("digest", 100.0), 1e-9, 2.0, 6e-6)
        >>> log.record_dropped(("digest", 100.0), 1e-8, 1.0, 1e-5)
        >>> log.dropped[("digest", 100.0)]
        (1e-08, 2.0, 6e-06)
        """
        known, worst, where = self.dropped.get(key, (0.0, 0.0, math.nan))
        if absorption > worst:
            worst, where = absorption, wavelength
        self.dropped[key] = (max(known, extinction), worst, where)

    def record_range(
        self, digest: str, material: str, name: str, parts: Sequence[str]
    ) -> None:
        """Add the parts of a band a model was evaluated at outside its intervals.

        Parameters
        ----------
        digest : str
            The model's digest.
        material : str
            The material it serves.
        name : str
            The model's name.
        parts : sequence of str
            The phrases naming the parts of the band outside the model's
            intervals; none when the band is not known.
        """
        _, _, known = self.ranges.setdefault(digest, (material, name, set()))
        known.update(parts)

    def copy(self) -> "MaterialLog":  # noqa: UP037 - the hook resolves class names eagerly
        """Return an independent log holding what this one holds so far.

        Returns
        -------
        MaterialLog
            The same warning keys and out-of-interval bands, in new containers.

        Examples
        --------
        >>> from solprop.physdata.selection import MaterialLog
        >>> parent = MaterialLog()
        >>> parent.seen.add(("range", "x"))
        >>> child = parent.copy()
        >>> child.seen.add(("range", "y"))
        >>> sorted(parent.seen), sorted(child.seen)
        ([('range', 'x')], [('range', 'x'), ('range', 'y')])
        """
        return MaterialLog(
            set(self.seen),
            {
                digest: (material, name, set(parts))
                for digest, (material, name, parts) in self.ranges.items()
            },
            dict(self.dropped),
        )

    def range_entries(self) -> tuple[dict[str, object], ...]:
        """Return the out-of-interval record, in a canonical order.

        Returns
        -------
        tuple of dict
            One entry per model: ``material``, ``model``, ``digest`` and the sorted
            ``outside`` phrases; sorted by material, model and digest.
        """
        return tuple(
            {
                "material": material,
                "model": name,
                "digest": digest,
                "outside": sorted(parts),
            }
            for digest, (material, name, parts) in sorted(
                self.ranges.items(), key=lambda item: (item[1][0], item[1][1], item[0])
            )
        )
