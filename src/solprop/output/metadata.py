# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run record: what produced an output file, and the resume key it is checked by.

An output file has to answer one question years later: *what produced this?*
:class:`RunMetadata` is that answer: the grid, the modes, the serialised physics
configuration, the package versions, the git provenance of the running checkout, the
noise seed and realisation, and the stepper settings.

**The resume key.** :meth:`RunMetadata.hashed` is the part of the record that says what
was computed: the schema version, the grid, the modes, the configuration, the noise, the
stepper settings with the stepping method when it is not the tableau (SEP 13), and the
identities and policies of the run's material manifest, and nothing else: not the
wall-clock time the run started, not the package versions, and not the git provenance.
:meth:`RunMetadata.configuration_hash` is a SHA-256 of its canonical JSON rather than
:func:`hash`, so it is stable across processes; it is stamped in every file and is the
exact identity a resume, a suspended writer, a snapshot and a scan row are checked by
(ADR-066): a resume with different physics or numerics of the same shape is refused,
naming the fields that changed (:meth:`RunMetadata.summarise_differences`). A record
read from a file is compared as it was written.

The manual page is ``docs/guide/results/provenance.md``.
"""

import dataclasses
import functools
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np

from solprop._display import Displayed, engineering, line
from solprop._identity import canonical_digest, canonical_json, text_digest
from solprop.exceptions import MaterialError, OutputError
from solprop.grid import EnvelopeGrid
from solprop.modes.material import JsonValue
from solprop.modes.protocol import TransverseMeasure
from solprop.output.provenance import BackendRecord, GitProvenance, package_versions
from solprop.output.records import (
    FIELD_LAYOUT,
    SPECTRAL_CONVENTION,
    DefaultsResolution,
    GridRecord,
    ModeRecord,
    NoiseRecord,
    NoiseResolution,
    SimulationKind,
    StepperRecord,
)
from solprop.output.serialisation import (
    _RECORD_ARRAY,
    _ArrayReference,
    _rebuild,
    _resolved,
    _stored_entry,
    serialise,
)
from solprop.physdata.selection import MaterialManifest, hashed_document

__all__ = [
    "INTERPOLATED_RESUMES_ANNOTATION",
    "PEAK_FIELDS_ANNOTATION",
    "RUN_SUMMARY_ANNOTATION",
    "SCHEMA_VERSION",
    "RunMetadata",
    "resolve_metadata_write",
]

logger = logging.getLogger(__name__)

_DIFFERENCES_SHOWN = 4
"""How many changed fields a refusal names in each part of the record before it counts
the rest of that part."""

_PATH_SEPARATOR = re.compile(r"[.\[]")
"""What ends a changed path's top-level key, the part of the record it belongs to."""

_ELEMENT_INDICES = re.compile(r"(?:\[\d+\])+$")
"""The trailing indices of a changed path into an array's element."""

_ARRAY_PART = re.compile(r"\.__(?:ndarray|imaginary)__$")
"""The key of an array record's real or imaginary part, dropped from its stem."""

SCHEMA_VERSION = 2
"""Version of the saved-output layout; incremented when a stored field changes meaning.

Schema 1 is the first released layout, 0.1.0's: the record types of this module, written
by :class:`solprop.output.memory.MemoryOutput` and
:class:`solprop.output.hdf5.HDF5Output` as ``docs/guide/results/output.md`` describes
them. Schema 2 records ``DataPulse.spectral_phase``. A record that declares any other
version is refused by name (:class:`RunMetadata`): files written by an earlier release
or by development versions of solprop, which declared output schemas up to 28, are not
read, and nor is a file from a later release. Read such a file with the solprop version
that wrote it.
"""

PEAK_FIELDS_ANNOTATION = "material_peak_fields"
"""Name of the post-run annotation holding the material manifest's peak-field report.

A list with one plain mapping per population with an optmat ionisation rate:
the largest field it reached against its model's maximum field, and the energy per
event there against the constant the run paid
(:class:`solprop.nonlinear.optmat_rates.PeakFieldReport`). Written by
:meth:`solprop.driver.simulation.Simulation.run` when a run closes, outside the
configuration hash (SEP 3, Q13).
"""

INTERPOLATED_RESUMES_ANNOTATION = "interpolated_resumes"
"""Name of the post-run annotation listing where a dense run was resumed, m.

A list of the positions of the interpolated save planes a run observed densely was
continued from with ``resume="interpolated"``: each continuation started afresh from a
field interpolated inside a step, not a state the march passed through (SEP 14).
Written by :meth:`solprop.driver.simulation.Simulation.run`, outside the configuration
hash.
"""

RUN_SUMMARY_ANNOTATION = "run_summary"
"""Name of the annotation holding the start-of-run summaries of a plane store.

A list with one mapping per start, ``{"title": str, "rows": [[key, value], ...]}``: a
run that is resumed adds the summary of its continuation to the first one's. Written
when a run starts, not when it closes, and outside the configuration hash.
"""


def _differences(
    left: JsonValue, right: JsonValue, path: str, found: list[str]
) -> None:
    """Collect the dotted paths at which two serialised structures differ.

    - ``right``: the second.
    """
    if isinstance(left, dict) and isinstance(right, dict):
        if _RECORD_ARRAY in left and _RECORD_ARRAY in right:
            # One entry, whatever changed in it: the array has no text to walk.
            if dict(left) != dict(right):
                found.append(path)
            return
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else key
            if key not in left or key not in right:
                found.append(child)
            else:
                _differences(left[key], right[key], child, found)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            found.append(f"{path}[length]")
            return
        for index, (one, other) in enumerate(zip(left, right, strict=True)):
            _differences(one, other, f"{path}[{index}]", found)
        return
    if canonical_json(left) != canonical_json(right):
        found.append(path)


def _refuse_schema(declared: object) -> None:
    """Refuse a record of any output schema but :data:`SCHEMA_VERSION`.

    One rule for every other number: an earlier release wrote a lower output schema, a
    development version of solprop wrote schemas up to 28, and a later release may write
    a higher one, and this release reads none of them (ADR-058).
    """
    if type(declared) is int and declared == SCHEMA_VERSION:
        return
    msg = (
        f"this record declares output schema_version {declared!r}, and this release "
        f"of solprop reads output schema {SCHEMA_VERSION} only: a file written by an "
        "earlier release, by a development version of solprop or by a later release is "
        "not read; read it with the solprop version that wrote it"
    )
    raise OutputError(msg)


_HASHED_KEYS: tuple[str, ...] = (
    "schema_version",
    "simulation_kind",
    "grid",
    "modes",
    "configuration",
    "noise",
    "stepper",
    "spatial",
    "power_convention",
    "modal",
    "materials",
    "initial_z",
    "tdse",
    "quadrature",
)
"""The entries of :meth:`RunMetadata.hashed`, in the order it builds them."""


def _stored_hashed(document: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Return the hashed part of a stored record, as the stored text holds it.

    The entries :meth:`RunMetadata.hashed` covers, taken from the document rather than
    from the objects rebuilt from it: the general modal identity without its accessor
    list (:func:`_hashed_modal`), and the material manifest's hashed part
    (:func:`solprop.physdata.selection.hashed_document`). An entry the document lacks
    stays absent, so the record digests to the stamp its writer computed.
    """
    hashed = {name: document[name] for name in _HASHED_KEYS if name in document}
    modal = hashed.get("modal")
    if isinstance(modal, dict):
        hashed["modal"] = {
            key: value for key, value in modal.items() if key != "accessors"
        }
    if "materials" in hashed:
        materials = hashed["materials"]
        hashed["materials"] = cast(
            "JsonValue",
            hashed_document(
                None if materials is None else cast("dict[str, object]", materials)
            ),
        )
    return hashed


def _hashed_modal(modal: Mapping[str, object] | None) -> JsonValue:
    """Return the general modal identity as the resume key sees it.

    Everything in the record except ``accessors``. Every other entry says what
    *was computed* -- which guide, which tables, which basis, which responses, how
    far from orthonormal it measured -- and a resume must refuse when one of them
    moves. ``accessors`` says what *this package version can compute*, which is a
    property of the build and not of the run: leaving it in the key would make
    every accessor this package gains invalidate every general modal file ever
    written, for a capability those files **gain** rather than lose. The list stays in
    the record for a reader, and the resume door checks it as a **subset** instead
    (:func:`solprop.output.hdf5.check_cache`).

    - ``modal``: the identity, as :func:`solprop.translate.general.modal_identity`
      writes it, or ``None`` for a run that is not a general modal one.
    - Returns: the serialised identity without ``accessors``, or ``None``.
    """
    if modal is None:
        return None
    return serialise({key: value for key, value in modal.items() if key != "accessors"})


def _guide_text(guide: object) -> str:
    """Return the one printed line describing the guide a general run was built on.

    - Returns: the provider's model, its region count and the resolution its reference
      was sampled at, or ``'none'`` when the entry is missing or malformed.
    """
    if not isinstance(guide, Mapping):
        return "none"
    provider = guide.get("provider")
    sampling = guide.get("sampling")
    regions = guide.get("regions")
    model = provider.get("model") if isinstance(provider, Mapping) else "unnamed"
    count = len(regions) if isinstance(regions, Sequence) else 0
    if not isinstance(sampling, Mapping):
        return f"{model}, {count} region(s)"
    return (
        f"{model}, {count} region(s), {sampling.get('n_knots')} knots on "
        f"{sampling.get('n_nodes')} nodes, profile error "
        f"{float(cast('float', sampling.get('profile_error', float('nan')))):.3e}"
    )


def _tables_text(tables: object) -> str:
    """Return the one printed line describing a general run's transform tables.

    The ``model_id`` of a general modal reference is the provider's whole physical
    identity as a JSON document -- 412 characters on the slice's one-channel
    fixture, and longer with more channels -- so printing it filled the row with
    the first eighty characters of a serialised mapping and said nothing. What a
    reader wants from this row is the gauge, the numeric policy and something short
    that two files can be compared by, which is what a digest is for. The full
    document is in ``metadata.modal['tables']['model_id']``.

    - Returns: a 12-hex digest of the model identity, the gauge and the policy, or
      ``'none'`` when the entry is missing or malformed.
    """
    if not isinstance(tables, Mapping):
        return "none"
    model = tables.get("model_id")
    digest = text_digest(str(model))[:12] if model is not None else "unnamed"
    # The digest first: the row is truncated to the block's width, and it is the
    # entry two files are compared by.
    return (
        f"model {digest}, gauge {tables.get('gauge_id')}, "
        f"policy {tables.get('policy_id')}"
    )


def _responses_text(responses: object) -> str:
    """Return the one printed line describing a general run's regional responses.

    - ``responses``: the ``responses`` entry of a general modal identity, or ``None``
      for a guide whose regions are all linear.
    - Returns: the compiled model, the responding region count and each region's
      declared susceptibility, or ``'none'`` when no region responds.
    """
    if not isinstance(responses, Mapping):
        return "none"
    chi3 = responses.get("chi3")
    values = (
        ", ".join(f"{float(one):.3e}" for one in chi3)
        if isinstance(chi3, Sequence) and not isinstance(chi3, str)
        else "unrecorded"
    )
    return (
        f"{responses.get('model_id')}, {responses.get('policy_id')}, "
        f"{responses.get('n_responses')} region(s), chi3 {values}"
    )


def _flow_text(flow: object) -> str:
    """Return the printed summary of a coupled-flow entry, or that there is none."""
    if not isinstance(flow, Mapping):
        return "diagonal"
    ratios = [
        float(cast("float", flow.get(name, 0.0)))
        for name in ("transverse_ratio", "vacuum_ratio", "quasi_static")
    ]
    methods = flow.get("section_methods")
    named = (
        ", ".join(str(one) for one in methods)
        if isinstance(methods, Sequence) and not isinstance(methods, str)
        else "none"
    )
    conditioning = float(cast("float", flow.get("inverse_condition_bound", 1.0)))
    return (
        f"{flow.get('model')}, {named}, worst ratio {max(ratios):.3e}, "
        f"conditioning {conditioning:.3e}"
    )


def _modal_rows(modal: Mapping[str, object] | None) -> list[tuple[str, object]]:
    """Return the printed rows of a general modal identity, or none at all.

    Every field of :attr:`RunMetadata.modal` appears in one of these rows,
    including the ``flow`` entry of a coherent wall,
    so that a printed record says which guide, which table and which basis produced
    the file; the rows are absent altogether for a route that has no such identity.

    - ``modal``: the identity, or ``None`` for a route that declares none.
    """
    if modal is None:
        return []
    tables = modal.get("tables")
    metric = modal.get("metric")
    guide = modal.get("guide")
    band = modal.get("active_band")
    channels = modal.get("channels")
    names = (
        ", ".join(str(one.get("display_label")) for one in channels)
        if isinstance(channels, Sequence) and not isinstance(channels, str)
        else "none"
    )
    published = modal.get("accessors")
    accessors = (
        cast("Sequence[object]", published)
        if isinstance(published, Sequence) and not isinstance(published, str)
        else None
    )
    return [
        ("basis", f"{modal.get('basis')}, {modal.get('n_components')} components"),
        ("channels", names),
        ("channel count", modal.get("n_channels")),
        ("tables", _tables_text(tables)),
        (
            "metric",
            f"{metric.get('kind')}, defect "
            f"{float(metric.get('defect', float('nan'))):.3e}, certificate "
            f"{metric.get('certificate_id')}"
            if isinstance(metric, Mapping)
            else "none",
        ),
        ("responses", _responses_text(modal.get("responses"))),
        ("guide", _guide_text(guide)),
        (
            "active band",
            f"{engineering(float(band[0]), 'rad/s')} to "
            f"{engineering(float(band[1]), 'rad/s')}"
            if isinstance(band, Sequence) and len(band) == 2
            else "none",
        ),
        ("frame", modal.get("frame")),
        ("power weight", modal.get("power_weight")),
        ("flow", _flow_text(modal.get("flow"))),
        ("noise basis", modal.get("noise_reference_basis") or "none"),
        (
            "accessors",
            # The count first: the row is truncated to the block's width, and a
            # reader who cannot see the last name can still see how many there are
            # and read the list off `metadata.modal['accessors']`.
            f"{len(accessors)}: {', '.join(str(one) for one in accessors)}"
            if accessors is not None
            else "unrestricted",
        ),
    ]


def _material_rows(manifest: MaterialManifest) -> list[tuple[str, object]]:
    """Return the printed row of a material manifest, or none for an empty one."""
    if manifest.empty:
        return []
    names = ", ".join(sorted({record.name for record in manifest.records})) or "none"
    return [("materials", f"optmat {manifest.optmat_version}: {names}")]


@dataclass(frozen=True, eq=False, kw_only=True)
class RunMetadata(Displayed):
    """Everything needed to say what produced a file, and to refuse a wrong resume.

    Equality is by identity (``eq=False``); two runs are the same run when their
    :meth:`configuration_hash` agree, which is what a resume actually asks.

    Attributes
    ----------
    grid : GridRecord
        The grid the run was computed on.
    modes : tuple
        One :class:`~solprop.output.records.ModeRecord` per propagated mode, in state
        order.
    simulation_kind : {'mode_averaged', 'multimode', 'radial', 'free', 'slab'}
        Which transform the run used.
    configuration : dict
        The physics objects that built the run -- responses, profiles, pulses,
        boundaries -- as :func:`~solprop.output.serialisation.serialise` records them.
        Nothing is silently dropped: a callable is present as its ``repr`` with
        ``reproducible: False``.
    stepper : StepperRecord
        The solver settings and the schedule taken.
    noise : NoiseRecord or None
        The shot-noise realisation, or ``None`` for a noiseless run.
    noise_resolution : NoiseResolution or None
        How the noise model was chosen (SEP 37): the requested and the resolved
        model and any fallback's reason. Outside the hash; ``None`` when the record's
        builder did not say, as for a record built by hand.
    defaults : DefaultsResolution or None
        Which of the integrator, the controller and the collar scheme took the
        route's default. Outside the hash; ``None`` when the record's builder did not
        say.
    backend : BackendRecord or None
        What executed the run, written by the driver from the backend's own
        ``metadata()``; ``None`` for a record built before a backend existed. Outside
        the hash, like the versions and the provenance beside it: a run continued on
        the other backend is the same run.
    versions : dict
        Package versions, from :func:`~solprop.output.provenance.package_versions`.
        Outside the hash.
    git : GitProvenance or None
        Provenance of the running checkout, or ``None``. Outside the hash.
    created : str
        When the run started, ISO 8601 in UTC. Outside the hash.
    schema_version : int
        Version of this record layout, :data:`SCHEMA_VERSION`; any other is refused.
    spectral_convention : str
        Sign convention of the stored field, always ``'optics'``.
    field_layout : str
        Axis order of the stored field.
    spatial : collections.abc.Mapping or None
        Spatial geometry, axis meanings, integration measures, field units and the
        plane measure (``flux_model`` and ``flux_plan``, SEP 34). Required for the
        radial, free and slab routes and ``None`` for modal runs. Included in the
        resume key.
    modal : collections.abc.Mapping or None
        The identity of a **general modal guide**: the ordered channel labels, the
        table, metric, response and provider identities, the sampling the reference
        is rebuilt from, the gauge, the frame, the active band, the noise reference
        basis and the accessors this basis admits, as
        :func:`solprop.translate.general.modal_identity` writes it. ``None`` for
        every other route. Included in the resume key, less the accessor list, so
        that a run continued against a different guide, a different table or a
        reordered mode axis is refused by the field that changed
        (:mod:`solprop.output.capability`).
    materials : solprop.physdata.selection.MaterialManifest
        The run's **material manifest**: the optmat version, the
        identity of every optmat model the run resolved, its species, compositions
        and solids, the policies in force and what was measured about them. Empty
        for a run of solprop's own materials. Its identities and policies are in the
        resume key (:meth:`~solprop.physdata.selection.MaterialManifest.hashed`), so
        a run continued with another model, policy or optmat version is refused by
        the entry that changed.
    initial_z : float
        Absolute launch position in metres; zero for a run launched at the entrance.
        Included in the resume key.
    tdse : collections.abc.Mapping or None
        Resolved scientific atomic descriptor, included in the resume key; ``None``
        for a run without a TDSE response. Provider execution and device facts
        belong to the backend record.
    quadrature : collections.abc.Mapping or None
        The transverse rule of a resolved modal run, as
        :meth:`solprop.modes.quadrature.TransverseQuadrature.record` writes it: the
        family, the counts, the nodes and the physical weights, in the resume key
        (SEP 19). ``None`` for the mode-averaged and spatial routes.
    reference : collections.abc.Mapping or None
        The plane-reference record (SEP 31): ``carrier``, the
        grid's reference frequency in rad/s; ``origin``, the reference of the run's
        path origin (phase, delay, calibrated), inherited from a parent plane for a
        run launched from one; ``parent``, that parent's provenance or ``None``;
        ``frame``, the frame integral of
        :func:`solprop.translate.coupling.frame_record`, or ``None`` for an operator
        supplied without frame tables; and ``basis``, the route and the basis
        identity (:func:`solprop.coupling.plane.basis_identity`) at each planned plane
        ``z``, the launch plane first. :func:`solprop.coupling.plane.reference_at`
        rebuilds any saved plane's reference from it. Derived numbers, so outside
        the hash; the inherited origin enters it through
        ``configuration["coupling"]``. Every run records one; ``None`` on a record
        built without it, as by hand, whose planes cannot be exchanged
        (:func:`solprop.processing.planes.field_plane`).
    transverse : collections.abc.Mapping or None
        A free-space run's resolved transverse boundary (SEP 40), as
        :meth:`solprop.spatial_absorber.SpatialAbsorberDesign.record` writes it: the
        safety ramp's start and strength, and a collar's strength, band, threshold
        and step limit. Derived numbers, so outside the hash; the specification is in
        ``configuration["spatial_absorber"]`` where one was given. ``None`` without
        a transverse boundary.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output.metadata import RunMetadata
    >>> from solprop.output.records import GridRecord, StepperRecord
    >>> from solprop.stepper import DOPRI5, PIController
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> metadata = RunMetadata(
    ...     grid=GridRecord.from_grid(grid),
    ...     modes=(),
    ...     simulation_kind="mode_averaged",
    ...     configuration={"gamma": 0.11},
    ...     stepper=StepperRecord.from_settings(
    ...         DOPRI5, PIController(rtol=1e-8, atol=0.0)
    ...     ),
    ... )
    >>> len(metadata.configuration_hash())
    64
    >>> metadata.spectral_convention
    'optics'
    """

    grid: GridRecord
    modes: tuple[ModeRecord, ...]
    simulation_kind: SimulationKind
    configuration: Mapping[str, object]
    stepper: StepperRecord
    noise: NoiseRecord | None = None
    backend: BackendRecord | None = None
    versions: Mapping[str, str] = dataclasses.field(default_factory=package_versions)
    git: GitProvenance | None = dataclasses.field(default_factory=GitProvenance.detect)
    created: str = ""
    schema_version: int = SCHEMA_VERSION
    spectral_convention: str = SPECTRAL_CONVENTION
    field_layout: str = FIELD_LAYOUT
    spatial: Mapping[str, object] | None = None
    modal: Mapping[str, object] | None = None
    materials: MaterialManifest = dataclasses.field(default_factory=MaterialManifest)
    initial_z: float = 0.0
    tdse: Mapping[str, object] | None = None
    quadrature: Mapping[str, object] | None = None
    reference: Mapping[str, object] | None = None
    transverse: Mapping[str, object] | None = None
    noise_resolution: NoiseResolution | None = None
    defaults: DefaultsResolution | None = None
    _stored: dict[str, JsonValue] | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False, metadata={"record": False}
    )
    _stored_arrays: dict[str, np.ndarray] | None = dataclasses.field(
        default=None, init=False, repr=False, compare=False, metadata={"record": False}
    )

    def __repr__(self) -> str:
        """Return a one-line description: what kind of run, on what, made when.

        Returns
        -------
        str
            For example ``RunMetadata(modal, field grid of 8192, 2 modes, native,
            2026-09-17T12:00:00+00:00)``.
        """
        backend = "no backend" if self.backend is None else self.backend.name
        return line(
            "RunMetadata",
            [
                ("", self.simulation_kind),
                ("", f"{self.grid.kind} grid of {self.grid.n_t}"),
                ("", f"{len(self.modes)} modes"),
                ("", backend),
                ("", self.created or "undated"),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        grid = self.grid
        kinds = [mode.kind.rsplit(".", 1)[-1] for mode in self.modes]
        versions = ", ".join(f"{name} {value}" for name, value in self.versions.items())
        return f"RunMetadata: a {self.simulation_kind} run", [
            (
                "grid",
                f"{grid.kind}, {grid.n_t} samples, reference "
                f"{engineering(grid.reference_wavelength, 'm')}, length "
                f"{engineering(grid.length, 'm')}",
            ),
            ("modes", ", ".join(kinds) or "none"),
            ("backend", "none recorded" if self.backend is None else self.backend.name),
            ("created", self.created or "undated"),
            (
                "convention",
                f"{self.spectral_convention} spectra, {self.power_convention} "
                f"amplitudes, schema {self.schema_version}",
            ),
            ("versions", versions),
            ("configuration", self.configuration_hash()),
            *_modal_rows(self.modal),
            *_material_rows(self.materials),
        ]

    def __post_init__(self) -> None:
        """Refuse another schema, stamp the creation time and adopt plain copies.

        Raises
        ------
        OutputError
            If the record declares an output schema other than :data:`SCHEMA_VERSION`,
            if `initial_z` is not finite and nonnegative, or if a spatial record is
            missing or names no plane measure.
        """
        _refuse_schema(self.schema_version)
        if not self.created:
            object.__setattr__(self, "created", datetime.now(UTC).isoformat())
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "configuration", dict(self.configuration))
        object.__setattr__(self, "versions", dict(self.versions))
        for name in (
            "spatial",
            "modal",
            "reference",
            "tdse",
            "quadrature",
            "transverse",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, dict(value))
        if not np.isfinite(self.initial_z) or self.initial_z < 0:
            raise OutputError("initial_z must be finite and nonnegative")
        if self.simulation_kind in ("radial", "free", "slab") and self.spatial is None:
            msg = "spatial metadata requires a spatial geometry record"
            raise OutputError(msg)
        if self.spatial is not None and "flux_model" not in self.spatial:
            msg = (
                "spatial metadata must name its flux_model and carry its flux_plan "
                "(SEP 34)"
            )
            raise OutputError(msg)

    @property
    def transverse_measure(self) -> TransverseMeasure:  # numpydoc ignore=RT01
        """What the run's transverse integrals measure.

        A run on modes uniform in y (:class:`solprop.modes.slab.SlabMode`) records
        ``configuration["transverse_measure"] = "per_unit_width"``: its powers are
        W/m, its energies J/m and its intensities still W/m². Every other run measures
        an area. The key sits in the free-form configuration and is written only
        where it is not the default.

        ``"area"`` or ``"per_unit_width"``.

        Raises
        ------
        OutputError
            If the stored value is neither.

        Examples
        --------
        >>> from solprop.output.metadata import _example_metadata
        >>> _example_metadata().transverse_measure
        'area'
        """
        value = self.configuration.get("transverse_measure", "area")
        if value not in ("area", "per_unit_width"):
            msg = (
                f"configuration.transverse_measure is {value!r}; a run measures "
                "'area' or 'per_unit_width'"
            )
            raise OutputError(msg)
        return cast("TransverseMeasure", value)

    @property
    def power_convention(self) -> str:  # numpydoc ignore=RT01
        """Amplitude label of the stored field.

        Always ``"poynting"``: the interface modal amplitude, whose squared
        modulus is the true power in watts (``docs/dev/conventions.md`` §4).
        """
        return "poynting"

    def hashed(self) -> dict[str, JsonValue]:
        """Return the part of the record the resume key is taken over.

        Everything that describes *what was computed*, and nothing that describes
        *when* or *with which build*: the wall-clock time, the package versions and the
        git provenance are excluded, so that a resume survives an unrelated commit or a
        dirty working tree, as long as they move no number the record holds; a new
        library or platform may move one, and the resume is then refused.
        ``modal.accessors`` is excluded for the same reason -- it is what this build can
        compute, not what this run did (``_hashed_modal``).

        Returns
        -------
        dict
            The serialised schema version, grid, modes, simulation kind, configuration,
            noise and stepper settings, the spatial record, the power convention, the
            general modal identity, the material manifest's identities and policies,
            the launch position, the TDSE descriptor and the transverse quadrature.

        Examples
        --------
        >>> from solprop.output.metadata import _example_metadata
        >>> sorted(_example_metadata().hashed())[:4]
        ['configuration', 'grid', 'initial_z', 'materials']
        """
        return dict(self._hashed_record)

    @functools.cached_property
    def _hashed_record(self) -> dict[str, JsonValue]:
        """The serialised hashed part, built once per record (:meth:`hashed`).

        A run asks for it several times -- the resume key twice and the stored JSON
        once -- and serialising the configuration was most of an output's opening
        (SEP 25 section 6.3). The record is frozen, so the first answer stands;
        :meth:`hashed` hands out a shallow copy, which its callers extend.
        """
        if self._stored is not None:
            return self._stored
        return {
            "schema_version": self.schema_version,
            "simulation_kind": self.simulation_kind,
            "grid": serialise(self.grid),
            "modes": serialise(list(self.modes)),
            "configuration": serialise(dict(self.configuration)),
            "noise": serialise(self.noise),
            "stepper": serialise(self.stepper),
            "spatial": serialise(self.spatial),
            "power_convention": self.power_convention,
            # All of the general-guide identity but the accessor list, which is the
            # build's capability and not the run's description (_hashed_modal).
            "modal": _hashed_modal(self.modal),
            # The material manifest's identities and policies; a run of solprop's
            # own materials keys on `None`.
            "materials": serialise(self.materials.hashed()),
            "initial_z": self.initial_z,
            "tdse": serialise(self.tdse),
            "quadrature": serialise(self.quadrature),
        }

    def configuration_hash(self) -> str:
        """Return the resume key: a digest of everything that was computed.

        A SHA-256 of the canonical JSON of :meth:`hashed`, so it is stable across
        processes -- unlike :func:`hash`, which is salted per interpreter. It covers the
        physics and the numerics, and not only the statistic names and the state shape,
        so a resume continues exactly when the two hashes are equal (ADR-066).

        Returns
        -------
        str
            Sixty-four lowercase hexadecimal characters.

        Examples
        --------
        >>> from solprop.output.metadata import _example_metadata
        >>> metadata = _example_metadata()
        >>> metadata.configuration_hash() == metadata.configuration_hash()
        True
        """
        return self._configuration_digest

    @functools.cached_property
    def _configuration_digest(self) -> str:
        """The resume key, computed once per record (:meth:`configuration_hash`)."""
        return canonical_digest(self._hashed_record)

    def with_backend(self, backend: BackendRecord) -> "RunMetadata":
        """Return this record with a run's execution facts, its resume key kept.

        The backend record is outside the hash, so the hashed part and the resume key
        already computed stand: a run opening its output does not serialise and digest
        its configuration a second time, which cost a capillary run 30 ms and a large
        launch seconds (solprop issue #347).

        Parameters
        ----------
        backend : BackendRecord
            The execution facts of the prepared backend.

        Returns
        -------
        RunMetadata
            A copy of this record with `backend` in place of its own.

        Examples
        --------
        >>> from solprop.output.metadata import _example_metadata
        >>> from solprop.output.provenance import BackendRecord
        >>> metadata = _example_metadata()
        >>> backend = BackendRecord(name="numpy_ref", propagator_mode="tabulate")
        >>> executed = metadata.with_backend(backend)
        >>> executed.backend.name
        'numpy_ref'
        >>> executed.configuration_hash() == metadata.configuration_hash()
        True
        """
        replaced = dataclasses.replace(self, backend=backend)
        # functools.cached_property stores in the instance dictionary, past the
        # frozen dataclass's __setattr__.
        for name in ("_hashed_record", "_configuration_digest"):
            if name in self.__dict__:
                replaced.__dict__[name] = self.__dict__[name]
        return replaced

    def differences(self, other: "RunMetadata") -> tuple[str, ...]:
        """Return the dotted paths at which two records describe different runs.

        This is what a refused resume names, so that the reason is the field that
        changed rather than "the hash does not agree".

        Parameters
        ----------
        other : RunMetadata
            The record to compare against.

        Returns
        -------
        tuple
            Dotted paths into :meth:`hashed`, in depth-first order; empty when the two
            describe the same run.

        Examples
        --------
        >>> import dataclasses
        >>> from solprop.output.metadata import _example_metadata
        >>> one = _example_metadata()
        >>> two = dataclasses.replace(one, configuration={"gamma": 0.22})
        >>> one.differences(two)
        ('configuration.gamma',)
        """
        found: list[str] = []
        _differences(self.hashed(), other.hashed(), "", found)
        return tuple(found)

    def summarise_differences(
        self,
        other: "RunMetadata",
        *,
        shown: int = _DIFFERENCES_SHOWN,
    ) -> str:
        """Say how two records differ, briefly enough to read in a refusal.

        The paths of :meth:`differences`, with every array's element paths grouped
        under the array's name and counted. Every part of the record that changed
        (``configuration``, ``materials``, ``noise``, ``schema_version``, ...) is
        named, with its first `shown` entries and a count of the rest, so that no
        cause is hidden behind another part's many entries. Where the noise differs
        and this record's realisation was drawn from a known seed, the summary says
        which seed draws it again.

        Parameters
        ----------
        other : RunMetadata
            The record to compare against; this one is the stored record.
        shown : int, optional
            How many entries to name in each part of the record.

        Returns
        -------
        str
            ``"these fields changed: ..."``, empty of entries when the two describe
            the same run.

        Examples
        --------
        >>> import dataclasses
        >>> from solprop.output.metadata import _example_metadata
        >>> one = _example_metadata()
        >>> two = dataclasses.replace(
        ...     one, configuration={"gamma": 0.22, "betas": [1.0, 2.0, 3.0]}
        ... )
        >>> one.summarise_differences(two)
        'these fields changed: configuration.betas, configuration.gamma'
        >>> three = dataclasses.replace(one, configuration={"gamma": [0.1, 0.2, 0.3]})
        >>> three.summarise_differences(
        ...     dataclasses.replace(one, configuration={"gamma": [0.4, 0.5, 0.6]})
        ... )
        'these fields changed: configuration.gamma (3 elements)'
        """
        # An array is recorded as its real and imaginary parts' nested lists, so an
        # element is a stem and its trailing indices, counted once for both parts.
        groups: dict[str, dict[str, str]] = {}
        for path in self.differences(other):
            container = _ELEMENT_INDICES.sub("", path)
            stem = _ARRAY_PART.sub("", container)
            groups.setdefault(stem, {}).setdefault(path[len(container) :], path)
        parts: dict[str, list[str]] = {}
        for stem, elements in groups.items():
            part = _PATH_SEPARATOR.split(stem, maxsplit=1)[0]
            parts.setdefault(part, []).append(
                next(iter(elements.values()))
                if len(elements) == 1
                else f"{stem} ({len(elements)} elements)"
            )
        named: list[str] = []
        for part, entries in parts.items():
            named.extend(entries[:shown])
            if len(entries) > shown:
                named[-1] += f" (and {len(entries) - shown} more in {part})"
        text = ", ".join(named)
        stored = self.noise
        if "noise" in parts and stored is not None and stored.seed is not None:
            text += (
                f"; the stored run drew its {stored.model} noise from seed "
                f"{stored.seed}: pass seed={stored.seed} to draw it again"
            )
        return f"these fields changed: {text}"

    def record_arrays(self) -> dict[str, np.ndarray]:
        """Return the arrays stored beside the record, by digest.

        What :meth:`to_json`'s references name: each numeric array of more than
        :data:`~solprop.output.serialisation.RECORD_ARRAY_LIMIT` elements. A store keeps
        them with the document, and :meth:`from_json` takes them back.

        Returns
        -------
        dict
            The arrays, not copied, each by the digest of its bytes with its dtype and
            shape.

        Examples
        --------
        >>> import dataclasses
        >>> import numpy as np
        >>> from solprop.output.metadata import _example_metadata
        >>> metadata = dataclasses.replace(
        ...     _example_metadata(), configuration={"rate": np.zeros(5000)}
        ... )
        >>> [array.size for array in metadata.record_arrays().values()]
        [5000]
        """
        found = dict(self._stored_arrays or {})

        def _walk(value: object) -> None:
            """Collect the arrays of a serialised structure."""
            if isinstance(value, _ArrayReference):
                found[str(value[_RECORD_ARRAY])] = value.array
            elif isinstance(value, dict):
                for item in value.values():
                    _walk(item)
            elif isinstance(value, list):
                for item in value:
                    _walk(item)

        _walk(self._document)
        return found

    def to_json(self) -> str:
        """Return the whole record as one JSON document.

        An array of more than :data:`~solprop.output.serialisation.RECORD_ARRAY_LIMIT`
        elements is a reference to what :meth:`record_arrays` holds.

        Returns
        -------
        str
            Canonical JSON carrying the hashed part plus the whole general modal
            identity, whose published accessor list the hashed part leaves out, the
            whole material manifest, the plane-reference record, the noise model's
            resolution, the conventions, the backend record, the versions, the git
            provenance and the creation time.

        Examples
        --------
        >>> import json
        >>> from solprop.output.metadata import _example_metadata
        >>> sorted(json.loads(_example_metadata().to_json()))[:3]
        ['backend', 'configuration', 'created']
        """
        return canonical_json(self._document)

    @functools.cached_property
    def _document(self) -> dict[str, JsonValue]:
        """The whole serialised record, built once per record (:meth:`to_json`).

        - Returns: the hashed part and what is stored beside it, JSON-ready.
        """
        document = self.hashed()
        # The whole identity, including the accessor list `hashed` leaves out: what a
        # file records and what its resume key is taken over are two different
        # questions, and the published accessors answer only the first
        # (:func:`_hashed_modal`).
        document["modal"] = serialise(self.modal)
        # The whole manifest, measurements included, beside its hashed part.
        document["materials"] = serialise(self.materials.to_dict())
        # Derived from the hashed problem, so stored beside the key, not in it: a
        # resume on another build may evaluate the frame to a different last bit.
        document["reference"] = serialise(self.reference)
        # The transverse boundary's derived values (SEP 40), from a root-finder:
        # beside the key, so that a resume on another build is not refused.
        document["transverse"] = serialise(self.transverse)
        # How the model was chosen, beside the key: the resolved model is in the
        # hashed noise record already, and a request is not part of the physics.
        document["noise_resolution"] = serialise(self.noise_resolution)
        # Which numerics took the route's default, beside the key for the same reason.
        document["defaults"] = serialise(self.defaults)
        document["spectral_convention"] = self.spectral_convention
        document["field_layout"] = self.field_layout
        document["backend"] = serialise(self.backend)
        document["versions"] = serialise(dict(self.versions))
        document["git"] = serialise(self.git)
        document["created"] = self.created
        return document

    @classmethod
    def from_json(
        cls,
        text: str,
        arrays: Mapping[str, object] | None = None,
    ) -> "RunMetadata":
        """Return the record a JSON document describes.

        The configuration comes back as the plain structures
        :func:`~solprop.output.serialisation.serialise` made of it, not as the physics
        objects: a callable was recorded by ``repr`` and cannot be rebuilt. That is
        enough for the hash, which is what a resume compares.

        Parameters
        ----------
        text : str
            A document produced by :meth:`to_json`.
        arrays : collections.abc.Mapping or None, optional
            What :meth:`record_arrays` gave for it, by digest, or an HDF5 group holding
            them. Each comes back in place of its reference, its digest checked.

        Returns
        -------
        RunMetadata
            The record, hashing identically to the one that produced ``text``.

        Raises
        ------
        OutputError
            If the document is not valid JSON, declares an output schema other than
            :data:`SCHEMA_VERSION`, a required field is missing, or an array it names
            is absent from `arrays` or does not have its digest.

        Examples
        --------
        >>> from solprop.output.metadata import _example_metadata
        >>> metadata = _example_metadata()
        >>> back = RunMetadata.from_json(metadata.to_json())
        >>> back.configuration_hash() == metadata.configuration_hash()
        True
        """
        try:
            document = json.loads(text)
        except json.JSONDecodeError as error:
            msg = f"stored metadata is not valid JSON: {error}"
            raise OutputError(msg) from error
        if not isinstance(document, dict) or "schema_version" not in document:
            msg = "stored metadata is missing or malformed: no schema_version"
            raise OutputError(msg)
        # The version first: a file of another schema is refused by its version,
        # not by the first field its layout lacks.
        _refuse_schema(document["schema_version"])
        # The record as written, references and all, is what digests to its stamp.
        written = document
        held: dict[str, np.ndarray] = {}
        document = cast("dict[str, Any]", _resolved(document, arrays or {}, held))

        # Another build of this schema may lack a top-level entry.
        stored = cast(
            "Callable[..., Any]", functools.partial(_stored_entry, document, path=())
        )

        try:
            noise = stored("noise")
            resolution = stored("noise_resolution")
            # Absent from a record written before the defaults were recorded.
            defaults = stored("defaults")
            git = stored("git")
            backend = stored("backend")
            metadata = cls(
                grid=_rebuild(GridRecord, document["grid"], ("grid",)),
                modes=tuple(
                    _rebuild(ModeRecord, one, ("modes", index))
                    for index, one in enumerate(document["modes"])
                ),
                simulation_kind=document["simulation_kind"],
                configuration=document["configuration"],
                stepper=StepperRecord.from_record(document["stepper"]),
                noise=None
                if noise is None
                else _rebuild(NoiseRecord, noise, ("noise",)),
                backend=None if backend is None else _rebuild(BackendRecord, backend),
                versions=cast("Mapping[str, str]", stored("versions", default={})),
                git=None if git is None else _rebuild(GitProvenance, git),
                created=str(stored("created", default="")),
                schema_version=document["schema_version"],
                spectral_convention=str(
                    stored("spectral_convention", default=SPECTRAL_CONVENTION)
                ),
                field_layout=str(stored("field_layout", default=FIELD_LAYOUT)),
                spatial=stored("spatial"),
                modal=stored("modal"),
                materials=MaterialManifest.from_dict(stored("materials")),
                initial_z=stored("initial_z", default=0.0),
                tdse=stored("tdse"),
                quadrature=stored("quadrature"),
                reference=stored("reference"),
                transverse=stored("transverse"),
                noise_resolution=None
                if resolution is None
                else _rebuild(NoiseResolution, resolution),
                defaults=None
                if defaults is None
                else _rebuild(DefaultsResolution, defaults),
            )
        except (KeyError, TypeError, MaterialError) as error:
            msg = f"stored metadata is missing or malformed: {error}"
            raise OutputError(msg) from error
        # The record as written: it digests to its own stamp, whatever today's code
        # would write.
        object.__setattr__(metadata, "_stored", _stored_hashed(written))
        object.__setattr__(metadata, "_stored_arrays", held)
        return metadata


def _example_metadata() -> RunMetadata:
    """Return a minimal record, so the examples above stay one line each.

    A documentation and experimentation helper, not part of the public surface: build a
    real record with the :class:`RunMetadata` constructor, whose own example shows how.
    """
    grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    return RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=(),
        simulation_kind="mode_averaged",
        configuration={"gamma": 0.11},
        stepper=StepperRecord(tableau="DOPRI5", controller={"rtol": 1e-8, "atol": 0.0}),
    )


def resolve_metadata_write(
    existing: RunMetadata | None, new: RunMetadata, *, force: bool = False
) -> bool:
    """Decide whether a handler may write metadata, under one rule for both handlers.

    Metadata is written once. A second write describing the same run is a no-op; one
    describing a different run is a mistake and raises, naming the fields that changed,
    unless the caller insists with ``force``, in which case the record is replaced and
    the replacement is logged.

    "The same run" means the same :meth:`RunMetadata.configuration_hash`, so a rerun of
    identical physics under a newer NumPy, or from a checkout that has since gained a
    commit, is still the same run.

    Parameters
    ----------
    existing : RunMetadata or None
        What the handler already holds, or ``None`` when it holds nothing.
    new : RunMetadata
        What the caller is offering.
    force : bool, optional
        Whether to overwrite a different record instead of refusing.

    Returns
    -------
    bool
        Whether the caller should write ``new``.

    Raises
    ------
    OutputError
        If a different record is already held and ``force`` is false.

    Examples
    --------
    >>> from solprop.output.metadata import _example_metadata, resolve_metadata_write
    >>> metadata = _example_metadata()
    >>> resolve_metadata_write(None, metadata)
    True
    >>> resolve_metadata_write(metadata, metadata)
    False
    """
    if existing is None:
        return True
    if existing.configuration_hash() == new.configuration_hash():
        return False
    changed = existing.summarise_differences(new)
    if not force:
        msg = (
            f"metadata has already been written for a different run; {changed}. "
            "Pass force=True to overwrite it."
        )
        raise OutputError(msg)
    logger.warning("overwriting run metadata; %s", changed)
    return True
