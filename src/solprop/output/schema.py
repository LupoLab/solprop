# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run metadata, saved-plane records and the one save protocol both handlers implement.

An output file has to answer one question years later: *what produced this?* The record
types here are that answer. :class:`RunMetadata` carries the grid, the modes, the
serialised physics configuration, the package versions, the git provenance of the
running checkout, the noise seed and realisation, and the stepper settings;
:class:`SavedPlane` carries one saved position with its field, its statistics and the
step history since the previous plane. :class:`Output` is the protocol
:class:`solprop.output.memory.MemoryOutput` and :class:`solprop.output.hdf5.HDF5Output`
both implement, so processing reads either through the same accessors.

**Conventions on the way out.** A ``SavedPlane`` holds its field in the internal FFT
convention, as a backend produces it; a handler converts it with
:func:`solprop.fourier.spectrum_to_optics` on the way to storage, so that what is stored
is the window-centred optics spectrum a spectrometer would agree with, marked
``spectral_convention = "optics"`` (``docs/dev/conventions.md`` §2; this module
and the handlers are boundary place 2). The noise realisation and ``noise_phase`` are
internal objects and stay in the FFT convention, marked so by
:data:`INTERNAL_CONVENTION`.

**Provenance, not source.** The provenance is the resolved configuration, the
package versions and the git commit with a dirty flag -- never the source text.

**The resume key.** :meth:`RunMetadata.configuration_hash` digests the schema version,
the grid, the modes, the configuration, the noise, the stepper settings, from schema 8
the identities and policies of the run's material manifest, and from schema 9 the
stepping method when it is not the tableau (SEP 13), and nothing else: not the
wall-clock time the run started, not the package versions, and not the git provenance,
so that a resume survives an unrelated commit or a dirty working tree. It is a SHA-256
of canonical JSON rather than :func:`hash`, so it is stable across processes, and it
covers the physics rather than only the statistic names and the state shape, so a
resume with different physics of the same shape is refused.

The manual page is ``docs/guide/results/output.md``.
"""

import dataclasses
import functools
import hashlib
import json
import logging
import math
import platform
import re
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Bool, Complex, Float, Int
from scipy.interpolate import CubicSpline

from solprop import fourier
from solprop._display import Displayed, engineering, line, summarise
from solprop.exceptions import MaterialError, OutputError
from solprop.grid import EnvelopeGrid, TimeGrid
from solprop.modes.material import JsonValue
from solprop.modes.protocol import Mode, TransverseMeasure
from solprop.noise import NoiseRealisation
from solprop.physdata.selection import MaterialManifest
from solprop.stepper import FixedSchedule, PIController, Tableau

__all__ = [
    "FIELD_LAYOUT",
    "HISTORY_FIELDS",
    "INLINE_ARRAY_LIMIT",
    "INTERNAL_CONVENTION",
    "INTERPOLATED_RESUMES_ANNOTATION",
    "PEAK_FIELDS_ANNOTATION",
    "SCHEMA_VERSION",
    "SPECTRAL_CONVENTION",
    "STEP_STATS_INDEX",
    "ArrayDigest",
    "BackendRecord",
    "CheckpointOutput",
    "Convention",
    "FieldRetention",
    "FieldsChoice",
    "GitProvenance",
    "GridRecord",
    "JsonValue",
    "MarchSnapshot",
    "ModeRecord",
    "NoiseRecord",
    "NoiseResolution",
    "Output",
    "PlaneLayout",
    "PlaneStore",
    "RunMetadata",
    "RunOutcome",
    "SavedPlane",
    "SimulationKind",
    "StepHistory",
    "StepLimits",
    "StepStats",
    "StepperRecord",
    "TerminalCheckpoint",
    "field_retention_refusal",
    "package_versions",
    "recorded_input",
    "resolve_metadata_write",
    "serialise",
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

SCHEMA_VERSION = 27
"""Version of the record layout; incremented whenever a stored field changes meaning.

Version 9 adds the SEP 13 stepping method. Independent version-10 branches added
SEP 12 launch/TDSE identities and terminal checkpoints, and SEP 14 route-dependent
step limits. Version 11 on main adds interpolated and natural-step save policies.
Version 12 combines both histories. Version 13 (SEP 16) lets
``StepperRecord.method`` name the Adams method's formulation: the exponential one
appears in its representation, while the default interaction formulation is left out,
so a version-12 method string and its configuration hash are unchanged. Version 14
records the controller's norm ``region``; a version-13 record had none, and scored the
requested band. Version 15 names the formulation in every Adams method string, the
interaction one included, now that the exponential one is the default; a method string
of an earlier version that names none was stepped in the interaction picture. Version
16 (SEP 18) records the field plasma column's band integration rule: the configuration
carries ``plasma_integration``, its taps, band edge and policies, and the file's root
attributes repeat the rule, ``K``, ``theta_b``, the taps and the certified error. A
run integrated with the plain trapezoid records neither, as before. Version 17 (SEP
19) records the transverse quadrature of a resolved modal run -- its family, requested
and realised counts, nodes and physical weights -- in ``quadrature`` and its resume key,
so that a run continued on another rule is refused. A record older than 17 has none,
which means "not recorded", never a field-scale rule. Version 18 (SEP 20) admits
``model="hacc"`` TDSE records, atomic windows and ``consistency`` provenance: a haCC or
windowed response serialises its ``model`` and ``atomic_window`` and its descriptor is
``solprop.tdse-descriptor.v3``, and the consistency record travels with the execution
facts, outside the hash. An SAE record without a window keeps its keys, so its
configuration and descriptor identities are those of version 17. Version 18 also
admits an optional raw-archive reference (SEP 20 phase 2): the TDSE execution record,
in ``backend.settings["tdse"]["archive"]`` and in each outcome's ``execution``, names
the archive's schema, run UUID and path, outside the hash; a run without an archive
records none. Version 19 (SEP 24 phase 1) lets ``StepperRecord.method`` name the
low-storage march, ``LowStorageRungeKutta(scheme='rk4ip')``, which replays a schedule
with no error estimate: its step history records the norm 0.0 on every step, meaning
"not estimated", never a step that met its tolerance exactly, and a record older than 19
never carries such a norm for that reason. Version 20 (SEP 22) makes the stored field
optional: a run keeps it on every plane, on the planes it names or on none
(:class:`FieldRetention`), the file's root attribute ``fields`` says which and a
``field_planes`` dataset lists the planes that carry one, and a field-free file has no
``field`` or ``noise_phase`` dataset. From version 20 an array input above
:data:`INLINE_ARRAY_LIMIT` elements is recorded by its content digest
(:class:`ArrayDigest`) rather than inline, and a field added to a recorded dataclass
after its record existed was left out while it held its default. Version 21 records
every field of a recorded dataclass, at its default too, and a scalar Raman response
with all its fields: no record leaves out what an earlier record lacked, so every
configuration hash moved. Loading older records preserves their original field
presence and configuration hash. Version 22 (SEP 27) changes the field-plasma source
to the held-current equation and records ``plasma_source`` in its configuration:
the source form, certificate limit and taper sample counts. The measured
full-transform steady-current certificate is execution evidence under
``backend.settings["plasma_source"]``, outside the hash. Every previous schema has a
different resume key. A private gated benchmark
carries ``_benchmark_plasma_source`` and refuses continuation even under this schema.
Version 23 (SEP 34) measures every spatial energy, power, fluence and intensity as the
axial Poynting flux of the medium at the plane: the spatial record names the
``flux_model`` and carries the run's plane-measure coefficients (``flux_plan``), from
which a saved plane's admittance at its own position is rebuilt. Earlier spatial
records used the vacuum measure and are refused by the processing that needs one.
Version 24 (SEP 31) carries the plane-reference record (``RunMetadata.reference``):
the antiderivatives of the frame quantities the run's linear operator removed, the
frequency origin of its linear term, the carrier, and the reference origin of its
path, which a run launched from another run's plane inherits with that parent's
identity (``configuration["coupling"]``). Any saved plane's absolute phase and time
reference is rebuilt from it alone. The tables are derived numbers, outside the hash;
the inherited origin is in the configuration, inside it. Planes of an earlier file
cannot be exchanged.

Version 25 (SEP 33) records every run's ``precision`` in its hashed configuration,
``"double"`` included, so a resume in the other precision is refused through the hash.
The saved arrays are binary64 either way. Version 26 (SEP 37) records how the run's
noise model was resolved (``RunMetadata.noise_resolution``): the model the caller
asked for, ``"default"`` when the keyword was omitted, the model the run carries, and
why a default fell back to none. It sits beside the hash, not in it, because the
resolved model is already there through the noise record. A record older than 26 has
none.

Version 27 (SEP 21) records ``Absorber.spatial``, and a GNLSE or spatial run that
declares no absorber records the default collar it now has rather than none.
"""

INLINE_ARRAY_LIMIT = 2**20
"""The most elements an array input is recorded with inline (SEP 22, Q4).

A larger array -- a gigabyte spatial launch -- is recorded by an :class:`ArrayDigest`
of its bytes, dtype and shape instead, which costs seconds of hashing per input rather
than tens of gigabytes of JSON, and still changes with any one sample.
"""

SPECTRAL_CONVENTION = "optics"
"""Sign convention of every spectrum a handler stores."""

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

INTERNAL_CONVENTION = "fft"
"""Sign convention of the arrays that stay internal: the noise field and its phase."""

FIELD_LAYOUT = "(n_z, n_modes, n_pol, n_omega)"
"""Axis order of the stored field, as ``docs/dev/conventions.md`` §6 fixes it."""

SimulationKind = Literal["mode_averaged", "multimode", "radial", "free", "slab"]
"""Which transform the run used."""

Convention = Literal["optics", "fft"]
"""Which sign convention an accessor should hand back."""

# JsonValue is imported above and re-exported here, where the file format is
# owned. It is defined in solprop.modes.material because this module imports
# solprop.modes.protocol: defining it here and importing it there would close a
# cycle through the output layer.

# The subprocess calls that read the git provenance are bounded: a hung git must not
# stall a run that is otherwise ready to start.
_GIT_TIMEOUT = 5.0


# ------------------------------------------------------------------- serialisation


def _tag_unreproducible(value: object, key: str) -> dict[str, JsonValue]:
    """Return the record of an object that cannot be rebuilt from the file.

    Parameters
    ----------
    value : object
        The object to describe.
    key : str
        Tag naming why it is irreducible, ``'__callable__'`` or ``'__repr__'``.

    Returns
    -------
    dict
        A record holding the object's ``repr`` and ``reproducible: False``.
    """
    logger.debug("configuration entry recorded by repr only: %s", key)
    return {key: repr(value), "reproducible": False}


def _serialise_dataclass(value: object) -> dict[str, JsonValue]:
    """Return a dataclass instance as a tagged dictionary of its constructor fields.

    Fields declared ``init=False`` are derived in ``__post_init__`` from the ones that
    are, so recording them would store the same information twice and make two runs
    that differ in nothing hash differently if a derived quantity ever changes form.

    A field whose metadata says ``record=False`` is a live handle that the record
    carries elsewhere, and is left out for the same reason: a provider channel's
    :class:`~solprop.modes.guide.ChannelSource` holds the provider object, whose
    identity the run records once beside its channels.

    Parameters
    ----------
    value : object
        A dataclass instance.

    Returns
    -------
    dict
        ``{'__class__': 'module.QualName', <field>: <serialised value>, ...}``.
    """
    kind = type(value)
    name = f"{kind.__module__}.{kind.__qualname__}"
    record: dict[str, JsonValue] = {"__class__": name}
    # The parameter stays `object` because the test session resolves annotations at
    # run time, and `fields` wants typeshed's `DataclassInstance`, which does not exist
    # there; the caller has already checked that this is a dataclass instance.
    for entry in dataclasses.fields(value):  # type: ignore[arg-type] - checked by the caller
        if not entry.init or not entry.metadata.get("record", True):
            continue
        record[entry.name] = serialise(getattr(value, entry.name))
    return record


def _serialise_enum(value: Enum) -> dict[str, JsonValue]:
    """Return an enumeration member as its qualified class name and its value.

    The class matters as much as the value: two enumerations may share a member value
    and mean different things, and the configuration hash has to tell them apart.

    Parameters
    ----------
    value : enum.Enum
        The member.

    Returns
    -------
    dict
        ``{'__enum__': 'module.QualName', 'value': <serialised value>}``.
    """
    kind = type(value)
    return {
        "__enum__": f"{kind.__module__}.{kind.__qualname__}",
        "value": serialise(value.value),
    }


def _serialise_numpy(value: np.ndarray | np.generic) -> JsonValue:
    """Return a NumPy array as a nested list with its dtype, or a scalar as a scalar.

    A complex array is split into its real and imaginary parts, because JSON has no
    complex number and a nested list of them would not survive a round trip.

    Parameters
    ----------
    value : numpy.ndarray or numpy.generic
        The array or the scalar.

    Returns
    -------
    dict or bool or int or float or str
        A ``__ndarray__`` record for an array, and the plain Python value of a scalar.
    """
    if isinstance(value, np.generic):
        return serialise(value.item())
    record: dict[str, JsonValue] = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    if np.iscomplexobj(value):
        record["__ndarray__"] = np.real(value).tolist()
        record["__imaginary__"] = np.imag(value).tolist()
    else:
        # tolist gives JSON primitives for every numeric and boolean dtype; anything
        # else goes back through serialise, which records what it cannot reduce.
        record["__ndarray__"] = serialise(value.tolist())
    return record


def _serialise_sequence(value: object) -> list[JsonValue]:
    """Return a sequence or a set as a list, ordering set members deterministically.

    Parameters
    ----------
    value : object
        A sequence, a set or a frozenset.

    Returns
    -------
    list
        The serialised members. A set's are sorted by their canonical JSON text, so
        that the configuration hash does not depend on set iteration order.
    """
    # `object` is not declared iterable; the caller has already checked that this is
    # a set or a sequence.
    members = [serialise(item) for item in value]  # type: ignore[union-attr] - checked by the caller
    if isinstance(value, set | frozenset):
        return sorted(members, key=_canonical_json)
    return members


def _serialise_object(value: object) -> JsonValue:
    """Return a structured or opaque object as a JSON-safe structure.

    Parameters
    ----------
    value : object
        Anything that is not a primitive, an enumeration or a NumPy value.

    Returns
    -------
    NoneType or bool or int or float or str or list or dict
        A structure of JSON primitives.
    """
    if isinstance(value, Path):
        return str(value)
    if type(value) is CubicSpline:
        # Material tables own these interpolants. Their repr contains an address,
        # so record the complete polynomial instead of an irreproducible callable.
        return {
            "__class__": "scipy.interpolate.CubicSpline",
            "x": serialise(value.x),
            "c": serialise(value.c),
            "axis": serialise(value.axis),
            "extrapolate": serialise(value.extrapolate),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _serialise_dataclass(value)
    if isinstance(value, Mapping):
        return {str(key): serialise(item) for key, item in value.items()}
    if isinstance(value, set | frozenset | Sequence) and not isinstance(
        value, str | bytes
    ):
        return _serialise_sequence(value)
    return _tag_unreproducible(value, "__callable__" if callable(value) else "__repr__")


def serialise(value: object) -> JsonValue:
    """Return any configuration object as a JSON-safe structure, dropping nothing.

    Dataclasses become tagged dictionaries of their constructor fields (less a live
    handle whose field metadata says ``record=False``), enumerations
    their qualified name and value, NumPy arrays a nested list with their dtype and
    shape, cubic splines their knots, coefficients, axis and extrapolation setting,
    complex numbers a real/imaginary pair, and mappings and sequences are
    recursed. Anything that cannot be reduced -- a function, a lambda, an object with
    no structure -- is recorded as its ``repr`` and marked ``reproducible: False``,
    because a configuration that is silently incomplete is worse than one that says
    where it stops.

    The function is idempotent: applying it to its own output returns that output, so
    a round-tripped :class:`RunMetadata` hashes exactly as the original does.

    Parameters
    ----------
    value : object
        Anything at all.

    Returns
    -------
    NoneType or bool or int or float or str or list or dict
        A structure of JSON primitives, lists and string-keyed dictionaries.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.schema import serialise
    >>> serialise({"radius": 125e-6})
    {'radius': 0.000125}
    >>> serialise(np.array([1.0, 2.0]))["dtype"]
    'float64'
    >>> serialise(abs)["reproducible"]
    False
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return _serialise_enum(value)
    if isinstance(value, int | float | str):
        return value
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, np.ndarray | np.generic):
        return _serialise_numpy(value)
    return _serialise_object(value)


@dataclass(frozen=True)
class ArrayDigest:
    """An array recorded by the digest of its contents rather than inline.

    What a run record holds for an array input above :data:`INLINE_ARRAY_LIMIT`
    elements (SEP 22 section 5.2, Q4): the SHA-256 of its bytes in little-endian order,
    with its dtype and shape. Two inputs that differ in one sample record different
    digests, so their runs have different configuration hashes; the array itself is
    not recoverable from the record.

    Attributes
    ----------
    sha256 : str
        Sixty-four lowercase hexadecimal characters.
    dtype : str
        The array's dtype, as NumPy names it (``"complex128"``).
    shape : tuple of int
        The array's shape.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.schema import ArrayDigest
    >>> digest = ArrayDigest.of(np.zeros((2, 3)))
    >>> digest.dtype, digest.shape, len(digest.sha256)
    ('float64', (2, 3), 64)
    """

    sha256: str
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def of(cls, values: np.ndarray) -> "ArrayDigest":  # noqa: UP037 - bound after the class body
        """Return the digest of an array's contents.

        Parameters
        ----------
        values : numpy.ndarray
            The array; read once, in little-endian byte order, without a copy when it
            is already contiguous and little-endian.

        Returns
        -------
        ArrayDigest
            The digest, the dtype and the shape.
        """
        array = np.asarray(values)
        # One byte order, so that the same numbers digest alike on every platform; a
        # copy only where the array is not contiguous in it already.
        ordered = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        return cls(
            sha256=hashlib.sha256(memoryview(ordered).cast("B")).hexdigest(),
            dtype=str(ordered.dtype),
            shape=tuple(int(size) for size in array.shape),
        )


def recorded_input(value: object) -> object:
    """Return an input as the run record carries it: inline, or by its digest.

    Parameters
    ----------
    value : object
        A pulse, a spatial pulse or an array input.

    Returns
    -------
    object
        `value` itself, unless it is an array of more than :data:`INLINE_ARRAY_LIMIT`
        elements, which is recorded as its :class:`ArrayDigest`.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.schema import recorded_input
    >>> recorded_input(np.zeros(4)).shape
    (4,)
    >>> type(recorded_input(np.zeros(2**20 + 1))).__name__
    'ArrayDigest'
    """
    if isinstance(value, np.ndarray) and value.size > INLINE_ARRAY_LIMIT:
        return ArrayDigest.of(value)
    return value


def _canonical_json(value: JsonValue) -> str:
    """Return the one textual form of a structure that a digest may be taken over.

    Parameters
    ----------
    value : NoneType or bool or int or float or str or list or dict
        A structure of JSON primitives.

    Returns
    -------
    str
        Compact JSON with sorted keys, so that two equal structures built in different
        orders produce identical text.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _decode_array(record: Mapping[str, object]) -> np.ndarray:
    """Return the array a ``__ndarray__`` record describes.

    The nested list :meth:`numpy.ndarray.tolist` produced already carries the shape, so
    only the dtype has to be restored, and the imaginary part rejoined when there is
    one.

    Parameters
    ----------
    record : collections.abc.Mapping
        A record produced by :func:`serialise` from a NumPy array.

    Returns
    -------
    numpy.ndarray
        The array, with its original dtype and shape.
    """
    dtype = np.dtype(str(record["dtype"]))
    if "__imaginary__" in record:
        real = np.array(record["__ndarray__"], dtype=np.float64)
        # Assigning the two views rather than forming real + 1j*imaginary: complex
        # arithmetic does not preserve a signed zero, and the digest is taken over the
        # text of these numbers, where -0.0 and 0.0 differ.
        values = np.empty(real.shape, dtype=dtype)
        values.real = real
        values.imag = np.array(record["__imaginary__"], dtype=np.float64)
        return values
    return np.array(record["__ndarray__"], dtype=dtype)


def _decode(value: JsonValue) -> object:
    """Return a serialised value with an array field rebuilt.

    An array is the only form a record type holds that JSON cannot carry directly, so
    it is the only one rebuilt; a nested configuration dictionary is left as plain
    JSON, because the physics objects it describes are not reconstructible in general
    (a callable was recorded by ``repr``).

    Parameters
    ----------
    value : NoneType or bool or int or float or str or list or dict
        A value produced by :func:`serialise`.

    Returns
    -------
    object
        The rebuilt value, or ``value`` itself when there is nothing to rebuild.
    """
    if isinstance(value, dict) and "__ndarray__" in value:
        return _decode_array(value)
    return value


def _rebuild[T](build: Callable[..., T], record: Mapping[str, JsonValue]) -> T:
    """Return a record instance from the dictionary :func:`serialise` made of it.

    Parameters
    ----------
    build : collections.abc.Callable
        The record class to rebuild, called with the fields as keyword arguments.
    record : collections.abc.Mapping
        Its serialised form.

    Returns
    -------
    object
        The rebuilt record.
    """
    arguments = {
        name: _decode(item) for name, item in record.items() if name != "__class__"
    }
    return build(**arguments)


# ---------------------------------------------------------------------- provenance


def _run_git(arguments: Sequence[str], root: Path) -> str | None:
    """Return the output of one read-only git command, or ``None`` if it cannot run.

    Parameters
    ----------
    arguments : collections.abc.Sequence
        Arguments after ``git``. Passed as a list, never through a shell.
    root : pathlib.Path
        Directory to run in.

    Returns
    -------
    str or None
        The stripped standard output, or ``None`` when git is absent, the directory is
        not a repository, or the command fails or times out.
    """
    try:
        finished = subprocess.run(
            ["git", *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    if finished.returncode != 0:
        return None
    return finished.stdout.strip()


@dataclass(frozen=True)
class BackendRecord:
    """What executed a run, as the backend's own ``metadata()`` reported it.

    A saved run has to say what produced it and not only what was computed: the thread
    configuration alone moved one step by a factor of nine
    (``docs/dev/backends.md`` section 3.4), and without this record a timing
    comparison has to read the realised kernel pool out of the backend itself.

    It is **outside the resume key**, like the package versions and the git provenance:
    a run continued on the other backend agrees with an uninterrupted one to
    2.2-2.7e-16 and is the same run. What it records is how,
    not what.

    A resumed file keeps the record it was opened with, creation time and all, so the
    backend it names is the one that **started** the run. That is append mode's rule
    for the whole record and not a special case for this group.

    Attributes
    ----------
    name : str
        The registry name of the backend, ``"numpy_ref"`` or ``"numba_cpu"``.
    propagator_mode : str
        Which the backend resolved the contract's ``"auto"`` to, or which was asked
        for: ``"tabulate"`` or ``"recompute"``.
    settings : dict
        Everything else the backend reported: the threading regime, the four pool
        counts, the plan flag, the wisdom digest, the column family. Free-form because
        it is a backend's own vocabulary, and JSON-serialisable because it is written
        to the file.

    Examples
    --------
    >>> from solprop.output.schema import BackendRecord
    >>> record = BackendRecord.from_metadata(
    ...     {"backend": "numba_cpu", "propagator_mode": "recompute",
    ...      "numba_threads": 4}
    ... )
    >>> record.name, record.propagator_mode, record.settings
    ('numba_cpu', 'recompute', {'numba_threads': 4})
    """

    name: str
    propagator_mode: str
    settings: Mapping[str, JsonValue] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Adopt a plain copy of the free-form settings."""
        object.__setattr__(self, "settings", dict(self.settings))

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object]) -> "BackendRecord":  # noqa: UP037 - bound after the class body
        """Return the record a backend's ``metadata()`` describes.

        Parameters
        ----------
        metadata : collections.abc.Mapping
            What :meth:`solprop.backends.protocol.Backend.metadata` returned.
            ``backend`` and ``propagator_mode`` are named fields here; everything else
            becomes `settings`.

        Returns
        -------
        BackendRecord
            The record.

        Raises
        ------
        OutputError
            If either named field is missing.
        """
        try:
            name = str(metadata["backend"])
            mode = str(metadata["propagator_mode"])
        except KeyError as error:
            msg = (
                "a backend's metadata must name the backend and the propagator mode; "
                f"this one is missing {error}"
            )
            raise OutputError(msg) from error
        rest = {
            key: cast("JsonValue", value)
            for key, value in metadata.items()
            if key not in {"backend", "propagator_mode"}
        }
        return cls(name=name, propagator_mode=mode, settings=rest)


@dataclass(frozen=True)
class GitProvenance:
    """Which commit of solprop produced a file, and whether it was modified.

    Attributes
    ----------
    commit : str
        Full 40-character hexadecimal commit hash of the running checkout.
    branch : str
        Branch name, or ``'HEAD'`` when the checkout is detached.
    dirty : bool
        Whether the working tree had uncommitted changes to tracked files. A dirty
        run is reproducible only up to those changes, which is the point of recording
        it. It is a separate field rather than a suffix on the hash string, so the
        commit stays parsable.

    Examples
    --------
    >>> from solprop.output.schema import GitProvenance
    >>> record = GitProvenance(commit="0" * 40, branch="main", dirty=False)
    >>> len(record.commit), record.dirty
    (40, False)
    """

    commit: str
    branch: str
    dirty: bool

    @classmethod
    def detect(cls) -> "GitProvenance | None":  # noqa: UP037 - bound after the class body
        """Return the provenance of the running solprop checkout, if there is one.

        An installed release is not a checkout, git may be absent, and the package may
        sit inside an unrelated repository's worktree -- a virtual environment inside a
        user's own project, whose ``git rev-parse`` answers with *that* project's
        commit; every one of those returns ``None`` rather than a placeholder string,
        so that a caller can tell "no provenance" from "this commit". The test that
        separates the last case from a development checkout is whether the repository
        found actually tracks this file.

        Returns
        -------
        GitProvenance or None
            The commit, branch and dirty flag, or ``None`` when they cannot be read.

        Examples
        --------
        >>> from solprop.output.schema import GitProvenance
        >>> record = GitProvenance.detect()
        >>> record is None or len(record.commit) == 40
        True
        """
        here = Path(__file__).resolve()
        root = here.parent
        # A repository was found, but is it solprop's? Inside a virtual environment that
        # lives in some other project's checkout, git answers for that project. Only a
        # repository that tracks this very file is the one to record; that cannot
        # change while the process runs, so it is asked once.
        if not _tracks(here):
            return None
        # One command gives the commit, the branch and the dirty flag: the per-run
        # cost of provenance is a single git process (SEP 25 section 6.3).
        status = _run_git(
            ["status", "--porcelain=v2", "--branch", "--untracked-files=no"], root
        )
        if status is None:
            return None
        headers = dict(
            line[2:].split(" ", 1)
            for line in status.splitlines()
            if line.startswith("# ") and " " in line[2:]
        )
        commit = headers.get("branch.oid", "")
        if len(commit) != 40:
            return None
        branch = headers.get("branch.head", "")
        dirty = any(line and not line.startswith("#") for line in status.splitlines())
        return cls(
            commit=commit,
            branch="HEAD" if branch in {"", "(detached)"} else branch,
            dirty=dirty,
        )


@functools.cache
def _tracks(path: Path) -> bool:
    """Return whether the git repository around `path` tracks it, asked once.

    Parameters
    ----------
    path : pathlib.Path
        A file of the package.

    Returns
    -------
    bool
        ``True`` when git runs, finds a repository and that repository tracks the file.
    """
    return _run_git(["ls-files", "--error-unmatch", path.name], path.parent) is not None


def _version_of(name: str) -> str:
    """Return an installed distribution's version, or ``'unknown'``.

    Parameters
    ----------
    name : str
        Distribution name.

    Returns
    -------
    str
        The version string, or ``'unknown'`` when the distribution is not installed.
    """
    try:
        return _package_version(name)
    except PackageNotFoundError:  # pragma: no cover - every dependency is declared
        return "unknown"


def package_versions() -> dict[str, str]:
    """Return the versions of solprop and of the packages a stored file depends on.

    Recorded so that a file can be read years later with the same numerics in view.
    The versions are deliberately *outside* the configuration hash: upgrading NumPy
    must not refuse to resume a run. optmat's version is here too, for a reader; a run
    that takes a material from optmat carries it again in its material manifest
    (:attr:`RunMetadata.materials`), where it is part of the key.

    Returns
    -------
    dict
        Version strings keyed by ``'solprop'``, ``'numpy'``, ``'scipy'``, ``'h5py'``,
        ``'optmat'`` (its full version string) and ``'python'``.

    Examples
    --------
    >>> from solprop.output.schema import package_versions
    >>> sorted(package_versions())
    ['h5py', 'numpy', 'optmat', 'python', 'scipy', 'solprop']
    """
    return {
        "solprop": _version_of("solprop"),
        "numpy": _version_of("numpy"),
        "scipy": _version_of("scipy"),
        "h5py": _version_of("h5py"),
        "optmat": _version_of("optmat"),
        "python": platform.python_version(),
    }


# ------------------------------------------------------------------------- records


@dataclass(frozen=True, kw_only=True)
class GridRecord:
    """The scalars that identify the grid a run was computed on.

    Only scalars: the four axis arrays a :class:`solprop.grid.TimeGrid` holds are
    reconstructible from these, and keeping them out of the record keeps the metadata
    JSON, and the configuration hash taken over it, small.

    Attributes
    ----------
    kind : {'field', 'envelope'}
        Spectral layout, which fixes what the stored field means and which conversion
        the storage boundary applies.
    n_t : int
        Coarse time samples.
    n_to : int
        Fine time samples.
    n_omega : int
        Coarse spectral bins, the length of the stored field's last axis.
    n_omega_fine : int
        Fine spectral bins.
    dt : float
        Coarse sample spacing, s.
    dt_fine : float
        Fine sample spacing, s.
    ffac : float
        Oversampling factor, dimensionless.
    omega0 : float
        Reference angular frequency, rad/s.
    d_omega : float
        Spectral bin spacing, rad/s.
    length : float
        Propagation length the grid was built for, m.
    reference_wavelength : float
        Wavelength of ``omega0``, m.
    omega_min : float
        Lower in-band angular frequency, rad/s.
    omega_max : float
        Upper in-band angular frequency, rad/s.
    omega_max_window : float
        Upper edge of the active band, rad/s.
    t_window_min : float
        Lower inner edge of the temporal collar, s: the window is flat between this
        and ``t_window_max`` and tapers to the ends of the axis outside them.
    t_window_max : float
        Upper inner edge of the temporal collar, s.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output.schema import GridRecord
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> record = GridRecord.from_grid(grid)
    >>> record.kind, record.n_omega == grid.n_omega
    ('envelope', True)
    """

    kind: fourier.GridKind
    n_t: int
    n_to: int
    n_omega: int
    n_omega_fine: int
    dt: float
    dt_fine: float
    ffac: float
    omega0: float
    d_omega: float
    length: float
    reference_wavelength: float
    omega_min: float
    omega_max: float
    omega_max_window: float
    t_window_min: float
    t_window_max: float

    @classmethod
    def from_grid(cls, grid: TimeGrid) -> "GridRecord":  # noqa: UP037 - bound after the class body
        """Return the record of a grid.

        Parameters
        ----------
        grid : solprop.grid.TimeGrid
            The grid the run used.

        Returns
        -------
        GridRecord
            Its scalar description.

        Examples
        --------
        >>> from solprop.grid import FieldGrid
        >>> from solprop.output.schema import GridRecord
        >>> grid = FieldGrid.from_limits(1.0, 800e-9, (400e-9, 2e-6), 1e-12)
        >>> GridRecord.from_grid(grid).kind
        'field'
        """
        return cls(
            kind=grid.kind,
            n_t=grid.n_t,
            n_to=grid.n_to,
            n_omega=grid.n_omega,
            n_omega_fine=grid.n_omega_fine,
            dt=grid.dt,
            dt_fine=grid.dt_fine,
            ffac=grid.ffac,
            omega0=grid.omega0,
            d_omega=grid.d_omega,
            length=grid.length,
            reference_wavelength=grid.reference_wavelength,
            omega_min=grid.omega_min,
            omega_max=grid.omega_max,
            omega_max_window=grid.omega_max_window,
            t_window_min=grid.t_collar.one_low,
            t_window_max=grid.t_collar.one_high,
        )


@dataclass(frozen=True, kw_only=True)
class ModeRecord:
    """One propagated mode, by class, effective area, channel label and configuration.

    Attributes
    ----------
    kind : str
        Qualified class name of the mode, ``'module.ClassName'``.
    effective_area : float
        ``A_eff`` in m^2 at the reference position, the one number processing needs
        most often and the one a plot label carries.
    configuration : dict
        The mode object as :func:`serialise` records it.
    label : collections.abc.Mapping or None
        The physical channel this row of the mode axis is, as
        :meth:`solprop.modes.protocol.ChannelLabel.as_mapping` writes it: the
        family, the azimuthal and radial orders, the parity, the handedness and
        the display label. **Output schema 7**; ``None`` on a
        record read back from an older file, which named its channels by class
        name and position and cannot be given quantum numbers after the fact.
        Outside :meth:`RunMetadata.hashed` below schema 7, so that every file
        written before this field existed keeps the digest it was stamped with.

    Examples
    --------
    >>> from solprop.modes import MarcatiliMode
    >>> from solprop.output.schema import ModeRecord
    >>> record = ModeRecord.from_mode(MarcatiliMode(125e-6, "He", 1.0))
    >>> record.kind
    'solprop.modes.capillary.MarcatiliMode'
    >>> None if record.label is None else record.label["display_label"]
    'HE1,1:even'
    """

    kind: str
    effective_area: float
    configuration: Mapping[str, JsonValue]
    label: Mapping[str, JsonValue] | None = None

    @classmethod
    def from_mode(cls, mode: Mode, z: float = 0.0) -> "ModeRecord":  # noqa: UP037 - bound after the class body
        """Return the record of a mode.

        Parameters
        ----------
        mode : solprop.modes.protocol.Mode
            The mode.
        z : float, optional
            Position along the waveguide the effective area is read at, m.

        Returns
        -------
        ModeRecord
            Its description.

        Examples
        --------
        >>> from solprop.modes import MarcatiliMode
        >>> from solprop.output.schema import ModeRecord
        >>> record = ModeRecord.from_mode(MarcatiliMode(75e-6, "He", 1.0))
        >>> f"{record.effective_area:.6e}"
        '8.421575e-09'
        """
        kind = type(mode)
        # serialise returns a dictionary for everything that is not a primitive, a
        # sequence or a callable, and a mode is none of those, so the cast is total.
        configuration = cast("dict[str, JsonValue]", serialise(mode))
        if getattr(mode, "mixture", None) is not None:
            # These legacy constructor slots are placeholders when a mixture owns
            # every species model and density path; they would imply a second fill.
            for name in ("gas", "pressure", "temperature", "source", "susceptibility"):
                configuration.pop(name, None)
        return cls(
            kind=f"{kind.__module__}.{kind.__qualname__}",
            effective_area=mode.aeff(z),
            configuration=configuration,
            label=dict(mode.channel_label.as_mapping()),
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class NoiseRecord:
    """The shot-noise realisation a run started from, and the seed that drew it.

    Equality is by identity (``eq=False``) because ``spectrum`` is an array; compare
    the seeds, which is what reproducibility means.

    The spectrum is stored in the **internal** FFT convention, unlike the field: it is
    not a user-facing spectrum but the object the stepper adds at every stage, and
    conjugating it would make it disagree with the file it has to be read back into.
    :attr:`spectral_convention` says so explicitly.

    Attributes
    ----------
    model : {'input', 'modified'}
        Which noise model the realisation belongs to.
    seed : int or None
        Entropy of the generator, or ``None`` when it could not be recovered.
    spectrum : numpy.ndarray
        The realisation, shape ``(n_modes, n_pol, n_omega)``, complex, in the square
        root of watts.
    spectral_convention : str
        Always ``'fft'``.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.noise import input_noise
    >>> from solprop.output.schema import NoiseRecord
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    >>> realisation = input_noise(grid, np.random.default_rng(3))
    >>> record = NoiseRecord.from_realisation(realisation)
    >>> record.seed, record.spectral_convention
    (3, 'fft')
    """

    model: str
    seed: int | None
    spectrum: Complex[np.ndarray, "n_modes n_pol n_omega"]
    spectral_convention: str = INTERNAL_CONVENTION

    def __post_init__(self) -> None:
        """Adopt a read-only copy of the spectrum, so a record cannot drift."""
        frozen = np.array(self.spectrum, dtype=np.complex128, copy=True)
        frozen.setflags(write=False)
        object.__setattr__(self, "spectrum", frozen)

    @classmethod
    def from_realisation(cls, realisation: NoiseRealisation) -> "NoiseRecord":  # noqa: UP037 - bound after the class body
        """Return the record of a drawn realisation.

        Parameters
        ----------
        realisation : solprop.noise.NoiseRealisation
            The realisation the run used.

        Returns
        -------
        NoiseRecord
            Its description.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.grid import EnvelopeGrid
        >>> from solprop.noise import modified_noise
        >>> from solprop.output.schema import NoiseRecord
        >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
        >>> NoiseRecord.from_realisation(modified_noise(grid, np.random.default_rng(1)))
        ... # doctest: +ELLIPSIS
        NoiseRecord(model='modified', seed=1, ...)
        """
        return cls(
            model=realisation.model,
            seed=realisation.seed,
            spectrum=realisation.spectrum,
        )


@dataclass(frozen=True)
class NoiseResolution:
    """How a run's noise model was chosen: what was asked for, what it carries, and why.

    Attributes
    ----------
    requested : {'default', 'input', 'modified'} or None
        The model the caller asked for: ``"default"`` when the keyword was omitted,
        ``None`` when noise was switched off explicitly.
    resolved : {'input', 'modified'} or None
        The model the run carries, or ``None`` for a noiseless run.
    reason : str or None
        Why a default resolved to no noise, for example that single precision carries
        none; ``None`` otherwise.

    Examples
    --------
    >>> from solprop.output.schema import NoiseResolution
    >>> NoiseResolution("default", "input")
    NoiseResolution(requested='default', resolved='input', reason=None)
    """

    requested: str | None
    resolved: str | None
    reason: str | None = None


@dataclass(frozen=True)
class StepLimits:
    """The caps a run put on the controller's step proposal.

    Grouped into one record because they travel together and because a constructor
    that took them one by one would need more arguments than the project allows.

    Attributes
    ----------
    dz_min : float
        Smallest step the controller was allowed to propose, m. Zero when the driver
        left it to :func:`solprop.stepper.controller.default_dz_min`.
    dz_max : float
        Largest step, m; infinite when uncapped.
    dz_init : float
        First step, m; zero when the driver chose it.
    stretch : float
        Landing stretch factor, dimensionless; 1 switches landing stretch off.
    interval_cap : bool
        Whether a step was capped at its operator-table interval's width (output
        schema 10; every earlier run was).
    auto_init : bool
        Whether the start-up rule took the step after the first from the first step's
        estimate (output schema 10; no earlier run did).
    interpolate_planes : bool
        Whether the save planes but the end and the hard kinks were interpolated
        inside accepted steps (output schema 11); ``False`` when every plane was
        landed on, as in every earlier run.
    step_planes : int
        A positive ``k`` when the end of every ``k``-th accepted step was saved too
        (output schema 11); ``0`` for the requested planes only.

    Examples
    --------
    >>> from solprop.output.schema import StepLimits
    >>> StepLimits(dz_max=0.01).dz_max
    0.01
    """

    dz_min: float = 0.0
    dz_max: float = math.inf
    dz_init: float = 0.0
    stretch: float = 1.1
    interval_cap: bool = True
    auto_init: bool = False
    interpolate_planes: bool = False
    step_planes: int = 0


@dataclass(frozen=True, eq=False, kw_only=True)
class StepperRecord:
    """The solver settings a run used, including the schedule it actually took.

    Equality is by identity (``eq=False``) because ``schedule`` is an array. Schedules
    are first class and re-loadable (see ``docs/dev/decisions.md``): a replayed run is
    bit-identical, and every ensemble member then carries the same numerical error as a
    common systematic.

    Build one from the objects a run held with :meth:`from_settings`, and from a stored
    document with :meth:`from_record`.

    Attributes
    ----------
    tableau : str
        Name of the Runge-Kutta scheme.
    method : str or None
        The integrator that stepped the run instead of the tableau, as its ``repr``:
        a low-call Adams method (SEP 13, recorded from output schema 9) or the
        low-storage march (SEP 24, from schema 19); ``None`` for the Runge-Kutta march.
    controller : dict
        The step controller as :func:`serialise` records it: tolerances, safety and
        gain factors, lag and the norm weights.
    limits : StepLimits
        The caps on the controller's proposal.
    schedule : numpy.ndarray or None
        Step lengths of shape ``(n_steps,)``, m, when a fixed schedule was replayed or
        recorded; ``None`` for a purely adaptive run.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.schema import StepLimits, StepperRecord
    >>> from solprop.stepper import DOPRI5, FixedSchedule, PIController
    >>> record = StepperRecord.from_settings(
    ...     DOPRI5,
    ...     PIController(rtol=1e-8, atol=0.0),
    ...     schedule=FixedSchedule(np.array([0.5, 0.5])),
    ...     limits=StepLimits(dz_max=0.1),
    ... )
    >>> record.tableau, record.limits.dz_max
    ('DOPRI5', 0.1)
    """

    tableau: str
    controller: Mapping[str, JsonValue]
    limits: StepLimits = dataclasses.field(default_factory=StepLimits)
    schedule: Float[np.ndarray, " n_steps"] | None = None
    method: str | None = None

    def __post_init__(self) -> None:
        """Adopt a read-only copy of the schedule, so a record cannot drift."""
        if self.schedule is not None:
            frozen = np.array(self.schedule, dtype=np.float64, copy=True)
            frozen.setflags(write=False)
            object.__setattr__(self, "schedule", frozen)

    @classmethod
    def from_settings(
        cls,
        tableau: Tableau,
        controller: PIController,
        *,
        schedule: FixedSchedule | None = None,
        limits: StepLimits | None = None,
        method: str | None = None,
    ) -> "StepperRecord":  # noqa: UP037 - bound after the class body
        """Return the record of the objects a run stepped with.

        Parameters
        ----------
        tableau : solprop.stepper.tableau.Tableau
            The scheme.
        controller : solprop.stepper.controller.PIController
            The step controller, which carries the tolerances and the norm choice.
        schedule : solprop.stepper.schedule.FixedSchedule, optional
            The fixed schedule replayed, if any.
        limits : StepLimits, optional
            The caps on the controller's proposal; the defaults when omitted.
        method : str or None, optional
            The Adams method that stepped the run instead of the tableau, as its
            ``repr``; ``None`` for the Runge-Kutta march.

        Returns
        -------
        StepperRecord
            The record.

        Examples
        --------
        >>> from solprop.output.schema import StepperRecord
        >>> from solprop.stepper import TSIT5, PIController
        >>> record = StepperRecord.from_settings(
        ...     TSIT5, PIController(rtol=1e-6, atol=0.0)
        ... )
        >>> record.tableau, record.schedule is None
        ('Tsit5', True)
        """
        # A PIController is a dataclass, so serialise always returns a dictionary.
        record = cast("dict[str, JsonValue]", serialise(controller))
        return cls(
            tableau=tableau.name,
            controller=record,
            limits=StepLimits() if limits is None else limits,
            schedule=None if schedule is None else schedule.steps,
            method=method,
        )

    @classmethod
    def from_record(cls, document: Mapping[str, JsonValue]) -> "StepperRecord":  # noqa: UP037 - bound after the class body
        """Return the record a stored document describes.

        Parameters
        ----------
        document : collections.abc.Mapping
            The form :func:`serialise` produced from a record.

        Returns
        -------
        StepperRecord
            The record.

        Examples
        --------
        >>> from solprop.output.schema import StepperRecord, serialise
        >>> from solprop.stepper import TSIT5, PIController
        >>> one = StepperRecord.from_settings(TSIT5, PIController(rtol=1e-6, atol=0.0))
        >>> StepperRecord.from_record(serialise(one)).tableau
        'Tsit5'
        """
        schedule = _decode(document["schedule"])
        return cls(
            tableau=str(document["tableau"]),
            controller=cast("dict[str, JsonValue]", document["controller"]),
            limits=_rebuild(
                StepLimits, cast("dict[str, JsonValue]", document["limits"])
            ),
            schedule=None if schedule is None else np.asarray(schedule, dtype=float),
            method=_optional_text(document.get("method")),
        )


HISTORY_FIELDS: Mapping[str, type] = {
    "z": np.float64,
    "h": np.float64,
    "norm": np.float64,
    "accepted": np.bool_,
    "wall_time": np.float64,
}
"""Dtype of each :class:`StepHistory` field, in the order they are laid out."""


@dataclass(frozen=True, eq=False, kw_only=True)
class StepHistory(Displayed):
    """What the stepper did between two saved planes.

    Recorded whether the run was adaptive or replayed a schedule, because the embedded
    error estimate costs nothing extra and a fixed run still needs it: a largest scaled
    norm above one means a step was too coarse for the tolerance the norm is scaled by
    (``docs/dev/stepper.md`` §8).

    Equality is by identity (``eq=False``) because every field is an array.

    Attributes
    ----------
    z : numpy.ndarray
        Position at the start of each attempt, shape ``(n_steps,)``, m.
    h : numpy.ndarray
        Length of each attempt, shape ``(n_steps,)``, m.
    norm : numpy.ndarray
        Scaled error norm each attempt scored as executed, shape ``(n_steps,)``,
        dimensionless; the controller's rescaled equivalent of a landing step is its
        own and is not recorded. The low-storage march estimates none and records 0.0,
        "not estimated" (output schema 19).
    accepted : numpy.ndarray
        Whether each attempt was accepted, shape ``(n_steps,)``, boolean.
    wall_time : numpy.ndarray
        Wall-clock time each attempt took, shape ``(n_steps,)``, s.

    Raises
    ------
    OutputError
        If the five arrays do not have equal shapes.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.schema import StepHistory
    >>> history = StepHistory(
    ...     z=np.array([0.0, 0.5]),
    ...     h=np.array([0.5, 0.5]),
    ...     norm=np.array([0.3, 0.4]),
    ...     accepted=np.array([True, True]),
    ...     wall_time=np.array([1e-3, 1e-3]),
    ... )
    >>> history.n_steps, history.n_rejected
    (2, 0)
    """

    # Independent axis names on purpose: their agreement is checked below with an
    # OutputError naming the culprit, and that check must also hold in production,
    # where the test session's runtime type checker is absent. One shared name would
    # let the checker forestall it.
    z: Float[np.ndarray, " n_history_z"]
    h: Float[np.ndarray, " n_history_h"]
    norm: Float[np.ndarray, " n_history_norm"]
    accepted: Bool[np.ndarray, " n_history_accepted"]
    wall_time: Float[np.ndarray, " n_history_wall"]

    def __repr__(self) -> str:
        """Return a one-line description: attempts, rejections and wall time.

        Returns
        -------
        str
            For example ``StepHistory(1842 attempts, 12 rejected, 12.2 s)``.
        """
        attempts = int(np.asarray(self.z).size)
        rejected = attempts - int(np.count_nonzero(self.accepted))
        seconds = float(np.sum(self.wall_time)) if attempts else 0.0
        return line(
            "StepHistory",
            [
                ("", f"{attempts} attempts"),
                ("", f"{rejected} rejected"),
                ("", engineering(seconds, "s")),
            ],
        )

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        attempts = int(np.asarray(self.z).size)
        if not attempts:
            return "StepHistory: no step attempted", []
        accepted = np.asarray(self.accepted, dtype=bool)
        steps = np.asarray(self.h, dtype=float)[accepted]
        return "StepHistory", [
            (
                "attempts",
                f"{attempts}, of which {attempts - int(accepted.sum())} rejected",
            ),
            ("accepted steps", summarise(steps, "m").replace("values", "steps")),
            ("largest error norm", f"{float(np.max(self.norm)):.3g}"),
            ("wall time", engineering(float(np.sum(self.wall_time)), "s")),
        ]

    def __post_init__(self) -> None:
        """Check that the five arrays describe one sequence, and freeze them.

        Raises
        ------
        OutputError
            If the five shapes disagree. That they are one-dimensional is fixed by the
            annotations, as it is for :class:`solprop.stepper.tableau.Tableau`.
        """
        shapes = {}
        for name, dtype in HISTORY_FIELDS.items():
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
            shapes[name] = values.shape
        if len(set(shapes.values())) > 1:
            msg = f"step history arrays must have equal shapes, got {shapes}"
            raise OutputError(msg)

    @classmethod
    def empty(cls) -> "StepHistory":  # noqa: UP037 - bound after the class body
        """Return the history of a plane no step led to, such as the starting plane.

        Returns
        -------
        StepHistory
            Five empty arrays.

        Examples
        --------
        >>> from solprop.output.schema import StepHistory
        >>> StepHistory.empty().n_steps
        0
        """
        return cls(
            z=np.empty(0),
            h=np.empty(0),
            norm=np.empty(0),
            accepted=np.empty(0, dtype=bool),
            wall_time=np.empty(0),
        )

    @property
    def n_steps(self) -> int:
        """Return the number of attempts recorded.

        Returns
        -------
        int
            Accepted and rejected attempts together.
        """
        return int(self.z.size)

    @property
    def n_rejected(self) -> int:
        """Return the number of rejected attempts.

        Returns
        -------
        int
            How many attempts the controller threw away.
        """
        return int(np.count_nonzero(~self.accepted))


STEP_STATS_INDEX: Mapping[str, type] = {"z": np.float64, "step": np.int64}
"""Dtype of each :class:`StepStats` index column, beside the statistics themselves."""


@dataclass(frozen=True, eq=False, kw_only=True)
class StepStats(Displayed):
    """The statistics a backend evaluated *between* two saved planes.

    A run whose :class:`solprop.stats.StatsSpec` names a ``periodic`` subset gets one
    row every ``period`` accepted steps, evaluated inside the backend's step loop and
    flushed here at each plane. Empty for a run that asked for none, which is
    the default and what every file written before schema version 3 holds.

    Stored the way the step history is -- flat columns concatenated over planes with a
    per-plane ``counts`` beside them -- so that a resume appends and a scan needs no NaN
    padding, which growing datasets would force on it.

    Equality is by identity (``eq=False``) because every field is an array.

    Attributes
    ----------
    z : numpy.ndarray
        Position of each row, m, float64, shape ``(n_rows,)``.
    step : numpy.ndarray
        Index of the accepted step each row was taken after, into this plane's
        :class:`StepHistory`, int64, shape ``(n_rows,)``. The history counts rejected
        attempts, so this is not ``row * period``.
    values : dict
        The statistics, keyed by catalogue name, each shaped ``(n_rows, *shape)`` with
        the statistic's own shape after the row axis -- the batch axis a backend
        carries is dropped here, as it is for the plane statistics.

    Raises
    ------
    OutputError
        If the columns do not all have the same number of rows.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.schema import StepStats
    >>> rows = StepStats(z=np.array([0.1, 0.2]), step=np.array([0, 2]),
    ...                  values={"energy": np.ones((2, 1, 1))})
    >>> rows.n_rows, rows.names
    (2, ('energy',))
    """

    z: Float[np.ndarray, " n_rows_z"]
    step: Int[np.ndarray, " n_rows_step"]
    values: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a one-line description: how many rows of which statistics.

        Returns
        -------
        str
            For example ``StepStats(1842 rows of energy, peak_power)``.
        """
        rows = int(np.asarray(self.z).size)
        return f"StepStats({rows} rows of {', '.join(self.values) or 'nothing'})"

    def __post_init__(self) -> None:
        """Freeze the columns and check that they describe the same rows.

        Raises
        ------
        OutputError
            Naming the columns whose row counts disagree.
        """
        shapes = {}
        for name, dtype in STEP_STATS_INDEX.items():
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
            shapes[name] = values.shape
        adopted = {}
        for name, value in self.values.items():
            rows = np.array(value, copy=True)
            rows.setflags(write=False)
            adopted[name] = rows
            shapes[name] = rows.shape[:1]
        object.__setattr__(self, "values", adopted)
        if len(set(shapes.values())) > 1:
            msg = f"step statistics columns must have equal row counts, got {shapes}"
            raise OutputError(msg)

    @classmethod
    def empty(cls) -> "StepStats":  # noqa: UP037 - bound after the class body
        """Return the channel a run without a periodic subset carries.

        Returns
        -------
        StepStats
            No rows and no statistics.
        """
        return cls(z=np.empty(0), step=np.empty(0, dtype=np.int64), values={})

    @property
    def n_rows(self) -> int:
        """Return how many rows belong to this plane.

        Returns
        -------
        int
            The row count.
        """
        return int(self.z.size)

    @property
    def names(self) -> tuple[str, ...]:
        """Return the statistics the channel carries, sorted.

        Returns
        -------
        tuple of str
            The catalogue names.
        """
        return tuple(sorted(self.values))


@dataclass(frozen=True, eq=False, kw_only=True)
class SavedPlane(Displayed):
    """One saved position: the field there, its statistics and how it was reached.

    The field is in the **internal** FFT convention, exactly as a backend produces it;
    a handler converts it on the way to storage. Equality is by identity
    (``eq=False``) because the field is an array. A **field-free** plane, of a run
    that keeps no field there (:class:`FieldRetention`), carries ``field=None`` and no
    noise phase: its position, statistics and step history only.

    Attributes
    ----------
    z : float
        Position along the waveguide, m.
    field : numpy.ndarray or None
        Stored spectrum, shape ``(n_modes, n_pol, n_omega)``, complex, in the square
        root of watts, internal convention; ``None`` on a field-free plane.
    stats : dict
        Statistics evaluated at this plane, keyed by catalogue name; the values are the
        arrays :func:`solprop.stats.collect` returns.
    step_history : StepHistory
        Every attempt the stepper made since the previous plane.
    step_stats : StepStats
        The rows a periodic statistics subset produced since the previous plane, empty
        unless the run asked for one.
    noise_phase : numpy.ndarray or None
        Accumulated unitary phase of the noise field at this plane, shape
        ``(n_modes, n_pol, n_omega)``, rad, internal convention, so that processing can
        restore the background; ``None`` when no noise model is active, and on a
        field-free plane.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.schema import SavedPlane, StepHistory
    >>> plane = SavedPlane(
    ...     z=0.0,
    ...     field=np.ones((1, 1, 5), dtype=complex),
    ...     stats={"energy": np.ones((1, 1))},
    ...     step_history=StepHistory.empty(),
    ... )
    >>> plane.shape
    (1, 1, 5)
    """

    z: float
    field: Complex[np.ndarray, "n_modes n_pol n_omega"] | None
    stats: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict)
    step_history: StepHistory = dataclasses.field(default_factory=StepHistory.empty)
    step_stats: StepStats = dataclasses.field(default_factory=StepStats.empty)
    noise_phase: Float[np.ndarray, "n_modes_phase n_pol_phase n_omega_phase"] | None = (
        None
    )

    def __repr__(self) -> str:
        """Return a one-line description: the position and the field's shape.

        Returns
        -------
        str
            For example ``SavedPlane(z=1.5 m, <2x1x8192 complex>, 14 statistics)``.
        """
        return line(
            "SavedPlane",
            [
                ("z", engineering(self.z, "m")),
                ("", self._field_text()),
                ("", f"{len(self.stats)} statistics"),
            ],
        )

    def _field_text(self) -> str:
        """Return the field summarised, or a note that the plane keeps none.

        Returns
        -------
        str
            The field's shape, or ``"no field"``.
        """
        return "no field" if self.field is None else summarise(self.field)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        return f"SavedPlane at z = {engineering(self.z, 'm')}", [
            ("field", self._field_text()),
            ("statistics", ", ".join(self.stats) or "none"),
            ("steps since last", repr(self.step_history)),
            ("between planes", repr(self.step_stats)),
        ]

    def __post_init__(self) -> None:
        """Freeze the arrays and check that a noise phase matches the field.

        Raises
        ------
        OutputError
            If ``z`` is not finite, ``noise_phase`` does not have the field's shape, or
            a field-free plane carries a noise phase.
        """
        if not math.isfinite(self.z):
            msg = f"a saved plane needs a finite position, got {self.z!r}"
            raise OutputError(msg)
        object.__setattr__(self, "stats", dict(self.stats))
        if self.field is None:
            if self.noise_phase is not None:
                msg = "a field-free plane carries no noise phase"
                raise OutputError(msg)
            return
        field = np.array(self.field, dtype=np.complex128, copy=True)
        field.setflags(write=False)
        object.__setattr__(self, "field", field)
        if self.noise_phase is not None:
            phase = np.array(self.noise_phase, dtype=np.float64, copy=True)
            if phase.shape != field.shape:
                msg = (
                    f"noise_phase has shape {phase.shape}, expected the field's "
                    f"{field.shape}"
                )
                raise OutputError(msg)
            phase.setflags(write=False)
            object.__setattr__(self, "noise_phase", phase)

    @property
    def shape(self) -> tuple[int, ...] | None:
        """Return the field's shape, or ``None`` on a field-free plane.

        Returns
        -------
        tuple or None
            ``(n_modes, n_pol, n_omega)``.
        """
        return None if self.field is None else self.field.shape


@dataclass(frozen=True)
class RunOutcome:
    """Record completion and cumulative attempted work separately from run identity.

    Attributes
    ----------
    status : {'running', 'complete', 'budget_exhausted', 'failed'}
        Current execution status. A budget stop is explicitly unfinished.
    z : float
        Last accepted or saved position, m.
    message : str
        Explanation of a stop or failure; empty on ordinary progress.
    logical_calls : collections.abc.Mapping
        Cumulative logical response ledger, including startup and failed attempts.
    atomic_calls : collections.abc.Mapping
        Cumulative provider ledger; empty when no atomic service is bound.
    execution : collections.abc.Mapping
        Current backend and provider facts, captured before resources close. Kept
        separately from the original run metadata so resumes retain their actual
        device, plan and diagnostic provenance without changing scientific identity.
    previous_executions : tuple of collections.abc.Mapping
        Closing records of earlier executions of the same resumed run, in order.
    """

    status: Literal["running", "complete", "budget_exhausted", "failed"]
    z: float
    message: str = ""
    logical_calls: Mapping[str, object] = dataclasses.field(default_factory=dict)
    atomic_calls: Mapping[str, object] = dataclasses.field(default_factory=dict)
    execution: Mapping[str, object] = dataclasses.field(default_factory=dict)
    previous_executions: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous status and nonfinite positions, and adopt ledger copies."""
        if self.status not in ("running", "complete", "budget_exhausted", "failed"):
            raise OutputError(f"unknown run outcome {self.status!r}")
        if not np.isfinite(self.z) or self.z < 0:
            raise OutputError("run outcome z must be finite and nonnegative")
        for name in ("logical_calls", "atomic_calls", "execution"):
            object.__setattr__(self, name, _outcome_record(getattr(self, name)))
        object.__setattr__(
            self,
            "previous_executions",
            tuple(_outcome_record(record) for record in self.previous_executions),
        )


def _outcome_record(record: Mapping[str, object]) -> dict[str, object]:
    """Own finite JSON values so later provider mutations cannot change a snapshot.

    Parameters
    ----------
    record : Mapping[str, object]
        Execution facts composed of finite JSON values.

    Returns
    -------
    dict
        Independently owned nested values, detached from provider state.
    """
    try:
        return cast(
            "dict[str, object]", json.loads(json.dumps(dict(record), allow_nan=False))
        )
    except (TypeError, ValueError) as error:
        raise OutputError(
            f"execution records must contain finite JSON values: {error}"
        ) from error


@dataclass(frozen=True, eq=False)
class TerminalCheckpoint:
    """Keep an accepted field between scheduled saves without adding an observation.

    Attributes
    ----------
    plane : SavedPlane
        Accepted field in the internal FFT convention and the output amplitude
        convention, with already available history and periodic statistics.
        Creating a checkpoint never requests fresh material statistics. It always
        carries the field, also in a run that keeps none at its planes, so that a
        stopped run resumes (SEP 22, Q1).
    next_save_index : int
        Index of the first scheduled save strictly beyond this field.
    landed : bool
        Whether the field is an accepted endpoint rather than a dense observation.
    """

    plane: SavedPlane
    next_save_index: int
    landed: bool = True

    def __post_init__(self) -> None:
        """Reject an unusable save index, a field-free or a nonfinite checkpoint."""
        if type(self.next_save_index) is not int or self.next_save_index < 0:
            raise OutputError(
                "checkpoint next_save_index must be a nonnegative integer"
            )
        if self.plane.field is None:
            raise OutputError(
                "a terminal checkpoint carries the field, also in a run that keeps "
                "none at its planes"
            )
        if (
            not np.isfinite(self.plane.z)
            or self.plane.z < 0
            or not np.all(np.isfinite(self.plane.field))
        ):
            raise OutputError("checkpoint position and field must be finite")


@dataclass(frozen=True, eq=False)
class MarchSnapshot:  # numpydoc ignore=PR02
    """An Adams march's state at an accepted position, for a resume to continue.

    What a resume needs to carry on as the uninterrupted run would: the accepted state
    exactly as the backend held it, the history's samples in their engine rows, which
    sample each history entry is, the order, the rejection count and the step the
    controller proposed next. The configuration hash already fixes the method, so a
    snapshot is never the record of what was computed, only of where a march stood;
    a resume that cannot use it starts the history afresh.

    Equality is by identity (``eq=False``) because the fields hold arrays.

    Attributes
    ----------
    z : float
        The accepted position, m.
    state : numpy.ndarray
        The accepted state in the canonical layout and the internal FFT convention,
        exactly as the backend held it.
    proposal : float
        The step the controller proposed next, m.
    method : str
        The method's representation, which the resuming run's must equal.
    order : int
        The order chosen for the next step.
    rejections : int
        Consecutive rejections so far.
    last_rejected : bool
        Whether the last trial was rejected.
    positions : numpy.ndarray
        Each history entry's position, m, oldest first.
    identities : tuple of str
        Each entry's kind (``"accepted"`` or ``"predicted"``).
    slots : numpy.ndarray
        Each entry's engine row.
    rows : collections.abc.Mapping
        The engine's rows by name: the samples (``"store"``) and, for a compensated
        exponential march, the sampled states and rates and each sample's flow
        weights.

    Raises
    ------
    OutputError
        If the entries disagree in number, or a position or the state is not finite.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.schema import MarchSnapshot
    >>> snapshot = MarchSnapshot(
    ...     z=0.5, state=np.ones((1, 1, 1, 4), complex), proposal=0.1, method="M()",
    ...     order=2, rejections=0, last_rejected=False,
    ...     positions=np.array([0.4, 0.5]), identities=("accepted", "predicted"),
    ...     slots=np.array([0, 1]), rows={"store": np.zeros((4, 1, 1, 1, 4), complex)},
    ... )
    >>> snapshot.nbytes
    320
    """

    z: float
    state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    proposal: float
    method: str
    order: int
    rejections: int
    last_rejected: bool
    positions: Float[np.ndarray, " n_entries"]
    identities: tuple[str, ...]
    slots: Int[np.ndarray, " n_entries"]
    rows: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        """Refuse a snapshot whose entries disagree or whose numbers are not finite.

        Raises
        ------
        OutputError
            Naming the inconsistency.
        """
        count = len(self.identities)
        if self.positions.shape != (count,) or self.slots.shape != (count,):
            raise OutputError("march snapshot entries disagree in number")
        if not (
            math.isfinite(self.z)
            and math.isfinite(self.proposal)
            and np.all(np.isfinite(self.positions))
            and np.all(np.isfinite(self.state))
        ):
            raise OutputError("march snapshot positions and state must be finite")

    @property
    def nbytes(self) -> int:
        """Return the bytes the snapshot's arrays hold.

        Returns
        -------
        int
            The state and every engine row.
        """
        return int(self.state.nbytes + sum(row.nbytes for row in self.rows.values()))


def _prepend_checkpoint_history(
    previous: SavedPlane, current: SavedPlane
) -> SavedPlane:
    """Retain attempts and periodic rows preceding a fresh-start resume.

    Parameters
    ----------
    previous : SavedPlane
        Pending checkpoint carrying attempts before the restart.
    current : SavedPlane
        New accepted field and attempts after the restart.

    Returns
    -------
    SavedPlane
        Current field with both histories and adjusted periodic-row indices.
    """
    history = StepHistory(
        **{
            name: np.concatenate(
                (
                    getattr(previous.step_history, name),
                    getattr(current.step_history, name),
                )
            )
            for name in HISTORY_FIELDS
        }
    )
    before, after = previous.step_stats, current.step_stats
    if before.n_rows and after.n_rows and before.names != after.names:
        raise OutputError("checkpoint periodic statistic names changed on resume")
    names = before.names if before.n_rows else after.names
    values = {}
    for name in names:
        if before.n_rows and after.n_rows:
            values[name] = np.concatenate((before.values[name], after.values[name]))
        else:
            values[name] = before.values[name] if before.n_rows else after.values[name]
    rows = StepStats(
        z=np.concatenate((before.z, after.z)),
        step=np.concatenate((before.step, after.step + previous.step_history.n_steps)),
        values=values,
    )
    return dataclasses.replace(current, step_history=history, step_stats=rows)


def _differences(
    left: JsonValue, right: JsonValue, path: str, found: list[str]
) -> None:
    """Collect the dotted paths at which two serialised structures differ.

    Parameters
    ----------
    left : NoneType or bool or int or float or str or list or dict
        The first structure.
    right : NoneType or bool or int or float or str or list or dict
        The second.
    path : str
        Dotted path of the position being compared.
    found : list
        Accumulator the paths are appended to, in depth-first order.
    """
    if isinstance(left, dict) and isinstance(right, dict):
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
    if _canonical_json(left) != _canonical_json(right):
        found.append(path)


def _optional_text(value: JsonValue) -> str | None:
    """Return a stored optional string, ``None`` when absent or null.

    Parameters
    ----------
    value : object
        The stored value.

    Returns
    -------
    str or None
        The string, or ``None``.
    """
    return None if value is None else str(value)


def _hashed_stepper(
    stepper: StepperRecord,
    schema_version: int,
    stored_limits: tuple[str, ...] | None = None,
) -> JsonValue:
    """Return the stepper record as the resume key takes it at a schema version.

    The method enters the key at schema 9 (SEP 13), the interval cap and start-up
    rule at schema 10 (SEP 14), and the controller's norm region at schema 14, and not
    before, so that a digest stamped at an earlier
    schema still verifies: those records have no such fields. Below schema 15 the
    method string leaves the interaction formulation out, as it was written then.

    Parameters
    ----------
    stepper : StepperRecord
        The record.
    schema_version : int
        The record's output schema.
    stored_limits : tuple of str or None
        Fields present in a loaded record, distinguishing the two schema-10 lineages.

    Returns
    -------
    object
        The serialised record, without ``method`` below schema 9 and without the
        interval cap and start-up rule below schema 10.
    """
    record = cast("dict[str, JsonValue]", serialise(stepper))
    if schema_version < 9:
        record.pop("method", None)
    method = record.get("method")
    if schema_version < 15 and isinstance(method, str):
        # Before schema 15 the interaction formulation, then the default, was left out
        # of the method's representation.
        record["method"] = method.replace(", formulation='interaction'", "", 1)
    if schema_version < 14:
        # The norm's region enters the key at schema 14; before it the norm scored the
        # requested band and recorded no region.
        controller = dict(cast("dict[str, JsonValue]", record["controller"]))
        controller.pop("region", None)
        record["controller"] = controller
    if stored_limits is not None:
        record["limits"] = {
            key: value
            for key, value in cast("dict[str, JsonValue]", record["limits"]).items()
            if key in stored_limits
        }
    elif schema_version < 11:
        limits = dict(cast("dict[str, JsonValue]", record["limits"]))
        limits.pop("interpolate_planes", None)
        limits.pop("step_planes", None)
        if schema_version < 10:
            limits.pop("interval_cap", None)
            limits.pop("auto_init", None)
        record["limits"] = limits
    return record


def _hashed_modes(modes: Sequence[ModeRecord], schema_version: int) -> list[JsonValue]:
    """Return the mode records as the resume key of one schema version sees them.

    :attr:`ModeRecord.label` is output schema 7's. A record read back from an older
    file carries none, and a record built here from a live mode carries one whatever
    the version says, so the key of a schema-6 or older record is taken with the
    field removed: the digests stamped into every file written before this field
    existed still verify, which is what keeps those files readable.

    Parameters
    ----------
    modes : collections.abc.Sequence
        The mode records, in state order.
    schema_version : int
        The version of the record the key is being taken for.

    Returns
    -------
    list
        The serialised records, without ``label`` below schema 7.
    """
    records = cast("list[dict[str, JsonValue]]", serialise(list(modes)))
    if schema_version < 7:
        for record in records:
            record.pop("label", None)
    return cast("list[JsonValue]", records)


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

    Parameters
    ----------
    modal : collections.abc.Mapping or None
        The identity, as :func:`solprop.translate.general.modal_identity` writes
        it, or ``None`` for a run that is not a general modal one.

    Returns
    -------
    object
        The serialised identity without ``accessors``, or ``None``.
    """
    if modal is None:
        return None
    return serialise({key: value for key, value in modal.items() if key != "accessors"})


def _guide_text(guide: object) -> str:
    """Return the one printed line describing the guide a general run was built on.

    Parameters
    ----------
    guide : object
        The ``guide`` entry of a general modal identity, as
        :func:`solprop.translate.general.guide_identity` writes it.

    Returns
    -------
    str
        The provider's model, its region count and the resolution its reference was
        sampled at, or ``'none'`` when the entry is missing or malformed.
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

    Parameters
    ----------
    tables : object
        The ``tables`` entry of a general modal identity, as
        :attr:`solprop.translate.modal_tables.ModalTransformTables.identity`
        writes it.

    Returns
    -------
    str
        A 12-hex digest of the model identity, the gauge and the policy, or
        ``'none'`` when the entry is missing or malformed.
    """
    if not isinstance(tables, Mapping):
        return "none"
    model = tables.get("model_id")
    digest = (
        hashlib.sha256(str(model).encode("utf-8")).hexdigest()[:12]
        if model is not None
        else "unnamed"
    )
    # The digest first: the row is truncated to the block's width, and it is the
    # entry two files are compared by.
    return (
        f"model {digest}, gauge {tables.get('gauge_id')}, "
        f"policy {tables.get('policy_id')}"
    )


def _responses_text(responses: object) -> str:
    """Return the one printed line describing a general run's regional responses.

    Parameters
    ----------
    responses : object
        The ``responses`` entry of a general modal identity, or ``None`` for a
        guide whose regions are all linear.

    Returns
    -------
    str
        The compiled model, the responding region count and each region's declared
        susceptibility, or ``'none'`` when no region responds.
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
    """Return the printed summary of a coupled-flow entry, or that there is none.

    Parameters
    ----------
    flow : object
        The ``modal["flow"]`` entry, a mapping for a run carrying a coupled flow and
        absent for the cumulative diagonal one every other route uses.

    Returns
    -------
    str
        For example ``mean impedance coherent wall, constant_schur, worst ratio
        9.012e-03, conditioning 1.000e+00``.

    Examples
    --------
    >>> from solprop.output.schema import _flow_text
    >>> _flow_text(None)
    'diagonal'
    """
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

    Every schema-7 field of :attr:`RunMetadata.modal` appears in one of these rows,
    including the ``flow`` entry of a coherent wall,
    so that a printed record says which guide, which table and which basis produced
    the file; the rows are absent altogether for a route that has no such identity.

    Parameters
    ----------
    modal : collections.abc.Mapping or None
        The identity, or ``None`` for a route that declares none.

    Returns
    -------
    list
        ``(key, value)`` rows for :meth:`RunMetadata._describe`.
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
    """Return the printed row of a material manifest, or none for an empty one.

    Parameters
    ----------
    manifest : solprop.physdata.selection.MaterialManifest
        The run's manifest.

    Returns
    -------
    list
        One ``("materials", ...)`` row naming the optmat version and the models.
    """
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
        One :class:`ModeRecord` per propagated mode, in state order.
    simulation_kind : {'mode_averaged', 'multimode', 'radial', 'free', 'slab'}
        Which transform the run used.
    configuration : dict
        The physics objects that built the run -- responses, profiles, pulses,
        boundaries -- as :func:`serialise` records them. Nothing is silently dropped:
        a callable is present as its ``repr`` with ``reproducible: False``.
    stepper : StepperRecord
        The solver settings and the schedule taken.
    noise : NoiseRecord or None
        The shot-noise realisation, or ``None`` for a noiseless run.
    noise_resolution : NoiseResolution or None
        How the noise model was chosen (schema 26): the requested and the resolved
        model and any fallback's reason. Outside the hash; ``None`` when not recorded.
    backend : BackendRecord or None
        What executed the run, written by the driver from the backend's own
        ``metadata()``; ``None`` for a record built before a backend existed. Outside
        the hash, like the versions and the provenance beside it: a run continued on
        the other backend is the same run.
    versions : dict
        Package versions, from :func:`package_versions`. Outside the hash.
    git : GitProvenance or None
        Provenance of the running checkout, or ``None``. Outside the hash.
    created : str
        When the run started, ISO 8601 in UTC. Outside the hash.
    schema_version : int
        Version of this record layout.
    spectral_convention : str
        Sign convention of the stored field, always ``'optics'``.
    field_layout : str
        Axis order of the stored field.
    spatial : collections.abc.Mapping or None
        Spatial geometry, axis meanings, integration measures and field units.
        None for modal runs and for legacy version-3 metadata. Included in the
        version-4 resume key.
    modal : collections.abc.Mapping or None
        The identity of a **general modal guide**: the ordered channel labels, the
        table, metric, response and provider identities, the sampling the reference
        is rebuilt from, the gauge, the frame, the active band, the noise reference
        basis and the accessors this basis admits, as
        :func:`solprop.translate.general.modal_identity` writes it. ``None`` for
        every other route, and for any record read back from a file older than
        **output schema 7**. Included in the schema-7 resume key, so that a run
        continued against a different guide, a different table or a reordered mode
        axis is refused by the field that changed
        (:mod:`solprop.output.capability`).
    materials : solprop.physdata.selection.MaterialManifest
        The run's **material manifest** of output schema 8: the optmat version, the
        identity of every optmat model the run resolved, its species, compositions
        and solids, the policies in force and what was measured about them. Empty
        for a run of solprop's own materials, and for any record read back from a
        file older than schema 8. Its identities and policies are in the schema-8
        resume key (:meth:`~solprop.physdata.selection.MaterialManifest.hashed`), so
        a run continued with another model, policy or optmat version is refused by
        the entry that changed.
    initial_z : float
        Absolute launch position in metres, zero when absent in a historical record.
    tdse : collections.abc.Mapping or None
        Resolved scientific atomic descriptor. Included in the SEP 12 schema-10
        lineage and in schema 12; provider execution and device facts belong to the
        backend record.
    quadrature : collections.abc.Mapping or None
        The transverse rule of a resolved modal run, as
        :meth:`solprop.modes.quadrature.TransverseQuadrature.record` writes it: the
        family, the counts, the nodes and the physical weights. Output schema 17 and
        in its resume key (SEP 19). ``None`` for the mode-averaged and spatial routes,
        and for any record older than schema 17, where it means "not recorded".
    reference : collections.abc.Mapping or None
        The plane-reference record of output schema 24 (SEP 31): ``carrier``, the
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
        ``configuration["coupling"]``.
        Written to JSON from schema 24 only; ``None`` for any record read back from
        an older file.

    Examples
    --------
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.output.schema import GridRecord, RunMetadata, StepperRecord
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
    noise_resolution: NoiseResolution | None = None
    # The two development branches both used schema 10. Preserve presence, not a
    # cached digest: changing a loaded scientific value must still change its hash.
    _stored_launch_fields: tuple[str, ...] | None = dataclasses.field(
        default=None, repr=False, metadata={"record": False}
    )
    _stored_step_limit_fields: tuple[str, ...] | None = dataclasses.field(
        default=None, repr=False, metadata={"record": False}
    )

    def __repr__(self) -> str:
        """Return a one-line description: what kind of run, on what, made when.

        Returns
        -------
        str
            For example ``RunMetadata(modal, field grid of 8192, 2 modes, numba_cpu,
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

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
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
        """Stamp the creation time when the caller did not, and adopt plain copies."""
        if not self.created:
            object.__setattr__(self, "created", datetime.now(UTC).isoformat())
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "configuration", dict(self.configuration))
        object.__setattr__(self, "versions", dict(self.versions))
        # The plane-reference record is outside the hash, so a record of an earlier
        # schema may hold one; it is written to JSON from schema 24 only.
        for name in ("spatial", "modal", "reference"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, dict(value))
        if not np.isfinite(self.initial_z) or self.initial_z < 0:
            raise OutputError("initial_z must be finite and nonnegative")
        if self.tdse is not None:
            object.__setattr__(self, "tdse", dict(self.tdse))
        if self.quadrature is not None:
            if self.schema_version < 17:
                msg = (
                    "a quadrature record requires output schema 17 and this record "
                    f"declares schema {self.schema_version}: below 17 it is outside "
                    "the resume key"
                )
                raise OutputError(msg)
            object.__setattr__(self, "quadrature", dict(self.quadrature))
        if self.schema_version < 10 and (
            self.initial_z != 0.0 or self.tdse is not None
        ):
            raise OutputError(
                "interior launch and TDSE metadata require output schema 10"
            )
        self._check_historical_layout()
        if self.simulation_kind in ("radial", "free", "slab") and (
            self.schema_version < 4 or self.spatial is None
        ):
            msg = "spatial metadata requires schema 4 and a spatial geometry record"
            raise OutputError(msg)
        if (
            self.spatial is not None
            and self.schema_version >= 23
            and "flux_model" not in self.spatial
        ):
            msg = (
                "spatial metadata of schema 23 or later must name its flux_model "
                "and carry its flux_plan (SEP 34)"
            )
            raise OutputError(msg)
        if self.modal is not None and self.schema_version < 7:
            # The identity would be carried by the record and left out of the
            # digest, so two runs on different guides would share a resume key.
            msg = (
                "a general modal identity requires output schema 7 and this record "
                f"declares schema {self.schema_version}: below 7 the identity is "
                "outside the resume key, so two runs on different guides would "
                "share one"
            )
            raise OutputError(msg)
        if not self.materials.empty and self.schema_version < 8:
            # As for the modal identity: below 8 the manifest is outside the key.
            msg = (
                "a material manifest requires output schema 8 and this record "
                f"declares schema {self.schema_version}: below 8 the manifest is "
                "outside the resume key, so two runs on different optmat models "
                "would share one"
            )
            raise OutputError(msg)

    def _check_historical_layout(self) -> None:
        """Refuse values a historical field layout cannot include in its identity."""
        launch_fields = self._stored_launch_fields
        if launch_fields is None:
            launch_fields = ("initial_z", "tdse") if self.schema_version >= 12 else ()
        if (self.initial_z != 0.0 and "initial_z" not in launch_fields) or (
            self.tdse is not None and "tdse" not in launch_fields
        ):
            raise OutputError(
                "this historical metadata layout cannot record launch or TDSE identity"
            )
        if self._stored_step_limit_fields is not None:
            for name, default in (
                ("interval_cap", True),
                ("auto_init", False),
                ("interpolate_planes", False),
                ("step_planes", 0),
            ):
                if (
                    name not in self._stored_step_limit_fields
                    and getattr(self.stepper.limits, name) != default
                ):
                    raise OutputError(
                        f"historical metadata cannot record changed step limit {name}"
                    )

    @property
    def transverse_measure(self) -> TransverseMeasure:
        """Return what the run's transverse integrals measure.

        A run on modes uniform in y (:class:`solprop.modes.slab.SlabMode`) records
        ``configuration["transverse_measure"] = "per_unit_width"``: its powers are
        W/m, its energies J/m and its intensities still W/m². Every other run, and
        every file written before the key existed, measures an area. The key sits
        in the free-form configuration and is written only where it is not the
        default, so no stored field changes meaning and the schema is unchanged.

        Returns
        -------
        str
            ``"area"`` or ``"per_unit_width"``.

        Raises
        ------
        OutputError
            If the stored value is neither.

        Examples
        --------
        >>> from solprop.output.schema import _example_metadata
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
    def power_convention(self) -> str:
        """Return the amplitude label implied by this record's schema version.

        Returns
        -------
        str
            ``poynting`` for schema 6 onward, ``vacuum`` for legacy files.
        """
        return "poynting" if self.schema_version >= 6 else "vacuum"

    def hashed(self) -> dict[str, JsonValue]:
        """Return the part of the record the resume key is taken over.

        Everything that describes *what was computed*, and nothing that describes
        *when* or *with which build*: the wall-clock time, the package versions and the
        git provenance are excluded, so that a resume survives an unrelated commit, a
        dirty working tree or a NumPy upgrade. ``modal.accessors`` is excluded for
        the same reason -- it is what this build can compute, not what this run did
        (``_hashed_modal``).

        Returns
        -------
        dict
            The serialised schema version, grid, modes, simulation kind, configuration,
            noise and stepper settings, and from schema 8 the material manifest's
            identities and policies.

        Examples
        --------
        >>> from solprop.output.schema import _example_metadata
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

        Returns
        -------
        dict
            The hashed part of the record, JSON-ready.
        """
        record: dict[str, JsonValue] = {
            "schema_version": self.schema_version,
            "simulation_kind": self.simulation_kind,
            "grid": serialise(self.grid),
            "modes": _hashed_modes(self.modes, self.schema_version),
            "configuration": serialise(dict(self.configuration)),
            "noise": serialise(self.noise),
            "stepper": _hashed_stepper(
                self.stepper, self.schema_version, self._stored_step_limit_fields
            ),
        }
        # Preserve the exact version-3 key so old files remain readable and their
        # stored hash still verifies. Continuing them with version 4 is refused.
        if self.schema_version >= 4:
            record["spatial"] = serialise(self.spatial)
        if self.schema_version >= 6:
            record["power_convention"] = self.power_convention
        if self.schema_version >= 7:
            # The same guard as the schema-6 power convention: the general-guide
            # identity enters the key at schema 7 and not before, so a schema-6
            # digest stays what it was stamped with. All of it but the accessor
            # list, which is the build's capability and not the run's description.
            record["modal"] = _hashed_modal(self.modal)
        if self.schema_version >= 8:
            # The material manifest's identities and policies, under the same guard,
            # so that every digest stamped before schema 8 still verifies. A run of
            # solprop's own materials keys on `None`.
            record["materials"] = serialise(self.materials.hashed())
        launch_fields = self._stored_launch_fields
        if launch_fields is None:
            launch_fields = ("initial_z", "tdse") if self.schema_version >= 12 else ()
        if "initial_z" in launch_fields:
            record["initial_z"] = self.initial_z
        if "tdse" in launch_fields:
            record["tdse"] = serialise(self.tdse)
        if self.schema_version >= 17:
            # The transverse rule (SEP 19), under the same guard as every addition,
            # so that each digest stamped before schema 17 still verifies.
            record["quadrature"] = serialise(self.quadrature)
        return record

    def configuration_hash(self) -> str:
        """Return the resume key: a digest of everything that was computed.

        A SHA-256 of the canonical JSON of :meth:`hashed`, so it is stable across
        processes -- unlike :func:`hash`, which is salted per interpreter. It covers the
        physics and not only the statistic names and the state shape, so a resume with
        different physics of the same shape is refused.

        Returns
        -------
        str
            Sixty-four lowercase hexadecimal characters.

        Examples
        --------
        >>> from solprop.output.schema import _example_metadata
        >>> metadata = _example_metadata()
        >>> metadata.configuration_hash() == metadata.configuration_hash()
        True
        """
        return self._configuration_digest

    @functools.cached_property
    def _configuration_digest(self) -> str:
        """The resume key, computed once per record (:meth:`configuration_hash`).

        Returns
        -------
        str
            The SHA-256 hex digest of the canonical JSON of the hashed part.
        """
        text = _canonical_json(self._hashed_record)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def differences(self, other: "RunMetadata") -> tuple[str, ...]:  # noqa: UP037 - bound after the class body
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
        >>> from solprop.output.schema import _example_metadata
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
        other: "RunMetadata",  # noqa: UP037 - bound after the class body
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
        >>> from solprop.output.schema import _example_metadata
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

    def to_json(self) -> str:
        """Return the whole record as one JSON document.

        Returns
        -------
        str
            Canonical JSON carrying the hashed part plus the versions, the git
            provenance, the creation time and -- at schema 7 and above -- the
            whole general modal identity, whose published accessor list the
            hashed part leaves out.

        Examples
        --------
        >>> import json
        >>> from solprop.output.schema import _example_metadata
        >>> sorted(json.loads(_example_metadata().to_json()))[:3]
        ['backend', 'configuration', 'created']
        """
        document = self.hashed()
        if self.schema_version >= 7:
            # The whole identity, including the accessor list `hashed` leaves out:
            # what a file records and what its resume key is taken over are two
            # different questions, and the published accessors answer only the
            # first (:func:`_hashed_modal`).
            document["modal"] = serialise(self.modal)
        if self.schema_version >= 8:
            # The whole manifest, measurements included, beside its hashed part.
            document["materials"] = serialise(self.materials.to_dict())
        if self.schema_version >= 24:
            # Derived from the hashed problem, so stored beside the key, not in it: a
            # resume on another build may evaluate the frame to a different last bit.
            document["reference"] = serialise(self.reference)
        if self.schema_version >= 26:
            # How the model was chosen, beside the key: the resolved model is in the
            # hashed noise record already, and a request is not part of the physics.
            document["noise_resolution"] = serialise(self.noise_resolution)
        document["spectral_convention"] = self.spectral_convention
        document["field_layout"] = self.field_layout
        document["backend"] = serialise(self.backend)
        document["versions"] = serialise(dict(self.versions))
        document["git"] = serialise(self.git)
        document["created"] = self.created
        return _canonical_json(document)

    @classmethod
    def from_json(cls, text: str) -> "RunMetadata":  # noqa: UP037 - bound after the class body
        """Return the record a JSON document describes.

        The configuration comes back as the plain structures :func:`serialise` made of
        it, not as the physics objects: a callable was recorded by ``repr`` and cannot
        be rebuilt. That is enough for the hash, which is what a resume compares.

        Parameters
        ----------
        text : str
            A document produced by :meth:`to_json`.

        Returns
        -------
        RunMetadata
            The record, hashing identically to the one that produced ``text``.

        Raises
        ------
        OutputError
            If the document is not valid JSON, or a required field is missing.

        Examples
        --------
        >>> from solprop.output.schema import _example_metadata
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
        try:
            noise = document["noise"]
            resolution = document.get("noise_resolution")
            git = document["git"]
            backend = document["backend"]
            return cls(
                grid=_rebuild(GridRecord, document["grid"]),
                modes=tuple(_rebuild(ModeRecord, one) for one in document["modes"]),
                simulation_kind=document["simulation_kind"],
                configuration=document["configuration"],
                stepper=StepperRecord.from_record(document["stepper"]),
                noise=None if noise is None else _rebuild(NoiseRecord, noise),
                backend=None if backend is None else _rebuild(BackendRecord, backend),
                versions=document["versions"],
                git=None if git is None else _rebuild(GitProvenance, git),
                created=document["created"],
                schema_version=document["schema_version"],
                spectral_convention=document["spectral_convention"],
                field_layout=document["field_layout"],
                spatial=document.get("spatial"),
                modal=document.get("modal"),
                materials=MaterialManifest.from_dict(document.get("materials")),
                initial_z=document.get("initial_z", 0.0),
                tdse=document.get("tdse"),
                quadrature=document.get("quadrature"),
                reference=document.get("reference"),
                noise_resolution=None
                if resolution is None
                else _rebuild(NoiseResolution, resolution),
                _stored_launch_fields=tuple(
                    key for key in ("initial_z", "tdse") if key in document
                ),
                _stored_step_limit_fields=tuple(document["stepper"]["limits"]),
            )
        except (KeyError, TypeError, MaterialError) as error:
            msg = f"stored metadata is missing or malformed: {error}"
            raise OutputError(msg) from error


def _example_metadata() -> RunMetadata:
    """Return a minimal record, so the examples above stay one line each.

    A documentation and experimentation helper, not part of the public surface: build a
    real record with the :class:`RunMetadata` constructor, whose own example shows how.

    Returns
    -------
    RunMetadata
        A mode-averaged record on a small envelope grid with one configuration entry.
    """
    grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    return RunMetadata(
        grid=GridRecord.from_grid(grid),
        modes=(),
        simulation_kind="mode_averaged",
        configuration={"gamma": 0.11},
        stepper=StepperRecord(tableau="DOPRI5", controller={"rtol": 1e-8, "atol": 0.0}),
    )


# ---------------------------------------------------------------- the save protocol


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
    >>> from solprop.output.schema import _example_metadata, resolve_metadata_write
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


FieldsChoice = Literal["none", "planes", "all"]
"""Which saved planes keep their field: none of them, the named ones, or all."""


def field_retention_refusal(fields: object, field_planes: object) -> str | None:
    """Say what is wrong with a field-retention request, or that nothing is.

    The one validator behind :class:`solprop.driver.simulation.SimulationSettings`,
    :func:`solprop.driver.run.run_problem`, :func:`solprop.driver.scan.run_scan` and
    :class:`FieldRetention`; each door raises its own layer's error with the message.

    Parameters
    ----------
    fields : object
        The request, one of ``"none"``, ``"planes"`` and ``"all"``.
    field_planes : object
        Saved-plane indices, the entrance as 0 and negative from the end; required
        exactly when `fields` is ``"planes"``.

    Returns
    -------
    str or None
        Why the request is refused, or ``None`` when it is valid.

    Examples
    --------
    >>> from solprop.output.schema import field_retention_refusal
    >>> field_retention_refusal("planes", [0, -1]) is None
    True
    >>> field_retention_refusal("planes", [])
    "field_planes is required exactly when fields='planes'"
    """
    if fields not in ("none", "planes", "all"):
        return f"fields must be none, planes or all, got {fields!r}"
    try:
        planes = tuple(cast("Sequence[object]", field_planes))
    except TypeError:
        return (
            "field_planes must be a sequence of saved-plane indices, got "
            f"{field_planes!r}"
        )
    if any(
        isinstance(index, bool) or not isinstance(index, int | np.integer)
        for index in planes
    ):
        return f"field_planes must hold integer saved-plane indices, got {planes!r}"
    if (fields == "planes") != bool(planes):
        return "field_planes is required exactly when fields='planes'"
    return None


@dataclass(frozen=True)
class FieldRetention:
    """Which saved planes of a run keep their field (SEP 22 section 5.1).

    A run is **homogeneous**: it keeps the field on every plane, on the planes it
    names, or on none, and its statistics on every plane either way. A plane without
    its field stores no ``field`` and no ``noise_phase``; what it saves is its
    position, its statistics and its step history.

    Attributes
    ----------
    fields : {'none', 'planes', 'all'}
        Keep no field, the fields of `field_planes`, or every field (the default).
    field_planes : tuple of int
        Saved-plane indices for ``"planes"``, the entrance as 0 and negative from the
        end until :meth:`resolved`; empty otherwise.

    Raises
    ------
    OutputError
        If the request is not valid (:func:`field_retention_refusal`).

    Examples
    --------
    >>> from solprop.output.schema import FieldRetention
    >>> kept = FieldRetention("planes", (0, -1)).resolved(5)
    >>> kept.field_planes, kept.keeps(4), kept.keeps(2), kept.kept(5)
    ((0, 4), True, False, 2)
    """

    fields: FieldsChoice = "all"
    field_planes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate the request and adopt plain integer indices.

        Raises
        ------
        OutputError
            Naming what is wrong.
        """
        refusal = field_retention_refusal(self.fields, self.field_planes)
        if refusal is not None:
            raise OutputError(refusal)
        object.__setattr__(
            self, "field_planes", tuple(int(index) for index in self.field_planes)
        )

    def resolved(self, n_planes: int) -> "FieldRetention":  # noqa: UP037 - bound after the class body
        """Return the rule with its indices counted from the entrance, sorted, unique.

        Parameters
        ----------
        n_planes : int
            The planes the run saves, the entrance included.

        Returns
        -------
        FieldRetention
            The same rule with non-negative indices.

        Raises
        ------
        OutputError
            If an index lies outside the run's planes.
        """
        resolved = set()
        for index in self.field_planes:
            plane = index + n_planes if index < 0 else index
            if not 0 <= plane < n_planes:
                msg = f"field plane {index} is outside a run of {n_planes} saved planes"
                raise OutputError(msg)
            resolved.add(plane)
        return FieldRetention(self.fields, tuple(sorted(resolved)))

    def keeps(self, index: int) -> bool:
        """Return whether the saved plane at `index` keeps its field.

        Parameters
        ----------
        index : int
            The plane's index, the entrance as 0; the rule must be :meth:`resolved`.

        Returns
        -------
        bool
            True for every plane of an ``"all"`` run, a named plane of a ``"planes"``
            run, and no plane of a ``"none"`` run.
        """
        if self.fields == "all":
            return True
        return index in self.field_planes

    def kept(self, n_planes: int) -> int:
        """Return how many of a run's planes keep their field.

        Parameters
        ----------
        n_planes : int
            The planes the run saves; the rule must be :meth:`resolved`.

        Returns
        -------
        int
            The planes whose field is stored.
        """
        if self.fields == "all":
            return n_planes
        return len(self.field_planes)

    def describe(self) -> str:
        """Return the rule in words, for a refusal naming it.

        Returns
        -------
        str
            For example ``"fields='planes' (saved planes 0, 4)"``.
        """
        if self.fields == "planes":
            planes = ", ".join(str(index) for index in self.field_planes)
            return f"fields='planes' (saved planes {planes})"
        return f"fields={self.fields!r}"


@dataclass(frozen=True)
class PlaneLayout:
    """What every plane of one run has in common, fixed by the first plane saved.

    Attributes
    ----------
    field_shape : tuple of int
        ``(n_modes, n_pol, n_omega)`` of the stored field, or of the run's state when
        it keeps no field; empty when neither is known.
    stat_names : tuple of str
        The statistic names, sorted.
    has_noise_phase : bool
        Whether a noise phase is stored beside each stored field.
    step_stat_names : tuple of str
        The statistics the between-planes channel carries, sorted; empty for a run
        without a periodic subset, which is the default.
    retention : FieldRetention
        Which planes keep their field, resolved; every plane by default.

    Examples
    --------
    >>> from solprop.output.schema import PlaneLayout
    >>> layout = PlaneLayout(
    ...     field_shape=(1, 1, 8), stat_names=("energy",), has_noise_phase=False
    ... )
    >>> layout.field_shape, layout.stat_names, layout.retention.fields
    ((1, 1, 8), ('energy',), 'all')
    """

    field_shape: tuple[int, ...]
    stat_names: tuple[str, ...]
    has_noise_phase: bool
    step_stat_names: tuple[str, ...] = ()
    retention: FieldRetention = FieldRetention()


@runtime_checkable
class Output(Protocol):
    """What every output handler does, and all that processing may assume of one.

    One protocol so that :class:`solprop.output.memory.MemoryOutput` and
    :class:`solprop.output.hdf5.HDF5Output` are interchangeable: a run is
    written to memory for a scan and to disk for a long propagation without the driver
    knowing which.

    Everything a handler hands back through ``__getitem__`` is *as stored*, so the field
    is in the optics convention; ``fields`` is the accessor that also offers the
    internal one, for direct comparison with a file stored in the FFT convention.
    """

    def open(self, metadata: RunMetadata, *, force: bool = False) -> None:
        """Begin a run, recording what produced it.

        Parameters
        ----------
        metadata : RunMetadata
            The record of the run about to start.
        force : bool, optional
            Whether to overwrite metadata already held for a different run.
        """
        ...

    def save(self, plane: SavedPlane) -> None:
        """Store one saved plane.

        Parameters
        ----------
        plane : SavedPlane
            The position, field, statistics and step history to store.
        """
        ...

    def close(self) -> None:
        """Finish the run, releasing any resource held. Calling it twice is harmless."""
        ...

    @property
    def n_saved(self) -> int:
        """Return how many planes have been stored.

        Returns
        -------
        int
            The number of planes.
        """
        ...

    @property
    def metadata(self) -> RunMetadata | None:
        """Return the record of the run, or ``None`` before it is opened.

        Returns
        -------
        RunMetadata or None
            What produced the stored data.
        """
        ...

    @property
    def stat_names(self) -> tuple[str, ...]:
        """Return the names of the statistics stored at every plane.

        Returns
        -------
        tuple of str
            The names, sorted; empty before the first plane is saved.
        """
        ...

    @property
    def step_stat_names(self) -> tuple[str, ...]:
        """Return the statistics stored between planes, if any.

        Returns
        -------
        tuple of str
            The names, sorted; empty when the run recorded no periodic subset.
        """
        ...

    def __getitem__(self, key: str) -> np.ndarray:
        """Return one stored array by name.

        Parameters
        ----------
        key : str
            ``'field'``, ``'z'``, ``'noise_phase'``, ``'field_planes'``,
            ``'stats/<name>'``, ``'step_history/<name>'`` or ``'step_stats/<name>'``.

        Returns
        -------
        numpy.ndarray
            The stored array, in the convention it is stored in.
        """
        ...

    def __contains__(self, key: object) -> bool:
        """Return whether ``self[key]`` would return an array, never raising.

        Parameters
        ----------
        key : object
            A candidate key.

        Returns
        -------
        bool
            ``True`` exactly when ``__getitem__`` would return an array for it.
        """
        ...

    def fields(self, convention: Convention = "optics") -> np.ndarray:
        """Return every stored field, in the requested sign convention.

        Parameters
        ----------
        convention : {'optics', 'fft'}, optional
            ``'optics'`` returns the arrays as stored; ``'fft'`` converts them back to
            the internal convention a backend works in, for direct comparison with a
            file stored in the FFT convention.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_z, n_modes, n_pol, n_omega)``, complex.
        """
        ...


@runtime_checkable
class CheckpointOutput(Protocol):
    """Declare optional outcome and terminal-checkpoint persistence for a handler."""

    def record_outcome(
        self, outcome: RunOutcome, checkpoint: TerminalCheckpoint | None = None
    ) -> None:
        """Persist execution status and an optional accepted terminal field.

        Parameters
        ----------
        outcome : RunOutcome
            Cumulative work and completion status.
        checkpoint : TerminalCheckpoint or None
            Accepted field between regular saves. None retains a pending checkpoint
            during progress and clears it when the run completes.
        """
        ...

    @property
    def outcome(self) -> RunOutcome | None:
        """Return recorded status, or None for legacy output.

        Returns
        -------
        RunOutcome or None
            Last persisted execution outcome.
        """
        ...

    @property
    def terminal_checkpoint(self) -> TerminalCheckpoint | None:
        """Return the latest accepted terminal field outside regular observations.

        Returns
        -------
        TerminalCheckpoint or None
            The checkpoint, if one has been recorded.
        """
        ...


class PlaneStore(Displayed, ABC):
    """The shared half of an output handler: one save protocol for every backing store.

    :class:`solprop.output.memory.MemoryOutput` and
    :class:`solprop.output.hdf5.HDF5Output` differ only in where the arrays go. What
    they must not differ in is *what* goes there: the sign convention of the stored
    field, the metadata overwrite rule, the key names and the step-history layout. Those
    live here, once, which is what makes the two handlers store a run bitwise
    identically, with one overwrite rule and one growth strategy.

    The base class also holds the run's **layout** -- the field's shape, the statistic
    names and whether a noise phase is stored -- fixed by the first plane saved and
    checked against every later one, so that a plane that does not fit is refused by
    name in either handler, rather than being dropped or raising deep inside the writer.

    A subclass implements four hooks: ``_write_metadata``, ``_append``, ``_read`` and
    ``_release``, and may implement a fifth, ``_write_annotation``, to persist the
    post-run annotations (:meth:`annotate`) a store held only in memory otherwise.

    A driver that knows how many planes the run will save says so with
    :meth:`reserve`, and each store sizes its first allocation for them; a run that
    saves more grows past it, so the number is a hint, never a limit.

    Examples
    --------
    >>> from solprop.output import MemoryOutput
    >>> from solprop.output.schema import Output, PlaneStore
    >>> handler = MemoryOutput()
    >>> isinstance(handler, PlaneStore), isinstance(handler, Output)
    (True, True)
    """

    def __init__(self) -> None:
        """Start with nothing stored and nothing opened."""
        self._metadata: RunMetadata | None = None
        self._n_saved = 0
        self._planned = 0
        self._closed = False
        self._layout: PlaneLayout | None = None
        # Which planes keep their field, and the state's shape for a run that keeps
        # none, told by the run before its first plane (`retain`).
        self._retention = FieldRetention()
        self._field_shape: tuple[int, ...] = ()
        self._annotations: dict[str, JsonValue] = {}
        self._outcome: RunOutcome | None = None
        self._terminal_checkpoint: TerminalCheckpoint | None = None
        self._checkpoint_pending = False
        self._resume_outcome_pending = False

    @property
    def outcome(self) -> RunOutcome | None:
        """Return the execution outcome, absent on legacy files.

        Returns
        -------
        RunOutcome or None
            The last recorded completion state and cumulative ledgers.
        """
        return self._outcome

    @property
    def terminal_checkpoint(self) -> TerminalCheckpoint | None:
        """Return the terminal checkpoint without treating it as a saved plane.

        Returns
        -------
        TerminalCheckpoint or None
            A field in the internal FFT convention and output amplitude units.
        """
        return self._terminal_checkpoint

    def record_outcome(
        self, outcome: RunOutcome, checkpoint: TerminalCheckpoint | None = None
    ) -> None:
        """Persist status and optional terminal field without extending regular saves.

        Parameters
        ----------
        outcome : RunOutcome
            Current status and cumulative call ledgers.
        checkpoint : TerminalCheckpoint or None
            Accepted terminal field, with no newly requested atomic statistics.

        Raises
        ------
        OutputError
            If the handler is closed, metadata is older than schema 10, or the
            checkpoint disagrees with the outcome or stored field layout.
        """
        self._check_usable()
        metadata = self._require_metadata()
        if metadata.schema_version < 10:
            raise OutputError(
                "outcomes and terminal checkpoints require output schema 10"
            )
        previous = self._outcome
        if previous is not None:
            history = previous.previous_executions
            if self._resume_outcome_pending or (
                previous.status != "running" and outcome.status == "running"
            ):
                closing = dataclasses.asdict(previous)
                closing.pop("previous_executions")
                history = (*history, closing)
            outcome = dataclasses.replace(outcome, previous_executions=history)
        if checkpoint is not None:
            if checkpoint.plane.z != outcome.z:
                raise OutputError("checkpoint and outcome positions must agree")
            if (
                self._layout is not None
                and self._layout.field_shape
                and checkpoint.plane.shape != self._layout.field_shape
            ):
                raise OutputError("checkpoint field shape differs from saved planes")
            if self._n_saved and checkpoint.plane.z < self._last_saved_z():
                raise OutputError("checkpoint precedes the last regular saved plane")
            if self._checkpoint_pending and self._terminal_checkpoint is not None:
                checkpoint = dataclasses.replace(
                    checkpoint,
                    plane=_prepend_checkpoint_history(
                        self._terminal_checkpoint.plane, checkpoint.plane
                    ),
                )
        elif outcome.status == "running":
            checkpoint = self._terminal_checkpoint
        self._write_outcome(outcome, checkpoint)
        self._outcome = outcome
        self._resume_outcome_pending = False
        self._terminal_checkpoint = checkpoint
        if outcome.status != "running":
            self._checkpoint_pending = False

    def _last_saved_z(self) -> float:
        """Return the position of the last regular saved plane.

        Returns
        -------
        float
            The position, m.
        """
        return float(self["z"][-1])

    def _write_outcome(
        self, outcome: RunOutcome | None, checkpoint: TerminalCheckpoint | None
    ) -> None:
        """Persist records; the memory store needs no additional storage hook.

        Parameters
        ----------
        outcome : RunOutcome or None
            Current execution outcome.
        checkpoint : TerminalCheckpoint or None
            Accepted checkpoint, or None to clear an older one.
        """

    @property
    def annotations(self) -> Mapping[str, JsonValue]:
        """Return the post-run annotations the store holds, by name.

        An annotation is something learned only once the run has closed -- the
        material manifest's peak-field report (:data:`PEAK_FIELDS_ANNOTATION`) is
        one -- so it is not part of the run's record: the configuration hash, a
        resume and a scan's reuse never see it. A store without any, a file written
        before annotations existed included, answers with an empty mapping.

        Returns
        -------
        collections.abc.Mapping
            Each annotation's plain JSON values, by name; a copy.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> dict(MemoryOutput().annotations)
        {}
        """
        return cast("dict[str, JsonValue]", json.loads(json.dumps(self._annotations)))

    def annotate(self, name: str, value: object) -> None:
        """Record a post-run annotation, replacing any of the same name.

        The run's record is not touched: an annotation stays outside the
        configuration hash, so a file annotated after its run is still the same run
        to :func:`solprop.output.hdf5.check_cache` and to a scan. A store may be
        annotated after it is closed; a file is then reopened briefly to write it.

        Parameters
        ----------
        name : str
            A non-empty name without ``/``, such as :data:`PEAK_FIELDS_ANNOTATION`.
        value : object
            Plain JSON values: mappings with string keys, lists, strings, numbers,
            booleans, None; anything else is refused here, the store's boundary.

        Raises
        ------
        OutputError
            If the store was never opened, the name is empty or holds ``/``, or the
            value is not plain JSON.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> from solprop.output.schema import _example_metadata
        >>> handler = MemoryOutput()
        >>> handler.open(_example_metadata())
        >>> handler.annotate("note", {"peak": 1.5})
        >>> handler.annotations["note"]
        {'peak': 1.5}
        """
        self._require_metadata()
        if not name or "/" in name:
            msg = (
                f"an annotation is named by a non-empty name without '/', got {name!r}"
            )
            raise OutputError(msg)
        try:
            text = json.dumps(value, sort_keys=True)
        except TypeError as error:
            msg = f"annotation {name!r} is not plain JSON: {error}"
            raise OutputError(msg) from error
        self._write_annotation(name, text)
        self._annotations[name] = cast("JsonValue", json.loads(text))

    def _write_annotation(self, name: str, text: str) -> None:
        """Persist one annotation; a store that lives in memory has nothing to do.

        Parameters
        ----------
        name : str
            The annotation's name.
        text : str
            Its JSON document.
        """

    @property
    def metadata(self) -> RunMetadata | None:
        """Return the record of the run, or ``None`` before it is opened.

        Returns
        -------
        RunMetadata or None
            What produced the stored data.
        """
        return self._metadata

    @property
    def n_saved(self) -> int:
        """Return how many planes have been stored.

        Returns
        -------
        int
            The number of planes.
        """
        return self._n_saved

    @property
    def closed(self) -> bool:
        """Return whether the handler has been closed.

        Returns
        -------
        bool
            True once :meth:`close` has run.
        """
        return self._closed

    @property
    def stat_names(self) -> tuple[str, ...]:
        """Return the names of the statistics stored at every plane.

        Read off the layout the first plane fixed, so that the two handlers answer
        alike, and a file continued or opened for reading answers from what it holds.

        Returns
        -------
        tuple of str
            The names, sorted; empty before the first plane is saved.
        """
        return () if self._layout is None else self._layout.stat_names

    @property
    def step_stat_names(self) -> tuple[str, ...]:
        """Return the statistics the between-planes channel carries.

        Empty for a run that asked for no periodic subset, and for a file written
        before schema version 3, which is how a reader tells whether the channel is
        there to prefer over the planes.

        Returns
        -------
        tuple of str
            The names, sorted; empty when the run has no channel.
        """
        return () if self._layout is None else self._layout.step_stat_names

    def _state_text(self) -> str:
        """Return whether the store is open, for printing.

        Returns
        -------
        str
            ``"open"`` or ``"closed"``.
        """
        return "closed" if self._closed else "open"

    def _span_text(self) -> str:
        """Return the positions the saved planes span, for printing.

        Returns
        -------
        str
            The first and last position, or a note that they cannot be read now.
        """
        if not self._n_saved:
            return "no plane saved yet"
        try:
            z = np.asarray(self["z"], dtype=float)
        except OutputError:
            return "positions unavailable while the store is closed"
        return f"{engineering(float(z[0]), 'm')} to {engineering(float(z[-1]), 'm')}"

    def __repr__(self) -> str:
        """Return a one-line description: how much is stored, and whether it is open.

        Returns
        -------
        str
            For example ``MemoryOutput(201 planes, 2 modes, closed)``.
        """
        fields: list[tuple[str, object]] = [("", f"{self._n_saved} planes")]
        if self._layout is not None:
            fields.append(("", "x".join(str(n) for n in self._layout.field_shape)))
        fields.append(("", self._state_text()))
        return line(type(self).__name__, fields)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        Returns
        -------
        tuple
            The title and ``(key, value)`` rows.
        """
        rows: list[tuple[str, object]] = [
            ("state", self._state_text()),
            ("planes", f"{self._n_saved}, {self._span_text()}"),
        ]
        if self._layout is not None:
            shape = " x ".join(str(n) for n in self._layout.field_shape)
            rows.append(("field per plane", f"{shape} (modes x components x omega)"))
            rows.append(("statistics", ", ".join(self._layout.stat_names) or "none"))
            between = ", ".join(self._layout.step_stat_names) or "none"
            rows.append(("between planes", between))
        if self._metadata is not None:
            rows.append(("run", repr(self._metadata)))
        return type(self).__name__, rows

    def __enter__(self) -> "PlaneStore":  # noqa: UP037 - bound after the class body
        """Return the handler, so that a ``with`` block closes it on the way out.

        Returns
        -------
        PlaneStore
            This handler.
        """
        return self

    def __exit__(self, *details: object) -> None:
        """Close the handler when the block ends, whatever happened inside it.

        Parameters
        ----------
        *details : object
            The exception being raised, if any; it is not suppressed.
        """
        self.close()

    def open(self, metadata: RunMetadata, *, force: bool = False) -> None:
        """Begin a run, recording what produced it.

        Parameters
        ----------
        metadata : RunMetadata
            The record of the run about to start.
        force : bool, optional
            Whether to overwrite metadata already held for a different run. Without it
            a second, different record raises; with it the record is replaced and the
            replacement logged (:func:`resolve_metadata_write`).

        Raises
        ------
        OutputError
            If the handler is closed, or a different record is already held and
            ``force`` is false.
        """
        self._check_usable()
        if resolve_metadata_write(self._metadata, metadata, force=force):
            self._write_metadata(metadata)
        self._metadata = metadata

    def save(self, plane: SavedPlane) -> None:
        """Store one saved plane, converting its field to the optics convention.

        Parameters
        ----------
        plane : SavedPlane
            The position, field, statistics and step history to store. Its field is in
            the internal FFT convention and is written out as
            :func:`solprop.fourier.spectrum_to_optics` of it; a field-free plane
            stores none.

        Raises
        ------
        OutputError
            If the handler is closed or was never opened, or the plane does not fit
            the run's layout: a field where the run keeps none, none where it keeps
            one (:meth:`retain`), another shape or other statistics.
        """
        self._check_usable()
        metadata = self._require_metadata()
        if self._checkpoint_pending and self._terminal_checkpoint is not None:
            plane = _prepend_checkpoint_history(self._terminal_checkpoint.plane, plane)
        stored = (
            None
            if plane.field is None
            else fourier.spectrum_to_optics(plane.field, kind=metadata.grid.kind)
        )
        self._check_layout(plane, stored)
        self._append(plane, stored)
        self._n_saved += 1
        if self._terminal_checkpoint is not None:
            self._write_outcome(self._outcome, None)
            self._terminal_checkpoint = None
            self._checkpoint_pending = False

    def reserve(self, n_planes: int) -> None:
        """Expect a run of `n_planes` saved planes in all, before the first is saved.

        A store sizes its plane-indexed arrays for this many planes when the first
        plane fixes their shapes, rather than growing them plane by plane; a run that
        saves more grows them past it (by doubling), and one that saves fewer uses
        only what it saved, so the count is a hint and changes nothing that is
        stored. The run loop (:func:`solprop.driver.run.run_problem`) reserves the
        entrance plane and every save plane. A later call can only raise the count.

        Parameters
        ----------
        n_planes : int
            The planes the run will save, the entrance included; non-negative.

        Raises
        ------
        OutputError
            If `n_planes` is negative.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> handler = MemoryOutput()
        >>> handler.reserve(201)
        >>> handler.n_saved
        0
        """
        if n_planes < 0:
            msg = f"a store reserves a non-negative number of planes, got {n_planes}"
            raise OutputError(msg)
        self._planned = max(self._planned, int(n_planes))

    def retain(self, retention: FieldRetention, field_shape: tuple[int, ...]) -> None:
        """Say which planes of the run keep their field, before the first is saved.

        The run loop (:func:`solprop.driver.run.run_problem`) calls it with the run's
        resolved rule. A store then refuses a plane that breaks the rule, sizes its
        field storage for the planes that keep one, and answers field reads on the
        others by naming the rule. A store continuing a run (a resumed file) already
        holds a rule and accepts only the same one.

        Parameters
        ----------
        retention : FieldRetention
            The rule, :meth:`~FieldRetention.resolved` against the run's planes.
        field_shape : tuple of int
            ``(n_modes, n_pol, n_omega)`` of the run's state, which a run that keeps no
            field records in its place.

        Raises
        ------
        OutputError
            If the store already holds planes under another rule.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> from solprop.output.schema import FieldRetention
        >>> handler = MemoryOutput()
        >>> handler.retain(FieldRetention("none"), (1, 1, 8))
        >>> handler.retention.fields, handler.field_shape
        ('none', (1, 1, 8))
        """
        if self._n_saved and retention != self._retention:
            msg = (
                f"this output holds planes saved with {self._retention.describe()}, "
                f"and a run continuing it asks for {retention.describe()}; a run keeps "
                "its fields by one rule, so continue it with the same fields and "
                "field_planes"
            )
            raise OutputError(msg)
        self._retention = retention
        self._field_shape = tuple(int(size) for size in field_shape)

    @property
    def retention(self) -> FieldRetention:
        """Return which planes of the run keep their field.

        Returns
        -------
        FieldRetention
            The rule the run declared (:meth:`retain`), every plane by default.
        """
        return self._retention

    @property
    def field_shape(self) -> tuple[int, ...]:
        """Return the shape of one plane's field, stored or not.

        Returns
        -------
        tuple of int
            ``(n_modes, n_pol, n_omega)``; empty before anything has said it.
        """
        if self._layout is not None and self._layout.field_shape:
            return self._layout.field_shape
        return self._field_shape

    def held_bytes(
        self,
        n_planes: int,
        plane_bytes: int,
        *,
        kept: int | None = None,
        field_bytes: int = 0,
    ) -> int:
        """Return the bytes the store keeps in memory for a run, for its admission.

        A store that writes each plane out keeps none beyond the plane being written,
        which is what this base answer says; one that keeps the run in memory
        overrides it (:meth:`solprop.output.memory.MemoryOutput.held_bytes`). The
        run's memory admission adds the answer to the backend's estimate
        (:func:`solprop.driver.memory.admit_memory`).

        Parameters
        ----------
        n_planes : int
            The planes the run will save, the entrance included.
        plane_bytes : int
            The bytes of one plane as the store holds it: the field, the noise phase
            and the statistics.
        kept : int or None, optional
            How many of the planes keep their field (:class:`FieldRetention`);
            ``None``, every one.
        field_bytes : int, optional
            The part of `plane_bytes` a plane without its field does not hold: the
            field and the noise phase.

        Returns
        -------
        int
            Zero: the plane in flight is a transient, not a persistent allocation.

        Examples
        --------
        >>> import tempfile
        >>> from pathlib import Path
        >>> from solprop.output import HDF5Output
        >>> with tempfile.TemporaryDirectory() as folder:
        ...     HDF5Output(Path(folder) / "run.h5").held_bytes(201, 2**20)
        0
        """
        del n_planes, plane_bytes, kept, field_bytes
        return 0

    def close(self) -> None:
        """Finish the run, releasing whatever the store holds open.

        Calling it a second time does nothing, so a driver may close in a ``finally``
        and a caller may close again without checking.
        """
        if self._closed:
            return
        self._release()
        self._closed = True

    def fields(self, convention: Convention = "optics") -> np.ndarray:
        """Return every stored field, in the requested sign convention.

        Parameters
        ----------
        convention : {'optics', 'fft'}, optional
            ``'optics'`` returns the arrays exactly as stored; ``'fft'`` converts them
            back to the internal convention a backend works in, so that a file stored in
            the FFT convention can be compared with them bin by bin.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_z, n_modes, n_pol, n_omega)``, complex.

        Raises
        ------
        OutputError
            If nothing has been stored yet, or the run did not keep the field on
            every plane (:meth:`plane_field` reads a plane that kept one).
        """
        stored = self["field"]
        if convention == "fft":
            kind = self._require_metadata().grid.kind
            return fourier.spectrum_to_internal(stored, kind=kind)
        return stored

    def __getitem__(self, key: str) -> np.ndarray:
        """Return one stored array by name, as stored.

        Parameters
        ----------
        key : str
            ``'field'``, ``'z'``, ``'noise_phase'``, ``'stats/<name>'``,
            ``'step_history/<name>'`` or ``'step_stats/<name>'``. The step history is
            concatenated over planes, with ``'step_history/counts'`` giving the number
            of attempts belonging to each; the between-planes statistics are laid out
            the same way, with ``'step_stats/z'``, ``'step_stats/step'`` and
            ``'step_stats/counts'`` beside the named statistics, and read back empty
            for a run that recorded none; ``'noise_phase'`` is there only when the run
            carried a noise model.

        Returns
        -------
        numpy.ndarray
            The stored array; the field in the optics convention.

        Raises
        ------
        OutputError
            If the key is not one of those forms, or names something not stored; for
            ``'field'`` and ``'noise_phase'``, also when the run did not keep the
            field on every plane, naming its ``fields`` (:meth:`plane_field` reads a
            plane that kept one). ``'field_planes'`` lists the saved planes that keep
            their field, for every run.
        """
        group, name = _split_key(key)
        if group == "" and name == "field_planes":
            return self._field_planes()
        if group == "" and name in ("field", "noise_phase"):
            self._refuse_partial_fields(name)
        return self._read(group, name)

    def _field_planes(self) -> Int[np.ndarray, " n_kept"]:
        """Return the indices of the saved planes that keep their field.

        Returns
        -------
        numpy.ndarray
            The indices, int64 and increasing; every saved plane of an ``"all"`` run.
        """
        indices = range(self._n_saved)
        return np.array(
            [index for index in indices if self._retention.keeps(index)], dtype=np.int64
        )

    def _refuse_partial_fields(self, name: str) -> None:
        """Refuse a whole-run field read on a run that kept fields on some planes only.

        Parameters
        ----------
        name : str
            ``'field'`` or ``'noise_phase'``.

        Raises
        ------
        OutputError
            Naming the run's ``fields``.
        """
        if self._retention.fields == "all":
            return
        where = (
            "no plane"
            if self._retention.fields == "none"
            else "only the saved planes listed by output['field_planes']"
        )
        msg = (
            f"this run was saved with {self._retention.describe()}: its {name} is "
            f"stored on {where}, so there is no {name} for every plane to read; "
            "read a plane that kept its field with plane_field(index), or rerun with "
            "fields='all'"
        )
        raise OutputError(msg)

    def plane_field(self, index: int, convention: Convention = "optics") -> np.ndarray:
        """Return one saved plane's field, in the requested sign convention.

        The accessor that works whatever the run kept: a plane that kept its field
        returns it, and one that did not is refused by name.

        Parameters
        ----------
        index : int
            The saved plane, the entrance as 0 and negative from the end.
        convention : {'optics', 'fft'}, optional
            As for :meth:`fields`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_modes, n_pol, n_omega)``, complex.

        Raises
        ------
        OutputError
            If the index is out of range, or that plane kept no field, naming the
            run's ``fields``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.output import MemoryOutput
        >>> from solprop.output.schema import SavedPlane, _example_metadata
        >>> handler = MemoryOutput()
        >>> handler.open(_example_metadata())
        >>> n_omega = handler.metadata.grid.n_omega
        >>> handler.save(SavedPlane(z=0.0, field=np.ones((1, 1, n_omega), complex)))
        >>> handler.plane_field(-1).shape
        (1, 1, 2560)
        """
        plane = index + self._n_saved if index < 0 else index
        if not 0 <= plane < self._n_saved:
            msg = (
                f"plane {index} is out of range for a run of {self._n_saved} saved "
                "planes"
            )
            raise OutputError(msg)
        if not self._retention.keeps(plane):
            msg = (
                f"saved plane {plane} kept no field: this run was saved with "
                f"{self._retention.describe()}"
            )
            raise OutputError(msg)
        slot = (
            plane
            if self._retention.fields == "all"
            else self._retention.field_planes.index(plane)
        )
        stored = self._read_field(slot)
        if convention == "fft":
            kind = self._require_metadata().grid.kind
            return fourier.spectrum_to_internal(stored, kind=kind)
        return stored

    def _read_field(self, slot: int) -> np.ndarray:
        """Return one stored field by its row in the field storage.

        Parameters
        ----------
        slot : int
            The row: the plane's index in an ``"all"`` run, its position among the
            kept planes otherwise.

        Returns
        -------
        numpy.ndarray
            The stored field, optics convention.
        """
        return self._read("", "field")[slot]

    def __contains__(self, key: object) -> bool:
        """Return whether ``self[key]`` would return an array.

        Answers for exactly the keys :meth:`__getitem__` accepts, and never raises:
        without it Python's fallback would iterate the handler with integer keys.

        Parameters
        ----------
        key : object
            A candidate key; anything but a string of one of :meth:`__getitem__`'s
            forms is not held.

        Returns
        -------
        bool
            ``True`` when the handler holds the array; ``False`` for an unknown form,
            a name not stored, nothing saved yet or a closed file.
        """
        if not isinstance(key, str):
            return False
        # Asking the reader is the one way to agree with it exactly; the cost is one
        # read of the array, which a membership test is followed by anyway.
        try:
            self[key]
        except OutputError:
            return False
        return True

    def _check_layout(
        self,
        plane: SavedPlane,
        stored: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    ) -> None:
        """Adopt the layout from the first plane, and refuse a later one that differs.

        Parameters
        ----------
        plane : SavedPlane
            The plane being appended.
        stored : numpy.ndarray or None
            Its field in the optics convention; ``None`` on a field-free plane.

        Raises
        ------
        OutputError
            If the plane carries a field where the run keeps none or none where it
            keeps one, naming the run's ``fields``, or the field's shape, the
            statistic names or the presence of a noise phase differ from the layout
            already fixed.
        """
        index = self._n_saved
        retention = self._retention
        if retention.keeps(index) != (stored is not None):
            has = "carries no field" if stored is None else "carries a field"
            msg = (
                f"plane {index} {has}, and the run was saved with "
                f"{retention.describe()}; a run keeps its fields by one rule, declared "
                "before its first plane (retain)"
            )
            raise OutputError(msg)
        layout = PlaneLayout(
            field_shape=self._field_shape if stored is None else stored.shape,
            stat_names=tuple(sorted(plane.stats)),
            has_noise_phase=plane.noise_phase is not None,
            step_stat_names=plane.step_stats.names,
            retention=retention,
        )
        if self._layout is None:
            self._layout = layout
            return
        held = self._layout
        # The entrance plane carries no rows, so the run's row-value names are learnt
        # from the first plane that has any.
        if not held.step_stat_names and layout.step_stat_names:
            held = dataclasses.replace(held, step_stat_names=layout.step_stat_names)
            self._layout = held
        # Whether a noise phase is stored beside the field is learnt from the first
        # plane that kept its field: a field-free plane carries none either way.
        if stored is not None and not self._fields_stored():
            held = dataclasses.replace(
                held,
                field_shape=stored.shape,
                has_noise_phase=layout.has_noise_phase,
            )
            self._layout = held
        if (
            stored is not None
            and held.field_shape
            and layout.field_shape != (held.field_shape)
        ):
            msg = (
                f"plane {index} has field shape {layout.field_shape}, but the run "
                f"stores {held.field_shape}"
            )
            raise OutputError(msg)
        if layout.stat_names != held.stat_names:
            msg = (
                f"plane {index} reports statistics {list(layout.stat_names)}, but the "
                f"run stores {list(held.stat_names)}"
            )
            raise OutputError(msg)
        if stored is not None and layout.has_noise_phase is not held.has_noise_phase:
            has = "carries" if layout.has_noise_phase else "does not carry"
            msg = f"plane {index} {has} a noise phase, unlike those before it"
            raise OutputError(msg)
        # A plane may legitimately carry no rows -- the entrance plane never does, and
        # at a period above one a short plane need not -- so the names are compared
        # only where there are rows to name.
        if layout.step_stat_names and layout.step_stat_names != held.step_stat_names:
            msg = (
                f"plane {index} reports step statistics "
                f"{list(layout.step_stat_names)}, but the run stores "
                f"{list(held.step_stat_names)}"
            )
            raise OutputError(msg)

    def _fields_stored(self) -> int:
        """Return how many of the planes saved so far kept their field.

        Returns
        -------
        int
            The rows the field storage holds, which is also the next field's row.
        """
        if self._retention.fields == "all":
            return self._n_saved
        return sum(1 for index in self._retention.field_planes if index < self._n_saved)

    def _check_usable(self) -> None:
        """Refuse to write to a closed handler.

        Raises
        ------
        OutputError
            If :meth:`close` has already run.
        """
        if self._closed:
            msg = "this output handler is closed; open a new one to store another run"
            raise OutputError(msg)

    def _require_metadata(self) -> RunMetadata:
        """Return the run's metadata, refusing to work without it.

        Returns
        -------
        RunMetadata
            The record given to :meth:`open`.

        Raises
        ------
        OutputError
            If :meth:`open` has not been called.
        """
        if self._metadata is None:
            msg = "call open(metadata) before saving or reading a plane"
            raise OutputError(msg)
        return self._metadata

    @abstractmethod
    def _write_metadata(self, metadata: RunMetadata) -> None:
        """Commit the run's metadata to the backing store.

        Parameters
        ----------
        metadata : RunMetadata
            The record to store.
        """
        ...

    @abstractmethod
    def _append(
        self,
        plane: SavedPlane,
        stored: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    ) -> None:
        """Append one plane to the backing store.

        Parameters
        ----------
        plane : SavedPlane
            The plane, whose field is still in the internal convention.
        stored : numpy.ndarray or None
            That field converted to the optics convention, which is what to store;
            ``None`` on a field-free plane, which stores no field.
        """
        ...

    @abstractmethod
    def _read(self, group: str, name: str) -> np.ndarray:
        """Return one stored array.

        Parameters
        ----------
        group : str
            ``''`` for a top-level array, else ``'stats'`` or ``'step_history'``.
        name : str
            Name within the group.

        Returns
        -------
        numpy.ndarray
            The stored array.
        """
        ...

    @abstractmethod
    def _release(self) -> None:
        """Release whatever the store holds open. Called once, by :meth:`close`."""
        ...


def _split_key(key: str) -> tuple[str, str]:
    """Split an accessor key into its group and its name.

    Parameters
    ----------
    key : str
        ``'field'``, ``'z'``, ``'noise_phase'``, ``'field_planes'``, ``'stats/<name>'``,
        ``'step_history/<name>'`` or ``'step_stats/<name>'``.

    Returns
    -------
    tuple of str
        ``(group, name)``, the group empty for a top-level array.

    Raises
    ------
    OutputError
        If the key has none of those forms.
    """
    if key in ("field", "z", "noise_phase", "field_planes"):
        return ("", key)
    group, _, name = key.partition("/")
    if group in ("stats", "step_history", "step_stats") and name:
        return (group, name)
    msg = (
        f"unknown key {key!r}; a handler holds 'field', 'z', 'noise_phase', "
        "'field_planes', 'stats/<name>', 'step_history/<name>' and 'step_stats/<name>'"
    )
    raise OutputError(msg)
