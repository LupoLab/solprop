# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Serialisation of configuration objects into the JSON a run record stores.

:func:`serialise` turns any configuration object into JSON primitives, dropping nothing:
what it cannot reduce it records by ``repr`` and marks irreproducible. A large numeric
array becomes a reference to its bytes (:data:`RECORD_ARRAY_LIMIT`), and a large array
input is recorded by its :class:`ArrayDigest` or by a :class:`StatedInput`
(:func:`recorded_input`). The private helpers here read a stored record back
(``_rebuild``, ``_resolved``) for the record types of :mod:`solprop.output.records` and
:mod:`solprop.output.metadata`.

The manual page is ``docs/guide/results/provenance.md``.
"""

import dataclasses
import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from solprop._identity import canonical_json
from solprop.exceptions import OutputError
from solprop.modes.material import JsonValue

__all__ = [
    "INLINE_ARRAY_LIMIT",
    "RECORD_ARRAY_LIMIT",
    "ArrayDigest",
    "JsonValue",
    "StatedInput",
    "recorded_input",
    "serialise",
]

logger = logging.getLogger(__name__)

# JsonValue is imported above and re-exported here, where the file format is
# owned. It is defined in solprop.modes.material because the record types import
# solprop.modes.protocol: defining it here and importing it there would close a
# cycle through the output layer.

RECORD_ARRAY_LIMIT = 4096
"""The most elements a numeric array of a run record holds as JSON (#309).

A larger one -- a per-node weight, a flux plan's coefficients -- is recorded by a
reference holding its digest, dtype and shape, and its bytes are stored beside the
record, named by that digest
(:meth:`~solprop.output.metadata.RunMetadata.record_arrays`). So the hash takes the
array's bytes rather than its text, which costs milliseconds rather than seconds, and
the array stays readable. The limit keeps a grid's axes in the text, for a person
reading it.
"""

_RECORD_ARRAY = "__record_array__"
"""The key of a reference to an array stored beside the record."""

INLINE_ARRAY_LIMIT = 2**20
"""The most elements an array input is recorded with inline (SEP 22, Q4).

A larger array -- a gigabyte spatial launch -- is recorded by an :class:`ArrayDigest`
of its bytes, dtype and shape instead, which costs seconds of hashing per input rather
than tens of gigabytes of JSON, and still changes with any one sample. A caller may
state its identity instead, with a :class:`StatedInput`.
"""


def _tag_unreproducible(value: object, key: str) -> dict[str, JsonValue]:
    """Return the record of an object that cannot be rebuilt from the file.

    - ``key``: tag naming why it is irreducible, ``'__callable__'`` or ``'__repr__'``.
    - Returns: a record holding the object's ``repr`` and ``reproducible: False``.
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

    - Returns: ``{'__class__': 'module.QualName', <field>: <serialised value>, ...}``.
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

    - Returns: ``{'__enum__': 'module.QualName', 'value': <serialised value>}``.
    """
    kind = type(value)
    return {
        "__enum__": f"{kind.__module__}.{kind.__qualname__}",
        "value": serialise(value.value),
    }


class _ArrayReference(dict):
    """A reference to a record array, in its JSON form, carrying the array it names.

    A serialised record is JSON-ready, so a reference is a plain mapping to :mod:`json`,
    and the array travels with it until the record is stored
    (:meth:`~solprop.output.metadata.RunMetadata.record_arrays`).

    Attributes
    ----------
    array : numpy.ndarray
        The array.
    """

    def __init__(self, values: np.ndarray) -> None:
        """Digest the array and hold it."""
        digest = ArrayDigest.of(values)
        super().__init__(
            {
                _RECORD_ARRAY: _array_name(digest),
                "dtype": digest.dtype,
                "shape": list(digest.shape),
            }
        )
        self.array = values


def _serialise_numpy(value: np.ndarray | np.generic) -> JsonValue:
    """Return a NumPy array as a nested list with its dtype, or a scalar as a scalar.

    A complex array is split into its real and imaginary parts, because JSON has no
    complex number and a nested list of them would not survive a round trip. A numeric
    array of more than :data:`RECORD_ARRAY_LIMIT` elements is a reference instead.

    - Returns: a ``__ndarray__`` record for an array, and the plain Python value of a
      scalar.
    """
    if isinstance(value, np.generic):
        return serialise(value.item())
    if value.size > RECORD_ARRAY_LIMIT and value.dtype.kind in "biufc":
        return _ArrayReference(value)
    return _inline_array(value)


def _inline_array(value: np.ndarray) -> dict[str, JsonValue]:
    """Return an array as a ``__ndarray__`` record, whatever its size.

    - Returns: its dtype, its shape and its values.
    """
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


def _inlined(value: JsonValue) -> JsonValue:
    """Return a serialised structure with each array reference's values in its text.

    For text with no store beside it, which has to stand alone: only a run record keeps
    its arrays beside its text
    (:meth:`~solprop.output.metadata.RunMetadata.record_arrays`). A reference already
    read back from text carries no array and stays as it is.
    """
    if isinstance(value, _ArrayReference):
        return _inline_array(value.array)
    if isinstance(value, dict):
        return {key: _inlined(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_inlined(item) for item in value]
    return value


def _serialise_sequence(value: object) -> list[JsonValue]:
    """Return a sequence or a set as a list, ordering set members deterministically.

    A set's members are sorted by their canonical JSON text, so that the configuration
    hash does not depend on set iteration order.
    """
    # `object` is not declared iterable; the caller has already checked that this is
    # a set or a sequence.
    members = [serialise(item) for item in value]  # type: ignore[union-attr] - checked by the caller
    if isinstance(value, set | frozenset):
        return sorted(members, key=canonical_json)
    return members


def _serialise_object(value: object) -> JsonValue:
    """Return a structured or opaque object as a JSON-safe structure."""
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
    shape, or above :data:`RECORD_ARRAY_LIMIT` elements a reference to their bytes,
    cubic splines their knots, coefficients, axis and extrapolation setting,
    complex numbers a real/imaginary pair, and mappings and sequences are
    recursed. Anything that cannot be reduced -- a function, a lambda, an object with
    no structure -- is recorded as its ``repr`` and marked ``reproducible: False``,
    because a configuration that is silently incomplete is worse than one that says
    where it stops.

    The function is idempotent: applying it to its own output returns that output, so a
    round-tripped :class:`~solprop.output.metadata.RunMetadata` hashes exactly as the
    original does.

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
    >>> from solprop.output.serialisation import serialise
    >>> serialise({"radius": 125e-6})
    {'radius': 0.000125}
    >>> serialise(np.array([1.0, 2.0]))["dtype"]
    'float64'
    >>> serialise(abs)["reproducible"]
    False
    """
    if value is None or isinstance(value, bool | _ArrayReference):
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


_DIGEST_BLOCK = 2**16
"""Elements a digest of an array not held in its order reads at a time."""


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
    >>> from solprop.output.serialisation import ArrayDigest
    >>> digest = ArrayDigest.of(np.zeros((2, 3)))
    >>> digest.dtype, digest.shape, len(digest.sha256)
    ('float64', (2, 3), 64)
    """

    sha256: str
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def of(cls, values: np.ndarray) -> "ArrayDigest":
        """Return the digest of an array's contents.

        Parameters
        ----------
        values : numpy.ndarray
            The array; read once, in C order and little-endian byte order, a block at
            a time unless it is held that way already, so that a broadcast input never
            becomes a whole volume.

        Returns
        -------
        ArrayDigest
            The digest, the dtype and the shape.
        """
        array = np.asarray(values)
        # One byte order, so that the same numbers digest alike on every platform.
        little = array.dtype.newbyteorder("<")
        digest = hashlib.sha256()
        if array.size == 0:
            pass
        elif array.flags.c_contiguous and array.dtype == little:
            digest.update(memoryview(array).cast("B"))
        else:
            # NumPy's stubs leave "contig" out of the flags they list.
            for block in np.nditer(  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
                array,
                flags=["external_loop", "buffered"],
                op_flags=["readonly", "contig"],  # pyright: ignore[reportArgumentType]
                op_dtypes=[little],
                casting="equiv",
                order="C",
                buffersize=_DIGEST_BLOCK,
            ):
                digest.update(np.asarray(block).tobytes())
        return cls(
            sha256=digest.hexdigest(),
            dtype=str(little),
            shape=tuple(int(size) for size in array.shape),
        )


def _array_name(digest: ArrayDigest) -> str:
    """Return the name a record array is stored under.

    The layout is part of the name because the digest covers only the bytes: arrays of
    equal bytes and different dtypes or shapes are different arrays, and one must not
    take the other's place in the store.

    - ``digest``: the array's digest, dtype and shape.
    """
    shape = "x".join(str(size) for size in digest.shape)
    return f"{digest.sha256}-{digest.dtype}-{shape}"


@dataclass(frozen=True)
class StatedInput:
    """An array input recorded by the identity its caller states, its bytes unread.

    What a run record holds under ``input_identity``
    (:meth:`solprop.driver.simulation.Simulation.with_input`): for a caller that
    already identifies its input, such as by a scan's delay and its own configuration,
    and has no use for a digest of it. The identity is taken on trust: unlike an
    :class:`ArrayDigest`, two arrays that differ record the same thing if they are
    stated alike, so a resume and a scan's cache accept one for the other.

    Attributes
    ----------
    identity : str
        What the caller says the input is.
    dtype : str
        The array's dtype, as NumPy names it (``"complex128"``), little-endian as an
        :class:`ArrayDigest` records it.
    shape : tuple of int
        The array's shape.

    Examples
    --------
    >>> import numpy as np
    >>> from solprop.output.serialisation import StatedInput
    >>> StatedInput.of(np.zeros((2, 3), complex), "delay 12 fs")
    StatedInput(identity='delay 12 fs', dtype='complex128', shape=(2, 3))
    """

    identity: str
    dtype: str
    shape: tuple[int, ...]

    @classmethod
    def of(cls, values: np.ndarray, identity: str) -> "StatedInput":
        """Return the record of an array input under a stated identity.

        Parameters
        ----------
        values : numpy.ndarray
            The array, of which only the dtype and the shape are read.
        identity : str
            What the caller says it is.

        Returns
        -------
        StatedInput
            The identity, the dtype and the shape.
        """
        return cls(
            identity=identity,
            dtype=str(values.dtype.newbyteorder("<")),
            shape=tuple(int(size) for size in values.shape),
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
    >>> from solprop.output.serialisation import recorded_input
    >>> recorded_input(np.zeros(4)).shape
    (4,)
    >>> type(recorded_input(np.zeros(2**20 + 1))).__name__
    'ArrayDigest'
    """
    if isinstance(value, np.ndarray) and not _recorded_inline(value):
        return ArrayDigest.of(value)
    return value


def _recorded_inline(values: np.ndarray) -> bool:
    """Return whether a run record carries an array input itself, not its digest.

    - Returns: True for at most :data:`INLINE_ARRAY_LIMIT` elements.
    """
    return values.size <= INLINE_ARRAY_LIMIT


def _decode_array(record: Mapping[str, object]) -> np.ndarray:
    """Return the array a ``__ndarray__`` record describes.

    The nested list :meth:`numpy.ndarray.tolist` produced already carries the shape, so
    only the dtype has to be restored, and the imaginary part rejoined when there is
    one.

    - Returns: the array, with its original dtype and shape.
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


def _decode(value: object) -> object:
    """Return a serialised value with an array field rebuilt.

    An array is the only form a record type holds that JSON cannot carry directly, so
    it is the only one rebuilt; a nested configuration dictionary is left as plain
    JSON, because the physics objects it describes are not reconstructible in general
    (a callable was recorded by ``repr``).

    - Returns: the rebuilt value, or ``value`` itself when there is nothing to rebuild.
    """
    if isinstance(value, dict) and "__ndarray__" in value:
        return _decode_array(value)
    return value


def _resolved(
    value: JsonValue,
    arrays: Mapping[str, object],
    found: dict[str, np.ndarray],
) -> object:
    """Return a stored document with each array reference replaced by its array."""
    if isinstance(value, list):
        return [_resolved(item, arrays, found) for item in value]
    if not isinstance(value, dict):
        return value
    if _RECORD_ARRAY not in value:
        return {key: _resolved(item, arrays, found) for key, item in value.items()}
    name = str(value[_RECORD_ARRAY])
    if name not in found:
        if name not in arrays:
            msg = (
                f"stored metadata names the array {name}, which is not beside it: "
                "read a record through the file that holds it"
            )
            raise OutputError(msg)
        array = np.asarray(arrays[name])
        if _array_name(ArrayDigest.of(array)) != name:
            msg = f"the array stored as {name} does not match that name"
            raise OutputError(msg)
        found[name] = array
    return found[name]


def _dotted(path: Sequence[str | int]) -> str:
    """Return a path into a stored record as a refusal writes it."""
    text = ""
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else f".{part}" if text else part
    return text


def _stored_entry(
    document: Mapping[str, object],
    name: str,
    path: tuple[str, ...],
    *,
    default: object = None,
    required: bool = False,
) -> object:
    """Return one entry of a stored record, reading an absent one as `default`."""
    if name in document:
        return document[name]
    if required:
        msg = f"stored metadata lacks {_dotted((*path, name))}"
        raise OutputError(msg)
    return default


def _rebuild[T](
    build: type[T],
    record: Mapping[str, object],
    path: tuple[str | int, ...] = (),
) -> T:
    """Return a record instance from the dictionary :func:`serialise` made of it.

    The instance is for reading, so a file another build of the same output schema wrote
    stays readable: a field the class no longer has is dropped, and an absent field
    takes the field's default. A resume never compares the rebuilt instance: it compares
    the stored text (:meth:`~solprop.output.metadata.RunMetadata.configuration_hash`).
    """
    fields = {entry.name: entry for entry in dataclasses.fields(build) if entry.init}  # pyright: ignore[reportArgumentType] - a record dataclass
    arguments = {name: _decode(item) for name, item in record.items() if name in fields}
    for name, entry in fields.items():
        if (
            name not in arguments
            and entry.default is dataclasses.MISSING
            and entry.default_factory is dataclasses.MISSING
        ):
            msg = (
                f"stored metadata lacks {_dotted((*path, name))}, which this build "
                "requires"
            )
            raise OutputError(msg)
    return build(**arguments)


def _optional_text(value: JsonValue) -> str | None:
    """Return a stored optional string, ``None`` when null.

    - Returns: the string, or ``None``.
    """
    return None if value is None else str(value)
