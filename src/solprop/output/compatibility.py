# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a resume compares: two run records, path by path (SEP 42).

A resume continues a stored run when the stored and requested records agree on what
matters to the run, not when their configuration hashes are equal. This module holds
the rules, as data, and the walk that applies them:

- the **class table** :data:`CLASS_TABLE` sorts every recorded path into one of four
  classes: ``layout`` (what gives the stored arrays their meaning; it must agree, with
  no override), ``compared`` (physics and launch parameters, the default for any path
  nobody classified; it must agree unless the user forces the resume), ``numerics``
  (how accurately the run steps; it may change while there is work left, and is
  recorded) and ``ignored`` (provenance, and launch realisations whose parameters are
  compared);
- the **neutral table** :data:`NEUTRAL_VALUES` gives, for a path one record lacks, the
  value that reproduces the code before the path existed. A path absent from one record
  with no entry is a difference: a field's default is deliberately not consulted, so a
  new field that changes what runs compute cannot agree with itself through it;
- :data:`CLASS_RENAMES` maps a class moved between modules to its new name.

Numbers agree per element within :data:`RECORD_RTOL` of the larger of the two, because
the elements of one array may be different quantities (``[β2, β3, β4]`` span 18
orders of magnitude). The paths the table marks *tail* add a floor of the same
tolerance times the array's largest element, for arrays with tails of rounding; the
paths it marks *exact* take no tolerance, because the driver does exact arithmetic on
them.

Before the walk, both records are normalised: optmat model records are keyed by
``material:kind:digest`` and solids by ``material:axis:digest``, so that a renamed
model with the same numbers is the same model, and a subtree whose ``__class__`` is an
optmat class is reduced to a marker, because the manifest records its digest.

The manual page is ``docs/guide/results/resuming.md``; the design is SEP 42.
"""

import dataclasses
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from enum import StrEnum
from typing import Final

import numpy as np

__all__ = [
    "CLASS_RENAMES",
    "CLASS_TABLE",
    "NEUTRAL_VALUES",
    "RECORD_RTOL",
    "Neutral",
    "PathChange",
    "PathClass",
    "RecordComparison",
    "Row",
    "classify",
    "compare_records",
    "dotted",
    "neutral_value",
]

type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)
type Component = str | int
type RecordPath = tuple[Component, ...]

RECORD_RTOL: Final = 1e-11
"""How far apart two recorded numbers may be and still agree, relative to the larger.

Chosen from the largest departure a change of libm, BLAS or compiler makes among a
run record's numbers: the ionisation threshold, found by bisection to a relative bracket
of 1e-12 (SEP 42 section 4). It is far below any stepper tolerance in use and any change
a user makes on purpose. The record-stability tests use the same constant.
"""

_OPTMAT_MARKER: Final = "<optmat object>"
"""What an optmat object's subtree is reduced to before the walk."""

_SHOWN: Final = 4
"""How many differences a summary names in each part of the record."""


class PathClass(StrEnum):
    """What a difference at a path means for a resume.

    Attributes
    ----------
    LAYOUT
        The stored arrays' meaning: it must agree, and ``resume="force"`` does not
        change that.
    COMPARED
        Physics and launch parameters: it must agree unless the resume is forced.
    NUMERICS
        How accurately the run steps: it may change while there is work left.
    IGNORED
        Provenance, and realisations whose parameters are compared.
    """

    LAYOUT = "layout"
    COMPARED = "compared"
    NUMERICS = "numerics"
    IGNORED = "ignored"


class _Any:
    """One path component of any value, the ``*`` of a pattern."""

    def __repr__(self) -> str:
        return "*"


class _AnyDepth:
    """Any number of path components, the ``**`` of a pattern."""

    def __repr__(self) -> str:
        return "**"


ANY: Final = _Any()
ANY_DEPTH: Final = _AnyDepth()

type PatternComponent = str | frozenset[str] | _Any | _AnyDepth


def _pattern(text: str) -> tuple[PatternComponent, ...]:
    """Return a pattern from its written form.

    Components are separated by ``.``; ``*`` matches one component (a list index
    included), ``**`` any number, and ``{a,b}`` lists alternatives. Patterns are
    written by solprop and never name a key holding a dot; the paths they are matched
    against are tuples, so a stored key with a dot in it is matched whole.

    Parameters
    ----------
    text : str
        The written pattern, such as ``"materials.records.*.{name,status}"``.

    Returns
    -------
    tuple
        Its components.
    """
    components: list[PatternComponent] = []
    for part in text.split("."):
        if part == "*":
            components.append(ANY)
        elif part == "**":
            components.append(ANY_DEPTH)
        elif part.startswith("{") and part.endswith("}"):
            components.append(frozenset(part[1:-1].split(",")))
        else:
            components.append(part)
    return tuple(components)


def _component_matches(pattern: PatternComponent, component: Component) -> bool:
    """Say whether one pattern component matches one path component.

    Parameters
    ----------
    pattern : str, frozenset, or the one-component wildcard
        The pattern's component.
    component : str or int
        The path's.

    Returns
    -------
    bool
        Whether they match.
    """
    if isinstance(pattern, _Any):
        return True
    if isinstance(component, int):
        return False
    if isinstance(pattern, frozenset):
        return component in pattern
    return pattern == component


def _prefix_matches(
    pattern: Sequence[PatternComponent], path: Sequence[Component]
) -> bool:
    """Say whether a pattern matches a path or one of its ancestors.

    Parameters
    ----------
    pattern : sequence
        The pattern's components.
    path : sequence
        The path's.

    Returns
    -------
    bool
        Whether the whole pattern matches the first components of `path`.
    """
    if not pattern:
        return True
    head, rest = pattern[0], pattern[1:]
    if isinstance(head, _AnyDepth):
        return any(_prefix_matches(rest, path[skip:]) for skip in range(len(path) + 1))
    if not path:
        return False
    return _component_matches(head, path[0]) and _prefix_matches(rest, path[1:])


def _full_matches(
    pattern: Sequence[PatternComponent], path: Sequence[Component]
) -> bool:
    """Say whether a pattern matches a path exactly, not only an ancestor of it.

    Parameters
    ----------
    pattern : sequence
        The pattern's components.
    path : sequence
        The path's.

    Returns
    -------
    bool
        Whether the pattern consumes the whole path.
    """
    if not pattern:
        return not path
    head, rest = pattern[0], pattern[1:]
    if isinstance(head, _AnyDepth):
        return any(_full_matches(rest, path[skip:]) for skip in range(len(path) + 1))
    if not path:
        return False
    return _component_matches(head, path[0]) and _full_matches(rest, path[1:])


def _lookup(record: JsonValue, path: Sequence[Component]) -> tuple[bool, JsonValue]:
    """Return whether a record holds a path, and the value there.

    Parameters
    ----------
    record : object
        A record, as JSON.
    path : sequence
        The path.

    Returns
    -------
    tuple
        ``(True, value)``, or ``(False, None)`` when the record lacks the path.
    """
    value = record
    for component in path:
        if isinstance(component, int):
            if not isinstance(value, list) or component >= len(value):
                return False, None
            value = value[component]
        else:
            if not isinstance(value, dict) or component not in value:
                return False, None
            value = value[component]
    return True, value


def _has_pulse(record: JsonValue) -> bool:
    """Say whether a record's launch was built from a pulse it records.

    Parameters
    ----------
    record : object
        A normalised record.

    Returns
    -------
    bool
        Whether ``configuration.pulse`` is present.
    """
    return _lookup(record, ("configuration", "pulse"))[0]


def _seeded_input_noise(record: JsonValue) -> bool:
    """Say whether a record's noise is input noise drawn from a recorded seed.

    Parameters
    ----------
    record : object
        A normalised record.

    Returns
    -------
    bool
        Whether ``noise.seed`` is set and ``noise.model`` is ``"input"``.
    """
    _, seed = _lookup(record, ("noise", "seed"))
    _, model = _lookup(record, ("noise", "model"))
    return seed is not None and model == "input"


@dataclasses.dataclass(frozen=True)
class Row:
    """One row of the class table.

    Attributes
    ----------
    pattern : str
        The written pattern (:func:`_pattern`).
    kind : PathClass
        The class of the paths it matches.
    reason : str
        Why, for the reader of the table.
    exact : bool
        Whether numbers here must be equal, with no tolerance.
    tail : bool
        Whether an array here adds a floor of the tolerance times its largest element.
    subtree : bool
        Whether the row covers the paths beneath its own; ``False`` for a row that
        classifies only a difference at the path itself, such as a list's length.
    when : callable or None
        A condition both records must meet for the row to apply.
    """

    pattern: str
    kind: PathClass
    reason: str
    exact: bool = False
    tail: bool = False
    subtree: bool = True
    when: Callable[[JsonValue], bool] | None = None

    @property
    def components(self) -> tuple[PatternComponent, ...]:
        """Return the parsed pattern.

        Returns
        -------
        tuple
            Its components.
        """
        return _pattern(self.pattern)

    def matches(
        self, path: RecordPath, stored: JsonValue, requested: JsonValue
    ) -> bool:
        """Say whether the row decides the class of a path.

        Parameters
        ----------
        path : tuple
            The path.
        stored, requested : object
            The two normalised records, for a conditional row.

        Returns
        -------
        bool
            Whether the row matches the path (or an ancestor, for a subtree row) and
            both records meet its condition.
        """
        components = _PARSED[self.pattern]
        hit = (
            _prefix_matches(components, path)
            if self.subtree
            else _full_matches(components, path)
        )
        if not hit:
            return False
        return self.when is None or (self.when(stored) and self.when(requested))


_LAYOUT = PathClass.LAYOUT
_COMPARED = PathClass.COMPARED
_NUMERICS = PathClass.NUMERICS
_IGNORED = PathClass.IGNORED

CLASS_TABLE: Final[tuple[Row, ...]] = (
    Row("simulation_kind", _LAYOUT, "what kind of run the file holds"),
    Row("power_convention", _LAYOUT, "what the stored amplitudes measure"),
    Row(
        "initial_z",
        _LAYOUT,
        "a resume finds its plane by exact arithmetic on the save planes",
        exact=True,
    ),
    Row(
        "grid.length",
        _LAYOUT,
        "a resume finds its plane by exact arithmetic on the save planes",
        exact=True,
    ),
    Row(
        "configuration.save_planes",
        _LAYOUT,
        "a resume finds its plane by searchsorted and lands on it",
        exact=True,
    ),
    Row("grid", _LAYOUT, "the stored arrays' axes"),
    Row("modes", _LAYOUT, "how many modes the state holds", subtree=False),
    Row(
        "modes.*.label.{family,azimuthal_order,radial_order,parity,handedness}",
        _LAYOUT,
        "which mode each row of the state is; a mode's model and data are compared",
    ),
    Row(
        "spatial.{kind,shape,axes,axis_names,axis_units,spectral_axes,"
        "spectral_axis_names,spectral_axis_units,column_order,physical_field_units,"
        "stored_field_units,spatial_weights,spectral_weights,forward_scale,"
        "inverse_scale}",
        _LAYOUT,
        "the transverse axes and transforms; flux plans and absorber rates are "
        "compared",
    ),
    Row(
        "modal.{basis,n_channels,n_components,channels}",
        _LAYOUT,
        "the basis's structure; the guide, tables and responses are compared",
    ),
    Row(
        "quadrature.{radial_rule,kind,full,requested_n_r,n_r,n_theta,n_nodes,"
        "embedded_rule}",
        _LAYOUT,
        "the rule and its counts, which per-node statistics are shaped by",
    ),
    Row(
        "noise",
        _LAYOUT,
        "a stored state carries input noise, and modified noise adds a source at "
        "every step",
        subtree=False,
    ),
    Row("noise.model", _LAYOUT, "which noise the stored state carries"),
    Row(
        "configuration.precision",
        _LAYOUT,
        "a run is not resumed in the other precision (SEP 42, Q7)",
    ),
    Row(
        "tdse",
        _LAYOUT,
        "compared as text: a snapshot binds to the descriptor",
        exact=True,
    ),
    Row("stepper", _NUMERICS, "how accurately the run steps, not what"),
    Row("configuration.propagator_mode", _NUMERICS, "how the linear step is taken"),
    Row(
        "configuration.initial",
        _IGNORED,
        "realised from a compared pulse on a compared grid; the stored field "
        "supersedes it",
        when=_has_pulse,
    ),
    Row(
        "configuration.initial",
        _COMPARED,
        "an array launch: this is its only record",
        tail=True,
    ),
    Row(
        "noise.spectrum",
        _IGNORED,
        "drawn from a compared model and seed, and carried by the stored field",
        when=_seeded_input_noise,
    ),
    Row(
        "noise.spectrum",
        _COMPARED,
        "a realisation passed in has no other record",
        tail=True,
    ),
    Row(
        "configuration.absorber.{spectral_rate,temporal_rate,temporal_rate_fine}",
        _COMPARED,
        "rates with rounding tails",
        tail=True,
    ),
    Row("spatial.absorber_rate", _COMPARED, "rates with rounding tails", tail=True),
    Row("materials.optmat_version", _IGNORED, "provenance"),
    Row(
        "materials.records.*.{name,status,status_reason,sources,optmat_version}",
        _IGNORED,
        "provenance; the digest identifies the numbers",
    ),
    Row(
        "materials.policies.ionisation.*.{model,status}",
        _IGNORED,
        "a model's name and status, outside the manifest's records",
    ),
    Row(
        "**.cladding.source",
        _IGNORED,
        "a model's name, outside the manifest's records",
    ),
    Row(
        "**.citation",
        _IGNORED,
        "a citation, which may carry a file's hash and the optmat release",
    ),
)
"""The class of every recorded path (SEP 42 section 5.2), in order.

The first row that matches a path decides its class; a path no row matches is
``compared``, so a field nobody has classified is held to agreement and the safe error
is a refusal. A row covers the paths beneath it unless it says otherwise. The table is
data, reviewed like code: a test asserts that every row matches a path in at least one
route's record.
"""

_PARSED: Final[dict[str, tuple[PatternComponent, ...]]] = {
    row.pattern: _pattern(row.pattern) for row in CLASS_TABLE
}


@dataclasses.dataclass(frozen=True)
class Neutral:
    """A recorded path's neutral value: what reproduces the code before it existed.

    Attributes
    ----------
    pattern : str
        The path, as a pattern (:func:`_pattern`).
    value : object
        The value an absent path reads as, in its serialised form.
    release : str
        The release that added the path; the entry is deleted with that release's
        frozen records, after the next minor release (SEP 42, Q3).
    """

    pattern: str
    value: JsonValue
    release: str


NEUTRAL_VALUES: Final[tuple[Neutral, ...]] = ()
"""One entry per recorded path a release declared neutral (SEP 42 section 4).

A path present in one record and absent from the other is a difference unless an entry
here gives its neutral value. Every new recorded field therefore needs a decision, and
the frozen records of ``tests/output/test_record_stability.py`` fail until someone
makes it. A default changed deliberately needs no entry: the earlier record holds the
old value, which the comparison sees as a change of physics.
"""

CLASS_RENAMES: Final[Mapping[str, str]] = {}
"""Classes moved between modules: the old qualified name to the new one.

Applied to ``__class__`` and ``__enum__`` tags of both records before the walk, so that
a move is not a change of physics.
"""

_NEUTRAL_PARSED: Final = tuple(
    (_pattern(entry.pattern), entry) for entry in NEUTRAL_VALUES
)


def neutral_value(path: Sequence[Component]) -> tuple[bool, JsonValue]:
    """Return the neutral value of a recorded path, if a release declared one.

    Parameters
    ----------
    path : sequence
        The path.

    Returns
    -------
    tuple
        ``(True, value)``, or ``(False, None)`` when :data:`NEUTRAL_VALUES` has no
        entry.

    Examples
    --------
    >>> from solprop.output.compatibility import neutral_value
    >>> neutral_value(("stepper", "controller", "no_such_field"))
    (False, None)
    """
    for components, entry in _NEUTRAL_PARSED:
        if _full_matches(components, tuple(path)):
            return True, entry.value
    return False, None


def classify(
    path: Sequence[Component],
    stored: JsonValue = None,
    requested: JsonValue = None,
) -> Row | None:
    """Return the row of the class table that decides a path's class.

    Parameters
    ----------
    path : sequence
        The path.
    stored, requested : object, optional
        The two normalised records, which a conditional row consults.

    Returns
    -------
    Row or None
        The first matching row; ``None`` for a path no row matches, which is
        compared.

    Examples
    --------
    >>> from solprop.output.compatibility import classify
    >>> classify(("stepper", "controller", "rtol")).kind
    <PathClass.NUMERICS: 'numerics'>
    >>> classify(("configuration", "gamma")) is None
    True
    """
    key = tuple(path)
    for row in CLASS_TABLE:
        if row.matches(key, stored, requested):
            return row
    return None


def dotted(path: Sequence[Component]) -> str:
    """Return a path as one readable string, keys joined by dots and indices bracketed.

    Parameters
    ----------
    path : sequence
        The path.

    Returns
    -------
    str
        For example ``"modes[0].label.family"``.

    Examples
    --------
    >>> from solprop.output.compatibility import dotted
    >>> dotted(("modes", 0, "label", "family"))
    'modes[0].label.family'
    """
    text = ""
    for component in path:
        if isinstance(component, int):
            text += f"[{component}]"
        else:
            text += f".{component}" if text else component
    return text


# ------------------------------------------------------------------ normalisation


def _renamed(value: JsonValue) -> JsonValue:
    """Return a record with every moved class tag renamed and optmat objects marked.

    Parameters
    ----------
    value : object
        A record, as JSON.

    Returns
    -------
    object
        A copy, wherever anything changed.
    """
    if isinstance(value, list):
        return [_renamed(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value.get("__class__")
    if isinstance(tag, str) and tag.startswith("optmat."):
        return _OPTMAT_MARKER
    result = {key: _renamed(item) for key, item in value.items()}
    for key in ("__class__", "__enum__"):
        name = result.get(key)
        if isinstance(name, str) and name in CLASS_RENAMES:
            result[key] = CLASS_RENAMES[name]
    return result


def _rekeyed(
    entries: JsonValue, fields: tuple[str, str], *, drop: str | None = None
) -> JsonValue:
    """Return a manifest's keyed entries keyed by two fields and their digest.

    Parameters
    ----------
    entries : object
        The stored mapping of entries, keyed by name.
    fields : tuple of str
        The two identifying fields the new key starts with.
    drop : str or None, optional
        A field each entry loses, because the new key replaces it.

    Returns
    -------
    object
        The entries keyed by ``field:field:digest``; `entries` itself when it is not
        a mapping of mappings.
    """
    if not isinstance(entries, dict):
        return entries
    keyed: dict[str, JsonValue] = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            keyed[name] = entry
            continue
        key = ":".join(str(entry.get(field) or "-") for field in (*fields, "digest"))
        if key in keyed:
            # Two entries with the same numbers under one identity: keep both,
            # distinguished by their stored names.
            key = f"{key}#{name}"
        keyed[key] = (
            {k: v for k, v in entry.items() if k != drop} if drop else dict(entry)
        )
    return keyed


def _normalised(record: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Return a record as the comparison sees it.

    Model records are keyed by ``material:kind:digest`` and solids by
    ``material:axis:digest`` (losing their ``model`` leaf), so that a renamed model with
    the same numbers is the same model; optmat objects are reduced to a marker, because
    the manifest records their digests; moved classes take their new names.

    Parameters
    ----------
    record : collections.abc.Mapping
        The hashed part of a run record.

    Returns
    -------
    dict
        The normalised copy.
    """
    result = _renamed(dict(record))
    assert isinstance(result, dict)
    materials = result.get("materials")
    if isinstance(materials, dict):
        materials = dict(materials)
        materials["records"] = _rekeyed(materials.get("records"), ("material", "kind"))
        if "solids" in materials:
            materials["solids"] = _rekeyed(
                materials["solids"], ("material", "axis"), drop="model"
            )
        result["materials"] = materials
    return result


# ------------------------------------------------------------------- leaf agreement

_TAGS: Final = ("__ndarray__", "__complex__", "__enum__", "__callable__", "__repr__")
_DIGEST_CLASS: Final = "solprop.output.schema.ArrayDigest"


def _is_tagged_leaf(value: JsonValue) -> bool:
    """Say whether a mapping is a leaf: an array, a complex number, a tag or a digest.

    Parameters
    ----------
    value : object
        A record value.

    Returns
    -------
    bool
        Whether the walk compares it whole.
    """
    if not isinstance(value, dict):
        return False
    return any(tag in value for tag in _TAGS) or value.get("__class__") == _DIGEST_CLASS


def _is_number(value: JsonValue) -> bool:
    """Say whether a value is a JSON number (a boolean is not).

    Parameters
    ----------
    value : object
        A record value.

    Returns
    -------
    bool
        Whether it is an int or a float.
    """
    return isinstance(value, int | float) and not isinstance(value, bool)


def _numeric_array(value: JsonValue) -> np.ndarray | None:
    """Return a record value as a numeric array, when it is one.

    A ``__ndarray__`` record of a numeric dtype, a ``__complex__`` scalar, or a nested
    list of numbers.

    Parameters
    ----------
    value : object
        A record value.

    Returns
    -------
    numpy.ndarray or None
        The numbers (complex where the record has an imaginary part), or ``None``.
    """
    if isinstance(value, list):
        return _list_array(value)
    if not isinstance(value, dict):
        return None
    if "__complex__" in value:
        pair = _list_array(value["__complex__"])
        valid = pair is not None and pair.shape == (2,)
        return np.asarray(complex(pair[0], pair[1])) if valid else None  # pyright: ignore[reportOptionalSubscript] - checked by valid
    real = _list_array(value.get("__ndarray__"))
    if real is None or "__imaginary__" not in value:
        return real
    imaginary = _list_array(value["__imaginary__"])
    if imaginary is None or imaginary.shape != real.shape:
        return None
    return real + 1j * imaginary


def _list_array(value: JsonValue) -> np.ndarray | None:
    """Return a nested list of numbers as an array, or ``None`` if it is not one.

    Parameters
    ----------
    value : object
        A record value.

    Returns
    -------
    numpy.ndarray or None
        The numbers, as integers when every one is an integer; ``None`` for anything
        holding a string, a boolean, ``None``, a mapping or a ragged nesting.
    """
    if not isinstance(value, list):
        return None
    try:
        array = np.asarray(value)
    except ValueError, TypeError:
        return None
    if array.dtype.kind not in "iuf":
        return None
    return array


@dataclasses.dataclass(frozen=True)
class PathChange:
    """One difference between a stored and a requested record.

    Attributes
    ----------
    path : tuple
        Where, as mapping keys and list indices.
    kind : PathClass
        Its class.
    stored : object
        The stored value, summarised: a scalar as itself, an array or a list as its
        shape, ``None`` where the stored record lacks the path.
    requested : object
        The requested value, summarised the same way.
    absent : str
        ``"stored"`` or ``"requested"`` when one record lacks the path, else ``""``.
    elements : int
        How many elements differ (1 for a scalar).
    largest : float or None
        The largest absolute difference between numbers, where both are numbers.
    note : str
        A reason the comparison adds, such as a repr that changes between processes.
    """

    path: RecordPath
    kind: PathClass
    stored: object
    requested: object
    absent: str = ""
    elements: int = 1
    largest: float | None = None
    note: str = ""

    def describe(self, *, relative_to: int = 0) -> str:
        """Return the one-line description a refusal or a warning shows.

        Parameters
        ----------
        relative_to : int, optional
            How many leading components to leave out of the path.

        Returns
        -------
        str
            For example ``"controller.rtol 1e-06 -> 1e-05"``.
        """
        name = dotted(self.path[relative_to:]) or dotted(self.path)
        if self.absent:
            side = "the stored record" if self.absent == "stored" else "this run's"
            return f"{name} (absent from {side})"
        if self.elements > 1 or isinstance(self.stored, tuple):
            largest = "" if self.largest is None else f", largest {self.largest:.3g}"
            count = f"{self.elements} elements" if self.elements > 1 else "shape"
            return f"{name} ({count}{largest})"
        return f"{name} {_short(self.stored)} -> {_short(self.requested)}"


def _short(value: object) -> str:
    """Return a value as a refusal shows it.

    Parameters
    ----------
    value : object
        A summarised value.

    Returns
    -------
    str
        Its ``repr``, cut to sixty characters.
    """
    text = repr(value)
    return text if len(text) <= 60 else text[:57] + "..."


def _summarised(value: JsonValue) -> object:
    """Return a record value as a :class:`PathChange` holds it.

    Parameters
    ----------
    value : object
        A record value.

    Returns
    -------
    object
        A scalar as itself, an enumeration as its value, a repr record as its repr,
        and anything else as its shape or kind.
    """
    if isinstance(value, list):
        array = _list_array(value)
        return (len(value),) if array is None else tuple(array.shape)
    if not isinstance(value, dict):
        return value
    tags = [tag for tag in ("__callable__", "__repr__") if tag in value]
    array = _numeric_array(value)
    if "__enum__" in value:
        summary: object = value.get("value")
    elif tags:
        summary = value[tags[0]]
    elif array is not None:
        summary = array.item() if array.ndim == 0 else tuple(array.shape)
    elif value.get("__class__") == _DIGEST_CLASS:
        summary = f"digest {str(value.get('sha256', ''))[:12]}"
    else:
        summary = f"<{value.get('__class__', 'mapping')}>"
    return summary


def _within(stored: np.ndarray, requested: np.ndarray, *, tail: bool) -> np.ndarray:
    """Return where two arrays of one shape agree element by element.

    ``|a - b| <= ε max(|a|, |b|)``, with ``ε`` = :data:`RECORD_RTOL`, complex values by
    modulus; a tail adds ``ε max_j(|a_j|, |b_j|)``. NaN agrees only with NaN in the same
    place, and an infinity only with the same infinity.

    Parameters
    ----------
    stored, requested : numpy.ndarray
        The two arrays.
    tail : bool
        Whether to add the array-scale floor.

    Returns
    -------
    numpy.ndarray
        A boolean array, true where the elements agree.
    """
    one = np.asarray(stored, dtype=np.complex128)
    two = np.asarray(requested, dtype=np.complex128)
    nan_one, nan_two = np.isnan(one), np.isnan(two)
    inf_one, inf_two = np.isinf(one), np.isinf(two)
    special = nan_one | nan_two | inf_one | inf_two
    with np.errstate(invalid="ignore", over="ignore"):
        # Infinities and NaNs compare by identity, below; their arithmetic is moot.
        size = np.maximum(np.abs(one), np.abs(two))
        bound = RECORD_RTOL * size
        if tail:
            finite = size[~special]
            bound = bound + RECORD_RTOL * (finite.max() if finite.size else 0.0)
        agree = np.abs(one - two) <= bound
    identical = (nan_one & nan_two) | (
        (inf_one | inf_two) & ~nan_one & ~nan_two & (one == two)
    )
    return np.where(special, identical, agree)


# --------------------------------------------------------------------- the walk


@dataclasses.dataclass
class _Walk:
    """The state of one comparison: the two records and what it found."""

    stored: dict[str, JsonValue]
    requested: dict[str, JsonValue]
    changes: list[PathChange] = dataclasses.field(default_factory=list)
    neutral: list[RecordPath] = dataclasses.field(default_factory=list)

    def row(self, path: RecordPath) -> Row | None:
        """Return the row deciding a path's class.

        Parameters
        ----------
        path : tuple
            The path.

        Returns
        -------
        Row or None
            The row, or ``None`` for a compared path.
        """
        return classify(path, self.stored, self.requested)

    def differ(self, path: RecordPath, change: dict[str, object]) -> None:
        """Record a difference at a path, with its class.

        Parameters
        ----------
        path : tuple
            The path.
        change : dict
            The :class:`PathChange` fields other than the path and the class.
        """
        row = self.row(path)
        kind = _COMPARED if row is None else row.kind
        if kind is _IGNORED:
            return
        self.changes.append(PathChange(path=path, kind=kind, **change))  # pyright: ignore[reportArgumentType] - the fields of PathChange

    def compare(self, path: RecordPath, one: JsonValue, two: JsonValue) -> None:
        """Compare two values at a path, recording every difference beneath it.

        Parameters
        ----------
        path : tuple
            Where the values sit.
        one, two : object
            The stored and the requested value.
        """
        if _is_tagged_leaf(one) or _is_tagged_leaf(two):
            self.leaf(path, one, two)
            return
        if isinstance(one, dict) and isinstance(two, dict):
            for key in sorted(set(one) | set(two)):
                self.child((*path, key), one, two, key)
            return
        if isinstance(one, list) and isinstance(two, list):
            self.sequence(path, one, two)
            return
        self.leaf(path, one, two)

    def child(
        self,
        path: RecordPath,
        one: dict[str, JsonValue],
        two: dict[str, JsonValue],
        key: str,
    ) -> None:
        """Compare one key of two mappings, reading an absent side as neutral.

        Parameters
        ----------
        path : tuple
            The key's path.
        one, two : dict
            The stored and the requested mapping.
        key : str
            The key.
        """
        if key in one and key in two:
            self.compare(path, one[key], two[key])
            return
        found, value = neutral_value(path)
        if found:
            self.neutral.append(path)
            self.compare(path, one.get(key, value), two.get(key, value))
            return
        present = one[key] if key in one else two[key]
        self.differ(
            path,
            {
                "stored": _summarised(present) if key in one else None,
                "requested": _summarised(present) if key in two else None,
                "absent": "requested" if key in one else "stored",
            },
        )

    def sequence(
        self, path: RecordPath, one: list[JsonValue], two: list[JsonValue]
    ) -> None:
        """Compare two lists: numbers as an array, anything else item by item.

        Parameters
        ----------
        path : tuple
            The lists' path.
        one, two : list
            The stored and the requested list.
        """
        if _list_array(one) is not None and _list_array(two) is not None:
            self.leaf(path, one, two)
            return
        if len(one) != len(two):
            self.differ(
                path,
                {"stored": (len(one),), "requested": (len(two),)},
            )
            return
        for index, (item, other) in enumerate(zip(one, two, strict=True)):
            self.compare((*path, index), item, other)

    def leaf(self, path: RecordPath, one: JsonValue, two: JsonValue) -> None:
        """Compare two leaves: numbers to the tolerance, anything else for equality.

        Parameters
        ----------
        path : tuple
            The leaves' path.
        one, two : object
            The stored and the requested leaf.
        """
        row = self.row(path)
        if row is not None and row.kind is _IGNORED:
            return
        exact = row is not None and row.exact
        tail = row is not None and row.tail
        first, second = _numeric_array(one), _numeric_array(two)
        if _is_number(one):
            first = np.asarray(one)
        if _is_number(two):
            second = np.asarray(two)
        if first is not None and second is not None and not exact:
            self.numbers(path, one, two, first, second, tail=tail)
            return
        if _canonical(one) == _canonical(two) or (
            exact and _is_number(one) and _is_number(two) and one == two
        ):
            return
        note = ""
        if any(
            isinstance(value, dict) and ("__callable__" in value or "__repr__" in value)
            for value in (one, two)
        ):
            note = (
                "recorded by its repr, which changes between processes: pass an "
                "object that carries its data"
            )
        self.differ(
            path,
            {
                "stored": _summarised(one),
                "requested": _summarised(two),
                "note": note,
            },
        )

    def numbers(
        self,
        path: RecordPath,
        one: JsonValue,
        two: JsonValue,
        first: np.ndarray,
        second: np.ndarray,
        *,
        tail: bool,
    ) -> None:
        """Compare two numeric leaves or arrays element by element.

        Parameters
        ----------
        path : tuple
            Their path.
        one, two : object
            The stored and the requested values, as recorded.
        first, second : numpy.ndarray
            The same, as arrays.
        tail : bool
            Whether the row adds the array-scale floor.
        """
        dtypes = tuple(
            value.get("dtype") if isinstance(value, dict) else None
            for value in (one, two)
        )
        if first.shape != second.shape or dtypes[0] != dtypes[1]:
            self.differ(
                path, {"stored": _summarised(one), "requested": _summarised(two)}
            )
            return
        if first.dtype.kind in "iu" and second.dtype.kind in "iu":
            # Integers count things; they agree when equal.
            agree = first == second
        else:
            agree = _within(first, second, tail=tail)
        if bool(np.all(agree)):
            return
        with np.errstate(invalid="ignore"):
            gaps = np.abs(
                np.asarray(first, dtype=np.complex128)
                - np.asarray(second, dtype=np.complex128)
            )[~agree]
        finite = gaps[np.isfinite(gaps)]
        largest = float(finite.max()) if finite.size else None
        self.differ(
            path,
            {
                "stored": _summarised(one),
                "requested": _summarised(two),
                "elements": int(np.count_nonzero(~agree)),
                "largest": largest,
            },
        )


def _canonical(value: JsonValue) -> str:
    """Return the one textual form of a leaf, for equality.

    Parameters
    ----------
    value : object
        A leaf.

    Returns
    -------
    str
        Its compact, key-sorted JSON.
    """
    import json  # noqa: PLC0415 - only for the rare non-numeric leaf

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


# ------------------------------------------------------------------ the comparison


@dataclasses.dataclass(frozen=True)
class RecordComparison:
    """What separates a stored run record from a requested one (SEP 42 section 5.1).

    Attributes
    ----------
    layout : tuple of PathChange
        Differences in what gives the stored arrays their meaning; no resume continues
        across them.
    refused : tuple of PathChange
        Differences in compared paths (physics and launch parameters); a resume
        continues across them only when it is forced.
    numerics : tuple of PathChange
        Differences in how accurately the run steps; a resume continues across them
        while there is work left, and records them.
    neutral : tuple of tuple
        The paths one record lacked and read as their neutral value.

    Examples
    --------
    >>> from solprop.output.compatibility import compare_records
    >>> stored = {"stepper": {"rtol": 1e-6}, "configuration": {"gamma": 0.11}}
    >>> requested = {"stepper": {"rtol": 1e-5}, "configuration": {"gamma": 0.11}}
    >>> comparison = compare_records(stored, requested)
    >>> comparison.continues, [change.describe() for change in comparison.numerics]
    (True, ['stepper.rtol 1e-06 -> 1e-05'])
    """

    layout: tuple[PathChange, ...] = ()
    refused: tuple[PathChange, ...] = ()
    numerics: tuple[PathChange, ...] = ()
    neutral: tuple[RecordPath, ...] = ()

    @property
    def continues(self) -> bool:
        """Return whether ``resume=True`` may continue across these differences.

        Returns
        -------
        bool
            True when neither the layout nor a compared path differs.
        """
        return not self.layout and not self.refused

    @property
    def identical(self) -> bool:
        """Return whether nothing but rounding, neutral values and ignored paths differ.

        Returns
        -------
        bool
            True when no class has a difference.
        """
        return self.continues and not self.numerics

    def numerics_record(self) -> dict[str, list[object]]:
        """Return the changed numerics as an execution record holds them (§5.3).

        Returns
        -------
        dict
            ``{dotted path: [stored, requested]}``; a non-finite number is written as
            text, because an execution record refuses one, and an array as its shape.
        """
        return {
            dotted(change.path): [
                _plain(change.stored),
                _plain(change.requested),
            ]
            for change in self.numerics
        }

    def summary(self, *, shown: int = _SHOWN) -> str:
        """Say how the records differ, briefly enough to read in a refusal or warning.

        Each class that differs has a line; within it every part of the record that
        changed is named, with its first `shown` entries and a count of the rest, so
        that no cause hides behind another part's many entries. A model whose numbers
        changed is named by its material, kind and stored name.

        Parameters
        ----------
        shown : int, optional
            How many entries to name in each part.

        Returns
        -------
        str
            One line per class that differs; empty when nothing does.
        """
        lines = []
        for title, changes in (
            ("the layout differs", self.layout),
            ("these compared values differ", self.refused),
            ("the numerics changed", self.numerics),
        ):
            if changes:
                lines.append(f"{title}: {_grouped(changes, shown)}")
        notes = sorted(
            {
                f"{dotted(change.path)} is {change.note}"
                for change in (*self.layout, *self.refused)
                if change.note
            }
        )
        lines.extend(notes)
        return "\n".join(lines)


def _plain(value: object) -> object:
    """Return a summarised value as JSON an execution record accepts.

    Parameters
    ----------
    value : object
        A :class:`PathChange` value.

    Returns
    -------
    object
        Finite numbers, strings, booleans and ``None`` as themselves; a non-finite
        number or anything else as text.
    """
    if value is None or isinstance(value, bool | str | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, tuple):
        return f"shape {list(value)}"  # pyright: ignore[reportUnknownArgumentType] - a shape
    return repr(value)


def _grouped(changes: Sequence[PathChange], shown: int) -> str:
    """Return a class's changes grouped by the part of the record they belong to.

    Parameters
    ----------
    changes : sequence of PathChange
        The changes of one class.
    shown : int
        How many to name in each part.

    Returns
    -------
    str
        ``"part: entry, entry ... and N more; part: ..."``.
    """
    parts: dict[str, list[str]] = {}
    for change in changes:
        part = str(change.path[0]) if change.path else ""
        parts.setdefault(part, []).append(_entry(change))
    named = []
    for part, entries in parts.items():
        text = ", ".join(entries[:shown])
        if len(entries) > shown:
            text += f", ... and {len(entries) - shown} more"
        named.append(f"{part}: {text}" if part else text)
    return "; ".join(named)


def _entry(change: PathChange) -> str:
    """Return one change as a summary names it.

    Parameters
    ----------
    change : PathChange
        The change.

    Returns
    -------
    str
        Its description relative to its part; a model record that is present on one
        side only is named as a model whose numbers changed.
    """
    path = change.path
    if len(path) == 3 and path[:2] == ("materials", "records") and change.absent:
        material, kind, digest = ([*str(path[2]).split(":"), "", ""])[:3]
        side = "this run" if change.absent == "stored" else "the stored run"
        return f"optmat model {material} {kind} (digest {digest[:12]}) only in {side}"
    return change.describe(relative_to=1)


def compare_records(
    stored: Mapping[str, JsonValue], requested: Mapping[str, JsonValue]
) -> RecordComparison:
    """Compare the hashed parts of two run records, path by path (SEP 42).

    Parameters
    ----------
    stored : collections.abc.Mapping
        The stored record's hashed part, as the file holds it.
    requested : collections.abc.Mapping
        The requested record's hashed part.

    Returns
    -------
    RecordComparison
        The differences, by class.

    Examples
    --------
    >>> from solprop.output.compatibility import compare_records
    >>> one = {"configuration": {"betas": [0.0, -2e-26, 1e-40]}}
    >>> two = {"configuration": {"betas": [0.0, -2e-26, -1e-40]}}
    >>> compare_records(one, two).continues
    False
    >>> three = {"configuration": {"betas": [0.0, -2e-26 * (1 + 1e-15), 1e-40]}}
    >>> compare_records(one, three).continues
    True
    """
    walk = _Walk(_normalised(stored), _normalised(requested))
    walk.compare((), walk.stored, walk.requested)
    by_kind: dict[PathClass, list[PathChange]] = {kind: [] for kind in PathClass}
    for change in walk.changes:
        by_kind[change.kind].append(change)
    return RecordComparison(
        layout=tuple(by_kind[_LAYOUT]),
        refused=tuple(by_kind[_COMPARED]),
        numerics=tuple(by_kind[_NUMERICS]),
        neutral=tuple(walk.neutral),
    )


def record_paths(record: Mapping[str, JsonValue]) -> Iterator[RecordPath]:
    """Yield every leaf path of a record, as the comparison walks it.

    Parameters
    ----------
    record : collections.abc.Mapping
        A run record's hashed part.

    Yields
    ------
    tuple
        Each leaf's path, after normalisation; an array or a list of numbers is one
        leaf.
    """
    normalised = _normalised(record)

    def walk(path: RecordPath, value: JsonValue) -> Iterator[RecordPath]:
        if _is_tagged_leaf(value) or (
            isinstance(value, list) and _list_array(value) is not None
        ):
            yield path
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from walk((*path, key), item)
        elif isinstance(value, list):
            yield path
            for index, item in enumerate(value):
                yield from walk((*path, index), item)
        else:
            yield path

    yield from walk((), normalised)
