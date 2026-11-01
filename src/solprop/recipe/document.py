# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The recipe document: its tables, its format version and the forms of its values.

A recipe is a TOML document with five fixed tables, and no other top-level name, so a
future keyword can never collide with the format:

``[recipe]``
    ``format`` (an integer) and ``route`` (a string), required; ``title`` and
    ``solprop`` (the version that wrote it), informative strings.
``[build]``
    The keywords of the route's ``Simulation.from_*`` constructor.
``[execution]``
    ``fields``, ``field_planes``, ``resume_history`` and ``memory_budget``.
``[pins]``
    Values set from a design target.
``[scan]``
    A scan over parameter paths.

This layer knows the document's shape and nothing of a route's keywords: those, and
which form each value takes, are the parameter metadata's. It reads text with
:mod:`tomllib`, checks the tables and the format version, and writes canonical text
with ``tomli_w``. Every value is a string, a boolean, a 64-bit integer, a float, a
table or an array: TOML's dates and times are refused, and so is an integer outside 64
bits, which a recipe writes as a decimal string (:func:`encode_integer`), since other
TOML readers refuse it.

The decoders and encoders here are the value forms the metadata calls: the ``"none"``
spelling, complex numbers, large integers, numeric arrays and data files.
"""

import difflib
import hashlib
import io
import re
import tomllib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
import tomli_w
from jaxtyping import Shaped

from solprop.recipe.errors import RecipeError

__all__ = [
    "EXECUTION_KEYS",
    "FORMAT",
    "MAX_ARRAY_LENGTH",
    "NONE",
    "RECIPE_KEYS",
    "REFUSED_BUILD_KEYS",
    "TABLES",
    "DataFile",
    "Document",
    "decode_array",
    "decode_complex",
    "decode_data_file",
    "decode_integer",
    "decode_none",
    "dump_text",
    "encode_complex",
    "encode_data_file",
    "encode_integer",
    "encode_none",
    "load_array",
    "parse_text",
    "read_data_file",
    "resolve_data_file",
    "sha256_digest",
    "unknown_key",
    "upgrade",
    "validate_document",
]

FORMAT = 1
"""The recipe format this solprop writes, and the newest it reads."""

TABLES = ("recipe", "build", "execution", "pins", "scan")
"""The top-level tables, in the order they are written."""

RECIPE_KEYS = ("format", "route", "title", "solprop")
"""The keys of ``[recipe]``, in the order they are written; the first two required."""

EXECUTION_KEYS = ("fields", "field_planes", "resume_history", "memory_budget")
"""The keys of ``[execution]``: the run's settings that are not build keywords."""

REFUSED_BUILD_KEYS = (
    "output",
    "progress",
    "progress_period",
    "resume",
    "stop",
    "snapshots",
    "tdse_archive",
)
"""Keywords a recipe may not hold: the caller or the command line supplies them."""

MAX_ARRAY_LENGTH = 10_000
"""The longest array a recipe holds inline; a longer one is a data file."""

NONE = "none"
"""The spelling of ``None`` for a keyword whose default is not ``None``."""

type Document = dict[str, dict[str, object]]
"""A recipe document: each present top-level table, as :mod:`tomllib` reads it."""

_INT64 = (-(2**63), 2**63 - 1)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DECIMAL_INTEGER = re.compile(r"[+-]?[0-9]+")


def unknown_key(
    key: str, valid: Collection[str], *, where: str, path: str | None = None
) -> RecipeError:
    """Return the error for an unknown key, naming the nearest valid one.

    Parameters
    ----------
    key : str
        The key found.
    valid : collection of str
        The keys allowed there.
    where : str
        Where the key was found, such as ``"[execution]"``.
    path : str or None, optional
        The parameter path, for the error.

    Returns
    -------
    RecipeError
        The error, to raise; it names the nearest valid key when one is close, and
        otherwise lists them all.

    Examples
    --------
    >>> print(unknown_key("field", ["fields", "field_planes"], where="[execution]"))
    unknown key 'field' in [execution]; did you mean 'fields'?
    """
    nearest = difflib.get_close_matches(key, sorted(valid), n=1)
    if nearest:
        hint = f"did you mean {nearest[0]!r}?"
    else:
        hint = f"expected one of {', '.join(sorted(valid))}"
    return RecipeError(f"unknown key {key!r} in {where}; {hint}", path=path)


def _location(table: str, keys: Sequence[str | int]) -> str:
    """Write where a value is, as ``[table] key.sub[0]``, for an error message.

    Parameters
    ----------
    table : str
        The top-level table.
    keys : sequence of str or int
        The keys and indices from the table to the value.

    Returns
    -------
    str
        The location.
    """
    parts: list[str] = []
    for key in keys:
        if isinstance(key, int):
            parts.append(f"[{key}]")
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            parts.append(f".{key}" if parts else key)
        else:
            parts.append(f"[{key!r}]")
    return f"[{table}] {''.join(parts)}" if parts else f"[{table}]"


def _check_values(node: object, table: str, keys: tuple[str | int, ...]) -> None:
    """Refuse a value that is not a recipe value: a date, a wide integer, an object.

    Parameters
    ----------
    node : object
        The value, checked with everything inside it.
    table : str
        The top-level table it is in, for the error.
    keys : tuple of str or int
        The keys and indices from the table to it, for the error.

    Raises
    ------
    RecipeError
        If ``node`` or a value inside it is not a string, boolean, 64-bit integer,
        float, table or array, or a table has a key that is not a string.
    """
    match node:
        case dict():
            for key, value in node.items():
                if not isinstance(key, str):
                    location = _location(table, keys)
                    raise RecipeError(f"a key in {location} is not a string: {key!r}")
                _check_values(value, table, (*keys, key))
        case list():
            for index, value in enumerate(node):
                _check_values(value, table, (*keys, index))
        case bool() | str() | float():
            pass
        case int():
            if not _INT64[0] <= node <= _INT64[1]:
                raise RecipeError(
                    f"{_location(table, keys)} is an integer outside 64 bits, which "
                    f"TOML cannot hold; write it as a decimal string, {str(node)!r}"
                )
        case _:
            raise RecipeError(
                f"{_location(table, keys)} is a {type(node).__name__}, which is not a "
                f"recipe value (a string, boolean, integer, float, table or array)"
            )


def upgrade(document: Mapping[str, Mapping[str, object]]) -> Document:
    """Return a document upgraded from its format to :data:`FORMAT`.

    Each format after the first adds one step, from format ``n`` to ``n + 1``, which
    rewrites what changed; a document is passed through the steps from its own
    format. Format 1 is the first, so for now this returns the document unchanged (as
    a new document).

    Parameters
    ----------
    document : Mapping
        A document whose ``[recipe] format`` is an integer between 1 and
        :data:`FORMAT`.

    Returns
    -------
    Document
        The document in format :data:`FORMAT`.

    Raises
    ------
    RecipeError
        If the format is not one this solprop reads.

    Examples
    --------
    >>> upgrade({"recipe": {"format": 1, "route": "gnlse"}})
    {'recipe': {'format': 1, 'route': 'gnlse'}}
    """
    upgraded: Document = {name: dict(table) for name, table in document.items()}
    version = upgraded["recipe"]["format"]
    if not isinstance(version, int) or not 1 <= version <= FORMAT:
        raise RecipeError(f"cannot upgrade a recipe of format {version!r}")
    while version < FORMAT:  # pragma: no cover - format 1 is the only format yet
        upgraded = _UPGRADES[version](upgraded)
        version += 1
        upgraded["recipe"]["format"] = version
    return upgraded


# format n -> the function that rewrites a format-n document as format n + 1.
_UPGRADES: dict[int, Callable[[Document], Document]] = {}


def _check_format(document: Mapping[str, object]) -> None:
    """Check ``[recipe] format`` first, so a newer recipe is refused as newer.

    Parameters
    ----------
    document : Mapping
        The document.

    Raises
    ------
    RecipeError
        If ``[recipe]`` or its format is missing, the format is not an integer, or
        it is newer than :data:`FORMAT` or below 1.
    """
    table = document.get("recipe")
    if table is None:
        raise RecipeError("a recipe needs a [recipe] table with format and route")
    if not isinstance(table, Mapping):
        raise RecipeError(f"[recipe] is a table, not {table!r}")
    version = table.get("format")
    if version is None:
        raise RecipeError(f"[recipe] needs format, the recipe format ({FORMAT} now)")
    if isinstance(version, bool) or not isinstance(version, int):
        raise RecipeError(f"[recipe] format is an integer, not {version!r}")
    if version > FORMAT:
        raise RecipeError(
            f"this recipe is in format {version}, newer than this solprop reads "
            f"(format {FORMAT}); update solprop to read it"
        )
    if version < 1:
        raise RecipeError(f"[recipe] format is at least 1, not {version}")


def _check_recipe_table(table: Mapping[str, object]) -> None:
    """Check ``[recipe]``'s keys and their types; the format is already checked.

    Parameters
    ----------
    table : Mapping
        The ``[recipe]`` table.

    Raises
    ------
    RecipeError
        If a key is unknown, ``route`` is missing, or a value is not a string.
    """
    for key in table:
        if key not in RECIPE_KEYS:
            raise unknown_key(key, RECIPE_KEYS, where="[recipe]")
    if "route" not in table:
        raise RecipeError("[recipe] needs route, the route the recipe builds")
    for key in ("route", "title", "solprop"):
        if key in table and not isinstance(table[key], str):
            raise RecipeError(f"[recipe] {key} is a string, not {table[key]!r}")


def validate_document(document: Mapping[str, object]) -> Document:
    """Check a recipe document's tables and format, and upgrade it to this format.

    Parameters
    ----------
    document : Mapping
        The document, as :mod:`tomllib` reads it.

    Returns
    -------
    Document
        A new document in format :data:`FORMAT`; the values are not copied.

    Raises
    ------
    RecipeError
        If a top-level name is not one of :data:`TABLES`, a table is not a table,
        ``[recipe]`` is missing or wrong, the format is newer than :data:`FORMAT`,
        ``[build]`` holds one of :data:`REFUSED_BUILD_KEYS`, ``[execution]`` holds a
        key not in :data:`EXECUTION_KEYS`, or a value is not a recipe value.

    Examples
    --------
    >>> document = validate_document(
    ...     {"recipe": {"format": 1, "route": "gnlse"}, "build": {"fwhm": "30 fs"}}
    ... )
    >>> document["build"]
    {'fwhm': '30 fs'}
    """
    _check_format(document)
    tables: Document = {}
    for name, table in document.items():
        if name not in TABLES:
            raise unknown_key(name, TABLES, where="the top level")
        if not isinstance(table, dict):
            raise RecipeError(f"[{name}] is a table, not {table!r}")
        tables[name] = table
    _check_recipe_table(tables["recipe"])
    for name, table in tables.items():
        _check_values(table, name, ())
    upgraded = upgrade(tables)
    for key in upgraded.get("build", {}):
        if key in REFUSED_BUILD_KEYS:
            raise RecipeError(
                f"[build] may not hold {key!r}: it chooses files, terminals or resume "
                f"behaviour, which the caller or the command line supplies",
                path=key,
            )
    for key in upgraded.get("execution", {}):
        if key not in EXECUTION_KEYS:
            raise unknown_key(key, EXECUTION_KEYS, where="[execution]")
    return upgraded


def parse_text(text: str) -> Document:
    """Read a recipe's TOML text into a checked document of this format.

    Parameters
    ----------
    text : str
        The TOML text.

    Returns
    -------
    Document
        The document, checked and upgraded by :func:`validate_document`.

    Raises
    ------
    RecipeError
        If the text is not TOML, or the document is refused by
        :func:`validate_document`.

    Examples
    --------
    >>> text = '''
    ... [recipe]
    ... format = 1
    ... route = "capillary"
    ...
    ... [build]
    ... radius = "125 um"
    ... '''
    >>> parse_text(text)["build"]
    {'radius': '125 um'}
    """
    try:
        document = tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise RecipeError(f"the recipe is not valid TOML: {error}") from error
    return validate_document(document)


def dump_text(document: Mapping[str, Mapping[str, object]]) -> str:
    """Write a recipe document as canonical TOML text.

    The tables are written in the order of :data:`TABLES` and ``[recipe]``'s keys in
    the order of :data:`RECIPE_KEYS`; every other table keeps its own order. Keys
    that are not bare TOML keys, such as parameter paths in ``[pins]``, are quoted.
    No comments are written.

    Parameters
    ----------
    document : Mapping
        The document, as :func:`parse_text` returns it; every value already encoded
        (a complex number as a table, a wide integer as a decimal string).

    Returns
    -------
    str
        The text, which :func:`parse_text` reads back to an equal document.

    Raises
    ------
    RecipeError
        If the document is refused by :func:`validate_document`.

    Examples
    --------
    >>> recipe = {"route": "gnlse", "format": 1}
    >>> print(dump_text({"build": {"fwhm": "30 fs"}, "recipe": recipe}))
    [recipe]
    format = 1
    route = "gnlse"
    <BLANKLINE>
    [build]
    fwhm = "30 fs"
    <BLANKLINE>
    """
    checked = validate_document(document)
    recipe = checked["recipe"]
    ordered: dict[str, dict[str, object]] = {
        "recipe": {key: recipe[key] for key in RECIPE_KEYS if key in recipe}
    }
    ordered |= {name: checked[name] for name in TABLES[1:] if name in checked}
    return tomli_w.dumps(ordered)


def decode_none(value: object) -> object | None:
    """Read the ``"none"`` spelling of ``None``.

    Only a keyword whose default is not ``None`` takes the spelling; the metadata
    says which.

    Parameters
    ----------
    value : object
        The recipe value.

    Returns
    -------
    object or None
        ``None`` for the string ``"none"``, and otherwise ``value`` itself.

    Examples
    --------
    >>> decode_none("none") is None
    True
    >>> decode_none(0.5)
    0.5
    """
    return None if value == NONE else value


def encode_none(value: object | None) -> object:
    """Write ``None`` as ``"none"``, the inverse of :func:`decode_none`.

    Parameters
    ----------
    value : object or None
        The value.

    Returns
    -------
    object
        ``"none"`` for ``None``, and otherwise ``value`` itself.

    Examples
    --------
    >>> encode_none(None)
    'none'
    """
    return NONE if value is None else value


def _real(value: object, part: str, path: str | None) -> float:
    """Read one part of a complex number.

    Parameters
    ----------
    value : object
        The part, an ``int`` or ``float``.
    part : str
        ``"real"`` or ``"imaginary"``, for the error.
    path : str or None
        The parameter path, for the error.

    Returns
    -------
    float
        The part.

    Raises
    ------
    RecipeError
        If ``value`` is a boolean or not a number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RecipeError(f"the {part} part {value!r} is not a number", path=path)
    return float(value)


def decode_complex(value: object, *, path: str | None = None) -> complex:
    """Read a complex number written as ``{re = ..., im = ...}``.

    Parameters
    ----------
    value : object
        A table with exactly the keys ``re`` and ``im``, each a number.
    path : str or None, optional
        The parameter path, for the error.

    Returns
    -------
    complex
        The number.

    Raises
    ------
    RecipeError
        If ``value`` is not such a table.

    Examples
    --------
    >>> decode_complex({"re": 1.45, "im": 0.0})
    (1.45+0j)
    """
    if not isinstance(value, Mapping) or set(value) != {"re", "im"}:
        raise RecipeError(
            f"a complex number is a table {{re = ..., im = ...}}, not {value!r}",
            path=path,
        )
    return complex(
        _real(value["re"], "real", path), _real(value["im"], "imaginary", path)
    )


def encode_complex(value: complex) -> dict[str, float]:
    """Write a complex number as ``{re, im}``, the inverse of :func:`decode_complex`.

    Parameters
    ----------
    value : complex
        The number.

    Returns
    -------
    dict
        ``{"re": value.real, "im": value.imag}``.

    Examples
    --------
    >>> encode_complex(1.45 + 2e-6j)
    {'re': 1.45, 'im': 2e-06}
    """
    return {"re": value.real, "im": value.imag}


def decode_integer(value: object, *, path: str | None = None) -> int:
    """Read an integer, given as a TOML integer or a decimal string.

    The string form holds an integer that TOML's 64 bits cannot, such as a drawn
    128-bit noise seed; any integer keyword accepts it.

    Parameters
    ----------
    value : object
        An ``int`` (not a boolean, and not a float even if integral) or a string of
        decimal digits with an optional sign.
    path : str or None, optional
        The parameter path, for the error.

    Returns
    -------
    int
        The integer.

    Raises
    ------
    RecipeError
        If ``value`` is neither.

    Examples
    --------
    >>> decode_integer("340282366920938463463374607431768211455")
    340282366920938463463374607431768211455
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str) and _DECIMAL_INTEGER.fullmatch(value.strip()):
        return int(value)
    raise RecipeError(
        f"expected an integer, or a decimal string for a large one, not {value!r}",
        path=path,
    )


def encode_integer(value: int) -> int | str:
    """Write an integer as TOML holds it, the inverse of :func:`decode_integer`.

    Parameters
    ----------
    value : int
        The integer.

    Returns
    -------
    int or str
        ``value`` itself within 64 bits, and its decimal string outside.

    Examples
    --------
    >>> encode_integer(2**64)
    '18446744073709551616'
    >>> encode_integer(42)
    42
    """
    return value if _INT64[0] <= value <= _INT64[1] else str(value)


def decode_array[T](
    value: object,
    decode_element: Callable[[object, int, str | None], T],
    *,
    path: str | None = None,
    max_length: int = MAX_ARRAY_LENGTH,
) -> list[T]:
    """Read a one-dimensional array held inline, element by element.

    Parameters
    ----------
    value : object
        A list of at most ``max_length`` elements, none of them a list.
    decode_element : callable
        Called as ``decode_element(element, index, element_path)`` for each element,
        where ``element_path`` is ``path[index]`` (or ``None``), so that the element's
        unit can depend on its index; it returns the decoded element.
    path : str or None, optional
        The parameter path of the array.
    max_length : int, optional
        The longest array held inline.

    Returns
    -------
    list
        The decoded elements.

    Raises
    ------
    RecipeError
        If ``value`` is not a list, is longer than ``max_length`` (a data file holds
        it instead) or has a list as an element; and whatever ``decode_element``
        raises.

    Examples
    --------
    >>> decode_array([1, 2], lambda element, index, path: 10 * element, path="x")
    [10, 20]
    """
    if not isinstance(value, list):
        raise RecipeError(f"expected an array, not {value!r}", path=path)
    if len(value) > max_length:
        raise RecipeError(
            f"an array of {len(value)} elements is longer than a recipe holds inline "
            f"({max_length}); save it as a .npy file beside the recipe and write "
            f'{{file = "name.npy", sha256 = "..."}}',
            path=path,
        )
    decoded: list[T] = []
    for index, element in enumerate(value):
        element_path = f"{path}[{index}]" if path else None
        if isinstance(element, list):
            raise RecipeError(
                "an array held inline is one-dimensional; a larger-dimensional one is "
                "a data file",
                path=element_path,
            )
        decoded.append(decode_element(element, index, element_path))
    return decoded


@dataclass(frozen=True)
class DataFile:
    """A reference to a data file beside the recipe, with its SHA-256 digest.

    Attributes
    ----------
    file : str
        The path, relative to the recipe's directory.
    sha256 : str
        The SHA-256 digest of the file's bytes, as 64 hexadecimal digits; stored in
        lower case.

    Raises
    ------
    RecipeError
        If ``file`` is empty or absolute, or ``sha256`` is not 64 hexadecimal digits.

    Examples
    --------
    >>> DataFile("table.npy", 64 * "A").sha256 == 64 * "a"
    True
    """

    file: str
    sha256: str

    def __post_init__(self) -> None:
        """Check the path is relative and the digest well formed."""
        if not self.file:
            raise RecipeError("a data file needs a file name")
        if PurePosixPath(self.file).is_absolute() or PureWindowsPath(self.file).anchor:
            raise RecipeError(
                f"a data file is named relative to the recipe, not as {self.file!r}"
            )
        digest = self.sha256.lower()
        if _SHA256.fullmatch(digest) is None:
            raise RecipeError(
                f"a data file's sha256 is 64 hexadecimal digits, not {self.sha256!r}"
            )
        object.__setattr__(self, "sha256", digest)


def decode_data_file(value: object, *, path: str | None = None) -> DataFile:
    """Read an array's data-file reference, ``{file = ..., sha256 = ...}``.

    Parameters
    ----------
    value : object
        A table with exactly the string keys ``file`` and ``sha256``.
    path : str or None, optional
        The parameter path, for the error.

    Returns
    -------
    DataFile
        The reference; the file is not read.

    Raises
    ------
    RecipeError
        If ``value`` is not such a table.

    Examples
    --------
    >>> decode_data_file({"file": "betas.npy", "sha256": 64 * "0"}).file
    'betas.npy'
    """
    if (
        not isinstance(value, Mapping)
        or set(value) != {"file", "sha256"}
        or not isinstance(file := value["file"], str)
        or not isinstance(digest := value["sha256"], str)
    ):
        raise RecipeError(
            f'a data file is a table {{file = "...", sha256 = "..."}}, not {value!r}',
            path=path,
        )
    try:
        return DataFile(file, digest)
    except RecipeError as error:
        raise RecipeError(error.message, path=path) from None


def encode_data_file(reference: DataFile) -> dict[str, str]:
    """Write a data-file reference, the inverse of :func:`decode_data_file`.

    Parameters
    ----------
    reference : DataFile
        The reference.

    Returns
    -------
    dict
        ``{"file": ..., "sha256": ...}``.

    Examples
    --------
    >>> encode_data_file(DataFile("betas.npy", 64 * "0"))["file"]
    'betas.npy'
    """
    return {"file": reference.file, "sha256": reference.sha256}


def sha256_digest(data: bytes) -> str:
    """Return the SHA-256 digest a data-file reference records for ``data``.

    Parameters
    ----------
    data : bytes
        The file's bytes.

    Returns
    -------
    str
        64 lower-case hexadecimal digits.

    Examples
    --------
    >>> sha256_digest(b"")[:16]
    'e3b0c44298fc1c14'
    """
    return hashlib.sha256(data).hexdigest()


def resolve_data_file(
    reference: DataFile,
    base_dir: Path,
    *,
    allow_outside: bool = False,
    path: str | None = None,
) -> Path:
    """Return the real path of a data file, refusing one outside the recipe's directory.

    Symbolic links are resolved first, in the directory and in the file's path, so a
    link inside the directory that points out of it is refused too.

    Parameters
    ----------
    reference : DataFile
        The reference.
    base_dir : Path
        The recipe's directory.
    allow_outside : bool, optional
        Accept a file outside ``base_dir``; only the caller, never the recipe, sets it.
    path : str or None, optional
        The parameter path, for the error.

    Returns
    -------
    Path
        The resolved path of an existing regular file.

    Raises
    ------
    RecipeError
        If the file does not exist or is not a regular file, or lies outside
        ``base_dir`` without ``allow_outside``.
    """
    base = Path(base_dir).resolve()
    try:
        target = (base / reference.file).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise RecipeError(
            f"cannot find the data file {reference.file!r} in {base}: {error}",
            path=path,
        ) from error
    if not allow_outside and not target.is_relative_to(base):
        raise RecipeError(
            f"the data file {reference.file!r} resolves to {target}, outside the "
            f"recipe's directory {base}; only the caller can allow that",
            path=path,
        )
    if not target.is_file():
        raise RecipeError(f"the data file {reference.file!r} is not a file", path=path)
    return target


def read_data_file(
    reference: DataFile,
    base_dir: Path,
    *,
    allow_outside: bool = False,
    path: str | None = None,
) -> bytes:
    """Read a data file's bytes, checking its SHA-256 digest.

    The bytes are read once and checked, so what the caller parses is exactly what
    was checked.

    Parameters
    ----------
    reference : DataFile
        The reference.
    base_dir : Path
        The recipe's directory.
    allow_outside : bool, optional
        Accept a file outside ``base_dir`` (see :func:`resolve_data_file`).
    path : str or None, optional
        The parameter path, for the error.

    Returns
    -------
    bytes
        The file's contents.

    Raises
    ------
    RecipeError
        If :func:`resolve_data_file` refuses the file, or its digest is not the one
        recorded.
    """
    target = resolve_data_file(
        reference, base_dir, allow_outside=allow_outside, path=path
    )
    data = target.read_bytes()
    digest = sha256_digest(data)
    if digest != reference.sha256:
        raise RecipeError(
            f"the data file {reference.file!r} has SHA-256 {digest}, not the "
            f"{reference.sha256} the recipe records; it has changed since the recipe "
            f"was written",
            path=path,
        )
    return data


def load_array(
    reference: DataFile,
    base_dir: Path,
    *,
    allow_outside: bool = False,
    path: str | None = None,
) -> Shaped[np.ndarray, "*shape"]:
    """Load a numeric array from a ``.npy`` data file, with pickling disabled.

    Parameters
    ----------
    reference : DataFile
        The reference.
    base_dir : Path
        The recipe's directory.
    allow_outside : bool, optional
        Accept a file outside ``base_dir`` (see :func:`resolve_data_file`).
    path : str or None, optional
        The parameter path, for the error.

    Returns
    -------
    numpy.ndarray
        The array, of a boolean, integer, float or complex dtype.

    Raises
    ------
    RecipeError
        If :func:`read_data_file` refuses the file, it is not a ``.npy`` file, it holds
        objects (which only pickling could load) or its dtype is not numeric.
    """
    data = read_data_file(reference, base_dir, allow_outside=allow_outside, path=path)
    if not data.startswith(b"\x93NUMPY"):
        raise RecipeError(
            f"the data file {reference.file!r} is not a .npy file", path=path
        )
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except (ValueError, EOFError, OSError) as error:
        raise RecipeError(
            f"cannot load {reference.file!r} with pickling disabled: {error}", path=path
        ) from error
    assert isinstance(array, np.ndarray)  # the magic number is a .npy file's
    if array.dtype.kind not in "biufc":
        raise RecipeError(
            f"the data file {reference.file!r} holds {array.dtype}, not numbers",
            path=path,
        )
    return array
