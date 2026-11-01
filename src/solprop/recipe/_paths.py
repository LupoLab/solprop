# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

r"""Parameter paths: the canonical name of every value in a recipe.

A path is a keyword, then any number of steps: ``.name`` for a keyword of an object,
``[i]`` for an element of an array and ``["key"]`` for a key of a mapping, as in
``pressure``, ``pulses[1].energy`` or ``launch["HE1,1:even"].share``. Paths are what
``Recipe.set`` and ``Recipe.get`` take, what errors name, what a scan varies and what a
pin sets.

In the tree a TOML document reads into, an object and a mapping are both dictionaries,
so ``.name`` and ``["name"]`` address the same entry; the parsed path keeps them apart
(:class:`Attribute` and :class:`Key`) because they are different texts, and the
canonical text is the one :func:`format_path` writes. An index is a non-negative
integer with no leading zero, and a key is a JSON string, so ``["a\"b"]`` is the key
``a"b``.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from solprop.recipe._errors import RecipeError

__all__ = [
    "Attribute",
    "Index",
    "Key",
    "Step",
    "format_path",
    "get_path",
    "parse_path",
    "set_path",
]


@dataclass(frozen=True)
class Attribute:
    """A keyword, or a keyword of an object: ``name`` or ``.name`` in a path.

    Attributes
    ----------
    name : str
        A Python identifier.
    """

    name: str


@dataclass(frozen=True)
class Index:
    """An element of an array: ``[i]`` in a path.

    Attributes
    ----------
    position : int
        The zero-based position, at least 0.
    """

    position: int


@dataclass(frozen=True)
class Key:
    """A key of a mapping: ``["key"]`` in a path.

    Attributes
    ----------
    key : str
        The key, any string.
    """

    key: str


type Step = Attribute | Index | Key
"""One step of a parameter path."""

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INDEX = re.compile(r"\[(0|[1-9][0-9]*)\]")
_KEY = re.compile(r'\[("(?:[^"\\]|\\.)*")\]', re.DOTALL)


def _refuse(text: str, column: int, reason: str) -> RecipeError:
    """Return the error for a path text that cannot be read.

    Parameters
    ----------
    text : str
        The path text.
    column : int
        The zero-based position where reading stopped.
    reason : str
        What was expected there.

    Returns
    -------
    RecipeError
        The error, to raise; it gives the column counted from 1.
    """
    return RecipeError(
        f"cannot read the parameter path {text!r} at column {column + 1}: {reason}"
    )


def parse_path(text: str) -> tuple[Step, ...]:
    """Read a parameter path into its steps.

    Parameters
    ----------
    text : str
        The path, such as ``'pulses[1].energy'``.

    Returns
    -------
    tuple of Attribute, Index or Key
        The steps; the first is always an :class:`Attribute`, the keyword.

    Raises
    ------
    RecipeError
        If ``text`` is not a path, naming the column where reading stopped.

    Examples
    --------
    >>> parse_path('launch["HE1,1:even"].share')
    (Attribute(name='launch'), Key(key='HE1,1:even'), Attribute(name='share'))
    >>> parse_path("pulses[1].energy")
    (Attribute(name='pulses'), Index(position=1), Attribute(name='energy'))
    """
    match = _NAME.match(text)
    if match is None:
        raise _refuse(text, 0, "a path starts with a keyword")
    steps: list[Step] = [Attribute(match[0])]
    column = match.end()
    while column < len(text):
        if text.startswith(".", column):
            match = _NAME.match(text, column + 1)
            if match is None:
                raise _refuse(text, column + 1, "expected a name after '.'")
            steps.append(Attribute(match[0]))
        elif (match := _INDEX.match(text, column)) is not None:
            steps.append(Index(int(match[1])))
        elif (match := _KEY.match(text, column)) is not None:
            try:
                key: object = json.loads(match[1])
            except json.JSONDecodeError as error:
                reason = f"the key is not a JSON string: {error}"
                raise _refuse(text, column, reason) from error
            assert isinstance(key, str)  # the pattern admits only a JSON string
            steps.append(Key(key))
        else:
            raise _refuse(
                text,
                column,
                'expected .name, [index] or ["key"], and an index has no sign or '
                "leading zero",
            )
        column = match.end()
    return tuple(steps)


def format_path(steps: Sequence[Step]) -> str:
    """Write the canonical text of a parameter path.

    Parameters
    ----------
    steps : sequence of Attribute, Index or Key
        The steps, starting with an :class:`Attribute`.

    Returns
    -------
    str
        The canonical text, which :func:`parse_path` reads back to ``steps``.

    Raises
    ------
    RecipeError
        If ``steps`` is empty, does not start with a keyword, or holds a name that is
        not an identifier or a negative index.

    Examples
    --------
    >>> format_path((Attribute("pulses"), Index(1), Attribute("energy")))
    'pulses[1].energy'
    >>> format_path((Attribute("launch"), Key("HE1,1:even")))
    'launch["HE1,1:even"]'
    """
    if not steps or not isinstance(steps[0], Attribute):
        raise RecipeError("a parameter path starts with a keyword")
    for step in steps:
        if isinstance(step, Attribute) and _NAME.fullmatch(step.name) is None:
            raise RecipeError(f"{step.name!r} is not a keyword name")
        if isinstance(step, Index) and step.position < 0:
            raise RecipeError(f"an index is at least 0, not {step.position}")
    # The first step is a keyword, written without the dot of an attribute.
    return "".join(_step_text(step) for step in steps).removeprefix(".")


def _steps(path: str | Sequence[Step]) -> tuple[tuple[Step, ...], str]:
    """Return the steps of a path given as text or steps, and its canonical text.

    Parameters
    ----------
    path : str or sequence of Attribute, Index or Key
        The path.

    Returns
    -------
    tuple
        The steps, and the canonical text.

    Raises
    ------
    RecipeError
        If the path cannot be read or written.
    """
    steps = parse_path(path) if isinstance(path, str) else tuple(path)
    return steps, format_path(steps)


def _step_text(step: Step) -> str:
    """Write one step as it appears after the keyword.

    Parameters
    ----------
    step : Attribute, Index or Key
        The step.

    Returns
    -------
    str
        ``.name``, ``[i]`` or ``["key"]``.
    """
    if isinstance(step, Attribute):
        return f".{step.name}"
    if isinstance(step, Index):
        return f"[{step.position}]"
    return f"[{json.dumps(step.key, ensure_ascii=False)}]"


def _child(node: object, step: Step, path: str, reached: Sequence[Step]) -> object:
    """Return the entry of a node that one step addresses.

    Parameters
    ----------
    node : object
        The table or array the step is taken in.
    step : Attribute, Index or Key
        The step.
    path : str
        The whole path's canonical text, for the error.
    reached : sequence of Attribute, Index or Key
        The steps that led to ``node``; empty at the top, where the step is a
        keyword.

    Returns
    -------
    object
        The entry.

    Raises
    ------
    RecipeError
        If ``node`` has no such entry, naming what was reached and the step.
    """
    if not reached:
        assert isinstance(step, Attribute)  # parse_path and format_path ensure it
        if not isinstance(node, Mapping) or step.name not in node:
            raise RecipeError(f"there is no keyword {step.name!r}", path=path)
        return node[step.name]
    owner = format_path(reached)
    if isinstance(step, Index):
        if not isinstance(node, list):
            raise RecipeError(
                f"{owner} is not an array, so it has no {_step_text(step)}", path=path
            )
        if step.position >= len(node):
            raise RecipeError(
                f"{owner} has {len(node)} elements, so no {_step_text(step)}",
                path=path,
            )
        return node[step.position]
    name = step.name if isinstance(step, Attribute) else step.key
    if not isinstance(node, Mapping):
        raise RecipeError(
            f"{owner} is not a table, so it has no {_step_text(step)}", path=path
        )
    if name not in node:
        raise RecipeError(f"{owner} has no {_step_text(step)}", path=path)
    return node[name]


def get_path(tree: Mapping[str, object], path: str | Sequence[Step]) -> object:
    """Return the value at a parameter path of a recipe table.

    Parameters
    ----------
    tree : Mapping
        The table the path starts in (a recipe's ``[build]`` table), as
        :mod:`tomllib` reads it: dictionaries, lists and values.
    path : str or sequence of Attribute, Index or Key
        The path, as text or steps. ``.name`` and ``["name"]`` both address the key
        ``name`` of a dictionary.

    Returns
    -------
    object
        The value; a table or array is returned itself, not a copy.

    Raises
    ------
    RecipeError
        If a step finds no entry, naming the path and the step.

    Examples
    --------
    >>> tree = {"pulses": [{"energy": "1 uJ"}, {"energy": "10 uJ"}]}
    >>> get_path(tree, "pulses[1].energy")
    '10 uJ'
    """
    steps, text = _steps(path)
    node: object = tree
    reached: list[Step] = []
    for step in steps:
        node = _child(node, step, text, reached)
        reached.append(step)
    return node


def _replace(
    node: object,
    steps: tuple[Step, ...],
    value: object,
    path: str,
    reached: tuple[Step, ...],
) -> object:
    """Return a copy of a node with a value set at the end of some steps.

    Parameters
    ----------
    node : object
        The table or array the steps start in.
    steps : tuple of Attribute, Index or Key
        The steps from ``node``, at least one.
    value : object
        The value to set.
    path : str
        The whole path's canonical text, for the error.
    reached : tuple of Attribute, Index or Key
        The steps that led to ``node``.

    Returns
    -------
    object
        The copy; the tables and arrays on the path are new and the rest shared.

    Raises
    ------
    RecipeError
        If a step before the last finds no entry, or an index is past the end.
    """
    step, rest = steps[0], steps[1:]
    if rest:
        child = _child(node, step, path, reached)
        value = _replace(child, rest, value, path, (*reached, step))
    if isinstance(step, Index):
        _child(node, step, path, reached)  # an element is replaced, never appended
        assert isinstance(node, list)
        copy = list(node)
        copy[step.position] = value
        return copy
    if not isinstance(node, Mapping):
        owner = format_path(reached)
        raise RecipeError(
            f"{owner} is not a table, so it has no {_step_text(step)}", path=path
        )
    name = step.name if isinstance(step, Attribute) else step.key
    return {**node, name: value}


def set_path(
    tree: Mapping[str, object], path: str | Sequence[Step], value: object
) -> dict[str, object]:
    """Return a copy of a recipe table with the value at a parameter path replaced.

    The tables and arrays on the path are copied and everything else is shared, so
    ``tree`` is not changed. The last step may add a key to a table, but every
    earlier step must exist, and an index must name an existing element: an array
    is never extended.

    Parameters
    ----------
    tree : Mapping
        The table the path starts in, as for :func:`get_path`.
    path : str or sequence of Attribute, Index or Key
        The path, as text or steps.
    value : object
        The new value.

    Returns
    -------
    dict
        The new table.

    Raises
    ------
    RecipeError
        If a step before the last finds no entry, or an index is past the end, naming
        the path and the step.

    Examples
    --------
    >>> tree = {"pulses": [{"energy": "1 uJ"}]}
    >>> set_path(tree, "pulses[0].energy", "2 uJ")
    {'pulses': [{'energy': '2 uJ'}]}
    >>> tree
    {'pulses': [{'energy': '1 uJ'}]}
    """
    steps, text = _steps(path)
    result = _replace(dict(tree), steps, value, text, ())
    assert isinstance(result, dict)
    return result
