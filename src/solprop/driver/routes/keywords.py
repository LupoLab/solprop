# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Name the keyword a refusal or a warning concerns (SEP 41, S10).

A keyword route builds physics objects from its keywords; an error or a warning those
objects raise is attributed here to the keyword the caller typed.
"""

import contextlib
import dataclasses
import warnings
from collections.abc import Iterator, Mapping
from typing import Any, Final, cast

from solprop.exceptions import (
    MaterialRangeWarning,
    MaterialStatusWarning,
    ParameterPath,
    SolpropError,
    SolpropWarning,
)

__all__ = ["keyword_request"]


@contextlib.contextmanager
def _about(
    parameter: ParameterPath | None = None,
    renamed: Mapping[str, str] | None = None,
) -> Iterator[None]:
    """Name the keyword a refusal raised in the block concerns (SEP 41, S10).

    An object built from a keyword's value refuses it by its own argument's name, or by
    none; the interface knows which keyword it passed, so it names that one here. The
    exception keeps its object, class, message and traceback.

    - ``parameter``: the path given to a :class:`~solprop.exceptions.SolpropError` that
      names no parameter; ``None`` leaves it unnamed.
    - ``renamed``: the inner call's argument names that differ from the keywords that
      supplied them, each mapped to its keyword: a path the error names already is
      renamed through it (each path of a conflict), and kept where it has no entry.
    """
    try:
        yield
    except SolpropError as error:
        if error.parameter is None:
            error.parameter = parameter
        elif renamed:
            error.parameter = (
                renamed.get(error.parameter, error.parameter)
                if isinstance(error.parameter, str)
                else tuple(renamed.get(path, path) for path in error.parameter)
            )
        raise


_REISSUED: Final[dict[str | tuple[str, type[Warning], int], int]] = {}
"""The registry of the warnings :func:`_warnings_about` issues again.

It stands in for the issuing module's own, so that a ``"default"`` filter still shows
a warning once per location and text rather than once per run.
"""


@contextlib.contextmanager
def _warnings_about(
    parameter: ParameterPath, categories: tuple[type[SolpropWarning], ...]
) -> Iterator[None]:
    """Name the keyword the value warnings issued in the block concern (SEP 41, S10).

    A material model warns deep inside the physics, where the keyword that chose the
    material is not known. The block's warnings are recorded and, when it ends, issued
    again in order at their own locations and through the caller's filters, so that an
    ``"error"`` filter still raises and an ``"ignore"`` one still hides them: a warning
    of one of `categories` that names no parameter is given `parameter` first, and
    every other warning is issued as it was.
    """
    caught: list[warnings.WarningMessage] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
    finally:
        for record in caught:
            # A recorded warning is an instance of its category, never a string.
            message = cast("Warning", record.message)
            if isinstance(message, categories) and message.parameter is None:
                message.parameter = parameter
            warnings.warn_explicit(
                message,
                record.category,
                record.filename,
                record.lineno,
                registry=_REISSUED,
                source=record.source,
            )


_MATERIAL_WARNINGS: Final = (MaterialRangeWarning, MaterialStatusWarning)
"""The warnings a material model issues about the conditions it is evaluated at."""

_GRID_KEYWORDS: Final[Mapping[str, str]] = {"reference_wavelength": "wavelength"}
"""The grid constructors' argument names that a keyword route spells otherwise."""

_PULSE_KEYWORDS: Final[Mapping[str, str]] = {"phase": "phases"}
"""The pulse constructors' argument names that a keyword route spells otherwise."""


def _path(*names: str) -> ParameterPath:
    """Return the path of one keyword, or the tuple of several in conflict.

    - ``*names``: the keywords, at least one.
    """
    return names[0] if len(names) == 1 else names


def _unset(**keywords: object) -> ParameterPath:
    """Return the path of the keywords among `keywords` left unset.

    - ``**keywords``: the keywords a refusal needs, by name; at least one is ``None``.
    """
    return _path(*(name for name, value in keywords.items() if value is None))


def keyword_request[R](
    kind: type[R], scope: Mapping[str, object], **given: object
) -> R:
    """Build a route's request from its constructor's locals.

    Parameters
    ----------
    kind : type
        The request, a frozen dataclass whose fields are keywords of the route.
    scope : collections.abc.Mapping
        The public constructor's ``locals()``: each field of `kind` it holds is taken
        from it, and anything else in it is ignored.
    **given : object
        Fields set otherwise than the constructor took them, overriding `scope`. A
        field in neither keeps its default.

    Returns
    -------
    object
        The request, an instance of `kind`.
    """
    # `kind` is a dataclass; the type variable cannot say so.
    names = {field.name for field in dataclasses.fields(cast("Any", kind))}
    return kind(**{name: scope[name] for name in names if name in scope} | given)
