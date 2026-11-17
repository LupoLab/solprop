# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Refuse an argument by name, in one wording across the package.

:func:`positive`, :func:`finite`, :func:`choice` and :func:`shape` each take the value,
the name the message gives it and the exception class of the calling layer, so a refusal
reads the same wherever it is raised and is still the caller's own error. Bespoke
physics checks stay where they are; these are only the plain conditions that recur
everywhere. A warning, which cannot refuse, is attributed to the caller's line the same
way everywhere too: :data:`PACKAGE_FILES` is the ``skip_file_prefixes`` that steps out
of the package.

This module imports nothing from the rest of the package except the exception hierarchy,
so that any module may use it.
"""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from solprop.exceptions import ParameterPath, SolpropError

__all__ = ["PACKAGE_FILES", "choice", "finite", "positive", "shape"]

PACKAGE_FILES: Final = (str(Path(__file__).resolve().parent),)
"""The package's own files, as ``skip_file_prefixes`` for :func:`warnings.warn`.

A warning raised several calls deep inside solprop -- through a constructor, a recipe,
a scan or the service -- skips every frame under the package root, so that it names the
caller's line; no fixed ``stacklevel`` reaches the user's code from all of those paths.
"""


def positive(
    value: float,
    name: str,
    error: type[SolpropError],
    *,
    parameter: ParameterPath | None = None,
    reject_bool: bool = False,
) -> float:
    """Return a finite, strictly positive number as a float, or refuse it by name.

    A string is refused rather than parsed.

    - ``parameter``: the recipe path the error carries, where the caller has one.
    - ``reject_bool``: also refuse ``True``, which would otherwise count as 1.
    """
    refused = isinstance(value, str | bytes) or (
        reject_bool and isinstance(value, bool)
    )
    number = math.nan if refused else float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise error(
            f"{name} must be finite and positive, got {value!r}", parameter=parameter
        )
    return number


def finite(
    value: float,
    name: str,
    error: type[SolpropError],
    *,
    parameter: ParameterPath | None = None,
) -> float:
    """Return a finite number as a float, or refuse it by name; a string is refused."""
    number = math.nan if isinstance(value, str | bytes) else float(value)
    if not math.isfinite(number):
        raise error(f"{name} must be finite, got {value!r}", parameter=parameter)
    return number


def choice(
    value: object,
    name: str,
    choices: Sequence[object],
    error: type[SolpropError],
    *,
    parameter: ParameterPath | None = None,
) -> None:
    """Refuse a keyword outside its documented values, listing them.

    The annotations already say which values are allowed; this is the check a caller
    without a type checker meets.
    """
    if value not in choices:
        *head, last = (repr(option) for option in choices)
        allowed = f"{', '.join(head)} or {last}" if head else last
        raise error(f"{name} must be {allowed}, got {value!r}", parameter=parameter)


def shape(
    actual: tuple[int, ...],
    expected: tuple[int, ...],
    name: str,
    error: type[SolpropError],
    *,
    parameter: ParameterPath | None = None,
) -> None:
    """Refuse an array whose shape is not the one expected."""
    if tuple(actual) != tuple(expected):
        raise error(
            f"{name} must have shape {tuple(expected)}, got {tuple(actual)}",
            parameter=parameter,
        )
