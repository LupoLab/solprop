# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The canonical identity of a record: its one JSON text, and the SHA-256 of that text.

Every digest that identifies a record -- a run's resume key, a scan's common record, a
coupling plane's basis, a material population, a TDSE basis or descriptor, a cache key
-- is taken over :func:`canonical_json`, so that two records equal as JSON have one
identity on every platform and in every process. The text has its keys sorted and no
whitespace. A NumPy scalar (a bool, an integer or a floating number, which records built
from arrays carry) is written as the Python number it holds; any other value JSON cannot
hold raises :class:`TypeError`: an identity that stringified an object would silently
merge records that differ (an array's ``str`` is abbreviated). Non-finite numbers are
written as JSON's ``NaN`` and ``Infinity``, which spell them deterministically (a
material's open interval bound is infinite), unless the caller asks for finite values
only.

Like ``_display``, this module imports nothing from the rest of the package (NumPy
alone), so any layer may use it.
"""

import hashlib
import json

import numpy as np

__all__ = ["canonical_digest", "canonical_json", "text_digest"]


def canonical_json(value: object, *, finite: bool = False) -> str:
    """Return the one JSON text of `value` an identity or an ordering is taken over.

    ``finite=True`` refuses NaN and infinity with :class:`ValueError`, for a record
    whose every number is a measured or configured finite quantity.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=not finite,
        default=_scalar,
    )


def _scalar(value: object) -> object:
    """Return a NumPy scalar as the Python value it holds; refuse anything else."""
    if isinstance(value, np.bool_ | np.integer | np.floating):
        return value.item()
    msg = f"Object of type {type(value).__name__} is not JSON serializable"
    raise TypeError(msg)


def text_digest(text: str) -> str:
    """Return the SHA-256 of a text's UTF-8 bytes: 64 lowercase hex digits."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_digest(value: object, *, finite: bool = False) -> str:
    """Return the identity of a JSON-ready record: the digest of its canonical JSON."""
    return text_digest(canonical_json(value, finite=finite))
