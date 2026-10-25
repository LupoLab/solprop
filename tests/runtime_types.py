# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Whether this test session checks array annotations at runtime (ADR-053).

``tests/conftest.py`` installs the jaxtyping and beartype import hook unless
``SOLPROP_TEST_TYPECHECK=0``. The checks cost a call-binding and a shape match on every
call of an annotated function, which in the reference backend's inner loops is most of
a test's time, so CI keeps them in one lane and leaves them out of the others.
"""

import os

import pytest

__all__ = ["TYPECHECKED", "requires_typecheck", "unchecked"]

TYPECHECKED = os.environ.get("SOLPROP_TEST_TYPECHECK", "1").strip() != "0"

requires_typecheck = pytest.mark.skipif(
    not TYPECHECKED, reason="the runtime type checks are off (SOLPROP_TEST_TYPECHECK=0)"
)
"""Skip a test that asserts what the import hook raises, when it is not installed."""


def unchecked(function):
    """Return `function` past the import hook's checks, as production calls it.

    The hook wraps every annotated function and keeps the original as ``__wrapped__``;
    without the hook there is nothing to unwrap.
    """
    return getattr(function, "__wrapped__", function)
