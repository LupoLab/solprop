# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Select execution backends or the legacy drivers required by particular tests."""

import os
from typing import Literal

from solprop.backends.registry import available, resolve_name

__all__ = ["legacy_backends", "selected_backend"]


def legacy_backends() -> tuple[Literal["numpy_ref", "native"], ...]:
    """Return installed backends whose legacy numerical methods a test exercises.

    Session-level conformance uses the registry's general ``available()`` instead.
    """
    names: tuple[Literal["numpy_ref", "native"], ...] = ("numpy_ref", "native")
    installed = available()
    return tuple(name for name in names if name in installed)


def selected_backend() -> str:
    """Return ``SOLPROP_TEST_BACKEND``, or what ``backend="auto"`` selects when unset.

    So a lane with the native package installed runs on ``native``, as a user's run
    would, and one without it on ``numpy_ref``; a lane names a backend to override.
    """
    return os.environ.get("SOLPROP_TEST_BACKEND") or resolve_name("auto")
