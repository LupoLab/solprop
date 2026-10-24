# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The backend the driver tests and direct ladders select: ``SOLPROP_TEST_BACKEND``."""

import os

from solprop.backends.registry import resolve_name

__all__ = ["selected_backend"]


def selected_backend() -> str:
    """Return ``SOLPROP_TEST_BACKEND``, or what ``backend="auto"`` selects when unset.

    So a lane with the native package installed runs on ``native``, as a user's run
    would, and one without it on ``numba_cpu``; a lane names a backend to override.
    """
    return os.environ.get("SOLPROP_TEST_BACKEND") or resolve_name("auto")
