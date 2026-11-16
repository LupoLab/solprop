# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native backend's per-stage evaluations, for solprop's shared code.

solprop's shared stage code takes a stage table's Chebyshev rows from the backend that
runs it (``NativeBackend.helpers``, solprop SEP 29 section 6.6), so that a native run
never reaches solprop's Python forms and the NumPy reference never reaches this
package. :data:`NATIVE` is this backend's record: the C++ transcription in
``include/solprop_native/chebyshev.hpp``, which performs solprop's operations in its
order under strict arithmetic.

The record names no Adams basis. The native Adams march is compiled and forms its
Lagrange basis and coefficient tables in C++ (``lagrange_basis.hpp``) every trial, so no
production path evaluates them in Python (solprop SEP 52); their bindings are private
test entry points in ``_core._testing``, which check them against solprop's plain forms.
"""

from __future__ import annotations

from solprop._helpers import Helpers, PointsFunction

from . import _core

__all__ = ["NATIVE"]


def _load_points() -> PointsFunction:
    """Return the native Chebyshev rows binding."""
    return _core.clenshaw_points


NATIVE = Helpers("native", _load_points)
"""The native backend's helper record (``NativeBackend.helpers``)."""
