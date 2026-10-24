# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The element-level sequence the transport tests run on a plain state.

It lives outside ``conftest.py`` because pytest discourages importing a conftest as a
module.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from solprop.coupling.elements import SpatialState
from solprop.coupling.transport import TransportReport, _sequence


@dataclass(frozen=True)
class Applied:
    """The state after a sequence and its ledger, for the element-level tests."""

    state: SpatialState
    report: TransportReport


def apply_sequence(state: SpatialState, elements: Sequence[object]) -> Applied:
    """Run a sequence on a plain state: the core that transport() wraps."""
    final, report = _sequence(state, elements)
    return Applied(final, report)
