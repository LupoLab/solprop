# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Select the historical gated plasma equation privately, for labelled comparisons.

Construct, translate, record and run a new simulation inside ``_gated_source()``.
Never reuse that simulation outside its context. Its configuration is explicitly
labelled and hashed, and its files refuse continuation through either resume API.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from solprop.translate.plasma_source import _BENCHMARK_GATED

__all__: list[str] = []


@contextmanager
def _gated_source() -> Iterator[None]:
    """Reproduce the old equation during one complete, private benchmark run."""
    token = _BENCHMARK_GATED.set(True)
    try:
        yield
    finally:
        _BENCHMARK_GATED.reset(token)
