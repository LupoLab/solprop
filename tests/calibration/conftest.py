# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fixtures for every calibration test."""

from __future__ import annotations

import pytest

from solprop.calibration import _native


@pytest.fixture(autouse=True)
def _fresh_store_warnings(monkeypatch: pytest.MonkeyPatch) -> None:  # pyright: ignore[reportUnusedFunction]
    """Give each test its own once-per-process store warnings.

    The store and the profile reader warn once per message per process, so a test
    expecting a warning would otherwise miss it after another test in its worker.
    """
    for module in (_native.profile, _native.store):
        monkeypatch.setattr(module, "_WARNED", set())
