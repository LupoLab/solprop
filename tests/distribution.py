# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve scientific test identities beside xdist's scheduling-only suffixes."""

import pytest

__all__ = ["original_nodeid", "remember_nodeid"]

_ORIGINAL_NODEID = pytest.StashKey[str]()


def remember_nodeid(item):
    """Retain the collected ID before a scheduler adds anything to it."""
    if _ORIGINAL_NODEID not in item.stash:
        item.stash[_ORIGINAL_NODEID] = item.nodeid


def original_nodeid(item):
    """Return the original ID, preserving parameter text including literal ``@``."""
    return item.stash.get(_ORIGINAL_NODEID, item.nodeid)
