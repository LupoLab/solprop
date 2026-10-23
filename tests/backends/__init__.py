# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the execution backends.

This package marker keeps the fixtures in ``conftest.py`` from colliding with the
session-wide ``tests/conftest.py``: ``--doctest-modules`` collects every module under
``testpaths``, and two files with the same basename cannot both be imported as top-level
modules in pytest's prepend import mode.
"""
