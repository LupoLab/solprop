# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The lifecycle conformance suite every execution session passes (SEP 47 section 7.1).

This package marker keeps ``conftest.py`` here from colliding with the session-wide
``tests/conftest.py`` under pytest's prepend import mode.
"""
