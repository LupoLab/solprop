# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Collect bounded, independently qualified benchmarks (SEP 30).

Importing the registry or planner never imports a scientific library. Workers
load those libraries only after their thread environment has been established.
"""

__all__: list[str] = []
