# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Recipes: a simulation written as a TOML document, built into a ``Simulation``.

A recipe holds the keywords of one route's constructor, with quantities written in
their units (``"125 um"``, ``"1.2 bar"``), and the run's execution settings. It is
loaded, checked against the parameter metadata, built, run, exported as a Python script
and saved. The format is SEP 41's (``docs/dev/seps/``).
"""

__all__: list[str] = []
