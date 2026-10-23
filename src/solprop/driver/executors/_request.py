# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Carry validated scan context into a durable executor's preparation step."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from solprop.driver._scan_model import Scan


@dataclass(frozen=True)
class ScanRequest:
    """Bind point identity, output retention and the worker before dispatch.

    Attributes
    ----------
    scan : Scan
        Global Cartesian scan or indexed selection.
    settings : mapping
        Retention and reduction contract shared by all controllers.
    output : pathlib.Path or str
        Collected HDF5 destination.
    work : callable
        Pickleable point worker with a backend memory probe.
    retry_failed : bool
        Whether previously failed points may be claimed again.
    """

    scan: Scan
    settings: Mapping[str, object]
    output: Path | str
    work: Callable
    retry_failed: bool = False
