# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native boundary watch against solprop's monitor (SEP 45 section 7.2).

A native march predicts, at each check it takes, what solprop's
:class:`~solprop.monitor.BoundaryMonitor` will make of the reading: the split collar's
new cap, the cadence, the quiet collar's verdict, a first warning or a stop. These
tests hand the same readings to a real monitor and to the native predictions
(``solprop_native._core.watch_check``) and require the same verdict and the same state,
field by field, on drawn readings around every threshold and on the review's witnesses.
"""

from __future__ import annotations

import importlib
import math
import warnings
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from solprop.backends.registry import available
from solprop.boundaries import SPLIT_RATE_BUCKETS, SplitCrossing
from solprop.exceptions import BackendError, BoundaryError, BoundaryWarning
from solprop.grid import EnvelopeGrid
from solprop.monitor import BoundaryCheck, BoundaryEnergies, BoundaryMonitor

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="the native backend is not installed"
)

STOP, WARN, QUIET, ERROR = 1, 2, 3, 5


def _core() -> Any:
    return importlib.import_module("solprop_native._core")


def _monitor(*, split: bool, quiet: bool, budget: float = 1e-3) -> BoundaryMonitor:
    """Return a launched monitor on a small envelope grid."""
    grid = EnvelopeGrid.from_limits(1.0, 1e-6, (0.8e-6, 1.25e-6), 1e-12)
    rate = np.where(np.abs(grid.t) > 0.4 * grid.t.max(), 1e14, 0.0)
    monitor = BoundaryMonitor.build(
        grid,
        rate,
        BoundaryCheck(budget=budget, quiet_collar=quiet),
        split=split,
    )
    if split:
        buckets = np.zeros((1, grid.n_omega), dtype=np.uint8)
        buckets[0, :4] = np.arange(4)
        monitor.crossing = SplitCrossing(buckets, 1e-13, launch_limit=0.05)
    pulse = np.exp(-((grid.t / 20e-15) ** 2)).astype(complex)
    spectrum = np.fft.fft(pulse)[None, None, None, :]
    energies = monitor.energies(spectrum)
    if split:
        energies = BoundaryEnergies(
            energies.total,
            energies.regions,
            found=np.array([energies.total, *energies.regions[:3]]),
            occupancy=np.full(SPLIT_RATE_BUCKETS, energies.total / SPLIT_RATE_BUCKETS),
        )
    monitor.launch(energies, 0.0)
    return monitor


def _reading(energies: BoundaryEnergies, monitor: BoundaryMonitor) -> np.ndarray:
    """Return a reading laid out as the native one is."""
    head = np.zeros(10)
    head[0] = energies.total
    head[1:5] = energies.regions
    head[5] = energies.removed
    if energies.found is not None:
        head[6:10] = energies.found
    if monitor.crossing is None:
        return head
    assert energies.occupancy is not None
    return np.concatenate([head, energies.occupancy])


def _python(monitor: BoundaryMonitor, energies: BoundaryEnergies, z: float) -> int:
    """Return the verdict solprop's monitor reaches on a check."""
    quiet = monitor.collar_quiet
    warned = monitor.watch_state()["warned"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", BoundaryWarning)
        try:
            monitor.step(energies, z)
        except BoundaryError:
            return STOP
        except BackendError:
            return ERROR
    if any(issubclass(one.category, BoundaryWarning) for one in caught) and not warned:
        return WARN
    return QUIET if monitor.collar_quiet != quiet else 0


def _agree(monitor: BoundaryMonitor, energies: BoundaryEnergies, z: float) -> int:
    """Check one reading on both, require the same verdict and state; return it."""
    core = _core()
    policy = monitor.watch_policy()
    before = monitor.watch_state()
    assert core.watch_due(policy, before, z) == monitor.due(z)
    native = core.watch_check(policy, before, _reading(energies, monitor), z)
    verdict = _python(monitor, energies, z)
    assert native["verdict"] == verdict
    if verdict in (STOP, ERROR):
        # The monitor raised part-way; only the cap, read first, is promised.
        assert native["state"]["cap"] == monitor.watch_state()["cap"]
        return verdict
    assert dict(native["state"]) == monitor.watch_state()
    return verdict


def _energies(
    monitor: BoundaryMonitor, total: float, shares: list[float], found: float
) -> BoundaryEnergies:
    """Return a reading with the given total and region shares of it."""
    regions = np.array(shares, dtype=np.float64) * total
    split = monitor.split
    occupancy = None
    if monitor.crossing is not None:
        occupancy = np.geomspace(1.0, 1e-9, SPLIT_RATE_BUCKETS) * total
    return BoundaryEnergies(
        total,
        regions,
        found=np.array([found * total, *(regions[:3] * found)]) if split else None,
        occupancy=occupancy,
    )


SHARE = st.one_of(
    st.just(0.0),
    st.floats(1e-20, 1e-1, allow_nan=False),
    # Around the activity, warning and stop thresholds of the default budget.
    st.sampled_from([1e-8, 1e-6, 1e-4, 1e-12]).map(lambda value: value * (1.0 + 1e-15)),
)


@settings(max_examples=60, deadline=None)
@given(
    split=st.booleans(),
    quiet=st.booleans(),
    steps=st.lists(
        st.tuples(
            st.floats(1e-6, 2e-2),
            st.lists(SHARE, min_size=4, max_size=4),
            st.sampled_from([1.0, 0.5, 1.0 + 1e-9]),
            st.booleans(),
            # Light, none at all, and less than the launch floor: the no-light path.
            st.sampled_from([1.0, 1.0, 1.0, 0.0, 1e-300]),
        ),
        min_size=1,
        max_size=12,
    ),
)
def test_the_native_predictions_are_the_monitor_s(split, quiet, steps):
    monitor = _monitor(split=split, quiet=quiet)
    z = 0.0
    for dz, shares, found, check, total in steps:
        z += dz
        # A step the draw leaves unchecked is skipped, unless the cadence makes it due.
        if not check and not monitor.due(z):
            monitor.skip(z)
            continue
        verdict = _agree(monitor, _energies(monitor, total, shares, found), z)
        if verdict in (STOP, ERROR):
            # The monitor raised part-way: the run ends here.
            return


@pytest.mark.parametrize("split", [False, True])
def test_the_thresholds_and_their_neighbours_agree(split):
    # The warning and stop thresholds exactly and at the doubles beside them, warned
    # and unwarned: compared on the amplitude, as `_judge` compares.
    for budget in (1e-3, 3e-3):
        for factor in (1.0, 10.0):
            edge = (factor * budget) ** 2
            for share in (math.nextafter(edge, 0.0), edge, math.nextafter(edge, 1.0)):
                for warned in (False, True):
                    monitor = _monitor(split=split, quiet=False, budget=budget)
                    if warned:
                        # An amplitude of twice the budget: a warning, not a stop.
                        first = [(2 * budget) ** 2, 0.0, 0.0, 0.0]
                        assert (
                            _agree(monitor, _energies(monitor, 1.0, first, 1.0), 1e-3)
                            == WARN
                        )
                    _agree(
                        monitor, _energies(monitor, 1.0, [share, 0, 0, 0], 1.0), 2e-3
                    )


def test_a_split_that_found_nothing_is_predicted():
    monitor = _monitor(split=True, quiet=False)
    _agree(monitor, _energies(monitor, 1.0, [0.0, 0.0, 0.0, 0.0], 0.0), 1e-3)


def test_the_split_limit_is_the_crossing_s():
    monitor = _monitor(split=True, quiet=False)
    crossing = monitor.crossing
    assert crossing is not None
    policy = monitor.watch_policy()
    occupancy = np.full(SPLIT_RATE_BUCKETS, 2.0**-54)
    occupancy[0] = float.fromhex("0x1.0c6f8ba2f9813p-20")
    occupancy[1] = 1.0
    for case in (occupancy, np.zeros(SPLIT_RATE_BUCKETS), np.ones(SPLIT_RATE_BUCKETS)):
        assert _core().watch_split_limit(policy, case) == crossing.limit(case)
