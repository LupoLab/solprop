# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's own boundary monitor against solprop's, on recorded readings.

SEP 47 sections 4.3 and 6.4: the native session owns the boundary monitor
(``solprop_native._core.Monitor``, ``watch::Monitor``), and this harness replaces
``_verify_watch_of``'s runtime replay. Both monitors are fed the same recorded readings
-- the launch, checked and skipped steps, saved planes, a resting collar, a resume --
and must make exactly the same decisions: the cadence (``due``), each verdict, its
warning or stop and the amplitude its message names, the quiet collar and the split
collar's cap, so the proposals that follow are the same. Their records must agree too:
exactly where the arithmetic is the same scalar sequence, and within a bound derived
below where a reading's reduction is summed in another order.

The one such reduction is the input floor's share of a split step's removal,
``sum_t (1 - exp(-alpha(t) h))`` over the ``n_t`` samples: NumPy sums it pairwise, the
native monitor in order. Each term is the same libm ``expm1``, nonnegative, so the two
sums differ by at most ``(n_t - 1 + ceil(log2 n_t)) u`` of either (Higham, *Accuracy and
Stability of Numerical Algorithms*, 2nd ed., section 4.2). Every later operation on it
is the same in both and adds at most ``u`` per operation on positive terms: over ``k``
steps, the floor's removal differs by at most ``(n_t + log2 n_t + 2 k + 4) u`` of
itself, and so does every removal it enters.
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

NONE, STOP, WARN, QUIET, REFUSE, ERROR = range(6)
U = 2.0**-53
"""The unit roundoff of binary64."""


def _modules() -> tuple[Any, Any]:
    return (
        importlib.import_module("solprop_native._core"),
        importlib.import_module("solprop_native._watch"),
    )


def _grid() -> EnvelopeGrid:
    return EnvelopeGrid.from_limits(1.0, 1e-6, (0.8e-6, 1.25e-6), 1e-12)


def _python(
    *, split: bool, quiet: bool, budget: float = 1e-3, crossing: bool = False
) -> BoundaryMonitor:
    """Return a fresh monitor on a small envelope grid with a collar."""
    grid = _grid()
    rate = np.where(np.abs(grid.t) > 0.4 * grid.t.max(), 1e14, 0.0)
    monitor = BoundaryMonitor.build(
        grid, rate, BoundaryCheck(budget=budget, quiet_collar=quiet), split=split
    )
    if crossing:
        buckets = np.zeros((1, grid.n_omega), dtype=np.uint8)
        buckets[0, :4] = np.arange(4)
        monitor.crossing = SplitCrossing(buckets, 1e-13, launch_limit=0.05)
    return monitor


class Pair:
    """solprop's monitor and the native one, fed the same readings."""

    def __init__(self, monitor: BoundaryMonitor, *, can_rest: bool) -> None:
        core, watch = _modules()
        self.python = monitor
        self.watch = watch
        self.native = core.Monitor(
            watch.monitor_config(monitor, monitor.crossing, can_rest=can_rest)
        )
        self.can_rest = can_rest
        self.steps = 0

    def reading(self, energies: BoundaryEnergies) -> np.ndarray:
        """Return a reading laid out as the native state's."""
        head = np.zeros(10)
        head[0] = energies.total
        head[1:5] = energies.regions
        head[5] = energies.removed
        if energies.found is not None:
            head[6:10] = energies.found
        if self.python.crossing is None:
            return head
        assert energies.occupancy is not None
        return np.concatenate([head, energies.occupancy])

    def _python_call(self, call: Any) -> tuple[int, str | None]:
        """Run a call on solprop's monitor; return its verdict and message."""
        warned = self.python.watch_state()["warned"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", BoundaryWarning)
            try:
                call()
            except BoundaryError as error:
                return STOP, str(error)
            except BackendError:
                return ERROR, None
        messages = [
            str(one.message)
            for one in caught
            if issubclass(one.category, BoundaryWarning)
        ]
        if messages and not warned:
            return WARN, messages[0]
        return NONE, None

    def _agree(self, verdict: int, message: str | None, judgement: dict) -> int:
        native = int(judgement["verdict"])
        assert native == verdict
        check = self.python.check
        if verdict in (STOP, WARN):
            # The amplitude the message names is the same, digit for digit.
            z = None if judgement["launch"] else judgement["z"]
            assert message == self.watch._guard_message(
                check, judgement["worst"], z, stop=verdict == STOP
            )
        if verdict in (STOP, ERROR):
            # solprop's monitor raised part-way; only the cap, read first, is promised.
            assert self.native.step_cap == self.python.step_cap
            return verdict
        self.state()
        return verdict

    def state(self) -> None:
        """Require the same decision state: cadence, quiet collar, cap, warning."""
        assert dict(self.native.watch_state()) == self.python.watch_state()
        assert self.native.stale == self.python.stale
        assert self.native.steps_to_check() == self.python.steps_to_check()
        assert self.native.collar_quiet == self.python.collar_quiet

    def collar(self) -> None:
        """Rest the collar where it is quiet and the route can rest it, on both."""
        rest = self.python.collar_quiet and self.can_rest
        assert self.native.rests == rest
        self.python.note_collar(rest)
        self.native.note_collar(rest)

    def resume(self, record: dict | None, *, plane: bool, unwatched=()) -> None:
        self.python.resume(record, plane=plane, unwatched=unwatched)
        self.native.resume(
            self.watch.resumption(record), plane, self.watch.unwatched_of(unwatched)
        )

    def launch(self, energies: BoundaryEnergies, z: float, floor=None) -> int:
        verdict, message = self._python_call(
            lambda: self.python.launch(energies, z, floor=floor)
        )
        regions = None
        energy = 0.0
        if floor is not None:
            # The floor is a reading of the noise draw; both take the same one.
            energy = self.python._floor
            regions = np.ascontiguousarray(self.python._floor_regions)
        judgement = self.native.launch(self.reading(energies), z, energy, regions)
        verdict = self._agree(verdict, message, judgement)
        if verdict not in (STOP, ERROR):
            self.collar()
        return verdict

    def due(self, z: float) -> bool:
        due = self.python.due(z)
        assert self.native.due(z) == due
        return due

    def skip(self, z: float) -> None:
        self.python.skip(z)
        self.native.skip(z)
        self.steps += 1
        self.state()

    def step(self, energies: BoundaryEnergies, z: float, *, plane=False) -> int:
        verdict, message = self._python_call(
            lambda: self.python.step(energies, z, plane=plane)
        )
        judgement = self.native.step(self.reading(energies), z, plane)
        if not plane:
            self.steps += 1
        verdict = self._agree(verdict, message, judgement)
        if verdict not in (STOP, ERROR):
            self.collar()
        return verdict

    def plane(self, z: float) -> None:
        self.python.plane(z)
        self.native.plane(z)

    def records(self, *, floor_steps: int | None = None) -> None:
        """Require the same record: exactly, or within the floor's bound (above)."""
        expected: dict[str, Any] = self.python.summary()
        got: dict[str, Any] = self.watch.record_of(
            self.python.check, self.native.record()
        )
        assert list(got) == list(expected)
        loose = (
            set()
            if floor_steps is None
            else {"removed_fraction", "removed_signed", "floor_removed_fraction"}
        )
        n_t = self.python.weights.shape[1]
        delta = (n_t + math.ceil(math.log2(n_t)) + 2 * (floor_steps or 0) + 4) * U
        scale = abs(float(expected["floor_removed_fraction"]))
        for name, value in expected.items():
            if name in loose:
                assert got[name] == pytest.approx(value, rel=0.0, abs=delta * scale)
            elif name == "planes":
                rows = got[name]
                for column, values in value.items():
                    if column == "removed_fraction" and loose:
                        np.testing.assert_allclose(
                            np.asarray(rows[column], dtype=float),
                            np.asarray(values, dtype=float),
                            rtol=0.0,
                            atol=delta * scale,
                        )
                    else:
                        assert rows[column] == values, column
            else:
                assert got[name] == value, name

    def finish(self) -> None:
        """Require the same finish verdict and message."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", BoundaryWarning)
            self.python.finish()
        finish = self.native.finish()
        messages = [str(one.message) for one in caught]
        assert bool(messages) == finish["warns"]
        if messages:
            assert messages[0] == self.watch._removal_message(
                self.python.check, finish["removed"]
            )


def _energies(
    monitor: BoundaryMonitor,
    total: float,
    shares: list[float],
    found: float,
    removed: float = 0.0,
) -> BoundaryEnergies:
    """Return a reading with the given total and region shares of it."""
    regions = np.array(shares, dtype=np.float64) * total
    occupancy = None
    if monitor.crossing is not None:
        occupancy = np.geomspace(1.0, 10.0 ** (-9 * found), SPLIT_RATE_BUCKETS) * total
    return BoundaryEnergies(
        total,
        regions,
        removed=removed * total if monitor.split else 0.0,
        found=np.array([found * total, *(regions[:3] * found)])
        if monitor.split
        else None,
        occupancy=occupancy,
    )


def _launch(pair: Pair, *, floor: np.ndarray | None = None) -> int:
    """Launch a centred pulse, its reading the host reduction's."""
    grid = _grid()
    pulse = np.exp(-((grid.t / 20e-15) ** 2)).astype(complex)
    energies = pair.python.energies(np.fft.fft(pulse)[None, None, None, :])
    if pair.python.split:
        energies = BoundaryEnergies(
            energies.total,
            energies.regions,
            found=np.zeros(4),
            occupancy=np.full(SPLIT_RATE_BUCKETS, energies.total / SPLIT_RATE_BUCKETS)
            if pair.python.crossing is not None
            else None,
        )
    return pair.launch(energies, 0.0, floor=floor)


SHARE = st.one_of(
    st.just(0.0),
    st.floats(1e-20, 1e-1, allow_nan=False),
    # Around the activity, warning and stop thresholds of the default budget, and the
    # quiet collar's.
    st.sampled_from([1e-8, 1e-6, 1e-4, 1e-12]).map(lambda value: value * (1.0 + 1e-15)),
)
STEP = st.tuples(
    st.floats(1e-6, 2e-2),
    st.lists(SHARE, min_size=4, max_size=4),
    st.sampled_from([1.0, 0.5, 1.0 + 1e-9]),
    st.booleans(),
    # Light, none at all, and less than the launch floor: the no-light path.
    st.sampled_from([1.0, 1.0, 1.0, 0.0, 1e-300]),
    # A saved plane after the step: none, or one.
    st.sampled_from([False, False, True]),
    st.floats(0.0, 1e-3),
)


def _run(pair: Pair, steps: list, z: float = 0.0) -> tuple[float, bool]:
    """Feed a drawn sequence to both; return the position and whether it ended."""
    for dz, shares, found, check, total, plane, removed in steps:
        z += dz
        energies = _energies(pair.python, total, shares, found, removed)
        # A step the draw leaves unchecked is skipped, unless the cadence makes it due.
        if not pair.due(z) and not check:
            pair.skip(z)
        elif pair.step(energies, z) in (STOP, ERROR):
            return z, True
        if plane:
            if pair.python.stale and pair.step(energies, z, plane=True) in (
                STOP,
                ERROR,
            ):
                return z, True
            pair.plane(z)
    return z, False


@settings(max_examples=60, deadline=None)
@given(
    split=st.booleans(),
    quiet=st.booleans(),
    can_rest=st.booleans(),
    crossing=st.booleans(),
    steps=st.lists(STEP, min_size=1, max_size=16),
)
def test_the_owner_decides_and_records_as_solprop_does(
    split, quiet, can_rest, crossing, steps
):
    pair = Pair(
        _python(split=split, quiet=quiet, crossing=split and crossing),
        can_rest=can_rest,
    )
    if _launch(pair) == STOP:
        return
    pair.plane(0.0)
    _, ended = _run(pair, steps)
    pair.records()
    if not ended:
        pair.finish()


@settings(max_examples=30, deadline=None)
@given(
    split=st.booleans(),
    quiet=st.booleans(),
    first=st.lists(STEP, min_size=1, max_size=10),
    then=st.lists(STEP, min_size=1, max_size=10),
    plane=st.booleans(),
    closing=st.booleans(),
)
def test_a_resume_continues_the_record_as_solprop_does(
    split, quiet, first, then, plane, closing
):
    # The first execution's record is solprop's own, as a run's file keeps it; a
    # running record (written at a plane) carries no rows.
    earlier = Pair(_python(split=split, quiet=quiet), can_rest=True)
    if _launch(earlier) == STOP:
        return
    z, ended = _run(earlier, first)
    if ended:
        return
    earlier.plane(z)
    record = earlier.python.summary()
    if not closing:
        del record["planes"]
    pair = Pair(_python(split=split, quiet=quiet), can_rest=True)
    pair.resume(record, plane=plane, unwatched=[(z, z + 1e-3), (z, z)])
    grid = _grid()
    pulse = np.exp(-((grid.t / 20e-15) ** 2)).astype(complex)
    energies = pair.python.energies(np.fft.fft(pulse)[None, None, None, :])
    pair.launch(energies, z)
    _, ended = _run(pair, then, z)
    pair.records()
    if not ended:
        pair.finish()


def test_a_resume_without_a_record_starts_again():
    pair = Pair(_python(split=False, quiet=False), can_rest=False)
    pair.resume(None, plane=True, unwatched=[(0.0, 0.1)])
    _launch(pair)
    pair.records()


def test_an_old_record_s_floor_removal_is_reconstructed():
    # A record from before the floor's removal and the signed removal were kept.
    earlier = Pair(_python(split=False, quiet=False), can_rest=False)
    _launch(earlier)
    record = earlier.python.summary()
    for name in ("floor_removed_fraction", "removed_signed", "quiet_checks"):
        del record[name]
    record["removed_fraction"] = 1e-9
    record["raw_removed_fraction"] = 3e-9
    pair = Pair(_python(split=False, quiet=False), can_rest=False)
    pair.resume(record, plane=False)
    _launch(pair)
    pair.records()


@pytest.mark.parametrize("split", [False, True])
def test_the_thresholds_and_their_neighbours_agree(split):
    # The warning and stop thresholds exactly and at the doubles beside them, warned
    # and unwarned: compared on the amplitude, as `_judge` compares.
    for budget in (1e-3, 3e-3):
        for factor in (1.0, 10.0):
            edge = (factor * budget) ** 2
            for share in (math.nextafter(edge, 0.0), edge, math.nextafter(edge, 1.0)):
                for warned in (False, True):
                    pair = Pair(
                        _python(split=split, quiet=False, budget=budget), can_rest=False
                    )
                    _launch(pair)
                    if warned:
                        first = [(2 * budget) ** 2, 0.0, 0.0, 0.0]
                        energies = _energies(pair.python, 1.0, first, 1.0)
                        assert pair.step(energies, 1e-3) == WARN
                    energies = _energies(pair.python, 1.0, [share, 0, 0, 0], 1.0)
                    pair.step(energies, 2e-3)
                    pair.records()


def test_a_launch_in_the_guard_is_judged_alike():
    for share, verdict in ((4e-6, WARN), (4e-4, STOP), (0.0, NONE)):
        pair = Pair(_python(split=False, quiet=False), can_rest=False)
        energies = _energies(pair.python, 1.0, [share, 0.0, 1e-3, 0.0], 1.0)
        assert pair.launch(energies, 0.0) == verdict
        pair.records()


def test_a_split_that_found_nothing_is_an_error_alike():
    pair = Pair(_python(split=True, quiet=False), can_rest=False)
    _launch(pair)
    energies = _energies(pair.python, 1.0, [0.0, 0.0, 0.0, 0.0], 0.0)
    assert pair.step(energies, 1e-3) == ERROR


def test_a_resting_collar_adds_no_removal_alike():
    pair = Pair(_python(split=False, quiet=True), can_rest=True)
    _launch(pair)
    calm = _energies(pair.python, 1.0, [0.0, 0.0, 0.0, 1e-3], 1.0)
    for k in range(1, 5):
        pair.step(calm, 0.01 * k)
    assert pair.native.rests
    near = _energies(pair.python, 1.0, [0.0, 0.0, 1e-6, 1e-3], 1.0)
    pair.step(near, 0.05)
    assert not pair.native.rests
    pair.step(calm, 0.06)
    pair.plane(0.06)
    pair.records()
    assert pair.native.record()["collar_left_out_checks"] > 0


@settings(max_examples=20, deadline=None)
@given(steps=st.lists(STEP, min_size=1, max_size=12), seed=st.integers(0, 2**16))
def test_the_floor_s_split_removal_is_within_its_bound(steps, seed):
    # An input floor under the split scheme: the one reading summed in another order.
    pair = Pair(_python(split=True, quiet=False), can_rest=False)
    grid = _grid()
    rng = np.random.default_rng(seed)
    floor = 1e-6 * (
        rng.standard_normal(grid.n_omega) + 1j * rng.standard_normal(grid.n_omega)
    )
    if _launch(pair, floor=floor[None, None, None, :]) == STOP:
        return
    _run(pair, steps)
    pair.records(floor_steps=pair.steps)
