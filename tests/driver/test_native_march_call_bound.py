# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Where a native Adams call returns moves no number (solprop SEP 39 section 6.5.7).

A native call returns at an event or at its bound, 256 attempts or 0.1 s of wall time,
so on a run with few events its return points depend on the machine's speed. The march's
state crosses each return and comes back, and a stop flag or a signal acts only there,
so a return has to be an exact no-op: every number a run stores is compared here, bit
for bit, under bounds that return at every attempt, at every seventh, at the wall
clock's every attempt, and never.

Every native run marches in the native execution session (SEP 47), whose march calls
take the session's call bound (``solprop_native.session.CALL_ATTEMPTS`` and
``CALL_SECONDS``), dense output and a transverse edge included.
"""

import dataclasses
import json
import math
from collections.abc import Callable
from typing import cast

import h5py
import numpy as np
import pytest

from backends.test_exponential_varying import EXPONENTIAL, gradient
from backends.test_transverse_backends import radial_collar
from legacy_geometry import LEGACY_GRID, bare_absorber
from solprop.backends import available
from solprop.driver import Simulation
from solprop.grid import EnvelopeGrid
from solprop.monitor import BoundaryCheck
from solprop.output.hdf5 import open_file
from solprop.stepper import FixedSchedule, Multistep
from test_split_crossing import _beam
from test_transverse_reads import REREVIEW, packet_run

pytestmark = [
    pytest.mark.skipif("native" not in available(), reason="native is not installed"),
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

NativeAdamsMarch = pytest.importorskip("solprop_native.backend")._NativeAdamsMarch
native_session = pytest.importorskip("solprop_native.session")

BOUNDS = {
    "every_attempt": (1, math.inf),
    "every_seventh": (7, math.inf),
    "clock_at_every_attempt": (256, 0.0),
    "none": (2**62, math.inf),
}
"""Each bound compared, as ``(MAX_ATTEMPTS, MAX_SECONDS)``."""

LENGTH = 0.05
"""The soliton's length, m: a few hundred steps at this order and tolerance."""

WINDOW = 4e-12
"""The soliton's time window, s: wide enough that the guards are quiet and the monitor
reads every eighth step, not every step, so a call has several attempts to bound."""

INTERACTION = Multistep(formulation="interaction")


def soliton(build_simulation, integrator=EXPONENTIAL, *, check=True, **keywords):
    """Return an N = 10 soliton with one plane, on the native march."""

    def build() -> Simulation:
        settings = {
            "order": 10,
            "length": LENGTH,
            "n_save": 2,
            "rtol": 1e-6,
            "trange": WINDOW,
            "backend": "native",
            "integrator": integrator,
        } | keywords
        simulation = build_simulation(**settings)
        if not check:
            simulation.settings = dataclasses.replace(
                simulation.settings, boundary_check=None
            )
        return simulation

    return build


def with_rows(build: Callable[[], Simulation]) -> Callable[[], Simulation]:
    """Return `build` with an energy row every third accepted step."""

    def built() -> Simulation:
        simulation = build()
        stats = simulation.settings.stats
        assert stats is not None
        simulation.settings = dataclasses.replace(
            simulation.settings,
            stats=dataclasses.replace(stats, periodic=("energy",), period=3),
        )
        return simulation

    return built


def with_settings(
    build: Callable[[], Simulation], **settings: object
) -> Callable[[], Simulation]:
    """Return `build` with `settings` replaced."""

    def built() -> Simulation:
        simulation = build()
        simulation.settings = dataclasses.replace(simulation.settings, **settings)
        return simulation

    return built


def replay(build_simulation, integrator) -> Callable[[], Simulation]:
    """Return the soliton on a uniform schedule, with no temporal collar to outrun."""
    grid = EnvelopeGrid.from_limits(
        LENGTH, 835e-9, (500e-9, 2e-6), 1e-12, **LEGACY_GRID
    )
    return soliton(
        build_simulation,
        integrator,
        check=False,
        grid=grid,
        trange=None,
        absorber=bare_absorber(grid),
        schedule=FixedSchedule.uniform(
            0.0, LENGTH, LENGTH / 1000, save_planes=[LENGTH]
        ),
    )


def _gnlse_cases(build_simulation) -> dict[str, Callable[[], Simulation]]:
    """Return the soliton's cases: both forms, monitored or not, and what returns."""
    base = soliton(build_simulation)
    interaction = soliton(build_simulation, INTERACTION)
    return {
        "exponential": base,
        "exponential_unmonitored": soliton(build_simulation, check=False),
        "interaction": interaction,
        "interaction_unmonitored": soliton(build_simulation, INTERACTION, check=False),
        "exponential_rows": with_rows(base),
        "interaction_rows": with_rows(interaction),
        # A saved plane every fifth step, the short way: each writes its field.
        "natural_stops": with_settings(
            soliton(build_simulation, length=LENGTH / 5, check=False), step_planes=5
        ),
        # A tight tolerance, and a first step far too long with no sizing of it: the
        # march starts by shrinking it, with a return between each rejection and retry.
        "rejections": with_settings(
            soliton(build_simulation, check=False, rtol=1e-9),
            dz_init=LENGTH / 4,
            auto_init=False,
        ),
        "replay": replay(build_simulation, EXPONENTIAL),
        "interaction_replay": replay(build_simulation, INTERACTION),
        "dense": with_settings(
            soliton(build_simulation, n_save=9), interpolate_planes=True
        ),
    }


def _route_cases() -> dict[str, Callable[[], Simulation]]:
    """Return the cases whose monitor carries state between the calls."""
    return {
        # A varying operator's compensated march, and the host's z-dependent bundles.
        "gradient": lambda: gradient(
            integrator=EXPONENTIAL, rtol=1e-6, backend="native"
        ),
        # The transverse edge: its reads, `read_due` and the ceiling they set.
        "transverse_edge": lambda: packet_run(
            REREVIEW,
            integrator=Multistep(),
            check=BoundaryCheck(transverse_budget=10.0),
            backend="native",
        ),
        # A requested collar's current, accumulated at every committed step.
        "requested_collar": lambda: radial_collar("native", Multistep()),
        # The split collar's crossing cap and removal, read at the checks.
        "split_collar": lambda: _beam("native", 32, 500e-15, 0.1),
    }


def _stored(path) -> dict[str, np.ndarray | str]:
    """Return every number a closed file stores, but the wall clock's and the record."""
    found: dict[str, np.ndarray | str] = {}

    def take(name: str, item: object) -> None:
        if isinstance(item, h5py.Dataset) and not (
            name.startswith("meta/") or name == "step_history/wall_time"
        ):
            found[name] = np.asarray(item)

    with h5py.File(path) as stored:
        stored.visititems(take)
    with open_file(path) as handler:
        outcome = handler.outcome
    assert outcome is not None
    assert outcome.status == "complete"
    assert outcome.execution["adams_march"] == "native"
    found["logical_calls"] = json.dumps(outcome.logical_calls, sort_keys=True)
    # The monitor's own record: what it read, removed and skipped.
    found["boundary_monitor"] = json.dumps(
        outcome.execution.get("boundary_monitor"), sort_keys=True
    )
    return found


@dataclasses.dataclass
class Calls:
    """The native session's march calls and attempts, which it counts itself."""

    session_calls: int = 0
    session_attempts: int = 0

    def __len__(self) -> int:
        """Return the native calls made."""
        return self.session_calls

    def held(self, most: int, *, exactly: bool = False) -> None:
        """Assert no call made more than `most` attempts (every call `most`, with
        `exactly`): each session call makes at least one attempt."""
        if self.session_calls:
            wanted = math.ceil(self.session_attempts / most)
            if exactly:
                assert self.session_calls == wanted
            else:
                assert self.session_calls >= wanted


def _counted_attempts(patch: pytest.MonkeyPatch, most: int, seconds: float) -> Calls:
    """Bound the session's native calls; return the record that fills with them."""
    # The session's own bound: a call's attempts and time; its advances are lifted,
    # so that only the call's bound returns it.
    patch.setattr(native_session, "CALL_ATTEMPTS", most)
    patch.setattr(native_session, "CALL_SECONDS", seconds if seconds > 0 else 1e-300)
    advance = native_session.NativeSession.advance
    calls = Calls()

    def unbounded(self, **_):
        before = self.native.status()
        try:
            return advance(self, work_limit=2**62, deadline=math.inf)
        finally:
            after = self.native.status()
            calls.session_calls += after.march_calls - before.march_calls
            calls.session_attempts += after.attempts - before.attempts

    patch.setattr(native_session.NativeSession, "advance", unbounded)
    return calls


def _assert_no_number_moved(build, tmp_path, monkeypatch) -> dict[str, Calls]:
    """Run `build` under every bound; assert the files' numbers are one set of bits.

    A return is an explicitly exact no-op (class P of ``docs/dev/testing.md``): the
    state crosses as floats and integers, unrounded, and the native state runs the same
    operations in the same order, one call or many, on one thread width.
    """
    stored = {}
    attempts = {}
    for name, (most, seconds) in BOUNDS.items():
        with monkeypatch.context() as patch:
            attempts[name] = _counted_attempts(patch, most, seconds)
            path = tmp_path / f"{name}.h5"
            build().run(path, progress="none")
            stored[name] = _stored(path)
    reference = stored["none"]
    for name, found in stored.items():
        assert found.keys() == reference.keys(), name
        for key, value in reference.items():
            np.testing.assert_array_equal(found[key], value, f"{name}: {key}")
    # Each bound held: nothing above was compared with itself by accident.
    attempts["every_attempt"].held(1, exactly=True)
    attempts["every_seventh"].held(7)
    attempts["clock_at_every_attempt"].held(1, exactly=True)
    return attempts


@pytest.mark.medium
@pytest.mark.parametrize(
    "case",
    [
        "exponential",
        "exponential_unmonitored",
        "interaction",
        "interaction_unmonitored",
        "exponential_rows",
        "interaction_rows",
        "natural_stops",
        "rejections",
        "replay",
        "interaction_replay",
        "dense",
    ],
)
def test_a_native_call_s_bound_moves_no_number_on_a_soliton(
    build_simulation, tmp_path, monkeypatch, case
):
    attempts = _assert_no_number_moved(
        _gnlse_cases(build_simulation)[case], tmp_path, monkeypatch
    )
    # The bound of one returned where the case's own events did not.
    assert len(attempts["every_attempt"]) > len(attempts["none"])
    if "unmonitored" in case:
        # One call to the end, which the bounds cut up.
        assert len(attempts["none"]) == 1
        assert len(attempts["every_seventh"]) > 1


@pytest.mark.medium
@pytest.mark.parametrize("case", sorted(_route_cases()))
def test_a_native_call_s_bound_moves_no_number_with_a_monitor_s_state(
    tmp_path, monkeypatch, case
):
    attempts = _assert_no_number_moved(_route_cases()[case], tmp_path, monkeypatch)
    assert len(attempts["every_attempt"]) >= len(attempts["none"]) > 0


def test_the_rejections_case_rejects_steps(build_simulation, tmp_path):
    # What makes the case above worth its name. Three rejections in a row restarting a
    # history that holds more than one sample is left to the scripted traces, which
    # return at every attempt too (`test_native_march_traces.py`): a real field does
    # not reach it on a case this small.
    path = tmp_path / "rejecting.h5"
    _gnlse_cases(build_simulation)["rejections"]().run(path, progress="none")
    with open_file(path) as handler:
        assert handler.outcome is not None
        rejected = cast("int", handler.outcome.logical_calls["rejected_steps"])
    assert rejected > 3


def test_the_call_bound_is_the_documented_one():
    # `docs/guide/results/resuming.md` and `docs/dev/kernel-contract.md` state these.
    assert (NativeAdamsMarch.MAX_ATTEMPTS, NativeAdamsMarch.MAX_SECONDS) == (256, 0.1)
