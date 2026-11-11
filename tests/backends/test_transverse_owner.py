# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The native session's own transverse edge against solprop's, on recorded readings.

SEP 47 sections 4.3 and 6.4 and the maintainer's decision 4 at its acceptance: the
native session owns a free-space run's transverse edge (``transverse::Edge``,
``solprop_native._core.TransverseEdge``) with every decision and record. Both edges are
fed the same recorded readings -- the launch with its clip, noise and schedule, reads
on and off their spacing, a collar's current at every step, a plane re-read, a resume
-- and must make exactly the same decisions: every warning and stop, in order and in
the same words, the read spacing and its ceiling, when a read is due and wanted, and
the status. Their records must be the same, number for number.

Three reductions belong to the reading (``transverse.hpp``): the launch's energy in the
safety ramp's band and its total, and the ramp's removal over a read interval,
``sum_c P_c (1 - exp(-alpha_c h))``. Here both edges take solprop's values of them, as
recorded readings, so the comparison is exact. The native edge's own values are held to
NumPy's in :func:`test_the_native_ramp_readings_are_within_their_bound`. Everything else
the edge computes from a reading -- the weighted columns, the reach's tail sums over
the columns as solprop sorted them, the worst face with its ties broken on the name --
is the same scalar sequence in both, and compared exactly on arbitrary readings.
"""

from __future__ import annotations

import importlib
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from solprop.backends.registry import available
from solprop.boundaries import SpatialAbsorber, SpatialAbsorberDesign
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.translate.transverse import _nodes_window
from solprop.transverse_monitor import TransverseEdge, _EdgeReading

pytestmark = pytest.mark.skipif(
    "native" not in available(), reason="the native backend is not installed"
)

K800 = 2 * math.pi / 800e-9
"""The band's smallest wavenumber, rad/m: one frequency at 800 nm."""

U = 2.0**-53
"""The unit roundoff of binary64."""


def _modules() -> tuple[Any, Any]:
    return (
        importlib.import_module("solprop_native._core"),
        importlib.import_module("solprop_native._transverse"),
    )


def _space(geometry: str) -> LineGrid | FreeGrid | RadialGrid:
    """Return a small grid of each kind of edge: two faces, four, or one."""
    if geometry == "line":
        return LineGrid(200e-6, 64, collar_fraction=0.5)
    if geometry == "box":
        return FreeGrid(200e-6, 16, 200e-6, 16, collar_fraction=0.5)
    return RadialGrid(200e-6, 24)


def _design(
    space: LineGrid | FreeGrid | RadialGrid, kind: str
) -> SpatialAbsorberDesign | None:
    """Return no boundary, a safety ramp on the top tenth of the wavenumbers, or a
    requested collar of 1e3 1/m outside the requested size."""
    if kind == "none":
        return None
    columns = int(np.asarray(space.spectral_weights).size)
    if kind == "ramp":
        if isinstance(space, RadialGrid):
            kappa = np.asarray(space.k_r)
        else:
            kappa = np.sqrt(np.asarray(space.kperp_sq))
        ramp = np.where(kappa > 0.9 * float(np.max(kappa)), 1e4, 0.0)
        return SpatialAbsorberDesign(
            SpatialAbsorber(real_space=False), ramp, 1e4, (0.9 * float(np.max(kappa)),)
        )
    assert not isinstance(space, RadialGrid)
    return SpatialAbsorberDesign(
        SpatialAbsorber(kappa=3e5),
        np.zeros(columns),
        1.0,
        (1e6,),
        collar_rate=1e3 * (1 - _nodes_window(space)),
        alpha0=1e3,
        width=(50e-6,),
        kappa_p=(3e5,),
    )


@dataclass
class Readings:
    """The arrays a drawn reading is made of, scaled by the launch energy."""

    faces: list[float]
    clip: list[float]
    collar: float
    outside: float
    columns: list[float]
    strip: list[float]
    current: float


class Pair:
    """solprop's edge and the native one, fed the same readings."""

    def __init__(
        self,
        geometry: str,
        design: str,
        *,
        budget: float = 1e-3,
        steps: np.ndarray | None = None,
        declared: bool = False,
        reads: bool = True,
    ) -> None:
        core, transverse = _modules()
        self.transverse = transverse
        space = _space(geometry)
        self.python = TransverseEdge(
            space, _design(space, design), budget, 10.0, 0.01, np.arange(1), K800
        )
        self.native = core.TransverseEdge(
            transverse.config_of(
                self.python, steps=steps, declared=declared, reads=reads
            ).native()
        )
        self.steps = steps
        self.declared = declared
        self.reads = reads
        self.lines = len(self.python.reduction.lines)
        self.columns = int(np.asarray(space.spectral_weights).size)
        self.launch_energy = 1.0

    # -- readings -----------------------------------------------------------------

    def reading(
        self, drawn: Readings, *, full: bool, faces: bool = True, current: bool = False
    ) -> tuple[_EdgeReading, dict[str, object]]:
        """Return a drawn reading as each edge takes it, laid out as a backend's."""
        energy = self.launch_energy
        collar = self.python.every_step
        columns = np.resize(np.asarray(drawn.columns), self.columns) * energy
        totals = np.array([energy, drawn.collar * energy, drawn.outside * energy, 0.0])
        strip = np.resize(np.asarray(drawn.strip), 2 * self.lines) * energy
        rate = drawn.current * energy if current else None
        if not faces:
            return _EdgeReading(current=rate), {"current": rate}
        values = {
            "faces": np.asarray(drawn.faces, dtype=np.float64) * energy,
            "columns": np.ascontiguousarray(columns),
            "clip": np.asarray(drawn.clip, dtype=np.float64) * energy if full else None,
            "totals": totals if full or collar else None,
            "strip": np.ascontiguousarray(strip) if collar else None,
            "current": rate,
        }
        python = _EdgeReading(
            faces=values["faces"],
            clip=values["clip"],
            totals=values["totals"],
            columns=values["columns"],
            strip=strip.reshape(-1, 2) if collar else None,
            current=rate,
        )
        return python, values

    def launch_sums(self, columns: np.ndarray) -> tuple[float, float] | None:
        """Return solprop's reading of the launch's energy in the ramp's band."""
        design = self.python.design
        if design is None:
            return None
        power = self.python._column_power(columns)  # pyright: ignore[reportPrivateUsage]
        return float(np.sum(power)), float(np.sum(power[design.ramp_rate > 0.0]))

    def read_sums(self, z: float) -> tuple[float, float] | None:
        """Return solprop's reading of the ramp's removal over the interval to `z`."""
        python = self.python
        design = python.design
        last = python._last_columns  # pyright: ignore[reportPrivateUsage]
        if design is None or last is None or not python.wants_read(z):
            return None
        gap = z - python._last_z  # pyright: ignore[reportPrivateUsage]
        taken = -np.expm1(-design.ramp_rate * gap)
        noise = python._noise_columns  # pyright: ignore[reportPrivateUsage]
        return (
            float(np.sum(last * taken)),
            0.0 if noise is None else float(np.sum(noise * taken)),
        )

    # -- calls ----------------------------------------------------------------------

    @staticmethod
    def _python_call(call: Any) -> list[tuple[str, str]]:
        """Run a call on solprop's edge; return its warnings and stop, in order."""
        stop = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", BoundaryWarning)
            try:
                call()
            except BoundaryError as error:
                stop = str(error)
        out = [
            ("warn", str(one.message))
            for one in caught
            if issubclass(one.category, BoundaryWarning)
        ]
        if stop is not None:
            out.append(("stop", stop))
        return out

    def _native_call(self, first: dict[str, object]) -> list[tuple[str, str]]:
        """Serve a native call's notices, each warning then the rest of its point."""
        out: list[tuple[str, str]] = []
        notice = self.transverse.EdgeNotice.of(first)
        while notice.kind != self.transverse.NoticeKind.NONE:
            if notice.stop:
                out.append(
                    ("stop", str(self.transverse.notice_error(self.python, notice)))
                )
                assert not self.native.pending
                break
            warning = self.transverse.notice_warning(self.python, notice)
            assert type(warning) is BoundaryWarning
            out.append(("warn", str(warning)))
            notice = self.transverse.EdgeNotice.of(self.native.proceed())
        return out

    def state(self, z: float, proposal: float) -> None:
        """Require the same decision state: the spacing, the next read and status."""
        python, native = self.python, self.native
        assert native.next_read == python.next_read
        assert native.ceiling() == python.ceiling(z)
        assert native.every_step == python.every_step
        assert native.resumed == python.resumed
        assert self.transverse.STATUSES[native.status] == python.status
        for ahead in (0.0, proposal, 2 * proposal, python.ceiling(z)):
            if math.isfinite(ahead):
                assert native.due(z, ahead) == python.due(z, ahead)
        for later in (z, math.nextafter(z, math.inf), z + proposal):
            assert native.wants_read(later) == python.wants_read(later)

    def record(self) -> None:
        """Require the same record, number for number."""
        expected = self.python.summary()
        got = self.transverse.record_of(self.python, self.native.record())
        assert list(got) == list(expected)
        assert got == expected

    def launch(
        self,
        drawn: Readings,
        z: float = 0.0,
        *,
        noise: np.ndarray | None = None,
        clean: np.ndarray | None = None,
    ) -> list[tuple[str, str]]:
        resumed = self.python.resumed
        collar = self.python.every_step
        python_reading, values = self.reading(
            drawn, full=not resumed, current=collar and resumed
        )
        columns = np.asarray(values["columns"])
        clean_columns = None if noise is None or resumed else clean
        expected = self._python_call(
            lambda: self.python.launch(
                python_reading,
                z,
                noise=noise,
                clean_columns=clean_columns,
                steps=self.steps,
                reads=self.reads,
                declared=self.declared,
            )
        )
        sums = None
        if not resumed and self.launch_energy > 0.0:
            sums = self.launch_sums(columns if clean_columns is None else clean_columns)
        carried = None if noise is None else self.noise(noise, clean_columns)
        got = self._native_call(self.native.launch(values, z, sums, carried))
        assert got == expected
        return got

    def noise(self, noise: np.ndarray, clean: np.ndarray | None) -> dict[str, object]:
        """Return the input noise's readings, as the native session forms them."""
        return self.transverse.noise_readings(self.python, noise, clean)

    def read(self, drawn: Readings, z: float) -> list[tuple[str, str]]:
        python_reading, values = self.reading(drawn, full=False)
        sums = self.read_sums(z)
        expected = self._python_call(lambda: self.python.read(python_reading, z))
        got = self._native_call(self.native.read(values, z, sums))
        assert got == expected
        return got

    def step(self, current: float, z: float) -> None:
        rate = current * self.launch_energy
        self.python.step(rate, z)
        self.native.step(rate, z)


SHARE = st.one_of(
    st.just(0.0),
    st.floats(1e-12, 1e-2, allow_nan=False),
    # Around the warning and the stop of the default budget.
    st.sampled_from([1e-6, 1e-4]).map(lambda value: value * (1.0 + 1e-15)),
)
COLUMN = st.one_of(
    st.just(0.0),
    st.floats(1e-14, 1.0, allow_nan=False),
    st.sampled_from([1e-6, 2e-6, 5e-7]),
)


def _readings(*, equal_faces: bool = False) -> st.SearchStrategy[Readings]:
    """Return drawn readings, shares of the launch energy."""
    faces = (
        SHARE.map(lambda share: [share] * 4)
        if equal_faces
        else st.lists(SHARE, min_size=4, max_size=4)
    )
    return st.builds(
        Readings,
        faces=faces,
        clip=st.lists(SHARE, min_size=4, max_size=4),
        collar=st.floats(0.0, 1e3),
        outside=st.floats(0.0, 1e-3),
        columns=st.lists(COLUMN, min_size=1, max_size=64),
        strip=st.lists(SHARE, min_size=2, max_size=8),
        current=st.one_of(st.just(0.0), st.floats(0.0, 1e-1)),
    )


STEP = st.tuples(
    # The step, m: within the read spacing's backstop and well past it.
    st.one_of(st.floats(1e-6, 1e-3), st.sampled_from([6.25e-4, 2.5e-3])),
    _readings() | _readings(equal_faces=True),
    # Read the edge here, or only take the current.
    st.sampled_from(["read", "read", "step", "plane"]),
)


def _run(pair: Pair, steps: list, z: float = 0.0) -> tuple[float, bool]:
    """Feed a drawn sequence to both; return the position and whether it stopped."""
    for dz, drawn, action in steps:
        z += dz
        if pair.python.every_step:
            pair.step(drawn.current, z)
        if action != "step":
            reported = pair.read(drawn, z)
            if reported and reported[-1][0] == "stop":
                return z, True
        if action == "plane":
            # A plane at the same position reads nothing more.
            assert pair.read(drawn, z) == []
        pair.state(z, dz)
    return z, False


@settings(max_examples=80, deadline=None)
@given(
    geometry=st.sampled_from(["line", "box", "radial"]),
    design=st.sampled_from(["none", "ramp", "collar"]),
    budget=st.sampled_from([1e-3, 3e-2]),
    launch=_readings(),
    steps=st.lists(STEP, min_size=1, max_size=12),
    schedule=st.one_of(st.none(), st.floats(1e-5, 2e-3)),
    declared=st.booleans(),
)
def test_the_owner_decides_and_records_as_solprop_does(  # noqa: PLR0913 - one draw each
    *, geometry, design, budget, launch, steps, schedule, declared
):
    if geometry == "radial" and design == "collar":
        design = "ramp"
    pair = Pair(
        geometry,
        design,
        budget=budget,
        steps=None if schedule is None else np.array([schedule, schedule / 2]),
        declared=declared and design == "ramp",
    )
    launched = pair.launch(launch)
    pair.record()
    if launched and launched[-1][0] == "stop":
        return
    pair.state(0.0, 1e-4)
    _run(pair, steps)
    pair.record()


@settings(max_examples=30, deadline=None)
@given(
    geometry=st.sampled_from(["line", "box"]),
    design=st.sampled_from(["none", "ramp", "collar"]),
    first=st.lists(STEP, min_size=1, max_size=6),
    then=st.lists(STEP, min_size=1, max_size=6),
    launch=_readings(),
    drop=st.sampled_from([None, "edge_max", "edge_baseline", "launch_energy"]),
)
def test_a_resume_continues_the_record_as_solprop_does(
    geometry, design, first, then, launch, drop
):
    # The first execution's record is solprop's own, as a run's file keeps it; a
    # record without a fitting baseline or launch starts the edge again.
    earlier = Pair(geometry, design)
    launched = earlier.launch(launch)
    if launched and launched[-1][0] == "stop":
        return
    z, stopped = _run(earlier, first)
    if stopped:
        return
    record = earlier.python.summary()
    if drop is not None:
        record[drop] = [0.0] * 7 if drop == "edge_max" else None
    pair = Pair(geometry, design)
    faces = len(pair.python._faces)  # pyright: ignore[reportPrivateUsage]
    pair.python.resume(record)
    pair.native.resume(pair.transverse.resumption(record, faces))
    pair.launch(launch, z)
    pair.state(z, 1e-4)
    pair.record()
    _run(pair, then, z)
    pair.record()


def test_tied_faces_stop_on_the_name_max_breaks_the_tie_on():
    # Four faces at one judged share past the stop: `max` of `(share, name)` tuples
    # picks the greatest name, "y-" ("+" sorts before "-").
    pair = Pair("box", "none")
    calm = Readings([0.0] * 4, [0.0] * 4, 0.0, 0.0, [1e-9], [0.0, 0.0], 0.0)
    assert pair.launch(calm) == []
    tied = Readings([2e-4] * 4, [0.0] * 4, 0.0, 0.0, [1e-9], [0.0, 0.0], 0.0)
    reported = pair.read(tied, 1e-4)
    assert reported[-1][0] == "stop"
    assert "on its y- face" in reported[-1][1]
    pair.record()


TOP = 32
"""The line's column of the largest transverse wavenumber, in the ramp's band."""


def test_a_launch_with_several_warnings_serves_each_in_turn():
    # The ramp's band, two clipped faces and a fixed schedule's long step: four
    # warnings at one launch, each its own notice, in solprop's order.
    pair = Pair("line", "ramp", steps=np.array([5e-3]))
    columns = [0.0] * 64
    columns[TOP] = 5e-5
    columns[0] = 1.0
    clipped = Readings([0.0] * 4, [3e-6, 4e-6, 0.0, 0.0], 0.0, 0.0, columns, [0.0], 0.0)
    reported = pair.launch(clipped)
    assert [kind for kind, _ in reported] == ["warn"] * 4
    assert "safety ramp's band" in reported[0][1]
    assert "x- face" in reported[1][1] and "x+ face" in reported[2][1]
    assert "longest step" in reported[3][1]
    pair.record()


def test_a_read_warns_on_a_face_then_stops_on_the_ramp():
    # The face's warning first; once the caller allows it, the ramp's stop.
    pair = Pair("line", "ramp")
    columns = [0.0] * 64
    columns[0] = 1.0
    assert pair.launch(Readings([0.0] * 4, [0.0] * 4, 0, 0, columns, [0.0], 0)) == []
    # The column's weight is 1.46e-7: a thousandth of the launch's energy.
    columns[TOP] = 1e4
    # The filament appears at a read the ramp has had no time to act on.
    assert (
        pair.read(Readings([0.0] * 4, [0.0] * 4, 0, 0, columns, [0.0], 0), 1e-9) == []
    )
    filament = Readings([3e-5, 0.0, 0, 0], [0.0] * 4, 0, 0, columns, [0.0], 0)
    reported = pair.read(filament, 2e-3)
    assert [kind for kind, _ in reported] == ["warn", "stop"]
    assert "x- face" in reported[0][1]
    assert "safety ramp has removed" in reported[1][1]
    pair.record()


def test_a_warning_the_caller_raises_leaves_the_rest_of_its_point_undone():
    # A raised warning ends the run there: the native record is the edge's as
    # solprop leaves it when its own warning raises, its later judgement not made.
    pair = Pair("line", "ramp")
    columns = [0.0] * 64
    columns[0] = 1.0
    assert pair.launch(Readings([0.0] * 4, [0.0] * 4, 0, 0, columns, [0.0], 0)) == []
    columns[TOP] = 0.5
    filament = Readings([3e-5, 0.0, 0, 0], [0.0] * 4, 0, 0, columns, [0.0], 0)
    pair.read(Readings([0.0] * 4, [0.0] * 4, 0, 0, columns, [0.0], 0), 1e-3)
    python_reading, values = pair.reading(filament, full=False)
    sums = pair.read_sums(2e-3)
    assert pair.native.status == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", BoundaryWarning)
        with pytest.raises(BoundaryWarning):
            pair.python.read(python_reading, 2e-3)
    notice = pair.native.read(values, 2e-3, sums)
    assert notice["kind"] == pair.transverse.NoticeKind.FACE and not notice["stop"]
    assert pair.native.pending
    pair.record()


def test_an_unread_edge_warns_once_and_records_why():
    pair = Pair("line", "none", reads=False)
    reported = pair.launch(Readings([0.0] * 4, [0.0] * 4, 0, 0, [1.0], [0.0], 0.0))
    assert reported == [("warn", reported[0][1])]
    assert "does not read the transverse edge" in reported[0][1]
    pair.state(0.0, 1e-4)
    pair.record()


@pytest.mark.parametrize("design", ["none", "ramp", "collar"])
def test_input_noise_is_judged_alike(design):
    # The noise's allowance, its columns in the reach and the ramp, and a collar's
    # strip floor: readings of a noise draw that both edges take.
    rng = np.random.default_rng(3)
    for scale in (1e-4, 1.0):
        pair = Pair("line", design)
        noise = scale * (
            rng.standard_normal((64, 1, 1)) + 1j * rng.standard_normal((64, 1, 1))
        )
        # Light at the low wavenumbers, a little in the ramp's band.
        clean = 1e-9 * np.abs(rng.standard_normal(64))
        clean[:4] = 1.0
        columns = list(clean + 1e-9 * np.abs(rng.standard_normal(64)))
        drawn = Readings([1e-7, 2e-7, 0, 0], [0.0] * 4, 1.0, 1e-5, columns, [1e-7], 0)
        reported = pair.launch(drawn, noise=noise, clean=clean)
        if scale > 1e-2:
            assert any("cannot be judged below" in text for _, text in reported)
        pair.record()
        if reported and reported[-1][0] == "stop":
            continue
        _run(pair, [(5e-4, drawn, "read"), (5e-4, drawn, "plane")])
        pair.record()


def test_the_thresholds_and_their_neighbours_agree():
    # A face's judged share exactly at the warning and the stop and at the doubles
    # beside them, as `>` compares them.
    for edge in (1e-6, 1e-4):
        for share in (math.nextafter(edge, 0.0), edge, math.nextafter(edge, 1.0)):
            pair = Pair("line", "none")
            calm = Readings([0.0] * 4, [0.0] * 4, 0, 0, [1e-9], [0.0], 0)
            assert pair.launch(calm) == []
            pair.read(
                Readings([share, 0, 0, 0], [0.0] * 4, 0, 0, [1e-9], [0.0], 0), 1e-4
            )
            pair.record()


@settings(max_examples=25, deadline=None)
@given(
    columns=st.lists(st.floats(0.0, 1.0), min_size=64, max_size=64),
    gap=st.floats(1e-7, 1e-2),
)
def test_the_native_ramp_readings_are_within_their_bound(columns, gap):
    # The native edge sums the ramp's band and its removal in order, with the C
    # library's expm1; NumPy pairwise, with its own. Every term is nonnegative, so the
    # sums differ by at most (n - 1 + ceil(log2 n)) u of either for the order (Higham,
    # Accuracy and Stability of Numerical Algorithms, 2nd ed., section 4.2), and each
    # removal term by the two expm1's errors, at most 4 u between them, and its product.
    # Declared content: the launch is not judged on its band, and always reads on.
    pair = Pair("line", "ramp", declared=True)
    drawn = Readings([0.0] * 4, [0.0] * 4, 0, 0, columns, [0.0], 0)
    _, values = pair.reading(drawn, full=True)
    launched = np.asarray(values["columns"])
    total, ramp = pair.native.launch_sums(launched)
    expected = pair.launch_sums(launched)
    assert expected is not None
    n = launched.size
    order = (n - 1 + math.ceil(math.log2(n))) * U
    assert total == pytest.approx(expected[0], rel=order, abs=0.0)
    assert ramp == pytest.approx(expected[1], rel=order, abs=1e-300)
    pair.launch(drawn)
    raw, _ = pair.native.read_sums(gap)
    found = pair.read_sums(gap)
    assert found is not None
    assert raw == pytest.approx(found[0], rel=order + 6 * U, abs=1e-300)


@pytest.mark.parametrize(
    "record",
    [
        {"launch_energy": None, "edge_baseline": [0.0, 0.0], "edge_max": [None, None]},
        {"launch_energy": 1.0, "edge_baseline": [0.0], "edge_max_raw": [None, None]},
        {"launch_energy": 0.0, "edge_baseline": [0.0, 0.0], "reads": math.nan},
    ],
    ids=["no-launch", "unfitting-baseline", "empty-launch"],
)
def test_a_record_solprop_s_resume_ignores_is_ignored_alike(record):
    # Past a launch without energy or a baseline that does not fit, solprop's resume
    # reads nothing more: the maxima and the reads are not converted, nor refused.
    pair = Pair("line", "none")
    pair.python.resume(record)
    pair.native.resume(pair.transverse.resumption(record, 2))
    assert not pair.python.resumed and not pair.native.resumed
    pair.launch(Readings([0.0] * 4, [0.0] * 4, 0, 0, [1.0], [0.0], 0.0))
    pair.record()
