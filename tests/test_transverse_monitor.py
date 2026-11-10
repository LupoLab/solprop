# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The boundary monitor's transverse edge (SEP 40 section 5.3, Part II's witnesses).

The scaled suite case (lengths scaled paraxially by 1/8, linear here) reaches its
wall within the run as configured; a bare wall at f = 1 reads an edge share near
2.9e-6 (Part II, factor 3) and warns, and a requested collar at f = 2.6 completes
clear.
"""

import dataclasses
import math
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from solprop.backends import driver as bridge
from solprop.backends.host import driver as host
from solprop.backends.registry import available
from solprop.boundaries import SpatialAbsorber, SpatialAbsorberDesign
from solprop.driver import Simulation
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import FreeGrid, LineGrid, RadialGrid
from solprop.log import NullReporter
from solprop.modes import PlanarGrid, SlabMode
from solprop.monitor import BoundaryCheck
from solprop.output.hdf5 import resume
from solprop.output.schema import PlaneStore
from solprop.translate.transverse import _nodes_window
from solprop.transverse_monitor import BACKSTOP_READS, TransverseEdge, _row_energy
from test_transverse_resolution import LENGTH, suite


def _launch(edge: TransverseEdge, state: np.ndarray, z: float, **options: Any) -> None:
    """Launch `edge` on a state's host reading, as a backend's load would."""
    noise = options.get("noise")
    clean = None if noise is None else edge.reduction.column_power(state - noise)
    reading = edge.reduction.reading(
        state,
        full=not edge.resumed,
        current=edge.every_step and edge.resumed,
    )
    edge.launch(reading, z, clean_columns=clean, **options)


def _read(edge: TransverseEdge, state: np.ndarray, z: float) -> None:
    """Read `edge` on a state's host reading."""
    edge.read(edge.reduction.reading(state), z)


def _current(edge: TransverseEdge, state: np.ndarray) -> float:
    """Return the state's inward current as a share of the launch."""
    return edge.reduction.current(state) / edge._launch  # pyright: ignore[reportPrivateUsage]


def _step(edge: TransverseEdge, state: np.ndarray, z: float) -> None:
    """Accumulate `edge`'s current on a state's host reading."""
    edge.step(edge.reduction.current(state), z)


def edge_record(output: object) -> dict[str, Any]:
    """Return a finished run's transverse edge record."""
    assert isinstance(output, PlaneStore)
    assert output.outcome is not None
    monitor = output.outcome.execution["boundary_monitor"]
    assert isinstance(monitor, dict)
    return cast("dict[str, Any]", monitor["transverse"])


def quietly(simulation: Simulation) -> tuple[object, list[str]]:
    """Run, returning the output and the boundary warnings' messages."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", BoundaryWarning)
        output = simulation.run(progress="none")
    return output, [
        str(one.message) for one in caught if issubclass(one.category, BoundaryWarning)
    ]


def test_a_run_reaching_its_wall_is_stopped_naming_a_size() -> None:
    with (
        warnings.catch_warnings(),
        pytest.raises(BoundaryError, match="transverse edge") as caught,
    ):
        warnings.simplefilter("ignore", BoundaryWarning)
        suite(0.0, None).run(progress="none")
    message = str(caught.value)
    assert "a radius of at least" in message
    assert "n_r of at least" in message
    assert "lower bound" in message


def test_a_bare_wall_at_twice_the_radius_warns_near_the_witness() -> None:
    output, messages = quietly(suite(1.0, None))
    record = edge_record(output)
    assert record["transverse_status"] == "warned"
    assert 2.9e-6 / 3 <= max(record["edge_max"]) <= 3 * 2.9e-6
    assert any("reached the transverse edge" in one for one in messages)


def test_a_bare_wall_further_out_completes_clear() -> None:
    output, messages = quietly(suite(1.5, None))
    record = edge_record(output)
    assert record["transverse_status"] == "clear"
    assert max(record["edge_max"]) < 1e-6
    assert not [one for one in messages if "transverse edge" in one]


def test_a_requested_collar_completes_clear_and_credits_its_removal() -> None:
    output, _ = quietly(suite(2.6, SpatialAbsorber()))
    record = edge_record(output)
    assert record["transverse_status"] == "clear"
    assert record["spatial_removed"] > 0.0
    assert record["edge_max"] == [0.0]


def test_the_reads_are_spaced_by_the_ceiling() -> None:
    # Section 5.3, with issue #396's backstop: at most L/4 apart, enforced as a step
    # ceiling, so a low-source run that would step further reads at least four times.
    output, _ = quietly(suite(1.5, None))
    record = edge_record(output)
    assert record["reads"] >= BACKSTOP_READS
    assert record["largest_gap"] <= LENGTH / BACKSTOP_READS * (1 + 1e-9)


def _spacer(reaches: list[float]) -> SimpleNamespace:
    """A monitor's spacing state over a guide of 1 m, guard 1 mm, reading `reaches`."""
    values = iter(reaches)
    return SimpleNamespace(
        length=1.0,
        _guard=1e-3,
        _reach=lambda _columns: next(values),
        _backstop=math.inf,
        _reach_reference=0.0,
        _spacing_now=math.inf,
    )


def test_the_backstop_grows_while_the_reach_holds_and_starts_again_when_it_grows():
    # Issue #396: the backstop starts at L/16 and doubles at each read whose reach has
    # not grown by a tenth since it started, to L/4; growth starts it again.
    reaches = [1e-3, 1.05e-3, 1.09e-3, 1.0e-3, 1.0e-3, 1.2e-3, 1.2e-3]
    monitor = _spacer(reaches)
    backstops, spacings = [], []
    for read, _ in enumerate(reaches):
        TransverseEdge._respace(monitor, np.zeros(1), launch=read == 0)  # pyright: ignore[reportArgumentType]
        backstops.append(monitor._backstop)
        spacings.append(monitor._spacing_now)
    assert backstops == [1 / 16, 1 / 8, 1 / 4, 1 / 4, 1 / 4, 1 / 16, 1 / 8]
    # 4 g / v_q bounds the spacing whatever the backstop: 4 mm / 1e-3 = 4 m here.
    assert spacings == backstops
    fast = _spacer([1.0, 1.0])
    TransverseEdge._respace(fast, np.zeros(1), launch=True)  # pyright: ignore[reportArgumentType]
    TransverseEdge._respace(fast, np.zeros(1), launch=False)  # pyright: ignore[reportArgumentType]
    assert fast._backstop == 1 / 8 and fast._spacing_now == pytest.approx(4e-3)


def test_a_creeping_reach_is_measured_from_the_backstop_s_start():
    # Each read grows the reach by 6 %, under the tenth, but by the third it has grown
    # 12 % since the backstop started, which starts it again.
    monitor = _spacer([1e-3, 1.06e-3, 1.1236e-3])
    for read in range(3):
        TransverseEdge._respace(monitor, np.zeros(1), launch=read == 0)  # pyright: ignore[reportArgumentType]
    assert monitor._backstop == 1 / 16
    assert monitor._reach_reference == pytest.approx(1.1236e-3)


def test_a_spectrum_without_reach_keeps_the_backstop_at_its_start():
    # Nothing above the warning (a budget of order one) says nothing about speeds.
    monitor = _spacer([1e-3, 1e-3, 0.0, 0.0])
    for read in range(4):
        TransverseEdge._respace(monitor, np.zeros(1), launch=read == 0)  # pyright: ignore[reportArgumentType]
    assert monitor._spacing_now == monitor._backstop == 1 / 16


@pytest.mark.skipif("native" not in available(), reason="native is not installed")
def test_native_reads_the_edge_as_numpy_ref_does() -> None:
    # A backend that does not read it is tested in tests/backends/
    # test_transverse_backends.py, with numpy_ref's reading switched off.
    records = []
    for backend in ("numpy_ref", "native"):
        simulation = suite(1.5, None)
        simulation.settings = dataclasses.replace(simulation.settings, backend=backend)
        output, messages = quietly(simulation)
        assert not any("does not read the transverse edge" in one for one in messages)
        records.append(edge_record(output))
    reference, found = records
    assert found["transverse_status"] == reference["transverse_status"]
    assert found["reads"] == reference["reads"] > 0


def test_the_transverse_budget_relaxes_the_edge_alone() -> None:
    simulation = suite(0.0, None)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        boundary_check=BoundaryCheck(transverse_budget=0.1),
    )
    # The edge reaches about 1e-4 of the launch: past the default stop, inside a
    # budget of 0.1 (1e-2 in energy). Time keeps its own budget.
    output, _ = quietly(simulation)
    assert edge_record(output)["transverse_status"] == "clear"


def _ramp_launch(share: float) -> Simulation:
    """Return the scaled suite with `share` of its launch in the top k_r column."""
    simulation = suite(1.5, None)
    problem = simulation.translate()
    field = np.array(problem.initial.field0)
    weights = problem.transform.geometry.spectral_weights  # pyright: ignore[reportOptionalMemberAccess]
    energy = float(np.sum(weights[:, None, None] * np.abs(field) ** 2))
    top = np.zeros_like(field)
    top[-1] = field[0]  # the beam's own spectrum, so the time window is not the issue
    top_energy = float(np.sum(weights[-1] * np.abs(top[-1]) ** 2))
    field = math.sqrt(1 - share) * field + math.sqrt(share * energy / top_energy) * top
    return simulation.with_input(field)


def test_the_edge_s_band_wavenumbers_are_found_once_per_run_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import solprop.driver.simulation as module  # noqa: PLC0415

    simulation = suite(1.5, None)
    calls: list[float] = []
    original = module.band_wavenumbers

    def counted(*args: Any) -> tuple[np.ndarray, np.ndarray]:
        calls.append(args[2])
        return original(*args)

    monkeypatch.setattr(module, "band_wavenumbers", counted)
    space, medium = simulation.spatial_grid, simulation.medium
    assert isinstance(space, RadialGrid | FreeGrid | PlanarGrid) and medium is not None
    first, again = (
        simulation._transverse_edge(BoundaryCheck(), space, medium) for _ in range(2)
    )
    assert len(calls) == 1
    assert first.k_min == again.k_min


def test_a_launch_in_the_ramp_s_band_is_refused() -> None:
    with pytest.raises(BoundaryError, match="safety ramp's band"):
        quietly(_ramp_launch(1e-2))


def test_a_small_share_in_the_ramp_s_band_warns() -> None:
    _, messages = quietly(_ramp_launch(1e-5))
    assert any("safety ramp's band" in one for one in messages)


class _StopAtPlane(NullReporter):
    """Raise at the second saved plane."""

    def __init__(self) -> None:
        self.seen = 0

    def plane(self, z: float, summary: object) -> None:
        del z, summary
        self.seen += 1
        if self.seen == 2:
            raise KeyError("stopped")


def test_a_resume_judges_against_the_original_launch(tmp_path: Path) -> None:
    def build() -> Simulation:
        simulation = suite(1.0, None)
        simulation.settings = dataclasses.replace(
            simulation.settings, save_planes=[LENGTH / 2, LENGTH]
        )
        return simulation

    whole, _ = quietly(build())
    path = tmp_path / "run.h5"
    with warnings.catch_warnings(), pytest.raises(KeyError):
        warnings.simplefilter("ignore", BoundaryWarning)
        build().run(path, progress=_StopAtPlane())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        build().run(path, resume=True, progress="none")
    outcome = resume(path).outcome
    assert outcome is not None
    resumed = cast("dict[str, Any]", outcome.execution["boundary_monitor"])[
        "transverse"
    ]
    expected = edge_record(whole)
    assert resumed["edge_baseline"] == expected["edge_baseline"]
    assert resumed["launch_energy"] == pytest.approx(expected["launch_energy"])
    assert max(resumed["edge_max"]) == pytest.approx(max(expected["edge_max"]), rel=0.1)


def test_a_resume_restores_a_collars_totals_and_continues_its_current() -> None:
    # A resumed collar run used to judge against zero baselines, with its current and
    # peaks restarted: nothing the resumed branch of the launch doesn't take again was
    # restored. The current's trapezoid continues from the resumed state's own sample.
    edge, space = _line_edge()
    start = _packet(space, 0.0, 0.0)
    # Returning, but faint enough to stay below the stop.
    packet = 3e-3 * _packet(space, 230e-6, 1e6)
    options: dict[str, Any] = {
        "noise": None,
        "steps": None,
        "reads": True,
        "declared": False,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        _launch(edge, start, 0.0, **options)
        _step(edge, packet, 1e-5)
        _step(edge, packet, 2e-5)
        _read(edge, packet, 2e-5)
    record = edge.summary()
    assert cast("float", record["current"]) > 0.0
    assert cast("float", record["strip_max"]) > 0.0
    resumed, _ = _line_edge()
    resumed.resume(record)
    for name in (
        "current",
        "current_baseline",
        "strip_baseline",
        "strip_floor",
        "strip_max",
        "returned_max",
    ):
        # Copied from the record: a structural contract, not arithmetic.
        assert resumed.summary()[name] == record[name], name
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        _launch(resumed, packet, 2e-5, **options)
        _step(resumed, packet, 3e-5)
        _step(edge, packet, 3e-5)
    # Two runs of the same arithmetic on the same numbers, in one process.
    assert resumed.summary()["current"] == pytest.approx(
        edge.summary()["current"], rel=1e-12
    )


@pytest.mark.parametrize("module", [host, bridge], ids=["host", "bridge"])
def test_the_clean_launch_columns_are_the_masked_launch_less_its_noise(
    module: ModuleType,
) -> None:
    # They stand in for `mask(field + noise) - noise`, which on a device would mean a
    # copy of the state. The two differ only by the rounding of adding and subtracting
    # the noise, relative to the column's energy.
    rng = np.random.default_rng(29)
    shape = (5, 2, 16)
    field = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    noise = 1e-3 * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    active = np.ones(16, dtype=bool)
    active[:3] = False
    expected = np.sum(np.abs(active * (field + noise) - noise) ** 2, axis=(1, 2))
    found = module._clean_launch_columns(field, noise, active)
    np.testing.assert_allclose(found, expected, rtol=1e-13)


def test_the_start_of_run_summary_says_what_the_edge_can_report() -> None:
    from solprop.driver.summary import run_summary  # noqa: PLC0415
    from solprop.output import MemoryOutput  # noqa: PLC0415

    simulation = suite(1.5, None)
    problem, metadata = simulation.translate(), simulation.metadata()

    def row(backend: str, check: BoundaryCheck | None) -> str:
        summary = run_summary(
            problem,
            metadata,
            backend=backend,
            execution={},
            output=MemoryOutput(),
            boundary_check=check,
        )
        return str(dict(summary.rows)["transverse edge"])

    assert "watched" in row("numpy_ref", BoundaryCheck())
    assert "unwatched on other" in row("other", BoundaryCheck())
    assert row("numpy_ref", None).endswith("off (boundary_check=None)")


def _line_edge() -> tuple[TransverseEdge, LineGrid]:
    """Return a line's edge: a collar of 100 um outside R = 200 um, 1 um spacing."""
    space = LineGrid(200e-6, 600, collar_fraction=0.5)
    design = SpatialAbsorberDesign(
        SpatialAbsorber(kappa=3e5),
        np.zeros(space.n_columns),
        1.0,
        (1e6,),
        collar_rate=1e3 * (1 - np.asarray(space.window)),
        alpha0=1e3,
        width=(100e-6,),
        kappa_p=(3e5,),
    )
    edge = TransverseEdge(
        space, design, 1e-3, 10.0, 0.01, np.arange(1), 2 * math.pi / 800e-9
    )
    return edge, space


def _packet(space: LineGrid, centre: float, kappa: float) -> np.ndarray:
    """Return a Gaussian packet at `centre` with transverse wavenumber `kappa`."""
    x = np.asarray(space.x)
    field = np.exp(-(((x - centre) / 10e-6) ** 2) + 1j * kappa * x)
    return np.fft.fft(field)[:, None, None]


def test_the_strip_reads_inward_flow_and_not_outward() -> None:
    # Section 5.3: the Kaiser(8)-windowed strip resolves direction above 16 / s; a
    # packet in the strip moving outward reads below 1e-6 of itself, one moving
    # inward reads most of itself.
    edge, space = _line_edge()
    launch = _packet(space, 0.0, 0.0)
    _launch(edge, launch, 0.0, noise=None, steps=None, reads=True, declared=False)
    # solprop propagates as exp(-i k_z z): a packet exp(i kappa x) travels to -x.
    outward, _ = edge._strip_shares(edge.reduction.strip(_packet(space, 230e-6, -1e6)))  # pyright: ignore[reportPrivateUsage]
    inward, _ = edge._strip_shares(edge.reduction.strip(_packet(space, 230e-6, 1e6)))  # pyright: ignore[reportPrivateUsage]
    assert outward < 1e-6
    assert inward > 0.1


def test_the_onset_current_counts_inward_flow_only() -> None:
    edge, space = _line_edge()
    _launch(
        edge,
        _packet(space, 0.0, 0.0),
        0.0,
        noise=None,
        steps=None,
        reads=True,
        declared=False,
    )
    leaving = _current(edge, _packet(space, 200e-6, -1e6))  # pyright: ignore[reportPrivateUsage]
    returning = _current(edge, _packet(space, 200e-6, 1e6))  # pyright: ignore[reportPrivateUsage]
    assert leaving < 1e-20
    assert returning > 1e3 * leaving


def test_the_ramp_s_removal_stops_a_run_at_ten_budgets() -> None:
    # Q8, decided on the under-resolved filament's measurements: the ramp's removal
    # warns above the budget and stops above ten, as the edge does (section 5.3).
    space = LineGrid(200e-6, 256)
    ramp = np.where(
        np.abs(np.asarray(space.k_x)) > 0.9 * np.max(np.abs(space.k_x)), 1e4, 0.0
    )
    design = SpatialAbsorberDesign(
        SpatialAbsorber(real_space=False), ramp, 1e4, (0.9 * float(np.max(space.k_x)),)
    )
    edge = TransverseEdge(
        space, design, 1e-3, 10.0, 0.01, np.arange(1), 2 * math.pi / 800e-9
    )
    clean = _packet(space, 0.0, 0.0)
    _launch(edge, clean, 0.0, noise=None, steps=None, reads=True, declared=False)
    filament = clean.copy()
    top = int(np.argmax(np.abs(np.asarray(space.k_x))))
    filament[top] = 0.05 * float(np.max(np.abs(clean)))  # content generated at the edge
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        _read(edge, filament, 1e-4)
        with pytest.raises(BoundaryError, match="safety ramp has removed"):
            _read(edge, filament, 1e-3)
    assert edge.status == "stopped"


# The edge's own bookkeeping, one member's state handed to it directly: its launch,
# its reads and its resume on each geometry, with no run around it.

K800 = 2 * math.pi / 800e-9


def _edge(
    space: LineGrid | FreeGrid | PlanarGrid,
    design: SpatialAbsorberDesign | None,
    *,
    budget: float = 1e-3,
) -> TransverseEdge:
    """Return an edge on `space`, one frequency at 800 nm over a 1 cm run."""
    return TransverseEdge(space, design, budget, 10.0, 0.01, np.arange(1), K800)


def _collar(
    space: LineGrid | FreeGrid | PlanarGrid, width: float
) -> SpatialAbsorberDesign:
    """Return a collar of 1e3 1/m in the room outside the requested size."""
    columns = int(np.asarray(space.spectral_weights).size)
    return SpatialAbsorberDesign(
        SpatialAbsorber(kappa=3e5),
        np.zeros(columns),
        1.0,
        (1e6,),
        collar_rate=1e3 * (1 - _nodes_window(space)),
        alpha0=1e3,
        width=(width,),
        kappa_p=(3e5,),
    )


def _launched(edge: TransverseEdge, state: np.ndarray, **options: Any) -> None:
    """Attach `state` at z = 0 as a read, undeclared launch without noise."""
    arguments: dict[str, Any] = {
        "noise": None,
        "steps": None,
        "reads": True,
        "declared": False,
    }
    _launch(edge, state, 0.0, **(arguments | options))


def _gaussian_line(space: LineGrid, centre: float, waist: float) -> np.ndarray:
    """Return a beam on a line's columns, one polarisation and frequency."""
    x = np.asarray(space.x)
    return np.fft.fft(np.exp(-(((x - centre) / waist) ** 2)))[:, None, None]


def test_a_launch_with_no_energy_is_never_read() -> None:
    edge = _edge(LineGrid(200e-6, 64), None)
    _launched(edge, np.zeros((64, 1, 1), dtype=complex))
    # Nothing to judge shares against: no read is due, and a read does nothing.
    assert edge.ceiling(0.0) == math.inf
    _read(edge, np.zeros((64, 1, 1), dtype=complex), 1e-3)
    assert edge.summary()["reads"] == 0


def test_a_launch_in_the_outer_twentieth_of_the_aperture_warns_it_is_clipped() -> None:
    # A beam centred at 190 um on a 200 um request puts most of its energy in the
    # outer 5 % of the aperture on its x+ face.
    space = LineGrid(200e-6, 128, collar_fraction=0.5)
    with pytest.warns(BoundaryWarning, match=r"on its x\+ face: the aperture clips"):
        _launched(_edge(space, None), _gaussian_line(space, 190e-6, 10e-6))


def _noisy(scale: float) -> tuple[TransverseEdge, np.ndarray, np.ndarray]:
    """Return a bare line's edge with a beam and white input noise of `scale`."""
    space = LineGrid(200e-6, 128)
    beam = _gaussian_line(space, 0.0, 20e-6)
    phases = np.random.default_rng(7).uniform(0.0, 2 * math.pi, beam.shape)
    noise = scale * float(np.max(np.abs(beam))) * np.exp(1j * phases)
    return _edge(space, None), beam, noise


def test_input_noise_above_the_warning_leaves_the_edge_insensitive() -> None:
    # White noise puts its share at the edge from the launch on: an edge that cannot
    # be judged below its allowance reports itself insensitive, and keeps that status
    # when light then reaches the edge and warns.
    edge, beam, noise = _noisy(3e-3)
    with pytest.warns(BoundaryWarning, match="cannot be judged below"):
        _launched(edge, beam + noise, noise=noise)
    record = edge.summary()
    allowance = cast("float", record["noise_allowance"])
    assert allowance > 1e-6
    assert edge.status == "insensitive"
    # A packet on the x+ face holding 3e-5 of the launch energy above the face's
    # launch share and the allowance: past the warning, short of the stop.
    space = cast("LineGrid", edge.space)
    packet = _gaussian_line(space, 195e-6, 4e-6)
    above = cast("list[float]", record["edge_baseline"])[1] + allowance + 3e-5
    launch = cast("float", record["launch_energy"])
    scale = math.sqrt(above * launch / _energy(edge, packet))
    with pytest.warns(BoundaryWarning, match="light reached the transverse edge"):
        _read(edge, beam + noise + scale * packet, 1e-3)
    assert edge.status == "insensitive"


def _energy(edge: TransverseEdge, state: np.ndarray) -> float:
    """Return a state's energy in the edge's own measure."""
    return float(np.sum(edge._column_power(edge.reduction.column_power(state))))  # pyright: ignore[reportPrivateUsage]


def _scaled(edge: TransverseEdge, state: np.ndarray, share: float) -> np.ndarray:
    """Return `state` scaled to hold `share` of the edge's launch energy."""
    launch = cast("float", edge.summary()["launch_energy"])
    return math.sqrt(share * launch / _energy(edge, state)) * state


def test_the_ramp_s_warning_keeps_an_earlier_verdict() -> None:
    # A line with a safety ramp on its top tenth of k_x: light reaching the x+ face
    # warns first; the ramp's removal of a filament at the top column then warns too,
    # and the run stays warned rather than being reset.
    space = LineGrid(200e-6, 256)
    ramp = np.where(
        np.abs(np.asarray(space.k_x)) > 0.9 * np.max(np.abs(space.k_x)), 1e4, 0.0
    )
    design = SpatialAbsorberDesign(
        SpatialAbsorber(real_space=False), ramp, 1e4, (0.9 * float(np.max(space.k_x)),)
    )
    edge = _edge(space, design)
    beam = _gaussian_line(space, 0.0, 20e-6)
    _launched(edge, beam)
    filament = np.zeros_like(beam)
    filament[int(np.argmax(np.abs(np.asarray(space.k_x))))] = 0.01 * float(
        np.max(np.abs(beam))
    )
    state = beam + filament + _scaled(edge, _gaussian_line(space, 196e-6, 2e-6), 3e-5)
    with pytest.warns(BoundaryWarning, match="light reached the transverse edge"):
        _read(edge, state, 1e-4)
    assert edge.status == "warned"
    with pytest.warns(BoundaryWarning, match="safety ramp has removed"):
        _read(edge, state, 2e-4)
    assert edge.status == "warned"
    # Without a collar there is no current across an onset to accumulate.
    _step(edge, state, 2e-4)
    assert edge.summary()["current"] == 0.0


def test_a_resume_without_a_fitting_record_judges_its_own_launch() -> None:
    space = LineGrid(200e-6, 128)
    beam = _gaussian_line(space, 0.0, 20e-6)
    for record in ({}, {"launch_energy": 1.0, "edge_baseline": [0.0]}):
        # No launch energy, or one baseline for a line's two faces: nothing is
        # restored, and the launch is recorded as a first execution's.
        edge = _edge(space, None)
        edge.resume(cast("dict[str, object]", record))
        _launched(edge, beam)
        assert edge.summary()["launch_energy"] == pytest.approx(_energy(edge, beam))


def test_a_resume_restores_the_record_and_takes_the_state_unjudged() -> None:
    edge, beam, noise = _noisy(3e-4)
    _launched(edge, beam + noise, noise=noise)
    record = edge.summary()
    space = cast("LineGrid", edge.space)
    # A stop is not carried into the execution that resumes after it: that one is
    # judged afresh against the original launch.
    stopped = _edge(space, None)
    stopped.resume(record | {"transverse_status": "stopped"})
    _launched(stopped, 2 * beam, noise=noise)
    assert stopped.status == "clear"
    assert stopped.summary()["launch_energy"] == record["launch_energy"]
    # A resume on a backend that does not read the edge is unwatched from there.
    unread = _edge(space, None)
    unread.resume(record)
    _launched(unread, beam + noise, noise=noise, reads=False)
    assert unread.status == "unwatched"
    assert unread.ceiling(0.0) == math.inf


def test_a_stop_on_a_slab_names_a_y_half_width() -> None:
    modes = [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]
    space = PlanarGrid(modes, 200e-6, 128)
    edge = _edge(space, None)
    _launched(edge, _slab(space, 0.0, 20e-6, 0.0))
    with pytest.raises(BoundaryError, match="a y half-width of at least") as caught:
        _read(edge, _scaled(edge, _slab(space, 195e-6, 4e-6, 0.0), 1e-2), 1e-3)
    assert "n_y of at least" in str(caught.value)


def _slab(space: PlanarGrid, centre: float, waist: float, kappa: float) -> np.ndarray:
    """Return a packet along y in every slab mode, on the slab's columns."""
    y = np.asarray(space.y)
    line = np.fft.fft(np.exp(-(((y - centre) / waist) ** 2) + 1j * kappa * y))
    return np.repeat(line, space.n_p)[:, None, None]


def _box(space: FreeGrid, centre: float, waist: float, kappa: float) -> np.ndarray:
    """Return a packet off-centre along x on a Cartesian box's columns."""
    x, y = np.meshgrid(np.asarray(space.x), np.asarray(space.y))
    field = np.exp(-(((x - centre) / waist) ** 2) - (y / waist) ** 2 + 1j * kappa * x)
    return np.fft.fft2(field).reshape(-1)[:, None, None]


@pytest.mark.parametrize("geometry", ["slab", "box"])
def test_a_collar_s_return_warns_then_stops_on_every_periodic_grid(
    geometry: str,
) -> None:
    # A packet in the strip travelling inward (exp(i kappa s) moves to smaller s
    # under exp(-i k_z z)): 1e-5 of the launch warns, 1e-2 stops (section 5.3).
    if geometry == "slab":
        modes = [SlabMode(63.5e-6, "Ar", 1.0, order=p) for p in (1, 3)]
        slab = PlanarGrid(modes, 200e-6, 240, collar_fraction=0.5)
        space: PlanarGrid | FreeGrid = slab

        def packet(centre: float, waist: float, kappa: float) -> np.ndarray:
            return _slab(slab, centre, waist, kappa)

    else:
        box = FreeGrid(200e-6, 120, 200e-6, 120, collar_fraction=0.5)
        space = box

        def packet(centre: float, waist: float, kappa: float) -> np.ndarray:
            return _box(box, centre, waist, kappa)

    edge = _edge(space, _collar(space, 100e-6))
    beam = packet(0.0, 20e-6, 0.0)
    _launched(edge, beam)
    inward = packet(230e-6, 10e-6, 3e5)
    with pytest.warns(BoundaryWarning, match="requested collar returns"):
        _read(edge, beam + _scaled(edge, inward, 1e-5), 1e-4)
    assert edge.status == "warned"
    with pytest.raises(BoundaryError, match="absorbing too little"):
        _read(edge, beam + _scaled(edge, inward, 1e-2), 2e-4)
    assert edge.status == "stopped"
    assert cast("float", edge.summary()["spatial_removed"]) > 0.0


def test_a_collar_narrower_than_its_strip_s_resolution_reads_no_return() -> None:
    # A collar of 2 um on a 10 um spacing: the strip holds fewer than the four nodes
    # a direction needs, and the onset's nearest node is the grid's last on both
    # faces, with no neighbour beyond it: neither reads anything.
    space = LineGrid(200e-6, 40, collar_fraction=0.01)
    edge = _edge(space, _collar(space, 2e-6))
    _launched(edge, _gaussian_line(space, 0.0, 20e-6))
    returning = _gaussian_line(space, 195e-6, 10e-6) * 10
    assert edge._strip_shares(edge.reduction.strip(returning)) == (0.0, 0.0)  # pyright: ignore[reportPrivateUsage]
    assert _current(edge, returning) == 0.0  # pyright: ignore[reportPrivateUsage]


def test_under_input_noise_a_collar_s_strip_is_judged_above_the_noise_s() -> None:
    # The strip's floor is three times what the input noise alone reads there.
    edge, space = _line_edge()
    beam = _packet(space, 0.0, 0.0)
    phases = np.random.default_rng(3).uniform(0.0, 2 * math.pi, beam.shape)
    noise = 1e-3 * float(np.max(np.abs(beam))) * np.exp(1j * phases)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        _launched(edge, beam + noise, noise=noise)
    floor = edge._strip_floor  # pyright: ignore[reportPrivateUsage]
    assert floor == pytest.approx(
        3 * edge._strip_shares(edge.reduction.strip(noise))[0]
    )  # pyright: ignore[reportPrivateUsage]
    assert floor > 0.0


@pytest.mark.parametrize("dtype", [np.complex128, np.complex64])
def test_row_energy_sums_each_rows_squared_magnitude(dtype: type) -> None:
    """A read's row sums equal ``sum |E|^2`` per row, in either precision.

    Double precision takes the real-view sum, including on a non-contiguous slice;
    single precision takes the plain one.
    """
    rng = np.random.default_rng(3)
    real, imaginary = rng.standard_normal((2, 5, 14))
    rows = (real + 1j * imaginary).astype(dtype)
    sliced = rows[:, ::2]
    expected = np.sum(np.abs(sliced.astype(np.complex128)) ** 2, axis=1)
    np.testing.assert_allclose(_row_energy(sliced), expected, rtol=1e-6)
