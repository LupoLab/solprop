# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The transverse edge's read spacing on Part II's witnesses (SEP 40 sections 5.3, 7.6).

A monitor reads only at committed steps, and a run with no source commits long ones, so
a packet can reach the wrap and come back between two reads. Section 5.3 bounds the gap
between two reads by ``min(L/16, 4 g / v_q)``, enforced as a step ceiling; a fixed
schedule cannot be held to it and reports ``coarse``.

Two of Part II's linear witnesses, on the line route in vacuum at 800 nm with no
collar (the default safety ramp alone) and noise off:

- the re-review's periodic packet: half-width 100 um, 256 samples, width 25 um, group
  velocity 0.2 (``kappa / k_z``), centred, over 1 mm;
- the scaled miss: half-width 1 mm, spacing 0.78 um (2560 samples), width 100 um,
  velocity 0.3, launched at -400 um, over 8.18 mm.

solprop propagates as ``exp(-i k_z z)``, so the launch ``exp(-i kappa x)`` travels
towards +x. A 100 fs pulse in a frame moving with the tilted packet keeps the time
window short. Part II's readings are at one frequency; where a reading sits on a
Gaussian tail, the band's spread of transverse velocities matters, and the readings are
taken on a 500 fs pulse instead.
"""

import dataclasses
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest

from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BoundaryError, BoundaryWarning
from solprop.grid import EnvelopeGrid, LineGrid
from solprop.linear import Frame
from solprop.monitor import BoundaryCheck
from solprop.output.schema import PlaneStore
from solprop.pulses import GaussianPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import FixedSchedule, Multistep, PIController, RungeKutta

C = 299792458.0
WAVELENGTH = 800e-9
K = 2 * math.pi / WAVELENGTH
GUARD = 0.05
"""The edge's width as a fraction of the outer half-width (section 5.3)."""
WARN, STOP = 1e-6, 1e-4
"""The default warning and stop, as energy shares of the launch."""


@dataclass(frozen=True)
class Packet:
    """A tilted Gaussian packet on a periodic line, and how far it runs.

    Attributes
    ----------
    half : float
        The periodic half-width, m.
    samples : int
        The line's samples.
    width : float
        The packet's 1/e field half-width, m.
    velocity : float
        Its group velocity ``kappa / k_z`` towards +x.
    centre : float
        Where it is launched, m.
    length : float
        The run's length, m.
    fwhm : float
        The pulse's duration, s.
    window : float
        The time window requested, s.
    """

    half: float
    samples: int
    width: float
    velocity: float
    centre: float
    length: float
    fwhm: float = 100e-15
    window: float = 0.5e-12

    @property
    def kappa(self) -> float:
        """Return the packet's transverse wavenumber, rad/m."""
        return K * self.velocity / math.sqrt(1 + self.velocity**2)

    def physical(self, space: LineGrid) -> np.ndarray:
        """Return the launch on `space`'s nodes, travelling towards +x."""
        x = np.asarray(space.x)
        return np.exp(-(((x - self.centre) / self.width) ** 2) - 1j * self.kappa * x)


REREVIEW = Packet(100e-6, 256, 25e-6, 0.2, 0.0, 1e-3)
MISS = Packet(1e-3, 2560, 100e-6, 0.3, -400e-6, 8.18e-3)


def packet_run(
    packet: Packet,
    *,
    integrator: Multistep | RungeKutta | None = None,
    schedule: FixedSchedule | None = None,
    check: BoundaryCheck | None = BoundaryCheck(),  # noqa: B008 - immutable
    planes: list[float] | None = None,
    space: LineGrid | None = None,
) -> Simulation:
    """Return `packet` as a free-space run on `space` (its own line by default).

    ``rtol = 1e-5``, ``dz_init = 1e-4 L``, ``auto_init`` and ``dz_max = L``: the
    settings of Part II's table of committed steps.
    """
    length = packet.length
    grid = EnvelopeGrid.from_limits(length, WAVELENGTH, (790e-9, 810e-9), packet.window)
    space = LineGrid(packet.half, packet.samples) if space is None else space
    pulse = GaussianPulse(wavelength=WAVELENGTH, fwhm=packet.fwhm, energy=1e-9)
    spectrum = pulse.build(grid)[0, 0]
    launch = np.fft.fft(packet.physical(space))[:, None, None] * spectrum[None, None, :]
    # The tilted packet's group delay is sqrt(1 + v^2) / c: moving with it keeps the
    # pulse in a short time window over the run.
    frame = Frame(
        2 * math.pi * C / WAVELENGTH, True, beta1=math.sqrt(1 + packet.velocity**2) / C
    )
    settings = SimulationSettings(
        controller=None if schedule is not None else PIController(rtol=1e-5, atol=0.0),
        save_planes=[length] if planes is None else planes,
        dz_max=length,
        dz_init=length * 1e-4,
        auto_init=True,
        integrator=RungeKutta() if integrator is None else integrator,
        backend="numpy_ref",
        schedule=schedule,
        boundary_check=check,
    )
    return Simulation.from_free(
        grid,
        space,
        SpatialMedium(1.0),
        [],
        launch,
        settings=settings,
        frame=frame,
        noise=None,
    )


def faces(space: LineGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return the line's two edge faces: the outer 5 % of the outer half-width."""
    x = np.asarray(space.x)
    dx = float(x[1] - x[0])
    outer = 0.5 * space.n_x * dx
    return x <= -(1 - GUARD) * outer, x >= (1 - GUARD) * outer - 0.5 * dx


def exact_faces(packet: Packet, z: np.ndarray) -> np.ndarray:
    """Return each face's share of the packet above its launch, ``(n_z, 2)``.

    The exact linear flow at 800 nm, ``E(kappa, z) = E(kappa, 0) exp(-i (k_z - k) z)``;
    the safety ramp is far above these packets and left out.
    """
    space = LineGrid(packet.half, packet.samples)
    kx = np.asarray(space.k_x)
    k_z = np.sqrt((K * K - kx * kx).astype(complex))
    start = np.fft.fft(packet.physical(space))
    fields = np.fft.ifft(start[None] * np.exp(-1j * np.outer(z, k_z - K)), axis=1)
    power = np.abs(fields) ** 2
    launch = np.abs(packet.physical(space)) ** 2
    total = float(np.sum(launch))
    shares = np.stack([np.sum(power[:, face], axis=1) for face in faces(space)], 1)
    baseline = np.array([np.sum(launch[face]) for face in faces(space)])
    return (shares - baseline[None, :]) / total


def true_stop(packet: Packet) -> float:
    """Return where a face's share above its launch first passes the stop, m."""
    z = np.linspace(0.0, packet.length, 4097)
    above = np.nonzero(np.max(exact_faces(packet, z), axis=1) > STOP)[0]
    assert above.size
    return float(z[above[0]])


def stopped_at(simulation: Simulation) -> float:
    """Run, expecting the transverse edge to stop it; return where it stopped, m."""
    with (
        warnings.catch_warnings(),
        pytest.raises(BoundaryError, match="reached the transverse edge") as caught,
    ):
        warnings.simplefilter("ignore", BoundaryWarning)
        simulation.run(progress="none")
    found = re.search(r"at z = ([0-9.eE+-]+) m", str(caught.value))
    assert found is not None
    return float(found.group(1))


def finished(simulation: Simulation) -> PlaneStore:
    """Run to the end with the boundary warnings silenced."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BoundaryWarning)
        output = simulation.run(progress="none")
    assert isinstance(output, PlaneStore)
    return output


def edge_record(output: PlaneStore) -> dict[str, Any]:
    """Return a finished run's transverse edge record."""
    assert output.outcome is not None
    monitor = output.outcome.execution["boundary_monitor"]
    assert isinstance(monitor, dict)
    return cast("dict[str, Any]", monitor["transverse"])


def committed(output: PlaneStore) -> np.ndarray:
    """Return a run's committed positions, m."""
    accepted = np.asarray(output["step_history/accepted"], dtype=bool)
    return np.cumsum(np.asarray(output["step_history/h"])[accepted])


# ------------------------------------------------------------- the re-review's packet
@pytest.mark.medium
def test_the_rereview_packet_reads_as_part_ii_tabulates() -> None:
    # Part II's witness: the edge share at z = 0, L/2 and L, and the field error at L
    # against a bare line eight times wider at the same spacing. +-5 %: the same linear
    # operator. At L the packet is back at the centre and the edge reads its tails,
    # where a 100 fs band's spread of positions (about 1 um) would add 20 %; on a
    # 500 fs pulse it adds under 1 %.
    packet = dataclasses.replace(REREVIEW, fwhm=500e-15, window=2e-12)
    length = packet.length
    space = LineGrid(packet.half, packet.samples)
    output = finished(
        packet_run(
            packet, integrator=Multistep(), check=None, planes=[length / 2, length]
        )
    )
    fields = [
        np.fft.ifft(np.asarray(output["field"][i])[:, 0, :], axis=0) for i in range(3)
    ]
    launch = float(np.sum(np.abs(fields[0]) ** 2))
    low, high = faces(space)
    edge = [float(np.sum(np.abs(f[low | high]) ** 2)) / launch for f in fields]
    assert edge[0] == pytest.approx(2.95e-14, rel=0.05, abs=0.0)
    assert edge[1] == pytest.approx(0.309, rel=0.05, abs=0.0)
    assert edge[2] == pytest.approx(3.92e-12, rel=0.05, abs=0.0)

    wide = LineGrid(8 * packet.half, 8 * packet.samples)
    reference = finished(_bare(packet, wide))
    first = (wide.n_x - space.n_x) // 2
    truth = np.fft.ifft(np.asarray(reference["field"][-1])[:, 0, :], axis=0)[
        first : first + space.n_x
    ]
    error = math.sqrt(float(np.sum(np.abs(fields[2] - truth) ** 2)) / launch)
    assert error == pytest.approx(1.00, rel=0.05)


def _bare(packet: Packet, space: LineGrid) -> Simulation:
    """Return `packet` on `space` with no transverse boundary at all, unmonitored."""
    simulation = packet_run(packet, integrator=Multistep(), check=None, space=space)
    simulation.settings = dataclasses.replace(
        simulation.settings, spatial_absorber=False
    )
    return simulation


@pytest.mark.parametrize("integrator", [Multistep(), RungeKutta()], ids=["ms", "rk"])
def test_the_rereview_packet_is_stopped_at_the_first_read_past_the_truth(
    integrator: Multistep | RungeKutta,
) -> None:
    # Section 7.6: under real adaptive runs it is read above the warning. Its edge
    # passes the stop near z = 0.24 mm, and the reads are L/16 apart: the stop is the
    # first read after the exact flow's crossing.
    packet = REREVIEW
    z = stopped_at(packet_run(packet, integrator=integrator))
    assert z - packet.length / 16 * (1 + 1e-9) < true_stop(packet) <= z


def test_a_long_fixed_schedule_over_the_packet_is_coarse() -> None:
    # Section 5.3: a fixed schedule cannot be held to the spacing (L/16 here). Steps
    # of 0.1, 0.8 and 0.1 mm read either side of the wrap, where the packet's edge
    # share stays below the warning, and at L, where it is back at the centre and the
    # run is wrong by its whole field: it reports coarse with its largest gap, never
    # clear.
    packet = REREVIEW
    schedule = FixedSchedule(np.array([0.1e-3, 0.8e-3, 0.1e-3]))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", BoundaryWarning)
        output = packet_run(packet, schedule=schedule).run(progress="none")
    assert isinstance(output, PlaneStore)
    record = edge_record(output)
    assert record["transverse_status"] == "coarse"
    assert record["reads"] == 3
    assert record["largest_gap"] == pytest.approx(0.8e-3, rel=1e-9, abs=0.0)
    assert max(record["edge_max"]) < WARN
    assert any("fixed schedule's longest step" in str(one.message) for one in caught)


def test_a_fixed_schedule_reading_on_the_wrap_stops() -> None:
    # The same schedule's coarseness is a report, not a pass: steps of L/4 read at
    # L/2, where the packet sits on the wrap, and the stop wins.
    packet = REREVIEW
    stopped_at(
        packet_run(
            packet,
            schedule=FixedSchedule.uniform(0.0, packet.length, packet.length / 4),
        )
    )


# ---------------------------------------------------------------- the scaled miss
@pytest.mark.parametrize(
    ("integrator", "count", "largest", "last"),
    [
        (Multistep(), 14, 0.295, [0.205, 0.41, 0.705, 1.0]),
        pytest.param(
            RungeKutta(),
            27,
            0.098,
            [0.796, 0.885, 0.984, 1.0],
            marks=pytest.mark.medium,
        ),
    ],
    ids=["ms", "rk"],
)
def test_the_scaled_miss_steps_as_part_ii_tabulates(
    integrator: Multistep | RungeKutta, count: int, largest: float, last: list[float]
) -> None:
    # Part II's table of committed steps, unmonitored so with no ceiling: their count,
    # the largest over L and the last four positions over L, to Part II's rounding
    # (1 %). Read there with the exact flow, the Multistep run never sees the packet
    # cross the wrap (Part II: 2e-8), though the truth passes the stop: why the
    # ceiling exists. RK's commits are dense enough to catch it.
    packet = MISS
    output = finished(packet_run(packet, integrator=integrator, check=None))
    positions = committed(output) / packet.length
    assert positions.size == count
    np.testing.assert_allclose(positions[-4:], last, rtol=0.01, atol=0.0)
    gaps = np.diff(np.r_[0.0, positions])
    assert float(np.max(gaps)) == pytest.approx(largest, rel=0.01, abs=0.0)
    reads = float(np.max(exact_faces(packet, np.r_[0.0, positions * packet.length])))
    assert true_stop(packet) < packet.length
    if isinstance(integrator, Multistep):
        assert reads < WARN
    else:
        assert reads > STOP


@pytest.mark.parametrize(
    "integrator",
    [Multistep(), pytest.param(RungeKutta(), marks=pytest.mark.medium)],
    ids=["ms", "rk"],
)
def test_the_scaled_miss_is_stopped_under_the_ceiling(
    integrator: Multistep | RungeKutta,
) -> None:
    # Part II's requirement: above the stop under the ceiling. The reads are
    # min(L/16, 4 g / v_q) = L/16 apart here, and the stop is the first read after
    # the exact flow's crossing.
    packet = MISS
    z = stopped_at(packet_run(packet, integrator=integrator))
    assert z - packet.length / 16 * (1 + 1e-9) < true_stop(packet) <= z


def test_the_scaled_miss_is_read_far_above_the_stop() -> None:
    # With the edge's budget relaxed the run completes, and the record shows what
    # the ceiling's reads saw: sixteen of them, at most L/16 apart, the largest far
    # above the stop (Part II: 0.54 over both faces at its own read positions; one
    # face here holds at most about 0.38 of a packet this wide).
    packet = MISS
    output = finished(
        packet_run(
            packet,
            integrator=Multistep(),
            check=BoundaryCheck(transverse_budget=10.0),
        )
    )
    record = edge_record(output)
    assert record["reads"] >= 16
    assert record["largest_gap"] <= packet.length / 16 * (1 + 1e-9)
    assert max(record["edge_max"]) > 1e3 * STOP


def test_the_scaled_miss_on_its_own_committed_steps_is_coarse() -> None:
    # Part II's committed positions as a fixed schedule: every read misses the
    # crossing, as unmonitored, but the run reports coarse with its largest gap,
    # 0.295 L against the spacing L/16, rather than clear.
    packet = MISS
    # The step doubles from dz_init = 1e-4 L up to 0.4095 L, then 0.70475 L and L.
    edges = np.r_[0.0, 1e-4 * (2.0 ** np.arange(1, 13) - 1), 0.70475, 1.0]
    schedule = FixedSchedule(np.diff(edges) * packet.length)
    output = finished(packet_run(packet, schedule=schedule))
    record = edge_record(output)
    assert record["transverse_status"] == "coarse"
    assert record["largest_gap"] == pytest.approx(
        0.29525 * packet.length, rel=1e-9, abs=0.0
    )
    assert max(record["edge_max"]) < WARN
