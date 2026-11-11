# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Split a field-free point of a Kerr scan: the march, the check, the crossings.

One input of a delay scan through :class:`solprop.driver.SimulationRunner`, with no
fields kept: a fused-silica Kerr beam at 260 nm on fixed low-storage RK4IP steps, with
and a stated input identity. It runs with ``boundary_check=None`` and
with the default check, budgets relaxed so that nothing stops, and profiles one point
of each. Then it times what a plane can ask of a device through the native session
(``solprop_native.session``): a state's copy to the host, the launch plane's field
exported in the internal convention and in the stored one, and the common statistics,
each the launch of a session whose request asks for that statistic alone.

Usage::

    uv run python -m benchmarks.device_split NX [--steps 8] [--planes 4] [--out FILE]

It prints one JSON object per section (``record``, ``setup``, ``point`` twice,
``calls``), and writes them to ``FILE`` too when given. ``point`` names each function's
calls, own and cumulative seconds, and ``calls`` each crossing's minor page faults.
The profiler inflates Python-heavy work, so compare ``walls_s``, which it doesn't run
under. A statistic's time is a session's start and launch plane, the statistic evaluated
natively there: subtract ``none``'s, which asks for none, for the statistic's share.
``calls`` also splits a width: its native moments, and what the same azimuthal average
takes on the host. The native backend is required, and a device build is what
it's for; on a CPU build the crossings are plain copies.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import platform
import pstats
import statistics
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from solprop.backends.events import ObservationReady
from solprop.backends.export import ExportRecipe
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.constants import EPS0, MU0, C
from solprop.driver import Simulation, SimulationSettings
from solprop.grid import EnvelopeGrid, FreeGrid
from solprop.monitor import BoundaryCheck
from solprop.nonlinear.responses import KerrResponse
from solprop.physdata.glasses import SellmeierGlass
from solprop.pulses import GaussianPulse, SpatialPulse
from solprop.spatial import SpatialMedium, cartesian_fluence_widths
from solprop.stats import StatsSpec, default_spec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, PIController

__all__ = ["main"]

STEP = 0.5e-6
"""The fixed step, m."""

NAMED = (
    "_read_transverse",
    "_edge_reading",
    "node_power",
    "column_power",
    "_spacing",
    "_observed_field",
    "_boundary_energies",
    "_take_step_stats",
    "_monitor_plane",
    "configuration_hash",
    "_hashed_record",
    "_check_replaced_input",
    "attach_monitor",
    "replay_step",
    "c_einsum",
    "field_to_host",
)
"""The functions a point's profile reports by name."""

GLASS = SellmeierGlass(
    name="malitson",
    source="Malitson 1965",
    wavelength_range=(0.21e-6, 3.71e-6),
    citation="Malitson, JOSA 55, 1205 (1965), doi:10.1364/JOSA.55.001205",
    poles=((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161)),
)


def _shell(command: str) -> str:
    return subprocess.run(
        command, shell=True, capture_output=True, text=True, check=False
    ).stdout.strip()


def _simulation(
    nx: int,
    steps: int,
    planes: int,
    check: BoundaryCheck | None,
    stats: StatsSpec | None = None,
) -> Simulation:
    length = STEP * steps
    half = 183e-6 * nx / 384 * 1.1
    grid = EnvelopeGrid.from_limits(
        length, 260e-9, (143e-9, 600e-9), 200e-15, thg=False, fft_size="smooth"
    )
    positions = [length * (index + 1) / planes for index in range(planes)]
    settings = SimulationSettings(
        spatial_absorber=False,
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=positions,
        dz_max=length,
        dz_init=STEP,
        schedule=FixedSchedule.uniform(0.0, length, STEP, save_planes=positions),
        integrator=LowStorageRungeKutta(),
        backend="native",
        boundary_check=check,
        fields="none",
        stats=StatsSpec(names=("energy",)) if stats is None else stats,
    )
    return Simulation.from_free(
        grid,
        FreeGrid(half, nx, half, nx, collar_fraction=0.0),
        SpatialMedium(GLASS),
        [KerrResponse(chi3=2.2e-22 * MU0 * EPS0 * C**2, thg=False)],
        SpatialPulse(GaussianPulse(wavelength=260e-9, fwhm=30e-15, energy=1e-7), 40e-6),
        settings=settings,
    )


def _profile_summary(profile: cProfile.Profile, top: int = 30) -> dict[str, Any]:
    rows, named = [], {}
    # `Stats.stats` is pstats' documented table, untyped in its stubs.
    table: dict[Any, Any] = pstats.Stats(profile).stats  # pyright: ignore[reportAttributeAccessIssue]
    for (path, line, function), (_, calls, own, cumulative, _) in table.items():
        where = f"{function}@{Path(path).name}:{line}"
        rows.append((own, cumulative, calls, where))
        if any(name in function for name in NAMED):
            named[where] = [calls, round(own, 3), round(cumulative, 3)]
    rows.sort(reverse=True)
    return {
        "named": named,
        "top_own": [[round(a, 3), round(b, 3), c, d] for a, b, c, d in rows[:top]],
    }


def _minor_faults() -> int:
    """Return this process's minor page faults so far: none are counted on Windows.

    Returns
    -------
    int
        The count.
    """
    if sys.platform == "win32":
        return 0
    import resource  # noqa: PLC0415 - not on Windows

    return resource.getrusage(resource.RUSAGE_SELF).ru_minflt


def _timed(call: Any, repeats: int) -> dict[str, float]:
    times, faults = [], 0
    for _ in range(repeats):
        before = _minor_faults()
        started = time.perf_counter()
        value = call()
        times.append(time.perf_counter() - started)
        faults = _minor_faults() - before
        del value
    return {
        "median_ms": round(1e3 * statistics.median(times), 2),
        "first_ms": round(1e3 * times[0], 2),
        "minor_faults": faults,
    }


def _native(simulation: Simulation) -> tuple[Any, Any, Any]:
    """Return a simulation's problem, run request and native backend.

    Parameters
    ----------
    simulation : solprop.driver.Simulation
        The simulation.

    Returns
    -------
    tuple
        The translated problem, its run request and a constructed native backend,
        which the caller closes.
    """
    problem = simulation.translate()
    request = simulation._run_request()
    backend = select(
        "native",
        problem,
        request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    return problem, request, backend


def _launch(backend: Any, problem: Any, request: Any) -> Any:
    """Return a call that starts a session and takes its launch plane.

    Parameters
    ----------
    backend : solprop_native.backend.NativeBackend
        The backend, which each session leaves open.
    problem, request : object
        The run's problem and request.

    Returns
    -------
    collections.abc.Callable
        The call, returning the launch's statistics.
    """
    prepared = backend.prepare(problem, request)

    def launch() -> object:
        with prepared.start(Start.entrance(problem), owns=False) as session:
            while True:
                batch = session.advance()
                for event in batch.events:
                    if isinstance(event, ObservationReady):
                        return dict(event.stats)
                session.acknowledge(batch.receipt)

    return launch


def _calls(nx: int, steps: int, planes: int) -> dict[str, Any]:
    specs = {"none": StatsSpec(names=())}
    specs.update(
        (name, StatsSpec(names=(name,)))
        for name in (
            "energy",
            "beam_spectrum",
            "absorbed_power",
            "on_axis_intensity",
            "peak_intensity",
            "beam_radius",
            "fwhm_r",
        )
    )
    specs["default"] = default_spec("free")
    problem, request, backend = _native(_simulation(nx, steps, planes, None))
    geometry = problem.transform.geometry
    x, y = geometry.axes
    state = backend._native_state
    try:
        with backend.prepare(problem, request).start(
            Start.entrance(problem), owns=False
        ) as session:
            batch = session.advance()
            ticket = None
            while True:
                ticket = next(
                    (
                        event.ticket
                        for event in batch.events
                        if isinstance(event, ObservationReady)
                    ),
                    None,
                )
                if ticket is not None:
                    break
                session.acknowledge(batch.receipt)
                batch = session.advance()
            out = np.empty(problem.shape, dtype=np.complex128)
            exports = {
                "export_internal": _timed(
                    lambda: session.export(
                        ticket, ExportRecipe(power=False, optics=False), out
                    ),
                    3,
                ),
                "export_stored": _timed(
                    lambda: session.export(
                        ticket, ExportRecipe(power=False, optics=True), out
                    ),
                    3,
                ),
            }
            session.acknowledge(batch.receipt)
        crossings = {"field_to_host": _timed(state.field_to_host, 3), **exports}
        # A width's parts: the fluence and its moments, one native call; and what the
        # host's azimuthal average of a Gaussian's fluence takes on this machine.
        widths = {
            "moments": _timed(
                lambda: state.spatial_statistics(32, 0.5 * EPS0 * C, 0.0), 5
            ),
            "host_average": _timed(
                lambda: cartesian_fluence_widths(
                    x,
                    y,
                    np.exp(-2 * (x[None, :] ** 2 + y[:, None] ** 2) / 40e-6**2).ravel(),
                ),
                3,
            ),
        }
    finally:
        backend.close()
    statistics_ms: dict[str, Any] = {}
    for name, spec in specs.items():
        problem, request, backend = _native(
            _simulation(nx, steps, planes, None, stats=spec)
        )
        try:
            statistics_ms[name] = _timed(_launch(backend, problem, request), 3)
        finally:
            backend.close()
    return {"crossings": crossings, "statistics": statistics_ms, "widths": widths}


def main(argv: list[str] | None = None) -> None:
    """Run the split and print its sections.

    Parameters
    ----------
    argv : list of str or None, optional
        The arguments, by default the process's.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nx", type=int, help="transverse samples per axis")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--planes", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args(argv)
    warnings.simplefilter("ignore")
    sink = arguments.out.open("w", encoding="utf-8") if arguments.out else None

    def emit(section: str, **values: Any) -> None:
        line = json.dumps({"section": section, "nx": arguments.nx, **values})
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
        if sink is not None:
            sink.write(line + "\n")
            sink.flush()

    import solprop_native  # noqa: PLC0415  # pyright: ignore[reportMissingImports]

    emit(
        "record",
        cpu=_shell("lscpu | grep 'Model name' | cut -d: -f2 | xargs"),
        cores=_shell("nproc"),
        uptime=_shell("uptime"),
        machine=platform.machine(),
        transparent_hugepages=_shell("cat /sys/kernel/mm/transparent_hugepage/enabled"),
        commit=_shell("git rev-parse --short HEAD"),
        device=solprop_native.build_info().device_backend,
    )
    started = time.perf_counter()
    base = _simulation(arguments.nx, arguments.steps, arguments.planes, None)
    built = time.perf_counter()
    problem = base.translate()
    translated = time.perf_counter()
    launch = np.array(problem.initial.field0)
    emit(
        "setup",
        from_free_s=round(built - started, 2),
        translate_s=round(translated - built, 2),
        field_GiB=round(launch.nbytes / 2**30, 3),
        shape=list(launch.shape),
    )
    relaxed = BoundaryCheck(budget=1.0, transverse_budget=1.0)
    for name, check in (("off", None), ("on", relaxed)):
        simulation = _simulation(arguments.nx, arguments.steps, arguments.planes, check)
        with simulation.runner() as runner:
            walls = []
            for point in range(2):
                field = launch * np.exp(1j * point)
                started = time.perf_counter()
                runner.run(field, progress="none", input_identity=f"point-{point}")
                walls.append(round(time.perf_counter() - started, 3))
            field = launch * np.exp(0.5j)
            profile = cProfile.Profile()
            profile.enable()
            runner.run(field, progress="none", input_identity="profiled")
            profile.disable()
            emit("point", check=name, walls_s=walls, **_profile_summary(profile))
    emit("calls", **_calls(arguments.nx, arguments.steps, arguments.planes))
    if sink is not None:
        sink.close()


if __name__ == "__main__":
    main()
