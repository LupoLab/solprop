# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The pinned backend comparison corpus: its cases and how one is recorded.

Before the reference is restructured and native's seam rewritten (SEP 52 phases 3 and
4), every march and every route family is run on short fixed schedules, on both
backends, and what each run leaves is recorded (``tests/data/backend_corpus.npz``,
written by ``tests/data/generate_backend_corpus.py``):

- the last saved field, the planes' positions and their statistics, and the step
  history;
- the boundary monitor's record, the split collar's removal and occupancies and the
  transverse edge's record among them;
- a resume: the run stopped at its middle plane and continued from its file, with the
  continuation's last field and monitor record;
- the timing probe's trials and their read-backs, or the low-storage march's refusal of
  a read-back.

Each backend is compared with its own record
(``tests/backends/test_backend_corpus.py``). Everything here runs through the public
paths -- :meth:`Simulation.run`, a file resumed by :meth:`Simulation.run`, and
:meth:`Backend.timing_probe` on a backend selected by the registry -- so that a
restructure behind them leaves the cases valid.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import platform
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from builders import grid_at, held_collar
from capability_routes import general_constant
from solprop.backends.registry import select
from solprop.backends.session import Start
from solprop.boundaries import SpatialAbsorber
from solprop.constants import C
from solprop.driver import Simulation, SimulationSettings
from solprop.exceptions import BackendError, SolpropWarning
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.log import NullReporter
from solprop.modes import PlanarGrid, SlabMode
from solprop.monitor import CHECKS, BoundaryCheck
from solprop.nonlinear.responses import KerrResponse
from solprop.output.hdf5 import open_file, read_resume_state
from solprop.output.records import HISTORY_FIELDS
from solprop.output.store import PlaneStore
from solprop.physdata import gamma3
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stats import StatsSpec
from solprop.stepper import FixedSchedule, LowStorageRungeKutta, RungeKutta
from solprop.stepper.multistep import Multistep

__all__ = [
    "CASES",
    "THREAD_VARIABLES",
    "Case",
    "Recorded",
    "identity",
    "record",
]

STEPS = 10
"""Fixed steps of every scheduled case (``AGENTS.md``: about ten steps)."""

RELAXED = BoundaryCheck(budget=1.0, stop_factor=1.0, transverse_budget=1.0)
"""A check that records everything and neither warns nor stops: the record is the
point, and the cases put light in the collar on purpose."""

HISTORY = tuple(name for name in HISTORY_FIELDS if name != "wall_time")
"""The step history's columns a run is compared on; its wall time is not a result."""


def _schedule(length: float, planes: tuple[float, ...]) -> FixedSchedule:
    """Return :data:`STEPS` equal steps over `length` that land on `planes`."""
    return FixedSchedule.uniform(0.0, length, length / STEPS, save_planes=planes)


# ----------------------------------------------------------------------- the routes

FIBRE = 0.02
"""The fibre's length, m: ten steps of 2 mm, a fiftieth of a metre's run."""

FIBRE_BETA2 = 2e-25
"""s^2/m: the walk comes from a carrier offset, and a 0.07 m dispersion length for the
200 fs pulse keeps it compact over the fibre."""

SPARSE = 100.0
"""How much longer than the fibre the run is that the fibre's checks and collar are
pinned to: the boundary monitor checks a quiet window every ``SPARSE FIBRE / 256``
(``monitor.CHECKS``, through ``BoundaryCheck.interval``), about every fourth step, and
the saved planes' states regardless, so the split collar's removal adds over several
accepted steps between two readings; and the collar keeps the default rate of a grid
that long (``builders.held_collar``), which is weaker than the fibre's own. The grid
itself ends at the fibre, where the run does (SEP 51)."""


def walking_fibre(
    backend: str,
    integrator: RungeKutta | Multistep | LowStorageRungeKutta,
    *,
    planes: tuple[float, ...] = (FIBRE / 2, FIBRE),
    scheduled: bool = True,
    interpolated: bool = False,
    rtol: float = 1e-7,
) -> Simulation:
    """Return a fibre whose pulse walks into the split collar, checked sparsely.

    The GNLSE route on an envelope grid. Launched 0.1 ps late at a carrier offset under
    :data:`FIBRE_BETA2`, the pulse's group delay walks a further 0.3 ps of its 1 ps
    window over the fibre, so the ``"split_step"`` collar takes some of it at every
    step from the first and the high guard fills: the split collar's removal and
    occupancies are not empty. Its
    nonlinear phase is about one radian. The schedule is :data:`STEPS` equal steps
    landing on `planes`, or with `interpolated` on its own edges only, the planes then
    observed by dense output; unscheduled, the march is adaptive at `rtol`.
    """
    pinned = EnvelopeGrid.from_limits(SPARSE * FIBRE, 1e-6, (0.8e-6, 1.25e-6), 1e-12)
    grid = grid_at(pinned, FIBRE)
    omega0 = 2 * np.pi * C / 1e-6
    offset = 0.3e-12 / (FIBRE_BETA2 * FIBRE)
    extra: dict[str, Any] = {"interpolate_planes": interpolated}
    if not scheduled:
        extra |= {"rtol": rtol, "atol": 0.0, "dz_init": FIBRE / 4}
    elif interpolated:
        extra["schedule"] = FixedSchedule(np.full(STEPS, FIBRE / STEPS))
    else:
        extra["schedule"] = _schedule(FIBRE, planes[:-1])
    return Simulation.from_gnlse(
        50.0, FIBRE, [0.0, 0.0, FIBRE_BETA2], wavelength=1e-6, grid=grid,
        pulses=[GaussianPulse(
            wavelength=2 * np.pi * C / (omega0 + offset), fwhm=200e-15,
            peak_power=1.0, phases=[0.0, 0.1e-12],
        )],
        noise=None, raman=False, shock=False, save_planes=list(planes),
        backend=backend, integrator=integrator, absorber_scheme="split_step",
        absorber=held_collar(grid, SPARSE * FIBRE),
        boundary_check=dataclasses.replace(RELAXED, interval=SPARSE * FIBRE / CHECKS),
        **extra,
    )  # fmt: skip


CAPILLARY = 2e-3
"""The capillary's length, m."""


def capillary(
    backend: str,
    integrator: Multistep | RungeKutta,
    *,
    modes: int = 1,
    envelope: bool = False,
    plasma: bool = False,
    interpolated: bool = False,
    rtol: float = 1e-7,
) -> Simulation:
    """Return a 2 mm argon capillary whose short pulse reaches the split collar.

    A 10 uJ, 15 fs pulse in a 50 um capillary of 2 bar of argon, on an 80 fs window:
    the pulse's tails and, with several modes, their group delays put light in the
    ``"split_step"`` collar within the ten steps. With `interpolated` the march is
    adaptive at `rtol` and saves at planes inside its steps, observed by dense output,
    which a fixed schedule does not offer.
    """
    limits = (300e-9, 2e-6) if not envelope else (500e-9, 1.3e-6)
    window = 80e-15
    grid = (
        EnvelopeGrid.from_limits(CAPILLARY, 800e-9, limits, window)
        if envelope
        else FieldGrid.from_limits(CAPILLARY, 800e-9, limits, window)
    )
    extra: dict[str, Any]
    if interpolated:
        # Dense observation is refused beside the split collar, whose factor is
        # applied once per accepted step: the in-stage collar, the default, instead.
        planes = (0.35 * CAPILLARY, 0.75 * CAPILLARY, CAPILLARY)
        extra = {"rtol": rtol, "atol": 1e-12, "interpolate_planes": True}
    else:
        planes = (CAPILLARY / 2, CAPILLARY)
        extra = {
            "schedule": _schedule(CAPILLARY, planes[:1]),
            "absorber_scheme": "split_step",
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        return Simulation.from_capillary(
            50e-6, CAPILLARY, "Ar", 2.0, grid=grid, wavelength=800e-9, fwhm=15e-15,
            energy=10e-6, modes=modes, envelope=envelope, raman=False, plasma=plasma,
            noise=None, save_planes=list(planes), backend=backend,
            integrator=integrator, boundary_check=RELAXED, **extra,
        )  # fmt: skip


SPATIAL = 2e-3
"""The free-space and planar cases' length, m."""


def _spatial_settings(
    backend: str,
    integrator: Multistep | RungeKutta | LowStorageRungeKutta,
    **overrides: Any,
) -> SimulationSettings:
    """Return :data:`STEPS` fixed steps over :data:`SPATIAL`, its middle plane saved."""
    planes = (SPATIAL / 2, SPATIAL)
    schedule = _schedule(SPATIAL, planes[:1])
    keywords: dict[str, Any] = {
        "save_planes": list(planes),
        "schedule": schedule,
        "dz_max": schedule.max_step,
        "dz_init": schedule.max_step,
        "backend": backend,
        "integrator": integrator,
        "absorber_scheme": "split_step",
        "boundary_check": RELAXED,
    }
    return SimulationSettings(**(keywords | overrides))


def _beam(energy: float, waist: float, propz: float = 0.0) -> SpatialPulse:
    """Return a 40 fs Gaussian beam at 800 nm, `propz` from its waist (m)."""
    return SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=40e-15, energy=energy),
        waist,
        propz=propz,
    )


def radial(
    backend: str, integrator: Multistep | RungeKutta | LowStorageRungeKutta
) -> Simulation:
    """Return a 20 uJ beam of 80 um waist in 5 bar of argon on 16 Hankel nodes.

    Kerr self-focusing over 2 mm; the pulse's spreading in time reaches the split
    collar of its 120 fs field window.
    """
    grid = FieldGrid.from_limits(SPATIAL, 800e-9, (600e-9, 1.2e-6), 120e-15)
    return Simulation.from_free(
        grid,
        RadialGrid(4e-4, 16),
        SpatialMedium("Ar", 5.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        _beam(20e-6, 80e-6),
        settings=_spatial_settings(backend, integrator, spatial_absorber=False),
        noise=None,
    )


def cartesian(
    backend: str, integrator: Multistep | RungeKutta | LowStorageRungeKutta
) -> Simulation:
    """Return a 2 uJ Kerr beam of 60 um waist in 5 bar of argon on an 8 by 8 plane.

    An envelope grid of a narrow band keeps the state small; no transverse collar is
    asked for (the coarse plane launches into the default safety ramp's band, as the
    catalogue's Cartesian route does).
    """
    grid = EnvelopeGrid.from_limits(SPATIAL, 800e-9, (700e-9, 900e-9), 100e-15)
    return Simulation.from_free(
        grid,
        FreeGrid(3e-4, 8, 3e-4, 8, collar_fraction=0),
        SpatialMedium("Ar", 5.0),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        _beam(2e-6, 60e-6),
        settings=_spatial_settings(backend, integrator, spatial_absorber=False),
        noise=None,
    )


def line_edge(
    backend: str, integrator: Multistep | RungeKutta | LowStorageRungeKutta
) -> Simulation:
    """Return a converging beam that reaches a real-space collar on the line route.

    SEP 40's requested collar outside a 300 um aperture of 40 points
    (``backends/edge_collar_cases.py``, whose Cartesian plane would be too large to
    record): the beam converges from 24 mm before its 60 um waist, so its wings reach
    the collar and the transverse edge's record holds an inward current above
    rounding. Linear, as the edge cases are.
    """
    grid = EnvelopeGrid.from_limits(SPATIAL, 800e-9, (700e-9, 900e-9), 100e-15)
    return Simulation.from_free(
        grid,
        LineGrid(300e-6, 40, collar_fraction=1.0),
        SpatialMedium(1.0),
        [],
        _beam(1e-9, 60e-6, propz=-2.4e-2),
        # Recorded without statistics, before the route took its default ones.
        settings=_spatial_settings(
            backend,
            integrator,
            spatial_absorber=SpatialAbsorber(kappa=8e4),
            stats=StatsSpec(names=()),
        ),
        noise=None,
    )


def planar(
    backend: str, integrator: Multistep | RungeKutta | LowStorageRungeKutta
) -> Simulation:
    """Return the hybrid planar route: two slab modes of an argon gap on 8 y points.

    ``capability_routes.planar`` on a 300 fs window and over 2 mm.
    """
    grid = EnvelopeGrid.from_limits(SPATIAL, 800e-9, (650e-9, 1000e-9), 300e-15)
    half_gap = 63.5e-6
    modes = [SlabMode(half_gap, "Ar", 1.0, order=p) for p in (1, 3)]
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=60e-15, energy=1e-3),
        400e-6,
        waist_x=0.735 * half_gap,
    )
    return Simulation.from_planar(
        grid,
        PlanarGrid(modes, 1e-3, 8),
        [KerrResponse(gamma3("Ar", wavelength=800e-9), thg=False)],
        pulse,
        # Recorded without statistics, before the route took its default ones.
        settings=_spatial_settings(
            backend, integrator, spatial_absorber=False, stats=StatsSpec(names=())
        ),
    )


def general_modal(
    backend: str, integrator: Multistep | RungeKutta | LowStorageRungeKutta
) -> Simulation:
    """Return the general modal route: ``capability_routes.general_constant`` scheduled.

    One HE11 channel of a 1.5 um strand with a Kerr core, over the catalogue's 0.2 mm.
    """
    simulation = general_constant(None)
    length = float(np.asarray(simulation.settings.save_planes)[-1])
    planes = (length / 2, length)
    schedule = _schedule(length, planes[:1])
    simulation.settings = dataclasses.replace(
        simulation.settings,
        backend=backend,
        integrator=integrator,
        schedule=schedule,
        save_planes=np.asarray(planes),
        dz_max=schedule.max_step,
        dz_init=schedule.max_step,
        absorber_scheme="split_step",
        boundary_check=RELAXED,
    )
    return simulation


FIXED_ADAMS = Multistep(variable_order=False)
"""The exponential Adams method at its top order: no order choice to fork."""

CARRIER = Multistep(carrier_frame="scalar")
"""The exponential Adams method under the carrier frame (SEP 35), adaptive."""


@dataclass(frozen=True)
class Case:
    """One corpus case: a route on a march, and how its run is recorded.

    Attributes
    ----------
    name : str
        Its key in the corpus.
    route : str
        The route family it covers.
    march : str
        The march it covers.
    build : Callable
        The simulation, given the backend's registry name.
    adaptive : bool
        Whether the march chooses its own steps; then a tighter run is recorded beside
        it, the accuracy envelope's reference.
    tighter : Callable or None
        The tighter run of an adaptive case, given the backend.
    resume : bool or str
        How its file is resumed (:meth:`Simulation.run`'s `resume`):
        ``"interpolated"`` where the middle plane lies inside a step.
    """

    name: str
    route: str
    march: str
    build: Callable[[str], Simulation]
    adaptive: bool = False
    tighter: Callable[[str], Simulation] | None = None
    resume: bool | str = True


CASES: tuple[Case, ...] = (
    Case(
        "fibre_rk_replay",
        "gnlse",
        "replayed Runge-Kutta",
        lambda backend: walking_fibre(backend, RungeKutta()),
    ),
    Case(
        "fibre_rk_adaptive",
        "gnlse",
        "adaptive Runge-Kutta",
        lambda backend: walking_fibre(backend, RungeKutta(), scheduled=False),
        adaptive=True,
        tighter=lambda backend: walking_fibre(
            backend, RungeKutta(), scheduled=False, rtol=1e-9
        ),
    ),
    Case(
        "fibre_low_storage",
        "gnlse",
        "low-storage",
        lambda backend: walking_fibre(backend, LowStorageRungeKutta()),
    ),
    Case(
        "capillary_adams_interaction",
        "capillary (three modes)",
        "fused Adams, interaction formulation (retained-history transport)",
        lambda backend: capillary(
            backend,
            Multistep(variable_order=False, formulation="interaction"),
            modes=3,
        ),
    ),
    Case(
        "capillary_adams_exponential",
        "capillary",
        "exponential Adams (untransported history)",
        lambda backend: capillary(backend, FIXED_ADAMS),
    ),
    Case(
        "capillary_adams_carrier",
        "capillary",
        "exponential Adams, carrier frame, dense output at interpolated planes",
        lambda backend: capillary(backend, CARRIER, interpolated=True),
        adaptive=True,
        tighter=lambda backend: capillary(
            backend, CARRIER, interpolated=True, rtol=1e-9
        ),
        resume="interpolated",
    ),
    Case(
        "capillary_envelope_plasma",
        "capillary (envelope plasma)",
        "exponential Adams",
        lambda backend: capillary(backend, FIXED_ADAMS, envelope=True, plasma=True),
    ),
    Case(
        "general_modal_adams",
        "general modal",
        "exponential Adams",
        lambda backend: general_modal(backend, FIXED_ADAMS),
    ),
    Case(
        "radial_rk",
        "free space, radial",
        "replayed Runge-Kutta",
        lambda backend: radial(backend, RungeKutta()),
    ),
    Case(
        "cartesian_adams",
        "free space, Cartesian",
        "exponential Adams",
        lambda backend: cartesian(backend, FIXED_ADAMS),
    ),
    Case(
        "line_edge_adams",
        "free space, line, real-space edge",
        "exponential Adams",
        lambda backend: line_edge(backend, FIXED_ADAMS),
    ),
    Case(
        "line_edge_low_storage",
        "free space, line, real-space edge",
        "low-storage",
        lambda backend: line_edge(backend, LowStorageRungeKutta()),
    ),
    Case(
        "planar_adams",
        "planar (hybrid slab)",
        "exponential Adams",
        lambda backend: planar(backend, FIXED_ADAMS),
    ),
)
"""Every case, keyed by :attr:`Case.name`."""


# ------------------------------------------------------------------- the recording

THREAD_VARIABLES = (
    "SOLPROP_THREADS",
    "SOLPROP_NATIVE_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
)
"""The thread settings of the default lane (``pyproject.toml``'s ``test`` task), each
pinned to one by the generator."""


def identity(backend: str) -> str:
    """Return a digest of what a backend's bits depend on beyond solprop's own code.

    The platform, NumPy's and SciPy's versions, the thread settings and, for native,
    its build: its source, compiler, libraries and thread count. Two runs with the same
    digest are expected to agree bit for bit where the order of operations is
    unchanged (SEP 25 allows no more as a diagnostic). Hashed so that no host path is
    recorded.
    """
    import scipy  # noqa: PLC0415

    found = {
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    } | {name: os.environ.get(name, "") for name in THREAD_VARIABLES}
    if backend == "native":
        from solprop_native import (  # noqa: PLC0415  # pyright: ignore[reportMissingImports]
            build_info,
        )

        info = build_info()
        found |= {
            "source": info.source_digest,
            "compiler": info.compiler,
            "flags": info.compiler_flags,
            "fft": info.fft_library,
            "blas": info.blas_library,
            "vector_math": info.vector_math,
            "threads": str(info.runtime_threads),
        }
    text = json.dumps(found, sort_keys=True).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


class _StopError(RuntimeError):
    """Raised by :class:`_StopAtPlane` to interrupt a run, leaving its file behind."""


class _StopAtPlane(NullReporter):
    """Interrupt a run at its `n`-th saved plane after the launch."""

    def __init__(self, n: int) -> None:
        self.n, self.seen = n, 0

    def plane(self, z: float, summary: object) -> None:
        del z, summary
        self.seen += 1
        if self.seen == self.n:
            raise _StopError


@dataclass
class Recorded:
    """What one case's run on one backend left.

    Attributes
    ----------
    arrays : dict
        Named arrays: ``field`` (the last saved plane, optics convention, as a stack
        of one plane), ``z``,
        ``stats/<name>``, ``step_history/<column>``, ``resumed/field`` and
        ``resumed/z``, ``probe/fields`` (the read-back of each trial, stacked, in
        internal units and the FFT convention), and for an adaptive case
        ``tighter/field``.
    records : dict
        The monitor's records (``monitor``, ``resumed/monitor``), how many executions
        the resumed file holds, the work counters, the probe's trials and refusal.
    """

    arrays: dict[str, np.ndarray]
    records: dict[str, Any]


def _monitor(execution: Mapping[str, object] | None) -> dict[str, Any]:
    """Return an execution record's boundary-monitor record, which every case keeps."""
    assert execution is not None
    found = execution["boundary_monitor"]
    assert isinstance(found, dict)
    return dict(found)


def _run(simulation: Simulation, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run a case into memory; add its arrays and return its records."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        output = simulation.run(progress="none")
    # The last plane, kept as a stack of one so that it is compared as one state.
    arrays["field"] = np.asarray(output["field"][-1:])
    arrays["z"] = np.asarray(output["z"])
    for name in output.stat_names:
        arrays[f"stats/{name}"] = np.asarray(output[f"stats/{name}"])
    for name in (*HISTORY, "counts"):
        arrays[f"step_history/{name}"] = np.asarray(output[f"step_history/{name}"])
    assert isinstance(output, PlaneStore)
    outcome = output.outcome
    assert outcome is not None
    execution = outcome.execution
    assert execution is not None
    return {
        "monitor": _monitor(execution),
        "source_evaluations": execution.get("source_evaluations"),
    }


def _resumed(
    case: Case, simulation: Simulation, path: Path, arrays: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Stop the case at its middle plane, resume its file to the end; return records.

    The stop leaves the file with the middle plane and its checkpoint (the march's
    history among it), which the second run continues from.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SolpropWarning)
        try:
            simulation.run(path, progress=_StopAtPlane(1))
        except _StopError:
            pass
        else:  # pragma: no cover - a case with no middle plane is a corpus defect
            raise AssertionError(f"{case.name} was not interrupted")
        simulation.run(path, resume=case.resume, progress="none")  # pyright: ignore[reportArgumentType]
    with open_file(path) as handle:
        arrays["resumed/field"] = np.asarray(handle["field"][-1:])
        arrays["resumed/z"] = np.asarray(handle["z"])
    outcome = read_resume_state(path).outcome
    assert outcome is not None
    return {
        "monitor": _monitor(outcome.execution),
        "executions": len(outcome.previous_executions) + 1,
    }


def _probed(simulation: Simulation, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Take the timing probe's trials from the launch; return what was asked.

    Two trials, at the default length and at half of it, each read back; the
    low-storage march's probe refuses the read-back, and that refusal is recorded.
    """
    problem = simulation.translate()
    run_request = simulation.run_request()
    backend = select(
        simulation.settings.backend,
        problem,
        run_request,
        grid=simulation.grid,
        modes=simulation.modes,
        profiles=simulation.profiles,
    )
    z = float(problem.initial.z)
    dz = float(run_request.stepper.dz_init)
    trials = [(z, dz), (z, 0.5 * dz)]
    fields = []
    refusal = None
    try:
        probe = backend.timing_probe(Start.entrance(problem))
        for start, length in trials:
            probe.trial(start, length)
            try:
                fields.append(np.asarray(probe.field()))
            except BackendError as error:
                refusal = str(error)
                break
    finally:
        backend.close()
    if fields:
        arrays["probe/fields"] = np.stack(fields)
    return {"trials": [list(one) for one in trials], "refusal": refusal}


def record(
    case: Case,
    backend: str,
    directory: Path,
    *,
    simulation: Simulation | None = None,
    tighter: bool = True,
) -> Recorded:
    """Run one case on one backend and return what it left.

    Parameters
    ----------
    case : Case
        The case.
    backend : str
        The backend's registry name.
    directory : pathlib.Path
        Where the resumed run's file is written.
    simulation : solprop.driver.Simulation or None, optional
        The case's simulation on `backend`, built once and run for each record, its
        translation reused; built here by default.
    tighter : bool, optional
        Whether an adaptive case's tighter run is made too; a comparison reads the
        recorded one.

    Returns
    -------
    Recorded
        Its arrays and records.
    """
    if simulation is None:
        simulation = case.build(backend)
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, Any] = {"run": _run(simulation, arrays)}
    path = directory / f"{case.name}-{backend}.h5"
    records["resumed"] = _resumed(case, simulation, path, arrays)
    records["probe"] = _probed(simulation, arrays)
    if tighter and case.tighter is not None:
        reference: dict[str, np.ndarray] = {}
        _run(case.tighter(backend), reference)
        arrays["tighter/field"] = reference["field"]
    for name in ("field", "resumed/field", "probe/fields", "tighter/field"):
        # A statistic may be undefined (NaN) by design; a field may not.
        if name in arrays:
            assert np.all(np.isfinite(arrays[name])), f"{case.name} {backend}: {name}"
    return Recorded(arrays, records)
