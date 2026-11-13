# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Every periodic-capable statistic, taken by the native session (SEP 47 section 6.4).

The native session evaluates and finishes every statistic natively, from a plan
lowered once (:mod:`solprop_native._statistics`): its z-dependent inputs are the
problem's own tables, or a profile or metric root lowered to a table with a recorded
error. Each family is run with a step plane after every accepted step and a periodic
row at every step, so that every row has an observation at its position, whose field
is exported. Three things are checked on each:

- the plane statistics against the NumPy reference (an independent implementation)
  evaluated on the exported field at the same position, inside the bound the matching
  backend statistics tests use for that statistic;
- the periodic row equal to the plane statistics at the same position, bit for bit:
  one native implementation of each;
- no return to Python caused by the rows, by solprop's native counters: the session
  returns at its observations and its end alone, each one batch.
"""

import dataclasses
import math
import re

import numpy as np
import pytest

from bounds import INTEGRATION_AGREEMENT, gamma
from sessions.executors import (
    PLANES,
    device_build,
    entrance,
    fibre,
    skip_device_refused,
    window,
)
from sessions.runs import load_native
from solprop.backends.events import ObservationReady, Terminal
from solprop.backends.export import ExportRecipe
from solprop.backends.protocol import stepping
from solprop.backends.registry import select
from solprop.exceptions import BackendError, SolpropError, TranslationError
from solprop.stats import (
    CATALOGUE,
    PLASMA_INTEGRATION_STATISTICS,
    StatsSpec,
    _half_max_range,
    _power,
)
from solprop.stepper import FixedSchedule, PIController
from solprop.translate.assembly import assemble

native_session = pytest.importorskip("solprop_native.session")
native_statistics = pytest.importorskip("solprop_native._statistics")
from solprop_native import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    _core,
    _counters,
)

RAW = ExportRecipe(power=False, optics=False)

PLASMA_RTOL = 1e-10
"""The node scans' parity tolerance, a compiled plasma column against the reference's
multipass scan over the window (``test_on_axis_statistics.py``)."""


@dataclasses.dataclass
class Seen:
    """One observation of a native session run.

    Attributes
    ----------
    z : float
        Its position, m.
    stats : dict
        Its plane statistics.
    row : dict or None
        The periodic row taken at the same position, if any.
    field : numpy.ndarray
        The state at it, exported losslessly.
    """

    z: float
    stats: dict[str, np.ndarray]
    row: dict[str, np.ndarray] | None
    field: np.ndarray


def _periodic(names):
    """Return the periodic-capable subset of `names`."""
    return tuple(name for name in names if CATALOGUE[name].between_planes)


def _run(problem, run_request, names, *, grid, modes=None, profiles=None, windows=None):
    """Run the native session with a plane and a row at every accepted step.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    run_request : solprop.translate.run_request.RunRequest
        Its run's request, whose statistics are replaced.
    names : tuple of str
        The plane statistics; their periodic-capable subset is periodic.
    grid : solprop.grid.TimeGrid
        The time grid.
    modes : sequence of Mode or None, optional
        The mode set.
    profiles : mapping or None, optional
        The z profiles.
    windows : numpy.ndarray or None, optional
        The energy windows.

    Returns
    -------
    tuple
        The observations, the native session's status, the counters and the plane
        lowering.
    """
    spec = StatsSpec(
        names=names,
        periodic=_periodic(names),
        period=1,
        energy_windows=np.empty((0, 2)) if windows is None else windows,
    )
    request = run_request.for_run(stats=spec)
    request = dataclasses.replace(
        request, observation=dataclasses.replace(request.observation, step_planes=1)
    )
    skip_device_refused(problem)
    session = native_session.NativeSession(
        problem, request, entrance(problem), grid=grid, modes=modes, profiles=profiles
    )
    seen: list[Seen] = []
    counters = _counters.enable()
    try:
        while not session.finished:
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady):
                    rows = event.step_stats
                    row = None
                    if rows.n_rows and rows.z[-1] == event.z:
                        row = {key: value[-1] for key, value in rows.values.items()}
                    field = np.array(session.export(event.ticket, RAW))
                    seen.append(Seen(event.z, dict(event.stats), row, field))
                elif isinstance(event, Terminal):
                    assert event.error is None, event.error
            session.acknowledge(batch.receipt)
        status = session.native.status()
        lowering = session._plane_lowering  # pyright: ignore[reportPrivateUsage]
    finally:
        _counters.disable()
        session.close()
    return seen, status, counters, lowering, spec


def _reference(problem, run_request, spec, seen, *, grid, modes=None, profiles=None):
    """Return the NumPy reference's plane statistics on each observed field.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The problem.
    run_request : solprop.translate.run_request.RunRequest
        Its run's request.
    spec : solprop.stats.StatsSpec
        The plane statistics.
    seen : list of Seen
        The observations.
    grid, modes, profiles
        As the backend takes them.

    Returns
    -------
    list of dict
        The reference's statistics, without a batch axis, one per observation.
    """
    backend = select(
        "numpy_ref", problem, run_request, grid=grid, modes=modes, profiles=profiles
    )
    start = float(problem.initial.z)
    out = []
    try:
        for record in seen:
            stepping(backend).set_initial(
                record.field, None, z=record.z, resume=record.z > start
            )
            values = stepping(backend).stats(spec)
            out.append({name: np.asarray(value) for name, value in values.items()})
    finally:
        backend.close()
    return out


def _check_rows(seen, names):
    """Assert each row equals its observation's plane statistics bit for bit."""
    periodic = _periodic(names)
    rows = [record for record in seen if record.row is not None]
    assert len(rows) >= len(seen) - 1 > 2
    for record in rows:
        assert record.row is not None
        assert set(record.row) == set(periodic)
        for name in periodic:
            np.testing.assert_array_equal(record.row[name], record.stats[name], name)


def _check_structure(seen, status, counters):
    """Assert the rows were native: returns at the observations and the end alone.

    Every row and plane statistic is the session's own, counted from its batches; the
    one field crossing an observation makes is the export the test asks for.
    """
    rows = sum(record.row is not None for record in seen)
    assert status.rows_taken >= rows > 0
    record = counters.record()
    assert record["statistics_rows"] == status.rows_taken
    assert record["statistics_evaluations"] == status.rows_taken + len(seen)
    assert record["native_returns"] == {
        "session_observation": len(seen),
        "session_terminal": 1,
    }
    assert sum(record["native_returns"].values()) == status.batches
    assert record["field_crossings"] == {"export": len(seen)}
    assert record["host_attempts"] == record["transverse_reads"] == 0


def _within(actual, expected, bound, what):
    """Assert agreement inside a bound fixed before the comparison."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, what
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isfinite(actual), finite, what)
    errors = np.abs(actual[finite] - expected[finite])
    assert np.all(errors <= bound), (what, float(np.max(errors, initial=0)), bound)


def _scalar_bound(problem, reference):
    """The free statistics test's bound of a scalar statistic: FFTs and reductions.

    ``gamma(256 (n_columns + n_to))`` relative to the statistic's largest value.
    """
    relative = gamma(256 * (problem.shape[0] * problem.shape[1] + problem.grids.n_to))
    return relative * float(np.max(np.abs(reference), initial=0.0))


def _width_bound(problem, grid, field, outer, weight=None):
    """Bound a half-maximum width from its trace's perturbation and its slopes.

    The trace moves by ``delta``, the scalar bound at its peak; each crossing
    ``t_lo + (L - P_lo)(t_hi - t_lo)/(P_hi - P_lo)`` then moves by at most
    ``4 delta dt / |P_hi - P_lo|`` (the level, both samples and the span), and the
    width by both crossings' sum.
    """
    whitened = field if weight is None else field * np.sqrt(weight)
    trace = np.sum(_power(grid, whitened), axis=(0, 1))
    delta = _scalar_bound(problem, trace)
    start, end = _half_max_range(grid.t, trace, outer=outer)
    if not math.isfinite(start):
        return 0.0
    dt = float(grid.t[1] - grid.t[0])
    total = 0.0
    for crossing in (start, end):
        low = int(np.searchsorted(grid.t, crossing)) - 1
        span = abs(float(trace[low + 1] - trace[low]))
        total += 4 * delta * dt / span
    return total


# --- the modal temporal family ---------------------------------------------------

TEMPORAL = (
    "omega0",
    "energy",
    "energy_window",
    "peak_power",
    "peak_power_window",
    "peak_power_total",
    "fwhm_t",
    "fwhm_t_outer",
)


def _soliton():
    """The lifecycle suite's soliton, replayed over 24 equal steps."""
    from sessions.executors import BETA2, GAMMA, TAU0, WAVELENGTH  # noqa: PLC0415
    from solprop.pulses import SechPulse  # noqa: PLC0415

    grid = window()
    peak = abs(BETA2) / (GAMMA * TAU0**2)
    return grid, *assemble(
        grid,
        fibre().mode,
        [fibre().kerr],
        SechPulse(wavelength=WAVELENGTH, width=TAU0, peak_power=peak).build(grid),
        save_planes=list(PLANES),
        controller=PIController(rtol=1e-6, atol=1e-16),
        dz_max=PLANES[-1],
        dz_init=1e-3,
        density=1.0,
        shock=False,
        prefactor_index=1.0,
        schedule=FixedSchedule(np.full(24, PLANES[-1] / 24)),
    )


def test_the_temporal_statistics_are_the_reference_s_and_rows_are_planes():
    """Widths finished natively; every temporal statistic, plane and row."""
    grid, problem, run_request = _soliton()
    windows = np.array([[0.99 * grid.omega0, 1.01 * grid.omega0]])
    seen, status, counters, _, spec = _run(
        problem, run_request, TEMPORAL, grid=grid, windows=windows
    )
    _check_rows(seen, TEMPORAL)
    _check_structure(seen, status, counters)
    for record, want in zip(
        seen, _reference(problem, run_request, spec, seen, grid=grid), strict=True
    ):
        for name in TEMPORAL:
            if name.startswith("fwhm"):
                bound = _width_bound(
                    problem, grid, record.field, name.endswith("outer")
                )
            else:
                bound = _scalar_bound(problem, want[name])
            _within(record.stats[name], want[name], bound, f"{name} at {record.z}")


# --- the native widths, against solprop's ------------------------------------------


@pytest.mark.parametrize("outer", [False, True], ids=["central", "outer"])
def test_the_native_width_is_solprop_s_crossing_to_the_bit(outer):
    """The half-maximum crossings ported, operation for operation, on traces with
    satellites, a pedestal, a trace filling the axis, a flat top and nothing."""
    rng = np.random.default_rng(47)
    t = np.linspace(-2e-12, 2e-12, 513)
    traces = [
        np.exp(-((t / 3e-13) ** 2)),
        np.exp(-((t / 3e-13) ** 2)) + 0.7 * np.exp(-(((t - 8e-13) / 1e-13) ** 2)),
        np.exp(-((t / 3e-13) ** 2)) + 0.6,
        np.ones_like(t),
        np.zeros_like(t),
        np.where(np.abs(t) < 5e-13, 1.0, 0.0),
        -np.exp(-((t / 3e-13) ** 2)),
        *(np.abs(rng.normal(size=t.size)) for _ in range(8)),
    ]
    for trace in traces:
        start, end = _half_max_range(t, trace, outer=outer)
        expected = end - start
        got = _core.half_width(t, trace, outer)
        if math.isnan(expected):
            assert math.isnan(got)
        else:
            assert got == expected


# --- a varying power weight, the plasma and the on-axis intensity -----------------

CAPILLARY = (
    "energy",
    "peak_power",
    "peak_power_total",
    "omega0",
    "peak_intensity",
    "electron_density",
    "peak_ionisation_rate",
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
    "plasma_undershoot",
    "plasma_work_defect",
    "plasma_boundary_level",
)
"""Every statistic of a gradient capillary of argon: the weight, the plasma tables, the
taper-free axis synthesis, each read from the problem's own tables."""

PLASMA = frozenset(
    {
        "electron_density",
        "peak_ionisation_rate",
        "species_ionisation",
        "species_electron_density",
        "species_peak_ionisation_rate",
        "species_density",
        "species_ionisation_energy",
        "plasma_undershoot",
        "plasma_work_defect",
        "plasma_boundary_level",
    }
)


def _capillary():
    """An ionising gradient capillary of argon over eight equal steps."""
    from solprop.grid import FieldGrid  # noqa: PLC0415
    from solprop.modes.capillary import MarcatiliMode  # noqa: PLC0415
    from solprop.nonlinear.optmat_rates import OptmatRate  # noqa: PLC0415
    from solprop.nonlinear.responses import (  # noqa: PLC0415
        KerrResponse,
        PlasmaResponse,
    )
    from solprop.physdata.gases import Gas  # noqa: PLC0415
    from solprop.profiles import Constant, PressureGradient  # noqa: PLC0415
    from solprop.pulses import GaussianPulse  # noqa: PLC0415
    from solprop.translate.ionisation import build_rate_table  # noqa: PLC0415

    grid = FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)
    gradient = PressureGradient.two_point(0.2, 2.0, 0.2)
    mode = MarcatiliMode(125e-6, "Ar", gradient, cladding=1.45)
    table = build_rate_table(OptmatRate(Gas.Ar, 800e-9, model="ppt_ilkov"), (1e9, 2e11))
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=100e-6).build(grid)
    problem, run_request = assemble(
        grid,
        mode,
        [
            KerrResponse(chi3=2.2e-52, thg=True),
            PlasmaResponse(Gas.Ar, table, preionisation=1e-3),
        ],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[2e-3, 4e-3],
        schedule=FixedSchedule.uniform(0.0, 4e-3, 5e-4),
        dz_max=0.01,
        dz_init=1e-4,
    )
    profiles = {
        "pressure": gradient,
        "density": mode.density,
        "core_radius": Constant(125e-6),
    }
    return grid, problem, run_request, [mode], profiles


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_the_plasma_and_weight_statistics_are_native():
    """Tables read natively at each row: the weight and the plasma."""
    grid, problem, run_request, modes, profiles = _capillary()
    assert problem.linear.power_weight_constant is None
    seen, status, counters, _, spec = _run(
        problem, run_request, CAPILLARY, grid=grid, modes=modes, profiles=profiles
    )
    _check_rows(seen, CAPILLARY)
    _check_structure(seen, status, counters)
    reference = _reference(
        problem, run_request, spec, seen, grid=grid, modes=modes, profiles=profiles
    )
    for record, want in zip(seen, reference, strict=True):
        for name in CAPILLARY:
            if name in PLASMA_INTEGRATION_STATISTICS:
                # Ratios of rounding-level sums: held absolutely, across backends.
                bound = INTEGRATION_AGREEMENT
            elif name in PLASMA:
                bound = PLASMA_RTOL * float(np.max(np.abs(want[name]), initial=0.0))
            else:
                bound = _scalar_bound(problem, want[name])
            _within(record.stats[name], want[name], bound, f"{name} at {record.z}")
    assert max(float(record.stats["electron_density"]) for record in seen) > 0.0


PROFILES = ("energy", "density", "pressure", "core_radius")


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_profile_table_short_of_the_target_is_the_closest_within_the_tolerance(
    monkeypatch,
):
    """A lowered profile no builder tolerance brings to ``1e-13`` of its scale (the
    platform's libm sets the last bits it is measured against) takes the closest table
    within :data:`PROFILE_TOLERANCE`, its error recorded; beyond it, refused by name."""
    from solprop.profiles import PiecewiseLinear  # noqa: PLC0415

    _, problem, run_request, _, _ = _capillary()
    profile = PiecewiseLinear(
        np.array([0.0, 1.3e-3, 3e-3]), np.array([125e-6, 119e-6, 122e-6])
    )
    z_end = float(run_request.observation.save_planes[-1])
    lowered = native_statistics._lowered_profile_table
    monkeypatch.setattr(native_statistics, "_FIT_TARGET", 0.0)
    _, record = lowered(problem, "core_radius", profile, z_end)
    assert record["source"] == "lowered profile"
    assert 0.0 < record["relative_error"] <= native_statistics.PROFILE_TOLERANCE
    monkeypatch.setattr(native_statistics, "PROFILE_TOLERANCE", 0.0)
    with pytest.raises(BackendError, match="the core_radius statistic: its profile"):
        lowered(problem, "core_radius", profile, z_end)


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize(
    "radius",
    ["constant", "kink-short"],
    ids=["constant radius", "a kinked radius shorter than the run"],
)
def test_the_profile_statistics_read_tables_with_a_recorded_error(radius):
    """SEP 47 section 6.4: a profile statistic reads a z table, the propagation's own
    where one reproduces the profile, else the profile lowered at preparation; its
    measured error is at rounding level and recorded in the execution record, bounds
    the difference from the reference (which reads the profile) and rows equal planes.

    The kinked radius is piecewise linear with a knot inside the run and ends before
    it, holding its last value beyond.
    """
    from solprop.profiles import Constant, PiecewiseLinear  # noqa: PLC0415

    grid, problem, run_request, modes, profiles = _capillary()
    profiles["core_radius"] = (
        Constant(125e-6)
        if radius == "constant"
        else PiecewiseLinear(
            np.array([0.0, 1.3e-3, 3e-3]), np.array([125e-6, 119e-6, 122e-6])
        )
    )
    seen, status, counters, lowering, spec = _run(
        problem, run_request, PROFILES, grid=grid, modes=modes, profiles=profiles
    )
    _check_rows(seen, PROFILES)
    _check_structure(seen, status, counters)
    tables = lowering.tables
    assert set(tables) == {"density", "pressure", "core_radius"}
    for name, record in tables.items():
        assert record["relative_error"] <= native_statistics.PROFILE_TOLERANCE, name
        assert record["check_points"] > 4000, name
    reference = _reference(
        problem, run_request, spec, seen, grid=grid, modes=modes, profiles=profiles
    )
    for record, want in zip(seen, reference, strict=True):
        for name in ("density", "pressure", "core_radius"):
            bound = tables[name]["max_error"]
            _within(record.stats[name], want[name], bound, f"{name} at {record.z}")
        _within(
            record.stats["energy"],
            want["energy"],
            _scalar_bound(problem, want["energy"]),
            f"energy at {record.z}",
        )
    if radius != "constant":
        assert tables["core_radius"]["source"] == "lowered profile"
        past = [record for record in seen if record.z > 3e-3]
        assert past
        for record in past:
            assert (
                abs(float(record.stats["core_radius"]) - 122e-6)
                <= tables["core_radius"]["max_error"]
            )
    # The record reaches the session's execution record.
    skip_device_refused(problem)
    session = native_session.NativeSession(
        problem,
        run_request.for_run(stats=spec),
        entrance(problem),
        grid=grid,
        modes=modes,
        profiles=profiles,
    )
    try:
        assert session.execution_record()["statistics_tables"] == tables
    finally:
        session.close()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_native_plan_refusal_is_the_reference_s_exception():
    """A position outside the plasma tables: native refuses as the reference's table
    does, with the solprop exception the reference raises. A state cannot be loaded
    there, so the native state evaluates the lowered plan there directly, and its
    refusal is translated as a session's is."""
    grid, problem, run_request, modes, _ = _capillary()
    far = 100.0 * float(run_request.stepper.z_end)
    channel = problem.responses.population_channels[0]
    with pytest.raises(SolpropError) as reference:
        channel.density(far)
    spec = StatsSpec(names=("electron_density",))
    lowering = native_statistics.lower(
        problem,
        spec,
        spec.names,
        grid=grid,
        modes=modes,
        z_end=float(run_request.stepper.z_end),
    )
    assert lowering.plan is not None
    backend = select("native", problem, run_request, grid=grid, modes=modes)
    try:
        state = load_native(backend, problem.initial.field0, clear_fsal=False)
        with pytest.raises(SolpropError) as caught:
            try:
                state.planned_statistics(lowering.plan, far)
            except (ValueError, IndexError, RuntimeError) as error:
                raise native_statistics.translated(error) from error
    finally:
        backend.close()
    assert type(caught.value) is type(reference.value) is TranslationError
    assert "outside a statistics table's span" in str(caught.value)


# --- two populations: the species reduction ----------------------------------------

SPECIES = (
    "energy",
    "electron_density",
    "peak_ionisation_rate",
    "species_ionisation",
    "species_electron_density",
    "species_peak_ionisation_rate",
    "species_density",
    "species_ionisation_energy",
)


def test_two_populations_are_reduced_natively():
    """The species reduction over two channels, finished natively."""
    from mixture_cases import mixture_translation  # noqa: PLC0415

    _, grid, mode, _, (problem, run_request) = mixture_translation(
        schedule=FixedSchedule.uniform(0.0, 1e-3, 1.25e-4)
    )
    assert len(problem.responses.population_channels) == 2
    seen, status, counters, _, spec = _run(
        problem, run_request, SPECIES, grid=grid, modes=[mode]
    )
    _check_rows(seen, SPECIES)
    _check_structure(seen, status, counters)
    reference = _reference(problem, run_request, spec, seen, grid=grid, modes=[mode])
    for record, want in zip(seen, reference, strict=True):
        for name in SPECIES:
            if name == "energy":
                bound = _scalar_bound(problem, want[name])
            else:
                bound = PLASMA_RTOL * float(np.max(np.abs(want[name]), initial=0.0))
            _within(record.stats[name], want[name], bound, f"{name} at {record.z}")
    assert max(float(record.stats["electron_density"]) for record in seen) > 0.0


# --- a general modal basis: functionals of a.H W a ----------------------------------

GENERAL = (
    "omega0",
    "physical_energy",
    "spectral_flux",
    "analytic_flux",
    "fwhm_t",
    "fwhm_t_outer",
)


def test_a_general_modal_basis_s_flux_statistics_are_native():
    """The metric's root, lowered once, whitens the columns natively."""
    from modal_provider_cases import GENERAL_Z_END, general_case  # noqa: PLC0415
    from solprop.pulses import GaussianPulse  # noqa: PLC0415
    from solprop.stats import _whitened  # noqa: PLC0415
    from solprop.translate.coupled import metric_at  # noqa: PLC0415

    case = general_case(
        (("HE", 1, 1, "even"), ("HE", 1, 1, "odd")), blocks=((0, 1),), circular=(0, 1)
    )
    grid = case.grid
    pulse = GaussianPulse(
        wavelength=2 * np.pi * 299792458.0 / grid.omega0, fwhm=30e-15, energy=1e-12
    ).build(grid)
    state = np.concatenate([pulse, 0.3j * pulse], axis=0)
    problem, run_request = assemble(
        grid,
        list(case.guide.mode_collection),
        [],
        state,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[GENERAL_Z_END],
        dz_max=GENERAL_Z_END,
        dz_init=GENERAL_Z_END / 8,
        schedule=FixedSchedule.uniform(0.0, GENERAL_Z_END, GENERAL_Z_END / 8),
        general=case.route,
    )
    metric = problem.linear.modal_metric
    assert metric is not None and metric.kind == "dense"
    seen, status, counters, lowering, spec = _run(
        problem, run_request, GENERAL, grid=grid
    )
    # The root is formed exactly, interval by interval, from the metric's own values.
    assert lowering.values["root"]["n_terms"] == 1
    _check_rows(seen, GENERAL)
    _check_structure(seen, status, counters)
    for record, want in zip(
        seen, _reference(problem, run_request, spec, seen, grid=grid), strict=True
    ):
        whitened = _whitened(record.field, metric_at(metric, record.z))
        for name in GENERAL:
            if name.startswith("fwhm"):
                bound = _width_bound(problem, grid, whitened, name.endswith("outer"))
            else:
                bound = _scalar_bound(problem, want[name])
            _within(record.stats[name], want[name], bound, f"{name} at {record.z}")


# --- a spatial beam: the axial flux and the far field -------------------------------

SPATIAL = (
    "energy",
    "peak_intensity",
    "on_axis_intensity",
    "absorbed_power",
    "on_axis_spectrum",
)


def test_a_beam_s_flux_and_far_field_statistics_are_native():
    """A Cartesian beam: the flux scalars and the far-field pass, plane and row."""
    from backends.test_radial_nonlinear import make_problem  # noqa: PLC0415
    from solprop.grid import EnvelopeGrid, FreeGrid  # noqa: PLC0415
    from solprop.pulses import GaussianPulse, SpatialPulse  # noqa: PLC0415
    from solprop.spatial import SpatialMedium  # noqa: PLC0415

    grid = EnvelopeGrid.from_limits(1e-4, 800e-9, (600e-9, 1.2e-6), 150e-15, thg=False)
    space = FreeGrid(4e-4, 32, 5e-4, 40, collar_fraction=0)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 100e-6
    )
    start = pulse.build(grid, space, medium=SpatialMedium()) * grid.active_mask
    problem, run_request = make_problem(grid, space, start, [])
    seen, status, counters, _, spec = _run(problem, run_request, SPATIAL, grid=grid)
    _check_rows(seen, SPATIAL)
    _check_structure(seen, status, counters)
    for record, want in zip(
        seen, _reference(problem, run_request, spec, seen, grid=grid), strict=True
    ):
        for name in SPATIAL:
            relative = gamma(256 * (space.n_columns + grid.n_to))
            bound = relative * float(np.max(np.abs(want[name]), initial=0.0))
            _within(record.stats[name], want[name], bound, f"{name} at {record.z}")


# --- the structural counter: rows return nothing to Python -------------------------


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_rows_of_every_family_cause_no_return_to_python():
    """SEP 47 section 7.3: a period-one run of the richest family returns to Python
    only at its observations and its end, whatever statistics its rows take: each row
    is taken and finished natively and arrives in bulk with its plane."""
    grid, problem, run_request, modes, profiles = _capillary()
    spec = StatsSpec(names=CAPILLARY, periodic=_periodic(CAPILLARY), period=1)
    request = run_request.for_run(stats=spec)
    skip_device_refused(problem)
    session = native_session.NativeSession(
        problem, request, entrance(problem), grid=grid, modes=modes, profiles=profiles
    )
    counters = _counters.enable()
    batches = []
    try:
        while not session.finished:
            batch = session.advance()
            batches.append(batch)
            session.acknowledge(batch.receipt)
        status = session.native.status()
    finally:
        _counters.disable()
        session.close()
    observations = [
        event
        for batch in batches
        for event in batch.events
        if isinstance(event, ObservationReady)
    ]
    # The launch and the two planes, then the end: nothing in between.
    assert len(observations) == 1 + len(run_request.stepper.save_planes)
    assert len(batches) == len(observations) + 1 == status.batches
    rows = sum(event.step_stats.n_rows for event in observations)
    accepted = sum(int(event.history.accepted.sum()) for event in observations)
    assert rows == accepted == status.rows_taken == 8
    assert set(observations[-1].step_stats.values) == set(_periodic(CAPILLARY))
    record = counters.record()
    assert record["statistics_rows"] == status.rows_taken
    assert record["statistics_evaluations"] == status.rows_taken + len(observations)
    assert record["native_returns"] == {
        "session_observation": len(observations),
        "session_terminal": 1,
    }
    assert record["field_crossings"] == {}


# --- a beam's fluence widths: plane-only, finished natively -----------------------

WIDTHS = ("energy", "on_axis_intensity", "beam_radius", "fwhm_r")
"""A spatial route's plane-only fluence widths beside two periodic statistics, which
the rows take."""


def _beam(route):
    """A beam on one width rule's geometry, over a few replayed steps."""
    from backends.planar_cases import planar_case  # noqa: PLC0415
    from solprop.driver import Simulation, SimulationSettings  # noqa: PLC0415
    from solprop.grid import (  # noqa: PLC0415
        EnvelopeGrid,
        FreeGrid,
        LineGrid,
        RadialGrid,
    )
    from solprop.pulses import GaussianPulse, SpatialPulse  # noqa: PLC0415
    from solprop.spatial import SpatialMedium  # noqa: PLC0415

    if route == "planar":
        grid, _, problem, run_request = planar_case("kerr", steps=4)
        return grid, problem, run_request
    grid = EnvelopeGrid.from_limits(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    space = {
        "radial": RadialGrid(3e-4, 32),
        "plane": FreeGrid(3e-4, 32, 4e-4, 24),
        "line": LineGrid(4e-4, 48),
    }[route]
    settings = SimulationSettings(
        controller=PIController(rtol=1e-6, atol=1e-14),
        save_planes=[1e-3],
        dz_max=2.5e-4,
        dz_init=2.5e-4,
        absorber=None,
        schedule=FixedSchedule(np.full(4, 2.5e-4)),
        stats=StatsSpec(names=WIDTHS),
    )
    # Off centre, so that a centroid and a two-sided average have work to do.
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 6e-5
    )
    simulation = Simulation.from_free(
        grid,
        space,
        SpatialMedium(1.45 + 2e-3j),
        [],
        pulse,
        settings=settings,
        noise=None,
    )
    return grid, simulation.translate(), simulation._run_request()


def _host_widths(problem, run_request, grid, record, monkeypatch):
    """The host finishing of the widths, on the native fluence passes.

    ``solprop.backends.spatial.signed_fluence_widths`` on the fluence block, or
    ``plane_fluence_widths`` on a plane's moments and its native trace, of the record's
    exported field loaded at its position.
    """
    from backends.test_native_widths import host_interp  # noqa: PLC0415
    from solprop.backends.spatial import (  # noqa: PLC0415
        fluence_budget,
        plane_fluence_widths,
        signed_fluence_widths,
    )
    from solprop.constants import EPS0, C  # noqa: PLC0415

    host_interp(monkeypatch)
    geometry = problem.transform.geometry
    assert geometry is not None
    plane = geometry.kind == "free" and geometry.axes[1].size > 1
    n_terms = problem.shape[1] * problem.grids.n_t + problem.transform.n_modes
    assert problem.initial.noise0 is None
    backend = select("native", problem, run_request, grid=grid)
    try:
        # The record's exported state, loaded into the native state itself.
        state = load_native(backend, record.field, clear_fsal=False)
        packed = np.ravel(
            state.spatial_statistics(32 if plane else 16, 0.5 * EPS0 * C, record.z)
        )
        if plane:
            return plane_fluence_widths(
                geometry, packed, n_terms, state.plane_fluence_trace
            )
        count = problem.transform.n_nodes
        budget = fluence_budget(np.full(count, packed[count + 1]), n_terms)
        return signed_fluence_widths(
            geometry, packed[:count], budget, float(packed[count])
        )
    finally:
        backend.close()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.filterwarnings(
    "ignore:the temporal absorber attenuates a packet walking off"
    ":solprop.exceptions.SolpropWarning"
)
@pytest.mark.parametrize("route", ["radial", "plane", "line", "planar"])
def test_a_beam_s_fluence_widths_are_native_at_its_planes(route, monkeypatch):
    """``beam_radius`` and ``fwhm_r``, plane-only, in the session's own plan.

    Against the host finishing of the widths on the same native passes: the same bits on
    a radial, line or slab profile, whose sums are NumPy's pairwise order
    (``numpy.interp`` held to its unfused rule off x86-64, where NumPy's may fuse);
    within a few units of rounding on a plane, whose extent is the platform's ``hypot``
    rather than Python's. Against the NumPy reference on each exported field within the
    backend widths test's bound. The rows, which take the periodic statistics beside
    them, return nothing to Python.
    """
    from backends.test_native_widths import _agree, _width_budget  # noqa: PLC0415

    grid, problem, run_request = _beam(route)
    request = run_request.for_run(stats=StatsSpec(names=WIDTHS))
    assert native_session.refusal(problem, request, entrance(problem)) is None
    seen, status, counters, _, spec = _run(problem, run_request, WIDTHS, grid=grid)
    _check_rows(seen, WIDTHS)
    _check_structure(seen, status, counters)
    reference = _reference(problem, run_request, spec, seen, grid=grid)
    geometry = problem.transform.geometry
    assert geometry is not None
    pixel = float(np.min(np.diff(geometry.axes[0])))
    extent = max(float(np.max(np.abs(axis))) for axis in geometry.axes)
    for record, want in zip(seen, reference, strict=True):
        got = (*record.stats["beam_radius"], float(record.stats["fwhm_r"]))
        host = _host_widths(problem, run_request, grid, record, monkeypatch)
        assert np.isfinite(got[0]), record.z
        if route == "plane":
            np.testing.assert_allclose(got, host, rtol=16 * np.finfo(float).eps)
        else:
            np.testing.assert_array_equal(got, host, err_msg=f"{route} at {record.z}")
        _agree(
            record.stats,
            want,
            _width_budget(grid, problem),
            pixel,
            f"{route} at {record.z}",
            extent,
        )


def test_a_negative_lobe_s_widths_are_refused_by_the_session_as_by_the_reference():
    """A fluence negative beyond its rounding has no width: the session's plane fails
    with the reference's exception and message (``test_native_widths``' lobe)."""
    from backends.test_native_widths import _simulation  # noqa: PLC0415
    from solprop.backends.session import Start  # noqa: PLC0415
    from solprop.grid import FreeGrid  # noqa: PLC0415
    from solprop.spatial import SpatialMedium  # noqa: PLC0415

    grid, problem, run_request = _simulation(
        "envelope", FreeGrid(6e-6, 24, 6e-6, 24), 1, SpatialMedium(1.45), waist=2e-6
    )
    columns = np.zeros(problem.shape[0], complex)
    columns[0] = 0.9
    columns[12] = 1.0
    spectrum = np.zeros((1, problem.shape[2]), complex)
    spectrum[0, 0] = 1.0
    field = np.einsum("r,pw->rpw", columns, spectrum) * 1e6
    request = run_request.for_run(stats=StatsSpec(names=("beam_radius",)))
    with pytest.raises(BackendError, match="nonnegative fluence profile") as want:
        reference = select("numpy_ref", problem, run_request, grid=grid)
        try:
            stepping(reference).set_initial(field, None)
            stepping(reference).stats(StatsSpec(names=("beam_radius",)))
        finally:
            reference.close()
    session = native_session.NativeSession(
        problem, request, Start(field=field, z=float(problem.initial.z)), grid=grid
    )
    errors = []
    try:
        while not session.finished:
            batch = session.advance()
            errors += [
                event.error
                for event in batch.events
                if isinstance(event, Terminal) and event.error is not None
            ]
            session.acknowledge(batch.receipt)
    finally:
        session.close()
    assert len(errors) == 1
    assert type(errors[0]) is BackendError
    got, expected = str(errors[0]), str(want.value)
    if device_build():
        # Two of the lobe's nodes hold the same flux to the message's four digits
        # (measured on an NVIDIA A40): which one a device's rounding takes beyond the
        # budget first is the device's transform's, not the session's.
        got, expected = (
            re.sub(r"at node \d+", "at node N", one) for one in (got, expected)
        )
    assert got == expected


# --- a metric that varies inside an interval: refused by name on native ------------


def _varying_metric():
    """The general basis of the flux test, its metric ``W0 (1 + T_1)`` on every
    interval: no longer constant inside one, positive away from each interval's left
    edge, and each coefficient as near the identity as ``W0``, which is what
    ``numpy_ref``'s admission measures."""
    from modal_provider_cases import GENERAL_Z_END, general_case  # noqa: PLC0415
    from solprop.pulses import GaussianPulse  # noqa: PLC0415

    case = general_case(
        (("HE", 1, 1, "even"), ("HE", 1, 1, "odd")), blocks=((0, 1),), circular=(0, 1)
    )
    grid = case.grid
    pulse = GaussianPulse(
        wavelength=2 * np.pi * 299792458.0 / grid.omega0, fwhm=30e-15, energy=1e-12
    ).build(grid)
    problem, run_request = assemble(
        grid,
        list(case.guide.mode_collection),
        [],
        np.concatenate([pulse, 0.3j * pulse], axis=0),
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[GENERAL_Z_END],
        dz_max=GENERAL_Z_END,
        dz_init=GENERAL_Z_END / 8,
        schedule=FixedSchedule.uniform(0.0, GENERAL_Z_END, GENERAL_Z_END / 8),
        general=case.route,
    )
    metric = problem.linear.modal_metric
    assert metric is not None and metric.dense is not None
    dense = np.concatenate([metric.dense, metric.dense], axis=1)
    varying = dataclasses.replace(metric, dense=dense)
    problem = dataclasses.replace(
        problem, linear=dataclasses.replace(problem.linear, modal_metric=varying)
    )
    assert native_statistics._metric_varies(problem)  # pyright: ignore[reportPrivateUsage]
    return grid, problem, run_request


def test_a_varying_metric_s_statistics_are_refused_by_native_before_allocation():
    """Native refuses a varying metric's statistics by name, at construction and at a
    run's preparation, and never leaves them to another session; numpy_ref
    computes them, and native still serves what the metric does not weigh."""
    grid, problem, run_request = _varying_metric()
    spec = StatsSpec(names=GENERAL)
    request = run_request.for_run(stats=spec)
    # A schedule edge inside the first interval, away from its left edge, where
    # W0 (1 + T_1) is zero.
    inside = float(run_request.stepper.save_planes[-1]) / 8
    reference = select("numpy_ref", problem, request, grid=grid)
    try:
        stepping(reference).set_initial(
            problem.initial.field0, None, z=inside, resume=True
        )
        values = stepping(reference).stats(spec)
    finally:
        reference.close()
    assert set(values) == set(GENERAL)
    assert np.isfinite(float(np.ravel(values["physical_energy"])[0]))
    refused = r"native refuses the metric statistics \('omega0', 'physical_energy'"
    with pytest.raises(BackendError, match=refused):
        select("native", problem, request, grid=grid)
    assert native_session.refusal(problem, request, entrance(problem)) is not None
    # Built for no statistics, asked for them by a run's request.
    backend = select("native", problem, run_request, grid=grid)
    try:
        with pytest.raises(BackendError, match=refused):
            backend.prepare(problem, request)
        # The channels' zero-dispersion wavelength weighs no metric.
        assert native_statistics.metric_refusal(problem, ("zdw",)) is None
    finally:
        backend.close()


# --- zdw: each mode's root along z, read from a table lowered once -----------------


def _zdw_capillary(pressure):
    """A Kerr capillary of helium whose band holds its zero-dispersion wavelength."""
    from solprop.grid import FieldGrid  # noqa: PLC0415
    from solprop.modes.capillary import MarcatiliMode  # noqa: PLC0415
    from solprop.nonlinear.responses import KerrResponse  # noqa: PLC0415
    from solprop.pulses import GaussianPulse  # noqa: PLC0415

    grid = FieldGrid.from_limits(0.2, 800e-9, (200e-9, 4e-6), 100e-15)
    mode = MarcatiliMode(125e-6, "He", pressure, cladding=1.45)
    start = GaussianPulse(wavelength=800e-9, fwhm=10e-15, energy=10e-6).build(grid)
    problem, run_request = assemble(
        grid,
        mode,
        [KerrResponse(chi3=2.2e-53, thg=True)],
        start,
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[2e-3, 4e-3],
        schedule=FixedSchedule.uniform(0.0, 4e-3, 1e-3),
        dz_max=0.01,
        dz_init=1e-4,
    )
    return grid, problem, run_request, [mode]


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_z_independent_mode_s_zdw_is_the_reference_s_to_the_bit():
    """A z-independent mode's table holds its one root: the reference's at every plane,
    bit for bit, with no root search in the run."""
    grid, problem, run_request, modes = _zdw_capillary(0.4)
    assert modes[0].z_constant
    names = ("energy", "zdw")
    seen, status, counters, lowering, spec = _run(
        problem, run_request, names, grid=grid, modes=modes
    )
    _check_rows(seen, names)
    _check_structure(seen, status, counters)
    record = lowering.tables["zdw"]
    assert record["source"] == "z-independent modes"
    assert record["max_error"] == 0.0
    reference = _reference(problem, run_request, spec, seen, grid=grid, modes=modes)
    for record_, want in zip(seen, reference, strict=True):
        assert np.isfinite(want["zdw"]).all()
        np.testing.assert_array_equal(record_.stats["zdw"], want["zdw"])


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_pressure_gradient_s_zdw_is_a_table_within_its_recorded_tolerance():
    """A mode that varies along z: its root lowered over the run, its measured error
    recorded and within ``ZDW_TOLERANCE``, and every plane within that of the
    reference's own root search."""
    from solprop.profiles import PressureGradient  # noqa: PLC0415

    grid, problem, run_request, modes = _zdw_capillary(
        PressureGradient.two_point(0.2, 0.3, 0.6)
    )
    assert not modes[0].z_constant
    names = ("energy", "zdw")
    seen, status, counters, lowering, spec = _run(
        problem, run_request, names, grid=grid, modes=modes
    )
    _check_rows(seen, names)
    _check_structure(seen, status, counters)
    record = lowering.tables["zdw"]
    tolerance = native_statistics.ZDW_TOLERANCE
    assert record["source"] == "lowered zdw"
    assert record["tolerance"] == tolerance
    assert 0.0 <= record["relative_error"] <= tolerance
    reference = _reference(problem, run_request, spec, seen, grid=grid, modes=modes)
    for record_, want in zip(seen, reference, strict=True):
        scale = float(np.max(np.abs(want["zdw"])))
        _within(
            record_.stats["zdw"], want["zdw"], tolerance * scale, f"zdw {record_.z}"
        )


@pytest.mark.medium
@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_few_planes_zdw_is_searched_there_and_is_the_reference_s_to_the_bit(
    monkeypatch,
):
    """A z-varying mode with only save planes: its root searched once at each of the
    run's positions before it starts, planes times modes searches and no fit, and the
    reference's at every plane, bit for bit."""
    import solprop.modes.base as modes_base  # noqa: PLC0415
    from solprop.profiles import PressureGradient  # noqa: PLC0415

    grid, problem, run_request, modes = _zdw_capillary(
        PressureGradient.two_point(0.2, 0.3, 0.6)
    )
    spec = StatsSpec(names=("energy", "zdw"))
    request = run_request.for_run(stats=spec)
    assert not request.observation.step_planes and not request.observation.dense
    searches = []
    search = modes_base.zdw

    def counted(*arguments, **keywords):
        searches.append(arguments[1])
        return search(*arguments, **keywords)

    monkeypatch.setattr(modes_base, "zdw", counted)
    session = native_session.NativeSession(
        problem, request, entrance(problem), grid=grid, modes=modes
    )
    seen: list[Seen] = []
    try:
        record = session._plane_lowering.tables["zdw"]  # pyright: ignore[reportPrivateUsage]
        while not session.finished:
            batch = session.advance()
            for event in batch.events:
                if isinstance(event, ObservationReady):
                    field = np.array(session.export(event.ticket, RAW))
                    seen.append(Seen(event.z, dict(event.stats), None, field))
                elif isinstance(event, Terminal):
                    assert event.error is None, event.error
            session.acknowledge(batch.receipt)
    finally:
        session.close()
    monkeypatch.undo()
    planes = 1 + len(run_request.stepper.save_planes)
    assert record["source"] == "save planes"
    assert record["max_error"] == 0.0
    assert len(seen) == planes
    assert sorted(searches) == sorted(record_.z for record_ in seen) * len(modes)
    reference = _reference(problem, run_request, spec, seen, grid=grid, modes=modes)
    for record_, want in zip(seen, reference, strict=True):
        assert np.isfinite(want["zdw"]).all()
        np.testing.assert_array_equal(record_.stats["zdw"], want["zdw"])
