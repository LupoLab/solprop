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
- no return to Python caused by the rows, by solprop's native counters.
"""

import dataclasses
import math

import numpy as np
import pytest

from bounds import gamma
from optmat_parity import INTEGRATION_AGREEMENT
from sessions.executors import PLANES, entrance, fibre, window
from solprop.backends.events import ObservationReady, Terminal
from solprop.backends.export import ExportRecipe
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
from solprop.translate.problem import assemble
from solprop.translate.run_request import describe_run

native_session = pytest.importorskip("solprop_native.session")
native_statistics = pytest.importorskip("solprop_native._statistics")
from solprop_native import _core, _counters  # noqa: E402

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


def _run(problem, names, *, grid, modes=None, profiles=None, windows=None):
    """Run the native session with a plane and a row at every accepted step.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
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
    request = describe_run(problem, stats=spec)
    request = dataclasses.replace(
        request, observation=dataclasses.replace(request.observation, step_planes=1)
    )
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


def _reference(problem, spec, seen, *, grid, modes=None, profiles=None):
    """Return the NumPy reference's plane statistics on each observed field.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.
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
    backend = select("numpy_ref", problem, grid=grid, modes=modes, profiles=profiles)
    start = float(problem.initial.z)
    out = []
    try:
        for record in seen:
            backend.set_initial(
                record.field[np.newaxis], None, z=record.z, resume=record.z > start
            )
            values = backend.stats(spec)
            out.append({name: np.asarray(value)[0] for name, value in values.items()})
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
    """Assert the rows were native: no return to Python, no statistics call."""
    rows = sum(record.row is not None for record in seen)
    assert status.rows_taken >= rows > 0
    record = counters.record()
    assert record["statistics_evaluations"] == 0
    assert record["statistics_rows"] == 0
    assert record["native_returns"] == {}
    assert record["field_crossings"] == {}


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
    relative = gamma(256 * (problem.shape[1] * problem.shape[2] + problem.grids.n_to))
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
    return grid, assemble(
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
    grid, problem = _soliton()
    windows = np.array([[0.99 * grid.omega0, 1.01 * grid.omega0]])
    seen, status, counters, _, spec = _run(
        problem, TEMPORAL, grid=grid, windows=windows
    )
    _check_rows(seen, TEMPORAL)
    _check_structure(seen, status, counters)
    for record, want in zip(
        seen, _reference(problem, spec, seen, grid=grid), strict=True
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
    problem = assemble(
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
    return grid, problem, [mode], profiles


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_the_plasma_and_weight_statistics_are_native():
    """Tables read natively at each row: the weight and the plasma."""
    grid, problem, modes, profiles = _capillary()
    assert problem.linear.power_weight_constant is None
    seen, status, counters, _, spec = _run(
        problem, CAPILLARY, grid=grid, modes=modes, profiles=profiles
    )
    _check_rows(seen, CAPILLARY)
    _check_structure(seen, status, counters)
    reference = _reference(
        problem, spec, seen, grid=grid, modes=modes, profiles=profiles
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


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
@pytest.mark.parametrize("name", ["density", "pressure", "core_radius"])
def test_a_profile_statistic_is_refused_by_name_and_exact_on_a_plane(name):
    """A profile statistic reads a Python profile the propagation does not tabulate:
    the session refuses it by name rather than approximate it, and the backend's
    plane statistic is the profile's own value, past the profile's span included (a
    piecewise-linear core radius shorter than the run, which clamps)."""
    from solprop.profiles import PiecewiseLinear  # noqa: PLC0415
    from solprop.stats import profile_value  # noqa: PLC0415

    grid, problem, modes, profiles = _capillary()
    profiles["core_radius"] = PiecewiseLinear(
        np.array([0.0, 1e-3]), np.array([125e-6, 120e-6])
    )
    spec = StatsSpec(names=("energy", name), periodic=("energy",), period=1)
    request = describe_run(problem, stats=spec)
    reason = native_session.refusal(problem, request)
    assert reason is not None and repr(name) in reason
    with pytest.raises(BackendError, match=name):
        native_session.NativeSession(
            problem,
            request,
            entrance(problem),
            grid=grid,
            modes=modes,
            profiles=profiles,
        )
    backend = select("native", problem, grid=grid, modes=modes, profiles=profiles)
    try:
        field = problem.initial.field0
        for z in (0.0, 2e-3, 4e-3):
            backend.set_initial(field, None, z=z, resume=z > 0.0)
            value = backend.stats(StatsSpec(names=(name,)))[name]
            assert value[0] == profile_value(profiles[name], z), z
    finally:
        backend.close()


@pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning")
def test_a_native_plan_refusal_is_the_reference_s_exception():
    """A position outside the plasma tables: native refuses as the reference's table
    does, with the solprop exception the reference raises. A state cannot be loaded
    there, so the backend's position is moved past the tables directly."""
    grid, problem, modes, _ = _capillary()
    far = 100.0 * float(problem.stepper.z_end)
    channel = problem.responses.population_channels[0]
    with pytest.raises(SolpropError) as reference:
        channel.density(far)
    backend = select("native", problem, grid=grid, modes=modes)
    try:
        backend.set_initial(problem.initial.field0, None)
        backend._z = far  # pyright: ignore[reportAttributeAccessIssue]
        with pytest.raises(SolpropError) as caught:
            backend.stats(StatsSpec(names=("electron_density",)))
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
    from mixture_cases import mixture_case  # noqa: PLC0415

    _, grid, mode, _, problem = mixture_case(
        schedule=FixedSchedule.uniform(0.0, 1e-3, 1.25e-4)
    )
    assert len(problem.responses.population_channels) == 2
    seen, status, counters, _, spec = _run(problem, SPECIES, grid=grid, modes=[mode])
    _check_rows(seen, SPECIES)
    _check_structure(seen, status, counters)
    reference = _reference(problem, spec, seen, grid=grid, modes=[mode])
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
    problem = assemble(
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
    seen, status, counters, lowering, spec = _run(problem, GENERAL, grid=grid)
    # The root is formed exactly, interval by interval, from the metric's own values.
    assert lowering.values["root"]["n_terms"] == 1
    _check_rows(seen, GENERAL)
    _check_structure(seen, status, counters)
    for record, want in zip(
        seen, _reference(problem, spec, seen, grid=grid), strict=True
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
    problem = make_problem(grid, space, start, [])
    seen, status, counters, _, spec = _run(problem, SPATIAL, grid=grid)
    _check_rows(seen, SPATIAL)
    _check_structure(seen, status, counters)
    for record, want in zip(
        seen, _reference(problem, spec, seen, grid=grid), strict=True
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
    grid, problem, modes, profiles = _capillary()
    spec = StatsSpec(names=CAPILLARY, periodic=_periodic(CAPILLARY), period=1)
    request = describe_run(problem, stats=spec)
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
    assert len(observations) == 1 + len(problem.stepper.save_planes)
    assert len(batches) == len(observations) + 1 == status.batches
    rows = sum(event.step_stats.n_rows for event in observations)
    accepted = sum(int(event.history.accepted.sum()) for event in observations)
    assert rows == accepted == status.rows_taken == 8
    assert set(observations[-1].step_stats.values) == set(_periodic(CAPILLARY))
    record = counters.record()
    assert record["statistics_evaluations"] == record["statistics_rows"] == 0
    assert record["native_returns"] == {}
    assert record["field_crossings"] == {}
