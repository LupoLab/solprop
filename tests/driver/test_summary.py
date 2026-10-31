# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The summary a run shows before its first step, per route, and where it goes."""

import dataclasses
import io
import logging
from functools import partial
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from solprop.driver import Simulation, SimulationSettings
from solprop.driver.run import run_problem
from solprop.driver.simulation import metadata_for
from solprop.driver.summary import (
    RUN_SUMMARY_ANNOTATION,
    RunSummary,
    _code,
    _compiled,
    _machine,
    _medium,
    _pressure,
    _rate,
    _representation,
    _response,
    _responses,
    _threads,
    memory_size,
    resources,
    run_summary,
)
from solprop.grid import EnvelopeGrid, FieldGrid, FreeGrid, LineGrid, RadialGrid
from solprop.log import (
    BarReporter,
    LogReporter,
    NullReporter,
    SummarisingReporter,
)
from solprop.modes import PlanarGrid, SlabMode
from solprop.nonlinear.envelope_plasma import EnvelopePlasmaResponse
from solprop.nonlinear.optmat_rates import OptmatRate
from solprop.nonlinear.raman import SilicaTensorResponse
from solprop.nonlinear.responses import KerrResponse, PlasmaResponse
from solprop.output import HDF5Output, MemoryOutput, open_file
from solprop.output.schema import GitProvenance
from solprop.physdata import Mixture
from solprop.profiles import Linear
from solprop.pulses import GaussianPulse, PlanarPulse, SpatialPulse
from solprop.spatial import SpatialMedium
from solprop.stepper import PIController, RungeKutta
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.stepper.schedule import FixedSchedule
from solprop.translate.precision import Precision

pytestmark = [
    pytest.mark.filterwarnings("ignore::solprop.exceptions.MaterialRangeWarning"),
    # The fixtures here launch noise-free into time windows kept tight for speed, which
    # hold more than a millionth of the energy in their guards or collar: the boundary
    # monitor warns (SEP 21), and the window's edge is not what this module tests.
    pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning"),
]

GNLSE = {
    "wavelength": 835e-9,
    "wavelength_limits": (500e-9, 2e-6),
    "trange": 1e-12,
    "width": 100e-15,
    "peak_power": 9.8,
    "shape": "sech",
    "n_save": 3,
    "backend": "numpy_ref",
}
CAPILLARY = {
    "wavelength": 800e-9,
    "wavelength_limits": (200e-9, 3e-6),
    "trange": 200e-15,
    "fwhm": 10e-15,
    "energy": 10e-6,
    "n_save": 2,
    "backend": "numpy_ref",
}


def gnlse(**changes):
    """Return a GNLSE simulation of a centimetre of fibre."""
    return Simulation.from_gnlse(
        0.11, 0.01, [0.0, 0.0, -1.183e-26], **{"noise": None, **GNLSE | changes}
    )


def capillary(pressure: Any = 1.0, gas: Any = "He", **changes):
    """Return a gas-filled capillary simulation a millimetre long."""
    return Simulation.from_capillary(
        125e-6, 1e-3, gas, pressure, **{"noise": None, **CAPILLARY | changes}
    )


def summary_of(simulation, *, backend="numpy_ref", execution=None, **keywords):
    """Return the summary of a simulation that has not run."""
    return run_summary(
        simulation.translate(),
        simulation.metadata(),
        backend=backend,
        execution={} if execution is None else execution,
        output=keywords.pop("output", MemoryOutput()),
        boundary_check=keywords.pop(
            "boundary_check", simulation.settings.boundary_check
        ),
        **keywords,
    )


def rows(simulation, **keywords):
    """Return the summary's rows as a mapping."""
    return dict(summary_of(simulation, **keywords).rows)


class Recording(NullReporter):
    """Record the order of the reporter calls a run makes."""

    def __init__(self):
        self.events = []
        self.text = None
        self.closing = None

    def summary(self, text):
        self.events.append("summary")
        if self.text is None:
            self.text = text
        else:
            self.closing = text

    def started(self, z_start, z_end, n_planes):
        self.events.append("started")

    def finished(self, total):
        self.events.append("finished")


# --------------------------------------------------------------------------- routes


def test_gnlse_summary_names_the_fibre_and_says_noise_and_raman_are_off():
    summary = summary_of(gnlse())
    table = dict(summary.rows)
    assert summary.title == "Starting propagation from z = 0 m to 10 mm"
    assert table["representation"] == "GNLSE, one mode, scalar"
    assert table["medium"] == "fibre, gamma 0.11 /W/m"
    assert table["responses"] == "Kerr\nRaman off"
    assert table["shot noise"] == "off"
    # The GNLSE takes the routes' default collar, with its edges watched (SEP 21).
    assert table["absorber"] == (
        "temporal, nonlinear scheme; "
        "edges watched (guards 0.05 of the window, budget 1e-03)"
    )
    # The window holds the request and the collar outside it, on a smooth size.
    assert table["grid"].startswith("envelope, 768 points, window ")
    assert table["start"] == "new run"
    assert table["output"] == "MemoryOutput"
    assert "pressure" not in table


def test_gnlse_raman_shows_the_fraction_held_back_from_kerr():
    table = rows(gnlse(raman=True))
    kerr, raman = table["responses"].split("\n")
    assert kerr == "Kerr"
    assert raman.startswith("Raman, fraction 0.21")
    assert raman.endswith("held back from Kerr")


def test_mode_averaged_capillary_with_noise_and_plasma():
    table = rows(capillary(mode_average=True, plasma=True, noise="input", seed=3))
    assert table["representation"] == "mode-averaged, one mode, scalar"
    assert table["medium"] == "He, 293.15 K, core radius 125 um"
    assert table["pressure"] == "1 bar"
    assert table["shot noise"] == "input model, seed 3"
    assert table["responses"].split("\n") == [
        "Kerr, third harmonic",
        "plasma (He), rate optmat recommended",
        "Raman off",
    ]
    assert table["absorber"] == (
        "temporal, nonlinear scheme; "
        "edges watched (guards 0.05 of the window, budget 1e-03)"
    )
    assert table["integrator"].startswith("Multistep, variable order up to ")
    assert table["tolerances"].startswith("rtol 1e-06, atol ")


def test_resolved_capillary_names_its_modal_integral_and_a_gradient_fill():
    table = rows(capillary(pressure=(1.0, 0.0), modes=2))
    assert table["representation"].startswith("2 modes, ")
    assert "modal integral on" in table["representation"]
    assert table["pressure"].startswith("PressureGradient(1 bar to ")
    one = rows(capillary(modes=1))
    assert one["representation"].startswith("1 mode, ")
    simulation = capillary(modes=1)
    unrecorded = dataclasses.replace(simulation.metadata(), quadrature=None)
    assert _representation(simulation.translate(), unrecorded) == "1 mode, scalar"


def test_a_mixture_lists_its_fractions():
    table = rows(capillary(mode_average=True, gas={"Ar": 0.5, "He": 0.5}))
    assert table["medium"].startswith("Ar 0.5 + He 0.5, 293.15 K")


def spatial_case(grid_kind="field"):
    constructor = (
        FieldGrid.from_limits
        if grid_kind == "field"
        else partial(EnvelopeGrid.from_limits, thg=False)
    )
    grid = constructor(1e-3, 800e-9, (600e-9, 1.2e-6), 150e-15)
    pulse = SpatialPulse(
        GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=3e-9), 60e-6
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-14),
        save_planes=[grid.length],
        dz_max=grid.length,
        dz_init=grid.length,
        backend="numpy_ref",
    )
    return grid, pulse, settings


def test_radial_cartesian_and_line_routes():
    grid, pulse, settings = spatial_case()
    radial = Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 32),
        SpatialMedium(),
        [],
        pulse,
        settings=settings,
        noise=None,
    )
    assert rows(radial)["representation"] == "radial, 32 points, scalar"
    assert rows(radial)["medium"].startswith("SpatialMedium(")
    free = Simulation.from_free(
        grid,
        FreeGrid(3e-4, 12, 4e-4, 16, window_factor=0),
        SpatialMedium(),
        [],
        pulse,
        settings=settings,
        noise=None,
    )
    assert rows(free)["representation"] == "Cartesian 12x16, scalar"
    line = Simulation.from_free(
        grid,
        LineGrid(1.5e-3, 64, window_factor=0),
        SpatialMedium(),
        [],
        pulse,
        settings=settings,
        noise=None,
    )
    assert rows(line)["representation"] == "Cartesian line, 64 points, scalar"


def test_planar_route():
    grid = EnvelopeGrid.from_limits(0.1, 800e-9, (650e-9, 1000e-9), 600e-15)
    space = PlanarGrid([SlabMode(63.5e-6, "Ar", 1.0, order=1)], 1e-3, 8)
    pulse = PlanarPulse(
        GaussianPulse(wavelength=800e-9, fwhm=100e-15, energy=1e-3),
        400e-6,
        waist_x=46e-6,
    )
    settings = SimulationSettings(
        controller=PIController(rtol=1e-8, atol=1e-12),
        save_planes=[0.1],
        dz_max=0.01,
        dz_init=0.01,
        backend="numpy_ref",
    )
    simulation = Simulation.from_planar(grid, space, [], pulse, settings=settings)
    assert rows(simulation)["representation"].startswith("planar, ")


# ------------------------------------------------------------------- the stepping


def test_the_integrator_row_names_each_march():
    assert rows(gnlse(integrator=RungeKutta()))["integrator"] == "RungeKutta (DOPRI5)"
    fixed = rows(gnlse(integrator=Multistep(order=4, variable_order=False)))
    assert fixed["integrator"].startswith("Multistep, order 4, ")
    low = rows(
        gnlse(
            integrator=LowStorageRungeKutta(),
            schedule=FixedSchedule(np.full(4, 0.0025)),
            # Four 2.5 mm steps are past the default collar's stability limit.
            absorber=False,
        )
    )
    assert low["integrator"] == "LowStorageRungeKutta (rk4ip)"
    assert low["absorber"].startswith("none; edges watched")
    assert low["tolerances"].endswith("replaying a fixed schedule of 4 steps")


def test_the_planes_row_names_per_step_and_interpolated_saving():
    simulation = gnlse()
    problem = simulation.translate()
    stepper = dataclasses.replace(
        problem.stepper, step_planes=3, interpolate_planes=True
    )
    summary = run_summary(
        dataclasses.replace(problem, stepper=stepper),
        simulation.metadata(),
        backend="numpy_ref",
        execution={},
        output=MemoryOutput(),
    )
    planes = dict(summary.rows)["save planes"]
    assert planes.endswith(", and every 3 accepted steps, interpolated")


def test_precision_is_the_state_s():
    simulation = gnlse()
    problem = simulation.translate()
    summary = run_summary(
        dataclasses.replace(problem, precision=Precision("single", "single")),
        simulation.metadata(),
        backend="numpy_ref",
        execution={},
        output=MemoryOutput(),
    )
    assert dict(summary.rows)["precision"] == "single"


def test_threads_come_from_the_backend_s_record():
    numba: dict[str, object] = {"numba_threads": 4, "fftw_threads": 2}
    numba |= {"thread_budget": 8, "thread_budget_source": "affinity"}
    assert _threads("numba_cpu", numba) == "4 kernel, 2 FFTW; budget 8 (affinity)"
    native = {"thread_budget": 6, "thread_budget_source": "SOLPROP_THREADS"}
    assert _threads("native", native) == "budget 6 (SOLPROP_THREADS)"
    assert _threads("numpy_ref", {}).startswith("1 (NumPy reference); budget ")
    assert _threads("other", {}).startswith("budget ")


def test_the_code_row_leads_with_the_commit_in_a_checkout():
    metadata = gnlse().metadata()
    release = dataclasses.replace(metadata, versions={"solprop": "1.2.0"}, git=None)
    assert _code(release) == "1.2.0"
    git = GitProvenance(commit="0123456789ab" + "c" * 28, branch="main", dirty=True)
    checkout = dataclasses.replace(
        release, versions={"solprop": "1.2.1.dev3+gabc"}, git=git
    )
    assert _code(checkout) == (
        "commit 0123456789ab, modified (main, installed as 1.2.1.dev3+gabc)"
    )
    clean = dataclasses.replace(checkout, git=dataclasses.replace(git, dirty=False))
    assert _code(clean).startswith("commit 0123456789ab (main, ")
    assert list(rows(gnlse()))[:3] == ["solprop", "machine", "backend"]


def test_the_machine_row_leaves_out_what_was_not_read():
    record = {
        "system": "Linux",
        "release": "6.8.0",
        "machine": "x86_64",
        "cpu_model": "AMD EPYC 9R45",
        "logical_cpus": 192,
        "usable_cpus": 16,
        "memory_bytes": 768 * 2**30,
    }
    assert _machine(record) == (
        "x86_64, AMD EPYC 9R45, 192 CPUs (16 usable), 768 GiB, Linux 6.8.0"
    )
    bare = record | {"cpu_model": None, "memory_bytes": None, "logical_cpus": None}
    assert _machine(bare) == "x86_64, Linux 6.8.0"


def test_resources_are_written_in_seconds_or_hours_and_binary_units():
    assert memory_size(512) == "512 B"
    assert memory_size(1536) == "1.5 KiB"
    assert memory_size(3 * 2**40) == "3 TiB"
    spent = {"wall_time_s": 4000.0, "cpu_time_s": 12.5, "peak_rss_bytes": 2**30}
    assert resources(spent) == (
        "resources: 1:06:40 wall, 12.5 s CPU, process peak memory 1 GiB"
    )


def test_a_file_records_each_execution_s_machine_and_resources(tmp_path):
    path = tmp_path / "run.h5"
    with pytest.raises(KeyboardInterrupt):
        gnlse().run(path, progress=Stop())
    with open_file(path) as stored:
        assert stored.outcome is not None
        failed: Any = stored.outcome.execution
    assert failed["host"]["usable_cpus"] >= 1
    assert failed["resources"]["wall_time_s"] >= 0.0
    gnlse().run(path, progress="none", resume=True)
    with open_file(path) as stored:
        assert stored.outcome is not None
        assert stored.outcome.status == "complete"
        execution: Any = stored.outcome.execution
        earlier: Any = stored.outcome.previous_executions[0]
        assert len(stored.outcome.previous_executions) == 1
    assert earlier["execution"]["resources"] == failed["resources"]
    assert execution["host"] == failed["host"]
    assert set(execution["resources"]) == {
        "wall_time_s",
        "cpu_time_s",
        "peak_rss_bytes",
    }


def test_a_resume_says_where_it_continues():
    table = rows(gnlse(), start_plane=1, z_start=0.005)
    assert table["start"] == "resuming at z = 5 mm, 1 of 2 planes to go"


def test_a_file_output_is_named_by_the_end_of_its_path(tmp_path):
    path = tmp_path / ("deep" * 30) / "run.h5"
    path.parent.mkdir()
    table = rows(gnlse(), output=HDF5Output(path))
    assert table["output"].startswith("HDF5Output ...")
    assert table["output"].endswith("run.h5")
    assert len(table["output"]) == len("HDF5Output ") + 52


def test_a_grid_reaching_zero_frequency_has_no_longest_wavelength():
    simulation = gnlse()
    metadata = simulation.metadata()
    zero = dataclasses.replace(
        metadata, grid=dataclasses.replace(metadata.grid, omega_min=0.0)
    )
    summary = run_summary(
        simulation.translate(),
        zero,
        backend="numpy_ref",
        execution={},
        output=MemoryOutput(),
    )
    assert dict(summary.rows)["grid"].endswith("band above 500 nm")


# ------------------------------------------------------- descriptions of the physics


def test_medium_without_a_record():
    assert _medium({}) == "not recorded"
    assert _medium({"medium": {"__class__": "Vacuum"}}) == "Vacuum"
    assert _medium({"medium": {}}) == "recorded"
    assert _medium({"gnlse": SimpleNamespace()}) == "fibre"
    air = Mixture.air(1.0)
    assert _medium({"medium": air}) == repr(air)
    assert _medium({"medium": {"labels": ["He"], "temperature_K": None}}) == "He"


def test_each_response_is_described_by_its_model():
    rate = OptmatRate("Ar", 800e-9)
    assert _response(KerrResponse(1.0, thg=False, species="Ar")) == "Kerr (Ar)"
    assert _response(PlasmaResponse("Ar", rate)) == (
        "plasma (Ar), rate optmat recommended"
    )
    assert _response(EnvelopePlasmaResponse("Ar", rate)) == (
        "envelope plasma (Ar), rate optmat recommended"
    )
    assert _response(SilicaTensorResponse()).startswith("Raman, fraction ")
    assert _rate(SimpleNamespace()) == "SimpleNamespace"
    assert _response(object()) == "object"
    tdse = type("TDSEGasResponse", (), {"target": "Ar", "model": "sae"})()
    assert _response(tdse) == "TDSE (Ar, sae)"
    dispersive = type(
        "DispersiveKerrResponse", (), {"model": SimpleNamespace(), "species": None}
    )()
    assert _response(dispersive) == "dispersive Kerr, SimpleNamespace"
    raman = type("OptmatRamanResponse", (), {"labels": ("N2",), "fraction": None})()
    assert _response(raman) == "Raman (N2)"


def test_responses_fall_back_to_what_was_compiled():
    problem = gnlse(raman=True).translate()
    assert _responses(problem, {}) == "Kerr\nRaman (1 terms)"
    assert _compiled(
        kerr=False, raman_terms=None, plasma=True, dispersive=True, tdse=True
    ) == ["plasma", "dispersive Kerr", "TDSE"]
    tdse = type("TDSEGasResponse", (), {"target": "Ar", "model": "sae"})()
    assert _responses(problem, {"responses": [tdse]}) == "TDSE (Ar, sae)\nRaman off"


def test_representation_of_a_general_guide_and_of_two_polarisations():
    simulation = gnlse()
    problem = simulation.translate()
    metadata = simulation.metadata()
    general = dataclasses.replace(
        metadata, configuration={}, simulation_kind="multimode", modal={"labels": []}
    )
    assert _representation(problem, general) == "general guide, 1 channels, scalar"
    grid, pulse, settings = spatial_case()
    circular = dataclasses.replace(pulse, jones=(1.0 + 0.0j, 1.0j))
    radial = Simulation.from_radial(
        grid,
        RadialGrid(6e-4, 16),
        SpatialMedium(),
        [],
        circular,
        settings=settings,
        noise=None,
    )
    assert rows(radial)["representation"] == "radial, 16 points, vector (x and y)"


def test_the_summary_prints_as_a_block_and_records_as_json():
    summary = RunSummary("Starting", (("noise", "off"),))
    assert repr(summary) == "RunSummary('Starting', 1 rows)"
    assert str(summary) == summary.text() == "Starting\n  noise: off"
    assert summary.record() == {"title": "Starting", "rows": [["noise", "off"]]}


# --------------------------------------------------------------------- where it goes


def test_a_run_shows_the_summary_before_it_starts_and_keeps_it():
    reporter = Recording()
    output = gnlse().run(progress=reporter)
    assert isinstance(output, MemoryOutput)
    assert reporter.events == ["summary", "started", "finished", "summary"]
    assert reporter.closing is not None
    assert reporter.closing.startswith("resources: ")
    assert " wall, " in reporter.closing and " CPU" in reporter.closing
    assert reporter.text is not None
    assert reporter.text.startswith("Starting propagation from z = 0 m to 10 mm\n")
    records: Any = output.annotations[RUN_SUMMARY_ANNOTATION]
    (record,) = records
    assert record["title"] == reporter.text.split("\n")[0]
    assert ["shot noise", "off"] in record["rows"]


def test_the_closing_line_of_a_bar_run_is_shown_on_its_stream():
    stream = io.StringIO()
    gnlse().run(progress=BarReporter(file=stream, ncols=80, ascii=True))
    text = stream.getvalue()
    assert text.startswith("Starting propagation from z = 0 m to 10 mm\n")
    assert "  shot noise:     off\n" in text
    assert "finished at z = 1.000000e-02 m: " in text
    assert "\nresources: " in text


def test_a_reporter_without_the_call_is_not_shown_the_summary():
    class Planes:
        def started(self, z_start, z_end, n_planes):
            pass

        def step(self, z, dz, accepted):
            pass

        def plane(self, z, summary):
            pass

        def finished(self, total):
            pass

    assert not isinstance(Planes(), SummarisingReporter)
    output = gnlse().run(progress=Planes())
    assert isinstance(output, MemoryOutput)
    records: Any = output.annotations[RUN_SUMMARY_ANNOTATION]
    assert len(records) == 1


def test_progress_none_shows_nothing(capsys, caplog):
    with caplog.at_level(logging.DEBUG, logger="solprop.log"):
        gnlse().run(progress="none")
    assert capsys.readouterr().err == ""
    assert not [r for r in caplog.records if r.name == "solprop.log"]


def test_progress_log_logs_the_summary_at_info(caplog):
    with caplog.at_level(logging.INFO, logger="solprop.log"):
        gnlse().run(progress="log", progress_period=0)
    messages = [r.getMessage() for r in caplog.records if r.name == "solprop.log"]
    assert messages[0].startswith("Starting propagation from z = 0 m to 10 mm\n")
    assert messages[-2].startswith("finished at z = ")
    assert messages[-1].startswith("resources: ")
    assert all(r.levelno == logging.INFO for r in caplog.records[:1])


def test_a_direct_run_of_a_problem_reports_what_was_compiled():
    simulation = gnlse()
    problem = simulation.translate()
    from solprop.backends.registry import select  # noqa: PLC0415 - only this test

    backend = select("numpy_ref", problem, grid=simulation.grid, modes=simulation.modes)
    backend.set_initial(problem.initial.field0, None)
    reporter = Recording()
    output = MemoryOutput()
    try:
        run_problem(
            problem,
            backend,
            output,
            metadata=metadata_for(problem, simulation.grid, modes=simulation.modes),
            progress=reporter,
        )
    finally:
        backend.close()
    assert reporter.text is not None
    assert "  responses:      Kerr\n                  Raman off\n" in reporter.text
    assert "  medium:         not recorded\n" in reporter.text


class Stop(NullReporter):
    """Stop a run at its first save plane."""

    def plane(self, z, summary):
        raise KeyboardInterrupt


def test_a_resumed_run_adds_its_summary(tmp_path):
    path = tmp_path / "run.h5"
    with pytest.raises(KeyboardInterrupt):
        gnlse().run(path, progress=Stop())
    gnlse().run(path, progress="none", resume=True)
    with open_file(path) as stored:
        starts: Any = stored.annotations[RUN_SUMMARY_ANNOTATION]
    assert len(starts) == 2
    assert ["start", "new run"] in starts[0]["rows"]
    assert dict(map(tuple, starts[1]["rows"]))["start"].startswith("resuming at z = ")


def test_the_noise_row_of_a_printed_simulation_gives_model_and_seed():
    assert "noise:     input model, seed 5" in str(gnlse(noise="input", seed=5))
    assert "noise:     none" in str(gnlse())


def test_the_line_reporter_summary_is_one_record():
    logger = logging.getLogger("solprop.test_summary_line")
    logger.propagate = False
    logger.setLevel(logging.INFO)
    stream = io.StringIO()
    logger.addHandler(logging.StreamHandler(stream))
    try:
        LogReporter(logger=logger).summary("Starting\n  noise: off")
    finally:
        logger.handlers.clear()
    assert stream.getvalue() == "Starting\n  noise: off\n"
    assert isinstance(LogReporter(), SummarisingReporter)
    assert NullReporter().summary("ignored") is None


def test_the_pressure_row_is_a_constant_or_the_profile_itself():
    assert _pressure({}) is None
    assert _pressure({"profiles": {}}) is None
    profile = Linear(1.0, 2.0, 1e-3)
    assert _pressure({"profiles": {"pressure": profile}}) == repr(profile)
