# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Memory admission against a declared budget, before the backend allocates.

:func:`solprop.driver.memory.admit_memory` refuses or warns on a run whose
construction-free estimate exceeds its budget, and the run's backend record carries the
admission.
"""

import math
import subprocess
import sys
import tracemalloc
import warnings

import pytest

import solprop.driver.memory as memory_module
from modal_provider_cases import GRIN_WAVELENGTH, grin_grid, grin_guide
from solprop.backends.memory import (
    MemoryEstimate,
    estimate_guide_memory,
    estimate_memory,
)
from solprop.driver import Simulation
from solprop.driver.executors import pool
from solprop.driver.memory import (
    DEFAULT_HEADROOM,
    MEMORY_BUDGET_VARIABLE,
    MemoryLimit,
    _over_budget,
    admit_memory,
    format_bytes,
    parse_memory_budget,
    resolve_memory_budget,
)
from solprop.exceptions import (
    DriverError,
    GeneralModalBasisWarning,
    MemoryBudgetError,
    MemoryBudgetWarning,
)
from solprop.grid import FieldGrid
from solprop.output.hdf5 import open_file
from solprop.stepper import Multistep, RungeKutta

# The fixtures here launch noise-free into time windows kept tight for speed, which
# hold more than a millionth of the energy in their guards or collar: the boundary
# monitor warns (SEP 21), and the window's edge is not what this module tests.
pytestmark = pytest.mark.filterwarnings("ignore::solprop.exceptions.BoundaryWarning")


@pytest.fixture
def no_declaration(monkeypatch):
    """Clear the environment's declaration, so the default is what is tested."""
    monkeypatch.delenv(MEMORY_BUDGET_VARIABLE, raising=False)


@pytest.mark.parametrize(
    ("declared", "limit", "action"),
    [
        (4096, 4096, "refuse"),
        ("4096", 4096, "refuse"),
        ("16 GiB", 16 * 2**30, "refuse"),
        ("12GB", 12 * 10**9, "refuse"),
        ("1.5 kib", 1536, "refuse"),
        ("8e9", 8 * 10**9, "refuse"),
        ("2 TiB", 2 * 2**40, "refuse"),
        ("warn: 64 MiB", 64 * 2**20, "warn"),
        ("WARN:3MB", 3 * 10**6, "warn"),
        ("none", None, "refuse"),
        ("warn:none", None, "warn"),
    ],
)
def test_parse_memory_budget_reads_units_and_actions(declared, limit, action):
    budget = parse_memory_budget(declared)
    assert (budget.limit, budget.action, budget.source) == (limit, action, "keyword")


@pytest.mark.parametrize("declared", ["lots", "16 GiBs", "-4 GiB", "", "warn:"])
def test_parse_memory_budget_refuses_what_it_cannot_read(declared):
    with pytest.raises(DriverError, match="cannot read the memory budget"):
        parse_memory_budget(declared)


@pytest.mark.parametrize("declared", [0, -1, "0 GiB", "0.1"])
def test_memory_limit_refuses_a_budget_that_is_not_positive(declared):
    with pytest.raises(DriverError, match="must be a positive byte count"):
        parse_memory_budget(declared)


def test_resolve_memory_budget_prefers_the_keyword_to_the_environment(monkeypatch):
    monkeypatch.setenv(MEMORY_BUDGET_VARIABLE, "1 GiB")
    assert resolve_memory_budget("2 GiB") == MemoryLimit(2 * 2**30, "refuse", "keyword")
    assert resolve_memory_budget() == MemoryLimit(
        2**30, "refuse", MEMORY_BUDGET_VARIABLE
    )


def test_resolve_memory_budget_names_the_variable_it_cannot_read(monkeypatch):
    monkeypatch.setenv(MEMORY_BUDGET_VARIABLE, "plenty")
    with pytest.raises(DriverError, match=MEMORY_BUDGET_VARIABLE):
        resolve_memory_budget()


def test_resolve_memory_budget_defaults_to_the_machine_with_headroom(
    monkeypatch, no_declaration
):
    monkeypatch.setattr(pool, "_memory_limit", lambda: 10_000)
    budget = resolve_memory_budget()
    assert budget == MemoryLimit(8_000, "warn", "machine")
    assert budget.limit == 10_000 * (1 - DEFAULT_HEADROOM)


def test_admit_memory_passes_a_run_within_its_budget(build_simulation):
    problem = build_simulation(integrator=RungeKutta()).translate()
    admission = admit_memory(problem, "numpy_ref", "1 GiB")
    assert admission.estimate == estimate_memory(problem, "numpy_ref")
    assert admission.record() == {
        "estimate": admission.estimate.total,
        "groups": {"base": admission.estimate.total},
        "budget": 2**30,
        "budget_source": "keyword",
    }


def test_admit_memory_refuses_by_name(build_simulation):
    problem = build_simulation(integrator=Multistep()).translate()
    total = estimate_memory(problem, "numba_cpu").total
    with pytest.raises(MemoryBudgetError) as refused:
        admit_memory(problem, "numba_cpu", total - 1)
    message = str(refused.value)
    assert f"estimated at {format_bytes(total)}" in message
    assert f"budget of {format_bytes(total - 1)} (keyword)" in message
    assert "Largest groups: base" in message
    assert "propagator_mode='recompute'" in message
    assert MEMORY_BUDGET_VARIABLE in message


def test_admit_memory_warns_when_the_declaration_asks(build_simulation):
    problem = build_simulation(integrator=Multistep()).translate()
    with pytest.warns(MemoryBudgetWarning, match="over the budget of 1.00 KiB"):
        admission = admit_memory(problem, "numpy_ref", "warn:1 KiB")
    assert admission.budget.action == "warn"


def test_admit_memory_logs_the_estimate_at_debug(build_simulation, caplog):
    problem = build_simulation(integrator=Multistep()).translate()
    with caplog.at_level("DEBUG", logger="solprop.driver.memory"):
        admit_memory(problem, "numpy_ref", "none")
    assert "against a budget of none (keyword)" in caplog.text


def test_admit_memory_checks_nothing_when_declared_none(build_simulation):
    admission = admit_memory(
        build_simulation(integrator=Multistep()).translate(), "numpy_ref", "none"
    )
    assert admission.record()["budget"] is None


def test_over_budget_names_the_resident_reference_and_the_route_advice():
    estimate = MemoryEstimate(
        "native",
        {"general_reference": 3 * 2**30, "general_maps": 2**30, "coupled_flow": 5},
        resident=3 * 2**30,
    )
    message = _over_budget(
        estimate, MemoryLimit(2**30, "refuse", "machine"), "the run's memory"
    )
    assert "3.00 GiB of it, the general route's reference" in message
    assert "general_reference 3.00 GiB, general_maps 1.00 GiB, coupled_flow 5 B" in (
        message
    )
    assert "a coarser node rule" in message
    assert "Chebyshev terms" in message


def _history_limit(problem, backend="numpy_ref"):
    """Return the estimate, its Adams history and a limit only the history exceeds."""
    estimate = estimate_memory(problem, backend)
    history = estimate.groups["adams_history"]
    assert history > 0
    return estimate, history, estimate.total - history // 2


def test_an_adams_history_that_does_not_fit_the_machine_is_refused(
    build_simulation, monkeypatch, no_declaration
):
    # The undeclared machine budget only warns, but a run that fits but for the Adams
    # history is refused, naming the Runge-Kutta march, which fits.
    problem = build_simulation(integrator=Multistep()).translate()
    _, history, limit = _history_limit(problem)
    monkeypatch.setattr(pool, "_memory_limit", lambda: limit / (1 - DEFAULT_HEADROOM))
    with pytest.raises(MemoryBudgetError) as refused:
        admit_memory(problem, "numpy_ref")
    message = str(refused.value)
    assert "integrator=RungeKutta()" in message
    assert f"{format_bytes(history)} of it is the Adams method's history" in message
    assert "the machine's memory" in message
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        admit_memory(build_simulation(integrator=RungeKutta()).translate(), "numpy_ref")


def test_a_run_over_the_machine_without_its_history_is_only_warned_on(
    build_simulation, monkeypatch, no_declaration
):
    problem = build_simulation(integrator=Multistep()).translate()
    estimate, history, _ = _history_limit(problem)
    limit = estimate.total - history - 1
    monkeypatch.setattr(pool, "_memory_limit", lambda: limit / (1 - DEFAULT_HEADROOM))
    with pytest.warns(MemoryBudgetWarning, match="RungeKutta"):
        admit_memory(problem, "numpy_ref")


def test_a_declared_warning_budget_is_not_turned_into_a_refusal(build_simulation):
    problem = build_simulation(integrator=Multistep()).translate()
    _, _, limit = _history_limit(problem)
    with pytest.warns(MemoryBudgetWarning, match="adams_history"):
        admit_memory(problem, "numpy_ref", f"warn:{limit}")


def test_an_adams_history_over_the_device_is_refused(build_simulation, monkeypatch):
    # The accelerator's free memory, less the headroom, is checked under any declared
    # budget but "none" and "warn:".
    problem = build_simulation(integrator=Multistep()).translate()
    _, _, limit = _history_limit(problem)
    free = math.ceil(limit / (1 - DEFAULT_HEADROOM))
    monkeypatch.setattr(memory_module, "available_device_memory", lambda _: free)
    with pytest.raises(MemoryBudgetError, match="accelerator's free memory"):
        admit_memory(problem, "numpy_ref", "1 TiB")
    assert admit_memory(problem, "numpy_ref", "none").budget.limit is None


def test_the_advice_for_an_adams_history_names_the_runge_kutta_march():
    estimate = MemoryEstimate("native", {"adams_history": 3 * 2**30, "base": 2**30})
    message = _over_budget(
        estimate, MemoryLimit(2**30, "refuse", "keyword"), "the run's memory"
    )
    assert "adams_history: integrator=RungeKutta()" in message


@pytest.mark.parametrize(
    ("size", "text"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.00 KiB"),
        (150 * 2**20, "150 MiB"),
        (5 * 2**40, "5.00 TiB"),
        (2000 * 2**40, "2e+03 TiB"),
    ],
)
def test_format_bytes_uses_binary_units(size, text):
    assert format_bytes(size) == text


def test_simulation_run_refuses_before_writing_anything(build_simulation, tmp_path):
    path = tmp_path / "refused.h5"
    with pytest.raises(MemoryBudgetError, match="over the budget"):
        build_simulation(integrator=Multistep()).run(
            path, progress="none", memory_budget="1 kB"
        )
    assert not path.exists()


@pytest.mark.medium
def test_simulation_run_records_its_admission(
    build_simulation, tmp_path, no_declaration
):
    simulation = build_simulation(backend="numpy_ref", integrator=Multistep())
    path = tmp_path / "admitted.h5"
    simulation.run(path, progress="none")
    with open_file(path) as stored:
        assert stored.metadata is not None
        assert stored.metadata.backend is not None
        record = stored.metadata.backend.settings["memory"]
    assert isinstance(record, dict)
    estimate = estimate_memory(simulation.translate(), "numpy_ref")
    # The default integrator's Adams history is its own group.
    history = estimate.groups["adams_history"]
    assert history > 0
    assert record == {
        "estimate": estimate.total,
        "groups": {"base": estimate.groups["base"], "adams_history": history},
        "budget": record["budget"],
        "budget_source": "machine",
    }
    budget = record["budget"]
    assert isinstance(budget, int)
    assert budget > estimate.total


@pytest.mark.medium
def test_simulation_run_warns_and_still_runs(build_simulation):
    with pytest.warns(MemoryBudgetWarning):
        handler = build_simulation(backend="numpy_ref", integrator=Multistep()).run(
            progress="none", memory_budget="warn:1 kB"
        )
    assert handler.n_saved == 4


def test_the_scan_output_imports_first_without_a_cycle():
    """`solprop.output.scan` imported before the driver loads, as autodoc imports it.

    It imports the driver package, whose simulation imports this module; a module-level
    import of the executors here re-entered the half-initialised scan output.
    """
    completed = subprocess.run(
        [sys.executable, "-c", "import solprop.output.scan, solprop.processing.scans"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


GUIDE_RUN = {
    "wavelength": GRIN_WAVELENGTH,
    "fwhm": 60e-15,
    "energy": 1e-12,
    "backend": "numpy_ref",
}
"""The graded-index pulse of the general-route tests, on the reference backend."""


@pytest.fixture(scope="module")
def guide():
    """The two-channel graded-index guide over its grid's active band."""
    return grin_guide(grin_grid(), knots=9)


def test_a_large_guide_is_refused_before_its_reference_is_built(guide):
    """A 40 ps window's reference is refused at construction, never allocated.

    Bounds, stated first. The estimate on this grid is 21 GiB, 18 GiB of it the
    reference (32769 bins, 512 nodes, two channels, two positions); the budget is
    8 GiB, declared to refuse. Everything construction allocates before the refusal
    -- the grid's axes and the save planes -- is a few arrays of the grid's 131072
    samples, so the traced peak must stay under a hundredth of the estimate: had the
    route been built, the peak would exceed the estimate itself.
    """
    grid = FieldGrid.from_limits(1e-3, GRIN_WAVELENGTH, (1.0e-6, 1.06e-6), 40e-12)
    estimate = estimate_guide_memory(guide, grid, z_max=1e-3, backend="numpy_ref")
    assert estimate.total > 8 * 2**30
    assert estimate.largest(1)[0][0] == "general_reference"
    tracemalloc.start()
    try:
        with pytest.raises(MemoryBudgetError, match="before they are built"):
            Simulation.from_waveguide(
                (),
                1e-3,
                provider=guide,
                grid=grid,
                memory_budget="8 GiB",
                **{"integrator": Multistep(), **({"noise": None, **GUIDE_RUN})},
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert 0 < peak < estimate.total / 100


@pytest.mark.medium
def test_a_guide_over_a_warning_budget_is_warned_on_and_built(guide, no_declaration):
    """A ``warn:`` declaration warns at construction and the route is built."""
    with (
        warnings.catch_warnings(),
        pytest.warns(MemoryBudgetWarning, match="general modal guide's reference"),
    ):
        # The guide's knots are the fixture's, and its basis warns over the active
        # band's far bins; that is not what is tested here.
        warnings.simplefilter("ignore", GeneralModalBasisWarning)
        simulation = Simulation.from_waveguide(
            (),
            1e-3,
            provider=guide,
            grid=grin_grid(),
            memory_budget="warn: 1 kB",
            **{"integrator": Multistep(), **({"noise": None, **GUIDE_RUN})},
        )
    assert simulation.general is not None


def test_a_memory_budget_without_a_guide_is_refused_by_name():
    """The gas-filled route builds no reference at construction, so it has no use."""
    with pytest.raises(DriverError, match="memory_budget= belongs to the general"):
        Simulation.from_waveguide(
            [],
            1e-3,
            "He",
            1.0,
            wavelength=800e-9,
            memory_budget="1 GiB",
            noise=None,
            integrator=Multistep(),
        )
