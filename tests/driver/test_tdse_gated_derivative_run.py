# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Run ``gated_derivative`` end to end with the fake providers (SEP 26 section 7.2).

The SAE runs use a manufactured host binding whose remainder is a known function of
time (``tests/edge_source_cases.py``); the haCC runs use the fake haCC provider of
``tests/hacc_cases.py``. Every run is a few fixed steps. Identities, resume keys,
ledgers and accounting are exact (class P); the edge ratio of the last evaluation is a
matched recomputation (class R) within its own recorded floors.
"""

import dataclasses
import json
import warnings
from typing import Any

import numpy as np
import pytest
from scipy.special import erf
from tests.edge_source_cases import ManufacturedBinding, case_from_problem, evaluate
from tests.hacc_cases import alpha_law, hacc_response, install, plain, preset_basis

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import Simulation
from solprop.driver import tdse as runtime_module
from solprop.exceptions import (
    MaterialRangeWarning,
    OutputError,
    ResponseBudgetError,
    TDSEDomainWarning,
    TDSEUnqualifiedModelWarning,
)
from solprop.grid import FieldGrid
from solprop.nonlinear.tdse import (
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEExecution,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.output import open_file, resume
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep
from solprop.translate.tdse_source import STEADY_CURRENT_CERTIFICATE_LIMIT

pytestmark = [
    pytest.mark.medium,
    pytest.mark.filterwarnings("ignore::solprop.exceptions.TDSECostWarning"),
]

_STEP = 2.0**-20
"""A binary-exact fixed step, so every run takes whole steps."""

_DRIFT = 1e-27
"""Per-atom drift current in A m: far below the optical scale, so no step rejects."""


def _drift(times):
    """A current that rises on the plateau at t = 0 and is steady through the taper."""
    return _DRIFT / 2 * (1 + erf(times / 1e-15))


def _sae(window, *, limit=40, steps=6):
    parent = Simulation.from_capillary(
        256e-6,
        steps * _STEP,
        "He",
        0.1,
        wavelength=800e-9,
        wavelength_limits=(400e-9, 1.6e-6),
        trange=60e-15,
        fwhm=6e-15,
        energy=1e-12,
        modes=1,
        n_r=4,
        n_save=2,
        plasma=False,
        noise=None,
        backend="numpy_ref",
        integrator=Multistep(formulation="interaction"),
        dz_init=_STEP,
        dz_max=_STEP,
    )
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 5, 1e-16, source="SEP 26 run fixture"
    )
    calibration = TDSECalibration(
        "sep26-run",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="manufactured drift; no atomic calculation",
        limitations=("no physical qualification",),
    )
    response = TDSEGasResponse(
        "helium",
        basis,
        1e-19,
        "tdse_native",
        1.0,
        calibration,
        # q = 8: a 969-sample collar taper and 180 samples for 5 fs.
        TDSESampling(8),
        atomic_window=window,
    )
    return Simulation(
        parent.grid,
        parent.modes,
        [response],
        parent._field0,
        settings=dataclasses.replace(
            parent.settings,
            tdse_execution=TDSEExecution(max_atomic_calls=limit),
            absorber=None,
            stats=StatsSpec(names=("energy",)),
        ),
        noise=None,
        profiles=parent.profiles,
        frame=parent.frame,
    )


@pytest.fixture
def bindings(monkeypatch):
    made = []

    def prepare(descriptor, execution, *args, **kwargs):
        del args, kwargs
        binding = ManufacturedBinding(descriptor, execution.max_atomic_calls, _drift)
        made.append(binding)
        return binding

    monkeypatch.setattr(runtime_module, "prepare_binding", prepare)
    return made


_SAE_GATES = pytest.mark.parametrize(
    "window", [None, TDSEAtomicWindow(40e-15, 5e-15)], ids=["collar", "window"]
)


@_SAE_GATES
def test_sae_run_records_certificate_round_trips_and_resumes(
    bindings, tmp_path, window
):
    simulation = _sae(window, limit=8)
    path = tmp_path / "held.h5"
    with pytest.raises(ResponseBudgetError):
        simulation.run(path, progress="none")
    first = resume(path)
    assert first.outcome is not None and first.outcome.status == "budget_exhausted"
    with open_file(path) as saved:
        assert saved.metadata is not None
        key = saved.metadata.configuration_hash()
        descriptor = json.loads(plain(saved.metadata.tdse)["configuration_json"])
    assert descriptor["schema"] == "solprop.tdse-descriptor.v5"
    assert "source_filter" not in descriptor["response"]
    execution = plain(first.outcome.execution)
    record = execution["tdse_source"]
    assert record["gated"] is True
    assert record["steady_current_certificate"] <= STEADY_CURRENT_CERTIFICATE_LIMIT
    edge = execution["tdse_last_evaluation"]["projection"]["edge"]
    assert edge["steady_current_certificate"] == record["steady_current_certificate"]
    simulation.settings = dataclasses.replace(
        simulation.settings, tdse_execution=TDSEExecution(max_atomic_calls=1000)
    )
    simulation.run(path, resume=True, progress="none")
    final = resume(path)
    assert final.outcome is not None and final.outcome.status == "complete"
    with open_file(path) as saved:
        assert saved.metadata is not None
        assert saved.metadata.configuration_hash() == key
    assert final.plane.field is not None
    assert np.all(np.isfinite(final.plane.field))


def test_a_run_refuses_to_resume_under_another_gate(bindings, tmp_path):
    # The gate is part of the source: the grid collar and a window are different runs.
    window = TDSEAtomicWindow(40e-15, 5e-15)
    for written, other in ((None, window), (window, None)):
        path = tmp_path / f"{'collar' if written is None else 'window'}.h5"
        with pytest.raises(ResponseBudgetError):
            _sae(written, limit=8).run(path, progress="none")
        with pytest.raises(OutputError):
            _sae(other, limit=1000).run(path, resume=True, progress="none")
    assert bindings


def test_the_fake_drift_keeps_its_rise_and_reports_its_edge_ratio(bindings):
    window = TDSEAtomicWindow(40e-15, 5e-15)
    simulation = _sae(window)
    output: Any = simulation.run(progress="none")
    assert output.outcome is not None
    last = plain(output.outcome.execution)["tdse_last_evaluation"]
    record = last["projection"]["edge"]["bands"]["active"]
    # T2 (c) on this run's own problem, position and drift, through the source
    # assembly alone and without the run's linear shadow.
    grid = simulation.grid
    assert isinstance(grid, FieldGrid)
    case = case_from_problem(simulation.translate(), grid, last["z"])
    direct = evaluate(case, _drift(case.times)).diagnostics["edge"]
    direct = direct["bands"]["active"]
    # Class R: the same remainder, gate and factors. The run's S carries the
    # rounding of J - j_1, which its recorded floors bound; the ratio moves by at
    # most twice the two norms' relative floors.
    allowance = 2 * (
        record["truncating_floor"] / record["truncating_norm"]
        + record["edge_floor"] / record["edge_norm"]
    )
    assert record["edge_over_truncating"] == pytest.approx(
        direct["edge_over_truncating"], rel=allowance
    )
    assert record["edge_norm"] > 0
    # The rise is kept: the corrected source is not the steady current's zero.
    assert direct["corrected_norm"] > 0.1 * direct["truncating_norm"]


def test_the_edge_entry_is_the_last_evaluation_and_keeps_no_history(bindings):
    simulation = _sae(TDSEAtomicWindow(40e-15, 5e-15))
    problem = simulation.translate()
    assert problem.tdse is not None
    binding = ManufacturedBinding(problem.tdse, 1000, _drift)
    backend = NumpyRefBackend(problem, grid=simulation.grid, tdse_binding=binding)
    try:
        backend.set_initial(problem.initial.field0, None, z=problem.initial.z)
        backend.propagate_to(2 * _STEP)
        first = plain(backend.metadata()["tdse_last_evaluation"])
        binding.law = lambda times: 3 * _drift(times)
        backend.propagate_to(4 * _STEP)
        second = plain(backend.metadata()["tdse_last_evaluation"])
    finally:
        backend.close()
    before = first["projection"]["edge"]["bands"]["active"]
    after = second["projection"]["edge"]["bands"]["active"]
    # A linear map of a remainder three times larger: three times the norms, to the
    # recorded floors; no list of earlier evaluations anywhere in the record.
    assert after["edge_norm"] == pytest.approx(
        3 * before["edge_norm"], rel=4 * before["edge_floor"] / before["edge_norm"]
    )
    assert set(first) == set(second) == {"purpose", "z", "projection"}
    assert not any(isinstance(value, list) for value in after.values())
    assert first["z"] < second["z"]


# --- haCC through the fake provider: windowed and on the grid collar -----------------

_HACC_LENGTH = 6 * 2.0**-20


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache"))
    fake = install(monkeypatch)
    fake.response = alpha_law(1.02, lambda5=2.0**-200)
    return fake


def _hacc(window, *, limit=1000):
    response = hacc_response(
        preset_basis(1),
        atomic_window=window,
        # q = 8 on a 40 fs window: 180 samples per 5 fs taper, 304 for the collar.
        sampling=TDSESampling(8),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        return Simulation.from_capillary(
            256e-6,
            _HACC_LENGTH,
            "He",
            0.1,
            wavelength=800e-9,
            wavelength_limits=(400e-9, 1.6e-6),
            trange=40e-15,
            fwhm=6e-15,
            energy=1e-12,
            modes=1,
            radial_rule="field_scale",
            n_r=2,
            noise=None,
            dz_init=2.0**-20,
            dz_max=2.0**-20,
            stats=StatsSpec(names=("energy",)),
            tdse=response,
            tdse_execution=TDSEExecution(max_atomic_calls=limit),
        )


def _quiet(simulation, path=None, **kwargs) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", TDSEDomainWarning)
        warnings.simplefilter("ignore", TDSEUnqualifiedModelWarning)
        return simulation.run(path, progress="none", **kwargs)


@pytest.mark.parametrize("window", ["auto", None], ids=["window", "collar"])
def test_hacc_run_round_trips_and_resumes(fake, tmp_path, window):
    path = tmp_path / "hacc.h5"
    simulation = _hacc(window, limit=40)
    simulation.settings = dataclasses.replace(simulation.settings, resume_history=False)
    with pytest.raises(ResponseBudgetError):
        _quiet(simulation, path)
    with open_file(path) as saved:
        assert saved.metadata is not None
        key = saved.metadata.configuration_hash()
        descriptor = json.loads(plain(saved.metadata.tdse)["configuration_json"])
    assert descriptor["schema"] == "solprop.tdse-descriptor.v5"
    assert descriptor["response"]["model"] == "hacc"
    assert ("atomic_window" in descriptor["lattice"]) is (window == "auto")
    simulation.settings = dataclasses.replace(
        simulation.settings, tdse_execution=TDSEExecution(max_atomic_calls=1000)
    )
    _quiet(simulation, path, resume=True)
    final = resume(path)
    assert final.outcome is not None and final.outcome.status == "complete"
    with open_file(path) as saved:
        assert saved.metadata is not None
        assert saved.metadata.configuration_hash() == key
    record = plain(final.outcome.execution)["tdse_source"]
    assert record["steady_current_certificate"] <= STEADY_CURRENT_CERTIFICATE_LIMIT
    with pytest.raises(OutputError):
        _quiet(_hacc(None if window == "auto" else "auto"), path, resume=True)
