# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin the held-source identity, admission and private historical comparison."""

import dataclasses

import numpy as np
import pytest
from tests.backends import test_numpy_ref_plasma as plasma_fixtures

from gated_source import _gated_source
from solprop.backends import registry
from solprop.backends.numba_cpu import NumbaCpuBackend
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver.simulation import _open_output, metadata_for
from solprop.exceptions import BackendError, DriverError, OutputError, TranslationError
from solprop.output import HDF5Output, check_cache, resume
from solprop.output.schema import SavedPlane
from solprop.translate.plasma_source import _BENCHMARK_GATED, plasma_source_record

field_grid = plasma_fixtures.field_grid
ionising_capillary = plasma_fixtures.ionising_capillary


def test_benchmark_identity_and_both_resume_doors(
    ionising_capillary, field_grid, tmp_path
):
    problem, _, _ = ionising_capillary()
    production = metadata_for(problem, field_grid)
    with _gated_source():
        historical = metadata_for(problem, field_grid)
        backend = NumpyRefBackend(problem)
        source_metadata = backend.metadata()["plasma_source"]
        assert isinstance(source_metadata, dict)
        assert source_metadata["form"] == "gated_benchmark"
        ledger = backend.memory_ledger()
        assert ledger["held_current_derivative"] == problem.grids.n_to * 8
        assert all(
            size == 0
            for name, size in ledger.items()
            if name.startswith("held_current_") and name != "held_current_derivative"
        )
        backend.close()
    assert not _BENCHMARK_GATED.get()
    assert historical.configuration_hash() != production.configuration_hash()
    assert historical.configuration["_benchmark_plasma_source"] == "gated_nonresumable"
    path = tmp_path / "gated.h5"
    with HDF5Output(path) as output:
        output.open(historical)
        output.save(SavedPlane(z=0.0, field=problem.initial.field0[0]))
    with pytest.raises(OutputError, match="gated plasma benchmark"):
        check_cache(path, historical)
    with pytest.raises(OutputError, match="gated plasma benchmark"):
        check_cache(path, production)
    with pytest.raises(OutputError, match="gated plasma benchmark"):
        resume(path)
    with pytest.raises(DriverError, match="gated plasma benchmark"):
        _open_output(path, historical, resume=True)


def test_gated_hook_refuses_compiled_execution_and_resets_after_exception(
    ionising_capillary,
):
    problem, _, _ = ionising_capillary()
    with pytest.raises(BackendError, match="requires numpy_ref"), _gated_source():
        NumbaCpuBackend(problem)
    assert not _BENCHMARK_GATED.get()
    with pytest.raises(BackendError, match="requires numpy_ref"), _gated_source():
        registry.select("numba_cpu", problem)
    assert not _BENCHMARK_GATED.get()


def test_metadata_and_memory_name_the_source(ionising_capillary, field_grid):
    problem, _, _ = ionising_capillary()
    backend = NumpyRefBackend(problem)
    record = plasma_source_record(problem.grids)
    assert backend.metadata()["plasma_source"] == record
    assert metadata_for(problem, field_grid).configuration["plasma_source"] == {
        key: value for key, value in record.items() if key != "certificate"
    }
    ledger = backend.memory_ledger()
    assert ledger["held_current_streams"] > 0
    assert ledger["held_current_spectra"] > 0
    assert ledger["held_current_derivative"] == problem.grids.n_to * 8
    backend.close()


def test_certificate_refuses_zero_gate_and_nonpositive_active_frequency(
    ionising_capillary,
):
    problem, _, _ = ionising_capillary()
    masks = problem.grids
    empty = dataclasses.replace(masks, t_window_fine=np.zeros(masks.n_to))
    with pytest.raises(TranslationError, match="nonzero temporal collar"):
        plasma_source_record(empty)
    active = masks.active_mask.copy()
    active[0] = True
    # The standalone certificate validates the contract even when called before
    # KernelProblem's general mask checks.
    broken = dataclasses.replace(masks, active_mask=active)
    with pytest.raises(TranslationError, match="positive active frequencies"):
        plasma_source_record(broken)


def test_private_historical_collar_cannot_bypass_production_admission(
    ionising_capillary, field_grid
):
    problem, _, _ = ionising_capillary()
    legacy = np.ones(problem.grids.n_to)
    legacy[0] = 0.0
    historical = dataclasses.replace(
        problem,
        grids=dataclasses.replace(problem.grids, t_window_fine=legacy),
    )
    with pytest.raises(TranslationError, match="collar certificate"):
        historical.validate()
    with _gated_source():
        historical.validate()
        metadata = metadata_for(historical, field_grid)
        assert metadata.configuration["plasma_source"] == {
            "form": "gated_benchmark",
            "resumable": False,
        }
        with pytest.raises(BackendError, match="requires numpy_ref"):
            NumbaCpuBackend(historical)
    with pytest.raises(TranslationError, match="collar certificate"):
        historical.validate()
