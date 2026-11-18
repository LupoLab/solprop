# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pin the held-source identity and its admission."""

import dataclasses

import numpy as np
import pytest

from backends import test_numpy_ref_plasma as plasma_fixtures
from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver.metadata import metadata_for
from solprop.exceptions import TranslationError
from solprop.translate.plasma_source import plasma_source_record

field_grid = plasma_fixtures.field_grid
ionising_capillary = plasma_fixtures.ionising_capillary


def test_metadata_and_memory_name_the_source(ionising_capillary, field_grid):
    problem, run_request, _, _ = ionising_capillary()
    backend = NumpyRefBackend(problem, run_request)
    record = plasma_source_record(problem.grids)
    assert backend.metadata()["plasma_source"] == record
    assert metadata_for(problem, run_request, field_grid).configuration[
        "plasma_source"
    ] == {key: value for key, value in record.items() if key != "certificate"}
    ledger = backend.memory_ledger()
    assert ledger["held_current_streams"] > 0
    assert ledger["held_current_spectra"] > 0
    assert ledger["held_current_derivative"] == problem.grids.n_to * 8
    backend.close()


def test_certificate_refuses_zero_gate_and_nonpositive_active_frequency(
    ionising_capillary,
):
    problem, _, _, _ = ionising_capillary()
    masks = problem.grids
    empty = dataclasses.replace(masks, t_window_fine=np.zeros(masks.n_to))
    with pytest.raises(TranslationError, match="nonzero temporal collar"):
        plasma_source_record(empty)
    active = masks.active_mask.copy()
    active[0] = True
    # The standalone certificate validates the contract even when called before
    # PropagationProblem's general mask checks.
    broken = dataclasses.replace(masks, active_mask=active)
    with pytest.raises(TranslationError, match="positive active frequencies"):
        plasma_source_record(broken)


def test_validation_refuses_an_unresolved_collar(ionising_capillary):
    problem, _, _, _ = ionising_capillary()
    legacy = np.ones(problem.grids.n_to)
    legacy[0] = 0.0
    unresolved = dataclasses.replace(
        problem,
        grids=dataclasses.replace(problem.grids, t_window_fine=legacy),
    )
    with pytest.raises(TranslationError, match="collar certificate"):
        unresolved.validate()
