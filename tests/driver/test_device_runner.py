# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Rebuild cached execution owners when a scan changes its execution choices."""

import dataclasses

import numpy as np
import pytest

from solprop.backends.numpy_ref import NumpyRefBackend
from solprop.driver import simulation as driver
from solprop.exceptions import DriverError


def _reference_owner(_name, problem, request, **keywords):
    # A real reference owner supplies lifecycle; only device selection is scripted.
    # These tests do not claim a GPU execution or advance a numerical state.
    return NumpyRefBackend(
        problem,
        request,
        grid=keywords["grid"],
        modes=keywords["modes"],
        profiles=keywords["profiles"],
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"device": "cpu"},
        {"backend_options": {"setup_id": "new", "wide": "df32"}},
        {"backend_options": {"setup_id": "one", "wide": "binary64"}},
        {"backend": "numpy_ref"},
        {"node_budget": 2**20},
    ],
)
def test_runner_refuses_changed_execution_before_consuming_input(
    changes, build_simulation, monkeypatch
):
    simulation = build_simulation(n_save=2)
    simulation.settings = dataclasses.replace(
        simulation.settings,
        backend="julia",
        device="cuda",
        backend_options={"setup_id": "one", "wide": "df32"},
    )
    monkeypatch.setattr(driver, "select", _reference_owner)
    with simulation.runner() as runner:
        assert runner._takes(simulation)
        field = simulation._field0.copy()
        saved = field.copy()
        simulation.settings = dataclasses.replace(simulation.settings, **changes)
        assert not runner._takes(simulation)

        def forbidden(*_args, **_kwargs):
            pytest.fail("changed execution choices reached input preparation")

        monkeypatch.setattr(simulation, "with_input", forbidden)
        with pytest.raises(DriverError, match="different execution settings"):
            runner.run(field, progress="none")
        np.testing.assert_array_equal(field, saved)
        assert runner.runs == 0
        assert not runner.closed


def test_runner_keeps_its_own_mapping_when_origin_options_mutate(
    build_simulation, monkeypatch
):
    simulation = build_simulation(n_save=2)
    simulation.settings = dataclasses.replace(
        simulation.settings, backend_options={"setup_id": "one"}
    )
    monkeypatch.setattr(driver, "select", _reference_owner)
    with simulation.runner() as runner:
        options = simulation.settings.backend_options
        assert isinstance(options, dict)
        options["setup_id"] = "changed"
        assert runner._execution_options.backend_options == {"setup_id": "one"}
        assert not runner._takes(simulation)


def test_runner_resolves_auto_and_native_to_the_same_selection(
    build_simulation, monkeypatch
):
    simulation = build_simulation(n_save=2)
    monkeypatch.setattr(driver, "select", _reference_owner)
    monkeypatch.setattr(
        driver, "resolve_name", lambda name: "native" if name == "auto" else name
    )
    simulation.settings = dataclasses.replace(simulation.settings, backend="auto")
    with simulation.runner() as runner:
        simulation.settings = dataclasses.replace(simulation.settings, backend="native")
        assert runner._takes(simulation)
