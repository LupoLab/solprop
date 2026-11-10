# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Carry owned execution choices without changing scientific identity."""

import dataclasses
import inspect
import multiprocessing
import pickle
import sys

import pytest

from solprop.backends.session import ExecutionOptions
from solprop.driver import SimulationSettings
from solprop.exceptions import BackendError


def _settings(**keywords):
    return SimulationSettings(
        save_planes=[0.001],
        dz_max=0.0001,
        dz_init=0.00001,
        backend="numpy_ref",
        **keywords,
    )


@pytest.mark.parametrize("construct", [_settings, ExecutionOptions])
def test_backend_option_mappings_are_owned_and_defaults_are_independent(construct):
    first, second = construct(), construct()
    assert first.backend_options == second.backend_options == {}
    assert first.backend_options is not second.backend_options
    original = {"wide": "float64", "setup_id": "cpu-identity"}
    selected = construct(backend_options=original)
    original["wide"] = "df32"
    assert selected.backend_options == {"wide": "float64", "setup_id": "cpu-identity"}
    restored = pickle.loads(pickle.dumps(selected))
    assert restored.backend_options == selected.backend_options
    assert restored.backend_options is not selected.backend_options


def test_execution_options_snapshot_the_settings_mapping():
    settings = _settings(backend_options={"setup_id": "one"})
    options = settings.execution_options()
    assert options.backend_options == {"setup_id": "one"}
    assert options.backend_options is not settings.backend_options
    changed = dataclasses.replace(settings, backend_options={"setup_id": "two"})
    assert changed.execution_options().backend_options == {"setup_id": "two"}
    assert options.backend_options == {"setup_id": "one"}


def test_backend_options_check_all_keys_against_the_backend_declaration():
    options = ExecutionOptions(backend_options={"wide": "df32", "setup_id": "one"})
    options.check_backend_options("julia", ("wide", "setup_id"))
    with pytest.raises(BackendError, match=r"julia.*setup_id"):
        options.check_backend_options("julia", ("wide",))


def test_production_execution_options_refuse_non_string_keys():
    # Exercise the production constructor without the test session's earlier type
    # refusal, as the other public-boundary validation tests do.
    initialize = inspect.unwrap(ExecutionOptions.__init__)
    with pytest.raises(BackendError, match="keys must be strings"):
        initialize(object.__new__(ExecutionOptions), backend_options={1: "invalid"})


def test_backend_options_change_after_translation_without_changing_hashes(
    build_simulation,
):
    simulation = build_simulation(n_save=2)
    problem = simulation.translate()
    metadata = simulation.metadata()
    simulation.settings = dataclasses.replace(
        simulation.settings, backend_options={"wide": "df32", "setup_id": "chosen"}
    )
    assert simulation.translate() is problem
    assert simulation.metadata().configuration_hash() == metadata.configuration_hash()
    assert simulation.settings.execution_options().backend_options == {
        "wide": "df32",
        "setup_id": "chosen",
    }
    fresh = build_simulation(n_save=2)
    fresh.settings = dataclasses.replace(
        fresh.settings, backend_options={"setup_id": "different"}
    )
    assert fresh.metadata().configuration_hash() == metadata.configuration_hash()


def _spawn_options(connection, settings, options):
    connection.send(
        (
            dict(settings.backend_options),
            dict(settings.execution_options().backend_options),
            dict(options.backend_options),
            "juliacall" in sys.modules,
        )
    )
    connection.close()


def test_spawned_worker_receives_options_without_starting_julia():
    settings = _settings(backend_options={"setup_id": "chosen", "wide": "df32"})
    options = settings.execution_options()
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_spawn_options, args=(sender, settings, options))
    process.start()
    sender.close()
    try:
        assert receiver.poll(20), "the spawned option round trip did not return"
        first, second, third, imported = receiver.recv()
        assert first == second == third == settings.backend_options
        assert not imported
        process.join(timeout=20)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        receiver.close()
