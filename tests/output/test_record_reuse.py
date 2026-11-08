# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run record is serialised once, however often its hashed part is asked for."""

import logging

from solprop.output import MemoryOutput, schema
from solprop.output.schema import _example_metadata


def test_the_hashed_part_is_serialised_once_and_handed_out_as_copies(monkeypatch):
    metadata = _example_metadata()
    calls = []
    serialise = schema.serialise

    def counted(value):
        calls.append(1)
        return serialise(value)

    monkeypatch.setattr(schema, "serialise", counted)
    first = metadata.hashed()
    after_first = len(calls)
    second = metadata.hashed()
    key = metadata.configuration_hash()
    assert len(calls) == after_first > 0
    assert first == second
    assert first is not second
    # A caller that extends its copy, as to_json does, leaves the record unchanged.
    first["extra"] = 1
    assert "extra" not in metadata.hashed()
    assert metadata.configuration_hash() == key


def test_a_memory_output_computes_no_hash_unless_it_logs_it(caplog):
    # The record caches its digest and serialised part, so what is cached is what ran.
    quiet = _example_metadata()
    MemoryOutput().open(quiet)
    assert "_configuration_digest" not in vars(quiet)
    assert "_hashed_record" not in vars(quiet)

    logged = _example_metadata()
    with caplog.at_level(logging.DEBUG, logger="solprop.output.memory"):
        MemoryOutput().open(logged)
    assert logged.configuration_hash()[:12] in caplog.text


def test_a_run_s_execution_facts_keep_its_resume_key(monkeypatch):
    # Opening a run adds the backend record, which is outside the hash: the key and
    # the hashed part already computed are reused, not serialised again.
    metadata = _example_metadata()
    key = metadata.configuration_hash()
    calls = []
    serialise = schema.serialise

    def counted(value):
        calls.append(1)
        return serialise(value)

    monkeypatch.setattr(schema, "serialise", counted)
    backend = schema.BackendRecord(name="native", propagator_mode="tabulate")
    executed = metadata.with_backend(backend)
    assert executed.configuration_hash() == key
    assert executed.hashed() == metadata.hashed()
    assert not calls
    # The stored document is the new record's own, backend included.
    assert '"native"' in executed.to_json()
    monkeypatch.undo()
    # A record whose key was never asked for computes it afresh, to the same value.
    fresh = _example_metadata().with_backend(backend)
    assert "_configuration_digest" not in vars(fresh)
    assert fresh.configuration_hash() == key
