# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A run record is serialised once, however often its hashed part is asked for."""

from solprop.output import schema
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
