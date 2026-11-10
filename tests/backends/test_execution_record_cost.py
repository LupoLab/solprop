# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The execution record is read at every plane; its immutable facts are taken once."""

import pytest


def test_the_wisdom_digest_is_taken_once_per_backend(
    monkeypatch, soliton_problem, soliton_grid
):
    """Every plan is built at construction, so the digest is taken there, once.

    Exporting FFTW's wisdom at every plane was most of the record's cost; the record
    the output keeps is unchanged.
    """
    native = pytest.importorskip("solprop_native")
    wisdom = pytest.importorskip("solprop_native._wisdom")
    calls = []
    digest = wisdom.digest

    def counted(applied):
        calls.append(1)
        return digest(applied)

    monkeypatch.setattr(wisdom, "digest", counted)
    problem, run_request = soliton_problem(soliton_grid, 1, save_planes=[0.01])
    backend = native.NativeBackend(problem, run_request, grid=soliton_grid)
    try:
        records = [backend.metadata() for _ in range(3)]
    finally:
        backend.close()
    assert len(calls) <= 1  # none on a device build
    assert records[0] == records[1] == records[2]
