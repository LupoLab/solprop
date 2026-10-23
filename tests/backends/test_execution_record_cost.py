# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The execution record is read at every plane; its immutable facts are taken once."""

from solprop.backends.numba_cpu import fft as numba_fft
from solprop.backends.numba_cpu.backend import NumbaCpuBackend


def test_the_wisdom_digest_is_exported_once_per_backend(
    monkeypatch, soliton_problem, soliton_grid
):
    """Every plan is built at construction, so the digest is taken at the first record.

    Exporting FFTW's wisdom at every plane was most of the record's cost; the record
    the output keeps is unchanged.
    """
    calls = []
    digest = numba_fft.wisdom_digest

    def counted():
        calls.append(1)
        return digest()

    monkeypatch.setattr(numba_fft, "wisdom_digest", counted)
    problem = soliton_problem(soliton_grid, 1, save_planes=[0.01])
    backend = NumbaCpuBackend(
        problem, grid=soliton_grid, threads=1, plan_flag="estimate"
    )
    try:
        records = [backend.metadata() for _ in range(3)]
    finally:
        backend.close()
    assert len(calls) == 1
    assert records[0] == records[1] == records[2]
    assert records[0]["wisdom_digest"] == digest()
