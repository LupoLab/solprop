# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A march snapshot's state is one trajectory's; its file keeps the leading axis of one.

SEP 47 section 6.7: the runtime snapshot carries ``(n_modes, n_pol, n_omega)``, and the
stored dataset keeps the ``(1, n_modes, n_pol, n_omega)`` layout every file has been
written with, so old files stay readable and a stacked state is never squeezed.
"""

import logging

import h5py
import numpy as np
import pytest

from solprop.output.hdf5 import _read_march, _write_march
from solprop.output.records import MarchSnapshot


def snapshot(state):
    """A snapshot at 0.5 m holding `state`."""
    return MarchSnapshot(
        z=0.5,
        state=state,
        proposal=0.1,
        method="Multistep()",
        order=2,
        rejections=0,
        last_rejected=False,
        positions=np.array([0.4, 0.5]),
        identities=("accepted", "predicted"),
        slots=np.array([0, 1]),
        rows={"store": np.zeros((4, 1, 1, 1, 4), complex)},
        history_key="adams",
    )


def test_a_snapshot_is_stored_with_its_leading_axis_and_read_back_without(tmp_path):
    state = np.arange(4, dtype=complex).reshape(1, 1, 4)
    with h5py.File(tmp_path / "run.h5", "w") as file:
        _write_march(file.require_group("march_snapshot"), snapshot(state))
        stored = file["march_snapshot/state"]
        assert isinstance(stored, h5py.Dataset)
        assert stored.shape == (1, 1, 1, 4)
        read = _read_march(file, 0.5)
    assert read is not None
    assert read.state.shape == (1, 1, 4)
    np.testing.assert_array_equal(read.state, state)


def _old_file(path, stored):
    """Write a snapshot group as the files before SEP 47 hold it, with `stored`."""
    with h5py.File(path, "w") as file:
        _write_march(
            file.require_group("march_snapshot"), snapshot(np.zeros((1, 1, 4), complex))
        )
        group = file["march_snapshot"]
        assert isinstance(group, h5py.Group)
        del group["state"]
        group.create_dataset("state", data=stored)


def test_an_existing_file_s_four_axis_state_stays_readable(tmp_path):
    stored = (1.0 + 2.0j) * np.ones((1, 2, 1, 4))
    _old_file(tmp_path / "old.h5", stored)
    with h5py.File(tmp_path / "old.h5", "r") as file:
        read = _read_march(file, 0.5)
    assert read is not None
    np.testing.assert_array_equal(read.state, stored[0])


def test_a_stacked_state_is_refused_rather_than_squeezed(tmp_path, caplog):
    _old_file(tmp_path / "stacked.h5", np.ones((2, 1, 1, 4), complex))
    with (
        caplog.at_level(logging.WARNING, logger="solprop.output.hdf5"),
        h5py.File(tmp_path / "stacked.h5", "r") as file,
    ):
        assert _read_march(file, 0.5) is None
    assert any("one trajectory's" in record.message for record in caplog.records)


@pytest.mark.parametrize("shape", [(1, 4), (1, 1, 1, 1, 4)])
def test_a_state_of_another_rank_is_refused(tmp_path, shape):
    _old_file(tmp_path / "rank.h5", np.ones(shape, complex))
    with h5py.File(tmp_path / "rank.h5", "r") as file:
        assert _read_march(file, 0.5) is None


def test_a_state_stored_without_the_axis_is_read_as_it_is(tmp_path):
    stored = np.ones((3, 1, 4), complex)
    _old_file(tmp_path / "bare.h5", stored)
    with h5py.File(tmp_path / "bare.h5", "r") as file:
        read = _read_march(file, 0.5)
    assert read is not None
    np.testing.assert_array_equal(read.state, stored)
