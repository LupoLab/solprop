# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""A march snapshot's state is one trajectory's, on disk as at run time.

SEP 52 section 5.4: from output schema 3 the stored dataset is ``(n_modes, n_pol,
n_omega)``, as the runtime snapshot carries it. A snapshot written before, with the
leading axis of one it then kept, is refused by name rather than squeezed, and so is a
state of any other shape.
"""

import h5py
import numpy as np
import pytest

from solprop.exceptions import OutputError
from solprop.output.hdf5 import _read_march, _write_march
from solprop.output.metadata import SCHEMA_VERSION
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
        rows={"store": np.zeros((4, 1, 1, 4), complex)},
        history_key="adams",
    )


def test_a_snapshot_is_stored_and_read_back_in_the_trajectory_s_layout(tmp_path):
    state = np.arange(4, dtype=complex).reshape(1, 1, 4)
    with h5py.File(tmp_path / "run.h5", "w") as file:
        _write_march(file.require_group("march_snapshot"), snapshot(state))
        stored = file["march_snapshot/state"]
        assert isinstance(stored, h5py.Dataset)
        assert stored.shape == (1, 1, 4)
        rows = file["march_snapshot/rows/store"]
        assert isinstance(rows, h5py.Dataset)
        assert rows.shape == (4, 1, 1, 4)
        read = _read_march(file, 0.5)
    assert read is not None
    assert read.state.shape == (1, 1, 4)
    np.testing.assert_array_equal(read.state, state)


def _stored_as(path, stored):
    """Write a snapshot group whose state dataset is replaced by `stored`."""
    with h5py.File(path, "w") as file:
        _write_march(
            file.require_group("march_snapshot"), snapshot(np.zeros((1, 1, 4), complex))
        )
        group = file["march_snapshot"]
        assert isinstance(group, h5py.Group)
        del group["state"]
        group.create_dataset("state", data=stored)


def test_a_snapshot_with_the_earlier_leading_axis_is_refused_by_name(tmp_path):
    _stored_as(tmp_path / "old.h5", (1.0 + 2.0j) * np.ones((1, 2, 1, 4)))
    with (
        h5py.File(tmp_path / "old.h5", "r") as file,
        pytest.raises(OutputError, match=f"before output schema {SCHEMA_VERSION}"),
    ):
        _read_march(file, 0.5)


@pytest.mark.parametrize("shape", [(2, 1, 1, 4), (1, 4), (1, 1, 1, 1, 4)])
def test_a_state_of_another_shape_is_refused_rather_than_squeezed(tmp_path, shape):
    _stored_as(tmp_path / "rank.h5", np.ones(shape, complex))
    with (
        h5py.File(tmp_path / "rank.h5", "r") as file,
        pytest.raises(OutputError, match="one trajectory's"),
    ):
        _read_march(file, 0.5)
