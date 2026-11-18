# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The recipe group of an output file round-trips its texts and data files (SEP 41)."""

import hashlib

import h5py
import numpy as np

from solprop.output.hdf5 import read_recipe, write_recipe
from solprop.output.records import RecipeFile, RecipeRecord


def test_a_recipe_and_its_files_read_back_as_written(tmp_path):
    record = RecipeRecord(
        1,
        'route = "gnlse"\n',
        'route = "gnlse"\nseed = "7"\n',
        {
            "spectrum-0.npy": RecipeFile(
                "spectrum.npy", "ab" * 32, b"\x93NUMPY\x00\x01"
            ),
            # Over the embedding budget: named and hashed, not stored.
            "large-1.npy": RecipeFile("large.npy", "cd" * 32, None),
        },
    )
    with h5py.File(tmp_path / "run.h5", "w") as file:
        assert read_recipe(file) is None
        write_recipe(file, record)
    with h5py.File(tmp_path / "run.h5", "r") as file:
        assert read_recipe(file) == record


def _dataset(group: h5py.Group, name: str) -> h5py.Dataset:
    found = group[name]
    assert isinstance(found, h5py.Dataset)
    return found


def test_the_recipe_group_has_its_documented_layout_and_digests(tmp_path):
    # Read with h5py alone, so the layout of docs/guide/results/output.md is pinned
    # independently of solprop's own reader.
    authored, resolved = 'route = "gnlse"\n', 'route = "gnlse"\nseed = "7"\n'
    data = b"\x93NUMPY\x00\x01"
    record = RecipeRecord(
        1,
        authored,
        resolved,
        {"spectrum-0.npy": RecipeFile("spectrum.npy", "ab" * 32, data)},
    )
    with h5py.File(tmp_path / "run.h5", "w") as file:
        write_recipe(file, record)
    with h5py.File(tmp_path / "run.h5", "r") as file:
        group = file["recipe"]
        assert isinstance(group, h5py.Group)
        assert set(group) == {"authored", "resolved", "files"}
        texts = {
            name: str(_dataset(group, name).asstr()[()])
            for name in ("authored", "resolved")
        }
        assert texts == {"authored": authored, "resolved": resolved}
        assert group.attrs["format"] == 1
        for name, text in texts.items():
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            assert group.attrs[f"{name}_sha256"] == digest
        entry = _dataset(group, "files/spectrum-0.npy")
        assert np.asarray(entry[()], dtype=np.uint8).tobytes() == data
        assert (entry.attrs["path"], entry.attrs["sha256"]) == (
            "spectrum.npy",
            "ab" * 32,
        )
