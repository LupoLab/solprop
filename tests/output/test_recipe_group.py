# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The recipe group of an output file round-trips its texts and data files (SEP 41)."""

import h5py

from solprop.output.hdf5 import read_recipe, write_recipe
from solprop.output.schema import RecipeFile, RecipeRecord


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
