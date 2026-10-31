# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Label saved spectra as true-power amplitudes, and convert to and from them."""

import h5py
import numpy as np
import pytest

from solprop.exceptions import GridError
from solprop.fourier import power_amplitude
from solprop.output import HDF5Output
from solprop.output.hdf5 import check_cache, resume
from solprop.output.schema import SavedPlane, _example_metadata


def test_the_hash_and_field_attribute_record_the_power_convention(tmp_path):
    metadata = _example_metadata()
    assert metadata.power_convention == "poynting"
    assert metadata.hashed()["power_convention"] == "poynting"
    path = tmp_path / "power.h5"
    output = HDF5Output(path)
    output.open(metadata)
    output.save(
        SavedPlane(z=0.0, field=np.ones((1, 1, metadata.grid.n_omega), complex))
    )
    output.close()
    with h5py.File(path) as file:
        assert file["field"].attrs["power_convention"] == "poynting"
    assert check_cache(path, metadata)
    assert resume(path).metadata.configuration_hash() == metadata.configuration_hash()


def test_power_amplitude_roundtrip_and_exact_unit_override():
    rng = np.random.default_rng(42)
    field = rng.normal(size=(2, 1, 17)) + 1j * rng.normal(size=(2, 1, 17))
    weight = rng.uniform(1, 2, size=field.shape)
    np.testing.assert_allclose(
        power_amplitude(power_amplitude(field, weight), weight, inverse=True),
        field,
        rtol=3 * np.finfo(float).eps,
        atol=0,
    )  # One multiplication and one division, with the same rounded square root.
    assert np.array_equal(power_amplitude(field, np.ones_like(weight)), field)


@pytest.mark.parametrize("weight", [0.0, -1.0, np.nan, np.inf])
def test_power_amplitude_refuses_invalid_weights(weight):
    with pytest.raises(GridError, match="positive and finite"):
        power_amplitude(np.ones((1, 1, 2), complex), np.full((1, 1, 2), weight))
