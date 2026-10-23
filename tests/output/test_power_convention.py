# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve schema-5 evidence and label schema-6 true-power spectra."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from solprop.exceptions import GridError, OutputError
from solprop.fourier import power_amplitude
from solprop.output import HDF5Output
from solprop.output.hdf5 import check_cache, resume
from solprop.output.schema import SavedPlane

FIXTURE = Path(__file__).parents[1] / "data" / "power-convention-schema5.h5"
LEGACY_HASH = "f407d314cb897acfd2a82216626013672ffca1b11dc42de2e1c008deeedd5c4c"


def test_schema5_fixture_stays_readable_with_its_original_digest_and_label():
    stored = resume(FIXTURE)
    assert stored.metadata.schema_version == 5
    assert stored.metadata.power_convention == "vacuum"
    assert "power_convention" not in stored.metadata.hashed()
    assert stored.metadata.configuration_hash() == LEGACY_HASH
    assert stored.plane.field is not None
    assert np.array_equal(stored.plane.field, np.ones_like(stored.plane.field))


def test_schema5_resume_is_refused_by_its_resume_key():
    stored = resume(FIXTURE)
    with pytest.raises(OutputError, match=r"different run.*schema_version"):
        check_cache(FIXTURE, replace(stored.metadata, schema_version=6))


def test_schema6_hash_and_field_attribute_record_the_power_convention(tmp_path):
    stored = resume(FIXTURE)
    metadata = replace(stored.metadata, schema_version=6)
    assert metadata.hashed()["power_convention"] == "poynting"
    assert metadata.configuration_hash() != LEGACY_HASH
    path = tmp_path / "power.h5"
    output = HDF5Output(path)
    output.open(metadata)
    output.save(SavedPlane(z=0.0, field=stored.plane.field))
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
