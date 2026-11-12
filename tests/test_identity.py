# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Golden digests: the identities stamped in files keep their values.

A resume, a scan's common record, a coupling plane's basis and the TDSE caches compare
these digests with the ones files already hold, so a change of the canonical text
changes what an old file is accepted against. Each value below was computed before the
digests were brought onto one helper; a change to any of them is a deliberate break of
stored identities and needs a release note.
"""

import math

import numpy as np
import pytest

from solprop._identity import canonical_digest, canonical_json, text_digest
from solprop.coupling.plane import basis_identity, lattice_identity, space_record
from solprop.driver import tdse_consistency
from solprop.grid import EnvelopeGrid, RadialGrid
from solprop.nonlinear.tdse import TDSEBasis
from solprop.output.metadata import _example_metadata
from solprop.output.scan import _hash, _json


def test_the_canonical_text_sorts_keys_and_drops_whitespace():
    record = {"b": [1, 2.5, None], "a": {"z": True, "y": "text"}, "c": math.inf}
    assert canonical_json(record) == (
        '{"a":{"y":"text","z":true},"b":[1,2.5,null],"c":Infinity}'
    )
    assert canonical_digest(record) == text_digest(canonical_json(record))
    assert text_digest("") == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_a_finite_identity_refuses_a_non_finite_number():
    with pytest.raises(ValueError, match="Out of range float"):
        canonical_digest({"intensity": math.nan}, finite=True)


def test_an_identity_refuses_a_value_json_cannot_hold():
    # Never a str() of it: an array's text is abbreviated, and two would merge.
    with pytest.raises(TypeError, match="not JSON serializable"):
        canonical_digest({"values": np.zeros(3)})


def test_a_numpy_scalar_is_written_as_the_number_it_holds():
    # A comparison of arrays gives np.bool_, an index np.int64: records carry both.
    record = {"zero": np.float64(0.0) == 0.0, "count": np.int64(3)}
    assert canonical_json(record) == '{"count":3,"zero":true}'
    assert canonical_json({"half": np.float32(0.5)}) == '{"half":0.5}'
    with pytest.raises(TypeError, match="not JSON serializable"):
        canonical_json({"value": np.complex128(1.0)})


@pytest.mark.parametrize(
    ("name", "digest"),
    [
        (
            "metadata",
            "9e1f945c0ab5246cc011e9bb713242ac19d8fb4e76b58ba4e95f23af58fc6964",
        ),
        ("scan", "80f9c7ee2c67bb3246215ab42ce1d49fa0cb9f93e774cd9b2d543229b4ae335b"),
        ("basis", "d7b11decb56fec6a504280a7c9e26f83c5f33d84fa111aa9a9b27f9e87481587"),
        (
            "lattice",
            "1b956475d651c1aedd23cc23d6d6c51ab37b58c7173ff491c9d16c1772c1bc7b",
        ),
        (
            "tdse_basis",
            "f8e756518be0c594c63712a70b5d0d969ce2d7b075828c3b7a5ebf2fe35dd0af",
        ),
        (
            "consistency",
            "a8bf3655b53dc839e2cebdf17c4c950a0061eeef6e7e0e3a38d75858904949a1",
        ),
    ],
)
def test_a_stored_identity_keeps_its_value(name, digest):
    computed = {
        "metadata": lambda: _example_metadata().configuration_hash(),
        "scan": lambda: _hash(_json({"gamma": 0.11, "modes": [1, 2], "inf": math.inf})),
        "basis": lambda: basis_identity("radial", space_record(RadialGrid(1e-3, 64))),
        "lattice": lambda: lattice_identity(
            EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
        ),
        "tdse_basis": lambda: (
            TDSEBasis(
                "manufactured", (0.0, 1e-9), (8,), 3, 1e-16, source="golden"
            ).identity
        ),
        "consistency": lambda: tdse_consistency._key(
            {"species": "helium", "intensity": 1e18, "order": [1, 2]}
        ),
    }[name]()
    assert computed == digest
