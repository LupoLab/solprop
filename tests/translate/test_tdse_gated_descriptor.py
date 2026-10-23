# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check the native TDSE descriptor, gated and edge-free (SEP 26), for every gate.

Every descriptor has the one ``solprop.tdse-descriptor.v5`` layout; a recorded source
filter and the earlier versions are refused. Identities are exact (class P): a
descriptor is canonical JSON and its SHA-256.
"""

import dataclasses
import hashlib
import json

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from tests.hacc_cases import hacc_response, preset_basis

from solprop.exceptions import MaterialError, TranslationError
from solprop.grid import FieldGrid, planck_taper_derivative
from solprop.nonlinear.tdse import (
    TDSEAtomicWindow,
    TDSEBasis,
    TDSECalibration,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.translate.tdse import (
    TDSEDescriptor,
    _response_from_json,
    lattice_times,
    refine_response_grid,
)

WINDOW = TDSEAtomicWindow(40e-15, 5e-15)
"""600 samples per taper on the refined lattice."""


@pytest.fixture(scope="module")
def grid():
    parent = FieldGrid.from_limits(0.01, 800e-9, (120e-9, 1.6e-6), 60e-15)
    # q = 8: 8.3e-18 s spacing, a 498-sample collar taper.
    return refine_response_grid(parent, TDSESampling(8))


def _sae(**changes):
    basis = TDSEBasis(
        "manufactured", (0.0, 1e-9), (8,), 5, 1e-16, source="v4 contract fixture"
    )
    calibration = TDSECalibration(
        "v4-fixture",
        "research",
        basis.identity,
        (400e-9, 2e-6),
        source="v4 contract fixture",
        limitations=("unqualified",),
    )
    values = {
        "atomic_window": None,
        "sampling": TDSESampling(8),
        **changes,
    }
    return TDSEGasResponse(
        "helium",
        basis,
        1e-19,
        "tdse_native",
        1.537,
        calibration,
        **values,  # pyright: ignore[reportArgumentType] - test keywords
    )


def _kinds():
    return {
        "sae_collar": _sae(),
        "sae_window": _sae(atomic_window=WINDOW),
        "hacc_window": hacc_response(
            preset_basis(1),
            atomic_window=WINDOW,
            sampling=TDSESampling(8),
        ),
        "hacc_collar": hacc_response(
            preset_basis(1),
            atomic_window=None,
            sampling=TDSESampling(8),
        ),
    }


@pytest.mark.parametrize("kind", list(_kinds()))
def test_a_gated_descriptor_round_trips_each_kind(grid, kind):
    declared = _kinds()[kind]
    descriptor = TDSEDescriptor.from_response(declared, grid)
    record = json.loads(descriptor.configuration_json)
    assert record["schema"] == "solprop.tdse-descriptor.v5"
    assert "source_filter" not in record["response"]
    assert record["response"]["model"] == declared.model
    assert "atomic_window" in record["response"]
    assert descriptor.gated
    # Exact round trips (class P): the record and the scalar fields reconstruct it.
    assert _response_from_json(descriptor.configuration_json) == declared
    assert TDSEDescriptor(**dataclasses.asdict(descriptor)) == descriptor
    assert (
        descriptor.identity
        == hashlib.sha256(descriptor.configuration_json.encode()).hexdigest()
    )
    windowed = kind.endswith("window")
    assert (descriptor.atomic_window is not None) is windowed
    assert (record["response"]["atomic_window"] is None) is not windowed
    if declared.model == "hacc":
        assert record["response"]["basis"]["kind"] == "hacc_helium"
    else:
        assert "kind" not in record["response"]["basis"]
    breakpoints = descriptor.source_breakpoints()
    expected = (
        descriptor.atomic_window.gate
        if descriptor.atomic_window is not None
        else tuple(
            record["lattice"]["time_collar"][name]
            for name in ("zero_low", "one_low", "one_high", "zero_high")
        )
    )
    assert breakpoints == tuple(expected)
    derivative = descriptor.source_gate_derivative()
    assert not derivative.flags.writeable
    assert_array_equal(
        derivative,
        planck_taper_derivative(
            lattice_times(descriptor.samples, descriptor.spacing), *breakpoints
        ),
    )
    error = descriptor.source_gate_derivative_error()
    assert not error.flags.writeable and np.all(error >= 0)


@pytest.mark.parametrize(
    "mutation",
    [
        "v4_schema",
        "v3_schema",
        "v2_schema",
        "v1_schema",
        "time_collar",
        "gated_derivative",
        "missing_model",
        "missing_window",
        "bad_collar",
    ],
)
def test_a_gated_descriptor_refuses_every_other_combination(grid, mutation):
    descriptor = TDSEDescriptor.from_response(_kinds()["sae_window"], grid)
    record = json.loads(descriptor.configuration_json)
    fields = {}
    if mutation.endswith("_schema"):
        record["schema"] = f"solprop.tdse-descriptor.{mutation.split('_')[0]}"
    elif mutation in ("time_collar", "gated_derivative"):
        # No filter is recorded any more: refused when the response is revalidated.
        record["response"]["source_filter"] = mutation
    elif mutation == "missing_model":
        del record["response"]["model"]
    elif mutation == "missing_window":
        del record["response"]["atomic_window"]
    else:
        collar = record["lattice"]["time_collar"]
        collar["one_high"] = collar["zero_high"]
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with pytest.raises(TranslationError, match="invalid"):
        dataclasses.replace(
            descriptor,
            configuration_json=encoded,
            identity=hashlib.sha256(encoded.encode()).hexdigest(),
            **fields,
        )


def test_a_descriptor_imposes_no_taper_sample_count(grid):
    # The certificate, computed when the problem is validated, is the whole admission
    # (tests/backends/test_tdse_edge_free.py, T7); a descriptor builds for any taper
    # and rebuilds from its own record.
    narrow = TDSEAtomicWindow(40e-15, 4 * grid.dt_fine)
    descriptor = TDSEDescriptor.from_response(_sae(atomic_window=narrow), grid)
    assert dataclasses.replace(descriptor) == descriptor


def test_a_native_lattice_must_be_the_standard_one(grid):
    shifted = dataclasses.replace(grid, t_fine=grid.t_fine + 1e-21, t=grid.t + 1e-21)
    with pytest.raises(TranslationError, match="standard fine response lattice"):
        TDSEDescriptor.from_response(_sae(), shifted)


# --- the response record: a window needs the native closure ---------------------------


@pytest.mark.parametrize("kerr_model", ["optmat_cubic", "tdse_shadow"])
def test_a_window_needs_the_native_closure(kerr_model):
    with pytest.raises(MaterialError, match="requires kerr_model='tdse_native'"):
        dataclasses.replace(
            _sae(atomic_window=WINDOW),
            kerr_model=kerr_model,
            cubic_scale=1.0 if kerr_model == "tdse_shadow" else None,
        )
