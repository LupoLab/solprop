# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Check SEP 26's SAE research record against the SEP 12 record it moves from.

SEP 26 section 7.5 and Q2: the SAE research record moves to ``gated_derivative``
under a new identity, and the recorded SEP 12 record is kept as recorded, never
relabelled. The SEP 12 record names the removed ``time_collar`` filter, so it is now
evidence only: it is checked by its hash and compared as a plain record, and solprop
no longer builds a response from it. SEP 26's record names ``"gated_derivative"``,
which is what every native response now is, so it is read without that field.
"""

import hashlib
import json
from pathlib import Path

import pytest

from solprop.grid import FieldGrid
from solprop.nonlinear.tdse import (
    TDSEBasis,
    TDSECalibration,
    TDSEGasResponse,
    TDSESampling,
)
from solprop.translate.tdse import TDSEDescriptor, refine_response_grid
from solprop.translate.tdse_source import (
    STEADY_CURRENT_CERTIFICATE_LIMIT,
    steady_current_certificate,
)

_ROOT = Path(__file__).resolve().parent / "data" / "tdse"
_SEP12 = _ROOT / "sep12-selected-filtered.response.json"
_SEP26 = _ROOT / "sep26-selected-gated-derivative.response.json"
# The SEP 12 record's hash as recorded in SEP 20's evidence manifest.
_SEP12_SHA256 = "f511ea3b4d42a3e004c6f46727b9f915da410653835bef75f05dba76ac201d98"


def _record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _response(path: Path) -> TDSEGasResponse:
    """Reconstruct a record with the public configuration classes.

    The record keeps the ``source_filter`` it was saved with; ``"gated_derivative"`` is
    what the native closure always does, so the reconstruction drops it.
    """
    record = _record(path)
    assert record.pop("source_filter") == "gated_derivative"
    basis_record = record.pop("basis")
    for key in ("radial_edges", "radial_orders", "cuts"):
        basis_record[key] = tuple(basis_record[key])
    calibration_record = record.pop("calibration")
    for key in ("wavelength_range", "limitations"):
        calibration_record[key] = tuple(calibration_record[key])
    sampling_record = record.pop("sampling")
    return TDSEGasResponse(
        basis=TDSEBasis(**basis_record),
        calibration=TDSECalibration(**calibration_record),
        sampling=TDSESampling(**sampling_record),
        **record,
    )


def test_the_sep12_record_is_kept_as_recorded():
    assert hashlib.sha256(_SEP12.read_bytes()).hexdigest() == _SEP12_SHA256
    old = _record(_SEP12)
    assert old["source_filter"] == "time_collar"
    assert old["calibration"]["identity"] == (
        "sep12-native1537-cut400-q32-time-collar-research-v1"
    )


def test_the_moved_record_differs_only_in_its_filter_and_calibration_text():
    old, new = _record(_SEP12), _record(_SEP26)
    assert _response(_SEP26).kerr_model == "tdse_native"
    assert new["calibration"]["identity"] == (
        "sep26-native1537-cut400-q32-gated-derivative-research-v1"
    )
    # Every scientific field but the filter is the SEP 12 record's, exactly.
    text = ("identity", "source", "limitations")
    assert {k: v for k, v in new["calibration"].items() if k not in text} == {
        k: v for k, v in old["calibration"].items() if k not in text
    }
    rest = ("source_filter", "calibration")
    assert {k: v for k, v in new.items() if k not in rest} == {
        k: v for k, v in old.items() if k not in rest
    }
    assert set(new) == set(old)


@pytest.fixture(scope="module")
def parent_grid() -> FieldGrid:
    # SEP 12's requested 60 fs, 120 nm-1.6 um parent; its collar taper spans 62
    # samples unrefined, about 2000 at q = 32.
    return FieldGrid.from_limits(0.01, 800e-9, (120e-9, 1.6e-6), 60e-15)


def test_the_moved_record_is_admitted_and_certified_on_its_parent(parent_grid):
    new = _response(_SEP26)
    grid = refine_response_grid(parent_grid, new.sampling)
    descriptor = TDSEDescriptor.from_response(new, grid)
    assert '"schema":"solprop.tdse-descriptor.v5"' in descriptor.configuration_json
    kappa = steady_current_certificate(descriptor, grid.omega, grid.active_mask)
    assert kappa is not None and kappa <= STEADY_CURRENT_CERTIFICATE_LIMIT
