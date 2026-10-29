# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Preserve the identities of independently developed historical output layouts."""

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from solprop.exceptions import OutputError
from solprop.output import HDF5Output, SavedPlane, open_file
from solprop.output.schema import RunMetadata, _example_metadata


@pytest.mark.parametrize("lineage", ["sep12-10", "sep14-10", "sep14-11", "combined-12"])
def test_independent_schema_lineages_preserve_hashes_and_field_presence(
    tmp_path, lineage
):
    # Construct the two historical layouts independently of the production reader.
    record = json.loads(_example_metadata().to_json())
    record["schema_version"] = int(lineage.rsplit("-", 1)[1])
    # Schema 17's quadrature record did not exist in these layouts.
    record.pop("quadrature")
    # Nor did schema 24's plane-reference record.
    record.pop("reference")
    # Nor schema 26's noise resolution (SEP 37).
    record.pop("noise_resolution")
    limits = record["stepper"]["limits"]
    if lineage == "sep12-10":
        for name in ("interval_cap", "auto_init", "interpolate_planes", "step_planes"):
            limits.pop(name)
        record["initial_z"] = 0.125
        record["tdse"] = {"identity": "historical native", "scale": 1.47}
    elif lineage.startswith("sep14"):
        record.pop("initial_z")
        record.pop("tdse")
        if lineage == "sep14-10":
            limits.pop("interpolate_planes")
            limits.pop("step_planes")
    scientific = {
        key: value
        for key, value in record.items()
        if key
        not in (
            "backend",
            "versions",
            "git",
            "created",
            "spectral_convention",
            "field_layout",
        )
    }
    expected = hashlib.sha256(
        json.dumps(
            scientific, sort_keys=True, separators=(",", ":"), allow_nan=True
        ).encode()
    ).hexdigest()
    loaded = RunMetadata.from_json(json.dumps(record))
    assert loaded.configuration_hash() == expected
    assert json.loads(loaded.to_json()) == record
    # Editing scientific data must recompute identity, not reuse a cached digest.
    assert (
        dataclasses.replace(loaded, configuration={"gamma": 0.12}).configuration_hash()
        != expected
    )
    path = tmp_path / "historical.h5"
    with HDF5Output(path) as output:
        output.open(loaded)
        output.save(
            SavedPlane(
                z=loaded.initial_z, field=np.zeros((1, 1, loaded.grid.n_omega), complex)
            )
        )
    with open_file(path) as output:
        assert output.metadata is not None
        assert output.metadata.configuration_hash() == expected


def test_legacy_layout_cannot_hide_new_scientific_settings():
    record = json.loads(
        dataclasses.replace(_example_metadata(), schema_version=10).to_json()
    )
    loaded = RunMetadata.from_json(json.dumps(record))
    with pytest.raises(OutputError, match="cannot record launch"):
        dataclasses.replace(loaded, initial_z=0.1)
    with pytest.raises(OutputError, match="cannot record changed step limit"):
        dataclasses.replace(
            loaded,
            stepper=dataclasses.replace(
                loaded.stepper,
                limits=dataclasses.replace(
                    loaded.stepper.limits, interpolate_planes=True
                ),
            ),
        )


# --- SEP 20: output schema 18 (section 7.1 item 18) ---------------------------------


def _tdse_metadata(response):
    import warnings  # noqa: PLC0415 - local to the SEP 20 lineage cases

    from solprop.driver import Simulation  # noqa: PLC0415
    from solprop.exceptions import MaterialRangeWarning  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MaterialRangeWarning)
        simulation = Simulation.from_capillary(
            256e-6,
            1e-5,
            "He",
            0.1,
            lambda0=800e-9,
            wavelength_limits=(400e-9, 1.6e-6),
            trange=40e-15,
            fwhm=6e-15,
            energy=1e-12,
            n_r=2,
            radial_rule="field_scale",
            noise=None,
            tdse=response,
        )
    return simulation.metadata()


def _consistency_record():
    from tests.hacc_cases import alpha_law, preset_basis  # noqa: PLC0415

    from solprop.driver.tdse_consistency import (  # noqa: PLC0415
        evaluate_consistency,
    )
    from solprop.nonlinear.tdse import TDSEConsistency  # noqa: PLC0415

    class Source:
        def diagnostic_currents(self, field, spacing):
            first, total = alpha_law(1.01)(field, spacing)
            return total[0], first[0]

    return evaluate_consistency(
        Source(),
        preset_basis(1),
        TDSEConsistency(),
        reference_wavelength=800e-9,
        provider_revision="272ef730a7dfa5caab8c0ad75235107b976408c1",
        computed_on={"execution_space": "OpenMP", "build": {}, "threads": 1},
        refresh=True,
    ).to_record()


@pytest.mark.parametrize("kind", ["preset", "custom", "sae"])
def test_schema_18_round_trips_hacc_and_sae_records(tmp_path, monkeypatch, kind):
    from tests.driver.test_tdse_entry import _sae  # noqa: PLC0415
    from tests.hacc_cases import (  # noqa: PLC0415
        hacc_response,
        plain,
        preset_basis,
        tiny_basis,
    )

    from solprop.nonlinear.tdse import TDSESampling  # noqa: PLC0415
    from solprop.output.schema import RunOutcome  # noqa: PLC0415

    monkeypatch.setenv("SOLPROP_CACHE_DIR", str(tmp_path / "cache"))
    response = {
        "preset": lambda: hacc_response(preset_basis(1), atomic_window="auto"),
        "custom": lambda: hacc_response(tiny_basis(), atomic_window="auto"),
        # The gated native source needs a lattice that resolves the grid collar.
        "sae": lambda: dataclasses.replace(_sae(), sampling=TDSESampling(4)),
    }[kind]()
    metadata = _tdse_metadata(response)
    assert metadata.schema_version == 28
    record = metadata.tdse
    assert record is not None
    descriptor = json.loads(str(record["configuration_json"]))
    assert descriptor["schema"] == "solprop.tdse-descriptor.v5"
    assert descriptor["response"]["model"] == response.model
    if kind != "sae":
        assert descriptor["response"]["basis"]["preset"] == (
            preset_basis(1).preset if kind == "preset" else None
        )
    consistency = None if kind == "sae" else _consistency_record()
    path = tmp_path / f"{kind}.h5"
    with HDF5Output(path) as output:
        output.open(metadata)
        output.save(
            SavedPlane(z=0.0, field=np.zeros((1, 1, metadata.grid.n_omega), complex))
        )
        output.record_outcome(
            RunOutcome(
                "complete",
                0.0,
                execution={
                    "tdse": {"model": response.model, "consistency": consistency}
                },
            )
        )
    with open_file(path) as stored:
        assert stored.metadata is not None
        assert stored.metadata.schema_version == 28
        assert stored.metadata.configuration_hash() == metadata.configuration_hash()
        assert stored.metadata.tdse == metadata.tdse
        assert stored.outcome is not None
        assert plain(stored.outcome.execution)["tdse"]["consistency"] == consistency
