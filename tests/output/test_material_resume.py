# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Output schema 8: the material manifest on record, and resume refused when it moves.

Each record is a capillary's metadata built without propagating; a file holds one
plane, which is all a resume check reads.
"""

import dataclasses
import json
from typing import Any

import numpy as np
import pytest
from optmat.gas.raman_response import raman_response

from solprop.boundaries import absorber
from solprop.constants import ROOM_TEMPERATURE
from solprop.driver import Simulation
from solprop.driver.scan import Scan
from solprop.exceptions import OutputError
from solprop.grid import FieldGrid
from solprop.output import HDF5Output, MemoryOutput, open_file
from solprop.output.hdf5 import check_cache, resume
from solprop.output.scan import PointResult, ScanWriter
from solprop.output.schema import SCHEMA_VERSION, RunMetadata, SavedPlane
from solprop.physdata import _optmat
from solprop.physdata.selection import MaterialManifest, MaterialSelection
from solprop.stepper import Multistep

pytestmark = pytest.mark.filterwarnings(
    "ignore::solprop.exceptions.MaterialRangeWarning"
)

LENGTH = 4e-4


SELECTION = MaterialSelection()


def metadata(materials: MaterialSelection | None = SELECTION) -> Any:
    """The record of a short argon capillary under a selection, or of today's."""
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 100e-15)
    return Simulation.from_capillary(
        50e-6,
        LENGTH,
        "Ar",
        1.0,
        lambda0=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0),
        fwhm=30e-15,
        energy=1e-6,
        plasma=False,
        n_r=8,
        n_save=2,
        backend="numpy_ref",
        materials=materials,
        noise=None,
        # The Adams march, so that the legacy records below carry a method string.
        integrator=Multistep(),
    ).metadata()


def written(path, record):
    """Write one plane of a run's record into an HDF5 file at `path`."""
    handler = HDF5Output(path)
    handler.open(record)
    handler.save(SavedPlane(z=0.0, field=np.ones((1, 1, record.grid.n_omega), complex)))
    handler.close()
    return path


@pytest.fixture(scope="module")
def selected():
    return metadata()


def test_the_current_schema_carries_the_manifest_through_memory_and_json(selected):
    assert SCHEMA_VERSION == 26 == selected.schema_version
    memory = MemoryOutput()
    memory.open(selected)
    stored: Any = memory.metadata
    assert stored.materials == selected.materials
    back = RunMetadata.from_json(selected.to_json())
    assert back.materials == selected.materials
    assert back.configuration_hash() == selected.configuration_hash()
    document = json.loads(selected.to_json())
    # The file keeps the measurements beside the hashed part.
    assert document["materials"]["dilute_scaling"]
    assert "dilute_scaling" not in selected.hashed()["materials"]


def test_schema_eight_round_trips_through_hdf5_and_resumes(tmp_path, selected):
    path = written(tmp_path / "run.h5", selected)
    stored: Any = open_file(path).metadata
    assert stored.materials == selected.materials
    assert resume(path).metadata.materials == selected.materials
    assert check_cache(path, selected)
    # An identical manifest rebuilt in the same process resumes too.
    assert check_cache(path, metadata())


def test_the_printed_record_names_optmat_and_its_models(selected):
    block = str(selected)
    assert "materials:" in block and "optmat " in block and "fit/Ar" in block


@pytest.mark.parametrize(
    ("materials", "entry"),
    [
        (MaterialSelection(electronic="carrier-kerr"), "materials.policies.electronic"),
        (MaterialSelection(carrier_wavelength=800e-9), "materials.policies.carrier"),
    ],
    ids=["electronic", "carrier"],
)
def test_a_changed_policy_refuses_resume(tmp_path, selected, materials, entry):
    path = written(tmp_path / "run.h5", selected)
    with pytest.raises(OutputError, match=entry.replace(".", r"\.")) as refused:
        check_cache(path, metadata(materials))
    assert "the material manifest differs" in str(refused.value)


def test_a_changed_optmat_version_refuses_resume(tmp_path, selected, monkeypatch):
    # Deliberately conservative: the version alone refuses, whatever moved.
    path = written(tmp_path / "run.h5", selected)
    monkeypatch.setattr(_optmat, "optmat_version", lambda: "0.1.2")
    with pytest.raises(OutputError, match=r"materials\.optmat_version"):
        check_cache(path, metadata())


def test_a_schema_seven_file_is_refused_by_its_resume_key(tmp_path, selected):
    # A schema-7 record carried no manifest (SEP 3 Phase 4b: the registry run that once
    # stood for it is deleted, so the empty manifest is written directly).
    legacy = dataclasses.replace(
        selected, schema_version=7, quadrature=None, materials=MaterialManifest()
    )
    path = written(tmp_path / "old.h5", legacy)
    # It still reads, and its stamped digest still verifies.
    stored: Any = open_file(path).metadata
    assert stored.schema_version == 7
    with pytest.raises(OutputError, match=r"different run.*schema_version"):
        check_cache(path, selected)


def test_a_schema_eight_file_is_refused_by_its_resume_key(tmp_path, selected):
    # Schema 9 (SEP 13) put the stepping method in the resume key; a schema-8 digest
    # omits it, so the two cannot be compared.
    legacy = dataclasses.replace(selected, schema_version=8, quadrature=None)
    path = written(tmp_path / "old.h5", legacy)
    stored: Any = open_file(path).metadata
    assert stored.schema_version == 8
    with pytest.raises(OutputError, match=r"different run.*schema_version"):
        check_cache(path, selected)


def test_a_schema_nine_file_is_refused_by_its_resume_key(tmp_path, selected):
    # Schema 10 (SEP 14) put the interval cap and the start-up rule in the resume key
    # when their defaults changed; a schema-9 run was stepped under the old ones.
    legacy = dataclasses.replace(selected, schema_version=9, quadrature=None)
    path = written(tmp_path / "old.h5", legacy)
    stored: Any = open_file(path).metadata
    assert stored.schema_version == 9
    with pytest.raises(OutputError, match=r"different run.*schema_version"):
        check_cache(path, selected)


def test_a_manifest_needs_schema_eight(selected):
    with pytest.raises(OutputError, match="requires output schema 8"):
        dataclasses.replace(selected, schema_version=7, quadrature=None)
    # Below 8 the hashed record has no materials entry at all.
    empty = MaterialManifest()
    legacy = dataclasses.replace(
        selected, schema_version=7, quadrature=None, materials=empty
    )
    assert "materials" not in legacy.hashed()


def test_a_malformed_stored_manifest_is_refused_on_reading(selected):
    document = json.loads(selected.to_json())
    record = next(iter(document["materials"]["records"].values()))
    del record["kind"]
    with pytest.raises(OutputError, match="missing or malformed"):
        RunMetadata.from_json(json.dumps(document))


def test_a_scan_conflict_without_a_record_keeps_the_workers_reason(tmp_path):
    writer = ScanWriter(tmp_path / "scan.h5", Scan("x", energy=[1.0]), {})
    with pytest.raises(OutputError, match=r"^point 0 differs$"):
        writer.commit(PointResult(0, "conflict", error="point 0 differs"))
    writer.close(consolidate=False)


def run_metadata(gas: str, materials: MaterialSelection, **changes: Any) -> Any:
    """The record of a short capillary of `gas` under `materials`."""
    options: dict[str, Any] = {"plasma": False} | changes
    grid = FieldGrid.from_limits(LENGTH, 800e-9, (300e-9, 3e-6), 100e-15)
    return Simulation.from_capillary(
        50e-6,
        LENGTH,
        gas,
        1.0,
        lambda0=800e-9,
        grid=grid,
        absorber=absorber(grid, boundary_length=1.0),
        fwhm=30e-15,
        energy=1e-6,
        n_r=8,
        n_save=2,
        backend="numpy_ref",
        materials=materials,
        **{"noise": None, **options},
    ).metadata()


@pytest.mark.parametrize(
    ("gas", "run", "changed"),
    [
        ("Ar", {"plasma": True}, MaterialSelection(ionisation="ppt_ilkov")),
        (
            "N2",
            {"raman": True},
            MaterialSelection(
                raman=raman_response(
                    "N2", ROOM_TEMPERATURE, wavelength=800e-9
                ).with_parameters(width_scale=0.8)
            ),
        ),
    ],
    ids=["ionisation", "raman"],
)
def test_a_changed_raman_or_ionisation_choice_refuses_resume(
    tmp_path, gas, run, changed
):
    # SEP 3, Q6: since Phases 1c and 1d resolve them, the selection's `raman` and
    # `ionisation` reach the key through the records and policies they resolve to.
    base = run_metadata(gas, SELECTION, **run)
    other = run_metadata(gas, changed, **run)
    assert other.configuration_hash() != base.configuration_hash()
    path = written(tmp_path / "run.h5", base)
    with pytest.raises(OutputError, match=r"materials\.") as refused:
        check_cache(path, other)
    assert "the material manifest differs" in str(refused.value)
    assert check_cache(path, run_metadata(gas, SELECTION, **run))


def test_a_choice_the_run_does_not_use_does_not_refuse_resume(tmp_path):
    # Keyed by what it resolves to, a choice that resolves nothing in a run is not a
    # different run: argon without plasma takes no rate.
    base = run_metadata("Ar", SELECTION)
    other = run_metadata("Ar", MaterialSelection(ionisation="ppt_ilkov"))
    assert other.configuration_hash() == base.configuration_hash()
    assert check_cache(written(tmp_path / "run.h5", base), other)


def test_a_schema_ten_file_is_refused_by_its_resume_key(tmp_path, selected):
    # Schema 11 (SEP 14 phase 3) put interpolated and step planes in the resume key.
    legacy = dataclasses.replace(selected, schema_version=10, quadrature=None)
    path = written(tmp_path / "old.h5", legacy)
    stored: Any = open_file(path).metadata
    assert stored.schema_version == 10
    with pytest.raises(OutputError, match=r"different run.*schema_version"):
        check_cache(path, selected)
