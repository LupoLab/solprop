# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Verify optical/archive pairing beyond individual file checksums."""

import hashlib
import json
from typing import Any

import h5py
import pytest
from tests.driver import test_snapshots as cases

from solprop.exceptions import OutputError
from solprop.output import SnapshotController, TDSEArchive, read_snapshot

fake = cases.fake
pytestmark = pytest.mark.medium


def hdf5(path) -> Any:
    """Expose h5py's heterogeneous tree for deliberate corruption in tests."""
    path.chmod(0o600)
    return h5py.File(path, "r+")


def edit_json(file, key, change):
    value = json.loads(file[key].asstr()[()])
    change(value)
    del file[key]
    file.create_dataset(key, data=json.dumps(value))


@pytest.fixture
def pair(fake, tmp_path):
    (tmp_path / "archive").mkdir()
    controller = SnapshotController(tmp_path / "snapshots")
    controller.request()
    cases._run(
        cases._simulation(),
        tmp_path / "working.h5",
        snapshots=controller,
        tdse_archive=TDSEArchive(tmp_path / "archive" / "raw.h5"),
    )
    return next(controller._directory.glob("snapshot-*"))


@pytest.mark.parametrize(
    "change",
    [
        "omitted",
        "relative",
        "filename",
        "uuid",
        "sessions",
        "session_index",
        "session_uuid",
        "closed",
        "counters",
        "configuration",
        "provider",
        "evaluations",
        "decision",
        "incomplete",
    ],
)
def test_mixed_pair_is_rejected_even_with_recomputed_checksums(pair, change):
    manifest_path = pair / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    optical, archive = [pair / row["path"] for row in manifest["files"]]
    if change == "omitted":
        manifest["files"].pop()
    elif change in ("relative", "filename"):
        with hdf5(optical) as file:

            def reference(outcome):
                ref = outcome["execution"]["tdse"]["archive"]
                ref["relative_to" if change == "relative" else "path"] = "wrong"

            edit_json(file, "meta/outcome", reference)
    else:
        _change_archive(archive, change)
    for record in manifest["files"]:
        path = pair / record["path"]
        record.update(
            size=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(OutputError):
        read_snapshot(pair)


def _change_archive(archive, change):
    with hdf5(archive) as file:
        sessions = file["sessions"]
        session = sessions[next(iter(sessions))]
        if change == "uuid":
            file.attrs["run_uuid"] = "different"
        elif change == "sessions":
            del sessions[next(iter(sessions))]
        elif change == "session_index":
            session.attrs["index"] = 9
        elif change == "session_uuid":
            session.attrs["run_uuid"] = "different"
        elif change == "closed":
            session.attrs["closed"] = True
        elif change == "counters":
            session.attrs["array_bytes"] += 1
        elif change in ("configuration", "provider"):
            name = (
                "optical_configuration_hash"
                if change == "configuration"
                else "descriptor_identity"
            )
            edit_json(
                file,
                session.name + "/context",
                lambda value: value.update({name: "different"}),
            )
        elif change == "evaluations":
            evaluations = session["evaluations"]
            del evaluations[next(iter(evaluations))]
        elif change == "decision":
            del session["trials"]
        else:
            evaluation = session["evaluations"][next(iter(session["evaluations"]))]
            del evaluation.attrs["outcome"]
