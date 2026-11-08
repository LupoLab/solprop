# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Reading, showing, exporting, importing and clearing machine profiles (SEP 46)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from solprop import calibration
from solprop.calibration import _native, command
from solprop.cli import main
from solprop.exceptions import BackendError, CalibrationError


@pytest.fixture
def environ(tmp_path: Path) -> dict[str, str]:
    """An environment whose store is empty and whose calibration is on."""
    return {"SOLPROP_CACHE_DIR": str(tmp_path / "store"), "SOLPROP_CALIBRATION": "use"}


@pytest.fixture
def cli_environ(
    environ: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> dict[str, str]:
    """The same environment, set on the process for the command line."""
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("SOLPROP_CALIBRATION_PATH", raising=False)
    return environ


def _publish(environ: dict[str, str], *, exact: bool = True) -> str:
    """Publish a generation for this machine and build; return its fingerprint."""
    found = calibration.choice(environ)
    assert found.fingerprint is not None and found.allocation is not None
    allocation = found.allocation.record()
    if not exact:
        assert isinstance(allocation["cores"], int)
        if allocation["cores"] < 2:
            pytest.skip("needs two cores for a smaller allocation")
        allocation["cores"] = allocation["cores"] - 1
        kinds = allocation["kinds"]
        assert isinstance(kinds, list)
        kinds[0] -= 1
    store = _native.store
    store.Store(calibration.store_directory(environ)).publish(
        store.Generation(
            fingerprint=found.fingerprint.digest,
            name="g1",
            created="2026-11-08T12:00:00Z",
            rows=(
                store.Row(
                    allocation=allocation,
                    level="standard",
                    process={"threads": 2, "wait": "spin"},
                    workspace={"stage_threshold": 1024, "stable_kernel_team": False},
                    fitted_under={"threads": 2, "wait": "spin"},
                    evidence={"geomean": 0.91},
                ),
            ),
        )
    )
    return found.fingerprint.digest


def test_the_pure_modules_load_without_the_native_extension() -> None:
    assert _native.store.MODES == ("auto", "use", "off")
    assert _native.variant().startswith("_core")
    assert _native.packages()
    with pytest.raises(AttributeError, match="no attribute 'extension'"):
        _native.__getattr__("extension")


def test_without_the_native_package_the_store_cannot_be_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_native.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(BackendError, match="install solprop-native"):
        _native.packages()


def test_no_profile_means_the_built_in_table(environ: dict[str, str]) -> None:
    assert calibration.profile(environ) is None
    assert calibration.profile(environ | {"SOLPROP_CALIBRATION": "off"}) is None
    with pytest.raises(CalibrationError, match="no calibration profile"):
        calibration.export_profile(Path("unused.json"), environ)
    assert calibration.clear(environ) == 0
    assert calibration.clear(environ | {"SOLPROP_CALIBRATION": "off"}) == 0
    text = command.describe(environ)
    assert "no profile for this machine and build" in text
    assert "SOLPROP_CALIBRATION=use" in text
    off = command.describe(environ | {"SOLPROP_CALIBRATION": "off"})
    assert "profile:      none (the built-in table applies)" in off


def test_a_published_profile_is_read_shown_exported_and_cleared(
    environ: dict[str, str], tmp_path: Path
) -> None:
    digest = _publish(environ)
    found = calibration.profile(environ)
    assert found is not None
    assert (found.fingerprint, found.generation, found.level) == (
        digest,
        "g1",
        "standard",
    )
    assert found.exact
    assert found.workspace["stage_threshold"] == 1024
    assert found.source is not None and found.source.name == "g1.json"
    text = command.describe(environ)
    assert "standard, generation g1" in text
    assert "measured on this allocation" in text
    assert "stage_threshold 1024 (built-in" in text
    assert "stable_kernel_team no (built-in" in text
    assert "evidence:     geomean: 0.91" in text
    # Export, clear, then import back.
    exported = tmp_path / "node.json"
    assert calibration.export_profile(exported, environ) == found.source
    assert json.loads(exported.read_text(encoding="utf-8"))["fingerprint"] == digest
    assert calibration.clear(environ) == 2
    assert calibration.profile(environ) is None
    assert calibration.import_profile(exported, environ) is True
    assert calibration.profile(environ) is not None


def test_a_profile_for_a_smaller_allocation_supplies_workspace_settings(
    environ: dict[str, str],
) -> None:
    _publish(environ, exact=False)
    found = calibration.profile(environ)
    assert found is not None
    assert not found.exact
    assert "workspace settings only" in command.describe(environ)


def test_another_machines_profile_is_imported_but_does_not_apply(
    environ: dict[str, str], tmp_path: Path
) -> None:
    foreign = tmp_path / "foreign.json"
    foreign.write_text(
        json.dumps(
            {
                "format": 1,
                "fingerprint": "0" * 24,
                "name": "g1",
                "created": "2026-11-08",
                "rows": [],
            }
        ),
        encoding="utf-8",
    )
    assert calibration.import_profile(foreign, environ) is False
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(CalibrationError, match="cannot import"):
        calibration.import_profile(broken, environ)


def test_the_command_line_shows_exports_imports_and_clears(
    cli_environ: dict[str, str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["calibrate", "--show"]) == 0
    assert "the built-in table applies" in capsys.readouterr().out
    exported = tmp_path / "out.json"
    assert main(["calibrate", "--export", str(exported)]) == 2
    assert "no calibration profile" in capsys.readouterr().err
    _publish(cli_environ)
    assert main(["calibrate", "--export", str(exported)]) == 0
    assert f"to {exported}" in capsys.readouterr().out
    assert main(["calibrate", "--clear"]) == 0
    assert "removed 2 file(s)" in capsys.readouterr().out
    assert main(["calibrate", "--import", str(exported)]) == 0
    assert "applies to this machine and build" in capsys.readouterr().out
    foreign = tmp_path / "foreign.json"
    record = json.loads(exported.read_text(encoding="utf-8"))
    foreign.write_text(json.dumps(record | {"fingerprint": "f" * 24}), encoding="utf-8")
    assert main(["calibrate", "--import", str(foreign)]) == 0
    assert "does not apply here" in capsys.readouterr().out
    # Measuring is not in this build yet.
    assert main(["calibrate"]) == 2
    assert "not available" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["calibrate", "--show", "--clear"])


def test_read_only_directories_are_listed(environ: dict[str, str]) -> None:
    text = command.describe(environ | {"SOLPROP_CALIBRATION_PATH": "/shared/profiles"})
    assert "read-only:    /shared/profiles" in text


def test_a_build_without_its_record_is_still_described(
    environ: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprint = _native.fingerprint
    original = fingerprint.build_identity

    def without_record(packages: object, stem: str = "_core") -> dict[str, object]:
        return original(packages, stem) | {"record": None}  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(_native.profile, "build_identity", without_record)
    assert "no build record" in command.describe(environ)


def test_the_summary_names_kinds_quota_and_spans() -> None:
    line = command._summary(  # pyright: ignore[reportPrivateUsage]
        {
            "cores": 18,
            "kinds": [6, 12],
            "sockets": 1,
            "numa_nodes": 1,
            "l3_groups": 1,
            "smt": False,
            "quota": 4,
        }
    )
    assert line == "18 cores, of kinds 6 + 12, quota 4"


def test_the_command_does_not_load_the_native_extension(tmp_path: Path) -> None:
    import subprocess  # noqa: PLC0415 - only here

    script = (
        "import sys; from solprop.cli import main; main(['calibrate', '--show']); "
        "print('solprop_native._core' in sys.modules, file=sys.stderr)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={"SOLPROP_CACHE_DIR": str(tmp_path), "PATH": "", "HOME": str(tmp_path)},
    )
    assert completed.stderr.strip().splitlines()[-1] == "False"
