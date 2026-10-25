# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the metadata schema, its serialisation, provenance and resume key."""

import dataclasses
import json
import math
import os
import subprocess
import sys
from enum import Enum
from typing import Any

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from solprop import modes
from solprop.exceptions import OutputError
from solprop.grid import EnvelopeGrid, FieldGrid
from solprop.modes import (
    IndexTable,
    MarcatiliMode,
    RectMode,
    StepIndexMode,
    TabulatedHollowMode,
)
from solprop.noise import input_noise
from solprop.nonlinear.gnlse import problem_from
from solprop.nonlinear.raman import RamanResponse, response_for
from solprop.nonlinear.responses import KerrResponse
from solprop.output import schema
from solprop.output.schema import (
    FIELD_LAYOUT,
    SCHEMA_VERSION,
    SPECTRAL_CONVENTION,
    GitProvenance,
    GridRecord,
    ModeRecord,
    NoiseRecord,
    RunMetadata,
    SavedPlane,
    StepHistory,
    StepLimits,
    StepperRecord,
    StepStats,
    package_versions,
    resolve_metadata_write,
    serialise,
)
from solprop.physdata import aluminium, fused_silica
from solprop.physdata.gases import Gas
from solprop.physdata.selection import MaterialSelection
from solprop.profiles import PressureGradient
from solprop.stepper import DOPRI5, FixedSchedule, PIController


def serialised(value: object) -> Any:
    """Return `serialise(value)`, typed loosely so that a test may index into it."""
    return serialise(value)


WAVELENGTH = 835e-9
LIMITS = (400e-9, 2e-6)
TRANGE = 2e-12
RADIUS = 125e-6
LENGTH = 1.0


@pytest.fixture
def envelope_grid():
    return EnvelopeGrid.from_limits(LENGTH, WAVELENGTH, LIMITS, TRANGE)


@pytest.fixture
def field_grid():
    return FieldGrid.from_limits(LENGTH, WAVELENGTH, LIMITS, TRANGE)


@pytest.fixture
def gradient_mode():
    return MarcatiliMode(
        RADIUS, "He", PressureGradient.two_point(LENGTH, 2.0, 0.5), model="reduced"
    )


@pytest.fixture
def stepper():
    return StepperRecord.from_settings(
        DOPRI5,
        PIController(rtol=1e-8, atol=0.0),
        schedule=FixedSchedule(np.array([0.5, 0.5])),
        limits=StepLimits(dz_max=0.1),
    )


def make_metadata(grid, stepper, **overrides):
    """Build a record over one grid, with any field replaced."""
    arguments = {
        "grid": GridRecord.from_grid(grid),
        "modes": (),
        "simulation_kind": "mode_averaged",
        "configuration": {"gamma": 0.11},
        "stepper": stepper,
    }
    return RunMetadata(**(arguments | overrides))


# ------------------------------------------------------------------- serialisation


class _Colour(Enum):
    RED = "red"


@dataclasses.dataclass(frozen=True)
class _Derived:
    given: float
    doubled: float = dataclasses.field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "doubled", 2.0 * self.given)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (3, 3),
        (2.5, 2.5),
        ("text", "text"),
        (np.float64(1.5), 1.5),
        (np.int64(7), 7),
        ((1, 2), [1, 2]),
        ([1, [2]], [1, [2]]),
        ({"a": 1}, {"a": 1}),
        ({2: 1}, {"2": 1}),
        (math.inf, math.inf),
    ],
)
def test_serialise_passes_json_primitives_and_containers_through(value, expected):
    assert serialise(value) == expected


def test_serialise_records_a_complex_number_as_a_pair():
    assert serialised(1.0 - 2.0j) == {"__complex__": [1.0, -2.0]}


def test_serialise_records_a_numpy_scalar_complex_as_a_pair():
    assert serialised(np.complex128(1 + 2j)) == {"__complex__": [1.0, 2.0]}


def test_serialise_records_an_array_with_its_dtype_and_shape():
    record = serialised(np.arange(6, dtype=np.int32).reshape(2, 3))
    assert record["dtype"] == "int32"
    assert record["shape"] == [2, 3]
    assert record["__ndarray__"] == [[0, 1, 2], [3, 4, 5]]


def test_serialise_records_an_enumeration_with_its_class():
    record = serialised(_Colour.RED)
    assert record["value"] == "red"
    assert record["__enum__"].endswith("_Colour")


def test_serialise_records_a_string_enumeration_as_an_enumeration():
    # Gas is a StrEnum, so a plain isinstance(value, str) test would lose the class.
    record = serialised(Gas.He)
    assert record == {"__enum__": "solprop.physdata.gases.Gas", "value": "He"}


def test_serialise_records_a_path_as_a_string(tmp_path):
    assert serialise(tmp_path) == str(tmp_path)


def test_serialise_orders_a_set_deterministically():
    assert serialise({3, 1, 2}) == serialise({2, 3, 1})


def test_serialise_skips_derived_dataclass_fields():
    record = serialised(_Derived(1.5))
    assert record["given"] == 1.5
    assert "doubled" not in record


def test_serialise_marks_a_callable_as_not_reproducible():
    record = serialised(math.sin)
    assert record["reproducible"] is False
    assert "sin" in record["__callable__"]


def test_serialise_marks_an_opaque_object_as_not_reproducible():
    record = serialised(object())
    assert record["reproducible"] is False
    assert "object" in record["__repr__"]


def test_serialise_is_idempotent(gradient_mode):
    once = serialised(gradient_mode)
    assert serialised(once) == once


def test_serialise_round_trips_a_gnlse_problem():
    problem = problem_from([0.0, 0.0, -11.83e-27], omega_ref=2.26e15, gamma=0.11)
    record = serialised(problem)
    assert record["__class__"] == "solprop.nonlinear.gnlse.GnlseProblem"
    assert record["gamma"] == 0.11
    assert record["kerr"]["__class__"] == "solprop.nonlinear.responses.KerrResponse"
    # The dispersion is a frozen callable dataclass, so the record carries the Taylor
    # coefficients themselves rather than a closure's repr, and the configuration hash
    # taken over it is stable across processes -- which is what a resume needs.
    dispersion = record["mode"]["dispersion"]
    assert dispersion["__class__"] == "solprop.nonlinear.gnlse.TaylorDispersion"
    assert dispersion["betas"]["__ndarray__"] == [0.0, 0.0, -11.83e-27]
    assert json.loads(json.dumps(record)) == record


def test_serialise_round_trips_a_capillary_with_a_pressure_gradient(gradient_mode):
    record = serialised(gradient_mode)
    pressure = record["pressure"]
    assert pressure["__class__"] == "solprop.profiles.PressureGradient"
    assert pressure["pressures"]["__ndarray__"] == [2.0, 0.5]
    assert json.loads(json.dumps(record)) == record


def test_serialise_round_trips_a_kerr_response():
    assert serialised(KerrResponse(chi3=1e-25, thg=False)) == {
        "__class__": "solprop.nonlinear.responses.KerrResponse",
        "chi3": 1e-25,
        "thg": False,
        "species": None,
    }


# ------------------------------------------------------------------- record classes


def test_grid_record_keeps_the_scalars_that_rebuild_the_axis(envelope_grid):
    record = GridRecord.from_grid(envelope_grid)
    assert record.kind == "envelope"
    assert record.n_omega == envelope_grid.n_omega
    # Everything stored is a scalar, so the record is JSON-safe without an array.
    assert "__ndarray__" not in json.dumps(serialised(record))


def test_mode_record_carries_the_class_and_the_effective_area(gradient_mode):
    record = ModeRecord.from_mode(gradient_mode)
    assert record.kind == "solprop.modes.capillary.MarcatiliMode"
    assert record.effective_area == pytest.approx(gradient_mode.aeff(0.0))


def test_noise_record_stays_in_the_internal_convention(envelope_grid):
    realisation = input_noise(envelope_grid, np.random.default_rng(5))
    record = NoiseRecord.from_realisation(realisation)
    assert record.spectral_convention == "fft"
    assert record.seed == 5
    assert np.array_equal(record.spectrum, realisation.spectrum)


def test_noise_record_spectrum_is_read_only(envelope_grid):
    record = NoiseRecord.from_realisation(
        input_noise(envelope_grid, np.random.default_rng(5))
    )
    with pytest.raises(ValueError, match="read-only"):
        record.spectrum[0, 0, 0] = 1.0


def test_stepper_record_reduces_its_objects_to_records(stepper):
    assert stepper.tableau == "DOPRI5"
    assert stepper.controller["rtol"] == 1e-8
    schedule = stepper.schedule
    assert schedule is not None
    assert np.array_equal(schedule, [0.5, 0.5])
    assert stepper.limits.dz_max == 0.1


def test_stepper_record_round_trips_through_its_stored_form(stepper):
    same = StepperRecord.from_record(serialised(stepper))
    assert same.tableau == stepper.tableau
    assert same.limits == stepper.limits
    stored, original = same.schedule, stepper.schedule
    assert stored is not None
    assert original is not None
    assert np.array_equal(stored, original)


def test_stepper_record_keeps_none_for_an_adaptive_run():
    record = StepperRecord.from_settings(DOPRI5, PIController(rtol=1e-6, atol=0.0))
    assert record.schedule is None
    assert record.limits == StepLimits()


def test_step_history_reports_its_length_and_rejections():
    history = StepHistory(
        z=np.array([0.0, 0.1, 0.1]),
        h=np.array([0.1, 0.2, 0.1]),
        norm=np.array([0.4, 1.7, 0.5]),
        accepted=np.array([True, False, True]),
        wall_time=np.array([1e-3, 1e-3, 1e-3]),
    )
    assert (history.n_steps, history.n_rejected) == (3, 1)


def test_step_history_rejects_unequal_lengths():
    with pytest.raises(OutputError, match="equal shapes"):
        StepHistory(
            z=np.array([0.0]),
            h=np.array([0.1, 0.2]),
            norm=np.array([0.4]),
            accepted=np.array([True]),
            wall_time=np.array([1e-3]),
        )


def test_step_stats_reports_its_rows_and_the_statistics_it_carries():
    rows = StepStats(
        z=np.array([0.1, 0.2]),
        step=np.array([0, 3]),
        values={"energy": np.ones((2, 1, 1))},
    )
    assert rows.n_rows == 2
    assert rows.names == ("energy",)
    assert rows.step.dtype == np.int64
    assert not rows.z.flags.writeable
    assert not rows.values["energy"].flags.writeable


def test_step_stats_rejects_columns_that_describe_different_rows():
    with pytest.raises(OutputError, match="equal row counts"):
        StepStats(
            z=np.array([0.1, 0.2]),
            step=np.array([0, 1]),
            values={"energy": np.ones((3, 1))},
        )


def test_an_empty_channel_carries_no_rows_and_no_names():
    rows = StepStats.empty()
    assert rows.n_rows == 0
    assert rows.names == ()
    assert rows.step.dtype == np.int64


def test_saved_plane_defaults_to_an_empty_history_and_no_noise():
    plane = SavedPlane(z=0.0, field=np.zeros((1, 1, 5), dtype=complex))
    assert plane.step_history.n_steps == 0
    assert plane.noise_phase is None
    assert plane.stats == {}


def test_saved_plane_rejects_a_non_finite_position():
    with pytest.raises(OutputError, match="finite position"):
        SavedPlane(z=math.nan, field=np.zeros((1, 1, 5), dtype=complex))


def test_saved_plane_rejects_a_noise_phase_of_the_wrong_shape():
    with pytest.raises(OutputError, match="noise_phase has shape"):
        SavedPlane(
            z=0.0,
            field=np.zeros((1, 1, 5), dtype=complex),
            noise_phase=np.zeros((1, 1, 4)),
        )


def test_saved_plane_keeps_a_read_only_copy_of_the_field():
    field = np.ones((1, 1, 5), dtype=complex)
    plane = SavedPlane(z=0.0, field=field, noise_phase=np.zeros((1, 1, 5)))
    field[0, 0, 0] = 99.0
    assert plane.field is not None
    assert plane.field[0, 0, 0] == 1.0
    phase = plane.noise_phase
    assert phase is not None
    with pytest.raises(ValueError, match="read-only"):
        phase[0, 0, 0] = 1.0


# ------------------------------------------------------------------- provenance


def test_package_versions_reports_the_six_it_promises():
    versions = package_versions()
    assert sorted(versions) == ["h5py", "numpy", "optmat", "python", "scipy", "solprop"]
    assert all(isinstance(value, str) and value for value in versions.values())


@pytest.fixture
def fresh_provenance():
    """Forget whether the checkout tracks the package, asked once per process."""
    schema._tracks.cache_clear()
    yield
    schema._tracks.cache_clear()


def _status(branch: str = "main", dirty: bool = False) -> str:
    """A ``git status --porcelain=v2 --branch`` answer."""
    lines = ["# branch.oid " + "a" * 40, f"# branch.head {branch}"]
    if dirty:
        lines.append("1 .M N... 100644 100644 100644 0 0 some/file.py")
    return "\n".join(lines)


def test_git_provenance_is_a_commit_or_none():
    record = GitProvenance.detect()
    assert record is None or len(record.commit) == 40


def test_git_provenance_is_none_when_git_is_absent(monkeypatch, fresh_provenance):
    def missing(*args, **kwargs):
        raise FileNotFoundError("no git here")

    monkeypatch.setattr(subprocess, "run", missing)
    assert GitProvenance.detect() is None


def test_git_provenance_is_none_outside_a_repository(monkeypatch, fresh_provenance):
    def failing(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=128, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", failing)
    assert GitProvenance.detect() is None


@pytest.mark.parametrize(
    ("branch", "dirty", "expected"),
    [("main", False, ("main", False)), ("(detached)", True, ("HEAD", True))],
)
def test_git_provenance_reads_one_status_per_run(
    monkeypatch, fresh_provenance, branch, dirty, expected
):
    # The checkout's identity is asked once per process; each run then costs one git
    # process, whose porcelain answer carries the commit, the branch and the flag.
    calls = []

    def answer(arguments, **kwargs):
        calls.append(arguments[1])
        text = "schema.py" if arguments[1] == "ls-files" else _status(branch, dirty)
        return subprocess.CompletedProcess(arguments, 0, stdout=text, stderr="")

    monkeypatch.setattr(subprocess, "run", answer)
    for _ in range(2):
        record = GitProvenance.detect()
        assert record is not None
        assert (record.commit, record.branch, record.dirty) == ("a" * 40, *expected)
    assert calls == ["ls-files", "status", "status"]


def test_git_provenance_is_none_without_a_commit(monkeypatch, fresh_provenance):
    # A repository before its first commit reports "(initial)" for the commit.
    def answer(arguments, **kwargs):
        text = (
            "schema.py"
            if arguments[1] == "ls-files"
            else "# branch.oid (initial)\n# branch.head main"
        )
        return subprocess.CompletedProcess(arguments, 0, stdout=text, stderr="")

    monkeypatch.setattr(subprocess, "run", answer)
    assert GitProvenance.detect() is None


def test_git_provenance_is_none_when_status_fails(monkeypatch, fresh_provenance):
    def answer(arguments, **kwargs):
        code = 0 if arguments[1] == "ls-files" else 128
        return subprocess.CompletedProcess(
            arguments, code, stdout="schema.py", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", answer)
    assert GitProvenance.detect() is None


def test_git_provenance_is_none_inside_a_repository_that_does_not_track_it(
    monkeypatch, fresh_provenance
):
    # A virtual environment inside a user's own project: git answers with that
    # project's commit, which is not solprop's provenance. The repository has to
    # track this file to count.
    def answer(arguments, **kwargs):
        if arguments[1] == "ls-files":
            return subprocess.CompletedProcess(arguments, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(arguments, 0, stdout=_status(), stderr="")

    monkeypatch.setattr(subprocess, "run", answer)
    assert GitProvenance.detect() is None


def test_git_provenance_is_none_when_git_times_out(monkeypatch, fresh_provenance):
    def slow(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=1.0)

    monkeypatch.setattr(subprocess, "run", slow)
    assert GitProvenance.detect() is None


# ------------------------------------------------------------------- run metadata


def test_metadata_states_the_conventions_it_promises(envelope_grid, stepper):
    metadata = make_metadata(envelope_grid, stepper)
    assert metadata.spectral_convention == SPECTRAL_CONVENTION == "optics"
    assert metadata.field_layout == FIELD_LAYOUT
    assert metadata.created


def test_metadata_hash_is_sixty_four_hexadecimal_characters(envelope_grid, stepper):
    digest = make_metadata(envelope_grid, stepper).configuration_hash()
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_metadata_hash_ignores_wall_time_versions_and_git(envelope_grid, stepper):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(
        envelope_grid,
        stepper,
        created="1999-01-01T00:00:00+00:00",
        versions={"solprop": "0.0.0"},
        git=GitProvenance(commit="f" * 40, branch="other", dirty=True),
    )
    assert one.configuration_hash() == two.configuration_hash()


@pytest.mark.parametrize(
    "overrides",
    [
        {"configuration": {"gamma": 0.22}},
        {"simulation_kind": "multimode"},
        {"schema_version": SCHEMA_VERSION + 1},
    ],
    ids=["configuration", "simulation_kind", "schema_version"],
)
def test_metadata_hash_changes_with_any_physics_field(
    envelope_grid, stepper, overrides
):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(envelope_grid, stepper, **overrides)
    assert one.configuration_hash() != two.configuration_hash()


def test_metadata_hash_changes_with_the_grid(envelope_grid, field_grid, stepper):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(field_grid, stepper)
    assert one.configuration_hash() != two.configuration_hash()


def test_metadata_hash_changes_with_the_noise_seed(envelope_grid, stepper):
    one = make_metadata(
        envelope_grid,
        stepper,
        noise=NoiseRecord.from_realisation(
            input_noise(envelope_grid, np.random.default_rng(1))
        ),
    )
    two = make_metadata(
        envelope_grid,
        stepper,
        noise=NoiseRecord.from_realisation(
            input_noise(envelope_grid, np.random.default_rng(2))
        ),
    )
    assert one.configuration_hash() != two.configuration_hash()


def test_metadata_hash_changes_with_a_mode(envelope_grid, stepper, gradient_mode):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(
        envelope_grid, stepper, modes=(ModeRecord.from_mode(gradient_mode),)
    )
    assert one.configuration_hash() != two.configuration_hash()


def test_metadata_hash_is_stable_across_processes(envelope_grid, stepper):
    expected = make_metadata(envelope_grid, stepper).configuration_hash()
    script = "\n".join(
        [
            "import numpy as np",
            "from solprop.grid import EnvelopeGrid",
            "from solprop.output.schema import GridRecord, RunMetadata, StepLimits",
            "from solprop.output.schema import StepperRecord",
            "from solprop.stepper import DOPRI5, FixedSchedule, PIController",
            "grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)",
            "stepper = StepperRecord.from_settings(",
            "    DOPRI5,",
            "    PIController(rtol=1e-8, atol=0.0),",
            "    schedule=FixedSchedule(np.array([0.5, 0.5])),",
            "    limits=StepLimits(dz_max=0.1),",
            ")",
            "print(",
            "    RunMetadata(",
            "        grid=GridRecord.from_grid(grid),",
            "        modes=(),",
            "        simulation_kind='mode_averaged',",
            "        configuration={'gamma': 0.11},",
            "        stepper=stepper,",
            "    ).configuration_hash()",
            ")",
        ]
    )
    # A different hash seed is the direct test that no builtin hash() is involved:
    # PYTHONHASHSEED salts str.__hash__ per interpreter and would change the answer.
    environment = dict(os.environ) | {"PYTHONHASHSEED": "12345"}
    finished = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    )
    assert finished.stdout.strip() == expected


def test_metadata_json_round_trips_to_the_same_hash(envelope_grid, stepper):
    metadata = make_metadata(
        envelope_grid,
        stepper,
        noise=NoiseRecord.from_realisation(
            input_noise(envelope_grid, np.random.default_rng(7))
        ),
    )
    back = RunMetadata.from_json(metadata.to_json())
    assert back.configuration_hash() == metadata.configuration_hash()
    assert back.created == metadata.created
    assert back.versions == metadata.versions
    stored, original = back.noise, metadata.noise
    assert stored is not None
    assert original is not None
    assert np.array_equal(stored.spectrum, original.spectrum)


def test_metadata_json_round_trips_a_mode_and_a_schedule(
    envelope_grid, stepper, gradient_mode
):
    metadata = make_metadata(
        envelope_grid, stepper, modes=(ModeRecord.from_mode(gradient_mode),)
    )
    back = RunMetadata.from_json(metadata.to_json())
    assert back.modes[0].kind == "solprop.modes.capillary.MarcatiliMode"
    stored, original = back.stepper.schedule, stepper.schedule
    assert stored is not None
    assert original is not None
    assert np.array_equal(stored, original)
    assert back.git == metadata.git


def test_metadata_json_round_trips_without_git(envelope_grid, stepper):
    metadata = make_metadata(envelope_grid, stepper, git=None)
    assert RunMetadata.from_json(metadata.to_json()).git is None


def test_metadata_from_json_rejects_a_document_that_is_not_json():
    with pytest.raises(OutputError, match="not valid JSON"):
        RunMetadata.from_json("{not json")


def test_metadata_from_json_rejects_a_document_missing_a_field():
    with pytest.raises(OutputError, match="missing or malformed"):
        RunMetadata.from_json('{"grid": {}}')


def test_metadata_differences_names_the_changed_path(envelope_grid, stepper):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(envelope_grid, stepper, configuration={"gamma": 0.22})
    assert one.differences(two) == ("configuration.gamma",)


def test_metadata_differences_names_an_added_key(envelope_grid, stepper):
    one = make_metadata(envelope_grid, stepper)
    changed = {"gamma": 0.11, "n2": 1.0}
    two = make_metadata(envelope_grid, stepper, configuration=changed)
    assert one.differences(two) == ("configuration.n2",)


def test_metadata_differences_names_a_list_entry(envelope_grid, stepper, gradient_mode):
    wider = MarcatiliMode(2.0 * RADIUS, "He", 1.0, model="reduced")
    one = make_metadata(
        envelope_grid, stepper, modes=(ModeRecord.from_mode(gradient_mode),)
    )
    two = make_metadata(envelope_grid, stepper, modes=(ModeRecord.from_mode(wider),))
    assert "modes[0].effective_area" in one.differences(two)


def test_metadata_differences_reports_a_length_change(envelope_grid, stepper):
    one = make_metadata(envelope_grid, stepper, configuration={"betas": [1.0, 2.0]})
    two = make_metadata(envelope_grid, stepper, configuration={"betas": [1.0]})
    assert one.differences(two) == ("configuration.betas[length]",)


def test_metadata_differences_is_empty_for_the_same_run(envelope_grid, stepper):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(envelope_grid, stepper, created="2000-01-01T00:00:00+00:00")
    assert one.differences(two) == ()


# ------------------------------------------------------------- the overwrite rule


def test_overwrite_rule_writes_when_nothing_is_held(envelope_grid, stepper):
    assert resolve_metadata_write(None, make_metadata(envelope_grid, stepper)) is True


def test_overwrite_rule_is_a_no_op_for_the_same_run(envelope_grid, stepper):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(envelope_grid, stepper)
    assert resolve_metadata_write(one, two) is False


def test_overwrite_rule_refuses_a_different_run_and_names_the_field(
    envelope_grid, stepper
):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(envelope_grid, stepper, configuration={"gamma": 0.22})
    with pytest.raises(OutputError, match=r"configuration\.gamma"):
        resolve_metadata_write(one, two)


def test_overwrite_rule_overwrites_and_logs_when_forced(envelope_grid, stepper, caplog):
    one = make_metadata(envelope_grid, stepper)
    two = make_metadata(envelope_grid, stepper, configuration={"gamma": 0.22})
    with caplog.at_level("WARNING", logger=schema.__name__):
        assert resolve_metadata_write(one, two, force=True) is True
    assert "configuration.gamma" in caplog.text


@pytest.mark.parametrize("class_name", ["TabulatedHollowMode", "TabulatedMode"])
def test_tabulated_mode_metadata_is_reproducible(class_name, envelope_grid, stepper):
    table = modes.IndexTable.from_wavelength(
        [400e-9, 600e-9, 800e-9, 1000e-9],
        [1.001 + 1e-8j] * 4,
        area=[1e-10] * 4,
    )
    mode = getattr(modes, class_name)(
        15e-6, "Ar", 10.0, table=table, reference_wavelength=800e-9
    )
    record = ModeRecord.from_mode(mode)
    metadata = make_metadata(envelope_grid, stepper, modes=(record,))
    back = RunMetadata.from_json(metadata.to_json())
    assert metadata.configuration_hash() == back.configuration_hash()
    assert (
        serialised(record.configuration)["table"]["neff_values"]["dtype"]
        == "complex128"
    )
    script = f"""
import numpy as np
from solprop.modes import IndexTable, {class_name}
from solprop.output.schema import ModeRecord, RunMetadata
metadata = RunMetadata.from_json({metadata.to_json()!r})
table = IndexTable.from_wavelength(
    [400e-9, 600e-9, 800e-9, 1000e-9], [1.001 + 1e-8j] * 4, area=[1e-10] * 4)
mode = {class_name}(15e-6, "Ar", 10., table=table, reference_wavelength=800e-9)
from dataclasses import replace
print(replace(metadata, modes=(ModeRecord.from_mode(mode),)).configuration_hash())
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ) | {"PYTHONHASHSEED": "3726"},
    )
    assert child.stdout.strip() == metadata.configuration_hash()


def test_rectangular_and_stepindex_records_round_trip_and_hash_alike(
    envelope_grid, stepper
):
    rectangular = RectMode(50e-6, 10e-6, "Ar", 5.0, cladding=1.45)
    strand = StepIndexMode(1.25e-6, profile_wavelength=835e-9)
    records = (ModeRecord.from_mode(rectangular), ModeRecord.from_mode(strand))
    assert records[0].kind == "solprop.modes.rectangular.RectMode"
    assert records[1].kind == "solprop.modes.stepindex.StepIndexMode"
    assert records[0].effective_area == pytest.approx(rectangular.aeff(0.0))
    assert records[1].effective_area == pytest.approx(strand.aeff(0.0))
    metadata = make_metadata(envelope_grid, stepper, modes=records)
    back = RunMetadata.from_json(metadata.to_json())
    assert metadata.configuration_hash() == back.configuration_hash()
    script = f"""
from dataclasses import replace
from solprop.modes import RectMode, StepIndexMode
from solprop.output.schema import ModeRecord, RunMetadata
metadata = RunMetadata.from_json({metadata.to_json()!r})
records = (
    ModeRecord.from_mode(RectMode(50e-6, 10e-6, "Ar", 5., cladding=1.45)),
    ModeRecord.from_mode(StepIndexMode(1.25e-6, profile_wavelength=835e-9)),
)
print(replace(metadata, modes=records).configuration_hash())
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ) | {"PYTHONHASHSEED": "3826"},
    )
    assert child.stdout.strip() == metadata.configuration_hash()


def test_a_metal_wall_record_round_trips_and_hashes_alike_across_processes(
    envelope_grid, stepper
):
    metal = ModeRecord.from_mode(
        RectMode(50e-6, 10e-6, "Ar", 5.0, cladding=aluminium())
    )
    document = json.dumps(serialised(metal))
    # The stitched table travels with the mode as its complete spline polynomials: 94
    # 085 bytes measured here against 1290 for the same mode with a constant wall. A
    # reporting threshold, not a record-format limit.
    assert 20_000 < len(document.encode("utf-8")) < 200_000
    # No spline is recorded by its repr any more, neither here nor for a silica-clad
    # capillary mode, so a metal-wall record hashes alike in another process.
    assert document.count('"reproducible": false') == 0
    silica = json.dumps(
        serialised(
            ModeRecord.from_mode(
                MarcatiliMode(125e-6, "He", 1.0, cladding=fused_silica())
            )
        )
    )
    assert silica.count('"reproducible": false') == 0
    metadata = make_metadata(envelope_grid, stepper, modes=(metal,))
    assert (
        RunMetadata.from_json(metadata.to_json()).configuration_hash()
        == metadata.configuration_hash()
    )
    script = f"""
from dataclasses import replace
from solprop.modes import RectMode
from solprop.output.schema import ModeRecord, RunMetadata
from solprop.physdata.metals import aluminium
metadata = RunMetadata.from_json({metadata.to_json()!r})
record = ModeRecord.from_mode(RectMode(50e-6, 10e-6, "Ar", 5., cladding=aluminium()))
print(replace(metadata, modes=(record,)).configuration_hash())
"""
    child = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env=dict(os.environ) | {"PYTHONHASHSEED": "3926"},
    )
    assert child.stdout.strip() == metadata.configuration_hash()


def test_ten_thousand_row_table_metadata_size(envelope_grid, stepper):
    omega = np.linspace(1e15, 5e15, 10000)
    table = IndexTable(
        omega, 1 - 1e25 / omega**2 + 1j * 1e36 / omega**3, area=np.full(10000, 1e-10)
    )
    mode = TabulatedHollowMode(15e-6, table=table, reference_wavelength=800e-9)
    metadata = make_metadata(
        envelope_grid, stepper, modes=(ModeRecord.from_mode(mode),)
    )
    size = len(metadata.to_json().encode("utf-8"))
    # A reporting threshold, not a record-format limit.
    assert 500_000 < size < 5_000_000


def test_cubic_spline_metadata_survives_independent_construction(
    envelope_grid, stepper
):
    knots = np.linspace(0.0, 1.0, 5)
    first = CubicSpline(knots, np.sin(knots), extrapolate=False)
    second = CubicSpline(knots, np.sin(knots), extrapolate=False)
    one = make_metadata(envelope_grid, stepper, configuration={"cladding": first})
    two = make_metadata(envelope_grid, stepper, configuration={"cladding": second})
    assert one.configuration_hash() == two.configuration_hash()
    assert (
        RunMetadata.from_json(one.to_json()).configuration_hash()
        == one.configuration_hash()
    )
    record = serialise(first)
    assert isinstance(record, dict)
    assert "reproducible" not in record
    script = (
        "import json, numpy as np; from scipy.interpolate import CubicSpline; "
        "from solprop.output.schema import serialise; "
        "x = np.linspace(0.0, 1.0, 5); "
        "print(json.dumps(serialise(CubicSpline(x, np.sin(x), extrapolate=False))))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == record


@pytest.mark.parametrize("member", ["x", "c", "axis", "extrapolate"])
def test_cubic_spline_metadata_hash_detects_polynomial_changes(
    envelope_grid, stepper, member
):
    knots = np.linspace(0.0, 1.0, 5)
    spline = CubicSpline(knots, np.sin(knots), extrapolate=False)
    before = make_metadata(
        envelope_grid, stepper, configuration={"cladding": spline}
    ).configuration_hash()
    if member in ("x", "c"):
        getattr(spline, member).flat[0] += 0.01
    else:
        setattr(spline, member, 1 if member == "axis" else True)
    after = make_metadata(
        envelope_grid, stepper, configuration={"cladding": spline}
    ).configuration_hash()
    assert after != before


def test_raman_tensor_identity_survives_metadata_round_trip_and_controls_resume(
    envelope_grid, stepper
):
    # optmat's H2 response disperses its strengths to the pump (an 800 nm carrier)
    # and its lines shift with density (a constant fill of 1e26 m^-3, 3.7 amagat,
    # inside optmat's validated range).
    selection = MaterialSelection(carrier_wavelength=800e-9)
    response = response_for("H2", selection=selection, density=1e26)
    original = make_metadata(envelope_grid, stepper, configuration={"raman": response})
    restored = RunMetadata.from_json(original.to_json())
    assert restored.configuration_hash() == original.configuration_hash()
    assert not resolve_metadata_write(restored, original)
    for changed in (
        RamanResponse(models=response.models),
        dataclasses.replace(
            response,
            tensor_terms=(
                dataclasses.replace(response.tensor_terms[0], contraction="isotropic"),
                *response.tensor_terms[1:],
            ),
        ),
    ):
        other = make_metadata(envelope_grid, stepper, configuration={"raman": changed})
        assert original.configuration_hash() != other.configuration_hash()
        with pytest.raises(OutputError, match=r"configuration\.raman"):
            resolve_metadata_write(restored, other)
    assert original.schema_version == 25


def test_spatial_metadata_of_schema_23_names_its_flux_model(envelope_grid, stepper):
    """From schema 23 (SEP 34) a spatial record carries its plane measure's model."""
    spatial = {"kind": "radial"}
    with pytest.raises(OutputError, match="must name its flux_model"):
        make_metadata(envelope_grid, stepper, simulation_kind="radial", spatial=spatial)
    # A record written before the measure existed is still read.
    older = make_metadata(
        envelope_grid,
        stepper,
        simulation_kind="radial",
        spatial=spatial,
        schema_version=22,
    )
    assert older.spatial == spatial
