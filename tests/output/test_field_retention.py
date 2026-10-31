# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Field retention in the two handlers: layout, storage, refusals and readers (SEP 22).

Storage is lossless, so a stored field is compared exactly with what went in (class P
of ``docs/dev/testing.md`` section 3), and so is every count and index.
"""

import dataclasses
import json
from typing import cast

import h5py
import numpy as np
import pytest

from solprop import fourier
from solprop.exceptions import OutputError
from solprop.output import HDF5Output, MemoryOutput, Output, open_file
from solprop.output.hdf5 import resume
from solprop.output.schema import (
    INLINE_ARRAY_LIMIT,
    ArrayDigest,
    FieldRetention,
    RunOutcome,
    SavedPlane,
    TerminalCheckpoint,
    _example_metadata,
    field_retention_refusal,
    recorded_input,
    serialise,
)
from solprop.stats import collect, default_spec

N_PLANES = 5


@pytest.fixture
def metadata():
    return _example_metadata()


def field(metadata, value: float) -> np.ndarray:
    """Return a distinct internal spectrum per plane, so a misplaced row is caught."""
    n_omega = metadata.grid.n_omega
    return np.full((1, 1, n_omega), value, complex) * np.exp(
        1j * np.linspace(0.0, 1.0, n_omega)
    )


def planes(metadata, retention: FieldRetention, *, noise: bool = False):
    """Return the run's planes, a field and a noise phase only where they are kept."""
    built = []
    for index in range(N_PLANES):
        keep = retention.keeps(index)
        built.append(
            SavedPlane(
                z=0.1 * index,
                field=field(metadata, index + 1.0) if keep else None,
                stats={"energy": np.full((1, 1), float(index))},
                noise_phase=np.full((1, 1, metadata.grid.n_omega), float(index))
                if keep and noise
                else None,
            )
        )
    return built


RULES = [
    FieldRetention(),
    FieldRetention("planes", (0, -1)),
    FieldRetention("planes", (1, 3)),
    FieldRetention("none"),
]


def store(handler, metadata, retention, *, noise=False):
    resolved = retention.resolved(N_PLANES)
    handler.open(metadata)
    handler.reserve(N_PLANES)
    handler.retain(resolved, (1, 1, metadata.grid.n_omega))
    for plane in planes(metadata, resolved, noise=noise):
        handler.save(plane)
    return resolved


# --------------------------------------------------------------- the rule itself


@pytest.mark.parametrize(
    ("fields", "field_planes", "message"),
    [
        ("some", (), "fields must be none, planes or all"),
        ("planes", (0.5,), "integer saved-plane indices"),
        ("planes", 3, "must be a sequence"),
        ("planes", (), "required exactly when fields='planes'"),
        ("none", (0,), "required exactly when fields='planes'"),
        ("planes", (True,), "integer saved-plane indices"),
    ],
)
def test_an_invalid_rule_is_refused_by_the_one_validator(fields, field_planes, message):
    assert message in str(field_retention_refusal(fields, field_planes))
    assert field_retention_refusal("planes", [np.int64(0), -1]) is None


@pytest.mark.parametrize(
    ("fields", "field_planes", "message"),
    [
        ("planes", (), "required exactly when fields='planes'"),
        ("none", (0,), "required exactly when fields='planes'"),
        ("planes", (True,), "integer saved-plane indices"),
    ],
)
def test_a_rule_of_the_right_types_is_still_validated(fields, field_planes, message):
    with pytest.raises(OutputError, match=message):
        FieldRetention(fields, field_planes)


def test_a_rule_resolves_negative_indices_once_and_refuses_one_outside():
    rule = FieldRetention("planes", (-1, 0, 4, 2))
    resolved = rule.resolved(5)
    assert resolved.field_planes == (0, 2, 4)
    assert [resolved.keeps(index) for index in range(5)] == [1, 0, 1, 0, 1]
    assert resolved.kept(5) == 3
    assert FieldRetention().kept(5) == 5
    assert FieldRetention("none").resolved(5).kept(5) == 0
    with pytest.raises(OutputError, match="outside a run of 5 saved planes"):
        FieldRetention("planes", (5,)).resolved(5)
    with pytest.raises(OutputError, match="outside a run of 5 saved planes"):
        FieldRetention("planes", (-6,)).resolved(5)
    assert FieldRetention("planes", (0, 4)).describe() == (
        "fields='planes' (saved planes 0, 4)"
    )
    assert FieldRetention("none").describe() == "fields='none'"


# ------------------------------------------------------------------ the handlers


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.describe())
@pytest.mark.parametrize("noise", [False, True])
def test_both_handlers_store_exactly_the_kept_fields(tmp_path, metadata, rule, noise):
    memory = MemoryOutput()
    resolved = store(memory, metadata, rule, noise=noise)
    path = tmp_path / "run.h5"
    with HDF5Output(path) as written:
        store(written, metadata, rule, noise=noise)
    kept = [index for index in range(N_PLANES) if resolved.keeps(index)]
    with open_file(path) as stored:
        for handler in (memory, stored):
            assert handler.n_saved == N_PLANES
            assert handler.retention == resolved
            assert handler.field_shape == (1, 1, metadata.grid.n_omega)
            # Lossless storage: the counts, indices and stored rows are exact.
            assert np.array_equal(handler["field_planes"], kept)
            assert np.array_equal(handler["z"], 0.1 * np.arange(N_PLANES))
            assert np.array_equal(
                handler["stats/energy"][:, 0, 0], np.arange(N_PLANES, dtype=float)
            )
            for index in kept:
                expected = field(metadata, index + 1.0)
                assert np.array_equal(handler.plane_field(index, "fft"), expected)
                assert np.array_equal(
                    handler.plane_field(index),
                    fourier.spectrum_to_optics(expected, kind=metadata.grid.kind),
                )
            for index in set(range(N_PLANES)) - set(kept):
                with pytest.raises(OutputError, match=rf"saved plane {index} kept no"):
                    handler.plane_field(index)
            with pytest.raises(OutputError, match="out of range"):
                handler.plane_field(N_PLANES)
            if resolved.fields == "all":
                assert handler["field"].shape == (N_PLANES, 1, 1, metadata.grid.n_omega)
                assert ("noise_phase" in handler) is noise
            else:
                assert "field" not in handler
                assert "noise_phase" not in handler
                with pytest.raises(OutputError, match=resolved.fields):
                    handler.fields()
    # The file holds what the rule keeps and nothing else.
    with h5py.File(path, "r") as file:
        assert str(file.attrs["fields"]) == resolved.fields
        assert tuple(np.asarray(file.attrs["field_shape"])) == (
            1,
            1,
            metadata.grid.n_omega,
        )
        if resolved.fields == "none":
            assert "field" not in file and "noise_phase" not in file
        else:
            assert cast("h5py.Dataset", file["field"]).shape[0] == len(kept)
            assert ("noise_phase" in file) is noise
        assert ("field_planes" in file) is (resolved.fields == "planes")


@pytest.mark.parametrize("rule", RULES, ids=lambda rule: rule.describe())
def test_a_memory_store_writes_the_file_the_other_handler_writes(
    tmp_path, metadata, rule
):
    memory = MemoryOutput()
    resolved = store(memory, metadata, rule, noise=True)
    returned = memory.planes()
    assert [plane.field is None for plane in returned] == [
        not resolved.keeps(index) for index in range(N_PLANES)
    ]
    memory.to_file(tmp_path / "copy.h5")
    with open_file(tmp_path / "copy.h5") as stored:
        assert stored.retention == resolved
        assert np.array_equal(stored["field_planes"], memory["field_planes"])
        for index in memory["field_planes"]:
            assert np.array_equal(
                stored.plane_field(int(index)), memory.plane_field(int(index))
            )


def test_a_memory_store_allocates_only_the_kept_fields(metadata):
    memory = MemoryOutput()
    store(memory, metadata, FieldRetention("planes", (1, 3)))
    # Two field rows, however many planes: the block is sized by the rule.
    assert memory._fields.shape[0] == 2
    none = MemoryOutput()
    store(none, metadata, FieldRetention("none"))
    assert none._fields.size == 0
    every = MemoryOutput(block=2)
    every.open(metadata)
    every.retain(FieldRetention(), (1, 1, metadata.grid.n_omega))
    for plane in planes(metadata, FieldRetention()):
        every.save(plane)
    # Grown by doubling past its block, as before.
    assert every._fields.shape[0] == 8


def test_held_bytes_charge_the_kept_fields_only():
    memory = MemoryOutput()
    assert memory.held_bytes(17, 1000) == 17000
    assert memory.held_bytes(17, 1000, kept=None, field_bytes=900) == 17000
    assert memory.held_bytes(17, 1000, kept=2, field_bytes=900) == 17 * 100 + 2 * 900
    assert memory.held_bytes(17, 1000, kept=0, field_bytes=900) == 1700


def test_a_plane_that_breaks_the_rule_is_refused_by_name(metadata):
    memory = MemoryOutput()
    memory.open(metadata)
    with pytest.raises(OutputError, match=r"carries no field.*fields='all'"):
        memory.save(SavedPlane(z=0.0, field=None))
    memory.retain(FieldRetention("planes", (1,)), (1, 1, metadata.grid.n_omega))
    with pytest.raises(OutputError, match=r"plane 0 carries a field.*saved planes 1"):
        memory.save(SavedPlane(z=0.0, field=field(metadata, 1.0)))
    memory.save(SavedPlane(z=0.0, field=None))
    with pytest.raises(OutputError, match=r"plane 1 carries no field"):
        memory.save(SavedPlane(z=0.1, field=None))
    memory.save(SavedPlane(z=0.1, field=field(metadata, 2.0)))
    # The rule is fixed once planes are held.
    with pytest.raises(OutputError, match="continue it with the same fields"):
        memory.retain(FieldRetention("none"), (1, 1, metadata.grid.n_omega))
    memory.retain(
        FieldRetention("planes", (1,)), (1, 1, metadata.grid.n_omega)
    )  # the same rule again


def test_a_kept_plane_after_field_free_ones_fixes_the_noise_phase_and_shape(metadata):
    n_omega = metadata.grid.n_omega
    memory = MemoryOutput()
    memory.open(metadata)
    memory.retain(FieldRetention("planes", (1, 2)), (1, 1, n_omega))
    memory.save(SavedPlane(z=0.0, field=None))
    phase = np.zeros((1, 1, n_omega))
    memory.save(SavedPlane(z=0.1, field=field(metadata, 1.0), noise_phase=phase))
    with pytest.raises(OutputError, match="does not carry a noise phase"):
        memory.save(SavedPlane(z=0.2, field=field(metadata, 2.0)))
    with pytest.raises(OutputError, match="field shape"):
        memory.save(SavedPlane(z=0.2, field=np.ones((1, 1, 4), complex)))
    with pytest.raises(OutputError, match="reports statistics"):
        memory.save(
            SavedPlane(
                z=0.2,
                field=field(metadata, 2.0),
                noise_phase=phase,
                stats={"energy": np.ones(1)},
            )
        )


def test_a_field_free_plane_carries_no_noise_phase():
    with pytest.raises(OutputError, match="field-free plane carries no noise phase"):
        SavedPlane(z=0.0, field=None, noise_phase=np.zeros((1, 1, 4)))
    plane = SavedPlane(z=0.0, field=None, stats={"energy": np.ones(1)})
    assert plane.shape is None
    assert "no field" in repr(plane) and "no field" in str(plane)


def test_statistics_cannot_be_formed_from_a_field_free_plane(metadata):
    from solprop.grid import EnvelopeGrid  # noqa: PLC0415

    grid = EnvelopeGrid.from_limits(1.0, 835e-9, (400e-9, 2e-6), 2e-12)
    with pytest.raises(OutputError, match="kept no field"):
        collect(default_spec("mode_averaged"), grid, SavedPlane(z=0.0, field=None))


def test_a_checkpoint_always_carries_the_field(metadata):
    with pytest.raises(OutputError, match="carries the field"):
        TerminalCheckpoint(SavedPlane(z=0.0, field=None), 0)


def test_a_field_free_file_resumes_only_from_its_checkpoint(tmp_path, metadata):
    path = tmp_path / "run.h5"
    resolved = FieldRetention("none").resolved(N_PLANES)
    with HDF5Output(path) as written:
        written.open(metadata)
        written.retain(resolved, (1, 1, metadata.grid.n_omega))
        for plane in planes(metadata, resolved)[:3]:
            written.save(plane)
    state = resume(path)
    assert state.n_saved == 3 and state.plane.field is None
    assert state.plane.z == pytest.approx(0.2)
    with HDF5Output(path, append=True) as again:
        again.open(metadata)
        # A continuing run with the same rule is accepted; another one is refused.
        again.retain(resolved, (1, 1, metadata.grid.n_omega))
        with pytest.raises(OutputError, match="continue it with the same fields"):
            again.retain(FieldRetention(), (1, 1, metadata.grid.n_omega))
        checkpoint = TerminalCheckpoint(
            SavedPlane(z=0.25, field=field(metadata, 9.0)), 3, landed=True
        )
        again.record_outcome(RunOutcome("failed", 0.25), checkpoint)
    state = resume(path)
    assert state.checkpoint is not None and state.plane.field is not None
    assert np.array_equal(state.plane.field, field(metadata, 9.0))


# ----------------------------------------------------------- record helpers


def test_an_array_above_the_inline_limit_is_recorded_by_its_digest():
    small = np.zeros(4)
    assert recorded_input(small) is small
    large = np.zeros(INLINE_ARRAY_LIMIT + 1, complex)
    digest = recorded_input(large)
    assert isinstance(digest, ArrayDigest)
    assert digest == ArrayDigest.of(large.copy())
    assert (digest.dtype, digest.shape) == ("complex128", (INLINE_ARRAY_LIMIT + 1,))
    changed = large.copy()
    changed[12345] = 1e-300
    assert ArrayDigest.of(changed).sha256 != digest.sha256
    # Byte order does not change the digest of the same numbers.
    numbers = np.arange(8.0)
    assert ArrayDigest.of(numbers.astype(">f8")) == ArrayDigest.of(numbers)
    record = serialise(digest)
    assert isinstance(record, dict)
    assert record["__class__"] == "solprop.output.schema.ArrayDigest"
    assert len(json.dumps(record)) < 300


@dataclasses.dataclass(frozen=True)
class _Recorded:
    """A recorded dataclass with defaulted fields."""

    value: float
    added: float = 1.5
    listed: tuple[int, ...] = ()


def test_every_field_is_recorded_at_its_default_too():
    # No field is left out of a record, whatever it holds.
    assert serialise(_Recorded(1.0)) == {
        "__class__": f"{__name__}._Recorded",
        "value": 1.0,
        "added": 1.5,
        "listed": [],
    }


def test_a_field_free_store_grows_its_planes_and_never_a_field_block(metadata):
    memory = MemoryOutput(block=1)
    memory.open(metadata)
    memory.retain(FieldRetention("none"), (1, 1, metadata.grid.n_omega))
    for plane in planes(metadata, FieldRetention("none")):
        memory.save(plane)
    assert memory.n_saved == N_PLANES and memory._fields.size == 0
    assert np.array_equal(memory["z"], 0.1 * np.arange(N_PLANES))
    # The noise-phase accessor answers None for a run without a noise model, and is
    # refused like the field for one that kept its phase on some planes only.
    assert memory.noise_phase is None
    noisy = MemoryOutput()
    store(noisy, metadata, FieldRetention("planes", (1,)), noise=True)
    with pytest.raises(OutputError, match="noise_phase"):
        noisy.noise_phase  # noqa: B018 - the property is what is tested


def test_a_figure_labels_the_columns_of_a_run_that_kept_no_field(metadata):
    from solprop.plotting.style import field_modes  # noqa: PLC0415

    memory = MemoryOutput()
    store(memory, metadata, FieldRetention("none"))
    assert field_modes(memory) == 1

    class _Plain(Output):
        """An output that is not a PlaneStore: the field's own shape answers."""

        def __getitem__(self, key):
            return np.zeros((2, 3, 1, 4))

    assert field_modes(_Plain()) == 3  # pyright: ignore[reportAbstractUsage]
