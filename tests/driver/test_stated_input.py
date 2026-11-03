# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""An array input recorded by the identity its caller states: nothing digests it.

The identity is taken on trust, so what these tests pin down is both sides of that: no
digest is taken, a run's hash follows the statement and not the bytes, and so a wrong
statement passes a resume and a scan's cache. Runs compare structurally, by the record.
"""

import numpy as np
import pytest

from driver.test_field_free_planes import cartesian
from driver.test_runner import capillary, inputs, stored
from solprop.driver import Simulation
from solprop.driver.executors import Local
from solprop.driver.scan import Scan, run_scan
from solprop.exceptions import DriverError, OutputError
from solprop.output import open_file
from solprop.output import schema as output_schema
from solprop.output.compatibility import compare_records
from solprop.output.schema import ArrayDigest, StatedInput, serialise
from solprop.physdata import Mixture
from solprop.pulses import GaussianPulse


@pytest.fixture
def no_digest(monkeypatch):
    """Fail any digest, and record every array input by more than the inline limit."""

    def refuse(values: np.ndarray) -> ArrayDigest:
        msg = f"a digest of {values.shape} was taken"
        raise AssertionError(msg)

    monkeypatch.setattr(output_schema.ArrayDigest, "of", refuse)
    monkeypatch.setattr(output_schema, "INLINE_ARRAY_LIMIT", 0)


@pytest.mark.usefixtures("no_digest")
def test_a_stated_input_is_recorded_for_its_digest_and_never_read():
    simulation = capillary("numpy_ref")
    field = inputs(simulation)[0]
    derived = simulation.with_input(
        field, input_identity="delay 12 fs", configuration={"delay": 12e-15}
    )
    recorded = derived.metadata().configuration["initial"]
    assert recorded == StatedInput("delay 12 fs", "complex128", field.shape)


def test_the_hash_follows_the_statement_and_not_the_bytes():
    simulation = capillary("numpy_ref")
    first, second = inputs(simulation)[:2]
    hashes = [
        simulation.with_input(field, input_identity=name)
        .metadata()
        .configuration_hash()
        for field, name in ((first, "a"), (second, "a"), (first, "b"))
    ]
    assert hashes[0] == hashes[1]
    assert hashes[0] != hashes[2]
    # A statement is not the array's own record, under either of its forms.
    assert hashes[0] != simulation.with_input(first).metadata().configuration_hash()


def test_without_a_statement_a_large_array_is_digested_as_ever(monkeypatch):
    simulation = capillary("numpy_ref")
    field = inputs(simulation)[0]
    monkeypatch.setattr(output_schema, "INLINE_ARRAY_LIMIT", 0)
    recorded = simulation.with_input(field).metadata().configuration["initial"]
    assert recorded == ArrayDigest.of(field)


def test_the_dtype_and_the_shape_are_those_given():
    simulation = capillary("numpy_ref")
    field = inputs(simulation)[0].astype(">c16")
    recorded = (
        simulation.with_input(field, input_identity="a")
        .metadata()
        .configuration["initial"]
    )
    assert recorded == StatedInput("a", "complex128", field.shape)


@pytest.mark.parametrize("backend", ["numpy_ref", "native"])
@pytest.mark.usefixtures("no_digest")
def test_a_runner_digests_no_input_when_each_is_stated(backend):
    """The parent is not digested either, to check the caller's keys against."""
    simulation = capillary(backend)
    fields = inputs(simulation)
    with simulation.runner() as runner:
        outputs = [
            stored(
                runner.run(
                    field,
                    configuration={"point": index},
                    input_identity=f"point {index}",
                    progress="none",
                )
            )
            for index, field in enumerate(fields)
        ]
    records = [output.metadata for output in outputs]
    assert all(record is not None for record in records)
    assert [
        record.configuration["initial"].identity  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
        for record in records
    ] == ["point 0", "point 1", "point 2"]
    assert all(output.n_saved > 1 for output in outputs)


def test_a_runner_digests_each_input_once_and_never_its_parent(monkeypatch):
    """Without a statement: a parent's digest would read a placeholder at full size."""
    simulation = capillary("numpy_ref")
    fields = inputs(simulation)
    monkeypatch.setattr(output_schema, "INLINE_ARRAY_LIMIT", 0)
    digest = ArrayDigest.of
    digested: list[tuple[int, ...]] = []

    def counting(values: np.ndarray) -> ArrayDigest:
        digested.append(values.shape)
        return digest(values)

    monkeypatch.setattr(output_schema.ArrayDigest, "of", counting)
    with simulation.runner() as runner:
        for index, field in enumerate(fields):
            runner.run(field, configuration={"point": index}, progress="none")
    assert digested == [field.shape for field in fields]


def test_a_runner_reads_a_small_stated_input_without_a_copy(monkeypatch):
    """The record keeps no array, so even one that would go inline is not copied."""
    simulation = cartesian("numpy_ref")
    field = 0.9 * simulation._field0
    with simulation.runner() as runner:
        load = runner.backend.set_initial
        shared: list[bool] = []

        def spy(given, noise, *, z=0.0, resume=False) -> None:
            shared.append(np.shares_memory(given, field))
            load(given, noise, z=z, resume=resume)

        monkeypatch.setattr(runner.backend, "set_initial", spy)
        runner.run(field, input_identity="a", progress="none")
    assert shared == [True]


@pytest.mark.parametrize(
    ("initial", "identity", "message"),
    [
        (
            GaussianPulse(wavelength=800e-9, fwhm=30e-15, energy=1e-6),
            "a",
            "array input",
        ),
        (None, "", "is empty"),
    ],
)
def test_an_identity_of_nothing_or_of_a_pulse_is_refused(initial, identity, message):
    simulation = capillary("numpy_ref")
    given = simulation._field0 if initial is None else initial
    with pytest.raises(DriverError, match=message):
        simulation.with_input(given, input_identity=identity)
    with simulation.runner() as runner, pytest.raises(DriverError, match=message):
        runner.run(given, input_identity=identity)


def test_a_stated_input_resumes_by_its_statement_alone(tmp_path):
    """A wrong statement passes, as documented; a different one is refused by name."""
    simulation = capillary("numpy_ref")
    first, second = inputs(simulation)[:2]
    path = tmp_path / "run.h5"
    simulation.with_input(first, input_identity="a").run(output=path, progress="none")
    assert open_file(path).n_saved > 1
    # The same statement of another array: nothing here can tell.
    simulation.with_input(second, input_identity="a").run(
        output=path, resume=True, progress="none"
    )
    with pytest.raises(OutputError, match="initial 'stated a' -> 'stated b'"):
        simulation.with_input(first, input_identity="b").run(
            output=path, resume=True, progress="none"
        )
    # Nor does the array's own record continue a stated file.
    with pytest.raises(OutputError, match=r"initial .*'stated a' -> "):
        simulation.with_input(first).run(output=path, resume=True, progress="none")


def test_the_leaf_rule_compares_a_statement_whole_and_names_it():
    stated = serialise(StatedInput("a", "complex128", (1, 2)))
    assert isinstance(stated, dict)
    assert stated["__class__"] == "solprop.output.schema.StatedInput"
    one = {"configuration": {"initial": stated}}
    two = {"configuration": {"initial": dict(stated, identity="b")}}
    comparison = compare_records(one, two)
    assert not comparison.continues
    assert "'stated a' -> 'stated b'" in comparison.summary()
    assert compare_records(one, {"configuration": {"initial": dict(stated)}}).continues


def _point(point) -> Simulation:
    """Return a scan point: one array input, stated as the point's power."""
    simulation = capillary("numpy_ref")
    return simulation.with_input(
        point.values["power"] * simulation._field0,
        input_identity=f"power {point.values['power']}",
    )


def test_a_scan_s_cache_keeps_a_point_stated_alike(tmp_path):
    scan = Scan("stated", power=[0.9, 1.1])
    path = tmp_path / "scan.h5"
    first = run_scan(scan, _point, output=path, executor=Local())
    assert first.status.tolist() == ["done", "done"]
    # Computed already, and recognised by its statement alone.
    again = run_scan(scan, _point, output=path, executor=Local())
    assert again.status.tolist() == ["done", "done"]


def test_a_mixture_update_keeps_the_statement():
    simulation = capillary("numpy_ref")
    derived = simulation.with_input(inputs(simulation)[0], input_identity="a")
    updated = derived.with_mixture(Mixture.fill({"He": 1.0}, 1.5))
    recorded = updated.metadata().configuration["initial"]
    assert isinstance(recorded, StatedInput)
    assert recorded.identity == "a"
