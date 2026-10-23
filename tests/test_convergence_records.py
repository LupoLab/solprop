# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The convergence records themselves: their schema, their provenance and their gate.

``tests/convergence_records.py`` is the mechanism; this file is what keeps the stored
files honest. Three properties matter and none of them is checked by the studies that
write the records:

* the files parse and describe what they claim to describe;
* every row names a commit on this tree's own line of history, and a test that still
  exists -- so a renamed writer or a row from an abandoned branch is caught here rather
  than surviving as a number on the accuracy page that nothing reproduces;
* the write gate refuses everything it is supposed to refuse.
"""

import dataclasses
import json
import math
import os
import subprocess

import pytest

import convergence_records as cr
from solprop.backends.registry import BACKENDS, available

RETIRED_BACKENDS = frozenset({"numba_cuda"})
"""Backends whose rows stay as evidence after the backend itself was deleted."""

TOP_LEVEL = {
    "schema",
    "study",
    "title",
    "case",
    "axis",
    "columns",
    "luna",
    "writes",
    "rows",
}

# Below this many collected files the run is a hand-picked selection rather than the
# suite, and the closure check below has nothing to be complete about. The same guard,
# and the same number, as `tests/test_tiers.py`.
SUITE_FILES = 5


@pytest.fixture(params=cr.studies())
def stored(request):
    """Every stored record in turn, so a failure names the study."""
    return cr.load(request.param)


def git(*arguments):
    """Return one line of read-only git output for this checkout, or None."""
    return cr._git(*arguments)


# ------------------------------------------------------------------ the schema


def test_there_is_at_least_one_record():
    """Otherwise every parametrised test below would pass over an empty set."""
    assert cr.studies()


def test_write_stamp_records_the_backends_actual_planner_setting(monkeypatch):
    monkeypatch.setenv("SOLPROP_FFTW_PLAN", "estimate")
    monkeypatch.setenv("SOLPROP_FFTW_PLANNER", "unused-spelling")
    assert cr._write_stamp("numba_cpu")["fftw_planner"] == "estimate"


def test_the_record_has_the_fields_the_figures_read(stored):
    assert set(stored) == TOP_LEVEL
    assert stored["schema"] == cr.SCHEMA_VERSION
    assert stored["title"] and stored["study"]
    assert set(stored["axis"]) == {"name", "unit", "label"}
    assert stored["rows"] and stored["writes"]


def test_every_column_states_a_unit_and_a_reason(stored):
    for column in stored["columns"]:
        assert set(column) == {"name", "unit", "why", "bound", "rtol", "atol"}
        assert column["unit"], column["name"]
        # docs/dev/testing.md section 3: a tolerance without a justification is a
        # defect, so the justification travels in the file beside the number.
        assert len(column["why"]) > 20, column["name"]
        # `None` is recorded-not-compared; zero is compared exactly, which is what
        # a column holding an input rather than a measurement wants.
        assert column["rtol"] is None or column["rtol"] >= 0.0
        assert column["atol"] >= 0.0


def test_every_row_carries_every_column_and_nothing_else(stored):
    names = {column["name"] for column in stored["columns"]}
    for row in stored["rows"]:
        assert set(row) == names | {"series", "axis", "write", "written_by"}
        assert isinstance(row["axis"], int | str)
        for name in names:
            if row[name] is None:
                continue
            assert math.isfinite(float(row[name])), (row["series"], row["axis"], name)


def test_no_two_rows_share_a_key(stored):
    keys = [(row["series"], repr(row["axis"])) for row in stored["rows"]]
    assert len(set(keys)) == len(keys)


def test_every_row_points_at_a_write_that_names_a_machine(stored):
    for row in stored["rows"]:
        write = stored["writes"][int(row["write"])]
        assert write["machine_id"] and write["os"]
        assert write["commit"] is None or len(write["commit"]) == 40
    # Every write is referenced: an orphan is a row that was deleted without its stamp.
    assert {int(row["write"]) for row in stored["rows"]} == set(
        range(len(stored["writes"]))
    )


def test_a_luna_quantity_names_the_witness_it_came_from(stored):
    """No record invents a Luna number: each is copied from a stored witness."""
    luna = stored["luna"]
    if luna is None:
        return
    assert len(luna["commit"]) == 40
    source = cr.REPO / "tests" / "data" / luna["source"]
    assert source.exists()


def test_the_file_on_disk_is_formatted_the_way_the_writer_writes_it(stored):
    """So that a regeneration is a diff of numbers, never of whitespace or order."""
    path = cr.path_of(stored["study"])
    text = json.dumps(stored, indent=1, sort_keys=True, allow_nan=False) + "\n"
    assert path.read_text() == text


# -------------------------------------------------------------- the provenance


def test_every_recorded_commit_is_on_this_tree_s_line_of_history(stored):
    """A row measured on an unrelated branch is a number nothing here reproduces.

    Asserted both ways round rather than "is an ancestor of HEAD": under ``git bisect``
    HEAD legitimately predates the record, and on a pull request HEAD is a merge commit
    the record is an ancestor of. What it still rejects is a record from a branch this
    one never touched.
    """
    if git("rev-parse", "HEAD") is None:
        pytest.skip("not a git checkout")
    if git("rev-parse", "--is-shallow-repository") == "true":
        pytest.skip("a shallow clone cannot answer an ancestry question")
    for write in stored["writes"]:
        commit = write["commit"]
        assert commit is not None, stored["study"]
        present = subprocess.run(
            ["git", "-C", str(cr.REPO), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
        )
        if present.returncode != 0:
            pytest.skip(f"{commit[:12]} is not in this checkout (squashed or rebased)")
        assert _related(commit, "HEAD"), f"{stored['study']}: {commit[:12]}"


def _related(one, other):
    def ancestor(first, second):
        return (
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(cr.REPO),
                    "merge-base",
                    "--is-ancestor",
                    first,
                    second,
                ],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )

    return ancestor(one, other) or ancestor(other, one)


def _writer_exists(node, collected):
    """Whether ``node`` is collected, or would be with an absent backend installed.

    An absent backend may be one coordinate of a compound parametrisation. Require
    a collected counterpart with every other coordinate unchanged, so a renamed
    case cannot pass merely because the same test function still exists.

    "Absent" covers both a backend that is registered but not installed here and one
    this package has deleted (:data:`RETIRED_BACKENDS`): the rows a deleted backend
    wrote are evidence of what it measured and stay, and nothing will ever collect
    their writer again.
    """
    if node in collected:
        return True
    base, _, parameter = node.partition("[")
    absent = (set(BACKENDS) | RETIRED_BACKENDS) - set(available())
    coordinates = parameter.rstrip("]").split("-")
    for index, coordinate in enumerate(coordinates):
        if coordinate not in absent:
            continue
        for present in available():
            counterpart = coordinates.copy()
            counterpart[index] = present
            if f"{base}[{'-'.join(counterpart)}]" in collected:
                return True
    return False


def test_every_row_was_written_by_a_test_that_still_exists(collected_tiers):
    """A renamed, deleted or re-parametrised writer leaves a row nothing reproduces.

    ``tests/conftest.py`` records every collected node id before ``-m`` and ``-k``
    deselect any of them, so this runs in the default lane and still sees the slow and
    benchmark writers. A writer parametrised on a backend that is registered but not
    installed here counts as existing (``_writer_exists``).
    """
    if len({node.split("::")[0] for node in collected_tiers}) < SUITE_FILES:
        pytest.skip("a hand-picked selection collects too little to be complete about")
    collected = set(collected_tiers)
    for study in cr.studies():
        for row in cr.load(study)["rows"]:
            assert _writer_exists(row["written_by"], collected), (
                study,
                row["written_by"],
            )


def test_a_writer_of_a_removed_backend_counts_as_existing():
    """A row of a deleted backend keeps its numbers without a writer to collect."""
    collected = {"tests/backends/test_x.py::test_y[numba_cpu]"}
    removed = next(iter(RETIRED_BACKENDS))
    assert removed not in BACKENDS
    assert _writer_exists(f"tests/backends/test_x.py::test_y[{removed}]", collected)
    assert not _writer_exists(f"tests/backends/test_z.py::test_y[{removed}]", collected)


def test_a_writer_of_an_absent_backend_counts_and_a_renamed_one_does_not():
    """The relaxation is exactly one parametrisation on an uninstalled backend."""
    collected = {"tests/backends/test_x.py::test_y[numba_cpu]"}
    absent = next(iter(set(BACKENDS) - set(available())), None)
    if absent is not None:
        assert _writer_exists(f"tests/backends/test_x.py::test_y[{absent}]", collected)
        assert not _writer_exists(
            f"tests/backends/test_z.py::test_y[{absent}]", collected
        )
    assert _writer_exists("tests/backends/test_x.py::test_y[numba_cpu]", collected)
    assert not _writer_exists("tests/backends/test_x.py::test_y[numpy_ref]", collected)
    assert not _writer_exists("tests/backends/test_x.py::test_gone", collected)


def test_absent_backend_writer_preserves_all_other_case_parameters(monkeypatch):
    """The optional-wheel lane still checks family, rate and test identity."""
    monkeypatch.setattr(
        "test_convergence_records.available", lambda: ("numpy_ref", "numba_cpu")
    )
    collected = {"tests/backends/test_x.py::test_y[envelope-numba_cpu-0.0]"}
    assert _writer_exists(
        "tests/backends/test_x.py::test_y[envelope-native-0.0]", collected
    )
    for node in (
        "tests/backends/test_x.py::test_y[field-native-0.0]",
        "tests/backends/test_x.py::test_y[envelope-native-1000.0]",
        "tests/backends/test_x.py::test_z[envelope-native-0.0]",
        "tests/backends/test_x.py::test_y[envelope-numpy_ref-0.0]",
    ):
        assert not _writer_exists(node, collected)


# ----------------------------------------------------------------- the gate


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """Point the writer at a scratch directory and pretend the tree is clean."""
    monkeypatch.setattr(cr, "RECORDS", tmp_path / "convergence")
    monkeypatch.setattr(cr, "_WRITTEN", set())
    monkeypatch.setattr(cr, "dirty_paths", lambda: [])
    monkeypatch.setenv(cr.WRITE_ENV, "1")
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    return tmp_path


def a_study(name="scratch"):
    return cr.Study(
        name=name,
        title="a scratch study",
        case={"what": "nothing at all"},
        axis={"name": "n", "unit": "1", "label": "rungs"},
        columns=(
            cr.Column(
                name="error",
                unit="1",
                why="a scratch column, compared at a tenth so the test is about the "
                "mechanism and not about a number",
                rtol=0.1,
            ),
        ),
    )


def rows(*pairs, series="only"):
    return [{"series": series, "axis": n, "error": e} for n, e in pairs]


def test_a_write_run_creates_the_file_and_a_read_run_agrees_with_it(elsewhere, request):
    study = a_study()
    cr.record(study, request=request, rows=rows((8, 1e-3), (16, 1e-4)))
    assert cr.path_of("scratch").exists()
    os.environ.pop(cr.WRITE_ENV)
    assert cr.record(study, request=request, rows=rows((8, 1e-3), (16, 1e-4)))["rows"]


def test_a_second_writer_merges_its_rows_and_keeps_the_others(elsewhere, request):
    study = a_study()
    cr.record(study, request=request, rows=rows((8, 1e-3)))
    cr.record(study, request=request, rows=rows((16, 1e-4), series="other"))
    stored = cr.load("scratch")
    assert {(row["series"], row["axis"]) for row in stored["rows"]} == {
        ("only", 8),
        ("other", 16),
    }


def test_a_rewritten_row_replaces_rather_than_duplicates(elsewhere, request):
    study = a_study()
    cr.record(study, request=request, rows=rows((8, 1e-3)))
    cr.record(study, request=request, rows=rows((8, 2e-3)))
    (row,) = cr.load("scratch")["rows"]
    assert row["error"] == 2e-3


def test_a_rung_that_gained_or_lost_a_quantity_fails(elsewhere, request):
    """`None` is "this rung has no observed order", not "this number is missing"."""
    study = a_study()
    cr.record(
        study, request=request, rows=[{"series": "only", "axis": 8, "error": None}]
    )
    os.environ.pop(cr.WRITE_ENV)
    cr.record(
        study, request=request, rows=[{"series": "only", "axis": 8, "error": None}]
    )
    with pytest.raises(cr.RecordError, match="gained or lost"):
        cr.record(study, request=request, rows=rows((8, 1e-3)))


def test_a_number_that_moved_beyond_the_column_s_tolerance_fails(elsewhere, request):
    study = a_study()
    cr.record(study, request=request, rows=rows((8, 1e-3)))
    os.environ.pop(cr.WRITE_ENV)
    cr.record(study, request=request, rows=rows((8, 1.05e-3)))  # inside rtol 0.1
    with pytest.raises(cr.RecordError, match="outside rtol"):
        cr.record(study, request=request, rows=rows((8, 2e-3)))


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ([], "vacuously"),
        ([{"series": "only", "axis": 0.5, "error": 1.0}], "not an int or a str"),
        ([{"series": "only", "axis": 8}], "carries"),
        ([{"series": "only", "axis": 8, "error": math.nan}], "non-finite"),
    ],
)
def test_a_row_set_that_could_not_be_compared_is_refused(
    elsewhere, request, bad, match
):
    with pytest.raises(cr.RecordError, match=match):
        cr.record(a_study(), request=request, rows=bad)


def test_a_missing_record_fails_rather_than_passing_vacuously(elsewhere, request):
    os.environ.pop(cr.WRITE_ENV)
    with pytest.raises(cr.RecordError, match="no convergence record"):
        cr.record(a_study("never-written"), request=request, rows=rows((8, 1.0)))


def test_a_row_the_record_does_not_have_fails(elsewhere, request):
    cr.record(a_study(), request=request, rows=rows((8, 1e-3)))
    os.environ.pop(cr.WRITE_ENV)
    with pytest.raises(cr.RecordError, match="no stored row"):
        cr.record(a_study(), request=request, rows=rows((16, 1e-4)))


def test_a_study_that_changed_shape_fails_rather_than_drawing_the_old_caption(
    elsewhere, request
):
    cr.record(a_study(), request=request, rows=rows((8, 1e-3)))
    os.environ.pop(cr.WRITE_ENV)
    moved = dataclasses.replace(a_study(), case={"what": "something else"})
    with pytest.raises(cr.RecordError, match="not the one this study"):
        cr.record(moved, request=request, rows=rows((8, 1e-3)))


@pytest.mark.parametrize(
    ("variable", "value", "match"),
    [
        ("CI", "true", "must not be set in CI"),
        ("PYTEST_XDIST_WORKER", "gw3", "under xdist"),
    ],
)
def test_the_writer_refuses_a_lane_that_would_race_or_rewrite_its_own_witness(
    elsewhere, request, monkeypatch, variable, value, match
):
    monkeypatch.setenv(variable, value)
    with pytest.raises(cr.RecordError, match=match):
        cr.record(a_study(), request=request, rows=rows((8, 1.0)))


def test_the_writer_refuses_a_dirty_tree(elsewhere, request, monkeypatch):
    """A record written from a tree that does not match its commit is not a record."""
    monkeypatch.setattr(cr, "dirty_paths", lambda: ["M src/solprop/stepper/tableau.py"])
    with pytest.raises(cr.RecordError, match="not clean"):
        cr.record(a_study(), request=request, rows=rows((8, 1.0)))


def test_the_writer_refuses_when_there_is_no_checkout(elsewhere, request, monkeypatch):
    monkeypatch.setattr(cr, "dirty_paths", lambda: None)
    with pytest.raises(cr.RecordError, match="no git checkout"):
        cr.record(a_study(), request=request, rows=rows((8, 1.0)))


def test_the_record_files_are_the_only_thing_a_write_run_may_dirty():
    """The gate excludes the records themselves, or only the first write would pass.

    Once the files are committed they are tracked, so a second writer in the same
    session -- and every writer in the second and third regeneration process -- would
    see a modified tracked file and refuse.
    """
    dirty = cr.dirty_paths()
    if dirty is None:
        pytest.skip("not a git checkout")
    assert not any(cr.RELATIVE in line for line in dirty)


def test_ci_never_writes_records():
    """Belt and braces for the gate above: a lane must never rewrite its own witness."""
    if os.environ.get("CI"):
        assert not cr.writing()
