# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Both backends against the pinned corpus (SEP 52 section 7, "A pinned corpus first").

``tests/data/backend_corpus.npz`` holds what every case of
``tests/backend_corpus_cases.py`` left on ``numpy_ref`` and on the native CPU build,
recorded before SEP 52 restructured the reference and native's seam
(``tests/data/generate_backend_corpus.py``): the last saved field, the planes'
statistics and step history, the boundary monitor's and transverse edge's records, a
resume from the middle plane, and the timing probe's read-backs or refusal. Each
backend's rerun is held to its own record.

**Bitwise is a diagnostic.** On the recording build and thread setting (the same
:func:`backend_corpus_cases.identity`) the names whose bits moved are reported as the
test's ``bitwise`` property, never failed on (SEP 25: no comparison needs bitwise
agreement across builds, thread counts or changes of operation order).

**What fails** is a departure beyond the computation's SEP 25 budget, fixed before the
comparison from the problem's operation counts:

- a fixed schedule is a matched discrete march (class R), held to
  :func:`march_bounds.march_bound` at the launch's nonlinear phase and the source's
  conditioning, both read by the reference when the corpus was written: the fields at
  that budget, the statistics at :data:`march_bounds.STATISTIC_FACTOR` times it (a rate
  statistic at :data:`RATE_FACTOR` times), the prescribed positions and steps, flags,
  counts and texts exactly, and every monitor reading within
  :func:`march_bounds.reading_tolerance`;
- the probe's read-backs are one trial each, held to the same budget for one step;
- an adaptive march (class T) meets the accuracy envelope of
  :func:`march_bounds.assert_in_envelope`: four times the recorded run's departure from
  the recorded run at a hundredth of its tolerance, and its accepted steps a work band
  (:func:`bounds.assert_work_within`); its decisions are compared bit for bit only in
  the diagnostic.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backend_corpus_cases import CASES, Case, Recorded, identity, record
from bounds import AccuracyEnvelope, assert_within, assert_work_within, departure
from march_bounds import (
    ENVELOPE_FACTOR,
    assert_matched,
    assert_record_within,
    assert_stored_matched,
    evaluation_rounding,
    march_bound,
)
from solprop.backends.registry import available
from solprop.driver import Simulation

CORPUS = Path(__file__).parents[1] / "data" / "backend_corpus.npz"

BACKENDS = [
    "numpy_ref",
    pytest.param(
        "native",
        marks=pytest.mark.skipif(
            "native" not in available(), reason="native is not installed"
        ),
    ),
]

RATE_FACTOR = 2 * 2 * 11
"""How many times the field's relative departure an ionisation statistic may move: its
rate's ``d ln W / d ln |E|`` is at most ``2 K`` in the multiphoton limit, with
``K = 11`` photons of 800 nm to ionise argon (15.76 eV), and a factor of two of margin;
the tunnelling regime is less sensitive (``bounds.rate_log_sensitivity``)."""

QUADRATURE = "stats/quadrature_error"
"""The transverse quadrature's relative error on the polarisation: a rounding residue
where the quadrature is exact (about 1e-15 on these cases), held to an absolute bound
(:func:`_assert_quadrature`)."""

EXACT = ("z", "resumed/z", "step_history/z", "step_history/h")
"""What a prescribed schedule fixes whatever the arithmetic (class P)."""

WORK_FACTOR = 1.5
"""The work band of an adaptive rerun's accepted steps: a rounding difference may flip
one decision, which on these runs of a dozen or a few hundred steps moves the count by
far less than half (SEP 25 section 4.4)."""


@functools.cache
def _corpus() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return the corpus's arrays by name and its records."""
    with np.load(CORPUS, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    records = json.loads(arrays.pop("records").tobytes().decode("utf-8"))
    return arrays, records


def _stored(case: Case, backend: str) -> Recorded:
    """Return what the corpus recorded of a case on a backend."""
    arrays, records = _corpus()
    prefix = f"{case.name}/{backend}/"
    own = {
        name.removeprefix(prefix): value
        for name, value in arrays.items()
        if name.startswith(prefix)
    }
    return Recorded(own, records["cases"][case.name]["backends"][backend])


CROSS_BUILD = 2.0
"""The factor on a matched march's budget for a record made by another build.

The corpus is recorded once, on one machine, and compared on every platform CI runs.
Two builds evaluate the same operations with different vector-maths libraries and
contraction, which :func:`march_bounds.march_bound` (two runs of one build) does not
count: the general modal case was measured on x86-64 Linux at 1.01 of the single-build
budget against a record made on arm64 macOS.
"""


def _budget(case: Case, simulation: Simulation, steps: int | None = None) -> float:
    """Return a case's rounding budget for two matched runs (class R).

    :func:`march_bounds.march_bound` at the recorded launch phase (no smaller than a
    tenth, as :func:`march_bounds.simulation_bound` takes it) and conditioning, over
    the schedule's steps or `steps`, times :data:`CROSS_BUILD`.
    """
    entry = _corpus()[1]["cases"][case.name]
    return CROSS_BUILD * march_bound(
        simulation.translate(),
        simulation._run_request(),
        steps,
        phi=max(float(entry["phi"]), 0.1),
        kappa=float(entry["kappa"]),
        backend=simulation.settings.backend,
    )


def _assert_quadrature(
    case: Case, simulation: Simulation, got: Recorded, expected: Recorded
):
    """Hold the quadrature's error to the rounding of the evaluations it compares.

    It is the relative difference of two evaluations of the polarisation, each within
    :func:`march_bounds.evaluation_rounding` of its value relative to it (both runs'
    roundings counted there); so a rerun's may move by the four evaluations' sum, an
    absolute bound on a dimensionless number that is itself rounding on an exact rule.
    Where it is undefined (the launch) it stays undefined.
    """
    if QUADRATURE not in expected.arrays:
        return
    one, other = got.arrays[QUADRATURE], expected.arrays[QUADRATURE]
    assert np.array_equal(np.isnan(one), np.isnan(other)), case.name
    problem = simulation.translate()
    limit = 4.0 * evaluation_rounding(
        int(problem.grids.n_to),
        int(problem.transform.n_nodes),
        int(problem.shape[0]),
        plasma=bool(problem.flags.has_plasma),
    )
    found = np.nan_to_num(np.abs(one - other))
    assert np.all(found <= limit), (case.name, float(np.max(found)), limit)


def _bitwise(got: Recorded, expected: Recorded) -> list[str]:
    """Return the names whose bits differ between a rerun and its record."""
    differ = []
    # An adaptive case's tighter run is read from the record, not rerun.
    for name in sorted((set(got.arrays) | set(expected.arrays)) - {"tighter/field"}):
        one, other = got.arrays.get(name), expected.arrays.get(name)
        if one is None or other is None or one.shape != other.shape:
            differ.append(name)
            continue
        nan = one.dtype.kind in "fc"
        if not np.array_equal(one, other, equal_nan=nan):
            differ.append(name)
    for name in sorted(expected.records):
        mine = json.dumps(got.records.get(name), sort_keys=True)
        if mine != json.dumps(expected.records[name], sort_keys=True):
            differ.append(f"records/{name}")
    return differ


def _compare_probe(
    case: Case, simulation: Simulation, got: Recorded, expected: Recorded
):
    """Hold the probe's read-backs to one step's budget, its refusal exactly."""
    assert got.records["probe"]["trials"] == expected.records["probe"]["trials"]
    refusal = expected.records["probe"]["refusal"]
    if refusal is not None:
        # The low-storage march has no trial field (SEP 52 section 5.5).
        assert "no trial field" in refusal
        assert got.records["probe"]["refusal"] is not None
        assert "probe/fields" not in got.arrays
        return
    assert got.records["probe"]["refusal"] is None
    assert_matched(
        got.arrays["probe/fields"],
        expected.arrays["probe/fields"],
        _budget(case, simulation, steps=1),
        what=f"{case.name} probe",
    )


def _compare_matched(
    case: Case, simulation: Simulation, got: Recorded, expected: Recorded
):
    """Hold a fixed-schedule case to its matched march's budget (class R)."""
    bound = _budget(case, simulation)
    # The error norms of a replay are a difference of nearly equal stage sums: their
    # relative rounding is the field's over the estimate's own size, which no budget
    # of the field bounds. They decide nothing on a schedule; the diagnostic reads
    # their bits.
    left_out = ("probe/fields", "step_history/norm", QUADRATURE)
    names = set(expected.arrays) - set(left_out)
    assert names == set(got.arrays) - set(left_out), case.name
    factors = {
        name: RATE_FACTOR for name in names if name.startswith("stats/species_")
    } | {
        name: RATE_FACTOR
        for name in ("stats/electron_density", "stats/peak_ionisation_rate")
        if name in names
    }
    assert_stored_matched(
        {name: expected.arrays[name] for name in names},
        {name: got.arrays[name] for name in names},
        bound,
        exact=EXACT,
        factors=factors,
        what=case.name,
    )
    steps = int(np.sum(expected.arrays["step_history/accepted"]))
    for part in ("run", "resumed"):
        assert_record_within(
            got.records[part]["monitor"],
            expected.records[part]["monitor"],
            bound,
            steps,
            what=f"{case.name} {part}",
            work=False,
        )
    # Accounting: the source evaluations a run made, where a backend counts them.
    assert (
        got.records["run"]["source_evaluations"]
        == expected.records["run"]["source_evaluations"]
    )


def _compare_adaptive(case: Case, got: Recorded, expected: Recorded):
    """Hold an adaptive case to its accuracy envelope (class T)."""
    reference = expected.arrays["tighter/field"]
    error = max(
        departure(expected.arrays["field"], reference),
        departure(expected.arrays["resumed/field"], reference),
    )
    assert error > 0.0, "the tighter run is the run itself"
    envelope = AccuracyEnvelope(
        reference=f"{case.name} at a hundredth of its tolerance",
        error=error,
        factor=ENVELOPE_FACTOR,
    )
    for name in ("field", "resumed/field"):
        assert_within(
            got.arrays[name], reference, envelope.bound, what=f"{case.name} {name}"
        )
    for name in ("z", "resumed/z"):
        # The planes are prescribed even where the steps are not.
        assert np.array_equal(got.arrays[name], expected.arrays[name]), name
    assert_work_within(
        int(np.sum(got.arrays["step_history/accepted"])),
        int(np.sum(expected.arrays["step_history/accepted"])),
        factor=WORK_FACTOR,
        what=f"{case.name} accepted steps",
    )
    # Two runs each inside the envelope differ by at most twice it.
    bound = 2.0 * envelope.bound
    steps = int(np.sum(expected.arrays["step_history/accepted"]))
    stats = [
        name
        for name in expected.arrays
        if name.startswith("stats/") and name != QUADRATURE
    ]
    assert_stored_matched(
        {name: expected.arrays[name] for name in stats},
        {name: got.arrays[name] for name in stats},
        bound,
        factors={name: RATE_FACTOR for name in stats if "ionisation" in name},
        what=case.name,
    )
    for part in ("run", "resumed"):
        assert_record_within(
            got.records[part]["monitor"],
            expected.records[part]["monitor"],
            bound,
            steps,
            what=f"{case.name} {part}",
            work=True,
        )


# Medium: each case runs three short propagations and a probe, under a second on
# native and up to a few seconds on the reference.
@pytest.mark.medium
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_a_rerun_reproduces_its_recorded_corpus(
    case: Case, backend: str, tmp_path: Path, record_property
) -> None:
    expected = _stored(case, backend)
    simulation = case.build(backend)
    got = record(case, backend, tmp_path, simulation=simulation, tighter=False)
    if identity(backend) == _corpus()[1]["identity"][backend]:
        differ = _bitwise(got, expected)
        report = "identical" if not differ else "differs: " + ", ".join(differ)
    else:
        report = "not compared: another build or thread setting"
    record_property("bitwise", report)
    print(f"{case.name} on {backend}, bitwise: {report}")  # noqa: T201 - the diagnostic
    assert got.records["resumed"]["executions"] == 2
    _compare_probe(case, simulation, got, expected)
    _assert_quadrature(case, simulation, got, expected)
    if case.adaptive:
        _compare_adaptive(case, got, expected)
    else:
        _compare_matched(case, simulation, got, expected)


def test_the_corpus_holds_what_it_was_recorded_for() -> None:
    # Every case on both backends, and the features the restructure must keep: the
    # split collar's removal added over several steps between readings, a filled guard,
    # a transverse edge read at every step, a resume, the probe's read-backs and the
    # low-storage refusal, and the adaptive marches' rejections.
    _, records = _corpus()
    assert set(records["cases"]) == {case.name for case in CASES}
    marches = {case.march for case in CASES}
    assert any("adaptive Runge-Kutta" in one for one in marches)
    assert any("replayed Runge-Kutta" in one for one in marches)
    assert any("interaction formulation" in one for one in marches)
    assert any("carrier frame" in one for one in marches)
    assert any("low-storage" in one for one in marches)
    for case in CASES:
        for backend in ("numpy_ref", "native"):
            stored = _stored(case, backend)
            monitor = stored.records["run"]["monitor"]
            assert stored.records["resumed"]["executions"] == 2
            assert stored.records["resumed"]["monitor"]["resumed_at"]
            assert monitor["launch_energy"] > 0.0
            assert ("tighter/field" in stored.arrays) == case.adaptive
            low_storage = case.march == "low-storage"
            assert (stored.records["probe"]["refusal"] is not None) == low_storage
            assert ("probe/fields" in stored.arrays) != low_storage
    fibre = _stored(CASES[0], "numpy_ref").records["run"]["monitor"]
    assert fibre["skipped"] > fibre["steps"] > 0
    assert fibre["raw_removed_fraction"] > 1e-5
    assert fibre["guard_high_max"] > 1e-6
    edge = _stored(
        next(case for case in CASES if case.name == "line_edge_adams"), "native"
    ).records["run"]["monitor"]["transverse"]
    assert edge["reads"] > 0 and edge["ramp_removed"] > 0.0
    adaptive = _stored(
        next(case for case in CASES if case.name == "fibre_rk_adaptive"), "numpy_ref"
    )
    assert not np.all(adaptive.arrays["step_history/accepted"])
