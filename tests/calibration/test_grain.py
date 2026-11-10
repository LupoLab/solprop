# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Operation-grain candidates from the problems' work sizes (SEP 46 phase 2)."""

from __future__ import annotations

from solprop.calibration import _native
from solprop.calibration.grain import (
    TIMED,
    Candidate,
    Operation,
    affected,
    best,
    operations,
    proposals,
    width,
)
from solprop.calibration.selection import Verdict

TABLE = _native.execution.builtin("Linux").native()


def test_sizing_keys_parse_into_operations() -> None:
    found = operations(
        {
            "kernels.stage.4096": {},
            "kernels.response.98304.samples.4096": {},
            "kernels.odd": {},
            "kernels.stage.many": {},
        }
    )
    assert found == [
        Operation("kernels", "response", 98304, 4096),
        Operation("kernels", "stage", 4096),
    ]


def test_widths_follow_the_native_rule() -> None:
    stage = Operation("kernels", "stage", 4096)
    assert width(TABLE, stage, 18) == 18
    assert width(TABLE, Operation("kernels", "stage", 100), 18) == 1
    copy = Operation("kernels", "copy", 65536)
    assert width(TABLE, copy, 18) == 18
    response = Operation("kernels", "response", 33 * 4096, 4096)
    assert width(TABLE, response, 18) == 4
    # Forty columns or 65536 samples take the full pool above the threshold.
    assert width(TABLE, Operation("kernels", "response", 40 * 1024, 1024), 18) == 18
    assert width(TABLE, Operation("kernels", "response", 65536, 65536), 18) == 18
    assert width(TABLE, Operation("kernels", "response", 40000, 0), 18) == 4
    assert width(TABLE, Operation("kernels", "response", 1000, 1000), 18) == 1


def test_a_record_affects_a_problem_only_where_a_width_moves() -> None:
    found = [Operation("kernels", "stage", 4096)]
    assert not affected(dict(TABLE, stage_threshold=2048), TABLE, found, 18)
    assert affected(
        dict(TABLE, stage_threshold=4097, stage_full=4097), TABLE, found, 18
    )


def test_each_class_moves_one_step_past_its_nearest_sizes() -> None:
    record = dict(TABLE, stage_threshold=1000, stage_full=10000)
    problems = {
        "a": [Operation("kernels", "stage", size) for size in (500, 5000, 50000)]
    }
    found = proposals(record, problems, 18)
    names = [candidate.name for candidate in found]
    assert names == [
        "medium 2",
        "medium 8",
        "stage threshold 500",
        "stage threshold 5001",
        "stage full 5000",
        "stage full 50001",
    ]
    moved = found[2].record(record)
    assert (moved["stage_threshold"], moved["stage_full"]) == (500, 10000)
    # Thresholds only, and a limit.
    assert [c.name for c in proposals(record, problems, 18, fulls=False)][2:] == [
        "stage threshold 500",
        "stage threshold 5001",
    ]
    assert len(proposals(record, problems, 18, limit=3)) == 3
    # A class whose sizes all run serially can only thread one more of them.
    serial = {"a": [Operation("kernels", "stage", 500)]}
    assert [c.name for c in proposals(record, serial, 18) if c.setting == "stage"] == [
        "stage threshold 500"
    ]


def test_the_medium_width_and_the_compact_team_have_short_ladders() -> None:
    problems = {"a": [Operation("kernels", "stage", 4096)]}
    # Every operation at full width: a medium width changes nothing.
    assert [c.setting for c in proposals(TABLE, problems, 18)] == ["stage", "stage"]
    compact = proposals(TABLE, problems, 18, compact=["a"])
    assert [c.name for c in compact][:2] == ["compact 4", "compact 16"]
    # On four threads the compact team of eight is already the whole pool.
    small = proposals(TABLE, problems, 4, compact=["a"])
    assert [c.name for c in small if c.setting == "compact"] == []
    # Classes outside the timed windows keep the table's values.
    adams = {"a": [Operation("kernels", "adams", 4096)]}
    assert proposals(TABLE, adams, 18) == []
    assert "adams" not in TIMED
    assert "fft" in TIMED


def test_transforms_thread_by_rows_and_row_length() -> None:
    # Linux: from 8192 samples a thread per 4096, at most one a row; rows of 16384
    # samples or more may be split up to the medium width; the full pool from fft_full.
    def threads(rows: int, length: int, capacity: int = 18) -> int:
        return width(TABLE, Operation("fft", "fft", rows * length, length), capacity)

    assert threads(1, 4096) == 1  # below the start
    assert threads(4, 3072) == 3  # 12288 samples, three threads' worth
    assert threads(4, 10240) == 4  # a thread a row
    assert threads(1, 10240) == 1  # one short row is not split
    assert threads(1, 32768) == 4  # a long row, split up to the medium width
    assert threads(3, 20480) == 4
    assert threads(12, 10240) == 12
    assert threads(128, 1152) == 18  # many short rows take the pool
    assert threads(289, 128) == 9  # 36992 samples
    assert threads(128, 1152, capacity=8) == 8
    # A batch of unknown row length keeps the element rule.
    assert width(TABLE, Operation("fft", "fft", 40960), 18) == 4
    assert width(TABLE, Operation("fft", "fft", 1 << 22, 1024), 36) == 36


def test_the_transforms_shape_settings_are_moved_where_they_change_a_width() -> None:
    transforms = {
        "a": [
            Operation("fft", "fft", 40960, 10240),
            Operation("fft", "fft", 36992, 128),
        ]
    }
    names = [c.name for c in proposals(TABLE, transforms, 18) if c.setting == "fft"]
    # The full size is a backstop and is not moved, and neither long row changes a
    # width here: four rows take four threads split or not, and 128 samples is short
    # either way.
    assert names == [
        "fft threshold 36993",
        "fft thread elements 2048",
        "fft thread elements 8192",
    ]
    # The transforms go last, after every other class's moves.
    assert TIMED[-1] == "fft"


def test_the_best_admitted_candidate_of_each_setting_is_kept() -> None:
    def verdict(score: float | None, admitted: bool) -> Verdict:
        return Verdict(admitted, score, score, 3, "")

    candidates = [
        Candidate("medium", "medium 2", {"medium_threads": 2}),
        Candidate("medium", "medium 8", {"medium_threads": 8}),
        Candidate("stage", "stage threshold 1", {"stage_threshold": 1}),
        Candidate("fft", "fft threshold 1", {"fft_threshold": 1}),
    ]
    verdicts = {
        "medium 2": verdict(0.9, True),
        "medium 8": verdict(0.8, True),
        "stage threshold 1": verdict(1.2, False),
        "fft threshold 1": verdict(None, True),
    }
    assert [c.name for c in best(candidates, verdicts)] == ["medium 8"]
    # A later candidate of the same setting with a worse score does not replace it.
    worse = {**verdicts, "medium 2": verdict(0.7, True), "medium 8": verdict(0.8, True)}
    assert [c.name for c in best(candidates, worse)] == ["medium 2"]
