# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The calibration's statistical rule: which setting replaces the table (SEP 46).

One rule throughout (SEP 46 section 6.5), the ``ab`` harness's:

- *Pairs.* Each candidate is timed interleaved with the built-in value, in rounds; a
  round times each once in a fresh order. A candidate's score is the median of its
  per-round ratios to the built-in.
- *Noise.* Each session also times the built-in against itself (A/A). The noise is the
  largest per-round deviation of that ratio from one, and at least
  :data:`NOISE_FLOOR`.
- *Admission.* A candidate replaces the built-in value only if its largest per-round
  ratio is below one minus the noise: it wins in every round by more than the noise.
  And no problem may be slower beyond the noise and :data:`TOLERANCE`, SEP 46's own
  bar (section 7 item 1): its median ratio on each problem timed must stay within
  ``1 + noise + TOLERANCE``, so that a large win on some problems cannot carry a real
  loss on another, while a loss within the bar does not veto a broad win. Ties keep
  the built-in. A failed round is dropped; fewer than the level's minimum rounds
  keeps the built-in.
- *Combined validation.* The admitted settings are timed together against the table on
  fresh rounds; the combination is published only if no window is slower than that
  same bar and the geometric mean is below one. Otherwise groups are dropped in
  :data:`DROP_ORDER` and the remainder validated again.

Every function here is pure: it reads timings and returns a verdict.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "DROP_ORDER",
    "MAX_CANDIDATES",
    "NOISE_FLOOR",
    "NOISE_LIMIT",
    "TOLERANCE",
    "Times",
    "Validation",
    "Verdict",
    "admit",
    "choose",
    "fallbacks",
    "noise",
    "ratios",
    "validate",
]

NOISE_FLOOR = 0.02
"""The least noise a session claims: a win must exceed 2 % in every round."""

TOLERANCE = 0.02
"""How much slower than its noise any one problem may run under a candidate: SEP 46
section 7 item 1's bar, no case slower than its A/A range plus 2 %."""

NOISE_LIMIT = 0.10
"""A/A noise above which a session is inconclusive and nothing is published."""

MAX_CANDIDATES = 4
"""The most candidates a setting is chosen from, so that the search cannot select noise
from a large set."""

DROP_ORDER = ("grain", "fft", "process")
"""The groups dropped, in this order, when a combination fails validation: operation
grain, FFT planning, then the waiting policy and the team."""

type Times = Sequence[float | None]
"""Seconds per round, ``None`` for a round that failed or was skipped."""


def ratios(candidate: Times, builtin: Times) -> list[float]:
    """Return the per-round ratios of `candidate` to `builtin`, dropping failed rounds.

    Parameters
    ----------
    candidate, builtin : sequence of float or None
        Seconds per round, in round order.

    Returns
    -------
    list of float
        One ratio per round where both were timed.

    Examples
    --------
    >>> ratios([0.9, None, 1.8], [1.0, 1.0, 2.0])
    [0.9, 0.9]
    """
    return [
        new / old
        for new, old in zip(candidate, builtin, strict=True)
        if new is not None and old is not None and old > 0 and math.isfinite(new)
    ]


def noise(builtin: Times, again: Times, minimum: int) -> float | None:
    """Return the A/A noise of a session, or ``None`` with too few rounds.

    Parameters
    ----------
    builtin, again : sequence of float or None
        The built-in setting timed twice in each round.
    minimum : int
        The fewest complete rounds that make a session.

    Returns
    -------
    float or None
        The largest per-round deviation of the ratio from one, at least
        :data:`NOISE_FLOOR`.

    Examples
    --------
    >>> noise([1.0, 1.0, 1.0], [1.01, 0.97, 1.0], minimum=3)
    0.03
    >>> noise([1.0, 1.0], [1.0, 1.0], minimum=3) is None
    True
    """
    measured = ratios(again, builtin)
    if len(measured) < minimum:
        return None
    return round(max(NOISE_FLOOR, *(abs(value - 1.0) for value in measured)), 12)


@dataclass(frozen=True)
class Verdict:
    """Whether one candidate replaces the built-in value.

    Attributes
    ----------
    admitted : bool
        Whether it won every round by more than the noise.
    score : float or None
        The median per-round ratio to the built-in; ``None`` without enough rounds.
    worst : float or None
        The largest per-round ratio.
    rounds : int
        The complete rounds compared.
    reason : str
        Why it was or was not admitted.
    windows : Mapping[str, float]
        Each problem's median per-round ratio, where the problems were timed apart.
    """

    admitted: bool
    score: float | None
    worst: float | None
    rounds: int
    reason: str
    windows: Mapping[str, float] = field(default_factory=dict)

    def record(self) -> dict[str, object]:
        """Return the verdict as JSON data.

        Returns
        -------
        dict
            The fields as JSON data.
        """
        return {
            "admitted": self.admitted,
            "score": self.score,
            "worst": self.worst,
            "rounds": self.rounds,
            "reason": self.reason,
            "windows": dict(self.windows),
        }


def admit(
    candidate: Times,
    builtin: Times,
    noise: float,
    minimum: int,
    windows: Mapping[str, Times] | None = None,
) -> Verdict:
    """Decide one candidate against the built-in value.

    Parameters
    ----------
    candidate, builtin : sequence of float or None
        Seconds per round, interleaved.
    noise : float
        The session's A/A noise (:func:`noise`).
    minimum : int
        The fewest complete rounds that decide.
    windows : Mapping[str, sequence of float or None] or None, optional
        Each problem's per-round ratio of the candidate to the built-in value.

    Returns
    -------
    Verdict
        Admitted only if every round's ratio is below ``1 - noise`` and no
        problem's median ratio is above ``1 + noise +`` :data:`TOLERANCE`.

    Examples
    --------
    >>> admit([0.9, 0.92, 0.91], [1.0, 1.0, 1.0], noise=0.02, minimum=3).admitted
    True
    >>> admit([0.8, 0.99, 0.8], [1.0, 1.0, 1.0], noise=0.02, minimum=3).reason
    'lost or tied in a round: worst ratio 0.990 against 0.980'
    >>> admit(
    ...     [0.8] * 3, [1.0] * 3, 0.02, 3, {"small": [0.5] * 3, "large": [1.3] * 3}
    ... ).reason
    'slower on large: median ratio 1.300 against 1.040'
    """
    measured = ratios(candidate, builtin)
    if len(measured) < minimum:
        return Verdict(
            False, None, None, len(measured), f"{len(measured)} of {minimum} rounds"
        )
    score, worst = statistics.median(measured), max(measured)
    medians = {
        name: statistics.median(found)
        for name, times in (windows or {}).items()
        if (found := ratios(times, [1.0] * len(times)))
    }
    if worst >= 1.0 - noise:
        return Verdict(
            False,
            score,
            worst,
            len(measured),
            f"lost or tied in a round: worst ratio {worst:.3f} against "
            f"{1.0 - noise:.3f}",
            medians,
        )
    slower = max(medians.items(), key=lambda item: item[1], default=None)
    if slower is not None and slower[1] > 1.0 + noise + TOLERANCE:
        return Verdict(
            False,
            score,
            worst,
            len(measured),
            f"slower on {slower[0]}: median ratio {slower[1]:.3f} against "
            f"{1.0 + noise + TOLERANCE:.3f}",
            medians,
        )
    return Verdict(True, score, worst, len(measured), "won every round", medians)


def choose(
    candidates: Mapping[str, Times],
    builtin: Times,
    noise: float,
    minimum: int,
    windows: Mapping[str, Mapping[str, Times]] | None = None,
) -> tuple[str | None, dict[str, Verdict]]:
    """Choose among a setting's candidates: the admitted one with the lowest score.

    Parameters
    ----------
    candidates : Mapping[str, sequence of float or None]
        At most :data:`MAX_CANDIDATES` candidates' seconds per round, by name.
    builtin : sequence of float or None
        The built-in value's, interleaved with them.
    noise : float
        The session's A/A noise.
    minimum : int
        The fewest complete rounds that decide.
    windows : Mapping[str, Mapping[str, sequence]] or None, optional
        Each candidate's per-problem per-round ratios (:func:`admit`).

    Returns
    -------
    tuple
        The chosen name, or ``None`` to keep the built-in value; and every
        candidate's verdict.

    Raises
    ------
    ValueError
        For more than :data:`MAX_CANDIDATES` candidates.
    """
    if len(candidates) > MAX_CANDIDATES:
        raise ValueError(
            f"{len(candidates)} candidates; a setting chooses among at most "
            f"{MAX_CANDIDATES}"
        )
    verdicts = {
        name: admit(times, builtin, noise, minimum, (windows or {}).get(name))
        for name, times in candidates.items()
    }
    admitted = [
        (verdict.score, name)
        for name, verdict in verdicts.items()
        if verdict.admitted and verdict.score is not None
    ]
    return (min(admitted)[1] if admitted else None), verdicts


@dataclass(frozen=True)
class Validation:
    """Whether a combination of admitted settings may be published.

    Attributes
    ----------
    passed : bool
        No window slower than ``1 + noise +`` :data:`TOLERANCE`, and a geometric
        mean below one.
    windows : Mapping[str, float]
        Each window's median per-round ratio to the table.
    geomean : float or None
        Their geometric mean; ``None`` when a window has too few rounds.
    reason : str
        Why it passed or failed.
    """

    passed: bool
    windows: Mapping[str, float]
    geomean: float | None
    reason: str

    def record(self) -> dict[str, object]:
        """Return the validation as JSON data.

        Returns
        -------
        dict
            The fields as JSON data.
        """
        return {
            "passed": self.passed,
            "windows": dict(self.windows),
            "geomean": self.geomean,
            "reason": self.reason,
        }


def validate(
    windows: Mapping[str, tuple[Times, Times]], noise: float, minimum: int
) -> Validation:
    """Validate a combination on fresh rounds of every calibration window.

    Parameters
    ----------
    windows : Mapping[str, tuple]
        Each window's seconds per round under the combination and under the table.
    noise : float
        The session's A/A noise.
    minimum : int
        The fewest complete rounds a window needs.

    Returns
    -------
    Validation
        The verdict, with each window's median ratio.

    Examples
    --------
    >>> validate({"a": ([0.9] * 3, [1.0] * 3)}, noise=0.02, minimum=3).passed
    True
    >>> validate(
    ...     {"a": ([0.8] * 3, [1.0] * 3), "b": ([1.05] * 3, [1.0] * 3)},
    ...     noise=0.02,
    ...     minimum=3,
    ... ).reason
    'b is slower: 1.050 against 1.040'
    """
    medians: dict[str, float] = {}
    for name, (combined, table) in windows.items():
        measured = ratios(combined, table)
        if len(measured) < minimum:
            return Validation(
                False, medians, None, f"{name}: {len(measured)} of {minimum} rounds"
            )
        medians[name] = statistics.median(measured)
    if not medians:
        return Validation(False, medians, None, "no windows")
    geomean = math.exp(statistics.fmean(math.log(value) for value in medians.values()))
    for name, value in medians.items():
        if value > 1.0 + noise + TOLERANCE:
            return Validation(
                False,
                medians,
                geomean,
                f"{name} is slower: {value:.3f} against {1.0 + noise + TOLERANCE:.3f}",
            )
    if geomean >= 1.0:
        return Validation(False, medians, geomean, f"geometric mean {geomean:.3f}")
    return Validation(True, medians, geomean, f"geometric mean {geomean:.3f}")


def fallbacks(groups: Sequence[str]) -> list[tuple[str, ...]]:
    """Return the combinations to validate in turn, dropping groups in order.

    Parameters
    ----------
    groups : sequence of str
        The groups with an admitted setting, among :data:`DROP_ORDER`.

    Returns
    -------
    list of tuple of str
        Every admitted group, then that set less grain, less grain and FFT, and so
        on; empty combinations are left out.

    Raises
    ------
    ValueError
        For a group outside :data:`DROP_ORDER`.

    Examples
    --------
    >>> fallbacks(["process", "grain", "fft"])
    [('grain', 'fft', 'process'), ('fft', 'process'), ('process',)]
    >>> fallbacks(["process"])
    [('process',)]
    """
    unknown = set(groups) - set(DROP_ORDER)
    if unknown:
        raise ValueError(f"unknown groups {sorted(unknown)}")
    ordered = [group for group in DROP_ORDER if group in groups]
    return [tuple(ordered[index:]) for index in range(len(ordered))]
