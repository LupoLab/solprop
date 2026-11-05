# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The route through a simulation: which stages are open and what comes next.

Home, then Setup, Run and Analyse (design 5.1). The window asks these rules rather than
holding its own, so that they are tested without a window.
"""

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["Primary", "Progress", "Stage", "primary_action", "stage_open"]


class Stage(StrEnum):
    """Where the window is."""

    HOME = "home"
    SETUP = "setup"
    RUN = "run"
    ANALYSE = "analyse"


@dataclass(frozen=True)
class Progress:
    """How far a simulation has got.

    Attributes
    ----------
    has_setup : bool
        Whether a setup is open.
    can_run : bool
        Whether solprop accepts it, so that it can run.
    running : bool
        Whether a run is going.
    has_run : bool
        Whether a run has been started in this window.
    has_results : bool
        Whether a results file is open.
    can_resume : bool
        Whether solprop says the last run can be continued, the setup unchanged since.
    """

    has_setup: bool = False
    can_run: bool = False
    running: bool = False
    has_run: bool = False
    has_results: bool = False
    can_resume: bool = False


@dataclass(frozen=True)
class Primary:
    """The button at the top right: what comes next.

    Attributes
    ----------
    action : str
        ``"run"``, ``"resume"`` or ``"stop"``.
    text : str
        What it says.
    enabled : bool
        Whether it can be pressed.
    """

    action: str
    text: str
    enabled: bool = True


def stage_open(stage: Stage, progress: Progress) -> bool:
    """Say whether a stage can be shown.

    Parameters
    ----------
    stage : Stage
        The stage.
    progress : Progress
        How far the simulation has got.

    Returns
    -------
    bool
        Whether its button is enabled.

    Examples
    --------
    >>> stage_open(Stage.ANALYSE, Progress(has_setup=True))
    False
    >>> stage_open(Stage.RUN, Progress(has_run=True))
    True
    """
    return {
        Stage.HOME: True,
        Stage.SETUP: progress.has_setup,
        Stage.RUN: progress.running or progress.has_run,
        Stage.ANALYSE: progress.has_results,
    }[stage]


def primary_action(progress: Progress) -> Primary:
    """Return the primary button, which is the same on every stage.

    Parameters
    ----------
    progress : Progress
        How far the simulation has got.

    Returns
    -------
    Primary
        Stop while a run goes, on any stage; Resume where the last run can go on; Run on
        a setup not yet run; otherwise Rerun.

    Examples
    --------
    >>> primary_action(Progress(has_setup=True, can_run=True)).text
    'Run'
    >>> primary_action(Progress(has_setup=True, running=True)).action
    'stop'
    >>> primary_action(Progress(has_run=True, has_results=True)).text
    'Rerun'
    >>> primary_action(Progress(has_setup=True)).enabled
    False
    >>> primary_action(Progress(has_setup=True, can_resume=True)).text
    'Resume'
    """
    if progress.running:
        return Primary("stop", "Stop")
    if progress.can_resume:
        return Primary("resume", "Resume")
    text = "Rerun" if progress.has_run else "Run"
    return Primary("run", text, progress.has_setup and progress.can_run)
