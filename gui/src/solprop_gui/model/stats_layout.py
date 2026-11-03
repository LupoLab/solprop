# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the Statistics view's four panels show.

The view is a 2 by 2 grid; each panel draws one quantity against z on its left axis
and, optionally, a second on its right. A quantity is a statistic the run recorded
(by its name in ``stats/``) or the run's step size (:data:`STEP`). A layout is kept
across files: a choice the next file also recorded stays, the rest are filled from
:func:`default_layout`.
"""

from collections.abc import Sequence
from typing import Final

__all__ = ["PANELS", "STEP", "Layout", "Panel", "default_layout", "kept_layout"]

STEP: Final = "step size"
"""The quantity that is the run's step size, accepted and rejected, against z."""

PANELS: Final = 4
"""The grid's panels."""

PREFERRED: Final = ("energy", "peak_power", "fwhm_t", "peak_intensity")
"""The statistics a new layout shows first, where the run recorded them."""

Panel = tuple[str | None, str | None]
"""A panel's left and right quantities; ``None`` for none."""

Layout = tuple[Panel, ...]
"""The four panels, row by row."""


def default_layout(names: Sequence[str]) -> Layout:
    """Return a first layout for a run's statistics: one per panel, none on the right.

    Parameters
    ----------
    names : Sequence of str
        The statistics the run recorded.

    Returns
    -------
    Layout
        The preferred statistics first, then the others in the run's order; a
        panel left empty when there are fewer than four. The step size is not
        shown unless chosen.

    Examples
    --------
    >>> default_layout(["zdw", "energy"])
    (('energy', None), ('zdw', None), (None, None), (None, None))
    """
    order = [n for n in PREFERRED if n in names] + [
        n for n in names if n not in PREFERRED
    ]
    chosen: list[str | None] = [*order[:PANELS]]
    chosen += [None] * (PANELS - len(chosen))
    return tuple((name, None) for name in chosen)


def kept_layout(layout: Layout, names: Sequence[str]) -> Layout:
    """Return a layout for another run: its choices where the run has them.

    Parameters
    ----------
    layout : Layout
        The layout shown.
    names : Sequence of str
        The statistics the new run recorded.

    Returns
    -------
    Layout
        Each choice the run also has (the step size always) kept; a panel whose
        left choice is gone moves its right one there, or takes the next statistic
        of :func:`default_layout` not shown elsewhere.

    Examples
    --------
    >>> kept_layout((("energy", "zdw"), ("peak_power", None), (None, None),
    ...              ("step size", None)), ["energy", "fwhm_t"])
    (('energy', None), ('fwhm_t', None), (None, None), ('step size', None))
    >>> kept_layout((("zdw", "energy"),) + ((None, None),) * 3, ["energy"])[0]
    ('energy', None)
    """

    def known(name: str | None) -> str | None:
        """Return a choice the run can show, or ``None``.

        Parameters
        ----------
        name : str or None
            The choice.

        Returns
        -------
        str or None
            It, when it is the step size or a statistic of the run.
        """
        return name if name == STEP or name in names else None

    kept = [(known(left), known(right)) for left, right in layout]
    kept = [(right, None) if left is None else (left, right) for left, right in kept]
    shown = {name for panel in kept for name in panel if name is not None}
    spare = [
        name
        for name, _ in default_layout(names)
        if name is not None and name not in shown
    ]
    for index, (left, right) in enumerate(kept):
        if left is None and layout[index][0] is not None and spare:
            kept[index] = (spare.pop(0), right)
    return tuple(kept)
