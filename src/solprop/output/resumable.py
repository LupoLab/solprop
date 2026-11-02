# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Whether a results file can continue, rebuild itself, and pass the gate (SEP 41, S17).

:func:`resumability` answers three questions about a file separately, so a front end
can say which applies without trying a resume:

- **Can it continue, and from where?** The position, and the ``resume=`` value it needs
  (``True``, ``"interpolated"`` or ``"force"``), or why not: no work left, no
  checkpoint written before a failure, another output schema, or a writer still alive.
- **Can it be rebuilt from itself?** Whether it records a recipe (SEP 41, S2), and
  whether this solprop reads its format. A file without one can still continue, from
  the script that wrote it.
- **Does the gate pass?** Only on request, at the cost of a rebuild: the comparison of
  the stored record with the one the recorded recipe rebuilds (SEP 42).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import h5py
import numpy as np

from solprop.exceptions import OutputError, SolpropError
from solprop.output._writer_lock import WriterLock
from solprop.output.hdf5 import resume
from solprop.output.schema import SCHEMA_VERSION, _decode

if TYPE_CHECKING:
    from solprop.output.compatibility import RecordComparison

__all__ = ["Resumability", "ResumeMode", "resumability"]

ResumeMode = Literal[True, "interpolated", "force"]
"""The ``resume=`` value a file needs to continue."""


@dataclass(frozen=True)
class Resumability:
    """What a results file allows: continuing, rebuilding, and the gate's verdict.

    Attributes
    ----------
    path : pathlib.Path
        The file.
    resumable : bool
        Whether a run can continue it.
    z : float or None
        Where it would continue from, m; ``None`` when it cannot.
    mode : {True, 'interpolated', 'force'} or None
        The ``resume=`` value it needs: ``"force"`` for a file a forced session
        wrote, ``"interpolated"`` when its last plane lies inside a step.
    reason : str or None
        Why it cannot continue; ``None`` when it can.
    recipe : {'stored', 'none', 'newer'}
        Whether it records a recipe this solprop reads, none (a file from a script),
        or one of a newer format.
    recipe_format : int or None
        The format of the recipe it records.
    comparison : solprop.output.compatibility.RecordComparison or None
        With ``gate=True`` and a readable recipe, what separates the stored record
        from the one the recipe rebuilds.
    gate_error : str or None
        With ``gate=True``, why the recipe could not be rebuilt.
    """

    path: Path
    resumable: bool
    z: float | None
    mode: ResumeMode | None
    reason: str | None
    recipe: Literal["stored", "none", "newer"]
    recipe_format: int | None = None
    comparison: "RecordComparison | None" = None  # noqa: UP037 - TYPE_CHECKING only
    gate_error: str | None = None

    @property
    def rebuildable(self) -> bool:
        """Return whether the file can rebuild itself.

        Returns
        -------
        bool
            True when it records a recipe this solprop reads.
        """
        return self.recipe == "stored"


def _writer_alive(path: Path) -> bool:
    """Return whether another writer holds a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    bool
        True when its writer lock cannot be taken.
    """
    lock = WriterLock(path)
    try:
        lock.acquire()
    except OutputError as error:
        return "another writer" in str(error)
    lock.release()
    return False


def _recipe_format(path: Path) -> int | None:
    """Return the format of the recipe a file records, or ``None``.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    int or None
        The ``recipe`` group's format attribute.
    """
    with h5py.File(path, "r") as file:
        group = file.get("recipe")
        if not isinstance(group, h5py.Group):
            return None
        return int(cast("int", group.attrs["format"]))


def _schema(path: Path) -> int | None:
    """Return the output schema a file declares.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    int or None
        Its ``schema_version`` attribute, or ``None`` when it has none.
    """
    with h5py.File(path, "r") as file:
        version = file.attrs.get("schema_version")
    return None if version is None else int(cast("int", version))


def resumability(path: str | Path, *, gate: bool = False) -> Resumability:
    """Say whether a results file can continue, rebuild itself, and pass the gate.

    Parameters
    ----------
    path : str or pathlib.Path
        The HDF5 output.
    gate : bool, optional
        Also rebuild the run from its recorded recipe and compare the records, as a
        resume would (SEP 42), so that, for example, a moved optmat digest shows
        before a resume is tried. It costs a rebuild.

    Returns
    -------
    Resumability
        The three answers.

    Raises
    ------
    OutputError
        If the file cannot be read as HDF5.
    """
    from solprop.recipe.document import FORMAT  # noqa: PLC0415 - the layer above

    path = Path(path)
    try:
        schema = _schema(path)
        found = _recipe_format(path)
    except OSError as error:
        msg = f"cannot read {path}: {error}"
        raise OutputError(msg) from error
    recipe: Literal["stored", "none", "newer"] = (
        "none" if found is None else "newer" if found > FORMAT else "stored"
    )

    def refused(reason: str) -> Resumability:
        """Return the answer for a file that cannot continue.

        Parameters
        ----------
        reason : str
            Why.

        Returns
        -------
        Resumability
            The answer.
        """
        return Resumability(path, False, None, None, reason, recipe, found)

    if schema != SCHEMA_VERSION:
        return refused(
            f"the file is of output schema {schema}, and this solprop reads "
            f"{SCHEMA_VERSION} only"
        )
    if _writer_alive(path):
        return refused("a writer still holds the file")
    try:
        state = resume(path)
    except OutputError as error:
        return refused(f"no state to continue from: {error}")
    planes = np.asarray(
        _decode(state.metadata.configuration["save_planes"]),  # type: ignore[arg-type]
        dtype=float,
    )
    position = float(state.plane.z)
    upcoming = int(np.searchsorted(planes, position, side="right"))
    if state.checkpoint is None and upcoming >= planes.size:
        return refused("the run is complete: no work is left")
    if state.plane.field is None:
        return refused(
            "its last plane kept no field and no terminal checkpoint follows it"
        )
    forced = state.outcome is not None and bool(state.outcome.forced_positions)
    mode: ResumeMode = (
        "force" if forced else "interpolated" if state.interpolated else True
    )
    comparison = None
    gate_error = None
    if gate and recipe == "stored":
        from solprop.recipe import Recipe  # noqa: PLC0415 - the layer above

        try:
            rebuilt = Recipe.from_output(path, resolved=True).build().metadata()
            comparison = state.metadata.compare(rebuilt)
        except SolpropError as error:
            gate_error = str(error)
    return Resumability(
        path, True, position, mode, None, recipe, found, comparison, gate_error
    )
