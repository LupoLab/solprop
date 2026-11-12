# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The one save protocol both output handlers implement, and the field retention.

:class:`Output` is the protocol :class:`solprop.output.memory.MemoryOutput` and
:class:`solprop.output.hdf5.HDF5Output` both implement, so processing reads either
through the same accessors; :class:`PlaneStore` is the base class both are built on,
and :class:`FieldRetention` says which planes keep their field. A handler converts each
plane's field to the optics convention on the way to storage
(``docs/dev/conventions.md`` §2; this module and the handlers are boundary place 2).

The manual page is ``docs/guide/results/output.md``.
"""

import dataclasses
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, cast, runtime_checkable

import numpy as np
from jaxtyping import Complex, Int

from solprop import fourier
from solprop._display import Displayed, engineering, line
from solprop._process import parallel_rows
from solprop.exceptions import OutputError
from solprop.modes.material import JsonValue
from solprop.output.metadata import RunMetadata, resolve_metadata_write
from solprop.output.records import (
    Convention,
    RecipeRecord,
    RunOutcome,
    SavedPlane,
    StoredField,
    TerminalCheckpoint,
    _amended_outcome,
    _holds_carried,
    _json_copy,
    _prepend_checkpoint_history,
    _with_phase,
)

__all__ = [
    "CheckpointOutput",
    "FieldRetention",
    "FieldsChoice",
    "Output",
    "PlaneLayout",
    "PlaneStore",
    "RetainingOutput",
    "field_retention_refusal",
    "stored_spectrum",
]

FieldsChoice = Literal["none", "planes", "all"]
"""Which saved planes keep their field: none of them, the named ones, or all."""


def field_retention_refusal(fields: object, field_planes: object) -> str | None:
    """Say what is wrong with a field-retention request, or that nothing is.

    The one validator behind :class:`solprop.driver.settings.SimulationSettings`,
    :func:`solprop.driver.run.run_problem`, :func:`solprop.driver.scan.run_scan` and
    :class:`FieldRetention`; each door raises its own layer's error with the message.

    Parameters
    ----------
    fields : object
        The request, one of ``"none"``, ``"planes"`` and ``"all"``.
    field_planes : object
        Saved-plane indices, the entrance as 0 and negative from the end; required
        exactly when `fields` is ``"planes"``.

    Returns
    -------
    str or None
        Why the request is refused, or ``None`` when it is valid.

    Examples
    --------
    >>> from solprop.output.store import field_retention_refusal
    >>> field_retention_refusal("planes", [0, -1]) is None
    True
    >>> field_retention_refusal("planes", [])
    "field_planes is required exactly when fields='planes'"
    """
    if fields not in ("none", "planes", "all"):
        return f"fields must be none, planes or all, got {fields!r}"
    try:
        planes = tuple(cast("Sequence[object]", field_planes))
    except TypeError:
        return (
            "field_planes must be a sequence of saved-plane indices, got "
            f"{field_planes!r}"
        )
    if any(
        isinstance(index, bool) or not isinstance(index, int | np.integer)
        for index in planes
    ):
        return f"field_planes must hold integer saved-plane indices, got {planes!r}"
    if (fields == "planes") != bool(planes):
        return "field_planes is required exactly when fields='planes'"
    return None


@dataclass(frozen=True)
class FieldRetention:
    """Which saved planes of a run keep their field (SEP 22 section 5.1).

    A run is **homogeneous**: it keeps the field on every plane, on the planes it
    names, or on none, and its statistics on every plane either way. A plane without
    its field stores no ``field`` and no ``noise_phase``; what it saves is its
    position, its statistics and its step history.

    Attributes
    ----------
    fields : {'none', 'planes', 'all'}
        Keep no field, the fields of `field_planes`, or every field (the default).
    field_planes : tuple of int
        Saved-plane indices for ``"planes"``, the entrance as 0 and negative from the
        end until :meth:`resolved`; empty otherwise.

    Raises
    ------
    OutputError
        If the request is not valid (:func:`field_retention_refusal`).

    Examples
    --------
    >>> from solprop.output.store import FieldRetention
    >>> kept = FieldRetention("planes", (0, -1)).resolved(5)
    >>> kept.field_planes, kept.keeps(4), kept.keeps(2), kept.kept(5)
    ((0, 4), True, False, 2)
    """

    fields: FieldsChoice = "all"
    field_planes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate the request and adopt plain integer indices.

        Raises
        ------
        OutputError
            Naming what is wrong.
        """
        refusal = field_retention_refusal(self.fields, self.field_planes)
        if refusal is not None:
            raise OutputError(refusal)
        object.__setattr__(
            self, "field_planes", tuple(int(index) for index in self.field_planes)
        )

    def resolved(self, n_planes: int) -> "FieldRetention":
        """Return the rule with its indices counted from the entrance, sorted, unique.

        Parameters
        ----------
        n_planes : int
            The planes the run saves, the entrance included.

        Returns
        -------
        FieldRetention
            The same rule with non-negative indices.

        Raises
        ------
        OutputError
            If an index lies outside the run's planes.
        """
        resolved = set()
        for index in self.field_planes:
            plane = index + n_planes if index < 0 else index
            if not 0 <= plane < n_planes:
                msg = f"field plane {index} is outside a run of {n_planes} saved planes"
                raise OutputError(msg)
            resolved.add(plane)
        return FieldRetention(self.fields, tuple(sorted(resolved)))

    def keeps(self, index: int) -> bool:
        """Return whether the saved plane at `index` keeps its field.

        Parameters
        ----------
        index : int
            The plane's index, the entrance as 0; the rule must be :meth:`resolved`.

        Returns
        -------
        bool
            True for every plane of an ``"all"`` run, a named plane of a ``"planes"``
            run, and no plane of a ``"none"`` run.
        """
        if self.fields == "all":
            return True
        return index in self.field_planes

    def kept(self, n_planes: int) -> int:
        """Return how many of a run's planes keep their field.

        Parameters
        ----------
        n_planes : int
            The planes the run saves; the rule must be :meth:`resolved`.

        Returns
        -------
        int
            The planes whose field is stored.
        """
        if self.fields == "all":
            return n_planes
        return len(self.field_planes)

    def describe(self) -> str:
        """Return the rule in words, for a refusal naming it.

        Returns
        -------
        str
            For example ``"fields='planes' (saved planes 0, 4)"``.
        """
        if self.fields == "planes":
            planes = ", ".join(str(index) for index in self.field_planes)
            return f"fields='planes' (saved planes {planes})"
        return f"fields={self.fields!r}"


@dataclass(frozen=True)
class PlaneLayout:
    """What every plane of one run has in common, fixed by the first plane saved.

    Attributes
    ----------
    field_shape : tuple of int
        ``(n_modes, n_pol, n_omega)`` of the stored field, or of the run's state when
        it keeps no field; empty when neither is known.
    stat_names : tuple of str
        The statistic names, sorted.
    has_noise_phase : bool
        Whether a noise phase is stored beside each stored field.
    step_stat_names : tuple of str
        The statistics the between-planes channel carries, sorted; empty for a run
        without a periodic subset, which is the default.
    retention : FieldRetention
        Which planes keep their field, resolved; every plane by default.

    Examples
    --------
    >>> from solprop.output.store import PlaneLayout
    >>> layout = PlaneLayout(
    ...     field_shape=(1, 1, 8), stat_names=("energy",), has_noise_phase=False
    ... )
    >>> layout.field_shape, layout.stat_names, layout.retention.fields
    ((1, 1, 8), ('energy',), 'all')
    """

    field_shape: tuple[int, ...]
    stat_names: tuple[str, ...]
    has_noise_phase: bool
    step_stat_names: tuple[str, ...] = ()
    retention: FieldRetention = FieldRetention()


@runtime_checkable
class Output(Protocol):
    """What every output handler does, and all that processing may assume of one.

    One protocol so that :class:`solprop.output.memory.MemoryOutput` and
    :class:`solprop.output.hdf5.HDF5Output` are interchangeable: a run is
    written to memory for a scan and to disk for a long propagation without the driver
    knowing which.

    Everything a handler hands back through ``__getitem__`` is *as stored*, so the field
    is in the optics convention; ``fields`` is the accessor that also offers the
    internal one, for direct comparison with a file stored in the FFT convention.
    """

    def open(self, metadata: RunMetadata, *, force: bool = False) -> None:
        """Begin a run, recording what produced it.

        Parameters
        ----------
        metadata : RunMetadata
            The record of the run about to start.
        force : bool, optional
            Whether to overwrite metadata already held for a different run.
        """
        ...

    def save(self, plane: SavedPlane) -> None:
        """Store one saved plane.

        A run gives the handler the backend's observation itself, not a copy: the
        plane's arrays are read-only, nothing writes to them again, and a handler may
        keep them or reduce them and let them go, with no copy of its own.

        Parameters
        ----------
        plane : SavedPlane
            The position, field, statistics and step history to store. On a plane that
            keeps no field, ``plane.field`` and ``plane.noise_phase`` are ``None``
            (:class:`RetainingOutput`).
        """
        ...

    def close(self) -> None:
        """Finish the run, releasing any resource held. Calling it twice is harmless."""
        ...

    @property
    def n_saved(self) -> int:  # numpydoc ignore=RT01
        """How many planes have been stored."""
        ...

    @property
    def metadata(self) -> RunMetadata | None:  # numpydoc ignore=RT01
        """Record of the run, or ``None`` before it is opened."""
        ...

    @property
    def stat_names(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The names of the statistics stored at every plane.

        The names, sorted; empty before the first plane is saved.
        """
        ...

    @property
    def step_stat_names(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The statistics stored between planes, if any.

        The names, sorted; empty when the run recorded no periodic subset.
        """
        ...

    def __getitem__(self, key: str) -> np.ndarray:
        """Return one stored array by name.

        Parameters
        ----------
        key : str
            ``'field'``, ``'z'``, ``'noise_phase'``, ``'field_planes'``,
            ``'stats/<name>'``, ``'step_history/<name>'`` or ``'step_stats/<name>'``.

        Returns
        -------
        numpy.ndarray
            The stored array, in the convention it is stored in.
        """
        ...

    def __contains__(self, key: object) -> bool:
        """Return whether ``self[key]`` would return an array, never raising.

        Parameters
        ----------
        key : object
            A candidate key.

        Returns
        -------
        bool
            ``True`` exactly when ``__getitem__`` would return an array for it.
        """
        ...

    def fields(self, convention: Convention = "optics") -> np.ndarray:
        """Return every stored field, in the requested sign convention.

        Parameters
        ----------
        convention : {'optics', 'fft'}, optional
            ``'optics'`` returns the arrays as stored; ``'fft'`` converts them back to
            the internal convention a backend works in, for direct comparison with a
            file stored in the FFT convention.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_z, n_modes, n_pol, n_omega)``, complex.
        """
        ...


@runtime_checkable
class CheckpointOutput(Protocol):
    """Declare optional outcome and terminal-checkpoint persistence for a handler."""

    def record_outcome(
        self, outcome: RunOutcome, checkpoint: TerminalCheckpoint | None = None
    ) -> None:
        """Persist execution status and an optional accepted terminal field.

        Parameters
        ----------
        outcome : RunOutcome
            Cumulative work and completion status.
        checkpoint : TerminalCheckpoint or None
            Accepted field between regular saves. None retains a pending checkpoint
            during progress and clears it when the run completes.
        """
        ...

    @property
    def outcome(self) -> RunOutcome | None:  # numpydoc ignore=RT01
        """Recorded status, or None when no outcome was recorded."""
        ...

    @property
    def terminal_checkpoint(self) -> TerminalCheckpoint | None:  # numpydoc ignore=RT01
        """Latest accepted terminal field outside regular observations."""
        ...


@runtime_checkable
class RetainingOutput(Protocol):
    """Declare that a handler accepts field-free planes, and is told the run's rule.

    Separate from :class:`Output`, as :class:`CheckpointOutput` is, because it is
    optional: a handler that stores every field need not implement it, and a run that
    keeps fewer than every field refuses a handler that does not
    (``fields="planes"`` and ``fields="none"``,
    :class:`~solprop.driver.settings.SimulationSettings`). On a plane that keeps no
    field, :meth:`Output.save` is given ``field=None`` and ``noise_phase=None``.

    Examples
    --------
    >>> from solprop.output.store import RetainingOutput
    >>> class Spectra:
    ...     def retain(self, retention, field_shape):
    ...         self.keeps = retention.field_planes
    >>> isinstance(Spectra(), RetainingOutput)
    True
    """

    def retain(self, retention: FieldRetention, field_shape: tuple[int, ...]) -> None:
        """Say which planes of the run keep their field, before the first is saved.

        The run loop (:func:`solprop.driver.run.run_problem`) calls it once per run,
        after :meth:`Output.open` and the handler's ``reserve`` if it has one and
        before any :meth:`Output.save`, and for ``fields="all"`` too.

        Parameters
        ----------
        retention : FieldRetention
            The rule, already :meth:`~FieldRetention.resolved` against the run's planes
            (its indices count from the entrance, and none is negative).
        field_shape : tuple of int
            ``(n_modes, n_pol, n_omega)`` of the run's state, known here even when no
            plane will carry a field.

        Raises
        ------
        OutputError
            A handler that already holds planes under a different rule raises this,
            rather than mixing two rules in one record.
        """
        ...


def stored_spectrum(
    field: Complex[np.ndarray, "*shape n_omega"],
    kind: fourier.GridKind,
    out: Complex[np.ndarray, "*shape n_omega"] | None = None,
) -> Complex[np.ndarray, "*shape n_omega"]:
    """Return :func:`solprop.fourier.spectrum_to_optics` of a large field, in threads.

    One thread's conjugation and sign flip cost more than the state's whole crossing
    from a device, and a state's rows are independent. The bits are the same.

    Parameters
    ----------
    field : numpy.ndarray
        A contiguous spectrum in the internal convention.
    kind : {'field', 'envelope'}
        The grid's spectral layout.
    out : numpy.ndarray or None, optional
        Contiguous room to write it to; a new array by default.

    Returns
    -------
    numpy.ndarray
        `out`, holding the spectrum in the optics convention.
    """
    target = np.empty_like(field) if out is None else out
    rows = field.reshape(-1, field.shape[-1])
    written = target.reshape(-1, field.shape[-1])

    def _convert(selected: slice) -> None:
        """Convert the selected rows into their room."""
        fourier.spectrum_to_optics(rows[selected], kind=kind, out=written[selected])

    parallel_rows(rows.shape[0], field.nbytes, _convert)
    return target


_EXPORT_HOOKS = ("save", "_append", "_stored_field")
"""The plane store methods that see a plane's field, whose owners must vouch for the
exported path, where a kept plane reaches them without one
(``PlaneStore.takes_exports``)."""


def _takes_exports_by_mro(cls: type) -> bool:
    """Return whether a plane store class takes exported fields, by its MRO.

    The nearest class that declares ``_takes_exports`` in its own body and says
    ``False`` opts the class out. Otherwise each of :data:`_EXPORT_HOOKS` is resolved
    as a call would resolve it, and the class that defines it must be the base store
    or declare ``_takes_exports = True`` in its own body: a ``save`` or a field hook
    from a subclass or a mixin that does not vouch for the exported path keeps the
    class on ``save``, which is given every plane with its field.
    """
    mro = cls.__mro__
    declared = next((klass for klass in mro if "_takes_exports" in vars(klass)), None)
    if declared is not None and vars(declared)["_takes_exports"] is not True:
        return False
    for name in _EXPORT_HOOKS:
        owner = next((klass for klass in mro if name in vars(klass)), None)
        if owner is None or owner is PlaneStore:
            continue
        if vars(owner).get("_takes_exports") is not True:
            return False
    return True


class PlaneStore(Displayed, ABC):
    """The shared half of an output handler: one save protocol for every backing store.

    :class:`solprop.output.memory.MemoryOutput` and
    :class:`solprop.output.hdf5.HDF5Output` differ only in where the arrays go. What
    they must not differ in is *what* goes there: the sign convention of the stored
    field, the metadata overwrite rule, the key names and the step-history layout. Those
    live here, once, which is what makes the two handlers store a run bitwise
    identically, with one overwrite rule and one growth strategy.

    The base class also holds the run's **layout** -- the field's shape, the statistic
    names and whether a noise phase is stored -- fixed by the first plane saved and
    checked against every later one, so that a plane that does not fit is refused by
    name in either handler, rather than being dropped or raising deep inside the writer.

    A subclass implements four hooks: ``_write_metadata``, ``_append``, ``_read`` and
    ``_release``, and may implement a fifth, ``_write_annotation``, to persist the
    post-run annotations (:meth:`annotate`) a store held only in memory otherwise.

    A driver that knows how many planes the run will save says so with
    :meth:`reserve`, and each store sizes its first allocation for them; a run that
    saves more grows past it, so the number is a hint, never a limit.

    A run's session exports a kept field straight into the stored form (SEP 47
    section 5.4): :meth:`field_destination` gives the room to fill, a store's own
    (`MemoryOutput`) or none, for a new array (`HDF5Output`), and
    :meth:`save_exported` takes the plane, so that the field is converted once and,
    in memory, written once, where it is kept.

    Examples
    --------
    >>> from solprop.output import MemoryOutput
    >>> from solprop.output.store import Output, PlaneStore
    >>> handler = MemoryOutput()
    >>> isinstance(handler, PlaneStore), isinstance(handler, Output)
    (True, True)
    """

    _takes_exports: ClassVar[bool] = True
    """Declared in a class's own body: whether its own ``save``, ``_append`` and
    ``_stored_field`` keep to the exported path's contract (:attr:`takes_exports`)."""

    _exports: ClassVar[bool] = True
    """Whether a run stores kept fields here through :meth:`save_exported`, resolved
    once per class from its method resolution order (:func:`_takes_exports_by_mro`)."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Resolve whether the subclass takes exported fields, by its MRO.

        Parameters
        ----------
        **kwargs : object
            Passed on to the next class.
        """
        super().__init_subclass__(**kwargs)
        cls._exports = _takes_exports_by_mro(cls)

    def __init__(self) -> None:
        """Start with nothing stored and nothing opened."""
        self._metadata: RunMetadata | None = None
        self._recipe: RecipeRecord | None = None
        self._n_saved = 0
        self._planned = 0
        self._closed = False
        self._layout: PlaneLayout | None = None
        # Which planes keep their field, and the state's shape for a run that keeps
        # none, told by the run before its first plane (`retain`).
        self._retention = FieldRetention()
        self._field_shape: tuple[int, ...] = ()
        self._annotations: dict[str, JsonValue] = {}
        self._outcome: RunOutcome | None = None
        self._terminal_checkpoint: TerminalCheckpoint | None = None
        # The attempts and rows of the checkpoint a resume started from, which no
        # regular plane holds yet: the next plane or checkpoint is preceded by them.
        self._carried: SavedPlane | None = None
        self._resume_outcome_pending = False

    @property
    def outcome(self) -> RunOutcome | None:  # numpydoc ignore=RT01
        """Execution outcome, or None when none was recorded.

        The last recorded completion state and cumulative ledgers.
        """
        return self._outcome

    @property
    def terminal_checkpoint(self) -> TerminalCheckpoint | None:  # numpydoc ignore=RT01
        """Terminal checkpoint without treating it as a saved plane.

        A field in the internal FFT convention and output amplitude units.
        """
        return self._terminal_checkpoint

    def record_recipe(self, record: RecipeRecord) -> None:
        """Record the recipe the run was built from, unless one is recorded already.

        Written once, at the first run start, and never rewritten (SEP 41, S2): a
        later session of the same file, from a script or from an edited recipe,
        leaves the first session's recipe in place and is recorded in its own
        execution record.

        Parameters
        ----------
        record : RecipeRecord
            The authored and resolved texts and the data files.
        """
        self._check_usable()
        if self.recipe is not None:
            return
        self._write_recipe(record)
        self._recipe = record

    def _write_recipe(self, record: RecipeRecord) -> None:
        """Persist the recipe; a store that lives in memory has nothing to do."""

    @property
    def recipe(self) -> RecipeRecord | None:  # numpydoc ignore=RT01
        """The recipe the run was built from, or ``None`` for a run from a script."""
        return self._recipe

    def record_outcome(
        self, outcome: RunOutcome, checkpoint: TerminalCheckpoint | None = None
    ) -> None:
        """Persist status and optional terminal field without extending regular saves.

        Parameters
        ----------
        outcome : RunOutcome
            Current status and cumulative call ledgers.
        checkpoint : TerminalCheckpoint or None
            Accepted field, with no newly requested atomic statistics. With a
            ``"running"`` outcome it is a running checkpoint (SEP 47 section 5.5): an
            accepted endpoint between saved planes, written so that a snapshot or a
            later failure has a state to resume from. Its attempts and periodic rows
            are not consumed: the next regular plane still carries them, and saving
            that plane supersedes the checkpoint. ``None`` keeps a pending checkpoint
            during progress and clears it otherwise; passing back the checkpoint
            held (:attr:`~PlaneStore.terminal_checkpoint`), or a copy, with a closing
            outcome keeps it as the state to resume from.

        Raises
        ------
        OutputError
            If the handler is closed or holds no metadata, or the checkpoint
            disagrees with the outcome or stored field layout.
        """
        self._check_usable()
        self._require_metadata()
        previous = self._outcome
        if previous is not None:
            history = previous.previous_executions
            if self._resume_outcome_pending or (
                previous.status != "running" and outcome.status == "running"
            ):
                closing = dataclasses.asdict(previous)
                closing.pop("previous_executions")
                history = (*history, closing)
            outcome = _amended_outcome(
                outcome,
                previous_executions=tuple(
                    cast("dict[str, object]", _json_copy(dict(record)))
                    for record in history
                ),
            )
        if checkpoint is not None:
            if checkpoint.plane.z != outcome.z:
                raise OutputError("checkpoint and outcome positions must agree")
            if (
                self._layout is not None
                and self._layout.field_shape
                and checkpoint.plane.shape != self._layout.field_shape
            ):
                raise OutputError("checkpoint field shape differs from saved planes")
            if self._n_saved and checkpoint.plane.z < self._last_saved_z():
                raise OutputError("checkpoint precedes the last regular saved plane")
            if self._carried is not None and not _holds_carried(
                checkpoint.plane, self._carried
            ):
                # A new checkpoint is preceded by what a resume carried; the held one,
                # or a copy of it, holds it already. Nothing is consumed: a running
                # checkpoint is superseded, and the next plane is preceded by the
                # same attempts.
                checkpoint = dataclasses.replace(
                    checkpoint,
                    plane=_prepend_checkpoint_history(self._carried, checkpoint.plane),
                )
        elif outcome.status == "running":
            checkpoint = self._terminal_checkpoint
        self._write_outcome(outcome, checkpoint)
        self._outcome = outcome
        self._resume_outcome_pending = False
        self._terminal_checkpoint = checkpoint
        if outcome.status != "running":
            self._carried = None

    def _last_saved_z(self) -> float:
        """Return the position of the last regular saved plane.

        - Returns: the position, m.
        """
        return float(self["z"][-1])

    def _write_outcome(
        self, outcome: RunOutcome | None, checkpoint: TerminalCheckpoint | None
    ) -> None:
        """Persist records; the memory store needs no additional storage hook.

        - ``checkpoint``: accepted checkpoint, or None to clear an older one.
        """

    @property
    def annotations(self) -> Mapping[str, JsonValue]:  # numpydoc ignore=RT01
        """Post-run annotations the store holds, by name.

        An annotation is something learned only once the run has closed -- the material
        manifest's peak-field report
        (:data:`~solprop.output.metadata.PEAK_FIELDS_ANNOTATION`) is one -- so it is not
        part of the run's record: the configuration hash, a resume and a scan's reuse
        never see it. A store without any, a file written before annotations existed
        included, answers with an empty mapping.

        Each annotation's plain JSON values, by name; a copy.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> dict(MemoryOutput().annotations)
        {}
        """
        return cast("dict[str, JsonValue]", json.loads(json.dumps(self._annotations)))

    def annotate(self, name: str, value: object) -> None:
        """Record a post-run annotation, replacing any of the same name.

        The run's record is not touched: an annotation stays outside the
        configuration hash, so a file annotated after its run is still the same run
        to :func:`solprop.output.hdf5.check_cache` and to a scan. A store may be
        annotated after it is closed; a file is then reopened briefly to write it.

        Parameters
        ----------
        name : str
            A non-empty name without ``/``, such as
            :data:`~solprop.output.metadata.PEAK_FIELDS_ANNOTATION`.
        value : object
            Plain JSON values: mappings with string keys, lists, strings, numbers,
            booleans, None; anything else is refused here, the store's boundary.

        Raises
        ------
        OutputError
            If the store was never opened, the name is empty or holds ``/``, or the
            value is not plain JSON.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> from solprop.output.metadata import _example_metadata
        >>> handler = MemoryOutput()
        >>> handler.open(_example_metadata())
        >>> handler.annotate("note", {"peak": 1.5})
        >>> handler.annotations["note"]
        {'peak': 1.5}
        """
        self._require_metadata()
        if not name or "/" in name:
            msg = (
                f"an annotation is named by a non-empty name without '/', got {name!r}"
            )
            raise OutputError(msg)
        try:
            text = json.dumps(value, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            msg = f"annotation {name!r} is not plain JSON: {error}"
            raise OutputError(msg) from error
        self._write_annotation(name, text)
        self._annotations[name] = cast("JsonValue", json.loads(text))

    def _write_annotation(self, name: str, text: str) -> None:
        """Persist one annotation; a store that lives in memory has nothing to do."""

    @property
    def metadata(self) -> RunMetadata | None:  # numpydoc ignore=RT01
        """Record of the run, or ``None`` before it is opened."""
        return self._metadata

    @property
    def n_saved(self) -> int:  # numpydoc ignore=RT01
        """How many planes have been stored."""
        return self._n_saved

    @property
    def closed(self) -> bool:  # numpydoc ignore=RT01
        """Whether the handler has been closed."""
        return self._closed

    @property
    def stat_names(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The names of the statistics stored at every plane.

        Read off the layout the first plane fixed, so that the two handlers answer
        alike, and a file continued or opened for reading answers from what it holds.

        The names, sorted; empty before the first plane is saved.
        """
        return () if self._layout is None else self._layout.stat_names

    @property
    def step_stat_names(self) -> tuple[str, ...]:  # numpydoc ignore=RT01
        """The statistics the between-planes channel carries.

        Empty for a run that asked for no periodic subset, which is how a reader
        tells whether the channel is there to prefer over the planes.

        The names, sorted; empty when the run has no channel.
        """
        return () if self._layout is None else self._layout.step_stat_names

    def _state_text(self) -> str:
        """Return whether the store is open, for printing.

        - Returns: ``"open"`` or ``"closed"``.
        """
        return "closed" if self._closed else "open"

    def _span_text(self) -> str:
        """Return the positions the saved planes span, for printing."""
        if not self._n_saved:
            return "no plane saved yet"
        try:
            z = np.asarray(self["z"], dtype=float)
        except OutputError:
            return "positions unavailable while the store is closed"
        return f"{engineering(float(z[0]), 'm')} to {engineering(float(z[-1]), 'm')}"

    def __repr__(self) -> str:
        """Return a one-line description: how much is stored, and whether it is open.

        Returns
        -------
        str
            For example ``MemoryOutput(201 planes, 2 modes, closed)``.
        """
        fields: list[tuple[str, object]] = [("", f"{self._n_saved} planes")]
        if self._layout is not None:
            fields.append(("", "x".join(str(n) for n in self._layout.field_shape)))
        fields.append(("", self._state_text()))
        return line(type(self).__name__, fields)

    def _describe(self) -> tuple[str, list[tuple[str, object]]]:
        """Return the title and rows of the printed block.

        - Returns: the title and ``(key, value)`` rows.
        """
        rows: list[tuple[str, object]] = [
            ("state", self._state_text()),
            ("planes", f"{self._n_saved}, {self._span_text()}"),
        ]
        if self._layout is not None:
            shape = " x ".join(str(n) for n in self._layout.field_shape)
            rows.append(("field per plane", f"{shape} (modes x components x omega)"))
            rows.append(("statistics", ", ".join(self._layout.stat_names) or "none"))
            between = ", ".join(self._layout.step_stat_names) or "none"
            rows.append(("between planes", between))
        if self._metadata is not None:
            rows.append(("run", repr(self._metadata)))
        return type(self).__name__, rows

    def __enter__(self) -> "PlaneStore":
        """Return the handler, so that a ``with`` block closes it on the way out.

        Returns
        -------
        PlaneStore
            This handler.
        """
        return self

    def __exit__(self, *details: object) -> None:
        """Close the handler when the block ends, whatever happened inside it.

        Parameters
        ----------
        *details : object
            The exception being raised, if any; it is not suppressed.
        """
        self.close()

    def open(self, metadata: RunMetadata, *, force: bool = False) -> None:
        """Begin a run, recording what produced it.

        Parameters
        ----------
        metadata : RunMetadata
            The record of the run about to start.
        force : bool, optional
            Whether to overwrite metadata already held for a different run. Without it a
            second, different record raises; with it the record is replaced and the
            replacement logged
            (:func:`~solprop.output.metadata.resolve_metadata_write`).

        Raises
        ------
        OutputError
            If the handler is closed, or a different record is already held and
            ``force`` is false.
        """
        self._check_usable()
        if resolve_metadata_write(self._metadata, metadata, force=force):
            self._write_metadata(metadata)
        self._metadata = metadata

    def save(self, plane: SavedPlane) -> None:
        """Store one saved plane, converting its field to the optics convention.

        Parameters
        ----------
        plane : SavedPlane
            The position, field, statistics and step history to store. Its field is in
            the internal FFT convention and is written out as
            :func:`solprop.fourier.spectrum_to_optics` of it; a field-free plane
            stores none.

        Raises
        ------
        OutputError
            If the handler is closed or was never opened, or the plane does not fit
            the run's layout: a field where the run keeps none, none where it keeps
            one (:meth:`retain`), another shape or other statistics.
        """
        self._store(plane, None)

    @property
    def takes_exports(self) -> bool:  # numpydoc ignore=RT01
        """Whether a run hands this store its kept fields already stored.

        A store that takes them is given each kept plane through
        :meth:`save_exported`, its field exported by the run's session straight into
        the stored form, and into :meth:`field_destination` (SEP 47 section 5.4). A
        subclass whose ``save``, ``_append`` or ``_stored_field`` comes from a class
        that does not vouch for the exported path -- its own override, or a mixin's
        ahead of the store in its method resolution order -- does not: it sees every
        plane, and its field, through ``save`` as before. A class vouches with
        ``_takes_exports = True`` in its own body, and opts out with ``False``.

        Whether :meth:`save_exported` is how the run stores kept fields here.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> MemoryOutput().takes_exports
        True
        """
        return self._exports

    def field_destination(
        self, plane: SavedPlane, shape: tuple[int, ...]
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return where the next plane's field is to be exported, or ``None``.

        The run's service asks before it exports a kept field (SEP 47 section 5.4). A
        store with room of its own for the field returns that room, a writeable,
        C-contiguous complex128 array of `shape`, and the export fills it once in the
        stored form, the true-power amplitude (or a spatial route's field measure) in
        the optics convention; :meth:`save_exported` then takes the plane. The room is
        the store's: no other plane is given it and nothing writes to it again once the
        plane is stored. A store without such room returns ``None`` and is handed a new
        array, which it takes as its own.

        Parameters
        ----------
        plane : SavedPlane
            The plane about to be stored, without its field: its statistics fix the
            layout of a store's first allocation.
        shape : tuple of int
            ``(n_modes, n_pol, n_omega)`` of the field.

        Returns
        -------
        numpy.ndarray or None
            The room, or ``None`` for a new array. This base class has none.

        Raises
        ------
        OutputError
            If the handler is closed or was never opened, the run keeps no field at
            this plane (:meth:`retain`), or `shape` is not the run's field shape.
        """
        self._check_destination(plane, shape)
        return None

    def save_exported(self, plane: SavedPlane, stored: StoredField) -> None:
        """Store one saved plane whose field was exported already in its stored form.

        The run's service calls it instead of :meth:`save` when the store
        :attr:`takes_exports`; what is stored is what :meth:`save` stores for the same
        field, since the export's conversion is the one :meth:`save` makes.

        Parameters
        ----------
        plane : SavedPlane
            The position, statistics and step history to store, with no field of its
            own.
        stored : StoredField
            Its field in the stored form, in the room :meth:`field_destination`
            returned or in any other array, which is copied in where the store keeps
            fields in its own room; and its noise phase.

        Raises
        ------
        OutputError
            If `plane` carries a field, or for any reason :meth:`save` refuses a
            plane.
        """
        if plane.field is not None:
            msg = (
                "a plane whose field was exported is stored without a field of its "
                "own; save() takes a plane with its field"
            )
            raise OutputError(msg)
        self._store(plane, stored)

    def _check_destination(self, plane: SavedPlane, shape: tuple[int, ...]) -> None:
        """Refuse to give room for a plane that cannot be stored with a field.

        - ``shape``: the field's shape.
        """
        self._check_usable()
        self._require_metadata()
        if plane.field is not None:
            msg = "a plane exported into a store's room carries no field of its own"
            raise OutputError(msg)
        index = self._n_saved
        if not self._retention.keeps(index):
            msg = (
                f"plane {index} keeps no field: the run was saved with "
                f"{self._retention.describe()}"
            )
            raise OutputError(msg)
        held = (
            self._layout.field_shape
            if self._layout is not None and self._fields_stored()
            else self._field_shape
        )
        if held and tuple(shape) != tuple(held):
            msg = (
                f"plane {index} has field shape {tuple(shape)}, but the run stores "
                f"{tuple(held)}"
            )
            raise OutputError(msg)

    def _store(self, plane: SavedPlane, exported: StoredField | None) -> None:
        """Store one plane, its field taken from the plane or already exported.

        - ``exported``: its field in the stored form (:meth:`save_exported`), or
          ``None`` to convert the plane's own field.
        """
        self._check_usable()
        metadata = self._require_metadata()
        if self._carried is not None:
            plane = _prepend_checkpoint_history(self._carried, plane)
        if exported is not None and exported.noise_phase is not None:
            plane = _with_phase(plane, exported.noise_phase)
        self._check_layout(plane, plane.field if exported is None else exported.values)
        if exported is not None:
            stored = self._take_exported(plane, exported.values)
        else:
            stored = (
                None
                if plane.field is None
                else self._stored_field(plane, metadata.grid.kind)
            )
        self._append(plane, stored)
        self._n_saved += 1
        self._carried = None
        if self._terminal_checkpoint is not None:
            # A terminal or running checkpoint is superseded by the plane beyond it
            # (SEP 47 section 5.5): a resume never goes back to it.
            self._write_outcome(self._outcome, None)
            self._terminal_checkpoint = None

    def _take_exported(
        self,
        plane: SavedPlane,
        exported: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return an exported field as the store holds it; this base class takes it.

        A store that keeps fields in a room of its own copies a field exported
        elsewhere into it (`MemoryOutput`).
        """
        del plane
        return exported

    def reserve(self, n_planes: int) -> None:
        """Expect a run of `n_planes` saved planes in all, before the first is saved.

        A store sizes its plane-indexed arrays for this many planes when the first
        plane fixes their shapes, rather than growing them plane by plane; a run that
        saves more grows them past it (by doubling), and one that saves fewer uses
        only what it saved, so the count is a hint and changes nothing that is
        stored. The run loop (:func:`solprop.driver.run.run_problem`) reserves the
        entrance plane and every save plane. A later call can only raise the count.

        Parameters
        ----------
        n_planes : int
            The planes the run will save, the entrance included; non-negative.

        Raises
        ------
        OutputError
            If `n_planes` is negative.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> handler = MemoryOutput()
        >>> handler.reserve(201)
        >>> handler.n_saved
        0
        """
        if n_planes < 0:
            msg = f"a store reserves a non-negative number of planes, got {n_planes}"
            raise OutputError(msg)
        self._planned = max(self._planned, int(n_planes))

    def retain(self, retention: FieldRetention, field_shape: tuple[int, ...]) -> None:
        """Say which planes of the run keep their field, before the first is saved.

        The run loop (:func:`solprop.driver.run.run_problem`) calls it with the run's
        resolved rule. A store then refuses a plane that breaks the rule, sizes its
        field storage for the planes that keep one, and answers field reads on the
        others by naming the rule. A store continuing a run (a resumed file) already
        holds a rule and accepts only the same one.

        Parameters
        ----------
        retention : FieldRetention
            The rule, :meth:`~FieldRetention.resolved` against the run's planes.
        field_shape : tuple of int
            ``(n_modes, n_pol, n_omega)`` of the run's state, which a run that keeps no
            field records in its place.

        Raises
        ------
        OutputError
            If the store already holds planes under another rule.

        Examples
        --------
        >>> from solprop.output import MemoryOutput
        >>> from solprop.output.store import FieldRetention
        >>> handler = MemoryOutput()
        >>> handler.retain(FieldRetention("none"), (1, 1, 8))
        >>> handler.retention.fields, handler.field_shape
        ('none', (1, 1, 8))
        """
        if self._n_saved and retention != self._retention:
            msg = (
                f"this output holds planes saved with {self._retention.describe()}, "
                f"and a run continuing it asks for {retention.describe()}; a run keeps "
                "its fields by one rule, so continue it with the same fields and "
                "field_planes"
            )
            raise OutputError(msg)
        self._retention = retention
        self._field_shape = tuple(int(size) for size in field_shape)

    @property
    def retention(self) -> FieldRetention:  # numpydoc ignore=RT01
        """Which planes of the run keep their field.

        The rule the run declared (:meth:`retain`), every plane by default.
        """
        return self._retention

    @property
    def field_shape(self) -> tuple[int, ...]:  # numpydoc ignore=RT01
        """Shape of one plane's field, stored or not.

        ``(n_modes, n_pol, n_omega)``; empty before anything has said it.
        """
        if self._layout is not None and self._layout.field_shape:
            return self._layout.field_shape
        return self._field_shape

    def close(self) -> None:
        """Finish the run, releasing whatever the store holds open.

        Calling it a second time does nothing, so a driver may close in a ``finally``
        and a caller may close again without checking.
        """
        if self._closed:
            return
        self._release()
        self._closed = True

    def fields(self, convention: Convention = "optics") -> np.ndarray:
        """Return every stored field, in the requested sign convention.

        Parameters
        ----------
        convention : {'optics', 'fft'}, optional
            ``'optics'`` returns the arrays exactly as stored; ``'fft'`` converts them
            back to the internal convention a backend works in, so that a file stored in
            the FFT convention can be compared with them bin by bin.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_z, n_modes, n_pol, n_omega)``, complex.

        Raises
        ------
        OutputError
            If nothing has been stored yet, or the run did not keep the field on
            every plane (:meth:`plane_field` reads a plane that kept one).
        """
        stored = self["field"]
        if convention == "fft":
            kind = self._require_metadata().grid.kind
            return fourier.spectrum_to_internal(stored, kind=kind)
        return stored

    def __getitem__(self, key: str) -> np.ndarray:
        """Return one stored array by name, as stored.

        Parameters
        ----------
        key : str
            ``'field'``, ``'z'``, ``'noise_phase'``, ``'stats/<name>'``,
            ``'step_history/<name>'`` or ``'step_stats/<name>'``. The step history is
            concatenated over planes, with ``'step_history/counts'`` giving the number
            of attempts belonging to each; the between-planes statistics are laid out
            the same way, with ``'step_stats/z'``, ``'step_stats/step'`` and
            ``'step_stats/counts'`` beside the named statistics, and read back empty
            for a run that recorded none; ``'noise_phase'`` is there only when the run
            carried a noise model.

        Returns
        -------
        numpy.ndarray
            The stored array; the field in the optics convention.

        Raises
        ------
        OutputError
            If the key is not one of those forms, or names something not stored; for
            ``'field'`` and ``'noise_phase'``, also when the run did not keep the
            field on every plane, naming its ``fields`` (:meth:`plane_field` reads a
            plane that kept one). ``'field_planes'`` lists the saved planes that keep
            their field, for every run.
        """
        group, name = _split_key(key)
        if group == "" and name == "field_planes":
            return self._field_planes()
        if group == "" and name in ("field", "noise_phase"):
            self._refuse_partial_fields(name)
        return self._read(group, name)

    def _field_planes(self) -> Int[np.ndarray, " n_kept"]:
        """Return the indices of the saved planes that keep their field."""
        indices = range(self._n_saved)
        return np.array(
            [index for index in indices if self._retention.keeps(index)], dtype=np.int64
        )

    def _refuse_partial_fields(self, name: str) -> None:
        """Refuse a whole-run field read on a run that kept fields on some planes only.

        - ``name``: ``'field'`` or ``'noise_phase'``.
        """
        if self._retention.fields == "all":
            return
        where = (
            "no plane"
            if self._retention.fields == "none"
            else "only the saved planes listed by output['field_planes']"
        )
        msg = (
            f"this run was saved with {self._retention.describe()}: its {name} is "
            f"stored on {where}, so there is no {name} for every plane to read; "
            "read a plane that kept its field with plane_field(index), or rerun with "
            "fields='all'"
        )
        raise OutputError(msg)

    def plane_field(self, index: int, convention: Convention = "optics") -> np.ndarray:
        """Return one saved plane's field, in the requested sign convention.

        The accessor that works whatever the run kept: a plane that kept its field
        returns it, and one that did not is refused by name.

        Parameters
        ----------
        index : int
            The saved plane, the entrance as 0 and negative from the end.
        convention : {'optics', 'fft'}, optional
            As for :meth:`fields`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_modes, n_pol, n_omega)``, complex.

        Raises
        ------
        OutputError
            If the index is out of range, or that plane kept no field, naming the
            run's ``fields``.

        Examples
        --------
        >>> import numpy as np
        >>> from solprop.output import MemoryOutput
        >>> from solprop.output.metadata import _example_metadata
        >>> from solprop.output.records import SavedPlane
        >>> handler = MemoryOutput()
        >>> handler.open(_example_metadata())
        >>> n_omega = handler.metadata.grid.n_omega
        >>> handler.save(SavedPlane(z=0.0, field=np.ones((1, 1, n_omega), complex)))
        >>> handler.plane_field(-1).shape
        (1, 1, 2560)
        """
        plane = index + self._n_saved if index < 0 else index
        if not 0 <= plane < self._n_saved:
            msg = (
                f"plane {index} is out of range for a run of {self._n_saved} saved "
                "planes"
            )
            raise OutputError(msg)
        if not self._retention.keeps(plane):
            msg = (
                f"saved plane {plane} kept no field: this run was saved with "
                f"{self._retention.describe()}"
            )
            raise OutputError(msg)
        slot = (
            plane
            if self._retention.fields == "all"
            else self._retention.field_planes.index(plane)
        )
        stored = self._read_field(slot)
        if convention == "fft":
            kind = self._require_metadata().grid.kind
            return fourier.spectrum_to_internal(stored, kind=kind)
        return stored

    def _read_field(self, slot: int) -> np.ndarray:
        """Return one stored field by its row in the field storage.

        - ``slot``: the row: the plane's index in an ``"all"`` run, its position among
          the kept planes otherwise.
        - Returns: the stored field, optics convention.
        """
        return self._read("", "field")[slot]

    def __contains__(self, key: object) -> bool:
        """Return whether ``self[key]`` would return an array.

        Answers for exactly the keys :meth:`__getitem__` accepts, and never raises:
        without it Python's fallback would iterate the handler with integer keys.

        Parameters
        ----------
        key : object
            A candidate key; anything but a string of one of :meth:`__getitem__`'s
            forms is not held.

        Returns
        -------
        bool
            ``True`` when the handler holds the array; ``False`` for an unknown form,
            a name not stored, nothing saved yet or a closed file.
        """
        if not isinstance(key, str):
            return False
        # Asking the reader is the one way to agree with it exactly; the cost is one
        # read of the array, which a membership test is followed by anyway.
        try:
            self[key]
        except OutputError:
            return False
        return True

    def _stored_field(
        self, plane: SavedPlane, kind: fourier.GridKind
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return a plane's field as it is stored, in the optics convention.

        A handler with room for it already converts it there (`MemoryOutput`).

        - ``kind``: the grid's spectral layout.
        """
        assert plane.field is not None
        return stored_spectrum(plane.field, kind)

    def _check_layout(
        self,
        plane: SavedPlane,
        stored: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    ) -> None:
        """Adopt the layout from the first plane, and refuse a later one that differs.

        - ``stored``: its field, in any convention, as only its shape is read; ``None``
          on a field-free plane.
        """
        index = self._n_saved
        retention = self._retention
        if retention.keeps(index) != (stored is not None):
            has = "carries no field" if stored is None else "carries a field"
            msg = (
                f"plane {index} {has}, and the run was saved with "
                f"{retention.describe()}; a run keeps its fields by one rule, declared "
                "before its first plane (retain)"
            )
            raise OutputError(msg)
        layout = PlaneLayout(
            field_shape=self._field_shape if stored is None else stored.shape,
            stat_names=tuple(sorted(plane.stats)),
            has_noise_phase=plane.noise_phase is not None,
            step_stat_names=plane.step_stats.names,
            retention=retention,
        )
        if self._layout is None:
            self._layout = layout
            return
        held = self._layout
        # The entrance plane carries no rows, so the run's row-value names are learnt
        # from the first plane that has any.
        if not held.step_stat_names and layout.step_stat_names:
            held = dataclasses.replace(held, step_stat_names=layout.step_stat_names)
            self._layout = held
        # Whether a noise phase is stored beside the field is learnt from the first
        # plane that kept its field: a field-free plane carries none either way.
        if stored is not None and not self._fields_stored():
            held = dataclasses.replace(
                held,
                field_shape=stored.shape,
                has_noise_phase=layout.has_noise_phase,
            )
            self._layout = held
        if (
            stored is not None
            and held.field_shape
            and layout.field_shape != (held.field_shape)
        ):
            msg = (
                f"plane {index} has field shape {layout.field_shape}, but the run "
                f"stores {held.field_shape}"
            )
            raise OutputError(msg)
        if layout.stat_names != held.stat_names:
            msg = (
                f"plane {index} reports statistics {list(layout.stat_names)}, but the "
                f"run stores {list(held.stat_names)}"
            )
            raise OutputError(msg)
        if stored is not None and layout.has_noise_phase is not held.has_noise_phase:
            has = "carries" if layout.has_noise_phase else "does not carry"
            msg = f"plane {index} {has} a noise phase, unlike those before it"
            raise OutputError(msg)
        # A plane may legitimately carry no rows -- the entrance plane never does, and
        # at a period above one a short plane need not -- so the names are compared
        # only where there are rows to name.
        if layout.step_stat_names and layout.step_stat_names != held.step_stat_names:
            msg = (
                f"plane {index} reports step statistics "
                f"{list(layout.step_stat_names)}, but the run stores "
                f"{list(held.step_stat_names)}"
            )
            raise OutputError(msg)

    def _fields_stored(self) -> int:
        """Return how many of the planes saved so far kept their field."""
        if self._retention.fields == "all":
            return self._n_saved
        return sum(1 for index in self._retention.field_planes if index < self._n_saved)

    def _check_usable(self) -> None:
        """Refuse to write to a closed handler."""
        if self._closed:
            msg = "this output handler is closed; open a new one to store another run"
            raise OutputError(msg)

    def _require_metadata(self) -> RunMetadata:
        """Return the run's metadata, refusing to work without it."""
        if self._metadata is None:
            msg = "call open(metadata) before saving or reading a plane"
            raise OutputError(msg)
        return self._metadata

    @abstractmethod
    def _write_metadata(self, metadata: RunMetadata) -> None:
        """Commit the run's metadata to the backing store."""
        ...

    @abstractmethod
    def _append(
        self,
        plane: SavedPlane,
        stored: Complex[np.ndarray, "n_modes n_pol n_omega"] | None,
    ) -> None:
        """Append one plane to the backing store.

        - ``plane``: the plane: its field in the internal convention, or ``None`` where
          the field was exported straight into the stored form
          (:meth:`~solprop.output.store.PlaneStore.save_exported`) and `stored` alone
          holds it. A subclass whose own ``_append`` reads the plane's field does not
          take exports (``PlaneStore.takes_exports``).
        - ``stored``: that field converted to the optics convention, which is what to
          store; ``None`` on a field-free plane, which stores no field.
        """
        ...

    @abstractmethod
    def _read(self, group: str, name: str) -> np.ndarray:
        """Return one stored array.

        - ``group``: ``''`` for a top-level array, else ``'stats'`` or
          ``'step_history'``.
        """
        ...

    @abstractmethod
    def _release(self) -> None:
        """Release whatever the store holds open. Called once, by :meth:`close`."""
        ...


def _split_key(key: str) -> tuple[str, str]:
    """Split an accessor key into its group and its name.

    - ``key``: ``'field'``, ``'z'``, ``'noise_phase'``, ``'field_planes'``,
      ``'stats/<name>'``, ``'step_history/<name>'`` or ``'step_stats/<name>'``.
    - Returns: ``(group, name)``, the group empty for a top-level array.
    """
    if key in ("field", "z", "noise_phase", "field_planes"):
        return ("", key)
    group, _, name = key.partition("/")
    if group in ("stats", "step_history", "step_stats") and name:
        return (group, name)
    msg = (
        f"unknown key {key!r}; a handler holds 'field', 'z', 'noise_phase', "
        "'field_planes', 'stats/<name>', 'step_history/<name>' and 'step_stats/<name>'"
    )
    raise OutputError(msg)
