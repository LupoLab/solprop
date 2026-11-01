# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Prepare compatible fixed-schedule scan members without changing their backend."""

import dataclasses
import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from solprop.backends.protocol import BackendHistory, Observation
from solprop.driver.run import _history_of, plane_from_observation
from solprop.driver.simulation import Simulation
from solprop.exceptions import DriverError
from solprop.output.scan import _json
from solprop.output.schema import SavedPlane
from solprop.translate.problem import KernelProblem

__all__ = [
    "BatchMember",
    "compatible_groups",
    "refuse_mixed_guides",
    "shared_table_bytes",
    "split_observation",
]


@dataclass(frozen=True)
class BatchMember:
    """Bind a global scan index to its independently translated simulation.

    Attributes
    ----------
    index : int
        Zero-based global scan index, retained after grouping and output splitting.
    simulation : Simulation
        Member with its own input, noise, parameter tables and backend selection.
    """

    index: int
    simulation: Simulation


def _signature(member: BatchMember) -> str:
    """Hash the execution layout while keeping member parameter tables independent.

    Parameters
    ----------
    member : BatchMember
        Independently constructed and translated point.

    Returns
    -------
    str
        Digest of the shared grid, algorithm, schedule and response structure.

    Raises
    ------
    DriverError
        If the point has no fixed schedule or uses an unsupported execution contract.
    """
    simulation = member.simulation
    backend = simulation.settings.backend
    if backend not in {"numpy_ref", "numba_cpu", "native"}:
        raise DriverError(
            f"Batch does not support backend {simulation.settings.backend!r}; "
            "use Local or Pool to preserve the selected backend"
        )
    problem = simulation.translate()
    if problem.shape[0] != 1:
        raise DriverError("Batch members must be independently translated singletons")
    if problem.transform.geometry is not None:
        raise DriverError("Batch currently requires fibre geometry")
    if problem.stepper.schedule is None:
        raise DriverError(
            "Batch requires an explicit fixed schedule; discover and calibrate "
            "with solprop.stepper before replay"
        )
    transform = problem.transform
    descriptor = {
        "backend": backend,
        "schema": problem.schema_version,
        "grid_kind": problem.grid_kind,
        "shape": problem.shape,
        "flags": problem.flags,
        "noise_model": problem.initial.noise_model,
        "grids": problem.grids,
        "stepper": problem.stepper,
        "statistics": problem.stats_spec,
        "nodes": transform.n_nodes,
        "node_chunk": transform.node_chunk,
        "transform_shape": transform.synthesis.shape,
        "coarse_projection": transform.projection_coarse is not None,
        "linear_structure": _structure(problem.linear),
        "response_structure": _structure(problem.responses),
        # The general modal route's narrow admission rule. Every other entry above
        # describes an execution *layout*, and two different guides share all of it
        # -- a 1.5 and a 1.6 um strand differ in no shape, no flag and no table
        # extent -- so a general-guide group is separated by what the guides
        # physically **are**: the table's three declared identities and its extents,
        # the metric's kind, gauge and certificate, and the compiled responses' model
        # and policy. Each is a scalar or a string, with the measured entries already
        # quantised by `ModalTransformTables.identity`, so a re-translation on another
        # BLAS does not split a batch that did not change.
        "modal_identity": _modal_identity(problem),
    }
    return hashlib.sha256(_json(descriptor).encode()).hexdigest()


def _modal_identity(problem: KernelProblem) -> Mapping[str, object] | None:
    """Describe which general modal guide a problem carries, or ``None`` for none.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A translated member.

    Returns
    -------
    collections.abc.Mapping or None
        The table, metric and response identities, all scalars and strings, or
        ``None`` where the problem is not a general modal run at all.
    """
    tables = problem.transform.modal_tables
    if tables is None:
        return None
    metric = problem.linear.modal_metric
    responses = problem.responses.regional_responses
    return {
        "tables": dict(tables.identity),
        "metric": None
        if metric is None
        else {
            "kind": metric.kind,
            "gauge_id": metric.gauge_id,
            "certificate_id": metric.certificate_id,
        },
        "responses": None
        if responses is None
        else {
            "model_id": responses.model_id,
            "policy_id": responses.policy_id,
            # The **whole** `(n_intervals, n_deg, n_responses)` array and not the
            # `[0, 0]` row. Today's geometry is constant, so every interval and
            # degree carries the same value and the row would separate the same
            # guides; a z-varying `chi3` would make that silently false and two
            # guides differing only away from the entrance would share a group.
            "chi3": [float(one) for one in np.ravel(responses.chi3)],
            "region": [int(one) for one in responses.response_region],
        },
    }


def refuse_mixed_guides(problems: Sequence[KernelProblem]) -> None:
    """Refuse a formed group whose members are not the same general modal guide.

    The defence the descriptor cannot give. ``_signature`` separates two guides
    into different groups, so this never fires on a group :func:`compatible_groups`
    built; it fires on one assembled by hand or by a future grouping rule, where a
    silently mixed group would propagate one guide's channels against another's
    reference and nothing downstream would say so.

    Parameters
    ----------
    problems : collections.abc.Sequence
        The members' translated problems, in group order.

    Raises
    ------
    DriverError
        Naming the field at which two members part company.
    """
    if not problems:
        return
    first = _modal_identity(problems[0])
    for index, problem in enumerate(problems[1:], start=1):
        other = _modal_identity(problem)
        if _json(other) == _json(first):
            continue
        which = "is a general modal guide and" if first is None else "differs from"
        msg = (
            f"general-guide batch requires identical guide/table/gauge identity: "
            f"member {index} {which} member 0. Members may differ in their launch "
            "and their noise seed and in nothing else; run the others on their own "
            "with Local or Pool"
        )
        raise DriverError(msg)


def _structure(value: object) -> object:
    """Describe table layouts without including their member-dependent values.

    Parameters
    ----------
    value : object
        Compiled response record or one of its data fields.

    Returns
    -------
    object
        Portable structure retaining types, field names and array layouts.
    """
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "type": type(value).__qualname__,
            "fields": {
                field.name: _structure(getattr(value, field.name))
                for field in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        return {key: _structure(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_structure(item) for item in value]
    if isinstance(value, (int, float, complex, np.generic)):
        return value if isinstance(value, bool) else type(value).__name__
    return value


def compatible_groups(
    members: Iterable[BatchMember], size: int
) -> Iterator[tuple[BatchMember, ...]]:
    """Group adjacent compatible points with bounded resident translation products.

    Parameters
    ----------
    members : iterable of BatchMember
        Members in scan dispatch order. Their global indices are never renumbered.
    size : int
        Maximum positive number of members per group.

    Yields
    ------
    tuple of BatchMember
        A full batch, a group ended by a layout change, or the final partial batch.

    Raises
    ------
    DriverError
        If size is invalid, an index is negative, a member is unsupported, or a
        formed group mixes general modal guides
        (:func:`refuse_mixed_guides`, which every yielded group passes through).
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise DriverError("Batch size must be a positive integer")
    current: list[BatchMember] = []
    signature = ""
    backend: str | None = None
    for member in members:
        if member.index < 0:
            raise DriverError("Batch indices must be nonnegative global indices")
        selected = member.simulation.settings.backend
        if backend is not None and selected != backend:
            raise DriverError(
                f"Batch refuses mixed backends {backend!r} and {selected!r}"
            )
        backend = selected
        key = _signature(member)
        if current and key != signature:
            yield _checked(current)
            current = []
        current.append(member)
        signature = key
        if len(current) == size:
            yield _checked(current)
            current = []
    if current:
        yield _checked(current)


def _checked(current: Sequence[BatchMember]) -> tuple[BatchMember, ...]:
    """Return a formed group, refusing one whose members are not the same guide.

    Defence in depth, and the reason the manual can say a mixed general-guide
    group is refused "at the driver and again at the backend": the guide identity
    is in ``_signature``, so this never fires on a group this module built,
    but a future grouping rule that dropped it would be caught here rather than
    propagating one guide's channels against another's reference. The check is a
    JSON comparison of identities the members' cached translations already
    carry, so it costs nothing per group.

    Parameters
    ----------
    current : collections.abc.Sequence
        The members of one formed group, in group order.

    Returns
    -------
    tuple
        The same members.

    Raises
    ------
    DriverError
        Naming the field at which two members part company.
    """
    refuse_mixed_guides([member.simulation.translate() for member in current])
    return tuple(current)


def shared_table_bytes(problems: Iterable[KernelProblem]) -> int:
    """Count host array storage once across shared translated member records.

    Parameters
    ----------
    problems : iterable of KernelProblem
        Translations whose resident host allocations are being admitted.

    Returns
    -------
    int
        Unique backing-array bytes. This excludes native workspace and outputs,
        which must be added from the backend ledger and retention plan separately.
    """
    visited: set[int] = set()
    storage: dict[int, int] = {}

    def visit(value: object) -> None:
        """Track backing allocations without recounting shared records or views.

        Parameters
        ----------
        value : object
            Translated record, collection or array whose storage is being counted.
        """
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, np.ndarray):
            root = value
            while isinstance(root.base, np.ndarray):
                root = root.base
            storage[id(root)] = root.nbytes
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    for problem in problems:
        visit(problem)
    return sum(storage.values())


def split_observation(
    observation: Observation,
    statistics: Mapping[str, np.ndarray],
    *,
    histories: Sequence[BackendHistory],
) -> tuple[SavedPlane, ...]:
    """Separate fields, statistics and error histories into ordinary saved planes.

    Each member retains its own error norms. The shared lockstep history describes
    the attempted positions and acceptance decisions, and is never substituted for
    a member's diagnostic history. No field or statistic gains a new storage axis.

    **The field of every returned plane is the backend's internal, vacuum-normalised
    modal amplitude, not the interface amplitude in true sqrt(W)**. This
    function splits an observation and nothing else; the conversion ``b = sqrt(w) a`` at
    the interface is ``solprop.driver.run._power_observation``, which a run applies to
    the observation before any plane is stored. A caller that wants stored units must
    convert first and split the converted observation.

    Parameters
    ----------
    observation : solprop.backends.protocol.Observation
        Host state with the leading batch axis and shared attempt itinerary.
    statistics : mapping of str to numpy.ndarray
        Saved-plane statistics, each with leading axis ``n_batch``.
    histories : sequence of solprop.backends.protocol.BackendHistory
        One history per member, with that member's error norms. Every itinerary must
        match the observation, including rejected attempts.

    Returns
    -------
    tuple of solprop.output.schema.SavedPlane
        One plane per member, in batch order, preserving species and other axes.

    Raises
    ------
    DriverError
        If the observation carries no field, a member history is missing,
        itineraries differ, or any statistic lacks the required batch axis.
        Validation completes before any plane is returned.
    """
    if observation.field is None:
        raise DriverError("a lockstep batch observes every member's field")
    count = observation.field.shape[0]
    if len(histories) != count:
        raise DriverError(
            f"Batch observation has {count} members but {len(histories)} histories"
        )
    for name, value in statistics.items():
        if value.ndim < 1 or value.shape[0] != count:
            raise DriverError(f"Batch statistic {name!r} must have {count} members")
    for name, value in observation.step_stats.values.items():
        if value.ndim < 2 or value.shape[1] != count:
            raise DriverError(
                f"Batch periodic statistic {name!r} must have {count} members"
            )
    reference = observation.step_history
    for index, history in enumerate(histories):
        if any(
            not np.array_equal(getattr(history, name), getattr(reference, name))
            for name in ("z", "dz", "accepted")
        ):
            raise DriverError(f"Batch member {index} has a different step itinerary")
    return tuple(
        dataclasses.replace(
            plane_from_observation(observation, statistics, member=index),
            step_history=_history_of(history),
        )
        for index, history in enumerate(histories)
    )
