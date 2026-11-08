# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Allocation classes: which part of a machine a process holds (solprop SEP 46).

The same machine behaves differently in different allocations: eight cores sharing a
cache are not eight cores across two sockets. A calibration's rows are therefore
keyed by an :class:`AllocationClass` (SEP 46 section 6.4): the cores held and their
kinds, how many shared L2 groups, L3 groups, NUMA nodes and sockets they span,
whether SMT siblings are in use, and a CPU quota below the core count. A process
computes its own class from its affinity mask and quota, and takes the row that
:func:`match` chooses.

This module is standard library only and loads nothing of the package.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ._topology import Topology

__all__ = ["AllocationClass", "Match", "classify", "match"]


@dataclass(frozen=True)
class AllocationClass:
    """The shape of the CPUs a process holds.

    Attributes
    ----------
    cores : int
        Physical cores among its CPUs.
    kinds : tuple of int
        Of those, the cores of each kind, fastest first.
    l2_groups : int
        Shared L2 groups spanned, zero where each core has its own L2.
    l3_groups : int
        L3 groups spanned.
    numa_nodes, sockets : int
        NUMA nodes and packages spanned.
    smt : bool
        Whether two CPUs of one core are held.
    quota : int or None
        A CPU quota below `cores` (control groups, a job's limit), or ``None``.
    """

    cores: int
    kinds: tuple[int, ...]
    l2_groups: int
    l3_groups: int
    numa_nodes: int
    sockets: int
    smt: bool
    quota: int | None = None

    def record(self) -> dict[str, object]:
        """Return the class as JSON data."""
        return {
            "cores": self.cores,
            "kinds": list(self.kinds),
            "l2_groups": self.l2_groups,
            "l3_groups": self.l3_groups,
            "numa_nodes": self.numa_nodes,
            "sockets": self.sockets,
            "smt": self.smt,
            "quota": self.quota,
        }

    @classmethod
    def read(cls, data: Mapping[str, object]) -> AllocationClass:
        """Read a class written by :meth:`record`.

        Raises
        ------
        ValueError
            When `data` is not a class.
        """
        try:
            kinds = data["kinds"]
            quota = data.get("quota")
            if not isinstance(kinds, list) or not all(
                type(kind) is int for kind in cast(list[object], kinds)
            ):
                raise TypeError("kinds")
            counts = {
                name: data[name]
                for name in ("cores", "l2_groups", "l3_groups", "numa_nodes", "sockets")
            }
            if not all(type(value) is int for value in counts.values()):
                raise TypeError("counts")
            if type(data["smt"]) is not bool or not (
                quota is None or type(quota) is int
            ):
                raise TypeError("smt or quota")
        except (KeyError, TypeError) as error:
            raise ValueError(f"not an allocation class: {error}") from error
        return cls(
            cores=counts["cores"],  # pyright: ignore[reportArgumentType] - checked
            kinds=tuple(kinds),  # pyright: ignore[reportUnknownArgumentType] - checked
            l2_groups=counts["l2_groups"],  # pyright: ignore[reportArgumentType]
            l3_groups=counts["l3_groups"],  # pyright: ignore[reportArgumentType]
            numa_nodes=counts["numa_nodes"],  # pyright: ignore[reportArgumentType]
            sockets=counts["sockets"],  # pyright: ignore[reportArgumentType]
            smt=bool(data["smt"]),
            quota=quota,  # pyright: ignore[reportArgumentType] - checked
        )

    def key(self) -> str:
        """Return a short stable name for the class, for claims and attempts."""
        text = json.dumps(self.record(), sort_keys=True)
        return hashlib.blake2b(text.encode(), digest_size=6).hexdigest()

    def span(self) -> tuple[object, ...]:
        """Return the class's span structure: everything but its core count."""
        return (
            len(self.kinds),
            self.l2_groups,
            self.l3_groups,
            self.numa_nodes,
            self.sockets,
            self.smt,
            self.quota is not None,
        )


def classify(
    topology: Topology, cpus: Sequence[int], quota: int | None = None
) -> AllocationClass:
    """Return the class of a process holding `cpus` under `quota` CPUs.

    CPUs `topology` does not list are ignored; an empty mask holds the whole machine.
    """
    held = [cpu for cpu in cpus if cpu in topology.core] or list(topology.cpus)
    cores = {topology.core[cpu] for cpu in held}
    kinds: dict[int, set[int]] = {}
    for cpu in held:
        kinds.setdefault(topology.kind.get(cpu, 0), set()).add(topology.core[cpu])
    machine_l2 = _group_sizes(topology, topology.l2)
    shared_l2 = any(size > 1 for size in machine_l2.values())
    smt = len(held) > len(cores)
    return AllocationClass(
        cores=len(cores),
        kinds=tuple(len(kinds[kind]) for kind in sorted(kinds)),
        l2_groups=len({topology.l2[cpu] for cpu in held}) if shared_l2 else 0,
        l3_groups=len({topology.l3[cpu] for cpu in held}),
        numa_nodes=len({topology.numa[cpu] for cpu in held}),
        sockets=len({topology.socket[cpu] for cpu in held}),
        smt=smt,
        quota=quota if quota is not None and quota < len(cores) else None,
    )


def _group_sizes(topology: Topology, groups: Mapping[int, int]) -> dict[int, int]:
    """Return each group's physical cores."""
    cores: dict[int, set[int]] = {}
    for cpu in topology.cpus:
        cores.setdefault(groups[cpu], set()).add(topology.core[cpu])
    return {group: len(members) for group, members in cores.items()}


@dataclass(frozen=True)
class Match[Row]:
    """The row a process takes, and how much of it applies.

    Attributes
    ----------
    row : Row
        The chosen row.
    exact : bool
        Whether its class is the process's own. Only an exact match supplies process
        settings; a nearer one supplies workspace settings only.
    """

    row: Row
    exact: bool

    @property
    def scope(self) -> Literal["all", "workspace"]:
        """``"all"`` for an exact match, else ``"workspace"``."""
        return "all" if self.exact else "workspace"


def match[Row](
    allocation: AllocationClass, rows: Sequence[tuple[AllocationClass, Row]]
) -> Match[Row] | None:
    """Choose the row for `allocation` among `rows`, by SEP 46 section 6.4.

    An exact match is used whole. Otherwise the row with the same span structure and
    the nearest smaller core count supplies workspace settings, decided by the
    maintainer (SEP 46 section 12 item 5). Otherwise ``None``: the built-in table.
    Among equal classes the last row wins, so a later, higher level replaces an
    earlier one.
    """
    exact = [row for cls, row in rows if cls == allocation]
    if exact:
        return Match(exact[-1], exact=True)
    smaller = [
        (cls.cores, index, row)
        for index, (cls, row) in enumerate(rows)
        if cls.span() == allocation.span() and cls.cores < allocation.cores
    ]
    if smaller:
        return Match(max(smaller, key=lambda item: (item[0], item[1]))[2], exact=False)
    return None
