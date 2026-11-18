# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What the installed build's execution space decides for a backend.

The build facts a backend reads once, whether the build runs on an accelerator, the
node chunk the native problem is built with (solprop SEP 47 section 6.5, issue #361)
and the low-storage march's resident flow volumes.
"""

from __future__ import annotations

import dataclasses
import functools
import types
from collections.abc import Mapping
from importlib.metadata import version

from solprop.backends.memory import low_storage_flows
from solprop.backends.node_chunk import NodeChunkPlan, plan_node_chunk
from solprop.backends.session import ExecutionOptions
from solprop.exceptions import BackendError
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.settings import StepperSettings
from solprop.translate.problem import PropagationProblem

from . import _core, build_info

__all__ = [
    "DEVICE_BUDGET_SHARE",
    "_build_facts",
    "_low_storage_flows",
    "_on_device",
    "_resident_flows",
    "device_node_chunk",
    "native_node_chunk",
]


@dataclasses.dataclass(frozen=True)
class _BuildFacts:
    """The parts of a backend's metadata that are fixed for the process."""

    info: Mapping[str, object]
    solprop_version: str
    device_team_size: int
    device_reduction_size: int


@functools.cache
def _build_facts() -> _BuildFacts:
    """Return the build record and the installed solprop version, read once.

    Reading them costs about a millisecond (the package metadata lookup and the native
    record), which a per-plane metadata record paid at every saved plane.
    """
    native_record = {
        key: value for key, value in _core.build_record().items() if value is not None
    }
    return _BuildFacts(
        info=types.MappingProxyType(dataclasses.asdict(build_info())),
        solprop_version=version("solprop"),
        device_team_size=int(native_record["device_team_size"]),
        device_reduction_size=int(native_record["device_reduction_size"]),
    )


def _on_device() -> bool:
    """Return whether the installed build executes on an accelerator."""
    return _build_facts().info["device_backend"] != "none"


DEVICE_BUDGET_SHARE = 4
"""The share of the accelerator's free bytes an automatic plan budgets: one quarter.

The budget is spent on the host plan's conservative inventory (twenty column buffers a
Cartesian column, about three times the live buffers, solprop issue #349), so the live
space-domain buffers take near a twelfth of the free bytes, leaving the rest for the
state and the march's registers."""


def device_node_chunk(
    problem: PropagationProblem, planned: int, free: int | None
) -> int:
    """Return the node chunk a device build takes when no budget or chunk is given.

    solprop issue #361: on a device a node chunk is a kernel launch, and fewer, larger
    launches are faster: one chunk took 0.74-0.92 of the default's time on AMD Instinct
    MI300X spatial routes (SEP 47 Part II), at the cost of space-domain buffers for
    every node at once. So an automatic plan budgets :data:`DEVICE_BUDGET_SHARE`'s share
    of the free device bytes, through the host plan's own rule and response admissions
    (:func:`solprop.backends.node_chunk.plan_node_chunk`): one chunk wherever that fits,
    otherwise the largest that does, and never fewer nodes than the host plan's chunk
    at the route's default budget. It is consulted only for an automatic plan: an
    explicit budget or chunk always wins. It runs at backend construction, after the
    translation, so no device runtime starts before a scan forks.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem.
    planned : int
        The host plan's chunk at the default budget
        (:func:`solprop.backends.node_chunk.plan_node_chunk`).
    free : int or None
        The accelerator's free bytes, or ``None`` where none is known.

    Returns
    -------
    int
        The device's chunk, between 1 and the node count.
    """
    if free is None:
        return planned
    budget = max(1, free // DEVICE_BUDGET_SHARE)
    try:
        roomy = plan_node_chunk(problem, ExecutionOptions(node_budget=budget))
    except BackendError:
        # A quarter of the free bytes cannot hold one node: the default plan stands,
        # and its own allocation reports what does not fit.
        return planned
    return max(planned, roomy.chunk)


def native_node_chunk(
    problem: PropagationProblem, plan: NodeChunkPlan, *, device: bool
) -> int:
    """Lower the host node-chunk plan to the chunk this backend executes.

    A device build's automatic plan goes through :func:`device_node_chunk`. The chunk
    is then bounded by the node count, and on the hybrid planar kind made whole y rows,
    as many as it holds and at least one, which is solprop's reference's own blocking.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem.
    plan : solprop.backends.node_chunk.NodeChunkPlan
        The host plan.
    device : bool
        Whether this build executes on an accelerator.

    Returns
    -------
    int
        The chunk the native problem is built with.
    """
    chunk = plan.chunk
    if device and plan.automatic:
        chunk = device_node_chunk(problem, chunk, _core.available_device_memory())
    chunk = min(chunk, problem.transform.n_nodes)
    geometry = problem.transform.geometry
    if geometry is not None and geometry.kind == "slab":
        n_x, n_y = geometry.shape
        chunk = min(n_y, max(1, chunk // n_x)) * n_x
    return chunk


def _low_storage_flows(
    problem: PropagationProblem, stepper: StepperSettings, *, device: bool
) -> int | None:
    """Return the low-storage march's flow volumes, or None for another march.

    The flows solprop's estimate counts (solprop kernel schema 32) -- except that a
    device build recomputes under ``"auto"``, where the estimate's CPU count
    over-states it by the table. A device's input crosses the state's layout through
    tiles; a CPU's has its own.
    """
    if not isinstance(stepper.integrator, LowStorageRungeKutta):
        return None
    return _resident_flows(low_storage_flows(problem, stepper), stepper, device=device)


def _resident_flows(flows: int, stepper: StepperSettings, *, device: bool) -> int:
    """Return the flow volumes the native state holds for a CPU's count `flows`.

    A device build recomputes under ``"auto"`` and holds none
    (:class:`~solprop.backends.records.LowStorageSettings` counts a CPU's).
    """
    return 0 if device and stepper.propagator_mode == "auto" else flows
