# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The run glue of a TDSE response (SEP 20): its entry, archive and atomic state.

The ``tdse=`` capillary keywords resolve here, and a run with a TDSE response opens its
archive, prepares and restores its atomic state, and passes the consistency gate before
its march.
"""

import dataclasses
import json
import logging
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

import solprop.driver.tdse as tdse_runtime
from solprop.backends.protocol import Backend
from solprop.backends.session import Start
from solprop.boundaries import absorber as collar
from solprop.boundaries import temporal_collar_rate
from solprop.driver import tdse_consistency
from solprop.driver.run import open_run_record
from solprop.driver.settings import SimulationSettings, _scheme
from solprop.exceptions import DriverError, TDSEArchiveError, TDSECostWarning
from solprop.grid import TimeGrid
from solprop.modes.material import JsonValue
from solprop.modes.quadrature import TransverseQuadrature
from solprop.nonlinear.tdse import TDSEAtomicWindow, TDSEExecution, TDSEGasResponse
from solprop.output.hdf5 import HDF5Output, ResumeState
from solprop.output.schema import Output, RunMetadata
from solprop.output.tdse_archive import TDSEArchive, TDSEArchiveWriter
from solprop.stats import StatsSpec
from solprop.stepper.multistep import Multistep
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import tdse_descriptor
from solprop.translate.tdse import AtomicCallLedger, TDSEBinding, refine_response_grid
from solprop.translate.tdse_source import TDSE_UNAVAILABLE_STATISTICS

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = []

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AtomicContext:
    """Hold what preparing a TDSE run's atomic resources needs from the run.

    Attributes
    ----------
    handler : Output
        The optical output, not yet opened.
    state : ResumeState or None
        What a resumed file holds.
    metadata : RunMetadata
        The run's record.
    """

    handler: Output
    state: ResumeState | None
    metadata: RunMetadata


def _prepare_atomic(
    simulation: "Simulation",
    problem: PropagationProblem,
    context: _AtomicContext,
    tdse_archive: TDSEArchive | None,
) -> tuple[TDSEBinding, TDSEArchiveWriter | None]:
    """Open a TDSE run's archive and prepare its binding.

    - Returns: the binding and the open archive session, or None without a request.
      A failure closes the archive before it propagates.
    """
    descriptor = tdse_descriptor(problem.route)
    assert descriptor is not None
    _warn_tdse_cost(simulation.settings.quadrature)
    execution = simulation.settings.tdse_execution
    if execution is None:
        raise DriverError("TDSE requires explicit tdse_execution settings")
    # Opened and validated before any atomic work (SEP 20, 5.8).
    archive = _open_tdse_archive(
        tdse_archive,
        problem,
        context.handler,
        context.state,
        context.metadata,
        execution,
    )
    try:
        # A haCC binding builds its propagation plans only after the
        # consistency check, so the check's one-row plan never adds to their
        # peak memory (_consistency_gate).
        binding = tdse_runtime.prepare_binding(
            descriptor,
            execution,
            problem.transform.n_nodes,
            defer_plans=descriptor.model == "hacc",
            archive=archive,
        )
    except BaseException as error:
        _close_tdse_archive(archive, error)
        raise
    return binding, archive


def _restore_atomic(binding: TDSEBinding | None, start: Start) -> None:
    """Restore a resumed run's cumulative atomic accounting into its binding.

    Before anything can charge the binding: a haCC run's consistency gate makes a
    diagnostic provider call when its local cache has no entry, and a budget that
    has begun work refuses to be restored. The start keeps only the logical
    ledger, which the session restores when it loads the start.

    - ``binding``: the run's open atomic binding; None for a run without one.
    """
    ledgers = start.ledgers
    if binding is None or not ledgers or not ledgers.get("atomic"):
        return
    atomic = cast("Mapping[str, object]", ledgers["atomic"])
    binding.restore_ledger(AtomicCallLedger.from_record(atomic))
    start.ledgers = {"logical": ledgers.get("logical", {})}


def _consistency_gate(  # noqa: PLR0917 - called positionally
    simulation: "Simulation",
    binding: TDSEBinding | None,
    problem: PropagationProblem,
    handler: Output,
    metadata: RunMetadata,
    backend: Backend,
) -> None:
    """Check a haCC run's weak-field linear response before propagating.

    The cached or computed result is attached to the binding's provenance, under
    ``tdse.consistency``. The binding was prepared with deferred plans, so the
    check's temporary one-row plan is closed before the propagation plans are
    built here. A refused or invalid result is written to the requested
    output, as the opened run record without any plane or checkpoint, before
    :class:`~solprop.exceptions.TDSEConsistencyError` is raised (SEP 20, 5.4).

    - ``binding``: the run's open atomic binding; None for a run without one.
    """
    descriptor = tdse_descriptor(problem.route)
    if binding is None or descriptor is None or descriptor.model != "hacc":
        return
    declared = next(
        one for one in simulation.responses if isinstance(one, TDSEGasResponse)
    )
    if not isinstance(binding, tdse_consistency.GateBinding):
        raise DriverError("a haCC run needs a binding with a diagnostic row")
    build = cast("dict[str, object]", binding.metadata().get("build", {}))
    native = cast("dict[str, object]", build.get("native", {}))
    result = tdse_consistency.gate_run(
        binding,
        declared,
        reference_wavelength=simulation.grid.reference_wavelength,
        provider_revision=tdse_runtime.pinned_revision(),
        computed_on={
            "execution_space": native.get("execution_space"),
            "build": build,
            "threads": native.get("requested_threads"),
        },
    )
    binding.record_consistency(result.to_record())
    if result.outcome in ("refused", "invalid"):
        try:
            open_run_record(handler, metadata, backend, binding)
        except Exception:
            logger.exception("could not record the refused consistency check")
    tdse_consistency.apply_outcome(result)
    # The diagnostic's temporary plan is closed: only now the propagation plans.
    if not binding.plans_built:
        binding.build_plans()


def _check_tdse_archive(
    archive: object, problem: PropagationProblem, handler: Output
) -> None:
    """Refuse an archive request this run cannot honour, before any work.

    - ``archive``: the request, checked here to be a ``TDSEArchive`` or None.
    """
    if archive is None:
        return
    if not isinstance(archive, TDSEArchive):
        raise DriverError("tdse_archive must be a TDSEArchive or None")
    if tdse_descriptor(problem.route) is None:
        raise DriverError(
            "tdse_archive archives a TDSE response's provider evaluations; this run "
            "has no TDSE response"
        )
    if isinstance(handler, HDF5Output) and _same_file(archive.path, handler.path):
        raise DriverError(
            f"the TDSE archive {archive.path} must differ from the optical HDF5 file"
        )


def _same_file(first: Path, second: Path) -> bool:
    """Return whether two paths name the same file, existing or not.

    - Returns: True when they resolve to one location.
    """
    if first.exists() and second.exists():
        return first.samefile(second)
    return first.resolve() == second.resolve()


def _recorded_archive_uuid(state: ResumeState) -> str | None:
    """Return the archive run UUID a resumed optical file records, if any.

    The latest execution's record wins, then the closing records of the earlier
    executions, latest first: an archive first requested by a resume is recorded
    in that execution's outcome, not in the file's original record. A TDSE resume
    always has an outcome (:func:`_validate_resume`).

    - Returns: the UUID, or None when no execution of the run archived.
    """
    outcome = state.outcome
    # `_validate_resume` has refused a TDSE file without its ledgers and outcome.
    assert outcome is not None
    records: list[object] = [outcome.execution]
    records.extend(
        previous.get("execution") for previous in reversed(outcome.previous_executions)
    )
    for record in records:
        tdse = cast("Mapping[str, object]", record).get("tdse")
        archive = tdse.get("archive") if isinstance(tdse, Mapping) else None
        value = (
            cast("Mapping[str, object]", archive).get("run_uuid")
            if isinstance(archive, Mapping)
            else None
        )
        if isinstance(value, str):
            return value
    return None


def _open_tdse_archive(  # noqa: PLR0917 - called positionally
    archive: TDSEArchive | None,
    problem: PropagationProblem,
    handler: Output,
    state: ResumeState | None,
    metadata: RunMetadata,
    execution: TDSEExecution,
) -> TDSEArchiveWriter | None:
    """Open the run's archive session, continuing its own run's archive on a resume.

    - ``state``: what a resumed file holds, or None for a fresh run.
    - Returns: the open session, or None without a request.
    """
    descriptor = tdse_descriptor(problem.route)
    if archive is None or descriptor is None:
        return None
    recorded = None if state is None else _recorded_archive_uuid(state)
    weights = problem.transform.node_weights
    context: dict[str, object] = {
        "descriptor": json.loads(descriptor.configuration_json),
        "descriptor_identity": descriptor.identity,
        "model": descriptor.model,
        "n_nodes": problem.transform.n_nodes,
        "node_weights_m2": None if weights is None else weights.tolist(),
        "quadrature": metadata.quadrature,
        "execution": dataclasses.asdict(execution),
        "provider_pin": tdse_runtime.pinned_revision(),
        # The file's own record on a resume, which the session holds and a snapshot
        # compares with.
        "optical_configuration_hash": (
            metadata if state is None else state.metadata
        ).configuration_hash(),
        "versions": dict(metadata.versions),
    }
    return TDSEArchiveWriter.open(
        archive,
        run_uuid=str(uuid.uuid4()) if recorded is None else recorded,
        kind="run" if state is None else "resume",
        optical=handler.path if isinstance(handler, HDF5Output) else None,
        context=context,
        resume=recorded is not None,
    )


def _close_run(
    backend: Backend | None,
    binding: TDSEBinding | None,
    archive: TDSEArchiveWriter | None,
    handler: Output,
    failure: BaseException | None,
) -> None:
    """Close a run's resources in order, each even when an earlier close fails.

    The backend before the atomic binding, the binding before the archive that
    records it, and the optical output last.

    - ``failure``: the error the run is raising, if any.
    """
    closing = failure
    for resource in (backend, binding, archive, handler):
        if resource is None:
            continue
        try:
            resource.close()
        except BaseException as error:
            if closing is None:
                closing = error
            else:
                closing.add_note(f"a run resource could not be closed: {error}")
    if failure is None and closing is not None:
        raise closing


def _close_tdse_archive(
    archive: TDSEArchiveWriter | None, failure: BaseException | None
) -> None:
    """Close the archive session, never masking the run's own error.

    - ``failure``: the error the run is raising, if any.
    """
    if archive is None:
        return
    try:
        archive.close()
    except TDSEArchiveError as error:
        if failure is None:
            raise
        failure.add_note(f"the TDSE archive could not be closed: {error}")
        logger.exception("could not close the TDSE archive")


def _resolve_tdse_grid(
    grid: TimeGrid, responses: Sequence[object], settings: SimulationSettings
) -> tuple[TimeGrid, SimulationSettings]:
    """Resolve the response lattice once, before building the optical input.

    - Returns: resolved grid and settings with a matching fine temporal collar.
    """
    atomic = [one for one in responses if isinstance(one, TDSEGasResponse)]
    if not atomic:
        if settings.tdse_execution is not None:
            raise DriverError("tdse_execution requires a TDSEGasResponse")
        return grid, settings
    if len(atomic) != 1 or len(responses) != 1:
        raise DriverError("TDSE owns the complete electronic response")
    if settings.tdse_execution is None:
        raise DriverError("TDSE requires explicit tdse_execution settings")
    refined = refine_response_grid(grid, atomic[0].sampling)
    return refined, _refined_tdse_collar(settings, grid, refined)


def _check_tdse_execution(
    simulation: "Simulation", settings: SimulationSettings
) -> None:
    """Refuse unsupported execution choices before any atomic preparation."""
    atomic = any(isinstance(one, TDSEGasResponse) for one in simulation.responses)
    if not atomic:
        if settings.tdse_execution is not None:
            raise DriverError("tdse_execution requires a TDSEGasResponse")
        return
    if (
        settings.tdse_execution is None
        or settings.backend != "numpy_ref"
        or not isinstance(settings.integrator, Multistep)
    ):
        raise DriverError(
            "TDSE requires explicit NumPy, Multistep and tdse_execution settings"
        )
    if simulation.noise is not None and simulation.noise.model == "modified":
        raise DriverError("TDSE refuses noise='modified'; use 'input' or None")
    if settings.schedule is not None or settings.interpolate_planes:
        raise DriverError("TDSE requires adaptive Adams with landed save planes")
    if _scheme(settings) != "nonlinear":
        raise DriverError("TDSE requires the in-stage temporal collar")
    if (
        simulation.spatial_grid is not None
        or simulation.general is not None
        or simulation.wall is not None
    ):
        raise DriverError("TDSE requires the uncoupled modal capillary route")


def _refined_tdse_collar(
    settings: SimulationSettings, parent: TimeGrid, refined: TimeGrid
) -> SimulationSettings:
    """Resample only a recognized analytic collar on the finer response lattice.

    Each rate is rebuilt from its own length (SEP 22 section 5.4): the spectral rate
    and the band from the reference length, the temporal rates from the collar's
    in-stage length, or as zeros for a spectral-only collar.

    - ``settings``: original absorber definition and its physical lengths in m.
    """
    boundary = settings.absorber
    if boundary is None or boundary.temporal_rate_fine.shape == (refined.n_to,):
        return settings
    original = collar(parent, boundary_length=boundary.reference_length)
    length = boundary.in_stage_length

    def temporal(grid: TimeGrid, axis: np.ndarray) -> np.ndarray:
        """Return the collar's temporal rate on a time axis of a grid, 1/m.

        - ``axis``: its coarse or fine time axis, s.
        - Returns: the bounded profile at the in-stage length, or zeros for a
          spectral-only collar.
        """
        if not boundary.temporal:
            return np.zeros(axis.size)
        return temporal_collar_rate(axis, grid.t_collar, length)

    expected = {
        "spectral_rate": original.spectral_rate,
        "band_mask": original.band_mask,
        "temporal_rate": temporal(parent, parent.t),
        "temporal_rate_fine": temporal(parent, parent.t_fine),
    }
    if not all(
        np.array_equal(getattr(boundary, name), value)
        for name, value in expected.items()
    ):
        raise DriverError(
            "TDSE refinement requires a standard analytic collar or a custom "
            "absorber already sampled on the refined response grid"
        )
    return dataclasses.replace(
        settings,
        absorber=dataclasses.replace(
            boundary, temporal_rate_fine=temporal(refined, refined.t_fine)
        ),
    )


@dataclasses.dataclass(frozen=True)
class _TDSEEntry:
    """Hold a TDSE capillary run's resolved response, resources and defaults.

    Attributes
    ----------
    response : TDSEGasResponse
        The declared response, its atomic window's duration resolved.
    execution : TDSEExecution
        Atomic runtime resources.
    backend : str
        The resolved backend, ``numpy_ref``.
    step_planes : int
        Resolved natural-step save interval.
    n_save : int or None
        Resolved requested-plane count; 2 (the entrance and the end) when neither
        `save_planes` nor `n_save` was given.
    record : dict
        The resolved defaults, recorded in the run's configuration.
    """

    response: TDSEGasResponse
    execution: TDSEExecution
    backend: str
    step_planes: int
    n_save: int | None
    record: dict[str, JsonValue]


def _tdse_entry(
    tdse: TDSEGasResponse | None,
    execution: TDSEExecution | None,
    *,
    trange: float | None,
    kerr: bool,
    mode_average: bool,
    unset: Mapping[str, object],
    backend: str,
    step_planes: int | None,
    save_planes: Sequence[float] | np.ndarray | None,
    n_save: int | None,
) -> _TDSEEntry | None:
    """Resolve the ``tdse=`` capillary keywords (SEP 20 sections 5.2 and 5.8).

    - ``tdse``: the declared response, or None for a rate-based run.
    - ``trange``: requested time window in s, the default atomic-window duration.
    - ``kerr``: the Kerr switch, which must stay true beside a TDSE response.
    - ``mode_average``: the mode-averaged switch, which must stay false beside a TDSE
      response: its translation needs the resolved radial quadrature.
    - ``unset``: Electronic-response keywords that must be left unset beside it.
    - ``save_planes``: requested save positions in m.
    - Returns: the resolution, or None without `tdse`.
    """
    if tdse is None:
        if execution is not None:
            raise DriverError(
                "tdse_execution requires a TDSEGasResponse",
                parameter=("tdse_execution", "tdse"),
                rule="tdse-execution-needs-tdse",
            )
        return None
    if not isinstance(tdse, TDSEGasResponse):
        raise DriverError("tdse must be a TDSEGasResponse", parameter="tdse")
    given = [name for name, value in unset.items() if value is not None]
    if not kerr:
        given.append("kerr=False")
    if mode_average:
        given.append("mode_average=True")
    if given:
        raise DriverError(
            f"the TDSE response owns the electronic response; leave "
            f"{', '.join(given)} unset beside tdse=",
            parameter=("tdse", *(name.removesuffix("=False") for name in given)),
            rule="tdse-owns-electronic",
        )
    response = tdse
    window = tdse.atomic_window
    duration_source = None
    if isinstance(window, TDSEAtomicWindow):
        duration_source = "explicit"
        if window.duration is None:
            if trange is None:
                raise DriverError(
                    "the atomic window's duration is unresolved: give trange or "
                    "TDSEAtomicWindow(duration=...)",
                    parameter=("tdse", "trange"),
                )
            response = dataclasses.replace(
                tdse, atomic_window=dataclasses.replace(window, duration=trange)
            )
            duration_source = "trange"
    resolved_backend = "numpy_ref" if backend == "auto" else backend
    if backend == "auto":
        logger.info("backend='auto' resolves to numpy_ref for a TDSE response")
    final_only = save_planes is None and n_save is None
    resolved_steps = 1 if step_planes is None else step_planes
    record: dict[str, JsonValue] = {
        "model": tdse.model,
        "backend": {"requested": backend, "resolved": resolved_backend},
        "step_planes": {"requested": step_planes, "resolved": resolved_steps},
        "requested_planes": "final_only" if final_only else "explicit",
        "atomic_window_duration": duration_source,
        "execution": "default" if execution is None else "explicit",
    }
    return _TDSEEntry(
        response=response,
        execution=TDSEExecution() if execution is None else execution,
        backend=resolved_backend,
        step_planes=resolved_steps,
        n_save=2 if final_only else n_save,
        record=record,
    )


def _entry_overrides[P, R](  # noqa: PLR0917 - called positionally
    atomic: _TDSEEntry | None,
    plasma: P,
    raman: R,
    backend: str,
    n_save: int | None,
    step_planes: int | None,
) -> tuple[_TDSEEntry | None, P, R, str, int | None, int]:
    """Return a TDSE entry with the capillary settings it overrides, or the originals.

    - ``step_planes``: requested natural-step interval; ``None`` is 0 without TDSE.
    - Returns: the entry, the plasma and Raman switches, the backend, the plane count
      and the step interval.
    """
    if atomic is None:
        steps = 0 if step_planes is None else step_planes
        return None, plasma, raman, backend, n_save, steps
    # False is a member of both switches' declared types.
    off = cast("P", False), cast("R", False)
    return atomic, *off, atomic.backend, atomic.n_save, atomic.step_planes


def _tdse_stats(spec: StatsSpec) -> StatsSpec:
    """Remove the statistics a TDSE response cannot supply from a default spec."""
    return dataclasses.replace(
        spec,
        names=tuple(
            name for name in spec.names if name not in TDSE_UNAVAILABLE_STATISTICS
        ),
    )


def _warn_tdse_cost(quadrature: TransverseQuadrature | None) -> None:
    """Warn once when a TDSE run integrates over a many-node radial rule."""
    if quadrature is None or quadrature.radial_rule in (
        "field_scale",
        "field_scale_radau",
    ):
        return
    warnings.warn(
        f"the TDSE response runs on the {quadrature.radial_rule!r} radial rule with "
        f"{quadrature.n_nodes} nodes; each node is one atom in every atomic call, so "
        "the atomic cost is proportional to it. A field-scale rule "
        "(radial_rule='field_scale', a few nodes) integrates the HE11 response at a "
        "fraction of the cost (docs/guide/waveguides/radial-quadrature.md)",
        TDSECostWarning,
        stacklevel=3,
    )
