# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Continue a run from the file it wrote: the resume decision and the restart.

A resumed run reads its file's last plane, checks that it may continue from it, and
loads the backend at that plane, with the history it needs for its stepper. The manual
page is ``docs/guide/setup/running.md``.
"""

import logging
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from solprop.backends.ledger import CallLedger
from solprop.backends.protocol import Backend
from solprop.backends.session import Start
from solprop.driver.inputs import _realisation
from solprop.driver.run import RESTART_RECORD, retention_at
from solprop.exceptions import DriverError, SolpropWarning
from solprop.fourier import power_amplitude
from solprop.nonlinear.tdse import TDSEExecution
from solprop.output.hdf5 import HDF5Output, ResumeState, _stored_metadata, check_cache
from solprop.output.hdf5 import resume as resume_state
from solprop.output.memory import MemoryOutput
from solprop.output.metadata import RunMetadata
from solprop.output.records import MarchSnapshot
from solprop.output.store import Output
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import tdse_descriptor
from solprop.translate.run_request import RunRequest
from solprop.translate.tdse import AtomicCallLedger

if TYPE_CHECKING:
    from solprop.driver.simulation import Simulation

__all__ = []

logger = logging.getLogger(__name__)

type _ResumeChoice = bool | Literal["interpolated"]
"""What ``resume`` takes: no, yes, or from a plane interpolated inside a step."""


def _adopt_recorded_seed(
    simulation: "Simulation", output: Output | Path | str | None
) -> None:
    """Take the noise seed of the file a resume continues, where the call gave none.

    The maintainers' ruling on resuming a default call (``seed=None``): the
    realisation is drawn again from the seed the file recorded, so the re-run
    call describes the run it continues. Nothing happens when there is no file,
    the file recorded no seed or another noise model, or the realisation was
    seeded or supplied by the caller; :func:`~solprop.output.hdf5.check_cache`
    then judges the run as before.
    """
    if (
        simulation._noise_redraw is None
        or simulation.noise is None
        or not isinstance(output, Path | str)
        or not Path(output).exists()
    ):
        return
    stored = _stored_metadata(output)[0].noise
    if stored is None or stored.seed is None or stored.model != simulation.noise.model:
        return
    if simulation._reseed(stored.seed):
        logger.info(
            "resuming %s: the noise was drawn without a seed, so it is drawn again "
            "from the seed the file recorded, %d",
            output,
            stored.seed,
        )


def _start(
    simulation: "Simulation",
    backend: Backend,
    problem: PropagationProblem,
    state: ResumeState | None,
    *,
    kept: bool = False,
    restoration: dict[str, object] | None = None,
) -> Start:
    """Return where the run starts: the entrance, or the state a file resumes.

    The run's session loads it (SEP 47): the field, the noise, a resume's call
    ledgers and the method history this run continues.

    - ``backend``: the backend about to step, asked only whether its march can carry
      a method history.
    - ``state``: what a file the run is continuing holds, or ``None`` to start at
      the entrance.
    - ``kept``: the backend was built for another input, so this one's noise has to
      be handed to it.
    """
    if state is None:
        return Start.entrance(problem, noise=_realisation(problem) if kept else None)
    request = simulation._run_request()
    next_plane = _validate_resume(
        problem, request, state, simulation.settings.tdse_execution
    )
    logger.info(
        "resuming at z = %.6e m, plane %d of %d",
        state.plane.z,
        state.n_saved,
        request.observation.save_planes.size + 1,
    )
    # A field-free plane resumes only a finished run, which never gets here.
    assert state.plane.field is not None
    field = state.plane.field
    if problem.transform.geometry is None:
        field = power_amplitude(
            field[np.newaxis], power_weight_at(problem, state.plane.z), inverse=True
        )[0]
    integrator = request.integration.method
    record, march = _continue_march(
        simulation,
        backend,
        state,
        adams=isinstance(integrator, Multistep),
        low_storage=isinstance(integrator, LowStorageRungeKutta),
    )
    if restoration is not None:
        restoration.update(record)
    return Start(
        field=field,
        z=float(state.plane.z),
        plane_index=next_plane,
        march=march,
        ledgers=None
        if state.outcome is None
        else {
            "logical": state.outcome.logical_calls,
            "atomic": state.outcome.atomic_calls,
        },
        resume=True,
    )


def _continue_march(
    simulation: "Simulation",
    backend: Backend,
    state: ResumeState,
    *,
    adams: bool = True,
    low_storage: bool = False,
) -> tuple[dict[str, object], MarchSnapshot | None]:
    """Decide whether the Adams history continues from the file's snapshot.

    Says what is restored, for the resumed execution record (SEP 47 section 5.5),
    and warns once when an Adams history the run expected is not restored:
    omitted by the automatic size limit, or kept where this backend's march
    cannot carry it. A history set aside on request (``resume_history=False``) is
    recorded without a warning. A history handed on is installed by the run's
    session, which warns and records a reset if it cannot
    (:func:`solprop.driver.run.run_problem`).

    - Returns ``record``: ``"state"`` (``"restored"``, ``"reset"`` or
      ``"not_applicable"``), ``"reason"`` and ``"z"``, m; for an Adams method also
      ``"retention"``, what the file states it kept there, ``{"state": "unknown"}``
      where it states nothing.
    - Returns ``march``: the history the run's session is to install, or ``None``.
    """
    z = state.plane.z
    if low_storage:
        record = {"state": "not_applicable", "reason": "no_method_history", "z": z}
        return record, None
    if not adams:
        # The proposal and the norm memory start afresh (issue #596).
        return {"state": "reset", "reason": "unsupported_method", "z": z}, None
    execution = None if state.outcome is None else state.outcome.execution
    recorded = retention_at(execution, z)
    retention: dict[str, object] = (
        dict(recorded)
        if recorded is not None
        else {
            "state": "unknown",
            # A file from before the restart record, or one that states nothing
            # at this position: never an invented reason.
            "reason": "legacy"
            if execution is None or RESTART_RECORD not in execution
            else "not_recorded",
        }
    )

    def reset(reason: str) -> dict[str, object]:
        """Return the restoration record of a history that starts afresh.

        - Returns: the record's ``"state"``, ``"reason"``, ``"z"`` and
          ``"retention"``.
        """
        return {"state": "reset", "reason": reason, "z": z, "retention": retention}

    if simulation.settings.resume_history is False:
        if state.march is not None:
            logger.info("the Adams history starts afresh: resume_history=False")
        return reset("explicit_policy"), None
    # A march that cannot carry its history restarts it without having been asked
    # to: say so once, as a warning (SEP 29 section 6.6).
    refusal = backend.march_export_refusal()
    if refusal is not None:
        warnings.warn(
            f"the Adams history starts afresh: {refusal}",
            SolpropWarning,
            stacklevel=2,
        )
        return reset("incompatible_representation"), None
    if state.march is None:
        return _missing_history(retention, reset), None
    restored: dict[str, object] = {
        "state": "restored",
        "reason": "retained_history",
        "z": z,
        "retention": retention,
    }
    return restored, state.march


def _missing_history(
    retention: Mapping[str, object],
    reset: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    """Record, and warn about, an Adams resume whose file holds no history."""
    kept, why = retention.get("state"), retention.get("reason")
    if kept == "omitted" and why == "automatic_size_limit":
        size = retention.get("nbytes")
        limit = retention.get("limit")
        warnings.warn(
            "the Adams history starts afresh: the run did not keep it for a "
            "resume, "
            + ("" if size is None else f"{size} bytes being ")
            + "over the "
            + ("automatic limit" if limit is None else f"{limit} bytes")
            + " kept by default (resume_history=True keeps it)",
            SolpropWarning,
            stacklevel=3,
        )
        return reset("automatic_size_limit")
    lost: tuple[str, str] | None = None
    if kept == "retained":
        lost = ("kept it here, but it cannot be read back", "unavailable_history")
    elif kept == "omitted" and why == "write_failed":
        lost = ("was keeping it here when the write failed", "unavailable_history")
    elif kept == "omitted" and why == "incompatible_representation":
        lost = (
            "could not keep it: the run's engine does not export it",
            "incompatible_representation",
        )
    if lost is not None:
        warnings.warn(
            f"the Adams history starts afresh: the file {lost[0]}",
            SolpropWarning,
            stacklevel=3,
        )
        return reset(lost[1])
    if kept == "omitted" and why == "explicit_policy":
        return reset("explicit_policy")
    # Nothing to continue (a position the march held no history at), or a file
    # that does not say: as before, the history starts afresh without a warning.
    return reset("unavailable_history")


def _validate_resume(
    problem: PropagationProblem,
    request: RunRequest,
    state: ResumeState,
    execution: TDSEExecution | None,
) -> int:
    """Validate restart position and budgets before preparing atomic resources.

    - ``request``: its run's request, with the absolute save positions in m.
    - ``execution``: current atomic budget, if this is a TDSE run.
    - Returns: index of the next scheduled save strictly beyond the accepted position.
    """
    planes = request.observation.save_planes
    next_plane = int(np.searchsorted(planes, state.plane.z, side="right"))
    if state.checkpoint is not None and state.checkpoint.next_save_index != next_plane:
        raise DriverError(
            "checkpoint next_save_index disagrees with its absolute position"
        )
    if not problem.initial.z <= state.plane.z <= planes[-1]:
        raise DriverError(
            "resume position lies outside the declared propagation interval"
        )
    if state.plane.field is None:
        if _work_left(request, state, next_plane):
            msg = (
                f"the run's last saved plane, at z = {state.plane.z:.6e} m, kept no "
                "field (it was saved with fields='none' or 'planes') and the file "
                "holds no terminal checkpoint, which always carries the field, so "
                "there is no state to continue from: a run stopped by an error or a "
                "signal writes one, a run killed outright does not. Rerun it, or keep "
                "the field on its last plane"
            )
            raise DriverError(msg)
    elif not np.all(np.isfinite(state.plane.field)):
        raise DriverError("resume requires a finite stored field")
    outcome = state.outcome
    if tdse_descriptor(problem.route) is not None and (
        outcome is None or not outcome.logical_calls or not outcome.atomic_calls
    ):
        raise DriverError("a TDSE resume requires persisted cumulative call ledgers")
    method = request.integration.method
    if outcome is not None and outcome.logical_calls and isinstance(method, Multistep):
        calls = CallLedger.from_record(outcome.logical_calls)
        if method.max_calls is not None and calls.total > method.max_calls:
            raise DriverError("logical call budget is below previously charged work")
    if outcome is not None and outcome.atomic_calls:
        atomic = AtomicCallLedger.from_record(outcome.atomic_calls)
        if tdse_descriptor(problem.route) is None or execution is None:
            raise DriverError(
                "an atomic call ledger requires an explicit TDSE execution"
            )
        if (
            execution.max_atomic_calls is not None
            and atomic.attempted > execution.max_atomic_calls
        ):
            raise DriverError("atomic call budget is below previously attempted work")
    return next_plane


def _interpolated(state: ResumeState) -> bool:
    """Return whether the plane a file resumes from was interpolated inside a step.

    A landed plane is the end of an accepted step of the run; an interpolated one lies
    inside a step. The entrance is the initial field.
    """
    return state.interpolated


def _open_output(
    output: Output | Path | str | None,
    metadata: RunMetadata,
    *,
    resume: _ResumeChoice,
) -> tuple[Output, ResumeState | None]:
    """Choose the handler a run writes to, and read what a resumed file holds.

    The resume gate (ADR-066): the file's configuration hash must equal `metadata`'s.
    Whether work is left is decided by :func:`_continues`.
    """
    if resume and metadata.configuration.get("_benchmark_plasma_source"):
        raise DriverError("gated plasma benchmark output cannot be resumed")
    if isinstance(output, Path | str):
        path = Path(output)
        if resume and check_cache(path, metadata):
            return HDF5Output(path, append=True), resume_state(path)
        return HDF5Output(path), None
    if resume:
        msg = (
            "only a path can be resumed: the file has to be opened in append mode, "
            "which is the driver's to decide; pass output=<path> with resume=True"
        )
        raise DriverError(msg)
    return (MemoryOutput() if output is None else output), None


def _check_resume_choice(resume: object) -> None:
    """Refuse a ``resume`` value that is not one of its three.

    A mistyped word would otherwise read as ``True``.
    """
    if resume is True or resume is False or resume == "interpolated":
        return
    msg = f"resume must be False, True or 'interpolated', got {resume!r}"
    raise ValueError(msg)


def _work_left(request: RunRequest, state: ResumeState, next_plane: int) -> bool:
    """Return whether a file has anything left to compute, by position.

    Decided by position, not status: a file killed after its last plane has no work
    left, though its status reads ``running``; a terminal checkpoint still owes its
    plane.
    """
    return (
        state.checkpoint is not None
        or next_plane < request.observation.save_planes.size
    )


def _refuse_interpolated(path: Path, state: ResumeState, resume: _ResumeChoice) -> None:
    """Refuse to continue from a plane interpolated inside a step, unless accepted.

    An interpolated plane is not a state the march passed through, so a run restarted
    there is not the run it continues (SEP 14 section 5); ``resume="interpolated"``
    accepts that.
    """
    if resume == "interpolated" or not _interpolated(state):
        return
    msg = (
        f"{path} ends on a plane interpolated inside a step at z = "
        f"{state.plane.z:.6e} m, not a state the march passed through; "
        "pass resume='interpolated' to continue afresh from it, or rerun"
    )
    raise DriverError(msg)


def _continues(
    path: Path,
    request: RunRequest,
    state: ResumeState,
    next_plane: int,
    resume: _ResumeChoice,
) -> bool:
    """Say whether a resume of a file computes anything, once its hash has passed.

    - Returns: ``True`` when work is left, and ``False`` for a finished file.
    """
    if not _work_left(request, state, next_plane):
        return False
    _refuse_interpolated(path, state, resume)
    return True
