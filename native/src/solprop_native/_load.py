# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""What a native run starts from, and the host scalars its stages read.

solprop SEP 47 phase 5 makes the native backend a resource holder that a session
(``solprop_native.session``) or a timing probe (``solprop_native._probe``) loads a start
into. This module holds the native backend's own forms of what solprop's legacy step
driver (``solprop.backends.driver.StepDriver``) did for it, written against the problem
and the request alone:

- :func:`load`: a start's field, noise realisation, position and replay cursor, as the
  native state is loaded with them. The ``input`` noise model adds its draw at the
  launch and never again, and the ``modified`` model keeps its realisation apart
  (``docs/dev/stepper.md``); a resume takes no second draw.
- :class:`StageCache`: a step's stage scalars
  (:func:`solprop.translate.stage.scalars_at`), the anchor hoisted once a step, kept
  until the next hoist.
- :func:`collar_rests` and :func:`propagator_mode`: the run's quiet-collar rule (SEP 21
  D15) and its flow treatment (SEP 24 section 4.3).

Every function is the driver's arithmetic, operation for operation, so a run loaded here
is bitwise the run the driver loaded.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from solprop._helpers import Helpers
from solprop.backends.driver import replay_tolerance
from solprop.exceptions import BackendError
from solprop.noise import NoiseRealisation
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.translate.masks import noise_spectrum
from solprop.translate.problem import (
    PropagationProblem,
    PropagatorMode,
    StepperSettings,
)
from solprop.translate.routes import transverse_of
from solprop.translate.stage import (
    StageScalars,
    StepAnchor,
    anchor,
    scalars,
    scalars_at,
)

type ComplexArray = npt.NDArray[np.complex128]
type RealArray = npt.NDArray[np.float64]

__all__ = [
    "Loaded",
    "StageCache",
    "collar_rests",
    "load",
    "propagator_mode",
    "schedule_index",
]

_EMPTY = np.empty(0, dtype=np.complex128)


@dataclasses.dataclass(frozen=True)
class Loaded:
    """A start as the native state is loaded with it.

    Attributes
    ----------
    field : numpy.ndarray
        The state, flat and contiguous, carrying the ``input`` draw at the launch; empty
        in the record a backend keeps once its native state holds it, so no host copy
        outlives the load.
    noise : numpy.ndarray or None
        The ``modified`` model's realisation, shape ``(1, n_modes, n_pol, n_omega)``,
        which every stage adds; ``None`` under the other models.
    floor : numpy.ndarray or None
        The ``input`` model's draw, the same shape, which rides in the state; the
        boundary monitor reads it as the vacuum floor (SEP 21, Q24).
    clean_columns : numpy.ndarray or None
        Each column's launched energy without the draw, formed from the parts on a
        transverse route's launch (SEP 40); ``None`` elsewhere.
    z : float
        The position, m.
    cursor : int
        The replayed schedule's edge index at `z`; zero without a schedule.
    """

    field: ComplexArray
    noise: ComplexArray | None
    floor: ComplexArray | None
    clean_columns: RealArray | None
    z: float
    cursor: int

    def native_noise(self) -> ComplexArray:
        """Return the realisation the native state adds at every stage, flat.

        Returns
        -------
        numpy.ndarray
            The ``modified`` model's realisation, or an empty array.
        """
        if self.noise is None:
            return _EMPTY
        return np.ascontiguousarray(self.noise.reshape(-1))


def load(
    problem: PropagationProblem,
    stepper: StepperSettings,
    field: ComplexArray,
    noise: NoiseRealisation | None,
    *,
    z: float,
    resume: bool,
) -> Loaded:
    """Return a start's state, noise and position as the native state takes them.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.
    stepper : solprop.translate.problem.StepperSettings
        Its stepping policy, for a replay's cursor.
    field : numpy.ndarray
        The state, shape ``(n_modes, n_pol, n_omega)``, of any strides.
    noise : solprop.noise.NoiseRealisation or None
        The realisation to carry, or ``None`` for the problem's own.
    z : float
        Where the state belongs, m.
    resume : bool
        Whether `field` came back from a saved plane of this run, and so never takes
        the ``input`` draw.

    Returns
    -------
    Loaded
        The load.

    Raises
    ------
    solprop.exceptions.BackendError
        If the field does not match the problem's shape, `z` lies outside the operator
        table, or a schedule is replayed and `z` is not one of its edges.
    """
    values = np.ascontiguousarray(field, dtype=np.complex128)
    if values.shape != problem.shape:
        msg = (
            f"the initial state has shape {values.shape}, expected "
            f"{problem.shape} from the contract"
        )
        raise BackendError(msg)
    # The kernels' layout, a leading axis of one (SEP 47 section 6.2).
    values = values[np.newaxis]
    position = _checked_position(problem, float(z))
    drawn = _noise_field(problem, noise)
    model = problem.initial.noise_model
    clean = None
    if (
        model == "input"
        and drawn is not None
        and not resume
        and position == problem.initial.z
    ):
        if transverse_of(problem.route) is not None:
            clean = _clean_launch_columns(
                values[0], drawn[0], problem.grids.active_mask
            )
        values = values + drawn
    return Loaded(
        field=np.ascontiguousarray(values.reshape(-1)),
        noise=drawn if model == "modified" else None,
        floor=drawn if model == "input" else None,
        clean_columns=clean,
        z=position,
        cursor=schedule_index(stepper, position),
    )


def schedule_index(stepper: StepperSettings, z: float) -> int:
    """Return how many steps of a replayed schedule a position has consumed.

    Parameters
    ----------
    stepper : solprop.translate.problem.StepperSettings
        The stepping policy.
    z : float
        The position, m.

    Returns
    -------
    int
        The index of the schedule's edge at `z`, and 0 without a schedule.

    Raises
    ------
    solprop.exceptions.BackendError
        If a schedule is replayed and `z` is not one of its edges.
    """
    schedule = stepper.schedule
    if schedule is None:
        return 0
    edges = schedule.edges
    index = int(np.argmin(np.abs(edges - z)))
    if abs(float(edges[index]) - z) > replay_tolerance(schedule, z):
        msg = (
            f"a replayed schedule can only be resumed at one of its edges, and "
            f"z = {z:.6g} m is not one; the nearest is {float(edges[index]):.6g} m"
        )
        raise BackendError(msg)
    return index


def _checked_position(problem: PropagationProblem, z: float) -> float:
    """Return `z`, refusing a position off the operator table."""
    low, high = problem.linear.z_span
    if not (math.isfinite(z) and low <= z <= high):
        msg = (
            f"a state can only be loaded inside the operator table, which covers "
            f"[{low:.6g}, {high:.6g}] m, and z = {z!r} does not"
        )
        raise BackendError(msg)
    return z


def _noise_field(
    problem: PropagationProblem, noise: NoiseRealisation | None
) -> ComplexArray | None:
    """Return the realisation to carry with a leading axis of one, or ``None``."""
    if problem.initial.noise_model == "none":
        return None
    if noise is not None:
        return noise_spectrum(noise, problem.shape)[np.newaxis]
    noise0 = problem.initial.noise0
    return None if noise0 is None else noise0[np.newaxis]


def _clean_launch_columns(
    field: ComplexArray, noise: ComplexArray, active: npt.NDArray[np.bool_]
) -> RealArray:
    """Return each column's energy in the masked launch, less its input draw.

    ``mask(field + noise) - noise``: the launch on the active band and the draw alone
    off it, formed from the parts so that no state crosses to subtract the draw.
    """
    inside = np.abs(field[..., active]) ** 2
    outside = np.abs(noise[..., ~active]) ** 2
    return np.sum(inside, axis=(1, 2)) + np.sum(outside, axis=(1, 2))


class StageCache:
    """A step's host stage scalars, its anchor hoisted once (solprop SEP 29 6.6).

    ``Phi`` is cumulative, so every stage of a step subtracts the same anchor:
    :meth:`hoist` evaluates it once. A trial knows its stage positions before its first
    stage, and :meth:`bundles` reads every scalar table once over all of them; the
    bundles equal :meth:`scalars`'s bit for bit and are kept, by anchor and position,
    until the next hoist.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.
    helpers : solprop._helpers.Helpers
        The backend's compiled Clenshaw and basis helpers.
    """

    def __init__(self, problem: PropagationProblem, helpers: Helpers) -> None:
        """Hoist the entrance's anchor; keep no bundle yet."""
        self._problem = problem
        self._helpers = helpers
        self._anchor: StepAnchor = anchor(problem, 0.0, helpers=helpers)
        self._kept: dict[tuple[float, float], StageScalars] = {}

    def hoist(self, z0: float) -> None:
        """Evaluate the anchor every stage of a step at `z0` subtracts; drop the kept.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        """
        self._anchor = anchor(self._problem, z0, helpers=self._helpers)
        self._kept.clear()

    def bundles(
        self, z0: float, positions: Sequence[float]
    ) -> tuple[StageScalars, ...]:
        """Return the scalars of several stages of one step, taken together.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        positions : sequence of float
            The stages' positions, m.

        Returns
        -------
        tuple of solprop.translate.stage.StageScalars
            One bundle per position, in order.
        """
        kept = self._kept
        missing = [z1 for z1 in dict.fromkeys(positions) if (z0, z1) not in kept]
        if missing:
            held = self._anchor if self._anchor.z == z0 else None
            for bundle in scalars_at(
                self._problem, z0, missing, at=held, helpers=self._helpers
            ):
                kept[z0, bundle.z1] = bundle
        return tuple(kept[z0, z1] for z1 in positions)

    def scalars(self, z0: float, z1: float) -> StageScalars:
        """Return one stage's scalars, a kept bundle where one was taken.

        Parameters
        ----------
        z0 : float
            The step's anchor, m.
        z1 : float
            The stage's position, m.

        Returns
        -------
        solprop.translate.stage.StageScalars
            The weights and the scalar path values.
        """
        kept = self._kept.get((z0, z1))
        if kept is not None:
            return kept
        held = self._anchor if self._anchor.z == z0 else None
        return scalars(self._problem, z0, z1, at=held, helpers=self._helpers)


def collar_rests(problem: PropagationProblem, stepper: StepperSettings) -> bool:
    """Return whether a run's stages may leave a quiet temporal collar out (SEP 21).

    A binary64 modal or radial stage with the in-stage (``"nonlinear"``) scheme and a
    temporal collar alone. The low-storage march, coupled flows, the general modal
    route, a real-space collar sharing the term, single precision, and the Cartesian,
    line and planar routes keep the collar in every stage (SEP 21 D15).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.
    stepper : solprop.translate.problem.StepperSettings
        Its stepping policy.

    Returns
    -------
    bool
        True when the term may be left out while the collar is quiet.
    """
    responses = problem.responses
    geometry = problem.transform.geometry
    resting_geometry = geometry is None or (
        geometry.kind == "radial" and not np.any(geometry.absorber_rate)
    )
    return (
        not isinstance(stepper.integrator, LowStorageRungeKutta)
        and responses.temporal_absorber is not None
        and responses.absorber_scheme == "nonlinear"
        and problem.linear.coupled_flow is None
        and problem.transform.modal_tables is None
        and not problem.precision.single
        and resting_geometry
    )


def propagator_mode(
    problem: PropagationProblem, stepper: StepperSettings, *, device: bool
) -> PropagatorMode:
    """Return the flow treatment a native run uses, ``"auto"`` resolved (SEP 24).

    Only the low-storage march holds flows, so every other march recomputes whatever
    was asked (#464). Under ``"auto"`` that march tabulates a constant operator's flow
    on a CPU, one volume reused while the step holds, and forms a varying operator's,
    or any on a device build, per element: there an extra field read costs more than
    the exponential (SEP 24 section 4.3).

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The run's problem.
    stepper : solprop.translate.problem.StepperSettings
        Its stepping policy.
    device : bool
        Whether the build executes on a device.

    Returns
    -------
    str
        ``"tabulate"`` or ``"recompute"``.
    """
    if not isinstance(stepper.integrator, LowStorageRungeKutta):
        return "recompute"
    asked = stepper.propagator_mode
    if asked != "auto":
        return asked
    if device:
        return "recompute"
    return "tabulate" if problem.flags.constant_linop else "recompute"
