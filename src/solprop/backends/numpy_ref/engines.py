# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's Runge-Kutta engines over its workspace (SEP 52 section 5.2).

Two :class:`~solprop.backends.engines.RungeKuttaEngine` implementations, the
counterparts of native's resident engines. :class:`NumpyRungeKutta` forms an
interaction-picture embedded Runge-Kutta trial from the workspace's committed state
and keeps its stages and end state until :meth:`~NumpyRungeKutta.accept` finishes and
commits it; a rejected trial leaves the committed state as it was.
:class:`NumpyLowStorage` is the low-storage RK4 step of SEP 24
(:mod:`solprop.backends.low_storage`), whose trial overwrites the state in place and
is in flight until it is accepted. The Adams engine is
:class:`solprop.backends.numpy_ref.multistep.HostHistory`.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Complex

from solprop.backends.engines import AcceptRecord, BoundaryReading
from solprop.backends.low_storage import schedule_key, table_steps
from solprop.backends.numpy_ref.passes.stepping import (
    accept_step,
    accumulate,
    stage_input,
)
from solprop.exceptions import BackendError
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.schedule import FixedSchedule
from solprop.stepper.tableau import Tableau, extension_coefficients

if TYPE_CHECKING:
    from solprop.backends.numpy_ref.workspace import NumpyWorkspace

__all__ = ["NumpyLowStorage", "NumpyRungeKutta"]

_State = Complex[np.ndarray, "n_modes n_pol n_omega"]
_Extension = tuple[float, float, list[np.ndarray]]
"""A continuous extension: its step's start and length, m, and its coefficients."""


class NumpyRungeKutta:
    """The interaction-picture embedded Runge-Kutta trial over a workspace.

    Every stage is evaluated. The tableau's first-same-as-last property would let the
    last stage of one step serve as the first of the next, after rebasing it by
    ``exp(dPhi)`` because the interaction picture is re-anchored every step; the
    reference does not take that saving, so that no stage it uses depends on the
    bookkeeping of the step before (``docs/dev/stepper.md`` section 10, trap 10).

    Parameters
    ----------
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The workspace whose state is stepped.
    tableau : solprop.stepper.tableau.Tableau
        The embedded pair.

    Attributes
    ----------
    evaluations : int
        Physical source evaluations of the trials since the engine was built, for one
        run, rejected trials included (SEP 30 records work rather than estimating it
        from steps).
    """

    def __init__(self, workspace: "NumpyWorkspace", tableau: Tableau) -> None:
        """Hold the workspace; no trial is held yet."""
        self._workspace = workspace
        self._tableau = tableau
        self._trial: tuple[_State, list[_State]] | None = None
        self._extension: tuple[float, float, list[np.ndarray]] | None = None
        self.evaluations = 0

    @property
    def in_flight(self) -> bool:  # numpydoc ignore=RT01
        """Whether a trial overwrote the committed state; never, on this engine."""
        return False

    @property
    def trial_state(self) -> _State:  # numpydoc ignore=RT01
        """The held trial's raw end state, before the accepted step's finish."""
        if self._trial is None:
            raise BackendError("take a trial before reading its state")
        return self._trial[0]

    @property
    def stages(self) -> list[_State]:  # numpydoc ignore=RT01
        """The held trial's stage derivatives, in its anchor's frame."""
        if self._trial is None:
            raise BackendError("take a trial before reading its stages")
        return list(self._trial[1])

    def trial(self, z: float, dz: float) -> float:
        """Form a trial from the committed state at `z` over `dz`; return its norm.

        A problem carrying ``linear.coupled_flow`` multiplies each stage's scalar
        propagator by the section's Schur matrix-function action; the two factors
        commute, so their product is the flow of the sum. A ``"frozen_residual"`` flow
        adds the explicit residual to each stage's lab-frame source, before the
        interaction picture's own factors carry it back, so it is integrated by the
        embedded pair exactly as the material response is.

        Parameters
        ----------
        z : float
            Where the trial starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            The scaled error norm of the step's embedded difference.
        """
        workspace = self._workspace
        state = workspace.committed()
        workspace.hoist(z)
        tableau = self._tableau
        # The whole step and every stage, read together: each table once.
        workspace.stage_bundles(
            z, [z + dz, *(z + float(node) * dz for node in tableau.nodes)]
        )
        exponents = workspace.exponents()
        step = workspace.stage_scalars(z, z + dz)
        actions = workspace.flow_actions(step.flow_section, dz)
        stages: list[_State] = []
        solution = np.zeros_like(state)
        error = np.zeros_like(state)
        for index, node in enumerate(tableau.nodes):
            fraction = float(node)
            combination = np.zeros_like(state)
            for previous in range(index):
                combination = accumulate(
                    combination,
                    stages[previous],
                    float(tableau.matrix[index, previous]),
                )
            bundle = workspace.stage_scalars(z, z + fraction * dz)
            exponent = exponents(fraction, bundle)
            forward, inverse = actions(fraction)
            lab = stage_input(
                workspace.channels(state, forward),
                workspace.channels(combination, forward),
                dz,
                np.exp(exponent),
                workspace.noise_at(bundle),
            )
            back = np.exp(-exponent)
            self.evaluations += 1
            drive = workspace.nonlinear(lab, bundle, back)
            residual = workspace.residual(lab, step.flow_section, bundle)
            if residual is not None:
                drive = drive + back * residual
            stage = workspace.channels(drive, inverse)
            stages.append(stage)
            solution = accumulate(
                solution, stage, float(tableau.solution_weights[index])
            )
            error = accumulate(error, stage, float(tableau.error_weights[index]))
        propagator = np.exp(exponents(1.0, step))
        whole = actions(1.0)[0]
        advanced = accept_step(
            workspace.channels(state, whole),
            propagator,
            dz,
            workspace.channels(solution, whole),
        )
        # The embedded difference is carried to the end of the step by the same
        # exponential as the solution, so the norm scales the local error of the
        # propagated state rather than of the interaction variable.
        norm = workspace.error_norm(
            propagator * (dz * workspace.channels(error, whole)), state, advanced
        )
        # Kept for the accept, and for the continuous extension of an accepted step
        # (SEP 14 phase 3).
        self._trial = (advanced, stages)
        return norm

    def accept(self, dz: float) -> AcceptRecord:
        """Finish the held trial and commit it as the workspace's state.

        Parameters
        ----------
        dz : float
            The accepted step, m.

        Returns
        -------
        solprop.backends.engines.AcceptRecord
            The finish's split collar's sums.
        """
        finished, record = self._workspace.finish(self.trial_state, dz)
        self._workspace.install(finished)
        return record

    def extend(self, z: float, dz: float) -> None:
        """Hold the held trial's continuous extension, before it is accepted.

        With ``v0`` the state at the step's anchor, ``k_i`` its stage derivatives in
        the anchor's frame and ``v1 = v0 + h sum_i b_i k_i``, the coefficients are
        DOPRI5's (Hairer, Noersett and Wanner, II.6): ``v0``, ``v1 - v0``, ``h k1 - (v1
        - v0)``, ``(v1 - v0) - h k7 - (h k1 - (v1 - v0))`` and ``h sum_i d_i k_i``.

        Parameters
        ----------
        z : float
            Where the trial started, m.
        dz : float
            The step, m.
        """
        workspace = self._workspace
        start = workspace.committed().reshape(workspace.problem.shape)
        coefficients = extension_coefficients(self._tableau, start, self.stages, dz)
        self._extension = (z, dz, coefficients)

    @property
    def extension(self) -> "_Extension | None":  # numpydoc ignore=RT01
        """The held extension: its step's start and length, m, and its coefficients."""
        return self._extension

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading.

        Returns
        -------
        solprop.backends.engines.BoundaryReading
            The workspace's reading.
        """
        return self._workspace.reading()


class NumpyLowStorage:
    """The host engine: the four register passes over host arrays (SEP 24 6.5).

    The workspace's state is the register ``A`` and is updated in place by the
    trial; ``B`` and ``C`` are this engine's host arrays, allocated at the first step.
    Each pass makes one source evaluation through the workspace's ``lab_source`` (the
    complete source, the noise draw advanced to the pass's position and the in-stage
    absorbers included) and takes ``P`` and ``Q`` from its ``propagator`` or the
    table, allocating transients as the reference does. The step's three positions
    are known before its first pass, so their host scalars are read together
    (``stage_bundles``) and the evaluations take the kept bundles.

    **The commit contract** (SEP 52 section 5.2): :meth:`trial` overwrites the state,
    and from its start until :meth:`accept` returns the engine is :attr:`in_flight`.
    An exception or a non-finite result in that interval leaves no committed state at
    that position; only :meth:`reset`, at a load, clears the flag.

    **The flows.** ``"tabulate"`` on a constant operator holds one volume,
    ``exp(h_t D / 2)``, built at the first step after a load and reused while the
    steps keep the same key ``h_t``, their run's mean step (:func:`table_steps`, SEP 24
    Q10): a uniform schedule's steps are differences of rounded positions and differ in
    their last bits. The table is built from the key alone, from the operator's origin
    to ``h_t / 2`` -- a constant operator's increment is the distance itself -- so its
    exponent carries one rounding of ``D h_t / 2`` and none of the step's position
    (``fl(z + h/2) - z`` would be off by up to half an ulp of ``z``, an error that
    accumulates). Each reused flow's exponent is then off by ``|D| |h - h_t| / 2``, a
    few ulps of the position times ``|D|``, which does not accumulate over the run. A
    varying operator's
    ``P`` and ``Q`` change every step, and ``"tabulate"`` forms the pair once per step;
    ``"recompute"`` forms each flow in the finish that applies it, from the positions,
    whose differences telescope. The treatments agree to rounding, not bit for bit.

    **The stiffness indicator** (SEP 24 section 6.3, Q1): ``h rho = 2 ||F3 - F2|| /
    ||F2 - g1||`` over the norm mask estimates ``|h lambda|`` of the linearised source
    on the content the step integrates (for ``u' = lambda u`` it is exactly
    ``|h lambda|``), the quantity the stability interval of section 4.6 bounds. It is
    kept in ``stiffness`` and never enters the norm column. It reads the content's
    margin, not the worst case: measured on an ionising capillary it passed 2.785 on
    the step before the replay diverged, and beside an in-stage collar it read 0.17 to
    0.65 of ``h alpha_max / 2`` as the collar held less or more of the field.

    Parameters
    ----------
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The reference's workspace, whose state, evaluations and finish are used. The
        native session keeps its registers resident and never takes this engine.
    method : solprop.stepper.low_storage.LowStorageRungeKutta
        The march.
    schedule : solprop.stepper.schedule.FixedSchedule or None
        The replayed schedule, whose runs key the flow table; ``None`` only for an
        engine that never tabulates.
    propagator_mode : {'tabulate', 'recompute'}
        Whether the step's flows are held for it.

    Attributes
    ----------
    stiffness : float
        ``h rho`` of the last completed trial, dimensionless; NaN before one, or when
        ``F2 - g1`` vanished on the norm mask.

    Notes
    -----
    What the engine holds between steps is the registers ``B`` and ``C`` and, on a
    constant operator tabulating, the table: :meth:`held` names them for the backend's
    memory ledger, and :meth:`close` releases them.
    """

    def __init__(
        self,
        workspace: "NumpyWorkspace",
        method: LowStorageRungeKutta,
        schedule: FixedSchedule | None,
        propagator_mode: str,
    ) -> None:
        """Hold the workspace with no registers yet."""
        self._workspace = workspace
        self._method = method
        self._schedule = schedule
        self._propagator_mode = propagator_mode
        self._accumulator: _State | None = None
        self._stage: _State | None = None
        self._table: _State | None = None
        self._table_step = math.nan
        self._keys: tuple[np.ndarray, np.ndarray] | None = None
        self._in_flight = False
        self.stiffness = math.nan

    @property
    def in_flight(self) -> bool:  # numpydoc ignore=RT01
        """Whether a trial overwrote the state and its accept has not returned."""
        return self._in_flight

    def reset(self) -> None:
        """Drop the flow table and a step left in flight: a state was loaded.

        A resume builds the table again from its own first step.
        """
        self._table = None
        self._table_step = math.nan
        self.stiffness = math.nan
        self._in_flight = False

    def close(self) -> None:
        """Release the registers and the flow table."""
        self.reset()
        self._accumulator = None
        self._stage = None

    def held(self) -> dict[str, int]:
        """Return the bytes of every array the engine holds between steps, by name.

        Returns
        -------
        dict
            ``register_accumulator``, ``register_stage`` and ``propagators`` (the
            table), each zero while it is not allocated.
        """
        return {
            "register_accumulator": _bytes(self._accumulator),
            "register_stage": _bytes(self._stage),
            "propagators": _bytes(self._table),
        }

    def trial(self, z: float, dz: float) -> float:
        """Take one RK4IP step in place, in the four passes of the register form.

        The engine is in flight from here until :meth:`accept` returns.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            0.0, "not estimated"; or, when the field has diverged, the accepted
            state's non-finite squared norm over the norm mask.
        """
        workspace = self._workspace
        state = workspace.committed()
        self._in_flight = True
        squared = self._step(state, z, dz)
        return squared if not math.isfinite(squared) else 0.0

    def accept(self, dz: float) -> AcceptRecord:
        """Finish the state the trial left and commit it; the step leaves flight.

        Parameters
        ----------
        dz : float
            The step, m, which the split collar's rate needs.

        Returns
        -------
        solprop.backends.engines.AcceptRecord
            The stiffness indicator and the finish's split collar's sums.
        """
        workspace = self._workspace
        finished, record = workspace.finish(workspace.committed(), dz)
        workspace.install(finished)
        self._in_flight = False
        return AcceptRecord(
            stiffness=self.stiffness, removed=record.removed, found=record.found
        )

    def extend(self, z: float, dz: float) -> None:
        """Refuse: the low-storage march replays and observes no plane densely.

        Parameters
        ----------
        z : float
            Where the step started, m.
        dz : float
            The step, m.

        Raises
        ------
        BackendError
            Always.
        """
        del z, dz
        raise BackendError("the low-storage engine holds no dense extension")

    def reading(self) -> BoundaryReading:
        """Return the committed state's boundary reading.

        Returns
        -------
        solprop.backends.engines.BoundaryReading
            The workspace's reading.
        """
        return self._workspace.reading()

    def _step(self, state: _State, z: float, dz: float) -> float:
        """Take the step's four passes on `state`; return its squared norm."""
        workspace = self._workspace
        midpoint, end = z + 0.5 * dz, z + dz
        first, second = self._flows(z, dz)
        mask = workspace.problem.grids.norm_mask

        # Pass 1: F1 at z_n; A <- P A, g <- P S, B <- A + h/6 g, C <- A + h/2 g.
        source = workspace.lab_source(state, z)
        flow = workspace.propagator(z, midpoint) if first is None else first
        if state.flags.writeable:
            np.multiply(state, flow, out=state)
        else:
            # An observation holds the accepted state read-only; it keeps its own.
            state = flow * state
            workspace.install(state)
        slope = flow * source
        accumulator, stage = self._registers(state)
        np.add(state, (dz / 6.0) * slope, out=accumulator)
        np.add(state, (0.5 * dz) * slope, out=stage)

        # Pass 2: F2 at z_m; B <- B + h/3 S, C <- A + h/2 S.
        source = workspace.lab_source(stage, midpoint)
        lower = _masked_norm(source - slope, mask)
        accumulator += (dz / 3.0) * source
        np.add(state, (0.5 * dz) * source, out=stage)
        previous = source

        # Pass 3: F3 at z_m; B <- B + h/3 S, C <- Q (A + h S).
        source = workspace.lab_source(stage, midpoint)
        upper = _masked_norm(source - previous, mask)
        accumulator += (dz / 3.0) * source
        flow = workspace.propagator(midpoint, end) if second is None else second
        np.multiply(flow, state + dz * source, out=stage)

        # Pass 4: F4 at z_n + h; A <- Q B + h/6 S.
        source = workspace.lab_source(stage, end)
        flow = workspace.propagator(midpoint, end) if second is None else second
        np.multiply(flow, accumulator, out=state)
        state += (dz / 6.0) * source
        self.stiffness = 2.0 * upper / lower if lower > 0.0 else math.nan
        accepted = state[..., mask]
        return float(np.vdot(accepted, accepted).real)

    def _flows(self, z: float, dz: float) -> tuple[_State | None, _State | None]:
        """Read the step's bundles and return the flows held for it, ``P`` and ``Q``.

        - ``z``: where the step starts, m.
        - ``dz``: the step, m, positive.
        - Returns: ``P`` and ``Q`` when they are held for the step -- the table on a
          constant operator, the pair on a varying one -- or ``None`` each when
          recomputed in the finishes that apply them.
        """
        workspace = self._workspace
        midpoint, end = z + 0.5 * dz, z + dz
        tabulating = self._propagator_mode == "tabulate"
        table = tabulating and workspace.problem.flags.constant_linop
        workspace.hoist(z)
        # Each evaluation reads its position as its own anchor and each flow its
        # interval, so the anchors' bundles are read here, once each.
        workspace.stage_bundles(z, [z] if table else [z, midpoint])
        workspace.stage_bundles(midpoint, [midpoint] if table else [midpoint, end])
        workspace.stage_bundles(end, [end])
        if table:
            if self._keys is None:
                schedule = self._schedule
                assert schedule is not None  # the translation refused it without one
                self._keys = (table_steps(schedule), schedule.edges)
            key = schedule_key(*self._keys, z, dz)
            if self._table is None or key != self._table_step:
                # From the operator's origin, where a constant operator's increment is
                # the half step itself, not fl(z + h/2) - z (class docstring).
                origin = workspace.problem.linear.z_span[0]
                half = origin + 0.5 * key
                workspace.stage_bundles(origin, [half])
                self._table = workspace.propagator(origin, half)
                self._table_step = key
            return self._table, self._table
        if tabulating:
            return (
                workspace.propagator(z, midpoint),
                workspace.propagator(midpoint, end),
            )
        return None, None

    def _registers(self, state: _State) -> tuple[_State, _State]:
        """Return ``B`` and ``C``, allocated like the state at the first step.

        - ``state``: the state, whose shape and type the registers take.
        - Returns: the accumulator and the stage register.
        """
        if self._accumulator is None or self._stage is None:
            self._accumulator = np.empty_like(state)
            self._stage = np.empty_like(state)
        return self._accumulator, self._stage


def _bytes(values: np.ndarray | None) -> int:
    """Return an optional array's bytes, zero for ``None``."""
    return 0 if values is None else int(values.nbytes)


def _masked_norm(values: _State, mask: np.ndarray) -> float:
    """Return the 2-norm of an array over the norm mask's bins.

    - ``values``: an array of the state's layout.
    - ``mask``: the norm mask, boolean of shape ``(n_omega,)``.
    """
    return float(np.linalg.norm(values[..., mask]))
