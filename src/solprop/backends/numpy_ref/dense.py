# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference's dense output: a plane interpolated inside the step that crossed it.

The counterparts of native's ``engines::RungeKuttaDense`` and ``engines::AdamsDense``
(``native/src/dense.cpp``), which implement the session's ``Dense``: the session
(:class:`~solprop.backends.host.session.HostSession`) decides when a plane is observed
inside the last accepted step, and asks one of these to hold the committed state aside,
make the state the interpolated field there, and restore the committed state before
the march goes on (SEP 14 phase 3, ADR-036, SEP 52 section 5.2). An observation
neither rewinds the controller nor advances the history, and evaluates no source.

- :class:`RungeKuttaDense` evaluates DOPRI5's continuous extension of Hairer, Norsett
  and Wanner II.6, which the engine held when the step was accepted
  (:meth:`~solprop.backends.numpy_ref.engines.NumpyRungeKutta.extend`), and carries
  it to the plane by the exact linear flow.
- :class:`AdamsDense` evaluates the Adams march's interpolant over its last accepted
  step: the weights at the plane's fraction from the march's record of the step
  (:class:`~solprop.backends.host.adams.AcceptedStep`), its carrier rotations applied
  to the samples the engine kept (SEP 35 section 4.1).
"""

import numpy as np
from jaxtyping import Complex

from solprop.backends.host.adams import AcceptedStep, HostAdamsMarch
from solprop.backends.numpy_ref.engines import NumpyRungeKutta
from solprop.backends.numpy_ref.multistep import (
    ExponentialKernels,
    HostHistory,
    _combine,
    _DenseStep,
    _ExponentialStep,
)
from solprop.backends.numpy_ref.workspace import NumpyWorkspace
from solprop.exceptions import BackendError
from solprop.stepper.multistep import integrated_weights, lagrange_coefficients
from solprop.stepper.schedule import reached
from solprop.stepper.tableau import extension_value

__all__ = ["AdamsDense", "RungeKuttaDense"]

_State = Complex[np.ndarray, "n_modes n_pol n_omega"]


class _Held:
    """The committed state a dense observation set aside, restored afterwards."""

    def __init__(self, workspace: NumpyWorkspace) -> None:
        self._workspace = workspace
        self._held: _State | None = None

    def hold(self, observed: _State) -> None:
        """Set the committed state aside and make `observed` the state."""
        workspace = self._workspace
        self._held = workspace.committed()
        workspace.install(observed)

    def restore(self) -> None:
        """Make the committed state current again."""
        held, self._held = self._held, None
        if held is None:
            raise BackendError("no dense observation holds the state")
        self._workspace.install(held)


class RungeKuttaDense:
    """DOPRI5's continuous extension of the last step that crossed a plane.

    Parameters
    ----------
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The workspace whose state is observed.
    engine : solprop.backends.numpy_ref.engines.NumpyRungeKutta
        The engine that held the extension.
    """

    def __init__(self, workspace: NumpyWorkspace, engine: NumpyRungeKutta) -> None:
        """Hold the workspace and the engine; nothing is observed yet."""
        self._workspace = workspace
        self._engine = engine
        self._held = _Held(workspace)

    @staticmethod
    def crosses(position: float, target: float) -> bool:
        """Return whether a march that returned at `position` stepped across `target`.

        Parameters
        ----------
        position : float
            The committed position, m.
        target : float
            The plane, m.

        Returns
        -------
        bool
            True past the plane's landing tolerance (``not reached(target, z)``).
        """
        return not reached(target, position)

    def observe(self, position: float) -> None:
        """Hold the committed state aside; make the state the field at `position`.

        Interpolated by the continuous extension and carried to `position` by the
        exact linear flow; no source is evaluated.

        Parameters
        ----------
        position : float
            The plane, m, inside the extended step.

        Raises
        ------
        BackendError
            If no extension is held.
        """
        extension = self._engine.extension
        if extension is None:
            raise BackendError("no accepted step is held for a dense observation")
        workspace = self._workspace
        z, dz, coefficients = extension
        start = coefficients[0]
        value = extension_value(coefficients, (position - z) / dz)
        # A bin the linear flow attenuates or amplifies by more than an e-fold over the
        # step -- the spectral collar's, outside the controlled band -- has an
        # interaction-picture derivative growing exponentially across it, which no
        # polynomial in theta follows: interpolated there it returns orders of
        # magnitude more than the field. Its source is negligible beside its own loss,
        # so the anchor's linear flow is its honest value.
        stiff = (
            np.abs(np.real(workspace.delta_phi(workspace.stage_scalars(z, z + dz))))
            > 1.0
        )
        value = np.where(stiff, start, value)
        self._held.hold(workspace.mask(workspace.propagator(z, position) * value))

    def restore(self) -> None:
        """Make the committed state current again."""
        self._held.restore()


class AdamsDense:
    """The Adams march's interpolant over its last accepted step.

    Parameters
    ----------
    workspace : solprop.backends.numpy_ref.workspace.NumpyWorkspace
        The workspace whose state is observed.
    engine : solprop.backends.numpy_ref.multistep.HostHistory
        The engine that kept the step's samples.
    march : solprop.backends.host.adams.HostAdamsMarch
        The march whose record of the step (:attr:`~HostAdamsMarch.last_step`) gives
        the weights' nodes and the carrier rotations.
    """

    def __init__(
        self, workspace: NumpyWorkspace, engine: HostHistory, march: HostAdamsMarch
    ) -> None:
        """Hold the workspace, the engine and the march; nothing is observed yet."""
        self._workspace = workspace
        self._engine = engine
        self._march = march
        self._exponential = engine.exponential
        self._held = _Held(workspace)

    @staticmethod
    def crosses(position: float, target: float) -> bool:
        """Return whether a march that returned at `position` stepped past `target`.

        Parameters
        ----------
        position : float
            The committed position, m.
        target : float
            The plane, m.

        Returns
        -------
        bool
            True when `position` is past it.
        """
        return position > target

    def observe(self, position: float) -> None:
        """Hold the committed state aside; make the state the field at `position`.

        Parameters
        ----------
        position : float
            The plane, m, inside the last accepted step.

        Raises
        ------
        BackendError
            If no accepted step is held.
        """
        step = self._march.last_step
        if step is None:
            raise BackendError("no accepted Adams step is held for a dense observation")
        if self._exponential:
            held = self._engine.exponential_dense
            assert held is not None
            observed = self._observe_exponential(held, step, position)
        else:
            held = self._engine.dense
            assert isinstance(held, _DenseStep)
            observed = self._observe_fused(held, step, position)
        self._held.hold(observed)

    def restore(self) -> None:
        """Make the committed state current again."""
        self._held.restore()

    def _observe_fused(
        self, held: _DenseStep, step: AcceptedStep, position: float
    ) -> _State:
        """Return the interaction formulation's field at `position`."""
        workspace = self._workspace
        theta = (position - held.z) / held.dz
        weights = integrated_weights(
            np.array([1.0, *step.nodes]),
            theta,
            check=False,
            helpers=workspace.helpers,
        )
        value = held.start + _combine(weights, held.slopes, held.dz)
        # A bin the linear flow changes by more than an e-fold over the step (the
        # spectral collar's) takes the anchor's linear flow: no polynomial follows its
        # exponentially growing derivative (as for the Runge-Kutta extension).
        exponent = workspace.delta_phi(
            workspace.stage_scalars(held.z, held.z + held.dz)
        )
        value = np.where(np.abs(np.real(exponent)) > 1.0, held.start, value)
        return workspace.mask(workspace.propagator(held.z, position, hold=True) * value)

    def _observe_exponential(
        self, held: _ExponentialStep, step: AcceptedStep, position: float
    ) -> _State:
        """Return the exponential formulation's field at `position`.

        The accepted corrector's interpolant integrated exactly against the flow to the
        fraction ``xi`` of the step (SEP 16 section 4.3):
        ``exp(xi x) u_n + h sum_k k! xi**(k+1) phi_{k+1}(xi x) sum_j c_jk F_j``; under
        a carrier frame the samples rotated by the step's rotations, ``G_j``, the
        moments at ``xi (x - sigma)`` and the factor ``exp(-sigma (1 - xi))``, with the
        accepted step's ``sigma`` (SEP 35 section 4.1). Every factor is a forward flow,
        so a strongly damped bin needs no guard; no source is evaluated and nothing
        enters the history.
        """
        # One evaluation per plane: the reference passes serve every backend.
        helpers = self._workspace.helpers
        kernels = ExponentialKernels(helpers=helpers)
        xi = (position - held.z) / held.dz
        # The accepted step's corrector, over (source, newest, ...), as its trial
        # formed it.
        coefficients = lagrange_coefficients(
            [1.0, *step.nodes], check=False, helpers=helpers
        )
        count = coefficients.shape[1]
        shift, values = _rotated(step.carrier, held.values)
        # Under a carrier frame (SEP 35 section 4.1) the accepted step's rotated
        # samples, integrated against the moments at xi (x - sigma) and carried back
        # by exp(-sigma (1 - xi)); every factor is bounded for an imaginary sigma.
        moments = kernels.moments(
            xi * (held.exponent if shift is None else held.exponent - shift), count
        )
        scaled = moments * np.array([xi ** (k + 1) for k in range(count)]).reshape(
            (count, *([1] * held.start.ndim))
        )
        increment = kernels.apply(scaled, coefficients, values, held.dz)
        if shift is not None:
            increment = np.exp(-shift * (1.0 - xi)) * increment
        value = np.exp(xi * held.exponent) * held.start + increment
        if held.compensation is not None:
            nodes, terms = held.compensation
            weights = integrated_weights(
                np.asarray(nodes), xi, check=False, helpers=helpers
            )
            # e^{(xi - theta_j) x} is a forward flow for every history sample; for
            # the endpoint sample it is an inverse one, which a bin the step damps by
            # more than an e-fold would amplify: such a bin takes the anchor's flow,
            # as the interaction formulation's interpolant does (ADR-039).
            for weight, node, term in zip(weights, nodes, terms, strict=True):
                value = (
                    value
                    + held.dz * weight * np.exp((xi - node) * held.exponent) * term
                )
            damped = np.real(held.exponent) < -1.0
            value = np.where(damped, np.exp(xi * held.exponent) * held.start, value)
        return self._workspace.mask(value)


def _rotated(
    carrier: np.ndarray, values: list[_State]
) -> tuple[np.ndarray | None, list[_State]]:
    """Return a step's carrier shift and its samples, rotated by its rotations.

    - ``carrier``: :meth:`~solprop.backends.engines.AdamsEngine.step_carrier`'s
      packing -- the shift's imaginary part, then each history row's rotation as its
      real and imaginary parts -- or empty without a carrier frame.
    - ``values``: the new sample, then the history rows the step read, newest first.
    - Returns: the shift, purely imaginary and shape ``(1, 1, 1)`` as the trial formed
      it, or ``None``; the new sample unrotated and each history row times its factor.
    """
    if carrier.size == 0:
        return None, values
    # As `CarrierReference.shift` forms it, so that it is the trial's own bits.
    shift = 1j * np.array(carrier[0]).reshape((1, 1, 1))
    factors = [
        np.array(complex(carrier[1 + 2 * row], carrier[2 + 2 * row])).reshape((1, 1, 1))
        for row in range(len(values) - 1)
    ]
    return shift, [
        values[0],
        *(factor * value for factor, value in zip(factors, values[1:], strict=True)),
    ]
