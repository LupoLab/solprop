# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The ``mlx`` backend: the single-precision oracle's step, on the Metal GPU.

:class:`MlxBackend` keeps everything the single-precision oracle
(:class:`~solprop.backends.numpy_ref.single.NumpySingleBackend`) does on the host --
admission, the narrowed tables and their exponents, the state scale, the stage
bundles, landing, statistics from the widened state, metadata -- and moves the state and
every per-step pass to the device: the stage combinations, the double-float propagators
and prefactors (:class:`~solprop.backends.mlx_gpu.arithmetic.DoubleFloatTerms`), the
transforms, the synthesis and projection products, the responses, the finish and the
error norm, reduced on the device to a few scalars per trial.

It admits the fibre route -- mode-averaged and multimode capillaries, solid-core and
GNLSE fibres on field and envelope grids, with Kerr, the envelope grid's harmonic,
scalar and tensor Raman, gas mixtures, the silica bundle, the field-grid plasma with its
held current, loss, both temporal absorber schemes, the polarisation taper and
device-ready gradients -- and radial and Cartesian free space, including a line, with
their transverse collars, stepped by Runge-Kutta. Everything else is refused by name.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx  # pyright: ignore[reportMissingImports] - macOS arm64 only
import numpy as np
from jaxtyping import Complex

from solprop.backends.memory import mlx_bounded, mlx_graph_bytes, mlx_stage_bytes
from solprop.backends.mlx_gpu import passes as device
from solprop.backends.mlx_gpu.arithmetic import (
    DoubleFloatTerms,
    fft_probe,
    gemm_probe,
    on_gpu,
    population,
    prefix_scan,
)
from solprop.backends.mlx_gpu.envelope import DeviceCycle, envelope_cycle
from solprop.backends.numpy_ref.passes import _kerr as compiled_kerr
from solprop.backends.numpy_ref.passes import _raman_thg as raman_thg
from solprop.backends.numpy_ref.single import NumpySingleBackend
from solprop.constants import EPS0
from solprop.exceptions import BackendError, ConvergenceError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.profiles import ZProfile
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.envelope_plasma import EnvelopePlasmaAssembly
from solprop.translate.linop import _separable_phase
from solprop.translate.precision import combine, narrow
from solprop.translate.problem import KernelProblem
from solprop.translate.responses import ResponseStage
from solprop.translate.spatial import separable_values
from solprop.translate.stage import StageScalars
from solprop.translate.tdse import TDSEBinding

__all__ = ["DeviceSources", "MlxBackend", "MlxLowStorage", "refuse_unsupported"]

_SOURCE_STAGES = frozenset(
    {
        ResponseStage.ENVELOPE_CYCLE,
        ResponseStage.ENVELOPE_POPULATION,
        ResponseStage.ENVELOPE_SOURCES,
        ResponseStage.KERR,
        ResponseStage.RAMAN_DRIVE,
        ResponseStage.PLASMA_CURRENT,
        ResponseStage.INTENSITY,
        ResponseStage.IONISATION_RATE,
        ResponseStage.PLASMA_SCAN,
        ResponseStage.ACCUMULATE,
    }
)


@dataclass(frozen=True)
class DeviceSources:
    """A chunk's windowed polarisation and ungated current, device arrays.

    Attributes
    ----------
    polarisation : mlx.core.array
        The windowed polarisation, divided by ``s``.
    current : mlx.core.array or None
        The ungated plasma current, divided by ``s``, or ``None`` without plasma.
    """

    polarisation: mx.array
    current: mx.array | None


def _summed(totals: list[mx.array], parts: list[mx.array]) -> list[mx.array]:
    """Return two lists of arrays summed element by element.

    Parameters
    ----------
    totals, parts : list of mlx.core.array
        Arrays of matching shapes.

    Returns
    -------
    list of mlx.core.array
        The sums.
    """
    return [a + b for a, b in zip(totals, parts, strict=True)]


def _restored(value: float, power: float, exponent: int) -> float:
    """Return a rule-6 scaled sum restored in binary64, ``value 2**(2 power + e)``.

    Parameters
    ----------
    value : float
        The sum of the scaled squares.
    power : float
        The power of two the values were divided by before squaring, an integer, or
        not finite when they were not.
    exponent : int
        Any further block exponent of the weights in the sum.

    Returns
    -------
    float
        The sum, or NaN when the values it was taken of were not finite.
    """
    if not math.isfinite(power):
        return math.nan
    return math.ldexp(value, 2 * int(power) + exponent)


def refuse_unsupported(problem: KernelProblem) -> None:
    """Refuse, by name, what the ``mlx`` backend does not run in this phase.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem.

    Raises
    ------
    BackendError
        Naming the refusal and the backend that runs it.
    """
    if not problem.precision.single:
        raise BackendError(
            "mlx computes in binary32 only, and Apple GPUs have no binary64 "
            "arithmetic: it refuses precision='double'; use precision='single', or a "
            "binary64 backend (native, numba_cpu, numpy_ref)"
        )
    stepper = problem.stepper
    refusals = [
        (
            not problem.device_ready,
            "an operator or prefactor that took the z table: it is not device-ready, "
            f"its table holding {problem.linear.nbytes / 2**20:.1f} MiB; use "
            "backend='numpy_ref'",
        ),
        (
            isinstance(stepper.integrator, Multistep),
            "the Adams integrator, which single precision does not admit",
        ),
        (
            stepper.interpolate_planes,
            "interpolated save planes (not yet on mlx); land on the planes instead",
        ),
    ]
    for refuse, what in refusals:
        if refuse:
            msg = f"mlx does not run {what}"
            raise BackendError(msg)


class MlxBackend(NumpySingleBackend):
    """Propagate a single-precision problem on an Apple-silicon GPU.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A validated single-precision fibre, radial or Cartesian problem.
    grid : solprop.grid.TimeGrid, optional
        The grid, needed only by ``stats``.
    modes : collections.abc.Sequence, optional
        The propagated modes, needed only by the ``zdw`` statistic.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Must be ``None``.

    Raises
    ------
    BackendError
        Naming what is refused, or a protected operation the device does not establish.
    """

    name = "mlx"

    _monitor_weights: mx.array | None = None
    """The boundary monitor's four weight rows on the device, each narrowed with its
    own exponent (:attr:`_monitor_exponents`), while a monitor is bound (SEP 21)."""

    _monitor_exponents: tuple[int, ...] = (0, 0, 0, 0)
    """The weight rows' block exponents (SEP 33 section 4.3, rule 1)."""

    _monitor_columns: mx.array | None = None
    """A radial run's column quadrature on the device, narrowed, shaped to broadcast
    against the state; ``None`` elsewhere."""

    _column_exponent: int = 0
    """The column quadrature's block exponent."""

    _pending_removal: tuple[mx.array, ...] = ()
    """What the split took at accepted steps not yet read: device scalars, each step's
    rule-6 exponent and its scaled sum (:meth:`_device_removal`), folded into
    :attr:`_split_removed` at the next synchronisation (:meth:`_synchronised`)."""

    def __init__(
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        tdse_binding: TDSEBinding | None = None,
    ) -> None:
        """Adopt a single-precision fibre problem and upload its tables."""
        refuse_unsupported(problem)
        super().__init__(
            problem,
            grid=grid,
            modes=modes,
            profiles=profiles,
            tdse_binding=tdse_binding,
        )
        grids, transform = problem.grids, problem.transform
        tables = self._tables
        self._crop = mx.array(grids.crop_index.astype(np.int32))
        crop = grids.crop_index
        contiguous = bool(np.array_equal(crop, np.arange(crop.size)))
        self._pad_crop = None if contiguous else self._crop
        self._norm_index = mx.array(np.flatnonzero(grids.norm_mask).astype(np.int32))
        self._active = mx.array(grids.active_mask.astype(np.float32))
        self._synthesis = mx.array(tables.synthesis)
        self._projection = mx.array(tables.projection)
        self._coarse = None if self._embedded is None else mx.array(self._embedded)
        self._t_window = mx.array(tables.t_window)
        self._t_window_derivative = mx.array(tables.t_window_derivative)
        self._omega_window = mx.array(tables.omega_window)
        self._temporal_rate = (
            None if tables.temporal_rate is None else mx.array(tables.temporal_rate)
        )
        self._device_carrier = (
            None if self._carrier is None else mx.array(self._carrier)
        )
        self._held_device: tuple[mx.array, mx.array] | None = None
        if self._held_factors is not None:
            inverse = np.zeros(grids.active_mask.shape, dtype=np.complex64)
            squared = np.zeros_like(inverse)
            inverse[grids.active_mask] = self._held_factors[0]
            squared[grids.active_mask] = self._held_factors[1]
            self._held_device = (mx.array(inverse), mx.array(squared))
        self._upload_operator(problem)
        self._prefactor_terms = (
            None
            if self._terms.prefactor is None
            else DoubleFloatTerms.upload(self._terms.prefactor)
        )
        self._prefactor_constant: mx.array | None = None
        responses = problem.responses
        self._rate_tables = tuple(
            device.DeviceRateTable.upload(channel.rate)
            for channel in responses.plasma_channels
        )
        self._upload_envelope(responses.envelope_plasma)
        self._raman_upload: tuple[object, mx.array] | None = None
        self._unit = mx.ones((1,), dtype=mx.complex64)
        # Whether each stage is evaluated as it is formed, bounding the live graph to
        # one stage; a batch evaluates its members' steps together instead.
        self.evaluate_stages = mlx_bounded(
            problem.stepper.tableau.n_stages, mlx_stage_bytes(problem)
        )
        n_to = grids.n_to
        n_pol = transform.n_pol
        chunk = min(transform.node_chunk, transform.n_nodes) * n_pol
        shapes = [(chunk, transform.n_modes, problem.shape[2] * n_to)]
        shapes.append((transform.n_modes, chunk, problem.shape[2] * n_to))
        # The Cartesian route has no quadrature products; its transverse transforms
        # are complex transforms along each axis, probed beside the temporal ones.
        separable = problem.linear.separable is not None
        self.gemm_error = gemm_probe(shapes) if on_gpu() and not separable else 0.0
        kind = problem.grid_kind
        lengths = [(kind, n_to), (kind, grids.n_t)]
        if separable and transform.geometry is not None:
            lengths += [("envelope", n) for n in transform.geometry.shape if n > 1]
        if problem.responses.raman_terms:
            lengths.append(("field", 2 * n_to))
        self.fft_error = fft_probe(lengths)
        # MLX keeps freed buffers for reuse; its cache is held to one stage's graph,
        # the admission estimate's own row (SEP 33 section 6.3).
        mx.set_cache_limit(mlx_graph_bytes(problem))

    # The state on the device.

    def _upload_envelope(self, envelope: EnvelopePlasmaAssembly | None) -> None:
        """Upload the envelope plasma's cycle representations and source factors.

        Parameters
        ----------
        envelope : solprop.translate.envelope_plasma.EnvelopePlasmaAssembly or None
            The compiled envelope plasma, if any.
        """
        self._cycles = (
            ()
            if envelope is None
            else tuple(DeviceCycle.upload(c.cycle) for c in envelope.channels)
        )
        self._envelope_factors = (
            None
            if envelope is None
            else (
                mx.array(self._rounded(envelope.factors.drude)),
                mx.array(self._rounded(envelope.factors.loss)),
            )
        )
        # The refinement failures of the trial being built, read with its norm.
        self._envelope_failures: list[mx.array] = []

    def _upload_operator(self, problem: KernelProblem) -> None:
        """Upload the linear operator's terms and the free-space tables.

        Parameters
        ----------
        problem : solprop.translate.problem.KernelProblem
            The adopted problem.

        Raises
        ------
        solprop.exceptions.BackendError
            When a fibre operator has no device-ready terms.
        """
        separable = problem.linear.separable
        self._free_prefactor: mx.array | None = None
        if separable is not None:
            # A separable operator is one term linear in the distance: the operator
            # at unit distance, weighted by the bundle's dz (translate.linop).
            self._phi = DoubleFloatTerms.upload(
                _separable_phase(separable, 1.0, imaginary=False)[None]
            )
            _, prefactor = separable_values(separable)
            self._free_prefactor = mx.array(self._cast(prefactor))
        elif self._terms.phi is None:
            raise BackendError("mlx needs the operator's device-ready terms")
        else:
            self._phi = DoubleFloatTerms.upload(self._terms.phi)
        self._spatial_matrix = (
            mx.array(self._spatial_absorber) if self._spatial_absorber.size else None
        )
        geometry = problem.transform.geometry
        self._radial_plasma = (
            geometry is not None
            and geometry.kind == "radial"
            and bool(problem.responses.plasma_channels)
        )

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: object,
        *,
        z: float = 0.0,
    ) -> None:
        """Load a state onto the device at the launch's scale.

        Parameters
        ----------
        field : numpy.ndarray
            The physical state in the canonical layout.
        noise : object
            Must be ``None``.
        z : float, optional
            The position, m.
        """
        super().set_initial(field, noise, z=z)
        self._state = cast(Any, mx.array(np.asarray(self._require_state())))

    def _observed_field(
        self, state: Any
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Copy the state to the host and widen it: ``s`` times it, exactly.

        Parameters
        ----------
        state : mlx.core.array or numpy.ndarray
            The scaled binary32 state.

        Returns
        -------
        numpy.ndarray
            An owned complex128 array.
        """
        return np.array(state).astype(np.complex128) * self.scale

    def _apply_mask(self, state):  # noqa: ANN001, ANN202 - a device or host state
        """Impose the hard band mask.

        Parameters
        ----------
        state : mlx.core.array or numpy.ndarray
            The state.

        Returns
        -------
        mlx.core.array or numpy.ndarray
            The masked state.
        """
        if isinstance(state, mx.array):
            return state * self._active
        return super()._apply_mask(state)

    def _finish(self, state: Any, dz: float) -> Any:
        """Mask the accepted state, apply a split-step collar, and check it is finite.

        While a boundary monitor of the ``"split_step"`` scheme is attached, what the
        split takes is reduced on the device from the time samples it forms anyway,
        and read with the next synchronisation (:meth:`_synchronised`).

        Parameters
        ----------
        state : mlx.core.array
            The accepted state.
        dz : float
            The step's length, m.

        Returns
        -------
        mlx.core.array
            The finished state.

        Raises
        ------
        ConvergenceError
            If the state is not finite.
        """
        finished = state * self._active
        responses = self._problem.responses
        rate = responses.temporal_absorber
        if rate is not None and responses.absorber_scheme == "split_step":
            # a + F(F^-1(a) (f - 1)) rather than F(F^-1(a) f): the factor is exactly one
            # outside the collar, so the transforms' rounding -- MLX's GPU round trip
            # shrinks by 1-2e-7 -- touches the collar's correction alone, not the state.
            change = mx.array(np.expm1(-0.5 * rate * dz).astype(np.float32))
            samples = self._time_samples(finished)
            if self._split_monitor is not None:
                self._pending_removal += self._device_removal(samples, change)
            time = samples * change
            if self._kind == "field":
                finished = finished + mx.fft.rfft(time, axis=-1) * self._active
            else:
                finished = finished + mx.fft.fft(time, axis=-1) * self._active
        # Finiteness was read with the trial's norm (one synchronisation per step):
        # a non-finite trial state scores an infinite norm, which every march refuses.
        return finished

    # The boundary monitor's reading (SEP 21).

    def _time_samples(self, state: mx.array) -> mx.array:
        """Return a device state's samples on the coarse time axis.

        Parameters
        ----------
        state : mlx.core.array
            A scaled state, canonical layout.

        Returns
        -------
        mlx.core.array
            Its inverse transform along the last axis, real on a field grid.
        """
        if self._kind == "field":
            return mx.fft.irfft(state, n=self._problem.grids.n_t, axis=-1)
        return mx.fft.ifft(state, axis=-1)

    def _scaled_density(self, samples: mx.array) -> tuple[mx.array, mx.array]:
        """Return time samples' energy density, scaled as SEP 33's rule 6 has it.

        The samples are divided by the power of two at or below their largest
        magnitude before squaring, so the squares and their sums cannot leave
        binary32's range; the columns' quadrature, narrowed with its own exponent,
        weighs each column on a radial run.

        Parameters
        ----------
        samples : mlx.core.array
            Time samples of a scaled state, canonical layout.

        Returns
        -------
        tuple of mlx.core.array
            The power of two ``p`` (a float32 scalar holding an integer) and the
            density, ``|samples / 2**p|**2`` times the narrowed column weights; the
            host restores ``2**(2 p)`` and the columns' exponent in binary64.
        """
        peak = mx.max(mx.abs(samples))
        power = mx.where(peak > 0, mx.floor(mx.log2(peak)), 0.0)
        scaled = samples * mx.power(2.0, -power)
        if self._kind == "field":
            density = scaled * scaled
        else:
            re, im = mx.real(scaled), mx.imag(scaled)
            density = re * re + im * im
        if self._monitor_columns is not None:
            density = density * self._monitor_columns
        return power, density

    def _device_removal(
        self, samples: mx.array, change: mx.array
    ) -> tuple[mx.array, ...]:
        """Return what one split takes of the samples it damps, lazily, scaled.

        Parameters
        ----------
        samples : mlx.core.array
            The masked accepted state's time samples, before the split.
        change : mlx.core.array
            ``exp(-alpha(t) h / 2) - 1`` on the coarse time axis, binary32.

        Returns
        -------
        tuple of mlx.core.array
            The scaling's power of two and ``sum_t (1 - exp(-alpha h)) P(t)`` of the
            scaled density (:meth:`_scaled_density`).
        """
        # 1 - exp(-alpha h) = -c (2 + c) with c = exp(-alpha h / 2) - 1, which the
        # split already holds: a product of two accurate factors, no cancellation.
        taken = -change * (2.0 + change)
        power, density = self._scaled_density(samples)
        return power, mx.sum(density * taken)

    def _synchronised(self, values: list[mx.array]) -> np.ndarray:
        """Read device values and the split's pending removals in one synchronisation.

        The removals are folded into :attr:`_split_removed` in binary64, physical
        units. Every read a march makes goes through here, so the lazy graph of a
        removal -- which holds its step's time samples -- never outlives the next
        synchronisation.

        Parameters
        ----------
        values : list of mlx.core.array
            The arrays the caller reads, of any shapes.

        Returns
        -------
        numpy.ndarray
            ``values`` flattened and concatenated, binary64.
        """
        pending, self._pending_removal = self._pending_removal, ()
        flat = [value.reshape(-1) for value in (*values, *pending)]
        read = np.array(mx.concatenate(flat), dtype=np.float64)
        count = sum(int(value.size) for value in values)
        for power, taken in read[count:].reshape(-1, 2):
            # The state is a / s, so its power is s**2 times the scaled one's.
            restored = _restored(float(taken), float(power), self._column_exponent)
            self._split_removed += restored * self.scale**2
        return read[:count]

    def _bind_monitor(self, monitor: BoundaryMonitor | None) -> None:
        """Upload a boundary monitor's weights, or release them.

        Each weight row and the columns' quadrature are narrowed with their own block
        exponents (SEP 33 section 4.3, rule 1), once per run.

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor or None
            The run's monitor, or ``None`` to stop.
        """
        super()._bind_monitor(monitor)
        self._pending_removal = ()
        if monitor is None:
            self._monitor_weights = self._monitor_columns = None
            return
        rows = [narrow(np.asarray(row, dtype=np.float64)) for row in monitor.weights]
        self._monitor_weights = mx.array(np.stack([row.values for row in rows]))
        self._monitor_exponents = tuple(row.exponent for row in rows)
        if monitor.columns is None:
            self._monitor_columns, self._column_exponent = None, 0
        else:
            columns = narrow(np.asarray(monitor.columns, dtype=np.float64))
            self._monitor_columns = mx.array(columns.values[None, :, None, None])
            self._column_exponent = columns.exponent

    def _boundary_energies(self, state: Any) -> BoundaryEnergies:
        """Return the boundary monitor's reading, reduced on the device.

        ``P(t)`` and its four weighted sums are formed on the device; five numbers
        and the scaling's exponent cross to the host, with any pending split
        removal, in one synchronisation.

        Parameters
        ----------
        state : mlx.core.array
            The scaled binary32 state.

        Returns
        -------
        solprop.monitor.BoundaryEnergies
            The reading, binary64, physical units.

        Raises
        ------
        BackendError
            If no monitor is attached.
        """
        weights = self._monitor_weights
        if self._monitor is None or weights is None:
            raise BackendError(f"{self.name}: no boundary monitor is attached")
        n_t = self._problem.grids.n_t
        power, density = self._scaled_density(self._time_samples(state))
        density_t = mx.sum(density.reshape(-1, n_t), axis=0)
        # Elementwise products and sums, not a matrix product, which MLX may round
        # to TF32 on the GPU.
        regions = mx.sum(weights * density_t[None, :], axis=1)
        values = self._synchronised([power, mx.sum(density_t), regions])
        # s**2 for the state's scale, 2**(2 p) for rule 6's, and the narrowed
        # weights' own exponents, restored in binary64.
        power, scale, own = float(values[0]), self.scale**2, self._column_exponent
        total = _restored(float(values[1]), power, own) * scale
        regions_host = np.array(
            [
                _restored(float(value), power, own + row) * scale
                for value, row in zip(values[2:], self._monitor_exponents, strict=True)
            ]
        )
        removed, self._split_removed = self._split_removed, 0.0
        return BoundaryEnergies(total, regions_host, removed)

    # The trial step.

    def _trial_graph(self, z: float, dz: float) -> tuple[mx.array, mx.array, mx.array]:
        """Build one interaction-picture DOPRI5 step on the device, unevaluated.

        The oracle's step (:meth:`NumpyRefBackend._trial_step`) with device arrays:
        every stage is evaluated, no stage is reused across steps.

        Parameters
        ----------
        z : float
            Position at the start of the step, m.
        dz : float
            Length of the trial step, m.

        Returns
        -------
        tuple of mlx.core.array
            The state at the anchor, the embedded difference carried to the end of the
            step and the trial state, as lazy device arrays: MLX evaluates them when
            the norm is read, so a batch builds every member's step first.
        """
        state = cast(mx.array, self._require_state())
        self._hoist(z)
        self._envelope_failures = []
        tableau = self._problem.stepper.tableau
        self._stage_bundles(
            z, [z + dz, *(z + float(node) * dz for node in tableau.nodes)]
        )
        step = self._stage_scalars(z, z + dz)
        stages: list[mx.array] = []
        solution = mx.zeros_like(state)
        error = mx.zeros_like(state)
        for index, node in enumerate(tableau.nodes):
            combination = mx.zeros_like(state)
            for previous in range(index):
                weight = float(tableau.matrix[index, previous])
                if weight != 0.0:
                    combination = combination + weight * stages[previous]
            bundle = self._stage_scalars(z, z + float(node) * dz)
            forward, back = self._propagators(bundle)
            lab = forward * (state + dz * combination)
            stage = self._device_nonlinear(lab, bundle, back)
            # Evaluated at each stage boundary: the graph of one stage -- its node
            # chunks and the scans' levels -- is then the most that is ever live
            # (SEP 33 section 6.3), where a whole step's graph would hold seven.
            if self.evaluate_stages:
                # The previous split's pending removal with the first stage, which
                # evaluates the state it was taken from: its time samples go then.
                mx.eval(stage, *self._pending_removal)
            stages.append(stage)
            solution = solution + float(tableau.solution_weights[index]) * stage
            error = error + float(tableau.error_weights[index]) * stage
        propagator, _ = self._propagators(step)
        advanced = propagator * (state + dz * solution)
        self._last_stages = stages
        return state, propagator * (dz * error), advanced

    def _trial_step(self, z: float, dz: float) -> tuple[float, mx.array]:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Take one interaction-picture DOPRI5 step on the device and score it.

        Parameters
        ----------
        z : float
            Position at the start of the step, m.
        dz : float
            Length of the trial step, m.

        Returns
        -------
        tuple
            The scaled error norm and the device state at the end of the step.
        """
        state, error, advanced = self._trial_graph(z, dz)
        return self._device_norm(error, state, advanced), advanced

    def _propagators(self, bundle: StageScalars) -> tuple[mx.array, mx.array]:
        """Return ``exp(dPhi)`` and ``exp(-dPhi)`` of a stage, double-float accurate.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        tuple of mlx.core.array
            The forward and inverse propagators, complex64.
        """
        assert bundle.phi_delta is not None  # device-ready, refused otherwise
        return (
            self._phi.exponential(bundle.phi_delta, 1),
            self._phi.exponential(bundle.phi_delta, -1),
        )

    def _device_prefactor(self, bundle: StageScalars) -> mx.array:
        """Return the stage's prefactor on the device.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        mlx.core.array
            The complex64 prefactor: the terms combined in double-float, or a constant
            prefactor evaluated once on the host and rounded.
        """
        if self._prefactor_terms is not None and bundle.prefactor is not None:
            return self._prefactor_terms.combine(bundle.prefactor)
        if self._prefactor_constant is None:
            self._prefactor_constant = mx.array(np.asarray(self._prefactor(bundle)))
        return self._prefactor_constant

    def _device_raman(self, bundle: StageScalars) -> mx.array | None:
        """Return the narrowed Raman kernel spectrum on the device.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        mlx.core.array or None
            The spectrum scaled by its block exponent, or ``None`` without Raman.
        """
        spectrum = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        if spectrum is None:
            return None
        if self._raman_upload is None or self._raman_upload[0] is not spectrum:
            self._raman_upload = (spectrum, mx.array(spectrum))
        return self._raman_upload[1]

    def _device_nonlinear(  # noqa: PLR0915 - the oracle's modal stage, pass by pass
        self, lab: mx.array, bundle: StageScalars, back: mx.array
    ) -> mx.array:
        """Evaluate one stage derivative on the device.

        Parameters
        ----------
        lab : mlx.core.array
            The lab-frame stage input.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        back : mlx.core.array
            ``exp(-dPhi)`` of the stage.

        Returns
        -------
        mlx.core.array
            The stage derivative in the interaction variable.
        """
        problem = self._problem
        grids, responses, transform = (
            problem.grids,
            problem.responses,
            problem.transform,
        )
        kind = self._kind
        if self._free_prefactor is not None:
            return self._device_free(lab, bundle, back)
        modal_t = device.to_time(lab[0], self._pad_crop, grids.n_to, kind=kind)
        absorber = self._device_absorber(modal_t)
        if self._radial_plasma:
            spectrum = self._device_radial_plasma(modal_t, bundle)
            return back * self._normalised(spectrum, bundle, absorber)[None]
        if len(responses.response_order) == 1:
            if absorber is None:
                return mx.zeros_like(lab)
            return back * absorber[None]
        raman = self._device_raman(bundle)
        taper = bundle.taper_scale
        n_pol = transform.n_pol
        # Per stream (the polarisation, the plasma's gated and edge currents, the
        # envelope plasma's drive and loss), the projection on the fine quadrature
        # and, for the alarm, on the embedded one: one matrix product each per chunk.
        fine: list[mx.array] | None = None
        embedded: list[mx.array] | None = None
        plasma = envelope = False
        for first in range(0, transform.n_nodes, transform.node_chunk):
            rows = slice(first * n_pol, (first + transform.node_chunk) * n_pol)
            field = device.synthesise(modal_t, self._synthesis[rows], taper, n_pol)
            sources = self._device_sources(field, bundle, raman)
            streams = [sources.polarisation]
            if sources.current is not None:
                plasma = True
                streams += [
                    self._t_window * sources.current,
                    self._t_window_derivative * sources.current,
                ]
            if self._cycles:
                envelope = True
                streams += list(self._device_envelope(field, bundle))
            parts = device.project_many(
                streams, self._projection[:, rows], taper, n_pol
            )
            fine = parts if fine is None else _summed(fine, parts)
            if self._coarse is not None:
                parts = device.project_many(
                    streams, self._coarse[:, rows], taper, n_pol
                )
                embedded = parts if embedded is None else _summed(embedded, parts)
        assert fine is not None
        projected, *rest = fine
        coarse, *coarse_rest = [None] * len(fine) if embedded is None else embedded
        gated = edge = None
        if plasma:
            gated, edge = rest[:2]
            self._last_plasma = (gated, edge, *coarse_rest[:2])  # pyright: ignore[reportAttributeAccessIssue]
            rest, coarse_rest = rest[2:], coarse_rest[2:]
        else:
            self._last_plasma = None
        drives = rest if envelope else None
        self._last_envelope = (  # pyright: ignore[reportAttributeAccessIssue]
            (*rest, *coarse_rest) if envelope else None
        )
        self._last_projected = projected  # pyright: ignore[reportAttributeAccessIssue]
        self._last_coarse = coarse  # pyright: ignore[reportAttributeAccessIssue]
        self._last_bundle = bundle
        spectrum = device.to_frequency(projected, self._crop, kind=kind)
        if gated is not None and edge is not None:
            spectrum = spectrum + self._device_held(gated, edge)
        if drives is not None:
            spectrum = spectrum + self._envelope_spectrum(drives)
        return back * self._normalised(spectrum, bundle, absorber)[None]

    def _normalised(
        self, spectrum: mx.array, bundle: StageScalars, absorber: mx.array | None
    ) -> mx.array:
        """Apply the prefactor and window to a source spectrum and add the collar.

        Parameters
        ----------
        spectrum : mlx.core.array
            The coarse source spectrum.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        absorber : mlx.core.array or None
            The in-stage collar's term.

        Returns
        -------
        mlx.core.array
            The stage derivative before the inverse propagator.
        """
        normalised = self._device_prefactor(bundle) * self._omega_window * spectrum
        return normalised if absorber is None else normalised + absorber

    def _device_held(self, gated: mx.array, edge: mx.array) -> mx.array:
        """Return the held plasma current's spectrum on the device.

        Parameters
        ----------
        gated : mlx.core.array
            The windowed current, projected.
        edge : mlx.core.array
            The current times the window's derivative, projected.

        Returns
        -------
        mlx.core.array
            Complex64 on the coarse grid, zero outside the active band.
        """
        assert self._held_device is not None
        inverse, squared = self._held_device
        current = mx.take(mx.fft.rfft(gated, axis=-1), self._crop, axis=-1)
        derivative = mx.take(mx.fft.rfft(edge, axis=-1), self._crop, axis=-1)
        return current * inverse + derivative * squared

    def _device_absorber(self, modal_t: mx.array) -> mx.array | None:
        """Return the in-stage temporal collar's term, or ``None``.

        Parameters
        ----------
        modal_t : mlx.core.array
            The modal field on the fine time grid.

        Returns
        -------
        mlx.core.array or None
            ``-alpha E / 2`` on the coarse grid, masked to the active band.
        """
        if not self._in_stage_absorber:
            return None
        term = mx.zeros_like(modal_t)
        if (
            self._temporal_rate is not None
            and self._problem.responses.absorber_scheme == "nonlinear"
        ):
            term = -0.5 * self._temporal_rate * modal_t
        if self._spatial_matrix is not None:
            n_modes = modal_t.shape[0]
            flat = modal_t.reshape(n_modes, -1)
            product = (self._spatial_matrix @ mx.real(flat)).astype(modal_t.dtype)
            if modal_t.dtype == mx.complex64:
                product = product + 1j * (self._spatial_matrix @ mx.imag(flat))
            term = term + product.reshape(modal_t.shape)
        return device.to_frequency(term, self._crop, kind=self._kind) * self._active

    def _device_radial_plasma(
        self, modal_t: mx.array, bundle: StageScalars
    ) -> mx.array:
        """Return the radial route's plasma-bearing source spectrum.

        The oracle's ``_radial_plasma_spectrum``: each node's polarisation and held
        current are transformed on the node, then projected onto the Hankel modes.

        Parameters
        ----------
        modal_t : mlx.core.array
            The modal field on the fine time grid.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        mlx.core.array
            The coarse source spectrum before the prefactor.
        """
        transform = self._problem.transform
        raman = self._device_raman(bundle)
        n_pol = transform.n_pol
        spectrum = None
        for first in range(0, transform.n_nodes, transform.node_chunk):
            rows = slice(first * n_pol, (first + transform.node_chunk) * n_pol)
            field = device.synthesise(
                modal_t, self._synthesis[rows], bundle.taper_scale, n_pol
            )
            sources = self._device_sources(field, bundle, raman)
            nodal = device.to_frequency(
                sources.polarisation, self._crop, kind=self._kind
            )
            assert sources.current is not None
            nodal = nodal + self._device_held(
                self._t_window * sources.current,
                self._t_window_derivative * sources.current,
            )
            part = device.project(
                nodal, self._projection[:, rows], bundle.taper_scale, n_pol
            )
            spectrum = part if spectrum is None else spectrum + part
        assert spectrum is not None
        return spectrum

    def _device_free(
        self, lab: mx.array, bundle: StageScalars, back: mx.array
    ) -> mx.array:
        """Evaluate a Cartesian (or line) stage derivative on the device.

        The oracle's ``_nonlinear_free``: the transverse inverse transform, the
        responses per chunk of columns on the fine time grid, the forward transforms,
        the per-column prefactor, and the in-stage collars.

        Parameters
        ----------
        lab : mlx.core.array
            The lab-frame stage input, ``(1, n_columns, n_pol, n_omega)``.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        back : mlx.core.array
            ``exp(-dPhi)`` of the stage.

        Returns
        -------
        mlx.core.array
            The stage derivative in the interaction variable.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        assert geometry is not None and self._free_prefactor is not None
        nx, ny = geometry.shape
        shape = (ny, nx, *lab.shape[2:])
        result = mx.zeros(lab.shape[1:], dtype=mx.complex64)
        kind = self._kind
        if len(responses.response_order) > 1:
            physical = mx.fft.ifftn(lab[0].reshape(shape), axes=[0, 1]).reshape(
                lab.shape[1:]
            )
            raman = self._device_raman(bundle)
            parts = []
            for first in range(0, transform.n_nodes, transform.node_chunk):
                rows = physical[first : first + transform.node_chunk]
                field = grids.pad_scale * device.to_time(
                    rows, self._pad_crop, grids.n_to, kind=kind
                )
                sources = self._device_sources(field, bundle, raman)
                part = device.to_frequency(sources.polarisation, self._crop, kind=kind)
                if sources.current is not None:
                    part = part + self._device_held(
                        self._t_window * sources.current,
                        self._t_window_derivative * sources.current,
                    )
                if self._cycles:
                    part = part + self._envelope_spectrum(
                        list(self._device_envelope(field, bundle))
                    )
                parts.append(part)
            spectrum = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            spectrum = mx.fft.fftn(spectrum.reshape(shape), axes=[0, 1]).reshape(
                lab.shape[1:]
            )
            result = self._free_prefactor * self._omega_window * spectrum
        if self._in_stage_absorber:
            physical = mx.fft.ifftn(lab[0].reshape(shape), axes=[0, 1]).reshape(
                lab.shape[1:]
            )
            temporal = self._temporal_rate
            if responses.absorber_scheme != "nonlinear":
                temporal = None
            spatial = mx.array(self._cast(geometry.absorber_rate))
            parts = []
            for first in range(0, transform.n_nodes, transform.node_chunk):
                rows = slice(first, first + transform.node_chunk)
                field = device.to_time(
                    physical[rows], self._pad_crop, grids.n_to, kind=kind
                )
                rate = spatial[rows][:, None, None]
                if temporal is not None:
                    rate = rate + temporal
                parts.append(
                    device.to_frequency(-0.5 * rate * field, self._crop, kind=kind)
                )
            collar = mx.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
            collar = mx.fft.fftn(collar.reshape(shape), axes=[0, 1]).reshape(
                lab.shape[1:]
            )
            result = result + self._active * collar
        return back * result[None]

    def _device_sources(
        self, field: mx.array, bundle: StageScalars, raman: mx.array | None
    ) -> DeviceSources:
        """Evaluate the responses at a chunk of nodes, the oracle's ``single_sources``.

        Parameters
        ----------
        field : mlx.core.array
            The scaled field at the nodes.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        raman : mlx.core.array or None
            The narrowed Raman kernel spectrum.

        Returns
        -------
        DeviceSources
            The windowed polarisation and the ungated current.

        Raises
        ------
        BackendError
            For a response this backend does not run.
        """
        responses = self._problem.responses
        k = self._scale_exponent
        total = mx.zeros_like(field)
        current = None
        for stage in responses.response_order:
            if stage is ResponseStage.KERR:
                kerr = compiled_kerr(responses)
                factor = combine(
                    kerr.coefficient, bundle.kerr_density, exponent=2 * k
                ).binary32()
                total = total + device.kerr(field, kerr, factor, self._device_carrier)
            elif stage is ResponseStage.RAMAN_DRIVE:
                assert raman is not None
                density = (
                    1.0
                    if len(responses.raman_terms) > 1
                    else bundle.raman_densities[0]
                    if bundle.raman_densities
                    else bundle.density
                )
                factor = combine(
                    EPS0, density, exponent=2 * k + self._raman_exponent
                ).binary32()
                total = total + device.raman(
                    field, raman, factor, kind=self._kind, thg=raman_thg(responses)
                )
            elif stage is ResponseStage.PLASMA_CURRENT:
                channels = responses.plasma_channels
                if len(channels) > 1:
                    densities = bundle.plasma_densities
                    initials = bundle.plasma_preionisation
                else:
                    densities = (
                        bundle.plasma_densities[0]
                        if bundle.plasma_densities
                        else bundle.density,
                    )
                    initials = (bundle.preionisation,)
                constants = tuple(
                    (
                        density
                        * channel.charge_mass_ratio
                        * channel.electrons_per_event,
                        density * channel.ionisation_potential,
                    )
                    for channel, density in zip(channels, densities, strict=True)
                )
                physical = field * float(2.0**k)
                current = device.plasma_current(
                    physical,
                    self._rate_tables,
                    constants,
                    tuple(initials),
                    self._problem.grids.dt_fine,
                    responses.plasma_integral,
                    evaluate=self.evaluate_stages,
                ) * float(2.0**-k)
            elif stage not in _SOURCE_STAGES:
                msg = f"mlx does not run the {stage.name} stage"
                raise BackendError(msg)
        return DeviceSources(total * self._t_window, current)

    def _device_envelope(
        self, field: mx.array, bundle: StageScalars
    ) -> tuple[mx.array, mx.array]:
        """Return the envelope plasma's windowed drive and loss at a chunk of nodes.

        The oracle's ``envelope_sources`` on the device: each channel's cycle moments by
        :func:`~solprop.backends.mlx_gpu.envelope.envelope_cycle`, its exposure by the
        trapezoid's doubling scan, its populations in their stable forms, and the drive
        ``(e^2/m) N f E`` and the loss ``I_p N sigma J_w``, both divided by ``s``. On
        MLX's CPU device the oracle's host evaluation, in binary64.

        Parameters
        ----------
        field : mlx.core.array
            The scaled envelope at the nodes, ``(n_nodes, n_pol, n_to)``.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        tuple of mlx.core.array
            The drive and the loss, complex64.
        """
        if not on_gpu():
            drive, loss = super()._envelope_sources(np.array(field), bundle)
            return mx.array(drive), mx.array(loss)
        envelope = self._problem.responses.envelope_plasma
        assert envelope is not None
        k = self._scale_exponent
        half_step = float(np.float32(0.5 * self._problem.grids.dt_fine))
        drive = mx.zeros_like(field)
        loss = mx.zeros_like(field)
        for channel, cycle, density, initial in zip(
            envelope.channels,
            self._cycles,
            bundle.plasma_densities,
            bundle.plasma_preionisation,
            strict=True,
        ):
            rate, current, failures = envelope_cycle(field, cycle, k)
            self._envelope_failures.append(failures)
            increments = (rate[..., :-1] + rate[..., 1:]) * half_step
            zero = mx.zeros((*rate.shape[:-1], 1), dtype=mx.float32)
            exposure = mx.concatenate([zero, prefix_scan(increments)], axis=-1)
            fraction, survival = population(exposure, initial)
            weight = combine(channel.charge_mass_ratio, density).binary32()
            work = combine(
                channel.ionisation_potential, density, exponent=-k
            ).binary32()
            drive = drive + (weight * fraction)[:, None, :] * field
            loss = loss + (work * survival)[:, None, :] * current
        return drive * self._t_window, loss * self._t_window

    def _envelope_populations(
        self,
        nodes: Complex[np.ndarray, "n_nodes n_pol n_to"],
        initials: tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return each channel's final fraction and peak rate, by the device's moments.

        The statistics' node scan with the stage's own formulation, so that a
        statistic and the propagation see the same populations; on MLX's CPU device
        the oracle's.

        Parameters
        ----------
        nodes : numpy.ndarray
            The physical envelope at the nodes, V/m.
        initials : tuple of float
            The channels' pre-ionised fractions here.

        Returns
        -------
        tuple of numpy.ndarray
            Final fractions and peak rates, ``(n_channels, n_nodes)``, binary64.

        Raises
        ------
        solprop.exceptions.ConvergenceError
            If a sample's refinement failed.
        """
        if not on_gpu():
            return super()._envelope_populations(nodes, initials)
        # The statistics' field is physical: scaled by the state's own power of two,
        # it is binary32 at the propagation's range.
        k = self._scale_exponent
        field = mx.array((nodes * math.ldexp(1.0, -k)).astype(np.complex64))
        half_step = float(np.float32(0.5 * self._problem.grids.dt_fine))
        fractions, peaks, failures = [], [], []
        for cycle, initial in zip(self._cycles, initials, strict=True):
            rate, _, failed = envelope_cycle(field, cycle, k)
            increments = (rate[..., :-1] + rate[..., 1:]) * half_step
            exposure = prefix_scan(increments)[..., -1]
            fraction, _ = population(exposure, initial)
            fractions.append(fraction)
            peaks.append(mx.max(rate, axis=-1))
            failures.append(failed)
        mx.eval(fractions, peaks, failures)
        self._raise_on_failures(float(sum(int(cast(int, f.item())) for f in failures)))
        return (
            np.array(mx.stack(fractions), dtype=np.float64),
            np.array(mx.stack(peaks), dtype=np.float64),
        )

    def _envelope_spectrum(self, drives: list[mx.array]) -> mx.array:
        """Return the drive's and the loss's spectrum times their factors.

        Parameters
        ----------
        drives : list of mlx.core.array
            The drive and the loss on the fine time grid.

        Returns
        -------
        mlx.core.array
            ``-drive / omega**2 - i loss / omega`` on the coarse grid.
        """
        assert self._envelope_factors is not None
        drude, loss = self._envelope_factors
        return drude * device.to_frequency(
            drives[0], self._crop, kind=self._kind
        ) + loss * device.to_frequency(drives[1], self._crop, kind=self._kind)

    def _failure_count(self) -> mx.array:
        """Return the trial's envelope refinement failures, float32 (zero without).

        Returns
        -------
        mlx.core.array
            A scalar.
        """
        total = mx.zeros((), dtype=mx.float32)
        for count in self._envelope_failures:
            total = total + count.astype(mx.float32)
        return total

    def _raise_on_failures(self, count: float) -> None:
        """Refuse a trial whose cycle moments did not converge, as the oracle does.

        Parameters
        ----------
        count : float
            The trial's failed samples.

        Raises
        ------
        solprop.exceptions.ConvergenceError
            If any sample's refinement failed by the 512-point rule.
        """
        if count:
            msg = (
                "envelope plasma directional moments did not converge by order 512 "
                f"at {int(count)} samples"
            )
            raise ConvergenceError(msg)

    def _device_norm(
        self, error: mx.array, state: mx.array, advanced: mx.array
    ) -> float:
        """Return the controller's norm from scaled sums of squares on the device.

        Parameters
        ----------
        error : mlx.core.array
            The embedded difference carried to the end of the step.
        state : mlx.core.array
            The state at the anchor.
        advanced : mlx.core.array
            The trial state.

        Returns
        -------
        float
            The norm, binary64; infinite if any value is not finite.
        """
        index = self._norm_index
        residual = mx.take(error, index, axis=-1)
        reductions = []
        for array in (
            residual,
            mx.take(state, index, axis=-1),
            mx.take(advanced, index, axis=-1),
        ):
            peak = mx.max(mx.abs(array))
            # The power of two at or below the peak, so the squares cannot overflow
            # (SEP 33 section 4.3, rule 6); the host restores it in binary64.
            power = mx.where(peak > 0, mx.floor(mx.log2(peak)), 0.0)
            scaled = array * mx.power(2.0, -power)
            re, im = mx.real(scaled), mx.imag(scaled)
            reductions += [power, mx.sum(re * re + im * im)]
        # One synchronisation per trial: the three sums, their powers and whether the
        # whole trial state is finite, which spares the finish a check of its own.
        reductions.append(mx.all(mx.isfinite(advanced)).astype(mx.float32))
        reductions.append(self._failure_count())
        values = self._synchronised(reductions)
        self._raise_on_failures(values[-1])
        if not np.all(np.isfinite(values[:-1])) or values[-2] != 1.0:
            return math.inf
        squares = [
            math.ldexp(values[2 * k + 1], 2 * int(values[2 * k])) for k in range(3)
        ]
        scale = self._controller.scale(math.sqrt(squares[1]), math.sqrt(squares[2]))
        if isinstance(scale, np.ndarray):
            weights = mx.array(
                (1.0 / scale[self._problem.grids.norm_mask]).astype(np.float32)
            )
            weighted = residual * weights
            re, im = mx.real(weighted), mx.imag(weighted)
            return math.sqrt(float(cast(float, mx.sum(re * re + im * im).item())))
        return math.sqrt(squares[0]) / scale

    def _update_alarm(self) -> None:
        """Evaluate the quadrature alarm on the host from the last stage's rows."""
        held = (
            self._last_projected,
            self._last_coarse,
            self._last_plasma,
            self._last_envelope,
        )
        to_host = lambda value: None if value is None else np.array(value)  # noqa: E731
        self._last_projected = to_host(held[0])  # pyright: ignore[reportAttributeAccessIssue]
        self._last_coarse = to_host(held[1])  # pyright: ignore[reportAttributeAccessIssue]
        if held[2] is not None:
            self._last_plasma = tuple(to_host(part) for part in held[2])  # pyright: ignore[reportAttributeAccessIssue]
        if held[3] is not None:
            self._last_envelope = tuple(to_host(part) for part in held[3])  # pyright: ignore[reportAttributeAccessIssue]
        try:
            super()._update_alarm()
        finally:
            (
                self._last_projected,
                self._last_coarse,
                self._last_plasma,
                self._last_envelope,
            ) = held  # pyright: ignore[reportAttributeAccessIssue]

    # The low-storage march's hooks and engine.

    def _adams_lab_source(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, lab: mx.array, position: float, *, purpose: str = "predictor"
    ) -> mx.array:
        """Evaluate the source at a lab-frame field on the device, untransported.

        Parameters
        ----------
        lab : mlx.core.array
            The lab-frame field.
        position : float
            Where it is evaluated, m.
        purpose : str, optional
            Unused; TDSE responses are refused.

        Returns
        -------
        mlx.core.array
            The source in the lab frame.
        """
        del purpose
        bundle = self._stage_scalars(position, position)
        return self._device_nonlinear(lab, bundle, self._unit)

    def _adams_propagator(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, anchor: float, position: float
    ) -> mx.array:
        """Return ``exp(Phi(position) - Phi(anchor))`` on the device.

        Parameters
        ----------
        anchor : float
            Where the flow starts, m.
        position : float
            Where it ends, m.

        Returns
        -------
        mlx.core.array
            The double-float accurate propagator, complex64.
        """
        return self._propagators(self._stage_scalars(anchor, position))[0]

    def _low_storage_engine(self, method: LowStorageRungeKutta) -> object:
        """Return the device engine of the low-storage march.

        Parameters
        ----------
        method : solprop.stepper.low_storage.LowStorageRungeKutta
            The method.

        Returns
        -------
        MlxLowStorage
            The engine.
        """
        del method
        return MlxLowStorage(self)

    def memory_ledger(self) -> dict[str, int]:
        """Return the reference's ledger with MLX's device memory beside it.

        Returns
        -------
        dict
            The oracle's rows, ``mlx_device_active`` and ``mlx_device_peak`` (MLX's
            own counters, measured) and ``mlx_device_cache``.
        """
        return super().memory_ledger() | {
            "mlx_device_active": int(mx.get_active_memory()),
            "mlx_device_peak": int(mx.get_peak_memory()),
            "mlx_device_cache": int(mx.get_cache_memory()),
        }

    def metadata(self) -> dict[str, object]:
        """Return the backend's metadata with MLX's version, device and GEMM probe.

        Returns
        -------
        dict
            The reference's metadata and the ``mlx`` entries.
        """
        return super().metadata() | {
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "mlx_device": str(mx.default_device()),
            "mlx_gemm_probe": self.gemm_error,
            "mlx_fft_probe": self.fft_error,
        }


class MlxLowStorage:
    """The low-storage RK4IP engine on the device (SEP 24 section 6.5).

    The host engine's four register passes
    (:class:`~solprop.backends.low_storage.HostLowStorage`) with device arrays; the
    half-step flows are formed every step by the double-float kernel, which agrees with
    the host's tabulated flows to rounding.

    Parameters
    ----------
    backend : MlxBackend
        The backend whose state is the register ``A``.
    """

    atomic = False

    def __init__(self, backend: MlxBackend) -> None:
        """Hold the backend."""
        self._backend = backend
        self.stiffness = math.nan

    def reset(self) -> None:
        """Forget the last step's indicator."""
        self.stiffness = math.nan

    def close(self) -> None:
        """Hold nothing between steps."""

    def held(self) -> dict[str, int]:
        """Return the bytes held between steps: none, the registers are transients.

        Returns
        -------
        dict
            Zero rows, as the host engine names them.
        """
        return {"register_accumulator": 0, "register_stage": 0, "propagators": 0}

    def _norm(self, values: mx.array) -> float:
        """Return the norm of values over the norm mask.

        Parameters
        ----------
        values : mlx.core.array
            A state-shaped array.

        Returns
        -------
        float
            The norm, binary64 on the host.
        """
        # Read at use: the march is built inside the backend's own construction.
        backend = self._backend
        picked = mx.take(values, backend._norm_index, axis=-1)
        re, im = mx.real(picked), mx.imag(picked)
        # Through the backend's read, which takes a pending split removal with it.
        return math.sqrt(float(backend._synchronised([mx.sum(re * re + im * im)])[0]))

    def step(self, z: float, dz: float) -> float:
        """Take one step, leaving the raw accepted state in the backend.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            The accepted state's squared norm over the norm mask.
        """
        backend = self._backend
        midpoint, end = z + 0.5 * dz, z + dz
        backend._hoist(z)
        backend._stage_bundles(z, [z, midpoint])
        backend._stage_bundles(midpoint, [midpoint, end])
        backend._stage_bundles(end, [end])
        backend._envelope_failures = []
        state = cast(mx.array, backend._state)
        source = backend._adams_lab_source(state, z)
        first = backend._adams_propagator(z, midpoint)
        state = first * state
        slope = first * source
        accumulator = state + (dz / 6.0) * slope
        stage = state + (0.5 * dz) * slope
        source = backend._adams_lab_source(stage, midpoint)
        lower = self._norm(source - slope)
        accumulator = accumulator + (dz / 3.0) * source
        stage = state + (0.5 * dz) * source
        previous = source
        source = backend._adams_lab_source(stage, midpoint)
        upper = self._norm(source - previous)
        accumulator = accumulator + (dz / 3.0) * source
        second = backend._adams_propagator(midpoint, end)
        stage = second * (state + dz * source)
        source = backend._adams_lab_source(stage, end)
        state = second * accumulator + (dz / 6.0) * source
        backend._state = cast(Any, state)
        self.stiffness = 2.0 * upper / lower if lower > 0.0 else math.nan
        if backend._envelope_failures:
            backend._raise_on_failures(
                float(cast(float, backend._failure_count().item()))
            )
        norm = self._norm(state)
        return norm * norm
