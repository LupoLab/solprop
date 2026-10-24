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

Phase 2 of SEP 33 admits the fibre route: mode-averaged and multimode capillaries,
solid-core and GNLSE fibres on field and envelope grids, with Kerr, the envelope grid's
harmonic, scalar and tensor Raman, gas mixtures, the silica bundle, the field-grid
plasma with its held current, loss, both temporal absorber schemes, the polarisation
taper and device-ready gradients, stepped by Runge-Kutta. Everything else is refused
by name.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import mlx.core as mx
import numpy as np
from jaxtyping import Complex

from solprop.backends.memory import mlx_graph_bytes
from solprop.backends.mlx_gpu import passes as device
from solprop.backends.mlx_gpu.arithmetic import DoubleFloatTerms, gemm_probe, on_gpu
from solprop.backends.numpy_ref.passes import NonlinearSources
from solprop.backends.numpy_ref.passes import _kerr as compiled_kerr
from solprop.backends.numpy_ref.passes import _raman_thg as raman_thg
from solprop.backends.numpy_ref.single import NumpySingleBackend
from solprop.constants import EPS0
from solprop.exceptions import BackendError, ConvergenceError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.precision import block_exponent, combine
from solprop.translate.problem import KernelProblem
from solprop.translate.responses import ResponseStage
from solprop.translate.stage import StageScalars
from solprop.translate.tdse import TDSEBinding

__all__ = ["MlxBackend", "refuse_unsupported"]

_SOURCE_STAGES = frozenset(
    {
        ResponseStage.KERR,
        ResponseStage.RAMAN_DRIVE,
        ResponseStage.PLASMA_CURRENT,
        ResponseStage.INTENSITY,
        ResponseStage.IONISATION_RATE,
        ResponseStage.PLASMA_SCAN,
        ResponseStage.ACCUMULATE,
    }
)


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
            problem.transform.geometry is not None,
            "radial and Cartesian free space (SEP 33 phase 3); use backend='numpy_ref'",
        ),
        (
            problem.responses.envelope_plasma is not None,
            "the envelope-grid plasma (SEP 33 phase 4); use backend='numpy_ref'",
        ),
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
    """Propagate a single-precision fibre problem on an Apple-silicon GPU.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A validated single-precision problem on the fibre route.
    grid : solprop.grid.TimeGrid, optional
        The grid, needed only by :meth:`stats`.
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
        if self._terms.phi is None:
            raise BackendError("mlx needs the operator's device-ready terms")
        self._phi = DoubleFloatTerms.upload(self._terms.phi)
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
        self._raman_upload: tuple[object, mx.array] | None = None
        self._unit = mx.ones((1,), dtype=mx.complex64)
        # Whether each stage is evaluated as it is formed, bounding the live graph to
        # one stage; a batch evaluates its members' steps together instead.
        self.evaluate_stages = True
        n_to = grids.n_to
        n_pol = transform.n_pol
        chunk = min(transform.node_chunk, transform.n_nodes) * n_pol
        shapes = [(chunk, transform.n_modes, problem.shape[2] * n_to)]
        shapes.append((transform.n_modes, chunk, problem.shape[2] * n_to))
        self.gemm_error = gemm_probe(shapes) if on_gpu() else 0.0
        # MLX keeps freed buffers for reuse; its cache is held to one stage's graph,
        # the admission estimate's own row (SEP 33 section 6.3).
        mx.set_cache_limit(mlx_graph_bytes(problem))

    # The state on the device.

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
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
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
            if self._kind == "field":
                time = mx.fft.irfft(finished, n=rate.size, axis=-1) * change
                finished = finished + mx.fft.rfft(time, axis=-1) * self._active
            else:
                time = mx.fft.ifft(finished, axis=-1) * change
                finished = finished + mx.fft.fft(time, axis=-1) * self._active
        if not bool(mx.all(mx.isfinite(finished)).item()):
            msg = (
                f"mlx: the state is not finite after a step of {dz:.4g} m from z = "
                f"{self._z:.6g} m: the field has diverged or left binary32's range"
            )
            raise ConvergenceError(msg)
        return finished

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
                mx.eval(stage)
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

    def _device_nonlinear(
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
        modal_t = device.to_time(lab[0], self._crop, grids.n_to, kind=kind)
        absorber = self._device_absorber(modal_t)
        if len(responses.response_order) == 1:
            if absorber is None:
                return mx.zeros_like(lab)
            return back * absorber[None]
        raman = self._device_raman(bundle)
        taper = bundle.taper_scale
        n_pol = transform.n_pol
        projected = None
        coarse = None
        gated = edge = coarse_gated = coarse_edge = None
        for first in range(0, transform.n_nodes, transform.node_chunk):
            rows = slice(first * n_pol, (first + transform.node_chunk) * n_pol)
            field = device.synthesise(modal_t, self._synthesis[rows], taper, n_pol)
            sources = self._device_sources(field, bundle, raman)
            polarisation = cast(mx.array, sources.polarisation)
            columns = self._projection[:, rows]
            part = device.project(polarisation, columns, taper, n_pol)
            projected = part if projected is None else projected + part
            if self._coarse is not None:
                reduced = device.project(
                    polarisation, self._coarse[:, rows], taper, n_pol
                )
                coarse = reduced if coarse is None else coarse + reduced
            if sources.current is not None:
                current = cast(mx.array, sources.current)
                g = device.project(self._t_window * current, columns, taper, n_pol)
                e = device.project(
                    self._t_window_derivative * current, columns, taper, n_pol
                )
                gated = g if gated is None else gated + g
                edge = e if edge is None else edge + e
                if self._coarse is not None:
                    cc = self._coarse[:, rows]
                    cg = device.project(self._t_window * current, cc, taper, n_pol)
                    ce = device.project(
                        self._t_window_derivative * current, cc, taper, n_pol
                    )
                    coarse_gated = cg if coarse_gated is None else coarse_gated + cg
                    coarse_edge = ce if coarse_edge is None else coarse_edge + ce
        assert projected is not None
        self._last_projected = projected  # pyright: ignore[reportAttributeAccessIssue]
        self._last_coarse = coarse  # pyright: ignore[reportAttributeAccessIssue]
        self._last_bundle = bundle
        self._last_plasma = (
            None if gated is None else (gated, edge, coarse_gated, coarse_edge)  # pyright: ignore[reportAttributeAccessIssue]
        )
        spectrum = device.to_frequency(projected, self._crop, kind=kind)
        if gated is not None and edge is not None:
            spectrum = spectrum + self._device_held(gated, edge)
        normalised = self._device_prefactor(bundle) * self._omega_window * spectrum
        if absorber is not None:
            normalised = normalised + absorber
        return back * normalised[None]

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
        if not self._in_stage_absorber or self._temporal_rate is None:
            return None
        if self._problem.responses.absorber_scheme != "nonlinear":
            return None
        term = -0.5 * self._temporal_rate * modal_t
        return device.to_frequency(term, self._crop, kind=self._kind) * self._active

    def _device_sources(
        self, field: mx.array, bundle: StageScalars, raman: mx.array | None
    ) -> NonlinearSources:
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
        solprop.backends.numpy_ref.passes.NonlinearSources
            The windowed polarisation and the ungated current, device arrays.

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
                ) * float(2.0**-k)
            elif stage not in _SOURCE_STAGES:
                msg = f"mlx does not run the {stage.name} stage"
                raise BackendError(msg)
        return NonlinearSources(total * self._t_window, current)  # pyright: ignore[reportArgumentType]

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
        before = mx.take(state, index, axis=-1)
        after = mx.take(advanced, index, axis=-1)
        peaks = mx.stack(
            [mx.max(mx.abs(residual)), mx.max(mx.abs(before)), mx.max(mx.abs(after))]
        )
        values = np.array(peaks, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            return math.inf
        totals = []
        for array, peak in zip((residual, before, after), values, strict=True):
            exponent = block_exponent(float(peak))
            scaled = array * float(2.0**-exponent)
            re, im = mx.real(scaled), mx.imag(scaled)
            totals.append((mx.sum(re * re + im * im), 2 * exponent))
        sums = np.array(mx.stack([total for total, _ in totals]), dtype=np.float64)
        squares = [
            math.ldexp(float(value), e)
            for value, (_, e) in zip(sums, totals, strict=True)
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
        held = (self._last_projected, self._last_coarse, self._last_plasma)
        to_host = lambda value: None if value is None else np.array(value)  # noqa: E731
        self._last_projected = to_host(held[0])  # pyright: ignore[reportAttributeAccessIssue]
        self._last_coarse = to_host(held[1])  # pyright: ignore[reportAttributeAccessIssue]
        if held[2] is not None:
            self._last_plasma = tuple(to_host(part) for part in held[2])  # pyright: ignore[reportAttributeAccessIssue]
        try:
            super()._update_alarm()
        finally:
            self._last_projected, self._last_coarse, self._last_plasma = held  # pyright: ignore[reportAttributeAccessIssue]

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
        }


class MlxLowStorage:
    """The low-storage RK4IP engine on the device (SEP 24 section 6.5).

    The host engine's four register passes (:class:`~solprop.backends.low_storage.
    HostLowStorage`) with device arrays; the half-step flows are formed every step by
    the double-float kernel, which agrees with the host's tabulated flows to rounding.

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
        picked = mx.take(values, self._backend._norm_index, axis=-1)
        re, im = mx.real(picked), mx.imag(picked)
        return math.sqrt(float(cast(float, mx.sum(re * re + im * im).item())))

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
        norm = self._norm(state)
        return norm * norm
