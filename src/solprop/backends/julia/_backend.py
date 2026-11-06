# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adapt the existing step driver to the experimental Julia-owned state."""

import dataclasses
import logging
import math
import os
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from jaxtyping import Complex

from solprop._process import _BlasLease
from solprop.backends.driver import StepDriver, refuse_general_leaves
from solprop.backends.exponential import refuse_carrier_frame
from solprop.backends.julia._helpers import JULIA
from solprop.backends.julia._runtime import (
    call_julia,
    runtime,
    source_digest,
    thread_limit,
)
from solprop.backends.multistep import source_absent
from solprop.backends.numpy_ref.passes import plasma_integration_statistics
from solprop.backends.spatial import collect_spatial, radial_absorber_matrix
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.output.schema import SavedPlane
from solprop.profiles import ZProfile
from solprop.stats import (
    _POYNTING_STATISTICS,
    StatsSpec,
    _power_half_width,
    _poynting_statistics,
    collect,
    default_spec,
)
from solprop.stepper import LowStorageRungeKutta, Multistep
from solprop.stepper.tableau import DOPRI5
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.linop import power_weight_at
from solprop.translate.precision import launch_peak, state_scale
from solprop.translate.problem import KernelProblem
from solprop.translate.stage import terms
from solprop.transverse_monitor import TransverseEdge, _EdgeReading

__all__ = ["JuliaBackend"]

logger = logging.getLogger(__name__)
_SPECIES_STATS = frozenset(
    (
        "species_density",
        "species_electron_density",
        "species_ionisation",
        "species_ionisation_energy",
        "species_peak_ionisation_rate",
        "electron_density",
        "peak_ionisation_rate",
        "plasma_undershoot",
        "plasma_work_defect",
        "plasma_boundary_level",
    )
)
_SUPPORTED_STATS = (
    frozenset(default_spec("multimode", with_profiles=True).names)
    | _SPECIES_STATS
    | {"peak_intensity", "quadrature_error"}
)
_SPATIAL_STATS = (
    frozenset(default_spec("radial", with_profiles=True).names) | _SPECIES_STATS
)


def _constant(table: PiecewiseChebyshev | None) -> bool:
    """Check the whole polynomial, not just samples at a few positions.

    Parameters
    ----------
    table : solprop.translate.chebyshev.PiecewiseChebyshev or None
        Translated scalar table.

    Returns
    -------
    bool
        Whether the table is absent or constant over every interval.
    """
    if table is None:
        return True
    coefficients = table.coefficients
    return bool(
        np.all(coefficients[:, 1:] == 0)
        and np.all(coefficients[:, 0] == coefficients[0, 0])
    )


def _refuse(problem: KernelProblem) -> None:
    """Refuse unsupported leaves before starting Julia or allocating its state.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Translated schema contract.

    Raises
    ------
    BackendError
        If a requested leaf is outside the experiment's scope.
    """
    response, transform, stepper = (
        problem.responses,
        problem.transform,
        problem.stepper,
    )
    reasons = {
        "envelope plasma": problem.grid_kind == "envelope" and problem.flags.has_plasma,
        "resolved modal or radial envelopes": problem.grid_kind == "envelope"
        and transform.kind not in {"free", "mode_averaged"},
        "envelope harmonics": problem.grid_kind == "envelope"
        and response.kerr_coefficients is not None
        and response.kerr_coefficients.harmonic_factor != 0,
        "single batch and scalar polarisation": problem.shape[0] != 1
        or problem.shape[2] != 1
        or transform.n_pol != 1,
        "planar transforms": transform.geometry is not None
        and transform.geometry.kind not in {"radial", "free"},
        "nonseparable longitudinal tables": not terms(problem).device_ready,
        "varying radius": transform.taper_scale is not None,
        "multiple plasma channels": len(response.plasma_channels) > 1,
        "multiple Raman components": len(response.raman_terms) > 1,
        "varying Raman coefficients": any(
            not _constant(r.density or response.density)
            or any(not _constant(v) for v in r.decay_rates)
            for r in response.raman_terms
        ),
        "Cartesian Raman": problem.flags.has_raman and transform.kind == "free",
        "spectral responses": problem.flags.has_spectral_responses,
        "dispersive Kerr": response.dispersive_kerr is not None,
        "separable responses off free space": response.separable is not None
        and transform.kind != "free",
        "TDSE": problem.tdse is not None,
        "noise": problem.flags.has_noise,
        "interaction or comparison Adams integration": isinstance(
            stepper.integrator, Multistep
        )
        and (
            type(stepper.integrator) is not Multistep
            or stepper.integrator.formulation != "exponential"
        ),
        "Cartesian Adams integration": isinstance(stepper.integrator, Multistep)
        and transform.kind == "free",
        "non-Cartesian low-storage integration": isinstance(
            stepper.integrator, LowStorageRungeKutta
        )
        and transform.kind != "free",
        "non-DOPRI5 tableau": stepper.tableau is not DOPRI5,
        "interpolated planes": stepper.interpolate_planes,
    }
    for reason, refused in reasons.items():
        if refused:
            raise BackendError(f"julia does not support {reason}")
    refuse_general_leaves(problem, "julia")
    refuse_carrier_frame(problem, "julia")
    _refuse_stats(problem.stats_spec, spatial=transform.geometry is not None)


def _compensated(problem: KernelProblem) -> bool:
    """Use the exact chord for structurally absent sources, as SEP 16 requires."""
    return problem.linear.phi_representation in {
        "expansion",
        "table",
    } and not source_absent(problem)


def _varying(problem: KernelProblem) -> bool:
    """Identify stage-dependent material scales and longitudinal expansions."""
    response = problem.responses
    return (
        problem.linear.phi_expansion is not None
        or response.prefactor_expansion is not None
        or not _constant(response.density)
        or (
            response.kerr_coefficients is not None
            and not _constant(response.kerr_coefficients.density)
        )
        or any(
            not _constant(c.density) or not _constant(c.preionisation)
            for c in response.plasma_channels
        )
    )


def _refuse_stats(spec: StatsSpec, *, spatial: bool = False) -> None:
    """Name statistics outside the prototype's declared scope.

    Parameters
    ----------
    spec : solprop.stats.StatsSpec
        Requested statistics.
    spatial : bool, optional
        Whether to use the spatial default/profile statistics subset.

    Raises
    ------
    BackendError
        If any requested statistic is unsupported.
    """
    if missing := set(spec.names) - (_SPATIAL_STATS if spatial else _SUPPORTED_STATS):
        raise BackendError(f"julia does not support statistics {sorted(missing)}")


def _configuration(problem: KernelProblem, threads: int) -> dict[str, object]:
    """Copy the validated constant seam into Julia's frequency-first layout.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Validated constant scalar modal or spatial contract.
    threads : int
        Positive Julia task limit.

    Returns
    -------
    dict
        Owning arrays and scalar configuration consumed once by Julia.
    """
    grids, responses, transform = problem.grids, problem.responses, problem.transform
    shape = problem.shape
    kerr = responses.kerr_coefficients
    density = (
        kerr.density
        if kerr is not None and kerr.density is not None
        else responses.density
    )
    factor = (
        0.0 if kerr is None else kerr.coefficient * float(density.coefficients[0, 0])
    )
    control, tableau = problem.stepper.controller, problem.stepper.tableau
    # One pool runs every pass, transforms and products included (julia/src/launch.jl):
    # a transform is split into at most this many blocks of whole columns.
    fft_threads = thread_limit("SOLPROP_JULIA_FFT_THREADS", threads)
    # A binary32 run holds its state as a / s (SEP 33, rule 2): the cubic responses
    # take s**2, the plasma reads the physical field s E (the engine's) and its loss
    # potential 1 / s**2, and the controller's absolute tolerance is divided by s.
    single = problem.precision.single
    scale = (
        state_scale(launch_peak(problem.initial.field0, problem.grid_kind))
        if single
        else 1.0
    )
    factor *= scale**2
    config: dict[str, object] = {
        "precision": "single" if single else "double",
        "state_scale": scale,
        "adams_order": problem.stepper.integrator.order
        if isinstance(problem.stepper.integrator, Multistep)
        else 0,
        "adams_compensated": _compensated(problem),
        "plasma": bool(responses.plasma_channels),
        "raman": bool(responses.raman_terms),
        "envelope": problem.grid_kind == "envelope",
        "low_storage": isinstance(problem.stepper.integrator, LowStorageRungeKutta),
        "free": transform.kind == "free",
        "linear": np.zeros(
            (0, 0) if transform.kind == "free" else (shape[-1], shape[1]), dtype=complex
        )
        if problem.linear.phi_constant is None
        else np.broadcast_to(problem.linear.phi_constant, shape)[0, :, 0, :].T.copy(),
        "prefactor": np.zeros(
            (0, 0) if transform.kind == "free" else (shape[-1], shape[1]), dtype=complex
        )
        if responses.prefactor_constant is None
        else np.broadcast_to(responses.prefactor_constant, shape[1:])[:, 0, :].T.copy(),
        # Mode averaging already folds the effective-area conversion into synthesis
        # and its reciprocal into prefactor; the shared one-node path uses both.
        "synthesis": transform.synthesis.T.copy(),
        "projection": transform.projection.T.copy(),
        "projection_coarse": transform.projection_coarse.T.copy(),
        "spatial_absorber": radial_absorber_matrix(transform, grids.pad_scale).T.copy(),
        "embedded": not np.array_equal(
            transform.projection, transform.projection_coarse
        ),
        "crop": grids.crop_index.copy(),
        "mask": grids.active_mask.copy(),
        "norm_mask": grids.norm_mask.copy(),
        "weights": np.ones(shape[-1])
        if control.weights is None
        else control.weights.copy(),
        "window": grids.t_window_fine.copy(),
        "omega_window": grids.omega_window.copy(),
        "rate": np.zeros(grids.n_to)
        if responses.temporal_absorber_fine is None
        else responses.temporal_absorber_fine.copy(),
        "coarse_rate": np.zeros(grids.n_t)
        if responses.temporal_absorber is None
        else responses.temporal_absorber.copy(),
        "a": tableau.matrix.copy(),
        "b": tableau.solution_weights.copy(),
        "e": tableau.error_weights.copy(),
        "c": tableau.nodes.copy(),
        "factor": factor,
        "analytic": kerr is not None and kerr.analytic_signal,
        "analytic_factor": 0.0
        if kerr is None
        else kerr.self_factor + kerr.conjugate_factor,
        "split": responses.absorber_scheme == "split_step",
        "rtol": control.rtol,
        "atol": control.atol / scale,
        "cache_propagators": transform.kind != "free"
        and problem.stepper.propagator_mode != "recompute",
        "threads": threads,
        "fft_threads": fft_threads,
        "raman_fft_threads": fft_threads,
        "transverse_fft_threads": fft_threads,
        "blas_threads": 1,
        "chunk": transform.node_chunk,
        "device": os.environ.get("SOLPROP_JULIA_DEVICE", "cpu"),
    }

    if _varying(problem):
        held = terms(problem)
        assert held.phi is not None and held.prefactor is not None
        config.update(
            phi_terms=np.broadcast_to(held.phi, (len(held.phi), *shape))[:, 0, :, 0, :]
            .transpose(2, 1, 0)
            .copy(),
            prefactor_terms=np.broadcast_to(
                held.prefactor, (len(held.prefactor), *shape[1:])
            )[:, :, 0, :]
            .transpose(2, 1, 0)
            .copy(),
        )

    if responses.plasma_channels:
        channel = responses.plasma_channels[0]
        table = channel.rate
        config.update(
            plasma_coefficients=table.coefficients.copy(),
            plasma_parameters=np.array(
                [
                    table.field_min,
                    table.field_max,
                    table.rate_max,
                    table.log_field_min,
                    table.d_log_field,
                    channel.charge_mass_ratio,
                    channel.ionisation_potential / scale**2,
                    float(channel.density(0.0)),
                    float(channel.preionisation(0.0)),
                ]
            ),
            plasma_taps=responses.plasma_integral.taps.copy(),
            plasma_floor=responses.plasma_integral.policy == "band",
            dt=grids.dt_fine,
            window_derivative=grids.t_window_fine_derivative.copy(),
            omega=grids.omega.copy(),
        )

    if responses.raman_terms:
        raman = responses.raman_terms[0]
        config.update(
            raman_undamped=raman.undamped.T.copy(),
            raman_rates=np.array([float(rate(0.0)) for rate in raman.decay_rates]),
            raman_fraction=raman.fraction,
            raman_factor=EPS0
            * float((raman.density or responses.density)(0.0))
            * scale**2,
            raman_thg=raman.thg,
            dt=grids.dt_fine,
        )

    if transform.kind == "free":
        data, geometry = problem.linear.separable, transform.geometry
        assert data is not None and geometry is not None
        config.update(
            spatial_shape=geometry.shape,
            pad_scale=grids.pad_scale,
            spatial_rate=geometry.absorber_rate.copy(),
            evanescent_rate=data.evanescent_rate,
        )
        for name in (
            "k_squared",
            "k_squared_low",
            "kperp_sq",
            "kperp_sq_low",
            "frame",
            "spectral_rate",
            "prefactor_numerator",
            "prefactor_clamped",
            "active",
            "kperp_rate",
        ):
            value = getattr(data, name)
            assert value is not None
            config[name] = value.copy()
    return config


class JuliaBackend(StepDriver[object]):
    """Propagate the experimental scalar core subset with resident Julia kernels.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Supported modal or spatial problem, in SI units and complex128.
    grid : solprop.grid.TimeGrid, optional
        Temporal grid, needed for statistics.
    modes : collections.abc.Sequence, optional
        Modes passed to the shared host statistics functions.
    profiles : collections.abc.Mapping, optional
        Profiles passed to the shared host statistics functions.

    Raises
    ------
    BackendError
        If the problem or the private Julia runtime is unsupported.
    """

    name = "julia"
    max_batch = 1
    # Shared scalar tables and bookkeeping remain plain Python. Numerical state
    # and complete propagation trials stay in Julia.
    helpers = JULIA
    propagator_default = "tabulate"
    # The transverse edge (SEP 40/44) is read by the driver's host reduction.
    reads_transverse_edge = True

    def __init__(
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
    ) -> None:
        _refuse(problem)
        problem.validate()
        super().__init__(problem, grid=grid, modes=modes, profiles=profiles)
        self._jl = runtime()
        # The loaded module is the one on disk: a source edit after the runtime
        # started would otherwise time or test stale code.
        self._digest = source_digest()
        if str(self._jl.loaded_digest()) != self._digest:
            raise BackendError("julia sources changed after the runtime loaded them")
        self._threads = thread_limit("SOLPROP_JULIA_THREADS")
        self._fixed: dict[str, object] | None = None
        self._poynting_weight: np.ndarray | None = None
        if not 1 <= self._threads <= int(self._jl.runtime()[1]):
            raise BackendError("julia thread count must fit the initialized runtime")
        self._engine = call_julia(
            self._jl.create, _configuration(problem, self._threads)
        )
        self._varying = _varying(problem)
        self._scale2 = (
            state_scale(launch_peak(problem.initial.field0, problem.grid_kind)) ** 2
            if problem.precision.single
            else 1.0
        )
        if self._varying:
            self._prepare_stages(0.0, 0.0)
        self._plan_flag = os.environ.get("SOLPROP_FFTW_PLAN") or "measure"
        self._weight = (
            np.empty(0)
            if problem.transform.geometry is not None
            else power_weight_at(problem, 0.0)
        )
        self._fsal = False
        self._recycle = _responses_recycle(problem)
        self._alarm = math.nan
        self._alarm_reported = False
        self._host_blas = _hold_host_blas()
        self._edge_key: tuple[int, bool] | None = None
        self._warm_bridge()
        if isinstance(problem.stepper.integrator, LowStorageRungeKutta):
            from solprop.backends.low_storage import LowStorageMarch  # noqa: PLC0415

            self._low_storage = LowStorageMarch(self, problem.stepper.integrator)

        if isinstance(problem.stepper.integrator, Multistep):
            from solprop.backends.multistep import AdamsMarch  # noqa: PLC0415

            self._adams = AdamsMarch(self, problem.stepper.integrator)

    def _warm_bridge(self) -> None:
        """Call each bridge entry the march reaches with the arguments it will pass.

        The engine compiled and ran its own paths at construction; what is left is
        the bridge's per-function wrappers and argument conversions, which otherwise
        compile inside a run's first boundary reading or edge read.
        """
        _, n_modes, _, n_omega = self._problem.shape
        jl, engine = self._jl, self._engine
        # A crossing's bucket table is read-only, which is part of its bridge type.
        buckets = np.zeros((n_modes, n_omega), dtype=np.uint8)
        for columns, table in ((None, None), (np.ones(n_modes), buckets.T)):
            call_julia(jl.bind_monitor_b, engine, columns, table)
        for name in ("state_power", "occupancy", "split_power", "column_energy"):
            np.asarray(call_julia(getattr(jl, name), engine))
        free = self._problem.transform.kind == "free"
        inverse = None if free else np.eye(n_modes)
        call_julia(jl.bind_edge_b, engine, inverse, np.zeros(1, dtype=np.int64))
        call_julia(jl.node_energy, engine)
        call_julia(jl.device_name, engine)
        self._observed_field(engine)
        if self._problem.flags.has_plasma:
            call_julia(jl.population_snapshot, engine, None)
        if self._weight.size:
            # The Poynting transforms' plan is measured here, not at the first plane.
            axis = None if self._modes is None else self._on_axis_synthesis(self._modes)
            self._bind_poynting(axis)
            for flux in (False, axis is not None):
                np.asarray(call_julia(jl.poynting_b, engine, flux)[0])
        call_julia(jl.bind_monitor_b, engine, None, None)
        self.metadata()

    def _adams_state(self) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Copy the resident state for rare host persistence and budget reports."""
        return self._observed_field(self._require_state())

    def _adams_restore(
        self, z: float, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> None:
        """Restore a snapshot field before continuing its resident history."""
        self._state = self._apply_mask(state)
        self._z = z

    def _adams_engine(
        self, method: Multistep, compensation: Literal["transported", "untransported"]
    ) -> object:
        """Keep exponential samples in Julia behind the shared Adams controller."""
        from solprop.backends.julia._integrators import JuliaAdams  # noqa: PLC0415

        if compensation != "transported":
            raise BackendError(
                "julia does not support untransported Adams compensation"
            )
        return JuliaAdams(self, method)

    def _low_storage_engine(self, method: LowStorageRungeKutta) -> object:
        """Keep RK4IP registers in Julia and reuse the shared schedule march."""
        from solprop.backends.julia._integrators import JuliaRK4  # noqa: PLC0415

        return JuliaRK4(self, method)

    def _apply_mask(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> object:
        """Load and mask an initial field in Julia-owned storage.

        Parameters
        ----------
        state : numpy.ndarray
            Internal field with the canonical seam layout.

        Returns
        -------
        object
            Opaque Julia engine handle.
        """
        # The engine reads a contiguous copy's column-major (n_omega, n_modes) view.
        values = np.ascontiguousarray(state[0, :, 0, :], dtype=np.complex128)
        call_julia(self._jl.load_b, self._engine, values.T)
        self._fsal = False
        self._alarm = math.nan
        return self._engine

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
        resume: bool = False,
    ) -> None:
        """Load an owning field and reset the controller and FSAL cache.

        Parameters
        ----------
        field : numpy.ndarray
            Internal complex128 field of shape (1, n_modes, 1, n_omega).
        noise : solprop.noise.NoiseRealisation or None
            Must be None for this prototype.
        z : float, optional
            Initial position in metres.
        resume : bool, optional
            Whether `field` came back from a saved plane of this run.

        Raises
        ------
        BackendError
            If noise is supplied or the field is nonfinite.
        """
        if noise is not None or not np.all(np.isfinite(field)):
            raise BackendError("julia requires finite initial fields without noise")
        super().set_initial(field, noise, z=z, resume=resume)
        if self._adams is not None:
            self._adams.reset()

    def _march_to(self, target: float) -> None:
        """Dispatch adaptive and replayed Adams runs to the existing march."""
        if self._adams is None:
            super()._march_to(target)
        else:
            self._adams.march_to(target)

    def _trial_step(self, z: float, dz: float) -> tuple[float, object]:
        """Evaluate a trial without changing the accepted state.

        Parameters
        ----------
        z : float
            Start position, m, at which the existing stage tables are evaluated.
        dz : float
            Trial step length, m.

        Returns
        -------
        float
            Dimensionless controller error norm.
        object
            Opaque engine whose candidate buffer awaits acceptance.

        Raises
        ------
        BackendError
            If the trial produces a nonfinite norm or state is unavailable.
        """
        self._require_state()
        if self._varying:
            self._prepare_stages(z, dz)
        norm = float(call_julia(self._jl.trial_b, self._engine, dz, nogil=True))
        self._rk_source_evaluations += 6 if self._fsal else 7
        self._fsal = True
        if not np.isfinite(norm):
            raise BackendError("julia produced a nonfinite trial error norm")
        return norm, self._engine

    def _prepare_stages(self, z: float, dz: float) -> None:
        """Transfer only the existing seam's scalar bundles before a Julia trial."""
        problem = self._problem
        nodes = problem.stepper.tableau.nodes
        # Resident Adams reads only the first material row and endpoint flow/
        # material row. Retain the fixed buffer shape without evaluating unused
        # DOPRI nodes; repeated positions (including dz=0) share a pure bundle.
        offsets = (
            [0.0] * (len(nodes) - 1) + [dz]
            if isinstance(problem.stepper.integrator, Multistep)
            else [float(c) * dz for c in nodes]
        )
        self._hoist(z)
        bundles = self._stage_bundles(z, [z + offset for offset in offsets])
        kerr = problem.responses.kerr_coefficients
        material = np.array(
            [
                [
                    0.0
                    if kerr is None
                    else kerr.coefficient * b.kerr_density * self._scale2,
                    b.plasma_densities[0] if b.plasma_densities else 0.0,
                    b.preionisation,
                ]
                for b in bundles
            ]
        ).T.copy()
        deltas, prefactors = [], []
        for bundle in bundles:
            assert bundle.phi_delta is not None and bundle.prefactor is not None
            deltas.append(bundle.phi_delta)
            prefactors.append(bundle.prefactor)
        call_julia(
            self._jl.configure_stages_b,
            self._engine,
            np.stack(deltas, axis=1),
            np.stack(prefactors, axis=1),
            material,
        )

    def _finish(self, state: object, dz: float) -> object:
        """Commit the candidate and apply accepted-step boundary treatments.

        Parameters
        ----------
        state : object
            Opaque candidate handle returned by the trial.
        dz : float
            Accepted step length, m.

        Returns
        -------
        object
            Opaque accepted state handle.
        """
        monitor = self._split_monitor if _splits(self._problem) else None
        self._alarm = float(
            call_julia(
                self._jl.commit_b, self._engine, dz, monitor is not None, nogil=True
            )
        )
        if monitor is not None:
            # What the split collar takes and finds, from the power of the masked
            # state before it, which the commit formed: BoundaryMonitor.split_sums.
            power = np.asarray(call_julia(self._jl.split_power, self._engine))
            rate = monitor.weights[3]
            half = power * np.exp(-0.5 * rate * dz)
            self._split_removed += float(power @ -np.expm1(-rate * dz))
            self._split_found = np.concatenate(
                [[half.sum()], monitor.weights[:3] @ half]
            )
        if self._alarm > 1e-6 and not self._alarm_reported:
            self._alarm_reported = True
            logger.warning(
                "transverse quadrature error %.2e exceeds 1e-6 at z=%.6g m; "
                "refine the quadrature",
                self._alarm,
                self._z + dz,
            )
        self._fsal = self._recycle
        return state

    def _begin_plane(self) -> None:
        """Reset the per-plane quadrature diagnostic's reporting flag."""
        self._alarm_reported = False

    def _observed_field(
        self, state: object
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Copy an accepted state back to the canonical host layout.

        Parameters
        ----------
        state : object
            Opaque accepted state handle.

        Returns
        -------
        numpy.ndarray
            Independent complex128 host array in the seam layout.
        """
        del state
        # The engine fills a host array, through its column-major (n_omega, n_modes)
        # view, with the pool's threads.
        _, n_modes, _, n_omega = self._problem.shape
        values = np.empty((n_modes, n_omega), dtype=np.complex128)
        call_julia(self._jl.copy_state_b, self._engine, values.T)
        return values[None, :, None, :]

    def _bind_monitor(self, monitor: BoundaryMonitor | None) -> None:
        """Hold the monitor's column weights and rate buckets in the engine.

        Parameters
        ----------
        monitor : solprop.monitor.BoundaryMonitor or None
            The run's monitor, or ``None`` to stop.
        """
        super()._bind_monitor(monitor)
        if monitor is None:
            return
        crossing = monitor.crossing
        call_julia(
            self._jl.bind_monitor_b,
            self._engine,
            None if monitor.columns is None else np.asarray(monitor.columns),
            None
            if crossing is None
            else np.ascontiguousarray(crossing.buckets, dtype=np.uint8).T,
        )

    def _edge_reading(
        self,
        state: object,
        edge: TransverseEdge,
        *,
        faces: bool = True,
        current: bool = False,
        full: bool = False,
    ) -> _EdgeReading:
        """Read the transverse edge from the engine's node and column energies.

        The state-sized sums are the engine's; the face tables act on the short
        vectors here. A collar's strip and inward current keep the host reduction.

        Parameters
        ----------
        state : object
            The backend's state handle.
        edge : solprop.transverse_monitor.TransverseEdge
            The run's edge.
        faces : bool, optional
            Whether the nodes, the columns and any collar's strip are wanted.
        current : bool, optional
            Whether a collar's inward current is wanted.
        full : bool, optional
            Whether every node is wanted, not only the rows a read uses.

        Returns
        -------
        solprop.transverse_monitor._EdgeReading
            The reading.
        """
        reduction = edge.reduction
        if current or (faces and reduction.lines):
            return super()._edge_reading(
                state, edge, faces=faces, current=current, full=full
            )
        if not faces:
            return _EdgeReading()
        tables = reduction.tables
        count = self._problem.shape[1]
        rows = (
            np.arange(count)
            if full or self._problem.transform.kind == "free"
            else (np.asarray(tables.rows))
        )
        key = (id(reduction), bool(full))
        if self._edge_key != key:
            inverse = reduction.synthesis()
            call_julia(
                self._jl.bind_edge_b,
                self._engine,
                None if inverse is None else np.asarray(inverse),
                np.asarray(rows, dtype=np.int64),
            )
            self._edge_key = key
        nodes = np.asarray(call_julia(self._jl.node_energy, self._engine))
        return _EdgeReading(
            faces=tables.faces @ nodes,
            clip=tables.clip @ nodes if full else None,
            totals=tables.totals @ nodes if full else None,
            columns=np.asarray(call_julia(self._jl.column_energy, self._engine)),
        )

    def _boundary_energies(self, state: object) -> BoundaryEnergies:
        """Read the monitor's energies from the engine's power, not a state copy.

        Parameters
        ----------
        state : object
            The accepted state handle.

        Returns
        -------
        solprop.monitor.BoundaryEnergies
            The reading.

        Raises
        ------
        BackendError
            If no monitor is attached.
        """
        del state
        monitor = self._monitor
        if monitor is None:
            raise BackendError("julia: no boundary monitor is attached")
        removed, self._split_removed = self._split_removed, 0.0
        power = np.asarray(call_julia(self._jl.state_power, self._engine))
        crossing = monitor.crossing
        return BoundaryEnergies(
            float(power.sum()),
            monitor.weights @ power,
            removed=removed,
            found=self._split_found,
            occupancy=None
            if crossing is None
            else np.asarray(call_julia(self._jl.occupancy, self._engine)),
        )

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Collect supported modal or spatial statistics through shared host functions.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Supported statistic names and any periodic subset.

        Returns
        -------
        dict
            Owning host arrays, with the leading singleton batch axis.

        Raises
        ------
        BackendError
            If a requested statistic is not supported or no state is loaded.
        """
        spatial = self._problem.transform.geometry is not None
        _refuse_stats(spec, spatial=spatial)
        if not spec.names:
            return {}
        if self._varying:
            self._weight = power_weight_at(self._problem, self._z)
            self._prepare_stages(self._z, 0.0)
        state = self._observed_field(self._require_state())[0]
        grid = self._require_grid()
        if spatial:
            delegated = dataclasses.replace(
                spec,
                names=tuple(n for n in spec.names if n not in _SPECIES_STATS),
                periodic=(),
            )
            values = collect_spatial(
                delegated,
                grid,
                self._problem.transform,
                state,
                z=self._z,
                flux=self._problem.flux,
                profiles=self._profiles,
                operator=self._problem.linear.separable,
            )
            if _SPECIES_STATS.intersection(spec.names):
                values.update(self._populations(spec, state))
            return {name: value[None] for name, value in values.items()}
        weighted = not np.all(self._weight == 1.0)
        owned = (
            (_POYNTING_STATISTICS if weighted else {"peak_intensity"})
            | _SPECIES_STATS
            | {"quadrature_error"}
        )
        delegated = dataclasses.replace(
            spec, names=tuple(n for n in spec.names if n not in owned), periodic=()
        )
        values = collect(
            delegated,
            grid,
            SavedPlane(z=self._z, field=state),
            modes=self._modes,
            profiles=self._profiles,
        )
        if weighted or "peak_intensity" in spec.names:
            axis = None
            if "peak_intensity" in spec.names:
                if self._modes is None:
                    raise BackendError("julia peak_intensity requires modes")
                axis = self._on_axis_synthesis(self._modes)
            values.update(self._poynting(spec, grid, state, axis))
        if "quadrature_error" in spec.names:
            values["quadrature_error"] = np.asarray(self._alarm)
        if _SPECIES_STATS.intersection(spec.names):
            values.update(self._populations(spec, state))
        return {name: value[None] for name, value in values.items()}

    def _poynting(
        self,
        spec: StatsSpec,
        grid: TimeGrid,
        state: np.ndarray,
        axis: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        """``stats._poynting_statistics``, its analytic signals formed by the engine.

        The transforms and the on-axis product are the engine's (``poynting``), the
        reductions over their short results the reference's own. Windowed statistics
        take the host function.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            The requested statistics.
        grid : solprop.grid.TimeGrid
            The run's grid.
        state : numpy.ndarray
            The state, (n_modes, n_pol, n_omega), the host copy the plane read.
        axis : numpy.ndarray or None
            The on-axis synthesis (2, n_modes), for ``peak_intensity``.

        Returns
        -------
        dict
            The requested Poynting statistics.
        """
        names = set(spec.names) & _POYNTING_STATISTICS
        if names & {"energy_window", "peak_power_window"}:
            return _poynting_statistics(spec, grid, state, self._weight, axis)
        values: dict[str, np.ndarray] = {}
        density = np.abs(state) ** 2 * self._weight
        if "omega0" in names:
            total = np.sum(density, axis=-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                values["omega0"] = np.sum(density * grid.omega, axis=-1) / total
        if "energy" in names:
            spectral = (
                density
                if grid.kind == "envelope"
                else density
                * np.concatenate(([1.0], np.full(grid.n_omega - 2, 4.0), [1.0]))
            )
            values["energy"] = (grid.dt / grid.n_t) * np.sum(spectral, axis=-1)
        if names - {"omega0", "energy"}:
            wanted = "peak_intensity" in names and axis is not None
            self._bind_poynting(axis)
            power, flux = call_julia(self._jl.poynting_b, self._engine, wanted)
            power = np.asarray(power, dtype=np.float64).T
            total_power = np.sum(power, axis=0)
            if "peak_power" in names:
                values["peak_power"] = np.max(power, axis=-1)
            if "peak_power_total" in names:
                values["peak_power_total"] = np.asarray(np.max(total_power))
            for name in ("fwhm_t", "fwhm_t_outer"):
                if name in names:
                    values[name] = _power_half_width(
                        grid, total_power, outermost=name.endswith("outer")
                    )
            if "peak_intensity" in names:
                if axis is None:
                    msg = (
                        "Poynting peak_intensity requires the on-axis synthesis matrix"
                    )
                    raise BackendError(msg)
                values["peak_intensity"] = np.asarray(
                    0.5 * EPS0 * C * np.max(np.asarray(flux))
                )
        return values

    def _bind_poynting(self, axis: np.ndarray | None) -> None:
        """Give the engine the power weight, when it is a new one, and the on-axis rows.

        Parameters
        ----------
        axis : numpy.ndarray or None
            The on-axis synthesis (2, n_modes); bound with the first weight.
        """
        if self._poynting_weight is self._weight:
            return
        call_julia(
            self._jl.bind_poynting_b,
            self._engine,
            np.ascontiguousarray(self._weight[:, 0, :].T),
            None if axis is None else np.ascontiguousarray(axis),
        )
        self._poynting_weight = self._weight

    def _populations(self, spec: StatsSpec, state: np.ndarray) -> dict[str, np.ndarray]:
        """Reduce fresh accepted-field populations at an observation."""
        from solprop.backends.populations import population_statistics  # noqa: PLC0415

        parts = np.zeros(6)
        if self._problem.flags.has_plasma:
            # The engine reads the state it holds, which is the one observed.
            del state
            raw = call_julia(self._jl.population_snapshot, self._engine, None)
            fractions, peaks = (np.asarray(raw[i], dtype=float)[None] for i in (0, 1))
            parts = np.asarray(raw[2], dtype=float)
        else:
            fractions = peaks = np.zeros((0, self._problem.transform.n_nodes))
        values = population_statistics(self._problem, self._z, fractions, peaks)
        if fractions.shape[0] == 1:
            values["peak_ionisation_rate"] = np.asarray(
                peaks[0, np.argmax(fractions[0])]
            )
        if any(name.startswith("plasma_") for name in spec.names):
            values.update(
                zip(
                    (
                        "plasma_undershoot",
                        "plasma_work_defect",
                        "plasma_boundary_level",
                    ),
                    (
                        np.asarray(value)
                        for value in plasma_integration_statistics(parts)
                    ),
                    strict=True,
                )
            )
        return {name: values[name] for name in spec.names if name in _SPECIES_STATS}

    def metadata(self) -> dict[str, object]:
        """Return runtime and thread provenance beside the shared step counters.

        Returns
        -------
        dict
            JSON-serializable execution metadata, without local paths.
        """
        self._check_open()
        assert self._engine is not None
        return (
            super().metadata()
            | self._provenance()
            | {"resident_source_evaluations": int(self._engine.calls)}
        )

    def _provenance(self) -> dict[str, object]:
        """Return the engine's fixed provenance, read from Julia once.

        A run records the metadata at every plane; these fields do not change.

        Returns
        -------
        dict
            The runtime, thread and plan fields of :meth:`metadata`.
        """
        if self._fixed is not None:
            return self._fixed
        assert self._engine is not None
        version, threads, fft, blas, library = self._jl.runtime()
        config = self._engine.config
        free = self._problem.transform.kind == "free"
        self._fixed = {
            "julia_version": str(version),
            "julia_source_digest": self._digest,
            "device": str(self._jl.device_name(self._engine)),
            "propagator_cache": (
                "schedule_run_mean"
                if self._low_storage is not None
                else "stage_expansion"
                if self._varying
                else "constant_table_exact_dz"
                if self._propagator_mode == "tabulate"
                else "disabled"
            ),
            "julia_threads": int(threads),
            "fftw_threads": int(fft),
            "modal_fftw_threads": int(config.fft_threads),
            "raman_fftw_threads": int(config.raman_fft_threads)
            if self._problem.flags.has_raman
            else None,
            "transverse_fftw_threads": int(config.transverse_fft_threads)
            if free
            else None,
            "blas_threads": int(blas),
            "blas_library": str(library),
            "plan_flag": self._plan_flag,
            "experimental": True,
        }
        return self._fixed

    def memory_ledger(self) -> dict[str, int]:
        """List owned numerical arrays, excluding opaque library plan internals.

        Returns
        -------
        dict
            Byte counts by Julia buffer name and host table.
        """
        self._check_open()
        result = self._jl.ledger(self._engine)
        return {str(name): int(result[name]) for name in result} | (
            {"host_power_weight": self._weight.nbytes} if self._weight.size else {}
        )

    def close(self) -> None:
        """Release this engine's Julia references; repeated calls are harmless."""
        if self._host_blas is not None:
            self._host_blas.restore_original_limits()
            self._host_blas = None
        self._state = None
        self._engine = None
        self._closed = True
        self._rows.clear()


def _hold_host_blas() -> _BlasLease | None:
    """Hold every host BLAS pool to one thread while the engine lives, as native does.

    NumPy's OpenBLAS spins after each small host product (the transverse edge's
    tables, the planes' statistics) on the cores the engine's pool waits on: on eight
    Xeon cores a radial filament's window took 4.26 s with its eight threads and 1.07 s
    with one. The lease is solprop's (``acquire_host_blas``), shared with native's.

    Returns
    -------
    solprop._process._BlasLease or None
        The lease, or None when no controllable BLAS is loaded.
    """
    import threadpoolctl  # noqa: PLC0415 - loaded with the engine only

    from solprop._process import acquire_host_blas  # noqa: PLC0415

    libraries = threadpoolctl.ThreadpoolController().select(user_api="blas")
    if not libraries.lib_controllers:
        return None
    return acquire_host_blas(libraries.lib_controllers, 1)


def _splits(problem: KernelProblem) -> bool:
    """Return whether accepted steps apply the split-step temporal collar."""
    response = problem.responses
    return response.absorber_scheme == "split_step" and (
        response.temporal_absorber is not None
    )


def _responses_recycle(problem: KernelProblem) -> bool:
    """Decide whether accepted-step absorption preserves the last source.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        Constant modal or spatial contract.

    Returns
    -------
    bool
        Whether the source can be reused after acceptance.
    """
    response = problem.responses
    return (
        response.absorber_scheme != "split_step"
        or response.temporal_absorber is None
        or not np.any(response.temporal_absorber)
    )
