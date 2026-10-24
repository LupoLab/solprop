# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Adapt the existing step driver to the experimental Julia-owned state."""

import dataclasses
import logging
import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from jaxtyping import Complex

from solprop._helpers import PLAIN
from solprop.backends.driver import StepDriver
from solprop.backends.julia._runtime import (
    _blas_provider,
    call_julia,
    runtime,
    thread_limit,
)
from solprop.backends.multistep import source_absent
from solprop.backends.numpy_ref.passes import plasma_integration_statistics
from solprop.backends.spatial import collect_spatial, radial_absorber_matrix
from solprop.constants import EPS0
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.noise import NoiseRealisation
from solprop.output.schema import SavedPlane
from solprop.profiles import ZProfile
from solprop.stats import (
    _POYNTING_STATISTICS,
    StatsSpec,
    _poynting_statistics,
    collect,
    default_spec,
)
from solprop.stepper import LowStorageRungeKutta, Multistep
from solprop.stepper.tableau import DOPRI5
from solprop.translate.chebyshev import PiecewiseChebyshev
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import SCHEMA_VERSION, KernelProblem
from solprop.translate.stage import scalars, terms

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
        If a requested leaf is outside the prototype's scope.
    """
    response, transform, linear, stepper = (
        problem.responses,
        problem.transform,
        problem.linear,
        problem.stepper,
    )
    reasons = {
        "kernel schema (requires 34)": problem.schema_version != 34,
        "envelope plasma": problem.grid_kind == "envelope" and problem.flags.has_plasma,
        "resolved modal or radial envelopes": problem.grid_kind == "envelope"
        and transform.kind not in {"free", "mode_averaged"},
        "envelope harmonics": problem.grid_kind == "envelope"
        and response.kerr_coefficients is not None
        and response.kerr_coefficients.harmonic_factor != 0,
        "single batch and scalar polarisation": problem.shape[0] != 1
        or problem.shape[2] != 1
        or transform.n_pol != 1,
        "planar or general modal transforms": (
            transform.geometry is not None
            and transform.geometry.kind not in {"radial", "free"}
        )
        or transform.modal_tables is not None,
        "coupled flow": linear.coupled_flow is not None,
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
        "separable or regional responses": (
            response.separable is not None and transform.kind != "free"
        )
        or response.regional_responses is not None,
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
            raise BackendError(f"julia prototype does not support {reason}")
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
        raise BackendError(
            f"julia prototype does not support statistics {sorted(missing)}"
        )


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
    chunk = min(transform.node_chunk, transform.n_nodes)
    temporal_columns = chunk if transform.kind == "free" else shape[1]
    # Short one-column transforms lose to task startup. The Xeon A/B sweep
    # retained serial transforms below 16k real-side samples, a four-thread cap
    # above it, and an independent cap for the doubled Raman convolution.
    fft_default = 1 if grids.n_to * temporal_columns < 16384 else min(threads, 4)
    raman_default = 1 if 2 * grids.n_to * chunk < 16384 else min(threads, 4)
    fft_threads = thread_limit("SOLPROP_FFTW_THREADS", fft_default)
    blas_default = (
        1
        if _blas_provider() == "accelerate" or transform.kind == "mode_averaged"
        else min(threads, 4)
    )
    config: dict[str, object] = {
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
        "atol": control.atol,
        "cache_propagators": transform.kind != "free"
        and problem.stepper.propagator_mode != "recompute",
        "threads": threads,
        "fft_threads": fft_threads,
        "raman_fft_threads": thread_limit(
            "SOLPROP_JULIA_RAMAN_FFT_THREADS", raman_default
        ),
        "transverse_fft_threads": thread_limit(
            "SOLPROP_TRANSVERSE_FFTW_THREADS", fft_threads
        ),
        "blas_threads": thread_limit(
            "SOLPROP_BLAS_THREADS",
            blas_default,
        ),
        "chunk": transform.node_chunk,
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
                    channel.ionisation_potential,
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
            raman_factor=EPS0 * float((raman.density or responses.density)(0.0)),
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
        Supported modal or spatial schema-34 problem, in SI units and complex128.
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
    helpers = PLAIN
    propagator_default = "tabulate"

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
        if int(self._jl.SCHEMA_VERSION) != SCHEMA_VERSION:
            raise BackendError("julia kernel schema does not match solprop")
        self._threads = thread_limit("SOLPROP_JULIA_THREADS")
        if not 1 <= self._threads <= int(self._jl.runtime()[1]):
            raise BackendError("julia thread count must fit the initialized runtime")
        self._engine = call_julia(
            self._jl.create, _configuration(problem, self._threads)
        )
        self._varying = _varying(problem)
        if self._varying:
            self._prepare_stages(0.0, 0.0)
        self._plan_flag = str(getattr(self._engine.config, "plan_flag", "estimate"))
        self._weight = (
            np.empty(0)
            if problem.transform.geometry is not None
            else power_weight_at(problem, 0.0)
        )
        self._fsal = False
        self._recycle = _responses_recycle(problem)
        self._alarm = math.nan
        self._alarm_reported = False
        if isinstance(problem.stepper.integrator, LowStorageRungeKutta):
            from solprop.backends.low_storage import LowStorageMarch  # noqa: PLC0415

            self._low_storage = LowStorageMarch(self, problem.stepper.integrator)

        if isinstance(problem.stepper.integrator, Multistep):
            from solprop.backends.multistep import AdamsMarch  # noqa: PLC0415

            self._adams = AdamsMarch(self, problem.stepper.integrator)

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
        call_julia(self._jl.load_b, self._engine, state[0, :, 0, :].T.copy())
        self._fsal = False
        self._alarm = math.nan
        return self._engine

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
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

        Raises
        ------
        BackendError
            If noise is supplied or the field is nonfinite.
        """
        if noise is not None or not np.all(np.isfinite(field)):
            raise BackendError("julia requires finite initial fields without noise")
        super().set_initial(field, noise, z=z)
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
        distinct = {
            offset: scalars(problem, z, z + offset, helpers=PLAIN)
            for offset in dict.fromkeys(offsets)
        }
        bundles = [distinct[offset] for offset in offsets]
        kerr = problem.responses.kerr_coefficients
        material = np.array(
            [
                [
                    0.0 if kerr is None else kerr.coefficient * b.kerr_density,
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
        self._alarm = float(call_julia(self._jl.commit_b, self._engine, dz, nogil=True))
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
        values = np.asarray(call_julia(self._jl.snapshot, state), dtype=np.complex128)
        return values.T.copy()[None, :, None, :]

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
                profiles=self._profiles,
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
            values.update(_poynting_statistics(spec, grid, state, self._weight, axis))
        if "quadrature_error" in spec.names:
            values["quadrature_error"] = np.asarray(self._alarm)
        if _SPECIES_STATS.intersection(spec.names):
            values.update(self._populations(spec, state))
        return {name: value[None] for name, value in values.items()}

    def _populations(self, spec: StatsSpec, state: np.ndarray) -> dict[str, np.ndarray]:
        """Reduce fresh accepted-field populations at an observation."""
        from solprop.backends.populations import population_statistics  # noqa: PLC0415

        parts = np.zeros(6)
        if self._problem.flags.has_plasma:
            raw = call_julia(
                self._jl.population_snapshot, self._engine, state[:, 0, :].T.copy()
            )
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
        version, threads, fft, blas, library = self._jl.runtime()
        accelerate = "Accelerate" in str(library)
        requested_blas = int(self._engine.config.blas_threads)
        return super().metadata() | {
            "julia_version": str(version),
            "plasma_channels": len(self._problem.responses.plasma_channels),
            "raman_terms": len(self._problem.responses.raman_terms),
            "raman_kernel_transforms": int(bool(self._problem.responses.raman_terms)),
            "resident_source_evaluations": int(self._engine.calls),
            "propagator_cache": (
                "schedule_run_mean"
                if self._low_storage is not None
                else "stage_expansion"
                if self._varying
                else "constant_table_exact_dz"
                if self._propagator_mode == "tabulate"
                else "disabled"
            ),
            "fft_provider": str(self._jl.FFTW.get_provider()),
            "parallel_fft_scale": bool(self._engine.config.parallel_fft_scale),
            "real_longitudinal_root": bool(self._engine.config.real_root),
            "parallel_copy": bool(self._engine.config.parallel_copy)
            if self._problem.transform.kind == "free"
            else None,
            "spatial_fft_layout": "strided"
            if self._problem.transform.kind == "free"
            else None,
            "julia_threads": int(threads),
            "fftw_threads": int(fft),
            "modal_fftw_threads": int(self._engine.config.fft_threads),
            "raman_fftw_threads": int(self._engine.config.raman_fft_threads)
            if self._problem.flags.has_raman
            else None,
            "transverse_fftw_threads": int(self._engine.config.transverse_fft_threads)
            if self._problem.transform.kind == "free"
            else None,
            "blas_threads": (1 if requested_blas == 1 else None)
            if accelerate
            else int(blas),
            "blas_thread_control": "single_or_managed" if accelerate else "fixed",
            "blas_library": str(library),
            "plan_flag": self._plan_flag,
            "wisdom_loaded": False,
            "statistics_execution": "julia_populations_shared_host_other",
            "schema_version": 34,
            "experimental": True,
        }

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
        self._state = None
        self._engine = None
        self._closed = True
        self._rows.clear()


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
