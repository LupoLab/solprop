# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt the scalar capillary contract to a resident CPU workspace."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
from jaxtyping import Complex, Float
from solprop._process import thread_budget, thread_count
from solprop.backends.driver import StepDriver
from solprop.backends.multistep import (
    AdamsMarch,
    compensation_tables,
    exponential_tables,
    source_absent,
)
from solprop.backends.numpy_ref.passes import plasma_integration_statistics
from solprop.backends.populations import population_statistics
from solprop.backends.spatial import fluence_budget, signed_fluence_widths
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.noise import NoiseRealisation
from solprop.profiles import ZProfile
from solprop.stats import (
    PLASMA_INTEGRATION_STATISTICS,
    PROFILE_STATISTICS,
    StatsSpec,
    profile_value,
)
from solprop.stepper.multistep import Multistep
from solprop.translate.linop import power_weight_at, rate_weights
from solprop.translate.problem import SCHEMA_VERSION, KernelProblem
from solprop.translate.spatial import axial_admittance
from solprop.translate.stage import StageScalars, StageTerms, terms

from . import (
    _core,  # pyright: ignore[reportMissingModuleSource]  # compiled extension
    build_info,
    check_build,
)
from ._helpers import CPUNATIVE

__all__ = ["CPUNativeBackend"]
_SCHEMA = 37
_FIELD_STATS = frozenset(
    {
        "omega0",
        "energy",
        "peak_power",
        "peak_power_total",
        "fwhm_t",
        "fwhm_t_outer",
        "peak_intensity",
    }
)
_RESPONSE_STATS = frozenset(
    {
        "quadrature_error",
        "electron_density",
        "peak_ionisation_rate",
        *PLASMA_INTEGRATION_STATISTICS,
    }
)
_SPATIAL_STATS = frozenset(
    {
        "energy",
        "on_axis_spectrum",
        "on_axis_intensity",
        "peak_intensity",
        "beam_radius",
        "fwhm_r",
    }
)


def _admit(problem: KernelProblem) -> None:
    """Refuse an unsupported capability before constructing the workspace."""
    check_build()

    def require(condition: bool, capability: str) -> None:
        if not condition:
            raise BackendError(f"cpunative does not support {capability}")

    require(
        problem.schema_version == SCHEMA_VERSION == _SCHEMA,
        f"kernel schema {problem.schema_version}; expected {_SCHEMA}",
    )
    require(not problem.precision.single, "single precision")
    require(problem.grid_kind == "field", "envelope propagation")
    require(problem.shape[0] == 1, "batch execution")
    require(problem.shape[2] == problem.transform.n_pol == 1, "vector polarisation")
    geometry = problem.transform.geometry
    require(
        geometry is None or getattr(geometry, "kind", None) == "radial",
        "Cartesian or planar spatial routes",
    )
    if geometry is not None:
        require(not np.any(geometry.absorber_rate), "radial spatial absorption")
        require(
            problem.flux is not None and problem.flux.material is None,
            "varying spatial plane measures",
        )
    require(problem.transform.modal_tables is None, "general modal guides")
    require(
        np.isrealobj(problem.transform.synthesis)
        and np.isrealobj(problem.transform.projection),
        "complex modal maps",
    )
    require(problem.transform.taper_scale is None, "varying radius")
    require(problem.linear.coupled_flow is None, "coupled operators")
    require(problem.tdse is None, "TDSE")
    require(
        problem.linear.phi_representation in ("constant", "expansion"),
        "tabulated linear operators",
    )
    require(
        problem.responses.prefactor_representation in ("constant", "expansion"),
        "tabulated prefactors",
    )
    require(not problem.stepper.interpolate_planes, "dense-output interpolation")
    require(problem.initial.noise_model != "modified", "modified noise")
    require(
        problem.stepper.controller.weights is None, "frequency-weighted error control"
    )
    method = problem.stepper.integrator
    require(
        method is None or isinstance(method, Multistep),
        "RK4IP or low-storage integration",
    )
    if isinstance(method, Multistep):
        require(method.formulation == "exponential", "alternative Adams formulations")
        require(method.carrier_frame == "off", "Adams carrier frames")
        require(method.order <= 7, "Adams orders above seven")
    else:
        require(
            problem.stepper.tableau.name == "DOPRI5",
            f"integrator {problem.stepper.tableau.name}",
        )
    responses = problem.responses
    kerr = responses.kerr_coefficients
    require(
        kerr is None
        or (kerr.cubic_form == "real" and not kerr.analytic_signal)
        or (
            kerr.cubic_form == "analytic"
            and kerr.analytic_signal
            and kerr.take_real
            and kerr.harmonic_factor == 0
        ),
        "this Kerr cubic form",
    )
    require(responses.dispersive_kerr is None, "dispersive Kerr")
    require(not responses.spectral_responses, "spectral responses")
    require(len(responses.plasma_channels) <= 1, "multiple plasma channels")
    require(len(responses.raman_terms) <= 1, "multiple molecular Raman terms")
    require(
        all(term.thg for term in responses.raman_terms), "cycle-averaged Raman drive"
    )
    names = set(problem.stats_spec.names) | set(problem.stats_spec.periodic)
    supported = (
        (_FIELD_STATS if geometry is None else _SPATIAL_STATS)
        | _RESPONSE_STATS
        | set(PROFILE_STATISTICS)
    )
    missing = sorted(
        name
        for name in names
        if name not in supported and not name.startswith("species_")
    )
    require(not missing, f"statistics {missing}")
    problem.validate()


def _stage(bundle: StageScalars) -> dict[str, object]:
    """Pack one stage's small scalar arrays."""
    return {
        "delta": np.ascontiguousarray(bundle.phi_delta, dtype=np.float64),
        "prefactor": np.ascontiguousarray(bundle.prefactor, dtype=np.float64),
        "raman_rates": np.ascontiguousarray(bundle.raman_rates, dtype=np.float64),
        "kerr_density": bundle.kerr_density,
        "plasma_density": bundle.plasma_densities[0]
        if bundle.plasma_densities
        else 0.0,
        "preionisation": bundle.plasma_preionisation[0]
        if bundle.plasma_preionisation
        else 0.0,
        "raman_density": bundle.raman_densities[0] if bundle.raman_densities else 0.0,
        "taper": bundle.taper_scale,
    }


def _payload(problem: KernelProblem, terms: StageTerms) -> dict[str, object]:
    """Copy immutable numerical tables; translation owns their physics."""
    responses, grids, transform = problem.responses, problem.grids, problem.transform
    plasma = responses.plasma_channels[0] if responses.plasma_channels else None
    raman = responses.raman_terms[0] if responses.raman_terms else None
    rate = None if plasma is None else plasma.rate
    kerr = responses.kerr_coefficients
    geometry = transform.geometry
    method = problem.stepper.integrator
    budget = thread_budget()
    threads = min(
        thread_count("SOLPROP_CPUNATIVE_THREADS") or budget.threads, budget.threads
    )
    planning = os.environ.get("SOLPROP_FFTW_PLAN", "measure").lower()
    if planning not in ("measure", "estimate"):
        raise BackendError("cpunative SOLPROP_FFTW_PLAN must be measure or estimate")
    zeros = np.empty(0, dtype=np.float64)
    result = {
        "profile": os.environ.get("SOLPROP_CPUNATIVE_PROFILE") == "1",
        "analytic_kerr": kerr is not None and kerr.analytic_signal,
        "modes": problem.shape[1],
        "nodes": transform.n_nodes,
        "chunk": min(transform.node_chunk, transform.n_nodes),
        "nt": grids.n_t,
        "nf": grids.n_to,
        "nw": grids.n_omega,
        "threads": threads,
        "slots": method.order + 2 if isinstance(method, Multistep) else 7,
        "plasma": plasma is not None,
        "band": responses.plasma_integral.policy == "band",
        "split": responses.absorber_scheme == "split_step"
        and responses.temporal_absorber is not None,
        "dt": grids.dt_fine,
        "kerr": 0.0
        if kerr is None
        else kerr.coefficient
        * (kerr.self_factor + kerr.conjugate_factor if kerr.analytic_signal else 1.0),
        "charge": 0.0 if plasma is None else plasma.charge_mass_ratio,
        "potential": 0.0 if plasma is None else plasma.ionisation_potential,
        "field_min": 0.0 if rate is None else rate.field_min,
        "field_max": 0.0 if rate is None else rate.field_max,
        "rate_max": 0.0 if rate is None else rate.rate_max,
        "log_min": 0.0 if rate is None else rate.log_field_min,
        "log_step": 1.0 if rate is None else rate.d_log_field,
        "raman_fraction": 0.0 if raman is None else raman.fraction,
        "epsilon0": EPS0,
        "rtol": problem.stepper.controller.rtol,
        "atol": problem.stepper.controller.atol,
        "phi": terms.phi,
        "prefactor": terms.prefactor,
        "synthesis": transform.synthesis,
        "projection": transform.projection,
        "coarse": zeros
        if transform.n_nodes == 1 or geometry is not None
        else transform.projection_coarse,
        "admittance": np.empty(0, dtype=np.complex128)
        if problem.flux is None
        else axial_admittance(problem.flux.base, grids.omega, slice(None)),
        "spatial_axis": np.empty(0, dtype=np.complex128)
        if geometry is None
        else geometry.on_axis,
        "spectral_weights": zeros if geometry is None else geometry.spectral_weights,
        "omega": grids.omega,
        "window": grids.t_window_fine,
        "window_derivative": grids.t_window_fine_derivative,
        "omega_window": grids.omega_window,
        "absorber": zeros
        if responses.temporal_absorber_fine is None
        else responses.temporal_absorber_fine,
        "absorber_coarse": zeros
        if responses.temporal_absorber is None
        else responses.temporal_absorber,
        "taps": responses.plasma_integral.taps,
        "rate": zeros if rate is None else rate.coefficients,
        "undamped": zeros if raman is None else raman.undamped,
        "tableau": problem.stepper.tableau.matrix,
        "solution": problem.stepper.tableau.solution_weights,
        "error": problem.stepper.tableau.error_weights,
        "crop": np.asarray(grids.crop_index, dtype=np.uintp),
        "active": np.asarray(grids.active_mask, dtype=np.uint8),
        "norm": np.asarray(grids.norm_mask, dtype=np.uint8),
        "planning": planning,
        "contraction": os.environ.get("SOLPROP_CPUNATIVE_CONTRACTION", "auto"),
    }
    return {
        name: np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
        for name, value in result.items()
    }


def memory_estimate(problem: KernelProblem) -> int:
    """Count persistent buffers and immutable tables without constructing plans."""
    if (
        problem.transform.geometry is not None
        and problem.transform.geometry.kind == "free"
    ):
        from .beam import memory_ledger

        return sum(memory_ledger(problem).values())
    _admit(problem)
    data = _payload(problem, terms(problem))
    return sum(_buffer_ledger(data, problem.stepper.tableau.n_stages).values())


def create_backend(
    problem: KernelProblem,
    *,
    grid: TimeGrid | None = None,
    modes: Sequence[Mode] | None = None,
    profiles: Mapping[str, ZProfile] | None = None,
) -> object:
    """Select the resident workspace for the translated geometry."""
    if (
        problem.transform.geometry is not None
        and problem.transform.geometry.kind == "free"
    ):
        from .beam import CartesianBackend

        return CartesianBackend(problem, grid=grid, modes=modes, profiles=profiles)
    return CPUNativeBackend(problem, grid=grid, modes=modes, profiles=profiles)


def _buffer_ledger(data: Mapping[str, object], stages: int) -> dict[str, int]:
    """Name each allocation using the same extents as the C++ constructor."""
    modes, nf, nt, nw = (cast(int, data[key]) for key in ("modes", "nf", "nt", "nw"))
    chunk, nodes, slots = (cast(int, data[key]) for key in ("chunk", "nodes", "slots"))
    size = modes * nw
    fine_bins = ((nf // 2 + 1) + 3) // 4 * 4
    raman_bins = (nf + 4) // 4 * 4
    streams = 3 if data["plasma"] else 1
    raman_rows = cast(np.ndarray, data["undamped"]).size // (2 * nf)
    complex_buffers = {
        **dict.fromkeys(
            ("state", "trial", "input", "source", "fsal", "endpoint"), size
        ),
        **dict.fromkeys(
            ("history", "history_state", "history_rate", "compensation"), size * slots
        ),
        "moments": size * (slots + 1),
        "stages": size * stages,
        "transform": modes * fine_bins,
        "projected_spectrum": streams * modes * fine_bins,
        "coarse_spectrum": streams * modes * fine_bins,
        "statistics_time": 2 * modes * nt,
        "statistics_nodes": 2 * chunk * nt
        if cast(np.ndarray, data["admittance"]).size
        else 0,
        "statistics_axis": 2 * nt if cast(np.ndarray, data["admittance"]).size else 0,
        "raman_frequency": chunk * raman_bins if raman_rows else 0,
        "raman_kernel": raman_bins if raman_rows else 0,
    }
    real_buffers = {
        **dict.fromkeys(
            (
                "node",
                "polarisation",
                "current",
                "edge",
                "rate",
                "fraction",
                "scan",
                "temporary",
            ),
            chunk * nf,
        ),
        "modal": modes * nf,
        "modal_quadrature": modes * nf if data["analytic_kerr"] else 0,
        "node_quadrature": chunk * nf if data["analytic_kerr"] else 0,
        "projected": streams * modes * nf,
        "coarse_projected": streams * modes * nf,
        "raman_time": 2 * chunk * nf if raman_rows else 0,
        "raman_kernel_time": 2 * nf if raman_rows else 0,
        "statistics_power": nt,
        "reductions": 8 * cast(int, data["threads"]),
        "last_rates": raman_rows,
        "node_populations": 8 * nodes,
        "chunk_populations": 8 * chunk,
    }
    return (
        {f"cpunative.{name}": 16 * count for name, count in complex_buffers.items()}
        | {f"cpunative.{name}": 8 * count for name, count in real_buffers.items()}
        | {
            f"cpunative.table.{name}": value.nbytes
            for name, value in data.items()
            if isinstance(value, np.ndarray)
        }
    )


class CPUNativeBackend(StepDriver[_core.State]):
    """Execute scalar field-grid capillaries in a resident C++ CPU workspace.

    Parameters
    ----------
    problem
        The validated double-precision kernel problem.
    grid, modes, profiles
        Host context for observations and profile statistics.
    """

    name = "cpunative"
    max_batch = 1
    helpers = CPUNATIVE
    far_field_statistics = frozenset()
    _workspace: _core.State | None
    _resident: _ResidentAdams | None

    def __init__(
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
    ) -> None:
        _admit(problem)
        super().__init__(problem, grid=grid, modes=modes, profiles=profiles)
        self._workspace = None
        configuration = _payload(problem, self._terms)
        self._contraction = cast(str, configuration["contraction"])
        self._planning = cast(str, configuration["planning"])
        self._ledger = _buffer_ledger(configuration, problem.stepper.tableau.n_stages)
        try:
            self._workspace = _core.State(configuration)
        except (ValueError, RuntimeError) as error:
            raise BackendError(f"cpunative construction failed: {error}") from error
        self._resident = None
        if isinstance(problem.stepper.integrator, Multistep):
            self._adams = AdamsMarch(self, problem.stepper.integrator)

    @property
    def _cpu(self) -> _core.State:
        if self._workspace is None:
            raise BackendError("cpunative workspace is closed")
        return self._workspace

    def _apply_mask(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> _core.State:
        try:
            self._cpu.load(np.ascontiguousarray(state, dtype=np.complex128))
        except ValueError as error:
            raise BackendError(
                f"cpunative initial field is invalid: {error}"
            ) from error
        return self._cpu

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
    ) -> None:
        """Load the initial spectrum and restart all integrator history."""
        if field.dtype != np.complex128 or field.shape != self._problem.shape:
            raise BackendError(
                "cpunative initial field must match the complex128 kernel shape"
            )
        super().set_initial(field, noise, z=z)
        if self._adams is not None:
            self._adams.reset()

    def _trial_step(self, z: float, dz: float) -> tuple[float, _core.State]:
        self._hoist(z)
        *stages, whole = self._stage_bundles(
            z,
            [*(z + float(c) * dz for c in self._problem.stepper.tableau.nodes), z + dz],
        )
        old, new, residual = self._cpu.trial(
            dz, [_stage(s) for s in stages], np.ascontiguousarray(whole.phi_delta)
        )
        self._rk_source_evaluations = self._cpu.evaluations
        divisor = self._problem.stepper.controller.scale(math.sqrt(old), math.sqrt(new))
        return math.sqrt(residual) / float(divisor), self._cpu

    def _finish(self, state: _core.State, dz: float) -> _core.State:
        self._cpu.accept(dz)
        return state

    def _observed_field(
        self, state: _core.State
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        return np.asarray(state.field()).reshape(self._problem.shape)

    def _march_to(self, target: float) -> None:
        if self._adams is None:
            super()._march_to(target)
        else:
            self._adams.march_to(target)

    def _adams_state(self) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        return self._observed_field(self._cpu)

    def _adams_engine(self, method: Multistep, compensation: str) -> object:
        if compensation != "transported":
            raise BackendError("cpunative requires transported Adams compensation")
        self._resident = _ResidentAdams(self, method)
        return self._resident

    def stats(self, spec: StatsSpec) -> dict[str, Float[np.ndarray, ...]]:
        """Return requested plane statistics with the protocol's leading batch axis."""
        self._require_state()
        values = {}
        geometry = self._problem.transform.geometry
        if geometry is not None and _SPATIAL_STATS.intersection(spec.names):
            packed = np.asarray(self._cpu.spatial_statistics(0.5 * EPS0 * C))
            nw, nodes = self._problem.shape[-1], self._problem.transform.n_nodes
            field_values = {
                "energy": packed[:1],
                "on_axis_intensity": packed[1:2],
                "peak_intensity": packed[2:3],
                "on_axis_spectrum": packed[4 : 4 + nw][None],
            }
            if {"beam_radius", "fwhm_r"}.intersection(spec.names):
                fluence = packed[4 + nw : 4 + nw + nodes]
                magnitudes = packed[4 + nw + nodes :]
                budget = fluence_budget(
                    magnitudes, self._problem.grids.n_t + self._problem.shape[1]
                )
                widths = signed_fluence_widths(
                    geometry, fluence, budget, float(packed[3])
                )
                field_values["beam_radius"] = np.asarray([widths[:2]])
                field_values["fwhm_r"] = np.asarray([widths[2]])
            values.update(
                {
                    name: value
                    for name, value in field_values.items()
                    if name in spec.names
                }
            )
        elif _FIELD_STATS.intersection(spec.names):
            modes = self._problem.shape[1]
            axis = (
                self._on_axis_synthesis(self._modes)
                if self._modes is not None
                else np.empty(0)
            )
            packed = np.asarray(
                self._cpu.statistics(
                    np.ascontiguousarray(power_weight_at(self._problem, self._z)),
                    np.ascontiguousarray(axis),
                    0.5 * EPS0 * C,
                )
            )
            field_values = {
                "omega0": packed[:modes].reshape(1, modes, 1),
                "energy": packed[modes : 2 * modes].reshape(1, modes, 1),
                "peak_power": packed[2 * modes : 3 * modes].reshape(1, modes),
                "peak_power_total": packed[3 * modes : 3 * modes + 1],
                "fwhm_t": packed[3 * modes + 1 : 3 * modes + 2],
                "fwhm_t_outer": packed[3 * modes + 2 : 3 * modes + 3],
                "peak_intensity": packed[3 * modes + 3 : 3 * modes + 4],
            }
            values.update(
                {
                    name: value
                    for name, value in field_values.items()
                    if name in spec.names
                }
            )
        populations = {
            name
            for name in spec.names
            if name.startswith("species_")
            or name in ("electron_density", "peak_ionisation_rate")
        }
        integration = set(spec.names).intersection(PLASMA_INTEGRATION_STATISTICS)
        if populations or integration:
            self._hoist(self._z)
            rows = np.asarray(
                self._cpu.populations(_stage(self._stage_scalars(self._z, self._z)))
            ).reshape(-1, 2)
            reduced = population_statistics(
                self._problem, self._z, rows[:, 0][None], rows[:, 1][None]
            )
            values.update({name: reduced[name][None] for name in populations})
            ratios: dict[str, float] = dict(
                zip(
                    PLASMA_INTEGRATION_STATISTICS,
                    plasma_integration_statistics(np.asarray(self._cpu.diagnostics())),
                    strict=True,
                )
            )
            values.update({name: np.asarray([ratios[name]]) for name in integration})
        for name in spec.names:
            if name in PROFILE_STATISTICS:
                if self._profiles is None or name not in self._profiles:
                    raise BackendError(f"cpunative requires the {name} profile")
                values[name] = np.asarray(
                    [profile_value(self._profiles[name], self._z)]
                )
            elif name == "quadrature_error":
                values[name] = np.asarray([self._cpu.quadrature])
        return values

    def metadata(self) -> dict[str, object]:
        """Return execution metadata and the extension's build provenance."""
        return super().metadata() | {
            "cpunative": build_info(),
            "threads": self._cpu.threads,
            "planning_seconds": self._cpu.planning_seconds,
            "contraction": self._cpu.contraction,
            "contraction_requested": self._contraction,
            "fft_planning": self._planning,
            "source_evaluations": self._cpu.evaluations,
            "response_work_seconds": dict(
                zip(
                    (
                        "kerr",
                        "raman_window",
                        "rate",
                        "exposure_scan",
                        "fraction",
                        "current_scan",
                        "current_assembly",
                    ),
                    self._cpu.response_timings,
                    strict=True,
                )
            ),
            "kernel_seconds": dict(
                zip(
                    (
                        "inverse_fft",
                        "raman_prepare",
                        "synthesis_response",
                        "projection",
                        "forward_fft",
                        "absorber",
                        "crop",
                        "quadrature",
                        "statistics",
                        "trial",
                    ),
                    self._cpu.timings,
                    strict=True,
                )
            ),
        }

    def memory_ledger(self) -> dict[str, int]:
        """Return bytes owned by the resident numerical workspace."""
        return {} if self._closed else dict(self._ledger)

    def available_device_memory(self) -> None:
        """Return None: this backend owns CPU memory only."""
        return None

    def close(self) -> None:
        """Release owned buffers and plans; repeated closure is harmless."""
        self._state = None
        self._noise = None
        self._held = None
        self._workspace = None
        self._closed = True


class _ResidentAdams:
    """Resident exponential Adams history controlled by the shared Python march."""

    dense = None

    def __init__(self, backend: CPUNativeBackend, method: Multistep) -> None:
        self.backend = backend
        self.method = method
        self.compensated = (
            backend._problem.linear.phi_representation == "expansion"
            and not source_absent(backend._problem)
        )
        self._trial = None

    @property
    def n_slots(self) -> int:
        return self.method.order + 2

    def reset(self) -> None:
        self._trial = None

    def snapshot_rows(self) -> dict[str, np.ndarray]:
        shape = (self.n_slots, *self.backend._problem.shape)
        names = ("store", "states", "rates") if self.compensated else ("store",)
        return {
            name: np.asarray(self.backend._cpu.history(name)).reshape(shape)
            for name in names
        }

    def load_rows(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        names = {"store", "states", "rates"} if self.compensated else {"store"}
        expected = (self.n_slots, *shape)
        if set(rows) != names or any(
            value.dtype != np.complex128
            or value.shape != expected
            or not np.isfinite(value).all()
            for value in rows.values()
        ):
            raise BackendError(
                "cpunative Adams snapshot rows do not match the resident history"
            )
        for name, value in rows.items():
            self.backend._cpu.load_history(name, np.ascontiguousarray(value))
        self.reset()

    def value(self, slot: int) -> None:
        return None

    def _rate(self, z: float, *, before: bool = False) -> np.ndarray:
        return np.ascontiguousarray(
            rate_weights(
                self.backend._problem.linear, z, before=before, helpers=CPUNATIVE
            )
        )

    def start(self, z: float, *, stored: bool) -> None:
        backend = self.backend
        backend._hoist(z)
        backend._cpu.adams_start(_stage(backend._stage_scalars(z, z)), self._rate(z))

    def exponential_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: np.ndarray,
        order: int,
        orders: list[int],
    ) -> np.ndarray:
        backend = self.backend
        backend._hoist(z)
        predictor, corrector, defects = exponential_tables(
            nodes, order, orders, helpers=CPUNATIVE
        )
        empty = np.empty(0)
        spans = carried = endpoint = comp_defects = empty
        rate = self._rate(z + dz, before=True)
        if self.compensated:
            spans, carried, endpoint, comp_defects = compensation_tables(
                nodes, order, orders, helpers=CPUNATIVE
            )
        width = corrector.shape[0]
        table = {
            "order": order,
            "width": width,
            "count": defects.shape[1] - width,
            "rows": len(orders),
            "slots": np.asarray(slots[: max(width - 1, order)], dtype=np.uintp),
            "predictor": predictor,
            "corrector": corrector,
            "defects": defects,
            "spans": spans,
            "carried": carried,
            "endpoint": endpoint,
            "comp_defects": comp_defects,
            "rate": rate,
        }
        self._trial = (dz, rate)
        return np.asarray(
            backend._cpu.adams_trial(
                dz,
                _stage(backend._stage_scalars(z, z + dz)),
                _stage(backend._stage_scalars(z + dz, z + dz)),
                table,
            )
        )

    def exponential_accept(
        self, keep: np.ndarray, free: int, *, interpolate: bool
    ) -> None:
        assert self._trial is not None
        assert not interpolate
        dz, rate = self._trial
        self.backend._cpu.adams_accept(dz, free, rate)
        self.backend._state = self.backend._cpu
        self._trial = None
