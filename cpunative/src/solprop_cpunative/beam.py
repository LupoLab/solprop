# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Execute scalar Cartesian Kerr envelopes with resident low-storage RK4IP."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np
from jaxtyping import Complex, Float
from solprop._process import thread_budget, thread_count
from solprop.backends.driver import StepDriver
from solprop.backends.low_storage import LowStorageMarch, schedule_key, table_steps
from solprop.backends.spatial import fluence_budget, signed_fluence_widths
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.profiles import ZProfile
from solprop.stats import StatsSpec
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.translate.problem import SCHEMA_VERSION, KernelProblem

from . import (
    _core,  # pyright: ignore[reportMissingModuleSource]  # compiled extension
    build_info,
    check_build,
)
from ._helpers import CPUNATIVE

__all__ = ["CartesianBackend"]

_STATISTICS = frozenset(
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
    check_build()

    def require(condition: bool, capability: str) -> None:
        if not condition:
            raise BackendError(
                f"cpunative Cartesian experiment does not support {capability}"
            )

    geometry = problem.transform.geometry
    response = problem.responses
    kerr = response.kerr_coefficients
    require(problem.schema_version == SCHEMA_VERSION == 37, "this kernel schema")
    require(not problem.precision.single, "single precision")
    require(geometry is not None and geometry.kind == "free", "this geometry")
    require(problem.shape[0] == problem.shape[2] == 1, "batch or vector fields")
    require(problem.grid_kind == "envelope", "real-field Cartesian propagation")
    require(problem.grids.n_t == problem.grids.n_to, "temporal oversampling")
    require(problem.linear.separable is not None, "nonseparable operators")
    require(
        problem.flux is not None and problem.flux.material is None,
        "varying plane measures",
    )
    require(
        geometry is not None and not np.any(geometry.absorber_rate),
        "spatial absorption",
    )
    require(response.absorber_scheme == "split_step", "in-stage temporal absorption")
    require(
        kerr is None
        or (
            kerr.cubic_form == "analytic"
            and not kerr.take_real
            and kerr.harmonic_factor == 0
        ),
        "this Kerr form",
    )
    require(
        not response.plasma_channels
        and not response.raman_terms
        and response.dispersive_kerr is None
        and not response.spectral_responses
        and problem.tdse is None,
        "responses other than instantaneous Kerr",
    )
    require(
        isinstance(problem.stepper.integrator, LowStorageRungeKutta),
        "integrators other than fixed RK4IP",
    )
    require(problem.stepper.schedule is not None, "adaptive schedules")
    require(
        problem.stepper.propagator_mode in ("auto", "tabulate"),
        "recomputed propagators",
    )
    require(not problem.stepper.interpolate_planes, "dense interpolation")
    require(problem.initial.noise_model != "modified", "modified noise")
    missing = (
        set(problem.stats_spec.names) | set(problem.stats_spec.periodic)
    ) - _STATISTICS
    require(not missing, f"statistics {sorted(missing)}")
    problem.validate()


def _payload(problem: KernelProblem) -> dict[str, object]:
    geometry = problem.transform.geometry
    operator = problem.linear.separable
    assert geometry is not None
    assert operator is not None
    grid, response = problem.grids, problem.responses
    kerr = response.kerr_coefficients
    budget = thread_budget().threads
    planning = os.environ.get("SOLPROP_FFTW_PLAN", "measure").lower()
    if planning not in ("measure", "estimate"):
        raise BackendError("cpunative SOLPROP_FFTW_PLAN must be measure or estimate")
    fft_schedule = os.environ.get("SOLPROP_CPUNATIVE_CARTESIAN_FFT", "auto")
    if fft_schedule not in ("auto", "direct", "split"):
        raise BackendError(
            "cpunative Cartesian FFT schedule must be auto, direct or split"
        )
    data = dict(
        nx=geometry.shape[0],
        ny=geometry.shape[1],
        nt=grid.n_t,
        threads=min(thread_count("SOLPROP_CPUNATIVE_THREADS") or budget, budget),
        dt=grid.dt,
        kerr=0.0
        if kerr is None
        else kerr.coefficient * (kerr.self_factor + kerr.conjugate_factor),
        light_speed=C,
        evanescent_rate=operator.evanescent_rate,
        split=response.temporal_absorber is not None,
        profile=os.environ.get("SOLPROP_CPUNATIVE_PROFILE") == "1",
        planning=planning,
        fft_schedule=fft_schedule,
        k_squared=operator.k_squared,
        k_squared_low=operator.k_squared_low,
        numerator=operator.prefactor_numerator,
        clamped=operator.prefactor_clamped,
        axis=geometry.on_axis,
        kperp=operator.kperp_sq,
        kperp_low=operator.kperp_sq_low,
        frame=operator.frame,
        spectral_rate=operator.spectral_rate,
        transverse_rate=operator.kperp_rate,
        omega=grid.omega,
        window=grid.t_window_fine,
        omega_window=grid.omega_window,
        absorber=np.empty(0)
        if response.temporal_absorber is None
        else response.temporal_absorber,
        spectral_weights=geometry.spectral_weights,
        active=np.asarray(grid.active_mask, dtype=np.uint8),
        norm=np.asarray(grid.norm_mask, dtype=np.uint8),
    )
    return {
        name: np.ascontiguousarray(value) if isinstance(value, np.ndarray) else value
        for name, value in data.items()
    }


def memory_ledger(problem: KernelProblem) -> dict[str, int]:
    _admit(problem)
    payload = _payload(problem)
    return {
        **dict.fromkeys(
            (
                f"cpunative.{name}"
                for name in ("state", "accumulator", "stage", "source", "flow")
            ),
            16 * int(np.prod(problem.shape)),
        ),
        "cpunative.axis": 32 * problem.grids.n_t,
        "cpunative.absorber_factor": 8 * problem.grids.n_t,
        **{
            f"cpunative.table.{name}": value.nbytes
            for name, value in payload.items()
            if isinstance(value, np.ndarray)
        },
    }


class CartesianBackend(StepDriver[_core.BeamState]):
    """Run the Cartesian Kerr experiment through the shared fixed-step march."""

    name = "cpunative"
    helpers = CPUNATIVE
    max_batch = 1
    far_field_statistics = frozenset()
    _workspace: _core.BeamState | None

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
        self._ledger = memory_ledger(problem)
        try:
            self._workspace = _core.BeamState(_payload(problem))
        except (ValueError, RuntimeError) as error:
            raise BackendError(
                f"cpunative Cartesian construction failed: {error}"
            ) from error
        method = problem.stepper.integrator
        assert isinstance(method, LowStorageRungeKutta)
        self._low_storage = LowStorageMarch(self, method)

    @property
    def _cpu(self) -> _core.BeamState:
        if self._workspace is None:
            raise BackendError("cpunative Cartesian workspace is closed")
        return self._workspace

    def _apply_mask(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> _core.BeamState:
        if state.dtype != np.complex128:
            raise BackendError("cpunative requires complex128 state")
        self._cpu.load(np.ascontiguousarray(state))
        return self._cpu

    def _trial_step(self, z: float, dz: float) -> tuple[float, _core.BeamState]:
        raise BackendError("cpunative Cartesian requires fixed RK4IP")

    def _finish(self, state: _core.BeamState, dz: float) -> _core.BeamState:
        state.finish(dz)
        return state

    def _observed_field(
        self, state: _core.BeamState
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        return np.asarray(state.field()).reshape(self._problem.shape)

    def _low_storage_engine(self, method: LowStorageRungeKutta) -> object:
        return _Engine(self)

    def stats(self, spec: StatsSpec) -> dict[str, Float[np.ndarray, ...]]:
        self._require_state()
        if not spec.names:
            return {}
        values = np.asarray(self._cpu.statistics(0.5 * EPS0 * C))
        nt, nodes = self._problem.grids.n_t, self._problem.transform.n_nodes
        result = dict(
            energy=values[:1],
            on_axis_intensity=values[1:2],
            peak_intensity=values[2:3],
            on_axis_spectrum=values[4 : 4 + nt][None],
        )
        if {"beam_radius", "fwhm_r"}.intersection(spec.names):
            geometry = self._problem.transform.geometry
            assert geometry is not None
            fluence, magnitude = (
                values[4 + nt : 4 + nt + nodes],
                values[4 + nt + nodes :],
            )
            widths = signed_fluence_widths(
                geometry,
                fluence,
                fluence_budget(magnitude, nt + nodes),
                float(values[3]),
            )
            result["beam_radius"] = np.asarray([widths[:2]])
            result["fwhm_r"] = np.asarray([widths[2]])
        return {name: result[name] for name in spec.names}

    def metadata(self) -> dict[str, object]:
        return super().metadata() | dict(
            cpunative=build_info(),
            threads=self._cpu.threads,
            source_evaluations=self._cpu.evaluations,
            planning_seconds=self._cpu.planning_seconds,
            fft_schedule=self._cpu.fft_schedule,
            kernel_seconds=dict(
                zip(
                    (
                        "inverse_fft",
                        "kerr",
                        "forward_fft",
                        "crop",
                        "absorber",
                        "statistics",
                    ),
                    self._cpu.timings,
                    strict=True,
                )
            ),
        )

    def memory_ledger(self) -> dict[str, int]:
        return {} if self._closed else dict(self._ledger)

    def available_device_memory(self) -> None:
        return None

    def close(self) -> None:
        self._state = self._workspace = None
        self._noise = self._held = None
        self._closed = True


class _Engine:
    atomic = True
    stiffness = float("nan")

    def __init__(self, backend: CartesianBackend) -> None:
        self.backend = backend
        self.schedule = backend._problem.stepper.schedule
        assert self.schedule is not None
        self.keys = table_steps(self.schedule)

    def step(self, z: float, dz: float) -> float:
        backend = self.backend
        assert self.schedule is not None
        bundles = backend._stage_bundles(z, [z, z + 0.5 * dz, z + dz])
        density = np.array([bundle.kerr_density for bundle in bundles])
        return backend._cpu.step(
            dz, schedule_key(self.keys, self.schedule.edges, z, dz), density
        )

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass
