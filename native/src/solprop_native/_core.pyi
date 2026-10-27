# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import overload

import numpy as np
import numpy.typing as npt

class State:
    def __init__(self, problem: dict[str, object]) -> None: ...
    def set_initial(
        self,
        field: npt.NDArray[np.complex128],
        noise: npt.NDArray[np.complex128],
    ) -> None: ...
    def trial_step(
        self,
        dz: float,
        phi_delta: npt.NDArray[np.float64],
        phi_at: npt.NDArray[np.float64],
        prefactor: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
        taper: npt.NDArray[np.float64],
        preionisation: npt.NDArray[np.float64],
        raman_rates: npt.NDArray[np.float64],
        material: dict[str, npt.NDArray[np.float64] | int] | None = None,
        prepare_only: bool = False,
    ) -> npt.NDArray[np.float64]: ...
    def prepare_raman(
        self,
        rates: npt.NDArray[np.float64],
        densities: npt.NDArray[np.float64] | None = None,
    ) -> None: ...
    def set_flow_actions(
        self,
        forward: npt.NDArray[np.complex128],
        inverse: npt.NDArray[np.complex128],
        identity: npt.NDArray[np.bool_],
    ) -> None: ...
    def form_flow_actions(
        self, section: int, dz: float, fractions: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def flow_actions_to_host(self, inverse: bool) -> npt.NDArray[np.complex128]: ...
    def weighted_error(
        self, scale: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    def last_stage_norms(self) -> npt.NDArray[np.float64]: ...
    def spatial_statistics(
        self, flags: int, intensity_factor: float, z: float
    ) -> npt.NDArray[np.float64]: ...
    def aperture_statistics(
        self,
        flags: int,
        apertures: npt.NDArray[np.float64],
        profile_code: int,
        exponent: int,
        regions: npt.NDArray[np.float64],
        intensity_factor: float,
    ) -> npt.NDArray[np.float64]: ...
    def statistics(
        self,
        flags: int,
        windows: npt.NDArray[np.float64],
        on_axis: npt.NDArray[np.float64],
        density: float,
        taper: float,
        preionisation: float,
        intensity_factor: float,
        channel_preionisation: npt.NDArray[np.float64] | None = None,
        power_weight: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.float64]: ...
    def adams_source(
        self,
        input: npt.NDArray[np.complex128],
        phi_delta: npt.NDArray[np.float64],
        phi_at: npt.NDArray[np.float64],
        prefactor: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
        taper: npt.NDArray[np.float64],
        preionisation: npt.NDArray[np.float64],
        raman_rates: npt.NDArray[np.float64],
        material: dict[str, npt.NDArray[np.float64] | int],
    ) -> npt.NDArray[np.complex128]: ...
    def adopt(self, field: npt.NDArray[np.complex128], dz: float) -> None: ...
    def low_storage_step(
        self,
        dz: float,
        phi_delta: npt.NDArray[np.float64],
        phi_at: npt.NDArray[np.float64],
        prefactor: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
        taper: npt.NDArray[np.float64],
        preionisation: npt.NDArray[np.float64],
        raman_rates: npt.NDArray[np.float64],
        material: dict[str, npt.NDArray[np.float64] | int],
        rebuild_table: bool,
    ) -> float: ...
    def hold_plane(self) -> None: ...
    def held_to_host(self) -> npt.NDArray[np.complex128]: ...
    def adams_allocate(
        self, slots: int, compensated: bool = False, interpolate: bool = True
    ) -> None: ...
    def adams_compensation_start(
        self, slot: int, rate: npt.NDArray[np.float64]
    ) -> None: ...
    def adams_start(
        self,
        slot: int,
        phi_delta: npt.NDArray[np.float64],
        phi_at: npt.NDArray[np.float64],
        prefactor: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
        taper: npt.NDArray[np.float64],
        preionisation: npt.NDArray[np.float64],
        raman_rates: npt.NDArray[np.float64],
        material: dict[str, npt.NDArray[np.float64] | int],
    ) -> None: ...
    def adams_magnitude(self) -> float: ...
    def adams_scale(
        self, rtol: float, atol: float, weights: npt.NDArray[np.float64]
    ) -> None: ...
    def adams_fused_trial(
        self,
        dz: float,
        slots: npt.NDArray[np.uint64],
        predictor: npt.NDArray[np.float64],
        corrector: npt.NDArray[np.float64],
        rows: npt.NDArray[np.float64],
        divisor: npt.NDArray[np.float64],
        phi_delta: npt.NDArray[np.float64],
        phi_at: npt.NDArray[np.float64],
        prefactor: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
        taper: npt.NDArray[np.float64],
        preionisation: npt.NDArray[np.float64],
        raman_rates: npt.NDArray[np.float64],
        material: dict[str, npt.NDArray[np.float64] | int],
    ) -> npt.NDArray[np.float64]: ...
    def adams_fused_accept(
        self,
        dz: float,
        keep: npt.NDArray[np.uint64],
        free: int,
        dense: npt.NDArray[np.uint64],
        interpolate: bool,
    ) -> None: ...
    def adams_exponential_trial(
        self,
        dz: float,
        slots: npt.NDArray[np.uint64],
        predictor: npt.NDArray[np.float64],
        corrector: npt.NDArray[np.float64],
        rows: npt.NDArray[np.float64],
        divisor: npt.NDArray[np.float64],
        phi_delta: npt.NDArray[np.float64],
        phi_at: npt.NDArray[np.float64],
        prefactor: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
        taper: npt.NDArray[np.float64],
        preionisation: npt.NDArray[np.float64],
        raman_rates: npt.NDArray[np.float64],
        material: dict[str, npt.NDArray[np.float64] | int],
        sample_phi_delta: npt.NDArray[np.float64],
        sample_phi_at: npt.NDArray[np.float64],
        sample_prefactor: npt.NDArray[np.float64],
        sample_density: npt.NDArray[np.float64],
        sample_taper: npt.NDArray[np.float64],
        sample_preionisation: npt.NDArray[np.float64],
        sample_raman_rates: npt.NDArray[np.float64],
        sample_material: dict[str, npt.NDArray[np.float64] | int],
        spans: npt.NDArray[np.float64],
        comp_predictor: npt.NDArray[np.float64],
        comp_corrector: npt.NDArray[np.float64],
        comp_rows: npt.NDArray[np.float64],
        rate: npt.NDArray[np.float64],
        carrier: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def adams_exponential_accept(
        self, dz: float, free: int, dense: npt.NDArray[np.uint64], interpolate: bool
    ) -> None: ...
    def adams_exponential_observe(
        self,
        xi: float,
        dz: float,
        coefficients: npt.NDArray[np.float64],
        dense_weight: npt.NDArray[np.float64],
        dense_theta: npt.NDArray[np.float64],
        carrier: npt.NDArray[np.float64],
    ) -> None: ...
    def adams_hold(self) -> None: ...
    def adams_restore(self) -> None: ...
    def adams_row_names(self) -> list[str]: ...
    def adams_rows_to_host(self, name: str) -> npt.NDArray[np.complex128]: ...
    def adams_rows_from_host(
        self, name: str, rows: npt.NDArray[np.complex128]
    ) -> None: ...
    def table_upload(
        self,
        kind: str,
        edges: npt.NDArray[np.float64],
        lower: float,
        upper: float,
        coefficients: npt.NDArray[np.complex128],
    ) -> None: ...
    def table_evaluate(
        self, slots: npt.NDArray[np.uint64], positions: npt.NDArray[np.float64]
    ) -> None: ...
    def table_rate(self, slot: int, z: float, before: bool) -> None: ...
    def table_terms_to_host(
        self, kind: str, slot: int
    ) -> npt.NDArray[np.complex128]: ...
    def phase_sum_to_host(
        self, weights: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]: ...
    def adams_observe(
        self,
        weights: npt.NDArray[np.float64],
        hermite: bool,
        phi_delta: npt.NDArray[np.float64],
        phi_at: npt.NDArray[np.float64],
        prefactor: npt.NDArray[np.float64],
        density: npt.NDArray[np.float64],
        taper: npt.NDArray[np.float64],
        preionisation: npt.NDArray[np.float64],
        raman_rates: npt.NDArray[np.float64],
        material: dict[str, npt.NDArray[np.float64] | int],
    ) -> None: ...
    def finish(self, dz: float) -> None: ...
    def promote_fsal(self) -> None: ...
    def clear_fsal(self) -> None: ...
    def field_to_host(self) -> npt.NDArray[np.complex128]: ...
    def stage_to_host(self, stage: int) -> npt.NDArray[np.complex128]: ...
    def projected_to_host(self, coarse: bool = False) -> npt.NDArray[np.complex128]: ...
    @property
    def dispersive_path(self) -> dict[str, bool]: ...
    def memory_ledger(self) -> dict[str, int]: ...
    @property
    def planning_seconds(self) -> float: ...
    @property
    def raman_planning_seconds(self) -> float: ...
    @property
    def raman_fft_provider(self) -> str: ...
    @property
    def raman_fft_tasks(self) -> dict[str, int]: ...
    @property
    def execution_sizing(self) -> dict[str, dict[str, int]]: ...
    @property
    def raman_kernel_transforms(self) -> int: ...
    @property
    def fsal_ready(self) -> bool: ...
    @property
    def rk_source_evaluations(self) -> int: ...
    @property
    def gemm_probe_error(self) -> float: ...
    @property
    def quadrature_error(self) -> float: ...
    def quadrature_peak(self) -> float: ...
    @property
    def synthesis_provider(self) -> str: ...
    @property
    def projection_provider(self) -> str: ...
    @property
    def gemm_shape_policy(self) -> str: ...
    @property
    def response_team_size(self) -> int: ...
    @property
    def closed(self) -> bool: ...
    def close(self) -> None: ...

def build_record() -> dict[str, str | bool | int | None]: ...
def fit_blas_threads(threads: int) -> str: ...
def initialize_runtime(
    threads: int,
    kernels: int = 0,
    fft: int = 0,
    raman_fft: int = 0,
    raman_tail_fft: int = 0,
    threaded_fft_min_samples: int = 0,
    band_tiles: int = -1,
) -> None: ...
def reference_fft3_pairs(
    values: npt.NDArray[np.complex128], warm: int, pairs: int
) -> list[float]: ...
def reference_stream_copies(elements: int, warm: int, repeats: int) -> list[float]: ...
def binary32_round_trip(
    values: npt.NDArray[np.complex128], real: bool
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]: ...
def combined_binary32(factors: list[float], exponent: int) -> float: ...
def set_blas_math_mode_selection(mode: str) -> None: ...
def blas_math_mode_settable() -> bool: ...
def shutdown_runtime() -> None: ...
@overload
def clenshaw_points(
    z: npt.NDArray[np.float64],
    edges: npt.NDArray[np.float64],
    lower: float,
    upper: float,
    c: npt.NDArray[np.float64],
    out: npt.NDArray[np.float64],
) -> int: ...
@overload
def clenshaw_points(
    z: npt.NDArray[np.float64],
    edges: npt.NDArray[np.float64],
    lower: float,
    upper: float,
    c: npt.NDArray[np.complex128],
    out: npt.NDArray[np.complex128],
) -> int: ...
def basis_rows(
    points: npt.NDArray[np.float64],
    upper: float,
    integrate: bool,
    rows: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
) -> None: ...

class BatchState:
    def __init__(self, members: list[State]) -> None: ...
    def accept(self, dz: float) -> npt.NDArray[np.float64]: ...
    def memory_ledger(self) -> dict[str, int]: ...
    @property
    def execution_sizing(self) -> dict[str, dict[str, int]]: ...
    def close(self) -> None: ...

def available_device_memory() -> int | None: ...

class EnvelopeConvergenceError(RuntimeError): ...
class GemmProbeError(RuntimeError): ...
