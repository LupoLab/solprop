# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import Any, overload

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
    def rk_trial(
        self, z: float, dz: float, nodes: list[float], prepare_only: bool = False
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
    def spatial_statistics(
        self, flags: int, intensity_factor: float, z: float
    ) -> npt.NDArray[np.float64]: ...
    def plane_fluence_trace(
        self,
        center_x: float,
        center_y: float,
        dx: float,
        dy: float,
        extent: float,
        spacing: float,
        n_radii: int,
        cosines: npt.NDArray[np.float64],
        sines: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def axis_spectrum(
        self, matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.complex128]: ...
    def aperture_statistics(
        self,
        flags: int,
        apertures: npt.NDArray[np.float64],
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
    def adams_stage_tables(self, tables: dict[str, object]) -> None: ...
    def has_stage_tables(self) -> bool: ...
    def adams_exponential_trial_at(
        self,
        z: float,
        dz: float,
        nodes: npt.NDArray[np.float64],
        slots: npt.NDArray[np.uint64],
        order: int,
        orders: list[int],
        compensated: bool,
        divisor: npt.NDArray[np.float64],
        carrier: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def adams_stage_scalars(self, z0: float, z1: float) -> dict[str, list[float]]: ...
    def adams_march_configure(
        self,
        policy: dict[str, object],
        fused: bool,
        compensated: bool,
        interpolate: bool,
        steps: npt.NDArray[np.float64],
        edges: npt.NDArray[np.float64],
    ) -> None: ...
    def adams_fused_trial_at(
        self,
        z: float,
        dz: float,
        nodes: npt.NDArray[np.float64],
        slots: npt.NDArray[np.uint64],
        order: int,
        orders: list[int],
        divisor: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]: ...
    def adams_march_run(
        self, state: dict[str, object], bounds: dict[str, object]
    ) -> dict[str, Any]: ...
    def adams_march_fault(self, stage: str, after: int, category: str) -> None: ...
    def rk_march_configure(
        self,
        policy: dict[str, object],
        nodes: list[float],
        rtol: float,
        atol: float,
        steps: npt.NDArray[np.float64],
        edges: npt.NDArray[np.float64],
    ) -> None: ...
    def rk_march_run(
        self, state: dict[str, object], bounds: dict[str, object]
    ) -> dict[str, Any]: ...
    def rk_weights_configure(self, weights: list[float]) -> None: ...
    def rk_weighted_score(
        self, partials: npt.NDArray[np.float64], rtol: float, atol: float
    ) -> float: ...
    def flow_guard_configure(
        self,
        row_sum: list[float],
        condition_bound: list[float],
        taylor_scale: float,
        max_squarings: int,
        allowance: float,
    ) -> None: ...
    def adams_stage_rate(self, z: float) -> list[float]: ...
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
    def table_slots(
        self, phi: bool, prefactor: bool, slots: int, rate: bool
    ) -> None: ...
    def table_load(self, positions: list[float]) -> list[int]: ...
    def table_rate_at(self, z: float, before: bool) -> None: ...
    def table_evaluations(self) -> int: ...
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
    def set_absorber_quiet(self, quiet: bool) -> None: ...
    def bind_monitor(
        self,
        weights: npt.NDArray[np.float64],
        column_weights: npt.NDArray[np.float64],
    ) -> None: ...
    def unbind_monitor(self) -> None: ...
    def bind_occupancy(self, buckets: npt.NDArray[np.uint8], count: int) -> None: ...
    def monitor_reading(self) -> npt.NDArray[np.float64]: ...
    def field_to_host(self) -> npt.NDArray[np.complex128]: ...
    def bind_edge(
        self,
        faces: npt.NDArray[np.float64],
        clip: npt.NDArray[np.float64],
        totals: npt.NDArray[np.float64],
        synthesis: npt.NDArray[np.float64],
        first_row: int,
    ) -> None: ...
    def crossed_bytes(self) -> tuple[int, int]: ...
    def bind_edge_lines(
        self,
        axes: npt.NDArray[np.uint64],
        counts: npt.NDArray[np.uint64],
        positions: npt.NDArray[np.uint64],
        windows: npt.NDArray[np.float64],
        inward: npt.NDArray[np.uint64],
        factors: npt.NDArray[np.float64],
        measures: npt.NDArray[np.float64],
        onsets: npt.NDArray[np.uint64],
        steps: npt.NDArray[np.float64],
        k_min: float,
        scale: npt.NDArray[np.float64],
        batch_bytes: int,
    ) -> None: ...
    def edge_reading(
        self, faces: bool, whole: bool, totals: bool, strip: bool, current: bool
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float
    ]: ...
    def set_crossing_workers(self, workers: int) -> None: ...
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
    def cpu_execution(self) -> dict[str, int | bool]: ...
    @property
    def fft_planning(self) -> dict[str, object]: ...
    @property
    def raman_kernel_transforms(self) -> int: ...
    @property
    def fsal_ready(self) -> bool: ...
    @property
    def rk_source_evaluations(self) -> int: ...
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
def export_fftw_wisdom() -> str | None: ...
def import_fftw_wisdom(text: str) -> bool: ...
def fit_blas_threads(threads: int) -> str: ...
def register_host_blas() -> None: ...
def rk_score(partials: npt.NDArray[np.float64], rtol: float, atol: float) -> float: ...
def strict_witness(
    a: float, b: float, c: float
) -> tuple[float, float, float, float]: ...
def watch_check(
    policy: dict[str, object],
    state: dict[str, object],
    reading: npt.NDArray[np.float64],
    z: float,
) -> dict[str, Any]: ...
def watch_due(
    policy: dict[str, object], state: dict[str, object], z: float
) -> bool: ...
def watch_split_limit(
    policy: dict[str, object], occupancy: npt.NDArray[np.float64]
) -> float: ...
def initialize_runtime(
    threads: int,
    kernels: int = 0,
    fft: int = 0,
    raman_fft: int = 0,
    raman_tail_fft: int = 0,
    band_tiles: int = -1,
) -> None: ...
def reference_fft3_pairs(
    values: npt.NDArray[np.complex128], warm: int, pairs: int
) -> list[float]: ...
def reference_stream_copies(elements: int, warm: int, repeats: int) -> list[float]: ...
def time_fft_plan(key: str, wisdom: str, effort: str) -> dict[str, float]: ...
def forget_fftw_wisdom() -> None: ...
def scoped_openmp_default() -> int: ...
def configure_native_spin(milliseconds: int) -> None: ...
def scoped_block_time() -> int: ...
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
def trial_tables(
    nodes: npt.NDArray[np.float64],
    order: int,
    orders: list[int],
    compensated: bool,
    predictor: npt.NDArray[np.float64],
    corrector: npt.NDArray[np.float64],
    factored: npt.NDArray[np.float64],
    spans: npt.NDArray[np.float64],
    carried: npt.NDArray[np.float64],
    endpoint: npt.NDArray[np.float64],
    defects: npt.NDArray[np.float64],
) -> None: ...
def vector_exp(x: npt.NDArray[np.float64], out: npt.NDArray[np.float64]) -> None: ...
def vector_log(x: npt.NDArray[np.float64], out: npt.NDArray[np.float64]) -> None: ...
def vector_sincos(
    x: npt.NDArray[np.float64],
    sine: npt.NDArray[np.float64],
    cosine: npt.NDArray[np.float64],
) -> None: ...
def vector_exp_complex(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    real: npt.NDArray[np.float64],
    imag: npt.NDArray[np.float64],
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
class BackendRefusal(RuntimeError): ...

class ScriptedMarch:
    def __init__(
        self,
        policy: dict[str, object],
        norms: npt.NDArray[np.float64],
        carrier_rate: float,
        steps: npt.NDArray[np.float64],
        edges: npt.NDArray[np.float64],
    ) -> None: ...
    def run(
        self, state: dict[str, object], bounds: dict[str, object]
    ) -> dict[str, Any]: ...
    def log(self) -> list[dict[str, Any]]: ...
