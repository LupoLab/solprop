# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The CPU backend: the seam, the stage driver and the statistics.

:class:`NumbaCpuBackend` steps the same problems as
:class:`solprop.backends.numpy_ref.backend.NumpyRefBackend` and agrees with it
to about 1e-15 rather than to the last bit (``docs/dev/backends.md`` section 2). It
inherits its step control from :class:`solprop.backends.driver.StepDriver` -- march,
replay, landing, history and the noise models are the same code on every backend --
and adds what is its own: the fused kernels of
:mod:`solprop.backends.numba_cpu.kernels`, the FFTW plans and thread pools of
:mod:`solprop.backends.numba_cpu.fft`, buffers allocated once at construction and never
again, and the four backend-owned statistics computed in compiled passes rather than
through the reference's per-plane re-synthesis.

**Nothing is allocated inside a step.** Every buffer is created here, named in
:meth:`NumbaCpuBackend.memory_ledger`, and written in place for the life of the run;
the plans are bound to those buffers, so a stage is a sequence of kernel calls,
in-place GEMMs and ``execute()`` calls with no array ever handed between them.

**The compiled step driver of** ``docs/dev/backends.md`` **section 3.5 is not built.**
The Python-level driver's dispatch overhead was measured below the run-to-run noise
floor on both target problems, and the deferral is by name.
"""

import functools
import logging
import math
import os
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

import numpy as np
from jaxtyping import Complex, Float, Num

from solprop._helpers import NUMBA
from solprop._process import release
from solprop.backends.bodies import RateArrays
from solprop.backends.driver import (
    StepDriver,
    refuse_general_leaves,
)
from solprop.backends.exponential import refuse_carrier_frame, refuse_formulation
from solprop.backends.low_storage import LowStorageMarch
from solprop.backends.memory import adams_memory
from solprop.backends.multistep import AdamsMarch, CoupledBlock
from solprop.backends.numba_cpu import fft, kernels, openmp
from solprop.backends.numba_cpu import planar as planar_kernels
from solprop.backends.numba_cpu import spatial as spatial_kernels
from solprop.backends.numba_cpu.coupled_flow import (
    FLOW_LEDGER_ROWS,
    FlowWorkspace,
    flow_kernels,
    flow_pad,
)
from solprop.backends.numba_cpu.dispersive import (
    DISPERSIVE_LEDGER_ROWS,
    DispersiveLayout,
    DispersiveMatrices,
    DispersiveWorkspace,
)
from solprop.backends.numba_cpu.dispersive_spatial import (
    SpatialDispersive,
    SpatialLayout,
)
from solprop.backends.numba_cpu.envelope_plasma import EnvelopeWorkspace
from solprop.backends.numba_cpu.field_plasma import HeldWorkspace
from solprop.backends.numba_cpu.general_modal import (
    GENERAL_LEDGER_ROWS,
    GeneralWorkspace,
)
from solprop.backends.numba_cpu.low_storage import NumbaLowStorage
from solprop.backends.numba_cpu.multistep import (
    NumbaAdamsKernels,
    NumbaExponentialKernels,
)
from solprop.backends.numba_cpu.pass_timers import PassTimers, timers_from_environment
from solprop.backends.numba_cpu.tables import HostTable
from solprop.backends.numpy_ref import passes
from solprop.backends.numpy_ref.admission import refuse_general_route
from solprop.backends.numpy_ref.backend import QUADRATURE_ALARM, BlockResidual
from solprop.backends.populations import axis_refusal, population_statistics
from solprop.backends.spatial import (
    FarFieldTables,
    collect_spatial,
    radial_absorber_matrix,
)
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError, SolpropWarning
from solprop.fourier import GridKind
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.noise import NoiseRealisation
from solprop.output.capability import (
    declared_capability,
    refusal_for,
    withheld_accessors,
)
from solprop.profiles import ZProfile
from solprop.spatial import ApertureTables
from solprop.stats import (
    _POYNTING_STATISTICS,
    CATALOGUE,
    METRIC_WEIGHTED_STATISTICS,
    ON_AXIS_PLASMA_STATISTICS,
    PLASMA_INTEGRATION_STATISTICS,
    WEIGHT_DEPENDENT_STATISTICS,
    StatsSpec,
    _poynting_statistics,
    analytic_field,
    collect,
    weight_refusal_for,
)
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import Multistep
from solprop.translate.coupled import metric_at
from solprop.translate.linop import power_weight_at
from solprop.translate.plasma_source import _BENCHMARK_GATED
from solprop.translate.problem import (
    MAX_N_COMPONENTS,
    MAX_N_POL,
    SCHEMA_VERSION,
    KernelProblem,
)
from solprop.translate.responses import ResponseStage
from solprop.translate.stage import StageScalars
from solprop.translate.tdse import TDSEPurpose

__all__ = ["NumbaCpuBackend"]

logger = logging.getLogger(__name__)

_STACKED_PROJECTION_MIN_NODES = 256
_STACKED_PROJECTION_MAX_MODES = 4


class NumbaCpuBackend(StepDriver[Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]]):
    """Propagate one Kerr, Raman and plasma problem with fused compiled kernels.

    The execution a bake-off of candidate backends chose (``docs/dev/backends.md``
    section 5). It implements :class:`solprop.backends.protocol.Backend` and steps a
    single problem (``max_batch = 1``) on either grid: mode-averaged or multimode on the
    fixed transverse quadrature, scalar or vector-isotropic.

    **Stage representations.** Constant and parameter-expanded operators use the
    per-stage bundle of host scalars. A z-tabulated ``Phi`` or prefactor uses
    preallocated host stage arrays and the reference's Clenshaw evaluation order
    through :class:`solprop.backends.numba_cpu.tables.HostTable`. Spatial constant
    operators use the separable representation. Device readiness is not required
    by this CPU backend.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to step. It is validated and refused by name here.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, which the statistics need and the contract does
        not carry.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, for the statistics that need them.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name.
    threads : int, optional
        Override for the kernels' thread pool. ``None`` takes the regime's choice and
        then the ``SOLPROP_NUMBA_THREADS`` environment variable.
    fftw_threads : int, optional
        Override for FFTW's pool, or ``SOLPROP_FFTW_THREADS``.
    transverse_fftw_threads : int, optional
        Override for Cartesian transverse FFTs, or
        ``SOLPROP_TRANSVERSE_FFTW_THREADS``. Otherwise apply the existing FFT size
        rule to that plan independently of the temporal plan.
    blas_threads : int, optional
        Override for the BLAS pool, or ``SOLPROP_BLAS_THREADS``.
    plan_flag : str, optional
        The FFTW planning effort, or ``SOLPROP_FFTW_PLAN``. ``"measure"`` by default.
    regime : {'small', 'large'}, optional
        Override for the threading regime, or ``SOLPROP_NUMBA_REGIME``. ``None`` reads
        it off the problem's element count, which is what a run should do; forcing it
        is for a measurement that wants a problem below the threshold stepped as a
        large one, so that it can be measured against thread count at all.

    Raises
    ------
    BackendError
        Naming what about the problem this backend does not execute.

    Examples
    --------
    >>> from solprop.backends.numba_cpu import NumbaCpuBackend
    >>> NumbaCpuBackend.name, NumbaCpuBackend.max_batch
    ('numba_cpu', 1)
    """

    name = "numba_cpu"
    max_batch = 1
    helpers = NUMBA
    # Both signs of the propagator are consumed by the stage that formed them -- the
    # forward in pass 1, the inverse in pass 7 -- and come from one real `exp`, one
    # `sincos` and a reciprocal, so `recompute` evaluates `dPhi` exactly as often as
    # `tabulate` does and the table buys residency alone: 561 us against 574 and
    # 128 KiB against 768 on the soliton case.
    propagator_default = "recompute"
    _wisdom_digest: str | None = None

    def __init__(  # noqa: PLR0913, PLR0915 - the seam's arguments, four pool overrides
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        threads: int | None = None,
        fftw_threads: int | None = None,
        transverse_fftw_threads: int | None = None,
        blas_threads: int | None = None,
        plan_flag: str | None = None,
        regime: str | None = None,
    ) -> None:
        """Adopt a validated problem, take the thread pools and plan every transform."""
        if _BENCHMARK_GATED.get():
            raise BackendError("the gated plasma benchmark requires numpy_ref")
        _refuse_unsupported(problem)
        problem.validate()
        super().__init__(problem, grid=grid, modes=modes, profiles=profiles)
        # SEP 28 phase 0's opt-in per-pass timers; ``None`` (the default) costs one
        # comparison per instrumented pass.
        self._pass_timers: PassTimers | None = timers_from_environment()
        # Formed once, so a geometry the far-field statistics refuse is refused here,
        # before any plan or step (SEP 23 section 6).
        self._far_field = FarFieldTables(problem)
        self._host_phi = (
            HostTable(problem.linear.polynomial)
            if problem.linear.phi_representation == "table"
            else None
        )
        self._host_prefactor = (
            HostTable(problem.responses.prefactor)
            if problem.responses.prefactor_representation == "table"
            and problem.responses.prefactor is not None
            else None
        )
        grids, transform = problem.grids, problem.transform
        _, n_modes, n_pol_state, n_omega = problem.shape
        self._n_columns = n_modes * n_pol_state
        self._n_pol_state = n_pol_state
        self._n_omega = n_omega
        self._n_to = grids.n_to
        self._kind: GridKind = problem.grid_kind
        self._crop = np.ascontiguousarray(grids.crop_index)
        self._contiguous = bool(
            np.array_equal(self._crop, np.arange(n_omega, dtype=self._crop.dtype))
        )
        # ADR-025's in-stage scheme is a second term on every stage rather than a
        # factor on the accepted state, so the kernel is compiled with or without it.
        self._spatial_absorber = radial_absorber_matrix(transform, grids.pad_scale)
        self._in_stage_absorber = bool(self._spatial_absorber.size) or (
            problem.responses.temporal_absorber_fine is not None
            and problem.responses.absorber_scheme == "nonlinear"
        )
        geometry = transform.geometry
        # A fibre problem has no spatial geometry; the threading policies below
        # (kernel cutoff, stacked projection, Raman tail, fraction reuse) were
        # measured on fibre transforms only, so a radial run keeps the general ones.
        self._fibre, self._free = geometry is None, problem.linear.separable is not None
        if self._free and geometry is not None:
            self._in_stage_absorber |= bool(np.any(geometry.absorber_rate))
        self._n_pol_node = transform.n_pol * n_pol_state
        (
            self._slab_block,
            self._slab_rows_per_y,
            self._chunk_nodes,
            self._temporal_rows,
        ) = _chunk_layout(problem)
        self._chunk_rows = self._chunk_nodes * self._n_pol_node
        # The plasma statistics between planes are read off what the accepted step
        # already computed, so the response column is specialised to leave them
        # behind -- but only for a run that asked for them, because it is one more
        # compilation of the package's hottest kernel. Not under the modified noise
        # model: there the last stage's input carries the noise draw, while the
        # statistics are defined on the committed field the reference scans.
        self._stash_plasma = problem.initial.noise_model != "modified" and bool(
            {
                "electron_density",
                "peak_ionisation_rate",
                "species_ionisation",
                "species_electron_density",
            }
            & set(problem.stats_spec.periodic)
        )
        self._spec = kernels.column_spec(
            problem.responses,
            self._kind,
            self._n_pol_node,
            parallel=False,
            stash=self._stash_plasma,
        )
        # The hybrid planar kind scores no quadrature alarm, as on the reference: its
        # transform carries no Kronrod projection yet.
        self._embedded = None if self._slab else _embedded_projection(problem)
        self._stacked_modal_projection = (
            self._fibre
            and self._kind == "field"
            and self._embedded is not None
            and _stacked_projection_shape(
                transform.n_modes, transform.n_nodes, self._chunk_nodes
            )
        )
        self._real_radial = (
            geometry is not None
            and geometry.kind == "radial"
            and self._kind == "envelope"
            and self._embedded is None
        )

        elements = _regime_elements(problem)
        self._regime_elements = (
            fft.MODAL_REGIME_ELEMENTS
            if self._fibre and transform.n_nodes > 1
            else fft.REGIME_ELEMENTS
        )
        pools = fft.choose_pools(
            elements,
            transform.synthesis.shape[0] * self._n_to,
            self._temporal_rows * self._n_to,
            threads=_override("SOLPROP_NUMBA_THREADS", threads),
            fftw_threads=_override("SOLPROP_FFTW_THREADS", fftw_threads),
            blas_threads=_override("SOLPROP_BLAS_THREADS", blas_threads),
            regime=regime or os.environ.get("SOLPROP_NUMBA_REGIME") or None,
            regime_elements=self._regime_elements,
            spatial=self._free,
        )
        self._raman_threads = fft.choose_pools(
            0,
            0,
            2 * self._chunk_nodes * self._spec.raman_drives * self._n_to
            if self._spec.raman
            else 0,
            fftw_threads=_override("SOLPROP_FFTW_THREADS", fftw_threads),
        ).fftw
        self._raman_tail_nodes = (
            transform.n_nodes % self._chunk_nodes
            if self._spec.raman and self._fibre
            else 0
        )
        self._raman_tail_threads = fft.choose_pools(
            0,
            0,
            2 * self._raman_tail_nodes * self._spec.raman_drives * self._n_to,
            fftw_threads=_override("SOLPROP_FFTW_THREADS", fftw_threads),
        ).fftw
        self._transverse_threads = fft.choose_pools(
            0,
            0,
            self._n_columns * self._n_omega if self._free else 0,
            fftw_threads=_override(
                "SOLPROP_TRANSVERSE_FFTW_THREADS", transverse_fftw_threads
            ),
            spatial=True,
        ).fftw
        # The node loop is a prange only in the large regime, and the specialisation is
        # part of the compiled kernel, so it is decided before anything is compiled.
        self._spec = self._spec._replace(parallel=pools.regime == "large")
        self._pools, self._limiter = fft.apply_pools(pools)
        self._parallel_spatial = (
            self._free
            and self._pools.numba > 1
            and self._pools.regime == "large"
            and self._n_columns * self._n_omega > fft.REGIME_ELEMENTS
        )
        self._spatial_layout = (
            self._free and self._kind == "envelope" and not self._slab
        )
        # The far-field statistics' banded pass (SEP 23 section 6), its bands in a
        # prange under the spatial stages' own rule; its partials are sized here.
        geometry = problem.transform.geometry
        self._far_field_pass: spatial_kernels.FarFieldPass | None = (
            spatial_kernels.FarFieldPass(problem, parallel=self._parallel_spatial)
            if geometry is not None and geometry.kind == "free"
            else None
        )
        self._flag = plan_flag or os.environ.get("SOLPROP_FFTW_PLAN", "measure")
        self._wisdom_loaded = fft.import_wisdom()
        family: fft.ColumnFamily = self._spec.family  # pyright: ignore[reportAssignmentType]
        # Every FFTW and Numba thread started from here on inherits this thread's CPUs.
        if release():  # pragma: no cover - Linux under an OpenMP binding only
            warnings.warn(
                "numba_cpu: an OpenMP runtime loaded earlier (solprop_native, imported "
                "first under OMP_PROC_BIND or OMP_PLACES) bound the main thread; it is "
                "released, but free-space FFTW transforms can still run several times "
                "slower. Unset OMP_PROC_BIND and OMP_PLACES for numba_cpu runs.",
                SolpropWarning,
                stacklevel=2,
            )
        self._plans = fft.build_plans(
            family,
            self._temporal_rows,
            self._n_to,
            grids.n_t,
            flag=self._flag,
            threads=self._pools.fftw,
            raman_threads=self._raman_threads,
            chunk_columns=self._chunk_nodes * self._spec.raman_drives
            if self._spec.raman
            else 0,
            tail_columns=self._raman_tail_nodes * self._spec.raman_drives,
            raman_tail_threads=self._raman_tail_threads,
            absorber=problem.responses.temporal_absorber is not None
            and (not self._free or problem.responses.absorber_scheme == "split_step"),
            in_stage_absorber=self._in_stage_absorber and not self._free,
            embedded=self._embedded is not None,
        )
        self._transverse = (
            fft.build_transverse_plans(
                (geometry.shape[1], geometry.shape[0], n_pol_state, n_omega),
                flag=self._flag,
                threads=self._transverse_threads,
                spatial_contiguous=self._spatial_layout,
            )
            if self._free and geometry is not None and not self._slab
            else None
        )
        data = problem.linear.separable
        self._separable = (
            None
            if data is None
            else (
                data.k_squared,
                data.kperp_sq,
                data.k_squared_low,
                data.kperp_sq_low,
                data.frame,
                data.spectral_rate,
                data.prefactor_numerator,
                data.prefactor_clamped,
                data.active,
                data.evanescent_rate,
                data.n_pol,
                data.kperp_rate,
            )
        )
        integrator = problem.stepper.integrator
        if isinstance(integrator, LowStorageRungeKutta):
            # Before the buffers: the march's resident engine replaces the
            # Runge-Kutta march's, which are then not allocated (SEP 24 section 6.6).
            self._low_storage = LowStorageMarch(self, integrator)
        self._allocate(_override("SOLPROP_FFTW_THREADS", fftw_threads))
        self._compile()
        logger.debug(
            "numba_cpu: %s grid, %s family, shape %s, %d nodes in chunks of %d, "
            "%s propagators, pools %s",
            self._kind,
            self._spec.family,
            problem.shape,
            transform.n_nodes,
            self._chunk_nodes,
            self._propagator_mode,
            self._pools.as_metadata(),
        )

    @property
    def _slab(self) -> bool:
        """Whether the problem is the hybrid planar kind (kernel schema 15).

        It is separable and so ``_free`` too, and branches off the Cartesian path
        wherever its columns are not its nodes.

        Returns
        -------
        bool
            True when whole y rows are blocked, which only the ``"slab"`` kind does.
        """
        return self._slab_block > 0

    def _plan_slab(self) -> None:
        """Plan the hybrid planar kind's y transforms and pack its operator, or not.

        The column volume carries a second half for the in-stage absorber's rows
        when that term is on (:func:`solprop.backends.numba_cpu.planar.
        build_slab_plans`).
        """
        geometry = self._problem.transform.geometry
        self._slab_plans = (
            planar_kernels.build_slab_plans(
                (geometry.shape[1], self._slab_rows_per_y, self._n_omega),
                absorber=self._in_stage_absorber,
                flag=self._flag,
                threads=self._transverse_threads,
            )
            if self._slab and geometry is not None
            else None
        )
        self._slab_data = planar_kernels.slab_data(self._problem) if self._slab else ()

    @property
    def _fsal(self) -> bool:
        """Return whether the last stage of a step may serve as the first of the next.

        The tableau's first-same-as-last property makes stage seven's *input* the accept
        pass's own result, so its derivative is stage one of the next step once it is
        rebased by ``exp(dPhi)`` for the re-anchored interaction picture
        (``docs/dev/stepper.md`` section 2). One evaluation in seven, 14.3 % of a step.

        A coupled flow does not take it either, because the reference does not: the
        identity holds there only through the whole step's forward action and only
        within one section, and ``numpy_ref`` evaluates every stage of a flow step
        (``docs/dev/stepper.md`` section 2); this backend keeps parity with it.

        Two things would break the identity and one does not. The hard band mask does
        not: the prefactor is exactly zero outside the active mask, so no stage ever
        writes there, the state never acquires content there and the mask is a bitwise
        no-op (measured at 9.9e-17). The ``"split_step"`` absorber does, because it
        changes the state after the last stage -- which is what the in-stage scheme was
        adopted to stop doing. A shot-noise realisation does not either: stage seven
        adds ``N(z + h)`` and so does stage one of the next step, at the same position
        and from the same draw.

        Returns
        -------
        bool
            Whether this problem's tableau and boundary treatments allow the saving.
        """
        responses = self._problem.responses
        split_step = (
            responses.temporal_absorber is not None
            and responses.absorber_scheme == "split_step"
        )
        # Only the Runge-Kutta march recycles a stage: an Adams march keeps its own
        # history and the low-storage march's last evaluation is at a predicted
        # state, not at the accepted one (SEP 24 Q13).
        return (
            self._problem.stepper.tableau.first_same_as_last
            and not split_step
            and self._problem.linear.coupled_flow is None
            and self._problem.stepper.integrator is None
        )

    # --- construction ---------------------------------------------------------

    def _allocate(self, fftw_threads: int | None = None) -> None:
        """Create every buffer the run will ever write, once.

        Parameters
        ----------
        fftw_threads : int, optional
            The FFTW pool override, which the general modal route's own plan family
            sizes itself by when it is ``None``.
        """
        problem = self._problem
        shape = (self._n_columns, self._n_omega)
        self._allocate_runge_kutta(shape)
        node_dtype = np.float64 if self._spec.family == "field_real" else np.complex128
        out_dtype = np.complex128 if self._kind == "envelope" else np.float64
        plans = self._plans
        # Plain free space reads its nodes from the chunk's time rows and writes its
        # polarisation into the rows the forward transform reads, so the two copies
        # between them are identities (SEP 28). Cleared once: rows past a short last
        # chunk are transformed but never read back, and must stay finite.
        self._aliased = (
            self._free
            and not self._slab
            and problem.responses.dispersive_kerr is None
            and plans.modal_time.shape == (self._chunk_rows, self._n_to)
            and plans.modal_time.dtype == node_dtype
            and plans.projected.shape == (self._chunk_rows, self._n_to)
            and plans.projected.dtype == out_dtype
        )
        plans.modal_time[:] = 0
        plans.projected[:] = 0
        if self._aliased:
            self._nodes = plans.modal_time
            self._polarisation = plans.projected
        else:
            self._nodes = np.zeros((self._chunk_rows, self._n_to), dtype=node_dtype)
            self._polarisation = np.zeros(
                (self._chunk_rows, self._n_to), dtype=out_dtype
            )
        self._chunk_projection = np.zeros(
            (0 if self._free else self._n_columns, self._n_to), dtype=out_dtype
        )
        transform = problem.transform
        self._allocate_matrices(node_dtype)
        self._scaled_embedded = (
            None if self._embedded is None else np.empty_like(self._embedded)
        )
        self._allocate_projection_chunks()
        self._allocate_channels()
        self._carrier = _harmonic_carrier(problem)
        self._raman_kernel: Complex[np.ndarray, "... n_omega_doubled"] | None = None
        self._raman_rates: tuple[float, ...] | None = None
        self._raman_densities: tuple[float, ...] | None = None
        self._raman_transforms = 0
        self._alarm = math.nan
        self._alarm_reported = False
        self._last_bundle: StageScalars | None = None
        self._stats_nodes: Float[np.ndarray, "n_rows n_to"] | None = None
        # One row per node, written by the last stage of every step and read when one
        # is accepted; a one-row dummy when the channel did not ask, which numba types
        # and the compiled branch never writes.
        self._stash = np.zeros(
            (
                problem.transform.n_nodes if self._spec.stash else 1,
                1 + 2 * self._spec.n_channels if self._spec.stash else 2,
            )
        )
        # Every population's two rows -- envelope ones included, which the spec's
        # field channel count does not see -- and the band rule's diagnostics.
        self._partial_rows = 2 * problem.flags.n_channels + (
            kernels.BAND_DIAGNOSTICS if self._spec.band else 0
        )
        self._population_partials = np.zeros(
            (1 + self._partial_rows, transform.n_nodes)
        )
        self._stash_current = False
        self._coarse_index = np.arange(self._n_omega, dtype=np.intp)
        self._free_temporal_rate = (
            problem.responses.temporal_absorber_fine
            if self._free
            and problem.responses.absorber_scheme == "nonlinear"
            and problem.responses.temporal_absorber_fine is not None
            else np.zeros(self._n_to if self._free else 0)
        )
        self._pad_scale = 1.0 / self._n_to
        self._raman_scale = 1.0 / (2 * self._n_to)
        # The radial collar's GEMM reads real rows; an analytic signal's real part is a
        # strided view, so it is copied here first rather than by numpy per stage.
        self._spatial_absorber_time = np.zeros(
            self._plans.modal_time.shape
            if self._spatial_absorber.size and self._spec.family == "field_analytic"
            else (0, 0)
        )
        self._absorber_spectrum = np.zeros(
            shape if self._in_stage_absorber and not self._free else (0, 0),
            dtype=np.complex128,
        )
        self._allocate_workspaces(fftw_threads)

    def _allocate_runge_kutta(self, shape: tuple[int, int]) -> None:
        """Create the Runge-Kutta march's stage rows, accumulators and propagators.

        The low-storage march's engine holds its own registers and flows, and these
        buffers are then empty: every route's stage reads them only under the
        Runge-Kutta march (SEP 24 section 6.6).

        Parameters
        ----------
        shape : tuple of int
            ``(n_columns, n_omega)``, one state's.
        """
        tableau = self._problem.stepper.tableau
        low_storage = self._low_storage is not None
        marched = (0, self._n_omega) if low_storage else shape
        rows = 0 if low_storage else tableau.n_stages
        self._stages = np.zeros((rows, *shape), dtype=np.complex128)
        self._solution = np.zeros(marched, dtype=np.complex128)
        self._error = np.zeros(marched, dtype=np.complex128)
        self._advanced = np.zeros(marched, dtype=np.complex128)
        self._partials = np.zeros((3, self._n_columns))
        # The recycled first-same-as-last stage, rebased into the next step's picture.
        # Held rather than recomputed only when `_fsal_ready` says the identity holds.
        self._recycled = np.zeros(marched, dtype=np.complex128)
        self._has_recycled = False
        self._pending: StageScalars | None = None
        self._fractions = np.array(
            sorted({float(node) for node in tableau.nodes} | {1.0})
        )
        self._tabulating = self._propagator_mode == "tabulate"
        distinct = 0 if low_storage else len(self._fractions) if self._tabulating else 1
        self._forward = np.zeros((distinct, *shape), dtype=np.complex128)
        self._inverse = np.zeros((distinct, *shape), dtype=np.complex128)
        self._exponent = np.zeros(marched, dtype=np.complex128)
        self._recomputed = math.nan

    def _allocate_matrices(self, node_dtype: type) -> None:
        """Hold the transform's matrices, their taper-scaled copies and the row rate.

        Parameters
        ----------
        node_dtype : type
            The node field's dtype, which a modal synthesis takes.
        """
        transform = self._problem.transform
        # The slab's matrices are real and act on real views of complex rows, as the
        # radial real GEMM does.
        real = self._real_radial or self._slab
        self._synthesis = np.ascontiguousarray(
            transform.synthesis, dtype=np.float64 if real else node_dtype
        )
        self._scaled_synthesis = np.empty_like(self._synthesis)
        self._projection = np.ascontiguousarray(
            transform.projection, dtype=np.float64 if self._slab else None
        )
        self._scaled_projection = np.empty_like(self._projection)
        # The y collar's in-stage rate on each (p, y) row: it depends on y alone (the
        # geometry checks it), so it is one entry per mode row.
        geometry = transform.geometry
        self._slab_row_rate = (
            np.repeat(
                geometry.absorber_rate.reshape(geometry.shape[1], -1)[:, 0],
                transform.synthesis.shape[1],
            )
            if self._slab and geometry is not None
            else np.zeros(0)
        )

    def _allocate_workspaces(self, fftw_threads: int | None) -> None:
        """Allocate the two route workspaces: envelope plasma and the general guide.

        Parameters
        ----------
        fftw_threads : int or None
            The FFTW pool override, which the general modal route's own plan family
            sizes itself by when it is ``None``.
        """
        self._allocate_envelope()
        self._allocate_field_plasma()
        self._plan_slab()
        self._allocate_dispersive(fftw_threads)
        problem = self._problem
        self._block_residual: BlockResidual | None = None
        self._flow = (
            None
            if problem.linear.coupled_flow is None
            else FlowWorkspace(
                problem,
                self._fractions,
                contiguous=self._contiguous,
                absorber=self._in_stage_absorber,
            )
        )
        self._general = (
            None
            if problem.transform.modal_tables is None
            else GeneralWorkspace(
                problem,
                flag=self._flag,
                fftw_threads=fftw_threads,
                parallel=self._spec.parallel,
            )
        )

    def _allocate_envelope(self) -> None:
        """Allocate the separate current streams and their forward plans."""
        assembly = self._problem.responses.envelope_plasma
        self._envelope = (
            None
            if assembly is None
            else EnvelopeWorkspace(
                assembly,
                (
                    self._chunk_nodes,
                    self._n_pol_node,
                    self._temporal_rows,
                    self._n_to,
                ),
                embedded=self._embedded is not None,
                parallel=self._spec.parallel,
                flag=self._flag,
                threads=self._pools.fftw,
            )
        )

    def _allocate_field_plasma(self) -> None:
        """Allocate the held-current products and their fixed transform plans."""
        self._field_plasma = (
            HeldWorkspace(
                self._problem,
                (self._chunk_nodes * self._n_pol_node, self._temporal_rows),
                embedded=self._embedded is not None,
                flag=self._flag,
                threads=self._pools.fftw,
            )
            if self._problem.responses.plasma_channels
            else None
        )

    def _allocate_dispersive(self, fftw_threads: int | None) -> None:
        """Allocate SEP 6's dispersive response's buffers and plans, when it is present.

        Parameters
        ----------
        fftw_threads : int or None
            The FFTW pool override, which sizes the response's batched plans when it is
            ``None``.
        """
        problem = self._problem
        responses = problem.responses
        self._dispersive: DispersiveWorkspace | None = None
        self._spatial_dispersive: SpatialDispersive | None = None
        if (
            responses.dispersive_kerr is None
            or problem.transform.modal_tables is not None
        ):
            # The general modal route's response belongs to its own workspace.
            return
        transform = problem.transform
        # Alone, the response owns the node field and the projected buffer: no column
        # runs and no other polarisation is projected beside it.
        stages = set(responses.response_order) - {ResponseStage.ACCUMULATE}
        alone = (
            stages == {ResponseStage.DISPERSIVE_KERR}
            and responses.envelope_plasma is None
        )
        if self._free:
            # Cartesian, line and planar: each chunk goes to time and back on its own.
            # Free space's in-stage absorber is a pass of its own, so only the planar
            # one reads the bare rows after the material.
            self._spatial_dispersive = SpatialDispersive(
                problem,
                self._plans,
                SpatialLayout(
                    rows=self._temporal_rows,
                    node_rows=self._chunk_rows,
                    n_pol_node=self._n_pol_node,
                    block=max(1, self._slab_block),
                    slab=self._slab,
                    alone=alone,
                    bare_time_free=not (self._slab and self._in_stage_absorber),
                    parallel=self._spec.parallel,
                ),
                flag=self._flag,
                fftw_threads=fftw_threads,
            )
            return
        self._dispersive = DispersiveWorkspace(
            problem,
            self._plans,
            self._nodes,
            self._chunk_projection,
            DispersiveMatrices(
                self._scaled_synthesis, self._scaled_projection, self._scaled_embedded
            ),
            DispersiveLayout(
                n_modes=transform.n_modes,
                n_pol_state=self._n_pol_state,
                chunk_rows=self._chunk_rows,
                chunks=-(-transform.n_nodes // self._chunk_nodes),
                alone=alone,
                embedded=self._embedded is not None,
                real_synthesis=self._synthesis.dtype == np.float64,
                bare_time_free=not self._in_stage_absorber,
                parallel=self._spec.parallel,
            ),
            flag=self._flag,
            fftw_threads=fftw_threads,
        )

    def _allocate_projection_chunks(self) -> None:
        """Pack the two real projection rules with each chunk's actual width."""
        self._projection_chunks: tuple[np.ndarray, ...] = ()
        self._scaled_projection_chunks: tuple[np.ndarray, ...] = ()
        self._projection_stacked = np.empty((0, 0))
        if not self._stacked_modal_projection or self._embedded is None:
            return
        transform = self._problem.transform
        stacked = np.vstack((self._projection, self._embedded))
        width = self._chunk_nodes * transform.n_pol
        self._projection_chunks = tuple(
            np.ascontiguousarray(stacked[:, first : first + width])
            for first in range(0, stacked.shape[1], width)
        )
        self._scaled_projection_chunks = tuple(
            np.empty_like(chunk) for chunk in self._projection_chunks
        )
        self._projection_stacked = np.empty((2 * self._n_columns, self._n_to))

    @property
    def _multipass_plasma(self) -> bool:
        """Whether the plasma runs in the multipass column rather than the fused one.

        Under the band rule (SEP 18), whose filters read ahead of the sample.

        Returns
        -------
        bool
            True for a field-grid plasma under the band rule.
        """
        return self._spec.plasma and self._spec.band

    def _compile(self) -> None:
        """Build every kernel this problem needs and compile it on its own buffers.

        The kernels are compiled eagerly, without written signatures. This is eager --
        nothing is left to the first step -- and it is stronger than a written
        signature, because the types come from the buffers the kernels will actually be
        handed and so cannot drift from them. What a hand-written signature would add is
        a compile-time error on a typo, which running the kernel here gives anyway.
        """
        self._combine = kernels.make_combine_and_pad(
            contiguous=self._contiguous,
            noise=self._problem.initial.noise_model == "modified",
            parallel=self._parallel_spatial,
        )
        self._spatial = spatial_kernels.make_kernels(
            parallel=self._parallel_spatial,
            parallel_transfers=self._parallel_spatial and self._spatial_layout,
        )
        self._finish_stage = kernels.make_finish_stage(
            contiguous=self._contiguous, absorber=self._in_stage_absorber
        )
        # Reuse pays in serial columns and in one-component parallel
        # columns; two-component parallel columns retain the original traversal.
        self._plasma_fraction_reuse = _reuse_plasma_fraction(
            self._spec, not self._fibre
        )
        self._planar = planar_kernels.make_kernels(parallel=self._parallel_spatial)
        multipass = self._multipass_plasma
        # Under the band rule the plasma writes J separately; the fused column
        # owns only the collared non-plasma polarisation (SEP 27).
        self._column = kernels.make_response_column(
            self._spec._replace(plasma=False, stash=False, preloaded=False)
            if multipass
            else self._spec,
            reuse_unchanged_fraction=self._plasma_fraction_reuse and not multipass,
        )
        self._band_column = (
            kernels.make_band_plasma_column(self._spec) if multipass else None
        )
        self._raman_finish = kernels.make_raman_finish(self._spec)
        # The accept pass and the weighted residual thread their columns only where a
        # state has enough of them to share (`kernels.ACCEPT_ELEMENTS`).
        self._accept, self._weighted_residual = kernels.accept_kernels(
            self._pools.numba, self._n_columns, self._n_omega
        )
        self._flow_combine, self._flow_accumulate, self._flow_accept = flow_kernels(
            self._pools.numba, self._n_columns, self._n_omega
        )
        self._stats_column = (
            kernels.make_band_stats_column(self._spec)
            if self._spec.band
            else kernels.make_stats_column(self._spec)
        )
        # The Adams march steps through these same kernels (SEP 14 phase 2).
        self._adams = self._adams_march()

    # --- the seam --------------------------------------------------------------

    def _apply_mask(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Zero everything outside the active band, out of place.

        Parameters
        ----------
        state : numpy.ndarray
            The state in the canonical layout.

        Returns
        -------
        numpy.ndarray
            The masked state.
        """
        out = np.empty_like(state)
        kernels.apply_hard_mask(
            state.reshape(self._n_columns, self._n_omega),
            self._problem.grids.active_mask,
            out.reshape(self._n_columns, self._n_omega),
        )
        return out

    def _begin_plane(self) -> None:
        """Forget the alarm of the plane just finished.

        The quadrature alarm describes the steps of one call and warns at most once
        about them: a rule too coarse for the integrand is coarse at every step of the
        plane and says so once.

        The recycled first-same-as-last stage is kept across the plane: the state the
        plane observed is the one the run adopted, so the identity holds there as
        between any two steps (SEP 14 phase 3). A resume reloads the state with no
        stage behind it and evaluates all seven, so a resumed replay agrees with an
        uninterrupted one to round-off rather than bitwise.
        """
        self._alarm = math.nan
        self._alarm_reported = False

    def _observe_inside(self, position: float) -> None:
        """Observe a plane inside the last step, and forget the stash of the march's.

        The plasma statistics stashed by the step's last column evaluation belong to
        the march's state, not to the field interpolated at the plane.

        Parameters
        ----------
        position : float
            The plane, m, inside the last accepted step.
        """
        super()._observe_inside(position)
        self._stash_current = False

    def _allocate_channels(self) -> None:
        """Allocate channel storage separately from the unchanged field pools."""
        self._statistics_power_weight = np.empty(
            self._problem.shape[1:]
            if self._problem.transform.geometry is None
            else (0,)
        )
        problem = self._problem
        plasma = problem.responses.plasma_coefficients
        self._channel_coefficients, self._channel_metadata = _channel_rate_arrays(
            problem
        )
        self._rate = (
            _rate_arrays(plasma)
            if self._spec.n_channels <= 1
            else (self._channel_coefficients, self._channel_metadata)
        )
        self._channel_scalars = np.zeros((4, self._spec.n_channels))
        for index, channel in enumerate(problem.responses.plasma_channels):
            self._channel_scalars[0, index] = channel.charge_mass_ratio
            self._channel_scalars[1, index] = channel.ionisation_potential
        self._channel_scratch = np.zeros(
            (
                self._spec.n_channels if self._spec.n_channels > 1 else 0,
                self._chunk_nodes,
                4,
            )
        )
        integral = problem.responses.plasma_integral
        self._taps = np.ascontiguousarray(integral.taps)
        # The multipass plasma column's scratch (SEP 25 section 6.6): rows of the fine
        # grid per node of a chunk.
        multipass = self._multipass_plasma
        self._plasma_work = np.zeros(
            (
                self._chunk_nodes if multipass else 0,
                kernels.band_work_rows(self._spec),
                self._n_to,
            )
        )

    def close(self) -> None:
        """Release the buffers and give the BLAS pool back. Idempotent."""
        if not self._closed:
            fft.export_wisdom()
            fft.release_pools(self._limiter)
        self._limiter = None
        self._state = None
        self._noise = None
        self._raman_kernel = None
        self._stats_nodes = None
        self._envelope = None
        self._field_plasma = None
        self._general = None
        self._flow = None
        self._dispersive = None
        self._spatial_dispersive = None
        if self._low_storage is not None:
            self._low_storage.close()
        self._spatial_absorber = np.empty((0, 0))
        self._spatial_absorber_time = np.empty((0, 0))
        self._absorber_spectrum = np.empty((0, 0), dtype=np.complex128)
        self._statistics_power_weight = np.empty(0)
        self._closed = True

    def metadata(self) -> dict[str, object]:
        """Return what this run's execution was, for the output's metadata.

        The thread configuration alone was measured to move one step by a factor of
        nine, so a result whose pools are not recorded is not reproducible.

        The wisdom digest is taken at the first record and kept: every plan this
        backend uses is built when it is constructed, so the wisdom they were chosen
        from cannot change afterwards, and exporting the wisdom again at every plane
        was most of the record's cost (SEP 25 section 6.3).

        Returns
        -------
        dict
            The base record, plus the pools, the plan flag, the wisdom digest and the
            specialisation.
        """
        if self._wisdom_digest is None:
            self._wisdom_digest = fft.wisdom_digest()
        return {
            **super().metadata(),
            "column_family": self._spec.family,
            "plasma_fraction_reuse": self._plasma_fraction_reuse,
            "column_plasma_magnitude": self._spec.plasma or self._spec.n_pol != 2,
            "parallel_nodes": self._spec.parallel,
            "kernel_regime_elements": self._regime_elements,
            "parallel_spatial": self._parallel_spatial,
            "parallel_spatial_transfers": self._parallel_spatial
            and self._spatial_layout,
            "transverse_layout": "spatial" if self._spatial_layout else "columns",
            "transverse_fftw_threads": self._transverse_threads if self._free else None,
            "raman_fftw_threads": self._raman_threads if self._spec.raman else None,
            "raman_tail_nodes": self._raman_tail_nodes,
            "raman_tail_fftw_threads": self._raman_tail_threads
            if self._raman_tail_nodes
            else None,
            "radial_real_gemm": self._real_radial,
            "modal_projection": "stacked_packed"
            if self._stacked_modal_projection
            else "separate",
            "stacked_projection_min_nodes": _STACKED_PROJECTION_MIN_NODES,
            "stacked_projection_max_modes": _STACKED_PROJECTION_MAX_MODES,
            "node_chunk": self._chunk_nodes,
            "slab_y_block": self._slab_block if self._slab else None,
            "general_node_chunk": None
            if self._general is None
            else self._general.maps.node_chunk,
            "general_fftw_threads": None
            if self._general is None
            else self._general.threads,
            "general_planning_seconds": 0.0
            if self._general is None
            else self._general.plans.planning_seconds,
            "plan_flag": self._flag,
            "planning_seconds": self._plans.planning_seconds,
            "envelope_planning_seconds": 0.0
            if self._envelope is None
            else self._envelope.planning_seconds,
            "held_current_planning_seconds": 0.0
            if self._field_plasma is None
            else self._field_plasma.planning_seconds,
            "dispersive_planning_seconds": 0.0
            if self._dispersive_workspace is None
            else self._dispersive_workspace.planning_seconds,
            "dispersive_alone": None
            if self._dispersive_workspace is None
            else self._dispersive_workspace.layout.alone,
            "wisdom_loaded": self._wisdom_loaded,
            "wisdom_digest": self._wisdom_digest,
            "fftw_wait_policy": openmp.WAIT_POLICY,
            **self._pools.as_metadata(),
        }

    @property
    def _dispersive_workspace(self) -> Any:
        """Return SEP 6's dispersive workspace of whichever route runs it, or ``None``.

        Returns
        -------
        DispersiveWorkspace or SpatialDispersive or None
            The fused path's, the Cartesian or planar route's, or the general modal
            route's, each with ``planning_seconds``, ``layout.alone`` and
            ``memory_ledger()``.
        """
        if self._dispersive is not None:
            return self._dispersive
        if self._spatial_dispersive is not None:
            return self._spatial_dispersive
        return None if self._general is None else self._general.dispersive

    @property
    def pass_timers(self) -> PassTimers | None:
        """The opt-in per-pass timers of a free-space evaluation (SEP 28 phase 0).

        Held when ``SOLPROP_PASS_TIMERS=1`` was set at construction
        (:mod:`solprop.backends.numba_cpu.pass_timers`), ``None`` otherwise. They
        record the Cartesian, line and planar evaluation's passes and the steppers'
        traversals around them; reset them before a timed window.

        Returns
        -------
        PassTimers or None
            The timers, or ``None`` when they are off.
        """
        return self._pass_timers

    def memory_ledger(self) -> dict[str, int]:
        """Return the size of every buffer the backend holds, in bytes.

        The names are the reference's for the buffers both backends have, so that the
        budget of ``docs/dev/architecture.md`` section 8 is measured against the same
        rows on either; the plan work areas are named beside them, which the reference
        has none of.

        Returns
        -------
        dict
            Buffer name to bytes.
        """
        field_bytes = self._n_columns * self._n_omega * 16
        march = (
            dict.fromkeys(("register_accumulator", "register_stage", "propagators"), 0)
            if self._low_storage is None
            else cast("NumbaLowStorage", self._low_storage.engine).held()
        )
        ledger = {
            "statistics_power_weight": self._statistics_power_weight.nbytes,
            "spatial_absorber_matrix": self._spatial_absorber.nbytes,
            "spatial_absorber_time": self._spatial_absorber_time.nbytes,
            "in_stage_absorber_term": self._absorber_spectrum.nbytes,
            "state": field_bytes,
            "stages": self._stages.nbytes,
            "accumulator_solution": self._solution.nbytes,
            "accumulator_error": self._error.nbytes,
            # The low-storage march's registers (SEP 24 section 6.6).
            "register_accumulator": march["register_accumulator"],
            "register_stage": march["register_stage"],
            "advanced": self._advanced.nbytes,
            "exponent": self._exponent.nbytes,
            "recycled_stage": self._recycled.nbytes,
            "step_partials": self._partials.nbytes,
            "population_partials": self._population_partials.nbytes,
            # The far-field pass's band partials and their sum (SEP 23 section 6).
            "statistics_partials": 0
            if self._far_field_pass is None
            else self._far_field_pass.nbytes,
            "host_phi_evaluation": 0
            if self._host_phi is None
            else self._host_phi.nbytes,
            "host_prefactor_evaluation": 0
            if self._host_prefactor is None
            else self._host_prefactor.nbytes,
            "channel_rate_coefficients": self._channel_coefficients.nbytes,
            "channel_rate_metadata": self._channel_metadata.nbytes,
            "channel_scalars": self._channel_scalars.nbytes,
            "plasma_work": self._plasma_work.nbytes,
            "channel_scratch": self._channel_scratch.nbytes,
            "plasma_stash": self._stash.nbytes,
            "coarse_index": self._coarse_index.nbytes,
            "free_temporal_rate": self._free_temporal_rate.nbytes,
            "carrier": self._carrier.nbytes,
            "noise": field_bytes if self._noise is not None else 0,
            # The Runge-Kutta march's pairs, or the low-storage march's table.
            "propagators": self._forward.nbytes
            + self._inverse.nbytes
            + march["propagators"],
            # Aliased views own no storage of their own (SEP 28).
            "node_field": 0 if self._aliased else self._nodes.nbytes,
            "node_polarisation": 0 if self._aliased else self._polarisation.nbytes,
            "chunk_projection": self._chunk_projection.nbytes,
            "synthesis": self._synthesis.nbytes + self._scaled_synthesis.nbytes,
            "projection": self._projection.nbytes + self._scaled_projection.nbytes,
            "embedded_projection": 0
            if self._embedded is None or self._scaled_embedded is None
            else self._embedded.nbytes + self._scaled_embedded.nbytes,
            "packed_projection": sum(
                chunk.nbytes
                for chunks in (
                    self._projection_chunks,
                    self._scaled_projection_chunks,
                )
                for chunk in chunks
            ),
            "stacked_projection": self._projection_stacked.nbytes,
            # What the run will hold, not what it has built yet: the kernel spectrum
            # is formed on the first stage that needs it and kept for the run.
            "raman_kernel_spectrum": (2 if self._spec.raman_drives == 3 else 1)
            * (self._n_to + 1)
            * 16
            if self._problem.flags.has_raman
            else 0,
            "stats_nodes": 0 if self._stats_nodes is None else self._stats_nodes.nbytes,
            # A low-storage march's state at its last field-free plane (SEP 22, Q14).
            "held_plane": 0 if self._held is None else self._held.nbytes,
        }
        if self._transverse is not None:
            ledger["transverse_scratch"] = self._transverse.spectrum.nbytes
            ledger["transverse_fft_plan_heap_growth"] = (
                self._transverse.plan_heap_growth
            )
        if self._slab_plans is not None:
            # Both halves of the column volume: the material's and, with the in-stage
            # absorber, the absorber's.
            ledger["transverse_scratch"] = self._slab_plans.spectrum.nbytes
            ledger["transverse_fft_plan_heap_growth"] = (
                self._slab_plans.plan_heap_growth
            )
        ledger["slab_row_rate"] = self._slab_row_rate.nbytes
        if self._envelope is not None:
            ledger.update(self._envelope.memory_ledger())
        if self._field_plasma is not None:
            ledger.update(self._field_plasma.memory_ledger())
            ledger["held_current_derivative"] = (
                self._problem.grids.t_window_fine_derivative.nbytes
            )
        # The general modal route's buffers, named on every route and zero where the
        # problem carries no modal tables.
        ledger.update(
            dict.fromkeys(GENERAL_LEDGER_ROWS, 0)
            if self._general is None
            else self._general.memory_ledger()
        )
        # The coupled flow's, likewise named on every route; its first three rows are
        # the reference's own.
        ledger.update(
            dict.fromkeys(FLOW_LEDGER_ROWS, 0)
            if self._flow is None
            else self._flow.memory_ledger()
        )
        # SEP 6's dispersive response, likewise named on every route.
        ledger.update(
            dict.fromkeys(DISPERSIVE_LEDGER_ROWS, 0)
            if self._dispersive_workspace is None
            else self._dispersive_workspace.memory_ledger()
        )
        # The Adams march's history (SEP 16 phase 3b), held by the host engine.
        ledger.update(adams_memory(self._problem, "numba_cpu"))
        return ledger | self._plans.ledger()

    # --- the step ---------------------------------------------------------------

    def _trial_step(
        self, z: float, dz: float
    ) -> tuple[float, Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]]:
        """Take one interaction-picture Runge-Kutta step and score it.

        The first stage is recycled from the previous step's last, rebased into this
        step's picture, whenever ``_fsal`` holds: the tableau is first-same-as-last and
        under the in-stage absorber nothing changes the state after the last stage.
        Otherwise -- ``split_step``, a fresh load, a new plane -- every stage is
        evaluated.

        Parameters
        ----------
        z : float
            Position at the start of the step, m.
        dz : float
            Length of the trial step, m, positive.

        Returns
        -------
        tuple
            The scaled error norm and the state at the end of the step.
        """
        state, whole, first = self._start_trial(z, dz)
        tableau = self._problem.stepper.tableau
        for index in range(first, tableau.n_stages):
            fraction = float(tableau.nodes[index])
            self._rk_source_evaluations += 1
            self._stage(state, index, dz, self._stage_scalars(z, z + fraction * dz))
        if self._flow is None:
            self._accept(
                state,
                self._propagator(1.0, whole),
                dz,
                self._solution,
                self._error,
                self._problem.grids.norm_mask,
                self._advanced,
                self._partials,
            )
        else:
            self._flow_accept(
                state,
                self._propagator(1.0, whole),
                dz,
                self._solution,
                self._error,
                self._flow.forward[self._slot(1.0)],
                self._flow.slots,
                self._problem.grids.norm_mask,
                self._advanced,
                self._flow.error,
                self._partials,
            )
        norm = self._score_trial(whole, dz)
        return norm, self._advanced.reshape(self._problem.shape).copy()

    def _start_trial(
        self, z: float, dz: float
    ) -> tuple[Complex[np.ndarray, "n_columns n_omega"], StageScalars, int]:
        """Prepare member state and FSAL accumulators for a trial.

        Parameters
        ----------
        z : float
            Start position, m.
        dz : float
            Positive step length, m.

        Returns
        -------
        tuple
            Flattened state, whole-step scalar bundle and first stage to evaluate.
        """
        state = self._require_state().reshape(self._n_columns, self._n_omega)
        self._hoist(z)
        tableau = self._problem.stepper.tableau
        # Every position this trial reads is known now: the tabulated fractions, the
        # whole step and the stages it evaluates. Their tables are read together, once
        # each, and the calls below take the bundles as kept.
        first = 1 if self._has_recycled else 0
        tabulated = self._fractions if self._tabulating else ()
        self._stage_bundles(
            z,
            [
                *(z + float(fraction) * dz for fraction in tabulated),
                z + dz,
                *(z + float(node) * dz for node in tableau.nodes[first:]),
            ],
        )
        self._prepare_propagators(z, dz)
        whole = self._stage_scalars(z, z + dz)
        if self._flow is not None:
            self._flow.prepare(whole.flow_section, dz)
        self._solution[:] = 0.0
        self._error[:] = 0.0
        if self._has_recycled:
            # The saving: stage one is the last stage of the accepted step before,
            # already rebased, so this step evaluates six of seven.
            self._stages[0] = self._recycled
            self._solution += float(tableau.solution_weights[0]) * self._recycled
            self._error += float(tableau.error_weights[0]) * self._recycled
        return state, whole, first

    def _score_trial(self, whole: StageScalars, dz: float) -> float:
        """Reduce this member's error partials and retain its pending FSAL bundle.

        Parameters
        ----------
        whole : solprop.translate.stage.StageScalars
            Whole-step propagator bundle.
        dz : float
            Positive step length, m.

        Returns
        -------
        float
            This member's scaled error norm, dimensionless.
        """
        controller = self._problem.stepper.controller
        # One partial per column, in any order: the norm is held to its summation
        # bound, not to bits (SEP 25).
        old = math.sqrt(float(np.sum(self._partials[0])))
        new = math.sqrt(float(np.sum(self._partials[1])))
        residual = math.sqrt(float(np.sum(self._partials[2])))
        scale = controller.scale(old, new)
        if isinstance(scale, np.ndarray):
            # Weighted: the division is per bin and inside the norm, so the single
            # summed residual above cannot express it and a second traversal does.
            self._weighted_residual(
                self._propagator(1.0, whole),
                dz,
                # A coupled flow's embedded difference is scored after the whole
                # step's action, as the accept pass left it.
                self._error if self._flow is None else self._flow.error,
                self._problem.grids.norm_mask,
                scale,
                self._partials[2],
            )
            norm = math.sqrt(float(np.sum(self._partials[2])))
        else:
            norm = residual / float(scale)
        # Held, not promoted: only an accepted step's last stage is the next step's
        # first, and `_on_accepted_step` is what says the step was kept.
        self._pending = whole if self._fsal else None
        return norm

    def _stage(
        self,
        state: Complex[np.ndarray, "n_columns n_omega"],
        index: int,
        dz: float,
        bundle: StageScalars,
    ) -> None:
        """Evaluate one stage derivative into ``self._stages[index]``.

        Parameters
        ----------
        state : numpy.ndarray
            The state at the start of the step.
        index : int
            Which stage of the tableau.
        dz : float
            Length of the whole step, m, which the stage combination scales by.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars, from the step's anchor to this node.
        """
        if self._slab:
            self._stage_slab(state, index, dz, bundle)
            return
        if self._free:
            self._stage_free(state, index, dz, bundle)
            return
        if self._general is not None:
            self._stage_general(state, index, dz, bundle)
            return
        tableau = self._problem.stepper.tableau
        fraction = float(tableau.nodes[index])
        if self._flow is None:
            self._combine(
                state,
                self._stages,
                np.ascontiguousarray(tableau.matrix[index, :index], dtype=np.float64),
                index,
                dz,
                self._propagator(fraction, bundle),
                self._noise_for(bundle),
                self._crop,
                self._plans.modal_spectrum,
                self._pad_scale,
            )
        else:
            self._flow_input(state, index, dz, bundle)
            flow_pad(
                self._flow.lab, self._crop, self._plans.modal_spectrum, self._pad_scale
            )
        if self._spec.family == "field_analytic":
            _double_positive_bins(self._plans.modal_spectrum, self._n_omega)
        if not self._nonlinear and not self._in_stage_absorber:
            # Nothing but the accumulation was compiled, so the derivative is
            # identically zero and the transforms would only add round-off. ADR-025's
            # in-stage absorber is a term of its own, so a problem with a collar and no
            # response at all still has a derivative and takes the path below. So is a
            # coupled flow's explicit residual, which the reference adds to a zero
            # source; a constant section's is identically zero and is not formed.
            if self._flow is None or not self._flow.residual:
                self._stages[index] = 0.0
                return
            self._plans.projected_spectrum[:] = 0.0
            self._flow_finish(self._plans.projected_spectrum, self._crop, index, bundle)
            return
        # The alarm is scored from what the *last* stage of an accepted step leaves in
        # the plan buffers, so it needs that stage's own scalars and not the step's.
        self._last_bundle = bundle
        self._modal_to_time(bundle)
        if self._nonlinear:
            self._polarisation_spectrum(bundle)
        else:
            # The absorber alone: no polarisation to transform, so the spectrum the
            # kernel multiplies by the prefactor is zero and only the term survives.
            self._plans.projected_spectrum[:] = 0.0
        if self._flow is not None:
            self._flow_finish(self._plans.projected_spectrum, self._crop, index, bundle)
            return
        self._finish_stage(
            self._plans.projected_spectrum,
            self._crop,
            self._prefactor_for(bundle),
            self._problem.grids.omega_window,
            self._propagator(fraction, bundle, forward=False),
            self._absorber_term(),
            float(tableau.solution_weights[index]),
            float(tableau.error_weights[index]),
            self._stages[index],
            self._solution,
            self._error,
        )

    def _polarisation_spectrum(self, bundle: StageScalars) -> None:
        """Run the nodes and transform the projected polarisation to frequency.

        SEP 6's dispersive response joins the spectrum after the stage's forward
        transform -- which it replaces when it is alone with buffers of its own -- and
        the envelope currents after both.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        """
        self._over_nodes(bundle)
        dispersive = self._dispersive
        held = self._field_plasma
        if (held is None or not held.radial) and (
            dispersive is None or not dispersive.replaces_transform
        ):
            self._plans.to_frequency.execute()
        if dispersive is not None:
            dispersive.combine()
        if self._envelope is not None:
            self._envelope.assemble(self._plans.projected_spectrum, self._crop)
        if held is not None and not held.radial:
            held.assemble(self._plans.projected_spectrum)

    def _modal_to_time(self, bundle: StageScalars) -> None:
        """Transform the padded modal spectrum to time, the dispersive fields with it.

        The dispersive response filters the spectrum first: the inverse transform
        destroys it, and the filtered fields are its multiples.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars, whose densities the dispersive response's
            filter is built from.
        """
        if self._dispersive is not None:
            self._dispersive.prepare(self._plans.modal_spectrum, bundle)
        self._plans.to_time.execute()

    def _stage_general(
        self,
        state: Complex[np.ndarray, "n_columns n_omega"],
        index: int,
        dz: float,
        bundle: StageScalars,
    ) -> None:
        """Evaluate one stage of a general modal guide into ``self._stages[index]``.

        The reference's ``_nonlinear_general`` on this backend's buffers: the stage
        input is combined once onto the **coarse** grid, which is where the general
        source synthesises, and the source's coarse spectrum goes through the same
        crop, prefactor, window, frame and accumulation kernel as every other route.
        A guide whose every region is linear skips the source, as the reference does,
        and the in-stage absorber is a term of its own: it takes the legacy
        combine and inverse transform of the modal columns, which is exactly what
        the reference's absorber reads.

        Parameters
        ----------
        state : numpy.ndarray
            The state at the start of the step.
        index : int
            Which stage of the tableau.
        dz : float
            Length of the whole step, m.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        """
        general = self._general
        assert general is not None
        tableau = self._problem.stepper.tableau
        fraction = float(tableau.nodes[index])
        weights = np.ascontiguousarray(tableau.matrix[index, :index], dtype=np.float64)
        propagator = self._propagator(fraction, bundle)
        drawn = self._noise_for(bundle)
        self._last_bundle = bundle
        flow = self._flow
        if flow is not None:
            self._flow_input(state, index, dz, bundle)
        if self._in_stage_absorber and flow is not None:
            flow_pad(flow.lab, self._crop, self._plans.modal_spectrum, self._pad_scale)
            self._plans.to_time.execute()
        elif self._in_stage_absorber:
            self._combine(
                state,
                self._stages,
                weights,
                index,
                dz,
                propagator,
                drawn,
                self._crop,
                self._plans.modal_spectrum,
                self._pad_scale,
            )
            self._plans.to_time.execute()
        if general.responds and flow is not None:
            flow_pad(flow.lab, self._coarse_index, general.lab, 1.0)
            spectrum = general.source(
                bundle.transform_weights, bundle.dispersive_kerr_densities
            )
        elif general.responds:
            self._combine(
                state,
                self._stages,
                weights,
                index,
                dz,
                propagator,
                drawn,
                self._coarse_index,
                general.lab,
                1.0,
            )
            spectrum = general.source(
                bundle.transform_weights, bundle.dispersive_kerr_densities
            )
        else:
            spectrum = general.output
            spectrum[:] = 0.0
        if flow is not None:
            self._flow_finish(spectrum, self._coarse_index, index, bundle)
            return
        self._finish_stage(
            spectrum,
            self._coarse_index,
            self._prefactor_for(bundle),
            self._problem.grids.omega_window,
            self._propagator(fraction, bundle, forward=False),
            self._absorber_term(),
            float(tableau.solution_weights[index]),
            float(tableau.error_weights[index]),
            self._stages[index],
            self._solution,
            self._error,
        )

    def _slot(self, fraction: float) -> int:
        """Return the propagator and flow-action slot of a stage fraction.

        Parameters
        ----------
        fraction : float
            The tableau node, dimensionless, in ``[0, 1]``.

        Returns
        -------
        int
            Its index in the step's distinct fractions.
        """
        return int(np.searchsorted(self._fractions, fraction))

    def _flow_input(
        self,
        state: Complex[np.ndarray, "n_columns n_omega"],
        index: int,
        dz: float,
        bundle: StageScalars,
    ) -> None:
        """Form one stage's lab-frame input through the forward flow action.

        The reference's ``stage_input`` of the flow-mixed state and combination
        (``NumpyRefBackend._trial_step``), into the flow's own buffer, which the pad
        and the explicit residual both read. No noise is added: a noise draw beside a
        coupled flow is refused.

        Parameters
        ----------
        state : numpy.ndarray
            The state at the start of the step.
        index : int
            Which stage of the tableau.
        dz : float
            Length of the whole step, m.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        """
        flow = self._flow
        assert flow is not None
        tableau = self._problem.stepper.tableau
        fraction = float(tableau.nodes[index])
        self._flow_combine(
            state,
            self._stages,
            np.ascontiguousarray(tableau.matrix[index, :index], dtype=np.float64),
            index,
            dz,
            self._propagator(fraction, bundle),
            flow.forward[self._slot(fraction)],
            flow.slots,
            fraction != 0.0,
            flow.lab,
        )

    def _flow_finish(
        self,
        spectrum: Complex[np.ndarray, "n_columns n_omega_out"],
        crop: np.ndarray,
        index: int,
        bundle: StageScalars,
    ) -> None:
        """Add the explicit residual, carry the source back by ``U^-1``, accumulate.

        Parameters
        ----------
        spectrum : numpy.ndarray
            The stage's nonlinear spectrum on the grid `crop` indexes, zero for a
            problem with no response.
        crop : numpy.ndarray
            Where each coarse bin sits in `spectrum`.
        index : int
            Which stage of the tableau.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars, whose ``flow_weights`` the residual is formed
            over.
        """
        flow = self._flow
        assert flow is not None
        tableau = self._problem.stepper.tableau
        fraction = float(tableau.nodes[index])
        flow.form_residual(bundle.flow_weights)
        flow.finish(
            spectrum,
            crop,
            self._prefactor_for(bundle),
            self._problem.grids.omega_window,
            self._propagator(fraction, bundle, forward=False),
            self._absorber_term(),
            flow.lab,
            flow.block,
            flow.residual_slots,
            flow.drive,
        )
        self._flow_accumulate(
            flow.drive,
            flow.inverse[self._slot(fraction)],
            flow.slots,
            fraction != 0.0,
            float(tableau.solution_weights[index]),
            float(tableau.error_weights[index]),
            self._stages[index],
            self._solution,
            self._error,
        )

    def _stage_free(  # noqa: PLR0912 - the opt-in pass timers' branches, free when off
        self, state: np.ndarray, index: int, dz: float, bundle: StageScalars
    ) -> None:
        """Execute one Cartesian stage with persistent FFTW plans.

        Parameters
        ----------
        state : numpy.ndarray
            Start-of-step spectral state, flattened by column.
        index : int
            Tableau stage index.
        dz : float
            Whole-step length in m.
        bundle : solprop.translate.stage.StageScalars
            Stage position and response scalars.
        """
        transverse, data = self._transverse, self._separable
        if transverse is None or data is None:
            msg = "Cartesian stage requires transverse plans and separable data"
            raise BackendError(msg)
        problem = self._problem
        geometry = problem.transform.geometry
        if geometry is None:
            msg = "Cartesian stage requires spatial geometry"
            raise BackendError(msg)
        tableau = problem.stepper.tableau
        fraction = float(tableau.nodes[index])
        forward = self._propagator(fraction, bundle)
        inverse = self._propagator(fraction, bundle, forward=False)
        scratch = transverse.spectrum.reshape(self._n_columns, self._n_omega)
        stage = self._stages[index]
        timers = self._pass_timers
        mark = timers.now() if timers is not None else 0.0
        stage[:] = 0
        if timers is not None:
            mark = timers.lap("rk.stage_clear", mark)
        self._last_bundle = bundle
        raman = (
            self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            if self._nonlinear
            else None
        )
        dispersive = self._spatial_dispersive
        if dispersive is not None and self._nonlinear:
            dispersive.begin(bundle)
        for material in (True, False):
            if (material and not self._nonlinear) or (
                not material and not self._in_stage_absorber
            ):
                continue
            if timers is not None:
                mark = timers.now()
            self._combine(
                state,
                self._stages,
                tableau.matrix[index, :index],
                index,
                dz,
                forward,
                self._noise_for(bundle),
                self._coarse_index,
                scratch,
                1.0 / problem.transform.n_nodes,
            )
            if timers is not None:
                mark = timers.lap("rk.input", mark)
            self._free_pass(bundle, material=material, raman=raman)
            if timers is not None:
                mark = timers.now()
            if material:
                self._spatial.finish_material(
                    data, scratch, problem.grids.omega_window, inverse, stage
                )
            else:
                np.multiply(scratch, problem.grids.active_mask, out=scratch)
                np.multiply(scratch, inverse, out=scratch)
                np.add(stage, scratch, out=stage)
            if timers is not None:
                mark = timers.lap("rk.finish", mark)
        if timers is not None:
            mark = timers.now()
        self._spatial.accumulate_stage(
            stage,
            (
                float(tableau.solution_weights[index]),
                float(tableau.error_weights[index]),
            ),
            self._solution,
            self._error,
        )
        if timers is not None:
            timers.lap("rk.accumulate", mark)

    def _free_pass(  # noqa: PLR0912 - the opt-in pass timers' branches, free when off
        self, bundle: StageScalars, *, material: bool, raman: np.ndarray | None
    ) -> None:
        """Take the input in the transverse scratch through one evaluation, in place.

        The shared core of a Cartesian or line evaluation (SEP 24 section 6.6), for
        either march: the transverse inverse transform, each node chunk to time, the
        material response or the in-stage absorber's rates, back to frequency, and
        the transverse forward transform. The scratch holds the input on entry and the
        source's raw spectrum on exit, before the prefactor or the band.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            Stage position and response scalars.
        material : bool
            Whether this is the material evaluation; otherwise the absorber's.
        raman : numpy.ndarray or None
            Cached Raman spectrum on the doubled temporal grid.
        """
        transverse, problem, plans = self._transverse, self._problem, self._plans
        geometry = problem.transform.geometry
        assert transverse is not None and geometry is not None
        scratch = transverse.spectrum.reshape(self._n_columns, self._n_omega)
        dispersive = self._spatial_dispersive
        timers = self._pass_timers
        mark = timers.now() if timers is not None else 0.0
        transverse.inverse.execute()
        if timers is not None:
            mark = timers.lap("free.transverse_inverse", mark)
        for first in range(0, problem.transform.n_nodes, self._chunk_nodes):
            last = min(first + self._chunk_nodes, problem.transform.n_nodes)
            rows = slice(first * self._n_pol_state, last * self._n_pol_state)
            count = (last - first) * self._n_pol_state
            self._spatial.pack_columns(
                scratch[rows],
                self._crop,
                plans.modal_spectrum,
                (problem.grids.pad_scale if material else 1.0) / self._n_to,
                self._spec.family == "field_analytic",
            )
            if timers is not None:
                mark = timers.lap("free.pack", mark)
            if material and dispersive is not None:
                dispersive.prepare(plans.modal_spectrum)
            plans.to_time.execute()
            if timers is not None:
                mark = timers.lap("free.to_time", mark)
            # The response column assigns every row it covers (the aliased
            # polarisation) and the absorber's rates do too; otherwise the rows are
            # cleared before the polarisation is copied in.
            if material and not self._aliased:
                plans.projected[:] = 0
            if timers is not None:
                mark = timers.lap("free.clear_projected", mark)
            if material:
                self._free_material(bundle, first, last, raman)
                if timers is not None:
                    mark = timers.now()
            else:
                samples = plans.modal_time[:count]
                if self._kind == "field":
                    samples = samples.real
                spatial_kernels.absorber_columns(
                    samples,
                    (geometry.absorber_rate, self._free_temporal_rate),
                    first,
                    self._n_pol_state,
                    plans.projected[:count],
                )
                if timers is not None:
                    mark = timers.lap("free.absorber", mark)
            self._transform_free_sources(material, scratch[rows])
            if timers is not None:
                mark = timers.now()
        transverse.forward.execute()
        if timers is not None:
            timers.lap("free.transverse_forward", mark)

    def _stage_slab(
        self, state: np.ndarray, index: int, dz: float, bundle: StageScalars
    ) -> None:
        """Execute one hybrid planar stage with persistent plans (kernel schema 15).

        The reference's ``_nonlinear_slab`` on this backend's buffers: the stage input
        is combined onto the ``(p, k_y)`` columns and taken to its ``(p, y)`` rows by
        the unnormalised inverse y transform; each block of whole y rows goes to time
        **once**, and from there the material source (synthesis in x, the response
        column, projection in x) and the in-stage absorber's term (the y collar's
        rate on each row plus the temporal one) each come back to frequency; both
        halves then return to columns through one forward y transform and the
        finish forms each column's term with its own prefactor. The x synthesis
        carries the temporal pad scale, so both passes pack with ``1/n_to`` alone.

        Parameters
        ----------
        state : numpy.ndarray
            Start-of-step spectral state, flattened by column.
        index : int
            Tableau stage index.
        dz : float
            Whole-step length in m.
        bundle : solprop.translate.stage.StageScalars
            Stage position and response scalars.
        """
        slab = self._slab_plans
        assert slab is not None
        problem = self._problem
        tableau = problem.stepper.tableau
        fraction = float(tableau.nodes[index])
        forward = self._propagator(fraction, bundle)
        inverse = self._propagator(fraction, bundle, forward=False)
        stage = self._stages[index]
        self._last_bundle = bundle
        material, absorber = self._nonlinear, self._in_stage_absorber
        if material or absorber:
            n_y = slab.spectrum.shape[1]
            self._combine(
                state,
                self._stages,
                tableau.matrix[index, :index],
                index,
                dz,
                forward,
                self._noise_for(bundle),
                self._coarse_index,
                slab.rows,
                1.0 / n_y,
            )
            self._slab_pass(bundle)
            self._planar.finish_stage(
                self._slab_data,
                slab.rows,
                slab.absorbed,
                problem.grids.omega_window,
                problem.grids.active_mask,
                inverse,
                stage,
                (material, absorber),
            )
        else:
            stage[:] = 0
        self._spatial.accumulate_stage(
            stage,
            (
                float(tableau.solution_weights[index]),
                float(tableau.error_weights[index]),
            ),
            self._solution,
            self._error,
        )

    def _slab_pass(self, bundle: StageScalars) -> None:  # noqa: PLR0912 - the opt-in pass timers' branches
        """Take the input in the planar column volume through one evaluation, in place.

        The shared core of a hybrid planar evaluation (SEP 24 section 6.6), for either
        march: the y inverse transform, each block of whole y rows to time once, the
        material source and the in-stage absorber's term each back to frequency into
        its half of the volume, and one forward y transform of both halves.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            Stage position and response scalars.
        """
        slab, plans = self._slab_plans, self._plans
        assert slab is not None
        material, absorber = self._nonlinear, self._in_stage_absorber
        n_y = slab.spectrum.shape[1]
        timers = self._pass_timers
        mark = timers.now() if timers is not None else 0.0
        slab.inverse.execute()
        if timers is not None:
            timers.lap("slab.transverse_inverse", mark)
        raman = (
            self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            if material
            else None
        )
        dispersive = self._spatial_dispersive if material else None
        if dispersive is not None:
            dispersive.begin(bundle)
        for first in range(0, n_y, self._slab_block):
            count = min(self._slab_block, n_y - first)
            rows = slice(
                first * self._slab_rows_per_y,
                (first + count) * self._slab_rows_per_y,
            )
            width = rows.stop - rows.start
            if timers is not None:
                mark = timers.now()
            spatial_kernels.pack_columns(
                slab.rows[rows],
                self._crop,
                plans.modal_spectrum,
                1.0 / self._n_to,
                self._spec.family == "field_analytic",
            )
            if timers is not None:
                mark = timers.lap("slab.pack", mark)
            if dispersive is not None:
                dispersive.prepare(plans.modal_spectrum)
            plans.to_time.execute()
            if timers is not None:
                mark = timers.lap("slab.to_time", mark)
            if material:
                self._slab_material(bundle, first, count, raman)
                if timers is not None:
                    mark = timers.lap("slab.material", mark)
                self._slab_output(width, slab.rows[rows], material=True)
                if timers is not None:
                    mark = timers.lap("slab.output", mark)
            if absorber:
                samples = plans.modal_time[:width]
                if self._kind == "field":
                    samples = samples.real
                spatial_kernels.absorber_columns(
                    samples,
                    (self._slab_row_rate, self._free_temporal_rate),
                    rows.start // self._n_pol_state,
                    self._n_pol_state,
                    plans.projected[:width],
                )
                self._slab_output(width, slab.absorbed[rows], material=False)
                if timers is not None:
                    mark = timers.lap("slab.absorber", mark)
        if timers is not None:
            mark = timers.now()
        slab.forward_all.execute()
        if timers is not None:
            timers.lap("slab.transverse_forward", mark)

    def _slab_material(
        self,
        bundle: StageScalars,
        first: int,
        count: int,
        raman: np.ndarray | None,
    ) -> None:
        """Synthesise one block's nodes in time, respond, and project back to rows.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            Stage response scalars.
        first, count : int
            The block's first y row and its number of y rows.
        raman : numpy.ndarray or None
            Cached Raman spectrum on the doubled temporal grid.
        """
        plans, dispersive = self._plans, self._spatial_dispersive
        width = count * self._slab_rows_per_y
        n_x = self._chunk_nodes // self._slab_block
        size = count * n_x * self._n_pol_node
        self._slab_product(
            self._synthesis, plans.modal_time[:width], self._nodes[:size], count
        )
        if dispersive is None or not dispersive.layout.alone:
            self._respond_chunk(bundle, first * n_x, (first + count) * n_x, raman)
            self._slab_product(
                self._projection,
                self._polarisation[:size],
                plans.projected[:width],
                count,
            )
        if dispersive is not None:
            dispersive.respond(self._nodes, count)
            dispersive.finish()
        if self._envelope is not None:
            for stream in range(2):
                self._slab_product(
                    self._projection,
                    self._envelope.sources[stream, :size],
                    self._envelope.projected[stream, :width],
                    count,
                )
            self._envelope.projected[:, width:] = 0
        if self._field_plasma is not None:
            for stream in range(2):
                self._slab_product(
                    self._projection,
                    self._field_plasma.sources[stream, :size],
                    self._field_plasma.projected[stream, :width],
                    count,
                )
            self._field_plasma.projected[:, width:] = 0

    @staticmethod
    def _slab_product(
        matrix: Float[np.ndarray, "n_out n_in"],
        source: np.ndarray,
        target: np.ndarray,
        count: int,
    ) -> None:
        """Apply a real slab matrix in x to every y row of a block, in time.

        One real GEMM per y row, batched: ``(n_out, n_in) @ (n_in, n_pol * n_to)``,
        a complex block taken as its interleaved real view, as the radial real GEMM
        takes it, so no product with a zero imaginary part is formed and nothing is
        allocated.

        Parameters
        ----------
        matrix : numpy.ndarray
            The synthesis ``(n_x, n_p)`` or the projection ``(n_p, n_x)``, real.
        source : numpy.ndarray
            ``count * n_in * n_pol`` rows of fine time samples, contiguous.
        target : numpy.ndarray
            ``count * n_out * n_pol`` rows, overwritten.
        count : int
            The block's y rows.
        """
        if np.iscomplexobj(source):
            source, target = source.view(np.float64), target.view(np.float64)
        np.matmul(
            matrix,
            source.reshape(count, matrix.shape[1], -1),
            out=target.reshape(count, matrix.shape[0], -1),
        )

    def _chunk_to_frequency(self, material: bool) -> None:
        """Transform a spatial chunk's polarisation, the dispersive response with it.

        Parameters
        ----------
        material : bool
            Whether this is the material pass, which the response belongs to.
        """
        dispersive = self._spatial_dispersive if material else None
        if dispersive is None or not dispersive.replaces_transform:
            self._plans.to_frequency.execute()
        if dispersive is not None:
            dispersive.combine()

    def _slab_output(self, width: int, target: np.ndarray, *, material: bool) -> None:
        """Take one block's rows back to frequency and crop them into place.

        Parameters
        ----------
        width : int
            Rows of the plan buffer this block filled.
        target : numpy.ndarray
            The block's rows of the material or the absorber half, overwritten.
        material : bool
            Whether the source includes the envelope plasma's current streams.
        """
        plans = self._plans
        # A final partial block leaves rows the transform still reads; they are
        # cleared so that what it reads is the block's own and finite.
        plans.projected[width:] = 0
        self._chunk_to_frequency(material)
        if material and self._envelope is not None:
            self._envelope.assemble(plans.projected_spectrum, self._crop)
        if material and self._field_plasma is not None:
            self._field_plasma.assemble(plans.projected_spectrum)
        spatial_kernels.crop_columns(plans.projected_spectrum, self._crop, target)

    def _free_material(
        self, bundle: StageScalars, first: int, last: int, raman: np.ndarray | None
    ) -> None:
        """Evaluate one free-space chunk's material polarisation in time.

        The response column on the chunk's columns, which are its nodes, and SEP 6's
        dispersive products beside it -- or alone, without the column.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            Stage response scalars.
        first, last : int
            The chunk's node interval.
        raman : numpy.ndarray or None
            Cached Raman spectrum on the doubled temporal grid.
        """
        plans, dispersive = self._plans, self._spatial_dispersive
        count = (last - first) * self._n_pol_state
        timers = self._pass_timers
        mark = timers.now() if timers is not None else 0.0
        if dispersive is None or not dispersive.layout.alone:
            if not self._aliased:
                self._nodes[:count] = plans.modal_time[:count]
            if timers is not None:
                mark = timers.lap("free.node_copy", mark)
            self._respond_chunk(bundle, first, last, raman)
            if timers is not None:
                mark = timers.lap("free.response", mark)
            if not self._aliased:
                plans.projected[:count] = self._polarisation[:count]
            if timers is not None:
                mark = timers.lap("free.polarisation_copy", mark)
        if dispersive is not None:
            dispersive.respond(plans.modal_time)
            dispersive.finish()
            if timers is not None:
                timers.lap("free.dispersive", mark)

    def _transform_free_sources(self, material: bool, output: np.ndarray) -> None:
        """Transform one material or absorber chunk using its prebuilt plans.

        Parameters
        ----------
        material : bool
            Whether the source includes the material's envelope plasma streams and
            SEP 6's dispersive response.
        output : numpy.ndarray
            Active rows of the physical-position coarse spectrum.
        """
        plans = self._plans
        timers = self._pass_timers
        mark = timers.now() if timers is not None else 0.0
        self._chunk_to_frequency(material)
        if timers is not None:
            mark = timers.lap("free.to_frequency", mark)
        if material and self._envelope is not None:
            # Fixed-width zero padding preserves the partial-chunk plan shape.
            self._envelope.projected[:] = self._envelope.sources
            self._envelope.assemble(plans.projected_spectrum, self._crop)
            if timers is not None:
                mark = timers.lap("free.plasma_streams", mark)
        if material and self._field_plasma is not None:
            self._field_plasma.projected[:] = self._field_plasma.sources
            self._field_plasma.assemble(plans.projected_spectrum)
            if timers is not None:
                mark = timers.lap("free.plasma_streams", mark)
        self._spatial.crop_columns(plans.projected_spectrum, self._crop, output)
        if timers is not None:
            timers.lap("free.crop", mark)

    def _over_nodes(self, bundle: StageScalars) -> None:
        """Run passes 3 to 5: synthesise, the fused response column, project.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars: every scalar path the response column reads.
        """
        problem = self._problem
        transform = problem.transform
        taper = bundle.taper_scale
        np.multiply(self._synthesis, taper, out=self._scaled_synthesis)
        np.multiply(self._projection, 1.0 / taper, out=self._scaled_projection)
        if self._embedded is not None and self._scaled_embedded is not None:
            np.multiply(self._embedded, 1.0 / taper, out=self._scaled_embedded)
        for packed, scaled in zip(
            self._projection_chunks, self._scaled_projection_chunks, strict=True
        ):
            np.multiply(packed, 1.0 / taper, out=scaled)
        dispersive = self._dispersive
        alone = dispersive is not None and dispersive.layout.alone
        # Alone and symmetric, the response reads only its filtered field, which it
        # synthesises itself into the node buffer.
        bare = not (alone and dispersive is not None and dispersive.symmetric)
        if dispersive is not None:
            dispersive.begin(bundle, taper)
        # Once per stage rather than once per chunk: the kernel spectrum does not
        # depend on the nodes.
        raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        modal = self._plans.modal_time.reshape(
            transform.n_modes, self._n_pol_state * self._n_to
        )
        if self._real_radial:
            # The raw QDHT is real. Its action on interleaved real/imaginary
            # columns is the same scalar transform for every polarisation, with
            # no packing allocation or extra products with zero imaginary parts.
            modal = modal.view(np.float64)
        for first in range(0, transform.n_nodes, self._chunk_nodes):
            last = min(first + self._chunk_nodes, transform.n_nodes)
            rows = slice(first * transform.n_pol, last * transform.n_pol)
            count = (last - first) * self._n_pol_node
            nodes = self._nodes[:count]
            target = nodes.reshape(rows.stop - rows.start, -1)
            if self._real_radial:
                target = target.view(np.float64)
            if bare:
                np.dot(self._scaled_synthesis[rows], modal, out=target)
            if not alone:
                self._respond_chunk(bundle, first, last, raman)
            if dispersive is not None:
                dispersive.respond_chunk(first // self._chunk_nodes)
            if alone:
                continue
            self._project_sources(rows, count, first == 0)
        if dispersive is not None:
            dispersive.finish()

    def _project_sources(self, rows: slice, count: int, first: bool) -> None:
        """Project a chunk in the order required by its plasma formulation.

        Parameters
        ----------
        rows : slice
            Physical-node projection columns for this chunk.
        count : int
            Active node-component rows.
        first : bool
            Whether this chunk overwrites the stage's accumulation buffers.
        """
        held = self._field_plasma
        if held is not None and held.radial:
            held.project_radial(
                self._polarisation,
                self._scaled_projection[:, rows],
                self._plans.projected_spectrum,
                first=first,
                count=count,
            )
        else:
            self._project(rows, count, first)
            if held is not None:
                held.project(
                    self._scaled_projection[:, rows],
                    None
                    if self._scaled_embedded is None
                    else self._scaled_embedded[:, rows],
                    self._problem.transform.n_modes,
                    first=first,
                    count=count,
                )
        if self._envelope is not None:
            self._envelope.project(
                self._scaled_projection[:, rows],
                None
                if self._scaled_embedded is None
                else self._scaled_embedded[:, rows],
                self._problem.transform.n_modes,
                first=first,
                count=count,
            )

    def _respond_chunk(
        self, bundle: StageScalars, first: int, last: int, raman: np.ndarray | None
    ) -> None:
        """Evaluate the common response column and optional Raman convolution.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            Stage response scalars.
        first, last : int
            Physical node interval occupied by the current chunk.
        raman : numpy.ndarray or None
            Cached Raman spectrum on the doubled temporal grid.
        """
        responses = self._problem.responses
        kerr, plasma = responses.kerr_coefficients, responses.plasma_coefficients
        density = (
            bundle.plasma_densities[0] if bundle.plasma_densities else bundle.density
        )
        preionisation = bundle.preionisation
        multiple = self._spec.n_channels > 1
        if multiple:
            self._channel_scalars[2] = bundle.plasma_densities
            self._channel_scalars[3] = bundle.plasma_preionisation
        count = (last - first) * self._n_pol_node
        nodes = self._nodes[:count]
        if self._band_column is not None:
            self._band_respond(nodes, count, first, last, density, preionisation)
        self._column(
            nodes,
            self._polarisation[:count],
            self._plans.drive if self._spec.raman else self._polarisation.real[:1],
            self._problem.grids.t_window_fine,
            self._rate,
            self._carrier,
            (kerr.coefficient * bundle.kerr_density) if kerr is not None else 0.0,
            kerr.self_factor if kerr is not None else 0.0,
            kerr.conjugate_factor if kerr is not None else 0.0,
            kerr.harmonic_factor if kerr is not None else 0.0,
            self._channel_scalars[0]
            if multiple
            else plasma.charge_mass_ratio
            if plasma is not None
            else 0.0,
            self._channel_scalars[1]
            if multiple
            else plasma.ionisation_potential
            if plasma is not None
            else 0.0,
            self._channel_scalars[2] if multiple else density,
            self._channel_scalars[3] if multiple else preionisation,
            self._problem.grids.dt_fine,
            self._stash[first:last] if self._spec.stash else self._stash,
            self._channel_scratch,
            None if self._field_plasma is None else self._field_plasma.current[:count],
        )
        if self._field_plasma is not None:
            self._field_plasma.products(
                self._problem.grids.t_window_fine,
                self._problem.grids.t_window_fine_derivative,
                count,
            )
        if raman is not None:
            self._convolve(
                last - first,
                raman,
                bundle.raman_densities[0] if self._spec.n_raman == 1 else 1.0,
            )
        if self._envelope is not None:
            self._envelope.respond(
                nodes,
                bundle,
                self._problem.grids.dt_fine,
                self._problem.grids.t_window_fine,
            )

    def _band_respond(
        self,
        nodes: Num[np.ndarray, "n_rows n_to"],
        count: int,
        first: int,
        last: int,
        density: float,
        preionisation: float,
    ) -> None:
        """Write the multipass column's physical current for one chunk.

        The fused column separately writes the collared Kerr and Raman terms.

        Parameters
        ----------
        nodes : numpy.ndarray
            The chunk's field on the fine grid.
        count : int
            The chunk's rows.
        first, last : int
            The chunk's node interval.
        density, preionisation : float
            The one channel's stage scalars; several channels read theirs from
            ``_channel_scalars``.
        """
        assert self._band_column is not None
        if self._spec.n_channels > 1:
            constants = tuple(self._channel_scalars)
        else:
            plasma = self._problem.responses.plasma_coefficients
            assert plasma is not None
            constants = (
                plasma.charge_mass_ratio,
                plasma.ionisation_potential,
                density,
                preionisation,
            )
        assert self._field_plasma is not None
        self._band_column(
            nodes,
            self._field_plasma.current[:count],
            self._rate,
            self._taps,
            *constants,
            self._problem.grids.dt_fine,
            self._stash[first:last] if self._spec.stash else self._stash,
            self._plasma_work[: last - first],
        )

    def _project(self, rows: slice, count: int, firstly: bool) -> None:
        """Accumulate one chunk's polarisation back onto the modal columns.

        Parameters
        ----------
        rows : slice
            The chunk's columns of the projection matrix.
        count : int
            Rows of the node buffer this chunk filled.
        firstly : bool
            Whether this is the first chunk, which writes rather than adds.
        """
        source = self._polarisation[:count].reshape(
            rows.stop - rows.start, self._n_pol_state * self._n_to
        )
        if self._stacked_modal_projection:
            self._project_stacked(rows, source, firstly)
            return
        target = self._plans.projected if firstly else self._chunk_projection
        target = target.reshape(self._problem.transform.n_modes, -1)
        if self._real_radial:
            source = source.view(np.float64)
            target = target.view(np.float64)
        np.dot(
            self._scaled_projection[:, rows],
            source,
            out=target,
        )
        if not firstly:
            np.add(
                self._plans.projected,
                self._chunk_projection,
                out=self._plans.projected,
            )
        if self._scaled_embedded is None or self._plans.coarse_projected is None:
            return
        coarse = self._plans.coarse_projected if firstly else self._chunk_projection
        np.dot(
            self._scaled_embedded[:, rows],
            source,
            out=coarse.reshape(self._problem.transform.n_modes, -1),
        )
        if not firstly:
            np.add(
                self._plans.coarse_projected,
                self._chunk_projection,
                out=self._plans.coarse_projected,
            )

    def _project_stacked(self, rows: slice, source: np.ndarray, firstly: bool) -> None:
        """Project both rules together without changing chunk accumulation order.

        Parameters
        ----------
        rows : slice
            Transform rows occupied by this physical node chunk.
        source : numpy.ndarray
            Real polarisation, shape ``(n_chunk_rows, n_pol * n_to)``.
        firstly : bool
            Whether this chunk overwrites the plan inputs instead of accumulating.
        """
        transform = self._problem.transform
        chunk = rows.start // (self._chunk_nodes * transform.n_pol)
        np.dot(
            self._scaled_projection_chunks[chunk],
            source,
            out=self._projection_stacked.reshape(2 * transform.n_modes, -1),
        )
        fine = self._projection_stacked[: self._n_columns]
        coarse = self._projection_stacked[self._n_columns :]
        assert self._plans.coarse_projected is not None
        if firstly:
            np.copyto(self._plans.projected, fine)
            np.copyto(self._plans.coarse_projected, coarse)
        else:
            np.add(self._plans.projected, fine, out=self._plans.projected)
            np.add(
                self._plans.coarse_projected, coarse, out=self._plans.coarse_projected
            )

    def _convolve(
        self,
        count: int,
        spectrum: Complex[np.ndarray, "... n_omega_doubled"],
        density: float,
    ) -> None:
        """Convolve one chunk and add the delayed term it makes to the total.

        Parameters
        ----------
        count : int
            Nodes in this chunk.
        spectrum : numpy.ndarray
            The damped kernel's spectrum on the doubled grid.
        density : float
            The density path at this position, m^-3.
        """
        plans = self._plans
        forward, inverse = plans.raman_to_frequency, plans.raman_to_time
        drive_spectrum = plans.drive_spectrum
        if count == self._raman_tail_nodes:
            forward, inverse = plans.raman_tail_to_frequency, plans.raman_tail_to_time
            drive_spectrum = drive_spectrum[: count * self._spec.raman_drives]
        forward.execute()
        if self._n_pol_node == 2:
            kernels.scale_tensor_spectrum(drive_spectrum, spectrum, self._raman_scale)
        else:
            kernels.scale_spectrum(drive_spectrum, spectrum, self._raman_scale)
        inverse.execute()
        self._raman_finish(
            self._polarisation[: count * self._n_pol_node],
            self._nodes[: count * self._n_pol_node],
            self._plans.convolved[: count * self._spec.raman_drives, : self._n_to],
            self._problem.grids.t_window_fine,
            EPS0 * density,
        )

    def _finish(
        self,
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        dz: float,
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Apply the once-per-accepted-step boundary treatments to a kept step.

        Parameters
        ----------
        state : numpy.ndarray
            The state at the end of the step.
        dz : float
            Length of the step, m.

        Returns
        -------
        numpy.ndarray
            The state with the hard band mask and, when one is present, the absorber,
            which the mask follows.
        """
        flat = state.reshape(self._n_columns, self._n_omega)
        kernels.apply_hard_mask(flat, self._problem.grids.active_mask, flat)
        responses = self._problem.responses
        rate = responses.temporal_absorber
        if rate is None or responses.absorber_scheme != "split_step":
            # Either there is no collar, or ADR-025's other scheme applied it inside
            # every stage, where it is already in the accepted state.
            return state
        if self._slab:
            self._finish_slab(flat, rate, dz)
            return state
        if self._free:
            physical = self._free_physical(state[0])
            plans = self._plans
            damping = np.exp(-0.5 * dz * rate)
            for first in range(0, len(physical), self._chunk_nodes):
                last = min(first + self._chunk_nodes, len(physical))
                count = (last - first) * self._n_pol_state
                plans.coarse_spectrum[:] = 0
                np.multiply(
                    physical[first:last].reshape(count, self._n_omega),
                    1.0 / self._problem.grids.n_t,
                    out=plans.coarse_spectrum[:count],
                )
                plans.absorber_to_time.execute()
                np.multiply(plans.coarse_time, damping, out=plans.coarse_time)
                plans.absorber_to_frequency.execute()
                physical[first:last] = plans.coarse_spectrum[:count].reshape(
                    last - first, self._n_pol_state, self._n_omega
                )
            assert self._transverse is not None
            self._transverse.forward.execute()
            kernels.apply_hard_mask(
                physical.reshape(self._n_columns, self._n_omega),
                self._problem.grids.active_mask,
                flat,
            )
            return state
        # The product in time spreads content across every bin, the DC among them, so
        # the band is imposed again after it (the reference's ``_finish``).
        absorbed = passes.apply_temporal_rate(state, rate, dz, kind=self._kind)
        columns = absorbed.reshape(self._n_columns, self._n_omega)
        kernels.apply_hard_mask(columns, self._problem.grids.active_mask, columns)
        return absorbed

    def _finish_slab(
        self,
        flat: Complex[np.ndarray, "n_columns n_omega"],
        rate: Float[np.ndarray, " n_t"],
        dz: float,
    ) -> None:
        """Apply the split-step temporal absorber to a planar state, row by row.

        A ``(p, y)`` row is a modal amplitude whose time signal is a real field's, and
        the rate is the same at every node, so the exact accepted-step factor acts on
        the rows (the reference's ``_finish``): to the rows, the coarse pair per chunk
        of rows, back to the columns, and the hard mask.

        Parameters
        ----------
        flat : numpy.ndarray
            The masked state, ``(n_columns, n_omega)``, updated in place.
        rate : numpy.ndarray
            The collar's power rate on the coarse time grid, 1/m.
        dz : float
            Length of the accepted step, m.
        """
        slab, plans = self._slab_plans, self._plans
        assert slab is not None and plans.coarse_spectrum is not None
        np.multiply(flat, 1.0 / slab.spectrum.shape[1], out=slab.rows)
        slab.inverse.execute()
        damping = np.exp(-0.5 * dz * rate)
        chunk = plans.coarse_spectrum.shape[0]
        for first in range(0, self._n_columns, chunk):
            last = min(first + chunk, self._n_columns)
            count = last - first
            plans.coarse_spectrum[:] = 0
            np.multiply(
                slab.rows[first:last],
                1.0 / self._problem.grids.n_t,
                out=plans.coarse_spectrum[:count],
            )
            plans.absorber_to_time.execute()
            np.multiply(plans.coarse_time, damping, out=plans.coarse_time)
            plans.absorber_to_frequency.execute()
            slab.rows[first:last] = plans.coarse_spectrum[:count]
        slab.forward.execute()
        kernels.apply_hard_mask(slab.rows, self._problem.grids.active_mask, flat)

    # --- the tables the stage reads ---------------------------------------------

    @property
    def _nonlinear(self) -> bool:
        """Whether anything but the accumulation was compiled into the responses.

        Returns
        -------
        bool
            False when the nonlinear term is identically zero, so that the step is the
            exact linear flow and the transforms would only add round-off.
        """
        return len(self._problem.responses.response_order) > 1

    def _prepare_propagators(self, z: float, dz: float) -> None:
        """Form ``exp(+-dPhi)`` for this step's stage fractions.

        Tabulating is the measured default (``docs/dev/backends.md`` section 3.6: whole
        steps of 17.8 ms against 21.7 on the RDW case), and the alternative is kept
        behind the contract's own flag rather than removed. Either way the pair comes
        from one real ``exp`` and one ``sincos``, where the reference evaluates two
        complex exponentials per stage.

        Parameters
        ----------
        z : float
            Position at the start of the step, m.
        dz : float
            Length of the step, m.
        """
        if not self._tabulating:
            self._recomputed = math.nan
            return
        for index, fraction in enumerate(self._fractions):
            self._write_propagator(
                self._stage_scalars(z, z + float(fraction) * dz), index
            )

    def _write_propagator(self, bundle: StageScalars, slot: int) -> None:
        """Evaluate one stage's exponent and exponentiate it into a slot.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars, whose weights the terms are combined over.
        slot : int
            Which row of the propagator tables to write.
        """
        if self._slab:
            self._planar.write_propagator(
                self._slab_data,
                bundle.z1 - bundle.z0,
                self._forward[slot],
                self._inverse[slot],
            )
            return
        if self._separable is not None:
            self._spatial.write_propagator(
                self._separable,
                bundle.z1 - bundle.z0,
                self._forward[slot],
                self._inverse[slot],
            )
            return
        exponent = self._delta_phi(bundle)
        self._exponent[:] = exponent.reshape(self._n_columns, self._n_omega)
        kernels.exponentiate(self._exponent, self._forward[slot], self._inverse[slot])

    def _propagator(
        self, fraction: float, bundle: StageScalars, *, forward: bool = True
    ) -> Complex[np.ndarray, "n_columns n_omega"]:
        """Return ``exp(+-dPhi)`` at one stage of the current step.

        Parameters
        ----------
        fraction : float
            The stage fraction, which indexes the table when tabulating.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars, needed only when recomputing.
        forward : bool, optional
            Whether the forward sign is wanted. True by default.

        Returns
        -------
        numpy.ndarray
            The propagator, in the table when tabulating and in the scratch pair when
            recomputing.
        """
        if self._tabulating:
            slot = int(np.searchsorted(self._fractions, fraction))
        else:
            slot = 0
            if fraction != self._recomputed:
                self._write_propagator(bundle, 0)
                self._recomputed = fraction
        return self._forward[slot] if forward else self._inverse[slot]

    def _delta_phi(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the stage's exponent, compiled where the route allows it.

        The general z-table route reads persistent host arrays, and a separable route
        evaluates its element formula in a compiled pass; anything else combines the
        terms on the host.

        Parameters
        ----------
        bundle : StageScalars
            Stage positions, weights and labelled response scalars.

        Returns
        -------
        numpy.ndarray
            Cumulative phase difference, dimensionless, in the canonical state layout.
        """
        if self._host_phi is not None:
            return self._host_phi.delta(bundle.z0, bundle.z1)
        if self._separable is not None and not self._slab:
            # The exponent the propagator kernel exponentiates, element by element,
            # rather than the host's separable volume, formed in NumPy at every trial.
            exponent = np.empty(self._problem.shape, dtype=np.complex128)
            self._spatial.write_exponent(
                self._separable,
                bundle.z1 - bundle.z0,
                exponent.reshape(self._n_columns, self._n_omega),
            )
            return exponent
        return super()._delta_phi(bundle)

    def _prefactor(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Evaluate a z-table prefactor on the host without stage array allocation.

        Parameters
        ----------
        bundle : StageScalars
            Stage positions, weights and labelled response scalars.

        Returns
        -------
        numpy.ndarray
            UPPE prefactor, shape (n_modes, n_pol, n_omega), in m/(V s).
        """
        if self._host_prefactor is not None:
            return self._host_prefactor.at(bundle.z1)
        return super()._prefactor(bundle)

    def _prefactor_for(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_columns n_omega"]:
        """Return the UPPE prefactor at a stage, as the kernel takes it.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        numpy.ndarray
            The prefactor times the crop scale, shape ``(n_columns, n_omega)``.
        """
        values = self._prefactor(bundle)
        return np.ascontiguousarray(
            np.broadcast_to(
                values.reshape(-1, self._n_omega), (self._n_columns, self._n_omega)
            )
        )

    def _absorber_term(self) -> Complex[np.ndarray, "n_columns n_omega"]:
        """Return the in-stage absorber term, or a buffer the kernel ignores.

        The modal field on the fine time grid is still in the plan's own buffer when
        this is called -- the forward transform of the *polarisation* does not destroy
        it -- so the term costs one product over ``(n_columns, n_to)`` (the radial
        collar's real GEMM, the temporal rate's multiply, or both) and one more planned
        forward transform of the same columns, into a spectrum of its own because the
        finish still reads the polarisation's. It is cropped and masked to the active
        band, as :func:`solprop.backends.numpy_ref.passes.absorber_term` is, into a
        buffer held for the run; nothing is allocated.

        Returns
        -------
        numpy.ndarray
            The term on the coarse grid, shape ``(n_columns, n_omega)``. When the
            scheme is not in use the kernel was compiled without the branch and never
            reads what comes back.
        """
        if not self._in_stage_absorber:
            return self._exponent
        plans, responses = self._plans, self._problem.responses
        rate = (
            responses.temporal_absorber_fine
            if responses.absorber_scheme == "nonlinear"
            else None
        )
        samples, term = plans.modal_time, plans.in_stage_time
        # The analytic family stores a complex analytic signal, while the field
        # equation's collars act on its real electric field.
        if self._spec.family == "field_analytic":
            samples = samples.real
        if self._spatial_absorber.size:
            if self._spatial_absorber_time.size:
                np.copyto(self._spatial_absorber_time, samples)
                samples = self._spatial_absorber_time
            self._radial_absorber_product(samples, term)
            if rate is not None:
                kernels.add_temporal_absorber(samples, rate, term)
        else:
            # The non-Cartesian stage reaches this path only for a temporal rate:
            # a radial rate took the branch above, an inactive absorber returned.
            assert rate is not None
            kernels.write_temporal_absorber(samples, rate, term)
        plans.in_stage_to_frequency.execute()
        spatial_kernels.crop_columns(
            plans.in_stage_spectrum, self._crop, self._absorber_spectrum
        )
        kernels.apply_hard_mask(
            self._absorber_spectrum,
            self._problem.grids.active_mask,
            self._absorber_spectrum,
        )
        return self._absorber_spectrum

    def _radial_absorber_product(
        self,
        samples: Num[np.ndarray, "n_columns n_to"],
        term: Num[np.ndarray, "n_columns n_to"],
    ) -> None:
        """Apply the composed radial collar to every mode's samples, into ``term``.

        The matrix is real (``translate/radial.py`` builds the Hankel pair real), so a
        complex row is multiplied through its interleaved real view: one real GEMM, no
        cast of the matrix to complex and none of the operand.

        Parameters
        ----------
        samples : numpy.ndarray
            Contiguous fine time samples, real or complex, shape
            ``(n_columns, n_to)`` with ``n_columns = n_r * n_pol``.
        term : numpy.ndarray
            The planned forward transform's input, of the same shape and type,
            overwritten.
        """
        if np.iscomplexobj(samples):
            samples, term = samples.view(np.float64), term.view(np.float64)
        modes = self._spatial_absorber.shape[0]
        np.matmul(
            self._spatial_absorber,
            samples.reshape(modes, -1),
            out=term.reshape(modes, -1),
        )

    def _free_physical(self, field: np.ndarray) -> np.ndarray:
        """Reconstruct physical columns into the persistent transverse scratch.

        Parameters
        ----------
        field : numpy.ndarray
            Spatial spectrum, shape (n_modes, n_pol, n_omega).

        Returns
        -------
        numpy.ndarray
            Borrowed physical-space spectrum with the same shape.
        """
        transverse = self._transverse
        assert transverse is not None
        scratch = transverse.spectrum.reshape(field.shape)
        np.multiply(field, 1.0 / self._problem.transform.n_nodes, out=scratch)
        transverse.inverse.execute()
        return scratch

    def _unitary_phase(self, bundle: StageScalars) -> np.ndarray:
        """Return the noise phase, evaluating separable data into existing scratch.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            Stage position and response scalars.

        Returns
        -------
        numpy.ndarray
            Phase in radians with the state shape.
        """
        if self._separable is None:
            return super()._unitary_phase(bundle)
        # The low-storage march holds no exponent: an observation's phase is formed
        # in a transient (its kernels form the draw's phase per element).
        phase = (
            self._exponent.real
            if self._exponent.size
            else np.empty((self._n_columns, self._n_omega))
        )
        self._write_phase(bundle.z1, phase)
        return phase.reshape(self._problem.shape)

    def _write_phase(
        self, z: float, phase: Float[np.ndarray, "n_columns n_omega"]
    ) -> None:
        """Evaluate the separable noise phase at ``z`` into a real buffer.

        Parameters
        ----------
        z : float
            Absolute propagation position in m.
        phase : numpy.ndarray
            Overwritten with the phase, radians.
        """
        if self._slab:
            planar_kernels.write_phase(self._slab_data, z, phase)
        else:
            spatial_kernels.write_phase(self._separable, z, phase)

    def _noise_for(
        self, bundle: StageScalars
    ) -> Complex[np.ndarray, "n_columns n_omega"]:
        """Return the advanced noise a stage adds, or a buffer the kernel ignores.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        numpy.ndarray
            The noise in the kernel's layout. When no noise model is active the kernel
            was compiled without the branch and never reads it.
        """
        if self._separable is not None:
            if self._noise is not None:
                self._write_phase(bundle.z1, self._exponent.real)
                spatial_kernels.rotate_noise(
                    self._noise.reshape(self._n_columns, self._n_omega),
                    self._exponent.real,
                    self._exponent,
                )
            return self._exponent
        drawn = self._noise_at(bundle)
        if drawn is None:
            return self._exponent
        return drawn.reshape(self._n_columns, self._n_omega)

    def _raman_spectrum(
        self, rates: tuple[float, ...], densities: tuple[float, ...] = ()
    ) -> Complex[np.ndarray, "... n_omega_doubled"] | None:
        """Return the damped Raman kernel spectrum for a set of rates, anew.

        Memoised on the **decay rates** rather than on the density, because the rates
        are what the spectrum depends on: a glass's are zero at every density, so its
        kernel is transformed once for the whole run, and a molecular gas on a gradient
        presents new rates at every stage and is transformed once per stage. Only the
        most recent set is kept, which is the right choice here -- a growing cache would
        hold thousands of spectra and hit none.

        Parameters
        ----------
        rates : tuple of float
            The decay rate of each term at the stage's position, s^-1, off the bundle.

        densities : tuple of float, optional
            Component amplitude densities in m^-3, in Raman-component order. Multiple
            components include every amplitude in the cache key; one keeps density
            outside the convolution and retains the original arithmetic.

        Returns
        -------
        numpy.ndarray or None
            The spectrum of shape ``(n_to + 1,)``, or ``None`` when no Raman kernel was
            compiled.
        """
        components = self._problem.responses.raman_terms
        if not components:
            return None
        amplitudes = densities if len(components) > 1 else ()
        if (
            self._raman_kernel is None
            or self._raman_rates != rates
            or self._raman_densities != amplitudes
        ):
            if self._n_pol_node == 2:
                self._raman_kernel = passes.tensor_raman_spectrum(
                    components,
                    rates,
                    densities if len(components) > 1 else (1.0,),
                    self._problem.grids.dt_fine,
                )
            elif len(components) == 1:
                compiled = components[0]
                self._raman_kernel = passes.raman_spectrum(
                    compiled.undamped,
                    rates,
                    compiled.fraction,
                    self._problem.grids.dt_fine,
                )
            else:
                self._raman_kernel = passes.mixture_raman_spectrum(
                    components, rates, densities, self._problem.grids.dt_fine
                )
            self._raman_rates = rates
            self._raman_densities = amplitudes
            self._raman_transforms += 1
        return self._raman_kernel

    # --- the statistics ---------------------------------------------------------

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
    ) -> None:
        """Load a state, and forget the stage the step before it left behind.

        Parameters
        ----------
        field : numpy.ndarray
            The state in the canonical layout.
        noise : solprop.noise.NoiseRealisation or None
            The realisation, or ``None`` to take the contract's.
        z : float, optional
            Where the state belongs, m. A positive value is a resume.
        """
        super().set_initial(field, noise, z=z)
        # A state that was handed in is not the state any stage was evaluated at, so
        # the first-same-as-last identity is broken until a step has been taken, and
        # so is what the last column evaluation left behind for the statistics.
        self._has_recycled = False
        self._pending = None
        self._stash_current = False
        # The entrance plane's quadrature_error is read before any march: the alarm
        # of a previous run on this backend is not its (SEP 22 section 4.7).
        self._alarm = math.nan
        self._alarm_reported = False
        if self._adams is not None:
            self._adams.reset()

    def _march_to(self, target: float) -> None:
        """Step to `target` with the Adams method when one is set, else Runge-Kutta.

        Parameters
        ----------
        target : float
            Position to reach, m.
        """
        if self._adams is None:
            super()._march_to(target)
        else:
            self._adams.march_to(target)

    def _low_storage_engine(self, method: LowStorageRungeKutta) -> object:
        """Return the resident engine that runs the low-storage march's register passes.

        Parameters
        ----------
        method : solprop.stepper.low_storage.LowStorageRungeKutta
            The march.

        Returns
        -------
        solprop.backends.numba_cpu.low_storage.NumbaLowStorage
            The engine, holding ``B``, ``C`` and the flows (SEP 24 section 6.6).
        """
        return NumbaLowStorage(self, method)

    def _adams_march(self) -> AdamsMarch | None:
        """Return the Adams march the problem asks for, with its buffers, or ``None``.

        Returns
        -------
        solprop.backends.multistep.AdamsMarch or None
            The march, when the problem carries an integrator.
        """
        integrator = self._problem.stepper.integrator
        if not isinstance(integrator, Multistep):
            return None
        shape = (self._n_columns, self._n_omega)
        # The lab-frame field goes in as it is: the combine kernel's propagator and
        # its empty stage row leave it untouched.
        self._identity = np.ones(shape, dtype=np.complex128)
        self._no_weights = np.zeros(0, dtype=np.float64)
        self._adams_out = np.zeros(shape, dtype=np.complex128)
        self._adams_held: tuple[float, float, StageScalars] | None = None
        self._last_scalars: StageScalars | None = None
        # The passes' rows run in a prange under the spatial stages' own rule, on
        # the state alone: a large problem, threads to run it, and enough elements
        # for a parallel region to pay.
        parallel = (
            self._pools.numba > 1
            and self._pools.regime == "large"
            and self._n_columns * self._n_omega > fft.REGIME_ELEMENTS
        )
        return AdamsMarch(
            self,
            integrator,
            kernels=NumbaAdamsKernels(parallel=parallel),
            exponential=NumbaExponentialKernels(parallel=parallel),
        )

    def _dense_stages(self) -> list[np.ndarray]:
        """Return the last trial's stage derivatives, in its anchor's frame.

        Stage one is the recycled one when the trial took it, which the trial wrote
        into its row, so every row is the accepted step's own.

        Returns
        -------
        list of numpy.ndarray
            One per tableau stage, copies in the canonical layout.
        """
        shape = self._problem.shape
        return [stage.reshape(shape).copy() for stage in self._stages]

    def _adams_propagator(
        self, anchor: float, position: float
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return ``U(position, anchor)`` from this backend's exponentiation kernel.

        Parameters
        ----------
        anchor, position : float
            The step's anchor and its end, m.

        Returns
        -------
        numpy.ndarray
            ``exp(Phi(position) - Phi(anchor))``, an array the caller owns.
        """
        bundle = self._stage_scalars(anchor, position)
        self._write_propagator(bundle, 0)
        self._recomputed = math.nan
        # The step's source call is evaluated at its end, so it reuses this bundle and
        # the pair just written to slot 0 rather than forming them again.
        self._adams_held = (anchor, position, bundle)
        return self._forward[0].reshape(self._problem.shape).copy()

    def _adams_lab_source(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Evaluate the source once through the route's own stage pipeline (SEP 16).

        A zero-length first stage: propagator slot 0 holds the flow over no distance,
        exactly one in both signs, so the route's stage -- modal, general modal,
        Cartesian, line, radial or planar, with the noise draw at `position` and the
        in-stage absorbers -- returns its source untransported. The Runge-Kutta
        accumulators it also touches are not read by the Adams march.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame field at `position`.
        position : float
            The evaluation position, m.
        purpose : str, optional
            Atomic-call accounting label; this backend executes no atomic response.

        Returns
        -------
        numpy.ndarray
            ``F(lab, position)`` in the lab frame.
        """
        del purpose
        # The trial has just evaluated the scalars ending at this position for its
        # flow: their node values are the sample's, so they are reused rather than
        # hoisting and evaluating every path again (SEP 16 section 6).
        held = self._last_scalars
        if held is not None and held.z1 == position:
            bundle = held
        else:
            self._hoist(position)
            bundle = self._stage_scalars(position, position)
        timers = self._pass_timers
        mark = timers.now() if timers is not None else 0.0
        # The flow over no distance, exactly one in both signs.
        self._forward[0].fill(1.0)
        self._inverse[0].fill(1.0)
        if timers is not None:
            mark = timers.lap("adams.identity_fill", mark)
        self._recomputed = 0.0
        self._adams_held = None
        # Beside a coupled flow the sample is the material source alone: the residual
        # joins the exponential formulation's compensation and the flow its moments
        # (SEP 16 phase 5), so the stage runs without the flow's pipeline.
        flow, self._flow = self._flow, None
        try:
            self._stage(
                np.ascontiguousarray(lab.reshape(self._n_columns, self._n_omega)),
                0,
                0.0,
                bundle,
            )
        finally:
            self._flow = flow
        if timers is None:
            return self._stages[0].reshape(self._problem.shape).copy()
        mark = timers.now()
        sample = self._stages[0].reshape(self._problem.shape).copy()
        timers.lap("adams.sample_copy", mark)
        return sample

    def _adams_block(self, z: float, dz: float) -> object:
        """Return the frozen block of the section a step from `z` is anchored in.

        The reference's factors, which this backend's flow holds
        (``FlowWorkspace.held``), and its residual.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m.

        Returns
        -------
        object
            The step's :class:`~solprop.backends.multistep.CoupledBlock`, or ``None``
            without a coupled flow.
        """
        if self._flow is None:
            return None
        held = self._flow.held
        bundle = self._stage_scalars(z, z + dz)
        section = bundle.flow_section
        assert section is not None
        residual = None
        if held.residual_terms is not None and not held.cf4[section]:
            if self._block_residual is None:
                self._block_residual = BlockResidual(held)
            residual = functools.partial(self._block_residual, section)
        return CoupledBlock(
            active=held.active,
            vectors=held.vectors[section],
            triangular=held.triangular[section],
            section=section,
            end_weights=bundle.flow_weights,
            residual=residual,
            # The flow actions' own eigenbasis and route: the trial's matrix
            # functions on the gated bins come from it, in one pass.
            basis=(
                self._flow.left[section],
                self._flow.right[section],
                self._flow.eigenvalues[section],
                self._flow.route[section],
            ),
        )

    def _stage_scalars(self, z0: float, z1: float) -> StageScalars:
        """Return a stage's host scalars, remembering the last for the Adams source.

        Parameters
        ----------
        z0, z1 : float
            The step's anchor and the stage's position, m.

        Returns
        -------
        solprop.translate.stage.StageScalars
            The weights and the scalar path values.
        """
        bundle = super()._stage_scalars(z0, z1)
        self._last_scalars = bundle
        return bundle

    def _adams_source(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        anchor: float,
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> tuple[
        Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ]:
        """Evaluate the source once through the modal route's stage kernels.

        The stage pipeline of :meth:`_stage` with the lab-frame field as its input: the
        combine kernel pads it with an identity propagator and no previous stages, the
        transforms and the response columns run as for any stage, and the finish kernel
        takes the result to the anchor's frame with zero tableau weights, so the
        Runge-Kutta accumulators are untouched.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame field at `position`.
        anchor, position : float
            The step's anchor and the evaluation position, m.
        purpose : str, optional
            Atomic-call accounting label; ignored when no atomic service is bound.

        Returns
        -------
        tuple
            ``U(anchor, position) N(lab, position)`` and ``U(position, anchor)``.
        """
        if self._free or self._slab or self._general is not None:
            # The other routes' stage pipelines: the lab-frame sample carried to the
            # anchor by the step's inverse flow, formed once more for this pair.
            source = self._adams_lab_source(lab, position, purpose=purpose)
            self._write_propagator(self._stage_scalars(anchor, position), 0)
            self._recomputed = math.nan
            shape = self._problem.shape
            return (
                source * self._inverse[0].reshape(shape),
                self._forward[0].reshape(shape).copy(),
            )
        del purpose
        held = self._adams_held
        if held is not None and held[0] == anchor and held[1] == position:
            bundle = held[2]
        else:
            self._hoist(anchor)
            bundle = self._stage_scalars(anchor, position)
            self._write_propagator(bundle, 0)
            self._recomputed = math.nan
        self._adams_held = None
        forward = self._forward[0].reshape(self._problem.shape).copy()
        inverse = self._inverse[0]
        self._combine(
            np.ascontiguousarray(lab.reshape(self._n_columns, self._n_omega)),
            self._stages,
            self._no_weights,
            0,
            0.0,
            self._identity,
            self._noise_for(bundle),
            self._crop,
            self._plans.modal_spectrum,
            self._pad_scale,
        )
        if self._spec.family == "field_analytic":
            _double_positive_bins(self._plans.modal_spectrum, self._n_omega)
        if not self._nonlinear and not self._in_stage_absorber:
            return np.zeros(self._problem.shape, dtype=np.complex128), forward
        self._last_bundle = bundle
        self._modal_to_time(bundle)
        if self._nonlinear:
            self._polarisation_spectrum(bundle)
        else:
            self._plans.projected_spectrum[:] = 0.0
        self._finish_stage(
            self._plans.projected_spectrum,
            self._crop,
            self._prefactor_for(bundle),
            self._problem.grids.omega_window,
            inverse,
            self._absorber_term(),
            0.0,
            0.0,
            self._adams_out,
            self._solution,
            self._error,
        )
        return self._adams_out.reshape(self._problem.shape).copy(), forward

    def _on_accepted_step(self) -> None:
        """Score the quadrature from the last stage of an accepted step.

        The relative difference between the projections of the same node polarisation
        with the Kronrod rule and with its embedded Gauss subset. It is a **statistic
        and a warning, never a controller input**. The fine rule's spectrum is already
        in hand from the last stage's pass 6, so this costs one forward transform where
        the reference pays two.

        The value is ``nan`` when the rule has no embedded pair, because the difference
        is then identically zero and reading that as convergence would be wrong; and
        ``nan`` where the polarisation itself vanishes, at a true vacuum end.
        """
        if self._pending is not None:
            # `k1^{n+1} = exp(dPhi^n) k7^n`: the picture is re-anchored every step, so
            # the recycled stage is rebased by the whole step's forward propagator
            # (``docs/dev/stepper.md`` section 2). Done here and not on the next
            # trial step, so that a rejected attempt cannot invalidate it: the state
            # it belongs to is the one the run just adopted.
            last = self._problem.stepper.tableau.n_stages - 1
            np.multiply(
                self._propagator(1.0, self._pending),
                self._stages[last],
                out=self._recycled,
            )
            self._has_recycled = True
            self._pending = None
        # The last column evaluation of this step scanned the state the run has just
        # adopted, so the plasma statistics may be read off what it left behind.
        self._stash_current = self._spec.stash and self._fsal and self._nonlinear
        if (
            self._plans.coarse_to_frequency is None
            or not self._nonlinear
            or self._last_bundle is None
        ):
            self._alarm = math.nan
            return
        grids = self._problem.grids
        weight = self._prefactor_for(self._last_bundle) * grids.omega_window
        self._coarse_spectrum(self._last_bundle)
        fine = weight * self._plans.projected_spectrum[:, self._crop]
        coarse = weight * self._plans.coarse_projected_spectrum[:, self._crop]
        scale = float(np.linalg.norm(fine))
        self._alarm = (
            math.nan if scale == 0.0 else float(np.linalg.norm(coarse - fine) / scale)
        )
        if self._alarm > QUADRATURE_ALARM and not self._alarm_reported:
            self._alarm_reported = True
            logger.warning(
                "the transverse quadrature reproduces the nonlinear polarisation to "
                "only %.2e at z = %.6g m, above the %.0e alarm; the integrand is not "
                "resolved by this rule (a kinked ionisation rate, or too few nodes for "
                "the mode set) and no step size can fix it",
                self._alarm,
                self._last_bundle.z1,
                QUADRATURE_ALARM,
            )

    def _coarse_spectrum(self, bundle: StageScalars) -> None:
        """Transform the embedded rule's projection of the last stage to frequency.

        SEP 6's dispersive response finishes its embedded products here, once per
        evaluation, and joins the spectrum as it does the fine rule's.

        Parameters
        ----------
        bundle : solprop.translate.stage.StageScalars
            The last stage's scalars.
        """
        plans, dispersive = self._plans, self._dispersive
        assert plans.coarse_to_frequency is not None
        if dispersive is not None:
            dispersive.finish_coarse(bundle)
        if dispersive is None or not dispersive.replaces_transform:
            plans.coarse_to_frequency.execute()
        if dispersive is not None:
            dispersive.combine_coarse()
        if self._envelope is not None:
            self._envelope.assemble(
                plans.coarse_projected_spectrum, self._crop, coarse=True
            )
        if self._field_plasma is not None:
            self._field_plasma.assemble(plans.coarse_projected_spectrum, coarse=True)

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Collect the requested statistics at the current position.

        The statistics of the saved field go to :func:`solprop.stats.collect`, as every
        backend's do; the four the catalogue marks ``backend`` are formed here from this
        backend's own buffers and its compiled passes.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Which statistics and which windows.

        Returns
        -------
        dict
            Small host arrays keyed by statistic name, each with a leading batch axis.

        Raises
        ------
        BackendError
            If the backend is closed, no initial state was loaded, no grid was given at
            construction, or a backend statistic needs something this run does not have.
        """
        import dataclasses  # noqa: PLC0415 - only the delegation needs it

        state = self._require_state()
        grid = self._require_grid()
        # The accessor whitelist's backend door, as in the reference backend: a
        # legacy name asked of a general basis, whose static synthesis matrix is a
        # checked zero placeholder, and a general name asked of an orthogonal or
        # spatial basis, which carries neither the certified metric nor the stored
        # E/H reference the name promises (`solprop.output.capability`). Before the
        # spatial branch, so a spatial run is refused rather than answered.
        general_modal = self._problem.transform.modal_tables is not None
        refusal = refusal_for(
            declared_capability(
                general_modal=general_modal,
                spatial=self._problem.transform.geometry is not None,
            ),
            spec.names,
            withheld=withheld_accessors(
                general_modal=general_modal,
                coupled_flow=self._problem.linear.coupled_flow is not None,
            ),
        )
        if refusal is not None:
            raise BackendError(refusal)
        refusal = axis_refusal(self._problem, spec.names)
        if refusal is not None:
            raise BackendError(refusal)
        if self._problem.transform.geometry is not None:
            response_names = (
                {"electron_density", "peak_ionisation_rate", "quadrature_error"}
                | set(PLASMA_INTEGRATION_STATISTICS)
                | {name for name in CATALOGUE if name.startswith("species_")}
            )
            spatial_spec = dataclasses.replace(
                spec,
                names=tuple(name for name in spec.names if name not in response_names),
                periodic=(),
            )
            tables = self._far_field.for_spec(spatial_spec)
            # Only the physical-space statistics read the node volume; the far-field
            # ones, absorbed_power between planes among them, read the stored state.
            volume = (
                self._free
                and not self._slab
                and bool(
                    set(spatial_spec.names)
                    & {"peak_intensity", "beam_radius", "fwhm_r"}
                )
            )
            members = [
                self._spatial_member(spatial_spec, grid, one, tables, volume=volume)
                for one in state
            ]
            return {
                name: np.stack([one[name] for one in members])
                for name in spatial_spec.names
            } | self._backend_statistics(
                tuple(name for name in spec.names if name in response_names), state
            )
        # As in the reference backend: the weight is a Chebyshev evaluation per column,
        # the whole of what the Poynting weighting adds to a CPU statistics call, so a
        # spec asking for none of the Poynting statistics never evaluates it.
        needed = bool(set(spec.names) & WEIGHT_DEPENDENT_STATISTICS)
        if needed:
            np.copyto(
                self._statistics_power_weight, power_weight_at(self._problem, self._z)
            )
        weight_refusal = weight_refusal_for(spec.names, self._statistics_power_weight)
        if weight_refusal is not None:
            raise BackendError(weight_refusal)
        weighted = needed and not np.all(self._statistics_power_weight == 1.0)
        power_names = tuple(
            name for name in spec.names if weighted and name in _POYNTING_STATISTICS
        )
        axis = None
        if "peak_intensity" in power_names:
            if self._modes is None:
                msg = "the peak on-axis intensity needs the mode set"
                raise BackendError(msg)
            axis = self._on_axis_synthesis(self._modes)
        power_values = (
            [
                _poynting_statistics(
                    spec, grid, member, self._statistics_power_weight, axis
                )
                for member in state
            ]
            if weighted
            else []
        )
        owned = tuple(
            name
            for name in spec.names
            if CATALOGUE[name].backend and name not in power_names
        )
        delegated = dataclasses.replace(
            spec,
            names=tuple(
                name
                for name in spec.names
                if name not in owned and name not in power_names
            ),
            periodic=tuple(
                name
                for name in spec.periodic
                if name not in owned and name not in power_names
            ),
        )
        from solprop.output.schema import SavedPlane  # noqa: PLC0415 - as above

        # The basis metric at this plane, as in the reference backend: `None` for an
        # identity metric and for every route without one, which is the statistics'
        # own bitwise-unchanged path.
        metric = (
            metric_at(self._problem.linear.modal_metric, self._z)
            if set(delegated.names) & METRIC_WEIGHTED_STATISTICS
            else None
        )
        members = [
            collect(
                delegated,
                grid,
                SavedPlane(z=self._z, field=state[member]),
                modes=self._modes,
                profiles=self._profiles,
                metric=metric,
            )
            for member in range(state.shape[0])
        ]
        values = {
            name: np.stack([member[name] for member in members])
            for name in delegated.names
        }
        values.update(
            {
                name: np.stack([member[name] for member in power_values])
                for name in power_names
            }
        )
        return values | self._backend_statistics(owned, state)

    def _spatial_member(
        self,
        spec: StatsSpec,
        grid: TimeGrid,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        tables: ApertureTables | None,
        *,
        volume: bool,
    ) -> dict[str, np.ndarray]:
        """Collect one member's spatial statistics, the far field by its banded pass.

        On a Cartesian or line problem the far-field statistics and
        ``on_axis_spectrum`` (its ``W = 1`` slot) come from the compiled pass of SEP 23
        section 6, whose on-axis amplitude ``on_axis_intensity`` then reads; every
        other statistic is the shared host function's.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            The spatial statistics asked for.
        grid : solprop.grid.TimeGrid
            The temporal grid.
        field : numpy.ndarray
            One member's spectrum.
        tables : solprop.spatial.ApertureTables or None
            The call's far-field tables.
        volume : bool
            Whether a physical-space statistic needs the node volume.

        Returns
        -------
        dict
            The member's statistics, without a batch axis.
        """
        folded: dict[str, np.ndarray] = {}
        axis = None
        if self._far_field_pass is not None:
            folded, axis = self._far_field_pass(spec.names, grid, field, tables)
        # The plane measure's energy and the closure the local statistics read
        # (SEP 34), in one compiled pass over the columns.
        wanted = set(spec.names)
        closure = None
        flux = self._problem.flux
        geometry = self._problem.transform.geometry
        local = bool(wanted & {"peak_intensity", "beam_radius", "fwhm_r"}) or (
            axis is None and bool(wanted & {"on_axis_spectrum", "on_axis_intensity"})
        )
        if flux is not None and geometry is not None and ("energy" in wanted or local):
            energy, closure = spatial_kernels.flux_measure(
                flux.at(self._z),
                grid,
                geometry.spectral_weights,
                field,
                closure=local,
                parallel=self._parallel_spatial,
            )
            if "energy" in wanted:
                folded["energy"] = np.asarray(energy)
        remaining = tuple(name for name in spec.names if name not in folded)
        import dataclasses  # noqa: PLC0415 - as in stats

        values = collect_spatial(
            spec
            if len(remaining) == len(spec.names)
            else dataclasses.replace(spec, names=remaining),
            grid,
            self._problem.transform,
            field,
            z=self._z,
            flux=self._problem.flux,
            profiles=self._profiles,
            tables=tables,
            operator=self._problem.linear.separable,
            axis=axis,
            closure=closure,
            # A slab's node volume is formed by the shared function, as on the
            # reference: it is n_x/n_p times the state.
            physical=self._free_physical(field) if volume else None,
        )
        return values | folded

    def _backend_statistics(
        self,
        names: tuple[str, ...],
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ) -> dict[str, np.ndarray]:
        """Share one population scan per member among the requested statistics.

        Every plasma statistic of a plane -- the five species statistics, the two
        totals, the on-axis rows and, under the band rule, the integration diagnostics
        -- reads the same scan of every node, so a member is scanned once, not once
        per statistic: a default capillary spec asked for eight scans of each plane,
        which was most of what the plane statistics cost. Under the band rule the
        diagnostics' column is the problem's own statistics column, so its population
        rows are the scan the species statistics would take, and that one scan serves
        both. The stash, where it is current, is still read in place of a scan.

        Parameters
        ----------
        names : tuple of str
            Backend-owned statistic names.
        state : numpy.ndarray
            Accepted state at this plane with its leading batch axis.

        Returns
        -------
        dict
            Requested values with leading batch axes and no retained field cache.
        """
        envelope = self._envelope is not None
        count = self._spec.n_channels
        totals = ("electron_density", "peak_ionisation_rate")
        # A field problem with no plasma channel leaves the totals to `_backend_stat`,
        # which refuses them by name.
        population_names = tuple(
            name
            for name in names
            if name.startswith("species_")
            or (name in totals and (envelope or count > 0))
        )
        integration_names = tuple(
            name for name in names if name in PLASMA_INTEGRATION_STATISTICS
        )
        axis_names = tuple(name for name in names if name in ON_AXIS_PLASMA_STATISTICS)
        if axis_names and not population_names:
            # As on the reference: the species statistics are the node scan on every
            # route, so sharing it with the on-axis statistics changes none of their
            # values and lets a node on the axis supply its row.
            population_names = tuple(
                name for name in names if name.startswith("species_")
            )
        # This observation's scans by member, read again by every statistic of it.
        scans: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        result = {
            name: self._backend_stat(name, state)
            for name in names
            if name not in population_names
            and name not in integration_names
            and name not in axis_names
        }
        if integration_names:
            share = bool(population_names) and self._spec.band and not envelope
            diagnostics = [
                self._integration_diagnostics(member, scans if share else None, index)
                for index, member in enumerate(state)
            ]
            result.update(
                {
                    name: np.stack([member[name] for member in diagnostics])
                    for name in integration_names
                }
            )
        if population_names:
            members = [
                self._species_statistics(member, scans, index)
                for index, member in enumerate(state)
            ]
            if not envelope and count == 1:
                # One field channel: the totals are read at the node of largest
                # ionised fraction (`_ionisation`), from the scan just shared.
                for index, (member, field) in enumerate(
                    zip(members, state, strict=True)
                ):
                    density, rate = self._ionisation(field, scans[index])
                    member["electron_density"] = density
                    member["peak_ionisation_rate"] = rate
            result.update(
                {
                    name: np.stack([member[name] for member in members])
                    for name in population_names
                }
            )
        if axis_names:
            axis = [
                self._axis_statistics(member, scans.get(index))
                for index, member in enumerate(state)
            ]
            result.update(
                {
                    name: np.stack([member[name] for member in axis])
                    for name in axis_names
                }
            )
        return result

    def _backend_stat(
        self,
        name: str,
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ) -> np.ndarray:
        """Compute one backend-owned statistic over the batch.

        Parameters
        ----------
        name : str
            The catalogue name.
        state : numpy.ndarray
            The state at this plane.

        Returns
        -------
        numpy.ndarray
            The value with a leading batch axis.
        """
        if name == "quadrature_error":
            return np.full(state.shape[0], self._alarm)
        if name == "peak_intensity":
            return np.stack(
                [self._peak_intensity(state[member]) for member in range(len(state))]
            )
        # Every plasma statistic of a problem with a population comes from the shared
        # scan of `_backend_statistics`; what reaches here is a field problem with no
        # plasma channel, which has no rate to integrate and is refused by name.
        msg = (
            "the electron density and the peak ionisation rate need a compiled "
            "plasma response, and this problem declares none"
        )
        raise BackendError(msg)

    def _plasma_stats(
        self,
        nodes: Num[np.ndarray, "n_rows n_to"],
        initial: float | Float[np.ndarray, " n_channels"],
        partials: Float[np.ndarray, "n_partials n_nodes"],
        kernel: Any = None,
    ) -> None:
        """Scan one block of nodes for their plasma statistics.

        The plain statistics column, or under the band rule (SEP 18) the band column,
        which also writes the diagnostics' parts and needs the taps and a scratch ring
        per node.

        Parameters
        ----------
        nodes : numpy.ndarray
            The block's field on the fine grid, rows ``node * n_pol + component``.
        initial : float or numpy.ndarray
            The initial fraction, one per channel when there are several.
        partials : numpy.ndarray
            Written with the block's columns of the statistics' rows.
        kernel : numba.core.dispatcher.Dispatcher or None, optional
            A band statistics column to use in place of the problem's own, which is
            how a trapezoid run asking for the diagnostics is answered.
        """
        column = self._stats_column if kernel is None else kernel
        if kernel is None and not self._spec.band:
            column(nodes, self._rate, initial, self._problem.grids.dt_fine, partials)
            return
        work = np.empty(
            (nodes.shape[0] // self._n_pol_node, kernels.BAND_STATS_ROWS, self._n_to)
        )
        column(
            nodes,
            self._rate,
            self._taps,
            initial,
            self._problem.grids.dt_fine,
            partials,
            work,
        )

    def _scan_nodes(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        initial: float | Float[np.ndarray, " n_channels"],
        partials: Float[np.ndarray, "n_partials n_nodes"],
        kernel: Any = None,
    ) -> None:
        """Scan every node of one member for its plasma statistics, on any geometry.

        A slab's nodes come in the stage's blocks, a free-space grid's in chunks of
        its physical points, and a modal or averaged problem's all at once.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.
        initial : float or numpy.ndarray
            The initial fraction, one per channel when there are several.
        partials : numpy.ndarray
            Written with the statistics' rows, one column per node.
        kernel : numba.core.dispatcher.Dispatcher or None, optional
            A band statistics column in place of the problem's own.
        """
        if self._slab:
            for first, last, nodes in self._slab_node_chunks(field):
                self._plasma_stats(nodes, initial, partials[:, first:last], kernel)
        elif self._free:
            physical = self._free_physical(field)
            for first in range(0, len(physical), self._chunk_nodes):
                last = min(first + self._chunk_nodes, len(physical))
                rows = (last - first) * self._n_pol_state
                spatial_kernels.pack_columns(
                    physical[first:last].reshape(rows, self._n_omega),
                    self._crop,
                    self._plans.modal_spectrum,
                    self._problem.grids.pad_scale / self._n_to,
                    # The analytic family doubles the one-sided spectrum, as the step
                    # path does; packing it as a real field scanned half the field.
                    self._spec.family == "field_analytic",
                )
                self._plans.to_time.execute()
                self._plasma_stats(
                    self._plans.modal_time[:rows],
                    initial,
                    partials[:, first:last],
                    kernel,
                )
        else:
            self._plasma_stats(self._nodes_at(field), initial, partials, kernel)

    def _integration_diagnostics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        scans: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        member: int = 0,
    ) -> dict[str, np.ndarray]:
        """Return the plasma column's integration diagnostics at this plane (SEP 18).

        The band statistics column over every node, whatever the problem's rule, then
        the largest of each part over the nodes and the three ratios of
        :func:`solprop.backends.numpy_ref.passes.plasma_integration_statistics`.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.
        scans : dict or None, optional
            Where to leave copies of the scan's channel fractions and peak rates,
            keyed by `member`, for the species statistics to read instead of scanning
            again. Only a band-rule problem passes it, whose own statistics column
            this is, and nothing is left when the stash is current.
        member : int, optional
            The batch member, the key into `scans`.

        Returns
        -------
        dict
            ``plasma_undershoot``, ``plasma_work_defect`` and
            ``plasma_boundary_level``, each zero-dimensional.

        Raises
        ------
        BackendError
            If the problem carries no field plasma channel.
        """
        count = self._spec.n_channels
        if count == 0 or self._envelope is not None:
            msg = (
                "the plasma integration diagnostics read the field-grid plasma "
                "column, and this problem carries none"
            )
            raise BackendError(msg)
        kernel = kernels.make_band_stats_column(self._spec)
        preion = tuple(
            float(c.preionisation(self._z))
            for c in self._problem.responses.plasma_channels
        )
        initial = preion[0] if count == 1 else np.asarray(preion)
        rows = 2 * count + kernels.BAND_DIAGNOSTICS
        partials = np.zeros((rows, self._problem.transform.n_nodes))
        self._scan_nodes(field, initial, partials, kernel)
        if scans is not None and not self._stash_is_current(field):
            scans[member] = (
                partials[: 2 * count : 2].copy(),
                partials[1 : 2 * count : 2].copy(),
            )
        undershoot, defect, boundary = passes.plasma_integration_statistics(
            np.max(partials[2 * count :], axis=1)
        )
        return {
            "plasma_undershoot": np.asarray(undershoot),
            "plasma_work_defect": np.asarray(defect),
            "plasma_boundary_level": np.asarray(boundary),
        }

    def _species_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        scans: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        member: int = 0,
    ) -> dict[str, np.ndarray]:
        """Read the accepted stash or scan independent channel populations.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member at this plane, shape (n_modes, n_pol, n_omega).
        scans : dict or None, optional
            This observation's channel fractions and peak rates at every node, keyed
            by `member`: read in place of a scan when the member is already there (a
            band-rule problem's diagnostics scanned it), and otherwise left there as
            copies for the totals and the on-axis statistics.
        member : int, optional
            The batch member, the key into `scans`.

        Returns
        -------
        dict
            The five species statistics and the two total plasma statistics.
        """
        if self._envelope is not None:
            return self._envelope_statistics(field, scans, member)
        count = self._spec.n_channels
        if count == 0:
            empty = np.zeros((0, self._problem.transform.n_nodes))
            return population_statistics(self._problem, self._z, empty, empty)
        if scans is not None and member in scans:
            fractions, peaks = scans[member]
            return population_statistics(self._problem, self._z, fractions, peaks)
        if self._stash_is_current(field):
            partials = self._stash.T
        else:
            partials = self._population_partials
            preion = tuple(
                float(c.preionisation(self._z))
                for c in self._problem.responses.plasma_channels
            )
            initial = preion[0] if count == 1 else np.asarray(preion)
            self._scan_nodes(field, initial, partials)
        if scans is not None:
            scans[member] = (
                partials[: 2 * count : 2].copy(),
                partials[1 : 2 * count : 2].copy(),
            )
        return population_statistics(
            self._problem,
            self._z,
            partials[: 2 * count : 2],
            partials[1 : 2 * count : 2],
        )

    def _envelope_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        scans: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        member: int = 0,
    ) -> dict[str, np.ndarray]:
        """Recompute cycle-averaged populations in full and final partial chunks.

        Parameters
        ----------
        field : numpy.ndarray
            One member's spectrum, shape (n_modes, n_pol, n_omega).
        scans : dict or None, optional
            Where to leave copies of the populations for the on-axis statistics.
        member : int, optional
            The batch member, the key into `scans`.

        Returns
        -------
        dict
            Species statistics and total electron density and peak rate.
        """
        assert self._envelope is not None
        channels = self._envelope.assembly.channels
        initials = tuple(float(c.preionisation(self._z)) for c in channels)
        partials = self._population_partials
        grids = self._problem.grids
        if self._slab:
            for first, last, nodes in self._slab_node_chunks(field):
                self._envelope.populations(
                    nodes,
                    initials,
                    grids.dt_fine,
                    grids.t_window_fine,
                    partials[:, first:last],
                )
        elif self._free:
            physical = self._free_physical(field)
            for first in range(0, len(physical), self._chunk_nodes):
                last = min(first + self._chunk_nodes, len(physical))
                rows = (last - first) * self._n_pol_state
                spatial_kernels.pack_columns(
                    physical[first:last].reshape(rows, self._n_omega),
                    self._crop,
                    self._plans.modal_spectrum,
                    grids.pad_scale / self._n_to,
                    False,
                )
                self._plans.to_time.execute()
                self._envelope.populations(
                    self._plans.modal_time[:rows],
                    initials,
                    grids.dt_fine,
                    grids.t_window_fine,
                    partials[:, first:last],
                )
        else:
            nodes = self._nodes_at(field)
            for first in range(0, self._problem.transform.n_nodes, self._chunk_nodes):
                last = min(first + self._chunk_nodes, self._problem.transform.n_nodes)
                self._envelope.populations(
                    nodes[first * self._n_pol_node : last * self._n_pol_node],
                    initials,
                    grids.dt_fine,
                    grids.t_window_fine,
                    partials[:, first:last],
                )
        count = len(channels)
        if scans is not None:
            scans[member] = (
                partials[: 2 * count : 2].copy(),
                partials[1 : 2 * count : 2].copy(),
            )
        return population_statistics(
            self._problem,
            self._z,
            partials[: 2 * count : 2],
            partials[1 : 2 * count : 2],
        )

    def _ionisation(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        scanned: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the electron density and the peak ionisation rate at this plane.

        Both are read at the **node of largest ionised fraction**, which is the
        reference's choice and its reasons; what differs is that the rate lookup and
        the exposure are one compiled traversal rather than a full re-synthesis
        followed by the multipass scan, which was measured at two thirds of a whole CtC
        step on an x86-64 server.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.
        scanned : tuple of numpy.ndarray or None, optional
            This observation's channel fractions and peak rates at every node, shape
            ``(1, n_nodes)`` each, when the species statistics have already scanned
            the member; otherwise the nodes are scanned here.

        Returns
        -------
        tuple of numpy.ndarray
            The electron density in m^-3 and the peak rate in s^-1.
        """
        responses = self._problem.responses
        plasma = responses.plasma_coefficients
        # Called only for a problem's one field channel (`_backend_statistics`).
        assert plasma is not None
        if scanned is None:
            fractions, rates = self._ionised_nodes(field, plasma)
        else:
            fractions, rates = scanned[0][0], scanned[1][0]
        # Both are read at the node of largest ionised fraction, which is the
        # reference's choice.
        densest = int(np.argmax(fractions))
        density = float(plasma.density(self._z)) * float(fractions[densest])
        return np.asarray(density), np.asarray(float(rates[densest]))

    def _ionised_nodes(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        plasma: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return each node's final ionised fraction and largest rate.

        Two routes to the same numbers. Between planes the accepted step's last stage
        has **already** scanned every node for exactly these two, so the compiled column
        left them in ``self._stash`` and this costs a read: that is what keeps the
        statistics channel affordable at ``period = 1``, because the alternative
        re-synthesises every node into a buffer of megabytes and evicts the step's own
        working set -- measured at 14 % of a reduced-M2 step against 0.4 % for the
        stash. Anywhere else -- at a save plane, where the field offered may not be the
        one the last stage saw -- the nodes are synthesised and scanned as before.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.
        plasma : solprop.translate.responses.PlasmaCoefficients
            The compiled plasma response.

        Returns
        -------
        tuple of numpy.ndarray
            The fraction at the end of the window and the largest rate, one per node.
        """
        if self._stash_is_current(field):
            return self._stash[:, 0], self._stash[:, 1]
        partials = np.empty((self._partial_rows, self._problem.transform.n_nodes))
        self._scan_nodes(field, float(plasma.preionisation(self._z)), partials)
        return partials[0], partials[1]

    def _slab_node_chunks(
        self, field: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> Iterator[tuple[int, int, np.ndarray]]:
        """Yield a planar state's field at the nodes of each block, on the fine grid.

        The stage's own route for the statistics: the columns to their ``(p, y)``
        rows, each block of rows to time, and the synthesis in x, into the node
        buffer the stage uses, which each yield overwrites.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.

        Yields
        ------
        tuple
            The block's first and one-past-last node, and its field in V/m, rows
            ``node * n_pol + component``.
        """
        slab, plans = self._slab_plans, self._plans
        assert slab is not None
        n_y = slab.spectrum.shape[1]
        n_x = self._chunk_nodes // self._slab_block
        np.multiply(
            field.reshape(self._n_columns, self._n_omega), 1.0 / n_y, out=slab.rows
        )
        slab.inverse.execute()
        for first in range(0, n_y, self._slab_block):
            count = min(self._slab_block, n_y - first)
            rows = slice(
                first * self._slab_rows_per_y, (first + count) * self._slab_rows_per_y
            )
            spatial_kernels.pack_columns(
                slab.rows[rows],
                self._crop,
                plans.modal_spectrum,
                1.0 / self._n_to,
                self._spec.family == "field_analytic",
            )
            plans.to_time.execute()
            size = count * n_x * self._n_pol_node
            self._slab_product(
                self._synthesis,
                plans.modal_time[: rows.stop - rows.start],
                self._nodes[:size],
                count,
            )
            yield first * n_x, (first + count) * n_x, self._nodes[:size]

    def _stash_is_current(
        self, field: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> bool:
        """Say whether the stash describes the state it is being asked about.

        Set by :meth:`_on_accepted_step` and cleared whenever the state moves without
        one -- a new plane, a fresh load. What makes it true is the tableau: with
        ``first_same_as_last`` the seventh stage's input **is** the accept pass's own
        result, so the last column evaluation of an accepted step scanned exactly the
        state the run then adopted. ``_fsal`` is also false for the ``split_step``
        absorber, which is right, because there ``_finish`` changes the state after
        the stages have run.

        The field is compared as well as the flag, because the stash describes the
        backend's own current state and nothing else: a caller handing in another
        plane's field gets the fresh scan it asked for.

        Parameters
        ----------
        field : numpy.ndarray
            The state the statistic is wanted at.

        Returns
        -------
        bool
            Whether the stash may be read instead of a fresh scan.
        """
        held = self._state
        return (
            self._stash_current
            and held is not None
            and field.shape == held.shape[1:]
            and np.shares_memory(field, held)
        )

    def _peak_intensity(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> np.ndarray:
        """Return the largest intensity on the axis of the waveguide, in W/m^2.

        The reference's definition exactly -- the analytic field synthesised at
        ``r = 0`` from the modes' own profiles, both Cartesian components -- with the
        reduction compiled. The on-axis matrix is built once and rescaled with the
        taper (:meth:`_on_axis_synthesis`), which is what makes this statistic cheap
        enough for the between-planes channel.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.

        Returns
        -------
        numpy.ndarray
            The peak intensity, zero-dimensional.

        Raises
        ------
        BackendError
            If the backend was built without the mode set the axis needs.
        """
        if self._modes is None:
            msg = (
                "the peak on-axis intensity needs the mode set, which the kernel "
                "contract does not carry; construct the backend with "
                "select('numba_cpu', problem, modes=modes)"
            )
            raise BackendError(msg)
        analytic = analytic_field(self._require_grid(), field)
        columns = np.tensordot(
            self._on_axis_synthesis(self._modes), analytic, axes=([1], [0])
        )
        partials = np.empty(1)
        kernels.peak_reduce(
            np.ascontiguousarray(columns.reshape(-1, columns.shape[-1])), partials
        )
        return np.asarray(0.5 * EPS0 * C * float(partials[0]))

    def _nodes_at(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> Num[np.ndarray, "n_rows n_to"]:
        """Return the field at every quadrature node, on the fine time grid.

        Unlike the stage loop this is not chunked: the statistics are read at the node
        of largest ionised fraction, which is not known until every node has been
        scanned. The buffer is allocated on the first call and kept.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.

        Returns
        -------
        numpy.ndarray
            The field in V/m, rows ``node * n_pol + component``.
        """
        transform = self._problem.transform
        rows = transform.synthesis.shape[0]
        if self._stats_nodes is None:
            self._stats_nodes = np.zeros(
                (rows * self._n_pol_state, self._n_to), dtype=self._nodes.dtype
            )
        self._plans.modal_spectrum[:] = 0.0
        self._plans.modal_spectrum[:, self._crop] = (
            field.reshape(self._n_columns, self._n_omega) * self._pad_scale
        )
        if self._spec.family == "field_analytic":
            _double_positive_bins(self._plans.modal_spectrum, self._n_omega)
        self._plans.to_time.execute()
        taper = (
            1.0
            if transform.taper_scale is None
            else float(transform.taper_scale(self._z))
        )
        np.multiply(self._synthesis, taper, out=self._scaled_synthesis)
        np.dot(
            self._scaled_synthesis,
            self._plans.modal_time.reshape(
                transform.n_modes, self._n_pol_state * self._n_to
            ),
            out=self._stats_nodes.reshape(rows, -1),
        )
        return self._stats_nodes


def _reuse_plasma_fraction(spec: kernels.ColumnSpec, spatial: bool) -> bool:
    """Select the measured one-channel fibre component and traversal bracket.

    Parameters
    ----------
    spec : solprop.backends.numba_cpu.kernels.ColumnSpec
        Compiled response family, component count and traversal policy.
    spatial : bool
        Whether the problem uses a spatial grid instead of fibre modes.

    Returns
    -------
    bool
        Whether the measured policy reuses each sample's channel fraction.
    """
    return (
        spec.plasma
        and spec.n_channels == 1
        and not spatial
        and (not spec.parallel or spec.n_pol == 1)
    )


def _chunk_layout(problem: KernelProblem) -> tuple[int, int, int, int]:
    """Return the chunk layout: the slab's y block and rows, the nodes, the rows.

    A node chunk of ``min(node_chunk, n_nodes)`` nodes on every kind but the hybrid
    planar one, which takes whole y rows at a time, as many as the node chunk holds,
    as the reference blocks them: the ``(p, y)`` rows of a block go to time together
    and their ``(x, y)`` nodes are one contiguous run of node indices.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The admitted problem.

    Returns
    -------
    tuple of int
        The y rows per block and the ``(p, y)`` rows per y row (both zero off the
        slab), the nodes per chunk, and the rows one temporal transform takes: a
        block's ``(p, y)`` rows, a node chunk's in free space, every modal column
        otherwise.
    """
    transform = problem.transform
    geometry = transform.geometry
    _, n_modes, n_pol, _ = problem.shape
    nodes = min(transform.node_chunk, transform.n_nodes)
    if geometry is not None and geometry.kind == "slab":
        n_x, n_y = geometry.shape
        block = min(n_y, max(1, transform.node_chunk // n_x))
        rows = transform.synthesis.shape[1] * n_pol
        return block, rows, block * n_x, block * rows
    if problem.linear.separable is not None:
        return 0, 0, nodes, nodes * transform.n_pol * n_pol
    return 0, 0, nodes, n_modes * n_pol


def _regime_elements(problem: KernelProblem) -> int:
    """Return the element count the threading regime is read off.

    ``n_to * n_columns * n_nodes`` of the transform; on the general modal route the
    node loop runs over the reference's own quadrature nodes, three components each,
    and the static transform is a checked placeholder of one node, so those are what
    is counted. The hybrid planar kind's synthesis is dense in x alone, one
    ``(n_x, n_p)`` product per y row, so it counts ``n_to * n_columns * n_x``.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The admitted problem.

    Returns
    -------
    int
        The element count.
    """
    _, n_modes, n_pol, _ = problem.shape
    tables = problem.transform.modal_tables
    if tables is not None:
        return problem.grids.n_to * MAX_N_COMPONENTS * int(tables.region_index.size)
    geometry = problem.transform.geometry
    if geometry is not None and geometry.kind == "slab":
        return problem.grids.n_to * n_modes * n_pol * geometry.shape[0]
    return problem.grids.n_to * n_modes * n_pol * problem.transform.n_nodes


def _override(name: str, given: int | None) -> int | None:
    """Return a thread-count override, from the argument or the environment.

    The registry constructs a backend with the seam's arguments alone, so a run that
    wants to set a pool without reaching past the seam sets an environment variable.
    The argument wins where both are given.

    Parameters
    ----------
    name : str
        The environment variable, for example ``"SOLPROP_NUMBA_THREADS"``.
    given : int or None
        The constructor's argument.

    Returns
    -------
    int or None
        The override, or ``None`` to leave the choice to the regime.

    Raises
    ------
    BackendError
        If the environment variable is set to something that is not a thread count.
    """
    if given is not None:
        return given
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        return int(value)
    except ValueError as error:
        msg = f"{name} must be a thread count, and is {value!r}"
        raise BackendError(msg) from error


def _channel_rate_arrays(problem: KernelProblem) -> tuple[np.ndarray, np.ndarray]:
    """Pack rates along a leading channel axis without resampling.

    Parameters
    ----------
    problem : KernelProblem
        Translated channels and their immutable rate tables.

    Returns
    -------
    tuple of numpy.ndarray
        Leading-channel coefficients and six scalar interpolation values per channel.
    """
    channels = problem.responses.plasma_channels
    if len(channels) <= 1:
        return np.empty((0, 0)), np.empty((0, 6))
    width = max(channel.rate.coefficients.size for channel in channels)
    storage = np.zeros((len(channels), width))
    metadata = np.empty((len(channels), 6))
    for index, channel in enumerate(channels):
        source = channel.rate.coefficients
        storage[index, : source.size] = source.ravel()
        rate = channel.rate
        metadata[index] = (
            rate.log_field_min,
            rate.d_log_field,
            rate.field_min,
            rate.field_max,
            rate.rate_max,
            source.shape[1],
        )
    return storage, metadata


def _rate_arrays(plasma: Any) -> RateArrays:
    """Return the compiled ionisation rate as a kernel takes it, or an empty stand-in.

    A kernel is compiled without the plasma branch when a problem has none, but numba
    still types the argument, so something of the right shape has to be passed.

    Parameters
    ----------
    plasma : solprop.translate.responses.PlasmaCoefficients or None
        The compiled plasma group.

    Returns
    -------
    RateArrays
        The rate, or a table no field can reach.
    """
    if plasma is not None:
        return RateArrays.of(plasma.rate)
    return RateArrays(
        coefficients=np.zeros((4, 1)),
        log_field_min=0.0,
        d_log_field=1.0,
        field_min=math.inf,
        field_max=math.inf,
        rate_max=0.0,
    )


def _double_positive_bins(
    spectrum: Complex[np.ndarray, "n_columns n_to"], n_omega: int
) -> None:
    """Turn a padded one-sided spectrum into the one whose inverse is analytic.

    The analytic signal of a real field is the complex inverse transform of its
    one-sided spectrum with the positive bins doubled and everything above them zero --
    which is what :func:`scipy.signal.hilbert` computes after first transforming the
    field it was given. On a field grid that does not radiate the harmonic the response
    column works on that analytic signal, so doing it here costs one plan and removes
    the reference's separate transform pair per node chunk.

    The pad has already zeroed everything the crop did not write, so only the coarse
    bins need scaling; the fine grid's Nyquist bin is never doubled, which matters only
    when the coarse grid is not oversampled (the same rule as the Cartesian
    ``pack_columns``).

    Parameters
    ----------
    spectrum : numpy.ndarray
        The padded fine spectrum, complex, full length. Modified in place.
    n_omega : int
        Coarse bins the crop wrote, whose first is the zero frequency.
    """
    spectrum[:, 1 : min(n_omega, spectrum.shape[1] // 2)] *= 2.0


def _harmonic_carrier(problem: KernelProblem) -> Complex[np.ndarray, " n_to"]:
    """Return ``exp(2 i omega_0 t)``, or a stand-in the kernel was compiled to ignore.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, whose compiled Kerr term says whether the harmonic is wanted.

    Returns
    -------
    numpy.ndarray
        The carrier of shape ``(n_to,)``, or an array of ones of that shape.

    Raises
    ------
    BackendError
        If a harmonic term is compiled and the contract carries no carrier for it.
    """
    kerr = problem.responses.kerr_coefficients
    if kerr is None or kerr.harmonic_factor == 0.0:
        return np.ones(problem.grids.n_to, dtype=np.complex128)
    if problem.grids.carrier_2w0 is None:
        msg = (
            "a third-harmonic Kerr term is compiled and grids.carrier_2w0 is absent; "
            "the envelope's harmonic sits an offset 2 omega_0 from its carrier and the "
            "response pass cannot place it without the factor"
        )
        raise BackendError(msg)
    return np.ascontiguousarray(problem.grids.carrier_2w0)


def _stacked_projection_shape(n_modes: int, n_nodes: int, node_chunk: int) -> bool:
    """Select the measured bracket for packed fine/coarse field projection.

    Parameters
    ----------
    n_modes : int
        Positive number of output modes.
    n_nodes : int
        Positive number of physical quadrature nodes.
    node_chunk : int
        Positive maximum physical nodes in one chunk.

    Returns
    -------
    bool
        Whether this shape lies in the measured bracket where the packed projection
        pays.
    """
    return (
        2 <= n_modes <= _STACKED_PROJECTION_MAX_MODES
        and n_nodes >= _STACKED_PROJECTION_MIN_NODES
        and n_nodes > node_chunk
    )


def _embedded_projection(
    problem: KernelProblem,
) -> Float[np.ndarray, "n_modes n_columns"] | None:
    """Return the embedded coarse-rule projection, when the quadrature has one.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, whose transform carries both projections.

    Returns
    -------
    numpy.ndarray or None
        The coarse projection, or ``None`` when it is the fine one and the alarm would
        be identically zero.
    """
    transform = problem.transform
    if np.array_equal(transform.projection, transform.projection_coarse):
        return None
    return np.ascontiguousarray(transform.projection_coarse)


_GENERAL_LEAVES = (
    "linear.coupled_flow",
    "linear.modal_metric",
    "transform.modal_tables",
    "responses.regional_responses",
)
"""The schema-13 leaves ``numba_cpu`` executes: the general modal route and the flow.

The three general-modal leaves, ``linear.coupled_flow`` and a moving guide's maps
(kernel schema 14's ``transform_weights``), in every combination ``numpy_ref`` admits
but a CF4 section, which is checked by the shared admission list
(:func:`solprop.backends.numpy_ref.admission.refuse_general_route`) so that the backends
cannot drift apart.
"""


def _refuse_flow_here(problem: KernelProblem) -> None:
    """Refuse the coupled flows this backend does not build and the reference does.

    Two, each by name, before the reference's own list runs under this backend's
    name: a ``"cf4"`` section, whose commutator-free exponentials of the whole
    generator at each stage's own Gauss nodes would need the node weights on this
    backend, and a flow beside a Cartesian
    free-space problem, which no translation builds and whose stage this backend runs
    through separate kernels that carry no flow.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to check.

    Raises
    ------
    BackendError
        Naming the section method or the route.
    """
    flow = problem.linear.coupled_flow
    if flow is None:
        return
    if "cf4" in flow.section_methods:
        msg = (
            "numba_cpu does not execute a 'cf4' section of linear.coupled_flow: its "
            "commutator-free exponentials are of the whole generator at each stage's "
            "own Gauss nodes, which this backend has no node weights for. Run "
            "CoherentWall(method='cf4') on numpy_ref, or declare the default "
            "method='frozen_residual', which numba_cpu executes"
        )
        raise BackendError(msg)
    if problem.linear.separable is not None:
        msg = (
            "numba_cpu executes linear.coupled_flow on the modal routes and this "
            "problem is a Cartesian free-space one: its stage runs through the "
            "spatial kernels, which carry no flow action"
        )
        raise BackendError(msg)


def _refuse_adams(problem: KernelProblem) -> None:
    """Refuse an integrator, or an Adams route, this backend's marches do not serve.

    Every route's source is evaluated through its own stage pipeline (SEP 16
    phase 3); a coupled flow is stepped by the exponential formulation only
    (phase 5) and refused by name otherwise, as the reference refuses it, at the
    same point of the list. The low-storage march (SEP 24) runs on every route the
    translation admits it on.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to check.

    Raises
    ------
    BackendError
        Naming what is unsupported.
    """
    integrator = problem.stepper.integrator
    if not isinstance(integrator, Multistep):
        # No integrator, or the low-storage march, which every admitted route runs
        # through its register kernels (SEP 24 phase 2).
        return
    if (
        problem.linear.coupled_flow is not None
        and integrator.formulation != "exponential"
    ):
        msg = (
            f"numba_cpu does not step a problem with a coupled linear flow with "
            f"the {integrator.name} integrator; use "
            "Multistep(formulation='exponential') or RungeKutta()"
        )
        raise BackendError(msg)
    refuse_formulation(problem, "numba_cpu")
    # SEP 35 phase 2: the scalar shift, fused into the compiled passes.
    refuse_carrier_frame(problem, "numba_cpu", delivered=frozenset({"scalar"}))


def _refuse_unsupported(problem: KernelProblem) -> None:
    """Refuse, by name, every problem this backend does not execute.

    The reference's refusals, for the same reasons and in the same order, plus this
    backend's own: a term silently dropped would run, converge and be wrong.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to check.

    Raises
    ------
    BackendError
        Naming what is unsupported.
    """
    if problem.schema_version != SCHEMA_VERSION:
        msg = (
            f"this backend implements kernel contract version {SCHEMA_VERSION} and "
            f"the problem declares version {problem.schema_version}"
        )
        raise BackendError(msg)
    if problem.precision.reduced:
        name = problem.precision.name
        raise BackendError(
            f"numba_cpu computes in binary64 only and refuses precision={name!r} "
            "(solprop SEP 33, SEP 36); use backend='native', or backend='numpy_ref', "
            "the oracle"
        )
    if problem.tdse is not None:
        raise BackendError("numba_cpu does not execute TDSE responses; use numpy_ref")
    refuse_general_leaves(problem, "numba_cpu", admits=_GENERAL_LEAVES)
    # The flow first, as the reference orders it: a metric or a table beside it is
    # the flow's refusal to make, and its message names the flow. A 'cf4' section is
    # refused before the integrator, since no integrator runs it here.
    _refuse_flow_here(problem)
    _refuse_adams(problem)
    refuse_general_route(
        problem,
        backend="numba_cpu",
        flow_methods=("constant_schur", "frozen_residual"),
        moving_maps=True,
    )
    if (
        problem.flags.has_spectral_responses
        and problem.responses.envelope_plasma is None
    ):
        msg = (
            "numba_cpu executes the Kerr, Raman and plasma terms and this problem "
            "declares spectral responses; a term silently dropped would run, converge "
            "and be wrong"
        )
        raise BackendError(msg)
    if (
        problem.flags.has_raman
        and problem.transform.n_pol * problem.shape[2] > 1
        and (
            not problem.responses.raman_terms
            or not all(term.tensor_capable for term in problem.responses.raman_terms)
        )
    ):
        raise BackendError(
            "vector Raman requires tensor-capable terms; found scalar_only"
        )
    n_batch, _, n_pol, _ = problem.shape
    if problem.transform.n_pol * n_pol > MAX_N_POL:
        msg = (
            f"the transform resolves {problem.transform.n_pol} transverse components "
            f"and the state carries {n_pol}; the components belong to one of the two "
            "and not to both, so this problem would count them twice"
        )
        raise BackendError(msg)
    if n_batch != NumbaCpuBackend.max_batch:
        msg = (
            f"numba_cpu steps one problem at a time (max_batch = "
            f"{NumbaCpuBackend.max_batch}) and this problem has a batch axis of "
            f"{n_batch}"
        )
        raise BackendError(msg)
