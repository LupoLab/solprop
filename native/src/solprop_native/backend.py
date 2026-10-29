# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The solprop-dependent native backend adapter."""

from __future__ import annotations

import dataclasses
import functools
import logging
import math
import os
import time
import types
import warnings
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from types import TracebackType
from typing import Any, ClassVar, Literal, Protocol, cast

import numpy as np
import numpy.typing as npt
from solprop._helpers import Helpers
from solprop._interrupts import committing
from solprop.backends.driver import StepDriver, reached, replay_tolerance
from solprop.backends.exponential import (
    CarrierReference,
    carrier_reference,
    refuse_carrier_frame,
    refuse_formulation,
)
from solprop.backends.low_storage import LowStorageMarch, schedule_key, table_steps
from solprop.backends.memory import (
    LOCAL_FLUX_STATISTICS,
    low_storage_flows,
    low_storage_stage_register,
    z_table_slots,
)
from solprop.backends.multistep import (
    AdamsKernels,
    AdamsMarch,
    CoupledBlock,
    ExponentialKernels,
    _Derivative,  # pyright: ignore[reportPrivateUsage] - the march's history entries
    fused_tables,
    source_absent,
    trial_tables,
)
from solprop.backends.numpy_ref import QUADRATURE_ALARM
from solprop.backends.numpy_ref.admission import refuse_general_route
from solprop.backends.numpy_ref.general_modal import (
    DELAYED_TERMS,
    general_dispersive_buffers,
    general_node_chunk,
    node_delayed_maps,
    node_kerr_form,
    node_kerr_tensor,
)
from solprop.backends.numpy_ref.passes import plasma_integration_statistics
from solprop.backends.populations import axis_refusal, population_statistics
from solprop.backends.spatial import (
    AXIS_SPECTRUM_ONLY,
    FarFieldTables,
    far_field_partials,
    fluence_budget,
    radial_absorber_matrix,
    signed_fluence_widths,
)
from solprop.constants import EPS0, C
from solprop.exceptions import (
    BackendError,
    ConvergenceError,
    HostMarchWarning,
)
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.monitor import BoundaryEnergies, BoundaryMonitor
from solprop.noise import NoiseRealisation
from solprop.output.capability import (
    declared_capability,
    refusal_for,
    withheld_accessors,
)
from solprop.output.schema import SavedPlane
from solprop.profiles import ZProfile
from solprop.spatial import (
    FAR_FIELD_STATISTICS,
    ApertureTables,
)
from solprop.stats import (
    METRIC_WEIGHTED_STATISTICS,
    ON_AXIS_PLASMA_STATISTICS,
    PLASMA_INTEGRATION_STATISTICS,
    PROFILE_STATISTICS,
    StatsSpec,
    collect,
    inband_wavelengths,
    profile_value,
    zdw,
)
from solprop.stepper.controller import PIController
from solprop.stepper.low_storage import LowStorageRungeKutta
from solprop.stepper.multistep import (
    Multistep,
    integrated_weights,
    lagrange_coefficients,
)
from solprop.translate.chebyshev import (
    _EDGE_SLACK_FRACTION,  # pyright: ignore[reportPrivateUsage] - the tables' own admitted span
    PiecewiseChebyshev,
)
from solprop.translate.coupled import (
    CoupledFlowTables,
    SectionMethod,
    flow_edge_slack,
    frozen_generator,
    metric_at,
    schur_blocks,
    schur_eigenbasis,
)
from solprop.translate.linop import power_weight_at, rate_weights
from solprop.translate.modal import DEFAULT_NODE_BUDGET
from solprop.translate.plasma_source import (
    _BENCHMARK_GATED,  # pyright: ignore[reportPrivateUsage] - refuse the private NumPy benchmark hook
)
from solprop.translate.problem import SCHEMA_VERSION, KernelProblem, PropagatorMode
from solprop.translate.stage import (
    StageScalars,
    StageTerms,
    combine,
    require_device_ready,
)
from solprop.translate.tdse import TDSEPurpose

from . import PROC_BIND, THREAD_BUDGET, WAIT_POLICY, _core, build_info, check_build
from ._helpers import NATIVE

type ComplexArray = npt.NDArray[np.complex128]
type RealArray = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


class _envelope_errors:
    """Translate refinement failures without importing solprop in the extension.

    A class rather than a generator: it wraps every native step call.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        """Enter the native call."""

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Raise a refinement failure leaving the call as solprop's error.

        Raises
        ------
        solprop.exceptions.ConvergenceError
            If the call raised the extension's ``EnvelopeConvergenceError``.
        """
        del kind, traceback
        if isinstance(error, _core.EnvelopeConvergenceError):
            raise ConvergenceError(str(error)) from error


ADAMS_MAX_CHANNELS = 16
"""The most channels of a coupled block the resident Adams engine steps.

The native ``adams_max_channels``: the engine's per-bin passes hold a block's channel
vectors in registers; a larger block takes solprop's host engine.
"""

_SPATIAL_STAT_FLAGS = {
    "energy": 1,
    "on_axis_intensity": 2,
    "peak_intensity": 4,
    "on_axis_spectrum": 8,
    "beam_radius": 16,
    "fwhm_r": 16,
}

_FAR_FIELD_FLAGS = {
    "beam_spectrum": 1 << 5,
    "absorbed_power": 1 << 7,
    "region_spectrum": 1 << 8,
    "aperture_spectrum": 1 << 9,
    "aperture_on_axis_spectrum": 1 << 10,
    "on_axis_spectrum": 1 << 11,
}
"""The native far-field pass's flags (solprop SEP 23 section 6), from 32.

``on_axis_spectrum`` is the pass's ``W = 1`` slot on a Cartesian or line grid; on the
other spatial kinds it stays ``spatial_statistics``' flag 8.
"""

_FAR_FIELD_ORDER = (
    "beam_spectrum",
    "absorbed_power",
    "region_spectrum",
    "aperture_spectrum",
    "aperture_on_axis_spectrum",
    "on_axis_spectrum",
)
"""The order in which the native pass packs its statistics."""

_STAT_FLAGS = {
    "omega0": 1 << 0,
    "energy": 1 << 1,
    "energy_window": 1 << 2,
    "peak_power": 1 << 3,
    "peak_power_window": 1 << 4,
    "peak_power_total": 1 << 5,
    "fwhm_t": 1 << 6,
    "fwhm_t_outer": 1 << 6,
    "electron_density": 1 << 7,
    "peak_ionisation_rate": 1 << 8,
    "peak_intensity": 1 << 9,
    "quadrature_error": 1 << 10,
}

_POPULATIONS_FLAG = 1 << 11
"""The native per-node, per-channel population rows, packed after the named values."""

_PLASMA_INTEGRATION_FLAG = 1 << 12
"""solprop SEP 18's run diagnostics: the native six maxima over every node, packed last.

The parts of :func:`solprop.backends.numpy_ref.passes.plasma_integration_diagnostics`,
from which :func:`solprop.backends.numpy_ref.passes.plasma_integration_statistics` forms
``plasma_undershoot``, ``plasma_work_defect`` and ``plasma_boundary_level``.
"""

_PLASMA_INTEGRATION_PARTS = 6


class NativeBackend(StepDriver[_core.State]):
    """Propagate modal, radial or Cartesian fields with persistent Kokkos state.

    Parameters
    ----------
    problem
        Validated schema-22 kernel problem. Accepts one batch member,
        modal or spatial scalar or two-component propagation (radial, Cartesian, or
        the hybrid planar kind of slab modes in x times an FFT in y), instantaneous
        Kerr, plasma, scalar time-domain Raman, weighted control, both absorber
        schemes, and device-owned statistics; and solprop's general modal route, whose
        statistics are solprop's own, on a constant guide or a moving one whose maps
        are combined per stage.
    grid, modes, profiles
        Host context used only for plane statistics.

    Raises
    ------
    solprop.exceptions.BackendError
        If the build or problem lies outside the implemented seam contract.
    """

    name = "native"
    max_batch = 1
    helpers = NATIVE
    propagator_default = "recompute"
    far_field_statistics: ClassVar[frozenset[str]] = frozenset(FAR_FIELD_STATISTICS)
    """The far-field statistics of solprop SEP 23 this build forms natively.

    solprop's registry admits them on this backend by this attribute and refuses
    them, by name, on a build without it.
    """

    def __init__(
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        adams_march: Literal["native", "host"] = "native",
    ) -> None:
        if _BENCHMARK_GATED.get():
            raise BackendError("the gated plasma benchmark requires numpy_ref")
        check_build(problem)
        _refuse_unsupported(problem)
        problem.validate()
        # solprop SEP 29 G1: a z table is evaluated natively on a CPU build; a device
        # build refuses it, naming the table's size and the CPU backends.
        if not problem.device_ready and _on_device():
            require_device_ready(problem, "native")
        super().__init__(problem, grid=grid, modes=modes, profiles=profiles)
        self._z_tables = None if problem.device_ready else _ZTables(problem)
        self._flow = _host_flow(problem)
        self._flow_key: tuple[int, float] | None = None
        tables = problem.transform.modal_tables
        self._moving_maps = (
            tables is not None
            and tables.synthesis is not None
            and tables.synthesis.shape[0] * tables.synthesis.shape[1] > 1
        )
        # solprop SEP 23: the far-field tables are formed once, which refuses a
        # geometry the far-field statistics do not admit before anything is built.
        self._far_field = FarFieldTables(problem)
        self._native_state = _core.State(
            _native_problem(problem, self._terms, self._flow, self._z_tables)
            | _low_storage_payload(problem, device=_on_device())
        )
        if self._z_tables is not None:
            self._z_tables.upload(self._native_state)
        _upload_flux_table(self._native_state, problem)
        self._last_loaded = np.empty(0, dtype=np.complex128)
        self._alarm = math.nan
        self._alarm_reported = False
        # solprop's Adams march (SEP 14 phase 2) steps on host arrays, with one native
        # source evaluation a step; the accepted state is kept on the host beside the
        # native one, which the statistics and observations read.
        integrator = problem.stepper.integrator
        self._resident: _ResidentAdams | None = None
        self._adams = (
            _NativeAdamsMarch(self, integrator, march=adams_march)
            if isinstance(integrator, Multistep)
            else None
        )
        self._host_state: ComplexArray | None = None
        self._adams_held: tuple[float, float, StageScalars, ComplexArray] | None = None
        # solprop's low-storage march (SEP 24 phase 3) steps through the resident
        # engine `_low_storage_engine` returns, built over the native state above.
        if isinstance(integrator, LowStorageRungeKutta):
            self._low_storage = LowStorageMarch(self, integrator)

    def _apply_mask(self, state: ComplexArray) -> _core.State:
        self._last_loaded = np.ascontiguousarray(state.reshape(-1))
        empty = np.empty(0, dtype=np.complex128)
        self._native_state.set_initial(self._last_loaded, empty)
        return self._native_state

    def set_initial(
        self,
        field: ComplexArray,
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
    ) -> None:
        """Load an exact host `complex128` state and optional noise realization."""
        _require_initial_array(field, self._problem.shape)
        super().set_initial(field, noise, z=z)
        native_noise = (
            np.empty(0, dtype=np.complex128)
            if self._noise is None
            else np.ascontiguousarray(self._noise.reshape(-1))
        )
        self._native_state.set_initial(self._last_loaded, native_noise)
        self._native_state.clear_fsal()
        if self._adams is not None:
            self._host_state = self._native_state.field_to_host().reshape(
                self._problem.shape
            )
            self._adams.reset()

    def _march_to(self, target: float) -> None:
        """Step to `target` with the Adams method when one is set, else Runge-Kutta."""
        try:
            if self._adams is None:
                super()._march_to(target)
            else:
                self._adams.march_to(target)
        finally:
            self._check_quadrature(target)

    def _adams_state(self) -> ComplexArray:
        """Return the host copy of the accepted state the Adams march reads.

        With the resident history (solprop SEP 16 phase 3b) the march reads it only to
        report a stopped run, so it is copied from the engine when asked for.
        """
        if self._host_state is None:
            raise BackendError("no initial state has been loaded")
        if self._resident is not None:
            return self._native_state.field_to_host().reshape(self._problem.shape)
        return self._host_state

    def _adams_engine(
        self,
        method: Multistep,
        compensation: Literal["transported", "untransported"],
    ) -> object:
        """Return the resident Adams history for the fused path (solprop SEP 16).

        The interaction formulation on the modal routes keeps its history, its step
        arithmetic and its norms in the native state (solprop ADR-043); the other
        paths take solprop's host engine over this backend's hooks.
        """
        # Beside a coupled flow the exponential formulation's matrix passes are
        # resident (solprop SEP 16 phase 6) when every bin of every section is on the
        # flow's eigenbasis route and the block has at most `ADAMS_MAX_CHANNELS`
        # channels; otherwise solprop's host engine runs them over this backend's hooks.
        flow = self._flow
        if flow is not None and (
            method.formulation != "exponential"
            or not bool(np.all(flow.route))
            or self._problem.shape[1] > ADAMS_MAX_CHANNELS
            or self._problem.shape[0] != 1
        ):
            return None
        # A varying operator's exponential formulation carries the transported
        # compensation natively, as does a frozen-residual flow's residual, whatever
        # the diagonal's representation; the untransported test witness stays on the
        # host.
        linear = self._problem.linear
        residual = flow is not None and flow.residual_terms is not None
        compensated = method.formulation == "exponential" and (
            residual
            or (
                linear.phi_representation in ("expansion", "table")
                and not source_absent(self._problem)
            )
        )
        if compensated and compensation != "transported":
            return None
        self._resident = _ResidentAdams(
            native=self._native_state,
            method=method,
            controller=self._problem.stepper.controller,
            n_omega=self._problem.grids.n_omega,
            arrays=self._resident_arrays,
            sample=self._resident_sample,
            hoist=self._hoist,
            adopted=self._resident_adopted,
            rate=self._rate_weights,
            shape=tuple(self._problem.shape),
            helpers=self.helpers,
            compensated=compensated,
            interpolate=self._problem.stepper.interpolate_planes,
            carrier=carrier_reference(self._problem, method),
        )
        # solprop SEP 39 phases 1 and 3: where the trial's stage rows are one bundle
        # repeated, the native state reads the scalar tables itself, uploaded once.
        tables = self._stage_tables()
        if tables is not None:
            self._native_state.adams_stage_tables(
                tables | _chord_tables(self._resident.carrier)
            )
            self._resident.staged = True
        return self._resident

    def _stage_tables(self) -> dict[str, object] | None:
        """Return the tables a resident Adams trial reads natively, or ``None``.

        solprop SEP 39 phases 1 and 4: every scalar table ``translate.stage.scalars``
        reads at a stage -- the weights of a constant or expanded ``Phi`` and
        prefactor, the density, the taper scale, the Raman decay rates and densities,
        the Kerr density, the plasma channels' densities and fractions, SEP 6's
        channel densities -- and an expansion's rate table, each packed as its batch
        pass reads it (``PiecewiseChebyshev.packed``). A separable operator's rows are
        a constant's, the distance and the position with a unit prefactor
        (`_trial_arrays`), and read no span. A z table's rows are its slots, which the
        native state loads through the cache `_ZTables` shares. A coupled flow's
        section and residual weights and a moving guide's map weights are sectioned
        Chebyshev rows (`flow_weights`, `transform_weights`), formed natively from
        their edges.
        """
        problem = self._problem
        linear, responses = problem.linear, problem.responses
        phi_weights = phi_rate = prefactor_weights = None
        separable = self._terms.separable is not None
        tables = self._z_tables
        if tables is not None and tables.phi:
            phi = _STAGE_TABLE
        elif separable or linear.phi_constant is not None:
            phi = _STAGE_CONSTANT
        elif linear.phi_expansion is not None:
            phi = _STAGE_EXPANSION
            phi_weights = linear.phi_expansion.weights.packed()
            phi_rate = linear.rate.packed()
        else:
            return None
        low = high = 0.0
        if tables is not None and tables.prefactor:
            prefactor = _STAGE_TABLE
        elif separable:
            prefactor = _STAGE_CONSTANT
            low, high = -math.inf, math.inf
        elif responses.prefactor_constant is not None:
            prefactor = _STAGE_CONSTANT
            low, high = responses.prefactor_span
        elif responses.prefactor_expansion is not None:
            prefactor = _STAGE_EXPANSION
            prefactor_weights = responses.prefactor_expansion.weights.packed()
        else:
            return None
        kerr = responses.kerr_coefficients
        taper = problem.transform.taper_scale
        return {
            "n_stages": problem.stepper.tableau.n_stages,
            "phi": phi,
            "phi_weights": phi_weights,
            "phi_rate": phi_rate,
            "prefactor": prefactor,
            "prefactor_weights": prefactor_weights,
            "prefactor_low": float(low),
            "prefactor_high": float(high),
            "density": responses.density.packed(),
            "taper": None if taper is None else taper.packed(),
            "raman_rates": [
                table.packed()
                for term in responses.raman_terms
                for table in term.decay_rates
            ],
            "kerr": 0 if kerr is None else 1 if kerr.density is None else 2,
            "kerr_density": None
            if kerr is None or kerr.density is None
            else kerr.density.packed(),
            "plasma_densities": [
                channel.density.packed() for channel in responses.population_channels
            ],
            "plasma_preionisation": [
                channel.preionisation.packed()
                for channel in responses.population_channels
            ],
            "raman_densities": [
                None if term.density is None else term.density.packed()
                for term in responses.raman_terms
            ],
            "dispersive_densities": []
            if responses.dispersive_kerr is None
            else [table.packed() for table in responses.dispersive_kerr.densities],
        } | _section_tables(
            problem,
            moving=self._moving_maps,
            residual=self._flow is not None and self._flow.residual_terms is not None,
        )

    def _low_storage_engine(self, method: LowStorageRungeKutta) -> object:
        """Return the resident low-storage engine (solprop SEP 24 section 6.7).

        Its step is one native call: four register passes accepted in place in the
        native state, which holds the registers and the flows (`_ResidentLowStorage`).
        """
        return _ResidentLowStorage(self, method)

    @property
    def _propagator_mode(self) -> PropagatorMode:
        """Return the flow treatment this run uses, ``"auto"`` resolved.

        As solprop's driver resolves it, except that a device build forms the low-
        storage march's flows per element under ``"auto"``: there an extra field read
        costs more than the exponential (SEP 24 section 4.3).
        """
        asked = self._problem.stepper.propagator_mode
        if asked == "auto" and self._low_storage is not None and _on_device():
            return "recompute"
        return super()._propagator_mode

    def _hold(self, state: _core.State | None) -> None:
        """Hold the field-free plane's state in the native memory space (SEP 22, Q15).

        One volume of the native state, allocated at the first such plane and counted
        in its ledger; nothing crosses to the host at the plane. A checkpoint after a
        step abandoned in flight copies it once (`_held_field`).
        """
        if state is None:
            self._held_z = None
            return
        if state is not self._native_state:
            raise BackendError("the held state does not belong to this backend")
        self._native_state.hold_plane()
        self._held_z = self._z

    def _held_field(self) -> ComplexArray:
        """Return the held plane's state, copied to the host once."""
        return self._native_state.held_to_host().reshape(self._problem.shape)

    def _resident_arrays(self, anchor: float, position: float) -> _TrialArrays:
        """Return a resident trial's arrays, every stage row at `position`."""
        bundle = self._stage_scalars(anchor, position)
        self._last_bundle = bundle
        stages = self._problem.stepper.tableau.n_stages
        arrays, material = self._repeated_arrays(bundle)
        flow = self._flow
        if flow is not None:
            # Beside a coupled flow the trial's section and a frozen-residual flow's
            # weights at `position`; the resident passes need no Runge-Kutta action.
            material |= {
                "flow_section": bundle.flow_section,
                "flow_weights": np.empty(0, dtype=np.float64)
                if flow.residual_terms is None
                else np.ascontiguousarray(
                    np.tile(_required(bundle.flow_weights, "flow_weights"), stages),
                    dtype=np.float64,
                ),
            }
        return arrays, material

    def _resident_sample(self, step: _TrialArrays, position: float) -> _TrialArrays:
        """Return the arrays of a trial at `position`, from a step's that ends there.

        An exponential trial evaluates its new sample at a trial anchored at the step's
        end. That bundle reads every table at the same position as the step's own, by
        the same evaluation, and its increment of ``Phi`` is a value less itself:
        exactly zero, also for a z table's slots. So its arrays are the step's with the
        increments zeroed, value for value, with no second evaluation of the tables.
        Beside a coupled flow the section and weights follow the anchor, so the bundle
        is formed.
        """
        if self._flow is not None:
            return self._resident_arrays(position, position)
        (phi_delta, *rows), material = step
        phi_at, prefactor, density, taper, preionisation, raman_rates = rows
        return (
            np.zeros_like(phi_delta),
            phi_at,
            prefactor,
            density,
            taper,
            preionisation,
            raman_rates,
        ), material

    def _repeated_arrays(self, bundle: StageScalars) -> _TrialArrays:
        """Return the trial arrays of a step whose every stage row is `bundle`.

        `_trial_arrays` of the bundle repeated, value for value. With the weights a
        constant's or an expansion's own and no further material rows (SEP 6's
        channels, a moving guide's maps), each array is one row repeated in one call
        rather than packed from a list of the same bundle, which costs a resident
        Adams trial more than its coefficient tables. The other problems take
        `_trial_arrays`.
        """
        stages = self._problem.stepper.tableau.n_stages
        if (
            self._z_tables is not None
            or self._terms.separable is not None
            or self._problem.responses.dispersive_kerr is not None
            or self._moving_maps
        ):
            return self._trial_arrays([bundle] * stages, bundle)
        phi_delta = _required(bundle.phi_delta, "phi_delta")[np.newaxis]
        phi_at = _required(bundle.phi_at, "phi_at")[np.newaxis]
        prefactor = _required(bundle.prefactor, "prefactor")[np.newaxis]
        return (
            np.repeat(phi_delta, stages + 1, axis=0),
            np.repeat(phi_at, stages, axis=0),
            np.repeat(prefactor, stages, axis=0),
            np.full(stages, bundle.density, dtype=np.float64),
            np.full(stages, bundle.taper_scale, dtype=np.float64),
            np.full(stages, bundle.preionisation, dtype=np.float64),
            _repeated_row(bundle.raman_rates, stages),
        ), {
            "kerr_density": np.full(stages, bundle.kerr_density, dtype=np.float64),
            "plasma_densities": _repeated_row(bundle.plasma_densities, stages),
            "plasma_preionisation": _repeated_row(bundle.plasma_preionisation, stages),
            "raman_densities": _repeated_row(bundle.raman_densities, stages),
        }

    def _adams_block(self, z: float, dz: float) -> object:
        """Return the frozen block of the section a step from `z` is anchored in.

        The host-side factors this backend decomposed its flow into (``_HostFlow``),
        and a frozen-residual flow's residual relative to the step's section, for
        solprop's exponential formulation beside a coupled flow (SEP 16 phase 5).
        """
        flow = self._flow
        if flow is None:
            return None
        bundle = self._stage_scalars(z, z + dz)
        section = bundle.flow_section
        assert section is not None
        residual = None
        if flow.residual_terms is not None:
            residual = functools.partial(_host_residual, flow, section)
        return CoupledBlock(
            active=flow.active,
            vectors=flow.vectors[section],
            triangular=flow.triangular[section],
            section=section,
            end_weights=bundle.flow_weights,
            residual=residual,
            # The flow's own eigenbasis and route, as numba_cpu passes its own.
            basis=(
                flow.left[section],
                flow.right[section],
                flow.eigenvalues[section],
                flow.route[section],
            ),
        )

    def _resident_adopted(self, z: float | None) -> None:
        """Make the native state the march's, at `z` when the march moved there."""
        self._state = self._native_state
        if z is not None:
            self._z = z

    def _adams_adopt(self, advanced: ComplexArray, dz: float) -> ComplexArray:
        """Adopt an Adams step through the native finish; keep its host copy."""
        self._native_state.adopt(np.ascontiguousarray(advanced.reshape(-1)), dz)
        self._state = self._native_state
        self._host_state = self._native_state.field_to_host().reshape(
            self._problem.shape
        )
        return self._host_state

    def _dense_stages(self) -> list[ComplexArray]:
        """Return the accepted trial's stage derivatives from the engine (SEP 14)."""
        stages = self._problem.stepper.tableau.n_stages
        return [
            self._native_state.stage_to_host(index).reshape(self._problem.shape)
            for index in range(stages)
        ]

    def _adams_restore(self, z: float, state: ComplexArray) -> None:
        """Resume the Adams march after a dense observation or a resume reloaded it.

        The modified noise model's realisation is part of the native state, and
        `_apply_mask` reloads the field alone, which clears it; it is reloaded with
        the field here, so that a restored march keeps adding it to its sources.
        """
        self._z = z
        self._host_state = state
        self._last_loaded = np.ascontiguousarray(state.reshape(-1))
        self._native_state.set_initial(
            self._last_loaded,
            np.empty(0, dtype=np.complex128)
            if self._noise is None
            else np.ascontiguousarray(self._noise.reshape(-1)),
        )
        self._state = self._native_state

    def _adams_propagator(self, anchor: float, position: float) -> ComplexArray:
        """Return ``U(position, anchor)`` on the host, held for the step's source."""
        bundle = self._stage_scalars(anchor, position)
        propagator = np.exp(self._delta_phi(bundle))
        self._adams_held = (anchor, position, bundle, propagator)
        return propagator

    def _adams_source(
        self,
        lab: ComplexArray,
        anchor: float,
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> tuple[ComplexArray, ComplexArray]:
        """Evaluate the source once natively, at a lab-frame field.

        The engine takes the field in the anchor's frame and evaluates stage zero of
        a trial whose every stage row is the evaluation position's, which returns
        ``U(anchor, position) N(lab, position)``. The shared march supplies a
        purpose label; the admitted optical source requires no atomic accounting.
        """
        held = self._adams_held
        if held is not None and held[0] == anchor and held[1] == position:
            bundle, propagator = held[2], held[3]
        else:
            self._hoist(anchor)
            bundle = self._stage_scalars(anchor, position)
            propagator = np.exp(self._delta_phi(bundle))
        self._adams_held = None
        self._last_bundle = bundle
        arrays, material = self._repeated_arrays(bundle)
        with _envelope_errors():
            source = self._native_state.adams_source(
                np.ascontiguousarray((lab / propagator).reshape(-1)),
                *arrays,
                material,
            )
        return source.reshape(self._problem.shape), propagator

    def _adams_lab_source(
        self,
        lab: ComplexArray,
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> ComplexArray:
        """Evaluate the source once natively, untransported (solprop SEP 16).

        The native stage-zero source with the anchor at the evaluation position, where
        the flow over no distance is exactly one: the sample the exponential
        formulation integrates against the linear flow.
        """
        self._hoist(position)
        source, _ = self._adams_source(lab, position, position, purpose=purpose)
        return source

    def _trial_step(self, z: float, dz: float) -> tuple[float, _core.State]:
        partials = self._prepare_trial(z, dz)
        return self._score_trial(partials), self._native_state

    def _prepare_trial(
        self, z: float, dz: float, *, defer_update: bool = False
    ) -> RealArray:
        """Prepare member stages, optionally deferring the shared accepted update."""
        self._hoist(z)
        tableau = self._problem.stepper.tableau
        # Every stage's scalars and the whole step's, each table read once (solprop
        # kernel schema 28).
        *bundles, whole = self._stage_bundles(
            z, [*(z + float(node) * dz for node in tableau.nodes), z + dz]
        )
        self._last_bundle = bundles[-1]
        arrays, material = self._trial_arrays(bundles, whole)
        phi_delta, phi_at, prefactor, density, taper, preionisation, raman_rates = (
            arrays
        )
        if self._flow is not None:
            material |= self._flow_trial(dz, whole, bundles)
        with _envelope_errors():
            partials = self._native_state.trial_step(
                dz,
                phi_delta,
                phi_at,
                prefactor,
                density,
                taper,
                preionisation,
                raman_rates,
                material,
                prepare_only=defer_update,
            )
        return partials

    def _trial_arrays(
        self, bundles: Sequence[StageScalars], whole: StageScalars
    ) -> tuple[
        tuple[
            RealArray, RealArray, RealArray, RealArray, RealArray, RealArray, RealArray
        ],
        dict[str, npt.NDArray[np.float64] | int],
    ]:
        """Pack the stages' host scalars into the engine's trial arrays."""
        if self._terms.separable is not None:
            phi_delta = np.array([[one.z1 - one.z0] for one in [*bundles, whole]])
            phi_at = np.array([[one.z1] for one in bundles])
            prefactor = np.ones((len(bundles), 1))
        else:
            phi_delta, phi_at, prefactor = self._weight_rows(bundles, whole)
        density = np.ascontiguousarray(
            np.array([bundle.density for bundle in bundles], dtype=np.float64)
        )
        taper = np.ascontiguousarray(
            np.array([bundle.taper_scale for bundle in bundles], dtype=np.float64)
        )
        preionisation = np.ascontiguousarray(
            np.array([bundle.preionisation for bundle in bundles], dtype=np.float64)
        )
        raman_rates = np.ascontiguousarray(
            np.array([bundle.raman_rates for bundle in bundles], dtype=np.float64)
        )
        material: dict[str, npt.NDArray[np.float64] | int] = {
            "kerr_density": np.asarray([one.kerr_density for one in bundles]),
            "plasma_densities": np.asarray([one.plasma_densities for one in bundles]),
            "plasma_preionisation": np.asarray(
                [one.plasma_preionisation for one in bundles]
            ),
            "raman_densities": np.asarray([one.raman_densities for one in bundles]),
        }
        if self._problem.responses.dispersive_kerr is not None:
            # solprop SEP 6's channel densities, each stage's row; the native response
            # weighs its channels and rebuilds a mixture's filter from them.
            material["dispersive_densities"] = np.ascontiguousarray(
                [one.dispersive_kerr_densities for one in bundles], dtype=np.float64
            )
        if self._moving_maps:
            # Schema 14's row: each stage's weights of a moving guide's maps, which the
            # engine combines natively per stage.
            material["transform_weights"] = np.ascontiguousarray(
                np.stack(
                    [
                        _required(one.transform_weights, "transform_weights")
                        for one in bundles
                    ]
                ),
                dtype=np.float64,
            )
        arrays = (
            phi_delta,
            phi_at,
            prefactor,
            density,
            taper,
            preionisation,
            raman_rates,
        )
        return arrays, material

    def _weight_rows(
        self, bundles: Sequence[StageScalars], whole: StageScalars
    ) -> tuple[RealArray, RealArray, RealArray]:
        """Return the stages' weights of ``Phi`` and of the prefactor over the terms.

        Each stage's increment, the whole step's and each stage's value of ``Phi``, and
        each stage's prefactor: the bundles' own weights of a constant or an expansion,
        or, for a z table, the +-1 weights of its slots (solprop SEP 29 G1), whose
        positions are evaluated natively first where no slot holds them.
        """
        tables = self._z_tables
        slot_of: dict[float, int] = {}
        if tables is not None:
            stages = [*bundles, whole]
            slot_of = tables.load(
                [one.z0 for one in stages if tables.phi] + [one.z1 for one in stages]
            )
        if tables is not None and tables.phi:
            phi_delta = np.stack(
                [tables.phi_row(slot_of, one.z0, one.z1) for one in [*bundles, whole]]
            )
            phi_at = np.stack(
                [tables.at_row(slot_of, one.z1, tables.n_phi) for one in bundles]
            )
        else:
            phi_delta = np.stack(
                [
                    *(_required(bundle.phi_delta, "phi_delta") for bundle in bundles),
                    _required(whole.phi_delta, "whole phi_delta"),
                ]
            )
            phi_at = np.stack(
                [_required(bundle.phi_at, "phi_at") for bundle in bundles]
            )
        if tables is not None and tables.prefactor:
            prefactor = np.stack(
                [tables.at_row(slot_of, one.z1, tables.slots) for one in bundles]
            )
        else:
            prefactor = np.stack(
                [_required(bundle.prefactor, "prefactor") for bundle in bundles]
            )
        return (
            np.ascontiguousarray(phi_delta),
            np.ascontiguousarray(phi_at),
            np.ascontiguousarray(prefactor),
        )

    def _phi_increment(self, z0: float, z1: float) -> RealArray:
        """Return the weights of ``Phi(z1) - Phi(z0)`` over the native terms.

        The bundle's own for a constant or an expansion; a z table's two slots,
        evaluated where no slot holds them, for a z table (solprop SEP 29 G1).
        """
        tables = self._z_tables
        if tables is not None and tables.phi:
            return tables.phi_row(tables.load([z0, z1]), z0, z1)
        return _required(self._stage_scalars(z0, z1).phi_delta, "phi_delta")

    def _rate_weights(self, z: float, *, before: bool = False) -> RealArray:
        """Return ``D(z)``'s weights over the native ``Phi`` terms (solprop SEP 16).

        solprop's :func:`~solprop.translate.linop.rate_weights` for a constant or an
        expansion; for a z table, the rate table evaluated natively into its slot
        (solprop SEP 29 G1).
        """
        tables = self._z_tables
        if tables is not None and tables.phi:
            return tables.rate(z, before=before)
        return rate_weights(
            self._problem.linear, z, before=before, helpers=self.helpers
        )

    def _delta_phi(self, bundle: StageScalars) -> ComplexArray:
        """Return the stage's exponent on the host, a z table's from its native slots.

        A host engine's propagator and the dense observation's stiffness test read it;
        a z table's two values are copied from the native slots and subtracted once, as
        the stage kernels subtract them (solprop SEP 29 G1).
        """
        tables = self._z_tables
        if tables is not None and tables.phi:
            here = tables.values("phi", bundle.z1)
            anchored = tables.values("phi", bundle.z0)
            return (here - anchored).reshape(self._problem.shape)
        return super()._delta_phi(bundle)

    def _unitary_phase(self, bundle: StageScalars) -> RealArray:
        """Return ``Im Phi`` at the stage's position, a z table's from its slot."""
        tables = self._z_tables
        if tables is not None and tables.phi:
            return np.imag(tables.values("phi", bundle.z1)).reshape(self._problem.shape)
        return super()._unitary_phase(bundle)

    def _flow_trial(
        self, dz: float, whole: StageScalars, bundles: Sequence[StageScalars]
    ) -> dict[str, npt.NDArray[np.float64] | int]:
        """Form the step's flow actions natively if its section or length changed.

        The generator is frozen across a step, so the actions depend on the section
        and the stage lengths alone, and a fixed schedule or a repeated accepted
        length reuses them: exactly one set is held, as in solprop's reference
        backend, and a step crossing into the next section rebuilds, which is the
        cache invalidation at a section edge. The engine forms them from the
        section's eigenbasis (or by scaling and squaring where it is ill-conditioned)
        and measures the whole step's conditioning; the host refuses a length whose
        exponent needs too many squarings before, and checks the measurement against
        the section's certificate after.
        """
        flow = self._flow
        assert flow is not None
        section = whole.flow_section
        key = (section, float(dz))
        if key != self._flow_key:
            # Forgotten first: a refusal below leaves no set of actions behind a key.
            self._flow_key = None
            fractions = np.array(
                [*(float(node) for node in self._problem.stepper.tableau.nodes), 1.0],
                dtype=np.float64,
            )
            for fraction in fractions:
                if fraction != 0.0:
                    _flow_squarings(
                        float(fraction) * float(dz) * float(flow.row_sum[section])
                    )
            measured = self._native_state.form_flow_actions(
                int(section), float(dz), fractions
            )
            _check_flow_conditioning(flow, section, measured[0], dz)
            self._flow_key = key
        if flow.residual_terms is None:
            weights = np.empty(0, dtype=np.float64)
        else:
            weights = np.ascontiguousarray(
                np.stack(
                    [_required(one.flow_weights, "flow_weights") for one in bundles]
                ),
                dtype=np.float64,
            )
        return {"flow_section": section, "flow_weights": weights}

    def _score_trial(self, partials: RealArray) -> float:
        """Reduce one member's partials in the established singleton order."""
        old = math.sqrt(float(np.asarray(partials[0], dtype=np.float64).sum()))
        new = math.sqrt(float(np.asarray(partials[1], dtype=np.float64).sum()))
        residual = math.sqrt(float(np.asarray(partials[2], dtype=np.float64).sum()))
        scale = self._problem.stepper.controller.scale(old, new)
        if isinstance(scale, np.ndarray):
            native_scale = np.ascontiguousarray(scale, dtype=np.float64)
            weighted = self._native_state.weighted_error(native_scale)
            return math.sqrt(float(np.asarray(weighted).sum()))
        return residual / float(scale)

    def _bind_monitor(self, monitor: BoundaryMonitor | None) -> None:
        """Hold the boundary monitor's weights in the native state (solprop SEP 21).

        From here on the engine's finish adds a split-step collar's removal at every
        accepted step, on every march, and :meth:`_boundary_energies` reads it.
        """
        super()._bind_monitor(monitor)
        if monitor is None:
            self._native_state.unbind_monitor()
            return
        columns = (
            np.empty(0, dtype=np.float64)
            if monitor.columns is None
            else np.ascontiguousarray(monitor.columns, dtype=np.float64)
        )
        self._native_state.bind_monitor(
            np.ascontiguousarray(monitor.weights, dtype=np.float64).reshape(-1), columns
        )

    def _boundary_energies(self, state: _core.State) -> BoundaryEnergies:
        """Reduce the monitor's five numbers where the state lives (solprop SEP 21).

        Only they and the split's removal since the last reading cross to the host.
        """
        if state is not self._native_state:
            raise BackendError("the monitored state does not belong to this backend")
        if self._monitor is None:
            raise BackendError(f"{self.name}: no boundary monitor is attached")
        reading = np.asarray(state.monitor_reading(), dtype=np.float64).reshape(-1)
        return BoundaryEnergies(
            float(reading[0]), reading[1:5].copy(), removed=float(reading[5])
        )

    def _finish(self, state: _core.State, dz: float) -> _core.State:
        if state is not self._native_state:
            raise BackendError("the advanced state does not belong to this backend")
        state.finish(dz)
        return state

    def _begin_plane(self) -> None:
        # The recycled stage is kept across a plane (solprop SEP 14 phase 3); a resume
        # or a dense observation reloads the state, which clears it.
        self._alarm = math.nan
        self._alarm_reported = False

    def _on_accepted_step(self) -> None:
        # The step's quadrature score stays on the device; `_check_quadrature` reads
        # its peak once a march, not once a step (a device synchronisation each).
        self._native_state.promote_fsal()

    def _check_quadrature(self, position: float) -> None:
        """Warn once a plane if any step's quadrature score passed the alarm."""
        peak = self._native_state.quadrature_peak()
        if math.isnan(peak):
            return
        # The last step's score, as the statistic reports it; the warning reads the
        # largest of the march's.
        self._alarm = self._native_state.quadrature_error
        if peak > QUADRATURE_ALARM and not self._alarm_reported:
            self._alarm_reported = True
            logger.warning(
                "the transverse quadrature reproduces the nonlinear polarisation to "
                "only %.2e by z = %.6g m, above the %.0e alarm; the integrand is not "
                "resolved by this rule (a kinked ionisation rate, or too few nodes "
                "for the mode set) and no step size can fix it",
                peak,
                position,
                QUADRATURE_ALARM,
            )

    def _collar_changed(self) -> None:
        """Pass the boundary monitor's quiet collar to the native stage (SEP 21).

        The reused stage was evaluated under the other setting, so it is dropped.
        """
        self._native_state.set_absorber_quiet(self._collar_quiet)
        self._native_state.clear_fsal()

    def _observed_field(self, state: _core.State) -> ComplexArray:
        if state is not self._native_state:
            raise BackendError("the observed state does not belong to this backend")
        return state.field_to_host().reshape(self._problem.shape)

    def _raman_spectrum(
        self, rates: tuple[float, ...], densities: tuple[float, ...] = ()
    ) -> None:
        """Prepare the native Raman kernel for private memoization diagnostics."""
        if not self._problem.responses.raman_terms:
            return None
        self._native_state.prepare_raman(
            np.ascontiguousarray(np.asarray(rates, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(densities, dtype=np.float64)),
        )

    @property
    def _raman_transforms(self) -> int:
        """Return the number of native Raman-kernel transforms."""
        return int(self._native_state.raman_kernel_transforms)

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Collect field and response statistics without observing the full field."""
        self._require_state()
        grid_value = self._require_grid()
        if self._problem.transform.modal_tables is not None:
            return self._general_statistics(spec, grid_value)
        integration_names = set(PLASMA_INTEGRATION_STATISTICS) & set(spec.names)
        if integration_names and (
            not self._problem.responses.population_channels
            or self._problem.responses.envelope_plasma is not None
        ):
            raise BackendError(
                "the plasma integration diagnostics read the field-grid plasma "
                "column, and this problem carries none"
            )
        spatial_names: set[str] = (
            set(spec.names) & (_SPATIAL_STAT_FLAGS.keys() | _FAR_FIELD_FLAGS.keys())
            if self._problem.transform.geometry is not None
            else set()
        )
        refusal = axis_refusal(self._problem, spec.names)
        if refusal is not None:
            raise BackendError(refusal)
        axis_names = set(spec.names) & set(ON_AXIS_PLASMA_STATISTICS)
        scan: tuple[RealArray, RealArray] | None = None
        population_names = {name for name in spec.names if name.startswith("species_")}
        if (
            len(self._problem.responses.population_channels) > 1
            or self._problem.responses.envelope_plasma is not None
        ):
            population_names.update(
                set(spec.names) & {"electron_density", "peak_ionisation_rate"}
            )
        native_names = tuple(
            name
            for name in spec.names
            if name in _STAT_FLAGS
            and name not in spatial_names
            and name not in population_names
        )
        if self._problem.transform.geometry is not None:
            # Mirror numpy_ref: on Hankel or DFT columns the temporal statistics
            # describe no observable, and the Cartesian native buffers hold one
            # node chunk, not every column.
            response_names = {
                "electron_density",
                "peak_ionisation_rate",
                "quadrature_error",
            }
            for name in native_names:
                if name not in response_names:
                    raise BackendError(
                        f"the {name!r} statistic is not a spatial saved-field statistic"
                    )
        plasma_names = {"electron_density", "peak_ionisation_rate"}
        if (
            plasma_names & set(native_names)
            and self._problem.responses.plasma_coefficients is None
        ):
            raise BackendError(
                "the electron density and the peak ionisation rate need a compiled "
                "plasma response, and this problem declares none"
            )
        on_axis = np.empty(0, dtype=np.float64)
        if "peak_intensity" in native_names:
            if self._modes is None:
                raise BackendError(
                    "the peak on-axis intensity needs the mode set at construction"
                )
            on_axis = np.ascontiguousarray(self._on_axis_synthesis(self._modes))
        plasma = self._problem.responses.plasma_coefficients
        density = 0.0 if plasma is None else float(plasma.density(self._z))
        preionisation = 0.0 if plasma is None else float(plasma.preionisation(self._z))
        taper_profile = self._problem.transform.taper_scale
        taper = 1.0 if taper_profile is None else float(taper_profile(self._z))
        flags = 0
        for name in native_names:
            flags |= _STAT_FLAGS[name]
        channels = self._problem.responses.population_channels
        if population_names and channels:
            flags |= _POPULATIONS_FLAG
        if integration_names:
            flags |= _PLASMA_INTEGRATION_FLAG
        with _envelope_errors():
            packed = (
                np.empty(0, dtype=np.float64)
                if flags == 0
                else np.asarray(
                    self._native_state.statistics(
                        flags,
                        np.ascontiguousarray(spec.energy_windows),
                        on_axis,
                        density,
                        taper,
                        preionisation,
                        0.5 * EPS0 * C,
                        np.asarray(
                            [
                                float(channel.preionisation(self._z))
                                for channel in channels
                            ]
                        ),
                        power_weight=(
                            np.empty(0)
                            if self._problem.transform.geometry is not None
                            else np.ascontiguousarray(
                                power_weight_at(self._problem, self._z)
                            )
                        ),
                    )
                ).ravel()
            )
        # Packed last: the six maxima the reference's three ratios are formed from.
        parts = packed[-_PLASMA_INTEGRATION_PARTS:] if integration_names else packed[:0]
        if integration_names:
            packed = packed[:-_PLASMA_INTEGRATION_PARTS]
        count = (
            2 * len(channels) * self._problem.transform.n_nodes
            if population_names and channels
            else 0
        )
        values = self._unpack_statistics(
            native_names, spec, packed[:-count] if count else packed, grid_value
        )
        if population_names:
            populations = (
                packed[-count:].reshape(
                    self._problem.transform.n_nodes, len(channels), 2
                )
                if count
                else np.empty((self._problem.transform.n_nodes, 0, 2))
            )
            reduced = population_statistics(
                self._problem, self._z, populations[:, :, 0].T, populations[:, :, 1].T
            )
            values.update({name: reduced[name][None] for name in population_names})
            if count:
                scan = (populations[:, :, 0].T, populations[:, :, 1].T)
        if axis_names:
            # SEP 19: the axis row is one host scan per saved plane, shared with the
            # other backends; a Radau rule's axis node comes from the device's own
            # node scan of this call when the species statistics asked for it.
            field = self._native_state.field_to_host().reshape(self._problem.shape)
            axis = self._axis_statistics(field[0], scan)
            values.update({name: axis[name][None] for name in axis_names})
        if integration_names:
            ratios = dict(
                zip(
                    PLASMA_INTEGRATION_STATISTICS,
                    plasma_integration_statistics(parts),
                    strict=True,
                )
            )
            values.update(
                {name: np.asarray([ratios[name]]) for name in integration_names}
            )
        if spatial_names:
            values.update(self._spatial_statistics(spatial_names, spec))
        values.update(self._profile_statistics(spec))
        return values

    def _general_statistics(
        self, spec: StatsSpec, grid: TimeGrid
    ) -> dict[str, np.ndarray]:
        """Collect a general modal run's statistics from the observed channel field.

        The route's admitted names are functionals of the physical flux ``a.H W a``
        of the basis metric, which solprop's own ``collect`` evaluates; none is a
        backend-owned response statistic. The accessor whitelist is checked first, as
        ``numpy_ref`` checks it, because a caller may ask a backend directly.
        """
        refusal = refusal_for(
            declared_capability(general_modal=True, spatial=False),
            spec.names,
            withheld=withheld_accessors(
                general_modal=True,
                coupled_flow=self._problem.linear.coupled_flow is not None,
            ),
        )
        if refusal is not None:
            raise BackendError(refusal)
        metric = (
            metric_at(self._problem.linear.modal_metric, self._z)
            if set(spec.names) & METRIC_WEIGHTED_STATISTICS
            else None
        )
        field = self._observed_field(self._native_state)
        members = [
            collect(
                spec,
                grid,
                SavedPlane(z=self._z, field=field[member]),
                modes=self._modes,
                profiles=self._profiles,
                metric=metric,
            )
            for member in range(field.shape[0])
        ]
        return {
            name: np.stack([member[name] for member in members]) for name in spec.names
        }

    def _spatial_statistics(
        self, names: set[str], spec: StatsSpec | None = None
    ) -> dict[str, np.ndarray]:
        """Reduce on the device and interpolate only the saved radial fluence.

        On a Cartesian or line grid the far-field statistics and ``on_axis_spectrum``
        come from the native far-field pass (solprop SEP 23 section 6).
        """
        geometry = self._problem.transform.geometry
        if geometry is None:
            raise BackendError("spatial statistics require spatial geometry")
        folded = (
            names & _FAR_FIELD_FLAGS.keys()
            if geometry.kind == "free"
            else names & set(FAR_FIELD_STATISTICS)
        )
        values: dict[str, np.ndarray] = {}
        if folded:
            values.update(self._far_field_statistics(folded, spec))
            names = names - folded
            if not names:
                return values
        flags = 0
        for name in names:
            flags |= _SPATIAL_STAT_FLAGS[name]
        packed = np.asarray(
            self._native_state.spatial_statistics(flags, 0.5 * EPS0 * C, self._z)
        ).ravel()
        offset = 0
        for name in (
            "energy",
            "on_axis_intensity",
            "peak_intensity",
            "on_axis_spectrum",
        ):
            if name in names:
                count = self._problem.shape[-1] if name == "on_axis_spectrum" else 1
                data = packed[offset : offset + count]
                values[name] = data[None] if name == "on_axis_spectrum" else data
                offset += count
        if names & {"beam_radius", "fwhm_r"}:
            # The signed fluence, the axis fluence and each node's magnitude, whose
            # largest value fixes the rounding budget (solprop SEP 34).
            count = self._problem.transform.n_nodes
            fluence = packed[offset : offset + count]
            axis_fluence = float(packed[offset + count])
            magnitude = packed[offset + count + 1 : offset + 2 * count + 1]
            n_pol = self._problem.shape[2]
            budget = fluence_budget(
                magnitude,
                n_pol * self._problem.grids.n_t + self._problem.transform.n_modes,
            )
            widths = signed_fluence_widths(geometry, fluence, budget, axis_fluence)
            if "beam_radius" in names:
                values["beam_radius"] = np.asarray([widths[:2]])
            if "fwhm_r" in names:
                values["fwhm_r"] = np.asarray([widths[2]])
            offset += 2 * count + 1
        if offset != packed.size:
            raise BackendError(
                f"native spatial statistics returned {packed.size} values; "
                f"consumed {offset}"
            )
        return values

    def _far_field_statistics(
        self, names: set[str], spec: StatsSpec | None
    ) -> dict[str, np.ndarray]:
        """Form the far-field statistics in the native banded pass (SEP 23 section 6).

        The tables are those solprop's :class:`~solprop.backends.spatial.FarFieldTables`
        forms for the call's spec, packed as ``(n_apertures, 6 + n_omega)`` rows of
        centre, half-width, reciprocal width, edge, distance and the scale row, and
        ``(n_regions, 4)`` boxes; the pass returns each statistic finished, in
        :data:`_FAR_FIELD_ORDER`.
        """
        far = names & set(FAR_FIELD_STATISTICS)
        tables = (
            None
            if not far
            else self._far_field.for_spec(
                self._problem.stats_spec if spec is None else spec
            )
        )
        if far and tables is None:
            raise BackendError(
                f"the far-field statistics {sorted(far)} need the tables the backend "
                "forms at construction"
            )
        n_omega = self._problem.shape[-1]
        apertures, regions, code, exponent = _aperture_payload(tables, n_omega)
        flags = 0
        for name in names:
            flags |= _FAR_FIELD_FLAGS[name]
        packed = np.asarray(
            self._native_state.aperture_statistics(
                flags, apertures, code, exponent, regions, 0.5 * EPS0 * C
            )
        ).ravel()
        n_apertures = apertures.shape[0]
        rows = {
            "beam_spectrum": None,
            "absorbed_power": 0,
            "region_spectrum": regions.shape[0],
            "aperture_spectrum": n_apertures,
            "aperture_on_axis_spectrum": n_apertures,
            "on_axis_spectrum": None,
        }
        values: dict[str, np.ndarray] = {}
        offset = 0
        for name in _FAR_FIELD_ORDER:
            if name not in names:
                continue
            count = rows[name]
            if name == "absorbed_power":
                size, shape = 4, (1, 4)
            elif count is None:
                size, shape = n_omega, (1, n_omega)
            else:
                size, shape = count * n_omega, (1, count, n_omega)
            values[name] = packed[offset : offset + size].reshape(shape)
            offset += size
        if offset != packed.size:
            raise BackendError(
                f"native far-field statistics returned {packed.size} values; "
                f"consumed {offset}"
            )
        return values

    def _unpack_statistics(
        self,
        names: tuple[str, ...],
        spec: StatsSpec,
        packed: RealArray,
        grid: TimeGrid,
    ) -> dict[str, np.ndarray]:
        """Unpack the native fixed-order scalar transfer into catalogue shapes."""
        requested = set(names)
        modes = self._problem.shape[1]
        polarisations = self._problem.shape[2]
        columns = modes * polarisations
        offset = 0
        values: dict[str, np.ndarray] = {}

        def take(size: int) -> RealArray:
            nonlocal offset
            result = packed[offset : offset + size]
            offset += size
            return result

        if "omega0" in requested:
            values["omega0"] = take(columns).reshape(1, modes, polarisations)
        if "energy" in requested:
            values["energy"] = take(columns).reshape(1, modes, polarisations)
        if "energy_window" in requested:
            values["energy_window"] = take(spec.n_windows * columns).reshape(
                1, spec.n_windows, modes, polarisations
            )
        if "peak_power" in requested:
            values["peak_power"] = take(modes).reshape(1, modes)
        if "peak_power_window" in requested:
            values["peak_power_window"] = take(spec.n_windows * modes).reshape(
                1, spec.n_windows, modes
            )
        if "peak_power_total" in requested:
            values["peak_power_total"] = take(1)
        for name in (
            "electron_density",
            "peak_ionisation_rate",
            "peak_intensity",
            "quadrature_error",
        ):
            if name in requested:
                values[name] = take(1)
        if {"fwhm_t", "fwhm_t_outer"} & requested:
            power = take(grid.n_t)
            if "fwhm_t" in requested:
                values["fwhm_t"] = np.array(
                    [_half_width(grid.t, power, outermost=False)]
                )
            if "fwhm_t_outer" in requested:
                values["fwhm_t_outer"] = np.array(
                    [_half_width(grid.t, power, outermost=True)]
                )
        if offset != packed.size:
            raise BackendError(
                f"native statistics returned {packed.size} values; consumed {offset}"
            )
        return values

    def _profile_statistics(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Evaluate profile-only and ZDW statistics on their owning host objects."""
        values: dict[str, np.ndarray] = {}
        for name in spec.names:
            if (
                name in _STAT_FLAGS
                or name in PLASMA_INTEGRATION_STATISTICS
                or name in ON_AXIS_PLASMA_STATISTICS
                or name.startswith("species_")
                or (
                    self._problem.transform.geometry is not None
                    and (name in _SPATIAL_STAT_FLAGS or name in _FAR_FIELD_FLAGS)
                )
            ):
                continue
            if name in PROFILE_STATISTICS:
                if self._profiles is None or name not in self._profiles:
                    raise BackendError(
                        f"the {name} statistic needs the {name!r} profile"
                    )
                values[name] = np.array([profile_value(self._profiles[name], self._z)])
            elif name == "zdw":
                if self._modes is None:
                    raise BackendError(
                        "the zero-dispersion wavelength needs the mode set"
                    )
                values[name] = np.stack(
                    [
                        zdw(
                            self._modes,
                            self._z,
                            wavelength_range=inband_wavelengths(self._require_grid()),
                        )
                    ]
                )
            else:
                raise BackendError(f"the native backend cannot collect {name!r}")
        return values

    def metadata(self) -> dict[str, object]:
        """Return the shared execution metadata plus the native build record.

        The build record and the installed solprop version cannot change within a
        process (thread settings are read once, at import), so they are read once
        (:func:`_build_facts`); the native state's counters are read at every call.
        """
        facts = _build_facts()
        work = super().metadata()
        if self._adams is None and self._low_storage is None:
            work["source_evaluations"] = self._native_state.rk_source_evaluations
        march = self._adams
        if isinstance(march, _NativeAdamsMarch):
            # Which march stepped the run (solprop SEP 39 section 5): a timing or a
            # rounding difference is traced to it.
            work["adams_march"] = "native" if march.native else "host"
        return {
            **work,
            **facts.info,
            "solprop_version": facts.solprop_version,
            "thread_budget": THREAD_BUDGET.threads,
            "thread_budget_source": THREAD_BUDGET.source,
            "openmp_wait_policy": WAIT_POLICY,
            "openmp_proc_bind": PROC_BIND,
            "planning_seconds": self._native_state.planning_seconds,
            "raman_fft_provider": self._native_state.raman_fft_provider,
            "raman_fft_tasks": self._native_state.raman_fft_tasks,
            "execution_sizing": self._native_state.execution_sizing,
            "raman_planning_seconds": self._native_state.raman_planning_seconds,
            "raman_enabled": self._problem.flags.has_raman,
            "raman_terms": sum(
                term.n_terms for term in self._problem.responses.raman_terms
            ),
            "plasma_channels": len(self._problem.responses.population_channels),
            "envelope_plasma_enabled": self._problem.responses.envelope_plasma
            is not None,
            "envelope_refinement_boundary": "scalar status read before transforms",
            "raman_components": len(self._problem.responses.raman_terms),
            "raman_drive": (
                "none"
                if not self._problem.responses.raman_terms
                else "field_squared"
                if self._problem.responses.raman_terms[0].thg
                and self._problem.grid_kind == "field"
                else "cycle_averaged"
            ),
            "raman_kernel_transforms": self._native_state.raman_kernel_transforms,
            "fsal_ready": self._native_state.fsal_ready,
            "dispersive_path": dict(self._native_state.dispersive_path) or None,
            "synthesis_provider": self._native_state.synthesis_provider,
            "projection_provider": self._native_state.projection_provider,
            "gemm_shape_policy": self._native_state.gemm_shape_policy,
            "device_team_size": self._native_state.response_team_size,
            "device_team_size_limit": facts.device_team_size,
            "device_team_size_policy": (
                "256 below n_to=8192; configured limit otherwise"
                if facts.info["device_backend"] == "cuda"
                else "fixed configured team width"
            ),
            "device_reduction_size": facts.device_reduction_size,
        }

    def available_device_memory(self) -> int | None:
        """Return free accelerator bytes, or None for a host build.

        Returns
        -------
        int or None
            Free bytes from the selected accelerator runtime.
        """
        return _core.available_device_memory()

    def memory_ledger(self) -> dict[str, int]:
        """Return every persistent native allocation's byte count by role."""
        return {
            str(name): int(value)
            for name, value in self._native_state.memory_ledger().items()
        }

    def close(self) -> None:
        """Release native plans and views; repeated calls have no effect."""
        if not self._closed:
            self._native_state.close()
        self._state = None
        self._noise = None
        self._closed = True


@dataclasses.dataclass(frozen=True)
class _BuildFacts:
    """The parts of a backend's metadata that are fixed for the process."""

    info: Mapping[str, object]
    solprop_version: str
    device_team_size: int
    device_reduction_size: int


@functools.cache
def _build_facts() -> _BuildFacts:
    """Return the build record and the installed solprop version, read once.

    Reading them costs about a millisecond (the package metadata lookup and the native
    record), which a per-plane metadata record paid at every saved plane.

    Returns
    -------
    _BuildFacts
        The build record as a mapping, solprop's version and the device team sizes.
    """
    native_record = {
        key: value for key, value in _core.build_record().items() if value is not None
    }
    return _BuildFacts(
        info=types.MappingProxyType(dataclasses.asdict(build_info())),
        solprop_version=version("solprop"),
        device_team_size=int(native_record["device_team_size"]),
        device_reduction_size=int(native_record["device_reduction_size"]),
    )


def _repeated_row(values: tuple[float, ...], count: int) -> RealArray:
    """Return a stage row of host scalars repeated `count` times, ``(count, n)``."""
    return np.repeat(np.array([values], dtype=np.float64), count, axis=0)


def _required(values: RealArray | None, name: str) -> RealArray:
    if values is None:
        raise BackendError(f"device-ready problem has no {name}")
    return values


def _half_width(t: RealArray, power: RealArray, *, outermost: bool) -> float:
    """Return the reference central or outer half-maximum width of a power trace."""
    peak = float(np.max(power))
    if peak <= 0.0:
        return math.nan
    level = 0.5 * peak
    apex = int(np.argmax(power))
    if outermost:
        above = np.flatnonzero(power > level)
        left, right = int(above[0]), int(above[-1])
        pairs = ((left - 1, left), (right + 1, right))
    else:
        low = np.flatnonzero(power[:apex] < level)
        high = np.flatnonzero(power[apex:] < level)
        if low.size == 0 or high.size == 0:
            return math.nan
        left, right = int(low[-1]), apex + int(high[0])
        pairs = ((left, left + 1), (right, right - 1))
    if pairs[0][0] < 0 or pairs[1][0] >= power.size:
        return math.nan

    def crossing(pair: tuple[int, int]) -> float:
        low, high = pair
        span = power[high] - power[low]
        return float(t[low] + (level - power[low]) * (t[high] - t[low]) / span)

    return crossing(pairs[1]) - crossing(pairs[0])


def _require_initial_array(field: object, shape: tuple[int, ...]) -> None:
    if not isinstance(field, np.ndarray):
        raise TypeError("field must be a NumPy ndarray")
    array = cast(npt.NDArray[np.generic], field)
    if array.dtype != np.dtype(np.complex128):
        raise TypeError(f"field must have dtype complex128, got {array.dtype}")
    if array.ndim != 4:
        raise ValueError(f"field must have rank 4, got rank {array.ndim}")
    if array.shape != shape:
        raise BackendError(
            f"the initial state has shape {array.shape}, expected {shape}"
        )
    if not array.flags.c_contiguous:
        raise ValueError("field must be C-contiguous")


type _TrialArrays = tuple[
    tuple[RealArray, RealArray, RealArray, RealArray, RealArray, RealArray, RealArray],
    dict[str, npt.NDArray[np.float64] | int],
]


@dataclasses.dataclass
class _ResidentAdams:
    """solprop's Adams history kept in the native state (solprop SEP 16 phase 3b).

    An engine for solprop's ``AdamsMarch`` (``solprop.backends.multistep.AdamsEngine``):
    the history's rows, a trial's predictor, its source, the corrected field and the
    defect norms, the adoption and transport of an accepted step and dense observation
    run in the engine, with only the trial's scalars, coefficient tables and norms
    crossing. solprop's host engine (``HostHistory``) is its reference. The backend
    hands it the native state and three of its own operations.
    """

    native: _core.State
    method: Multistep
    controller: object
    n_omega: int
    arrays: Callable[[float, float], _TrialArrays]
    """``(anchor, position) -> trial arrays`` with every stage row at `position`."""
    sample: Callable[[_TrialArrays, float], _TrialArrays]
    """``(step arrays, end) -> trial arrays`` anchored at the step's end, `end`."""
    hoist: Callable[[float], None]
    adopted: Callable[[float | None], None]
    """Make the native state the backend's, at a new position when one is given."""
    rate: Callable[..., RealArray]
    """``(z, *, before=False) -> D(z)``'s weights over the native ``Phi`` terms."""
    shape: tuple[int, ...]
    """The state's shape, ``(n_batch, n_modes, n_pol, n_omega)``."""
    helpers: Helpers
    """The backend's Chebyshev and Adams-basis helpers (``NativeBackend.helpers``)."""
    interpolate: bool
    compensated: bool = False
    """Whether trials carry a varying operator's transported compensation."""
    carrier: CarrierReference | None = None
    """The run's carrier frame (solprop SEP 35): its shift is formed on the host from
    the reference element's chord, one scalar a trial, and nothing is read back."""
    staged: bool = False
    """Whether the native state forms an exponential trial's tables, stage scalars
    and rate itself, from tables uploaded once (solprop SEP 39 phase 1)."""
    dense: None = None
    _trial: tuple[float, float, list[float], npt.NDArray[np.int64], int] | None = None
    _step: tuple[float, float, npt.NDArray[np.float64]] | None = None
    _exponential: bool = False
    _device_scale: bool = False
    _coefficients: npt.NDArray[np.float64] | None = None
    _dense_nodes: npt.NDArray[np.float64] | None = None
    _trial_carrier: RealArray = dataclasses.field(default_factory=lambda: _NO_CARRIER)
    _dense_carrier: RealArray = dataclasses.field(default_factory=lambda: _NO_CARRIER)

    def __post_init__(self) -> None:
        """Allocate the history's rows in the native state."""
        self.native.adams_allocate(self.n_slots, self.compensated, self.interpolate)
        self._exponential = self.method.formulation == "exponential"
        # solprop's own controller's scale is formed on the device each trial, so that
        # the state's magnitude never crosses; any other controller keeps the host's.
        controller = self.controller
        self._device_scale = isinstance(controller, PIController) and (
            type(controller).scale is PIController.scale
        )
        if self._device_scale:
            assert isinstance(controller, PIController)
            weights = controller.weights
            self.native.adams_scale(
                float(controller.rtol),
                float(controller.atol),
                np.empty(0, dtype=np.float64)
                if weights is None
                else np.ascontiguousarray(weights, dtype=np.float64),
            )

    @property
    def n_slots(self) -> int:
        """The history's rows: the method's order and two more."""
        return self.method.order + 2

    @property
    def device_scale(self) -> bool:
        """Whether the norm's scale is solprop's own, formed in the native state."""
        return self._device_scale

    def march_stepped(
        self,
        step: tuple[float, float, list[float]] | None,
        *,
        fused: bool,
        carrier: Sequence[float],
    ) -> None:
        """Adopt what a native march call leaves (solprop SEP 39 phase 2).

        No trial is held, and the last accepted step's anchor, step and nodes ``(1,
        theta...)``, if the call accepted one, set the dense observation's
        interpolant: the fused path's nodes, or the exponential corrector's
        coefficients on them, with the step's carrier rotations.
        """
        self._trial = None
        if step is not None:
            z, dz, nodes = step
            interpolant = (
                np.array(nodes)
                if fused
                else lagrange_coefficients(nodes, check=False, helpers=self.helpers)
            )
            self._step = (z, dz, interpolant) if self.interpolate else None
            self._dense_nodes = np.array(nodes)
            self._dense_carrier = (
                np.asarray(carrier, dtype=np.float64) if carrier else _NO_CARRIER
            )
        self.adopted(None)

    def reset(self) -> None:
        """Forget the last accepted step and any trial in flight."""
        self._trial = None
        self._step = None
        self._dense_carrier = _NO_CARRIER

    def _carrier(self, z: float, dz: float, nodes: list[float], rows: int) -> RealArray:
        """Return a trial's packed carrier frame, empty without one.

        ``(Im sigma, Re f_0, Im f_0, ...)`` over the `rows` history rows the trial
        reads, ``f_t = exp(sigma (1 - theta_t))``; ``sigma`` is the reference element's
        chord over the step (solprop SEP 35 section 6.1), formed from the operator's
        own representation, so a device build reads nothing back.
        """
        carrier = self.carrier
        if carrier is None or not carrier.active:
            return _NO_CARRIER
        assert carrier.chord is not None
        # One batch member per native state.
        shift = float(np.imag(carrier.chord.exponent(z, z + dz))[0])
        packed = np.empty(1 + 2 * rows, dtype=np.float64)
        packed[0] = shift
        # exp(i shift s) as NumPy's complex exponential forms it: exp(0) (cos + i sin).
        spans = 1.0 - np.asarray(nodes[:rows], dtype=np.float64)
        packed[1::2] = np.cos(shift * spans)
        packed[2::2] = np.sin(shift * spans)
        return packed

    def hold(self) -> object:
        """Keep the accepted state aside in the engine across a dense observation."""
        self.native.adams_hold()
        return None

    def restore(self, z: float, held: object) -> None:
        """Make the held accepted state current again at `z`."""
        del held
        self.native.adams_restore()
        self.adopted(z)

    def start(self, z: float, *, stored: bool) -> None:
        """Evaluate the history's first sample at the accepted state into row zero."""
        del stored
        self.hoist(z)
        arrays, material = self.arrays(z, z)
        with _envelope_errors():
            self.native.adams_start(0, *arrays, material)
        if self.compensated:
            # The startup sample's state and rate, read on the far side of a kink.
            self.native.adams_compensation_start(0, self.rate(z))

    def value(self, slot: int) -> None:
        """The rows are resident: there is no host row to name."""
        del slot

    # --- persistence across a resume (solprop SEP 29 G2) ------------------------------

    def snapshot_rows(self) -> dict[str, ComplexArray]:
        """Return host copies of the rows a resume needs to continue the history.

        The rows ``HostHistory`` also writes keep its names and layout,
        ``(n_slots, n_batch, n_modes, n_pol, n_omega)``: ``"store"`` and, compensated,
        ``"states"`` and ``"rates"``. A frozen-residual flow's history holds each
        sample's generator applied to its state, ``"applied"``, where the host engine
        holds the sample's weights (``"weights_<row>"``); the two are not
        interchangeable, and each engine refuses the other's. The copies are lossless.
        Only the trial's scratch stays behind; the step's bookkeeping is the march's.
        """
        expected = (self.n_slots, *self.shape)
        return {
            name: self.native.adams_rows_to_host(name).reshape(expected)
            for name in self.native.adams_row_names()
        }

    def load_rows(self, rows: Mapping[str, np.ndarray], shape: tuple[int, ...]) -> None:
        """Install rows `snapshot_rows` returned, for a state of `shape`.

        Everything is checked before any row is copied, so a refused snapshot leaves
        the history as it was. The trial in flight and the last step's interpolant
        are forgotten, as ``HostHistory.load_rows`` forgets them: the march's next step
        is a new trial.

        Raises
        ------
        BackendError
            Naming the native backend and the mismatch: rows of another engine (a host
            engine's frozen-residual weights), a row this history lacks or does not
            keep, or one of the wrong shape or type.
        """
        names = self.native.adams_row_names()
        weights = sorted(name for name in rows if name.startswith("weights_"))
        if weights:
            listed = ", ".join(weights)
            raise BackendError(
                "the native backend cannot continue this Adams history: the snapshot "
                f"holds the host engine's frozen-residual weights ({listed}), and the "
                "native engine keeps each sample's applied generator ('applied') "
                "instead"
            )
        foreign = sorted(set(rows) - set(names))
        missing = [name for name in names if name not in rows]
        if foreign or missing:
            raise BackendError(
                "the native backend cannot continue this Adams history: its engine "
                f"keeps the rows {names}, and the snapshot "
                + (f"lacks {missing}" if missing else f"also holds {foreign}")
            )
        expected = (self.n_slots, *shape)
        for name in names:
            row = rows[name]
            if row.shape != expected or not np.iscomplexobj(row):
                raise BackendError(
                    f"the snapshot's {name!r} rows ({row.dtype}, shape {row.shape}) do "
                    f"not fit the native backend's complex store of shape {expected}"
                )
        self.reset()
        for name in names:
            flat = np.ascontiguousarray(rows[name], dtype=np.complex128)
            self.native.adams_rows_from_host(
                name, flat.reshape(self.n_slots, -1, shape[-1])
            )

    def fused_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: npt.NDArray[np.int64],
        order: int,
        orders: list[int],
    ) -> RealArray:
        """Form an interaction-formulation trial natively; return its scored norms.

        A staged engine forms the tables and the stage scalars in the native state from
        the nodes alone (solprop SEP 39 phase 3), by the same operations as the host's.
        """
        if self.staged:
            with _envelope_errors():
                norms = self.native.adams_fused_trial_at(
                    z,
                    dz,
                    np.asarray(nodes, dtype=np.float64),
                    np.ascontiguousarray(
                        slots[: max(max(orders), order)], dtype=np.uint64
                    ),
                    order,
                    list(orders),
                    self._divisor(),
                )
            self._trial = (z, dz, nodes, slots, order)
            return np.asarray(norms, dtype=np.float64).reshape(-1)
        self.hoist(z)
        predictor, corrector, rows = fused_tables(
            nodes, order, orders, helpers=self.helpers
        )
        divisor = self._divisor()
        width = rows.shape[1]
        arrays, material = self.arrays(z, z + dz)
        with _envelope_errors():
            norms = self.native.adams_fused_trial(
                dz,
                np.ascontiguousarray(slots[: max(width - 1, order)], dtype=np.uint64),
                np.ascontiguousarray(predictor, dtype=np.float64),
                np.ascontiguousarray(corrector, dtype=np.float64),
                np.ascontiguousarray(rows, dtype=np.float64).reshape(-1),
                divisor,
                *arrays,
                material,
            )
        self._trial = (z, dz, nodes, slots, order)
        return np.asarray(norms, dtype=np.float64).reshape(-1)

    def fused_accept(
        self, keep: npt.NDArray[np.int64], free: int, *, interpolate: bool
    ) -> None:
        """Adopt the held trial in the engine; keep its interpolant there."""
        assert self._trial is not None
        z, dz, nodes, slots, order = self._trial
        self._trial = None
        self.native.adams_fused_accept(
            dz,
            np.ascontiguousarray(keep, dtype=np.uint64),
            free,
            np.ascontiguousarray(slots[:order], dtype=np.uint64),
            interpolate,
        )
        self.adopted(None)
        self._step = (z, dz, np.array([1.0, *nodes[:order]])) if interpolate else None

    def _divisor(self) -> RealArray:
        """Return the norm's scale per bin: the accepted state's before the step.

        Empty when the device forms it (`__post_init__`).
        """
        if self._device_scale:
            return _NO_DIVISOR
        magnitude = self.native.adams_magnitude()
        scale = cast("ControllerLike", self.controller).scale(magnitude, magnitude)
        return np.ascontiguousarray(
            np.broadcast_to(np.asarray(scale, dtype=np.float64), (self.n_omega,))
        )

    def exponential_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: npt.NDArray[np.int64],
        order: int,
        orders: list[int],
    ) -> RealArray:
        """Form an exponential trial natively; return its scored norms.

        The step's exponent, flow and moments come from the trial's own arrays; the
        predicted field's source is evaluated in the laboratory frame, at a trial
        whose anchor is the step's end. A staged engine forms the tables, the stage
        scalars and the rate in the native state from the nodes alone (solprop SEP 39
        phase 1), by the same operations as the host's.
        """
        if self.staged:
            return self._staged_trial(z, dz, nodes, slots, order, orders)
        self.hoist(z)
        # Every table in one call of the backend's helpers; each is a fresh
        # C-contiguous float64 array, as the native call takes it.
        predictor, corrector, rows, spans, carried, endpoint, defects = trial_tables(
            nodes, order, orders, compensated=self.compensated, helpers=self.helpers
        )
        divisor = self._divisor()
        width = corrector.shape[0]
        step, material = self.arrays(z, z + dz)
        sample, sample_material = self.sample((step, material), z + dz)
        rate = _NO_RATE
        if self.compensated:
            rate = np.ascontiguousarray(
                self.rate(z + dz, before=True), dtype=np.float64
            )
        carrier = self._carrier(z, dz, nodes, width - 1)
        with _envelope_errors():
            norms = self.native.adams_exponential_trial(
                dz,
                np.ascontiguousarray(slots[: max(width - 1, order)], dtype=np.uint64),
                predictor.reshape(-1),
                corrector.reshape(-1),
                rows.reshape(-1),
                divisor,
                *step,
                material,
                *sample,
                sample_material,
                spans,
                carried,
                endpoint,
                defects.reshape(-1),
                rate,
                carrier,
            )
        self._trial = (z, dz, nodes, slots, order)
        self._trial_carrier = carrier
        self._coefficients = corrector[: order + 1, : order + 1].copy()
        self._dense_nodes = np.array([1.0, *nodes[:order]])
        return np.asarray(norms, dtype=np.float64).reshape(-1)

    def _staged_trial(
        self,
        z: float,
        dz: float,
        nodes: list[float],
        slots: npt.NDArray[np.int64],
        order: int,
        orders: list[int],
    ) -> RealArray:
        """Form an exponential trial from its nodes alone; return its scored norms.

        The native state forms the coefficient tables (``lagrange::trial_tables``),
        the stage scalars and the rate from the tables uploaded once; the divisor and
        the carrier frame's shift are the host's, as on the other path. A run that
        interpolates its planes keeps the corrector's coefficients on the host for its
        observations, from the same basis.
        """
        width = 1 + max(orders)
        carrier = self._carrier(z, dz, nodes, width - 1)
        with _envelope_errors():
            norms = self.native.adams_exponential_trial_at(
                z,
                dz,
                np.asarray(nodes, dtype=np.float64),
                np.ascontiguousarray(slots[: max(width - 1, order)], dtype=np.uint64),
                order,
                list(orders),
                self.compensated,
                self._divisor(),
                carrier,
            )
        self._trial = (z, dz, nodes, slots, order)
        self._trial_carrier = carrier
        self._coefficients = (
            lagrange_coefficients(
                [1.0, *nodes[:order]], check=False, helpers=self.helpers
            )
            if self.interpolate
            else _NO_COEFFICIENTS
        )
        self._dense_nodes = np.array([1.0, *nodes[:order]])
        return np.asarray(norms, dtype=np.float64).reshape(-1)

    def exponential_accept(
        self, keep: npt.NDArray[np.int64], free: int, *, interpolate: bool
    ) -> None:
        """Adopt the held trial in the engine; the new sample takes row `free`."""
        del keep  # untransported samples are never re-anchored
        assert self._trial is not None
        z, dz, _, slots, order = self._trial
        self._trial = None
        self.native.adams_exponential_accept(
            dz,
            free,
            np.ascontiguousarray(slots[:order], dtype=np.uint64),
            interpolate,
        )
        self.adopted(None)
        coefficients = self._coefficients
        assert coefficients is not None
        self._step = (z, dz, coefficients) if interpolate else None
        # The accepted step's shift and the rotations of the `order` history rows its
        # interpolant reads, held across later trials and rejections.
        trial = self._trial_carrier
        self._dense_carrier = trial[: 1 + 2 * order] if trial.size else _NO_CARRIER

    def observe(self, position: float) -> object:
        """Interpolate the last accepted step natively into the engine's state."""
        assert self._step is not None
        z, dz, nodes = self._step
        if self._exponential:
            # `nodes` holds the accepted corrector's coefficients here; a varying
            # operator's compensation is interpolated over the step's own samples.
            xi = (position - z) / dz
            weights = thetas = np.empty(0, dtype=np.float64)
            if self.compensated:
                assert self._dense_nodes is not None
                thetas = self._dense_nodes
                weights = integrated_weights(
                    thetas, xi, check=False, helpers=self.helpers
                )
            self.native.adams_exponential_observe(
                xi,
                dz,
                np.ascontiguousarray(nodes, dtype=np.float64).reshape(-1),
                np.ascontiguousarray(weights, dtype=np.float64),
                np.ascontiguousarray(thetas, dtype=np.float64),
                np.ascontiguousarray(self._dense_carrier),
            )
            return self.native
        theta = (position - z) / dz
        if getattr(self.method, "interpolant", "adams") == "adams":
            weights = dz * integrated_weights(
                nodes, theta, check=False, helpers=self.helpers
            )
            hermite = False
        else:
            theta2 = theta * theta
            theta3 = theta2 * theta
            weights = np.array(
                [
                    2 * theta3 - 3 * theta2 + 1,
                    dz * (theta3 - 2 * theta2 + theta),
                    -2 * theta3 + 3 * theta2,
                    dz * (theta3 - theta2),
                ]
            )
            hermite = True
        arrays, material = self.arrays(z, position)
        self.native.adams_observe(
            np.ascontiguousarray(weights, dtype=np.float64), hermite, *arrays, material
        )
        return self.native


_NO_DIVISOR = np.empty(0, dtype=np.float64)
"""The empty divisor that asks the native trial to form the scale itself."""

_NO_SCHEDULE = np.empty(0, dtype=np.float64)
"""The empty schedule of an adaptive march's configuration."""

_NO_CARRIER = np.empty(0, dtype=np.float64)
"""The empty carrier frame: the native trial takes the unshifted arithmetic."""

_NO_RATE = np.empty(0, dtype=np.float64)
"""The empty endpoint rate of an uncompensated trial."""

_NO_COEFFICIENTS = np.empty((0, 0), dtype=np.float64)
"""The corrector's coefficients a staged trial keeps when no plane is interpolated."""

_STAGE_CONSTANT = 1
_STAGE_EXPANSION = 2
_STAGE_TABLE = 3
"""How the stage tables name a constant, an expanded or a z table's ``Phi`` and
prefactor (``stage::Terms``)."""


def _section_tables(
    problem: KernelProblem, *, moving: bool, residual: bool
) -> dict[str, object]:
    """Return a coupled flow's and a moving guide's sections as the stage tables take
    them: their edges, degrees and the edges' slack (`flow_edge_slack`,
    `transform_section`), and whether the flow carries a frozen residual."""
    flow = problem.linear.coupled_flow
    maps = problem.transform.modal_tables if moving else None
    out: dict[str, object] = {
        "flow_edges": [],
        "flow_degrees": 0,
        "flow_slack": 0.0,
        "flow_residual": False,
        "transform_edges": [],
        "transform_degrees": 0,
        "transform_slack": 0.0,
    }
    if flow is not None:
        out |= {
            "flow_edges": np.asarray(flow.z_edges, dtype=np.float64).tolist(),
            "flow_degrees": int(flow.n_deg),
            "flow_slack": float(flow_edge_slack(flow)),
            "flow_residual": residual,
        }
    if maps is not None and maps.synthesis is not None:
        edges = np.asarray(maps.z_edges, dtype=np.float64)
        out |= {
            "transform_edges": edges.tolist(),
            "transform_degrees": int(maps.synthesis.shape[1]),
            "transform_slack": _EDGE_SLACK_FRACTION * float(edges[-1] - edges[0]),
        }
    return out


def _chord_tables(carrier: CarrierReference | None) -> dict[str, object]:
    """Return the carrier frame's chord as the native stage tables take it.

    The imaginary part of the reference element's operator for the one batch member
    (`CarrierChord`): a constant's rate, or an expansion's terms over the ``Phi``
    weights the tables already hold. None without an active frame, and none for a z
    table's chord, whose run steps solprop's march.
    """
    chord = None if carrier is None or not carrier.active else carrier.chord
    if chord is not None and chord.constant is not None:
        return {
            "chord": 1,
            "chord_rate": float(np.imag(chord.constant)[0]),
            "chord_terms": [],
        }
    if chord is not None and chord.expansion is not None:
        terms = np.imag(np.asarray(chord.expansion.terms))[:, 0]
        return {"chord": 2, "chord_rate": 0.0, "chord_terms": terms.tolist()}
    return {"chord": 0, "chord_rate": 0.0, "chord_terms": []}


def _on_device() -> bool:
    """Return whether the installed build executes on an accelerator."""
    return _build_facts().info["device_backend"] != "none"


def _low_storage_payload(problem: KernelProblem, *, device: bool) -> dict[str, object]:
    """Return the native problem's low-storage rows (solprop kernel schema 32).

    The march's own allocation: its registers in place of the Runge-Kutta views, the
    stage register elided where solprop's estimate elides it, and the flows solprop's
    estimate counts -- except that a device build recomputes under ``"auto"``, where
    the estimate's CPU count over-states it by the table.
    """
    if not isinstance(problem.stepper.integrator, LowStorageRungeKutta):
        return {}
    flows = low_storage_flows(problem, "native")
    if device and problem.stepper.propagator_mode == "auto":
        flows = 0
    return {
        "low_storage": True,
        "low_storage_stage_register": low_storage_stage_register(problem, "native"),
        "low_storage_flows": flows,
    }


_ADAMS_MARCH = os.environ.get("SOLPROP_ADAMS_MARCH", "native")
"""Which march steps an admitted exponential ``Multistep()`` run: ``"native"``, or
``"host"`` for solprop's Python march (solprop SEP 39 section 5). A private switch for
the benchmark harness's A/B comparisons, read once at import."""

_MARCH_COUNTERS = (
    "startup",
    "predictor",
    "rejected",
    "accepted_steps",
    "rejected_steps",
    "restarts",
    "failure_restarts",
    "dense_observations",
)
"""The call ledger's counters the native march keeps."""

_REACHED, _HOST_STEP, _MONITOR, _BOUND, _BUDGET, _FLOOR, _ERROR, _DIVERGED = range(8)
"""Why a native march call returned (``march::Reason``)."""

_CONVERGENCE, _BACKEND, _VALUE = 1, 2, 3
"""Which error a native failure maps to (``march::Category``)."""


class _MarchRunner(Protocol):
    """What runs the native march: the native state, or a script in the tests."""

    def run(
        self, state: dict[str, object], bounds: dict[str, object]
    ) -> dict[str, Any]: ...


class _NativeAdamsMarch(AdamsMarch):
    """solprop's exponential Adams march, executed natively (solprop SEP 39 phase 2).

    The algorithm is ``AdamsMarch``'s, which stays the reference; the native march
    performs it on the resident engine for a run of attempts and returns only at a host
    event: the target reached or crossed, an accepted step on which the driver has work
    (a statistics row, a natural stop, a monitor check), the call's bound, the budget,
    the floor or a native error. Between calls the march's state is this object's, as
    the Python march keeps it, so a snapshot, a load, a new run and a reset are the
    reference's own; each call takes the state and hands it back.

    One ``committing()`` spans each call, the state's synchronisation and the step
    history's rows, so a signal arriving meanwhile is raised with the host's records in
    step with the native state. The progress callbacks, the debug lines and the event
    step's work run after it, and an error is raised last.

    A configuration the native march does not cover runs the Python march
    (:attr:`refusal`), with a warning unless the private switch asked for it.
    """

    MAX_ATTEMPTS: ClassVar[int] = 256
    """A call's most attempts, the bound on a signal's wait (solprop SEP 39 Q3)."""
    MAX_SECONDS: ClassVar[float] = 0.1
    """A call's most wall time, s."""

    def __init__(
        self,
        backend: StepDriver[Any],
        method: Multistep,
        *,
        march: Literal["native", "host"] = "native",
        runner: _MarchRunner | None = None,
        kernels: AdamsKernels | None = None,
        exponential: ExponentialKernels | None = None,
    ) -> None:
        """Build the Python march, then decide which march steps the run.

        `runner`, the scripted march of the decision-trace tests, takes the native
        state's place; `kernels` and `exponential` are the Python march's own.
        """
        # solprop's signature names the driver unparameterised.
        super().__init__(backend, method, kernels, exponential)  # pyright: ignore[reportUnknownMemberType]
        # The driver's protected bookkeeping, which this march books into exactly as
        # solprop's does (``DRIVER_SURFACE``): held untyped, since it is the driver's
        # own surface the march reaches, as `AdamsMarch` itself reaches it.
        self._driver: Any = backend
        self._runner = runner
        self._configured: tuple[float | None] | None = None
        refusal, warn = self._native_refusal(march)
        self.refusal = refusal
        """Why the Python march steps the run, or ``None`` when the native one does."""
        if refusal is not None and warn:
            warnings.warn(
                f"solprop-native steps this {method.name} run with solprop's Python "
                f"march: {refusal}. The native march (SEP 39) covers Multistep() in "
                "either formulation and the one-order-higher PEC comparisons on "
                "every route with the resident engine and PIController's norm",
                HostMarchWarning,
                stacklevel=4,
            )

    @property
    def native(self) -> bool:
        """Whether the native march steps the run."""
        return self.refusal is None

    def _native_refusal(self, march: str) -> tuple[str | None, bool]:
        """Return why the native march does not step the run, and whether to warn."""
        if march == "host" or _ADAMS_MARCH == "host":
            return "the private switch asked for it", False
        engine = self._engine
        reasons = (
            (
                self._general is not None,
                "a comparison method's general path (PECE, explicit or matched pairs)",
            ),
        )
        for failed, reason in reasons:
            if failed:
                return reason, True
        if self._runner is not None:
            return None, False
        if not isinstance(engine, _ResidentAdams):
            return (
                "the host engine, which steps a coupled flow off the eigenbasis "
                "route or wider than 16 channels"
            ), True
        # Device builds are qualified one vendor at a time (SEP 39 section 6.7): CUDA
        # on NVIDIA A40s; HIP waits for its own GPU segment.
        if _build_facts().info["device_backend"] == "hip":
            return "a HIP build, which the native march is not yet qualified on", True
        assert isinstance(engine, _ResidentAdams)
        chord = None if engine.carrier is None else engine.carrier.chord
        resident = (
            (
                not engine.staged,
                "an operator or prefactor representation the native stage tables "
                "do not read",
            ),
            (not engine.device_scale, "a controller with a norm scale of its own"),
            (
                self._policy.carrier and (chord is None or chord.table is not None),
                "the carrier frame on a z table",
            ),
        )
        for failed, reason in resident:
            if failed:
                return reason, True
        return None, False

    def march_to(self, target: float) -> None:
        """Step the backend's state to `target`, natively when the march is admitted.

        Parameters
        ----------
        target : float
            Position to reach, m.

        Raises
        ------
        ConvergenceError
            If the proposal falls below the contract's ``dz_min``, or a native
            refinement fails.
        ResponseBudgetError
            If the next trial would exceed the method's call budget.
        """
        if not self.native:
            super().march_to(target)
            return
        backend = self._driver
        if self._ahead is not None:
            # The march is ahead of the last observation; resume it there.
            self._engine.restore(*self._ahead)
            self._ahead = None
        if backend._natural_stop_due(target):
            return
        self._configure()
        schedule = backend._problem.stepper.schedule
        if schedule is not None:
            # A replay takes the schedule's steps as given (`AdamsMarch._replay_to`).
            slack = replay_tolerance(schedule, target)
            while (
                backend._schedule_index < schedule.n_steps
                and backend._z < target - slack
            ):
                if self._call(target, slack):
                    return
            if backend._z < target - slack:
                msg = (
                    f"the replayed schedule ends at {float(schedule.z_end):.6g} m and "
                    f"cannot reach z = {target:.6g} m"
                )
                raise BackendError(msg)
            return
        while not reached(backend._z, target):
            if self._call(target):
                return
        if self._interpolate and backend._z > target:
            self._ahead = (backend._z, self._engine.hold())
            backend._state = cast("Any", self._observe(target))
            backend._z = target
            self.ledger.dense_observations += 1

    def _configure(self) -> None:
        """Hand the native state the run's policy, again when the monitor changed."""
        monitor = self._driver._monitor
        interval = None if monitor is None else float(monitor.interval)
        if self._configured == (interval,):
            return
        policy = dataclasses.replace(self._policy, quiet_interval=interval)
        if self._runner is None:
            backend = self._driver
            engine = cast("_ResidentAdams", self._engine)
            schedule = backend._problem.stepper.schedule
            steps = edges = _NO_SCHEDULE
            if schedule is not None:
                steps = np.ascontiguousarray(schedule.steps, dtype=np.float64)
                edges = np.ascontiguousarray(schedule.edges, dtype=np.float64)
            backend._native_state.adams_march_configure(
                dataclasses.asdict(policy),
                self._fused,
                engine.compensated,
                engine.interpolate,
                steps,
                edges,
            )
        self._configured = (interval,)

    def _bounds(self, target: float, slack: float | None) -> dict[str, object]:
        """Return a call's bounds: the target, the steps the driver has work on and,
        for a replay, the edges' slack."""
        backend = self._driver
        steps: list[int] = []
        if backend._row_spec is not None:
            period = backend._row_period
            steps.append(period - backend._accepted_steps % period)
        every = backend._problem.stepper.step_planes
        if every:
            steps.append(every - backend._natural_count % every)
        monitor = backend._monitor
        checked = None
        if monitor is not None:
            steps.append(monitor.steps_to_check())
            checked = monitor.checked_at
        return {
            "target": float(target),
            "quiet": bool(backend._collar_quiet),
            "host_step": int(min(steps)) if steps else 0,
            "monitor": checked is not None,
            "first_check": False,
            "check_interval": 0.0 if monitor is None else float(monitor.interval),
            "last_check": 0.0 if checked is None else float(checked),
            "max_attempts": self.MAX_ATTEMPTS,
            "max_seconds": self.MAX_SECONDS,
            "replay": slack is not None,
            "slack": 0.0 if slack is None else float(slack),
        }

    def _state(self) -> dict[str, object]:
        """Return the march's state as the native call takes it."""
        backend, ledger = self._driver, self.ledger
        return {
            "z": float(backend._z),
            "proposal": float(backend._proposal),
            "order": self._order,
            "rejections": self._rejections,
            "last_rejected": self._last_rejected,
            "collar_was_quiet": self._collar_was_quiet,
            "cursor": backend._schedule_index,
            "history": [
                (entry.z, entry.identity == "predicted", entry.slot)
                for entry in self._history
            ],
            "ledger": {name: getattr(ledger, name) for name in _MARCH_COUNTERS},
        }

    def _call(self, target: float, slack: float | None = None) -> bool:
        """Make one native call and drain it; return whether a natural stop is due.

        `slack` is a replay's (:func:`solprop.backends.driver.replay_tolerance`), or
        ``None`` for an adaptive march.
        """
        backend = self._driver
        bounds = self._bounds(target, slack)
        started = time.perf_counter() - backend._clock
        with committing():
            result = self._native_run(self._state(), bounds)
            self._synchronise(result)
            rows = np.asarray(result["rows"], dtype=np.float64).reshape(-1, 6)
            reason = int(result["reason"])
            # Only the last accepted row of an event can have the driver's work; the
            # others take the skip path, as the reference's would.
            event = reason in (_REACHED, _HOST_STEP, _MONITOR)
            last = len(rows) - 1
            for index, (end, dz, norm, accepted, _, wall) in enumerate(rows.tolist()):
                backend._record_attempt(end, dz, norm, bool(accepted), started + wall)
                if slack is not None:
                    backend._accepted_norms.append(norm)
                final = index == last and (event or reason == _FLOOR)
                if accepted and not final:
                    backend._skip_step(end)
        progress = backend._progress
        for end, dz, norm, accepted, order, _ in rows.tolist():
            progress.step(end, dz, bool(accepted))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "native Adams attempt to z = %.9g m: dz = %.4g m, order %d, "
                    "norm %.4g, %s",
                    end,
                    dz,
                    int(order),
                    norm,
                    "accepted" if accepted else "rejected",
                )
        if event and rows.size and rows[-1, 3]:
            backend._take_step_stats()
            if backend._natural_stop(target):
                return True
        self._raise(reason, result)
        return False

    def _native_run(
        self, state: dict[str, object], bounds: dict[str, object]
    ) -> dict[str, Any]:
        """Run the native march from `state` to an event within `bounds`."""
        if self._runner is not None:
            return self._runner.run(state, bounds)
        native = self._driver._native_state
        with _envelope_errors():
            return native.adams_march_run(state, bounds)

    def _synchronise(self, result: dict[str, Any]) -> None:
        """Adopt the state a native call handed back."""
        backend, engine = self._driver, self._engine
        state = result["state"]
        backend._z = float(state["z"])
        backend._proposal = float(state["proposal"])
        backend._schedule_index = int(state["cursor"])
        self._order = int(state["order"])
        self._rejections = int(state["rejections"])
        self._last_rejected = bool(state["last_rejected"])
        self._collar_was_quiet = bool(state["collar_was_quiet"])
        self._history = [
            _Derivative(
                float(z), None, "predicted" if predicted else "accepted", int(slot)
            )
            for z, predicted, slot in state["history"]
        ]
        ledger = self.ledger
        for name, value in state["ledger"].items():
            setattr(ledger, name, int(value))
        ledger.orders.extend(int(order) for order in result["orders"])
        if isinstance(engine, _ResidentAdams):
            step = None
            if result["stepped"]:
                order = int(result["step_order"])
                nodes = [1.0, *(float(node) for node in result["step_nodes"][:order])]
                step = (float(result["step_z"]), float(result["step_dz"]), nodes)
            engine.march_stepped(
                step, fused=self._fused, carrier=list(result["step_carrier"])
            )

    def _raise(self, reason: int, result: dict[str, Any]) -> None:
        """Raise the error a call returned with, if any."""
        if reason == _BUDGET:
            raise self._budget_error(int(result["cost"]))
        if reason == _FLOOR:
            proposed = float(result["floor_proposal"])
            msg = (
                f"step control asked for {proposed:.3e} m, below the floor dz_min ="
                f" {self._policy.dz_min:.3e} m; the tolerance cannot be met here"
            )
            raise ConvergenceError(msg)
        if reason == _DIVERGED:
            dz, end = float(result["replay_dz"]), float(result["replay_end"])
            norm = float(result["replay_norm"])
            msg = (
                f"{self._driver.name}: a replayed {self._method.name} step of "
                f"{dz:.4g} m ending at z = {end:.6g} m has a non-finite error norm "
                f"({norm!r}): the field has diverged. A replayed schedule takes "
                "its steps as given and cannot reject one, so a step beyond an "
                "explicit stability limit -- the in-stage temporal collar's or a "
                "coupled flow's advisory step -- grows until it overflows; the "
                "translation refuses a declared step past the collar's limit and "
                "warns on one past the flow's. Take shorter steps, or let the "
                "controller choose them"
            )
            raise ConvergenceError(msg)
        if reason == _ERROR:
            message = str(result["message"])
            category = int(result["category"])
            if category == _CONVERGENCE:
                raise ConvergenceError(message)
            if category == _VALUE:
                raise ValueError(message)
            raise RuntimeError(message)


class _ResidentLowStorage:
    """solprop's low-storage RK4IP engine on the native state (SEP 24 section 6.7).

    A ``solprop.backends.low_storage.LowStorageEngine``: the native state is the
    register ``A`` and holds ``B``, ``C`` (unless elided, solprop's Q4) and the flows;
    a step is one native call making the four register passes in the route's own
    scratch (the transverse scratch, or the modal fine spectrum) through the stage core
    the Runge-Kutta march shares, accepted in place. Only the step's host scalars
    cross: one bundle per position (the start, the midpoint, the end) and the flows'
    intervals.

    **The flows.** A constant operator tabulating holds one volume, ``exp(h_t D / 2)``
    for the key ``h_t`` of solprop's ``table_steps`` (the step's run's mean, SEP 24
    Q10), built from the operator's origin as solprop's engines build it and rebuilt
    only when the key changes; a varying operator tabulating holds the step's ``P``
    and ``Q``; recomputing holds none and forms each flow per element in the finish
    that applies it -- on a modal route from the phase terms, so a varying operator's
    flows are never transients outside the ledger.

    **The stiffness indicator** (SEP 24 Q11) is not formed, as on ``numba_cpu``:
    ``stiffness`` is NaN. Recovering ``g1`` and ``F2`` needs ``C``, which is elided on
    the Cartesian and line routes the march is for or split across the exchange's two
    evaluations, and two more reductions a step.

    Parameters
    ----------
    backend : NativeBackend
        The backend whose native state this engine steps.
    method : solprop.stepper.low_storage.LowStorageRungeKutta
        The march.

    Attributes
    ----------
    stiffness : float
        Always NaN: not formed on this backend.
    atomic : bool
        Always ``True``: the step is one native call, during which no signal handler
        runs, so solprop's march commits the step, the finish and the move together.
    """

    def __init__(self, backend: NativeBackend, method: LowStorageRungeKutta) -> None:
        """Hold the backend and the schedule's table keys."""
        self._backend = backend
        self._method = method
        self.stiffness = math.nan
        self.atomic = True
        problem = backend._problem  # pyright: ignore[reportPrivateUsage] - the owner's problem
        payload = _low_storage_payload(problem, device=_on_device())
        self._flows = int(cast(int, payload["low_storage_flows"]))
        schedule = problem.stepper.schedule
        assert schedule is not None  # the translation refused the march without one
        self._keys, self._edges = table_steps(schedule), schedule.edges
        self._table_step = math.nan
        self._separable = backend._terms.separable is not None  # pyright: ignore[reportPrivateUsage] - the owner's fixed terms

    def reset(self) -> None:
        """Drop the table's key: a resume builds it again from its own step."""
        self._table_step = math.nan

    def close(self) -> None:
        """Forget the table; the native state releases the views when it closes."""
        self.reset()

    def step(self, z: float, dz: float) -> float:
        """Take one step in four native register passes, accepted in place.

        Parameters
        ----------
        z : float
            Where the step starts, m.
        dz : float
            The step, m, positive.

        Returns
        -------
        float
            The accepted state's squared norm over the norm mask, before the boundary
            treatments.
        """
        backend = self._backend
        midpoint, end = z + 0.5 * dz, z + dz
        # The engine reads its owner's tables, as solprop's engines read theirs.
        hoist = backend._hoist  # pyright: ignore[reportPrivateUsage]
        bundles_at = backend._stage_bundles  # pyright: ignore[reportPrivateUsage]
        # A z table's increments are its slots' (solprop SEP 29 G1).
        increment = backend._phi_increment  # pyright: ignore[reportPrivateUsage]
        problem = backend._problem  # pyright: ignore[reportPrivateUsage]
        hoist(z)
        # Each evaluation reads its position as its own anchor (solprop's engines);
        # a modal flow formed for the step reads its interval.
        modal = not self._separable and self._flows != 1
        bundles = [
            bundles_at(z, [z, midpoint] if modal else [z])[0],
            bundles_at(midpoint, [midpoint, end] if modal else [midpoint])[0],
            bundles_at(end, [end])[0],
        ]
        stages = problem.stepper.tableau.n_stages
        padded = bundles + [bundles[-1]] * (stages - len(bundles))
        arrays, material = backend._trial_arrays(padded, bundles[-1])  # pyright: ignore[reportPrivateUsage]
        phi_delta = np.zeros_like(arrays[0])
        rebuild = False
        key = math.nan
        if self._flows == 1:
            key = schedule_key(self._keys, self._edges, z, dz)
            rebuild = key != self._table_step
        if self._separable:
            # A constant operator's increment is the distance itself.
            phi_delta[0:2, 0] = 0.5 * dz
            if rebuild:
                phi_delta[2, 0] = 0.5 * key
        else:
            if modal:
                phi_delta[0] = increment(z, midpoint)
                phi_delta[1] = increment(midpoint, end)
            if rebuild:
                # From the operator's origin, not fl(z + h/2) - z (solprop's engines).
                origin = float(problem.linear.z_span[0])
                half = origin + 0.5 * key
                bundles_at(origin, [half])
                phi_delta[2] = increment(origin, half)
        with _envelope_errors():
            squared = backend._native_state.low_storage_step(  # pyright: ignore[reportPrivateUsage]
                dz,
                np.ascontiguousarray(phi_delta),
                *arrays[1:],
                material,
                rebuild,
            )
        if rebuild:
            self._table_step = key
        return float(squared)


_TABLE_POINTS = 16
"""The most positions one native table evaluation takes (``z_table::max_points``)."""


class _ZTables:
    """The general z tables of one native state (solprop SEP 29 section 6.6, G1).

    solprop falls back to a Chebyshev table of the state's size for ``Phi``
    (``linear.phi_coefficients``) or the response prefactor (``responses.prefactor``)
    when neither is a weighted sum of fixed arrays. The coefficients cross once, at
    construction (``table_upload``), and stay resident; a trial's positions are then
    evaluated natively into the native state's term slots, ``slots`` of them, and the
    stage weights select them: ``Phi(z1) - Phi(z0)`` is the row ``e_1 - e_0`` and
    ``Phi(z1)`` the row ``e_1``, so every native consumer of the terms (the Runge-Kutta
    stages, the low-storage flows, the Adams phase and the noise) reads the table
    unchanged, as one rounded subtraction of two values, as solprop's reference forms
    it. The prefactor's slots are its values at the same positions, one-hot weighted.

    The slots are kept as a least-recently-used set of positions, in the native state
    (``stage::ZSlots``, solprop SEP 39 phase 4), where the native Adams march reads
    them too: each call's positions are evaluated only where no slot holds them yet,
    so an accepted step's end, which is the next step's anchor, is never evaluated
    twice. A call's positions
    number at most ``slots`` (the tableau's distinct stage fractions, five for the
    low-storage march), so none of one call's positions is evicted by another of its
    own. A compensated Adams march reads ``D = dPhi/dz`` from the derivative table
    (``linear.rate``) into one more ``Phi`` slot, ``rate_slot``.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A problem that took the z table for ``Phi``, the prefactor or both.

    Attributes
    ----------
    phi, prefactor : bool
        Which quantities are z tables.
    slots : int
        The position slots, each one value of each table.
    rate_slot : int or None
        The ``Phi`` term that holds the rate, or ``None`` when no march reads it.
    """

    def __init__(self, problem: KernelProblem) -> None:
        """Size the slots from the problem; nothing crosses until :meth:`upload`."""
        self._problem = problem
        self.phi = problem.linear.phi_representation == "table"
        self.prefactor = problem.responses.prefactor_representation == "table"
        integrator = problem.stepper.integrator
        self.slots = z_table_slots(problem)
        if self.slots > _TABLE_POINTS:
            raise BackendError(
                f"native evaluates a z table at {_TABLE_POINTS} positions a call, and "
                f"this tableau's stages need {self.slots}"
            )
        compensated = (
            self.phi
            and isinstance(integrator, Multistep)
            and integrator.formulation == "exponential"
            and not source_absent(problem)
        )
        self.rate_slot = self.slots if compensated else None
        self._native: _core.State | None = None

    @property
    def evaluations(self) -> int:
        """Positions evaluated so far, a count for the tests and the benchmarks."""
        return 0 if self._native is None else self._native.table_evaluations()

    @property
    def n_phi(self) -> int:
        """The native ``Phi`` term count: the slots, and the rate's when it is read."""
        return self.slots + (self.rate_slot is not None)

    @property
    def nbytes(self) -> int:
        """The bytes of the coefficients that cross to the native state."""
        coefficients = 0
        if self.phi:
            phi = self._problem.linear.phi_coefficients
            assert phi is not None
            coefficients += phi.nbytes
            if self.rate_slot is not None:
                coefficients += self._problem.linear.rate.coefficients.nbytes
        if self.prefactor:
            table = self._problem.responses.prefactor
            assert table is not None
            coefficients += table.coefficients.nbytes
        return int(coefficients)

    def upload(self, native: _core.State) -> None:
        """Make the tables resident in `native`, once, before its first trial."""
        self._native = native
        native.table_slots(
            self.phi, self.prefactor, self.slots, self.rate_slot is not None
        )
        if self.phi:
            self._upload("phi", self._problem.linear.polynomial)
            if self.rate_slot is not None:
                self._upload("rate", self._problem.linear.rate)
        if self.prefactor:
            table = self._problem.responses.prefactor
            assert table is not None
            self._upload("prefactor", table)

    def _upload(self, kind: str, table: PiecewiseChebyshev) -> None:
        """Send one table's breakpoints and coefficients, and its admitted span."""
        assert self._native is not None
        low, high = table.z_span
        slack = _EDGE_SLACK_FRACTION * (high - low)
        n_intervals, n_terms = table.coefficients.shape[:2]
        self._native.table_upload(
            kind,
            np.ascontiguousarray(table.breakpoints, dtype=np.float64),
            low - slack,
            high + slack,
            np.ascontiguousarray(table.coefficients, dtype=np.complex128).reshape(
                n_intervals, n_terms, -1
            ),
        )

    def load(self, positions: Sequence[float]) -> dict[float, int]:
        """Return each position's slot, evaluating those no slot holds, in one call.

        Parameters
        ----------
        positions : sequence of float
            The positions one native call reads, m, at most ``slots`` distinct.

        Returns
        -------
        dict
            Position to slot.
        """
        assert self._native is not None
        wanted = [float(z) for z in positions]
        try:
            slots = self._native.table_load(wanted)
        except ValueError as error:
            raise BackendError(str(error)) from error
        return dict(zip(wanted, slots, strict=True))

    def phi_row(self, slot_of: Mapping[float, int], z0: float, z1: float) -> RealArray:
        """Return the weights of ``Phi(z1) - Phi(z0)`` over the native terms."""
        row = np.zeros(self.n_phi)
        row[slot_of[z1]] += 1.0
        row[slot_of[z0]] -= 1.0
        return row

    def at_row(self, slot_of: Mapping[float, int], z: float, count: int) -> RealArray:
        """Return the one-hot weights of a table's value at `z` over `count` terms."""
        row = np.zeros(count)
        row[slot_of[z]] = 1.0
        return row

    def rate(self, z: float, *, before: bool) -> RealArray:
        """Evaluate ``D(z)`` into the rate slot; return its weights over the terms.

        Parameters
        ----------
        z : float
            Position, m.
        before : bool
            Read an interior breakpoint from the interval ending there.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_phi,)``, one at the rate slot.
        """
        assert self._native is not None
        assert self.rate_slot is not None
        self._native.table_rate_at(float(z), before)
        row = np.zeros(self.n_phi)
        row[self.rate_slot] = 1.0
        return row

    def values(self, kind: Literal["phi", "prefactor"], z: float) -> ComplexArray:
        """Return a table's value at `z` on the host, from its native slot."""
        assert self._native is not None
        slot = self.load([z])[float(z)]
        return self._native.table_terms_to_host(kind, slot)


class ControllerLike(Protocol):
    """The step controller's norm scale, as the resident engine reads it."""

    def scale(self, before: float, after: float) -> float | RealArray:
        """Return the norm's scale for two state magnitudes."""
        ...


def _refuse_adams(problem: KernelProblem) -> None:
    integrator = problem.stepper.integrator
    # The low-storage march (solprop SEP 24 phase 3) runs on every route the
    # translation admits it on: no coupled flow and no TDSE response.
    if integrator is None or isinstance(integrator, LowStorageRungeKutta):
        return
    # solprop SEP 16 phase 3: every route's stage source serves both formulations,
    # with the modified noise. A coupled flow is stepped by the exponential
    # formulation only (phase 5), as solprop's reference does; its CF4 sections are
    # refused by the shared check.
    if (
        problem.linear.coupled_flow is not None
        and integrator.formulation != "exponential"
    ):
        raise BackendError(
            "solprop-native does not step a problem with a coupled linear flow with "
            f"the {integrator.name} integrator; use Multistep(formulation="
            "'exponential') or the Runge-Kutta march"
        )
    refuse_formulation(problem, "native")
    # SEP 35 phase 3: the scalar shift on CPU builds; device builds wait for their
    # qualification.
    device = _on_device()
    refuse_carrier_frame(
        problem,
        "native (device build)" if device else "native",
        delivered=frozenset() if device else frozenset({"scalar"}),
    )


def _refuse_unsupported(problem: KernelProblem) -> None:
    if problem.schema_version != SCHEMA_VERSION:
        raise BackendError(
            f"solprop-native implements schema {SCHEMA_VERSION}; problem declares "
            f"{problem.schema_version}"
        )
    if problem.precision.single:
        raise BackendError(
            "solprop-native computes in binary64 only and refuses precision='single' "
            "(solprop SEP 33); use backend='numpy_ref', the single-precision oracle"
        )
    if problem.tdse is not None:
        raise BackendError(
            "solprop-native does not execute TDSEGasResponse (solprop SEP 12); "
            "use backend='numpy_ref' with the independent solprop-tdse atomic engine"
        )
    # Schema 16's low-call Adams integrator (solprop SEP 13) runs here on the modal
    # fibre route (SEP 14 phase 2), its source evaluated by the general engine's stage
    # pipeline; elsewhere it is refused by name rather than stepped by this backend's
    # Runge-Kutta march, which would silently be a different method. A 'cf4' section
    # is refused first, by the shared list: no integrator runs one here, so the
    # integrator's advice to take the Runge-Kutta march would mislead.
    flow = problem.linear.coupled_flow
    if flow is not None and "cf4" in flow.section_methods:
        refuse_general_route(
            problem, backend="native", flow_methods=_FLOW_METHODS, moving_maps=True
        )
    _refuse_adams(problem)
    # One list for every backend: the general modal route and the coupled flow, each
    # refusal naming this backend.
    # A moving guide's synthesis and projection, combined per stage over schema 14's
    # `StageScalars.transform_weights`, run here beside the frozen-residual flow that
    # carries their connection.
    refuse_general_route(
        problem, backend="native", flow_methods=_FLOW_METHODS, moving_maps=True
    )
    _refuse_native_route(problem)
    flags = problem.flags
    if not flags.single_batch:
        raise BackendError(
            "NativeBackend owns one member (max_batch = 1, n_batch == 1); "
            "use NativeBatchBackend for fixed-schedule batches"
        )
    if problem.transform.n_pol * problem.shape[2] > 2:
        raise BackendError(
            "the polarisation components must belong to one of the two supported "
            "representations; Kokkos supports at most two effective node "
            "components, but the transform and state request "
            f"{problem.transform.n_pol * problem.shape[2]}"
        )
    if (
        flags.has_raman
        and problem.transform.n_pol * problem.shape[2] != 1
        and (
            not problem.responses.raman_terms
            or not all(term.tensor_capable for term in problem.responses.raman_terms)
        )
    ):
        raise BackendError(
            "vector Raman requires tensor-capable terms; found scalar_only"
        )
    if (
        flags.has_spectral_responses and problem.responses.envelope_plasma is None
    ) or any(
        response is not problem.responses.envelope_plasma
        for response in problem.responses.spectral_responses
    ):
        raise BackendError(
            "spectral responses other than envelope plasma are unsupported by Kokkos"
        )


_FLOW_METHODS: tuple[SectionMethod, ...] = ("constant_schur", "frozen_residual")
"""The section methods of ``linear.coupled_flow`` this backend executes.

The constant Schur section and the frozen-residual varying flow. The commutator-free
CF4 flow is solprop's reference backend's alone, and solprop's shared admission list
refuses it here by name.
"""


def _refuse_native_route(problem: KernelProblem) -> None:
    """Refuse what solprop's shared list admits and this backend has not built.

    The general modal route and the coupled flow are admitted by solprop's one list,
    :func:`solprop.backends.numpy_ref.admission.refuse_general_route`, run under this
    backend's name before anything is allocated. What remains
    here is this backend's own: the native delayed response contracts the six pair
    histories of three Cartesian components, which the reference's arithmetic does
    not require of a record.
    """
    tables = problem.transform.modal_tables
    regional = problem.responses.regional_responses
    if tables is None or regional is None or not regional.n_terms:
        return
    components = int(tables.electric.shape[2])
    if regional.n_terms != DELAYED_TERMS or components != 3:
        raise BackendError(
            f"native executes the {DELAYED_TERMS} delayed channels of the silica "
            "tensor Raman model on three components, and "
            f"responses.regional_responses carries {regional.n_terms} term(s) on "
            f"{components}"
        )


@dataclasses.dataclass(frozen=True)
class _HostFlow:
    """A coupled flow's Schur factors per section on its active bins, host-side.

    Formed as solprop's reference backend forms its own (``_coupled_flow_maps``): the
    frozen block of each section read out of the leaf, decomposed per active bin, the
    declared conditioning row, and a frozen-residual flow's terms; with each
    section's eigenbasis of its triangular factor (solprop's
    ``schur_eigenbasis``), whose route marks the bins formed from it, and the largest
    row-sum norm of the factor, which the squaring refusal reads. The eigenbasis
    crosses to the native engine once, with the Schur factors of the bins it does not
    cover; the engine forms the actions per section and step length. ``scored``
    marks, per active bin, the requested band (``grids.inband_mask``): the
    explicit residual is applied there only, as solprop's reference applies it
    (``residual_bins``).
    """

    active: npt.NDArray[np.bool_]
    scored: npt.NDArray[np.bool_]
    vectors: ComplexArray
    triangular: ComplexArray
    frozen: ComplexArray
    condition_bound: RealArray
    residual_terms: ComplexArray | None
    left: ComplexArray
    right: ComplexArray
    eigenvalues: ComplexArray
    route: npt.NDArray[np.bool_]
    row_sum: RealArray


def _host_residual(
    flow: _HostFlow, section: int, state: ComplexArray, weights: RealArray
) -> ComplexArray:
    """Return ``R_s(z) state`` for the flow weights of a position (solprop SEP 16).

    solprop's reference residual (``numpy_ref.backend._residual_action`` and
    ``_residual_drive``): the terms combined over the weights less the step's section's
    frozen block, applied on the active bins the error norm scores, zero elsewhere.
    """
    assert flow.residual_terms is not None
    action = combine(weights, flow.residual_terms) - flow.frozen[section]
    drive = np.zeros_like(state)
    bins = np.flatnonzero(flow.active)[flow.scored]
    drive[:, :, :, bins] = np.einsum(
        "wij,bjpw->bipw", action[flow.scored], state[:, :, :, bins]
    )
    return drive


def _host_flow(problem: KernelProblem) -> _HostFlow | None:
    """Decompose an admitted coupled flow once, or return ``None`` without one."""
    flow: CoupledFlowTables | None = problem.linear.coupled_flow
    if flow is None:
        return None
    generator = flow.generator
    assert generator is not None
    sections = len(flow.section_methods)
    active = np.asarray(np.any(generator != 0.0, axis=(0, 1, 2, 3)), dtype=np.bool_)
    frozen = np.stack(
        [
            np.moveaxis(frozen_generator(flow, index)[:, :, active], -1, 0)
            for index in range(sections)
        ]
    )
    factors = [schur_blocks(np.moveaxis(block, 0, -1)) for block in frozen]
    bases = [schur_eigenbasis(*pair) for pair in factors]
    residual: ComplexArray | None = None
    if flow.section_methods != ("constant_schur",):
        terms = generator[:, :, :, :, active]
        if flow.connection is not None:
            terms = terms - flow.connection[:, :, :, :, active]
        residual = np.ascontiguousarray(
            np.moveaxis(terms, (2, 3), (-2, -1)).reshape(
                sections * flow.n_deg, -1, flow.n_modes, flow.n_modes
            ),
            dtype=np.complex128,
        )
    return _HostFlow(
        active=active,
        scored=np.ascontiguousarray(problem.grids.inband_mask[active], dtype=np.bool_),
        vectors=np.stack([pair[0] for pair in factors]),
        triangular=np.stack([pair[1] for pair in factors]),
        frozen=np.ascontiguousarray(frozen, dtype=np.complex128),
        condition_bound=np.ascontiguousarray(
            flow.inverse_condition_bound[:, active], dtype=np.float64
        ),
        residual_terms=residual,
        left=np.ascontiguousarray(np.stack([basis[0] for basis in bases])),
        right=np.ascontiguousarray(np.stack([basis[1] for basis in bases])),
        eigenvalues=np.ascontiguousarray(np.stack([basis[2] for basis in bases])),
        route=np.stack([basis[3] for basis in bases]),
        row_sum=np.asarray(
            [
                float(np.max(np.sum(abs(pair[1]), axis=-1), initial=0.0))
                for pair in factors
            ],
            dtype=np.float64,
        ),
    )


_TAYLOR_SCALE = 0.5
"""Row-sum norm solprop's reference scales an exponent below before its series."""

_MAX_FLOW_SQUARINGS = 30
"""Largest number of squarings a flow action may need before it is refused."""


def _flow_squarings(largest: float) -> None:
    """Refuse a length whose exponent would need more squarings than admitted.

    solprop's reference rule (``numpy_ref.backend._flow_squarings``), applied to
    the section's largest row-sum norm times the length whichever route the engine
    forms a bin by, so a step is refused here exactly where the reference refuses it,
    to the rounding of one product.
    """
    squarings = 0
    if largest > _TAYLOR_SCALE:
        squarings = math.ceil(math.log2(largest / _TAYLOR_SCALE))
    if squarings > _MAX_FLOW_SQUARINGS:
        raise BackendError(
            f"native cannot take a coupled flow action whose exponent has row-sum "
            f"norm {largest:.4g}: it would need {squarings} squarings, above the "
            f"{_MAX_FLOW_SQUARINGS} this backend admits. Cap dz_max"
        )


def _check_flow_conditioning(
    flow: _HostFlow,
    section: int,
    measured: RealArray,
    dz: float,
) -> None:
    """Refuse a step whose measured ``norm(U) norm(U^-1)`` exceeds the certificate.

    The certificate row ``exp(dz_max (mu_max - mu_min))`` is checked per step as the
    reference checks it, with the same allowance for the rounding of two norms and
    nothing more; the two spectral norms of the whole step's actions are measured by
    the engine per bin, each an upper bound to the rounding of a Jacobi iteration on
    its Gram matrix.
    """
    if not measured.size:
        return
    declared = flow.condition_bound[section]
    modes = int(flow.vectors.shape[-1])
    allowance = 1.0 + 64.0 * modes * float(np.finfo(float).eps)
    excess = measured / (declared * allowance)
    worst = int(np.argmax(excess))
    if excess[worst] > 1.0:
        raise BackendError(
            f"native measured a coupled flow conditioning of "
            f"{float(measured[worst]):.12g} over a step of {dz:.4g} m, above the "
            f"{float(declared[worst]):.12g} the certificate declares at "
            f"active bin {worst} of section {section} -- exceeding it by a factor "
            f"{float(excess[worst]):.12g} of the rounding allowance: the inverse "
            "action carries the nonlinear source back by the measured factor. Lower "
            "dz_max, which is what the bound was declared over"
        )


def _flow_payload(flow: _HostFlow | None) -> dict[str, object]:
    """Pack a coupled flow's bins and residual data; its actions follow per step."""
    if flow is None:
        return {"coupled_flow": False}
    slot = np.zeros(flow.active.size, dtype=np.uint64)
    slot[flow.active] = np.arange(1, int(np.count_nonzero(flow.active)) + 1)
    residual = flow.residual_terms
    empty = np.empty(0, dtype=np.complex128)
    # The bins formed by scaling and squaring are numbered 1, 2, ... in row-major
    # (section, bin) order and carry their Schur factors; the rest take the
    # eigenbasis, whose `W` and `Y` are zero where it is not used.
    fallback = ~flow.route
    route = np.zeros(flow.route.shape, dtype=np.uint64)
    route[fallback] = np.arange(1, int(np.count_nonzero(fallback)) + 1)
    # The residual's data are zero on the bins the error norm does not score, so the
    # engine's residual there is an exact zero and the frozen flow alone carries them,
    # as in solprop's reference. Every term is finite, so each
    # product with a zero is a signed zero and the running sums stay +0: the source is
    # what adding nothing leaves, but for the sign of a zero.
    scored = flow.scored[None, :, None, None]
    return {
        "coupled_flow": True,
        "flow_active": int(np.count_nonzero(flow.active)),
        "flow_sections": int(flow.frozen.shape[0]),
        "flow_terms": 0 if residual is None else int(residual.shape[0]),
        "flow_slot": slot,
        "flow_frozen": empty
        if residual is None
        else np.ascontiguousarray(np.where(scored, flow.frozen, 0.0)),
        "flow_residual_terms": empty
        if residual is None
        else np.ascontiguousarray(np.where(scored, residual, 0.0)),
        "flow_left": flow.left,
        "flow_right": flow.right,
        "flow_eigenvalues": flow.eigenvalues,
        "flow_route": route.reshape(-1),
        "flow_fallback_vectors": np.ascontiguousarray(flow.vectors[fallback]),
        "flow_fallback_triangular": np.ascontiguousarray(flow.triangular[fallback]),
    }


def _aperture_payload(
    tables: ApertureTables | None, n_omega: int
) -> tuple[RealArray, RealArray, int, int]:
    """Pack far-field tables as the native pass reads them (solprop SEP 23 section 6).

    Returns the ``(n_apertures, 6 + n_omega)`` aperture rows (centre x and y, the
    half-width, the reciprocal edge width, the edge or order, the distance, then the
    scale row ``z c / omega`` that every backend reads to the bit), the
    ``(n_regions, 4)`` boxes, the profile code and the exponent; empty tables without
    any.
    """
    if tables is None:
        return np.empty((0, 6 + n_omega)), np.empty((0, 4)), 0, 2
    constants = np.column_stack(
        (
            tables.centres,
            tables.half_widths,
            tables.inverse_widths,
            tables.edges,
            tables.distances,
        )
    )
    apertures = np.ascontiguousarray(np.hstack((constants, tables.scale)))
    return (
        apertures,
        np.ascontiguousarray(tables.regions, dtype=np.float64),
        tables.profile_code,
        tables.exponent,
    )


def _native_problem(
    problem: KernelProblem,
    fixed: StageTerms,
    flow: _HostFlow | None = None,
    tables: _ZTables | None = None,
) -> dict[str, object]:
    kerr = problem.responses.kerr_coefficients
    dispersive = problem.responses.dispersive_kerr
    # solprop SEP 6's number-conserving sector is formed on analytic signals, of the
    # filtered fields and, for the polynomial, of the bare one; a Raman drive beside it
    # is the cycle-averaged one, which reads the analytic signal too.
    # The general modal route synthesises its own analytic signal on the real family.
    family = (
        "envelope"
        if problem.grid_kind == "envelope"
        else "field_analytic"
        if (kerr is not None and kerr.analytic_signal)
        or (
            dispersive is not None
            and dispersive.sector == "conserving"
            and problem.transform.modal_tables is None
        )
        else "field_real"
    )
    absorber = problem.responses.temporal_absorber
    absorber_fine = problem.responses.temporal_absorber_fine
    spatial_absorber = radial_absorber_matrix(
        problem.transform, problem.grids.pad_scale
    )
    temporal_in_stage = (
        absorber_fine is not None and problem.responses.absorber_scheme == "nonlinear"
    )
    carrier = problem.grids.carrier_2w0
    n_omega = problem.shape[-1]
    phi_terms = fixed.phi
    prefactor_terms = fixed.prefactor
    spatial = fixed.separable
    if spatial is not None:
        phi = np.empty((1, 0, n_omega), dtype=np.complex128)
        prefactor = np.empty((1, 0, n_omega), dtype=np.complex128)
    else:
        # A z table's terms are slots the native evaluation fills (solprop SEP 29 G1):
        # an empty array whose leading extent is their count crosses in their place.
        if tables is not None and tables.phi:
            phi = np.empty((tables.n_phi, 0), dtype=np.complex128)
        elif phi_terms is None:
            raise BackendError("Kokkos requires a device-ready fixed-term problem")
        else:
            phi = np.ascontiguousarray(
                phi_terms.reshape(phi_terms.shape[0], -1, n_omega)
            )
        if tables is not None and tables.prefactor:
            prefactor = np.empty((tables.slots, 0), dtype=np.complex128)
        elif prefactor_terms is None:
            raise BackendError("Kokkos requires a device-ready fixed-term problem")
        else:
            prefactor = np.ascontiguousarray(
                prefactor_terms.reshape(prefactor_terms.shape[0], -1, n_omega)
            )
    channels = problem.responses.population_channels
    raman_terms = problem.responses.raman_terms
    plasma = channels[0] if channels else None
    raman = raman_terms[0] if raman_terms else None
    rate = None if plasma is None else plasma.rate
    coarse = (
        np.empty(0, dtype=np.float64)
        if problem.transform.n_nodes == 1
        or np.array_equal(
            problem.transform.projection_coarse, problem.transform.projection
        )
        else problem.transform.projection_coarse
    )
    geometry = problem.transform.geometry
    # Free space and schema 15's hybrid planar kind share the separable operator and
    # the transverse FFT engine; the planar kind adds its slab rule in x.
    free = geometry is not None and geometry.kind in ("free", "slab")
    spatial_absorption = bool(spatial_absorber.size) or (
        free and geometry is not None and bool(np.any(geometry.absorber_rate))
    )
    node_chunk = min(problem.transform.node_chunk, problem.transform.n_nodes)
    coarse_matrix = coarse
    spatial_data: dict[str, object] = {}
    if free and geometry is not None and spatial is not None:
        spatial_data = {
            "n_x": geometry.shape[0],
            "n_y": geometry.shape[1],
            "spatial_k_squared": spatial.k_squared,
            "spatial_k_squared_low": spatial.k_squared_low,
            "spatial_kperp_sq": spatial.kperp_sq,
            "spatial_kperp_sq_low": spatial.kperp_sq_low,
            "spatial_frame": spatial.frame,
            "spatial_spectral_rate": spatial.spectral_rate,
            "spatial_numerator": spatial.prefactor_numerator,
            "spatial_clamped": spatial.prefactor_clamped,
            "spatial_evanescent_rate": spatial.evanescent_rate,
            "spatial_absorber_rate": geometry.absorber_rate,
            # A transverse collar's rate per column (solprop kernel schema 31).
            "spatial_kperp_rate": spatial.kperp_rate,
        }
        if geometry.kind == "free":
            # solprop SEP 23: the stored axes and the slots of the problem's own
            # far-field request, which size the native band partials. The pass folds
            # on_axis_spectrum only: on_axis_intensity is spatial_statistics' own.
            k_x, k_y = geometry.spectral_axes
            slots, _ = far_field_partials(problem, axis_folded=AXIS_SPECTRUM_ONLY)
            spatial_data |= {
                "spatial_k_x": np.ascontiguousarray(k_x, dtype=np.float64),
                "spatial_k_y": np.ascontiguousarray(k_y, dtype=np.float64),
                "spatial_far_field_slots": slots,
            }
        if geometry.kind == "slab":
            spatial_data |= _slab_payload(problem)
            n_x, n_y = geometry.shape
            # Whole y rows per chunk, as many as the node chunk holds (at least one),
            # which is solprop's reference's own blocking.
            node_chunk = min(n_y, max(1, problem.transform.node_chunk // n_x)) * n_x
            # The reference scores no quadrature alarm on this kind, so no embedded
            # Kronrod projection is packed.
            coarse_matrix = np.empty(0, dtype=np.float64)
    # The accepted step's last stage leaves the plasma statistics behind only when
    # that stage's input is the committed field, as on numba_cpu: a first-same-as-last
    # tableau, no split-step absorber (applied to the field after the last stage)
    # and no coupled flow; and not under the modified noise model, where that input
    # carries the noise draw while solprop defines the statistics on the committed
    # field its reference scans. Otherwise they are scanned afresh.
    stash_plasma = (
        plasma is not None
        and problem.responses.envelope_plasma is None
        and problem.initial.noise_model != "modified"
        and problem.stepper.tableau.first_same_as_last
        and not (
            absorber is not None and problem.responses.absorber_scheme == "split_step"
        )
        and problem.linear.coupled_flow is None
        and bool(
            {
                "electron_density",
                "peak_ionisation_rate",
                "species_ionisation",
                "species_electron_density",
            }
            & set(problem.stats_spec.periodic)
        )
    )
    return {
        **spatial_data,
        **_channel_payload(problem),
        **_envelope_payload(problem),
        **_general_payload(problem),
        **_flow_payload(flow),
        **_dispersive_payload(problem),
        **_flux_payload(problem),
        "n_omega": n_omega,
        "n_t": problem.grids.n_t,
        "n_to": problem.grids.n_to,
        "n_stages": problem.stepper.tableau.n_stages,
        "n_modes": problem.shape[1],
        "n_pol_state": problem.shape[2],
        "n_pol_transform": problem.transform.n_pol,
        "n_nodes": problem.transform.n_nodes,
        "node_chunk": node_chunk,
        "family": family,
        "nonlinear": kerr is not None,
        "modified_noise": problem.initial.noise_model == "modified",
        "in_stage_absorber": temporal_in_stage or spatial_absorption,
        "split_step_absorber": absorber is not None
        and problem.responses.absorber_scheme == "split_step",
        "first_same_as_last": problem.stepper.tableau.first_same_as_last,
        "crop_index": np.ascontiguousarray(problem.grids.crop_index, dtype=np.uint64),
        "active_mask": np.ascontiguousarray(problem.grids.active_mask),
        "norm_mask": np.ascontiguousarray(problem.grids.norm_mask),
        "omega_window": np.ascontiguousarray(problem.grids.omega_window),
        "omega": np.ascontiguousarray(problem.grids.omega),
        "time_window": np.ascontiguousarray(problem.grids.t_window_fine),
        "time_window_derivative": np.ascontiguousarray(
            problem.grids.t_window_fine_derivative
            if problem.grid_kind == "field" and problem.responses.plasma_channels
            else np.empty(0)
        ),
        "absorber_fine": np.ascontiguousarray(
            absorber_fine
            if temporal_in_stage
            else np.zeros(problem.grids.n_to)
            if spatial_absorption
            else np.empty(0),
            dtype=np.float64,
        ),
        "spatial_absorber_matrix": spatial_absorber,
        "spatial_spectral_weights": np.empty(0)
        if geometry is None
        else geometry.spectral_weights,
        "spatial_on_axis": np.empty(0, dtype=np.complex128)
        if geometry is None
        else geometry.on_axis,
        "absorber_coarse": np.ascontiguousarray(
            np.empty(0) if absorber is None else absorber, dtype=np.float64
        ),
        "carrier": np.ascontiguousarray(
            np.ones(problem.grids.n_to, dtype=np.complex128)
            if carrier is None
            else carrier
        ),
        "synthesis_matrix": np.ascontiguousarray(problem.transform.synthesis),
        "projection_matrix": np.ascontiguousarray(problem.transform.projection),
        "projection_coarse_matrix": np.ascontiguousarray(coarse_matrix),
        "phi_terms": phi,
        "prefactor_terms": prefactor,
        "phi_table": tables is not None and tables.phi,
        "prefactor_table": tables is not None and tables.prefactor,
        "power_weight": np.ascontiguousarray(
            np.empty(0) if geometry is not None else power_weight_at(problem, 0.0)
        ),
        "tableau_matrix": np.ascontiguousarray(problem.stepper.tableau.matrix),
        "tableau_nodes": np.ascontiguousarray(problem.stepper.tableau.nodes),
        "solution_weights": np.ascontiguousarray(
            problem.stepper.tableau.solution_weights
        ),
        "error_weights": np.ascontiguousarray(problem.stepper.tableau.error_weights),
        "synthesis": 0.0 if free else float(problem.transform.synthesis[0, 0]),
        "projection": 0.0 if free else float(problem.transform.projection[0, 0]),
        "kerr_coefficient": 0.0 if kerr is None else kerr.coefficient,
        "self_factor": 0.0 if kerr is None else kerr.self_factor,
        "conjugate_factor": 0.0 if kerr is None else kerr.conjugate_factor,
        "harmonic_factor": 0.0 if kerr is None else kerr.harmonic_factor,
        "cubic_real": kerr is not None and kerr.cubic_form == "real",
        "take_real": kerr is not None and kerr.take_real,
        "plasma": plasma is not None,
        "raman": raman is not None,
        "raman_thg": raman is not None
        and bool(raman.thg)
        and problem.grid_kind == "field",
        "raman_tensor_capable": bool(raman_terms)
        and all(term.tensor_capable for term in raman_terms),
        "raman_tensor_weights": np.ascontiguousarray(
            np.concatenate([term.tensor_weights for term in raman_terms], axis=0)
            if raman_terms
            else np.empty((0, 2), dtype=np.float64)
        ),
        "stash_plasma": stash_plasma,
        "weighted_controller": problem.stepper.controller.weights is not None,
        "phase_b_engine": True,
        "dt_fine": problem.grids.dt_fine,
        "dt": problem.grids.dt,
        "raman_fraction": 0.0 if raman is None else float(raman.fraction),
        "raman_epsilon0": EPS0 if raman is not None else 0.0,
        "raman_undamped": np.ascontiguousarray(
            np.empty((0, 2 * problem.grids.n_to), dtype=np.float64)
            if raman is None
            else np.concatenate([term.undamped for term in raman_terms], axis=0)
        ),
        "charge_mass_ratio": 0.0 if plasma is None else plasma.charge_mass_ratio,
        "ionisation_potential": 0.0 if plasma is None else plasma.ionisation_potential,
        "log_field_min": 0.0 if rate is None else rate.log_field_min,
        "d_log_field": 0.0 if rate is None else rate.d_log_field,
        "field_min": 0.0 if rate is None else rate.field_min,
        "field_max": 0.0 if rate is None else rate.field_max,
        "rate_max": 0.0 if rate is None else rate.rate_max,
        "rate_coefficients": np.ascontiguousarray(
            np.empty((4, 0), dtype=np.float64) if rate is None else rate.coefficients
        ),
        # solprop SEP 18: the running-integral rule, one per problem.
        "plasma_band": problem.responses.plasma_integral.policy == "band",
        "plasma_taps": problem.responses.plasma_integral.taps,
    }


def _dispersive_payload(problem: KernelProblem) -> dict[str, object]:
    """Lay out solprop SEP 6's dispersive response for the native engine.

    The layout of solprop's ``numba_cpu`` (``dispersive_terms_layout``), whose
    structure the native response takes on every route: the channels grouped as the
    leaf groups them (``DispersiveKerrTerms.groups``) -- the symmetric evaluator's one
    mixture per node region, the polynomial's species of equal leg weight and region --
    with one filtered field and one product per symmetric group, and for the
    polynomial one ``E^3`` per distinct region followed by two fields and two products
    per group; each output term's multiplier, product and factor, every multiplier
    carrying the sector's ``Pi_+`` and factor (one on a field grid's full cubic,
    ``Pi_+ / 8`` for its conserving sector, ``Pi_+ / 4`` on an envelope grid), and the
    input multipliers ``Pi_+`` on a field grid's analytic family. A mixture's rows are
    rebuilt natively at every stage whose densities change, from its channels' ``g^4``
    (``mixture_filter``'s arithmetic); the first species' stand until then. On the
    general modal route each group's key is its region and every node's region is
    packed beside them; the route's grid is a field grid.
    """
    leaf = problem.responses.dispersive_kerr
    if leaf is None:
        return {"dispersive": False}
    tables = problem.transform.modal_tables
    general = tables is not None
    kind = "field" if general else problem.grid_kind
    form = (
        ("real" if leaf.sector == "full" else "conserving")
        if kind == "field"
        else ("harmonic" if leaf.sector == "full" else "conserving")
    )
    analytic = form == "conserving" and kind == "field"
    symmetric = leaf.evaluator == "symmetric"
    n_channels = leaf.n_channels
    positive = leaf.positive.astype(np.float64)
    if kind == "envelope":
        scale = positive / 4.0
    elif analytic:
        scale = positive / 8.0
    else:
        scale = np.ones(positive.shape)
    input_scale = positive if analytic else np.ones(positive.shape)
    groups = leaf.groups()
    # A group's key is its general modal region; every other route has one, so one
    # cube is shared by every polynomial group there.
    regions = leaf.regions if general else None
    keys = [0 if regions is None else int(regions[group[0]]) for group in groups]
    distinct = list(dict.fromkeys(keys))
    cubes = [distinct.index(key) for key in keys]
    n_cubes = 0 if symmetric else len(distinct)
    mixtures = tuple(
        index for index, group in enumerate(groups) if symmetric and len(group) > 1
    )
    fields: list[RealArray] = []
    rows: list[int] = []
    channels: list[int] = []
    multipliers: list[RealArray] = []
    for index, group in enumerate(groups):
        weight = leaf.filters[group[0]]
        if symmetric:
            fields.append(weight)
            rows.append(index)
            channels.append(
                n_channels + 1 + mixtures.index(index)
                if index in mixtures
                else group[0]
            )
            multipliers.append(weight * scale)
            continue
        assert leaf.outputs is not None
        fields.extend([weight, weight * weight])
        first = n_cubes + 2 * index
        for channel in group:
            output = leaf.outputs[channel]
            rows.extend([cubes[index], first])
            channels.extend([channel, channel])
            multipliers.extend([output[0] * scale, output[1] * scale])
        rows.append(first + 1)
        channels.append(n_channels)
        multipliers.append(scale.copy())
    stacked = np.array(fields, dtype=np.float64)
    if analytic:
        stacked = stacked * leaf.positive
    constants = (
        np.zeros((n_channels, 2))
        if leaf.products is None
        else np.asarray(leaf.products, dtype=np.float64)
    )
    if mixtures:
        squares = leaf.filters * leaf.filters
        quartics = np.ascontiguousarray(squares * squares, dtype=np.float64)
    else:
        quartics = np.empty(0, dtype=np.float64)
    offsets = np.cumsum([0, *(len(group) for group in groups)])
    route: dict[str, object] = {}
    if tables is not None:
        route = {
            "dispersive_group_keys": np.asarray(keys, dtype=np.uint64),
            "dispersive_node_region": np.ascontiguousarray(
                tables.region_index, dtype=np.uint64
            ),
        }
    return {
        **route,
        "dispersive": True,
        "dispersive_symmetric": symmetric,
        "dispersive_form": {"real": 0, "conserving": 1, "harmonic": 2}[form],
        "dispersive_inputs": np.ascontiguousarray(stacked),
        "dispersive_term_multipliers": np.ascontiguousarray(
            np.array(multipliers, dtype=np.float64)
        ),
        "dispersive_term_rows": np.asarray(rows, dtype=np.uint64),
        "dispersive_term_channels": np.asarray(channels, dtype=np.uint64),
        "dispersive_coefficients": np.asarray(leaf.coefficients, dtype=np.float64),
        "dispersive_pair_constants": np.ascontiguousarray(constants),
        "dispersive_group_offsets": np.asarray(offsets, dtype=np.uint64),
        "dispersive_group_members": np.asarray(
            [channel for group in groups for channel in group], dtype=np.uint64
        ),
        "dispersive_quartics": quartics,
        "dispersive_input_scale": np.ascontiguousarray(input_scale),
        "dispersive_output_scale": np.ascontiguousarray(scale),
    }


def _flux_payload(problem: KernelProblem) -> dict[str, object]:
    """Pack the plane measure of solprop SEP 34 (kernel schema 35).

    A separable run (Cartesian, line, planar) is measured by its own operator's root,
    which the native state already holds: nothing more is packed, and a plan that is
    not that operator is refused. A radial run packs the entrance's material and
    transverse squares and whether a material z table follows
    (:func:`_upload_flux_table`); the admission record is not read. ``flux_closure``
    sizes the magnetic closure's buffers for the problem's own local statistics.
    """
    flux = problem.flux
    if flux is None:
        return {}
    closure = bool(set(problem.stats_spec.names) & LOCAL_FLUX_STATISTICS)
    if problem.linear.separable is not None:
        if flux.base is not problem.linear.separable or flux.material is not None:
            raise BackendError(
                "solprop-native measures a separable run by its own operator's root, "
                "and this plane measure is another"
            )
        return {"flux_closure": closure}
    base = flux.base
    return {
        "flux_k_squared": np.ascontiguousarray(base.k_squared, dtype=np.complex128),
        "flux_k_squared_low": np.ascontiguousarray(
            base.k_squared_low, dtype=np.complex128
        ),
        "flux_kperp_sq": np.ascontiguousarray(base.kperp_sq, dtype=np.float64),
        "flux_kperp_sq_low": np.ascontiguousarray(base.kperp_sq_low, dtype=np.float64),
        "flux_table": flux.material is not None,
        "flux_closure": closure,
    }


def _upload_flux_table(native: _core.State, problem: KernelProblem) -> None:
    """Make a radial plane measure's material z table resident, once (SEP 34)."""
    flux = problem.flux
    if flux is None or flux.material is None:
        return
    table = flux.material
    low, high = table.z_span
    slack = _EDGE_SLACK_FRACTION * (high - low)
    n_intervals, n_terms = table.coefficients.shape[:2]
    native.table_upload(
        "flux",
        np.ascontiguousarray(table.breakpoints, dtype=np.float64),
        low - slack,
        high + slack,
        np.ascontiguousarray(table.coefficients, dtype=np.complex128).reshape(
            n_intervals, n_terms, -1
        ),
    )


def _slab_payload(problem: KernelProblem) -> dict[str, object]:
    """Pack the hybrid planar kind's channels, wall terms and per-column collar rate.

    Schema 15's ``SeparableOperator.kappa_sq`` is ``(n_channels, n_omega)``, column
    ``c`` belonging to channel ``c % n_channels``; the geometry's collar rate depends on
    y alone (solprop checks it), so it is packed per column ``p + n_p * iy``, the index
    of the temporal row it acts on.
    """
    geometry = problem.transform.geometry
    separable = problem.linear.separable
    assert geometry is not None
    assert separable is not None
    assert separable.kappa_sq is not None
    n_x, n_y = geometry.shape
    n_p = separable.n_channels
    return {
        "n_p": n_p,
        "spatial_kappa_sq": np.ascontiguousarray(separable.kappa_sq),
        "spatial_absorber_rate": np.ascontiguousarray(
            np.repeat(geometry.absorber_rate.reshape(n_y, n_x)[:, 0], n_p)
        ),
    }


def _general_payload(problem: KernelProblem) -> dict[str, object]:
    """Pack the general modal route's maps, derived as ``numpy_ref`` derives them.

    The per-node tensor, the form and the delayed maps come from solprop's own
    public derivations (``solprop.backends.numpy_ref.general_modal``), so this
    backend executes exactly the arrays the reference executes; the chunk is the
    reference's own memory strategy at the same budget. Nothing is packed for a
    problem without ``transform.modal_tables``.
    """
    tables = problem.transform.modal_tables
    if tables is None:
        return {"general_modal": False}
    assert tables.synthesis is not None
    assert tables.projection is not None
    regional = problem.responses.regional_responses
    components = int(tables.electric.shape[2])
    nodes = int(tables.region_index.size)
    # The reference's chunk, which counts SEP 6's dispersive buffers as both
    # compiled backends hold them.
    chunk = general_node_chunk(
        components,
        problem.grids.n_to,
        int(tables.omega_fine.size),
        DEFAULT_NODE_BUDGET,
        dispersive=general_dispersive_buffers(problem.responses.dispersive_kerr),
    )
    owner, delayed_tensor, delayed_spectrum = node_delayed_maps(tables, regional)
    terms = 0 if delayed_tensor is None else int(delayed_tensor.shape[1])
    intervals, n_deg = (int(one) for one in tables.synthesis.shape[:2])
    return {
        "general_modal": True,
        "general_real_form": node_kerr_form(regional) == "real",
        # A moving guide's maps are every Chebyshev coefficient along z, flattened
        # over (n_intervals, n_deg) in the order of the stage bundle's
        # transform_weights; a constant guide's are the one coefficient.
        "general_map_intervals": intervals,
        "general_map_n_deg": n_deg,
        "general_components": components,
        "general_nodes": nodes,
        "general_node_chunk": min(chunk, nodes),
        "general_responses": 0 if delayed_tensor is None else delayed_tensor.shape[0],
        "general_terms": terms,
        "general_pad_scale": float(problem.grids.pad_scale),
        "general_epsilon0": EPS0,
        "general_synthesis": np.ascontiguousarray(tables.synthesis),
        "general_projection": np.ascontiguousarray(tables.projection),
        "general_tensor": np.ascontiguousarray(node_kerr_tensor(tables, regional)),
        "general_frequency_factor": np.ascontiguousarray(
            tables.source_frequency_factor, dtype=np.complex128
        ),
        "general_delayed_tensor": np.ascontiguousarray(
            np.empty(0) if delayed_tensor is None else delayed_tensor
        ),
        "general_delayed_spectrum": np.ascontiguousarray(
            np.empty(0, dtype=np.complex128)
            if delayed_spectrum is None
            else delayed_spectrum
        ),
        "general_node_response": np.ascontiguousarray(
            np.empty(0, dtype=np.uint64) if not terms else owner.astype(np.int64) + 1,
            dtype=np.uint64,
        ),
    }


def _envelope_payload(problem: KernelProblem) -> dict[str, object]:
    """Pack the accepted cycle rules and already masked spectral factors."""
    assembly = problem.responses.envelope_plasma
    if assembly is None:
        return {"envelope_plasma": False}
    nodes = np.zeros((len(assembly.channels), 4, 512))
    weights = np.zeros_like(nodes)
    for channel_index, channel in enumerate(assembly.channels):
        for level, rule in enumerate(channel.cycle.rules):
            nodes[channel_index, level, : len(rule.nodes)] = rule.nodes
            weights[channel_index, level, : len(rule.weights)] = rule.weights
    return {
        "envelope_plasma": True,
        "envelope_rule_nodes": nodes,
        "envelope_rule_weights": weights,
        "envelope_drude_factor": assembly.factors.drude,
        "envelope_loss_factor": assembly.factors.loss,
    }


def _channel_payload(problem: KernelProblem) -> dict[str, object]:
    """Pack independent rate tables without resampling their coefficients."""
    parameters: list[list[float]] = []
    coefficients: list[np.ndarray] = []
    offset = 0
    for channel in problem.responses.population_channels:
        rate = channel.rate
        parameters.append(
            [
                rate.log_field_min,
                rate.d_log_field,
                rate.field_min,
                rate.field_max,
                rate.rate_max,
                rate.coefficients.shape[1],
                offset,
                channel.charge_mass_ratio,
                channel.ionisation_potential,
            ]
        )
        coefficients.append(rate.coefficients.ravel())
        offset += rate.coefficients.size
    terms = problem.responses.raman_terms
    return {
        "n_channels": len(parameters),
        "channel_parameters": np.asarray(parameters, dtype=np.float64).reshape(-1, 9),
        "channel_coefficients": np.concatenate(coefficients)
        if coefficients
        else np.empty(0),
        "n_raman_components": len(terms),
        "raman_row_component": np.asarray(
            [component for component, term in enumerate(terms) for _ in term.undamped],
            dtype=np.float64,
        ),
        "raman_component_fraction": np.asarray([term.fraction for term in terms]),
    }
