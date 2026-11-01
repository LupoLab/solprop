# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The reference backend: the step loop, the seam and the refusals.

One :class:`NumpyRefBackend` steps one mode-averaged problem with the Kerr term, in
plain NumPy, calling the named passes of :mod:`solprop.backends.numpy_ref.passes` in the
order of ``docs/dev/architecture.md`` section 8. The step control follows
``docs/dev/stepper.md`` section 3 exactly: the controller's *proposal* is its state, the
executed step's own norm decides acceptance, and the equivalent norm of the proposal is
what the history and a rejection see.

The contract is consumed only through its evaluation functions --
:func:`solprop.translate.linop.delta_phi`,
:func:`solprop.translate.linop.unitary_phase`,
:func:`solprop.translate.responses.prefactor_at`, and the scalar tables
``responses.density`` and ``transform.taper_scale`` -- never through the coefficient
arrays, so the representation of the z tables can change without touching a backend.
"""

import collections
import dataclasses
import functools
import logging
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from jaxtyping import Bool, Complex, Float, Inexact, Num

from solprop._helpers import PLAIN
from solprop.backends.driver import StepDriver, refuse_general_leaves
from solprop.backends.exponential import refuse_carrier_frame, refuse_formulation
from solprop.backends.low_storage import LowStorageMarch
from solprop.backends.memory import adams_memory, reference_base_memory
from solprop.backends.numpy_ref import passes, transforms
from solprop.backends.numpy_ref.admission import (
    refuse_general_route,
)
from solprop.backends.numpy_ref.envelope_plasma import (
    envelope_populations,
    envelope_sources,
)
from solprop.backends.numpy_ref.general_modal import (
    GeneralMaps,
    general_dispersive_buffers,
    general_memory,
    general_node_chunk,
    general_source,
    node_delayed_maps,
    node_kerr_form,
    node_kerr_tensor,
)
from solprop.backends.numpy_ref.tdse import (
    atomic_edge_source,
    atomic_sources,
    project_atomic_sources,
)
from solprop.backends.populations import axis_refusal, population_statistics
from solprop.backends.spatial import (
    FarFieldTables,
    collect_spatial,
    radial_absorber_matrix,
    slab_columns,
    slab_rows,
    slab_synthesis,
)
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError, ResponseBudgetError
from solprop.fourier import GridKind
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.noise import NoiseRealisation
from solprop.output.capability import (
    declared_capability,
    refusal_for,
    withheld_accessors,
)
from solprop.output.schema import SavedPlane
from solprop.profiles import ZProfile
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
from solprop.translate.coupled import (
    CF4_NODES,
    CF4_WEIGHTS,
    CoupledFlowTables,
    flow_weights,
    frozen_generator,
    metric_at,
    schur_blocks,
)
from solprop.translate.linop import power_weight_at
from solprop.translate.modal import DEFAULT_NODE_BUDGET
from solprop.translate.plasma_source import _BENCHMARK_GATED
from solprop.translate.problem import (
    KERNEL_SCHEMA_VERSION,
    MAX_N_POL,
    KernelProblem,
)
from solprop.translate.spatial import separable_values
from solprop.translate.stage import StageScalars, combine
from solprop.translate.tdse import AtomicCallLedger, TDSEBinding, TDSEPurpose
from solprop.translate.tdse_source import (
    STEADY_CURRENT_CERTIFICATE_LIMIT,
    TDSE_UNAVAILABLE_STATISTICS,
    steady_current_certificate,
)

__all__ = ["QUADRATURE_ALARM", "BlockResidual", "NumpyRefBackend", "SourceTables"]

logger = logging.getLogger(__name__)

QUADRATURE_ALARM = 1e-6
"""Relative quadrature error above which the alarm is logged, once per plane.

The fixed rule integrates the Kerr quartic of a mode set to the reference's own
rounding -- 2e-14 at ``n_r = 32`` for four HE_1m modes
(``docs/guide/waveguides/mode-sets.md``) -- so
anything a millionth of the polarisation is not the rule converging slowly but the
integrand being unresolved, which no step size can fix. It also sits at the loose end
of the tolerances a run asks the controller for, so an alarm means the quadrature has
become the largest error in the step. The number is a judgement, not a measurement.
"""


class NumpyRefBackend(StepDriver[Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]]):
    """Propagate one Kerr and plasma problem with interaction-picture DOPRI5.

    The oracle of ``docs/dev/backends.md`` section 2. It implements
    :class:`solprop.backends.protocol.Backend` and steps a single problem
    (``max_batch = 1``) on either grid: mode-averaged or multimode on the fixed
    transverse quadrature, scalar or vector-isotropic, with the Kerr term, the delayed
    Raman term and the plasma. Mode-averaged propagation is the one-node case of the
    same pipeline and not a second code path.

    **Refusals.** A problem is refused by name, with
    :class:`solprop.exceptions.BackendError`, when its ``schema_version`` is not the one
    this backend knows, when its flags declare a spectral response, when a Raman kernel
    meets two polarisation components, when the transform and the state would both carry
    the transverse components, or when its batch axis is larger than one. A term
    silently dropped would run, converge and be wrong.

    **Convention.** Every array held or returned is in the FFT convention of
    ``docs/dev/conventions.md`` section 2; the backend never conjugates. The
    output layer converts on the way to disk and the processing accessors on the way to
    a user.

    **The grid.** :meth:`stats` delegates to :func:`solprop.stats.collect`, which needs
    a :class:`solprop.grid.TimeGrid`; the kernel contract carries the array group
    :class:`solprop.translate.masks.Grids` and no grid object, so the grid is supplied
    here instead. This is a known gap in the contract.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The validated kernel problem. :meth:`~.KernelProblem.validate` is called again
        here: validation is the backend's defence, not the caller's courtesy.
    grid : solprop.grid.TimeGrid, optional
        The grid the state lives on, needed only by :meth:`stats`.
    modes : collections.abc.Sequence, optional
        The propagated modes, in state order, needed only by the ``zdw`` statistic.
    profiles : collections.abc.Mapping, optional
        The z-profiles keyed by statistic name, needed only by the profile statistics.
    tdse_binding : solprop.translate.tdse.TDSEBinding or None, optional
        Driver-owned persistent atomic plans, required exactly when the problem
        declares a TDSE response. The driver closes this binding after the backend.

    Raises
    ------
    BackendError
        Naming what the backend refuses.

    Examples
    --------
    >>> from solprop.backends.numpy_ref import NumpyRefBackend
    >>> from solprop.grid import EnvelopeGrid
    >>> from solprop.nonlinear.gnlse import problem_from
    >>> from solprop.pulses import SechPulse
    >>> from solprop.stepper import PIController
    >>> from solprop.translate.problem import assemble
    >>> grid = EnvelopeGrid.from_limits(1.0, 835e-9, (500e-9, 2e-6), 1e-12)
    >>> gnlse = problem_from([0.0, 0.0, -1.183e-26], omega_ref=grid.omega0,
    ...                      gamma=0.11, shock=False)
    >>> start = SechPulse(wavelength=835e-9, width=100e-15, peak_power=9.8).build(grid)
    >>> problem = assemble(grid, gnlse.mode, [gnlse.kerr], start,
    ...                    controller=PIController(rtol=1e-8, atol=1e-16),
    ...                    save_planes=[0.01], dz_max=0.01, dz_init=1e-3,
    ...                    density=1.0, shock=False, prefactor_index=1.0)
    >>> backend = NumpyRefBackend(problem, grid=grid)
    >>> backend.set_initial(problem.initial.field0, None)
    >>> summary = backend.propagate_to(0.01)
    >>> summary.z, summary.n_rejected, summary.met_tolerance
    (0.01, 0, True)
    >>> backend.observe().field.shape
    (1, 1, 1, 768)
    >>> backend.close()
    """

    reads_transverse_edge = True

    name = "numpy_ref"
    max_batch = 1
    # The oracle evaluates its stage tables and Adams bases with the uncompiled forms,
    # never with the compiled code it exists to check (SEP 29 section 6.6).
    helpers = PLAIN
    # The reference forms `exp(+dPhi)` and `exp(-dPhi)` afresh at every stage, so
    # keeping the six distinct exponents of a step is worth its memory here where it is
    # not on a backend that takes both signs from one transcendental.
    propagator_default = "tabulate"

    def __init__(
        self,
        problem: KernelProblem,
        *,
        grid: TimeGrid | None = None,
        modes: Sequence[Mode] | None = None,
        profiles: Mapping[str, ZProfile] | None = None,
        tdse_binding: TDSEBinding | None = None,
    ) -> None:
        """Adopt a validated problem, refuse what this backend does not execute."""
        _refuse_unsupported(problem)
        problem.validate()
        if (problem.tdse is None) != (tdse_binding is None):
            raise BackendError(
                "TDSE problems require an explicit host response binding"
            )
        self._tdse_binding = tdse_binding
        self._tdse_last_purpose: TDSEPurpose | None = None
        self._tdse_last_z: float | None = None
        self._tdse_last_projection: dict[str, object] | None = None
        super().__init__(problem, grid=grid, modes=modes, profiles=profiles)
        self._statistics_power_weight = np.empty(
            problem.shape[1:] if problem.transform.geometry is None else (0,)
        )
        # Formed once, so a geometry the far-field statistics refuse is refused here,
        # before any step (SEP 23 section 6).
        self._far_field = FarFieldTables(problem)
        self._gated_plasma = _BENCHMARK_GATED.get()
        self._last_plasma: (
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None
        ) = None
        self._kind: GridKind = problem.grid_kind
        # The tables a source evaluation reads, as the problem holds them; the
        # single-precision path narrows them on upload (SEP 33).
        self._tables = SourceTables.of(problem)
        self._carrier = _harmonic_carrier(problem)
        self._embedded = _embedded_projection(problem)
        self._general = _general_maps(problem)
        self._general_responds = self._general is not None and (
            bool(np.any(self._general.tensor)) or self._general.dispersive is not None
        )
        self._flow = _coupled_flow_maps(problem)
        self._block_residual: BlockResidual | None = None
        self._flow_cache: (
            tuple[
                tuple[int, float, float | None],
                dict[float, tuple[np.ndarray | None, np.ndarray | None]],
            ]
            | None
        ) = None
        self._last_projected: Num[np.ndarray, "n_modes n_pol n_to"] | None = None
        self._last_coarse: Num[np.ndarray, "n_modes n_pol n_to"] | None = None
        self._last_bundle: StageScalars | None = None
        self._last_envelope: (
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None] | None
        ) = None
        self._raman_rates: tuple[float, ...] | None = None
        self._raman_densities: tuple[float, ...] | None = None
        self._raman_kernel: Complex[np.ndarray, "... n_omega_doubled"] | None = None
        self._raman_transforms = 0
        self._alarm = math.nan
        self._alarm_reported = False
        # ADR-025's in-stage scheme is a second term on every stage rather than a
        # factor on the accepted state, so a problem that has no response at all still
        # has a derivative when it is active.
        self._spatial_absorber = radial_absorber_matrix(
            problem.transform, problem.grids.pad_scale
        )
        self._in_stage_absorber = bool(self._spatial_absorber.size) or (
            problem.responses.temporal_absorber_fine is not None
            and problem.responses.absorber_scheme == "nonlinear"
        )
        geometry = problem.transform.geometry
        if geometry is not None and geometry.kind in ("free", "slab"):
            self._in_stage_absorber |= bool(np.any(geometry.absorber_rate))
        integrator = problem.stepper.integrator
        self._adams = None
        if isinstance(integrator, Multistep):
            # The shared march reads this package's passes, so it is imported when a
            # run asks for it rather than with the package.
            from solprop.backends.multistep import AdamsMarch  # noqa: PLC0415

            self._adams = AdamsMarch(self, integrator)
        elif isinstance(integrator, LowStorageRungeKutta):
            # The host engine is the oracle (SEP 24 section 6.5).
            self._low_storage = LowStorageMarch(self, integrator)
        self._adams_held: (
            tuple[float, float, StageScalars, Complex[np.ndarray, ...]] | None
        ) = None
        # The lab-frame source's unit "propagator": one element broadcast to the state's
        # shape, so it holds no volume beside the state.
        self._lab_frame = np.broadcast_to(
            np.ones((), dtype=np.complex128), problem.shape
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "numpy_ref: %s grid, shape %s, %d Phi intervals, %s propagators, "
                "%s schedule",
                problem.grid_kind,
                problem.shape,
                problem.linear.n_intervals,
                self._propagator_mode,
                "replayed" if problem.stepper.schedule is not None else "adaptive",
            )

    def _propagator(
        self, exponent: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return the stage propagator ``exp(dPhi)`` of a binary64 exponent.

        The single-precision path rounds the binary64 exponential once (SEP 33
        section 4.2); here it is the exponential itself.

        Parameters
        ----------
        exponent : numpy.ndarray
            ``dPhi`` or its negative, complex128.

        Returns
        -------
        numpy.ndarray
            ``exp(exponent)``.
        """
        return np.exp(exponent)

    def _envelope_sources(
        self, field_t: Complex[np.ndarray, "n_nodes n_pol n_to"], bundle: StageScalars
    ) -> tuple[
        Complex[np.ndarray, "n_nodes n_pol n_to"],
        Complex[np.ndarray, "n_nodes n_pol n_to"],
    ]:
        """Return the envelope-grid plasma's windowed drive and loss at some nodes.

        Parameters
        ----------
        field_t : numpy.ndarray
            The field at the nodes, V/m.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        tuple of numpy.ndarray
            :func:`solprop.backends.numpy_ref.envelope_plasma.envelope_sources`.
        """
        envelope = self._problem.responses.envelope_plasma
        assert envelope is not None
        return envelope_sources(
            field_t,
            envelope,
            bundle,
            self._problem.grids.dt_fine,
            self._tables.t_window,
        )

    def _envelope_populations(
        self,
        nodes: Complex[np.ndarray, "n_nodes n_pol n_to"],
        initials: tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return each envelope channel's final fraction and peak rate at the nodes.

        Parameters
        ----------
        nodes : numpy.ndarray
            The physical envelope at the nodes, V/m.
        initials : tuple of float
            The channels' pre-ionised fractions here.

        Returns
        -------
        tuple of numpy.ndarray
            :func:`solprop.backends.numpy_ref.envelope_plasma.envelope_populations`.
        """
        envelope = self._problem.responses.envelope_plasma
        assert envelope is not None
        return envelope_populations(
            nodes, envelope, initials, self._problem.grids.dt_fine
        )

    def _error_norm(
        self,
        error: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        advanced: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ) -> float:
        """Return the controller's scaled norm of a trial step's embedded difference.

        Parameters
        ----------
        error : numpy.ndarray
            The embedded difference carried to the end of the step.
        state : numpy.ndarray
            The state at the anchor.
        advanced : numpy.ndarray
            The trial state at the end of the step.

        Returns
        -------
        float
            :func:`solprop.backends.numpy_ref.passes.error_norm` of the three.
        """
        return passes.error_norm(
            error,
            state,
            advanced,
            self._problem.grids.norm_mask,
            self._problem.stepper.controller,
        )

    def _rounded(
        self, values: Inexact[np.ndarray, "*shape"]
    ) -> Inexact[np.ndarray, "*shape"]:
        """Return a binary64 array a pass multiplies by, in the run's precision.

        The identity here; the single-precision path rounds it once (SEP 33).

        Parameters
        ----------
        values : numpy.ndarray
            A binary64 coefficient array evaluated on the host.

        Returns
        -------
        numpy.ndarray
            `values`.
        """
        return values

    def stats(self, spec: StatsSpec) -> dict[str, np.ndarray]:
        """Collect the requested statistics at the current position.

        The statistics of the saved field go to :func:`solprop.stats.collect`, one call
        per batch member, stacked along the leading batch axis the protocol promises;
        there is exactly one definition of each in the package and that is it. The
        statistics :class:`solprop.stats.StatDefinition` marks ``backend`` are of
        quantities a saved plane does not carry -- the electron density, the peak
        ionisation rate, the on-axis intensity, the quadrature error -- and are formed
        here from the backend's own buffers.

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
        state = self._require_state()
        grid = self._require_grid()
        if self._tdse_binding is not None:
            unavailable = TDSE_UNAVAILABLE_STATISTICS.intersection(spec.names)
            if unavailable:
                raise BackendError(
                    f"TDSE does not define legacy statistics {sorted(unavailable)}"
                )
        # Before the spatial branch, as in `numba_cpu`: a spatial run asked for a
        # general name would otherwise reach `collect_spatial` and be answered from a
        # geometry that carries no certified metric at all.
        _refuse_inadmissible(self._problem, spec)
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
            members = [
                collect_spatial(
                    spatial_spec,
                    grid,
                    self._problem.transform,
                    one,
                    z=self._z,
                    flux=self._problem.flux,
                    profiles=self._profiles,
                    tables=tables,
                    operator=self._problem.linear.separable,
                )
                for one in state
            ]
            return {
                name: np.stack([one[name] for one in members])
                for name in spatial_spec.names
            } | self._backend_statistics(
                tuple(name for name in spec.names if name in response_names), state
            )
        # The weight is a Chebyshev evaluation per column, which is the whole of what
        # the Poynting weighting adds to a statistics call on a CPU (0.053 ms of
        # 0.056 ms on the reduced-M2 case), so a spec that asks for none of the
        # Poynting statistics -- a periodic plasma subset, say -- never
        # evaluates it.
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
        # The basis metric at this plane, evaluated once for every member: the
        # statistics of `METRIC_WEIGHTED_STATISTICS` are functionals of the physical
        # same-frequency flux `a.H W a`, and on a declared change of basis W is not
        # the identity. `metric_at` returns None for an identity metric, which is
        # the statistics' own bitwise-unchanged path, so an ordinary run pays
        # nothing.
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

    def _backend_statistics(
        self,
        names: tuple[str, ...],
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ) -> dict[str, np.ndarray]:
        """Share each fresh envelope population scan among the requested statistics.

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
        population_names = (
            tuple(
                name
                for name in names
                if name.startswith("species_")
                or name in ("electron_density", "peak_ionisation_rate")
            )
            if self._problem.responses.envelope_plasma is not None
            else ()
        )
        integration_names = tuple(
            name for name in names if name in PLASMA_INTEGRATION_STATISTICS
        )
        axis_names = tuple(name for name in names if name in ON_AXIS_PLASMA_STATISTICS)
        if axis_names and not population_names:
            # The species statistics are the node scan of `_species_statistics` on
            # every route, so sharing one scan with the on-axis statistics changes
            # none of their values and lets a node on the axis supply its row.
            population_names = tuple(
                name for name in names if name.startswith("species_")
            )
        # Node scans of this observation only, by batch member: a rule with a node on
        # the axis hands its row to the on-axis statistics, and nothing outlives the
        # call, so no stale stage or rejected step can be reused.
        scans: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        result = {
            name: self._backend_stat(name, state)
            for name in names
            if name not in population_names
            and name not in integration_names
            and name not in axis_names
        }
        if integration_names:
            diagnostics = [self._integration_diagnostics(member) for member in state]
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
        if (
            name.startswith("species_")
            or self._problem.flags.n_channels > 1
            or self._problem.responses.envelope_plasma is not None
        ):
            return np.stack(
                [self._species_statistics(member)[name] for member in state]
            )
        ionised = [self._ionisation(state[member]) for member in range(len(state))]
        index = 0 if name == "electron_density" else 1
        return np.stack([one[index] for one in ionised])

    def _species_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        scans: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
        member: int = 0,
    ) -> dict[str, np.ndarray]:
        """Evaluate independent populations on the actual nonlinear nodes.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member at this plane, shape (n_modes, n_pol, n_omega).
        scans : dict or None, optional
            Where to leave this observation's channel fractions and peak rates at
            every node, keyed by `member`, for the on-axis statistics to reuse.
        member : int, optional
            The batch member, the key into `scans`.

        Returns
        -------
        dict
            The five species statistics and the two total plasma statistics.
        """
        responses = self._problem.responses
        transform, grids = self._problem.transform, self._problem.grids
        channels = responses.population_channels
        if channels:
            if transform.geometry is not None and transform.geometry.kind == "free":
                nx, ny = transform.geometry.shape
                physical = transforms.ifft2(
                    field.reshape(ny, nx, *field.shape[1:]), axes=(0, 1)
                ).reshape(field.shape)
                nodes = passes.to_time(
                    physical * grids.pad_scale,
                    grids.crop_index,
                    grids.n_to,
                    kind=self._kind,
                )
            elif transform.geometry is not None and transform.geometry.kind == "slab":
                # The slab synthesis carries the pad scale already.
                nodes = passes.to_time(
                    slab_synthesis(transform, field),
                    grids.crop_index,
                    grids.n_to,
                    kind=self._kind,
                )
            else:
                nodes = self._nodes_at(field)
            if responses.envelope_plasma is not None:
                fractions, peaks = self._envelope_populations(
                    nodes, tuple(float(c.preionisation(self._z)) for c in channels)
                )
                if scans is not None:
                    scans[member] = (fractions, peaks)
                return population_statistics(self._problem, self._z, fractions, peaks)
            rates, population = passes.plasma_channel_fractions(
                nodes,
                responses.plasma_channels,
                tuple(float(c.preionisation(self._z)) for c in channels),
                grids.dt_fine,
                responses.plasma_integral,
            )
            if scans is not None:
                scans[member] = (population[:, :, -1], rates.max(axis=-1))
            return population_statistics(
                self._problem, self._z, population[:, :, -1], rates.max(axis=-1)
            )
        empty = np.zeros((0, transform.n_nodes))
        return population_statistics(self._problem, self._z, empty, empty)

    def _integration_diagnostics(
        self, field: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> dict[str, np.ndarray]:
        """Return the plasma column's integration diagnostics at this plane (SEP 18).

        The undershoot of the exposure increments, the relative work defect of the
        loss current and the boundary level of the free-current integrand, over every
        node and field plasma channel
        (:data:`solprop.stats.PLASMA_INTEGRATION_STATISTICS`).

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.

        Returns
        -------
        dict
            The three statistics, each zero-dimensional and dimensionless.

        Raises
        ------
        BackendError
            If the problem carries no field plasma channel.
        """
        responses = self._problem.responses
        channels = responses.plasma_channels
        if not channels:
            msg = (
                "the plasma integration diagnostics read the field-grid plasma "
                "column, and this problem carries none"
            )
            raise BackendError(msg)
        initial = tuple(float(channel.preionisation(self._z)) for channel in channels)
        parts = np.zeros(6)
        for nodes in self._plasma_nodes(field):
            parts = np.maximum(
                parts,
                passes.plasma_integration_diagnostics(
                    nodes,
                    channels,
                    initial,
                    self._problem.grids.dt_fine,
                    responses.plasma_integral,
                ),
            )
        undershoot, defect, boundary = passes.plasma_integration_statistics(parts)
        return {
            "plasma_undershoot": np.asarray(undershoot),
            "plasma_work_defect": np.asarray(defect),
            "plasma_boundary_level": np.asarray(boundary),
        }

    def _plasma_nodes(
        self, field: Complex[np.ndarray, "n_modes n_pol n_omega"]
    ) -> Iterator[Float[np.ndarray, "n_nodes_chunk n_pol n_to"]]:
        """Yield the real node fields the plasma column is evaluated on, by chunk.

        The same synthesis as :meth:`_ionisation`: the physical nodes of a separable
        geometry, chunk by chunk, or the modal quadrature nodes at once.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.

        Yields
        ------
        numpy.ndarray
            Node fields on the fine time grid, V/m.
        """
        grids = self._problem.grids
        if self._problem.linear.separable is None:
            yield self._nodes_at(field)
            return
        geometry = self._problem.transform.geometry
        assert geometry is not None
        nx, ny = geometry.shape
        physical = (
            slab_synthesis(self._problem.transform, field) / grids.pad_scale
            if geometry.kind == "slab"
            else transforms.ifft2(
                field.reshape(ny, nx, *field.shape[1:]), axes=(0, 1)
            ).reshape(field.shape)
        )
        chunk = self._problem.transform.node_chunk
        for first in range(0, len(physical), chunk):
            yield passes.to_time(
                physical[first : first + chunk] * grids.pad_scale,
                grids.crop_index,
                grids.n_to,
                kind=self._kind,
            )

    def _ionisation(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the electron density and the peak ionisation rate at this plane.

        Both are read at the **node of largest ionised fraction**, which is where the
        plasma the run carries is densest and the only node either number can honestly
        describe; for a mode-averaged run that is the one node. The axis is not a
        quadrature node at all and is not where the polarisation was evaluated, so the
        field is not re-synthesised there.

        The electron density is ``density(z)`` times the fraction at the **end** of the
        window: what the pulse leaves behind, not what it reaches in passing.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.

        Returns
        -------
        tuple of numpy.ndarray
            The electron density in m^-3 and the peak rate in s^-1, both
            zero-dimensional.

        Raises
        ------
        BackendError
            If the problem declares no plasma, so there is no rate to integrate.
        """
        responses = self._problem.responses
        plasma = responses.plasma_coefficients
        if plasma is None:
            msg = (
                "the electron density and the peak ionisation rate need a compiled "
                "plasma response, and this problem declares none"
            )
            raise BackendError(msg)
        if self._problem.linear.separable is not None:
            geometry = self._problem.transform.geometry
            assert geometry is not None
            nx, ny = geometry.shape
            physical = (
                # The slab synthesis carries the pad scale, multiplied back below.
                slab_synthesis(self._problem.transform, field)
                / self._problem.grids.pad_scale
                if geometry.kind == "slab"
                else transforms.ifft2(
                    field.reshape(ny, nx, *field.shape[1:]), axes=(0, 1)
                ).reshape(field.shape)
            )
            largest, peak = -1.0, 0.0
            grids = self._problem.grids
            chunk = self._problem.transform.node_chunk
            for first in range(0, len(physical), chunk):
                nodes = passes.to_time(
                    physical[first : first + chunk] * grids.pad_scale,
                    grids.crop_index,
                    grids.n_to,
                    kind=self._kind,
                )
                rate = passes.ionisation_rate(nodes, plasma.rate)
                fraction, _ = passes.plasma_scan(
                    nodes,
                    rate,
                    plasma,
                    preionisation=float(plasma.preionisation(self._z)),
                    dt=grids.dt_fine,
                    integral=responses.plasma_integral,
                )
                densest = int(np.argmax(fraction[:, -1]))
                if fraction[densest, -1] > largest:
                    largest = float(fraction[densest, -1])
                    peak = float(np.max(rate[densest]))
            return np.asarray(float(plasma.density(self._z)) * largest), np.asarray(
                peak
            )
        nodes = self._nodes_at(field)
        rate = passes.ionisation_rate(nodes, plasma.rate)
        fraction, _ = passes.plasma_scan(
            nodes,
            rate,
            plasma,
            preionisation=float(plasma.preionisation(self._z)),
            dt=self._problem.grids.dt_fine,
            integral=responses.plasma_integral,
        )
        densest = int(np.argmax(fraction[:, -1]))
        density = float(plasma.density(self._z)) * float(fraction[densest, -1])
        return np.asarray(density), np.asarray(float(np.max(rate[densest])))

    def _peak_intensity(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> np.ndarray:
        """Return the largest intensity on the axis of the waveguide, in W/m^2.

        ``eps0 c/2 max_t sum_c |E_{a,c}(0, t)|^2`` with the analytic field synthesised
        at ``r = 0`` from the modes' own profiles, both Cartesian components. The axis
        is not a quadrature node, so the mode fields are evaluated there rather than
        interpolated, and both components are taken so that the answer does not depend
        on which one the transform resolves or the state carries. One definition for
        every run: a mode-averaged run synthesises its one mode the same way, which is
        ``P / A_eff`` for a top-hat profile and the true on-axis value for a real one
        -- 1.77 times the average for HE_11, so ``P / A_eff`` would under-report it by
        that factor. The on-axis matrix is built once and rescaled with the taper
        (:meth:`_on_axis_synthesis`), which is what makes this statistic cheap enough
        for the between-planes channel.

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
                "NumpyRefBackend(problem, modes=modes)"
            )
            raise BackendError(msg)
        analytic = analytic_field(self._require_grid(), field)
        columns = np.tensordot(
            self._on_axis_synthesis(self._modes), analytic, axes=([1], [0])
        )
        power = np.sum(np.abs(columns) ** 2, axis=tuple(range(columns.ndim - 1)))
        return np.asarray(0.5 * EPS0 * C * float(np.max(power)))

    def _nodes_at(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
    ) -> Num[np.ndarray, "n_nodes n_components n_to"]:
        """Return the physical field at every quadrature node on the fine time grid.

        Parameters
        ----------
        field : numpy.ndarray
            One batch member's state, shape ``(n_modes, n_pol, n_omega)``.

        Returns
        -------
        numpy.ndarray
            The field in V/m, shape ``(n_nodes, n_components, n_to)``. Here
            ``n_components = transform.n_pol * n_pol`` counts physical components,
            independently of the state's polarisation axis.
        """
        grids, transform = self._problem.grids, self._problem.transform
        taper = (
            1.0
            if transform.taper_scale is None
            else float(transform.taper_scale(self._z))
        )
        modal_t = passes.to_time(field, grids.crop_index, grids.n_to, kind=self._kind)
        return passes.synthesise(modal_t, transform.synthesis, taper, transform.n_pol)

    def metadata(self) -> dict[str, object]:
        """Return execution facts and cumulative logical and atomic call ledgers.

        Returns
        -------
        dict
            Provider-free, serializable facts. A PEC predictor is explicitly
            labelled and is never reported as an accepted-field diagnostic.
        """
        record = super().metadata()
        if self._gated_plasma:
            record["_benchmark_plasma_source"] = "gated_nonresumable"
            record["plasma_source"] = {"form": "gated_benchmark", "resumable": False}
        if self._tdse_binding is not None:
            record["atomic_calls"] = dataclasses.asdict(self._tdse_binding.ledger)
            record["tdse_execution"] = dict(self._tdse_binding.metadata())
            record["tdse_last_evaluation"] = {
                "purpose": self._tdse_last_purpose,
                "z": self._tdse_last_z,
                "projection": self._tdse_last_projection,
            }
            record["tdse_source"] = _source_record(self._problem)
        return record

    def restore_call_ledgers(
        self, logical: Mapping[str, object], atomic: Mapping[str, object] | None = None
    ) -> None:
        """Restore cumulative counters while leaving the Adams history empty.

        Parameters
        ----------
        logical : mapping
            Persisted logical call counters and accepted-order record.
        atomic : mapping or None, optional
            Persisted atomic attempts; unused reservations are released.

        Raises
        ------
        BackendError
            If a ledger is incompatible with this run.
        """
        super().restore_call_ledgers(logical)
        if atomic:
            if self._tdse_binding is None:
                raise BackendError("atomic call ledger requires a TDSE binding")
            self._tdse_binding.restore_ledger(AtomicCallLedger.from_record(atomic))

    def memory_ledger(self) -> dict[str, int]:
        """Return the size of every buffer the backend works in, in bytes.

        The reference backend allocates its stage and transform buffers per pass instead
        of keeping a preallocated pool, so this reports what one step has resident at
        its peak -- which is what the optimised backends hold for the whole run, and
        what the budget of ``docs/dev/architecture.md`` section 8 is measured against.
        Every buffer is named even when it is empty.

        The space-domain entries are sized by ``transform.node_chunk`` and not by the
        node count, which is the whole point of the chunk loop (``architecture.md``
        section 13): holding ``(n_to, n_pol n_nodes)`` for every node at once would be
        210 MB against 45 MB on the comb-to-continuum target.

        Returns
        -------
        dict
            Buffer name to bytes.
        """
        problem = self._problem
        distinct = int(np.unique(np.append(problem.stepper.tableau.nodes, 1.0)).size)
        general: dict[str, int] = dict.fromkeys(_GENERAL_LEDGER_ROWS, 0)
        if self._general is not None:
            # The route exists exactly when the tables do (`_general_maps`).
            tables = problem.transform.modal_tables
            assert tables is not None
            general = general_memory(self._general, tables, problem.grids.n_to)
        ledger = adams_memory(problem, "numpy_ref") | {
            # One definition of these rows, which the construction-free estimate of
            # `solprop.backends.memory` sums too.
            **reference_base_memory(
                problem,
                tabulating=self._propagator_mode == "tabulate",
                embedded=self._embedded is not None,
                noise=self._noise is not None,
                statistics_weight=self._statistics_power_weight.nbytes,
                spatial_absorber=self._spatial_absorber.nbytes,
            ),
            # The coupled flow's own resident arrays: the frozen block of every
            # section with its Schur factors, the residual's terms, and the one
            # forward/inverse action pair held for the section and length last seen.
            # All of them are on the active bins alone.
            "coupled_flow_factors": 0
            if self._flow is None
            else self._flow.vectors.nbytes
            + self._flow.triangular.nbytes
            + self._flow.frozen.nbytes,
            "coupled_flow_residual_terms": 0
            if self._flow is None or self._flow.residual_terms is None
            else self._flow.residual_terms.nbytes,
            "coupled_flow_actions": 0
            if self._flow is None
            else 2 * distinct * self._flow.frozen[0].nbytes,
            # The general modal route's buffers are its own: the two execution maps,
            # the per-node tensor and the record's resident E/H reference are
            # whole-problem allocations no chunk shrinks, and its chunk buffers are
            # three components wide and complex on a grid where the legacy ones are
            # one or two and real. Every row is zero on every
            # other route.
            **general,
            # A low-storage march's state at its last field-free plane (SEP 22, Q14),
            # held from the first such plane; the admission charges it beforehand.
            "held_plane": 0 if self._held is None else self._held.nbytes,
            **(
                {}
                if self._tdse_binding is None
                else {
                    f"tdse_{name}": size
                    for name, size in self._tdse_binding.memory_ledger().items()
                }
            ),
        }
        if self._gated_plasma:
            for name in ledger:
                if (
                    name.startswith("held_current_")
                    and name != "held_current_derivative"
                ):
                    ledger[name] = 0
        return ledger

    def close(self) -> None:
        """Release the buffers. Idempotent."""
        self._state = None
        self._noise = None
        self._raman_kernel = None
        self._last_plasma = None
        self._spatial_absorber = np.empty((0, 0))
        self._raman_rates = None
        self._raman_densities = None
        self._flow_cache = None
        self._statistics_power_weight = np.empty(0)
        self._adams_held = None
        if self._low_storage is not None:
            self._low_storage.close()
        self._closed = True

    def set_initial(
        self,
        field: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        noise: NoiseRealisation | None,
        *,
        z: float = 0.0,
        resume: bool = False,
    ) -> None:
        """Load a state; an Adams march restarts its history from it.

        Parameters
        ----------
        field : numpy.ndarray
            The state in the canonical layout.
        noise : solprop.noise.NoiseRealisation or None
            The realisation to carry, or ``None`` for the problem's own.
        z : float, optional
            The position the state belongs to, m; positive for a resume, which starts
            the history afresh unless
            :meth:`~solprop.backends.driver.StepDriver.restore_march` then continues
            it.
        resume : bool, optional
            Whether `field` came back from a saved plane, and so never takes the
            ``input`` draw.
        """
        super().set_initial(field, noise, z=z, resume=resume)
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

    def _reserve_source_calls(self, count: int) -> None:
        """Reserve every atomic chunk required by one complete Adams trial.

        Parameters
        ----------
        count : int
            Number of logical source evaluations required by the trial.
        """
        binding = self._tdse_binding
        if binding is None:
            return
        try:
            binding.reserve(count)
        except ResponseBudgetError as error:
            raise ResponseBudgetError(
                "the TDSE atomic-call budget cannot reserve the next trial "
                f"at z = {self._z:.6g} m ({error.calls} charged); "
                "the run stops at its last accepted step, unfinished",
                z=self._z,
                field=self._adams_state().copy(),
                calls=error.calls,
            ) from None

    def _release_source_calls(self) -> None:
        """Release unused atomic reservations after a completed or failed trial."""
        if self._tdse_binding is not None:
            self._tdse_binding.release_reservation()

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
        """Evaluate the source once through the reference passes.

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
        held = self._adams_held
        if held is not None and held[0] == anchor and held[1] == position:
            bundle, propagator = held[2], held[3]
        else:
            bundle = self._stage_scalars(anchor, position)
            propagator = self._propagator(self._delta_phi(bundle))
        self._adams_held = None
        noise = self._noise_at(bundle)
        field_in = lab if noise is None else lab + noise
        if self._tdse_binding is not None:
            return (
                self._nonlinear(field_in, bundle, 1.0 / propagator, purpose=purpose),
                propagator,
            )
        return self._nonlinear(field_in, bundle, 1.0 / propagator), propagator

    def _dense_stages(self) -> list[np.ndarray]:
        """Return the last trial's stage derivatives, in its anchor's frame.

        Returns
        -------
        list of numpy.ndarray
            One per tableau stage, in the canonical layout.
        """
        return list(self._last_stages)

    def _adams_block(self, z: float, dz: float) -> object:
        """Return the frozen block of the section a step from `z` is anchored in.

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
        from solprop.backends.multistep import CoupledBlock  # noqa: PLC0415 - cycle

        flow = self._flow
        if flow is None:
            return None
        bundle = self._stage_scalars(z, z + dz)
        section = bundle.flow_section
        assert section is not None
        residual = None
        if flow.residual_terms is not None and not flow.cf4[section]:
            if self._block_residual is None:
                self._block_residual = BlockResidual(flow)
            residual = functools.partial(self._block_residual, section)
        return CoupledBlock(
            active=flow.active,
            vectors=flow.vectors[section],
            triangular=flow.triangular[section],
            section=section,
            end_weights=bundle.flow_weights,
            residual=residual,
        )

    def _adams_lab_source(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        position: float,
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Evaluate the source once through the reference passes, untransported.

        The same passes as :meth:`_adams_source`, with the stage's inverse propagator
        replaced by ones: a multiplication by one is exact, so the sample is the
        lab-frame source bit for bit, and every route's evaluation stays the one the
        Runge-Kutta march uses.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame field at `position`.
        position : float
            The evaluation position, m.
        purpose : str, optional
            Atomic-call accounting label; ignored when no atomic service is bound.

        Returns
        -------
        numpy.ndarray
            ``F(lab, position)`` in the lab frame.
        """
        bundle = self._stage_scalars(position, position)
        noise = self._noise_at(bundle)
        field_in = lab if noise is None else lab + noise
        if self._tdse_binding is not None:
            return self._nonlinear(field_in, bundle, self._lab_frame, purpose=purpose)
        return self._nonlinear(field_in, bundle, self._lab_frame)

    def _adams_propagator(
        self, anchor: float, position: float
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return ``U(position, anchor)``, held for the Adams step's source call.

        Only an Adams march reads it back (:meth:`_adams_source`), so only an Adams
        run holds it: the low-storage march applies its flows at once, and a held one
        would be a volume its ledger does not name.

        Parameters
        ----------
        anchor, position : float
            The step's anchor and its end, m.

        Returns
        -------
        numpy.ndarray
            ``exp(Phi(position) - Phi(anchor))``.
        """
        bundle = self._stage_scalars(anchor, position)
        propagator = self._propagator(self._delta_phi(bundle))
        if self._adams is not None:
            self._adams_held = (anchor, position, bundle, propagator)
        return propagator

    def _trial_step(
        self, z: float, dz: float
    ) -> tuple[float, Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]]:
        """Take one interaction-picture Runge-Kutta step and score it.

        Every stage is evaluated. The tableau's first-same-as-last property would let
        the last stage of one step serve as the first of the next, after rebasing it by
        ``exp(dPhi)`` because the interaction picture is re-anchored every step; the
        reference backend does not take that saving, so that no stage it uses depends on
        the bookkeeping of the step before (``docs/dev/stepper.md`` section 10, trap
        10). A coupled flow would need the analogous certified forward action of its
        own to reuse a stage, so the saving stays refused there too, and is **absent**
        here rather than skipped by a flag.

        A problem carrying ``linear.coupled_flow`` multiplies each stage's scalar
        propagator by the section's Schur matrix-function action. The two factors
        commute -- the cumulative diagonal beside a coupled flow is required to be a
        multiple of the identity (:func:`_refuse_flow_problem`) -- so their product is
        the flow of the sum, exactly, and the shared scalar stays analytically
        integrated in ``Phi``.

        A ``"frozen_residual"`` flow adds the **explicit residual**
        ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)`` to each stage's lab-frame
        source, before the interaction picture's own factors carry it back, so it is
        integrated by the embedded Runge-Kutta pair exactly as the material response
        is. Freezing is then a rearrangement of the original equation and not a
        second-order approximation: whatever block the exponential carries, the
        residual is the rest of the generator.

        A step anchored in a ``"cf4"`` section carries **no** residual: its flow
        actions are the commutator-free exponentials of the whole generator over each
        stage's own length (:func:`_cf4_action`), so the source is the material
        response alone and the flow's own error is outside what the embedded pair
        measures: a known limitation of the controller on a CF4 section.

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
        state = self._require_state()
        self._hoist(z)
        tableau = self._problem.stepper.tableau
        # The whole step and every stage, read together: each table once.
        self._stage_bundles(
            z, [z + dz, *(z + float(node) * dz for node in tableau.nodes)]
        )
        exponents = self._exponents(z, dz)
        step = self._stage_scalars(z, z + dz)
        actions = self._flow_actions(z, step.flow_section, dz)
        stages: list[Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]] = []
        solution = np.zeros_like(state)
        error = np.zeros_like(state)
        for index, node in enumerate(tableau.nodes):
            fraction = float(node)
            combination = np.zeros_like(state)
            for previous in range(index):
                combination = passes.accumulate(
                    combination,
                    stages[previous],
                    float(tableau.matrix[index, previous]),
                )
            bundle = self._stage_scalars(z, z + fraction * dz)
            exponent = exponents(fraction, bundle)
            forward, inverse = actions(fraction)
            lab = passes.stage_input(
                self._channels(state, forward),
                self._channels(combination, forward),
                dz,
                self._propagator(exponent),
                self._noise_at(bundle),
            )
            back = self._propagator(-exponent)
            self._rk_source_evaluations += 1
            drive = self._nonlinear(lab, bundle, back)
            residual = self._residual(lab, step.flow_section, bundle)
            if residual is not None:
                drive = drive + back * residual
            stage = self._channels(drive, inverse)
            stages.append(stage)
            solution = passes.accumulate(
                solution, stage, float(tableau.solution_weights[index])
            )
            error = passes.accumulate(error, stage, float(tableau.error_weights[index]))
        # Kept for the continuous extension of an accepted step (SEP 14 phase 3).
        self._last_stages = stages
        propagator = self._propagator(exponents(1.0, step))
        whole = actions(1.0)[0]
        advanced = passes.accept_step(
            self._channels(state, whole),
            propagator,
            dz,
            self._channels(solution, whole),
        )
        # The embedded difference is carried to the end of the step by the same
        # exponential as the solution, so the norm scales the local error of the
        # propagated state rather than of the interaction variable.
        norm = self._error_norm(
            propagator * (dz * self._channels(error, whole)), state, advanced
        )
        return norm, advanced

    def _channels(
        self,
        values: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        action: Complex[np.ndarray, "n_active_c n_modes_c_out n_modes_c_in"] | None,
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Mix the channel axis by one coupled-flow action, or return the values.

        Parameters
        ----------
        values : numpy.ndarray
            Any array of the state's layout.
        action : numpy.ndarray or None
            The action on the active bins, or ``None`` for a problem with no coupled
            flow, which is every route but the coherent wall.

        Returns
        -------
        numpy.ndarray
            The values, mixed or untouched.
        """
        if self._flow is None or action is None:
            return values
        return _apply_flow(values, action, self._flow.active)

    def _residual(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        section: int,
        bundle: StageScalars,
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"] | None:
        """Return the explicit residual's lab-frame source term, or nothing.

        ``None`` for every problem but a ``"frozen_residual"`` coupled flow: the
        constant section has an identically zero residual and is not
        multiplied by one, so its arithmetic is bit for bit what it was; a step
        anchored in a ``"cf4"`` section has none either, its whole generator being in
        the exponential.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame stage input.
        section : int
            The step's section, from the step bundle's ``flow_section``.
        bundle : solprop.translate.stage.StageScalars
            The stage's own bundle, whose ``flow_weights`` the terms are combined over.

        Returns
        -------
        numpy.ndarray or None
            ``R_s(z_1) u`` in the state's layout, or ``None``.
        """
        flow = self._flow
        if flow is None or flow.residual_terms is None or flow.cf4[section]:
            return None
        assert bundle.flow_weights is not None
        return _residual_drive(
            lab,
            _residual_action(flow, section, bundle.flow_weights),
            flow.active,
            flow.residual_bins,
        )

    def _flow_actions(
        self, z: float, section: int, dz: float
    ) -> Callable[
        [float],
        tuple[
            Complex[np.ndarray, "n_active_a n_modes_a_out n_modes_a_in"] | None,
            Complex[np.ndarray, "n_active_b n_modes_b_out n_modes_b_in"] | None,
        ],
    ]:
        """Return the memoised forward and inverse flow actions of one step.

        The generator is **frozen** across the step -- at the midpoint of the section
        the step's anchor lies in -- so an action depends on the stage fraction only
        through the length ``c_i h`` it covers, and the six distinct DOPRI5 fractions
        need at most six pairs; the tableau uses the last one twice and the zero
        fraction is the identity. The whole step's pair is checked against that
        section's declared conditioning row, which bounds every shorter one because
        ``exp(c dz (mu_max - mu_min))`` increases with ``c``.

        The pairs also survive **between** steps, for the one section and trial length
        last seen: a fixed schedule and an accepted adaptive step both repeat their
        ``dz``, and an action that depends on nothing but ``dz`` and a frozen generator
        is then the same matrix. Exactly one pair of them is held, so what a step has
        resident at its peak -- which is what :meth:`memory_ledger` reports -- is
        unchanged, and a step that crosses into the next section rebuilds, which is a
        cache invalidation at the section edge and is what makes the edge a kink of
        the flow (``docs/dev/stepper.md`` section 2). The conditioning of a held
        length was measured when it was built.

        A step anchored in a ``"cf4"`` section is not frozen: its actions are the
        commutator-free exponentials of the generator at Gauss nodes **inside** each
        stage's own interval, so they depend on where the step starts as well as on
        its length, and the key carries `z`. Within one step the six fractions are
        still computed once each; between steps nothing is reused. Every one of them
        is checked against the section's declared pair bound, factor by factor
        (:func:`_check_cf4_growth`).

        Parameters
        ----------
        z : float
            Where the step starts, m.
        section : int
            Which section the step is anchored in, from the stage bundle's
            ``flow_section``.
        dz : float
            Length of the trial step, m.

        Returns
        -------
        callable
            ``fraction -> (U, U^-1)``, each ``None`` when no coupled flow is carried.
        """
        flow = self._flow
        if flow is None:
            return lambda fraction: (None, None)  # noqa: ARG005
        held = self._flow_cache
        cf4 = bool(flow.cf4[section])
        key = (section, dz, float(z) if cf4 else None)
        cache: dict[float, tuple[np.ndarray | None, np.ndarray | None]] = (
            held[1] if held is not None and held[0] == key else {}
        )
        self._flow_cache = (key, cache)

        def action(
            fraction: float,
        ) -> tuple[np.ndarray | None, np.ndarray | None]:
            """Return the pair at a stage fraction, computing it at most once.

            Parameters
            ----------
            fraction : float
                The tableau node ``c_i``, dimensionless, in ``[0, 1]``.

            Returns
            -------
            tuple
                The forward and inverse actions, or ``(None, None)`` at fraction zero,
                where both are the identity.
            """
            if fraction == 0.0:
                return None, None
            if fraction not in cache:
                if cf4:
                    forward, inverse, growth = _cf4_action(
                        flow, float(z), fraction * dz, section
                    )
                    _check_cf4_growth(flow, section, growth, fraction * dz)
                    cache[fraction] = (forward, inverse)
                else:
                    pair = _flow_action(flow, section, fraction * dz)
                    if fraction == 1.0:
                        _check_flow_conditioning(flow, section, pair, dz)
                    cache[fraction] = pair
            return cache[fraction]

        return action

    def _apply_mask(
        self, state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Zero the state outside the active band.

        Parameters
        ----------
        state : numpy.ndarray
            The state in the canonical layout.

        Returns
        -------
        numpy.ndarray
            The masked state.
        """
        return passes.apply_hard_mask(state, self._problem.grids.active_mask)

    def _begin_plane(self) -> None:
        """Forget the last plane's quadrature alarm, so each plane reports its own."""
        self._alarm = math.nan
        self._alarm_reported = False

    def _on_accepted_step(self) -> None:
        """Score the quadrature alarm from what the last stage left behind."""
        self._update_alarm()

    def _record(self, z: float, dz: float, norm: float, accepted: bool) -> None:
        """Record one attempt, and tell an atomic binding the trial's outcome.

        Every attempt of the Adams march opens one trial with its reservation
        (:meth:`_reserve_source_calls`) and ends here, accepted or rejected, so an
        archive can label a rejected trial's evaluations (SEP 20 section 5.8). A
        replayed step is recorded after its reservation is released; it is still
        the latest trial. Nothing about the step or its accounting changes.

        Parameters
        ----------
        z : float
            Position at the end of the step, m.
        dz : float
            Length of the step, m.
        norm : float
            The scaled error norm the attempt scored.
        accepted : bool
            Whether the step was kept.
        """
        super()._record(z, dz, norm, accepted)
        if self._tdse_binding is not None:
            self._tdse_binding.record_trial(accepted=accepted)

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
            The state with the hard band mask and, when one is present, the temporal
            absorber applied. The ``"split_step"`` absorber is followed by the mask, so
            the accepted state never holds content outside the active band.
        """
        masked = self._apply_mask(state)
        responses = self._problem.responses
        rate = responses.temporal_absorber
        if rate is None or responses.absorber_scheme != "split_step":
            # Either there is no collar, or ADR-025's other scheme applied it inside
            # every stage, where it is already in the accepted state.
            return masked
        if (monitor := self._split_monitor) is not None:
            # The time factor acts alike on every column of any transverse basis, so
            # the stored state's power is what the split takes of, wherever it acts;
            # read as the physical state, which a single-precision state is scaled
            # from.
            self._split_removed += monitor.split_removal(
                self._observed_field(masked), dz
            )
        geometry = self._problem.transform.geometry
        if geometry is not None and geometry.kind == "slab":
            # A (p, y) row is a modal amplitude whose time signal is a real field's,
            # and the rate is the same at every node, so it is applied row by row.
            transform = self._problem.transform
            physical = slab_rows(transform, masked[0])
            chunk = transform.node_chunk
            for first in range(0, len(physical), chunk):
                rows = slice(first, first + chunk)
                physical[rows] = passes.apply_temporal_rate(
                    physical[rows][None], rate, dz, kind=self._kind
                )[0]
            return self._apply_mask(slab_columns(transform, physical)[None])
        if geometry is not None and geometry.kind == "free":
            nx, ny = geometry.shape
            shape = (ny, nx, *masked.shape[2:])
            physical = transforms.ifft2(masked[0].reshape(shape), axes=(0, 1)).reshape(
                masked.shape[1:]
            )
            chunk = self._problem.transform.node_chunk
            for first in range(0, len(physical), chunk):
                rows = slice(first, first + chunk)
                physical[rows] = passes.apply_temporal_rate(
                    physical[rows][None], rate, dz, kind=self._kind
                )[0]
            result = transforms.fft2(physical.reshape(shape), axes=(0, 1)).reshape(
                masked.shape
            )
            return self._apply_mask(result)
        # The product in time spreads content across every bin, the DC among them, so
        # the band is imposed again after it, as the in-stage term is masked.
        return self._apply_mask(
            passes.apply_temporal_rate(masked, rate, dz, kind=self._kind)
        )

    def _update_alarm(self) -> None:
        """Score the quadrature from the last stage of an accepted step.

        The relative difference between the projections of the same node polarisation
        with the Kronrod rule and with its embedded Gauss subset, at the cost of one
        extra forward transform per accepted step rather than a whole second evaluation
        of the transform.
        It is a **statistic and a warning, never a controller input**: a
        quadrature too coarse for the integrand is a modelling error the step size
        cannot fix, and feeding it to the controller would only shrink the step until
        the run stopped.

        The value is ``nan`` when the rule has no embedded pair, because the difference
        is then identically zero and reading that as convergence would be wrong; and
        ``nan`` where the polarisation itself vanishes, at a true vacuum end, because a
        relative measure of nothing says nothing.
        """
        if (
            self._last_coarse is None
            or self._last_projected is None
            or self._last_bundle is None
        ):
            self._alarm = math.nan
            return
        grids = self._problem.grids
        weight = self._prefactor(self._last_bundle) * self._tables.omega_window
        fine = weight * passes.to_frequency(
            self._last_projected, grids.crop_index, kind=self._kind
        )
        coarse = weight * passes.to_frequency(
            self._last_coarse, grids.crop_index, kind=self._kind
        )
        if self._last_envelope is not None:
            assembly = self._problem.responses.envelope_plasma
            assert assembly is not None
            drive, loss, coarse_drive, coarse_loss = self._last_envelope
            assert coarse_drive is not None and coarse_loss is not None
            for factor, full, reduced in (
                (assembly.factors.drude, drive, coarse_drive),
                (assembly.factors.loss, loss, coarse_loss),
            ):
                fine += (
                    weight
                    * factor
                    * passes.to_frequency(full, grids.crop_index, kind=self._kind)
                )
                coarse += (
                    weight
                    * factor
                    * passes.to_frequency(reduced, grids.crop_index, kind=self._kind)
                )
        if self._last_plasma is not None:
            gated, edge, coarse_gated, coarse_edge = self._last_plasma
            assert coarse_gated is not None and coarse_edge is not None
            fine += weight * self._held_streams(gated, edge)
            coarse += weight * self._held_streams(coarse_gated, coarse_edge)
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

    def _nonlinear(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        *,
        purpose: TDSEPurpose = "predictor",
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Evaluate one stage derivative through the transform passes.

        The pipeline of ``architecture.md`` section 8 for any node count: one batched
        inverse transform of every modal column, then, per chunk of
        ``transform.node_chunk`` nodes, the synthesis GEMM, the fused response pass and
        the projection GEMM accumulated over the chunks, then one batched forward
        transform, the crop and the prefactor. Mode-averaged propagation is the
        one-node, one-column case of exactly this code.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame stage input from
            :func:`solprop.backends.numpy_ref.passes.stage_input`.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars: the prefactor's weights and every scalar path the
            responses read, all at the stage's own position.
        inverse_propagator : numpy.ndarray
            ``exp(-dPhi_i)``.
        purpose : str, optional
            Source-evaluation label for atomic accounting and diagnostics.

        Returns
        -------
        numpy.ndarray
            The stage derivative in the interaction variable.
        """
        responses = self._problem.responses
        grids = self._problem.grids
        if self._tdse_binding is not None:
            return self._nonlinear_tdse(lab, bundle, inverse_propagator, purpose)
        if self._problem.linear.separable is not None:
            geometry = self._problem.transform.geometry
            evaluate = (
                self._nonlinear_slab
                if geometry is not None and geometry.kind == "slab"
                else self._nonlinear_free
            )
            return evaluate(lab, bundle, inverse_propagator)
        if self._general is not None:
            return self._nonlinear_general(lab, bundle, inverse_propagator)
        if len(responses.response_order) == 1:
            # Nothing but the accumulation was compiled, so the polarisation is
            # identically zero and the step is the exact linear flow. Running the
            # transforms to multiply by nothing would only add round-off -- unless
            # ADR-025's in-stage absorber is active, which is a term of its own and
            # would otherwise be dropped from a problem that has no response at all.
            if not self._in_stage_absorber:
                return np.zeros_like(lab)
            alone = passes.to_time(
                lab[0], grids.crop_index, grids.n_to, kind=self._kind
            )
            return passes.stage_output(
                np.zeros_like(lab[0]),
                np.zeros_like(lab[0]),
                self._tables.omega_window,
                inverse_propagator,
                self._absorber_term(alone, may_quiet=True),
            )
        modal_t = passes.to_time(lab[0], grids.crop_index, grids.n_to, kind=self._kind)
        geometry = self._problem.transform.geometry
        if (
            geometry is not None
            and geometry.kind == "radial"
            and bool(responses.plasma_channels)
            and not self._gated_plasma
        ):
            spectrum = self._radial_plasma_spectrum(modal_t, bundle)
        else:
            projected, coarse = self._over_nodes(modal_t, bundle)
            self._last_projected = projected
            self._last_coarse = coarse
            self._last_bundle = bundle
            spectrum = passes.to_frequency(projected, grids.crop_index, kind=self._kind)
            if self._last_plasma is not None:
                gated, edge, _, _ = self._last_plasma
                spectrum += self._held_streams(gated, edge)
            if responses.envelope_plasma is not None:
                assert self._last_envelope is not None
                drive, loss, _, _ = self._last_envelope
                factors = responses.envelope_plasma.factors
                spectrum += self._rounded(factors.drude) * passes.to_frequency(
                    drive, grids.crop_index, kind=self._kind
                )
                spectrum += self._rounded(factors.loss) * passes.to_frequency(
                    loss, grids.crop_index, kind=self._kind
                )
        return passes.stage_output(
            spectrum,
            self._prefactor(bundle),
            self._tables.omega_window,
            inverse_propagator,
            self._absorber_term(modal_t, may_quiet=True),
        )

    def _nonlinear_tdse(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        purpose: TDSEPurpose,
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Evaluate the declared atomic binding and combine independent P/J sources.

        Parameters
        ----------
        lab : numpy.ndarray
            Lab-frame modal input, shape ``(n_batch, n_modes, n_pol, n_omega)``.
        bundle : StageScalars
            Stage density, modal weights and absolute position.
        inverse_propagator : numpy.ndarray
            Inverse linear propagation from this stage to its anchor.
        purpose : str
            Evaluation label used by the atomic call ledger.

        Returns
        -------
        numpy.ndarray
            Modal derivative in the anchor frame, with the in-stage collar.
        """
        problem = self._problem
        grids, transform = problem.grids, problem.transform
        binding = self._tdse_binding
        assert binding is not None and problem.current_numerator is not None
        modal_t = passes.to_time(lab[0], grids.crop_index, grids.n_to, kind="field")
        fields = np.ascontiguousarray(
            passes.synthesise(modal_t, transform.synthesis, bundle.taper_scale, 1)[
                :, 0, :
            ]
        )
        result = binding.evaluate(
            fields, z=bundle.z1, purpose=purpose, density=bundle.density
        )
        polarisation, current = atomic_sources(fields, result, problem, bundle)
        source, projection = project_atomic_sources(
            polarisation,
            current,
            result,
            problem,
            bundle,
            self._prefactor(bundle),
            edge=atomic_edge_source(result, problem, bundle),
        )
        self._tdse_last_purpose, self._tdse_last_z = purpose, bundle.z1
        self._tdse_last_projection = projection
        return passes.stage_output(
            source,
            np.ones_like(source),
            grids.omega_window,
            inverse_propagator,
            self._absorber_term(modal_t),
        )

    def _nonlinear_general(
        self,
        lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        bundle: StageScalars,
        inverse_propagator: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Evaluate one stage derivative through a general modal guide's maps.

        The route's own execution path, in
        :mod:`solprop.backends.numpy_ref.general_modal`, beside the fused pass and
        not an extension of it. The synthesis is a complex
        frequency-dependent map of three Cartesian components, so it happens in the
        frequency domain and the field a response sees is the analytic signal. What
        is shared with the legacy path is everything after the projection -- the
        prefactor, which is the crop scale alone here, the spectral collar, the
        in-stage absorber and the return to the interaction picture.

        The quadrature alarm is not scored on this route: the general node rule has
        no embedded coarse pair, and a difference that is identically zero read as
        convergence would be wrong, as for any unpaired rule.

        A guide with no responding region is short-circuited, as the legacy path
        short-circuits a problem whose `response_order` is the accumulation alone.

        Parameters
        ----------
        lab : numpy.ndarray
            The lab-frame stage input.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.
        inverse_propagator : numpy.ndarray
            ``exp(-dPhi_i)``.

        Returns
        -------
        numpy.ndarray
            The stage derivative in the interaction variable.
        """
        assert self._general is not None
        grids = self._problem.grids
        # A guide whose every material region is linear has an identically zero
        # tensor, so the source is zero for every state and every position: the
        # synthesis, the 3 x n_nodes transforms, the contraction and the projection
        # would spend a tenth of a second per step producing it (measured 108 ms on
        # the 288-node strand). ADR-025's in-stage absorber is a term of its own and
        # is not dropped with it.
        spectrum = (
            general_source(
                lab[0],
                self._general,
                grids.crop_index,
                n_to=grids.n_to,
                pad_scale=grids.pad_scale,
                transform_weights=bundle.transform_weights,
                dispersive_densities=bundle.dispersive_kerr_densities,
            )
            if self._general_responds
            else np.zeros_like(lab[0])
        )
        absorber = None
        if self._in_stage_absorber:
            absorber = self._absorber_term(
                passes.to_time(lab[0], grids.crop_index, grids.n_to, kind=self._kind)
            )
        return passes.stage_output(
            spectrum,
            self._prefactor(bundle),
            grids.omega_window,
            inverse_propagator,
            absorber,
        )

    def _column_sources(
        self,
        field_t: np.ndarray,
        bundle: StageScalars,
        raman: np.ndarray | None,
    ) -> passes.NonlinearSources:
        """Evaluate one column; the historical integral is benchmark-only.

        Parameters
        ----------
        field_t : numpy.ndarray
            Physical node fields in V/m, with time last.
        bundle : StageScalars
            Densities, populations and other response scalars at this position.
        raman : numpy.ndarray or None
            Damped Raman convolution spectrum, when required by the response.

        Returns
        -------
        NonlinearSources
            Collared local polarisation in C/m² and ungated current in A/m².
        """
        responses, grids = self._problem.responses, self._problem.grids
        sources = passes.nonlinear_sources(
            field_t,
            responses,
            self._tables.t_window,
            kind=self._kind,
            density=bundle.density,
            preionisation=bundle.preionisation,
            dt=grids.dt_fine,
            carrier=self._carrier,
            raman_spectrum=raman,
            scalars=bundle,
        )
        if not self._gated_plasma or sources.current is None:
            return sources
        return passes.NonlinearSources(
            sources.polarisation
            + self._tables.t_window
            * responses.plasma_integral.integrate(sources.current, grids.dt_fine),
            None,
        )

    def _held_streams(self, gated: np.ndarray, edge: np.ndarray) -> np.ndarray:
        """Transform and combine the current products in polarisation units.

        Parameters
        ----------
        gated, edge : numpy.ndarray
            Fine-time gJ and g'J rows, optionally projected before this call.

        Returns
        -------
        numpy.ndarray
            Cropped raw-DFT polarisation-equivalent source, zero on inactive bins.
        """
        grids = self._problem.grids
        return passes.held_plasma_spectrum(
            gated, edge, grids.omega, grids.crop_index, grids.active_mask
        )

    def _held_current(self, current: np.ndarray) -> np.ndarray:
        """Form and transform the two collared products of a nodal current.

        Parameters
        ----------
        current : numpy.ndarray
            Ungated nodal current in A/m², with fine time last.

        Returns
        -------
        numpy.ndarray
            Cropped raw-DFT polarisation-equivalent source at the nodes.
        """
        return self._held_streams(
            self._tables.t_window * current, self._tables.t_window_derivative * current
        )

    def _radial_plasma_spectrum(
        self,
        modal_t: np.ndarray,
        bundle: StageScalars,
    ) -> np.ndarray:
        """Transform three nodal streams, combine, then project once in frequency.

        Parameters
        ----------
        modal_t : numpy.ndarray
            Radial spectral-column fields on the fine time grid.
        bundle : StageScalars
            Densities, modal weights and other stage response scalars.

        Returns
        -------
        numpy.ndarray
            Combined projected non-plasma and held-current source spectrum.
        """
        problem = self._problem
        transform, grids = problem.transform, problem.grids
        # The field's own precision: complex128, or complex64 under SEP 33.
        spectrum = np.zeros(
            problem.shape[1:], dtype=np.result_type(modal_t, np.complex64)
        )
        raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        for first in range(0, transform.n_nodes, transform.node_chunk):
            rows = slice(
                first * transform.n_pol,
                (first + transform.node_chunk) * transform.n_pol,
            )
            field = passes.synthesise(
                modal_t,
                self._tables.synthesis[rows],
                bundle.taper_scale,
                transform.n_pol,
            )
            sources = self._column_sources(field, bundle, raman)
            assert sources.current is not None
            nodal = passes.to_frequency(
                sources.polarisation, grids.crop_index, kind="field"
            )
            nodal += self._held_current(sources.current)
            spectrum += passes.project(
                nodal,
                self._tables.projection[:, rows],
                bundle.taper_scale,
                transform.n_pol,
            )
        return spectrum

    def _nonlinear_free(
        self,
        lab: np.ndarray,
        bundle: StageScalars,
        inverse_propagator: np.ndarray,
    ) -> np.ndarray:
        """Apply transverse FFTs before bounded temporal response transforms.

        Parameters
        ----------
        lab : numpy.ndarray
            Lab-frame stage spectrum, including the batch axis.
        bundle : solprop.translate.stage.StageScalars
            Response scalars at the stage position.
        inverse_propagator : numpy.ndarray
            Interaction-picture inverse factor.

        Returns
        -------
        numpy.ndarray
            Stage derivative in the interaction picture.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        if geometry is None:
            msg = "Cartesian nonlinear evaluation requires spatial geometry"
            raise BackendError(msg)
        shape = (geometry.shape[1], geometry.shape[0], *lab.shape[2:])
        result = np.zeros_like(lab)
        if len(responses.response_order) == 1 and not self._in_stage_absorber:
            return result
        scratch = np.empty_like(lab[0])
        if len(responses.response_order) > 1:
            transforms.ifft2(
                lab[0].reshape(shape), axes=(0, 1), out=scratch.reshape(shape)
            )
            raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            for first in range(0, transform.n_nodes, transform.node_chunk):
                rows = slice(first, first + transform.node_chunk)
                field_t = grids.pad_scale * passes.to_time(
                    scratch[rows], grids.crop_index, grids.n_to, kind=self._kind
                )
                sources = self._column_sources(field_t, bundle, raman)
                polarisation_t = sources.polarisation
                scratch[rows] = passes.to_frequency(
                    polarisation_t, grids.crop_index, kind=self._kind
                )
                if sources.current is not None:
                    scratch[rows] += self._held_current(sources.current)
                if responses.envelope_plasma is not None:
                    drive, loss = self._envelope_sources(field_t, bundle)
                    factors = responses.envelope_plasma.factors
                    scratch[rows] += self._rounded(factors.drude) * passes.to_frequency(
                        drive, grids.crop_index, kind=self._kind
                    )
                    scratch[rows] += self._rounded(factors.loss) * passes.to_frequency(
                        loss, grids.crop_index, kind=self._kind
                    )
            view = scratch.reshape(shape)
            transforms.fft2(view, axes=(0, 1), out=view)
            data = problem.linear.separable
            assert data is not None
            for first in range(0, transform.n_nodes, transform.node_chunk):
                rows = slice(first, first + transform.node_chunk)
                _, prefactor = separable_values(data, rows)
                result[0, rows] = (
                    self._rounded(prefactor) * self._tables.omega_window * scratch[rows]
                )
        if self._in_stage_absorber:
            # The rate is additive to the normalised material term. Reusing the
            # transverse scratch avoids a second retained physical spectral volume.
            transforms.ifft2(
                lab[0].reshape(shape), axes=(0, 1), out=scratch.reshape(shape)
            )
            temporal = self._tables.temporal_rate
            timed = temporal is not None and responses.absorber_scheme == "nonlinear"
            for first in range(0, transform.n_nodes, transform.node_chunk):
                rows = slice(first, first + transform.node_chunk)
                rate = self._rounded(geometry.absorber_rate[rows, None, None])
                if not timed:
                    # A rate of the node alone commutes with the temporal transforms:
                    # a spatial-only pass needs no temporal pair (SEP 40 section 4.7).
                    scratch[rows] = -0.5 * rate * scratch[rows]
                    continue
                field_t = passes.to_time(
                    scratch[rows], grids.crop_index, grids.n_to, kind=self._kind
                )
                scratch[rows] = passes.to_frequency(
                    -0.5 * (rate + temporal) * field_t,
                    grids.crop_index,
                    kind=self._kind,
                )
            view = scratch.reshape(shape)
            transforms.fft2(view, axes=(0, 1), out=view)
            result[0] += grids.active_mask * scratch
        return result * inverse_propagator

    def _nonlinear_slab(
        self,
        lab: np.ndarray,
        bundle: StageScalars,
        inverse_propagator: np.ndarray,
    ) -> np.ndarray:
        """Evaluate one hybrid planar stage: slab synthesis in x, an FFT in y.

        The ``(p, k_y)`` columns go to ``(p, y)`` rows by the inverse DFT in y, and
        the rows to time -- a row is a slab amplitude, whose time signal is a real
        field's on a field grid -- so that, as on the modal route, only ``n_p * n_y``
        temporal transforms are taken each way. In time, the slab synthesis (which
        carries the pad scale) gives the field at the ``(x, y)`` nodes, the responses
        are the Cartesian ones node by node, and the slab rule's dual basis projects
        the polarisation back onto the rows; the rows return to frequency and to
        ``(p, k_y)``, and each column takes its own prefactor from the separable leaf
        (kernel schema 15). The y collar's in-stage rate depends on y alone, so it
        acts on the rows directly.

        Parameters
        ----------
        lab : numpy.ndarray
            Lab-frame stage spectrum, including the batch axis.
        bundle : solprop.translate.stage.StageScalars
            Response scalars at the stage position.
        inverse_propagator : numpy.ndarray
            Interaction-picture inverse factor.

        Returns
        -------
        numpy.ndarray
            Stage derivative in the interaction picture.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        assert geometry is not None
        result = np.zeros_like(lab)
        if len(responses.response_order) == 1 and not self._in_stage_absorber:
            return result
        n_x, n_y = geometry.shape
        n_p = transform.synthesis.shape[1]
        rows_f = slab_rows(transform, lab[0])
        temporal = responses.temporal_absorber_fine
        timed = temporal is not None and responses.absorber_scheme == "nonlinear"
        rows_t = (
            passes.to_time(rows_f, grids.crop_index, grids.n_to, kind=self._kind)
            if len(responses.response_order) > 1 or timed
            else None
        )
        if len(responses.response_order) > 1:
            assert rows_t is not None
            raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
            envelope = responses.envelope_plasma is not None
            plasma = bool(responses.plasma_channels) and not self._gated_plasma
            projected = np.zeros_like(rows_t)
            # The envelope plasma's Drude drive and loss current, projected as the
            # polarisation is; empty unless the response is present.
            drives = [
                np.zeros_like(rows_t) for _ in range(2 if envelope or plasma else 0)
            ]
            # Whole y rows at a time, as many as the node chunk holds.
            block = max(1, transform.node_chunk // n_x)
            for first in range(0, n_y, block):
                rows = slice(first * n_p, min(first + block, n_y) * n_p)
                count = (rows.stop - rows.start) // n_p
                field_t = np.matmul(
                    transform.synthesis, rows_t[rows].reshape(count, n_p, -1)
                ).reshape(count * n_x, *rows_t.shape[1:])
                sources = self._column_sources(field_t, bundle, raman)
                current = sources.current
                source_rows = [sources.polarisation]
                if envelope:
                    assert responses.envelope_plasma is not None
                    source_rows.extend(
                        envelope_sources(
                            field_t,
                            responses.envelope_plasma,
                            bundle,
                            grids.dt_fine,
                            grids.t_window_fine,
                        )
                    )
                if current is not None:
                    source_rows.extend(
                        (
                            grids.t_window_fine * current,
                            grids.t_window_fine_derivative * current,
                        )
                    )
                for target, source in zip(
                    (projected, *drives), source_rows, strict=True
                ):
                    target[rows] = np.matmul(
                        transform.projection, source.reshape(count, n_x, -1)
                    ).reshape(count * n_p, *rows_t.shape[1:])
            spectrum = passes.to_frequency(projected, grids.crop_index, kind=self._kind)
            if envelope:
                assert responses.envelope_plasma is not None
                factors = responses.envelope_plasma.factors
                spectrum += factors.drude * passes.to_frequency(
                    drives[0], grids.crop_index, kind=self._kind
                )
                spectrum += factors.loss * passes.to_frequency(
                    drives[1], grids.crop_index, kind=self._kind
                )
            if plasma:
                spectrum += self._held_streams(drives[0], drives[1])
            columns = slab_columns(transform, spectrum)
            data = problem.linear.separable
            assert data is not None
            for first in range(0, transform.n_modes, transform.node_chunk):
                rows = slice(first, first + transform.node_chunk)
                _, prefactor = separable_values(data, rows)
                result[0, rows] = prefactor * grids.omega_window * columns[rows]
        if self._in_stage_absorber:
            result[0] += self._slab_absorber(rows_f, rows_t)
        return result * inverse_propagator

    def _slab_absorber(
        self,
        rows_f: Complex[np.ndarray, "n_modes n_pol n_omega"],
        rows_t: Inexact[np.ndarray, "n_modes n_pol n_to"] | None,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"]:
        """Return the planar in-stage collar's term on the ``(p, k_y)`` columns.

        The collar's rate depends on y alone, so it acts on the ``(p, y)`` rows. With
        no temporal rate in-stage it commutes with the temporal transforms, and the
        pass needs no temporal pair (SEP 40 section 4.7).

        Parameters
        ----------
        rows_f : numpy.ndarray
            The stage's rows in frequency.
        rows_t : numpy.ndarray or None
            The same rows in time; required when a temporal rate applies in-stage.

        Returns
        -------
        numpy.ndarray
            The term, masked to the active band.
        """
        problem = self._problem
        transform, grids, responses = (
            problem.transform,
            problem.grids,
            problem.responses,
        )
        geometry = transform.geometry
        assert geometry is not None
        n_x, n_y = geometry.shape
        n_p = transform.synthesis.shape[1]
        rate = np.repeat(geometry.absorber_rate.reshape(n_y, n_x)[:, 0], n_p)[
            :, None, None
        ]
        temporal = responses.temporal_absorber_fine
        if temporal is not None and responses.absorber_scheme == "nonlinear":
            assert rows_t is not None
            term = passes.to_frequency(
                -0.5 * (rate + temporal) * rows_t, grids.crop_index, kind=self._kind
            )
        else:
            term = -0.5 * rate * rows_f
        return grids.active_mask * slab_columns(transform, term)

    def _absorber_term(
        self,
        modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
        *,
        may_quiet: bool = False,
    ) -> Complex[np.ndarray, "n_modes n_pol n_omega"] | None:
        """Return the in-stage absorber's contribution, or ``None`` in the other scheme.

        On the modal stage (`may_quiet`) a temporal collar alone is left out while the
        boundary monitor finds it quiet (SEP 21): the term is then below
        :data:`solprop.monitor.QUIET`, and its transform is the stage's largest cost
        after the response's.

        Parameters
        ----------
        modal_t : numpy.ndarray
            The lab-frame modal field on the fine time grid.
        may_quiet : bool, optional
            Whether this stage honours the monitor's quiet collar.

        Returns
        -------
        numpy.ndarray or None
            The absorber's term on the coarse grid, or ``None`` when the
            ``"split_step"`` scheme is active and the rate is applied to the state
            after the step instead.
        """
        rate = self._tables.temporal_rate
        if not self._in_stage_absorber:
            return None
        if may_quiet and self._collar_quiet and not self._spatial_absorber.size:
            return None
        grids = self._problem.grids
        term = np.zeros_like(modal_t)
        if rate is not None and self._problem.responses.absorber_scheme == "nonlinear":
            term = -0.5 * rate * modal_t
        if self._spatial_absorber.size:
            term = term + (
                self._spatial_absorber @ modal_t.reshape(modal_t.shape[0], -1)
            ).reshape(modal_t.shape)
        return (
            passes.to_frequency(term, grids.crop_index, kind=self._kind)
            * grids.active_mask
        )

    def _over_nodes(
        self,
        modal_t: Inexact[np.ndarray, "n_modes n_pol n_to"],
        bundle: StageScalars,
    ) -> tuple[
        Num[np.ndarray, "n_modes n_pol n_to"],
        Num[np.ndarray, "n_modes_c n_pol_c n_to_c"] | None,
    ]:
        """Run the response pass over the quadrature in chunks and project it back.

        The chunk loop of ``architecture.md`` section 13: the space-domain buffers are
        sized by ``transform.node_chunk`` and not by the node count, so a chunk of one
        and a chunk of every node differ only in round-off. The embedded coarse rule is
        projected in the same loop, from the same node polarisation, because that is
        what makes the quadrature alarm free: the two rules share every node the coarse
        one has.

        Parameters
        ----------
        modal_t : numpy.ndarray
            The modal field on the fine time grid, shape ``(n_modes, n_pol, n_to)``.
        bundle : solprop.translate.stage.StageScalars
            The stage's host scalars.

        Returns
        -------
        projected : numpy.ndarray
            The modal polarisation on the fine time grid, shape
            ``(n_modes, n_pol, n_to)``.
        coarse : numpy.ndarray or None
            The same under the embedded coarse rule, or ``None`` when the quadrature
            has no embedded pair and the estimate would be identically zero.
        """
        responses, grids = self._problem.responses, self._problem.grids
        transform = self._problem.transform
        taper = bundle.taper_scale
        # Once per stage rather than once per chunk: the kernel spectrum does not depend
        # on the nodes, and along a gradient it is one real transform of 2 n_to against
        # n_modes n_pol transforms of n_to that the stage costs anyway.
        raman = self._raman_spectrum(bundle.raman_rates, bundle.raman_densities)
        n_pol = transform.n_pol
        projected = np.zeros(
            (transform.n_modes, self._problem.shape[2], grids.n_to),
            dtype=modal_t.dtype,
        )
        coarse = None if self._embedded is None else np.zeros_like(projected)
        envelope = self._problem.responses.envelope_plasma
        envelope_projected = (
            None
            if envelope is None
            else (
                np.zeros_like(projected),
                np.zeros_like(projected),
                None if coarse is None else np.zeros_like(coarse),
                None if coarse is None else np.zeros_like(coarse),
            )
        )
        plasma_projected = (
            (
                np.zeros_like(projected),
                np.zeros_like(projected),
                None if coarse is None else np.zeros_like(coarse),
                None if coarse is None else np.zeros_like(coarse),
            )
            if bool(responses.plasma_channels) and not self._gated_plasma
            else None
        )
        for first in range(0, transform.n_nodes, transform.node_chunk):
            rows = slice(first * n_pol, (first + transform.node_chunk) * n_pol)
            field_t = passes.synthesise(
                modal_t, self._tables.synthesis[rows], taper, n_pol
            )
            sources = self._column_sources(field_t, bundle, raman)
            polarisation_t = sources.polarisation
            projected = projected + passes.project(
                polarisation_t, self._tables.projection[:, rows], taper, n_pol
            )
            if coarse is not None and self._embedded is not None:
                coarse = coarse + passes.project(
                    polarisation_t, self._embedded[:, rows], taper, n_pol
                )
            if sources.current is not None and plasma_projected is not None:
                gated, edge, coarse_gated, coarse_edge = plasma_projected
                for target, reduced, window in (
                    (gated, coarse_gated, self._tables.t_window),
                    (edge, coarse_edge, self._tables.t_window_derivative),
                ):
                    current = window * sources.current
                    target[...] += passes.project(
                        current, self._tables.projection[:, rows], taper, n_pol
                    )
                    if reduced is not None and self._embedded is not None:
                        reduced[...] += passes.project(
                            current, self._embedded[:, rows], taper, n_pol
                        )
            if envelope is not None and envelope_projected is not None:
                drive, loss = self._envelope_sources(field_t, bundle)
                full_drive, full_loss, coarse_drive, coarse_loss = envelope_projected
                full_drive += passes.project(
                    drive, self._tables.projection[:, rows], taper, n_pol
                )
                full_loss += passes.project(
                    loss, self._tables.projection[:, rows], taper, n_pol
                )
                if (
                    self._embedded is not None
                    and coarse_drive is not None
                    and coarse_loss is not None
                ):
                    coarse_drive += passes.project(
                        drive, self._embedded[:, rows], taper, n_pol
                    )
                    coarse_loss += passes.project(
                        loss, self._embedded[:, rows], taper, n_pol
                    )
        self._last_plasma = plasma_projected
        self._last_envelope = envelope_projected
        return projected, coarse

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
            if self._problem.transform.n_pol * self._problem.shape[2] == 2:
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

    def _exponents(
        self, z: float, dz: float
    ) -> Callable[
        [float, StageScalars], Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]
    ]:
        """Return the memoised ``dPhi`` of a step, keyed by the stage fraction.

        The anchor is hoisted once per step and each exponent is evaluated once and
        negated for the inverse, never evaluated forward and backward independently
        (``docs/dev/stepper.md`` section 4). ``propagator_mode`` decides whether the
        exponents are kept for the whole step or evaluated per stage; the exponent
        itself is one linear combination of the terms over the bundle's weights,
        whatever the representation (:mod:`solprop.translate.stage`).

        Parameters
        ----------
        z : float
            Position at the start of the step, m.
        dz : float
            Length of the step, m.

        Returns
        -------
        callable
            ``(fraction, bundle) -> dPhi``, of the state's broadcast shape.
        """
        del z, dz
        if self._propagator_mode == "recompute":
            return lambda fraction, bundle: self._delta_phi(bundle)  # noqa: ARG005
        cache: dict[float, Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]] = {}

        def exponent(
            fraction: float, bundle: StageScalars
        ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
            """Return ``dPhi`` at a stage fraction, computing it at most once.

            Parameters
            ----------
            fraction : float
                The tableau node ``c_i``, dimensionless, in ``[0, 1]``.
            bundle : solprop.translate.stage.StageScalars
                The stage's host scalars, whose weights the terms are combined over.

            Returns
            -------
            numpy.ndarray
                ``Phi(z + c_i h) - Phi(z)``, dimensionless.
            """
            if fraction not in cache:
                cache[fraction] = self._delta_phi(bundle)
            return cache[fraction]

        return exponent


@dataclass(frozen=True)
class SourceTables:
    """The problem's tables a source evaluation multiplies by.

    Read through this record rather than off the problem so that the single-precision
    path can hold them narrowed (SEP 33 section 4.2) while the passes stay the same.

    Attributes
    ----------
    synthesis : numpy.ndarray
        ``transform.synthesis``.
    projection : numpy.ndarray
        ``transform.projection``.
    t_window : numpy.ndarray
        ``grids.t_window_fine``.
    t_window_derivative : numpy.ndarray
        ``grids.t_window_fine_derivative``.
    omega_window : numpy.ndarray
        ``grids.omega_window``.
    temporal_rate : numpy.ndarray or None
        ``responses.temporal_absorber_fine``.
    """

    synthesis: Float[np.ndarray, "n_rows n_modes_s"]
    projection: Float[np.ndarray, "n_modes n_columns"]
    t_window: Float[np.ndarray, " n_to"]
    t_window_derivative: Float[np.ndarray, " n_to"]
    omega_window: Float[np.ndarray, " n_omega"]
    temporal_rate: Float[np.ndarray, " n_to"] | None

    @classmethod
    def of(cls, problem: KernelProblem) -> "SourceTables":  # noqa: UP037 - the runtime type hook resolves this while the class body runs
        """Return the problem's own tables.

        Parameters
        ----------
        problem : solprop.translate.problem.KernelProblem
            The problem.

        Returns
        -------
        SourceTables
            References to its arrays, uncopied.
        """
        grids = problem.grids
        return cls(
            problem.transform.synthesis,
            problem.transform.projection,
            grids.t_window_fine,
            grids.t_window_fine_derivative,
            grids.omega_window,
            problem.responses.temporal_absorber_fine,
        )


def _refuse_adams(problem: KernelProblem) -> None:
    """Refuse what the Adams march does not execute (SEP 13 sections 2 and 6.2).

    Its history is of source samples along a diagonal flow, so a coupled flow is
    refused by name rather than stepped with a stale history, and the exponential
    formulation refuses a varying operator until SEP 16's phase 4
    (:func:`solprop.backends.exponential.refuse_formulation`). The ``modified`` noise is
    admitted for both formulations (SEP 16 section 4.4): its draw is fixed for the run
    and added to each sample's field at that sample's own position, never to the
    propagating state, so conditional on the draw the source is a deterministic,
    nonautonomous function of position that the history interpolates. The split-step
    temporal collar is admitted: it changes
    the state only inside the collar, where its first-order splitting error already
    lies (``docs/dev/stepper.md`` section 6), and it removes the collar's stability
    limit, which binds the Adams methods long before Runge-Kutta's (SEP 13 phase 2).

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
        return
    if (
        problem.linear.coupled_flow is not None
        and integrator.formulation != "exponential"
    ):
        # SEP 16 Q3: only the exponential formulation steps a coupled flow; the
        # interaction one would transport its history through inverse actions.
        msg = (
            f"the {integrator.name} integrator does not step a problem with a "
            "coupled linear flow; use Multistep(formulation='exponential') or the "
            "Runge-Kutta march"
        )
        raise BackendError(msg)
    refuse_formulation(problem, "numpy_ref")
    # SEP 35: the scalar shift, on the guided routes.
    refuse_carrier_frame(problem, "numpy_ref", delivered=frozenset({"scalar"}))


def _refuse_unsupported(problem: KernelProblem) -> None:
    """Refuse, by name, every problem this backend does not execute.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem to check.

    Raises
    ------
    BackendError
        Naming what is unsupported.
    """
    if problem.schema_version != KERNEL_SCHEMA_VERSION:
        msg = (
            "this backend implements kernel contract version "
            f"{KERNEL_SCHEMA_VERSION} and the problem declares version "
            f"{problem.schema_version}"
        )
        raise BackendError(msg)
    refuse_general_leaves(problem, "numpy_ref", admits=_GENERAL_LEAVES)
    _refuse_adams(problem)
    # One list for every backend: the coupled flow first, then the
    # general modal route, each refusal naming this backend.
    refuse_general_route(problem, backend="numpy_ref", moving_maps=True)
    if (
        problem.flags.has_spectral_responses
        and problem.responses.envelope_plasma is None
    ):
        msg = (
            "numpy_ref executes the Kerr, Raman and plasma terms and this problem "
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
    if n_batch != NumpyRefBackend.max_batch:
        msg = (
            f"numpy_ref steps one problem at a time (max_batch = "
            f"{NumpyRefBackend.max_batch}) and this problem has a batch axis of "
            f"{n_batch}"
        )
        raise BackendError(msg)


_GENERAL_LEDGER_ROWS = (
    "general_synthesis",
    "general_projection",
    "general_stage_maps",
    "general_tensor",
    "general_electric",
    "general_magnetic",
    "general_backward_electric",
    "general_backward_magnetic",
    "general_reciprocal_gram",
    "general_coordinate_dual",
    "general_scaled_state",
    "general_node_spectrum",
    "general_node_field",
    "general_node_pair",
    "general_node_weighted",
    "general_node_polarisation",
    "general_node_source",
    "general_node_drive",
    "general_node_histories",
    "general_node_drive_spectrum",
    "general_node_delayed",
    "general_delayed_tensor",
    "general_delayed_spectrum",
)
"""The ledger rows of the general modal route, named even when they are empty.

``memory_ledger`` names every buffer on every route, so a problem that is not a
general modal guide reports these as zero rather than omitting them
(:func:`solprop.backends.numpy_ref.general_modal.general_memory` fills them).
"""


def _refuse_inadmissible(problem: KernelProblem, spec: StatsSpec) -> None:
    """Refuse a statistic this problem's basis does not define, by name.

    The second door of the accessor whitelist (:mod:`solprop.output.capability`),
    after the run loop's own admission: a caller may build a backend and ask it for
    statistics directly, and a general guide's static synthesis matrix is a checked
    **zero** placeholder, so an on-axis intensity taken from one would be zero
    rather than obviously wrong.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem this backend was built from.
    spec : solprop.stats.StatsSpec
        The statistics asked for.

    Raises
    ------
    BackendError
        Naming the statistic and the basis, with the replacement to use.
    """
    refusal = refusal_for(
        declared_capability(
            general_modal=problem.transform.modal_tables is not None,
            spatial=problem.transform.geometry is not None,
        ),
        spec.names,
        withheld=withheld_accessors(
            general_modal=problem.transform.modal_tables is not None,
            coupled_flow=problem.linear.coupled_flow is not None,
        ),
    )
    if refusal is not None:
        raise BackendError(refusal)
    refusal = axis_refusal(problem, spec.names)
    if refusal is not None:
        raise BackendError(refusal)


def _general_maps(problem: KernelProblem) -> GeneralMaps | None:
    """Derive the general modal route's maps, or return ``None`` for another route.

    The maps are read out of the leaves once, here, because the route propagates
    constant geometry: one interval of one Chebyshev degree along z, so there is
    nothing to evaluate per stage. The chunk is the backend's own memory strategy
    and not a contract field -- ``transform.node_chunk`` is the checked placeholder
    of the static matrices this route replaces.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        A validated problem this backend has already admitted.

    Returns
    -------
    GeneralMaps or None
        The derived maps, or ``None`` when the problem carries no modal tables.
    """
    tables = problem.transform.modal_tables
    if tables is None:
        return None
    assert tables.synthesis is not None and tables.projection is not None
    chunk = general_node_chunk(
        int(tables.electric.shape[2]),
        problem.grids.n_to,
        int(tables.omega_fine.size),
        DEFAULT_NODE_BUDGET,
        dispersive=general_dispersive_buffers(problem.responses.dispersive_kerr),
    )
    regional = problem.responses.regional_responses
    owner, delayed_tensor, delayed_spectrum = node_delayed_maps(tables, regional)
    moving = tables.synthesis.shape[0] * tables.synthesis.shape[1] > 1
    maps = GeneralMaps(
        tables.synthesis[0, 0],
        tables.projection[0, 0],
        node_kerr_tensor(tables, regional),
        tables.source_frequency_factor,
        min(chunk, int(tables.region_index.size)),
        node_kerr_form(regional),
        owner,
        delayed_tensor,
        delayed_spectrum,
        # A moving guide's coefficients, flattened over (n_intervals, n_deg) in the
        # order the stage bundle's transform_weights are.
        tables.synthesis.reshape(-1, *tables.synthesis.shape[2:]) if moving else None,
        tables.projection.reshape(-1, *tables.projection.shape[2:]) if moving else None,
        problem.responses.dispersive_kerr,
        None if problem.responses.dispersive_kerr is None else tables.region_index,
    )
    logger.debug(
        "numpy_ref general modal route: %d channels on %d nodes, chunk %d, "
        "%d fine bins, gauge %s, %s cubic, %d delayed term(s)",
        maps.n_modes,
        maps.n_nodes,
        maps.node_chunk,
        int(tables.omega_fine.size),
        tables.gauge_id,
        maps.form,
        maps.n_terms,
    )
    return maps


@dataclass(frozen=True)
class _CoupledFlow:
    """What ``numpy_ref`` holds to execute one coherent block, frozen per section.

    The Schur factors are the backend's own: they are computed here from the leaf's
    generator rather than carried across the seam, because a decomposition is not a
    contract row and no schema has one for it. The
    **section** axis leads them and the frequency axis follows, and only the bins the
    generator is non-zero at are held: the rest of the band has an identity action and
    the state is masked to zero there anyway.

    A ``"constant_schur"`` leaf is the one-section case with no residual at all, and
    goes through exactly the constant-section arithmetic. A ``"frozen_residual"`` leaf
    adds `residual_terms`: the coefficients of ``G(z) - C(z)`` along z, flattened over
    ``(n_intervals, n_deg)`` in that order, so that the stage bundle's
    ``flow_weights`` combine them into ``G(z_1) - C(z_1)`` exactly as ``phi_delta``
    combines the terms of ``dPhi`` (``docs/dev/kernel-contract.md`` section 1). The
    explicit residual is that combination minus the frozen block of the step's own
    section. It is applied only on `residual_bins`, the active bins of the requested
    band; in the spectral collar outside them the frozen flow alone carries the state
    (``docs/guide/waveguides/general-modal.md``).

    Attributes
    ----------
    active : numpy.ndarray
        Which bins carry a block, bool, shape ``(n_omega,)``.
    residual_bins : numpy.ndarray
        Which bins the explicit residual is applied at, bool, shape ``(n_omega,)``:
        `active` and the problem's ``grids.inband_mask``, the requested band: in the
        collar beyond it the wall model is outside its validity, and the residual is
        left out whether or not the error norm scores there.
    vectors, triangular : numpy.ndarray
        ``Z`` and ``T`` of ``G_s = Z T Z^H`` per section, shape
        ``(n_sections, n_active, n_modes, n_modes)``.
    frozen : numpy.ndarray
        The frozen generator of each section in 1/m, same layout.
    condition_bound : numpy.ndarray
        The leaf's declared ``norm(U) norm(U^-1)`` per section and active bin,
        dimensionless, shape ``(n_sections, n_active)``.
    residual_terms : numpy.ndarray or None
        The coefficients of ``G(z) - C(z)`` in 1/m, shape
        ``(n_intervals * n_deg, n_active, n_modes, n_modes)``; ``None`` for a leaf with
        no explicit residual. A ``"cf4"`` section exponentiates the same combination
        at its Gauss nodes.
    n_modes : int
        Channels the block couples.
    cf4 : numpy.ndarray
        Which sections are ``"cf4"``, bool, shape ``(n_sections,)``.
    leaf : solprop.translate.coupled.CoupledFlowTables
        The leaf itself, whose edges and degree axis a CF4 node's weights are built
        on (:func:`solprop.translate.coupled.flow_weights`).
    """

    active: Bool[np.ndarray, " n_omega_flow"]
    residual_bins: Bool[np.ndarray, " n_omega_scored"]
    vectors: Complex[np.ndarray, "n_sections_v n_active_v n_modes_v_out n_modes_v_in"]
    triangular: Complex[
        np.ndarray, "n_sections_t n_active_t n_modes_t_out n_modes_t_in"
    ]
    frozen: Complex[np.ndarray, "n_sections_f n_active_f n_modes_f_out n_modes_f_in"]
    condition_bound: Float[np.ndarray, "n_sections_c n_active_c"]
    residual_terms: (
        Complex[np.ndarray, "n_terms_r n_active_r n_modes_r_out n_modes_r_in"] | None
    )
    n_modes: int
    cf4: Bool[np.ndarray, " n_sections_cf4"]
    leaf: CoupledFlowTables


_TAYLOR_ORDER = 16
"""Degree of the Taylor series a scaled triangular exponential is summed to.

Scaling brings the row-sum norm below :data:`_TAYLOR_SCALE`, so the truncated tail is
at most ``_TAYLOR_SCALE**(n+1)/(n+1)!``, which is 2.1e-20 at this degree -- below the
rounding of the squarings that follow it and of the two unitary products around it.
The alternative, a Pade approximant, buys a lower degree at the cost of a solve per
frequency; the oracle takes the series (``docs/dev/architecture.md`` section 3:
physics and reference code carry no performance tricks).
"""

_TAYLOR_SCALE = 0.5
"""Row-sum norm a block is scaled below before the series is summed."""

_MAX_FLOW_SQUARINGS = 30
"""Largest number of squarings a flow action may need before it is refused.

``2**30`` times the scale is a step whose own generator spans 21 orders of magnitude
in the exponent; a problem that needs it has not been certified by anything the
translation layer measured.
"""


def _flow_squarings(largest: float, *, backend: str = "numpy_ref") -> int:
    """Return how many squarings an exponent of this row-sum norm needs, or refuse.

    Parameters
    ----------
    largest : float
        The largest row-sum norm of the exponent over the stack, dimensionless.
    backend : str, optional
        The executing backend's registry name, used in the refusal: the compiled
        backends refuse a step on this rule too, whichever route forms their actions.

    Returns
    -------
    int
        The number of squarings that brings the norm below :data:`_TAYLOR_SCALE`.

    Raises
    ------
    BackendError
        If it exceeds :data:`_MAX_FLOW_SQUARINGS`.
    """
    squarings = 0
    if largest > _TAYLOR_SCALE:
        squarings = math.ceil(math.log2(largest / _TAYLOR_SCALE))
    if squarings > _MAX_FLOW_SQUARINGS:
        msg = (
            f"{backend} cannot take a coupled flow action whose exponent has row-sum "
            f"norm {largest:.4g}: it would need {squarings} squarings, above the "
            f"{_MAX_FLOW_SQUARINGS} this backend admits. Cap dz_max"
        )
        raise BackendError(msg)
    return squarings


def _triangular_exponential(
    blocks: Complex[np.ndarray, "n_active_e n_modes_e_out n_modes_e_in"],
    *,
    backend: str = "numpy_ref",
) -> Complex[np.ndarray, "n_active_x n_modes_x_out n_modes_x_in"]:
    """Return ``exp(A)`` for a stack of blocks, by scaling, a series and squaring.

    Parameters
    ----------
    blocks : numpy.ndarray
        The stack to exponentiate, shape ``(n_active, n_modes, n_modes)``.
    backend : str, optional
        The executing backend's registry name, used in the refusal: ``numba_cpu``
        forms its actions with this function on the host.

    Returns
    -------
    numpy.ndarray
        ``exp(A)`` in the same layout.

    Raises
    ------
    BackendError
        If a block needs more than :data:`_MAX_FLOW_SQUARINGS` squarings.
    """
    largest = float(np.max(np.sum(abs(blocks), axis=-1), initial=0.0))
    squarings = _flow_squarings(largest, backend=backend)
    scaled = blocks / float(2**squarings)
    identity = np.broadcast_to(
        np.eye(blocks.shape[1], dtype=np.complex128), blocks.shape
    )
    series = identity.copy()
    for term in range(_TAYLOR_ORDER, 0, -1):
        series = identity + scaled @ series / term
    for _ in range(squarings):
        series = series @ series
    return series


def _flow_action(
    flow: _CoupledFlow, section: int, length: float, *, backend: str = "numpy_ref"
) -> tuple[
    Complex[np.ndarray, "n_active_f n_modes_f_out n_modes_f_in"],
    Complex[np.ndarray, "n_active_i n_modes_i_out n_modes_i_in"],
]:
    """Return the forward and inverse flow actions of one section over one length.

    ``U = Z exp(l T) Z^H`` and ``U^-1 = Z exp(-l T) Z^H``: the inverse is the matrix
    function of the negated exponent, never a numerical inverse of ``U``, so the
    interaction picture's backward action is as accurate as its forward one and does
    not inherit the conditioning of a solve (``docs/dev/stepper.md`` section 4: no
    unbounded inverse propagation).

    Parameters
    ----------
    flow : _CoupledFlow
        The held Schur factors.
    section : int
        Which section's frozen block the step is anchored in.
    length : float
        Distance in m, positive for the forward action.
    backend : str, optional
        The executing backend's registry name, used in a refusal.

    Returns
    -------
    tuple of numpy.ndarray
        The two actions, shape ``(n_active, n_modes, n_modes)``.
    """
    vectors = flow.vectors[section]
    exponent = float(length) * flow.triangular[section]
    adjoint = np.conj(np.swapaxes(vectors, -1, -2))
    forward = vectors @ _triangular_exponential(exponent, backend=backend) @ adjoint
    inverse = vectors @ _triangular_exponential(-exponent, backend=backend) @ adjoint
    return forward, inverse


def _cf4_action(
    flow: _CoupledFlow, z: float, length: float, section: int
) -> tuple[
    Complex[np.ndarray, "n_active_c4f n_modes_c4f_out n_modes_c4f_in"],
    Complex[np.ndarray, "n_active_c4i n_modes_c4i_out n_modes_c4i_in"],
    Float[np.ndarray, " n_active_c4g"],
]:
    """Return the CF4 forward and inverse actions over ``[z, z + length]``.

    ``V = exp(B_2) exp(B_1)`` with ``B_1 = l (b A_1 + a A_2)`` acting first and
    ``B_2 = l (a A_1 + b A_2)`` second, ``A_j`` the whole generator ``G - C`` at the
    Gauss nodes ``z + c_j l`` (:data:`solprop.translate.coupled.CF4_NODES`, ``a, b``
    :data:`solprop.translate.coupled.CF4_WEIGHTS`). The inverse is
    ``exp(-B_1) exp(-B_2)``: the exact inverse of the approximation, formed from the
    negated exponents and never by a solve, so the interaction picture's forward and
    backward actions are inverse to rounding and a zero source reproduces ``V``
    exactly.

    The generator at a node is the stage bundle's own evaluation form, applied at a
    position that is not a stage: ``flow_weights`` of the node, resolved against the
    step's section, combined over the held terms. Every node lies strictly inside the
    step, so no edge rule is ever exercised by it.

    Parameters
    ----------
    flow : _CoupledFlow
        The held terms and the leaf.
    z : float
        Where the step starts, m.
    length : float
        The stage's own length ``c_i h``, m, positive.
    section : int
        The step's section.

    Returns
    -------
    tuple
        The two actions, shape ``(n_active, n_modes, n_modes)``, and the product of
        the four factors' spectral norms per active bin -- the quantity the CF4
        factor-norm guard is declared on.
    """
    assert flow.residual_terms is not None
    early, late = (
        combine(
            flow_weights(flow.leaf, float(z) + node * float(length), section),
            flow.residual_terms,
        )
        for node in CF4_NODES
    )
    small, large = CF4_WEIGHTS
    first = float(length) * (large * early + small * late)
    second = float(length) * (small * early + large * late)
    factors = (
        _triangular_exponential(first),
        _triangular_exponential(second),
        _triangular_exponential(-first),
        _triangular_exponential(-second),
    )
    growth = np.prod(
        [np.linalg.norm(factor, ord=2, axis=(-2, -1)) for factor in factors], axis=0
    )
    return factors[1] @ factors[0], factors[2] @ factors[3], growth


def _check_cf4_growth(
    flow: _CoupledFlow,
    section: int,
    growth: Float[np.ndarray, " n_active_g"],
    length: float,
) -> None:
    """Refuse a CF4 action whose internal factors exceed the declared pair bound.

    The CF4 factor-norm guard at the executor. The leaf declares, for a ``"cf4"``
    section, ``exp((2/sqrt(3)) dz_max (m_hi - m_lo))``: a bound on the product of the
    four factor norms of any forward and inverse pair a step anchored there can form
    (:func:`solprop.translate.coupled.cf4_flow`). Here the four norms are **measured**,
    from the factors actually formed, and compared with it, so a certificate declared
    over too short a step or a generator it was not built from is caught per step rather
    than trusted. It bounds the forward product's excess over the exact action and the
    inverse action the source is carried back by, together.

    Parameters
    ----------
    flow : _CoupledFlow
        The held declaration.
    section : int
        The step's section.
    growth : numpy.ndarray
        The measured product per active bin, dimensionless.
    length : float
        The stage length it was formed over, m, used in the refusal.

    Raises
    ------
    BackendError
        If the measured product exceeds the declared bound at any bin.
    """
    declared = flow.condition_bound[section]
    # The same allowance as the frozen check: the bound is on the exact factors, and
    # what exceeds it by less than the rounding of four spectral norms is arithmetic.
    allowance = 1.0 + 64.0 * flow.n_modes * float(np.finfo(float).eps)
    excess = growth / (declared * allowance)
    if np.any(excess > 1.0):
        worst = int(np.argmax(excess))
        msg = (
            f"numpy_ref measured a CF4 factor-norm product of "
            f"{float(growth[worst]):.12g} over a stage of {length:.4g} m, above the "
            f"{float(declared[worst]):.12g} the certificate declares at active bin "
            f"{worst} of section {section}: the negative CF4 weight lets a factor of a "
            "dissipative flow grow beyond the exact action. Lower "
            "dz_max, which is what the bound was declared over"
        )
        raise BackendError(msg)


def _residual_action(
    flow: _CoupledFlow, section: int, weights: Float[np.ndarray, " n_flow_terms"]
) -> Complex[np.ndarray, "n_active_r n_modes_r_out n_modes_r_in"]:
    """Return the explicit residual of one stage, ``R_s(z_1)``, on the active bins.

    ``R_s(z) = D(z) - D(z_s) + H(z) - H(z_s) - C(z)``, evaluated as one linear
    combination of the held terms over the bundle's weights minus the frozen block of
    the **step's** section. The weights are zero outside the section containing the
    stage's own position, so no section's expansion is extrapolated; the frozen block
    is the step's, because the exponential the interaction picture is built on is, and
    freezing is a rearrangement of the same equation whichever block it names.

    Parameters
    ----------
    flow : _CoupledFlow
        The held terms and frozen blocks.
    section : int
        The step's section, from the stage bundle's ``flow_section``.
    weights : numpy.ndarray
        The stage's flow weights, float64, shape ``(n_intervals * n_deg,)``.

    Returns
    -------
    numpy.ndarray
        The residual in 1/m, shape ``(n_active, n_modes, n_modes)``.
    """
    assert flow.residual_terms is not None
    return combine(weights, flow.residual_terms) - flow.frozen[section]


class BlockResidual:
    """``R_s(z) state`` for a held flow's positions (SEP 16 phases 5 and 6).

    A history sample's residual is formed against the section of every trial it serves,
    about six; its action depends only on the section and the sample's flow weights, so
    each pair's action is formed once and kept, the least recently used dropped first:
    each trial's endpoint, at a new position, is used once.

    Parameters
    ----------
    flow : _CoupledFlow
        The held flow.
    """

    _KEPT = 64
    """Most actions kept: well above a history's samples times the sections it spans."""

    def __init__(self, flow: _CoupledFlow) -> None:
        """Hold the flow, with no action formed yet."""
        self._flow = flow
        self._actions: collections.OrderedDict[tuple[int, bytes], np.ndarray] = (
            collections.OrderedDict()
        )

    def __call__(
        self,
        section: int,
        state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
        weights: Float[np.ndarray, " n_flow_terms"],
    ) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
        """Return ``R_s(z) state`` for the flow weights of a position.

        Parameters
        ----------
        section : int
            The step's section, whose frozen block the residual is relative to.
        state : numpy.ndarray
            The state the residual acts on.
        weights : numpy.ndarray
            The flow's term weights at the position.

        Returns
        -------
        numpy.ndarray
            The residual's term, zero off the scored active bins.
        """
        key = (section, np.ascontiguousarray(weights).tobytes())
        action = self._actions.get(key)
        if action is None:
            if len(self._actions) >= self._KEPT:
                self._actions.popitem(last=False)
            action = _residual_action(self._flow, section, weights)
            self._actions[key] = action
        else:
            self._actions.move_to_end(key)
        flow = self._flow
        return _residual_drive(state, action, flow.active, flow.residual_bins)


def _residual_drive(
    lab: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    action: Complex[np.ndarray, "n_active_d n_modes_d_out n_modes_d_in"],
    active: Bool[np.ndarray, " n_omega"],
    scored: Bool[np.ndarray, " n_omega_scored"],
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Return the explicit residual's contribution to one stage's lab-frame source.

    Unlike :func:`_apply_flow`, which *transforms* a state and therefore leaves the
    bins it does not cover alone, this **adds a term** and is exactly zero off them:
    the residual is zero at every bin the wall formula was never evaluated at, and a
    copy of the state there would enter the source as a derivative of its own size.

    It is also zero on the active bins the error norm does not score, the spectral
    collar. There the residual can reach a spectral radius the
    embedded pair is unstable at, at steps the in-band error allows, and nothing
    would see the growth; the frozen flow, exact at any step, carries those bins
    alone. The block is diagonal in frequency, so no in-band bin reads them through
    it; only the nonlinear source couples them in, at the size of the collar's
    content.

    Parameters
    ----------
    lab : numpy.ndarray
        The lab-frame stage input.
    action : numpy.ndarray
        ``R_s(z_1)`` on the active bins, 1/m.
    active : numpy.ndarray
        Which bins those are, bool, shape ``(n_omega,)``.
    scored : numpy.ndarray
        Which bins the residual is applied at, bool, shape ``(n_omega,)``: the active
        bins inside the error norm's band.

    Returns
    -------
    numpy.ndarray
        ``R_s(z_1) u`` in the state's layout, zero off the scored active bins.
    """
    drive = np.zeros_like(lab)
    drive[:, :, :, scored] = np.einsum(
        "wij,bjpw->bipw", action[scored[active]], lab[:, :, :, scored]
    )
    return drive


def _apply_flow(
    state: Complex[np.ndarray, "n_batch n_modes n_pol n_omega"],
    action: Complex[np.ndarray, "n_active_a n_modes_a_out n_modes_a_in"],
    active: Bool[np.ndarray, " n_omega"],
) -> Complex[np.ndarray, "n_batch n_modes n_pol n_omega"]:
    """Apply a per-frequency channel action to the canonical state.

    Parameters
    ----------
    state : numpy.ndarray
        The state, or any array of its layout.
    action : numpy.ndarray
        The action on the active bins. The identity case is the caller's: a problem
        with no coupled flow, or the zero stage fraction, never reaches here.
    active : numpy.ndarray
        Which bins the action covers, bool, shape ``(n_omega,)``.

    Returns
    -------
    numpy.ndarray
        The state with its channel axis mixed on the active bins and untouched off
        them, where the block is zero and the action is the identity.
    """
    carried = state.copy()
    carried[:, :, :, active] = np.einsum(
        "wij,bjpw->bipw", action, state[:, :, :, active]
    )
    return carried


_GENERAL_LEAVES = (
    "linear.coupled_flow",
    "linear.modal_metric",
    "transform.modal_tables",
    "responses.regional_responses",
)
"""The schema-12 leaves ``numpy_ref`` executes.

The metric, the tables and the regional responses are the general modal route;
``coupled_flow`` is executed in the combinations
:func:`solprop.backends.numpy_ref.admission.refuse_coupled_flow` admits. Naming a leaf
here only says the field may be present; which contents are executed is the shared
list's, :func:`solprop.backends.numpy_ref.admission.refuse_general_route`, which every
backend runs under its own name.
"""


def _check_flow_conditioning(
    flow: _CoupledFlow,
    section: int,
    pair: tuple[
        Complex[np.ndarray, "n_active_p n_modes_p_out n_modes_p_in"],
        Complex[np.ndarray, "n_active_q n_modes_q_out n_modes_q_in"],
    ],
    dz: float,
    *,
    backend: str = "numpy_ref",
) -> None:
    """Refuse a step whose flow action exceeds the leaf's declared conditioning.

    The certificate row the translation layer declared is
    ``exp(dz_max (mu_max - mu_min))``, a bound on ``norm(U)_2 norm(U^-1)_2``; here the
    two spectral norms are **measured**, from the singular values of the actual
    action, and compared with it. That is the point of a certificate: the row is a
    claim the executor checks rather than a number it trusts, and a step longer than
    the ``dz_max`` the row was declared over is exactly what this catches.

    The inverse is what the interaction picture carries the nonlinear source back by,
    so an unbounded one is an unbounded amplification of that source (the
    ``norm(g(b) - V^-1 N(V b))`` inequality). Local re-anchoring is what keeps it
    bounded at all: the picture is re-anchored at every accepted step, so no inverse
    action is ever taken over more than one of them.

    Parameters
    ----------
    flow : _CoupledFlow
        The held factors and the declared row.
    section : int
        Which section the step is anchored in.
    pair : tuple of numpy.ndarray
        The forward and inverse actions over the whole step.
    dz : float
        Length of the step, m, used in the refusal.
    backend : str, optional
        The executing backend's registry name, used in the refusal: ``numba_cpu``
        checks its steps with this function.

    Raises
    ------
    BackendError
        If the measured conditioning exceeds the declared bound at any bin.
    """
    forward, inverse = pair
    if not forward.size:
        return
    measured = np.linalg.norm(forward, ord=2, axis=(-2, -1)) * np.linalg.norm(
        inverse, ord=2, axis=(-2, -1)
    )
    _refuse_flow_conditioning(flow, section, measured, dz, backend=backend)


def _refuse_flow_conditioning(
    flow: _CoupledFlow,
    section: int,
    measured: Float[np.ndarray, " n_active_m"],
    dz: float,
    *,
    backend: str = "numpy_ref",
) -> None:
    """Refuse a step whose measured ``norm(U) norm(U^-1)`` exceeds the certificate.

    The decision of :func:`_check_flow_conditioning`, on a measurement made
    elsewhere: the compiled backends measure the two spectral norms in their own
    kernels and hand the per-bin products here, so every backend
    compares them with the declared row by the same allowance and refuses in the same
    words.

    Parameters
    ----------
    flow : _CoupledFlow
        The held factors and the declared row.
    section : int
        Which section the step is anchored in.
    measured : numpy.ndarray
        ``norm(U)_2 norm(U^-1)_2`` of the whole step's actions per active bin.
    dz : float
        Length of the step, m, used in the refusal.
    backend : str, optional
        The executing backend's registry name, used in the refusal.

    Raises
    ------
    BackendError
        If the measured conditioning exceeds the declared bound at any bin.
    """
    if not measured.size:
        return
    declared = flow.condition_bound[section]
    # A bin whose block is normal has a bound of exactly one and a measured product
    # that is one to within the rounding of two spectral norms; the allowance is that
    # rounding and nothing more, because the bound is a true bound on the exact
    # action and any excess above it is arithmetic.
    allowance = 1.0 + 64.0 * flow.n_modes * float(np.finfo(float).eps)
    excess = measured / (declared * allowance)
    worst = int(np.argmax(excess))
    if excess[worst] > 1.0:
        msg = (
            # Enough digits, and the ratio itself: the two quantities differ by as
            # little as 1e-13 at the step that first exceeds the bound, and `.4g`
            # printed both as the same number with no excess to read.
            f"{backend} measured a coupled flow conditioning of "
            f"{float(measured[worst]):.12g} over a step of {dz:.4g} m, above the "
            f"{float(declared[worst]):.12g} the certificate declares at "
            f"active bin {worst} of section {section} -- exceeding it by a factor "
            f"{float(excess[worst]):.12g} of the rounding allowance: the inverse "
            "action carries the nonlinear source back by the measured factor. Lower "
            "dz_max, which is what the bound was declared over"
        )
        raise BackendError(msg)


def _coupled_flow_maps(problem: KernelProblem) -> _CoupledFlow | None:
    """Decompose the coupled flow this backend will execute, once.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, already admitted by
        :func:`solprop.backends.numpy_ref.admission.refuse_coupled_flow`.

    Returns
    -------
    _CoupledFlow or None
        The Schur factors and the certificate, or ``None`` for a problem carrying no
        coupled flow.
    """
    flow = problem.linear.coupled_flow
    if flow is None:
        return None
    assert flow.generator is not None
    sections = len(flow.section_methods)
    active = np.any(flow.generator != 0.0, axis=(0, 1, 2, 3))
    frozen = np.stack(
        [
            np.moveaxis(frozen_generator(flow, index)[:, :, active], -1, 0)
            for index in range(sections)
        ]
    )
    factors = [schur_blocks(np.moveaxis(block, 0, -1)) for block in frozen]
    residual = None
    if flow.section_methods != ("constant_schur",):
        terms = flow.generator[:, :, :, :, active]
        if flow.connection is not None:
            terms = terms - flow.connection[:, :, :, :, active]
        residual = np.ascontiguousarray(
            np.moveaxis(terms, (2, 3), (-2, -1)).reshape(
                sections * flow.n_deg, -1, flow.n_modes, flow.n_modes
            )
        )
    held = _CoupledFlow(
        active=active,
        residual_bins=active & problem.grids.inband_mask,
        vectors=np.stack([pair[0] for pair in factors]),
        triangular=np.stack([pair[1] for pair in factors]),
        frozen=frozen,
        condition_bound=np.ascontiguousarray(flow.inverse_condition_bound[:, active]),
        residual_terms=residual,
        n_modes=flow.n_modes,
        cf4=np.array([method == "cf4" for method in flow.section_methods]),
        leaf=flow,
    )
    logger.debug(
        "coupled flow: %d channel(s) on %d of %d bins over %d section(s), %s, largest "
        "declared conditioning %.3e, largest declared residual %.3e 1/m, largest "
        "declared representation error %.3e",
        held.n_modes,
        int(np.count_nonzero(active)),
        len(active),
        sections,
        "no explicit residual" if residual is None else f"{residual.shape[0]} terms",
        float(np.max(held.condition_bound, initial=1.0)),
        float(np.max(flow.residual_norm_bound)),
        float(np.max(flow.generator_error)),
    )
    return held


def _source_record(problem: KernelProblem) -> dict[str, object]:
    """Record a TDSE run's source filter and, for the gated one, its certificate.

    The steady-current certificate (SEP 26 section 4.4) is refused above its limit at
    translation. It is run metadata, never part of the descriptor or any identity.

    Parameters
    ----------
    problem : KernelProblem
        A TDSE problem.

    Returns
    -------
    dict
        ``gated``, and for a gated source the certificate, its limit
        and the taper breakpoints in s.
    """
    descriptor = problem.tdse
    assert descriptor is not None
    record: dict[str, object] = {"gated": descriptor.gated}
    if descriptor.gated:
        record |= {
            "steady_current_certificate": steady_current_certificate(
                descriptor, problem.grids.omega, problem.grids.active_mask
            ),
            "certificate_limit": STEADY_CURRENT_CERTIFICATE_LIMIT,
            "gate_breakpoints": list(descriptor.source_breakpoints()),
        }
    return record


def _harmonic_carrier(
    problem: KernelProblem,
) -> Complex[np.ndarray, " n_to"] | None:
    """Return ``exp(2 i omega_0 t)`` on the fine grid when the harmonic is radiated.

    The envelope-grid third-harmonic term needs the carrier the coarse grid does not
    resolve. Until contract version 3 the contract carried neither it nor a fine time
    axis and a backend built it from ``omega0``, ``n_to`` and ``dt_fine`` -- one piece
    of physics inside a backend, against ``docs/dev/architecture.md`` section 3, which
    is why it now arrives as ``grids.carrier_2w0``. The sign is the FFT convention's
    (``docs/guide/media/nonlinear.md``).

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, whose compiled Kerr term says whether the harmonic is wanted.

    Returns
    -------
    numpy.ndarray or None
        The carrier of shape ``(n_to,)``, or ``None`` when no harmonic term is present.

    Raises
    ------
    BackendError
        If a harmonic term is compiled and the contract carries no carrier for it.
        ``validate`` refuses that pairing, so only a hand-built problem reaches here.
    """
    if not problem.responses.harmonic_carrier:
        return None
    if problem.grids.carrier_2w0 is None:
        msg = (
            "a third-harmonic Kerr term is compiled and grids.carrier_2w0 is absent; "
            "the envelope's harmonic sits an offset 2 omega_0 from its carrier and the "
            "response pass cannot place it without the factor"
        )
        raise BackendError(msg)
    return problem.grids.carrier_2w0


def _embedded_projection(
    problem: KernelProblem,
) -> Float[np.ndarray, "n_modes n_columns"] | None:
    """Return the embedded coarse-rule projection, when the quadrature has one.

    ``transform.projection_coarse`` equals ``transform.projection`` when the rule has no
    embedded pair, in which case the difference of the two projections is identically
    zero and reading it as convergence would be wrong
    (:class:`solprop.modes.quadrature.ModeMatrices`). The alarm reports ``nan`` there
    instead, and the second projection is not computed at all.

    Parameters
    ----------
    problem : solprop.translate.problem.KernelProblem
        The problem, whose transform carries both projections.

    Returns
    -------
    numpy.ndarray or None
        The coarse projection, or ``None`` when it is the fine one.
    """
    transform = problem.transform
    if np.array_equal(transform.projection, transform.projection_coarse):
        return None
    return transform.projection_coarse
