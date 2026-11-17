# SPDX-FileCopyrightText: 2026 John C. Travers and contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""The statistics the reference forms itself, from a committed state (SEP 9, SEP 19).

:class:`ReferenceStatistics` answers :meth:`NumpyWorkspace.stats
<solprop.backends.numpy_ref.workspace.NumpyWorkspace.stats>`. The statistics of the
saved field are :func:`solprop.stats.collect`'s (or, on a spatial geometry,
:func:`solprop.backends.spatial.collect_spatial`'s), with the Poynting weight and the
basis metric at the plane; the ones a saved plane does not carry are formed here: the
electron density and peak ionisation rate at the densest node, the species and
mixture populations, the plasma integration diagnostics (SEP 18), the peak on-axis
intensity and the four on-axis plasma statistics, which scan the field synthesised on
the axis unless a quadrature node sits there. :meth:`ReferenceStatistics.node_fields`
is the one definition of the node fields every population scan reads, the fields the
stage's response pass evaluates.
"""

import dataclasses
from collections.abc import Iterator, Mapping, Sequence

import numpy as np
from jaxtyping import Complex, Float, Inexact, Num

from solprop.backends.numpy_ref.passes.plasma import (
    envelope_populations,
    ionisation_rate,
    plasma_channel_fractions,
    plasma_integration_diagnostics,
    plasma_integration_statistics,
    plasma_scan,
)
from solprop.backends.numpy_ref.passes.transforms import synthesise, to_time
from solprop.backends.populations import (
    axis_node,
    axis_refusal,
    axis_row_matrix,
    axis_statistics,
    population_statistics,
)
from solprop.backends.spatial import FarFieldTables, collect_spatial, slab_synthesis
from solprop.constants import EPS0, C
from solprop.exceptions import BackendError
from solprop.fourier import GridKind
from solprop.grid import TimeGrid
from solprop.modes.protocol import Mode
from solprop.modes.quadrature import mode_matrix, on_axis_rule
from solprop.output.capability import (
    declared_capability,
    refusal_for,
    withheld_accessors,
)
from solprop.output.records import SavedPlane
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
from solprop.translate.coupled import metric_at
from solprop.translate.linop import power_weight_at
from solprop.translate.problem import PropagationProblem
from solprop.translate.routes import flux_of
from solprop.translate.run_request import RunRequest

__all__ = ["ReferenceStatistics"]


class ReferenceStatistics:
    """The reference's statistics of a state, and the tables they read.

    Built once by the workspace from the validated problem and request. It holds no
    state of the run: :meth:`collect` is handed the committed state and the quadrature
    alarm, and keeps only what does not depend on them -- the far-field tables, the
    Poynting weight's buffer, the on-axis synthesis matrix and the axis node.

    Parameters
    ----------
    problem : solprop.translate.problem.PropagationProblem
        The validated problem.
    request : solprop.translate.run_request.RunRequest
        Its run's request, whose statistics the far-field tables are formed for.
    node_chunk : int
        The nodes a population scan holds at once.
    grid : solprop.grid.TimeGrid or None
        The grid the state lives on, which the statistics need.
    modes : collections.abc.Sequence or None
        The propagated modes, in state order, for the on-axis statistics.
    profiles : collections.abc.Mapping or None
        The z-profiles keyed by statistic name.
    name : str
        The backend's name, for messages.

    Attributes
    ----------
    axis_scans : int
        The on-axis plasma scans actually run (SEP 19), so that a test can see that a
        rule with a node on the axis supplies that row instead of a fresh scan.
    """

    def __init__(
        self,
        problem: PropagationProblem,
        request: RunRequest,
        *,
        node_chunk: int,
        grid: TimeGrid | None,
        modes: Sequence[Mode] | None,
        profiles: Mapping[str, ZProfile] | None,
        name: str,
    ) -> None:
        """Form the tables; refuse a geometry the far-field statistics refuse."""
        self._problem = problem
        self._kind: GridKind = problem.grid_kind
        self._node_chunk = node_chunk
        self._grid = grid
        self._modes = modes
        self._profiles = profiles
        self.name = name
        self._power_weight = np.empty(
            problem.shape if problem.transform.geometry is None else (0,)
        )
        # Formed once, so a geometry the far-field statistics refuse is refused here,
        # before any step (SEP 23 section 6).
        self._far_field = FarFieldTables(problem, request.observation.stats)
        self._on_axis: Float[np.ndarray, "n_rows n_modes"] | None = None
        self._axis_node: int | None = None
        self._axis_node_known = False
        self.axis_scans = 0

    @property
    def power_weight_bytes(self) -> int:  # numpydoc ignore=RT01
        """The Poynting weight buffer's size, which the memory ledger names."""
        return self._power_weight.nbytes

    def close(self) -> None:
        """Release the weight buffer. Idempotent."""
        self._power_weight = np.empty(0)

    def collect(
        self,
        spec: StatsSpec,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
        *,
        alarm: float,
    ) -> dict[str, np.ndarray]:
        """Collect the requested statistics of a state at a position.

        The statistics of the saved field go to :func:`solprop.stats.collect`; there is
        exactly one definition of each in the package and that is it. The statistics
        :class:`solprop.stats.StatDefinition` marks ``backend`` are of quantities a
        saved plane does not carry -- the electron density, the peak ionisation rate,
        the on-axis intensity, the quadrature error -- and are formed here from the
        state and the workspace's tables.

        Parameters
        ----------
        spec : solprop.stats.StatsSpec
            Which statistics and which windows.
        state : numpy.ndarray
            The committed state, shape ``(n_modes, n_pol, n_omega)``.
        z : float
            The state's position, m.
        alarm : float
            The quadrature alarm the last accepted step scored, which the
            ``quadrature_error`` statistic reports.

        Returns
        -------
        dict
            Small host arrays keyed by statistic name.
        """
        grid = self._require_grid()
        # Before the spatial branch: a spatial run asked for a
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
            spatial = collect_spatial(
                spatial_spec,
                grid,
                self._problem.transform,
                state,
                z=z,
                flux=flux_of(self._problem.route),
                profiles=self._profiles,
                tables=tables,
                operator=self._problem.linear.separable,
                node_chunk=self._node_chunk,
            )
            return {
                name: np.array(spatial[name]) for name in spatial_spec.names
            } | self._backend_statistics(
                tuple(name for name in spec.names if name in response_names),
                state,
                z,
                alarm,
            )
        # The weight is a Chebyshev evaluation per column, which is the whole of what
        # the Poynting weighting adds to a statistics call on a CPU (0.053 ms of
        # 0.056 ms on the reduced-M2 case), so a spec that asks for none of the
        # Poynting statistics -- a periodic plasma subset, say -- never
        # evaluates it.
        needed = bool(set(spec.names) & WEIGHT_DEPENDENT_STATISTICS)
        if needed:
            np.copyto(self._power_weight, power_weight_at(self._problem, z))
        weight_refusal = weight_refusal_for(spec.names, self._power_weight)
        if weight_refusal is not None:
            raise BackendError(weight_refusal)
        weighted = needed and not np.all(self._power_weight == 1.0)
        power_names = tuple(
            name for name in spec.names if weighted and name in _POYNTING_STATISTICS
        )
        axis = None
        if "peak_intensity" in power_names:
            if self._modes is None:
                msg = "the peak on-axis intensity needs the mode set"
                raise BackendError(msg)
            axis = self._on_axis_synthesis(self._modes, z)
        power_values = (
            _poynting_statistics(spec, grid, state, self._power_weight, axis)
            if weighted
            else {}
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
        # The basis metric at this plane: the
        # statistics of `METRIC_WEIGHTED_STATISTICS` are functionals of the physical
        # same-frequency flux `a.H W a`, and on a declared change of basis W is not
        # the identity. `metric_at` returns None for an identity metric, which is
        # the statistics' own bitwise-unchanged path, so an ordinary run pays
        # nothing.
        metric = (
            metric_at(self._problem.linear.modal_metric, z)
            if set(delegated.names) & METRIC_WEIGHTED_STATISTICS
            else None
        )
        collected = collect(
            delegated,
            grid,
            SavedPlane(z=z, field=state),
            modes=self._modes,
            profiles=self._profiles,
            metric=metric,
        )
        values = {name: np.array(collected[name]) for name in delegated.names}
        values.update({name: np.array(power_values[name]) for name in power_names})
        return values | self._backend_statistics(owned, state, z, alarm)

    def _backend_statistics(
        self,
        names: tuple[str, ...],
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
        alarm: float,
    ) -> dict[str, np.ndarray]:
        """Share each fresh envelope population scan among the requested statistics.

        - ``state``: accepted state at this plane.
        - Returns: requested values, with no retained field cache.
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
        # Node scans of this observation only: a rule with a node on the axis hands
        # its row to the on-axis statistics, and nothing outlives the call, so no
        # stale stage or rejected step can be reused.
        scans: list[tuple[np.ndarray, np.ndarray]] = []
        result = {
            name: self._backend_stat(name, state, z, alarm)
            for name in names
            if name not in population_names
            and name not in integration_names
            and name not in axis_names
        }
        if integration_names:
            diagnostics = self._integration_diagnostics(state, z)
            result.update(
                {name: np.array(diagnostics[name]) for name in integration_names}
            )
        if population_names:
            species = self._species_statistics(state, z, scans)
            result.update({name: np.array(species[name]) for name in population_names})
        if axis_names:
            axis = self._axis_statistics(state, scans[0] if scans else None, z)
            result.update({name: np.array(axis[name]) for name in axis_names})
        return result

    def _backend_stat(
        self,
        name: str,
        state: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
        alarm: float,
    ) -> np.ndarray:
        """Compute one backend-owned statistic."""
        if name == "quadrature_error":
            return np.full((), alarm)
        if name == "peak_intensity":
            return np.array(self._peak_intensity(state, z))
        if (
            name.startswith("species_")
            or self._problem.flags.n_channels > 1
            or self._problem.responses.envelope_plasma is not None
        ):
            return np.array(self._species_statistics(state, z)[name])
        return np.array(
            self._ionisation(state, z)[0 if name == "electron_density" else 1]
        )

    def _species_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
        scans: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate independent populations on the actual nonlinear nodes.

        Every channel's final fraction and peak rate at every node of
        :meth:`node_fields`: cycle-averaged on an envelope grid, the field grid's
        exposure scan otherwise.

        - ``scans``: where to leave this observation's channel fractions and peak rates
          at every node, for the on-axis statistics to reuse.
        - Returns: the five species statistics and the two total plasma statistics.
        """
        responses = self._problem.responses
        channels = responses.population_channels
        if not channels:
            empty = np.zeros((0, self._problem.transform.n_nodes))
            return population_statistics(self._problem, z, empty, empty)
        initials = tuple(float(channel.preionisation(z)) for channel in channels)
        dt = self._problem.grids.dt_fine
        finals, peaks = [], []
        for nodes in self.node_fields(field, z):
            if responses.envelope_plasma is not None:
                final, peak = envelope_populations(
                    nodes, responses.envelope_plasma, initials, dt
                )
            else:
                rates, population = plasma_channel_fractions(
                    nodes,
                    responses.plasma_channels,
                    initials,
                    dt,
                    responses.plasma_integral,
                )
                final, peak = population[:, :, -1], rates.max(axis=-1)
            finals.append(final)
            peaks.append(peak)
        fractions, largest = (
            np.concatenate(finals, axis=1),
            np.concatenate(peaks, axis=1),
        )
        if scans is not None:
            scans.append((fractions, largest))
        return population_statistics(self._problem, z, fractions, largest)

    def _integration_diagnostics(
        self, field: Complex[np.ndarray, "n_modes n_pol n_omega"], z: float
    ) -> dict[str, np.ndarray]:
        """Return the plasma column's integration diagnostics at this plane (SEP 18).

        The undershoot of the exposure increments, the relative work defect of the
        loss current and the boundary level of the free-current integrand, over every
        node and field plasma channel
        (:data:`solprop.stats.PLASMA_INTEGRATION_STATISTICS`).

        - Returns: the three statistics, each zero-dimensional and dimensionless.
        """
        responses = self._problem.responses
        channels = responses.plasma_channels
        if not channels:
            msg = (
                "the plasma integration diagnostics read the field-grid plasma "
                "column, and this problem carries none"
            )
            raise BackendError(msg)
        initial = tuple(float(channel.preionisation(z)) for channel in channels)
        parts = np.zeros(6)
        for nodes in self.node_fields(field, z):
            parts = np.maximum(
                parts,
                plasma_integration_diagnostics(
                    nodes,
                    channels,
                    initial,
                    self._problem.grids.dt_fine,
                    responses.plasma_integral,
                ),
            )
        undershoot, defect, boundary = plasma_integration_statistics(parts)
        return {
            "plasma_undershoot": np.asarray(undershoot),
            "plasma_work_defect": np.asarray(defect),
            "plasma_boundary_level": np.asarray(boundary),
        }

    def node_fields(
        self, state: Complex[np.ndarray, "n_modes n_pol n_omega"], z: float
    ) -> Iterator[Num[np.ndarray, "n_nodes_chunk n_components n_to"]]:
        """Yield the physical field at the nodes the responses see, chunk by chunk.

        The fields the stage's response pass evaluates, on the fine time grid: the
        modal quadrature's nodes at once (``_nodes_at``), or a separable
        geometry's physical nodes by the node chunk -- the slab synthesis in x, which
        carries the pad scale, or the inverse transverse DFT followed by the pad scale.

        Parameters
        ----------
        state : numpy.ndarray
            A state, shape ``(n_modes, n_pol, n_omega)``.
        z : float
            Its position, m.

        Yields
        ------
        numpy.ndarray
            The field at a chunk of nodes, V/m, shape
            ``(n_nodes_chunk, n_components, n_to)``: real on a field grid, complex on
            an envelope grid.
        """
        problem = self._problem
        if problem.linear.separable is None:
            yield self._nodes_at(state, z)
            return
        grids, transform = problem.grids, problem.transform
        geometry = transform.geometry
        assert geometry is not None
        slab = geometry.kind == "slab"
        if slab:
            physical = slab_synthesis(transform, state)
        else:
            nx, ny = geometry.shape
            physical = np.fft.ifft2(
                state.reshape(ny, nx, *state.shape[1:]), axes=(0, 1)
            ).reshape(state.shape)
        for first in range(0, len(physical), self._node_chunk):
            rows = physical[first : first + self._node_chunk]
            yield to_time(
                rows if slab else rows * grids.pad_scale,
                grids.crop_index,
                grids.n_to,
                kind=self._kind,
            )

    def _ionisation(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the electron density and the peak ionisation rate at this plane.

        Both are read at the **node of largest ionised fraction**, which is where the
        plasma the run carries is densest and the only node either number can honestly
        describe; for a mode-averaged run that is the one node. The axis is not a
        quadrature node at all and is not where the polarisation was evaluated, so the
        field is not re-synthesised there.

        The electron density is ``density(z)`` times the fraction at the **end** of the
        window: what the pulse leaves behind, not what it reaches in passing.

        - Returns: the electron density in m^-3 and the peak rate in s^-1, both
          zero-dimensional.
        """
        responses = self._problem.responses
        plasma = responses.plasma_coefficients
        if plasma is None:
            msg = (
                "the electron density and the peak ionisation rate need a compiled "
                "plasma response, and this problem declares none"
            )
            raise BackendError(msg)
        largest: float | None = None
        peak = 0.0
        for nodes in self.node_fields(field, z):
            rate = ionisation_rate(nodes, plasma.rate)
            fraction, _ = plasma_scan(
                nodes,
                rate,
                plasma,
                preionisation=float(plasma.preionisation(z)),
                dt=self._problem.grids.dt_fine,
                integral=responses.plasma_integral,
            )
            densest = int(np.argmax(fraction[:, -1]))
            if largest is None or fraction[densest, -1] > largest:
                largest = float(fraction[densest, -1])
                peak = float(np.max(rate[densest]))
        assert largest is not None
        return np.asarray(float(plasma.density(z)) * largest), np.asarray(peak)

    def _peak_intensity(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
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
        (``_on_axis_synthesis``), which is what makes this statistic cheap enough
        for the between-planes channel.
        """
        if self._modes is None:
            msg = (
                "the peak on-axis intensity needs the mode set, which the kernel "
                "contract does not carry; construct the backend with "
                "NumpyRefBackend(problem, request, modes=modes)"
            )
            raise BackendError(msg)
        analytic = analytic_field(self._require_grid(), field)
        columns = np.tensordot(
            self._on_axis_synthesis(self._modes, z), analytic, axes=([1], [0])
        )
        power = np.sum(np.abs(columns) ** 2, axis=tuple(range(columns.ndim - 1)))
        return np.asarray(0.5 * EPS0 * C * float(np.max(power)))

    def _nodes_at(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        z: float,
    ) -> Num[np.ndarray, "n_nodes n_components n_to"]:
        """Return the physical field at every quadrature node on the fine time grid.

        - Returns: the field in V/m, shape ``(n_nodes, n_components, n_to)``. Here
          ``n_components = transform.n_pol * n_pol`` counts physical components,
          independently of the state's polarisation axis.
        """
        grids, transform = self._problem.grids, self._problem.transform
        taper = (
            1.0 if transform.taper_scale is None else float(transform.taper_scale(z))
        )
        modal_t = to_time(field, grids.crop_index, grids.n_to, kind=self._kind)
        return synthesise(modal_t, transform.synthesis, taper, transform.n_pol)

    def _axis_statistics(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        scan: tuple[np.ndarray, np.ndarray] | None,
        z: float,
    ) -> dict[str, np.ndarray]:
        """Return the four on-axis plasma statistics of the state (SEP 19).

        A rule with a node on the axis hands over that node's row from this
        observation's node scan; otherwise the axis row is scanned once
        (``_axis_populations``).

        - ``field``: the state at this plane, shape (n_modes, n_pol, n_omega).
        - ``scan``: this observation's channel fractions and peak rates at every node,
          when the node statistics already scanned them.
        - Returns: the four ``on_axis_*`` statistics.
        """
        if self._modes is None:
            msg = (
                "the on-axis plasma statistics need the mode set, which the kernel "
                "contract does not carry; construct the backend with "
                f"select({self.name!r}, problem, request, modes=modes)"
            )
            raise BackendError(msg)
        if not self._axis_node_known:
            self._axis_node_known = True
            self._axis_node = axis_node(
                self._problem,
                mode_matrix(
                    self._modes,
                    on_axis_rule(self._modes[0].dimlimits(0.0)[0]),
                    0.0,
                    components=("x", "y"),
                ).synthesis,
            )
        if scan is not None and self._axis_node is not None:
            fractions, peaks = scan
            node = self._axis_node
            return axis_statistics(self._problem, z, fractions[:, node], peaks[:, node])
        self.axis_scans += 1
        fractions, peaks = self._axis_populations(
            field, self._on_axis_synthesis(self._modes, z), z
        )
        return axis_statistics(self._problem, z, fractions, peaks)

    def _axis_populations(
        self,
        field: Complex[np.ndarray, "n_modes n_pol n_omega"],
        axis: Float[np.ndarray, "2 n_modes"],
        z: float,
    ) -> tuple[Float[np.ndarray, " n_channels"], Float[np.ndarray, " n_channels"]]:
        """Scan the field synthesised on the axis: one row, every channel (SEP 19).

        The axis is one row per saved plane, against every node of every stage. The
        field is the same as the node fields' -- the same pad scale, taper and fine
        grid -- with both Cartesian components of the mode profiles, as for
        ``peak_intensity``; when the state carries the components itself, the
        profile's magnitude on the axis is the scalar synthesis. The tables,
        preionisation and running-integral rule are the node scan's.

        - ``axis``: the mode set's synthesis on the axis at `z`, taper included, rows x
          and y (``_on_axis_synthesis``).
        - Returns: the final ionised fraction and the largest rate (cycle-averaged on
          an envelope grid) on the axis, one per channel.
        """
        problem = self._problem
        responses, grids = problem.responses, problem.grids
        modal_t = to_time(field, grids.crop_index, grids.n_to, kind=self._kind)
        row: Inexact[np.ndarray, "1 n_components n_to"] = synthesise(
            modal_t,
            axis_row_matrix(problem, axis),
            1.0,
            2 if field.shape[1] == 1 else 1,
        )
        preionisation = tuple(
            float(channel.preionisation(z)) for channel in responses.population_channels
        )
        if responses.envelope_plasma is not None:
            fractions, peaks = envelope_populations(
                row, responses.envelope_plasma, preionisation, grids.dt_fine
            )
            return fractions[:, 0], peaks[:, 0]
        rates, population = plasma_channel_fractions(
            row,
            responses.plasma_channels,
            preionisation,
            grids.dt_fine,
            responses.plasma_integral,
        )
        return population[:, 0, -1], rates[:, 0].max(axis=-1)

    def _on_axis_synthesis(
        self, modes: Sequence[Mode], z: float
    ) -> Float[np.ndarray, "n_rows n_modes"]:
        """Return the mode set's synthesis matrix on the axis, at the current position.

        Built once at ``z = 0`` and rescaled by the transform's own taper factor: the
        axis is the single node ``r = 0`` of
        :func:`solprop.modes.quadrature.on_axis_rule`, so the law that rescales the
        transform's matrices -- a scale-invariant profile goes as ``1/a`` and its
        normalisation as ``a^2`` -- applies to it exactly, and a set that is neither
        constant nor scale invariant never reaches a backend (``translate.modal``).
        Rebuilding the matrix at every call was what kept ``peak_intensity`` out of the
        between-planes channel.

        - Returns: shape ``(n_rows, n_modes)``, one row per Cartesian component on the
          axis.
        """
        if self._on_axis is None:
            rule = on_axis_rule(modes[0].dimlimits(0.0)[0])
            matrices = mode_matrix(modes, rule, 0.0, components=("x", "y"))
            self._on_axis = np.ascontiguousarray(matrices.synthesis)
        taper = self._problem.transform.taper_scale
        if taper is None:
            return self._on_axis
        return self._on_axis * float(taper(z))

    def _require_grid(self) -> TimeGrid:
        """Return the grid the state lives on, or say why statistics cannot be taken."""
        if self._grid is None:
            msg = (
                "statistics need the grid the state lives on, which the kernel "
                f"contract does not carry; construct the backend with select("
                f"{self.name!r}, problem, request, grid=grid)"
            )
            raise BackendError(msg)
        return self._grid


def _refuse_inadmissible(problem: PropagationProblem, spec: StatsSpec) -> None:
    """Refuse a statistic this problem's basis does not define, by name.

    The second door of the accessor whitelist (:mod:`solprop.output.capability`),
    after the run loop's own admission: a caller may build a backend and ask it for
    statistics directly, and a general guide's static synthesis matrix is a checked
    **zero** placeholder, so an on-axis intensity taken from one would be zero
    rather than obviously wrong.
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
